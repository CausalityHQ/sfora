#!/usr/bin/env python3
"""Train-only In-Shop query/gallery check of three pretrained L/14 head executions."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from evaluate_sop_cub_transfer import gpu_compute_pids
from PIL import Image
from probe_l14_head_lowrank import factorize_linear
from probe_l14_parallel_fold import assert_no_foreign_gpu_processes, load_authenticated_l14
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from train_sop_compact_backbone import publish_file_noreplace, sha256

from sfora.inference_head import fold_eval_affine_head
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca
from sfora.unicom_audit_io import load_embedding_bundle
from sfora.unicom_inshop import parse_inshop_partition

PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
SOURCE_ARCHIVE_SHA256 = "6eae13715e18d7eb99450bade5056538f8f08f1e9b550d0f24ee09e52bb25d0e"
FIT_FRACTION = 0.9
SPLIT_SEED = 179019


def split_query_gallery_rows(labels: tuple[int, ...]) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Put alternating images of each identity in separate gallery/query roles."""

    if not labels or min(Counter(labels).values()) < 2:
        raise ValueError("In-Shop constructed validation needs two images per identity")
    positions: dict[int, int] = {}
    query, gallery = [], []
    for index, label in enumerate(labels):
        offset = positions.get(label, 0)
        positions[label] = offset + 1
        (gallery if offset % 2 == 0 else query).append(index)
    return tuple(query), tuple(gallery)


def score_query_gallery(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    *,
    query_inverse_norms: torch.Tensor | None = None,
    gallery_inverse_norms: torch.Tensor | None = None,
    prefix_euclidean_dimensions: int | None = None,
) -> dict[str, object]:
    """Exact stable ordinal ranking, Recall@1 and mAP@R for disjoint roles."""

    if (
        query.dtype != torch.float32
        or gallery.dtype != torch.float32
        or query.ndim != 2
        or gallery.ndim != 2
        or query.shape[1] != gallery.shape[1]
        or query_labels.dtype != torch.int64
        or gallery_labels.dtype != torch.int64
        or query_labels.shape != (len(query),)
        or gallery_labels.shape != (len(gallery),)
        or len({query.device, gallery.device, query_labels.device, gallery_labels.device}) != 1
        or not bool(torch.isfinite(query).all())
        or not bool(torch.isfinite(gallery).all())
    ):
        raise ValueError("In-Shop query/gallery scoring geometry differs")
    packed = query_inverse_norms is not None or gallery_inverse_norms is not None
    if packed and (
        query_inverse_norms is None
        or gallery_inverse_norms is None
        or query_inverse_norms.shape != (len(query),)
        or gallery_inverse_norms.shape != (len(gallery),)
        or query_inverse_norms.device != query.device
        or gallery_inverse_norms.device != gallery.device
        or prefix_euclidean_dimensions is not None
    ):
        raise ValueError("In-Shop packed scorer geometry differs")
    if prefix_euclidean_dimensions is not None and not (
        2 <= prefix_euclidean_dimensions < query.shape[1]
    ):
        raise ValueError("In-Shop prefix scorer geometry differs")
    if not packed:
        query = F.normalize(query, dim=1)
        gallery = F.normalize(gallery, dim=1)
    if prefix_euclidean_dimensions is not None:
        query = query[:, :prefix_euclidean_dimensions].contiguous()
        gallery = gallery[:, :prefix_euclidean_dimensions].contiguous()
        gallery_norms = (gallery * gallery).sum(dim=1)
    else:
        gallery_norms = None
    counts = Counter(gallery_labels.cpu().tolist())
    relevant = torch.tensor(
        [counts[int(label)] for label in query_labels.cpu().tolist()],
        dtype=torch.int64,
        device=query.device,
    )
    if len(query) < 1 or len(gallery) < 1 or int(relevant.min()) < 1:
        raise ValueError("In-Shop query lacks a gallery positive")
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1, device=query.device)
    per_r1: list[float] = []
    per_ap: list[float] = []
    top1: list[int] = []
    for start in range(0, len(query), 64):
        stop = min(start + 64, len(query))
        scores = query[start:stop] @ gallery.T
        if packed:
            scores = (
                scores
                * query_inverse_norms[start:stop, None].float()
                * gallery_inverse_norms[None, :].float()
            )
        if gallery_norms is not None:
            scores = 2 * scores - gallery_norms[None, :]
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width]
        matches = gallery_labels[ranked] == query_labels[start:stop, None]
        precision = torch.cumsum(matches, dim=1) / ranks[None, :]
        valid = ranks[None, :] <= relevant[start:stop, None]
        ap = (precision * matches * valid).sum(dim=1) / relevant[start:stop]
        per_r1.extend(float(value) for value in matches[:, 0].cpu().tolist())
        per_ap.extend(float(value) for value in ap.cpu().tolist())
        top1.extend(int(value) for value in ranked[:, 0].cpu().tolist())
    return {
        "queries": len(query),
        "gallery": len(gallery),
        "recall_at_1": sum(per_r1) / len(per_r1),
        "map_at_r": sum(per_ap) / len(per_ap),
        "per_query_r1": per_r1,
        "per_query_ap": per_ap,
        "top1_gallery_ordinals": top1,
    }


class VerifiedTrainImages(Dataset):
    def __init__(self, records: tuple, transform: object, manifest: bytes) -> None:
        if len(manifest) != 32 * len(records):
            raise ValueError("In-Shop train image manifest differs")
        self.records = records
        self.transform = transform
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        row = self.records[index]
        data = row.image_path.read_bytes()
        if hashlib.sha256(data).digest() != self.manifest[index * 32 : (index + 1) * 32]:
            raise ValueError("In-Shop train image content changed")
        with Image.open(BytesIO(data)) as image:
            return self.transform(image.convert("RGB")), row.label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("unicom-checkout", "checkpoint", "inshop-root", "source-archive", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--execute-l14-inshop-train-head-screen", action="store_true", required=True
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.workers < 0
        or not torch.cuda.is_available()
        or gpu_compute_pids()
        or sha256(args.inshop_root / "Eval" / "list_eval_partition.txt") != PARTITION_SHA256
        or sha256(args.source_archive) != SOURCE_ARCHIVE_SHA256
    ):
        raise ValueError("In-Shop L/14 head screen invocation differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    all_records = parse_inshop_partition(args.inshop_root)
    train = tuple(row for row in all_records if row.split == "train")
    source_archive = load_embedding_bundle(args.source_archive)
    if (
        source_archive.metadata["model_identifier"] != "UNICOM-ViT-L/14@336px"
        or source_archive.metadata["checkpoint_sha256"]
        != "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
        or not np.array_equal(
            source_archive.train_labels, np.asarray([row.label for row in train])
        )
    ):
        raise ValueError("In-Shop L/14 source archive differs")
    counts = Counter(row.label for row in train)
    eligible_indexes = tuple(index for index, row in enumerate(train) if counts[row.label] >= 2)
    identities = {label: index for index, label in enumerate(sorted(counts))}
    eligible_labels = tuple(identities[train[index].label] for index in eligible_indexes)
    partition = deterministic_class_partition(
        eligible_labels, fit_fraction=FIT_FRACTION, seed=SPLIT_SEED
    )
    fit_indexes = tuple(eligible_indexes[index] for index in partition.fit_row_indexes)
    validation_indexes = tuple(
        eligible_indexes[index] for index in partition.validation_row_indexes
    )
    selected = tuple(train[index] for index in validation_indexes)
    selected_labels = tuple(identities[row.label] for row in selected)
    query_indexes, gallery_indexes = split_query_gallery_rows(selected_labels)
    manifest = b"".join(hashlib.sha256(row.image_path.read_bytes()).digest() for row in selected)
    fit_features = torch.from_numpy(
        np.ascontiguousarray(source_archive.train_embeddings[list(fit_indexes)]).copy()
    )
    pca_started = time.perf_counter()
    pca = fit_centered_pca(F.normalize(fit_features, dim=1), dimensions=128)
    pca_fit_seconds = time.perf_counter() - pca_started
    del fit_features
    model, transform = load_authenticated_l14(args.unicom_checkout, args.checkpoint)
    rank_first, spectral_energy = factorize_linear(model.feature[0], 512)
    model = model.cuda().eval()
    source_head = model.feature
    rank_head = nn.Sequential(
        rank_first.cuda(), source_head[1], source_head[2], source_head[3]
    ).eval()
    fused_head = fold_eval_affine_head(source_head)
    arms = {"original": source_head, "rank512": rank_head, "exact_fused": fused_head}
    assert_no_foreign_gpu_processes()
    loader = DataLoader(
        VerifiedTrainImages(selected, transform, manifest),
        batch_size=32,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    encoded = {name: [] for name in arms}
    encode_started = time.perf_counter()
    with torch.inference_mode():
        for images, _ in loader:
            flattened = model.forward_features(images.cuda(non_blocking=True))
            for name, head in arms.items():
                encoded[name].append(head(flattened).float().cpu())
    features = {name: torch.cat(parts).contiguous() for name, parts in encoded.items()}
    if any(value.shape != (len(selected), 768) for value in features.values()):
        raise ValueError("In-Shop L/14 encoded validation geometry differs")
    archive_validation = torch.from_numpy(
        np.ascontiguousarray(source_archive.train_embeddings[list(validation_indexes)]).copy()
    )
    archive_max_abs = float((features["original"] - archive_validation).abs().max())
    if archive_max_abs > 1e-4:
        raise ValueError(f"In-Shop source archive feature mismatch: {archive_max_abs}")
    exact_max_abs = float((features["original"] - features["exact_fused"]).abs().max())
    labels = torch.tensor(selected_labels, dtype=torch.int64, device="cuda")
    query_labels = labels[list(query_indexes)]
    gallery_labels = labels[list(gallery_indexes)]
    results: dict[str, object] = {}
    for name, values in features.items():
        full = values.cuda()
        packed_full = pack_int8_unit_embeddings(F.normalize(values, dim=1))
        reduced = pca.apply(F.normalize(values, dim=1))
        packed_128 = pack_int8_unit_embeddings(reduced)
        arms_to_score = {
            "full_float_cosine": (full, None, None, None),
            "upstream_prefix512_euclidean": (full, None, None, 512),
            "full_packed_cosine": (
                packed_full.codes.float().cuda(),
                packed_full.inverse_norms.cuda(),
                packed_full.inverse_norms.cuda(),
                None,
            ),
            "pca128_packed_cosine": (
                packed_128.codes.float().cuda(),
                packed_128.inverse_norms.cuda(),
                packed_128.inverse_norms.cuda(),
                None,
            ),
        }
        results[name] = {}
        for scorer, (matrix, query_inverse, gallery_inverse, prefix) in arms_to_score.items():
            results[name][scorer] = score_query_gallery(
                matrix[list(query_indexes)],
                matrix[list(gallery_indexes)],
                query_labels,
                gallery_labels,
                query_inverse_norms=(
                    None if query_inverse is None else query_inverse[list(query_indexes)]
                ),
                gallery_inverse_norms=(
                    None if gallery_inverse is None else gallery_inverse[list(gallery_indexes)]
                ),
                prefix_euclidean_dimensions=prefix,
            )
        print(
            json.dumps(
                {
                    "arm": name,
                    "quality": {
                        scorer: {metric: result[metric] for metric in ("recall_at_1", "map_at_r")}
                        for scorer, result in results[name].items()
                    },
                }
            ),
            flush=True,
        )
    assert_no_foreign_gpu_processes()
    result = {
        "schema": "sfora-l14-heads-inshop-train-query-gallery-v1",
        "claim_eligible": False,
        "split": "In-Shop official train identities only; 90/10 class-disjoint fit/validation",
        "split_seed": SPLIT_SEED,
        "singleton_train_identities_excluded": sum(count == 1 for count in counts.values()),
        "fit_images": len(fit_indexes),
        "validation_images": len(selected),
        "query_images": len(query_indexes),
        "gallery_images": len(gallery_indexes),
        "validation_image_paths": [
            str(row.image_path.relative_to(args.inshop_root)) for row in selected
        ],
        "validation_labels": list(selected_labels),
        "query_indexes": query_indexes,
        "gallery_indexes": gallery_indexes,
        "validation_image_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "spectral_energy_fraction_rank512": spectral_energy,
        "pca128_fit_seconds": pca_fit_seconds,
        "pca128_mean_sha256": hashlib.sha256(pca.mean.numpy().tobytes()).hexdigest(),
        "pca128_components_sha256": hashlib.sha256(pca.components.numpy().tobytes()).hexdigest(),
        "archive_validation_max_abs_difference": archive_max_abs,
        "exact_fused_max_abs_difference": exact_max_abs,
        "encoding_seconds_all_arms_shared_trunk_including_decode": time.perf_counter()
        - encode_started,
        "quality": results,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "total_seconds": time.perf_counter() - started,
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "inputs": {
            "checkpoint_sha256": sha256(args.checkpoint),
            "partition_sha256": PARTITION_SHA256,
            "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
            "source_sha256": sha256(Path(__file__)),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish_file_noreplace(
        args.output,
        lambda stream: stream.write(
            (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )
    print(json.dumps({"output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
