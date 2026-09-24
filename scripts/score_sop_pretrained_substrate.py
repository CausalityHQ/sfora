#!/usr/bin/env python3
"""Fit-only PCA and exact packed SOP TRAIN score for pretrained substrates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from score_sop_b16_resolution_train import compare
from screen_sop_finite_gallery_head import score_heldout_against_all
from torch.nn import functional as F

from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.representation_ceiling import (
    CenteredPcaTransform,
    deterministic_class_partition,
    fit_centered_pca,
)
from sfora.sop_evaluation import score_gallery_r1, score_symmetric

UNICOM_L14_ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
SEED = 179019
DEPENDENCIES = (
    "scripts/score_sop_pretrained_substrate.py",
    "scripts/score_sop_b16_resolution_train.py",
    "scripts/screen_sop_finite_gallery_head.py",
    "scripts/diagnose_sop_seen_gallery_effect.py",
    "src/sfora/joint_relational_compaction.py",
    "src/sfora/representation_ceiling.py",
    "src/sfora/sop_evaluation.py",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def publish_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


@dataclass(frozen=True)
class ProjectedCorpus:
    pca: CenteredPcaTransform
    normalized_source: torch.Tensor
    projected: torch.Tensor
    packed: PackedInt8Embeddings


def require_aligned_rows(
    source_ids: np.ndarray,
    source_labels: np.ndarray,
    source_paths: np.ndarray,
    candidate_ids: np.ndarray,
    candidate_labels: np.ndarray,
    candidate_paths: np.ndarray,
) -> None:
    """Reject a descriptor matrix attached to different ordered TRAIN rows."""

    if (
        source_ids.ndim != 1
        or source_labels.shape != source_ids.shape
        or source_paths.shape != source_ids.shape
        or candidate_ids.shape != source_ids.shape
        or candidate_labels.shape != source_ids.shape
        or candidate_paths.shape != source_ids.shape
        or not np.array_equal(source_ids, candidate_ids)
        or not np.array_equal(source_labels, candidate_labels)
        or not np.array_equal(source_paths.astype(str), candidate_paths.astype(str))
    ):
        raise ValueError("SOP substrate row alignment differs")


def fit_project_pack(
    features: np.ndarray, fit_rows: np.ndarray, *, dimensions: int = 128
) -> ProjectedCorpus:
    """Fit PCA solely on fit products and pack all TRAIN descriptors."""

    if (
        not isinstance(features, np.ndarray)
        or features.dtype != np.float32
        or features.ndim != 2
        or features.shape[1] < dimensions
        or not np.isfinite(features).all()
        or type(fit_rows) is not np.ndarray
        or fit_rows.dtype != np.int64
        or fit_rows.ndim != 1
        or len(fit_rows) <= dimensions
        or int(fit_rows.min()) < 0
        or int(fit_rows.max()) >= len(features)
    ):
        raise ValueError("SOP substrate projection inventory differs")
    source = F.normalize(torch.from_numpy(np.array(features, copy=True, order="C")), dim=1)
    if not bool(torch.isfinite(source).all()):
        raise ValueError("SOP substrate source features nonfinite")
    pca = fit_centered_pca(source[fit_rows].contiguous(), dimensions=dimensions)
    projected = pca.apply(source.contiguous())
    return ProjectedCorpus(
        pca=pca,
        normalized_source=source,
        projected=projected,
        packed=pack_int8_unit_embeddings(projected),
    )


def score_corpus(
    name: str,
    features: np.ndarray,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
) -> dict[str, Any]:
    begin = time.perf_counter()
    corpus = fit_project_pack(features, fit_rows)
    pca_seconds = time.perf_counter() - begin
    device = torch.device("cuda")
    all_labels = torch.from_numpy(labels.copy())
    held = torch.from_numpy(held_rows.copy())
    fit = torch.from_numpy(fit_rows.copy())
    held_labels = all_labels[held]
    begin = time.perf_counter()
    normalized = corpus.normalized_source.cuda()
    projected = corpus.projected.cuda()
    code = corpus.packed.codes.float()
    inverse = corpus.packed.inverse_norms
    packed_full = score_heldout_against_all(code, inverse, all_labels, held, fit, device=device)
    score = {
        "float_full_width": {
            "holdout_only": score_symmetric(normalized[held.cuda()], held_labels.cuda()),
        },
        "float_pca128": {
            "holdout_only": score_symmetric(projected[held.cuda()], held_labels.cuda()),
            "full_train_gallery": score_gallery_r1(projected, all_labels.cuda(), held.cuda()),
        },
        "packed_pca128": {
            "holdout_only": score_symmetric(
                code[held].cuda(), held_labels.cuda(), inverse_norms=inverse[held].cuda()
            ),
            "full_train_gallery": packed_full,
        },
    }
    score_seconds = time.perf_counter() - begin
    return {
        "schema": "sfora-sop-pretrained-substrate-arm-v1",
        "name": name,
        "split": "SOP official TRAIN product-disjoint fit/holdout only",
        "seed": SEED,
        "fit_images": len(fit_rows),
        "holdout_queries": len(held_rows),
        "full_gallery_images": len(labels),
        "source_dimensions": features.shape[1],
        "pca_fit_seconds": pca_seconds,
        "score_seconds": score_seconds,
        "pca_sha256": hashlib.sha256(
            corpus.pca.mean.numpy().tobytes() + corpus.pca.components.numpy().tobytes()
        ).hexdigest(),
        "gallery_bytes_per_item": {
            "float_full_width": features.shape[1] * 4,
            "float_pca128": 512,
            "packed_pca128": 130,
        },
        "score": score,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-l14-archive", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or args.output_dir.is_symlink()
        or not torch.cuda.is_available()
        or sha256(args.unicom_l14_archive) != UNICOM_L14_ARCHIVE_SHA256
    ):
        raise ValueError("SOP pretrained substrate source authority differs")
    candidate_receipt_path = args.candidate_dir / "receipt.json"
    candidate_receipt = json.loads(candidate_receipt_path.read_text())
    candidate_path = args.candidate_dir / "train_features.npy"
    if (
        candidate_receipt.get("schema") != "sfora-sop-siglip2-train-feature-export-v1"
        or candidate_receipt.get("full_train") is not True
        or candidate_receipt.get("rows") != 59_551
        or candidate_receipt.get("unicom_train_archive_sha256") != UNICOM_L14_ARCHIVE_SHA256
        or sha256(candidate_path) != candidate_receipt.get("features_sha256")
    ):
        raise ValueError("SOP candidate source receipt differs")
    with np.load(args.unicom_l14_archive, allow_pickle=False) as archive:
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        paths = np.asarray(archive["train_relative_paths"]).astype(str)
        baseline = np.asarray(archive["train_embeddings"], dtype=np.float32)
    ordered_hash = hashlib.sha256(
        "\n".join(
            f"{int(i)}\0{int(y)}\0{p}" for i, y, p in zip(ids, labels, paths, strict=True)
        ).encode()
    ).hexdigest()
    candidate = np.load(candidate_path, mmap_mode="r", allow_pickle=False)
    if (
        ids.shape != (59_551,)
        or labels.shape != ids.shape
        or paths.shape != ids.shape
        or baseline.shape != (59_551, 768)
        or candidate.shape != (59_551, 1024)
        or candidate.dtype != np.float32
        or ordered_hash != candidate_receipt.get("ordered_rows_sha256")
    ):
        raise ValueError("SOP substrate TRAIN row alignment differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    fit_rows = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_rows = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(fit_rows) != 53_700 or len(held_rows) != 5_851:
        raise ValueError("SOP substrate partition differs")
    source_root = Path(__file__).resolve().parents[1]
    source_hashes = {relative: sha256(source_root / relative) for relative in DEPENDENCIES}
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    baseline_result = score_corpus("unicom_l14_336", baseline, labels, fit_rows, held_rows)
    baseline_packed_holdout = baseline_result["score"]["packed_pca128"]["holdout_only"][
        "recall_at_1"
    ]
    if abs(baseline_packed_holdout - 0.861220) > 0.002:
        publish_json(args.output_dir / "unicom_l14_336.json", baseline_result)
        raise ValueError("SOP L/14 architecture holdout baseline parity differs")
    publish_json(args.output_dir / "unicom_l14_336.json", baseline_result)
    print(
        json.dumps(
            {
                "arm": "unicom_l14_336",
                "full_packed_r1": baseline_result["score"]["packed_pca128"]["full_train_gallery"][
                    "recall_at_1"
                ],
            }
        ),
        flush=True,
    )
    candidate_result = score_corpus("siglip2_l16_256", candidate, labels, fit_rows, held_rows)
    publish_json(args.output_dir / "siglip2_l16_256.json", candidate_result)
    print(
        json.dumps(
            {
                "arm": "siglip2_l16_256",
                "full_packed_r1": candidate_result["score"]["packed_pca128"]["full_train_gallery"][
                    "recall_at_1"
                ],
            }
        ),
        flush=True,
    )
    paired = {
        representation: {
            gallery: compare(
                candidate_result["score"][representation][gallery],
                baseline_result["score"][representation][gallery],
                labels[held_rows],
            )
            for gallery in ("holdout_only", "full_train_gallery")
        }
        for representation in ("float_pca128", "packed_pca128")
    }
    if {relative: sha256(source_root / relative) for relative in DEPENDENCIES} != source_hashes:
        raise ValueError("SOP substrate source changed during screen")
    receipt = {
        "schema": "sfora-sop-pretrained-substrate-screen-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN fit/holdout only; no TEST rows read",
        "seed": SEED,
        "fit_images": len(fit_rows),
        "holdout_queries": len(held_rows),
        "heldout_products": len(partition.validation_class_ids),
        "full_gallery_images": len(labels),
        "source_archive_sha256": UNICOM_L14_ARCHIVE_SHA256,
        "candidate_feature_sha256": candidate_receipt["features_sha256"],
        "candidate_receipt_sha256": sha256(candidate_receipt_path),
        "ordered_rows_sha256": ordered_hash,
        "source_sha256": source_hashes,
        "arms": {
            name: {"receipt": f"{name}.json", "sha256": sha256(args.output_dir / f"{name}.json")}
            for name in ("unicom_l14_336", "siglip2_l16_256")
        },
        "paired_product_bootstrap": paired,
        "total_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    publish_json(args.output_dir / "receipt.json", receipt)
    print(json.dumps({"result": "completed", "seconds": receipt["total_wall_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
