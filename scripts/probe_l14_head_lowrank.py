#!/usr/bin/env python3
"""Train-only, reject-only low-rank flatten-head screen for authenticated UNICOM L/14."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from evaluate_sop_reference_checkpoint import EXPECTED_SOP_RECORD_SHA256
from export_unicom_sop_embeddings import ordered_record_sha256, parse_sop_records
from PIL import Image
from probe_l14_parallel_fold import (
    assert_no_foreign_gpu_processes,
    load_authenticated_l14,
    timed_call,
    timing_summary,
)
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from train_sop_compact_backbone import (
    FIT_FRACTION,
    SPLIT_SEED,
    publish_file_noreplace,
    sha256,
)

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_evaluation import score_symmetric


def factorize_linear(source: nn.Linear, rank: int) -> tuple[nn.Sequential, float]:
    """Return the optimal rank-r Frobenius approximation U(U.T W)."""

    if (
        not isinstance(source, nn.Linear)
        or source.bias is not None
        or source.weight.device.type != "cpu"
        or source.weight.dtype != torch.float32
        or not isinstance(rank, int)
        or not 1 <= rank <= min(source.in_features, source.out_features)
    ):
        raise ValueError("L/14 low-rank head rank or source geometry differs")
    weight = source.weight.detach()
    if not bool(torch.isfinite(weight).all()):
        raise ValueError("L/14 low-rank head has nonfinite weight")
    gram = weight @ weight.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    selected = eigenvectors[:, -rank:].contiguous()
    captured = float(eigenvalues[-rank:].sum() / eigenvalues.clamp_min(0).sum())
    first = nn.Linear(source.in_features, rank, bias=False)
    second = nn.Linear(rank, source.out_features, bias=False)
    with torch.no_grad():
        first.weight.copy_(selected.T @ weight)
        second.weight.copy_(selected)
    return nn.Sequential(first, second).eval(), captured


class PinnedTrainImages(Dataset):
    def __init__(self, records: tuple, transform: object, manifest: bytes) -> None:
        if len(manifest) != 32 * len(records):
            raise ValueError("SOP train holdout image manifest differs")
        self.records = records
        self.transform = transform
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.records[index]
        data = row.image_path.read_bytes()
        if hashlib.sha256(data).digest() != self.manifest[index * 32 : (index + 1) * 32]:
            raise ValueError("SOP train holdout image content changed")
        with Image.open(BytesIO(data)) as image:
            return self.transform(image.convert("RGB")), row.label


@torch.inference_mode()
def encode(model: nn.Module, loader: DataLoader, expected_rows: int) -> tuple[torch.Tensor, float]:
    started = time.perf_counter()
    outputs = [model(images.cuda(non_blocking=True)).float().cpu() for images, _ in loader]
    values = torch.cat(outputs).contiguous()
    if values.shape != (expected_rows, 768) or not bool(torch.isfinite(values).all()):
        raise ValueError("L/14 low-rank descriptor geometry differs")
    return values, time.perf_counter() - started


def score_features(values: torch.Tensor, labels: tuple[int, ...]) -> dict[str, object]:
    gpu_values = values.cuda()
    gpu_labels = torch.tensor(labels, dtype=torch.int64, device="cuda")
    packed = pack_int8_unit_embeddings(F.normalize(gpu_values, dim=1))
    return {
        "full_float_cosine": score_symmetric(gpu_values, gpu_labels),
        "full_packed_cosine": score_symmetric(
            packed.codes.float(), gpu_labels, inverse_norms=packed.inverse_norms
        ),
        "upstream_prefix512_euclidean": score_symmetric(
            gpu_values, gpu_labels, prefix_euclidean_dimensions=512
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("unicom-checkout", "checkpoint", "sop-root", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--rank", type=int, default=512)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--calls-per-block", type=int, default=10)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--execute-l14-lowrank-head-screen", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or args.rank != 512
        or args.blocks < 2
        or args.calls_per_block < 2
        or args.workers < 0
        or not torch.cuda.is_available()
    ):
        raise ValueError("L/14 low-rank head invocation differs")
    from evaluate_sop_cub_transfer import gpu_compute_pids

    if gpu_compute_pids():
        raise ValueError("L/14 low-rank head GPU occupied")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    started = time.perf_counter()
    all_records = parse_sop_records(args.sop_root)
    if (
        ordered_record_sha256(all_records) != EXPECTED_SOP_RECORD_SHA256
        or sha256(args.sop_root / "Ebay_train.txt")
        != "77abb1e82af49f2f1f272dc6bd0b8480904f58742091764f9511b152cad1824e"
    ):
        raise ValueError("L/14 low-rank SOP source differs")
    train = tuple(row for row in all_records if row.split == "train")
    split = deterministic_class_partition(
        tuple(row.label for row in train), fit_fraction=FIT_FRACTION, seed=SPLIT_SEED
    )
    selected = tuple(train[index] for index in split.validation_row_indexes)
    manifest = b"".join(hashlib.sha256(row.image_path.read_bytes()).digest() for row in selected)
    model, transform = load_authenticated_l14(args.unicom_checkout, args.checkpoint)
    source_head = model.feature[0]
    if source_head.weight.shape != (1024, 589824):
        raise ValueError("L/14 low-rank head source differs")
    compressed_head, spectral_energy = factorize_linear(source_head, args.rank)
    compressed_head = compressed_head.cuda().eval()
    model = model.cuda().eval()
    source_head = model.feature[0]
    assert_no_foreign_gpu_processes()
    loader = DataLoader(
        PinnedTrainImages(selected, transform, manifest),
        batch_size=32,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )
    diagnostics = {}
    for arm, head in (("original", source_head), ("rank512", compressed_head)):
        model.feature[0] = head
        features, seconds = encode(model, loader, len(selected))
        diagnostics[arm] = {
            "encode_seconds_including_decode": seconds,
            "quality": score_features(features, tuple(row.label for row in selected)),
        }
        if arm == "original":
            reference_features = features
        else:
            normalized = torch.nn.functional.normalize(reference_features, dim=1)
            candidate = torch.nn.functional.normalize(features, dim=1)
            diagnostics[arm]["mean_descriptor_cosine_to_original"] = float(
                (normalized * candidate).sum(dim=1).mean()
            )
        del features
    sampled_indexes = np.linspace(0, len(selected) - 1, 32, dtype=int)
    sample = torch.stack([loader.dataset[index][0] for index in sampled_indexes]).cuda()
    timings: dict[str, object] = {}
    for size in (1, 32):
        inputs = sample[:size]
        for head in (source_head, compressed_head):
            model.feature[0] = head
            for _ in range(5):
                timed_call(model, inputs, (size, 768))
        samples = {
            "original": {"gpu_ms": [], "wall_ms": []},
            "rank512": {"gpu_ms": [], "wall_ms": []},
        }
        order = []
        for block in range(args.blocks):
            arms = (("original", source_head), ("rank512", compressed_head))
            if block % 2:
                arms = tuple(reversed(arms))
            for arm, head in arms:
                order.append(arm)
                model.feature[0] = head
                for _ in range(args.calls_per_block):
                    gpu_ms, wall_ms = timed_call(model, inputs, (size, 768))
                    samples[arm]["gpu_ms"].append(gpu_ms)
                    samples[arm]["wall_ms"].append(wall_ms)
        timings[str(size)] = {
            "samples": samples,
            "interleaved_order": order,
            "summary": {
                arm: {kind: timing_summary(values) for kind, values in columns.items()}
                for arm, columns in samples.items()
            },
        }
        print(json.dumps({"batch": size, "summary": timings[str(size)]["summary"]}), flush=True)
    assert_no_foreign_gpu_processes()
    receipt = {
        "schema": "sfora-l14-lowrank-head-train-holdout-screen-v1",
        "claim_eligible": False,
        "p99_contract_satisfied": False,
        "split": "SOP train identities, disjoint 90/10 class split",
        "split_seed": SPLIT_SEED,
        "rank": args.rank,
        "spectral_energy_fraction": spectral_energy,
        "selected_train_image_ids": [row.image_id for row in selected],
        "selected_train_labels": [row.label for row in selected],
        "selected_train_image_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "quality": diagnostics,
        "timing": timings,
        "timing_scope": (
            "preprocessed SOP training tensors resident on GPU; fp32 weights with "
            "source internal fp16 autocast"
        ),
        "hardware": {
            "gpu": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "inputs": {
            "checkpoint_sha256": sha256(args.checkpoint),
            "ordered_sop_records_sha256": EXPECTED_SOP_RECORD_SHA256,
            "source_sha256": sha256(Path(__file__)),
        },
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "elapsed_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    publish_file_noreplace(
        args.output,
        lambda stream: stream.write(
            (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
        ),
    )
    print(json.dumps({"output": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
