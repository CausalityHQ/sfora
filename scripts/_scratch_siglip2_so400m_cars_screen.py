#!/usr/bin/env python3
"""Fail-fast Cars196 frozen SigLIP2 SO400M compact-quality screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from sfora.deterministic_similarity_runtime import (
    configure_deterministic_similarity_runtime,
)

_SUPPORT_PATH = Path(__file__).with_name(
    "_scratch_unicom_lastblock_rank_finish.py"
)
_SUPPORT_SPEC = importlib.util.spec_from_file_location(
    "scratch_unicom_lastblock_rank_finish_support", _SUPPORT_PATH
)
if _SUPPORT_SPEC is None or _SUPPORT_SPEC.loader is None:
    raise RuntimeError("rank-finish support module is unavailable")
_SUPPORT = importlib.util.module_from_spec(_SUPPORT_SPEC)
_SUPPORT_SPEC.loader.exec_module(_SUPPORT)
_fit_score = _SUPPORT._fit_score
paired_class_bootstrap = _SUPPORT.paired_class_bootstrap
sha256 = _SUPPORT.sha256

DATASET_ID = "tanganke/stanford_cars"
DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
MODEL_ID = "google/siglip2-so400m-patch14-384"
MODEL_REVISION = "e8e487298228002f3d8a82e0cd5c8ea9c567f57f"
MODEL_SHA256 = "9f4f4a49f908ef0c979bce8ff5a5c0e88882dc6c5dc4304387cbbd152558e2c2"
CONFIG_SHA256 = "73477b47ae9a395f008f993ebd8fb5c16ce0b5e8419ffdf2f5b2183b260f9cff"
PROCESSOR_SHA256 = "fb2817d3523ca3b666c859f15320c7138416bc38ffc515e2963f78c868c51c90"
BASELINE_FEATURE_SHA256 = "6ddfd7e2c9fd489dff51fa33697c62abf92a45247b52b336c0371f5482231ab3"
BASELINE = {"map_at_r": 0.860662, "recall_at_1": 0.974542}


def _metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("SigLIP2 metric authority differs")
    return value


def classify_candidate(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    interval: Mapping[str, object],
) -> dict[str, object]:
    """Apply the frozen joint Cars teacher-screen gate."""

    map_delta = _metric(candidate, "map_at_r") - _metric(baseline, "map_at_r")
    recall_delta = _metric(candidate, "recall_at_1") - _metric(
        baseline, "recall_at_1"
    )
    lower = interval.get("lower")
    if type(lower) is not float or not math.isfinite(lower):
        raise ValueError("SigLIP2 interval authority differs")
    return {
        "map_at_r_delta": map_delta,
        "recall_at_1_delta": recall_delta,
        "minimum_map_at_r_delta": 0.005,
        "require_positive_interval_lower": True,
        "minimum_recall_at_1_delta": 0.0,
        "passed": map_delta >= 0.005 and lower > 0.0 and recall_delta >= 0.0,
    }


def validate_processor_authority(config: Mapping[str, object]) -> None:
    """Reject preprocessing drift from the frozen upstream snapshot."""

    expected = {
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
        "resample": 2,
        "rescale_factor": 1 / 255,
        "size": {"height": 384, "width": 384},
    }
    if any(config.get(key) != value for key, value in expected.items()):
        raise ValueError("SigLIP2 processor authority differs")


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(
        self, source: object, indices: Sequence[int], processor: object
    ) -> None:
        self.source = source
        self.indices = tuple(indices)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> torch.Tensor:
        row = self.source[self.indices[index]]  # type: ignore[index]
        image = row["image"].convert("RGB")
        return self.processor(images=image, return_tensors="pt")[  # type: ignore[operator]
            "pixel_values"
        ][0]


def _encode(
    model: torch.nn.Module,
    source: object,
    indices: Sequence[int],
    processor: object,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(
        _ImageDataset(source, indices, processor),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    rows = []
    model.eval()
    with torch.inference_mode():
        for step, images in enumerate(loader, start=1):
            output = model(images.to(device=device, dtype=torch.float16))
            rows.append(output.pooler_output.float().cpu())
            if step % 25 == 0:
                print(
                    json.dumps(
                        {"encoded_batches": step, "total_batches": len(loader)},
                        sort_keys=True,
                    ),
                    flush=True,
                )
    features = torch.cat(rows).contiguous()
    if features.shape != (len(indices), 1152) or not bool(
        torch.isfinite(features).all()
    ):
        raise ValueError("SigLIP2 feature authority differs")
    return features


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--baseline-features", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute-screen", action="store_true", required=True)
    return parser.parse_args(arguments)


def _feature_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    model_file = args.model_snapshot / "model.safetensors"
    config_file = args.model_snapshot / "config.json"
    processor_file = args.model_snapshot / "preprocessor_config.json"
    if (
        args.output.exists()
        or args.model_snapshot.name != MODEL_REVISION
        or args.batch_size != 32
        or args.workers != 8
        or sha256(model_file) != MODEL_SHA256
        or sha256(config_file) != CONFIG_SHA256
        or sha256(processor_file) != PROCESSOR_SHA256
        or sha256(args.baseline_features) != BASELINE_FEATURE_SHA256
        or sha256(args.preregistration) != args.preregistration_sha256
    ):
        raise ValueError("SigLIP2 screen input authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration.get("schema") != "sfora-siglip2-so400m-cars-screen-v1"
        or preregistration.get("script_sha256") != sha256(Path(__file__))
    ):
        raise ValueError("SigLIP2 screen preregistration authority differs")
    processor_config = json.loads(processor_file.read_text())
    validate_processor_authority(processor_config)

    configure_deterministic_similarity_runtime(17, cpu_threads=8)
    from datasets import concatenate_datasets, load_dataset
    from transformers import AutoImageProcessor, SiglipVisionModel

    source_bundle = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    source = concatenate_datasets((source_bundle["train"], source_bundle["test"]))
    labels = np.asarray(source["label"], dtype=np.int64)
    fit_indices = np.flatnonzero(labels < 98).tolist()
    evaluation_indices = np.flatnonzero(labels >= 98).tolist()
    if (
        labels.shape != (16_185,)
        or len(fit_indices) != 8_054
        or len(evaluation_indices) != 8_131
    ):
        raise ValueError("Cars196 row authority differs")

    with np.load(args.baseline_features, allow_pickle=False) as archive:
        baseline_fit = torch.from_numpy(
            np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32)
        )
        fit_labels = torch.from_numpy(
            np.ascontiguousarray(archive["fit_labels"], dtype=np.int64)
        )
        baseline_evaluation = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_embeddings"], dtype=np.float32)
        )
        evaluation_labels = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_labels"], dtype=np.int64)
        )
    if (
        baseline_fit.shape != (8_054, 768)
        or baseline_evaluation.shape != (8_131, 768)
        or not np.array_equal(fit_labels.numpy(), labels[fit_indices])
        or not np.array_equal(evaluation_labels.numpy(), labels[evaluation_indices])
    ):
        raise ValueError("baseline feature authority differs")

    device = torch.device("cuda")
    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True
    )
    model = SiglipVisionModel.from_pretrained(
        args.model_snapshot,
        local_files_only=True,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to(device)
    parameter_count = sum(value.numel() for value in model.parameters())
    if parameter_count != 428_225_600:
        raise ValueError("SigLIP2 parameter authority differs")

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    candidate_fit = _encode(
        model,
        source,
        fit_indices,
        processor,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    candidate_evaluation = _encode(
        model,
        source,
        evaluation_indices,
        processor,
        device=device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    baseline = _fit_score(
        baseline_fit,
        fit_labels,
        baseline_evaluation,
        evaluation_labels,
        device=device,
    )
    candidate = _fit_score(
        candidate_fit,
        fit_labels,
        candidate_evaluation,
        evaluation_labels,
        device=device,
    )
    interval = paired_class_bootstrap(
        candidate["per_query_ap"],  # type: ignore[arg-type]
        baseline["per_query_ap"],  # type: ignore[arg-type]
        evaluation_labels.numpy(),
    )
    reproduction_ok = (
        abs(float(baseline["map_at_r"]) - BASELINE["map_at_r"]) <= 0.001
        and abs(float(baseline["recall_at_1"]) - BASELINE["recall_at_1"]) <= 0.001
    )
    decision = classify_candidate(baseline, candidate, interval)
    decision["baseline_reproduction"] = reproduction_ok
    decision["passed"] = bool(decision["passed"] and reproduction_ok)
    del baseline["per_query_ap"]
    del candidate["per_query_ap"]

    result = {
        "schema": "sfora-siglip2-so400m-cars-screen-result-v1",
        "claim_eligible": False,
        "dataset": "cars196-class-disjoint",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "model_sha256": MODEL_SHA256,
        "config_sha256": CONFIG_SHA256,
        "processor_sha256": PROCESSOR_SHA256,
        "parameter_count": parameter_count,
        "candidate_feature_sha256": {
            "fit": _feature_sha256(candidate_fit),
            "evaluation": _feature_sha256(candidate_evaluation),
        },
        "preregistration_sha256": args.preregistration_sha256,
        "script_sha256": sha256(Path(__file__)),
        "compact_representation": {
            "kind": "power-whitening-int8-128-plus-f16-inverse-norm",
            "bytes_per_item": 130,
            "alpha": 0.75,
            "regularization": 1.0,
        },
        "baseline": baseline,
        "candidate": candidate,
        "paired_class_bootstrap_map_delta_95": interval,
        "decision": decision,
        "next_action": "replicate-unchanged-on-cub" if decision["passed"] else "close-family",
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    print(json.dumps({"output": str(args.output), "decision": decision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
