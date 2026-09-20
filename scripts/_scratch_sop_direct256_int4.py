#!/usr/bin/env python3
"""Throwaway learned 768-to-256 SOP projection with equal-byte int4 packing."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import positive_coverage_artifacts as artifacts
import run_sop_positive_coverage_metric as coverage
import run_sop_similarity_loss_controls as controls
import torch
from probe_representation_ceiling import load_paired_train_archives
from torch import nn
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.representation_ceiling import deterministic_class_partition

SOURCE_SHA256 = "6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818"
TEACHER_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
DIRECT128_SHA256 = "d472e6af89eccff408b5816bef64cb762bde133c7dd54077e4e84eb3a4cbaf99"
DIRECT128_RECEIPT_SHA256 = "d635a3f708aced91a0639ebe0c0139179ab883f38dd639ff153e3a69ab3c1d25"
OUTPUT_DIMENSIONS = 256
SCALE_QUANTILE = 0.999
MINIMUM_FLOAT_GAIN = 0.010
MINIMUM_PACKED_GAIN = 0.010
MAXIMUM_QUANTIZATION_LOSS = 0.003


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_pca_affine(
    rows: torch.Tensor, *, output_dimensions: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a centered PCA projection and return affine weight and bias."""

    if (
        type(rows) is not torch.Tensor
        or rows.ndim != 2
        or rows.dtype != torch.float32
        or output_dimensions <= 0
        or output_dimensions > min(rows.shape)
        or not bool(torch.isfinite(rows).all())
    ):
        raise ValueError("PCA affine authority differs")
    centre = rows.mean(dim=0)
    centred = rows - centre
    covariance = centred.T.double().matmul(centred.double()) / (rows.shape[0] - 1)
    values, vectors = torch.linalg.eigh(covariance)
    order = torch.argsort(values, descending=True)[:output_dimensions]
    weight = vectors[:, order].T.float().contiguous()
    bias = torch.mv(weight, -centre).contiguous()
    return weight, bias


def fit_pack_decode_int4(
    fit_rows: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Fit shared train-only scales, pack signed nibbles, and decode them."""

    if (
        fit_rows.ndim != 2
        or rows.ndim != 2
        or fit_rows.shape[1] != OUTPUT_DIMENSIONS
        or rows.shape[1] != OUTPUT_DIMENSIONS
        or fit_rows.dtype != torch.float32
        or rows.dtype != torch.float32
        or not bool(torch.isfinite(fit_rows).all())
        or not bool(torch.isfinite(rows).all())
    ):
        raise ValueError("int4 packing requires finite 256 dimensions")
    scales = (
        np.quantile(fit_rows.abs().numpy(), SCALE_QUANTILE, axis=0, method="linear")
        .astype(np.float32)
        / 7.0
    )
    if not np.isfinite(scales).all() or bool((scales <= 0.0).any()):
        raise ValueError("int4 scale authority differs")
    codes = torch.clamp(torch.round(rows / torch.from_numpy(scales)), -7, 7).to(torch.int8)
    unsigned = codes.numpy().view(np.uint8) & 0x0F
    packed = np.ascontiguousarray(unsigned[:, 0::2] | (unsigned[:, 1::2] << 4))
    signed = packed.view(np.int8)
    low = (signed << 4) >> 4
    high = signed >> 4
    restored = np.stack((low, high), axis=2).reshape(codes.shape)
    if packed.shape != (rows.shape[0], 128) or not np.array_equal(restored, codes.numpy()):
        raise ValueError("int4 wire authority differs")
    decoded = F.normalize(
        torch.from_numpy(restored.astype(np.float32) * scales), dim=1
    ).contiguous()
    return decoded, {
        "persistent_bytes_per_item": int(packed.shape[1]),
        "scale_fit_quantile": SCALE_QUANTILE,
        "shared_scale_bytes": int(scales.nbytes),
    }


def score(codes: torch.Tensor, labels: tuple[int, ...]) -> dict[str, object]:
    counts = Counter(labels)
    return coverage._score(
        codes, labels, torch.device("cuda")
    ) if codes.shape[1] == 128 else _score_float(codes, labels, max(counts.values()) - 1)


def _score_float(
    codes: torch.Tensor, labels: tuple[int, ...], candidate_width: int
) -> dict[str, object]:
    from probe_sop_relational_linear import score_symmetric

    result = score_symmetric(
        codes, labels, candidate_width=candidate_width, device=torch.device("cuda")
    )
    return {"map_at_r": float(result["map_at_r"]), "r1": float(result["r1"])}


def train_direct256(
    teacher_fit: torch.Tensor,
    teacher_validation: torch.Tensor,
    fit_labels: tuple[int, ...],
    validation_labels: tuple[int, ...],
) -> tuple[nn.Linear, dict[str, object]]:
    device = torch.device("cuda")
    weight, bias = fit_pca_affine(teacher_fit, output_dimensions=OUTPUT_DIMENSIONS)
    model = nn.Linear(teacher_fit.shape[1], OUTPUT_DIMENSIONS, bias=True, device=device)
    with torch.no_grad():
        model.weight.copy_(weight.to(device))
        assert model.bias is not None
        model.bias.copy_(bias.to(device))
    fit_device = teacher_fit.to(device)
    with torch.inference_mode():
        initial_fit = F.normalize(model(fit_device).float(), dim=1).contiguous()
    frozen_negatives = controls.frozen_hard_negative_index(
        initial_fit.cpu(),
        fit_labels,
        device=device,
        k=coverage.HARD_NEGATIVES,
        block_size=128,
    )
    schedule_authority = coverage._schedule(fit_labels)
    fit_label_array = np.asarray(fit_labels, dtype=np.int64)
    positive_groups = {
        label: np.flatnonzero(fit_label_array == label) for label in set(fit_labels)
    }
    learning_rate = coverage.LEARNING_RATE * math.sqrt(OUTPUT_DIMENSIONS / 768.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    started = time.monotonic()
    for step, anchors_np in enumerate(schedule_authority.row_indexes, start=1):
        anchors = torch.from_numpy(anchors_np).to(device)
        mined = frozen_negatives[anchors].to(device)
        positives_np, positive_mask_np = coverage._positive_rows(
            anchors_np, fit_label_array, positive_groups
        )
        positives = torch.from_numpy(positives_np).to(device)
        positive_mask = torch.from_numpy(positive_mask_np).to(device).contiguous()
        anchor_codes = F.normalize(model(fit_device[anchors]).float(), dim=1)
        positive_codes = F.normalize(
            model(fit_device[positives.reshape(-1)]).float(), dim=1
        ).reshape(len(anchors), positives.shape[1], -1)
        negative_codes = F.normalize(
            model(fit_device[mined.reshape(-1)]).float(), dim=1
        ).reshape(len(anchors), frozen_negatives.shape[1], -1)
        loss = controls.matched_control_loss(
            "mean_logit",
            torch.einsum("bd,bpd->bp", anchor_codes, positive_codes),
            positive_mask,
            torch.einsum("bd,bnd->bn", anchor_codes, negative_codes),
            torch.einsum("bd,bd->b", anchor_codes, initial_fit[anchors]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 100 == 0:
            print(
                json.dumps(
                    {
                        "elapsed_seconds": time.monotonic() - started,
                        "loss": losses[-1],
                        "step": step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    with torch.inference_mode():
        fit_codes = F.normalize(model(fit_device).float(), dim=1).cpu().contiguous()
        validation_codes = F.normalize(
            model(teacher_validation.to(device)).float(), dim=1
        ).cpu().contiguous()
    return model, {
        "final_loss": losses[-1],
        "fit_codes": fit_codes,
        "learning_rate": learning_rate,
        "mean_last_100_loss": artifacts.mean_recent_loss(losses, window=100),
        "schedule_sha256": schedule_authority.sha256,
        "validation_codes": validation_codes,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--direct128-checkpoint", type=Path, required=True)
    parser.add_argument("--direct128-receipt", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--execute-learned-width-gate", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preregistration = json.loads(args.preregistration.read_text())
    if (
        args.output.exists()
        or args.checkpoint.exists()
        or file_sha256(Path(__file__)) != args.script_sha256
        or file_sha256(args.preregistration) != args.preregistration_sha256
        or preregistration["script_sha256"] != args.script_sha256
        or file_sha256(args.source_snapshot) != SOURCE_SHA256
        or file_sha256(args.teacher_snapshot) != TEACHER_SHA256
        or file_sha256(args.direct128_checkpoint) != DIRECT128_SHA256
        or file_sha256(args.direct128_receipt) != DIRECT128_RECEIPT_SHA256
    ):
        raise ValueError("learned width experiment authority differs")
    configure_deterministic_similarity_runtime(0, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.monotonic()
    pair = load_paired_train_archives(
        args.source_snapshot,
        SOURCE_SHA256,
        args.teacher_snapshot,
        TEACHER_SHA256,
    )
    teacher = F.normalize(pair["teacher_train"].float(), dim=1).contiguous()
    partition = deterministic_class_partition(
        pair["train_labels"], fit_fraction=0.8, seed=coverage.SPLIT_SEED
    )
    fit_indexes = list(partition.fit_row_indexes)
    validation_indexes = list(partition.validation_row_indexes)
    fit_labels = tuple(pair["train_labels"][row] for row in fit_indexes)
    validation_labels = tuple(pair["train_labels"][row] for row in validation_indexes)
    direct128_state = torch.load(args.direct128_checkpoint, map_location="cpu", weights_only=True)
    if type(direct128_state) is not dict or set(direct128_state) != {"weight", "bias"}:
        raise ValueError("direct128 checkpoint schema differs")
    with torch.inference_mode():
        direct128_validation = F.normalize(
            F.linear(
                teacher[validation_indexes],
                direct128_state["weight"].float(),
                direct128_state["bias"].float(),
            ),
            dim=1,
        ).contiguous()
    direct128_score = coverage._score(
        direct128_validation, validation_labels, torch.device("cuda")
    )
    model, training = train_direct256(
        teacher[fit_indexes], teacher[validation_indexes], fit_labels, validation_labels
    )
    float_codes = training.pop("validation_codes")
    fit_codes = training.pop("fit_codes")
    assert isinstance(float_codes, torch.Tensor) and isinstance(fit_codes, torch.Tensor)
    int4_codes, packing = fit_pack_decode_int4(fit_codes, float_codes)
    candidate_width = max(Counter(validation_labels).values()) - 1
    float_score = _score_float(float_codes, validation_labels, candidate_width)
    int4_score = _score_float(int4_codes, validation_labels, candidate_width)
    float_gain = float(float_score["map_at_r"]) - float(direct128_score["float_map_at_r"])
    packed_gain = float(int4_score["map_at_r"]) - float(direct128_score["packed_map_at_r"])
    quantization_loss = float(float_score["map_at_r"]) - float(int4_score["map_at_r"])
    passed = (
        float_gain >= MINIMUM_FLOAT_GAIN
        and packed_gain >= MINIMUM_PACKED_GAIN
        and quantization_loss <= MAXIMUM_QUANTIZATION_LOSS
    )
    state = {
        "weight": model.weight.detach().cpu().float().contiguous(),
        "bias": model.bias.detach().cpu().float().contiguous(),
    }
    partial_checkpoint = args.checkpoint.with_suffix(args.checkpoint.suffix + ".partial")
    torch.save(state, partial_checkpoint)
    os.replace(partial_checkpoint, args.checkpoint)
    result = {
        "schema": "scratch-sop-learned-direct256-int4-v1",
        "claim_eligible": False,
        "dataset": "stanford-online-products-official-train-class-disjoint-validation",
        "official_test_touched": False,
        "fit_rows": len(fit_indexes),
        "validation_rows": len(validation_indexes),
        "inputs": {
            "direct128_checkpoint_sha256": DIRECT128_SHA256,
            "direct128_receipt_sha256": DIRECT128_RECEIPT_SHA256,
            "source_sha256": SOURCE_SHA256,
            "teacher_sha256": TEACHER_SHA256,
        },
        "method": {
            "initialization": "train-only-pca256-affine",
            "objective": "existing-frozen-mean-logit",
            "optimizer": "adam-constant-no-weight-decay-clip1",
            "output_dimensions": OUTPUT_DIMENSIONS,
            "updates": coverage.UPDATES,
            **training,
            **packing,
        },
        "direct128": {
            "float_map_at_r": direct128_score["float_map_at_r"],
            "float_r1": direct128_score["float_r1"],
            "packed_map_at_r": direct128_score["packed_map_at_r"],
            "packed_r1": direct128_score["packed_r1"],
        },
        "direct256_float": float_score,
        "direct256_int4": int4_score,
        "observed": {
            "float_map_gain": float_gain,
            "packed_map_gain": packed_gain,
            "quantization_map_loss": quantization_loss,
        },
        "gates": {
            "minimum_float_map_gain": MINIMUM_FLOAT_GAIN,
            "minimum_packed_map_gain": MINIMUM_PACKED_GAIN,
            "maximum_quantization_map_loss": MAXIMUM_QUANTIZATION_LOSS,
        },
        "passed": passed,
        "next": (
            "one claim-ineligible official positioning evaluation"
            if passed
            else "close exact learned-width construction"
        ),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "script_sha256": args.script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "elapsed_seconds": time.monotonic() - started,
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
