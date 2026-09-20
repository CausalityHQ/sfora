#!/usr/bin/env python3
"""Matched int4-aware continuation gate over the frozen OML SOP 256-D head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import run_sop_positive_coverage_metric as coverage
import run_sop_similarity_loss_controls as controls
import torch
from _scratch_oml_sop_compact import paired_bootstrap
from _scratch_same_teacher_ladder import _score
from torch import nn
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

DIMENSIONS = 256
PERSISTENT_BYTES = 128
SCALE_QUANTILE = 0.999
MINIMUM_MAP_GAIN = 0.002
MAXIMUM_R1_LOSS = 0.0
MAXIMUM_SOURCE_MAP_GAP = 0.003


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_scales(rows: torch.Tensor) -> torch.Tensor:
    scales = np.quantile(rows.abs().numpy(), SCALE_QUANTILE, axis=0, method="linear")
    scales = np.ascontiguousarray(scales.astype(np.float32) / 7.0)
    if scales.shape != (DIMENSIONS,) or not np.isfinite(scales).all() or (scales <= 0).any():
        raise ValueError("QAT scale authority differs")
    return torch.from_numpy(scales)


def hard_int4(rows: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    codes = torch.clamp(torch.round(rows / scales), -7, 7)
    return F.normalize(codes * scales, dim=1).contiguous()


def ste_int4(rows: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    normalized = rows / scales
    hard = torch.clamp(torch.round(normalized), -7, 7)
    codes = normalized + (hard - normalized).detach()
    return F.normalize(codes * scales, dim=1)


def validate_wire(rows: torch.Tensor, scales: torch.Tensor) -> None:
    codes = torch.clamp(torch.round(rows / scales), -7, 7).to(torch.int8).numpy()
    unsigned = codes.view(np.uint8) & 0x0F
    packed = np.ascontiguousarray(unsigned[:, 0::2] | (unsigned[:, 1::2] << 4))
    signed = packed.view(np.int8)
    restored = np.stack(((signed << 4) >> 4, signed >> 4), axis=2).reshape(codes.shape)
    if packed.shape != (len(rows), PERSISTENT_BYTES) or not np.array_equal(restored, codes):
        raise ValueError("QAT int4 wire authority differs")


def continue_head(
    state: dict[str, torch.Tensor],
    fit: torch.Tensor,
    labels: tuple[int, ...],
    frozen_negatives: torch.Tensor,
    scales: torch.Tensor,
    *,
    quantization_aware: bool,
) -> tuple[nn.Linear, dict[str, object]]:
    device = torch.device("cuda")
    model = nn.Linear(fit.shape[1], DIMENSIONS, bias=True, device=device)
    with torch.no_grad():
        model.weight.copy_(state["weight"].to(device))
        assert model.bias is not None
        model.bias.copy_(state["bias"].to(device))
    fit_device = fit.to(device)
    scales_device = scales.to(device)
    with torch.inference_mode():
        initial_fit = F.normalize(model(fit_device).float(), dim=1).contiguous()
    schedule = coverage._schedule(labels)
    label_array = np.asarray(labels, dtype=np.int64)
    positive_groups = {
        label: np.flatnonzero(label_array == label) for label in set(labels)
    }
    learning_rate = coverage.LEARNING_RATE * math.sqrt(DIMENSIONS / 768.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    started = time.monotonic()
    transform = ste_int4 if quantization_aware else (lambda rows, _scales: rows)
    for step, anchors_np in enumerate(schedule.row_indexes, start=1):
        anchors = torch.from_numpy(anchors_np).to(device)
        mined = frozen_negatives[anchors].to(device)
        positives_np, positive_mask_np = coverage._positive_rows(
            anchors_np, label_array, positive_groups
        )
        positives = torch.from_numpy(positives_np).to(device)
        positive_mask = torch.from_numpy(positive_mask_np).to(device).contiguous()
        anchor_float = F.normalize(model(fit_device[anchors]).float(), dim=1)
        positive_float = F.normalize(
            model(fit_device[positives.reshape(-1)]).float(), dim=1
        )
        negative_float = F.normalize(
            model(fit_device[mined.reshape(-1)]).float(), dim=1
        )
        anchor_codes = transform(anchor_float, scales_device)
        positive_codes = transform(positive_float, scales_device).reshape(
            len(anchors), positives.shape[1], -1
        )
        negative_codes = transform(negative_float, scales_device).reshape(
            len(anchors), frozen_negatives.shape[1], -1
        )
        loss = controls.matched_control_loss(
            "mean_logit",
            torch.einsum("bd,bpd->bp", anchor_codes, positive_codes),
            positive_mask,
            torch.einsum("bd,bnd->bn", anchor_codes, negative_codes),
            torch.einsum("bd,bd->b", anchor_float, initial_fit[anchors]),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 250 == 0:
            print(
                json.dumps(
                    {
                        "arm": "qat" if quantization_aware else "float_control",
                        "elapsed_seconds": time.monotonic() - started,
                        "loss": losses[-1],
                        "step": step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    return model, {
        "final_loss": losses[-1],
        "learning_rate": learning_rate,
        "mean_last_100_loss": float(np.mean(losses[-100:])),
        "schedule_sha256": schedule.sha256,
        "updates": len(losses),
    }


def evaluate(
    model: nn.Linear,
    fit: torch.Tensor,
    test: torch.Tensor,
    scales: torch.Tensor,
    labels: tuple[int, ...],
) -> tuple[dict[str, object], dict[str, object]]:
    device = torch.device("cuda")
    with torch.inference_mode():
        fit_float = F.normalize(model(fit.to(device)).float(), dim=1).cpu().contiguous()
        test_float = F.normalize(model(test.to(device)).float(), dim=1).cpu().contiguous()
    validate_wire(fit_float, scales)
    decoded = hard_int4(test_float, scales)
    float_score = _score(
        test_float, test_float, labels, labels, same_rows=True, device=device
    )
    int4_score = _score(decoded, decoded, labels, labels, same_rows=True, device=device)
    return float_score, int4_score


def summary(score: dict[str, object]) -> dict[str, float]:
    return {
        "map_at_r": float(score["map_at_r"]),
        "recall_at_1": float(score["recall_at_1"]),
    }


def contrast(candidate: dict[str, object], baseline: dict[str, object]) -> dict[str, object]:
    return {
        "map_at_r": paired_bootstrap(candidate["per_query_ap"], baseline["per_query_ap"]),
        "recall_at_1": paired_bootstrap(
            candidate["per_query_r1"], baseline["per_query_r1"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--parent-result", type=Path, required=True)
    parser.add_argument("--parent-result-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-int4-qat-gate", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    for path, expected in (
        (args.features, args.features_sha256),
        (args.checkpoint, args.checkpoint_sha256),
        (args.parent_result, args.parent_result_sha256),
        (args.preregistration, args.preregistration_sha256),
        (Path(__file__), args.script_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError("QAT authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    expected = {
        "checkpoint_sha256": args.checkpoint_sha256,
        "features_sha256": args.features_sha256,
        "parent_result_sha256": args.parent_result_sha256,
        "script_sha256": args.script_sha256,
        "source_commit": args.source_commit,
    }
    if any(preregistration.get(key) != value for key, value in expected.items()):
        raise ValueError("QAT preregistration binding differs")
    parent = json.loads(args.parent_result.read_text())
    if (
        parent.get("representation", {}).get("persistent_bytes_per_item") != PERSISTENT_BYTES
        or parent.get("checkpoint_sha256") != args.checkpoint_sha256
    ):
        raise ValueError("QAT parent result differs")

    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    with np.load(args.features, allow_pickle=False) as archive:
        fit = F.normalize(
            torch.from_numpy(np.ascontiguousarray(archive["train_features"])).float(), dim=1
        ).contiguous()
        fit_labels = tuple(
            int(value)
            for value in np.ascontiguousarray(archive["train_labels"], dtype=np.int64).tolist()
        )
        test = F.normalize(
            torch.from_numpy(np.ascontiguousarray(archive["test_features"])).float(), dim=1
        ).contiguous()
        test_labels = tuple(
            int(value)
            for value in np.ascontiguousarray(archive["test_labels"], dtype=np.int64).tolist()
        )
    if fit.shape != (59_551, 384) or test.shape != (60_502, 384):
        raise ValueError("QAT feature shape differs")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if type(state) is not dict or set(state) != {"bias", "weight"}:
        raise ValueError("QAT checkpoint schema differs")
    initial_model = nn.Linear(384, DIMENSIONS, bias=True, device=device)
    with torch.no_grad():
        initial_model.weight.copy_(state["weight"].to(device))
        assert initial_model.bias is not None
        initial_model.bias.copy_(state["bias"].to(device))
        initial_fit = F.normalize(initial_model(fit.to(device)).float(), dim=1).cpu().contiguous()
    scales = fit_scales(initial_fit)
    frozen_negatives = controls.frozen_hard_negative_index(
        initial_fit,
        fit_labels,
        device=device,
        k=coverage.HARD_NEGATIVES,
        block_size=128,
    )
    source_score = _score(test, test, test_labels, test_labels, same_rows=True, device=device)
    initial_float, initial_int4 = evaluate(initial_model, fit, test, scales, test_labels)
    float_model, float_training = continue_head(
        state,
        fit,
        fit_labels,
        frozen_negatives,
        scales,
        quantization_aware=False,
    )
    float_float, float_int4 = evaluate(float_model, fit, test, scales, test_labels)
    del float_model
    torch.cuda.empty_cache()
    qat_model, qat_training = continue_head(
        state,
        fit,
        fit_labels,
        frozen_negatives,
        scales,
        quantization_aware=True,
    )
    qat_float, qat_int4 = evaluate(qat_model, fit, test, scales, test_labels)
    qat_vs_control = contrast(qat_int4, float_int4)
    qat_vs_initial = contrast(qat_int4, initial_int4)
    qat_vs_source = contrast(qat_int4, source_score)
    passed = bool(
        float(qat_vs_control["map_at_r"]["delta"]) >= MINIMUM_MAP_GAIN
        and float(qat_vs_control["map_at_r"]["ci95"][0]) > 0.0
        and float(qat_vs_control["recall_at_1"]["delta"]) >= MAXIMUM_R1_LOSS
        and float(qat_vs_initial["map_at_r"]["delta"]) >= MINIMUM_MAP_GAIN
        and float(qat_vs_initial["map_at_r"]["ci95"][0]) > 0.0
        and float(qat_vs_initial["recall_at_1"]["delta"]) >= MAXIMUM_R1_LOSS
        and float(source_score["map_at_r"]) - float(qat_int4["map_at_r"])
        <= MAXIMUM_SOURCE_MAP_GAP
        and float(qat_vs_source["map_at_r"]["ci95"][0]) >= -MAXIMUM_SOURCE_MAP_GAP
    )
    result = {
        "schema": "scratch-oml-sop-int4-qat-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "dataset": "Stanford Online Products",
        "split": "official train fit; official leave-one-out test evaluation",
        "representation": {
            "dimensions": DIMENSIONS,
            "persistent_bytes_per_item": PERSISTENT_BYTES,
            "scale_fit_quantile": SCALE_QUANTILE,
            "shared_scale_bytes": int(scales.numel() * scales.element_size()),
            "symmetric_query_and_gallery_quantization": True,
        },
        "arms": {
            "source_float384": summary(source_score),
            "initial_float256": summary(initial_float),
            "initial_int4": summary(initial_int4),
            "float_continuation_float256": summary(float_float),
            "float_continuation_int4": summary(float_int4),
            "qat_float256": summary(qat_float),
            "qat_int4": summary(qat_int4),
        },
        "contrasts": {
            "qat_int4_minus_float_continuation_int4": qat_vs_control,
            "qat_int4_minus_initial_int4": qat_vs_initial,
            "qat_int4_minus_source_float384": qat_vs_source,
        },
        "training": {
            "float_control": float_training,
            "qat": qat_training,
            "frozen_negative_count": coverage.HARD_NEGATIVES,
            "frozen_scale_sha256": hashlib.sha256(scales.numpy().tobytes()).hexdigest(),
        },
        "gate": {
            "minimum_map_gain": MINIMUM_MAP_GAIN,
            "maximum_recall_at_1_loss": MAXIMUM_R1_LOSS,
            "maximum_source_map_gap": MAXIMUM_SOURCE_MAP_GAP,
            "passed": passed,
            "next": "replicate_unchanged_on_oml_inshop" if passed else "kill_exact_family",
        },
        "elapsed_seconds": time.perf_counter() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated(device)),
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="", flush=True)


if __name__ == "__main__":
    main()
