#!/usr/bin/env python3
"""Throwaway 1,024-bit mixed-precision margin-damage probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import run_sop_positive_coverage_metric as coverage
import run_sop_similarity_loss_controls as controls
import torch
from _scratch_cub_direct256_int4 import remap_labels, score_float, train_width
from _scratch_sop_direct256_int4 import fit_pack_decode_int4
from scipy.optimize import linear_sum_assignment
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

TOTAL_BITS = 1_024
LEVEL_COUNTS = {0: 64, 4: 128, 8: 64}
ANCHOR_COUNT = 4_096
SCALE_QUANTILE = 0.999
MINIMUM_MAP_GAIN = 0.002
MAXIMUM_MAP_LOSS = 0.002
MAXIMUM_R1_LOSS = 0.002


def _scale(rows: torch.Tensor, maximum: int) -> torch.Tensor:
    values = np.quantile(rows.abs().numpy(), SCALE_QUANTILE, axis=0, method="linear").astype(
        np.float32
    ) / float(maximum)
    if not np.isfinite(values).all() or bool((values <= 0.0).any()):
        raise ValueError("mixed-precision scale authority differs")
    return torch.from_numpy(values)


def _quantize(
    rows: torch.Tensor, scales: dict[int, torch.Tensor], levels: torch.Tensor
) -> torch.Tensor:
    result = torch.zeros_like(rows)
    for bits, maximum in ((4, 7), (8, 127)):
        mask = levels == bits
        result[:, mask] = (
            torch.clamp(torch.round(rows[:, mask] / scales[bits][mask]), -maximum, maximum)
            * scales[bits][mask]
        )
    return F.normalize(result, dim=1).contiguous()


def _margin_damage(
    rows: torch.Tensor, labels: tuple[int, ...]
) -> tuple[dict[int, torch.Tensor], torch.Tensor]:
    scales = {4: _scale(rows, 7), 8: _scale(rows, 127)}
    schedule = coverage._schedule(labels)
    anchors_np = np.concatenate(schedule.row_indexes)[:ANCHOR_COUNT]
    label_array = np.asarray(labels, dtype=np.int64)
    groups = {label: np.flatnonzero(label_array == label) for label in set(labels)}
    positives_np, positive_mask = coverage._positive_rows(anchors_np, label_array, groups)
    first_positive = positive_mask.argmax(axis=1)
    positives_np = positives_np[np.arange(len(anchors_np)), first_positive]
    negatives = controls.frozen_hard_negative_index(
        rows,
        labels,
        device=torch.device("cuda"),
        k=1,
        block_size=128,
    )[torch.from_numpy(anchors_np).to("cuda"), 0].cpu()
    anchors = rows[torch.from_numpy(anchors_np)]
    positives = rows[torch.from_numpy(positives_np)]
    negative_rows = rows[negatives]
    reference = anchors * (positives - negative_rows)
    damage = torch.empty((rows.shape[1], 3), dtype=torch.float64)
    damage[:, 0] = reference.double().square().mean(dim=0)
    quantized: dict[int, torch.Tensor] = {}
    for column, (bits, maximum) in enumerate(((4, 7), (8, 127)), start=1):
        quantized[bits] = (
            torch.clamp(torch.round(rows / scales[bits]), -maximum, maximum) * scales[bits]
        )
        candidate = quantized[bits][torch.from_numpy(anchors_np)] * (
            quantized[bits][torch.from_numpy(positives_np)] - quantized[bits][negatives]
        )
        damage[:, column] = (candidate - reference).double().square().mean(dim=0)
    return scales, damage


def _allocate(damage: torch.Tensor) -> torch.Tensor:
    if damage.shape != (256, 3):
        raise ValueError("mixed-precision damage shape differs")
    slots = np.repeat(np.asarray((0, 1, 2), dtype=np.int64), (64, 128, 64))
    costs = damage.numpy()[:, slots]
    row_indexes, column_indexes = linear_sum_assignment(costs)
    levels = torch.empty(256, dtype=torch.int64)
    levels[torch.from_numpy(row_indexes)] = torch.from_numpy(
        np.asarray((0, 4, 8), dtype=np.int64)[slots[column_indexes]].copy()
    )
    if (
        int(levels.sum()) != TOTAL_BITS
        or {bits: int((levels == bits).sum()) for bits in LEVEL_COUNTS} != LEVEL_COUNTS
    ):
        raise ValueError("mixed-precision bit budget differs")
    return levels


def _control_levels(fit: torch.Tensor, levels: torch.Tensor, *, random: bool) -> torch.Tensor:
    counts = {bits: int((levels == bits).sum()) for bits in (0, 4, 8)}
    if random:
        order = torch.randperm(256, generator=torch.Generator().manual_seed(20260920))
    else:
        order = torch.argsort(fit.var(dim=0), descending=True, stable=True)
    result = torch.zeros(256, dtype=torch.int64)
    result[order[: counts[8]]] = 8
    result[order[counts[8] : counts[8] + counts[4]]] = 4
    return result


def _load(path: Path) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...], tuple[int, ...]]:
    archive = np.load(path, allow_pickle=False)
    return (
        F.normalize(torch.from_numpy(archive["fit_embeddings"].copy()).float(), dim=1),
        F.normalize(torch.from_numpy(archive["evaluation_embeddings"].copy()).float(), dim=1),
        remap_labels(archive["fit_labels"]),
        remap_labels(archive["evaluation_labels"]),
    )


def _run(name: str, path: Path) -> dict[str, object]:
    fit, evaluation, fit_labels, evaluation_labels = _load(path)
    _, direct128, _ = train_width(fit, evaluation, fit_labels, width=128)
    baseline128 = coverage._score(direct128, evaluation_labels, torch.device("cuda"))
    direct256_fit, direct256, _ = train_width(fit, evaluation, fit_labels, width=256)
    int4, _ = fit_pack_decode_int4(direct256_fit, direct256)
    baseline256 = score_float(int4, evaluation_labels)
    scales, damage = _margin_damage(direct256_fit, fit_labels)
    levels = _allocate(damage)
    candidates = {}
    for key, candidate_levels in (
        ("margin", levels),
        ("variance", _control_levels(direct256_fit, levels, random=False)),
        ("random", _control_levels(direct256_fit, levels, random=True)),
    ):
        candidates[key] = score_float(
            _quantize(direct256, scales, candidate_levels), evaluation_labels
        )
    best_baseline_map = max(float(baseline128["packed_map_at_r"]), baseline256["map_at_r"])
    best_baseline_r1 = max(float(baseline128["packed_r1"]), baseline256["r1"])
    margin = candidates["margin"]
    result = {
        "dataset": name,
        "rows": {"fit": len(fit_labels), "evaluation": len(evaluation_labels)},
        "direct128_int8": {
            "map_at_r": float(baseline128["packed_map_at_r"]),
            "recall_at_1": float(baseline128["packed_r1"]),
        },
        "direct256_int4": baseline256,
        "allocation_counts": {str(bits): int((levels == bits).sum()) for bits in (0, 4, 8)},
        "candidates": candidates,
        "margin_vs_best": {
            "map_at_r": float(margin["map_at_r"]) - best_baseline_map,
            "recall_at_1": float(margin["r1"]) - best_baseline_r1,
        },
    }
    result["passed"] = (
        float(result["margin_vs_best"]["map_at_r"]) >= MINIMUM_MAP_GAIN
        and float(result["margin_vs_best"]["recall_at_1"]) >= -MAXIMUM_R1_LOSS
        and float(margin["map_at_r"]) >= float(candidates["variance"]["map_at_r"])
        and float(margin["map_at_r"]) >= float(candidates["random"]["map_at_r"])
    )
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--cars", type=Path, required=True)
    parser.add_argument("--cub", type=Path, required=True)
    args = parser.parse_args()
    configure_deterministic_similarity_runtime(0, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    results = [_run("cars", args.cars), _run("cub", args.cub)]
    output = {
        "schema": "scratch-mixed-precision-margin-probe-v1",
        "claim_eligible": False,
        "persistent_bits_per_item": TOTAL_BITS,
        "gates": {
            "minimum_map_gain_over_best": MINIMUM_MAP_GAIN,
            "maximum_r1_loss": MAXIMUM_R1_LOSS,
            "must_beat_controls": True,
        },
        "results": results,
        "passed": all(bool(result["passed"]) for result in results),
    }
    print(json.dumps(output, sort_keys=True, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
