#!/usr/bin/env python3
"""Replicate the frozen learned-width hypothesis on class-disjoint CUB."""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import run_sop_positive_coverage_metric as coverage
import run_sop_similarity_loss_controls as controls
import torch
from _scratch_sop_direct256_int4 import (
    file_sha256,
    fit_pack_decode_int4,
    fit_pca_affine,
)
from probe_sop_relational_linear import score_symmetric
from torch import nn
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

ARCHIVE_SHA256 = "847a40bd8c0c2a5289de9b5a8eca93c935d4eafed6507e1360da3a3bfee6f62e"
MINIMUM_WIDTH_GAIN = 0.005
MAXIMUM_QUANTIZATION_LOSS = 0.003
MAXIMUM_R1_DECLINE = 0.002


def remap_labels(values: np.ndarray) -> tuple[int, ...]:
    mapping = {value: index + 1 for index, value in enumerate(sorted(np.unique(values)))}
    return tuple(mapping[value] for value in values)


def score_float(codes: torch.Tensor, labels: tuple[int, ...]) -> dict[str, float]:
    raw = score_symmetric(
        codes,
        labels,
        candidate_width=max(Counter(labels).values()) - 1,
        device=torch.device("cuda"),
    )
    return {"map_at_r": float(raw["map_at_r"]), "r1": float(raw["r1"])}


def train_width(
    fit: torch.Tensor,
    evaluation: torch.Tensor,
    fit_labels: tuple[int, ...],
    *,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float | int | str]]:
    device = torch.device("cuda")
    weight, bias = fit_pca_affine(fit, output_dimensions=width)
    model = nn.Linear(fit.shape[1], width, bias=True, device=device)
    with torch.no_grad():
        model.weight.copy_(weight.to(device))
        assert model.bias is not None
        model.bias.copy_(bias.to(device))
    fit_device = fit.to(device)
    with torch.inference_mode():
        initial_fit = F.normalize(model(fit_device).float(), dim=1).contiguous()
    negatives = controls.frozen_hard_negative_index(
        initial_fit.cpu(),
        fit_labels,
        device=device,
        k=coverage.HARD_NEGATIVES,
        block_size=128,
    )
    schedule = coverage._schedule(fit_labels)
    label_array = np.asarray(fit_labels, dtype=np.int64)
    positive_groups = {
        label: np.flatnonzero(label_array == label) for label in set(fit_labels)
    }
    learning_rate = coverage.LEARNING_RATE * math.sqrt(width / 768.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    losses: list[float] = []
    started = time.monotonic()
    for step, anchors_np in enumerate(schedule.row_indexes, start=1):
        anchors = torch.from_numpy(anchors_np).to(device)
        mined = negatives[anchors]
        positives_np, mask_np = coverage._positive_rows(
            anchors_np, label_array, positive_groups
        )
        positives = torch.from_numpy(positives_np).to(device)
        mask = torch.from_numpy(mask_np).to(device).contiguous()
        anchor_codes = F.normalize(model(fit_device[anchors]).float(), dim=1)
        positive_codes = F.normalize(
            model(fit_device[positives.reshape(-1)]).float(), dim=1
        ).reshape(len(anchors), positives.shape[1], -1)
        negative_codes = F.normalize(
            model(fit_device[mined.reshape(-1)]).float(), dim=1
        ).reshape(len(anchors), negatives.shape[1], -1)
        loss = controls.matched_control_loss(
            "mean_logit",
            torch.einsum("bd,bpd->bp", anchor_codes, positive_codes),
            mask,
            torch.einsum("bd,bnd->bn", anchor_codes, negative_codes),
            torch.einsum("bd,bd->b", anchor_codes, initial_fit[anchors]),
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
                        "elapsed_seconds": time.monotonic() - started,
                        "loss": losses[-1],
                        "step": step,
                        "width": width,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    with torch.inference_mode():
        fit_codes = F.normalize(model(fit_device).float(), dim=1).cpu().contiguous()
        evaluation_codes = F.normalize(
            model(evaluation.to(device)).float(), dim=1
        ).cpu().contiguous()
    return fit_codes, evaluation_codes, {
        "elapsed_seconds": time.monotonic() - started,
        "final_loss": losses[-1],
        "learning_rate": learning_rate,
        "schedule_sha256": schedule.sha256,
        "updates": coverage.UPDATES,
        "width": width,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-cub-replication", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    preregistration = json.loads(args.preregistration.read_text())
    if (
        args.output.exists()
        or file_sha256(Path(__file__)) != args.script_sha256
        or file_sha256(args.preregistration) != args.preregistration_sha256
        or preregistration["script_sha256"] != args.script_sha256
        or file_sha256(args.archive) != ARCHIVE_SHA256
    ):
        raise ValueError("CUB learned-width authority differs")
    configure_deterministic_similarity_runtime(0, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.monotonic()
    archive = np.load(args.archive, allow_pickle=False)
    metadata = json.loads(str(archive["metadata_json"].item()))
    if (
        metadata.get("schema") != "scratch-cub-unicom-class-disjoint-v1"
        or metadata.get("dimensions") != 768
        or metadata.get("counts") != {"evaluation": 2857, "fit": 2997}
    ):
        raise ValueError("CUB feature schema differs")
    fit = F.normalize(torch.from_numpy(archive["fit_embeddings"]).float(), dim=1).contiguous()
    evaluation = F.normalize(
        torch.from_numpy(archive["evaluation_embeddings"]).float(), dim=1
    ).contiguous()
    fit_labels = remap_labels(archive["fit_labels"].astype(np.int64))
    evaluation_labels = remap_labels(archive["evaluation_labels"].astype(np.int64))
    pca128_weight, pca128_bias = fit_pca_affine(fit, output_dimensions=128)
    pca128_eval = F.normalize(
        F.linear(evaluation, pca128_weight, pca128_bias), dim=1
    ).contiguous()
    pca128_score = coverage._score(pca128_eval, evaluation_labels, torch.device("cuda"))
    _, direct128_eval, direct128_training = train_width(
        fit, evaluation, fit_labels, width=128
    )
    direct128_score = coverage._score(
        direct128_eval, evaluation_labels, torch.device("cuda")
    )
    direct256_fit, direct256_eval, direct256_training = train_width(
        fit, evaluation, fit_labels, width=256
    )
    direct256_int4, packing = fit_pack_decode_int4(direct256_fit, direct256_eval)
    direct256_float_score = score_float(direct256_eval, evaluation_labels)
    direct256_int4_score = score_float(direct256_int4, evaluation_labels)
    width_gain = float(direct256_int4_score["map_at_r"]) - float(
        direct128_score["packed_map_at_r"]
    )
    quantization_loss = float(direct256_float_score["map_at_r"]) - float(
        direct256_int4_score["map_at_r"]
    )
    r1_delta = float(direct256_int4_score["r1"]) - float(direct128_score["packed_r1"])
    passed = (
        width_gain >= MINIMUM_WIDTH_GAIN
        and quantization_loss <= MAXIMUM_QUANTIZATION_LOSS
        and r1_delta >= -MAXIMUM_R1_DECLINE
    )
    result = {
        "schema": "scratch-cub-learned-direct256-int4-replication-v1",
        "claim_eligible": False,
        "dataset": "cub-200-2011-sha256-class-disjoint-fit-evaluation",
        "rows": {"fit": len(fit_labels), "evaluation": len(evaluation_labels)},
        "inputs": {"archive_sha256": ARCHIVE_SHA256},
        "pca128_int8": {
            "map_at_r": pca128_score["packed_map_at_r"],
            "r1": pca128_score["packed_r1"],
        },
        "direct128_int8": {
            "map_at_r": direct128_score["packed_map_at_r"],
            "r1": direct128_score["packed_r1"],
            "training": direct128_training,
        },
        "direct256_float": direct256_float_score,
        "direct256_int4": {
            **direct256_int4_score,
            **packing,
            "training": direct256_training,
        },
        "observed": {
            "packed_width_map_gain": width_gain,
            "quantization_map_loss": quantization_loss,
            "packed_r1_delta": r1_delta,
        },
        "gates": {
            "minimum_packed_width_map_gain": MINIMUM_WIDTH_GAIN,
            "maximum_quantization_map_loss": MAXIMUM_QUANTIZATION_LOSS,
            "maximum_packed_r1_decline": MAXIMUM_R1_DECLINE,
        },
        "passed": passed,
        "next": (
            "freeze recipe for compact multi-domain panel"
            if passed
            else "close learned-width generality claim"
        ),
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
