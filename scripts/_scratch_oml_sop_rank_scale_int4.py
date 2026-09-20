#!/usr/bin/env python3
"""Fit-only ranking-aware int4 scale diagnostic for the learned OML SOP head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from _scratch_oml_sop_compact import paired_bootstrap
from _scratch_same_teacher_ladder import _score
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

ANCHORS = 4096
CANDIDATE_POOL = 8192
HARD_NEGATIVES = 8
UPDATES = 200
LEARNING_RATE = 0.01
SCALE_QUANTILE = 0.999
MINIMUM_MAP_GAIN = 0.002
MAXIMUM_R1_LOSS = 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def initial_scales(rows: torch.Tensor) -> torch.Tensor:
    scales = np.quantile(rows.abs().numpy(), SCALE_QUANTILE, axis=0, method="linear")
    scales = scales.astype(np.float32) / 7.0
    if not np.isfinite(scales).all() or bool((scales <= 0).any()):
        raise ValueError("rank-scale initialization differs")
    return torch.from_numpy(scales)


def hard_quantize(rows: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    return F.normalize(
        torch.clamp(torch.round(rows / scales), -7, 7) * scales, dim=1
    ).contiguous()


def ste_quantize(rows: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    normalized = rows / scales
    hard = torch.clamp(torch.round(normalized), -7, 7)
    codes = normalized + (hard - normalized).detach()
    return F.normalize(codes * scales, dim=1)


def calibration_pairs(
    rows: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20_260_920)
    anchors = torch.randperm(len(rows), generator=generator)[:ANCHORS]
    pool = torch.randperm(len(rows), generator=generator)[:CANDIDATE_POOL]
    groups: dict[int, torch.Tensor] = {}
    for label in torch.unique(labels).tolist():
        groups[int(label)] = torch.nonzero(labels == label, as_tuple=False).flatten()
    positives = []
    for anchor in anchors.tolist():
        candidates = groups[int(labels[anchor])]
        candidates = candidates[candidates != anchor]
        if len(candidates) == 0:
            raise ValueError("rank-scale positive authority differs")
        positives.append(int(candidates[0]))
    positives_tensor = torch.tensor(positives, dtype=torch.int64)
    device = torch.device("cuda")
    bank = rows[pool].to(device)
    bank_labels = labels[pool].to(device)
    negative_parts = []
    with torch.inference_mode():
        for start in range(0, len(anchors), 128):
            selected = anchors[start : start + 128]
            scores = rows[selected].to(device) @ bank.T
            scores = scores.masked_fill(
                labels[selected].to(device).unsqueeze(1) == bank_labels.unsqueeze(0),
                -torch.inf,
            )
            negative_parts.append(
                torch.topk(scores, HARD_NEGATIVES, dim=1, largest=True, sorted=True).indices.cpu()
            )
    negatives = pool[torch.cat(negative_parts)]
    return anchors, positives_tensor, negatives, pool


def fit_scales(
    rows: torch.Tensor,
    labels: torch.Tensor,
    *,
    objective: str,
) -> tuple[torch.Tensor, list[float]]:
    device = torch.device("cuda")
    base = initial_scales(rows).to(device)
    log_scales = torch.nn.Parameter(base.log())
    optimizer = torch.optim.Adam([log_scales], lr=LEARNING_RATE)
    anchors, positives, negatives, pool = calibration_pairs(rows, labels)
    anchor_rows = rows[anchors].to(device)
    positive_rows = rows[positives].to(device)
    negative_rows = rows[negatives].to(device)
    reconstruction_rows = rows[pool].to(device)
    reference_margins = torch.einsum(
        "bd,bnd->bn", anchor_rows, positive_rows[:, None, :] - negative_rows
    ).detach()
    weights = torch.exp(-reference_margins.abs() / 0.05).detach()
    weights = weights / weights.mean()
    losses = []
    for _ in range(UPDATES):
        scales = log_scales.exp()
        if objective == "margin":
            quantized_positive = ste_quantize(positive_rows, scales)
            quantized_negative = ste_quantize(
                negative_rows.reshape(-1, rows.shape[1]), scales
            ).reshape_as(negative_rows)
            margins = torch.einsum(
                "bd,bnd->bn",
                anchor_rows,
                quantized_positive[:, None, :] - quantized_negative,
            )
            loss = (weights * (margins - reference_margins).square()).mean()
        elif objective == "reconstruction":
            quantized = ste_quantize(reconstruction_rows, scales)
            loss = (quantized - reconstruction_rows).square().mean()
        else:
            raise ValueError("rank-scale objective differs")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach()))
    return log_scales.detach().exp().cpu(), losses


def arm(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    labels: tuple[int, ...],
    scales: torch.Tensor,
) -> dict[str, object]:
    score = _score(
        queries,
        hard_quantize(gallery, scales),
        labels,
        labels,
        same_rows=True,
        device=torch.device("cuda"),
    )
    return score


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-rank-scale-gate", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    for path, expected in (
        (args.features, args.features_sha256),
        (args.checkpoint, args.checkpoint_sha256),
        (args.preregistration, args.preregistration_sha256),
        (Path(__file__), args.script_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError("rank-scale authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if preregistration.get("script_sha256") != args.script_sha256:
        raise ValueError("rank-scale preregistration differs")
    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    with np.load(args.features, allow_pickle=False) as archive:
        train = F.normalize(
            torch.from_numpy(np.ascontiguousarray(archive["train_features"])).float(), dim=1
        ).contiguous()
        train_labels = torch.from_numpy(
            np.ascontiguousarray(archive["train_labels"], dtype=np.int64)
        )
        test = F.normalize(
            torch.from_numpy(np.ascontiguousarray(archive["test_features"])).float(), dim=1
        ).contiguous()
        test_labels = tuple(
            int(value)
            for value in np.ascontiguousarray(archive["test_labels"], dtype=np.int64).tolist()
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weight, bias = checkpoint["weight"], checkpoint["bias"]
    fit_float = F.normalize(F.linear(train, weight, bias), dim=1).contiguous()
    test_float = F.normalize(F.linear(test, weight, bias), dim=1).contiguous()
    quantile_scales = initial_scales(fit_float)
    reconstruction_scales, reconstruction_losses = fit_scales(
        fit_float, train_labels, objective="reconstruction"
    )
    margin_scales, margin_losses = fit_scales(fit_float, train_labels, objective="margin")
    baseline = arm(test_float, test_float, test_labels, quantile_scales)
    reconstruction = arm(test_float, test_float, test_labels, reconstruction_scales)
    margin = arm(test_float, test_float, test_labels, margin_scales)
    map_gain = paired_bootstrap(margin["per_query_ap"], baseline["per_query_ap"])
    r1_gain = paired_bootstrap(margin["per_query_r1"], baseline["per_query_r1"])
    margin_vs_reconstruction = paired_bootstrap(
        margin["per_query_ap"], reconstruction["per_query_ap"]
    )
    passed = bool(
        float(map_gain["delta"]) >= MINIMUM_MAP_GAIN
        and float(map_gain["ci95"][0]) > 0.0
        and float(r1_gain["delta"]) >= MAXIMUM_R1_LOSS
        and float(margin_vs_reconstruction["delta"]) > 0.0
    )
    result = {
        "schema": "scratch-oml-sop-rank-scale-int4-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "persistent_gallery_bytes_per_item": 128,
        "arms": {
            name: {"map_at_r": value["map_at_r"], "recall_at_1": value["recall_at_1"]}
            for name, value in (
                ("quantile", baseline),
                ("reconstruction", reconstruction),
                ("margin", margin),
            )
        },
        "contrasts": {
            "margin_minus_quantile_map_at_r": map_gain,
            "margin_minus_quantile_recall_at_1": r1_gain,
            "margin_minus_reconstruction_map_at_r": margin_vs_reconstruction,
        },
        "training": {
            "anchors": ANCHORS,
            "candidate_pool": CANDIDATE_POOL,
            "hard_negatives": HARD_NEGATIVES,
            "updates": UPDATES,
            "learning_rate": LEARNING_RATE,
            "reconstruction_final_loss": reconstruction_losses[-1],
            "margin_final_loss": margin_losses[-1],
        },
        "gate": {"passed": passed},
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
