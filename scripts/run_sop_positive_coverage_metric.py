#!/usr/bin/env python3
"""Run the authenticated matched-arm SOP positive-coverage experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from positive_coverage_artifacts import (
    PositiveCoverageArtifactPaths,
    mean_recent_loss,
    positive_coverage_source_identity,
    write_positive_coverage_artifacts,
)
from probe_representation_ceiling import load_paired_train_archives
from probe_sop_relational_linear import score_symmetric
from torch import nn

from sfora.deterministic_similarity_runtime import (
    configure_deterministic_similarity_runtime,
)
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.teacher_anchored_distillation import (
    ClassBalancedAnchorSchedule,
    class_balanced_anchor_schedule,
    positive_coverage_hard_negative_loss,
    stable_different_class_topk,
)

SEED = 0
SPLIT_SEED = 17
CLASS_COUNT = 64
ANCHORS_PER_CLASS = 2
HARD_NEGATIVES = 256
UPDATES = 2_000
TEMPERATURE = 0.05
MARGIN = 0.02
ANCHOR_WEIGHT = 10.0
LEARNING_RATE = 1e-4
BOOTSTRAP_SAMPLES = 10_000
BASE_MAP = 0.5556011035874895
BASE_R1 = 0.8111758251034017
MAP_GATE = BASE_MAP + 0.003
R1_GATE = BASE_R1
EXPECTED_BASE_PARAMETER_SHA256 = "acd49a3854a238a84d760c5499ab5f07e53a48c9ef995253a636ad3276f8b6ee"


def _unit(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(value.float(), dim=-1).contiguous()


def _parameter_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        cpu = value.detach().cpu().float().contiguous()
        digest.update(struct.pack("<I", cpu.ndim))
        digest.update(struct.pack(f"<{cpu.ndim}Q", *cpu.shape))
        digest.update(cpu.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict[str, object]) -> bytes:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return (text + "\n").encode()


def _schedule(labels: tuple[int, ...]) -> ClassBalancedAnchorSchedule:
    label_array = np.asarray(labels, dtype=np.int64)
    return class_balanced_anchor_schedule(
        label_array,
        seed=SEED,
        updates=UPDATES,
        classes_per_update=CLASS_COUNT,
        rows_per_class=ANCHORS_PER_CLASS,
    )


def _stable_hard_negatives(
    anchor_codes: torch.Tensor,
    bank: torch.Tensor,
    anchor_labels: torch.Tensor,
    bank_labels: torch.Tensor,
) -> torch.Tensor:
    return stable_different_class_topk(
        anchor_codes,
        bank,
        anchor_labels,
        bank_labels,
        k=HARD_NEGATIVES,
    )


def _positive_rows(
    anchors: np.ndarray,
    label_array: np.ndarray,
    groups: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows = [
        groups[int(label_array[anchor])][groups[int(label_array[anchor])] != anchor]
        for anchor in anchors
    ]
    width = max(len(value) for value in rows)
    padded = np.zeros((len(rows), width), dtype=np.int64)
    mask = np.zeros((len(rows), width), dtype=np.bool_)
    for index, value in enumerate(rows):
        padded[index, : len(value)] = value
        mask[index, : len(value)] = True
    return padded, mask


def _score(codes: torch.Tensor, labels: tuple[int, ...], device: torch.device) -> dict[str, object]:
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    width = max(counts.values()) - 1
    floating = score_symmetric(codes, labels, candidate_width=width, device=device)
    packed = score_symmetric(
        pack_int8_unit_embeddings(codes), labels, candidate_width=width, device=device
    )
    return {
        "float_map_at_r": float(floating["map_at_r"]),
        "float_r1": float(floating["r1"]),
        "packed_map_at_r": float(packed["map_at_r"]),
        "packed_per_query_ap": [float(value) for value in packed["per_query_ap"]],
        "packed_r1": float(packed["r1"]),
    }


def _class_cluster_lower_bound(
    treatment: list[float], control: list[float], labels: tuple[int, ...]
) -> float:
    difference = np.asarray(treatment, dtype=np.float64) - np.asarray(control, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    classes = np.asarray(sorted(set(labels)), dtype=np.int64)
    sums = np.asarray([difference[label_array == label].sum() for label in classes])
    counts = np.asarray([(label_array == label).sum() for label in classes])
    rng = np.random.Generator(np.random.PCG64(SEED))
    values = np.empty(BOOTSTRAP_SAMPLES, dtype=np.float64)
    for start in range(0, BOOTSTRAP_SAMPLES, 128):
        stop = min(start + 128, BOOTSTRAP_SAMPLES)
        draws = rng.integers(0, len(classes), size=(stop - start, len(classes)))
        values[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(np.quantile(values, 0.025, method="lower"))


def _train_arm(
    name: str,
    base_fit: torch.Tensor,
    base_validation: torch.Tensor,
    fit_labels: tuple[int, ...],
    validation_labels: tuple[int, ...],
    schedule: tuple[np.ndarray, ...],
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    transform = nn.Linear(base_fit.shape[1], base_fit.shape[1], bias=False, device=device)
    with torch.no_grad():
        transform.weight.copy_(torch.eye(base_fit.shape[1], device=device))
    optimizer = torch.optim.Adam(transform.parameters(), lr=LEARNING_RATE)
    fit_cuda = base_fit.to(device)
    fit_label_tensor = torch.tensor(fit_labels, dtype=torch.int64, device=device)
    fit_label_array = np.asarray(fit_labels, dtype=np.int64)
    positive_groups = {label: np.flatnonzero(fit_label_array == label) for label in set(fit_labels)}
    started = time.monotonic()
    losses: list[float] = []
    for step, anchors_np in enumerate(schedule, start=1):
        anchors = torch.from_numpy(anchors_np).to(device)
        with torch.inference_mode():
            bank = _unit(transform(fit_cuda))
            mined = _stable_hard_negatives(
                bank[anchors], bank, fit_label_tensor[anchors], fit_label_tensor
            )
        positives_np, positive_mask_np = _positive_rows(
            anchors_np, fit_label_array, positive_groups
        )
        positives = torch.from_numpy(positives_np).to(device)
        positive_mask = torch.from_numpy(positive_mask_np).to(device).contiguous()
        anchor_codes = _unit(transform(fit_cuda[anchors]))
        positive_codes = _unit(
            transform(fit_cuda[positives.reshape(-1)]).reshape(len(anchors), positives.shape[1], -1)
        )
        negative_codes = _unit(
            transform(fit_cuda[mined.reshape(-1)]).reshape(len(anchors), HARD_NEGATIVES, -1)
        )
        positive_similarities = torch.einsum(
            "bd,bpd->bp", anchor_codes, positive_codes
        ).contiguous()
        negative_similarities = torch.einsum(
            "bd,bnd->bn", anchor_codes, negative_codes
        ).contiguous()
        self_similarities = torch.einsum("bd,bd->b", anchor_codes, fit_cuda[anchors]).contiguous()
        loss = positive_coverage_hard_negative_loss(
            positive_similarities,
            positive_mask,
            negative_similarities,
            self_similarities,
            temperature=TEMPERATURE,
            margin=MARGIN,
            anchor_weight=ANCHOR_WEIGHT,
            positive_aggregation=name,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transform.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % 100 == 0:
            print(
                json.dumps(
                    {
                        "arm": name,
                        "elapsed_seconds": time.monotonic() - started,
                        "loss": losses[-1],
                        "step": step,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    with torch.inference_mode():
        validation_codes = _unit(transform(base_validation.to(device)).cpu())
    result = {
        "elapsed_seconds": time.monotonic() - started,
        "final_loss": losses[-1],
        "mean_last_100_loss": mean_recent_loss(losses, window=100),
        "parameter_sha256": _parameter_sha256(transform.weight),
        "score": _score(validation_codes, validation_labels, device),
    }
    state = {"weight": transform.weight.detach().cpu()}
    return result, state


def main() -> None:
    global SEED
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pooled-checkpoint", type=Path, required=True)
    parser.add_argument("--coverage-checkpoint", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--execute-positive-coverage", action="store_true", required=True)
    args = parser.parse_args()
    SEED = args.seed
    artifact_paths = PositiveCoverageArtifactPaths(
        pooled_checkpoint=args.pooled_checkpoint,
        coverage_checkpoint=args.coverage_checkpoint,
        complete_receipt=args.receipt,
    )
    source_identity = positive_coverage_source_identity(
        driver=Path(__file__).resolve(),
        driver_sha256=args.driver_sha256,
        source_revision=args.source_revision,
    )

    configure_deterministic_similarity_runtime(SEED, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if _file_sha256(args.base_checkpoint) != args.base_checkpoint_sha256:
        raise ValueError("base checkpoint digest differs")
    device = torch.device("cuda")
    pair = load_paired_train_archives(
        args.source_snapshot,
        args.source_sha256,
        args.teacher_snapshot,
        args.teacher_sha256,
    )
    teacher = _unit(pair["teacher_train"])
    partition = deterministic_class_partition(
        pair["train_labels"], fit_fraction=0.8, seed=SPLIT_SEED
    )
    fit_indexes = list(partition.fit_row_indexes)
    validation_indexes = list(partition.validation_row_indexes)
    fit_labels = tuple(pair["train_labels"][row] for row in fit_indexes)
    validation_labels = tuple(pair["train_labels"][row] for row in validation_indexes)
    state = torch.load(args.base_checkpoint, map_location="cpu", weights_only=True)
    if set(state) != {"weight", "bias"}:
        raise ValueError("base checkpoint schema differs")
    weight = cast(torch.Tensor, state["weight"]).float().contiguous()
    bias = cast(torch.Tensor, state["bias"]).float().contiguous()
    if _parameter_sha256(weight, bias) != EXPECTED_BASE_PARAMETER_SHA256:
        raise ValueError("base checkpoint parameters differ")
    with torch.inference_mode():
        base_fit = _unit(torch.nn.functional.linear(teacher[fit_indexes], weight, bias))
        base_validation = _unit(
            torch.nn.functional.linear(teacher[validation_indexes], weight, bias)
        )
    base_score = _score(base_validation, validation_labels, device)
    if (
        abs(float(base_score["packed_map_at_r"]) - BASE_MAP) > 1e-12
        or abs(float(base_score["packed_r1"]) - BASE_R1) > 1e-12
    ):
        raise ValueError("base score differs")
    schedule_authority = _schedule(fit_labels)
    schedule = tuple(row.copy() for row in schedule_authority.row_indexes)
    arms: dict[str, dict[str, object]] = {}
    arm_states: dict[str, dict[str, torch.Tensor]] = {}
    for name in ("pooled", "coverage"):
        result, arm_state = _train_arm(
            name,
            base_fit,
            base_validation,
            fit_labels,
            validation_labels,
            schedule,
            device=device,
        )
        arms[name] = result
        arm_states[name] = arm_state
    pooled_score = cast(dict[str, object], arms["pooled"]["score"])
    treatment_score = cast(dict[str, object], arms["coverage"]["score"])
    lower_bound = _class_cluster_lower_bound(
        cast(list[float], treatment_score["packed_per_query_ap"]),
        cast(list[float], pooled_score["packed_per_query_ap"]),
        validation_labels,
    )
    passes = (
        float(treatment_score["packed_map_at_r"]) >= MAP_GATE
        and float(treatment_score["packed_r1"]) >= R1_GATE
        and lower_bound > 0.0
    )
    receipt = {
        "arms": arms,
        "base": {
            "checkpoint_sha256": args.base_checkpoint_sha256,
            "parameter_sha256": EXPECTED_BASE_PARAMETER_SHA256,
            "score": base_score,
        },
        "bootstrap": {
            "class_clustered": True,
            "lower_quantile": 0.025,
            "samples": BOOTSTRAP_SAMPLES,
            "seed": SEED,
            "treatment_minus_control_lower_bound": lower_bound,
        },
        "claim_eligible": False,
        "dataset": "sop-official-train-class-disjoint-validation",
        "fitting_rows": len(fit_indexes),
        "gates": {
            "packed_map_at_r": MAP_GATE,
            "packed_r1": R1_GATE,
            "treatment_minus_control_lower_bound": 0.0,
        },
        "method": {
            "anchor_weight": ANCHOR_WEIGHT,
            "classes_per_update": CLASS_COUNT,
            "hard_negatives": HARD_NEGATIVES,
            "learning_rate": LEARNING_RATE,
            "margin": MARGIN,
            "optimizer": "adam-constant-no-weight-decay-clip1",
            "temperature": TEMPERATURE,
            "updates_per_arm": UPDATES,
            "schedule_sha256": schedule_authority.sha256,
        },
        "inputs": {
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "official_test_touched": False,
        "passes": passes,
        "schema": "sfora-positive-coverage-metric-screen-v2",
        "seed": SEED,
        "source": source_identity,
        "validation_classes": len(partition.validation_class_ids),
        "validation_rows": len(validation_indexes),
    }
    write_positive_coverage_artifacts(
        states=arm_states,
        receipt=receipt,
        paths=artifact_paths,
    )


if __name__ == "__main__":
    main()
