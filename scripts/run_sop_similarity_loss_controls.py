#!/usr/bin/env python3
"""Run matched established-loss controls for the SOP compact adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import struct
import time
from pathlib import Path
from typing import cast

import numpy as np
import run_sop_positive_coverage_metric as coverage
import torch
from positive_coverage_artifacts import (
    MatchedLossPanelArtifactPaths,
    mean_recent_loss,
    positive_coverage_source_identity,
    write_matched_loss_panel_artifacts,
)
from probe_representation_ceiling import load_paired_train_archives
from torch import nn

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.representation_ceiling import deterministic_class_partition
from sfora.teacher_anchored_distillation import (
    multi_similarity_hard_negative_loss,
    positive_coverage_hard_negative_loss,
    stable_different_class_topk,
    supervised_contrastive_hard_negative_loss,
)

SEED = 0
TEMPERATURE = coverage.TEMPERATURE
MARGIN = coverage.MARGIN
ANCHOR_WEIGHT = coverage.ANCHOR_WEIGHT
MULTI_SIMILARITY_BASE = 0.5
_ARM_NAMES = ("pooled", "coverage", "mean_logit", "supcon", "multi_similarity")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_base_authority(
    *, parameter_sha256: str, expected_map: float, expected_r1: float
) -> tuple[str, float, float]:
    """Validate caller-registered base identity and validation scores."""

    if (
        type(parameter_sha256) is not str
        or _SHA256.fullmatch(parameter_sha256) is None
        or type(expected_map) is not float
        or type(expected_r1) is not float
        or not math.isfinite(expected_map)
        or not math.isfinite(expected_r1)
        or not 0.0 <= expected_map <= 1.0
        or not 0.0 <= expected_r1 <= 1.0
    ):
        raise ValueError("base authority differs")
    return parameter_sha256, expected_map, expected_r1


def integer_tensor_sha256(value: torch.Tensor) -> str:
    """Hash one integer tensor with explicit dtype, rank, shape, and byte order."""

    if (
        type(value) is not torch.Tensor
        or value.dtype != torch.int64
        or value.ndim not in (1, 2)
        or value.numel() == 0
    ):
        raise ValueError("integer tensor authority differs")
    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256(b"sfora-int64-le-v1\0")
    digest.update(struct.pack("<I", cpu.ndim))
    digest.update(struct.pack(f"<{cpu.ndim}Q", *cpu.shape))
    digest.update(cpu.numpy().astype("<i8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def load_base_receipt_authority(
    *,
    path: Path,
    sha256: str,
    parameter_sha256: str,
    expected_map: float,
    expected_r1: float,
) -> dict[str, str]:
    """Authenticate the parent base receipt used by the SOP control panel."""

    if (
        not isinstance(path, Path)
        or not path.is_absolute()
        or not path.is_file()
        or _SHA256.fullmatch(sha256) is None
        or coverage._file_sha256(path) != sha256
    ):
        raise ValueError("base receipt authority differs")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("base receipt authority differs") from error
    if (
        type(value) is not dict
        or value.get("schema") != "sfora-retrieval-local-rank-replay-v1"
        or value.get("claim_eligible") is not False
        or value.get("official_test_touched") is not False
        or value.get("dataset") != "sop-official-train-class-disjoint-validation"
        or value.get("final_head_sha256") != parameter_sha256
        or type(value.get("expected_packed_map_at_r")) is not float
        or type(value.get("expected_packed_r1")) is not float
        or abs(value["expected_packed_map_at_r"] - expected_map) > 1e-12
        or abs(value["expected_packed_r1"] - expected_r1) > 1e-12
        or type(value.get("source_revision")) is not str
        or re.fullmatch(r"[0-9a-f]{40}", value["source_revision"]) is None
    ):
        raise ValueError("base receipt authority differs")
    return {"sha256": sha256, "source_revision": value["source_revision"]}


def frozen_hard_negative_index(
    base_codes: torch.Tensor,
    labels: tuple[int, ...],
    *,
    device: torch.device,
    k: int,
    block_size: int,
) -> torch.Tensor:
    """Mine one deterministic frozen negative table for every fitting row."""

    if (
        base_codes.ndim != 2
        or len(labels) != base_codes.shape[0]
        or type(k) is not int
        or not 0 < k < base_codes.shape[0]
        or type(block_size) is not int
        or block_size <= 0
    ):
        raise ValueError("frozen negative authority differs")
    bank = base_codes.to(device)
    label_tensor = torch.tensor(labels, dtype=torch.int64, device=device)
    blocks = []
    with torch.inference_mode():
        for start in range(0, bank.shape[0], block_size):
            stop = min(start + block_size, bank.shape[0])
            blocks.append(
                stable_different_class_topk(
                    bank[start:stop],
                    bank,
                    label_tensor[start:stop],
                    label_tensor,
                    k=k,
                )
            )
    return torch.cat(blocks).contiguous()


def canonical_arm_result(
    *, losses: list[float], weight: torch.Tensor, score: dict[str, object]
) -> dict[str, object]:
    """Build deterministic scientific arm evidence without wall-clock fields."""

    if not losses or any(not math.isfinite(value) for value in losses):
        raise ValueError("arm result authority differs")
    return {
        "final_loss": losses[-1],
        "mean_last_100_loss": mean_recent_loss(losses, window=100),
        "parameter_sha256": coverage._parameter_sha256(weight),
        "score": score,
    }


def matched_control_loss(
    name: str,
    positive_similarities: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_similarities: torch.Tensor,
    self_similarities: torch.Tensor,
) -> torch.Tensor:
    """Evaluate one preregistered established-loss control."""

    if name in {"pooled", "coverage", "mean_logit"}:
        return positive_coverage_hard_negative_loss(
            positive_similarities,
            positive_mask,
            negative_similarities,
            self_similarities,
            temperature=TEMPERATURE,
            margin=MARGIN,
            anchor_weight=ANCHOR_WEIGHT,
            positive_aggregation=name,
        )
    if name == "supcon":
        return supervised_contrastive_hard_negative_loss(
            positive_similarities,
            positive_mask,
            negative_similarities,
            self_similarities,
            temperature=TEMPERATURE,
            margin=MARGIN,
            anchor_weight=ANCHOR_WEIGHT,
        )
    if name == "multi_similarity":
        return multi_similarity_hard_negative_loss(
            positive_similarities,
            positive_mask,
            negative_similarities,
            self_similarities,
            alpha=2.0,
            beta=50.0,
            base=MULTI_SIMILARITY_BASE,
            mining_margin=0.1,
            anchor_weight=ANCHOR_WEIGHT,
        )
    raise ValueError("matched control objective differs")


def _train_arm(
    name: str,
    base_fit: torch.Tensor,
    base_validation: torch.Tensor,
    fit_labels: tuple[int, ...],
    validation_labels: tuple[int, ...],
    schedule: tuple[np.ndarray, ...],
    frozen_negatives: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    transform = nn.Linear(base_fit.shape[1], base_fit.shape[1], bias=False, device=device)
    with torch.no_grad():
        transform.weight.copy_(torch.eye(base_fit.shape[1], device=device))
    optimizer = torch.optim.Adam(transform.parameters(), lr=coverage.LEARNING_RATE)
    fit_cuda = base_fit.to(device)
    fit_label_array = np.asarray(fit_labels, dtype=np.int64)
    positive_groups = {label: np.flatnonzero(fit_label_array == label) for label in set(fit_labels)}
    started = time.monotonic()
    losses: list[float] = []
    for step, anchors_np in enumerate(schedule, start=1):
        anchors = torch.from_numpy(anchors_np).to(device)
        mined = frozen_negatives[anchors]
        positives_np, positive_mask_np = coverage._positive_rows(
            anchors_np, fit_label_array, positive_groups
        )
        positives = torch.from_numpy(positives_np).to(device)
        positive_mask = torch.from_numpy(positive_mask_np).to(device).contiguous()
        anchor_codes = coverage._unit(transform(fit_cuda[anchors]))
        positive_codes = coverage._unit(
            transform(fit_cuda[positives.reshape(-1)]).reshape(len(anchors), positives.shape[1], -1)
        )
        negative_codes = coverage._unit(
            transform(fit_cuda[mined.reshape(-1)]).reshape(
                len(anchors), coverage.HARD_NEGATIVES, -1
            )
        )
        positive_similarities = torch.einsum(
            "bd,bpd->bp", anchor_codes, positive_codes
        ).contiguous()
        negative_similarities = torch.einsum(
            "bd,bnd->bn", anchor_codes, negative_codes
        ).contiguous()
        self_similarities = torch.einsum("bd,bd->b", anchor_codes, fit_cuda[anchors]).contiguous()
        loss = matched_control_loss(
            name,
            positive_similarities,
            positive_mask,
            negative_similarities,
            self_similarities,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
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
        validation_codes = coverage._unit(transform(base_validation.to(device)).cpu())
    result = canonical_arm_result(
        losses=losses,
        weight=transform.weight,
        score=coverage._score(validation_codes, validation_labels, device),
    )
    return result, {"weight": transform.weight.detach().cpu()}


def main() -> None:
    global SEED
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint-sha256", required=True)
    parser.add_argument("--base-receipt", type=Path, required=True)
    parser.add_argument("--base-receipt-sha256", required=True)
    parser.add_argument("--base-parameter-sha256", required=True)
    parser.add_argument("--expected-base-map", type=float, required=True)
    parser.add_argument("--expected-base-r1", type=float, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--pooled-checkpoint", type=Path, required=True)
    parser.add_argument("--coverage-checkpoint", type=Path, required=True)
    parser.add_argument("--mean-logit-checkpoint", type=Path, required=True)
    parser.add_argument("--supcon-checkpoint", type=Path, required=True)
    parser.add_argument("--multi-similarity-checkpoint", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--execute-matched-controls", action="store_true", required=True)
    args = parser.parse_args()
    SEED = args.seed
    coverage.SEED = args.seed
    paths = MatchedLossPanelArtifactPaths(
        pooled_checkpoint=args.pooled_checkpoint,
        coverage_checkpoint=args.coverage_checkpoint,
        mean_logit_checkpoint=args.mean_logit_checkpoint,
        supcon_checkpoint=args.supcon_checkpoint,
        multi_similarity_checkpoint=args.multi_similarity_checkpoint,
        complete_receipt=args.receipt,
    )
    source_identity = positive_coverage_source_identity(
        driver=Path(__file__).resolve(),
        driver_sha256=args.driver_sha256,
        source_revision=args.source_revision,
    )
    base_parameter_sha256, expected_base_map, expected_base_r1 = validate_base_authority(
        parameter_sha256=args.base_parameter_sha256,
        expected_map=args.expected_base_map,
        expected_r1=args.expected_base_r1,
    )

    runtime = configure_deterministic_similarity_runtime(SEED, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if coverage._file_sha256(args.base_checkpoint) != args.base_checkpoint_sha256:
        raise ValueError("base checkpoint digest differs")
    base_receipt = load_base_receipt_authority(
        path=args.base_receipt,
        sha256=args.base_receipt_sha256,
        parameter_sha256=base_parameter_sha256,
        expected_map=expected_base_map,
        expected_r1=expected_base_r1,
    )
    device = torch.device("cuda")
    pair = load_paired_train_archives(
        args.source_snapshot,
        args.source_sha256,
        args.teacher_snapshot,
        args.teacher_sha256,
    )
    teacher = coverage._unit(pair["teacher_train"])
    partition = deterministic_class_partition(
        pair["train_labels"], fit_fraction=0.8, seed=coverage.SPLIT_SEED
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
    if coverage._parameter_sha256(weight, bias) != base_parameter_sha256:
        raise ValueError("base checkpoint parameters differ")
    with torch.inference_mode():
        base_fit = coverage._unit(torch.nn.functional.linear(teacher[fit_indexes], weight, bias))
        base_validation = coverage._unit(
            torch.nn.functional.linear(teacher[validation_indexes], weight, bias)
        )
    base_score = coverage._score(base_validation, validation_labels, device)
    observed_base_map = base_score.get("packed_map_at_r")
    observed_base_r1 = base_score.get("packed_r1")
    if (
        type(observed_base_map) is not float
        or type(observed_base_r1) is not float
        or abs(observed_base_map - expected_base_map) > 1e-12
        or abs(observed_base_r1 - expected_base_r1) > 1e-12
    ):
        raise ValueError("base score differs")
    schedule_authority = coverage._schedule(fit_labels)
    schedule = tuple(row.copy() for row in schedule_authority.row_indexes)
    frozen_negatives = frozen_hard_negative_index(
        base_fit,
        fit_labels,
        device=device,
        k=coverage.HARD_NEGATIVES,
        block_size=128,
    )
    frozen_negative_sha256 = integer_tensor_sha256(frozen_negatives)
    arms: dict[str, dict[str, object]] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name in _ARM_NAMES:
        arms[name], states[name] = _train_arm(
            name,
            base_fit,
            base_validation,
            fit_labels,
            validation_labels,
            schedule,
            frozen_negatives,
            device=device,
        )
    receipt = {
        "arms": arms,
        "base": {
            "checkpoint_sha256": args.base_checkpoint_sha256,
            "parameter_sha256": base_parameter_sha256,
            "parent_receipt": base_receipt,
            "score": base_score,
        },
        "claim_eligible": False,
        "dataset": "sop-official-train-class-disjoint-validation",
        "fitting_rows": len(fit_indexes),
        "method": {
            "anchor_weight": ANCHOR_WEIGHT,
            "classes_per_update": coverage.CLASS_COUNT,
            "hard_negatives": coverage.HARD_NEGATIVES,
            "learning_rate": coverage.LEARNING_RATE,
            "margin": MARGIN,
            "multi_similarity": {
                "alpha": 2.0,
                "base": MULTI_SIMILARITY_BASE,
                "beta": 50.0,
                "epsilon": 0.1,
                "setting": "authors-released-implementation",
            },
            "negative_mining": "frozen-base-cached-once",
            "negative_table": {
                "columns": frozen_negatives.shape[1],
                "rows": frozen_negatives.shape[0],
                "sha256": frozen_negative_sha256,
            },
            "optimizer": "adam-constant-no-weight-decay-clip1",
            "schedule_sha256": schedule_authority.sha256,
            "temperature": TEMPERATURE,
            "updates_per_arm": coverage.UPDATES,
        },
        "inputs": {
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "official_test_touched": False,
        "partition": {
            "fit_labels_sha256": integer_tensor_sha256(torch.tensor(fit_labels, dtype=torch.int64)),
            "fit_rows_sha256": integer_tensor_sha256(torch.tensor(fit_indexes, dtype=torch.int64)),
            "validation_labels_sha256": integer_tensor_sha256(
                torch.tensor(validation_labels, dtype=torch.int64)
            ),
            "validation_rows_sha256": integer_tensor_sha256(
                torch.tensor(validation_indexes, dtype=torch.int64)
            ),
        },
        "runtime": runtime._asdict(),
        "schema": "sfora-matched-loss-controls-v1",
        "seed": SEED,
        "source": source_identity,
        "validation_classes": len(partition.validation_class_ids),
        "validation_rows": len(validation_indexes),
    }
    write_matched_loss_panel_artifacts(states=states, receipt=receipt, paths=paths)


if __name__ == "__main__":
    main()
