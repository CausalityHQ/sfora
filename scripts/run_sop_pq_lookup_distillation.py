#!/usr/bin/env python3
"""Run one frozen label-free query-dependent PQ lookup-table diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import time
from collections import Counter
from pathlib import Path
from typing import cast

import torch
from positive_coverage_artifacts import positive_coverage_source_identity
from probe_representation_ceiling import load_paired_train_archives
from probe_sop_relational_linear import score_symmetric
from run_sop_additive_codec_preflight import (
    score_asymmetric_optimized_product,
    score_asymmetric_product,
)
from run_sop_pq_candidate_reranking import (
    _canonical_json,
    _file_sha256,
    _load_parent_quantizers,
    score_ranked_candidates,
    validate_parent_receipt,
)

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.pq_candidate_reranking import (
    aligned_product_squared_distances,
    exact_product_shortlists,
)
from sfora.pq_lookup_distillation import (
    LOOKUP_MAXIMUM_GRADIENT_NORM,
    LOOKUP_WEIGHT_DECAY,
    LookupTableDistillationSpec,
    QueryDependentLookupScorer,
    fit_query_dependent_lookup_scorer,
    lookup_table_distillation_loss,
)
from sfora.product_quantization import fit_product_quantizer
from sfora.representation_ceiling import deterministic_class_partition

DIMENSIONS = 128
BLOCKS = 24
CODEBOOK_SIZE = 256
TEACHER_NEAR_WIDTH = 128
COMPRESSED_NEAR_WIDTH = 128
COMPRESSED_MINING_WIDTH = 256
UNIFORM_TAIL_WIDTH = 128
UNIFORM_TAIL_SEED = 2901
PAIRS_PER_STRATUM = 128
HIDDEN_DIMENSIONS = 128
CORRECTION_DIMENSIONS = 32
EPOCHS = 4
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
TEMPERATURE = 0.03
PAIRWISE_WEIGHT = 0.25
RANK_SCORE_WEIGHT = 0.1
MAXIMUM_ITERATIONS = 25
SPLIT_SEED = 17
TARGET_MAP = 0.58563
MAXIMUM_R1_REGRESSION = 0.001
PRIOR_RERANKER_MAP = 0.5786155156952365
PQ_SEEDS = (0, 1, 2, 3, 4)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_U64_MASK = (1 << 64) - 1
_PRIOR_RECEIPT_KEYS = {
    "claim_eligible",
    "dataset",
    "decision",
    "fit_seconds",
    "fit_teacher_loss",
    "inputs",
    "matched_controls",
    "model_checkpoint",
    "model_parameter_bytes",
    "official_test_touched",
    "partition",
    "pq24_seed_stability",
    "recipe",
    "rankings",
    "reranked",
    "reranked_paired_delta",
    "reranker_seed_stability",
    "runtime",
    "schema",
    "seed",
    "source",
}


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _U64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _U64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _U64_MASK
    return value ^ (value >> 31)


def validate_prior_reranker_receipt(
    value: object,
    *,
    direct_checkpoint_sha256: str,
    parent_codec_checkpoint_sha256: str,
    parent_codec_receipt_sha256: str,
    source_snapshot_sha256: str,
    teacher_snapshot_sha256: str,
) -> None:
    """Bind the fixed comparator to its exact prior experiment and inputs."""

    expected_inputs = {
        "direct_checkpoint_sha256": direct_checkpoint_sha256,
        "parent_codec_checkpoint_sha256": parent_codec_checkpoint_sha256,
        "parent_codec_receipt_sha256": parent_codec_receipt_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "teacher_snapshot_sha256": teacher_snapshot_sha256,
    }
    if (
        type(value) is not dict
        or set(value) != _PRIOR_RECEIPT_KEYS
        or value.get("claim_eligible") is not False
        or value.get("dataset") != "sop-official-train-class-disjoint-validation"
        or value.get("official_test_touched") is not False
        or value.get("schema") != "sfora-pq-candidate-set-reranking-v1"
        or type(value.get("seed")) is not int
        or value["seed"] != 0
        or value.get("inputs") != expected_inputs
        or type(value.get("partition")) is not dict
        or value["partition"].get("split_seed") != SPLIT_SEED
        or type(value.get("decision")) is not dict
        or value["decision"].get("classification") != "candidate-set-reranker-failed-pq32"
        or type(value.get("reranked")) is not dict
        or type(value["reranked"].get("map_at_r")) is not float
        or value["reranked"]["map_at_r"] != PRIOR_RERANKER_MAP
        or type(value["reranked"].get("r1")) is not float
    ):
        raise ValueError("prior reranker receipt authority differs")


def _concrete_tensor_device(value: torch.Tensor, *, expected_type: str) -> torch.device:
    """Return the indexed runtime device represented by an allocated tensor."""

    if (
        type(value) is not torch.Tensor
        or type(expected_type) is not str
        or expected_type not in {"cpu", "cuda"}
        or value.device.type != expected_type
    ):
        raise ValueError("lookup execution device differs")
    return value.device


def build_label_free_candidate_pool(
    float_rankings: torch.Tensor,
    compressed_rankings: torch.Tensor,
    *,
    compressed_width: int,
    uniform_width: int,
    uniform_seed: int,
) -> torch.Tensor:
    """Combine teacher-near, student-near, and deterministic uniform-tail rows."""

    if (
        type(float_rankings) is not torch.Tensor
        or float_rankings.dtype != torch.int64
        or float_rankings.ndim != 2
        or float_rankings.shape[0] < 2
        or float_rankings.shape[1] < 1
        or type(compressed_rankings) is not torch.Tensor
        or compressed_rankings.dtype != torch.int64
        or compressed_rankings.ndim != 2
        or compressed_rankings.shape[0] != float_rankings.shape[0]
        or compressed_rankings.shape[1] < compressed_width
        or compressed_rankings.device != float_rankings.device
        or type(compressed_width) is not int
        or compressed_width < 1
        or type(uniform_width) is not int
        or uniform_width < 1
        or type(uniform_seed) is not int
        or uniform_seed < 0
        or bool((float_rankings < 0).any())
        or bool((compressed_rankings < 0).any())
        or bool((float_rankings >= len(float_rankings)).any())
        or bool((compressed_rankings >= len(float_rankings)).any())
    ):
        raise ValueError("lookup candidate pool authority differs")
    rows = len(float_rankings)
    target = float_rankings.shape[1] + compressed_width + uniform_width
    if target > rows - 1:
        raise ValueError("lookup candidate pool authority differs")
    result: list[list[int]] = []
    for row, (teacher_near, compressed_near) in enumerate(
        zip(float_rankings.tolist(), compressed_rankings.tolist(), strict=True)
    ):
        selected: list[int] = []
        seen = {row}
        for candidate in teacher_near:
            if candidate not in seen:
                seen.add(candidate)
                selected.append(candidate)
        if len(selected) != len(teacher_near):
            raise ValueError("lookup candidate pool authority differs")
        compressed_added = 0
        for candidate in compressed_near:
            if candidate not in seen:
                seen.add(candidate)
                selected.append(candidate)
                compressed_added += 1
            if compressed_added == compressed_width:
                break
        if compressed_added != compressed_width:
            raise ValueError("lookup candidate pool authority differs")
        state = (uniform_seed ^ (row * 0xD2B74407B1CE6E93)) & _U64_MASK
        limit = (1 << 64) - ((1 << 64) % rows)
        uniform_added = 0
        while uniform_added < uniform_width:
            random_word = _splitmix64(state)
            state = (state + 0x9E3779B97F4A7C15) & _U64_MASK
            if random_word >= limit:
                continue
            candidate = random_word % rows
            if candidate not in seen:
                seen.add(candidate)
                selected.append(candidate)
                uniform_added += 1
        if len(selected) != target:
            raise RuntimeError("lookup candidate pool is incomplete")
        result.append(selected)
    return torch.tensor(result, dtype=torch.int64, device=float_rankings.device)


def registered_pair_positions(
    *,
    rows: int,
    candidates: int,
    teacher_width: int,
    compressed_width: int,
    uniform_width: int,
    pairs_per_stratum: int,
) -> torch.Tensor:
    """Return fixed contested and near-versus-tail pair positions."""

    if (
        type(rows) is not int
        or rows < 1
        or type(candidates) is not int
        or type(teacher_width) is not int
        or teacher_width < 32
        or type(compressed_width) is not int
        or compressed_width < 32
        or type(uniform_width) is not int
        or type(pairs_per_stratum) is not int
        or pairs_per_stratum < 1
        or pairs_per_stratum != PAIRS_PER_STRATUM
        or uniform_width < pairs_per_stratum
        or candidates != teacher_width + compressed_width + uniform_width
    ):
        raise ValueError("lookup pair authority differs")
    ordinal = torch.arange(pairs_per_stratum, dtype=torch.int64)
    combinations = torch.combinations(torch.arange(32, dtype=torch.int64), r=2)
    spread = torch.div(
        torch.arange(64, dtype=torch.int64) * len(combinations), 64, rounding_mode="floor"
    )
    teacher_inside = combinations[spread]
    student_inside = teacher_inside + teacher_width
    inside = torch.cat((teacher_inside, student_inside), dim=0)
    contrast_left = torch.where(
        (ordinal // 32) % 2 == 0, ordinal % 32, teacher_width + ordinal % 32
    )
    contrast_right = teacher_width + compressed_width + ordinal
    base = torch.stack(
        (
            inside,
            torch.stack((contrast_left, contrast_right), dim=-1),
        )
    ).reshape(2 * pairs_per_stratum, 2)
    return base.unsqueeze(0).expand(rows, -1, -1)


def lookup_distillation_decision(
    *,
    exhaustive_map_at_r: float,
    fixed_candidate_map_at_r: float,
    r1: float,
    pq24_r1: float,
    pq32_map_at_r: float,
    prior_reranker_map_at_r: float,
    pq_codebook_variation: float,
    objective_improved: bool,
) -> dict[str, object]:
    """Classify the frozen fixed-code experiment without tuning on its result."""

    values = (
        exhaustive_map_at_r,
        fixed_candidate_map_at_r,
        r1,
        pq24_r1,
        pq32_map_at_r,
        prior_reranker_map_at_r,
        pq_codebook_variation,
    )
    if type(objective_improved) is not bool or any(
        type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("lookup distillation result differs")
    minimum_fixed_candidate_signal = prior_reranker_map_at_r + pq_codebook_variation
    minimum_r1 = pq24_r1 - MAXIMUM_R1_REGRESSION
    if not objective_improved:
        classification = "fixed-code-lookup-optimization-failed"
    elif r1 < minimum_r1:
        classification = "fixed-code-lookup-r1-regressed"
    elif exhaustive_map_at_r < pq32_map_at_r:
        classification = "fixed-code-lookup-failed-pq32"
    elif fixed_candidate_map_at_r <= minimum_fixed_candidate_signal:
        classification = "fixed-code-lookup-within-pq-variation"
    elif exhaustive_map_at_r < TARGET_MAP:
        classification = "fixed-code-lookup-improved"
    else:
        classification = "fixed-code-lookup-target-screen"
    return {
        "arm": "ranking",
        "classification": classification,
        "gates": {
            "minimum_fixed_pq24_top32_map_at_r_over_prior_and_variation": (
                minimum_fixed_candidate_signal
            ),
            "minimum_r1": minimum_r1,
            "pq32_map_at_r": pq32_map_at_r,
            "target_map_at_r": TARGET_MAP,
        },
        "observed": {
            "exhaustive_map_at_r": exhaustive_map_at_r,
            "fixed_pq24_top32_map_at_r": fixed_candidate_map_at_r,
            "r1": r1,
        },
        "passed": False,
        "reason": "seed-0 screen; lookup-model seed variation is not yet measured",
        "scope": "exhaustive",
    }


def _exact_float_shortlists(
    values: torch.Tensor, *, width: int, query_block_size: int
) -> torch.Tensor:
    if (
        type(values) is not torch.Tensor
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] <= width
        or not bool(torch.isfinite(values).all())
        or type(width) is not int
        or width < 1
        or type(query_block_size) is not int
        or query_block_size < 1
    ):
        raise ValueError("float shortlist authority differs")
    result = []
    with torch.inference_mode():
        for start in range(0, len(values), query_block_size):
            stop = min(start + query_block_size, len(values))
            similarities = values[start:stop] @ values.T
            similarities[
                torch.arange(stop - start, device=values.device),
                torch.arange(start, stop, device=values.device),
            ] = -torch.inf
            retained = width + 1
            scores, indexes = torch.topk(
                similarities, k=retained, dim=1, largest=True, sorted=False
            )
            ordinal_order = torch.argsort(indexes, dim=1, stable=True)
            indexes = indexes.gather(1, ordinal_order)
            scores = scores.gather(1, ordinal_order)
            score_order = torch.argsort(scores, dim=1, descending=True, stable=True)
            indexes = indexes.gather(1, score_order)
            scores = scores.gather(1, score_order)
            ambiguous = scores[:, width - 1] == scores[:, width]
            for block_row in torch.nonzero(ambiguous, as_tuple=False).flatten().tolist():
                boundary = scores[block_row, width - 1]
                candidates = torch.nonzero(
                    similarities[block_row] >= boundary, as_tuple=False
                ).flatten()
                candidates = torch.sort(candidates).values
                candidate_scores = similarities[block_row, candidates]
                order = torch.argsort(candidate_scores, descending=True, stable=True)
                indexes[block_row, :width] = candidates[order[:width]]
            result.append(indexes[:, :width].clone())
    return torch.cat(result, dim=0).contiguous()


def _training_tensors(
    values: torch.Tensor,
    codes: torch.Tensor,
    candidates: torch.Tensor,
    quantizer: object,
    *,
    block_size: int = 256,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    candidate_codes = []
    baselines = []
    teachers = []
    with torch.inference_mode():
        for start in range(0, len(values), block_size):
            stop = min(start + block_size, len(values))
            indexes = candidates[start:stop]
            selected_codes = codes[indexes]
            candidate_codes.append(selected_codes)
            baselines.append(
                -0.5
                * aligned_product_squared_distances(
                    values[start:stop],
                    selected_codes,
                    quantizer,  # type: ignore[arg-type]
                )
            )
            teachers.append(torch.einsum("bd,bcd->bc", values[start:stop], values[indexes]))
    return (
        torch.cat(candidate_codes, dim=0).contiguous(),
        torch.cat(baselines, dim=0).contiguous(),
        torch.cat(teachers, dim=0).contiguous(),
    )


def _loss_diagnostics(
    model: QueryDependentLookupScorer | None,
    queries: torch.Tensor,
    codes: torch.Tensor,
    baseline: torch.Tensor,
    teacher: torch.Tensor,
    pairs: torch.Tensor,
    *,
    listwise_weight: float,
    pairwise_weight: float,
    score_weight: float,
    block_size: int = 512,
) -> dict[str, float]:
    totals: dict[str, list[float]] = {
        "listwise_kl": [],
        "pairwise_bce": [],
        "score_mse": [],
        "total": [],
    }
    with torch.inference_mode():
        for start in range(0, len(queries), block_size):
            stop = min(start + block_size, len(queries))
            scores = (
                baseline[start:stop]
                if model is None
                else model(queries[start:stop], codes[start:stop], baseline[start:stop])
            )
            loss = lookup_table_distillation_loss(
                scores,
                teacher[start:stop],
                pairs[start:stop],
                temperature=TEMPERATURE,
                listwise_weight=listwise_weight,
                pairwise_weight=pairwise_weight,
                score_weight=score_weight,
            )
            count = stop - start
            for name in totals:
                totals[name].append(float(getattr(loss, name)) * count)
    return {name: math.fsum(parts) / len(queries) for name, parts in totals.items()}


def _fixed_candidate_rankings(
    model: QueryDependentLookupScorer,
    queries: torch.Tensor,
    gallery_codes: torch.Tensor,
    quantizer: object,
    candidates: torch.Tensor,
    *,
    block_size: int = 256,
) -> torch.Tensor:
    result = []
    with torch.inference_mode():
        for start in range(0, len(queries), block_size):
            stop = min(start + block_size, len(queries))
            indexes = candidates[start:stop]
            selected_codes = gallery_codes[indexes]
            baseline = -0.5 * aligned_product_squared_distances(
                queries[start:stop],
                selected_codes,
                quantizer,  # type: ignore[arg-type]
            )
            scores = model(queries[start:stop], selected_codes, baseline)
            ordinal_order = torch.argsort(indexes, dim=1, stable=True)
            indexes = indexes.gather(1, ordinal_order)
            scores = scores.gather(1, ordinal_order)
            order = torch.argsort(scores, dim=1, descending=True, stable=True)
            result.append(torch.gather(indexes, 1, order))
    return torch.cat(result, dim=0).cpu().contiguous()


def _exhaustive_lookup_rankings(
    model: QueryDependentLookupScorer,
    queries: torch.Tensor,
    gallery_codes: torch.Tensor,
    quantizer: object,
    *,
    width: int,
    block_size: int = 64,
) -> torch.Tensor:
    result = []
    with torch.inference_mode():
        for start in range(0, len(queries), block_size):
            stop = min(start + block_size, len(queries))
            baseline = (
                -0.5
                * quantizer.asymmetric_squared_distances(  # type: ignore[attr-defined]
                    queries[start:stop], gallery_codes
                )
            )
            scores = model.score_gallery(queries[start:stop], gallery_codes, baseline)
            scores[
                torch.arange(stop - start, device=queries.device),
                torch.arange(start, stop, device=queries.device),
            ] = -torch.inf
            retained = width + 1
            values, indexes = torch.topk(scores, k=retained, dim=1, largest=True, sorted=False)
            ordinal_order = torch.argsort(indexes, dim=1, stable=True)
            indexes = indexes.gather(1, ordinal_order)
            values = values.gather(1, ordinal_order)
            score_order = torch.argsort(values, dim=1, descending=True, stable=True)
            indexes = indexes.gather(1, score_order)
            values = values.gather(1, score_order)
            ambiguous = values[:, width - 1] == values[:, width]
            for block_row in torch.nonzero(ambiguous, as_tuple=False).flatten().tolist():
                boundary = values[block_row, width - 1]
                candidates = torch.nonzero(scores[block_row] >= boundary, as_tuple=False).flatten()
                candidates = torch.sort(candidates).values
                candidate_scores = scores[block_row, candidates]
                order = torch.argsort(candidate_scores, descending=True, stable=True)
                indexes[block_row, :width] = candidates[order[:width]]
            result.append(indexes[:, :width].clone().cpu())
    return torch.cat(result, dim=0).contiguous()


def _pairwise_agreement(
    model: QueryDependentLookupScorer,
    queries: torch.Tensor,
    codes: torch.Tensor,
    baseline: torch.Tensor,
    teacher: torch.Tensor,
    pairs: torch.Tensor,
) -> float:
    agreements = []
    with torch.inference_mode():
        for start in range(0, len(queries), 512):
            stop = min(start + 512, len(queries))
            scores = model(queries[start:stop], codes[start:stop], baseline[start:stop])
            selected = pairs[start:stop]
            student_delta = scores.gather(1, selected[:, :, 0]) - scores.gather(
                1, selected[:, :, 1]
            )
            teacher_delta = teacher[start:stop].gather(1, selected[:, :, 0]) - teacher[
                start:stop
            ].gather(1, selected[:, :, 1])
            agreements.append(
                (torch.sign(student_delta) == torch.sign(teacher_delta)).float().flatten()
            )
    return float(torch.cat(agreements).mean())


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-checkpoint-sha256", required=True)
    parser.add_argument("--parent-codec-checkpoint", type=Path, required=True)
    parser.add_argument("--parent-codec-checkpoint-sha256", required=True)
    parser.add_argument("--parent-codec-receipt", type=Path, required=True)
    parser.add_argument("--parent-codec-receipt-sha256", required=True)
    parser.add_argument("--prior-reranker-receipt", type=Path, required=True)
    parser.add_argument("--prior-reranker-receipt-sha256", required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--execute-pq-lookup-distillation", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = (args.model_output, args.receipt)
    if (
        any(
            not path.is_absolute() or not path.parent.is_dir() or path.exists() or path.is_symlink()
            for path in outputs
        )
        or outputs[0] == outputs[1]
        or args.seed != 0
        or any(
            _SHA256.fullmatch(value) is None
            for value in (
                args.source_sha256,
                args.teacher_sha256,
                args.direct_checkpoint_sha256,
                args.parent_codec_checkpoint_sha256,
                args.parent_codec_receipt_sha256,
                args.prior_reranker_receipt_sha256,
                args.driver_sha256,
            )
        )
        or _file_sha256(Path(__file__).resolve()) != args.driver_sha256
        or _file_sha256(args.direct_checkpoint) != args.direct_checkpoint_sha256
        or _file_sha256(args.parent_codec_checkpoint) != args.parent_codec_checkpoint_sha256
        or _file_sha256(args.parent_codec_receipt) != args.parent_codec_receipt_sha256
        or _file_sha256(args.prior_reranker_receipt) != args.prior_reranker_receipt_sha256
    ):
        raise ValueError("lookup distillation authority differs")
    parent_wire = args.parent_codec_receipt.read_bytes()
    parent_receipt = json.loads(parent_wire)
    if type(parent_receipt) is not dict or _canonical_json(parent_receipt) != parent_wire:
        raise ValueError("parent codec receipt authority differs")
    validate_parent_receipt(
        parent_receipt,
        parent_checkpoint_sha256=args.parent_codec_checkpoint_sha256,
        direct_checkpoint_sha256=args.direct_checkpoint_sha256,
        source_snapshot_sha256=args.source_sha256,
        teacher_snapshot_sha256=args.teacher_sha256,
    )
    prior_wire = args.prior_reranker_receipt.read_bytes()
    prior_receipt = json.loads(prior_wire)
    if type(prior_receipt) is not dict or _canonical_json(prior_receipt) != prior_wire:
        raise ValueError("prior reranker receipt authority differs")
    validate_prior_reranker_receipt(
        prior_receipt,
        direct_checkpoint_sha256=args.direct_checkpoint_sha256,
        parent_codec_checkpoint_sha256=args.parent_codec_checkpoint_sha256,
        parent_codec_receipt_sha256=args.parent_codec_receipt_sha256,
        source_snapshot_sha256=args.source_sha256,
        teacher_snapshot_sha256=args.teacher_sha256,
    )
    runtime = configure_deterministic_similarity_runtime(args.seed, cpu_threads=1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    source_identity = positive_coverage_source_identity(
        driver=Path(__file__).resolve(),
        driver_sha256=args.driver_sha256,
        source_revision=args.source_revision,
    )
    pair = load_paired_train_archives(
        args.source_snapshot,
        args.source_sha256,
        args.teacher_snapshot,
        args.teacher_sha256,
    )
    partition = deterministic_class_partition(
        pair["train_labels"], fit_fraction=0.8, seed=SPLIT_SEED
    )
    fit_indexes = list(partition.fit_row_indexes)
    validation_indexes = list(partition.validation_row_indexes)
    validation_labels = tuple(pair["train_labels"][row] for row in validation_indexes)
    if prior_receipt["partition"] != {
        "fit_rows": len(fit_indexes),
        "split_seed": SPLIT_SEED,
        "validation_rows": len(validation_indexes),
    }:
        raise ValueError("prior reranker receipt partition differs")
    state = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=True)
    if type(state) is not dict or set(state) != {"weight", "bias"}:
        raise ValueError("direct checkpoint schema differs")
    weight = state["weight"]
    bias = state["bias"]
    if (
        type(weight) is not torch.Tensor
        or weight.dtype != torch.float32
        or weight.shape != (DIMENSIONS, 768)
        or type(bias) is not torch.Tensor
        or bias.dtype != torch.float32
        or bias.shape != (DIMENSIONS,)
        or not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("direct checkpoint geometry differs")
    teacher = torch.nn.functional.normalize(pair["teacher_train"].float(), dim=1)
    with torch.inference_mode():
        embeddings = torch.nn.functional.normalize(
            torch.nn.functional.linear(teacher, weight, bias), dim=1
        ).contiguous()
    fit_rows = embeddings[fit_indexes].contiguous()
    validation_rows = embeddings[validation_indexes].contiguous()
    pq24, pq32, opq24 = _load_parent_quantizers(args.parent_codec_checkpoint)
    device = torch.device("cuda")
    validation = validation_rows.to(device)
    device = _concrete_tensor_device(validation, expected_type="cuda")
    fit_values = fit_rows.to(device)
    candidate_width = max(Counter(validation_labels).values()) - 1

    stability = []
    for seed in PQ_SEEDS:
        fitted_cpu = fit_product_quantizer(
            fit_rows, pq24.spec, seed=seed, maximum_iterations=MAXIMUM_ITERATIONS
        )
        if seed == 0 and any(
            not torch.equal(left, right)
            for left, right in zip(
                fitted_cpu.detached_codebooks(), pq24.detached_codebooks(), strict=True
            )
        ):
            raise RuntimeError("lookup distillation parent PQ24 codebooks differ")
        fitted = fitted_cpu.to(device)
        fitted_codes = fitted.hard_encode(validation)
        score = score_asymmetric_product(
            validation, fitted_codes, fitted, validation_labels, device=device
        )
        stability.append({"map_at_r": score["map_at_r"], "r1": score["r1"], "seed": seed})

    pq24 = pq24.to(device).eval()
    pq32 = pq32.to(device).eval()
    opq24 = opq24.to(device).eval()
    fit_codes = pq24.hard_encode(fit_values)
    validation_codes = pq24.hard_encode(validation)
    pq24_score = score_asymmetric_product(
        validation, validation_codes, pq24, validation_labels, device=device
    )
    pq32_codes = pq32.hard_encode(validation)
    pq32_score = score_asymmetric_product(
        validation, pq32_codes, pq32, validation_labels, device=device
    )
    opq24_codes = opq24.hard_encode(validation)
    opq24_score = score_asymmetric_optimized_product(
        validation, opq24_codes, opq24, validation_labels, device=device
    )
    float_raw = score_symmetric(
        validation_rows, validation_labels, candidate_width=candidate_width, device=device
    )
    float_score = {
        "map_at_r": float(float_raw["map_at_r"]),
        "per_query_ap": [float(value) for value in float_raw["per_query_ap"]],
        "r1": float(float_raw["r1"]),
    }

    fit_float = _exact_float_shortlists(fit_values, width=TEACHER_NEAR_WIDTH, query_block_size=256)
    fit_compressed = exact_product_shortlists(
        fit_values,
        fit_codes,
        pq24,
        width=COMPRESSED_MINING_WIDTH,
        query_block_size=256,
    )
    fit_candidates = build_label_free_candidate_pool(
        fit_float,
        fit_compressed,
        compressed_width=COMPRESSED_NEAR_WIDTH,
        uniform_width=UNIFORM_TAIL_WIDTH,
        uniform_seed=UNIFORM_TAIL_SEED,
    )
    candidate_codes, baseline, teacher_scores = _training_tensors(
        fit_values, fit_codes, fit_candidates, pq24
    )
    base_pairs = registered_pair_positions(
        rows=1,
        candidates=fit_candidates.shape[1],
        teacher_width=TEACHER_NEAR_WIDTH,
        compressed_width=COMPRESSED_NEAR_WIDTH,
        uniform_width=UNIFORM_TAIL_WIDTH,
        pairs_per_stratum=PAIRS_PER_STRATUM,
    ).to(device)
    pairs = base_pairs.expand(len(fit_values), -1, -1)
    spec = LookupTableDistillationSpec(
        query_dimensions=DIMENSIONS,
        blocks=BLOCKS,
        codebook_size=CODEBOOK_SIZE,
        hidden_dimensions=HIDDEN_DIMENSIONS,
        correction_dimensions=CORRECTION_DIMENSIONS,
    )
    arms = {
        "teacher_score_mse": {"listwise_weight": 0.0, "pairwise_weight": 0.0, "score_weight": 1.0},
        "ranking": {
            "listwise_weight": 1.0,
            "pairwise_weight": PAIRWISE_WEIGHT,
            "score_weight": RANK_SCORE_WEIGHT,
        },
    }
    models: dict[str, QueryDependentLookupScorer] = {}
    fit_receipts: dict[str, object] = {}
    updates_per_epoch = math.ceil(len(fit_values) / BATCH_SIZE)
    updates = EPOCHS * updates_per_epoch
    for name, weights in arms.items():
        initial_before = _loss_diagnostics(
            None,
            fit_values,
            candidate_codes,
            baseline,
            teacher_scores,
            pairs,
            listwise_weight=weights["listwise_weight"],
            pairwise_weight=weights["pairwise_weight"],
            score_weight=weights["score_weight"],
        )
        latest = {
            "baseline": baseline,
            "candidate_codes": candidate_codes,
            "pairs": pairs,
            "teacher_scores": teacher_scores,
        }
        refreshes: list[dict[str, int]] = []

        def refresh_candidates(
            current: QueryDependentLookupScorer,
            refresh_index: int,
            _latest: dict[str, torch.Tensor] = latest,
            _refreshes: list[dict[str, int]] = refreshes,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            current_student = _exhaustive_lookup_rankings(
                current,
                fit_values,
                fit_codes,
                pq24,
                width=COMPRESSED_MINING_WIDTH,
            ).to(device)
            current_candidates = build_label_free_candidate_pool(
                fit_float,
                current_student,
                compressed_width=COMPRESSED_NEAR_WIDTH,
                uniform_width=UNIFORM_TAIL_WIDTH,
                uniform_seed=UNIFORM_TAIL_SEED + refresh_index,
            )
            refreshed_codes, refreshed_baseline, refreshed_teacher = _training_tensors(
                fit_values, fit_codes, current_candidates, pq24
            )
            refreshed_pairs = base_pairs.expand(len(fit_values), -1, -1)
            _latest.update(
                {
                    "baseline": refreshed_baseline,
                    "candidate_codes": refreshed_codes,
                    "pairs": refreshed_pairs,
                    "teacher_scores": refreshed_teacher,
                }
            )
            _refreshes.append(
                {
                    "candidate_width": current_candidates.shape[1],
                    "index": refresh_index,
                    "uniform_seed": UNIFORM_TAIL_SEED + refresh_index,
                }
            )
            return refreshed_codes, refreshed_baseline, refreshed_teacher, refreshed_pairs

        started = time.monotonic()
        model = fit_query_dependent_lookup_scorer(
            fit_values,
            candidate_codes,
            baseline,
            teacher_scores,
            pairs,
            spec=spec,
            seed=args.seed,
            updates=updates,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            temperature=TEMPERATURE,
            listwise_weight=weights["listwise_weight"],
            pairwise_weight=weights["pairwise_weight"],
            score_weight=weights["score_weight"],
            refresh_interval=updates_per_epoch,
            refresh=refresh_candidates,
        )
        elapsed = time.monotonic() - started
        final_before = _loss_diagnostics(
            None,
            fit_values,
            latest["candidate_codes"],
            latest["baseline"],
            latest["teacher_scores"],
            latest["pairs"],
            listwise_weight=weights["listwise_weight"],
            pairwise_weight=weights["pairwise_weight"],
            score_weight=weights["score_weight"],
        )
        after = _loss_diagnostics(
            model,
            fit_values,
            latest["candidate_codes"],
            latest["baseline"],
            latest["teacher_scores"],
            latest["pairs"],
            listwise_weight=weights["listwise_weight"],
            pairwise_weight=weights["pairwise_weight"],
            score_weight=weights["score_weight"],
        )
        after_on_initial_pool = _loss_diagnostics(
            model,
            fit_values,
            candidate_codes,
            baseline,
            teacher_scores,
            pairs,
            listwise_weight=weights["listwise_weight"],
            pairwise_weight=weights["pairwise_weight"],
            score_weight=weights["score_weight"],
        )
        objective_improved = (
            after["total"] < final_before["total"]
            and after_on_initial_pool["total"] < initial_before["total"]
        )
        models[name] = model
        fit_receipts[name] = {
            "after": after,
            "after_on_initial_pool": after_on_initial_pool,
            "before_on_final_pool": final_before,
            "candidate_refreshes": refreshes,
            "initial_pool_before": initial_before,
            "objective_improved": objective_improved,
            "seconds": elapsed,
        }

    baseline_top32 = exact_product_shortlists(
        validation, validation_codes, pq24, width=32, query_block_size=256
    )
    evaluation: dict[str, object] = {}
    output_rankings: dict[str, object] = {}
    for name, model in models.items():
        fixed_rankings = _fixed_candidate_rankings(
            model, validation, validation_codes, pq24, baseline_top32
        )
        exhaustive_rankings = _exhaustive_lookup_rankings(
            model,
            validation,
            validation_codes,
            pq24,
            width=max(32, candidate_width),
        )
        fixed_score = score_ranked_candidates(fixed_rankings, validation_labels)
        exhaustive_score = score_ranked_candidates(exhaustive_rankings, validation_labels)
        top32_codes, top32_baseline, top32_teacher = _training_tensors(
            validation, validation_codes, baseline_top32, pq24
        )
        # Report agreement across every unordered pair in the fixed top-32 candidate set.
        near_ordinal = torch.arange(32, device=device, dtype=torch.int64)
        top32_pairs = (
            torch.combinations(near_ordinal, r=2).unsqueeze(0).expand(len(validation), -1, -1)
        )
        evaluation[name] = {
            "exhaustive": exhaustive_score,
            "fixed_pq24_top32": fixed_score,
            "teacher_pairwise_agreement": _pairwise_agreement(
                model,
                validation,
                top32_codes,
                top32_baseline,
                top32_teacher,
                top32_pairs,
            ),
        }
        output_rankings[name] = {
            "exhaustive": exhaustive_rankings.tolist(),
            "fixed_pq24_top32": fixed_rankings.tolist(),
        }

    seed_maps = [cast(float, row["map_at_r"]) for row in stability]
    variation = float(torch.tensor(seed_maps, dtype=torch.float64).std(unbiased=False))
    ranking_evaluation = cast(dict[str, object], evaluation["ranking"])
    ranking_score = cast(dict[str, object], ranking_evaluation["exhaustive"])
    ranking_fixed_score = cast(dict[str, object], ranking_evaluation["fixed_pq24_top32"])
    decision = lookup_distillation_decision(
        exhaustive_map_at_r=cast(float, ranking_score["map_at_r"]),
        fixed_candidate_map_at_r=cast(float, ranking_fixed_score["map_at_r"]),
        r1=cast(float, ranking_score["r1"]),
        pq24_r1=cast(float, pq24_score["r1"]),
        pq32_map_at_r=cast(float, pq32_score["map_at_r"]),
        prior_reranker_map_at_r=PRIOR_RERANKER_MAP,
        pq_codebook_variation=variation,
        objective_improved=cast(
            bool,
            cast(dict[str, object], fit_receipts["ranking"])["objective_improved"],
        ),
    )
    checkpoint = {
        "models": {
            name: {key: value.detach().cpu() for key, value in model.state_dict().items()}
            for name, model in models.items()
        },
        "parent_codec_checkpoint_sha256": args.parent_codec_checkpoint_sha256,
        "recipe": {
            "batch_size": BATCH_SIZE,
            "candidate_pool": [
                TEACHER_NEAR_WIDTH,
                COMPRESSED_NEAR_WIDTH,
                UNIFORM_TAIL_WIDTH,
            ],
            "compressed_mining_width": COMPRESSED_MINING_WIDTH,
            "correction_dimensions": CORRECTION_DIMENSIONS,
            "epochs": EPOCHS,
            "hidden_dimensions": HIDDEN_DIMENSIONS,
            "learning_rate": LEARNING_RATE,
            "maximum_gradient_norm": LOOKUP_MAXIMUM_GRADIENT_NORM,
            "maximum_pq_iterations": MAXIMUM_ITERATIONS,
            "pairwise_weight": PAIRWISE_WEIGHT,
            "pairs_per_stratum": PAIRS_PER_STRATUM,
            "pq_seeds": list(PQ_SEEDS),
            "rank_score_weight": RANK_SCORE_WEIGHT,
            "ranking_listwise_weight": 1.0,
            "temperature": TEMPERATURE,
            "uniform_tail_seed": UNIFORM_TAIL_SEED,
            "updates": updates,
            "updates_per_epoch": updates_per_epoch,
            "weight_decay": LOOKUP_WEIGHT_DECAY,
        },
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint, checkpoint_buffer)
    checkpoint_wire = checkpoint_buffer.getvalue()
    checkpoint_sha256 = hashlib.sha256(checkpoint_wire).hexdigest()
    with publish_bytes_noreplace(
        args.model_output,
        checkpoint_wire,
        validator=lambda wire: (
            None
            if hashlib.sha256(wire).hexdigest() == checkpoint_sha256
            else (_ for _ in ()).throw(ValueError("lookup checkpoint differs"))
        ),
    ):
        pass
    receipt: dict[str, object] = {
        "claim_eligible": False,
        "dataset": "sop-official-train-class-disjoint-validation",
        "decision": decision,
        "evaluation": evaluation,
        "fit": fit_receipts,
        "inputs": {
            "direct_checkpoint_sha256": args.direct_checkpoint_sha256,
            "parent_codec_checkpoint_sha256": args.parent_codec_checkpoint_sha256,
            "parent_codec_receipt_sha256": args.parent_codec_receipt_sha256,
            "prior_reranker_receipt_sha256": args.prior_reranker_receipt_sha256,
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "matched_controls": {
            "float": float_score,
            "opq24": opq24_score,
            "pq24": pq24_score,
            "pq32": pq32_score,
        },
        "model_checkpoint": {"bytes": len(checkpoint_wire), "sha256": checkpoint_sha256},
        "model_parameter_bytes": {
            name: sum(
                parameter.numel() * parameter.element_size() for parameter in model.parameters()
            )
            for name, model in models.items()
        },
        "serving_cost_evidence": {
            "combined_pq_and_correction_kernel_implemented": False,
            "correction_lookup_additions_per_candidate": BLOCKS,
            "correction_table_bytes_per_query_f32": BLOCKS * CODEBOOK_SIZE * 4,
            "pq_baseline_lookup_additions_per_candidate": BLOCKS,
        },
        "official_test_touched": False,
        "partition": {
            "fit_rows": len(fit_indexes),
            "split_seed": SPLIT_SEED,
            "validation_rows": len(validation_indexes),
        },
        "pq24_seed_stability": {
            "map_at_r_population_standard_deviation": variation,
            "rows": stability,
        },
        "lookup_model_seed_stability": {
            "measured": False,
            "reason": (
                "seed-0 kill screen; target success requires a separate frozen "
                "multi-seed confirmation"
            ),
        },
        "rankings": output_rankings,
        "recipe": checkpoint["recipe"],
        "runtime": runtime._asdict(),
        "schema": "sfora-pq-lookup-distillation-v1",
        "seed": args.seed,
        "source": source_identity,
    }
    receipt_wire = _canonical_json(receipt)
    with publish_bytes_noreplace(
        args.receipt,
        receipt_wire,
        validator=lambda wire: (
            None
            if wire == receipt_wire
            else (_ for _ in ()).throw(ValueError("lookup receipt differs"))
        ),
    ):
        pass
    print(receipt_wire.decode(), end="")


if __name__ == "__main__":
    main()
