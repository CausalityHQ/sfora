#!/usr/bin/env python3
"""Run one frozen label-free 24-byte additive-quantization screen."""

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
from run_sop_pq_lookup_distillation import (
    _concrete_tensor_device,
    _exact_float_shortlists,
)

from sfora.additive_quantization import (
    AdditiveFitSpec,
    AdditiveQuantizationSpec,
    AdditiveQuantizer,
    fit_additive_quantizer,
    padded_product_codebooks,
)
from sfora.atomic_publication import publish_bytes_noreplace
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.representation_ceiling import deterministic_class_partition

DIMENSIONS = 128
STAGES = 24
CODEBOOK_SIZE = 256
ROUNDS = 8
ASSIGNMENT_SWEEPS = 2
CODEBOOK_EPOCHS = 2
BATCH_SIZE = 1_024
LEARNING_RATES = (1e-3,) * 4 + (3e-4,) * 4
HELDOUT_SWEEPS = 8
SPLIT_SEED = 17
QUERY_BLOCK_SIZE = 128
TARGET_MAP = 0.58563
MINIMUM_RECOVERY_MAP = 0.5809
MAXIMUM_R1_REGRESSION = 0.002
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_failed_lookup_receipt(
    value: object,
    *,
    direct_checkpoint_sha256: str,
    parent_codec_checkpoint_sha256: str,
    parent_codec_receipt_sha256: str,
    source_snapshot_sha256: str,
    teacher_snapshot_sha256: str,
) -> None:
    """Require the exact failed fixed-code experiment that motivates re-encoding."""

    expected_inputs = {
        "direct_checkpoint_sha256": direct_checkpoint_sha256,
        "parent_codec_checkpoint_sha256": parent_codec_checkpoint_sha256,
        "parent_codec_receipt_sha256": parent_codec_receipt_sha256,
        "source_snapshot_sha256": source_snapshot_sha256,
        "teacher_snapshot_sha256": teacher_snapshot_sha256,
    }
    if (
        type(value) is not dict
        or value.get("schema") != "sfora-pq-lookup-distillation-v1"
        or value.get("claim_eligible") is not False
        or value.get("official_test_touched") is not False
        or type(value.get("decision")) is not dict
        or value["decision"].get("classification") != "fixed-code-lookup-r1-regressed"
        or type(value.get("inputs")) is not dict
        or any(value["inputs"].get(key) != expected for key, expected in expected_inputs.items())
    ):
        raise ValueError("failed lookup receipt authority differs")


def exact_additive_rankings(
    queries: torch.Tensor,
    gallery_codes: torch.Tensor,
    quantizer: AdditiveQuantizer,
    *,
    width: int,
    query_block_size: int,
) -> torch.Tensor:
    """Rank an aligned gallery with stable row-ordinal ties and self exclusion."""

    if (
        type(queries) is not torch.Tensor
        or queries.dtype != torch.float32
        or queries.ndim != 2
        or queries.shape[0] != gallery_codes.shape[0]
        or type(width) is not int
        or width < 1
        or width >= queries.shape[0]
        or type(query_block_size) is not int
        or query_block_size < 1
    ):
        raise ValueError("additive ranking authority differs")
    rankings = []
    for start in range(0, queries.shape[0], query_block_size):
        stop = min(start + query_block_size, queries.shape[0])
        scores = quantizer.asymmetric_dot_scores(queries[start:stop], gallery_codes)
        local = torch.arange(stop - start, device=scores.device)
        scores[local, torch.arange(start, stop, device=scores.device)] = -torch.inf
        rankings.append(
            torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width].clone()
        )
    return torch.cat(rankings, dim=0).contiguous()


def metric_rankings(rankings: torch.Tensor) -> torch.Tensor:
    """Materialize integer rankings on the CPU metric boundary."""

    if type(rankings) is not torch.Tensor or rankings.dtype != torch.int64 or rankings.ndim != 2:
        raise ValueError("metric ranking authority differs")
    return rankings.detach().cpu().contiguous()


def publish_checkpoint_or_validate_existing(path: Path, wire: bytes) -> None:
    """Publish a checkpoint once or accept its exact prior crash-recovery bytes."""

    if not isinstance(path, Path) or type(wire) is not bytes or not wire or path.is_symlink():
        raise ValueError("joint codec checkpoint differs")
    if path.exists():
        if not path.is_file() or path.read_bytes() != wire:
            raise ValueError("joint codec checkpoint differs")
        return
    expected_sha256 = hashlib.sha256(wire).hexdigest()
    with publish_bytes_noreplace(
        path,
        wire,
        validator=lambda observed: (
            None
            if hashlib.sha256(observed).hexdigest() == expected_sha256
            else (_ for _ in ()).throw(ValueError("joint codec checkpoint differs"))
        ),
    ):
        pass


def joint_codec_decision(
    *,
    best_map_at_r: float,
    best_r1: float,
    pq32_map_at_r: float,
    matched_pq24_dot_r1: float,
    padded_pq24_dot_map_at_r: float,
) -> dict[str, object]:
    """Classify one seed-zero screen against frozen quality gates."""

    if not all(
        type(value) is float and math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (
            best_map_at_r,
            best_r1,
            pq32_map_at_r,
            matched_pq24_dot_r1,
            padded_pq24_dot_map_at_r,
        )
    ):
        raise ValueError("joint codec decision authority differs")
    if best_map_at_r < pq32_map_at_r:
        classification = "additive-codec-failed-pq32"
    elif best_map_at_r < MINIMUM_RECOVERY_MAP:
        classification = "additive-codec-insufficient-recovery"
    elif best_r1 < matched_pq24_dot_r1 - MAXIMUM_R1_REGRESSION:
        classification = "additive-codec-r1-regressed"
    elif best_map_at_r < TARGET_MAP:
        classification = "additive-codec-seed0-continue"
    else:
        classification = "additive-codec-target-screen"
    return {
        "classification": classification,
        "continuation_eligible": classification
        in {"additive-codec-seed0-continue", "additive-codec-target-screen"},
        "gain_over_padded_pq24_dot": best_map_at_r - padded_pq24_dot_map_at_r,
        "matched_pq24_dot_r1": matched_pq24_dot_r1,
        "maximum_r1_regression_from_matched_pq24_dot": MAXIMUM_R1_REGRESSION,
        "minimum_recovery_map_at_r": MINIMUM_RECOVERY_MAP,
        "pq32_cross_functional_quality_bar": pq32_map_at_r,
        "passed": classification == "additive-codec-target-screen",
        "target_map_at_r": TARGET_MAP,
    }


def select_joint_codec_arm(
    arm_qualities: dict[str, dict[str, float]],
    *,
    pq32_map_at_r: float,
    matched_pq24_dot_r1: float,
    padded_pq24_dot_map_at_r: float,
) -> tuple[str, dict[str, dict[str, object]]]:
    """Select the highest-mAP arm among those satisfying every frozen gate."""

    if (
        type(arm_qualities) is not dict
        or not arm_qualities
        or any(
            type(name) is not str
            or not name
            or type(quality) is not dict
            or set(quality) != {"map_at_r", "r1"}
            for name, quality in arm_qualities.items()
        )
    ):
        raise ValueError("joint codec arm authority differs")
    decisions = {
        name: joint_codec_decision(
            best_map_at_r=quality["map_at_r"],
            best_r1=quality["r1"],
            pq32_map_at_r=pq32_map_at_r,
            matched_pq24_dot_r1=matched_pq24_dot_r1,
            padded_pq24_dot_map_at_r=padded_pq24_dot_map_at_r,
        )
        for name, quality in arm_qualities.items()
    }
    eligible = [
        name for name, decision in decisions.items() if decision["continuation_eligible"] is True
    ]
    candidates = eligible if eligible else list(arm_qualities)
    selected = sorted(candidates, key=lambda name: (-arm_qualities[name]["map_at_r"], name))[0]
    return selected, decisions


def _float_rerank_within_candidates(values: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
    """Order candidate rows by exact float scores and stable row ties."""

    result = []
    for start in range(0, values.shape[0], QUERY_BLOCK_SIZE):
        stop = min(start + QUERY_BLOCK_SIZE, values.shape[0])
        ordered_candidates = torch.sort(candidates[start:stop], dim=1, stable=True).values
        candidate_vectors = values[ordered_candidates]
        scores = torch.einsum("bd,bkd->bk", values[start:stop], candidate_vectors)
        result.append(
            ordered_candidates.gather(
                1, torch.argsort(scores, dim=1, descending=True, stable=True)
            ).clone()
        )
    return torch.cat(result, dim=0).contiguous()


def _ordering_diagnostics(
    values: torch.Tensor,
    codes: torch.Tensor,
    quantizer: AdditiveQuantizer,
    rankings: torch.Tensor,
    float_reference_rankings: torch.Tensor,
) -> dict[str, object]:
    """Measure local ordering, containment, and exhaustive positive-error tails."""

    widths = (1, 8, 32, 128)
    overlap_totals = {width: 0 for width in widths}
    pair_agreements = 0
    pair_total = 0
    intruders = 0
    maxima = []
    pair_positions = torch.combinations(torch.arange(32, device=values.device), r=2)
    for start in range(0, values.shape[0], QUERY_BLOCK_SIZE):
        stop = min(start + QUERY_BLOCK_SIZE, values.shape[0])
        local_rankings = rankings[start:stop]
        local_reference = float_reference_rankings[start:stop]
        for width in widths:
            matches = (local_rankings[:, :width, None] == local_reference[:, None, :width]).any(
                dim=2
            )
            overlap_totals[width] += int(matches.sum())
        intruders += int(
            (~(local_rankings[:, :32, None] == local_reference[:, None, :128]).any(dim=2)).sum()
        )
        student_scores = quantizer.asymmetric_dot_scores(values[start:stop], codes)
        teacher_scores = values[start:stop] @ values.T
        local = torch.arange(stop - start, device=values.device)
        aligned = torch.arange(start, stop, device=values.device)
        student_scores[local, aligned] = -torch.inf
        teacher_scores[local, aligned] = -torch.inf
        positive_error = student_scores - teacher_scores
        positive_error[local, aligned] = -torch.inf
        maxima.extend(positive_error.amax(dim=1).cpu().tolist())
        top32 = local_reference[:, :32]
        selected = student_scores.gather(1, top32)
        left = pair_positions[:, 0]
        right = pair_positions[:, 1]
        left_scores = selected[:, left]
        right_scores = selected[:, right]
        left_rows = top32[:, left]
        right_rows = top32[:, right]
        pair_agreements += int(
            (
                (left_scores > right_scores)
                | ((left_scores == right_scores) & (left_rows < right_rows))
            ).sum()
        )
        pair_total += selected.shape[0] * pair_positions.shape[0]
    maxima_tensor = torch.tensor(maxima, dtype=torch.float64)
    return {
        "float_reranked_top32": _float_rerank_within_candidates(values, rankings[:, :32]).tolist(),
        "mean_coded_top32_intruders_outside_float_head_top128": intruders / values.shape[0],
        "score_positive_error_maximum": float(maxima_tensor.max()),
        "score_positive_error_per_query_maximum_p99": float(torch.quantile(maxima_tensor, 0.99)),
        "float_head_top32_pairwise_agreement": pair_agreements / pair_total,
        "float_head_topk_overlap": {
            str(width): overlap_totals[width] / (values.shape[0] * width) for width in widths
        },
    }


def _reconstruction_diagnostics(
    values: torch.Tensor,
    codes: torch.Tensor,
    initial_codes: torch.Tensor,
    quantizer: AdditiveQuantizer,
) -> dict[str, object]:
    decoded = quantizer.hard_decode(codes)
    residual = values - decoded
    parallel = (values * residual).sum(dim=1).square()
    squared = residual.square().sum(dim=1)
    utilization = [
        int(torch.unique(codes[:, stage]).numel()) / quantizer.spec.codebook_size
        for stage in range(quantizer.spec.stages)
    ]
    return {
        "assignment_churn_ppm": int((codes != initial_codes).sum()) * 1_000_000 // codes.numel(),
        "mean_parallel_squared_error": float(parallel.mean()),
        "mean_squared_error": float(squared.mean()),
        "mean_tangential_squared_error": float((squared - parallel).clamp_min(0.0).mean()),
        "minimum_stage_utilization": min(utilization),
        "mean_stage_utilization": sum(utilization) / len(utilization),
    }


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
    parser.add_argument("--failed-lookup-receipt", type=Path, required=True)
    parser.add_argument("--failed-lookup-receipt-sha256", required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--execute-joint-product-quantization", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = (args.model_output, args.receipt)
    digest_values = (
        args.source_sha256,
        args.teacher_sha256,
        args.direct_checkpoint_sha256,
        args.parent_codec_checkpoint_sha256,
        args.parent_codec_receipt_sha256,
        args.failed_lookup_receipt_sha256,
        args.driver_sha256,
    )
    if (
        any(not path.is_absolute() or not path.parent.is_dir() for path in outputs)
        or args.receipt.exists()
        or args.receipt.is_symlink()
        or args.model_output.is_symlink()
        or (args.model_output.exists() and not args.model_output.is_file())
        or outputs[0] == outputs[1]
        or args.seed != 0
        or any(_SHA256.fullmatch(value) is None for value in digest_values)
        or _file_sha256(Path(__file__).resolve()) != args.driver_sha256
        or _file_sha256(args.direct_checkpoint) != args.direct_checkpoint_sha256
        or _file_sha256(args.parent_codec_checkpoint) != args.parent_codec_checkpoint_sha256
        or _file_sha256(args.parent_codec_receipt) != args.parent_codec_receipt_sha256
        or _file_sha256(args.failed_lookup_receipt) != args.failed_lookup_receipt_sha256
    ):
        raise ValueError("joint product quantization authority differs")
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
    failed_wire = args.failed_lookup_receipt.read_bytes()
    failed_receipt = json.loads(failed_wire)
    if type(failed_receipt) is not dict or _canonical_json(failed_receipt) != failed_wire:
        raise ValueError("failed lookup receipt authority differs")
    validate_failed_lookup_receipt(
        failed_receipt,
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
    expected_partition = {
        "fit_rows": len(fit_indexes),
        "split_seed": SPLIT_SEED,
        "validation_rows": len(validation_indexes),
    }
    if failed_receipt.get("partition") != expected_partition:
        raise ValueError("failed lookup receipt partition differs")
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
    pq24 = pq24.to(device).eval()
    pq32 = pq32.to(device).eval()
    opq24 = opq24.to(device).eval()
    if (
        pq24.spec.dimensions != DIMENSIONS
        or pq24.spec.bytes_per_vector != STAGES
        or pq24.spec.codebook_size != CODEBOOK_SIZE
    ):
        raise ValueError("parent PQ24 geometry differs")
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
    candidate_width = max(Counter(validation_labels).values()) - 1
    float_raw = score_symmetric(
        validation_rows, validation_labels, candidate_width=candidate_width, device=device
    )
    float_score = {
        "map_at_r": float(float_raw["map_at_r"]),
        "per_query_ap": [float(value) for value in float_raw["per_query_ap"]],
        "r1": float(float_raw["r1"]),
    }
    float_reference_rankings = _exact_float_shortlists(
        validation, width=128, query_block_size=QUERY_BLOCK_SIZE
    )
    additive_initial = padded_product_codebooks(
        pq24.spec.block_dimensions, pq24.detached_codebooks()
    ).to(device)
    initial_additive = AdditiveQuantizer.from_codebooks(
        AdditiveQuantizationSpec(dimensions=DIMENSIONS, stages=STAGES, codebook_size=CODEBOOK_SIZE),
        additive_initial,
    ).eval()
    initial_additive_rankings = exact_additive_rankings(
        validation,
        validation_codes,
        initial_additive,
        width=max(128, candidate_width),
        query_block_size=QUERY_BLOCK_SIZE,
    )
    initial_additive_score = score_ranked_candidates(
        metric_rankings(initial_additive_rankings[:, :candidate_width]), validation_labels
    )
    initial_ordering = _ordering_diagnostics(
        validation,
        validation_codes,
        initial_additive,
        initial_additive_rankings,
        float_reference_rankings,
    )
    initial_float_reranked = score_ranked_candidates(
        metric_rankings(
            torch.tensor(initial_ordering.pop("float_reranked_top32"), device=device)[
                :, :candidate_width
            ]
        ),
        validation_labels,
    )
    heldout_generator = torch.Generator(device="cpu")
    heldout_generator.manual_seed(args.seed + 10_000)
    heldout_stage_order = tuple(
        int(value) for value in torch.randperm(STAGES, generator=heldout_generator).tolist()
    )
    reencoded_control_codes = initial_additive.coordinate_descent_encode(
        validation,
        initial_codes=validation_codes,
        sweeps=HELDOUT_SWEEPS,
        parallel_weight=3.0,
        stage_order=heldout_stage_order,
    )
    reencoded_control_rankings = exact_additive_rankings(
        validation,
        reencoded_control_codes,
        initial_additive,
        width=max(128, candidate_width),
        query_block_size=QUERY_BLOCK_SIZE,
    )
    reencoded_control_score = score_ranked_candidates(
        metric_rankings(reencoded_control_rankings[:, :candidate_width]), validation_labels
    )
    reencoded_control_ordering = _ordering_diagnostics(
        validation,
        reencoded_control_codes,
        initial_additive,
        reencoded_control_rankings,
        float_reference_rankings,
    )
    reencoded_control_float_reranked = score_ranked_candidates(
        metric_rankings(
            torch.tensor(reencoded_control_ordering.pop("float_reranked_top32"), device=device)[
                :, :candidate_width
            ]
        ),
        validation_labels,
    )
    arms: dict[str, object] = {}
    checkpoint_arms: dict[str, object] = {}
    for name, parallel_weight in (("isotropic", 0.0), ("anisotropic", 3.0)):
        fit_spec = AdditiveFitSpec(
            rounds=ROUNDS,
            assignment_sweeps=ASSIGNMENT_SWEEPS,
            codebook_epochs=CODEBOOK_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rates=LEARNING_RATES,
            parallel_weight=parallel_weight,
            seed=args.seed,
        )
        started = time.monotonic()
        fitted = fit_additive_quantizer(
            fit_values,
            initial_codebooks=additive_initial,
            initial_codes=fit_codes,
            fit_spec=fit_spec,
        )
        fitted.quantizer.eval()
        heldout_codes = fitted.quantizer.coordinate_descent_encode(
            validation,
            initial_codes=validation_codes,
            sweeps=HELDOUT_SWEEPS,
            parallel_weight=parallel_weight,
            stage_order=heldout_stage_order,
        )
        elapsed = time.monotonic() - started
        rankings = exact_additive_rankings(
            validation,
            heldout_codes,
            fitted.quantizer,
            width=max(128, candidate_width),
            query_block_size=QUERY_BLOCK_SIZE,
        )
        score = score_ranked_candidates(
            metric_rankings(rankings[:, :candidate_width]), validation_labels
        )
        ordering = _ordering_diagnostics(
            validation, heldout_codes, fitted.quantizer, rankings, float_reference_rankings
        )
        float_reranked = score_ranked_candidates(
            metric_rankings(
                torch.tensor(ordering.pop("float_reranked_top32"), device=device)[
                    :, :candidate_width
                ]
            ),
            validation_labels,
        )
        arms[name] = {
            "elapsed_seconds": elapsed,
            "fit_assignment_churn_ppm": list(fitted.assignment_churn_ppm),
            "fit_objectives": list(fitted.objectives),
            "fit_stage_utilization_ppm": [
                list(values) for values in fitted.fit_stage_utilization_ppm
            ],
            "float_reranked_within_top32": float_reranked,
            "ordering": ordering,
            "quality": score,
            "rankings_top32": rankings[:, :32].tolist(),
            "reconstruction": _reconstruction_diagnostics(
                validation, heldout_codes, validation_codes, fitted.quantizer
            ),
        }
        checkpoint_arms[name] = {
            "codebooks": fitted.quantizer.detached_codebooks(),
            "parallel_weight": parallel_weight,
        }
    arm_qualities = {
        name: {
            "map_at_r": cast(
                float, cast(dict[str, object], cast(dict[str, object], arm)["quality"])["map_at_r"]
            ),
            "r1": cast(
                float, cast(dict[str, object], cast(dict[str, object], arm)["quality"])["r1"]
            ),
        }
        for name, arm in arms.items()
    }
    best_name, arm_decisions = select_joint_codec_arm(
        arm_qualities,
        pq32_map_at_r=cast(float, pq32_score["map_at_r"]),
        matched_pq24_dot_r1=cast(float, initial_additive_score["r1"]),
        padded_pq24_dot_map_at_r=cast(float, initial_additive_score["map_at_r"]),
    )
    decision = dict(arm_decisions[best_name])
    decision["arm_decisions"] = arm_decisions
    decision["best_arm"] = best_name
    decision["selection_policy"] = "preregistered-best-of-two-claim-ineligible"
    checkpoint = {
        "arms": checkpoint_arms,
        "parent_codec_checkpoint_sha256": args.parent_codec_checkpoint_sha256,
        "recipe": {
            "assignment_sweeps_per_round": ASSIGNMENT_SWEEPS,
            "batch_size": BATCH_SIZE,
            "codebook_epochs_per_round": CODEBOOK_EPOCHS,
            "codebook_size": CODEBOOK_SIZE,
            "dimensions": DIMENSIONS,
            "heldout_coordinate_sweeps": HELDOUT_SWEEPS,
            "learning_rates": list(LEARNING_RATES),
            "parallel_weights": {"anisotropic": 3.0, "isotropic": 0.0},
            "rounds": ROUNDS,
            "stages": STAGES,
        },
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint, checkpoint_buffer)
    checkpoint_wire = checkpoint_buffer.getvalue()
    checkpoint_sha256 = hashlib.sha256(checkpoint_wire).hexdigest()
    receipt: dict[str, object] = {
        "arms": arms,
        "claim_eligible": False,
        "dataset": "sop-official-train-class-disjoint-validation",
        "decision": decision,
        "inputs": {
            "direct_checkpoint_sha256": args.direct_checkpoint_sha256,
            "failed_lookup_receipt_sha256": args.failed_lookup_receipt_sha256,
            "parent_codec_checkpoint_sha256": args.parent_codec_checkpoint_sha256,
            "parent_codec_receipt_sha256": args.parent_codec_receipt_sha256,
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "method_family": {
            "anisotropic_objective": "ScaNN-style parallel-error weighting",
            "codec": "full-dimensional additive quantization",
            "encoder": "incumbent-preserving local-search coordinate descent",
            "novelty_scope": "exact-24-byte no-sidecar causal screen",
        },
        "matched_controls": {
            "float": float_score,
            "opq24": opq24_score,
            "padded_pq24_anisotropic_reencoded": {
                "float_reranked_within_top32": reencoded_control_float_reranked,
                "ordering": reencoded_control_ordering,
                "quality": reencoded_control_score,
                "rankings_top32": reencoded_control_rankings[:, :32].tolist(),
                "reconstruction": _reconstruction_diagnostics(
                    validation,
                    reencoded_control_codes,
                    validation_codes,
                    initial_additive,
                ),
            },
            "padded_pq24_dot": {
                "float_reranked_within_top32": initial_float_reranked,
                "ordering": initial_ordering,
                "quality": initial_additive_score,
                "rankings_top32": initial_additive_rankings[:, :32].tolist(),
                "reconstruction": _reconstruction_diagnostics(
                    validation,
                    validation_codes,
                    validation_codes,
                    initial_additive,
                ),
            },
            "pq24": pq24_score,
            "pq24_adc_r1_reference": pq24_score["r1"],
            "pq32": pq32_score,
        },
        "model_checkpoint": {"bytes": len(checkpoint_wire), "sha256": checkpoint_sha256},
        "official_test_touched": False,
        "partition": expected_partition,
        "recipe": checkpoint["recipe"],
        "runtime": runtime._asdict(),
        "schema": "sfora-joint-product-quantization-v1",
        "seed": args.seed,
        "serving": {
            "code_bytes_per_vector": STAGES,
            "database_sidecar_bytes_per_vector": 0,
            "lookup_additions_per_candidate": STAGES,
            "shared_codebook_bytes_per_arm": STAGES * CODEBOOK_SIZE * DIMENSIONS * 4,
            "table_bytes_per_query_f32": STAGES * CODEBOOK_SIZE * 4,
        },
        "source": source_identity,
    }
    receipt_wire = _canonical_json(receipt)
    publish_checkpoint_or_validate_existing(args.model_output, checkpoint_wire)
    with publish_bytes_noreplace(
        args.receipt,
        receipt_wire,
        validator=lambda wire: (
            None
            if wire == receipt_wire
            else (_ for _ in ()).throw(ValueError("joint codec receipt differs"))
        ),
    ):
        pass
    print(receipt_wire.decode(), end="")


if __name__ == "__main__":
    main()
