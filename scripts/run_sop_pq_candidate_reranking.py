#!/usr/bin/env python3
"""Train one frozen label-free candidate-set reranker on SOP fitting classes."""

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

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.pq_candidate_reranking import (
    CandidateSetReranker,
    ProductCodeResidualStatistics,
    aligned_product_squared_distances,
    candidate_set_distillation_loss,
    exact_product_shortlists,
    fit_candidate_set_reranker,
)
from sfora.product_quantization import (
    OptimizedProductQuantizer,
    ProductQuantizationSpec,
    ProductQuantizer,
    fit_product_quantizer,
)
from sfora.representation_ceiling import deterministic_class_partition

DIMENSIONS = 128
BLOCKS = 24
CODEBOOK_SIZE = 256
SHORTLIST_WIDTH = 32
MODEL_DIMENSIONS = 128
HEADS = 4
LAYERS = 3
UPDATES = 20_000
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
TEMPERATURE = 0.05
SCORE_WEIGHT = 0.1
MAXIMUM_ITERATIONS = 25
SPLIT_SEED = 17
TARGET_MAP = 0.58563
MAXIMUM_R1_REGRESSION = 0.002
PQ_SEEDS = (0, 1, 2, 3, 4)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_PARENT_KEYS = {
    "claim_eligible",
    "codec",
    "codebook_checkpoint",
    "dataset",
    "decision",
    "fit_seconds",
    "fit_seconds_by_arm",
    "implementation",
    "inputs",
    "matched_controls",
    "official_test_touched",
    "optimized_product",
    "partition",
    "relative_validation_squared_error",
    "restored_norms",
    "runtime",
    "schema",
    "score",
    "seed",
    "source",
    "used_codewords_per_stage",
}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: dict[str, object]) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def validate_parent_receipt(
    value: object,
    *,
    parent_checkpoint_sha256: str,
    direct_checkpoint_sha256: str,
    source_snapshot_sha256: str,
    teacher_snapshot_sha256: str,
) -> None:
    """Bind the inherited PQ codebooks to their authenticated experiment inputs."""

    if (
        type(value) is not dict
        or set(value) != _PARENT_KEYS
        or value.get("claim_eligible") is not False
        or value.get("dataset") != "sop-official-train-class-disjoint-validation"
        or value.get("official_test_touched") is not False
        or value.get("schema") != "sfora-additive-codec-preflight-v2"
        or type(value.get("seed")) is not int
        or value["seed"] != 0
        or type(value.get("codebook_checkpoint")) is not dict
        or value["codebook_checkpoint"].get("sha256") != parent_checkpoint_sha256
        or value.get("inputs")
        != {
            "direct_checkpoint_sha256": direct_checkpoint_sha256,
            "source_snapshot_sha256": source_snapshot_sha256,
            "teacher_snapshot_sha256": teacher_snapshot_sha256,
        }
        or type(value.get("partition")) is not dict
        or value["partition"].get("split_seed") != SPLIT_SEED
    ):
        raise ValueError("parent codec receipt authority differs")


def candidate_reranking_decision(
    *,
    map_at_r: float,
    r1: float,
    pq24_map_at_r: float,
    pq32_map_at_r: float,
    pq24_r1: float,
    seed_standard_deviation: float,
) -> dict[str, object]:
    """Classify the frozen candidate-set screen without tuning on its outcome."""

    values = (map_at_r, r1, pq24_map_at_r, pq32_map_at_r, pq24_r1, seed_standard_deviation)
    if any(
        type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in values
    ):
        raise ValueError("candidate reranking result differs")
    minimum_r1 = pq24_r1 - MAXIMUM_R1_REGRESSION
    if map_at_r < pq32_map_at_r:
        classification = "candidate-set-reranker-failed-pq32"
    elif r1 < minimum_r1:
        classification = "candidate-set-reranker-r1-regressed"
    elif map_at_r - pq24_map_at_r <= seed_standard_deviation:
        classification = "candidate-set-reranker-within-seed-variation"
    elif map_at_r < TARGET_MAP:
        classification = "candidate-set-reranker-improved"
    else:
        classification = "candidate-set-reranker-target-passed"
    return {
        "classification": classification,
        "gates": {
            "match_pq32_map_at_r": pq32_map_at_r,
            "minimum_r1": minimum_r1,
            "minimum_signal_over_seed_standard_deviation": seed_standard_deviation,
            "target_map_at_r": TARGET_MAP,
        },
        "passed": classification == "candidate-set-reranker-target-passed",
    }


def score_ranked_candidates(rankings: torch.Tensor, labels: tuple[int, ...]) -> dict[str, object]:
    """Recompute leave-self-out mAP@R and R@1 from bounded row rankings."""

    if (
        type(rankings) is not torch.Tensor
        or rankings.dtype != torch.int64
        or rankings.ndim != 2
        or rankings.shape[0] < 1
        or rankings.shape[1] < 1
        or rankings.device.type != "cpu"
        or type(labels) is not tuple
        or len(labels) != len(rankings)
        or any(type(label) is not int for label in labels)
        or bool((rankings < 0).any())
        or bool((rankings >= len(labels)).any())
    ):
        raise ValueError("candidate ranking score authority differs")
    counts = Counter(labels)
    if not counts or min(counts.values()) < 2 or rankings.shape[1] < max(counts.values()) - 1:
        raise ValueError("candidate ranking score authority differs")
    average_precisions = []
    hits = []
    for row, (ranking, label) in enumerate(zip(rankings.tolist(), labels, strict=True)):
        if row in ranking or len(ranking) != len(set(ranking)):
            raise ValueError("candidate ranking score authority differs")
        positives = counts[label] - 1
        found = 0
        terms = []
        for rank, index in enumerate(ranking[:positives], start=1):
            if labels[index] == label:
                found += 1
                terms.append(found / rank)
        average_precisions.append(math.fsum(terms) / positives)
        hits.append(float(labels[ranking[0]] == label))
    return {
        "map_at_r": math.fsum(average_precisions) / len(labels),
        "per_query_ap": average_precisions,
        "r1": math.fsum(hits) / len(labels),
    }


def _candidate_tensors(
    values: torch.Tensor,
    codes: torch.Tensor,
    shortlists: torch.Tensor,
    quantizer: ProductQuantizer,
    statistics: ProductCodeResidualStatistics,
    *,
    block_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    baselines = []
    feature_rows = []
    pairwise_rows = []
    with torch.no_grad():
        for start in range(0, len(values), block_size):
            stop = min(start + block_size, len(values))
            indexes = shortlists[start:stop]
            candidate_codes = codes[indexes]
            observed = statistics.candidate_features(values[start:stop], candidate_codes)
            baseline = -0.5 * aligned_product_squared_distances(
                values[start:stop], candidate_codes, quantizer
            )
            baselines.append(baseline)
            feature_rows.append(observed.concatenated())
            pairwise_rows.append(observed.pairwise_code_similarity)
    return tuple(
        torch.cat(rows, dim=0).contiguous() for rows in (baselines, feature_rows, pairwise_rows)
    )  # type: ignore[return-value]


def _teacher_scores(
    values: torch.Tensor, shortlists: torch.Tensor, *, block_size: int = 1024
) -> torch.Tensor:
    rows = []
    with torch.no_grad():
        for start in range(0, len(values), block_size):
            stop = min(start + block_size, len(values))
            rows.append(
                torch.einsum("bd,bcd->bc", values[start:stop], values[shortlists[start:stop]])
            )
    return torch.cat(rows, dim=0).contiguous()


def _reranked_candidates(
    values: torch.Tensor,
    codes: torch.Tensor,
    quantizer: ProductQuantizer,
    statistics: ProductCodeResidualStatistics,
    model: CandidateSetReranker,
) -> tuple[torch.Tensor, torch.Tensor]:
    shortlists = exact_product_shortlists(
        values, codes, quantizer, width=SHORTLIST_WIDTH, query_block_size=256
    )
    baseline, features, pairwise = _candidate_tensors(
        values, codes, shortlists, quantizer, statistics
    )
    score_blocks = []
    with torch.inference_mode():
        for start in range(0, len(values), 512):
            stop = min(start + 512, len(values))
            score_blocks.append(
                model(baseline[start:stop], features[start:stop], pairwise[start:stop])
            )
        scores = torch.cat(score_blocks, dim=0)
        order = torch.argsort(scores, dim=1, descending=True, stable=True)
    return shortlists.cpu(), torch.gather(shortlists, 1, order).cpu()


def _fit_diagnostics(
    model: CandidateSetReranker | None,
    baseline: torch.Tensor,
    features: torch.Tensor,
    pairwise: torch.Tensor,
    teacher_scores: torch.Tensor,
    *,
    block_size: int = 512,
) -> dict[str, float]:
    """Measure the frozen fit objective without one full-set attention allocation."""

    if type(block_size) is not int or block_size < 1:
        raise ValueError("candidate reranker diagnostic block differs")
    weighted: dict[str, list[float]] = {
        "listwise_kl": [],
        "score_mse": [],
        "total": [],
    }
    entropy_terms = []
    with torch.inference_mode():
        for start in range(0, len(baseline), block_size):
            stop = min(start + block_size, len(baseline))
            scores = (
                baseline[start:stop]
                if model is None
                else model(
                    baseline[start:stop],
                    features[start:stop],
                    pairwise[start:stop],
                )
            )
            loss = candidate_set_distillation_loss(
                scores,
                teacher_scores[start:stop],
                temperature=TEMPERATURE,
                score_weight=SCORE_WEIGHT,
            )
            rows = stop - start
            weighted["listwise_kl"].append(float(loss.listwise_kl) * rows)
            weighted["score_mse"].append(float(loss.score_mse) * rows)
            weighted["total"].append(float(loss.total) * rows)
            probabilities = torch.softmax(teacher_scores[start:stop] / TEMPERATURE, dim=-1)
            entropy_terms.append(
                float(
                    (-(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum(dim=-1)).sum()
                )
            )
    rows = len(baseline)
    return {
        "listwise_kl": math.fsum(weighted["listwise_kl"]) / rows,
        "score_mse": math.fsum(weighted["score_mse"]) / rows,
        "target_effective_support": math.exp(math.fsum(entropy_terms) / rows),
        "total": math.fsum(weighted["total"]) / rows,
    }


def _load_parent_quantizers(
    path: Path,
) -> tuple[ProductQuantizer, ProductQuantizer, OptimizedProductQuantizer]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if type(value) is not dict or set(value) != {
        "codebooks",
        "codebook_size",
        "dimensions",
        "opq24_codebooks",
        "opq24_rotation",
        "pq24_codebooks",
        "pq32_codebooks",
        "stages",
    }:
        raise ValueError("parent codec checkpoint schema differs")
    if (
        value["codebook_size"] != CODEBOOK_SIZE
        or value["dimensions"] != DIMENSIONS
        or value["stages"] != BLOCKS
    ):
        raise ValueError("parent codec checkpoint geometry differs")
    pq24_spec = ProductQuantizationSpec((5,) * 16 + (6,) * 8)
    pq32_spec = ProductQuantizationSpec((4,) * 32)
    pq24 = ProductQuantizer.from_codebooks(pq24_spec, value["pq24_codebooks"])
    pq32 = ProductQuantizer.from_codebooks(pq32_spec, value["pq32_codebooks"])
    opq24 = OptimizedProductQuantizer.from_components(
        pq24_spec, value["opq24_rotation"], value["opq24_codebooks"]
    )
    return pq24, pq32, opq24


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
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--execute-candidate-set-reranking", action="store_true", required=True)
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
                args.driver_sha256,
            )
        )
        or _file_sha256(Path(__file__).resolve()) != args.driver_sha256
        or _file_sha256(args.direct_checkpoint) != args.direct_checkpoint_sha256
        or _file_sha256(args.parent_codec_checkpoint) != args.parent_codec_checkpoint_sha256
        or _file_sha256(args.parent_codec_receipt) != args.parent_codec_receipt_sha256
    ):
        raise ValueError("candidate reranking authority differs")
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
    device = validation.device
    candidate_width = max(Counter(validation_labels).values()) - 1

    stability = []
    seed0_validation_codes: torch.Tensor | None = None
    for seed in PQ_SEEDS:
        fitted_cpu = fit_product_quantizer(
            fit_rows,
            pq24.spec,
            seed=seed,
            maximum_iterations=MAXIMUM_ITERATIONS,
        )
        if seed == 0 and any(
            not torch.equal(left, right)
            for left, right in zip(
                fitted_cpu.detached_codebooks(), pq24.detached_codebooks(), strict=True
            )
        ):
            raise RuntimeError("candidate reranking parent PQ24 codebooks differ")
        fitted = fitted_cpu.to(device)
        fitted_codes = fitted.hard_encode(validation)
        if seed == 0:
            seed0_validation_codes = fitted_codes
        result = score_asymmetric_product(
            validation, fitted_codes, fitted, validation_labels, device=device
        )
        stability.append({"map_at_r": result["map_at_r"], "r1": result["r1"], "seed": seed})

    pq24 = pq24.to(device).eval()
    pq32 = pq32.to(device).eval()
    opq24 = opq24.to(device).eval()
    pq24_codes = pq24.hard_encode(validation)
    if seed0_validation_codes is None or not torch.equal(seed0_validation_codes, pq24_codes):
        raise RuntimeError("candidate reranking parent PQ24 codes differ")
    pq32_codes = pq32.hard_encode(validation)
    opq24_codes = opq24.hard_encode(validation)
    pq24_score = score_asymmetric_product(
        validation, pq24_codes, pq24, validation_labels, device=device
    )
    pq32_score = score_asymmetric_product(
        validation, pq32_codes, pq32, validation_labels, device=device
    )
    opq24_score = score_asymmetric_optimized_product(
        validation, opq24_codes, opq24, validation_labels, device=device
    )
    float_value = score_symmetric(
        validation_rows,
        validation_labels,
        candidate_width=candidate_width,
        device=device,
    )
    float_score = {
        "map_at_r": float(float_value["map_at_r"]),
        "per_query_ap": [float(value) for value in float_value["per_query_ap"]],
        "r1": float(float_value["r1"]),
    }
    baseline_rankings = exact_product_shortlists(
        validation,
        pq24_codes,
        pq24,
        width=SHORTLIST_WIDTH,
        query_block_size=256,
    ).cpu()
    baseline_shortlist_score = score_ranked_candidates(baseline_rankings, validation_labels)
    if baseline_shortlist_score != pq24_score or stability[0] != {
        "map_at_r": pq24_score["map_at_r"],
        "r1": pq24_score["r1"],
        "seed": 0,
    }:
        raise RuntimeError("candidate reranking matched PQ24 control differs")

    pq24_cpu = pq24.cpu()
    fit_codes_cpu = pq24_cpu.hard_encode(fit_rows)
    statistics = ProductCodeResidualStatistics.fit(pq24_cpu, fit_rows, fit_codes_cpu).to(device)
    fit_device = fit_rows.to(device)
    fit_codes = fit_codes_cpu.to(device)
    fit_shortlists = exact_product_shortlists(
        fit_device,
        fit_codes,
        pq24_cpu.to(device),
        width=SHORTLIST_WIDTH,
        query_block_size=256,
    )
    baseline, features, pairwise = _candidate_tensors(
        fit_device, fit_codes, fit_shortlists, pq24, statistics
    )
    teacher_scores = _teacher_scores(fit_device, fit_shortlists)
    started = time.monotonic()
    model = fit_candidate_set_reranker(
        baseline,
        features,
        pairwise,
        teacher_scores,
        blocks=BLOCKS,
        model_dimensions=MODEL_DIMENSIONS,
        heads=HEADS,
        layers=LAYERS,
        seed=args.seed,
        updates=UPDATES,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        temperature=TEMPERATURE,
        score_weight=SCORE_WEIGHT,
    )
    fit_seconds = time.monotonic() - started
    fit_loss_before = _fit_diagnostics(None, baseline, features, pairwise, teacher_scores)
    fit_loss_after = _fit_diagnostics(model, baseline, features, pairwise, teacher_scores)
    if not fit_loss_after["total"] < fit_loss_before["total"]:
        raise RuntimeError("candidate reranker training did not improve the frozen objective")
    del baseline, features, pairwise, teacher_scores, fit_device, fit_codes, fit_shortlists
    validation_statistics = statistics
    pq24 = pq24_cpu.to(device).eval()
    validation_codes = pq24.hard_encode(validation)
    repeated_baseline_rankings, reranked = _reranked_candidates(
        validation, validation_codes, pq24, validation_statistics, model
    )
    if not torch.equal(repeated_baseline_rankings, baseline_rankings):
        raise RuntimeError("candidate reranking shortlist replay differs")
    validation_for_ceiling = validation
    baseline_rankings_device = baseline_rankings.to(device)
    with torch.inference_mode():
        ceiling_scores = torch.einsum(
            "bd,bcd->bc",
            validation_for_ceiling,
            validation_for_ceiling[baseline_rankings_device],
        )
        ceiling_order = torch.argsort(ceiling_scores, dim=1, descending=True, stable=True)
        ceiling_rankings = torch.gather(baseline_rankings_device, 1, ceiling_order).cpu()
    top32_float_ceiling = score_ranked_candidates(ceiling_rankings, validation_labels)
    reranked_score = score_ranked_candidates(reranked, validation_labels)
    seed_maps = [cast(float, row["map_at_r"]) for row in stability]
    seed_standard_deviation = float(
        torch.tensor(seed_maps, dtype=torch.float64).std(unbiased=False)
    )
    decision = candidate_reranking_decision(
        map_at_r=cast(float, reranked_score["map_at_r"]),
        r1=cast(float, reranked_score["r1"]),
        pq24_map_at_r=cast(float, pq24_score["map_at_r"]),
        pq32_map_at_r=cast(float, pq32_score["map_at_r"]),
        pq24_r1=cast(float, pq24_score["r1"]),
        seed_standard_deviation=seed_standard_deviation,
    )
    checkpoint = {
        "model": {name: value.detach().cpu() for name, value in model.state_dict().items()},
        "parent_codec_checkpoint_sha256": args.parent_codec_checkpoint_sha256,
        "recipe": {
            "batch_size": BATCH_SIZE,
            "heads": HEADS,
            "layers": LAYERS,
            "learning_rate": LEARNING_RATE,
            "model_dimensions": MODEL_DIMENSIONS,
            "score_weight": SCORE_WEIGHT,
            "shortlist_width": SHORTLIST_WIDTH,
            "temperature": TEMPERATURE,
            "updates": UPDATES,
        },
        "statistics": {name: value.cpu() for name, value in statistics.state_dict().items()},
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    checkpoint_wire = buffer.getvalue()
    checkpoint_sha256 = hashlib.sha256(checkpoint_wire).hexdigest()
    with publish_bytes_noreplace(
        args.model_output,
        checkpoint_wire,
        validator=lambda wire: (
            None
            if hashlib.sha256(wire).hexdigest() == checkpoint_sha256
            else (_ for _ in ()).throw(ValueError("candidate reranker checkpoint differs"))
        ),
    ):
        pass
    map_gain = cast(float, reranked_score["map_at_r"]) - cast(float, pq24_score["map_at_r"])
    r1_gain = cast(float, reranked_score["r1"]) - cast(float, pq24_score["r1"])
    receipt: dict[str, object] = {
        "claim_eligible": False,
        "dataset": "sop-official-train-class-disjoint-validation",
        "decision": decision,
        "fit_seconds": fit_seconds,
        "fit_teacher_loss": {
            "after": fit_loss_after,
            "before": fit_loss_before,
            "residual_scale": float(model.residual_scale),
        },
        "inputs": {
            "direct_checkpoint_sha256": args.direct_checkpoint_sha256,
            "parent_codec_checkpoint_sha256": args.parent_codec_checkpoint_sha256,
            "parent_codec_receipt_sha256": args.parent_codec_receipt_sha256,
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "matched_controls": {
            "float": float_score,
            "opq24": opq24_score,
            "pq24": pq24_score,
            "pq24_top32": baseline_shortlist_score,
            "pq32": pq32_score,
            "top32_float_ceiling": top32_float_ceiling,
        },
        "model_checkpoint": {"bytes": len(checkpoint_wire), "sha256": checkpoint_sha256},
        "model_parameter_bytes": sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        ),
        "official_test_touched": False,
        "partition": {
            "fit_rows": len(fit_indexes),
            "split_seed": SPLIT_SEED,
            "validation_rows": len(validation_indexes),
        },
        "pq24_seed_stability": {
            "gain_exceeds_one_standard_deviation": map_gain > seed_standard_deviation,
            "gain_in_standard_deviations": (
                map_gain / seed_standard_deviation if seed_standard_deviation > 0 else None
            ),
            "map_at_r_population_standard_deviation": seed_standard_deviation,
            "rows": stability,
        },
        "reranker_seed_stability": {
            "measured": False,
            "reason": "seed-0 kill screen; multi-seed confirmation is required before a claim",
        },
        "recipe": checkpoint["recipe"],
        "reranked": reranked_score,
        "reranked_paired_delta": {"map_at_r": map_gain, "r1": r1_gain},
        "rankings": {
            "baseline": baseline_rankings.tolist(),
            "reranked": reranked.tolist(),
            "top32_float_ceiling": ceiling_rankings.tolist(),
        },
        "runtime": runtime._asdict(),
        "schema": "sfora-pq-candidate-set-reranking-v1",
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
            else (_ for _ in ()).throw(ValueError("candidate reranking receipt differs"))
        ),
    ):
        pass
    print(receipt_wire.decode(), end="")


if __name__ == "__main__":
    main()
