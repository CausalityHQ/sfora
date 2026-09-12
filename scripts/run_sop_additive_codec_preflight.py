#!/usr/bin/env python3
"""Run the frozen SOP development preflight for a 24-byte additive codec."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import cast

import torch
from positive_coverage_artifacts import positive_coverage_source_identity
from probe_representation_ceiling import load_paired_train_archives
from probe_sop_relational_linear import _lexicographic_candidates, score_symmetric

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.product_quantization import (
    ProductQuantizationSpec,
    ProductQuantizer,
    fit_product_quantizer,
)
from sfora.representation_ceiling import deterministic_class_partition
from sfora.residual_quantization import (
    ResidualQuantizationSpec,
    ResidualQuantizer,
    fit_residual_quantizer,
)

STAGES = 24
CODEBOOK_SIZE = 256
DIMENSIONS = 128
MAXIMUM_ITERATIONS = 25
SPLIT_SEED = 17
TARGET_MAP = 0.58563
_SHA256 = re.compile(r"[0-9a-f]{64}")


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


def _library_sha256(name: str) -> str:
    return _file_sha256(Path(__file__).resolve().parents[1] / "src" / "sfora" / name)


def _score_asymmetric(
    queries: torch.Tensor,
    gallery_codes: torch.Tensor,
    labels: tuple[int, ...],
    score_block: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    *,
    device: torch.device,
) -> dict[str, object]:
    """Evaluate leave-self-out SOP mAP@R from registered asymmetric score blocks."""

    norms = torch.linalg.vector_norm(queries.detach().double(), dim=1)
    if (
        type(queries) is not torch.Tensor
        or queries.device != gallery_codes.device
        or queries.device != device
        or queries.dtype != torch.float32
        or queries.ndim != 2
        or gallery_codes.device != device
        or gallery_codes.dtype != torch.uint8
        or gallery_codes.shape[0] != queries.shape[0]
        or type(labels) is not tuple
        or len(labels) != len(queries)
        or any(type(label) is not int for label in labels)
        or not bool((torch.abs(norms - 1.0) <= 2e-5).all())
    ):
        raise ValueError("additive SOP score authority differs")
    counts = Counter(labels)
    if not counts or min(counts.values()) < 2:
        raise ValueError("additive SOP score authority differs")
    candidate_width = max(counts.values()) - 1
    rankings = []
    with torch.inference_mode():
        for start in range(0, len(queries), 256):
            stop = min(start + 256, len(queries))
            scores = score_block(queries[start:stop], gallery_codes)
            rows = torch.arange(stop - start, device=device)
            columns = torch.arange(start, stop, device=device)
            scores[rows, columns] = -torch.inf
            rankings.append(_lexicographic_candidates(scores, candidate_width).cpu())
    ranked = torch.cat(rankings)
    average_precisions = []
    hits = []
    for ranking, label in zip(ranked.tolist(), labels, strict=True):
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


def score_asymmetric_additive(
    queries: torch.Tensor,
    gallery_codes: torch.Tensor,
    quantizer: ResidualQuantizer,
    labels: tuple[int, ...],
    *,
    device: torch.device,
) -> dict[str, object]:
    """Score exact additive-dot residual codes against float queries."""

    if type(quantizer) is not ResidualQuantizer:
        raise ValueError("additive SOP score authority differs")
    return _score_asymmetric(
        queries,
        gallery_codes,
        labels,
        quantizer.asymmetric_dot_scores,
        device=device,
    )


def score_asymmetric_product(
    queries: torch.Tensor,
    gallery_codes: torch.Tensor,
    quantizer: ProductQuantizer,
    labels: tuple[int, ...],
    *,
    device: torch.device,
) -> dict[str, object]:
    """Score exact negative-squared-distance product codes against float queries."""

    if type(quantizer) is not ProductQuantizer:
        raise ValueError("product SOP score authority differs")
    return _score_asymmetric(
        queries,
        gallery_codes,
        labels,
        lambda query, codes: -quantizer.asymmetric_squared_distances(query, codes),
        device=device,
    )


def additive_preflight_decision(
    *,
    map_at_r: float,
    r1: float,
    pq32_map_at_r: float,
    pq32_r1: float,
) -> dict[str, object]:
    """Classify one frozen additive-codec screen without selecting a new recipe."""

    if (
        type(map_at_r) is not float
        or type(r1) is not float
        or type(pq32_map_at_r) is not float
        or type(pq32_r1) is not float
        or not math.isfinite(map_at_r)
        or not math.isfinite(r1)
        or not math.isfinite(pq32_map_at_r)
        or not math.isfinite(pq32_r1)
        or not 0.0 <= map_at_r <= 1.0
        or not 0.0 <= r1 <= 1.0
        or not 0.0 <= pq32_map_at_r <= 1.0
        or not 0.0 <= pq32_r1 <= 1.0
    ):
        raise ValueError("additive preflight result differs")
    if map_at_r < pq32_map_at_r:
        classification = "greedy-residual-failed"
    elif map_at_r >= TARGET_MAP and r1 >= pq32_r1:
        classification = "greedy-residual-target-passed"
    else:
        classification = "greedy-residual-viable"
    return {
        "classification": classification,
        "gates": {
            "match_pq32_map_at_r": pq32_map_at_r,
            "match_pq32_r1": pq32_r1,
            "recover_0_015_map_at_r": TARGET_MAP,
        },
        "passed": classification == "greedy-residual-target-passed",
    }


def validate_output_paths(codebook_output: Path, receipt: Path) -> None:
    """Reject aliases and invalid no-clobber destinations before expensive fitting."""

    if (
        not isinstance(codebook_output, Path)
        or not isinstance(receipt, Path)
        or not codebook_output.is_absolute()
        or not receipt.is_absolute()
        or not codebook_output.parent.is_dir()
        or not receipt.parent.is_dir()
        or codebook_output.exists()
        or codebook_output.is_symlink()
        or receipt.exists()
        or receipt.is_symlink()
        or codebook_output == receipt
    ):
        raise ValueError("additive preflight output authority differs")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-checkpoint-sha256", required=True)
    parser.add_argument("--codebook-output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--execute-additive-codec-preflight", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    outputs = (args.codebook_output, args.receipt)
    validate_output_paths(*outputs)
    if (
        type(args.seed) is not int
        or args.seed != 0
        or any(
            _SHA256.fullmatch(value) is None
            for value in (
                args.source_sha256,
                args.teacher_sha256,
                args.direct_checkpoint_sha256,
                args.driver_sha256,
            )
        )
        or _file_sha256(Path(__file__).resolve()) != args.driver_sha256
        or _file_sha256(args.direct_checkpoint) != args.direct_checkpoint_sha256
    ):
        raise ValueError("additive preflight authority differs")
    source_identity = positive_coverage_source_identity(
        driver=Path(__file__).resolve(),
        driver_sha256=args.driver_sha256,
        source_revision=args.source_revision,
    )
    runtime = configure_deterministic_similarity_runtime(args.seed, cpu_threads=1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
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
    raw_weight = state["weight"]
    raw_bias = state["bias"]
    if (
        type(raw_weight) is not torch.Tensor
        or type(raw_bias) is not torch.Tensor
        or raw_weight.dtype != torch.float32
        or raw_bias.dtype != torch.float32
        or raw_weight.shape != (DIMENSIONS, 768)
        or raw_bias.shape != (DIMENSIONS,)
        or not bool(torch.isfinite(raw_weight).all())
        or not bool(torch.isfinite(raw_bias).all())
    ):
        raise ValueError("direct checkpoint geometry differs")
    weight = raw_weight.contiguous()
    bias = raw_bias.contiguous()
    teacher = torch.nn.functional.normalize(pair["teacher_train"].float(), dim=1)
    with torch.inference_mode():
        embeddings = torch.nn.functional.normalize(
            torch.nn.functional.linear(teacher, weight, bias), dim=1
        ).contiguous()
    spec = ResidualQuantizationSpec(
        dimensions=DIMENSIONS, stages=STAGES, codebook_size=CODEBOOK_SIZE
    )
    pq24_spec = ProductQuantizationSpec(block_dimensions=(5,) * 16 + (6,) * 8)
    pq32_spec = ProductQuantizationSpec(block_dimensions=(4,) * 32)
    fit_rows = embeddings[fit_indexes].contiguous()
    started = time.monotonic()
    pq24 = fit_product_quantizer(
        fit_rows, pq24_spec, seed=args.seed, maximum_iterations=MAXIMUM_ITERATIONS
    )
    pq32 = fit_product_quantizer(
        fit_rows, pq32_spec, seed=args.seed, maximum_iterations=MAXIMUM_ITERATIONS
    )
    quantizer = fit_residual_quantizer(
        fit_rows,
        spec,
        seed=args.seed,
        maximum_iterations=MAXIMUM_ITERATIONS,
    )
    fit_seconds = time.monotonic() - started
    device = torch.device("cuda")
    pq24 = pq24.to(device).eval()
    pq32 = pq32.to(device).eval()
    quantizer = quantizer.to(device).eval()
    validation = embeddings[validation_indexes].to(device)
    device = validation.device
    with torch.inference_mode():
        candidate_width = max(Counter(validation_labels).values()) - 1
        float_score_value = score_symmetric(
            validation.cpu(),
            validation_labels,
            candidate_width=candidate_width,
            device=device,
        )
        float_score = {
            "map_at_r": float(float_score_value["map_at_r"]),
            "per_query_ap": [float(value) for value in float_score_value["per_query_ap"]],
            "r1": float(float_score_value["r1"]),
        }
        codes = quantizer.hard_encode(validation)
        score = score_asymmetric_additive(
            validation, codes, quantizer, validation_labels, device=device
        )
        pq24_codes = pq24.hard_encode(validation)
        pq24_score = score_asymmetric_product(
            validation, pq24_codes, pq24, validation_labels, device=device
        )
        pq32_codes = pq32.hard_encode(validation)
        pq32_score = score_asymmetric_product(
            validation, pq32_codes, pq32, validation_labels, device=device
        )
        restored = quantizer.hard_decode(codes)
        relative_squared_error = float(
            (validation - restored).square().sum() / validation.square().sum()
        )
        restored_norms = torch.linalg.vector_norm(restored, dim=1)
        fit_codes = quantizer.hard_encode(fit_rows.to(device))
        usage = [int(torch.unique(fit_codes[:, stage]).numel()) for stage in range(STAGES)]
    decision = additive_preflight_decision(
        map_at_r=cast(float, score["map_at_r"]),
        r1=cast(float, score["r1"]),
        pq32_map_at_r=cast(float, pq32_score["map_at_r"]),
        pq32_r1=cast(float, pq32_score["r1"]),
    )
    checkpoint_state = {
        "codebooks": quantizer.codebooks.detach().cpu().contiguous(),
        "codebook_size": CODEBOOK_SIZE,
        "dimensions": DIMENSIONS,
        "stages": STAGES,
        "pq24_codebooks": pq24.detached_codebooks(),
        "pq32_codebooks": pq32.detached_codebooks(),
    }
    checkpoint_buffer = io.BytesIO()
    torch.save(checkpoint_state, checkpoint_buffer)
    checkpoint_wire = checkpoint_buffer.getvalue()
    checkpoint_sha256 = hashlib.sha256(checkpoint_wire).hexdigest()
    with publish_bytes_noreplace(
        args.codebook_output,
        checkpoint_wire,
        validator=lambda wire: (
            None
            if hashlib.sha256(wire).hexdigest() == checkpoint_sha256
            else (_ for _ in ()).throw(ValueError("checkpoint digest differs"))
        ),
    ):
        pass
    receipt: dict[str, object] = {
        "claim_eligible": False,
        "codec": {
            "bytes_per_vector": STAGES,
            "codebook_size": CODEBOOK_SIZE,
            "dimensions": DIMENSIONS,
            "encoding": "greedy-residual",
            "maximum_lloyd_iterations": MAXIMUM_ITERATIONS,
            "score": "float-query-additive-dot",
            "stages": STAGES,
        },
        "codebook_checkpoint": {
            "bytes": len(checkpoint_wire),
            "sha256": checkpoint_sha256,
        },
        "dataset": "sop-official-train-class-disjoint-validation",
        "decision": decision,
        "matched_controls": {
            "float": float_score,
            "pq24": pq24_score,
            "pq32": pq32_score,
        },
        "fit_seconds": fit_seconds,
        "inputs": {
            "direct_checkpoint_sha256": args.direct_checkpoint_sha256,
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "official_test_touched": False,
        "implementation": {
            "product_quantization_sha256": _library_sha256("product_quantization.py"),
            "residual_quantization_sha256": _library_sha256("residual_quantization.py"),
        },
        "partition": {
            "fit_fraction": 0.8,
            "fit_rows": len(fit_indexes),
            "split_seed": SPLIT_SEED,
            "validation_rows": len(validation_indexes),
        },
        "relative_validation_squared_error": relative_squared_error,
        "restored_norms": {
            "maximum": float(restored_norms.max()),
            "mean": float(restored_norms.mean()),
            "minimum": float(restored_norms.min()),
            "standard_deviation": float(restored_norms.std(unbiased=False)),
        },
        "runtime": runtime._asdict(),
        "schema": "sfora-additive-codec-preflight-v1",
        "score": score,
        "seed": args.seed,
        "source": source_identity,
        "used_codewords_per_stage": usage,
    }
    receipt_wire = _canonical_json(receipt)
    with publish_bytes_noreplace(
        args.receipt,
        receipt_wire,
        validator=lambda wire: (
            None
            if wire == receipt_wire
            else (_ for _ in ()).throw(ValueError("receipt bytes differ"))
        ),
    ):
        pass
    print(receipt_wire.decode(), end="")


if __name__ == "__main__":
    main()
