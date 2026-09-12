#!/usr/bin/env python3
"""Run the frozen single-stage differential-PQ development screen."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import time
from pathlib import Path
from typing import cast

import torch
from positive_coverage_artifacts import positive_coverage_source_identity
from probe_representation_ceiling import class_cluster_lower_bound, load_paired_train_archives
from run_sop_additive_codec_preflight import score_asymmetric_product
from run_sop_pq_candidate_reranking import (
    _file_sha256,
    _load_parent_quantizers,
    validate_parent_receipt,
)
from run_sop_pq_lookup_distillation import (
    _concrete_tensor_device,
    _exact_float_shortlists,
    build_label_free_candidate_pool,
)

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.joint_pq_projection import (
    JointPqProjection,
    JointPqTrainingSpec,
    fit_joint_pq_projection,
)
from sfora.pq_candidate_reranking import exact_product_shortlists
from sfora.product_quantization import ProductQuantizer
from sfora.representation_ceiling import deterministic_class_partition

JOINT_PQ_ARM_NAMES = (
    "restricted_rank",
    "restricted_differential",
    "full_rank",
    "full_differential",
)
PQ32_MAP_AT_R = 0.5789231844
PQ32_R1 = 0.8241748966
KILL_MAP_AT_R = 0.5771
GO_MAP_AT_R = 0.5830
STRONG_GO_MAP_AT_R = 0.5875
STRONG_GO_R1 = 0.8310
MINIMUM_R1 = PQ32_R1 - 0.001
SPLIT_SEED = 17
CANDIDATE_WIDTH = 128
TEACHER_WIDTH = 64
COMPRESSED_WIDTH = 32
UNIFORM_WIDTH = 32
QUERY_BLOCK_SIZE = 128
BOOTSTRAP_SAMPLES = 100_000
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _clone_quantizer(quantizer: ProductQuantizer, device: torch.device) -> ProductQuantizer:
    return ProductQuantizer.from_codebooks(
        quantizer.spec,
        tuple(codebook.to(device) for codebook in quantizer.detached_codebooks()),
    )


def projection_row_space_basis(weight: torch.Tensor) -> torch.Tensor:
    """Return an orthonormal basis for the incumbent projection's row space."""

    if (
        type(weight) is not torch.Tensor
        or weight.dtype != torch.float32
        or weight.ndim != 2
        or weight.shape[0] >= weight.shape[1]
        or not bool(torch.isfinite(weight).all())
    ):
        raise ValueError("projection row-space authority differs")
    try:
        basis = torch.linalg.qr(weight.double().T, mode="reduced")[0].T.float().contiguous()
    except RuntimeError as error:
        raise ValueError("projection row-space authority differs") from error
    identity = torch.eye(weight.shape[0], dtype=torch.float32, device=weight.device)
    if not bool(torch.isfinite(basis).all()) or not torch.allclose(
        basis @ basis.T, identity, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("projection row-space authority differs")
    return cast(torch.Tensor, basis)


def initialize_joint_pq_arms(
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
    quantizer: ProductQuantizer,
    *,
    row_space_basis: torch.Tensor,
    device: torch.device,
) -> dict[str, JointPqProjection]:
    """Create four independent arms with deployment-identical step-zero outputs."""

    if (
        type(base_weight) is not torch.Tensor
        or type(base_bias) is not torch.Tensor
        or base_weight.dtype != torch.float32
        or base_bias.dtype != torch.float32
        or base_weight.ndim != 2
        or base_bias.shape != (base_weight.shape[0],)
        or base_weight.shape[0] != quantizer.spec.dimensions
        or not bool(torch.isfinite(base_weight).all())
        or not bool(torch.isfinite(base_bias).all())
        or type(row_space_basis) is not torch.Tensor
        or row_space_basis.dtype != torch.float32
        or row_space_basis.shape != base_weight.shape
        or not isinstance(device, torch.device)
    ):
        raise ValueError("joint PQ initialization authority differs")
    full_weight = base_weight.to(device)
    full_bias = base_bias.to(device)
    result: dict[str, JointPqProjection] = {}
    for name in JOINT_PQ_ARM_NAMES:
        result[name] = JointPqProjection.from_components(
            input_dimensions=base_weight.shape[1],
            initial_weight=full_weight,
            initial_bias=full_bias,
            quantizer=_clone_quantizer(quantizer, device),
            row_space_basis=(
                row_space_basis.to(device) if name.startswith("restricted_") else None
            ),
        )
    return result


def joint_pq_deployment_contract() -> dict[str, object]:
    """Return the immutable inference representation and arithmetic budget."""

    return {
        "bytes_per_vector": 24,
        "codebook_entries": 256,
        "database_sidecar_bytes_per_vector": 0,
        "distance_accumulations_per_candidate": 24,
        "lookup_tables_per_query": 24,
        "stored_code_dtype": "uint8",
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    if type(value) is not torch.Tensor or value.layout != torch.strided:
        raise ValueError("joint PQ tensor authority differs")
    wire = value.detach().cpu().contiguous().numpy().tobytes()
    return hashlib.sha256(wire).hexdigest()


def _code_diagnostics(codes: torch.Tensor) -> dict[str, object]:
    if type(codes) is not torch.Tensor or codes.dtype != torch.uint8 or codes.ndim != 2:
        raise ValueError("joint PQ code authority differs")
    used: list[int] = []
    entropy_bits: list[float] = []
    for stage in range(codes.shape[1]):
        counts = torch.bincount(codes[:, stage].long(), minlength=256).double()
        nonzero = counts[counts > 0]
        probabilities = nonzero / nonzero.sum()
        used.append(len(nonzero))
        entropy_bits.append(float(-(probabilities * torch.log2(probabilities)).sum()))
    return {"entropy_bits_by_stage": entropy_bits, "used_codewords_by_stage": used}


def _validate_arm_qualities(value: object) -> dict[str, dict[str, float]]:
    if type(value) is not dict or set(value) != set(JOINT_PQ_ARM_NAMES):
        raise ValueError("joint PQ quality authority differs")
    for quality in value.values():
        if type(quality) is not dict or set(quality) != {
            "map_at_r",
            "pq32_paired_lower",
            "r1",
        }:
            raise ValueError("joint PQ quality authority differs")
        if (
            any(
                type(quality[key]) is not float
                or not math.isfinite(quality[key])
                or not 0.0 <= quality[key] <= 1.0
                for key in ("map_at_r", "r1")
            )
            or type(quality["pq32_paired_lower"]) is not float
            or not math.isfinite(quality["pq32_paired_lower"])
            or not -1.0 <= quality["pq32_paired_lower"] <= 1.0
        ):
            raise ValueError("joint PQ quality authority differs")
    return value


def joint_pq_decision(arm_qualities: object) -> dict[str, object]:
    """Classify the frozen four-arm seed-zero screen."""

    qualities = _validate_arm_qualities(arm_qualities)
    selected = sorted(
        JOINT_PQ_ARM_NAMES,
        key=lambda name: (-qualities[name]["map_at_r"], name),
    )[0]
    primary = qualities["full_differential"]
    if all(qualities[name]["map_at_r"] <= KILL_MAP_AT_R for name in JOINT_PQ_ARM_NAMES):
        classification = "kill"
    elif (
        primary["map_at_r"] >= STRONG_GO_MAP_AT_R
        and primary["r1"] >= STRONG_GO_R1
        and primary["pq32_paired_lower"] > 0.0
    ):
        classification = "strong-go"
    elif (
        primary["map_at_r"] >= GO_MAP_AT_R
        and primary["r1"] >= MINIMUM_R1
        and primary["pq32_paired_lower"] > 0.0
    ):
        classification = "go"
    elif primary["map_at_r"] >= GO_MAP_AT_R:
        classification = "positive-not-significant"
    else:
        classification = "ambiguous"
    return {
        "classification": classification,
        "differential_mechanism_supported": (
            qualities["full_differential"]["map_at_r"] - qualities["full_rank"]["map_at_r"] >= 0.002
        ),
        "passed": classification in {"go", "strong-go"},
        "primary_arm": "full_differential",
        "selected_arm": selected,
        "single_stage_mechanism_supported": (
            qualities["full_rank"]["map_at_r"] - qualities["restricted_rank"]["map_at_r"] >= 0.004
        ),
    }


def canonical_joint_pq_receipt(value: object) -> bytes:
    """Validate and encode the claim-ineligible scientific receipt."""

    if (
        type(value) is not dict
        or set(value)
        != {
            "arms",
            "claim_eligible",
            "dataset",
            "decision",
            "deployment",
            "inputs",
            "model_checkpoint",
            "official_test_touched",
            "partition",
            "recipe",
            "runtime",
            "schema",
            "seed",
            "source",
        }
        or value.get("schema") != "sfora-single-stage-differential-pq-v1"
        or value.get("claim_eligible") is not False
        or value.get("official_test_touched") is not False
        or value.get("deployment") != joint_pq_deployment_contract()
        or type(value.get("arms")) is not dict
        or set(value["arms"]) != set(JOINT_PQ_ARM_NAMES)
        or type(value.get("decision")) is not dict
        or type(value.get("seed")) is not int
        or value["seed"] < 0
    ):
        raise ValueError("joint PQ receipt authority differs")
    try:
        encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("joint PQ receipt authority differs") from error
    return (encoded + "\n").encode()


def _checkpoint_bytes(arms: dict[str, JointPqProjection]) -> bytes:
    value = {
        "arms": {
            name: {
                "bias": model.projection.bias.detach().cpu().float().contiguous(),
                "codebooks": model.quantizer.detached_codebooks(),
                "input_dimensions": model.projection.in_features,
                "weight": model.effective_weight().detach().cpu().float().contiguous(),
            }
            for name, model in arms.items()
        },
        "deployment": joint_pq_deployment_contract(),
        "schema": "sfora-single-stage-differential-pq-model-v1",
    }
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


def _publish_exact(path: Path, wire: bytes, *, role: str) -> None:
    if path.is_symlink() or (path.exists() and (not path.is_file() or path.read_bytes() != wire)):
        raise FileExistsError(f"{role} already exists")
    if path.exists():
        return
    digest = hashlib.sha256(wire).hexdigest()
    with publish_bytes_noreplace(
        path,
        wire,
        validator=lambda observed: (
            None
            if hashlib.sha256(observed).hexdigest() == digest
            else (_ for _ in ()).throw(ValueError(f"{role} differs"))
        ),
    ):
        pass


def parse_arguments(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit-only local scientific command line."""

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
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--model-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument(
        "--execute-single-stage-differential-pq",
        action="store_true",
        required=True,
    )
    return parser.parse_args(arguments)


def main() -> None:
    """Authenticate, fit, score, and publish one frozen scientific screen."""

    args = parse_arguments()
    paths = (
        args.source_snapshot,
        args.teacher_snapshot,
        args.direct_checkpoint,
        args.parent_codec_checkpoint,
        args.parent_codec_receipt,
    )
    digests = (
        args.source_sha256,
        args.teacher_sha256,
        args.direct_checkpoint_sha256,
        args.parent_codec_checkpoint_sha256,
        args.parent_codec_receipt_sha256,
        args.driver_sha256,
    )
    if (
        any(not path.is_absolute() or not path.is_file() or path.is_symlink() for path in paths)
        or any(_SHA256.fullmatch(value) is None for value in digests)
        or args.receipt.resolve() == args.model_output.resolve()
        or args.receipt.is_symlink()
        or args.model_output.is_symlink()
        or _file_sha256(Path(__file__).resolve()) != args.driver_sha256
        or any(
            _file_sha256(path) != digest for path, digest in zip(paths, digests[:5], strict=True)
        )
    ):
        raise ValueError("single-stage differential PQ authority differs")
    parent_wire = args.parent_codec_receipt.read_bytes()
    parent_receipt = json.loads(parent_wire)
    if (
        type(parent_receipt) is not dict
        or (json.dumps(parent_receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
        != parent_wire
    ):
        raise ValueError("parent codec receipt authority differs")
    validate_parent_receipt(
        parent_receipt,
        parent_checkpoint_sha256=args.parent_codec_checkpoint_sha256,
        direct_checkpoint_sha256=args.direct_checkpoint_sha256,
        source_snapshot_sha256=args.source_sha256,
        teacher_snapshot_sha256=args.teacher_sha256,
    )
    source_identity = positive_coverage_source_identity(
        driver=Path(__file__).resolve(),
        driver_sha256=args.driver_sha256,
        source_revision=args.source_revision,
    )
    runtime = configure_deterministic_similarity_runtime(args.seed, cpu_threads=1)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
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
    direct = torch.load(args.direct_checkpoint, map_location="cpu", weights_only=True)
    if type(direct) is not dict or set(direct) != {"bias", "weight"}:
        raise ValueError("direct checkpoint schema differs")
    weight = cast(torch.Tensor, direct["weight"])
    bias = cast(torch.Tensor, direct["bias"])
    if (
        type(weight) is not torch.Tensor
        or type(bias) is not torch.Tensor
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or weight.shape != (128, 768)
        or bias.shape != (128,)
        or not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("direct checkpoint geometry differs")
    teacher = torch.nn.functional.normalize(pair["teacher_train"].float(), dim=1)
    fit_teacher = teacher[fit_indexes].to(device)
    validation_teacher = teacher[validation_indexes].to(device)
    with torch.inference_mode():
        fit_incumbent = torch.nn.functional.normalize(
            torch.nn.functional.linear(fit_teacher, weight.to(device), bias.to(device)), dim=1
        ).contiguous()
        validation_incumbent = torch.nn.functional.normalize(
            torch.nn.functional.linear(validation_teacher, weight.to(device), bias.to(device)),
            dim=1,
        ).contiguous()
    device = _concrete_tensor_device(validation_teacher, expected_type="cuda")
    pq24, pq32, _opq24 = _load_parent_quantizers(args.parent_codec_checkpoint)
    pq24 = pq24.to(device).eval()
    pq32 = pq32.to(device).eval()
    if pq24.spec.bytes_per_vector != 24 or pq24.spec.codebook_size != 256:
        raise ValueError("parent PQ24 geometry differs")
    with torch.inference_mode():
        teacher_rankings = _exact_float_shortlists(
            fit_teacher,
            width=TEACHER_WIDTH,
            query_block_size=QUERY_BLOCK_SIZE,
        )
        incumbent_codes = pq24.hard_encode(fit_incumbent)
        compressed_rankings = exact_product_shortlists(
            fit_incumbent,
            incumbent_codes,
            pq24,
            width=TEACHER_WIDTH + COMPRESSED_WIDTH,
            query_block_size=QUERY_BLOCK_SIZE,
        )
        candidate_indexes = build_label_free_candidate_pool(
            teacher_rankings,
            compressed_rankings,
            compressed_width=COMPRESSED_WIDTH,
            uniform_width=UNIFORM_WIDTH,
            uniform_seed=args.seed,
        )
        if candidate_indexes.shape != (len(fit_indexes), CANDIDATE_WIDTH):
            raise RuntimeError("joint PQ candidate pool is incomplete")
    neighbor_pairs = torch.combinations(torch.arange(8, device=device), r=2)
    row_space_basis = projection_row_space_basis(weight.to(device))
    arms = initialize_joint_pq_arms(
        weight, bias, pq24, row_space_basis=row_space_basis, device=device
    )
    arm_results: dict[str, object] = {}
    qualities: dict[str, dict[str, float]] = {}
    validation_pq32_codes = pq32.hard_encode(validation_incumbent)
    pq32_score = score_asymmetric_product(
        validation_incumbent,
        validation_pq32_codes,
        pq32,
        validation_labels,
        device=device,
    )
    if (
        abs(cast(float, pq32_score["map_at_r"]) - PQ32_MAP_AT_R) > 1e-9
        or abs(cast(float, pq32_score["r1"]) - PQ32_R1) > 1e-9
    ):
        raise ValueError("PQ32 frozen control differs")
    step_zero_codes: dict[str, torch.Tensor] = {}
    step_zero_rows: dict[str, torch.Tensor] = {}
    with torch.inference_mode():
        for name, model in arms.items():
            step_zero_rows[name] = model.project(fit_teacher)
            step_zero_codes[name] = model.quantizer.hard_encode(step_zero_rows[name])
    baseline_step_zero = step_zero_codes[JOINT_PQ_ARM_NAMES[0]]
    if any(
        not torch.equal(baseline_step_zero, step_zero_codes[name]) for name in JOINT_PQ_ARM_NAMES
    ):
        raise ValueError("joint PQ step-zero codes differ")
    step_zero_evidence = {
        "code_sha256_by_arm": {
            name: _tensor_sha256(step_zero_codes[name]) for name in JOINT_PQ_ARM_NAMES
        },
        "maximum_row_delta_by_arm": {
            name: float(torch.max(torch.abs(step_zero_rows[name] - fit_incumbent)))
            for name in JOINT_PQ_ARM_NAMES
        },
    }
    for name, model in arms.items():
        inputs = fit_teacher
        differential_weight = 0.1 if name.endswith("_differential") else 0.0
        training_spec = JointPqTrainingSpec(
            updates=1_000,
            batch_size=256,
            projection_learning_rate=3e-4,
            codebook_learning_rate=1e-3,
            weight_decay=1e-4,
            temperature=0.05,
            float_weight=0.25,
            reconstruction_weight=0.1,
            differential_weight=differential_weight,
            gradient_norm_cap=5.0,
            seed=args.seed,
        )
        started = time.monotonic()
        fitted = fit_joint_pq_projection(
            model,
            inputs=inputs,
            teacher_values=fit_teacher,
            candidate_indexes=candidate_indexes,
            neighbor_pairs=neighbor_pairs,
            spec=training_spec,
        )
        with torch.inference_mode():
            validation_rows = fitted.model.project(validation_teacher)
            codes = fitted.model.quantizer.hard_encode(validation_rows)
            fit_codes = fitted.model.quantizer.hard_encode(fitted.model.project(inputs))
        quality = score_asymmetric_product(
            validation_rows,
            codes,
            fitted.model.quantizer,
            validation_labels,
            device=device,
        )
        paired_lower = class_cluster_lower_bound(
            tuple(float(value) for value in cast(list[float], quality["per_query_ap"])),
            tuple(float(value) for value in cast(list[float], pq32_score["per_query_ap"])),
            validation_labels,
            seed=args.seed,
            samples=BOOTSTRAP_SAMPLES,
        )
        qualities[name] = {
            "map_at_r": cast(float, quality["map_at_r"]),
            "pq32_paired_lower": paired_lower,
            "r1": cast(float, quality["r1"]),
        }
        arm_results[name] = {
            "code_churn_ppm": int(
                round(
                    float((fit_codes != step_zero_codes[name]).any(dim=1).float().mean())
                    * 1_000_000
                )
            ),
            "differential_weight": differential_weight,
            "elapsed_seconds": time.monotonic() - started,
            "final_code_diagnostics": _code_diagnostics(fit_codes),
            "losses": list(fitted.losses),
            "objective_trace": [
                {
                    "adc_kl": point.adc_kl,
                    "differential": point.differential,
                    "float_kl": point.float_kl,
                    "reconstruction": point.reconstruction,
                    "total": point.total,
                }
                for point in fitted.trace
            ],
            "quality": quality,
        }
    decision = joint_pq_decision(qualities)
    checkpoint_wire = _checkpoint_bytes(arms)
    checkpoint_sha256 = hashlib.sha256(checkpoint_wire).hexdigest()
    receipt: dict[str, object] = {
        "arms": arm_results,
        "claim_eligible": False,
        "dataset": "sop-official-train-class-disjoint-validation",
        "decision": {
            **decision,
            "pq32_map_at_r": cast(float, pq32_score["map_at_r"]),
            "pq32_r1": cast(float, pq32_score["r1"]),
            "quality_inputs": qualities,
        },
        "deployment": joint_pq_deployment_contract(),
        "inputs": {
            "direct_checkpoint_sha256": args.direct_checkpoint_sha256,
            "parent_codec_checkpoint_sha256": args.parent_codec_checkpoint_sha256,
            "parent_codec_receipt_sha256": args.parent_codec_receipt_sha256,
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "model_checkpoint": {"bytes": len(checkpoint_wire), "sha256": checkpoint_sha256},
        "official_test_touched": False,
        "partition": {
            "fit_rows": len(fit_indexes),
            "split_seed": SPLIT_SEED,
            "validation_rows": len(validation_indexes),
        },
        "recipe": {
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "candidate_width": CANDIDATE_WIDTH,
            "candidate_indexes_sha256": _tensor_sha256(candidate_indexes),
            "compressed_width": COMPRESSED_WIDTH,
            "teacher_width": TEACHER_WIDTH,
            "uniform_width": UNIFORM_WIDTH,
            "updates": 1_000,
            "neighbor_pairs_sha256": _tensor_sha256(neighbor_pairs),
            "step_zero": step_zero_evidence,
        },
        "runtime": runtime._asdict(),
        "schema": "sfora-single-stage-differential-pq-v1",
        "seed": args.seed,
        "source": source_identity,
    }
    receipt_wire = canonical_joint_pq_receipt(receipt)
    if (
        positive_coverage_source_identity(
            driver=Path(__file__).resolve(),
            driver_sha256=args.driver_sha256,
            source_revision=args.source_revision,
        )
        != source_identity
    ):
        raise ValueError("source identity authority differs")
    _publish_exact(args.model_output, checkpoint_wire, role="joint PQ model")
    _publish_exact(args.receipt, receipt_wire, role="joint PQ receipt")
    print(receipt_wire.decode(), end="")


if __name__ == "__main__":
    main()
