#!/usr/bin/env python3
"""Compare restricted and direct trainable projections on SOP validation."""

from __future__ import annotations

import argparse
import json
import math
import re
import time
from pathlib import Path
from typing import cast

import numpy as np
import positive_coverage_artifacts as artifacts
import run_sop_positive_coverage_metric as coverage
import run_sop_similarity_loss_controls as controls
import torch
from probe_representation_ceiling import load_paired_train_archives
from torch import nn

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.foldable_linear import FoldableLinear
from sfora.representation_ceiling import deterministic_class_partition

_ARM_NAMES = ("restricted_adapter", "factorized_adapter", "direct_projection")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def validate_scientific_projection_shape(
    base_weight: torch.Tensor, base_bias: torch.Tensor
) -> None:
    """Require the preregistered SOP teacher and deployed code widths."""

    if base_weight.shape != (128, 768) or base_bias.shape != (128,):
        raise ValueError("scientific projection shape differs")


def initialize_matched_parameterizations(
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, nn.Module]:
    """Initialize both parameterizations to the same effective affine map."""

    if (
        type(base_weight) is not torch.Tensor
        or type(base_bias) is not torch.Tensor
        or base_weight.dtype != torch.float32
        or base_bias.dtype != torch.float32
        or base_weight.ndim != 2
        or base_bias.ndim != 1
        or base_weight.shape[0] != base_bias.shape[0]
        or not bool(torch.isfinite(base_weight).all())
        or not bool(torch.isfinite(base_bias).all())
    ):
        raise ValueError("base projection authority differs")
    restricted = nn.Linear(base_weight.shape[0], base_weight.shape[0], bias=False, device=device)
    factorized = FoldableLinear(
        input_dim=base_weight.shape[0],
        hidden_dim=3 * base_weight.shape[0],
        output_dim=base_weight.shape[0],
    ).to(device)
    direct = nn.Linear(base_weight.shape[1], base_weight.shape[0], bias=True, device=device)
    with torch.no_grad():
        restricted.weight.copy_(torch.eye(base_weight.shape[0], device=device))
        factorized.initialize_repeated_identity()
        direct.weight.copy_(base_weight.to(device))
        assert direct.bias is not None
        direct.bias.copy_(base_bias.to(device))
    return {
        "restricted_adapter": restricted,
        "factorized_adapter": factorized,
        "direct_projection": direct,
    }


def encode_parameterization(
    name: str,
    model: nn.Module,
    teacher_rows: torch.Tensor,
    base_rows: torch.Tensor,
) -> torch.Tensor:
    """Encode rows through one registered matched parameterization."""

    if name in ("restricted_adapter", "factorized_adapter"):
        source = base_rows
    elif name == "direct_projection":
        source = teacher_rows
    else:
        raise ValueError("parameterization arm differs")
    return torch.nn.functional.normalize(model(source).float(), dim=-1).contiguous()


def fold_restricted_adapter(
    adapter_weight: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold the restricted adapter and frozen base into one affine head."""

    if (
        adapter_weight.ndim != 2
        or base_weight.ndim != 2
        or base_bias.ndim != 1
        or adapter_weight.shape[0] != adapter_weight.shape[1]
        or adapter_weight.shape[1] != base_weight.shape[0]
        or base_bias.shape[0] != base_weight.shape[0]
    ):
        raise ValueError("projection folding authority differs")
    return (
        torch.matmul(adapter_weight, base_weight).contiguous(),
        torch.mv(adapter_weight, base_bias).contiguous(),
    )


def parameterization_state(name: str, model: nn.Linear) -> tuple[dict[str, torch.Tensor], str]:
    """Return a CPU checkpoint and its role-specific parameter digest."""

    weight = model.weight.detach().cpu().float().contiguous()
    if name == "restricted_adapter" and model.bias is None:
        state = {"weight": weight}
        return state, artifacts.linear_weight_sha256(weight)
    if name == "direct_projection" and model.bias is not None:
        bias = model.bias.detach().cpu().float().contiguous()
        state = {"weight": weight, "bias": bias}
        return state, artifacts.affine_parameters_sha256(weight, bias)
    raise ValueError("parameterization arm differs")


def trainable_parameter_count(model: nn.Module) -> int:
    """Count trainable scalar parameters without counting frozen buffers."""

    if not isinstance(model, nn.Module):
        raise ValueError("parameterization arm differs")
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def parameterization_learning_rate(name: str) -> float:
    """Match first-order deployed-map motion across parameter widths."""

    if name == "restricted_adapter":
        return coverage.LEARNING_RATE
    if name in ("factorized_adapter", "direct_projection"):
        return coverage.LEARNING_RATE * math.sqrt(128.0 / 768.0)
    raise ValueError("parameterization arm differs")


def _relative_affine_displacement(
    weight: torch.Tensor,
    bias: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
) -> float:
    numerator = torch.sum((weight.double() - base_weight.double()).square()) + torch.sum(
        (bias.double() - base_bias.double()).square()
    )
    denominator = torch.sum(base_weight.double().square()) + torch.sum(base_bias.double().square())
    value = float(torch.sqrt(numerator / denominator))
    if not math.isfinite(value):
        raise ValueError("deployed displacement differs")
    return value


def projection_decision(
    restricted: dict[str, object],
    factorized: dict[str, object],
    direct: dict[str, object],
    *,
    validation_labels: tuple[int, ...],
) -> dict[str, object]:
    """Apply the preregistered paired capacity-screen decision rule."""

    restricted_score = cast(dict[str, object], restricted["score"])
    factorized_score = cast(dict[str, object], factorized["score"])
    direct_score = cast(dict[str, object], direct["score"])
    restricted_ap = cast(list[float], restricted_score["packed_per_query_ap"])
    direct_ap = cast(list[float], direct_score["packed_per_query_ap"])
    map_gain = cast(float, direct_score["packed_map_at_r"]) - cast(
        float, restricted_score["packed_map_at_r"]
    )
    r1_gain = cast(float, direct_score["packed_r1"]) - cast(float, restricted_score["packed_r1"])
    lower_bound = coverage._class_cluster_lower_bound(direct_ap, restricted_ap, validation_labels)
    factorized_gain = cast(float, factorized_score["packed_map_at_r"]) - cast(
        float, restricted_score["packed_map_at_r"]
    )
    closure_fraction = factorized_gain / map_gain if map_gain > 0.0 else 0.0
    direct_factorized_gain = cast(float, direct_score["packed_map_at_r"]) - cast(
        float, factorized_score["packed_map_at_r"]
    )
    if map_gain < 0.003 or lower_bound <= 0.0 or r1_gain < 0.0:
        classification = "direct-screen-failed"
    elif factorized_gain >= map_gain - 0.002:
        classification = "parameterization-sufficient"
    elif direct_factorized_gain >= 0.005 and closure_fraction < 0.6:
        classification = "mixed-or-information-leading"
    else:
        classification = "mixed"
    return {
        "capacity_control": {
            "classification": classification,
            "closure_fraction": closure_fraction,
            "direct_minus_factorized_packed_map_at_r": direct_factorized_gain,
            "factorized_minus_restricted_packed_map_at_r": factorized_gain,
        },
        "gates": {
            "class_clustered_lower_bound": 0.0,
            "packed_map_at_r_gain": 0.003,
            "packed_r1_gain": 0.0,
        },
        "observed": {
            "class_clustered_lower_bound": lower_bound,
            "packed_map_at_r_gain": map_gain,
            "packed_r1_gain": r1_gain,
        },
        "passed": lower_bound > 0.0 and map_gain >= 0.003 and r1_gain >= 0.0,
    }


def train_parameterization_arm(
    name: str,
    model: nn.Module,
    teacher_fit: torch.Tensor,
    teacher_validation: torch.Tensor,
    base_fit: torch.Tensor,
    base_validation: torch.Tensor,
    base_weight: torch.Tensor,
    base_bias: torch.Tensor,
    fit_labels: tuple[int, ...],
    validation_labels: tuple[int, ...],
    schedule: tuple[np.ndarray, ...],
    frozen_negatives: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Train one matched parameterization with the frozen mean-logit objective."""

    if name not in _ARM_NAMES or frozen_negatives.shape[0] != teacher_fit.shape[0]:
        raise ValueError("parameterization arm differs")
    optimizer = torch.optim.Adam(model.parameters(), lr=parameterization_learning_rate(name))
    teacher_fit_device = teacher_fit.to(device)
    base_fit_device = base_fit.to(device)
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
        anchor_codes = encode_parameterization(
            name, model, teacher_fit_device[anchors], base_fit_device[anchors]
        )
        positive_codes = encode_parameterization(
            name,
            model,
            teacher_fit_device[positives.reshape(-1)],
            base_fit_device[positives.reshape(-1)],
        ).reshape(len(anchors), positives.shape[1], -1)
        negative_codes = encode_parameterization(
            name,
            model,
            teacher_fit_device[mined.reshape(-1)],
            base_fit_device[mined.reshape(-1)],
        ).reshape(len(anchors), frozen_negatives.shape[1], -1)
        positive_similarities = torch.einsum(
            "bd,bpd->bp", anchor_codes, positive_codes
        ).contiguous()
        negative_similarities = torch.einsum(
            "bd,bnd->bn", anchor_codes, negative_codes
        ).contiguous()
        self_similarities = torch.einsum(
            "bd,bd->b", anchor_codes, coverage._unit(base_fit_device[anchors])
        ).contiguous()
        loss = controls.matched_control_loss(
            "mean_logit",
            positive_similarities,
            positive_mask,
            negative_similarities,
            self_similarities,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        validation_codes = encode_parameterization(
            name,
            model,
            teacher_validation.to(device),
            base_validation.to(device),
        ).cpu()
    if name == "factorized_adapter":
        if not isinstance(model, FoldableLinear):
            raise ValueError("parameterization arm differs")
        adapter_weight, adapter_bias = model.fold()
        folded_weight = torch.matmul(adapter_weight.detach().cpu(), base_weight).contiguous()
        folded_bias = (
            torch.mv(adapter_weight.detach().cpu(), base_bias) + adapter_bias.detach().cpu()
        ).contiguous()
        state = {"weight": folded_weight, "bias": folded_bias}
        parameter_sha256 = artifacts.affine_parameters_sha256(
            adapter_weight.detach().cpu(), adapter_bias.detach().cpu()
        )
    else:
        if not isinstance(model, nn.Linear):
            raise ValueError("parameterization arm differs")
        state, parameter_sha256 = parameterization_state(name, model)
    if name == "restricted_adapter":
        folded_weight, folded_bias = fold_restricted_adapter(
            state["weight"], base_weight, base_bias
        )
        deployed_head_sha256 = artifacts.affine_parameters_sha256(folded_weight, folded_bias)
    elif name == "direct_projection":
        folded_weight = state["weight"]
        folded_bias = state["bias"]
        deployed_head_sha256 = artifacts.affine_parameters_sha256(folded_weight, folded_bias)
    else:
        deployed_head_sha256 = artifacts.affine_parameters_sha256(folded_weight, folded_bias)
    with torch.inference_mode():
        deployment_codes = coverage._unit(
            torch.nn.functional.linear(
                teacher_validation.to(device),
                folded_weight.to(device),
                folded_bias.to(device),
            )
        )
        deployment_delta = float(
            torch.max(torch.abs(deployment_codes - validation_codes.to(device))).cpu()
        )
    if not math.isfinite(deployment_delta):
        raise ValueError("folded deployment differs")
    if name == "restricted_adapter":
        state["base_head_weight"] = base_weight.detach().cpu().float().contiguous()
        state["base_head_bias"] = base_bias.detach().cpu().float().contiguous()
        deployment_equivalence: object = {"maximum_absolute_code_delta": deployment_delta}
    elif name == "factorized_adapter":
        deployment_equivalence = {"maximum_absolute_code_delta": deployment_delta}
    else:
        deployment_equivalence = "identical-parameterization"
    result: dict[str, object] = {
        "deployed_relative_frobenius_displacement": _relative_affine_displacement(
            folded_weight, folded_bias, base_weight, base_bias
        ),
        "deployment_equivalence": deployment_equivalence,
        "deployed_head_sha256": deployed_head_sha256,
        "final_loss": losses[-1],
        "mean_last_100_loss": artifacts.mean_recent_loss(losses, window=100),
        "parameter_sha256": parameter_sha256,
        "score": coverage._score(deployment_codes.cpu(), validation_labels, device),
        "trainable_parameters": trainable_parameter_count(model),
    }
    if name == "restricted_adapter":
        result["base_head_sha256"] = artifacts.affine_parameters_sha256(
            state["base_head_weight"], state["base_head_bias"]
        )
    return result, state


def _parse_args() -> argparse.Namespace:
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
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--restricted-checkpoint", type=Path, required=True)
    parser.add_argument("--factorized-checkpoint", type=Path, required=True)
    parser.add_argument("--direct-checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument(
        "--execute-projection-parameterizations", action="store_true", required=True
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if (
        type(args.seed) is not int
        or args.seed < 0
        or _SHA256.fullmatch(args.source_sha256) is None
        or _SHA256.fullmatch(args.teacher_sha256) is None
        or _SHA256.fullmatch(args.base_checkpoint_sha256) is None
    ):
        raise ValueError("experiment authority differs")
    coverage.SEED = args.seed
    controls.SEED = args.seed
    base_parameter_sha256, expected_base_map, expected_base_r1 = controls.validate_base_authority(
        parameter_sha256=args.base_parameter_sha256,
        expected_map=args.expected_base_map,
        expected_r1=args.expected_base_r1,
    )
    paths = artifacts.ProjectionParameterizationArtifactPaths(
        restricted_checkpoint=args.restricted_checkpoint,
        factorized_checkpoint=args.factorized_checkpoint,
        direct_checkpoint=args.direct_checkpoint,
        complete_receipt=args.receipt,
    )
    source_identity = artifacts.positive_coverage_source_identity(
        driver=Path(__file__).resolve(),
        driver_sha256=args.driver_sha256,
        source_revision=args.source_revision,
    )
    runtime = configure_deterministic_similarity_runtime(args.seed, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if artifacts._file_sha256(args.base_checkpoint) != args.base_checkpoint_sha256:
        raise ValueError("base checkpoint digest differs")
    base_receipt = controls.load_base_receipt_authority(
        path=args.base_receipt,
        sha256=args.base_receipt_sha256,
        parameter_sha256=base_parameter_sha256,
        expected_map=expected_base_map,
        expected_r1=expected_base_r1,
        source_snapshot_sha256=args.source_sha256,
        teacher_snapshot_sha256=args.teacher_sha256,
    )
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
    if type(state) is not dict or set(state) != {"weight", "bias"}:
        raise ValueError("base checkpoint schema differs")
    base_weight = cast(torch.Tensor, state["weight"]).float().contiguous()
    base_bias = cast(torch.Tensor, state["bias"]).float().contiguous()
    validate_scientific_projection_shape(base_weight, base_bias)
    if artifacts.affine_parameters_sha256(base_weight, base_bias) != base_parameter_sha256:
        raise ValueError("base checkpoint parameters differ")
    with torch.inference_mode():
        base_fit_raw = torch.nn.functional.linear(
            teacher[fit_indexes], base_weight, base_bias
        ).contiguous()
        base_validation_raw = torch.nn.functional.linear(
            teacher[validation_indexes], base_weight, base_bias
        ).contiguous()
        base_fit = coverage._unit(base_fit_raw)
        base_validation = coverage._unit(base_validation_raw)
    device = torch.device("cuda")
    base_score = coverage._score(base_validation, validation_labels, device)
    observed_map = base_score.get("packed_map_at_r")
    observed_r1 = base_score.get("packed_r1")
    if (
        type(observed_map) is not float
        or type(observed_r1) is not float
        or abs(observed_map - expected_base_map) > 1e-12
        or abs(observed_r1 - expected_base_r1) > 1e-12
    ):
        raise ValueError("base score differs")
    schedule_authority = coverage._schedule(fit_labels)
    schedule = tuple(row.copy() for row in schedule_authority.row_indexes)
    frozen_negatives = controls.frozen_hard_negative_index(
        base_fit,
        fit_labels,
        device=device,
        k=coverage.HARD_NEGATIVES,
        block_size=128,
    )
    models = initialize_matched_parameterizations(base_weight, base_bias, device=device)
    with torch.inference_mode():
        teacher_fit_device = teacher[fit_indexes].to(device)
        base_fit_device = coverage._unit(
            base_fit_raw_device := torch.nn.functional.linear(
                teacher_fit_device, base_weight.to(device), base_bias.to(device)
            ).contiguous()
        )
        initial_delta = float(
            torch.max(
                torch.abs(
                    encode_parameterization(
                        "restricted_adapter",
                        models["restricted_adapter"],
                        teacher_fit_device,
                        base_fit_device,
                    )
                    - encode_parameterization(
                        "direct_projection",
                        models["direct_projection"],
                        teacher_fit_device,
                        base_fit_device,
                    )
                )
            ).cpu()
        )
        factorized_delta = float(
            torch.max(
                torch.abs(
                    encode_parameterization(
                        "restricted_adapter",
                        models["restricted_adapter"],
                        teacher_fit_device,
                        base_fit_device,
                    )
                    - encode_parameterization(
                        "factorized_adapter",
                        models["factorized_adapter"],
                        teacher_fit_device,
                        base_fit_raw_device,
                    )
                )
            ).cpu()
        )
    if (
        not math.isfinite(initial_delta)
        or not math.isfinite(factorized_delta)
        or max(initial_delta, factorized_delta) > 2e-6
        or trainable_parameter_count(models["factorized_adapter"])
        != trainable_parameter_count(models["direct_projection"])
    ):
        raise ValueError("matched initialization differs")
    arms: dict[str, dict[str, object]] = {}
    states: dict[str, dict[str, torch.Tensor]] = {}
    for name in _ARM_NAMES:
        arm_base_fit = base_fit_raw_device if name == "factorized_adapter" else base_fit_device
        arm_base_validation = (
            base_validation_raw if name == "factorized_adapter" else base_validation
        )
        arms[name], states[name] = train_parameterization_arm(
            name,
            models[name],
            teacher_fit_device,
            teacher[validation_indexes],
            arm_base_fit,
            arm_base_validation,
            base_weight,
            base_bias,
            fit_labels,
            validation_labels,
            schedule,
            frozen_negatives,
            device=device,
        )
    decision = projection_decision(
        arms["restricted_adapter"],
        arms["factorized_adapter"],
        arms["direct_projection"],
        validation_labels=validation_labels,
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
        "decision": decision,
        "fitting_rows": len(fit_indexes),
        "initialization": {
            "direct_maximum_absolute_code_delta": initial_delta,
            "factorized_maximum_absolute_code_delta": factorized_delta,
            "maximum_allowed_delta": 2e-6,
        },
        "inputs": {
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "method": {
            "anchor_weight": controls.ANCHOR_WEIGHT,
            "classes_per_update": coverage.CLASS_COUNT,
            "deployment": "single-affine-768-to-128-then-unit-int8",
            "factorized_adapter": {
                "base_input": "raw-affine-128-before-unit-normalization",
                "hidden_dimensions": 384,
                "initialization": "three-balanced-repeated-identity-frames",
                "nonlinearity": "none",
            },
            "hard_negatives": coverage.HARD_NEGATIVES,
            "learning_rates": {name: parameterization_learning_rate(name) for name in _ARM_NAMES},
            "margin": controls.MARGIN,
            "negative_mining": "frozen-base-cached-once",
            "negative_table": {
                "columns": frozen_negatives.shape[1],
                "rows": frozen_negatives.shape[0],
                "sha256": controls.integer_tensor_sha256(frozen_negatives),
            },
            "objective": "mean_logit",
            "optimizer": "adam-constant-no-weight-decay-clip1",
            "schedule_sha256": schedule_authority.sha256,
            "temperature": controls.TEMPERATURE,
            "updates_per_arm": coverage.UPDATES,
            "trainable_parameters": {
                name: trainable_parameter_count(models[name]) for name in _ARM_NAMES
            },
        },
        "official_test_touched": False,
        "partition": {
            "fit_labels_sha256": controls.integer_tensor_sha256(
                torch.tensor(fit_labels, dtype=torch.int64)
            ),
            "fit_rows_sha256": controls.integer_tensor_sha256(
                torch.tensor(fit_indexes, dtype=torch.int64)
            ),
            "validation_labels_sha256": controls.integer_tensor_sha256(
                torch.tensor(validation_labels, dtype=torch.int64)
            ),
            "validation_rows_sha256": controls.integer_tensor_sha256(
                torch.tensor(validation_indexes, dtype=torch.int64)
            ),
        },
        "runtime": runtime._asdict(),
        "schema": "sfora-projection-parameterizations-v2",
        "seed": args.seed,
        "source": source_identity,
        "validation_classes": len(partition.validation_class_ids),
        "validation_rows": len(validation_indexes),
    }
    artifacts.write_projection_parameterization_artifacts(
        states=states, receipt=receipt, paths=paths
    )


if __name__ == "__main__":
    main()
