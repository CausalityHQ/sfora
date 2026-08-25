#!/usr/bin/env python3
"""Build the frozen seed-0 decision for the UniCOM full-width objective."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

STEP_TIME_METRIC = "step_wall"
ARMS = ("sampled_512", "full_768")
EPOCHS = (4, 8, 12, 16)


INPUT_KEYS = (
    "run_config",
    "pair_inventory",
    "pair_result",
    "profile_comparison",
    "control_receipt",
    "candidate_receipt",
)
INPUT_BINDING_KEYS = ("path", "sha256", "bytes")
TOP_KEYS = (
    "schema_version",
    "inputs",
    "metric_authority",
    "evidence",
    "decision",
    "status",
)
METRIC_AUTHORITY_KEYS = ("step_time", "memory", "checkpoint_bytes")
EVIDENCE_KEYS = (
    "primary_map_delta",
    "control_top1_count",
    "candidate_top1_count",
    "candidate_primary_by_epoch",
    "control_epoch16_primary",
    "abba_step_time_ratio",
    "observed_cuda_step_ratio",
    "peak_allocated_ratio",
    "peak_reserved_ratio",
    "checkpoint_bytes_equal",
    "control_checkpoint_bytes",
    "candidate_checkpoint_bytes",
)


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py").resolve()
    spec = importlib.util.spec_from_file_location(f"_{name}_for_decision", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError(f"loaded {name} source differs")
    return module


EVALUATOR = _load_sibling("evaluate_unicom_full_width_objective")
COMPARATOR = _load_sibling("compare_unicom_full_width_profiles")
TRAINER = _load_sibling("train_unicom_inshop")


def _finite_float(value: object, name: str, *, positive: bool = False) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError(f"{name} must be a finite builtin float")
    if positive and value <= 0.0:
        raise ValueError(f"{name} must be positive")
    return value


def _natural(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < int(positive):
        raise TypeError(f"{name} must be a builtin integer")
    return value


def _top1_count(value: object, name: str) -> int:
    if type(value) is not list or not value or any(type(item) is not bool for item in value):
        raise TypeError(f"{name} must be a nonempty builtin-boolean list")
    return sum(value)


def _file_binding(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("decision input must be a real file")
    payload = path.read_bytes()
    return {
        "path": str(path),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _placeholder_inputs() -> dict[str, object]:
    return {
        key: {"path": f"/{key}.json", "sha256": "0" * 64, "bytes": 1}
        for key in INPUT_KEYS
    }


def _receipt_evidence(
    receipt: object, *, arm: str
) -> tuple[int, int, tuple[int, ...]]:
    if type(receipt) is not dict or receipt.get("arm") != arm:
        raise ValueError(f"{arm} receipt differs")
    allocated = _natural(receipt.get("peak_allocated_bytes"), f"{arm} allocated")
    reserved = _natural(receipt.get("peak_reserved_bytes"), f"{arm} reserved")
    if allocated <= 0 or reserved < allocated:
        raise ValueError(f"{arm} memory differs")
    checkpoints = receipt.get("checkpoints")
    if type(checkpoints) is not list or len(checkpoints) != len(EPOCHS):
        raise ValueError(f"{arm} checkpoints differ")
    sizes: list[int] = []
    for expected_epoch, row in zip(EPOCHS, checkpoints, strict=True):
        if type(row) is not dict or row.get("epoch") != expected_epoch:
            raise ValueError(f"{arm} checkpoint order differs")
        size = _natural(row.get("bytes"), f"{arm} checkpoint bytes", positive=True)
        sizes.append(size)
    return allocated, reserved, tuple(sizes)


def build_seed0_decision(
    *,
    pair_result: object,
    profile_comparison: object,
    control_receipt: object,
    candidate_receipt: object,
    inputs: object | None = None,
) -> dict[str, object]:
    """Recompute the seed-0 quality and operational decision from validated inputs."""

    control_allocated, control_reserved, control_sizes = _receipt_evidence(
        control_receipt, arm=ARMS[0]
    )
    candidate_allocated, candidate_reserved, candidate_sizes = _receipt_evidence(
        candidate_receipt, arm=ARMS[1]
    )
    if type(pair_result) is not dict or type(pair_result.get("rows")) is not list:
        raise ValueError("paired result differs")
    rows = pair_result["rows"]
    if len(rows) != len(EPOCHS):
        raise ValueError("paired result row count differs")

    candidate_primary_by_epoch: dict[int, float] = {}
    observed_control_sizes: list[int] = []
    observed_candidate_sizes: list[int] = []
    control_epoch16_primary = 0.0
    control_top1_count = 0
    candidate_top1_count = 0
    for expected_epoch, row in zip(EPOCHS, rows, strict=True):
        if type(row) is not dict or row.get("epoch") != expected_epoch:
            raise ValueError("paired result epoch order differs")
        arms = row.get("arms")
        if type(arms) is not dict or tuple(arms) != ARMS:
            raise ValueError("paired result arms differ")
        size_lists = (observed_control_sizes, observed_candidate_sizes)
        for arm, sizes in zip(ARMS, size_lists, strict=True):
            arm_row = arms[arm]
            if type(arm_row) is not dict:
                raise ValueError("paired result arm row differs")
            sizes.append(
                _natural(
                    arm_row.get("checkpoint_bytes"),
                    f"{arm} paired checkpoint bytes",
                    positive=True,
                )
            )
            primary = arm_row.get("primary")
            if type(primary) is not dict:
                raise ValueError("paired primary metrics differ")
            _finite_float(primary.get("map_at_r"), f"{arm} primary mAP")
            _top1_count(primary.get("top1_correct"), f"{arm} top-1")
        control_primary = arms[ARMS[0]]["primary"]
        candidate_primary = arms[ARMS[1]]["primary"]
        candidate_primary_by_epoch[expected_epoch] = candidate_primary["map_at_r"]
        if expected_epoch == 16:
            control_epoch16_primary = control_primary["map_at_r"]
            control_top1_count = _top1_count(
                control_primary["top1_correct"], "control top-1"
            )
            candidate_top1_count = _top1_count(
                candidate_primary["top1_correct"], "candidate top-1"
            )

    if (
        tuple(observed_control_sizes) != control_sizes
        or tuple(observed_candidate_sizes) != candidate_sizes
    ):
        raise ValueError("paired result and receipt checkpoint bytes differ")
    checkpoint_bytes_equal = control_sizes == candidate_sizes

    if type(profile_comparison) is not dict:
        raise ValueError("profile comparison differs")
    ratios = profile_comparison.get("ratios")
    if type(ratios) is not dict:
        raise ValueError("profile comparison ratios differ")
    wall_ratio = _finite_float(ratios.get(STEP_TIME_METRIC), "wall-step ratio", positive=True)
    cuda_ratio = _finite_float(ratios.get("cuda_step"), "CUDA-step ratio", positive=True)
    if type(profile_comparison.get("checkpoint_bytes_equal")) is not bool:
        raise TypeError("profile checkpoint equality must be a builtin boolean")
    if profile_comparison["checkpoint_bytes_equal"] != (
        control_sizes[-1] == candidate_sizes[-1]
    ):
        raise ValueError("profile and receipt checkpoint equality differ")

    primary_map_delta = float(
        candidate_primary_by_epoch[16] - control_epoch16_primary
    )
    allocated_ratio = float(candidate_allocated / control_allocated)
    reserved_ratio = float(candidate_reserved / control_reserved)
    decision = EVALUATOR.selection_decision(
        primary_map_delta=primary_map_delta,
        control_top1_count=control_top1_count,
        candidate_top1_count=candidate_top1_count,
        candidate_primary_by_epoch=candidate_primary_by_epoch,
        control_epoch16_primary=control_epoch16_primary,
        abba_step_time_ratio=wall_ratio,
        peak_allocated_ratio=allocated_ratio,
        peak_reserved_ratio=reserved_ratio,
        control_checkpoint_bytes=sum(control_sizes),
        candidate_checkpoint_bytes=sum(candidate_sizes),
        checkpoint_bytes_equal=checkpoint_bytes_equal,
    )
    return {
        "schema_version": "unicom-full-width-seed0-decision-v1",
        "inputs": _placeholder_inputs() if inputs is None else inputs,
        "metric_authority": {
            "step_time": STEP_TIME_METRIC,
            "memory": "training_run_receipts",
            "checkpoint_bytes": "paired_result_and_training_run_receipts",
        },
        "evidence": {
            "primary_map_delta": primary_map_delta,
            "control_top1_count": control_top1_count,
            "candidate_top1_count": candidate_top1_count,
            "candidate_primary_by_epoch": [
                {"epoch": epoch, "map_at_r": candidate_primary_by_epoch[epoch]}
                for epoch in EPOCHS
            ],
            "control_epoch16_primary": control_epoch16_primary,
            "abba_step_time_ratio": wall_ratio,
            "observed_cuda_step_ratio": cuda_ratio,
            "peak_allocated_ratio": allocated_ratio,
            "peak_reserved_ratio": reserved_ratio,
            "checkpoint_bytes_equal": checkpoint_bytes_equal,
            "control_checkpoint_bytes": list(control_sizes),
            "candidate_checkpoint_bytes": list(candidate_sizes),
        },
        "decision": decision,
        "status": decision["decision"],
    }


def validate_seed0_decision(value: object) -> None:
    """Strictly validate and recompute a persisted seed-0 decision."""

    if type(value) is not dict or tuple(value) != TOP_KEYS:
        raise ValueError("seed-0 decision schema differs")
    if value["schema_version"] != "unicom-full-width-seed0-decision-v1":
        raise ValueError("seed-0 decision version differs")
    inputs = value["inputs"]
    if type(inputs) is not dict or tuple(inputs) != INPUT_KEYS:
        raise ValueError("seed-0 decision input schema differs")
    for binding in inputs.values():
        if (
            type(binding) is not dict
            or tuple(binding) != INPUT_BINDING_KEYS
            or type(binding["path"]) is not str
            or not binding["path"]
            or type(binding["sha256"]) is not str
            or len(binding["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in binding["sha256"])
            or type(binding["bytes"]) is not int
            or binding["bytes"] <= 0
        ):
            raise ValueError("seed-0 decision input binding differs")
    authority = value["metric_authority"]
    if (
        type(authority) is not dict
        or tuple(authority) != METRIC_AUTHORITY_KEYS
        or tuple(authority.values())
        != (
            STEP_TIME_METRIC,
            "training_run_receipts",
            "paired_result_and_training_run_receipts",
        )
    ):
        raise ValueError("seed-0 metric authority differs")
    evidence = value["evidence"]
    if type(evidence) is not dict or tuple(evidence) != EVIDENCE_KEYS:
        raise ValueError("seed-0 evidence schema differs")
    trajectory = evidence["candidate_primary_by_epoch"]
    if type(trajectory) is not list or len(trajectory) != len(EPOCHS):
        raise ValueError("seed-0 trajectory differs")
    by_epoch: dict[int, float] = {}
    for expected_epoch, row in zip(EPOCHS, trajectory, strict=True):
        if (
            type(row) is not dict
            or tuple(row) != ("epoch", "map_at_r")
            or row["epoch"] != expected_epoch
        ):
            raise ValueError("seed-0 trajectory row differs")
        by_epoch[expected_epoch] = _finite_float(
            row["map_at_r"], "seed-0 trajectory mAP"
        )
    primary_map_delta = _finite_float(
        evidence["primary_map_delta"], "primary mAP delta"
    )
    control_epoch16_primary = _finite_float(
        evidence["control_epoch16_primary"], "control epoch-16 primary"
    )
    if primary_map_delta != by_epoch[16] - control_epoch16_primary:
        raise ValueError("seed-0 primary mAP delta relation differs")
    checkpoint_equal = evidence["checkpoint_bytes_equal"]
    if type(checkpoint_equal) is not bool:
        raise TypeError("seed-0 checkpoint equality must be a builtin boolean")
    size_vectors: list[tuple[int, ...]] = []
    for key in ("control_checkpoint_bytes", "candidate_checkpoint_bytes"):
        sizes = evidence[key]
        if type(sizes) is not list or len(sizes) != len(EPOCHS):
            raise ValueError("seed-0 checkpoint sizes differ")
        size_vectors.append(
            tuple(_natural(item, "seed-0 checkpoint bytes", positive=True) for item in sizes)
        )
    if checkpoint_equal != (size_vectors[0] == size_vectors[1]):
        raise ValueError("seed-0 checkpoint equality relation differs")
    expected = EVALUATOR.selection_decision(
        primary_map_delta=primary_map_delta,
        control_top1_count=_natural(evidence["control_top1_count"], "control top-1"),
        candidate_top1_count=_natural(evidence["candidate_top1_count"], "candidate top-1"),
        candidate_primary_by_epoch=by_epoch,
        control_epoch16_primary=control_epoch16_primary,
        abba_step_time_ratio=_finite_float(
            evidence["abba_step_time_ratio"], "A-B-B-A step-time", positive=True
        ),
        peak_allocated_ratio=_finite_float(
            evidence["peak_allocated_ratio"], "allocated ratio", positive=True
        ),
        peak_reserved_ratio=_finite_float(
            evidence["peak_reserved_ratio"], "reserved ratio", positive=True
        ),
        control_checkpoint_bytes=sum(size_vectors[0]),
        candidate_checkpoint_bytes=sum(size_vectors[1]),
        checkpoint_bytes_equal=checkpoint_equal,
    )
    _finite_float(
        evidence["observed_cuda_step_ratio"], "observed CUDA-step ratio", positive=True
    )
    if value["decision"] != expected or value["status"] != expected["decision"]:
        raise ValueError("seed-0 decision recomputation differs")


def _validate_run_config(
    config: object,
    args: argparse.Namespace,
    *,
    observed_command: list[str] | None = None,
) -> None:
    if (
        type(config) is not dict
        or config.get("schema_version") != "unicom-full-width-objective-run-v2"
    ):
        raise ValueError("run configuration version differs")
    operational = config.get("thresholds", {}).get("operational")
    if (
        type(operational) is not dict
        or operational.get("step_time_metric") != STEP_TIME_METRIC
        or operational.get("step_time_ratio") != 1.02
        or operational.get("peak_allocated_ratio") != 1.02
        or operational.get("peak_reserved_ratio") != 1.02
        or operational.get("checkpoint_bytes_equal") is not True
    ):
        raise ValueError("run configuration operational gate differs")
    downstream = config.get("seed0_downstream")
    if (
        type(downstream) is not dict
        or downstream.get("decision_path") != str(args.output)
    ):
        raise ValueError("run configuration decision output differs")
    if downstream.get("pair_inventory", {}).get("path") != str(args.pair_inventory):
        raise ValueError("run configuration pair inventory differs")
    if downstream.get("pair_result") != str(args.pair_result):
        raise ValueError("run configuration pair result differs")
    if downstream.get("profile_comparison") != str(args.profile_comparison):
        raise ValueError("run configuration profile comparison differs")
    schedule = config.get("run_schedule")
    if type(schedule) is not list or not schedule or schedule[0].get("seed") != 0:
        raise ValueError("run configuration seed-0 schedule differs")
    receipts = [run.get("receipt") for run in schedule[0].get("runs", [])]
    if receipts != [str(args.control_receipt), str(args.candidate_receipt)]:
        raise ValueError("run configuration seed-0 receipts differ")
    if observed_command is not None:
        templates = config.get("command_templates")
        if (
            type(templates) is not dict
            or templates.get("decision_command") != observed_command
            or any(type(token) is not str or not token for token in observed_command)
        ):
            raise ValueError("run configuration decision command differs")


def _cross_bind_inputs(
    *,
    config: dict[str, object],
    pair_inventory: dict[str, object],
    pair_result: dict[str, object],
    profile_comparison: dict[str, object],
    control_receipt: dict[str, object],
    candidate_receipt: dict[str, object],
    run_config_sha256: str,
) -> None:
    receipts = {ARMS[0]: control_receipt, ARMS[1]: candidate_receipt}
    source = config.get("source")
    if type(source) is not dict or type(source.get("commit")) is not str:
        raise ValueError("run configuration source differs")
    source_commit = source["commit"]
    expected_runtime = config["environment"]
    if type(expected_runtime) is not dict:
        raise ValueError("run configuration environment differs")
    runtime = {
        "python": expected_runtime.get("python"),
        "torch": expected_runtime.get("torch"),
        "cuda": expected_runtime.get("cuda"),
    }
    for arm, receipt in receipts.items():
        if (
            receipt.get("seed") != 0
            or receipt.get("arm") != arm
            or receipt.get("source_commit") != source_commit
            or receipt.get("config_sha256") != run_config_sha256
            or receipt.get("runtime") != runtime
        ):
            raise ValueError("training receipt run binding differs")
    if (
        profile_comparison.get("config_sha256") != run_config_sha256
        or profile_comparison.get("source_commit")
        != control_receipt.get("source_commit")
        or candidate_receipt.get("source_commit")
        != control_receipt.get("source_commit")
    ):
        raise ValueError("profile comparison run binding differs")
    inventory = pair_inventory.get("inventory")
    rows = pair_result.get("rows")
    if (
        pair_inventory.get("seed") != 0
        or type(inventory) is not list
        or type(rows) is not list
    ):
        raise ValueError("paired inventory binding differs")
    for epoch_index, epoch in enumerate(EPOCHS):
        for arm_index, arm in enumerate(ARMS):
            inventory_row = inventory[epoch_index * len(ARMS) + arm_index]
            receipt_row = receipts[arm]["checkpoints"][epoch_index]
            pair_row = rows[epoch_index]["arms"][arm]
            if (
                inventory_row.get("arm") != arm
                or inventory_row.get("epoch") != epoch
                or tuple(
                    inventory_row.get(key) for key in ("path", "sha256", "bytes")
                )
                != tuple(receipt_row.get(key) for key in ("path", "sha256", "bytes"))
                or tuple(
                    pair_row.get(key)
                    for key in (
                        "checkpoint_path",
                        "checkpoint_sha256",
                        "checkpoint_bytes",
                    )
                )
                != tuple(receipt_row.get(key) for key in ("path", "sha256", "bytes"))
            ):
                raise ValueError("paired checkpoint authority differs")


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-config", required=True, type=Path)
    parser.add_argument("--pair-inventory", required=True, type=Path)
    parser.add_argument("--pair-result", required=True, type=Path)
    parser.add_argument("--profile-comparison", required=True, type=Path)
    parser.add_argument("--control-receipt", required=True, type=Path)
    parser.add_argument("--candidate-receipt", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        paths = {
            "run_config": args.run_config,
            "pair_inventory": args.pair_inventory,
            "pair_result": args.pair_result,
            "profile_comparison": args.profile_comparison,
            "control_receipt": args.control_receipt,
            "candidate_receipt": args.candidate_receipt,
        }
        loaded = {key: EVALUATOR.strict_json_object(path) for key, path in paths.items()}
        observed_command = list(sys.orig_argv) if arguments is None else None
        _validate_run_config(
            loaded["run_config"], args, observed_command=observed_command
        )
        EVALUATOR.validate_pair_result(loaded["pair_result"], loaded["pair_inventory"])
        COMPARATOR.validate_comparison_result(loaded["profile_comparison"])
        TRAINER.validate_training_run_receipt(loaded["control_receipt"])
        TRAINER.validate_training_run_receipt(loaded["candidate_receipt"])
        run_config_sha256 = hashlib.sha256(args.run_config.read_bytes()).hexdigest()
        _cross_bind_inputs(
            config=loaded["run_config"],
            pair_inventory=loaded["pair_inventory"],
            pair_result=loaded["pair_result"],
            profile_comparison=loaded["profile_comparison"],
            control_receipt=loaded["control_receipt"],
            candidate_receipt=loaded["candidate_receipt"],
            run_config_sha256=run_config_sha256,
        )
        control_sha = hashlib.sha256(args.control_receipt.read_bytes()).hexdigest()
        candidate_sha = hashlib.sha256(args.candidate_receipt.read_bytes()).hexdigest()
        if loaded["profile_comparison"].get("receipt_sha256s") != [
            control_sha,
            candidate_sha,
            candidate_sha,
            control_sha,
        ]:
            raise ValueError("profile comparison receipt bindings differ")
        result = build_seed0_decision(
            pair_result=loaded["pair_result"],
            profile_comparison=loaded["profile_comparison"],
            control_receipt=loaded["control_receipt"],
            candidate_receipt=loaded["candidate_receipt"],
            inputs={key: _file_binding(path) for key, path in paths.items()},
        )
        validate_seed0_decision(result)
        EVALUATOR.publish_result(result, args.output, validate=validate_seed0_decision)
    except Exception as error:
        print(f"seed-0 decision failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
