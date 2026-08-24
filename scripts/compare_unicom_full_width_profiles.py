#!/usr/bin/env python3
"""Compare registered UniCOM full-width profiles in symmetric A-B-B-A order."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

ARMS = ("sampled_512", "full_768", "full_768", "sampled_512")
BOOTSTRAP_SEED = 768
BOOTSTRAP_REPLICATES = 10_000
RESULT_KEYS = (
    "schema_version",
    "arms",
    "profile_sha256s",
    "receipt_sha256s",
    "source_commit",
    "config_sha256",
    "checkpoint_epoch",
    "ratios",
    "ratio_bootstrap_95",
    "checkpoint_bytes_equal",
    "kernel_gate",
)
RATIO_KEYS = (
    "step_wall",
    "cuda_step",
    "objective_ceiling",
    "peak_allocated",
    "peak_reserved",
    "fusible_non_backbone",
)
BOOTSTRAP_KEYS = ("step_wall", "cuda_step", "objective_ceiling")
KERNEL_GATE_KEYS = ("control_lower_95", "candidate_lower_95", "candidate_eligible")


def _load_sibling(name: str):
    path = Path(__file__).with_name(f"{name}.py").resolve()
    spec = importlib.util.spec_from_file_location(f"_{name}_for_full_width", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    if Path(module.__file__).resolve() != path:
        raise ValueError(f"loaded {name} source differs")
    return module


PROFILER = _load_sibling("profile_unicom_training_step")
TRAINER = _load_sibling("train_unicom_inshop")


def _reject_constant(value: str) -> None:
    raise ValueError(f"nonfinite JSON constant is forbidden: {value}")


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if type(key) is not str or key in result:
            raise ValueError("JSON object keys differ")
        result[key] = value
    return result


def strict_json_object(path: Path) -> dict[str, object]:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError("comparison input path differs")
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("comparison input exceeds 64 MiB")
    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=_strict_pairs,
        parse_constant=_reject_constant,
    )
    if type(value) is not dict:
        raise ValueError("comparison input root differs")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values or any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("comparison samples differ")
    return float(math.fsum(values) / len(values))


def _ratio(candidate: Sequence[float], control: Sequence[float]) -> float:
    denominator = _mean(control)
    if denominator <= 0.0:
        raise ValueError("comparison control mean must be positive")
    return float(_mean(candidate) / denominator)


def _bootstrap_ratio(
    candidate: tuple[float, ...], control: tuple[float, ...], *, seed: int
) -> list[float]:
    candidate_values = np.asarray(candidate, dtype=np.float64)
    control_values = np.asarray(control, dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(seed))
    candidate_draws = candidate_values[
        generator.integers(
            0,
            candidate_values.size,
            size=(BOOTSTRAP_REPLICATES, candidate_values.size),
        )
    ]
    control_draws = control_values[
        generator.integers(
            0,
            control_values.size,
            size=(BOOTSTRAP_REPLICATES, control_values.size),
        )
    ]
    ratios = candidate_draws.mean(axis=1) / control_draws.mean(axis=1)
    return [float(value) for value in np.quantile(ratios, (0.025, 0.975))]


def _is_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_comparison_result(payload: object) -> None:
    if type(payload) is not dict or tuple(payload) != RESULT_KEYS:
        raise ValueError("comparison result schema differs")
    if (
        payload["schema_version"] != "unicom-full-width-abba-v1"
        or payload["arms"] != list(ARMS)
        or type(payload["profile_sha256s"]) is not list
        or len(payload["profile_sha256s"]) != 4
        or type(payload["receipt_sha256s"]) is not list
        or len(payload["receipt_sha256s"]) != 4
        or len(set(payload["profile_sha256s"])) != 4
        or len(set(payload["receipt_sha256s"])) != 4
        or any(not _is_hex(value, 64) for value in payload["profile_sha256s"])
        or any(not _is_hex(value, 64) for value in payload["receipt_sha256s"])
        or not _is_hex(payload["source_commit"], 40)
        or not _is_hex(payload["config_sha256"], 64)
        or payload["checkpoint_epoch"] != 16
        or type(payload["checkpoint_bytes_equal"]) is not bool
    ):
        raise ValueError("comparison result binding differs")
    ratios = payload["ratios"]
    if (
        type(ratios) is not dict
        or tuple(ratios) != RATIO_KEYS
        or any(
            type(value) is not float or not math.isfinite(value) or value < 0.0
            for value in ratios.values()
        )
    ):
        raise ValueError("comparison result ratios differ")
    intervals = payload["ratio_bootstrap_95"]
    if type(intervals) is not dict or tuple(intervals) != BOOTSTRAP_KEYS:
        raise ValueError("comparison result intervals differ")
    for interval in intervals.values():
        if (
            type(interval) is not list
            or len(interval) != 2
            or any(type(value) is not float or not math.isfinite(value) for value in interval)
            or interval[0] > interval[1]
        ):
            raise ValueError("comparison result interval differs")
    kernel = payload["kernel_gate"]
    if (
        type(kernel) is not dict
        or tuple(kernel) != KERNEL_GATE_KEYS
        or any(
            type(kernel[key]) is not float
            or not math.isfinite(kernel[key])
            or kernel[key] < 0.0
            for key in KERNEL_GATE_KEYS[:2]
        )
        or type(kernel["candidate_eligible"]) is not bool
    ):
        raise ValueError("comparison result kernel gate differs")


def compare_abba(
    profile_paths: tuple[Path, ...], receipt_paths: tuple[Path, ...]
) -> dict[str, object]:
    """Authenticate four artifacts and compute the registered symmetric ratios."""

    if (
        type(profile_paths) is not tuple
        or type(receipt_paths) is not tuple
        or len(profile_paths) != 4
        or len(receipt_paths) != 4
    ):
        raise ValueError("A-B-B-A input inventory differs")
    profile_hashes = tuple(_sha256_file(path) for path in profile_paths)
    receipt_hashes = tuple(_sha256_file(path) for path in receipt_paths)
    if len(set(profile_hashes)) != 4 or len(set(receipt_hashes)) != 4:
        raise ValueError("A-B-B-A artifacts must have distinct bytes")
    profiles = tuple(strict_json_object(path) for path in profile_paths)
    receipts = tuple(strict_json_object(path) for path in receipt_paths)
    for receipt in receipts:
        TRAINER.validate_training_run_receipt(receipt)
    if tuple(receipt["arm"] for receipt in receipts) != ARMS:
        raise ValueError("A-B-B-A arm order differs")
    validated_profiles = tuple(
        PROFILER._profile_samples(profile, "imprinted") for profile in profiles
    )
    if any(profile["checkpoint_epoch"] != 16 for profile in profiles):
        raise ValueError("A-B-B-A checkpoint epoch differs")
    if any(
        earlier["finished_unix_ns"] > later["started_unix_ns"]
        for earlier, later in zip(profiles, profiles[1:], strict=False)
    ):
        raise ValueError("A-B-B-A execution order differs")
    source_commits = {receipt["source_commit"] for receipt in receipts}
    config_hashes = {receipt["config_sha256"] for receipt in receipts}
    trainer_hashes = {receipt["trainer_sha256"] for receipt in receipts}
    receipt_runtimes = {json.dumps(receipt["runtime"], sort_keys=True) for receipt in receipts}
    profile_runtimes = {json.dumps(profile["runtime"], sort_keys=True) for profile in profiles}
    if (
        len(source_commits) != 1
        or len(config_hashes) != 1
        or len(trainer_hashes) != 1
        or len(receipt_runtimes) != 1
        or len(profile_runtimes) != 1
        or any(profile["trainer_sha256"] not in trainer_hashes for profile in profiles)
    ):
        raise ValueError("A-B-B-A source or runtime differs")
    receipt_runtime = receipts[0]["runtime"]
    profile_runtime = profiles[0]["runtime"]
    if (
        receipt_runtime["python"] != profile_runtime["python_version"]
        or receipt_runtime["torch"] != profile_runtime["torch_version"]
        or receipt_runtime["cuda"] != profile_runtime["cuda_version"]
    ):
        raise ValueError("A-B-B-A runtime binding differs")
    for profile, receipt in zip(profiles, receipts, strict=True):
        if profile["run_checkpoint_sha256"] != receipt["checkpoints"][-1]["sha256"]:
            raise ValueError("A-B-B-A checkpoint binding differs")

    control_indices = (0, 3)
    candidate_indices = (1, 2)

    def timing_samples(indices: tuple[int, int]) -> tuple[dict[str, float], ...]:
        return tuple(
            sample
            for index in indices
            for sample in validated_profiles[index][0]
        )

    def timing_values(indices: tuple[int, int], key: str) -> tuple[float, ...]:
        return tuple(
            float(sample[key])
            for index in indices
            for sample in validated_profiles[index][0]
        )

    def objective_values(indices: tuple[int, int]) -> tuple[float, ...]:
        return tuple(
            float(sample["objective_forward_seconds"] + sample["head_backward_seconds"])
            for index in indices
            for sample in validated_profiles[index][0]
        )

    control_timing = timing_samples(control_indices)
    candidate_timing = timing_samples(candidate_indices)
    control_wall = tuple(float(sample["step_wall_seconds"]) for sample in control_timing)
    candidate_wall = tuple(float(sample["step_wall_seconds"]) for sample in candidate_timing)
    control_cuda = timing_values(control_indices, "cuda_step_seconds")
    candidate_cuda = timing_values(candidate_indices, "cuda_step_seconds")
    control_objective = objective_values(control_indices)
    candidate_objective = objective_values(candidate_indices)
    control_allocated = tuple(
        float(receipts[index]["peak_allocated_bytes"]) for index in control_indices
    )
    candidate_allocated = tuple(
        float(receipts[index]["peak_allocated_bytes"]) for index in candidate_indices
    )
    control_reserved = tuple(
        float(receipts[index]["peak_reserved_bytes"]) for index in control_indices
    )
    candidate_reserved = tuple(
        float(receipts[index]["peak_reserved_bytes"]) for index in candidate_indices
    )
    control_fusible = tuple(
        float(value) for index in control_indices for value in validated_profiles[index][1]
    )
    candidate_fusible = tuple(
        float(value) for index in candidate_indices for value in validated_profiles[index][1]
    )
    control_kernel = PROFILER.summarize_profile(control_timing, control_fusible)
    candidate_kernel = PROFILER.summarize_profile(candidate_timing, candidate_fusible)
    checkpoint_sizes = tuple(receipt["checkpoints"][-1]["bytes"] for receipt in receipts)
    result = {
        "schema_version": "unicom-full-width-abba-v1",
        "arms": list(ARMS),
        "profile_sha256s": list(profile_hashes),
        "receipt_sha256s": list(receipt_hashes),
        "source_commit": next(iter(source_commits)),
        "config_sha256": next(iter(config_hashes)),
        "checkpoint_epoch": 16,
        "ratios": {
            "step_wall": _ratio(candidate_wall, control_wall),
            "cuda_step": _ratio(candidate_cuda, control_cuda),
            "objective_ceiling": _ratio(candidate_objective, control_objective),
            "peak_allocated": _ratio(candidate_allocated, control_allocated),
            "peak_reserved": _ratio(candidate_reserved, control_reserved),
            "fusible_non_backbone": _ratio(candidate_fusible, control_fusible),
        },
        "ratio_bootstrap_95": {
            "step_wall": _bootstrap_ratio(candidate_wall, control_wall, seed=BOOTSTRAP_SEED),
            "cuda_step": _bootstrap_ratio(candidate_cuda, control_cuda, seed=BOOTSTRAP_SEED + 1),
            "objective_ceiling": _bootstrap_ratio(
                candidate_objective, control_objective, seed=BOOTSTRAP_SEED + 2
            ),
        },
        "checkpoint_bytes_equal": len(set(checkpoint_sizes)) == 1,
        "kernel_gate": {
            "control_lower_95": control_kernel["fusible_fraction_bootstrap_lower_95"],
            "candidate_lower_95": candidate_kernel["fusible_fraction_bootstrap_lower_95"],
            "candidate_eligible": candidate_kernel["kernel_gate_passed"],
        },
    }
    validate_comparison_result(result)
    return result


def write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    validate_comparison_result(payload)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(temporary)
    encoded = (json.dumps(payload, indent=2, allow_nan=False) + "\n").encode()
    linked = False
    published = False
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        persisted = strict_json_object(temporary)
        validate_comparison_result(persisted)
        if persisted != payload:
            raise RuntimeError("persisted A-B-B-A result differs")
        os.link(temporary, path)
        linked = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
            published = True
        finally:
            os.close(directory)
    finally:
        if linked and not published:
            try:
                temporary_info = temporary.lstat()
                output_info = path.lstat()
            except FileNotFoundError:
                pass
            else:
                if (
                    temporary_info.st_dev,
                    temporary_info.st_ino,
                ) == (output_info.st_dev, output_info.st_ino):
                    path.unlink()
                    cleanup_directory = os.open(
                        path.parent, os.O_RDONLY | os.O_DIRECTORY
                    )
                    try:
                        os.fsync(cleanup_directory)
                    except OSError:
                        pass
                    finally:
                        os.close(cleanup_directory)
        temporary.unlink(missing_ok=True)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", nargs=4, required=True, type=Path)
    parser.add_argument("--receipts", nargs=4, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        result = compare_abba(tuple(args.profiles), tuple(args.receipts))
        write_json_atomic(args.output, result)
    except Exception as error:
        print(f"profile comparison failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
