#!/usr/bin/env python3
"""Run the frozen train-only AHNCR held-out-label falsifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

from sfora.asymmetric_cohort_residual import (
    ArmComparison,
    DirectComparison,
    Evaluation,
    decide_ahncr,
    evaluate_ahncr,
    exact_mcnemar,
    reporting_shards,
    select_query_and_gallery,
    split_heldout_labels,
)

SCHEMA_VERSION = "inshop-ahncr-v1"
EXPECTED_TRAIN_SHA256 = "67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea"
K = 50
PERMUTATION_SEED = 20260814
PERMUTATION_COUNT = 20
ARM_NAMES = (
    "raw",
    "ahncr",
    "global_mean",
    "positive_expansion",
    "unary_cohort_density",
    "symmetric_ad_norm",
)
DIRECT_CONTROL_NAMES = ARM_NAMES[2:]
PREDICATE_NAMES = (
    "pooled_gain",
    "paired_significance",
    "shard_consistency",
    "global_mean_control",
    "positive_expansion_control",
    "unary_density_control",
    "symmetric_normalization_control",
    "assignment_null",
    "transition_direction",
    "global_mean_paired",
    "positive_expansion_paired",
    "unary_cohort_density_paired",
    "symmetric_ad_norm_paired",
)


class TrainBundle(NamedTuple):
    embeddings: np.ndarray
    labels: np.ndarray
    example_ids: np.ndarray
    checkpoint_sha256: str


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.ascontiguousarray(value, dtype=dtype).tobytes()).hexdigest()


def _ids_sha256(value: np.ndarray) -> str:
    digest = hashlib.sha256()
    for item in value.tolist():
        encoded = item.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def _is_sha(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_train_archive(path: Path) -> TrainBundle:
    if not path.is_file() or path.is_symlink():
        raise ValueError("train input must be a regular nonsymlink file")
    actual_sha = _sha256_path(path)
    if actual_sha != EXPECTED_TRAIN_SHA256:
        raise ValueError("train input SHA-256 differs from the frozen archive")
    with np.load(path, allow_pickle=False) as payload:
        core_keys = (
            "embeddings",
            "labels",
            "example_ids",
            "split",
            "checkpoint_sha256",
        )
        registered_keys = (
            "embeddings",
            "labels",
            "example_ids",
            "source_paths",
            "artifact_selection",
            "split",
            "checkpoint_sha256",
            "report_sha256",
        )
        if tuple(payload.files) not in (core_keys, registered_keys):
            raise ValueError("train archive keys or order differ")
        embeddings = np.array(payload["embeddings"], copy=True)
        labels = np.array(payload["labels"], copy=True)
        example_ids = np.array(payload["example_ids"], copy=True)
        split_value = payload["split"]
        checkpoint_value = payload["checkpoint_sha256"]
    if split_value.shape != () or type(split_value.item()) is not str:
        raise ValueError("split must be a scalar concrete string")
    if split_value.item() != "train":
        raise ValueError("train archive split differs")
    if checkpoint_value.shape != () or type(checkpoint_value.item()) is not str:
        raise ValueError("checkpoint_sha256 must be a scalar concrete string")
    checkpoint_sha256 = checkpoint_value.item()
    if not _is_sha(checkpoint_sha256):
        raise ValueError("checkpoint_sha256 must be lowercase 64-hex")
    if embeddings.ndim != 2 or embeddings.dtype != np.float32:
        raise ValueError("embeddings must be a rank-2 float32 array")
    if labels.shape != (embeddings.shape[0],) or labels.dtype != np.int64:
        raise ValueError("labels must be a row-aligned int64 vector")
    if example_ids.shape != (embeddings.shape[0],) or example_ids.dtype.kind != "U":
        raise ValueError("example_ids must be a row-aligned Unicode vector")
    if np.any(example_ids == "") or np.unique(example_ids).size != example_ids.size:
        raise ValueError("example_ids must be nonempty and unique")
    if not np.isfinite(embeddings).all() or not np.allclose(
        np.linalg.norm(embeddings.astype(np.float64), axis=1),
        1.0,
        rtol=0.0,
        atol=2e-4,
    ):
        raise ValueError("embeddings must be finite and unit normalized")
    return TrainBundle(embeddings, labels, example_ids, checkpoint_sha256)


def _arm_dict(value: ArmComparison) -> dict[str, Any]:
    return {
        "raw_recall": value.raw_recall,
        "recall": value.recall,
        "gain": value.gain,
        "raw_correct": value.raw_correct,
        "correct": value.correct,
        "wrong_to_right": value.wrong_to_right,
        "right_to_wrong": value.right_to_wrong,
        "p_value": value.p_value,
        "shard_counts": list(value.shard_counts),
        "shard_raw_correct": list(value.shard_raw_correct),
        "shard_correct": list(value.shard_correct),
        "shard_gains": list(value.shard_gains),
    }


def _direct_dict(value: DirectComparison) -> dict[str, int | float]:
    return {
        "control_wrong_ahncr_right": value.control_wrong_ahncr_right,
        "control_right_ahncr_wrong": value.control_right_ahncr_wrong,
        "p_value": value.p_value,
    }


def build_ahncr_report(train_path: Path, *, block_size: int = 256) -> dict[str, Any]:
    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive concrete integer")
    thread_environment = {
        name: os.environ.get(name)
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    }
    if tuple(thread_environment.values()) != ("1", "1", "1"):
        raise ValueError("BLAS thread environment must be pinned to 1")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CUDA_VISIBLE_DEVICES must be empty")
    bundle = load_train_archive(train_path)
    split = split_heldout_labels(bundle.labels)
    cohort_mask = np.isin(bundle.labels, split.cohort_labels)
    cohort = np.asarray(bundle.embeddings[cohort_mask], dtype=np.float32)
    retrieval = select_query_and_gallery(bundle.labels, bundle.example_ids, split.evaluation_labels)
    queries = np.asarray(bundle.embeddings[retrieval.query_indices], dtype=np.float32)
    gallery = np.asarray(bundle.embeddings[retrieval.gallery_indices], dtype=np.float32)
    gallery_labels = np.asarray(bundle.labels[retrieval.gallery_indices], dtype=np.int64)
    shards = reporting_shards(split.evaluation_labels)
    evaluation = evaluate_ahncr(
        queries,
        retrieval.query_labels,
        gallery,
        gallery_labels,
        cohort,
        shards,
        k=K,
        block_size=block_size,
        permutation_seed=PERMUTATION_SEED,
    )
    passed, predicates = decide_ahncr(evaluation)
    report = {
        "schema_version": SCHEMA_VERSION,
        "input": {
            "path": str(train_path),
            "sha256": _sha256_path(train_path),
            "checkpoint_sha256": bundle.checkpoint_sha256,
            "row_count": int(bundle.labels.size),
            "dimension": int(bundle.embeddings.shape[1]),
        },
        "environment": {
            "numpy_version": str(np.__version__),
            "CUDA_VISIBLE_DEVICES": "",
            **thread_environment,
        },
        "configuration": {
            "k": K,
            "block_size": block_size,
            "fit_fraction": 0.8,
            "permutation_seed": PERMUTATION_SEED,
            "permutation_count": PERMUTATION_COUNT,
        },
        "split": {
            "cohort_label_count": int(split.cohort_labels.size),
            "evaluation_label_count": int(split.evaluation_labels.size),
            "cohort_row_count": int(cohort.shape[0]),
            "query_count": int(retrieval.query_indices.size),
            "gallery_count": int(retrieval.gallery_indices.size),
            "cohort_labels_sha256": _array_sha256(split.cohort_labels, "<i8"),
            "evaluation_labels_sha256": _array_sha256(split.evaluation_labels, "<i8"),
            "query_ids_sha256": _ids_sha256(bundle.example_ids[retrieval.query_indices]),
            "gallery_ids_sha256": _ids_sha256(bundle.example_ids[retrieval.gallery_indices]),
            "shard_counts": [int(np.sum(shards == shard)) for shard in range(4)],
        },
        "arms": {name: _arm_dict(evaluation.arms[name]) for name in ARM_NAMES},
        "control_comparisons": {
            name: _direct_dict(evaluation.control_comparisons[name])
            for name in DIRECT_CONTROL_NAMES
        },
        "null": {
            "shuffled_gains": list(evaluation.shuffled_gains),
            "shuffled_p95": evaluation.shuffled_p95,
        },
        "decision": {
            "predicates": predicates,
            "passes_falsifier": passed,
        },
    }
    return validate_ahncr_report(report)


def _keys(value: object, expected: tuple[str, ...], name: str) -> dict[str, Any]:
    if type(value) is not dict or tuple(value) != expected:
        raise ValueError(f"{name} keys or order differ")
    return value


def _finite_float(value: object, name: str) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite concrete float")
    return value


def _int(value: object, name: str, *, positive: bool = False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{name} must be a valid concrete integer")
    return value


def validate_ahncr_report(value: object) -> dict[str, Any]:
    report = _keys(
        value,
        (
            "schema_version",
            "input",
            "environment",
            "configuration",
            "split",
            "arms",
            "control_comparisons",
            "null",
            "decision",
        ),
        "report",
    )
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version differs")
    input_value = _keys(
        report["input"],
        ("path", "sha256", "checkpoint_sha256", "row_count", "dimension"),
        "input",
    )
    if type(input_value["path"]) is not str or not input_value["path"]:
        raise ValueError("input.path differs")
    if not _is_sha(input_value["sha256"]) or input_value["sha256"] != EXPECTED_TRAIN_SHA256:
        raise ValueError("input.sha256 differs")
    if not _is_sha(input_value["checkpoint_sha256"]):
        raise ValueError("input.checkpoint_sha256 differs")
    _int(input_value["row_count"], "input.row_count", positive=True)
    _int(input_value["dimension"], "input.dimension", positive=True)
    environment = _keys(
        report["environment"],
        (
            "numpy_version",
            "CUDA_VISIBLE_DEVICES",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
        ),
        "environment",
    )
    if type(environment["numpy_version"]) is not str or not environment["numpy_version"]:
        raise ValueError("environment.numpy_version differs")
    if tuple(environment[name] for name in tuple(environment)[1:]) != ("", "1", "1", "1"):
        raise ValueError("environment values differ")
    configuration = _keys(
        report["configuration"],
        ("k", "block_size", "fit_fraction", "permutation_seed", "permutation_count"),
        "configuration",
    )
    if configuration["k"] != K or configuration["fit_fraction"] != 0.8:
        raise ValueError("configuration constants differ")
    _int(configuration["block_size"], "configuration.block_size", positive=True)
    if (
        configuration["permutation_seed"] != PERMUTATION_SEED
        or configuration["permutation_count"] != PERMUTATION_COUNT
    ):
        raise ValueError("permutation configuration differs")
    split = _keys(
        report["split"],
        (
            "cohort_label_count",
            "evaluation_label_count",
            "cohort_row_count",
            "query_count",
            "gallery_count",
            "cohort_labels_sha256",
            "evaluation_labels_sha256",
            "query_ids_sha256",
            "gallery_ids_sha256",
            "shard_counts",
        ),
        "split",
    )
    for name in (
        "cohort_label_count",
        "evaluation_label_count",
        "cohort_row_count",
        "query_count",
        "gallery_count",
    ):
        _int(split[name], f"split.{name}", positive=True)
    if split["query_count"] != split["evaluation_label_count"]:
        raise ValueError("query and evaluation-label counts differ")
    for name in (
        "cohort_labels_sha256",
        "evaluation_labels_sha256",
        "query_ids_sha256",
        "gallery_ids_sha256",
    ):
        if not _is_sha(split[name]):
            raise ValueError(f"split.{name} differs")
    if type(split["shard_counts"]) is not list or len(split["shard_counts"]) != 4:
        raise ValueError("split.shard_counts differs")
    for count in split["shard_counts"]:
        _int(count, "split.shard_counts item", positive=True)
    if sum(split["shard_counts"]) != split["query_count"]:
        raise ValueError("split.shard_counts total differs")
    arms_value = _keys(report["arms"], ARM_NAMES, "arms")
    arm_objects: dict[str, ArmComparison] = {}
    for name in ARM_NAMES:
        arm = _keys(
            arms_value[name],
            (
                "raw_recall",
                "recall",
                "gain",
                "raw_correct",
                "correct",
                "wrong_to_right",
                "right_to_wrong",
                "p_value",
                "shard_counts",
                "shard_raw_correct",
                "shard_correct",
                "shard_gains",
            ),
            f"arms.{name}",
        )
        floats = {
            key: _finite_float(arm[key], f"arms.{name}.{key}")
            for key in ("raw_recall", "recall", "gain", "p_value")
        }
        ints = {
            key: _int(arm[key], f"arms.{name}.{key}")
            for key in ("raw_correct", "correct", "wrong_to_right", "right_to_wrong")
        }
        if (
            ints["raw_correct"] > split["query_count"]
            or ints["correct"] > split["query_count"]
            or ints["wrong_to_right"] > split["query_count"]
            or ints["right_to_wrong"] > split["query_count"]
            or not 0.0 <= floats["raw_recall"] <= 1.0
            or not 0.0 <= floats["recall"] <= 1.0
            or not 0.0 <= floats["p_value"] <= 1.0
            or not -1.0 <= floats["gain"] <= 1.0
        ):
            raise ValueError(f"arms.{name} bounds differ")
        if (
            floats["raw_recall"] != ints["raw_correct"] / split["query_count"]
            or floats["recall"] != ints["correct"] / split["query_count"]
        ):
            raise ValueError(f"arms.{name} recall differs")
        if floats["gain"] != floats["recall"] - floats["raw_recall"]:
            raise ValueError(f"arms.{name}.gain differs")
        if ints["correct"] - ints["raw_correct"] != ints["wrong_to_right"] - ints["right_to_wrong"]:
            raise ValueError(f"arms.{name} transitions differ")
        if floats["p_value"] != exact_mcnemar(ints["wrong_to_right"], ints["right_to_wrong"]):
            raise ValueError(f"arms.{name}.p_value differs")
        if any(
            type(arm[key]) is not list or len(arm[key]) != 4
            for key in (
                "shard_counts",
                "shard_raw_correct",
                "shard_correct",
                "shard_gains",
            )
        ):
            raise ValueError(f"arms.{name} shard arrays differ")
        for index in range(4):
            count = arm["shard_counts"][index]
            raw_correct = arm["shard_raw_correct"][index]
            correct = arm["shard_correct"][index]
            gain = arm["shard_gains"][index]
            _int(count, "arm shard count", positive=True)
            _int(raw_correct, "arm shard raw_correct")
            _int(correct, "arm shard correct")
            _finite_float(gain, "arm shard gain")
            if raw_correct > count or correct > count:
                raise ValueError(f"arms.{name} shard bounds differ")
            if count != split["shard_counts"][index] or gain != (correct - raw_correct) / count:
                raise ValueError(f"arms.{name} shard relation differs")
        if (
            sum(arm["shard_counts"]) != split["query_count"]
            or sum(arm["shard_raw_correct"]) != ints["raw_correct"]
            or sum(arm["shard_correct"]) != ints["correct"]
        ):
            raise ValueError(f"arms.{name} shard totals differ")
        arm_objects[name] = ArmComparison(
            floats["raw_recall"],
            floats["recall"],
            floats["gain"],
            ints["wrong_to_right"],
            ints["right_to_wrong"],
            floats["p_value"],
            tuple(arm["shard_gains"]),
            ints["raw_correct"],
            ints["correct"],
            tuple(arm["shard_counts"]),
            tuple(arm["shard_raw_correct"]),
            tuple(arm["shard_correct"]),
        )
    raw = arm_objects["raw"]
    if (
        raw.recall != raw.raw_recall
        or raw.correct != raw.raw_correct
        or raw.wrong_to_right != 0
        or raw.right_to_wrong != 0
        or raw.gain != 0.0
        or raw.shard_correct != raw.shard_raw_correct
    ):
        raise ValueError("raw arm differs from its baseline")
    for name in ARM_NAMES[1:]:
        arm = arm_objects[name]
        if (
            arm.raw_recall != raw.raw_recall
            or arm.raw_correct != raw.raw_correct
            or arm.shard_counts != raw.shard_counts
            or arm.shard_raw_correct != raw.shard_raw_correct
        ):
            raise ValueError(f"arms.{name} shared raw baseline differs")
    direct_value = _keys(
        report["control_comparisons"],
        DIRECT_CONTROL_NAMES,
        "control_comparisons",
    )
    direct_objects: dict[str, DirectComparison] = {}
    for name in DIRECT_CONTROL_NAMES:
        comparison = _keys(
            direct_value[name],
            (
                "control_wrong_ahncr_right",
                "control_right_ahncr_wrong",
                "p_value",
            ),
            f"control_comparisons.{name}",
        )
        favor = _int(
            comparison["control_wrong_ahncr_right"],
            f"control_comparisons.{name}.control_wrong_ahncr_right",
        )
        against = _int(
            comparison["control_right_ahncr_wrong"],
            f"control_comparisons.{name}.control_right_ahncr_wrong",
        )
        p_value = _finite_float(comparison["p_value"], f"control_comparisons.{name}.p_value")
        if (
            favor > split["query_count"]
            or against > split["query_count"]
            or p_value != exact_mcnemar(favor, against)
            or arm_objects["ahncr"].correct - arm_objects[name].correct != favor - against
        ):
            raise ValueError(f"control_comparisons.{name} relations differ")
        direct_objects[name] = DirectComparison(favor, against, p_value)
    null = _keys(report["null"], ("shuffled_gains", "shuffled_p95"), "null")
    if type(null["shuffled_gains"]) is not list or len(null["shuffled_gains"]) != PERMUTATION_COUNT:
        raise ValueError("null.shuffled_gains differs")
    gains = tuple(
        _finite_float(item, "null.shuffled_gains item") for item in null["shuffled_gains"]
    )
    p95 = _finite_float(null["shuffled_p95"], "null.shuffled_p95")
    if p95 != float(np.quantile(gains, 0.95, method="linear")):
        raise ValueError("null.shuffled_p95 differs")
    expected_pass, expected_predicates = decide_ahncr(
        Evaluation(arm_objects, gains, p95, direct_objects)
    )
    decision = _keys(report["decision"], ("predicates", "passes_falsifier"), "decision")
    if _keys(decision["predicates"], PREDICATE_NAMES, "decision.predicates") != expected_predicates:
        raise ValueError("decision.predicates differ")
    if (
        type(decision["passes_falsifier"]) is not bool
        or decision["passes_falsifier"] is not expected_pass
    ):
        raise ValueError("decision.passes_falsifier differs")
    return report


def write_json_atomic(
    path: Path, payload: dict[str, Any], *, validator: Callable[[object], object]
) -> None:
    if not path.parent.is_dir() or path.is_symlink():
        raise ValueError("output parent must exist and output must not be a symlink")
    validator(payload)
    encoded = (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode()
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor: int | None = None
    temporary_owned = False
    owned: os.stat_result | None = None
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        temporary_owned = True
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        owned = temporary.stat(follow_symlinks=False)
        os.link(temporary, path)
        published = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        validator(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        if published and owned is not None:
            try:
                destination = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if stat.S_ISREG(destination.st_mode) and (
                    destination.st_dev,
                    destination.st_ino,
                ) == (owned.st_dev, owned.st_ino):
                    path.unlink()
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if temporary_owned:
            temporary.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parse_args(argv)
        report = build_ahncr_report(arguments.train, block_size=arguments.block_size)
        write_json_atomic(arguments.output, report, validator=validate_ahncr_report)
    except Exception as error:
        print(f"structural failure: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
