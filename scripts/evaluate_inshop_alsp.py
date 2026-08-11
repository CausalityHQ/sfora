#!/usr/bin/env python3
"""Fit and falsify an amortized local-scale potential on frozen embeddings."""

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

from sfora.amortized_local_scale import (
    RetrievalComparison,
    compare_potentials,
    decide_alsp,
    density_diagnostics,
    fit_ridge_potential,
    nonself_density,
    predict_potential,
    select_ridge_lambda,
    split_labels,
)

SCHEMA_VERSION = "inshop-alsp-v1"
K = 50
FIT_FRACTION = 0.8
RIDGE_GRID = (1e-6, 1e-4, 1e-2, 1.0, 100.0)
PERMUTATION_SEED = 20260811
RANDOM_DIRECTION_SEED = 20260812
RANDOM_DIRECTION_COUNT = 20
ARM_NAMES = ("raw", "alsp", "constant", "permuted", "oracle")
COMPARISON_KEYS = (
    "raw_recall",
    "corrected_recall",
    "gain",
    "wrong_to_right",
    "right_to_wrong",
    "p_value",
)


class EmbeddingBundle(NamedTuple):
    embeddings: np.ndarray
    labels: np.ndarray
    example_ids: np.ndarray
    split: str
    checkpoint_sha256: str


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def float64_sha256(value: np.ndarray) -> str:
    """Hash canonical little-endian contiguous float64 bytes."""

    if (
        not isinstance(value, np.ndarray)
        or value.ndim != 1
        or not np.isfinite(value).all()
    ):
        raise ValueError("value must be a finite rank-1 array")
    canonical = np.ascontiguousarray(value, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _label_sha256(value: np.ndarray) -> str:
    canonical = np.ascontiguousarray(value, dtype="<i8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _comparison_dict(value: RetrievalComparison) -> dict[str, float | int]:
    return {
        "raw_recall": value.raw_recall,
        "corrected_recall": value.corrected_recall,
        "gain": value.gain,
        "wrong_to_right": value.wrong_to_right,
        "right_to_wrong": value.right_to_wrong,
        "p_value": value.p_value,
    }


def load_embedding_bundle(path: Path, *, expected_split: str) -> EmbeddingBundle:
    """Load an exact frozen embedding archive used by the ALSP falsifier."""

    if not path.is_file() or path.is_symlink():
        raise ValueError(f"embedding path is not a regular file: {path}")
    with np.load(path, allow_pickle=False) as payload:
        required = (
            "embeddings",
            "labels",
            "example_ids",
            "split",
            "checkpoint_sha256",
        )
        missing = [key for key in required if key not in payload.files]
        if missing:
            raise ValueError(f"embedding archive lacks {missing[0]}")
        embeddings = np.array(payload["embeddings"], copy=True)
        labels = np.array(payload["labels"], copy=True)
        example_ids = np.array(payload["example_ids"], copy=True)
        split_value = payload["split"]
        checkpoint_value = payload["checkpoint_sha256"]
        if split_value.shape != () or checkpoint_value.shape != ():
            raise ValueError("split and checkpoint_sha256 must be scalar strings")
        split = str(split_value.item())
        checkpoint_sha256 = str(checkpoint_value.item())
    if embeddings.ndim != 2 or embeddings.dtype != np.float32:
        raise ValueError("embeddings must be a rank-2 float32 array")
    if labels.ndim != 1 or labels.dtype != np.int64:
        raise ValueError("labels must be a rank-1 int64 array")
    if example_ids.ndim != 1 or example_ids.dtype.kind != "U":
        raise ValueError("example_ids must be a rank-1 Unicode array")
    if not (embeddings.shape[0] == labels.size == example_ids.size):
        raise ValueError("embedding archive row counts differ")
    if embeddings.shape[0] < 2 or embeddings.shape[1] == 0:
        raise ValueError("embedding archive must contain at least two rows")
    if not np.isfinite(embeddings).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if not np.allclose(norms, 1.0, rtol=0.0, atol=2e-4):
        raise ValueError("embeddings must be unit-normalized")
    if np.any(example_ids == "") or np.unique(example_ids).size != example_ids.size:
        raise ValueError("example_ids must be nonempty and unique")
    if split != expected_split:
        raise ValueError(f"embedding split is {split!r}, expected {expected_split!r}")
    if len(checkpoint_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in checkpoint_sha256
    ):
        raise ValueError("checkpoint_sha256 must be lowercase 64-hex")
    return EmbeddingBundle(
        embeddings=embeddings,
        labels=labels,
        example_ids=example_ids,
        split=split,
        checkpoint_sha256=checkpoint_sha256,
    )


def build_alsp_report(
    train_path: Path,
    query_path: Path,
    gallery_path: Path,
    *,
    block_size: int = 256,
) -> dict[str, Any]:
    """Fit on train-only densities, freeze, then evaluate the seed-0 test pair."""

    if type(block_size) is not int or block_size <= 0:
        raise ValueError("block_size must be a positive integer")
    train = load_embedding_bundle(train_path, expected_split="train")
    query = load_embedding_bundle(query_path, expected_split="query")
    gallery = load_embedding_bundle(gallery_path, expected_split="gallery")
    if not (
        train.checkpoint_sha256
        == query.checkpoint_sha256
        == gallery.checkpoint_sha256
    ):
        raise ValueError("train/query/gallery checkpoint hashes differ")
    if not (
        train.embeddings.shape[1]
        == query.embeddings.shape[1]
        == gallery.embeddings.shape[1]
    ):
        raise ValueError("train/query/gallery embedding dimensions differ")

    fit_labels, validation_labels = split_labels(train.labels, FIT_FRACTION)
    fit_mask = np.isin(train.labels, fit_labels)
    validation_mask = np.isin(train.labels, validation_labels)
    all_train_targets = nonself_density(
        train.embeddings,
        train.example_ids,
        k=K,
        block_size=block_size,
    )
    selected_model, grid_rows = select_ridge_lambda(
        train.embeddings[fit_mask],
        all_train_targets[fit_mask],
        train.embeddings[validation_mask],
        all_train_targets[validation_mask],
        RIDGE_GRID,
    )

    model = fit_ridge_potential(
        train.embeddings, all_train_targets, selected_model.ridge_lambda
    )
    predicted_potential = predict_potential(model, gallery.embeddings)

    permutation = np.random.Generator(np.random.PCG64(PERMUTATION_SEED)).permutation(
        all_train_targets.size
    )
    permuted_model = fit_ridge_potential(
        train.embeddings,
        all_train_targets[permutation],
        selected_model.ridge_lambda,
    )
    permuted_potential = predict_potential(permuted_model, gallery.embeddings)
    constant_potential = np.full(
        gallery.labels.size,
        float(np.mean(all_train_targets, dtype=np.float64)),
        dtype=np.float64,
    )

    true_gallery_density = nonself_density(
        gallery.embeddings,
        gallery.example_ids,
        k=K,
        block_size=block_size,
    )
    diagnostics = density_diagnostics(predicted_potential, true_gallery_density)
    zero_potential = np.zeros(gallery.labels.size, dtype=np.float64)
    random_generator = np.random.Generator(np.random.PCG64(RANDOM_DIRECTION_SEED))
    predicted_mean = float(np.mean(predicted_potential, dtype=np.float64))
    predicted_std = float(np.std(predicted_potential, dtype=np.float64))
    random_potentials: dict[str, np.ndarray] = {}
    for index in range(RANDOM_DIRECTION_COUNT):
        direction = random_generator.normal(size=gallery.embeddings.shape[1])
        direction /= np.linalg.norm(direction)
        projection = gallery.embeddings.astype(np.float64) @ direction
        projection_mean = float(np.mean(projection, dtype=np.float64))
        projection_std = float(np.std(projection, dtype=np.float64))
        if projection_std == 0.0 or predicted_std == 0.0:
            scaled = np.full(gallery.labels.size, predicted_mean, dtype=np.float64)
        else:
            scaled = predicted_mean + predicted_std * (
                (projection - projection_mean) / projection_std
            )
        random_potentials[f"random_{index:02d}"] = np.asarray(scaled, dtype=np.float64)
    potentials = {
        "raw": zero_potential,
        "alsp": predicted_potential,
        "constant": constant_potential,
        "permuted": permuted_potential,
        "oracle": true_gallery_density,
        **random_potentials,
    }
    comparisons = compare_potentials(
        query.embeddings,
        query.labels,
        gallery.embeddings,
        gallery.labels,
        potentials,
        block_size,
    )
    arms = {name: _comparison_dict(comparisons[name]) for name in ARM_NAMES}
    random_gains = [
        comparisons[f"random_{index:02d}"].gain
        for index in range(RANDOM_DIRECTION_COUNT)
    ]
    random_null_p95 = float(
        np.quantile(np.asarray(random_gains, dtype=np.float64), 0.95, method="linear")
    )
    passed, predicates = decide_alsp(
        correlation=diagnostics["pearson"],
        alsp_gain=float(arms["alsp"]["gain"]),
        oracle_gain=float(arms["oracle"]["gain"]),
        alsp_p_value=float(arms["alsp"]["p_value"]),
        permuted_gain=float(arms["permuted"]["gain"]),
        random_null_p95=random_null_p95,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "inputs": {
            "train": {"path": str(train_path), "sha256": _sha256_path(train_path)},
            "query": {"path": str(query_path), "sha256": _sha256_path(query_path)},
            "gallery": {
                "path": str(gallery_path),
                "sha256": _sha256_path(gallery_path),
            },
            "checkpoint_sha256": train.checkpoint_sha256,
        },
        "configuration": {
            "k": K,
            "block_size": block_size,
            "fit_fraction": FIT_FRACTION,
            "ridge_grid": list(RIDGE_GRID),
            "permutation_seed": PERMUTATION_SEED,
            "random_direction_seed": RANDOM_DIRECTION_SEED,
            "random_direction_count": RANDOM_DIRECTION_COUNT,
        },
        "split": {
            "fit_label_count": int(fit_labels.size),
            "validation_label_count": int(validation_labels.size),
            "fit_labels_sha256": _label_sha256(fit_labels),
            "validation_labels_sha256": _label_sha256(validation_labels),
            "fit_row_count": int(np.sum(fit_mask)),
            "validation_row_count": int(np.sum(validation_mask)),
        },
        "selection": {
            "selected_ridge_lambda": selected_model.ridge_lambda,
            "grid": grid_rows,
        },
        "fit": {
            "row_count": int(train.labels.size),
            "target_mean": float(np.mean(all_train_targets, dtype=np.float64)),
            "target_std": float(np.std(all_train_targets, dtype=np.float64)),
            "weights": model.weights.tolist(),
            "weights_sha256": float64_sha256(model.weights),
            "intercept": model.intercept,
        },
        "test": {
            "density_diagnostics": diagnostics,
            "arms": arms,
            "random_direction_null": {
                "gains": random_gains,
                "linear_p95": random_null_p95,
            },
        },
        "decision": {"predicates": predicates, "passes_falsifier": passed},
    }
    return validate_alsp_report(report)


def _require_keys(value: object, keys: tuple[str, ...], name: str) -> dict[str, Any]:
    if type(value) is not dict or list(value) != list(keys):
        raise ValueError(f"{name} schema differs")
    return value


def _require_float(value: object, name: str) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite float")
    return value


def validate_alsp_report(value: object) -> dict[str, Any]:
    """Validate the exact persisted ALSP report and all decision relations."""

    report = _require_keys(
        value,
        (
            "schema_version",
            "inputs",
            "configuration",
            "split",
            "selection",
            "fit",
            "test",
            "decision",
        ),
        "report",
    )
    if report["schema_version"] != SCHEMA_VERSION:
        raise ValueError("schema_version differs")
    inputs = _require_keys(
        report["inputs"], ("train", "query", "gallery", "checkpoint_sha256"), "inputs"
    )
    for name in ("train", "query", "gallery"):
        row = _require_keys(inputs[name], ("path", "sha256"), f"inputs.{name}")
        if type(row["path"]) is not str or not row["path"]:
            raise ValueError(f"inputs.{name}.path differs")
        if type(row["sha256"]) is not str or len(row["sha256"]) != 64:
            raise ValueError(f"inputs.{name}.sha256 differs")
    if type(inputs["checkpoint_sha256"]) is not str or len(
        inputs["checkpoint_sha256"]
    ) != 64:
        raise ValueError("inputs.checkpoint_sha256 differs")
    configuration = _require_keys(
        report["configuration"],
        (
            "k",
            "block_size",
            "fit_fraction",
            "ridge_grid",
            "permutation_seed",
            "random_direction_seed",
            "random_direction_count",
        ),
        "configuration",
    )
    if configuration["k"] != K or type(configuration["block_size"]) is not int:
        raise ValueError("configuration differs")
    if (
        configuration["fit_fraction"] != FIT_FRACTION
        or configuration["ridge_grid"] != list(RIDGE_GRID)
        or configuration["permutation_seed"] != PERMUTATION_SEED
        or configuration["random_direction_seed"] != RANDOM_DIRECTION_SEED
        or configuration["random_direction_count"] != RANDOM_DIRECTION_COUNT
    ):
        raise ValueError("configuration differs")
    split = _require_keys(
        report["split"],
        (
            "fit_label_count",
            "validation_label_count",
            "fit_labels_sha256",
            "validation_labels_sha256",
            "fit_row_count",
            "validation_row_count",
        ),
        "split",
    )
    for name in (
        "fit_label_count",
        "validation_label_count",
        "fit_row_count",
        "validation_row_count",
    ):
        if type(split[name]) is not int or split[name] <= 0:
            raise ValueError(f"split.{name} differs")
    for name in ("fit_labels_sha256", "validation_labels_sha256"):
        if type(split[name]) is not str or len(split[name]) != 64:
            raise ValueError(f"split.{name} differs")
    selection = _require_keys(
        report["selection"], ("selected_ridge_lambda", "grid"), "selection"
    )
    selected_lambda = _require_float(
        selection["selected_ridge_lambda"], "selection.selected_ridge_lambda"
    )
    if selected_lambda not in RIDGE_GRID or type(selection["grid"]) is not list:
        raise ValueError("selection differs")
    if len(selection["grid"]) != len(RIDGE_GRID):
        raise ValueError("selection.grid differs")
    observed_best: float | None = None
    observed_selected: float | None = None
    for expected_lambda, item in zip(RIDGE_GRID, selection["grid"], strict=True):
        row = _require_keys(item, ("ridge_lambda", "validation_mse"), "selection.grid")
        if row["ridge_lambda"] != expected_lambda:
            raise ValueError("selection.grid ridge order differs")
        mse = _require_float(row["validation_mse"], "selection.grid.validation_mse")
        if mse < 0.0:
            raise ValueError("selection.grid validation_mse differs")
        if observed_best is None or mse < observed_best:
            observed_best = mse
            observed_selected = expected_lambda
    if selected_lambda != observed_selected:
        raise ValueError("selection selected lambda differs")
    fit = _require_keys(
        report["fit"],
        ("row_count", "target_mean", "target_std", "weights", "weights_sha256", "intercept"),
        "fit",
    )
    if type(fit["row_count"]) is not int or fit["row_count"] <= 0:
        raise ValueError("fit.row_count differs")
    _require_float(fit["target_mean"], "fit.target_mean")
    if _require_float(fit["target_std"], "fit.target_std") < 0.0:
        raise ValueError("fit.target_std differs")
    if type(fit["weights"]) is not list or not fit["weights"]:
        raise ValueError("fit.weights differs")
    weights = np.asarray([_require_float(item, "fit.weights") for item in fit["weights"]])
    if fit["weights_sha256"] != float64_sha256(weights):
        raise ValueError("fit.weights_sha256 differs")
    _require_float(fit["intercept"], "fit.intercept")
    test = _require_keys(
        report["test"],
        ("density_diagnostics", "arms", "random_direction_null"),
        "test",
    )
    diagnostics = _require_keys(
        test["density_diagnostics"], ("pearson", "spearman", "mse"), "test.diagnostics"
    )
    for name in diagnostics:
        _require_float(diagnostics[name], f"test.diagnostics.{name}")
    if diagnostics["mse"] < 0.0:
        raise ValueError("test.diagnostics.mse differs")
    arms = _require_keys(test["arms"], ARM_NAMES, "test.arms")
    for arm_name in ARM_NAMES:
        arm = _require_keys(arms[arm_name], COMPARISON_KEYS, f"test.arms.{arm_name}")
        for name in ("raw_recall", "corrected_recall", "gain", "p_value"):
            _require_float(arm[name], f"test.arms.{arm_name}.{name}")
        for name in ("wrong_to_right", "right_to_wrong"):
            if type(arm[name]) is not int or arm[name] < 0:
                raise ValueError(f"test.arms.{arm_name}.{name} differs")
        if arm["gain"] != arm["corrected_recall"] - arm["raw_recall"]:
            raise ValueError(f"test.arms.{arm_name}.gain differs")
    if arms["constant"] != arms["raw"]:
        raise ValueError("constant arm is not a ranking identity")
    random_null = _require_keys(
        test["random_direction_null"], ("gains", "linear_p95"), "test.random_null"
    )
    if type(random_null["gains"]) is not list or len(
        random_null["gains"]
    ) != RANDOM_DIRECTION_COUNT:
        raise ValueError("test.random_null.gains differs")
    random_gains = np.asarray(
        [_require_float(item, "test.random_null.gains") for item in random_null["gains"]],
        dtype=np.float64,
    )
    expected_random_p95 = float(np.quantile(random_gains, 0.95, method="linear"))
    if random_null["linear_p95"] != expected_random_p95:
        raise ValueError("test.random_null.linear_p95 differs")
    decision = _require_keys(
        report["decision"], ("predicates", "passes_falsifier"), "decision"
    )
    expected_passed, expected_predicates = decide_alsp(
        correlation=diagnostics["pearson"],
        alsp_gain=arms["alsp"]["gain"],
        oracle_gain=arms["oracle"]["gain"],
        alsp_p_value=arms["alsp"]["p_value"],
        permuted_gain=arms["permuted"]["gain"],
        random_null_p95=expected_random_p95,
    )
    if decision["predicates"] != expected_predicates:
        raise ValueError("decision predicates differ")
    if type(decision["passes_falsifier"]) is not bool or decision[
        "passes_falsifier"
    ] is not expected_passed:
        raise ValueError("decision passes_falsifier differs")
    return report


def write_json_atomic(
    path: Path,
    payload: dict[str, Any],
    *,
    validator: Callable[[object], object],
) -> None:
    """Publish validated canonical JSON without replacing an existing path."""

    if not path.parent.is_dir() or path.is_symlink():
        raise ValueError("output parent must exist and output must not be a symlink")
    validator(payload)
    encoded = (
        json.dumps(payload, sort_keys=False, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor: int | None = None
    owned_stat: os.stat_result | None = None
    published = False
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        owned_stat = temporary.stat(follow_symlinks=False)
        os.link(temporary, path)
        published = True
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        persisted = json.loads(path.read_text(encoding="utf-8"))
        validator(persisted)
    except Exception:
        if published and owned_stat is not None:
            try:
                destination_stat = path.stat(follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                if (
                    stat.S_ISREG(destination_stat.st_mode)
                    and destination_stat.st_dev == owned_stat.st_dev
                    and destination_stat.st_ino == owned_stat.st_ino
                ):
                    path.unlink()
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        temporary.unlink(missing_ok=True)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        arguments = _parse_args(argv)
        report = build_alsp_report(
            arguments.train,
            arguments.query,
            arguments.gallery,
            block_size=arguments.block_size,
        )
        write_json_atomic(
            arguments.output, report, validator=validate_alsp_report
        )
    except Exception as error:
        print(f"structural failure: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
