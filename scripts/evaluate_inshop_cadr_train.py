#!/usr/bin/env python3
"""Run the frozen train-only CADR model-selection gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from sfora.cadr import (
    balanced_log_loss,
    build_pairs,
    calibrated_log_loss,
    decide_train_gate,
    fit_score_calibration,
    select_lambda,
    split_labels,
)

EXPECTED_TRAIN_SHA256 = "67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea"
LAMBDA_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0)
TAU_FACTORS = (0.1, 0.3, 1.0, 3.0, 10.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha(value: np.ndarray, dtype: str) -> str:
    return hashlib.sha256(np.asarray(value, dtype=dtype).tobytes(order="C")).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file() or _sha256(path) != EXPECTED_TRAIN_SHA256:
        raise ValueError("train archive path or SHA differs")
    with np.load(path, allow_pickle=False) as archive:
        expected = [
            "embeddings",
            "labels",
            "example_ids",
            "source_paths",
            "artifact_selection",
            "split",
            "checkpoint_sha256",
            "report_sha256",
        ]
        if archive.files != expected or archive["split"].item() != "train":
            raise ValueError("train archive schema differs")
        return {name: np.array(archive[name], copy=True) for name in expected}


def _within_variance(embeddings: np.ndarray, labels: np.ndarray, allowed: np.ndarray) -> np.ndarray:
    total = np.zeros(embeddings.shape[1], dtype=np.float64)
    count = 0
    for label in allowed.tolist():
        rows = embeddings[labels == label].astype(np.float64)
        centered = rows - rows.mean(axis=0)
        total += np.sum(centered * centered, axis=0)
        count += rows.shape[0]
    return total / count


def build_train_report(path: Path, *, k: int = 10, block_size: int = 256) -> dict[str, Any]:
    payload = _load(path)
    embeddings = payload["embeddings"]
    labels = payload["labels"]
    ids = payload["example_ids"]
    split = split_labels(labels)
    fit_pairs = build_pairs(embeddings, labels, ids, split.fit_labels, k=k, block_size=block_size)
    val_pairs = build_pairs(
        embeddings, labels, ids, split.validation_labels, k=k, block_size=block_size
    )
    selected, records = select_lambda(
        fit_pairs.features, fit_pairs.targets, val_pairs.features, val_pairs.targets, LAMBDA_GRID
    )
    raw_fit = fit_pairs.features.sum(axis=1)
    raw_val = val_pairs.features.sum(axis=1)
    platt = fit_score_calibration(raw_fit, fit_pairs.targets)
    platt_loss = calibrated_log_loss(raw_val, val_pairs.targets, platt)
    within = _within_variance(embeddings, labels, split.fit_labels)
    median = float(np.median(within))
    wccn_records: list[dict[str, float]] = []
    for factor in TAU_FACTORS:
        weights = 1.0 / (within + factor * median)
        weights /= weights.mean()
        calibration = fit_score_calibration(fit_pairs.features @ weights, fit_pairs.targets)
        loss = calibrated_log_loss(val_pairs.features @ weights, val_pairs.targets, calibration)
        wccn_records.append({"tau_factor": factor, "validation_loss": loss})
    best_wccn = min(wccn_records, key=lambda item: (item["validation_loss"], -item["tau_factor"]))
    selected_lambda = records[
        min(
            range(len(records)),
            key=lambda i: (records[i].validation_loss, -records[i].lambda_value),
        )
    ].lambda_value
    cadr_loss = balanced_log_loss(val_pairs.features, val_pairs.targets, selected)
    passed, predicates = decide_train_gate(
        selected_lambda, selected.weights, cadr_loss, platt_loss, best_wccn["validation_loss"]
    )
    return {
        "schema_version": "inshop-cadr-train-v1",
        "input": {
            "path": str(path),
            "sha256": _sha256(path),
            "checkpoint_sha256": str(payload["checkpoint_sha256"].item()),
            "rows": int(labels.size),
            "dimension": int(embeddings.shape[1]),
        },
        "environment": {
            "numpy_version": str(np.__version__),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "configuration": {"k": k, "block_size": block_size, "lambda_grid": list(LAMBDA_GRID)},
        "split": {
            "fit_labels": int(split.fit_labels.size),
            "validation_labels": int(split.validation_labels.size),
            "fit_labels_sha256": _array_sha(split.fit_labels, "<i8"),
            "validation_labels_sha256": _array_sha(split.validation_labels, "<i8"),
        },
        "pairs": {
            "fit_positive": int(fit_pairs.positive_indices.shape[0]),
            "fit_negative": int(fit_pairs.negative_indices.shape[0]),
            "validation_positive": int(val_pairs.positive_indices.shape[0]),
            "validation_negative": int(val_pairs.negative_indices.shape[0]),
        },
        "cadr": {
            "selected_lambda": selected_lambda,
            "validation_loss": cadr_loss,
            "weights_sha256": _array_sha(selected.weights, "<f8"),
            "weights_mean": float(selected.weights.mean()),
            "weights_std": float(selected.weights.std()),
            "records": [record.__dict__ for record in records],
        },
        "controls": {
            "platt_validation_loss": platt_loss,
            "wccn": wccn_records,
            "selected_wccn_validation_loss": best_wccn["validation_loss"],
        },
        "decision": {"predicates": predicates, "status": "PASS" if passed else "KILL"},
    }


def validate_train_report(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or list(value) != [
        "schema_version",
        "input",
        "environment",
        "configuration",
        "split",
        "pairs",
        "cadr",
        "controls",
        "decision",
    ]:
        raise ValueError("report schema differs")
    if value["schema_version"] != "inshop-cadr-train-v1":
        raise ValueError("schema version differs")
    environment = value["environment"]
    if list(environment) != [
        "numpy_version",
        "cuda_visible_devices",
        "omp_num_threads",
        "mkl_num_threads",
        "openblas_num_threads",
    ] or [
        environment["cuda_visible_devices"],
        environment["omp_num_threads"],
        environment["mkl_num_threads"],
        environment["openblas_num_threads"],
    ] != ["", "1", "1", "1"]:
        raise ValueError("environment differs")
    configuration = value["configuration"]
    if configuration["lambda_grid"] != list(LAMBDA_GRID):
        raise ValueError("lambda grid differs")
    cadr = value["cadr"]
    records = cadr["records"]
    if len(records) != len(LAMBDA_GRID) or [r["lambda_value"] for r in records] != list(
        LAMBDA_GRID
    ):
        raise ValueError("lambda records differ")
    best_index = min(
        range(len(records)),
        key=lambda i: (records[i]["validation_loss"], -records[i]["lambda_value"]),
    )
    if (
        cadr["selected_lambda"] != records[best_index]["lambda_value"]
        or cadr["validation_loss"] != records[best_index]["validation_loss"]
    ):
        raise ValueError("selected CADR record differs")
    controls = value["controls"]
    if [r["tau_factor"] for r in controls["wccn"]] != list(TAU_FACTORS):
        raise ValueError("WCCN records differ")
    selected_wccn = min(
        controls["wccn"], key=lambda item: (item["validation_loss"], -item["tau_factor"])
    )["validation_loss"]
    if controls["selected_wccn_validation_loss"] != selected_wccn:
        raise ValueError("selected WCCN differs")
    decision = value["decision"]
    expected_pass, expected_predicates = decide_train_gate(
        cadr["selected_lambda"],
        np.asarray(
            [cadr["weights_mean"] - cadr["weights_std"], cadr["weights_mean"] + cadr["weights_std"]]
        ),
        cadr["validation_loss"],
        controls["platt_validation_loss"],
        selected_wccn,
    )
    if decision["predicates"] != expected_predicates or decision["status"] != (
        "PASS" if expected_pass else "KILL"
    ):
        raise ValueError("decision differs")

    def check_finite(node: object) -> None:
        if type(node) is float and not np.isfinite(node):
            raise ValueError("nonfinite report value")
        if isinstance(node, dict):
            for child in node.values():
                check_finite(child)
        elif isinstance(node, list):
            for child in node:
                check_finite(child)

    check_finite(value)
    return value


def _write(path: Path, report: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    owned = False
    published = False
    try:
        data = (json.dumps(report, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        with temp.open("xb") as handle:
            owned = True
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
        published = True
        loaded = json.loads(path.read_text())
        validate_train_report(loaded)
    except Exception:
        if (
            published
            and path.exists()
            and temp.exists()
            and path.stat().st_ino == temp.stat().st_ino
        ):
            path.unlink()
        raise
    finally:
        if owned:
            temp.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = build_train_report(args.train)
        validate_train_report(report)
        _write(args.output, report)
    except Exception as error:
        print(f"structural failure: {error}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
