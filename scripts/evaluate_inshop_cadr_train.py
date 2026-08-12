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
        "environment": {"numpy_version": str(np.__version__)},
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
    decision = value["decision"]
    if decision["status"] not in {"PASS", "KILL"} or (decision["status"] == "PASS") != all(
        decision["predicates"].values()
    ):
        raise ValueError("decision differs")
    return value


def _write(path: Path, report: dict[str, Any]) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    owned = False
    try:
        data = (json.dumps(report, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
        with temp.open("xb") as handle:
            owned = True
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
        loaded = json.loads(path.read_text())
        validate_train_report(loaded)
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
