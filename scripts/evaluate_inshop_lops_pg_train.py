#!/usr/bin/env python3
"""Run the frozen train-only LOPS-PG confirmation on folds 1-3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from sfora.idgp import build_cohorts
from sfora.lops_pg import (
    ARM_ORDER,
    CohortEvaluation,
    decide_lops_pg,
    evaluate_cohort,
    summarize_evaluations,
)

EXPECTED_TRAIN_SHA256 = "67aa387c9815fd300e7db0da9f1a781e4b95191bc9db715e7d06850c9a7e6fea"
EXPECTED_ARCHIVE_KEYS = [
    "embeddings",
    "labels",
    "example_ids",
    "source_paths",
    "artifact_selection",
    "split",
    "checkpoint_sha256",
    "report_sha256",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, *, require_registered_hash: bool) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("train input must be a regular nonsymlink file")
    digest = _sha256(path)
    if require_registered_hash and digest != EXPECTED_TRAIN_SHA256:
        raise ValueError("train input SHA-256 differs")
    with np.load(path, allow_pickle=False) as archive:
        if require_registered_hash and archive.files != EXPECTED_ARCHIVE_KEYS:
            raise ValueError("train archive schema differs")
        required = {"embeddings", "labels", "example_ids"}
        if not required.issubset(archive.files):
            raise ValueError("train archive lacks required arrays")
        result = {name: np.array(archive[name], copy=True) for name in required}
    embeddings = result["embeddings"]
    labels = result["labels"]
    ids = result["example_ids"]
    if embeddings.ndim != 2 or embeddings.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError("train embeddings must be a float matrix")
    if labels.dtype != np.dtype(np.int64) or labels.shape != (embeddings.shape[0],):
        raise ValueError("train labels must be aligned int64")
    if ids.dtype.kind != "U" or ids.shape != labels.shape or np.unique(ids).size != ids.size:
        raise ValueError("train IDs must be aligned unique Unicode")
    if not np.isfinite(embeddings).all():
        raise ValueError("train embeddings must be finite")
    result["sha256"] = digest
    return result


def _cohort_record(value: CohortEvaluation) -> dict[str, Any]:
    return {
        "fold": value.fold,
        "index": value.index,
        "example_ids": value.example_ids.tolist(),
        "labels": value.labels.tolist(),
        "pre_margins": value.pre_margins.tolist(),
        "bottom_mask": value.bottom_mask.tolist(),
        "conflict_mask": value.conflict_mask.tolist(),
        "skipped_mask": value.skipped_mask.tolist(),
        "primary_mask": value.primary_mask.tolist(),
        "constraint_dots": value.constraint_dots.tolist(),
        "positive_tangents": value.positive_tangents.tolist(),
        "directions": {arm: value.directions[arm].tolist() for arm in ARM_ORDER},
        "margin_changes": {arm: value.margin_changes[arm].tolist() for arm in ARM_ORDER},
        "positive_similarity_changes": {
            arm: value.positive_similarity_changes[arm].tolist() for arm in ARM_ORDER
        },
    }


def _from_record(value: Mapping[str, Any]) -> CohortEvaluation:
    return CohortEvaluation(
        fold=value["fold"],
        index=value["index"],
        example_ids=np.asarray(value["example_ids"]),
        labels=np.asarray(value["labels"], dtype=np.int64),
        pre_margins=np.asarray(value["pre_margins"], dtype=np.float64),
        bottom_mask=np.asarray(value["bottom_mask"], dtype=np.bool_),
        conflict_mask=np.asarray(value["conflict_mask"], dtype=np.bool_),
        skipped_mask=np.asarray(value["skipped_mask"], dtype=np.bool_),
        primary_mask=np.asarray(value["primary_mask"], dtype=np.bool_),
        constraint_dots=np.asarray(value["constraint_dots"], dtype=np.float64),
        positive_tangents=np.asarray(value["positive_tangents"], dtype=np.float64),
        directions={
            arm: np.asarray(value["directions"][arm], dtype=np.float64) for arm in ARM_ORDER
        },
        margin_changes={
            arm: np.asarray(value["margin_changes"][arm], dtype=np.float64) for arm in ARM_ORDER
        },
        positive_similarity_changes={
            arm: np.asarray(value["positive_similarity_changes"][arm], dtype=np.float64)
            for arm in ARM_ORDER
        },
    )


def build_report(
    path: Path, *, require_registered_hash: bool = True, bootstrap_replicates: int = 10_000
) -> dict[str, Any]:
    payload = _load(path, require_registered_hash=require_registered_hash)
    embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
    norms = np.linalg.norm(embeddings, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("train embeddings contain zero rows")
    embeddings /= norms[:, None]
    labels = payload["labels"]
    ids = payload["example_ids"]
    evaluations = []
    for cohort in build_cohorts(labels, ids):
        if cohort.fold == 0:
            continue
        rows = cohort.row_indices
        evaluations.append(
            evaluate_cohort(
                embeddings[rows], labels[rows], ids[rows], fold=cohort.fold, index=cohort.index
            )
        )
    summary = summarize_evaluations(evaluations, bootstrap_replicates=bootstrap_replicates)
    predicates, status = decide_lops_pg(summary)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()
    return {
        "schema_version": "inshop-lops-pg-confirmation-v1",
        "status": status,
        "input": {"path": str(path), "sha256": payload["sha256"]},
        "environment": {"numpy_version": str(np.__version__), "device": "cpu"},
        "protocol": {
            "folds": [1, 2, 3],
            "discovery_fold": 0,
            "epsilon": 0.01,
            "source_commit": revision,
        },
        "cohorts": [{"fold": item.fold, "index": item.index} for item in evaluations],
        "rows": [_cohort_record(item) for item in evaluations],
        "summary": summary,
        "bootstrap": summary["bootstrap"],
        "predicates": list(predicates),
    }


def validate_report(value: object, *, bootstrap_replicates: int = 10_000) -> dict[str, Any]:
    if type(value) is not dict:
        raise ValueError("report must be an object")
    required = (
        "schema_version",
        "status",
        "input",
        "environment",
        "protocol",
        "cohorts",
        "rows",
        "summary",
        "bootstrap",
        "predicates",
    )
    if tuple(value) != required or value["schema_version"] != "inshop-lops-pg-confirmation-v1":
        raise ValueError("report schema differs")
    if value["protocol"]["folds"] != [1, 2, 3] or value["protocol"]["discovery_fold"] != 0:
        raise ValueError("confirmation fold protocol differs")
    evaluations = tuple(_from_record(record) for record in value["rows"])
    if any(item.fold not in (1, 2, 3) for item in evaluations):
        raise ValueError("discovery fold leaked into confirmation")
    recomputed = summarize_evaluations(evaluations, bootstrap_replicates=bootstrap_replicates)
    predicates, status = decide_lops_pg(recomputed)
    if value["summary"] != recomputed or value["bootstrap"] != recomputed["bootstrap"]:
        raise ValueError("report summaries differ")
    if value["predicates"] != list(predicates) or value["status"] != status:
        raise ValueError("report decision differs")
    if value["cohorts"] != [{"fold": item.fold, "index": item.index} for item in evaluations]:
        raise ValueError("report cohort registry differs")
    return value


def write_report(
    path: Path, report: dict[str, Any], *, validator: Callable[[object], object] = validate_report
) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    published = False
    try:
        data = (json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with temp.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
        published = True
        validator(json.loads(path.read_text()))
    except Exception:
        if published and path.exists() and not path.is_symlink():
            path.unlink()
        raise
    finally:
        if temp.exists() and not temp.is_symlink():
            temp.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = build_report(args.train)
    validate_report(report)
    write_report(args.output, report)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
