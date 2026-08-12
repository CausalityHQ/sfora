#!/usr/bin/env python3
"""Run the frozen train-only LE-IDGP mechanism falsifier."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from sfora.idgp import (
    ARM_ORDER,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    COHORT_LABELS,
    CONTROL_RANDOM_SEED,
    CONTROL_SHUFFLE_SEED,
    LOCAL_K,
    ROWS_PER_LABEL,
    cluster_bootstrap,
    decide_idgp,
    evaluate_idgp,
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
THREAD_ENVIRONMENT = [
    "cuda_visible_devices",
    "omp_num_threads",
    "mkl_num_threads",
    "openblas_num_threads",
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_train_archive(path: Path) -> dict[str, Any]:
    """Load only the exact regular train archive."""

    if path.is_symlink() or not path.is_file():
        raise ValueError("train input must be a regular nonsymlink file")
    digest = _sha256(path)
    if digest != EXPECTED_TRAIN_SHA256:
        raise ValueError("train input SHA-256 differs")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != EXPECTED_ARCHIVE_KEYS:
            raise ValueError("train archive schema differs")
        payload = {name: np.array(archive[name], copy=True) for name in EXPECTED_ARCHIVE_KEYS}
    embeddings = payload["embeddings"]
    labels = payload["labels"]
    ids = payload["example_ids"]
    if payload["split"].shape != () or payload["split"].item() != "train":
        raise ValueError("train archive split differs")
    if embeddings.ndim != 2 or embeddings.dtype != np.dtype(np.float32):
        raise ValueError("train embeddings must be a float32 matrix")
    if labels.shape != (embeddings.shape[0],) or labels.dtype != np.dtype(np.int64):
        raise ValueError("train labels must be an aligned int64 vector")
    if ids.shape != (embeddings.shape[0],) or ids.dtype.kind != "U":
        raise ValueError("train example IDs must be aligned Unicode values")
    if np.unique(ids).size != ids.size or any(not value for value in ids.tolist()):
        raise ValueError("train example IDs must be unique and nonempty")
    if not np.isfinite(embeddings).all():
        raise ValueError("train embeddings must be finite")
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1)
    if np.max(np.abs(norms - 1.0)) > 5e-5:
        raise ValueError("train embedding rows must be unit normalized")
    payload["sha256"] = digest
    return payload


def _cohort_record(evaluation: Any) -> dict[str, Any]:
    primary_records: list[dict[str, Any]] = []
    for row in np.flatnonzero(evaluation.primary_mask).tolist():
        primary_records.append(
            {
                "example_id": str(evaluation.example_ids[row]),
                "label": int(evaluation.labels[row]),
                "zero": float(evaluation.margin_changes["zero"][row]),
                "le_idgp": float(evaluation.margin_changes["le_idgp"][row]),
                "shuffled_local": float(evaluation.margin_changes["shuffled_local"][row]),
                "random_tangent": float(evaluation.margin_changes["random_tangent"][row]),
                "global_centering": float(evaluation.margin_changes["global_centering"][row]),
                "two_sided": float(evaluation.margin_changes["two_sided"][row]),
                "positive_zero": float(evaluation.positive_similarity_changes["zero"][row]),
                "positive_le_idgp": float(evaluation.positive_similarity_changes["le_idgp"][row]),
                "analytic_density_reduction": float(evaluation.analytic_density_reduction[row]),
            }
        )
    ordered_ids = "\0".join(evaluation.example_ids.tolist()).encode("utf-8")
    return {
        "fold": evaluation.fold,
        "index": evaluation.index,
        "ordered_sha256": hashlib.sha256(ordered_ids).hexdigest(),
        "eligible_rows": int(evaluation.labels.size),
        "conflicting_rows": int(
            np.count_nonzero((evaluation.conflict_dots < 0.0) & ~evaluation.skipped_mask)
        ),
        "skipped_rows": int(np.count_nonzero(evaluation.skipped_mask)),
        "primary_records": primary_records,
        "diagnostics": {
            "local_norm_mean": float(evaluation.local_norms.mean()),
            "global_norm_mean": float(evaluation.global_norms.mean()),
            "local_global_cosine_mean": evaluation.local_global_cosine_mean,
            "collective_idgp_advantage": evaluation.collective_idgp_advantage,
            "reference_conflict_fraction": evaluation.reference_conflict_fraction,
            "reference_cosine_mean": evaluation.reference_cosine_mean,
        },
    }


def build_report(path: Path) -> dict[str, Any]:
    """Evaluate the registered train archive and build an exact JSON-ready report."""

    payload = load_train_archive(path)
    evaluation = evaluate_idgp(payload["embeddings"], payload["labels"], payload["example_ids"])
    summary = evaluation.summary
    cohorts = [_cohort_record(value) for value in evaluation.evaluations]
    diagnostics = {
        "local_norm_mean": float(np.mean([c["diagnostics"]["local_norm_mean"] for c in cohorts])),
        "global_norm_mean": float(np.mean([c["diagnostics"]["global_norm_mean"] for c in cohorts])),
        "local_global_cosine_mean": float(
            np.mean([c["diagnostics"]["local_global_cosine_mean"] for c in cohorts])
        ),
        "collective_idgp_advantage_mean": float(
            np.mean([c["diagnostics"]["collective_idgp_advantage"] for c in cohorts])
        ),
        "reference_conflict_fraction_mean": float(
            np.mean([c["diagnostics"]["reference_conflict_fraction"] for c in cohorts])
        ),
        "reference_cosine_mean": float(
            np.mean([c["diagnostics"]["reference_cosine_mean"] for c in cohorts])
        ),
    }
    return {
        "schema_version": "inshop-le-idgp-train-v1",
        "input": {
            "path": str(path),
            "sha256": payload["sha256"],
            "checkpoint_sha256": str(payload["checkpoint_sha256"].item()),
            "report_sha256": str(payload["report_sha256"].item()),
            "rows": int(payload["embeddings"].shape[0]),
            "dimension": int(payload["embeddings"].shape[1]),
        },
        "environment": {
            "numpy_version": str(np.__version__),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
            "mkl_num_threads": os.environ.get("MKL_NUM_THREADS"),
            "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        },
        "configuration": {
            "cohort_labels": COHORT_LABELS,
            "rows_per_label": ROWS_PER_LABEL,
            "cohort_rows": COHORT_LABELS * ROWS_PER_LABEL,
            "local_k": LOCAL_K,
            "epsilon": 0.01,
            "tangent_cutoff": 1e-8,
            "shuffle_seed": CONTROL_SHUFFLE_SEED,
            "random_seed": CONTROL_RANDOM_SEED,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        "cohorts": cohorts,
        "pooled": {
            "coverage": summary["coverage"],
            "fold_mean_advantages": summary["fold_mean_advantages"],
            "metrics": summary["pooled"],
        },
        "controls": {"arm_order": list(ARM_ORDER)},
        "bootstrap": summary["bootstrap"],
        "diagnostics": diagnostics,
        "decision": {"predicates": list(evaluation.predicates), "status": evaluation.status},
    }


def _bootstrap_json(values: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, Any]:
    summary = cluster_bootstrap(values, labels, seed=seed, replicates=BOOTSTRAP_REPLICATES)
    return {
        "mean": summary.mean,
        "standard_error": summary.standard_error,
        "lower_bound": summary.lower_bound,
        "mde": summary.mde,
        "label_count": summary.label_count,
        "median_rows_per_label": summary.median_rows_per_label,
        "replicate_sha256": summary.replicate_sha256,
    }


def _recompute_pooled(value: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    records = [record for cohort in value["cohorts"] for record in cohort["primary_records"]]
    eligible = sum(cohort["eligible_rows"] for cohort in value["cohorts"])
    conflicting = sum(cohort["conflicting_rows"] for cohort in value["cohorts"])
    if not records:
        empty_hash = hashlib.sha256(b"").hexdigest()
        bootstrap = {
            name: {
                "mean": 0.0,
                "standard_error": 0.0,
                "lower_bound": 0.0,
                "mde": 0.0,
                "label_count": 0,
                "median_rows_per_label": 0.0,
                "replicate_sha256": empty_hash,
            }
            for name in ("raw", "shuffled", "random", "global", "positive")
        }
        return (
            {
                "coverage": {
                    "eligible_rows": eligible,
                    "conflicting_rows": conflicting,
                    "conflict_fraction": conflicting / eligible,
                    "primary_rows": 0,
                    "primary_labels": 0,
                    "folds_with_primary": 0,
                },
                "fold_mean_advantages": [],
                "metrics": {
                    "raw_advantage_mean": 0.0,
                    "raw_advantage_lower": 0.0,
                    "shuffled_contrast_lower": 0.0,
                    "random_contrast_lower": 0.0,
                    "global_contrast_lower": 0.0,
                    "zero_abs_margin_change_median": 0.0,
                    "positive_similarity_mean": 0.0,
                    "positive_similarity_lower": 0.0,
                    "le_minus_two_sided_mean": 0.0,
                },
            },
            bootstrap,
        )
    labels = np.asarray([record["label"] for record in records], dtype=np.int64)
    zero = np.asarray([record["zero"] for record in records], dtype=np.float64)
    le = np.asarray([record["le_idgp"] for record in records], dtype=np.float64)
    values = {
        "raw": le - zero,
        "shuffled": le
        - np.asarray([record["shuffled_local"] for record in records], dtype=np.float64),
        "random": le
        - np.asarray([record["random_tangent"] for record in records], dtype=np.float64),
        "global": le
        - np.asarray([record["global_centering"] for record in records], dtype=np.float64),
        "positive": np.asarray(
            [record["positive_le_idgp"] - record["positive_zero"] for record in records],
            dtype=np.float64,
        ),
    }
    bootstrap = {
        name: _bootstrap_json(array, labels, BOOTSTRAP_SEED + offset)
        for offset, (name, array) in enumerate(values.items())
    }
    fold_means = []
    for fold in sorted({cohort["fold"] for cohort in value["cohorts"]}):
        fold_records = [
            record
            for cohort in value["cohorts"]
            if cohort["fold"] == fold
            for record in cohort["primary_records"]
        ]
        if fold_records:
            fold_means.append(
                float(np.mean([record["le_idgp"] - record["zero"] for record in fold_records]))
            )
    raw = values["raw"]
    positive = values["positive"]
    two = le - np.asarray([record["two_sided"] for record in records], dtype=np.float64)
    return (
        {
            "coverage": {
                "eligible_rows": eligible,
                "conflicting_rows": conflicting,
                "conflict_fraction": conflicting / eligible,
                "primary_rows": len(records),
                "primary_labels": int(np.unique(labels).size),
                "folds_with_primary": len(fold_means),
            },
            "fold_mean_advantages": fold_means,
            "metrics": {
                "raw_advantage_mean": float(raw.mean()),
                "raw_advantage_lower": bootstrap["raw"]["lower_bound"],
                "shuffled_contrast_lower": bootstrap["shuffled"]["lower_bound"],
                "random_contrast_lower": bootstrap["random"]["lower_bound"],
                "global_contrast_lower": bootstrap["global"]["lower_bound"],
                "zero_abs_margin_change_median": float(np.median(np.abs(zero))),
                "positive_similarity_mean": float(positive.mean()),
                "positive_similarity_lower": bootstrap["positive"]["lower_bound"],
                "le_minus_two_sided_mean": float(two.mean()),
            },
        },
        bootstrap,
    )


def validate_report(value: object) -> dict[str, Any]:
    """Strictly validate report structure and all recomputable decisions."""

    expected_top = [
        "schema_version",
        "input",
        "environment",
        "configuration",
        "cohorts",
        "pooled",
        "controls",
        "bootstrap",
        "diagnostics",
        "decision",
    ]
    if not isinstance(value, dict) or list(value) != expected_top:
        raise ValueError("report schema differs")
    if value["schema_version"] != "inshop-le-idgp-train-v1":
        raise ValueError("report version differs")
    if list(value["input"]) != [
        "path",
        "sha256",
        "checkpoint_sha256",
        "report_sha256",
        "rows",
        "dimension",
    ]:
        raise ValueError("report input schema differs")
    environment = value["environment"]
    if list(environment) != ["numpy_version", *THREAD_ENVIRONMENT] or [
        environment[name] for name in THREAD_ENVIRONMENT
    ] != ["", "1", "1", "1"]:
        raise ValueError("report environment differs")
    expected_configuration = {
        "cohort_labels": 45,
        "rows_per_label": 4,
        "cohort_rows": 180,
        "local_k": 50,
        "epsilon": 0.01,
        "tangent_cutoff": 1e-8,
        "shuffle_seed": 20260812,
        "random_seed": 20260814,
        "bootstrap_seed": 20260813,
        "bootstrap_replicates": 10_000,
    }
    if (
        list(value["configuration"]) != list(expected_configuration)
        or value["configuration"] != expected_configuration
    ):
        raise ValueError("report configuration differs")
    if list(value["controls"]) != ["arm_order"] or value["controls"] != {
        "arm_order": list(ARM_ORDER)
    }:
        raise ValueError("report controls differ")
    if not isinstance(value["cohorts"], list) or not value["cohorts"]:
        raise ValueError("report cohorts differ")
    cohort_keys = [
        "fold",
        "index",
        "ordered_sha256",
        "eligible_rows",
        "conflicting_rows",
        "skipped_rows",
        "primary_records",
        "diagnostics",
    ]
    diagnostic_keys = [
        "local_norm_mean",
        "global_norm_mean",
        "local_global_cosine_mean",
        "collective_idgp_advantage",
        "reference_conflict_fraction",
        "reference_cosine_mean",
    ]
    primary_keys = [
        "example_id",
        "label",
        "zero",
        "le_idgp",
        "shuffled_local",
        "random_tangent",
        "global_centering",
        "two_sided",
        "positive_zero",
        "positive_le_idgp",
        "analytic_density_reduction",
    ]
    previous_pair: tuple[int, int] | None = None
    primary_ids: set[str] = set()
    for cohort in value["cohorts"]:
        if not isinstance(cohort, dict) or list(cohort) != cohort_keys:
            raise ValueError("report cohort schema differs")
        if type(cohort["fold"]) is not int or type(cohort["index"]) is not int:
            raise ValueError("report cohort coordinates differ")
        pair = (cohort["fold"], cohort["index"])
        if previous_pair is not None and pair <= previous_pair:
            raise ValueError("report cohort order differs")
        previous_pair = pair
        if (
            type(cohort["ordered_sha256"]) is not str
            or len(cohort["ordered_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in cohort["ordered_sha256"])
        ):
            raise ValueError("report cohort hash differs")
        if (
            type(cohort["eligible_rows"]) is not int
            or cohort["eligible_rows"] != 180
            or type(cohort["conflicting_rows"]) is not int
            or not 0 <= cohort["conflicting_rows"] <= 180
            or type(cohort["skipped_rows"]) is not int
            or not 0 <= cohort["skipped_rows"] <= 180
        ):
            raise ValueError("report cohort counts differ")
        if not isinstance(cohort["primary_records"], list):
            raise ValueError("report cohort primary records differ")
        for record in cohort["primary_records"]:
            if not isinstance(record, dict) or list(record) != primary_keys:
                raise ValueError("report cohort primary schema differs")
            if (
                type(record["example_id"]) is not str
                or not record["example_id"]
                or record["example_id"] in primary_ids
                or type(record["label"]) is not int
            ):
                raise ValueError("report cohort primary identity differs")
            primary_ids.add(record["example_id"])
            if any(type(record[key]) is not float for key in primary_keys[2:]):
                raise ValueError("report cohort primary scalar types differ")
        diagnostics = cohort["diagnostics"]
        if not isinstance(diagnostics, dict) or list(diagnostics) != diagnostic_keys:
            raise ValueError("report cohort diagnostics schema differs")
        if any(type(diagnostics[key]) is not float for key in diagnostic_keys):
            raise ValueError("report cohort diagnostics types differ")
    recomputed_pooled, recomputed_bootstrap = _recompute_pooled(value)
    if value["pooled"] != recomputed_pooled:
        raise ValueError("report pooled values differ")
    if value["bootstrap"] != recomputed_bootstrap:
        raise ValueError("report bootstrap values differ")
    expected_diagnostics = {
        "local_norm_mean": float(
            np.mean([cohort["diagnostics"]["local_norm_mean"] for cohort in value["cohorts"]])
        ),
        "global_norm_mean": float(
            np.mean([cohort["diagnostics"]["global_norm_mean"] for cohort in value["cohorts"]])
        ),
        "local_global_cosine_mean": float(
            np.mean(
                [cohort["diagnostics"]["local_global_cosine_mean"] for cohort in value["cohorts"]]
            )
        ),
        "collective_idgp_advantage_mean": float(
            np.mean(
                [cohort["diagnostics"]["collective_idgp_advantage"] for cohort in value["cohorts"]]
            )
        ),
        "reference_conflict_fraction_mean": float(
            np.mean(
                [
                    cohort["diagnostics"]["reference_conflict_fraction"]
                    for cohort in value["cohorts"]
                ]
            )
        ),
        "reference_cosine_mean": float(
            np.mean([cohort["diagnostics"]["reference_cosine_mean"] for cohort in value["cohorts"]])
        ),
    }
    if (
        list(value["diagnostics"]) != list(expected_diagnostics)
        or value["diagnostics"] != expected_diagnostics
    ):
        raise ValueError("report diagnostics differ")
    passed, predicates = decide_idgp(
        {
            "coverage": value["pooled"]["coverage"],
            "fold_mean_advantages": value["pooled"]["fold_mean_advantages"],
            "pooled": value["pooled"]["metrics"],
        }
    )
    if value["decision"] != {
        "predicates": list(predicates),
        "status": "PASS" if passed else "KILL",
    }:
        raise ValueError("report decision differs")

    def check(node: object) -> None:
        if type(node) is float and not np.isfinite(node):
            raise ValueError("report contains nonfinite float")
        if isinstance(node, dict):
            for child in node.values():
                check(child)
        elif isinstance(node, list):
            for child in node:
                check(child)

    check(value)
    return value


def write_report(path: Path, report: dict[str, Any]) -> None:
    """Publish canonical JSON without replacement and strictly reload it."""

    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise FileExistsError(path)
    temp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    owned_temp = False
    published_inode: tuple[int, int] | None = None
    try:
        data = (json.dumps(report, ensure_ascii=False, separators=(",", ":")) + "\n").encode()
        with temp.open("xb") as handle:
            owned_temp = True
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp, path)
        stat = path.stat(follow_symlinks=False)
        published_inode = (stat.st_dev, stat.st_ino)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        validate_report(json.loads(path.read_text()))
    except Exception:
        if published_inode is not None and path.exists() and not path.is_symlink():
            current = path.stat(follow_symlinks=False)
            if (current.st_dev, current.st_ino) == published_inode:
                path.unlink()
        raise
    finally:
        if owned_temp and temp.exists() and not temp.is_symlink():
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
    return 0 if report["decision"]["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
