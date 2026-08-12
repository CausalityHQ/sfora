"""Strict bundle evaluation and no-clobber reporting for the Lorentz rider."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import secrets
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from sfora.lorentz_rider import (
    apply_frozen_pca,
    column_permutation_null,
    fit_frozen_pca,
    gromov_delta_rel,
    l0_subsample_indices,
    l1_decision,
    medoid_index,
    paired_identity_max_interval,
    pairwise_chord_distances,
    scale_for_target_median,
    score_control_blocks,
    spectrum_gaussian_null,
)
from sfora.unicom_audit_io import EmbeddingBundle
from sfora.unicom_retrieval_audit import l2_normalize, retrieval_metrics_from_score_chunks

REPORT_KEYS = (
    "schema_version",
    "input",
    "constants",
    "l0",
    "l1",
    "runtime",
    "decision",
    "timing",
)
ARRAY_KEYS = (
    "train_embeddings",
    "train_labels",
    "query_embeddings",
    "query_labels",
    "gallery_embeddings",
    "gallery_labels",
)
CONTROL_ORDER = (
    "lorentz",
    "pca_euclidean",
    "pca_cosine",
    "spatial_only",
    "power_1",
    "power_3",
)
DIMENSIONS = (8, 16, 32, 64)
TARGET_RADII = (0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
HEX = frozenset("0123456789abcdef")


def _exact_dict(value: object, keys: tuple[str, ...], *, name: str) -> dict:
    if type(value) is not dict or tuple(value) != keys:
        raise ValueError(f"{name} key order differs")
    return value


def _builtin_int(value: object, *, name: str, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a builtin integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} is below its minimum")
    return value


def _finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise TypeError(f"{name} must be a finite builtin float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _sha256(value: object, *, name: str) -> str:
    if type(value) is not str or len(value) != 64 or not set(value) <= HEX:
        raise ValueError(f"{name} must be a lowercase SHA-256")
    return value


def _float_list(value: object, expected: tuple[float, ...], *, name: str) -> None:
    if type(value) is not list or len(value) != len(expected):
        raise ValueError(f"{name} differs")
    for actual, registered in zip(value, expected, strict=True):
        if type(actual) is not float or actual != registered:
            raise ValueError(f"{name} differs")


def _validate_constants(value: object) -> dict:
    constants = _exact_dict(
        value,
        (
            "dimensions",
            "target_median_radii",
            "clip",
            "curvature",
            "l0_subsample_count",
            "l0_subsample_size",
            "l0_seed_formula",
            "l0_null_seed_formula",
            "l0_relative_convention",
            "l0_close_threshold",
            "mips_query_transform",
            "bootstrap_samples",
            "bootstrap_seed",
            "norm_correlation_threshold",
        ),
        name="constants",
    )
    if constants["dimensions"] != list(DIMENSIONS):
        raise ValueError("registered dimensions differ")
    _float_list(constants["target_median_radii"], TARGET_RADII, name="target radii")
    registered = {
        "clip": 2.5,
        "curvature": -1.0,
        "l0_subsample_count": 10,
        "l0_subsample_size": 2000,
        "l0_seed_formula": "7000 + 100 * dataset_index + replicate",
        "l0_null_seed_formula": "7500 + 100 * dataset_index + replicate",
        "l0_relative_convention": "2 * delta / diameter",
        "l0_close_threshold": 0.3,
        "mips_query_transform": "(-q0, qs)",
        "bootstrap_samples": 10000,
        "bootstrap_seed": 205,
        "norm_correlation_threshold": 0.05,
    }
    for key, expected in registered.items():
        if type(constants[key]) is not type(expected) or constants[key] != expected:
            raise ValueError(f"registered constant {key} differs")
    return constants


def _validate_input(value: object) -> dict:
    result = _exact_dict(
        value,
        (
            "bundle_path",
            "bundle_sha256",
            "metadata_sha256",
            "array_sha256",
            "embedding_dimension",
            "train_count",
            "query_count",
            "gallery_count",
            "identical_query_gallery",
            "dataset_index",
        ),
        name="input",
    )
    if type(result["bundle_path"]) is not str or not Path(result["bundle_path"]).is_absolute():
        raise ValueError("bundle path must be absolute")
    _sha256(result["bundle_sha256"], name="bundle SHA")
    _sha256(result["metadata_sha256"], name="metadata SHA")
    arrays = _exact_dict(result["array_sha256"], ARRAY_KEYS, name="array SHA mapping")
    for key in arrays:
        _sha256(arrays[key], name=f"{key} SHA")
    _builtin_int(result["embedding_dimension"], name="embedding dimension", minimum=1)
    _builtin_int(result["train_count"], name="train count", minimum=2000)
    _builtin_int(result["query_count"], name="query count", minimum=1)
    _builtin_int(result["gallery_count"], name="gallery count", minimum=1)
    expected_identical = (
        result["query_count"] == result["gallery_count"]
        and arrays["query_embeddings"] == arrays["gallery_embeddings"]
    )
    if (
        type(result["identical_query_gallery"]) is not bool
        or result["identical_query_gallery"] is not expected_identical
    ):
        raise TypeError("identical query/gallery flag must be boolean")
    if type(result["dataset_index"]) is not int or result["dataset_index"] not in {0, 1, 2}:
        raise ValueError("dataset index differs")
    return result


def _validate_l0(value: object, *, dataset_index: int) -> dict:
    result = _exact_dict(
        value,
        ("replicates", "observed_mean", "permutation_mean", "spectrum_mean"),
        name="l0",
    )
    if type(result["replicates"]) is not list or len(result["replicates"]) != 10:
        raise ValueError("L0 replicate count differs")
    observed: list[float] = []
    permutation: list[float] = []
    spectrum: list[float] = []
    for index, raw in enumerate(result["replicates"]):
        item = _exact_dict(
            raw,
            (
                "replicate",
                "seed",
                "null_seed",
                "indices_sha256",
                "base",
                "diameter",
                "delta",
                "delta_rel",
                "permutation_delta_rel",
                "spectrum_delta_rel",
            ),
            name=f"L0 replicate {index}",
        )
        if item["replicate"] != index or type(item["replicate"]) is not int:
            raise ValueError("L0 replicate index differs")
        if item["seed"] != 7000 + 100 * dataset_index + index:
            raise ValueError("L0 seed differs")
        if item["null_seed"] != 7500 + 100 * dataset_index + index:
            raise ValueError("L0 null seed differs")
        _sha256(item["indices_sha256"], name="L0 indices SHA")
        _builtin_int(item["base"], name="L0 base", minimum=0)
        diameter = _finite_float(item["diameter"], name="L0 diameter", nonnegative=True)
        delta = _finite_float(item["delta"], name="L0 delta", nonnegative=True)
        relative = _finite_float(item["delta_rel"], name="L0 relative", nonnegative=True)
        if diameter <= 0.0 or relative != 2.0 * delta / diameter:
            raise ValueError("L0 relative convention differs")
        observed.append(relative)
        permutation.append(
            _finite_float(
                item["permutation_delta_rel"], name="permutation relative", nonnegative=True
            )
        )
        spectrum.append(
            _finite_float(item["spectrum_delta_rel"], name="spectrum relative", nonnegative=True)
        )
    expected_means = (
        float(np.mean(observed)),
        float(np.mean(permutation)),
        float(np.mean(spectrum)),
    )
    for key, expected in zip(
        ("observed_mean", "permutation_mean", "spectrum_mean"), expected_means, strict=True
    ):
        if _finite_float(result[key], name=key, nonnegative=True) != expected:
            raise ValueError(f"{key} differs")
    return result


def _validate_control(value: object, *, query_count: int, name: str) -> dict:
    control = _exact_dict(value, ("recall_at_1", "correct", "correct_sha256"), name=name)
    correct = control["correct"]
    if (
        type(correct) is not list
        or len(correct) != query_count
        or any(type(item) is not bool for item in correct)
    ):
        raise ValueError(f"{name} correctness differs")
    expected_recall = float(sum(correct) / query_count)
    if (
        _finite_float(control["recall_at_1"], name=f"{name} R@1", nonnegative=True)
        != expected_recall
    ):
        raise ValueError(f"{name} R@1 differs")
    expected_sha = hashlib.sha256(np.asarray(correct, dtype=np.bool_).tobytes()).hexdigest()
    if _sha256(control["correct_sha256"], name=f"{name} correctness SHA") != expected_sha:
        raise ValueError(f"{name} correctness SHA differs")
    return control


def _validate_l1(value: object, *, query_count: int, embedding_dimension: int) -> dict:
    result = _exact_dict(value, ("probe", "dimensions"), name="l1")
    probe = _exact_dict(
        result["probe"],
        (
            "dimension",
            "norm_identity_frequency_spearman",
            "norm_cosine_margin_spearman",
            "status",
        ),
        name="norm probe",
    )
    if probe["dimension"] != 32 or type(probe["dimension"]) is not int:
        raise ValueError("probe dimension differs")
    first = _finite_float(probe["norm_identity_frequency_spearman"], name="frequency Spearman")
    second = _finite_float(probe["norm_cosine_margin_spearman"], name="margin Spearman")
    expected_status = (
        "NORM_CHANNEL_UNINFORMATIVE"
        if abs(first) < 0.05 and abs(second) < 0.05
        else "NORM_CHANNEL_INFORMATIVE"
    )
    if probe["status"] != expected_status:
        raise ValueError("norm probe status differs")
    if type(result["dimensions"]) is not list or not result["dimensions"]:
        raise ValueError("L1 dimensions differ")
    seen: list[int] = []
    for raw in result["dimensions"]:
        item = _exact_dict(
            raw,
            (
                "dimension",
                "pca_mean_sha256",
                "pca_components_sha256",
                "train_median_radius",
                "scales",
                "baseline_name",
                "bootstrap",
                "best_target_median_radius",
                "best_endpoint_gain",
                "best_spatial_gain",
                "best_power_gain",
                "decision",
            ),
            name="L1 dimension",
        )
        dimension = _builtin_int(item["dimension"], name="L1 dimension", minimum=1)
        if dimension not in DIMENSIONS or dimension in seen:
            raise ValueError("L1 dimension order differs")
        seen.append(dimension)
        _sha256(item["pca_mean_sha256"], name="PCA mean SHA")
        _sha256(item["pca_components_sha256"], name="PCA component SHA")
        train_median_radius = _finite_float(
            item["train_median_radius"], name="train median radius", nonnegative=True
        )
        if train_median_radius <= 0.0:
            raise ValueError("train median radius must be positive")
        if type(item["scales"]) is not list or len(item["scales"]) != len(TARGET_RADII):
            raise ValueError("L1 scales differ")
        scale_rows: list[tuple[float, dict[str, dict]]] = []
        for expected_target, raw_scale in zip(TARGET_RADII, item["scales"], strict=True):
            scale = _exact_dict(
                raw_scale,
                ("target_median_radius", "scale", "controls"),
                name="L1 scale",
            )
            target = _finite_float(
                scale["target_median_radius"], name="target radius", nonnegative=True
            )
            if target != expected_target:
                raise ValueError("target radius differs")
            actual_scale = _finite_float(scale["scale"], name="scale", nonnegative=True)
            if actual_scale != target / train_median_radius:
                raise ValueError("registered scale differs")
            controls = _exact_dict(scale["controls"], CONTROL_ORDER, name="controls")
            scale_rows.append(
                (
                    target,
                    {
                        key: _validate_control(controls[key], query_count=query_count, name=key)
                        for key in controls
                    },
                )
            )
        if item["baseline_name"] not in {"pca_euclidean", "pca_cosine"}:
            raise ValueError("baseline name differs")
        baseline_name = item["baseline_name"]
        first_controls = scale_rows[0][1]
        expected_baseline = (
            "pca_euclidean"
            if first_controls["pca_euclidean"]["recall_at_1"]
            >= first_controls["pca_cosine"]["recall_at_1"]
            else "pca_cosine"
        )
        if baseline_name != expected_baseline:
            raise ValueError("baseline endpoint selection differs")
        best = max(
            scale_rows,
            key=lambda row: row[1]["lorentz"]["recall_at_1"] - row[1][baseline_name]["recall_at_1"],
        )
        controls = best[1]
        endpoint_gain = controls["lorentz"]["recall_at_1"] - controls[baseline_name]["recall_at_1"]
        spatial_gain = controls["lorentz"]["recall_at_1"] - controls["spatial_only"]["recall_at_1"]
        power_gain = controls["lorentz"]["recall_at_1"] - max(
            controls["power_1"]["recall_at_1"], controls["power_3"]["recall_at_1"]
        )
        relations = {
            "best_target_median_radius": best[0],
            "best_endpoint_gain": endpoint_gain,
            "best_spatial_gain": spatial_gain,
            "best_power_gain": power_gain,
        }
        for key, expected in relations.items():
            if _finite_float(item[key], name=key) != expected:
                raise ValueError(f"{key} differs")
        bootstrap = _exact_dict(
            item["bootstrap"],
            (
                "point",
                "lower",
                "upper",
                "standard_errors",
                "samples",
                "seed",
                "replicates",
                "replicates_sha256",
            ),
            name="bootstrap",
        )
        if _finite_float(bootstrap["point"], name="bootstrap point") != endpoint_gain:
            raise ValueError("bootstrap point differs")
        if (
            bootstrap["samples"] != 10000
            or type(bootstrap["samples"]) is not int
            or bootstrap["seed"] != 205
            or type(bootstrap["seed"]) is not int
        ):
            raise ValueError("bootstrap constants differ")
        raw_replicates = bootstrap["replicates"]
        if (
            type(raw_replicates) is not list
            or len(raw_replicates) != bootstrap["samples"]
            or any(type(value) is not float or not np.isfinite(value) for value in raw_replicates)
        ):
            raise ValueError("bootstrap replicates differ")
        replicates = np.asarray(raw_replicates, dtype=np.float64)
        expected_lower, expected_upper = np.percentile(replicates, [2.5, 97.5])
        lower = _finite_float(bootstrap["lower"], name="bootstrap lower")
        upper = _finite_float(bootstrap["upper"], name="bootstrap upper")
        if lower != float(expected_lower) or upper != float(expected_upper):
            raise ValueError("bootstrap interval differs")
        deviation = float(np.std(replicates, ddof=1))
        expected_se = endpoint_gain / deviation if deviation > 0.0 else 0.0
        standard_errors = _finite_float(bootstrap["standard_errors"], name="bootstrap SE")
        if standard_errors != expected_se:
            raise ValueError("bootstrap standard errors differ")
        expected_sha = hashlib.sha256(replicates.tobytes(order="C")).hexdigest()
        if _sha256(bootstrap["replicates_sha256"], name="bootstrap replicate SHA") != expected_sha:
            raise ValueError("bootstrap replicate SHA differs")
        expected_decision = l1_decision(
            endpoint_gain=endpoint_gain,
            endpoint_lower=lower,
            standard_errors=standard_errors,
            spatial_gain=spatial_gain,
            power_gain=power_gain,
        )
        decision = _exact_dict(
            item["decision"],
            ("status", "endpoint_passed", "spatial_passed", "power_passed"),
            name="L1 decision",
        )
        if decision != {
            "status": expected_decision.status,
            "endpoint_passed": expected_decision.endpoint_passed,
            "spatial_passed": expected_decision.spatial_passed,
            "power_passed": expected_decision.power_passed,
        }:
            raise ValueError("L1 decision differs")
    if seen[0] != 32:
        raise ValueError("dimension 32 must be evaluated first")
    expected_expanded = [32] + [value for value in (8, 16, 64) if value <= embedding_dimension]
    first_passed = result["dimensions"][0]["decision"]["status"] == "GEOMETRY_SURVIVES"
    if seen != (expected_expanded if first_passed else [32]):
        raise ValueError("conditional dimension expansion differs")
    return result


def validate_lorentz_report(value: object) -> None:
    """Validate one exact Lorentz L0/L1 persisted report."""

    report = _exact_dict(value, REPORT_KEYS, name="report")
    if report["schema_version"] != 1 or type(report["schema_version"]) is not int:
        raise ValueError("schema version differs")
    input_value = _validate_input(report["input"])
    _validate_constants(report["constants"])
    l0 = _validate_l0(report["l0"], dataset_index=input_value["dataset_index"])
    l1 = _validate_l1(
        report["l1"],
        query_count=input_value["query_count"],
        embedding_dimension=input_value["embedding_dimension"],
    )
    runtime = _exact_dict(report["runtime"], ("python", "numpy"), name="runtime")
    if any(type(runtime[key]) is not str or not runtime[key] for key in runtime):
        raise ValueError("runtime differs")
    decision = _exact_dict(
        report["decision"],
        (
            "status",
            "l0_mean_delta_rel",
            "l0_above_close_threshold",
            "evaluated_dimensions",
        ),
        name="global decision",
    )
    if decision["l0_mean_delta_rel"] != l0["observed_mean"]:
        raise ValueError("global L0 value differs")
    expected_l0_close = l0["observed_mean"] > 0.3
    if (
        type(decision["l0_above_close_threshold"]) is not bool
        or decision["l0_above_close_threshold"] is not expected_l0_close
    ):
        raise ValueError("global L0 threshold predicate differs")
    evaluated = [item["dimension"] for item in l1["dimensions"]]
    if decision["evaluated_dimensions"] != evaluated:
        raise ValueError("evaluated dimensions differ")
    any_pass = any(item["decision"]["status"] == "GEOMETRY_SURVIVES" for item in l1["dimensions"])
    expected_status = "L1_DATASET_PASS" if any_pass else "CLOSE_GEOMETRY_DATASET"
    if decision["status"] != expected_status:
        raise ValueError("global decision differs")
    timing = _exact_dict(
        report["timing"], ("l0_seconds", "l1_seconds", "total_seconds"), name="timing"
    )
    l0_seconds = _finite_float(timing["l0_seconds"], name="L0 seconds", nonnegative=True)
    l1_seconds = _finite_float(timing["l1_seconds"], name="L1 seconds", nonnegative=True)
    if (
        _finite_float(timing["total_seconds"], name="total seconds", nonnegative=True)
        != l0_seconds + l1_seconds
    ):
        raise ValueError("total timing differs")


def _strict_json_object(data: bytes) -> dict[str, object]:
    def pairs(items):
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise ValueError("report contains a duplicate key")
            result[key] = value
        return result

    def reject(token: str):
        raise ValueError(f"report contains nonfinite JSON number {token}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    if type(value) is not dict:
        raise TypeError("report must decode to an object")
    return value


def publish_lorentz_report(path: Path, payload: Mapping[str, object]) -> None:
    """Publish one strict report with exclusive same-directory hard-linking."""

    validate_lorentz_report(payload)
    path = Path(path)
    if not path.parent.is_dir() or path.exists() or path.is_symlink():
        raise FileExistsError(path)
    data = (json.dumps(payload, allow_nan=False, indent=2) + "\n").encode()
    temporary = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    published = False
    try:
        with temporary.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        published = True
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        persisted = _strict_json_object(path.read_bytes())
        validate_lorentz_report(persisted)
        if persisted != payload:
            raise ValueError("persisted report differs from input payload")
    except Exception:
        if published:
            try:
                if path.stat().st_ino == temporary.stat().st_ino:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        temporary.unlink(missing_ok=True)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)


def runtime_payload() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": str(np.__version__)}


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _metadata_sha256(value: Mapping[str, object]) -> str:
    data = json.dumps(value, allow_nan=False, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(data).hexdigest()


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and values[order[stop]] == values[order[start]]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("Spearman vectors differ")
    first = _average_ranks(left.astype(np.float64))
    second = _average_ranks(right.astype(np.float64))
    first -= np.mean(first)
    second -= np.mean(second)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    return 0.0 if denominator == 0.0 else float(np.dot(first, second) / denominator)


def _score_view(
    chunks,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    exclude_same_row: bool,
):
    def prepared_chunks():
        row_start = 0
        for raw in chunks:
            chunk = np.ascontiguousarray(raw.astype(np.float64))
            if exclude_same_row:
                rows = np.arange(chunk.shape[0], dtype=np.int64)
                chunk[rows, row_start + rows] = np.finfo(np.float64).min
            row_start += chunk.shape[0]
            yield chunk

    return retrieval_metrics_from_score_chunks(prepared_chunks(), query_labels, gallery_labels)


def _control_payload(view) -> dict[str, object]:
    correct = view.top1_correct.astype(np.bool_, copy=False)
    return {
        "recall_at_1": view.recall[1],
        "correct": correct.tolist(),
        "correct_sha256": _array_sha256(correct),
    }


def _evaluate_l0(train_embeddings: np.ndarray, *, dataset_index: int) -> dict[str, object]:
    train = l2_normalize(train_embeddings)
    rows: list[dict[str, object]] = []
    for replicate in range(10):
        indices = l0_subsample_indices(train.shape[0], dataset_index, replicate)
        sample = np.ascontiguousarray(train[indices])
        distances = pairwise_chord_distances(sample)
        estimate = gromov_delta_rel(distances, medoid_index(distances))
        null_seed = 7500 + 100 * dataset_index + replicate
        permuted_distances = pairwise_chord_distances(column_permutation_null(sample, null_seed))
        permuted = gromov_delta_rel(permuted_distances, medoid_index(permuted_distances))
        spectrum_distances = pairwise_chord_distances(spectrum_gaussian_null(sample, null_seed))
        spectrum = gromov_delta_rel(spectrum_distances, medoid_index(spectrum_distances))
        rows.append(
            {
                "replicate": replicate,
                "seed": 7000 + 100 * dataset_index + replicate,
                "null_seed": null_seed,
                "indices_sha256": _array_sha256(indices),
                "base": estimate.base,
                "diameter": estimate.diameter,
                "delta": estimate.delta,
                "delta_rel": estimate.relative,
                "permutation_delta_rel": permuted.relative,
                "spectrum_delta_rel": spectrum.relative,
            }
        )
    return {
        "replicates": rows,
        "observed_mean": float(np.mean([row["delta_rel"] for row in rows])),
        "permutation_mean": float(np.mean([row["permutation_delta_rel"] for row in rows])),
        "spectrum_mean": float(np.mean([row["spectrum_delta_rel"] for row in rows])),
    }


def _norm_probe(
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    query_projected: np.ndarray,
    gallery_projected: np.ndarray,
) -> dict[str, object]:
    counts: dict[object, int] = {}
    for label in gallery_labels.tolist():
        counts[label] = counts.get(label, 0) + 1
    frequencies = np.asarray(
        [counts.get(label, 0) for label in query_labels.tolist()], dtype=np.float64
    )
    norms = np.linalg.norm(query_projected.astype(np.float64), axis=1)
    query_unit = l2_normalize(query_projected)
    gallery_unit = l2_normalize(gallery_projected)
    exclude_same_row = query_projected.shape == gallery_projected.shape and np.array_equal(
        query_projected, gallery_projected
    )
    gallery64 = gallery_unit.astype(np.float64)
    margins: list[np.ndarray] = []
    for start in range(0, query_unit.shape[0], 256):
        cosine = query_unit[start : start + 256].astype(np.float64) @ gallery64.T
        if exclude_same_row:
            rows = np.arange(cosine.shape[0], dtype=np.int64)
            cosine[rows, start + rows] = np.finfo(np.float64).min
        if cosine.shape[1] == 1:
            margins.append(cosine[:, 0])
        else:
            top_two = np.partition(cosine, -2, axis=1)[:, -2:]
            margins.append(np.max(top_two, axis=1) - np.min(top_two, axis=1))
    margin = np.concatenate(margins)
    first = _spearman(norms, frequencies)
    second = _spearman(norms, margin)
    return {
        "dimension": 32,
        "norm_identity_frequency_spearman": first,
        "norm_cosine_margin_spearman": second,
        "status": (
            "NORM_CHANNEL_UNINFORMATIVE"
            if abs(first) < 0.05 and abs(second) < 0.05
            else "NORM_CHANNEL_INFORMATIVE"
        ),
    }


def _evaluate_dimension(bundle: EmbeddingBundle, dimension: int) -> dict[str, object]:
    fit = fit_frozen_pca(bundle.train_embeddings, dimension)
    train = apply_frozen_pca(bundle.train_embeddings, fit)
    query = apply_frozen_pca(bundle.query_embeddings, fit)
    gallery = apply_frozen_pca(bundle.gallery_embeddings, fit)
    exclude_same_row = (
        bundle.query_embeddings.shape == bundle.gallery_embeddings.shape
        and np.array_equal(bundle.query_embeddings, bundle.gallery_embeddings)
    )
    train_median_radius = float(
        np.median(np.sqrt(np.sum(train.astype(np.float64) ** 2, axis=1, dtype=np.float64)))
    )
    scale_rows: list[dict[str, object]] = []
    correctness: list[dict[str, np.ndarray]] = []
    for target in TARGET_RADII:
        scale = scale_for_target_median(train, target)
        chunks = score_control_blocks(query, gallery, scale, 2.5)
        views = {
            name: _score_view(
                value,
                bundle.query_labels,
                bundle.gallery_labels,
                exclude_same_row=exclude_same_row,
            )
            for name, value in chunks.items()
        }
        correctness.append(
            {name: view.top1_correct.astype(np.bool_, copy=False) for name, view in views.items()}
        )
        scale_rows.append(
            {
                "target_median_radius": target,
                "scale": scale,
                "controls": {name: _control_payload(views[name]) for name in CONTROL_ORDER},
            }
        )
    euclidean = correctness[0]["pca_euclidean"]
    cosine = correctness[0]["pca_cosine"]
    baseline_name = (
        "pca_euclidean" if float(np.mean(euclidean)) >= float(np.mean(cosine)) else "pca_cosine"
    )
    baseline = correctness[0][baseline_name]
    interiors = np.stack([row["lorentz"] for row in correctness])
    interval = paired_identity_max_interval(
        baseline, interiors, bundle.query_labels, samples=10_000, seed=205
    )
    endpoint_gains = np.mean(interiors, axis=1) - float(np.mean(baseline))
    best_index = int(np.argmax(endpoint_gains))
    best = correctness[best_index]
    lorentz_r1 = float(np.mean(best["lorentz"]))
    spatial_gain = lorentz_r1 - float(np.mean(best["spatial_only"]))
    power_gain = lorentz_r1 - max(float(np.mean(best["power_1"])), float(np.mean(best["power_3"])))
    decision = l1_decision(
        endpoint_gain=interval.point,
        endpoint_lower=interval.lower,
        standard_errors=interval.standard_errors,
        spatial_gain=spatial_gain,
        power_gain=power_gain,
    )
    return {
        "dimension": dimension,
        "pca_mean_sha256": _array_sha256(fit.mean),
        "pca_components_sha256": _array_sha256(fit.components),
        "train_median_radius": train_median_radius,
        "scales": scale_rows,
        "baseline_name": baseline_name,
        "bootstrap": {
            "point": interval.point,
            "lower": interval.lower,
            "upper": interval.upper,
            "standard_errors": interval.standard_errors,
            "samples": interval.samples,
            "seed": interval.seed,
            "replicates": list(interval.replicates),
            "replicates_sha256": interval.replicates_sha256,
        },
        "best_target_median_radius": TARGET_RADII[best_index],
        "best_endpoint_gain": interval.point,
        "best_spatial_gain": spatial_gain,
        "best_power_gain": power_gain,
        "decision": {
            "status": decision.status,
            "endpoint_passed": decision.endpoint_passed,
            "spatial_passed": decision.spatial_passed,
            "power_passed": decision.power_passed,
        },
    }


def evaluate_l0_l1(bundle: EmbeddingBundle, *, dataset_index: int) -> dict[str, object]:
    """Evaluate the registered train-only L0 then no-training L1 ladder."""

    if type(bundle) is not EmbeddingBundle:
        raise TypeError("bundle must be an EmbeddingBundle")
    if type(dataset_index) is not int or dataset_index not in {0, 1, 2}:
        raise ValueError("dataset_index must be 0, 1, or 2")
    if bundle.train_embeddings.shape[1] < 32:
        raise ValueError("Lorentz rider requires at least 32 embedding dimensions")
    identical_query_gallery = (
        bundle.query_embeddings.shape == bundle.gallery_embeddings.shape
        and np.array_equal(bundle.query_embeddings, bundle.gallery_embeddings)
    )
    started = time.perf_counter()
    l0 = _evaluate_l0(bundle.train_embeddings, dataset_index=dataset_index)
    after_l0 = time.perf_counter()
    fit32 = fit_frozen_pca(bundle.train_embeddings, 32)
    probe = _norm_probe(
        bundle.query_labels,
        bundle.gallery_labels,
        apply_frozen_pca(bundle.query_embeddings, fit32),
        apply_frozen_pca(bundle.gallery_embeddings, fit32),
    )
    dimensions = [_evaluate_dimension(bundle, 32)]
    if dimensions[0]["decision"]["status"] == "GEOMETRY_SURVIVES":
        for dimension in (8, 16, 64):
            if dimension <= bundle.train_embeddings.shape[1]:
                dimensions.append(_evaluate_dimension(bundle, dimension))
    finished = time.perf_counter()
    l0_seconds = after_l0 - started
    l1_seconds = finished - after_l0
    report = {
        "schema_version": 1,
        "input": {
            "bundle_path": str(bundle.path),
            "bundle_sha256": bundle.sha256,
            "metadata_sha256": _metadata_sha256(bundle.metadata),
            "array_sha256": {key: _array_sha256(getattr(bundle, key)) for key in ARRAY_KEYS},
            "embedding_dimension": bundle.train_embeddings.shape[1],
            "train_count": bundle.train_embeddings.shape[0],
            "query_count": bundle.query_embeddings.shape[0],
            "gallery_count": bundle.gallery_embeddings.shape[0],
            "identical_query_gallery": identical_query_gallery,
            "dataset_index": dataset_index,
        },
        "constants": {
            "dimensions": list(DIMENSIONS),
            "target_median_radii": list(TARGET_RADII),
            "clip": 2.5,
            "curvature": -1.0,
            "l0_subsample_count": 10,
            "l0_subsample_size": 2000,
            "l0_seed_formula": "7000 + 100 * dataset_index + replicate",
            "l0_null_seed_formula": "7500 + 100 * dataset_index + replicate",
            "l0_relative_convention": "2 * delta / diameter",
            "l0_close_threshold": 0.3,
            "mips_query_transform": "(-q0, qs)",
            "bootstrap_samples": 10000,
            "bootstrap_seed": 205,
            "norm_correlation_threshold": 0.05,
        },
        "l0": l0,
        "l1": {"probe": probe, "dimensions": dimensions},
        "runtime": runtime_payload(),
        "decision": {
            "status": (
                "L1_DATASET_PASS"
                if any(row["decision"]["status"] == "GEOMETRY_SURVIVES" for row in dimensions)
                else "CLOSE_GEOMETRY_DATASET"
            ),
            "l0_mean_delta_rel": l0["observed_mean"],
            "l0_above_close_threshold": l0["observed_mean"] > 0.3,
            "evaluated_dimensions": [row["dimension"] for row in dimensions],
        },
        "timing": {
            "l0_seconds": l0_seconds,
            "l1_seconds": l1_seconds,
            "total_seconds": l0_seconds + l1_seconds,
        },
    }
    validate_lorentz_report(report)
    return report
