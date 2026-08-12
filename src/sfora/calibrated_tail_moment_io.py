"""Strict result validation and atomic publication for CTM evaluation."""

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

from sfora.calibrated_tail_moment import (
    CONTROL_ORDER,
    EQUAL_TOTAL_STORAGE_CONTROLS,
    MATCHED_ROW_WIDTH_CONTROLS,
    cluster_lambda_interval,
    ctm_decision,
    evaluate_width,
    fit_projection_basis,
    fit_tail_moment,
    permuted_tail_null,
    project_unit,
    query_identity_interval,
)
from sfora.unicom_audit_io import EmbeddingBundle
from sfora.unicom_retrieval_audit import l2_normalize

_REPORT_KEYS = (
    "schema_version",
    "phase",
    "input",
    "constants",
    "fits",
    "views",
    "primary",
    "timing",
    "storage",
    "runtime",
    "decision",
)
_HEX = frozenset("0123456789abcdef")
_ARRAY_KEYS = (
    "train_embeddings",
    "train_labels",
    "query_embeddings",
    "query_labels",
    "gallery_embeddings",
    "gallery_labels",
)


def _exact_dict(value: object, keys: tuple[str, ...], *, name: str) -> dict:
    if type(value) is not dict or tuple(value) != keys:
        raise ValueError(f"{name} key order differs")
    return value


def _finite_float(value: object, *, name: str, nonnegative: bool = False) -> float:
    if type(value) is not float or not np.isfinite(value):
        raise TypeError(f"{name} must be a finite builtin float")
    if nonnegative and value < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _probability(value: object, *, name: str) -> float:
    result = _finite_float(value, name=name, nonnegative=True)
    if result > 1.0:
        raise ValueError(f"{name} must not exceed one")
    return result


def _builtin_int(value: object, *, name: str, nonnegative: bool = False) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be a builtin integer")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")
    return value


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def _validate_fit_grid(
    value: object,
    *,
    name: str,
    widths: tuple[int, ...],
    dimension: int,
    neighbors: int,
    basis_kind: str,
) -> dict:
    grid = _exact_dict(value, tuple(str(width) for width in widths), name=name)
    for width in widths:
        item = _exact_dict(
            grid[str(width)],
            (
                "dimension",
                "basis_kind",
                "width",
                "neighbors",
                "lambda_raw",
                "lambda_value",
            ),
            name=f"{name} {width}",
        )
        if (
            item["dimension"] != dimension
            or item["basis_kind"] != basis_kind
            or item["width"] != width
            or item["neighbors"] != neighbors
        ):
            raise ValueError(f"{name} {width} constants differ")
        raw = _finite_float(item["lambda_raw"], name=f"{name} {width} raw")
        clipped = _finite_float(
            item["lambda_value"], name=f"{name} {width} value", nonnegative=True
        )
        if clipped != min(1.0, max(0.0, raw)):
            raise ValueError(f"{name} {width} clipping differs")
    return grid


def _validate_view(
    value: object,
    *,
    name: str,
    query_count: int,
    gallery_count: int,
    expected_values: int,
    expected_fixed: int,
) -> dict:
    view = _exact_dict(
        value,
        (
            "recall",
            "map_at_r",
            "top1_indices",
            "top1_correct",
            "values_per_row",
            "fixed_bytes",
            "total_bytes",
            "query_projection_seconds",
            "descriptor_build_seconds",
            "search_seconds",
        ),
        name=name,
    )
    recall = _exact_dict(view["recall"], ("1", "10", "20", "30"), name=f"{name} recall")
    for key in recall:
        _probability(recall[key], name=f"{name} recall {key}")
    _probability(view["map_at_r"], name=f"{name} mAP@R")
    indices = view["top1_indices"]
    correct = view["top1_correct"]
    if (
        type(indices) is not list
        or len(indices) != query_count
        or any(type(index) is not int or not 0 <= index < gallery_count for index in indices)
    ):
        raise ValueError(f"{name} top1 indices differ")
    if (
        type(correct) is not list
        or len(correct) != query_count
        or any(type(item) is not bool for item in correct)
    ):
        raise ValueError(f"{name} top1 correctness differs")
    expected_r1 = float(sum(correct) / query_count)
    if recall["1"] != expected_r1:
        raise ValueError(f"{name} recall@1 differs from correctness")
    if view["values_per_row"] != expected_values:
        raise ValueError(f"{name} row width differs")
    if view["fixed_bytes"] != expected_fixed:
        raise ValueError(f"{name} fixed storage differs")
    if view["total_bytes"] != gallery_count * expected_values * 4 + expected_fixed:
        raise ValueError(f"{name} total storage differs")
    _finite_float(
        view["query_projection_seconds"],
        name=f"{name} projection seconds",
        nonnegative=True,
    )
    _finite_float(
        view["descriptor_build_seconds"],
        name=f"{name} descriptor build seconds",
        nonnegative=True,
    )
    _finite_float(view["search_seconds"], name=f"{name} search seconds", nonnegative=True)
    return view


def validate_ctm_report(
    value: object,
    *,
    expected_widths: tuple[int, ...] = (64, 128, 256, 512),
    expected_neighbors: int = 50,
    expected_null_count: int = 32,
) -> None:
    """Validate the exact persisted CTM report union."""

    report = _exact_dict(value, _REPORT_KEYS, name="report")
    if report["schema_version"] != 1:
        raise ValueError("report schema version differs")
    if report["phase"] not in {"FIT_CLOSED", "EVALUATED"}:
        raise ValueError("report phase differs")
    input_value = _exact_dict(
        report["input"],
        (
            "bundle_path",
            "bundle_sha256",
            "metadata",
            "array_sha256",
            "embedding_dimension",
            "query_count",
            "gallery_count",
        ),
        name="input",
    )
    if type(input_value["bundle_path"]) is not str or not input_value["bundle_path"]:
        raise TypeError("bundle path must be a nonempty string")
    sha = input_value["bundle_sha256"]
    if type(sha) is not str or len(sha) != 64 or not set(sha) <= _HEX:
        raise ValueError("bundle SHA-256 differs")
    if type(input_value["metadata"]) is not dict:
        raise TypeError("bundle metadata must be an object")
    array_sha = _exact_dict(input_value["array_sha256"], _ARRAY_KEYS, name="array SHA-256")
    if any(
        type(array_sha[key]) is not str
        or len(array_sha[key]) != 64
        or not set(array_sha[key]) <= _HEX
        for key in array_sha
    ):
        raise ValueError("array SHA-256 differs")
    dimension = input_value["embedding_dimension"]
    if type(dimension) is not int or dimension <= 0:
        raise TypeError("embedding dimension must be a positive builtin integer")
    if input_value["metadata"].get("embedding_dimension") != dimension:
        raise ValueError("metadata embedding dimension differs")
    query_count = _builtin_int(input_value["query_count"], name="query count")
    gallery_count = _builtin_int(input_value["gallery_count"], name="gallery count")
    if query_count <= 0 or gallery_count <= 0:
        raise ValueError("query and gallery counts must be positive")
    constants = _exact_dict(
        report["constants"],
        ("widths", "neighbors", "bootstrap_samples", "bootstrap_seed", "null_seeds"),
        name="constants",
    )
    active_widths = tuple(width for width in expected_widths if width < dimension - 1)
    if constants["widths"] != list(active_widths):
        raise ValueError("registered widths differ")
    if constants["neighbors"] != expected_neighbors:
        raise ValueError("registered neighbor count differs")
    if constants["bootstrap_samples"] != 10_000 or constants["bootstrap_seed"] != 205:
        raise ValueError("bootstrap constants differ")
    null_seeds = constants["null_seeds"]
    if (
        type(null_seeds) is not list
        or len(null_seeds) != expected_null_count
        or any(type(seed) is not int for seed in null_seeds)
    ):
        raise ValueError("null seed constants differ")
    fits = _exact_dict(
        report["fits"],
        ("primary_width", "native", "pca", "primary_scalar_nonzero", "null"),
        name="fits",
    )
    expected_primary = 128 if 128 in active_widths else active_widths[0]
    if fits["primary_width"] != expected_primary:
        raise ValueError("primary width differs")
    native_fits = _validate_fit_grid(
        fits["native"],
        name="native fits",
        widths=active_widths,
        dimension=dimension,
        neighbors=expected_neighbors,
        basis_kind="native",
    )
    _validate_fit_grid(
        fits["pca"],
        name="PCA fits",
        widths=active_widths,
        dimension=dimension,
        neighbors=expected_neighbors,
        basis_kind="pca",
    )
    raw = native_fits[str(expected_primary)]["lambda_raw"]
    clipped = native_fits[str(expected_primary)]["lambda_value"]
    if type(fits["primary_scalar_nonzero"]) is not bool:
        raise TypeError("primary scalar flag must be a builtin bool")
    null = _exact_dict(fits["null"], ("seeds", "lambda_raw_values", "p_value"), name="tail null")
    if null["seeds"] != null_seeds:
        raise ValueError("tail null seeds differ")
    null_values = null["lambda_raw_values"]
    if type(null_values) is not list or len(null_values) != expected_null_count:
        raise ValueError("tail null values differ")
    for item in null_values:
        _finite_float(item, name="tail null value")
    _probability(null["p_value"], name="tail null p-value")
    timing = _exact_dict(
        report["timing"],
        (
            "normalization_seconds",
            "basis_fit_seconds",
            "coefficient_fit_seconds",
            "falsifier_seconds",
            "evaluation_seconds",
        ),
        name="timing",
    )
    for key in timing:
        _finite_float(timing[key], name=key, nonnegative=True)
    runtime = _exact_dict(report["runtime"], ("python", "numpy"), name="runtime")
    if any(type(runtime[key]) is not str or not runtime[key] for key in runtime):
        raise TypeError("runtime values must be nonempty builtin strings")
    decision = _exact_dict(report["decision"], ("status", "reason"), name="decision")
    if any(type(decision[key]) is not str or not decision[key] for key in decision):
        raise TypeError("decision fields must be nonempty builtin strings")
    if report["phase"] == "FIT_CLOSED":
        if any(report[key] is not None for key in ("views", "primary", "storage")):
            raise ValueError("FIT_CLOSED cannot contain evaluation payloads")
        if timing["evaluation_seconds"] != 0.0:
            raise ValueError("FIT_CLOSED evaluation timing differs")
        if clipped != 0.0 or fits["primary_scalar_nonzero"] is not False:
            raise ValueError("FIT_CLOSED coefficient differs")
        if decision["status"] != "CLOSE":
            raise ValueError("FIT_CLOSED decision differs")
    else:
        if any(report[key] is None for key in ("views", "primary", "storage")):
            raise ValueError("EVALUATED payload is incomplete")
        views = _exact_dict(
            report["views"], tuple(str(width) for width in active_widths), name="views"
        )
        pca_fixed = dimension * 4 + dimension * dimension * 4
        official_width = min(512, dimension)
        for width in active_widths:
            width_views = _exact_dict(views[str(width)], CONTROL_ORDER, name=f"width {width} views")
            for control in CONTROL_ORDER:
                expected_values = (
                    official_width
                    if control == "official_512"
                    else dimension
                    if control == "full_width_768"
                    else width + 1
                )
                expected_fixed = (
                    pca_fixed if control in {"pca_ctm", "pca_renormalized_prefix_plus_zero"} else 0
                )
                _validate_view(
                    width_views[control],
                    name=f"width {width} {control}",
                    query_count=query_count,
                    gallery_count=gallery_count,
                    expected_values=expected_values,
                    expected_fixed=expected_fixed,
                )
        primary = _exact_dict(
            report["primary"],
            ("paired", "lambda_interval", "null_p_value", "width_gains", "replication_status"),
            name="primary",
        )
        paired = _exact_dict(
            primary["paired"], ("point", "lower", "upper", "samples", "seed"), name="paired"
        )
        for key in ("point", "lower", "upper"):
            _finite_float(paired[key], name=f"paired {key}")
        if paired["samples"] != 10_000 or paired["seed"] != 205:
            raise ValueError("paired bootstrap constants differ")
        if not paired["lower"] <= paired["point"] <= paired["upper"]:
            raise ValueError("paired interval order differs")
        primary_views = views[str(expected_primary)]
        candidate_correct = np.asarray(
            primary_views["native_ctm"]["top1_correct"], dtype=np.float64
        )
        baseline_correct = np.asarray(
            primary_views["renormalized_prefix_plus_zero"]["top1_correct"],
            dtype=np.float64,
        )
        expected_point = float(np.mean(candidate_correct - baseline_correct))
        if paired["point"] != expected_point:
            raise ValueError("paired point differs")
        interval = _exact_dict(
            primary["lambda_interval"], ("point", "lower", "upper"), name="lambda interval"
        )
        for key in interval:
            _finite_float(interval[key], name=f"lambda interval {key}")
        if interval["point"] != raw or interval["lower"] > interval["upper"]:
            raise ValueError("lambda interval differs")
        _probability(primary["null_p_value"], name="null p-value")
        if primary["null_p_value"] != null["p_value"]:
            raise ValueError("primary null p-value differs")
        gains = primary["width_gains"]
        if type(gains) is not list or len(gains) != len(active_widths):
            raise ValueError("width gains differ")
        for item, width in zip(gains, active_widths, strict=True):
            if type(item) is not list or len(item) != 2 or item[0] != width:
                raise ValueError("width gain key differs")
            gain = _finite_float(item[1], name="width gain")
            expected_gain = (
                views[str(width)]["native_ctm"]["recall"]["1"]
                - views[str(width)]["renormalized_prefix_plus_zero"]["recall"]["1"]
            )
            if gain != expected_gain:
                raise ValueError("width gain differs")
        if primary["replication_status"] not in {"PENDING", "PASSED", "FAILED"}:
            raise ValueError("replication status differs")
        storage = _exact_dict(
            report["storage"],
            ("native_ctm_bytes", "official_512_bytes", "pca_fixed_bytes"),
            name="storage",
        )
        if storage != {
            "native_ctm_bytes": primary_views["native_ctm"]["total_bytes"],
            "official_512_bytes": primary_views["official_512"]["total_bytes"],
            "pca_fixed_bytes": pca_fixed,
        }:
            raise ValueError("storage summary differs")
        recomputed = ctm_decision(
            ctm_r1=primary_views["native_ctm"]["recall"]["1"],
            ctm_map_at_r=primary_views["native_ctm"]["map_at_r"],
            renormalized_r1=primary_views["renormalized_prefix_plus_zero"]["recall"]["1"],
            renormalized_map_at_r=primary_views["renormalized_prefix_plus_zero"]["map_at_r"],
            official_512_r1=primary_views["official_512"]["recall"]["1"],
            paired_lower=paired["lower"],
            control_r1={
                key: primary_views[key]["recall"]["1"] for key in MATCHED_ROW_WIDTH_CONTROLS
            },
            ctm_total_bytes=primary_views["native_ctm"]["total_bytes"],
            control_total_bytes={
                key: primary_views[key]["total_bytes"] for key in EQUAL_TOTAL_STORAGE_CONTROLS
            },
            lambda_raw=raw,
            lambda_lower=interval["lower"],
            null_p_value=primary["null_p_value"],
            width_gains=tuple((item[0], item[1]) for item in gains),
            replication_status=primary["replication_status"],
        )
        if decision != {"status": recomputed.status, "reason": recomputed.reason}:
            raise ValueError("EVALUATED decision differs")


def _strict_json_object(data: bytes) -> dict[str, object]:
    def pairs(items):
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise ValueError("report contains a duplicate key")
            result[key] = item
        return result

    def reject(token: str):
        raise ValueError(f"report contains nonfinite JSON number {token}")

    value = json.loads(data, object_pairs_hook=pairs, parse_constant=reject)
    if type(value) is not dict:
        raise TypeError("report must decode to an object")
    return value


def publish_ctm_report(
    path: Path,
    payload: Mapping[str, object],
    *,
    expected_widths: tuple[int, ...] = (64, 128, 256, 512),
    expected_neighbors: int = 50,
    expected_null_count: int = 32,
) -> None:
    """Atomically publish one strict report without replacing any path."""

    validate_ctm_report(
        payload,
        expected_widths=expected_widths,
        expected_neighbors=expected_neighbors,
        expected_null_count=expected_null_count,
    )
    path = Path(path)
    if not path.parent.is_dir() or path.exists() or path.is_symlink():
        raise FileExistsError(path)
    data = (json.dumps(payload, allow_nan=False, indent=2) + "\n").encode()
    temporary = path.parent / f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    try:
        with temporary.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    persisted = _strict_json_object(path.read_bytes())
    validate_ctm_report(
        persisted,
        expected_widths=expected_widths,
        expected_neighbors=expected_neighbors,
        expected_null_count=expected_null_count,
    )


def runtime_payload() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": str(np.__version__)}


def _view_payload(view) -> dict[str, object]:
    return {
        "recall": {str(key): value for key, value in view.recall.items()},
        "map_at_r": view.map_at_r,
        "top1_indices": view.top1_indices.tolist(),
        "top1_correct": view.top1_correct.tolist(),
        "values_per_row": view.values_per_row,
        "fixed_bytes": view.fixed_bytes,
        "total_bytes": view.total_bytes,
        "query_projection_seconds": view.query_projection_seconds,
        "descriptor_build_seconds": view.descriptor_build_seconds,
        "search_seconds": view.search_seconds,
    }


def _fit_payload(fit) -> dict[str, object]:
    return {
        "dimension": fit.dimension,
        "basis_kind": fit.basis_kind,
        "width": fit.width,
        "neighbors": fit.neighbors,
        "lambda_raw": fit.lambda_raw,
        "lambda_value": fit.lambda_value,
    }


def evaluate_bundle(
    bundle: EmbeddingBundle,
    *,
    widths: tuple[int, ...] = (64, 128, 256, 512),
    neighbors: int = 50,
    null_seeds: range = range(206, 238),
) -> dict[str, object]:
    """Fit CTM on train rows and evaluate the frozen UNICOM control grid."""

    if type(bundle) is not EmbeddingBundle:
        raise TypeError("bundle must be an EmbeddingBundle")
    dimension = bundle.train_embeddings.shape[1]
    active_widths = tuple(width for width in widths if width < dimension - 1)
    if not active_widths:
        raise ValueError("no registered width fits the embedding dimension")
    primary_width = 128 if 128 in active_widths else active_widths[0]
    phase_start = time.perf_counter()
    train_native = l2_normalize(bundle.train_embeddings)
    query_native = l2_normalize(bundle.query_embeddings)
    gallery_native = l2_normalize(bundle.gallery_embeddings)
    normalization_seconds = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    pca_basis = fit_projection_basis(train_native, kind="pca")
    basis_fit_seconds = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    train_pca = project_unit(train_native, pca_basis)
    query_projection_start = time.perf_counter()
    query_pca = project_unit(query_native, pca_basis)
    query_projection_seconds = time.perf_counter() - query_projection_start
    gallery_pca = project_unit(gallery_native, pca_basis)
    native_fits = {
        width: fit_tail_moment(train_native, width=width, basis_kind="native", neighbors=neighbors)
        for width in active_widths
    }
    pca_fits = {
        width: fit_tail_moment(train_pca, width=width, basis_kind="pca", neighbors=neighbors)
        for width in active_widths
    }
    coefficient_fit_seconds = time.perf_counter() - phase_start
    primary_fit = native_fits[primary_width]
    phase_start = time.perf_counter()
    interval = cluster_lambda_interval(
        train_native,
        bundle.train_labels,
        width=primary_width,
        basis_kind="native",
        neighbors=neighbors,
    )
    null = permuted_tail_null(
        train_native,
        width=primary_width,
        basis_kind="native",
        neighbors=neighbors,
        seeds=null_seeds,
    )
    tail = train_native[:, primary_width:].astype(np.float64)
    radius = np.sqrt(np.sum(tail * tail, axis=1, dtype=np.float64))
    scalar = (np.sqrt(primary_fit.lambda_value) * radius).astype(np.float32)
    scalar_nonzero = bool(np.any(scalar != 0.0))
    falsifier_seconds = time.perf_counter() - phase_start
    base = {
        "schema_version": 1,
        "phase": "FIT_CLOSED",
        "input": {
            "bundle_path": str(bundle.path),
            "bundle_sha256": bundle.sha256,
            "metadata": bundle.metadata,
            "array_sha256": {
                "train_embeddings": _array_sha256(bundle.train_embeddings),
                "train_labels": _array_sha256(bundle.train_labels),
                "query_embeddings": _array_sha256(bundle.query_embeddings),
                "query_labels": _array_sha256(bundle.query_labels),
                "gallery_embeddings": _array_sha256(bundle.gallery_embeddings),
                "gallery_labels": _array_sha256(bundle.gallery_labels),
            },
            "embedding_dimension": dimension,
            "query_count": bundle.query_embeddings.shape[0],
            "gallery_count": bundle.gallery_embeddings.shape[0],
        },
        "constants": {
            "widths": list(active_widths),
            "neighbors": neighbors,
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 205,
            "null_seeds": list(null_seeds),
        },
        "fits": {
            "primary_width": primary_width,
            "native": {str(width): _fit_payload(native_fits[width]) for width in active_widths},
            "pca": {str(width): _fit_payload(pca_fits[width]) for width in active_widths},
            "primary_scalar_nonzero": scalar_nonzero,
            "null": {
                "seeds": list(null.seeds),
                "lambda_raw_values": list(null.lambda_raw_values),
                "p_value": null.p_value,
            },
        },
        "views": None,
        "primary": None,
        "timing": {
            "normalization_seconds": normalization_seconds,
            "basis_fit_seconds": basis_fit_seconds,
            "coefficient_fit_seconds": coefficient_fit_seconds,
            "falsifier_seconds": falsifier_seconds,
            "evaluation_seconds": 0.0,
        },
        "storage": None,
        "runtime": runtime_payload(),
        "decision": {"status": "CLOSE", "reason": "primary coefficient clipped to zero"},
    }
    if primary_fit.lambda_value == 0.0 or not scalar_nonzero:
        validate_ctm_report(
            base,
            expected_widths=widths,
            expected_neighbors=neighbors,
            expected_null_count=len(null_seeds),
        )
        return base

    evaluation_start = time.perf_counter()
    pca_fixed_bytes = pca_basis.mean.nbytes + pca_basis.matrix.nbytes
    official_width = 512 if dimension >= 512 else dimension
    views = {
        width: evaluate_width(
            query_native,
            gallery_native,
            query_pca,
            gallery_pca,
            bundle.query_labels,
            bundle.gallery_labels,
            width=width,
            native_fit=native_fits[width],
            pca_fit=pca_fits[width],
            pca_fixed_bytes=pca_fixed_bytes,
            official_width=official_width,
            pca_query_projection_seconds=query_projection_seconds,
        )
        for width in active_widths
    }
    primary_views = views[primary_width]
    paired = query_identity_interval(
        primary_views["renormalized_prefix_plus_zero"].top1_correct,
        primary_views["native_ctm"].top1_correct,
        bundle.query_labels,
    )
    control_r1 = {key: primary_views[key].recall[1] for key in MATCHED_ROW_WIDTH_CONTROLS}
    control_bytes = {key: primary_views[key].total_bytes for key in EQUAL_TOTAL_STORAGE_CONTROLS}
    width_gains = tuple(
        (
            width,
            width_views["native_ctm"].recall[1]
            - width_views["renormalized_prefix_plus_zero"].recall[1],
        )
        for width, width_views in views.items()
    )
    decision = ctm_decision(
        ctm_r1=primary_views["native_ctm"].recall[1],
        ctm_map_at_r=primary_views["native_ctm"].map_at_r,
        renormalized_r1=primary_views["renormalized_prefix_plus_zero"].recall[1],
        renormalized_map_at_r=primary_views["renormalized_prefix_plus_zero"].map_at_r,
        official_512_r1=primary_views["official_512"].recall[1],
        paired_lower=paired.lower,
        control_r1=control_r1,
        ctm_total_bytes=primary_views["native_ctm"].total_bytes,
        control_total_bytes=control_bytes,
        lambda_raw=primary_fit.lambda_raw,
        lambda_lower=interval.lower,
        null_p_value=null.p_value,
        width_gains=width_gains,
        replication_status="PENDING",
    )
    base["phase"] = "EVALUATED"
    base["views"] = {
        str(width): {key: _view_payload(view) for key, view in width_views.items()}
        for width, width_views in views.items()
    }
    base["primary"] = {
        "paired": {
            "point": paired.point,
            "lower": paired.lower,
            "upper": paired.upper,
            "samples": paired.samples,
            "seed": paired.seed,
        },
        "lambda_interval": {
            "point": interval.point,
            "lower": interval.lower,
            "upper": interval.upper,
        },
        "null_p_value": null.p_value,
        "width_gains": [[width, gain] for width, gain in width_gains],
        "replication_status": "PENDING",
    }
    base["timing"]["evaluation_seconds"] = time.perf_counter() - evaluation_start
    base["storage"] = {
        "native_ctm_bytes": primary_views["native_ctm"].total_bytes,
        "official_512_bytes": primary_views["official_512"].total_bytes,
        "pca_fixed_bytes": pca_fixed_bytes,
    }
    base["decision"] = {"status": decision.status, "reason": decision.reason}
    validate_ctm_report(
        base,
        expected_widths=widths,
        expected_neighbors=neighbors,
        expected_null_count=len(null_seeds),
    )
    return base
