"""Strict embedding-bundle IO and outcome reporting for the UNICOM audit."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sfora.unicom_retrieval_audit import GeometryAudit, RetrievalView
from sfora.unicom_shard_audit import ShardAudit

_BUNDLE_KEYS = (
    "metadata_json",
    "train_embeddings",
    "train_labels",
    "query_embeddings",
    "query_labels",
    "gallery_embeddings",
    "gallery_labels",
)
_ARRAY_KEYS = _BUNDLE_KEYS[1:]
_METADATA_KEYS = (
    "schema_version",
    "model_identifier",
    "model_revision",
    "checkpoint_sha256",
    "image_list_sha256",
    "transform",
    "embedding_dimension",
    "split_counts",
    "array_sha256",
)
_REPORT_KEYS = (
    "schema_version",
    "input",
    "constants",
    "geometry",
    "sharding",
    "runtime",
    "warnings",
)
_HEX = frozenset("0123456789abcdef")


@dataclass(frozen=True)
class EmbeddingBundle:
    path: Path
    sha256: str
    metadata: dict[str, object]
    train_embeddings: np.ndarray
    train_labels: np.ndarray
    query_embeddings: np.ndarray
    query_labels: np.ndarray
    gallery_embeddings: np.ndarray
    gallery_labels: np.ndarray


def _is_sha256(value: object) -> bool:
    return type(value) is str and len(value) == 64 and set(value) <= _HEX


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(values: np.ndarray) -> str:
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def _require_exact_keys(value: object, keys: tuple[str, ...], *, name: str) -> dict:
    if type(value) is not dict or tuple(value) != keys:
        raise ValueError(f"{name} key order differs")
    return value


def _strict_json_object(text: str, *, name: str) -> dict[str, object]:
    def reject_constant(token: str) -> object:
        raise ValueError(f"{name} contains nonfinite JSON number {token}")

    try:
        value = json.loads(text, parse_constant=reject_constant)
    except (json.JSONDecodeError, TypeError) as error:
        raise ValueError(f"{name} is not strict JSON") from error
    if type(value) is not dict:
        raise TypeError(f"{name} must decode to an object")
    return value


def _require_fp32_matrix(values: np.ndarray, *, rows: int, dimension: int, name: str) -> None:
    if type(values) is not np.ndarray or values.dtype != np.float32:
        raise TypeError(f"{name} must be a float32 NumPy array")
    if values.shape != (rows, dimension) or not values.flags.c_contiguous:
        raise ValueError(f"{name} shape/order differs")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")
    norms = np.linalg.norm(values.astype(np.float64), axis=1)
    if np.any(norms == 0.0):
        raise ValueError(f"{name} contains a zero-norm row")


def _require_labels(values: np.ndarray, *, rows: int, name: str) -> None:
    if type(values) is not np.ndarray or values.ndim != 1 or values.shape != (rows,):
        raise ValueError(f"{name} shape differs")
    if values.dtype.kind not in {"U", "S"}:
        raise TypeError(f"{name} must be a fixed-width string array")
    if any(type(label) is not str or not label for label in values.tolist()):
        raise TypeError(f"{name} must contain nonempty builtin strings")


def load_embedding_bundle(
    path: Path,
    *,
    expected_counts: tuple[int, int, int] = (25_882, 14_218, 12_612),
    expected_dimension: int = 768,
) -> EmbeddingBundle:
    """Load and authenticate one immutable NumPy embedding bundle."""

    path = Path(path)
    if (
        type(expected_counts) is not tuple
        or len(expected_counts) != 3
        or any(type(value) is not int or value <= 0 for value in expected_counts)
        or type(expected_dimension) is not int
        or expected_dimension <= 0
    ):
        raise ValueError("expected shape constants are invalid")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or path.is_symlink():
        raise ValueError("bundle path must be a regular non-symlink file")

    with np.load(path, allow_pickle=False) as archive:
        if tuple(archive.files) != _BUNDLE_KEYS:
            raise ValueError("bundle key order differs")
        raw_metadata = archive["metadata_json"]
        if raw_metadata.shape != () or raw_metadata.dtype.kind not in {"U", "S"}:
            raise TypeError("metadata_json must be one scalar string")
        metadata = _strict_json_object(str(raw_metadata), name="metadata_json")
        arrays = {name: np.ascontiguousarray(archive[name]) for name in _ARRAY_KEYS}

    _require_exact_keys(metadata, _METADATA_KEYS, name="metadata")
    if metadata["schema_version"] != 1:
        raise ValueError("metadata schema version differs")
    for key in ("model_identifier", "transform"):
        if type(metadata[key]) is not str or not metadata[key]:
            raise TypeError(f"metadata {key} must be a nonempty string")
    if not _is_sha256(metadata["checkpoint_sha256"]) or not _is_sha256(
        metadata["image_list_sha256"]
    ):
        raise ValueError("metadata SHA-256 differs")
    revision = metadata["model_revision"]
    if type(revision) is not str or len(revision) != 40 or not set(revision) <= _HEX:
        raise ValueError("metadata model revision differs")
    if metadata["embedding_dimension"] != expected_dimension:
        raise ValueError("embedding dimension differs")
    split_counts = _require_exact_keys(
        metadata["split_counts"], ("train", "query", "gallery"), name="split count"
    )
    if tuple(split_counts.values()) != expected_counts:
        raise ValueError("split count differs")
    hashes = _require_exact_keys(metadata["array_sha256"], _ARRAY_KEYS, name="array hash")

    train_count, query_count, gallery_count = expected_counts
    _require_fp32_matrix(
        arrays["train_embeddings"],
        rows=train_count,
        dimension=expected_dimension,
        name="train_embeddings",
    )
    _require_fp32_matrix(
        arrays["query_embeddings"],
        rows=query_count,
        dimension=expected_dimension,
        name="query_embeddings",
    )
    _require_fp32_matrix(
        arrays["gallery_embeddings"],
        rows=gallery_count,
        dimension=expected_dimension,
        name="gallery_embeddings",
    )
    for split, count in zip(("train", "query", "gallery"), expected_counts, strict=True):
        _require_labels(arrays[f"{split}_labels"], rows=count, name=f"{split}_labels")
    gallery_identities = set(arrays["gallery_labels"].tolist())
    if not set(arrays["query_labels"].tolist()) <= gallery_identities:
        raise ValueError("query label membership differs from gallery")
    for name, values in arrays.items():
        if not _is_sha256(hashes[name]) or hashes[name] != _array_sha256(values):
            raise ValueError(f"array hash differs for {name}")
        values.flags.writeable = False

    return EmbeddingBundle(
        path=path.resolve(),
        sha256=_file_sha256(path),
        metadata=metadata,
        train_embeddings=arrays["train_embeddings"],
        train_labels=arrays["train_labels"],
        query_embeddings=arrays["query_embeddings"],
        query_labels=arrays["query_labels"],
        gallery_embeddings=arrays["gallery_embeddings"],
        gallery_labels=arrays["gallery_labels"],
    )


def _view_payload(view: RetrievalView, *, include_rows: bool) -> dict[str, object]:
    payload: dict[str, object] = {
        "recall": {str(key): value for key, value in view.recall.items()},
        "map_at_r": view.map_at_r,
    }
    if include_rows:
        payload["top1_indices"] = view.top1_indices.tolist()
        payload["top1_correct"] = view.top1_correct.tolist()
    return payload


def build_audit_report(
    bundle: EmbeddingBundle,
    geometry: GeometryAudit,
    sharding: ShardAudit,
    *,
    selected_coordinates: int = 512,
    geometry_bootstrap_samples: int = 10_000,
    panel_classes: int = 64,
    examples_per_class: int = 4,
) -> dict[str, object]:
    geometry_payload = {
        "official": _view_payload(geometry.official, include_rows=True),
        "prefix_unit": _view_payload(geometry.prefix_unit, include_rows=True),
        "full_unit": _view_payload(geometry.full_unit, include_rows=False),
        "random_units": [
            _view_payload(view, include_rows=False) for view in geometry.random_units
        ],
        "reproduction_passed": geometry.reproduction_passed,
        "delta_norm": geometry.delta_norm,
        "norm_interval": list(geometry.norm_interval),
        "delta_mask": geometry.delta_mask,
        "mask_wins": geometry.mask_wins,
        "disagree": geometry.disagree,
        "energy_disagreement_count": geometry.energy_disagreement_count,
        "energy_gap_mean": geometry.energy_gap_mean,
        "energy_gap_median": geometry.energy_gap_median,
        "energy_gap_negative_fraction": geometry.energy_gap_negative_fraction,
        "energy_gap_interval": (
            list(geometry.energy_gap_interval) if geometry.energy_gap_interval is not None else None
        ),
        "error_energy_point_biserial": geometry.error_energy_point_biserial,
        "decision": {
            "primary": geometry.decision.primary,
            "evaluator_repair": geometry.decision.evaluator_repair,
            "coordinate_nonexchangeability": (
                geometry.decision.coordinate_nonexchangeability
            ),
        },
    }
    sharding_payload = {
        "trials": sharding.trials,
        "permutations_per_trial": sharding.permutations_per_trial,
        "independent_loss_range": sharding.independent_loss_range,
        "independent_loss_std": sharding.independent_loss_std,
        "independent_gradient_mse": sharding.independent_gradient_mse,
        "coherent_gradient_mse": sharding.coherent_gradient_mse,
        "independent_gradient_cosine_distance": (
            sharding.independent_gradient_cosine_distance
        ),
        "coherent_gradient_cosine_distance": sharding.coherent_gradient_cosine_distance,
        "coherent_invariance_error": sharding.coherent_invariance_error,
        "prediction_change_rate": sharding.prediction_change_rate,
        "mask_union_coverage": sharding.mask_union_coverage,
        "all_finite": sharding.all_finite,
        "decision": sharding.decision,
    }
    return {
        "schema_version": 1,
        "input": {
            "bundle_path": str(bundle.path),
            "bundle_sha256": bundle.sha256,
            "metadata": bundle.metadata,
        },
        "constants": {
            "selected_coordinates": selected_coordinates,
            "random_masks": len(geometry.random_units),
            "geometry_bootstrap_samples": geometry_bootstrap_samples,
            "panel_classes": panel_classes,
            "examples_per_class": examples_per_class,
            "shard_trials": sharding.trials,
            "permutations_per_trial": sharding.permutations_per_trial,
        },
        "geometry": geometry_payload,
        "sharding": sharding_payload,
        "runtime": {"python": platform.python_version(), "numpy": str(np.__version__)},
        "warnings": [],
    }


def _finite_json_tree(value: object) -> None:
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not np.isfinite(value):
            raise ValueError("report contains a nonfinite float")
        return
    if type(value) is list:
        for item in value:
            _finite_json_tree(item)
        return
    if type(value) is dict:
        if any(type(key) is not str for key in value):
            raise TypeError("report object keys must be strings")
        for item in value.values():
            _finite_json_tree(item)
        return
    raise TypeError("report contains a non-JSON concrete type")


def validate_audit_report(value: object) -> None:
    report = _require_exact_keys(value, _REPORT_KEYS, name="report")
    _finite_json_tree(report)
    if report["schema_version"] != 1:
        raise ValueError("report schema version differs")
    input_value = _require_exact_keys(
        report["input"], ("bundle_path", "bundle_sha256", "metadata"), name="report input"
    )
    if type(input_value["bundle_path"]) is not str or not input_value["bundle_path"]:
        raise TypeError("report bundle path differs")
    if not _is_sha256(input_value["bundle_sha256"]):
        raise ValueError("report bundle SHA-256 differs")
    _require_exact_keys(input_value["metadata"], _METADATA_KEYS, name="report metadata")
    constants = _require_exact_keys(
        report["constants"],
        (
            "selected_coordinates",
            "random_masks",
            "geometry_bootstrap_samples",
            "panel_classes",
            "examples_per_class",
            "shard_trials",
            "permutations_per_trial",
        ),
        name="report constants",
    )
    if any(type(value) is not int or value <= 0 for value in constants.values()):
        raise ValueError("report constants differ")
    geometry = _require_exact_keys(
        report["geometry"],
        (
            "official",
            "prefix_unit",
            "full_unit",
            "random_units",
            "reproduction_passed",
            "delta_norm",
            "norm_interval",
            "delta_mask",
            "mask_wins",
            "disagree",
            "energy_disagreement_count",
            "energy_gap_mean",
            "energy_gap_median",
            "energy_gap_negative_fraction",
            "energy_gap_interval",
            "error_energy_point_biserial",
            "decision",
        ),
        name="report geometry",
    )
    if (
        type(geometry["random_units"]) is not list
        or len(geometry["random_units"]) != constants["random_masks"]
        or type(geometry["decision"]) is not dict
    ):
        raise ValueError("report geometry count/schema differs")
    if geometry["decision"].get("primary") not in {
        "EVALUATOR_REPAIR",
        "COORDINATE_NONEXCHANGEABILITY",
        "GEOMETRY_NULL",
        "REPRODUCTION_FAILED",
    }:
        raise ValueError("geometry decision differs")
    sharding = _require_exact_keys(
        report["sharding"],
        (
            "trials",
            "permutations_per_trial",
            "independent_loss_range",
            "independent_loss_std",
            "independent_gradient_mse",
            "coherent_gradient_mse",
            "independent_gradient_cosine_distance",
            "coherent_gradient_cosine_distance",
            "coherent_invariance_error",
            "prediction_change_rate",
            "mask_union_coverage",
            "all_finite",
            "decision",
        ),
        name="report sharding",
    )
    if sharding["decision"] not in {
        "SHARD_SENSITIVE",
        "SHARD_NULL",
    }:
        raise ValueError("sharding decision differs")
    if (
        sharding["all_finite"] is not True
        or sharding["trials"] != constants["shard_trials"]
        or sharding["permutations_per_trial"] != constants["permutations_per_trial"]
    ):
        raise ValueError("sharding finite gate failed")
    if type(report["runtime"]) is not dict or tuple(report["runtime"]) != (
        "python",
        "numpy",
    ):
        raise ValueError("runtime schema differs")
    if type(report["warnings"]) is not list or any(
        type(item) is not str for item in report["warnings"]
    ):
        raise TypeError("warnings schema differs")


def publish_json_no_clobber(path: Path, payload: Mapping[str, object]) -> None:
    """Publish a validated JSON report with exclusive hard-link installation."""

    path = Path(path)
    validate_audit_report(payload)
    parent_info = path.parent.lstat()
    if not stat.S_ISDIR(parent_info.st_mode) or path.parent.is_symlink():
        raise ValueError("output parent must be a real directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    encoded = (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor: int | None = None
    owned: tuple[int, int] | None = None
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        temporary_info = os.fstat(descriptor)
        owned = (temporary_info.st_dev, temporary_info.st_ino)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = None
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        persisted = _strict_json_object(path.read_text(encoding="utf-8"), name="report")
        validate_audit_report(persisted)
        if persisted != payload:
            raise ValueError("persisted report differs from input")
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            info = temporary.lstat()
        except FileNotFoundError:
            pass
        else:
            if owned is not None and (info.st_dev, info.st_ino) == owned:
                temporary.unlink()
