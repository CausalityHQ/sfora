"""Pure NumPy diagnostics for frozen UNICOM retrieval embeddings."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.unicom_inshop import InshopRecord

RECALL_AT_K = (1, 10, 20, 30)


def strict_typed_equal(left: object, right: object) -> bool:
    """Compare canonical JSON-like values without Python numeric coercion."""

    if type(left) is not type(right):
        return False
    if type(left) is dict:
        left_items = tuple(left.items())
        right_items = tuple(right.items())
        return len(left_items) == len(right_items) and all(
            strict_typed_equal(left_key, right_key)
            and strict_typed_equal(left_value, right_value)
            for (left_key, left_value), (right_key, right_value) in zip(
                left_items, right_items, strict=True
            )
        )
    if type(left) in (list, tuple):
        return len(left) == len(right) and all(
            strict_typed_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    return bool(left == right)


def _canonical_posix_relative_path(value: object, *, name: str) -> str:
    if type(value) is not str or not value or "\\" in value:
        raise ValueError(f"{name} differs")
    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    canonical = posix.as_posix()
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or bool(windows.drive)
        or canonical != value
        or any(part in {"", ".", ".."} for part in posix.parts)
    ):
        raise ValueError(f"{name} differs")
    return canonical


@dataclass(frozen=True)
class RetrievalView:
    recall: dict[int, float]
    map_at_r: float
    top1_indices: np.ndarray
    top1_correct: np.ndarray
    average_precision: np.ndarray | None = None


@dataclass(frozen=True)
class LogicalInshopRecord:
    """Relocation-stable In-Shop record stored in evaluation evidence."""

    image_name: str
    label: str


@dataclass(frozen=True)
class GeometryDecision:
    primary: str
    full_dimension_control: bool
    evaluator_repair: bool
    coordinate_nonexchangeability: bool


@dataclass(frozen=True)
class GeometryConfig:
    selected_coordinates: int
    random_mask_count: int
    bootstrap_samples: int
    expected_official_r1: float
    reproduction_tolerance: float
    delta_threshold: float = 0.002
    mask_wins_threshold: int = 24
    disagreement_threshold: float = 0.10
    norm_bootstrap_seed: int = 205
    energy_bootstrap_seed: int = 205
    full_bootstrap_seed: int = 206
    random_mask_seed_start: int = 0


@dataclass(frozen=True, eq=False)
class GeometryAudit:
    config: GeometryConfig
    official: RetrievalView
    prefix_unit: RetrievalView
    full_unit: RetrievalView
    random_units: tuple[RetrievalView, ...]
    reproduction_passed: bool
    delta_norm: float
    norm_interval: tuple[float, float]
    delta_full: float
    full_interval: tuple[float, float]
    delta_mask: float
    mask_wins: int
    disagree: float
    energy_disagreement_count: int
    energy_gap_mean: float | None
    energy_gap_median: float | None
    energy_gap_negative_fraction: float | None
    energy_gap_interval: tuple[float, float] | None
    error_energy_point_biserial: float | None
    decision: GeometryDecision

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GeometryAudit):
            return NotImplemented

        def view_equal(left: RetrievalView, right: RetrievalView) -> bool:
            return (
                left.recall == right.recall
                and left.map_at_r == right.map_at_r
                and np.array_equal(left.top1_indices, right.top1_indices)
                and np.array_equal(left.top1_correct, right.top1_correct)
                and (
                    (left.average_precision is None and right.average_precision is None)
                    or np.array_equal(left.average_precision, right.average_precision)
                )
            )

        return (
            self.config == other.config
            and view_equal(self.official, other.official)
            and view_equal(self.prefix_unit, other.prefix_unit)
            and view_equal(self.full_unit, other.full_unit)
            and len(self.random_units) == len(other.random_units)
            and all(
                view_equal(left, right)
                for left, right in zip(self.random_units, other.random_units, strict=True)
            )
            and self.reproduction_passed == other.reproduction_passed
            and self.delta_norm == other.delta_norm
            and self.norm_interval == other.norm_interval
            and self.delta_full == other.delta_full
            and self.full_interval == other.full_interval
            and self.delta_mask == other.delta_mask
            and self.mask_wins == other.mask_wins
            and self.disagree == other.disagree
            and self.energy_disagreement_count == other.energy_disagreement_count
            and self.energy_gap_mean == other.energy_gap_mean
            and self.energy_gap_median == other.energy_gap_median
            and self.energy_gap_negative_fraction == other.energy_gap_negative_fraction
            and self.energy_gap_interval == other.energy_gap_interval
            and self.error_energy_point_biserial == other.error_energy_point_biserial
            and self.decision == other.decision
        )


def _require_fp32_matrix(values: np.ndarray, *, name: str = "embeddings") -> None:
    if type(values) is not np.ndarray:
        raise TypeError(f"{name} must be a NumPy array")
    if values.dtype != np.float32:
        raise TypeError(f"{name} must have dtype float32")
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError(f"{name} must be a nonempty matrix")
    if not values.flags.c_contiguous:
        raise ValueError(f"{name} must be C-contiguous")
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite")


def l2_normalize(values: np.ndarray) -> np.ndarray:
    """Normalize FP32 rows with ordered FP64 reductions."""

    _require_fp32_matrix(values)
    squared = values.astype(np.float64) ** 2
    norms = np.sqrt(np.sum(squared, axis=1, keepdims=True, dtype=np.float64))
    if np.any(norms == 0.0):
        raise ValueError("embedding row has zero L2 norm")
    return np.ascontiguousarray((values.astype(np.float64) / norms).astype(np.float32))


def random_masks(
    *, dimension: int, selected: int, count: int, seed_start: int = 0
) -> tuple[np.ndarray, ...]:
    if any(type(value) is not int for value in (dimension, selected, count, seed_start)):
        raise TypeError("mask parameters must be builtin integers")
    if dimension <= 0 or selected <= 0 or selected > dimension or count <= 0 or seed_start < 0:
        raise ValueError("invalid random-mask dimensions")
    return tuple(
        np.sort(
            np.random.Generator(np.random.PCG64(seed_start + offset)).choice(
                dimension, selected, replace=False
            )
        ).astype(np.int64)
        for offset in range(count)
    )


def _validate_labels(labels: np.ndarray, rows: int, *, name: str) -> None:
    if type(labels) is not np.ndarray or labels.ndim != 1 or labels.shape[0] != rows:
        raise ValueError(f"{name} must be a one-dimensional array matching embedding rows")
    if labels.dtype.kind not in {"U", "S", "O"}:
        raise TypeError(f"{name} must contain identity strings")
    if any(type(value) is not str or not value for value in labels.tolist()):
        raise TypeError(f"{name} must contain nonempty builtin strings")


def _validate_coordinates(coordinates: np.ndarray, dimension: int) -> np.ndarray:
    if type(coordinates) is not np.ndarray or coordinates.dtype != np.int64:
        raise TypeError("coordinates must be an int64 NumPy array")
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("coordinates must be a nonempty vector")
    if np.any(coordinates < 0) or np.any(coordinates >= dimension):
        raise ValueError("coordinate is outside the embedding dimension")
    if not np.array_equal(coordinates, np.unique(coordinates)):
        raise ValueError("coordinates must be unique and sorted")
    return coordinates


def _selected_view(
    values: np.ndarray,
    coordinates: np.ndarray,
    *,
    normalize_before: bool,
) -> np.ndarray:
    if type(normalize_before) is not bool:
        raise TypeError("normalize_before must be a builtin bool")
    if normalize_before:
        return np.ascontiguousarray(l2_normalize(values)[:, coordinates])
    selected = np.ascontiguousarray(values[:, coordinates])
    return l2_normalize(selected)


def _stable_top_indices(distances: np.ndarray, count: int) -> np.ndarray:
    """Select the exact stable distance prefix without sorting the full gallery."""

    if count >= distances.size:
        candidates = np.arange(distances.size, dtype=np.int64)
    else:
        partition = np.argpartition(distances, count - 1)[:count]
        boundary = np.max(distances[partition])
        lower = np.flatnonzero(distances < boundary)
        tied = np.flatnonzero(distances == boundary)[: count - lower.size]
        candidates = np.concatenate((lower, tied))
    return candidates[np.lexsort((candidates, distances[candidates]))]


def _stable_top_score_indices(scores: np.ndarray, count: int) -> np.ndarray:
    """Select the exact stable descending-score prefix."""

    if count >= scores.size:
        candidates = np.arange(scores.size, dtype=np.int64)
    else:
        partition = np.argpartition(scores, scores.size - count)[-count:]
        boundary = np.min(scores[partition])
        higher = np.flatnonzero(scores > boundary)
        tied = np.flatnonzero(scores == boundary)[: count - higher.size]
        candidates = np.concatenate((higher, tied))
    return candidates[np.lexsort((candidates, -scores[candidates]))]


def canonical_logical_record(
    record: InshopRecord, dataset_root: Path
) -> LogicalInshopRecord:
    """Convert one existing image into its real Img-relative logical identity."""

    if type(record) is not InshopRecord or not isinstance(dataset_root, Path):
        raise TypeError("In-Shop logical record input differs")
    if type(record.label) is not str or not record.label:
        raise ValueError("In-Shop logical record label differs")
    absolute_root = dataset_root.absolute()
    resolved_root = dataset_root.resolve()
    image_root = resolved_root / "Img"
    if (
        absolute_root != resolved_root
        or not resolved_root.is_dir()
        or resolved_root.is_symlink()
        or not image_root.is_dir()
        or image_root.is_symlink()
    ):
        raise ValueError("In-Shop dataset root differs")
    image_path = record.image_path
    if not isinstance(image_path, Path):
        raise TypeError("In-Shop image path differs")
    absolute_path = image_path.absolute()
    resolved_path = image_path.resolve()
    try:
        relative = resolved_path.relative_to(image_root)
    except ValueError as error:
        raise ValueError("In-Shop image path is not a real Img descendant") from error
    unresolved = image_root
    for part in relative.parts:
        if part in {"", ".", ".."}:
            raise ValueError("In-Shop image path differs")
        unresolved /= part
        if unresolved.is_symlink():
            raise ValueError("In-Shop image path contains a symlink")
    if (
        absolute_path != resolved_path
        or not relative.parts
        or not resolved_path.is_file()
        or resolved_path.is_symlink()
    ):
        raise ValueError("In-Shop image path is not a real Img descendant")
    return LogicalInshopRecord(image_name=relative.as_posix(), label=record.label)


def _canonical_logical_records(
    records: Sequence[InshopRecord], dataset_root: Path, *, name: str
) -> tuple[LogicalInshopRecord, ...]:
    if type(records) is not tuple or not records:
        raise ValueError(f"{name} must be a nonempty ordered tuple")
    logical = tuple(canonical_logical_record(record, dataset_root) for record in records)
    paths = tuple(record.image_name for record in logical)
    if len(paths) != len(set(paths)):
        raise ValueError(f"{name} contains duplicate image paths")
    return logical


def tensor_sha256(values: np.ndarray) -> str:
    """Hash exact canonical NumPy tensor bytes."""

    if type(values) is not np.ndarray or not values.flags.c_contiguous:
        raise ValueError("tensor hash input differs")
    return hashlib.sha256(values.tobytes(order="C")).hexdigest()


def ranking_sha256(scores: np.ndarray, indices: np.ndarray) -> str:
    """Hash a complete ascending FP64-distance/int64-index ranking."""

    if (
        type(scores) is not np.ndarray
        or scores.dtype != np.float64
        or scores.ndim != 1
        or not scores.flags.c_contiguous
        or not np.isfinite(scores).all()
        or type(indices) is not np.ndarray
        or indices.dtype != np.int64
        or indices.shape != scores.shape
        or not indices.flags.c_contiguous
    ):
        raise ValueError("ranking hash input differs")
    digest = hashlib.sha256(b"unicom-squared-euclidean-ranking-v1\0")
    digest.update(scores.tobytes(order="C"))
    digest.update(indices.tobytes(order="C"))
    return digest.hexdigest()


def query_evidence(
    *,
    query_values: np.ndarray,
    gallery_values: np.ndarray,
    query_records: tuple[InshopRecord, ...],
    gallery_records: tuple[InshopRecord, ...],
    dataset_root: Path,
    coordinates: np.ndarray,
    normalize_before: bool,
) -> tuple[dict[str, object], ...]:
    """Build exact per-query evidence under the registered retrieval geometry."""

    _require_fp32_matrix(query_values, name="query_values")
    _require_fp32_matrix(gallery_values, name="gallery_values")
    if query_values.shape[1] != gallery_values.shape[1]:
        raise ValueError("query and gallery descriptor dimensions differ")
    coordinates = _validate_coordinates(coordinates, query_values.shape[1])
    if type(normalize_before) is not bool:
        raise TypeError("normalize_before must be a builtin bool")
    query_logical = _canonical_logical_records(
        query_records, dataset_root, name="query_records"
    )
    gallery_logical = _canonical_logical_records(
        gallery_records, dataset_root, name="gallery_records"
    )
    if len(query_logical) != query_values.shape[0]:
        raise ValueError("query tensor/record row count differs")
    if len(gallery_logical) != gallery_values.shape[0]:
        raise ValueError("gallery tensor/record row count differs")
    all_paths = tuple(record.image_name for record in query_logical + gallery_logical)
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("query/gallery image paths overlap")
    return _query_evidence_from_logical(
        query_values=query_values,
        gallery_values=gallery_values,
        query_records=query_logical,
        gallery_records=gallery_logical,
        coordinates=coordinates,
        normalize_before=normalize_before,
    )


def _query_evidence_from_logical(
    *,
    query_values: np.ndarray,
    gallery_values: np.ndarray,
    query_records: tuple[LogicalInshopRecord, ...],
    gallery_records: tuple[LogicalInshopRecord, ...],
    coordinates: np.ndarray,
    normalize_before: bool,
) -> tuple[dict[str, object], ...]:
    gallery_label_counts = Counter(record.label for record in gallery_records)
    if any(record.label not in gallery_label_counts for record in query_records):
        raise ValueError("query label is absent from gallery label map")

    normalized_query = l2_normalize(query_values)
    normalized_gallery = l2_normalize(gallery_values)
    if normalize_before:
        selected_query = np.ascontiguousarray(normalized_query[:, coordinates])
        selected_gallery = np.ascontiguousarray(normalized_gallery[:, coordinates])
    else:
        selected_query = l2_normalize(np.ascontiguousarray(query_values[:, coordinates]))
        selected_gallery = l2_normalize(
            np.ascontiguousarray(gallery_values[:, coordinates])
        )
        normalized_query = selected_query
    gallery64 = selected_gallery.astype(np.float64)
    gallery_norms = np.sum(gallery64 * gallery64, axis=1, dtype=np.float64)
    gallery_indices = np.arange(gallery_values.shape[0], dtype=np.int64)
    rows: list[dict[str, object]] = []
    for query_index, (query_record, selected_row) in enumerate(
        zip(query_records, selected_query, strict=True)
    ):
        query64 = selected_row.astype(np.float64)
        distances = np.ascontiguousarray(
            np.sum(query64 * query64, dtype=np.float64)
            + gallery_norms
            - 2.0 * (gallery64 @ query64)
        )
        order = np.ascontiguousarray(
            np.lexsort((gallery_indices, distances)).astype(np.int64, copy=False)
        )
        ordered_scores = np.ascontiguousarray(distances[order])
        relevant = gallery_label_counts[query_record.label]
        prefix_length = min(max(max(RECALL_AT_K), relevant), len(gallery_records))
        ranked_prefix = tuple(
            {
                "gallery_index": int(gallery_index),
                "gallery_path": gallery_records[gallery_index].image_name,
                "gallery_label": gallery_records[gallery_index].label,
                "score": float(distances[gallery_index]),
                "correct": gallery_records[gallery_index].label == query_record.label,
            }
            for gallery_index in order[:prefix_length]
        )
        matches = np.asarray(
            [row["correct"] for row in ranked_prefix[:relevant]], dtype=np.bool_
        )
        precision = np.cumsum(matches, dtype=np.int64) / np.arange(1, relevant + 1)
        average_precision = float(np.sum(precision * matches) / relevant)
        rows.append(
            {
                "query_path": query_record.image_name,
                "query_label": query_record.label,
                "relevant_gallery_count": relevant,
                "ranked_prefix": ranked_prefix,
                "ap_at_r": average_precision,
                "query_sha256": tensor_sha256(
                    np.ascontiguousarray(normalized_query[query_index])
                ),
                "complete_ranking_sha256": ranking_sha256(ordered_scores, order),
            }
        )
    return tuple(rows)


def recompute_query_metrics(
    rows: tuple[dict[str, object], ...],
) -> dict[str, float]:
    """Strictly recompute aggregate metrics from ranked per-query prefixes."""

    if type(rows) is not tuple or not rows:
        raise ValueError("query evidence must be a nonempty ordered tuple")
    recalls = {key: [] for key in RECALL_AT_K}
    average_precisions: list[float] = []
    query_paths: set[str] = set()
    expected_row_keys = (
        "query_path",
        "query_label",
        "relevant_gallery_count",
        "ranked_prefix",
        "ap_at_r",
        "query_sha256",
        "complete_ranking_sha256",
    )
    expected_rank_keys = (
        "gallery_index",
        "gallery_path",
        "gallery_label",
        "score",
        "correct",
    )
    for row in rows:
        if type(row) is not dict or tuple(row) != expected_row_keys:
            raise ValueError("query evidence row differs")
        query_path = _canonical_posix_relative_path(
            row["query_path"], name="query evidence row"
        )
        if (
            query_path in query_paths
            or type(row["query_label"]) is not str
            or not row["query_label"]
            or type(row["relevant_gallery_count"]) is not int
            or row["relevant_gallery_count"] <= 0
            or type(row["ranked_prefix"]) is not tuple
            or len(row["ranked_prefix"]) < row["relevant_gallery_count"]
            or type(row["ap_at_r"]) is not float
            or not np.isfinite(row["ap_at_r"])
            or not 0.0 <= row["ap_at_r"] <= 1.0
            or type(row["query_sha256"]) is not str
            or len(row["query_sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["query_sha256"])
            or type(row["complete_ranking_sha256"]) is not str
            or len(row["complete_ranking_sha256"]) != 64
            or any(
                character not in "0123456789abcdef"
                for character in row["complete_ranking_sha256"]
            )
        ):
            raise ValueError("query evidence row differs")
        query_paths.add(query_path)
        indices: set[int] = set()
        gallery_paths: set[str] = set()
        previous: tuple[float, int] | None = None
        matches = []
        for ranked in row["ranked_prefix"]:
            if type(ranked) is not dict or tuple(ranked) != expected_rank_keys:
                raise ValueError("ranked query evidence differs")
            gallery_path = _canonical_posix_relative_path(
                ranked["gallery_path"], name="ranked query evidence"
            )
            if (
                type(ranked["gallery_index"]) is not int
                or ranked["gallery_index"] < 0
                or ranked["gallery_index"] in indices
                or gallery_path in gallery_paths
                or type(ranked["gallery_label"]) is not str
                or not ranked["gallery_label"]
                or type(ranked["score"]) is not float
                or not np.isfinite(ranked["score"])
                or type(ranked["correct"]) is not bool
                or ranked["correct"] != (
                    ranked["gallery_label"] == row["query_label"]
                )
            ):
                raise ValueError("ranked query evidence differs")
            key = (ranked["score"], ranked["gallery_index"])
            if previous is not None and key < previous:
                raise ValueError("ranked query evidence differs")
            previous = key
            indices.add(ranked["gallery_index"])
            gallery_paths.add(gallery_path)
            matches.append(ranked["correct"])
        relevant = row["relevant_gallery_count"]
        truncated = np.asarray(matches[:relevant], dtype=np.bool_)
        precision = np.cumsum(truncated, dtype=np.int64) / np.arange(1, relevant + 1)
        ap_at_r = float(np.sum(precision * truncated) / relevant)
        if row["ap_at_r"] != ap_at_r:
            raise ValueError("query AP@R differs")
        for key in recalls:
            recalls[key].append(bool(np.any(matches[:key])))
        average_precisions.append(ap_at_r)
    return {
        "recall_at_1": float(np.mean(recalls[1])),
        "recall_at_10": float(np.mean(recalls[10])),
        "recall_at_20": float(np.mean(recalls[20])),
        "recall_at_30": float(np.mean(recalls[30])),
        "map_at_r": float(np.mean(average_precisions)),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _descriptor_binding(path: Path, *, evidence_root: Path) -> dict[str, object]:
    values = np.load(path, allow_pickle=False)
    _require_fp32_matrix(values, name="persisted descriptors")
    relative = path.resolve().relative_to(evidence_root.resolve())
    return {
        "path": relative.as_posix(),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
        "shape": list(values.shape),
        "dtype": "float32",
        "c_order": True,
    }


def _json_query_rows(rows: tuple[dict[str, object], ...]) -> list[dict[str, object]]:
    return [
        {
            **{key: value for key, value in row.items() if key != "ranked_prefix"},
            "ranked_prefix": [dict(ranked) for ranked in row["ranked_prefix"]],
        }
        for row in rows
    ]


def _query_rows_from_json(value: object) -> tuple[dict[str, object], ...]:
    json_row_keys = (
        "query_path",
        "query_label",
        "relevant_gallery_count",
        "ap_at_r",
        "query_sha256",
        "complete_ranking_sha256",
        "ranked_prefix",
    )
    ranked_keys = (
        "gallery_index",
        "gallery_path",
        "gallery_label",
        "score",
        "correct",
    )
    if type(value) is not list or not value:
        raise ValueError("per-query evaluation evidence differs")
    rows = []
    for row in value:
        if (
            type(row) is not dict
            or tuple(row) != json_row_keys
            or type(row["ranked_prefix"]) is not list
        ):
            raise ValueError("per-query evaluation evidence differs")
        ranked_prefix = []
        for ranked in row["ranked_prefix"]:
            if type(ranked) is not dict or tuple(ranked) != ranked_keys:
                raise ValueError("per-query evaluation evidence differs")
            ranked_prefix.append(dict(ranked))
        rows.append(
            {
                "query_path": row["query_path"],
                "query_label": row["query_label"],
                "relevant_gallery_count": row["relevant_gallery_count"],
                "ranked_prefix": tuple(ranked_prefix),
                "ap_at_r": row["ap_at_r"],
                "query_sha256": row["query_sha256"],
                "complete_ranking_sha256": row["complete_ranking_sha256"],
            }
        )
    return tuple(rows)


def _validate_evaluation_metrics(value: object) -> None:
    keys = (
        "recall_at_1",
        "recall_at_10",
        "recall_at_20",
        "recall_at_30",
        "map_at_r",
    )
    if (
        type(value) is not dict
        or tuple(value) != keys
        or any(
            type(value[key]) is not float
            or not math.isfinite(value[key])
            or not 0.0 <= value[key] <= 1.0
            for key in keys
        )
    ):
        raise ValueError("evaluation aggregate metrics differ")


def _encode_npy(values: np.ndarray) -> bytes:
    import io

    output = io.BytesIO()
    np.save(output, values, allow_pickle=False)
    return output.getvalue()


def _write_npy_exclusive(path: Path, values: np.ndarray):
    return _publish_bytes_retained(path, _encode_npy(values))


def _write_json_exclusive(path: Path, value: object):
    payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
    return _publish_bytes_retained(path, payload)


def _publish_bytes_retained(path: Path, payload: bytes):
    return publish_bytes_noreplace(
        path,
        payload,
        validator=lambda persisted: (
            None
            if persisted == payload
            else (_ for _ in ()).throw(ValueError("evaluation evidence differs"))
        ),
    )


def write_evaluation_evidence(
    *,
    query_values: np.ndarray,
    gallery_values: np.ndarray,
    query_records: tuple[InshopRecord, ...],
    gallery_records: tuple[InshopRecord, ...],
    dataset_root: Path,
    coordinates: np.ndarray,
    normalize_before: bool,
    epoch: int,
    evidence_root: Path,
    publication_guard=lambda _component, _destination, _payload: None,
) -> dict[str, object]:
    """Persist immutable descriptor preimages and their recomputable receipt."""

    if type(epoch) is not int or epoch not in (4, 8, 12, 16):
        raise ValueError("evaluation epoch differs")
    if not isinstance(evidence_root, Path):
        raise TypeError("evaluation evidence root must be a Path")
    absolute_root = evidence_root.absolute()
    resolved_root = evidence_root.resolve()
    if (
        absolute_root != resolved_root
        or not resolved_root.is_dir()
        or resolved_root.is_symlink()
    ):
        raise ValueError("evaluation evidence root differs")
    _require_fp32_matrix(query_values, name="query_values")
    _require_fp32_matrix(gallery_values, name="gallery_values")
    if query_values.shape[1] != 768 or gallery_values.shape[1] != 768:
        raise ValueError("evaluation descriptor width differs")
    expected_coordinates = np.arange(512, dtype=np.int64)
    if not np.array_equal(coordinates, expected_coordinates) or normalize_before is not True:
        raise ValueError("evaluation geometry differs")
    logical_query = _canonical_logical_records(
        query_records, dataset_root, name="query_records"
    )
    logical_gallery = _canonical_logical_records(
        gallery_records, dataset_root, name="gallery_records"
    )
    rows = query_evidence(
        query_values=query_values,
        gallery_values=gallery_values,
        query_records=query_records,
        gallery_records=gallery_records,
        dataset_root=dataset_root,
        coordinates=coordinates,
        normalize_before=normalize_before,
    )
    metrics = recompute_query_metrics(rows)
    stem = f"evaluation-epoch-{epoch:04d}"
    query_path = resolved_root / f"{stem}-query.npy"
    gallery_path = resolved_root / f"{stem}-gallery.npy"
    ranked_path = resolved_root / f"{stem}-ranked-prefix.json"
    receipt_path = resolved_root / f"{stem}.json"
    created = []
    try:
        query_payload = _encode_npy(np.ascontiguousarray(query_values))
        publication_guard("query", query_path, query_payload)
        published = _write_npy_exclusive(
            query_path, np.ascontiguousarray(query_values)
        )
        if published.payload != query_payload:
            raise RuntimeError("query publication bytes differ")
        created.append((query_path, published))
        gallery_payload = _encode_npy(np.ascontiguousarray(gallery_values))
        publication_guard("gallery", gallery_path, gallery_payload)
        published = _write_npy_exclusive(
            gallery_path, np.ascontiguousarray(gallery_values)
        )
        if published.payload != gallery_payload:
            raise RuntimeError("gallery publication bytes differ")
        created.append((gallery_path, published))
        ranked_rows = _json_query_rows(rows)
        ranked_payload = (
            json.dumps(ranked_rows, indent=2, allow_nan=False) + "\n"
        ).encode()
        publication_guard("ranked-prefix", ranked_path, ranked_payload)
        published = _write_json_exclusive(ranked_path, ranked_rows)
        if published.payload != ranked_payload:
            raise RuntimeError("ranked-prefix publication bytes differ")
        created.append((ranked_path, published))
        receipt: dict[str, object] = {
            "schema": "unicom-evaluation-evidence-v1",
            "epoch": epoch,
            "geometry": {
                "input_dimension": 768,
                "coordinates": list(range(512)),
                "normalize_before": True,
                "ranking": "ascending_squared_euclidean",
            },
            "query_descriptors": _descriptor_binding(
                query_path, evidence_root=resolved_root
            ),
            "gallery_descriptors": _descriptor_binding(
                gallery_path, evidence_root=resolved_root
            ),
            "query_records": [record.__dict__ for record in logical_query],
            "gallery_records": [record.__dict__ for record in logical_gallery],
            "ranked_prefix_evidence": {
                "path": ranked_path.relative_to(resolved_root).as_posix(),
                "sha256": hashlib.sha256(ranked_payload).hexdigest(),
                "bytes": len(ranked_payload),
            },
            "metrics": metrics,
            "evaluation_signature": {
                "descriptor_dtype": "float32",
                "descriptor_dimension": 512,
                "descriptor_sha256": tensor_sha256(
                    np.ascontiguousarray(
                        np.concatenate(
                            (l2_normalize(query_values), l2_normalize(gallery_values))
                        )[:, :512]
                    )
                ),
                "operations": [
                    "official_forward",
                    "full768_l2",
                    "prefix512",
                    "squared_euclidean",
                ],
            },
        }
        validate_evaluation_evidence(receipt, resolved_root)
        receipt_payload = (json.dumps(receipt, indent=2, allow_nan=False) + "\n").encode()
        publication_guard("receipt", receipt_path, receipt_payload)
        published = _write_json_exclusive(receipt_path, receipt)
        if published.payload != receipt_payload:
            raise RuntimeError("evaluation receipt publication bytes differ")
        created.append((receipt_path, published))
        persisted = json.loads(receipt_path.read_bytes())
        validate_evaluation_evidence(persisted, resolved_root)
        for _path, publication in created:
            publication.close()
        return receipt
    except Exception:
        directory_descriptor = os.open(resolved_root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for path, publication in reversed(created):
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    publication.close()
                    continue
                if (info.st_dev, info.st_ino) == publication.identity:
                    path.unlink()
                    os.fsync(directory_descriptor)
                publication.close()
        finally:
            os.close(directory_descriptor)
        raise


def _logical_inventory(value: object, *, name: str) -> tuple[LogicalInshopRecord, ...]:
    if type(value) is not list or not value:
        raise ValueError(f"{name} inventory differs")
    records = []
    paths: set[str] = set()
    for row in value:
        if type(row) is not dict or tuple(row) != ("image_name", "label"):
            raise ValueError(f"{name} inventory differs")
        relative = _canonical_posix_relative_path(
            row["image_name"], name=f"{name} inventory"
        )
        if (
            relative in paths
            or type(row["label"]) is not str
            or not row["label"]
        ):
            raise ValueError(f"{name} inventory differs")
        paths.add(relative)
        records.append(LogicalInshopRecord(relative, row["label"]))
    return tuple(records)


def _load_bound_descriptors(
    binding: object, *, evidence_root: Path, name: str
) -> np.ndarray:
    keys = ("path", "sha256", "bytes", "shape", "dtype", "c_order")
    if type(binding) is not dict or tuple(binding) != keys:
        raise ValueError(f"{name} descriptor binding differs")
    relative = Path(binding["path"])
    if (
        type(binding["path"]) is not str
        or not binding["path"]
        or relative.is_absolute()
        or any(part in {"", ".", ".."} for part in relative.parts)
    ):
        raise ValueError(f"{name} descriptor path differs")
    unresolved = evidence_root
    for part in relative.parts:
        unresolved /= part
        if unresolved.is_symlink():
            raise ValueError(f"{name} descriptor path differs")
    resolved = unresolved.resolve()
    try:
        resolved.relative_to(evidence_root)
    except ValueError as error:
        raise ValueError(f"{name} descriptor path differs") from error
    try:
        with resolved.open("rb") as handle:
            version = np.lib.format.read_magic(handle)
            if version != (1, 0):
                raise ValueError(f"{name} descriptor NumPy format differs")
            header_shape, fortran_order, header_dtype = (
                np.lib.format.read_array_header_1_0(handle)
            )
            data_offset = handle.tell()
    except Exception as error:
        raise ValueError(f"{name} descriptor NumPy format differs") from error
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or type(binding["sha256"]) is not str
        or len(binding["sha256"]) != 64
        or type(binding["bytes"]) is not int
        or binding["bytes"] <= 0
        or resolved.stat().st_size != binding["bytes"]
        or _sha256_file(resolved) != binding["sha256"]
        or type(binding["shape"]) is not list
        or len(binding["shape"]) != 2
        or any(type(dimension) is not int or dimension <= 0 for dimension in binding["shape"])
        or binding["dtype"] != "float32"
        or binding["c_order"] is not True
        or list(header_shape) != binding["shape"]
        or fortran_order is not False
        or header_dtype != np.dtype(np.float32)
        or resolved.stat().st_size
        != data_offset + math.prod(header_shape) * header_dtype.itemsize
    ):
        raise ValueError(f"{name} descriptor bytes differ")
    try:
        values = np.load(resolved, allow_pickle=False)
    except Exception as error:
        raise ValueError(f"{name} descriptor payload differs") from error
    _require_fp32_matrix(values, name=f"{name} descriptors")
    if list(values.shape) != binding["shape"]:
        raise ValueError(f"{name} descriptor shape differs")
    return values


def validate_evaluation_evidence(receipt: object, evidence_root: Path) -> None:
    """Strict-load descriptor preimages and rebuild every recorded metric."""

    keys = (
        "schema",
        "epoch",
        "geometry",
        "query_descriptors",
        "gallery_descriptors",
        "query_records",
        "gallery_records",
        "ranked_prefix_evidence",
        "metrics",
        "evaluation_signature",
    )
    if (
        type(receipt) is not dict
        or tuple(receipt) != keys
        or receipt["schema"] != "unicom-evaluation-evidence-v1"
        or type(receipt["epoch"]) is not int
        or receipt["epoch"] not in (4, 8, 12, 16)
    ):
        raise ValueError("evaluation evidence receipt differs")
    if not isinstance(evidence_root, Path):
        raise TypeError("evaluation evidence root must be a Path")
    absolute_root = evidence_root.absolute()
    resolved_root = evidence_root.resolve()
    if (
        absolute_root != resolved_root
        or not resolved_root.is_dir()
        or resolved_root.is_symlink()
    ):
        raise ValueError("evaluation evidence root differs")
    geometry = receipt["geometry"]
    if (
        type(geometry) is not dict
        or tuple(geometry)
        != ("input_dimension", "coordinates", "normalize_before", "ranking")
        or type(geometry["input_dimension"]) is not int
        or geometry["input_dimension"] != 768
        or type(geometry["coordinates"]) is not list
        or len(geometry["coordinates"]) != 512
        or any(type(coordinate) is not int for coordinate in geometry["coordinates"])
        or not strict_typed_equal(geometry["coordinates"], list(range(512)))
        or type(geometry["normalize_before"]) is not bool
        or not geometry["normalize_before"]
        or type(geometry["ranking"]) is not str
        or geometry["ranking"] != "ascending_squared_euclidean"
    ):
        raise ValueError("evaluation geometry differs")
    signature = receipt["evaluation_signature"]
    if (
        type(signature) is not dict
        or tuple(signature)
        != (
            "descriptor_dtype",
            "descriptor_dimension",
            "descriptor_sha256",
            "operations",
        )
        or type(signature["descriptor_dtype"]) is not str
        or signature["descriptor_dtype"] != "float32"
        or type(signature["descriptor_dimension"]) is not int
        or signature["descriptor_dimension"] != 512
        or type(signature["descriptor_sha256"]) is not str
        or len(signature["descriptor_sha256"]) != 64
        or any(
            character not in "0123456789abcdef"
            for character in signature["descriptor_sha256"]
        )
        or type(signature["operations"]) is not list
        or any(type(operation) is not str for operation in signature["operations"])
        or not strict_typed_equal(
            signature["operations"],
            [
                "official_forward",
                "full768_l2",
                "prefix512",
                "squared_euclidean",
            ],
        )
    ):
        raise ValueError("evaluation signature differs")
    query_records = _logical_inventory(receipt["query_records"], name="query")
    gallery_records = _logical_inventory(receipt["gallery_records"], name="gallery")
    all_paths = tuple(record.image_name for record in query_records + gallery_records)
    if len(all_paths) != len(set(all_paths)):
        raise ValueError("evaluation record paths overlap")
    stem = f"evaluation-epoch-{receipt['epoch']:04d}"
    if (
        type(receipt["query_descriptors"]) is not dict
        or receipt["query_descriptors"].get("path") != f"{stem}-query.npy"
    ):
        raise ValueError("query descriptor path differs")
    if (
        type(receipt["gallery_descriptors"]) is not dict
        or receipt["gallery_descriptors"].get("path") != f"{stem}-gallery.npy"
    ):
        raise ValueError("gallery descriptor path differs")
    ranked_binding = receipt["ranked_prefix_evidence"]
    ranked_path = resolved_root / f"{stem}-ranked-prefix.json"
    if (
        type(ranked_binding) is not dict
        or tuple(ranked_binding) != ("path", "sha256", "bytes")
        or ranked_binding["path"] != ranked_path.name
        or ranked_path.is_symlink()
        or not ranked_path.is_file()
    ):
        raise ValueError("ranked-prefix evidence authority differs")
    ranked_payload = ranked_path.read_bytes()
    if (
        hashlib.sha256(ranked_payload).hexdigest() != ranked_binding["sha256"]
        or len(ranked_payload) != ranked_binding["bytes"]
    ):
        raise ValueError("ranked-prefix evidence authority differs")
    try:
        ranked_value = json.loads(ranked_payload)
    except (TypeError, ValueError) as error:
        raise ValueError("ranked-prefix evidence authority differs") from error
    if ranked_payload != (
        json.dumps(ranked_value, indent=2, allow_nan=False) + "\n"
    ).encode():
        raise ValueError("ranked-prefix evidence authority differs")
    query_values = _load_bound_descriptors(
        receipt["query_descriptors"], evidence_root=resolved_root, name="query"
    )
    gallery_values = _load_bound_descriptors(
        receipt["gallery_descriptors"], evidence_root=resolved_root, name="gallery"
    )
    if (
        query_values.shape != (len(query_records), 768)
        or gallery_values.shape != (len(gallery_records), 768)
    ):
        raise ValueError("evaluation descriptor inventory differs")
    deployed_descriptors = np.ascontiguousarray(
        np.concatenate((l2_normalize(query_values), l2_normalize(gallery_values)))[:, :512]
    )
    if signature["descriptor_sha256"] != tensor_sha256(deployed_descriptors):
        raise ValueError("evaluation signature descriptor differs")
    rebuilt = _query_evidence_from_logical(
        query_values=query_values,
        gallery_values=gallery_values,
        query_records=query_records,
        gallery_records=gallery_records,
        coordinates=np.arange(512, dtype=np.int64),
        normalize_before=True,
    )
    supplied_rows = _query_rows_from_json(ranked_value)
    supplied_metrics = recompute_query_metrics(supplied_rows)
    _validate_evaluation_metrics(receipt["metrics"])
    if not strict_typed_equal(supplied_metrics, receipt["metrics"]):
        raise ValueError("evaluation aggregate metrics differ")
    if not strict_typed_equal(ranked_value, _json_query_rows(rebuilt)):
        raise ValueError("per-query evaluation evidence differs")
    metrics = recompute_query_metrics(rebuilt)
    if not strict_typed_equal(receipt["metrics"], metrics):
        raise ValueError("evaluation aggregate metrics differ")


def retrieval_metrics_from_score_chunks(
    score_chunks: Iterable[np.ndarray],
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    recall_at_k: tuple[int, ...] | None = None,
) -> RetrievalView:
    """Reduce ordered FP64 score chunks with the repository retrieval contract."""

    if type(query_labels) is not np.ndarray or query_labels.ndim != 1:
        raise ValueError("query_labels must be a one-dimensional array")
    if type(gallery_labels) is not np.ndarray or gallery_labels.ndim != 1:
        raise ValueError("gallery_labels must be a one-dimensional array")
    _validate_labels(query_labels, query_labels.shape[0], name="query_labels")
    _validate_labels(gallery_labels, gallery_labels.shape[0], name="gallery_labels")
    if recall_at_k is None:
        recall_at_k = RECALL_AT_K
    if (
        type(recall_at_k) is not tuple
        or not recall_at_k
        or any(type(key) is not int or key <= 0 for key in recall_at_k)
        or tuple(sorted(set(recall_at_k))) != recall_at_k
    ):
        raise ValueError("recall-at-k contract differs")
    recall_hits = {key: [] for key in recall_at_k}
    average_precisions: list[float] = []
    top1_indices: list[int] = []
    gallery_label_counts = Counter(gallery_labels.tolist())
    query_index = 0
    for chunk in score_chunks:
        if (
            type(chunk) is not np.ndarray
            or chunk.dtype != np.float64
            or chunk.ndim != 2
            or chunk.shape[1] != gallery_labels.shape[0]
            or not chunk.flags.c_contiguous
            or not np.isfinite(chunk).all()
        ):
            raise ValueError("score chunk contract differs")
        for row in chunk:
            if query_index >= query_labels.shape[0]:
                raise ValueError("score chunks contain too many query rows")
            relevant = gallery_label_counts.get(query_labels[query_index], 0)
            if relevant == 0:
                raise ValueError("query identity has no relevant gallery item")
            required = min(max(max(recall_hits), relevant), gallery_labels.shape[0])
            order = _stable_top_score_indices(row, required)
            matches = gallery_labels[order] == query_labels[query_index]
            top1_indices.append(int(order[0]))
            for key in recall_hits:
                recall_hits[key].append(bool(np.any(matches[: min(key, matches.size)])))
            truncated = matches[:relevant]
            precision = np.cumsum(truncated, dtype=np.int64) / np.arange(1, relevant + 1)
            average_precisions.append(float(np.sum(precision * truncated) / relevant))
            query_index += 1
    if query_index != query_labels.shape[0]:
        raise ValueError("score chunks contain too few query rows")
    top1 = np.asarray(top1_indices, dtype=np.int64)
    top1_correct = np.asarray(gallery_labels[top1] == query_labels, dtype=np.bool_)
    return RetrievalView(
        recall={key: float(np.mean(values)) for key, values in recall_hits.items()},
        map_at_r=float(np.mean(average_precisions)),
        top1_indices=top1,
        top1_correct=top1_correct,
        average_precision=np.asarray(average_precisions, dtype=np.float64),
    )


def retrieval_view(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    coordinates: np.ndarray,
    normalize_before: bool,
    chunk_size: int = 256,
    recall_at_k: tuple[int, ...] | None = None,
) -> RetrievalView:
    """Evaluate one deterministic selected-coordinate retrieval view."""

    _require_fp32_matrix(query_embeddings, name="query_embeddings")
    _require_fp32_matrix(gallery_embeddings, name="gallery_embeddings")
    if query_embeddings.shape[1] != gallery_embeddings.shape[1]:
        raise ValueError("query and gallery embedding dimensions differ")
    _validate_labels(query_labels, query_embeddings.shape[0], name="query_labels")
    _validate_labels(gallery_labels, gallery_embeddings.shape[0], name="gallery_labels")
    coordinates = _validate_coordinates(coordinates, query_embeddings.shape[1])
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive builtin integer")

    query = _selected_view(query_embeddings, coordinates, normalize_before=normalize_before)
    gallery = _selected_view(gallery_embeddings, coordinates, normalize_before=normalize_before)
    gallery64 = gallery.astype(np.float64)
    gallery_norms = np.sum(gallery64 * gallery64, axis=1, dtype=np.float64)

    def score_chunks() -> Iterable[np.ndarray]:
        for start in range(0, query.shape[0], chunk_size):
            query_chunk = query[start : start + chunk_size].astype(np.float64)
            query_norms = np.sum(query_chunk * query_chunk, axis=1, dtype=np.float64)
            distances = (
                query_norms[:, None] + gallery_norms[None, :] - 2.0 * (query_chunk @ gallery64.T)
            )
            yield np.ascontiguousarray(-distances)

    if recall_at_k is None:
        return retrieval_metrics_from_score_chunks(score_chunks(), query_labels, gallery_labels)
    return retrieval_metrics_from_score_chunks(
        score_chunks(), query_labels, gallery_labels, recall_at_k=recall_at_k
    )


def paired_r1_interval(
    baseline_correct: np.ndarray,
    candidate_correct: np.ndarray,
    *,
    samples: int = 10_000,
    seed: int = 205,
) -> tuple[float, float]:
    if (
        type(baseline_correct) is not np.ndarray
        or type(candidate_correct) is not np.ndarray
        or baseline_correct.dtype != np.bool_
        or candidate_correct.dtype != np.bool_
        or baseline_correct.ndim != 1
        or candidate_correct.shape != baseline_correct.shape
        or baseline_correct.size == 0
    ):
        raise TypeError("paired correctness arrays must be matching nonempty boolean vectors")
    if type(samples) is not int or samples <= 0 or type(seed) is not int:
        raise ValueError("samples and seed must be builtin integers with samples positive")
    differences = candidate_correct.astype(np.float64) - baseline_correct.astype(np.float64)
    return _bootstrap_mean_interval(differences, samples=samples, seed=seed)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    seed: int,
    chunk_size: int = 64,
) -> tuple[float, float]:
    generator = np.random.Generator(np.random.PCG64(seed))
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, chunk_size):
        stop = min(samples, start + chunk_size)
        indices = generator.integers(0, values.size, size=(stop - start, values.size))
        means[start:stop] = values[indices].mean(axis=1, dtype=np.float64)
    bounds = np.percentile(means, [2.5, 97.5])
    return float(bounds[0]), float(bounds[1])


def geometry_decision(
    *,
    delta_norm: float,
    norm_lower_bound: float,
    delta_full: float,
    full_lower_bound: float,
    delta_mask: float,
    mask_wins: int,
    disagree: float,
    delta_threshold: float = 0.002,
    mask_wins_threshold: int = 24,
    disagreement_threshold: float = 0.10,
) -> GeometryDecision:
    values = (
        delta_norm,
        norm_lower_bound,
        delta_full,
        full_lower_bound,
        delta_mask,
        disagree,
    )
    if any(type(value) is not float or not np.isfinite(value) for value in values):
        raise TypeError("decision metrics must be finite builtin floats")
    if type(mask_wins) is not int or not 0 <= mask_wins <= 32:
        raise TypeError("mask_wins must be a builtin integer in [0, 32]")
    if any(
        type(value) is not float or not np.isfinite(value)
        for value in (delta_threshold, disagreement_threshold)
    ):
        raise TypeError("decision thresholds must be finite builtin floats")
    if type(mask_wins_threshold) is not int or not 0 <= mask_wins_threshold <= 32:
        raise TypeError("mask-wins threshold must be a builtin integer in [0, 32]")
    full_dimension_control = delta_full >= delta_threshold and full_lower_bound > 0.0
    evaluator_repair = delta_norm >= delta_threshold and norm_lower_bound > 0.0
    coordinate = (
        delta_mask >= delta_threshold
        and mask_wins >= mask_wins_threshold
        and disagree >= disagreement_threshold
    )
    if full_dimension_control:
        primary = "FULL_DIMENSION_CONTROL"
    elif evaluator_repair:
        primary = "EVALUATOR_REPAIR"
    elif coordinate:
        primary = "COORDINATE_NONEXCHANGEABILITY"
    else:
        primary = "GEOMETRY_NULL"
    return GeometryDecision(
        primary=primary,
        full_dimension_control=full_dimension_control,
        evaluator_repair=evaluator_repair,
        coordinate_nonexchangeability=coordinate,
    )


def audit_deployment_geometry(
    query_embeddings: np.ndarray,
    gallery_embeddings: np.ndarray,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    selected: int = 512,
    random_count: int = 32,
    bootstrap_samples: int = 10_000,
    expected_official_r1: float = 0.746,
    reproduction_tolerance: float = 0.002,
) -> GeometryAudit:
    """Run all registered retrieval views and reduce them to one audit result."""

    if type(selected) is not int or selected <= 0 or selected > query_embeddings.shape[1]:
        raise ValueError("selected must be within the embedding dimension")
    if type(random_count) is not int or random_count <= 0:
        raise ValueError("random_count must be positive")
    if type(expected_official_r1) is not float or type(reproduction_tolerance) is not float:
        raise TypeError("reproduction values must be builtin floats")
    if not np.isfinite(expected_official_r1) or not np.isfinite(reproduction_tolerance):
        raise ValueError("reproduction values must be finite")
    if reproduction_tolerance < 0.0:
        raise ValueError("reproduction_tolerance must be nonnegative")
    config = GeometryConfig(
        selected_coordinates=selected,
        random_mask_count=random_count,
        bootstrap_samples=bootstrap_samples,
        expected_official_r1=expected_official_r1,
        reproduction_tolerance=reproduction_tolerance,
    )

    prefix = np.arange(config.selected_coordinates, dtype=np.int64)
    full = np.arange(query_embeddings.shape[1], dtype=np.int64)
    official = retrieval_view(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        coordinates=prefix,
        normalize_before=True,
    )
    prefix_unit = retrieval_view(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        coordinates=prefix,
        normalize_before=False,
    )
    full_unit = retrieval_view(
        query_embeddings,
        gallery_embeddings,
        query_labels,
        gallery_labels,
        coordinates=full,
        normalize_before=False,
    )
    random_units = tuple(
        retrieval_view(
            query_embeddings,
            gallery_embeddings,
            query_labels,
            gallery_labels,
            coordinates=mask,
            normalize_before=False,
        )
        for mask in random_masks(
            dimension=query_embeddings.shape[1],
            selected=config.selected_coordinates,
            count=config.random_mask_count,
            seed_start=config.random_mask_seed_start,
        )
    )

    delta_norm = float(prefix_unit.recall[1] - official.recall[1])
    norm_interval = paired_r1_interval(
        official.top1_correct,
        prefix_unit.top1_correct,
        samples=config.bootstrap_samples,
        seed=config.norm_bootstrap_seed,
    )
    delta_full = float(full_unit.recall[1] - prefix_unit.recall[1])
    full_interval = paired_r1_interval(
        prefix_unit.top1_correct,
        full_unit.top1_correct,
        samples=config.bootstrap_samples,
        seed=config.full_bootstrap_seed,
    )
    random_r1 = np.asarray([view.recall[1] for view in random_units], dtype=np.float64)
    delta_mask = float(np.median(random_r1) - prefix_unit.recall[1])
    mask_wins = int(np.count_nonzero(random_r1 > prefix_unit.recall[1]))
    disagreement_rates = np.asarray(
        [np.mean(view.top1_indices != prefix_unit.top1_indices) for view in random_units],
        dtype=np.float64,
    )
    disagree = float(np.median(disagreement_rates))

    gallery_full_unit = l2_normalize(gallery_embeddings)
    prefix_energy = np.sum(
        gallery_full_unit[:, : config.selected_coordinates].astype(np.float64) ** 2,
        axis=1,
        dtype=np.float64,
    )
    disagreement = official.top1_indices != prefix_unit.top1_indices
    energy_gaps = (
        prefix_energy[official.top1_indices[disagreement]]
        - prefix_energy[prefix_unit.top1_indices[disagreement]]
    )
    if energy_gaps.size:
        energy_gap_mean: float | None = float(np.mean(energy_gaps, dtype=np.float64))
        energy_gap_median: float | None = float(np.median(energy_gaps))
        energy_gap_negative_fraction: float | None = float(np.mean(energy_gaps < 0.0))
        energy_gap_interval: tuple[float, float] | None = _bootstrap_mean_interval(
            energy_gaps,
            samples=config.bootstrap_samples,
            seed=config.energy_bootstrap_seed,
        )
    else:
        energy_gap_mean = None
        energy_gap_median = None
        energy_gap_negative_fraction = None
        energy_gap_interval = None

    errors = (~official.top1_correct).astype(np.float64)
    selected_energy = prefix_energy[official.top1_indices]
    if np.std(errors) == 0.0 or np.std(selected_energy) == 0.0:
        association: float | None = None
    else:
        association = float(np.corrcoef(errors, selected_energy)[0, 1])

    reproduction_passed = (
        abs(full_unit.recall[1] - config.expected_official_r1) <= config.reproduction_tolerance
    )
    if reproduction_passed:
        decision = geometry_decision(
            delta_norm=delta_norm,
            norm_lower_bound=norm_interval[0],
            delta_full=delta_full,
            full_lower_bound=full_interval[0],
            delta_mask=delta_mask,
            mask_wins=mask_wins,
            disagree=disagree,
            delta_threshold=config.delta_threshold,
            mask_wins_threshold=config.mask_wins_threshold,
            disagreement_threshold=config.disagreement_threshold,
        )
    else:
        decision = GeometryDecision(
            primary="REPRODUCTION_FAILED",
            full_dimension_control=False,
            evaluator_repair=False,
            coordinate_nonexchangeability=False,
        )
    return GeometryAudit(
        config=config,
        official=official,
        prefix_unit=prefix_unit,
        full_unit=full_unit,
        random_units=random_units,
        reproduction_passed=reproduction_passed,
        delta_norm=delta_norm,
        norm_interval=norm_interval,
        delta_full=delta_full,
        full_interval=full_interval,
        delta_mask=delta_mask,
        mask_wins=mask_wins,
        disagree=disagree,
        energy_disagreement_count=int(energy_gaps.size),
        energy_gap_mean=energy_gap_mean,
        energy_gap_median=energy_gap_median,
        energy_gap_negative_fraction=energy_gap_negative_fraction,
        energy_gap_interval=energy_gap_interval,
        error_energy_point_biserial=association,
        decision=decision,
    )
