"""Deterministic geometry for Local-Excess Iso-Density Gradient Projection."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FOLD_DOMAIN = b"LE-IDGP-fold-v1:"
ROW_DOMAIN = b"LE-IDGP-row-v1:"
COHORT_LABELS = 45
ROWS_PER_LABEL = 4


@dataclass(frozen=True)
class Cohort:
    """One deterministic 45-label by four-row cohort and its disjoint reference."""

    fold: int
    index: int
    row_indices: NDArray[np.int64]
    reference_row_indices: NDArray[np.int64]
    ordered_sha256: str


def normalize_rows(value: NDArray[np.floating]) -> NDArray[np.float64]:
    """Return finite nonzero rows normalized in float64."""

    rows = np.asarray(value, dtype=np.float64)
    if rows.ndim != 2:
        raise ValueError("embeddings must be two-dimensional")
    if not np.isfinite(rows).all():
        raise ValueError("embeddings must be finite")
    norms = np.linalg.norm(rows, axis=1)
    if np.any(norms == 0.0):
        raise ValueError("embedding rows must be nonzero")
    return rows / norms[:, None]


def _validate_labels(labels: NDArray[np.int64]) -> NDArray[np.int64]:
    value = np.asarray(labels)
    if value.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    if value.dtype != np.dtype(np.int64):
        raise ValueError("labels must have exact int64 dtype")
    return value


def _label_digest(label: int) -> bytes:
    encoded = np.asarray(label, dtype="<i8").tobytes()
    return hashlib.sha256(FOLD_DOMAIN + encoded).digest()


def assign_label_folds(labels: NDArray[np.int64]) -> dict[int, int]:
    """Assign each exact int64 label using the low two bits of its frozen hash."""

    value = _validate_labels(labels)
    return {
        int(label): _label_digest(int(label))[0] & 3 for label in sorted(np.unique(value).tolist())
    }


def _row_digest(example_id: str) -> bytes:
    return hashlib.sha256(ROW_DOMAIN + example_id.encode("utf-8")).digest()


def build_cohorts(labels: NDArray[np.int64], example_ids: NDArray[np.str_]) -> tuple[Cohort, ...]:
    """Build complete deterministic cohorts and cyclic disjoint references."""

    label_array = _validate_labels(labels)
    id_array = np.asarray(example_ids)
    if id_array.ndim != 1 or id_array.shape[0] != label_array.shape[0]:
        raise ValueError("example IDs must align with labels")
    if id_array.dtype.kind != "U":
        raise ValueError("example IDs must have exact Unicode dtype")
    if any(not value for value in id_array.tolist()):
        raise ValueError("example IDs must be nonempty")
    if np.unique(id_array).size != id_array.size:
        raise ValueError("example IDs must be unique")

    folds = assign_label_folds(label_array)
    complete_rows: list[tuple[int, list[NDArray[np.int64]]]] = []
    for fold in range(4):
        eligible = [
            label
            for label, assigned in folds.items()
            if assigned == fold and np.count_nonzero(label_array == label) >= ROWS_PER_LABEL
        ]
        eligible.sort(key=lambda label: (_label_digest(label), label))
        groups: list[NDArray[np.int64]] = []
        for start in range(0, len(eligible) - COHORT_LABELS + 1, COHORT_LABELS):
            group = eligible[start : start + COHORT_LABELS]
            selected: list[int] = []
            for label in group:
                candidates = np.flatnonzero(label_array == label).tolist()
                candidates.sort(
                    key=lambda row: (_row_digest(str(id_array[row])), str(id_array[row]))
                )
                selected.extend(candidates[:ROWS_PER_LABEL])
            groups.append(np.asarray(selected, dtype=np.int64))
        if groups:
            if len(groups) < 2:
                raise ValueError("each represented fold requires two complete cohorts")
            complete_rows.append((fold, groups))

    cohorts: list[Cohort] = []
    for fold, groups in complete_rows:
        for index, rows in enumerate(groups):
            reference = groups[(index + 1) % len(groups)]
            ordered_ids = "\0".join(str(value) for value in id_array[rows]).encode("utf-8")
            cohorts.append(
                Cohort(
                    fold=fold,
                    index=index,
                    row_indices=rows,
                    reference_row_indices=reference,
                    ordered_sha256=hashlib.sha256(ordered_ids).hexdigest(),
                )
            )
    return tuple(cohorts)


def _validate_geometry_inputs(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.int64],
) -> tuple[NDArray[np.float64], NDArray[np.int64]]:
    z = normalize_rows(embeddings)
    label_array = _validate_labels(labels)
    if z.shape[0] != label_array.shape[0]:
        raise ValueError("embeddings and labels must align")
    return z, label_array


def global_tangent(
    embeddings: NDArray[np.floating], labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return the stopped-peer all-foreign mean tangent for every anchor."""

    z, label_array = _validate_geometry_inputs(embeddings, labels)
    result = np.empty_like(z)
    for index in range(z.shape[0]):
        foreign = z[label_array != label_array[index]]
        if foreign.shape[0] == 0:
            raise ValueError("every anchor requires a foreign label")
        direction = foreign.mean(axis=0)
        result[index] = direction - z[index] * float(z[index] @ direction)
    return result


def local_excess_tangent(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.int64],
    example_ids: NDArray[np.str_],
    *,
    k: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return top-k-minus-global density tangents and scalar densities."""

    z, label_array = _validate_geometry_inputs(embeddings, labels)
    id_array = np.asarray(example_ids)
    if id_array.ndim != 1 or id_array.shape[0] != z.shape[0] or id_array.dtype.kind != "U":
        raise ValueError("example IDs must be aligned Unicode values")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive builtin int")
    tangent = np.empty_like(z)
    density = np.empty(z.shape[0], dtype=np.float64)
    for index in range(z.shape[0]):
        foreign_indices = np.flatnonzero(label_array != label_array[index]).tolist()
        if len(foreign_indices) < k:
            raise ValueError("anchor has fewer than k foreign peers")
        similarities = z[foreign_indices] @ z[index]
        ranked = sorted(
            range(len(foreign_indices)),
            key=lambda position: (
                -float(similarities[position]),
                str(id_array[foreign_indices[position]]),
            ),
        )
        top_positions = ranked[:k]
        foreign = z[foreign_indices]
        top = foreign[top_positions]
        global_mean = foreign.mean(axis=0)
        local_mean = top.mean(axis=0)
        direction = local_mean - global_mean
        tangent[index] = direction - z[index] * float(z[index] @ direction)
        density[index] = float((top @ z[index]).mean() - similarities.mean())
    return tangent, density
