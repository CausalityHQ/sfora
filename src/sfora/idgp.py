"""Deterministic geometry for Local-Excess Iso-Density Gradient Projection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sfora.training import ProjectionTrainingConfig, _proxy_anchor_gradient

FOLD_DOMAIN = b"LE-IDGP-fold-v1:"
ROW_DOMAIN = b"LE-IDGP-row-v1:"
COHORT_LABELS = 45
ROWS_PER_LABEL = 4
LOCAL_K = 50
CONTROL_SHUFFLE_SEED = 20260812
CONTROL_RANDOM_SEED = 20260814
ARM_ORDER = (
    "zero",
    "le_idgp",
    "shuffled_local",
    "random_tangent",
    "global_centering",
    "two_sided",
)


@dataclass(frozen=True)
class Cohort:
    """One deterministic 45-label by four-row cohort and its disjoint reference."""

    fold: int
    index: int
    row_indices: NDArray[np.int64]
    reference_row_indices: NDArray[np.int64]
    ordered_sha256: str


@dataclass(frozen=True)
class CohortEvaluation:
    """Complete deterministic virtual-step evidence for one cohort."""

    fold: int
    index: int
    example_ids: NDArray[np.str_]
    labels: NDArray[np.int64]
    pre_margins: NDArray[np.float64]
    conflict_dots: NDArray[np.float64]
    skipped_mask: NDArray[np.bool_]
    primary_mask: NDArray[np.bool_]
    margin_changes: Mapping[str, NDArray[np.float64]]
    positive_similarity_changes: Mapping[str, NDArray[np.float64]]
    analytic_density_reduction: NDArray[np.float64]
    local_norms: NDArray[np.float64]
    global_norms: NDArray[np.float64]
    local_global_cosine_mean: float
    collective_idgp_advantage: float
    reference_conflict_fraction: float
    reference_cosine_mean: float


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


def _aligned_float64_pair(
    first: NDArray[np.floating], second: NDArray[np.floating]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    left = np.asarray(first, dtype=np.float64)
    right = np.asarray(second, dtype=np.float64)
    if left.ndim != 2 or left.shape != right.shape:
        raise ValueError("vector arrays must be aligned and two-dimensional")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError("vector arrays must be finite")
    return left, right


def proxy_anchor_surrogate_tangent(
    embeddings: NDArray[np.floating], labels: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Return the repository's centroid-proxy surrogate gradient on the sphere."""

    z, label_array = _validate_geometry_inputs(embeddings, labels)
    config = ProjectionTrainingConfig(objective="proxy_anchor")
    gradient = np.asarray(_proxy_anchor_gradient(z, label_array, config), dtype=np.float64)
    return gradient - z * np.sum(gradient * z, axis=1, keepdims=True)


def project_one_sided(
    gradients: NDArray[np.floating],
    nuisance_tangents: NDArray[np.floating],
    *,
    cutoff: float = 1e-8,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.bool_]]:
    """Project conflicting gradients onto a nuisance half-space boundary."""

    g, h = _aligned_float64_pair(gradients, nuisance_tangents)
    if type(cutoff) is not float or not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be a positive builtin float")
    norms = np.linalg.norm(h, axis=1)
    skipped = norms < cutoff
    dots = np.sum(g * h, axis=1)
    conflicting = (dots < 0.0) & ~skipped
    projected = g.copy()
    projected[conflicting] -= (dots[conflicting] / np.square(norms[conflicting]))[:, None] * h[
        conflicting
    ]
    return projected, dots, skipped


def project_two_sided(
    gradients: NDArray[np.floating],
    nuisance_tangents: NDArray[np.floating],
    *,
    cutoff: float = 1e-8,
) -> NDArray[np.float64]:
    """Remove the nuisance component regardless of its sign."""

    g, h = _aligned_float64_pair(gradients, nuisance_tangents)
    if type(cutoff) is not float or not np.isfinite(cutoff) or cutoff <= 0.0:
        raise ValueError("cutoff must be a positive builtin float")
    norms = np.linalg.norm(h, axis=1)
    active = norms >= cutoff
    projected = g.copy()
    dots = np.sum(g * h, axis=1)
    projected[active] -= (dots[active] / np.square(norms[active]))[:, None] * h[active]
    return projected


def geodesic_step(
    embeddings: NDArray[np.floating],
    directions: NDArray[np.floating],
    *,
    epsilon: float = 0.01,
) -> NDArray[np.float64]:
    """Move each unit row by an equal sphere-geodesic arc in descent direction."""

    z = normalize_rows(embeddings)
    direction = np.asarray(directions, dtype=np.float64)
    if direction.shape != z.shape or not np.isfinite(direction).all():
        raise ValueError("directions must be finite and align with embeddings")
    if type(epsilon) is not float or not np.isfinite(epsilon) or epsilon <= 0.0:
        raise ValueError("epsilon must be a positive builtin float")
    tangent = direction - z * np.sum(direction * z, axis=1, keepdims=True)
    norms = np.linalg.norm(tangent, axis=1)
    active = norms >= 1e-8
    result = z.copy()
    units = tangent[active] / norms[active, None]
    result[active] = np.cos(epsilon) * z[active] - np.sin(epsilon) * units
    return result


def retrieval_geometry(
    anchor: NDArray[np.floating],
    peers: NDArray[np.floating],
    peer_labels: NDArray[np.int64],
    anchor_label: int,
) -> tuple[float, float]:
    """Return margin and nearest-positive cosine for one anchor and frozen peers."""

    anchor_array = np.asarray(anchor, dtype=np.float64)
    if anchor_array.ndim != 1 or not np.isfinite(anchor_array).all():
        raise ValueError("anchor must be a finite vector")
    normalized_anchor = normalize_rows(anchor_array[None, :])[0]
    normalized_peers = normalize_rows(peers)
    labels = _validate_labels(peer_labels)
    if normalized_peers.shape[0] != labels.shape[0]:
        raise ValueError("peers and labels must align")
    positive = labels == anchor_label
    foreign = ~positive
    if not np.any(positive) or not np.any(foreign):
        raise ValueError("anchor requires positive and foreign peers")
    similarities = normalized_peers @ normalized_anchor
    nearest_positive = float(np.max(similarities[positive]))
    nearest_foreign = float(np.max(similarities[foreign]))
    return nearest_positive - nearest_foreign, nearest_positive


def _cross_local_excess_tangent(
    anchors: NDArray[np.float64],
    anchor_labels: NDArray[np.int64],
    peers: NDArray[np.float64],
    peer_labels: NDArray[np.int64],
    peer_ids: NDArray[np.str_],
    *,
    k: int = LOCAL_K,
) -> NDArray[np.float64]:
    result = np.empty_like(anchors)
    for index in range(anchors.shape[0]):
        foreign_indices = np.flatnonzero(peer_labels != anchor_labels[index]).tolist()
        if len(foreign_indices) < k:
            raise ValueError("reference cohort has fewer than k foreign peers")
        similarities = peers[foreign_indices] @ anchors[index]
        ranked = sorted(
            range(len(foreign_indices)),
            key=lambda position: (
                -float(similarities[position]),
                str(peer_ids[foreign_indices[position]]),
            ),
        )
        foreign = peers[foreign_indices]
        direction = foreign[ranked[:k]].mean(axis=0) - foreign.mean(axis=0)
        result[index] = direction - anchors[index] * float(anchors[index] @ direction)
    return result


def _all_retrieval_geometry(
    embeddings: NDArray[np.float64], labels: NDArray[np.int64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    margins = np.empty(embeddings.shape[0], dtype=np.float64)
    positives = np.empty(embeddings.shape[0], dtype=np.float64)
    for index in range(embeddings.shape[0]):
        keep = np.arange(embeddings.shape[0]) != index
        margins[index], positives[index] = retrieval_geometry(
            embeddings[index], embeddings[keep], labels[keep], int(labels[index])
        )
    return margins, positives


def _cosine_mean(first: NDArray[np.float64], second: NDArray[np.float64]) -> float:
    first_norm = np.linalg.norm(first, axis=1)
    second_norm = np.linalg.norm(second, axis=1)
    active = (first_norm >= 1e-8) & (second_norm >= 1e-8)
    if not np.any(active):
        return 0.0
    values = np.sum(first[active] * second[active], axis=1) / (
        first_norm[active] * second_norm[active]
    )
    return float(values.mean())


def evaluate_cohort(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.int64],
    example_ids: NDArray[np.str_],
    *,
    fold: int,
    index: int,
    reference_embeddings: NDArray[np.floating],
    reference_labels: NDArray[np.int64],
    reference_ids: NDArray[np.str_],
) -> CohortEvaluation:
    """Evaluate all single-anchor arms and non-gating system diagnostics."""

    z, label_array = _validate_geometry_inputs(embeddings, labels)
    id_array = np.asarray(example_ids)
    if z.shape[0] != COHORT_LABELS * ROWS_PER_LABEL:
        raise ValueError("primary cohort must contain exactly 180 rows")
    if id_array.shape != (z.shape[0],) or id_array.dtype.kind != "U":
        raise ValueError("primary example IDs must be aligned Unicode values")
    reference_z, reference_label_array = _validate_geometry_inputs(
        reference_embeddings, reference_labels
    )
    reference_id_array = np.asarray(reference_ids)
    if reference_z.shape[0] != z.shape[0] or reference_id_array.shape != (z.shape[0],):
        raise ValueError("reference cohort must align with primary cohort size")
    if reference_id_array.dtype.kind != "U":
        raise ValueError("reference IDs must be Unicode values")

    gradients = proxy_anchor_surrogate_tangent(z, label_array)
    local, _ = local_excess_tangent(z, label_array, id_array, k=LOCAL_K)
    global_value = global_tangent(z, label_array)
    le_idgp, dots, skipped = project_one_sided(gradients, local)
    rng_shuffle = np.random.default_rng(CONTROL_SHUFFLE_SEED + 10_000 * fold + index)
    shuffled = local[rng_shuffle.permutation(z.shape[0])]
    shuffled -= z * np.sum(shuffled * z, axis=1, keepdims=True)
    shuffled_projected, _, _ = project_one_sided(gradients, shuffled)
    rng_random = np.random.default_rng(CONTROL_RANDOM_SEED + 10_000 * fold + index)
    random_value = rng_random.normal(size=z.shape)
    random_value -= z * np.sum(random_value * z, axis=1, keepdims=True)
    random_projected, _, _ = project_one_sided(gradients, random_value)
    global_projected, _, _ = project_one_sided(gradients, global_value)
    two_sided = project_two_sided(gradients, local)
    directions = {
        "zero": gradients,
        "le_idgp": le_idgp,
        "shuffled_local": shuffled_projected,
        "random_tangent": random_projected,
        "global_centering": global_projected,
        "two_sided": two_sided,
    }
    pre_margins, pre_positives = _all_retrieval_geometry(z, label_array)
    margin_changes: dict[str, NDArray[np.float64]] = {}
    positive_changes: dict[str, NDArray[np.float64]] = {}
    for arm in ARM_ORDER:
        arm_margins = np.empty(z.shape[0], dtype=np.float64)
        arm_positives = np.empty(z.shape[0], dtype=np.float64)
        for row in range(z.shape[0]):
            moved = geodesic_step(z[row : row + 1], directions[arm][row : row + 1])[0]
            keep = np.arange(z.shape[0]) != row
            arm_margins[row], arm_positives[row] = retrieval_geometry(
                moved, z[keep], label_array[keep], int(label_array[row])
            )
        margin_changes[arm] = arm_margins - pre_margins
        positive_changes[arm] = arm_positives - pre_positives

    bottom = np.zeros(z.shape[0], dtype=np.bool_)
    ordered = sorted(range(z.shape[0]), key=lambda row: (pre_margins[row], str(id_array[row])))
    bottom[ordered[: z.shape[0] // 4]] = True
    primary = bottom & (dots < 0.0) & ~skipped
    gradient_norms = np.linalg.norm(gradients, axis=1)
    analytic = np.zeros(z.shape[0], dtype=np.float64)
    active_gradient = gradient_norms >= 1e-8
    analytic[active_gradient & (dots < 0.0)] = (
        0.01
        * np.abs(dots[active_gradient & (dots < 0.0)])
        / gradient_norms[active_gradient & (dots < 0.0)]
    )

    moved_zero = geodesic_step(z, gradients)
    moved_idgp = geodesic_step(z, le_idgp)
    zero_collective, _ = _all_retrieval_geometry(moved_zero, label_array)
    idgp_collective, _ = _all_retrieval_geometry(moved_idgp, label_array)
    collective_advantage = float(np.mean(idgp_collective - zero_collective))

    reference_local = _cross_local_excess_tangent(
        z,
        label_array,
        reference_z,
        reference_label_array,
        reference_id_array,
    )
    reference_dots = np.sum(gradients * reference_local, axis=1)
    reference_norms = np.linalg.norm(reference_local, axis=1)
    reference_active = reference_norms >= 1e-8
    reference_conflict_fraction = float(np.mean((reference_dots < 0.0) & reference_active))

    return CohortEvaluation(
        fold=fold,
        index=index,
        example_ids=id_array.copy(),
        labels=label_array.copy(),
        pre_margins=pre_margins,
        conflict_dots=dots,
        skipped_mask=skipped,
        primary_mask=primary,
        margin_changes=margin_changes,
        positive_similarity_changes=positive_changes,
        analytic_density_reduction=analytic,
        local_norms=np.linalg.norm(local, axis=1),
        global_norms=np.linalg.norm(global_value, axis=1),
        local_global_cosine_mean=_cosine_mean(local, global_value),
        collective_idgp_advantage=collective_advantage,
        reference_conflict_fraction=reference_conflict_fraction,
        reference_cosine_mean=_cosine_mean(local, reference_local),
    )
