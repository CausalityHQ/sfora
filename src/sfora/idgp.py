"""Deterministic geometry for Local-Excess Iso-Density Gradient Projection."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sfora.training import ProjectionTrainingConfig, _proxy_anchor_gradient

FOLD_DOMAIN = b"LE-IDGP-fold-v1:"
ROW_DOMAIN = b"LE-IDGP-row-v1:"
POOL_DOMAIN = b"LE-IDGP-pool-v1:"
COHORT_LABELS = 45
ROWS_PER_LABEL = 4
LOCAL_K = 50
DENSITY_POOL_SIZE = 12_612
CONTROL_SHUFFLE_SEED = 20260812
CONTROL_RANDOM_SEED = 20260814
BOOTSTRAP_SEED = 20260813
BOOTSTRAP_REPLICATES = 10_000
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
    """One deterministic 45-label by four-row PA-surrogate cohort."""

    fold: int
    index: int
    row_indices: NDArray[np.int64]
    ordered_sha256: str


@dataclass(frozen=True)
class DensityPools:
    """Deterministically ordered primary and alternate density pools."""

    primary_row_indices: NDArray[np.int64]
    alternate_row_indices: NDArray[np.int64]
    primary_sha256: str
    alternate_sha256: str


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
    bottom_mask: NDArray[np.bool_]
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


@dataclass(frozen=True)
class BootstrapSummary:
    """One label-cluster bootstrap distribution and its frozen summaries."""

    mean: float
    standard_error: float
    lower_bound: float
    mde: float
    label_count: int
    median_rows_per_label: float
    replicate_sha256: str
    replicates: NDArray[np.float64]


@dataclass(frozen=True)
class IDGPEvaluation:
    """Full core evaluation before JSON report conversion."""

    evaluations: tuple[CohortEvaluation, ...]
    density_pool_rows: int
    density_pool_sha256: str
    alternate_pool_rows: int
    alternate_pool_sha256: str
    summary: Mapping[str, object]
    predicates: tuple[dict[str, object], ...]
    passed: bool
    status: str


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
            complete_rows.append((fold, groups))

    cohorts: list[Cohort] = []
    for fold, groups in complete_rows:
        for index, rows in enumerate(groups):
            ordered_ids = "\0".join(str(value) for value in id_array[rows]).encode("utf-8")
            cohorts.append(
                Cohort(
                    fold=fold,
                    index=index,
                    row_indices=rows,
                    ordered_sha256=hashlib.sha256(ordered_ids).hexdigest(),
                )
            )
    return tuple(cohorts)


def build_density_pools(
    labels: NDArray[np.int64],
    example_ids: NDArray[np.str_],
    *,
    pool_size: int = DENSITY_POOL_SIZE,
) -> DensityPools:
    """Split eligible rows into one fixed primary pool and an alternate remainder."""

    label_array = _validate_labels(labels)
    id_array = np.asarray(example_ids)
    if id_array.shape != label_array.shape or id_array.dtype.kind != "U":
        raise ValueError("density-pool IDs must be aligned Unicode values")
    if np.unique(id_array).size != id_array.size or any(not value for value in id_array.tolist()):
        raise ValueError("density-pool IDs must be unique and nonempty")
    if type(pool_size) is not int or pool_size < LOCAL_K or pool_size >= label_array.size:
        raise ValueError("density pool size must leave a nonempty alternate pool")
    ordered = sorted(
        range(label_array.size),
        key=lambda row: (
            hashlib.sha256(POOL_DOMAIN + str(id_array[row]).encode("utf-8")).digest(),
            str(id_array[row]),
        ),
    )
    primary = np.asarray(ordered[:pool_size], dtype=np.int64)
    alternate = np.asarray(ordered[pool_size:], dtype=np.int64)

    def digest(rows: NDArray[np.int64]) -> str:
        data = "\0".join(str(value) for value in id_array[rows]).encode("utf-8")
        return hashlib.sha256(data).hexdigest()

    return DensityPools(
        primary_row_indices=primary,
        alternate_row_indices=alternate,
        primary_sha256=digest(primary),
        alternate_sha256=digest(alternate),
    )


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
    anchors: NDArray[np.floating],
    anchor_labels: NDArray[np.int64],
    pool: NDArray[np.floating],
    pool_labels: NDArray[np.int64],
) -> NDArray[np.float64]:
    """Return the stopped-peer all-foreign mean tangent for every anchor."""

    z, label_array = _validate_geometry_inputs(anchors, anchor_labels)
    peer_z, peer_label_array = _validate_geometry_inputs(pool, pool_labels)
    if peer_z.shape[1] != z.shape[1]:
        raise ValueError("density pool dimension differs from anchors")
    result = np.empty_like(z)
    total = peer_z.sum(axis=0, dtype=np.float64)
    label_sums = {
        int(label): peer_z[peer_label_array == label].sum(axis=0, dtype=np.float64)
        for label in np.unique(label_array)
    }
    label_counts = {
        int(label): int(np.count_nonzero(peer_label_array == label))
        for label in np.unique(label_array)
    }
    for index in range(z.shape[0]):
        label = int(label_array[index])
        same_count = label_counts.get(label, 0)
        foreign_count = peer_z.shape[0] - same_count
        if foreign_count == 0:
            raise ValueError("every anchor requires a foreign label")
        direction = (total - label_sums.get(label, 0.0)) / foreign_count
        result[index] = direction - z[index] * float(z[index] @ direction)
    return result


def local_excess_tangent(
    anchors: NDArray[np.floating],
    anchor_labels: NDArray[np.int64],
    pool: NDArray[np.floating],
    pool_labels: NDArray[np.int64],
    pool_ids: NDArray[np.str_],
    *,
    k: int = 50,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return top-k-minus-global density tangents and scalar densities."""

    z, label_array = _validate_geometry_inputs(anchors, anchor_labels)
    peer_z, peer_label_array = _validate_geometry_inputs(pool, pool_labels)
    if peer_z.shape[1] != z.shape[1]:
        raise ValueError("density pool dimension differs from anchors")
    id_array = np.asarray(pool_ids)
    if id_array.ndim != 1 or id_array.shape[0] != peer_z.shape[0] or id_array.dtype.kind != "U":
        raise ValueError("pool IDs must be aligned Unicode values")
    if type(k) is not int or k < 1:
        raise ValueError("k must be a positive builtin int")
    tangent = np.empty_like(z)
    density = np.empty(z.shape[0], dtype=np.float64)
    similarity_matrix = z @ peer_z.T
    for index in range(z.shape[0]):
        foreign_indices = np.flatnonzero(peer_label_array != label_array[index]).tolist()
        if len(foreign_indices) < k:
            raise ValueError("anchor has fewer than k foreign peers")
        similarities = similarity_matrix[index, foreign_indices]
        top_positions = np.lexsort(
            (id_array[foreign_indices], -similarities)
        )[:k]
        foreign = peer_z[foreign_indices]
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
    density_embeddings: NDArray[np.floating],
    density_labels: NDArray[np.int64],
    density_ids: NDArray[np.str_],
    alternate_embeddings: NDArray[np.floating],
    alternate_labels: NDArray[np.int64],
    alternate_ids: NDArray[np.str_],
) -> CohortEvaluation:
    """Evaluate all single-anchor arms and non-gating system diagnostics."""

    z, label_array = _validate_geometry_inputs(embeddings, labels)
    id_array = np.asarray(example_ids)
    if z.shape[0] != COHORT_LABELS * ROWS_PER_LABEL:
        raise ValueError("primary cohort must contain exactly 180 rows")
    if id_array.shape != (z.shape[0],) or id_array.dtype.kind != "U":
        raise ValueError("primary example IDs must be aligned Unicode values")
    density_z, density_label_array = _validate_geometry_inputs(
        density_embeddings, density_labels
    )
    density_id_array = np.asarray(density_ids)
    alternate_z, alternate_label_array = _validate_geometry_inputs(
        alternate_embeddings, alternate_labels
    )
    alternate_id_array = np.asarray(alternate_ids)
    if density_z.shape[1] != z.shape[1] or alternate_z.shape[1] != z.shape[1]:
        raise ValueError("density pool dimensions must match primary cohort")
    if density_id_array.shape != (density_z.shape[0],) or density_id_array.dtype.kind != "U":
        raise ValueError("density IDs must be aligned Unicode values")
    if alternate_id_array.shape != (alternate_z.shape[0],) or alternate_id_array.dtype.kind != "U":
        raise ValueError("alternate IDs must be aligned Unicode values")

    gradients = proxy_anchor_surrogate_tangent(z, label_array)
    local, _ = local_excess_tangent(
        z, label_array, density_z, density_label_array, density_id_array, k=LOCAL_K
    )
    global_value = global_tangent(z, label_array, density_z, density_label_array)
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

    reference_local, _ = local_excess_tangent(
        z,
        label_array,
        alternate_z,
        alternate_label_array,
        alternate_id_array,
        k=LOCAL_K,
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
        bottom_mask=bottom,
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


def cluster_bootstrap(
    values: NDArray[np.floating],
    labels: NDArray[np.int64],
    *,
    seed: int,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> BootstrapSummary:
    """Bootstrap a mean by resampling complete identity clusters."""

    value_array = np.asarray(values, dtype=np.float64)
    label_array = _validate_labels(labels)
    if value_array.ndim != 1 or value_array.shape != label_array.shape:
        raise ValueError("bootstrap values and labels must be aligned vectors")
    if value_array.size == 0 or not np.isfinite(value_array).all():
        raise ValueError("bootstrap values must be nonempty and finite")
    if type(seed) is not int or type(replicates) is not int or replicates < 2:
        raise ValueError("bootstrap seed and replicate count must be builtin ints")
    unique = np.asarray(sorted(np.unique(label_array).tolist()), dtype=np.int64)
    grouped = [value_array[label_array == label] for label in unique]
    rng = np.random.default_rng(seed)
    distribution = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        selected = rng.integers(0, unique.size, size=unique.size)
        distribution[replicate] = float(
            np.concatenate([grouped[position] for position in selected]).mean()
        )
    little_endian = distribution.astype("<f8", copy=False)
    row_counts = np.asarray([group.size for group in grouped], dtype=np.float64)
    standard_error = float(distribution.std(ddof=1))
    return BootstrapSummary(
        mean=float(value_array.mean()),
        standard_error=standard_error,
        lower_bound=float(np.percentile(distribution, 1.0, method="linear")),
        mde=2.576 * standard_error,
        label_count=int(unique.size),
        median_rows_per_label=float(np.median(row_counts)),
        replicate_sha256=hashlib.sha256(little_endian.tobytes(order="C")).hexdigest(),
        replicates=distribution,
    )


def _bootstrap_record(summary: BootstrapSummary) -> dict[str, object]:
    return {
        "mean": summary.mean,
        "standard_error": summary.standard_error,
        "lower_bound": summary.lower_bound,
        "mde": summary.mde,
        "label_count": summary.label_count,
        "median_rows_per_label": summary.median_rows_per_label,
        "replicate_sha256": summary.replicate_sha256,
    }


def summarize_evaluations(
    evaluations: Sequence[CohortEvaluation],
    *,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> dict[str, object]:
    """Aggregate primary-row effects and all paired control contrasts."""

    if not evaluations:
        raise ValueError("at least one cohort evaluation is required")
    ordered = sorted(evaluations, key=lambda value: (value.fold, value.index))
    primary_labels: list[NDArray[np.int64]] = []
    raw_values: list[NDArray[np.float64]] = []
    shuffled_values: list[NDArray[np.float64]] = []
    random_values: list[NDArray[np.float64]] = []
    global_values: list[NDArray[np.float64]] = []
    positive_values: list[NDArray[np.float64]] = []
    two_sided_values: list[NDArray[np.float64]] = []
    zero_changes: list[NDArray[np.float64]] = []
    fold_values: dict[int, list[NDArray[np.float64]]] = {}
    for evaluation in ordered:
        specificity = evaluation.bottom_mask & ~evaluation.skipped_mask
        if np.any(specificity):
            two_sided_values.append(
                evaluation.margin_changes["le_idgp"][specificity]
                - evaluation.margin_changes["two_sided"][specificity]
            )
        primary = evaluation.primary_mask
        if not np.any(primary):
            continue
        le = evaluation.margin_changes["le_idgp"][primary]
        zero = evaluation.margin_changes["zero"][primary]
        raw = le - zero
        labels = evaluation.labels[primary]
        primary_labels.append(labels)
        raw_values.append(raw)
        shuffled_values.append(le - evaluation.margin_changes["shuffled_local"][primary])
        random_values.append(le - evaluation.margin_changes["random_tangent"][primary])
        global_values.append(le - evaluation.margin_changes["global_centering"][primary])
        positive_values.append(
            evaluation.positive_similarity_changes["le_idgp"][primary]
            - evaluation.positive_similarity_changes["zero"][primary]
        )
        zero_changes.append(zero)
        fold_values.setdefault(evaluation.fold, []).append(raw)
    if not primary_labels:
        eligible_rows = sum(evaluation.labels.size for evaluation in ordered)
        conflicting_rows = sum(
            int(np.count_nonzero((evaluation.conflict_dots < 0.0) & ~evaluation.skipped_mask))
            for evaluation in ordered
        )
        empty_hash = hashlib.sha256(b"").hexdigest()
        empty_bootstrap = {
            "mean": 0.0,
            "standard_error": 0.0,
            "lower_bound": 0.0,
            "mde": 0.0,
            "label_count": 0,
            "median_rows_per_label": 0.0,
            "replicate_sha256": empty_hash,
        }
        return {
            "coverage": {
                "eligible_rows": eligible_rows,
                "conflicting_rows": conflicting_rows,
                "conflict_fraction": conflicting_rows / eligible_rows,
                "primary_rows": 0,
                "primary_labels": 0,
                "folds_with_primary": 0,
            },
            "fold_mean_advantages": [],
            "pooled": {
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
            "bootstrap": {
                name: dict(empty_bootstrap)
                for name in ("raw", "shuffled", "random", "global", "positive")
            },
        }

    labels = np.concatenate(primary_labels)
    contrasts = {
        "raw": np.concatenate(raw_values),
        "shuffled": np.concatenate(shuffled_values),
        "random": np.concatenate(random_values),
        "global": np.concatenate(global_values),
        "positive": np.concatenate(positive_values),
    }
    bootstraps = {
        name: cluster_bootstrap(
            values,
            labels,
            seed=BOOTSTRAP_SEED + offset,
            replicates=bootstrap_replicates,
        )
        for offset, (name, values) in enumerate(contrasts.items())
    }
    raw = contrasts["raw"]
    positive = contrasts["positive"]
    two_sided = np.concatenate(two_sided_values) if two_sided_values else np.asarray([0.0])
    zero = np.concatenate(zero_changes)
    eligible_rows = sum(evaluation.labels.size for evaluation in ordered)
    conflicting_rows = sum(
        int(np.count_nonzero((evaluation.conflict_dots < 0.0) & ~evaluation.skipped_mask))
        for evaluation in ordered
    )
    folds_with_primary = len(fold_values)
    fold_means = [float(np.concatenate(fold_values[fold]).mean()) for fold in sorted(fold_values)]
    return {
        "coverage": {
            "eligible_rows": eligible_rows,
            "conflicting_rows": conflicting_rows,
            "conflict_fraction": conflicting_rows / eligible_rows,
            "primary_rows": int(labels.size),
            "primary_labels": int(np.unique(labels).size),
            "folds_with_primary": folds_with_primary,
        },
        "fold_mean_advantages": fold_means,
        "pooled": {
            "raw_advantage_mean": float(raw.mean()),
            "raw_advantage_lower": bootstraps["raw"].lower_bound,
            "shuffled_contrast_lower": bootstraps["shuffled"].lower_bound,
            "random_contrast_lower": bootstraps["random"].lower_bound,
            "global_contrast_lower": bootstraps["global"].lower_bound,
            "zero_abs_margin_change_median": float(np.median(np.abs(zero))),
            "positive_similarity_mean": float(positive.mean()),
            "positive_similarity_lower": bootstraps["positive"].lower_bound,
            "le_minus_two_sided_mean": float(two_sided.mean()),
        },
        "bootstrap": {name: _bootstrap_record(value) for name, value in bootstraps.items()},
    }


def decide_idgp(
    summary: Mapping[str, object],
) -> tuple[bool, tuple[dict[str, object], ...]]:
    """Apply the seven frozen LE-IDGP predicates in exact order."""

    coverage = summary["coverage"]
    pooled = summary["pooled"]
    fold_means = summary["fold_mean_advantages"]
    if (
        not isinstance(coverage, Mapping)
        or not isinstance(pooled, Mapping)
        or not isinstance(fold_means, Sequence)
    ):
        raise ValueError("summary structure differs")
    predicates = (
        {
            "name": "coverage",
            "passed": coverage["conflict_fraction"] >= 0.1
            and coverage["primary_labels"] >= 100
            and coverage["folds_with_primary"] >= 3,
        },
        {
            "name": "raw_advantage",
            "passed": pooled["raw_advantage_mean"] > 0.0 and pooled["raw_advantage_lower"] > 0.0,
        },
        {
            "name": "fold_consistency",
            "passed": sum(value > 0.0 for value in fold_means) >= 3,
        },
        {
            "name": "control_superiority",
            "passed": pooled["shuffled_contrast_lower"] > 0.0
            and pooled["random_contrast_lower"] > 0.0
            and pooled["global_contrast_lower"] > 0.0,
        },
        {
            "name": "material_effect",
            "passed": pooled["raw_advantage_mean"]
            >= 0.05 * pooled["zero_abs_margin_change_median"],
        },
        {
            "name": "positive_similarity",
            "passed": pooled["positive_similarity_mean"] >= -1e-4
            and pooled["positive_similarity_lower"] >= -5e-4,
        },
        {
            "name": "one_sided_specificity",
            "passed": pooled["le_minus_two_sided_mean"] >= 0.0,
        },
    )
    if any(type(predicate["passed"]) is not bool for predicate in predicates):
        raise ValueError("predicate values must be builtin booleans")
    return all(predicate["passed"] is True for predicate in predicates), predicates


def evaluate_idgp(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.int64],
    example_ids: NDArray[np.str_],
    *,
    density_pool_size: int = DENSITY_POOL_SIZE,
    bootstrap_replicates: int = BOOTSTRAP_REPLICATES,
) -> IDGPEvaluation:
    """Evaluate every complete deterministic cohort and apply the frozen gate."""

    z, label_array = _validate_geometry_inputs(embeddings, labels)
    id_array = np.asarray(example_ids)
    if id_array.shape != (z.shape[0],) or id_array.dtype.kind != "U":
        raise ValueError("example IDs must be aligned Unicode values")
    unique, counts = np.unique(label_array, return_counts=True)
    eligible_labels = unique[counts >= ROWS_PER_LABEL]
    eligible = np.isin(label_array, eligible_labels)
    eligible_z = z[eligible]
    eligible_label_array = label_array[eligible]
    eligible_ids = id_array[eligible]
    cohorts = build_cohorts(eligible_label_array, eligible_ids)
    if not cohorts:
        raise ValueError("input has no complete LE-IDGP cohorts")
    pools = build_density_pools(
        eligible_label_array,
        eligible_ids,
        pool_size=density_pool_size,
    )
    density_rows = pools.primary_row_indices
    alternate_rows = pools.alternate_row_indices
    evaluations: list[CohortEvaluation] = []
    for cohort in cohorts:
        evaluations.append(
            evaluate_cohort(
                eligible_z[cohort.row_indices],
                eligible_label_array[cohort.row_indices],
                eligible_ids[cohort.row_indices],
                fold=cohort.fold,
                index=cohort.index,
                density_embeddings=eligible_z[density_rows],
                density_labels=eligible_label_array[density_rows],
                density_ids=eligible_ids[density_rows],
                alternate_embeddings=eligible_z[alternate_rows],
                alternate_labels=eligible_label_array[alternate_rows],
                alternate_ids=eligible_ids[alternate_rows],
            )
        )
    summary = summarize_evaluations(evaluations, bootstrap_replicates=bootstrap_replicates)
    passed, predicates = decide_idgp(summary)
    return IDGPEvaluation(
        evaluations=tuple(evaluations),
        density_pool_rows=int(density_rows.size),
        density_pool_sha256=pools.primary_sha256,
        alternate_pool_rows=int(alternate_rows.size),
        alternate_pool_sha256=pools.alternate_sha256,
        summary=summary,
        predicates=predicates,
        passed=passed,
        status="PASS" if passed else "KILL",
    )
