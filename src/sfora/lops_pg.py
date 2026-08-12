"""Deterministic geometry for Leave-One-Out Positive-Safe Proxy Gradient."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sfora.idgp import cluster_bootstrap, geodesic_step, proxy_anchor_surrogate_tangent

ARM_ORDER = (
    "proxy_anchor",
    "lops_pg",
    "shuffled_centroid",
    "positive_only",
    "nearest_positive_safe",
    "batch_hard_triplet",
)


@dataclass(frozen=True)
class CohortEvaluation:
    fold: int
    index: int
    example_ids: NDArray[np.str_]
    labels: NDArray[np.int64]
    pre_margins: NDArray[np.float64]
    bottom_mask: NDArray[np.bool_]
    conflict_mask: NDArray[np.bool_]
    skipped_mask: NDArray[np.bool_]
    primary_mask: NDArray[np.bool_]
    constraint_dots: NDArray[np.float64]
    positive_tangents: NDArray[np.float64]
    directions: Mapping[str, NDArray[np.float64]]
    margin_changes: Mapping[str, NDArray[np.float64]]
    positive_similarity_changes: Mapping[str, NDArray[np.float64]]


def _vector(value: NDArray[np.floating], *, name: str) -> NDArray[np.float64]:
    result = np.asarray(value)
    if result.dtype != np.dtype(np.float64) or result.ndim != 1:
        raise ValueError(f"{name} must be an exact float64 vector")
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _unit(value: NDArray[np.float64], *, name: str) -> NDArray[np.float64]:
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError(f"{name} is degenerate")
    return value / norm


def positive_centroid_tangent(
    anchor: NDArray[np.floating], siblings: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Return the stopped leave-one-out centroid's spherical ascent tangent."""

    z = _unit(_vector(anchor, name="anchor"), name="anchor")
    peers = np.asarray(siblings)
    if peers.dtype != np.dtype(np.float64) or peers.ndim != 2 or peers.shape[1:] != z.shape:
        raise ValueError("siblings must be aligned exact float64 rows")
    if peers.shape[0] < 1 or not np.isfinite(peers).all():
        raise ValueError("siblings must be nonempty and finite")
    centroid = _unit(peers.mean(axis=0, dtype=np.float64), name="sibling centroid")
    return (centroid - z * float(z @ centroid)).copy()


def project_positive_safe(
    gradient: NDArray[np.floating], tangent: NDArray[np.floating]
) -> tuple[NDArray[np.float64], bool]:
    """Project a conflicting gradient onto the positive-cohesion half-space."""

    g = _vector(gradient, name="gradient")
    p = _vector(tangent, name="positive tangent")
    if g.shape != p.shape:
        raise ValueError("gradient and positive tangent must align")
    squared_norm = float(p @ p)
    if squared_norm < 1e-16:
        raise ValueError("positive tangent is degenerate")
    dot = float(g @ p)
    if dot <= 0.0:
        return g.copy(), False
    return (g - (dot / squared_norm) * p).copy(), True


def batch_hard_triplet_tangent(
    anchor: NDArray[np.floating],
    anchor_label: int,
    peers: NDArray[np.floating],
    peer_labels: NDArray[np.int64],
    peer_ids: NDArray[np.str_],
) -> NDArray[np.float64]:
    """Return the tangent gradient of max-negative minus max-positive similarity."""

    z = _unit(_vector(anchor, name="anchor"), name="anchor")
    rows = np.asarray(peers)
    labels = np.asarray(peer_labels)
    ids = np.asarray(peer_ids)
    if rows.dtype != np.dtype(np.float64) or rows.ndim != 2 or rows.shape[1:] != z.shape:
        raise ValueError("peers must be aligned exact float64 rows")
    if labels.dtype != np.dtype(np.int64) or labels.shape != (rows.shape[0],):
        raise ValueError("peer labels must be aligned exact int64 values")
    if ids.dtype.kind != "U" or ids.shape != labels.shape:
        raise ValueError("peer IDs must be aligned Unicode values")
    if not np.isfinite(rows).all():
        raise ValueError("peers must be finite")
    similarities = rows @ z
    positive = np.flatnonzero(labels == anchor_label).tolist()
    negative = np.flatnonzero(labels != anchor_label).tolist()
    if not positive or not negative:
        raise ValueError("triplet tangent requires positive and negative peers")
    positive.sort(key=lambda index: (-similarities[index], str(ids[index])))
    negative.sort(key=lambda index: (-similarities[index], str(ids[index])))
    gradient = rows[negative[0]] - rows[positive[0]]
    return (gradient - z * float(z @ gradient)).copy()


def _geometry(
    anchor: NDArray[np.float64],
    anchor_label: int,
    peers: NDArray[np.float64],
    peer_labels: NDArray[np.int64],
) -> tuple[float, float]:
    similarities = peers @ anchor
    positive = peer_labels == anchor_label
    negative = ~positive
    if not np.any(positive) or not np.any(negative):
        raise ValueError("each anchor requires positive and negative peers")
    return (
        float(np.max(similarities[positive]) - np.max(similarities[negative])),
        float(np.mean(similarities[positive], dtype=np.float64)),
    )


def _nearest_positive_tangent(
    anchor: NDArray[np.float64],
    peers: NDArray[np.float64],
    labels: NDArray[np.int64],
    ids: NDArray[np.str_],
    anchor_label: int,
) -> NDArray[np.float64]:
    candidates = np.flatnonzero(labels == anchor_label).tolist()
    candidates.sort(key=lambda row: (-float(peers[row] @ anchor), str(ids[row])))
    target = peers[candidates[0]]
    return (target - anchor * float(anchor @ target)).copy()


def evaluate_cohort(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.int64],
    example_ids: NDArray[np.str_],
    *,
    fold: int,
    index: int,
) -> CohortEvaluation:
    """Evaluate LOPS-PG and registered controls with single-anchor virtual steps."""

    z = np.asarray(embeddings)
    label_array = np.asarray(labels)
    ids = np.asarray(example_ids)
    if z.dtype != np.dtype(np.float64) or z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError("embeddings must be finite exact float64 rows")
    norms = np.linalg.norm(z, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("embedding rows must be nondegenerate")
    z = z / norms[:, None]
    if label_array.dtype != np.dtype(np.int64) or label_array.shape != (z.shape[0],):
        raise ValueError("labels must be aligned exact int64 values")
    if ids.dtype.kind != "U" or ids.shape != label_array.shape or np.unique(ids).size != ids.size:
        raise ValueError("example IDs must be aligned unique Unicode values")
    if type(fold) is not int or fold not in (1, 2, 3) or type(index) is not int or index < 0:
        raise ValueError("confirmation cohorts require fold 1, 2, or 3 and nonnegative index")
    gradients = proxy_anchor_surrogate_tangent(z, label_array)
    positive = np.empty_like(z)
    nearest = np.empty_like(z)
    triplet = np.empty_like(z)
    pre_margins = np.empty(z.shape[0], dtype=np.float64)
    pre_positive = np.empty(z.shape[0], dtype=np.float64)
    for row in range(z.shape[0]):
        keep = np.arange(z.shape[0]) != row
        peers = z[keep]
        peer_labels = label_array[keep]
        peer_ids = ids[keep]
        same = peers[peer_labels == label_array[row]]
        positive[row] = positive_centroid_tangent(z[row], same)
        nearest[row] = _nearest_positive_tangent(
            z[row], peers, peer_labels, peer_ids, int(label_array[row])
        )
        triplet[row] = batch_hard_triplet_tangent(
            z[row], int(label_array[row]), peers, peer_labels, peer_ids
        )
        pre_margins[row], pre_positive[row] = _geometry(
            z[row], int(label_array[row]), peers, peer_labels
        )
    positive_norms = np.linalg.norm(positive, axis=1)
    skipped = positive_norms < 1e-8
    dots = np.sum(gradients * positive, axis=1)
    conflict = (dots > 0.0) & ~skipped
    lops = gradients.copy()
    lops[conflict] -= (dots[conflict] / np.square(positive_norms[conflict]))[:, None] * positive[
        conflict
    ]

    rng = np.random.Generator(np.random.PCG64(20260821 + index))
    for _ in range(10_000):
        permutation = rng.permutation(z.shape[0])
        if np.all(label_array[permutation] != label_array):
            break
    else:
        raise ValueError("could not construct shuffled-label derangement")
    shuffled_tangent = positive[permutation].copy()
    shuffled_tangent -= z * np.sum(shuffled_tangent * z, axis=1, keepdims=True)
    shuffled = gradients.copy()
    shuffled_dots = np.sum(shuffled * shuffled_tangent, axis=1)
    shuffled_norms = np.linalg.norm(shuffled_tangent, axis=1)
    shuffled_active = (shuffled_dots > 0.0) & (shuffled_norms >= 1e-8)
    shuffled[shuffled_active] -= (
        shuffled_dots[shuffled_active] / np.square(shuffled_norms[shuffled_active])
    )[:, None] * shuffled_tangent[shuffled_active]
    nearest_safe = gradients.copy()
    nearest_dots = np.sum(nearest_safe * nearest, axis=1)
    nearest_norms = np.linalg.norm(nearest, axis=1)
    nearest_active = (nearest_dots > 0.0) & (nearest_norms >= 1e-8)
    nearest_safe[nearest_active] -= (
        nearest_dots[nearest_active] / np.square(nearest_norms[nearest_active])
    )[:, None] * nearest[nearest_active]
    directions = {
        "proxy_anchor": gradients,
        "lops_pg": lops,
        "shuffled_centroid": shuffled,
        "positive_only": -positive,
        "nearest_positive_safe": nearest_safe,
        "batch_hard_triplet": triplet,
    }
    margin_changes: dict[str, NDArray[np.float64]] = {}
    positive_changes: dict[str, NDArray[np.float64]] = {}
    for arm in ARM_ORDER:
        arm_margin = np.empty(z.shape[0], dtype=np.float64)
        arm_positive = np.empty(z.shape[0], dtype=np.float64)
        for row in range(z.shape[0]):
            moved = geodesic_step(z[row : row + 1], directions[arm][row : row + 1])[0]
            keep = np.arange(z.shape[0]) != row
            arm_margin[row], arm_positive[row] = _geometry(
                moved, int(label_array[row]), z[keep], label_array[keep]
            )
        margin_changes[arm] = arm_margin - pre_margins
        positive_changes[arm] = arm_positive - pre_positive
    ordered = sorted(range(z.shape[0]), key=lambda row: (pre_margins[row], str(ids[row])))
    bottom = np.zeros(z.shape[0], dtype=np.bool_)
    bottom[ordered[: z.shape[0] // 4]] = True
    return CohortEvaluation(
        fold=fold,
        index=index,
        example_ids=ids.copy(),
        labels=label_array.copy(),
        pre_margins=pre_margins,
        bottom_mask=bottom,
        conflict_mask=conflict,
        skipped_mask=skipped,
        primary_mask=bottom & ~skipped,
        constraint_dots=dots,
        positive_tangents=positive.copy(),
        directions={name: value.copy() for name, value in directions.items()},
        margin_changes=margin_changes,
        positive_similarity_changes=positive_changes,
    )


def _bootstrap_record(
    values: NDArray[np.float64], labels: NDArray[np.int64], seed: int, replicates: int
) -> dict[str, object]:
    result = cluster_bootstrap(values, labels, seed=seed, replicates=replicates)
    return {
        "mean": result.mean,
        "lower_bound": result.lower_bound,
        "standard_error": result.standard_error,
        "label_count": result.label_count,
        "median_rows_per_label": result.median_rows_per_label,
        "replicate_sha256": result.replicate_sha256,
        "replicates": replicates,
        "seed": seed,
    }


def summarize_evaluations(
    evaluations: Sequence[CohortEvaluation], *, bootstrap_replicates: int = 10_000
) -> dict[str, object]:
    """Aggregate the frozen paired confirmation populations."""

    ordered = sorted(evaluations, key=lambda value: (value.fold, value.index))
    if not ordered or any(value.fold not in (1, 2, 3) for value in ordered):
        raise ValueError("confirmation requires folds 1, 2, and 3 only")
    labels: list[NDArray[np.int64]] = []
    raw: list[NDArray[np.float64]] = []
    shuffled: list[NDArray[np.float64]] = []
    positive_only: list[NDArray[np.float64]] = []
    positive: list[NDArray[np.float64]] = []
    pa: list[NDArray[np.float64]] = []
    hard: list[NDArray[np.float64]] = []
    nearest: list[NDArray[np.float64]] = []
    fold_raw: dict[int, list[NDArray[np.float64]]] = {}
    constraint_ok = True
    for value in ordered:
        mask = value.primary_mask
        if np.any(mask):
            lops = value.margin_changes["lops_pg"][mask]
            zero = value.margin_changes["proxy_anchor"][mask]
            contrast = lops - zero
            labels.append(value.labels[mask])
            raw.append(contrast)
            shuffled.append(lops - value.margin_changes["shuffled_centroid"][mask])
            positive_only.append(lops - value.margin_changes["positive_only"][mask])
            positive.append(
                value.positive_similarity_changes["lops_pg"][mask]
                - value.positive_similarity_changes["proxy_anchor"][mask]
            )
            pa.append(zero)
            hard.append(value.margin_changes["batch_hard_triplet"][mask] - zero)
            nearest.append(value.margin_changes["nearest_positive_safe"][mask] - zero)
            fold_raw.setdefault(value.fold, []).append(contrast)
        projected_dots = np.sum(value.directions["lops_pg"] * value.positive_tangents, axis=1)
        constraint_ok &= bool(np.all(projected_dots[value.conflict_mask] <= 1e-12))
        constraint_ok &= bool(np.all(value.constraint_dots[value.conflict_mask] > 0.0))
    if not labels:
        raise ValueError("confirmation has no primary rows")
    label_array = np.concatenate(labels)
    arrays = {
        "raw": np.concatenate(raw),
        "shuffled": np.concatenate(shuffled),
        "positive_only": np.concatenate(positive_only),
        "positive": np.concatenate(positive),
    }
    bootstraps = {
        name: _bootstrap_record(values, label_array, 20260822 + offset, bootstrap_replicates)
        for offset, (name, values) in enumerate(arrays.items())
    }
    eligible = sum(value.labels.size for value in ordered)
    conflicts = sum(int(np.count_nonzero(value.conflict_mask)) for value in ordered)
    primary = arrays["raw"].size
    return {
        "coverage": {
            "eligible_rows": eligible,
            "conflicting_rows": conflicts,
            "conflict_fraction": conflicts / eligible,
            "primary_rows": primary,
            "primary_labels": int(np.unique(label_array).size),
            "folds_with_primary": len(fold_raw),
        },
        "fold_mean_advantages": [
            {"fold": fold, "mean": float(np.concatenate(fold_raw[fold]).mean())}
            for fold in sorted(fold_raw)
        ],
        "pooled": {
            "raw_advantage_mean": float(arrays["raw"].mean()),
            "shuffled_contrast_mean": float(arrays["shuffled"].mean()),
            "positive_only_contrast_mean": float(arrays["positive_only"].mean()),
            "positive_similarity_mean": float(arrays["positive"].mean()),
            "pa_abs_margin_change_median": float(np.median(np.abs(np.concatenate(pa)))),
            "hard_triplet_advantage_mean": float(np.concatenate(hard).mean()),
            "nearest_positive_advantage_mean": float(np.concatenate(nearest).mean()),
        },
        "bootstrap": bootstraps,
        "constraint_integrity": constraint_ok,
    }


def decide_lops_pg(summary: Mapping[str, object]) -> tuple[tuple[dict[str, object], ...], str]:
    """Apply the seven frozen confirmation predicates."""

    coverage = summary["coverage"]
    pooled = summary["pooled"]
    bootstrap = summary["bootstrap"]
    assert (
        isinstance(coverage, Mapping)
        and isinstance(pooled, Mapping)
        and isinstance(bootstrap, Mapping)
    )
    raw = bootstrap["raw"]
    shuffled = bootstrap["shuffled"]
    positive_only = bootstrap["positive_only"]
    positive = bootstrap["positive"]
    assert all(isinstance(value, Mapping) for value in (raw, shuffled, positive_only, positive))
    checks = (
        (
            "coverage",
            coverage["conflict_fraction"] >= 0.5
            and coverage["primary_labels"] >= 100
            and coverage["folds_with_primary"] == 3,
        ),
        ("raw_advantage", pooled["raw_advantage_mean"] > 0.0 and raw["lower_bound"] > 0.0),
        (
            "fold_consistency",
            len(summary["fold_mean_advantages"]) == 3
            and all(value["mean"] > 0.0 for value in summary["fold_mean_advantages"]),
        ),
        (
            "control_superiority",
            shuffled["mean"] > 0.0
            and shuffled["lower_bound"] > 0.0
            and positive_only["mean"] > 0.0
            and positive_only["lower_bound"] > 0.0,
        ),
        (
            "material_effect",
            pooled["raw_advantage_mean"] >= 0.1 * pooled["pa_abs_margin_change_median"],
        ),
        (
            "positive_similarity",
            pooled["positive_similarity_mean"] > 0.0 and positive["lower_bound"] >= -1e-4,
        ),
        ("constraint_integrity", summary["constraint_integrity"] is True),
    )
    predicates = tuple({"name": name, "passed": bool(passed)} for name, passed in checks)
    return predicates, "PASS" if all(value["passed"] for value in predicates) else "KILL"
