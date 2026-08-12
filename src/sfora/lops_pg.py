"""Deterministic geometry for Leave-One-Out Positive-Safe Proxy Gradient."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sfora.idgp import geodesic_step, proxy_anchor_surrogate_tangent

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
    lops[conflict] -= (
        dots[conflict] / np.square(positive_norms[conflict])
    )[:, None] * positive[conflict]

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
        directions={name: value.copy() for name, value in directions.items()},
        margin_changes=margin_changes,
        positive_similarity_changes=positive_changes,
    )
