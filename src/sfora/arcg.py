"""Pure diagnostics for augmentation-response compatibility graphs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class ARCGDiagnostics:
    eligible_edges: int
    total_edges: int
    density: float
    multicomponent_fraction: float
    closest_quartile_rejected_fraction: float
    farthest_quartile_accepted_fraction: float


def normalized_response_signatures(
    anchor_embeddings: NDArray[np.floating],
    transformed_embeddings: NDArray[np.floating],
    *,
    floor: float = 1e-6,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Return registered MAD-standardized, within-image-centred signatures."""
    anchor = _row_normalize(np.asarray(anchor_embeddings, dtype=np.float64), floor)
    transformed = np.asarray(transformed_embeddings, dtype=np.float64)
    if transformed.ndim != 3 or transformed.shape[0] != anchor.shape[0]:
        raise ValueError("transformed embeddings must have shape (images, views, dimensions)")
    transformed = transformed / np.maximum(
        np.linalg.norm(transformed, axis=2, keepdims=True), floor
    )
    responses = 1.0 - np.einsum("nd,nvd->nv", anchor, transformed)
    median = np.median(responses, axis=0)
    mad = np.median(np.abs(responses - median), axis=0)
    standardized = (responses - median) / np.maximum(mad, floor)
    centred = standardized - standardized.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1, keepdims=True)
    valid = norms[:, 0] >= floor
    signatures = np.zeros_like(centred)
    signatures[valid] = centred[valid] / norms[valid]
    return signatures, valid


def diagnose_arcg_graph(
    anchor_embeddings: NDArray[np.floating],
    signatures: NDArray[np.floating],
    labels: NDArray[np.integer],
    valid: NDArray[np.bool_],
    *,
    agreement_threshold: float = 0.5,
) -> ARCGDiagnostics:
    """Measure graph selectivity and distance/response asymmetry within classes."""
    anchor = _row_normalize(np.asarray(anchor_embeddings, dtype=np.float64), 1e-6)
    signatures = np.asarray(signatures, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    valid = np.asarray(valid, dtype=bool)
    eligible_edges = total_edges = 0
    multicomp = multicomp_denominator = 0
    closest_rejected = closest_total = 0
    farthest_accepted = farthest_total = 0

    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        size = len(indices)
        if size < 2:
            continue
        rows, cols = np.triu_indices(size, k=1)
        left, right = indices[rows], indices[cols]
        agreements = np.einsum("nd,nd->n", signatures[left], signatures[right])
        eligible = valid[left] & valid[right] & (agreements >= agreement_threshold)
        distances = 1.0 - np.einsum("nd,nd->n", anchor[left], anchor[right])
        total_edges += len(rows)
        eligible_edges += int(eligible.sum())

        order = np.argsort(distances, kind="stable")
        quartile_size = max(1, len(order) // 4)
        closest = order[:quartile_size]
        farthest = order[-quartile_size:]
        closest_total += len(closest)
        closest_rejected += int((~eligible[closest]).sum())
        farthest_total += len(farthest)
        farthest_accepted += int(eligible[farthest].sum())

        if size >= 3:
            multicomp_denominator += 1
            parent = list(range(size))

            def find(node: int, forest: list[int] = parent) -> int:
                while forest[node] != node:
                    forest[node] = forest[forest[node]]
                    node = forest[node]
                return node

            for row, col, keep in zip(rows, cols, eligible, strict=True):
                if not keep:
                    continue
                root_a, root_b = find(int(row)), find(int(col))
                parent[root_a] = root_b
            multicomp += int(len({find(node) for node in range(size)}) > 1)

    return ARCGDiagnostics(
        eligible_edges=eligible_edges,
        total_edges=total_edges,
        density=eligible_edges / total_edges if total_edges else 0.0,
        multicomponent_fraction=(
            multicomp / multicomp_denominator if multicomp_denominator else 0.0
        ),
        closest_quartile_rejected_fraction=(
            closest_rejected / closest_total if closest_total else 0.0
        ),
        farthest_quartile_accepted_fraction=(
            farthest_accepted / farthest_total if farthest_total else 0.0
        ),
    )


def _row_normalize(values: NDArray[np.float64], floor: float) -> NDArray[np.float64]:
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), floor)
