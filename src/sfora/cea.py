"""Counterfactual-evidence agreement graphs for train-time DML supervision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CEAGraph:
    """Detached same-class eligibility graph and its operating diagnostics."""

    neighbours: tuple[tuple[tuple[int, float], ...], ...]
    edge_density: float
    multicomponent_fraction: float
    close_rejected_fraction: float
    far_accepted_fraction: float


def distance_budget_control(
    embeddings: NDArray[np.floating],
    labels: NDArray[np.integer],
    reference: CEAGraph,
) -> CEAGraph:
    """Select the same per-class edge budget using embedding distance only."""
    z = np.asarray(embeddings, dtype=np.float64)
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    y = np.asarray(labels, dtype=np.int64)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(len(y))]
    eligible_edges = total_edges = 0
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        if len(members) < 2:
            continue
        rows, cols = np.triu_indices(len(members), k=1)
        total_edges += len(rows)
        budget = sum(1 for i in members for j, _ in reference.neighbours[int(i)] if j > int(i) and y[j] == label)
        order = np.argsort(
            -np.einsum("nd,nd->n", z[members[rows]], z[members[cols]]), kind="stable"
        )[:budget]
        eligible_edges += len(order)
        for k in order:
            i, j = int(members[rows[k]]), int(members[cols[k]])
            neighbours[i].append((j, 1.0))
            neighbours[j].append((i, 1.0))
    return CEAGraph(
        neighbours=tuple(tuple(row) for row in neighbours),
        edge_density=eligible_edges / total_edges if total_edges else 0.0,
        multicomponent_fraction=0.0,
        close_rejected_fraction=0.0,
        far_accepted_fraction=0.0,
    )


def build_cea_graph(
    embeddings: NDArray[np.floating],
    signatures: NDArray[np.floating],
    labels: NDArray[np.integer],
    *,
    agreement_threshold: float,
) -> CEAGraph:
    """Gate same-class edges by agreement of detached evidence signatures.

    The graph is deliberately binary: an eligible edge is a positive relation;
    every other labelled same-class pair is positive-to-unknown.  This function
    performs no soft weighting and does not alter inference descriptors.
    """
    z = np.asarray(embeddings, dtype=np.float64)
    s = np.asarray(signatures, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if z.ndim != 2 or s.ndim != 2 or z.shape[0] != s.shape[0] or z.shape[0] != y.shape[0]:
        raise ValueError("embeddings, signatures, and labels must share their first dimension")
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    s /= np.maximum(np.linalg.norm(s, axis=1, keepdims=True), 1e-12)
    neighbours: list[list[tuple[int, float]]] = [[] for _ in range(len(y))]
    eligible_edges = total_edges = 0
    closest_rejected = closest_total = 0
    farthest_accepted = farthest_total = 0
    multicomp = multicomp_total = 0
    for label in np.unique(y):
        members = np.flatnonzero(y == label)
        if len(members) < 2:
            continue
        rows, cols = np.triu_indices(len(members), k=1)
        left, right = members[rows], members[cols]
        agreement = np.einsum("nd,nd->n", s[left], s[right])
        eligible = agreement >= agreement_threshold
        distances = 1.0 - np.einsum("nd,nd->n", z[left], z[right])
        total_edges += len(rows)
        eligible_edges += int(eligible.sum())
        for i, j in zip(left[eligible], right[eligible], strict=True):
            neighbours[int(i)].append((int(j), 1.0))
            neighbours[int(j)].append((int(i), 1.0))
        order = np.argsort(distances, kind="stable")
        quartile = max(1, len(order) // 4)
        close = order[:quartile]
        far = order[-quartile:]
        closest_total += len(close)
        closest_rejected += int((~eligible[close]).sum())
        farthest_total += len(far)
        farthest_accepted += int(eligible[far].sum())
        if len(members) >= 3:
            multicomp_total += 1
            parent = list(range(len(members)))

            def find(node: int) -> int:
                while parent[node] != node:
                    parent[node] = parent[parent[node]]
                    node = parent[node]
                return node

            for row, col, keep in zip(rows, cols, eligible, strict=True):
                if keep:
                    parent[find(int(row))] = find(int(col))
            multicomp += int(len({find(i) for i in range(len(members))}) > 1)
    return CEAGraph(
        neighbours=tuple(tuple(row) for row in neighbours),
        edge_density=eligible_edges / total_edges if total_edges else 0.0,
        multicomponent_fraction=multicomp / multicomp_total if multicomp_total else 0.0,
        close_rejected_fraction=closest_rejected / closest_total if closest_total else 0.0,
        far_accepted_fraction=farthest_accepted / farthest_total if farthest_total else 0.0,
    )
