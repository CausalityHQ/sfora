"""Confusion-edge margin primitives for candidate 320.

The graph is deliberately built from two independent observations: a sample's
nearest foreign training image and nearest foreign class proxy.  It is not a
hardness scalar; only agreement creates a directed class relation.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np


def build_confusion_edges(
    embeddings: np.ndarray,
    labels: np.ndarray,
    proxies: np.ndarray,
    proxy_labels: np.ndarray,
    *,
    min_support: int = 2,
) -> dict[int, tuple[int, float]]:
    """Return source-class -> target-class confusion edges and support rate."""
    x = np.asarray(embeddings, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(proxies, dtype=np.float64)
    pl = np.asarray(proxy_labels, dtype=np.int64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
    votes: dict[int, Counter[int]] = {}
    for start in range(0, len(x), 512):
        stop = min(start + 512, len(x))
        rows = np.arange(start, stop)
        image_sim = x[rows] @ x.T
        image_sim[np.arange(stop - start), rows] = -np.inf
        foreign_image = np.where(y[rows, None] == y[None, :], -np.inf, image_sim)
        image_target = y[foreign_image.argmax(axis=1)]
        proxy_sim = x[rows] @ p.T
        proxy_sim[pl[None, :] == y[rows, None]] = -np.inf
        proxy_target = pl[proxy_sim.argmax(axis=1)]
        for source, a, b in zip(y[rows], image_target, proxy_target, strict=True):
            if int(a) == int(b):
                votes.setdefault(int(source), Counter())[int(a)] += 1
    edges: dict[int, tuple[int, float]] = {}
    for source, counts in votes.items():
        target, support = counts.most_common(1)[0]
        if support >= min_support:
            edges[source] = (target, float(support / sum(counts.values())))
    return edges


def confusion_edge_margin_loss(
    similarities: Any,
    labels: Any,
    proxy_labels: Any,
    edges: dict[int, tuple[int, float]],
    *,
    margin: float = 0.1,
    torch_module: Any,
) -> Any:
    """Softplus margin on registered directed source/target proxy relations."""
    rows: list[int] = []
    target_positions: list[int] = []
    weights: list[float] = []
    position = {int(label): i for i, label in enumerate(proxy_labels.detach().cpu().tolist())}
    for row, label in enumerate(labels.detach().cpu().tolist()):
        edge = edges.get(int(label))
        if edge is None or int(label) not in position or edge[0] not in position:
            continue
        rows.append(row)
        target_positions.append(position[edge[0]])
        weights.append(edge[1])
    if not rows:
        return similarities.sum() * 0.0
    row_tensor = torch_module.as_tensor(rows, device=similarities.device)
    own_tensor = torch_module.as_tensor([position[int(labels[i])] for i in rows], device=similarities.device)
    target_tensor = torch_module.as_tensor(target_positions, device=similarities.device)
    weight_tensor = torch_module.as_tensor(weights, dtype=similarities.dtype, device=similarities.device)
    gap = similarities[row_tensor, target_tensor] - similarities[row_tensor, own_tensor] + float(margin)
    return (torch_module.nn.functional.softplus(gap) * weight_tensor).sum() / weight_tensor.sum().clamp_min(1e-8)
