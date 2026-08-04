#!/usr/bin/env python3
"""Independently recompute ARCG/IPSR Gate-0 statistics from a raw embedding pack.

This intentionally imports no project diagnostic or preference-construction code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def normalize(values: np.ndarray, *, axis: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values / np.maximum(np.linalg.norm(values, axis=axis, keepdims=True), 1.0e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pack", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    with np.load(args.pack, allow_pickle=False) as pack:
        anchors = normalize(pack["anchor_embeddings"], axis=1)
        transformed = normalize(pack["transformed_embeddings"], axis=2)
        labels = np.asarray(pack["labels"], dtype=np.int64)
        view_names = [str(item) for item in pack["view_names"].tolist()]

    if anchors.shape != (25_882, 512):
        raise RuntimeError(f"unexpected corrected In-Shop anchor shape: {anchors.shape}")
    if transformed.shape != (25_882, 5, 512):
        raise RuntimeError(f"unexpected transformed shape: {transformed.shape}")
    if view_names != ["flip", "left", "right", "top", "bottom"]:
        raise RuntimeError(f"unexpected deterministic view order: {view_names}")
    unique, counts = np.unique(labels, return_counts=True)
    if len(unique) != 3_997 or int(counts.sum()) != 25_882:
        raise RuntimeError("pack does not contain the official In-Shop training partition")

    responses = 1.0 - np.einsum("nd,nvd->nv", anchors, transformed)
    medians = np.median(responses, axis=0)
    mads = np.median(np.abs(responses - medians), axis=0)
    standardized = (responses - medians) / np.maximum(mads, 1.0e-6)
    centred = standardized - standardized.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centred, axis=1)
    valid = norms >= 1.0e-6
    signatures = np.zeros_like(centred)
    signatures[valid] = centred[valid] / norms[valid, None]

    preferred = np.full(len(labels), -1, dtype=np.int64)
    unknown = np.full(len(labels), -1, dtype=np.int64)
    initial_losses: list[float] = []
    eligible_classes = covered_classes = 0
    eligible_edges = total_edges = 0
    closest_rejected = closest_total = 0
    farthest_accepted = farthest_total = 0
    multicomponent = multicomponent_denominator = 0

    for label in unique:
        members = np.flatnonzero(labels == label)
        size = len(members)
        if size >= 2:
            rows, cols = np.triu_indices(size, k=1)
            left, right = members[rows], members[cols]
            agreement = np.einsum("nd,nd->n", signatures[left], signatures[right])
            keep = valid[left] & valid[right] & (agreement >= 0.5)
            distance = 1.0 - np.einsum("nd,nd->n", anchors[left], anchors[right])
            eligible_edges += int(keep.sum())
            total_edges += len(keep)
            order = np.argsort(distance, kind="stable")
            width = max(1, len(order) // 4)
            closest, farthest = order[:width], order[-width:]
            closest_rejected += int((~keep[closest]).sum())
            closest_total += len(closest)
            farthest_accepted += int(keep[farthest].sum())
            farthest_total += len(farthest)
            if size >= 3:
                multicomponent_denominator += 1
                parent = np.arange(size, dtype=np.int64)

                def find(node: int) -> int:
                    while parent[node] != node:
                        parent[node] = parent[parent[node]]
                        node = int(parent[node])
                    return node

                for row, col, retained in zip(rows, cols, keep, strict=True):
                    if retained:
                        parent[find(int(row))] = find(int(col))
                multicomponent += int(len({find(node) for node in range(size)}) > 1)

        if size < 3:
            continue
        eligible_classes += 1
        class_covered = False
        for anchor_index in members:
            if not valid[anchor_index]:
                continue
            peers = members[members != anchor_index]
            agreement = signatures[peers] @ signatures[anchor_index]
            similarity = anchors[peers] @ anchors[anchor_index]
            compatible = valid[peers] & (agreement >= 0.5)
            incompatible = valid[peers] & (agreement < 0.5)
            if not compatible.any() or not incompatible.any():
                continue
            incompatible_rows = np.flatnonzero(incompatible)
            u_row = incompatible_rows[int(np.argmax(similarity[incompatible_rows]))]
            contradicted = compatible & (similarity < similarity[u_row])
            if not contradicted.any():
                continue
            compatible_rows = np.flatnonzero(contradicted)
            p_row = compatible_rows[int(np.argmax(similarity[compatible_rows]))]
            preferred[anchor_index] = peers[p_row]
            unknown[anchor_index] = peers[u_row]
            initial_losses.append(
                float(np.logaddexp(0.0, similarity[u_row] - similarity[p_row]))
            )
            class_covered = True
        covered_classes += int(class_covered)

    preference_count = int((preferred >= 0).sum())
    payload = {
        "eligible_edges": eligible_edges,
        "total_edges": total_edges,
        "density": eligible_edges / total_edges,
        "multicomponent_fraction": multicomponent / multicomponent_denominator,
        "closest_quartile_rejected_fraction": closest_rejected / closest_total,
        "farthest_quartile_accepted_fraction": farthest_accepted / farthest_total,
        "valid_signature_fraction": float(valid.mean()),
        "preference_count": preference_count,
        "anchor_coverage": preference_count / len(labels),
        "class_coverage": covered_classes / eligible_classes,
        "mean_initial_loss": float(np.mean(initial_losses)),
        "pack": str(args.pack),
    }
    rendered = json.dumps(payload, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
