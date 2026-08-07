#!/usr/bin/env python3
"""Measure whether learned proxies over-credit their own training images.

This is a CPU-only Gate-1 diagnostic for Pass 108.  The cross-fitted target for
an image is the centroid of the other deterministic half of its identity;
the image is never included in its own target.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _norm(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embeddings", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch

    pack = np.load(args.embeddings)
    z = _norm(np.asarray(pack["embeddings"], dtype=np.float32))
    labels = np.asarray(pack["labels"], dtype=np.int64)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    proxies = _norm(state["metric_proxies"].detach().cpu().numpy().astype(np.float32))
    proxy_labels = state["metric_proxy_labels"].detach().cpu().numpy().astype(np.int64)
    proxy_by_label = {int(c): proxies[i] for i, c in enumerate(proxy_labels)}
    proxy_index = {int(c): i for i, c in enumerate(proxy_labels)}
    # One dense 25k x 4k multiply is substantially cheaper than rebuilding a
    # foreign-proxy matrix inside every identity loop.
    all_proxy_scores = z @ proxies.T
    rng = np.random.default_rng(args.seed)
    proxy_scores: list[float] = []
    cross_scores: list[float] = []
    proxy_margins: list[float] = []
    cross_margins: list[float] = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        if len(idx) < 4 or int(c) not in proxy_by_label:
            continue
        shuffled = idx[rng.permutation(len(idx))]
        half = len(shuffled) // 2
        a, b = shuffled[:half], shuffled[half:]
        for query, support in ((a, b), (b, a)):
            target = _norm(z[support].mean(axis=0, keepdims=True))[0]
            p = proxy_by_label[int(c)]
            foreign_scores = all_proxy_scores[query].copy()
            foreign_scores[:, proxy_index[int(c)]] = -np.inf
            foreign_max = foreign_scores.max(axis=1)
            proxy_pos = z[query] @ p
            cross_pos = z[query] @ target
            proxy_scores.extend(proxy_pos.tolist())
            cross_scores.extend(cross_pos.tolist())
            proxy_margins.extend((proxy_pos - foreign_max).tolist())
            cross_margins.extend((cross_pos - foreign_max).tolist())
    proxy_scores_a = np.asarray(proxy_scores)
    cross_scores_a = np.asarray(cross_scores)
    proxy_margins_a = np.asarray(proxy_margins)
    cross_margins_a = np.asarray(cross_margins)
    result = {
        "eligible_images": int(len(proxy_scores_a)),
        "proxy_minus_crossfit_mean": float(np.mean(proxy_scores_a - cross_scores_a)),
        "proxy_minus_crossfit_median": float(np.median(proxy_scores_a - cross_scores_a)),
        "fraction_proxy_minus_crossfit_ge_0.03": float(np.mean(proxy_scores_a - cross_scores_a >= 0.03)),
        "proxy_margin_median": float(np.median(proxy_margins_a)),
        "crossfit_margin_median": float(np.median(cross_margins_a)),
        "crossfit_minus_proxy_margin_median": float(np.median(cross_margins_a) - np.median(proxy_margins_a)),
        "gate_pass": bool(
            np.mean(proxy_scores_a - cross_scores_a >= 0.03) >= 0.20
            and np.median(cross_margins_a) - np.median(proxy_margins_a) >= -0.01
        ),
        "seed": args.seed,
    }
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
