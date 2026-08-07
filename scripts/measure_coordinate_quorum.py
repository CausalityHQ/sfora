#!/usr/bin/env python3
"""Measure how much training-class identity leaks through coordinate subsets.

This is a training-only diagnostic for the SQLS proposal. It uses an
within-class image split and a nearest-class-centroid probe; official test
images are never loaded.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("pack", type=Path)
    ap.add_argument("--seed", type=int, default=8107)
    ap.add_argument("--repeats", type=int, default=20)
    args = ap.parse_args()
    a = np.load(args.pack, allow_pickle=True)
    x = np.asarray(a["embeddings"], dtype=np.float64)
    y = np.asarray(a["labels"])
    rng = np.random.default_rng(args.seed)
    classes = np.asarray([c for c in np.unique(y) if np.sum(y == c) >= 2])
    train: list[int] = []
    valid: list[int] = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        rng.shuffle(idx)
        n = min(max(1, int(np.floor(0.8 * len(idx)))), len(idx) - 1)
        train.extend(idx[:n].tolist())
        valid.extend(idx[n:].tolist())
    train_idx = np.asarray(train)
    valid_idx = np.asarray(valid)
    centroids = np.stack(
        [x[train_idx[y[train_idx] == c]].mean(axis=0) for c in classes]
    )
    global_mean = x[train_idx].mean(axis=0)
    between = np.zeros(x.shape[1])
    within = np.zeros(x.shape[1])
    for c in classes:
        z = x[train_idx[y[train_idx] == c]]
        between += len(z) * (z.mean(axis=0) - global_mean) ** 2
        within += ((z - z.mean(axis=0)) ** 2).sum(axis=0)
    order = np.argsort(between / (within + 1e-12))[::-1]
    truth = y[valid_idx]
    rows = []
    for k in (32, 64, 128, 256, 512):
        scores = []
        for _ in range(args.repeats):
            mask = np.arange(x.shape[1]) if k == x.shape[1] else rng.choice(
                x.shape[1], k, replace=False
            )
            z = x[valid_idx][:, mask]
            c = centroids[:, mask]
            z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
            c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
            scores.append(float(np.mean(classes[np.argmax(z @ c.T, axis=1)] == truth)))
        mask = order[:k]
        z = x[valid_idx][:, mask]
        c = centroids[:, mask]
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
        c /= np.linalg.norm(c, axis=1, keepdims=True) + 1e-12
        rows.append(
            {
                "k": k,
                "random_mean": float(np.mean(scores)),
                "random_sd": float(np.std(scores, ddof=0)),
                "top_between_within_mean": float(
                    np.mean(classes[np.argmax(z @ c.T, axis=1)] == truth)
                ),
            }
        )
    print(
        json.dumps(
            {
                "pack": str(args.pack),
                "seed": args.seed,
                "train_images": int(len(train_idx)),
                "validation_images": int(len(valid_idx)),
                "classes": int(len(classes)),
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
