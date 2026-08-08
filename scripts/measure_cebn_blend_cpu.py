#!/usr/bin/env python3
"""CPU hill-climb diagnostic for the CE-BN head probe.

This applies leave-own-class-out moments to saved corrected In-Shop descriptors,
then blends that transform with the untouched descriptor. It is a mechanism
diagnostic only; it does not authorize a GPU arm by itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-8)


def _cebn(x: np.ndarray, labels: np.ndarray, batch_size: int) -> np.ndarray:
    out = np.empty_like(x, dtype=np.float32)
    for start in range(0, len(x), batch_size):
        stop = min(start + batch_size, len(x))
        xb = x[start:stop].astype(np.float32, copy=False)
        yb = labels[start:stop]
        mask = yb[:, None] != yb[None, :]
        count = mask.sum(axis=1, keepdims=True).astype(np.float32)
        safe = np.maximum(count, 2.0)
        sums = mask.astype(np.float32) @ xb
        sums2 = mask.astype(np.float32) @ (xb * xb)
        mean = sums / safe
        var = np.maximum(sums2 / safe - mean * mean, 0.0)
        out[start:stop] = (xb - mean) / np.sqrt(var + 1e-5)
    return _normalize(out)


def _r1(query: np.ndarray, gallery: np.ndarray, qlabels: np.ndarray, glabels: np.ndarray) -> float:
    correct = 0
    for start in range(0, len(query), 512):
        scores = query[start : start + 512] @ gallery.T
        correct += int(np.sum(qlabels[start : start + 512] == glabels[scores.argmax(axis=1)]))
    return correct / len(query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/tmp"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arrays = {
        split: np.load(args.root / f"inshop_corrected_pa_seed0_{split}_final.npz")
        for split in ("query", "gallery")
    }
    raw_q = _normalize(arrays["query"]["embeddings"].astype(np.float32))
    raw_g = _normalize(arrays["gallery"]["embeddings"].astype(np.float32))
    ce_q = _cebn(raw_q, arrays["query"]["labels"], 64)
    ce_g = _cebn(raw_g, arrays["gallery"]["labels"], 64)
    result = {"baseline_r1": _r1(raw_q, raw_g, arrays["query"]["labels"], arrays["gallery"]["labels"])}
    for lam in (0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        q = _normalize((1.0 - lam) * raw_q + lam * ce_q)
        g = _normalize((1.0 - lam) * raw_g + lam * ce_g)
        result[f"lambda_{lam:.2f}"] = _r1(q, g, arrays["query"]["labels"], arrays["gallery"]["labels"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
