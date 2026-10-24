#!/usr/bin/env python3
"""Paired product-bootstrap screen of two In-Shop pilot receipts (candidate - control) on the same held queries.

Resamples held PRODUCTS (5,000 draws, seed 179019) and reports mean delta and 95% interval for R@1 and mAP@R.
Decision per docs/inshop_so400m_384_pilot_2026-10-24.md: PASS iff R@1 lower95 > 0 and mAP@R upper95 >= 0.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_inshop_so400m_pilot import inshop_split  # noqa: E402


def paired(candidate: np.ndarray, control: np.ndarray, clusters: np.ndarray, draws: int = 5000) -> dict:
    products, inverse = np.unique(clusters, return_inverse=True)
    diff_sum = np.bincount(inverse, weights=candidate - control)
    count = np.bincount(inverse).astype(np.float64)
    rng = np.random.default_rng(179019)
    picks = rng.integers(0, len(products), size=(draws, len(products)))
    boot = diff_sum[picks].sum(1) / count[picks].sum(1) * 100
    return {"delta_pp": float((candidate - control).mean() * 100),
            "lower95": float(np.percentile(boot, 2.5)), "upper95": float(np.percentile(boot, 97.5))}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    a = p.parse_args()
    _, labels, _, held, _, _ = inshop_split(a.dataset_root)
    c, b = (json.loads((d / "receipt.json").read_text())["quality"] for d in (a.candidate, a.control))
    r1 = paired(np.asarray(c["per_query_r1"]), np.asarray(b["per_query_r1"]), labels[held])
    ap = paired(np.asarray(c["per_query_ap"]), np.asarray(b["per_query_ap"]), labels[held])
    out = {"candidate_r1": c["recall_at_1"], "control_r1": b["recall_at_1"], "candidate_map": c["map_at_r"],
           "control_map": b["map_at_r"], "r1": r1, "map_at_r": ap,
           "decision": "PASS" if r1["lower95"] > 0 and ap["upper95"] >= 0 else "KILL"}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
