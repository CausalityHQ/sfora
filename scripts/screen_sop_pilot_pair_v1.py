#!/usr/bin/env python3
"""Paired product-bootstrap screen of two SOP pilot receipts (candidate - control) on the 5,851 TRAIN-held queries.

Labels come from the UNICOM SOP TRAIN archive and the seed-179019 fit 0.9 partition used by every SOP pilot.
Decision as in docs/sop_exposure_384d512_2026-10-25.md: PASS iff R@1 lower95 > 0 and mAP@R upper95 >= 0.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from screen_inshop_pilot_pair_v1 import paired
from sfora.representation_ceiling import deterministic_class_partition


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--control", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    a = p.parse_args()
    with np.load(a.archive, allow_pickle=False) as z:
        labels = np.asarray(z["train_labels"], dtype=np.int64)
    held = np.asarray(deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=179019)
                      .validation_row_indexes, dtype=np.int64)
    c, b = (json.loads((d / "receipt.json").read_text())["quality"] for d in (a.candidate, a.control))
    assert len(c["per_query_r1"]) == len(b["per_query_r1"]) == len(held) == 5851
    r1 = paired(np.asarray(c["per_query_r1"]), np.asarray(b["per_query_r1"]), labels[held])
    ap = paired(np.asarray(c["per_query_ap"]), np.asarray(b["per_query_ap"]), labels[held])
    print(json.dumps({"candidate_r1": c["recall_at_1"], "control_r1": b["recall_at_1"], "candidate_map": c["map_at_r"],
                      "control_map": b["map_at_r"], "r1": r1, "map_at_r": ap,
                      "decision": "PASS" if r1["lower95"] > 0 and ap["upper95"] >= 0 else "KILL"}, indent=1))


if __name__ == "__main__":
    main()
