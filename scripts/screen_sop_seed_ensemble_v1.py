#!/usr/bin/env python3
"""Two-seed embedding-ensemble screen for SOP So400m@384/512 (TRAIN-held, pre-registered 2026-10-25).

Ensemble = renormalized mean of two seeds' unit train_embeddings.npy. It is packed int8 512-D and scored exactly
like the pilots: 5,851 held queries vs the full 59,551 TRAIN gallery, self excluded, score_packed_full_gallery.
Paired product bootstrap vs the single seed-179024 receipt.
Rule: PASS iff R@1 lower95 > 0 and mAP upper95 >= 0 => ONE official read of the 2-seed ensemble
(new SOP incumbent iff paired official R@1 vs 93.967 lower95 > 0). Cost: 2x encode (quality-first policy).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_sop_so400m_384_dim512_pilot as dim512  # noqa: E402
from screen_inshop_pilot_pair_v1 import paired  # noqa: E402
from sfora.joint_relational_compaction import pack_int8_unit_embeddings  # noqa: E402
from sfora.representation_ceiling import deterministic_class_partition  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, nargs=2, required=True, help="seed-179024 run dir first")
    p.add_argument("--archive", type=Path, required=True)
    a = p.parse_args()
    with np.load(a.archive, allow_pickle=False) as z:
        labels = np.asarray(z["train_labels"], dtype=np.int64)
    held = np.asarray(deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=179019)
                      .validation_row_indexes, dtype=np.int64)
    e = sum(torch.from_numpy(np.load(r / "train_embeddings.npy")) for r in a.runs)
    e = torch.nn.functional.normalize(e, dim=1)
    packed = pack_int8_unit_embeddings(e)
    q = dim512.score_packed_full_gallery(packed.codes.float(), packed.inverse_norms, torch.from_numpy(labels.copy()),
                                         torch.from_numpy(held.copy()), device=torch.device("cuda"))
    base = json.loads((a.runs[0] / "receipt.json").read_text())["quality"]
    other = json.loads((a.runs[1] / "receipt.json").read_text())["quality"]
    out = {"ensemble": [q["recall_at_1"], q["map_at_r"]], "seed_179024": [base["recall_at_1"], base["map_at_r"]],
           "seed_other": [other["recall_at_1"], other["map_at_r"]],
           "r1": paired(np.asarray(q["per_query_r1"]), np.asarray(base["per_query_r1"]), labels[held]),
           "map": paired(np.asarray(q["per_query_ap"]), np.asarray(base["per_query_ap"]), labels[held])}
    out["decision"] = "PASS" if out["r1"]["lower95"] > 0 and out["map"]["upper95"] >= 0 else "KILL"
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
