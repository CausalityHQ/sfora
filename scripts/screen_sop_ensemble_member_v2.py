#!/usr/bin/env python3
"""Add-a-member ensemble screen for SOP So400m@384/512 (TRAIN-held, pre-registered 2026-10-31).

Incumbent = renormalised mean of --base runs (two-seed ensemble 179024+179025; official 94.129).
Primary arm = incumbent + detail-crop model (receipt 923c69eb; same R@1 as seed 179024 but a disjoint error set,
34 fixed / 34 broken), i.e. the renormalised mean of --base + --add. --extra arms are DESCRIPTIVE only (no selection).
Scored like the pilots: packed int8 512-D, 5,851 held queries vs the full 59,551 TRAIN gallery, self excluded.
Rule: PASS iff primary-vs-incumbent paired product-bootstrap R@1 lower95 > 0 and mAP upper95 >= 0
=> ONE official read of the three-member ensemble (new SOP incumbent iff paired official R@1 vs 94.129 lower95 > 0).
Cost: 3x encode, 1,542 B/row (quality-first policy). Else KILL.
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


def score(runs, labels, held):
    e = torch.nn.functional.normalize(sum(torch.from_numpy(np.load(r / "train_embeddings.npy")) for r in runs), dim=1)
    packed = pack_int8_unit_embeddings(e)
    return dim512.score_packed_full_gallery(packed.codes.float(), packed.inverse_norms, torch.from_numpy(labels.copy()),
                                            torch.from_numpy(held.copy()), device=torch.device("cuda"))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", type=Path, nargs="+", required=True)
    p.add_argument("--add", type=Path, required=True)
    p.add_argument("--extra", type=Path, nargs="*", default=[])
    p.add_argument("--archive", type=Path, required=True)
    a = p.parse_args()
    with np.load(a.archive, allow_pickle=False) as z:
        labels = np.asarray(z["train_labels"], dtype=np.int64)
    held = np.asarray(deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=179019)
                      .validation_row_indexes, dtype=np.int64)
    base = score(a.base, labels, held)
    out = {"incumbent": [base["recall_at_1"], base["map_at_r"]]}

    def vs_base(q):
        return {"quality": [q["recall_at_1"], q["map_at_r"]],
                "r1": paired(np.asarray(q["per_query_r1"]), np.asarray(base["per_query_r1"]), labels[held]),
                "map": paired(np.asarray(q["per_query_ap"]), np.asarray(base["per_query_ap"]), labels[held])}

    out["primary"] = vs_base(score([*a.base, a.add], labels, held))
    out["descriptive"] = {str(x): vs_base(score([*a.base, x], labels, held)) for x in a.extra}
    pr = out["primary"]
    out["decision"] = "PASS" if pr["r1"]["lower95"] > 0 and pr["map"]["upper95"] >= 0 else "KILL"
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
