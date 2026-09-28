#!/usr/bin/env python3
"""Reject source-preservation hypotheses from pinned full-gallery TRAIN ranks."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from preflight_inshop_siglip2_unseen_gallery import split
from probe_inshop_warmstart_packing import EVIDENCE
from score_inshop_crop_view_pair import PARTITION_SHA, QUERY_SHA, roles, sha256

PACKING_SHA = "52aeec228915e26b5272d868627b905f29b933a633519fd6388dc8279bd8fd0e"
SOURCE_SHA = "d57e2acbf8dbe559d7b9f7dced64df90d2fe32c8c56f020d08ced1c1a6c9ca82"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--partition", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    packing_path = EVIDENCE / "inshop-sop-warmstart-packing-diagnostic-v1.json"
    source_path = EVIDENCE / "inshop-sop-warmstart-smoke-v1/matched_source_quality.json"
    if (
        args.output.exists()
        or sha256(args.partition) != PARTITION_SHA
        or sha256(packing_path) != PACKING_SHA
        or sha256(source_path) != SOURCE_SHA
    ):
        raise ValueError("persistent-source authority differs")
    packing, source = json.loads(packing_path.read_text()), json.loads(source_path.read_text())
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    _, held = split(tuple(row[1] for row in train))
    labels = tuple(train[row][1] for row in held)
    root = Path("/dataset")
    query, gallery = roles(labels, tuple(root / "Img" / train[row][0] for row in held), root)
    import hashlib

    if hashlib.sha256(np.asarray(query, dtype="<i4").tobytes()).hexdigest() != QUERY_SHA:
        raise ValueError("persistent-source query order differs")
    counts = Counter(labels[row] for row in gallery)
    strata = np.asarray([counts[labels[row]] for row in query])
    misses = np.asarray(packing["treatment_persistent_misses"])
    core = np.asarray(packing["all_six_persistent_misses"])
    stable_hits = np.all(
        [
            np.asarray(row["treatment"]["packed"]["per_query_r1"]) == 1
            for row in packing["results"].values()
        ],
        axis=0,
    )
    rng = np.random.default_rng(179019)
    controls = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(stable_hits & (strata == count)),
                size=int((strata[misses] == count).sum()),
                replace=False,
            )
            for count in np.unique(strata[misses])
        ]
    )
    if len(misses) != 84 or len(core) != 64 or len(controls) != len(misses):
        raise ValueError("persistent-source inventory differs")
    results = {}
    for arm in ("pretrained", "sop"):
        hits = np.asarray(source["quality"][arm]["per_query_r1"])
        if hits.shape != (6354,) or not np.isin(hits, [0, 1]).all():
            raise ValueError("persistent-source ranks differ")
        results[arm] = {
            "persistent_miss_hits": int(hits[misses].sum()),
            "all_six_miss_hits": int(hits[core].sum()),
            "matched_control_hits": int(hits[controls].sum()),
        }
    report = {
        "schema": "sfora-inshop-persistent-source-diagnostic-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN held products; full gallery source ranks",
        "source_sha256": sha256(Path(__file__)),
        "packing_receipt_sha256": PACKING_SHA,
        "source_rank_receipt_sha256": SOURCE_SHA,
        "results": results,
        "persistent_miss_query_ordinals": misses.tolist(),
        "all_six_miss_query_ordinals": core.tolist(),
        "matched_hit_control_query_ordinals": controls.tolist(),
        "source_preservation_diagnostic_advance": results["sop"]["persistent_miss_hits"] >= 42,
        "encoder_export_or_training": False,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"results": results, "advance": report["source_preservation_diagnostic_advance"]}
        )
    )


if __name__ == "__main__":
    main()
