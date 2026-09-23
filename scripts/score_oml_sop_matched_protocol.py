#!/usr/bin/env python3
"""Rescore the authenticated OML SOP features with Sfora's stable-ordinal protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.sop_evaluation import score_symmetric

FEATURES_SHA256 = "8f565027b20e55923826a5240d97c564171428a5f1cc7290e680d2223fd28da4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-matched-protocol", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.features) != FEATURES_SHA256:
        raise ValueError("OML SOP matched scorer authority differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    with np.load(args.features, allow_pickle=False) as archive:
        features = np.ascontiguousarray(archive["test_features"])
        labels = np.ascontiguousarray(archive["test_labels"])
        ids = np.ascontiguousarray(archive["test_ids"])
    if (
        features.shape != (60_502, 384)
        or labels.shape != (60_502,)
        or ids.shape != (60_502,)
        or len(np.unique(labels)) != 11_316
        or len(np.unique(ids)) != 60_502
    ):
        raise ValueError("OML SOP matched scorer rows differ")
    torch.backends.cuda.matmul.allow_tf32 = False
    values = F.normalize(torch.from_numpy(features).to("cuda"), dim=1)
    result = score_symmetric(values, torch.from_numpy(labels).to("cuda"))
    receipt = {
        "schema": "sfora-oml-sop-matched-scorer-v1",
        "claim_eligible": False,
        "split": "official SOP test; already observed; leave-one-out stable ordinal scorer",
        "features_sha256": FEATURES_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "scorer_sha256": sha256(Path(__file__).parents[1] / "src/sfora/sop_evaluation.py"),
        "gpu": torch.cuda.get_device_name(),
        "score": result,
    }
    payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    with args.output.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"recall_at_1": result["recall_at_1"], "map_at_r": result["map_at_r"]}))


if __name__ == "__main__":
    main()
