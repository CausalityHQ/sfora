#!/usr/bin/env python3
"""Frozen float-query/int4-gallery gate over the learned OML SOP head."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from _scratch_oml_sop_compact import paired_bootstrap
from _scratch_same_teacher_ladder import _score
from _scratch_sop_direct256_int4 import fit_pack_decode_int4
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

MINIMUM_MAP_GAIN = 0.002
MAXIMUM_FLOAT_MAP_GAP = 0.003
MINIMUM_R1_GAIN = 0.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--parent-receipt", type=Path, required=True)
    parser.add_argument("--parent-receipt-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-asymmetric-gate", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    for path, expected in (
        (args.features, args.features_sha256),
        (args.checkpoint, args.checkpoint_sha256),
        (args.parent_receipt, args.parent_receipt_sha256),
        (args.preregistration, args.preregistration_sha256),
        (Path(__file__), args.script_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError("OML asymmetric authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    for key, expected in (
        ("features_sha256", args.features_sha256),
        ("checkpoint_sha256", args.checkpoint_sha256),
        ("parent_receipt_sha256", args.parent_receipt_sha256),
        ("script_sha256", args.script_sha256),
    ):
        if preregistration.get(key) != expected:
            raise ValueError("OML asymmetric preregistration differs")
    parent = json.loads(args.parent_receipt.read_text())
    if (
        parent.get("checkpoint_sha256") != args.checkpoint_sha256
        or parent.get("gate", {}).get("passed") is not False
        or parent.get("representation", {}).get("persistent_bytes_per_item") != 128
    ):
        raise ValueError("OML asymmetric parent differs")

    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    with np.load(args.features, allow_pickle=False) as archive:
        train = F.normalize(
            torch.from_numpy(np.ascontiguousarray(archive["train_features"])).float(), dim=1
        ).contiguous()
        test = F.normalize(
            torch.from_numpy(np.ascontiguousarray(archive["test_features"])).float(), dim=1
        ).contiguous()
        labels = tuple(
            int(value)
            for value in np.ascontiguousarray(archive["test_labels"], dtype=np.int64).tolist()
        )
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    weight = checkpoint["weight"]
    bias = checkpoint["bias"]
    if weight.shape != (256, 384) or bias.shape != (256,):
        raise ValueError("OML asymmetric checkpoint shape differs")
    fit_float = F.normalize(F.linear(train, weight, bias), dim=1).contiguous()
    test_float = F.normalize(F.linear(test, weight, bias), dim=1).contiguous()
    test_int4, packing = fit_pack_decode_int4(fit_float, test_float)
    symmetric = _score(
        test_int4, test_int4, labels, labels, same_rows=True, device=device
    )
    asymmetric = _score(
        test_float, test_int4, labels, labels, same_rows=True, device=device
    )
    floating = _score(
        test_float, test_float, labels, labels, same_rows=True, device=device
    )
    map_gain = paired_bootstrap(asymmetric["per_query_ap"], symmetric["per_query_ap"])
    r1_gain = paired_bootstrap(asymmetric["per_query_r1"], symmetric["per_query_r1"])
    float_gap = float(floating["map_at_r"]) - float(asymmetric["map_at_r"])
    passed = bool(
        float(map_gain["delta"]) >= MINIMUM_MAP_GAIN
        and float(map_gain["ci95"][0]) > 0.0
        and float(r1_gain["delta"]) >= MINIMUM_R1_GAIN
        and float_gap <= MAXIMUM_FLOAT_MAP_GAP
    )
    result = {
        "schema": "scratch-oml-sop-asymmetric-int4-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "split": "official train fit; official leave-one-out test evaluation",
        "representation": {
            "persistent_gallery_bytes_per_item": 128,
            "transient_query_dimensions": 256,
            **packing,
        },
        "arms": {
            "symmetric_int4": {
                "map_at_r": symmetric["map_at_r"],
                "recall_at_1": symmetric["recall_at_1"],
            },
            "asymmetric_float_query_int4_gallery": {
                "map_at_r": asymmetric["map_at_r"],
                "recall_at_1": asymmetric["recall_at_1"],
            },
            "float_query_float_gallery": {
                "map_at_r": floating["map_at_r"],
                "recall_at_1": floating["recall_at_1"],
            },
        },
        "contrasts": {
            "asymmetric_minus_symmetric_map_at_r": map_gain,
            "asymmetric_minus_symmetric_recall_at_1": r1_gain,
            "float_minus_asymmetric_map_at_r": float_gap,
        },
        "gate": {
            "minimum_map_gain": MINIMUM_MAP_GAIN,
            "maximum_float_map_gap": MAXIMUM_FLOAT_MAP_GAP,
            "minimum_recall_at_1_gain": MINIMUM_R1_GAIN,
            "passed": passed,
        },
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
