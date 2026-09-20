#!/usr/bin/env python3
"""Throwaway SOP float ceiling for reallocating 1,024 code bits to 256 dims."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.nn import functional as F


SOURCE_SHA256 = "6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818"
TEACHER_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
HELPER_SHA256 = "0ebd8bd47f4ee9b1606d5ca06dbf799f8c95b2a9894d1ae1d06208c4e23f68cf"
PCA128_FLOAT_MAP = 0.45763895695082696
MAP_GAIN_GATE = 0.006


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-float-ceiling", action="store_true", required=True)
    args = parser.parse_args()
    preregistration = json.loads(args.preregistration.read_text())
    if (
        args.output.exists()
        or sha256(Path(__file__)) != args.script_sha256
        or sha256(args.preregistration) != args.preregistration_sha256
        or preregistration["script_sha256"] != args.script_sha256
        or sha256(args.source_snapshot) != SOURCE_SHA256
        or sha256(args.teacher_snapshot) != TEACHER_SHA256
        or sha256(args.helper) != HELPER_SHA256
    ):
        raise ValueError("SOP PCA256 float ceiling authority differs")

    sys.path.insert(0, str(args.helper.parent))
    from probe_sop_relational_linear import load_paired_archives, score_symmetric
    from sfora.deterministic_similarity_runtime import (
        configure_deterministic_similarity_runtime,
    )

    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    started = time.monotonic()
    pair = load_paired_archives(
        args.source_snapshot,
        SOURCE_SHA256,
        args.teacher_snapshot,
        TEACHER_SHA256,
    )
    train = F.normalize(pair["teacher_train"].float(), dim=1).contiguous()
    test = F.normalize(pair["teacher_test"].float(), dim=1).contiguous()
    labels = pair["test_labels"]
    centre = train.mean(dim=0)
    centred = train - centre
    covariance = (centred.T @ centred) / (centred.shape[0] - 1)
    values, vectors = torch.linalg.eigh(covariance.double())
    order = torch.argsort(values, descending=True)[:256]
    components = vectors[:, order].float().contiguous()
    projected = F.normalize((test - centre) @ components, dim=1).contiguous()
    score = score_symmetric(
        projected,
        labels,
        candidate_width=max(Counter(labels).values()) - 1,
        device=torch.device("cuda"),
    )
    pca256_map = float(score["map_at_r"])
    gain = pca256_map - PCA128_FLOAT_MAP
    result = {
        "schema": "scratch-sop-pca256-float-ceiling-v1",
        "claim_eligible": False,
        "dataset": "stanford-online-products-official-test-already-observed",
        "fit_rows": int(train.shape[0]),
        "evaluation_rows": int(test.shape[0]),
        "source_sha256": SOURCE_SHA256,
        "teacher_sha256": TEACHER_SHA256,
        "helper_sha256": HELPER_SHA256,
        "script_sha256": args.script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "pca128_float_map_at_r_reference": PCA128_FLOAT_MAP,
        "pca256_float": {
            "map_at_r": pca256_map,
            "recall_at_1": float(score["r1"]),
        },
        "pca256_minus_pca128_map_at_r": gain,
        "gate": MAP_GAIN_GATE,
        "passed": gain >= MAP_GAIN_GATE,
        "next": "test global-scale PCA256 int4" if gain >= MAP_GAIN_GATE else "close rate-reallocation lane",
        "elapsed_seconds": time.monotonic() - started,
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
