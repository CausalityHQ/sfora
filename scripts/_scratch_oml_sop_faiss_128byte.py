#!/usr/bin/env python3
"""Matched 128-byte Faiss PQ/OPQ controls over official OML SOP features."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from _scratch_oml_sop_compact import paired_bootstrap
from _scratch_same_teacher_ladder import _score

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime

SPECS = ("PQ128x8", "OPQ128_384,PQ128x8")
MINIMUM_MAP_GAIN = 0.002
MAXIMUM_R1_LOSS = 0.0
MAXIMUM_FLOAT_MAP_GAP = 0.003


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
    parser.add_argument("--int4-result", type=Path, required=True)
    parser.add_argument("--int4-result-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-faiss-controls", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    for path, expected in (
        (args.features, args.features_sha256),
        (args.int4_result, args.int4_result_sha256),
        (args.preregistration, args.preregistration_sha256),
        (Path(__file__), args.script_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError("OML Faiss authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if preregistration.get("script_sha256") != args.script_sha256:
        raise ValueError("OML Faiss preregistration differs")
    int4_result = json.loads(args.int4_result.read_text())
    if int4_result.get("representation", {}).get("persistent_bytes_per_item") != 128:
        raise ValueError("OML Faiss parent differs")

    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    faiss = importlib.import_module("faiss")
    faiss.omp_set_num_threads(1)
    with np.load(args.features, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_features"], dtype=np.float32)
        test = np.ascontiguousarray(archive["test_features"], dtype=np.float32)
        labels = tuple(
            int(value)
            for value in np.ascontiguousarray(archive["test_labels"], dtype=np.int64).tolist()
        )
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    test /= np.linalg.norm(test, axis=1, keepdims=True)
    train = np.ascontiguousarray(train)
    test = np.ascontiguousarray(test)
    source = torch.from_numpy(test.copy())
    source_score = _score(
        source, source, labels, labels, same_rows=True, device=torch.device("cuda")
    )
    parent_score = int4_result["arms"]["learned256_int4"]
    arms: dict[str, object] = {}
    best_name = ""
    best_score: dict[str, object] | None = None
    for spec in SPECS:
        started = time.perf_counter()
        codec = faiss.index_factory(384, spec)
        codec.train(train)
        codes = codec.sa_encode(test)
        if codes.shape != (len(test), 128):
            raise ValueError("OML Faiss code width differs")
        reconstructed = codec.sa_decode(codes)
        reconstructed /= np.linalg.norm(reconstructed, axis=1, keepdims=True)
        gallery = torch.from_numpy(np.ascontiguousarray(reconstructed, dtype=np.float32))
        symmetric = _score(
            gallery, gallery, labels, labels, same_rows=True, device=torch.device("cuda")
        )
        asymmetric = _score(
            source, gallery, labels, labels, same_rows=True, device=torch.device("cuda")
        )
        arms[spec] = {
            "persistent_bytes_per_item": int(codes.shape[1]),
            "serialized_model_bytes": int(np.asarray(faiss.serialize_index(codec)).nbytes),
            "symmetric": {
                "map_at_r": symmetric["map_at_r"],
                "recall_at_1": symmetric["recall_at_1"],
            },
            "asymmetric": {
                "map_at_r": asymmetric["map_at_r"],
                "recall_at_1": asymmetric["recall_at_1"],
            },
            "asymmetric_minus_source": {
                "map_at_r": paired_bootstrap(
                    asymmetric["per_query_ap"], source_score["per_query_ap"]
                ),
                "recall_at_1": paired_bootstrap(
                    asymmetric["per_query_r1"], source_score["per_query_r1"]
                ),
            },
            "elapsed_seconds": time.perf_counter() - started,
        }
        if best_score is None or float(asymmetric["map_at_r"]) > float(best_score["map_at_r"]):
            best_name = spec
            best_score = asymmetric
    assert best_score is not None
    best_map = float(best_score["map_at_r"])
    best_r1 = float(best_score["recall_at_1"])
    int4_map = float(parent_score["map_at_r"])
    int4_r1 = float(parent_score["recall_at_1"])
    # The parent receipt does not retain per-query vectors. Aggregate promotion
    # therefore requires both a point margin over it and paired noninferiority
    # to the authenticated float source.
    source_contrast = arms[best_name]["asymmetric_minus_source"]
    passed = bool(
        best_map - int4_map >= MINIMUM_MAP_GAIN
        and best_r1 - int4_r1 >= MAXIMUM_R1_LOSS
        and float(source_score["map_at_r"]) - best_map <= MAXIMUM_FLOAT_MAP_GAP
        and float(source_contrast["map_at_r"]["ci95"][0]) >= -MAXIMUM_FLOAT_MAP_GAP
    )
    result = {
        "schema": "scratch-oml-sop-faiss-128byte-v1",
        "claim_eligible": False,
        "dataset": "Stanford Online Products",
        "faiss_version": faiss.__version__,
        "omp_threads": faiss.omp_get_max_threads(),
        "source": {
            "map_at_r": source_score["map_at_r"],
            "recall_at_1": source_score["recall_at_1"],
        },
        "learned256_int4_parent": parent_score,
        "arms": arms,
        "winner": best_name,
        "winner_minus_learned256_int4": {
            "map_at_r": best_map - int4_map,
            "recall_at_1": best_r1 - int4_r1,
        },
        "gate": {"passed": passed},
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
