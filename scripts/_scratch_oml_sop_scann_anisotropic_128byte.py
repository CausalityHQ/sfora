#!/usr/bin/env python3
"""Matched 128-byte ScaNN anisotropic-hashing gate on official OML SOP."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import scann

BLOCK_DIMENSIONS = 3
HASH_TYPE = "lut256"
TRAINING_ITERATIONS = 20
ANISOTROPIC_THRESHOLD = 0.2
PERSISTENT_BYTES = 128
BOOTSTRAPS = 10_000
MINIMUM_MAP_GAIN = 0.002
MAXIMUM_SOURCE_MAP_GAP = 0.003
MAXIMUM_SOURCE_R1_GAP = 0.001
SCANN_VERSION = "1.4.2"
SCANN_EXTENSION_SHA256 = "6bf70cff2ac0d7646f91664cab0292117490ada30a6dddf8afaefa920ee79725"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bootstrap(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, object]:
    if (
        candidate.shape != baseline.shape
        or candidate.ndim != 1
        or candidate.size != 60_502
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
    ):
        raise ValueError("ScaNN bootstrap authority differs")
    differences = candidate.astype(np.float64) - baseline.astype(np.float64)
    generator = np.random.Generator(np.random.PCG64(20_260_920))
    means = np.empty(BOOTSTRAPS, dtype=np.float64)
    for start in range(0, BOOTSTRAPS, 50):
        stop = min(start + 50, BOOTSTRAPS)
        indexes = generator.integers(0, len(differences), size=(stop - start, len(differences)))
        means[start:stop] = differences[indexes].mean(axis=1, dtype=np.float64)
    return {
        "delta": float(differences.mean(dtype=np.float64)),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
    }


def build(gallery: np.ndarray, threshold: float) -> tuple[object, float]:
    started = time.perf_counter()
    builder = scann.scann_ops_pybind.builder(gallery, 16, "dot_product")
    builder.set_n_training_threads(1)
    searcher = builder.score_ah(
        BLOCK_DIMENSIONS,
        anisotropic_quantization_threshold=threshold,
        training_sample_size=len(gallery),
        hash_type=HASH_TYPE,
        training_iterations=TRAINING_ITERATIONS,
    ).build()
    return searcher, time.perf_counter() - started


def score(
    searcher: object,
    queries: np.ndarray,
    labels: np.ndarray,
) -> tuple[dict[str, object], float]:
    counts = Counter(int(value) for value in labels.tolist())
    maximum_r = max(counts.values()) - 1
    started = time.perf_counter()
    indexes, _scores = searcher.search_batched(
        queries, final_num_neighbors=maximum_r + 1
    )
    elapsed = time.perf_counter() - started
    if indexes.shape != (len(queries), maximum_r + 1):
        raise ValueError("ScaNN ranking shape differs")
    per_query_ap = np.empty(len(queries), dtype=np.float64)
    per_query_r1 = np.empty(len(queries), dtype=np.float64)
    for query_index, raw in enumerate(indexes):
        ranking = raw[raw != query_index]
        relevant = counts[int(labels[query_index])] - 1
        if relevant < 1 or len(ranking) < relevant:
            raise ValueError("ScaNN leave-one-out authority differs")
        matches = labels[ranking[:relevant]] == labels[query_index]
        precision = np.cumsum(matches, dtype=np.float64) / np.arange(1, relevant + 1)
        per_query_ap[query_index] = float((precision * matches).sum() / relevant)
        per_query_r1[query_index] = float(matches[0])
    return {
        "map_at_r": float(per_query_ap.mean(dtype=np.float64)),
        "recall_at_1": float(per_query_r1.mean(dtype=np.float64)),
        "per_query_ap": per_query_ap,
        "per_query_r1": per_query_r1,
    }, elapsed


def public_summary(score_value: dict[str, object]) -> dict[str, float]:
    return {
        "map_at_r": float(score_value["map_at_r"]),
        "recall_at_1": float(score_value["recall_at_1"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--faiss-result", type=Path, required=True)
    parser.add_argument("--faiss-result-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-scann-gate", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    for path, expected in (
        (args.features, args.features_sha256),
        (args.faiss_result, args.faiss_result_sha256),
        (args.preregistration, args.preregistration_sha256),
        (Path(__file__), args.script_sha256),
    ):
        if sha256_file(path) != expected:
            raise ValueError("ScaNN gate authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    expected = {
        "faiss_result_sha256": args.faiss_result_sha256,
        "features_sha256": args.features_sha256,
        "script_sha256": args.script_sha256,
        "source_commit": args.source_commit,
    }
    if any(preregistration.get(key) != value for key, value in expected.items()):
        raise ValueError("ScaNN preregistration binding differs")
    faiss_result = json.loads(args.faiss_result.read_text())
    if faiss_result.get("faiss_version") != "1.12.0":
        raise ValueError("ScaNN Faiss control differs")
    opq = faiss_result["arms"]["OPQ128_384,PQ128x8"]["asymmetric"]

    with np.load(args.features, allow_pickle=False) as archive:
        gallery = np.ascontiguousarray(archive["test_features"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["test_labels"], dtype=np.int64)
        source_ap = np.ascontiguousarray(archive["test_ap_at_r"], dtype=np.float64)
        source_r1 = np.ascontiguousarray(archive["test_recall_at_1"], dtype=np.float64)
    if gallery.shape != (60_502, 384) or labels.shape != (60_502,):
        raise ValueError("ScaNN feature shape differs")
    gallery /= np.linalg.norm(gallery, axis=1, keepdims=True)
    gallery = np.ascontiguousarray(gallery)
    source = {
        "map_at_r": float(source_ap.mean(dtype=np.float64)),
        "recall_at_1": float(source_r1.mean(dtype=np.float64)),
        "per_query_ap": source_ap,
        "per_query_r1": source_r1,
    }
    arms: dict[str, object] = {}
    private_scores: dict[str, dict[str, object]] = {}
    for name, threshold in (("isotropic", math.nan), ("anisotropic", ANISOTROPIC_THRESHOLD)):
        searcher, build_seconds = build(gallery, threshold)
        score_value, search_seconds = score(searcher, gallery, labels)
        private_scores[name] = score_value
        arms[name] = {
            **public_summary(score_value),
            "build_seconds": build_seconds,
            "search_seconds": search_seconds,
        }
        print(json.dumps({"completed": name, **arms[name]}, sort_keys=True), flush=True)
        del searcher
    anisotropic = private_scores["anisotropic"]
    isotropic = private_scores["isotropic"]
    versus_isotropic = {
        "map_at_r": bootstrap(anisotropic["per_query_ap"], isotropic["per_query_ap"]),
        "recall_at_1": bootstrap(anisotropic["per_query_r1"], isotropic["per_query_r1"]),
    }
    versus_source = {
        "map_at_r": bootstrap(anisotropic["per_query_ap"], source["per_query_ap"]),
        "recall_at_1": bootstrap(anisotropic["per_query_r1"], source["per_query_r1"]),
    }
    passed = bool(
        float(versus_isotropic["map_at_r"]["delta"]) >= MINIMUM_MAP_GAIN
        and float(versus_isotropic["map_at_r"]["ci95"][0]) > 0.0
        and float(versus_isotropic["recall_at_1"]["delta"]) >= 0.0
        and float(anisotropic["map_at_r"]) > float(opq["map_at_r"])
        and float(anisotropic["recall_at_1"]) > float(opq["recall_at_1"])
        and float(source["map_at_r"]) - float(anisotropic["map_at_r"])
        <= MAXIMUM_SOURCE_MAP_GAP
        and float(versus_source["map_at_r"]["ci95"][0]) >= -MAXIMUM_SOURCE_MAP_GAP
        and float(source["recall_at_1"]) - float(anisotropic["recall_at_1"])
        <= MAXIMUM_SOURCE_R1_GAP
    )
    result = {
        "schema": "scratch-oml-sop-scann-anisotropic-128byte-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "dataset": "Stanford Online Products",
        "split": "official leave-one-out test gallery; unsupervised gallery-fit index",
        "scann": {
            "version": SCANN_VERSION,
            "extension_sha256": SCANN_EXTENSION_SHA256,
            "hash_type": HASH_TYPE,
            "dimensions_per_block": BLOCK_DIMENSIONS,
            "blocks": gallery.shape[1] // BLOCK_DIMENSIONS,
            "training_iterations": TRAINING_ITERATIONS,
            "persistent_bytes_per_item": PERSISTENT_BYTES,
            "training_threads": 1,
        },
        "source_float384": public_summary(source),
        "faiss_opq128x8_asymmetric": opq,
        "arms": arms,
        "anisotropic_minus_isotropic": versus_isotropic,
        "anisotropic_minus_source": versus_source,
        "gate": {
            "minimum_map_gain": MINIMUM_MAP_GAIN,
            "maximum_source_map_gap": MAXIMUM_SOURCE_MAP_GAP,
            "maximum_source_recall_at_1_gap": MAXIMUM_SOURCE_R1_GAP,
            "must_beat_faiss_opq_in_both_metrics": True,
            "passed": passed,
            "next": "replicate_unchanged_on_oml_inshop" if passed else "kill_exact_family",
        },
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="", flush=True)


if __name__ == "__main__":
    main()
