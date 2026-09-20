#!/usr/bin/env python3
"""Throwaway powered MET-small exact OPQ64x8 baseline measurement."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np

FEATURES_SHA256 = "0277f717e73416e9cb7d1a8d57c3db08e05aea20cf9803f1fcedbfc502b81105"
SPEC = "OPQ64_768,PQ64x8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def pseudoquery_mask(labels: np.ndarray, paths: np.ndarray) -> np.ndarray:
    mask = np.zeros(len(labels), dtype=np.bool_)
    for label in np.unique(labels):
        indexes = np.flatnonzero(labels == label)
        if len(indexes) >= 2:
            mask[indexes[np.argsort(paths[indexes], kind="stable")[0]]] = True
    return mask


def quality(
    rankings: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray
) -> dict[str, float]:
    if rankings.shape != (len(query_labels), 5):
        raise ValueError("powered OPQ64 ranking shape differs")
    counts = Counter(int(label) for label in gallery_labels.tolist())
    matches = gallery_labels[rankings] == query_labels[:, None]
    mmp = []
    r1 = []
    for row, label in zip(matches, query_labels, strict=True):
        relevant = min(counts[int(label)], 5)
        if relevant < 1:
            raise ValueError("powered OPQ64 query lacks gallery positive")
        mmp.append(float(row.sum()) / relevant)
        r1.append(float(row[0]))
    return {
        "mmp_at_5": float(np.mean(mmp, dtype=np.float64)),
        "recall_at_1": float(np.mean(r1, dtype=np.float64)),
    }


def self_test() -> None:
    rankings = np.asarray([[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]], dtype=np.int64)
    query_labels = np.asarray([7, 8], dtype=np.int64)
    gallery_labels = np.asarray([7, 9, 7, 10, 11, 8], dtype=np.int64)
    metric = quality(rankings, query_labels, gallery_labels)
    assert metric == {"mmp_at_5": 1.0, "recall_at_1": 1.0}
    print("SELF_TEST_OK")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--features-sha256")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if None in (args.features, args.features_sha256, args.output):
        raise ValueError("MET powered OPQ64 scientific arguments required")
    if (
        args.features_sha256 != FEATURES_SHA256
        or sha256_file(args.features) != FEATURES_SHA256
        or args.output.exists()
    ):
        raise ValueError("MET powered OPQ64 authority differs")
    faiss = importlib.import_module("faiss")

    with np.load(args.features, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["train_labels"], dtype=np.int64)
        paths = np.ascontiguousarray(archive["train_paths"])
        official = np.ascontiguousarray(archive["val_embeddings"], dtype=np.float32)
        official_labels = np.ascontiguousarray(archive["val_labels"], dtype=np.int64)
    held_out = pseudoquery_mask(labels, paths)
    if train.shape != (38_307, 768) or official.shape != (129, 768) or held_out.sum() != 3_050:
        raise ValueError("MET powered OPQ64 population differs")
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    official /= np.linalg.norm(official, axis=1, keepdims=True)
    gallery = np.ascontiguousarray(train[~held_out])
    gallery_labels = np.ascontiguousarray(labels[~held_out])
    pseudo = np.ascontiguousarray(train[held_out])
    pseudo_labels = np.ascontiguousarray(labels[held_out])

    faiss.omp_set_num_threads(1)
    float_index = faiss.IndexFlatL2(768)
    float_index.add(gallery)
    float_started = time.monotonic()
    _, pseudo_float_ranking = float_index.search(pseudo, 5)
    _, official_float_ranking = float_index.search(official, 5)
    float_seconds = time.monotonic() - float_started

    fit_started = time.monotonic()
    codec = faiss.index_factory(768, SPEC)
    if codec.sa_code_size() != 64:
        raise ValueError("MET powered OPQ64 code width differs")
    codec.train(gallery)
    fit_seconds = time.monotonic() - fit_started
    codes = codec.sa_encode(gallery)
    if codes.shape != (35_257, 64):
        raise ValueError("MET powered OPQ64 code shape differs")
    decoded = np.ascontiguousarray(codec.sa_decode(codes), dtype=np.float32)
    codec.add(gallery)
    score_started = time.monotonic()
    _, pseudo_opq_ranking = codec.search(pseudo, 5)
    _, official_opq_ranking = codec.search(official, 5)
    score_seconds = time.monotonic() - score_started

    float_pseudo = quality(pseudo_float_ranking, pseudo_labels, gallery_labels)
    opq_pseudo = quality(pseudo_opq_ranking, pseudo_labels, gallery_labels)
    float_official = quality(official_float_ranking, official_labels, gallery_labels)
    opq_official = quality(official_opq_ranking, official_labels, gallery_labels)
    pseudo_penalty = {
        key: float_pseudo[key] - opq_pseudo[key] for key in ("mmp_at_5", "recall_at_1")
    }
    payload = {
        "schema": "scratch-met-small-powered-opq64-baseline-v1",
        "claim_eligible": False,
        "features_sha256": FEATURES_SHA256,
        "fit_gallery_rows": len(gallery),
        "pseudo_queries": len(pseudo),
        "official_validation_queries": len(official),
        "faiss_version": faiss.__version__,
        "omp_threads": faiss.omp_get_max_threads(),
        "spec": SPEC,
        "code_bytes_per_item": int(codes.shape[1]),
        "fit_seconds": fit_seconds,
        "float_score_seconds": float_seconds,
        "opq_score_seconds": score_seconds,
        "reconstruction_mse": float(np.mean((gallery - decoded) ** 2, dtype=np.float64)),
        "float": {"pseudo": float_pseudo, "official": float_official},
        "opq64x8": {"pseudo": opq_pseudo, "official": opq_official},
        "pseudo_float_minus_opq": pseudo_penalty,
        "codec_near_ceiling": bool(
            pseudo_penalty["mmp_at_5"] <= 0.005
            and pseudo_penalty["recall_at_1"] <= 0.01
        ),
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
