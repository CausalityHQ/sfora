#!/usr/bin/env python3
"""Throwaway MET-small 64-byte product-residual codec falsifier."""

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
OPQ_SPEC = "OPQ64_768,PQ64x8"
SPLITS = 32
STAGES = 2
BITS = 8
BEAM = 16
BOOTSTRAP_REPLICATES = 10_000


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


def quality_rows(
    rankings: np.ndarray, query_labels: np.ndarray, gallery_labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    if rankings.shape != (len(query_labels), 5):
        raise ValueError("product-residual ranking shape differs")
    counts = Counter(int(label) for label in gallery_labels.tolist())
    matches = gallery_labels[rankings] == query_labels[:, None]
    relevant = np.asarray(
        [min(counts[int(label)], 5) for label in query_labels], dtype=np.float64
    )
    if bool((relevant < 1).any()):
        raise ValueError("product-residual query lacks gallery positive")
    return matches.sum(axis=1, dtype=np.float64) / relevant, matches[:, 0].astype(np.float64)


def summary(rows: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    return {
        "mmp_at_5": float(np.mean(rows[0], dtype=np.float64)),
        "recall_at_1": float(np.mean(rows[1], dtype=np.float64)),
    }


def paired_interval(
    deltas: np.ndarray, *, seed: int, replicates: int = BOOTSTRAP_REPLICATES
) -> tuple[float, float, float]:
    values = np.asarray(deltas, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("paired bootstrap authority differs")
    rng = np.random.default_rng(seed)
    means = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 256):
        stop = min(start + 256, replicates)
        indexes = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indexes].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(means, (0.025, 0.975))
    return float(lower), float(values.mean(dtype=np.float64)), float(upper)


def exact_l2_ranking(faiss: object, queries: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    index = faiss.IndexFlatL2(gallery.shape[1])
    index.add(np.ascontiguousarray(gallery, dtype=np.float32))
    _, ranking = index.search(np.ascontiguousarray(queries, dtype=np.float32), 5)
    return ranking


def self_test() -> None:
    constant = np.full(17, 0.25, dtype=np.float64)
    assert paired_interval(constant, seed=7) == (0.25, 0.25, 0.25)
    rankings = np.asarray([[0, 1, 2, 3, 4], [5, 4, 3, 2, 1]], dtype=np.int64)
    labels = np.asarray([7, 9, 7, 10, 11, 8], dtype=np.int64)
    rows = quality_rows(rankings, np.asarray([7, 8], dtype=np.int64), labels)
    assert summary(rows) == {"mmp_at_5": 1.0, "recall_at_1": 1.0}
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
        raise ValueError("product-residual scientific arguments required")
    if (
        args.features_sha256 != FEATURES_SHA256
        or sha256_file(args.features) != FEATURES_SHA256
        or args.output.exists()
    ):
        raise ValueError("product-residual authority differs")

    faiss = importlib.import_module("faiss")
    with np.load(args.features, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["train_labels"], dtype=np.int64)
        paths = np.ascontiguousarray(archive["train_paths"])
        official = np.ascontiguousarray(archive["val_embeddings"], dtype=np.float32)
        official_labels = np.ascontiguousarray(archive["val_labels"], dtype=np.int64)
    held_out = pseudoquery_mask(labels, paths)
    if train.shape != (38_307, 768) or official.shape != (129, 768) or held_out.sum() != 3_050:
        raise ValueError("product-residual population differs")
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    official /= np.linalg.norm(official, axis=1, keepdims=True)
    gallery = np.ascontiguousarray(train[~held_out])
    gallery_labels = np.ascontiguousarray(labels[~held_out])
    pseudo = np.ascontiguousarray(train[held_out])
    pseudo_labels = np.ascontiguousarray(labels[held_out])

    faiss.omp_set_num_threads(1)
    opq_started = time.monotonic()
    opq = faiss.index_factory(768, OPQ_SPEC)
    opq.train(gallery)
    opq.add(gallery)
    opq_fit_seconds = time.monotonic() - opq_started
    transform = faiss.downcast_VectorTransform(opq.chain.at(0))
    rotated_gallery = np.ascontiguousarray(transform.apply_py(gallery), dtype=np.float32)
    rotated_pseudo = np.ascontiguousarray(transform.apply_py(pseudo), dtype=np.float32)
    rotated_official = np.ascontiguousarray(transform.apply_py(official), dtype=np.float32)

    _, opq_pseudo_ranking = opq.search(pseudo, 5)
    _, opq_official_ranking = opq.search(official, 5)
    opq_codes = np.ascontiguousarray(opq.sa_encode(gallery), dtype=np.uint8)
    base_index = faiss.downcast_index(opq.index)
    opq_decoded = np.ascontiguousarray(base_index.sa_decode(opq_codes), dtype=np.float32)
    if opq_codes.shape != (35_257, 64):
        raise ValueError("product-residual OPQ code width differs")
    explicit_opq_ranking = exact_l2_ranking(faiss, rotated_pseudo, opq_decoded)
    if not np.array_equal(explicit_opq_ranking, opq_pseudo_ranking):
        raise ValueError("product-residual OPQ exact-score control differs")

    prq = faiss.ProductResidualQuantizer(768, SPLITS, STAGES, BITS)
    for split in range(SPLITS):
        residual = faiss.downcast_AdditiveQuantizer(prq.subquantizer(split))
        residual.max_beam_size = BEAM
    prq_started = time.monotonic()
    prq.train(rotated_gallery)
    prq_fit_seconds = time.monotonic() - prq_started
    prq_codes = np.ascontiguousarray(prq.compute_codes(rotated_gallery), dtype=np.uint8)
    if prq.code_size != 64 or prq_codes.shape != (35_257, 64):
        raise ValueError("product-residual payload width differs")
    prq_decoded = np.ascontiguousarray(prq.decode(prq_codes), dtype=np.float32)
    prq_pseudo_ranking = exact_l2_ranking(faiss, rotated_pseudo, prq_decoded)
    prq_official_ranking = exact_l2_ranking(faiss, rotated_official, prq_decoded)

    opq_pseudo_rows = quality_rows(opq_pseudo_ranking, pseudo_labels, gallery_labels)
    prq_pseudo_rows = quality_rows(prq_pseudo_ranking, pseudo_labels, gallery_labels)
    opq_official_rows = quality_rows(opq_official_ranking, official_labels, gallery_labels)
    prq_official_rows = quality_rows(prq_official_ranking, official_labels, gallery_labels)
    mmp_interval = paired_interval(prq_pseudo_rows[0] - opq_pseudo_rows[0], seed=20_260_920)
    r1_interval = paired_interval(prq_pseudo_rows[1] - opq_pseudo_rows[1], seed=20_260_921)
    codebook_bytes = int(np.asarray(faiss.vector_to_array(prq.codebooks)).nbytes)
    pair_norm_table_bytes = SPLITS * (1 << BITS) ** STAGES * 4
    opq_codebook_bytes = int(
        np.asarray(faiss.vector_to_array(base_index.pq.centroids)).nbytes
    )
    payload = {
        "schema": "scratch-met-small-product-residual-codec-v1",
        "claim_eligible": False,
        "features_sha256": FEATURES_SHA256,
        "faiss_version": faiss.__version__,
        "omp_threads": faiss.omp_get_max_threads(),
        "gallery_rows": len(gallery),
        "pseudo_queries": len(pseudo),
        "official_validation_queries": len(official),
        "payload_bytes_per_item": int(prq_codes.shape[1]),
        "product_residual": {
            "splits": SPLITS,
            "stages_per_split": STAGES,
            "bits_per_stage": BITS,
            "beam": BEAM,
            "fit_seconds": prq_fit_seconds,
            "codebook_bytes": codebook_bytes,
            "pair_norm_table_bytes": pair_norm_table_bytes,
            "reconstruction_mse": float(
                np.mean((rotated_gallery - prq_decoded) ** 2, dtype=np.float64)
            ),
            "pseudo": summary(prq_pseudo_rows),
            "official": summary(prq_official_rows),
        },
        "opq64x8": {
            "spec": OPQ_SPEC,
            "fit_seconds": opq_fit_seconds,
            "codebook_bytes": opq_codebook_bytes,
            "reconstruction_mse": float(
                np.mean((rotated_gallery - opq_decoded) ** 2, dtype=np.float64)
            ),
            "pseudo": summary(opq_pseudo_rows),
            "official": summary(opq_official_rows),
        },
        "paired_pseudo_delta_product_residual_minus_opq": {
            "mmp_at_5_lower_mean_upper_95": mmp_interval,
            "recall_at_1_lower_mean_upper_95": r1_interval,
        },
        "promotion_passed": bool(
            mmp_interval[1] >= 0.002
            and mmp_interval[0] > 0.0
            and r1_interval[0] > -0.001
        ),
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
