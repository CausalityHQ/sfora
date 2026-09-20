#!/usr/bin/env python3
"""Throwaway exact-inner-product second-moment OPQ64 precoder falsifier."""

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


def second_moment_pair(
    gallery: np.ndarray, queries: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    if (
        gallery.dtype != np.float32
        or queries.dtype != np.float32
        or gallery.ndim != 2
        or queries.ndim != 2
        or gallery.shape[0] < gallery.shape[1]
        or queries.shape[1] != gallery.shape[1]
        or not np.isfinite(gallery).all()
        or not np.isfinite(queries).all()
    ):
        raise ValueError("second-moment precoder authority differs")
    covariance = gallery.astype(np.float64).T @ gallery.astype(np.float64) / len(gallery)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    floor = float(eigenvalues[-1] * 1e-6)
    floored = np.maximum(eigenvalues, floor)
    root = eigenvectors * np.sqrt(floored)[None, :]
    inverse_root = eigenvectors * (1.0 / np.sqrt(floored))[None, :]
    encoded = np.ascontiguousarray(gallery.astype(np.float64) @ root, dtype=np.float32)
    transformed_queries = np.ascontiguousarray(
        queries.astype(np.float64) @ inverse_root, dtype=np.float32
    )
    sample_rows = min(len(gallery), 256)
    sample_queries = min(len(queries), 64)
    exact = queries[:sample_queries].astype(np.float64) @ gallery[:sample_rows].astype(np.float64).T
    observed = (
        transformed_queries[:sample_queries].astype(np.float64)
        @ encoded[:sample_rows].astype(np.float64).T
    )
    diagnostics = {
        "eigenvalue_min": float(eigenvalues[0]),
        "eigenvalue_max": float(eigenvalues[-1]),
        "eigenvalue_floor": floor,
        "floored_eigenvalues": int((eigenvalues < floor).sum()),
        "sample_inner_product_max_abs_error": float(np.max(np.abs(exact - observed))),
        "stored_precoder_bytes": int(
            eigenvectors.astype(np.float32).nbytes
            + floored.astype(np.float32).nbytes
        ),
    }
    return encoded, transformed_queries, diagnostics


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
        raise ValueError("second-moment ranking shape differs")
    counts = Counter(int(label) for label in gallery_labels.tolist())
    matches = gallery_labels[rankings] == query_labels[:, None]
    relevant = np.asarray(
        [min(counts[int(label)], 5) for label in query_labels], dtype=np.float64
    )
    if bool((relevant < 1).any()):
        raise ValueError("second-moment query lacks gallery positive")
    return matches.sum(axis=1, dtype=np.float64) / relevant, matches[:, 0].astype(np.float64)


def summarize(rows: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    return {
        "mmp_at_5": float(rows[0].mean(dtype=np.float64)),
        "recall_at_1": float(rows[1].mean(dtype=np.float64)),
    }


def paired_interval(values: np.ndarray, *, seed: int) -> tuple[float, float, float]:
    deltas = np.asarray(values, dtype=np.float64)
    if deltas.ndim != 1 or len(deltas) < 2 or not np.isfinite(deltas).all():
        raise ValueError("second-moment paired bootstrap authority differs")
    rng = np.random.default_rng(seed)
    replicates = np.empty(10_000, dtype=np.float64)
    for start in range(0, len(replicates), 256):
        stop = min(start + 256, len(replicates))
        indexes = rng.integers(0, len(deltas), size=(stop - start, len(deltas)))
        replicates[start:stop] = deltas[indexes].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(replicates, (0.025, 0.975))
    return float(lower), float(deltas.mean(dtype=np.float64)), float(upper)


def top5(faiss: object, gallery: np.ndarray, queries: np.ndarray, *, metric: str) -> np.ndarray:
    if metric == "l2":
        index = faiss.IndexFlatL2(gallery.shape[1])
    elif metric == "ip":
        index = faiss.IndexFlatIP(gallery.shape[1])
    else:
        raise ValueError("second-moment score metric differs")
    index.add(np.ascontiguousarray(gallery, dtype=np.float32))
    _, ranking = index.search(np.ascontiguousarray(queries, dtype=np.float32), 5)
    return ranking


def self_test() -> None:
    values = np.asarray(
        [[1.0, 2.0], [2.0, -1.0], [-1.0, 0.5], [0.5, 0.25]], dtype=np.float32
    )
    queries = np.asarray([[0.25, -0.75], [1.0, 1.0]], dtype=np.float32)
    encoded, transformed_queries, _ = second_moment_pair(values, queries)
    assert np.allclose(transformed_queries @ encoded.T, queries @ values.T, atol=1e-5)
    constant = np.full(11, 0.125, dtype=np.float64)
    assert paired_interval(constant, seed=3) == (0.125, 0.125, 0.125)
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
        raise ValueError("second-moment scientific arguments required")
    if (
        args.features_sha256 != FEATURES_SHA256
        or sha256_file(args.features) != FEATURES_SHA256
        or args.output.exists()
    ):
        raise ValueError("second-moment scientific authority differs")
    faiss = importlib.import_module("faiss")
    with np.load(args.features, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["train_labels"], dtype=np.int64)
        paths = np.ascontiguousarray(archive["train_paths"])
        official = np.ascontiguousarray(archive["val_embeddings"], dtype=np.float32)
        official_labels = np.ascontiguousarray(archive["val_labels"], dtype=np.int64)
    held_out = pseudoquery_mask(labels, paths)
    if train.shape != (38_307, 768) or official.shape != (129, 768) or held_out.sum() != 3_050:
        raise ValueError("second-moment scientific population differs")
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    official /= np.linalg.norm(official, axis=1, keepdims=True)
    gallery = np.ascontiguousarray(train[~held_out])
    gallery_labels = np.ascontiguousarray(labels[~held_out])
    pseudo = np.ascontiguousarray(train[held_out])
    pseudo_labels = np.ascontiguousarray(labels[held_out])

    faiss.omp_set_num_threads(1)
    baseline_started = time.monotonic()
    baseline = faiss.index_factory(768, SPEC)
    baseline.train(gallery)
    baseline.add(gallery)
    baseline_fit_seconds = time.monotonic() - baseline_started
    _, baseline_pseudo_rank = baseline.search(pseudo, 5)
    _, baseline_official_rank = baseline.search(official, 5)
    baseline_codes = np.ascontiguousarray(baseline.sa_encode(gallery), dtype=np.uint8)
    baseline_decoded = np.ascontiguousarray(baseline.sa_decode(baseline_codes), dtype=np.float32)

    transformed_gallery, transformed_pseudo, diagnostics = second_moment_pair(gallery, pseudo)
    _, transformed_official, official_diagnostics = second_moment_pair(gallery, official)
    if diagnostics != official_diagnostics:
        differing = {
            key
            for key in diagnostics
            if key != "sample_inner_product_max_abs_error"
            and diagnostics[key] != official_diagnostics[key]
        }
        if differing:
            raise ValueError("second-moment transform authority differs")
    float_rank = top5(faiss, gallery, pseudo, metric="ip")
    transformed_float_rank = top5(
        faiss, transformed_gallery, transformed_pseudo, metric="ip"
    )
    if not np.array_equal(float_rank, transformed_float_rank):
        raise ValueError("second-moment exact inner-product ranking differs")

    candidate_started = time.monotonic()
    candidate = faiss.index_factory(768, SPEC)
    candidate.train(transformed_gallery)
    candidate_fit_seconds = time.monotonic() - candidate_started
    candidate_codes = np.ascontiguousarray(candidate.sa_encode(transformed_gallery), dtype=np.uint8)
    if baseline_codes.shape != (35_257, 64) or candidate_codes.shape != (35_257, 64):
        raise ValueError("second-moment payload width differs")
    candidate_decoded = np.ascontiguousarray(candidate.sa_decode(candidate_codes), dtype=np.float32)
    candidate_pseudo_rank = top5(
        faiss, candidate_decoded, transformed_pseudo, metric="ip"
    )
    candidate_official_rank = top5(
        faiss, candidate_decoded, transformed_official, metric="ip"
    )

    baseline_pseudo_rows = quality_rows(
        baseline_pseudo_rank, pseudo_labels, gallery_labels
    )
    candidate_pseudo_rows = quality_rows(
        candidate_pseudo_rank, pseudo_labels, gallery_labels
    )
    baseline_official_rows = quality_rows(
        baseline_official_rank, official_labels, gallery_labels
    )
    candidate_official_rows = quality_rows(
        candidate_official_rank, official_labels, gallery_labels
    )
    mmp_interval = paired_interval(
        candidate_pseudo_rows[0] - baseline_pseudo_rows[0], seed=20_260_922
    )
    r1_interval = paired_interval(
        candidate_pseudo_rows[1] - baseline_pseudo_rows[1], seed=20_260_923
    )
    payload = {
        "schema": "scratch-met-small-second-moment-precoder-v1",
        "claim_eligible": False,
        "features_sha256": FEATURES_SHA256,
        "faiss_version": faiss.__version__,
        "omp_threads": faiss.omp_get_max_threads(),
        "gallery_rows": len(gallery),
        "pseudo_queries": len(pseudo),
        "official_validation_queries": len(official),
        "payload_bytes_per_item": int(candidate_codes.shape[1]),
        "transform": diagnostics,
        "baseline": {
            "spec": SPEC,
            "fit_seconds": baseline_fit_seconds,
            "reconstruction_mse": float(
                np.mean((gallery - baseline_decoded) ** 2, dtype=np.float64)
            ),
            "pseudo": summarize(baseline_pseudo_rows),
            "official": summarize(baseline_official_rows),
        },
        "second_moment_precoded": {
            "spec": SPEC,
            "fit_seconds": candidate_fit_seconds,
            "transformed_reconstruction_mse": float(
                np.mean((transformed_gallery - candidate_decoded) ** 2, dtype=np.float64)
            ),
            "pseudo": summarize(candidate_pseudo_rows),
            "official": summarize(candidate_official_rows),
        },
        "paired_pseudo_delta_candidate_minus_baseline": {
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
