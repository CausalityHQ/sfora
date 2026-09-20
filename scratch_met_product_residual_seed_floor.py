#!/usr/bin/env python3
"""Throwaway three-seed stability check for the MET-small PRQ mechanism."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import time
from pathlib import Path

import numpy as np

from scratch_met_product_residual_codec import (
    BEAM,
    BITS,
    FEATURES_SHA256,
    OPQ_SPEC,
    SPLITS,
    pseudoquery_mask,
    quality_rows,
    sha256_file,
    summary,
)
from scratch_met_product_residual_diagnostic import exact_ranking, fidelity_rows

SEEDS = (1234, 20_260_920, 314_159)


def summarize_seed_rows(rows: list[dict[str, float]]) -> dict[str, float]:
    keys = tuple(rows[0])
    if not rows or any(tuple(row) != keys for row in rows):
        raise ValueError("product-residual seed summary differs")
    return {
        f"{key}_{stat}": float(function([row[key] for row in rows]))
        for key in keys
        for stat, function in (("min", min), ("mean", np.mean), ("max", max))
    }


def self_test() -> None:
    result = summarize_seed_rows([{"x": 1.0, "y": 4.0}, {"x": 3.0, "y": 2.0}])
    assert result == {
        "x_min": 1.0,
        "x_mean": 2.0,
        "x_max": 3.0,
        "y_min": 2.0,
        "y_mean": 3.0,
        "y_max": 4.0,
    }
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
        raise ValueError("product-residual seed-floor scientific arguments required")
    if (
        args.features_sha256 != FEATURES_SHA256
        or sha256_file(args.features) != FEATURES_SHA256
        or args.output.exists()
    ):
        raise ValueError("product-residual seed-floor authority differs")

    faiss = importlib.import_module("faiss")
    with np.load(args.features, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["train_labels"], dtype=np.int64)
        paths = np.ascontiguousarray(archive["train_paths"])
    held_out = pseudoquery_mask(labels, paths)
    if train.shape != (38_307, 768) or held_out.sum() != 3_050:
        raise ValueError("product-residual seed-floor population differs")
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    gallery = np.ascontiguousarray(train[~held_out])
    gallery_labels = np.ascontiguousarray(labels[~held_out])
    pseudo = np.ascontiguousarray(train[held_out])
    pseudo_labels = np.ascontiguousarray(labels[held_out])

    faiss.omp_set_num_threads(1)
    started = time.monotonic()
    opq = faiss.index_factory(768, OPQ_SPEC)
    opq.train(gallery)
    transform = faiss.downcast_VectorTransform(opq.chain.at(0))
    rotated_gallery = np.ascontiguousarray(transform.apply_py(gallery), dtype=np.float32)
    rotated_pseudo = np.ascontiguousarray(transform.apply_py(pseudo), dtype=np.float32)
    opq_codes = np.ascontiguousarray(opq.sa_encode(gallery), dtype=np.uint8)
    base_index = faiss.downcast_index(opq.index)
    opq_decoded = np.ascontiguousarray(base_index.sa_decode(opq_codes), dtype=np.float32)
    exact_top10 = exact_ranking(faiss, rotated_pseudo, rotated_gallery, metric="l2", k=10)

    baseline: dict[str, dict[str, float]] = {}
    for metric in ("l2", "ip"):
        ranking = exact_ranking(faiss, rotated_pseudo, opq_decoded, metric=metric, k=10)
        baseline[metric] = {
            **summary(quality_rows(ranking[:, :5], pseudo_labels, gallery_labels)),
            "exact_top1_match": float(fidelity_rows(exact_top10, ranking)[0].mean()),
            "exact_top10_overlap": float(fidelity_rows(exact_top10, ranking)[1].mean()),
        }
    seeds: list[dict[str, object]] = []
    delta_rows: list[dict[str, float]] = []
    for seed in SEEDS:
        prq = faiss.ProductResidualQuantizer(768, SPLITS, 2, BITS)
        for split in range(SPLITS):
            residual = faiss.downcast_AdditiveQuantizer(prq.subquantizer(split))
            residual.max_beam_size = BEAM
            residual.cp.seed = seed
        fit_started = time.monotonic()
        prq.train(rotated_gallery)
        fit_seconds = time.monotonic() - fit_started
        codes = np.ascontiguousarray(prq.compute_codes(rotated_gallery), dtype=np.uint8)
        decoded = np.ascontiguousarray(prq.decode(codes), dtype=np.float32)
        metrics: dict[str, dict[str, float]] = {}
        seed_delta: dict[str, float] = {}
        for metric in ("l2", "ip"):
            ranking = exact_ranking(faiss, rotated_pseudo, decoded, metric=metric, k=10)
            fidelity = fidelity_rows(exact_top10, ranking)
            metrics[metric] = {
                **summary(quality_rows(ranking[:, :5], pseudo_labels, gallery_labels)),
                "exact_top1_match": float(fidelity[0].mean()),
                "exact_top10_overlap": float(fidelity[1].mean()),
            }
            for key in ("mmp_at_5", "recall_at_1", "exact_top1_match", "exact_top10_overlap"):
                seed_delta[f"{metric}_{key}"] = metrics[metric][key] - baseline[metric][key]
        delta_rows.append(seed_delta)
        seeds.append(
            {
                "seed": seed,
                "fit_seconds": fit_seconds,
                "reconstruction_mse": float(
                    np.mean((rotated_gallery - decoded) ** 2, dtype=np.float64)
                ),
                "metrics": metrics,
                "delta_product_residual_minus_opq": seed_delta,
            }
        )

    delta_summary = summarize_seed_rows(delta_rows)
    payload = {
        "schema": "scratch-met-small-product-residual-seed-floor-v1",
        "claim_eligible": False,
        "features_sha256": FEATURES_SHA256,
        "faiss_version": faiss.__version__,
        "gallery_rows": len(gallery),
        "pseudo_queries": len(pseudo),
        "seeds": seeds,
        "opq_fixed_baseline": baseline,
        "delta_across_seeds": delta_summary,
        "fresh_replication_gate": bool(
            delta_summary["l2_exact_top10_overlap_min"] > 0.0
            and delta_summary["ip_exact_top10_overlap_min"] > 0.0
            and delta_summary["l2_mmp_at_5_min"] > 0.0
            and delta_summary["ip_mmp_at_5_min"] > 0.0
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
