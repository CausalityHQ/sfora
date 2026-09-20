#!/usr/bin/env python3
"""Throwaway mechanism diagnostic for the MET-small product-residual probe."""

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
    paired_interval,
    pseudoquery_mask,
    quality_rows,
    sha256_file,
    summary,
)

BOOTSTRAP_SEED = 20_260_922


def exact_ranking(
    faiss: object,
    queries: np.ndarray,
    gallery: np.ndarray,
    *,
    metric: str,
    k: int,
) -> np.ndarray:
    if metric == "l2":
        index = faiss.IndexFlatL2(gallery.shape[1])
    elif metric == "ip":
        index = faiss.IndexFlatIP(gallery.shape[1])
    else:
        raise ValueError("product-residual diagnostic metric differs")
    index.add(np.ascontiguousarray(gallery, dtype=np.float32))
    _, ranking = index.search(np.ascontiguousarray(queries, dtype=np.float32), k)
    return np.asarray(ranking, dtype=np.int64)


def fidelity_rows(reference: np.ndarray, candidate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if reference.shape != candidate.shape or reference.ndim != 2 or reference.shape[1] != 10:
        raise ValueError("product-residual fidelity shape differs")
    top1 = (reference[:, 0] == candidate[:, 0]).astype(np.float64)
    overlap = np.fromiter(
        (
            len(set(reference[row].tolist()).intersection(candidate[row].tolist())) / 10.0
            for row in range(len(reference))
        ),
        dtype=np.float64,
        count=len(reference),
    )
    return top1, overlap


def error_decomposition(
    queries: np.ndarray,
    exact_top10: np.ndarray,
    gallery: np.ndarray,
    decoded: np.ndarray,
) -> dict[str, float]:
    directional: list[np.ndarray] = []
    norm_delta: list[np.ndarray] = []
    for start in range(0, len(queries), 256):
        stop = min(start + 256, len(queries))
        neighbors = exact_top10[start:stop]
        truth = gallery[neighbors]
        reconstruction = decoded[neighbors]
        error = reconstruction - truth
        directional.append(
            np.einsum("bd,bkd->bk", queries[start:stop], error, dtype=np.float64).reshape(-1)
        )
        norm_delta.append(
            (
                np.sum(reconstruction * reconstruction, axis=2, dtype=np.float64)
                - np.sum(truth * truth, axis=2, dtype=np.float64)
            ).reshape(-1)
        )
    direction = np.concatenate(directional)
    norm = np.concatenate(norm_delta)
    return {
        "directional_mean": float(direction.mean(dtype=np.float64)),
        "directional_standard_deviation": float(direction.std(dtype=np.float64)),
        "directional_rms": float(np.sqrt(np.mean(direction * direction, dtype=np.float64))),
        "norm_delta_mean": float(norm.mean(dtype=np.float64)),
        "norm_delta_standard_deviation": float(norm.std(dtype=np.float64)),
        "norm_delta_rms": float(np.sqrt(np.mean(norm * norm, dtype=np.float64))),
    }


def self_test() -> None:
    reference = np.asarray([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]], dtype=np.int64)
    candidate = np.asarray([[0, 1, 12, 13, 14, 15, 16, 17, 18, 19]], dtype=np.int64)
    top1, overlap = fidelity_rows(reference, candidate)
    assert top1.tolist() == [1.0]
    assert overlap.tolist() == [0.2]
    queries = np.asarray([[1.0, 0.0]], dtype=np.float32)
    gallery = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    decoded = np.asarray([[0.5, 0.0], [0.0, 2.0]], dtype=np.float32)
    decomposition = error_decomposition(
        queries,
        np.asarray([[0, 1, 0, 1, 0, 1, 0, 1, 0, 1]], dtype=np.int64),
        gallery,
        decoded,
    )
    assert decomposition["directional_mean"] == -0.25
    assert decomposition["norm_delta_mean"] == 1.125
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
        raise ValueError("product-residual diagnostic scientific arguments required")
    if (
        args.features_sha256 != FEATURES_SHA256
        or sha256_file(args.features) != FEATURES_SHA256
        or args.output.exists()
    ):
        raise ValueError("product-residual diagnostic authority differs")

    faiss = importlib.import_module("faiss")
    with np.load(args.features, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["train_labels"], dtype=np.int64)
        paths = np.ascontiguousarray(archive["train_paths"])
    held_out = pseudoquery_mask(labels, paths)
    if train.shape != (38_307, 768) or held_out.sum() != 3_050:
        raise ValueError("product-residual diagnostic population differs")
    train /= np.linalg.norm(train, axis=1, keepdims=True)
    gallery = np.ascontiguousarray(train[~held_out])
    gallery_labels = np.ascontiguousarray(labels[~held_out])
    pseudo = np.ascontiguousarray(train[held_out])
    pseudo_labels = np.ascontiguousarray(labels[held_out])

    faiss.omp_set_num_threads(1)
    started = time.monotonic()
    opq = faiss.index_factory(768, OPQ_SPEC)
    opq.train(gallery)
    opq.add(gallery)
    transform = faiss.downcast_VectorTransform(opq.chain.at(0))
    rotated_gallery = np.ascontiguousarray(transform.apply_py(gallery), dtype=np.float32)
    rotated_pseudo = np.ascontiguousarray(transform.apply_py(pseudo), dtype=np.float32)
    opq_codes = np.ascontiguousarray(opq.sa_encode(gallery), dtype=np.uint8)
    base_index = faiss.downcast_index(opq.index)
    opq_decoded = np.ascontiguousarray(base_index.sa_decode(opq_codes), dtype=np.float32)

    prq = faiss.ProductResidualQuantizer(768, SPLITS, 2, BITS)
    for split in range(SPLITS):
        residual = faiss.downcast_AdditiveQuantizer(prq.subquantizer(split))
        residual.max_beam_size = BEAM
    prq.train(rotated_gallery)
    prq_codes = np.ascontiguousarray(prq.compute_codes(rotated_gallery), dtype=np.uint8)
    prq_decoded = np.ascontiguousarray(prq.decode(prq_codes), dtype=np.float32)

    exact_top10 = exact_ranking(faiss, rotated_pseudo, rotated_gallery, metric="l2", k=10)
    rankings: dict[str, np.ndarray] = {}
    quality: dict[str, dict[str, float]] = {}
    fidelity: dict[str, dict[str, float]] = {}
    fidelity_rows_by_arm: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for codec, decoded in (("opq", opq_decoded), ("product_residual", prq_decoded)):
        for metric in ("l2", "ip"):
            arm = f"{codec}_{metric}"
            ranking10 = exact_ranking(faiss, rotated_pseudo, decoded, metric=metric, k=10)
            rankings[arm] = ranking10
            quality[arm] = summary(quality_rows(ranking10[:, :5], pseudo_labels, gallery_labels))
            rows = fidelity_rows(exact_top10, ranking10)
            fidelity_rows_by_arm[arm] = rows
            fidelity[arm] = {
                "exact_top1_match": float(rows[0].mean(dtype=np.float64)),
                "exact_top10_overlap": float(rows[1].mean(dtype=np.float64)),
            }

    fidelity_delta = {}
    for metric in ("l2", "ip"):
        opq_rows = fidelity_rows_by_arm[f"opq_{metric}"]
        prq_rows = fidelity_rows_by_arm[f"product_residual_{metric}"]
        fidelity_delta[metric] = {
            "exact_top1_match_lower_mean_upper_95": paired_interval(
                prq_rows[0] - opq_rows[0], seed=BOOTSTRAP_SEED + (0 if metric == "l2" else 2)
            ),
            "exact_top10_overlap_lower_mean_upper_95": paired_interval(
                prq_rows[1] - opq_rows[1], seed=BOOTSTRAP_SEED + (1 if metric == "l2" else 3)
            ),
        }

    payload = {
        "schema": "scratch-met-small-product-residual-diagnostic-v1",
        "claim_eligible": False,
        "features_sha256": FEATURES_SHA256,
        "faiss_version": faiss.__version__,
        "gallery_rows": len(gallery),
        "pseudo_queries": len(pseudo),
        "elapsed_seconds": time.monotonic() - started,
        "quality": quality,
        "float_rank_fidelity": fidelity,
        "paired_product_residual_minus_opq_fidelity": fidelity_delta,
        "exact_top10_error": {
            "opq": error_decomposition(rotated_pseudo, exact_top10, rotated_gallery, opq_decoded),
            "product_residual": error_decomposition(
                rotated_pseudo, exact_top10, rotated_gallery, prq_decoded
            ),
        },
        "interpretation_gate": {
            "product_residual_l2_top10_overlap_lower_above_zero": bool(
                fidelity_delta["l2"]["exact_top10_overlap_lower_mean_upper_95"][0] > 0.0
            ),
            "product_residual_label_gain_survives_ip": bool(
                quality["product_residual_ip"]["mmp_at_5"] > quality["opq_ip"]["mmp_at_5"]
            ),
        },
    }
    wire = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
