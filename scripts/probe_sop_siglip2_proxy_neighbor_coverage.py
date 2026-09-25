#!/usr/bin/env python3
"""Fit-only proxy-neighbor coverage preflight for SOP SigLIP2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features


@torch.inference_mode()
def proxy_top_four(classifier: torch.Tensor) -> torch.Tensor:
    """Return four closest normalized class proxies with ordinal tie breaking."""

    if classifier.ndim != 2 or classifier.shape[0] < 6:
        raise ValueError("fit-only proxy graph geometry differs")
    unit = F.normalize(classifier.float(), dim=1)
    neighbors = []
    for start in range(0, len(unit), 256):
        stop = min(start + 256, len(unit))
        scores = unit[start:stop] @ unit.T
        scores[
            torch.arange(stop - start, device=unit.device),
            torch.arange(start, stop, device=unit.device),
        ] = -torch.inf
        neighbors.append(torch.argsort(scores, dim=1, descending=True, stable=True)[:, :4])
    return torch.cat(neighbors)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "source-archive",
        "source-features",
        "training-receipt",
        "training-checkpoint",
        "train-embeddings",
        "fit-census",
        "frozen-member-graph",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--expected-fit-census-sha256", required=True)
    parser.add_argument("--expected-frozen-member-graph-sha256", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not torch.cuda.is_available()
        or sha256(args.training_receipt) != args.expected_training_receipt_sha256
        or sha256(args.fit_census) != args.expected_fit_census_sha256
        or sha256(args.frozen_member_graph) != args.expected_frozen_member_graph_sha256
    ):
        raise ValueError("fit-only proxy-neighbor authority differs")
    receipt = json.loads(args.training_receipt.read_text())
    census = json.loads(args.fit_census.read_text())
    frozen_graph = json.loads(args.frozen_member_graph.read_text())
    if (
        any(
            sha256(path) != receipt[key]
            for path, key in (
                (args.source_archive, "source_archive_sha256"),
                (args.source_features, "source_features_sha256"),
                (args.training_checkpoint, "checkpoint_sha256"),
                (args.train_embeddings, "train_embeddings_sha256"),
            )
        )
        or frozen_graph.get("fit_census_sha256") != args.expected_fit_census_sha256
    ):
        raise ValueError("fit-only proxy-neighbor artifacts differ")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        all_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    fit_rows = np.asarray(
        deterministic_class_partition(
            tuple(map(int, all_labels)), fit_fraction=0.9, seed=179019
        ).fit_row_indexes,
        dtype=np.int64,
    )
    labels = all_labels[fit_rows]
    products, class_indexes = np.unique(labels, return_inverse=True)
    if len(fit_rows) != 53_700 or len(products) != 10_186:
        raise ValueError("fit-only proxy-neighbor partition differs")
    source = np.load(args.source_features, mmap_mode="r", allow_pickle=False)
    all_values = np.load(args.train_embeddings, mmap_mode="r", allow_pickle=False)
    if source.shape != (len(all_labels), 1024) or all_values.shape != (len(all_labels), 128):
        raise ValueError("fit-only proxy-neighbor embedding geometry differs")
    from train_sop_siglip2_compact import initialize_head_and_classifier

    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    fit_source = torch.from_numpy(np.asarray(source[fit_rows]).copy())
    head, initial_classifier, pca_sha = initialize_head_and_classifier(
        fit_source, tuple(map(int, labels))
    )
    head_sha = hashlib.sha256(
        head.weight.detach().numpy().tobytes() + head.bias.detach().numpy().tobytes()
    ).hexdigest()
    if pca_sha != receipt["source_pca_sha256"] or head_sha != receipt["initial_head_sha256"]:
        raise ValueError("fit-only proxy-neighbor initial head differs")
    initial_classifier_sha = hashlib.sha256(
        initial_classifier.detach().numpy().tobytes()
    ).hexdigest()
    if initial_classifier_sha != receipt["initial_classifier_sha256"]:
        raise ValueError("fit-only proxy-neighbor initial classifier differs")
    frozen_values = F.normalize(compact_head_features(fit_source, head), dim=1)
    packed = pack_int8_unit_embeddings(frozen_values)
    checkpoint = torch.load(args.training_checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("arm") != "arcface"
        or checkpoint["classifier"].shape != initial_classifier.shape
    ):
        raise ValueError("fit-only proxy-neighbor final classifier differs")
    device = torch.device("cuda:0")
    initial_neighbors = proxy_top_four(initial_classifier.detach().to(device)).cpu().numpy()
    final_neighbors = proxy_top_four(checkpoint["classifier"].to(device)).cpu().numpy()

    codes = packed.codes.to(device=device, dtype=torch.float32)
    inverse_norms = packed.inverse_norms.to(device=device, dtype=torch.float32)
    labels_gpu = torch.from_numpy(labels.copy()).to(device)
    frozen_query_errors: list[int] = []
    frozen_impostors: list[int] = []
    for start in range(0, len(labels), 64):
        stop = min(start + 64, len(labels))
        query_ordinals = torch.arange(start, stop, device=device)
        scores = (
            (codes[start:stop] @ codes.T) * inverse_norms[start:stop, None] * inverse_norms[None, :]
        )
        scores[torch.arange(stop - start, device=device), query_ordinals] = -torch.inf
        nearest = scores.argmax(dim=1)
        wrong = labels_gpu[nearest] != labels_gpu[query_ordinals]
        frozen_query_errors.extend(map(int, query_ordinals[wrong].cpu()))
        frozen_impostors.extend(map(int, nearest[wrong].cpu()))
    if len(frozen_query_errors) != frozen_graph["frozen_fit_errors"]:
        raise ValueError("fit-only proxy-neighbor frozen error inventory differs")
    trained_errors = census["trained_arcface"]["errors"]
    trained_queries = np.asarray([int(error["query_fit_ordinal"]) for error in trained_errors])
    trained_impostors = np.asarray([int(error["impostor_fit_ordinal"]) for error in trained_errors])
    frozen_queries = np.asarray(frozen_query_errors)
    frozen_impostor_rows = np.asarray(frozen_impostors)

    def coverage(graph: np.ndarray, queries: np.ndarray, impostors: np.ndarray) -> int:
        return int(
            np.any(graph[class_indexes[queries]] == class_indexes[impostors, None], axis=1).sum()
        )

    initial_own_covered = coverage(initial_neighbors, frozen_queries, frozen_impostor_rows)
    final_trained_covered = coverage(final_neighbors, trained_queries, trained_impostors)
    result = {
        "schema": "sfora-sop-siglip2-fit-proxy-neighbor-coverage-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN fit products only, seed 179019",
        "source_sha256": sha256(Path(__file__)),
        "training_receipt_sha256": args.expected_training_receipt_sha256,
        "fit_census_sha256": args.expected_fit_census_sha256,
        "frozen_member_graph_sha256": args.expected_frozen_member_graph_sha256,
        "source_pca_sha256": pca_sha,
        "initial_classifier_sha256": initial_classifier_sha,
        "final_classifier_sha256": hashlib.sha256(
            checkpoint["classifier"].detach().numpy().tobytes()
        ).hexdigest(),
        "fit_rows_sha256": hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest(),
        "initial_errors": len(frozen_queries),
        "initial_own_error_covered": initial_own_covered,
        "initial_own_error_coverage": initial_own_covered / len(frozen_queries),
        "trained_errors": len(trained_queries),
        "final_trained_error_covered": final_trained_covered,
        "final_trained_error_coverage": final_trained_covered / len(trained_queries),
        "initial_graph_trained_error_coverage": coverage(
            initial_neighbors, trained_queries, trained_impostors
        )
        / len(trained_queries),
        "initial_graph": initial_neighbors.tolist(),
        "final_graph": final_neighbors.tolist(),
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "initial_errors",
                    "initial_own_error_coverage",
                    "trained_errors",
                    "final_trained_error_coverage",
                    "initial_graph_trained_error_coverage",
                    "wall_seconds",
                    "peak_cuda_allocated_bytes",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
