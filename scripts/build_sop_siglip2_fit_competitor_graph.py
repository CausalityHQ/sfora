#!/usr/bin/env python3
"""Build a frozen fit-only competitor graph for the next SOP training gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features


def ten_negative_neighbors(
    scores: torch.Tensor, labels: torch.Tensor, query_ordinals: torch.Tensor
) -> torch.Tensor:
    """Select ten negative rows with lowest gallery ordinal winning score ties."""

    if scores.shape != (len(query_ordinals), len(labels)) or len(labels) < 12:
        raise ValueError("fit-only competitor graph score geometry differs")
    negative = scores.masked_fill(labels[None, :] == labels[query_ordinals, None], -torch.inf)
    cutoff = negative.topk(10, dim=1).values[:, -1]
    if not bool(torch.isfinite(cutoff).all()):
        raise ValueError("fit-only competitor graph lacks ten negatives")
    higher = negative > cutoff[:, None]
    needed_equal = 10 - higher.sum(dim=1)
    equal = negative == cutoff[:, None]
    selected = higher | (equal & (equal.cumsum(dim=1) <= needed_equal[:, None]))
    return torch.nonzero(selected, as_tuple=False)[:, 1].reshape(len(query_ordinals), 10)


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
        "fit-census",
        "native-library",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    parser.add_argument("--expected-fit-census-sha256", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not torch.cuda.is_available()
        or sha256(args.training_receipt) != args.expected_training_receipt_sha256
        or sha256(args.fit_census) != args.expected_fit_census_sha256
    ):
        raise ValueError("fit-only competitor graph authority differs")
    receipt = json.loads(args.training_receipt.read_text())
    census = json.loads(args.fit_census.read_text())
    if (
        any(
            sha256(path) != receipt[key]
            for path, key in (
                (args.source_archive, "source_archive_sha256"),
                (args.source_features, "source_features_sha256"),
                (args.native_library, "native_library_sha256"),
            )
        )
        or census.get("training_receipt_sha256") != args.expected_training_receipt_sha256
    ):
        raise ValueError("fit-only competitor graph inputs differ")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        all_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    fit_rows = np.asarray(
        deterministic_class_partition(
            tuple(map(int, all_labels)), fit_fraction=0.9, seed=179019
        ).fit_row_indexes,
        dtype=np.int64,
    )
    labels = all_labels[fit_rows]
    source = np.load(args.source_features, mmap_mode="r", allow_pickle=False)
    if source.shape != (len(all_labels), 1024) or len(fit_rows) != 53_700:
        raise ValueError("fit-only competitor graph source geometry differs")
    from train_sop_siglip2_compact import initialize_head_and_classifier

    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    fit_features = torch.from_numpy(np.asarray(source[fit_rows]).copy())
    head, classifier, pca_sha = initialize_head_and_classifier(
        fit_features, tuple(map(int, labels))
    )
    if pca_sha != receipt["source_pca_sha256"]:
        raise ValueError("fit-only competitor graph PCA differs")
    head_sha = hashlib.sha256(
        head.weight.detach().numpy().tobytes() + head.bias.detach().numpy().tobytes()
    ).hexdigest()
    classifier_sha = hashlib.sha256(classifier.detach().numpy().tobytes()).hexdigest()
    if (
        head_sha != receipt["initial_head_sha256"]
        or classifier_sha != receipt["initial_classifier_sha256"]
    ):
        raise ValueError("fit-only competitor graph initial head differs")
    frozen_values = F.normalize(compact_head_features(fit_features, head), dim=1)
    packed = pack_int8_unit_embeddings(frozen_values)
    device = torch.device("cuda:0")
    codes = packed.codes.to(device=device, dtype=torch.float32)
    inverse_norms = packed.inverse_norms.to(device=device, dtype=torch.float32)
    labels_gpu = torch.from_numpy(labels.copy()).to(device)
    selected = np.linspace(0, len(labels) - 1, 32, dtype=np.int64)
    selected_packed = PackedInt8Embeddings(
        packed.codes[selected].contiguous(), packed.inverse_norms[selected].contiguous()
    )
    with CutilePackedInt8Gallery.open_packed(args.native_library, packed) as native:
        native_rows, native_scores = native.search_packed(selected_packed)
    selected_scores = (
        (codes[selected] @ codes.T) * inverse_norms[selected, None] * inverse_norms[None, :]
    )
    selected_sorted = torch.argsort(selected_scores, dim=1, descending=True, stable=True)[:, :10]
    if not np.array_equal(selected_sorted.cpu().numpy(), native_rows) or not np.array_equal(
        selected_scores.gather(1, selected_sorted).cpu().numpy().view(np.uint32),
        native_scores.view(np.uint32),
    ):
        raise ValueError("fit-only competitor graph scores differ from native")

    directed: Counter[tuple[int, int]] = Counter()
    frozen_wrong = np.zeros(len(labels), dtype=np.bool_)
    for start in range(0, len(labels), 64):
        stop = min(start + 64, len(labels))
        query_ordinals = torch.arange(start, stop, device=device)
        scores = (
            (codes[start:stop] @ codes.T) * inverse_norms[start:stop, None] * inverse_norms[None, :]
        )
        own_excluded = scores.clone()
        own_excluded[torch.arange(len(query_ordinals), device=device), query_ordinals] = -torch.inf
        nearest = own_excluded.argmax(dim=1)
        frozen_wrong[start:stop] = (labels_gpu[nearest] != labels_gpu[query_ordinals]).cpu().numpy()
        neighbors = ten_negative_neighbors(scores, labels_gpu, query_ordinals).cpu().numpy()
        for offset, row in enumerate(neighbors):
            anchor_product = int(labels[start + offset])
            for competitor in {int(labels[index]) for index in row}:
                directed[(anchor_product, competitor)] += 1
    symmetric: Counter[tuple[int, int]] = Counter()
    for (product, competitor), count in directed.items():
        symmetric[tuple(sorted((product, competitor)))] += count
    adjacent: dict[int, list[tuple[int, int]]] = {int(product): [] for product in np.unique(labels)}
    for (left, right), weight in symmetric.items():
        adjacent[left].append((right, weight))
        adjacent[right].append((left, weight))
    graph = {
        str(product): [
            {"product": neighbor, "weight": weight}
            for neighbor, weight in sorted(items, key=lambda item: (-item[1], item[0]))[:4]
        ]
        for product, items in adjacent.items()
    }
    trained_errors = census["trained_arcface"]["errors"]
    graph_targets = {
        int(product): {int(entry["product"]) for entry in items} for product, items in graph.items()
    }
    covered = sum(
        int(all_labels[int(error["impostor_train_row"])])
        in graph_targets[int(all_labels[int(error["query_train_row"])])]
        for error in trained_errors
    )
    inherited = sum(frozen_wrong[int(error["query_fit_ordinal"])] for error in trained_errors)
    result = {
        "schema": "sfora-sop-siglip2-fit-competitor-graph-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN fit products only, seed 179019",
        "source_sha256": sha256(Path(__file__)),
        "training_receipt_sha256": args.expected_training_receipt_sha256,
        "fit_census_sha256": args.expected_fit_census_sha256,
        "initial_head_sha256": head_sha,
        "initial_classifier_sha256": classifier_sha,
        "source_pca_sha256": pca_sha,
        "fit_rows_sha256": hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest(),
        "fit_products": len(graph),
        "frozen_fit_errors": int(frozen_wrong.sum()),
        "trained_errors": len(trained_errors),
        "trained_error_frozen_inherited": int(inherited),
        "trained_error_frozen_top4_covered": int(covered),
        "trained_error_frozen_top4_coverage": covered / len(trained_errors),
        "products_with_at_least_one_neighbor": sum(bool(items) for items in graph.values()),
        "native_selected_top10_exact": True,
        "graph": graph,
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
                    "fit_products",
                    "frozen_fit_errors",
                    "trained_error_frozen_inherited",
                    "trained_error_frozen_top4_covered",
                    "trained_error_frozen_top4_coverage",
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
