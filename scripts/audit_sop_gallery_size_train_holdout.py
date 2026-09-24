"""Measure the gallery-size penalty on identical SOP training-identity queries.

The extra rows are class-disjoint fit identities, so this is a diagnostic of
unseen distractors for a frozen pretrained encoder, not an official-test score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from export_unicom_sop_embeddings import load_sop_embedding_archive
from torch.nn import functional as F

from sfora.representation_ceiling import deterministic_class_partition

ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SPLIT_SEED = 179019
BOOTSTRAP_SEED = 179019


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_top1(
    vectors: torch.Tensor, labels: np.ndarray, holdout: np.ndarray, *, block_rows: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Return hits against holdout-only and all-training galleries.

    ``holdout`` is in source-gallery ordinal order. Torch argmax chooses the
    first ordinal on exact score ties, as does the deployed stable tie rule.
    """

    if vectors.ndim != 2 or len(vectors) != len(labels) or len(holdout) < 2:
        raise ValueError("SOP gallery inputs differ")
    if block_rows < 1 or not bool(torch.isfinite(vectors).all()):
        raise ValueError("SOP gallery score inputs differ")
    if len(np.unique(holdout)) != len(holdout) or not np.all(holdout[:-1] < holdout[1:]):
        raise ValueError("SOP holdout indexes must be unique and ordered")
    if holdout[0] < 0 or holdout[-1] >= len(vectors):
        raise ValueError("SOP holdout index is out of range")

    source = F.normalize(vectors.float(), dim=1).contiguous()
    gallery = source.T.contiguous()
    small_hits = np.empty(len(holdout), dtype=np.bool_)
    full_hits = np.empty(len(holdout), dtype=np.bool_)
    holdout_torch = torch.from_numpy(holdout)
    for start in range(0, len(holdout), block_rows):
        stop = min(start + block_rows, len(holdout))
        query_ordinals = holdout_torch[start:stop]
        scores = source[query_ordinals] @ gallery
        scores[torch.arange(stop - start), query_ordinals] = -torch.inf
        full_top1 = scores.argmax(dim=1).numpy()
        small_top1 = holdout[scores[:, holdout_torch].argmax(dim=1).numpy()]
        targets = labels[holdout[start:stop]]
        full_hits[start:stop] = labels[full_top1] == targets
        small_hits[start:stop] = labels[small_top1] == targets
    return small_hits, full_hits


def cluster_interval(
    labels: np.ndarray, small: np.ndarray, full: np.ndarray
) -> tuple[float, float]:
    """Product-clustered percentile interval for full minus small Recall@1."""

    _, inverse = np.unique(labels, return_inverse=True)
    classes = int(inverse.max()) + 1
    counts = np.bincount(inverse, minlength=classes)
    deltas = np.bincount(inverse, weights=full.astype(int) - small.astype(int), minlength=classes)
    generator = np.random.default_rng(BOOTSTRAP_SEED)
    results = np.empty(10_000, dtype=np.float64)
    for sample in range(len(results)):
        chosen = generator.integers(0, classes, size=classes)
        results[sample] = deltas[chosen].sum() / counts[chosen].sum()
    lower, upper = np.quantile(results, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.archive) != ARCHIVE_SHA256:
        raise ValueError("SOP gallery-size output or source authority differs")
    torch.set_num_threads(8)
    started = time.perf_counter()
    archive = load_sop_embedding_archive(args.archive)
    labels = np.asarray(archive["train_labels"], dtype=np.int64)
    vectors = torch.from_numpy(np.ascontiguousarray(archive["train_embeddings"]))
    if vectors.shape != (59_551, 768) or labels.shape != (59_551,):
        raise ValueError("SOP training source shape differs")
    partition = deterministic_class_partition(
        tuple(labels.tolist()), fit_fraction=0.9, seed=SPLIT_SEED
    )
    holdout = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(holdout) != 5_851 or len(partition.fit_row_indexes) != 53_700:
        raise ValueError("SOP class-disjoint split differs")
    if not set(labels[holdout]).isdisjoint(labels[list(partition.fit_row_indexes)]):
        raise ValueError("SOP fit and holdout class identities overlap")
    with torch.inference_mode():
        small, full = paired_top1(vectors, labels, holdout)
    if bool(np.any(full & ~small)):
        raise ValueError("adding class-disjoint distractors increased a top-1 hit")
    receipt = {
        "schema": "sfora-sop-train-holdout-gallery-size-audit-v1",
        "claim_eligible": False,
        "model": "pretrained UNICOM ViT-B/16, full 768-D normalized float cosine",
        "dataset": "Stanford Online Products",
        "split": "official training identities; deterministic class-disjoint 90/10 split",
        "split_seed": SPLIT_SEED,
        "query_rows": len(holdout),
        "small_gallery_rows": len(holdout),
        "full_gallery_rows": len(labels),
        "added_fit_identity_distractors": len(partition.fit_row_indexes),
        "small_recall_at_1": float(small.mean()),
        "full_recall_at_1": float(full.mean()),
        "full_minus_small_recall_at_1": float(full.mean() - small.mean()),
        "full_minus_small_cluster_bootstrap_ci95": cluster_interval(labels[holdout], small, full),
        "small_only_hits": int(np.count_nonzero(small & ~full)),
        "full_only_hits": int(np.count_nonzero(full & ~small)),
        "query_source_ordinals": holdout.tolist(),
        "small_hits": small.astype(int).tolist(),
        "full_hits": full.astype(int).tolist(),
        "tie_rule": "lowest source-gallery ordinal among equal float32 cosine scores",
        "self_excluded": True,
        "torch_threads": torch.get_num_threads(),
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": {"processor": platform.processor(), "torch": torch.__version__},
        "archive_sha256": ARCHIVE_SHA256,
        "script_sha256": sha256(Path(__file__)),
    }
    args.output.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "small_recall_at_1",
                    "full_recall_at_1",
                    "full_minus_small_recall_at_1",
                    "full_minus_small_cluster_bootstrap_ci95",
                    "elapsed_seconds",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
