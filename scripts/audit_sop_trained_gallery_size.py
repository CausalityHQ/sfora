"""Compare a trained SOP checkpoint on holdout-only and seen-distractor galleries.

The larger gallery contains identities used to train the checkpoint. Its
result is a diagnostic bracket, never a checkpoint-selection metric.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import BinaryIO

import numpy as np
import torch
from audit_sop_gallery_size_train_holdout import cluster_interval, paired_top1
from export_unicom_sop_embeddings import load_sop_embedding_archive
from train_sop_compact_backbone import CHECKPOINT_SHA256, publish_file_noreplace

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition

SOURCE_ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SPLIT_SEED = 179019
SOURCE_RELATIVES = (
    "scripts/audit_sop_trained_gallery_size.py",
    "scripts/audit_sop_gallery_size_train_holdout.py",
    "scripts/export_unicom_sop_embeddings.py",
    "scripts/train_sop_compact_backbone.py",
    "src/sfora/joint_relational_compaction.py",
    "src/sfora/representation_ceiling.py",
    "src/sfora/sop_evaluation.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {relative: sha256(root / relative) for relative in SOURCE_RELATIVES}


def packed_pair(
    vectors: torch.Tensor, labels: np.ndarray, holdout: np.ndarray, *, block_rows: int = 64
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact signed-byte/f16 forward score, with self masked before top-1."""

    if (
        vectors.ndim != 2
        or len(vectors) != len(labels)
        or len(holdout) < 2
        or block_rows < 1
        or not bool(torch.isfinite(vectors).all())
    ):
        raise ValueError("SOP trained packed input differs")
    packed = pack_int8_unit_embeddings(vectors.float().contiguous())
    codes = packed.codes.float()
    inverse = packed.inverse_norms.float()
    gallery = codes.T.contiguous()
    holdout_torch = torch.from_numpy(holdout)
    small_hits = np.empty(len(holdout), dtype=np.bool_)
    full_hits = np.empty(len(holdout), dtype=np.bool_)
    full_winners = np.empty(len(holdout), dtype=np.int64)
    for start in range(0, len(holdout), block_rows):
        stop = min(start + block_rows, len(holdout))
        queries = holdout_torch[start:stop]
        scores = (codes[queries] @ gallery) * inverse[queries, None] * inverse[None, :]
        scores[torch.arange(stop - start), queries] = -torch.inf
        full_top1 = scores.argmax(dim=1).numpy()
        small_top1 = holdout[scores[:, holdout_torch].argmax(dim=1).numpy()]
        targets = labels[holdout[start:stop]]
        full_hits[start:stop] = labels[full_top1] == targets
        small_hits[start:stop] = labels[small_top1] == targets
        full_winners[start:stop] = full_top1
    if bool(np.any(full_hits & ~small_hits)):
        raise ValueError("adding SOP distractors increased packed top-1 hits")
    return small_hits, full_hits, full_winners


def arm_result(
    small: np.ndarray, full: np.ndarray, winners: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    if not (len(small) == len(full) == len(winners) == len(labels)):
        raise ValueError("SOP trained gallery comparison rows differ")
    return {
        "small_recall_at_1": float(small.mean()),
        "full_recall_at_1": float(full.mean()),
        "full_minus_small_recall_at_1": float(full.mean() - small.mean()),
        "full_minus_small_cluster_bootstrap_ci95": cluster_interval(labels, small, full),
        "small_only_hits": int(np.count_nonzero(small & ~full)),
        "full_only_hits": int(np.count_nonzero(full & ~small)),
        "small_hits": small.astype(int).tolist(),
        "full_hits": full.astype(int).tolist(),
        "full_top1_ordinals": winners.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--trained-features", type=Path, required=True)
    parser.add_argument("--trained-features-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.source_archive) != SOURCE_ARCHIVE_SHA256
        or sha256(args.trained_features) != args.trained_features_sha256
    ):
        raise ValueError("SOP trained gallery input authority differs")
    initial_source_manifest = source_manifest()
    source = load_sop_embedding_archive(args.source_archive)
    with np.load(args.trained_features, allow_pickle=False) as archive:
        features = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        image_ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if (
        metadata.get("schema") != "sfora-sop-trained-train-features-v1"
        or metadata.get("rows") != 59_551
        or metadata.get("width") not in (128, 768)
        or metadata.get("source_checkpoint_sha256") != CHECKPOINT_SHA256
        or not isinstance(metadata.get("trained_checkpoint_sha256"), str)
        or len(metadata["trained_checkpoint_sha256"]) != 64
        or not isinstance(metadata.get("training_receipt_sha256"), str)
        or len(metadata["training_receipt_sha256"]) != 64
        or features.shape != (59_551, metadata["width"])
        or labels.shape != (59_551,)
        or image_ids.shape != (59_551,)
        or not np.array_equal(labels, np.asarray(source["train_labels"]))
        or not np.array_equal(image_ids, np.asarray(source["train_image_ids"]))
        or not bool(np.isfinite(features).all())
    ):
        raise ValueError("SOP trained gallery feature inventory differs")
    partition = deterministic_class_partition(
        tuple(labels.tolist()), fit_fraction=0.9, seed=SPLIT_SEED
    )
    holdout = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(holdout) != 5_851 or len(partition.fit_row_indexes) != 53_700:
        raise ValueError("SOP trained gallery partition differs")
    torch.set_num_threads(8)
    started = time.perf_counter()
    vectors = torch.from_numpy(features)
    with torch.inference_mode():
        float_small, float_full, float_winners, _, _ = paired_top1(vectors, labels, holdout)
        packed_small, packed_full, packed_winners = packed_pair(vectors, labels, holdout)
    if abs(float(packed_small.mean()) - float(metadata["holdout_packed_recall_at_1"])) > 1e-6:
        raise ValueError("SOP trained gallery packed holdout parity differs")
    result = {
        "schema": "sfora-sop-trained-gallery-size-diagnostic-v1",
        "claim_eligible": False,
        "checkpoint_selection_eligible": False,
        "larger_gallery_bias": "extra fit identities were seen during model training",
        "dataset": "Stanford Online Products",
        "split": "official train identities; deterministic class-disjoint fit/holdout",
        "split_seed": SPLIT_SEED,
        "query_rows": len(holdout),
        "small_gallery_rows": len(holdout),
        "large_gallery_rows": len(labels),
        "query_source_ordinals": holdout.tolist(),
        "query_labels": labels[holdout].tolist(),
        "float": arm_result(float_small, float_full, float_winners, labels[holdout]),
        "packed": arm_result(packed_small, packed_full, packed_winners, labels[holdout]),
        "trained_feature_archive_sha256": args.trained_features_sha256,
        "trained_feature_metadata": metadata,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "source_sha256": initial_source_manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "hardware": {"processor": platform.processor(), "torch": torch.__version__},
    }
    if source_manifest() != initial_source_manifest:
        raise ValueError("SOP trained gallery scoring source changed during execution")
    wire = (json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode()

    def write_receipt(stream: BinaryIO) -> None:
        stream.write(wire)

    publish_file_noreplace(args.output, write_receipt)
    print(
        json.dumps(
            {arm: result[arm]["full_minus_small_recall_at_1"] for arm in ("float", "packed")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
