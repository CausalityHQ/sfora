#!/usr/bin/env python3
"""Rescore frozen SOP TRAIN embeddings with unseen-product gallery only."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

from compare_sop_siglip2_member_bank_arms import product_bootstrap
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.unicom_rank_finish import identity_balanced_batches
from train_sop_siglip2_compact import score_packed_full_gallery

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
COMPARISON_SHA256 = "6646adc6a0c72838f398268876d41d317e94c1c0c1bde47b09eb424a251ac097"
SEEDS = (179023, 179024, 179025)
ARMS = {
    "bank": "sfora-siglip2-bf16-member-bank-{seed}-bank-v1",
    "matched_float": "sfora-siglip2-bf16-rankmatched-{seed}-float-v1",
    "original_float": "sfora-siglip2-bf16-member-bank-{seed}-float_rank-v1",
}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def fit_impostors(
    packed: PackedInt8Embeddings, labels: np.ndarray, held: np.ndarray, old_r1: np.ndarray
) -> dict[str, int]:
    """Classify the already recorded full-gallery top-1 errors by row origin."""
    codes = packed.codes.float()
    inverse = packed.inverse_norms.float()
    errors = held[old_r1 == 0]
    fit_errors = 0
    held_errors = 0
    held_mask = np.zeros(len(labels), dtype=np.bool_)
    held_mask[held] = True
    for start in range(0, len(errors), 64):
        block = errors[start : start + 64]
        rows = torch.from_numpy(block.copy())
        scores = (codes[rows] @ codes.T) * inverse[rows, None] * inverse[None, :]
        scores[torch.arange(len(rows)), rows] = -torch.inf
        top1 = torch.argmax(scores, dim=1).numpy()
        if np.any(labels[top1] == labels[block]):
            raise ValueError("full-gallery error status differs from frozen receipt")
        held_errors += int(held_mask[top1].sum())
        fit_errors += int((~held_mask[top1]).sum())
    return {"fit_row": fit_errors, "heldout_row": held_errors}


def self_test() -> None:
    values = torch.zeros((6, 128), dtype=torch.float32)
    values[0, 0] = values[4, 0] = 1
    values[1, 0] = -1
    values[2, 1] = values[5, 1] = 1
    values[3, 1] = -1
    packed = pack_int8_unit_embeddings(values)
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    held = np.asarray([4, 5], dtype=np.int64)
    assert fit_impostors(packed, labels, held, np.zeros(2)) == {"fit_row": 2, "heldout_row": 0}
    assert fit_impostors(packed, labels, held, np.ones(2)) == {"fit_row": 0, "heldout_row": 0}
    subset = score_packed_full_gallery(
        packed.codes[held], packed.inverse_norms[held], torch.from_numpy(labels[held]),
        torch.arange(2), device=torch.device("cpu"),
    )
    assert subset["recall_at_1"] == subset["map_at_r"] == 1.0


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--run-base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
        or sha256(args.comparison) != COMPARISON_SHA256
    ):
        raise ValueError("unseen-gallery input or output authority differs")
    torch.set_num_threads(16)
    comparison = json.loads(args.comparison.read_text())
    if comparison.get("bank_specific_screen_pass") is not True or comparison.get("seeds") != list(SEEDS):
        raise ValueError("unseen-gallery comparison authority differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
    if labels.shape != ids.shape or labels.shape != (59_551,):
        raise ValueError("unseen-gallery TRAIN inventory differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=179019)
    fit = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(fit) != 53_700 or len(held) != 5_851 or len(np.unique(labels[held])) != 1_132:
        raise ValueError("unseen-gallery product partition differs")
    rows: dict[str, dict] = {}
    per_query: dict[str, dict[str, list[np.ndarray]]] = {
        name: {"r1": [], "map_at_r": []} for name in ARMS
    }
    refresh: dict[str, dict] = {}
    for seed in SEEDS:
        rows[str(seed)] = {}
        for name, pattern in ARMS.items():
            run = args.run_base / pattern.format(seed=seed)
            receipt_path = run / "receipt.json"
            receipt_digest = sha256(receipt_path)
            if receipt_digest != comparison["arms"][str(seed)][f"{name}_receipt_sha256"]:
                raise ValueError("unseen-gallery training receipt differs")
            receipt = json.loads(receipt_path.read_text())
            embedding_path = run / "train_embeddings.npy"
            if (
                receipt.get("source_archive_sha256") != ARCHIVE_SHA256
                or receipt.get("seed") != seed
                or receipt.get("query_image_ids_sha256") != hashlib.sha256(ids[held].tobytes()).hexdigest()
                or sha256(embedding_path) != receipt.get("train_embeddings_sha256")
                or receipt.get("quality", {}).get("native_top10_exact") is not True
            ):
                raise ValueError("unseen-gallery export authority differs")
            values = np.load(embedding_path, mmap_mode="r", allow_pickle=False)
            if values.shape != (59_551, 128) or values.dtype != np.float32 or not np.isfinite(values).all():
                raise ValueError("unseen-gallery embedding geometry differs")
            packed = pack_int8_unit_embeddings(torch.from_numpy(np.asarray(values).copy()))
            old_r1 = np.asarray(receipt["quality"]["per_query_r1"], dtype=np.float64)
            if (
                old_r1.shape != (len(held),)
                or not np.isin(old_r1, (0, 1)).all()
                or not np.isclose(old_r1.mean(), receipt["quality"]["recall_at_1"])
            ):
                raise ValueError("unseen-gallery old query quality differs")
            impostors = fit_impostors(packed, labels, held, old_r1)
            if sum(impostors.values()) != int((1 - old_r1).sum()):
                raise ValueError("unseen-gallery full-gallery error inventory differs")
            subset = score_packed_full_gallery(
                packed.codes[held],
                packed.inverse_norms[held],
                torch.from_numpy(labels[held].copy()),
                torch.arange(len(held)),
                device=torch.device("cpu"),
            )
            per_query[name]["r1"].append(np.asarray(subset["per_query_r1"], dtype=np.float64))
            per_query[name]["map_at_r"].append(np.asarray(subset["per_query_ap"], dtype=np.float64))
            rows[str(seed)][name] = {
                "receipt_sha256": receipt_digest,
                "embeddings_sha256": sha256(embedding_path),
                "full_gallery_r1": receipt["quality"]["recall_at_1"],
                "heldout_only_r1": subset["recall_at_1"],
                "heldout_only_map_at_r": subset["map_at_r"],
                "full_gallery_top1_errors": impostors,
            }
            if name == "bank":
                schedule = identity_balanced_batches(
                    tuple(map(str, labels[fit])), batch_size=64, images_per_identity=4,
                    seed=seed, epoch=1, steps=1_000,
                )
                if hashlib.sha256(np.asarray(schedule, dtype="<i4").tobytes()).hexdigest() != receipt["schedule_sha256"]:
                    raise ValueError("unseen-gallery bank refresh schedule differs")
                counts = np.bincount(np.asarray(schedule).ravel(), minlength=len(fit))
                refresh[str(seed)] = {
                    "never": int(np.count_nonzero(counts == 0)),
                    "once": int(np.count_nonzero(counts == 1)),
                    "multiple": int(np.count_nonzero(counts > 1)),
                    "presentations": int(counts.sum()),
                }
    comparisons = {}
    for control in ("matched_float", "original_float"):
        comparisons[f"bank_minus_{control}"] = {
            metric: product_bootstrap(
                np.mean(np.stack(per_query["bank"][metric]), axis=0)
                - np.mean(np.stack(per_query[control][metric]), axis=0),
                labels[held],
            )
            for metric in ("r1", "map_at_r")
        }
    matched = comparisons["bank_minus_matched_float"]
    original = comparisons["bank_minus_original_float"]
    # Frozen mechanistic screen on the already-used holdout; never a SOTA gate.
    survives = (
        matched["r1"]["point"] >= 0.0036
        and matched["r1"]["lower_95"] > 0
        and matched["map_at_r"]["point"] >= 0
        and original["r1"]["lower_95"] > 0
    )
    result = {
        "schema": "sfora-sop-siglip2-unseen-gallery-diagnostic-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN selected holdout; heldout-product-only gallery; self excluded",
        "source_sha256": sha256(Path(__file__)),
        "dependency_source_sha256": {
            "packed_wire": sha256(Path(inspect.getsourcefile(pack_int8_unit_embeddings))),
            "packed_scorer": sha256(
                Path(inspect.getsourcefile(inspect.unwrap(score_packed_full_gallery)))
            ),
            "partition": sha256(Path(inspect.getsourcefile(deterministic_class_partition))),
            "schedule": sha256(Path(inspect.getsourcefile(identity_balanced_batches))),
            "bootstrap": sha256(Path(inspect.getsourcefile(product_bootstrap))),
        },
        "source_archive_sha256": ARCHIVE_SHA256,
        "comparison_sha256": COMPARISON_SHA256,
        "holdout_queries": len(held),
        "holdout_products": 1_132,
        "seeds": SEEDS,
        "arms": rows,
        "bank_refresh_counts": refresh,
        "comparisons": comparisons,
        "bank_port_screen_survives": survives,
        "interpretation": "Retrospective mechanism diagnostic; repeated TRAIN holdout cannot confirm an external quality claim",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"bank_port_screen_survives": survives, "comparisons": comparisons}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
