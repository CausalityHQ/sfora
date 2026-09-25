#!/usr/bin/env python3
"""Frozen SOP TRAIN fit-only member-bank stale-impostor coverage gate."""

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

from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def contains_impostor_in_topk(
    scores: torch.Tensor,
    query_classes: torch.Tensor,
    gallery_classes: torch.Tensor,
    impostor_ordinals: torch.Tensor,
    *,
    k: int = 64,
) -> torch.Tensor:
    """Tie-stable top-k negative membership, excluding every same-product row."""

    if (
        scores.ndim != 2
        or scores.shape != (len(query_classes), len(gallery_classes))
        or len(impostor_ordinals) != len(query_classes)
        or not 0 < k < scores.shape[1]
        or bool((impostor_ordinals < 0).any())
        or bool((impostor_ordinals >= scores.shape[1]).any())
    ):
        raise ValueError("member-bank score geometry differs")
    if bool((gallery_classes[impostor_ordinals] == query_classes).any()):
        raise ValueError("member-bank impostor is from query product")
    negative = scores.masked_fill(gallery_classes[None, :] == query_classes[:, None], -torch.inf)
    boundary = negative.topk(k, dim=1).values[:, -1]
    if not bool(torch.isfinite(boundary).all()):
        raise ValueError("member-bank has fewer than k negatives")
    impostor_scores = negative.gather(1, impostor_ordinals[:, None]).squeeze(1)
    higher = (negative > impostor_scores[:, None]).sum(dim=1)
    earlier_equal = (
        (negative == impostor_scores[:, None])
        & (
            torch.arange(scores.shape[1], device=scores.device)[None, :]
            < impostor_ordinals[:, None]
        )
    ).sum(dim=1)
    return higher + earlier_equal < k


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
        raise ValueError("member-bank authority differs")
    receipt = json.loads(args.training_receipt.read_text())
    census = json.loads(args.fit_census.read_text())
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
        or census.get("training_receipt_sha256") != args.expected_training_receipt_sha256
    ):
        raise ValueError("member-bank artifacts differ")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        all_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    fit_rows = np.asarray(
        deterministic_class_partition(
            tuple(map(int, all_labels)), fit_fraction=0.9, seed=179019
        ).fit_row_indexes,
        dtype=np.int64,
    )
    labels = all_labels[fit_rows]
    _, classes = np.unique(labels, return_inverse=True)
    if len(fit_rows) != 53_700 or len(np.unique(labels)) != 10_186:
        raise ValueError("member-bank partition differs")
    source = np.load(args.source_features, mmap_mode="r", allow_pickle=False)
    trained = np.load(args.train_embeddings, mmap_mode="r", allow_pickle=False)
    if source.shape != (len(all_labels), 1024) or trained.shape != (len(all_labels), 128):
        raise ValueError("member-bank embedding geometry differs")
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
    initial_proxy_sha = hashlib.sha256(initial_classifier.detach().numpy().tobytes()).hexdigest()
    if (
        pca_sha != receipt["source_pca_sha256"]
        or head_sha != receipt["initial_head_sha256"]
        or initial_proxy_sha != receipt["initial_classifier_sha256"]
    ):
        raise ValueError("member-bank initial geometry differs")
    checkpoint = torch.load(args.training_checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("arm") != "arcface"
        or checkpoint["classifier"].shape != initial_classifier.shape
    ):
        raise ValueError("member-bank control checkpoint differs")
    errors = census["trained_arcface"]["errors"]
    if len(errors) != 4_296:
        raise ValueError("member-bank error census differs")
    query_rows = np.asarray([int(x["query_fit_ordinal"]) for x in errors], dtype=np.int64)
    impostor_rows = np.asarray([int(x["impostor_fit_ordinal"]) for x in errors], dtype=np.int64)
    device = torch.device("cuda:0")
    bank_a = F.normalize(compact_head_features(fit_source, head), dim=1).to(device)
    w0 = F.normalize(initial_classifier.float(), dim=1).to(device)
    w1 = F.normalize(checkpoint["classifier"].float(), dim=1).to(device)
    class_gpu = torch.from_numpy(classes.copy()).to(device)
    bank_b = F.normalize(bank_a - w0[class_gpu] + w1[class_gpu], dim=1)
    current = F.normalize(
        torch.from_numpy(np.asarray(trained[fit_rows][query_rows]).copy()).to(device), dim=1
    )
    query_gpu = torch.from_numpy(query_rows).to(device)
    impostor_gpu = torch.from_numpy(impostor_rows).to(device)
    hits = [0, 0]
    for start in range(0, len(query_rows), 32):
        stop = min(start + 32, len(query_rows))
        q = current[start:stop]
        qclass = class_gpu[query_gpu[start:stop]]
        impostor = impostor_gpu[start:stop]
        for index, bank in enumerate((bank_a, bank_b)):
            hits[index] += int(
                contains_impostor_in_topk(q @ bank.T, qclass, class_gpu, impostor).sum().item()
            )
    if hits[0] / len(errors) >= 0.60:
        decision = "raw_bank_allowed"
    elif hits[1] / len(errors) >= 0.60:
        decision = "reanchored_bank_allowed"
    else:
        decision = "stop_before_training"
    result = {
        "schema": "sfora-sop-siglip2-member-bank-preflight-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN fit products only, seed 179019",
        "source_sha256": sha256(Path(__file__)),
        "training_receipt_sha256": args.expected_training_receipt_sha256,
        "fit_census_sha256": args.expected_fit_census_sha256,
        "source_pca_sha256": pca_sha,
        "initial_classifier_sha256": initial_proxy_sha,
        "fit_rows_sha256": hashlib.sha256(fit_rows.astype("<i8").tobytes()).hexdigest(),
        "errors": len(errors),
        "k": 64,
        "gate": 0.60,
        "raw_hits": hits[0],
        "raw_coverage": hits[0] / len(errors),
        "reanchored_hits": hits[1],
        "reanchored_coverage": hits[1] / len(errors),
        "decision": decision,
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
                    "errors",
                    "raw_hits",
                    "raw_coverage",
                    "reanchored_hits",
                    "reanchored_coverage",
                    "decision",
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
