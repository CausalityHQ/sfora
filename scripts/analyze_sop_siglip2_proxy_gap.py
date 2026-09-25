#!/usr/bin/env python3
"""Fit-only proxy versus gallery-member diagnostic for trained SOP ArcFace."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.representation_ceiling import deterministic_class_partition


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "source-archive",
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
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not torch.cuda.is_available()
        or sha256(args.training_receipt) != args.expected_training_receipt_sha256
        or sha256(args.fit_census) != args.expected_fit_census_sha256
    ):
        raise ValueError("fit-only proxy-gap authority differs")
    receipt = json.loads(args.training_receipt.read_text())
    census = json.loads(args.fit_census.read_text())
    for path, key in (
        (args.source_archive, "source_archive_sha256"),
        (args.training_checkpoint, "checkpoint_sha256"),
        (args.train_embeddings, "train_embeddings_sha256"),
    ):
        if sha256(path) != receipt[key]:
            raise ValueError("fit-only proxy-gap artifacts differ")
    if census["training_receipt_sha256"] != args.expected_training_receipt_sha256:
        raise ValueError("fit-only proxy-gap census differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        all_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    fit_rows = np.asarray(
        deterministic_class_partition(
            tuple(map(int, all_labels)), fit_fraction=0.9, seed=179019
        ).fit_row_indexes,
        dtype=np.int64,
    )
    labels = all_labels[fit_rows]
    products, targets = np.unique(labels, return_inverse=True)
    all_values = np.load(args.train_embeddings, mmap_mode="r", allow_pickle=False)
    checkpoint = torch.load(args.training_checkpoint, map_location="cpu", weights_only=True)
    if (
        len(products) != checkpoint["classifier"].shape[0]
        or len(fit_rows) != census["trained_arcface"]["fit_queries"]
    ):
        raise ValueError("fit-only proxy-gap classifier geometry differs")
    device = torch.device("cuda:0")
    values = F.normalize(
        torch.from_numpy(np.asarray(all_values[fit_rows]).copy()).to(device), dim=1
    )
    proxies = F.normalize(checkpoint["classifier"].to(device), dim=1)
    targets_gpu = torch.from_numpy(targets).to(device)
    correct_proxy = torch.zeros(len(fit_rows), dtype=torch.bool, device=device)
    for start in range(0, len(fit_rows), 512):
        stop = min(start + 512, len(fit_rows))
        scores = values[start:stop] @ proxies.T
        correct_proxy[start:stop] = scores.argmax(dim=1) == targets_gpu[start:stop]
    errors = census["trained_arcface"]["errors"]
    q = torch.tensor([int(e["query_fit_ordinal"]) for e in errors], device=device)
    impostor_fit_ordinals = torch.tensor(
        [int(e["impostor_fit_ordinal"]) for e in errors], device=device
    )
    mate_fit_ordinals = torch.tensor([int(e["mate_fit_ordinal"]) for e in errors], device=device)
    impostor_rows = np.asarray([int(e["impostor_train_row"]) for e in errors])
    impostor_targets = torch.from_numpy(
        np.searchsorted(products, all_labels[impostor_rows]).astype(np.int64)
    ).to(device)
    ranks: list[int] = []
    for start in range(0, len(q), 512):
        stop = min(start + 512, len(q))
        scores = values[q[start:stop]] @ proxies.T
        impostor_score = scores.gather(1, impostor_targets[start:stop, None])
        ordinals = torch.arange(len(products), device=device)[None, :]
        rank = (
            1
            + (scores > impostor_score).sum(dim=1)
            + ((scores == impostor_score) & (ordinals < impostor_targets[start:stop, None])).sum(
                dim=1
            )
        )
        ranks.extend(map(int, rank.cpu()))
    is_error = torch.zeros(len(fit_rows), dtype=torch.bool, device=device)
    is_error[q] = True
    if (
        int(correct_proxy[is_error].sum())
        != census["trained_arcface"]["proxy_correct_member_wrong_count"]
    ):
        raise ValueError("fit-only proxy-gap proxy classification differs")
    product_sums = torch.zeros_like(proxies)
    product_sums.index_add_(0, targets_gpu, values)
    product_means = F.normalize(product_sums, dim=1)
    coherence = (values * product_means[targets_gpu]).sum(dim=1)
    product_coherence_sums = torch.zeros(len(products), device=device)
    product_coherence_sums.index_add_(0, targets_gpu, coherence)
    product_counts = torch.bincount(targets_gpu, minlength=len(products))
    product_coherence = product_coherence_sums / product_counts
    query_values = values[q]
    mate_values = values[mate_fit_ordinals]
    mean_values = product_means[targets_gpu[q]]
    mate_tangent = F.normalize(
        mate_values - (mate_values * query_values).sum(dim=1, keepdim=True) * query_values,
        dim=1,
    )
    mean_tangent = F.normalize(
        mean_values - (mean_values * query_values).sum(dim=1, keepdim=True) * query_values,
        dim=1,
    )
    tangent_alignment = (mate_tangent * mean_tangent).sum(dim=1)
    impostor_own_proxy_cosine = (values[impostor_fit_ordinals] * proxies[impostor_targets]).sum(
        dim=1
    )
    proxy_correct_errors = correct_proxy[q].cpu().numpy()
    proxy_correct_ranks = np.asarray(ranks)[proxy_correct_errors]
    tangent_misaligned = (tangent_alignment < 0).cpu().numpy()
    orphan_flags = np.asarray([int(error["mate_rank"]) > 10 for error in errors])
    result = {
        "schema": "sfora-sop-siglip2-fit-proxy-gap-v2",
        "claim_eligible": False,
        "split": "SOP official TRAIN fit products only, seed 179019",
        "source_sha256": sha256(Path(__file__)),
        "training_receipt_sha256": args.expected_training_receipt_sha256,
        "fit_census_sha256": args.expected_fit_census_sha256,
        "fit_queries": len(fit_rows),
        "fit_errors": len(errors),
        "proxy_correct_all_fit": int(correct_proxy.sum()),
        "proxy_correct_gallery_correct": int(correct_proxy[~is_error].sum()),
        "proxy_correct_gallery_wrong": int(correct_proxy[is_error].sum()),
        "impostor_proxy_rank_at_most_2": sum(rank <= 2 for rank in ranks),
        "impostor_proxy_rank_at_most_10": sum(rank <= 10 for rank in ranks),
        "impostor_proxy_rank_over_100": sum(rank > 100 for rank in ranks),
        "impostor_proxy_rank_median": float(np.median(ranks)),
        "proxy_correct_error_impostor_proxy_rank_at_most_2": int(np.sum(proxy_correct_ranks <= 2)),
        "proxy_correct_error_impostor_proxy_rank_at_most_5": int(np.sum(proxy_correct_ranks <= 5)),
        "proxy_correct_error_impostor_proxy_rank_at_most_10": int(
            np.sum(proxy_correct_ranks <= 10)
        ),
        "proxy_correct_error_impostor_proxy_rank_median": float(np.median(proxy_correct_ranks)),
        "impostor_own_proxy_cosine_median": float(impostor_own_proxy_cosine.median()),
        "impostor_own_proxy_cosine_p10": float(torch.quantile(impostor_own_proxy_cosine, 0.1)),
        "product_coherence_median": float(product_coherence.median()),
        "product_coherence_p10": float(torch.quantile(product_coherence, 0.1)),
        "tangent_misaligned_error_count": int(tangent_misaligned.sum()),
        "tangent_misaligned_orphan_count": int((tangent_misaligned & orphan_flags).sum()),
        "tangent_alignment_median": float(tangent_alignment.median()),
        "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
