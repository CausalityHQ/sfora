#!/usr/bin/env python3
"""Fixed full-source MAIN/compact-rank gradient kill screen, TRAIN fit cache only."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import digest_rows, schedule, split
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from probe_inshop_source_classifier import unused_gradient
from score_inshop_crop_view_pair import PARTITION_SHA, sha256
from torch import nn
from torch.nn import functional as F
from train_sop_siglip2_compact import initialize_head_and_classifier, member_bank_positive_ordinals

from sfora.deployed_code_rank import smooth_ap_bank_loss
from sfora.unicom_training import sharded_mask_arcface_loss


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("partition", "source-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.partition) != PARTITION_SHA
        or sha256(args.source_cache) != SOURCE_CACHE_SHA
    ):
        raise ValueError("source-main cache authority differs")
    labels = tuple(
        line.split()[1]
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    )
    fit, _ = split(labels)
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("source-main fit authority differs")
    names = sorted({labels[row] for row in fit})
    lookup = {name: i for i, name in enumerate(names)}
    target = torch.tensor([lookup[labels[row]] for row in fit])
    cache = np.load(args.source_cache, mmap_mode="r", allow_pickle=False)
    if cache.shape != (25882, 1024) or cache.dtype != np.float32:
        raise ValueError("source-main source geometry differs")
    source = torch.from_numpy(cache[list(fit)].copy())
    torch.set_num_threads(8)
    torch.manual_seed(179024)
    head, native_classifier, pca_sha = initialize_head_and_classifier(
        source, tuple(target.tolist()), allow_singletons=True
    )
    if pca_sha != "f387aae10a5fe080ebbc9a8ec0e77ff9e86b4812b6e1066048de1cf7cf1e4ea8":
        raise ValueError("source-main native128 initializer differs")
    unit = F.normalize(source, dim=1)
    mean = unit.mean(0)
    centered = F.normalize(unit - mean, dim=1)
    sums = torch.zeros(len(names), 1024)
    sums.index_add_(0, target, centered)
    classifier = nn.Parameter(F.normalize(sums, dim=1))
    bank = F.normalize(head(unit).detach(), dim=1)
    positives = member_bank_positive_ordinals(target.numpy(), allow_singletons=True)
    counts = torch.bincount(target)
    batches = schedule(tuple(labels[row] for row in fit), 1000, seed=179024)
    schedule_sha = hashlib.sha256(np.asarray(batches, dtype="<i4").tobytes()).hexdigest()
    if schedule_sha != "c12def923f604ec0c72fe52f498dda80985905760428d4c8ad8485972a5996b9":
        raise ValueError("source-main schedule differs")
    unused, ratios, pressure, residuals, head_ratios, history = [], [], [], [], [], []
    for rows in batches[:17]:
        indexes = torch.tensor(rows)
        query = source[indexes].detach().clone().requires_grad_()
        features = head(F.normalize(query, dim=1))
        native = sharded_mask_arcface_loss(
            features,
            native_classifier,
            target[indexes],
            torch.arange(128).reshape(1, -1),
            margin=0.3,
            scale=64,
        )
        source_main = sharded_mask_arcface_loss(
            F.normalize(F.normalize(query, dim=1) - mean, dim=1),
            classifier,
            target[indexes],
            torch.arange(1024).reshape(1, -1),
            margin=0.3,
            scale=64,
        )
        rank = native.new_zeros(())
        active = bool((counts[target[indexes]] > 1).all())
        if active:
            positive = positives[indexes]
            width = int((positive >= 0).sum(1).max())
            rank = 8 * smooth_ap_bank_loss(
                F.normalize(features, dim=1), bank, positive[:, :width], indexes
            )
        control = native + rank
        total = source_main + rank
        old_query, old_head = torch.autograd.grad(control, (query, head.weight), retain_graph=True)
        main_query = torch.autograd.grad(source_main, query, retain_graph=True)[0]
        new_query, new_head, new_proxy = torch.autograd.grad(
            total, (query, head.weight, classifier), allow_unused=True
        )
        gradients = (old_query, old_head, main_query, new_query, new_proxy)
        if any(not torch.isfinite(value).all() for value in gradients) or new_proxy.norm() <= 0:
            raise ValueError("source-main finite/proxy gradient failed")
        new_norm = new_query.double().norm(dim=1)
        old_norm = old_query.double().norm(dim=1)
        if (new_norm <= 0).any() or (old_norm <= 0).any():
            raise ValueError("source-main total gradient missing")
        unit_query = F.normalize(query.detach(), dim=1)
        unused.extend(
            (
                unused_gradient(new_query, head.weight.detach(), unit_query).double().norm(dim=1)
                / new_norm
            ).tolist()
        )
        residuals.extend(
            (
                unused_gradient(old_query, head.weight.detach(), unit_query).double().norm(dim=1)
                / old_norm
            ).tolist()
        )
        ratios.extend((new_norm / old_norm).tolist())
        pressure.extend((main_query.double().norm(dim=1) / new_norm).tolist())
        if active:
            if new_head is None or not torch.isfinite(new_head).all() or new_head.norm() <= 0:
                raise ValueError("source-main compact head gradient missing")
            head_ratios.append(float(new_head.double().norm() / old_head.double().norm()))
        elif new_head is not None:
            raise ValueError("source-main inactive head policy differs")
        history.append(
            {
                "native_arcface": float(native.detach()),
                "source_arcface": float(source_main.detach()),
                "rank8": float(rank.detach()),
                "rank_active": active,
                "source_proxy_gradient_norm": float(new_proxy.norm()),
            }
        )
    wall = time.perf_counter() - started
    criteria = {
        "inventory": len(unused) == 1088 and len(head_ratios) == 15,
        "unused_route": float(np.median(unused)) >= 0.2,
        "main_pressure": float(np.median(pressure)) >= 0.1,
        "bounded_total": 0.25 <= float(np.median(ratios)) <= 4,
        "head_pressure": float(np.median(head_ratios)) >= 0.25,
        "native_residual": max(residuals) <= 1e-5,
        "cpu_budget": wall <= 120,
    }
    result = {
        "schema": "sfora-inshop-source-main-cache-v1",
        "claim_eligible": False,
        "decision": "GO_ENCODER_DESIGN_REVIEW" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "median_unused_norm_fraction": float(np.median(unused)),
        "median_main_total_norm_fraction": float(np.median(pressure)),
        "median_total_native_norm_ratio": float(np.median(ratios)),
        "median_head_native_norm_ratio": float(np.median(head_ratios)),
        "unused_norm_fractions": unused,
        "main_total_norm_fractions": pressure,
        "total_native_norm_ratios": ratios,
        "head_native_norm_ratios": head_ratios,
        "history": history,
        "cpu_wall_seconds": wall,
        "source_sha256": sha256(Path(__file__)),
        "partition_sha256": PARTITION_SHA,
        "source_cache_sha256": SOURCE_CACHE_SHA,
        "pca_sha256": pca_sha,
        "fit_sha256": digest_rows(fit),
        "schedule_sha256": schedule_sha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "decision",
                    "criteria",
                    "median_unused_norm_fraction",
                    "median_main_total_norm_fraction",
                    "median_total_native_norm_ratio",
                    "median_head_native_norm_ratio",
                    "cpu_wall_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
