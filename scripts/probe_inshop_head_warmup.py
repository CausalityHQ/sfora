#!/usr/bin/env python3
"""Fit-cache-only rejection gate for one fixed head-first training phase."""

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import digest_rows, schedule, split
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from score_inshop_crop_view_pair import PARTITION_SHA, sha256
from torch.nn import functional as F
from train_sop_siglip2_compact import (
    initialize_head_and_classifier,
    member_bank_initial_values,
    member_bank_positive_ordinals,
    member_bank_rank_loss,
)

from sfora.unicom_training import sharded_mask_arcface_loss


def warm_head(source, head, classifier, target, batches):
    if len(batches) != 100:
        raise ValueError("head warm-up requires exactly100 cached steps")
    unit = F.normalize(source.detach().float(), dim=1)
    masks = torch.arange(head.out_features).unsqueeze(0)
    optimizer = torch.optim.AdamW(
        list(head.parameters()) + [classifier], lr=1e-4, weight_decay=0.05
    )
    losses = []
    for rows in batches:
        optimizer.zero_grad(set_to_none=True)
        loss = sharded_mask_arcface_loss(
            head(unit[list(rows)]), classifier, target[list(rows)], masks, margin=0.3, scale=64
        )
        if not torch.isfinite(loss):
            raise ValueError("head warm-up loss nonfinite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(head.parameters()) + [classifier], 1, error_if_nonfinite=True
        )
        optimizer.step()
        if any(not torch.isfinite(value).all() for value in list(head.parameters()) + [classifier]):
            raise ValueError("head warm-up parameter nonfinite")
        losses.append(float(loss.detach()))
    return losses


def probe(source, head, classifier, target, batches):
    bank = member_bank_initial_values(source, head, live_head=False)
    positives = member_bank_positive_ordinals(target.numpy(), allow_singletons=True)
    counts = torch.bincount(target)
    norms, ce_losses, features = [], [], []
    for rows in batches:
        index = torch.tensor(rows)
        query = source[index].detach().clone().requires_grad_()
        compact = head(F.normalize(query, dim=1))
        ce = sharded_mask_arcface_loss(
            compact, classifier, target[index], torch.arange(128).unsqueeze(0), margin=0.3, scale=64
        )
        loss = ce
        if bool((counts[target[index]] > 1).all()):
            positive = positives[index]
            width = int((positive >= 0).sum(1).max())
            loss = loss + 8 * member_bank_rank_loss(
                compact, bank, head, positive[:, :width], index, live_head=False
            )
        gradient = torch.autograd.grad(loss, query)[0]
        norm = gradient.double().norm(dim=1)
        if not torch.isfinite(norm).all() or bool((norm == 0).any()):
            raise ValueError("head warm-up source gradient undefined")
        norms.extend(norm.tolist())
        ce_losses.append(float(ce.detach()))
        features.append(F.normalize(compact.detach(), dim=1))
    compact = torch.cat(features)
    centered = compact - compact.mean(0)
    spectrum = torch.linalg.svdvals(centered).square()
    return {
        "source_gradient_norms": norms,
        "native_arcface_losses": ce_losses,
        "compact_variance": float(centered.square().mean()),
        "compact_effective_rank": float(spectrum.sum().square() / spectrum.square().sum()),
    }


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
        raise ValueError("head warm-up cached authority differs")
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    labels = tuple(row[1] for row in train)
    fit, _ = split(labels)
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("head warm-up fit authority differs")
    names = sorted({labels[row] for row in fit})
    index = {name: position for position, name in enumerate(names)}
    target = torch.tensor([index[labels[row]] for row in fit])
    cache = np.load(args.source_cache, mmap_mode="r", allow_pickle=False)
    if cache.shape != (25882, 1024) or cache.dtype != np.float32:
        raise ValueError("head warm-up cache geometry differs")
    source = torch.from_numpy(cache[list(fit)].copy())
    torch.set_num_threads(8)
    torch.manual_seed(179024)
    head, classifier, pca_sha = initialize_head_and_classifier(
        source, tuple(target.tolist()), allow_singletons=True
    )
    if pca_sha != "f387aae10a5fe080ebbc9a8ec0e77ff9e86b4812b6e1066048de1cf7cf1e4ea8":
        raise ValueError("head warm-up PCA authority differs")
    batches = schedule(tuple(labels[row] for row in fit), 1000, seed=179024)
    schedule_sha = hashlib.sha256(np.asarray(batches, dtype="<i4").tobytes()).hexdigest()
    if schedule_sha != "c12def923f604ec0c72fe52f498dda80985905760428d4c8ad8485972a5996b9":
        raise ValueError("head warm-up schedule authority differs")
    before = probe(source, head, classifier, target, batches[:17])
    warm = copy.deepcopy(head)
    warm_classifier = torch.nn.Parameter(classifier.detach().clone())
    losses = warm_head(source, warm, warm_classifier, target, batches[:100])
    after = probe(source, warm, warm_classifier, target, batches[:17])
    gradient_ratio = float(
        np.median(
            np.array(after["source_gradient_norms"]) / np.array(before["source_gradient_norms"])
        )
    )
    ce_ratio = float(
        np.mean(after["native_arcface_losses"]) / np.mean(before["native_arcface_losses"])
    )
    criteria = {
        "pressure": gradient_ratio <= 0.75,
        "classification_loss": ce_ratio <= 0.8,
        "variance": after["compact_variance"] >= 0.8 * before["compact_variance"],
        "effective_rank": after["compact_effective_rank"] >= 0.8 * before["compact_effective_rank"],
    }
    wall = time.perf_counter() - started
    report = {
        "schema": "sfora-inshop-head-first-cache-v1",
        "claim_eligible": False,
        "split": "official TRAIN fit only, no held read",
        "before": before,
        "after": after,
        "gradient_norm_ratio_median": gradient_ratio,
        "native_arcface_loss_ratio": ce_ratio,
        "cached_losses": losses,
        "warm_steps": 100,
        "probe_batches": 17,
        "criteria": criteria,
        "advance": all(criteria.values()) and wall <= 120,
        "cpu_main_wall_seconds": wall,
        "source_sha256": sha256(Path(__file__)),
        "features_sha256": SOURCE_CACHE_SHA,
        "partition_sha256": PARTITION_SHA,
        "fit_sha256": digest_rows(fit),
        "pca_sha256": pca_sha,
        "schedule_sha256": schedule_sha,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("before", "after", "cached_losses")
            }
        )
    )


if __name__ == "__main__":
    main()
