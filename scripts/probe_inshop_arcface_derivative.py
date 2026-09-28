#!/usr/bin/env python3
"""Fit-cache-only gradient contract for one analytic-interior ArcFace variant."""

import argparse
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

from sfora.unicom_training import sharded_mask_arcface_logits


def analytic_interior_arcface_logits(embeddings, weights, labels, masks, *, margin=0.3, scale=64.0):
    """Reference forward, analytic interior derivative, zero target derivative at poles."""
    reference = sharded_mask_arcface_logits(
        embeddings, weights, labels, masks, margin=margin, scale=scale
    )
    cosine = sharded_mask_arcface_logits(embeddings, weights, labels, masks, margin=0.0, scale=1.0)
    rows = torch.arange(len(labels), device=labels.device)
    target = cosine[rows, labels].clamp(-1 + 1e-6, 1 - 1e-6)
    analytic = torch.cos(torch.acos(target) + margin) * scale
    result = reference.clone()
    result[rows, labels] = reference[rows, labels].detach() + (analytic - analytic.detach())
    return result


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
        raise ValueError("ArcFace derivative cache authority differs")
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    labels = tuple(row[1] for row in train)
    fit, _ = split(labels)
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("ArcFace derivative fit inventory differs")
    names = sorted({labels[row] for row in fit})
    lookup = {name: index for index, name in enumerate(names)}
    target = torch.tensor([lookup[labels[row]] for row in fit])
    cache = np.load(args.source_cache, mmap_mode="r", allow_pickle=False)
    if cache.shape != (25882, 1024) or cache.dtype != np.float32:
        raise ValueError("ArcFace derivative source geometry differs")
    source = torch.from_numpy(cache[list(fit)].copy())
    torch.set_num_threads(8)
    torch.manual_seed(179024)
    head, classifier, pca_sha = initialize_head_and_classifier(
        source, tuple(target.tolist()), allow_singletons=True
    )
    if pca_sha != "f387aae10a5fe080ebbc9a8ec0e77ff9e86b4812b6e1066048de1cf7cf1e4ea8":
        raise ValueError("ArcFace derivative PCA differs")
    batches = schedule(tuple(labels[row] for row in fit), 1000, seed=179024)
    schedule_sha = hashlib.sha256(np.asarray(batches, dtype="<i4").tobytes()).hexdigest()
    if schedule_sha != "c12def923f604ec0c72fe52f498dda80985905760428d4c8ad8485972a5996b9":
        raise ValueError("ArcFace derivative schedule differs")
    bank = member_bank_initial_values(source, head, live_head=False)
    positives = member_bank_positive_ordinals(target.numpy(), allow_singletons=True)
    counts = torch.bincount(target)
    masks = torch.arange(128).reshape(1, -1)
    angle, relative, norm_ratio, radial, target_cosines = [], [], [], [], []
    identical, finite, loss_rows = True, True, []
    for rows in batches[:17]:
        indexes = torch.tensor(rows)
        query = source[indexes].detach().clone().requires_grad_()
        compact = head(F.normalize(query, dim=1))
        actual = analytic_interior_arcface_logits(compact, classifier, target[indexes], masks)
        reference = sharded_mask_arcface_logits(
            compact, classifier, target[indexes], masks, margin=0.3, scale=64
        )
        identical = identical and torch.equal(reference, actual)
        rank = compact.sum() * 0
        if bool((counts[target[indexes]] > 1).all()):
            positive = positives[indexes]
            width = int((positive >= 0).sum(1).max())
            rank = 8 * member_bank_rank_loss(
                compact, bank, head, positive[:, :width], indexes, live_head=False
            )
        baseline = F.cross_entropy(reference, target[indexes]) + rank
        treatment = F.cross_entropy(actual, target[indexes]) + rank
        base_gradient = torch.autograd.grad(baseline, query, retain_graph=True)[0].double()
        new_gradient = torch.autograd.grad(treatment, query)[0].double()
        base_norm, new_norm = base_gradient.norm(dim=1), new_gradient.norm(dim=1)
        valid = (
            (base_norm > 0) & (new_norm > 0) & torch.isfinite(base_norm) & torch.isfinite(new_norm)
        )
        finite = finite and bool(valid.all())
        cosine = (base_gradient * new_gradient).sum(1)[valid] / (base_norm[valid] * new_norm[valid])
        angle.extend(torch.rad2deg(torch.acos(cosine.clamp(-1, 1))).tolist())
        relative.extend(
            ((new_gradient - base_gradient).norm(dim=1)[valid] / base_norm[valid]).tolist()
        )
        norm_ratio.extend((new_norm[valid] / base_norm[valid]).tolist())
        radial.extend((new_gradient * query.detach().double()).sum(1).abs().tolist())
        with torch.no_grad():
            c = sharded_mask_arcface_logits(
                compact, classifier, target[indexes], masks, margin=0, scale=1
            )
            target_cosines.extend(c[torch.arange(len(rows)), target[indexes]].tolist())
        loss_rows.append(
            {"baseline": float(baseline.detach()), "analytic": float(treatment.detach())}
        )
    wall = time.perf_counter() - started
    criteria = {
        "authority_forward_parity": identical,
        "all_1088_finite_nonzero_gradients": finite and len(angle) == 1088,
        "material_direction": float(np.median(angle)) >= 7.5,
        "material_relative_gradient": float(np.median(relative)) >= 0.25,
        "bounded_pressure": 0.5 <= float(np.median(norm_ratio)) <= 3,
        "tangent": max(radial) < 1e-5,
        "cpu_budget": wall <= 120,
    }
    receipt = {
        "schema": "sfora-inshop-arcface-derivative-cache-v1",
        "split": "official TRAIN original fit only; no held read",
        "decision": "GO_BOUNDED_ENCODER_SMOKE" if all(criteria.values()) else "KILL",
        "claim_eligible": False,
        "criteria": criteria,
        "margin": 0.3,
        "scale": 64,
        "derivative_pole_epsilon": 1e-6,
        "source_gradient_angles_degrees": angle,
        "source_gradient_relative_changes": relative,
        "source_gradient_norm_ratios": norm_ratio,
        "target_cosines": target_cosines,
        "radial_dot_absolute": radial,
        "losses": loss_rows,
        "median_angle_degrees": float(np.median(angle)),
        "median_relative_change": float(np.median(relative)),
        "median_norm_ratio": float(np.median(norm_ratio)),
        "cpu_main_wall_seconds": wall,
        "source_sha256": sha256(Path(__file__)),
        "native_source_sha256": sha256(Path(sharded_mask_arcface_logits.__code__.co_filename)),
        "partition_sha256": PARTITION_SHA,
        "cache_sha256": SOURCE_CACHE_SHA,
        "fit_sha256": digest_rows(fit),
        "pca_sha256": pca_sha,
        "schedule_sha256": schedule_sha,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                key: receipt[key]
                for key in (
                    "decision",
                    "criteria",
                    "median_angle_degrees",
                    "median_relative_change",
                    "median_norm_ratio",
                    "cpu_main_wall_seconds",
                )
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
