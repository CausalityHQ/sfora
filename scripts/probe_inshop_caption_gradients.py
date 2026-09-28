#!/usr/bin/env python3
"""Cached CPU rejection gate for fixed soft caption supervision."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from score_inshop_crop_view_pair import PARTITION_SHA, bootstrap_lower, sha256
from train_sop_siglip2_compact import initialize_head_and_classifier

from sfora.frozen_text_supervision import frozen_text_loss
from sfora.sop_compact_training import compact_head_features
from sfora.unicom_training import sharded_mask_arcface_loss

TEACHER_SHA = "d20a12cdbe1428f52ec192d182046f5f9882a2ba2e42ff82ec3aeac89a40c975"
ALIGNMENT_SHA = "b6f77d2eaacac7e170e2d688b03db9e1ff3455f1ebfaed5829e107856f554aab"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("partition", "teacher", "source-cache", "alignment", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if args.output.exists() or any(
        sha256(path) != digest
        for path, digest in (
            (args.partition, PARTITION_SHA),
            (args.teacher, TEACHER_SHA),
            (args.source_cache, SOURCE_CACHE_SHA),
            (args.alignment, ALIGNMENT_SHA),
        )
    ):
        raise ValueError("caption gradient authority differs")
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    labels = tuple(row[1] for row in train)
    fit, _ = split(labels)
    cache = torch.load(args.teacher, map_location="cpu", weights_only=True)
    names = tuple(sorted({labels[row] for row in fit}))
    teacher_names = cache["labels"]
    text = cache["features"]
    if (
        cache["fit_rows_sha256"] != digest_rows(fit)
        or teacher_names != sorted(set(teacher_names))
        or len(set(names) - set(teacher_names)) != 1
        or not set(teacher_names) <= set(names)
        or text.shape != (2003, 1024)
        or text.dtype != torch.float32
        or not torch.isfinite(text).all()
        or not torch.allclose(text.norm(dim=1), torch.ones(2003), atol=2e-6, rtol=0)
    ):
        raise ValueError("caption teacher fit geometry differs")
    alignment = json.loads(args.alignment.read_text())
    query = [row["query_train_row"] for row in alignment["manifest"]]
    if (
        len(query) != 512
        or len({labels[row] for row in query}) != 512
        or not set(query) <= set(fit)
    ):
        raise ValueError("caption cached query inventory differs")
    teacher_index = {name: ordinal for ordinal, name in enumerate(teacher_names)}
    class_index = {name: ordinal for ordinal, name in enumerate(names)}
    class_map = np.asarray([teacher_index.get(name, -1) for name in names], dtype="<i4")
    source_cache = np.load(args.source_cache, mmap_mode="r", allow_pickle=False)
    if source_cache.shape != (25882, 1024) or source_cache.dtype != np.float32:
        raise ValueError("caption cached source shape differs")
    torch.set_num_threads(8)
    torch.manual_seed(179024)
    head, classifier, pca_sha = initialize_head_and_classifier(
        torch.from_numpy(source_cache[list(fit)].copy()),
        tuple(class_index[labels[row]] for row in fit),
        allow_singletons=True,
    )
    source = torch.from_numpy(source_cache[query].copy()).requires_grad_()
    target = torch.tensor([teacher_index[labels[row]] for row in query])
    sham = torch.tensor(
        [teacher_index[alignment["shuffled_caption_names"][labels[row]]] for row in query]
    )
    true_ce = frozen_text_loss(source, text, target, reduction="none")
    sham_ce = frozen_text_loss(source, text, sham, reduction="none")
    true_grad = torch.autograd.grad(0.1 * true_ce.mean(), source)[0]
    sham_grad = torch.autograd.grad(0.1 * sham_ce.mean(), source)[0]
    base_loss = sharded_mask_arcface_loss(
        compact_head_features(source, head),
        classifier,
        torch.tensor([class_index[labels[row]] for row in query]),
        torch.arange(128).unsqueeze(0),
        margin=0.3,
        scale=64.0,
    )
    base_grad = torch.autograd.grad(base_loss, source)[0]
    specific = true_grad - sham_grad
    distinct = specific.norm(dim=1) / true_grad.norm(dim=1)
    cosine = torch.nn.functional.cosine_similarity(specific, base_grad, dim=1)
    similarity = torch.nn.functional.normalize(source.detach(), dim=1) @ text.T
    order = similarity.argsort(dim=1, descending=True, stable=True)
    ranks = order.argsort(dim=1)
    rank_delta = (ranks[torch.arange(512), sham] - ranks[torch.arange(512), target]).numpy()
    products = np.asarray([labels[row] for row in query])
    ce_delta = (sham_ce - true_ce).detach().numpy()
    ce_lower, rank_lower = (
        bootstrap_lower(ce_delta, products),
        bootstrap_lower(rank_delta, products),
    )
    residual = text.double() - text.double().mean(dim=0)
    covariance = residual.T @ residual
    effective_rank = float(covariance.trace().square() / covariance.square().sum())
    categories = len({"/".join(Path(train[row][0]).parts[:3]) for row in fit})
    neighbors = text @ text.T
    neighbors.fill_diagonal_(-torch.inf)
    med_distinct = float(distinct.median())
    med_cosine = float(cosine.median())
    negative_fraction = float((cosine < 0).float().mean())
    criteria = {
        "C1": ce_lower > 0 and rank_lower > 0,
        "C2": med_distinct >= 0.30,
        "C3": med_cosine >= 0 and negative_fraction <= 0.60,
        "C4": effective_rank > 2 * categories,
    }
    wall = time.perf_counter() - started
    report = {
        "schema": "sfora-inshop-caption-cached-gradients-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN fit only; 512 preselected distinct products",
        "source_sha256": sha256(Path(__file__)),
        "teacher_sha256": TEACHER_SHA,
        "features_sha256": SOURCE_CACHE_SHA,
        "alignment_sha256": ALIGNMENT_SHA,
        "loss_source_sha256": sha256(Path(frozen_text_loss.__code__.co_filename)),
        "pca_sha256": pca_sha,
        "class_to_caption_map_sha256": hashlib.sha256(class_map.tobytes()).hexdigest(),
        "mean_true_minus_sham_ce_gain": float(ce_delta.mean()),
        "ce_gain_lower95": ce_lower,
        "mean_true_rank_gain": float(rank_delta.mean()),
        "rank_gain_lower95": rank_lower,
        "caption_specific_gradient_ratio_median": med_distinct,
        "caption_specific_base_cosine_median": med_cosine,
        "negative_cosine_fraction": negative_fraction,
        "weighted_true_aux_base_gradient_ratio_median": float(
            (true_grad.norm(dim=1) / base_grad.norm(dim=1)).median()
        ),
        "text_residual_effective_rank": effective_rank,
        "clothing_category_groups": categories,
        "exact_duplicate_prototypes": len(text) - len(torch.unique(text, dim=0)),
        "nearest_text_cosine_median": float(neighbors.max(dim=1).values.median()),
        "criteria": criteria,
        "cpu_wall_seconds": wall,
        "advance": all(criteria.values()) and wall <= 120,
        "encoder_training_or_export": False,
        "torch": str(torch.__version__),
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
