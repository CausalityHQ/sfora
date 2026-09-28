#!/usr/bin/env python3
"""CPU-only fit prototype/gradient screen for a training-only source classifier."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from probe_inshop_description_alignment import shuffled_caption_names
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from score_inshop_crop_view_pair import PARTITION_SHA, bootstrap_lower, sha256
from train_sop_siglip2_compact import initialize_head_and_classifier


def unused_gradient(
    gradient: torch.Tensor, weight: torch.Tensor, unit: torch.Tensor
) -> torch.Tensor:
    """Project outside orthonormal head rows and the source normalization axis."""
    outside = gradient - (gradient @ weight.T) @ weight
    radial = unit - (unit @ weight.T) @ weight
    coefficient = (outside * radial).sum(1) / radial.square().sum(1).clamp_min(1e-12)
    return outside - coefficient[:, None] * radial


def prototype_scores(
    query: torch.Tensor,
    fit: torch.Tensor,
    classes: torch.Tensor,
    positions: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    sums = torch.zeros(int(classes.max()) + 1, fit.shape[1])
    sums.index_add_(0, classes, fit)
    positive = torch.nn.functional.normalize(sums[target] - fit[positions], dim=1)
    weights = torch.nn.functional.normalize(sums, dim=1)
    if not bool((positive.norm(dim=1) > 0).all()) or not bool((weights.norm(dim=1) > 0).all()):
        raise ValueError("source-classifier positive inventory differs")
    scores = query @ weights.T
    scores[torch.arange(len(query)), target] = (query * positive).sum(1)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("partition", "source-cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if (
        args.output.exists()
        or sha256(args.partition) != PARTITION_SHA
        or sha256(args.source_cache) != SOURCE_CACHE_SHA
    ):
        raise ValueError("source-classifier authority differs")
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    labels = tuple(row[1] for row in train)
    fit, _ = split(labels)
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("source-classifier fit split differs")
    names = sorted({labels[row] for row in fit})
    counts = Counter(labels[row] for row in fit)
    categories = {labels[row]: "/".join(Path(train[row][0]).parts[:3]) for row in fit}
    category_counts = Counter(categories.values())
    eligible = [
        name for name in names if counts[name] >= 2 and category_counts[categories[name]] >= 2
    ]
    selected = sorted(
        eligible,
        key=lambda name: hashlib.sha256(b"inshop-source-classifier-v1\0" + name.encode()).digest(),
    )[:512]
    query = [
        min(
            (row for row in fit if labels[row] == name),
            key=lambda row: hashlib.sha256(train[row][0].encode()).digest(),
        )
        for name in selected
    ]
    if len(selected) != 512 or len(names) != 2004:
        raise ValueError("source-classifier query class inventory differs")
    index = {name: position for position, name in enumerate(names)}
    positions = torch.tensor([fit.index(row) for row in query])
    classes = torch.tensor([index[labels[row]] for row in fit])
    target = torch.tensor([index[name] for name in selected])
    mapping = shuffled_caption_names(
        {name: category for name, category in categories.items() if category_counts[category] >= 2},
        np.random.default_rng(179019),
    )
    sham = torch.tensor([index[mapping[name]] for name in selected])
    values = np.load(args.source_cache, mmap_mode="r", allow_pickle=False)
    if values.shape != (25882, 1024) or values.dtype != np.float32:
        raise ValueError("source-classifier feature shape differs")
    torch.set_num_threads(8)
    torch.manual_seed(179024)
    fit_source = torch.from_numpy(values[list(fit)].copy())
    head, _, pca_sha = initialize_head_and_classifier(
        fit_source, tuple(classes.tolist()), allow_singletons=True
    )
    if pca_sha != "f387aae10a5fe080ebbc9a8ec0e77ff9e86b4812b6e1066048de1cf7cf1e4ea8":
        raise ValueError("source-classifier PCA authority differs")
    weight = head.weight.detach()
    if not torch.allclose(weight @ weight.T, torch.eye(128), rtol=0, atol=2e-5):
        raise ValueError("source-classifier PCA rows are not orthonormal")
    fit_unit = torch.nn.functional.normalize(fit_source, dim=1)
    mean = fit_unit.mean(0)
    high_fit = torch.nn.functional.normalize(fit_unit - mean, dim=1)
    low_fit = torch.nn.functional.normalize(head(fit_unit).detach(), dim=1)
    source = torch.from_numpy(values[query].copy()).requires_grad_()
    unit = torch.nn.functional.normalize(source, dim=1)
    high_query = torch.nn.functional.normalize(unit - mean, dim=1)
    low_query = torch.nn.functional.normalize(head(unit), dim=1)
    high_scores = prototype_scores(high_query, high_fit, classes, positions, target)
    low_scores = prototype_scores(low_query, low_fit, classes, positions, target)
    high_grad = torch.autograd.grad(
        torch.nn.functional.cross_entropy(64 * high_scores, target), source, retain_graph=True
    )[0]
    sham_grad = torch.autograd.grad(
        torch.nn.functional.cross_entropy(64 * high_scores, sham), source, retain_graph=True
    )[0]
    low_grad = torch.autograd.grad(
        torch.nn.functional.cross_entropy(64 * low_scores, target), source
    )[0]
    high_unused = unused_gradient(high_grad, weight, unit.detach())
    sham_unused = unused_gradient(sham_grad, weight, unit.detach())
    low_unused = unused_gradient(low_grad, weight, unit.detach())

    def norm(tensor):
        return tensor.double().norm(dim=1)

    fraction = norm(high_unused) / norm(high_grad)
    specificity = norm(high_unused - sham_unused) / norm(high_unused)
    low_residual = norm(low_unused) / norm(low_grad)
    if not all(torch.isfinite(value).all() for value in (fraction, specificity, low_residual)):
        raise ValueError("source-classifier gradient statistic undefined")
    high_hits = (high_scores.argmax(dim=1) == target).int().numpy()
    low_hits = (low_scores.argmax(dim=1) == target).int().numpy()
    delta = high_hits - low_hits
    lower, upper = (
        bootstrap_lower(delta, np.asarray(selected)),
        -bootstrap_lower(-delta, np.asarray(selected)),
    )
    criteria = {
        "classification": float(delta.mean()) >= 0.01 and lower > 0,
        "unused_identity_gradient": float(fraction.median()) >= 0.30,
        "unused_label_specificity": float(specificity.median()) >= 0.30,
        "compact_span_check": float(low_residual.max()) <= 1e-4,
    }
    wall = time.perf_counter() - started
    report = {
        "schema": "sfora-inshop-source-classifier-cached-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN fit only; 512 classes, leave-query-out positive centroids",
        "source_sha256": sha256(Path(__file__)),
        "features_sha256": SOURCE_CACHE_SHA,
        "partition_sha256": PARTITION_SHA,
        "fit_rows_sha256": digest_rows(fit),
        "pca_sha256": pca_sha,
        "query_train_rows": query,
        "sham_label_mapping": mapping,
        "source_prototype_accuracy": float(high_hits.mean()),
        "compact_prototype_accuracy": float(low_hits.mean()),
        "per_query_source_hit": high_hits.tolist(),
        "per_query_compact_hit": low_hits.tolist(),
        "gain_pp": float(100 * delta.mean()),
        "product_bootstrap_95_pp": [100 * lower, 100 * upper],
        "unused_identity_gradient_fraction_median": float(fraction.median()),
        "unused_gradient_specificity_median": float(specificity.median()),
        "compact_gradient_outside_span_max": float(low_residual.max()),
        "criteria": criteria,
        "cpu_wall_seconds": wall,
        "advance": all(criteria.values()) and wall <= 120,
        "encoder_run": False,
        "torch": str(torch.__version__),
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if not isinstance(value, list) and key != "sham_label_mapping"
            }
        )
    )


if __name__ == "__main__":
    main()
