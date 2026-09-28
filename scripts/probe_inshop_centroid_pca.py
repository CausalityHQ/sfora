#!/usr/bin/env python3
"""One fixed TRAIN-fit supervised-subspace initialization falsifier."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_unseen_gallery import digest_rows, split
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from probe_inshop_source_classifier import prototype_scores
from score_inshop_crop_view_pair import PARTITION_SHA, bootstrap_lower, sha256
from torch.nn import functional as F

from sfora.representation_ceiling import fit_centered_pca


def centroid_pca(values, classes, dimensions):
    count = torch.bincount(classes)
    if not bool((count > 0).all()):
        raise ValueError("centroid basis has missing product")
    means = torch.zeros(len(count), values.shape[1], dtype=values.dtype)
    means.index_add_(0, classes, values)
    return fit_centered_pca(means / count[:, None], dimensions=dimensions)


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("partition", "source-cache", "panel", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    torch.set_num_threads(8)
    if (
        args.output.exists()
        or sha256(args.partition) != PARTITION_SHA
        or sha256(args.source_cache) != SOURCE_CACHE_SHA
        or sha256(args.panel) != "ace3f13e92bc0357c195b307da3fde128780f33a9f7a9a8c12af7b0b8200f29f"
    ):
        raise ValueError("centroid PCA authority differs")
    train = [
        line.split()
        for line in args.partition.read_text().splitlines()[2:]
        if line.split()[-1] == "train"
    ]
    labels = tuple(row[1] for row in train)
    fit, _ = split(labels)
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("centroid PCA fit differs")
    queries = json.loads(args.panel.read_text())["query_train_rows"]
    if (
        len(queries) != 512
        or len(set(queries)) != 512
        or not set(queries).issubset(fit)
        or len({labels[i] for i in queries}) != 512
    ):
        raise ValueError("centroid PCA panel differs")
    mapping = {label: i for i, label in enumerate(sorted({labels[i] for i in fit}))}
    target = torch.tensor([mapping[labels[i]] for i in fit])
    positions = torch.tensor([fit.index(i) for i in queries])
    basis_rows = torch.tensor([i for i, row in enumerate(fit) if row not in set(queries)])
    cache = np.load(args.source_cache, mmap_mode="r", allow_pickle=False)
    if cache.shape != (25882, 1024) or cache.dtype != np.float32:
        raise ValueError("centroid PCA cache geometry differs")
    unit = F.normalize(torch.from_numpy(cache[list(fit)].copy()), dim=1)
    basis = unit[basis_rows]
    classes = target[basis_rows]
    generator = torch.Generator().manual_seed(179024)
    shuffled = classes[torch.randperm(len(classes), generator=generator)]
    assert torch.equal(torch.bincount(classes), torch.bincount(shuffled))
    heads = {
        "native": fit_centered_pca(basis, dimensions=128),
        "centroid": centroid_pca(basis, classes, 128),
        "sham": centroid_pca(basis, shuffled, 128),
    }
    hits, evidence = {}, {}
    for name, head in heads.items():
        values = F.normalize(head.apply(unit), dim=1)
        if not bool(torch.isfinite(values).all()) or not bool((values.norm(dim=1) > 0).all()):
            raise ValueError("centroid PCA output differs")
        scores = prototype_scores(values[positions], values, target, positions, target[positions])
        hits[name] = (scores.argmax(1) == target[positions]).int().numpy()
        evidence[name] = {
            "basis_sha256": hashlib.sha256(head.components.numpy().tobytes()).hexdigest(),
            "mean_sha256": hashlib.sha256(head.mean.numpy().tobytes()).hexdigest(),
            "hits": int(hits[name].sum()),
            "per_query_hit": hits[name].tolist(),
        }
    qlabels = np.asarray([labels[i] for i in queries])
    criteria, deltas = {}, {}
    for baseline in ("native", "sham"):
        delta = hits["centroid"] - hits[baseline]
        lower = bootstrap_lower(delta, qlabels)
        upper = -bootstrap_lower(-delta, qlabels)
        deltas[baseline] = {"point": float(delta.mean()), "lower95": lower, "upper95": upper}
        criteria[baseline] = float(delta.mean()) >= 0.01 and lower > 0
    criteria["cpu_budget"] = time.perf_counter() - started <= 120
    result = {
        "schema": "sfora-inshop-centroid-pca-init-v1",
        "claim_eligible": False,
        "decision": "GO_ENCODER_SMOKE_DESIGN" if all(criteria.values()) else "KILL",
        "criteria": criteria,
        "arms": evidence,
        "deltas": deltas,
        "query_train_rows": queries,
        "split": "officialTRAIN fit only",
        "basis_train_rows_sha256": digest_rows(tuple(fit[i] for i in basis_rows.tolist())),
        "source_cache_sha256": SOURCE_CACHE_SHA,
        "partition_sha256": PARTITION_SHA,
        "panel_sha256": sha256(args.panel),
        "source_sha256": sha256(Path(__file__)),
        "cpu_wall_seconds": time.perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("arms", "query_train_rows")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
