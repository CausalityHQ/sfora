#!/usr/bin/env python3
"""Read-only fit-only joint-data audit; no feature values or quality reads."""

import argparse
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
from export_unicom_sop_embeddings import _parse_split
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from score_inshop_crop_view_pair import sha256
from train_inshop_siglip2_unseen_gallery import half_fit_products
from train_sop_siglip2_compact import MODEL_HASHES, ordered_rows_sha256

from sfora.representation_ceiling import deterministic_class_partition


def select_external(labels, eligible, count):
    names = sorted(
        set(labels[i] for i in eligible),
        key=lambda name: (hashlib.sha256(f"sfora-joint-f0-179033:{name}".encode()).digest(), name),
    )
    if len(names) < count:
        raise ValueError("external identity inventory too small")
    chosen = set(names[:count])
    return tuple(i for i in eligible if labels[i] in chosen)


def self_check():
    labels = (1, 1, 2, 2, 3, 3)
    selected = select_external(labels, (0, 1, 2, 3), 1)
    assert len(selected) == 2 and len({labels[i] for i in selected}) == 1
    assert not set(selected) & {4, 5}
    assert selected == select_external(labels, (3, 2, 1, 0), 1)[::-1]
    try:
        select_external(labels, (0, 1), 2)
    except ValueError:
        pass
    else:
        raise AssertionError("missing identity accepted")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--self-check", action="store_true")
    parser.add_argument("--sop-root", type=Path)
    parser.add_argument("--sop-cache", type=Path)
    parser.add_argument("--partition", type=Path)
    parser.add_argument("--inshop-cache", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    self_check()
    if args.self_check:
        return
    if any(
        v is None
        for v in (args.sop_root, args.sop_cache, args.partition, args.inshop_cache, args.output)
    ):
        parser.error("all data paths and output required")
    if args.output.exists():
        raise ValueError("audit receipt already exists")
    started = time.perf_counter()
    sop = _parse_split(args.sop_root, "train")
    labels = tuple(row.label for row in sop)
    receipt = json.loads((args.sop_cache.parent / "receipt.json").read_text())
    row_sha = ordered_rows_sha256(
        np.asarray([r.image_id for r in sop]),
        np.asarray(labels),
        np.asarray([r.relative_path for r in sop]),
    )
    if (
        receipt.get("schema") != "sfora-sop-siglip2-train-feature-export-v1"
        or receipt.get("full_train") is not True
        or receipt.get("model_file_sha256") != MODEL_HASHES
        or receipt.get("ordered_rows_sha256") != row_sha
        or receipt.get("features_sha256")
        != "d15f76e459b90836df2807260a34d06692e2f2e5b8ba17b3449ed56500b8357a"
        or sha256(args.sop_cache) != receipt["features_sha256"]
        or sha256(args.partition) != PARTITION_SHA
        or sha256(args.inshop_cache) != SOURCE_CACHE_SHA
    ):
        raise ValueError("joint source authority differs")
    rows = [
        r.split() for r in args.partition.read_text().splitlines()[2:] if r.split()[-1] == "train"
    ]
    inshop_labels = tuple(r[1] for r in rows)
    fit, outer = split(inshop_labels)
    if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
        raise ValueError("original In-Shop fit differs")
    inner = half_fit_products(inshop_labels, fit)
    counts = Counter(inshop_labels[i] for i in inner)
    training = tuple(i for i in inner if counts[inshop_labels[i]] >= 2)
    inner_names = {inshop_labels[i] for i in inner}
    validation = tuple(i for i in fit if inshop_labels[i] not in inner_names)
    counts_held = Counter(inshop_labels[i] for i in validation)
    validation = tuple(i for i in validation if counts_held[inshop_labels[i]] >= 2)
    sop_split = deterministic_class_partition(labels, fit_fraction=0.9, seed=179019)
    external = select_external(
        labels, sop_split.fit_row_indexes, len(set(inshop_labels[i] for i in training))
    )
    a_names = {"inshop:" + inshop_labels[i] for i in training}
    b_names = {"sop:" + str(labels[i]) for i in external}
    positives = max(Counter(inshop_labels[i] for i in training).values()) - 1
    positives = max(positives, max(Counter(labels[i] for i in external).values()) - 1)
    shapes = [np.load(p, mmap_mode="r").shape for p in (args.sop_cache, args.inshop_cache)]
    criteria = {
        "source_shapes": shapes == [(59551, 1024), (25882, 1024)],
        "target_inner_inventory": len(training) == 6757
        and len(a_names) == 995
        and len(validation) == 6514,
        "target_product_disjoint": not {inshop_labels[i] for i in training}
        & {inshop_labels[i] for i in validation},
        "outer_held_unread": not set(training + validation) & set(outer),
        "external_fit_only": not set(external) & set(sop_split.validation_row_indexes),
        "external_class_matched": len(b_names) == len(a_names),
        "label_namespaces_disjoint": not a_names & b_names,
        "bounded_bank": len(training) + len(external) <= 20000 and positives <= 32,
        "bounded_wall": time.perf_counter() - started <= 60,
    }
    result = {
        "schema": "sfora-sop-inshop-joint-inventory-f0-v1",
        "claim_eligible": False,
        "decision": "GO_CACHED_JOINT_DESIGN" if all(criteria.values()) else "KILL_DATA_READINESS",
        "criteria": criteria,
        "audit_seconds": time.perf_counter() - started,
        "inshop_training_rows": list(training),
        "inshop_validation_rows": list(validation),
        "sop_external_rows": list(external),
        "inshop_products": len(a_names),
        "sop_products": len(b_names),
        "max_other_positives": positives,
        "sop_metadata_sha256": sha256(args.sop_root / "Ebay_train.txt"),
        "sop_ordered_rows_sha256": row_sha,
        "sop_features_sha256": receipt["features_sha256"],
        "inshop_features_sha256": SOURCE_CACHE_SHA,
        "partition_sha256": PARTITION_SHA,
        "source_sha256": sha256(Path(__file__)),
        "scope": "metadata/file hashes only; no pixels, features, gradients, quality or serving",
        "cross_dataset_pixel_duplicates": "not audited; namespaces are not a pixel-duplicate proof",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if not k.endswith("_rows")}))


if __name__ == "__main__":
    main()
