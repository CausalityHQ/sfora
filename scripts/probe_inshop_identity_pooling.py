#!/usr/bin/env python3
"""Fixed-budget, inner-fit-only category pooling falsifier; no encoder work."""

import argparse
import copy
import json
import signal
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from compare_inshop_sop_warmstart_100 import packed_quality
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from score_inshop_crop_view_pair import bootstrap_lower, sha256
from torch import nn
from torch.nn import functional as F
from train_sop_siglip2_compact import (
    initialize_head_and_classifier,
    member_bank_initial_values,
    member_bank_positive_ordinals,
    member_bank_rank_loss,
    member_bank_refresh_rows,
)

from sfora.sop_compact_training import compact_head_features
from sfora.unicom_rank_finish import identity_balanced_batches
from sfora.unicom_training import sharded_mask_arcface_loss

SEED = 179031
STEPS = 100
FIT_SHA = "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be"
INNER_SHA = "28589624414c9bc0b084aa1187cf5285db9f18ab32ba82f75580cc0218df4776"
VALIDATION_SHA = "abc407cbfb1eb6ba4fbbf067223a829e36ee2ccb7bfa4134e45deef9991256be"
GROUP_A = {"Dresses", "Skirts", "Rompers_Jumpsuits", "Pants", "Shorts", "Denim", "Leggings"}


def shuffled_b_labels(labels, protected, seed=SEED):
    result = labels.copy()
    indexes = np.flatnonzero(~protected)
    result[indexes] = labels[np.random.default_rng(seed).permutation(indexes)]
    return result


def joint_class_ids(target, external):
    labels = tuple("inshop:" + str(i) for i in target) + tuple("sop:" + str(i) for i in external)
    mapping = {name: i for i, name in enumerate(sorted(set(labels)))}
    return np.asarray([mapping[name] for name in labels], dtype=np.int64)


def joint_self_check():
    ids = joint_class_ids(("1", "1", "2", "2"), ("1", "1", "2", "2"))
    assert len(set(ids)) == 4 and not set(ids[:4]) & set(ids[4:])
    protected = np.arange(len(ids)) < 4
    shuffled = shuffled_b_labels(ids, protected, seed=179033)
    assert np.array_equal(ids[:4], shuffled[:4]) and Counter(ids) == Counter(shuffled)
    assert np.array_equal(ids, joint_class_ids(("1", "1", "2", "2"), ("1", "1", "2", "2")))


def joint_data(inventory_path, sop_cache_path, sop_metadata_path, labels, cache, fit, outer):
    if sha256(inventory_path) != "853bdac748c891e90804a52199fd90a007e0492c7041245a650dc768fec5237b":
        raise ValueError("joint inventory authority differs")
    inventory = json.loads(inventory_path.read_text())
    if (
        sha256(sop_cache_path) != inventory["sop_features_sha256"]
        or sha256(sop_metadata_path) != inventory["sop_metadata_sha256"]
    ):
        raise ValueError("joint SOP source authority differs")
    sop = np.load(sop_cache_path, mmap_mode="r", allow_pickle=False)
    sop_rows = [line.split() for line in sop_metadata_path.read_text().splitlines()[1:]]
    target_rows = tuple(inventory["inshop_training_rows"])
    validation = tuple(inventory["inshop_validation_rows"])
    external_rows = tuple(inventory["sop_external_rows"])
    if (
        sop.shape != (59551, 1024)
        or sop.dtype != np.float32
        or len(sop_rows) != 59551
        or len(target_rows) != 6757
        or len(validation) != 6514
        or len(external_rows) != 5190
        or len(set(target_rows)) != 6757
        or len(set(validation)) != 6514
        or len(set(external_rows)) != 5190
        or not set(target_rows + validation) <= set(fit)
        or set(target_rows + validation) & set(outer)
        or set(labels[i] for i in target_rows) & set(labels[i] for i in validation)
    ):
        raise ValueError("joint fit-only inventories differ")
    source = torch.from_numpy(np.concatenate((cache[list(target_rows)], sop[list(external_rows)])))
    if not bool(torch.isfinite(source).all()):
        raise ValueError("joint source nonfinite")
    classes = joint_class_ids(
        tuple(labels[i] for i in target_rows), tuple(sop_rows[i][1] for i in external_rows)
    )
    a = np.arange(len(source)) < len(target_rows)
    if len(set(classes[a])) != 995 or len(set(classes[~a])) != 995:
        raise ValueError("joint class counts differ")
    return source, torch.from_numpy(cache[list(validation)].copy()), a, classes, validation


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--joint-self-check", action="store_true")
    parser.add_argument("--joint-inventory", type=Path)
    parser.add_argument("--sop-cache", type=Path)
    parser.add_argument("--sop-metadata", type=Path)
    parser.add_argument(
        "--inshop-cache", type=Path, default=Path("/tmp/sfora-inshop-pretrained-features.npy")
    )
    parser.add_argument(
        "--partition", type=Path, default=Path("/tmp/sfora-inshop-partition-replay.txt")
    )
    args = parser.parse_args()
    if args.joint_self_check:
        joint_self_check()
        return
    if args.output is None or (
        args.joint_inventory and (args.sop_cache is None or args.sop_metadata is None)
    ):
        parser.error("output and joint data paths required")
    seed = 179033 if args.joint_inventory else SEED
    if args.output.exists():
        raise ValueError("pooling output already exists")
    started = time.perf_counter()
    result = {
        "schema": "sfora-inshop-category-pooling-f0-v1",
        "arms": {},
        "claim_eligible": False,
        "updates_per_arm": STEPS,
        "seed": seed,
        "dataset_split": "In-Shop TRAIN original fit only; internal product-disjoint A validation",
        "serving_latency_measured": False,
        "encoder_training": False,
    }
    if args.joint_inventory:
        result.update(
            {
                "schema": "sfora-sop-inshop-joint-cached-f1-v1",
                "dataset_split": "In-Shop TRAIN fit internal holdout; external SOP TRAIN fit",
                "joint_inventory_sha256": sha256(args.joint_inventory),
                "sop_cache_sha256": sha256(args.sop_cache),
                "sop_metadata_sha256": sha256(args.sop_metadata),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def deadline(_signum, _frame):
        raise TimeoutError("fixed 120-second CPU budget exceeded; no adaptive steps")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    try:
        torch.set_num_threads(8)
        torch.manual_seed(seed)
        partition = args.partition
        cache_path = args.inshop_cache
        if sha256(partition) != PARTITION_SHA or sha256(cache_path) != SOURCE_CACHE_SHA:
            raise ValueError("pooling cache/partition authority differs")
        rows = [
            s.split() for s in partition.read_text().splitlines()[2:] if s.split()[-1] == "train"
        ]
        labels = tuple(row[1] for row in rows)
        fit, outer = split(labels)
        if digest_rows(fit) != FIT_SHA:
            raise ValueError("pooling original fit differs")
        domains = defaultdict(set)
        for path, label, _ in rows:
            domains[label].add("A" if path.split("/")[2] in GROUP_A else "B")
        eligible = tuple(i for i in fit if len(domains[labels[i]]) == 1)
        inner, held = split(tuple(labels[i] for i in eligible))
        training = tuple(eligible[i] for i in inner)
        validation = tuple(eligible[i] for i in held if domains[labels[eligible[i]]] == {"A"})
        if not args.joint_inventory and (
            digest_rows(training) != INNER_SHA or digest_rows(validation) != VALIDATION_SHA
        ):
            raise ValueError("pooling internal split differs")
        counts = Counter(labels[i] for i in training)
        # Native member-bank anchors require another image of the same product.
        training = tuple(i for i in training if counts[labels[i]] >= 2)
        assert set(training).isdisjoint(outer) and set(validation).isdisjoint(outer)
        assert set(labels[i] for i in training).isdisjoint(labels[i] for i in validation)
        cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        if cache.shape != (25882, 1024) or cache.dtype != np.float32:
            raise ValueError("pooling source geometry differs")
        source = torch.from_numpy(cache[list(training)].copy())
        validation_source = torch.from_numpy(cache[list(validation)].copy())
        a = np.asarray([domains[labels[i]] == {"A"} for i in training])
        a_indexes, b_indexes = np.flatnonzero(a), np.flatnonzero(~a)
        names = sorted(set(labels[i] for i in training))
        mapping = {label: i for i, label in enumerate(names)}
        classes = np.asarray([mapping[labels[i]] for i in training], dtype=np.int64)
        if args.joint_inventory:
            source, validation_source, a, classes, validation = joint_data(
                args.joint_inventory, args.sop_cache, args.sop_metadata, labels, cache, fit, outer
            )
            training = tuple(range(len(source)))
            a_indexes, b_indexes = np.flatnonzero(a), np.flatnonzero(~a)
        sham = shuffled_b_labels(classes, a, seed=seed)
        assert np.array_equal(sham[a], classes[a]) and Counter(sham) == Counter(classes)
        if args.joint_inventory and float(np.mean(sham[~a] != classes[~a])) < 0.9:
            raise ValueError("joint sham changes too few external labels")
        common, _, pca_sha = initialize_head_and_classifier(source[a_indexes], tuple(classes[a]))
        batches_a = identity_balanced_batches(
            tuple(str(i) for i in classes[a]),
            batch_size=64,
            images_per_identity=4,
            seed=seed,
            epoch=1,
            steps=STEPS,
            coverage_first=True,
        )
        batches_b = identity_balanced_batches(
            tuple(str(i) for i in classes[~a]),
            batch_size=32,
            images_per_identity=4,
            seed=seed + 1,
            epoch=1,
            steps=STEPS,
            coverage_first=True,
        )
        pooled_batches = tuple(
            tuple(a_indexes[list(x[:32])]) + tuple(b_indexes[list(y)])
            for x, y in zip(batches_a, batches_b, strict=True)
        )
        validation_labels = tuple(labels[i] for i in validation)
        members = defaultdict(list)
        for i, label in enumerate(validation_labels):
            members[label].append(i)
        query = sorted(i for group in members.values() for i in group[::2])
        gallery = sorted(i for group in members.values() for i in group[1::2])
        assert all(len(group) >= 2 for group in members.values())
        if args.joint_inventory and (len(query), len(gallery), len(members)) != (3440, 3074, 997):
            raise ValueError("joint internal held roles differ")
        result.update(
            {
                "source_cache_sha256": SOURCE_CACHE_SHA,
                "partition_sha256": PARTITION_SHA,
                "source_sha256": sha256(Path(__file__)),
                "helper_source_sha256": {
                    f.__module__ + "." + f.__name__: sha256(Path(f.__code__.co_filename))
                    for f in (
                        initialize_head_and_classifier,
                        compact_head_features,
                        sharded_mask_arcface_loss,
                        member_bank_rank_loss,
                        member_bank_positive_ordinals,
                        member_bank_refresh_rows,
                        packed_quality,
                        bootstrap_lower,
                        identity_balanced_batches,
                    )
                },
                "initial_pca_sha256": pca_sha,
                "training_rows_sha256": digest_rows(training),
                "validation_rows_sha256": digest_rows(validation),
                "query_rows_sha256": digest_rows(tuple(validation[i] for i in query)),
                "gallery_rows_sha256": digest_rows(tuple(validation[i] for i in gallery)),
                "query_count": len(query),
                "gallery_count": len(gallery),
                "validation_products": len(members),
                "a_fit_products": len(set(classes[a])),
                "b_fit_products": len(set(classes[~a])),
                "sham_changed_fraction": float(np.mean(sham[~a] != classes[~a])),
                "pooled_feature_schedule_sha256": digest_rows(
                    tuple(i for batch in pooled_batches for i in batch)
                ),
            }
        )
        terminal_values = {}
        for name in ("control", "pooled", "sham"):
            arm_started = time.perf_counter()
            arm_source = source[a_indexes] if name == "control" else source
            arm_classes = (
                classes[a] if name == "control" else (classes if name == "pooled" else sham)
            )
            arm_names = sorted(set(arm_classes.tolist()))
            ids = torch.tensor([arm_names.index(i) for i in arm_classes], dtype=torch.long)
            head = copy.deepcopy(common)
            with torch.no_grad():
                initial = compact_head_features(arm_source, head)
                sums = torch.zeros(len(arm_names), 128).index_add_(0, ids, initial)
            proxies = nn.Parameter(F.normalize(sums, dim=1))
            bank = member_bank_initial_values(arm_source, head, live_head=False)
            positives = member_bank_positive_ordinals(ids.numpy())
            batches = batches_a if name == "control" else pooled_batches
            optimizer = torch.optim.AdamW(
                [head.weight, head.bias, proxies], lr=1e-4, weight_decay=0.05
            )
            masks = torch.arange(128).unsqueeze(0)
            losses, gradients = [], []
            for step, batch in enumerate(batches, 1):
                indexes = torch.tensor(batch, dtype=torch.long)
                optimizer.zero_grad(set_to_none=True)
                projected = compact_head_features(arm_source[indexes], head)
                control = sharded_mask_arcface_loss(
                    projected, proxies, ids[indexes], masks, margin=0.3, scale=64.0
                )
                positive = positives[indexes]
                width = int((positive >= 0).sum(1).max())
                rank = member_bank_rank_loss(
                    projected, bank, head, positive[:, :width], indexes, live_head=False
                )
                loss = control + 8 * rank
                if not bool(torch.isfinite(loss)):
                    raise ValueError("pooling loss is nonfinite")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [head.weight, head.bias, proxies], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                refresh, positions = member_bank_refresh_rows(tuple(int(i) for i in batch))
                bank[list(refresh)] = F.normalize(projected.detach()[list(positions)], dim=1)
                if not bool(torch.isfinite(bank).all()):
                    raise ValueError("pooling bank is nonfinite")
                losses.append(float(loss.detach()))
                gradients.append(float(norm))
                if step in (1, 50, STEPS):
                    print(
                        json.dumps(
                            {
                                "arm": name,
                                "step": step,
                                "loss": losses[-1],
                                "elapsed": time.perf_counter() - started,
                            }
                        ),
                        flush=True,
                    )
            with torch.no_grad():
                terminal_values[name] = F.normalize(
                    compact_head_features(validation_source, head), dim=1
                ).numpy()
            result["arms"][name] = {
                "updates": len(losses),
                "losses": losses,
                "gradient_norms": gradients,
                "train_and_init_seconds": time.perf_counter() - arm_started,
                "bank_rows": len(bank),
                "parameters": sum(p.numel() for p in [head.weight, head.bias, proxies]),
            }
        # Never compare a partially trained campaign.
        for name, values in terminal_values.items():
            result["arms"][name]["quality"] = packed_quality(
                values, validation_labels, query, gallery, device=torch.device("cpu")
            )
        qlabels = np.asarray([validation_labels[i] for i in query])
        deltas, criteria = {}, {}
        for baseline, floor in (("control", 0.003), ("sham", 0.002)):
            candidate = result["arms"]["pooled"]["quality"]
            reference = result["arms"][baseline]["quality"]
            delta = np.asarray(candidate["per_query_r1"]) - np.asarray(reference["per_query_r1"])
            lower, upper = bootstrap_lower(delta, qlabels), -bootstrap_lower(-delta, qlabels)
            ap_delta = candidate["map_at_r"] - reference["map_at_r"]
            deltas[baseline] = {
                "recall_delta": float(delta.mean()),
                "lower95": lower,
                "upper95": upper,
                "map_delta": ap_delta,
            }
            criteria[baseline] = float(delta.mean()) >= floor and lower > 0 and ap_delta >= 0
        result.update(
            {
                "deltas": deltas,
                "criteria": criteria,
                "decision": "GO_JOINT_ENCODER_SMOKE_DESIGN"
                if all(criteria.values())
                else (
                    "KILL_ACTUAL_JOINT_CACHED_PROXY"
                    if args.joint_inventory
                    else "KILL_CATEGORY_POOLING_PROXY"
                ),
            }
        )
    except TimeoutError as error:
        result.update({"decision": "KILL_CPU_FEASIBILITY", "error": str(error)})
    except Exception as error:
        result.update({"decision": "KILL_INSTRUMENT", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        signal.alarm(0)
        result["cpu_wall_seconds"] = time.perf_counter() - started
        args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "arms"}), flush=True)


if __name__ == "__main__":
    main()
