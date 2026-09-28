#!/usr/bin/env python3
"""One fixed inner-fit CPU quality smoke for repeated coverage sampling."""

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
from preflight_inshop_siglip2_unseen_gallery import digest_rows, digest_schedule, split
from probe_inshop_sop_product_prior import SOURCE_CACHE_SHA
from score_inshop_crop_view_pair import PARTITION_SHA, bootstrap_lower, sha256
from torch import nn
from torch.nn import functional as F
from train_inshop_siglip2_unseen_gallery import half_fit_products
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

SEED, STEPS = 179032, 400


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("repeat coverage output already exists")
    started = time.perf_counter()
    result = {
        "schema": "sfora-repeat-coverage-cached-f1-v1",
        "arms": {},
        "claim_eligible": False,
        "seed": SEED,
        "updates_per_arm": STEPS,
        "dataset_split": "In-Shop original TRAIN-fit only; internal product-disjoint validation",
        "encoder_training": False,
        "serving_latency_measured": False,
    }

    def deadline(_signal, _frame):
        raise TimeoutError("120-second fixed CPU budget exceeded")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.set_num_threads(8)
        torch.manual_seed(SEED)
        partition = Path("/tmp/sfora-inshop-partition-replay.txt")
        cache_path = Path("/tmp/sfora-inshop-pretrained-features.npy")
        if sha256(partition) != PARTITION_SHA or sha256(cache_path) != SOURCE_CACHE_SHA:
            raise ValueError("repeat coverage cache authority differs")
        rows = [
            s.split() for s in partition.read_text().splitlines()[2:] if s.split()[-1] == "train"
        ]
        labels = tuple(row[1] for row in rows)
        fit, outer = split(labels)
        if digest_rows(fit) != "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be":
            raise ValueError("repeat coverage original fit differs")
        inner = half_fit_products(labels, fit)
        names = set(labels[i] for i in inner)
        validation = tuple(i for i in fit if labels[i] not in names)
        counts = Counter(labels[i] for i in inner)
        inner = tuple(i for i in inner if counts[labels[i]] >= 2)
        counts = Counter(labels[i] for i in validation)
        validation = tuple(i for i in validation if counts[labels[i]] >= 2)
        assert set(inner).isdisjoint(outer) and set(validation).isdisjoint(outer)
        assert set(labels[i] for i in inner).isdisjoint(labels[i] for i in validation)
        cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
        assert cache.shape == (25882, 1024) and cache.dtype == np.float32
        source = torch.from_numpy(cache[list(inner)].copy())
        held = torch.from_numpy(cache[list(validation)].copy())
        fit_labels = tuple(labels[i] for i in inner)
        classes = {label: i for i, label in enumerate(sorted(set(fit_labels)))}
        ids = torch.tensor([classes[label] for label in fit_labels])
        common, proxy, pca_sha = initialize_head_and_classifier(source, tuple(ids.tolist()))
        positive_table = member_bank_positive_ordinals(ids.numpy())
        schedules = {
            name: identity_balanced_batches(
                fit_labels,
                batch_size=64,
                images_per_identity=4,
                seed=SEED,
                epoch=1,
                steps=STEPS,
                coverage_first=True,
                repeat_coverage=repeat,
            )
            for name, repeat in [("control", False), ("repeated", True)]
        }
        boundary = next(
            i for i, (a, b) in enumerate(zip(*schedules.values(), strict=True)) if a != b
        )
        assert boundary > 17
        held_labels = tuple(labels[i] for i in validation)
        members = defaultdict(list)
        for i, label in enumerate(held_labels):
            members[label].append(i)
        query = sorted(i for group in members.values() for i in group[::2])
        gallery = sorted(i for group in members.values() for i in group[1::2])
        result.update(
            {
                "source_sha256": sha256(Path(__file__)),
                "sampler_source_sha256": sha256(
                    Path(identity_balanced_batches.__code__.co_filename)
                ),
                "source_cache_sha256": SOURCE_CACHE_SHA,
                "partition_sha256": PARTITION_SHA,
                "initial_pca_sha256": pca_sha,
                "training_rows_sha256": digest_rows(inner),
                "validation_rows_sha256": digest_rows(validation),
                "training_rows": len(inner),
                "training_products": len(classes),
                "query_count": len(query),
                "gallery_count": len(gallery),
                "validation_products": len(members),
                "first_changed_update": boundary + 1,
            }
        )
        terminal = {}
        for name, batches in schedules.items():
            arm_started = time.perf_counter()
            head = copy.deepcopy(common)
            classifier = nn.Parameter(proxy.detach().clone())
            bank = member_bank_initial_values(source, head, live_head=False)
            optimizer = torch.optim.AdamW(
                [head.weight, head.bias, classifier], lr=1e-4, weight_decay=0.05
            )
            masks = torch.arange(128).unsqueeze(0)
            losses, gradients = [], []
            for step, batch in enumerate(batches, 1):
                indexes = torch.tensor(batch)
                optimizer.zero_grad(set_to_none=True)
                projected = compact_head_features(source[indexes], head)
                arcface = sharded_mask_arcface_loss(
                    projected, classifier, ids[indexes], masks, margin=0.3, scale=64.0
                )
                positive = positive_table[indexes]
                width = int((positive >= 0).sum(1).max())
                rank = member_bank_rank_loss(
                    projected, bank, head, positive[:, :width], indexes, live_head=False
                )
                loss = arcface + 8 * rank
                assert bool(torch.isfinite(loss))
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    [head.weight, head.bias, classifier], 1.0, error_if_nonfinite=True
                )
                optimizer.step()
                refresh, positions = member_bank_refresh_rows(batch)
                bank[list(refresh)] = F.normalize(projected.detach()[list(positions)], dim=1)
                assert bool(torch.isfinite(bank).all())
                losses.append(float(loss.detach()))
                gradients.append(float(norm))
                if step in (1, 100, 200, 400):
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
                terminal[name] = F.normalize(compact_head_features(held, head), dim=1).numpy()
            result["arms"][name] = {
                "updates": len(losses),
                "losses": losses,
                "gradient_norms": gradients,
                "schedule_sha256": digest_schedule(batches),
                "training_and_init_seconds": time.perf_counter() - arm_started,
            }
        assert (
            result["arms"]["control"]["losses"][:boundary]
            == result["arms"]["repeated"]["losses"][:boundary]
        )
        result["identical_prefix_losses"] = boundary
        for name, values in terminal.items():
            result["arms"][name]["quality"] = packed_quality(
                values, held_labels, query, gallery, device=torch.device("cpu")
            )
        qlabels = np.asarray([held_labels[i] for i in query])
        deltas = {}
        for metric, vector in [("recall", "per_query_r1"), ("map", "per_query_ap")]:
            a = result["arms"]["repeated"]["quality"][vector]
            b = result["arms"]["control"]["quality"][vector]
            delta = np.asarray(a) - np.asarray(b)
            deltas[metric] = {
                "point": float(delta.mean()),
                "lower95": bootstrap_lower(delta, qlabels),
                "upper95": -bootstrap_lower(-delta, qlabels),
            }
        criteria = {
            "recall": deltas["recall"]["point"] >= 0.003 and deltas["recall"]["lower95"] > 0,
            "map": deltas["map"]["point"] >= 0.002 and deltas["map"]["lower95"] > 0,
        }
        result.update(
            {
                "deltas": deltas,
                "criteria": criteria,
                "decision": "GO_ENCODER_BOUNDARY_SMOKE_DESIGN"
                if all(criteria.values())
                else "KILL_CACHED_REPEAT_COVERAGE_PROXY",
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
        args.output.write_text(json.dumps(result, allow_nan=False, sort_keys=True) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "arms"}), flush=True)


if __name__ == "__main__":
    main()
