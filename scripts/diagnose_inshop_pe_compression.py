#!/usr/bin/env python3
"""Fixed CPU-only source1024/PCA128 counterfactual on authenticated TRAIN-fit caches."""

import argparse
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import probe_inshop_pe_training_smoke as smoke
from audit_inshop_pe_fit_features import PREFLIGHT_SHA
from compare_inshop_sop_warmstart_100 import packed_quality
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, split
from score_inshop_crop_view_pair import bootstrap_lower, roles
from sfora.unicom_inshop import parse_inshop_partition
from train_inshop_pe_pair import CACHE_SHA


def retained_variance(source, components):
    assert torch.isfinite(source).all() and (source.norm(dim=1) > 0).all()
    assert torch.allclose(
        components @ components.T, torch.eye(len(components)), atol=1e-5, rtol=0
    )
    unit = F.normalize(source, dim=1).double()
    centered = unit - unit.mean(0)
    ratio = float(
        (centered @ components.double().T).square().sum() / centered.square().sum()
    )
    assert np.isfinite(ratio) and 0 <= ratio <= 1 + 1e-6
    return ratio


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("source-cache", "training-dir", "dataset-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    started = time.perf_counter()
    assert (
        os.environ.get("CUDA_VISIBLE_DEVICES") == "" and not torch.cuda.is_available()
    )
    torch.set_num_threads(8)
    assert smoke.sha(args.source_cache / "receipt.json") == CACHE_SHA
    assert smoke.sha(args.source_cache / "preflight.json") == PREFLIGHT_SHA
    assert (
        smoke.sha(args.training_dir / "receipt.json")
        == "d02a1c176f023e81bd9a0bb526ff3294e677e88dcc0724b9306e1ca884177dbb"
    )
    assert (
        smoke.sha(args.training_dir / "preflight.json")
        == "41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293"
    )
    assert (
        smoke.sha(args.dataset_root / "Eval/list_eval_partition.txt") == PARTITION_SHA
    )
    stop = json.loads((args.training_dir / "receipt.json").read_text())
    assert stop["decision"] == "STOP" and not stop["advance"]
    frozen = json.loads((args.training_dir / "preflight.json").read_text())
    assert (
        smoke.sha(args.training_dir / "initializers.npz")
        == frozen["initializers_sha256"]
    )
    cache = json.loads((args.source_cache / "receipt.json").read_text())
    manifest = json.loads((args.source_cache / "preflight.json").read_text())[
        "manifest"
    ]
    rows = tuple(
        r for r in parse_inshop_partition(args.dataset_root) if r.split == "train"
    )
    fit, held = split(tuple(r.label for r in rows))
    assert tuple(m["train_row"] for m in manifest) == fit and len(fit) == 13283
    labels = tuple(rows[i].label for i in fit)
    assert labels == tuple(m["product"] for m in manifest)
    assert tuple(m["relative_path"] for m in manifest) == tuple(
        str(rows[i].image_path.relative_to(args.dataset_root)) for i in fit
    )
    assert set(labels).isdisjoint(rows[i].label for i in held)
    query, gallery = roles(
        labels, tuple(rows[i].image_path for i in fit), args.dataset_root
    )
    assert set(query).isdisjoint(gallery) and sorted(query + gallery) == list(
        range(len(fit))
    )
    assert len(set(labels[i] for i in query)) == 1992
    args.output.mkdir(exist_ok=False)
    quality, retained = {}, {}
    with np.load(args.training_dir / "initializers.npz", allow_pickle=False) as init:
        for arm in ("large", "pe"):
            path = args.source_cache / (arm + ".fit.npy")
            assert smoke.sha(path) == cache["arms"][arm]["features_sha256"]
            source = torch.from_numpy(np.load(path, allow_pickle=False))
            assert source.shape == (13283, 1024) and source.dtype == torch.float32
            assert torch.isfinite(source).all()
            assert torch.allclose(
                source.norm(dim=1), torch.ones(13283), atol=1e-5, rtol=0
            )
            head = nn.Linear(1024, 128)
            head.load_state_dict(
                {
                    k: torch.from_numpy(init[arm + ".head." + k])
                    for k in ("weight", "bias")
                }
            )
            projected = F.normalize(smoke.compact_head_features(source, head), dim=1)
            assert torch.allclose(
                projected, torch.from_numpy(init[arm + ".bank"]), atol=1e-6, rtol=0
            )
            retained[arm] = retained_variance(source, head.weight)
            for name, values in (("source1024", source), ("pca128", projected)):
                key = arm + "." + name
                tick = time.perf_counter()
                quality[key] = packed_quality(
                    values.numpy(), labels, query, gallery, device=torch.device("cpu")
                )
                smoke.save(
                    args.output / (key + ".json"),
                    {
                        "quality": quality[key],
                        "score_seconds": time.perf_counter() - tick,
                        "width": values.shape[1],
                        "quality_read": "TRAIN-fit diagnostic only",
                    },
                )
                print(
                    json.dumps(
                        {
                            "case": key,
                            "r1": quality[key]["recall_at_1"],
                            "map_at_r": quality[key]["map_at_r"],
                        }
                    ),
                    flush=True,
                )
    products = np.asarray([labels[i] for i in query])
    contrasts = {}
    for metric, field in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        values = {k: np.asarray(v[field]) for k, v in quality.items()}
        delta = {
            "pe_minus_large_source": values["pe.source1024"]
            - values["large.source1024"],
            "pe_minus_large_pca": values["pe.pca128"] - values["large.pca128"],
            "large_compression": values["large.pca128"] - values["large.source1024"],
            "pe_compression": values["pe.pca128"] - values["pe.source1024"],
        }
        delta["differential_compression"] = (
            delta["pe_compression"] - delta["large_compression"]
        )
        contrasts[metric] = {
            k: {
                "delta_pp": 100 * float(v.mean()),
                "product_bootstrap_95_pp": [
                    100 * bootstrap_lower(v, products),
                    -100 * bootstrap_lower(-v, products),
                ],
            }
            for k, v in delta.items()
        }
    assert time.perf_counter() - started < 180
    smoke.save(
        args.output / "receipt.json",
        {
            "schema": "sfora-pe-fixed-fit-compression-diagnostic-v1",
            "quality": quality,
            "contrasts": contrasts,
            "retained_centered_variance": retained,
            "query_rows": len(query),
            "gallery_rows": len(gallery),
            "query_products": len(set(products)),
            "query_sha256": hashlib.sha256(
                np.asarray(query, dtype="<i4").tobytes()
            ).hexdigest(),
            "gallery_sha256": hashlib.sha256(
                np.asarray(gallery, dtype="<i4").tobytes()
            ).hexdigest(),
            "source_files_sha256": smoke.authority(),
            "input_files_sha256": {
                str(path): smoke.sha(path)
                for path in (
                    args.source_cache / "receipt.json",
                    args.source_cache / "preflight.json",
                    args.source_cache / "large.fit.npy",
                    args.source_cache / "pe.fit.npy",
                    args.training_dir / "receipt.json",
                    args.training_dir / "preflight.json",
                    args.training_dir / "initializers.npz",
                    args.dataset_root / "Eval/list_eval_partition.txt",
                )
            },
            "prior_stop_preserved": True,
            "whole_wall_seconds": time.perf_counter() - started,
            "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "quality_read": "TRAIN-fit diagnostic only",
            "claim_eligible": False,
        },
    )
    print(
        json.dumps({"contrasts": contrasts, "retained_variance": retained}), flush=True
    )


if __name__ == "__main__":
    main()
