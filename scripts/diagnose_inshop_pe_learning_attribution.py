#!/usr/bin/env python3
"""Fixed initial/final encoder/head counterfactuals; no training or selection."""

import argparse
import gc
import json
import resource
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import probe_inshop_pe_training_smoke as smoke
import train_inshop_pe_pair as pair
from compare_inshop_sop_warmstart_100 import packed_quality
from export_sop_siglip2_train import export_features
from score_inshop_crop_view_pair import bootstrap_lower

PRIOR_SHA = "d02a1c176f023e81bd9a0bb526ff3294e677e88dcc0724b9306e1ca884177dbb"
PRIOR_PREFLIGHT = "41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293"
AUDIT_SHA = "9bb2f7baa3dd8fdee010528a7a3828a8ea9abf1aa8056faff40846d729257deb"


def decomposition(states):
    parts = {
        "total": states["11"] - states["00"],
        "encoder": states["10"] - states["00"],
        "head": states["01"] - states["00"],
        "interaction": states["11"] - states["10"] - states["01"] + states["00"],
    }
    assert np.allclose(
        parts["total"],
        parts["encoder"] + parts["head"] + parts["interaction"],
        atol=1e-12,
        rtol=0,
    )
    return parts


def golden_parity(old, new, old_quality, new_quality):
    assert old.shape == new.shape and np.isfinite(new).all()
    cosine = np.sum(old.astype(np.float64) * new, axis=1) / (
        np.linalg.norm(old.astype(np.float64), axis=1)
        * np.linalg.norm(new.astype(np.float64), axis=1)
    )
    assert float(cosine.min()) >= 0.999999
    for key in ("per_query_r1", "per_query_ap"):
        assert np.allclose(old_quality[key], new_quality[key], atol=1e-6, rtol=0), key
    return float(cosine.min())


def inputs(args):
    assert smoke.sha(args.training_dir / "receipt.json") == PRIOR_SHA
    assert smoke.sha(args.training_dir / "checkpoint-score-audit.json") == AUDIT_SHA
    prior_args = SimpleNamespace(
        **{
            **vars(args),
            "output": args.training_dir,
            "preflight_sha256": PRIOR_PREFLIGHT,
        }
    )
    frozen = pair.check_startup(prior_args)
    prior = json.loads((args.training_dir / "receipt.json").read_text())
    assert prior["decision"] == "STOP" and not prior["advance"]
    paths = [
        args.training_dir / name
        for name in (
            "receipt.json",
            "preflight.json",
            "initializers.npz",
            "checkpoint-score-audit.json",
        )
    ]
    for arm in ("large", "pe"):
        for name, key in (
            (arm + ".pt", "checkpoint_sha256"),
            (arm + ".held.npy", "held_sha256"),
        ):
            path = args.training_dir / name
            assert smoke.sha(path) == prior["arms"][arm][key]
            paths.append(path)
    audit = json.loads((args.training_dir / "checkpoint-score-audit.json").read_text())
    assert audit["decision"] == "STOP" and audit["pass"]
    return frozen, prior, {str(path): smoke.sha(path) for path in paths}


def heads(args, arm, saved):
    result = []
    with np.load(args.training_dir / "initializers.npz", allow_pickle=False) as init:
        for values in (
            {
                k: torch.from_numpy(init[arm + ".head." + k].copy())
                for k in ("weight", "bias")
            },
            saved["head"],
        ):
            head = nn.Linear(1024, 128)
            head.load_state_dict(values, strict=True)
            assert all(torch.isfinite(p).all() for p in head.parameters())
            result.append(head.eval().requires_grad_(False))
    return result


def preflight(args):
    assert not torch.cuda.is_available()
    frozen, prior, hashes = inputs(args)
    for arm in ("large", "pe"):
        saved = torch.load(
            args.training_dir / (arm + ".pt"),
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )
        hs = heads(args, arm, saved)
        vision, processor = smoke.load_arm(args, arm)
        vision.load_state_dict(saved["vision"], strict=True)
        assert all(torch.isfinite(p).all() for p in vision.parameters())
        if arm == "pe":
            vision.rope.rope.load_state_dict(saved["rope"], strict=True)
            vision.rope.update_grid(torch.device("cpu"), 14, 14)
            assert torch.equal(vision.rope.freq, saved["rope_freq"])
        del vision, processor, hs, saved
        gc.collect()
    args.output.mkdir(exist_ok=False)
    code = smoke.authority()
    code[Path(__file__).name] = smoke.sha(Path(__file__))
    smoke.save(
        args.output / "preflight.json",
        {
            "code": code,
            "inputs_sha256": hashes,
            "prior_stop_preserved": True,
            "query_rows": len(frozen["query"]),
            "gallery_rows": len(frozen["gallery"]),
        },
    )
    print("PASS original/final source, saved head, input/code authority; no CUDA")


def state_digest(vision, hs, arm):
    values = dict(vision.state_dict())
    for i, head in enumerate(hs):
        values.update({f"head{i}.{k}": v for k, v in head.state_dict().items()})
    if arm == "pe":
        values.update(
            {"foreign." + k: v for k, v in vision.rope.rope.state_dict().items()}
        )
        values["foreign_grid"] = vision.rope.freq
    return smoke.digest(values)


@torch.inference_mode()
def export_state(args, frozen, arm, state):
    start = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    saved = torch.load(
        args.training_dir / (arm + ".pt"),
        weights_only=True,
        map_location="cpu",
        mmap=True,
    )
    hs = heads(args, arm, saved)
    vision, processor = smoke.load_arm(args, arm)
    if state == "1":
        vision.load_state_dict(saved["vision"], strict=True)
        if arm == "pe":
            vision.rope.rope.load_state_dict(saved["rope"], strict=True)
    if arm == "pe":
        vision.rope.update_grid(torch.device("cpu"), 14, 14)
        if state == "1":
            assert torch.equal(vision.rope.freq, saved["rope_freq"])
    vision.eval().requires_grad_(False)
    before = state_digest(vision, hs, arm)
    del saved
    vision.cuda()
    hs = [h.cuda() for h in hs]

    def pixels(batch):
        images, _ = pair.augmented_images(
            args.dataset_root, batch, tuple(range(len(batch))), None
        )
        return pair.pixels(processor, images, arm).cuda()

    x = pixels(frozen["held_manifest"][:4])
    f32 = smoke.encode(vision, x, arm).float()
    with torch.autocast("cuda", dtype=torch.float16):
        f16 = smoke.encode(vision, x, arm).float()
    cosine = F.cosine_similarity(f32, f16).tolist()
    print(
        json.dumps({"arm": arm, "vision": state, "fp16_fp32_cosine": cosine}),
        flush=True,
    )
    assert min(cosine) >= 0.999 and torch.cuda.max_memory_allocated() < 10_000_000_000
    del x, f32, f16

    def encode(batch):
        with torch.autocast("cuda", dtype=torch.float16):
            raw = smoke.encode(vision, pixels(batch), arm)
        parts = [F.normalize(raw.float(), dim=1)] + [
            F.normalize(smoke.compact_head_features(raw, h).float(), dim=1) for h in hs
        ]
        assert all(
            torch.isfinite(v).all()
            and torch.allclose(
                v.norm(dim=1), torch.ones(len(v), device="cuda"), atol=1e-5, rtol=0
            )
            for v in parts
        )
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        return torch.cat(parts, dim=1).cpu().numpy()

    path = args.output / (arm + ".vision" + state + ".npy")
    export_features(frozen["held_manifest"], encode, path, width=1280, batch_size=32)
    assert before == state_digest(vision, hs, arm)
    report = {
        "features_sha256": smoke.sha(path),
        "state_sha256": before,
        "fp16_fp32_cosine": cosine,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "state_wall_seconds": time.perf_counter() - start,
    }
    smoke.save(args.output / (arm + ".vision" + state + ".json"), report)
    return report


def score(args, frozen, prior):
    quality, parity, contrasts = {}, {}, {}
    labels = tuple(r["product"] for r in frozen["held_manifest"])
    products = np.asarray([labels[i] for i in frozen["query"]])
    for arm in ("large", "pe"):
        for state in ("0", "1"):
            values = np.load(
                args.output / (arm + ".vision" + state + ".npy"), allow_pickle=False
            )
            assert values.shape == (12599, 1280) and values.dtype == np.float32
            for h, lo in (("0", 1024), ("1", 1152)):
                key = arm + "." + state + h
                features = values[:, lo : lo + 128].copy()
                quality[key] = packed_quality(
                    features,
                    labels,
                    frozen["query"],
                    frozen["gallery"],
                    device=torch.device("cpu"),
                )
                smoke.save(args.output / (key + ".quality.json"), quality[key])
                if state + h == "11":
                    parity[arm] = golden_parity(
                        np.load(
                            args.training_dir / (arm + ".held.npy"), allow_pickle=False
                        ),
                        features,
                        prior["arms"][arm]["quality"],
                        quality[key],
                    )
    for metric, field in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        arrays = {
            arm: decomposition(
                {
                    s: np.asarray(quality[arm + "." + s][field])
                    for s in ("00", "01", "10", "11")
                }
            )
            for arm in ("large", "pe")
        }
        deltas = {
            arm + "." + part: v
            for arm, parts in arrays.items()
            for part, v in parts.items()
        }
        deltas.update(
            {
                "pe_minus_large." + part: arrays["pe"][part] - arrays["large"][part]
                for part in arrays["pe"]
            }
        )
        deltas.update(
            {
                "pe_minus_large.state" + s: np.asarray(quality["pe." + s][field])
                - quality["large." + s][field]
                for s in ("00", "01", "10", "11")
            }
        )
        contrasts[metric] = {
            key: {
                "delta_pp": 100 * float(delta.mean()),
                "product_bootstrap_95_pp": [
                    100 * bootstrap_lower(delta, products),
                    -100 * bootstrap_lower(-delta, products),
                ],
                "query_bootstrap_95_pp": [
                    100 * bootstrap_lower(delta, np.arange(len(delta))),
                    -100 * bootstrap_lower(-delta, np.arange(len(delta))),
                ],
            }
            for key, delta in deltas.items()
        }
    return quality, parity, contrasts


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in (
        "root",
        "cache",
        "training-dir",
        "output",
        "dataset-root",
        "large-snapshot",
        "mechanics-dir",
    ):
        p.add_argument("--" + n, type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--check-startup-only", action="store_true")
    p.add_argument("--preflight-sha256")
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.preflight_only:
        preflight(args)
        return
    assert (
        args.preflight_sha256
        and smoke.sha(args.output / "preflight.json") == args.preflight_sha256
    )
    external = json.loads((args.output / "preflight.json").read_text())
    assert Path(__file__).resolve() == args.root.resolve() / Path(__file__).name
    assert smoke.sha(Path(__file__)) == external["code"][Path(__file__).name]
    frozen, prior, hashes = inputs(args)
    assert hashes == external["inputs_sha256"]
    pair.executing_authority(args.root, external["code"])
    assert all(smoke.sha(args.root / n) == h for n, h in external["code"].items())
    if args.check_startup_only:
        print("PASS external startup and executed code authority")
        return
    assert set(p.name for p in args.output.iterdir()) == {"preflight.json"}
    smoke.save(
        args.output / "attempt.json",
        {"training": False, "states": ["00", "01", "10", "11"]},
    )
    start = time.perf_counter()
    reports = {}
    for arm in ("large", "pe"):
        for state in ("0", "1"):
            reports[arm + "." + state] = export_state(args, frozen, arm, state)
            torch._C._cuda_clearCublasWorkspaces()
            gc.collect()
            torch.cuda.empty_cache()
            cleanup = torch.cuda.memory_allocated()
            assert cleanup < 8 * 1024**2
            reports[arm + "." + state]["cleanup_bytes"] = cleanup
    quality, parity, contrasts = score(args, frozen, prior)
    assert inputs(args)[2] == hashes
    assert all(smoke.sha(args.root / n) == h for n, h in external["code"].items())
    assert time.perf_counter() - start < 600
    smoke.save(
        args.output / "receipt.json",
        {
            "reports": reports,
            "quality": quality,
            "golden_cosine_min": parity,
            "contrasts": contrasts,
            "prior_stop_preserved": True,
            "quality_read": "TRAIN-held saved-state attribution only",
            "claim_eligible": False,
            "preflight_sha256": args.preflight_sha256,
            "whole_wall_seconds": time.perf_counter() - start,
            "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        },
    )
    print(
        json.dumps(
            {
                "parity": parity,
                "states": {
                    k: {m: v[m] for m in ("recall_at_1", "map_at_r")}
                    for k, v in quality.items()
                },
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
