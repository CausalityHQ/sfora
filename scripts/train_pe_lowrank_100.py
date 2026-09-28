#!/usr/bin/env python3
"""One fresh100-update rank32 PE TRAIN pilot under the frozen quality gate."""

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pe_lowrank_adaptation as lowrank
import train_inshop_pe_pair as pair
from pe_core_training import frozen_state, named_training_parameters


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-startup-only", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    code = json.loads(args.authority.read_text())
    assert set(code) == {
        "train_pe_lowrank_100.py",
        "pe_lowrank_adaptation.py",
        "qualify_pe_lowrank_native_cpu.py",
        "run_pe_lowrank_100.sh",
        "native-cpu-v1.json",
        "mechanics-receipt.json",
    }
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert Path(lowrank.__file__).resolve() == root / "pe_lowrank_adaptation.py"
    cpu_result = root / "native-cpu-v1.json"
    cpu = json.loads(cpu_result.read_text())
    assert cpu["pass"]
    mechanics = json.loads((root / "mechanics-receipt.json").read_text())
    assert mechanics["pass"] and mechanics["updates"] == 17
    assert mechanics["updated_gpu_strict_reload_exact"]
    assert all(
        cpu["authority"][n] == code[n]
        for n in ("pe_lowrank_adaptation.py", "qualify_pe_lowrank_native_cpu.py")
    )
    source_args = SimpleNamespace(
        root=root,
        output=Path("/home/riomus/runs/sfora-pe-augmented-100-v2"),
        preflight_sha256="41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293",
        cache=Path("/home/riomus/runs/sfora-pe-fullfit-cache-v3"),
        mechanics_dir=Path("/home/riomus/runs/sfora-pe-fp16-smoke-v1"),
        dataset_root=Path("/home/riomus/datasets/inshop_official_standard"),
        large_snapshot=Path(
            "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c"
        ),
    )
    frozen = pair.check_startup(source_args)
    prior = json.loads((source_args.output / "pe.json").read_text())
    assert (
        pair.sha(source_args.output / "receipt.json")
        == "d02a1c176f023e81bd9a0bb526ff3294e677e88dcc0724b9306e1ca884177dbb"
    )
    assert (
        prior
        == json.loads((source_args.output / "receipt.json").read_text())["arms"]["pe"]
    )
    if args.check_startup_only:
        assert not torch.cuda.is_available()
        print(
            "PASS native CPU receipt binding, required authority set and archived controls; no CUDA",
            flush=True,
        )
        return
    args.output.mkdir(exist_ok=False)
    pair.smoke.save(args.output / "attempt.json", {"code": code, "updates": 100})
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    vision, processor = pair.smoke.load_arm(source_args, "pe")
    pair.executing_authority(root, frozen["code"])
    inventory = pair.smoke.freeze_prefix(vision, "pe")
    assert inventory == {
        k: tuple(v) for k, v in frozen["initializers"]["pe"]["inventory"].items()
    }
    sites = lowrank.install(vision, inventory)
    vision.cuda().train()
    groups = lowrank.parameter_groups(vision, sites)
    factors = [(n, getattr(m.parametrizations, k)[0]) for n, (m, k) in sites.items()]

    def original_digest(sites):
        return pair.smoke.digest(
            {n: getattr(m.parametrizations, k).original for n, (m, k) in sites.items()}
        )

    original_sha = original_digest(sites)
    with np.load(source_args.output / "initializers.npz", allow_pickle=False) as init:
        head = nn.Linear(1024, 128)
        head.load_state_dict(
            {k: torch.from_numpy(init["pe.head." + k]) for k in ("weight", "bias")}
        )
        head = head.cuda().train()
        classifier = nn.Parameter(torch.from_numpy(init["pe.classifier"]).cuda())
        bank = torch.from_numpy(init["pe.bank"]).cuda()
    target = torch.tensor(frozen["target"], device="cuda")
    positives = pair.smoke.member_bank_positive_ordinals(
        np.asarray(frozen["target"], dtype=np.int64), allow_singletons=True
    ).cuda()
    params = (
        [p for g in groups for p in g["params"]]
        + list(head.parameters())
        + [classifier]
    )
    assert len({id(p) for p in params}) == len(params)
    optimizer = torch.optim.AdamW(
        groups
        + [
            {"params": list(head.parameters()), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    dtype, scaler = pair.smoke.training_precision("fp16", device="cuda")
    assert scaler.get_scale() == 128
    images, _ = pair.augmented_images(
        source_args.dataset_root, frozen["fit_manifest"], (0, 1, 2, 3), None
    )
    x = pair.pixels(processor, images, "pe").cuda()
    with torch.no_grad():
        fp32 = vision(x).float()
        with torch.autocast("cuda", dtype=dtype):
            fp16 = vision(x).float()
        cached = torch.from_numpy(
            np.load(source_args.cache / "pe.fit.npy", mmap_mode="r")[:4].copy()
        ).cuda()
        calibration = {
            "fp16_fp32": F.cosine_similarity(fp16, fp32).tolist(),
            "cached_fresh_fp16": F.cosine_similarity(cached, fp16).tolist(),
        }
        assert all(min(v) >= 0.999 for v in calibration.values())
    del images, x, fp32, fp16, cached
    pair.cuda_budget("pe")
    frozen_sha = pair.smoke.digest(frozen_state(vision, "pe", inventory))
    initial_groups = {
        n: pair.smoke.digest(v)
        for n, v in pair.smoke.groups(vision, head, classifier, "pe").items()
    }
    seconds, losses, rgb_hashes, diagnostics, scales = [], [], [], [], []
    for step, batch in enumerate(frozen["batches"], 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        images, rgb_sha = pair.augmented_images(
            source_args.dataset_root, frozen["fit_manifest"], batch, step
        )
        assert rgb_sha == prior["rgb_sha256"][step - 1]
        x = pair.pixels(processor, images, "pe")
        if step == 1:
            assert (
                pair.smoke.digest({"pixels": x})
                == frozen["initializers"]["pe"]["first_pixels_sha256"]
            )
        x = x.cuda()
        index = torch.tensor(batch, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            source = vision(x)
        raw = pair.smoke.compact_head_features(source, head)
        ce = pair.smoke.sharded_mask_arcface_loss(
            raw,
            classifier,
            target[index],
            torch.arange(128, device="cuda").unsqueeze(0),
            margin=0.3,
            scale=64,
        )
        rank = (
            pair.smoke.member_bank_rank_loss(
                raw, bank, head, positives[index], index, live_head=False
            )
            if frozen["rank_active"][step - 1]
            else ce.new_zeros(())
        )
        loss = ce + 8 * rank
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        for n, p in (
            list(named_training_parameters(vision, "pe"))
            + [("head." + n, p) for n, p in head.named_parameters()]
            + [("classifier", classifier)]
        ):
            assert (p.grad is None) == (not p.requires_grad), n
            if p.grad is not None:
                assert torch.isfinite(p.grad).all(), n
                if step in (1, 100) and ".parametrizations." not in n:
                    assert p.grad.norm() > 0, n
        if step in (1, 2, 100):
            norms = {}
            for n, factor in factors:
                a, b = factor.A.grad, factor.B.grad
                assert (torch.count_nonzero(a) == 0) if step == 1 else (a.norm() > 0)
                assert b.norm() > 0
                norms[n] = [float(a.norm()), float(b.norm())]
            diagnostics.append({"step": step, "factor_data_gradient_norms": norms})
        torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= scale, "skipped update"
        rows, positions = pair.smoke.member_bank_refresh_rows(tuple(batch))
        bank[torch.tensor(rows, device="cuda")] = pair.smoke.member_bank_refresh_values(
            source, raw, torch.tensor(positions, device="cuda"), live_head=False
        )
        assert all(torch.isfinite(p).all() for p in params)
        assert all(
            torch.isfinite(v).all()
            for state in optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        )
        assert torch.isfinite(bank).all()
        pair.cuda_budget("pe")
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - tick)
        losses.append(float(loss.detach()))
        rgb_hashes.append(rgb_sha)
        scales.append(scaler.get_scale())
        print(
            json.dumps({"step": step, "seconds": seconds[-1], "loss": losses[-1]}),
            flush=True,
        )
    assert original_digest(sites) == original_sha
    assert frozen_sha == pair.smoke.digest(frozen_state(vision, "pe", inventory))
    assert all(
        pair.smoke.digest(pair.smoke.groups(vision, head, classifier, "pe")[n]) != h
        for n, h in initial_groups.items()
    )
    median = float(np.median(seconds[2:]))
    # Persist cost before the frozen stop rule, including negative outcomes.
    pair.smoke.save(
        args.output / "training.json",
        {
            "median_step_3_100_seconds": median,
            "step_seconds": seconds,
            "losses": losses,
            "rgb_sha256": rgb_hashes,
            "scales": scales,
            "diagnostics": diagnostics,
            "calibration": calibration,
            "original_sha256": original_sha,
            "frozen_sha256": frozen_sha,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "quality_read": False,
        },
    )
    if median > 0.71769696:
        pair.smoke.save(
            args.output / "receipt.json",
            {
                "advance": False,
                "reason": "fixed training cost gate failed",
                "quality_read": False,
                "updates": 100,
                "median_step_3_100_seconds": median,
            },
        )
        print("KILL cost; stopped before held read", flush=True)
        return
    del optimizer, groups, params, source, raw, ce, rank, loss
    vision.zero_grad(set_to_none=True)
    vision.eval()
    head.eval()

    def export(model, compact, path):
        @torch.inference_mode()
        def encode(batch):
            images, _ = pair.augmented_images(
                source_args.dataset_root, batch, tuple(range(len(batch))), None
            )
            with torch.autocast("cuda", dtype=dtype):
                raw_source = model(pair.pixels(processor, images, "pe").cuda())
            values = F.normalize(
                pair.smoke.compact_head_features(raw_source, compact).float(), dim=1
            )
            assert torch.isfinite(values).all() and (values.norm(dim=1) > 0).all()
            pair.cuda_budget("pe")
            return values.cpu().numpy()

        pair.export_features(
            frozen["held_manifest"], encode, path, width=128, batch_size=32
        )
        return np.load(path, allow_pickle=False)

    live = export(vision, head, args.output / "inprocess.held.npy")
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        updated = vision(x).float()
    lowrank.merge(sites)
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        merged = vision(x).float()
    assert torch.equal(updated, merged), "updated native GPU merge differs"
    pair.cuda_budget("pe")
    checkpoint = args.output / "pe.pt"
    torch.save(
        {
            "vision": vision.state_dict(),
            "head": head.state_dict(),
            "classifier": classifier.detach(),
            "bank": bank,
            "rope": vision.rope.rope.state_dict(),
            "rope_freq": vision.rope.freq,
        },
        checkpoint,
    )
    checkpoint_sha = pair.sha(checkpoint)
    from core.vision_encoder.pe import VisionTransformer

    fresh = (
        VisionTransformer.from_config("PE-Core-B16-224", pretrained=False)
        .float()
        .eval()
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    fresh.load_state_dict(saved["vision"], strict=True)
    fresh.rope.rope.load_state_dict(saved["rope"], strict=True)
    fresh.rope.update_grid(torch.device("cpu"), 14, 14)
    assert torch.equal(fresh.rope.freq, saved["rope_freq"])
    fresh.cuda()
    loaded_head = nn.Linear(1024, 128)
    loaded_head.load_state_dict(saved["head"], strict=True)
    loaded_head.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        reloaded = fresh(x).float()
    assert torch.equal(updated, reloaded), "updated strict-loaded GPU output differs"
    pair.cuda_budget("pe")
    pair.executing_authority(root, frozen["code"])
    loaded = export(fresh, loaded_head, args.output / "pe.held.npy")
    minimum_cosine = float(
        np.min(
            np.sum(live.astype(np.float64) * loaded, axis=1)
            / (
                np.linalg.norm(live.astype(np.float64), axis=1)
                * np.linalg.norm(loaded.astype(np.float64), axis=1)
            )
        )
    )
    assert minimum_cosine >= 0.999999
    labels = tuple(r["product"] for r in frozen["held_manifest"])
    quality = pair.packed_quality(loaded, labels, frozen["query"], frozen["gallery"])
    live_quality = pair.packed_quality(live, labels, frozen["query"], frozen["gallery"])
    assert all(
        np.max(np.abs(np.asarray(quality[k]) - np.asarray(live_quality[k]))) <= 1e-6
        for k in ("per_query_r1", "per_query_ap")
    )
    products = np.asarray(labels)[frozen["query"]]
    intervals = {}
    for name in ("per_query_r1", "per_query_ap"):
        delta = np.asarray(quality[name]) - np.asarray(prior["quality"][name])
        intervals[name] = {
            "mean_delta": float(np.mean(delta)),
            "product_lower95": pair.bootstrap_lower(delta, products),
            "product_upper95": -pair.bootstrap_lower(-delta, products),
        }
    advance = bool(
        quality["recall_at_1"] >= 0.951720176
        and quality["map_at_r"] >= 0.776237120
        and all(v["product_lower95"] > 0 for v in intervals.values())
    )
    pair.cuda_budget("pe")
    pair.executing_authority(root, frozen["code"])
    assert all(pair.sha(root / n) == h for n, h in code.items())
    del fresh, saved, loaded_head

    pair.smoke.save(
        args.output / "receipt.json",
        {
            "advance": advance,
            "updates": 100,
            "quality_read": "In-Shop TRAIN-held only",
            "checkpoint_sha256": checkpoint_sha,
            "held_sha256": pair.sha(args.output / "pe.held.npy"),
            "quality": quality,
            "paired_dense_pe_intervals": intervals,
            "updated_gpu_merge_exact": True,
            "updated_gpu_strict_reload_exact": True,
            "updated_loaded_minimum_cosine": minimum_cosine,
            "packed_per_query_parity": True,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "median_step_3_100_seconds": median,
            "seconds": time.perf_counter() - started,
            "official_read": False,
            "claim_eligible": False,
        },
    )
    del vision, head, classifier, bank, sites, factors, x, p, factor
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "GO TRAIN feasibility" if advance else "KILL fixed TRAIN quality gate",
        flush=True,
    )


if __name__ == "__main__":
    main()
