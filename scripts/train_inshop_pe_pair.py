#!/usr/bin/env python3
"""Frozen 100-update native augmented PE/Large TRAIN-held feasibility pair."""

import argparse
import gc
import hashlib
import json
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import probe_inshop_pe_training_smoke as smoke
from audit_inshop_pe_fit_features import PREFLIGHT_SHA
from compare_inshop_sop_warmstart_100 import packed_quality
from export_inshop_pe_fit_features import source_authority
from export_sop_siglip2_train import export_features
from pe_core_authority import WEIGHT_BYTES, WEIGHT_SHA
from preflight_inshop_siglip2_unseen_gallery import schedule, split
from score_inshop_crop_view_pair import GALLERY_SHA, QUERY_SHA, bootstrap_lower, roles
from sfora.unicom_inshop import parse_inshop_partition
from train_sop_siglip2_compact import ImageRows

CACHE_SHA = "f71852c782bc700f5a269eba120e291e1d4d1a5c2e20b5410effa4284d121add"
SEED = 179032
sha = smoke.sha


def augmented_images(dataset_root, manifest, batch, step):
    paths = tuple(dataset_root / manifest[i]["relative_path"] for i in batch)
    assert all(
        sha(p) == manifest[i]["image_sha256"] for p, i in zip(paths, batch, strict=True)
    )
    dataset = ImageRows(paths, tuple(batch), augment=step is not None)
    with torch.random.fork_rng(devices=[]):
        if step is not None:
            torch.random.default_generator.manual_seed(SEED * 100_000 + step)
        images = [dataset[i][0] for i in range(len(batch))]
    h = hashlib.sha256()
    for image in images:
        h.update(str(image.size).encode())
        h.update(image.tobytes())
    return images, h.hexdigest()


def pixels(processor, images, arm):
    return (
        torch.stack([processor(image) for image in images])
        if arm == "pe"
        else processor(images=images, return_tensors="pt")["pixel_values"]
    )


def rank_active(batch, counts):
    return all(counts[i] > 1 for i in batch)


def executing_authority(root, code):
    root = root.resolve()
    assert Path(__file__).resolve() == root / "train_inshop_pe_pair.py"
    assert sha(Path(__file__)) == code["train_inshop_pe_pair.py"]
    scripts = {Path(n).stem for n in code if "/" not in n and n.endswith(".py")}
    for name, module in list(sys.modules.items()):
        if name in scripts or name.split(".")[0] in {"sfora", "core", "einops", "ftfy"}:
            file = getattr(module, "__file__", None)
            if file:
                path = Path(file).resolve()
                assert path.is_relative_to(root), name
                relative = str(path.relative_to(root))
                assert relative in code and sha(path) == code[relative], name


def authenticate_models(args):
    checkpoint = Path(
        json.loads((args.root / "acquisition.json").read_text())["checkpoint"]
    )
    assert checkpoint.stat().st_size == WEIGHT_BYTES and sha(checkpoint) == WEIGHT_SHA
    assert args.large_snapshot.name == smoke.MODEL_REVISION
    assert all(sha(args.large_snapshot / n) == h for n, h in smoke.MODEL_HASHES.items())


def reserve_attempt(output):
    for name in (
        "attempt.json",
        "receipt.json",
        "large.pt",
        "pe.pt",
        "large.held.npy",
        "pe.held.npy",
        "large.held.npy.partial",
        "pe.held.npy.partial",
        "large.json",
        "pe.json",
        "large.cleanup.json",
        "pe.cleanup.json",
    ):
        path = output / name
        assert not path.exists() and not path.is_symlink(), name
    smoke.save(output / "attempt.json", {"seed": SEED, "updates": 100})


def cuda_budget(arm):
    assert torch.cuda.max_memory_allocated() < (
        16_000_000_000 if arm == "large" else 10_000_000_000
    )


def check_startup(args):
    assert (
        args.preflight_sha256
        and sha(args.output / "preflight.json") == args.preflight_sha256
    )
    frozen = json.loads((args.output / "preflight.json").read_text())
    from einops import _torch_specific

    from core.vision_encoder.pe import VisionTransformer

    assert (
        callable(VisionTransformer.from_config)
        and Path(_torch_specific.__file__).is_file()
    )
    executing_authority(args.root, frozen["code"])
    source_authority(args)
    authenticate_models(args)
    assert sha(args.cache / "receipt.json") == CACHE_SHA
    assert sha(args.cache / "preflight.json") == PREFLIGHT_SHA
    cache = json.loads((args.cache / "receipt.json").read_text())
    assert all(
        sha(args.cache / (a + ".fit.npy")) == r["features_sha256"]
        for a, r in cache["arms"].items()
    )
    assert sha(args.output / "initializers.npz") == frozen["initializers_sha256"]
    assert all(sha(args.root / name) == value for name, value in frozen["code"].items())
    return frozen


def preflight(args):
    from einops import _torch_specific

    assert Path(_torch_specific.__file__).is_file()
    source_authority(args)
    authenticate_models(args)
    assert sha(args.cache / "receipt.json") == CACHE_SHA
    assert sha(args.cache / "preflight.json") == PREFLIGHT_SHA
    cache = json.loads((args.cache / "receipt.json").read_text())
    manifest = json.loads((args.cache / "preflight.json").read_text())["manifest"]
    names = sorted({r["product"] for r in manifest})
    codec = {name: i for i, name in enumerate(names)}
    target = tuple(codec[r["product"]] for r in manifest)
    counts = tuple(Counter(target)[i] for i in range(len(names)))
    assert len(target) == 13283 and len(names) == 2004 and counts.count(1) == 12
    batches = schedule(tuple(r["product"] for r in manifest), 1000, seed=SEED)[:100]
    active = [rank_active(tuple(target[i] for i in batch), counts) for batch in batches]
    rows = tuple(
        r for r in parse_inshop_partition(args.dataset_root) if r.split == "train"
    )
    fit, held = split(tuple(r.label for r in rows))
    assert tuple(r["train_row"] for r in manifest) == fit
    held_manifest = [
        {
            "relative_path": str(rows[i].image_path.relative_to(args.dataset_root)),
            "image_sha256": sha(rows[i].image_path),
            "product": rows[i].label,
        }
        for i in held
    ]
    query, gallery = roles(
        tuple(rows[i].label for i in held),
        tuple(rows[i].image_path for i in held),
        args.dataset_root,
    )
    assert len(query) == 6354 and len(gallery) == 6245
    for values, expected in ((query, QUERY_SHA), (gallery, GALLERY_SHA)):
        assert (
            hashlib.sha256(np.asarray(values, dtype="<i4").tobytes()).hexdigest()
            == expected
        )
    args.output.mkdir(exist_ok=False)
    initializers, info = {}, {}
    for arm in ("large", "pe"):
        path = args.cache / (arm + ".fit.npy")
        assert sha(path) == cache["arms"][arm]["features_sha256"]
        source = torch.from_numpy(np.load(path, allow_pickle=False))
        head, classifier, pca_sha = smoke.initialize_head_and_classifier(
            source, target, allow_singletons=True
        )
        bank = F.normalize(smoke.compact_head_features(source, head).detach(), dim=1)
        assert torch.isfinite(bank).all() and (bank.norm(dim=1) > 0).all()
        batch = batches[active.index(True)]
        index = torch.tensor(batch)
        query_source = source[index].clone().requires_grad_()
        raw = smoke.compact_head_features(query_source, head)
        positives = smoke.member_bank_positive_ordinals(
            np.asarray(target, dtype=np.int64), allow_singletons=True
        )
        ce = smoke.sharded_mask_arcface_loss(
            raw,
            classifier,
            torch.tensor(target)[index],
            torch.arange(128).unsqueeze(0),
            margin=0.3,
            scale=64,
        )
        objective = ce + 8 * smoke.member_bank_rank_loss(
            raw, bank, head, positives[index], index, live_head=False
        )
        gradients = torch.autograd.grad(
            objective, (query_source, head.weight, head.bias, classifier)
        )
        assert torch.isfinite(objective)
        assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
        gradient_norms = [float(g.norm()) for g in gradients]
        objective_value = float(objective.detach())
        del objective, ce, raw, query_source, gradients, positives
        for name, value in {
            "head.weight": head.weight,
            "head.bias": head.bias,
            "classifier": classifier,
            "bank": bank,
        }.items():
            initializers[arm + "." + name] = value.detach().numpy().copy()
        vision, processor = smoke.load_arm(args, arm)
        inventory = smoke.freeze_prefix(vision, arm)
        images, rgb_sha = augmented_images(args.dataset_root, manifest, batches[0], 1)
        native_pixels = pixels(processor, images, arm)
        info[arm] = {
            "cached_full_bank_objective": objective_value,
            "cached_full_bank_gradient_norms": gradient_norms,
            "pca_sha256": pca_sha,
            "inventory": inventory,
            "first_rgb_sha256": rgb_sha,
            "first_pixels_sha256": smoke.digest({"pixels": native_pixels}),
            "head_sha256": smoke.digest(head.state_dict()),
            "classifier_sha256": smoke.digest({"classifier": classifier}),
            "bank_sha256": smoke.digest({"bank": bank}),
        }
        del source, vision, processor, head, classifier, bank, images, native_pixels
        gc.collect()
    assert info["large"]["first_rgb_sha256"] == info["pe"]["first_rgb_sha256"]
    with (args.output / "initializers.npz").open("xb") as stream:
        np.savez(stream, **initializers)
    used = sorted({i for batch in batches for i in batch})
    smoke.save(
        args.output / "preflight.json",
        {
            "code": smoke.authority(),
            "initializers": info,
            "initializers_sha256": sha(args.output / "initializers.npz"),
            "target": target,
            "counts": counts,
            "batches": batches,
            "rank_active": active,
            "fit_manifest": manifest,
            "held_manifest": held_manifest,
            "query": query,
            "gallery": gallery,
            "seed": SEED,
            "updates": 100,
            "distinct_images": len(used),
            "distinct_products": len({target[i] for i in used}),
            "presentations": 6400,
            "quality_read": False,
            "claim_eligible": False,
        },
    )
    print(
        "PASS full-fit initializers, matched augmented pixels, held role and code authority"
    )


def run_arm(args, arm, frozen):
    torch.manual_seed(SEED)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    vision, processor = smoke.load_arm(args, arm)
    executing_authority(args.root, frozen["code"])
    inventory = smoke.freeze_prefix(vision, arm)
    assert inventory == {
        k: tuple(v) for k, v in frozen["initializers"][arm]["inventory"].items()
    }
    vision = vision.cuda().train()
    with np.load(args.output / "initializers.npz", allow_pickle=False) as init:
        head = nn.Linear(1024, 128)
        head.load_state_dict(
            {k: torch.from_numpy(init[arm + ".head." + k]) for k in ("weight", "bias")}
        )
        head = head.cuda().train()
        classifier = nn.Parameter(torch.from_numpy(init[arm + ".classifier"]).cuda())
        bank = torch.from_numpy(init[arm + ".bank"]).cuda()
    target = torch.tensor(frozen["target"], device="cuda")
    positives = smoke.member_bank_positive_ordinals(
        np.asarray(frozen["target"], dtype=np.int64), allow_singletons=True
    ).cuda()
    trainable = [p for p in vision.parameters() if p.requires_grad]
    params = trainable + list(head.parameters()) + [classifier]
    optimizer = torch.optim.AdamW(
        [
            {"params": trainable, "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    dtype, scaler = smoke.training_precision("fp16", device="cuda")
    assert scaler.get_scale() == 128
    images, _ = augmented_images(
        args.dataset_root, frozen["fit_manifest"], (0, 1, 2, 3), None
    )
    x = pixels(processor, images, arm).cuda()
    with torch.no_grad():
        fp32 = smoke.encode(vision, x, arm).float()
        with torch.autocast("cuda", dtype=dtype):
            fp16 = smoke.encode(vision, x, arm).float()
        cached = torch.from_numpy(
            np.load(args.cache / (arm + ".fit.npy"), mmap_mode="r")[:4].copy()
        ).cuda()
        precision = {
            "fp16_fp32": F.cosine_similarity(fp16, fp32).tolist(),
            "cached_fresh_fp16": F.cosine_similarity(cached, fp16).tolist(),
        }
        print(json.dumps({"arm": arm, "calibration": precision}), flush=True)
        assert all(min(v) >= 0.999 for v in precision.values())
        cuda_budget(arm)
    del x, images, fp32, fp16, cached
    before = smoke.digest(smoke.frozen_state(vision, arm, inventory))
    groups_before = {
        n: smoke.digest(v)
        for n, v in smoke.groups(vision, head, classifier, arm).items()
    }
    seconds, losses, scales, norms, rgb_hashes, diagnostics = [], [], [], [], [], []
    input_seconds = []
    for step, batch in enumerate(frozen["batches"], 1):
        torch.cuda.synchronize()
        tick = time.perf_counter()
        images, rgb_sha = augmented_images(
            args.dataset_root, frozen["fit_manifest"], batch, step
        )
        x = pixels(processor, images, arm)
        if step == 1:
            assert rgb_sha == frozen["initializers"][arm]["first_rgb_sha256"]
            assert (
                smoke.digest({"pixels": x})
                == frozen["initializers"][arm]["first_pixels_sha256"]
            )
        input_seconds.append(time.perf_counter() - tick)
        x = x.cuda()
        index = torch.tensor(batch, device="cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=dtype):
            source = smoke.encode(vision, x, arm)
        raw = smoke.compact_head_features(source, head)
        ce = smoke.sharded_mask_arcface_loss(
            raw,
            classifier,
            target[index],
            torch.arange(128, device="cuda").unsqueeze(0),
            margin=0.3,
            scale=64,
        )
        rank = (
            smoke.member_bank_rank_loss(
                raw, bank, head, positives[index], index, live_head=False
            )
            if frozen["rank_active"][step - 1]
            else ce.new_zeros(())
        )
        loss = ce + 8 * rank
        assert torch.isfinite(loss)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        assert scaler.get_scale() >= scale, "optimizer update skipped"
        rows, positions = smoke.member_bank_refresh_rows(tuple(batch))
        bank[torch.tensor(rows, device="cuda")] = smoke.member_bank_refresh_values(
            source, raw, torch.tensor(positions, device="cuda"), live_head=False
        )
        assert all(
            torch.isfinite(p).all()
            for _, p in smoke.named_training_parameters(vision, arm)
        )
        assert all(
            torch.isfinite(p).all() for p in list(head.parameters()) + [classifier]
        )
        assert all(
            torch.isfinite(v).all()
            for state in optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        )
        assert torch.isfinite(bank).all()
        cuda_budget(arm)
        if step in (1, 100):
            diagnostics.append(
                {
                    "step": step,
                    "per_parameter_grad_norm": smoke.gradients(
                        vision, head, classifier, arm
                    ),
                }
            )
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - tick)
        losses.append(float(loss.detach()))
        scales.append(scaler.get_scale())
        norms.append(float(norm))
        rgb_hashes.append(rgb_sha)
        if step % 10 == 0:
            print(
                json.dumps({"arm": arm, "step": step, "loss": losses[-1]}), flush=True
            )
    after = smoke.digest(smoke.frozen_state(vision, arm, inventory))
    groups_after = {
        n: smoke.digest(v)
        for n, v in smoke.groups(vision, head, classifier, arm).items()
    }
    assert after == before and all(
        groups_after[n] != v for n, v in groups_before.items()
    )
    checkpoint = args.output / (arm + ".pt")
    torch.save(
        {
            "vision": vision.state_dict(),
            "head": head.state_dict(),
            "classifier": classifier.detach(),
            "bank": bank,
            "rope": vision.rope.rope.state_dict() if arm == "pe" else None,
            "rope_freq": vision.rope.freq if arm == "pe" else None,
        },
        checkpoint,
    )
    del optimizer, source, raw, ce, rank, loss, x, index, images
    vision.zero_grad(set_to_none=True)
    head.zero_grad(set_to_none=True)
    classifier.grad = None
    vision.eval()
    head.eval()

    @torch.inference_mode()
    def encode(batch):
        images, _ = augmented_images(
            args.dataset_root, batch, tuple(range(len(batch))), None
        )
        with torch.autocast("cuda", dtype=dtype):
            source = smoke.encode(vision, pixels(processor, images, arm).cuda(), arm)
        values = F.normalize(smoke.compact_head_features(source, head).float(), dim=1)
        assert torch.isfinite(values).all() and (values.norm(dim=1) > 0).all()
        cuda_budget(arm)
        return values.cpu().numpy()

    held_path = args.output / (arm + ".held.npy")
    export_features(
        frozen["held_manifest"], encode, held_path, width=128, batch_size=32
    )
    quality = packed_quality(
        np.load(held_path, allow_pickle=False),
        tuple(r["product"] for r in frozen["held_manifest"]),
        frozen["query"],
        frozen["gallery"],
    )
    cuda_budget(arm)
    report = {
        "inventory": inventory,
        "frozen_sha256": before,
        "final_frozen_sha256": after,
        "initial_groups": groups_before,
        "final_groups": groups_after,
        "diagnostics": diagnostics,
        "losses": losses,
        "step_seconds": seconds,
        "input_seconds": input_seconds,
        "guarded_update_seconds": [
            s - i for s, i in zip(seconds, input_seconds, strict=True)
        ],
        "preclip_norms": norms,
        "grad_scaler_scales": scales,
        "rgb_sha256": rgb_hashes,
        "rank_active_updates": sum(frozen["rank_active"]),
        "calibration": precision,
        "median_step_3_100_seconds": float(np.median(seconds[2:])),
        "guarded_images_per_second": 6400 / sum(seconds),
        "training_seconds": sum(seconds),
        "arm_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "checkpoint_sha256": sha(checkpoint),
        "held_sha256": sha(held_path),
        "quality": quality,
        "quality_read": "TRAIN-held only",
        "claim_eligible": False,
    }
    smoke.save(args.output / (arm + ".json"), report)
    assert report["peak_cuda_allocated_bytes"] < (
        16_000_000_000 if arm == "large" else 10_000_000_000
    )
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "root",
        "cache",
        "output",
        "dataset-root",
        "large-snapshot",
        "mechanics-dir",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--preflight-sha256")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--preflight-only", action="store_true")
    mode.add_argument("--check-startup-only", action="store_true")
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.preflight_only:
        preflight(args)
        return
    frozen = check_startup(args)
    if args.check_startup_only:
        print("PASS trusted external preflight/source/cache/initializer/code authority")
        return
    reserve_attempt(args.output)
    assert torch.cuda.is_available() and not (args.output / "receipt.json").exists()
    started = time.perf_counter()
    reports = {}
    for arm in ("large", "pe"):
        reports[arm] = run_arm(args, arm, frozen)
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        cleanup = torch.cuda.memory_allocated()
        smoke.save(
            args.output / (arm + ".cleanup.json"), {"allocated_cuda_bytes": cleanup}
        )
        assert cleanup < 8 * 1024**2
    assert reports["large"]["rgb_sha256"] == reports["pe"]["rgb_sha256"]
    products = np.asarray(
        [frozen["held_manifest"][i]["product"] for i in frozen["query"]]
    )
    paired = {}
    for metric, key in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        delta = (
            np.asarray(reports["pe"]["quality"][key]) - reports["large"]["quality"][key]
        )
        paired[metric] = {
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
    ratio = (
        reports["pe"]["median_step_3_100_seconds"]
        / reports["large"]["median_step_3_100_seconds"]
    )
    advance = (
        paired["r1"]["delta_pp"] >= -0.5
        and paired["map_at_r"]["delta_pp"] >= -1
        and ratio <= 0.8
    )
    current = smoke.authority()
    assert set(current) <= set(frozen["code"])
    assert all(current.get(n, h) == h for n, h in frozen["code"].items())
    assert time.perf_counter() - started < 600
    smoke.save(
        args.output / "receipt.json",
        {
            "arms": reports,
            "executed_code": current,
            "paired": paired,
            "step_time_ratio": ratio,
            "advance": bool(advance),
            "decision": "FEASIBILITY_PASS" if advance else "STOP",
            "preflight_sha256": args.preflight_sha256,
            "whole_wall_seconds": time.perf_counter() - started,
            "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "quality_read": "TRAIN-held only",
            "claim_eligible": False,
        },
    )
    print(
        json.dumps(
            {
                "decision": "FEASIBILITY_PASS" if advance else "STOP",
                "paired": paired,
                "step_time_ratio": ratio,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
