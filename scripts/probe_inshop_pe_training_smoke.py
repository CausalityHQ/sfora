#!/usr/bin/env python3
"""Matched native PE/Large image-gradient and training-cost mechanics only."""

import argparse
import gc
import hashlib
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from transformers import AutoImageProcessor, AutoModel

from core.vision_encoder.transforms import get_image_transform
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from pe_core_authority import load_visual, sha
from pe_core_training import freeze_prefix, frozen_state, named_training_parameters
from preflight_inshop_siglip2_unseen_gallery import schedule
from sfora.sop_compact_training import compact_head_features
from sfora.unicom_training import sharded_mask_arcface_loss
from train_sop_siglip2_compact import (
    initialize_head_and_classifier,
    member_bank_positive_ordinals,
    member_bank_rank_loss,
    member_bank_refresh_rows,
    member_bank_refresh_values,
)


def digest(values):
    h = hashlib.sha256()
    for name, value in sorted(values.items()):
        h.update(name.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def authority():
    root = Path(__file__).resolve().parent
    return {
        str(Path(m.__file__).resolve().relative_to(root)): sha(m.__file__)
        for m in list(sys.modules.values())
        if getattr(m, "__file__", None)
        and Path(m.__file__).is_file()
        and Path(m.__file__).resolve().is_relative_to(root)
    }


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def check_startup(args, frozen):
    root = args.root.resolve()
    for name, value in frozen["code"].items():
        path = (root / name).resolve()
        assert path.is_relative_to(root) and path.is_file(), name
        assert sha(path) == value, name
    current = authority()
    assert all(current.get(name, value) == value for name, value in frozen["code"].items())
    assert sha(args.output / "initializers.npz") == frozen["initializers_sha256"]
    assert sha(args.root / "pilot.features.npz") == frozen["source_features_sha256"]
    assert sha(args.root / "cpu-preflight-v2.json") == frozen["source_preflight_sha256"]


def metadata(args):
    p = json.loads((args.root / "cpu-preflight-v2.json").read_text())
    report = json.loads((args.root / "pilot.json").read_text())
    assert report["advance"] and sha(args.root / "pilot.features.npz") == report["features_sha256"]
    assert sha(args.root / "cpu-preflight-v2.json") == report["preflight_sha256"]
    labels = tuple(row["product"] for row in p["image_manifest"])
    codec = {name: i for i, name in enumerate(sorted(set(labels)))}
    target = torch.tensor([codec[name] for name in labels])
    assert len(codec) == 512 and torch.bincount(target).tolist() == [2] * 512
    batches = schedule(labels, 16, seed=179032)
    updated = sorted({i for batch in batches for i in batch})
    assert len(updated) == 512 and len({labels[i] for i in updated}) == 256
    paths = [args.dataset_root / row["relative_path"] for row in p["image_manifest"]]
    assert all(
        sha(path) == row["image_sha256"]
        for path, row in zip(paths, p["image_manifest"], strict=True)
    )
    cache = np.load(args.root / "pilot.features.npz", allow_pickle=False)
    initializers, info = {}, {}
    for arm in ("large", "pe"):
        source = torch.from_numpy(cache[arm].copy())
        head, classifier, pca_sha = initialize_head_and_classifier(source, tuple(target.tolist()))
        bank = F.normalize(compact_head_features(source, head).detach(), dim=1)
        initializers.update(
            {
                arm + ".head.weight": head.weight.detach().numpy(),
                arm + ".head.bias": head.bias.detach().numpy(),
                arm + ".classifier": classifier.detach().numpy(),
                arm + ".bank": bank.numpy(),
            }
        )
        info[arm] = {
            "pca_sha256": pca_sha,
            "head_sha256": digest(head.state_dict()),
            "classifier_sha256": digest({"classifier": classifier}),
            "bank_sha256": digest({"bank": bank}),
        }
        index = torch.tensor(batches[0])
        query = source[index].clone().requires_grad_()
        raw = compact_head_features(query, head)
        positives = member_bank_positive_ordinals(target.numpy())
        loss = sharded_mask_arcface_loss(
            raw, classifier, target[index], torch.arange(128).unsqueeze(0), margin=0.3, scale=64
        ) + 8 * member_bank_rank_loss(raw, bank, head, positives[index], index, live_head=False)
        grads = torch.autograd.grad(loss, (query, head.weight, head.bias, classifier))
        assert torch.isfinite(loss) and all(torch.isfinite(g).all() and g.norm() > 0 for g in grads)
        info[arm]["cached_objective_loss"] = float(loss.detach())
        info[arm]["cached_objective_gradient_norms"] = [float(g.norm()) for g in grads]
        vision, processor = load_arm(args, arm)
        inventory = freeze_prefix(vision, arm)
        pixel_rows = []
        for i in updated:
            with Image.open(paths[i]) as image:
                image = image.convert("RGB")
                pixel_rows.append(
                    processor(image)
                    if arm == "pe"
                    else processor(images=[image], return_tensors="pt")["pixel_values"][0]
                )
        pixel_cache = torch.stack(pixel_rows)
        info[arm]["pixel_sha256"] = digest({str(i): pixel_cache[j] for j, i in enumerate(updated)})
        info[arm]["cpu_setup_max_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        info[arm]["inventory"] = inventory
        info[arm]["trainable_parameters"] = sum(
            p.numel() for _, p in named_training_parameters(vision, arm) if p.requires_grad
        )
        info[arm]["frozen_parameters"] = sum(
            p.numel() for _, p in named_training_parameters(vision, arm) if not p.requires_grad
        )
        del vision, processor, pixel_cache, pixel_rows
        gc.collect()
    out = args.output / "initializers.npz"
    with out.open("xb") as stream:
        np.savez(stream, **initializers)
    rows, positions = member_bank_refresh_rows((3, 0, 3, 1))
    assert rows == (0, 1, 3) and positions == (1, 3, 2)
    raw = torch.eye(4, 128)
    refresh = member_bank_refresh_values(
        torch.zeros(4, 1024), raw, torch.tensor(positions), live_head=False
    )
    assert torch.equal(refresh, raw[torch.tensor(positions)])
    result = {
        "code": authority(),
        "initializers_sha256": sha(out),
        "initializers": info,
        "batches": batches,
        "updated_rows": updated,
        "target": target.tolist(),
        "source_preflight_sha256": sha(args.root / "cpu-preflight-v2.json"),
        "source_features_sha256": report["features_sha256"],
        "seed": 179032,
        "epoch": 1,
        "products_updated": 256,
        "images_updated": 512,
        "presentations": 1024,
        "claim_eligible": False,
        "quality_read": False,
    }
    save(args.output / "preflight.json", result)
    print("PASS initializers, distinct-valued duplicate fixture and frozen coverage/code authority")


def load_arm(args, arm):
    if arm == "pe":
        acquired = json.loads((args.root / "acquisition.json").read_text())
        vision, _ = load_visual(args.root, Path(acquired["checkpoint"]))
        processor = get_image_transform(224)
    else:
        assert args.large_snapshot.name == MODEL_REVISION
        assert all(sha(args.large_snapshot / name) == value for name, value in MODEL_HASHES.items())
        full = AutoModel.from_pretrained(
            args.large_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float32
        )
        vision = full.vision_model
        del full
        processor = AutoImageProcessor.from_pretrained(
            args.large_snapshot, local_files_only=True, backend="torchvision"
        )
    assert all(p.dtype == torch.float32 for p in vision.parameters())
    return vision, processor


def encode(vision, pixels, arm):
    return vision(pixels) if arm == "pe" else vision(pixel_values=pixels).pooler_output


def groups(vision, head, classifier, arm):
    blocks = vision.transformer.resblocks if arm == "pe" else vision.encoder.layers
    result = {
        f"block{i}": dict(blocks[i].named_parameters())
        for i in range(len(blocks) // 2, len(blocks))
    }
    for name in ("ln_post", "attn_pool") if arm == "pe" else ("post_layernorm", "head"):
        result[name] = dict(getattr(vision, name).named_parameters())
    if arm == "pe":
        result["proj"] = {"proj": vision.proj}
    result["compact_head"], result["classifier"] = (
        dict(head.named_parameters()),
        {"classifier": classifier},
    )
    return result


def gradients(vision, head, classifier, arm):
    result = {}
    params = (
        list(named_training_parameters(vision, arm))
        + [("head." + n, p) for n, p in head.named_parameters()]
        + [("classifier", classifier)]
    )
    for name, p in params:
        assert (p.grad is None) == (not p.requires_grad), name
        if p.grad is not None:
            norm = float(p.grad.detach().float().norm())
            assert np.isfinite(norm) and norm > 0, name
            result[name] = norm
    return result


def run_arm(args, arm, frozen):
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    vision, processor = load_arm(args, arm)
    inventory = freeze_prefix(vision, arm)
    source_manifest = json.loads((args.root / "cpu-preflight-v2.json").read_text())[
        "image_manifest"
    ]
    pixels = {}
    for i in frozen["updated_rows"]:
        with Image.open(args.dataset_root / source_manifest[i]["relative_path"]) as image:
            image = image.convert("RGB")
            value = (
                processor(image)
                if arm == "pe"
                else processor(images=[image], return_tensors="pt")["pixel_values"][0]
            )
            pixels[i] = value
    pixel_sha = digest({str(i): value for i, value in pixels.items()})
    assert pixel_sha == frozen["initializers"][arm]["pixel_sha256"]
    assert inventory == {k: tuple(v) for k, v in frozen["initializers"][arm]["inventory"].items()}
    vision = vision.cuda().train()
    init = np.load(args.output / "initializers.npz", allow_pickle=False)
    head = nn.Linear(1024, 128)
    head.load_state_dict(
        {key: torch.from_numpy(init[arm + ".head." + key]) for key in ("weight", "bias")}
    )
    head = head.cuda().train()
    classifier = nn.Parameter(torch.from_numpy(init[arm + ".classifier"]).cuda())
    bank = torch.from_numpy(init[arm + ".bank"]).cuda()
    target = torch.tensor(frozen["target"], device="cuda")
    positives = member_bank_positive_ordinals(np.asarray(frozen["target"])).cuda()
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
    subset = frozen["updated_rows"][:4]
    check_pixels = torch.stack([pixels[i] for i in subset]).cuda()
    with torch.no_grad():
        fp32 = encode(vision, check_pixels, arm).float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bf16 = encode(vision, check_pixels, arm).float()
        cached = torch.from_numpy(
            np.load(args.root / "pilot.features.npz", allow_pickle=False)[arm][subset]
        ).cuda()
        cos32 = F.cosine_similarity(fp32, bf16).tolist()
        coscache = F.cosine_similarity(cached, bf16).tolist()
        print(
            json.dumps(
                {
                    "arm": arm,
                    "rows": subset,
                    "bf16_fp32_cosine": cos32,
                    "cache_fp16_bf16_cosine": coscache,
                }
            ),
            flush=True,
        )
        assert min(cos32) >= 0.999 and min(coscache) >= 0.999
        readout_difference = float(
            (compact_head_features(cached, head) - compact_head_features(bf16, head)).abs().max()
        )
    del check_pixels, fp32, bf16, cached
    initial_frozen = digest(frozen_state(vision, arm, inventory))
    initial_groups = {
        name: digest(values) for name, values in groups(vision, head, classifier, arm).items()
    }
    losses, seconds, preclip, diagnostics = [], [], [], []
    for step, batch in enumerate(frozen["batches"], 1):
        index = torch.tensor(batch, device="cuda")
        torch.cuda.synchronize()
        tick = time.perf_counter()
        x = torch.stack([pixels[i] for i in batch]).cuda()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            source = encode(vision, x, arm)
        raw = compact_head_features(source, head)
        ce = sharded_mask_arcface_loss(
            raw,
            classifier,
            target[index],
            torch.arange(128, device="cuda").unsqueeze(0),
            margin=0.3,
            scale=64,
        )
        rank = member_bank_rank_loss(raw, bank, head, positives[index], index, live_head=False)
        loss = ce + 8 * rank
        assert torch.isfinite(loss)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
        optimizer.step()
        rows, positions = member_bank_refresh_rows(tuple(batch))
        bank[torch.tensor(rows, device="cuda")] = member_bank_refresh_values(
            source, raw, torch.tensor(positions, device="cuda"), live_head=False
        )
        assert all(torch.isfinite(p).all() for _, p in named_training_parameters(vision, arm))
        assert all(torch.isfinite(p).all() for p in list(head.parameters()) + [classifier])
        assert all(
            torch.isfinite(v).all()
            for state in optimizer.state.values()
            for v in state.values()
            if isinstance(v, torch.Tensor)
        )
        assert torch.isfinite(bank).all()
        torch.cuda.synchronize()
        seconds.append(time.perf_counter() - tick)
        losses.append(float(loss.detach()))
        preclip.append(float(norm))
        if step in (1, 16):
            diagnostics.append(
                {"step": step, "per_parameter_grad_norm": gradients(vision, head, classifier, arm)}
            )
    final_frozen = digest(frozen_state(vision, arm, inventory))
    final_groups = {
        name: digest(values) for name, values in groups(vision, head, classifier, arm).items()
    }
    assert final_frozen == initial_frozen
    assert all(final_groups[name] != initial for name, initial in initial_groups.items())
    checkpoint = {
        "vision": vision.state_dict(),
        "head": head.state_dict(),
        "classifier": classifier.detach(),
        "bank": bank,
        "rope": vision.rope.rope.state_dict() if arm == "pe" else None,
        "rope_freq": vision.rope.freq if arm == "pe" else None,
    }
    path = args.output / (arm + ".pt")
    torch.save(checkpoint, path)
    peak = torch.cuda.max_memory_allocated()
    assert peak < (10_000_000_000 if arm == "pe" else 16_000_000_000)
    return {
        "inventory": inventory,
        "frozen_sha256": initial_frozen,
        "final_frozen_sha256": final_frozen,
        "initial_groups": initial_groups,
        "final_groups": final_groups,
        "diagnostics": diagnostics,
        "losses": losses,
        "step_seconds": seconds,
        "preclip_norms": preclip,
        "pixel_sha256": pixel_sha,
        "bf16_fp32_cosine": cos32,
        "cache_fp16_bf16_cosine": coscache,
        "readout_max_abs_delta": readout_difference,
        "median_step_3_16_seconds": float(np.median(seconds[2:])),
        "peak_cuda_allocated_bytes": peak,
        "guarded_images_per_second": 1024 / sum(seconds),
        "arm_wall_seconds": time.perf_counter() - started,
        "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_sha256": sha(path),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("root", "dataset-root", "large-snapshot", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--check-startup-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(179032)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.preflight_only:
        args.output.mkdir(exist_ok=False)
        metadata(args)
        return
    frozen = json.loads((args.output / "preflight.json").read_text())
    if args.check_startup_only:
        check_startup(args, frozen)
        for key in ("source_preflight_sha256", "initializers_sha256", "source_features_sha256"):
            try:
                check_startup(args, {**frozen, key: "0" * 64})
            except AssertionError:
                pass
            else:
                raise AssertionError("changed authority accepted: " + key)
        altered = {**frozen, "code": {**frozen["code"], "core/vision_encoder/pe.py": "0" * 64}}
        try:
            check_startup(args, altered)
        except AssertionError:
            pass
        else:
            raise AssertionError("changed lazy code accepted")
        print("PASS startup authority; changed source/initializer/features/code rejected")
        return
    started = time.perf_counter()
    assert torch.cuda.is_available() and not (args.output / "receipt.json").exists()
    check_startup(args, frozen)
    reports = {}
    for arm in ("large", "pe"):
        reports[arm] = run_arm(args, arm, frozen)
        torch._C._cuda_clearCublasWorkspaces()
        gc.collect()
        torch.cuda.empty_cache()
        reports[arm]["post_cleanup_allocated_bytes"] = torch.cuda.memory_allocated()
        assert reports[arm]["post_cleanup_allocated_bytes"] < 8 * 1024**2
    ratio = reports["pe"]["median_step_3_16_seconds"] / reports["large"]["median_step_3_16_seconds"]
    result = {
        "arms": reports,
        "step_ratio": ratio,
        "advance": ratio <= 0.8,
        "whole_wall_seconds": time.perf_counter() - started,
        "claim_eligible": False,
        "quality_read": False,
        "preflight_sha256": sha(args.output / "preflight.json"),
    }
    assert result["whole_wall_seconds"] <= 180
    save(args.output / "receipt.json", result)
    print(
        json.dumps(
            {
                "advance": result["advance"],
                "step_ratio": ratio,
                "whole_wall_seconds": result["whole_wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
