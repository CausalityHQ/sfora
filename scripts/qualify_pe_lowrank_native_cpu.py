#!/usr/bin/env python3
"""Actual native PE rank32 qualification: fit-only, no optimizer or CUDA."""

import argparse
import gc
import json
import time
from pathlib import Path
from types import SimpleNamespace

import torch

import pe_lowrank_adaptation as lowrank
import train_inshop_pe_pair as pair
from pe_core_training import freeze_prefix, frozen_state


def bits_equal(a, b):
    return torch.equal(
        a.contiguous().view(torch.int32), b.contiguous().view(torch.int32)
    )


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--authority", type=Path, required=True)
    cli.add_argument("--result", type=Path, required=True)
    args = cli.parse_args()
    started = time.perf_counter()
    assert not torch.cuda.is_available()
    authority = json.loads(args.authority.read_text())
    root = Path(__file__).resolve().parent
    for name, digest in authority.items():
        assert pair.sha(root / name) == digest, name
    assert Path(lowrank.__file__).resolve() == root / "pe_lowrank_adaptation.py"
    assert not args.result.exists()
    attempt = args.result.with_suffix(".attempt.json")
    with attempt.open("x") as file:
        json.dump(authority, file)
    torch.set_num_threads(4)
    source = SimpleNamespace(
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
    pair.check_startup(source)
    manifest = json.loads((source.cache / "preflight.json").read_text())["manifest"]
    images, rgb_sha = pair.augmented_images(source.dataset_root, manifest, (0, 1), None)
    vision, processor = pair.smoke.load_arm(source, "pe")
    vision.eval()
    pixels = pair.pixels(processor, images, "pe")
    with torch.no_grad():
        original = vision(pixels).detach()
    native_shapes = {n: tuple(v.shape) for n, v in vision.state_dict().items()}
    inventory = freeze_prefix(vision, "pe")
    frozen_sha = pair.smoke.digest(frozen_state(vision, "pe", inventory))
    sites = lowrank.install(vision, inventory)
    groups = lowrank.parameter_groups(vision, sites)
    assert [g["lr"] for g in groups] == [1e-5, 1e-4]
    originals = {
        n: getattr(m.parametrizations, k).original.detach().clone()
        for n, (m, k) in sites.items()
    }
    with torch.no_grad():
        zero = vision(pixels)
    assert bits_equal(original, zero), "zero-B native source output differs"
    print("PASS native source bitwise zero-B and optimizer coverage", flush=True)
    factors = [getattr(m.parametrizations, k)[0] for m, k in sites.values()]

    def gradients(vision, factors, sites, first):
        vision.zero_grad(set_to_none=True)
        output = vision(pixels)
        assert torch.isfinite(output).all()
        (output * torch.linspace(-1, 1, output.shape[1])).sum().backward()
        norms = []
        for factor in factors:
            assert factor.A.grad is not None and factor.B.grad is not None
            a, b = factor.A.grad, factor.B.grad
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            assert (torch.count_nonzero(a) == 0) if first else (a.norm() > 0)
            assert b.norm() > 0
            norms.append([float(a.norm()), float(b.norm())])
        assert all(
            getattr(m.parametrizations, k).original.grad is None
            for m, k in sites.values()
        )
        return norms

    first_norms = gradients(vision, factors, sites, True)
    with torch.no_grad():
        for factor in factors:
            # Synthetic nonzero fixture only: no optimizer/data update.
            factor.B.fill_(1e-4)
    later_norms = gradients(vision, factors, sites, False)
    vision.zero_grad(set_to_none=True)
    with torch.no_grad():
        updated = vision(pixels).detach()
    assert not bits_equal(original, updated)
    assert frozen_sha == pair.smoke.digest(frozen_state(vision, "pe", inventory))
    assert all(
        bits_equal(originals[n], getattr(m.parametrizations, k).original)
        for n, (m, k) in sites.items()
    )
    lowrank.merge(sites)
    assert native_shapes == {n: tuple(v.shape) for n, v in vision.state_dict().items()}
    with torch.no_grad():
        assert bits_equal(updated, vision(pixels)), "native merge output differs"
    checkpoint = args.result.with_suffix(".pt")
    assert not checkpoint.exists()
    torch.save(
        {
            "vision": vision.state_dict(),
            "rope": vision.rope.rope.state_dict(),
            "rope_freq": vision.rope.freq,
        },
        checkpoint,
    )
    del groups, originals, factors, sites, vision
    gc.collect()
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
    assert bits_equal(fresh.rope.freq, saved["rope_freq"])
    with torch.no_grad():
        assert bits_equal(updated, fresh(pixels)), (
            "strict-loaded updated native output differs"
        )
    assert frozen_sha == pair.smoke.digest(frozen_state(fresh, "pe", inventory))
    pair.executing_authority(
        root, json.loads((source.output / "preflight.json").read_text())["code"]
    )
    result = {
        "pass": True,
        "cuda": False,
        "optimizer_updates": 0,
        "held_images": 0,
        "fit_rows": [0, 1],
        "rgb_sha256": rgb_sha,
        "authority": authority,
        "sites": 24,
        "factors": 48,
        "factor_elements": 2359296,
        "native_state_keys": len(native_shapes),
        "first_gradient_norms": first_norms,
        "nonzero_fixture_gradient_norms": later_norms,
        "bitwise_source_merge_reload": True,
        "checkpoint_sha256": pair.sha(checkpoint),
        "seconds": time.perf_counter() - started,
    }
    pair.smoke.save(args.result, result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
