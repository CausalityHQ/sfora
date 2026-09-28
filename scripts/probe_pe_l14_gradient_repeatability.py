#!/usr/bin/env python3
"""Measure identical uncheckpointed native FP16 gradient repeatability; no candidate."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import pe_l14_training as l14
import pe_l14_checkpoint as replay
import train_inshop_pe_pair as pair
from pe_core_training import named_training_parameters

PREFLIGHT_SHA = "be5a8bdcfd1749340f220d5669cebfb1c0d8b30d3c08667a0b72a4fb38701277"
INIT = Path("/home/riomus/runs/sfora-pe-l14-init-v1")


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--script-sha256", required=True)
    p.add_argument("--checkpoint-helper-sha256", required=True)
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    assert pair.sha(Path(__file__)) == args.script_sha256
    assert pair.sha(root / "pe_l14_checkpoint.py") == args.checkpoint_helper_sha256
    assert Path(replay.__file__).resolve() == root / "pe_l14_checkpoint.py"
    assert not args.output.exists()
    if args.device == "cpu":
        assert not torch.cuda.is_available()
    else:
        assert torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    source_args, frozen, _ = l14.control(root)
    assert pair.sha(INIT / "preflight.json") == PREFLIGHT_SHA
    initial = json.loads((INIT / "preflight.json").read_text())
    assert pair.sha(INIT / "initializers.npz") == initial["initializers_sha256"]
    assert all(pair.sha(root / n) == h for n, h in initial["code"].items())
    model, processor = l14.load()
    inventory = l14.freeze(model)
    assert inventory == {
        k: tuple(v) for k, v in initial["initializers"]["inventory"].items()
    }
    model.to(args.device).train()
    head = nn.Linear(1024, 128)
    with np.load(INIT / "initializers.npz", allow_pickle=False) as a:
        head.load_state_dict(
            {k: torch.from_numpy(a["pe.head." + k]) for k in ("weight", "bias")}
        )
        classifier = nn.Parameter(
            torch.from_numpy(a["pe.classifier"].copy()).to(args.device)
        )
        bank = torch.from_numpy(a["pe.bank"].copy()).to(args.device)
    head.to(args.device)
    # Same actual first2 fit files, their own labels and fixed unaugmented RGB.
    rows = (0, 1)
    images, rgb_sha = pair.augmented_images(
        source_args.dataset_root, frozen["fit_manifest"], rows, None
    )
    pixels = pair.pixels(processor, images, "pe").to(args.device)
    labels = torch.tensor(frozen["target"], device=args.device)
    indices = torch.tensor(rows, device=args.device)
    positives = pair.smoke.member_bank_positive_ordinals(
        np.asarray(frozen["target"], dtype=np.int64), allow_singletons=True
    ).to(args.device)
    names_params = (
        [(n, p) for n, p in named_training_parameters(model, "pe") if p.requires_grad]
        + [("head." + n, p) for n, p in head.named_parameters()]
        + [("classifier", classifier)]
    )
    params = tuple(p for _, p in names_params)
    frozen_hash = pair.smoke.digest(l14.frozen_state(model, inventory))
    state_hash = pair.smoke.digest(model.state_dict())
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def run():
        with torch.autocast(
            args.device, dtype=torch.float16, enabled=args.device == "cuda"
        ):
            source = model(pixels)
        raw = pair.smoke.compact_head_features(source, head)
        loss = pair.smoke.sharded_mask_arcface_loss(
            raw,
            classifier,
            labels[indices],
            torch.arange(128, device=args.device).unsqueeze(0),
            margin=0.3,
            scale=64,
        ) + 8 * pair.smoke.member_bank_rank_loss(
            raw, bank, head, positives[indices], indices, live_head=False
        )
        assert torch.isfinite(loss)
        gradients = torch.autograd.grad(
            loss * (128 if args.device == "cuda" else 1), params
        )
        assert all(torch.isfinite(g).all() and g.norm() > 0 for g in gradients)
        return (
            source.detach().cpu(),
            raw.detach().cpu(),
            float(loss.detach()),
            tuple(g.cpu() for g in gradients),
        )

    baseline = run()
    candidate = run()
    assert not pixels.requires_grad
    assert (
        torch.equal(baseline[0], candidate[0])
        and torch.equal(baseline[1], candidate[1])
        and baseline[2] == candidate[2]
    )
    differences = {}
    for (name, _), a, b in zip(names_params, baseline[3], candidate[3], strict=True):
        delta = a.double() - b.double()
        differences[name] = {
            "exact": torch.equal(a, b),
            "max_abs": float(delta.abs().max()),
            "relative_l2": float(delta.norm() / a.double().norm()),
        }
    gradient_repeatable = all(v["exact"] for v in differences.values())

    assert pair.smoke.digest(model.state_dict()) == state_hash
    assert pair.smoke.digest(l14.frozen_state(model, inventory)) == frozen_hash
    model.eval()
    with (
        torch.no_grad(),
        torch.autocast(args.device, dtype=torch.float16, enabled=args.device == "cuda"),
    ):
        serving = model(pixels).float().cpu()
    assert torch.equal(serving, baseline[0].float())
    peak = torch.cuda.max_memory_allocated() if args.device == "cuda" else 0
    print(
        json.dumps({"phase": "parity-complete", "peak_cuda_allocated_bytes": peak}),
        flush=True,
    )
    assert peak < 10_000_000_000 and time.perf_counter() - started < 120
    pair.smoke.save(
        args.output,
        {
            "pass": True,
            "device": args.device,
            "precision": "fp16-autocast/scaled128" if args.device == "cuda" else "fp32",
            "native_output_exact": True,
            "compact_output_exact": True,
            "loss_exact": True,
            "baseline_gradient_bitwise_repeatable": gradient_repeatable,
            "gradient_differences": differences,
            "checkpoint_candidate_executed": False,
            "parameter_tensors": len(params),
            "source_state_unchanged": True,
            "frozen_rotary_unchanged": True,
            "normal_serving_output_exact": True,
            "input_requires_grad": False,
            "rgb_sha256": rgb_sha,
            "gradient_sha256": pair.smoke.digest(
                dict(zip((n for n, p in names_params), baseline[3], strict=True))
            ),
            "loss": baseline[2],
            "seconds": time.perf_counter() - started,
            "peak_cuda_allocated_bytes": peak,
            "optimizer_updates": 0,
            "held_images": 0,
            "quality_read": False,
            "script_sha256": args.script_sha256,
            "checkpoint_helper_sha256": args.checkpoint_helper_sha256,
            "initialization_preflight_sha256": PREFLIGHT_SHA,
            "torch_checkpoint_sha256": pair.sha(Path(torch.utils.checkpoint.__file__)),
        },
    )
    print(
        "COMPLETE identical native baseline repeatability diagnostic; no candidate or optimizer update",
        flush=True,
    )


if __name__ == "__main__":
    main()
