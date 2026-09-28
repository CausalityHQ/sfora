#!/usr/bin/env python3
"""CPU-only independent original/final checkpoint integrity and cost replay."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import probe_inshop_pe_training_smoke as smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("root", "output", "dataset-root", "large-snapshot"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--vision-precision", choices=("fp16", "bf16"), required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    frozen = json.loads((args.output / "preflight.json").read_text())
    receipt = json.loads((args.output / "receipt.json").read_text())
    smoke.check_startup(args, frozen)
    assert smoke.sha(args.output / "preflight.json") == receipt["preflight_sha256"]
    assert not receipt["quality_read"] and not receipt["claim_eligible"]
    init = np.load(args.output / "initializers.npz", allow_pickle=False)
    reports = {}
    for arm in ("large", "pe"):
        r = receipt["arms"][arm]
        vision, processor = smoke.load_arm(args, arm)
        inventory = smoke.freeze_prefix(vision, arm)
        assert inventory == {k: tuple(v) for k, v in r["inventory"].items()}
        if arm == "pe":
            vision.rope.update_grid(torch.device("cpu"), 14, 14)
        head = nn.Linear(1024, 128)
        head.load_state_dict(
            {k: torch.from_numpy(init[arm + ".head." + k]) for k in ("weight", "bias")}
        )
        classifier = nn.Parameter(torch.from_numpy(init[arm + ".classifier"].copy()))
        assert smoke.digest(smoke.frozen_state(vision, arm, inventory)) == r["frozen_sha256"]
        initial_groups = {
            n: smoke.digest(v) for n, v in smoke.groups(vision, head, classifier, arm).items()
        }
        assert initial_groups == r["initial_groups"]
        path = args.output / (arm + ".pt")
        assert smoke.sha(path) == r["checkpoint_sha256"]
        saved = torch.load(path, weights_only=True, map_location="cpu", mmap=True)
        vision.load_state_dict(saved["vision"], strict=True)
        head.load_state_dict(saved["head"], strict=True)
        classifier = nn.Parameter(saved["classifier"])
        if arm == "pe":
            vision.rope.rope.load_state_dict(saved["rope"], strict=True)
            assert torch.equal(vision.rope.freq, saved["rope_freq"])
        assert (
            smoke.digest(smoke.frozen_state(vision, arm, inventory))
            == r["final_frozen_sha256"]
            == r["frozen_sha256"]
        )
        final_groups = {
            n: smoke.digest(v) for n, v in smoke.groups(vision, head, classifier, arm).items()
        }
        assert final_groups == r["final_groups"]
        assert all(final_groups[n] != h for n, h in initial_groups.items())
        assert saved["bank"].shape == (1024, 128) and saved["classifier"].shape == (512, 128)
        assert torch.isfinite(saved["bank"]).all() and torch.isfinite(classifier).all()
        assert torch.allclose(saved["bank"].norm(dim=1), torch.ones(1024), atol=1e-5, rtol=0)
        expected = set(inventory["trainable"]) | {"head.weight", "head.bias", "classifier"}
        nulls = set(r["null_key_bias_parameters"])
        assert len(nulls) == (12 if arm == "large" else 0)
        assert all(
            n.startswith("encoder.layers.") and n.endswith(".self_attn.k_proj.bias") for n in nulls
        )
        assert [d["step"] for d in r["diagnostics"]] == [1, 16]
        for d in r["diagnostics"]:
            values = d["per_parameter_grad_norm"]
            assert set(values) == expected
            assert all(np.isfinite(v) and (v > 0 or n in nulls) for n, v in values.items())
        assert r["grad_scaler_scales"] == [128.0] * 16
        assert (
            min(r["precision_fp32_cosine"]) >= 0.999
            and min(r["cache_fp16_precision_cosine"]) >= 0.999
        )
        assert len(r["step_seconds"]) == len(r["losses"]) == len(r["preclip_norms"]) == 16
        assert all(np.isfinite(x) and x > 0 for x in r["step_seconds"] + r["preclip_norms"])
        assert all(np.isfinite(x) for x in r["losses"])
        assert float(np.median(r["step_seconds"][2:])) == r["median_step_3_16_seconds"]
        assert 1024 / sum(r["step_seconds"]) == r["guarded_images_per_second"]
        assert r["peak_cuda_allocated_bytes"] < (
            16_000_000_000 if arm == "large" else 10_000_000_000
        )
        assert r["post_cleanup_allocated_bytes"] < 8 * 1024**2
        reports[arm] = {
            "checkpoint_sha256": r["checkpoint_sha256"],
            "frozen_sha256": r["frozen_sha256"],
            "changed_groups": sorted(final_groups),
        }
        del vision, processor, head, classifier, saved
        gc.collect()
    ratio = (
        receipt["arms"]["pe"]["median_step_3_16_seconds"]
        / receipt["arms"]["large"]["median_step_3_16_seconds"]
    )
    assert ratio == receipt["step_ratio"] and receipt["advance"] == (ratio <= 0.8)
    assert receipt["whole_wall_seconds"] <= 180
    smoke.save(
        args.output / "checkpoint-audit.json",
        {"arms": reports, "step_ratio": ratio, "pass": True, "quality_read": False},
    )
    print(
        "PASS checkpoint bytes, frozen/group/gradient/scaler integrity and cost"
    )


if __name__ == "__main__":
    main()
