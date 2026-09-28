#!/usr/bin/env python3
"""CPU replay of actual augmented-pair checkpoints, packed held scores and decision."""

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import probe_inshop_pe_training_smoke as smoke
from compare_inshop_sop_warmstart_100 import packed_quality
from score_inshop_crop_view_pair import bootstrap_lower
from train_inshop_pe_pair import check_startup


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
    p.add_argument("--preflight-sha256", required=True)
    p.add_argument("--receipt-sha256", required=True)
    args = p.parse_args()
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    assert smoke.sha(args.output / "receipt.json") == args.receipt_sha256
    frozen = check_startup(args)
    receipt = json.loads((args.output / "receipt.json").read_text())
    assert receipt["preflight_sha256"] == args.preflight_sha256
    assert (
        receipt["quality_read"] == "TRAIN-held only" and not receipt["claim_eligible"]
    )
    assert receipt["whole_wall_seconds"] < 600
    init = np.load(args.output / "initializers.npz", allow_pickle=False)
    quality, audits = {}, {}
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
        assert (
            smoke.digest(smoke.frozen_state(vision, arm, inventory))
            == r["frozen_sha256"]
        )
        before = {
            n: smoke.digest(v)
            for n, v in smoke.groups(vision, head, classifier, arm).items()
        }
        assert before == r["initial_groups"]
        assert smoke.sha(args.output / (arm + ".pt")) == r["checkpoint_sha256"]
        saved = torch.load(
            args.output / (arm + ".pt"),
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )
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
        after = {
            n: smoke.digest(v)
            for n, v in smoke.groups(vision, head, classifier, arm).items()
        }
        assert after == r["final_groups"] and all(
            after[n] != v for n, v in before.items()
        )
        assert saved["bank"].shape == (13283, 128) and saved["classifier"].shape == (
            2004,
            128,
        )
        assert torch.isfinite(saved["bank"]).all()
        assert torch.allclose(
            saved["bank"].norm(dim=1), torch.ones(13283), atol=1e-5, rtol=0
        )
        assert all(
            torch.isfinite(p).all()
            for _, p in smoke.named_training_parameters(vision, arm)
        )
        assert all(
            torch.isfinite(p).all() for p in list(head.parameters()) + [classifier]
        )
        expected = set(inventory["trainable"]) | {
            "head.weight",
            "head.bias",
            "classifier",
        }
        nulls = {
            n
            for n in inventory["trainable"]
            if arm == "large"
            and n.startswith("encoder.layers.")
            and n.endswith(".self_attn.k_proj.bias")
        }
        assert len(nulls) == (12 if arm == "large" else 0)
        assert [d["step"] for d in r["diagnostics"]] == [1, 100]
        for d in r["diagnostics"]:
            values = d["per_parameter_grad_norm"]
            assert set(values) == expected
            assert all(
                np.isfinite(v) and (v > 0 or n in nulls) for n, v in values.items()
            )
        assert r["grad_scaler_scales"] == [128.0] * 100
        assert all(min(v) >= 0.999 for v in r["calibration"].values())
        assert (
            len(r["step_seconds"]) == len(r["losses"]) == len(r["preclip_norms"]) == 100
        )
        assert all(
            np.isfinite(v) and v > 0 for v in r["step_seconds"] + r["preclip_norms"]
        )
        assert all(np.isfinite(v) for v in r["losses"])
        assert np.median(r["step_seconds"][2:]) == r["median_step_3_100_seconds"]
        assert 6400 / sum(r["step_seconds"]) == r["guarded_images_per_second"]
        assert r["peak_cuda_allocated_bytes"] < (
            16_000_000_000 if arm == "large" else 10_000_000_000
        )
        assert (
            json.loads((args.output / (arm + ".cleanup.json")).read_text())[
                "allocated_cuda_bytes"
            ]
            < 8 * 1024**2
        )
        assert r["rank_active_updates"] == sum(frozen["rank_active"])
        assert len(r["rgb_sha256"]) == 100
        path = args.output / (arm + ".held.npy")
        assert smoke.sha(path) == r["held_sha256"]
        values = np.load(path, allow_pickle=False)
        assert values.shape == (12599, 128) and values.dtype == np.float32
        assert np.isfinite(values).all() and np.allclose(
            np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0
        )
        quality[arm] = packed_quality(
            values,
            tuple(m["product"] for m in frozen["held_manifest"]),
            frozen["query"],
            frozen["gallery"],
            device=torch.device("cpu"),
        )
        for key in ("per_query_r1", "per_query_ap", "recall_at_1", "map_at_r"):
            assert (
                np.max(np.abs(np.asarray(quality[arm][key]) - r["quality"][key])) < 1e-6
            )
        audits[arm] = {
            "checkpoint_sha256": r["checkpoint_sha256"],
            "held_sha256": r["held_sha256"],
            "changed_groups": sorted(after),
        }
        del vision, processor, head, classifier, saved, values
        gc.collect()
    assert receipt["arms"]["large"]["rgb_sha256"] == receipt["arms"]["pe"]["rgb_sha256"]
    products = np.asarray(
        [frozen["held_manifest"][i]["product"] for i in frozen["query"]]
    )
    for name, key in (("r1", "per_query_r1"), ("map_at_r", "per_query_ap")):
        delta = np.asarray(quality["pe"][key]) - quality["large"][key]
        pair = receipt["paired"][name]
        assert abs(100 * delta.mean() - pair["delta_pp"]) < 1e-4
        for labels, field in (
            (products, "product_bootstrap_95_pp"),
            (np.arange(len(delta)), "query_bootstrap_95_pp"),
        ):
            interval = [
                100 * bootstrap_lower(delta, labels),
                -100 * bootstrap_lower(-delta, labels),
            ]
            assert np.max(np.abs(np.asarray(interval) - pair[field])) < 1e-4
    ratio = (
        receipt["arms"]["pe"]["median_step_3_100_seconds"]
        / receipt["arms"]["large"]["median_step_3_100_seconds"]
    )
    assert ratio == receipt["step_time_ratio"]
    advance = (
        receipt["paired"]["r1"]["delta_pp"] >= -0.5
        and receipt["paired"]["map_at_r"]["delta_pp"] >= -1
        and ratio <= 0.8
    )
    assert receipt["advance"] == advance
    smoke.save(
        args.output / "checkpoint-score-audit.json",
        {
            "arms": audits,
            "pass": True,
            "decision": receipt["decision"],
            "quality_read": "TRAIN-held score replay only",
            "claim_eligible": False,
        },
    )
    print(
        "PASS actual checkpoints/frozen state/groups/bank/scales/packed held-score/CI/decision replay"
    )


if __name__ == "__main__":
    main()
