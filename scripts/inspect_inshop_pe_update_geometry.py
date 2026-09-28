#!/usr/bin/env python3
"""CPU-only saved parameter-update geometry of the stopped PE/Large pair."""

import argparse
import gc
import json
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

import probe_inshop_pe_training_smoke as smoke
from diagnose_inshop_pe_learning_attribution import PRIOR_PREFLIGHT, PRIOR_SHA
from train_inshop_pe_pair import check_startup


def update_geometry(initial, final):
    assert initial.keys() == final.keys()
    old2 = delta2 = 0.0
    for name, before in initial.items():
        after = final[name]
        assert before.shape == after.shape
        assert torch.isfinite(before).all() and torch.isfinite(after).all()
        old2 += float(before.double().square().sum())
        delta2 += float((after.double() - before.double()).square().sum())
    assert old2 > 0
    return {
        "initial_norm": old2**0.5,
        "update_norm": delta2**0.5,
        "relative_update": (delta2 / old2) ** 0.5,
    }


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for n in (
        "root",
        "cache",
        "output",
        "dataset-root",
        "large-snapshot",
        "mechanics-dir",
        "result",
    ):
        p.add_argument("--" + n, type=Path, required=True)
    args = p.parse_args()
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    started = time.perf_counter()
    assert smoke.sha(args.output / "receipt.json") == PRIOR_SHA
    args.preflight_sha256 = PRIOR_PREFLIGHT
    frozen = check_startup(args)
    prior = json.loads((args.output / "receipt.json").read_text())
    assert prior["decision"] == "STOP" and not prior["advance"]
    reports = {}
    with np.load(args.output / "initializers.npz", allow_pickle=False) as init:
        for arm in ("large", "pe"):
            r = prior["arms"][arm]
            assert smoke.sha(args.output / (arm + ".pt")) == r["checkpoint_sha256"]
            vision, processor = smoke.load_arm(args, arm)
            smoke.freeze_prefix(vision, arm)
            head = nn.Linear(1024, 128)
            head.load_state_dict(
                {
                    k: torch.from_numpy(init[arm + ".head." + k].copy())
                    for k in ("weight", "bias")
                }
            )
            classifier = nn.Parameter(
                torch.from_numpy(init[arm + ".classifier"].copy())
            )
            before = {
                g: {k: v.clone() for k, v in values.items()}
                for g, values in smoke.groups(vision, head, classifier, arm).items()
            }
            assert {g: smoke.digest(v) for g, v in before.items()} == r[
                "initial_groups"
            ]
            saved = torch.load(
                args.output / (arm + ".pt"),
                weights_only=True,
                map_location="cpu",
                mmap=True,
            )
            vision.load_state_dict(saved["vision"], strict=True)
            head.load_state_dict(saved["head"], strict=True)
            classifier = nn.Parameter(saved["classifier"])
            after = smoke.groups(vision, head, classifier, arm)
            assert {g: smoke.digest(v) for g, v in after.items()} == r["final_groups"]
            reports[arm] = {g: update_geometry(before[g], v) for g, v in after.items()}
            assert all(v["update_norm"] > 0 for v in reports[arm].values())
            del before, after, saved, vision, processor, head, classifier
            gc.collect()
    assert smoke.sha(args.output / "receipt.json") == PRIOR_SHA
    assert smoke.sha(args.output / "preflight.json") == PRIOR_PREFLIGHT
    assert all(smoke.sha(args.root / n) == h for n, h in frozen["code"].items())
    smoke.save(
        args.result,
        {
            "groups": reports,
            "prior_receipt_sha256": PRIOR_SHA,
            "executed_script_sha256": smoke.sha(Path(__file__)),
            "whole_wall_seconds": time.perf_counter() - started,
            "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "quality_read": False,
            "claim_eligible": False,
            "prior_stop_preserved": True,
        },
    )
    print(json.dumps(reports), flush=True)


if __name__ == "__main__":
    main()
