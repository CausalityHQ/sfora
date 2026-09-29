#!/usr/bin/env python3
"""Fresh full TRAIN-held checkpoint diagnosis; never rescues the closed pilot."""

import argparse
import copy
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pe_large_pool as native


def startup(root, execution_sha):
    pair = native.pair
    manifest = root / "large-pool-checkpoint-validation-execution.json"
    assert pair.sha(manifest) == execution_sha
    code = json.loads(manifest.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items()), "validation code differs"
    control, frozen, prior = native.control(root)
    checkpoint = Path("/home/riomus/runs/sfora-large-native-pool-pilot-v1/pe.pt")
    assert pair.sha(checkpoint) == "a7e3d8d413aba67dd0d1472d5b70537a72de902ecbdac505833af9cbcaa1a796"
    training = checkpoint.parent / "training.json"
    assert pair.sha(training) == "ebf8521b70e46280824ada916fa83bb7488eb856afb865edc6630b636d24d409"
    assert not (checkpoint.parent / "receipt.json").exists()
    t = json.loads(training.read_text())
    assert len(t["step_seconds"]) == 100 and t["scales"] == [128] * 100
    assert t["rgb_sha256"] == prior["rgb_sha256"]
    assert np.median(t["step_seconds"][2:]) == t["median_step_3_100_seconds"] <= 0.71769696
    assert t["caller_cpu_rng_unchanged"] and t["one_pending_cpu_batch"]
    assert [d["step"] for d in t["diagnostics"]] == [1, 100]
    assert all(len(d["data_gradient_norms"]) == 14 and all(v > 0 for v in d["data_gradient_norms"].values()) for d in t["diagnostics"])
    mechanics = Path("/home/riomus/runs/sfora-large-native-pool-mechanics-v1")
    assert pair.sha(mechanics / "receipt.json") == "4158bb3be1bc5d88285cd567c4de8e37d7215aee2469b19f66e5b5c63d893be0"
    m = json.loads((mechanics / "receipt.json").read_text())
    assert pair.sha(mechanics / "training.json") == m["training_sha256"]
    mt = json.loads((mechanics / "training.json").read_text())
    assert t["losses"][:17] == mt["losses"] and t["scales"][:17] == mt["scales"]
    cpu = Path("/home/riomus/runs/sfora-large-immutable-amp-cpu-v1/preflight.json")
    assert pair.sha(cpu) == "f48b025eb5921d51cfb7ca98f8500c2c32e23840bf77cf6fc37621b966173591"
    info = json.loads(cpu.read_text())
    assert all(pair.sha(root / n) == h for n, h in info["code"].items())
    for p, digest in info["native_environment"]["native_files"].items():
        assert pair.sha(Path(p)) == digest
    dense = json.loads((native.INIT / "receipt.json").read_text())["arms"]["pe"]
    return control, frozen, checkpoint, t, info, dense, code


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-startup-only", action="store_true")
    args = parser.parse_args()
    pair = native.pair
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    control, frozen, checkpoint, t, info, dense, code = startup(root, args.execution_sha256)
    if args.check_startup_only:
        assert not torch.cuda.is_available()
        print("PASS fresh validation CPU startup: source/control/checkpoint/training/mechanics/code; no images/scores/CUDA")
        return
    assert not args.output.exists() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, processor = pair.smoke.load_arm(control, "large")
    native.freeze(model)
    assert pair.smoke.digest(native.frozen_state(model)) == t["frozen_sha256"]
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict(saved["vision"], strict=True)
    model.eval()
    assert pair.smoke.digest(native.whole_state(model)) == info["whole_checkpoint_state_sha256"]
    other = type(model)(copy.deepcopy(model.config)).float().eval()
    other.load_state_dict(saved["vision"], strict=True)
    native.freeze(other)
    heads = [nn.Linear(1024, 128).eval() for _ in range(2)]
    for h in heads:
        h.load_state_dict(saved["head"], strict=True)
        assert pair.smoke.digest(h.state_dict()) == t["final_group_sha256"]["compact_head"]
        h.cuda()
    assert pair.smoke.digest({"classifier": saved["classifier"]}) == t["final_group_sha256"]["classifier"]
    assert saved["bank"].shape == (13283, 128) and torch.isfinite(saved["bank"]).all()
    assert torch.allclose(saved["bank"].norm(dim=1), torch.ones(13283), atol=1e-5, rtol=0)
    del saved
    models = (model.cuda(), other.cuda())
    runtime = native.runtime_identity(model)
    assert runtime == native.runtime_identity(other)
    original = info["whole_checkpoint_state_sha256"]
    assert all(pair.smoke.digest(native.whole_state(m)) == original for m in models)
    assert json.loads(json.dumps(native.environment(model, processor))) == info["native_environment"]
    torch.cuda.reset_peak_memory_stats()
    rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    args.output.mkdir(exist_ok=False)
    chunks = []

    @torch.inference_mode()
    def encode(batch):
        images, _ = pair.augmented_images(control.dataset_root, batch, tuple(range(len(batch))), None)
        x = pair.pixels(processor, images, "large").cuda()
        a, b = native.verified_features(other, model, x)
        va = F.normalize(pair.smoke.compact_head_features(a, heads[1]).float(), dim=1)
        vb = F.normalize(pair.smoke.compact_head_features(b, heads[0]).float(), dim=1)
        assert torch.equal(va, vb)
        chunks.append(vb.cpu().numpy())
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        assert runtime == native.runtime_identity(model) == native.runtime_identity(other)
        return va.cpu().numpy()

    pair.export_features(frozen["held_manifest"], encode, args.output / "loaded-held.npy", width=128, batch_size=32)
    values = np.load(args.output / "loaded-held.npy", allow_pickle=False)
    independent = np.concatenate(chunks)
    assert values.shape == (12599, 128) and np.array_equal(values, independent)
    np.save(args.output / "reference-held.npy", independent, allow_pickle=False)
    labels = tuple(r["product"] for r in frozen["held_manifest"])
    quality = pair.packed_quality(values, labels, frozen["query"], frozen["gallery"])
    reference = pair.packed_quality(independent, labels, frozen["query"], frozen["gallery"])
    assert all(np.array_equal(np.asarray(quality[k]), np.asarray(reference[k])) for k in quality)
    products = np.asarray(labels)[frozen["query"]]
    intervals = {}
    for k in ("per_query_r1", "per_query_ap"):
        delta = np.asarray(quality[k]) - np.asarray(dense["quality"][k])
        intervals[k] = {"mean_delta": float(np.mean(delta)), "product_lower95": pair.bootstrap_lower(delta, products), "product_upper95": -pair.bootstrap_lower(-delta, products), "query_lower95": pair.bootstrap_lower(delta, np.arange(len(delta))), "query_upper95": -pair.bootstrap_lower(-delta, np.arange(len(delta)))}
    floors_pass = bool(quality["recall_at_1"] >= 0.951720176 and quality["map_at_r"] >= 0.776237120 and all(v["product_lower95"] > 0 for v in intervals.values()))
    assert all(pair.smoke.digest(native.whole_state(m)) == original for m in models)
    assert all(pair.smoke.digest(native.frozen_state(m)) == t["frozen_sha256"] for m in models)
    assert all(p.dtype == torch.float32 for m in models for p in m.parameters())
    assert runtime == native.runtime_identity(model) == native.runtime_identity(other)
    assert json.loads(json.dumps(native.environment(model, processor))) == info["native_environment"]
    assert torch.equal(rng, torch.random.get_rng_state())
    assert all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    assert all(pair.sha(root / n) == h for n, h in code.items())
    pair.smoke.save(args.output / "receipt.json", {"diagnostic_floors_pass": floors_pass, "original_pilot_status": "terminal whole-job timeout; not rescued", "checkpoint_sha256": pair.sha(checkpoint), "execution_sha256": args.execution_sha256, "quality": quality, "paired_dense_pe_intervals": intervals, "loaded_held_sha256": pair.sha(args.output / "loaded-held.npy"), "reference_held_sha256": pair.sha(args.output / "reference-held.npy"), "all394_fresh_whole_independent_checkpoint_copy_batches_exact": True, "state_runtime_frozen_complement_preserved": True, "packed_per_query_exact": True, "cpu_cuda_rng_unchanged": True, "quality_read": "TRAIN-held fresh extra diagnostic only", "lost_original_trained_live_instance_used": False, "optimizer_updates": 0, "official_read": False, "claim_eligible": False, "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated()})
    print("Fresh checkpoint TRAIN-held diagnostic complete; closed original pilot remains a mechanics blocker")


if __name__ == "__main__":
    main()
