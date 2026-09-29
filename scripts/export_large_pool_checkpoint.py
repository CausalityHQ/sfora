#!/usr/bin/env python3
"""Qualify and export two independent copies of the retained pooling checkpoint."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")

import argparse
import copy
import json
import os
import resource
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pe_large_pool as native
from sfora.joint_relational_compaction import pack_int8_unit_embeddings

pair = native.pair
sha = pair.sha
TRAIN_ROOT = Path("/home/riomus/runs/sfora-large-native-pool-v1")
EXPORT_ROOT = Path("/home/riomus/runs/sfora-large-pool-checkpoint-export-source-v3")
SCORE_ROOT = Path("/home/riomus/runs/sfora-large-pool-checkpoint-score-source-v3")
CPU_PROOF = Path("/home/riomus/runs/sfora-large-pool-checkpoint-cpu-v3/proof.json")
WIRES = Path("/home/riomus/runs/sfora-large-pool-checkpoint-wires-v3")
CHECKPOINT = Path("/home/riomus/runs/sfora-large-native-pool-pilot-v1/pe.pt")
TRAIN_CODE = "1c867a64e4fb3eeac04daf0db4eb642ec0041d0ed43a91948f52c968547dd0b0"
CHECKPOINT_SHA = "a7e3d8d413aba67dd0d1472d5b70537a72de902ecbdac505833af9cbcaa1a796"
TRAIN_SHA = "ebf8521b70e46280824ada916fa83bb7488eb856afb865edc6630b636d24d409"
GPU_PROOF = Path("/home/riomus/runs/sfora-large-native-pool-gpu-v1/preflight.json")
GPU_SHA = "9f49b23c1128eca8b62af6330abaf34f189882d3051bc148beb7223c65f2aa85"
MECHANICS = Path("/home/riomus/runs/sfora-large-native-pool-mechanics-v1/receipt.json")
MECHANICS_SHA = "4158bb3be1bc5d88285cd567c4de8e37d7215aee2469b19f66e5b5c63d893be0"
INPUT_PROOF = Path("/home/riomus/runs/sfora-large-native-pool-prefetch-cpu-v1/preflight.json")
INPUT_SHA = "f85a525d7a366476d861af70cac1137b6fed8a5fa3b9696a69927849d344cca0"


def checked_map(root, code):
    for name, digest in code.items():
        path = (root / name).resolve()
        assert path.is_relative_to(root.resolve()) and sha(path) == digest, name


def execution(root, expected):
    manifest = root / "large-pool-export-execution.json"
    assert sha(manifest) == expected
    code = json.loads(manifest.read_text())
    previous_path = root / "large-pool-training-execution.json"
    assert sha(previous_path) == sha(TRAIN_ROOT / previous_path.name) == TRAIN_CODE
    previous = json.loads(previous_path.read_text())
    assert len(previous) == 71 and len(code) == 72
    assert set(code) == set(previous) | {"export_large_pool_checkpoint.py"}
    assert all(code[n] == h for n, h in previous.items())
    checked_map(TRAIN_ROOT, previous)
    checked_map(root, code)
    assert Path(__file__).resolve() == root / "export_large_pool_checkpoint.py"
    return code


def closure(root, code):
    checked_map(root, code)
    pair.executing_authority(root, code)
    assert all(n in code and code[n] == h for n, h in pair.smoke.authority().items())
    assert Path(native.__file__).resolve() == root / "pe_large_pool.py"


def authority(root, expected, loaded_code=None):
    code = execution(root, expected)
    active = code if loaded_code is None else loaded_code
    assert all(active[n] == h for n, h in code.items())
    closure(root, active)
    # Extend only the executing-source closure; retain every original authority hash.
    original = pair.executing_authority
    with patch.object(pair, "executing_authority", lambda r, old: original(r, {**old, **active})):
        control, frozen, prior = native.control(root)
    gpu = json.loads(GPU_PROOF.read_text())
    assert sha(GPU_PROOF) == GPU_SHA
    cpu_path = Path("/home/riomus/runs/sfora-large-native-pool-cpu-v3/preflight.json")
    assert sha(cpu_path) == gpu["native_cpu_authority_sha256"]
    cpu = json.loads(cpu_path.read_text())
    assert sha(INPUT_PROOF) == INPUT_SHA
    inputs = json.loads(INPUT_PROOF.read_text())
    assert inputs["original_native_cpu_sha256"] == sha(cpu_path)
    assert inputs["serial_worker_pixels_exact"] and inputs["caller_cpu_rng_unchanged"]
    for proof in (gpu, cpu, inputs):
        checked_map(root, proof["code"])
    assert gpu["updated_native_roundtrip_exact"] and gpu["temporary_pool_restored"]
    for path, digest in gpu["environment"]["native_files"].items():
        assert sha(Path(path)) == digest
    assert sha(CHECKPOINT) == CHECKPOINT_SHA
    assert sha(CHECKPOINT.parent / "training.json") == TRAIN_SHA
    assert not (CHECKPOINT.parent / "receipt.json").exists()
    training = json.loads((CHECKPOINT.parent / "training.json").read_text())
    assert training["execution_sha256"] == TRAIN_CODE and training["preflight_sha256"] == GPU_SHA
    assert not training["quality_read"] and not training["training_state_discarded"]
    assert training["frozen_sha256"] == cpu["frozen_complement_sha256"]
    assert training["attributed_attention_autograd_nodes"] == gpu["attention_autograd_nodes"]
    assert training["caller_cpu_rng_unchanged"] and training["one_pending_cpu_batch"]
    for key in ("step_seconds", "losses", "scales", "rgb_sha256", "worker_input_seconds", "preclip_gradient_norms"):
        assert len(training[key]) == 100
    assert all(np.isfinite(training[k]).all() for k in ("step_seconds", "losses", "preclip_gradient_norms"))
    assert training["scales"] == [128] * 100 and training["rgb_sha256"] == prior["rgb_sha256"]
    assert statistics.median(training["step_seconds"][2:]) == training["median_step_3_100_seconds"] <= .71769696
    assert training["training_wall_seconds_including_fill_drain"] >= sum(training["step_seconds"])
    assert training["peak_cuda_allocated_bytes"] < 10_000_000_000
    assert [d["step"] for d in training["diagnostics"]] == [1, 100]
    assert all(len(d["data_gradient_norms"]) == 14 and all(np.isfinite(v) and v > 0 for v in d["data_gradient_norms"].values()) for d in training["diagnostics"])
    assert sha(MECHANICS) == MECHANICS_SHA
    mechanics = json.loads(MECHANICS.read_text())
    assert mechanics["execution_sha256"] == TRAIN_CODE and mechanics["preflight_sha256"] == GPU_SHA
    assert mechanics["advance"] and mechanics["updates"] == 17
    assert mechanics["updated_gpu_strict_reload_exact"] and mechanics["fit_only_export_path_exact"]
    assert mechanics["median_step_3_17_seconds"] <= .71769696
    assert not (MECHANICS.parent / "pe.pt").exists() and not (MECHANICS.parent / "mechanics-fit.npy").exists()
    assert sha(MECHANICS.parent / "training.json") == mechanics["training_sha256"]
    mt = json.loads((MECHANICS.parent / "training.json").read_text())
    assert training["losses"][:17] == mt["losses"] and training["scales"][:17] == mt["scales"]
    log = TRAIN_ROOT / "large-pool-pilot-v1.log"
    terminal = log.read_text()
    assert all(s in terminal for s in ("Finished with result: timeout", "code=killed/status=TERM", "Memory swap peak: 0B"))
    assert terminal.count('"phase": "strict-loaded-verified-batch"') == 373
    assert len(frozen["held_manifest"]) == 12599 and len(frozen["fit_manifest"]) == 13283
    assert len(frozen["query"]) == 6354 and len(frozen["gallery"]) == 6245
    assert set(frozen["query"]).isdisjoint(frozen["gallery"])
    assert sorted(frozen["query"] + frozen["gallery"]) == list(range(12599))
    assert {r["product"] for r in frozen["held_manifest"]}.isdisjoint(r["product"] for r in frozen["fit_manifest"])
    proofs = {str(p): sha(p) for p in (GPU_PROOF, cpu_path, INPUT_PROOF, MECHANICS,
        MECHANICS.parent / "training.json", CHECKPOINT, CHECKPOINT.parent / "training.json",
        log, native.INIT / "receipt.json", native.INIT / "initializers.npz",
        root / "large-pool-training-execution.json", root / "large-pool-export-execution.json")}
    return control, frozen, prior, code, training, gpu, cpu, inputs, proofs


def flags():
    return {"deterministic": torch.are_deterministic_algorithms_enabled(),
        "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "autocast_cuda": torch.is_autocast_enabled("cuda"),
        "autocast_cpu": torch.is_autocast_enabled("cpu")}


def rng():
    return pair.smoke.digest({"cpu": torch.random.get_rng_state(),
        **{f"cuda.{i}": v for i, v in enumerate(torch.cuda.get_rng_state_all())}})


def runtime(model):
    value = native.runtime_identity(model)
    assert all(v == str(next(model.parameters()).device) for v in value["buffer_devices"].values())
    return json.loads(json.dumps({k: v for k, v in value.items() if k != "buffer_devices"}))


def checkpoint_pair(control, training, gpu, cpu, device):
    # One original loader authenticates the actual native class/config/nonpersistent buffers.
    with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device.type == "cuda" else []):
        model, processor = pair.smoke.load_arm(control, "large")
        inventory = native.freeze(model)
        assert json.loads(json.dumps(native.environment(model, processor))) == gpu["environment"]
        assert inventory == {k: tuple(v) for k, v in cpu["inventory"].items()}
        assert len(model.state_dict()) == 400 and len(inventory["frozen"]) == 389
        assert sum(p.numel() for p in model.parameters()) == 315956224
        assert pair.smoke.digest(native.whole_state(model)) == cpu["whole_original_source_sha256"]
        assert pair.smoke.digest(native.frozen_state(model)) == training["frozen_sha256"]
        assert runtime(model) == {k: v for k, v in cpu["runtime_identity"].items() if k != "buffer_devices"}
        saved = torch.load(CHECKPOINT, map_location="cpu", weights_only=True, mmap=True)
        assert set(saved) == {"vision", "head", "classifier", "bank"}
        assert set(saved["vision"]) == set(model.state_dict()) and len(saved["vision"]) == 400
        assert all(v.dtype == torch.float32 and torch.isfinite(v).all() for group in ("vision", "head") for v in saved[group].values())
        other = type(model)(copy.deepcopy(model.config)).float().eval()
        native.freeze(other)
        models = (model.eval(), other)
        heads = tuple(nn.Linear(1024, 128).float().eval() for _ in range(2))
        for m, h in zip(models, heads, strict=True):
            m.load_state_dict(saved["vision"], strict=True)
            h.load_state_dict(saved["head"], strict=True)
            assert pair.smoke.digest(native.frozen_state(m)) == training["frozen_sha256"]
            assert pair.smoke.digest(dict(m.head.named_parameters())) == training["final_group_sha256"]["native_pool"]
            assert pair.smoke.digest(h.state_dict()) == training["final_group_sha256"]["compact_head"]
            m.to(device).requires_grad_(False)
            h.to(device).requires_grad_(False)
        assert saved["classifier"].shape == (2004, 128) and saved["bank"].shape == (13283, 128)
        assert saved["classifier"].dtype == saved["bank"].dtype == torch.float32
        assert torch.isfinite(saved["classifier"]).all() and torch.isfinite(saved["bank"]).all()
        assert torch.allclose(saved["bank"].norm(dim=1), torch.ones(13283), atol=1e-5, rtol=0)
        assert pair.smoke.digest({"classifier": saved["classifier"]}) == training["final_group_sha256"]["classifier"]
        assert all(training["final_group_sha256"][k] != v for k, v in training["initial_group_sha256"].items())
        assert runtime(model) == runtime(other) == {k: v for k, v in training["runtime_identity"].items() if k != "buffer_devices"}
        assert pair.smoke.digest(native.whole_state(model)) == pair.smoke.digest(native.whole_state(other))
    return models, heads, processor


def state(models, heads):
    return {"whole_sha256": [pair.smoke.digest(native.whole_state(m)) for m in models],
        "frozen_sha256": [pair.smoke.digest(native.frozen_state(m)) for m in models],
        "head_sha256": [pair.smoke.digest(h.state_dict()) for h in heads],
        "runtime": [runtime(m) for m in models],
        "buffer_layouts": [{n: {"shape": list(v.shape), "dtype": str(v.dtype)}
            for n, v in m.named_buffers()} for m in (*models, *heads)]}


def unchanged(root, code, control, frozen, proofs, models, heads, processor, before, environment, numerical, random):
    assert state(models, heads) == before
    assert all(p.dtype == torch.float32 and p.grad is None for m in (*models, *heads) for p in m.parameters())
    assert all(not m.training for model in (*models, *heads) for m in model.modules())
    assert all(json.loads(json.dumps(native.environment(m, processor))) == environment for m in models)
    assert flags() == numerical and rng() == random
    usage = resource.getrusage(resource.RUSAGE_SELF)
    assert usage.ru_maxrss * 1024 < 8 * 1024**3 and usage.ru_nswap == 0
    assert all(sha(Path(p)) == digest for p, digest in proofs.items())
    assert sha(TRAIN_ROOT / "large-pool-training-execution.json") == TRAIN_CODE
    checked_map(TRAIN_ROOT, json.loads((TRAIN_ROOT / "large-pool-training-execution.json").read_text()))
    original = pair.executing_authority
    with patch.object(pair, "executing_authority", lambda r, old: original(r, {**old, **code})):
        pair.check_startup(control)
    assert all(sha(control.dataset_root / r["relative_path"]) == r["image_sha256"] for r in frozen["fit_manifest"] + frozen["held_manifest"])
    closure(root, code)


def packed_equal(a, b):
    pa, pb = (pack_int8_unit_embeddings(v.detach().cpu()) for v in (a, b))
    assert np.array_equal(pa.codes, pb.codes) and np.array_equal(pa.inverse_norms, pb.inverse_norms)


@torch.no_grad()
def encode(models, heads, pixels):
    a, b = native.verified_features(*models, pixels)
    assert torch.isfinite(a).all() and a.shape == (len(pixels), 1024)
    raw = tuple(pair.smoke.compact_head_features(v, h).float() for v, h in zip((a, b), heads, strict=True))
    assert torch.equal(*raw)
    vectors = tuple(F.normalize(v, dim=1) for v in raw)
    assert torch.equal(*vectors) and torch.isfinite(vectors[0]).all()
    assert vectors[0].shape == (len(pixels), 128)
    packed_equal(*vectors)
    return (a, b), vectors


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualify-cpu", action="store_true")
    parser.add_argument("--cpu-proof", type=Path)
    parser.add_argument("--cpu-sha256")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    assert root == EXPORT_ROOT
    assert not args.output.exists() and not args.output.is_symlink()
    assert args.output == (CPU_PROOF.parent if args.qualify_cpu else WIRES)
    assert (not args.cpu_proof and not args.cpu_sha256) if args.qualify_cpu else (args.cpu_proof == CPU_PROOF and args.cpu_sha256)
    assert torch.cuda.is_available() != args.qualify_cpu
    started = time.perf_counter()
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if not args.qualify_cpu:
        assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
        torch.cuda.reset_peak_memory_stats()
    control, frozen, _, code, training, gpu, cpu, inputs, proofs = authority(root, args.execution_sha256)
    print(json.dumps({"phase": "authority", "seconds": time.perf_counter() - started}), flush=True)
    if not args.qualify_cpu:
        assert sha(CPU_PROOF) == args.cpu_sha256
        proof = json.loads(CPU_PROOF.read_text())
    numerical, random = flags(), rng()
    tick = time.perf_counter()
    models, heads, processor = checkpoint_pair(control, training, gpu, cpu, torch.device("cpu" if args.qualify_cpu else "cuda"))
    before = state(models, heads)
    assert before["whole_sha256"][0] == before["whole_sha256"][1]
    assert before["head_sha256"][0] == before["head_sha256"][1]
    print(json.dumps({"phase": "independent_strict400_reload", "seconds": time.perf_counter() - tick}), flush=True)
    binding = {"execution_sha256": args.execution_sha256, "code": code, "proof_hashes": proofs,
        "checkpoint_sha256": CHECKPOINT_SHA, "training_sha256": TRAIN_SHA,
        "training_execution_sha256": TRAIN_CODE, "state": before, "environment": gpu["environment"],
        "numerical_flags": numerical, "fit_manifest": frozen["fit_manifest"],
        "original_pilot_status": "terminal timeout at373/394; training100 cost PASS",
        "historical_live_reload_batches": 373, "lost_original_trained_live_instance_used": False,
        "quality_read": False, "official_read": False, "claim_eligible": False,
        "public_latency_measured": False, "full_production_goal_met": False, "optimizer_updates": 0}
    if args.qualify_cpu:
        tick = time.perf_counter()
        images, rgb = pair.augmented_images(control.dataset_root, frozen["fit_manifest"], frozen["batches"][0], 1)
        first = pair.pixels(processor, images, "large")
        assert rgb == training["rgb_sha256"][0]
        assert pair.smoke.digest({"pixels": first}) == inputs["pixels_sha256"][0]
        del first, images
        images, _ = pair.augmented_images(control.dataset_root, frozen["fit_manifest"], (0, 1), None)
        pixels = pair.pixels(processor, images, "large")
        outputs, vectors = encode(models, heads, pixels)
        with torch.no_grad():
            direct = models[0](pixel_values=pixels).pooler_output.float()
        assert torch.equal(direct, outputs[0])
        print(json.dumps({"phase": "cpu_fit_pixels_direct_whole", "seconds": time.perf_counter() - tick}), flush=True)
        unchanged(root, code, control, frozen, proofs, models, heads, processor, before, gpu["environment"], numerical, random)
        original_sha = sha
        with patch.dict(execution.__globals__, {"sha": lambda p: "changed" if Path(p).resolve() == Path(__file__).resolve() else original_sha(p)}):
            try:
                execution(root, args.execution_sha256)
            except AssertionError:
                pass
            else:
                raise AssertionError("changed export driver accepted")
        assert time.perf_counter() - started < 120
        args.output.mkdir()
        pair.smoke.save(CPU_PROOF, {**binding, "pass": True, "changed_driver_rejected": True,
            "strict400_native_head_reload_and_direct_whole_calibration_exact": True,
            "all_named_buffers_including_nonpersistent_preserved": True,
            "source_state_rng_flags_data_preserved": True, "cpu_cuda_rng_unchanged": True,
            "first_two_fit_pixels_sha256": pair.smoke.digest({"pixels": pixels}),
            "first_augmented_pixels_sha256": inputs["pixels_sha256"][0],
            "held_images": 0, "read_only": True, "seconds": time.perf_counter() - started,
            "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        print("PASS CPU retained checkpoint authority; independent native copies, no quality", flush=True)
        return
    assert all(proof[k] == v for k, v in binding.items())
    assert proof["pass"] and proof["changed_driver_rejected"] and proof["read_only"]
    assert proof["strict400_native_head_reload_and_direct_whole_calibration_exact"]
    assert proof["source_state_rng_flags_data_preserved"] and proof["cpu_cuda_rng_unchanged"]
    proofs[str(CPU_PROOF)] = args.cpu_sha256
    tick = time.perf_counter()
    images, _ = pair.augmented_images(control.dataset_root, frozen["fit_manifest"], (0, 1), None)
    pixels = pair.pixels(processor, images, "large")
    assert pair.smoke.digest({"pixels": pixels}) == proof["first_two_fit_pixels_sha256"]
    x = pixels.cuda()
    (a, _), (va, _) = encode(models, heads, x)
    with torch.no_grad():
        full = models[0](pixel_values=x).pooler_output.float()
        vf = F.normalize(pair.smoke.compact_head_features(full, heads[0]).float(), dim=1)
    calibration = {"pooled": F.cosine_similarity(a, full).tolist(), "compact": F.cosine_similarity(va, vf).tolist()}
    assert all(min(v) >= .999 for v in calibration.values())
    print(json.dumps({"phase": "fp16_fp32_fit_calibration", "seconds": time.perf_counter() - tick}), flush=True)
    chunks, reference, prep_seconds, forward_seconds = [], [], [], []
    held_started = time.perf_counter()
    for start in range(0, 12599, 32):
        tick = time.perf_counter()
        rows = frozen["held_manifest"][start:start + 32]
        images, _ = pair.augmented_images(control.dataset_root, rows, tuple(range(len(rows))), None)
        pixels = pair.pixels(processor, images, "large")
        prep_seconds.append(time.perf_counter() - tick)
        tick = time.perf_counter()
        _, vectors = encode(models, heads, pixels.cuda())
        chunks.append(vectors[0].cpu().numpy())
        reference.append(vectors[1].cpu().numpy())
        torch.cuda.synchronize()
        forward_seconds.append(time.perf_counter() - tick)
        assert torch.cuda.max_memory_allocated() < 10_000_000_000
        if len(chunks) % 32 == 0 or start + 32 >= 12599:
            print(json.dumps({"phase": "independent_whole_held", "images": min(start + 32, 12599), "seconds": time.perf_counter() - held_started}), flush=True)
    held_seconds = time.perf_counter() - held_started
    tick = time.perf_counter()
    unchanged(root, code, control, frozen, proofs, models, heads, processor, before, gpu["environment"], numerical, random)
    print(json.dumps({"phase": "exit_authority", "seconds": time.perf_counter() - tick}), flush=True)
    values, other = np.concatenate(chunks), np.concatenate(reference)
    assert len(chunks) == 394 and values.dtype == other.dtype == np.float32
    assert values.shape == other.shape == (12599, 128) and np.array_equal(values, other) and np.isfinite(values).all()
    assert np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0)
    packed_equal(torch.from_numpy(values), torch.from_numpy(other))
    assert time.perf_counter() - started < 300
    args.output.mkdir()
    for name, array in (("held.npy", values), ("reference-held.npy", other)):
        with (args.output / name).open("xb") as stream:
            np.save(stream, array, allow_pickle=False)
    pair.smoke.save(args.output / "receipt.json", {**binding, "pass": True,
        "cpu_authority_sha256": args.cpu_sha256, "source_code": code,
        "held_sha256": sha(args.output / "held.npy"), "reference_held_sha256": sha(args.output / "reference-held.npy"),
        "held_manifest": frozen["held_manifest"], "query": frozen["query"], "gallery": frozen["gallery"],
        "held_images": 12599, "batches": 394, "batch_size": 32,
        "two_independent_checkpoint_copies_whole_head_normalized_packed_exact": True,
        "source_state_rng_flags_data_preserved": True,
        "all_named_buffers_including_nonpersistent_preserved": True,
        "fp32_parameters_fp16_fresh_separate_scopes_heads_outside": True,
        "fp16_fp32_fit_calibration": calibration, "held_forward_seconds": held_seconds,
        "cpu_prepare_seconds": prep_seconds, "paired_forward_seconds": forward_seconds,
        "export_wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "process_swap_count": resource.getrusage(resource.RUSAGE_SELF).ru_nswap,
        "training_cost": {k: training[k] for k in ("median_step_3_100_seconds", "training_wall_seconds_including_fill_drain", "peak_cuda_allocated_bytes")}})
    print("PASS complete independent checkpoint-copy export; no quality", flush=True)


if __name__ == "__main__":
    main()
