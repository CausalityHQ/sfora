#!/usr/bin/env python3
"""Native S16 full-encoder learning with bounded actual CPU input."""

import argparse
from contextlib import closing
import os
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pe_s16_training as l14
import pe_s16_training as pool
from qualify_pe_s16_gpu import sdpa_nodes
from pe_l14_prefetch import prefetch
import train_inshop_pe_pair as pair
from pe_core_training import named_training_parameters


def cuda_budget(phase):
    counters = {
        "phase": phase,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "current_cuda_allocated_bytes": torch.cuda.memory_allocated(),
    }
    print(json.dumps(counters), flush=True)
    assert counters["peak_cuda_allocated_bytes"] < 10_000_000_000, counters


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preflight-sha256")
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--mechanics-receipt-sha256")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check-startup-only", action="store_true")
    parser.add_argument("--updates", type=int, choices=(17, 100), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    init_dir = Path("/home/riomus/runs/sfora-pe-s16-init-v1")
    authority_dir = Path("/home/riomus/runs/sfora-pe-s16-gpu-v1")
    assert (
        args.preflight_sha256
        and pair.sha(authority_dir / "preflight.json") == args.preflight_sha256
    )
    initial = json.loads((authority_dir / "preflight.json").read_text())
    native_cpu = Path("/home/riomus/runs/sfora-pe-s16-cpu-v1/preflight.json")
    assert pair.sha(native_cpu) == initial["native_cpu_authority_sha256"]
    native_initial = json.loads(native_cpu.read_text())
    input_cpu = Path("/home/riomus/runs/sfora-pe-l14-prefetch-cpu-v1/preflight.json")
    assert (
        pair.sha(input_cpu)
        == "234d7ece495c74a294ff18e3cc7f12cf301244d6cc322783d8e26a9db0e2c273"
    )
    input_initial = json.loads(input_cpu.read_text())
    assert (
        input_initial["serial_worker_pixels_exact"]
        and input_initial["caller_cpu_rng_unchanged"]
    )
    assert (
        initial["updated_native_roundtrip_exact"]
        and initial["temporary_stem_and_pool_restored"]
    )
    execution = root / "s16-training-execution.json"
    assert pair.sha(execution) == args.execution_sha256
    code = json.loads(execution.read_text())
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert (
        pair.sha(init_dir / "initializers.npz") == native_initial["initializers_sha256"]
    )
    assert Path(l14.__file__).resolve() == root / "pe_s16_training.py"
    source_args, frozen, prior = l14.control(root)
    info = native_initial
    if args.check_startup_only:
        assert not torch.cuda.is_available()
        print(
            "PASS CPU startup source/control/initializers/qualification/execution authority",
            flush=True,
        )
        return
    if args.updates == 100:
        mechanics_path = Path(
            "/home/riomus/runs/sfora-pe-s16-mechanics-v1/receipt.json"
        )
        assert not (mechanics_path.parent / "pe.pt").exists()
        assert (
            args.mechanics_receipt_sha256
            and pair.sha(mechanics_path) == args.mechanics_receipt_sha256
        )
        mechanics = json.loads(mechanics_path.read_text())
        assert (
            mechanics["advance"]
            and mechanics["updates"] == 17
            and mechanics["updated_gpu_strict_reload_exact"]
        )
        assert mechanics["median_step_3_17_seconds"] <= 0.71769696
        assert mechanics["preflight_sha256"] == args.preflight_sha256
    args.output.mkdir(exist_ok=False)
    pair.smoke.save(
        args.output / "attempt.json",
        {
            "preflight_sha256": args.preflight_sha256,
            "updates": args.updates,
            "execution_sha256": args.execution_sha256,
            "mechanics_receipt_sha256": pair.sha(mechanics_path)
            if args.updates == 100
            else None,
        },
    )
    timing_field = f"median_step_3_{args.updates}_seconds"
    torch.set_num_threads(8)
    torch.manual_seed(pair.SEED)
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    assert not torch.is_deterministic_algorithms_warn_only_enabled()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    vision, processor = l14.load()
    pair.executing_authority(root, frozen["code"])
    inventory = pool.freeze(vision)
    assert inventory == {k: tuple(v) for k, v in info["inventory"].items()}
    vision.cuda().train()
    with np.load(init_dir / "initializers.npz", allow_pickle=False) as init:
        head = nn.Linear(512, 128)
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
    native_params = [
        p for _, p in named_training_parameters(vision, "pe") if p.requires_grad
    ]
    params = native_params + list(head.parameters()) + [classifier]
    assert len({id(p) for p in params}) == len(params)
    optimizer = torch.optim.AdamW(
        [
            {"params": native_params, "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in params
    }
    assert len(params) == 166 and not optimizer.state
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
            np.load(l14.CACHE / "s16.fit.npy", mmap_mode="r")[:4].copy()
        ).cuda()
        calibration = {
            "fp16_fp32": F.cosine_similarity(fp16, fp32).tolist(),
            "cached_fresh_fp16": F.cosine_similarity(cached, fp16).tolist(),
        }
        pair.smoke.save(args.output / "calibration.json", calibration)
        print(json.dumps({"calibration": calibration}), flush=True)
        assert all(min(v) >= 0.999 for v in calibration.values())
    del images, x, fp32, fp16, cached
    cuda_budget("resource-check")
    frozen_sha = pair.smoke.digest(pool.frozen_state(vision))
    assert frozen_sha == info["frozen_complement_sha256"]
    initial_groups = {
        n: pair.smoke.digest(v)
        for n, v in {
            "full_native_encoder": dict(vision.named_parameters()),
            "compact_head": dict(head.named_parameters()),
            "classifier": {"classifier": classifier},
        }.items()
    }
    seconds, losses, rgb_hashes, diagnostics, scales, norms = [], [], [], [], [], []

    def prepare_input(index):
        tick = time.perf_counter()
        batch = frozen["batches"][index]
        images, rgb_sha = pair.augmented_images(
            source_args.dataset_root, frozen["fit_manifest"], batch, index + 1
        )
        assert rgb_sha == prior["rgb_sha256"][index]
        pixels = pair.pixels(processor, images, "pe")
        assert pixels.device.type == "cpu" and not pixels.requires_grad
        if index < 17:
            assert (
                pair.smoke.digest({"pixels": pixels})
                == input_initial["pixels_sha256"][index]
            )
        return pixels, rgb_sha, time.perf_counter() - tick

    training_rng = torch.random.get_rng_state().clone()
    worker_input_seconds = []
    runtime = pool.runtime_identity(vision)
    attention_backends = {}
    training_started = time.perf_counter()
    with closing(prefetch(prepare_input, args.updates)) as inputs:
        for step, batch in enumerate(frozen["batches"][: args.updates], 1):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            x, rgb_sha, worker_seconds = next(inputs)
            assert rgb_sha == prior["rgb_sha256"][step - 1]
            assert x.device.type == "cpu" and not x.requires_grad
            assert x.is_contiguous()
            if step == 1:
                assert (
                    pair.smoke.digest({"pixels": x})
                    == input_initial["pixels_sha256"][0]
                )
            x = x.cuda()
            index = torch.tensor(batch, device="cuda")
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                source = vision(x)
            if step == 1:
                nodes = sdpa_nodes(source.grad_fn)
                attention_backends = sorted(type(n).__name__ for n in nodes)
                nodes.clear()
                assert attention_backends == initial["sdpa_autograd_nodes"]
            raw = l14.compact_features(source, head)
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
            for n, p in (
                list(named_training_parameters(vision, "pe"))
                + [("head." + n, p) for n, p in head.named_parameters()]
                + [("classifier", classifier)]
            ):
                assert (p.grad is None) == (not p.requires_grad), n
                if p.grad is not None:
                    assert torch.isfinite(p.grad).all(), n
                    if step in (1, args.updates) and ".parametrizations." not in n:
                        assert p.grad.norm() > 0, n
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(params, 1, error_if_nonfinite=True)
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            assert scaler.get_scale() >= scale_before, "optimizer update skipped"
            if step in (1, args.updates):
                diagnostics.append(
                    {
                        "step": step,
                        "data_gradient_norms": pair.smoke.gradients(
                            vision, head, classifier, "pe"
                        ),
                    }
                )
            rows, positions = pair.smoke.member_bank_refresh_rows(tuple(batch))
            bank[torch.tensor(rows, device="cuda")] = (
                pair.smoke.member_bank_refresh_values(
                    source,
                    raw,
                    torch.tensor(positions, device="cuda"),
                    live_head=False,
                )
            )
            assert all(torch.isfinite(p).all() for p in params)
            assert all(
                torch.isfinite(v).all()
                for state in optimizer.state.values()
                for v in state.values()
                if isinstance(v, torch.Tensor)
            )
            assert torch.isfinite(bank).all()
            cuda_budget("resource-check")
            torch.cuda.synchronize()
            seconds.append(time.perf_counter() - tick)
            worker_input_seconds.append(worker_seconds)
            losses.append(float(loss.detach()))
            rgb_hashes.append(rgb_sha)
            scales.append(scaler.get_scale())
            norms.append(float(norm))
            print(
                json.dumps({"step": step, "seconds": seconds[-1], "loss": losses[-1]}),
                flush=True,
            )
    assert pool.runtime_identity(vision) == runtime
    training_wall_seconds = time.perf_counter() - training_started
    assert torch.equal(training_rng, torch.random.get_rng_state())
    assert frozen_sha == pair.smoke.digest(pool.frozen_state(vision))
    assert all(
        pair.smoke.digest(
            {
                "full_native_encoder": dict(vision.named_parameters()),
                "compact_head": dict(head.named_parameters()),
                "classifier": {"classifier": classifier},
            }[n]
        )
        != h
        for n, h in initial_groups.items()
    )
    median = float(np.median(seconds[2:]))
    # Persist cost before the frozen stop rule, including negative outcomes.
    pair.smoke.save(
        args.output / "training.json",
        {
            timing_field: median,
            "step_seconds": seconds,
            "attributed_sdpa_autograd_nodes": attention_backends,
            "runtime_identity": runtime,
            "execution_sha256": args.execution_sha256,
            "worker_input_seconds": worker_input_seconds,
            "training_wall_seconds_including_fill_drain": training_wall_seconds,
            "one_pending_cpu_batch": True,
            "caller_cpu_rng_unchanged": True,
            "losses": losses,
            "rgb_sha256": rgb_hashes,
            "scales": scales,
            "preclip_gradient_norms": norms,
            "diagnostics": diagnostics,
            "calibration": calibration,
            "frozen_sha256": frozen_sha,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "initial_group_sha256": initial_groups,
            "final_group_sha256": {
                n: pair.smoke.digest(v)
                for n, v in {
                    "full_native_encoder": dict(vision.named_parameters()),
                    "compact_head": dict(head.named_parameters()),
                    "classifier": {"classifier": classifier},
                }.items()
            },
            "preflight_sha256": args.preflight_sha256,
            "training_state_discarded": args.updates == 17,
            "quality_read": False,
        },
    )
    if median > 0.71769696:
        pair.smoke.save(
            args.output / "receipt.json",
            {
                "advance": False,
                "reason": "fixed training cost gate failed",
                "preflight_sha256": args.preflight_sha256,
                "discard_training_state": args.updates == 17,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "quality_read": False,
                "updates": args.updates,
                timing_field: median,
            },
        )
        print("KILL cost; stopped before held read", flush=True)
        return
    del optimizer, params, source, raw, ce, rank, loss
    vision.zero_grad(set_to_none=True)
    vision.eval()
    head.eval()
    runtime = pool.runtime_identity(vision)

    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        updated = vision(x).float()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        merged = vision(x).float()
    assert torch.equal(updated, merged), "native evaluation changed"
    cuda_budget("resource-check")
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
        VisionTransformer.from_config("PE-Core-S16-384", pretrained=False)
        .float()
        .eval()
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    fresh.load_state_dict(saved["vision"], strict=True)
    fresh.rope.rope.load_state_dict(saved["rope"], strict=True)
    fresh.rope.update_grid(torch.device("cpu"), 14, 14)
    assert torch.equal(fresh.rope.freq, saved["rope_freq"])
    fresh.cuda()
    loaded_head = nn.Linear(512, 128)
    loaded_head.load_state_dict(saved["head"], strict=True)
    loaded_head.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
        reloaded = fresh(x).float()
    assert pair.smoke.digest(l14.whole_state(fresh)) == pair.smoke.digest(
        l14.whole_state(vision)
    )
    assert pool.runtime_identity(fresh) == pool.runtime_identity(vision) == runtime
    assert torch.equal(updated, reloaded), "updated strict-loaded GPU output differs"
    with torch.no_grad():
        updated_readout = F.normalize(
            l14.compact_features(updated, head).float(), dim=1
        )
        loaded_readout = F.normalize(
            l14.compact_features(reloaded, loaded_head).float(), dim=1
        )
    assert torch.equal(updated_readout, loaded_readout), (
        "updated strict-loaded compact GPU output differs"
    )
    cuda_budget("resource-check")
    pair.executing_authority(root, frozen["code"])
    assert pair.smoke.digest(pool.frozen_state(fresh)) == frozen_sha
    assert all(pair.sha(root / n) == h for n, h in code.items())
    if args.updates == 17:
        checkpoint.unlink()
        pair.smoke.save(
            args.output / "receipt.json",
            {
                "advance": True,
                "preflight_sha256": args.preflight_sha256,
                "updates": 17,
                "quality_read": False,
                "discard_training_state": True,
                "updated_gpu_strict_reload_exact": True,
                timing_field: median,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                "seconds": time.perf_counter() - started,
            },
        )
        print(
            "PASS native S16 full-encoder adaptation mechanics; trained state discarded",
            flush=True,
        )
        return

    live_chunks = []

    @torch.inference_mode()
    def encode_verified(
        batch,
        model=fresh,
        live_model=vision,
        live_head=head,
        reloaded_head=loaded_head,
        recorded_live=live_chunks,
    ):
        images, _ = pair.augmented_images(
            source_args.dataset_root, batch, tuple(range(len(batch))), None
        )
        loaded_source, live_source = pool.verified_features(
            model, live_model, pair.pixels(processor, images, "pe").cuda()
        )
        live_vectors = F.normalize(
            l14.compact_features(live_source, live_head).float(), dim=1
        )
        loaded_vectors = F.normalize(
            l14.compact_features(loaded_source, reloaded_head).float(),
            dim=1,
        )
        assert torch.equal(live_vectors, loaded_vectors), (
            "actual whole live/loaded native encoder/head vectors differ"
        )
        assert (
            pool.runtime_identity(model) == pool.runtime_identity(live_model) == runtime
        )
        recorded_live.append(live_vectors.cpu().numpy())
        cuda_budget("strict-loaded-full-held")
        return loaded_vectors.cpu().numpy()

    assert pair.smoke.digest(pool.frozen_state(fresh)) == frozen_sha
    pair.export_features(
        frozen["held_manifest"],
        encode_verified,
        args.output / "pe.held.npy",
        width=128,
        batch_size=32,
    )
    loaded = np.load(args.output / "pe.held.npy", allow_pickle=False)
    live = np.concatenate(live_chunks)
    assert live.shape == loaded.shape and np.array_equal(live, loaded)
    np.save(args.output / "pe.live-held.npy", live, allow_pickle=False)
    del live_chunks
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
            "query_lower95": pair.bootstrap_lower(delta, np.arange(len(delta))),
            "query_upper95": -pair.bootstrap_lower(-delta, np.arange(len(delta))),
        }
    advance = bool(
        quality["recall_at_1"] >= 0.951720176
        and quality["map_at_r"] >= 0.776237120
        and all(v["product_lower95"] > 0 for v in intervals.values())
    )
    cuda_budget("resource-check")
    pair.executing_authority(root, frozen["code"])
    assert all(pair.sha(root / n) == h for n, h in code.items())
    assert pair.smoke.digest(l14.whole_state(fresh)) == pair.smoke.digest(
        l14.whole_state(vision)
    )
    assert pair.smoke.digest(pool.frozen_state(fresh)) == frozen_sha
    assert pool.runtime_identity(fresh) == pool.runtime_identity(vision) == runtime
    del fresh, saved, loaded_head

    pair.smoke.save(
        args.output / "receipt.json",
        {
            "advance": advance,
            "updates": args.updates,
            "quality_read": "In-Shop TRAIN-held only",
            "checkpoint_sha256": checkpoint_sha,
            "held_sha256": pair.sha(args.output / "pe.held.npy"),
            "quality": quality,
            "paired_dense_pe_intervals": intervals,
            "native_gpu_output_exact": True,
            "updated_gpu_strict_reload_exact": True,
            "updated_loaded_minimum_cosine": minimum_cosine,
            "packed_per_query_parity": True,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            timing_field: median,
            "seconds": time.perf_counter() - started,
            "frozen_complement_unchanged": True,
            "full_live_loaded_native_state_exact": True,
            "full_held_independent_whole_encoder_exact": True,
            "live_held_sha256": pair.sha(args.output / "pe.live-held.npy"),
            "execution_sha256": args.execution_sha256,
            "preflight_sha256": args.preflight_sha256,
            "full_held_head_reload_exact": True,
            "official_read": False,
            "claim_eligible": False,
        },
    )
    del vision, head, classifier, bank, x, p
    gc.collect()
    torch.cuda.empty_cache()
    print(
        "GO TRAIN feasibility" if advance else "KILL fixed TRAIN quality gate",
        flush=True,
    )


if __name__ == "__main__":
    main()
