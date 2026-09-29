#!/usr/bin/env python3
"""One discarded 17-update F5 Large lower-adapter mechanics run."""

import argparse
import hashlib
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import pe_large_optimization as old
import pe_native_valid_anchor as lane
from large_lower_adapter import install, merge
from sfora.joint_relational_compaction import pack_int8_unit_embeddings


CPU = Path("/home/riomus/runs/sfora-large-lower-adapter-cpu-v2")
CPU_SHA = "b0b2140c2ec56e9b94227efbe60b473690428ce15cddfd25b394efe3bff5ee4e"
CPU_CODE_SHA = "7dfc0bb4dbe391bcacb1ae861fa21406a5ac3b85155e567bbb5150ae38679a01"
OLD_HALF = Path("/home/riomus/runs/sfora-large-coverage-half-100-v1")
VALID_CPU = Path("/home/riomus/runs/sfora-native-valid-anchor-cpu-v2/valid-anchor-cpu-proof.json")
VALID_CPU_SHA = "8d78e6104f453dc62563e446cd0e668c9a158890b1267c02b0097d98c2c09f5d"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def startup(root, execution_sha):
    assert sha(root / "large-lower-gpu-execution.json") == execution_sha
    code = json.loads((root / "large-lower-gpu-execution.json").read_text())
    previous = json.loads((root / "large-optimization-execution.json").read_text())
    assert set(code) == set(previous) | {"large_lower_adapter.py", "check_large_lower_adapter.py", "train_large_lower_mechanics.py", "pe_native_valid_anchor.py"}
    assert all(code[k] == v for k, v in previous.items())
    assert all(sha(root / k) == v for k, v in code.items())
    assert code["pe_native_valid_anchor.py"] == "3733cbde3951217a6ed68d24c0d42b7b26c8bd80a966103fa4c32c186789d1ca"
    assert sha(VALID_CPU) == VALID_CPU_SHA
    valid_cpu = json.loads(VALID_CPU.read_text())
    assert valid_cpu["pass"] and valid_cpu["optimizer_updates"] == 0 and not valid_cpu["quality_read"]
    assert sha(CPU / "lower-cpu-proof-v2.json") == CPU_SHA
    assert sha(CPU / "large-lower-execution.json") == CPU_CODE_SHA
    cpu = json.loads((CPU / "lower-cpu-proof-v2.json").read_text())
    assert cpu["pass"] and cpu["site_count"] == 24 and cpu["optimizer_member_count"] == 256
    assert not cpu["quality_read"] and not cpu["gpu_training_qualified"]
    cpu_code = json.loads((CPU / "large-lower-execution.json").read_text())
    source = json.loads((root / "large-coverage-execution.json").read_text())
    assert all(cpu_code[k] == v for k, v in source.items())
    assert cpu_code["large_lower_adapter.py"] == code["large_lower_adapter.py"]
    old_helpers = old.previous.selected.helpers
    with patch.object(old.previous.selected, "helpers", lambda r, _: old_helpers(r, code)):
        control, native, prior, proof, _ = old.startup(root, sha(root / "large-optimization-execution.json"))
    return control, native, prior, proof, code


def fresh(control, native, proof):
    model, head, processor, inventory = old.coverage.load_native(control, native)
    sites = install(model, rank=8, seed=179032)
    model.cuda().train()
    head.cuda().train()
    values = old.initializers(proof, "half")
    classifier = nn.Parameter(values["classifier"].cuda())
    bank, target = values["bank"].cuda(), values["target"].cuda()
    factors = [p for module, _ in sites for p in module.parametrizations.weight[0].parameters()]
    factor_ids = {id(p) for p in factors}
    dense = [p for p in model.parameters() if p.requires_grad and id(p) not in factor_ids]
    params = old.coverage.parameters(model, head, classifier)
    assert len(factors) == 48 and len(params) == 256
    optimizer = torch.optim.AdamW([{"params": dense, "lr": 1e-5},
                                   {"params": factors, "lr": 1e-4},
                                   {"params": list(head.parameters()) + [classifier], "lr": 1e-4}], weight_decay=.05)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {id(p) for _, p in params}
    scaler = old.pair.smoke.training_precision("fp16", device="cuda")[1]
    return {"model": model, "head": head, "processor": processor, "inventory": inventory,
            "classifier": classifier, "bank": bank, "target": target,
            "positive": old.pair.smoke.member_bank_positive_ordinals(target.cpu().numpy(), allow_singletons=True).cuda(),
            "params": params, "optimizer": optimizer, "scaler": scaler, "counter": 0, "sites": sites}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--startup-only", action="store_true")
    args = parser.parse_args()
    assert not args.output.exists()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(old.pair.SEED)
    control, native, prior, proof, code = startup(root, args.execution_sha256)
    if args.startup_only:
        assert not torch.cuda.is_available()
        assert np.array_equal(old.schedule(old.initializers(proof, "half")["target"].numpy())[:100], old.coverage.schedule(old.initializers(proof, "half")["target"].numpy()))
        args.output.write_text(json.dumps({"pass": True, "execution_sha256": args.execution_sha256, "quality_read": False, "gpu_training_qualified": False}) + "\n")
        return
    assert torch.cuda.is_available() and os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    assert old.coverage.teacher.qualified.numerical_flags() == prior["numerical_flags"]
    state = fresh(control, native, proof)
    arm = proof["arms"]["half"]
    batches = old.schedule(state["target"].cpu().numpy())[:17]
    assert len(batches) == 17
    old_receipt = json.loads((OLD_HALF / "receipt.json").read_text())
    old_weights = {n: p.detach().cpu().clone() for n, p in state["model"].named_parameters() if not p.requires_grad}
    rows = []
    durations = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step, batch in enumerate(batches, 1):
        tick = time.perf_counter()
        ids = tuple(map(int, batch))
        images, rgb = old.pair.augmented_images(control.dataset_root, arm["rows"], ids, step)
        pixels = old.pair.pixels(state["processor"], images, "large")
        pixel_sha = old.pair.smoke.digest({"pixels": pixels})
        rank_active = arm["rank_active"][step - 1]
        with patch.object(old.coverage, "terms", lane.terms):
            row = old.step(state, pixels, ids, rank_active)
        valid_count = int((state["positive"][torch.tensor(ids, device="cuda")] >= 0).any(dim=1).sum())
        row.update(rgb_sha256=rgb, pixels_sha256=pixel_sha, rank_active_before=rank_active,
                   valid_anchors=valid_count, rank_recovered=not rank_active)
        assert rank_active or valid_count == 0 or row["rank"] > 0
        assert row["rgb_sha256"] == old_receipt["steps"][step - 1]["rgb_sha256"]
        assert row["pixels_sha256"] == old_receipt["steps"][step - 1]["pixels_sha256"]
        lower = [float(module.parametrizations.weight[0].B.grad.norm()) for module, _ in state["sites"]]
        assert all(v > 0 and np.isfinite(v) for v in lower)
        row["lower_factor_b_min_gradient"] = min(lower)
        rows.append(row)
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - tick)
        row["seconds"] = durations[-1]
        if step == 17:
            calibration = pixels[:2].cuda()
    training_wall = time.perf_counter() - started
    recovered = sum(row["rank_recovered"] for row in rows)
    assert recovered > 0, "mechanics did not exercise corrected inactive-batch routing"
    args.output.with_suffix(".training.json").write_text(json.dumps({"steps": rows,
        "training_wall_seconds": training_wall, "recovered_rank_updates": recovered,
        "quality_read": False}, sort_keys=True, indent=2) + "\n")
    assert all(torch.equal(p.detach().cpu(), old_weights[n]) for n, p in state["model"].named_parameters() if not p.requires_grad)
    model = state["model"].eval()
    head = state["head"].eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        before = model(pixel_values=calibration).pooler_output.float()
    merge(state["sites"])
    assert len(model.state_dict()) == 400
    del state
    torch.cuda.empty_cache()
    strict, reload_head, _, _ = old.coverage.load_native(control, native)
    strict.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()}, strict=True)
    reload_head.load_state_dict({k: v.detach().cpu() for k, v in head.state_dict().items()}, strict=True)
    strict.cuda().eval()
    reload_head.cuda().eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        after = strict(pixel_values=calibration).pooler_output.float()
    assert torch.equal(before, after)
    a = F.normalize(old.pair.smoke.compact_head_features(before, head), dim=1)
    b = F.normalize(old.pair.smoke.compact_head_features(after, reload_head), dim=1)
    assert torch.equal(a, b)
    packed_a, packed_b = pack_int8_unit_embeddings(a.cpu()), pack_int8_unit_embeddings(b.cpu())
    assert np.array_equal(packed_a.codes, packed_b.codes)
    assert np.array_equal(packed_a.inverse_norms, packed_b.inverse_norms)
    median = statistics.median(durations[2:])
    assert median * 100 < 269, "100-update training admission exceeds fixed cap"
    result = {"pass": True, "execution_sha256": args.execution_sha256, "cpu_proof_sha256": CPU_SHA,
              "source_code": code, "steps": rows, "training_wall_seconds": training_wall,
              "images_per_second": 17 * 64 / training_wall, "median_step_3_to_17_seconds": median,
              "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(), "quality_read": False,
              "checkpoint_saved": False, "merged_native_strict400_exact": True,
              "recovered_rank_updates": recovered}
    assert result["cuda_peak_allocated_bytes"] < 10_000_000_000
    args.output.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print("PASS 17 native lower-adapter updates, exact inputs and merged strict400 parity; mechanics discarded")


if __name__ == "__main__":
    main()
