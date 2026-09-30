#!/usr/bin/env python3
"""Actual source/layout/gradient qualification; DGX CPU only, no quality read."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import resource
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace
from unittest.mock import patch

STARTED = time.perf_counter()
ORIGINAL = "c9cec7b71d0c2ba90de2eb0d5c9810f111d4e7feb88f10e63055fbabf8a6d8b3"
ADDED = {"token_residual_readout.py", "test_token_residual_readout.py",
         "qualify_token_residual_cpu.py", "test_token_residual_cpu.py"}
SIGLIP_SHA = "274eafdb9bff99607f15eb4c4fb6fc47256b07358807653baa48539d13376b31"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closure(root, expected):
    manifest = root / "token-residual-cpu-execution.json"
    assert sha(manifest) == expected
    code = json.loads(manifest.read_text())
    original = root / "late-dense-execution.json"
    assert sha(original) == ORIGINAL
    old = json.loads(original.read_text())
    assert len(old) == 114 and len(code) == 118
    assert set(code) - set(old) == ADDED
    assert all(code[n] == h for n, h in old.items())
    assert all(sha(root / n) == h for n, h in code.items()), "execution source changed"
    return code


def verify(state, base, arm):
    """Verify unchanged common state without relabelling its 208-member proof."""
    import train_late_dense_adaptation as native
    import token_residual_readout as residual
    residual.validate_weight(state["residual"], arm)
    assert state["residual"].requires_grad == (arm == "candidate")
    expected = 208 + (arm == "candidate")
    assert len(state["params"]) == expected
    ids = [id(p) for g in state["optimizer"].param_groups for p in g["params"]]
    assert ids == [id(p) for _, p in state["params"]] and len(ids) == len(set(ids))
    common = {**state, "params": state["params"][:208],
              "optimizer": SimpleNamespace(param_groups=state["optimizer"].param_groups[:3])}
    native.verify(common, base)
    if arm == "candidate":
        assert len(state["optimizer"].param_groups) == 4
        assert state["params"][-1][0] == "residual" and state["params"][-1][1] is state["residual"]
        group = state["optimizer"].param_groups[3]
        assert group["lr"] == 1e-4 and group["weight_decay"] == .05
    else:
        assert len(state["optimizer"].param_groups) == 3
        assert state["residual"].count_nonzero() == 0


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--execution-sha256", required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    root = Path(__file__).resolve().parent
    code = closure(root, args.execution_sha256)
    cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip().lstrip("/")
    assert int((cgroup / "memory.max").read_text()) == 8 * 1024**3
    assert int((cgroup / "memory.swap.max").read_text()) == 0
    assert str(os.getpid()) in (cgroup / "cgroup.procs").read_text().split()
    invocation = os.environ["INVOCATION_ID"]
    import torch
    from torch.nn import functional as F
    import inspect
    import qualify_late_dense_adaptation_cpu as qualified
    import train_late_dense_adaptation as native
    import token_residual_readout as residual
    assert not torch.cuda.is_available()
    torch.set_num_threads(8); torch.manual_seed(179032)
    selected = qualified.archived.training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, "helpers", lambda r, _: helpers(r, code)):
        control, source, _, proof, _, _ = qualified.startup(root, ORIGINAL)
    native.pair.executing_authority(root, code)
    initial_values = []
    for arm in ("control", "candidate"):
        state, initial = qualified.fresh(control, source, proof, 12, "cpu")
        schedule = native.old.coverage.schedule(state["target"].tolist(), seed=179032)
        schedule_sha = native.old.fingerprint({"batches": torch.tensor(schedule)})
        base = native.identity(state, SimpleNamespace(seed=179032, boundary=12,
            execution_sha256=args.execution_sha256, cpu_sha256=None,
            cpu_log_sha256=None, cpu_execution_sha256=None), initial, schedule_sha)
        rng = torch.random.get_rng_state().clone()
        residual.attach(state, arm)
        assert torch.equal(rng, torch.random.get_rng_state())
        assert state["residual"].count_nonzero() == 0
        verify(state, base, arm)
        initial_values.append(initial)
        if arm == "control":
            del state; gc.collect()
    assert initial_values[0] == initial_values[1]
    assert state["counter"] == 0 and not state["optimizer"].state
    model = state["model"]
    module = Path(inspect.getfile(type(model)))
    assert sha(module) == SIGLIP_SHA
    assert model.config.hidden_size == 1024
    assert model.embeddings.num_patches == 256
    assert model.config.image_size // model.config.patch_size == 16
    model.eval(); state["head"].eval()
    with TemporaryDirectory(dir=root) as temporary:
        path = Path(temporary) / "native.pt"
        torch.save({"vision": model.state_dict(), "head": state["head"].state_dict(),
            "buffers": dict(model.named_buffers()), "residual": state["residual"].detach()}, path)
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        with torch.random.fork_rng(devices=[]):
            reference = type(model)(copy.deepcopy(model.config)).float().eval()
            head = torch.nn.Linear(1024, 128).eval()
            weight = residual.new_weight("cpu", False)
        reference.load_state_dict(saved["vision"], strict=True)
        head.load_state_dict(saved["head"], strict=True)
        with torch.no_grad():
            weight.copy_(saved["residual"])
            for n, value in reference.named_buffers(): value.copy_(saved["buffers"][n])
        assert len(saved["vision"]) == 400
        assert native.old.fingerprint(reference.state_dict()) == native.old.fingerprint(saved["vision"])
        assert native.old.fingerprint(head.state_dict()) == native.old.fingerprint(saved["head"])
        assert native.old.fingerprint(dict(reference.named_buffers())) == base["buffers_sha256"]
        del saved; gc.collect()
        images, rgb = native.pair.augmented_images(control.dataset_root, proof["arms"]["half"]["rows"], (0, 1), 1001)
        pixels = native.pair.pixels(state["processor"], images, "large")
        with torch.no_grad():
            a = model(pixel_values=pixels); b = reference(pixel_values=pixels)
            assert torch.equal(a.pooler_output, b.pooler_output)
            assert torch.equal(a.last_hidden_state, b.last_hidden_state)
            patch_values = model.embeddings.patch_embedding(pixels)
            assert torch.equal(patch_values.flatten(2).transpose(1, 2).reshape(2, 16, 16, 1024), patch_values.permute(0, 2, 3, 1))
            grid = a.last_hidden_state.float().reshape(2, 16, 16, 1024)
            independent = torch.cat([grid[:, r:r+8, c:c+8].mean(dim=(1, 2)) for r, c in ((0,0),(0,8),(8,0),(8,8))], dim=1)
            independent = F.normalize(independent, dim=1)
            assert torch.equal(residual.quadrant_features(a.last_hidden_state), independent)
            old_raw = native.pair.smoke.compact_head_features(a.pooler_output, state["head"])
            raw = residual.raw_features(a.pooler_output, a.last_hidden_state, state["head"], state["residual"])
            replay = residual.raw_features(b.pooler_output, b.last_hidden_state, head, weight)
            assert torch.equal(old_raw, raw) and torch.equal(raw, replay)
            native.old.previous.training.packed_equal(F.normalize(raw, dim=1), F.normalize(replay, dim=1))
            token_dtype = str(a.last_hidden_state.dtype)
        del reference, head, weight, a, b, grid, independent, patch_values, raw, old_raw, replay
    gc.collect()
    # Actual live source tokens, with no second encoder pass in the residual branch.
    result = model(pixel_values=pixels)
    tokens = result.last_hidden_state
    assert tokens.requires_grad and tokens.shape == (2, 256, 1024)
    raw = residual.raw_features(result.pooler_output, tokens, state["head"], state["residual"])
    index = torch.tensor([0, 1])
    target = state["target"][index]
    ce = native.pair.smoke.sharded_mask_arcface_loss(raw, state["classifier"], target,
         torch.arange(128).unsqueeze(0), margin=.3, scale=64)
    rank = qualified.archived.training.lane.valid_rank(raw, state["bank"], state["head"], state["positive"][index], index)
    gradient = torch.autograd.grad(ce + 8 * rank, state["residual"], retain_graph=True)[0]
    assert torch.isfinite(gradient).all() and gradient.norm() > 0
    global_raw = native.pair.smoke.compact_head_features(result.pooler_output, state["head"])
    original_ce = native.pair.smoke.sharded_mask_arcface_loss(global_raw, state["classifier"], target,
         torch.arange(128).unsqueeze(0), margin=.3, scale=64)
    candidate_head = torch.autograd.grad(ce, state["head"].weight, retain_graph=True)[0]
    original_head = torch.autograd.grad(original_ce, state["head"].weight, retain_graph=True)[0]
    assert torch.equal(candidate_head, original_head)
    with torch.no_grad(): state["residual"][0, 0] = .01
    witness = F.linear(residual.quadrant_features(tokens), state["residual"])
    assert witness.count_nonzero() > 0
    live_gradient = torch.autograd.grad(witness.square().sum(), tokens)[0]
    assert torch.isfinite(live_gradient).all() and live_gradient.norm() > 0
    with torch.no_grad(): state["residual"].zero_()
    verify(state, base, "candidate")
    members = state["optimizer"].param_groups[0]["params"]
    removed = members.pop()
    qualified.rejects(lambda: verify(state, base, "candidate")); members.append(removed)
    members.append(removed)
    qualified.rejects(lambda: verify(state, base, "candidate")); members.pop()
    for tensor in (next(iter(qualified.late.frozen_state(model, 12).values())), next(iter(dict(model.named_buffers()).values()))):
        before = tensor.detach().clone()
        with torch.no_grad(): tensor.add_(1)
        qualified.rejects(lambda: verify(state, base, "candidate"))
        with torch.no_grad(): tensor.copy_(before)
    verify(state, base, "candidate")
    assert torch.equal(rng, torch.random.get_rng_state())
    assert state["counter"] == 0 and not state["optimizer"].state
    wrong = torch.zeros(128, 4095)
    qualified.rejects(lambda: residual.validate_weight(wrong, "candidate"))
    qualified.rejects(lambda: residual.validate_weight(state["residual"].double(), "candidate"))
    qualified.rejects(lambda: residual.validate_weight(state["residual"], "foreign"))
    original_sha = sha
    with patch(__name__ + ".sha", lambda p: "changed" if p.resolve() == Path(__file__).resolve() else original_sha(p)):
        qualified.rejects(lambda: closure(root, args.execution_sha256))
    assert closure(root, args.execution_sha256) == code
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    swap = int(next(s for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmSwap:")).split()[1])
    elapsed = time.perf_counter() - STARTED
    assert 0 < rss <= 8 * 1024 * 1024 and swap == 0 and elapsed < 120
    facts = {"schema": "token-residual-cpu-v1", "pass": True, "code": code,
        "execution_sha256": args.execution_sha256, "source_checkpoint_sha256": qualified.late.SOURCE_SHA,
        "initial_state_sha256": initial, "boundary": 12, "optimizer_members": {"control":208,"candidate":209},
        "module_sha256": SIGLIP_SHA, "token_dtype":token_dtype, "token_shape":[2,256,1024],
        "same_call_tokens_and_layout_exact":True, "independent_quadrants_exact":True,
        "zero_raw_and_packed_reload_exact":True, "nonzero_W_gradient":True,
        "nonzero_W_live_token_witness":True, "shared_head_zero_gradient_exact":True,
        "RNG_preserved":True, "source_and_W_negatives_rejected":True,
        "rgb_sha256":rgb, "pixels_sha256":native.pair.smoke.digest({"pixels":pixels}),
        "unit_invocation_id":invocation,"unit_cgroup":str(cgroup), "unit_memory_max_bytes":8*1024**3,
        "unit_memory_swap_max_bytes":0,"host_max_rss_kib":rss,"host_swap_kib":swap,
        "total_seconds":elapsed,"peak_cuda_allocated_bytes":0,"optimizer_updates":0,"quality_read":False}
    native.atomic_write(args.output, lambda stream:stream.write((json.dumps(facts,indent=2)+'\n').encode()))


if __name__ == "__main__":
    main()
