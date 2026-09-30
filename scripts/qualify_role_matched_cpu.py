#!/usr/bin/env python3
"""Actual corrected native128 source and role-mask admission; DGX CPU only."""
if not __debug__:
    raise SystemExit("Qualification requires assertions")

import argparse
import copy
import gc
import hashlib
import inspect
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
ADDED = {"role_matched_bank_rank.py", "test_role_matched_bank_rank.py",
         "qualify_role_matched_cpu.py", "test_role_matched_cpu.py"}
INITIAL = "77c26114a74472682f7f511d732dfac2679d99bfe120ee52d3f78c032a9d9867"
SIGLIP = "274eafdb9bff99607f15eb4c4fb6fc47256b07358807653baa48539d13376b31"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def witness_ordinals(batch, inventory):
    return [int(i) for i in batch if any(p >= 0 for p in inventory[int(i)])][:2]


def closure(root, expected):
    manifest = root / "role-matched-cpu-execution.json"
    assert sha(manifest) == expected
    code = json.loads(manifest.read_text())
    previous = root / "late-dense-execution.json"
    assert sha(previous) == ORIGINAL
    original = json.loads(previous.read_text())
    assert len(original) == 114 and len(code) == 118
    assert set(code) - set(original) == ADDED
    assert all(code[n] == h for n, h in original.items())
    assert all(sha(root / n) == h for n, h in code.items()), "execution source changed"
    return code


def strict_reload(state, pixels, base, root):
    import torch
    from torch.nn import functional as F
    import train_late_dense_adaptation as native
    model = state["model"]
    with TemporaryDirectory(dir=root) as temporary:
        path = Path(temporary) / "native.pt"
        torch.save({"vision": model.state_dict(), "head": state["head"].state_dict(),
                    "buffers": dict(model.named_buffers())}, path)
        with torch.random.fork_rng(devices=[]):
            reference = type(model)(copy.deepcopy(model.config)).float().eval()
            head = torch.nn.Linear(1024, 128).eval()
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        assert len(saved["vision"]) == 400
        reference.load_state_dict(saved["vision"], strict=True)
        head.load_state_dict(saved["head"], strict=True)
        with torch.no_grad():
            for name, value in reference.named_buffers():
                value.copy_(saved["buffers"][name])
        assert native.old.fingerprint(reference.state_dict()) == native.old.fingerprint(saved["vision"])
        assert native.old.fingerprint(head.state_dict()) == native.old.fingerprint(saved["head"])
        assert native.old.fingerprint(dict(reference.named_buffers())) == base["buffers_sha256"]
        del saved
        gc.collect()
        with torch.no_grad():
            a = model(pixel_values=pixels).pooler_output
            b = reference(pixel_values=pixels).pooler_output
            assert torch.equal(a, b)
            a = native.pair.smoke.compact_head_features(a, state["head"])
            b = native.pair.smoke.compact_head_features(b, head)
            assert torch.equal(a, b)
            native.old.previous.training.packed_equal(F.normalize(a, dim=1), F.normalize(b, dim=1))
        del reference, head, a, b
    gc.collect()


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists() and not args.output.is_symlink()
    root = Path(__file__).resolve().parent
    code = closure(root, args.execution_sha256)
    cgroup = Path("/sys/fs/cgroup") / Path("/proc/self/cgroup").read_text().split("0::", 1)[1].strip().lstrip("/")
    assert int((cgroup / "memory.max").read_text()) == 8 * 1024**3
    assert int((cgroup / "memory.swap.max").read_text()) == 0
    assert str(os.getpid()) in (cgroup / "cgroup.procs").read_text().split()
    invocation = os.environ["INVOCATION_ID"]
    import torch
    import role_matched_bank_rank as role
    import test_role_matched_bank_rank as witnesses
    from sfora.deployed_code_rank import smooth_ap_bank_loss as original_loss
    import qualify_late_dense_adaptation_cpu as qualified
    import train_late_dense_adaptation as native
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(179032)
    selected = qualified.archived.training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, "helpers", lambda r, _: helpers(r, code)):
        control, source, _, proof, frozen, _ = qualified.startup(root, ORIGINAL)
    native.pair.executing_authority(root, code)
    state, initial = qualified.fresh(control, source, proof, 12, "cpu")
    assert initial == INITIAL
    assert sha(Path(inspect.getfile(type(state["model"])))) == SIGLIP
    schedule = native.old.coverage.schedule(state["target"].tolist(), seed=179032)
    schedule_sha = native.pair.smoke.digest({"batches": torch.tensor(schedule)})
    base = native.identity(state, SimpleNamespace(seed=179032, boundary=12,
        execution_sha256=args.execution_sha256, cpu_sha256=None,
        cpu_log_sha256=None, cpu_execution_sha256=None), initial, schedule_sha)
    native.verify(state, base)
    assert len(state["params"]) == 208 and not state["optimizer"].state and state["counter"] == 0
    rng = torch.random.get_rng_state().clone()
    fit = proof["arms"]["half"]["rows"]
    held = frozen["held_manifest"]
    facts = witnesses.held_metadata_checks(
        tuple(r["product"] for r in held), tuple(r["relative_path"] for r in held),
        frozen["query"], frozen["gallery"],
        fit_labels=tuple(r["product"] for r in fit),
        fit_paths=tuple(r["relative_path"] for r in fit))
    roles, inventory = role.build_role_inventory(
        tuple(r["product"] for r in fit), tuple(r["relative_path"] for r in fit))
    role_tensor = torch.tensor(roles, dtype=torch.long)
    positives = torch.tensor(inventory, dtype=torch.long)
    mask_sha = native.old.fingerprint({"roles": role_tensor, "positive": positives})
    assert sha(Path(inspect.getfile(original_loss))) == code["src/sfora/deployed_code_rank.py"]
    facts.update(witnesses.native_cpu_checks(original_loss))
    # Metadata-only first eligible images from the original frozen first batch.
    ordinals = witness_ordinals(schedule[0], inventory)
    assert len(ordinals) == 2
    index = torch.tensor(ordinals, dtype=torch.long)
    images, rgb = native.pair.augmented_images(control.dataset_root, fit, tuple(ordinals), 1001)
    pixels = native.pair.pixels(state["processor"], images, "large")
    state["model"].eval()
    state["head"].eval()
    strict_reload(state, pixels, base, root)
    result = state["model"](pixel_values=pixels)
    raw = native.pair.smoke.compact_head_features(result.pooler_output, state["head"])
    control_rank = role.bank_rank_loss(raw, state["bank"], state["positive"][index], index,
        arm="control", labels=state["target"], original_loss=original_loss)
    original_rank = qualified.archived.training.lane.valid_rank(
        raw, state["bank"], state["head"], state["positive"][index], index)
    assert torch.equal(control_rank, original_rank)
    a = torch.autograd.grad(control_rank, raw, retain_graph=True)[0]
    b = torch.autograd.grad(original_rank, raw, retain_graph=True)[0]
    assert torch.equal(a, b)
    bank_version = state["bank"]._version
    candidate_rank = role.bank_rank_loss(raw, state["bank"], positives[index], index,
        arm="role_matched", labels=state["target"], roles=role_tensor)
    assert torch.isfinite(candidate_rank) and candidate_rank >= 0
    members = [p for _, p in state["params"]]
    gradients = torch.autograd.grad(candidate_rank, members, retain_graph=True, allow_unused=True)
    assert gradients[-1] is None, "rank must not update ArcFace proxies"
    assert all(g is not None and bool(torch.isfinite(g).all()) for g in gradients[:-1])
    gradient_by_id = {id(p): g for p, g in zip(members, gradients, strict=True)}
    layer_norms = {str(i): sum(float(gradient_by_id[id(p)].double().norm())
                  for p in state["model"].encoder.layers[i].parameters()) for i in range(12, 24)}
    assert all(v > 0 for v in layer_norms.values())
    head_norm = sum(float(gradient_by_id[id(p)].double().norm()) for p in state["head"].parameters())
    assert head_norm > 0
    del gradients, gradient_by_id
    ce = native.pair.smoke.sharded_mask_arcface_loss(raw, state["classifier"], state["target"][index],
         torch.arange(128).unsqueeze(0), margin=.3, scale=64)
    head_ce = torch.autograd.grad(ce, state["head"].weight, retain_graph=True)[0]
    head_rank = torch.autograd.grad(candidate_rank, state["head"].weight)[0]
    gradient_facts = {"rank_head_norm": float(head_rank.double().norm()),
                     "ce_head_norm": float(head_ce.double().norm()),
                     "rank_layer_norms": layer_norms, "rank_loss": float(candidate_rank.detach()),
                     "control_rank_loss": float(control_rank.detach()), "ce_loss": float(ce.detach())}
    del result, raw, a, b, head_ce, head_rank, ce, candidate_rank, control_rank, original_rank
    assert state["bank"]._version == bank_version and state["bank"].grad is None
    native.verify(state, base)
    source_values = {"vision": state["model"].state_dict(), "head": state["head"].state_dict(),
        "classifier": state["classifier"].detach(), "bank": state["bank"],
        "buffers": dict(state["model"].named_buffers())}
    assert native.old.fingerprint(source_values) == initial
    assert torch.equal(rng, torch.random.get_rng_state())
    assert state["counter"] == 0 and not state["optimizer"].state
    assert closure(root, args.execution_sha256) == code
    elapsed = time.perf_counter() - STARTED
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    swap = int(next(s for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmSwap:")).split()[1])
    assert elapsed < 120 and 0 < rss <= 8 * 1024 * 1024 and swap == 0
    value = {"schema": "role-matched-cpu-v1", "pass": True, "code": code,
        "execution_sha256": args.execution_sha256, "source_checkpoint_sha256": qualified.late.SOURCE_SHA,
        "initial_state_sha256": initial, "boundary": 12, "optimizer_members": 208,
        "siglip_module_sha256": SIGLIP, "schedule_sha256": schedule_sha,
        "role_positive_sha256": mask_sha, "module_sha256": code["role_matched_bank_rank.py"],
        "fit_query_images": roles.count(role.QUERY), "fit_gallery_images": roles.count(role.GALLERY),
        "native400_whole_head_packed_reload_exact": True, "control_loss_gradient_exact": True,
        "rank_live_source_head_layers_gradient": True, "bank_detached_unchanged": True,
        "source_roles_complete_state_exact": True, "RNG_preserved": True,
        "witnesses": facts, "gradient_facts": gradient_facts, "fit_ordinals": ordinals,
        "rgb_sha256": rgb, "pixels_sha256": native.pair.smoke.digest({"pixels": pixels}),
        "unit_invocation_id": invocation, "unit_cgroup": str(cgroup),
        "unit_memory_max_bytes": 8 * 1024**3, "unit_memory_swap_max_bytes": 0,
        "host_max_rss_kib": rss, "host_swap_kib": swap, "total_seconds": elapsed,
        "peak_cuda_allocated_bytes": 0, "optimizer_updates": 0, "quality_read": False}
    native.atomic_write(args.output, lambda stream: stream.write((json.dumps(value, indent=2) + "\n").encode()))


if __name__ == "__main__":
    main()
