#!/usr/bin/env python3
"""Actual fit-only CPU admission for the fixed image-queue procedure."""
if not __debug__:
    raise SystemExit("Native qualification requires assertions")

import argparse
import collections
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
ADDED = {"image_queue_sampler.py", "test_image_queue_sampler.py",
         "qualify_image_queue_adaptation_cpu.py", "test_image_queue_adaptation_cpu.py"}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def closure(root, expected):
    manifest = root / "image-queue-cpu-execution.json"
    assert sha(manifest) == expected
    code = json.loads(manifest.read_text())
    previous = root / "late-dense-execution.json"
    assert sha(previous) == ORIGINAL
    old = json.loads(previous.read_text())
    assert len(old) == 114 and len(code) == 118
    assert set(code) - set(old) == ADDED and all(code[n] == h for n, h in old.items())
    assert all(sha(root / n) == h for n, h in code.items()), "execution source changed"
    return code


def schedules(target, original, seed):
    from image_queue_sampler import queue_schedule
    candidate = queue_schedule(target, original, seed)
    assert [[target[i] for i in b] for b in original] == [[target[i] for i in b] for b in candidate]
    uses = collections.defaultdict(list)
    sizes = collections.Counter(target)
    for batch in candidate:
        for i in batch:
            uses[target[i]].append(i)
    for label, rows in uses.items():
        for offset in range(0, len(rows), sizes[label]):
            cycle = rows[offset:offset + sizes[label]]
            assert len(cycle) == len(set(cycle)), "repeat before queue exhaustion"
    unique = len({i for b in candidate for i in b})
    assert unique == sum(min(len(rows), sizes[c]) for c, rows in uses.items())
    assert unique == {179032: 6330, 179041: 6331}[seed]
    assert candidate == queue_schedule(target, original, seed)
    return candidate


def main():
    assert __debug__, "assertions required"
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
    import qualify_late_dense_adaptation_cpu as qualified
    import train_late_dense_adaptation as native
    assert not torch.cuda.is_available()
    torch.set_num_threads(8)
    torch.manual_seed(179032)
    control, source, _, proof, _, _ = qualified.startup(root, ORIGINAL)
    native.pair.executing_authority(root, code)
    fingerprints = []
    for _ in range(2):
        state, initial = qualified.fresh(control, source, proof, 12, "cpu")
        fingerprints.append(initial)
        assert state["counter"] == 0 and not state["optimizer"].state and len(state["params"]) == 208
        del state
        gc.collect()
    assert fingerprints[0] == fingerprints[1]
    state, initial = qualified.fresh(control, source, proof, 12, "cpu")
    target = state["target"].tolist()
    rng = torch.random.get_rng_state().clone()
    queue_proofs = {}
    for seed in (179032, 179041):
        original = native.old.coverage.schedule(target, seed=seed).tolist()
        candidate = schedules(target, original, seed)
        queue_proofs[str(seed)] = {"unique_images": len({i for b in candidate for i in b}),
            "control_schedule_sha256": native.old.fingerprint({"batches": torch.tensor(original)}),
            "queue_schedule_sha256": native.old.fingerprint({"batches": torch.tensor(candidate)}),
            "class_sequence_sha256": sha_bytes([[target[i] for i in b] for b in original])}
    identity_args = SimpleNamespace(seed=179032, boundary=12, execution_sha256=args.execution_sha256,
        cpu_sha256=None, cpu_log_sha256=None, cpu_execution_sha256=None)
    base = native.identity(state, identity_args, initial, queue_proofs["179032"]["queue_schedule_sha256"])
    native.verify(state, base)
    state["model"].eval(); state["head"].eval()
    with TemporaryDirectory(dir=root) as temporary:
        path = Path(temporary) / "native.pt"
        torch.save({"vision": state["model"].state_dict(), "head": state["head"].state_dict(),
                    "buffers": dict(state["model"].named_buffers())}, path)
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        with torch.random.fork_rng(devices=[]):
            reference = type(state["model"])(copy.deepcopy(state["model"].config)).float().eval()
            head = torch.nn.Linear(1024, 128).eval()
        reference.load_state_dict(saved["vision"], strict=True); head.load_state_dict(saved["head"], strict=True)
        assert len(saved["vision"]) == 400
        with torch.no_grad():
            for n, b in reference.named_buffers():
                b.copy_(saved["buffers"][n])
        assert native.old.fingerprint(reference.state_dict()) == native.old.fingerprint(saved["vision"])
        assert native.old.fingerprint(head.state_dict()) == native.old.fingerprint(saved["head"])
        assert native.old.fingerprint(dict(reference.named_buffers())) == base["buffers_sha256"]
        del saved; gc.collect()
        images, rgb = native.pair.augmented_images(control.dataset_root, proof["arms"]["half"]["rows"], (0, 1), 1001)
        pixels = native.pair.pixels(state["processor"], images, "large")
        with torch.no_grad():
            a = state["model"](pixel_values=pixels).pooler_output
            b = reference(pixel_values=pixels).pooler_output
            assert torch.equal(a, b)
            va = F.normalize(native.pair.smoke.compact_head_features(a, state["head"]), dim=1)
            vb = F.normalize(native.pair.smoke.compact_head_features(b, head), dim=1)
            assert torch.equal(va, vb)
            native.old.previous.training.packed_equal(va, vb)
        del reference, head
    members = state["optimizer"].param_groups[0]["params"]
    removed = members.pop(); qualified.rejects(lambda: native.verify(state, base)); members.append(removed)
    members.append(removed); qualified.rejects(lambda: native.verify(state, base)); members.pop()
    for tensor in (next(iter(qualified.late.frozen_state(state["model"], 12).values())), next(iter(dict(state["model"].named_buffers()).values()))):
        before = tensor.detach().clone()
        with torch.no_grad(): tensor.add_(1)
        qualified.rejects(lambda: native.verify(state, base))
        with torch.no_grad(): tensor.copy_(before)
    native.verify(state, base)
    assert torch.equal(rng, torch.random.get_rng_state())
    original_sha = sha
    with patch(__name__ + ".sha", lambda path: "changed" if path.resolve() == Path(__file__).resolve() else original_sha(path)):
        qualified.rejects(lambda: closure(root, args.execution_sha256))
    assert closure(root, args.execution_sha256) == code
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    swap = int(next(s for s in Path("/proc/self/status").read_text().splitlines() if s.startswith("VmSwap:")).split()[1])
    elapsed = time.perf_counter() - STARTED
    assert 0 < rss <= 8 * 1024 * 1024 and swap == 0 and elapsed < 120
    native.atomic_write(args.output, lambda stream: stream.write((json.dumps({"schema": "image-queue-cpu-v1",
        "pass": True, "execution_sha256": args.execution_sha256, "code": code,
        "source_checkpoint_sha256": qualified.late.SOURCE_SHA, "initial_state_sha256": initial,
        "schedules": queue_proofs, "boundary": 12, "optimizer_members": 208,
        "matched_control_candidate_initial_state": True, "native400_whole_head_packed_reload_exact": True,
        "optimizer_and_frozen_buffer_negatives_rejected": True, "changed_driver_rejected": True, "RNG_preserved": True,
        "rgb_sha256": rgb, "pixels_sha256": native.pair.smoke.digest({"pixels": pixels}),
        "unit_invocation_id": invocation, "unit_cgroup": str(cgroup), "unit_memory_max_bytes": 8 * 1024**3,
        "unit_memory_swap_max_bytes": 0, "host_max_rss_kib": rss, "host_swap_kib": swap,
        "total_seconds": elapsed, "peak_cuda_allocated_bytes": 0, "optimizer_updates": 0,
        "quality_read": False}, indent=2) + "\n").encode()))


def sha_bytes(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


if __name__ == "__main__":
    main()
