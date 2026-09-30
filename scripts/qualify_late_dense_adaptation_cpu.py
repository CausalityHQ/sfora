#!/usr/bin/env python3
"""Actual native late-stage source admission; DGX CPU only, no quality/update."""
if not __debug__:
    raise SystemExit("Native qualification requires assertions")

import argparse
import gc
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.nn import functional as F

import qualify_pe_teacher_retained256 as archived
import late_dense_boundary as late

old = archived.driver
SOURCE_ROOT = Path("/home/riomus/runs/sfora-late-dense-source-v1")
ORIGINAL_CODE = "82ada79acff29017c324626214448844e31dba81a386e56b122cf0f25fcd8df7"
RESUME = archived.run_path(128) / "resume.pt"
REQUIRED = {"large_dense_boundary.py", "late_dense_boundary.py",
            "qualify_late_dense_adaptation_cpu.py", "inshop_late_dense_adaptation_gate_2026-09-30.md"}
ALLOWED = REQUIRED | {"test_late_dense_boundary.py", "train_late_dense_adaptation.py",
                      "test_late_dense_adaptation.py"}


def startup(root, expected):
    manifest = root / "late-dense-execution.json"
    assert old.pair.sha(manifest) == expected
    code = json.loads(manifest.read_text())
    previous = root / "teacher-retained256-qualification-execution.json"
    assert old.pair.sha(previous) == ORIGINAL_CODE
    original = json.loads(previous.read_text())
    assert all(code[n] == h for n, h in original.items())
    assert REQUIRED <= set(code) - set(original) <= ALLOWED
    assert all(old.pair.sha(root / n) == h for n, h in code.items())
    selected = archived.training.previous.cpu.previous.qualified.confirmation.selected
    helpers = selected.helpers
    with patch.object(selected, "helpers", lambda r, _: helpers(r, code)):
        control, source, prior, proof, frozen, _, receipts = archived.authority(root, ORIGINAL_CODE)
    terminal = receipts[128][-1]
    assert terminal["checkpoint_sha256"] == late.SOURCE_SHA
    assert terminal["completed_step"] == 1000 and terminal["width"] == 128
    assert old.pair.sha(RESUME) == late.SOURCE_SHA
    fit, held = proof["arms"]["half"]["rows"], frozen["held_manifest"]
    assert fit == frozen["fit_manifest"] and len(fit) == 13283 and len(held) == 12599
    assert len({r["product"] for r in fit}) == 2004
    assert len({r["product"] for r in held}) == 1993
    assert {r["product"] for r in fit}.isdisjoint(r["product"] for r in held)
    assert {r["relative_path"] for r in fit}.isdisjoint(r["relative_path"] for r in held)
    return control, source, prior, proof, frozen, code


def fresh(control, source, proof, boundary, device):
    state = old.fresh(control, source, proof, "half", device)
    fingerprint = late.initialize(state, proof, RESUME, boundary)
    return state, fingerprint


def rejects(action):
    try:
        action()
    except AssertionError:
        return
    raise AssertionError("negative authority case accepted")


def main():
    assert not torch.cuda.is_available()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(179032)
    control, source, prior, proof, frozen, code = startup(root, args.execution_sha256)
    fingerprints = {}
    for boundary in (12, 10):
        state, fingerprints[boundary] = fresh(control, source, proof, boundary, "cpu")
        assert state["counter"] == 0 and not state["optimizer"].state
        assert len(state["params"]) == (208 if boundary == 12 else 240)
        del state
        gc.collect()
    assert fingerprints[12] == fingerprints[10]
    state, fingerprint = fresh(control, source, proof, 10, "cpu")
    import large_dense_boundary as dense
    dense.verify_optimizer(state)
    before = {n: v.detach().clone() for n, v in late.frozen_state(state["model"], 10).items()}
    buffers = dict(state["model"].named_buffers())
    assert buffers and all(n in before for n in buffers)
    rng = torch.random.get_rng_state().clone()
    state["model"].eval(); state["head"].eval()
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "native.pt"
        torch.save({"vision": state["model"].state_dict(), "head": state["head"].state_dict(), "buffers": buffers}, path)
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        with torch.random.fork_rng(devices=[]):
            reference = type(state["model"])(state["model"].config).eval()
            head = torch.nn.Linear(1024, 128).eval()
        reference.load_state_dict(saved["vision"], strict=True)
        head.load_state_dict(saved["head"], strict=True)
        assert old.fingerprint(dict(reference.named_buffers())) == old.fingerprint(saved["buffers"])
        images, _ = old.pair.augmented_images(control.dataset_root, proof["arms"]["half"]["rows"], (0, 1), None)
        pixels = old.pair.pixels(state["processor"], images, "large")
        with torch.no_grad():
            a = state["model"](pixel_values=pixels).pooler_output
            b = reference(pixel_values=pixels).pooler_output
            assert torch.equal(a, b)
            a = F.normalize(old.pair.smoke.compact_head_features(a, state["head"]), dim=1)
            b = F.normalize(old.pair.smoke.compact_head_features(b, head), dim=1)
            old.previous.training.packed_equal(a, b)
        del reference, head, saved
    members = state["optimizer"].param_groups[0]["params"]
    removed = members.pop()
    rejects(lambda: dense.verify_optimizer(state))
    members.append(removed); members.append(removed)
    rejects(lambda: dense.verify_optimizer(state))
    members.pop()
    for tensor in (next(iter(late.frozen_state(state["model"], 10).values())), next(iter(buffers.values()))):
        original = tensor.detach().clone()
        with torch.no_grad():
            tensor.add_(1)
        rejects(lambda: dense.verify_frozen(state["model"], before))
        with torch.no_grad():
            tensor.copy_(original)
    dense.verify_optimizer(state); dense.verify_frozen(state["model"], before)
    assert torch.equal(rng, torch.random.get_rng_state())
    sha = old.pair.sha
    with patch.object(old.pair, "sha", lambda p: "changed" if Path(p).resolve() == Path(__file__).resolve() else sha(p)):
        rejects(lambda: startup(root, args.execution_sha256))
    old.pair.smoke.save(args.output, {"pass": True, "code": code, "execution_sha256": args.execution_sha256,
        "source_checkpoint_sha256": late.SOURCE_SHA, "initial_state_sha256": fingerprint,
        "matched_control_candidate_initial_state": True, "fit_held_identity_and_path_disjoint": True,
        "native400_whole_head_packed_reload_exact": True, "optimizer_members": {"12": 208, "10": 240},
        "missing_duplicate_optimizer_and_frozen_buffer_mutation_rejected": True,
        "changed_driver_rejected": True, "RNG_preserved": True, "optimizer_updates": 0, "quality_read": False})
    print("PASS actual native late-stage CPU constructor/reload; no training or quality", flush=True)


if __name__ == "__main__":
    main()
