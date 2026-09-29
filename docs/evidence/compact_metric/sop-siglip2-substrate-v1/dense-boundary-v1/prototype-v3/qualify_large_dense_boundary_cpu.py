#!/usr/bin/env python3
"""Actual dense10 constructor qualification; no optimizer updates or quality reads."""
import argparse
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import train_large_lower_pilot as pilot
import large_dense_boundary as dense

TRAIN_CODE = "7a19df8b60267c928ac487642e3bd44b898205227019938af8cb806ba5a95a7d"


def startup(root, expected, seed):
    sha = pilot.mechanics.sha
    assert sha(root / "dense-boundary-execution.json") == expected
    code = json.loads((root / "dense-boundary-execution.json").read_text())
    prior = json.loads((root / "lower-pilot-execution.json").read_text())
    assert set(code) == set(prior) | {"large_dense_boundary.py", "check_large_dense_boundary.py", "qualify_large_dense_boundary_cpu.py"}
    assert all(code[n] == h for n, h in prior.items())
    assert all(sha(root / n) == h for n, h in code.items())
    helpers = pilot.old.previous.selected.helpers
    with patch.object(pilot.old.previous.selected, "helpers", lambda r, _: helpers(r, code)):
        values = pilot.startup(root, TRAIN_CODE, seed)
    return values, code


def rejects(action):
    try:
        action()
    except AssertionError:
        return
    raise AssertionError("negative qualification accepted")


def main():
    assert __debug__ and not torch.cuda.is_available()
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--seed", type=int, choices=tuple(pilot.CONTROLS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    root = Path(__file__).resolve().parent
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    (control, native, _, proof, _, archived, _), code = startup(root, args.execution_sha256, args.seed)
    state = dense.fresh(control, native, proof, "cpu")
    assert pilot.initial_fingerprint(state, args.seed) == archived["initial_state_sha256"]
    dense.verify_optimizer(state)
    frozen = {n: v.detach().clone() for n, v in dense.frozen_state(state["model"]).items()}
    rng = torch.random.get_rng_state().clone()
    state["model"].eval()
    state["head"].eval()
    reference = copy.deepcopy(state["model"])
    reference_head = copy.deepcopy(state["head"])
    with TemporaryDirectory() as temporary:
        path = Path(temporary) / "native.pt"
        torch.save({"vision": state["model"].state_dict(), "head": state["head"].state_dict()}, path)
        saved = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        assert len(saved["vision"]) == 400
        reference.load_state_dict(saved["vision"], strict=True)
        reference_head.load_state_dict(saved["head"], strict=True)
        rows = proof["arms"]["half"]["rows"]
        images, _ = pilot.old.pair.augmented_images(control.dataset_root, rows, (0, 1), None)
        pixels = pilot.old.pair.pixels(state["processor"], images, "large")
        assert pilot.old.pair.smoke.digest({"pixels": pixels}) == native["first_two_fit_pixels_sha256"]
        with torch.no_grad():
            a = state["model"](pixel_values=pixels).pooler_output
            b = reference(pixel_values=pixels).pooler_output
            assert torch.equal(a, b)
            va = pilot.old.pair.smoke.compact_head_features(a, state["head"])
            vb = pilot.old.pair.smoke.compact_head_features(b, reference_head)
            assert torch.equal(va, vb)
            pilot.old.previous.training.packed_equal(torch.nn.functional.normalize(va, dim=1), torch.nn.functional.normalize(vb, dim=1))
    del reference, reference_head, saved
    members = state["optimizer"].param_groups[0]["params"]
    removed = members.pop()
    rejects(lambda: dense.verify_optimizer(state))
    members.append(removed)
    members.append(removed)
    rejects(lambda: dense.verify_optimizer(state))
    members.pop()
    state["model"].encoder.layers[9].requires_grad_(True)
    rejects(lambda: dense.verify_optimizer(state))
    state["model"].encoder.layers[9].requires_grad_(False)
    before = {n: v.detach().clone() for n, v in dense.frozen_state(state["model"]).items()}
    first = next(iter(dense.frozen_state(state["model"]).values()))
    with torch.no_grad():
        first.add_(1)
    rejects(lambda: dense.verify_frozen(state["model"], before))
    with torch.no_grad():
        first.copy_(next(iter(before.values())))
    dense.verify_optimizer(state)
    dense.verify_frozen(state["model"], frozen)
    assert torch.equal(rng, torch.random.get_rng_state())
    original_sha = pilot.mechanics.sha
    with patch.object(pilot.mechanics, "sha", lambda p: "changed" if Path(p).resolve() == Path(__file__).resolve() else original_sha(p)):
        rejects(lambda: startup(root, args.execution_sha256, args.seed))
    pilot.old.pair.smoke.save(args.output, {"pass": True, "seed": args.seed, "execution_sha256": args.execution_sha256,
        "initial_state_sha256": archived["initial_state_sha256"], "optimizer_members": len(state["params"]),
        "frozen_tensor_count": len(frozen), "native400_whole_head_packed_reload_exact": True,
        "wrong_boundary_missing_duplicate_optimizer_and_frozen_mutation_rejected": True,
        "changed_driver_rejected": True, "rng_preserved": True, "optimizer_updates": 0, "quality_read": False})
    print("PASS actual dense10 CPU constructor/optimizer/native400/head/packed and negative authority", flush=True)


if __name__ == "__main__":
    main()
