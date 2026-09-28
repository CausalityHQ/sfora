#!/usr/bin/env python3
"""Pinned PE acquisition and strict CPU visual-state authority; no CUDA use."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

REVISION = "a16450b46fef32363459920c2685a1b4ef13dcd9"
WEIGHT_SHA = "0a5c220aa083488e0fc9221f766dced8a576b7074662d9b9da923da1e8844fce"
WEIGHT_BYTES = 1790786632


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def visual_state(state):
    if "state_dict" in state:
        state = state["state_dict"]
    elif "weights" in state:
        state = state["weights"]
    canonical = {}
    for name, tensor in state.items():
        if not isinstance(name, str):
            raise ValueError("non-string state key")
        key = name.removeprefix("module.")
        if key in canonical:
            raise ValueError("prefix-removal collision")
        canonical[key] = tensor
    if any(key.startswith("visual.") for key in canonical):
        canonical = {
            key.removeprefix("visual."): value
            for key, value in canonical.items()
            if key.startswith("visual.")
        }
    if not canonical or not all(isinstance(value, torch.Tensor) for value in canonical.values()):
        raise ValueError("visual state absent or invalid")
    return canonical


def load_visual(native_root, checkpoint):
    from core.vision_encoder.pe import VisionTransformer

    authority = json.loads((native_root / "source-manifest.json").read_text())
    assert all(sha(native_root / name) == expected for name, expected in authority.items())
    assert checkpoint.stat().st_size == WEIGHT_BYTES and sha(checkpoint) == WEIGHT_SHA
    state = visual_state(torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True))
    model = VisionTransformer.from_config("PE-Core-B16-224", pretrained=False).float()
    expected = model.state_dict()
    assert state.keys() == expected.keys()
    assert all(state[key].shape == expected[key].shape for key in expected)
    assert all(torch.isfinite(value).all() for value in state.values())
    dtypes = sorted({str(value.dtype) for value in state.values()})
    model.load_state_dict(state, strict=True)
    assert all(p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters())
    return model.eval(), dtypes


def self_test():
    model = torch.nn.Linear(2, 3)
    state = {"module.visual." + key: value for key, value in model.state_dict().items()}
    state["module.text.weight"] = torch.zeros(1)
    model.load_state_dict(visual_state({"state_dict": state}), strict=True)
    for bad in (
        {"module.visual.weight": model.weight, "visual.weight": model.weight},
        {"visual.weight": model.weight},
    ):
        try:
            model.load_state_dict(visual_state(bad), strict=True)
        except (ValueError, RuntimeError):
            pass
        else:
            raise AssertionError("collision or missing-state trap failed")
    print("PASS anchored visual state, prefix collision and missing-key rejection")


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    torch.set_num_threads(8)
    if args.self_test:
        self_test()
        return
    assert not args.output.exists()
    checkpoint = Path(
        hf_hub_download("facebook/PE-Core-B16-224", "PE-Core-B16-224.pt", revision=REVISION)
    )
    model, dtypes = load_visual(args.native_root, checkpoint)
    result = {
        "checkpoint": str(checkpoint),
        "revision": REVISION,
        "weight_sha256": WEIGHT_SHA,
        "weight_bytes": WEIGHT_BYTES,
        "checkpoint_visual_dtypes": dtypes,
        "parameter_dtype": "torch.float32",
        "strict_visual_keys": len(model.state_dict()),
        "vision_parameters": sum(p.numel() for p in model.parameters()),
        "script_sha256": sha(__file__),
        "claim_eligible": False,
        "cuda_used": False,
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
