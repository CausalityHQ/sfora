#!/usr/bin/env python3
"""Pinned public PE-L14 acquisition and CPU224 native authority only."""

import argparse
import json
import time
from pathlib import Path

import torch
import pe_core_authority
from huggingface_hub import hf_hub_download
from PIL import Image

from pe_core_authority import sha, visual_state

REVISION = "bafb0f76541d399057e980a25947f67acec76575"
WEIGHT_SHA = "0cdab5b338cbaa1e7a5dcd1b2fb4c9f4d5df1abd289564658edbab64a650e7e8"
WEIGHT_BYTES = 2684747432


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--source-manifest-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    assert not torch.cuda.is_available()
    root = Path(__file__).resolve().parent
    assert sha(Path(__file__)) == args.script_sha256
    helpers = {
        "pe_core_authority.py": "b84697d272198c72fd2161d8018dd5b3de8f3a89880536163ea1e16a37e688ba",
        "probe_inshop_pe_training_smoke.py": "45d89389e02231434b073734e22a309189f7f6c566d2da87b703f7ff06001954",
    }
    assert Path(pe_core_authority.__file__).resolve() == root / "pe_core_authority.py"
    assert all(sha(root / n) == h for n, h in helpers.items())
    manifest_path = root / "source-manifest.json"
    assert sha(manifest_path) == args.source_manifest_sha256
    authority = json.loads(manifest_path.read_text())
    assert all(sha(root / n) == h for n, h in authority.items())
    torch.set_num_threads(8)
    torch.manual_seed(179036)
    output = root / "l14-acquisition-v1.json"
    attempt = root / "l14-acquisition-attempt-v1.json"
    assert not output.exists()
    with attempt.open("x") as file:
        json.dump(
            {
                "revision": REVISION,
                "weight_sha256": WEIGHT_SHA,
                "source_manifest_sha256": args.source_manifest_sha256,
                "script_sha256": args.script_sha256,
            },
            file,
        )
    checkpoint = Path(
        hf_hub_download(
            "facebook/PE-Core-L14-336",
            "PE-Core-L14-336.pt",
            revision=REVISION,
            token=False,
        )
    )
    assert checkpoint.stat().st_size == WEIGHT_BYTES and sha(checkpoint) == WEIGHT_SHA
    from einops import _torch_specific
    from core.vision_encoder.pe import VisionTransformer
    from core.vision_encoder.transforms import get_image_transform

    assert Path(_torch_specific.__file__).is_relative_to(root)
    assert Path(
        __import__("core.vision_encoder.pe", fromlist=["x"]).__file__
    ).is_relative_to(root)
    state = visual_state(
        torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    )
    model = (
        VisionTransformer.from_config("PE-Core-L14-336", pretrained=False)
        .float()
        .eval()
    )
    assert model.image_size == 336 and model.width == 1024 and model.layers == 24
    expected = model.state_dict()
    assert state.keys() == expected.keys()
    assert all(
        state[n].shape == expected[n].shape and torch.isfinite(state[n]).all()
        for n in expected
    )
    model.load_state_dict(state, strict=True)
    dtypes = sorted({str(v.dtype) for v in state.values()})
    assert all(
        p.dtype == torch.float32 and p.device.type == "cpu" for p in model.parameters()
    )
    fit_path = Path("/home/riomus/runs/sfora-pe-fullfit-cache-v3/preflight.json")
    assert (
        sha(fit_path)
        == "4e6c886e87ec8abca45455c5790e35e252cf60694f98d043a1c55d5d21aea3ff"
    )
    fit = json.loads(fit_path.read_text())["manifest"][:2]
    processor = get_image_transform(224)
    images = []
    for row in fit:
        path = (
            Path("/home/riomus/datasets/inshop_official_standard")
            / row["relative_path"]
        )
        assert sha(path) == row["image_sha256"]
        with Image.open(path) as image:
            images.append(processor(image.convert("RGB")))
    pixels = torch.stack(images)
    assert pixels.shape == (2, 3, 224, 224)
    # Authenticate source weights before/after interpolation and forward.
    import probe_inshop_pe_training_smoke
    from probe_inshop_pe_training_smoke import digest

    assert (
        Path(probe_inshop_pe_training_smoke.__file__).resolve()
        == root / "probe_inshop_pe_training_smoke.py"
    )

    before = digest(model.state_dict())
    with torch.inference_mode():
        values = model(pixels)
    assert (
        values.shape == (2, 1024)
        and torch.isfinite(values).all()
        and (values.norm(dim=1) > 0).all()
    )
    assert digest(model.state_dict()) == before and model.image_size == 336
    assert all(sha(root / n) == h for n, h in authority.items())
    assert all(sha(root / n) == h for n, h in helpers.items())
    result = {
        "pass": True,
        "checkpoint": str(checkpoint),
        "revision": REVISION,
        "weight_sha256": WEIGHT_SHA,
        "weight_bytes": WEIGHT_BYTES,
        "vision_parameters": sum(p.numel() for p in model.parameters()),
        "strict_visual_keys": len(expected),
        "checkpoint_visual_dtypes": dtypes,
        "parameter_dtype": "torch.float32",
        "pretrained_image_size": 336,
        "actual_input_size": 224,
        "actual_output_shape": list(values.shape),
        "source_weights_unchanged": True,
        "fit_rows": fit,
        "processor_pixels_sha256": digest({"pixels": pixels}),
        "source_state_sha256": before,
        "foreign_rope_sha256": digest(model.rope.rope.state_dict()),
        "foreign_grid_sha256": digest({"freq": model.rope.freq}),
        "script_sha256": args.script_sha256,
        "source_manifest_sha256": args.source_manifest_sha256,
        "helper_sha256": helpers,
        "seconds": time.perf_counter() - started,
        "optimizer_updates": 0,
        "held_images": 0,
        "cuda_used": False,
        "quality_read": False,
    }
    with output.open("x") as file:
        json.dump(result, file, indent=2, allow_nan=False)
        file.write("\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
