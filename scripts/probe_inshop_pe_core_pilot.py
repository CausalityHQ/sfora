#!/usr/bin/env python3
"""Frozen PE versus SigLIP2 Large fit-only prototype/cost pilot, no learning."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

from core.vision_encoder import pe as pe_module
from core.vision_encoder.transforms import get_image_transform
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from pe_core_authority import load_visual, sha
from probe_inshop_siglip2_base_pilot import packed_hits
from sfora.representation_ceiling import fit_centered_pca


def lower_bound(delta):
    rng = np.random.Generator(np.random.PCG64(179031))
    draws = rng.integers(len(delta), size=(5000, len(delta)))
    return float(np.quantile(delta[draws].mean(1), 0.025))


def code_authority():
    root = Path(__file__).resolve().parent
    return {
        str(Path(module.__file__).resolve().relative_to(root)): sha(module.__file__)
        for name, module in sorted(sys.modules.items())
        if getattr(module, "__file__", None)
        and Path(module.__file__).is_file()
        and Path(module.__file__).resolve().is_relative_to(root)
    }


def main():
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "native-root",
        "dataset-root",
        "large-snapshot",
        "preflight",
        "acquisition",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    assert code_authority() == json.loads(
        (args.native_root / "pilot-code-authority.json").read_text()
    )
    assert torch.cuda.is_available() and not args.output.exists()
    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    frozen = json.loads(args.preflight.read_text())
    acquisition = json.loads(args.acquisition.read_text())
    assert args.large_snapshot.name == MODEL_REVISION
    assert all(
        sha(args.large_snapshot / name) == expected for name, expected in MODEL_HASHES.items()
    )
    assert all(
        sha(args.native_root / name) == expected
        for name, expected in frozen["source_sha256"].items()
    )
    paths = [args.dataset_root / row["relative_path"] for row in frozen["image_manifest"]]
    assert len(paths) == 1024 and all(
        sha(path) == row["image_sha256"]
        for path, row in zip(paths, frozen["image_manifest"], strict=True)
    )
    pe, pe_dtypes = load_visual(args.native_root, Path(acquisition["checkpoint"]))
    assert isinstance(pe, pe_module.VisionTransformer)
    assert sum(p.numel() for p in pe.parameters()) == frozen["vision_parameters"]
    full = AutoModel.from_pretrained(
        args.large_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float32
    )
    large = full.vision_model
    del full
    assert len(large.encoder.layers) == 24 and large.config.hidden_size == 1024
    assert all(p.dtype == torch.float32 for p in large.parameters())
    large_processor = AutoImageProcessor.from_pretrained(
        args.large_snapshot, local_files_only=True, backend="torchvision"
    )
    pe_processor = get_image_transform(224)
    models = {"large": large.cuda().eval(), "pe": pe.cuda().eval()}
    outputs = {arm: [] for arm in models}
    fixed = {}
    cosine = None
    export_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, 1024, 32):
            images = []
            for path in paths[start : start + 32]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            assert all(image.mode == "RGB" for image in images)
            pixels = {
                "large": large_processor(images=images, return_tensors="pt")["pixel_values"],
                "pe": torch.stack([pe_processor(image) for image in images]),
            }
            assert torch.equal(
                pixels["large"], large_processor(images=images, return_tensors="pt")["pixel_values"]
            )
            assert torch.equal(pixels["pe"], torch.stack([pe_processor(image) for image in images]))
            pixels = {arm: value.cuda() for arm, value in pixels.items()}
            if not fixed:
                fixed = pixels
                fp32 = models["pe"](pixels["pe"][:4])
                with torch.autocast("cuda", dtype=torch.float16):
                    fp16 = models["pe"](pixels["pe"][:4])
                assert torch.isfinite(fp32).all() and torch.isfinite(fp16).all()
                assert bool((fp32.norm(dim=1) > 0).all()) and bool(
                    (fp16.float().norm(dim=1) > 0).all()
                )
                cosine = torch.nn.functional.cosine_similarity(fp32, fp16.float()).tolist()
                assert min(cosine) >= 0.999
            with torch.autocast("cuda", dtype=torch.float16):
                for arm, model in models.items():
                    values = (
                        model(pixels[arm])
                        if arm == "pe"
                        else model(pixel_values=pixels[arm]).pooler_output
                    )
                    assert values.shape == (32, 1024) and torch.isfinite(values).all()
                    assert bool((values.float().norm(dim=1) > 0).all())
                    outputs[arm].append(torch.nn.functional.normalize(values.float(), dim=1).cpu())
        torch.cuda.synchronize()
        export_wall = time.perf_counter() - export_started
        times = {arm: [] for arm in models}
        with torch.autocast("cuda", dtype=torch.float16):

            def forward(arm):
                return (
                    models[arm](fixed[arm]) if arm == "pe" else models[arm](pixel_values=fixed[arm])
                )

            for _ in range(3):
                for arm in models:
                    forward(arm)
            torch.cuda.synchronize()
            for block in range(10):
                for arm in ("large", "pe") if block % 2 == 0 else ("pe", "large"):
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    forward(arm)
                    torch.cuda.synchronize()
                    times[arm].append(1000 * (time.perf_counter() - tick))
    arms = {}
    source_artifacts = {}
    for arm in models:
        source = torch.cat(outputs[arm])
        source_artifacts[arm] = source.numpy()
        query, gallery = source[:512], source[512:]
        pca = fit_centered_pca(gallery, dimensions=128)
        hit = packed_hits(
            torch.nn.functional.normalize(pca.apply(query), dim=1),
            torch.nn.functional.normalize(pca.apply(gallery), dim=1),
        ).numpy()
        arms[arm] = {
            "packed_hits": hit.tolist(),
            "packed_r1": float(hit.mean()),
            "raw_float_r1": float(
                ((query @ gallery.T).argmax(1) == torch.arange(512)).float().mean()
            ),
            "encoder_batch32_ms": times[arm],
            "encoder_batch32_p50_ms": float(np.median(times[arm])),
            "source_sha256": hashlib.sha256(source.numpy().tobytes()).hexdigest(),
        }
    delta = np.array(arms["pe"]["packed_hits"]) - np.array(arms["large"]["packed_hits"])
    lower = lower_bound(delta)
    features_path = args.output.with_suffix(".features.npz")
    with features_path.open("xb") as stream:
        np.savez(stream, **source_artifacts)
    ratio = arms["pe"]["encoder_batch32_p50_ms"] / arms["large"]["encoder_batch32_p50_ms"]
    wall = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated()
    criteria = {
        "quality_point": float(delta.mean()) >= -0.03,
        "quality_lower": lower >= -0.05,
        "encoder_time": ratio <= 0.8,
        "wall": wall <= 120,
        "cuda": peak < 10_000_000_000,
    }
    result = {
        "arms": arms,
        "gain_pp": 100 * float(delta.mean()),
        "gain_lower95_pp": 100 * lower,
        "encoder_batch32_p50_ratio": ratio,
        "criteria": criteria,
        "advance": all(criteria.values()),
        "main_wall_seconds": wall,
        "export_wall_seconds": export_wall,
        "peak_cuda_allocated_bytes": peak,
        "fp16_fp32_cosine_first4": cosine,
        "pe_checkpoint_dtypes": pe_dtypes,
        "parameter_dtype": "torch.float32",
        "preflight_sha256": sha(args.preflight),
        "acquisition_sha256": sha(args.acquisition),
        "script_sha256": sha(__file__),
        "claim_eligible": False,
    }
    result["features_sha256"] = sha(features_path)
    result["code_authority"] = code_authority()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "arms"}))


if __name__ == "__main__":
    main()
