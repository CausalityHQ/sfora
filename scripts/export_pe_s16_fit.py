#!/usr/bin/env python3
"""Fixed224 native S16 FP16 fit-only cache; no training or quality read."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from einops import _torch_specific
from core.vision_encoder.pe import VisionTransformer
from core.vision_encoder.transforms import get_image_transform

import acquire_pe_s16 as source
import probe_inshop_pe_training_smoke as smoke
from export_inshop_pe_fit_features import FIT_SHA, validate_membership
from export_sop_siglip2_train import export_features
from pe_core_authority import sha, visual_state
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from sfora.unicom_inshop import parse_inshop_partition

ROOT = Path(__file__).resolve().parent
DATA = Path("/home/riomus/datasets/inshop_official_standard")
FIT = Path("/home/riomus/runs/sfora-pe-fullfit-cache-v3/preflight.json")
FIT_SHA256 = "4e6c886e87ec8abca45455c5790e35e252cf60694f98d043a1c55d5d21aea3ff"
SOURCE_SHA256 = "f9c9a7748d04f6d512dc602f33029b736d149847b49b54e65488c5c78fe04600"
VERSIONS = (
    "torch",
    "torchvision",
    "Pillow",
    "numpy",
    "timm",
    "einops",
    "huggingface_hub",
)


def source_authority():
    path = ROOT / "s16-acquisition-v1.json"
    assert sha(path) == SOURCE_SHA256
    receipt = json.loads(path.read_text())
    assert receipt["pass"] and receipt["revision"] == source.REVISION
    assert receipt["weight_sha256"] == source.WEIGHT_SHA
    assert sha(ROOT / "acquire_pe_s16.py") == receipt["script_sha256"]
    assert sha(ROOT / "source-manifest.json") == receipt["source_manifest_sha256"]
    native = json.loads((ROOT / "source-manifest.json").read_text())
    assert all(
        sha(ROOT / n) == h for n, h in {**native, **receipt["helper_sha256"]}.items()
    )
    for module in (source, smoke, _torch_specific):
        assert Path(module.__file__).resolve().is_relative_to(ROOT)
    assert (
        Path(__import__("core.vision_encoder.pe", fromlist=["x"]).__file__)
        .resolve()
        .is_relative_to(ROOT)
    )
    checkpoint = Path(receipt["checkpoint"])
    assert (
        checkpoint.stat().st_size == source.WEIGHT_BYTES
        and sha(checkpoint) == source.WEIGHT_SHA
    )
    assert (
        sha(FIT) == FIT_SHA256
        and sha(DATA / "Eval/list_eval_partition.txt") == PARTITION_SHA
    )
    return receipt


def preflight(output):
    assert not torch.cuda.is_available()
    source_authority()
    frozen = json.loads(FIT.read_text())
    rows = tuple(r for r in parse_inshop_partition(DATA) if r.split == "train")
    labels = tuple(r.label for r in rows)
    fit, held = split(labels)
    validate_membership(labels, fit, held)
    manifest = frozen["manifest"]
    assert (
        tuple(m["train_row"] for m in manifest) == fit and digest_rows(fit) == FIT_SHA
    )
    for i, m in zip(fit, manifest, strict=True):
        assert m["product"] == rows[i].label and m["relative_path"] == str(
            rows[i].image_path.relative_to(DATA)
        )
        assert sha(rows[i].image_path) == m["image_sha256"]
    output.mkdir(exist_ok=False)
    smoke.save(
        output / "preflight.json",
        {
            "manifest": manifest,
            "code": smoke.authority(),
            "versions": {n: importlib.metadata.version(n) for n in VERSIONS},
            "source_receipt_sha256": SOURCE_SHA256,
            "prior_fit_manifest_sha256": FIT_SHA256,
            "fit_sha256": FIT_SHA,
            "quality_read": False,
        },
    )
    print(
        "PASS exact full fit/source/code/environment/image authority; no CUDA or held decoding"
    )


def startup(path, expected_sha):
    assert expected_sha and sha(path) == expected_sha
    frozen = json.loads(path.read_text())
    source_authority()
    assert all(sha(ROOT / n) == h for n, h in frozen["code"].items())
    current = smoke.authority()
    assert set(current) <= set(frozen["code"])
    assert all(current[n] == frozen["code"][n] for n in current)
    assert {n: importlib.metadata.version(n) for n in VERSIONS} == frozen["versions"]
    assert frozen["manifest"] == json.loads(FIT.read_text())["manifest"]
    return frozen


def check_features(path, rows):
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    assert (
        values.shape == (rows, 512)
        and values.dtype == np.float32
        and np.isfinite(values).all()
    )
    error = float(np.max(np.abs(np.linalg.norm(values.astype(np.float64), axis=1) - 1)))
    assert error < 1e-5
    return error


def acquire(output, expected_sha):
    started = time.perf_counter()
    frozen = startup(output / "preflight.json", expected_sha)
    assert torch.cuda.is_available() and not (output / "receipt.json").exists()
    smoke.save(output / "attempt.json", {"preflight_sha256": expected_sha})
    source_receipt = source_authority()
    state = visual_state(
        torch.load(
            source_receipt["checkpoint"],
            weights_only=True,
            map_location="cpu",
            mmap=True,
        )
    )
    model = (
        VisionTransformer.from_config("PE-Core-S16-384", pretrained=False)
        .float()
        .eval()
    )
    expected = model.state_dict()
    assert state.keys() == expected.keys() and len(expected) == 163
    assert all(
        state[n].shape == expected[n].shape and torch.isfinite(state[n]).all()
        for n in expected
    )
    model.load_state_dict(state, strict=True)
    assert smoke.digest(model.state_dict()) == source_receipt["source_state_sha256"]
    assert model.image_size == 384 and model.width == 384 and model.layers == 12
    del state, expected
    model = model.cuda()
    torch.cuda.reset_peak_memory_stats()
    processor = get_image_transform(224)

    def pixels(batch):
        images = []
        for row in batch:
            path = DATA / row["relative_path"]
            assert sha(path) == row["image_sha256"]
            with Image.open(path) as image:
                images.append(processor(image.convert("RGB")))
        return torch.stack(images)

    x = pixels(frozen["manifest"][:4])
    assert smoke.digest({"pixels": x[:2]}) == source_receipt["processor_pixels_sha256"]
    x = x.cuda()
    with torch.inference_mode():
        fp32 = model(x).float()
        with torch.autocast("cuda", dtype=torch.float16):
            fp16 = model(x).float()
        cosine = F.cosine_similarity(fp32, fp16).tolist()
    smoke.save(output / "precision.json", {"fp16_fp32_cosine": cosine})
    assert (
        min(cosine) >= 0.999
        and torch.isfinite(fp32).all()
        and torch.isfinite(fp16).all()
    )
    with (output / "references.npz").open("xb") as stream:
        np.savez(
            stream,
            fp32=F.normalize(fp32, dim=1).cpu().numpy(),
            fp16=F.normalize(fp16, dim=1).cpu().numpy(),
        )
    del x, fp32, fp16

    @torch.inference_mode()
    def encode(batch):
        with torch.autocast("cuda", dtype=torch.float16):
            raw = model(pixels(batch).cuda()).float()
        assert torch.isfinite(raw).all() and (raw.norm(dim=1) > 0).all()
        return F.normalize(raw, dim=1).cpu().numpy()

    feature_path = output / "s16.fit.npy"
    tick = time.perf_counter()
    export_features(frozen["manifest"], encode, feature_path, width=512, batch_size=32)
    export_seconds = time.perf_counter() - tick
    norm_error = check_features(feature_path, 13283)
    values = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    with np.load(output / "references.npz", allow_pickle=False) as refs:
        cache_cosines = {
            n: F.cosine_similarity(
                torch.from_numpy(values[:4].copy()), torch.from_numpy(refs[n])
            ).tolist()
            for n in ("fp32", "fp16")
        }
    peak = torch.cuda.max_memory_allocated()
    smoke.save(
        output / "cache-parity.json",
        {"cosines": cache_cosines, "peak_cuda_allocated_bytes": peak},
    )
    assert (
        min(v for a in cache_cosines.values() for v in a) >= 0.999
        and peak < 10_000_000_000
    )
    assert smoke.digest(model.state_dict()) == source_receipt["source_state_sha256"]
    assert (
        smoke.digest(model.rope.rope.state_dict())
        == source_receipt["foreign_rope_sha256"]
    )
    assert (
        smoke.digest({"freq": model.rope.freq}) == source_receipt["foreign_grid_sha256"]
    )
    startup(output / "preflight.json", expected_sha)
    elapsed = time.perf_counter() - started
    assert elapsed < 300
    smoke.save(
        output / "receipt.json",
        {
            "pass": True,
            "source_receipt_sha256": SOURCE_SHA256,
            "preflight_sha256": expected_sha,
            "features_sha256": sha(feature_path),
            "references_sha256": sha(output / "references.npz"),
            "shape": [13283, 512],
            "dtype": "float32",
            "max_norm_error_float64": norm_error,
            "fp16_fp32_cosine": cosine,
            "cache_reference_cosines": cache_cosines,
            "export_seconds": export_seconds,
            "export_images_per_second": 13283 / export_seconds,
            "whole_seconds": elapsed,
            "peak_cuda_allocated_bytes": peak,
            "source_state_unchanged": True,
            "optimizer_updates": 0,
            "held_images": 0,
            "quality_read": False,
            "public_latency_measured": False,
        },
    )
    print(
        "PASS native FP16 parity and full-fit S16 feature acquisition; no quality read",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--preflight-only", action="store_true")
    p.add_argument("--preflight-sha256")
    args = p.parse_args()
    torch.set_num_threads(8)
    torch.manual_seed(179036)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.preflight_only:
        preflight(args.output)
    else:
        acquire(args.output, args.preflight_sha256)


if __name__ == "__main__":
    main()
