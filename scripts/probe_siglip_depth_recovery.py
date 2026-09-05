#!/usr/bin/env python3
"""Authenticated optimization-image-only speed preflight for fixed SigLIP surgery."""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import platform
import resource
import stat
import sys
from collections.abc import Callable, Mapping
from functools import partial
from pathlib import Path
from time import perf_counter_ns
from typing import Any

import torch
from PIL import Image

from sfora.data import materialize_image
from sfora.siglip_depth_recovery import RETAINED_BLOCKS, prune_siglip_student, speed_gate
from sfora.siglip_proxy_control import PooledProxyAnchorModel, SiglipProxyControlConfig

_SCRIPTS = str(Path(__file__).resolve().parent)
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)
import run_siglip_proxy_control as control  # noqa: E402

SEED_RECEIPT_SHA256 = "30d990f09bc3ce3c83c514c724aa3c2b2fd0c6fcceea01db97a557e3257e72ba"
CHECKPOINT_SHA256 = "cb9c768fbb254bb164432ac92f756ca588cb1f33ac3eea86d4057d075ce2ef6e"
CHECKPOINT_BYTES = 5146653305
MANIFEST_SHA256 = "6c053b820202fb5deccfba06360e8506f201ce9eedbc6569384abd8fc30004ac"


def _canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _file_sha(path: Path) -> str:
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("preflight input must be a regular file")
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def authenticate_control(root: Path) -> tuple[dict[str, Any], Path]:
    """Authenticate the fixed immutable seed receipt and tensor bytes before parsing."""
    receipt_path = root / "seed-017.receipt.json"
    if _file_sha(receipt_path) != SEED_RECEIPT_SHA256:
        raise ValueError("seed17 receipt SHA differs")
    receipt = json.loads(receipt_path.read_bytes())
    config = SiglipProxyControlConfig()
    if (
        receipt["schema"] != "sfora-siglip-proxy-control-seed-v1"
        or receipt["claim_eligible"] is not False
        or type(receipt["seed"]) is not int
        or receipt["seed"] != 17
        or receipt["config_sha256"] != control._config_sha256(config)
        or receipt["dataset"]["manifest_sha256"] != MANIFEST_SHA256
    ):
        raise ValueError("seed17 control authority differs")
    ref = receipt["checkpoint"]
    if (
        ref["basename"] != "seed-017-epoch-060.pt"
        or ref["sha256"] != CHECKPOINT_SHA256
        or type(ref["bytes"]) is not int
        or ref["bytes"] != CHECKPOINT_BYTES
        or type(ref["epoch"]) is not int
        or ref["epoch"] != 60
    ):
        raise ValueError("seed17 checkpoint reference differs")
    path = root / "seed-017/checkpoints/seed-017-epoch-060.pt"
    if path.stat().st_size != CHECKPOINT_BYTES or _file_sha(path) != CHECKPOINT_SHA256:
        raise ValueError("seed17 checkpoint bytes differ")
    return receipt, path


def select_speed_images(
    bands: control.ControlExampleBands,
) -> tuple[tuple[Image.Image, ...], list[dict[str, Any]], str]:
    """Take only the first128 sorted optimization images, validating full ID authority."""
    if (
        control.control_manifest_sha256(bands.ordered_manifest) != MANIFEST_SHA256
        or len(bands.optimization) != 3963
        or len(bands.clean_validation) != 2746
        or len(bands.burned_diagnostic) != 1345
    ):
        raise ValueError("speed preflight population authority differs")
    selected = tuple(sorted(bands.optimization, key=lambda e: e.example_id))[:128]
    if any(type(e.label) is not int or not 0 <= e.label < 49 for e in selected):
        raise ValueError("speed preflight must use optimization classes only")
    rows, images = [], []
    digest = hashlib.sha256(b"sfora-depth-speed-original-rgb-v1\0")
    for example in selected:
        image = materialize_image(example.image)
        if not isinstance(image, Image.Image):
            raise ValueError("speed source is not a PIL image")
        rgb = image.convert("RGB")
        rows.append({"example_id": example.example_id, "label": example.label})
        digest.update(_canonical({"example_id": example.example_id, "size": list(rgb.size)}))
        digest.update(rgb.tobytes())
        images.append(rgb)
    return tuple(images), rows, digest.hexdigest()


def measure_pair(
    forwards: dict[str, Callable[[Any], torch.Tensor]],
    bank: Any,
    *,
    window: int,
    synchronize: Callable[[], None],
    clock: Callable[[], int] = perf_counter_ns,
    progress: Callable[[int], None] | None = None,
) -> dict[str, list[int]]:
    """Alternate paired full/student calls within rounds; retain100 postwarmup samples."""
    if set(forwards) != {"full", "student"} or len(bank) != 128 or window not in (0, 1, 2):
        raise ValueError("speed pair inventory differs")
    samples: dict[str, list[int]] = {"full": [], "student": []}
    with torch.no_grad():
        for index in range(110):
            start = (index * 8) % 121
            batch = bank[start : start + 8]
            order = ("full", "student") if (index + window) % 2 == 0 else ("student", "full")
            for name in order:
                synchronize()
                began = clock()
                output = forwards[name](batch)
                synchronize()
                elapsed = clock() - began
                if not isinstance(output, torch.Tensor) or not bool(torch.isfinite(output).all()):
                    raise ValueError("speed output is nonfinite")
                if type(elapsed) is not int or elapsed <= 0:
                    raise ValueError("speed clock failed to advance")
                if torch.cuda.is_available() and torch.cuda.max_memory_reserved() >= 96 * 1024**3:
                    raise RuntimeError("depth preflight CUDA memory limit")
                if index >= 10:
                    samples[name].append(elapsed)
                del output
            if progress is not None and (index + 1) % 10 == 0:
                progress(index + 1)
    return samples


def _release_cpu_pages() -> None:
    gc.collect()
    if sys.platform == "linux":
        trim = getattr(ctypes.CDLL(None), "malloc_trim", None)
        if trim is not None:
            trim.argtypes, trim.restype = [ctypes.c_size_t], ctypes.c_int
            trim(0)


def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    """Load one authenticated incumbent and its independent student; never train."""
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("the scientific speed preflight requires CUDA")
    control.require_control_determinism(torch.device("cuda"))
    import transformers

    if transformers.__version__ != "5.12.1":
        raise RuntimeError("depth preflight Transformers version differs from5.12.1")
    began = perf_counter_ns()
    receipt, checkpoint = authenticate_control(args.control_root)
    print("depth-speed: fixed checkpoint authenticated", flush=True)
    bands = control.load_control_examples()
    images, rows, pixels_sha = select_speed_images(bands)
    del bands
    _release_cpu_pages()
    print(
        "depth-speed:128 optimization RGB images selected; unused source pixels released",
        flush=True,
    )
    config = SiglipProxyControlConfig()
    tower, processor = control.load_siglip_control_components(config=config)
    full = PooledProxyAnchorModel(
        tower=tower, input_dimensions=1152, embedding_dimensions=512, class_count=49
    ).float()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    if (
        payload.get("schema") != "sfora-siglip-proxy-checkpoint-payload-v1"
        or payload.get("claim_eligible") is not False
        or type(payload.get("seed")) is not int
        or payload["seed"] != 17
        or type(payload.get("completed_epoch")) is not int
        or payload["completed_epoch"] != 60
        or payload.get("config_sha256") != receipt["config_sha256"]
        or not isinstance(payload.get("model_state"), Mapping)
    ):
        raise ValueError("depth preflight inference payload differs")
    full.load_state_dict(payload["model_state"], strict=True)
    del payload, tower
    full = full.to("cuda").eval().requires_grad_(False)
    student = prune_siglip_student(full)
    models = {"full": full, "student": student}
    _release_cpu_pages()
    print("depth-speed: full27/student18 resident; no optimization or evaluation", flush=True)
    pixels = control.preprocess_control_evaluation(processor, list(images)).to("cuda")

    def pipeline(model: PooledProxyAnchorModel, batch: Any) -> torch.Tensor:
        inputs = control.preprocess_control_evaluation(processor, list(batch)).to("cuda")
        return model.encode(inputs)

    windows = []
    for window in range(3):
        scopes = {}
        for scope in ("pipeline", "encoder"):
            forwards: dict[str, Callable[[Any], torch.Tensor]] = {
                name: partial(pipeline, model) if scope == "pipeline" else model.tower
                for name, model in models.items()
            }

            def progress(n: int, w: int = window, s: str = scope) -> None:
                print(f"depth-speed:window={w} scope={s} rounds={n}/110", flush=True)

            scopes[scope] = measure_pair(
                forwards,
                images if scope == "pipeline" else pixels,
                window=window,
                synchronize=torch.cuda.synchronize,
                progress=progress,
            )
        windows.append(scopes)
    core = Path(__file__).resolve().parents[1] / "src/sfora/siglip_depth_recovery.py"
    return {
        "schema": "sfora-siglip-depth-speed-preflight-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "optimization_steps": 0,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "seed_receipt_sha256": SEED_RECEIPT_SHA256,
        "seed": 17,
        "dataset_manifest_sha256": MANIFEST_SHA256,
        "input_rows": rows,
        "original_rgb_sha256": pixels_sha,
        "retained_one_indexed_blocks": list(RETAINED_BLOCKS),
        "source_sha256": {
            str(path.relative_to(core.parents[2])): _file_sha(path)
            for path in (
                Path(__file__).resolve(),
                core,
                Path(control.__file__),
                core.parent / "siglip_proxy_control.py",
                core.parent / "data.py",
            )
        },
        "model": {
            "name": config.model_name,
            "revision": config.model_revision,
            "input_resolution": 384,
            "descriptor_dimensions": 512,
        },
        "parameter_counts": {
            name: sum(p.numel() for p in model.parameters()) for name, model in models.items()
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
            "torch_threads": torch.get_num_threads(),
        },
        "timing": {
            "batch_size": 8,
            "warmups_per_window": 10,
            "samples_per_window": 100,
            "pair_order": "reverse-on-odd-round-plus-window",
            "pipeline_scope": (
                "resident-original-RGB,processor-resize,transfer,tower,projection,normalize"
            ),
            "encoder_scope": "resident-device-pixels,tower-with-pooler",
            "windows": windows,
        },
        "speed_passed": speed_gate(windows),
        "resources": {
            "elapsed_ns": perf_counter_ns() - began,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(),
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-speed-preflight", action="store_true", required=True)
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    result = run_probe(args)
    raw = _canonical(result)
    control._write_new(args.output, raw)
    print(
        f"depth-speed:COMPLETE speed_passed={result['speed_passed']} "
        f"sha256={hashlib.sha256(raw).hexdigest()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
