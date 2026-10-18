#!/usr/bin/env python3
"""GPU-only So400m encode sweep; this is not an official retrieval-quality read.

Simulated perspectives, arguments, objections and falsifiable predictions (not
expert testimony or measurements):
* GPU systems engineer: favour fused SDPA, native low precision and larger
  batches; object to including compilation or transfers in steady-state timing.
  Predict the best SDPA/native configuration beats the matching eager/fp32-AMP
  b64 configuration by >=10% at each resolution; compilation may not help b256.
* Numerics specialist: keep the trained head and both normalisations in FP32;
  object to calling a dtype/backend change exact. Predict FP16 has smaller
  maximum cosine drift than BF16 for otherwise identical configurations, and
  at least one native/SDPA/compiled configuration changes packed bytes.
* Retrieval researcher: inspect neighbours as well as cosine errors; object to
  treating synthetic-noise agreement as evidence of SOP accuracy. Predict some
  configuration with max cosine drift <1e-3 still changes a packed top-10 set
  on real samples because of quantisation boundaries and near ties.
* Skeptical production engineer: report warmup cost, memory and failures; object
  to assuming b256 or compilation fits. Predict at least one compiled/b256 arm
  either fails or is slower than its eager-execution b64 counterpart including
  warmup for a 2048-image job. All predictions can be refuted by this report.

Reference: installed HF default attention, FP32 vision weights with FP16 CUDA
autocast, b64, no compile. Strictly restore checkpoint weights in FP32 (the
snapshot's initial FP16 weights are entirely replaced by the checkpoint).
As in evaluate_sop_so400m_res280_official.export(), interpolate positions,
normalise pooled.float(), apply the trained FP32 linear head outside autocast,
then L2-normalise. Native FP16/BF16 changes only vision weights/input dtype.
HF's attention switch applies to transformer attention; its pooling head uses
nn.MultiheadAttention independently. TF32 is disabled for matmul and cuDNN.

Time only the resident GPU tensor -> 128-D unit output, excluding preprocessing,
H2D/D2H and CPU int8 packing. Five warmups precede twenty timed iterations.
Use the SAME first 2048 inputs for every drift check. Cosine drift and float
top-10 agreement use all sample pairs; also report packed top-10 agreement with
the repository's int8 codes and FP16 inverse norms. Self is excluded; ties
prefer the lowest row index. Agreement is mean |candidate set & reference set|/10.
Synthetic inputs are deterministic uint8 noise; --images-npy accepts uint8
NHWC RGB, >=2048 rows, without decoding or fetching any images.

Run separately for --resolution 280 and 384, only when the GPU is authorised.
GB10 measured b64 baselines (context, not a controlled speedup denominator):
87.9 img/s at 280; 47.5 img/s at 384. Never infer exactness from these predictions.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoImageProcessor, SiglipVisionModel
import transformers

from sfora.joint_relational_compaction import pack_int8_unit_embeddings

SCHEMA = "sfora-sop-so400m-encode-variants-v1"
DRIFT_ROWS = 2048
WARMUP_ITERS = 5
TIMED_ITERS = 20
PRECISIONS = {"fp32_autocast": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def configurations() -> list[dict]:
    return [dict(zip(("attention", "precision", "batch_size", "compile_mode"), values))
            for values in itertools.product(("eager", "sdpa"), PRECISIONS, (64, 128, 256),
                                            (None, "default", "max-autotune-no-cudagraphs"))]


def reference_config(attention: str = "default") -> dict:
    return {"attention": attention, "precision": "fp32_autocast", "batch_size": 64, "compile_mode": None}


class Encoder(nn.Module):
    def __init__(self, vision: nn.Module, head: nn.Linear, precision: str) -> None:
        super().__init__()
        self.vision, self.head, self.precision = vision, head, precision

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        with torch.autocast(pixels.device.type, dtype=torch.float16,
                            enabled=pixels.is_cuda and self.precision == "fp32_autocast"):
            pooled = self.vision(pixel_values=pixels, interpolate_pos_encoding=True).pooler_output
        with torch.autocast(pixels.device.type, enabled=False):
            return F.normalize(self.head(F.normalize(pooled.float(), dim=1)), dim=1)


def build_encoder(config, state: dict, variant: dict, device: torch.device) -> Encoder:
    config = copy.deepcopy(config)
    config._attn_implementation = None if variant["attention"] == "default" else variant["attention"]
    vision = SiglipVisionModel(config)
    head = nn.Linear(config.hidden_size, 128)
    vision.load_state_dict(state["vision"], strict=True)
    head.load_state_dict(state["head"], strict=True)
    vision.to(device=device, dtype=PRECISIONS[variant["precision"]])
    head.to(device=device, dtype=torch.float32)
    return Encoder(vision, head, variant["precision"]).eval()


def load_samples(images_npy: Path | None, resolution: int) -> np.ndarray:
    if images_npy is None:
        return np.random.default_rng(179024).integers(
            0, 256, (DRIFT_ROWS, resolution, resolution, 3), dtype=np.uint8)
    samples = np.load(images_npy, mmap_mode="r", allow_pickle=False)
    if (samples.dtype != np.uint8 or samples.ndim != 4 or len(samples) < DRIFT_ROWS
            or samples.shape[-1] != 3 or min(samples.shape[1:3]) < 1):
        raise ValueError("--images-npy requires uint8 NHWC RGB with >=2048 rows")
    return samples[:DRIFT_ROWS]


def pixels_for(samples: np.ndarray, processor, variant: dict, device: torch.device) -> torch.Tensor:
    pixels = processor(images=list(samples), return_tensors="pt")["pixel_values"]
    return pixels.to(device=device, dtype=PRECISIONS[variant["precision"]])


@torch.inference_mode()
def encode_samples(model, samples: np.ndarray, processor, variant: dict,
                   device: torch.device) -> torch.Tensor:
    batch = variant["batch_size"]
    return torch.cat([model(pixels_for(samples[start:start + batch], processor, variant, device))
                      .float().cpu() for start in range(0, len(samples), batch)]).contiguous()


def neighbours(scores: torch.Tensor, start: int) -> torch.Tensor:
    scores = scores.clone()
    scores[torch.arange(len(scores)), torch.arange(start, start + len(scores))] = -torch.inf
    return torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10]


def set_agreement(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left[:, :, None] == right[:, None, :]).any(dim=2).float().mean())


def drift_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    if (reference.device.type != "cpu" or candidate.device.type != "cpu"
            or reference.dtype != torch.float32 or candidate.dtype != torch.float32
            or reference.ndim != 2 or reference.shape != candidate.shape
            or reference.shape[0] < 11 or reference.shape[1] != 128):
        raise ValueError("drift requires matching CPU FP32 (N>=11, 128) unit embeddings")
    # Packing validates finite unit rows and rejects zero vectors before any ranking.
    ref_packed = pack_int8_unit_embeddings(reference)
    cand_packed = pack_int8_unit_embeddings(candidate)
    ref_unit, cand_unit = F.normalize(reference, dim=1), F.normalize(candidate, dim=1)
    ref_codes, cand_codes = ref_packed.codes.float(), cand_packed.codes.float()
    ref_inv, cand_inv = ref_packed.inverse_norms.float(), cand_packed.inverse_norms.float()
    maximum, float_agreement, packed_agreement = 0.0, 0.0, 0.0
    for start in range(0, len(reference), 256):
        stop = min(start + 256, len(reference))
        ref_scores = ref_unit[start:stop] @ ref_unit.T
        cand_scores = cand_unit[start:stop] @ cand_unit.T
        maximum = max(maximum, float((ref_scores - cand_scores).abs().max()))
        float_agreement += (stop - start) * set_agreement(
            neighbours(ref_scores, start), neighbours(cand_scores, start))
        ref_scores = (ref_codes[start:stop] @ ref_codes.T) * ref_inv[start:stop, None] * ref_inv[None, :]
        cand_scores = (cand_codes[start:stop] @ cand_codes.T) * cand_inv[start:stop, None] * cand_inv[None, :]
        packed_agreement += (stop - start) * set_agreement(
            neighbours(ref_scores, start), neighbours(cand_scores, start))
    return {"rows": len(reference), "max_abs_cosine_delta": maximum,
            "byte_identical_code_fraction": float((ref_packed.codes == cand_packed.codes).all(dim=1).float().mean()),
            "top10_neighbour_set_agreement": float_agreement / len(reference),
            "packed_top10_neighbour_set_agreement": packed_agreement / len(reference)}


def make_report(environment: dict, reference: dict, results: list[dict]) -> dict:
    return {"schema": SCHEMA, "official_read": False, "environment": environment,
            "reference": reference, "results": results,
            "metrics": {"timing": "GPU-resident pixels to FP32 unit 128-D output; no transfer/packing",
                        "top10": "self excluded; stable row-index ties; mean intersection/10",
                        "peak_cuda_bytes": "max allocated bytes including warmup and drift; excludes reserved cache"}}


@torch.inference_mode()
def benchmark_variant(config, state: dict, variant: dict, samples: np.ndarray, processor,
                      reference: torch.Tensor, device: torch.device) -> dict:
    model = build_encoder(config, state, variant, device)
    effective = model.vision.config._attn_implementation
    if effective != variant["attention"]:
        raise RuntimeError(f"requested attention {variant['attention']}, got {effective}")
    if variant["compile_mode"] is not None:
        model = torch.compile(model, mode=variant["compile_mode"])
    pixels = pixels_for(samples[:variant["batch_size"]], processor, variant, device)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for _ in range(WARMUP_ITERS):
        model(pixels)
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - started
    started = time.perf_counter()
    for _ in range(TIMED_ITERS):
        model(pixels)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    candidate = encode_samples(model, samples, processor, variant, device)
    metrics = drift_metrics(reference, candidate)
    return {"config": variant, "status": "ok", "effective_attention": effective,
            "warmup_iters": WARMUP_ITERS, "warmup_seconds": warmup_seconds,
            "timed_iters": TIMED_ITERS, "timed_seconds": seconds,
            "gpu_img_per_s": TIMED_ITERS * len(pixels) / seconds,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "drift": metrics}


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("checkpoint", "model-snapshot", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(280, 384), required=True)
    parser.add_argument("--images-npy", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError("GPU benchmark requires CUDA; run the separate tiny CPU test instead")
    if not args.model_snapshot.is_dir():
        raise ValueError("--model-snapshot must be a local snapshot directory")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    config = AutoConfig.from_pretrained(args.model_snapshot, local_files_only=True).vision_config
    if (config.hidden_size, config.num_hidden_layers, config.patch_size, config.image_size) != (1152, 27, 14, 384):
        raise ValueError("snapshot must describe So400m/14, 27 layers, width 1152, native 384px")
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    processor = AutoImageProcessor.from_pretrained(args.model_snapshot, local_files_only=True, backend="torchvision")
    processor.size = {"height": args.resolution, "width": args.resolution}
    samples = load_samples(args.images_npy, args.resolution)
    model = build_encoder(config, state, reference_config(), device)
    default_attention = model.vision.config._attn_implementation
    reference_variant = reference_config(default_attention)
    with torch.inference_mode():
        reference = encode_samples(model, samples, processor, reference_variant, device)
    reference_info = {"config": reference_variant, "self_drift": drift_metrics(reference, reference)}
    del model
    gc.collect()
    torch.cuda.empty_cache()
    environment = {"torch": torch.__version__, "transformers": transformers.__version__,
                   "python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
                   "compute_capability": list(torch.cuda.get_device_capability()),
                   "allow_tf32": False, "hf_default_attention": default_attention,
                   "resolution": args.resolution, "drift_rows": len(samples),
                   "checkpoint": str(args.checkpoint), "checkpoint_sha256": sha256(args.checkpoint),
                   "checkpoint_seed": state.get("seed"), "checkpoint_updates": state.get("updates"),
                   "model_snapshot": str(args.model_snapshot),
                   "model_config_sha256": sha256(args.model_snapshot / "config.json"),
                   "preprocessor_sha256": sha256(args.model_snapshot / "preprocessor_config.json"),
                   "sample_source": str(args.images_npy) if args.images_npy else "synthetic uint8 noise seed 179024",
                   "sample_rows_sha256": hashlib.sha256(np.ascontiguousarray(samples)).hexdigest(),
                   "vision_precisions": list(PRECISIONS), "head_dtype": "float32"}
    results = []
    for variant in configurations():
        try:
            result = benchmark_variant(config, state, variant, samples, processor, reference, device)
        except Exception as error:
            result = {"config": variant, "status": "unsupported", "error": f"{type(error).__name__}: {error}"}
        finally:
            torch.compiler.reset()
            gc.collect()
            torch.cuda.empty_cache()
        results.append(result)
        print(json.dumps(result, allow_nan=False), flush=True)
    report = make_report(environment, reference_info, results)
    with args.output.open("x") as stream:
        stream.write(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
