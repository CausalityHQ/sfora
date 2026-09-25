#!/usr/bin/env python3
"""Paired diagnostic image-to-top-10 timing: selected SigLIP2 vs faithful UNICOM."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from probe_l14_parallel_fold import load_authenticated_l14
from torch import nn
from torch.nn import functional as F
from train_sop_siglip2_compact import make_collate, paths_from_archive

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.packed_int8_search import CpuPackedInt8Gallery
from sfora.sop_compact_training import compact_head_features

ARCHIVE_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
MANIFEST_SHA256 = "28a3ec0561cd83ee426f3d1c301c70799316af91c1e9083a5a1ffdf3414327c1"
DECISION_SHA256 = "ccfdb7055643f2a16124ecab62b30d31a2b025f380319e950cb4a481066f1852"
OFFICIAL_SHA256 = "86b5840728218364ccf1584965f019c462f224dc491deb6172e16aa8bf252665"
NATIVE_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
STAGES = ("decode_preprocess", "host_to_device", "encode", "pack", "search", "image_to_top10")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(samples: list[int]) -> dict[str, float]:
    values = np.asarray(samples, dtype=np.float64) / 1e6
    return {
        "p50_ms": float(np.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(values.mean()),
    }


@torch.inference_mode()
def call(arm: str, paths: tuple[Path, ...], state: dict) -> tuple[dict[str, int], np.ndarray]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    images = []
    for path in paths:
        with Image.open(path) as image:
            images.append(image.convert("RGB"))
    if arm == "candidate":
        batch, _ = make_collate(state["processor"])([(image, 0) for image in images])
        prepared = batch["pixel_values"]
    else:
        prepared = torch.stack([state["transform"](image) for image in images])
    preprocessed = time.perf_counter_ns()
    if arm == "candidate":
        device = {"pixel_values": prepared.cuda(non_blocking=False)}
    else:
        device = prepared.cuda(non_blocking=False)
    torch.cuda.synchronize()
    transferred = time.perf_counter_ns()
    if arm == "candidate":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            pooled = state["vision"](**device).pooler_output
        values = F.normalize(compact_head_features(pooled, state["head"]), dim=1).cpu()
    else:
        values = state["model"](device).float()
    torch.cuda.synchronize()
    encoded = time.perf_counter_ns()
    if arm == "candidate":
        query = pack_int8_unit_embeddings(values)
    else:
        query = F.normalize(values, dim=1)[:, :512].contiguous()
    torch.cuda.synchronize()
    packed = time.perf_counter_ns()
    if arm == "candidate":
        ordinals, _scores = state["gallery"].search_packed(query)
    else:
        scores = 2 * (query @ state["gallery"].T) - state["gallery_norms"][None, :]
        ordinals = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10].cpu().numpy()
    torch.cuda.synchronize()
    finished = time.perf_counter_ns()
    if ordinals.shape != (len(paths), 10):
        raise ValueError("paired live top-10 geometry differs")
    stages = (
        preprocessed - started,
        transferred - preprocessed,
        encoded - transferred,
        packed - encoded,
        finished - packed,
        finished - started,
    )
    return dict(zip(STAGES, stages, strict=True)), ordinals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "archive",
        "manifest",
        "dataset-root",
        "decision",
        "training-receipt",
        "checkpoint",
        "official-receipt",
        "official-embeddings",
        "model-snapshot",
        "unicom-checkout",
        "unicom-checkpoint",
        "native-library",
        "output",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--blocks", type=int, default=10)
    parser.add_argument("--calls-per-block", type=int, default=10)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.blocks != 10
        or args.calls_per_block != 10
        or not torch.cuda.is_available()
    ):
        raise ValueError("paired live timing invocation differs")
    for path, digest in (
        (args.archive, ARCHIVE_SHA256),
        (args.manifest, MANIFEST_SHA256),
        (args.decision, DECISION_SHA256),
        (args.official_receipt, OFFICIAL_SHA256),
        (args.native_library, NATIVE_SHA256),
    ):
        if sha256(path) != digest:
            raise ValueError(f"paired live source differs: {path}")
    decision = json.loads(args.decision.read_text())
    trained = json.loads(args.training_receipt.read_text())
    official = json.loads(args.official_receipt.read_text())
    if (
        decision.get("replication_gate_pass") is not True
        or sha256(args.training_receipt) != decision["arms"]["179023"]["bank"]["receipt_sha256"]
        or trained.get("seed") != 179023
        or trained.get("arm") != "float_rank_member_bank"
        or sha256(args.checkpoint) != trained.get("checkpoint_sha256")
        or official.get("training_receipt_sha256") != sha256(args.training_receipt)
        or sha256(args.official_embeddings) != official.get("test_embeddings_sha256")
        or official.get("native_top10_exact") is not True
        or official.get("source_archive_sha256") != ARCHIVE_SHA256
        or official.get("test_image_manifest_sha256") != MANIFEST_SHA256
    ):
        raise ValueError("selected checkpoint or official embedding authority differs")
    with np.load(args.archive, allow_pickle=False) as archive:
        reference = np.asarray(archive["test_embeddings"], dtype=np.float32)
        relatives = np.asarray(archive["test_relative_paths"])
        ids = np.asarray(archive["test_image_ids"], dtype=np.int64)
    candidate = np.load(args.official_embeddings, mmap_mode="r")
    if (
        reference.shape != (60_502, 768)
        or candidate.shape != (60_502, 128)
        or ids.shape != (60_502,)
    ):
        raise ValueError("paired live gallery inventory differs")
    paths = paths_from_archive(args.dataset_root, relatives[:32])
    manifest = args.manifest.read_bytes()
    if len(manifest) != 60_502 * 32 or any(
        hashlib.sha256(path.read_bytes()).digest() != manifest[32 * index : 32 * (index + 1)]
        for index, path in enumerate(paths)
    ):
        raise ValueError("paired live query image bytes differ")
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    import transformers

    processor = transformers.AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = transformers.AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    head = nn.Linear(1024, 128).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if checkpoint.get("seed") != 179023 or checkpoint.get("arm") != "float_rank_member_bank":
        raise ValueError("paired live checkpoint differs")
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    reference_model, transform = load_authenticated_l14(
        args.unicom_checkout, args.unicom_checkpoint
    )
    reference_model = reference_model.cuda().eval()
    candidate_gallery = pack_int8_unit_embeddings(
        torch.from_numpy(np.asarray(candidate[32:]).copy())
    )
    reference_gallery = F.normalize(torch.from_numpy(reference[32:].copy()).cuda(), dim=1)[
        :, :512
    ].contiguous()
    state = {
        "candidate": {"processor": processor, "vision": vision, "head": head},
        "reference": {
            "model": reference_model,
            "transform": transform,
            "gallery": reference_gallery,
            "gallery_norms": (reference_gallery * reference_gallery).sum(dim=1),
        },
    }
    with ExitStack() as stack:
        state["candidate"]["gallery"] = stack.enter_context(
            CutilePackedInt8Gallery.open_packed(args.native_library, candidate_gallery)
        )
        parity = {}
        for arm in ("candidate", "reference"):
            _timing, ordinals = call(arm, paths, state[arm])
            if arm == "candidate":
                cached_query = pack_int8_unit_embeddings(
                    torch.from_numpy(np.asarray(candidate[:32]).copy())
                )
                cached_ordinals = state[arm]["gallery"].search_packed(cached_query)[0]
                scalar_ordinals = CpuPackedInt8Gallery.open_packed(candidate_gallery).search_packed(
                    cached_query
                )[0]
                padded_ordinals = call(arm, paths + paths, state[arm])[1][:32]
                if not np.array_equal(ordinals, cached_ordinals):
                    print(
                        json.dumps(
                            {
                                "candidate_live_cached_top10_equal_rows": int(
                                    np.all(ordinals == cached_ordinals, axis=1).sum()
                                ),
                                "candidate_live_cached_top1_equal_rows": int(
                                    (ordinals[:, 0] == cached_ordinals[:, 0]).sum()
                                ),
                                "candidate_cached_native_scalar_equal": bool(
                                    np.array_equal(cached_ordinals, scalar_ordinals)
                                ),
                                "candidate_padded64_cached_top10_equal_rows": int(
                                    np.all(padded_ordinals == cached_ordinals, axis=1).sum()
                                ),
                                "candidate_live_first_top10": ordinals[0].tolist(),
                                "candidate_cached_first_top10": cached_ordinals[0].tolist(),
                            }
                        ),
                        flush=True,
                    )
                if (
                    not np.array_equal(cached_ordinals, scalar_ordinals)
                    or not np.array_equal(padded_ordinals, cached_ordinals)
                    or not np.array_equal(ordinals[:, 0], cached_ordinals[:, 0])
                ):
                    raise ValueError("selected live query differs from scored packed path")
            else:
                query = F.normalize(torch.from_numpy(reference[:32].copy()).cuda(), dim=1)[:, :512]
                scores = 2 * query @ reference_gallery.T - state[arm]["gallery_norms"][None, :]
                cached_ordinals = (
                    torch.argsort(scores, dim=1, descending=True, stable=True)[:, :10].cpu().numpy()
                )
                if not np.array_equal(ordinals, cached_ordinals):
                    raise ValueError("reference live query differs from cached prefix scorer")
            parity[arm] = hashlib.sha256(ordinals.tobytes()).hexdigest()
        raw = {}
        for batch in (1, 32):
            selected = paths[:batch]
            for arm in ("candidate", "reference"):
                for _ in range(3):
                    call(arm, selected, state[arm])
            samples = {arm: {stage: [] for stage in STAGES} for arm in ("candidate", "reference")}
            hashes = {}
            for block in range(args.blocks):
                order = ("candidate", "reference") if block % 2 == 0 else ("reference", "candidate")
                for arm in order:
                    for _ in range(args.calls_per_block):
                        durations, ordinals = call(arm, selected, state[arm])
                        digest = hashlib.sha256(ordinals.tobytes()).hexdigest()
                        if arm in hashes and hashes[arm] != digest:
                            raise ValueError("paired live result changed during timing")
                        hashes[arm] = digest
                        for stage, value in durations.items():
                            samples[arm][stage].append(value)
                print(json.dumps({"batch": batch, "completed_blocks": block + 1}), flush=True)
            raw[str(batch)] = {
                arm: {
                    "samples_ns": samples[arm],
                    "summary": {stage: metrics(values) for stage, values in samples[arm].items()},
                    "result_sha256": hashes[arm],
                }
                for arm in samples
            }
    if any(
        hashlib.sha256(path.read_bytes()).digest() != manifest[32 * index : 32 * (index + 1)]
        for index, path in enumerate(paths)
    ):
        raise ValueError("paired live query images changed")
    receipt = {
        "schema": "sfora-sop-coverage-vs-fullwidth-unicom-diagnostic-v1",
        "claim_eligible": False,
        "p99_certified": False,
        "split": (
            "already-observed SOP official TEST, first 32 image queries, "
            "shared gallery rows 32:60502, ten nonself results"
        ),
        "gallery_rows": 60_470,
        "candidate_gallery_wire_bytes": 60_470 * 130,
        "reference_gallery_fp32_bytes": 60_470 * 512 * 4,
        "candidate_precision": (
            "fp32 weights; fp16 autocast vision; fp32 head; int8 packed exact search"
        ),
        "reference_scorer": (
            "normalize full 768-D; prefix 512 without renormalization; "
            "2*q*g - squared gallery norm; stable ordinal ties"
        ),
        "query_image_ids": ids[:32].tolist(),
        "live_cached_top10_parity_sha256": parity,
        "raw": raw,
        "inputs": {
            "archive_sha256": ARCHIVE_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "decision_sha256": DECISION_SHA256,
            "training_receipt_sha256": sha256(args.training_receipt),
            "checkpoint_sha256": sha256(args.checkpoint),
            "official_receipt_sha256": OFFICIAL_SHA256,
            "official_embeddings_sha256": sha256(args.official_embeddings),
            "unicom_checkpoint_sha256": sha256(args.unicom_checkpoint),
            "native_sha256": NATIVE_SHA256,
            "script_sha256": sha256(Path(__file__)),
        },
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                batch: {arm: raw[batch][arm]["summary"]["image_to_top10"] for arm in raw[batch]}
                for batch in raw
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
