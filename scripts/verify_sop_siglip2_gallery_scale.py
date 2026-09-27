#!/usr/bin/env python3
"""TRAIN-only full SOP gallery build, exact top-10 parity, and public latency."""

from __future__ import annotations

import hashlib
import json
import resource
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image
from train_sop_siglip2_compact import paths_from_archive

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.siglip2_compact_serving import Siglip2CompactEncoder, Siglip2CompactIndex

RUN = Path("/home/riomus/runs/sfora-sop-public-scale-train-v3")
TRAINING = Path("/home/riomus/runs/sfora-sop-true-freeze-freeze-179024-1000-v1")
ARCHIVE = Path("/home/riomus/sfora-relational-sop-e1/unicom-l14-sop-v1.npz")
DATA = Path("/home/riomus/datasets/Stanford_Online_Products")
MODEL = Path(
    "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/"
    "snapshots/787800c8990e6f058423089178e718139608408c"
)
LIBRARY = Path(
    "/home/riomus/sfora-rc5-pointer-b7c57022/libsfora_cutile_int8_score_sha39602d0e.so"
)
RECEIPT_SHA = "07b4716b42d1291b9c195774ebd48d9df89a3b578ee54efdb94662c5e125d1c5"
CHECKPOINT_SHA = "2c838561b6c23242d74eb29329fd026cc8fba9bf965dcc4529348028dfe6d172"
ARCHIVE_SHA = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
LIBRARY_SHA = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
SERVING_SHA = "5fcf261053136c916e0fe6c51119036b8114ea3f599d69c1080d7225d3ebe73a"


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    output = RUN / "receipt.json"
    if (
        output.exists()
        or not torch.cuda.is_available()
        or sha(TRAINING / "receipt.json") != RECEIPT_SHA
        or sha(TRAINING / "checkpoint.pt") != CHECKPOINT_SHA
        or sha(ARCHIVE) != ARCHIVE_SHA
        or sha(LIBRARY) != LIBRARY_SHA
        or sha(Path(__import__("sfora.siglip2_compact_serving", fromlist=["x"]).__file__))
        != SERVING_SHA
    ):
        raise ValueError("SOP public scale authority differs")
    training = json.loads((TRAINING / "receipt.json").read_text())
    if training.get("checkpoint_sha256") != CHECKPOINT_SHA or training.get("seed") != 179024:
        raise ValueError("SOP public scale training source differs")
    with np.load(ARCHIVE, allow_pickle=False) as archive:
        paths = paths_from_archive(DATA, np.asarray(archive["train_relative_paths"]))
    if len(paths) != 59_551:
        raise ValueError("SOP TRAIN gallery inventory differs")
    gallery_digest = hashlib.sha256()
    unique = []
    seen = set()
    for path in paths:
        digest = sha(path)
        gallery_digest.update(bytes.fromhex(digest))
        if digest not in seen and len(unique) < 1_000:
            seen.add(digest)
            unique.append(path)
    if len(unique) != 1_000:
        raise ValueError("SOP latency image byte inventory differs")
    torch.set_num_threads(20)
    torch.backends.cuda.matmul.allow_tf32 = False
    load_started = time.perf_counter()
    encoder = Siglip2CompactEncoder.from_checkpoint(
        model_snapshot=MODEL,
        checkpoint=TRAINING / "checkpoint.pt",
        expected_checkpoint_sha256=CHECKPOINT_SHA,
        model_file_sha256=training["model_file_sha256"],
        precision="fp16_native",
        device=torch.device("cuda:0"),
    )
    torch.cuda.synchronize()
    load_wall = time.perf_counter() - load_started
    torch.cuda.reset_peak_memory_stats()
    captured = []
    original = CutilePackedInt8Gallery.open_packed

    def capture(library: Path, packed: object):
        captured.append(packed)
        return original(library, packed)

    build_started = time.perf_counter()
    with patch.object(CutilePackedInt8Gallery, "open_packed", staticmethod(capture)):
        index = Siglip2CompactIndex.from_image_paths(
            encoder=encoder,
            native_library=LIBRARY,
            image_paths=paths,
            expected_native_library_sha256=LIBRARY_SHA,
        )
    with index:
        torch.cuda.synchronize()
        build_wall = time.perf_counter() - build_started
        if len(captured) != 1 or index.gallery is None or index.encoder is None:
            raise ValueError("SOP public gallery did not open")
        packed = captured[0]
        with ExitStack() as stack:
            images = [stack.enter_context(Image.open(path)) for path in unique[:32]]
            queries = encoder.encode_images(images)
            rows, scores = index.search_images(images)
        code = packed.codes.float().cuda()
        inverse = packed.inverse_norms.float().cuda()
        query_code = queries.codes.float().cuda()
        query_inverse = queries.inverse_norms.float().cuda()
        oracle_scores = (query_code @ code.T) * query_inverse[:, None] * inverse[None, :]
        oracle_rows = torch.argsort(oracle_scores, dim=1, descending=True, stable=True)[:, :10]
        oracle_values = oracle_scores.gather(1, oracle_rows).cpu().numpy()
        parity = bool(np.array_equal(rows, oracle_rows.cpu().numpy()))
        maximum_score_error = float(np.max(np.abs(scores - oracle_values)))
        for path in unique[:32]:
            with Image.open(path) as image:
                index.search_images([image])
        torch.cuda.synchronize()
        latency_ns = []
        for path in unique:
            torch.cuda.synchronize()
            started = time.perf_counter_ns()
            with Image.open(path) as image:
                index.search_images([image])
            torch.cuda.synchronize()
            latency_ns.append(time.perf_counter_ns() - started)
        throughput_started = time.perf_counter()
        for start in range(0, len(unique), 32):
            with ExitStack() as stack:
                images = [
                    stack.enter_context(Image.open(path)) for path in unique[start : start + 32]
                ]
                index.search_images(images)
        torch.cuda.synchronize()
        throughput_wall = time.perf_counter() - throughput_started
        peak_cuda = torch.cuda.max_memory_allocated()
    ms = np.asarray(latency_ns, dtype=np.float64) / 1e6
    percentiles = {f"p{p}_ms": float(np.percentile(ms, p)) for p in (50, 95, 99)}
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    gates = {
        "top10_exact": parity and maximum_score_error <= 1e-5,
        "build": build_wall <= 600,
        "batch1_p99": percentiles["p99_ms"] <= 30,
        "peak_cuda": peak_cuda < 3_000_000_000,
        "peak_rss": peak_rss < 6_000_000_000,
    }
    result = {
        "schema": "sfora-sop-public-custom-gallery-train-scale-v1",
        "claim_eligible": False,
        "source_sha256": sha(Path(__file__)),
        "serving_sha256": SERVING_SHA,
        "training_receipt_sha256": RECEIPT_SHA,
        "checkpoint_sha256": CHECKPOINT_SHA,
        "archive_sha256": ARCHIVE_SHA,
        "native_library_sha256": LIBRARY_SHA,
        "gallery_image_digests_sha256": gallery_digest.hexdigest(),
        "gallery_rows": len(paths),
        "gallery_wire_bytes": 130 * len(paths),
        "query_image_hashes": [sha(path) for path in unique],
        "oracle_rows": oracle_rows.cpu().tolist(),
        "public_rows": rows.tolist(),
        "maximum_score_abs_error": maximum_score_error,
        "model_load_wall_seconds": load_wall,
        "gallery_build_wall_seconds": build_wall,
        "batch1_latency_ns": latency_ns,
        "batch1_percentiles": percentiles,
        "batch32_images_per_second": len(unique) / throughput_wall,
        "peak_cuda_allocated_bytes": peak_cuda,
        "peak_parent_host_rss_bytes": peak_rss,
        "gates": gates,
        "advance": all(gates.values()),
        "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
    }
    output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(
        json.dumps({"gates": gates, "build_s": build_wall, "p99_ms": percentiles["p99_ms"]}),
        flush=True,
    )
    if not result["advance"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
