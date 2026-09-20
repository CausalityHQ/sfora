#!/usr/bin/env python3
"""Benchmark the persistent packed-int8 cuTile device top-k path."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import tempfile
import time
from pathlib import Path
from statistics import fmean

import numpy as np
import torch

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings

_DIMENSIONS = 128
_TOP_K = 10
_GALLERY_ROWS = 1_000_000
_EXACTNESS_ROWS = 1_000_003
_WARMUPS = 5
_SAMPLES = 50
_PERSISTENT_BYTES_PER_ITEM = 130
_TEMPORARY_BYTES_BATCH_32 = 32 * math.ceil(_GALLERY_ROWS / 128) * 16 * 8
_BASELINE_SHA256 = "c7b1403de61a3fe457fa92eeab08f814fbfcc5cd60a1d4f1fe241c6eb4374fb4"
_BASELINES = {
    "1": {"current_materializing_p99_ns": 20_048_894, "resident_f32_p99_ns": 2_596_715},
    "32": {"current_materializing_p99_ns": 25_635_013, "resident_f32_p99_ns": 8_665_042},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _nearest_rank(values: list[int], fraction: float) -> int:
    if len(values) != _SAMPLES or any(value <= 0 for value in values):
        raise ValueError("cuTile top-k timing authority differs")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _summary(values: list[int], *, batch: int) -> dict[str, object]:
    mean_ns = fmean(values)
    return {
        "mean_ns": mean_ns,
        "p50_ns": _nearest_rank(values, 0.50),
        "p99_ns": _nearest_rank(values, 0.99),
        "queries_per_second": batch * 1e9 / mean_ns,
        "raw_ns": values,
        "sample_count": len(values),
    }


def _decision(batches: dict[str, dict[str, object]], *, exact: bool, peak_rss: int) -> bool:
    if set(batches) != {"1", "32"} or not exact or peak_rss >= 2 * 1024**3:
        return False
    for batch, evidence in batches.items():
        p99 = int(evidence["native"]["p99_ns"])  # type: ignore[index]
        baseline = _BASELINES[batch]
        if not (
            p99 * 5 < baseline["current_materializing_p99_ns"] * 4
            and p99 * 5 < baseline["resident_f32_p99_ns"] * 4
        ):
            return False
    return True


def _packed_random(rows: int, *, seed: int) -> PackedInt8Embeddings:
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(-127, 128, (rows, _DIMENSIONS), generator=generator, dtype=torch.int8)
    norms = torch.linalg.vector_norm(codes.float(), dim=1)
    return PackedInt8Embeddings(codes.contiguous(), norms.reciprocal().half().contiguous())


def _numpy(packed: PackedInt8Embeddings) -> tuple[np.ndarray, np.ndarray]:
    return packed.codes.numpy(), packed.inverse_norms.numpy()


def _exactness(library: Path) -> dict[str, object]:
    gallery = _packed_random(_EXACTNESS_ROWS, seed=1701)
    gallery_codes, gallery_norms = _numpy(gallery)
    device = torch.device("cuda")
    resident_codes = gallery.codes.to(device=device, dtype=torch.float32).T.contiguous()
    resident_norms = gallery.inverse_norms.to(device=device, dtype=torch.float32)
    batches: dict[str, object] = {}
    with CutilePackedInt8Gallery.open(library, gallery_codes, gallery_norms) as native:
        for batch in (1, 32):
            queries = _packed_random(batch, seed=1701 + batch)
            query_codes, query_norms = _numpy(queries)
            ordinals, observed_scores = native.search(query_codes, query_norms, k=_TOP_K)
            q_codes = queries.codes.to(device=device, dtype=torch.float32)
            q_norms = queries.inverse_norms.to(device=device, dtype=torch.float32)
            dense = (q_codes @ resident_codes) * q_norms[:, None] * resident_norms[None, :]
            expected = torch.argsort(dense, dim=1, descending=True, stable=True)[:, :_TOP_K]
            expected_ordinals = expected.cpu().numpy().astype(np.int64, copy=False)
            exact_ordinals = bool(np.array_equal(ordinals, expected_ordinals))
            selected_gallery = gallery_codes[ordinals]
            dots = np.sum(query_codes[:, None, :].astype(np.int32) * selected_gallery, axis=2)
            scalar_scores = dots.astype(np.float32) * query_norms.astype(np.float32)[:, None]
            scalar_scores *= gallery_norms[ordinals].astype(np.float32)
            exact_scores = bool(
                np.array_equal(observed_scores.view(np.uint32), scalar_scores.view(np.uint32))
            )
            batches[str(batch)] = {
                "exact_ordinals": exact_ordinals,
                "exact_score_bits": exact_scores,
            }
            del dense, expected, q_codes, q_norms
    return {"gallery_rows": _EXACTNESS_ROWS, "batches": batches}


def benchmark(library: Path, source_commit: str) -> dict[str, object]:
    if not library.is_absolute() or not library.is_file() or len(source_commit) != 40:
        raise ValueError("cuTile top-k benchmark authority differs")
    gallery = _packed_random(_GALLERY_ROWS, seed=1701)
    gallery_codes, gallery_norms = _numpy(gallery)
    batches: dict[str, dict[str, object]] = {}
    with CutilePackedInt8Gallery.open(library, gallery_codes, gallery_norms) as native:
        for batch in (1, 32):
            queries = _packed_random(batch, seed=1701 + batch)
            query_codes, query_norms = _numpy(queries)
            started = time.perf_counter_ns()
            first = native.search(query_codes, query_norms, k=_TOP_K)
            compile_ns = time.perf_counter_ns() - started
            if first[0].shape != (batch, _TOP_K):
                raise RuntimeError("cuTile top-k result shape differs")
            for _ in range(_WARMUPS):
                native.search(query_codes, query_norms, k=_TOP_K)
            samples: list[int] = []
            for _ in range(_SAMPLES):
                started = time.perf_counter_ns()
                native.search(query_codes, query_norms, k=_TOP_K)
                samples.append(time.perf_counter_ns() - started)
            batches[str(batch)] = {
                "compile_ns": compile_ns,
                "native": _summary(samples, batch=batch),
                "frozen_baseline": _BASELINES[str(batch)],
            }
    del gallery, gallery_codes, gallery_norms
    exactness = _exactness(library)
    exact = all(
        bool(item["exact_ordinals"]) and bool(item["exact_score_bits"])
        for item in exactness["batches"].values()  # type: ignore[union-attr]
    )
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    advance = _decision(batches, exact=exact, peak_rss=peak_rss)
    return {
        "schema": "sfora-packed-int8-cutile-topk-v1",
        "claim_eligible": False,
        "advance": advance,
        "source_commit": source_commit,
        "library_sha256": _sha256(library),
        "gallery_rows": _GALLERY_ROWS,
        "dimensions": _DIMENSIONS,
        "top_k": _TOP_K,
        "warmups": _WARMUPS,
        "samples": _SAMPLES,
        "batches": batches,
        "exactness": exactness,
        "persistent_bytes_per_item": _PERSISTENT_BYTES_PER_ITEM,
        "persistent_gallery_bytes": _GALLERY_ROWS * _PERSISTENT_BYTES_PER_ITEM,
        "maximum_temporary_device_bytes": _TEMPORARY_BYTES_BATCH_32,
        "process_peak_rss_bytes": peak_rss,
        "baseline_summary_sha256": _BASELINE_SHA256,
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "platform": platform.platform(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-million-topk", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("cuTile top-k output already exists")
    receipt = benchmark(args.library.resolve(), args.source_commit)
    payload = (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    print(payload.decode(), end="")


if __name__ == "__main__":
    main()
