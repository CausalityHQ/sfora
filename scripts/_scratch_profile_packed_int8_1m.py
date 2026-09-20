#!/usr/bin/env python3
"""Profile the exact packed-int8 scorer at one million resident rows."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import tempfile
import time
from pathlib import Path
from statistics import fmean

import torch

from sfora.joint_relational_compaction import PackedInt8Embeddings


def _nearest_rank(values: list[int], fraction: float) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _latency_summary(values: list[int], *, query_batch: int) -> dict[str, float | int]:
    if not values or query_batch < 1:
        raise ValueError("packed int8 profile samples differ")
    mean_ns = fmean(values)
    return {
        "mean_ns": mean_ns,
        "p50_ns": _nearest_rank(values, 0.50),
        "p99_ns": _nearest_rank(values, 0.99),
        "queries_per_second": query_batch * 1e9 / mean_ns,
    }


def _packed_random(rows: int, dimensions: int, *, seed: int) -> PackedInt8Embeddings:
    generator = torch.Generator().manual_seed(seed)
    codes = torch.randint(
        -127,
        128,
        (rows, dimensions),
        generator=generator,
        dtype=torch.int8,
    ).contiguous()
    norms = torch.linalg.vector_norm(codes.float(), dim=1)
    return PackedInt8Embeddings(codes, norms.reciprocal().to(torch.float16).contiguous())


def _timed(runner: object) -> tuple[int, torch.Tensor]:
    torch.cuda.synchronize()
    started = time.perf_counter_ns()
    output = runner()
    torch.cuda.synchronize()
    return time.perf_counter_ns() - started, output


def _peak_rss_bytes() -> int:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


def profile(
    *,
    gallery_count: int,
    dimensions: int,
    samples: int,
    warmups: int,
) -> dict[str, object]:
    if gallery_count != 1_000_000 or dimensions != 128 or samples < 10 or warmups < 1:
        raise ValueError("packed int8 profile authority differs")
    if not torch.cuda.is_available():
        raise RuntimeError("packed int8 profile requires CUDA")
    device = torch.device("cuda")
    gallery = _packed_random(gallery_count, dimensions, seed=1701)
    query_sets = {batch: _packed_random(batch, dimensions, seed=1701 + batch) for batch in (1, 32)}

    resident_codes = gallery.codes.to(device=device, dtype=torch.float32).T.contiguous()
    resident_norms = gallery.inverse_norms.to(device=device, dtype=torch.float32).contiguous()
    resident_queries = {
        batch: (
            packed.codes.to(device=device, dtype=torch.float32).contiguous(),
            packed.inverse_norms.to(device=device, dtype=torch.float32).contiguous(),
        )
        for batch, packed in query_sets.items()
    }
    resident_gpu_bytes = (
        resident_codes.numel() * resident_codes.element_size()
        + resident_norms.numel() * resident_norms.element_size()
    )
    measurements: dict[str, object] = {}
    score_equivalence: dict[str, object] = {}

    for batch, queries in query_sets.items():
        query_codes, query_norms = resident_queries[batch]

        def current(queries: PackedInt8Embeddings = queries) -> torch.Tensor:
            return torch.topk(
                queries.cosine_similarity(gallery, device=device),
                k=10,
                dim=1,
                largest=True,
                sorted=True,
            ).indices

        def resident(
            query_codes: torch.Tensor = query_codes,
            query_norms: torch.Tensor = query_norms,
        ) -> torch.Tensor:
            scores = (query_codes @ resident_codes) * query_norms[:, None] * resident_norms[None, :]
            return torch.topk(scores, k=10, dim=1, largest=True, sorted=True).indices

        for _ in range(warmups):
            current()
            resident()
        current_reference = current()
        resident_reference = resident()
        torch.cuda.synchronize()
        equal = torch.equal(current_reference, resident_reference)
        if not equal:
            raise RuntimeError("packed int8 resident top-k differs")

        elapsed = {"current_materializing": [], "resident_float32": []}
        torch.cuda.reset_peak_memory_stats()
        for _ in range(samples):
            for name, runner in (
                ("current_materializing", current),
                ("resident_float32", resident),
            ):
                duration, _ = _timed(runner)
                elapsed[name].append(duration)
        summaries = {
            name: _latency_summary(values, query_batch=batch) for name, values in elapsed.items()
        }
        current_mean = float(summaries["current_materializing"]["mean_ns"])
        resident_mean = float(summaries["resident_float32"]["mean_ns"])
        conversion_share = max(0.0, (current_mean - resident_mean) / current_mean)
        measurements[str(batch)] = {
            "arms": summaries,
            "conversion_share_of_current_mean": conversion_share,
            "kernel_eligible_over_30_percent": conversion_share > 0.30,
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "raw_samples_ns": elapsed,
        }
        score_equivalence[str(batch)] = {"top10_indices_equal": equal}

    return {
        "schema": "sfora-packed-int8-million-profile-v1",
        "claim_eligible": False,
        "gallery_count": gallery_count,
        "dimensions": dimensions,
        "top_k": 10,
        "persistent_bytes_per_item": gallery.bytes_per_vector,
        "persistent_gallery_bytes": gallery_count * gallery.bytes_per_vector,
        "resident_float32_gpu_bytes": resident_gpu_bytes,
        "measurements": measurements,
        "score_equivalence": score_equivalence,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "platform": platform.platform(),
        "samples": samples,
        "warmups": warmups,
    }


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gallery-count", type=int, default=1_000_000)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--samples", type=int, default=50)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--execute-million-profile", action="store_true", required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("packed int8 profile output already exists")
    receipt = profile(
        gallery_count=args.gallery_count,
        dimensions=args.dimensions,
        samples=args.samples,
        warmups=args.warmups,
    )
    serialized = json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False)
    payload = f"{serialized}\n".encode()
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
