#!/usr/bin/env python3
"""Benchmark packed-int4 CPU execution against honest resident controls."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

import sfora.packed_int4 as packed_int4
from sfora.packed_int4 import ResidentInt4Gallery, pack_int4_unit_embeddings


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _cpu_model() -> str:
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        identifiers: dict[str, str] = {}
        for line in cpuinfo.read_text(errors="replace").splitlines():
            key, separator, value = line.partition(":")
            if (
                separator
                and key.strip() in {"model name", "Hardware", "Processor"}
                and value.strip()
            ):
                return value.strip()
            if separator and key.strip() in {"CPU implementer", "CPU part"} and value.strip():
                identifiers[key.strip()] = value.strip()
        if identifiers:
            return "; ".join(f"{key} {identifiers[key]}" for key in sorted(identifiers))
    return platform.processor() or platform.machine()


def _process_peak_rss_bytes() -> int:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    raise RuntimeError("Linux process peak RSS authority is unavailable")


def _percentile(samples: list[int], fraction: float) -> int:
    ordered = sorted(samples)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _arm(samples: list[int]) -> dict[str, Any]:
    return {
        "p50_ns": _percentile(samples, 0.50),
        "p95_ns": _percentile(samples, 0.95),
        "p99_ns": _percentile(samples, 0.99),
        "samples_ns": samples,
    }


def benchmark_resident_int4(
    *,
    gallery_count: int,
    dimensions: int,
    query_count: int,
    samples: int,
    warmups: int,
    threads: int,
    seed: int,
) -> dict[str, Any]:
    """Run three CPU arms and return a self-describing claim-ineligible receipt."""

    values = (gallery_count, dimensions, query_count, samples, warmups, threads, seed)
    if (
        any(type(value) is not int for value in values)
        or gallery_count < 1
        or dimensions < 2
        or dimensions % 2
        or query_count < 1
        or samples < 1
        or warmups < 0
        or threads < 1
        or seed < 0
    ):
        raise ValueError("resident int4 benchmark authority differs")
    original_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(threads)
        observed_threads = torch.get_num_threads()
        if observed_threads != threads:
            raise RuntimeError("resident int4 benchmark thread authority differs")
        return _benchmark_resident_int4_configured(
            gallery_count=gallery_count,
            dimensions=dimensions,
            query_count=query_count,
            samples=samples,
            warmups=warmups,
            threads=observed_threads,
            seed=seed,
        )
    finally:
        torch.set_num_threads(original_threads)


def _benchmark_resident_int4_configured(
    *,
    gallery_count: int,
    dimensions: int,
    query_count: int,
    samples: int,
    warmups: int,
    threads: int,
    seed: int,
) -> dict[str, Any]:
    """Run the benchmark after validation and temporary thread configuration."""

    generator = torch.Generator().manual_seed(seed)
    gallery = pack_int4_unit_embeddings(
        F.normalize(torch.randn((gallery_count, dimensions), generator=generator), dim=1)
        .to(torch.float32)
        .contiguous()
    )
    queries = pack_int4_unit_embeddings(
        F.normalize(torch.randn((query_count, dimensions), generator=generator), dim=1)
        .to(torch.float32)
        .contiguous()
    )
    construction_started = time.perf_counter_ns()
    resident = ResidentInt4Gallery.from_packed(gallery)
    resident_construction_ns = time.perf_counter_ns() - construction_started
    integer_backend_observed = resident.integer_backend_available
    if not integer_backend_observed:
        raise RuntimeError("resident int4 integer kernel is unavailable")
    float_gallery = resident.gallery_codes_transposed.float().contiguous()
    float_gallery_norms = resident.inverse_norms.float().contiguous()
    query_codes = queries.signed_codes().float().contiguous()
    query_norms = queries.inverse_norms.float().contiguous()

    def packed_decode_float() -> torch.Tensor:
        return queries.cosine_similarity(gallery)

    def resident_int8() -> torch.Tensor:
        return resident.score_queries(queries=queries, require_integer=True)

    def resident_float32() -> torch.Tensor:
        return (query_codes @ float_gallery) * query_norms[:, None] * float_gallery_norms[None, :]

    runners = {
        "packed_decode_float": packed_decode_float,
        "resident_float32": resident_float32,
        "resident_int8": resident_int8,
    }
    for _ in range(warmups):
        for runner in runners.values():
            runner()
    outputs = {name: runner() for name, runner in runners.items()}
    scores_bit_exact = all(
        torch.equal(outputs["packed_decode_float"], value) for value in outputs.values()
    )
    if not scores_bit_exact:
        raise RuntimeError("resident int4 benchmark score parity differs")
    timings: dict[str, list[int]] = {name: [] for name in runners}
    for _ in range(samples):
        for name, runner in runners.items():
            started = time.perf_counter_ns()
            runner()
            timings[name].append(time.perf_counter_ns() - started)
    packed_bytes = dimensions // 2 + 2
    return {
        "arms": {name: _arm(values) for name, values in timings.items()},
        "claim_eligible": False,
        "cpu_model": _cpu_model(),
        "dimensions": dimensions,
        "execution_backend": (
            "torch-private-int-mm-cpu-observed" if integer_backend_observed else "unavailable"
        ),
        "gallery_count": gallery_count,
        "integer_backend_required": True,
        "memory_bytes_per_vector": {
            "packed_persistent": packed_bytes,
            "resident_float32_layout": dimensions * 4 + 4,
            "resident_int8_layout": dimensions + 2,
            "resident_int8_plus_packed_if_coexisting": dimensions + 2 + packed_bytes,
        },
        "platform": platform.platform(),
        "packed_int4_source_sha256": _sha256(Path(packed_int4.__file__)),
        "python": platform.python_version(),
        "query_count": query_count,
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "resident_construction_ns": resident_construction_ns,
        "schema": "sfora-packed-int4-resident-benchmark-v2",
        "scores_bit_exact": scores_bit_exact,
        "script_sha256": _sha256(Path(__file__)),
        "seed": seed,
        "threads": threads,
        "torch": torch.__version__,
        "warmups": warmups,
    }


def canonical_benchmark_bytes(receipt: dict[str, Any]) -> bytes:
    """Serialize a benchmark receipt as sorted compact JSON with one newline."""

    text = json.dumps(receipt, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return f"{text}\n".encode()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit benchmark command line."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gallery-count", type=int, default=5_924)
    parser.add_argument("--dimensions", type=int, default=128)
    parser.add_argument("--query-count", type=int, default=1)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--warmups", type=int, default=100)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--execute-resident-int4-benchmark", action="store_true", required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run once and atomically publish the canonical receipt."""

    args = parse_args(argv)
    receipt = benchmark_resident_int4(
        gallery_count=args.gallery_count,
        dimensions=args.dimensions,
        query_count=args.query_count,
        samples=args.samples,
        warmups=args.warmups,
        threads=args.threads,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_benchmark_bytes(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.output)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
