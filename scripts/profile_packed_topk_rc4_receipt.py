"""Validate the authenticated RC4 packed top-k profiling receipt."""

from __future__ import annotations

import copy
import math
import re
from typing import Any

_STAGES = {
    "host_to_device",
    "score",
    "block_selection",
    "merge",
    "buffer_initialization",
    "device_to_host",
    "host_and_api",
}
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


def _hash(value: object, name: str) -> None:
    if type(value) is not str or _DIGEST.fullmatch(value) is None:
        raise ValueError(f"{name} must be a SHA-256 digest")


def _positive_int(value: object, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nearest_rank(samples: list[int], percentile: int) -> int:
    ordered = sorted(samples)
    return ordered[math.ceil(len(ordered) * percentile / 100) - 1]


def summarize_receipt(value: object, *, expected_samples: int = 50) -> dict[str, Any]:
    _positive_int(expected_samples, "expected_samples")
    if type(value) is not dict or value.get("schema") != "sfora-packed-topk-rc4-stage-v1":
        raise ValueError("RC4 profiler receipt schema differs")
    for name in (
        "source_commit",
        "gallery_sha256",
        "baseline_receipt_sha256",
        "diagnostic_binary_sha256",
        "rc3_library_sha256",
        "fixture_manifest_sha256",
    ):
        _hash(value.get(name), name)
    for name in ("query_sha256", "profiler_report_sha256"):
        hashes = value.get(name)
        if type(hashes) is not dict or set(hashes) != {"1", "32"}:
            raise ValueError(f"{name} batches differ")
        for batch in ("1", "32"):
            _hash(hashes[batch], f"{name}.{batch}")
    rows = value.get("batches")
    if type(rows) is not dict or set(rows) != {"1", "32"}:
        raise ValueError("RC4 profiler batches differ")
    result = copy.deepcopy(value)
    for batch in ("1", "32"):
        row = rows[batch]
        if type(row) is not dict:
            raise TypeError(f"batch {batch} differs")
        if row.get("exact_score_bits") is not True or row.get("exact_ordinals") is not True:
            raise ValueError(f"batch {batch} exactness failed")
        samples = row.get("end_to_end_ns")
        if type(samples) is not list or len(samples) != expected_samples:
            raise ValueError(f"batch {batch} sample count differs")
        samples = [_positive_int(sample, f"batch {batch} sample") for sample in samples]
        stages = row.get("stage_mean_ns")
        if type(stages) is not dict or set(stages) != _STAGES:
            raise ValueError(f"batch {batch} stages differ")
        for name, measured in stages.items():
            _positive_int(measured, f"batch {batch} {name}")
        for name in ("gpu_peak_memory_bytes", "process_peak_rss_bytes"):
            _positive_int(row.get(name), f"batch {batch} {name}")
        summary = result["batches"][batch]
        summary["p50_ns"] = _nearest_rank(samples, 50)
        summary["p95_ns"] = _nearest_rank(samples, 95)
        summary["p99_ns"] = _nearest_rank(samples, 99)
        summary["queries_per_second"] = int(batch) * 1e9 / (sum(samples) / len(samples))
    return result
