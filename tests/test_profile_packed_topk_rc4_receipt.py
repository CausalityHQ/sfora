"""The RC4 profiler receipt must fail closed on incomplete evidence."""

from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/profile_packed_topk_rc4_receipt.py"
_SPEC = importlib.util.spec_from_file_location("profile_packed_topk_rc4_receipt", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
summarize_receipt = _MODULE.summarize_receipt


def _receipt() -> dict[str, object]:
    digest = "a" * 64
    row = {
        "exact_score_bits": True,
        "exact_ordinals": True,
        "end_to_end_ns": [100, 110, 120, 130, 140],
        "stage_mean_ns": {
            "host_to_device": 5,
            "score": 30,
            "block_selection": 40,
            "merge": 10,
            "buffer_initialization": 10,
            "device_to_host": 5,
            "host_and_api": 20,
        },
        "gpu_peak_memory_bytes": 256,
        "process_peak_rss_bytes": 1024,
    }
    return {
        "schema": "sfora-packed-topk-rc4-stage-v1",
        "source_commit": "b" * 40,
        "gallery_sha256": digest,
        "query_sha256": {"1": digest, "32": digest},
        "baseline_receipt_sha256": digest,
        "diagnostic_binary_sha256": digest,
        "rc3_library_sha256": digest,
        "fixture_manifest_sha256": digest,
        "profiler_report_sha256": {"1": digest, "32": digest},
        "batches": {"1": copy.deepcopy(row), "32": copy.deepcopy(row)},
    }


def test_receipt_recomputes_nearest_rank_percentiles_and_throughput() -> None:
    result = summarize_receipt(_receipt(), expected_samples=5)

    assert result["batches"]["1"]["p50_ns"] == 120
    assert result["batches"]["1"]["p95_ns"] == 140
    assert result["batches"]["1"]["p99_ns"] == 140
    assert result["batches"]["32"]["queries_per_second"] == 32e9 / 120


@pytest.mark.parametrize(
    "mutation",
    [
        lambda r: r["batches"].pop("32"),
        lambda r: r["batches"]["1"]["end_to_end_ns"].pop(),
        lambda r: r["batches"]["1"].update(exact_ordinals=False),
        lambda r: r["batches"]["1"]["stage_mean_ns"].pop("block_selection"),
        lambda r: r.update(gallery_sha256="unbound"),
        lambda r: r["batches"]["1"].update(gpu_peak_memory_bytes=None),
    ],
)
def test_receipt_rejects_missing_or_failed_evidence(mutation: Any) -> None:
    receipt = _receipt()
    mutation(receipt)

    with pytest.raises((TypeError, ValueError)):
        summarize_receipt(receipt, expected_samples=5)
