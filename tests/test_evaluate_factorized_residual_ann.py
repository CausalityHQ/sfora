from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evaluate_factorized_residual_ann.py"
SPEC = importlib.util.spec_from_file_location("evaluate_factorized_residual_ann", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
subject = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(subject)


def test_latency_summary_uses_higher_percentiles_and_binds_raw_samples() -> None:
    raw = [40, 10, 30, 20]

    summary = subject.summarize_latency_ns(raw)

    expected_raw = np.asarray(raw, dtype="<i8").tobytes()
    assert summary == {
        "maximum": 40,
        "mean": 25.0,
        "minimum": 10,
        "p50": 30,
        "p95": 40,
        "p99": 40,
        "raw_sha256": hashlib.sha256(expected_raw).hexdigest(),
    }


def test_matrix_header_rejects_trailing_bytes(tmp_path: Path) -> None:
    path = tmp_path / "queries.u8bin"
    payload = np.arange(6, dtype=np.uint8).reshape(2, 3)
    path.write_bytes(np.asarray([2, 3], dtype="<u4").tobytes() + payload.tobytes())

    assert subject._read_matrix_header(path, np.dtype(np.uint8)) == (2, 3)

    path.write_bytes(path.read_bytes() + b"\x00")
    with pytest.raises(ValueError, match="factorized residual benchmark receipt differs"):
        subject._read_matrix_header(path, np.dtype(np.uint8))


def test_authenticated_matrix_is_a_stable_owned_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "queries.u8bin"
    payload = np.arange(6, dtype=np.uint8).reshape(2, 3)
    wire = np.asarray([2, 3], dtype="<u4").tobytes() + payload.tobytes()
    path.write_bytes(wire)

    observed = subject._read_authenticated_matrix(
        path,
        np.dtype(np.uint8),
        hashlib.sha256(wire).hexdigest(),
    )
    path.write_bytes(np.asarray([2, 3], dtype="<u4").tobytes() + b"\xff" * 6)

    np.testing.assert_array_equal(observed, payload)
    assert observed.flags.owndata
    assert not observed.flags.writeable


def test_authenticated_matrix_rejects_same_inode_mutation_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "queries.u8bin"
    payload = np.arange(6, dtype=np.uint8).reshape(2, 3)
    wire = np.asarray([2, 3], dtype="<u4").tobytes() + payload.tobytes()
    path.write_bytes(wire)
    real_pread = subject.os.pread
    mutated = False

    def mutate_after_read(descriptor: int, size: int, offset: int) -> bytes:
        nonlocal mutated
        block = real_pread(descriptor, size, offset)
        if not mutated:
            mutated = True
            with path.open("r+b") as stream:
                stream.seek(8)
                stream.write(b"\xff")
                stream.flush()
        return block

    monkeypatch.setattr(subject.os, "pread", mutate_after_read)

    with pytest.raises(ValueError, match="factorized residual benchmark receipt differs"):
        subject._read_authenticated_matrix(
            path,
            np.dtype(np.uint8),
            hashlib.sha256(wire).hexdigest(),
        )


def test_authenticated_matrix_accepts_explicit_trailing_distance_plane(
    tmp_path: Path,
) -> None:
    path = tmp_path / "truth.bin"
    ids = np.asarray([[4, 7], [1, 9]], dtype="<u4")
    distances = np.asarray([[0.0, 1.5], [2.0, 3.0]], dtype="<f4")
    wire = (
        np.asarray([2, 2], dtype="<u4").tobytes()
        + ids.tobytes()
        + distances.tobytes()
    )
    path.write_bytes(wire)

    observed = subject._read_authenticated_matrix(
        path,
        np.dtype("<u4"),
        hashlib.sha256(wire).hexdigest(),
        trailing_dtype=np.dtype("<f4"),
    )

    np.testing.assert_array_equal(observed, ids)


def test_canonical_receipt_recomputes_quality_latency_and_stage_totals() -> None:
    receipt = {
        "artifact_manifest_sha256": "11" * 32,
        "claim_eligible": False,
        "dataset": "fixture",
        "io": {"logical_bytes": 16, "physical_bytes": 32, "rows_scanned": 9},
        "latency_ns": subject.summarize_latency_ns([10, 20]),
        "latency_raw_ns": [10, 20],
        "memory_ledger": {
            "artifact_resident_bytes": 100,
            "context_bytes": 10,
            "fixed_service_overhead_bytes": 20,
            "memory_limit_bytes": 200,
            "safety_headroom_bytes": 30,
            "total_reserved_bytes": 160,
            "vector_store_resident_bytes": 0,
        },
        "native_binary_sha256": "22" * 32,
        "native_source_sha256": "33" * 32,
        "outputs_sha256": "44" * 32,
        "peak_rss_bytes": 150,
        "per_query_hits": [2, 1],
        "query_count": 2,
        "query_order_seed": None,
        "query_start": 7,
        "query_sha256": "55" * 32,
        "return_width": 2,
        "schema": "sfora-factorized-residual-benchmark-v1",
        "split": "fixture queries 0..1",
        "stage_ns": {
            "candidate_raw": [4, 8],
            "candidate_total": 12,
            "exact_raw": [5, 10],
            "exact_total": 15,
        },
        "strict_id_recall_ppm": 750_000,
        "thread_count": 1,
        "truth_sha256": "66" * 32,
        "truth_format": "ids-u32",
        "vector_sha256": "77" * 32,
        "warmup_queries": 1,
    }

    wire = subject.canonical_benchmark_receipt_bytes(receipt)

    assert wire == (
        json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    for mutation in (
        {"strict_id_recall_ppm": 750_001},
        {"query_start": True},
        {"truth_format": "auto"},
        {"latency_ns": {**receipt["latency_ns"], "p99": 19}},
        {
            "stage_ns": {
                **receipt["stage_ns"],
                "candidate_total": 13,
            }
        },
    ):
        with pytest.raises(ValueError, match="factorized residual benchmark receipt differs"):
            subject.canonical_benchmark_receipt_bytes({**receipt, **mutation})


def test_cli_rejects_remote_paths_and_requires_explicit_execution(tmp_path: Path) -> None:
    common = [
        "--artifact",
        str(tmp_path / "artifact"),
        "--manifest-sha256",
        "11" * 32,
        "--vectors",
        str(tmp_path / "vectors"),
        "--query",
        str(tmp_path / "query"),
        "--query-sha256",
        "22" * 32,
        "--truth",
        str(tmp_path / "truth"),
        "--truth-sha256",
        "33" * 32,
        "--truth-format",
        "ids-u32",
        "--native-cache",
        str(tmp_path / "cache"),
        "--output",
        str(tmp_path / "receipt.json"),
        "--dataset",
        "fixture",
        "--split",
        "fixture split",
        "--query-start",
        "0",
        "--query-count",
        "2",
    ]

    with pytest.raises(SystemExit):
        subject.parse_args(common)
    with pytest.raises(SystemExit):
        subject.parse_args(
            ["--artifact", "s3://bucket/artifact", *common[2:], "--execute"]
        )
    parsed = subject.parse_args([*common, "--execute"])
    assert parsed.artifact == tmp_path / "artifact"
