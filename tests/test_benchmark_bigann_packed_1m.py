"""Distinct-source scaling inputs preserve the byte-level BIGANN layout."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_bigann_packed_1m.py"
SPEC = importlib.util.spec_from_file_location("benchmark_bigann_packed_1m", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_reads_real_payload_length_despite_larger_header_and_packs_centered_codes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "base.u8bin"
    rows = np.full((4, 128), 128, dtype=np.uint8)
    rows[0, 0] = 129
    rows[1, 0] = 127
    rows[2, 1] = 130
    rows[3, 2] = 131
    path.write_bytes(struct.pack("<II", 1_000_000_000, 128) + rows.tobytes())
    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    raw = MODULE.read_u8bin_prefix(
        path, digest, header_rows=1_000_000_000, file_rows=4, take_rows=2
    )
    codes, inverse_norms = MODULE.pack_centered_u8(raw)

    assert codes.shape == (2, 128)
    assert codes.dtype == np.int8
    assert codes[0, 0] == 1
    assert codes[1, 0] == -1
    assert np.all(codes[:, 1:] == 0)
    assert inverse_norms.dtype == np.dtype("<f2")
    assert inverse_norms.tolist() == [1.0, 1.0]


def test_rejects_truncated_or_changed_bigann_source(tmp_path: Path) -> None:
    path = tmp_path / "base.u8bin"
    path.write_bytes(struct.pack("<II", 1_000_000_000, 128) + bytes(128 * 4))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source"):
        MODULE.read_u8bin_prefix(path, digest, header_rows=1_000_000_000, file_rows=5, take_rows=2)
    with pytest.raises(ValueError, match="source"):
        MODULE.read_u8bin_prefix(
            path, "0" * 64, header_rows=1_000_000_000, file_rows=4, take_rows=2
        )


def test_zero_centered_vector_cannot_be_a_cosine_code() -> None:
    with pytest.raises(ValueError, match="zero"):
        MODULE.pack_centered_u8(np.full((1, 128), 128, dtype=np.uint8))


def test_exact_oracle_breaks_score_ties_by_source_row_ordinal() -> None:
    codes = np.zeros((12, 128), dtype=np.int8)
    codes[:, 0] = 1
    codes[2, 0] = -1
    norms = np.ones(12, dtype="<f2")
    query = np.zeros(128, dtype=np.int8)
    query[0] = 1

    ordinals, scores = MODULE.exact_oracle_topk(codes, norms, query, np.float16(1), k=10)

    assert ordinals.tolist() == [0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    assert scores.tolist() == [1.0] * 10


def test_gpu_snapshot_requires_exclusive_visible_process_and_allows_missing_memory() -> None:
    assert MODULE.exclusive_gpu_snapshot("55, 256\n", pid=55) == {
        "compute_pids": [55],
        "own_memory_bytes": 256 * 1024 * 1024,
    }
    assert MODULE.exclusive_gpu_snapshot("55, [N/A]\n", pid=55)["own_memory_bytes"] is None
    with pytest.raises(ValueError, match="occupied"):
        MODULE.exclusive_gpu_snapshot("42, 8083\n55, 256\n", pid=55)
    with pytest.raises(ValueError, match="visible"):
        MODULE.exclusive_gpu_snapshot("42, 8083\n", pid=55)
    with pytest.raises(ValueError, match="telemetry"):
        MODULE.exclusive_gpu_snapshot("55, unavailable\n", pid=55)


def test_gpu_must_be_idle_before_native_gallery_creation() -> None:
    MODULE.require_idle_gpu_output("")
    with pytest.raises(ValueError, match="occupied"):
        MODULE.require_idle_gpu_output("3495844, 8083\n")


def test_native_search_parity_checks_ids_and_score_arithmetic() -> None:
    ids = np.array([0, 1], dtype=np.int64)
    scores = np.array([1.0, 0.5], dtype=np.float32)
    MODULE.require_native_oracle_parity(ids, scores, ids.copy(), scores.copy())
    with pytest.raises(ValueError, match="oracle"):
        MODULE.require_native_oracle_parity(ids, scores, ids[::-1].copy(), scores.copy())
    with pytest.raises(ValueError, match="oracle"):
        MODULE.require_native_oracle_parity(
            ids, scores, ids.copy(), np.array([1.0, 0.4], dtype=np.float32)
        )


def test_source_commit_must_match_clean_tracked_benchmark(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "benchmark.py").write_text("print('version one')\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "benchmark.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Verifier",
            "-c",
            "user.email=verifier@example.invalid",
            "commit",
            "-qm",
            "source",
        ],
        check=True,
    )
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    MODULE.require_clean_source_commit(tmp_path, commit, ("benchmark.py",))
    with pytest.raises(ValueError, match="source commit"):
        MODULE.require_clean_source_commit(tmp_path, "0" * 40, ("benchmark.py",))
    (tmp_path / "benchmark.py").write_text("print('changed')\n")
    with pytest.raises(ValueError, match="source commit"):
        MODULE.require_clean_source_commit(tmp_path, commit, ("benchmark.py",))
