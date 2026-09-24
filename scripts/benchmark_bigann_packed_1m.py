#!/usr/bin/env python3
"""Distinct BIGANN source-item serving replay; no image-quality claim."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.cutile_int8 import CutilePackedInt8Gallery

BASE_SHA256 = "24f9c72cb7cf4cb388a2e6e2eeb939f671739be16e84edc9b3b52813063c918f"
QUERY_SHA256 = "eca755831fc9a8004e14886df48b81109a8ed3bfc6b632509c93c0460a30d552"
LIBRARY_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
API_SHA256 = "b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409"
GALLERY_ROWS = 1_000_000
QUERY_ROWS = 32


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def require_clean_source_commit(
    root: Path, expected_commit: str, relative_paths: tuple[str, ...]
) -> None:
    """Bind executed files to the named immutable Git revision."""

    if (
        len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
        or not relative_paths
    ):
        raise ValueError("BIGANN source commit differs")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    try:
        revision = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        ).stdout.strip()
        checkout_root = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
            env=environment,
        ).stdout.strip()
        if revision != expected_commit or Path(checkout_root).resolve() != root.resolve():
            raise ValueError("BIGANN source commit differs")
        for relative in relative_paths:
            published = subprocess.run(
                ["git", "-C", str(root), "show", f"HEAD:{relative}"],
                check=True,
                capture_output=True,
                timeout=10,
                env=environment,
            ).stdout
            if (root / relative).read_bytes() != published:
                raise ValueError("BIGANN source commit differs")
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("BIGANN source commit differs") from error


def read_u8bin_prefix(
    path: Path,
    expected_sha256: str,
    *,
    header_rows: int,
    file_rows: int,
    take_rows: int,
) -> np.ndarray:
    """Read a local BIGANN prefix whose header still names the parent corpus."""

    if (
        not isinstance(path, Path)
        or not path.is_file()
        or not 0 < take_rows <= file_rows <= header_rows
        or path.stat().st_size != 8 + file_rows * 128
        or sha256(path) != expected_sha256
    ):
        raise ValueError("BIGANN source authority differs")
    with path.open("rb") as stream:
        header = stream.read(8)
    if struct.unpack("<II", header) != (header_rows, 128):
        raise ValueError("BIGANN source header differs")
    return np.memmap(path, dtype=np.uint8, mode="r", offset=8, shape=(file_rows, 128))[:take_rows]


def pack_centered_u8(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map BIGANN bytes to the native signed-code and f16-inverse-norm wire."""

    if (
        not isinstance(values, np.ndarray)
        or values.dtype != np.uint8
        or values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] != 128
    ):
        raise ValueError("BIGANN byte-vector inventory differs")
    codes = np.empty(values.shape, dtype=np.int8)
    inverse_norms = np.empty(values.shape[0], dtype="<f2")
    for start in range(0, len(values), 65_536):
        stop = min(start + 65_536, len(values))
        centered = values[start:stop].astype(np.int16) - 128
        codes[start:stop] = centered.astype(np.int8)
        floats = centered.astype(np.float32)
        squared = (floats * floats).sum(axis=1)
        if np.any(squared == 0):
            raise ValueError("BIGANN centered zero vector cannot use cosine")
        inverse_norms[start:stop] = (1.0 / np.sqrt(squared)).astype("<f2")
    if not np.isfinite(inverse_norms).all() or not (inverse_norms > 0).all():
        raise ValueError("BIGANN inverse norms differ")
    return codes, inverse_norms


def exact_oracle_topk(
    codes: np.ndarray,
    inverse_norms: np.ndarray,
    query_code: np.ndarray,
    query_inverse_norm: np.float16,
    *,
    k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Score one query in int32 and break equal scores by source ordinal."""

    if (
        codes.dtype != np.int8
        or codes.ndim != 2
        or codes.shape[1] != 128
        or inverse_norms.dtype != np.dtype("<f2")
        or inverse_norms.shape != (len(codes),)
        or query_code.dtype != np.int8
        or query_code.shape != (128,)
        or not 1 <= k <= len(codes)
    ):
        raise ValueError("BIGANN exact oracle inventory differs")
    query = query_code.astype(np.int32)
    scores = np.empty(len(codes), dtype=np.float32)
    for start in range(0, len(codes), 65_536):
        stop = min(start + 65_536, len(codes))
        dots = codes[start:stop].astype(np.int32) @ query
        scores[start:stop] = (
            dots.astype(np.float32)
            * np.float32(query_inverse_norm)
            * inverse_norms[start:stop].astype(np.float32)
        )
    ranked = np.argsort(-scores, kind="stable")[:k]
    return ranked.astype(np.int64), scores[ranked]


def require_native_oracle_parity(
    actual_ids: np.ndarray,
    actual_scores: np.ndarray,
    expected_ids: np.ndarray,
    expected_scores: np.ndarray,
) -> None:
    if (
        not np.array_equal(actual_ids, expected_ids)
        or actual_scores.shape != expected_scores.shape
        or not np.allclose(actual_scores, expected_scores, rtol=0, atol=1e-4)
    ):
        raise ValueError("BIGANN scaling exact top-k oracle differs")


def percentile(samples: list[int], percent: int) -> int:
    return sorted(samples)[math.ceil(len(samples) * percent / 100) - 1]


def exclusive_gpu_snapshot(output: str, *, pid: int) -> dict[str, object]:
    """Require that the only visible compute process is this benchmark."""

    processes: list[tuple[int, int | None]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 2 or not fields[0].isdigit():
            raise ValueError("BIGANN GPU telemetry differs")
        memory = fields[1]
        if memory.isdigit():
            memory_bytes = int(memory) * 1024 * 1024
        elif memory in ("[N/A]", "[Not Supported]", "N/A", "Not Supported"):
            memory_bytes = None
        else:
            raise ValueError("BIGANN GPU telemetry differs")
        processes.append((int(fields[0]), memory_bytes))
    own = [memory for process, memory in processes if process == pid]
    if len(own) != 1:
        raise ValueError("BIGANN own GPU process is not visible")
    if len(processes) != 1:
        raise ValueError("BIGANN GPU is occupied")
    return {"compute_pids": [process for process, _ in processes], "own_memory_bytes": own[0]}


def require_idle_gpu_output(output: str) -> None:
    if output.strip():
        raise ValueError("BIGANN GPU is occupied before gallery creation")


def gpu_process_output() -> str:
    return subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout


def gpu_process_snapshot() -> dict[str, object]:
    output = gpu_process_output()
    return exclusive_gpu_snapshot(output, pid=os.getpid())


def gpu_device_snapshot() -> str:
    """Retain the device, driver, clock, power, and temperature as observed."""

    return subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,clocks.sm,power.draw,temperature.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--execute-distinct-1m-scaling", action="store_true", required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or not args.output.parent.is_dir()
        or not 1 <= args.calls <= 500
        or len(args.source_commit) != 40
    ):
        raise ValueError("BIGANN scaling invocation differs")
    initial_script_sha256 = sha256(Path(__file__))
    if sha256(args.library) != LIBRARY_SHA256:
        raise ValueError("BIGANN scaling library differs")
    import sfora.cutile_int8 as api

    if sha256(Path(api.__file__)) != API_SHA256:
        raise ValueError("BIGANN scaling API differs")
    source_root = Path(__file__).resolve().parents[1]
    if Path(api.__file__).resolve() != (source_root / "src/sfora/cutile_int8.py").resolve():
        raise ValueError("BIGANN imported API source differs")
    require_clean_source_commit(
        source_root,
        args.source_commit,
        ("scripts/benchmark_bigann_packed_1m.py", "src/sfora/cutile_int8.py"),
    )
    base = read_u8bin_prefix(
        args.base,
        BASE_SHA256,
        header_rows=1_000_000_000,
        file_rows=10_000_000,
        take_rows=GALLERY_ROWS,
    )
    query = read_u8bin_prefix(
        args.query,
        QUERY_SHA256,
        header_rows=10_000,
        file_rows=10_000,
        take_rows=QUERY_ROWS,
    )
    gallery_codes, gallery_norms = pack_centered_u8(base)
    query_codes, query_norms = pack_centered_u8(query)
    gallery_code_sha256 = hashlib.sha256(gallery_codes.tobytes()).hexdigest()
    gallery_norm_sha256 = hashlib.sha256(gallery_norms.tobytes()).hexdigest()
    query_code_sha256 = hashlib.sha256(query_codes.tobytes()).hexdigest()
    query_norm_sha256 = hashlib.sha256(query_norms.tobytes()).hexdigest()
    measurements = []
    oracle_checks = []
    require_idle_gpu_output(gpu_process_output())
    with CutilePackedInt8Gallery.open(args.library, gallery_codes, gallery_norms) as gallery:
        gpu_after_gallery_open = gpu_process_snapshot()
        full_query_ids, full_query_scores = gallery.search(query_codes, query_norms, k=10)
        for query_index in range(QUERY_ROWS):
            expected_ids, expected_scores = exact_oracle_topk(
                gallery_codes,
                gallery_norms,
                query_codes[query_index],
                query_norms[query_index],
                k=10,
            )
            require_native_oracle_parity(
                full_query_ids[query_index],
                full_query_scores[query_index],
                expected_ids,
                expected_scores,
            )
            oracle_checks.append(
                {
                    "query_ordinal": query_index,
                    "top10_ordinals": expected_ids.tolist(),
                    "top10_scores": expected_scores.tolist(),
                }
            )
        one_ids, one_scores = gallery.search(query_codes[:1], query_norms[:1], k=10)
        require_native_oracle_parity(
            one_ids[0], one_scores[0], full_query_ids[0], full_query_scores[0]
        )
        gpu_before_timing = gpu_device_snapshot()
        gpu_process_before_timing = gpu_process_snapshot()
        for pair, order in enumerate(((1, 32), (32, 1)), start=1):
            for batch in order:
                qcodes = query_codes[:batch]
                qnorms = query_norms[:batch]
                first_ids, first_scores = gallery.search(qcodes, qnorms, k=10)
                first_hash = hashlib.sha256(
                    first_ids.tobytes() + first_scores.tobytes()
                ).hexdigest()
                for _ in range(5):
                    gallery.search(qcodes, qnorms, k=10)
                samples: list[int] = []
                for _ in range(args.calls):
                    started = time.perf_counter_ns()
                    ids, scores = gallery.search(qcodes, qnorms, k=10)
                    samples.append(time.perf_counter_ns() - started)
                    if hashlib.sha256(ids.tobytes() + scores.tobytes()).hexdigest() != first_hash:
                        raise ValueError("BIGANN scaling search result changed")
                measurements.append(
                    {
                        "pair": pair,
                        "batch": batch,
                        "warmups": 5,
                        "samples_ns": samples,
                        "p50_ns": percentile(samples, 50),
                        "p95_ns": percentile(samples, 95),
                        "p99_ns": percentile(samples, 99) if args.calls >= 100 else None,
                        "max_ns": max(samples),
                        "queries_per_second": batch * 1e9 / (sum(samples) / len(samples)),
                        "result_sha256": first_hash,
                    }
                )
        gpu_process_after_timing = gpu_process_snapshot()
        gpu_after_timing = gpu_device_snapshot()
    if (
        sha256(args.base) != BASE_SHA256
        or sha256(args.query) != QUERY_SHA256
        or sha256(args.library) != LIBRARY_SHA256
        or sha256(Path(api.__file__)) != API_SHA256
        or sha256(Path(__file__)) != initial_script_sha256
    ):
        raise ValueError("BIGANN scaling inputs changed during timing")
    require_clean_source_commit(
        source_root,
        args.source_commit,
        ("scripts/benchmark_bigann_packed_1m.py", "src/sfora/cutile_int8.py"),
    )
    receipt = {
        "schema": "sfora-bigann-source-ordinal-packed-exact-1m-diagnostic-v1",
        "claim_eligible": False,
        "scope": (
            "packed exact-search scaling on one million BIGANN source ordinals; "
            "no learned image encoder or quality metric; "
            "transformed cosine is not the official BIGANN L2 protocol"
        ),
        "source_commit": args.source_commit,
        "script_sha256": initial_script_sha256,
        "library_sha256": LIBRARY_SHA256,
        "api_sha256": API_SHA256,
        "source": {
            "base_sha256": BASE_SHA256,
            "query_sha256": QUERY_SHA256,
            "base_file_rows": 10_000_000,
            "base_parent_header_rows": 1_000_000_000,
            "query_file_rows": 10_000,
            "gallery_rows": GALLERY_ROWS,
            "query_rows": QUERY_ROWS,
            "gallery_source_ordinals": "base[0:1000000]",
            "query_source_ordinals": "query[0:32]",
            "construction": (
                "uint8 coordinate minus 128, signed int8 code, reciprocal l2 norm rounded to f16"
            ),
            "gallery_code_sha256": gallery_code_sha256,
            "gallery_norm_sha256": gallery_norm_sha256,
            "query_code_sha256": query_code_sha256,
            "query_norm_sha256": query_norm_sha256,
            "gallery_bytes": gallery_codes.nbytes + gallery_norms.nbytes,
            "gallery_bytes_per_item": 130,
            "official_bigann_metric": "L2 on original uint8 vectors; not measured here",
            "measured_metric": "exact cosine on centered signed int8 and f16 inverse norms",
            "production_code_range": (
                "Sfora learned-code quantizer uses -127..127; "
                "this source transform can include -128"
            ),
            "retrieval_semantics": "top-10 is an arithmetic oracle, not a semantic quality result",
        },
        "oracle_checks": oracle_checks,
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "gpu_process_after_gallery_open": gpu_after_gallery_open,
        "gpu_process_before_timing": gpu_process_before_timing,
        "gpu_process_after_timing": gpu_process_after_timing,
        "gpu_before_timing": gpu_before_timing,
        "gpu_after_timing": gpu_after_timing,
        "measurements": measurements,
        "argv": sys.argv,
    }
    payload = (json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode()
    with publish_bytes_noreplace(
        args.output, payload, validator=lambda persisted: _check_publication(persisted, payload)
    ):
        pass
    print(json.dumps({"receipt": str(args.output), "rows": len(measurements)}), flush=True)
    return 0


def _check_publication(persisted: bytes, expected: bytes) -> None:
    if persisted != expected:
        raise ValueError("BIGANN scaling publication differs")


if __name__ == "__main__":
    raise SystemExit(main())
