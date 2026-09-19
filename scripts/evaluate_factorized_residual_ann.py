#!/usr/bin/env python3
"""Evaluate the authenticated local factorized-residual ANN public API."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import stat
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sfora.factorized_residual_ann import FactorizedResidualArtifact
from sfora.factorized_residual_index import FactorizedResidualIndex
from sfora.factorized_residual_native import compile_factorized_residual_backend
from sfora.vector_store import DirectIoVectorStore

_ERROR = "factorized residual benchmark receipt differs"
_SHA_FIELDS = (
    "artifact_manifest_sha256",
    "native_binary_sha256",
    "native_source_sha256",
    "outputs_sha256",
    "query_sha256",
    "truth_sha256",
    "vector_sha256",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def _read_matrix_header(path: Path, dtype: np.dtype[Any]) -> tuple[int, int]:
    """Read a two-u32 matrix header and reject truncated or appended bytes."""

    header = np.fromfile(path, dtype="<u4", count=2)
    if header.shape != (2,):
        raise ValueError(_ERROR)
    rows, columns = map(int, header)
    if rows < 1 or columns < 1 or path.stat().st_size != 8 + rows * columns * dtype.itemsize:
        raise ValueError(_ERROR)
    return rows, columns


def _read_authenticated_matrix(
    path: Path,
    dtype: np.dtype[Any],
    expected_sha256: str,
    *,
    trailing_dtype: np.dtype[Any] | None = None,
) -> np.ndarray[Any, np.dtype[Any]]:
    """Return an owned matrix snapshot authenticated from one stable descriptor."""

    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_CLOEXEC | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(before.st_mode)
            or before.st_size < 9
            or before.st_size > 1 << 30
        ):
            raise ValueError(_ERROR)
        wire = bytearray()
        while len(wire) < before.st_size:
            block = os.pread(
                descriptor,
                min(8 << 20, before.st_size - len(wire)),
                len(wire),
            )
            if not block:
                raise ValueError(_ERROR)
            wire.extend(block)
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
        if (
            any(getattr(before, field) != getattr(after, field) for field in stable)
            or hashlib.sha256(wire).hexdigest() != expected_sha256
        ):
            raise ValueError(_ERROR)
    finally:
        os.close(descriptor)
    header = np.frombuffer(wire, dtype="<u4", count=2)
    rows, columns = map(int, header)
    entries = rows * columns
    trailing_itemsize = 0 if trailing_dtype is None else trailing_dtype.itemsize
    if (
        rows < 1
        or columns < 1
        or len(wire) != 8 + entries * (dtype.itemsize + trailing_itemsize)
    ):
        raise ValueError(_ERROR)
    if trailing_dtype is not None:
        trailing = np.frombuffer(
            wire,
            dtype=trailing_dtype,
            count=entries,
            offset=8 + entries * dtype.itemsize,
        )
        if not bool(np.isfinite(trailing).all()) or bool((trailing < 0).any()):
            raise ValueError(_ERROR)
    result = (
        np.frombuffer(wire, dtype=dtype, count=entries, offset=8)
        .reshape(rows, columns)
        .copy()
    )
    result.flags.writeable = False
    return result


def summarize_latency_ns(raw: list[int]) -> dict[str, int | float | str]:
    """Summarize nonempty nanosecond samples with the registered higher rule."""

    if not raw or any(type(value) is not int or value < 0 for value in raw):
        raise ValueError(_ERROR)
    values = np.asarray(raw, dtype="<i8")
    return {
        "maximum": int(values.max()),
        "mean": float(values.mean()),
        "minimum": int(values.min()),
        "p50": int(np.quantile(values, 0.50, method="higher")),
        "p95": int(np.quantile(values, 0.95, method="higher")),
        "p99": int(np.quantile(values, 0.99, method="higher")),
        "raw_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def canonical_benchmark_receipt_bytes(receipt: dict[str, object]) -> bytes:
    """Recompute derivable evidence and return sorted newline JSON."""

    expected_keys = {
        *_SHA_FIELDS,
        "claim_eligible",
        "dataset",
        "io",
        "latency_ns",
        "latency_raw_ns",
        "memory_ledger",
        "peak_rss_bytes",
        "per_query_hits",
        "query_count",
        "query_order_seed",
        "query_start",
        "return_width",
        "schema",
        "split",
        "stage_ns",
        "strict_id_recall_ppm",
        "thread_count",
        "truth_format",
        "warmup_queries",
    }
    if type(receipt) is not dict or set(receipt) != expected_keys:
        raise ValueError(_ERROR)
    query_count = receipt["query_count"]
    return_width = receipt["return_width"]
    hits = receipt["per_query_hits"]
    latency_raw = receipt["latency_raw_ns"]
    stage = receipt["stage_ns"]
    ledger = receipt["memory_ledger"]
    io = receipt["io"]
    if (
        receipt["schema"] != "sfora-factorized-residual-benchmark-v1"
        or receipt["claim_eligible"] is not False
        or type(receipt["dataset"]) is not str
        or not receipt["dataset"]
        or type(receipt["split"]) is not str
        or not receipt["split"]
        or type(query_count) is not int
        or query_count < 1
        or type(receipt["query_start"]) is not int
        or receipt["query_start"] < 0
        or type(return_width) is not int
        or return_width < 1
        or type(receipt["thread_count"]) is not int
        or not 1 <= receipt["thread_count"] <= 256
        or type(receipt["warmup_queries"]) is not int
        or receipt["warmup_queries"] < 0
        or receipt["query_order_seed"] is not None
        and type(receipt["query_order_seed"]) is not int
        or any(not _is_sha256(receipt[field]) for field in _SHA_FIELDS)
        or receipt["truth_format"] not in {"ids-u32", "ids-u32-distances-f32"}
        or type(hits) is not list
        or len(hits) != query_count
        or any(type(value) is not int or not 0 <= value <= return_width for value in hits)
        or type(latency_raw) is not list
        or len(latency_raw) != query_count
        or receipt["latency_ns"] != summarize_latency_ns(latency_raw)
        or type(stage) is not dict
        or set(stage) != {"candidate_raw", "candidate_total", "exact_raw", "exact_total"}
        or type(stage["candidate_raw"]) is not list
        or type(stage["exact_raw"]) is not list
        or len(stage["candidate_raw"]) != query_count
        or len(stage["exact_raw"]) != query_count
        or any(
            type(value) is not int or value < 0
            for value in (*stage["candidate_raw"], *stage["exact_raw"])
        )
        or stage["candidate_total"] != sum(stage["candidate_raw"])
        or stage["exact_total"] != sum(stage["exact_raw"])
        or receipt["strict_id_recall_ppm"]
        != sum(hits) * 1_000_000 // (query_count * return_width)
        or type(io) is not dict
        or set(io) != {"logical_bytes", "physical_bytes", "rows_scanned"}
        or any(type(value) is not int or value < 0 for value in io.values())
        or io["physical_bytes"] < io["logical_bytes"]
        or type(ledger) is not dict
        or set(ledger)
        != {
            "artifact_resident_bytes",
            "context_bytes",
            "fixed_service_overhead_bytes",
            "memory_limit_bytes",
            "safety_headroom_bytes",
            "total_reserved_bytes",
            "vector_store_resident_bytes",
        }
        or any(type(value) is not int or value < 0 for value in ledger.values())
        or ledger["total_reserved_bytes"]
        != ledger["artifact_resident_bytes"]
        + ledger["vector_store_resident_bytes"]
        + ledger["fixed_service_overhead_bytes"]
        + ledger["safety_headroom_bytes"]
        + ledger["context_bytes"]
        or ledger["total_reserved_bytes"] > ledger["memory_limit_bytes"]
        or type(receipt["peak_rss_bytes"]) is not int
        or receipt["peak_rss_bytes"] < 0
    ):
        raise ValueError(_ERROR)
    return (json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if "://" in value or not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute and local")
    return path


def _sha256_argument(value: str) -> str:
    if not _is_sha256(value):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the strict local-only benchmark interface."""

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--artifact", type=_absolute_path, required=True)
    parser.add_argument("--manifest-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--vectors", type=_absolute_path, required=True)
    parser.add_argument("--query", type=_absolute_path, required=True)
    parser.add_argument("--query-sha256", type=_sha256_argument, required=True)
    parser.add_argument("--truth", type=_absolute_path, required=True)
    parser.add_argument("--truth-sha256", type=_sha256_argument, required=True)
    parser.add_argument(
        "--truth-format",
        choices=("ids-u32", "ids-u32-distances-f32"),
        required=True,
    )
    parser.add_argument("--native-cache", type=_absolute_path, required=True)
    parser.add_argument("--output", type=_absolute_path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--query-start", type=int, required=True)
    parser.add_argument("--query-count", type=int, required=True)
    parser.add_argument("--query-order-seed", type=int)
    parser.add_argument("--warmup-queries", type=int, default=50)
    parser.add_argument("--thread-count", type=int, default=20)
    parser.add_argument("--memory-limit-bytes", type=int, default=3 * 1024**3)
    parser.add_argument("--fixed-service-overhead-bytes", type=int, default=128 * 1024**2)
    parser.add_argument("--safety-headroom-bytes", type=int, default=64 * 1024**2)
    parser.add_argument("--execute", action="store_true")
    parsed = parser.parse_args(arguments)
    if not parsed.execute:
        parser.error("--execute is required")
    return parsed


def run(args: argparse.Namespace) -> dict[str, object]:
    """Authenticate inputs and execute one serialized public-API evaluation."""

    if args.output.exists():
        raise ValueError(_ERROR)
    artifact = FactorizedResidualArtifact.open(
        args.artifact, manifest_sha256=args.manifest_sha256
    )
    store: DirectIoVectorStore | None = None
    backend = None
    index: FactorizedResidualIndex | None = None
    try:
        spec = artifact.spec
        query_dtype = np.dtype(np.uint8 if spec.vector_dtype == "uint8" else "<f4")
        query_matrix = _read_authenticated_matrix(
            args.query, query_dtype, args.query_sha256
        )
        truth_matrix = _read_authenticated_matrix(
            args.truth,
            np.dtype("<u4"),
            args.truth_sha256,
            trailing_dtype=(
                np.dtype("<f4")
                if args.truth_format == "ids-u32-distances-f32"
                else None
            ),
        )
        query_rows, query_dimensions = query_matrix.shape
        truth_rows, truth_width = truth_matrix.shape
        if (
            query_dimensions != spec.dimensions
            or truth_rows != query_rows
            or truth_width < spec.return_width
            or args.query_start < 0
            or args.query_count < 1
            or args.query_start + args.query_count > query_rows
            or not 0 <= args.warmup_queries <= args.query_count
        ):
            raise ValueError(_ERROR)
        queries = query_matrix[args.query_start : args.query_start + args.query_count]
        truth = truth_matrix[
            args.query_start : args.query_start + args.query_count,
            : spec.return_width,
        ]
        if args.query_order_seed is not None:
            order = np.random.default_rng(args.query_order_seed).permutation(args.query_count)
            queries = np.ascontiguousarray(queries[order])
            truth = np.ascontiguousarray(truth[order])
        store = DirectIoVectorStore(args.vectors, artifact.vector_store_identity)
        backend = compile_factorized_residual_backend(args.native_cache)
        index = FactorizedResidualIndex.open(
            artifact,
            store,
            candidate_backend=backend,
            thread_count=args.thread_count,
            memory_limit_bytes=args.memory_limit_bytes,
            fixed_service_overhead_bytes=args.fixed_service_overhead_bytes,
            safety_headroom_bytes=args.safety_headroom_bytes,
        )
        for query in queries[: args.warmup_queries]:
            index.search(np.array(query, dtype=query_dtype, order="C", copy=True))
        raw: list[int] = []
        candidate_raw: list[int] = []
        exact_raw: list[int] = []
        hits: list[int] = []
        logical_bytes = physical_bytes = rows_scanned = 0
        outputs = hashlib.sha256()
        for query, expected in zip(queries, truth, strict=True):
            owned_query = np.array(query, dtype=query_dtype, order="C", copy=True)
            started = time.perf_counter_ns()
            result = index.search(owned_query)
            raw.append(time.perf_counter_ns() - started)
            candidate_raw.append(result.candidate_ns)
            exact_raw.append(result.exact_ns)
            hits.append(len(set(map(int, result.ids)) & set(map(int, expected))))
            logical_bytes += result.vector_reads.logical_bytes
            physical_bytes += result.vector_reads.physical_bytes
            rows_scanned += result.candidates.evidence.rows_scanned
            outputs.update(result.ids.astype("<i8").tobytes())
            outputs.update(result.squared_distances.astype("<f4").tobytes())
        ledger = index.memory_ledger()
        receipt: dict[str, object] = {
            "artifact_manifest_sha256": args.manifest_sha256,
            "claim_eligible": False,
            "dataset": args.dataset,
            "io": {
                "logical_bytes": logical_bytes,
                "physical_bytes": physical_bytes,
                "rows_scanned": rows_scanned,
            },
            "latency_ns": summarize_latency_ns(raw),
            "latency_raw_ns": raw,
            "memory_ledger": {
                name: getattr(ledger, name) for name in ledger.__slots__
            },
            "native_binary_sha256": backend.binary_sha256,
            "native_source_sha256": backend.source_sha256,
            "outputs_sha256": outputs.hexdigest(),
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "per_query_hits": hits,
            "query_count": args.query_count,
            "query_order_seed": args.query_order_seed,
            "query_start": args.query_start,
            "query_sha256": args.query_sha256,
            "return_width": spec.return_width,
            "schema": "sfora-factorized-residual-benchmark-v1",
            "split": args.split,
            "stage_ns": {
                "candidate_raw": candidate_raw,
                "candidate_total": sum(candidate_raw),
                "exact_raw": exact_raw,
                "exact_total": sum(exact_raw),
            },
            "strict_id_recall_ppm": sum(hits) * 1_000_000 // (args.query_count * spec.return_width),
            "thread_count": args.thread_count,
            "truth_sha256": args.truth_sha256,
            "truth_format": args.truth_format,
            "vector_sha256": artifact.vector_store_identity.sha256,
            "warmup_queries": args.warmup_queries,
        }
        return receipt
    finally:
        if index is not None:
            index.close()
        else:
            if store is not None:
                store.close()
            artifact.close()
            if backend is not None:
                backend.close()


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    receipt = run(args)
    wire = canonical_benchmark_receipt_bytes(receipt)
    with args.output.open("xb") as stream:
        stream.write(wire)
        stream.flush()
        os.fsync(stream.fileno())
    print(wire.decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
