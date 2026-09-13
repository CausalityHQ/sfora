from __future__ import annotations

import ctypes
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import numpy as np


PACKED = Path(os.environ["SFORA_PACKED_DIR"])
BASE = Path(os.environ["SFORA_BASE"])
QUERY = Path(os.environ["SFORA_QUERY"])
TRUTH = Path(os.environ["SFORA_TRUTH"])
OUT = Path(os.environ["SFORA_OUT"])
START = int(os.environ.get("SFORA_QUERY_START", "0"))
COUNT = int(os.environ.get("SFORA_QUERY_COUNT", "1000"))
ORDER_SEED = int(os.environ.get("SFORA_QUERY_ORDER_SEED", "-1"))
NPROBE = int(os.environ["SFORA_NPROBE"])
SHORTLIST = int(os.environ.get("SFORA_SHORTLIST", "1024"))
RETURN = 100
LIBRARY = Path("/tmp/libone_lut_scanner.so")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def latency(values: list[int]) -> dict[str, int | float | str]:
    a = np.asarray(values, dtype=np.int64)
    return {
        "minimum": int(a.min()),
        "p50": int(np.quantile(a, 0.50, method="higher")),
        "p95": int(np.quantile(a, 0.95, method="higher")),
        "p99": int(np.quantile(a, 0.99, method="higher")),
        "maximum": int(a.max()),
        "mean": float(a.mean()),
        "raw_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def rss_bytes() -> int:
    resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
    return resident_pages * os.sysconf("SC_PAGE_SIZE")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    manifest_bytes = (PACKED / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest["schema"] != "sfora-factorized-residual-adc-packed-v1":
        raise ValueError("manifest schema differs")
    for name, role in manifest["roles"].items():
        path = PACKED / name
        if path.stat().st_size != role["bytes"] or sha256(path) != role["sha256"]:
            raise ValueError(f"packed role differs: {name}")
    rows = int(manifest["rows"])
    dimensions = int(manifest["dimensions"])
    nlist = int(manifest["nlist"])
    subquantizers = int(manifest["subquantizers"])
    code_bytes = int(manifest["code_bytes"])
    coarse = np.memmap(PACKED / "coarse.f32", np.float32, "r", shape=(nlist, dimensions))
    pq = np.memmap(PACKED / "pq.f32", np.float32, "r", shape=(subquantizers, 64, dimensions // subquantizers))
    offsets = np.memmap(PACKED / "offsets.u64", np.uint64, "r", shape=(nlist + 1,))
    ids = np.memmap(PACKED / "ids.u32", np.uint32, "r", shape=(rows,))
    codes = np.memmap(PACKED / "codes.u8", np.uint8, "r", shape=(rows, code_bytes))
    norms = np.memmap(PACKED / "norms.u8", np.uint8, "r", shape=(rows,))
    norm_low = np.memmap(PACKED / "norm-low.f32", np.float32, "r", shape=(nlist,))
    norm_scale = np.memmap(PACKED / "norm-scale.f32", np.float32, "r", shape=(nlist,))
    residency_checksum = 0
    for array in (coarse, pq, offsets, ids, codes, norms, norm_low, norm_scale):
        raw = array.reshape(-1).view(np.uint8)
        for begin in range(0, raw.size, 64 << 20):
            residency_checksum ^= int(raw[begin : begin + (64 << 20)].sum(dtype=np.uint64))
    rss_after_prefault = rss_bytes()
    queries = np.ascontiguousarray(
        np.memmap(QUERY, np.uint8, "r", offset=8, shape=(10_000, dimensions))[START : START + COUNT],
        dtype=np.float32,
    )
    truth = np.memmap(TRUTH, np.uint32, "r", offset=8, shape=(10_000, 100))[
        START : START + COUNT
    ]
    if ORDER_SEED >= 0:
        order = np.random.default_rng(ORDER_SEED).permutation(COUNT)
        queries = np.ascontiguousarray(queries[order])
        truth = np.ascontiguousarray(truth[order])
    library = ctypes.CDLL(str(LIBRARY))
    search = library.one_lut_search
    fp = ctypes.POINTER(ctypes.c_float)
    u8p = ctypes.POINTER(ctypes.c_uint8)
    u64p = ctypes.POINTER(ctypes.c_uint64)
    i64p = ctypes.POINTER(ctypes.c_int64)
    search.argtypes = [
        fp, fp, fp, u64p, ctypes.c_void_p, ctypes.c_int, u8p, u8p, fp, fp,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int,
        ctypes.c_int, i64p, fp,
    ]
    search.restype = ctypes.c_int
    output_ids = np.empty(RETURN, dtype=np.int64)
    output_distances = np.empty(RETURN, dtype=np.float32)
    direct_fd = os.open(BASE, os.O_RDONLY | os.O_DIRECT)

    def run(query: np.ndarray) -> None:
        status = search(
            query.ctypes.data_as(fp), coarse.ctypes.data_as(fp), pq.ctypes.data_as(fp),
            offsets.ctypes.data_as(u64p), ctypes.c_void_p(ids.ctypes.data), 1,
            codes.ctypes.data_as(u8p), norms.ctypes.data_as(u8p),
            norm_low.ctypes.data_as(fp), norm_scale.ctypes.data_as(fp), nlist,
            dimensions, subquantizers, 6, NPROBE, SHORTLIST, RETURN, 2,
            None, direct_fd, 1, output_ids.ctypes.data_as(i64p),
            output_distances.ctypes.data_as(fp),
        )
        if status:
            raise RuntimeError(f"native scanner status {status}")

    for query in queries[:50]:
        run(query)
    elapsed: list[int] = []
    hits = 0
    output_hash = hashlib.sha256()
    for ordinal, (query, expected) in enumerate(zip(queries, truth, strict=True)):
        began = time.perf_counter_ns()
        run(query)
        elapsed.append(time.perf_counter_ns() - began)
        hits += len(set(map(int, output_ids)) & set(map(int, expected)))
        output_hash.update(output_ids.tobytes())
        output_hash.update(output_distances.tobytes())
        if ordinal % 100 == 99:
            print(f"progress={ordinal + 1}", flush=True)
    os.close(direct_fd)
    payload = {
        "schema": "sfora-factorized-adc-direct-refine-scale-v1",
        "claim_eligible": False,
        "dataset": "bigann100m",
        "split": f"query ordinals {START}..{START + COUNT - 1}",
        "query_order_seed": None if ORDER_SEED < 0 else ORDER_SEED,
        "strict_id_recall_at_100": hits / (COUNT * RETURN),
        "latency_ns": latency(elapsed),
        "protocol": {
            "rows": rows,
            "dimensions": dimensions,
            "nlist": nlist,
            "nprobe": NPROBE,
            "pq": "32x6bit residual",
            "shortlist": SHORTLIST,
            "return": RETURN,
            "exact_refinement": "io_uring O_DIRECT",
        },
        "packed_index_bytes": manifest["total_bytes"],
        "rss_after_prefault_bytes": rss_after_prefault,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "residency_checksum": residency_checksum,
        "outputs_sha256": output_hash.hexdigest(),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "library_sha256": sha256(LIBRARY),
    }
    wire = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(wire)
    print(wire.decode(), end="")


if __name__ == "__main__":
    main()
