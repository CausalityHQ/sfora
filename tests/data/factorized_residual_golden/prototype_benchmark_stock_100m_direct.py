from __future__ import annotations

import ctypes
import hashlib
import json
import os
import resource
import time
from pathlib import Path

import faiss
import numpy as np


INDEX = Path(os.environ["SFORA_INDEX"])
BASE = Path(os.environ["SFORA_BASE"])
QUERY = Path(os.environ["SFORA_QUERY"])
TRUTH = Path(os.environ["SFORA_TRUTH"])
OUT = Path(os.environ["SFORA_OUT"])
START = int(os.environ.get("SFORA_QUERY_START", "0"))
COUNT = int(os.environ.get("SFORA_QUERY_COUNT", "1000"))
NPROBE = int(os.environ.get("SFORA_NPROBE", "640"))
SHORTLIST = int(os.environ.get("SFORA_SHORTLIST", "1024"))
ROWS, DIMENSIONS, RETURN = 100_000_000, 128, 100
LIBRARY = Path("/tmp/libone_lut_scanner.so")


def latency(values: list[int]) -> dict[str, int | float | str]:
    data = np.asarray(values, dtype=np.int64)
    return {
        "minimum": int(data.min()),
        "p50": int(np.quantile(data, 0.50, method="higher")),
        "p95": int(np.quantile(data, 0.95, method="higher")),
        "p99": int(np.quantile(data, 0.99, method="higher")),
        "maximum": int(data.max()),
        "mean": float(data.mean()),
        "raw_sha256": hashlib.sha256(data.tobytes()).hexdigest(),
    }


def rss_bytes() -> int:
    pages = int(Path("/proc/self/statm").read_text().split()[1])
    return pages * os.sysconf("SC_PAGE_SIZE")


def main() -> None:
    if OUT.exists():
        raise FileExistsError(OUT)
    faiss.omp_set_num_threads(20)
    index = faiss.read_index(str(INDEX))
    if index.ntotal != ROWS or index.d != DIMENSIONS or not index.by_residual:
        raise ValueError("index authority differs")
    index.nprobe = NPROBE
    quantizer = faiss.downcast_index(index.quantizer)
    quantizer.hnsw.efSearch = 64
    coarse = np.ascontiguousarray(quantizer.reconstruct_n(0, index.nlist), dtype=np.float32)
    exact_coarse = faiss.IndexFlatL2(DIMENSIONS)
    exact_coarse.add(coarse)
    index.use_precomputed_table = 1
    index.precompute_table()
    rss_after_load = rss_bytes()
    queries = np.ascontiguousarray(
        np.memmap(QUERY, np.uint8, "r", offset=8, shape=(10_000, DIMENSIONS))[
            START : START + COUNT
        ],
        dtype=np.float32,
    )
    truth = np.memmap(TRUTH, np.uint32, "r", offset=8, shape=(10_000, 100))[
        START : START + COUNT
    ]
    library = ctypes.CDLL(str(LIBRARY))
    rerank = library.exact_rerank_direct
    fp = ctypes.POINTER(ctypes.c_float)
    i64p = ctypes.POINTER(ctypes.c_int64)
    rerank.argtypes = [
        fp,
        i64p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        i64p,
        fp,
    ]
    rerank.restype = ctypes.c_int
    output_ids = np.empty(RETURN, dtype=np.int64)
    output_distances = np.empty(RETURN, dtype=np.float32)
    direct_fd = os.open(BASE, os.O_RDONLY | os.O_DIRECT)

    def search(query: np.ndarray) -> tuple[int, int]:
        began = time.perf_counter_ns()
        coarse_distances, coarse_lists = exact_coarse.search(query[None], NPROBE)
        _, found = index.search_preassigned(
            query[None], SHORTLIST, coarse_lists, coarse_distances
        )
        candidate_done = time.perf_counter_ns()
        candidate_ids = np.ascontiguousarray(found[0], dtype=np.int64)
        status = rerank(
            query.ctypes.data_as(fp),
            candidate_ids.ctypes.data_as(i64p),
            SHORTLIST,
            DIMENSIONS,
            RETURN,
            direct_fd,
            1,
            output_ids.ctypes.data_as(i64p),
            output_distances.ctypes.data_as(fp),
        )
        if status:
            raise RuntimeError(f"direct reranker status {status}")
        ended = time.perf_counter_ns()
        return candidate_done - began, ended - began

    for query in queries[:50]:
        search(query)
    candidate_ns: list[int] = []
    end_to_end_ns: list[int] = []
    hits = 0
    outputs = hashlib.sha256()
    for ordinal, (query, expected) in enumerate(zip(queries, truth, strict=True)):
        candidate_elapsed, total_elapsed = search(query)
        candidate_ns.append(candidate_elapsed)
        end_to_end_ns.append(total_elapsed)
        hits += len(set(map(int, output_ids)) & set(map(int, expected)))
        outputs.update(output_ids.tobytes())
        outputs.update(output_distances.tobytes())
        if ordinal % 100 == 99:
            print(f"progress={ordinal + 1}", flush=True)
    os.close(direct_fd)
    payload = {
        "schema": "sfora-stock-ivfpq-direct-refine-baseline-v1",
        "claim_eligible": False,
        "dataset": "bigann100m",
        "split": f"query ordinals {START}..{START + COUNT - 1}",
        "strict_id_recall_at_100": hits / (COUNT * RETURN),
        "candidate_latency_ns": latency(candidate_ns),
        "batch1_end_to_end_latency_ns": latency(end_to_end_ns),
        "protocol": {
            "rows": ROWS,
            "dimensions": DIMENSIONS,
            "nlist": index.nlist,
            "nprobe": NPROBE,
            "pq": "32x6bit residual",
            "shortlist": SHORTLIST,
            "return": RETURN,
            "threads": 20,
            "coarse_router": "exact flat all centroids",
            "exact_refinement": "same io_uring O_DIRECT helper",
        },
        "serialized_index_bytes": INDEX.stat().st_size,
        "rss_after_load_bytes": rss_after_load,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "outputs_sha256": outputs.hexdigest(),
        "library_sha256": hashlib.sha256(LIBRARY.read_bytes()).hexdigest(),
    }
    wire = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    OUT.write_bytes(wire)
    print(wire.decode(), end="")


if __name__ == "__main__":
    main()
