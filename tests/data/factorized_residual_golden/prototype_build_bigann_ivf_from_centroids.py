from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import faiss
import numpy as np


ROWS = int(os.environ.get("SFORA_ROWS", "10000000"))
NLIST = int(os.environ.get("SFORA_NLIST", "32768"))
TRAIN_ROWS = int(os.environ.get("SFORA_TRAIN_ROWS", "1277952"))
BASE = Path(os.environ.get("SFORA_BASE", "/home/riomus/datasets/bigann/base.first10m.raw.u8bin"))
CENTROIDS = Path(os.environ.get("SFORA_CENTROIDS", f"/tmp/sfora-bigann-ivf{NLIST}-centroids-seed50.npy"))
INDEX_OUT = Path(os.environ.get("SFORA_INDEX_OUT", f"/tmp/sfora-bigann{ROWS}-ivf{NLIST}-hnsw32-residual-pq32x6-seed50.index"))
RECEIPT = INDEX_OUT.with_suffix(".json")
SEED, DIMS, M, NBITS = 50, 128, 32, 6


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    if INDEX_OUT.exists() or RECEIPT.exists():
        raise FileExistsError("output already exists")
    faiss.omp_set_num_threads(20)
    base = np.memmap(BASE, np.uint8, "r", offset=8, shape=(ROWS, DIMS))
    centers = np.load(CENTROIDS, allow_pickle=False)
    if centers.shape != (NLIST, DIMS) or centers.dtype != np.float32:
        raise ValueError("centroid authority differs")
    quantizer = faiss.IndexHNSWFlat(DIMS, 32, faiss.METRIC_L2)
    quantizer.hnsw.efConstruction = 80
    began = time.monotonic()
    quantizer.add(centers)
    graph_seconds = time.monotonic() - began
    quantizer.hnsw.efSearch = 128
    rng = np.random.default_rng(SEED)
    train_ids = np.sort(rng.choice(ROWS, TRAIN_ROWS, replace=False))
    train = np.ascontiguousarray(base[train_ids], dtype=np.float32)
    began = time.monotonic()
    _, assignments = quantizer.search(train, 1)
    residual = train - centers[assignments[:, 0]]
    assign_train_seconds = time.monotonic() - began
    pq = faiss.ProductQuantizer(DIMS, M, NBITS)
    pq.cp.seed = SEED
    pq.cp.niter = 25
    began = time.monotonic()
    pq.train(np.ascontiguousarray(residual, dtype=np.float32))
    pq_seconds = time.monotonic() - began
    del train, residual, assignments
    index = faiss.IndexIVFPQ(quantizer, DIMS, NLIST, M, NBITS, faiss.METRIC_L2)
    index.by_residual = True
    index.pq = pq
    index.is_trained = True
    quantizer.hnsw.efSearch = 64
    began = time.monotonic()
    for start in range(0, ROWS, 100_000):
        block = np.ascontiguousarray(base[start : start + 100_000], dtype=np.float32)
        index.add_with_ids(block, np.arange(start, start + len(block), dtype=np.int64))
        if start % 1_000_000 == 0:
            print(f"add={start + len(block)}", flush=True)
    add_seconds = time.monotonic() - began
    faiss.write_index(index, str(INDEX_OUT))
    payload = {
        "schema": "sfora-bigann-ivf-from-centroids-dev-v1",
        "claim_eligible": False,
        "source": str(BASE),
        "rows": ROWS,
        "dimensions": DIMS,
        "train_rows": TRAIN_ROWS,
        "seed": SEED,
        "nlist": NLIST,
        "quantizer": "HNSW32 efConstruction80 add-efSearch64",
        "pq": "32x6bit residual",
        "construction_seconds": {
            "hnsw_graph": graph_seconds,
            "training_assignment": assign_train_seconds,
            "pq": pq_seconds,
            "add": add_seconds,
        },
        "serialized_index_bytes": INDEX_OUT.stat().st_size,
        "index_sha256": sha256(INDEX_OUT),
        "centroids_sha256": sha256(CENTROIDS),
    }
    wire = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    RECEIPT.write_bytes(wire)
    print(wire.decode(), end="")


if __name__ == "__main__":
    main()
