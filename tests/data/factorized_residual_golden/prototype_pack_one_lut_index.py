from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import faiss
import numpy as np


INDEX = Path(os.environ["SFORA_INDEX"])
OUT = Path(os.environ["SFORA_PACKED_DIR"])
EXPECTED_ROWS = int(os.environ["SFORA_ROWS"])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    index = faiss.read_index(str(INDEX))
    if not index.by_residual or index.ntotal != EXPECTED_ROWS or index.pq.nbits != 6:
        raise ValueError("source index authority differs")
    nlist, dimensions, subquantizers = index.nlist, index.d, index.pq.M
    coarse = np.ascontiguousarray(
        faiss.downcast_index(index.quantizer).reconstruct_n(0, nlist), dtype=np.float32
    )
    pq = np.ascontiguousarray(
        faiss.vector_to_array(index.pq.centroids).reshape(subquantizers, 64, -1),
        dtype=np.float32,
    )
    offsets = np.zeros(nlist + 1, dtype=np.uint64)
    for list_id in range(nlist):
        offsets[list_id + 1] = offsets[list_id] + index.invlists.list_size(list_id)
    if int(offsets[-1]) != EXPECTED_ROWS:
        raise ValueError("posting count differs")
    coarse.tofile(OUT / "coarse.f32")
    pq.tofile(OUT / "pq.f32")
    offsets.tofile(OUT / "offsets.u64")
    ids = np.memmap(OUT / "ids.u32", np.uint32, "w+", shape=(EXPECTED_ROWS,))
    codes = np.memmap(
        OUT / "codes.u8", np.uint8, "w+", shape=(EXPECTED_ROWS, index.code_size)
    )
    norms = np.memmap(OUT / "norms.u8", np.uint8, "w+", shape=(EXPECTED_ROWS,))
    norm_low = np.memmap(OUT / "norm-low.f32", np.float32, "w+", shape=(nlist,))
    norm_scale = np.memmap(OUT / "norm-scale.f32", np.float32, "w+", shape=(nlist,))
    for list_id in range(nlist):
        begin, end = map(int, offsets[list_id : list_id + 2])
        size = end - begin
        if size == 0:
            norm_low[list_id] = 0.0
            norm_scale[list_id] = 0.0
            continue
        raw_ids = faiss.rev_swig_ptr(index.invlists.get_ids(list_id), size)
        if raw_ids.min() < 0 or raw_ids.max() > np.iinfo(np.uint32).max:
            raise ValueError("u32 id authority differs")
        list_codes = faiss.rev_swig_ptr(
            index.invlists.get_codes(list_id), size * index.code_size
        ).reshape(size, index.code_size)
        ids[begin:end] = raw_ids
        codes[begin:end] = list_codes
        reconstructed = index.pq.decode(np.ascontiguousarray(list_codes)) + coarse[list_id]
        squared_norms = np.einsum("ij,ij->i", reconstructed, reconstructed)
        low, high = float(squared_norms.min()), float(squared_norms.max())
        scale = (high - low) / 255.0
        norm_low[list_id] = low
        norm_scale[list_id] = scale
        norms[begin:end] = np.rint(
            np.divide(
                squared_norms - low,
                scale,
                out=np.zeros_like(squared_norms),
                where=scale > 0,
            )
        ).clip(0, 255).astype(np.uint8)
        if list_id % 4096 == 4095:
            print(f"lists={list_id + 1} rows={end}", flush=True)
    for array in (ids, codes, norms, norm_low, norm_scale):
        array.flush()
    del ids, codes, norms, norm_low, norm_scale
    roles = {}
    for path in sorted(OUT.iterdir()):
        if path.name == "manifest.json":
            continue
        roles[path.name] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    manifest = {
        "schema": "sfora-factorized-residual-adc-packed-v1",
        "claim_eligible": False,
        "source_index_sha256": sha256(INDEX),
        "rows": EXPECTED_ROWS,
        "dimensions": dimensions,
        "nlist": nlist,
        "subquantizers": subquantizers,
        "bits_per_subquantizer": 6,
        "code_bytes": index.code_size,
        "id_bytes": 4,
        "norm_bytes": 1,
        "roles": roles,
        "total_bytes": sum(role["bytes"] for role in roles.values()),
    }
    wire = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
    (OUT / "manifest.json").write_bytes(wire)
    print(wire.decode(), end="")


if __name__ == "__main__":
    main()
