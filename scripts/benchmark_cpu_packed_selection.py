"""Paired local diagnostic for the exact CPU packed-search selector."""

from __future__ import annotations

import hashlib
import json
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import sfora.packed_int8_search as packed_search
from sfora.joint_relational_compaction import pack_int8_unit_embeddings


def full_sort(scores: np.ndarray, ordinals: np.ndarray, k: int) -> np.ndarray:
    """The previous selector, retained here as the paired control."""

    return np.lexsort((ordinals, -scores))[:k]


def main() -> None:
    if packed_search.__file__ is None:
        raise ValueError("CPU packed search source path is unavailable")
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(179019)
    gallery = pack_int8_unit_embeddings(
        torch.nn.functional.normalize(torch.randn(59_519, 128, generator=generator), dim=1)
    )
    index = packed_search.CpuPackedInt8Gallery.open_packed(gallery)
    partition = packed_search._ordered_topk_indexes
    cases: dict[str, object] = {}
    receipt: dict[str, object] = {
        "schema": "sfora-cpu-packed-selection-synthetic-v1",
        "claim_eligible": False,
        "gallery_rows": 59_519,
        "dimensions": 128,
        "top_k": 10,
        "blocks": 10,
        "torch_threads": torch.get_num_threads(),
        "cpu": platform.processor(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(Path(packed_search.__file__).read_bytes()).hexdigest(),
        "cases": cases,
    }
    try:
        for batch in (1, 32):
            queries = pack_int8_unit_embeddings(
                torch.nn.functional.normalize(torch.randn(batch, 128, generator=generator), dim=1)
            )
            packed_search._ordered_topk_indexes = partition
            selected = index.search_packed(queries)
            packed_search._ordered_topk_indexes = full_sort
            control = index.search_packed(queries)
            if not (
                np.array_equal(selected[0], control[0]) and np.array_equal(selected[1], control[1])
            ):
                raise ValueError("paired CPU packed search differs")
            samples: dict[str, list[int]] = {"full_sort": [], "partition": []}
            for block in range(10):
                order = (
                    (("full_sort", full_sort), ("partition", partition))
                    if block % 2 == 0
                    else (("partition", partition), ("full_sort", full_sort))
                )
                for name, selector in order:
                    packed_search._ordered_topk_indexes = selector
                    started = time.perf_counter_ns()
                    index.search_packed(queries)
                    samples[name].append(time.perf_counter_ns() - started)
            cases[str(batch)] = {
                "exact_ordinal_and_score_parity": True,
                "samples_ns": samples,
                "median_ms": {
                    name: statistics.median(values) / 1_000_000 for name, values in samples.items()
                },
            }
    finally:
        packed_search._ordered_topk_indexes = partition
    print(json.dumps(receipt, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
