#!/usr/bin/env python3
"""GPU parity check: sfora.so400m_compact_serving vs one So400m training run's export and receipt.

Encodes all 59,551 SOP TRAIN images through the library (batch 64, archive row order), reports bitwise
equality with <run-dir>/train_embeddings.npy, packs, searches the 5,851 held queries against the full
packed gallery (self excluded) and requires per-query top-1 correctness == receipt quality.per_query_r1.
Also times end-to-end encode img/s (disk read + preprocess + forward) and single-image latency
(in-memory PIL, preprocess + forward + host copy) p50/p99 over 200 images after warmup.
Exit 1 iff per-query R@1 differs. Not run at authoring time.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.so400m_compact_serving import BATCH_SIZE, So400mCompactEncoder, So400mCompactIndex

FIT_SEED = 179019
TRAIN_ROWS = 59_551
HELD_ROWS = 5_851


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:  # same as train_sop_siglip2_substrate_pair.ImageRows
        return image.convert("RGB")


class Rows(Dataset):  # type: ignore[misc]
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Image.Image:
        return load_rgb(self.paths[index])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--resolution", type=int, choices=(280, 384), required=True)
    p.add_argument("--model-snapshot", type=Path, required=True)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists() or not torch.cuda.is_available():
        raise SystemExit("output exists or CUDA unavailable")
    receipt = json.loads((a.run_dir / "receipt.json").read_text())
    quality = receipt["quality"]
    output_dim = int(quality.get("output_width", 128))
    if receipt.get("substrate_config", {}).get("crop") not in (None, a.resolution):
        raise SystemExit("receipt crop differs from --resolution")
    with np.load(a.archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"]).astype(str)
    if labels.shape != (TRAIN_ROWS,) or relatives.shape != (TRAIN_ROWS,):
        raise SystemExit("SOP TRAIN inventory differs")
    held = np.asarray(
        deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=FIT_SEED)
        .validation_row_indexes, dtype=np.int64)
    expected_r1 = np.asarray(quality["per_query_r1"], dtype=np.float64)
    if held.shape != (HELD_ROWS,) or expected_r1.shape != (HELD_ROWS,):
        raise SystemExit("SOP held partition differs")
    paths = [a.dataset_root / relative for relative in relatives]

    device = torch.device("cuda")
    encoder = So400mCompactEncoder.from_checkpoint(
        a.model_snapshot, a.run_dir / "checkpoint.pt",
        resolution=a.resolution, output_dim=output_dim, device=device)

    loader = DataLoader(Rows(paths), batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
                        collate_fn=list)
    torch.cuda.synchronize()
    started = time.perf_counter()
    values = torch.cat([encoder.encode_images(batch) for batch in loader])
    encode_seconds = time.perf_counter() - started

    reference = np.load(a.run_dir / "train_embeddings.npy", allow_pickle=False)
    library = values.numpy()
    bitwise = bool(reference.shape == library.shape and reference.dtype == library.dtype
                   and np.array_equal(reference.view(np.uint32), library.view(np.uint32)))
    max_abs = float(np.max(np.abs(reference.astype(np.float64) - library.astype(np.float64))))
    differing_rows = int(np.any(reference != library, axis=1).sum())

    packed = pack_int8_unit_embeddings(values)
    reference_packed = pack_int8_unit_embeddings(torch.from_numpy(reference.copy()))
    packed_equal = bool(torch.equal(packed.codes, reference_packed.codes)
                        and torch.equal(packed.inverse_norms, reference_packed.inverse_norms))
    index = So400mCompactIndex(packed, device=device)
    queries = type(packed)(packed.codes[held].contiguous(), packed.inverse_norms[held].contiguous())
    started = time.perf_counter()
    ordinals, _scores = index.search(queries, 1, exclude=held)
    search_seconds = time.perf_counter() - started
    observed_r1 = (labels[ordinals[:, 0]] == labels[held]).astype(np.float64)
    r1_equal = bool(np.array_equal(observed_r1, expected_r1))

    latency_images = [load_rgb(path) for path in paths[:220]]
    for image in latency_images[:20]:
        encoder.encode_images([image])
    latencies = []
    for image in latency_images[20:]:
        torch.cuda.synchronize()
        t = time.perf_counter()
        encoder.encode_images([image])  # returns a CPU tensor, so the device work is complete
        latencies.append(time.perf_counter() - t)
    latency_ms = np.asarray(latencies) * 1e3

    result = {
        "schema": "sfora-so400m-serving-parity-v1",
        "run_dir": str(a.run_dir), "resolution": a.resolution, "output_dim": output_dim,
        "rows": TRAIN_ROWS, "held_queries": HELD_ROWS,
        "train_embeddings_bitwise_equal": bitwise,
        "train_embeddings_max_abs_diff": max_abs,
        "train_embeddings_differing_rows": differing_rows,
        "packed_codes_equal": packed_equal,
        "per_query_r1_equal": r1_equal,
        "per_query_r1_mismatches": int((observed_r1 != expected_r1).sum()),
        "observed_r1": float(observed_r1.mean()), "receipt_r1": float(expected_r1.mean()),
        "receipt_recall_at_1": quality.get("recall_at_1"),
        "encode_seconds": encode_seconds, "encode_images_per_second": TRAIN_ROWS / encode_seconds,
        "encode_batch_size": BATCH_SIZE, "search_seconds": search_seconds,
        "single_image_latency_ms": {"p50": float(np.percentile(latency_ms, 50)),
                                    "p99": float(np.percentile(latency_ms, 99)),
                                    "count": len(latencies), "warmup": 20},
        "gallery_index_bytes": len(index.to_bytes()),
        "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("x") as stream:
        stream.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    print(json.dumps({k: result[k] for k in ("train_embeddings_bitwise_equal", "train_embeddings_max_abs_diff",
                                             "per_query_r1_equal", "encode_images_per_second")}), flush=True)
    return 0 if r1_equal else 1


if __name__ == "__main__":
    sys.exit(main())
