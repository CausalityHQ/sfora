"""CPU-only train-descriptor probe of expanded float32 Euclidean ranking."""

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

ARCHIVE = Path(sys.argv[1])
OUTPUT = Path(sys.argv[2])
EXPECTED = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"

if OUTPUT.exists():
    raise ValueError("precision probe output already exists")
torch.set_num_threads(2)
started = time.perf_counter()
with ARCHIVE.open("rb") as stream:
    digest = hashlib.file_digest(stream, "sha256").hexdigest()
if digest != EXPECTED:
    raise ValueError("authenticated SOP train feature archive differs")
with np.load(ARCHIVE, allow_pickle=False) as archive:
    raw = archive["train_embeddings"]
    labels = archive["train_labels"]
    if raw.shape != (59551, 768) or raw.dtype != np.float32 or labels.shape != (59551,):
        raise ValueError("SOP train descriptor inventory differs")
    values = F.normalize(torch.from_numpy(raw.copy()), dim=1)[:, :512].contiguous()
gallery_squared = (values * values).sum(dim=1)
values64 = values.double()
gallery_squared64 = (values64 * values64).sum(dim=1)
queries = np.linspace(0, len(values) - 1, num=1024, dtype=np.int64)
q32 = values[queries]
q64 = values64[queries]
batch32 = 2 * (q32 @ values.T) - gallery_squared[None, :]
batch64 = 2 * (q64 @ values64.T) - gallery_squared64[None, :]
records = []
for local, ordinal in enumerate(queries.tolist()):
    single32 = 2 * (q32[local : local + 1] @ values.T)[0] - gallery_squared
    row32 = batch32[local]
    row64 = batch64[local]
    single32[ordinal] = -torch.inf
    row32[ordinal] = -torch.inf
    row64[ordinal] = -torch.inf
    selected = [int(row.argmax()) for row in (single32, row32, row64)]
    records.append(
        {
            "query_ordinal": ordinal,
            "single_float32_top1": selected[0],
            "batch_float32_top1": selected[1],
            "batch_float64_top1": selected[2],
            "single_float32_correct": bool(labels[selected[0]] == labels[ordinal]),
            "batch_float32_correct": bool(labels[selected[1]] == labels[ordinal]),
            "batch_float64_correct": bool(labels[selected[2]] == labels[ordinal]),
            "single_vs_batch_score_bits_differ": bool(not torch.equal(single32, row32)),
            "float64_winner_distance_squared": float(
                (q64[local] - values64[selected[2]]).square().sum()
            ),
        }
    )
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
result = {
    "schema": "sfora-sop-train-prefix-precision-probe-v1",
    "claim_eligible": False,
    "archive_sha256": digest,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    "source_split": "SOP train only; pretrained UNICOM ViT-B/16 descriptors",
    "query_count": len(records),
    "gallery_rows": len(values),
    "prefix_dimensions": 512,
    "single_float32_vs_float64_top1_differences": sum(
        r["single_float32_top1"] != r["batch_float64_top1"] for r in records
    ),
    "batch_float32_vs_float64_top1_differences": sum(
        r["batch_float32_top1"] != r["batch_float64_top1"] for r in records
    ),
    "single_vs_batch_float32_top1_differences": sum(
        r["single_float32_top1"] != r["batch_float32_top1"] for r in records
    ),
    "single_float32_r1": sum(r["single_float32_correct"] for r in records) / len(records),
    "batch_float32_r1": sum(r["batch_float32_correct"] for r in records) / len(records),
    "batch_float64_r1": sum(r["batch_float64_correct"] for r in records) / len(records),
    "elapsed_seconds": time.perf_counter() - started,
    "python": platform.python_version(),
    "torch": torch.__version__,
    "records": records,
}
OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({k: v for k, v in result.items() if k != "records"}), flush=True)
