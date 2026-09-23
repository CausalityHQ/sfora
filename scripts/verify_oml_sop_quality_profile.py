#!/usr/bin/env python3
"""Revalidate the bundled OML SOP profile with the exact 130-byte packed score."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.compact_metric import _compact_metric_lexicographic_topk
from sfora.model_profiles import load_oml_sop_compact_encoder

FEATURE_SHA256 = "8f565027b20e55923826a5240d97c564171428a5f1cc7290e680d2223fd28da4"
MODEL_SHA256 = "2701830538f31bd2dabb06622475cc889b1095580fc57218d0293e7a6bce53a7"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--external-checkpoint", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.features) != FEATURE_SHA256:
        raise ValueError("SOP profile feature authority differs")
    if sha256(args.external_checkpoint) != MODEL_SHA256:
        raise ValueError("SOP external checkpoint differs")
    if not torch.cuda.is_available():
        raise RuntimeError("SOP quality verification requires CUDA")
    started = time.perf_counter()
    with np.load(args.features, allow_pickle=False) as archive:
        values = torch.from_numpy(np.ascontiguousarray(archive["test_features"]))
        labels = torch.from_numpy(np.ascontiguousarray(archive["test_labels"]))
    if values.shape != (60_502, 384) or labels.shape != (60_502,):
        raise ValueError("SOP profile test inventory differs")
    torch.cuda.reset_peak_memory_stats()
    profile = load_oml_sop_compact_encoder()
    packed = profile.encode_packed(values)
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = False
    codes = packed.codes.float().to(device)
    inverse = packed.inverse_norms.float().to(device)
    label_gpu = labels.to(device)
    _, inverse_label, counts = torch.unique(
        label_gpu, sorted=True, return_inverse=True, return_counts=True
    )
    relevant = counts[inverse_label] - 1
    if int(relevant.min()) < 1:
        raise ValueError("SOP profile positive inventory differs")
    retained = int(relevant.max())
    aps: list[float] = []
    hits: list[int] = []
    with torch.inference_mode():
        for start in range(0, len(codes), 256):
            stop = min(start + 256, len(codes))
            scores = (codes[start:stop] @ codes.T) * inverse[start:stop, None] * inverse[None, :]
            scores[
                torch.arange(stop - start, device=device),
                torch.arange(start, stop, device=device),
            ] = -torch.inf
            ranking = _compact_metric_lexicographic_topk(scores, retained)
            matches = label_gpu[ranking] == label_gpu[start:stop, None]
            ranks = torch.arange(1, retained + 1, device=device, dtype=torch.float64)
            precision = torch.cumsum(matches, dim=1).to(torch.float64) / ranks
            mask = torch.arange(retained, device=device)[None, :] < relevant[start:stop, None]
            batch_ap = (precision * matches * mask).sum(dim=1) / relevant[start:stop]
            aps.extend(float(value) for value in batch_ap.cpu().tolist())
            hits.extend(int(value) for value in matches[:, 0].cpu().tolist())
    result = {
        "schema": "sfora-oml-sop-quality-profile-verification-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": sha256(Path(__file__)),
        "feature_sha256": FEATURE_SHA256,
        "external_checkpoint_sha256": MODEL_SHA256,
        "projection_sha256": profile.sha256,
        "test_rows": len(values),
        "served_bytes_per_row": 130,
        "map_at_r": float(math.fsum(aps) / len(aps)),
        "recall_at_1": float(math.fsum(hits) / len(hits)),
        "per_query_ap": aps,
        "per_query_r1": hits,
        "test_code_sha256": hashlib.sha256(packed.codes.numpy().tobytes()).hexdigest(),
        "test_inverse_norm_sha256": hashlib.sha256(
            packed.inverse_norms.numpy().tobytes()
        ).hexdigest(),
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
        "elapsed_seconds": float(time.perf_counter() - started),
    }
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()

    def validate(persisted: bytes) -> None:
        if persisted != payload:
            raise ValueError("SOP quality result publication differs")

    published = publish_bytes_noreplace(args.output, payload, validator=validate)
    published.close()
    print(json.dumps({"map_at_r": result["map_at_r"], "recall_at_1": result["recall_at_1"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
