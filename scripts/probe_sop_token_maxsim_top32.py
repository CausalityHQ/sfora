#!/usr/bin/env python3
"""Train-only frozen-token MaxSim falsifier for compact SOP retrieval."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features
from sfora.unicom_tail_adapter import output_from_last_block_input

SOURCE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
CACHE_SHA256 = "f7835576efaf513c59895ba99eb4b839649df972f62aecbf674e9542a118b0d9"
HEAD_SHA256 = "9eaf0cdbd757f1f83b00e5415349b4608a62769a56ace81983d67ea2b87781fd"
ORIGINAL_RECEIPT_SHA256 = "331b8055b970704b1269b3229e0b2114f594033bfe1690c9af736b4631b46aa1"
SEED = 179019
SHORTLIST = 32


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def metrics(top_rows: torch.Tensor, labels: torch.Tensor, relevant: torch.Tensor) -> dict[str, object]:
    if top_rows.shape != (len(labels), SHORTLIST) or int(relevant.max()) >= SHORTLIST:
        raise ValueError("SOP token MaxSim metric geometry differs")
    width = int(relevant.max())
    matches = labels[top_rows[:, :width]] == labels[:, None]
    ranks = torch.arange(1, width + 1, device=labels.device)
    precision = matches.cumsum(dim=1) / ranks[None, :]
    ap = (precision * matches * (ranks[None, :] <= relevant[:, None])).sum(dim=1) / relevant
    r1 = matches[:, 0]
    return {
        "recall_at_1": float(r1.float().mean().item()),
        "map_at_r": float(ap.mean().item()),
        "per_query_r1": [int(x) for x in r1.cpu().tolist()],
        "per_query_ap": [float(x) for x in ap.cpu().tolist()],
    }


@torch.inference_mode()
def maxsim_top32(tokens: torch.Tensor, shortlisted: torch.Tensor, *, block: int) -> torch.Tensor:
    """Symmetric mean-of-max-patch cosine; exact ordinal tie ordering."""

    n = len(tokens)
    if tokens.shape != (n, 196, 768) or shortlisted.shape != (n, SHORTLIST):
        raise ValueError("SOP token MaxSim input geometry differs")
    chosen = torch.empty((n, SHORTLIST), dtype=torch.int64, device=tokens.device)
    for start in range(0, n, block):
        stop = min(start + block, n)
        candidate = shortlisted[start:stop]
        queries = tokens[start:stop, None].expand(-1, SHORTLIST, -1, -1)
        queries = queries.reshape(-1, 196, 768)
        gallery = tokens[candidate.reshape(-1)]
        pair = torch.bmm(queries, gallery.transpose(1, 2))
        local = (pair.amax(dim=2).mean(dim=1) + pair.amax(dim=1).mean(dim=1)) * 0.5
        local = local.reshape(stop - start, SHORTLIST)
        if not bool(torch.isfinite(local).all()):
            raise ValueError("SOP token MaxSim nonfinite score")
        ordinal = torch.argsort(candidate, dim=1, stable=True)
        ordinal_candidate = torch.gather(candidate, 1, ordinal)
        ordinal_local = torch.gather(local, 1, ordinal)
        order = torch.argsort(ordinal_local, dim=1, descending=True, stable=True)
        chosen[start:stop] = torch.gather(ordinal_candidate, 1, order)
        if (start // block) % 40 == 0:
            print(json.dumps({"scored_queries": stop, "of": n}), flush=True)
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--head-checkpoint", type=Path, required=True)
    parser.add_argument("--original-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--execute-sop-token-maxsim-top32", action="store_true", required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != SOURCE_SHA256
        or sha256(args.token_cache) != CACHE_SHA256
        or sha256(args.head_checkpoint) != HEAD_SHA256
        or sha256(args.original_receipt) != ORIGINAL_RECEIPT_SHA256
        or not torch.cuda.is_available()
    ):
        raise ValueError("SOP token MaxSim authority differs")
    original = json.loads(args.original_receipt.read_text())
    if (
        original["schema"] != "sfora-sop-cached-teacher-tail-screen-v1"
        or original["results"]["head_arcface"]["checkpoint_sha256"] != HEAD_SHA256
    ):
        raise ValueError("SOP token MaxSim head control differs")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    held_ids = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(held_ids) != 5_851:
        raise ValueError("SOP token MaxSim heldout count differs")
    cache = np.load(args.token_cache, mmap_mode="r", allow_pickle=False)
    if cache.shape != (59_551, 196, 768) or cache.dtype != np.float32:
        raise ValueError("SOP token MaxSim cache geometry differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    state = torch.load(args.head_checkpoint, map_location="cpu", weights_only=True)
    if state.get("arm") != "head_arcface" or state.get("cache_sha256") != CACHE_SHA256:
        raise ValueError("SOP token MaxSim head metadata differs")
    source = load_authenticated_source_model(args.unicom_checkout, args.source_checkpoint)
    model = source.encoder.cuda().eval()
    model.load_state_dict(state["model"], strict=True)
    head = nn.Linear(768, 128)
    head.load_state_dict(state["head"], strict=True)
    head = head.cuda().eval()
    device = torch.device("cuda")
    held_labels = torch.from_numpy(labels[held_ids].copy()).to(device)
    all_codes = torch.empty((len(held_ids), 128), dtype=torch.float32, device=device)
    all_inv = torch.empty(len(held_ids), dtype=torch.float32, device=device)
    for start in range(0, len(held_ids), 64):
        stop = min(start + 64, len(held_ids))
        raw = torch.from_numpy(np.asarray(cache[held_ids[start:stop]]).copy()).to(device)
        values = F.normalize(compact_head_features(output_from_last_block_input(model, raw), head), dim=1)
        packed = pack_int8_unit_embeddings(values.cpu())
        all_codes[start:stop] = packed.codes.float().to(device)
        all_inv[start:stop] = packed.inverse_norms.float().to(device)
    scores = (all_codes @ all_codes.T) * all_inv[:, None] * all_inv[None, :]
    scores.fill_diagonal_(-torch.inf)
    shortlist = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :SHORTLIST]
    relevant = torch.bincount(held_labels)[held_labels] - 1
    baseline = metrics(shortlist, held_labels, relevant)
    if (
        abs(baseline["recall_at_1"] - original["results"]["head_arcface"]["holdout_only"]["recall_at_1"]) > 1e-6
        or abs(baseline["map_at_r"] - original["results"]["head_arcface"]["holdout_only"]["map_at_r"]) > 1e-6
    ):
        raise ValueError("SOP token MaxSim baseline does not reproduce")
    if args.preflight_only:
        token_sample = F.normalize(
            torch.from_numpy(np.asarray(cache[held_ids[:8]]).copy()).to(device), dim=2
        )
        sample_rows = shortlist[:8].clone()
        # A small standalone sample cannot address full-index shortlist entries.
        sample_rows %= 8
        if not bool(torch.isfinite(maxsim_top32(token_sample, sample_rows, block=4).float()).all()):
            raise ValueError("SOP token MaxSim preflight differs")
        print(json.dumps({"preflight": "passed", "baseline_r1": baseline["recall_at_1"]}), flush=True)
        return
    tokens = F.normalize(
        torch.from_numpy(np.asarray(cache[held_ids]).copy()).to(device), dim=2
    )
    maxsim_started = time.perf_counter()
    maxsim_rows = maxsim_top32(tokens, shortlist, block=8)
    maxsim_seconds = time.perf_counter() - maxsim_started
    candidate = metrics(maxsim_rows, held_labels, relevant)
    positive_in_shortlist = (held_labels[shortlist] == held_labels[:, None]).any(dim=1)
    receipt = {
        "schema": "sfora-sop-token-maxsim-top32-v1",
        "claim_eligible": False,
        "source_archive_sha256": SOURCE_SHA256,
        "cache_sha256": CACHE_SHA256,
        "head_checkpoint_sha256": HEAD_SHA256,
        "original_receipt_sha256": ORIGINAL_RECEIPT_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "seed": SEED,
        "heldout_rows": len(held_ids),
        "heldout_products": len(np.unique(labels[held_ids])),
        "shortlist": SHORTLIST,
        "positive_in_shortlist": int(positive_in_shortlist.sum().item()),
        "baseline": baseline,
        "maxsim": candidate,
        "maxsim_seconds": maxsim_seconds,
        "total_seconds_after_hashes": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"baseline_r1": baseline["recall_at_1"], "maxsim_r1": candidate["recall_at_1"], "receipt": str(args.output)}), flush=True)


if __name__ == "__main__":
    main()
