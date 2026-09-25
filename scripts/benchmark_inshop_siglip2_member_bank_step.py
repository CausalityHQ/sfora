#!/usr/bin/env python3
"""Time the frozen In-Shop member-bank loss at its actual fit geometry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from preflight_inshop_siglip2_coverage import PARTITION_SHA256, fit_and_holdout, sha256
from torch.nn import functional as F
from train_sop_siglip2_compact import member_bank_positive_ordinals

import sfora.deployed_code_rank as rank_module
from sfora.deployed_code_rank import smooth_ap_bank_loss
from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_rank_finish import identity_balanced_batches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.preflight) != args.expected_preflight_sha256
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
        or not torch.cuda.is_available()
    ):
        raise ValueError("In-Shop bank timing authority differs")
    preflight = json.loads(args.preflight.read_text())
    labels = tuple(
        row.label for row in parse_inshop_partition(args.dataset_root) if row.split == "train"
    )
    fit, _ = fit_and_holdout(labels)
    fit_labels = tuple(labels[index] for index in fit)
    counts = Counter(fit_labels)
    rows = len(fit)
    positives_width = max(counts.values()) - 1
    if (
        preflight.get("schema") != "sfora-inshop-siglip2-coverage-preflight-v1"
        or preflight.get("fit_row_indexes_sha256")
        != hashlib.sha256(np.asarray(fit, dtype="<i8").tobytes()).hexdigest()
        or rows != 23_342
        or positives_width < 1
    ):
        raise ValueError("In-Shop bank timing fit geometry differs")
    names = {name: index for index, name in enumerate(sorted(counts))}
    class_ids = np.asarray([names[name] for name in fit_labels], dtype=np.int64)
    positive_table = member_bank_positive_ordinals(class_ids, allow_singletons=True)
    if positive_table.shape != (rows, positives_width):
        raise ValueError("In-Shop bank timing positive table differs")
    sample_batches = []
    for seed in (179023, 179024, 179025):
        schedule = identity_balanced_batches(
            fit_labels,
            batch_size=64,
            images_per_identity=4,
            seed=seed,
            epoch=1,
            steps=1000,
            coverage_first=True,
        )
        expected = preflight["schedules"][str(seed)]
        if (
            hashlib.sha256(np.asarray(schedule, dtype="<i4").tobytes()).hexdigest()
            != expected["sha256"]
        ):
            raise ValueError("In-Shop bank timing schedule differs")
        active = [batch for batch in schedule if all(counts[fit_labels[row]] > 1 for row in batch)]
        sample_batches.extend(
            active[index] for index in np.linspace(0, len(active) - 1, 20, dtype=int)
        )
    torch.manual_seed(179019)
    torch.backends.cuda.matmul.allow_tf32 = False
    device = torch.device("cuda:0")
    bank = F.normalize(torch.randn((rows, 128), device=device), dim=1)
    anchors = F.normalize(torch.randn((64, 128), device=device), dim=1).requires_grad_()
    positive_table = positive_table.to(device)

    def one_step(batch: tuple[int, ...]) -> tuple[float, int]:
        anchors.grad = None
        self_ordinals = torch.tensor(batch, device=device)
        positives = positive_table[self_ordinals]
        width = int((positives >= 0).sum(dim=1).max())
        positives = positives[:, :width]
        started = time.perf_counter()
        loss = smooth_ap_bank_loss(anchors, bank, positives, self_ordinals)
        loss.backward()
        bank[self_ordinals] = anchors.detach()
        torch.cuda.synchronize(device)
        return time.perf_counter() - started, width

    for batch in sample_batches[:5]:
        one_step(batch)
    torch.cuda.reset_peak_memory_stats(device)
    timed = [one_step(batch) for batch in sample_batches]
    samples = [seconds for seconds, _ in timed]
    widths = [width for _, width in timed]
    median = statistics.median(samples)
    result = {
        "schema": "sfora-inshop-siglip2-member-bank-step-cost-v2",
        "claim_eligible": False,
        "preflight_sha256": args.expected_preflight_sha256,
        "partition_sha256": PARTITION_SHA256,
        "source_sha256": sha256(Path(__file__)),
        "loss_source_sha256": sha256(Path(rank_module.__file__)),
        "rows": rows,
        "anchors": 64,
        "positives_per_anchor": positives_width,
        "dimension": 128,
        "warmups": 5,
        "timed_steps": len(samples),
        "sampled_batch_positive_widths": widths,
        "wall_seconds": samples,
        "median_wall_seconds": median,
        "p95_wall_seconds": float(np.percentile(samples, 95)),
        "gate_wall_seconds": 0.06,
        "gate_pass": median <= 0.06,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "hardware": {"gpu": torch.cuda.get_device_name(device), "torch": torch.__version__},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("median_wall_seconds", "gate_pass", "positives_per_anchor")
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
