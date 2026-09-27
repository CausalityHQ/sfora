#!/usr/bin/env python3
"""Fit-only learned-token residual against a deranged-token control."""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, split
from score_inshop_crop_view_pair import bootstrap_lower, sha256
from train_inshop_siglip2_unseen_gallery import half_fit_products
from train_sop_siglip2_compact import export_all, score_packed_full_gallery

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition

CHECKPOINT_SHA = "2140583a6f6e9b832935fa9f2740ebcda285235db173b920c94a0f54832d62a3"
RECEIPT_SHA = "f07f8ccbf4163e6c711946196c74feed988d8b2805d37f1a90143b9f9e7bd631"
FULL_FIT_SHA = "f23783a513f0bce23c4ea6126f8a868ee600fe88a23a5e797778a2a508ca89be"
HALF_FIT_SHA = "90bb48e566cd2144c74ed6c638362b227f115967393b13c8f6485e0c773fc9e0"
HELD_SHA = "9b1151e8cf65343682bd10b92885e79447d418ee6c664e2c005efccf7ebb5d1b"
SEEDS = (17, 23, 29)


class TokenResidual(torch.nn.Module):
    def __init__(self, seed: int) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        self.down = torch.nn.Linear(1024, 16, bias=False)
        self.up = torch.nn.Linear(16, 128, bias=False)
        with torch.no_grad():
            self.down.weight.copy_(torch.randn((16, 1024), generator=generator) / 32)
            self.up.weight.zero_()

    def forward(self, base: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        summary = torch.nn.functional.layer_norm(tokens.float(), (1024,))
        residual = self.up(torch.tanh(self.down(summary)))
        return torch.nn.functional.normalize(base + residual, dim=1)


def schedule(labels: torch.Tensor, seed: int) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = {
        int(label): torch.nonzero(labels == label, as_tuple=False).flatten()
        for label in torch.unique(labels).tolist()
    }
    eligible = tuple(label for label, rows in sorted(groups.items()) if len(rows) >= 4)
    if len(eligible) < 16:
        raise ValueError("inner-fit product schedule is too small")
    batches = []
    for _ in range(250):
        picked = torch.tensor(eligible)[torch.randperm(len(eligible), generator=generator)[:16]]
        batches.append(
            torch.cat(
                [
                    groups[int(label)][
                        torch.randperm(len(groups[int(label)]), generator=generator)[:4]
                    ]
                    for label in picked
                ]
            )
        )
    return tuple(batches)


def derangement(rows: int, seed: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.randperm(rows, generator=generator)
    donor = torch.empty(rows, dtype=torch.int64)
    donor[order] = torch.roll(order, shifts=1)
    if bool((donor == torch.arange(rows)).any()):
        raise ValueError("inner-fit token donor derangement failed")
    return donor


def contrastive(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    similarities = values @ values.T / 0.05
    self_mask = torch.eye(len(values), dtype=torch.bool, device=values.device)
    positives = labels[:, None].eq(labels[None, :]) & ~self_mask
    if bool((positives.sum(dim=1) == 0).any()):
        raise ValueError("inner-fit batch has no positive")
    return (
        torch.logsumexp(similarities.masked_fill(self_mask, -torch.inf), dim=1)
        - torch.logsumexp(similarities.masked_fill(~positives, -torch.inf), dim=1)
    ).mean()


def score(values: torch.Tensor, labels: torch.Tensor) -> dict[str, Any]:
    packed = pack_int8_unit_embeddings(values)
    result = score_packed_full_gallery(
        packed.codes.float(),
        packed.inverse_norms,
        labels,
        torch.arange(len(values)),
        device=torch.device("cuda"),
    )
    return result


def train_arm(
    base_fit: torch.Tensor,
    tokens_fit: torch.Tensor,
    labels_fit: torch.Tensor,
    base_val: torch.Tensor,
    tokens_val: torch.Tensor,
    seed: int,
    batches: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, float, int, float]:
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model = TokenResidual(seed).cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    final_loss = math.nan
    for indexes in batches:
        optimizer.zero_grad(set_to_none=True)
        values = model(base_fit[indexes].cuda(), tokens_fit[indexes].cuda())
        loss = contrastive(values, labels_fit[indexes].cuda())
        if not bool(torch.isfinite(loss)):
            raise ValueError("inner-fit residual loss is nonfinite")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if not bool(torch.isfinite(grad_norm)):
            raise ValueError("inner-fit residual gradient is nonfinite")
        optimizer.step()
        final_loss = float(loss.detach())
    model.eval()
    with torch.inference_mode():
        result = torch.cat(
            [
                model(
                    base_val[start : start + 128].cuda(), tokens_val[start : start + 128].cuda()
                ).cpu()
                for start in range(0, len(base_val), 128)
            ]
        )
    torch.cuda.synchronize()
    return result, time.perf_counter() - started, torch.cuda.max_memory_allocated(), final_loss


def save_new(path: Path, value: dict[str, Any]) -> None:
    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    if sys.argv[1:] == ["--self-test"]:
        base = torch.nn.functional.normalize(torch.randn(4, 128), dim=1)
        tokens = torch.randn(4, 1024)
        model = TokenResidual(17)
        assert torch.allclose(model(base, tokens), base, rtol=0, atol=1e-6)
        assert bool(torch.isfinite(contrastive(base, torch.tensor([0, 0, 1, 1]))))
        assert len(schedule(torch.tensor([0] * 4 + [1] * 4 + list(range(2, 18)) * 4), 17)) == 250
        assert not bool((derangement(101, 17) == torch.arange(101)).any())
        return
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("dataset-root", "model-snapshot", "checkpoint", "training-receipt", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        (args.output_dir / "receipt.json").exists()
        or not torch.cuda.is_available()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or sha256(args.checkpoint) != CHECKPOINT_SHA
        or sha256(args.training_receipt) != RECEIPT_SHA
        or args.model_snapshot.name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
    ):
        raise ValueError("inner-fit token source authority differs")
    receipt = json.loads(args.training_receipt.read_text())
    train = tuple(row for row in parse_inshop_partition(args.dataset_root) if row.split == "train")
    labels = tuple(row.label for row in train)
    full_fit, outer_held = split(labels)
    half_fit = half_fit_products(labels, full_fit)
    half_set = set(half_fit)
    inner_val = tuple(row for row in full_fit if row not in half_set)
    inner_counts = Counter(labels[row] for row in inner_val)
    scored_val = tuple(row for row in inner_val if inner_counts[labels[row]] > 1)
    if (
        len(train) != 25_882
        or len(full_fit) != 13_283
        or len(half_fit) != 6_764
        or len(inner_val) != 6_519
        or len(scored_val) != 6_514
        or len(set(labels[row] for row in scored_val)) != 997
        or len(outer_held) != 12_599
        or digest_rows(full_fit) != FULL_FIT_SHA
        or digest_rows(half_fit) != HALF_FIT_SHA
        or digest_rows(outer_held) != HELD_SHA
        or receipt.get("checkpoint_sha256") != CHECKPOINT_SHA
        or receipt.get("fit_rows_sha256") != HALF_FIT_SHA
        or receipt.get("half_fit_products") is not True
        or set(labels[row] for row in half_fit) & set(labels[row] for row in inner_val)
    ):
        raise ValueError("inner-fit product partition differs")

    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    full = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full.vision_model.float().cuda().eval()
    del full
    head = torch.nn.Linear(1024, 128).cuda().eval()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    vision.load_state_dict(checkpoint["vision"], strict=True)
    head.load_state_dict(checkpoint["head"], strict=True)
    if any(
        not torch.equal(value.cpu(), checkpoint["vision"][name])
        for name, value in vision.state_dict().items()
    ) or any(
        not torch.equal(value.cpu(), checkpoint["head"][name])
        for name, value in head.state_dict().items()
    ):
        raise ValueError("inner-fit FP32 checkpoint rounded")
    del checkpoint
    token_means: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
        tokens = output.last_hidden_state
        if tokens is None or tokens.ndim != 3 or tokens.shape[1:] != (256, 1024):
            raise ValueError("inner-fit final patch geometry differs")
        token_means.append(tokens.float().mean(dim=1).cpu())

    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = tuple(train[row].image_path for row in full_fit)
    started = time.perf_counter()
    hook = vision.register_forward_hook(capture)
    try:
        base = export_all(
            vision, head, paths, tuple(range(len(paths))), processor, workers=4, batch_size=64
        )
    finally:
        hook.remove()
    torch.cuda.synchronize()
    export_wall = time.perf_counter() - started
    token = torch.cat(token_means)
    if base.shape != (len(full_fit), 128) or token.shape != (len(full_fit), 1024):
        raise ValueError("inner-fit cache geometry differs")
    del vision, head, processor
    torch.cuda.empty_cache()
    positions = {row: i for i, row in enumerate(full_fit)}
    fit_positions = torch.tensor([positions[row] for row in half_fit])
    val_positions = torch.tensor([positions[row] for row in scored_val])
    classes = {name: i for i, name in enumerate(sorted(set(labels[row] for row in full_fit)))}
    fit_labels = torch.tensor([classes[labels[row]] for row in half_fit])
    val_labels = torch.tensor([classes[labels[row]] for row in scored_val])
    base_fit, base_val = base[fit_positions], base[val_positions]
    token_fit, token_val = token[fit_positions], token[val_positions]
    baseline = score(base_val, val_labels)
    donor_fit = token_fit[derangement(len(token_fit), 20260920)]
    donor_val = token_val[derangement(len(token_val), 20260921)]
    reports: list[dict[str, Any]] = []
    deltas = []
    for seed in SEEDS:
        batches = schedule(fit_labels, seed)
        arms = {}
        for name, fit_tokens, val_tokens in (
            ("treatment", token_fit, token_val),
            ("donor", donor_fit, donor_val),
        ):
            values, wall, peak, loss = train_arm(
                base_fit, fit_tokens, fit_labels, base_val, val_tokens, seed, batches
            )
            arms[name] = {
                "quality": score(values, val_labels),
                "fit_wall_seconds": wall,
                "peak_cuda_allocated_bytes": peak,
                "final_loss": loss,
            }
        delta = np.asarray(arms["treatment"]["quality"]["per_query_r1"]) - np.asarray(
            baseline["per_query_r1"]
        )
        deltas.append(delta)
        report = {
            "seed": seed,
            "arms": arms,
            "recall_delta_pp": 100 * float(delta.mean()),
            "map_delta": arms["treatment"]["quality"]["map_at_r"] - baseline["map_at_r"],
        }
        save_new(args.output_dir / f"seed-{seed}.json", report)
        reports.append(report)
        print(
            json.dumps(
                {
                    "seed": seed,
                    "recall_delta_pp": report["recall_delta_pp"],
                    "map_delta": report["map_delta"],
                }
            ),
            flush=True,
        )
        if seed == 17 and (report["recall_delta_pp"] <= 0 or report["map_delta"] < 0.002):
            break
    mean_delta = np.mean(np.stack(deltas), axis=0)
    lower = 100 * bootstrap_lower(mean_delta, np.asarray([labels[row] for row in scored_val]))
    mean_map = float(np.mean([row["map_delta"] for row in reports]))
    mean_donor_gap = float(
        np.mean(
            [
                row["arms"]["treatment"]["quality"]["map_at_r"]
                - row["arms"]["donor"]["quality"]["map_at_r"]
                for row in reports
            ]
        )
    )
    gates = {
        "three_seeds": len(reports) == 3,
        "every_recall_nonnegative": all(row["recall_delta_pp"] >= 0 for row in reports),
        "mean_recall_gain": 100 * float(mean_delta.mean()) >= 0.25,
        "paired_lower_positive": lower > 0,
        "mean_map_gain": mean_map >= 0.003,
        "donor_map_gap": mean_donor_gap >= 0.003,
        "fit_cost": all(
            row["arms"][arm]["fit_wall_seconds"] < 75
            and row["arms"][arm]["peak_cuda_allocated_bytes"] < 3_000_000_000
            for row in reports
            for arm in ("treatment", "donor")
        ),
    }
    save_new(
        args.output_dir / "receipt.json",
        {
            "schema": "sfora-inshop-token-residual-inner-fit-v1",
            "claim_eligible": False,
            "source_sha256": sha256(Path(__file__)),
            "helper_source_sha256": {
                name: sha256(Path(importlib.import_module(name).__file__))
                for name in (
                    "train_inshop_siglip2_unseen_gallery",
                    "train_sop_siglip2_compact",
                    "sfora.joint_relational_compaction",
                )
            },
            "checkpoint_sha256": CHECKPOINT_SHA,
            "training_receipt_sha256": RECEIPT_SHA,
            "full_fit_sha256": FULL_FIT_SHA,
            "half_fit_sha256": HALF_FIT_SHA,
            "outer_held_sha256": HELD_SHA,
            "inner_validation_rows": len(scored_val),
            "singleton_rows_excluded_from_scoring": len(inner_val) - len(scored_val),
            "baseline": baseline,
            "export_wall_seconds": export_wall,
            "seeds_run": [row["seed"] for row in reports],
            "mean_recall_delta_pp": 100 * float(mean_delta.mean()),
            "paired_product_bootstrap_lower_pp": lower,
            "mean_map_delta": mean_map,
            "mean_treatment_minus_donor_map": mean_donor_gap,
            "gates": gates,
            "advance": all(gates.values()),
            "hardware": {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
        },
    )


if __name__ == "__main__":
    main()
