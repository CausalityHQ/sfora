#!/usr/bin/env python3
"""Fit-only SOP cached-token falsifier for a content-addressed slot readout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from score_sop_b16_resolution_train import compare
from screen_sop_cached_teacher_tail import arcface_term
from screen_sop_finite_gallery_head import score_heldout_against_all
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from train_sop_compact_backbone import initialize_head_and_classifier

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.product_pair_schedule import product_pair_batches
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features
from sfora.sop_evaluation import score_gallery_r1, score_symmetric
from sfora.token_slot_readout import MeanTokenReadout, SlotTokenReadout
from sfora.unicom_tail_adapter import output_from_last_block_input

SOURCE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
CACHE_SHA256 = "f7835576efaf513c59895ba99eb4b839649df972f62aecbf674e9542a118b0d9"
CHECKPOINT_SHA256 = "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef"
SEED = 179019
ARMS = ("flatten_arcface", "flatten_anchor", "mean_anchor", "slot4_anchor")
ANCHOR_WEIGHT = 1.0
HEAD_LR = 1e-4
TAIL_LR = 1e-5
PROXY_LR = 1e-3
DEPENDENCIES = (
    "scripts/screen_sop_token_slot_readout.py",
    "scripts/score_sop_b16_resolution_train.py",
    "scripts/screen_sop_cached_teacher_tail.py",
    "scripts/screen_sop_finite_gallery_head.py",
    "scripts/sop_teacher_anchored_runtime.py",
    "scripts/train_sop_compact_backbone.py",
    "src/sfora/joint_relational_compaction.py",
    "src/sfora/product_pair_schedule.py",
    "src/sfora/representation_ceiling.py",
    "src/sfora/sop_compact_training.py",
    "src/sfora/sop_evaluation.py",
    "src/sfora/token_slot_readout.py",
    "src/sfora/unicom_tail_adapter.py",
)


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {relative: sha256(root / relative) for relative in DEPENDENCIES}


def publish_json(path: Path, value: dict[str, Any]) -> None:
    """Publish a completed per-arm receipt without replacing prior evidence."""

    with path.open("xb") as stream:
        stream.write((json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())


def ridge_mean_map(
    model: Any,
    cache: np.ndarray,
    fit_rows: np.ndarray,
    target: np.ndarray,
    *,
    batch_size: int = 64,
) -> tuple[MeanTokenReadout, dict[str, float]]:
    """Fit a centered linear source-descriptor map from fit-only token means."""

    if fit_rows.ndim != 1 or target.shape != (len(fit_rows), 768) or cache.shape[1:] != (196, 768):
        raise ValueError("SOP slot ridge inventory differs")
    model.eval()
    pooled = torch.empty((len(fit_rows), 768), dtype=torch.float32)
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(fit_rows), batch_size):
            stop = min(start + batch_size, len(fit_rows))
            tokens = torch.from_numpy(np.asarray(cache[fit_rows[start:stop]]).copy()).cuda()
            final = model.norm(model.blocks[-1](tokens).float())
            pooled[start:stop] = final.mean(dim=1).cpu()
    extraction_seconds = time.perf_counter() - started
    x = pooled.double().cuda()
    y = torch.from_numpy(target.copy()).double().cuda()
    x_mean = x.mean(dim=0)
    y_mean = y.mean(dim=0)
    centered_x = x - x_mean
    centered_y = y - y_mean
    gram = centered_x.T @ centered_x
    ridge_lambda = 0.001 * float(torch.trace(gram)) / 768
    weights = torch.linalg.solve(
        gram + ridge_lambda * torch.eye(768, device="cuda", dtype=torch.float64),
        centered_x.T @ centered_y,
    )
    bias = y_mean - x_mean @ weights
    mean = MeanTokenReadout(width=768, output_width=768)
    with torch.no_grad():
        mean.projection.weight.copy_(weights.T.float().cpu())
        mean.projection.bias.copy_(bias.float().cpu())
    prediction = x @ weights + bias
    cosine = F.cosine_similarity(prediction.float(), y.float(), dim=1)
    fit_cosine = float(cosine.mean())
    del x, y, gram, weights, prediction
    torch.cuda.synchronize()
    return mean, {
        "fit_token_mean_seconds": extraction_seconds,
        "total_ridge_seconds": time.perf_counter() - started,
        "ridge_lambda": ridge_lambda,
        "fit_source_cosine": fit_cosine,
    }


def source_from_cache(model: Any, tokens: torch.Tensor, readout: nn.Module | None) -> torch.Tensor:
    if readout is None:
        return output_from_last_block_input(model, tokens)
    final = model.norm(model.blocks[-1](tokens).float())
    return cast(torch.Tensor, readout(final))


@torch.inference_mode()
def evaluate(
    model: Any,
    head: nn.Linear,
    readout: nn.Module | None,
    cache: np.ndarray,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    held_rows: np.ndarray,
) -> dict[str, Any]:
    model.eval()
    head.eval()
    if readout is not None:
        readout.eval()
    values = torch.empty((len(labels), 128), dtype=torch.float32)
    for start in range(0, len(labels), 64):
        stop = min(start + 64, len(labels))
        tokens = torch.from_numpy(np.asarray(cache[start:stop]).copy()).cuda()
        source = source_from_cache(model, tokens, readout)
        features = compact_head_features(source, head)
        values[start:stop] = F.normalize(features, dim=1).cpu()
    if not bool(torch.isfinite(values).all()):
        raise ValueError("SOP slot nonfinite evaluation features")
    packed = pack_int8_unit_embeddings(values)
    held = torch.from_numpy(held_rows)
    held_labels = torch.from_numpy(labels[held_rows].copy())
    all_labels = torch.from_numpy(labels.copy())
    code_float = packed.codes.float()
    packed_holdout = score_symmetric(
        code_float[held], held_labels, inverse_norms=packed.inverse_norms[held]
    )
    packed_full = score_heldout_against_all(
        code_float,
        packed.inverse_norms,
        all_labels,
        held,
        torch.from_numpy(fit_rows.copy()),
        device=torch.device("cuda"),
    )
    float_gpu = values.cuda()
    labels_gpu = all_labels.cuda()
    held_gpu = held.cuda()
    return {
        "float": {
            "holdout_only": score_symmetric(float_gpu[held_gpu], labels_gpu[held_gpu]),
            "full_train_gallery": score_gallery_r1(float_gpu, labels_gpu, held_gpu),
        },
        "packed_130b": {
            "holdout_only": packed_holdout,
            "full_train_gallery": packed_full,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("unicom-checkout", "checkpoint", "source-archive", "token-cache", "output-dir"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or args.output_dir.is_symlink()
        or not torch.cuda.is_available()
        or sha256(args.checkpoint) != CHECKPOINT_SHA256
        or sha256(args.source_archive) != SOURCE_SHA256
        or sha256(args.token_cache) != CACHE_SHA256
    ):
        raise ValueError("SOP slot screen source/output authority differs")
    manifest = source_manifest()
    started = time.perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        source_features = np.asarray(archive["train_embeddings"], dtype=np.float32)
    cache = np.load(args.token_cache, mmap_mode="r", allow_pickle=False)
    if (
        labels.shape != (59_551,)
        or source_features.shape != (59_551, 768)
        or cache.shape != (59_551, 196, 768)
        or cache.dtype != np.float32
    ):
        raise ValueError("SOP slot source inventory differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    fit_rows = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_rows = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    fit_labels = labels[fit_rows]
    batches = product_pair_batches(fit_labels, products_per_batch=32, passes=5, seed=SEED)
    schedule_sha = hashlib.sha256(batches.astype("<i8").tobytes()).hexdigest()
    class_names, class_ids_np = np.unique(fit_labels, return_inverse=True)
    if (
        len(fit_rows) != 53_700
        or len(held_rows) != 5_851
        or len(class_names) != 10_186
        or len(batches) != 1_595
    ):
        raise ValueError("SOP slot partition or schedule differs")
    fit_source = torch.from_numpy(source_features[fit_rows].copy())
    initial_head, initial_proxy = initialize_head_and_classifier(
        fit_source, tuple(map(int, fit_labels))
    )
    class_ids = torch.from_numpy(class_ids_np.copy()).cuda()
    authenticated = load_authenticated_source_model(args.unicom_checkout, args.checkpoint)
    model: Any = authenticated.encoder.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tail_state = {k: v.detach().clone() for k, v in model.blocks[-1].state_dict().items()}
    ridge, ridge_receipt = ridge_mean_map(model, cache, fit_rows, source_features[fit_rows].copy())
    slot_init = SlotTokenReadout(width=768, output_width=768, slots=4)
    slot_init.initialize_from_mean(ridge, seed=SEED)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    results: dict[str, Any] = {}
    arm_files: dict[str, dict[str, str]] = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        model.blocks[-1].load_state_dict(tail_state)
        model.eval()
        for parameter in model.blocks[-1].parameters():
            parameter.requires_grad_(True)
            parameter.grad = None
        head = nn.Linear(768, 128, device="cuda")
        head.load_state_dict(initial_head.state_dict())
        proxy = nn.Parameter(initial_proxy.detach().clone().cuda())
        readout: nn.Module | None = None
        if arm == "mean_anchor":
            readout = MeanTokenReadout(width=768, output_width=768).cuda()
            readout.load_state_dict(ridge.state_dict())
        elif arm == "slot4_anchor":
            readout = SlotTokenReadout(width=768, output_width=768, slots=4).cuda()
            readout.load_state_dict(slot_init.state_dict())
        groups: list[dict[str, Any]] = [
            {"params": head.parameters(), "lr": HEAD_LR},
            {"params": [proxy], "lr": PROXY_LR},
            {"params": model.blocks[-1].parameters(), "lr": TAIL_LR},
        ]
        if readout is not None:
            groups.append({"params": readout.parameters(), "lr": HEAD_LR})
        optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
        torch.cuda.reset_peak_memory_stats()
        begin = time.perf_counter()
        first_loss = None
        last_loss = None
        last_anchor = None
        for step, local_rows in enumerate(batches, 1):
            global_rows = fit_rows[local_rows]
            tokens = torch.from_numpy(np.asarray(cache[global_rows]).copy()).cuda()
            source = source_from_cache(model, tokens, readout)
            features = compact_head_features(source, head)
            targets = class_ids[torch.from_numpy(local_rows).cuda()]
            arcface = arcface_term(features, proxy, targets)
            if arm == "flatten_arcface":
                anchor = arcface.new_zeros(())
            else:
                teacher = torch.from_numpy(source_features[global_rows].copy()).cuda()
                anchor = (1.0 - F.cosine_similarity(source.float(), teacher, dim=1)).mean()
            loss = arcface + ANCHOR_WEIGHT * anchor
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"SOP slot {arm} loss nonfinite")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            parameters = list(head.parameters()) + [proxy] + list(model.blocks[-1].parameters())
            if readout is not None:
                parameters += list(readout.parameters())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            optimizer.step()
            if first_loss is None:
                first_loss = float(loss.detach())
            last_loss = float(loss.detach())
            last_anchor = float(anchor.detach())
            if step % 319 == 0:
                print(json.dumps({"arm": arm, "step": step, "loss": last_loss}), flush=True)
        torch.cuda.synchronize()
        train_seconds = time.perf_counter() - begin
        train_peak = torch.cuda.max_memory_allocated()
        expected_state = sum(len(group["params"]) for group in optimizer.param_groups)
        if len(optimizer.state) != expected_state or any(
            int(state["step"].item()) != len(batches) for state in optimizer.state.values()
        ):
            raise ValueError(f"SOP slot {arm} optimizer update count differs")
        begin = time.perf_counter()
        score = evaluate(model, head, readout, cache, labels, fit_rows, held_rows)
        eval_seconds = time.perf_counter() - begin
        if arm == "flatten_arcface" and (
            abs(score["packed_130b"]["holdout_only"]["recall_at_1"] - 0.885490) > 0.002
            or abs(score["packed_130b"]["full_train_gallery"]["recall_at_1"] - 0.750812) > 0.002
        ):
            raise ValueError("SOP slot flatten control does not reproduce prior result")
        checkpoint = args.output_dir / f"{arm}.pt"
        torch.save(
            {
                "tail": {k: v.detach().cpu() for k, v in model.blocks[-1].state_dict().items()},
                "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                "proxy": proxy.detach().cpu(),
                "readout": None
                if readout is None
                else {k: v.detach().cpu() for k, v in readout.state_dict().items()},
                "arm": arm,
                "seed": SEED,
                "updates": len(batches),
                "source_checkpoint_sha256": CHECKPOINT_SHA256,
            },
            checkpoint,
        )
        arm_result = {
            "arm": arm,
            "schema": "sfora-sop-slot-readout-arm-v1",
            "split": "SOP official TRAIN fit/holdout only",
            "seed": SEED,
            "updates": len(batches),
            "train_seconds": train_seconds,
            "evaluation_seconds": eval_seconds,
            "train_peak_cuda_allocated_bytes": train_peak,
            "readout_parameters": 0
            if readout is None
            else sum(p.numel() for p in readout.parameters()),
            "first_loss": first_loss,
            "last_loss": last_loss,
            "last_anchor": last_anchor,
            "score": score,
            "checkpoint_sha256": sha256(checkpoint),
            "source_manifest_sha256": hashlib.sha256(
                json.dumps(manifest, sort_keys=True).encode()
            ).hexdigest(),
        }
        arm_path = args.output_dir / f"{arm}.json"
        publish_json(arm_path, arm_result)
        results[arm] = score
        arm_files[arm] = {
            "receipt": arm_path.name,
            "receipt_sha256": sha256(arm_path),
            "checkpoint": checkpoint.name,
            "checkpoint_sha256": sha256(checkpoint),
        }
        print(
            json.dumps(
                {
                    "arm": arm,
                    "train_seconds": train_seconds,
                    "packed_full_r1": score["packed_130b"]["full_train_gallery"]["recall_at_1"],
                }
            ),
            flush=True,
        )
        if arm == "flatten_anchor":
            # Candidates have no dependency on the 115M-parameter source head.
            model.feature = nn.Identity()
            torch.cuda.empty_cache()
    held_labels = labels[held_rows]
    paired = {}
    for candidate, control in (
        ("slot4_anchor", "flatten_arcface"),
        ("slot4_anchor", "flatten_anchor"),
        ("slot4_anchor", "mean_anchor"),
    ):
        paired[f"{candidate}_minus_{control}"] = {
            representation: {
                gallery: compare(
                    results[candidate][representation][gallery],
                    results[control][representation][gallery],
                    held_labels,
                )
                for gallery in ("holdout_only", "full_train_gallery")
            }
            for representation in ("float", "packed_130b")
        }
    if source_manifest() != manifest:
        raise ValueError("SOP slot source changed during screen")
    receipt = {
        "schema": "sfora-sop-slot-readout-screen-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint fit/holdout; no TEST rows read",
        "fit_images": 53_700,
        "holdout_queries": 5_851,
        "heldout_products": 1_132,
        "full_gallery_images": 59_551,
        "seed": SEED,
        "updates_per_arm": len(batches),
        "schedule_sha256": schedule_sha,
        "ridge": ridge_receipt,
        "source_sha256": manifest,
        "source_archive_sha256": SOURCE_SHA256,
        "token_cache_sha256": CACHE_SHA256,
        "source_checkpoint_sha256": CHECKPOINT_SHA256,
        "arms": arm_files,
        "paired_product_bootstrap": paired,
        "total_seconds": time.perf_counter() - started,
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    publish_json(args.output_dir / "receipt.json", receipt)
    print(
        json.dumps({"result": "completed", "total_seconds": receipt["total_seconds"]}), flush=True
    )


if __name__ == "__main__":
    main()
