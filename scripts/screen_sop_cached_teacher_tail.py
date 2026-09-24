#!/usr/bin/env python3
"""Matched SOP train-only B/16 tail adaptation with cached L/14 relations."""

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
from screen_sop_finite_gallery_head import score_heldout_against_all
from sop_teacher_anchored_runtime import load_authenticated_source_model
from torch import nn
from torch.nn import functional as F
from train_sop_compact_backbone import initialize_head_and_classifier

from sfora.deployed_code_rank import packed_cosine_ste
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.product_pair_schedule import product_pair_batches
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features
from sfora.sop_evaluation import score_symmetric
from sfora.unicom_tail_adapter import output_from_last_block_input

SOURCE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
TEACHER_SHA256 = "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
CACHE_SHA256 = "f7835576efaf513c59895ba99eb4b839649df972f62aecbf674e9542a118b0d9"
SEED = 179019
PRODUCTS_PER_BATCH = 32
PASSES = 5
ARMS = ("head_arcface", "head_teacher", "tail_arcface", "tail_teacher")
MARGIN = 0.3
SCALE = 64.0
TEMPERATURE = 0.1
TEACHER_COEFFICIENT = 1.0
HEAD_LR = 1e-4
TAIL_LR = 1e-5
PROXY_LR = 1e-3
DEPENDENCIES = (
    "scripts/screen_sop_cached_teacher_tail.py",
    "scripts/screen_sop_finite_gallery_head.py",
    "scripts/train_sop_compact_backbone.py",
    "scripts/sop_teacher_anchored_runtime.py",
    "src/sfora/deployed_code_rank.py",
    "src/sfora/joint_relational_compaction.py",
    "src/sfora/product_pair_schedule.py",
    "src/sfora/representation_ceiling.py",
    "src/sfora/sop_compact_training.py",
    "src/sfora/sop_evaluation.py",
    "src/sfora/unicom_tail_adapter.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def arcface_term(features: torch.Tensor, proxy: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    cosine = F.normalize(features, dim=1) @ F.normalize(proxy, dim=1).T
    row = torch.arange(len(features), device=features.device)
    selected = cosine[row, target].clamp(-1 + 1e-6, 1 - 1e-6)
    margin_cosine = selected * np.cos(MARGIN) - torch.sqrt(1 - selected.square()) * np.sin(MARGIN)
    logits = cosine.scatter(1, target[:, None], margin_cosine[:, None]) * SCALE
    return F.cross_entropy(logits, target)


def relational_term(student_features: torch.Tensor, teacher_features: torch.Tensor) -> torch.Tensor:
    """Rowwise KL on all nonself pair scores, through the deployed packed wire."""

    if (
        student_features.ndim != 2
        or student_features.shape[1] != 128
        or teacher_features.shape != (len(student_features), 768)
        or not student_features.requires_grad
        or teacher_features.requires_grad
    ):
        raise ValueError("SOP teacher relation geometry differs")
    student = packed_cosine_ste(student_features)
    teacher = F.normalize(teacher_features, dim=1) @ F.normalize(teacher_features, dim=1).T
    mask = ~torch.eye(len(student_features), dtype=torch.bool, device=student.device)
    student_logits = student[mask].reshape(len(student), len(student) - 1) / TEMPERATURE
    teacher_logits = teacher[mask].reshape(len(teacher), len(teacher) - 1) / TEMPERATURE
    return F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )


@torch.inference_mode()
def evaluate_arm(
    model: nn.Module,
    head: nn.Linear,
    cache: np.ndarray,
    labels: np.ndarray,
    fit_ids: np.ndarray,
    held_ids: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, object]:
    model.eval()
    head.eval()
    all_codes = torch.empty((len(labels), 128), dtype=torch.float32)
    all_inverse = torch.empty(len(labels), dtype=torch.float16)
    for start in range(0, len(labels), 64):
        stop = min(start + 64, len(labels))
        tokens = torch.from_numpy(np.asarray(cache[start:stop]).copy()).to(device)
        source = output_from_last_block_input(model, tokens)
        feature = F.normalize(compact_head_features(source, head), dim=1).cpu().contiguous()
        packed = pack_int8_unit_embeddings(feature)
        all_codes[start:stop] = packed.codes.float()
        all_inverse[start:stop] = packed.inverse_norms
    held_index = torch.from_numpy(held_ids)
    held_labels = torch.from_numpy(labels[held_ids].copy())
    held = score_symmetric(
        all_codes[held_index], held_labels, inverse_norms=all_inverse[held_index]
    )
    full = score_heldout_against_all(
        all_codes,
        all_inverse,
        torch.from_numpy(labels.copy()),
        held_index,
        torch.from_numpy(fit_ids),
        device=device,
    )
    return {
        "holdout_only": {
            key: held[key] for key in ("recall_at_1", "map_at_r", "per_query_r1", "per_query_ap")
        },
        "all_train_gallery": full,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--unicom-checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--teacher-archive", type=Path, required=True)
    parser.add_argument("--token-cache", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--tail-eval-control-only", action="store_true")
    parser.add_argument("--execute-sop-cached-teacher-tail", action="store_true", required=True)
    args = parser.parse_args()
    receipt_path = args.output_dir / "receipt.json"
    if (
        receipt_path.exists()
        or args.output_dir.is_symlink()
        or sha256(args.source_archive) != SOURCE_SHA256
        or sha256(args.teacher_archive) != TEACHER_SHA256
        or sha256(args.token_cache) != CACHE_SHA256
        or not torch.cuda.is_available()
    ):
        raise ValueError("SOP cached teacher-tail source/output authority differs")
    started = time.perf_counter()
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    with (
        np.load(args.source_archive, allow_pickle=False) as source,
        np.load(args.teacher_archive, allow_pickle=False) as teacher,
    ):
        labels = np.asarray(source["train_labels"], dtype=np.int64)
        source_features = np.asarray(source["train_embeddings"], dtype=np.float32)
        teacher_features = np.asarray(teacher["train_embeddings"], dtype=np.float32)
        for key in ("train_labels", "train_image_ids", "train_relative_paths"):
            if not np.array_equal(source[key], teacher[key]):
                raise ValueError("SOP B/16-L/14 teacher rows differ")
    if (
        labels.shape != (59_551,)
        or source_features.shape != (59_551, 768)
        or teacher_features.shape != (59_551, 768)
    ):
        raise ValueError("SOP cached teacher-tail feature inventory differs")
    cache = np.load(args.token_cache, mmap_mode="r", allow_pickle=False)
    if cache.shape != (59_551, 196, 768) or cache.dtype != np.float32:
        raise ValueError("SOP cached teacher-tail token geometry differs")
    partition = deterministic_class_partition(tuple(map(int, labels)), fit_fraction=0.9, seed=SEED)
    fit_ids = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_ids = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if (len(fit_ids), len(held_ids)) != (53_700, 5_851):
        raise ValueError("SOP cached teacher-tail identity partition differs")
    fit_labels = labels[fit_ids]
    batches = product_pair_batches(
        fit_labels, products_per_batch=PRODUCTS_PER_BATCH, passes=PASSES, seed=SEED
    )
    schedule_sha256 = hashlib.sha256(batches.astype("<i8").tobytes()).hexdigest()
    fit_source = torch.from_numpy(source_features[fit_ids].copy())
    initial_head, initial_proxy = initialize_head_and_classifier(
        fit_source, tuple(map(int, fit_labels))
    )
    class_names, class_ids_np = np.unique(fit_labels, return_inverse=True)
    if len(class_names) != 10_186 or len(batches) != 1_595:
        raise ValueError("SOP cached teacher-tail schedule differs")
    class_ids = torch.from_numpy(class_ids_np.copy()).cuda()
    teacher_gpu = torch.from_numpy(teacher_features.copy()).cuda()
    source = load_authenticated_source_model(args.unicom_checkout, args.checkpoint)
    model = source.encoder.cuda().eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    original_tail = {
        key: value.detach().clone() for key, value in model.blocks[-1].state_dict().items()
    }
    device = torch.device("cuda")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    active_arms = ("tail_arcface_eval",) if args.tail_eval_control_only else ARMS
    for arm in active_arms:
        torch.manual_seed(SEED)
        model.blocks[-1].load_state_dict(original_tail)
        model.eval()
        tail_trainable = arm.startswith("tail_")
        for parameter in model.blocks[-1].parameters():
            parameter.requires_grad_(tail_trainable)
            parameter.grad = None
        if tail_trainable and not args.tail_eval_control_only:
            model.blocks[-1].train()
        head = nn.Linear(768, 128, device=device)
        head.load_state_dict(initial_head.state_dict())
        proxy = nn.Parameter(initial_proxy.detach().clone().to(device))
        optimizer_groups: list[dict[str, object]] = [
            {"params": head.parameters(), "lr": HEAD_LR},
            {"params": [proxy], "lr": PROXY_LR},
        ]
        if tail_trainable:
            optimizer_groups.append({"params": model.blocks[-1].parameters(), "lr": TAIL_LR})
        optimizer = torch.optim.AdamW(optimizer_groups, weight_decay=0.0)
        torch.cuda.synchronize()
        arm_started = time.perf_counter()
        last_arcface = 0.0
        last_teacher = 0.0
        train_batches = batches[:1] if args.preflight_only else batches
        for step, local_rows in enumerate(train_batches):
            global_rows = fit_ids[local_rows]
            tokens = torch.from_numpy(np.asarray(cache[global_rows]).copy()).to(device)
            source_output = output_from_last_block_input(model, tokens)
            compact = compact_head_features(source_output, head)
            targets = class_ids[torch.from_numpy(local_rows).to(device)]
            arcface = arcface_term(compact, proxy, targets)
            teacher_term = (
                relational_term(compact, teacher_gpu[torch.from_numpy(global_rows).to(device)])
                if arm.endswith("teacher")
                else torch.zeros((), device=device)
            )
            loss = arcface + TEACHER_COEFFICIENT * teacher_term
            if not bool(torch.isfinite(loss)):
                raise ValueError(f"SOP cached teacher-tail nonfinite loss in {arm}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            parameters = list(head.parameters()) + [proxy]
            if tail_trainable:
                parameters.extend(model.blocks[-1].parameters())
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            last_arcface = float(arcface.detach())
            last_teacher = float(teacher_term.detach())
            if (step + 1) % 319 == 0:
                print(
                    json.dumps(
                        {
                            "arm": arm,
                            "steps": step + 1,
                            "arcface": last_arcface,
                            "teacher": last_teacher,
                        }
                    ),
                    flush=True,
                )
        torch.cuda.synchronize()
        training_seconds = time.perf_counter() - arm_started
        if args.preflight_only:
            tail_gradient = [parameter.grad for parameter in model.blocks[-1].parameters()]
            if tail_trainable != any(
                gradient is not None and bool((gradient != 0).any()) for gradient in tail_gradient
            ):
                raise ValueError(f"SOP cached teacher-tail gradient gate differs in {arm}")
            results[arm] = {
                "training_seconds": training_seconds,
                "last_arcface_loss": last_arcface,
                "last_teacher_loss": last_teacher,
                "tail_gradient_present": tail_trainable,
            }
            print(json.dumps({"arm": arm, "preflight": results[arm]}), flush=True)
            continue
        eval_started = time.perf_counter()
        score = evaluate_arm(model, head, cache, labels, fit_ids, held_ids, device=device)
        evaluation_seconds = time.perf_counter() - eval_started
        checkpoint = args.output_dir / f"{arm}.pt"
        if checkpoint.exists() or checkpoint.is_symlink():
            raise ValueError("SOP cached teacher-tail checkpoint exists")
        torch.save(
            {
                "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "proxy": proxy.detach().cpu(),
                "arm": arm,
                "source_checkpoint_sha256": source.checkpoint_sha256,
                "cache_sha256": CACHE_SHA256,
            },
            checkpoint,
        )
        results[arm] = {
            "training_seconds": training_seconds,
            "evaluation_seconds": evaluation_seconds,
            "last_arcface_loss": last_arcface,
            "last_teacher_loss": last_teacher,
            "checkpoint_sha256": sha256(checkpoint),
            **score,
        }
        print(
            json.dumps(
                {
                    "arm": arm,
                    "training_seconds": training_seconds,
                    "holdout_r1": score["holdout_only"]["recall_at_1"],
                    "full_r1": score["all_train_gallery"]["recall_at_1"],
                }
            ),
            flush=True,
        )
    receipt = {
        "schema": (
            "sfora-sop-tail-eval-control-preflight-v1"
            if args.tail_eval_control_only and args.preflight_only
            else "sfora-sop-tail-eval-control-screen-v1"
            if args.tail_eval_control_only
            else "sfora-sop-cached-teacher-tail-preflight-v1"
            if args.preflight_only
            else "sfora-sop-cached-teacher-tail-screen-v1"
        ),
        "claim_eligible": False,
        "source_archive_sha256": SOURCE_SHA256,
        "teacher_archive_sha256": TEACHER_SHA256,
        "cache_sha256": CACHE_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "dependencies_sha256": {
            relative: sha256(Path(__file__).resolve().parents[1] / relative)
            for relative in DEPENDENCIES
        },
        "product_pair_schedule_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/sfora/product_pair_schedule.py"
        ),
        "tail_adapter_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/sfora/unicom_tail_adapter.py"
        ),
        "seed": SEED,
        "fit_rows": len(fit_ids),
        "holdout_rows": len(held_ids),
        "fit_products": len(class_names),
        "products_per_batch": PRODUCTS_PER_BATCH,
        "passes": PASSES,
        "updates_per_arm": 1 if args.preflight_only else len(batches),
        "schedule_sha256": schedule_sha256,
        "arcface_margin": MARGIN,
        "arcface_scale": SCALE,
        "teacher_temperature": TEMPERATURE,
        "teacher_coefficient": TEACHER_COEFFICIENT,
        "learning_rates": {"head": HEAD_LR, "tail": TAIL_LR, "proxy": PROXY_LR},
        "wire_bytes_per_gallery_item": 130,
        "results": results,
        "total_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        },
    }
    with receipt_path.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps({"receipt": str(receipt_path), "total_seconds": receipt["total_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
