#!/usr/bin/env python3
"""Paired SigLIP2 compact-head training on In-Shop TRAIN identities."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from export_inshop_siglip2_train_features import MODEL_HASHES
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_coverage import PARTITION_SHA256, fit_and_holdout
from torch import nn
from torch.utils.data import DataLoader
from train_sop_siglip2_compact import (
    FixedBatches,
    ImageRows,
    export_all,
    initialize_head_and_classifier,
    make_collate,
    member_bank_initial_values,
    member_bank_positive_ordinals,
    member_bank_rank_loss,
    member_bank_refresh_rows,
    member_bank_refresh_values,
    score_packed_full_gallery,
    sha256,
    training_precision,
)

from sfora.deployed_code_rank import smooth_ap_float_loss
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.sop_compact_training import compact_head_features
from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_rank_finish import identity_balanced_batches
from sfora.unicom_training import sharded_mask_arcface_loss

SEEDS = (179023, 179024, 179025)
UPDATES = 1_000
BATCH_SIZE = 64
RANK_COEFFICIENT = {"bank": 8.0, "float": 21.93}
FIT_SHA256 = "dedc5de723ec62f60c3fe8be87fbae10ab041764bac6a1b3c4c5d9a95de2d2b6"
HELD_SHA256 = "ac320387fb58c1583c334bfd72844975df36f5a46879ec4ce878b7f1409b2953"


def singleton_batch(labels: tuple[str, ...], batch: tuple[int, ...], singleton: set[str]) -> bool:
    return any(labels[row] in singleton for row in batch)


def source_manifest() -> dict[str, str]:
    used = (
        main,
        fit_and_holdout,
        smooth_ap_float_loss,
        pack_int8_unit_embeddings,
        compact_head_features,
        parse_inshop_partition,
        identity_balanced_batches,
        sharded_mask_arcface_loss,
        initialize_head_and_classifier,
        member_bank_positive_ordinals,
        member_bank_rank_loss,
        export_all,
        score_packed_full_gallery,
    )
    paths = {
        Path(filename).resolve()
        for function in used
        if (filename := sys.modules[function.__module__].__file__) is not None
    }
    if len(paths) != 9:
        raise ValueError("In-Shop source manifest differs")
    return {str(path): sha256(path) for path in sorted(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--expected-preflight-sha256", required=True)
    parser.add_argument("--bank-cost", type=Path, required=True)
    parser.add_argument("--expected-bank-cost-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=("bank", "float"), required=True)
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--updates", type=int, choices=(2, 17, UPDATES), default=UPDATES)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if (
        args.output_dir.exists()
        or args.output_dir.is_symlink()
        or args.workers < 0
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA256
        or args.model_snapshot.resolve().name != MODEL_REVISION
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in MODEL_HASHES.items()
        )
        or sha256(args.preflight) != args.expected_preflight_sha256
        or sha256(args.bank_cost) != args.expected_bank_cost_sha256
    ):
        raise ValueError("In-Shop paired training authority differs")
    sources = source_manifest()
    preflight = json.loads(args.preflight.read_text())
    cost = json.loads(args.bank_cost.read_text())
    cache = json.loads((args.features_dir / "receipt.json").read_text())
    if (
        preflight.get("schema") != "sfora-inshop-siglip2-coverage-preflight-v1"
        or preflight.get("partition_sha256") != PARTITION_SHA256
        or cost.get("schema") != "sfora-inshop-siglip2-member-bank-step-cost-v2"
        or cost.get("preflight_sha256") != args.expected_preflight_sha256
        or cost.get("gate_pass") is not True
        or cost.get("rows") != 23_342
        or cache.get("schema") != "sfora-inshop-siglip2-train-feature-export-v1"
        or cache.get("partition_sha256") != PARTITION_SHA256
        or cache.get("model_file_sha256") != MODEL_HASHES
        or sha256(args.features_dir / "train_features.npy") != cache.get("features_sha256")
    ):
        raise ValueError("In-Shop paired training inputs differ")
    records = parse_inshop_partition(args.dataset_root)
    train = tuple(row for row in records if row.split == "train")
    labels = tuple(row.label for row in train)
    fit, held = fit_and_holdout(labels)
    fit_sha = hashlib.sha256(np.asarray(fit, dtype="<i8").tobytes()).hexdigest()
    held_sha = hashlib.sha256(np.asarray(held, dtype="<i8").tobytes()).hexdigest()
    if (
        len(train) != 25_882
        or fit_sha != FIT_SHA256
        or held_sha != HELD_SHA256
        or preflight.get("fit_row_indexes_sha256") != fit_sha
        or preflight.get("holdout_row_indexes_sha256") != held_sha
    ):
        raise ValueError("In-Shop paired training class split differs")
    source = np.load(args.features_dir / "train_features.npy", mmap_mode="r")
    if source.shape != (len(train), 1024) or source.dtype != np.float32:
        raise ValueError("In-Shop feature geometry differs")
    fit_labels = tuple(labels[row] for row in fit)
    names = tuple(sorted(set(fit_labels)))
    class_index = {name: index for index, name in enumerate(names)}
    class_ids = np.asarray([class_index[label] for label in fit_labels], dtype=np.int64)
    counts = Counter(fit_labels)
    singleton = {name for name, count in counts.items() if count == 1}
    schedule = identity_balanced_batches(
        fit_labels,
        batch_size=BATCH_SIZE,
        images_per_identity=4,
        seed=args.seed,
        epoch=1,
        steps=UPDATES,
        coverage_first=True,
    )
    schedule_sha = hashlib.sha256(np.asarray(schedule, dtype="<i4").tobytes()).hexdigest()
    inactive = tuple(
        step
        for step, batch in enumerate(schedule, start=1)
        if singleton_batch(fit_labels, batch, singleton)
    )
    expected = preflight["schedules"][str(args.seed)]
    if (
        schedule_sha != expected["sha256"]
        or list(inactive) != expected["singleton_steps"]
        or len(schedule) - len(inactive) != expected["rank_active_updates"]
        or len({row for batch in schedule for row in batch}) != len(fit)
    ):
        raise ValueError("In-Shop paired training schedule differs")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    features_cpu = torch.from_numpy(np.asarray(source[list(fit)]).copy()).float()
    head, classifier, pca_sha = initialize_head_and_classifier(
        features_cpu, tuple(class_ids.tolist()), allow_singletons=True
    )
    bank_started = time.perf_counter()
    bank_cpu = (
        member_bank_initial_values(features_cpu, head, live_head=False)
        if args.arm == "bank"
        else None
    )
    import transformers
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        args.model_snapshot, local_files_only=True, backend="torchvision"
    )
    if (
        type(processor).__name__ != "SiglipImageProcessor"
        or processor.size["height"] != 256
        or processor.size["width"] != 256
        or processor.resample != 2
    ):
        raise ValueError("In-Shop processor differs")
    full_model = AutoModel.from_pretrained(
        args.model_snapshot, local_files_only=True, use_safetensors=True, dtype=torch.float16
    )
    vision = full_model.vision_model
    del full_model
    vision = vision.float().cuda().train()
    head = head.cuda().train()
    classifier = nn.Parameter(classifier.cuda())
    if args.arm == "bank":
        assert bank_cpu is not None
        bank = bank_cpu.cuda()
        positives = member_bank_positive_ordinals(class_ids, allow_singletons=True).cuda()
        if positives.shape[1] != cost["positives_per_anchor"]:
            raise ValueError("In-Shop bank positive width differs from timing gate")
        schedule_gpu = torch.tensor(schedule, dtype=torch.long, device="cuda")
        class_ids_gpu = torch.from_numpy(class_ids).cuda()
    else:
        bank = None
        positives = None
        schedule_gpu = None
        class_ids_gpu = None
    bank_init_seconds = time.perf_counter() - bank_started if args.arm == "bank" else 0.0
    paths = tuple(row.image_path for row in train)
    dataset = ImageRows(tuple(paths[row] for row in fit), tuple(class_ids.tolist()), augment=True)
    loader = DataLoader(
        dataset,
        batch_sampler=FixedBatches(schedule[: args.updates]),
        num_workers=args.workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=make_collate(processor),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": vision.parameters(), "lr": 1e-5},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    dtype, scaler = training_precision("bf16", device="cuda")
    masks = torch.arange(128, device="cuda", dtype=torch.int64).unsqueeze(0)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses: list[float] = []
    step_seconds: list[float] = []
    first_input_batch_sha256: list[str] = []
    for step, (batch, target) in enumerate(loader, start=1):
        step_started = time.perf_counter()
        if step <= 10:
            digest = hashlib.sha256()
            for key in sorted(batch):
                digest.update(key.encode())
                digest.update(batch[key].contiguous().numpy().tobytes())
            digest.update(target.contiguous().numpy().tobytes())
            first_input_batch_sha256.append(digest.hexdigest())
        tensors = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
        target = target.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=dtype):  # type: ignore[attr-defined]
            source_features = vision(**tensors).pooler_output
        if source_features is None:
            raise ValueError("In-Shop train pooler missing")
        features = compact_head_features(source_features, head)
        control = sharded_mask_arcface_loss(
            features.float(), classifier, target, masks, margin=0.3, scale=64.0
        )
        rank = control.new_zeros(())
        if step not in inactive:
            if args.arm == "bank":
                assert bank is not None and positives is not None
                assert schedule_gpu is not None and class_ids_gpu is not None
                ordinals = schedule_gpu[step - 1]
                if not torch.equal(target, class_ids_gpu[ordinals]):
                    raise ValueError("In-Shop bank schedule labels differ")
                batch_positives = positives[ordinals]
                batch_width = int((batch_positives >= 0).sum(dim=1).max())
                rank = member_bank_rank_loss(
                    features,
                    bank,
                    head,
                    batch_positives[:, :batch_width],
                    ordinals,
                    live_head=False,
                )
            else:
                rank = smooth_ap_float_loss(features.float(), target.tolist())
        loss = control + RANK_COEFFICIENT[args.arm] * rank
        if not bool(torch.isfinite(loss)):
            raise ValueError("In-Shop training loss nonfinite")
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(
            list(vision.parameters()) + list(head.parameters()) + [classifier],
            1.0,
            error_if_nonfinite=True,
        )
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if scaler.get_scale() < before:
            raise ValueError("In-Shop optimizer step skipped")
        if args.arm == "bank":
            assert bank is not None
            refresh_rows, refresh_positions = member_bank_refresh_rows(schedule[step - 1])
            bank[torch.tensor(refresh_rows, device="cuda")] = member_bank_refresh_values(
                source_features,
                features,
                torch.tensor(refresh_positions, device="cuda"),
                live_head=False,
            )
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_started)
        losses.append(float(loss.detach()))
        if step % 50 == 0 or step == args.updates:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": losses[-1],
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                ),
                flush=True,
            )
    training_seconds = time.perf_counter() - started
    training_peak_cuda = torch.cuda.max_memory_allocated()
    if source_manifest() != sources:
        raise ValueError("In-Shop training source changed during execution")
    checkpoint_path = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "vision": {key: value.detach().cpu() for key, value in vision.state_dict().items()},
            "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            "classifier": classifier.detach().cpu(),
            "seed": args.seed,
            "arm": args.arm,
            "updates": args.updates,
        },
        checkpoint_path,
    )
    quality = None
    export_seconds = None
    score_seconds = None
    if args.updates == UPDATES:
        export_started = time.perf_counter()
        values = export_all(
            vision,
            head,
            paths,
            tuple(range(len(train))),
            processor,
            workers=args.workers,
            batch_size=BATCH_SIZE,
        )
        export_seconds = time.perf_counter() - export_started
        packed = pack_int8_unit_embeddings(values)
        encoded = {name: index for index, name in enumerate(sorted(set(labels)))}
        label_ids = torch.tensor([encoded[name] for name in labels], dtype=torch.int64)
        score_started = time.perf_counter()
        quality = score_packed_full_gallery(
            packed.codes.float(),
            packed.inverse_norms,
            label_ids,
            torch.tensor(held, dtype=torch.int64),
            device=torch.device("cuda"),
        )
        score_seconds = time.perf_counter() - score_started
    if source_manifest() != sources:
        raise ValueError("In-Shop evaluation source changed during execution")
    receipt = {
        "schema": "sfora-inshop-siglip2-compact-paired-train-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN; product-disjoint fit/holdout; no official query/gallery",
        "arm": args.arm,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": BATCH_SIZE,
        "rank_coefficient": RANK_COEFFICIENT[args.arm],
        "rank_inactive_steps": list(inactive),
        "rank_active_updates": sum(step not in inactive for step in range(1, args.updates + 1)),
        "source_sha256": sha256(Path(__file__)),
        "source_files_sha256": sources,
        "preflight_sha256": args.expected_preflight_sha256,
        "bank_cost_sha256": args.expected_bank_cost_sha256,
        "feature_receipt_sha256": sha256(args.features_dir / "receipt.json"),
        "features_sha256": cache["features_sha256"],
        "partition_sha256": PARTITION_SHA256,
        "model_file_sha256": MODEL_HASHES,
        "fit_rows_sha256": fit_sha,
        "held_rows_sha256": held_sha,
        "schedule_sha256": schedule_sha,
        "pca_sha256": pca_sha,
        "first_input_batch_sha256": first_input_batch_sha256,
        "training_wall_seconds": training_seconds,
        "training_wall_including_member_bank_init_seconds": training_seconds + bank_init_seconds,
        "member_bank_init_seconds": bank_init_seconds,
        "step_seconds": step_seconds,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "training_peak_cuda_allocated_bytes": training_peak_cuda,
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "export_seconds": export_seconds,
        "score_seconds": score_seconds,
        "quality": quality,
        "checkpoint_sha256": sha256(checkpoint_path),
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "arm": args.arm,
                "seed": args.seed,
                "training_seconds": training_seconds,
                "holdout_r1": quality["recall_at_1"] if quality else None,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
