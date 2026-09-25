#!/usr/bin/env python3
"""Fit-only SOP probe of ArcFace against a live-head full source bank."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from train_sop_siglip2_compact import (
    ARCHIVE_SHA256,
    NATIVE_SHA256,
    OUTPUT_WIDTH,
    TILEIRAS_SHA256,
    evaluate_packed,
    initialize_head_and_classifier,
    member_bank_positive_ordinals,
    ordered_rows_sha256,
    sha256,
)

from sfora.live_head_bank import live_head_bank_loss
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import (
    CompactTrainingArm,
    compact_head_features,
    compact_training_terms,
)
from sfora.unicom_rank_finish import identity_balanced_batches

FEATURE_SHA256 = "d15f76e459b90836df2807260a34d06692e2f2e5b8ba17b3449ed56500b8357a"
SEED = 179023
STEPS = 1_000
ARMS = ("arcface", "live_head_bank")
EXPECTED_SCHEDULE_SHA256 = "82ec2c5394892596cf111c2f0f152c94e7e4469e1ec11b6b23124f4c09f95a62"
EXPECTED_HEAD_SHA256 = "8205919e7d74ac75fe41d7b495a883f39ef24b8acfe7514e77f97ef1d98d3d59"
EXPECTED_CLASSIFIER_SHA256 = "9cc5193ff242252e5beb1a2716ea72563bcf758a4663e83089180396f1c3568e"
EXPECTED_PCA_SHA256 = "7ce07200c115e241f775c535e0ebe08fa3f2fdb28dfcfb3699f61af3a16fbcc9"
SOURCE_RELATIVES = (
    "scripts/train_sop_siglip2_cached_head_probe.py",
    "scripts/train_sop_siglip2_compact.py",
    "src/sfora/live_head_bank.py",
    "src/sfora/sop_compact_training.py",
    "src/sfora/unicom_rank_finish.py",
    "src/sfora/unicom_training.py",
    "src/sfora/representation_ceiling.py",
    "src/sfora/joint_relational_compaction.py",
    "src/sfora/cutile_int8.py",
)


def validate_export(receipt: dict, archive: Path, features: Path, native: Path) -> float:
    """Bind the cached descriptors and their acquisition cost."""

    if (
        receipt.get("schema") != "sfora-sop-siglip2-train-feature-export-v1"
        or receipt.get("rows") != 59_551
        or receipt.get("dimension") != 1_024
        or receipt.get("features_sha256") != FEATURE_SHA256
        or receipt.get("unicom_train_archive_sha256") != ARCHIVE_SHA256
        or not isinstance(receipt.get("encode_wall_seconds"), (int, float))
        or not math.isfinite(receipt["encode_wall_seconds"])
        or receipt["encode_wall_seconds"] <= 0
        or sha256(archive) != ARCHIVE_SHA256
        or sha256(features) != FEATURE_SHA256
        or sha256(native) != NATIVE_SHA256
    ):
        raise ValueError("SOP cached-head source authority differs")
    return float(receipt["encode_wall_seconds"])


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    return {relative: sha256(root / relative) for relative in SOURCE_RELATIVES}


def initial_hashes(head: nn.Linear, classifier: nn.Parameter) -> tuple[str, str]:
    weight = head.weight.detach().cpu().numpy().tobytes()
    bias = head.bias.detach().cpu().numpy().tobytes()
    proxy = classifier.detach().cpu().numpy().tobytes()
    return hashlib.sha256(weight + bias).hexdigest(), hashlib.sha256(proxy).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--native-library", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--arm", choices=ARMS, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink() or not torch.cuda.is_available():
        raise ValueError("SOP cached-head output or CUDA differs")
    receipt_path = args.candidate_dir / "receipt.json"
    features_path = args.candidate_dir / "train_features.npy"
    export_receipt = json.loads(receipt_path.read_text())
    cache_seconds = validate_export(
        export_receipt, args.source_archive, features_path, args.native_library
    )
    with np.load(args.source_archive, allow_pickle=False) as archive:
        labels = np.asarray(archive["train_labels"], dtype=np.int64)
        ids = np.asarray(archive["train_image_ids"], dtype=np.int64)
        relatives = np.asarray(archive["train_relative_paths"])
    if (
        labels.shape != (59_551,)
        or ids.shape != labels.shape
        or relatives.shape != labels.shape
        or ordered_rows_sha256(ids, labels, relatives) != export_receipt.get("ordered_rows_sha256")
    ):
        raise ValueError("SOP cached-head TRAIN rows differ")
    cached = np.load(features_path, mmap_mode="r", allow_pickle=False)
    if cached.shape != (59_551, 1_024) or cached.dtype != np.float32:
        raise ValueError("SOP cached-head features differ")
    partition = deterministic_class_partition(
        tuple(map(int, labels)), fit_fraction=0.9, seed=179019
    )
    fit = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if len(fit) != 53_700 or len(held) != 5_851 or len(set(map(int, labels[fit]))) != 10_186:
        raise ValueError("SOP cached-head fit split differs")
    fit_labels = tuple(map(int, labels[fit]))
    class_names = tuple(sorted(set(fit_labels)))
    class_index = {name: index for index, name in enumerate(class_names)}
    fit_class_ids = np.asarray([class_index[label] for label in fit_labels], dtype=np.int64)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    initial_started = time.perf_counter()
    fit_features = torch.from_numpy(np.asarray(cached[fit]).copy()).float()
    head, classifier, pca_sha = initialize_head_and_classifier(fit_features, fit_labels)
    head_sha, classifier_sha = initial_hashes(head, classifier)
    schedule = identity_balanced_batches(
        tuple(map(str, fit_labels)),
        batch_size=64,
        images_per_identity=4,
        seed=SEED,
        epoch=1,
        steps=STEPS,
    )
    schedule_sha = hashlib.sha256(np.asarray(schedule, dtype="<i4").tobytes()).hexdigest()
    if (
        schedule_sha != EXPECTED_SCHEDULE_SHA256
        or head_sha != EXPECTED_HEAD_SHA256
        or classifier_sha != EXPECTED_CLASSIFIER_SHA256
        or pca_sha != EXPECTED_PCA_SHA256
    ):
        raise ValueError("SOP cached-head initial weights or schedule differ")
    source_bank = F.normalize(fit_features.cuda(), dim=1)
    all_features = torch.from_numpy(np.asarray(cached).copy()).cuda()
    head = head.cuda().train()
    classifier = nn.Parameter(classifier.cuda())
    class_ids = torch.from_numpy(fit_class_ids).cuda()
    positives = member_bank_positive_ordinals(fit_class_ids).cuda()
    masks = torch.arange(OUTPUT_WIDTH, dtype=torch.int64, device="cuda").unsqueeze(0)
    initial_seconds = time.perf_counter() - initial_started
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if not tileiras or sha256(Path(tileiras)) != TILEIRAS_SHA256:
        raise ValueError("SOP cached-head CuTile toolchain differs")
    source_files = source_manifest()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    optimizer = torch.optim.AdamW(
        [{"params": head.parameters(), "lr": 1e-4}, {"params": [classifier], "lr": 1e-4}],
        weight_decay=0.05,
    )
    torch.cuda.reset_peak_memory_stats()
    step_seconds: list[float] = []
    first_batch_hashes: list[str] = []
    first_loss = math.nan
    last_loss = math.nan
    last_arcface_loss = math.nan
    last_rank_loss = math.nan
    train_started = time.perf_counter()
    for step, row_indexes in enumerate(schedule, start=1):
        step_started = time.perf_counter()
        ordinals = torch.tensor(row_indexes, dtype=torch.long, device="cuda")
        target = class_ids[ordinals]
        if step <= 10:
            first_batch_hashes.append(
                hashlib.sha256(
                    np.asarray(row_indexes, dtype="<i4").tobytes()
                    + target.cpu().numpy().astype("<i8").tobytes()
                ).hexdigest()
            )
        optimizer.zero_grad(set_to_none=True)
        features = compact_head_features(source_bank[ordinals], head)
        control, _ = compact_training_terms(
            features,
            classifier,
            target,
            masks,
            arm=CompactTrainingArm.ARCFACE,
            arcface_margin=0.3,
            arcface_scale=64.0,
        )
        rank = (
            live_head_bank_loss(
                F.normalize(features, dim=1), source_bank, head, positives[ordinals], ordinals
            )
            if args.arm == "live_head_bank"
            else control.new_zeros(())
        )
        loss = control + 8.0 * rank
        if not bool(torch.isfinite(loss)):
            raise ValueError("SOP cached-head loss is nonfinite")
        if step == 1:
            first_loss = float(loss.detach())
        if step == STEPS:
            last_loss = float(loss.detach())
            last_arcface_loss = float(control.detach())
            last_rank_loss = float(rank.detach())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(head.parameters()) + [classifier], 1.0, error_if_nonfinite=True
        )
        optimizer.step()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_started)
        if step % 100 == 0:
            print(
                json.dumps({"step": step, "elapsed_seconds": time.perf_counter() - train_started}),
                flush=True,
            )
    training_seconds = time.perf_counter() - train_started
    training_peak = torch.cuda.max_memory_allocated()
    head.eval()
    values = []
    export_started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(all_features), 512):
            values.append(compact_head_features(all_features[start : start + 512], head).cpu())
        embeddings = F.normalize(torch.cat(values), dim=1)
    embedding_seconds = time.perf_counter() - export_started
    quality = evaluate_packed(embeddings, labels, fit, held, args.native_library)
    if source_manifest() != source_files:
        raise ValueError("SOP cached-head source changed during probe")
    checkpoint = args.output_dir / "checkpoint.pt"
    torch.save(
        {
            "arm": args.arm,
            "seed": SEED,
            "steps": STEPS,
            "head": head.cpu().state_dict(),
            "classifier": classifier.cpu(),
        },
        checkpoint,
    )
    result = {
        "schema": "sfora-sop-siglip2-cached-head-probe-v1",
        "claim_eligible": False,
        "split": "SOP official TRAIN product-disjoint holdout; full TRAIN gallery",
        "arm": args.arm,
        "seed": SEED,
        "steps": STEPS,
        "source_archive_sha256": ARCHIVE_SHA256,
        "query_image_ids_sha256": hashlib.sha256(ids[held].tobytes()).hexdigest(),
        "source_cache_sha256": FEATURE_SHA256,
        "export_receipt_sha256": sha256(receipt_path),
        "source_files_sha256": source_files,
        "native_library_sha256": NATIVE_SHA256,
        "tileiras_sha256": TILEIRAS_SHA256,
        "initial_head_sha256": head_sha,
        "initial_classifier_sha256": classifier_sha,
        "source_pca_sha256": pca_sha,
        "schedule_sha256": schedule_sha,
        "first_input_batch_sha256": first_batch_hashes,
        "first_loss": first_loss,
        "last_loss": last_loss,
        "last_arcface_loss": last_arcface_loss,
        "last_rank_loss": last_rank_loss,
        "cache_encode_seconds": cache_seconds,
        "head_initialization_seconds": initial_seconds,
        "training_wall_seconds": training_seconds,
        "accounted_cache_plus_head_train_seconds": cache_seconds
        + initial_seconds
        + training_seconds,
        "head_export_seconds": embedding_seconds,
        "step_seconds": step_seconds,
        "training_peak_cuda_allocated_bytes": training_peak,
        "train_plus_evaluation_peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(),
        "quality": quality,
        "checkpoint_sha256": sha256(checkpoint),
        "hardware": {
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    with (args.output_dir / "receipt.json").open("xb") as stream:
        stream.write((json.dumps(result, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(json.dumps({"arm": args.arm, "packed_r1": quality["recall_at_1"]}), flush=True)


if __name__ == "__main__":
    main()
