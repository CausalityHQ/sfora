#!/usr/bin/env python3
"""Compare budget and lower-block freezing on an unseen In-Shop TRAIN gallery."""

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
from export_inshop_siglip2_train_features import MODEL_HASHES, load_vision_init
from export_sop_siglip2_train import MODEL_REVISION
from preflight_inshop_siglip2_unseen_gallery import PARTITION_SHA, digest_rows, schedule, split
from probe_inshop_source_classifier import unused_gradient
from probe_inshop_spatial_parts import CHECKPOINT_SHA
from probe_inshop_teacher_transfer import label_preserving_sham
from probe_inshop_wide_training_head import fold_uncentered_head
from qualify_inshop_wide_checkpoint import matched_checkpoint_checks
from run_inshop_wide_head_smoke import packed_fit_images
from siglip2_base_authority import BASE_HASHES, BASE_REVISION, TRANSFER_SMOKE_SHA
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

from sfora.deployed_code_rank import smooth_ap_bank_loss
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.live_head_bank import live_head_bank_loss
from sfora.representation_ceiling import fit_centered_pca
from sfora.siglip2_compact_serving import Siglip2CompactEncoder
from sfora.sop_compact_training import compact_head_features
from sfora.teacher_anchored_distillation import (
    cross_dimensional_relational_distillation_loss,
    embedding_geometry_diagnostics,
)
from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_rank_finish import identity_balanced_batches
from sfora.unicom_training import sharded_mask_arcface_loss

SEED = 179023
BATCH_SIZE = 64
RANK_COEFFICIENT = 8.0
PREFLIGHT_SHA = "d5e22c6a331acbdf3b143b9836597593c2317f9488b99b72f61a4c54baf18eb8"


def validate_cache_vision_init(cache: dict, expected_sha256: str | None) -> None:
    if cache.get("vision_init_sha256") != expected_sha256:
        raise ValueError("In-Shop cache vision initialization differs")


def singleton_batch(labels: tuple[str, ...], batch: tuple[int, ...], singleton: set[str]) -> bool:
    return any(labels[row] in singleton for row in batch)


def half_fit_products(labels: tuple[str, ...], fit: tuple[int, ...]) -> tuple[int, ...]:
    """Keep a deterministic nested half of the existing fit products."""

    names = sorted(
        {labels[row] for row in fit},
        key=lambda name: (
            hashlib.sha256(b"sfora-inshop-fit-half-v1\0" + name.encode()).digest(),
            name,
        ),
    )
    selected = set(names[: len(names) // 2])
    return tuple(row for row in fit if labels[row] in selected)


def valid_anchor_rank_loss(
    features: torch.Tensor,
    bank: torch.Tensor,
    head: nn.Linear,
    positives: torch.Tensor,
    ordinals: torch.Tensor,
) -> torch.Tensor:
    """Rank eligible anchors while keeping the original batch denominator."""

    valid = (positives >= 0).any(dim=1)
    if not bool(valid.any()):
        raise ValueError("rank batch has no valid anchor")
    return member_bank_rank_loss(
        features[valid], bank, head, positives[valid], ordinals[valid], live_head=False
    ) * (valid.sum() / len(valid))


def source_manifest() -> dict[str, str]:
    used = (
        main,
        load_vision_init,
        split,
        schedule,
        smooth_ap_bank_loss,
        live_head_bank_loss,
        pack_int8_unit_embeddings,
        compact_head_features,
        parse_inshop_partition,
        identity_balanced_batches,
        sharded_mask_arcface_loss,
        initialize_head_and_classifier,
        fit_centered_pca,
        member_bank_positive_ordinals,
        member_bank_rank_loss,
        export_all,
        score_packed_full_gallery,
        unused_gradient,
        label_preserving_sham,
        cross_dimensional_relational_distillation_loss,
        fold_uncentered_head,
        matched_checkpoint_checks,
        packed_fit_images,
        Siglip2CompactEncoder.from_checkpoint,
    )
    paths = {
        Path(filename).resolve()
        for function in used
        if (filename := sys.modules[function.__module__].__file__) is not None
    }
    if len(paths) != 19:
        raise ValueError("In-Shop source manifest differs")
    return {str(path): sha256(path) for path in sorted(paths)}


def schedule_horizon(updates: int) -> int:
    return 3_000 if updates > 1_000 else 1_000


def parameter_digest(module, *, frozen):
    digest = hashlib.sha256()
    for name, value in module.named_parameters():
        if value.requires_grad != frozen:
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def gradient_norm(parameters):
    return float(
        torch.stack(
            [
                parameter.grad.detach().float().norm().square()
                for parameter in parameters
                if parameter.grad is not None
            ]
        )
        .sum()
        .sqrt()
    )


def width_geometry(features):
    unit = torch.nn.functional.normalize(features.float(), dim=1).detach().cpu().contiguous()
    if not torch.isfinite(unit).all() or bool((features.float().norm(dim=1) <= 1e-10).any()):
        raise ValueError("wide smoke projected vector degenerate")
    return {
        "variance": float(unit.var(0, correction=0).sum()),
        "effective_rank": embedding_geometry_diagnostics(unit).effective_rank,
    }


@torch.no_grad()
def calibrate_fit_fold(vision, head, paths, processor, destination, workers, rows_sha):
    """Fit once on the frozen TRAIN-fit rows, never the held gallery."""
    started = time.perf_counter()
    loader = DataLoader(
        ImageRows(paths, tuple(range(len(paths))), augment=False),
        batch_size=32,
        shuffle=False,
        num_workers=workers,
        collate_fn=make_collate(processor),
    )
    outputs, first_sources = [], []
    for batch, _ in loader:
        tensors = {key: value.cuda() for key, value in batch.items()}
        with torch.autocast("cuda", dtype=torch.float16):
            pooled = vision(**tensors).pooler_output
        outputs.append(
            torch.nn.functional.normalize(
                compact_head_features(pooled, head, output_dim=256), dim=1
            ).cpu()
        )
        if len(first_sources) < 2:
            first_sources.append(pooled.cpu())
    first_source = torch.cat(first_sources)
    values = torch.cat(outputs)
    _, singular, right = torch.linalg.svd(values.double(), full_matrices=False)
    if not torch.isfinite(singular).all() or singular[127] <= 1e-10:
        raise ValueError("wide smoke compactor rank differs")
    components = right[:128].clone()
    for row in components:
        if row[row.abs().argmax()] < 0:
            row.neg_()
    components = components.float().contiguous()
    if not torch.allclose(components @ components.T, torch.eye(128), atol=2e-5, rtol=0):
        raise ValueError("wide smoke compactor orthogonality differs")
    wide_cpu = nn.Linear(1024, 256)
    wide_cpu.load_state_dict({name: value.cpu() for name, value in head.state_dict().items()})
    folded = fold_uncentered_head(wide_cpu, components)
    source = torch.nn.functional.normalize(first_source.float(), dim=1)
    composed = torch.nn.functional.normalize(
        torch.nn.functional.normalize(wide_cpu(source), dim=1) @ components.T, dim=1
    )
    folded_values = torch.nn.functional.normalize(folded(source), dim=1)
    error = float((composed - folded_values).abs().max())
    if error > 1e-5 or bool((folded(source).norm(dim=1) <= 1e-10).any()):
        raise ValueError("wide smoke fold identity differs")
    folded_gpu = folded.cuda()
    folded_values = torch.cat(
        [
            torch.nn.functional.normalize(
                compact_head_features(part.cuda(), folded_gpu), dim=1
            ).cpu()
            for part in first_sources
        ]
    )
    composed = torch.cat(
        [
            torch.nn.functional.normalize(
                torch.nn.functional.normalize(
                    compact_head_features(part.cuda(), head, output_dim=256), dim=1
                )
                @ components.cuda().T,
                dim=1,
            ).cpu()
            for part in first_sources
        ]
    )
    error = max(error, float((composed - folded_values).abs().max()))
    if error > 1e-5:
        raise ValueError("wide smoke GPU fold identity differs")
    state = torch.load(destination / "checkpoint.pt", map_location="cpu", weights_only=True)
    state["head"] = {name: value.cpu() for name, value in folded.state_dict().items()}
    state["compactor"] = components
    state["calibration_rows_sha256"] = rows_sha
    state["parent_checkpoint_sha256"] = sha256(destination / "checkpoint.pt")
    torch.save(state, destination / "folded_checkpoint.pt")
    torch.save(
        {"pooled": first_source, "folded": folded_values, "composed": composed},
        destination / "fold_probe.pt",
    )
    return folded_gpu, {
        "fit_rows": len(paths),
        "calibration_wall_seconds": time.perf_counter() - started,
        "fit_rows_sha256": rows_sha,
        "singular_values": singular.tolist(),
        "retained_energy": float(singular[:128].square().sum() / singular.square().sum()),
        "compactor_sha256": hashlib.sha256(components.numpy().tobytes()).hexdigest(),
        "fold_error": error,
        "serving_width": 128,
        "folded_checkpoint_sha256": sha256(destination / "folded_checkpoint.pt"),
    }


def frozen_source_centroid_loss(
    source: torch.Tensor,
    mean: torch.Tensor,
    prototypes: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Training-only cosine CE64 against fixed, fit-centered source prototypes."""
    if (
        source.ndim != 2
        or mean.shape != (source.shape[1],)
        or prototypes.ndim != 2
        or prototypes.shape[1] != source.shape[1]
        or not all(torch.isfinite(value).all() for value in (source, mean, prototypes))
        or bool((source.float().norm(dim=1) == 0).any())
        or bool((prototypes.float().norm(dim=1) == 0).any())
    ):
        raise ValueError("frozen source centroid geometry differs")
    with torch.autocast(device_type=source.device.type, enabled=False):
        centered = torch.nn.functional.normalize(source.float(), dim=1) - mean.detach().float()
        if bool((centered.norm(dim=1) == 0).any()):
            raise ValueError("frozen source centered vector is zero")
        return sharded_mask_arcface_loss(
            centered,
            prototypes.detach().float(),
            target,
            torch.arange(source.shape[1], device=source.device).unsqueeze(0),
            margin=0,
            scale=64,
        )


def source_smoke_probe(
    vision,
    head,
    classifier,
    mean,
    prototypes,
    tensors,
    target,
    bank,
    positives,
    ordinals,
    rank_active,
    sham_index,
):
    """Separate objective gradients on the SAME fixed fit batch, before clipping."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        source = vision(**tensors).pooler_output
    features = compact_head_features(source, head)
    masks = torch.arange(128, device="cuda").unsqueeze(0)
    main = sharded_mask_arcface_loss(
        features.float(), classifier, target, masks, margin=0.3, scale=64
    )
    if rank_active:
        width = int((positives >= 0).sum(1).max())
        main = main + 8 * member_bank_rank_loss(
            features, bank, head, positives[:, :width], ordinals, live_head=False
        )
    auxiliary = frozen_source_centroid_loss(source, mean, prototypes, target)
    sham_loss = frozen_source_centroid_loss(source, mean, prototypes, sham_index[target])
    source_grad = torch.autograd.grad(auxiliary, source, retain_graph=True)[0].float()
    sham_source_grad = torch.autograd.grad(sham_loss, source, retain_graph=True)[0].float()
    # Head rows cease to be orthonormal after the first optimizer step.
    basis = torch.linalg.qr(head.weight.detach().T, mode="reduced").Q.T
    unit_source = torch.nn.functional.normalize(source.detach().float(), dim=1)
    unused = unused_gradient(source_grad, basis, unit_source)
    sham_unused = unused_gradient(sham_source_grad, basis, unit_source)
    parameters = tuple(parameter for parameter in vision.parameters() if parameter.requires_grad)
    main_grad = torch.autograd.grad(main, parameters, retain_graph=True)
    auxiliary_grad = torch.autograd.grad(
        0.1 * auxiliary,
        parameters + tuple(head.parameters()) + (classifier,),
        retain_graph=True,
        allow_unused=True,
    )
    if any(value is not None for value in auxiliary_grad[len(parameters) :]):
        raise ValueError("source auxiliary reached compact head/classifier")
    auxiliary_grad = auxiliary_grad[: len(parameters)]
    if any(value is None or not torch.isfinite(value).all() for value in auxiliary_grad):
        raise ValueError("source auxiliary encoder gradient missing/nonfinite")

    def norm(values):
        return torch.stack([value.float().square().sum() for value in values]).sum().sqrt()

    main_norm, auxiliary_norm = norm(main_grad), norm(auxiliary_grad)
    cosine = torch.stack(
        [(a.float() * b.float()).sum() for a, b in zip(main_grad, auxiliary_grad, strict=True)]
    ).sum()
    cosine = cosine / (main_norm * auxiliary_norm)
    with torch.no_grad():
        centered_source = torch.nn.functional.normalize(source.float(), dim=1) - mean
        logits = 64 * (torch.nn.functional.normalize(centered_source, dim=1) @ prototypes.T)
        probabilities = logits.softmax(1)[torch.arange(len(target), device="cuda"), target]
        compact = torch.nn.functional.normalize(features.float(), dim=1)
        centered = compact - compact.mean(0)
        spectrum = torch.linalg.svdvals(centered).square()
    report = {
        "main_loss": float(main.detach()),
        "auxiliary_loss": float(auxiliary.detach()),
        "sham_auxiliary_loss": float(sham_loss.detach()),
        "outside_span_norm_fraction": float(
            (unused.norm(dim=1) / source_grad.norm(dim=1).clamp_min(1e-30)).median()
        ),
        "unused_true_sham_cosine": float(
            torch.nn.functional.cosine_similarity(unused, sham_unused, dim=1).median()
        ),
        "main_encoder_gradient_norm": float(main_norm),
        "weighted_auxiliary_encoder_gradient_norm": float(auxiliary_norm),
        "weighted_auxiliary_to_main_gradient_ratio": float(auxiliary_norm / main_norm),
        "encoder_gradient_cosine": float(cosine),
        "saturated_target_fraction": float((probabilities > 1 - 1e-6).float().mean()),
        "compact_effective_rank": float(spectrum.sum().square() / spectrum.square().sum()),
        "compact_centered_variance": float(centered.square().mean()),
        "auxiliary_head_gradient": False,
    }
    if not all(np.isfinite(value) for value in report.values()):
        raise ValueError("source smoke diagnostic nonfinite")
    return report


def main() -> None:
    entire_started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--vision-init-checkpoint", type=Path)
    parser.add_argument("--vision-init-sha256")
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--arm",
        choices=(
            "control",
            "budget",
            "freeze",
            "freeze_emb",
            "freeze_emb_rank",
            "freeze_emb_mapr",
            "freeze_emb_live",
            "subspace",
        ),
        required=True,
    )
    parser.add_argument(
        "--updates", type=int, choices=(17, 100, 1_000, 1_533, 3_000), required=True
    )
    parser.add_argument(
        "--seed", type=int, choices=(179023, 179024, 179025, 179026, 179027), default=SEED
    )
    parser.add_argument("--vision-lr", type=float, choices=(1e-5, 3e-5), default=1e-5)
    parser.add_argument("--preflight-sha256", default=PREFLIGHT_SHA)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--half-fit-products", action="store_true")
    parser.add_argument("--tail-blocks-to-drop", type=int, choices=(0, 2), default=0)
    parser.add_argument("--freeze-first-blocks", type=int, choices=(12, 16), default=12)
    parser.add_argument("--source-centroid-smoke", choices=("control", "auxiliary"))
    parser.add_argument("--source-centroid-receipt", type=Path)
    parser.add_argument("--teacher-transfer", choices=("control", "true", "sham"))
    parser.add_argument("--teacher-transfer-smoke-receipt", type=Path)
    parser.add_argument("--teacher-model-snapshot", type=Path)
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument("--training-width", type=int, choices=(128, 256), default=128)
    parser.add_argument("--wide-head-smoke-receipt", type=Path)
    parser.add_argument("--wide-head-qualification", type=Path)
    args = parser.parse_args()
    transfer = args.teacher_transfer is not None
    width_quality = args.wide_head_qualification is not None
    if width_quality and (
        args.wide_head_smoke_receipt is not None
        or sha256(args.wide_head_qualification)
        != "eb4c8c1bf2a059277b92493fc71612d1f4ce9b53504c3ae1d6346155bf01591e"
    ):
        raise ValueError("wide100 qualification authority differs")
    wide_smoke = args.wide_head_smoke_receipt is not None or width_quality
    if (args.training_width != 128 and not wide_smoke) or (
        wide_smoke
        and (
            args.arm != "freeze_emb"
            or args.seed != 179024
            or args.updates != (100 if width_quality else 17)
            or args.freeze_first_blocks != 12
            or args.half_fit_products
            or args.tail_blocks_to_drop
            or args.vision_init_checkpoint
            or args.source_centroid_smoke
            or transfer
            or (
                not width_quality
                and sha256(args.wide_head_smoke_receipt)
                != "2698063c7c5720b85dbd643bde678d61550c6a65dffbb4c598910d49779d474e"
            )
        )
    ):
        raise ValueError("wide head smoke authority differs")
    if transfer:
        if (
            args.arm != "control"
            or args.seed != 179024
            or args.updates != 100
            or args.half_fit_products
            or args.tail_blocks_to_drop
            or args.vision_init_checkpoint
            or args.source_centroid_smoke
            or args.teacher_transfer_smoke_receipt is None
            or sha256(args.teacher_transfer_smoke_receipt) != TRANSFER_SMOKE_SHA
            or args.teacher_checkpoint is None
            or sha256(args.teacher_checkpoint) != CHECKPOINT_SHA
            or args.teacher_model_snapshot is None
            or args.teacher_model_snapshot.name != MODEL_REVISION
            or any(
                sha256(args.teacher_model_snapshot / name) != digest
                for name, digest in MODEL_HASHES.items()
            )
        ):
            raise ValueError("teacher transfer protocol authority differs")
    elif any(
        value is not None
        for value in (
            args.teacher_transfer_smoke_receipt,
            args.teacher_model_snapshot,
            args.teacher_checkpoint,
        )
    ):
        raise ValueError("teacher options require transfer method")
    model_revision, model_hashes, source_width, depth = (
        (BASE_REVISION, BASE_HASHES, 768, 12)
        if transfer
        else (MODEL_REVISION, MODEL_HASHES, 1024, 24)
    )
    if (
        args.output_dir.exists()
        or ((args.source_centroid_smoke is None) != (args.source_centroid_receipt is None))
        or (
            args.source_centroid_smoke is not None
            and (
                args.arm != "freeze_emb"
                or args.seed != 179024
                or args.updates != 17
                or args.freeze_first_blocks != 12
                or args.half_fit_products
                or args.tail_blocks_to_drop
                or args.vision_init_checkpoint is not None
            )
        )
        or args.output_dir.is_symlink()
        or (args.vision_init_checkpoint is None) != (args.vision_init_sha256 is None)
        or (args.arm == "budget") != (args.updates == 3_000)
        or (
            args.updates == 1_533
            and (args.arm != "freeze_emb" or args.seed not in (179024, 179026))
        )
        or (args.vision_lr != 1e-5 and (args.arm != "control" or args.seed == 179023))
        or (args.arm == "subspace" and (args.seed == 179023 or args.vision_lr != 1e-5))
        or (
            args.arm in ("freeze_emb", "freeze_emb_rank", "freeze_emb_mapr", "freeze_emb_live")
            and (args.seed == 179023 or args.vision_lr != 1e-5)
        )
        or args.workers < 0
        or (
            args.half_fit_products
            and (args.arm != "freeze_emb" or args.seed != 179024 or args.updates not in (17, 1_000))
        )
        or (
            args.tail_blocks_to_drop
            and (args.arm != "freeze_emb" or args.seed != 179026 or args.updates not in (17, 100))
        )
        or (
            args.freeze_first_blocks == 16
            and (
                args.arm not in ("freeze_emb", "freeze_emb_mapr", "freeze_emb_live")
                or args.seed != 179024
                or args.updates not in (17, 100, 1_000)
                or args.half_fit_products
                or args.tail_blocks_to_drop
            )
        )
        or (
            args.arm == "freeze_emb_mapr"
            and (
                args.freeze_first_blocks != 16
                or args.seed != 179024
                or args.updates not in (17, 1_000)
                or args.half_fit_products
                or args.tail_blocks_to_drop
            )
        )
        or (
            args.arm == "freeze_emb_live"
            and (args.freeze_first_blocks != 16 or args.seed != 179024)
        )
        or not torch.cuda.is_available()
        or not torch.cuda.is_bf16_supported()
        or sha256(args.dataset_root / "Eval/list_eval_partition.txt") != PARTITION_SHA
        or args.model_snapshot.resolve().name != model_revision
        or any(
            sha256(args.model_snapshot / name) != digest for name, digest in model_hashes.items()
        )
        or sha256(args.preflight) != args.preflight_sha256
    ):
        raise ValueError("In-Shop paired training authority differs")
    sources = source_manifest()
    preflight = json.loads(args.preflight.read_text())
    cache = json.loads((args.features_dir / "receipt.json").read_text())
    validate_cache_vision_init(cache, args.vision_init_sha256)
    if (
        preflight.get("schema") != "sfora-inshop-siglip2-unseen-gallery-preflight-v1"
        or preflight.get("seed") != args.seed
        or preflight.get("partition_sha256") != PARTITION_SHA
        or cache.get("schema") != "sfora-inshop-siglip2-train-feature-export-v1"
        or cache.get("partition_sha256") != PARTITION_SHA
        or cache.get("model_file_sha256") != model_hashes
        or cache.get("tail_blocks_dropped", 0) != args.tail_blocks_to_drop
        or sha256(args.features_dir / "train_features.npy") != cache.get("features_sha256")
    ):
        raise ValueError("In-Shop paired training inputs differ")
    records = parse_inshop_partition(args.dataset_root)
    train = tuple(row for row in records if row.split == "train")
    labels = tuple(row.label for row in train)
    ordered_rows_sha = hashlib.sha256(
        "\n".join(
            f"{row.label}\0{row.image_path.relative_to(args.dataset_root)}" for row in train
        ).encode()
    ).hexdigest()
    if cache.get("ordered_rows_sha256") != ordered_rows_sha:
        raise ValueError("In-Shop cached features do not match training row order")
    full_fit, held = split(labels)
    fit_sha = digest_rows(full_fit)
    held_sha = digest_rows(held)
    if (
        len(train) != 25_882
        or len(full_fit) != 13_283
        or len(held) != 12_599
        or preflight.get("fit_sha256") != fit_sha
        or preflight.get("held_sha256") != held_sha
    ):
        raise ValueError("In-Shop paired training class split differs")
    fit = half_fit_products(labels, full_fit) if args.half_fit_products else full_fit
    fit_sha = digest_rows(fit)
    source = np.load(args.features_dir / "train_features.npy", mmap_mode="r")
    if source.shape != (len(train), source_width) or source.dtype != np.float32:
        raise ValueError("In-Shop feature geometry differs")
    if transfer and (
        cache.get("exported_fit_only") is not True or cache.get("fit_sha256") != fit_sha
    ):
        raise ValueError("student fit-only cache authority differs")
    fit_labels = tuple(labels[row] for row in fit)
    names = tuple(sorted(set(fit_labels)))
    class_index = {name: index for index, name in enumerate(names)}
    class_ids = np.asarray([class_index[label] for label in fit_labels], dtype=np.int64)
    counts = Counter(fit_labels)
    singleton = {name for name, count in counts.items() if count == 1}
    schedule_updates = schedule_horizon(args.updates)
    batches = schedule(fit_labels, schedule_updates, seed=args.seed)
    if schedule_updates == 3_000 and batches[:1_000] != schedule(fit_labels, 1_000, seed=args.seed):
        raise ValueError("In-Shop budget schedule does not extend control")
    schedule_sha = hashlib.sha256(np.asarray(batches, dtype="<i4").tobytes()).hexdigest()
    inactive = tuple(
        step
        for step, batch in enumerate(batches, start=1)
        if singleton_batch(fit_labels, batch, singleton)
    )
    expected = preflight["schedules"][str(schedule_updates)]
    if (
        not args.half_fit_products
        and (
            schedule_sha != expected["sha256"]
            or list(inactive) != expected["rank_inactive_steps"]
            or len(batches) - len(inactive) != expected["rank_active_updates"]
        )
    ) or len({row for batch in batches for row in batch}) != len(fit):
        raise ValueError("In-Shop paired training schedule differs")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(16)
    torch.backends.cuda.matmul.allow_tf32 = False
    features_cpu = torch.from_numpy(np.asarray(source[list(fit)]).copy()).float()
    head_init_started = time.perf_counter()
    head, classifier, pca_sha = initialize_head_and_classifier(
        features_cpu, tuple(class_ids.tolist()), allow_singletons=True
    )
    if args.training_width == 256:
        narrow_weight = head.weight.detach().clone()
        with torch.random.fork_rng(devices=[]):
            head, classifier, pca_sha = initialize_head_and_classifier(
                features_cpu, tuple(class_ids.tolist()), allow_singletons=True, output_dim=256
            )
        if not torch.equal(head.weight[:128], narrow_weight):
            raise ValueError("wide initialization changes native subspace")
    head_init_seconds = time.perf_counter() - head_init_started
    source_mean = source_prototypes = None
    source_sham_index = None
    if args.source_centroid_smoke:
        if (
            sha256(args.source_centroid_receipt)
            != "ace3f13e92bc0357c195b307da3fde128780f33a9f7a9a8c12af7b0b8200f29f"
        ):
            raise ValueError("source centroid cached gate authority differs")
        sham_mapping = json.loads(args.source_centroid_receipt.read_text())["sham_label_mapping"]
        source_sham_index = torch.tensor(
            [class_index[sham_mapping.get(name, name)] for name in names], device="cuda"
        )
        unit = torch.nn.functional.normalize(features_cpu, dim=1)
        source_mean = unit.mean(0).cuda()
        centered = torch.nn.functional.normalize(unit - source_mean.cpu(), dim=1)
        sums = torch.zeros(len(names), 1024)
        sums.index_add_(0, torch.from_numpy(class_ids), centered)
        source_prototypes = torch.nn.functional.normalize(sums, dim=1).cuda()
    bank_started = time.perf_counter()
    live_head = args.arm == "freeze_emb_live"
    bank_cpu = member_bank_initial_values(
        features_cpu, head, live_head=live_head, output_dim=args.training_width
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
        args.model_snapshot,
        local_files_only=True,
        use_safetensors=True,
        dtype=torch.float32 if transfer else torch.float16,
    )
    vision = full_model.vision_model
    del full_model
    if len(vision.encoder.layers) != depth:
        raise ValueError("In-Shop SigLIP2 encoder depth differs")
    if args.tail_blocks_to_drop:
        vision.encoder.layers = nn.ModuleList(
            list(vision.encoder.layers[: -args.tail_blocks_to_drop])
        )
    if args.vision_init_checkpoint is not None:
        load_vision_init(vision, args.vision_init_checkpoint, args.vision_init_sha256)
    vision = vision.float().cuda().train()
    teacher = teacher_head = None
    if transfer and args.teacher_transfer != "control":
        from transformers import AutoConfig, SiglipVisionModel

        teacher = SiglipVisionModel(
            AutoConfig.from_pretrained(
                args.teacher_model_snapshot, local_files_only=True
            ).vision_config
        ).float()
        teacher_head = nn.Linear(1024, 128)
        teacher_state = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=True)
        teacher.load_state_dict(teacher_state["vision"], strict=True)
        teacher_head.load_state_dict(teacher_state["head"], strict=True)
        if any(
            not torch.equal(value, teacher_state["vision"][name])
            for name, value in teacher.state_dict().items()
        ) or any(
            not torch.equal(value, teacher_state["head"][name])
            for name, value in teacher_head.state_dict().items()
        ):
            raise ValueError("teacher FP32 load parity differs")
        del teacher_state
        teacher = teacher.cuda().eval().requires_grad_(False)
        teacher_head = teacher_head.cuda().eval().requires_grad_(False)
        torch.manual_seed(args.seed)
    if args.arm in (
        "freeze",
        "freeze_emb",
        "freeze_emb_rank",
        "freeze_emb_mapr",
        "freeze_emb_live",
    ):
        for block in vision.encoder.layers[: args.freeze_first_blocks]:
            block.requires_grad_(False)
    if args.arm in ("freeze_emb", "freeze_emb_rank", "freeze_emb_mapr", "freeze_emb_live"):
        vision.embeddings.requires_grad_(False)
    head = head.cuda().train()
    classifier = nn.Parameter(classifier.cuda())
    bank = bank_cpu.cuda()
    positives = member_bank_positive_ordinals(class_ids, allow_singletons=True).cuda()
    schedule_gpu = torch.tensor(batches, dtype=torch.long, device="cuda")
    class_ids_gpu = torch.from_numpy(class_ids).cuda()
    bank_init_seconds = time.perf_counter() - bank_started
    paths = tuple(row.image_path for row in train)
    dataset = ImageRows(tuple(paths[row] for row in fit), tuple(class_ids.tolist()), augment=True)
    loader = DataLoader(
        dataset,
        batch_sampler=FixedBatches(batches[: args.updates]),
        num_workers=args.workers,
        pin_memory=True,
        generator=torch.Generator().manual_seed(args.seed),
        collate_fn=make_collate(processor),
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": vision.parameters(), "lr": args.vision_lr},
            {"params": head.parameters(), "lr": 1e-4},
            {"params": [classifier], "lr": 1e-4},
        ],
        weight_decay=0.05,
    )
    dtype, scaler = training_precision("bf16", device="cuda")
    masks = torch.arange(args.training_width, device="cuda", dtype=torch.int64).unsqueeze(0)
    mask_rng = torch.Generator().manual_seed(args.seed + 128_000)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    losses: list[float] = []
    step_seconds: list[float] = []
    preclip_grad_norms: list[float] = []
    first_input_batch_sha256: list[str] = []
    rank_active_updates = 0
    recovered_rank_updates = 0
    smoke_probes = []
    auxiliary_losses = []
    transfer_losses = []
    teacher_forward_seconds = []
    fixed_smoke_batch = None
    width_history = []
    width_initial_geometry = None
    frozen_sha = parameter_digest(vision, frozen=True) if wide_smoke else None
    initial_trainable_sha = parameter_digest(vision, frozen=False) if wide_smoke else None
    started = time.perf_counter()
    for step, (batch, target) in enumerate(loader, start=1):
        step_started = time.perf_counter()
        if step <= 10 or args.source_centroid_smoke or transfer or wide_smoke:
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
        features = compact_head_features(source_features, head, output_dim=args.training_width)
        if wide_smoke and step == 1:
            fixed_smoke_batch = tensors, target
            width_initial_geometry = width_geometry(features.detach())
        if args.arm == "subspace":
            coordinates = torch.randperm(128, generator=mask_rng)[:64].sort().values.cuda()
            masks = coordinates.unsqueeze(0)
        control = sharded_mask_arcface_loss(
            features.float(), classifier, target, masks, margin=0.3, scale=64.0
        )
        rank = control.new_zeros(())
        if step not in inactive or args.arm == "freeze_emb_rank":
            ordinals = schedule_gpu[step - 1]
            if not torch.equal(target, class_ids_gpu[ordinals]):
                raise ValueError("In-Shop bank schedule labels differ")
            batch_positives = positives[ordinals]
            batch_width = int((batch_positives >= 0).sum(dim=1).max())
            if args.arm == "subspace":
                rank = smooth_ap_bank_loss(
                    torch.nn.functional.normalize(
                        features.float().index_select(1, coordinates), dim=1
                    ),
                    torch.nn.functional.normalize(bank.index_select(1, coordinates), dim=1),
                    batch_positives[:, :batch_width],
                    ordinals,
                )
            elif step in inactive:
                rank = valid_anchor_rank_loss(
                    features, bank, head, batch_positives[:, :batch_width], ordinals
                )
                recovered_rank_updates += 1
            else:
                rank = member_bank_rank_loss(
                    features,
                    bank,
                    head,
                    batch_positives[:, :batch_width],
                    ordinals,
                    live_head=live_head,
                    truncate_at_r=args.arm == "freeze_emb_mapr",
                )
            rank_active_updates += 1
        loss = control + RANK_COEFFICIENT * rank
        if teacher is not None:
            teacher_started = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype):
                teacher_source = teacher(**tensors).pooler_output
            teacher_codes = (
                torch.nn.functional.normalize(
                    compact_head_features(teacher_source, teacher_head), dim=1
                )
                .detach()
                .contiguous()
            )
            torch.cuda.synchronize()
            teacher_forward_seconds.append(time.perf_counter() - teacher_started)
            if args.teacher_transfer == "sham":
                teacher_codes = label_preserving_sham(teacher_codes, target)
            relation = cross_dimensional_relational_distillation_loss(
                torch.nn.functional.normalize(features.float(), dim=1).contiguous(),
                teacher_codes,
                temperatures=(0.20,),
            )
            transfer_losses.append(float(relation.detach()))
            loss = loss + 0.1 * relation
        if args.source_centroid_smoke:
            if step == 1:
                fixed_smoke_batch = (tensors, target)
                smoke_probes.append(
                    source_smoke_probe(
                        vision,
                        head,
                        classifier,
                        source_mean,
                        source_prototypes,
                        tensors,
                        target,
                        bank,
                        positives[schedule_gpu[0]],
                        schedule_gpu[0],
                        1 not in inactive,
                        source_sham_index,
                    )
                )
                (args.output_dir / "source_probe_initial.json").write_text(
                    json.dumps(smoke_probes[-1], sort_keys=True, allow_nan=False) + "\n"
                )
                if smoke_probes[-1]["weighted_auxiliary_to_main_gradient_ratio"] > 0.25:
                    raise ValueError("source auxiliary gradient dominates main objective")
            auxiliary = frozen_source_centroid_loss(
                source_features, source_mean, source_prototypes, target
            )
            auxiliary_losses.append(float(auxiliary.detach()))
            if args.source_centroid_smoke == "auxiliary":
                loss = loss + 0.1 * auxiliary
        if not bool(torch.isfinite(loss)):
            raise ValueError("In-Shop training loss nonfinite")
        scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
        scaler.unscale_(optimizer)
        if wide_smoke:
            width_history.append(
                {
                    "arcface": float(control.detach()),
                    "bank": float(rank.detach()),
                    "group_gradient_norms": {
                        name: gradient_norm(parameters)
                        for name, parameters in (
                            ("vision", vision.parameters()),
                            ("head", head.parameters()),
                            ("classifier", (classifier,)),
                        )
                    },
                }
            )
        preclip_grad_norm = torch.nn.utils.clip_grad_norm_(
            list(vision.parameters()) + list(head.parameters()) + [classifier],
            1.0,
            error_if_nonfinite=True,
        )
        preclip_grad_norms.append(float(preclip_grad_norm))
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        if (args.source_centroid_smoke or transfer or wide_smoke) and any(
            not torch.isfinite(parameter).all()
            for parameter in list(vision.parameters()) + list(head.parameters()) + [classifier]
        ):
            raise ValueError("source smoke parameter nonfinite")
        if wide_smoke and (
            not torch.isfinite(bank).all()
            or any(
                not torch.isfinite(value).all()
                for state in optimizer.state.values()
                for value in state.values()
                if isinstance(value, torch.Tensor)
            )
        ):
            raise ValueError("wide smoke optimizer/bank nonfinite")
        if scaler.get_scale() < before:
            raise ValueError("In-Shop optimizer step skipped")
        refresh_rows, refresh_positions = member_bank_refresh_rows(batches[step - 1])
        bank[torch.tensor(refresh_rows, device="cuda")] = member_bank_refresh_values(
            source_features,
            features,
            torch.tensor(refresh_positions, device="cuda"),
            live_head=live_head,
        )
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - step_started)
        losses.append(float(loss.detach()))
        if args.source_centroid_smoke and time.perf_counter() - entire_started > 120:
            raise ValueError("source smoke whole-arm wall exceeded paired budget")
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
    expected_recovered = (
        sum(step <= args.updates for step in inactive) if args.arm == "freeze_emb_rank" else 0
    )
    if (
        recovered_rank_updates != expected_recovered
        or rank_active_updates
        != args.updates - sum(step <= args.updates for step in inactive) + expected_recovered
    ):
        raise ValueError("In-Shop rank execution count differs")
    training_peak_cuda = torch.cuda.max_memory_allocated()
    width_terminal_geometry = None
    if wide_smoke:
        if frozen_sha != parameter_digest(
            vision, frozen=True
        ) or initial_trainable_sha == parameter_digest(vision, frozen=False):
            raise ValueError("wide smoke frozen/update authority differs")
        vision.eval()
        with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
            pooled = vision(**fixed_smoke_batch[0]).pooler_output
        with torch.no_grad():
            width_terminal_geometry = width_geometry(
                compact_head_features(pooled, head, output_dim=args.training_width)
            )
    if args.source_centroid_smoke:
        tensors, target = fixed_smoke_batch
        smoke_probes.append(
            source_smoke_probe(
                vision,
                head,
                classifier,
                source_mean,
                source_prototypes,
                tensors,
                target,
                bank,
                positives[schedule_gpu[0]],
                schedule_gpu[0],
                1 not in inactive,
                source_sham_index,
            )
        )
        training_peak_cuda = torch.cuda.max_memory_allocated()
        (args.output_dir / "source_probe_terminal.json").write_text(
            json.dumps(smoke_probes[-1], sort_keys=True, allow_nan=False) + "\n"
        )
    if source_manifest() != sources:
        raise ValueError("In-Shop training source changed during execution")
    checkpoint_path = args.output_dir / "checkpoint.pt"
    if not args.source_centroid_smoke:
        torch.save(
            {
                "vision": {key: value.detach().cpu() for key, value in vision.state_dict().items()},
                "head": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "training_evidence": {
                    "updates": args.updates,
                    "first_input_batch_sha256": first_input_batch_sha256,
                    "all_step_losses": losses,
                    "training_wall_seconds": training_seconds,
                    "training_peak_cuda_allocated_bytes": training_peak_cuda,
                    "source_files_sha256": sources,
                    "fit_rows_sha256": fit_sha,
                    "schedule_sha256": schedule_sha,
                }
                if wide_smoke
                else None,
                "training_width": args.training_width,
                "classifier": classifier.detach().cpu(),
                "seed": args.seed,
                "arm": args.arm,
                "updates": args.updates,
                "teacher_transfer": args.teacher_transfer,
                "model_revision": model_revision,
                "source_width": source_width,
                "teacher_checkpoint_sha256": CHECKPOINT_SHA if transfer else None,
                "vision_lr": args.vision_lr,
                "vision_init_sha256": args.vision_init_sha256,
                "half_fit_products": args.half_fit_products,
                "tail_blocks_dropped": args.tail_blocks_to_drop,
                "freeze_first_blocks": args.freeze_first_blocks,
                "live_head_bank": live_head,
            },
            checkpoint_path,
        )
    width_fold = None
    export_head = head
    if wide_smoke and args.training_width == 256:
        calibration_rows = fit if width_quality else fit[:2048]
        export_head, width_fold = calibrate_fit_fold(
            vision,
            head,
            tuple(paths[row] for row in calibration_rows),
            processor,
            args.output_dir,
            args.workers,
            digest_rows(calibration_rows),
        )
    quality = None
    matched_parity = None
    if width_quality:
        deployed_path = (
            args.output_dir / "folded_checkpoint.pt"
            if args.training_width == 256
            else checkpoint_path
        )
        matched_parity = matched_checkpoint_checks(
            args.model_snapshot,
            checkpoint_path,
            deployed_path,
            sha256(checkpoint_path),
            sha256(deployed_path),
            tuple(paths[row] for row in fit[:64]),
        )
        if not all(matched_parity.values()):
            raise ValueError("wide100 matched public32 parity failed")
    export_seconds = None
    score_seconds = None
    if args.updates >= 100:
        export_started = time.perf_counter()
        values = export_all(
            vision,
            export_head,
            tuple(paths[row] for row in held),
            tuple(range(len(held))),
            processor,
            workers=args.workers,
            batch_size=32 if width_quality else BATCH_SIZE,
        )
        export_seconds = time.perf_counter() - export_started
        np.save(args.output_dir / "held_values.npy", values.numpy())
        packed = pack_int8_unit_embeddings(values)
        held_labels = tuple(labels[row] for row in held)
        encoded = {name: index for index, name in enumerate(sorted(set(held_labels)))}
        label_ids = torch.tensor([encoded[name] for name in held_labels], dtype=torch.int64)
        score_started = time.perf_counter()
        quality = score_packed_full_gallery(
            packed.codes.float(),
            packed.inverse_norms,
            label_ids,
            torch.arange(len(held), dtype=torch.int64),
            device=torch.device("cuda"),
        )
        score_seconds = time.perf_counter() - score_started
    if source_manifest() != sources:
        raise ValueError("In-Shop evaluation source changed during execution")
    receipt = {
        "schema": "sfora-inshop-siglip2-unseen-gallery-train-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN; product-disjoint half split; held-only symmetric gallery",
        "arm": args.arm,
        "seed": args.seed,
        "updates": args.updates,
        "batch_size": BATCH_SIZE,
        "workers": args.workers,
        "rank_coefficient": RANK_COEFFICIENT,
        "vision_lr": args.vision_lr,
        "vision_init_sha256": args.vision_init_sha256,
        "training_coordinates": 64 if args.arm == "subspace" else 128,
        "rank_inactive_steps": (
            []
            if args.arm == "freeze_emb_rank"
            else [step for step in inactive if step <= args.updates]
        ),
        "formerly_rank_inactive_steps": (
            [step for step in inactive if step <= args.updates]
            if args.arm == "freeze_emb_rank"
            else []
        ),
        "rank_active_updates": rank_active_updates,
        "source_sha256": sha256(Path(__file__)),
        "source_files_sha256": sources,
        "preflight_sha256": args.preflight_sha256,
        "feature_receipt_sha256": sha256(args.features_dir / "receipt.json"),
        "features_sha256": cache["features_sha256"],
        "partition_sha256": PARTITION_SHA,
        "model_file_sha256": model_hashes,
        "teacher_transfer": args.teacher_transfer,
        "training_width": args.training_width,
        "wide_head_smoke": wide_smoke,
        "wide_head_qualification_sha256": sha256(args.wide_head_qualification)
        if width_quality
        else None,
        "export_batch_size": 32 if width_quality else BATCH_SIZE,
        "matched_public32_parity": matched_parity,
        "width_history": width_history,
        "width_initial_geometry": width_initial_geometry,
        "width_terminal_geometry": width_terminal_geometry,
        "width_fold": width_fold,
        "teacher_checkpoint_sha256": CHECKPOINT_SHA if transfer else None,
        "teacher_transfer_losses": transfer_losses,
        "teacher_forward_seconds": teacher_forward_seconds,
        "fit_rows_sha256": fit_sha,
        "full_fit_rows_sha256": digest_rows(full_fit),
        "half_fit_products": args.half_fit_products,
        "tail_blocks_dropped": args.tail_blocks_to_drop,
        "freeze_first_blocks": args.freeze_first_blocks,
        "live_head_bank": live_head,
        "bank_width": bank.shape[1],
        "held_rows_sha256": held_sha,
        "schedule_sha256": schedule_sha,
        "executed_schedule_sha256": hashlib.sha256(
            np.asarray(batches[: args.updates], dtype="<i4").tobytes()
        ).hexdigest(),
        "fit_rows": len(fit),
        "held_rows": len(held),
        "frozen_encoder_blocks": list(range(args.freeze_first_blocks))
        if args.arm
        in ("freeze", "freeze_emb", "freeze_emb_rank", "freeze_emb_mapr", "freeze_emb_live")
        else [],
        "frozen_embeddings": args.arm
        in ("freeze_emb", "freeze_emb_rank", "freeze_emb_mapr", "freeze_emb_live"),
        "recovered_rank_updates": recovered_rank_updates,
        "pca_sha256": pca_sha,
        "head_classifier_init_seconds": head_init_seconds,
        "first_input_batch_sha256": first_input_batch_sha256,
        "training_wall_seconds": training_seconds,
        "training_wall_including_member_bank_init_seconds": training_seconds + bank_init_seconds,
        "member_bank_init_seconds": bank_init_seconds,
        "step_seconds": step_seconds,
        "preclip_grad_norms": preclip_grad_norms,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "all_step_losses": losses if args.source_centroid_smoke or transfer or wide_smoke else None,
        "source_centroid_smoke": args.source_centroid_smoke,
        "source_centroid_probes": smoke_probes,
        "source_centroid_auxiliary_losses": auxiliary_losses,
        "whole_arm_wall_seconds": time.perf_counter() - entire_started,
        "training_peak_cuda_allocated_bytes": training_peak_cuda,
        "peak_parent_host_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "export_seconds": export_seconds,
        "score_seconds": score_seconds,
        "held_values_sha256": sha256(args.output_dir / "held_values.npy") if quality else None,
        "quality": quality,
        "checkpoint_sha256": sha256(checkpoint_path) if checkpoint_path.exists() else None,
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
