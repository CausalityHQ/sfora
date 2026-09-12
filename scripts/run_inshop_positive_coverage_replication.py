#!/usr/bin/env python3
"""Run the frozen In-Shop confirmation of exposure-normalized positive coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import cast

import numpy as np
import probe_inshop_relational_linear as inshop
import replay_sop_retrieval_local_rank as replay
import run_sop_positive_coverage_metric as coverage
import torch
from positive_coverage_artifacts import (
    PositiveCoverageArtifactPaths,
    positive_coverage_source_identity,
    write_positive_coverage_artifacts,
)
from torch import nn

from sfora.deterministic_similarity_runtime import (
    configure_deterministic_similarity_runtime,
)
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.teacher_anchored_distillation import (
    ClassBalancedAnchorSchedule,
    class_balanced_anchor_schedule,
    exposure_normalized_update_count,
    positive_coverage_hard_negative_loss,
    retrieval_local_rank_distillation_loss,
)

B16_SHA256 = "730764705dc7dbacefd9c5d0ba1d9f2d65f1b9cbbebd62da84e97fd0d9548a29"
L14_SHA256 = "6eae13715e18d7eb99450bade5056538f8f08f1e9b550d0f24ee09e52bb25d0e"
SOP_REFERENCE_CLASSES = 9_054
SOP_REFERENCE_UPDATES = 2_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, digest: str) -> dict[str, object]:
    if _sha256(path) != digest:
        raise ValueError("In-Shop archive digest differs")
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "metadata_json",
            "train_embeddings",
            "train_labels",
            "query_embeddings",
            "query_labels",
            "gallery_embeddings",
            "gallery_labels",
        }
        if set(archive.files) != expected:
            raise ValueError("In-Shop archive schema differs")
        result = {key: archive[key].copy() for key in expected - {"metadata_json"}}
        result["metadata"] = json.loads(str(archive["metadata_json"].item()))
    for split, rows in (("train", 25_882), ("query", 14_218), ("gallery", 12_612)):
        embeddings = cast(np.ndarray, result[f"{split}_embeddings"])
        labels = cast(np.ndarray, result[f"{split}_labels"])
        if (
            embeddings.dtype != np.float32
            or embeddings.shape != (rows, 768)
            or not np.isfinite(embeddings).all()
            or labels.shape != (rows,)
            or labels.dtype.kind != "U"
        ):
            raise ValueError("In-Shop archive arrays differ")
    return result


def _pair(left: dict[str, object], right: dict[str, object]) -> None:
    for split in ("train", "query", "gallery"):
        if not np.array_equal(left[f"{split}_labels"], right[f"{split}_labels"]):
            raise ValueError("In-Shop paired labels differ")


def _eligible_schedule(labels: tuple[int, ...], *, updates: int) -> ClassBalancedAnchorSchedule:
    label_array = np.asarray(labels, dtype=np.int64)
    return class_balanced_anchor_schedule(
        label_array,
        seed=coverage.SEED,
        updates=updates,
        classes_per_update=coverage.CLASS_COUNT,
        rows_per_class=coverage.ANCHORS_PER_CLASS,
    )


def _score(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_labels: tuple[str, ...],
    gallery_labels: tuple[str, ...],
    *,
    device: torch.device,
) -> dict[str, object]:
    floating = inshop._score(query, gallery, query_labels, gallery_labels, device=device)
    packed = inshop._score(
        pack_int8_unit_embeddings(query),
        pack_int8_unit_embeddings(gallery),
        query_labels,
        gallery_labels,
        device=device,
    )
    return {
        "float_map_at_r": float(floating["map_at_r"]),
        "float_r1": float(floating["r1"]),
        "packed_map_at_r": float(packed["map_at_r"]),
        "packed_per_query_ap": [float(value) for value in packed["per_query_ap"]],
        "packed_r1": float(packed["r1"]),
    }


def _fit_rank_head(train: torch.Tensor, *, device: torch.device) -> nn.Linear:
    pca = fit_centered_pca(train, dimensions=replay.DIMENSIONS)
    components = pca.components
    mean = pca.mean
    head = nn.Linear(768, replay.DIMENSIONS, device=device, dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(components.to(device))
        head.bias.copy_((-(components.double() @ mean.double())).float().to(device))
    train_cuda = train.to(device)
    teacher_neighbors = replay._nearest_rows(train_cuda, count=replay.TEACHER_NEIGHBORS)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=replay.EPOCHS, eta_min=1e-5
    )
    rng = np.random.Generator(np.random.PCG64(replay.SEED))
    for epoch in range(replay.EPOCHS):
        with torch.inference_mode():
            current = replay._unit(head(train_cuda))
            student_neighbors = replay._nearest_rows(current, count=replay.STUDENT_NEIGHBORS)
        candidates = replay._candidate_rows(teacher_neighbors, student_neighbors, rng=rng)
        order = rng.permutation(len(train))
        total = 0.0
        for start in range(0, len(order), replay.BATCH_SIZE):
            anchors_np = order[start : start + replay.BATCH_SIZE]
            anchors = torch.from_numpy(anchors_np).to(device)
            candidate = torch.from_numpy(candidates[anchors_np]).to(device)
            anchor_codes = replay._unit(head(train_cuda[anchors]))
            candidate_codes = replay._unit(
                head(train_cuda[candidate.reshape(-1)]).reshape(
                    len(anchors), replay.CANDIDATE_WIDTH, -1
                )
            )
            student = torch.einsum("bd,bkd->bk", anchor_codes, candidate_codes).contiguous()
            teacher = (
                torch.einsum(
                    "bd,bkd->bk",
                    train_cuda[anchors],
                    train_cuda[candidate.reshape(-1)].reshape(
                        len(anchors), replay.CANDIDATE_WIDTH, -1
                    ),
                )
                .detach()
                .contiguous()
            )
            loss = retrieval_local_rank_distillation_loss(
                student,
                teacher,
                teacher_neighbor_count=replay.TEACHER_NEIGHBORS,
                margin_cap=0.05,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(anchors_np)
        scheduler.step()
        print(
            json.dumps(
                {"epoch": epoch + 1, "loss": total / len(order), "phase": "rank"},
                sort_keys=True,
            ),
            flush=True,
        )
    return head


def _fit_transform(
    name: str,
    base_train: torch.Tensor,
    train_labels: tuple[int, ...],
    schedule: tuple[np.ndarray, ...],
    *,
    device: torch.device,
) -> tuple[nn.Linear, float]:
    transform = nn.Linear(128, 128, bias=False, device=device)
    with torch.no_grad():
        transform.weight.copy_(torch.eye(128, device=device))
    optimizer = torch.optim.Adam(transform.parameters(), lr=coverage.LEARNING_RATE)
    fit = base_train.to(device)
    label_tensor = torch.tensor(train_labels, dtype=torch.int64, device=device)
    label_array = np.asarray(train_labels, dtype=np.int64)
    groups = {label: np.flatnonzero(label_array == label) for label in set(train_labels)}
    final_loss = math.nan
    for step, anchors_np in enumerate(schedule, start=1):
        anchors = torch.from_numpy(anchors_np).to(device)
        with torch.inference_mode():
            bank = replay._unit(transform(fit))
            negatives = coverage._stable_hard_negatives(
                bank[anchors], bank, label_tensor[anchors], label_tensor
            )
        positives_np, mask_np = coverage._positive_rows(anchors_np, label_array, groups)
        positives = torch.from_numpy(positives_np).to(device)
        mask = torch.from_numpy(mask_np).to(device).contiguous()
        anchor_codes = replay._unit(transform(fit[anchors]))
        positive_codes = replay._unit(
            transform(fit[positives.reshape(-1)]).reshape(len(anchors), positives.shape[1], -1)
        )
        negative_codes = replay._unit(
            transform(fit[negatives.reshape(-1)]).reshape(len(anchors), coverage.HARD_NEGATIVES, -1)
        )
        loss = positive_coverage_hard_negative_loss(
            torch.einsum("bd,bpd->bp", anchor_codes, positive_codes).contiguous(),
            mask,
            torch.einsum("bd,bnd->bn", anchor_codes, negative_codes).contiguous(),
            torch.einsum("bd,bd->b", anchor_codes, fit[anchors]).contiguous(),
            temperature=coverage.TEMPERATURE,
            margin=coverage.MARGIN,
            anchor_weight=coverage.ANCHOR_WEIGHT,
            positive_aggregation=name,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(transform.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach())
        if step == 1 or step % 100 == 0 or step == len(schedule):
            print(
                json.dumps({"arm": name, "loss": final_loss, "step": step}, sort_keys=True),
                flush=True,
            )
    return transform, final_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b16", type=Path, required=True)
    parser.add_argument("--l14", type=Path, required=True)
    parser.add_argument("--pooled-checkpoint", type=Path, required=True)
    parser.add_argument("--coverage-checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--execute-positive-coverage", action="store_true", required=True)
    args = parser.parse_args()
    coverage.SEED = args.seed
    artifact_paths = PositiveCoverageArtifactPaths(
        pooled_checkpoint=args.pooled_checkpoint,
        coverage_checkpoint=args.coverage_checkpoint,
        complete_receipt=args.receipt,
    )
    source_identity = positive_coverage_source_identity(
        driver=Path(__file__).resolve(),
        driver_sha256=args.driver_sha256,
        source_revision=args.source_revision,
    )
    configure_deterministic_similarity_runtime(coverage.SEED, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    started = time.monotonic()
    b16 = _load(args.b16, B16_SHA256)
    l14 = _load(args.l14, L14_SHA256)
    _pair(b16, l14)
    device = torch.device("cuda")
    train = replay._unit(torch.from_numpy(cast(np.ndarray, l14["train_embeddings"])))
    query = replay._unit(torch.from_numpy(cast(np.ndarray, l14["query_embeddings"])))
    gallery = replay._unit(torch.from_numpy(cast(np.ndarray, l14["gallery_embeddings"])))
    train_names = tuple(str(value) for value in cast(np.ndarray, l14["train_labels"]).tolist())
    query_labels = tuple(str(value) for value in cast(np.ndarray, l14["query_labels"]).tolist())
    gallery_labels = tuple(str(value) for value in cast(np.ndarray, l14["gallery_labels"]).tolist())
    if set(train_names) & (set(query_labels) | set(gallery_labels)):
        raise ValueError("In-Shop train/test identities overlap")
    train_identity = {name: index + 1 for index, name in enumerate(sorted(set(train_names)))}
    train_labels = tuple(train_identity[name] for name in train_names)
    teacher_score = _score(query, gallery, query_labels, gallery_labels, device=device)
    head = _fit_rank_head(train, device=device)
    with torch.inference_mode():
        base_train = replay._unit(head(train.to(device)).cpu())
        base_query = replay._unit(head(query.to(device)).cpu())
        base_gallery = replay._unit(head(gallery.to(device)).cpu())
    base_score = _score(base_query, base_gallery, query_labels, gallery_labels, device=device)
    base_head_sha256 = replay._parameter_sha256(head.weight, head.bias)
    base_head_state = {
        "base_head_weight": head.weight.detach().cpu().contiguous(),
        "base_head_bias": head.bias.detach().cpu().contiguous(),
    }
    _labels, class_counts = np.unique(np.asarray(train_labels), return_counts=True)
    eligible_classes = int(np.sum(class_counts >= coverage.ANCHORS_PER_CLASS))
    updates = exposure_normalized_update_count(
        eligible_class_count=eligible_classes,
        classes_per_update=coverage.CLASS_COUNT,
        reference_updates=SOP_REFERENCE_UPDATES,
        reference_class_count=SOP_REFERENCE_CLASSES,
        reference_classes_per_update=coverage.CLASS_COUNT,
    )
    schedule_authority = _eligible_schedule(train_labels, updates=updates)
    schedule = tuple(row.copy() for row in schedule_authority.row_indexes)
    arms: dict[str, dict[str, object]] = {}
    arm_states: dict[str, dict[str, torch.Tensor]] = {}
    for name in ("pooled", "coverage"):
        transform, final_loss = _fit_transform(
            name, base_train, train_labels, schedule, device=device
        )
        with torch.inference_mode():
            transformed_query = replay._unit(transform(base_query.to(device)).cpu())
            transformed_gallery = replay._unit(transform(base_gallery.to(device)).cpu())
        arms[name] = {
            "base_head_sha256": base_head_sha256,
            "final_loss": final_loss,
            "parameter_sha256": replay._parameter_sha256(transform.weight),
            "score": _score(
                transformed_query,
                transformed_gallery,
                query_labels,
                gallery_labels,
                device=device,
            ),
        }
        arm_states[name] = {
            "weight": transform.weight.detach().cpu().contiguous(),
            **base_head_state,
        }
    control = cast(dict[str, object], arms["pooled"]["score"])
    treatment = cast(dict[str, object], arms["coverage"]["score"])
    query_identity = {name: index + 1 for index, name in enumerate(sorted(set(query_labels)))}
    lower = coverage._class_cluster_lower_bound(
        cast(list[float], treatment["packed_per_query_ap"]),
        cast(list[float], control["packed_per_query_ap"]),
        tuple(query_identity[name] for name in query_labels),
    )
    gates = {
        "packed_map_at_r": float(base_score["packed_map_at_r"]) + 0.003,
        "packed_r1": float(base_score["packed_r1"]),
        "treatment_minus_control_lower_bound": 0.0,
    }
    passes = (
        float(treatment["packed_map_at_r"]) >= gates["packed_map_at_r"]
        and float(treatment["packed_r1"]) >= gates["packed_r1"]
        and lower > 0.0
    )
    receipt = {
        "archives": {"b16_sha256": B16_SHA256, "l14_sha256": L14_SHA256},
        "arms": arms,
        "base": {
            "parameter_sha256": base_head_sha256,
            "score": base_score,
        },
        "bootstrap": {
            "note": "query-identity-cluster bootstrap with fixed asymmetric gallery",
            "treatment_minus_control_lower_bound": lower,
        },
        "claim_eligible": False,
        "dataset": "in-shop-clothes-retrieval-official-train-query-gallery",
        "elapsed_seconds": time.monotonic() - started,
        "gates": gates,
        "passes": passes,
        "recipe_origin": "sop-class-exposure-rule-frozen-before-inshop",
        "schema": "sfora-positive-coverage-inshop-replication-v2",
        "seed": args.seed,
        "source": source_identity,
        "teacher": teacher_score,
        "training_dose": {
            "eligible_classes": eligible_classes,
            "reference_classes": SOP_REFERENCE_CLASSES,
            "reference_updates": SOP_REFERENCE_UPDATES,
            "replication_updates": updates,
            "schedule_sha256": schedule_authority.sha256,
        },
    }
    write_positive_coverage_artifacts(
        states=arm_states,
        receipt=receipt,
        paths=artifact_paths,
    )


if __name__ == "__main__":
    main()
