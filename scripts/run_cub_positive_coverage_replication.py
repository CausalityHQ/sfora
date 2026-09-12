#!/usr/bin/env python3
"""Run the frozen CUB replication of the positive-coverage recipe."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import cast

import numpy as np
import replay_sop_retrieval_local_rank as replay
import run_sop_positive_coverage_metric as coverage
import torch
from positive_coverage_artifacts import (
    PositiveCoverageArtifactPaths,
    positive_coverage_source_identity,
    write_positive_coverage_artifacts,
)
from probe_sop_relational_linear import score_symmetric
from torch import nn

from sfora.deterministic_similarity_runtime import (
    configure_deterministic_similarity_runtime,
)
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import fit_centered_pca
from sfora.teacher_anchored_distillation import (
    class_balanced_anchor_schedule,
    exposure_normalized_update_count,
    retrieval_local_rank_distillation_loss,
)

EXPECTED_B16_SHA256 = "289bf02a28392b788775c4955e0b334dd5c89ca3723bd7c143fcfe622f4a38de"
EXPECTED_L14_SHA256 = "6d8b9ae5d4f546a9324e8b130a5486b6d7fe31945f0f984f43747269b6218852"
SOP_REFERENCE_CLASSES = 9_054
SOP_REFERENCE_UPDATES = 2_000


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path, expected_sha256: str) -> dict[str, object]:
    if _sha256(path) != expected_sha256:
        raise ValueError("CUB archive digest differs")
    with np.load(path, allow_pickle=False) as archive:
        expected = {
            "metadata_json",
            "train_embeddings",
            "train_labels",
            "train_image_ids",
            "train_relative_paths",
            "train_class_names",
            "test_embeddings",
            "test_labels",
            "test_image_ids",
            "test_relative_paths",
            "test_class_names",
        }
        if set(archive.files) != expected:
            raise ValueError("CUB archive schema differs")
        result: dict[str, object] = {
            key: archive[key].copy() for key in expected - {"metadata_json"}
        }
        result["metadata"] = json.loads(str(archive["metadata_json"].item()))
    for split, rows in (("train", 5_864), ("test", 5_924)):
        embeddings = cast(np.ndarray, result[f"{split}_embeddings"])
        labels = cast(np.ndarray, result[f"{split}_labels"])
        identifiers = cast(np.ndarray, result[f"{split}_image_ids"])
        paths = cast(np.ndarray, result[f"{split}_relative_paths"])
        names = cast(np.ndarray, result[f"{split}_class_names"])
        if (
            embeddings.dtype != np.float32
            or embeddings.shape != (rows, 768)
            or not np.isfinite(embeddings).all()
            or labels.dtype != np.int64
            or labels.shape != (rows,)
            or identifiers.dtype != np.int64
            or identifiers.shape != (rows,)
            or paths.shape != (rows,)
            or names.shape != (rows,)
        ):
            raise ValueError("CUB archive arrays differ")
    return result


def _pair(b16: dict[str, object], l14: dict[str, object]) -> None:
    for split in ("train", "test"):
        for suffix in ("labels", "image_ids", "relative_paths", "class_names"):
            left = cast(np.ndarray, b16[f"{split}_{suffix}"])
            right = cast(np.ndarray, l14[f"{split}_{suffix}"])
            if not np.array_equal(left, right):
                raise ValueError("CUB paired archive identity differs")
    train_labels = cast(np.ndarray, b16["train_labels"])
    test_labels = cast(np.ndarray, b16["test_labels"])
    if len(set(train_labels.tolist())) != 100 or len(set(test_labels.tolist())) != 100:
        raise ValueError("CUB class count differs")
    if set(train_labels.tolist()) & set(test_labels.tolist()):
        raise ValueError("CUB train/test classes overlap")


def _score(codes: torch.Tensor, labels: tuple[int, ...], device: torch.device) -> dict[str, object]:
    counts: dict[int, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    width = max(counts.values()) - 1
    floating = score_symmetric(codes, labels, candidate_width=width, device=device)
    packed = score_symmetric(
        pack_int8_unit_embeddings(codes), labels, candidate_width=width, device=device
    )
    return {
        "float_map_at_r": float(floating["map_at_r"]),
        "float_r1": float(floating["r1"]),
        "packed_map_at_r": float(packed["map_at_r"]),
        "packed_per_query_ap": [float(value) for value in packed["per_query_ap"]],
        "packed_r1": float(packed["r1"]),
    }


def _rank_base(
    teacher_train: torch.Tensor,
    teacher_test: torch.Tensor,
    test_labels: tuple[int, ...],
    *,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    dict[str, object],
    dict[str, torch.Tensor],
]:
    pca = fit_centered_pca(teacher_train, dimensions=replay.DIMENSIONS)
    components = pca.components
    mean = pca.mean
    head = nn.Linear(768, replay.DIMENSIONS, device=device, dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(components.to(device))
        head.bias.copy_((-(components.double() @ mean.double())).float().to(device))
    train_cuda = teacher_train.to(device)
    teacher_neighbors = replay._nearest_rows(train_cuda, count=replay.TEACHER_NEIGHBORS)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=replay.EPOCHS, eta_min=1e-5
    )
    rng = np.random.Generator(np.random.PCG64(replay.SEED))
    trajectory: list[dict[str, object]] = []
    for epoch in range(replay.EPOCHS):
        with torch.inference_mode():
            current = replay._unit(head(train_cuda))
            student_neighbors = replay._nearest_rows(current, count=replay.STUDENT_NEIGHBORS)
        candidates = replay._candidate_rows(teacher_neighbors, student_neighbors, rng=rng)
        order = rng.permutation(len(teacher_train))
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
        row = {"epoch": epoch + 1, "mean_training_loss": total / len(order)}
        trajectory.append(row)
        print(json.dumps({"phase": "rank", **row}, sort_keys=True), flush=True)
    with torch.inference_mode():
        base_train = replay._unit(head(train_cuda).cpu())
        base_test = replay._unit(head(teacher_test.to(device)).cpu())
    return (
        base_train,
        base_test,
        {
            "parameter_sha256": replay._parameter_sha256(head.weight, head.bias),
            "score": _score(base_test, test_labels, device),
            "trajectory": trajectory,
        },
        {
            "base_head_weight": head.weight.detach().cpu().contiguous(),
            "base_head_bias": head.bias.detach().cpu().contiguous(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b16", type=Path, required=True)
    parser.add_argument("--l14", type=Path, required=True)
    parser.add_argument("--pooled-checkpoint", type=Path, required=True)
    parser.add_argument("--coverage-checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--execute-positive-coverage", action="store_true", required=True)
    args = parser.parse_args()
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
    b16 = _load(args.b16, EXPECTED_B16_SHA256)
    l14 = _load(args.l14, EXPECTED_L14_SHA256)
    _pair(b16, l14)
    train = replay._unit(torch.from_numpy(cast(np.ndarray, l14["train_embeddings"])))
    test = replay._unit(torch.from_numpy(cast(np.ndarray, l14["test_embeddings"])))
    train_labels = tuple(int(value) for value in cast(np.ndarray, l14["train_labels"]).tolist())
    test_labels = tuple(int(value) for value in cast(np.ndarray, l14["test_labels"]).tolist())
    device = torch.device("cuda")
    teacher_score = _score(test, test_labels, device)
    base_train, base_test, base, base_head_state = _rank_base(
        train, test, test_labels, device=device
    )
    eligible_classes = len(set(train_labels))
    replication_updates = exposure_normalized_update_count(
        eligible_class_count=eligible_classes,
        classes_per_update=coverage.CLASS_COUNT,
        reference_updates=SOP_REFERENCE_UPDATES,
        reference_class_count=SOP_REFERENCE_CLASSES,
        reference_classes_per_update=coverage.CLASS_COUNT,
    )
    schedule_value = class_balanced_anchor_schedule(
        np.asarray(train_labels, dtype=np.int64),
        seed=coverage.SEED,
        updates=replication_updates,
        classes_per_update=coverage.CLASS_COUNT,
        rows_per_class=coverage.ANCHORS_PER_CLASS,
    )
    schedule = tuple(row.copy() for row in schedule_value.row_indexes)
    arms: dict[str, dict[str, object]] = {}
    arm_states: dict[str, dict[str, torch.Tensor]] = {}
    for name in ("pooled", "coverage"):
        arm, state = coverage._train_arm(
            name,
            base_train,
            base_test,
            train_labels,
            test_labels,
            schedule,
            device=device,
        )
        arm["base_head_sha256"] = base["parameter_sha256"]
        state.update(base_head_state)
        arms[name] = arm
        arm_states[name] = state
    control_score = cast(dict[str, object], arms["pooled"]["score"])
    treatment_score = cast(dict[str, object], arms["coverage"]["score"])
    lower = coverage._class_cluster_lower_bound(
        cast(list[float], treatment_score["packed_per_query_ap"]),
        cast(list[float], control_score["packed_per_query_ap"]),
        test_labels,
    )
    base_score = cast(dict[str, object], base["score"])
    gates = {
        "packed_map_at_r": float(base_score["packed_map_at_r"]) + 0.003,
        "packed_r1": float(base_score["packed_r1"]),
        "treatment_minus_control_lower_bound": 0.0,
    }
    passes = (
        float(treatment_score["packed_map_at_r"]) >= gates["packed_map_at_r"]
        and float(treatment_score["packed_r1"]) >= gates["packed_r1"]
        and lower > 0.0
    )
    receipt = {
        "archives": {
            "b16_sha256": EXPECTED_B16_SHA256,
            "l14_sha256": EXPECTED_L14_SHA256,
        },
        "arms": arms,
        "base": base,
        "bootstrap": {
            "lower_quantile": 0.025,
            "samples": coverage.BOOTSTRAP_SAMPLES,
            "treatment_minus_control_lower_bound": lower,
        },
        "claim_eligible": False,
        "dataset": "cub-200-2011-official-train-test",
        "elapsed_seconds": time.monotonic() - started,
        "gates": gates,
        "official_test_touched": True,
        "passes": passes,
        "recipe_origin": "sop-frozen-before-cub",
        "schema": "sfora-positive-coverage-class-exposure-diagnostic-v2",
        "source": source_identity,
        "teacher": teacher_score,
        "test_classes": len(set(test_labels)),
        "test_rows": len(test),
        "train_classes": len(set(train_labels)),
        "train_rows": len(train),
        "training_dose": {
            "reference_classes": SOP_REFERENCE_CLASSES,
            "reference_updates": SOP_REFERENCE_UPDATES,
            "replication_updates": replication_updates,
            "rule": "ceil(reference_updates*dataset_classes/reference_classes)",
            "schedule_sha256": schedule_value.sha256,
        },
    }
    write_positive_coverage_artifacts(
        states=arm_states,
        receipt=receipt,
        paths=artifact_paths,
    )


if __name__ == "__main__":
    main()
