#!/usr/bin/env python3
"""Reproduce the sealed SOP retrieval-local-rank base checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
from positive_coverage_artifacts import (
    positive_coverage_source_identity,
    write_canonical_receipt_no_clobber,
    write_linear_replay_artifacts,
)
from probe_representation_ceiling import load_paired_train_archives
from probe_sop_relational_linear import score_symmetric
from torch import nn

from sfora.deterministic_similarity_runtime import (
    configure_deterministic_similarity_runtime,
)
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import (
    deterministic_class_partition,
    fit_centered_pca,
)
from sfora.teacher_anchored_distillation import (
    retrieval_local_rank_distillation_loss,
)

SEED = 17
DIMENSIONS = 128
TEACHER_NEIGHBORS = 64
STUDENT_NEIGHBORS = 64
CANDIDATE_WIDTH = 192
BATCH_SIZE = 128
EPOCHS = 10
EXPECTED_MAP = 0.5556011035874895
EXPECTED_R1 = 0.8111758251034017
EXPECTED_HEAD_SHA256 = "acd49a3854a238a84d760c5499ab5f07e53a48c9ef995253a636ad3276f8b6ee"
REPLAY_TOLERANCE = 1e-4


def _unit(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(value.float(), dim=-1).contiguous()


def _parameter_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        cpu = value.detach().cpu().float().contiguous()
        digest.update(struct.pack("<I", cpu.ndim))
        digest.update(struct.pack(f"<{cpu.ndim}Q", *cpu.shape))
        digest.update(cpu.numpy().astype("<f4", copy=False).tobytes(order="C"))
    return digest.hexdigest()


@torch.inference_mode()
def _nearest_rows(value: torch.Tensor, *, count: int, block: int = 256) -> np.ndarray:
    result = np.empty((len(value), count), dtype=np.int64)
    transposed = value.T.contiguous()
    for start in range(0, len(value), block):
        stop = min(start + block, len(value))
        scores = value[start:stop] @ transposed
        local = torch.arange(stop - start, device=value.device)
        scores[local, torch.arange(start, stop, device=value.device)] = -torch.inf
        # Similarity ties are vanishingly rare for these normalized float embeddings.
        rows = torch.topk(scores, count, dim=1, largest=True, sorted=True).indices
        result[start:stop] = rows.cpu().numpy()
    return result


def _candidate_rows(
    teacher_rows: np.ndarray,
    student_rows: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    row_count = len(teacher_rows)
    output = np.empty((row_count, CANDIDATE_WIDTH), dtype=np.int64)
    for anchor in range(row_count):
        chosen = list(int(row) for row in teacher_rows[anchor])
        present = set(chosen)
        present.add(anchor)
        for row in student_rows[anchor]:
            candidate = int(row)
            if candidate not in present:
                chosen.append(candidate)
                present.add(candidate)
        while len(chosen) < CANDIDATE_WIDTH:
            candidate = int(rng.integers(0, row_count))
            if candidate not in present:
                chosen.append(candidate)
                present.add(candidate)
        output[anchor] = chosen[:CANDIDATE_WIDTH]
    return output


def _score(codes: torch.Tensor, labels: tuple[int, ...], device: torch.device) -> dict[str, float]:
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
        "packed_r1": float(packed["r1"]),
    }


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-snapshot", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--teacher-snapshot", type=Path, required=True)
    parser.add_argument("--teacher-sha256", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--driver-sha256", required=True)
    parser.add_argument("--execute-positive-coverage", action="store_true", required=True)
    args = parser.parse_args()
    outputs = (args.checkpoint, args.receipt)
    partials = tuple(path.with_name(f"{path.name}.partial") for path in outputs)
    if (
        not args.checkpoint.is_absolute()
        or not args.receipt.is_absolute()
        or len(set((*outputs, *partials))) != 4
        or any(not path.parent.is_dir() for path in outputs)
        or any(path.exists() for path in (*outputs, *partials))
    ):
        raise ValueError("replay artifact path authority differs")
    source_identity = positive_coverage_source_identity(
        driver=Path(__file__).resolve(),
        driver_sha256=args.driver_sha256,
        source_revision=args.source_revision,
    )

    configure_deterministic_similarity_runtime(SEED, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    started = time.monotonic()
    pair = load_paired_train_archives(
        args.source_snapshot,
        args.source_sha256,
        args.teacher_snapshot,
        args.teacher_sha256,
    )
    teacher = _unit(pair["teacher_train"])
    partition = deterministic_class_partition(pair["train_labels"], fit_fraction=0.8, seed=SEED)
    fit_indexes = list(partition.fit_row_indexes)
    validation_indexes = list(partition.validation_row_indexes)
    fit = teacher[fit_indexes].contiguous()
    validation = teacher[validation_indexes].contiguous()
    validation_labels = tuple(pair["train_labels"][row] for row in validation_indexes)

    pca = fit_centered_pca(fit, dimensions=DIMENSIONS)
    components = pca.components
    mean = pca.mean
    head = nn.Linear(fit.shape[1], DIMENSIONS, bias=True, device=device, dtype=torch.float32)
    with torch.no_grad():
        head.weight.copy_(components.to(device))
        head.bias.copy_((-(components.double() @ mean.double())).float().to(device))
    fit_cuda = fit.to(device)
    teacher_neighbors = _nearest_rows(fit_cuda, count=TEACHER_NEIGHBORS)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    rng = np.random.Generator(np.random.PCG64(SEED))
    trajectory: list[dict[str, object]] = []

    def evaluate(epoch: int, loss: float | None = None) -> None:
        with torch.inference_mode():
            codes = _unit(head(validation.to(device)).cpu())
        row: dict[str, object] = {"epoch": epoch, **_score(codes, validation_labels, device)}
        if loss is not None:
            row["mean_training_loss"] = loss
        trajectory.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    evaluate(0)
    for epoch in range(EPOCHS):
        with torch.inference_mode():
            current = _unit(head(fit_cuda))
            student_neighbors = _nearest_rows(current, count=STUDENT_NEIGHBORS)
        candidates = _candidate_rows(teacher_neighbors, student_neighbors, rng=rng)
        order = rng.permutation(len(fit_indexes))
        total_loss = 0.0
        total_rows = 0
        head.train()
        for start in range(0, len(order), BATCH_SIZE):
            anchors_np = order[start : start + BATCH_SIZE]
            anchors = torch.from_numpy(anchors_np).to(device)
            candidate = torch.from_numpy(candidates[anchors_np]).to(device)
            anchor_codes = _unit(head(fit_cuda[anchors]))
            candidate_codes = _unit(
                head(fit_cuda[candidate.reshape(-1)]).reshape(len(anchors), CANDIDATE_WIDTH, -1)
            )
            student_similarities = torch.einsum(
                "bd,bkd->bk", anchor_codes, candidate_codes
            ).contiguous()
            teacher_similarities = (
                torch.einsum(
                    "bd,bkd->bk",
                    fit_cuda[anchors],
                    fit_cuda[candidate.reshape(-1)].reshape(len(anchors), CANDIDATE_WIDTH, -1),
                )
                .detach()
                .contiguous()
            )
            loss = retrieval_local_rank_distillation_loss(
                student_similarities,
                teacher_similarities,
                teacher_neighbor_count=TEACHER_NEIGHBORS,
                margin_cap=0.05,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach()) * len(anchors_np)
            total_rows += len(anchors_np)
        scheduler.step()
        evaluate(epoch + 1, total_loss / total_rows)

    final = trajectory[-1]
    delta_map = abs(float(final["packed_map_at_r"]) - EXPECTED_MAP)
    delta_r1 = abs(float(final["packed_r1"]) - EXPECTED_R1)
    head_sha256 = _parameter_sha256(head.weight, head.bias)
    matched = (
        delta_map <= REPLAY_TOLERANCE
        and delta_r1 <= REPLAY_TOLERANCE
        and head_sha256 == EXPECTED_HEAD_SHA256
    )
    receipt = {
        "claim_eligible": False,
        "dataset": "sop-official-train-class-disjoint-validation",
        "elapsed_seconds": time.monotonic() - started,
        "expected_packed_map_at_r": EXPECTED_MAP,
        "expected_packed_r1": EXPECTED_R1,
        "expected_head_sha256": EXPECTED_HEAD_SHA256,
        "final_head_sha256": head_sha256,
        "matched": matched,
        "official_test_touched": False,
        "replay_tolerance": REPLAY_TOLERANCE,
        "schema": "sfora-retrieval-local-rank-replay-v2",
        "source": source_identity,
        "inputs": {
            "source_snapshot_sha256": args.source_sha256,
            "teacher_snapshot_sha256": args.teacher_sha256,
        },
        "trajectory": trajectory,
    }
    if not matched:
        write_canonical_receipt_no_clobber(receipt, args.receipt)
        raise RuntimeError(f"replay mismatch: map={delta_map:.9g}, r1={delta_r1:.9g}")
    write_linear_replay_artifacts(
        state={
            "weight": head.weight.detach().cpu().contiguous(),
            "bias": head.bias.detach().cpu().contiguous(),
        },
        receipt=receipt,
        checkpoint=args.checkpoint,
        complete_receipt=args.receipt,
    )


if __name__ == "__main__":
    main()
