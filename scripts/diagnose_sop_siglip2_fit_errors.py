#!/usr/bin/env python3
"""Fit-only diagnosis of deployed SOP packed-retrieval top-1 errors."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition
from sfora.sop_compact_training import compact_head_features


def classify_scores(
    scores: torch.Tensor, labels: torch.Tensor, query_ordinals: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return top row, nearest same-label mate, and mate rank after self-exclusion."""

    if (
        scores.ndim != 2
        or labels.ndim != 1
        or scores.shape[1] != labels.shape[0]
        or query_ordinals.shape != (scores.shape[0],)
    ):
        raise ValueError("fit-only packed score geometry differs")
    score = scores.clone()
    row_ordinals = torch.arange(score.shape[0], device=score.device)
    score[row_ordinals, query_ordinals] = -torch.inf
    same = labels[None, :] == labels[query_ordinals, None]
    mate_scores = score.masked_fill(~same, -torch.inf)
    best_mate = mate_scores.argmax(dim=1)
    best_score = mate_scores[row_ordinals, best_mate]
    if not bool(torch.isfinite(best_score).all()):
        raise ValueError("fit-only product has no other gallery image")
    ordinals = torch.arange(score.shape[1], device=score.device)[None, :]
    rank = (
        1
        + (score > best_score[:, None]).sum(dim=1)
        + ((score == best_score[:, None]) & (ordinals < best_mate[:, None])).sum(dim=1)
    )
    return score.argmax(dim=1), best_mate, rank


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def packed_scores(
    queries: torch.Tensor,
    gallery_transposed: torch.Tensor,
    query_inverse_norms: torch.Tensor,
    gallery_inverse_norms: torch.Tensor,
) -> torch.Tensor:
    """Match the native int32 dot, then sequential fp32 norm multiplications."""

    return (queries @ gallery_transposed) * query_inverse_norms[:, None] * gallery_inverse_norms


@torch.inference_mode()
def diagnose(
    values: np.ndarray,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    source_features: np.ndarray,
    classifier: torch.Tensor,
    native_library: Path,
) -> dict[str, object]:
    device = torch.device("cuda:0")
    packed = pack_int8_unit_embeddings(torch.from_numpy(np.asarray(values, dtype=np.float32)))
    codes = packed.codes.to(device=device, dtype=torch.float32)
    inverse_norms = packed.inverse_norms.to(device=device, dtype=torch.float32)
    gallery_transposed = codes.T.contiguous()
    product_labels = torch.from_numpy(labels.copy()).to(device)
    names, class_indexes = torch.unique(product_labels, sorted=True, return_inverse=True)
    if names.numel() != classifier.shape[0]:
        raise ValueError("fit-only product proxy inventory differs")
    unit_values = F.normalize(torch.from_numpy(np.asarray(values).copy()).to(device), dim=1)
    classifier = F.normalize(classifier.to(device), dim=1)
    frozen = F.normalize(torch.from_numpy(np.asarray(source_features).copy()).to(device), dim=1)
    product_sums = torch.zeros((names.numel(), values.shape[1]), device=device)
    product_sums.index_add_(0, class_indexes, unit_values)
    product_means = F.normalize(product_sums, dim=1)

    selected = np.linspace(0, len(labels) - 1, 32, dtype=np.int64)
    selected_packed = PackedInt8Embeddings(
        packed.codes[selected].contiguous(), packed.inverse_norms[selected].contiguous()
    )
    with CutilePackedInt8Gallery.open_packed(native_library, packed) as native:
        native_rows, native_scores = native.search_packed(selected_packed)
    selected_scores = packed_scores(
        codes[selected], gallery_transposed, inverse_norms[selected], inverse_norms
    )
    sorted_rows = torch.argsort(selected_scores, dim=1, descending=True, stable=True)[:, :10]
    exact_rows = np.array_equal(sorted_rows.cpu().numpy(), native_rows)
    exact_scores = np.array_equal(
        selected_scores.gather(1, sorted_rows).cpu().numpy().view(np.uint32),
        native_scores.view(np.uint32),
    )
    if not exact_rows or not exact_scores:
        raise ValueError("fit-only packed score differs from exact native top-10")

    errors: list[dict[str, object]] = []
    correct = 0
    for start in range(0, len(labels), 64):
        stop = min(start + 64, len(labels))
        query_ordinals = torch.arange(start, stop, device=device)
        scores = packed_scores(
            codes[start:stop], gallery_transposed, inverse_norms[start:stop], inverse_norms
        )
        top, mate, rank = classify_scores(scores, product_labels, query_ordinals)
        wrong = product_labels[top] != product_labels[query_ordinals]
        correct += int((~wrong).sum().item())
        if not bool(wrong.any()):
            continue
        error_query = query_ordinals[wrong]
        error_top = top[wrong]
        error_mate = mate[wrong]
        error_rank = rank[wrong]
        proxy_scores = unit_values[error_query] @ classifier.T
        proxy_correct = proxy_scores.argmax(dim=1) == class_indexes[error_query]
        frozen_cosine = (frozen[error_query] * frozen[error_top]).sum(dim=1)
        mate_direction = F.normalize(unit_values[error_mate] - unit_values[error_query], dim=1)
        mean_direction = F.normalize(
            product_means[class_indexes[error_query]] - unit_values[error_query], dim=1
        )
        alignment = (mate_direction * mean_direction).sum(dim=1)
        for position in range(len(error_query)):
            query = int(error_query[position])
            impostor = int(error_top[position])
            same_product_mate = int(error_mate[position])
            errors.append(
                {
                    "query_fit_ordinal": query,
                    "query_train_row": int(fit_rows[query]),
                    "impostor_fit_ordinal": impostor,
                    "impostor_train_row": int(fit_rows[impostor]),
                    "mate_fit_ordinal": same_product_mate,
                    "mate_train_row": int(fit_rows[same_product_mate]),
                    "mate_rank": int(error_rank[position]),
                    "proxy_correct": bool(proxy_correct[position]),
                    "frozen_impostor_cosine": float(frozen_cosine[position]),
                    "duplicate_impostor": bool(frozen_cosine[position] >= 0.97),
                    "mate_mean_direction_cosine": float(alignment[position]),
                    "direction_misaligned": bool(alignment[position] < 0.0),
                }
            )
    error_count = len(errors)
    duplicate = sum(bool(error["duplicate_impostor"]) for error in errors)
    proxy_correct_count = sum(bool(error["proxy_correct"]) for error in errors)
    near_miss = sum(2 <= int(error["mate_rank"]) <= 10 for error in errors)
    orphan = sum(int(error["mate_rank"]) > 10 for error in errors)
    misaligned_orphan = sum(
        int(error["mate_rank"]) > 10 and bool(error["direction_misaligned"]) for error in errors
    )
    return {
        "fit_queries": len(labels),
        "fit_correct": correct,
        "fit_errors": error_count,
        "fit_recall_at_1": correct / len(labels),
        "native_selected_top10_exact": exact_rows and exact_scores,
        "duplicate_impostor_count": duplicate,
        "proxy_correct_member_wrong_count": proxy_correct_count,
        "near_miss_count": near_miss,
        "orphan_count": orphan,
        "misaligned_orphan_count": misaligned_orphan,
        "duplicate_impostor_share_of_errors": duplicate / error_count if error_count else None,
        "proxy_correct_member_wrong_share_of_errors": (
            proxy_correct_count / error_count if error_count else None
        ),
        "near_miss_plus_misaligned_orphan_share_of_errors": (
            (near_miss + misaligned_orphan) / error_count if error_count else None
        ),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in (
        "source-archive",
        "source-features",
        "training-receipt",
        "training-checkpoint",
        "train-embeddings",
        "native-library",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-training-receipt-sha256", required=True)
    args = parser.parse_args()
    started = time.perf_counter()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or not torch.cuda.is_available()
        or sha256(args.training_receipt) != args.expected_training_receipt_sha256
    ):
        raise ValueError("fit-only diagnosis authority differs")
    receipt = json.loads(args.training_receipt.read_text())
    if (
        any(
            sha256(path) != receipt[key]
            for path, key in (
                (args.source_archive, "source_archive_sha256"),
                (args.source_features, "source_features_sha256"),
                (args.training_checkpoint, "checkpoint_sha256"),
                (args.train_embeddings, "train_embeddings_sha256"),
                (args.native_library, "native_library_sha256"),
            )
        )
        or receipt.get("arm") != "arcface"
    ):
        raise ValueError("fit-only diagnosis artifacts differ")
    with np.load(args.source_archive, allow_pickle=False) as archive:
        all_labels = np.asarray(archive["train_labels"], dtype=np.int64)
    partition = deterministic_class_partition(
        tuple(map(int, all_labels)), fit_fraction=0.9, seed=179019
    )
    fit_rows = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    labels = all_labels[fit_rows]
    if len(fit_rows) != receipt.get("fit_images") or len(np.unique(labels)) != receipt.get(
        "fit_products"
    ):
        raise ValueError("fit-only diagnosis split differs")
    all_values = np.load(args.train_embeddings, mmap_mode="r", allow_pickle=False)
    all_source = np.load(args.source_features, mmap_mode="r", allow_pickle=False)
    if all_values.shape != (len(all_labels), 128) or all_source.shape != (len(all_labels), 1024):
        raise ValueError("fit-only diagnosis embedding geometry differs")
    checkpoint = torch.load(args.training_checkpoint, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("arm") != "arcface"
        or checkpoint.get("seed") != receipt.get("seed")
        or checkpoint.get("updates") != receipt.get("updates")
    ):
        raise ValueError("fit-only diagnosis checkpoint identity differs")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    trained = diagnose(
        all_values[fit_rows],
        labels,
        fit_rows,
        all_source[fit_rows],
        checkpoint["classifier"],
        args.native_library,
    )
    decision_basis = "trained_arcface"
    fallback = None
    if trained["fit_errors"] < 200:
        from train_sop_siglip2_compact import initialize_head_and_classifier

        fit_source = torch.from_numpy(np.asarray(all_source[fit_rows]).copy())
        head, initial_classifier, _ = initialize_head_and_classifier(
            fit_source, tuple(map(int, labels))
        )
        frozen_values = F.normalize(compact_head_features(fit_source, head), dim=1).numpy()
        fallback = diagnose(
            frozen_values,
            labels,
            fit_rows,
            all_source[fit_rows],
            initial_classifier.detach(),
            args.native_library,
        )
        decision_basis = "frozen_pca"
    basis = trained if fallback is None else fallback
    if basis["fit_errors"] < 200:
        decision = "insufficient_errors"
    elif basis["duplicate_impostor_share_of_errors"] >= 0.5:
        decision = "duplicate_impostors_dominate"
    elif basis["proxy_correct_member_wrong_share_of_errors"] >= 0.4:
        decision = "revise_competitor_field"
    elif basis["near_miss_plus_misaligned_orphan_share_of_errors"] >= 0.3:
        decision = "run_existential_mate_matched_arm"
    else:
        decision = "no_preregistered_mechanism_gate"
    result = {
        "schema": "sfora-sop-siglip2-fit-error-census-v1",
        "split": "SOP official TRAIN fit products only, seed 179019",
        "claim_eligible": False,
        "training_receipt_sha256": args.expected_training_receipt_sha256,
        "source_sha256": sha256(Path(__file__)),
        "source_archive_sha256": sha256(args.source_archive),
        "source_features_sha256": sha256(args.source_features),
        "checkpoint_sha256": sha256(args.training_checkpoint),
        "train_embeddings_sha256": sha256(args.train_embeddings),
        "native_library_sha256": sha256(args.native_library),
        "trained_arcface": trained,
        "frozen_pca_fallback": fallback,
        "decision_basis": decision_basis,
        "registered_decision": decision,
        "wall_seconds": time.perf_counter() - started,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "hardware": {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__},
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
                for key in (
                    "schema",
                    "decision_basis",
                    "registered_decision",
                    "wall_seconds",
                    "peak_cuda_allocated_bytes",
                )
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
