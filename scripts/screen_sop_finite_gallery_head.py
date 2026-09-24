#!/usr/bin/env python3
"""Train-only matched head screen of finite-gallery recall surrogates."""

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
from torch import nn
from torch.nn import functional as F

from sfora.gallery_risk import finite_gallery_log_success, finite_gallery_success
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition, fit_centered_pca
from sfora.sop_evaluation import score_symmetric
from sfora.unicom_rank_finish import identity_balanced_batches

ARCHIVE_SHA256 = "16b4554d3868363905f1e1cd385783a8033513835723a7b89f4b762d893d757f"
SEED = 179019
STEPS = 256
BATCH_SIZE = 64
SAMPLE_FRACTION = 5_000 / 53_700
TEMPERATURE = 0.03
HEAD_LR = 1e-4
PROXY_LR = 1e-4
ARCFACE_MARGIN = 0.3
ARCFACE_SCALE = 64.0
ARMS = ("pca", "arcface", "inbatch_probability", "bank_log", "bank_probability")


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def packed_unit_ste(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Use the deployed int8 and f16 wire in forward, straight-through backward."""

    unit = F.normalize(value.float(), dim=1)
    scaled = 127.0 * unit
    rounded = scaled.round().clamp(-127, 127)
    codes = scaled + (rounded - scaled).detach()
    inverse = torch.linalg.vector_norm(codes, dim=1).reciprocal()
    f16 = inverse.half().float()
    return codes, inverse + (f16 - inverse).detach()


def packed_scores(
    query: tuple[torch.Tensor, torch.Tensor], gallery: tuple[torch.Tensor, torch.Tensor]
) -> torch.Tensor:
    qcodes, qinv = query
    gcodes, ginv = gallery
    return (qcodes @ gcodes.T) * qinv[:, None] * ginv[None, :]


def gallery_loss(
    head: nn.Linear,
    fit: torch.Tensor,
    labels: torch.Tensor,
    batch: torch.Tensor,
    *,
    arm: str,
) -> torch.Tensor:
    batch_labels = labels[batch]
    live = packed_unit_ste(head(fit[batch]))
    local = packed_scores(live, live)
    positive = batch_labels[:, None] == batch_labels[None, :]
    positive.fill_diagonal_(False)
    if not bool(positive.any(dim=1).all()):
        raise ValueError("SOP scheduled batch lacks positives")
    strongest = local.masked_fill(~positive, -torch.inf).max(dim=1).values
    if arm == "inbatch_probability":
        scores = local
        valid = batch_labels[:, None] != batch_labels[None, :]
        pool_size = len(batch) - 1
        sample_size = max(1, round(pool_size * SAMPLE_FRACTION))
    else:
        with torch.no_grad():
            bank = packed_unit_ste(head(fit))
        scores = packed_scores(live, bank)
        valid = batch_labels[:, None] != labels[None, :]
        pool_size = len(fit)
        sample_size = 5_000
    count = torch.sigmoid((scores - strongest[:, None]) / TEMPERATURE)
    count = (count * valid).sum(dim=1)
    if arm == "bank_log":
        # This arm tests the original objective, after the prior-art critique.
        return -finite_gallery_log_success(
            count, pool_size=pool_size, sample_size=sample_size
        ).mean()
    return (
        1.0 - finite_gallery_success(count, pool_size=pool_size, sample_size=sample_size)
    ).mean()


def arcface_loss(
    head: nn.Linear,
    proxy: nn.Parameter,
    fit: torch.Tensor,
    class_ids: torch.Tensor,
    batch: torch.Tensor,
) -> torch.Tensor:
    embedding = F.normalize(head(fit[batch]), dim=1)
    cosine = embedding @ F.normalize(proxy, dim=1).T
    target = class_ids[batch]
    row = torch.arange(len(batch), device=batch.device)
    selected = cosine[row, target].clamp(-1 + 1e-6, 1 - 1e-6)
    sine = (1 - selected.square()).sqrt()
    margin_cosine = selected * np.cos(ARCFACE_MARGIN) - sine * np.sin(ARCFACE_MARGIN)
    logits = cosine.scatter(1, target[:, None], margin_cosine[:, None]) * ARCFACE_SCALE
    return F.cross_entropy(logits, target)


@torch.inference_mode()
def packed_gallery(head: nn.Linear, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    packed_codes: list[torch.Tensor] = []
    packed_inverse: list[torch.Tensor] = []
    for block in rows.split(1024):
        normalized = F.normalize(head(block), dim=1).cpu().contiguous()
        packed = pack_int8_unit_embeddings(normalized)
        packed_codes.append(packed.codes.float())
        packed_inverse.append(packed.inverse_norms)
    return torch.cat(packed_codes), torch.cat(packed_inverse)


@torch.inference_mode()
def score_heldout_against_all(
    codes: torch.Tensor,
    inverse: torch.Tensor,
    labels: torch.Tensor,
    query_rows: torch.Tensor,
    fit_rows: torch.Tensor,
    *,
    device: torch.device,
    sample_size: int = 5_000,
) -> dict[str, object]:
    """Score heldout queries against all train rows, excluding each query itself."""

    if query_rows.ndim != 1 or fit_rows.ndim != 1:
        raise ValueError("SOP scoring partition differs")
    code_gpu = codes.to(device)
    inv_gpu = inverse.to(device).float()
    label_gpu = labels.to(device)
    query_gpu = query_rows.to(device)
    fit_gpu = fit_rows.to(device)
    relevant = torch.bincount(label_gpu[query_gpu])[label_gpu[query_gpu]] - 1
    width = int(relevant.max())
    ranks = torch.arange(1, width + 1, device=device)
    per_r1: list[float] = []
    per_ap: list[float] = []
    outranking: list[int] = []
    for block in query_gpu.split(64):
        scores = (code_gpu[block] @ code_gpu.T) * inv_gpu[block, None] * inv_gpu[None, :]
        scores[torch.arange(len(block), device=device), block] = -torch.inf
        qlabels = label_gpu[block]
        positive = qlabels[:, None] == label_gpu[None, :]
        positive_scores = scores.masked_fill(~positive, -torch.inf)
        strongest = positive_scores.max(dim=1).values
        ordinals = torch.arange(len(labels), device=device)
        best_ordinal = (
            torch.where(positive_scores == strongest[:, None], ordinals[None, :], len(labels))
            .min(dim=1)
            .values
        )
        seen_scores = scores[:, fit_gpu]
        # This readout is the exact sampled-negative probability, not a loss.
        bad = (
            (seen_scores > strongest[:, None])
            | ((seen_scores == strongest[:, None]) & (fit_gpu[None, :] < best_ordinal[:, None]))
        ).sum(dim=1)
        outranking.extend(int(x) for x in bad.cpu().tolist())
        ranked = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :width]
        matches = label_gpu[ranked] == qlabels[:, None]
        rel = relevant[len(per_r1) : len(per_r1) + len(block)]
        precision = matches.cumsum(dim=1) / ranks[None, :]
        ap = (precision * matches * (ranks[None, :] <= rel[:, None])).sum(dim=1) / rel
        per_r1.extend(float(x) for x in matches[:, 0].cpu().tolist())
        per_ap.extend(float(x) for x in ap.cpu().tolist())
    from diagnose_sop_seen_gallery_effect import expected_recall_random_negatives

    expected = expected_recall_random_negatives(
        np.full(len(outranking), len(fit_rows), dtype=np.int64),
        np.asarray(outranking, dtype=np.int64),
        sample_size,
    )
    return {
        "recall_at_1": float(np.mean(per_r1)),
        "map_at_r": float(np.mean(per_ap)),
        "per_query_r1": per_r1,
        "per_query_ap": per_ap,
        "seen_outranking_counts": outranking,
        "expected_r1_sampled_fit_negatives_fixed_positives": float(expected.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-sop-finite-gallery-head", action="store_true", required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or args.output.is_symlink()
        or sha256(args.source_archive) != ARCHIVE_SHA256
    ):
        raise ValueError("SOP head screen source/output authority differs")
    if not torch.cuda.is_available():
        raise ValueError("SOP head screen requires DGX GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    torch.set_num_threads(16)
    device = torch.device("cuda")
    started = time.perf_counter()
    with np.load(args.source_archive, allow_pickle=False) as source:
        features = np.asarray(source["train_embeddings"], dtype=np.float32)
        labels_np = np.asarray(source["train_labels"], dtype=np.int64)
    if features.shape != (59_551, 768) or labels_np.shape != (59_551,):
        raise ValueError("SOP pretrained feature inventory differs")
    features = np.ascontiguousarray(features / np.linalg.norm(features, axis=1, keepdims=True))
    partition = deterministic_class_partition(
        tuple(map(int, labels_np)), fit_fraction=0.9, seed=SEED
    )
    fit_ids = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    held_ids = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    if (len(fit_ids), len(held_ids)) != (53_700, 5_851):
        raise ValueError("SOP train identity partition differs")
    pca = fit_centered_pca(torch.from_numpy(features[fit_ids].copy()), dimensions=128)
    head_weight = pca.components.to(device)
    head_bias = -(head_weight @ pca.mean.to(device))
    fit = torch.from_numpy(features[fit_ids].copy()).to(device)
    held = torch.from_numpy(features[held_ids].copy()).to(device)
    fit_labels_np = labels_np[fit_ids]
    fit_labels = torch.from_numpy(fit_labels_np.copy()).to(device)
    all_labels = torch.from_numpy(labels_np.copy())
    schedule = identity_balanced_batches(
        tuple(map(str, fit_labels_np)),
        batch_size=BATCH_SIZE,
        images_per_identity=4,
        seed=SEED,
        epoch=1,
        steps=STEPS,
    )
    schedule_array = np.asarray(schedule, dtype="<i4")
    class_ids = torch.unique(fit_labels, sorted=True, return_inverse=True)[1]
    results: dict[str, object] = {}
    for arm in ARMS:
        torch.manual_seed(SEED)
        head = nn.Linear(768, 128, bias=True, device=device)
        with torch.no_grad():
            head.weight.copy_(head_weight)
            head.bias.copy_(head_bias)
        proxy = (
            nn.Parameter(torch.randn(len(partition.fit_class_ids), 128, device=device))
            if arm == "arcface"
            else None
        )
        params = [{"params": head.parameters(), "lr": HEAD_LR}]
        if proxy is not None:
            params.append({"params": [proxy], "lr": PROXY_LR})
        optimizer = torch.optim.Adam(params)
        arm_started = time.perf_counter()
        last_loss = None
        if arm != "pca":
            for batch_tuple in schedule:
                batch = torch.tensor(batch_tuple, device=device, dtype=torch.int64)
                if proxy is not None:
                    loss = arcface_loss(head, proxy, fit, class_ids, batch)
                else:
                    loss = gallery_loss(head, fit, fit_labels, batch, arm=arm)
                if not bool(torch.isfinite(loss)):
                    raise ValueError(f"SOP {arm} loss became nonfinite")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                optimizer.step()
                last_loss = float(loss.detach())
        training_seconds = time.perf_counter() - arm_started
        held_codes, held_inverse = packed_gallery(head, held)
        fit_codes, fit_inverse = packed_gallery(head, fit)
        held_labels = torch.from_numpy(labels_np[held_ids].copy())
        held_score = score_symmetric(held_codes, held_labels, inverse_norms=held_inverse)
        all_codes = torch.empty((len(features), 128), dtype=torch.float32)
        all_inverse = torch.empty(len(features), dtype=torch.float16)
        all_codes[held_ids] = held_codes
        all_codes[fit_ids] = fit_codes
        all_inverse[held_ids] = held_inverse
        all_inverse[fit_ids] = fit_inverse
        full_score = score_heldout_against_all(
            all_codes,
            all_inverse,
            all_labels,
            torch.from_numpy(held_ids),
            torch.from_numpy(fit_ids),
            device=device,
        )
        results[arm] = {
            "holdout_only": {
                key: held_score[key]
                for key in ("recall_at_1", "map_at_r", "per_query_r1", "per_query_ap")
            },
            "all_train_gallery": full_score,
            "training_seconds": training_seconds,
            "last_loss": last_loss,
            "head_sha256": hashlib.sha256(
                head.weight.detach().cpu().contiguous().numpy().astype("<f4").tobytes()
            ).hexdigest(),
        }
        print(
            json.dumps(
                {
                    "arm": arm,
                    "training_seconds": training_seconds,
                    "holdout_r1": held_score["recall_at_1"],
                    "full_r1": full_score["recall_at_1"],
                }
            ),
            flush=True,
        )
    receipt = {
        "schema": "sfora-sop-finite-gallery-head-screen-v1",
        "claim_eligible": False,
        "source_archive_sha256": ARCHIVE_SHA256,
        "script_sha256": sha256(Path(__file__)),
        "gallery_risk_sha256": sha256(
            Path(__file__).resolve().parents[1] / "src/sfora/gallery_risk.py"
        ),
        "seed": SEED,
        "steps": STEPS,
        "schedule_sha256": hashlib.sha256(schedule_array.tobytes()).hexdigest(),
        "fit_rows": len(fit_ids),
        "holdout_rows": len(held_ids),
        "temperature": TEMPERATURE,
        "sample_fraction": SAMPLE_FRACTION,
        "head_learning_rate": HEAD_LR,
        "proxy_learning_rate": PROXY_LR,
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
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write((json.dumps(receipt, sort_keys=True, allow_nan=False) + "\n").encode())
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps({"output": str(args.output), "total_seconds": receipt["total_seconds"]}),
        flush=True,
    )


if __name__ == "__main__":
    main()
