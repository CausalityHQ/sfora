#!/usr/bin/env python3
"""One fit-only In-Shop screen for SmoothAP continuation in the deployed code geometry."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora import compact_metric, joint_relational_compaction, unicom_rank_finish, unicom_training
from sfora.atomic_publication import publish_bytes_noreplace
from sfora.compact_metric import (
    CompactMetricConfig,
    CompactMetricEncoder,
    fit_compact_metric_projection,
)
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_rank_finish import identity_balanced_batches, smooth_ap_finish_loss
from sfora.unicom_training import identity_holdout

ARCHIVE_SHA256 = "2045eecd26805e3cfa0ff36b25f102101f199595de847d59508af0b077bbf5ee"
PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
MODEL_SHA256 = "ad7e16d28daf32c3ae8d6258444e18c142cdf2e1816448a615be653e9545697b"
RUN_RECEIPT_SHA256 = "62276d5bdd9df2e29378ac7d59b0728cefdd82237b181e6db589874fbaec4667"
FINISH_RESULT_SHA256 = "60b14fc4342f32ec9b59fc8dc3424e054b32f2652b53dc43993199de1b446ca6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def publish_result(path: Path, payload: bytes) -> None:
    def validate(persisted: bytes) -> None:
        if persisted != payload:
            raise ValueError("fit-only result publication differs")

    published = publish_bytes_noreplace(path, payload, validator=validate)
    published.close()


def fake_quantized_unit(value: torch.Tensor) -> torch.Tensor:
    """Return the packed code's unit geometry with a straight-through gradient."""

    unit = F.normalize(value.float(), dim=1)
    rounded = torch.round(unit * 127.0).clamp(-127, 127) / 127.0
    straight_through = unit + (rounded - unit).detach()
    return F.normalize(straight_through, dim=1)


def packed_query_gallery_metrics(
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_labels: np.ndarray,
    gallery_labels: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, object]:
    if not set(query_labels.tolist()).issubset(set(gallery_labels.tolist())):
        raise ValueError("holdout query/gallery labels differ")
    query_wire = pack_int8_unit_embeddings(query)
    gallery_wire = pack_int8_unit_embeddings(gallery)
    q = query_wire.codes.float().to(device)
    g = gallery_wire.codes.float().to(device)
    qinv = query_wire.inverse_norms.float().to(device)
    ginv = gallery_wire.inverse_norms.float().to(device)
    scores = (q @ g.T) * qinv[:, None] * ginv[None, :]
    order = torch.argsort(scores, dim=1, descending=True, stable=True).cpu().numpy()
    counts = Counter(gallery_labels.tolist())
    aps: list[float] = []
    hits: list[float] = []
    for label, ranking in zip(query_labels.tolist(), order, strict=True):
        positives = counts[label]
        relevant = gallery_labels[ranking[:positives]] == label
        precision_terms = np.cumsum(relevant) * relevant / np.arange(1, positives + 1)
        aps.append(float(np.sum(precision_terms) / positives))
        hits.append(float(gallery_labels[ranking[0]] == label))
    return {
        "map_at_r": float(math.fsum(aps) / len(aps)),
        "recall_at_1": float(math.fsum(hits) / len(hits)),
        "per_query_ap": aps,
        "per_query_r1": hits,
        "code_sha256": hashlib.sha256(gallery_wire.codes.numpy().tobytes()).hexdigest(),
        "inverse_norm_sha256": hashlib.sha256(
            gallery_wire.inverse_norms.numpy().tobytes()
        ).hexdigest(),
    }


def score_encoder(
    encoder: CompactMetricEncoder,
    features: torch.Tensor,
    query_indexes: np.ndarray,
    gallery_indexes: np.ndarray,
    labels: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, object]:
    query = encoder.transform(features[query_indexes].contiguous())
    gallery = encoder.transform(features[gallery_indexes].contiguous())
    return packed_query_gallery_metrics(
        query, gallery, labels[query_indexes], labels[gallery_indexes], device=device
    )


def continue_head(
    initial: CompactMetricEncoder,
    bank: torch.Tensor,
    schedule: tuple[tuple[tuple[int, ...], ...], ...],
    schedule_labels: tuple[str, ...],
    eligible_positions: tuple[int, ...],
    *,
    quantized: bool,
    device: torch.device,
) -> tuple[CompactMetricEncoder, float]:
    module = torch.nn.Linear(768, 128, device=device, dtype=torch.float32)
    with torch.no_grad():
        module.weight.copy_(initial.weight.to(device))
        module.bias.copy_(initial.bias.to(device))
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4, weight_decay=0.0)
    losses: list[float] = []
    for epoch, batches in enumerate(schedule, start=1):
        for indexes in batches:
            rows = torch.tensor([eligible_positions[i] for i in indexes], device=device)
            projected = module(bank[rows])
            coded = fake_quantized_unit(projected) if quantized else F.normalize(projected, dim=1)
            padded = F.pad(coded, (0, 384))
            batch_labels = tuple(schedule_labels[i] for i in indexes)
            loss = smooth_ap_finish_loss(padded, batch_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(module.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach()))
        print(
            json.dumps({"arm": "coded" if quantized else "float", "epoch": epoch,
                        "last_loss": losses[-1]}), flush=True,
        )
    encoder = CompactMetricEncoder(
        weight=module.weight.detach().cpu().contiguous(),
        bias=module.bias.detach().cpu().contiguous(),
    )
    return encoder, float(math.fsum(losses) / len(losses))


def run(args: argparse.Namespace) -> dict[str, object]:
    started = time.perf_counter()
    if len(args.source_commit) != 40 or any(
        c not in "0123456789abcdef" for c in args.source_commit
    ):
        raise ValueError("source commit differs")
    if sha256(args.archive) != ARCHIVE_SHA256:
        raise ValueError("rank-finished archive differs")
    if sha256(args.dataset_root / "Eval" / "list_eval_partition.txt") != PARTITION_SHA256:
        raise ValueError("In-Shop partition differs")
    if sha256(args.run_receipt) != RUN_RECEIPT_SHA256:
        raise ValueError("rank-finish parent run receipt differs")
    if sha256(args.finish_result) != FINISH_RESULT_SHA256:
        raise ValueError("rank-finish result differs")
    receipt = json.loads(args.run_receipt.read_text())
    finish_result = json.loads(args.finish_result.read_text())
    if (
        receipt["holdout_fraction"] != 0.2
        or receipt["holdout_seed"] != 0
        or finish_result["model_artifact"]["sha256"] != MODEL_SHA256
        or finish_result["run_receipt"]["sha256"] != RUN_RECEIPT_SHA256
        or finish_result["finish_seed"] != 1
    ):
        raise ValueError("rank-finish training lineage differs")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda")
    with np.load(args.archive, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata_json"]))
        if metadata["checkpoint_sha256"] != MODEL_SHA256:
            raise ValueError("rank-finished model differs")
        # Deliberately do not read query_embeddings or gallery_embeddings.
        features_np = np.ascontiguousarray(archive["train_embeddings"])
        labels = np.asarray(archive["train_labels"])
    records = tuple(
        row for row in parse_inshop_partition(args.dataset_root) if row.split == "train"
    )
    if tuple(labels.tolist()) != tuple(row.label for row in records):
        raise ValueError("feature row order differs from official train partition")
    optimization, query, gallery, _ = identity_holdout(
        records, fraction=receipt["holdout_fraction"], seed=receipt["holdout_seed"]
    )
    index = {row.image_path: i for i, row in enumerate(records)}
    fit_indexes = np.asarray([index[row.image_path] for row in optimization], dtype=np.int64)
    query_indexes = np.asarray([index[row.image_path] for row in query], dtype=np.int64)
    gallery_indexes = np.asarray([index[row.image_path] for row in gallery], dtype=np.int64)
    assert not (set(labels[fit_indexes].tolist()) & set(labels[query_indexes].tolist()))
    assert set(labels[query_indexes].tolist()) == set(labels[gallery_indexes].tolist())
    features = torch.from_numpy(features_np)
    classes = {label: n for n, label in enumerate(sorted(set(labels[fit_indexes].tolist())))}
    fit_labels = torch.tensor([classes[label] for label in labels[fit_indexes]], dtype=torch.int64)
    torch.cuda.reset_peak_memory_stats()
    base_fit = fit_compact_metric_projection(
        features[fit_indexes].contiguous(), fit_labels, device=device
    )
    baseline = score_encoder(
        base_fit.encoder, features, query_indexes, gallery_indexes, labels, device=device
    )
    bank = F.normalize(features[fit_indexes].to(device), dim=1)
    fit_label_strings = tuple(labels[fit_indexes].tolist())
    fit_class_counts = Counter(fit_label_strings)
    eligible_positions = tuple(
        i for i, label in enumerate(fit_label_strings) if fit_class_counts[label] >= 2
    )
    schedule_labels = tuple(fit_label_strings[i] for i in eligible_positions)
    steps = math.ceil(len(eligible_positions) / 128)
    schedule = tuple(
        identity_balanced_batches(
            schedule_labels, batch_size=128, images_per_identity=4,
            seed=0, epoch=epoch, steps=steps,
        )
        for epoch in (1, 2)
    )
    schedule_sha256 = hashlib.sha256(
        json.dumps(schedule, separators=(",", ":")).encode()
    ).hexdigest()
    float_encoder, float_loss = continue_head(
        base_fit.encoder, bank, schedule, schedule_labels, eligible_positions,
        quantized=False, device=device,
    )
    candidate_encoder, coded_loss = continue_head(
        base_fit.encoder, bank, schedule, schedule_labels, eligible_positions,
        quantized=True, device=device,
    )
    float_control = score_encoder(
        float_encoder, features, query_indexes, gallery_indexes, labels, device=device
    )
    candidate = score_encoder(
        candidate_encoder, features, query_indexes, gallery_indexes, labels, device=device
    )
    ap_delta = np.asarray(candidate["per_query_ap"]) - np.asarray(baseline["per_query_ap"])
    r1_delta = np.asarray(candidate["per_query_r1"]) - np.asarray(baseline["per_query_r1"])
    coded_minus_float = float(
        np.mean(np.asarray(candidate["per_query_ap"]) - np.asarray(float_control["per_query_ap"]))
    )
    float_minus_baseline = float(
        np.mean(np.asarray(float_control["per_query_ap"]) - np.asarray(baseline["per_query_ap"]))
    )
    float_r1_delta = np.asarray(float_control["per_query_r1"]) - np.asarray(
        baseline["per_query_r1"]
    )
    generator = np.random.Generator(np.random.PCG64(17))
    draws = generator.integers(0, len(ap_delta), size=(10_000, len(ap_delta)))
    interval = np.quantile(ap_delta[draws].mean(axis=1), (0.025, 0.975)).tolist()
    delta_map = float(np.mean(ap_delta))
    delta_r1 = float(np.mean(r1_delta))
    gained_r1 = int(np.count_nonzero(r1_delta > 0))
    lost_r1 = int(np.count_nonzero(r1_delta < 0))
    advance = delta_map >= 0.003 and gained_r1 >= lost_r1
    code_attribution = coded_minus_float >= 0.001
    float_advance = float_minus_baseline >= 0.003 and float(np.sum(float_r1_delta)) >= 0.0
    return {
        "schema": "sfora-deployed-code-rank-finish-f0-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": sha256(Path(__file__)),
        "library_sha256": {
            module.__name__: sha256(Path(module.__file__))
            for module in (
                compact_metric, joint_relational_compaction, unicom_rank_finish,
                unicom_training,
            )
        },
        "archive_sha256": ARCHIVE_SHA256,
        "partition_sha256": PARTITION_SHA256,
        "run_receipt_sha256": RUN_RECEIPT_SHA256,
        "finish_result_sha256": FINISH_RESULT_SHA256,
        "holdout": {"fraction": receipt["holdout_fraction"],
                    "seed": receipt["holdout_seed"],
                    "query_paths_sha256": hashlib.sha256(
                        "\n".join(str(row.image_path) for row in query).encode()
                    ).hexdigest()},
        "model_sha256": MODEL_SHA256,
        "rows": {"fit": len(fit_indexes), "holdout_queries": len(query_indexes),
                 "holdout_gallery": len(gallery_indexes)},
        "baseline_encoder_sha256": base_fit.encoder.sha256,
        "baseline_fit": {"eligible_classes": base_fit.eligible_class_count,
                         "total_updates": base_fit.total_updates,
                         "schedule_sha256": base_fit.schedule_sha256,
                         "excluded_singleton_rows": len(fit_indexes) - len(eligible_positions),
                         "config": dataclasses.asdict(CompactMetricConfig())},
        "float_encoder_sha256": float_encoder.sha256,
        "candidate_encoder_sha256": candidate_encoder.sha256,
        "baseline": baseline,
        "float_control": float_control,
        "candidate": candidate,
        "delta": {"map_at_r": delta_map, "recall_at_1": delta_r1,
                  "map_at_r_ci95": interval, "coded_minus_float_map_at_r": coded_minus_float,
                  "float_minus_baseline_map_at_r": float_minus_baseline,
                  "float_minus_baseline_net_r1_flips": int(np.sum(float_r1_delta)),
                  "r1_gained_queries": gained_r1, "r1_lost_queries": lost_r1},
        "gate": {"minimum_map_at_r": 0.003, "minimum_net_r1_flips": 0,
                 "minimum_coded_minus_float_map_at_r": 0.001,
                 "advance_code_geometry": advance and code_attribution,
                 "advance_float_rank_only": float_advance and not (advance and code_attribution),
                 "head_continuation_closed": not advance and not float_advance},
        "training": {"epochs": 2, "steps_per_epoch": steps,
                     "excluded_singleton_rows": len(fit_indexes) - len(eligible_positions),
                     "schedule_sha256": schedule_sha256,
                     "optimizer": "AdamW", "learning_rate": 1e-4,
                     "weight_decay": 0.0, "gradient_clip": 1.0,
                     "batch_size": 128, "images_per_identity": 4,
                     "float_loss_mean": float_loss, "coded_loss_mean": coded_loss},
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.perf_counter() - started,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--run-receipt", required=True, type=Path)
    parser.add_argument("--finish-result", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = run(args)
    payload = (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()
    publish_result(args.output, payload)
    print(json.dumps({"output": str(args.output), "gate": result["gate"],
                      "delta": result["delta"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
