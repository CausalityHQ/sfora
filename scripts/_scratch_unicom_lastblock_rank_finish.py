#!/usr/bin/env python3
"""Fail-fast Cars196 UniCOM final-block Smooth-AP rank-finishing probe."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from sfora.compact_metric import fit_power_whitening_projection
from sfora.deterministic_similarity_runtime import (
    configure_deterministic_similarity_runtime,
)
from sfora.joint_relational_compaction import (
    PackedInt8Embeddings,
    pack_int8_unit_embeddings,
)
from sfora.unicom_rank_finish import identity_balanced_batches, smooth_ap_finish_loss

DATASET_ID = "tanganke/stanford_cars"
DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
CHECKPOINT_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
BASELINE_FEATURE_SHA256 = "6ddfd7e2c9fd489dff51fa33697c62abf92a45247b52b336c0371f5482231ab3"
BASELINE = {"map_at_r": 0.860662, "recall_at_1": 0.974542}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze_except_final_block(model: torch.nn.Module) -> dict[str, object]:
    """Expose only UniCOM transformer block 23 to optimization."""

    blocks = getattr(model, "blocks", None)
    if not isinstance(blocks, torch.nn.ModuleList) or len(blocks) != 24:
        raise ValueError("UniCOM transformer authority differs")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in blocks[23].parameters():
        parameter.requires_grad_(True)
    names = [name for name, value in model.named_parameters() if value.requires_grad]
    if not names or any(not name.startswith("blocks.23.") for name in names):
        raise ValueError("UniCOM final-block authority differs")
    return {
        "block_index": 23,
        "parameter_names": names,
        "trainable_parameters": sum(
            value.numel() for value in model.parameters() if value.requires_grad
        ),
        "total_parameters": sum(value.numel() for value in model.parameters()),
    }


def _metric(metrics: Mapping[str, object], name: str) -> float:
    value = metrics.get(name)
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("rank-finish metric authority differs")
    return value


def classify_candidate(
    baseline: Mapping[str, object],
    candidate: Mapping[str, object],
    interval: Mapping[str, object],
) -> dict[str, object]:
    """Apply the frozen joint Cars promotion gate."""

    map_delta = _metric(candidate, "map_at_r") - _metric(baseline, "map_at_r")
    recall_delta = _metric(candidate, "recall_at_1") - _metric(
        baseline, "recall_at_1"
    )
    lower = interval.get("lower")
    if type(lower) is not float or not math.isfinite(lower):
        raise ValueError("rank-finish interval authority differs")
    return {
        "map_at_r_delta": map_delta,
        "recall_at_1_delta": recall_delta,
        "minimum_map_at_r_delta": 0.005,
        "require_positive_interval_lower": True,
        "minimum_recall_at_1_delta": 0.0,
        "passed": map_delta >= 0.005 and lower > 0.0 and recall_delta >= 0.0,
    }


def proxy_anchor_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    proxies: torch.Tensor,
    *,
    alpha: float = 32.0,
    delta: float = 0.1,
) -> torch.Tensor:
    """Compute the canonical Proxy Anchor objective with learned class proxies."""

    if (
        embeddings.ndim != 2
        or proxies.ndim != 2
        or embeddings.shape[1] != proxies.shape[1]
        or labels.shape != (embeddings.shape[0],)
        or labels.dtype != torch.int64
        or int(labels.min()) < 0
        or int(labels.max()) >= proxies.shape[0]
        or type(alpha) is not float
        or alpha != 32.0
        or type(delta) is not float
        or delta != 0.1
        or not bool(torch.isfinite(embeddings).all())
        or not bool(torch.isfinite(proxies).all())
    ):
        raise ValueError("Proxy Anchor authority differs")
    normalized_embeddings = torch.nn.functional.normalize(embeddings.float(), dim=1)
    normalized_proxies = torch.nn.functional.normalize(proxies.float(), dim=1)
    similarities = normalized_embeddings @ normalized_proxies.T
    positive = torch.nn.functional.one_hot(
        labels, num_classes=proxies.shape[0]
    ).bool()
    negative = ~positive
    positive_terms = torch.exp(-alpha * (similarities - delta)) * positive
    negative_terms = torch.exp(alpha * (similarities + delta)) * negative
    present = positive.any(dim=0)
    loss = torch.log1p(positive_terms.sum(dim=0))[present].mean()
    loss = loss + torch.log1p(negative_terms.sum(dim=0)).mean()
    if not torch.isfinite(loss):
        raise ValueError("Proxy Anchor loss is nonfinite")
    return loss


def paired_class_bootstrap(
    candidate: np.ndarray, control: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    """Return a fixed 10k paired evaluation-class bootstrap interval."""

    if (
        candidate.ndim != 1
        or control.shape != candidate.shape
        or labels.shape != candidate.shape
        or len(candidate) == 0
    ):
        raise ValueError("paired bootstrap authority differs")
    classes = np.unique(labels)
    differences = np.asarray(
        [
            (candidate[labels == label] - control[labels == label]).mean()
            for label in classes
        ],
        dtype=np.float64,
    )
    generator = np.random.Generator(np.random.PCG64(17))
    values = np.empty(10_000, dtype=np.float64)
    for start in range(0, len(values), 100):
        stop = min(start + 100, len(values))
        indexes = generator.integers(0, len(classes), size=(stop - start, len(classes)))
        values[start:stop] = differences[indexes].mean(axis=1)
    lower, median, upper = np.quantile(values, (0.025, 0.5, 0.975))
    return {"lower": float(lower), "median": float(median), "upper": float(upper)}


def _score(
    packed: PackedInt8Embeddings, labels: torch.Tensor, *, device: torch.device
) -> dict[str, object]:
    label_values = tuple(int(value) for value in labels.tolist())
    counts = Counter(label_values)
    positive_counts = [counts[label] - 1 for label in label_values]
    if min(positive_counts) < 1:
        raise ValueError("retrieval positive authority differs")
    width = max(positive_counts)
    codes = packed.codes.to(device=device, dtype=torch.float32)
    inverse_norms = packed.inverse_norms.to(device=device, dtype=torch.float32)
    rankings = []
    with torch.inference_mode():
        for start in range(0, len(codes), 256):
            stop = min(start + 256, len(codes))
            scores = (
                (codes[start:stop] @ codes.T)
                * inverse_norms[start:stop].unsqueeze(1)
                * inverse_norms.unsqueeze(0)
            )
            scores[
                torch.arange(stop - start, device=device),
                torch.arange(start, stop, device=device),
            ] = -torch.inf
            ordinal = torch.arange(len(codes), device=device).expand(stop - start, -1)
            ordinal_order = torch.argsort(ordinal, dim=1, stable=True)
            score_order = torch.argsort(
                scores.gather(1, ordinal_order), dim=1, descending=True, stable=True
            )
            rankings.append(ordinal_order.gather(1, score_order)[:, :width].cpu())
    aps: list[float] = []
    hits: list[float] = []
    for ranking, label, positives in zip(
        torch.cat(rankings).tolist(), label_values, positive_counts, strict=True
    ):
        found = 0
        terms = []
        for rank, index in enumerate(ranking[:positives], start=1):
            if label_values[index] == label:
                found += 1
                terms.append(found / rank)
        aps.append(math.fsum(terms) / positives)
        hits.append(float(label_values[ranking[0]] == label))
    return {
        "map_at_r": math.fsum(aps) / len(aps),
        "recall_at_1": math.fsum(hits) / len(hits),
        "per_query_ap": np.asarray(aps, dtype=np.float64),
    }


def _fit_score(
    fit: torch.Tensor,
    fit_labels: torch.Tensor,
    evaluation: torch.Tensor,
    evaluation_labels: torch.Tensor,
    *,
    device: torch.device,
) -> dict[str, object]:
    fitted = fit_power_whitening_projection(
        fit,
        fit_labels,
        alpha=0.75,
        regularization=1.0,
        output_dimensions=128,
    )
    packed = pack_int8_unit_embeddings(fitted.encoder.transform(evaluation))
    result = _score(packed, evaluation_labels, device=device)
    result["encoder_sha256"] = fitted.encoder.sha256
    return result


class _ImageDataset(torch.utils.data.Dataset):
    def __init__(self, source: object, indices: Sequence[int], transform: object) -> None:
        self.source = source
        self.indices = tuple(indices)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.source[self.indices[index]]  # type: ignore[index]
        image = row["image"].convert("RGB")
        return self.transform(image), int(row["label"])  # type: ignore[operator]


def _encode(
    model: torch.nn.Module,
    source: object,
    indices: Sequence[int],
    transform: object,
    *,
    device: torch.device,
    batch_size: int,
    workers: int,
) -> torch.Tensor:
    loader = torch.utils.data.DataLoader(
        _ImageDataset(source, indices, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
    )
    rows = []
    model.eval()
    with torch.inference_mode():
        for images, _ in loader:
            rows.append(model(images.to(device)).float().cpu())
    output = torch.cat(rows).contiguous()
    if output.shape != (len(indices), 768) or not bool(torch.isfinite(output).all()):
        raise ValueError("candidate feature authority differs")
    return output


def _parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-features", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-output", type=Path, required=True)
    parser.add_argument(
        "--objective", choices=("smooth-ap", "imprinted-proxy-anchor"), required=True
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--execute-rank-finish", action="store_true", required=True)
    return parser.parse_args(arguments)


def _require_checkout(checkout: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != UNICOM_REVISION:
        raise ValueError("teacher checkout authority differs")


def main(arguments: Sequence[str] | None = None) -> int:
    args = _parse_args(arguments)
    if (
        args.output.exists()
        or args.checkpoint_output.exists()
        or args.batch_size != 128
        or args.workers != 8
        or sha256(args.checkpoint) != CHECKPOINT_SHA256
        or sha256(args.baseline_features) != BASELINE_FEATURE_SHA256
        or sha256(args.preregistration) != args.preregistration_sha256
    ):
        raise ValueError("rank-finish input authority differs")
    _require_checkout(args.checkout)
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration.get("schema") != "sfora-unicom-lastblock-rank-finish-v1"
        or preregistration.get("script_sha256") != sha256(Path(__file__))
    ):
        raise ValueError("rank-finish preregistration authority differs")

    configure_deterministic_similarity_runtime(17, cpu_threads=8)
    from datasets import concatenate_datasets, load_dataset
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    source_bundle = load_dataset(DATASET_ID, revision=DATASET_REVISION)
    source = concatenate_datasets((source_bundle["train"], source_bundle["test"]))
    labels = np.asarray(source["label"], dtype=np.int64)
    fit_indices = np.flatnonzero(labels < 98).tolist()
    evaluation_indices = np.flatnonzero(labels >= 98).tolist()
    if (
        labels.shape != (16_185,)
        or len(fit_indices) != 8_054
        or len(evaluation_indices) != 8_131
    ):
        raise ValueError("Cars196 row authority differs")

    with np.load(args.baseline_features, allow_pickle=False) as archive:
        baseline_fit = torch.from_numpy(
            np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32)
        )
        fit_labels = torch.from_numpy(
            np.ascontiguousarray(archive["fit_labels"], dtype=np.int64)
        )
        baseline_evaluation = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_embeddings"], dtype=np.float32)
        )
        evaluation_labels = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_labels"], dtype=np.int64)
        )
    if (
        baseline_fit.shape != (8_054, 768)
        or baseline_evaluation.shape != (8_131, 768)
        or not np.array_equal(fit_labels.numpy(), labels[fit_indices])
        or not np.array_equal(evaluation_labels.numpy(), labels[evaluation_indices])
    ):
        raise ValueError("baseline feature authority differs")

    package_root = (args.checkout / "unicom").resolve()
    sys.path.insert(0, str(package_root))
    try:
        unicom = importlib.import_module("unicom")
    finally:
        sys.path.pop(0)
    model, evaluation_transform = unicom.load(
        "ViT-L/14@336px", download_root=str(args.checkpoint.parent)
    )
    authority = freeze_except_final_block(model)
    device = torch.device("cuda")
    model = model.to(device)
    training_transform = transforms.Compose(
        (
            transforms.RandomResizedCrop(
                336,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.Lambda(lambda image: image.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize(
                (0.48145466, 0.4578275, 0.40821073),
                (0.26862954, 0.26130258, 0.27577711),
            ),
        )
    )
    training_dataset = _ImageDataset(source, fit_indices, training_transform)
    training_labels = tuple(str(int(labels[index])) for index in fit_indices)
    steps_per_epoch = len(fit_indices) // args.batch_size
    parameters = [value for value in model.parameters() if value.requires_grad]
    proxies = None
    parameter_groups: list[dict[str, object]] = [{"params": parameters, "lr": 1e-5}]
    maximum_learning_rates = [1e-5]
    if args.objective == "imprinted-proxy-anchor":
        normalized_baseline = torch.nn.functional.normalize(baseline_fit, dim=1)
        imprinted = torch.stack(
            [
                normalized_baseline[fit_labels == label].mean(dim=0)
                for label in range(98)
            ]
        )
        proxies = torch.nn.Parameter(
            torch.nn.functional.normalize(imprinted, dim=1).to(device)
        )
        parameter_groups.append({"params": [proxies], "lr": 1e-3})
        maximum_learning_rates.append(1e-3)
    optimizer = torch.optim.AdamW(parameter_groups, lr=1e-5, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=maximum_learning_rates,
        steps_per_epoch=steps_per_epoch,
        epochs=4,
        pct_start=0.1,
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    history = []
    for epoch in range(1, 5):
        batches = identity_balanced_batches(
            training_labels,
            batch_size=128,
            images_per_identity=4,
            seed=17,
            epoch=epoch,
            steps=steps_per_epoch,
        )
        loader = torch.utils.data.DataLoader(
            training_dataset,
            batch_sampler=batches,
            num_workers=args.workers,
            pin_memory=True,
            generator=torch.Generator().manual_seed(17_000 + epoch),
        )
        model.eval()
        model.blocks[23].train()
        losses = []
        for step, (images, batch_labels) in enumerate(loader, start=1):
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images.to(device)).float()
            if args.objective == "smooth-ap":
                loss = smooth_ap_finish_loss(
                    embeddings,
                    tuple(int(value) for value in batch_labels.tolist()),
                )
            else:
                assert proxies is not None
                loss = proxy_anchor_loss(
                    embeddings,
                    batch_labels.to(device=device, dtype=torch.int64),
                    proxies,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [*parameters, *((proxies,) if proxies is not None else ())], 1.0
            )
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            if step % 10 == 0:
                print(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "step": step,
                            "steps": steps_per_epoch,
                            "loss": losses[-1],
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                    flush=True,
                )
        history.append(
            {
                "epoch": epoch,
                "steps": len(losses),
                "mean_loss": math.fsum(losses) / len(losses),
            }
        )

    candidate_fit = _encode(
        model,
        source,
        fit_indices,
        evaluation_transform,
        device=device,
        batch_size=128,
        workers=args.workers,
    )
    candidate_evaluation = _encode(
        model,
        source,
        evaluation_indices,
        evaluation_transform,
        device=device,
        batch_size=128,
        workers=args.workers,
    )
    baseline = _fit_score(
        baseline_fit,
        fit_labels,
        baseline_evaluation,
        evaluation_labels,
        device=device,
    )
    candidate = _fit_score(
        candidate_fit,
        fit_labels,
        candidate_evaluation,
        evaluation_labels,
        device=device,
    )
    interval = paired_class_bootstrap(
        candidate["per_query_ap"],  # type: ignore[arg-type]
        baseline["per_query_ap"],  # type: ignore[arg-type]
        evaluation_labels.numpy(),
    )
    reproduction_ok = (
        abs(float(baseline["map_at_r"]) - BASELINE["map_at_r"]) <= 0.001
        and abs(float(baseline["recall_at_1"]) - BASELINE["recall_at_1"]) <= 0.001
    )
    decision = classify_candidate(baseline, candidate, interval)
    decision["baseline_reproduction"] = reproduction_ok
    decision["passed"] = bool(decision["passed"] and reproduction_ok)
    del baseline["per_query_ap"]
    del candidate["per_query_ap"]

    checkpoint = {
        "schema": "sfora-unicom-lastblock-rank-finish-checkpoint-v1",
        "source_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip(),
        "official_checkpoint_sha256": CHECKPOINT_SHA256,
        "final_block": {
            name.removeprefix("blocks.23."): value.detach().cpu()
            for name, value in model.state_dict().items()
            if name.startswith("blocks.23.")
        },
    }
    torch.save(checkpoint, args.checkpoint_output)
    result = {
        "schema": "sfora-unicom-lastblock-rank-finish-result-v1",
        "claim_eligible": False,
        "dataset": "cars196-class-disjoint",
        "source_commit": checkpoint["source_commit"],
        "dataset_id": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "unicom_revision": UNICOM_REVISION,
        "official_checkpoint_sha256": CHECKPOINT_SHA256,
        "baseline_feature_sha256": BASELINE_FEATURE_SHA256,
        "preregistration_sha256": args.preregistration_sha256,
        "script_sha256": sha256(Path(__file__)),
        "checkpoint_sha256": sha256(args.checkpoint_output),
        "parameter_authority": authority,
        "method": {
            "objective": args.objective,
            "trainable_scope": "transformer-block-23-only",
            "epochs": 4,
            "batch_size": 128,
            "identities_per_batch": 32,
            "images_per_identity": 4,
            "loss": (
                "smooth-ap-deployment-prefix-v1"
                if args.objective == "smooth-ap"
                else "imprinted-proxy-anchor-v1"
            ),
            "loss_dimensions": 512 if args.objective == "smooth-ap" else 768,
            "temperature": 0.01 if args.objective == "smooth-ap" else None,
            "proxy_anchor_alpha": 32.0 if proxies is not None else None,
            "proxy_anchor_delta": 0.1 if proxies is not None else None,
            "proxy_learning_rate": 1e-3 if proxies is not None else None,
            "optimizer": "adamw",
            "maximum_learning_rate": 1e-5,
            "weight_decay": 0.0,
            "gradient_norm_cap": 1.0,
            "augmentation": "random-resized-crop-336-scale-0.8-1.0-plus-horizontal-flip",
        },
        "compact_representation": {
            "kind": "power-whitening-int8-128-plus-f16-inverse-norm",
            "bytes_per_item": 130,
            "alpha": 0.75,
            "regularization": 1.0,
        },
        "baseline": baseline,
        "candidate": candidate,
        "paired_class_bootstrap_map_delta_95": interval,
        "decision": decision,
        "next_action": "replicate-unchanged-on-cub" if decision["passed"] else "close-family",
        "history": history,
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": int(torch.cuda.max_memory_allocated()),
    }
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    )
    print(json.dumps({"output": str(args.output), "decision": decision}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
