#!/usr/bin/env python3
"""Fail-fast power-initialized mixed-positive-objective diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path
from types import ModuleType

import torch

import sfora.compact_metric as compact_metric
from sfora.compact_metric import CompactMetricEncoder, _score_compact_metric_codes
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.teacher_anchored_distillation import positive_coverage_hard_negative_loss


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def class_key(dataset: str, label: int) -> bytes:
    return hashlib.sha256(f"power-whitening-v1:{dataset}:{label}".encode()).digest()


def mixed_loss(
    positive_similarities: torch.Tensor,
    positive_mask: torch.Tensor,
    negative_similarities: torch.Tensor,
    self_similarities: torch.Tensor,
    *,
    temperature: float,
    margin: float,
    anchor_weight: float,
    positive_aggregation: str,
) -> torch.Tensor:
    if positive_aggregation != "mean_logit":
        raise ValueError("power mixed objective authority differs")
    common = {
        "temperature": temperature,
        "margin": margin,
        "anchor_weight": anchor_weight,
    }
    coverage = positive_coverage_hard_negative_loss(
        positive_similarities,
        positive_mask,
        negative_similarities,
        self_similarities,
        positive_aggregation="coverage",
        **common,
    )
    nearest = positive_coverage_hard_negative_loss(
        positive_similarities,
        positive_mask,
        negative_similarities,
        self_similarities,
        positive_aggregation="pooled",
        **common,
    )
    return 0.5 * coverage + 0.5 * nearest


def fit_mixed(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    initial_encoder: CompactMetricEncoder,
    device: torch.device,
) -> CompactMetricEncoder:
    original = compact_metric.positive_coverage_hard_negative_loss
    compact_metric.positive_coverage_hard_negative_loss = mixed_loss
    try:
        return compact_metric.fit_compact_metric_projection(
            embeddings, labels, initial_encoder=initial_encoder, device=device
        ).encoder
    finally:
        compact_metric.positive_coverage_hard_negative_loss = original


def score(
    encoder: CompactMetricEncoder,
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    values = _score_compact_metric_codes(encoder.encode(embeddings), labels, device=device)
    return float(values[0]), float(values[1])


def run_dataset(
    name: str,
    data: dict[str, torch.Tensor],
    selected_cell: str,
    source_module: ModuleType,
    device: torch.device,
) -> dict[str, object]:
    fit = data["fit_embeddings"]
    labels = data["fit_labels"]
    classes = sorted(
        (int(value) for value in torch.unique(labels)), key=lambda value: class_key(name, value)
    )
    arms = ("pca", "learned", "power", "power_mean", "power_mixed")
    pooled = {arm: [0.0, 0.0, 0] for arm in arms}
    folds = []
    for fold in range(2):
        training_mask = torch.isin(labels, torch.tensor(classes[fold::2], dtype=torch.int64))
        validation_mask = torch.isin(
            labels, torch.tensor(classes[1 - fold :: 2], dtype=torch.int64)
        )
        training = fit[training_mask].contiguous()
        training_labels = labels[training_mask].contiguous()
        validation = fit[validation_mask].contiguous()
        validation_labels = labels[validation_mask].contiguous()
        power_cells = source_module.fit_cells(
            training,
            training_labels,
            "within",
            f"power-mixed-objective-v1:{name}:{fold}",
        )
        power = power_cells[selected_cell]
        fitted = {
            "pca": power_cells["0.00:0.00"],
            "learned": compact_metric.fit_compact_metric_projection(
                training, training_labels, device=device
            ).encoder,
            "power": power,
            "power_mean": compact_metric.fit_compact_metric_projection(
                training, training_labels, initial_encoder=power, device=device
            ).encoder,
            "power_mixed": fit_mixed(training, training_labels, power, device),
        }
        scores = {}
        for arm, encoder in fitted.items():
            map_at_r, recall = score(encoder, validation, validation_labels, device)
            scores[arm] = {"map_at_r": map_at_r, "recall_at_1": recall}
            pooled[arm][0] += len(validation) * map_at_r
            pooled[arm][1] += len(validation) * recall
            pooled[arm][2] += len(validation)
        folds.append({"fold": fold, "validation_rows": len(validation), "scores": scores})
    combined = {
        arm: {"map_at_r": row[0] / row[2], "recall_at_1": row[1] / row[2]}
        for arm, row in pooled.items()
    }
    incumbent = max(
        ("learned", "power", "power_mean"),
        key=lambda arm: (combined[arm]["map_at_r"], combined[arm]["recall_at_1"], arm),
    )
    return {
        "selected_cell": selected_cell,
        "folds": folds,
        "pooled": combined,
        "incumbent": incumbent,
        "mixed_minus_incumbent": {
            metric: combined["power_mixed"][metric] - combined[incumbent][metric]
            for metric in ("map_at_r", "recall_at_1")
        },
    }


def outer(
    data: dict[str, torch.Tensor],
    selected_cell: str,
    source_module: ModuleType,
    device: torch.device,
) -> dict[str, object]:
    fit = data["fit_embeddings"]
    labels = data["fit_labels"]
    power = source_module.fit_cells(
        fit, labels, "within", "power-mixed-objective-v1:outer"
    )[selected_cell]
    mixed = fit_mixed(fit, labels, power, device)
    result = {}
    for name, encoder in (("power", power), ("power_mixed", mixed)):
        map_at_r, recall = score(
            encoder, data["evaluation_embeddings"], data["evaluation_labels"], device
        )
        result[name] = {
            "map_at_r": map_at_r,
            "recall_at_1": recall,
            "encoder_sha256": encoder.sha256,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-script", type=Path, required=True)
    parser.add_argument("--source-script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset", action="append", nargs=3, metavar=("NAME", "PATH", "SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(Path(__file__)) != args.script_sha256
        or sha256(args.preregistration) != args.preregistration_sha256
        or sha256(args.source_script) != args.source_script_sha256
        or not args.dataset
    ):
        raise ValueError("power mixed objective authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    supplied = {name: (Path(path), digest) for name, path, digest in args.dataset}
    if (
        preregistration["schema"] != "sfora-power-mixed-objective-preregistration-v1"
        or preregistration["source_commit"] != args.source_commit
        or preregistration["script_sha256"] != args.script_sha256
        or preregistration["source_script_sha256"] != args.source_script_sha256
        or set(supplied) != set(preregistration["datasets"])
    ):
        raise ValueError("power mixed objective authority differs")
    for name, (path, digest) in supplied.items():
        if preregistration["datasets"][name] != digest or sha256(path) != digest:
            raise ValueError("power mixed objective authority differs")
    source_module = load_module("power_source_control", args.source_script)
    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    started = time.monotonic()
    selected_cells = preregistration["selected_cells"]
    data = {
        name: source_module.load_food(path) if name == "food101" else source_module.load_cars(path)
        for name, (path, _digest) in sorted(supplied.items())
    }
    datasets = {
        name: run_dataset(name, row, selected_cells[name], source_module, device)
        for name, row in data.items()
    }
    map_deltas = [row["mixed_minus_incumbent"]["map_at_r"] for row in datasets.values()]
    recall_deltas = [row["mixed_minus_incumbent"]["recall_at_1"] for row in datasets.values()]
    passed = (
        math.fsum(map_deltas) / len(map_deltas) >= 0.002
        and math.fsum(recall_deltas) / len(recall_deltas) >= 0.0
        and min(map_deltas) >= -0.003
        and min(recall_deltas) >= -0.001
    )
    result = {
        "schema": "sfora-power-mixed-objective-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": args.script_sha256,
        "source_script_sha256": args.source_script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "datasets": datasets,
        "decision": {
            "passed": passed,
            "macro_mixed_minus_incumbent_map_at_r": math.fsum(map_deltas) / len(map_deltas),
            "macro_mixed_minus_incumbent_recall_at_1": math.fsum(recall_deltas)
            / len(recall_deltas),
        },
        "outer": (
            {
                name: outer(row, selected_cells[name], source_module, device)
                for name, row in data.items()
            }
            if passed
            else None
        ),
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
