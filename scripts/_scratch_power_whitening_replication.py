#!/usr/bin/env python3
"""Prospective three-dataset replication of the frozen power-whitening selector."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.compact_metric import _compact_metric_lexicographic_topk, _score_compact_metric_codes
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime


class FloatEncoder:
    def __init__(self, encoder: object) -> None:
        self.encoder = encoder

    def encode(self, values: torch.Tensor) -> torch.Tensor:
        return self.encoder.transform(values)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered(values: np.ndarray, namespace: str) -> list[str]:
    return sorted(
        (str(value) for value in np.unique(values)),
        key=lambda value: hashlib.sha256(f"{namespace}:{value}".encode()).digest(),
    )


def integer_labels(values: np.ndarray, universe: list[str]) -> torch.Tensor:
    mapping = {value: index for index, value in enumerate(universe)}
    return torch.tensor([mapping[str(value)] for value in values], dtype=torch.int64)


def score_same(
    encoder: object, embeddings: torch.Tensor, labels: torch.Tensor, device: torch.device
) -> tuple[float, float]:
    score = _score_compact_metric_codes(encoder.encode(embeddings), labels, device=device)
    return float(score[0]), float(score[1])


def score_query_gallery(
    encoder: object,
    query: torch.Tensor,
    gallery: torch.Tensor,
    query_labels: torch.Tensor,
    gallery_labels: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    query_values = F.normalize(encoder.encode(query).float(), dim=1).to(device)
    gallery_values = F.normalize(encoder.encode(gallery).float(), dim=1).to(device)
    gallery_label_values = tuple(int(value) for value in gallery_labels.tolist())
    counts = Counter(gallery_label_values)
    width = max(counts.values())
    rankings = []
    with torch.inference_mode():
        for start in range(0, len(query_values), 256):
            similarities = query_values[start : start + 256] @ gallery_values.T
            rankings.append(_compact_metric_lexicographic_topk(similarities, width).cpu())
    average_precision = []
    recall_at_1 = []
    for ranking, label in zip(torch.cat(rankings).tolist(), query_labels.tolist(), strict=True):
        positives = counts[int(label)]
        found = 0
        terms = []
        for rank, row in enumerate(ranking[:positives], start=1):
            if gallery_label_values[row] == label:
                found += 1
                terms.append(found / rank)
        average_precision.append(math.fsum(terms) / positives)
        recall_at_1.append(float(gallery_label_values[ranking[0]] == label))
    return (
        math.fsum(average_precision) / len(average_precision),
        math.fsum(recall_at_1) / len(recall_at_1),
    )


def load_class_disjoint(path: Path, dataset: str) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        features = np.ascontiguousarray(archive["features"], dtype=np.float32)
        labels = np.ascontiguousarray(archive["labels"])
    if dataset == "food101":
        if features.shape != (25_250, 768) or len(np.unique(labels)) != 101:
            raise ValueError("power whitening replication dataset differs")
        values = ordered(labels, "food-whitening-selector-v1")
        fit_values = set(values[:50])
    elif dataset == "flowers102":
        if features.shape != (6_149, 768) or len(np.unique(labels)) != 102:
            raise ValueError("power whitening replication dataset differs")
        values = ordered(labels, "flowers-whitening-selector-v1")
        fit_values = set(values[:51])
    else:
        raise ValueError("power whitening replication dataset differs")
    fit_mask = np.asarray([str(value) in fit_values for value in labels])
    evaluation_mask = ~fit_mask
    mapped = integer_labels(labels, values)
    return {
        "fit": torch.from_numpy(features[fit_mask].copy()),
        "fit_labels": mapped[torch.from_numpy(fit_mask)].contiguous(),
        "evaluation": torch.from_numpy(features[evaluation_mask].copy()),
        "evaluation_labels": mapped[torch.from_numpy(evaluation_mask)].contiguous(),
    }


def load_inshop(path: Path) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        train = np.ascontiguousarray(archive["train_embeddings"], dtype=np.float32)
        train_raw = np.ascontiguousarray(archive["train_labels"])
        query = np.ascontiguousarray(archive["query_embeddings"], dtype=np.float32)
        query_raw = np.ascontiguousarray(archive["query_labels"])
        gallery = np.ascontiguousarray(archive["gallery_embeddings"], dtype=np.float32)
        gallery_raw = np.ascontiguousarray(archive["gallery_labels"])
    values, counts = np.unique(train_raw, return_counts=True)
    eligible_values = values[counts >= 2]
    eligible = np.isin(train_raw, eligible_values)
    train = train[eligible].copy()
    train_raw = train_raw[eligible].copy()
    train_values = sorted(str(value) for value in np.unique(train_raw))
    evaluation_values = sorted(str(value) for value in np.unique(query_raw))
    if (
        train.shape != (25_870, 768)
        or query.shape != (14_218, 768)
        or gallery.shape != (12_612, 768)
        or set(train_values).intersection(evaluation_values)
        or set(evaluation_values) != set(str(value) for value in np.unique(gallery_raw))
    ):
        raise ValueError("power whitening replication dataset differs")
    return {
        "fit": torch.from_numpy(train),
        "fit_labels": integer_labels(train_raw, train_values),
        "evaluation": torch.from_numpy(query),
        "evaluation_labels": integer_labels(query_raw, evaluation_values),
        "gallery": torch.from_numpy(gallery),
        "gallery_labels": integer_labels(gallery_raw, evaluation_values),
    }


def screen(
    name: str, data: dict[str, torch.Tensor], module: object, device: torch.device
) -> dict[str, object]:
    labels = data["fit_labels"]
    classes = sorted(
        (int(value) for value in torch.unique(labels)),
        key=lambda value: hashlib.sha256(f"power-whitening-v1:{name}:{value}".encode()).digest(),
    )
    pooled = {
        f"{alpha:.2f}:{regularization:.2f}": [0.0, 0.0, 0]
        for alpha, regularization in module.CELLS
    }
    folds = []
    for fold in range(2):
        training_classes = classes[fold::2]
        validation_classes = classes[1 - fold :: 2]
        training_mask = torch.isin(labels, torch.tensor(training_classes, dtype=torch.int64))
        validation_mask = torch.isin(labels, torch.tensor(validation_classes, dtype=torch.int64))
        training = data["fit"][training_mask].contiguous()
        training_labels = labels[training_mask].contiguous()
        validation = data["fit"][validation_mask].contiguous()
        validation_labels = labels[validation_mask].contiguous()
        scores = {}
        for cell, encoder in module.fit_cells(training, training_labels).items():
            key = f"{cell[0]:.2f}:{cell[1]:.2f}"
            map_at_r, recall = score_same(encoder, validation, validation_labels, device)
            scores[key] = {"map_at_r": map_at_r, "recall_at_1": recall}
            pooled[key][0] += len(validation) * map_at_r
            pooled[key][1] += len(validation) * recall
            pooled[key][2] += len(validation)
        folds.append(
            {
                "fold": fold,
                "training_classes": len(training_classes),
                "training_rows": len(training),
                "validation_classes": len(validation_classes),
                "validation_rows": len(validation),
                "scores": scores,
            }
        )
    combined = {
        key: {"map_at_r": value[0] / value[2], "recall_at_1": value[1] / value[2]}
        for key, value in pooled.items()
    }
    selected = max(
        combined, key=lambda key: (combined[key]["map_at_r"], combined[key]["recall_at_1"], key)
    )
    return {"folds": folds, "pooled": combined, "selected": selected}


def outer(
    data: dict[str, torch.Tensor], selected_key: str, module: object, device: torch.device
) -> dict[str, object]:
    alpha, regularization = (float(value) for value in selected_key.split(":"))
    encoders = module.fit_cells(data["fit"], data["fit_labels"])
    arms = {}
    for name, encoder in (
        ("pca", encoders[(0.0, 0.0)]),
        ("selected", encoders[(alpha, regularization)]),
    ):
        if "gallery" in data:
            float_score = score_query_gallery(
                FloatEncoder(encoder),
                data["evaluation"],
                data["gallery"],
                data["evaluation_labels"],
                data["gallery_labels"],
                device,
            )
            int8_score = score_query_gallery(
                encoder,
                data["evaluation"],
                data["gallery"],
                data["evaluation_labels"],
                data["gallery_labels"],
                device,
            )
        else:
            float_score = _score_compact_metric_codes(
                encoder.transform(data["evaluation"]), data["evaluation_labels"], device=device
            )[:2]
            int8_score = _score_compact_metric_codes(
                encoder.encode(data["evaluation"]), data["evaluation_labels"], device=device
            )[:2]
        arms[name] = {
            "float_map_at_r": float(float_score[0]),
            "float_recall_at_1": float(float_score[1]),
            "int8_map_at_r": float(int8_score[0]),
            "int8_recall_at_1": float(int8_score[1]),
            "encoder_sha256": encoder.sha256,
        }
    return {"selected_cell": selected_key, "arms": arms}


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--screen-script", type=Path, required=True)
    parser.add_argument("--screen-script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset", action="append", nargs=3, metavar=("NAME", "PATH", "SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(Path(__file__)) != args.script_sha256
        or sha256(args.preregistration) != args.preregistration_sha256
        or sha256(args.screen_script) != args.screen_script_sha256
        or not args.dataset
    ):
        raise ValueError("power whitening replication authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    supplied = {name: (Path(path), digest) for name, path, digest in args.dataset}
    if (
        preregistration["schema"] != "sfora-power-whitening-replication-preregistration-v1"
        or preregistration["script_sha256"] != args.script_sha256
        or preregistration["screen_script_sha256"] != args.screen_script_sha256
        or preregistration["source_commit"] != args.source_commit
        or set(supplied) != set(preregistration["datasets"])
    ):
        raise ValueError("power whitening replication authority differs")
    for name, (path, digest) in supplied.items():
        if digest != preregistration["datasets"][name] or sha256(path) != digest:
            raise ValueError("power whitening replication authority differs")
    spec = importlib.util.spec_from_file_location("power_whitening_screen", args.screen_script)
    if spec is None or spec.loader is None:
        raise ImportError(args.screen_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    started = time.monotonic()
    data = {
        name: load_inshop(path) if name == "inshop" else load_class_disjoint(path, name)
        for name, (path, _digest) in sorted(supplied.items())
    }
    screens = {name: screen(name, row, module, device) for name, row in data.items()}
    pca_key = "0.00:0.00"
    gains = {
        name: row["pooled"][row["selected"]]["map_at_r"] - row["pooled"][pca_key]["map_at_r"]
        for name, row in screens.items()
    }
    recall_gains = {
        name: row["pooled"][row["selected"]]["recall_at_1"]
        - row["pooled"][pca_key]["recall_at_1"]
        for name, row in screens.items()
    }
    passed = (
        math.fsum(gains.values()) / len(gains) >= 0.002
        and min(gains.values()) >= -0.003
        and min(recall_gains.values()) >= -0.003
    )
    result = {
        "schema": "sfora-power-whitening-replication-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": args.script_sha256,
        "screen_script_sha256": args.screen_script_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "screens": screens,
        "decision": {
            "passed": passed,
            "macro_selected_minus_pca_map_at_r": math.fsum(gains.values()) / len(gains),
            "per_dataset_selected_minus_pca_map_at_r": gains,
            "per_dataset_selected_minus_pca_recall_at_1": recall_gains,
        },
        "outer": (
            {name: outer(data[name], screens[name]["selected"], module, device) for name in data}
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
