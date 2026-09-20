#!/usr/bin/env python3
"""Compare sealed power-code top-1 errors with their frozen float teacher."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.compact_metric import _compact_metric_lexicographic_topk
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime


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


def load_class_disjoint(path: Path, name: str) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as archive:
        if "fit_embeddings" in archive:
            return {
                "fit": torch.from_numpy(
                    np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32)
                ),
                "fit_labels": torch.from_numpy(
                    np.ascontiguousarray(archive["fit_labels"], dtype=np.int64)
                ),
                "query": torch.from_numpy(
                    np.ascontiguousarray(archive["evaluation_embeddings"], dtype=np.float32)
                ),
                "query_labels": torch.from_numpy(
                    np.ascontiguousarray(archive["evaluation_labels"], dtype=np.int64)
                ),
            }
        features = np.ascontiguousarray(archive["features"], dtype=np.float32)
        raw_labels = np.ascontiguousarray(archive["labels"])
    if name != "food101" or features.shape != (25_250, 768):
        raise ValueError("compact teacher census dataset differs")
    values = ordered(raw_labels, "food-whitening-selector-v1")
    fit_values = set(values[:50])
    fit_mask = np.asarray([str(value) in fit_values for value in raw_labels])
    labels = integer_labels(raw_labels, values)
    return {
        "fit": torch.from_numpy(features[fit_mask].copy()),
        "fit_labels": labels[torch.from_numpy(fit_mask)].contiguous(),
        "query": torch.from_numpy(features[~fit_mask].copy()),
        "query_labels": labels[torch.from_numpy(~fit_mask)].contiguous(),
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
    eligible = np.isin(train_raw, values[counts >= 2])
    train = train[eligible].copy()
    train_raw = train_raw[eligible].copy()
    train_values = sorted(str(value) for value in np.unique(train_raw))
    evaluation_values = sorted(str(value) for value in np.unique(query_raw))
    return {
        "fit": torch.from_numpy(train),
        "fit_labels": integer_labels(train_raw, train_values),
        "query": torch.from_numpy(query),
        "query_labels": integer_labels(query_raw, evaluation_values),
        "gallery": torch.from_numpy(gallery),
        "gallery_labels": integer_labels(gallery_raw, evaluation_values),
    }


def top1(
    query: torch.Tensor,
    gallery: torch.Tensor,
    device: torch.device,
    same_set: bool,
) -> torch.Tensor:
    query = F.normalize(query.float(), dim=1).to(device)
    gallery = F.normalize(gallery.float(), dim=1).to(device)
    rows = []
    with torch.inference_mode():
        for start in range(0, len(query), 512):
            similarities = query[start : start + 512] @ gallery.T
            if same_set:
                stop = min(start + len(similarities), len(gallery))
                local = torch.arange(stop - start, device=device)
                similarities[local, torch.arange(start, stop, device=device)] = -torch.inf
            rows.append(_compact_metric_lexicographic_topk(similarities, 1)[:, 0].cpu())
    return torch.cat(rows)


def distribution(values: torch.Tensor) -> dict[str, float]:
    array = values.double().numpy()
    return {
        "mean": float(array.mean()) if len(array) else 0.0,
        "median": float(np.median(array)) if len(array) else 0.0,
        "p90": float(np.quantile(array, 0.9)) if len(array) else 0.0,
    }


def census(
    data: dict[str, torch.Tensor],
    encoder: object,
    device: torch.device,
) -> dict[str, object]:
    query = data["query"]
    gallery = data.get("gallery", query)
    query_labels = data["query_labels"]
    gallery_labels = data.get("gallery_labels", query_labels)
    same_set = "gallery" not in data
    teacher_top1 = top1(query, gallery, device, same_set)
    compact_top1 = top1(encoder.encode(query), encoder.encode(gallery), device, same_set)
    teacher_ok = gallery_labels[teacher_top1] == query_labels
    compact_ok = gallery_labels[compact_top1] == query_labels
    indegree = torch.bincount(compact_top1, minlength=len(gallery))
    compact_error = ~compact_ok
    shared_error = compact_error & ~teacher_ok
    return {
        "query_rows": len(query),
        "gallery_rows": len(gallery),
        "teacher_recall_at_1": float(teacher_ok.double().mean()),
        "compact_recall_at_1": float(compact_ok.double().mean()),
        "compact_error_count": int(compact_error.sum()),
        "teacher_error_count": int((~teacher_ok).sum()),
        "shared_error_count": int(shared_error.sum()),
        "compact_only_error_count": int((compact_error & teacher_ok).sum()),
        "teacher_only_error_count": int((compact_ok & ~teacher_ok).sum()),
        "compact_error_teacher_error_share": (
            float(shared_error.sum() / compact_error.sum()) if compact_error.any() else 1.0
        ),
        "same_top1_share": float((teacher_top1 == compact_top1).double().mean()),
        "wrong_target_indegree": distribution(indegree[compact_top1[compact_error]]),
        "correct_target_indegree": distribution(indegree[compact_top1[compact_ok]]),
        "compact_target_max_indegree": int(indegree.max()),
    }


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
        raise ValueError("compact teacher census authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    supplied = {name: (Path(path), digest) for name, path, digest in args.dataset}
    if (
        preregistration["schema"] != "sfora-compact-teacher-error-census-preregistration-v1"
        or preregistration["source_commit"] != args.source_commit
        or preregistration["script_sha256"] != args.script_sha256
        or preregistration["screen_script_sha256"] != args.screen_script_sha256
        or set(supplied) != set(preregistration["datasets"])
    ):
        raise ValueError("compact teacher census authority differs")
    for name, (path, digest) in supplied.items():
        if preregistration["datasets"][name] != digest or sha256(path) != digest:
            raise ValueError("compact teacher census authority differs")
    spec = importlib.util.spec_from_file_location("power_whitening_screen", args.screen_script)
    if spec is None or spec.loader is None:
        raise ImportError(args.screen_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    started = time.monotonic()
    datasets = {}
    for name, (path, _digest) in sorted(supplied.items()):
        data = load_inshop(path) if name == "inshop" else load_class_disjoint(path, name)
        alpha, regularization = preregistration["selected_cells"][name]
        encoder = module.fit_cells(data["fit"], data["fit_labels"])[
            (alpha, regularization)
        ]
        datasets[name] = census(data, encoder, device)
    overlap = [row["compact_error_teacher_error_share"] for row in datasets.values()]
    result = {
        "schema": "sfora-compact-teacher-error-census-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "preregistration_sha256": args.preregistration_sha256,
        "datasets": datasets,
        "decision": {
            "all_teacher_inherited_above_0.90": all(value >= 0.9 for value in overlap),
            "macro_compact_error_teacher_error_share": math.fsum(overlap) / len(overlap),
        },
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
