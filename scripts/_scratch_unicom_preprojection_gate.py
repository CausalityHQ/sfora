#!/usr/bin/env python3
"""Fail-fast UniCOM pre-projection information gate on authenticated Cars features."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from sfora.compact_metric import fit_power_whitening_projection
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_width_control(
    rows: torch.Tensor, *, output_dimensions: int, seed: int
) -> torch.Tensor:
    """Add deterministic nonlinear random features without adding image information."""

    if (
        rows.ndim != 2
        or rows.dtype != torch.float32
        or output_dimensions <= rows.shape[1]
        or seed < 0
    ):
        raise ValueError("fixed-width control authority differs")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weight = torch.randn(
        rows.shape[1], output_dimensions - rows.shape[1], generator=generator
    ) / math.sqrt(rows.shape[1])
    return torch.cat((rows, torch.tanh(rows @ weight)), dim=1).contiguous()


def paired_class_bootstrap(
    candidate: np.ndarray, control: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
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
    rankings: list[torch.Tensor] = []
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
            ordered_scores = scores.gather(1, ordinal_order)
            score_order = torch.argsort(ordered_scores, dim=1, descending=True, stable=True)
            rankings.append(ordinal_order.gather(1, score_order)[:, :width].cpu())
    aps: list[float] = []
    hits: list[float] = []
    for ranking, label, positives in zip(
        torch.cat(rankings).tolist(), label_values, positive_counts, strict=True
    ):
        found = 0
        terms: list[float] = []
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


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--features-sha256", required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.features) != args.features_sha256
        or sha256(args.preregistration) != args.preregistration_sha256
        or sha256(Path(__file__)) != args.script_sha256
    ):
        raise ValueError("pre-projection gate authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration.get("schema") != "sfora-unicom-preprojection-gate-v1"
        or preregistration.get("source_commit") != args.source_commit
        or preregistration.get("script_sha256") != args.script_sha256
    ):
        raise ValueError("pre-projection gate authority differs")
    with np.load(args.features, allow_pickle=False) as archive:
        if set(archive.files) != {
            "metadata_json",
            "fit_preprojection",
            "fit_final",
            "fit_labels",
            "evaluation_preprojection",
            "evaluation_final",
            "evaluation_labels",
        }:
            raise ValueError("pre-projection feature authority differs")
        fit_final = torch.from_numpy(np.ascontiguousarray(archive["fit_final"], dtype=np.float32))
        fit_preprojection = torch.from_numpy(
            np.ascontiguousarray(archive["fit_preprojection"], dtype=np.float32)
        )
        fit_labels = torch.from_numpy(np.ascontiguousarray(archive["fit_labels"], dtype=np.int64))
        evaluation_final = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_final"], dtype=np.float32)
        )
        evaluation_preprojection = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_preprojection"], dtype=np.float32)
        )
        evaluation_labels = torch.from_numpy(
            np.ascontiguousarray(archive["evaluation_labels"], dtype=np.int64)
        )
    configure_deterministic_similarity_runtime(17, cpu_threads=8)
    device = torch.device("cuda")
    arms_input = {
        "final_768": (fit_final, evaluation_final),
        "final_derived_1024": (
            fixed_width_control(fit_final, output_dimensions=1_024, seed=17),
            fixed_width_control(evaluation_final, output_dimensions=1_024, seed=17),
        ),
        "preprojection_1024": (fit_preprojection, evaluation_preprojection),
    }
    started = time.monotonic()
    arms = {
        name: _fit_score(
            fit,
            fit_labels,
            evaluation,
            evaluation_labels,
            device=device,
        )
        for name, (fit, evaluation) in arms_input.items()
    }
    candidate = arms["preprojection_1024"]
    control_name = max(
        ("final_768", "final_derived_1024"),
        key=lambda name: (arms[name]["map_at_r"], arms[name]["recall_at_1"], name),
    )
    control = arms[control_name]
    interval = paired_class_bootstrap(
        candidate["per_query_ap"], control["per_query_ap"], evaluation_labels.numpy()
    )
    map_delta = float(candidate["map_at_r"]) - float(control["map_at_r"])
    recall_delta = float(candidate["recall_at_1"]) - float(control["recall_at_1"])
    reproduction_ok = (
        abs(float(arms["final_768"]["map_at_r"]) - 0.860658) <= 0.001
        and abs(float(arms["final_768"]["recall_at_1"]) - 0.974419) <= 0.001
    )
    control_ok = (
        float(arms["final_derived_1024"]["map_at_r"])
        - float(arms["final_768"]["map_at_r"])
        <= 0.003
    )
    passed = (
        reproduction_ok
        and control_ok
        and map_delta >= 0.005
        and interval["lower"] > 0.0
        and recall_delta >= 0.0
    )
    for arm in arms.values():
        del arm["per_query_ap"]
    result = {
        "schema": "sfora-unicom-preprojection-gate-result-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "features_sha256": args.features_sha256,
        "preregistration_sha256": args.preregistration_sha256,
        "script_sha256": args.script_sha256,
        "dataset": "cars196-class-disjoint",
        "representation": "signed-int8-128-plus-f16-inverse-norm",
        "bytes_per_item": 130,
        "fixed_power_whitening": {"alpha": 0.75, "regularization": 1.0},
        "arms": arms,
        "candidate": "preprojection_1024",
        "strongest_control": control_name,
        "map_at_r_delta": map_delta,
        "recall_at_1_delta": recall_delta,
        "paired_class_bootstrap_map_delta_95": interval,
        "gate": {
            "baseline_reproduction": reproduction_ok,
            "derived_width_control": control_ok,
            "minimum_map_at_r_delta": 0.005,
            "require_positive_interval_lower": True,
            "minimum_recall_at_1_delta": 0.0,
            "passed": passed,
        },
        "next_action": "freeze-unseen-replication" if passed else "close-preprojection-family",
        "elapsed_seconds": time.monotonic() - started,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.write_text(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
