#!/usr/bin/env python3
"""Frozen four-arm, five-dataset same-teacher representation ladder."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from _scratch_cub_direct256_int4 import remap_labels, train_width
from _scratch_sop_direct256_int4 import fit_pack_decode_int4, fit_pca_affine
from probe_sop_relational_linear import _lexicographic_candidates
from torch.nn import functional as F

from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.representation_ceiling import deterministic_class_partition

BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20_260_920
MINIMUM_MAP_EFFECT = 0.005
ARM_NAMES = (
    "pca128_int8",
    "pca256_int4",
    "learned128_int8",
    "learned256_int4",
)
EXPECTED_SHA256 = {
    "cars": "6ddfd7e2c9fd489dff51fa33697c62abf92a45247b52b336c0371f5482231ab3",
    "cub": "847a40bd8c0c2a5289de9b5a8eca93c935d4eafed6507e1360da3a3bfee6f62e",
    "sop": "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a",
    "inshop": "05cd5901425210c06a3972f5a67acf41c961b2f4b59d6535744f3e3536d036ad",
    "food101": "277c192d91ae05ec5389f5cb70e43e5b88d8898140a2a06c64f6d8961e9893b5",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def paired_query_bootstrap(
    candidate: np.ndarray,
    baseline: np.ndarray,
    *,
    seed: int,
    replicates: int,
) -> dict[str, Any]:
    """Return a deterministic paired query bootstrap interval for a mean delta."""

    if (
        type(candidate) is not np.ndarray
        or type(baseline) is not np.ndarray
        or candidate.ndim != 1
        or baseline.shape != candidate.shape
        or len(candidate) == 0
        or not np.isfinite(candidate).all()
        or not np.isfinite(baseline).all()
        or type(seed) is not int
        or type(replicates) is not int
        or replicates < 2
    ):
        raise ValueError("paired query authority differs")
    difference = candidate.astype(np.float64) - baseline.astype(np.float64)
    rng = np.random.Generator(np.random.PCG64(seed))
    values = np.empty(replicates, dtype=np.float64)
    for start in range(0, replicates, 64):
        stop = min(start + 64, replicates)
        indexes = rng.integers(0, len(difference), size=(stop - start, len(difference)))
        values[start:stop] = difference[indexes].mean(axis=1)
    low, high = np.quantile(values, (0.025, 0.975), method="linear")
    delta = float(difference.mean())
    return {
        "delta": delta,
        "ci95": [float(low), float(high)],
        "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
        "meets_minimum_effect": bool(delta >= MINIMUM_MAP_EFFECT),
        "survives": bool(delta >= MINIMUM_MAP_EFFECT and low > 0.0),
    }


def _normalise(array: np.ndarray) -> torch.Tensor:
    value = F.normalize(torch.from_numpy(array.copy()).float(), dim=1).contiguous()
    if value.ndim != 2 or value.shape[1] != 768 or not bool(torch.isfinite(value).all()):
        raise ValueError("teacher embedding authority differs")
    return value


def _score(
    queries: torch.Tensor | PackedInt8Embeddings,
    gallery: torch.Tensor | PackedInt8Embeddings,
    query_labels: tuple[int, ...],
    gallery_labels: tuple[int, ...],
    *,
    same_rows: bool,
    device: torch.device,
) -> dict[str, Any]:
    if isinstance(queries, PackedInt8Embeddings):
        if not isinstance(gallery, PackedInt8Embeddings):
            raise ValueError("ladder scoring representation differs")
        query_rows = len(queries.codes)
        gallery_rows = len(gallery.codes)
        dimensions = queries.codes.shape[1]
        gallery_dimensions = gallery.codes.shape[1]
        packed = True
    elif type(queries) is torch.Tensor and type(gallery) is torch.Tensor:
        query_rows = len(queries)
        gallery_rows = len(gallery)
        dimensions = queries.shape[1]
        gallery_dimensions = gallery.shape[1]
        packed = False
    else:
        raise ValueError("ladder scoring representation differs")
    if (
        dimensions != gallery_dimensions
        or query_rows != len(query_labels)
        or gallery_rows != len(gallery_labels)
        or (same_rows and (query_rows != gallery_rows or query_labels != gallery_labels))
    ):
        raise ValueError("ladder scoring authority differs")
    counts = Counter(gallery_labels)
    positive_counts = [counts[label] - int(same_rows) for label in query_labels]
    if min(positive_counts) < 1:
        raise ValueError("ladder positive authority differs")
    width = max(positive_counts)
    if packed:
        assert isinstance(queries, PackedInt8Embeddings)
        assert isinstance(gallery, PackedInt8Embeddings)
        q_codes = queries.codes.to(device=device, dtype=torch.float32)
        g_codes = gallery.codes.to(device=device, dtype=torch.float32)
        q_norms = queries.inverse_norms.to(device=device, dtype=torch.float32)
        g_norms = gallery.inverse_norms.to(device=device, dtype=torch.float32)
    else:
        assert isinstance(queries, torch.Tensor)
        assert isinstance(gallery, torch.Tensor)
        q = queries.to(device)
        g = gallery.to(device)
    rankings: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, query_rows, 256):
            stop = min(start + 256, query_rows)
            if packed:
                scores = (
                    (q_codes[start:stop] @ g_codes.T)
                    * q_norms[start:stop].unsqueeze(1)
                    * g_norms.unsqueeze(0)
                )
            else:
                scores = q[start:stop] @ g.T
            if same_rows:
                scores[
                    torch.arange(stop - start, device=device),
                    torch.arange(start, stop, device=device),
                ] = -torch.inf
            rankings.append(_lexicographic_candidates(scores, width).cpu())
    aps: list[float] = []
    hits: list[float] = []
    for ranking, label, positives in zip(
        torch.cat(rankings).tolist(), query_labels, positive_counts, strict=True
    ):
        found = 0
        terms: list[float] = []
        for rank, index in enumerate(ranking[:positives], start=1):
            if gallery_labels[index] == label:
                found += 1
                terms.append(found / rank)
        aps.append(math.fsum(terms) / positives)
        hits.append(float(gallery_labels[ranking[0]] == label))
    return {
        "map_at_r": math.fsum(aps) / len(aps),
        "recall_at_1": math.fsum(hits) / len(hits),
        "per_query_ap": aps,
        "per_query_r1": hits,
    }


def _class_disjoint(path: Path) -> tuple[Any, ...]:
    with np.load(path, allow_pickle=False) as archive:
        fit = _normalise(archive["fit_embeddings"])
        evaluation = _normalise(archive["evaluation_embeddings"])
        fit_labels = remap_labels(archive["fit_labels"].astype(np.int64))
        evaluation_labels = remap_labels(archive["evaluation_labels"].astype(np.int64))
    return fit, fit_labels, evaluation, evaluation_labels, evaluation, evaluation_labels, True


def _sop(path: Path) -> tuple[Any, ...]:
    with np.load(path, allow_pickle=False) as archive:
        fit = _normalise(archive["train_embeddings"])
        evaluation = _normalise(archive["test_embeddings"])
        fit_labels = remap_labels(archive["train_labels"].astype(np.int64))
        evaluation_labels = remap_labels(archive["test_labels"].astype(np.int64))
    return fit, fit_labels, evaluation, evaluation_labels, evaluation, evaluation_labels, True


def _food(path: Path) -> tuple[Any, ...]:
    with np.load(path, allow_pickle=False) as archive:
        features = _normalise(archive["features"])
        labels = archive["labels"].astype(np.int64)
    partition = deterministic_class_partition(
        tuple(int(value) for value in labels), fit_fraction=0.8, seed=17
    )
    fit_indexes = np.asarray(partition.fit_row_indexes, dtype=np.int64)
    evaluation_indexes = np.asarray(partition.validation_row_indexes, dtype=np.int64)
    fit = features[torch.from_numpy(fit_indexes)]
    evaluation = features[torch.from_numpy(evaluation_indexes)]
    fit_labels = remap_labels(labels[fit_indexes])
    evaluation_labels = remap_labels(labels[evaluation_indexes])
    return fit, fit_labels, evaluation, evaluation_labels, evaluation, evaluation_labels, True


def _inshop(path: Path) -> tuple[Any, ...]:
    with np.load(path, allow_pickle=False) as archive:
        raw_fit_labels = archive["train_labels"].copy()
        values, counts = np.unique(raw_fit_labels, return_counts=True)
        eligible = np.isin(raw_fit_labels, values[counts >= 2])
        fit = _normalise(archive["train_embeddings"][eligible])
        fit_labels = remap_labels(raw_fit_labels[eligible])
        query = _normalise(archive["query_embeddings"])
        gallery = _normalise(archive["gallery_embeddings"])
        query_raw = archive["query_labels"].copy()
        gallery_raw = archive["gallery_labels"].copy()
    authority = {
        value: index + 1
        for index, value in enumerate(sorted(set(query_raw.tolist()) | set(gallery_raw.tolist())))
    }
    query_labels = tuple(authority[value] for value in query_raw.tolist())
    gallery_labels = tuple(authority[value] for value in gallery_raw.tolist())
    return fit, fit_labels, query, query_labels, gallery, gallery_labels, False


def _evaluate_dataset(name: str, loaded: tuple[Any, ...]) -> dict[str, Any]:
    fit, fit_labels, query, query_labels, gallery, gallery_labels, same_rows = loaded
    device = torch.device("cuda")
    started = time.monotonic()

    pca128_weight, pca128_bias = fit_pca_affine(fit, output_dimensions=128)
    pca128_query = F.normalize(F.linear(query, pca128_weight, pca128_bias), dim=1)
    pca128_gallery = F.normalize(F.linear(gallery, pca128_weight, pca128_bias), dim=1)
    pca128_query = pack_int8_unit_embeddings(pca128_query)
    pca128_gallery = pack_int8_unit_embeddings(pca128_gallery)

    pca256_weight, pca256_bias = fit_pca_affine(fit, output_dimensions=256)
    pca256_fit = F.normalize(F.linear(fit, pca256_weight, pca256_bias), dim=1)
    pca256_query = F.normalize(F.linear(query, pca256_weight, pca256_bias), dim=1)
    pca256_gallery = F.normalize(F.linear(gallery, pca256_weight, pca256_bias), dim=1)
    pca256_query, _ = fit_pack_decode_int4(pca256_fit, pca256_query)
    pca256_gallery, _ = fit_pack_decode_int4(pca256_fit, pca256_gallery)

    evaluation = query if same_rows else torch.cat((query, gallery))
    learned128_fit, learned128, training128 = train_width(
        fit, evaluation, fit_labels, width=128
    )
    learned256_fit, learned256, training256 = train_width(
        fit, evaluation, fit_labels, width=256
    )
    if same_rows:
        learned128_query = learned128_gallery = learned128
        learned256_query = learned256_gallery = learned256
    else:
        learned128_query, learned128_gallery = learned128.split((len(query), len(gallery)))
        learned256_query, learned256_gallery = learned256.split((len(query), len(gallery)))
    learned128_query = pack_int8_unit_embeddings(learned128_query)
    learned128_gallery = pack_int8_unit_embeddings(learned128_gallery)
    learned256_query, _ = fit_pack_decode_int4(learned256_fit, learned256_query)
    learned256_gallery, _ = fit_pack_decode_int4(learned256_fit, learned256_gallery)

    vectors = {
        "pca128_int8": (pca128_query, pca128_gallery),
        "pca256_int4": (pca256_query, pca256_gallery),
        "learned128_int8": (learned128_query, learned128_gallery),
        "learned256_int4": (learned256_query, learned256_gallery),
    }
    arms = {
        arm: _score(
            values[0],
            values[1],
            query_labels,
            gallery_labels,
            same_rows=same_rows,
            device=device,
        )
        for arm, values in vectors.items()
    }
    baseline_ap = np.asarray(arms["pca128_int8"]["per_query_ap"], dtype=np.float64)
    contrasts = {
        arm: paired_query_bootstrap(
            np.asarray(arms[arm]["per_query_ap"], dtype=np.float64),
            baseline_ap,
            seed=BOOTSTRAP_SEED,
            replicates=BOOTSTRAP_REPLICATES,
        )
        for arm in ARM_NAMES[1:]
    }
    contrasts["learned256_int4_minus_learned128_int8"] = paired_query_bootstrap(
        np.asarray(arms["learned256_int4"]["per_query_ap"], dtype=np.float64),
        np.asarray(arms["learned128_int8"]["per_query_ap"], dtype=np.float64),
        seed=BOOTSTRAP_SEED,
        replicates=BOOTSTRAP_REPLICATES,
    )
    return {
        "dataset": name,
        "split": {
            "fit_rows": len(fit),
            "query_rows": len(query),
            "gallery_rows": len(gallery),
            "same_rows": same_rows,
        },
        "arms": arms,
        "representation": {
            "pca128_int8": {
                "persistent_bytes_per_item": pca128_query.bytes_per_vector,
                "dimensions": 128,
            },
            "pca256_int4": {
                "persistent_bytes_per_item": 128,
                "shared_scale_bytes": 1024,
                "dimensions": 256,
            },
            "learned128_int8": {
                "persistent_bytes_per_item": learned128_query.bytes_per_vector,
                "dimensions": 128,
            },
            "learned256_int4": {
                "persistent_bytes_per_item": 128,
                "shared_scale_bytes": 1024,
                "dimensions": 256,
            },
        },
        "contrasts_against_pca128_int8": {
            arm: contrasts[arm] for arm in ARM_NAMES[1:]
        },
        "learned_width_contrast": contrasts[
            "learned256_int4_minus_learned128_int8"
        ],
        "training": {"learned128_int8": training128, "learned256_int4": training256},
        "elapsed_seconds": time.monotonic() - started,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    for name in ("cars", "cub", "sop", "inshop", "food101"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument("--preregistration-sha256", required=True)
    parser.add_argument("--script-sha256", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-frozen-ladder", action="store_true", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    if _sha256(Path(__file__)) != args.script_sha256:
        raise ValueError("ladder script authority differs")
    if _sha256(args.preregistration) != args.preregistration_sha256:
        raise ValueError("ladder preregistration authority differs")
    preregistration = json.loads(args.preregistration.read_text())
    if (
        preregistration.get("schema")
        != "scratch-same-teacher-four-arm-five-dataset-ladder-preregistration-v1"
        or preregistration.get("script_sha256") != args.script_sha256
        or preregistration.get("source_commit") != args.source_commit
        or preregistration.get("bootstrap")
        != {
            "estimand": "paired per-query AP@R difference",
            "interval": "two-sided percentile 95%",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        }
    ):
        raise ValueError("ladder preregistration contract differs")
    for name in EXPECTED_SHA256:
        if _sha256(getattr(args, name)) != EXPECTED_SHA256[name]:
            raise ValueError(f"{name} artifact authority differs")
    configure_deterministic_similarity_runtime(0, cpu_threads=2)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    loaders = {
        "cars": _class_disjoint(args.cars),
        "cub": _class_disjoint(args.cub),
        "sop": _sop(args.sop),
        "inshop": _inshop(args.inshop),
        "food101": _food(args.food101),
    }
    datasets = {name: _evaluate_dataset(name, value) for name, value in loaders.items()}
    panel_decision = {}
    for arm in ARM_NAMES[1:]:
        failed = [
            name
            for name, row in datasets.items()
            if not row["contrasts_against_pca128_int8"][arm]["survives"]
        ]
        panel_decision[arm] = {
            "survives_all_datasets": not failed,
            "failed_datasets": failed,
        }
    result = {
        "schema": "scratch-same-teacher-four-arm-five-dataset-ladder-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "inputs": EXPECTED_SHA256,
        "bootstrap": {
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "minimum_map_effect": MINIMUM_MAP_EFFECT,
        },
        "datasets": datasets,
        "panel_decision": panel_decision,
    }
    wire = json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    partial = args.output.with_suffix(args.output.suffix + ".partial")
    partial.write_text(wire)
    os.replace(partial, args.output)
    print(wire, end="")


if __name__ == "__main__":
    main()
