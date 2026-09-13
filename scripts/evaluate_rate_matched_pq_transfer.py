#!/usr/bin/env python3
"""Evaluate fixed-rate representation-plus-OPQ transfer on local embeddings."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import torch

import sfora as sfora_package
from sfora.product_quantization import (
    balanced_product_quantization_spec,
    fit_optimized_product_quantizer,
)
from sfora.rate_matched_product_quantization import fit_rate_matched_product_quantizer


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _lower_hex(value: str, width: int) -> str:
    if len(value) != width or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError(f"value must be {width} lowercase hexadecimal characters")
    return value


def parse_arguments(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the strict, explicit experiment-only command line."""

    values = list(arguments) if arguments is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--archive-sha256", required=True, type=lambda value: _lower_hex(value, 64))
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--source-commit", required=True, type=lambda value: _lower_hex(value, 40))
    parser.add_argument("--device", required=True, choices=("cpu", "cuda"))
    parser.add_argument("--encode-batch-size", required=True, type=_positive_integer)
    parser.add_argument("--query-batch-size", required=True, type=_positive_integer)
    parser.add_argument("--bootstrap-samples", required=True, type=_positive_integer)
    parser.add_argument("--execute-transfer-evaluation", required=True, action="store_true")
    flags = [value.split("=", 1)[0] for value in values if value.startswith("--")]
    duplicates = [flag for flag, count in Counter(flags).items() if count > 1]
    if duplicates:
        parser.error(f"duplicate argument: {sorted(duplicates)[0]}")
    parsed = parser.parse_args(values)
    if not parsed.input_uri:
        parser.error("--input-uri must be nonempty")
    return parsed


class AdcRetrievalScore(TypedDict):
    """Aggregate and paired per-query retrieval evidence."""

    map_at_r: float
    per_query_ap: tuple[float, ...]
    per_query_r1: tuple[float, ...]
    r1: float


class FloatRetrievalScore(AdcRetrievalScore):
    """Float retrieval evidence plus retained exact-neighbor ordinals."""

    neighbor_ordinals: tuple[tuple[int, ...], ...]


@dataclass(frozen=True, slots=True)
class TransferArchive:
    """Authenticated normalized train/test embedding archive."""

    archive_keys: tuple[str, ...]
    sha256: str
    test_embeddings: torch.Tensor
    test_labels: tuple[int, ...]
    train_embeddings: torch.Tensor
    train_labels: tuple[int, ...]


def normalize_embedding_rows(values: torch.Tensor) -> torch.Tensor:
    """Validate and L2-normalize one float32 embedding matrix."""

    if (
        type(values) is not torch.Tensor
        or values.dtype != torch.float32
        or values.ndim != 2
        or values.shape[0] < 1
        or values.shape[1] < 2
        or not bool(torch.isfinite(values).all())
    ):
        raise ValueError("embedding row authority differs")
    norms = torch.linalg.vector_norm(values.double(), dim=1, keepdim=True)
    if not bool(torch.isfinite(norms).all()) or bool((norms <= 1e-12).any()):
        raise ValueError("embedding row authority differs")
    normalized = (values.double() / norms).float().contiguous()
    output_norms = torch.linalg.vector_norm(normalized.double(), dim=1)
    if bool((torch.abs(output_norms - 1.0) > 2e-6).any()):
        raise ValueError("embedding row authority differs")
    return cast(torch.Tensor, normalized)


def load_transfer_archive(path: Path, *, sha256: str) -> TransferArchive:
    """Authenticate and load a generic NPZ embedding-pair archive."""

    if (
        not isinstance(path, Path)
        or type(sha256) is not str
        or len(sha256) != 64
        or any(character not in "0123456789abcdef" for character in sha256)
        or not path.is_file()
    ):
        raise ValueError("transfer archive authority differs")
    snapshot = path.read_bytes()
    if hashlib.sha256(snapshot).hexdigest() != sha256:
        raise ValueError("transfer archive authority differs")
    required = {
        "train_embeddings",
        "train_labels",
        "test_embeddings",
        "test_labels",
    }
    try:
        with np.load(io.BytesIO(snapshot), allow_pickle=False) as payload:
            keys = tuple(sorted(payload.files))
            if not required.issubset(keys):
                raise ValueError("transfer archive authority differs")
            train_array = np.asarray(payload["train_embeddings"])
            test_array = np.asarray(payload["test_embeddings"])
            train_labels_array = np.asarray(payload["train_labels"])
            test_labels_array = np.asarray(payload["test_labels"])
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise ValueError("transfer archive authority differs") from error
    if (
        train_array.dtype != np.float32
        or test_array.dtype != np.float32
        or train_array.ndim != 2
        or test_array.ndim != 2
        or train_array.shape[0] < 2
        or test_array.shape[0] < 2
        or train_array.shape[1] < 2
        or test_array.shape[1] != train_array.shape[1]
        or train_labels_array.dtype != np.int64
        or test_labels_array.dtype != np.int64
        or train_labels_array.shape != (train_array.shape[0],)
        or test_labels_array.shape != (test_array.shape[0],)
    ):
        raise ValueError("transfer archive authority differs")
    train_labels = tuple(int(value) for value in train_labels_array.tolist())
    test_labels = tuple(int(value) for value in test_labels_array.tolist())
    if (
        len(set(train_labels)) < 2
        or any(count < 2 for count in Counter(test_labels).values())
        or len(set(test_labels)) < 2
    ):
        raise ValueError("transfer archive authority differs")
    try:
        train = normalize_embedding_rows(torch.from_numpy(train_array.copy()))
        test = normalize_embedding_rows(torch.from_numpy(test_array.copy()))
    except ValueError as error:
        raise ValueError("transfer archive authority differs") from error
    return TransferArchive(
        archive_keys=keys,
        sha256=sha256,
        test_embeddings=test,
        test_labels=test_labels,
        train_embeddings=train,
        train_labels=train_labels,
    )


def encode_in_batches(
    values: torch.Tensor,
    *,
    encoder: Callable[[torch.Tensor], torch.Tensor],
    batch_size: int,
) -> torch.Tensor:
    """Encode consecutive bounded batches into one uint8 code matrix."""

    if (
        type(values) is not torch.Tensor
        or values.ndim != 2
        or values.shape[0] < 1
        or type(batch_size) is not int
        or batch_size < 1
        or not callable(encoder)
    ):
        raise ValueError("batched encoding authority differs")
    batches: list[torch.Tensor] = []
    width: int | None = None
    for start in range(0, len(values), batch_size):
        batch = encoder(values[start : start + batch_size])
        if (
            type(batch) is not torch.Tensor
            or batch.dtype != torch.uint8
            or batch.ndim != 2
            or batch.shape[0] != min(batch_size, len(values) - start)
            or batch.shape[1] < 1
            or not batch.is_contiguous()
            or (width is not None and batch.shape[1] != width)
        ):
            raise ValueError("batched encoding authority differs")
        width = batch.shape[1]
        batches.append(batch)
    return torch.cat(batches, dim=0).contiguous()


def _lowest_distance_candidates(distances: torch.Tensor, width: int) -> torch.Tensor:
    retained = min(width + 1, distances.shape[1])
    values, indexes = torch.topk(distances, k=retained, dim=1, largest=False, sorted=False)
    ordinal_order = torch.argsort(indexes, dim=1, stable=True)
    indexes = indexes.gather(1, ordinal_order)
    values = values.gather(1, ordinal_order)
    distance_order = torch.argsort(values, dim=1, stable=True)
    indexes = indexes.gather(1, distance_order)
    values = values.gather(1, distance_order)
    if retained > width:
        for row in torch.nonzero(values[:, width - 1] == values[:, width], as_tuple=False):
            row_index = int(row.item())
            boundary = values[row_index, width - 1]
            candidates = torch.nonzero(distances[row_index] <= boundary, as_tuple=False).flatten()
            candidates = torch.sort(candidates).values
            candidate_values = distances[row_index, candidates]
            order = torch.argsort(candidate_values, stable=True)
            indexes[row_index, :width] = candidates[order[:width]]
    return indexes[:, :width]


def score_adc_retrieval(
    queries: torch.Tensor,
    gallery_codes: torch.Tensor,
    labels: tuple[int, ...],
    *,
    distance: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    batch_size: int,
) -> AdcRetrievalScore:
    """Evaluate leave-self-out MAP@R and Recall@1 from bounded ADC batches."""

    if (
        type(queries) is not torch.Tensor
        or queries.dtype != torch.float32
        or queries.ndim != 2
        or type(gallery_codes) is not torch.Tensor
        or gallery_codes.dtype != torch.uint8
        or gallery_codes.ndim != 2
        or len(queries) != len(gallery_codes)
        or type(labels) is not tuple
        or len(labels) != len(queries)
        or any(type(label) is not int for label in labels)
        or type(batch_size) is not int
        or batch_size < 1
        or not callable(distance)
    ):
        raise ValueError("ADC retrieval authority differs")
    counts = Counter(labels)
    if len(counts) < 2 or any(count < 2 for count in counts.values()):
        raise ValueError("ADC retrieval authority differs")
    width = max(counts.values()) - 1
    per_query_ap: list[float] = []
    per_query_r1: list[float] = []
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        distances = distance(queries[start:stop], gallery_codes)
        if (
            type(distances) is not torch.Tensor
            or distances.dtype != torch.float32
            or distances.shape != (stop - start, len(gallery_codes))
            or distances.device != queries.device
            or not bool(torch.isfinite(distances).all())
        ):
            raise ValueError("ADC retrieval authority differs")
        distances = distances.clone()
        distances[
            torch.arange(stop - start, device=distances.device),
            torch.arange(start, stop, device=distances.device),
        ] = torch.inf
        rankings = _lowest_distance_candidates(distances, width).cpu().tolist()
        for ranking, label in zip(rankings, labels[start:stop], strict=True):
            positive_count = counts[label] - 1
            found = 0
            terms: list[float] = []
            for rank, index in enumerate(ranking[:positive_count], start=1):
                if labels[index] == label:
                    found += 1
                    terms.append(found / rank)
            per_query_ap.append(math.fsum(terms) / positive_count)
            per_query_r1.append(float(labels[ranking[0]] == label))
    return AdcRetrievalScore(
        map_at_r=math.fsum(per_query_ap) / len(per_query_ap),
        per_query_ap=tuple(per_query_ap),
        per_query_r1=tuple(per_query_r1),
        r1=math.fsum(per_query_r1) / len(per_query_r1),
    )


def score_float_retrieval(
    queries: torch.Tensor,
    gallery: torch.Tensor,
    labels: tuple[int, ...],
    *,
    batch_size: int,
    neighbor_width: int,
) -> FloatRetrievalScore:
    """Evaluate bounded exact squared-L2 retrieval with stable ordinal ties."""

    if (
        type(queries) is not torch.Tensor
        or queries.dtype != torch.float32
        or queries.ndim != 2
        or type(gallery) is not torch.Tensor
        or gallery.dtype != torch.float32
        or gallery.ndim != 2
        or gallery.shape != queries.shape
        or queries.device != gallery.device
        or not bool(torch.isfinite(queries).all())
        or not bool(torch.isfinite(gallery).all())
        or type(labels) is not tuple
        or len(labels) != len(queries)
        or any(type(label) is not int for label in labels)
        or type(batch_size) is not int
        or batch_size < 1
        or type(neighbor_width) is not int
        or neighbor_width < 1
        or neighbor_width >= len(gallery)
    ):
        raise ValueError("float retrieval authority differs")
    counts = Counter(labels)
    if len(counts) < 2 or any(count < 2 for count in counts.values()):
        raise ValueError("float retrieval authority differs")
    retained = max(neighbor_width, max(counts.values()) - 1)
    per_query_ap: list[float] = []
    per_query_r1: list[float] = []
    neighbor_ordinals: list[tuple[int, ...]] = []
    for start in range(0, len(queries), batch_size):
        stop = min(start + batch_size, len(queries))
        batch = queries[start:stop]
        distances = (batch[:, None, :] - gallery[None, :, :]).square().sum(dim=2)
        distances[
            torch.arange(stop - start, device=distances.device),
            torch.arange(start, stop, device=distances.device),
        ] = torch.inf
        rankings = _lowest_distance_candidates(distances, retained).cpu().tolist()
        for ranking, label in zip(rankings, labels[start:stop], strict=True):
            positive_count = counts[label] - 1
            found = 0
            terms: list[float] = []
            for rank, index in enumerate(ranking[:positive_count], start=1):
                if labels[index] == label:
                    found += 1
                    terms.append(found / rank)
            per_query_ap.append(math.fsum(terms) / positive_count)
            per_query_r1.append(float(labels[ranking[0]] == label))
            neighbor_ordinals.append(tuple(ranking[:neighbor_width]))
    return FloatRetrievalScore(
        map_at_r=math.fsum(per_query_ap) / len(per_query_ap),
        neighbor_ordinals=tuple(neighbor_ordinals),
        per_query_ap=tuple(per_query_ap),
        per_query_r1=tuple(per_query_r1),
        r1=math.fsum(per_query_r1) / len(per_query_r1),
    )


def paired_class_bootstrap_interval(
    candidate: Sequence[float],
    baseline: Sequence[float],
    labels: tuple[int, ...],
    *,
    seed: int,
    samples: int,
) -> tuple[float, float]:
    """Return a deterministic 95% paired bootstrap interval over complete classes."""

    if (
        len(candidate) != len(baseline)
        or len(candidate) != len(labels)
        or len(candidate) < 2
        or any(type(value) is not float or not math.isfinite(value) for value in candidate)
        or any(type(value) is not float or not math.isfinite(value) for value in baseline)
        or any(type(label) is not int for label in labels)
        or type(seed) is not int
        or seed < 0
        or type(samples) is not int
        or samples < 1
    ):
        raise ValueError("paired bootstrap authority differs")
    classes = tuple(sorted(set(labels)))
    if len(classes) < 2:
        raise ValueError("paired bootstrap authority differs")
    deltas = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    sums = np.asarray([deltas[label_array == label].sum() for label in classes])
    counts = np.asarray([(label_array == label).sum() for label in classes], dtype=np.float64)
    generator = np.random.default_rng(seed)
    replicates = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 128):
        stop = min(start + 128, samples)
        selected = generator.integers(0, len(classes), size=(stop - start, len(classes)))
        replicates[start:stop] = sums[selected].sum(axis=1) / counts[selected].sum(axis=1)
    lower, upper = np.quantile(replicates, (0.025, 0.975), method="linear")
    return float(lower), float(upper)


def _exact_keys(value: object, keys: set[str]) -> bool:
    return type(value) is dict and set(value) == keys


def _finite_float(value: object, *, positive: bool = False) -> bool:
    return type(value) is float and math.isfinite(value) and (not positive or value > 0.0)


def _valid_metric_rows(value: object, count: int, *, binary: bool = False) -> bool:
    return (
        type(value) is list
        and len(value) == count
        and all(
            _finite_float(item) and 0.0 <= item <= 1.0 and (not binary or item in (0.0, 1.0))
            for item in value
        )
    )


def _balanced_widths(dimensions: int, blocks: int) -> list[int]:
    width, wider = divmod(dimensions, blocks)
    return [width] * (blocks - wider) + [width + 1] * wider


def validate_transfer_result(value: object, *, labels: tuple[int, ...]) -> None:
    """Recompute and validate one complete paired transfer receipt."""

    try:
        if (
            type(labels) is not tuple
            or len(labels) < 2
            or any(type(label) is not int for label in labels)
            or not _exact_keys(
                value,
                {
                    "arms",
                    "bootstrap",
                    "claim_eligible",
                    "config",
                    "contrasts",
                    "execution",
                    "input",
                    "provenance",
                    "schema",
                    "timing_scope",
                },
            )
        ):
            raise ValueError
        root = cast(dict[str, Any], value)
        if (
            root["schema"] != "sfora-rate-matched-pq-transfer-v1"
            or root["claim_eligible"] is not False
            or root["timing_scope"] != "full-corpus diagnostic evaluation; not serving latency"
        ):
            raise ValueError
        config = root["config"]
        if config != {
            "bytes_per_vector": 24,
            "codebook_size": 256,
            "dimensions": 80,
            "maximum_iterations": 20,
            "rotation_iterations": 4,
            "seed": 50,
        }:
            raise ValueError
        bootstrap = root["bootstrap"]
        if (
            not _exact_keys(bootstrap, {"samples", "seed"})
            or type(bootstrap["seed"]) is not int
            or bootstrap["seed"] < 0
            or type(bootstrap["samples"]) is not int
            or bootstrap["samples"] < 1
        ):
            raise ValueError
        execution = root["execution"]
        if (
            not _exact_keys(execution, {"encode_batch_size", "query_batch_size"})
            or type(execution["encode_batch_size"]) is not int
            or execution["encode_batch_size"] < 1
            or type(execution["query_batch_size"]) is not int
            or execution["query_batch_size"] < 1
        ):
            raise ValueError
        input_value = root["input"]
        required_archive_keys = {
            "test_embeddings",
            "test_labels",
            "train_embeddings",
            "train_labels",
        }
        if (
            not _exact_keys(
                input_value,
                {"archive_keys", "dimensions", "sha256", "test_rows", "train_rows"},
            )
            or type(input_value["archive_keys"]) is not list
            or input_value["archive_keys"] != sorted(set(input_value["archive_keys"]))
            or any(type(key) is not str or not key for key in input_value["archive_keys"])
            or not required_archive_keys.issubset(input_value["archive_keys"])
            or type(input_value["sha256"]) is not str
            or len(input_value["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in input_value["sha256"])
            or type(input_value["dimensions"]) is not int
            or input_value["dimensions"] < 80
            or type(input_value["train_rows"]) is not int
            or input_value["train_rows"] < 256
            or input_value["test_rows"] != len(labels)
        ):
            raise ValueError
        provenance = root["provenance"]
        if (
            not _exact_keys(
                provenance,
                {
                    "cuda_version",
                    "device",
                    "device_name",
                    "evaluator_sha256",
                    "input_uri",
                    "numpy_version",
                    "scikit_learn_version",
                    "source_commit",
                    "torch_version",
                },
            )
            or provenance["device"] not in ("cpu", "cuda")
            or any(
                type(provenance[key]) is not str or not provenance[key]
                for key in (
                    "cuda_version",
                    "device_name",
                    "numpy_version",
                    "scikit_learn_version",
                    "torch_version",
                )
            )
            or type(provenance["input_uri"]) is not str
            or not provenance["input_uri"]
            or type(provenance["source_commit"]) is not str
            or len(provenance["source_commit"]) != 40
            or any(character not in "0123456789abcdef" for character in provenance["source_commit"])
            or type(provenance["evaluator_sha256"]) is not str
            or len(provenance["evaluator_sha256"]) != 64
            or any(
                character not in "0123456789abcdef" for character in provenance["evaluator_sha256"]
            )
            or type(provenance["torch_version"]) is not str
            or not provenance["torch_version"]
        ):
            raise ValueError
        arms = root["arms"]
        if type(arms) is not list or len(arms) != 3:
            raise ValueError
        expected_names = ("pca80-opq24", "opq24", "opq32")
        expected_widths = (24, 24, 32)
        by_name: dict[str, dict[str, Any]] = {}
        arm_keys = {
            "block_dimensions",
            "code_bytes_per_vector",
            "evaluation_seconds",
            "fit_seconds",
            "map_at_r",
            "name",
            "parameter_bytes",
            "per_query_ap",
            "per_query_r1",
            "r1",
        }
        for arm, name, width in zip(arms, expected_names, expected_widths, strict=True):
            represented_dimensions = 80 if name == "pca80-opq24" else input_value["dimensions"]
            expected_parameter_bytes = (
                4 * (81 * input_value["dimensions"] + 26880)
                if name == "pca80-opq24"
                else 4
                * (
                    input_value["dimensions"] * input_value["dimensions"]
                    + 256 * input_value["dimensions"]
                )
            )
            if (
                not _exact_keys(arm, arm_keys)
                or arm["name"] != name
                or type(arm["code_bytes_per_vector"]) is not int
                or arm["code_bytes_per_vector"] != width
                or arm["block_dimensions"] != _balanced_widths(represented_dimensions, width)
                or type(arm["parameter_bytes"]) is not int
                or arm["parameter_bytes"] != expected_parameter_bytes
                or not _finite_float(arm["fit_seconds"], positive=True)
                or not _finite_float(arm["evaluation_seconds"], positive=True)
                or not _valid_metric_rows(arm["per_query_ap"], len(labels))
                or not _valid_metric_rows(arm["per_query_r1"], len(labels), binary=True)
                or not _finite_float(arm["map_at_r"])
                or not _finite_float(arm["r1"])
            ):
                raise ValueError
            map_at_r = math.fsum(arm["per_query_ap"]) / len(labels)
            r1 = math.fsum(arm["per_query_r1"]) / len(labels)
            if arm["map_at_r"] != map_at_r or arm["r1"] != r1:
                raise ValueError
            if any(
                Counter(labels)[label] == 2 and ap != r1_value
                for ap, r1_value, label in zip(
                    arm["per_query_ap"], arm["per_query_r1"], labels, strict=True
                )
            ):
                raise ValueError
            by_name[name] = arm
        contrasts = root["contrasts"]
        if type(contrasts) is not list or len(contrasts) != 2:
            raise ValueError
        candidate = by_name["pca80-opq24"]
        contrast_keys = {
            "baseline",
            "map_at_r_delta",
            "map_at_r_interval_95",
            "r1_delta",
            "r1_interval_95",
        }
        for contrast, baseline_name in zip(contrasts, ("opq24", "opq32"), strict=True):
            baseline = by_name[baseline_name]
            map_interval = paired_class_bootstrap_interval(
                candidate["per_query_ap"],
                baseline["per_query_ap"],
                labels,
                seed=bootstrap["seed"],
                samples=bootstrap["samples"],
            )
            r1_interval = paired_class_bootstrap_interval(
                candidate["per_query_r1"],
                baseline["per_query_r1"],
                labels,
                seed=bootstrap["seed"],
                samples=bootstrap["samples"],
            )
            if (
                not _exact_keys(contrast, contrast_keys)
                or contrast["baseline"] != baseline_name
                or not _finite_float(contrast["map_at_r_delta"])
                or not _finite_float(contrast["r1_delta"])
                or type(contrast["map_at_r_interval_95"]) is not list
                or len(contrast["map_at_r_interval_95"]) != 2
                or any(not _finite_float(bound) for bound in contrast["map_at_r_interval_95"])
                or type(contrast["r1_interval_95"]) is not list
                or len(contrast["r1_interval_95"]) != 2
                or any(not _finite_float(bound) for bound in contrast["r1_interval_95"])
                or contrast["map_at_r_delta"] != candidate["map_at_r"] - baseline["map_at_r"]
                or contrast["r1_delta"] != candidate["r1"] - baseline["r1"]
                or contrast["map_at_r_interval_95"] != list(map_interval)
                or contrast["r1_interval_95"] != list(r1_interval)
            ):
                raise ValueError
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("transfer result authority differs") from error


def canonical_transfer_result_bytes(value: object, *, labels: tuple[int, ...]) -> bytes:
    """Return validated sorted compact JSON with exactly one trailing newline."""

    validate_transfer_result(value, labels=labels)
    try:
        return (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError("transfer result authority differs") from error


def _parameter_bytes(module: torch.nn.Module) -> int:
    return sum(value.numel() * value.element_size() for value in module.state_dict().values())


def evaluate_transfer(
    archive: TransferArchive,
    *,
    source_commit: str,
    input_uri: str,
    device: torch.device,
    encode_batch_size: int,
    query_batch_size: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    """Fit and score the frozen rate-matched transfer comparison."""

    if (
        type(archive) is not TransferArchive
        or type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or type(input_uri) is not str
        or not input_uri
        or type(device) is not torch.device
        or device.type not in ("cpu", "cuda")
        or (device.type == "cuda" and not torch.cuda.is_available())
        or type(encode_batch_size) is not int
        or encode_batch_size < 1
        or type(query_batch_size) is not int
        or query_batch_size < 1
        or type(bootstrap_samples) is not int
        or bootstrap_samples < 1
        or archive.train_embeddings.shape[1] < 80
    ):
        raise ValueError("transfer evaluation authority differs")
    scientific_config = {
        "bytes_per_vector": 24,
        "codebook_size": 256,
        "dimensions": 80,
        "maximum_iterations": 20,
        "rotation_iterations": 4,
        "seed": 50,
    }
    rate_spec = balanced_product_quantization_spec(
        dimensions=80, bytes_per_vector=24, codebook_size=256
    )
    ordinary_specs = {
        width: balanced_product_quantization_spec(
            dimensions=archive.train_embeddings.shape[1],
            bytes_per_vector=width,
            codebook_size=256,
        )
        for width in (24, 32)
    }
    arms: list[dict[str, Any]] = []
    for name in ("pca80-opq24", "opq24", "opq32"):
        fit_started = time.perf_counter_ns()
        codec: Any
        encoder: Callable[[torch.Tensor], torch.Tensor]
        if name == "pca80-opq24":
            codec = fit_rate_matched_product_quantizer(
                archive.train_embeddings,
                rate_spec,
                seed=50,
                maximum_iterations=20,
                rotation_iterations=4,
            )
            encoder = codec.encode
        else:
            width = 24 if name == "opq24" else 32
            codec = fit_optimized_product_quantizer(
                archive.train_embeddings,
                ordinary_specs[width],
                seed=50,
                maximum_iterations=20,
                rotation_iterations=4,
            )
            encoder = codec.hard_encode
        fit_seconds = (time.perf_counter_ns() - fit_started) / 1_000_000_000.0
        parameter_bytes = _parameter_bytes(codec)
        codec = codec.to(device)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_started = time.perf_counter_ns()

        def encode_batch(
            batch: torch.Tensor, encode: Callable[[torch.Tensor], torch.Tensor] = encoder
        ) -> torch.Tensor:
            return encode(batch.to(device)).cpu()

        with torch.inference_mode():
            codes = encode_in_batches(
                archive.test_embeddings,
                encoder=encode_batch,
                batch_size=encode_batch_size,
            ).to(device)
            queries = archive.test_embeddings.to(device)
            if name == "pca80-opq24":
                distance = codec.score_codes
            else:
                distance = codec.asymmetric_squared_distances
            score = score_adc_retrieval(
                queries,
                codes,
                archive.test_labels,
                distance=distance,
                batch_size=query_batch_size,
            )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        evaluation_seconds = (time.perf_counter_ns() - evaluation_started) / 1_000_000_000.0
        arms.append(
            {
                "code_bytes_per_vector": 24 if name != "opq32" else 32,
                "block_dimensions": list(codec.spec.block_dimensions),
                "evaluation_seconds": evaluation_seconds,
                "fit_seconds": fit_seconds,
                "map_at_r": score["map_at_r"],
                "name": name,
                "parameter_bytes": parameter_bytes,
                "per_query_ap": list(score["per_query_ap"]),
                "per_query_r1": list(score["per_query_r1"]),
                "r1": score["r1"],
            }
        )
        del codec, codes, queries
        if device.type == "cuda":
            torch.cuda.empty_cache()
    candidate = arms[0]
    contrasts: list[dict[str, Any]] = []
    for baseline in arms[1:]:
        contrasts.append(
            {
                "baseline": baseline["name"],
                "map_at_r_delta": candidate["map_at_r"] - baseline["map_at_r"],
                "map_at_r_interval_95": list(
                    paired_class_bootstrap_interval(
                        candidate["per_query_ap"],
                        baseline["per_query_ap"],
                        archive.test_labels,
                        seed=50,
                        samples=bootstrap_samples,
                    )
                ),
                "r1_delta": candidate["r1"] - baseline["r1"],
                "r1_interval_95": list(
                    paired_class_bootstrap_interval(
                        candidate["per_query_r1"],
                        baseline["per_query_r1"],
                        archive.test_labels,
                        seed=50,
                        samples=bootstrap_samples,
                    )
                ),
            }
        )
    result: dict[str, Any] = {
        "arms": arms,
        "bootstrap": {"samples": bootstrap_samples, "seed": 50},
        "claim_eligible": False,
        "config": scientific_config,
        "contrasts": contrasts,
        "execution": {
            "encode_batch_size": encode_batch_size,
            "query_batch_size": query_batch_size,
        },
        "input": {
            "archive_keys": list(archive.archive_keys),
            "dimensions": archive.train_embeddings.shape[1],
            "sha256": archive.sha256,
            "test_rows": len(archive.test_embeddings),
            "train_rows": len(archive.train_embeddings),
        },
        "provenance": {
            "cuda_version": str(torch.version.cuda) if torch.version.cuda is not None else "none",
            "device": device.type,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
            "evaluator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "input_uri": input_uri,
            "numpy_version": str(np.__version__),
            "scikit_learn_version": version("scikit-learn"),
            "source_commit": source_commit,
            "torch_version": str(torch.__version__),
        },
        "schema": "sfora-rate-matched-pq-transfer-v1",
        "timing_scope": "full-corpus diagnostic evaluation; not serving latency",
    }
    validate_transfer_result(result, labels=archive.test_labels)
    return result


@dataclass(slots=True)
class _OutputReservation:
    path: Path
    partial: Path
    descriptor: int | None

    def abort(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
        if self.partial.exists():
            self.partial.unlink()

    def publish(self, payload: bytes) -> None:
        if self.descriptor is None:
            raise RuntimeError("transfer output reservation is closed")
        descriptor = self.descriptor
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written < 1:
                    raise OSError("transfer result write made no progress")
                view = view[written:]
            os.fsync(descriptor)
            os.close(descriptor)
            self.descriptor = None
            os.link(self.partial, self.path)
            self.partial.unlink()
        except BaseException:
            self.abort()
            raise


def _reserve_output(path: Path) -> _OutputReservation:
    partial = path.with_name(f".{path.name}.partial")
    if path.exists():
        raise FileExistsError(path)
    try:
        descriptor = os.open(partial, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise FileExistsError(path) from error
    return _OutputReservation(path=path, partial=partial, descriptor=descriptor)


def _verify_source_authority(source_commit: str) -> str:
    repository = Path(__file__).resolve().parents[1]
    package_path = Path(sfora_package.__file__).resolve()
    try:
        head = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("transfer source authority differs") from error
    if (
        head != source_commit
        or status
        or not package_path.is_relative_to(repository / "src" / "sfora")
    ):
        raise ValueError("transfer source authority differs")
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def main(arguments: Sequence[str] | None = None) -> int:
    """Run one explicit fixed transfer evaluation and publish its receipt."""

    parsed = parse_arguments(arguments)
    evaluator_sha256 = _verify_source_authority(parsed.source_commit)
    reservation = _reserve_output(parsed.output)
    try:
        archive = load_transfer_archive(parsed.archive, sha256=parsed.archive_sha256)
        result = evaluate_transfer(
            archive,
            source_commit=parsed.source_commit,
            input_uri=parsed.input_uri,
            device=torch.device(parsed.device),
            encode_batch_size=parsed.encode_batch_size,
            query_batch_size=parsed.query_batch_size,
            bootstrap_samples=parsed.bootstrap_samples,
        )
        payload = canonical_transfer_result_bytes(result, labels=archive.test_labels)
        if (
            result["provenance"]["evaluator_sha256"] != evaluator_sha256
            or _verify_source_authority(parsed.source_commit) != evaluator_sha256
        ):
            raise ValueError("transfer source authority differs")
        reservation.publish(payload)
    except BaseException:
        reservation.abort()
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileExistsError, OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1) from error
