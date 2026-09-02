#!/usr/bin/env python3
"""Authenticate and execute the local weight-space transfer diagnostic."""

from __future__ import annotations

import hashlib
import io
import math
import random
import stat
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from scripts.run_siglip_proxy_control import (
    ControlRunAuthority,
    SiglipProxyControlConfig,
    _config_sha256,
    _run_authority_sha256,
    read_control_seed_receipt,
)


@dataclass(frozen=True)
class EndpointMetrics:
    """One normalized burned-band endpoint row."""

    correct: int
    queries: int
    recall_at_1: float
    mean_nearest_positive_cosine: float
    mean_nearest_negative_cosine: float
    mean_margin: float


@dataclass(frozen=True)
class SeedEndpointAuthority:
    """Outcome-minimized seed receipt authority released to the child."""

    seed: int
    initial_state_sha256: str
    initial_raw: EndpointMetrics
    initial_projected: EndpointMetrics
    trained_raw: EndpointMetrics
    trained_projected: EndpointMetrics
    checkpoint_basename: str
    checkpoint_sha256: str
    checkpoint_bytes: int
    config_sha256: str
    run_authority_sha256: str


@dataclass(frozen=True)
class LoadedTransferCheckpoint:
    """Authenticated model-only checkpoint projection."""

    seed: int
    initial_snapshot_sha256: str
    model_state: Mapping[str, torch.Tensor]


def _read_regular(path: Path, *, role: str) -> bytes:
    if not isinstance(path, Path) or path.is_symlink() or not path.exists():
        raise ValueError(f"{role} must be one regular file")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role} must be one regular file")
    return path.read_bytes()


def _lower_hex(value: object, *, role: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{role} digest differs")
    return value


def _object(value: object, keys: set[str], *, role: str) -> dict[str, object]:
    if type(value) is not dict or set(value) != keys:
        raise ValueError(f"{role} schema differs")
    return value


def _endpoint_metric(value: object, *, queries: int, role: str) -> EndpointMetrics:
    row = _object(
        value,
        {
            "correct",
            "queries",
            "recall_at_1",
            "mean_nearest_positive_cosine",
            "mean_nearest_negative_cosine",
            "mean_margin",
        },
        role=role,
    )
    correct = row["correct"]
    observed_queries = row["queries"]
    floats = (
        row["recall_at_1"],
        row["mean_nearest_positive_cosine"],
        row["mean_nearest_negative_cosine"],
        row["mean_margin"],
    )
    if (
        type(correct) is not int
        or type(observed_queries) is not int
        or observed_queries != queries
        or not 0 <= correct <= observed_queries
        or any(type(item) is not float or not math.isfinite(item) for item in floats)
        or row["recall_at_1"] != correct / observed_queries
    ):
        raise ValueError(f"{role} arithmetic differs")
    return EndpointMetrics(
        correct=correct,
        queries=observed_queries,
        recall_at_1=row["recall_at_1"],
        mean_nearest_positive_cosine=row["mean_nearest_positive_cosine"],
        mean_nearest_negative_cosine=row["mean_nearest_negative_cosine"],
        mean_margin=row["mean_margin"],
    )


def _snapshot_metrics(value: object, *, role: str) -> dict[str, tuple[EndpointMetrics, ...]]:
    snapshot = _object(
        value,
        {"optimization", "clean_validation", "burned_diagnostic"},
        role=role,
    )
    result: dict[str, tuple[EndpointMetrics, ...]] = {}
    for band, queries in (
        ("optimization", 3_963),
        ("clean_validation", 2_746),
        ("burned_diagnostic", 1_345),
    ):
        values = _object(snapshot[band], {"raw", "projected"}, role=f"{role} {band}")
        result[band] = (
            _endpoint_metric(values["raw"], queries=queries, role=f"{role} {band} raw"),
            _endpoint_metric(
                values["projected"], queries=queries, role=f"{role} {band} projected"
            ),
        )
    return result


def load_seed_endpoint_authority(
    *,
    path: Path,
    expected_sha256: str,
    expected_bytes: int,
    expected_seed: int,
) -> SeedEndpointAuthority:
    """Authenticate one full seed result and project only endpoint authority."""

    _lower_hex(expected_sha256, role="seed result")
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("seed result byte length differs")
    if type(expected_seed) is not int or expected_seed not in (17, 29, 43):
        raise ValueError("seed result seed differs")
    raw = _read_regular(path, role="seed result")
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("seed result identity differs")
    value = read_control_seed_receipt(raw)
    config = SiglipProxyControlConfig()
    source = _object(value["source"], {"revision", "tree_digest", "dirty"}, role="source")
    dataset = _object(
        value["dataset"],
        {
            "name",
            "revision",
            "manifest_sha256",
            "optimization_examples",
            "clean_validation_examples",
            "burned_diagnostic_examples",
        },
        role="dataset",
    )
    model = _object(
        value["model"],
        {"name", "revision", "resolved_revision", "initial_state_sha256"},
        role="model",
    )
    initial_state_sha256 = _lower_hex(model["initial_state_sha256"], role="initial state")
    config_sha256 = _config_sha256(config)
    if (
        type(value["seed"]) is not int
        or value["seed"] != expected_seed
        or source["dirty"] is not False
        or dataset
        != {
            "name": config.dataset_name,
            "revision": config.dataset_revision,
            "manifest_sha256": dataset["manifest_sha256"],
            "optimization_examples": 3_963,
            "clean_validation_examples": 2_746,
            "burned_diagnostic_examples": 1_345,
        }
        or _lower_hex(dataset["manifest_sha256"], role="manifest")
        != dataset["manifest_sha256"]
        or model["name"] != config.model_name
        or model["revision"] != config.model_revision
        or model["resolved_revision"] != config.model_revision
        or value["config_sha256"] != config_sha256
    ):
        raise ValueError("seed result model or dataset authority differs")
    environment = _object(
        value["environment"], set(ControlRunAuthority.__dataclass_fields__), role="environment"
    )
    try:
        run_authority = ControlRunAuthority(**environment)
    except (TypeError, ValueError) as error:
        raise ValueError("seed result environment differs") from error
    if (
        run_authority.source_revision != source["revision"]
        or run_authority.source_tree_digest != source["tree_digest"]
        or run_authority.manifest_sha256 != dataset["manifest_sha256"]
    ):
        raise ValueError("seed result cross-object authority differs")
    evaluation = _object(value["evaluation"], {"initial", "final"}, role="evaluation")
    initial = _snapshot_metrics(evaluation["initial"], role="initial")
    trained = _snapshot_metrics(evaluation["final"], role="final")
    checkpoint = _object(
        value["checkpoint"],
        {"basename", "receipt_basename", "sha256", "bytes", "epoch"},
        role="checkpoint",
    )
    checkpoint_sha256 = _lower_hex(checkpoint["sha256"], role="checkpoint")
    if (
        type(checkpoint["basename"]) is not str
        or not checkpoint["basename"]
        or type(checkpoint["receipt_basename"]) is not str
        or not checkpoint["receipt_basename"]
        or type(checkpoint["bytes"]) is not int
        or checkpoint["bytes"] <= 0
        or type(checkpoint["epoch"]) is not int
        or checkpoint["epoch"] != 60
    ):
        raise ValueError("checkpoint receipt authority differs")
    return SeedEndpointAuthority(
        seed=expected_seed,
        initial_state_sha256=initial_state_sha256,
        initial_raw=initial["burned_diagnostic"][0],
        initial_projected=initial["burned_diagnostic"][1],
        trained_raw=trained["burned_diagnostic"][0],
        trained_projected=trained["burned_diagnostic"][1],
        checkpoint_basename=checkpoint["basename"],
        checkpoint_sha256=checkpoint_sha256,
        checkpoint_bytes=checkpoint["bytes"],
        config_sha256=config_sha256,
        run_authority_sha256=_run_authority_sha256(run_authority),
    )


def load_transfer_checkpoint(
    *,
    path: Path,
    expected_sha256: str,
    expected_bytes: int,
    expected_seed: int,
    expected_config_sha256: str,
    expected_run_authority_sha256: str,
) -> LoadedTransferCheckpoint:
    """Authenticate and project model state without changing process RNG."""

    for role, digest in (
        ("checkpoint", expected_sha256),
        ("config", expected_config_sha256),
        ("run authority", expected_run_authority_sha256),
    ):
        _lower_hex(digest, role=role)
    if type(expected_bytes) is not int or expected_bytes <= 0:
        raise ValueError("checkpoint byte length differs")
    if type(expected_seed) is not int or expected_seed not in (17, 29, 43):
        raise ValueError("checkpoint seed differs")
    raw = _read_regular(path, role="checkpoint")
    if len(raw) != expected_bytes or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("checkpoint identity differs")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(list(cuda_states))
    expected_keys = {
        "claim_eligible",
        "completed_epoch",
        "config_sha256",
        "cpu_rng_state",
        "cuda_rng_states",
        "final_objective",
        "initial_snapshot_sha256",
        "maximum_score_disagreement",
        "model_state",
        "optimizer_state",
        "run_authority_sha256",
        "sampler_cycles",
        "sampler_positions",
        "schema",
        "seed",
    }
    if type(payload) is not dict or set(payload) != expected_keys:
        raise ValueError("checkpoint payload schema differs")
    initial_snapshot = _lower_hex(payload["initial_snapshot_sha256"], role="initial snapshot")
    model_state = payload["model_state"]
    cycles = payload["sampler_cycles"]
    positions = payload["sampler_positions"]
    if (
        payload["schema"] != "sfora-siglip-proxy-checkpoint-payload-v1"
        or payload["claim_eligible"] is not False
        or type(payload["seed"]) is not int
        or payload["seed"] != expected_seed
        or type(payload["completed_epoch"]) is not int
        or payload["completed_epoch"] != 60
        or payload["config_sha256"] != expected_config_sha256
        or payload["run_authority_sha256"] != expected_run_authority_sha256
        or type(payload["final_objective"]) is not float
        or not math.isfinite(payload["final_objective"])
        or type(payload["maximum_score_disagreement"]) is not float
        or not 0.0 <= payload["maximum_score_disagreement"] <= 2.0e-5
        or type(cycles) is not tuple
        or type(positions) is not tuple
        or len(cycles) != 49
        or len(positions) != 49
        or any(type(value) is not int or value < 0 for value in cycles + positions)
        or not isinstance(payload["cpu_rng_state"], torch.Tensor)
        or payload["cpu_rng_state"].dtype != torch.uint8
        or type(payload["cuda_rng_states"]) is not tuple
        or any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
            for value in payload["cuda_rng_states"]
        )
        or type(payload["optimizer_state"]) is not dict
        or type(model_state) is not OrderedDict
        or not model_state
        or any(type(name) is not str or not name for name in model_state)
        or any(not isinstance(value, torch.Tensor) for value in model_state.values())
        or any(
            value.is_floating_point() and not bool(torch.isfinite(value).all())
            for value in model_state.values()
        )
    ):
        raise ValueError("checkpoint payload authority differs")
    copied = OrderedDict(
        (name, tensor.detach().cpu().clone()) for name, tensor in model_state.items()
    )
    return LoadedTransferCheckpoint(
        seed=expected_seed,
        initial_snapshot_sha256=initial_snapshot,
        model_state=MappingProxyType(copied),
    )
