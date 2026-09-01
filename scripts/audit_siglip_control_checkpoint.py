#!/usr/bin/env python3
"""Authenticate and audit one terminal SigLIP control checkpoint locally."""

from __future__ import annotations

import hashlib
import stat
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_SCRIPT_DIRECTORY = Path(__file__).resolve().parent
if str(_SCRIPT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIRECTORY))

from run_siglip_proxy_control import (  # noqa: E402
    CheckpointAuthority,
    ControlRunAuthority,
    SiglipProxyControlConfig,
    _checkpoint_authority_from_receipt,
    _config_sha256,
    _json_compatible,
    control_aggregate_receipt_bytes,
    read_control_seed_receipt,
)

_SELECTED_SEED = 17
_FINAL_EPOCH = 60
_BURNED_QUERIES = 1_345


@dataclass(frozen=True, slots=True)
class AuthenticatedControlCampaign:
    """Exact terminal campaign and selected checkpoint authority."""

    seed: int
    seed_receipt: dict[str, Any]
    seed_receipt_sha256: str
    aggregate_sha256: str
    run_authority: ControlRunAuthority
    checkpoint: CheckpointAuthority
    final_burned_correct: dict[str, int]


def _read_regular(path: Path, *, role: str) -> bytes:
    if not isinstance(path, Path) or path.is_symlink() or not path.exists():
        raise ValueError(f"{role} must be a regular file")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{role} must be a regular file")
    return path.read_bytes()


def _require_keys(value: object, expected: set[str], *, role: str) -> dict[str, Any]:
    if type(value) is not dict or set(value) != expected:
        raise ValueError(f"{role} schema differs")
    return cast(dict[str, Any], value)


def _burned_correct(seed_receipt: dict[str, Any]) -> dict[str, int]:
    evaluation = _require_keys(
        seed_receipt.get("evaluation"), {"initial", "final"}, role="evaluation"
    )
    final = _require_keys(
        evaluation["final"],
        {"optimization", "clean_validation", "burned_diagnostic"},
        role="final evaluation",
    )
    burned = _require_keys(
        final["burned_diagnostic"], {"raw", "projected"}, role="burned evaluation"
    )
    result: dict[str, int] = {}
    scalar_keys = {
        "correct",
        "queries",
        "recall_at_1",
        "mean_nearest_positive_cosine",
        "mean_nearest_negative_cosine",
        "mean_margin",
    }
    for name in ("raw", "projected"):
        row = _require_keys(burned[name], scalar_keys, role=f"{name} burned metrics")
        correct = row["correct"]
        queries = row["queries"]
        recall = row["recall_at_1"]
        if (
            type(correct) is not int
            or type(queries) is not int
            or type(recall) is not float
            or queries != _BURNED_QUERIES
            or not 0 <= correct <= queries
            or recall != correct / queries
        ):
            raise ValueError(f"{name} burned metrics differ")
        result[name] = correct
    return result


def _validate_selected_seed_receipt(
    value: dict[str, Any],
) -> tuple[ControlRunAuthority, dict[str, int]]:
    config = SiglipProxyControlConfig()
    source = _require_keys(value.get("source"), {"revision", "tree_digest", "dirty"}, role="source")
    dataset = _require_keys(
        value.get("dataset"),
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
    model = _require_keys(
        value.get("model"),
        {"name", "revision", "resolved_revision", "initial_state_sha256"},
        role="model",
    )
    config_payload = _json_compatible(vars(config))
    source_manifest = dataset["manifest_sha256"]
    if (
        value.get("seed") != _SELECTED_SEED
        or type(value.get("seed")) is not int
        or source["dirty"] is not False
        or dataset
        != {
            "name": config.dataset_name,
            "revision": config.dataset_revision,
            "manifest_sha256": source_manifest,
            "optimization_examples": 3_963,
            "clean_validation_examples": 2_746,
            "burned_diagnostic_examples": _BURNED_QUERIES,
        }
        or model["name"] != config.model_name
        or model["revision"] != config.model_revision
        or model["resolved_revision"] != config.model_revision
        or value.get("config") != config_payload
        or value.get("config_sha256") != _config_sha256(config)
    ):
        raise ValueError("selected seed model, dataset, or config authority differs")
    environment = _require_keys(
        value.get("environment"), set(ControlRunAuthority.__dataclass_fields__), role="environment"
    )
    try:
        run_authority = ControlRunAuthority(**environment)
    except (TypeError, ValueError) as error:
        raise ValueError("selected seed environment differs") from error
    training = _require_keys(
        value.get("training"),
        {
            "optimizer_steps",
            "steps_per_epoch",
            "microbatch_size",
            "final_objective",
            "maximum_score_disagreement",
        },
        role="training",
    )
    if (
        run_authority.source_revision != source["revision"]
        or run_authority.source_tree_digest != source["tree_digest"]
        or run_authority.manifest_sha256 != source_manifest
        or run_authority.steps_per_epoch != 33
        or run_authority.evaluation_batch_size != 32
        or run_authority.query_block != 128
        or training["steps_per_epoch"] != run_authority.steps_per_epoch
        or training["microbatch_size"] != run_authority.microbatch_size
    ):
        raise ValueError("selected seed run authority differs")
    return run_authority, _burned_correct(value)


def read_authenticated_control_campaign(
    *,
    aggregate: Path,
    seed_receipts: tuple[Path, ...],
    checkpoint_directory: Path,
    selected_seed: int,
) -> AuthenticatedControlCampaign:
    """Cross-bind the terminal aggregate, seed receipt, and final checkpoint."""

    if (
        type(seed_receipts) is not tuple
        or len(seed_receipts) != 3
        or type(selected_seed) is not int
        or selected_seed != _SELECTED_SEED
        or not isinstance(checkpoint_directory, Path)
        or checkpoint_directory.is_symlink()
        or not checkpoint_directory.is_dir()
    ):
        raise ValueError("control campaign paths or selected seed differ")
    aggregate_raw = _read_regular(aggregate, role="control aggregate")
    receipt_raws = tuple(
        _read_regular(path, role="control seed receipt") for path in seed_receipts
    )
    expected_aggregate = control_aggregate_receipt_bytes(receipt_raws)
    if aggregate_raw != expected_aggregate:
        raise ValueError("control aggregate authority differs")
    parsed = tuple(read_control_seed_receipt(raw) for raw in receipt_raws)
    selected = next((value for value in parsed if value.get("seed") == selected_seed), None)
    if selected is None:
        raise ValueError("selected seed receipt is absent")
    run_authority, final_burned_correct = _validate_selected_seed_receipt(selected)
    checkpoint_value = _require_keys(
        selected.get("checkpoint"),
        {"basename", "receipt_basename", "sha256", "bytes", "epoch"},
        role="checkpoint",
    )
    if (
        type(checkpoint_value["epoch"]) is not int
        or checkpoint_value["epoch"] != _FINAL_EPOCH
        or type(checkpoint_value["basename"]) is not str
        or type(checkpoint_value["receipt_basename"]) is not str
        or type(checkpoint_value["sha256"]) is not str
        or type(checkpoint_value["bytes"]) is not int
    ):
        raise ValueError("selected checkpoint authority differs")
    checkpoint = _checkpoint_authority_from_receipt(
        checkpoint_directory / checkpoint_value["receipt_basename"],
        directory=checkpoint_directory,
        expected_seed=selected_seed,
    )
    if (
        checkpoint.epoch != checkpoint_value["epoch"]
        or checkpoint.path.name != checkpoint_value["basename"]
        or checkpoint.receipt_path.name != checkpoint_value["receipt_basename"]
        or checkpoint.sha256 != checkpoint_value["sha256"]
        or checkpoint.bytes != checkpoint_value["bytes"]
    ):
        raise ValueError("selected checkpoint binding differs")
    return AuthenticatedControlCampaign(
        seed=selected_seed,
        seed_receipt=selected,
        seed_receipt_sha256=hashlib.sha256(receipt_raws[0]).hexdigest(),
        aggregate_sha256=hashlib.sha256(aggregate_raw).hexdigest(),
        run_authority=run_authority,
        checkpoint=checkpoint,
        final_burned_correct=final_burned_correct,
    )
