#!/usr/bin/env python3
"""Evaluate SOP nested-rank checkpoints from authenticated local artifacts."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn

from sfora.nested_rank_evaluation import (
    _bounded_top_indices,
    class_bootstrap_lower_bound,
    classify_promotion,
    evaluate_paired_rankings,
    rank_query_gallery,
    rank_self_retrieval,
    rank_self_retrieval_int8,
    recompute_self_retrieval,
)

__all__ = (
    "_bounded_top_indices",
    "class_bootstrap_lower_bound",
    "classify_promotion",
    "evaluate_paired_rankings",
    "load_training_artifact",
    "rank_self_retrieval",
    "rank_self_retrieval_int8",
    "rank_query_gallery",
    "recompute_self_retrieval",
    "restore_candidate_state",
)


_RESULT_KEYS = {
    "schema",
    "claim_eligible",
    "status",
    "arm",
    "split_seed",
    "temperature",
    "optimization_rows",
    "optimization_classes",
    "epochs",
    "history",
    "authority",
    "model_artifact",
    "run_receipt",
}
_ARMS = {
    "proxy-anchor",
    "neighborhood",
    "combined",
    "proxy-anchor-768",
    "s2sd-768-to-128",
}


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _artifact_descriptor(value: object, *, expected_path: str) -> bool:
    return (
        type(value) is dict
        and set(value) == {"path", "sha256", "bytes"}
        and value["path"] == expected_path
        and _is_sha256(value["sha256"])
        and type(value["bytes"]) is int
        and value["bytes"] > 0
    )


def load_training_artifact(
    result_path: Path,
    result_sha256: str,
    model_path: Path,
) -> tuple[dict[str, object], dict[str, object]]:
    """Authenticate and load one completed local NNRL training artifact."""

    if (
        not isinstance(result_path, Path)
        or not isinstance(model_path, Path)
        or not _is_sha256(result_sha256)
        or result_path.is_symlink()
        or model_path.is_symlink()
        or not result_path.is_file()
        or not model_path.is_file()
    ):
        raise ValueError("NNRL evaluation artifact path differs")
    result_bytes = result_path.read_bytes()
    if hashlib.sha256(result_bytes).hexdigest() != result_sha256:
        raise ValueError("NNRL result digest differs")
    try:
        result = json.loads(result_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("NNRL result JSON differs") from error
    if (
        type(result) is not dict
        or set(result) != _RESULT_KEYS
        or _canonical_json_bytes(result) != result_bytes
        or result["schema"] != "sfora-nnrl-sop-training-result-v1"
        or result["claim_eligible"] is not False
        or result["status"] != "COMPLETE"
        or result["arm"] not in _ARMS
        or type(result["split_seed"]) is not int
        or type(result["temperature"]) is not float
        or not math.isfinite(result["temperature"])
        or result["temperature"] <= 0.0
        or type(result["optimization_rows"]) is not int
        or result["optimization_rows"] <= 0
        or type(result["optimization_classes"]) is not int
        or result["optimization_classes"] <= 1
        or type(result["epochs"]) is not int
        or result["epochs"] <= 0
        or type(result["history"]) is not list
        or len(result["history"]) != result["epochs"]
        or type(result["authority"]) is not dict
        or not result["authority"]
        or not _artifact_descriptor(result["model_artifact"], expected_path=model_path.name)
        or not _artifact_descriptor(result["run_receipt"], expected_path="run-receipt.json")
    ):
        raise ValueError("NNRL result authority differs")
    for epoch, row in enumerate(result["history"], start=1):
        if (
            type(row) is not dict
            or set(row) != {"epoch", "steps", "attempted_steps", "skipped_updates", "mean_loss"}
            or row["epoch"] != epoch
            or type(row["steps"]) is not int
            or row["steps"] <= 0
            or row["attempted_steps"] != row["steps"]
            or row["skipped_updates"] != 0
            or type(row["mean_loss"]) is not float
            or not math.isfinite(row["mean_loss"])
        ):
            raise ValueError("NNRL result history differs")
    model_authority = result["model_artifact"]
    model_bytes = model_path.read_bytes()
    if (
        len(model_bytes) != model_authority["bytes"]
        or hashlib.sha256(model_bytes).hexdigest() != model_authority["sha256"]
    ):
        raise ValueError("NNRL model authority differs")
    model = torch.load(model_path, map_location="cpu", weights_only=True)
    if type(model) is not dict or set(model) != {"encoder", "head", "raw_proxies"}:
        raise ValueError("NNRL model schema differs")
    return result, model


def _validate_state_dict(module: nn.Module, value: object) -> dict[str, torch.Tensor]:
    expected = module.state_dict()
    if (
        type(value) is not dict
        or set(value) != set(expected)
        or any(
            type(name) is not str
            or type(tensor) is not torch.Tensor
            or tensor.shape != expected[name].shape
            or tensor.dtype != expected[name].dtype
            or not torch.isfinite(tensor).all()
            for name, tensor in value.items()
        )
    ):
        raise ValueError("NNRL model state differs")
    return value


def restore_candidate_state(
    encoder: nn.Module,
    head: nn.Module,
    artifact: object,
) -> torch.Tensor:
    """Strictly restore an authenticated candidate encoder and projection head."""

    if (
        not isinstance(encoder, nn.Module)
        or not isinstance(head, nn.Module)
        or type(artifact) is not dict
        or set(artifact) != {"encoder", "head", "raw_proxies"}
    ):
        raise ValueError("NNRL model state differs")
    encoder_state = _validate_state_dict(encoder, artifact["encoder"])
    head_state = _validate_state_dict(head, artifact["head"])
    raw_proxies = artifact["raw_proxies"]
    if (
        type(raw_proxies) is not torch.Tensor
        or raw_proxies.ndim != 2
        or raw_proxies.shape[0] <= 1
        or raw_proxies.shape[1] <= 1
        or not torch.isfinite(raw_proxies).all()
    ):
        raise ValueError("NNRL model state differs")
    encoder.load_state_dict(encoder_state, strict=True)
    head.load_state_dict(head_state, strict=True)
    return raw_proxies
