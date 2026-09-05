#!/usr/bin/env python3
"""Local authenticated boundaries for the fixed class-language pilot."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

from sfora.siglip_language_guidance import standardized_text_gram
from sfora.siglip_recovery_evaluation import recovery_decision

EVALUATOR_SHA256 = "e60d8fa318a17b69985deb2c8f43339427b34576f34f48bf66a21cf683ab6985"
EVALUATION_SOURCE_SHA256 = {
    "runner": EVALUATOR_SHA256,
    "evaluation_core": "d8952c1ea5e6ea9c747379ee07e25330aab9632829ae90fc179ebde8ffad0568",
    "retrieval_core": "fa2f06d1fa78a8058b1a90f4103eb3966e90931504acd990fbb5440f61bee34c",
}


def _authenticate(path: Path, expected_sha256: str) -> None:
    if (
        type(expected_sha256) is not str
        or len(expected_sha256) != 64
        or any(c not in "0123456789abcdef" for c in expected_sha256)
    ):
        raise ValueError("input digest authority invalid")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("input digest differs")


def load_text_state(path: Path, expected_sha256: str, model: torch.nn.Module) -> None:
    """Authenticate the full container, then strictly restore only its text state."""
    _authenticate(path, expected_sha256)
    expected = model.state_dict()
    with safe_open(path, framework="pt", device="cpu") as container:
        container_keys = container.keys()
        keys = {k for k in container_keys if k.startswith("text_model.")}
        if keys != {f"text_model.{k}" for k in expected}:
            raise ValueError("text container key authority differs")
        state = {}
        for key in sorted(keys):
            name = key.removeprefix("text_model.")
            value = container.get_tensor(key)
            reference = expected[name]
            if (
                value.shape != reference.shape
                or value.dtype != torch.float32
                or reference.dtype != torch.float32
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError("text container tensor authority differs")
            state[name] = value
    model.load_state_dict(state, strict=True)


def official_optimization_prompts(path: Path, expected_sha256: str) -> tuple[str, ...]:
    """Bind the official name mapping but construct prompts for IDs0..48 only."""
    _authenticate(path, expected_sha256)
    value = json.loads(path.read_bytes())
    try:
        label = value["features"]["label"]
        names = label["names"]
        valid = (
            label["_type"] == "ClassLabel"
            and type(names) is list
            and len(names) == 196
            and all(type(name) is str and name.strip() for name in names)
        )
    except (KeyError, TypeError):
        valid = False
    if not valid:
        raise ValueError("official class-name metadata differs")
    return tuple(f"a photo of a {names[i]}." for i in range(49))


def encode_frozen_text(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    expected_tokens_sha256: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode one authenticated49x64 batch without a synthesized attention mask."""
    vocab_size = getattr(getattr(model, "config", None), "vocab_size", None)
    if (
        input_ids.shape != (49, 64)
        or input_ids.dtype != torch.int64
        or type(vocab_size) is not int
        or vocab_size <= 0
        or bool((input_ids < 0).any())
        or bool((input_ids >= vocab_size).any())
        or hashlib.sha256(input_ids.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        != expected_tokens_sha256
    ):
        raise ValueError("text token authority differs")
    model.eval().requires_grad_(False)
    with torch.no_grad():
        pooled = model(input_ids=input_ids).pooler_output.float()
        norms = pooled.norm(dim=1, keepdim=True)
        if not bool(torch.isfinite(pooled).all()) or bool((norms <= 0).any()):
            raise ValueError("invalid frozen text descriptors")
        vectors = pooled / norms
        gram = standardized_text_gram(vectors)
    return vectors, gram


def recovery_initialization_choice(
    evaluation: dict[str, Any],
    monitor: dict[str, Any],
    *,
    pair_sha256: str,
    pair_monitor_sha256: str,
    evaluation_sha256: str,
) -> dict[str, Any]:
    """Recompute selection from authenticated result views, not copied decisions.

    The caller must first authenticate full files and the pair/checkpoint chain.
    This function validates the distinct evaluation monitor and measured gates;
    no remaining time from the recovery campaign becomes pilot time.
    """
    expected_evaluation = {
        "schema": "sfora-siglip-depth-recovery-evaluation-v1",
        "claim_eligible": False,
        "quality_measured": True,
        "seed": 17,
        "surface": "exploratory-reuse-49..81",
        "pair_sha256": pair_sha256,
        "monitor_sha256": pair_monitor_sha256,
        "runner_sha256": EVALUATOR_SHA256,
        "source_sha256": EVALUATION_SOURCE_SHA256,
    }
    expected_monitor = {
        "schema": "sfora-recovery-evaluation-monitor-v1",
        "claim_eligible": False,
        "exit_code": 0,
        "stop_reason": None,
        "result_sha256": evaluation_sha256,
        "pair_monitor_sha256": pair_monitor_sha256,
        "monitor_sha256": "db5ae1d9293d004eee5755f60fc2a392f0107f98678cb7ff18c5e8dee0c753dc",
        "script_sha256": "b9adad0f2bdf980e76197e7bad0b7c6012c4730bc66ffed7db7be506640953ec",
    }
    for actual, expected in ((evaluation, expected_evaluation), (monitor, expected_monitor)):
        for key, wanted in expected.items():
            got = actual.get(key)
            if type(got) is not type(wanted) or got != wanted:
                raise ValueError(f"recovery evaluation {key} authority differs")
    try:
        elapsed = evaluation["resources"]["elapsed_seconds"]
        monitor_elapsed = monitor["elapsed_s"]
        if any(
            type(x) is not float or not math.isfinite(x) or x <= 0
            for x in (elapsed, monitor_elapsed)
        ):
            raise ValueError("recovery evaluation elapsed evidence invalid")
        projection = 2 * elapsed + 300
        if not math.isfinite(projection) or projection > 1800:
            raise ValueError("four-gallery evaluation reserve is insufficient")
        cells = evaluation["cells"]
        if set(cells) != {"teacher", "pa", "relational"}:
            raise ValueError("recovery evaluation cell authority differs")
        teacher = cells["teacher"]
        if teacher["map_at_r"] != 0.7913744556922272:
            raise ValueError("recovery teacher MAP reproduction differs")
        decision = recovery_decision(
            teacher,
            {k: cells[k] for k in ("pa", "relational")},
            evaluation["search_profile"]["samples_ns"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("recovery evaluation evidence incomplete") from error
    selected = decision["selected_arm"]
    return {
        "initialization": selected or "teacher",
        "layers": 18 if selected else 27,
        "relational_base": selected == "relational",
        "recomputed_decision": decision,
        "evaluation_projection_seconds": projection,
        "pilot_total_seconds": 7200,
    }
