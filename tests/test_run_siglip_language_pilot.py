"""Real local tensor containers exercise text-state and prompt authority."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file
from transformers import SiglipTextConfig, SiglipTextModel

SPEC = importlib.util.spec_from_file_location(
    "run_siglip_language_pilot",
    Path(__file__).parents[1] / "scripts/run_siglip_language_pilot.py",
)
assert SPEC and SPEC.loader
subject = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = subject
SPEC.loader.exec_module(subject)


def _text_model() -> SiglipTextModel:
    torch.manual_seed(17)
    config = SiglipTextConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=64,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=1,
    )
    config._attn_implementation = "eager"
    return SiglipTextModel(config)


def _container(path: Path, model: torch.nn.Module) -> dict[str, torch.Tensor]:
    tensors = {f"text_model.{k}": v.detach().clone() for k, v in model.state_dict().items()}
    tensors["vision_model.unused"] = torch.tensor([123.0])
    save_file(tensors, path)
    return tensors


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_text_container_load_strips_one_prefix_and_matches_real_model(tmp_path: Path) -> None:
    original = _text_model()
    path = tmp_path / "model.safetensors"
    _container(path, original)
    destination = _text_model()
    with torch.no_grad():
        for p in destination.parameters():
            p.zero_()
    subject.load_text_state(path, _sha(path), destination)
    assert all(
        torch.equal(original.state_dict()[k], v) for k, v in destination.state_dict().items()
    )
    ids = torch.arange(64).remainder(32).unsqueeze(0)
    with torch.no_grad():
        assert torch.equal(original(ids).pooler_output, destination(ids).pooler_output)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "nan", "double-prefix"])
def test_text_container_semantic_mutations_reject_after_rehash(
    tmp_path: Path,
    mutation: str,
) -> None:
    model = _text_model()
    path = tmp_path / "model.safetensors"
    tensors = _container(path, model)
    key = "text_model.embeddings.token_embedding.weight"
    if mutation == "missing":
        del tensors[key]
    elif mutation == "extra":
        tensors["text_model.extra"] = torch.ones(1)
    elif mutation == "shape":
        tensors[key] = tensors[key][:1]
    elif mutation == "dtype":
        tensors[key] = tensors[key].half()
    elif mutation == "nan":
        tensors[key][0, 0] = float("nan")
    else:
        tensors["text_model." + key] = tensors.pop(key)
    save_file(tensors, path)
    with pytest.raises(ValueError):
        subject.load_text_state(path, _sha(path), model)


def test_corrupt_text_container_fails_hash_before_parsing(tmp_path: Path) -> None:
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"not a tensor container")
    with pytest.raises(ValueError, match="digest"):
        subject.load_text_state(path, "0" * 64, _text_model())


def test_prompts_use_only_exact_optimization_names_from_authenticated_metadata(
    tmp_path: Path,
) -> None:
    path = tmp_path / "dataset_info.json"
    info: dict[str, Any] = {
        "features": {
            "label": {"names": [f"Official model {i}" for i in range(196)], "_type": "ClassLabel"}
        },
    }
    path.write_text(json.dumps(info), encoding="utf-8")
    prompts = subject.official_optimization_prompts(path, _sha(path))
    assert len(prompts) == 49
    assert prompts[0] == "a photo of a Official model 0."
    assert prompts[-1] == "a photo of a Official model 48."
    assert all("model 49." not in p for p in prompts)
    info["features"]["label"]["names"][49] = "unseen changed name"
    path.write_text(json.dumps(info), encoding="utf-8")
    assert subject.official_optimization_prompts(path, _sha(path)) == prompts
    with pytest.raises(ValueError, match="digest"):
        subject.official_optimization_prompts(path, "0" * 64)


@pytest.mark.parametrize("mutation", ["count", "empty", "nonstring", "wrong-type"])
def test_prompt_metadata_must_match_class_mapping(tmp_path: Path, mutation: str) -> None:
    path = tmp_path / "dataset_info.json"
    names: list[Any] = [f"model {i}" for i in range(196)]
    if mutation == "count":
        names.pop()
    elif mutation == "empty":
        names[1] = ""
    elif mutation == "nonstring":
        names[1] = 1
    info = {
        "features": {
            "label": {
                "names": names,
                "_type": "Value" if mutation == "wrong-type" else "ClassLabel",
            }
        }
    }
    path.write_text(json.dumps(info), encoding="utf-8")
    with pytest.raises(ValueError):
        subject.official_optimization_prompts(path, _sha(path))


def _tokens() -> torch.Tensor:
    return torch.arange(49 * 64).reshape(49, 64).add(torch.arange(49)[:, None]).remainder(32)


def _token_sha(ids: torch.Tensor) -> str:
    return hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()


def test_frozen_text_encoding_matches_direct_pooling_without_gradients() -> None:
    model = _text_model()
    ids = _tokens()
    with torch.no_grad():
        pooled = model(ids).pooler_output
        expected = pooled / pooled.norm(dim=1, keepdim=True)
        similarities = expected @ expected.T
        off_diagonal = similarities[~torch.eye(49, dtype=torch.bool)]
        mean = off_diagonal.sum() / (49 * 48)
        variance = ((off_diagonal - mean) ** 2).sum() / (49 * 48)
        expected_gram = (similarities - mean) / variance.sqrt()
        expected_gram.fill_diagonal_(0)
    vectors, gram = subject.encode_frozen_text(model, ids, _token_sha(ids))
    torch.testing.assert_close(vectors, expected, rtol=0, atol=1e-6)
    torch.testing.assert_close(gram, expected_gram, rtol=0, atol=1e-5)
    assert not vectors.requires_grad and not gram.requires_grad
    assert not model.training
    assert all(not p.requires_grad and p.grad is None for p in model.parameters())


@pytest.mark.parametrize("mutation", ["hash", "rows", "length", "dtype", "negative", "vocab"])
def test_token_authority_rejects_before_text_encoding(mutation: str) -> None:
    model = _text_model()
    ids = _tokens()
    if mutation == "rows":
        ids = ids[:48]
    elif mutation == "length":
        ids = ids[:, :63]
    elif mutation == "dtype":
        ids = ids.int()
    elif mutation == "negative":
        ids[0, 0] = -1
    elif mutation == "vocab":
        ids[0, 0] = 32
    digest = "0" * 64 if mutation == "hash" else _token_sha(ids)
    with pytest.raises(ValueError):
        subject.encode_frozen_text(model, ids, digest)
    # Validation occurs before changing the caller's trainable model state.
    assert model.training and all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("value", [0.0, float("nan")])
def test_degenerate_text_pooling_is_invalid_not_a_training_target(value: float) -> None:
    model = _text_model()
    with torch.no_grad():
        model.head.weight.fill_(value)
        model.head.bias.fill_(value)
    ids = _tokens()
    with pytest.raises(ValueError):
        subject.encode_frozen_text(model, ids, _token_sha(ids))


def _selection_evidence() -> tuple[dict[str, Any], dict[str, Any]]:
    cells = {
        name: {
            "queries": 2746,
            "correct": 2596,
            "map_at_r": 0.7913744556922272,
            "descriptor_bytes": 5623808,
        }
        for name in ("teacher", "pa", "relational")
    }
    value: dict[str, Any] = {
        "schema": "sfora-siglip-depth-recovery-evaluation-v1",
        "claim_eligible": False,
        "quality_measured": True,
        "seed": 17,
        "surface": "exploratory-reuse-49..81",
        "pair_sha256": "1" * 64,
        "monitor_sha256": "2" * 64,
        "runner_sha256": "e60d8fa318a17b69985deb2c8f43339427b34576f34f48bf66a21cf683ab6985",
        "source_sha256": {
            "runner": "e60d8fa318a17b69985deb2c8f43339427b34576f34f48bf66a21cf683ab6985",
            "evaluation_core": "d8952c1ea5e6ea9c747379ee07e25330aab9632829ae90fc179ebde8ffad0568",
            "retrieval_core": "fa2f06d1fa78a8058b1a90f4103eb3966e90931504acd990fbb5440f61bee34c",
        },
        "cells": cells,
        "search_profile": {"samples_ns": {k: [100] * 100 for k in cells}},
        "resources": {"elapsed_seconds": 300.0},
    }
    monitor = {
        "schema": "sfora-recovery-evaluation-monitor-v1",
        "claim_eligible": False,
        "exit_code": 0,
        "stop_reason": None,
        "result_sha256": "3" * 64,
        "pair_monitor_sha256": "2" * 64,
        "elapsed_s": 305.0,
        "monitor_sha256": "db5ae1d9293d004eee5755f60fc2a392f0107f98678cb7ff18c5e8dee0c753dc",
        "script_sha256": "b9adad0f2bdf980e76197e7bad0b7c6012c4730bc66ffed7db7be506640953ec",
    }
    return value, monitor


def _select(value: dict[str, Any], monitor: dict[str, Any]) -> dict[str, Any]:
    return cast(
        dict[str, Any],
        subject.recovery_initialization_choice(
            value,
            monitor,
            pair_sha256="1" * 64,
            pair_monitor_sha256="2" * 64,
            evaluation_sha256="3" * 64,
        ),
    )


def test_initialization_recomputes_all_gates_and_prefers_pa() -> None:
    value, monitor = _selection_evidence()
    assert _select(value, monitor)["initialization"] == "pa"
    value["cells"]["pa"]["correct"] = 2590
    assert _select(value, monitor)["initialization"] == "relational"
    value["search_profile"]["samples_ns"]["relational"] = [106] * 100
    choice = _select(value, monitor)
    assert choice["initialization"] == "teacher"
    assert choice["layers"] == 27 and choice["relational_base"] is False
    value["decision"] = {"selected_arm": "pa", "passed": True}
    # A copied selection claim cannot override the measured cells/search gate.
    assert _select(value, monitor)["initialization"] == "teacher"


@pytest.mark.parametrize(
    ("role", "key", "bad"),
    [
        ("monitor", "schema", "sfora-recovery-pair-monitor-v1"),
        ("monitor", "exit_code", False),
        ("monitor", "exit_code", 1),
        ("monitor", "stop_reason", "psi-cap"),
        ("monitor", "result_sha256", "9" * 64),
        ("monitor", "pair_monitor_sha256", "9" * 64),
        ("monitor", "monitor_sha256", "9" * 64),
        ("monitor", "script_sha256", "9" * 64),
        ("monitor", "claim_eligible", 0),
        ("monitor", "elapsed_s", float("nan")),
        ("result", "schema", "wrong"),
        ("result", "claim_eligible", 0),
        ("result", "seed", True),
        ("result", "quality_measured", False),
        ("result", "pair_sha256", "9" * 64),
        ("result", "monitor_sha256", "9" * 64),
        ("result", "runner_sha256", "9" * 64),
        ("result", "source_sha256", {}),
        ("result", "surface", "official-test"),
    ],
)
def test_initialization_requires_its_own_successful_evaluation_monitor(
    role: str,
    key: str,
    bad: object,
) -> None:
    value, monitor = _selection_evidence()
    (monitor if role == "monitor" else value)[key] = bad
    with pytest.raises(ValueError):
        _select(value, monitor)


def test_four_gallery_reserve_is_an_admission_gate_not_extra_budget() -> None:
    value, monitor = _selection_evidence()
    value["resources"]["elapsed_seconds"] = 750.0
    monitor["elapsed_s"] = 755.0
    choice = _select(value, monitor)
    assert choice["evaluation_projection_seconds"] == 1800.0
    assert choice["pilot_total_seconds"] == 7200
    bad = copy.deepcopy(value)
    bad["resources"]["elapsed_seconds"] = 750.001
    with pytest.raises(ValueError, match="reserve"):
        _select(bad, monitor)
    for elapsed in (False, -1.0, 0.0, float("nan"), float("inf")):
        bad["resources"]["elapsed_seconds"] = elapsed
        with pytest.raises(ValueError):
            _select(bad, monitor)
