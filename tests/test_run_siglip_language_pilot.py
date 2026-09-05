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

from sfora.siglip_depth_recovery import relational_cross_entropy
from sfora.siglip_language_guidance import language_centroid_cross_entropy, standardized_text_gram
from sfora.siglip_proxy_control import PooledProxyAnchorModel
from sfora.token_set_proxy_anchor import proxy_anchor_loss

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
    monitor["elapsed_s"] = 750.0
    choice = _select(value, monitor)
    assert choice["evaluation_projection_seconds"] == 1800.0
    assert choice["pilot_total_seconds"] == 7200
    slower_monitor = copy.deepcopy(monitor)
    slower_monitor["elapsed_s"] = 755.0
    with pytest.raises(ValueError, match="reserve"):
        _select(value, slower_monitor)
    bad = copy.deepcopy(value)
    bad["resources"]["elapsed_seconds"] = 750.001
    with pytest.raises(ValueError, match="reserve"):
        _select(bad, monitor)
    for elapsed in (False, -1.0, 0.0, float("nan"), float("inf")):
        bad["resources"]["elapsed_seconds"] = elapsed
        with pytest.raises(ValueError):
            _select(bad, monitor)


def _image_model() -> PooledProxyAnchorModel:
    torch.manual_seed(17)
    return PooledProxyAnchorModel(
        tower=torch.nn.Sequential(torch.nn.Linear(4, 5), torch.nn.Tanh()),
        input_dimensions=5,
        embedding_dimensions=3,
        class_count=4,
    )


@pytest.mark.parametrize("arm", ["base", "correct", "permuted"])
@pytest.mark.parametrize("relational", [False, True])
def test_twenty_language_updates_match_direct_full_batch_adamw(
    arm: str,
    relational: bool,
) -> None:
    direct = _image_model()
    replay = copy.deepcopy(direct)
    teacher = copy.deepcopy(direct).eval().requires_grad_(False) if relational else None
    torch.manual_seed(31)
    pixels = torch.randn(6, 4)
    labels = torch.tensor([0, 0, 1, 1, 3, 3])
    gram = (
        None
        if arm == "base"
        else standardized_text_gram(torch.nn.functional.normalize(torch.randn(4, 5), dim=1))
    )
    if arm == "permuted":
        assert gram is not None
        permutation = torch.tensor([1, 3, 0, 2])
        gram = gram[permutation][:, permutation]
    # Literal original recipe for this tiny topology; independent of the runner.
    linear = cast(torch.nn.Linear, cast(torch.nn.Sequential, direct.tower)[0])
    optimizer = torch.optim.AdamW(
        [
            {"params": [linear.bias], "lr": 1e-5, "initial_lr": 1e-5, "weight_decay": 0.0},
            {"params": [linear.weight], "lr": 1e-5, "initial_lr": 1e-5, "weight_decay": 1e-4},
            {
                "params": [direct.projection.weight],
                "lr": 1e-4,
                "initial_lr": 1e-4,
                "weight_decay": 1e-4,
            },
            {"params": [direct.proxies], "lr": 1e-2, "initial_lr": 1e-2, "weight_decay": 0.0},
        ],
        betas=(0.9, 0.999),
        eps=1e-8,
        foreach=False,
    )
    expected_losses = []
    for update in range(1, 21):
        import math

        multiplier = (
            update / 10
            if update <= 10
            else (0.1 + 0.45 * (1 + math.cos(math.pi * (update - 10) / 188)))
        )
        optimizer.zero_grad(set_to_none=True)
        for group in optimizer.param_groups:
            group["lr"] = group["initial_lr"] * multiplier
        vectors = direct.encode(pixels)
        loss = proxy_anchor_loss(
            vectors @ torch.nn.functional.normalize(direct.proxies, dim=1).T,
            labels,
            alpha=32.0,
            delta=0.1,
        )
        if gram is not None:
            loss = loss + language_centroid_cross_entropy(vectors, labels, gram)
        if teacher is not None:
            with torch.no_grad():
                teacher_vectors = teacher.encode(pixels)
            loss = loss + relational_cross_entropy(vectors, teacher_vectors)
        expected_losses.append(float(loss.detach()))
        torch.autograd.backward(loss)
        torch.nn.utils.clip_grad_norm_(
            direct.parameters(), 10.0, error_if_nonfinite=True, foreach=False
        )
        optimizer.step()
    events: list[dict[str, Any]] = []
    batch_sha = hashlib.sha256(
        b"sfora-recovery-batch-v1\0"
        + b'{"dtype":"torch.float32","shape":[6,4]}\n'
        + pixels.numpy().tobytes()
        + b'{"dtype":"torch.int64","shape":[6]}\n'
        + labels.numpy().tobytes()
    ).hexdigest()
    evidence = subject.train_language_arm(
        replay,
        teacher,
        lambda update: (pixels, labels),
        arm=arm,
        gram=gram,
        expected_input_hashes=None if arm == "base" else [batch_sha] * 20,
        microbatch_size=2,
        progress=events.append,
        synchronize=lambda: None,
    )
    assert evidence["completed_updates"] == 20
    expected_gram_sha = (
        None
        if gram is None
        else hashlib.sha256(
            b"sfora-language-gram-v1\0"
            + b'{"dtype":"torch.float32","shape":[4,4]}\n'
            + gram.numpy().tobytes()
        ).hexdigest()
    )
    assert evidence["consumed_gram_sha256"] == expected_gram_sha
    assert len(events) == 20 and [e["update"] for e in events] == list(range(1, 21))
    assert evidence["input_sha256"] == [batch_sha] * 20
    assert evidence["initial_state_sha256"] != evidence["final_state_sha256"]
    for expected, actual in zip(expected_losses, events, strict=True):
        assert actual["loss"] == pytest.approx(expected, abs=2e-4)
        assert actual["maximum_descriptor_disagreement"] <= 2e-5
        assert actual["elapsed_ns"] > 0
    for a, b in zip(direct.parameters(), replay.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=2e-5, rtol=2e-5)
    if teacher is not None:
        assert all(p.grad is None and not p.requires_grad for p in teacher.parameters())


def test_language_arm_rejects_paired_input_drift_before_first_update() -> None:
    model = _image_model()
    before = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pixels, labels = torch.randn(6, 4), torch.tensor([0, 0, 1, 1, 3, 3])
    gram = standardized_text_gram(torch.nn.functional.normalize(torch.randn(4, 5), dim=1))
    events: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="input"):
        subject.train_language_arm(
            model,
            None,
            lambda update: (pixels, labels),
            arm="correct",
            gram=gram,
            expected_input_hashes=["0" * 64] * 20,
            microbatch_size=2,
            progress=events.append,
            synchronize=lambda: None,
        )
    assert events == []
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


def test_disposable_preflight_measures_only_six_updates_and_preserves_initialization() -> None:
    initial = _image_model()
    before = {k: v.detach().clone() for k, v in initial.state_dict().items()}
    instances: list[PooledProxyAnchorModel] = []

    def factory() -> PooledProxyAnchorModel:
        model = copy.deepcopy(initial)
        instances.append(model)
        return model

    pixels, labels = torch.randn(6, 4), torch.tensor([0, 0, 1, 1, 3, 3])
    gram = standardized_text_gram(torch.nn.functional.normalize(torch.randn(4, 5), dim=1))
    events: list[dict[str, Any]] = []
    result = subject.measure_language_preflight(
        factory,
        None,
        lambda update: (pixels, labels),
        gram=gram,
        microbatch_size=2,
        progress=events.append,
        synchronize=lambda: None,
    )
    assert len(instances) == 2 and instances[0] is not instances[1]
    assert [(e["arm"], e["update"]) for e in events] == [
        ("base", 1),
        ("base", 2),
        ("base", 3),
        ("correct", 1),
        ("correct", 2),
        ("correct", 3),
    ]
    assert len(result["update_seconds"]) == 6 and all(t > 0 for t in result["update_seconds"])
    assert (
        result["arms"]["base"]["initial_state_sha256"]
        == result["arms"]["correct"]["initial_state_sha256"]
    )
    assert result["arms"]["base"]["input_sha256"] == result["arms"]["correct"]["input_sha256"]
    assert all(torch.equal(before[k], v) for k, v in initial.state_dict().items())


def test_disposable_preflight_rejects_a_factory_that_advances_initial_weights() -> None:
    initial = _image_model()
    instances = 0

    def factory() -> PooledProxyAnchorModel:
        nonlocal instances
        model = copy.deepcopy(initial)
        if instances:
            with torch.no_grad():
                model.proxies.add_(0.01)
        instances += 1
        return model

    pixels, labels = torch.randn(6, 4), torch.tensor([0, 0, 1, 1, 3, 3])
    gram = standardized_text_gram(torch.nn.functional.normalize(torch.randn(4, 5), dim=1))
    events: list[dict[str, Any]] = []
    with pytest.raises(ValueError, match="initial"):
        subject.measure_language_preflight(
            factory,
            None,
            lambda update: (pixels, labels),
            gram=gram,
            microbatch_size=2,
            progress=events.append,
            synchronize=lambda: None,
        )
    assert len(events) == 3


def _trained_fixture() -> tuple[PooledProxyAnchorModel, dict[str, Any]]:
    model = _image_model()
    pixels, labels = torch.randn(6, 4), torch.tensor([0, 0, 1, 1, 3, 3])
    result = subject.train_language_arm(
        model,
        None,
        lambda update: (pixels, labels),
        arm="base",
        gram=None,
        expected_input_hashes=None,
        microbatch_size=2,
        progress=lambda event: None,
        synchronize=lambda: None,
    )
    return model, cast(dict[str, Any], result)


def test_language_checkpoint_seal_requires_final_twenty_step_state(tmp_path: Path) -> None:
    model, evidence = _trained_fixture()
    path = tmp_path / "base-final.pt"
    seal = subject.seal_language_checkpoint(path, model, evidence, target_sha256="a" * 64)
    assert seal["sha256"] == _sha(path) and seal["bytes"] == path.stat().st_size
    assert seal["basename"] == "base-final.pt" and seal["completed_updates"] == 20
    payload = torch.load(path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "sfora-siglip-language-final-v1"
    assert payload["target_sha256"] == "a" * 64
    assert payload["final_state_sha256"] == evidence["final_state_sha256"]
    assert payload["completed_updates"] == 20 and payload["arm"] == "base"
    assert payload["claim_eligible"] is False
    assert all(torch.equal(v, payload["model_state"][k]) for k, v in model.state_dict().items())
    with pytest.raises(FileExistsError):
        subject.seal_language_checkpoint(path, model, evidence, target_sha256="a" * 64)
    assert _sha(path) == seal["sha256"]


@pytest.mark.parametrize(
    "mutation", ["partial", "order", "inputs", "state", "target", "arm", "nan-loss"]
)
def test_checkpoint_rejects_incomplete_or_drifted_evidence_before_write(
    tmp_path: Path,
    mutation: str,
) -> None:
    model, evidence = _trained_fixture()
    target = "a" * 64
    if mutation == "partial":
        evidence["completed_updates"] = 3
    elif mutation == "order":
        evidence["steps"][1]["update"] = 1
    elif mutation == "inputs":
        evidence["input_sha256"].pop()
    elif mutation == "state":
        with torch.no_grad():
            model.proxies.add_(0.01)
    elif mutation == "target":
        target = "invalid"
    elif mutation == "arm":
        evidence["arm"] = "unknown"
    else:
        evidence["steps"][0]["loss"] = float("nan")
    path = tmp_path / "base-final.pt"
    with pytest.raises(ValueError):
        subject.seal_language_checkpoint(path, model, evidence, target_sha256=target)
    assert not path.exists()


def test_checkpoint_cannot_claim_an_unconsumed_language_target(tmp_path: Path) -> None:
    model, evidence = _trained_fixture()
    # Base consumed no Gram. A valid-looking digest must not be accepted as its target.
    path = tmp_path / "base-final.pt"
    with pytest.raises(ValueError, match="Gram"):
        subject.seal_language_checkpoint(
            path,
            model,
            evidence,
            target_sha256="a" * 64,
            expected_gram_sha256="b" * 64,
        )
    assert not path.exists()


def _target_fixture() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...], dict[str, str]]:
    generator = torch.Generator().manual_seed(17)
    vectors = torch.nn.functional.normalize(torch.randn(49, 1152, generator=generator), dim=1)
    ids = torch.arange(49 * 64, dtype=torch.int64).reshape(49, 64) % 32000
    prompts = tuple(f"a photo of a training class {i}." for i in range(49))
    sources = {
        k: "a" * 64
        for k in (
            "model",
            "config",
            "tokenizer",
            "tokenizer_config",
            "spiece",
            "special_tokens",
            "dataset_info",
            "runner",
            "guidance",
            "protocol",
        )
    }
    return vectors, ids, prompts, sources


def test_target_bundle_binds_both_axes_and_roundtrips_before_training(tmp_path: Path) -> None:
    vectors, ids, prompts, sources = _target_fixture()
    receipt = subject.seal_language_targets(
        tmp_path,
        vectors,
        ids,
        prompts,
        sources,
        expected_tokens_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        elapsed_seconds=1.25,
    )
    path = tmp_path / "language-targets.json"
    loaded = subject.load_language_targets(path, _sha(path))
    assert loaded["receipt"] == receipt
    assert loaded["receipt"]["claim_eligible"] is False
    assert loaded["receipt"]["quality_measured"] is False
    assert loaded["receipt"]["class_ids"] == list(range(49))
    assert loaded["receipt"]["prompts"] == list(prompts)
    assert loaded["receipt"]["input_sha256"] == sources
    expected = standardized_text_gram(vectors)
    permutation = torch.tensor(
        [
            15,
            6,
            24,
            23,
            43,
            13,
            40,
            39,
            21,
            42,
            33,
            14,
            7,
            11,
            16,
            1,
            19,
            20,
            29,
            8,
            32,
            17,
            27,
            45,
            46,
            12,
            48,
            41,
            0,
            30,
            38,
            4,
            25,
            31,
            2,
            3,
            28,
            18,
            36,
            34,
            26,
            47,
            10,
            5,
            35,
            9,
            22,
            37,
            44,
        ]
    )
    assert torch.equal(loaded["tensors"]["correct"], expected)
    assert torch.equal(loaded["tensors"]["permuted"], expected[permutation][:, permutation])
    assert not torch.equal(loaded["tensors"]["correct"], loaded["tensors"]["permuted"])
    assert torch.equal(loaded["tensors"]["input_ids"], ids)
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        subject.seal_language_targets(
            tmp_path,
            vectors,
            ids,
            prompts,
            sources,
            expected_tokens_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
            elapsed_seconds=1.25,
        )
    assert path.read_bytes() == original


@pytest.mark.parametrize("mutation", ["tokens", "vectors", "names", "source", "elapsed"])
def test_target_bundle_refuses_invalid_inputs_before_any_write(
    tmp_path: Path, mutation: str
) -> None:
    vectors, ids, prompts, sources = _target_fixture()
    expected_tokens = hashlib.sha256(ids.numpy().tobytes()).hexdigest()
    elapsed = 1.25
    if mutation == "tokens":
        ids[0, 0] += 1
    elif mutation == "vectors":
        vectors[0, 0] = float("nan")
    elif mutation == "names":
        prompts += ("a photo of an evaluation class.",)
    elif mutation == "source":
        sources.pop("model")
    else:
        elapsed = float("inf")
    with pytest.raises(ValueError):
        subject.seal_language_targets(
            tmp_path,
            vectors,
            ids,
            prompts,
            sources,
            expected_tokens_sha256=expected_tokens,
            elapsed_seconds=elapsed,
        )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("mutation", ["missing", "digest", "blob", "permutation", "duplicate"])
def test_target_loader_rejects_partial_or_coherently_rehashed_control_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    from safetensors.torch import load_file

    vectors, ids, prompts, sources = _target_fixture()
    receipt = subject.seal_language_targets(
        tmp_path,
        vectors,
        ids,
        prompts,
        sources,
        expected_tokens_sha256=hashlib.sha256(ids.numpy().tobytes()).hexdigest(),
        elapsed_seconds=1.25,
    )
    path = tmp_path / "language-targets.json"
    blob = tmp_path / "language-targets.safetensors"
    if mutation == "missing":
        blob.unlink()
    elif mutation == "digest":
        receipt["tensors_sha256"] = "b" * 64
    elif mutation == "blob":
        blob.write_bytes(b"corrupt")
    elif mutation == "permutation":
        receipt["permutation"][0], receipt["permutation"][1] = (
            receipt["permutation"][1],
            receipt["permutation"][0],
        )
    else:
        tensors = load_file(blob)
        tensors["permuted"] = tensors["correct"].clone()
        save_file(tensors, blob)
        receipt["tensors_sha256"] = _sha(blob)
        receipt["tensors_bytes"] = blob.stat().st_size
    path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.load_language_targets(path, _sha(path))
