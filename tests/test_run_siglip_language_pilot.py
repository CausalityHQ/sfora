"""Real local tensor containers exercise text-state and prompt authority."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from safetensors.torch import save_file
from transformers import BatchEncoding, SiglipTextConfig, SiglipTextModel

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


def test_evaluation_live_source_gate_uses_only_clean_evaluator_dependencies() -> None:
    assert Path(subject.evaluator.__file__).name == "siglip_language_evaluation.py"
    assert _sha(Path(subject.evaluator.__file__)) == subject.LANGUAGE_EVALUATOR_SHA256
    assert (
        _sha(Path(subject.evaluation_core.__file__))
        == subject.LANGUAGE_EVALUATION_SOURCE_SHA256["evaluation_core"]
    )
    assert not hasattr(subject, "retrieval_core")


def test_real_retrieval_evidence_reproduces_and_is_canonical_json_native(tmp_path: Path) -> None:
    vectors = torch.nn.functional.normalize(
        torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 1.0, 0.0],
                [0.1, 0.9, 0.0],
            ]
        ),
        dim=1,
    )
    cell = subject.evaluator._retrieval_cell(vectors, (0, 0, 1, 1))
    reproduction = subject.evaluator.require_teacher_reproduction(cell, cell["retrieval"])
    assert reproduction == {
        "aggregate_reproduced": True,
        "per_query_bitwise_reproduced": True,
        "first_differing_ordinal": None,
    }
    output = tmp_path / "evaluation.json"
    subject._publish_language_evaluation(
        output,
        {"cell": cell},
        deadline_ns=10**30,
    )
    assert output.read_bytes() == subject.control._canonical_bytes({"cell": cell})


def test_recovery_dependency_gate_rejects_live_source_drift() -> None:
    expected = subject.evaluator.recovery_dependency_sha256()
    subject.evaluator.verify_recovery_dependencies(expected)
    drifted = dict(expected)
    drifted["depth_core"] = "f" * 64
    with pytest.raises(ValueError, match="dependencies"):
        subject.evaluator.verify_recovery_dependencies(drifted)


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


def test_text_header_matches_model_without_materializing_tensor_payloads(tmp_path: Path) -> None:
    model = _text_model()
    path = tmp_path / "model.safetensors"
    _container(path, model)
    subject.validate_text_header(path, _sha(path), model)
    tensors = _container(path, model)
    key = "text_model.embeddings.token_embedding.weight"
    tensors[key] = tensors[key].half()
    save_file(tensors, path)
    with pytest.raises(ValueError, match="header"):
        subject.validate_text_header(path, _sha(path), model)


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
        "monitor_sha256": "2b542c65ab9ec644693559fafddcfdccbddff1aedf297307fe2abb2d22792987",
        "script_sha256": "140a8160dc81c3585247ff5e7227e87a99f8ffff6839fb0f6ba11db601e3072d",
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


def test_initialization_authenticates_complete_pair_chain_and_ignores_copied_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation, evaluation_monitor = _selection_evidence()
    evaluation["selected_arm"] = "relational"  # copied convenience field is not authority
    pair_receipt: dict[str, Any] = {
        "resources": {},
        "arms": {},
        "checkpoints": {},
        "dependencies": subject.evaluator.recovery_dependency_sha256(),
    }
    pair_monitor = {"schema": "fixture-pair-monitor"}
    smoke = {"schema": "fixture-smoke"}

    def write(name: str, value: dict[str, Any]) -> tuple[Path, str]:
        path = tmp_path / name
        path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
        return path, _sha(path)

    pair_path, pair_sha = write("pair-complete.json", pair_receipt)
    pair_monitor_path, pair_monitor_sha = write("pair-monitor.json", pair_monitor)
    evaluation["pair_sha256"] = pair_sha
    evaluation["monitor_sha256"] = pair_monitor_sha
    evaluation_path, evaluation_sha = write("evaluation.json", evaluation)
    evaluation_monitor["result_sha256"] = evaluation_sha
    evaluation_monitor["pair_monitor_sha256"] = pair_monitor_sha
    evaluation_monitor_path, evaluation_monitor_sha = write(
        "evaluation-monitor.json",
        evaluation_monitor,
    )
    smoke_path, smoke_sha = write("smoke.json", smoke)
    checkpoint_paths = {arm: tmp_path / f"{arm}-final.pt" for arm in ("pa", "relational")}
    for path in checkpoint_paths.values():
        path.write_bytes(b"payload")
    calls: list[tuple[str, Any]] = []

    def fake_smoke(path: Path, digest: str) -> dict[str, Any]:
        calls.append(("smoke", (path, digest)))
        return smoke

    def fake_budget(value: Any, monitor: Any, digest: str) -> float:
        calls.append(("pair-monitor", (monitor, digest)))
        return 1.0

    def fake_checkpoints(directory: Path, value: Any) -> dict[str, Path]:
        calls.append(("checkpoint-bytes", directory))
        return checkpoint_paths

    monkeypatch.setattr(
        subject.pair,
        "read_smoke_authority",
        fake_smoke,
    )
    monkeypatch.setattr(
        subject.evaluator,
        "validate_pair_receipt",
        lambda value, authority: calls.append(("pair", (value, authority))),
    )
    monkeypatch.setattr(
        subject.evaluator,
        "evaluation_budget_seconds",
        fake_budget,
    )
    monkeypatch.setattr(
        subject.evaluator,
        "authenticate_checkpoint_files",
        fake_checkpoints,
    )
    monkeypatch.setattr(subject.torch, "load", lambda path, **kwargs: {"path": path.name})
    monkeypatch.setattr(
        subject.evaluator,
        "validate_student_payload",
        lambda payload, value, arm: calls.append(("payload", (payload, arm))),
    )
    result = subject.read_initialization(
        SimpleNamespace(
            pair_directory=tmp_path,
            pair_sha256=pair_sha,
            pair_monitor=pair_monitor_path,
            pair_monitor_sha256=pair_monitor_sha,
            smoke_result=smoke_path,
            smoke_sha256=smoke_sha,
            evaluation=evaluation_path,
            evaluation_sha256=evaluation_sha,
            evaluation_monitor=evaluation_monitor_path,
            evaluation_monitor_sha256=evaluation_monitor_sha,
            teacher_checkpoint=tmp_path / "teacher.pt",
        )
    )
    assert result["selected_arm"] == "pa"
    assert result["initialization_path"] == checkpoint_paths["pa"]
    assert [c[0] for c in calls] == [
        "smoke",
        "pair",
        "pair-monitor",
        "checkpoint-bytes",
        "payload",
        "payload",
    ]
    assert result["choice"]["evaluation_projection_seconds"] == 910.0


def test_initialization_rejects_digest_before_any_tensor_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "pair-complete.json"
    path.write_text("{}\n")
    parsed: list[bool] = []
    monkeypatch.setattr(subject.torch, "load", lambda *a, **k: parsed.append(True))
    with pytest.raises(ValueError):
        subject.read_initialization(SimpleNamespace(pair_directory=tmp_path, pair_sha256="0" * 64))
    assert parsed == []


@pytest.mark.parametrize("selected", ["teacher", "pa", "relational"])
def test_language_initialization_factory_restores_only_the_recomputed_choice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
) -> None:
    full = _image_model().eval().requires_grad_(False)
    student = copy.deepcopy(full).train().requires_grad_(True)
    checkpoint = tmp_path / f"{selected}-final.pt"
    if selected != "teacher":
        torch.save({"model_state": student.state_dict()}, checkpoint)
    validations: list[tuple[str, str]] = []
    monkeypatch.setattr(subject.pair, "load_teacher", lambda root: copy.deepcopy(full))
    monkeypatch.setattr(subject, "prune_siglip_student", lambda model: copy.deepcopy(student))
    monkeypatch.setattr(
        subject.evaluator,
        "validate_student_payload",
        lambda payload, receipt, arm: validations.append((receipt["schema"], arm)),
    )
    factory, teacher, initial_sha = subject.language_initialization_factory(
        {
            "selected_arm": selected,
            "initialization_path": checkpoint,
            "pair_receipt": {"schema": "fixture-pair"},
        },
        tmp_path / "control",
        torch.device("cpu"),
    )
    first, second = factory(), factory()
    assert first is not second
    assert subject.control._model_state_sha256(first) == initial_sha
    assert subject.control._model_state_sha256(second) == initial_sha
    assert validations == ([] if selected == "teacher" else [("fixture-pair", selected)])
    assert (teacher is not None) is (selected == "relational")
    if teacher is not None:
        assert not any(p.requires_grad for p in teacher.parameters())


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


def test_execute_training_uses_fresh_equal_models_and_seals_all_three_arms(
    tmp_path: Path,
) -> None:
    initial = _image_model()
    instances: list[PooledProxyAnchorModel] = []

    def factory() -> PooledProxyAnchorModel:
        model = copy.deepcopy(initial)
        instances.append(model)
        return model

    pixels, labels = torch.randn(6, 4), torch.tensor([0, 0, 1, 1, 3, 3])
    correct = standardized_text_gram(torch.nn.functional.normalize(torch.randn(4, 5), dim=1))
    permuted = correct[[1, 0, 3, 2]][:, [1, 0, 3, 2]].contiguous()
    events: list[dict[str, Any]] = []
    result = subject.execute_language_training(
        factory,
        None,
        lambda update: (pixels, labels),
        output_dir=tmp_path,
        target_sha256="a" * 64,
        grams={"correct": correct, "permuted": permuted},
        gram_sha256={
            "correct": subject._gram_digest(correct),
            "permuted": subject._gram_digest(permuted),
        },
        spent_seconds=1.0,
        microbatch_size=2,
        progress=events.append,
        synchronize=lambda: None,
    )
    assert len(instances) == 5
    assert list(result["arms"]) == ["base", "correct", "permuted"]
    assert list(result["checkpoints"]) == ["base", "correct", "permuted"]
    assert all(path.exists() for path in (tmp_path / f"{arm}-final.pt" for arm in result["arms"]))
    initials = {evidence["initial_state_sha256"] for evidence in result["arms"].values()}
    inputs = {tuple(evidence["input_sha256"]) for evidence in result["arms"].values()}
    assert len(initials) == len(inputs) == 1
    assert result["projection_seconds"] <= 7200
    assert result["teacher_unchanged"] is True
    assert len(events) == 66


def test_execute_training_rejects_duplicate_controls_before_model_construction(
    tmp_path: Path,
) -> None:
    gram = standardized_text_gram(torch.nn.functional.normalize(torch.randn(4, 5), dim=1))
    calls: list[bool] = []
    with pytest.raises(ValueError, match="distinct"):
        subject.execute_language_training(
            lambda: calls.append(True),
            None,
            lambda update: (torch.randn(6, 4), torch.tensor([0, 0, 1, 1, 3, 3])),
            output_dir=tmp_path,
            target_sha256="a" * 64,
            grams={"correct": gram, "permuted": gram.clone()},
            gram_sha256={
                "correct": subject._gram_digest(gram),
                "permuted": subject._gram_digest(gram),
            },
            spent_seconds=1.0,
            microbatch_size=2,
            progress=lambda event: None,
            synchronize=lambda: None,
        )
    assert calls == [] and list(tmp_path.iterdir()) == []


def test_run_training_authenticates_then_trains_and_seals_one_terminal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "out"
    calls: list[str] = []
    gram = standardized_text_gram(torch.nn.functional.normalize(torch.randn(4, 5), dim=1))
    permuted = gram[[1, 0, 3, 2]][:, [1, 0, 3, 2]].contiguous()
    target_receipt = {
        "schema": "fixture-targets",
        "gram_sha256": {
            "correct": subject._gram_digest(gram),
            "permuted": subject._gram_digest(permuted),
        },
    }
    initialization = {
        "selected_arm": "pa",
        "pair_receipt": {
            "schema": "fixture-pair",
            "dependencies": subject.evaluator.recovery_dependency_sha256(),
        },
        "pair_sha256": "1" * 64,
        "pair_monitor_sha256": "2" * 64,
        "evaluation_sha256": "3" * 64,
        "evaluation_monitor_sha256": "4" * 64,
    }
    monkeypatch.setattr(
        subject,
        "read_initialization",
        lambda args: calls.append("initialization") or initialization,
    )
    monkeypatch.setattr(subject.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        subject.control,
        "require_control_determinism",
        lambda device: calls.append("determinism"),
    )

    def prepare(args: Any, device: torch.device, progress: Any) -> dict[str, Any]:
        calls.append("targets")
        (args.output_dir / "language-targets.json").write_bytes(b"{}\n")
        return target_receipt

    monkeypatch.setattr(subject, "prepare_text_targets", prepare)
    monkeypatch.setattr(
        subject,
        "load_language_targets",
        lambda path, digest: {
            "receipt": target_receipt,
            "tensors": {"correct": gram, "permuted": permuted},
        },
    )
    monkeypatch.setattr(
        subject,
        "language_initialization_factory",
        lambda *args: (lambda: _image_model(), None, "5" * 64),
    )
    monkeypatch.setattr(
        subject,
        "language_training_batch_source",
        lambda receipt: (
            calls.append("images") or (lambda update: (torch.randn(6, 4), torch.arange(6)))
        ),
    )
    monkeypatch.setattr(
        subject,
        "execute_language_training",
        lambda *args, **kwargs: (
            calls.append("training")
            or {
                "arms": {arm: {"completed_updates": 20} for arm in ("base", "correct", "permuted")},
                "checkpoints": {
                    arm: {"sha256": str(i) * 64}
                    for i, arm in enumerate(("base", "correct", "permuted"), 6)
                },
                "projection_seconds": 100.0,
                "initial_state_sha256": "5" * 64,
                "input_sha256": ["9" * 64] * 20,
                "teacher_state_sha256": None,
                "teacher_unchanged": True,
                "preflight": {"quality_measured": False},
            }
        ),
    )
    result = subject.run_training(SimpleNamespace(output_dir=output, control_root=tmp_path))
    assert calls[:4] == ["initialization", "determinism", "targets", "images"]
    assert calls[-1] == "training"
    assert result["schema"] == "sfora-siglip-language-training-v1"
    terminal = output / "training-complete.json"
    assert terminal.read_bytes() == subject.control._canonical_bytes(result)


def test_run_training_rejects_initialization_before_cuda_or_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cuda: list[bool] = []
    monkeypatch.setattr(
        subject,
        "read_initialization",
        lambda args: (_ for _ in ()).throw(ValueError("authority")),
    )
    monkeypatch.setattr(subject.torch.cuda, "is_available", lambda: cuda.append(True))
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="authority"):
        subject.run_training(SimpleNamespace(output_dir=output))
    assert cuda == [] and not output.exists()


def test_language_batch_source_replays_exact_authenticated_first_twenty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = [f"{index:064x}" for index in range(20)]
    receipt = {
        "arms": {arm: {"input_sha256": expected + ["f" * 64] * 178} for arm in ("pa", "relational")}
    }
    calls: list[int] = []
    monkeypatch.setattr(subject, "load_optimization_images", lambda: ("images",))
    monkeypatch.setattr(subject.smoke, "recovery_batches", lambda examples: tuple(range(198)))
    monkeypatch.setattr(subject.control, "build_control_train_transform", lambda: "transform")
    monkeypatch.setattr(
        subject.smoke,
        "paired_training_batch",
        lambda examples, batch, transform, update: (
            calls.append(update) or torch.tensor([float(update)]),
            torch.tensor([update]),
        ),
    )
    monkeypatch.setattr(
        subject.smoke,
        "_batch_sha",
        lambda pixels, labels: expected[int(labels[0]) - 1],
    )
    batch = subject.language_training_batch_source(receipt)
    assert int(batch(1)[1][0]) == 1
    assert int(batch(20)[1][0]) == 20
    assert calls == [1, 20]
    with pytest.raises(ValueError):
        batch(0)
    monkeypatch.setattr(subject.smoke, "_batch_sha", lambda pixels, labels: "e" * 64)
    with pytest.raises(ValueError, match="crops"):
        batch(1)


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


@pytest.mark.parametrize(
    "mutation", ["missing", "digest", "blob", "permutation", "duplicate", "receipt-symlink"]
)
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
        if mutation == "duplicate":
            tensors = load_file(blob)
            tensors["permuted"] = tensors["correct"].clone()
            save_file(tensors, blob)
            receipt["tensors_sha256"] = _sha(blob)
            receipt["tensors_bytes"] = blob.stat().st_size
        else:
            target = tmp_path / "language-targets-real.json"
            path.rename(target)
            path.symlink_to(target.name)
    if mutation != "receipt-symlink":
        path.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    with pytest.raises((ValueError, FileNotFoundError)):
        subject.load_language_targets(path, _sha(path))


def test_prepare_targets_authenticates_local_sources_and_never_uses_evaluation_names(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshot = tmp_path / "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
    snapshot.mkdir()
    files = {}
    for role, name in {
        "model": "model.safetensors",
        "config": "config.json",
        "tokenizer": "tokenizer.json",
        "tokenizer_config": "tokenizer_config.json",
        "spiece": "spiece.model",
        "special_tokens": "special_tokens_map.json",
    }.items():
        path = snapshot / name
        path.write_bytes(role.encode())
        files[role] = path
    dataset_info = tmp_path / "dataset_info.json"
    dataset_info.write_text(
        json.dumps(
            {
                "features": {
                    "label": {"_type": "ClassLabel", "names": [f"name-{i}" for i in range(196)]}
                }
            }
        )
    )
    files["dataset_info"] = dataset_info
    digests = {role: _sha(path) for role, path in files.items()}
    ids = _tokens()
    model = _text_model()
    model.config.hidden_size = 1152
    model.config.num_hidden_layers = 27
    model.config.vocab_size = 32000
    pooled = torch.nn.functional.normalize(torch.randn(49, 1152), dim=1)
    model.forward = lambda **kwargs: SimpleNamespace(pooler_output=pooled)  # type: ignore[method-assign]
    seen: dict[str, Any] = {}
    monkeypatch.setattr(subject, "SiglipTextModel", lambda config: model, raising=False)
    monkeypatch.setattr(
        subject,
        "SiglipTextConfig",
        SimpleNamespace(
            from_json_file=lambda path: SimpleNamespace(
                hidden_size=1152,
                num_hidden_layers=27,
                vocab_size=32000,
            ),
        ),
        raising=False,
    )
    monkeypatch.setattr(subject, "load_text_state", lambda *args: seen.update(loaded=args))
    monkeypatch.setattr(
        subject,
        "validate_text_header",
        lambda *args: seen.update(header=args),
    )

    class Tokenizer:
        def __call__(self, prompts: tuple[str, ...], **kwargs: Any) -> BatchEncoding:
            seen["prompts"] = prompts
            seen["tokenizer_kwargs"] = kwargs
            return BatchEncoding({"input_ids": ids})

    monkeypatch.setattr(
        subject,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda path, local_files_only: Tokenizer()),
        raising=False,
    )
    monkeypatch.setattr(
        subject,
        "seal_language_targets",
        lambda directory, vectors, input_ids, prompts, source, **kwargs: (
            seen.update(
                sealed=(directory, vectors, input_ids, prompts, source, kwargs),
            )
            or {"schema": "fixture-seal"}
        ),
    )
    args = SimpleNamespace(
        snapshot=snapshot,
        dataset_info=dataset_info,
        output_dir=tmp_path / "out",
        expected_tokens_sha256=_token_sha(ids),
        **{role: path for role, path in files.items() if role != "dataset_info"},
        **{f"{role}_sha256": digest for role, digest in digests.items()},
    )
    events: list[dict[str, Any]] = []
    result = subject.prepare_text_targets(args, torch.device("cpu"), events.append)
    assert result["schema"] == "fixture-seal"
    assert len(events) == 2 and events[0]["stage"] == "text-targets-authenticated"
    assert events[1]["stage"] == "text-targets-sealed"
    assert seen["prompts"] == tuple(f"a photo of a name-{i}." for i in range(49))
    assert all("name-49" not in prompt for prompt in seen["prompts"])
    assert seen["tokenizer_kwargs"] == {
        "padding": "max_length",
        "truncation": True,
        "max_length": 64,
        "return_tensors": "pt",
    }
    source = seen["sealed"][4]
    assert {k: source[k] for k in digests} == digests
    assert set(source) == {*digests, "runner", "guidance", "protocol"}


def test_prepare_targets_rejects_source_digest_before_model_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = tmp_path / "model.safetensors"
    model.write_bytes(b"real")
    constructed: list[bool] = []
    monkeypatch.setattr(
        subject,
        "SiglipTextModel",
        lambda config: constructed.append(True),
        raising=False,
    )
    with pytest.raises(ValueError):
        subject.prepare_text_targets(
            SimpleNamespace(model=model, model_sha256="0" * 64),
            torch.device("cpu"),
            lambda event: None,
        )
    assert constructed == []


def _language_train_arguments(tmp_path: Path) -> list[str]:
    paths = {
        name: tmp_path / name
        for name in (
            "pair-directory",
            "pair-monitor",
            "smoke-result",
            "evaluation",
            "evaluation-monitor",
            "control-root",
            "teacher-checkpoint",
            "snapshot",
            "model",
            "config",
            "tokenizer",
            "tokenizer-config",
            "spiece",
            "special-tokens",
            "dataset-info",
            "output-dir",
        )
    }
    arguments = ["train"]
    for name, path in paths.items():
        arguments.extend((f"--{name}", str(path)))
    for name in (
        "pair",
        "pair-monitor",
        "smoke",
        "evaluation",
        "evaluation-monitor",
        "model",
        "config",
        "tokenizer",
        "tokenizer-config",
        "spiece",
        "special-tokens",
        "dataset-info",
    ):
        arguments.extend((f"--{name}-sha256", "a" * 64))
    arguments.extend(("--expected-tokens-sha256", "b" * 64, "--execute-language-training"))
    return arguments


def test_cli_has_only_strict_separate_train_and_evaluate_phases(tmp_path: Path) -> None:
    train = subject.parse_args(_language_train_arguments(tmp_path))
    assert train.phase == "train"
    assert train.output_dir == tmp_path / "output-dir"
    assert train.execute_language_training is True

    evaluation = subject.parse_args(
        [
            "evaluate",
            "--training-directory",
            str(tmp_path / "training"),
            "--training-sha256",
            "c" * 64,
            "--training-monitor",
            str(tmp_path / "training-monitor.json"),
            "--training-monitor-sha256",
            "d" * 64,
            "--audit-result",
            str(tmp_path / "audit.json"),
            "--audit-sha256",
            "e" * 64,
            "--control-root",
            str(tmp_path / "control"),
            "--output",
            str(tmp_path / "result.json"),
            "--execute-language-evaluation",
        ]
    )
    assert evaluation.phase == "evaluate"
    assert evaluation.training_directory == tmp_path / "training"
    assert evaluation.execute_language_evaluation is True

    forbidden = _language_train_arguments(tmp_path) + ["--train-steps", "21"]
    with pytest.raises(SystemExit):
        subject.parse_args(forbidden)
    duplicate = _language_train_arguments(tmp_path) + ["--pair-sha256", "e" * 64]
    with pytest.raises(SystemExit):
        subject.parse_args(duplicate)
    equals_duplicate = _language_train_arguments(tmp_path) + ["--pair-sha256=" + "e" * 64]
    with pytest.raises(SystemExit):
        subject.parse_args(equals_duplicate)
    abbreviated = _language_train_arguments(tmp_path)
    abbreviated[abbreviated.index("--output-dir")] = "--output-d"
    with pytest.raises(SystemExit):
        subject.parse_args(abbreviated)


def test_main_dispatches_only_the_selected_phase(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[str] = []
    training = SimpleNamespace(
        phase="train",
        output_dir=tmp_path / "training",
        execute_language_training=True,
    )
    training.output_dir.mkdir()
    (training.output_dir / "training-complete.json").write_bytes(b"{}\n")
    monkeypatch.setattr(subject, "parse_args", lambda arguments: training)
    monkeypatch.setattr(subject, "run_training", lambda args: calls.append("train") or {})
    subject.main(["train"])
    assert calls == ["train"]
    assert "TRAINING COMPLETE" in capsys.readouterr().out

    output = tmp_path / "evaluation.json"
    evaluation = SimpleNamespace(
        phase="evaluate",
        output=output,
        execute_language_evaluation=True,
    )
    monkeypatch.setattr(subject, "parse_args", lambda arguments: evaluation)
    monkeypatch.setattr(
        subject,
        "run_pilot_evaluation",
        lambda args: (
            setattr(args, "_evaluation_deadline_ns", 10**30),
            calls.append("evaluate"),
            {"schema": "fixture-evaluation"},
        )[-1],
        raising=False,
    )
    subject.main(["evaluate"])
    assert calls == ["train", "evaluate"]
    assert evaluation._evaluation_process_started_ns == subject._PROCESS_STARTED_NS
    assert output.read_bytes() == subject.control._canonical_bytes({"schema": "fixture-evaluation"})
    assert "EVALUATION COMPLETE" in capsys.readouterr().out


def test_evaluation_publication_is_create_exclusive_under_a_destination_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "evaluation.json"
    args = SimpleNamespace(
        phase="evaluate",
        output=output,
        execute_language_evaluation=True,
    )
    monkeypatch.setattr(subject, "parse_args", lambda arguments: args)

    def race(_: Any) -> dict[str, Any]:
        args._evaluation_deadline_ns = 10**30
        output.write_bytes(b"concurrent-writer\n")
        return {"schema": "fixture-evaluation"}

    monkeypatch.setattr(subject, "run_pilot_evaluation", race)
    with pytest.raises(FileExistsError):
        subject.main(["evaluate"])
    assert output.read_bytes() == b"concurrent-writer\n"


@pytest.mark.parametrize("clock", [(100, 100), (99, 100)])
def test_evaluation_publication_fails_closed_when_deadline_expires(
    tmp_path: Path,
    clock: tuple[int, int],
) -> None:
    output = tmp_path / "evaluation.json"
    ticks = iter(clock)
    with pytest.raises(RuntimeError, match="wall cap"):
        subject._publish_language_evaluation(
            output,
            {"schema": "fixture-evaluation"},
            deadline_ns=100,
            clock=lambda: next(ticks),
        )
    assert not output.exists()


def test_evaluation_rejects_training_authority_before_loading_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    training = tmp_path / "training"
    training.mkdir()
    terminal = training / "training-complete.json"
    terminal.write_bytes(b"{}\n")
    monitor = tmp_path / "monitor.json"
    monitor.write_bytes(subject.control._canonical_bytes({"schema": "fixture-monitor"}))
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"{}\n")
    monkeypatch.setattr(subject.evaluator, "AUDIT_SHA256", _sha(audit))
    loaded: list[bool] = []
    monkeypatch.setattr(
        subject.evaluation_core,
        "load_recovery_evaluation_images",
        lambda: loaded.append(True),
    )
    with pytest.raises(ValueError, match="training"):
        subject.run_pilot_evaluation(
            SimpleNamespace(
                training_directory=training,
                training_sha256=_sha(terminal),
                training_monitor=monitor,
                training_monitor_sha256=_sha(monitor),
                audit_result=audit,
                control_root=tmp_path / "control",
            )
        )
    assert loaded == []


def _language_evaluation_training_fixture(
    directory: Path,
    *,
    selected: str = "teacher",
) -> tuple[Path, Path]:
    directory.mkdir()
    torch.manual_seed(17)
    model = PooledProxyAnchorModel(
        tower=torch.nn.Sequential(torch.nn.Linear(4, 1152), torch.nn.Tanh()),
        input_dimensions=1152,
        embedding_dimensions=512,
        class_count=49,
    )
    initial_state_sha256 = subject.control._model_state_sha256(model)
    torch.save(model.state_dict(), directory / "teacher-initial.pt")
    with torch.no_grad():
        model.proxies.add_(0.001)
    state_sha256 = subject.control._model_state_sha256(model)
    assert state_sha256 != initial_state_sha256
    vectors = torch.nn.functional.normalize(torch.randn(49, 1152), dim=1)
    input_ids = torch.arange(49 * 64, dtype=torch.int64).reshape(49, 64) % 32000
    tokens_sha256 = hashlib.sha256(input_ids.numpy().tobytes()).hexdigest()
    target_receipt = subject.seal_language_targets(
        directory,
        vectors,
        input_ids,
        tuple(f"a photo of a fixture car {index}." for index in range(49)),
        {
            "model": "1" * 64,
            "config": "2" * 64,
            "tokenizer": "3" * 64,
            "tokenizer_config": "4" * 64,
            "spiece": "5" * 64,
            "special_tokens": "6" * 64,
            "dataset_info": "7" * 64,
            "runner": _sha(Path(subject.__file__)),
            "guidance": _sha(Path(subject.language_guidance.__file__)),
            "protocol": _sha(Path(subject.language_protocol.__file__)),
        },
        expected_tokens_sha256=tokens_sha256,
        elapsed_seconds=1.0,
    )
    target_sha256 = _sha(directory / "language-targets.json")
    assert target_receipt["gram_sha256"]["correct"] != target_receipt["gram_sha256"]["permuted"]
    checkpoints: dict[str, dict[str, Any]] = {}
    arm_summaries: dict[str, dict[str, Any]] = {}
    for arm in ("base", "correct", "permuted"):
        path = directory / f"{arm}-final.pt"
        payload = {
            "schema": "sfora-siglip-language-final-v1",
            "claim_eligible": False,
            "seed": 17,
            "arm": arm,
            "completed_updates": 20,
            "target_sha256": target_sha256,
            "consumed_gram_sha256": None if arm == "base" else target_receipt["gram_sha256"][arm],
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": state_sha256,
            "input_sha256": [f"{index:064x}" for index in range(20)],
            "input_dimensions": model.projection.in_features,
            "embedding_dimensions": model.projection.out_features,
            "class_count": model.class_count,
            "model_state": model.state_dict(),
        }
        torch.save(payload, path)
        checkpoints[arm] = {
            "basename": path.name,
            "sha256": _sha(path),
            "bytes": path.stat().st_size,
            "arm": arm,
            "completed_updates": 20,
        }
        arm_summaries[arm] = {
            "arm": arm,
            "completed_updates": 20,
            "initial_state_sha256": initial_state_sha256,
            "final_state_sha256": state_sha256,
            "steps": [
                {
                    "arm": arm,
                    "update": index,
                    "elapsed_ns": 1,
                    "input_sha256": f"{index - 1:064x}",
                    "loss": 1.0,
                    "proxy_loss": 1.0,
                    "relational_loss": 1.0 if selected == "relational" else 0.0,
                    "language_loss": 0.0 if arm == "base" else 1.0,
                    "gradient_norm": 1.0,
                    "maximum_descriptor_disagreement": 0.0,
                    "lr_multiplier": subject.recovery_multiplier(index),
                }
                for index in range(1, 21)
            ],
            "input_sha256": [f"{index:064x}" for index in range(20)],
            "consumed_gram_sha256": payload["consumed_gram_sha256"],
            "teacher_state_sha256": initial_state_sha256 if selected == "relational" else None,
            "teacher_unchanged": True,
        }
    receipt = {
        "schema": "sfora-siglip-language-training-v1",
        "claim_eligible": False,
        "quality_measured": False,
        "seed": 17,
        "status": "complete",
        "selected_initialization": selected,
        "pair_sha256": "1" * 64,
        "pair_monitor_sha256": "2" * 64,
        "evaluation_sha256": "3" * 64,
        "evaluation_monitor_sha256": "4" * 64,
        "recovery_dependencies": subject.evaluator.recovery_dependency_sha256(),
        "target_sha256": target_sha256,
        "runner_sha256": _sha(Path(subject.__file__)),
        "elapsed_seconds": 100.0,
        "arms": arm_summaries,
        "checkpoints": checkpoints,
        "preflight": {
            "arms": {
                arm: {
                    **arm_summaries[arm],
                    "completed_updates": 3,
                    "steps": arm_summaries[arm]["steps"][:3],
                    "input_sha256": arm_summaries[arm]["input_sha256"][:3],
                }
                for arm in ("base", "correct")
            },
            "update_seconds": [1e-9] * 6,
            "quality_measured": False,
        },
        "projection_seconds": 3600.0,
        "initial_state_sha256": initial_state_sha256,
        "input_sha256": [f"{index:064x}" for index in range(20)],
        "teacher_state_sha256": initial_state_sha256 if selected == "relational" else None,
        "teacher_unchanged": True,
    }
    terminal = directory / "training-complete.json"
    terminal.write_bytes(subject.control._canonical_bytes(receipt))
    monitor = directory.parent / "training-monitor.json"
    monitor.write_bytes(
        subject.control._canonical_bytes(
            {
                "schema": "sfora-siglip-language-training-monitor-v1",
                "claim_eligible": False,
                "exit_code": 0,
                "stop_reason": None,
                "result_sha256": _sha(terminal),
                "elapsed_s": 110.0,
            }
        )
    )
    return terminal, monitor


def test_evaluation_authenticates_all_three_final_payloads_before_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal, monitor = _language_evaluation_training_fixture(tmp_path / "training")
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"{}\n")
    monkeypatch.setattr(subject.evaluator, "AUDIT_SHA256", _sha(audit))
    teacher = PooledProxyAnchorModel(
        tower=torch.nn.Sequential(torch.nn.Linear(4, 1152), torch.nn.Tanh()),
        input_dimensions=1152,
        embedding_dimensions=512,
        class_count=49,
    )
    teacher.load_state_dict(
        torch.load(
            terminal.parent / "teacher-initial.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    monkeypatch.setattr(subject.evaluator, "evaluation_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        subject.evaluator,
        "load_teacher_and_processor",
        lambda root: (teacher, "processor"),
    )
    monkeypatch.setattr(
        subject.evaluation_core,
        "load_recovery_evaluation_images",
        lambda: (_ for _ in ()).throw(RuntimeError("IMAGES_REACHED")),
    )
    with pytest.raises(RuntimeError, match="IMAGES_REACHED"):
        subject.run_pilot_evaluation(
            SimpleNamespace(
                training_directory=terminal.parent,
                training_sha256=_sha(terminal),
                training_monitor=monitor,
                training_monitor_sha256=_sha(monitor),
                audit_result=audit,
                audit_sha256=_sha(audit),
                control_root=tmp_path / "control",
            )
        )


def test_evaluation_uses_monitored_elapsed_not_training_projection_as_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal, monitor = _language_evaluation_training_fixture(tmp_path / "training")
    receipt = json.loads(terminal.read_bytes())
    receipt["elapsed_seconds"] = 3_700.0
    receipt["projection_seconds"] = 3_600.0
    terminal.write_bytes(subject.control._canonical_bytes(receipt))
    monitor_value = json.loads(monitor.read_bytes())
    monitor_value["elapsed_s"] = 3_701.0
    monitor_value["result_sha256"] = _sha(terminal)
    monitor.write_bytes(subject.control._canonical_bytes(monitor_value))
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"{}\n")
    monkeypatch.setattr(subject.evaluator, "AUDIT_SHA256", _sha(audit))
    teacher = PooledProxyAnchorModel(
        tower=torch.nn.Sequential(torch.nn.Linear(4, 1152), torch.nn.Tanh()),
        input_dimensions=1152,
        embedding_dimensions=512,
        class_count=49,
    )
    teacher.load_state_dict(
        torch.load(
            terminal.parent / "teacher-initial.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    monkeypatch.setattr(subject.evaluator, "evaluation_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        subject.evaluator,
        "load_teacher_and_processor",
        lambda root: (teacher, "processor"),
    )
    monkeypatch.setattr(
        subject.evaluation_core,
        "load_recovery_evaluation_images",
        lambda: (_ for _ in ()).throw(RuntimeError("IMAGES_REACHED")),
    )
    with pytest.raises(RuntimeError, match="IMAGES_REACHED"):
        subject.run_pilot_evaluation(
            SimpleNamespace(
                training_directory=terminal.parent,
                training_sha256=_sha(terminal),
                training_monitor=monitor,
                training_monitor_sha256=_sha(monitor),
                audit_result=audit,
                audit_sha256=_sha(audit),
                control_root=tmp_path / "control",
            )
        )


@pytest.mark.parametrize("selected", ["teacher", "pa", "relational"])
def test_evaluation_recomputes_three_arm_decision_and_discordances(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    selected: str,
) -> None:
    terminal, monitor = _language_evaluation_training_fixture(
        tmp_path / "training",
        selected=selected,
    )
    ids = [f"image-{index}" for index in range(2746)]
    audit_value = {
        "common_image_ids": ids,
        "common_to_native": list(range(2746)),
        "decoded_native_sha256": "d" * 64,
        "reproductions": {
            "siglip-projected-17": {
                "retrieval": {
                    "fixture": True,
                    "labels": [49 + index % 33 for index in range(2746)],
                }
            }
        },
    }
    audit = tmp_path / "audit.json"
    audit.write_bytes(subject.control._canonical_bytes(audit_value))
    monkeypatch.setattr(subject.evaluator, "AUDIT_SHA256", _sha(audit))
    examples = tuple(
        SimpleNamespace(example_id=example_id, label=49 + index % 33)
        for index, example_id in enumerate(ids)
    )
    teacher = PooledProxyAnchorModel(
        tower=torch.nn.Sequential(torch.nn.Linear(4, 1152), torch.nn.Tanh()),
        input_dimensions=1152,
        embedding_dimensions=512,
        class_count=49,
    )
    teacher.load_state_dict(
        torch.load(
            terminal.parent / "teacher-initial.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    monkeypatch.setattr(subject.evaluator, "evaluation_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        subject.evaluator,
        "load_teacher_and_processor",
        lambda root: (teacher, "processor"),
    )
    pruned: list[bool] = []
    monkeypatch.setattr(
        subject,
        "prune_siglip_student",
        lambda model: pruned.append(True) or copy.deepcopy(model),
    )
    monkeypatch.setattr(
        subject.evaluation_core, "load_recovery_evaluation_images", lambda: examples
    )
    monkeypatch.setattr(subject.evaluator, "decoded_native_digest", lambda rows, order: "d" * 64)
    monkeypatch.setattr(
        subject.evaluator,
        "embed_recovery_model",
        lambda model, rows, processor, device, check_time: torch.nn.functional.normalize(
            torch.ones(2746, 512), dim=1
        ),
    )
    correctness = {
        "teacher": [True] * 2596 + [False] * 150,
        "base": [True] * 2596 + [False] * 150,
        "correct": [True] * 2611 + [False] * 135,
        "permuted": [True] * 2597 + [False] * 149,
    }
    maps = {"teacher": 0.7913744556922272, "base": 0.792, "correct": 0.793, "permuted": 0.7925}
    names = iter(("teacher", "base", "correct", "permuted"))
    monkeypatch.setattr(
        subject.evaluator,
        "_retrieval_cell",
        lambda vectors, labels: (
            lambda name: {
                "queries": 2746,
                "correct": sum(correctness[name]),
                "map_at_r": maps[name],
                "retrieval": {"correct": correctness[name]},
            }
        )(next(names)),
    )
    monkeypatch.setattr(
        subject.evaluator,
        "require_teacher_reproduction",
        lambda cell, baseline: {"aggregate_reproduced": True},
    )
    result = subject.run_pilot_evaluation(
        SimpleNamespace(
            training_directory=terminal.parent,
            training_sha256=_sha(terminal),
            training_monitor=monitor,
            training_monitor_sha256=_sha(monitor),
            audit_result=audit,
            audit_sha256=_sha(audit),
            control_root=tmp_path / "control",
        )
    )
    assert result["schema"] == "sfora-siglip-language-evaluation-v1"
    assert result["selected_initialization"] == selected
    assert len(pruned) == (0 if selected == "teacher" else 3)
    assert result["decision"]["passed"] is True
    assert result["cells"]["correct"]["correct"] == 2611
    assert result["paired_discordances"]["correct-vs-base"] == {
        "both_correct": 2596,
        "control_only": 0,
        "correct_only": 15,
        "both_wrong": 135,
    }
    assert result["paired_discordances"]["correct-vs-permuted"]["correct_only"] == 14
    assert "search_profile" not in result


def test_evaluation_rejects_audit_label_drift_before_embedding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    terminal, monitor = _language_evaluation_training_fixture(tmp_path / "training")
    ids = [f"image-{index}" for index in range(2746)]
    audit = tmp_path / "audit.json"
    audit.write_bytes(
        subject.control._canonical_bytes(
            {
                "common_image_ids": ids,
                "common_to_native": list(range(2746)),
                "decoded_native_sha256": "d" * 64,
                "reproductions": {
                    "siglip-projected-17": {
                        "retrieval": {"labels": [50] * 2746},
                    }
                },
            }
        )
    )
    monkeypatch.setattr(subject.evaluator, "AUDIT_SHA256", _sha(audit))
    examples = tuple(
        SimpleNamespace(example_id=example_id, label=49 + index % 33)
        for index, example_id in enumerate(ids)
    )
    teacher = PooledProxyAnchorModel(
        tower=torch.nn.Sequential(torch.nn.Linear(4, 1152), torch.nn.Tanh()),
        input_dimensions=1152,
        embedding_dimensions=512,
        class_count=49,
    )
    teacher.load_state_dict(
        torch.load(
            terminal.parent / "teacher-initial.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    monkeypatch.setattr(subject.evaluator, "evaluation_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        subject.evaluator,
        "load_teacher_and_processor",
        lambda root: (teacher, "processor"),
    )
    monkeypatch.setattr(
        subject.evaluation_core,
        "load_recovery_evaluation_images",
        lambda: examples,
    )
    monkeypatch.setattr(subject.evaluator, "decoded_native_digest", lambda rows, order: "d" * 64)
    monkeypatch.setattr(
        subject.evaluator,
        "embed_recovery_model",
        lambda *args: (_ for _ in ()).throw(RuntimeError("EMBED_REACHED")),
    )
    with pytest.raises(ValueError, match="audit"):
        subject.run_pilot_evaluation(
            SimpleNamespace(
                training_directory=terminal.parent,
                training_sha256=_sha(terminal),
                training_monitor=monitor,
                training_monitor_sha256=_sha(monitor),
                audit_result=audit,
                audit_sha256=_sha(audit),
                control_root=tmp_path / "control",
            )
        )


@pytest.mark.parametrize(
    "mutation",
    ["selected", "digest", "arms", "projection", "teacher-state", "teacher-state-type"],
)
def test_evaluation_rejects_training_semantic_drift_before_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    terminal, monitor = _language_evaluation_training_fixture(tmp_path / "training")
    receipt = json.loads(terminal.read_bytes())
    if mutation == "selected":
        receipt["selected_initialization"] = "unknown"
    elif mutation == "digest":
        receipt["pair_sha256"] = "not-a-digest"
    elif mutation == "arms":
        del receipt["arms"]["permuted"]
    elif mutation == "projection":
        receipt["projection_seconds"] = 7200.0001
    elif mutation == "teacher-state":
        receipt["teacher_state_sha256"] = "f" * 64
    else:
        receipt["teacher_state_sha256"] = False
    terminal.write_bytes(subject.control._canonical_bytes(receipt))
    monitor_value = json.loads(monitor.read_bytes())
    monitor_value["result_sha256"] = _sha(terminal)
    monitor.write_bytes(subject.control._canonical_bytes(monitor_value))
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"{}\n")
    loaded: list[bool] = []
    monkeypatch.setattr(
        subject.evaluation_core,
        "load_recovery_evaluation_images",
        lambda: loaded.append(True),
    )
    with pytest.raises(ValueError, match="training"):
        subject.run_pilot_evaluation(
            SimpleNamespace(
                training_directory=terminal.parent,
                training_sha256=_sha(terminal),
                training_monitor=monitor,
                training_monitor_sha256=_sha(monitor),
                audit_result=audit,
                audit_sha256=_sha(audit),
                control_root=tmp_path / "control",
            )
        )
    assert loaded == []


@pytest.mark.parametrize(
    "mutation",
    [
        "arm-summary",
        "step-order",
        "step-semantics",
        "elapsed-under-steps",
        "preflight-step",
        "monitor-elapsed",
        "audit-digest",
        "target-bundle",
        "teacher-model",
        "recovery-dependency",
    ],
)
def test_evaluation_rejects_cross_receipt_drift_before_images(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    terminal, monitor = _language_evaluation_training_fixture(tmp_path / "training")
    if mutation == "arm-summary":
        receipt = json.loads(terminal.read_bytes())
        receipt["arms"]["correct"]["final_state_sha256"] = "f" * 64
        terminal.write_bytes(subject.control._canonical_bytes(receipt))
        monitor_value = json.loads(monitor.read_bytes())
        monitor_value["result_sha256"] = _sha(terminal)
        monitor.write_bytes(subject.control._canonical_bytes(monitor_value))
    elif mutation == "step-order":
        receipt = json.loads(terminal.read_bytes())
        receipt["arms"]["correct"]["steps"][0]["update"] = 2
        terminal.write_bytes(subject.control._canonical_bytes(receipt))
        monitor_value = json.loads(monitor.read_bytes())
        monitor_value["result_sha256"] = _sha(terminal)
        monitor.write_bytes(subject.control._canonical_bytes(monitor_value))
    elif mutation == "step-semantics":
        receipt = json.loads(terminal.read_bytes())
        receipt["arms"]["base"]["steps"][0]["language_loss"] = 0.25
        terminal.write_bytes(subject.control._canonical_bytes(receipt))
        monitor_value = json.loads(monitor.read_bytes())
        monitor_value["result_sha256"] = _sha(terminal)
        monitor.write_bytes(subject.control._canonical_bytes(monitor_value))
    elif mutation == "elapsed-under-steps":
        receipt = json.loads(terminal.read_bytes())
        receipt["arms"]["base"]["steps"][0]["elapsed_ns"] = 101_000_000_000
        terminal.write_bytes(subject.control._canonical_bytes(receipt))
        monitor_value = json.loads(monitor.read_bytes())
        monitor_value["result_sha256"] = _sha(terminal)
        monitor.write_bytes(subject.control._canonical_bytes(monitor_value))
    elif mutation == "preflight-step":
        receipt = json.loads(terminal.read_bytes())
        receipt["preflight"]["arms"]["correct"]["steps"][0]["update"] = 2
        terminal.write_bytes(subject.control._canonical_bytes(receipt))
        monitor_value = json.loads(monitor.read_bytes())
        monitor_value["result_sha256"] = _sha(terminal)
        monitor.write_bytes(subject.control._canonical_bytes(monitor_value))
    elif mutation == "monitor-elapsed":
        monitor_value = json.loads(monitor.read_bytes())
        monitor_value["elapsed_s"] = 99.0
        monitor.write_bytes(subject.control._canonical_bytes(monitor_value))
    elif mutation == "target-bundle":
        (terminal.parent / "language-targets.safetensors").unlink()
    elif mutation == "recovery-dependency":
        receipt = json.loads(terminal.read_bytes())
        receipt["recovery_dependencies"]["runner"] = "f" * 64
        terminal.write_bytes(subject.control._canonical_bytes(receipt))
        monitor_value = json.loads(monitor.read_bytes())
        monitor_value["result_sha256"] = _sha(terminal)
        monitor.write_bytes(subject.control._canonical_bytes(monitor_value))
    audit = tmp_path / "audit.json"
    audit.write_bytes(b"{}\n")
    monkeypatch.setattr(
        subject.evaluator,
        "AUDIT_SHA256",
        "e" * 64 if mutation == "audit-digest" else _sha(audit),
    )
    teacher = PooledProxyAnchorModel(
        tower=torch.nn.Sequential(torch.nn.Linear(4, 1152), torch.nn.Tanh()),
        input_dimensions=1152,
        embedding_dimensions=512,
        class_count=49,
    )
    teacher.load_state_dict(
        torch.load(
            terminal.parent / "teacher-initial.pt",
            map_location="cpu",
            weights_only=True,
        )
    )
    if mutation == "teacher-model":
        with torch.no_grad():
            teacher.proxies.add_(0.01)
    monkeypatch.setattr(subject.evaluator, "evaluation_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        subject.evaluator,
        "load_teacher_and_processor",
        lambda root: (teacher, "processor"),
    )
    monkeypatch.setattr(
        subject.evaluation_core,
        "load_recovery_evaluation_images",
        lambda: (_ for _ in ()).throw(RuntimeError("IMAGES_REACHED")),
    )
    with pytest.raises(ValueError, match="language"):
        subject.run_pilot_evaluation(
            SimpleNamespace(
                training_directory=terminal.parent,
                training_sha256=_sha(terminal),
                training_monitor=monitor,
                training_monitor_sha256=_sha(monitor),
                audit_result=audit,
                audit_sha256=_sha(audit),
                control_root=tmp_path / "control",
            )
        )
