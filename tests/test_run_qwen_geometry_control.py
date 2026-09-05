"""Behavioral tests for exact logical-batch Qwen geometry replay."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from sfora.qwen_geometry_control import learning_rate_multiplier
from sfora.token_set_proxy_anchor import proxy_anchor_loss

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_qwen_geometry_control.py"
_SPEC = importlib.util.spec_from_file_location("run_qwen_geometry_control", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


class _TinyDescriptor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.first = nn.Linear(5, 7)
        self.output = nn.Linear(7, 3, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.output(torch.tanh(self.first(inputs))), dim=-1)


class _TupleDescriptor(_TinyDescriptor):
    def forward(self, inputs: tuple[torch.Tensor, ...]) -> torch.Tensor:
        return super().forward(torch.stack(inputs))


def _fixture() -> tuple[nn.Module, nn.Parameter, torch.optim.AdamW]:
    torch.manual_seed(90210)
    model = _TinyDescriptor().double()
    proxies = nn.Parameter(torch.randn(3, 3, dtype=torch.float64))
    groups = [
        {
            "params": list(model.parameters()),
            "lr": 1.0e-3,
            "base_lr": 1.0e-3,
            "schedule_update": 0,
            "role": "model",
        },
        {
            "params": [proxies],
            "lr": 1.0e-2,
            "base_lr": 1.0e-2,
            "schedule_update": 0,
            "role": "proxies",
            "weight_decay": 0.0,
        },
    ]
    return model, proxies, torch.optim.AdamW(groups, weight_decay=1.0e-4)


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [
                [0.5, -0.2, 0.3, 0.8, -0.1],
                [-0.4, 0.7, 0.2, -0.3, 0.9],
                [0.6, 0.1, -0.8, 0.4, 0.2],
                [-0.2, -0.5, 0.9, 0.3, 0.7],
            ],
            dtype=torch.float64,
        ),
        torch.tensor([0, 1, 2, 0], dtype=torch.int64),
    )


def test_replayed_step_matches_full_batch_chain_rule_and_adamw_state() -> None:
    inputs, labels = _inputs()
    oracle_model, oracle_proxies, oracle_optimizer = _fixture()
    replay_model = copy.deepcopy(oracle_model)
    replay_proxies = nn.Parameter(oracle_proxies.detach().clone())
    replay_groups = copy.deepcopy(oracle_optimizer.param_groups)
    replay_groups[0]["params"] = list(replay_model.parameters())
    replay_groups[1]["params"] = [replay_proxies]
    replay_optimizer = torch.optim.AdamW(replay_groups)

    scores = oracle_model(inputs) @ F.normalize(oracle_proxies, dim=-1).T
    loss = proxy_anchor_loss(scores, labels, alpha=32.0, delta=0.1)
    (score_gradients,) = torch.autograd.grad(loss, scores, retain_graph=True)
    loss.backward()
    expected_gradients = tuple(
        parameter.grad.detach().clone()
        for parameter in (*oracle_model.parameters(), oracle_proxies)
    )
    expected_norm = torch.nn.utils.clip_grad_norm_(
        (*oracle_model.parameters(), oracle_proxies), 1.0
    )
    multiplier = learning_rate_multiplier(0)
    for group in oracle_optimizer.param_groups:
        group["lr"] = group["base_lr"] * multiplier
        group["schedule_update"] = 1
    oracle_optimizer.step()

    evidence = _MODULE.replayed_proxy_anchor_step(
        model=replay_model,
        proxies=replay_proxies,
        inputs=inputs,
        labels=labels,
        optimizer=replay_optimizer,
        microbatch_size=2,
        update_index=0,
    )

    assert type(evidence) is _MODULE.GeometryStepEvidence
    torch.testing.assert_close(evidence.scores, scores.detach(), rtol=1.0e-12, atol=1.0e-12)
    torch.testing.assert_close(
        evidence.score_gradients, score_gradients.detach(), rtol=1.0e-12, atol=1.0e-12
    )
    assert evidence.loss == pytest.approx(float(loss.detach()), rel=1.0e-12)
    assert evidence.gradient_norm == pytest.approx(float(expected_norm), rel=1.0e-10)
    assert evidence.sampled_parameter_values > 0
    assert 0 < evidence.changed_sampled_parameter_values <= evidence.sampled_parameter_values
    assert tuple(role for role, _, _ in evidence.parameter_displacement) == (
        "model",
        "proxies",
    )
    assert sum(total for _, total, _ in evidence.parameter_displacement) == (
        evidence.sampled_parameter_values
    )
    for actual, expected in zip(evidence.parameter_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1.0e-10, atol=1.0e-11)
    for actual, expected in zip(
        (*replay_model.parameters(), replay_proxies),
        (*oracle_model.parameters(), oracle_proxies),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected, rtol=1.0e-10, atol=1.0e-11)
    assert all(group["schedule_update"] == 1 for group in replay_optimizer.param_groups)
    assert evidence.updated_state_sha256 == _MODULE.state_sha256(
        (*replay_model.parameters(), replay_proxies)
    )


def test_replayed_step_is_byte_deterministic_from_restored_state() -> None:
    inputs, labels = _inputs()
    results = []
    for _ in range(2):
        model, proxies, optimizer = _fixture()
        results.append(
            _MODULE.replayed_proxy_anchor_step(
                model=model,
                proxies=proxies,
                inputs=inputs,
                labels=labels,
                optimizer=optimizer,
                microbatch_size=2,
                update_index=0,
            )
        )
    assert results[0].updated_state_sha256 == results[1].updated_state_sha256
    assert results[0].optimizer_state_sha256 == results[1].optimizer_state_sha256


def test_replayed_step_accepts_a_sliceable_non_tensor_image_batch() -> None:
    inputs, labels = _inputs()
    model, proxies, optimizer = _fixture()
    tuple_model = _TupleDescriptor().double()
    tuple_model.load_state_dict(model.state_dict())
    tuple_optimizer = torch.optim.AdamW(
        [
            {
                "params": list(tuple_model.parameters()),
                "lr": 1.0e-3,
                "base_lr": 1.0e-3,
                "schedule_update": 0,
            },
            {
                "params": [proxies],
                "lr": 1.0e-2,
                "base_lr": 1.0e-2,
                "schedule_update": 0,
                "weight_decay": 0.0,
            },
        ],
        weight_decay=1.0e-4,
    )

    evidence = _MODULE.replayed_proxy_anchor_step(
        model=tuple_model,
        proxies=proxies,
        inputs=tuple(inputs.unbind()),
        labels=labels,
        optimizer=tuple_optimizer,
        microbatch_size=2,
        update_index=0,
    )

    assert evidence.update_index == 0
    assert evidence.scores.shape == (4, 3)


def test_rejected_step_cannot_advance_schedule_or_mutate_state() -> None:
    inputs, labels = _inputs()
    model, proxies, optimizer = _fixture()
    before = _MODULE.state_sha256((*model.parameters(), proxies))
    inputs[0, 0] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        _MODULE.replayed_proxy_anchor_step(
            model=model,
            proxies=proxies,
            inputs=inputs,
            labels=labels,
            optimizer=optimizer,
            microbatch_size=2,
            update_index=0,
        )
    assert _MODULE.state_sha256((*model.parameters(), proxies)) == before
    assert all(group["schedule_update"] == 0 for group in optimizer.param_groups)

    for group in optimizer.param_groups:
        group["schedule_update"] = 1
    with pytest.raises(ValueError, match="schedule"):
        _MODULE.replayed_proxy_anchor_step(
            model=model,
            proxies=proxies,
            inputs=torch.nan_to_num(inputs),
            labels=labels,
            optimizer=optimizer,
            microbatch_size=2,
            update_index=0,
        )


class _FakeImageProcessor:
    def image_processor(
        self, *, images: list[torch.Tensor], return_tensors: str
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        return {
            "pixel_values": torch.stack(images),
            "image_grid_thw": torch.tensor([[1, 2, 2]] * len(images)),
        }


class _ForbiddenLanguage(nn.Module):
    def forward(self, _inputs: torch.Tensor) -> torch.Tensor:
        raise AssertionError("language execution is forbidden")


class _FakeVisual(nn.Module):
    spatial_merge_size = 2

    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(4, 4, bias=False)
        self.deepstack_merger_list = nn.ModuleList([nn.Linear(4, 4)])

    def forward(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert return_dict
        assert grid_thw.shape == (pixel_values.shape[0], 3)
        rows = self.projection(pixel_values.mean(dim=1))
        return SimpleNamespace(pooler_output=rows)


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = _FakeVisual()
        self.model.language_model = _ForbiddenLanguage()
        self.lm_head = _ForbiddenLanguage()

@pytest.mark.parametrize("arm", ["mean", "attention"])
def test_qwen_geometry_model_executes_only_vision_and_registered_pooler(arm: str) -> None:
    qwen = _FakeQwen()
    qwen.model.visual.to(dtype=torch.bfloat16)
    wrapper = _MODULE.QwenVisionGeometryModel(
        model=qwen,
        processor=_FakeImageProcessor(),
        token_dimensions=4,
        arm=arm,
    )
    images = tuple(torch.arange(16, dtype=torch.float32).reshape(4, 4) + row for row in range(2))

    tokens = wrapper.visual_tokens(images)
    descriptors = wrapper(images)

    assert tokens.shape == (2, 1, 4)
    assert descriptors.shape == (2, 4096)
    torch.testing.assert_close(
        torch.linalg.vector_norm(descriptors, dim=-1), torch.ones(2), rtol=0, atol=1.0e-6
    )
    assert all(not parameter.requires_grad for parameter in qwen.model.language_model.parameters())
    assert all(not parameter.requires_grad for parameter in qwen.lm_head.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in qwen.model.visual.deepstack_merger_list.parameters()
    )
    assert all(parameter.dtype == torch.float32 for parameter in wrapper.visual.parameters())
    assert "_qwen_model" not in wrapper.__dict__
    assert all(not isinstance(module, _ForbiddenLanguage) for module in wrapper.modules())
    wrapper_ids = {id(parameter) for parameter in wrapper.parameters() if parameter.requires_grad}
    assert wrapper_ids == {
        id(parameter)
        for parameter in (*qwen.model.visual.parameters(), *wrapper.pooler.parameters())
        if parameter.requires_grad
    }

    _MODULE._validate_qwen_parameter_roles(wrapper)
    wrapper.pooler.output.weight.requires_grad_(False)
    with pytest.raises(ValueError, match="role"):
        _MODULE._validate_qwen_parameter_roles(wrapper)


def test_full_tower_displacement_is_streamed_and_grouped_by_block() -> None:
    tower = nn.Module()
    tower.patch_embed = nn.Linear(2, 2, bias=False)
    tower.blocks = nn.ModuleList([nn.Linear(2, 2, bias=False) for _ in range(2)])
    baseline = _MODULE._capture_trainable_snapshot(tower)
    with torch.no_grad():
        tower.patch_embed.weight.add_(2.0e-6)
        tower.blocks[0].weight.add_(3.0e-6)

    summary = _MODULE._summarize_tower_displacement(tower, baseline)

    assert summary["changed_elements"] == 8
    assert summary["total_elements"] == 12
    assert summary["relative_l2"] > 1.0e-6
    assert tuple(block["block"] for block in summary["blocks"]) == (
        "blocks.0",
        "blocks.1",
        "patch_embed",
    )
    assert summary["moving_transformer_blocks"] == 1
    assert summary["transformer_blocks"] == 2
    assert summary["moving_block_fraction"] == 0.5


def test_visual_token_displacement_requires_functional_tower_change() -> None:
    before = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    unchanged = before.clone()
    after = before + 2.0e-5

    summary = _MODULE._summarize_token_displacement(before, unchanged, after)

    assert summary["unchanged_repeat_discrepancy"] == 0.0
    assert summary["relative_l2"] > 1.0e-6
    assert summary["passed"] is True


def test_smoke_cli_is_explicit_and_refuses_network_or_test_capabilities(
    tmp_path: Path,
) -> None:
    base = [
        "smoke",
        "--model-root",
        str(tmp_path),
        "--snapshot-manifest",
        str(tmp_path / "snapshot.json"),
        "--fixture",
        str(tmp_path / "fixture.json"),
        "--output",
        str(tmp_path / "result.json"),
        "--source-commit",
        "1" * 40,
        "--arm",
        "mean",
        "--seed",
        "17",
        "--microbatch-size",
        "1",
        "--execute-smoke",
    ]
    parsed = _MODULE.parse_args(base)
    assert parsed.arm == "mean"
    assert parsed.seed == 17
    for forbidden in (
        "--official-test",
        "--hub-token",
        "--dataset-uri",
        "--generate",
        "--language-model",
    ):
        with pytest.raises(SystemExit):
            _MODULE.parse_args([*base, forbidden, "value"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([token for token in base if token != "--execute-smoke"])

    (tmp_path / "result.json").write_text("occupied")
    with pytest.raises(SystemExit):
        _MODULE.parse_args(base)


def test_train_cli_requires_distinct_checkpoint_and_explicit_execution(tmp_path: Path) -> None:
    base = [
        "train",
        "--model-root",
        str(tmp_path),
        "--snapshot-manifest",
        str(tmp_path / "snapshot.json"),
        "--fixture",
        str(tmp_path / "fixture.json"),
        "--output",
        str(tmp_path / "receipt.json"),
        "--checkpoint-output",
        str(tmp_path / "checkpoint.pt"),
        "--source-commit",
        "1" * 40,
        "--arm",
        "attention",
        "--seed",
        "29",
        "--microbatch-size",
        "1",
        "--execute-train",
    ]
    parsed = _MODULE.parse_args(base)
    assert parsed.phase == "train"
    assert parsed.checkpoint_output == tmp_path / "checkpoint.pt"
    with pytest.raises(SystemExit):
        _MODULE.parse_args([token for token in base if token != "--execute-train"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*base, "--execute-smoke"])
    same_path = [
        str(tmp_path / "receipt.json") if token == str(tmp_path / "checkpoint.pt") else token
        for token in base
    ]
    with pytest.raises(SystemExit):
        _MODULE.parse_args(same_path)


def test_rgb_preprocessing_returns_an_owned_writable_224_array() -> None:
    image = Image.new("RGB", (8, 6), color=(1, 2, 3))
    value = _MODULE._rgb_224(image)
    assert value.shape == (224, 224, 3)
    assert value.flags.c_contiguous
    assert value.flags.owndata
    assert value.flags.writeable


def test_checkpoint_round_trip_restores_exact_training_state(tmp_path: Path) -> None:
    inputs, labels = _inputs()
    model, proxies, optimizer = _fixture()
    _MODULE.replayed_proxy_anchor_step(
        model=model,
        proxies=proxies,
        inputs=inputs,
        labels=labels,
        optimizer=optimizer,
        microbatch_size=2,
        update_index=0,
    )
    checkpoint = tmp_path / "state.pt"

    authority = _MODULE.write_geometry_checkpoint(
        path=checkpoint,
        model=model,
        proxies=proxies,
        optimizer=optimizer,
        source_commit="1" * 40,
        arm="mean",
        seed=17,
        completed_updates=1,
        epoch_plan_digests=("2" * 64, "3" * 64, "4" * 64),
    )

    restored_model, restored_proxies, restored_optimizer = _fixture()
    next_update = _MODULE.restore_geometry_checkpoint(
        path=checkpoint,
        authority=authority,
        model=restored_model,
        proxies=restored_proxies,
        optimizer=restored_optimizer,
        source_commit="1" * 40,
        arm="mean",
        seed=17,
        epoch_plan_digests=("2" * 64, "3" * 64, "4" * 64),
    )

    assert next_update == 1
    assert authority.basename == checkpoint.name
    assert authority.byte_length == checkpoint.stat().st_size
    assert authority.sha256 == _MODULE._sha256_path(checkpoint)
    assert _MODULE.state_sha256((*restored_model.parameters(), restored_proxies)) == (
        _MODULE.state_sha256((*model.parameters(), proxies))
    )
    assert _MODULE._optimizer_state_sha256(restored_optimizer) == (
        _MODULE._optimizer_state_sha256(optimizer)
    )


def test_checkpoint_rejects_byte_and_identity_drift(tmp_path: Path) -> None:
    model, proxies, optimizer = _fixture()
    checkpoint = tmp_path / "state.pt"
    kwargs = {
        "source_commit": "1" * 40,
        "arm": "attention",
        "seed": 29,
        "completed_updates": 0,
        "epoch_plan_digests": ("2" * 64, "3" * 64, "4" * 64),
    }
    authority = _MODULE.write_geometry_checkpoint(
        path=checkpoint,
        model=model,
        proxies=proxies,
        optimizer=optimizer,
        **kwargs,
    )
    restore = {
        "path": checkpoint,
        "authority": authority,
        "model": model,
        "proxies": proxies,
        "optimizer": optimizer,
        "source_commit": kwargs["source_commit"],
        "arm": kwargs["arm"],
        "seed": kwargs["seed"],
        "epoch_plan_digests": kwargs["epoch_plan_digests"],
    }

    for mutation in (
        {"authority": dataclasses.replace(authority, sha256="0" * 64)},
        {"source_commit": "5" * 40},
        {"arm": "mean"},
        {"seed": 43},
        {"epoch_plan_digests": ("6" * 64, "3" * 64, "4" * 64)},
    ):
        with pytest.raises(ValueError):
            _MODULE.restore_geometry_checkpoint(**(restore | mutation))

    with checkpoint.open("ab") as stream:
        stream.write(b"drift")
    with pytest.raises(ValueError, match="authority"):
        _MODULE.restore_geometry_checkpoint(**restore)


def test_run_train_executes_all_registered_updates_before_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _TrainModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(2, 2)
            self.pooler = nn.Linear(2, 2)

    examples = tuple(
        SimpleNamespace(label=label, image=index)
        for label in range(49)
        for index in range(4)
    )
    model = _TrainModel()
    calls: list[int] = []

    monkeypatch.setattr(_MODULE, "load_image_retrieval_examples", lambda **_kwargs: examples)
    monkeypatch.setattr(_MODULE, "_rgb_224", lambda image: image)
    monkeypatch.setattr(_MODULE, "_load_real_geometry_model", lambda _args: model)
    monkeypatch.setattr(
        _MODULE,
        "optimizer_groups",
        lambda **_kwargs: [
            {"params": list(model.parameters()), "lr": 1.0e-3, "role": "model"},
        ],
    )

    def fake_step(**kwargs: object) -> SimpleNamespace:
        calls.append(int(kwargs["update_index"]))
        return SimpleNamespace(
            loss=12.0 - len(calls) / 1000.0,
            gradient_norm=1.0,
            maximum_score_disagreement=0.0,
            sampled_parameter_values=9,
            changed_sampled_parameter_values=8,
            parameter_displacement=(("tower", 3, 2), ("pooler", 3, 3), ("proxies", 3, 3)),
            updated_state_sha256="a" * 64,
            optimizer_state_sha256="b" * 64,
        )

    monkeypatch.setattr(_MODULE, "replayed_proxy_anchor_step", fake_step)
    checkpoint = tmp_path / "checkpoint.pt"
    receipt = tmp_path / "receipt.json"
    args = SimpleNamespace(
        arm="mean",
        checkpoint_output=checkpoint,
        microbatch_size=1,
        output=receipt,
        seed=17,
        source_commit="1" * 40,
    )

    raw = _MODULE.run_train(args)
    value = __import__("json").loads(raw)

    assert calls == list(range(183))
    assert value["optimizer_updates"] == 183
    assert value["checkpoint"]["completed_updates"] == 183
    assert checkpoint.is_file()
    assert len(value["epochs"]) == 3


def test_smoke_reloads_initial_state_and_repeats_exact_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _SmokeModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = nn.Linear(2, 2)
            self.pooler = nn.Linear(2, 2)

        def visual_tokens(self, _images: object) -> torch.Tensor:
            return torch.ones(1, 1, 2)

    examples = tuple(
        SimpleNamespace(label=label, image=index)
        for label in range(49)
        for index in range(4)
    )
    loads: list[_SmokeModel] = []
    calls: list[int] = []
    monkeypatch.setattr(_MODULE, "load_image_retrieval_examples", lambda **_kwargs: examples)
    monkeypatch.setattr(_MODULE, "_rgb_224", lambda image: image)

    def load(_args: object) -> _SmokeModel:
        model = _SmokeModel()
        loads.append(model)
        return model

    monkeypatch.setattr(_MODULE, "_load_real_geometry_model", load)
    monkeypatch.setattr(
        _MODULE,
        "optimizer_groups",
        lambda **kwargs: [
            {"params": list(kwargs["tower"].parameters()), "lr": 1.0e-3, "role": "tower"}
        ],
    )

    def fake_step(**kwargs: object) -> SimpleNamespace:
        update = int(kwargs["update_index"])
        calls.append(update)
        return SimpleNamespace(
            loss=12.0 - update,
            gradient_norm=1.0 + update,
            maximum_score_disagreement=0.0,
            sampled_parameter_values=9,
            changed_sampled_parameter_values=8,
            parameter_displacement=(("tower", 3, 2), ("pooler", 3, 3), ("proxies", 3, 3)),
            updated_state_sha256=f"{update + 1:064x}",
            optimizer_state_sha256=f"{update + 4:064x}",
        )

    monkeypatch.setattr(_MODULE, "replayed_proxy_anchor_step", fake_step)
    monkeypatch.setattr(
        _MODULE,
        "_summarize_tower_displacement",
        lambda *_args: {
            "relative_l2": 2.0e-6,
            "moving_block_fraction": 1.0,
            "blocks": [],
        },
    )
    monkeypatch.setattr(
        _MODULE,
        "_summarize_token_displacement",
        lambda *_args: {"passed": True, "relative_l2": 2.0e-6},
    )
    args = SimpleNamespace(arm="mean", microbatch_size=1, seed=17, source_commit="1" * 40)

    value = __import__("json").loads(_MODULE.run_smoke(args))

    assert len(loads) == 3
    assert calls == [0, 1, 2, 0, 1, 2]
    assert value["checkpoint_resume_equal"] is True
    assert value["restored_repetition_equal"] is True
    assert value["trials"][0]["updates"] == value["trials"][1]["updates"]
    assert value["trials"][0]["resume_after_two"] is False
    assert value["trials"][1]["resume_after_two"] is True
