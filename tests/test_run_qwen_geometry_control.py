"""Behavioral tests for exact logical-batch Qwen geometry replay."""

from __future__ import annotations

import copy
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


class _FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.visual = nn.Linear(4, 4, bias=False)
        self.model.visual.deepstack_merger_list = nn.ModuleList([nn.Linear(4, 4)])
        self.model.language_model = _ForbiddenLanguage()
        self.lm_head = _ForbiddenLanguage()

    def get_image_features(
        self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor
    ) -> SimpleNamespace:
        assert image_grid_thw.shape == (pixel_values.shape[0], 3)
        rows = self.model.visual(pixel_values)
        return SimpleNamespace(pooler_output=tuple(rows.unbind()))


@pytest.mark.parametrize("arm", ["mean", "attention"])
def test_qwen_geometry_model_executes_only_vision_and_registered_pooler(arm: str) -> None:
    qwen = _FakeQwen()
    wrapper = _MODULE.QwenVisionGeometryModel(
        model=qwen,
        processor=_FakeImageProcessor(),
        token_dimensions=4,
        arm=arm,
    )
    images = tuple(torch.arange(16, dtype=torch.float32).reshape(4, 4) + row for row in range(2))

    descriptors = wrapper(images)

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
    wrapper_ids = {id(parameter) for parameter in wrapper.parameters() if parameter.requires_grad}
    assert wrapper_ids == {
        id(parameter)
        for parameter in (*qwen.model.visual.parameters(), *wrapper.pooler.parameters())
        if parameter.requires_grad
    }


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


def test_rgb_preprocessing_returns_an_owned_writable_224_array() -> None:
    image = Image.new("RGB", (8, 6), color=(1, 2, 3))
    value = _MODULE._rgb_224(image)
    assert value.shape == (224, 224, 3)
    assert value.flags.c_contiguous
    assert value.flags.owndata
    assert value.flags.writeable
