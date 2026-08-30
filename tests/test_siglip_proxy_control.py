"""Frozen authority tests for the SigLIP pooled Proxy Anchor control."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, cast

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from sfora.siglip_proxy_control import (
    PooledProxyAnchorModel,
    SiglipProxyControlConfig,
    recomputed_proxy_anchor_backward,
    validate_control_partition,
)
from sfora.substrate_screen import SUBSTRATE_F0_CLASSES
from sfora.token_set_proxy_anchor import proxy_anchor_loss
from sfora.token_set_screen import F1_TRAIN_CLASSES, F1_VALIDATION_CLASSES


def _labels(classes: frozenset[int], count: int) -> torch.Tensor:
    return torch.tensor(
        [label for label in sorted(classes) for _ in range(count)],
        dtype=torch.int64,
    )


def test_control_partition_reuses_exact_existing_band_authority() -> None:
    split = validate_control_partition(
        optimization_labels=_labels(F1_TRAIN_CLASSES, 4),
        clean_validation_labels=_labels(F1_VALIDATION_CLASSES, 2),
        burned_diagnostic_labels=_labels(SUBSTRATE_F0_CLASSES, 2),
    )

    assert split.optimization_classes is F1_TRAIN_CLASSES
    assert split.clean_validation_classes is F1_VALIDATION_CLASSES
    assert split.burned_diagnostic_classes is SUBSTRATE_F0_CLASSES
    assert split.optimization_examples == 49 * 4
    assert split.clean_validation_examples == 33 * 2
    assert split.burned_diagnostic_examples == 16 * 2


@pytest.mark.parametrize(
    ("role", "labels", "message"),
    [
        ("optimization_labels", _labels(F1_TRAIN_CLASSES, 4)[:-1], "at least four"),
        (
            "clean_validation_labels",
            _labels(F1_VALIDATION_CLASSES, 2)[:-1],
            "at least two",
        ),
        (
            "burned_diagnostic_labels",
            _labels(SUBSTRATE_F0_CLASSES, 2)[:-1],
            "at least two",
        ),
        (
            "optimization_labels",
            _labels(frozenset(range(48)), 4),
            "exact optimization classes",
        ),
        (
            "clean_validation_labels",
            torch.cat((_labels(F1_VALIDATION_CLASSES, 2), torch.tensor([98]))),
            "exact clean-validation classes",
        ),
        (
            "burned_diagnostic_labels",
            torch.cat((_labels(SUBSTRATE_F0_CLASSES, 2), torch.tensor([81]))),
            "exact burned-diagnostic classes",
        ),
    ],
)
def test_control_partition_rejects_count_and_band_drift(
    role: str,
    labels: torch.Tensor,
    message: str,
) -> None:
    arguments = {
        "optimization_labels": _labels(F1_TRAIN_CLASSES, 4),
        "clean_validation_labels": _labels(F1_VALIDATION_CLASSES, 2),
        "burned_diagnostic_labels": _labels(SUBSTRATE_F0_CLASSES, 2),
    }
    arguments[role] = labels

    with pytest.raises(ValueError, match=message):
        validate_control_partition(**arguments)


@pytest.mark.parametrize(
    "labels",
    [
        torch.zeros((2, 2), dtype=torch.int64),
        torch.zeros(196, dtype=torch.float32),
        torch.zeros(196, dtype=torch.bool),
        torch.empty(0, dtype=torch.int64),
    ],
)
def test_control_partition_rejects_noncanonical_label_tensors(labels: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="integer vector"):
        validate_control_partition(
            optimization_labels=labels,
            clean_validation_labels=_labels(F1_VALIDATION_CLASSES, 2),
            burned_diagnostic_labels=_labels(SUBSTRATE_F0_CLASSES, 2),
        )


def test_control_config_is_the_frozen_reviewed_contract() -> None:
    config = SiglipProxyControlConfig()

    assert config.model_name == "google/siglip-so400m-patch14-384"
    assert config.model_revision == "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
    assert config.dataset_name == "tanganke/stanford_cars"
    assert config.dataset_revision == "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
    assert config.seeds == (17, 29, 43)
    assert config.input_dimensions == 1152
    assert config.embedding_dimensions == 512
    assert config.logical_batch_size == 120
    assert config.images_per_class == 4
    assert config.classes_per_batch == 30
    assert config.proxy_anchor_alpha == 32.0
    assert config.proxy_anchor_delta == 0.1
    assert config.tower_learning_rate == 1.0e-5
    assert config.projection_learning_rate == 1.0e-4
    assert config.proxy_learning_rate == 1.0e-2
    assert config.weight_decay == 1.0e-4
    assert config.train_epochs == 60
    assert config.warmup_epochs == 5
    assert config.decay_epochs == (10, 20, 30, 40, 50)
    assert config.decay_gamma == 0.5
    assert config.gradient_clip_norm == 10.0
    assert config.replay_score_tolerance == 2.0e-5
    assert config.smoke_microbatch_ladder == (
        120,
        60,
        40,
        30,
        24,
        20,
        15,
        12,
        10,
        8,
        6,
        5,
        4,
        3,
        2,
        1,
    )
    assert config.combined_memory_limit_bytes == 96 * 1024**3
    assert config.maximum_projected_seed_hours == 24.0
    assert config.steps_per_epoch(119) == 0
    assert config.steps_per_epoch(120) == 1
    assert config.steps_per_epoch(239) == 1
    assert config.steps_per_epoch(240) == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_name", "google/siglip-base-patch16-224"),
        ("model_revision", "main"),
        ("dataset_name", "cars"),
        ("dataset_revision", "main"),
        ("seeds", (17, 29)),
        ("input_dimensions", 512),
        ("embedding_dimensions", 1152),
        ("logical_batch_size", 119),
        ("images_per_class", 3),
        ("classes_per_batch", 29),
        ("proxy_anchor_alpha", 31.0),
        ("proxy_anchor_delta", 0.2),
        ("tower_learning_rate", 2.0e-5),
        ("projection_learning_rate", 2.0e-4),
        ("proxy_learning_rate", 2.0e-2),
        ("weight_decay", 0.0),
        ("train_epochs", 61),
        ("warmup_epochs", 4),
        ("decay_epochs", (10, 20, 30, 40)),
        ("decay_gamma", 0.1),
        ("gradient_clip_norm", 1.0),
        ("replay_score_tolerance", 1.0e-3),
        ("smoke_microbatch_ladder", (8, 4, 2, 1)),
        ("combined_memory_limit_bytes", 96_000_000_000),
        ("maximum_projected_seed_hours", 48.0),
    ],
)
def test_control_config_rejects_every_frozen_field_drift(field: str, value: object) -> None:
    with pytest.raises(ValueError, match="frozen SigLIP pooled-control contract"):
        replace(SiglipProxyControlConfig(), **cast(Any, {field: value}))


@pytest.mark.parametrize("examples", [-1, 0, True, 119.0])
def test_steps_per_epoch_requires_enough_concrete_examples(examples: object) -> None:
    with pytest.raises(ValueError, match="optimization example count"):
        SiglipProxyControlConfig().steps_per_epoch(examples)  # type: ignore[arg-type]


class _ToyTower(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(5, 7, bias=True),
            nn.GELU(),
            nn.Linear(7, 4, bias=False),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.layers(inputs))


def _toy_control() -> PooledProxyAnchorModel:
    torch.manual_seed(917)
    return PooledProxyAnchorModel(
        tower=_ToyTower(),
        input_dimensions=4,
        embedding_dimensions=3,
        class_count=3,
    ).double()


@pytest.mark.parametrize("microbatch_size", [1, 2, 4])
def test_recomputed_proxy_anchor_matches_one_pass_chain_rule(
    microbatch_size: int,
) -> None:
    inputs = torch.tensor(
        [
            [0.5, -0.2, 0.3, 0.8, -0.1],
            [-0.4, 0.7, 0.2, -0.3, 0.9],
            [0.6, 0.1, -0.8, 0.4, 0.2],
            [-0.2, -0.5, 0.9, 0.3, 0.7],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
    oracle = _toy_control()
    replay = copy.deepcopy(oracle)
    oracle_optimizer = torch.optim.SGD(oracle.parameters(), lr=0.031)
    replay_optimizer = torch.optim.SGD(replay.parameters(), lr=0.031)

    oracle_scores = oracle.class_scores(inputs)
    oracle_loss = proxy_anchor_loss(oracle_scores, labels, alpha=32.0, delta=0.1)
    (oracle_score_gradients,) = torch.autograd.grad(
        oracle_loss,
        oracle_scores,
        retain_graph=True,
    )
    torch.autograd.backward(oracle_loss)
    oracle_gradients: dict[str, torch.Tensor] = {}
    for name, parameter in oracle.named_parameters():
        assert parameter.grad is not None
        oracle_gradients[name] = parameter.grad.detach().clone()

    evidence = recomputed_proxy_anchor_backward(
        replay,
        inputs,
        labels,
        microbatch_size=microbatch_size,
        alpha=32.0,
        delta=0.1,
        score_tolerance=2.0e-5,
    )
    replay_gradients: dict[str, torch.Tensor] = {}
    for name, parameter in replay.named_parameters():
        assert parameter.grad is not None
        replay_gradients[name] = parameter.grad.detach().clone()

    torch.testing.assert_close(evidence.loss, oracle_loss.detach(), rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(evidence.scores, oracle_scores.detach(), rtol=1.0e-6, atol=1.0e-7)
    torch.testing.assert_close(
        evidence.score_gradients,
        oracle_score_gradients.detach(),
        rtol=1.0e-6,
        atol=1.0e-7,
    )
    assert evidence.maximum_score_disagreement <= 1.0e-12
    assert replay_gradients.keys() == oracle_gradients.keys()
    for name in oracle_gradients:
        torch.testing.assert_close(
            replay_gradients[name],
            oracle_gradients[name],
            rtol=1.0e-6,
            atol=1.0e-7,
        )

    oracle_optimizer.step()
    replay_optimizer.step()
    for (oracle_name, oracle_parameter), (replay_name, replay_parameter) in zip(
        oracle.named_parameters(),
        replay.named_parameters(),
        strict=True,
    ):
        assert replay_name == oracle_name
        torch.testing.assert_close(
            replay_parameter,
            oracle_parameter,
            rtol=1.0e-6,
            atol=1.0e-7,
        )


def _parameter_gradients(model: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def test_forbidden_microbatch_objectives_and_cotangent_rescaling_diverge() -> None:
    inputs = torch.tensor(
        [
            [0.5, -0.2, 0.3, 0.8, -0.1],
            [-0.4, 0.7, 0.2, -0.3, 0.9],
            [0.6, 0.1, -0.8, 0.4, 0.2],
            [-0.2, -0.5, 0.9, 0.3, 0.7],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
    correct = _toy_control()
    independent = copy.deepcopy(correct)

    recomputed_proxy_anchor_backward(
        correct,
        inputs,
        labels,
        microbatch_size=2,
        alpha=32.0,
        delta=0.1,
        score_tolerance=2.0e-5,
    )
    for start in (0, 2):
        local_scores = independent.class_scores(inputs[start : start + 2])
        local_loss = proxy_anchor_loss(
            local_scores,
            labels[start : start + 2],
            alpha=32.0,
            delta=0.1,
        )
        torch.autograd.backward(local_loss)

    correct_gradients = _parameter_gradients(correct)
    independent_gradients = _parameter_gradients(independent)
    diverged = False
    for name, correct_gradient in correct_gradients.items():
        independent_gradient = independent_gradients[name]
        if (
            correct_gradient is not None
            and independent_gradient is not None
            and not torch.allclose(correct_gradient, independent_gradient)
        ):
            diverged = True
    assert diverged
    assert any(
        gradient is not None and not torch.allclose(gradient, gradient * 0.5)
        for gradient in correct_gradients.values()
    )


def test_reordered_labels_and_detached_proxies_fail_the_full_batch_oracle() -> None:
    inputs = torch.arange(20, dtype=torch.float64).reshape(4, 5) / 17.0
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.int64)
    model = _toy_control()
    scores = model.class_scores(inputs)
    correct_loss = proxy_anchor_loss(scores, labels, alpha=32.0, delta=0.1)
    reordered_loss = proxy_anchor_loss(scores, labels.roll(1), alpha=32.0, delta=0.1)
    assert not torch.allclose(correct_loss, reordered_loss)

    detached_model = _toy_control()
    embeddings = detached_model.encode(inputs)
    detached_scores = embeddings @ F.normalize(detached_model.proxies.detach().float(), dim=1).T
    detached_loss = proxy_anchor_loss(detached_scores, labels, alpha=32.0, delta=0.1)
    torch.autograd.backward(detached_loss)
    assert detached_model.proxies.grad is None


@pytest.mark.parametrize("microbatch_size", [0, 3, 5, True])
def test_recomputed_backward_rejects_nondivisor_microbatches(microbatch_size: object) -> None:
    with pytest.raises(ValueError, match="positive divisor"):
        recomputed_proxy_anchor_backward(
            _toy_control(),
            torch.ones((4, 5), dtype=torch.float64),
            torch.tensor([0, 1, 2, 0]),
            microbatch_size=microbatch_size,  # type: ignore[arg-type]
            alpha=32.0,
            delta=0.1,
            score_tolerance=2.0e-5,
        )


class _StochasticTower(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.input = nn.Linear(5, 4)
        self.stochastic = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, self.stochastic(self.input(inputs)))


@pytest.mark.parametrize("module", [nn.Dropout(0.1), nn.BatchNorm1d(4)])
def test_recomputed_backward_refuses_stochastic_or_batch_dependent_towers(
    module: nn.Module,
) -> None:
    model = PooledProxyAnchorModel(
        tower=_StochasticTower(module),
        input_dimensions=4,
        embedding_dimensions=3,
        class_count=3,
    )
    model.train()
    with pytest.raises(ValueError, match="dropout|batch normalization"):
        recomputed_proxy_anchor_backward(
            model,
            torch.ones((4, 5)),
            torch.tensor([0, 1, 2, 0]),
            microbatch_size=2,
            alpha=32.0,
            delta=0.1,
            score_tolerance=2.0e-5,
        )
