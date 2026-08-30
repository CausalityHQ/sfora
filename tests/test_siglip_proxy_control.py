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
    SeedControlEvidence,
    SiglipProxyControlConfig,
    evaluate_control_band,
    nearest_class_margins,
    recomputed_proxy_anchor_backward,
    summarize_control_seeds,
    validate_control_partition,
)
from sfora.substrate_screen import SUBSTRATE_F0_CLASSES, SubstrateScreenMetrics
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


def _margin_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [[1.0, 0.0], [0.8, 0.6], [-1.0, 0.0], [-0.8, 0.6]],
            dtype=torch.float32,
        ),
        torch.tensor([0, 0, 1, 1], dtype=torch.int64),
    )


def test_nearest_class_margins_are_blockwise_and_exact() -> None:
    embeddings, labels = _margin_fixture()

    evidence = nearest_class_margins(embeddings, labels, query_block=1)

    torch.testing.assert_close(
        evidence.nearest_positive_cosine,
        torch.tensor([0.8, 0.8, 0.8, 0.8]),
    )
    torch.testing.assert_close(
        evidence.nearest_negative_cosine,
        torch.tensor([-0.8, -0.28, -0.8, -0.28]),
    )
    torch.testing.assert_close(evidence.margin, torch.tensor([1.6, 1.08, 1.6, 1.08]))
    assert evidence.mean_nearest_positive_cosine == pytest.approx(0.8)
    assert evidence.mean_nearest_negative_cosine == pytest.approx(-0.54)
    assert evidence.mean_margin == pytest.approx(1.34)


def test_control_band_calls_the_existing_recall_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings, labels = _margin_fixture()
    expected = SubstrateScreenMetrics(correct=3, queries=4, recall_at_1=0.75)
    calls: list[tuple[torch.Tensor, torch.Tensor, int]] = []

    def fake_score(
        incoming_embeddings: torch.Tensor,
        incoming_labels: torch.Tensor,
        *,
        query_block: int,
    ) -> SubstrateScreenMetrics:
        calls.append((incoming_embeddings, incoming_labels, query_block))
        return expected

    monkeypatch.setattr("sfora.siglip_proxy_control.score_frozen_substrate", fake_score)

    evidence = evaluate_control_band(embeddings, labels, query_block=2)

    assert evidence.retrieval is expected
    assert calls == [(embeddings, labels, 2)]
    assert evidence.margins.mean_margin == pytest.approx(1.34)


def _seed_evidence(seed: int, *, train_change: float) -> SeedControlEvidence:
    return SeedControlEvidence(
        seed=seed,
        train_initial_margin=0.1,
        train_final_margin=0.1 + train_change,
        clean_initial_recall_at_1=0.4,
        clean_final_recall_at_1=0.55,
        clean_initial_margin=0.2,
        clean_final_margin=0.27,
        burned_initial_margin=0.3,
        burned_final_margin=0.34,
    )


def test_control_seed_summary_computes_registered_changes_and_ratios() -> None:
    summaries = summarize_control_seeds(
        (
            _seed_evidence(17, train_change=0.2),
            _seed_evidence(29, train_change=0.1),
            _seed_evidence(43, train_change=0.0),
        )
    )

    assert tuple(summary.seed for summary in summaries) == (17, 29, 43)
    assert summaries[0].clean_recall_change == pytest.approx(0.15)
    assert summaries[0].clean_margin_change == pytest.approx(0.07)
    assert summaries[0].burned_margin_change == pytest.approx(0.04)
    assert summaries[0].memorization_to_transfer_ratio == pytest.approx(0.2)
    assert summaries[1].memorization_to_transfer_ratio == pytest.approx(0.4)
    assert summaries[2].memorization_to_transfer_ratio is None


def test_control_seed_summary_rejects_cardinality_order_and_nonfinite_drift() -> None:
    with pytest.raises(ValueError, match="exact seeds"):
        summarize_control_seeds((_seed_evidence(17, train_change=0.2),))
    with pytest.raises(ValueError, match="exact seeds"):
        summarize_control_seeds(
            (
                _seed_evidence(29, train_change=0.2),
                _seed_evidence(17, train_change=0.2),
                _seed_evidence(43, train_change=0.2),
            )
        )
    with pytest.raises(ValueError, match="finite"):
        summarize_control_seeds(
            (
                _seed_evidence(17, train_change=0.2),
                replace(_seed_evidence(29, train_change=0.2), clean_final_margin=float("nan")),
                _seed_evidence(43, train_change=0.2),
            )
        )
