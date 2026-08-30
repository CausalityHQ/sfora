"""Frozen authority tests for the SigLIP pooled Proxy Anchor control."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest
import torch

from sfora.siglip_proxy_control import (
    SiglipProxyControlConfig,
    validate_control_partition,
)
from sfora.substrate_screen import SUBSTRATE_F0_CLASSES
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
