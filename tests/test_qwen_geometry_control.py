from __future__ import annotations

import hashlib
import struct
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from sfora.qwen_geometry_control import (
    GeometryBatchPlan,
    MeanProjectionPooler,
    QwenGeometryProtocol,
    SingleQueryAttentionPooler,
    build_geometry_pooler,
    derive_epoch_batches,
    initialize_geometry_pooler,
    initialize_geometry_proxies,
    learning_rate_multiplier,
    optimizer_groups,
    parameter_role_manifest,
    pool_patch_tokens,
)


def test_protocol_freezes_the_paired_qwen_geometry_experiment() -> None:
    protocol = QwenGeometryProtocol()

    assert protocol.model_name == "Qwen/Qwen3-VL-8B-Instruct"
    assert protocol.model_revision == "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"
    assert protocol.arms == ("mean", "attention")
    assert protocol.seeds == (17, 29, 43)
    assert protocol.optimization_classes == tuple(range(49))
    assert protocol.clean_development_classes == tuple(range(49, 82))
    assert protocol.burned_diagnostic_classes == tuple(range(82, 98))
    assert protocol.image_size == 224
    assert protocol.embedding_dimensions == 4096
    assert protocol.logical_batch_size == 64
    assert protocol.classes_per_batch == 16
    assert protocol.images_per_class == 4
    assert protocol.epochs == 3
    assert protocol.steps_per_epoch == 61
    assert protocol.optimizer_updates == 183
    assert protocol.proxy_anchor_alpha == 32.0
    assert protocol.proxy_anchor_delta == 0.1
    assert protocol.tower_learning_rate == 2.0e-5
    assert protocol.pooler_learning_rate == 1.0e-4
    assert protocol.proxy_learning_rate == 1.0e-2
    assert protocol.adamw_betas == (0.9, 0.999)
    assert protocol.adamw_epsilon == 1.0e-8
    assert protocol.weight_decay == 1.0e-4
    assert protocol.warmup_updates == 10
    assert protocol.gradient_clip_norm == 1.0
    assert protocol.trainable_parameter_precision == "float32"
    assert protocol.visual_compute_precision == "bfloat16-autocast"
    assert protocol.optimizer_foreach is False
    assert protocol.tower_displacement_floor == 1.0e-6
    assert protocol.minimum_moving_block_fraction == 0.9


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_revision", "f" * 40),
        ("arms", ("attention", "mean")),
        ("seeds", (17, 29)),
        ("optimization_classes", tuple(range(48))),
        ("clean_development_classes", tuple(range(50, 82))),
        ("burned_diagnostic_classes", tuple(range(83, 98))),
        ("image_size", 384),
        ("embedding_dimensions", 512),
        ("logical_batch_size", 63),
        ("classes_per_batch", 8),
        ("images_per_class", 8),
        ("epochs", 4),
        ("steps_per_epoch", 60),
        ("optimizer_updates", 182),
        ("proxy_anchor_alpha", 31.0),
        ("tower_learning_rate", 1.0e-3),
        ("adamw_betas", (0.8, 0.999)),
        ("warmup_updates", 9),
        ("gradient_clip_norm", 10.0),
        ("trainable_parameter_precision", "bfloat16"),
        ("visual_compute_precision", "float32"),
        ("optimizer_foreach", True),
        ("tower_displacement_floor", 0.0),
        ("minimum_moving_block_fraction", 0.8),
    ],
)
def test_protocol_rejects_scientific_constant_drift(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field.replace("_", " ")):
        replace(QwenGeometryProtocol(), **{field: value})


def test_mean_pooler_matches_hand_derived_mean_then_projection() -> None:
    pooler = MeanProjectionPooler(token_dimensions=4)
    with torch.no_grad():
        pooler.output.weight.copy_(
            torch.arange(4096 * 4, dtype=torch.float32).reshape(4096, 4).remainder(17) / 17
        )
    tokens = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [2.0, 4.0, 1.0, 3.0]]]
    )

    actual, weights = pool_patch_tokens(pooler, tokens)
    expected = F.normalize(F.linear(tokens.mean(dim=1), pooler.output.weight), dim=-1)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert weights is None
    assert actual.dtype == torch.float32
    assert actual.shape == (1, 4096)


def test_attention_pooler_matches_hand_derived_query_key_softmax() -> None:
    pooler = SingleQueryAttentionPooler(token_dimensions=4)
    with torch.no_grad():
        pooler.query.copy_(torch.tensor([1.0, -2.0, 0.5, 3.0]))
        pooler.key.weight.copy_(torch.eye(4))
        pooler.output.weight.copy_(
            torch.arange(4096 * 4, dtype=torch.float32).reshape(4096, 4).remainder(19) / 19
        )
    tokens = torch.tensor(
        [[[1.0, 2.0, 0.0, 1.0], [2.0, 0.0, 1.0, 3.0], [0.0, 1.0, 4.0, 2.0]]]
    )

    actual, weights = pool_patch_tokens(pooler, tokens)
    logits = torch.einsum("d,bpd->bp", pooler.query, F.linear(tokens, pooler.key.weight)) / 2
    expected_weights = logits.softmax(dim=-1)
    expected_pooled = torch.einsum("bp,bpd->bd", expected_weights, tokens)
    expected = F.normalize(F.linear(expected_pooled, pooler.output.weight), dim=-1)

    assert weights is not None
    torch.testing.assert_close(weights, expected_weights, rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    torch.testing.assert_close(weights.sum(dim=-1), torch.ones(1), rtol=0.0, atol=0.0)


def test_pooler_factory_exposes_only_the_two_registered_arms() -> None:
    assert type(build_geometry_pooler("mean", token_dimensions=4)) is MeanProjectionPooler
    assert (
        type(build_geometry_pooler("attention", token_dimensions=4))
        is SingleQueryAttentionPooler
    )
    with pytest.raises(ValueError, match="registered geometry arm"):
        build_geometry_pooler("max", token_dimensions=4)


def test_role_separated_initialization_keeps_shared_projection_byte_identical() -> None:
    mean = build_geometry_pooler("mean", token_dimensions=4)
    attention = build_geometry_pooler("attention", token_dimensions=4)

    initialize_geometry_pooler(mean, seed=17)
    initialize_geometry_pooler(attention, seed=17)

    assert torch.equal(mean.output.weight, attention.output.weight)
    assert torch.count_nonzero(attention.query) > 0
    assert torch.count_nonzero(attention.key.weight) > 0
    repeated = build_geometry_pooler("attention", token_dimensions=4)
    initialize_geometry_pooler(repeated, seed=17)
    for left, right in zip(attention.parameters(), repeated.parameters(), strict=True):
        assert torch.equal(left, right)

    with pytest.raises(ValueError, match="seed"):
        initialize_geometry_pooler(mean, seed=18)

    proxies = torch.nn.Parameter(torch.empty(49, 4096))
    initialize_geometry_proxies(proxies, seed=17)
    repeated_proxies = torch.nn.Parameter(torch.empty(49, 4096))
    initialize_geometry_proxies(repeated_proxies, seed=17)
    assert torch.equal(proxies, repeated_proxies)


@pytest.mark.parametrize(
    "tokens",
    [
        torch.ones(2, 4),
        torch.ones(2, 3, 5),
        torch.tensor([[[1.0, 2.0, float("nan"), 4.0]]]),
        torch.empty(1, 0, 4),
    ],
)
def test_poolers_reject_malformed_or_nonfinite_patch_tokens(tokens: torch.Tensor) -> None:
    for arm in QwenGeometryProtocol().arms:
        pooler = build_geometry_pooler(arm, token_dimensions=4)
        with pytest.raises(ValueError, match="patch tokens"):
            pool_patch_tokens(pooler, tokens)


def test_poolers_reject_a_zero_descriptor_before_normalization() -> None:
    tokens = torch.ones(1, 2, 4)
    for arm in QwenGeometryProtocol().arms:
        pooler = build_geometry_pooler(arm, token_dimensions=4)
        with torch.no_grad():
            pooler.output.weight.zero_()
        with pytest.raises(ValueError, match="nonzero"):
            pool_patch_tokens(pooler, tokens)


def _batch_digest(indices: tuple[int, ...]) -> str:
    encoded = b"".join(struct.pack("<Q", index) for index in indices)
    return hashlib.sha256(encoded).hexdigest()


def test_epoch_batches_are_balanced_deterministic_and_arm_invariant() -> None:
    members = {
        label: tuple(range(label * 5, label * 5 + 5))
        for label in QwenGeometryProtocol().optimization_classes
    }

    plan = derive_epoch_batches(members, seed=17, epoch=0)
    repeated = derive_epoch_batches(members, seed=17, epoch=0)

    assert type(plan) is GeometryBatchPlan
    assert plan == repeated
    assert len(plan.batches) == QwenGeometryProtocol().steps_per_epoch
    assert plan.digest == hashlib.sha256("".join(plan.batch_digests).encode()).hexdigest()
    for batch, digest in zip(plan.batches, plan.batch_digests, strict=True):
        assert len(batch) == 64
        assert digest == _batch_digest(batch)
        labels = [index // 5 for index in batch]
        assert len(set(labels)) == 16
        assert all(labels.count(label) == 4 for label in set(labels))
        assert len(set(batch)) == 64

    assert derive_epoch_batches(members, seed=29, epoch=0) != plan
    assert derive_epoch_batches(members, seed=17, epoch=1) != plan


@pytest.mark.parametrize(
    "members",
    [
        {label: tuple(range(4)) for label in range(48)},
        {label: tuple(range(4)) for label in range(50)},
        {label: tuple(range(3)) for label in range(49)},
        {label: (0, 1, 2, 2) for label in range(49)},
    ],
)
def test_epoch_batches_reject_noncanonical_class_members(members: object) -> None:
    with pytest.raises(ValueError, match="optimization class members"):
        derive_epoch_batches(members, seed=17, epoch=0)  # type: ignore[arg-type]


def test_learning_rate_schedule_has_registered_warmup_and_cosine_endpoints() -> None:
    assert learning_rate_multiplier(0) == pytest.approx(0.1)
    assert learning_rate_multiplier(9) == pytest.approx(1.0)
    assert learning_rate_multiplier(10) == pytest.approx(1.0)
    assert learning_rate_multiplier(182) == pytest.approx(0.0, abs=1.0e-15)
    multipliers = [learning_rate_multiplier(update) for update in range(183)]
    assert all(0.0 <= value <= 1.0 for value in multipliers)
    assert multipliers[10:] == sorted(multipliers[10:], reverse=True)
    for invalid in (-1, 183, True, 1.0):
        with pytest.raises(ValueError, match="update index"):
            learning_rate_multiplier(invalid)  # type: ignore[arg-type]


class _RoleFixture(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tower = torch.nn.Sequential(
            torch.nn.Linear(4, 4), torch.nn.LayerNorm(4), torch.nn.Linear(4, 4, bias=False)
        )
        self.pooler = MeanProjectionPooler(4)
        self.proxies = torch.nn.Parameter(torch.ones(49, 4096))


def test_parameter_roles_and_optimizer_groups_are_complete_disjoint_and_exact() -> None:
    fixture = _RoleFixture()
    manifest = parameter_role_manifest(
        tower=fixture.tower, pooler=fixture.pooler, proxies=fixture.proxies
    )
    groups = optimizer_groups(
        tower=fixture.tower, pooler=fixture.pooler, proxies=fixture.proxies
    )

    named_ids = {
        id(parameter)
        for parameter in (
            *fixture.tower.parameters(),
            *fixture.pooler.parameters(),
            fixture.proxies,
        )
    }
    grouped = [parameter for group in groups for parameter in group["params"]]
    assert {id(parameter) for parameter in grouped} == named_ids
    assert len({id(parameter) for parameter in grouped}) == len(grouped)
    assert {role for _, role in manifest.roles} == {"tower", "pooler", "proxies"}
    assert {group["role"] for group in groups} == {"tower", "pooler", "proxies"}
    for group in groups:
        expected_lr = {
            "tower": 2.0e-5,
            "pooler": 1.0e-4,
            "proxies": 1.0e-2,
        }[group["role"]]
        assert group["lr"] == expected_lr
        if group["role"] == "proxies" or group["decay"] is False:
            assert group["weight_decay"] == 0.0
        else:
            assert group["weight_decay"] == 1.0e-4


def test_parameter_roles_reject_frozen_or_aliased_parameters() -> None:
    fixture = _RoleFixture()
    fixture.tower[0].weight.requires_grad_(False)
    with pytest.raises(ValueError, match="trainable"):
        parameter_role_manifest(
            tower=fixture.tower, pooler=fixture.pooler, proxies=fixture.proxies
        )

    fixture = _RoleFixture()
    fixture.pooler.output.weight = fixture.tower[0].weight
    with pytest.raises(ValueError, match="duplicated"):
        parameter_role_manifest(
            tower=fixture.tower, pooler=fixture.pooler, proxies=fixture.proxies
        )
