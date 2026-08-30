from __future__ import annotations

import pytest
import torch

from sfora.token_set_training import (
    F1TrainingConfig,
    FrozenTokenSetSplit,
    PooledProxyAnchorHead,
    initialize_paired_f1_heads,
    train_f1_arm,
)


def _split(labels: torch.Tensor) -> FrozenTokenSetSplit:
    generator = torch.Generator().manual_seed(101 + int(labels[0]))
    count = labels.numel()
    attention = torch.rand((count, 2), generator=generator)
    return FrozenTokenSetSplit(
        global_features=torch.randn((count, 4), generator=generator),
        token_features=torch.randn((count, 2, 4), generator=generator).to(torch.float16),
        pretrained_attention=attention / attention.sum(dim=1, keepdim=True),
        labels=labels,
    )


def test_paired_heads_share_global_initialization_and_tspa_state() -> None:
    pooled, tspa, shuffled = initialize_paired_f1_heads(
        input_dimensions=4,
        classes=49,
        global_dimensions=3,
        token_dimensions=2,
        token_proxies_per_class=2,
        set_weight=0.25,
        seed=17,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(pooled.projection.weight, tspa.global_projection.weight)
    torch.testing.assert_close(pooled.proxies, tspa.global_proxies)
    for name, value in tspa.state_dict().items():
        torch.testing.assert_close(value, shuffled.state_dict()[name])


def test_pooled_proxy_head_returns_normalized_embeddings_and_scores() -> None:
    head = PooledProxyAnchorHead(input_dimensions=4, dimensions=3, classes=2)
    scores, embeddings = head(torch.randn(5, 4))

    assert scores.shape == (5, 2)
    torch.testing.assert_close(embeddings.norm(dim=1), torch.ones(5))


def test_one_epoch_f1_arm_returns_final_only_validation_evidence() -> None:
    train = _split(torch.arange(49).repeat_interleave(2))
    validation = _split(torch.arange(49, 82).repeat_interleave(2))
    config = F1TrainingConfig(
        epochs=1,
        batch_size=32,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        global_dimensions=3,
        token_dimensions=2,
        token_proxies_per_class=2,
        set_weight=0.25,
        query_block=16,
    )
    pooled, _, _ = initialize_paired_f1_heads(
        input_dimensions=4,
        classes=49,
        global_dimensions=config.global_dimensions,
        token_dimensions=config.token_dimensions,
        token_proxies_per_class=config.token_proxies_per_class,
        set_weight=config.set_weight,
        seed=17,
        device=torch.device("cpu"),
    )

    result = train_f1_arm(
        arm="pooled",
        head=pooled,
        train=train,
        validation=validation,
        config=config,
        seed=17,
        device=torch.device("cpu"),
    )

    assert result.arm == "pooled"
    assert 0.0 <= result.validation_recall_at_1 <= 1.0
    assert result.final_training_objective > 0.0
    assert result.objective_kind == "proxy-anchor"
    assert result.mean_token_proxy_cosine is None
    assert result.collapse_exceeded is None


def test_one_epoch_tspa_accepts_frozen_half_precision_tokens() -> None:
    train = _split(torch.arange(49).repeat_interleave(2))
    validation = _split(torch.arange(49, 82).repeat_interleave(2))
    config = F1TrainingConfig(
        epochs=1,
        batch_size=64,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        global_dimensions=3,
        token_dimensions=2,
        token_proxies_per_class=2,
        set_weight=0.25,
        query_block=32,
    )
    _, tspa, _ = initialize_paired_f1_heads(
        input_dimensions=4,
        classes=49,
        global_dimensions=3,
        token_dimensions=2,
        token_proxies_per_class=2,
        set_weight=0.25,
        seed=17,
        device=torch.device("cpu"),
    )

    result = train_f1_arm(
        arm="tspa",
        head=tspa,
        train=train,
        validation=validation,
        config=config,
        seed=17,
        device=torch.device("cpu"),
    )

    assert 0.0 <= result.validation_recall_at_1 <= 1.0
    assert result.mean_token_proxy_cosine is not None
    assert result.collapse_exceeded is False


def test_one_epoch_token_shuffled_arm_executes_bound_token_attention_permutation() -> None:
    train = _split(torch.arange(49).repeat_interleave(2))
    validation = _split(torch.arange(49, 82).repeat_interleave(2))
    config = F1TrainingConfig(
        epochs=1,
        batch_size=98,
        learning_rate=1.0e-3,
        weight_decay=0.0,
        global_dimensions=3,
        token_dimensions=2,
        token_proxies_per_class=2,
        set_weight=0.25,
        query_block=32,
    )
    _, _, shuffled = initialize_paired_f1_heads(
        input_dimensions=4,
        classes=49,
        global_dimensions=3,
        token_dimensions=2,
        token_proxies_per_class=2,
        set_weight=0.25,
        seed=17,
        device=torch.device("cpu"),
    )

    result = train_f1_arm(
        arm="token-shuffled-tspa",
        head=shuffled,
        train=train,
        validation=validation,
        config=config,
        seed=17,
        device=torch.device("cpu"),
    )

    assert result.arm == "token-shuffled-tspa"
    assert result.objective_kind == "proxy-anchor-plus-proxy-diversity"


def test_training_rejects_head_and_evaluation_weight_drift() -> None:
    train = _split(torch.arange(49).repeat_interleave(2))
    validation = _split(torch.arange(49, 82).repeat_interleave(2))
    config = F1TrainingConfig(
        epochs=1,
        batch_size=98,
        global_dimensions=3,
        token_dimensions=2,
        token_proxies_per_class=2,
        set_weight=0.25,
    )
    _, tspa, _ = initialize_paired_f1_heads(
        input_dimensions=4,
        classes=49,
        global_dimensions=3,
        token_dimensions=2,
        token_proxies_per_class=2,
        set_weight=0.5,
        seed=17,
        device=torch.device("cpu"),
    )

    with pytest.raises(ValueError, match="differs from the F1 configuration"):
        train_f1_arm(
            arm="tspa",
            head=tspa,
            train=train,
            validation=validation,
            config=config,
            seed=17,
            device=torch.device("cpu"),
        )
