from __future__ import annotations

import pytest
import torch

from sfora.token_set_proxy_anchor import (
    TokenSetProxyAnchorHead,
    proxy_anchor_loss,
    select_attention_tokens,
    token_proxy_diversity,
    token_set_class_scores,
    token_set_proxy_anchor_objective,
)


def test_token_set_class_scores_match_global_and_maxsim_equation() -> None:
    global_embeddings = torch.tensor([[1.0, 0.0]])
    token_embeddings = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    token_weights = torch.tensor([[0.75, 0.25]])
    global_proxies = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    token_proxies = torch.tensor(
        [
            [[1.0, 0.0], [-1.0, 0.0]],
            [[0.0, 1.0], [0.0, -1.0]],
        ]
    )

    scores = token_set_class_scores(
        global_embeddings,
        token_embeddings,
        token_weights,
        global_proxies,
        token_proxies,
        set_weight=0.25,
    )

    torch.testing.assert_close(scores, torch.tensor([[0.9375, 0.0625]]), rtol=0, atol=0)


def test_proxy_anchor_loss_rewards_the_correct_class() -> None:
    labels = torch.tensor([0, 1])
    good = torch.tensor([[0.9, -0.2], [-0.1, 0.8]])
    bad = torch.tensor([[-0.2, 0.9], [0.8, -0.1]])

    assert proxy_anchor_loss(good, labels, alpha=32.0, delta=0.1) < proxy_anchor_loss(
        bad,
        labels,
        alpha=32.0,
        delta=0.1,
    )


def test_proxy_anchor_loss_matches_closed_form() -> None:
    scores = torch.tensor([[0.2, -0.1], [0.4, 0.3], [-0.2, 0.1]])
    labels = torch.tensor([0, 0, 1])
    alpha = 3.0
    delta = 0.1
    one_hot = torch.nn.functional.one_hot(labels, num_classes=2).bool()
    positive = torch.exp(-alpha * (scores - delta)).masked_fill(~one_hot, 0.0)
    negative = torch.exp(alpha * (scores + delta)).masked_fill(one_hot, 0.0)
    expected = torch.log1p(positive.sum(dim=0)).mean() + torch.log1p(negative.sum(dim=0)).mean()

    actual = proxy_anchor_loss(scores, labels, alpha=alpha, delta=delta)

    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_token_set_proxy_anchor_head_produces_normalized_trainable_state() -> None:
    generator = torch.Generator().manual_seed(11)
    head = TokenSetProxyAnchorHead(
        input_dimensions=8,
        global_dimensions=6,
        token_dimensions=4,
        classes=3,
        token_proxies_per_class=2,
        set_weight=0.25,
    )
    global_features = torch.randn(5, 8, generator=generator)
    token_features = torch.randn(5, 3, 8, generator=generator)
    pretrained_attention = torch.softmax(torch.randn(5, 3, generator=generator), dim=1)

    output = head(global_features, token_features, pretrained_attention)

    assert output.class_scores.shape == (5, 3)
    assert output.global_embeddings.shape == (5, 6)
    assert output.token_embeddings.shape == (5, 3, 4)
    torch.testing.assert_close(output.token_weights.sum(dim=1), torch.ones(5))
    torch.testing.assert_close(output.global_embeddings.norm(dim=1), torch.ones(5))
    torch.testing.assert_close(output.token_embeddings.norm(dim=2), torch.ones(5, 3))
    torch.testing.assert_close(output.token_weights, pretrained_attention, rtol=1e-6, atol=1e-7)
    assert "set_weight_tensor" in head.state_dict()
    output.class_scores.sum().backward()
    assert head.saliency_residual.weight.grad is not None
    assert float(head.saliency_residual.weight.grad.abs().max()) > 0.0
    assert head.global_proxies.grad is not None
    assert head.token_proxies.grad is not None
    with torch.no_grad():
        head.set_weight_tensor.fill_(0.5)
    rescored = head(global_features, token_features, pretrained_attention)
    assert not torch.equal(output.class_scores, rescored.class_scores)


def test_token_proxy_diversity_detects_collapsed_proxy_sets() -> None:
    collapsed = torch.tensor([[[1.0, 0.0], [1.0, 0.0]]])
    diverse = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    collapsed_penalty, collapsed_cosine = token_proxy_diversity(collapsed, margin=0.5)
    diverse_penalty, diverse_cosine = token_proxy_diversity(diverse, margin=0.5)

    assert collapsed_penalty > diverse_penalty
    assert collapsed_cosine == 1.0
    assert diverse_cosine == 0.0


def test_select_attention_tokens_is_shared_and_stable_for_ties() -> None:
    tokens = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    selected, weights, indices = select_attention_tokens(
        tokens,
        torch.tensor([[0.4, 0.4, 0.1, 0.1]]),
        top_k=3,
    )

    assert indices.tolist() == [[0, 1, 2]]
    torch.testing.assert_close(selected, tokens[:, :3])
    torch.testing.assert_close(weights.sum(dim=1), torch.ones(1))


def test_objective_binds_diversity_and_collapse_gate() -> None:
    scores = torch.tensor([[0.8, -0.2], [-0.1, 0.7]])
    labels = torch.tensor([0, 1])
    collapsed = torch.tensor([[[1.0, 0.0], [1.0, 0.0]], [[0.0, 1.0], [0.0, 1.0]]])

    result = token_set_proxy_anchor_objective(scores, labels, collapsed)

    torch.testing.assert_close(result.total, result.proxy_anchor + 0.1 * result.diversity)
    assert result.mean_token_proxy_cosine == 1.0
    assert bool(result.collapse_exceeded) is True


def test_training_surfaces_reject_empty_or_invalid_authority() -> None:
    with pytest.raises(ValueError, match="empty batch"):
        proxy_anchor_loss(torch.empty(0, 2), torch.empty(0, dtype=torch.int64))
    with pytest.raises(ValueError, match="nonzero"):
        token_set_class_scores(
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[[1.0e-20, 0.0]]]),
            torch.ones(1, 1),
            torch.tensor([[1.0, 0.0]]),
            torch.tensor([[[1.0, 0.0]]]),
            set_weight=0.25,
        )
