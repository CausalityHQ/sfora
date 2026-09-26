"""The synthetic classes must stay outside the retrieval bank and keep gradients."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import train_inshop_siglip2_unseen_gallery as trainer  # noqa: E402


def test_proxy_synthesis_interpolates_distinct_classes_with_gradients() -> None:
    features = torch.randn(8, 128, requires_grad=True)
    proxies = torch.nn.Parameter(torch.randn(4, 128))
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    result = trainer.synthesize_proxy_batch(
        features, proxies, labels, np.random.default_rng(179024)
    )
    all_features, all_proxies, all_labels, partners, lambdas = result
    assert all_features.shape == (16, 128)
    assert all_proxies.shape == (12, 128)
    assert all_labels.tolist() == labels.tolist() + list(range(4, 12))
    assert all(labels[partner] != labels[row] for row, partner in enumerate(partners))
    assert np.all((lambdas > 0) & (lambdas < 1))
    for row, partner in enumerate(partners):
        expected = lambdas[row] * features[row] + (1 - lambdas[row]) * features[partner]
        torch.testing.assert_close(all_features[8 + row], expected)
        expected_proxy = (
            lambdas[row] * proxies[labels[row]] + (1 - lambdas[row]) * proxies[labels[partner]]
        )
        torch.testing.assert_close(all_proxies[4 + row], expected_proxy)
    (all_features[8:].square().sum() + all_proxies[4:].square().sum()).backward()
    assert features.grad is not None and bool(torch.isfinite(features.grad).all())
    assert proxies.grad is not None and bool(torch.isfinite(proxies.grad).all())
    assert float(features.grad.abs().sum()) > 0
    assert float(proxies.grad.abs().sum()) > 0


def test_proxy_synthesis_rejects_single_class_batch() -> None:
    with pytest.raises(ValueError, match="different classes"):
        trainer.synthesize_proxy_batch(
            torch.randn(2, 128),
            torch.randn(1, 128),
            torch.zeros(2, dtype=torch.long),
            np.random.default_rng(1),
        )
