from __future__ import annotations

import pytest

from sfora.pfml import pfml_potential_loss

torch = pytest.importorskip("torch")


def test_pfml_matches_hand_computed_energy() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.8, 0.6]], dtype=torch.float32)
    labels = torch.tensor([0, 1])
    proxies = torch.tensor([[0.6, 0.8], [-1.0, 0.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([0, 1])

    loss = pfml_potential_loss(
        embeddings,
        labels,
        proxy_embeddings=proxies,
        proxy_labels=proxy_labels,
        delta=0.5,
        alpha=2.0,
        torch_module=torch,
    )

    expected = 2.0 * (4.0 - 1.25 + 4.0 + 12.5 - 1.0 / 3.6 + 4.0)
    assert float(loss.detach()) == pytest.approx(expected, rel=1e-5)


def test_pfml_updates_live_proxy_pairs_and_saturates_constant_pairs() -> None:
    embeddings = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, requires_grad=True)
    proxies = torch.nn.Parameter(
        torch.tensor([[1.0, 0.0, 0.0], [0.96, 0.28, 0.0]], dtype=torch.float32)
    )
    loss = pfml_potential_loss(
        embeddings,
        torch.tensor([2]),
        proxy_embeddings=proxies,
        proxy_labels=torch.tensor([0, 1]),
        delta=0.5,
        alpha=2.0,
        torch_module=torch,
    )
    loss.backward()
    assert proxies.grad is not None and float(proxies.grad.norm()) > 0.0
    assert embeddings.grad is not None and float(embeddings.grad.norm()) == pytest.approx(0.0)


def test_pfml_requires_coherent_proxy_and_tensor_authority() -> None:
    embeddings = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    labels = torch.tensor([0])
    with pytest.raises(ValueError, match="requires class proxies"):
        pfml_potential_loss(
            embeddings,
            labels,
            proxy_embeddings=None,
            proxy_labels=None,
            delta=0.5,
            alpha=2.0,
            torch_module=torch,
        )
    with pytest.raises(ValueError, match="authority"):
        pfml_potential_loss(
            embeddings,
            labels,
            proxy_embeddings=torch.tensor([[float("nan"), 0.0]]),
            proxy_labels=torch.tensor([0]),
            delta=0.5,
            alpha=2.0,
            torch_module=torch,
        )
