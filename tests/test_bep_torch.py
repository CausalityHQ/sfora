import pytest
import numpy as np


torch = pytest.importorskip("torch")

from sfora.image_end_to_end import _barrier_energy_loss
from sfora.losses import barrier_energy_loss


def test_barrier_energy_torch_is_finite_and_has_gradient() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [0.7, 0.7], [-1.0, 0.0]],
        dtype=torch.float32,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    proxies = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
    )
    proxy_labels = torch.tensor([0, 1, 2], dtype=torch.long)
    value = _barrier_energy_loss(
        embeddings,
        labels,
        proxies,
        proxy_labels,
        temperature=0.1,
        path_points=9,
        min_class_size=2,
        torch_module=torch,
    )
    assert torch.isfinite(value)
    assert float(value) > 0.0
    value.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert float(embeddings.grad.abs().sum()) > 0.0


def test_barrier_energy_torch_skips_classes_without_a_proxy() -> None:
    embeddings = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    labels = torch.tensor([7, 7], dtype=torch.long)
    proxies = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    proxy_labels = torch.tensor([3], dtype=torch.long)
    value = _barrier_energy_loss(
        embeddings,
        labels,
        proxies,
        proxy_labels,
        temperature=0.1,
        path_points=3,
        min_class_size=2,
        torch_module=torch,
    )
    assert float(value) == pytest.approx(0.0)


def test_barrier_energy_torch_matches_numpy_reference() -> None:
    embeddings = torch.tensor(
        [[1.0, 0.0], [0.3, 0.95], [-1.0, 0.0], [-0.3, 0.95]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    proxies = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float64
    )
    proxy_labels = torch.tensor([0, 1, 2], dtype=torch.long)
    got = _barrier_energy_loss(
        embeddings, labels, proxies, proxy_labels, temperature=0.1,
        path_points=5, min_class_size=2, torch_module=torch
    )
    # The Torch objective uses cyclic within-class pairs; construct the same
    # aligned NumPy pairs explicitly for a direct reference comparison.
    expected = barrier_energy_loss(
        np.asarray([[1.0, 0.0], [0.3, 0.95], [-1.0, 0.0], [-0.3, 0.95]]),
        np.asarray([[0.3, 0.95], [1.0, 0.0], [-0.3, 0.95], [-1.0, 0.0]]),
        proxies.numpy(), np.asarray([0, 0, 1, 1]), temperature=0.1, path_points=5,
    )
    assert float(got.detach()) == pytest.approx(expected, rel=1e-10, abs=1e-10)
