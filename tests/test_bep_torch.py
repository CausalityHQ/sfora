import pytest


torch = pytest.importorskip("torch")

from sfora.image_end_to_end import _barrier_energy_loss


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
