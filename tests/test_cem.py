import numpy as np
import torch

from sfora.cem import build_confusion_edges, confusion_edge_margin_loss


def test_build_confusion_edges_requires_cross_level_agreement() -> None:
    x = np.array([[1, 0], [1, 0], [0.9, 0.1], [0.9, 0.1], [0, 1], [0, 1]], dtype=float)
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.array([[1, 0], [0, 1], [0.9, 0.1]], dtype=float)
    proxy_labels = np.array([0, 1, 2])
    edges = build_confusion_edges(x, y, p, proxy_labels, min_support=1)
    assert 0 not in edges
    assert 1 not in edges


def test_cem_loss_only_moves_registered_target_proxy() -> None:
    sim = torch.tensor([[0.8, 0.2, 0.1]], requires_grad=True)
    labels = torch.tensor([0])
    proxy_labels = torch.tensor([0, 1, 2])
    loss = confusion_edge_margin_loss(sim, labels, proxy_labels, {0: (1, 1.0)}, torch_module=torch)
    loss.backward()
    assert sim.grad is not None
    assert sim.grad[0, 1] > 0
    assert sim.grad[0, 0] < 0
    assert sim.grad[0, 2] == 0
