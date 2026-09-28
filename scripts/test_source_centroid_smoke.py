import torch
from run_inshop_source_centroid_smoke import probe_failures
from torch.nn import functional as F
from train_inshop_siglip2_unseen_gallery import frozen_source_centroid_loss

from sfora.unicom_training import sharded_mask_arcface_loss


def test_smoke_requires_both_weak_probes_but_stops_either_dominant_probe():
    def probe(ratio, saturated=0):
        return {
            "weighted_auxiliary_to_main_gradient_ratio": ratio,
            "saturated_target_fraction": saturated,
        }

    assert probe_failures([probe(0.005), probe(0.005)]) == ["weak_auxiliary_encoder_gradient"]
    assert probe_failures([probe(0.005), probe(0.02)]) == []
    assert probe_failures([probe(0.02), probe(0.26)]) == ["dominant_auxiliary_encoder_gradient"]
    assert probe_failures([probe(0.02, 0.6), probe(0.02, 0.6)]) == ["saturated_auxiliary_targets"]


def test_frozen_source_loss_native_parity_and_gradient_routing():
    torch.manual_seed(7)
    source = torch.randn(4, 6, requires_grad=True)
    mean = torch.randn(6, requires_grad=True)
    prototypes = torch.randn(3, 6, requires_grad=True)
    head = torch.nn.Linear(6, 2)
    target = torch.tensor([0, 1, 2, 0])
    head(source)  # Main-path parameters must receive no auxiliary gradient.
    loss = frozen_source_centroid_loss(source, mean, prototypes, target)
    expected = sharded_mask_arcface_loss(
        F.normalize(source, dim=1) - mean.detach(),
        prototypes.detach(),
        target,
        torch.arange(6).unsqueeze(0),
        margin=0,
        scale=64,
    )
    assert torch.allclose(loss, expected)
    actual_grad = torch.autograd.grad(loss, source, retain_graph=True)[0]
    expected_grad = torch.autograd.grad(expected, source)[0]
    assert torch.allclose(actual_grad, expected_grad)
    loss.backward()
    assert source.grad is not None and source.grad.norm() > 0
    assert mean.grad is None and prototypes.grad is None and head.weight.grad is None
