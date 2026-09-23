"""The training scorer must use the deployed packed-code forward arithmetic."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from sfora.deployed_code_rank import (
    packed_cosine_ste,
    smooth_ap_float_loss,
    smooth_ap_packed_loss,
)
from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.unicom_rank_finish import smooth_ap_finish_loss


def test_packed_cosine_forward_matches_130_byte_wire() -> None:
    torch.manual_seed(14)
    features = torch.randn(8, 128, dtype=torch.float32, requires_grad=True)
    observed = packed_cosine_ste(features)
    wire = pack_int8_unit_embeddings(F.normalize(features.detach(), dim=1))
    expected = wire.cosine_similarity(wire)
    assert torch.equal(observed.detach(), expected)
    assert torch.all(observed.diag() > 0.999)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast parity")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_packed_cosine_forward_matches_wire_inside_cuda_autocast(dtype: torch.dtype) -> None:
    torch.manual_seed(25)
    features = torch.randn(32, 128, device="cuda", dtype=torch.float32)
    normalized = F.normalize(features, dim=1).cpu()
    wire = pack_int8_unit_embeddings(normalized)
    expected = wire.cosine_similarity(wire)
    with torch.autocast(device_type="cuda", dtype=dtype):
        observed = packed_cosine_ste(features)
    assert observed.dtype == torch.float32
    assert torch.equal(observed.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for autocast parity")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_float_and_packed_rank_losses_ignore_outer_cuda_autocast(dtype: torch.dtype) -> None:
    torch.manual_seed(29)
    features = torch.randn(8, 128, device="cuda", dtype=torch.float32)
    labels = (0, 0, 1, 1, 2, 2, 3, 3)
    expected_float = smooth_ap_float_loss(features, labels)
    expected_packed = smooth_ap_packed_loss(features, labels)
    with torch.autocast(device_type="cuda", dtype=dtype):
        observed_float = smooth_ap_float_loss(features, labels)
        observed_packed = smooth_ap_packed_loss(features, labels)
    assert torch.equal(observed_float, expected_float)
    assert torch.equal(observed_packed, expected_packed)


def test_packed_smooth_ap_has_finite_nonzero_gradient() -> None:
    torch.manual_seed(19)
    features = torch.randn(8, 128, dtype=torch.float32, requires_grad=True)
    loss = smooth_ap_packed_loss(features, (0, 0, 1, 1, 2, 2, 3, 3))
    loss.backward()
    assert torch.isfinite(loss)
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad) > 0


def test_float_control_preserves_128d_smooth_ap_objective() -> None:
    torch.manual_seed(21)
    features = torch.randn(8, 128, dtype=torch.float32, requires_grad=True)
    labels = (0, 0, 1, 1, 2, 2, 3, 3)
    observed = smooth_ap_float_loss(features, labels)
    reference = smooth_ap_finish_loss(F.pad(features, (0, 384)), labels)
    torch.testing.assert_close(observed, reference, rtol=0, atol=2e-6)
    observed.backward()
    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad) > 0


@pytest.mark.parametrize(
    "features,labels",
    [
        (torch.zeros(4, 128), (0, 0, 1, 1)),
        (torch.ones(4, 127), (0, 0, 1, 1)),
        (torch.ones(4, 128), (0, 1, 2, 3)),
    ],
)
def test_packed_smooth_ap_rejects_unusable_training_batches(
    features: torch.Tensor, labels: tuple[int, ...]
) -> None:
    with pytest.raises(ValueError):
        smooth_ap_packed_loss(features, labels)
