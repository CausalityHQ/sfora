"""Matched objective controls for full-backbone SOP compact training."""

from __future__ import annotations

import pytest
import torch

from sfora.deployed_code_rank import smooth_ap_float_loss, smooth_ap_packed_loss
from sfora.sop_compact_training import (
    CompactTrainingArm,
    compact_head_features,
    compact_training_loss,
    compact_training_terms,
    initialize_full_width_head_and_classifier,
)
from sfora.unicom_training import sharded_mask_arcface_loss


def _batch():
    torch.manual_seed(19)
    features = torch.randn(8, 128, requires_grad=True)
    weights = torch.nn.Parameter(torch.randn(2, 128))
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    masks = torch.arange(128, dtype=torch.int64).unsqueeze(0)
    return features, weights, labels, masks


@pytest.mark.parametrize(
    ("arm", "rank"),
    [
        (CompactTrainingArm.ARCFACE, None),
        (CompactTrainingArm.FLOAT_RANK, smooth_ap_float_loss),
        (CompactTrainingArm.PACKED_RANK, smooth_ap_packed_loss),
    ],
)
def test_compact_loss_retains_same_arcface_control(arm, rank):
    features, weights, labels, masks = _batch()
    result = compact_training_loss(features, weights, labels, masks, arm=arm)
    expected = sharded_mask_arcface_loss(features, weights, labels, masks, margin=0.3, scale=64.0)
    if rank is not None:
        expected = expected + 2.0 * rank(features, labels.tolist())
    torch.testing.assert_close(result, expected)
    result.backward()
    assert bool(torch.isfinite(features.grad).all())
    assert bool(torch.isfinite(weights.grad).all())
    assert float(features.grad.abs().sum()) > 0.0
    assert float(weights.grad.abs().sum()) > 0.0


def test_compact_loss_rejects_mismatched_positive_inventory():
    features, weights, labels, masks = _batch()
    labels[1:4] = 1
    with pytest.raises(ValueError, match="positive inventory"):
        compact_training_loss(features, weights, labels, masks, arm=CompactTrainingArm.PACKED_RANK)


def test_compact_head_is_invariant_to_positive_source_scale():
    torch.manual_seed(27)
    source = torch.randn(6, 768)
    head = torch.nn.Linear(768, 128)
    scales = torch.tensor([0.1, 0.4, 1.0, 2.0, 10.0, 25.0])
    actual = compact_head_features(source, head)
    scaled = compact_head_features(source * scales[:, None], head)
    torch.testing.assert_close(actual, scaled, rtol=1e-5, atol=1e-5)


def test_compact_head_accepts_siglip2_width_and_backpropagates():
    torch.manual_seed(2701)
    source = torch.randn(6, 1024, requires_grad=True)
    head = torch.nn.Linear(1024, 128)
    actual = compact_head_features(source, head)
    scaled = compact_head_features(source * 2.0, head)
    torch.testing.assert_close(actual, scaled, rtol=1e-5, atol=1e-5)
    actual.square().sum().backward()
    assert source.grad is not None and bool(torch.isfinite(source.grad).all())
    assert head.weight.grad is not None and bool(torch.isfinite(head.weight.grad).all())


@pytest.mark.parametrize("arm", tuple(CompactTrainingArm))
def test_compact_terms_reconstruct_total_without_extra_objective(arm):
    features, weights, labels, masks = _batch()
    control, rank = compact_training_terms(features, weights, labels, masks, arm=arm)
    total = compact_training_loss(features, weights, labels, masks, arm=arm)
    torch.testing.assert_close(control + 2.0 * rank, total)
    assert bool(torch.isfinite(control))
    assert bool(torch.isfinite(rank))
    if arm is CompactTrainingArm.ARCFACE:
        assert float(rank) == 0.0


def test_reference_scale_changes_only_the_shared_arcface_control():
    features, weights, labels, masks = _batch()
    control, rank = compact_training_terms(
        features,
        weights,
        labels,
        masks,
        arm=CompactTrainingArm.PACKED_RANK,
        arcface_margin=0.25,
        arcface_scale=32.0,
    )
    expected = sharded_mask_arcface_loss(features, weights, labels, masks, margin=0.25, scale=32.0)
    torch.testing.assert_close(control, expected)
    torch.testing.assert_close(rank, smooth_ap_packed_loss(features, labels.tolist()))


def test_full_width_identity_head_and_fit_only_class_imprint():
    torch.manual_seed(31)
    source = torch.randn(8, 768)
    labels = (9, 9, 9, 9, 3, 3, 3, 3)
    head, classifier = initialize_full_width_head_and_classifier(source, labels)
    projected = compact_head_features(source, head, output_dim=768)
    torch.testing.assert_close(projected, torch.nn.functional.normalize(source, dim=1))
    expected = torch.stack([projected[4:].sum(dim=0), projected[:4].sum(dim=0)])
    torch.testing.assert_close(classifier, torch.nn.functional.normalize(expected, dim=1))


def test_full_width_arcface_control_matches_direct_objective_and_gradients():
    torch.manual_seed(32)
    features = torch.randn(8, 768, requires_grad=True)
    classifier = torch.nn.Parameter(torch.randn(2, 768))
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    masks = torch.arange(768, dtype=torch.int64).unsqueeze(0)
    actual, rank = compact_training_terms(
        features,
        classifier,
        labels,
        masks,
        arm=CompactTrainingArm.ARCFACE,
        arcface_margin=0.25,
        arcface_scale=32.0,
        output_dim=768,
    )
    expected = sharded_mask_arcface_loss(
        features, classifier, labels, masks, margin=0.25, scale=32.0
    )
    torch.testing.assert_close(actual, expected)
    assert float(rank) == 0.0
    actual.backward()
    assert bool(torch.isfinite(features.grad).all())
    assert bool(torch.isfinite(classifier.grad).all())


def test_full_width_control_rejects_compact_only_rank_objective():
    features = torch.randn(8, 768)
    classifier = torch.randn(2, 768)
    labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1], dtype=torch.int64)
    masks = torch.arange(768, dtype=torch.int64).unsqueeze(0)
    with pytest.raises(ValueError, match="full-width control only supports ArcFace"):
        compact_training_terms(
            features,
            classifier,
            labels,
            masks,
            arm=CompactTrainingArm.PACKED_RANK,
            output_dim=768,
        )
