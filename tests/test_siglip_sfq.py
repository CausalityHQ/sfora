"""Tests for the optimization-only Shrunk-Fisher-Quotient diagnostic."""

from __future__ import annotations

import torch

from sfora.siglip_head_screen import build_feature_split_authority
from sfora.siglip_sfq import build_sfq_fold_schedule


def _four_pair_features() -> tuple[torch.Tensor, torch.Tensor]:
    centers = (
        (1.00, 0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
        (1.00, -0.05, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00),
        (0.00, 0.00, 1.00, 0.05, 0.00, 0.00, 0.00, 0.00),
        (0.00, 0.00, 1.00, -0.05, 0.00, 0.00, 0.00, 0.00),
        (0.00, 0.00, 0.00, 0.00, 1.00, 0.05, 0.00, 0.00),
        (0.00, 0.00, 0.00, 0.00, 1.00, -0.05, 0.00, 0.00),
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, 0.05),
        (0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 1.00, -0.05),
    )
    rows: list[torch.Tensor] = []
    labels: list[int] = []
    for label, center in enumerate(centers):
        base = torch.tensor(center, dtype=torch.float32)
        for offset in (-0.01, 0.0, 0.01):
            row = base.clone()
            row[(2 * (label // 2) + 1) % row.numel()] += offset
            rows.append(row)
            labels.append(label)
    return torch.stack(rows), torch.tensor(labels, dtype=torch.int64)


def _authority(features: torch.Tensor, *, role: str = "optimization-train"):
    return build_feature_split_authority(
        source_manifest_sha256="1" * 64,
        role=role,
        official_test_access=False,
        ordered_example_ids=tuple(f"row-{row:04d}" for row in range(features.shape[0])),
        features=features,
    )


def test_sfq_folds_are_deterministic_class_disjoint_and_keep_twin_pairs() -> None:
    """A broken edge order or allocator must not leak classes or split nearest twins."""

    features, labels = _four_pair_features()

    first = build_sfq_fold_schedule(features, labels, _authority(features), fold_count=4)
    second = build_sfq_fold_schedule(features, labels, _authority(features), fold_count=4)

    assert first == second
    assert len(first.sha256) == 64
    assert sorted(label for fold in first.folds for label in fold.validation_labels) == list(
        range(8)
    )
    assert all(set(fold.fit_labels).isdisjoint(fold.validation_labels) for fold in first.folds)
    assert {frozenset(fold.validation_labels) for fold in first.folds} == {
        frozenset((0, 1)),
        frozenset((2, 3)),
        frozenset((4, 5)),
        frozenset((6, 7)),
    }


def test_sfq_authority_rejects_evaluation_role_and_feature_drift() -> None:
    """Removing role or digest validation must expose this test to forbidden input."""

    features, labels = _four_pair_features()
    try:
        build_sfq_fold_schedule(
            features,
            labels,
            _authority(features, role="clean-validation"),
            fold_count=4,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("SFQ accepted evaluation-role authority")

    authority = _authority(features)
    features[0, 0] += 0.25
    try:
        build_sfq_fold_schedule(features, labels, authority, fold_count=4)
    except ValueError:
        pass
    else:
        raise AssertionError("SFQ accepted post-authority feature drift")
