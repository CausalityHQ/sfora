"""Tests for the optimization-only class-disjoint gradient diagnostic."""

from __future__ import annotations

import math

import pytest
import torch

from sfora.siglip_cdga import (
    build_cdga_domain_split,
    symmetric_conflict_projection,
)


class TestCDGAPrimitiveTests:
    """Mutation locks for the pseudo-domain and gradient arithmetic boundary."""

    def test_domain_split_is_disjoint_complete_and_seed_bound(self) -> None:
        first = build_cdga_domain_split(
            fit_labels=(0, 1, 2, 3, 4, 5),
            validation_labels=(6, 7),
            master_seed_sha256="0" * 64,
        )
        repeated = build_cdga_domain_split(
            fit_labels=(0, 1, 2, 3, 4, 5),
            validation_labels=(6, 7),
            master_seed_sha256="0" * 64,
        )
        rotated = build_cdga_domain_split(
            fit_labels=(0, 1, 2, 3, 4, 5),
            validation_labels=(6, 7),
            master_seed_sha256="0" * 62 + "01",
        )

        assert first == repeated
        assert first.domain_a_labels == (0, 2, 4)
        assert first.domain_b_labels == (1, 3, 5)
        assert set(first.domain_a_labels).isdisjoint(first.domain_b_labels)
        assert sorted(first.domain_a_labels + first.domain_b_labels) == list(range(6))
        assert set(first.validation_labels).isdisjoint(first.fit_labels)
        assert rotated.domain_a_labels == (1, 3, 5)
        assert rotated.domain_b_labels == (0, 2, 4)
        assert rotated.sha256 != first.sha256

    @pytest.mark.parametrize(
        ("fit_labels", "validation_labels", "seed"),
        [
            ((0, 1, 2), (3,), "0" * 64),
            ((0, 1, 1, 2), (3,), "0" * 64),
            ((0, 1, 2, 3), (3, 4), "0" * 64),
            ((0, 1, 2, 3), (4,), "g" * 64),
            ((False, 1, 2, 3), (4,), "0" * 64),
            ((0, 1, 2, 3), (4,), 0),
        ],
    )
    def test_domain_split_rejects_insufficient_overlap_or_type_drift(
        self,
        fit_labels: object,
        validation_labels: object,
        seed: object,
    ) -> None:
        with pytest.raises(ValueError, match="CDGA domain authority differs"):
            build_cdga_domain_split(
                fit_labels=fit_labels,  # type: ignore[arg-type]
                validation_labels=validation_labels,  # type: ignore[arg-type]
                master_seed_sha256=seed,  # type: ignore[arg-type]
            )

    def test_symmetric_projection_is_identity_without_conflict(self) -> None:
        left = torch.tensor([1.0, 0.0], dtype=torch.float32)
        right = torch.tensor([0.0, 2.0], dtype=torch.float32)

        result = symmetric_conflict_projection(left, right, epsilon=1.0e-12)

        assert result.conflict is False
        assert result.pre_projection_cosine == 0.0
        assert torch.equal(result.left, left)
        assert torch.equal(result.right, right)

    def test_symmetric_projection_removes_negative_cross_components(self) -> None:
        left = torch.tensor([1.0, 0.0], dtype=torch.float32)
        right = torch.tensor([-1.0, 1.0], dtype=torch.float32)

        result = symmetric_conflict_projection(left, right, epsilon=1.0e-12)

        assert result.conflict is True
        assert result.pre_projection_cosine == pytest.approx(-1.0 / math.sqrt(2.0))
        assert torch.equal(result.left, torch.tensor([0.5, 0.5]))
        assert torch.equal(result.right, torch.tensor([0.0, 1.0]))
        assert float(torch.dot(result.left, right)) == 0.0
        assert float(torch.dot(result.right, left)) == 0.0

    def test_symmetric_projection_handles_zero_and_rejects_invalid_inputs(self) -> None:
        zero = torch.zeros(3, dtype=torch.float32)
        finite = torch.tensor([1.0, -2.0, 3.0], dtype=torch.float32)
        result = symmetric_conflict_projection(zero, finite, epsilon=1.0e-12)
        assert result.conflict is False
        assert result.pre_projection_cosine == 0.0
        assert torch.equal(result.left, zero)
        assert torch.equal(result.right, finite)

        bad_cases = (
            (finite.double(), finite, 1.0e-12),
            (finite.reshape(1, 3), finite, 1.0e-12),
            (finite, finite[:2], 1.0e-12),
            (torch.tensor([math.nan, 0.0, 0.0]), finite, 1.0e-12),
            (finite, finite, 0.0),
            (finite, finite, True),
        )
        for left, right, epsilon in bad_cases:
            with pytest.raises(ValueError, match="CDGA gradient authority differs"):
                symmetric_conflict_projection(left, right, epsilon=epsilon)
