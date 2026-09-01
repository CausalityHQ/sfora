"""Tests for the optimization-only class-disjoint gradient diagnostic."""

from __future__ import annotations

import json
import math

import pytest
import torch

import sfora.siglip_cdga as cdga_module
from sfora.siglip_cdga import (
    CDGAGradientProjection,
    _batch_schedule,
    _matched_gradients,
    build_cdga_domain_split,
    run_cdga_fold_diagnostic,
    symmetric_conflict_projection,
    train_cdga_fold,
    validate_cdga_result_bytes,
)
from sfora.siglip_head_screen import build_feature_split_authority
from sfora.siglip_sfq import build_sfq_fold_schedule


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

    def test_matched_gradient_reducer_changes_only_conflicting_projection(self) -> None:
        left_projection = torch.tensor([1.0, 0.0], dtype=torch.float32)
        right_projection = torch.tensor([0.0, 2.0], dtype=torch.float32)
        left_proxy = torch.tensor([3.0, -1.0], dtype=torch.float32)
        right_proxy = torch.tensor([1.0, 5.0], dtype=torch.float32)

        no_conflict = _matched_gradients(
            left_projection,
            right_projection,
            left_proxy,
            right_proxy,
            epsilon=1.0e-12,
        )

        assert torch.equal(no_conflict.comparator_projection, no_conflict.cdga_projection)
        assert torch.equal(no_conflict.proxy, torch.tensor([2.0, 2.0]))
        assert no_conflict.conflict is False

        conflict = _matched_gradients(
            left_projection,
            torch.tensor([-1.0, 1.0]),
            left_proxy,
            right_proxy,
            epsilon=1.0e-12,
        )
        assert conflict.conflict is True
        assert not torch.equal(conflict.comparator_projection, conflict.cdga_projection)
        assert torch.equal(conflict.proxy, no_conflict.proxy)


def _cached_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    rows = []
    labels = []
    for label in range(8):
        for instance in range(4):
            row = torch.zeros(8, dtype=torch.float32)
            row[label] = 2.0
            row[(label + 1) % 8] = 0.03 * (instance - 1.5)
            row[(label + 3) % 8] = 0.01 * (instance + 1)
            rows.append(row)
            labels.append(label)
    return torch.stack(rows), torch.tensor(labels, dtype=torch.int64)


def _authority(features: torch.Tensor):
    return build_feature_split_authority(
        source_manifest_sha256="1" * 64,
        role="optimization-train",
        official_test_access=False,
        ordered_example_ids=tuple(f"row-{index:03d}" for index in range(features.shape[0])),
        features=features,
    )


class TestCDGATrainingTests:
    """The matched arms must differ only at the projection-gradient reducer."""

    def test_fold_training_is_deterministic_and_excludes_validation_labels(self) -> None:
        features, labels = _cached_fixture()
        authority = _authority(features)
        fold = build_sfq_fold_schedule(features, labels, authority, fold_count=4).folds[0]

        first = train_cdga_fold(
            features,
            labels,
            fold=fold,
            master_seed_sha256="2" * 64,
            output_dimensions=4,
            train_steps=3,
            examples_per_class=2,
            projection_learning_rate=1.0e-3,
            proxy_learning_rate=1.0e-2,
            weight_decay=1.0e-4,
            alpha=32.0,
            delta=0.1,
            device=torch.device("cpu"),
        )
        repeated = train_cdga_fold(
            features,
            labels,
            fold=fold,
            master_seed_sha256="2" * 64,
            output_dimensions=4,
            train_steps=3,
            examples_per_class=2,
            projection_learning_rate=1.0e-3,
            proxy_learning_rate=1.0e-2,
            weight_decay=1.0e-4,
            alpha=32.0,
            delta=0.1,
            device=torch.device("cpu"),
        )

        assert first.fit_labels == fold.fit_labels
        assert first.validation_labels == fold.validation_labels
        assert set(first.fit_labels).isdisjoint(first.validation_labels)
        assert first.trained_example_count == 24
        assert first.train_steps == 3
        assert first.examples_per_class == 2
        assert first.batch_schedule_sha256 == repeated.batch_schedule_sha256
        assert torch.equal(first.initial_weight, repeated.initial_weight)
        assert torch.equal(first.comparator_weight, repeated.comparator_weight)
        assert torch.equal(first.cdga_weight, repeated.cdga_weight)
        assert math.isfinite(first.mean_pre_projection_cosine)
        assert 0 <= first.conflict_count <= first.train_steps
        assert first.comparator_final_loss <= first.comparator_initial_loss
        assert first.cdga_final_loss <= first.cdga_initial_loss

    def test_fold_training_rejects_validation_or_hyperparameter_drift(self) -> None:
        features, labels = _cached_fixture()
        authority = _authority(features)
        fold = build_sfq_fold_schedule(features, labels, authority, fold_count=4).folds[0]
        for mutation in (
            {"train_steps": 0},
            {"examples_per_class": True},
            {"projection_learning_rate": 0.0},
            {"device": torch.device("meta")},
        ):
            arguments = {
                "fold": fold,
                "master_seed_sha256": "2" * 64,
                "output_dimensions": 4,
                "train_steps": 2,
                "examples_per_class": 2,
                "projection_learning_rate": 1.0e-3,
                "proxy_learning_rate": 1.0e-2,
                "weight_decay": 1.0e-4,
                "alpha": 32.0,
                "delta": 0.1,
                "device": torch.device("cpu"),
            }
            arguments.update(mutation)
            with pytest.raises(ValueError, match="CDGA training authority differs"):
                train_cdga_fold(features, labels, **arguments)

    def test_fold_training_is_invariant_to_held_feature_values(self) -> None:
        features, labels = _cached_fixture()
        authority = _authority(features)
        fold = build_sfq_fold_schedule(features, labels, authority, fold_count=4).folds[0]
        mutated = features.clone()
        validation_mask = torch.isin(
            labels, torch.tensor(fold.validation_labels, dtype=torch.int64)
        )
        mutated[validation_mask] = torch.flip(mutated[validation_mask], dims=(1,)) * 17.0
        arguments = {
            "fold": fold,
            "master_seed_sha256": "2" * 64,
            "output_dimensions": 4,
            "train_steps": 2,
            "examples_per_class": 2,
            "projection_learning_rate": 1.0e-3,
            "proxy_learning_rate": 1.0e-2,
            "weight_decay": 1.0e-4,
            "alpha": 32.0,
            "delta": 0.1,
            "device": torch.device("cpu"),
        }

        original = train_cdga_fold(features, labels, **arguments)
        changed = train_cdga_fold(mutated, labels, **arguments)

        assert original.batch_schedule_sha256 == changed.batch_schedule_sha256
        assert torch.equal(original.initial_weight, changed.initial_weight)
        assert torch.equal(original.comparator_weight, changed.comparator_weight)
        assert torch.equal(original.cdga_weight, changed.cdga_weight)

    def test_batches_use_only_fit_rows_and_identity_reducer_keeps_arms_equal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        features, labels = _cached_fixture()
        authority = _authority(features)
        fold = build_sfq_fold_schedule(features, labels, authority, fold_count=4).folds[0]
        fit_mask = torch.isin(labels, torch.tensor(fold.fit_labels, dtype=torch.int64))
        fit_labels = labels[fit_mask].contiguous()
        split = build_cdga_domain_split(
            fit_labels=fold.fit_labels,
            validation_labels=fold.validation_labels,
            master_seed_sha256="2" * 64,
        )
        schedule = _batch_schedule(
            fit_labels,
            split,
            train_steps=2,
            examples_per_class=2,
            fold_ordinal=fold.ordinal,
        )
        assert all(
            int(fit_labels[row]) in fold.fit_labels
            for pair in schedule
            for rows in pair
            for row in rows
        )

        def identity_projection(
            left: torch.Tensor, right: torch.Tensor, *, epsilon: float
        ) -> CDGAGradientProjection:
            assert epsilon == 1.0e-12
            return CDGAGradientProjection(left.clone(), right.clone(), False, 0.0)

        monkeypatch.setattr(cdga_module, "symmetric_conflict_projection", identity_projection)
        trained = train_cdga_fold(
            features,
            labels,
            fold=fold,
            master_seed_sha256="2" * 64,
            output_dimensions=4,
            train_steps=2,
            examples_per_class=2,
            projection_learning_rate=1.0e-3,
            proxy_learning_rate=1.0e-2,
            weight_decay=1.0e-4,
            alpha=32.0,
            delta=0.1,
            device=torch.device("cpu"),
        )
        assert trained.conflict_count == 0
        assert torch.equal(trained.comparator_weight, trained.cdga_weight)


class TestCDGAResultTests:
    """Canonical evidence must be reconstructed from integer fold primitives."""

    def test_result_recomputes_fold_counts_gates_and_canonical_bytes(self) -> None:
        features, labels = _cached_fixture()
        raw = run_cdga_fold_diagnostic(
            features,
            labels,
            split_authority=_authority(features),
            feature_cache_manifest_sha256="3" * 64,
            master_seed_sha256="2" * 64,
            output_dimensions=4,
            fold_count=4,
            train_steps=2,
            examples_per_class=2,
            device="cpu",
        )
        result = validate_cdga_result_bytes(raw)

        assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
        assert result.schema == "sfora-siglip-cdga-fold-diagnostic-v1"
        assert result.claim_eligible is False
        assert result.official_test_access is False
        assert result.query_count == 32
        assert len(result.folds) == 4
        assert result.raw_hits == sum(fold.raw_hits for fold in result.folds)
        assert result.comparator_hits == sum(fold.comparator_hits for fold in result.folds)
        assert result.cdga_hits == sum(fold.cdga_hits for fold in result.folds)
        assert result.cdga_minus_comparator_ppm == (
            result.cdga_recall_ppm - result.comparator_recall_ppm
        )
        assert result.passed is (
            result.valid
            and result.cdga_minus_comparator_ppm >= 2_000
            and result.cdga_hits >= result.spectral_hits
            and all(fold.cdga_minus_comparator_ppm >= -10_000 for fold in result.folds)
        )

    @pytest.mark.parametrize("field", ["query_count", "cdga_hits", "passed", "fold_count"])
    def test_result_rejects_aggregate_and_gate_mutations(self, field: str) -> None:
        features, labels = _cached_fixture()
        raw = run_cdga_fold_diagnostic(
            features,
            labels,
            split_authority=_authority(features),
            feature_cache_manifest_sha256="3" * 64,
            master_seed_sha256="2" * 64,
            output_dimensions=4,
            fold_count=4,
            train_steps=1,
            examples_per_class=2,
            device="cpu",
        )
        value = json.loads(raw)
        if field == "passed":
            value[field] = not value[field]
        else:
            value[field] += 1
        mutated = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with pytest.raises(ValueError):
            validate_cdga_result_bytes(mutated)

    def test_result_rejects_noncomplement_fit_partition(self) -> None:
        features, labels = _cached_fixture()
        raw = run_cdga_fold_diagnostic(
            features,
            labels,
            split_authority=_authority(features),
            feature_cache_manifest_sha256="3" * 64,
            master_seed_sha256="2" * 64,
            output_dimensions=4,
            fold_count=4,
            train_steps=1,
            examples_per_class=2,
            device="cpu",
        )
        value = json.loads(raw)
        value["folds"][0]["fit_labels"] = value["folds"][0]["fit_labels"][:-1]
        mutated = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

        with pytest.raises(ValueError, match="CDGA fold partition differs"):
            validate_cdga_result_bytes(mutated)

    def test_result_rejects_domain_split_digest_drift(self) -> None:
        features, labels = _cached_fixture()
        raw = run_cdga_fold_diagnostic(
            features,
            labels,
            split_authority=_authority(features),
            feature_cache_manifest_sha256="3" * 64,
            master_seed_sha256="2" * 64,
            output_dimensions=4,
            fold_count=4,
            train_steps=1,
            examples_per_class=2,
            device="cpu",
        )
        value = json.loads(raw)
        value["folds"][0]["domain_split_sha256"] = "f" * 64
        mutated = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()

        with pytest.raises(ValueError, match="CDGA domain split evidence differs"):
            validate_cdga_result_bytes(mutated)
