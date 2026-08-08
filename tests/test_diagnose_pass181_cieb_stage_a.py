"""Tiny focused tests for the frozen Pass181 CIEB Stage-A diagnostic."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_pass181_cieb_stage_a.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_pass181_cieb_stage_a", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_ownership_entropy_uses_class_balance_and_assigns_constant_coordinate_one() -> None:
    # Duplicating class 0 must not change a class-balanced estimator.
    descriptors = np.asarray(
        [
            [0.0, 7.0],
            [0.0, 7.0],
            [0.0, 7.0],
            [0.0, 7.0],
            [0.0, 7.0],
            [0.0, 7.0],
            [3.0, 7.0],
        ]
    )
    labels = np.asarray([0, 0, 0, 0, 1, 1, 2])

    result = _MODULE.ownership_entropy(descriptors, labels)

    expected_first = -(
        2.0 * (1.0 / 6.0) * np.log(1.0 / 6.0) + (4.0 / 6.0) * np.log(4.0 / 6.0)
    ) / np.log(3.0)
    assert result[0] == pytest.approx(expected_first)
    assert result[1] == pytest.approx(1.0)


def test_ownership_entropy_rejects_one_class_and_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="at least two"):
        _MODULE.ownership_entropy(np.asarray([[0.0], [1.0]]), np.asarray([4, 4]))
    with pytest.raises(ValueError, match="finite"):
        _MODULE.ownership_entropy(np.asarray([[0.0], [np.nan]]), np.asarray([4, 5]))


def test_fold_and_gallery_query_split_are_domain_separated_and_order_invariant() -> None:
    # Literals are independently SHA-256 checked from the frozen domains.
    assert [_MODULE.class_fold(label) for label in [0, 1, -7, 3996]] == [0, 0, 2, 0]
    ids = np.asarray(["coat-c", "coat-a", "coat-e", "coat-b", "coat-d"])

    gallery, query = _MODULE.split_identity(ids)

    assert ids[gallery].tolist() == ["coat-a", "coat-b", "coat-c"]
    assert ids[query].tolist() == ["coat-e", "coat-d"]
    permutation = np.asarray([3, 0, 4, 1, 2])
    shuffled = ids[permutation]
    gallery_2, query_2 = _MODULE.split_identity(shuffled)
    assert set(ids[gallery]) == set(shuffled[gallery_2])
    assert set(ids[query]) == set(shuffled[query_2])


def test_spearman_uses_average_ranks_and_maps_a_constant_vector_to_zero() -> None:
    assert _MODULE.spearman(
        np.asarray([1.0, 1.0, 3.0]), np.asarray([1.0, 2.0, 3.0])
    ) == pytest.approx(0.8660254037844387)
    assert _MODULE.spearman(np.ones(4), np.arange(4.0)) == 0.0


def test_matched_masks_preserve_every_stratum_count_and_are_reproducible() -> None:
    strata = np.repeat(np.arange(8, dtype=np.int64), 8)
    target = np.asarray([0, 8, 16, 24, 32, 40, 48, 56], dtype=np.int64)

    first = _MODULE.build_matched_masks(
        strata,
        target,
        seed=2,
        fold=4,
        mask_count=20,
    )
    second = _MODULE.build_matched_masks(
        strata,
        target,
        seed=2,
        fold=4,
        mask_count=20,
    )

    assert first.shape == (20, 8)
    assert first.dtype == np.uint16
    np.testing.assert_array_equal(first, second)
    assert len({tuple(row.tolist()) for row in first}) == 20
    assert all(tuple(row.tolist()) != tuple(target.tolist()) for row in first)
    for row in first:
        assert np.bincount(strata[row], minlength=8).tolist() == [1] * 8
    assert (
        _MODULE.canonical_mask_sha256(first)
        == "eb7856ca1bf3cf3a601e331374695dd1ef36f33df2704955c5c3a7801f53a094"
    )


def test_margin_effects_use_frozen_foreign_rows_for_target_and_controls() -> None:
    descriptors = np.asarray(
        [
            [2.0, 1.0, 1.0, 0.0],  # query, class 10
            [2.0, 1.0, 0.0, 0.0],  # positive gallery
            [0.0, 1.0, 1.0, 0.0],  # foreign gallery selected before ablation
            [0.0, 0.0, 1.0, 1.0],  # another foreign gallery
        ]
    )
    descriptors /= np.linalg.norm(descriptors, axis=1, keepdims=True)
    labels = np.asarray([10, 10, 20, 30])
    ids = np.asarray(["q", "p", "f1", "f2"])
    gallery = np.asarray([1, 2, 3])
    queries = np.asarray([0])

    result = _MODULE.score_mask_effects(
        descriptors,
        labels,
        ids,
        gallery,
        queries,
        np.asarray([[0], [3]], dtype=np.uint16),
        foreign_k=1,
        tau=0.05,
    )

    assert result["eligible_labels"].tolist() == [10]
    assert result["identity_effects"].shape == (1, 2)
    assert result["frozen_foreign_ids"] == [["f1"]]
    assert result["unablated_recall_at_1"] == pytest.approx(1.0)
    assert result["target_recall_at_1"] == pytest.approx(0.0)
    assert np.isfinite(result["identity_effects"]).all()


def test_vectorized_masked_similarities_equal_scalar_ablation() -> None:
    rng = np.random.default_rng(17)
    query = rng.normal(size=12)
    query /= np.linalg.norm(query)
    supports = rng.normal(size=(5, 12))
    supports /= np.linalg.norm(supports, axis=1, keepdims=True)
    masks = np.asarray([[0, 3, 8], [1, 6, 11], [2, 4, 9]], dtype=np.uint16)

    observed = _MODULE.masked_similarity_matrix(query, supports, masks, chunk_size=2)
    expected = np.asarray(
        [
            _MODULE._masked_rows(supports, mask)
            @ _MODULE._masked_rows(query[None, :], mask)[0]
            for mask in masks
        ]
    )

    np.testing.assert_allclose(observed, expected, atol=1.0e-12, rtol=1.0e-12)


def test_cross_seed_binding_rejects_label_changes_even_when_ids_match() -> None:
    first = SimpleNamespace(
        train_example_ids=np.asarray(["a", "b", "c"]),
        train_labels=np.asarray([1, 1, 2]),
    )
    changed = SimpleNamespace(
        train_example_ids=np.asarray(["a", "b", "c"]),
        train_labels=np.asarray([1, 2, 2]),
    )
    with pytest.raises(ValueError, match="labels differ"):
        _MODULE.validate_cross_seed_training_binding([first, changed])


def test_result_serialization_omits_full_control_matrix_but_keeps_verdict_inputs() -> None:
    payload = {
        "seed": 0,
        "labels": np.asarray([1, 2]),
        "target_identity_effects": np.asarray([0.2, 0.4]),
        "control_identity_effects": np.asarray([[0.1, 0.3], [0.2, 0.2]]),
        "D": 0.1,
    }

    serialized = _MODULE._json_seed_full(payload)

    assert "control_identity_effects" not in serialized
    assert serialized["matched_mean_identity_effects"] == pytest.approx([0.2, 0.2])
    assert serialized["identity_advantages"] == pytest.approx([0.0, 0.2])
    assert serialized["control_means"] == pytest.approx([0.15, 0.25])
    assert len(serialized["control_identity_effects_sha256"]) == 64


def _seed_effects(seed: int, target_shift: float) -> dict[str, object]:
    labels = np.arange(12, dtype=np.int64)
    controls = np.asarray(
        [
            [-0.02 + 0.0005 * label + 0.0002 * replicate for replicate in range(6)]
            for label in labels
        ],
        dtype=np.float64,
    )
    target = controls.mean(axis=1) + target_shift + 0.0001 * seed
    return {
        "seed": seed,
        "labels": labels,
        "target_identity_effects": target,
        "control_identity_effects": controls,
    }


def test_joint_verdict_passes_only_with_all_seed_deltas_and_positive_lower_bound() -> None:
    result = _MODULE.full_stage_a_verdict(
        [_seed_effects(seed, 0.01) for seed in range(4)],
        stabilities=[0.8] * 4,
        cvs=[0.2] * 4,
        bootstrap_replicates=500,
    )

    assert result["stage_a"] == "PASS_ONWARD"
    assert result["D_pooled"] == pytest.approx(0.01015)
    assert result["bootstrap_95_lower_bound"] > 0.0
    assert all(value > 0.0 for value in result["D_by_seed"].values())


def test_early_entropy_fail_precedes_mask_stage_and_full_nonpositive_effect_fails() -> None:
    early = _MODULE.entropy_stage_verdict(
        stabilities=[0.29, 0.20, 0.10, 0.80],
        cvs=[0.2, 0.2, 0.2, 0.2],
    )
    assert early["stop_before_matched_masks"] is True
    assert early["stage_a"] == "FAIL"

    full = _MODULE.full_stage_a_verdict(
        [_seed_effects(seed, -0.001) for seed in range(4)],
        stabilities=[0.8] * 4,
        cvs=[0.2] * 4,
        bootstrap_replicates=100,
    )
    assert full["stage_a"] == "FAIL"
    assert full["D_pooled"] < 0.0
