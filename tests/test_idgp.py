from __future__ import annotations

import hashlib
import struct

import numpy as np
import pytest

from sfora.idgp import (
    assign_label_folds,
    build_cohorts,
    global_tangent,
    local_excess_tangent,
    normalize_rows,
)


def _independent_fold(label: int) -> int:
    digest = hashlib.sha256(b"LE-IDGP-fold-v1:" + struct.pack("<q", label)).digest()
    return digest[0] & 3


def _labels_in_fold(fold: int, count: int) -> list[int]:
    result: list[int] = []
    label = 0
    while len(result) < count:
        if _independent_fold(label) == fold:
            result.append(label)
        label += 1
    return result


def test_assign_label_folds_uses_frozen_low_bit_hash() -> None:
    labels = np.asarray([91, -7, 0, 91, 12], dtype=np.int64)
    actual = assign_label_folds(labels)
    assert actual == {label: _independent_fold(label) for label in sorted(set(labels.tolist()))}

    with pytest.raises(ValueError, match="int64"):
        assign_label_folds(labels.astype(np.int32))
    with pytest.raises(ValueError, match="one-dimensional"):
        assign_label_folds(labels[:, None])


def test_build_cohorts_is_order_invariant_and_uses_exact_rows() -> None:
    selected_labels = _labels_in_fold(2, 91)
    rows: list[tuple[int, str]] = []
    for label in selected_labels:
        for suffix in ("echo", "alpha", "delta", "bravo", "charlie"):
            rows.append((label, f"label-{label}-{suffix}"))
    labels = np.asarray([row[0] for row in rows], dtype=np.int64)
    ids = np.asarray([row[1] for row in rows])

    cohorts = build_cohorts(labels, ids)
    assert len(cohorts) == 2
    assert [(cohort.fold, cohort.index) for cohort in cohorts] == [(2, 0), (2, 1)]
    assert all(cohort.row_indices.shape == (180,) for cohort in cohorts)
    assert all(cohort.reference_row_indices.shape == (180,) for cohort in cohorts)
    assert set(labels[cohorts[0].row_indices].tolist()).isdisjoint(
        labels[cohorts[0].reference_row_indices].tolist()
    )

    for label in np.unique(labels[cohorts[0].row_indices]):
        candidates = ids[labels == label].tolist()
        expected = sorted(
            candidates,
            key=lambda value: (
                hashlib.sha256(b"LE-IDGP-row-v1:" + value.encode("utf-8")).digest(),
                value,
            ),
        )[:4]
        assert sorted(
            ids[cohorts[0].row_indices][labels[cohorts[0].row_indices] == label]
        ) == sorted(expected)

    permutation = np.random.default_rng(8).permutation(labels.size)
    permuted = build_cohorts(labels[permutation], ids[permutation])
    assert [ids[c.row_indices].tolist() for c in cohorts] == [
        ids[permutation][c.row_indices].tolist() for c in permuted
    ]
    assert [c.ordered_sha256 for c in cohorts] == [c.ordered_sha256 for c in permuted]


def test_build_cohorts_rejects_invalid_ids_and_insufficient_complete_cohorts() -> None:
    labels = np.repeat(np.asarray(_labels_in_fold(0, 45), dtype=np.int64), 4)
    ids = np.asarray([f"id-{index}" for index in range(labels.size)])
    with pytest.raises(ValueError, match="two complete cohorts"):
        build_cohorts(labels, ids)

    enough_labels = np.repeat(np.asarray(_labels_in_fold(0, 90), dtype=np.int64), 4)
    duplicate_ids = np.asarray([f"id-{index}" for index in range(enough_labels.size)])
    duplicate_ids[1] = duplicate_ids[0]
    with pytest.raises(ValueError, match="unique"):
        build_cohorts(enough_labels, duplicate_ids)
    with pytest.raises(ValueError, match="Unicode"):
        build_cohorts(enough_labels, np.arange(enough_labels.size, dtype=np.int64))


def _density(anchor: np.ndarray, peers: np.ndarray, peer_ids: np.ndarray, k: int) -> float:
    similarities = peers @ anchor
    order = sorted(range(peers.shape[0]), key=lambda index: (-similarities[index], peer_ids[index]))
    return float(similarities[order[:k]].mean() - similarities.mean())


def test_local_excess_tangent_matches_stopped_peer_finite_difference() -> None:
    z = normalize_rows(
        np.asarray(
            [
                [1.0, 0.2, -0.1],
                [0.9, 0.3, 0.1],
                [0.8, 0.5, -0.2],
                [0.1, 1.0, 0.1],
                [-0.4, 0.7, 0.6],
                [-0.8, 0.1, 0.5],
            ],
            dtype=np.float64,
        )
    )
    labels = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    ids = np.asarray(["a", "b", "c", "d", "e", "f"])
    tangent, density = local_excess_tangent(z, labels, ids, k=2)

    peers = z[labels != labels[0]]
    peer_ids = ids[labels != labels[0]]
    numeric = np.empty(z.shape[1], dtype=np.float64)
    for axis in range(z.shape[1]):
        offset = np.zeros(z.shape[1], dtype=np.float64)
        offset[axis] = 1e-6
        plus = normalize_rows((z[0] + offset)[None, :])[0]
        minus = normalize_rows((z[0] - offset)[None, :])[0]
        numeric[axis] = (
            _density(plus, peers, peer_ids, 2) - _density(minus, peers, peer_ids, 2)
        ) / (2e-6)
    np.testing.assert_allclose(tangent[0], numeric, rtol=2e-5, atol=2e-7)
    np.testing.assert_allclose(np.sum(tangent * z, axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(density[0], _density(z[0], peers, peer_ids, 2), rtol=0.0, atol=1e-15)


def test_local_plus_global_recovers_topk_mean_tangent_with_stable_ties() -> None:
    z = normalize_rows(
        np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.5, 0.5], [0.5, -0.5], [-1.0, 0.0]],
            dtype=np.float64,
        )
    )
    labels = np.asarray([0, 0, 1, 2, 3], dtype=np.int64)
    ids = np.asarray(["anchor", "positive", "z-tie", "a-tie", "far"])
    local, _ = local_excess_tangent(z, labels, ids, k=1)
    global_value = global_tangent(z, labels)
    selected = z[3]
    expected_top = selected - z[0] * float(z[0] @ selected)
    np.testing.assert_allclose(local[0] + global_value[0], expected_top, atol=1e-12)
