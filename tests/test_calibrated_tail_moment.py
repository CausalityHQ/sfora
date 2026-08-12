from __future__ import annotations

import numpy as np
import pytest

from sfora.calibrated_tail_moment import (
    encode_tail_moment,
    fit_tail_moment,
    head_neighbor_pairs,
)


def _unit_fixture() -> np.ndarray:
    return np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.8, 0.0, 0.0, 0.6],
            [0.0, 0.8, 0.6, 0.0],
            [0.0, 0.8, 0.0, 0.6],
        ],
        dtype=np.float32,
    )


def test_head_pairs_use_only_head_inner_product_and_stable_row_ties() -> None:
    unit = np.asarray(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )

    pairs = head_neighbor_pairs(unit, width=2, neighbors=1)

    assert pairs.tolist() == [[0, 1], [1, 0], [2, 0]]


def test_fit_matches_the_registered_ordered_fp64_formula() -> None:
    unit = np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.8, 0.0, 0.0, 0.6],
            [0.0, 0.8, 0.6, 0.0],
        ],
        dtype=np.float32,
    )
    fit = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)
    pairs = fit.pairs
    tail = unit[:, 2:].astype(np.float64)
    radius = np.linalg.norm(tail, axis=1)
    x = radius[pairs[:, 0]] * radius[pairs[:, 1]]
    y = np.sum(tail[pairs[:, 0]] * tail[pairs[:, 1]], axis=1, dtype=np.float64)
    expected = float(np.sum(x * y, dtype=np.float64) / np.sum(x * x, dtype=np.float64))

    assert fit.lambda_raw == expected
    assert fit.lambda_value == min(1.0, max(0.0, expected))
    encoded = encode_tail_moment(unit, fit, basis_kind="native")
    assert encoded.shape == (3, 3)
    assert encoded.dtype == np.float32


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_fit_rejects_nonfinite_values(bad: float) -> None:
    values = np.eye(3, dtype=np.float32)
    values[0, 0] = bad

    with pytest.raises(ValueError, match="finite"):
        fit_tail_moment(values, width=1, basis_kind="native", neighbors=1)


def test_zero_denominator_has_zero_coefficient_and_exact_zero_scalar() -> None:
    unit = np.eye(3, dtype=np.float32)
    fit = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)

    encoded = encode_tail_moment(unit, fit, basis_kind="native")

    assert fit.lambda_value == 0.0
    assert encoded[:, -1].tolist() == [0.0, 0.0, 0.0]


def test_pair_selection_cannot_observe_tail() -> None:
    unit = _unit_fixture()
    changed = unit.copy()
    changed[:, 2:] = changed[[1, 0, 3, 2], 2:]

    assert np.array_equal(
        head_neighbor_pairs(unit, width=2, neighbors=1),
        head_neighbor_pairs(changed, width=2, neighbors=1),
    )


def test_negative_raw_fit_clips_to_zero_and_encoding_is_exact_lambda_zero() -> None:
    unit = np.asarray(
        [[0.8, 0.0, 0.6, 0.0], [0.8, 0.0, -0.6, 0.0]], dtype=np.float32
    )
    fit = fit_tail_moment(unit, width=2, basis_kind="native", neighbors=1)

    encoded = encode_tail_moment(unit, fit, basis_kind="native")

    assert fit.lambda_raw < 0.0
    assert fit.lambda_value == 0.0
    assert encoded[:, -1].tobytes() == np.zeros(2, dtype=np.float32).tobytes()


def test_encoding_rejects_wrong_dimension_and_basis() -> None:
    fit = fit_tail_moment(_unit_fixture(), width=2, basis_kind="native", neighbors=1)

    with pytest.raises(ValueError, match="basis/dimension"):
        encode_tail_moment(_unit_fixture(), fit, basis_kind="pca")
    with pytest.raises(ValueError, match="basis/dimension"):
        encode_tail_moment(np.eye(3, dtype=np.float32), fit, basis_kind="native")


def test_neighbor_selection_only_sorts_the_requested_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    angles = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False, dtype=np.float32)
    unit = np.stack((np.cos(angles), np.sin(angles), np.zeros(64)), axis=1).astype(
        np.float32
    )
    original = np.lexsort
    sizes: list[int] = []

    def bounded(keys):
        sizes.append(len(keys[0]))
        assert len(keys[0]) == 4
        return original(keys)

    monkeypatch.setattr(np, "lexsort", bounded)

    pairs = head_neighbor_pairs(unit, width=2, neighbors=4, chunk_size=7)

    assert pairs.shape == (256, 2)
    assert sizes == [4] * 64


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda value: value.astype(np.float64), "float32"),
        (lambda value: np.asfortranarray(value), "C-contiguous"),
        (lambda value: value[0], "matrix"),
        (lambda value: value * np.float32(0.5), "unit"),
    ],
)
def test_fit_rejects_invalid_embedding_contract(mutation, message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        fit_tail_moment(
            mutation(_unit_fixture()), width=2, basis_kind="native", neighbors=1
        )


@pytest.mark.parametrize(
    "width, neighbors",
    [(True, 1), (0, 1), (4, 1), (2, True), (2, 0), (2, 4)],
)
def test_fit_rejects_invalid_width_or_neighbor_count(width, neighbors) -> None:
    with pytest.raises((TypeError, ValueError)):
        fit_tail_moment(
            _unit_fixture(), width=width, basis_kind="native", neighbors=neighbors
        )
