from __future__ import annotations

import numpy as np
import pytest

from sfora.gallery_hubness import (
    corrected_cosine_top1,
    gallery_local_density,
    top1_hubness,
)


def _unit(rows: list[list[float]]) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    return values / np.linalg.norm(values, axis=1, keepdims=True)


GALLERY = _unit([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])


def test_gallery_density_excludes_self_and_is_block_size_invariant() -> None:
    expected = np.asarray([0.8, 0.8, 0.6, 0.0], dtype=np.float32)
    assert gallery_local_density(GALLERY, k=1, block_size=1) == pytest.approx(
        expected
    )
    assert gallery_local_density(GALLERY, k=1, block_size=3) == pytest.approx(
        expected
    )


def test_all_gallery_density_uses_every_nonself_item() -> None:
    expected = np.asarray([-1.0 / 15.0, 0.2, 0.2, -0.6], dtype=np.float32)
    assert gallery_local_density(GALLERY, k="all", block_size=2) == pytest.approx(
        expected, abs=1e-7
    )


def test_corrected_top1_uses_two_cosine_minus_density_and_index_ties() -> None:
    queries = _unit([[1.0, 0.0], [0.0, 1.0]])
    density = np.asarray([1.0, 0.0, 0.4, 0.0], dtype=np.float32)
    indices, scores = corrected_cosine_top1(
        queries, GALLERY, density, block_size=1
    )
    assert indices.tolist() == [1, 2]
    assert scores.tolist() == pytest.approx([1.6, 1.6])

    tied_indices, _ = corrected_cosine_top1(
        _unit([[1.0, 1.0]]),
        _unit([[1.0, 0.0], [0.0, 1.0]]),
        np.zeros(2, dtype=np.float32),
        block_size=1,
    )
    assert tied_indices.tolist() == [0]


def test_top1_hubness_returns_exact_counts_and_population_skewness() -> None:
    queries = _unit([[1.0, 0.0], [0.99, 0.1], [0.0, 1.0]])
    summary = top1_hubness(queries, GALLERY, block_size=2)
    assert summary.counts.tolist() == [2, 0, 1, 0]
    assert summary.maximum_count == 2
    assert summary.skewness == pytest.approx(0.49338220021815865)


@pytest.mark.parametrize(
    ("gallery", "k", "message"),
    [
        (np.asarray([[1, 0]], dtype=np.int64), 1, "floating"),
        (np.asarray([[np.nan, 0]], dtype=np.float32), 1, "finite"),
        (_unit([[1, 0]]), 1, "available"),
        (GALLERY, 0, "positive integer"),
        (GALLERY, True, "positive integer"),
        (GALLERY, "bad", "integer or 'all'"),
    ],
)
def test_gallery_density_rejects_invalid_inputs(
    gallery: np.ndarray, k: int | str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        gallery_local_density(gallery, k=k)


def test_corrected_top1_rejects_dimension_and_density_mismatches() -> None:
    with pytest.raises(ValueError, match="dimensions"):
        corrected_cosine_top1(
            _unit([[1, 0, 0]]), GALLERY, np.zeros(4, dtype=np.float32)
        )
    with pytest.raises(ValueError, match="density"):
        corrected_cosine_top1(
            _unit([[1, 0]]), GALLERY, np.zeros(3, dtype=np.float32)
        )
