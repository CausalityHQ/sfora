from __future__ import annotations

import numpy as np
import pytest
import torch

import sfora
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.packed_int8_search import CpuPackedInt8Gallery, _ordered_topk_indexes


def _packed(rows: int, *, seed: int) -> PackedInt8Embeddings:
    generator = torch.Generator().manual_seed(seed)
    values = torch.nn.functional.normalize(
        torch.randn(rows, 128, generator=generator), dim=1
    ).contiguous()
    return pack_int8_unit_embeddings(values)


def test_cpu_packed_gallery_matches_full_exact_scores_across_blocks() -> None:
    gallery = _packed(37, seed=211)
    queries = _packed(3, seed=223)
    reference = queries.cosine_similarity(gallery).numpy()
    ordinals = np.arange(reference.shape[1], dtype=np.int64)
    expected_ordinals = np.stack([np.lexsort((ordinals, -row))[:10] for row in reference])
    expected_scores = np.take_along_axis(reference, expected_ordinals, axis=1)

    assert sfora.CpuPackedInt8Gallery is CpuPackedInt8Gallery
    index = CpuPackedInt8Gallery.open_packed(gallery, block_rows=7)
    actual_ordinals, actual_scores = index.search_packed(queries)

    np.testing.assert_array_equal(actual_ordinals, expected_ordinals)
    np.testing.assert_array_equal(actual_scores, expected_scores)


def test_cpu_packed_gallery_resolves_equal_scores_by_lowest_ordinal() -> None:
    vector = torch.nn.functional.normalize(torch.arange(1, 129, dtype=torch.float32), dim=0)
    gallery = pack_int8_unit_embeddings(vector.repeat(12, 1).contiguous())
    query = pack_int8_unit_embeddings(vector.reshape(1, 128).contiguous())

    index = CpuPackedInt8Gallery.open_packed(gallery, block_rows=5)
    ordinals, _scores = index.search_packed(query)

    np.testing.assert_array_equal(ordinals[0], np.arange(10, dtype=np.int64))


def test_cpu_packed_gallery_matches_full_order_for_variable_k_and_block_edges() -> None:
    gallery = _packed(1003, seed=227)
    queries = _packed(4, seed=229)
    reference = queries.cosine_similarity(gallery).numpy()
    ordinals = np.arange(reference.shape[1], dtype=np.int64)
    index = CpuPackedInt8Gallery.open_packed(gallery, block_rows=127)

    for k in (1, 7, 10, 100, 1003):
        expected = np.stack([np.lexsort((ordinals, -row))[:k] for row in reference])
        actual, scores = index.search_packed(queries, k=k)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(scores, np.take_along_axis(reference, expected, axis=1))


@pytest.mark.parametrize("corrupt_norm", [float("nan"), float("inf"), 0.0])
def test_cpu_packed_gallery_rejects_mutated_invalid_query_norm(corrupt_norm: float) -> None:
    gallery = _packed(50, seed=233)
    queries = _packed(2, seed=239)
    queries.inverse_norms[0] = corrupt_norm
    index = CpuPackedInt8Gallery.open_packed(gallery, block_rows=16)

    with pytest.raises(ValueError, match="CPU packed query authority differs"):
        index.search_packed(queries)


def test_cpu_topk_cutoff_tie_uses_ordinal_with_shuffled_candidates() -> None:
    scores = np.asarray([0.5, 0.5, 0.5, 0.4], dtype=np.float32)
    ordinals = np.asarray([9, 1, 5, 0], dtype=np.int64)

    np.testing.assert_array_equal(
        _ordered_topk_indexes(scores, ordinals, 2), np.asarray([1, 2], dtype=np.intp)
    )


def test_cpu_packed_gallery_preserves_order_across_query_tiles(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gallery = _packed(257, seed=241)
    queries = _packed(67, seed=251)
    reference = queries.cosine_similarity(gallery).numpy()
    ordinals = np.arange(gallery.codes.shape[0], dtype=np.int64)
    expected = np.stack([np.lexsort((ordinals, -row))[:10] for row in reference])
    index = CpuPackedInt8Gallery.open_packed(gallery, block_rows=33)
    matmul = torch.Tensor.__matmul__
    observed_query_rows: list[int] = []

    def tracked_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        observed_query_rows.append(left.shape[0])
        return matmul(left, right)

    monkeypatch.setattr(torch.Tensor, "__matmul__", tracked_matmul)

    actual, scores = index.search_packed(queries)

    assert observed_query_rows and max(observed_query_rows) <= 64
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(scores, np.take_along_axis(reference, expected, axis=1))
