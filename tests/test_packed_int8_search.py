from __future__ import annotations

import numpy as np
import torch

import sfora
from sfora.joint_relational_compaction import PackedInt8Embeddings, pack_int8_unit_embeddings
from sfora.packed_int8_search import CpuPackedInt8Gallery


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
