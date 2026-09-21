from __future__ import annotations

import numpy as np
import torch

import sfora
from sfora.joint_relational_compaction import pack_int8_unit_embeddings


def _packed(rows: int, *, seed: int) -> object:
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

    index = sfora.CpuPackedInt8Gallery.open_packed(gallery, block_rows=7)
    actual_ordinals, actual_scores = index.search_packed(queries)

    np.testing.assert_array_equal(actual_ordinals, expected_ordinals)
    np.testing.assert_array_equal(actual_scores, expected_scores)


def test_cpu_packed_gallery_resolves_equal_scores_by_lowest_ordinal() -> None:
    vector = torch.nn.functional.normalize(torch.arange(1, 129, dtype=torch.float32), dim=0)
    gallery = pack_int8_unit_embeddings(vector.repeat(12, 1).contiguous())
    query = pack_int8_unit_embeddings(vector.reshape(1, 128).contiguous())

    index = sfora.CpuPackedInt8Gallery.open_packed(gallery, block_rows=5)
    ordinals, _scores = index.search_packed(query)

    np.testing.assert_array_equal(ordinals[0], np.arange(10, dtype=np.int64))
