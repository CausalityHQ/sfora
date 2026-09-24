import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_l14_exact_head_sop_train import gallery_rows_without_queries, scalar_packed_topk


def test_gallery_rows_exclude_only_selected_queries() -> None:
    rows = gallery_rows_without_queries(7, (1, 4))
    assert rows == (0, 2, 3, 5, 6)
    assert len(rows) == 5


def test_scalar_packed_topk_resolves_duplicate_scores_by_gallery_ordinal() -> None:
    query = np.full(128, 1, dtype=np.int8)
    gallery = np.zeros((12, 128), dtype=np.int8)
    gallery[0] = 1
    gallery[1] = 1
    inverse = np.full(12, 1 / np.sqrt(128), dtype=np.float16)
    ranked = scalar_packed_topk(query, np.float16(1 / np.sqrt(128)), gallery, inverse)
    assert ranked.tolist()[:2] == [0, 1]
