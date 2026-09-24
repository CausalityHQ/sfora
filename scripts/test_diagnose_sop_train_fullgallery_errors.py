import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_sop_train_fullgallery_errors import nearest_full_and_holdout


def test_full_gallery_and_holdout_mask_self_and_choose_first_tied_row() -> None:
    vectors = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.8, 0.6]])
    queries = np.asarray([2, 3], dtype=np.int64)
    fit = np.asarray([0, 1], dtype=np.int64)
    full_idx, full_score, hold_idx, hold_score = nearest_full_and_holdout(
        vectors, queries, fit, block_rows=1
    )
    assert full_idx.tolist() == [0, 0]
    assert hold_idx.tolist() == [3, 2]
    assert np.allclose(full_score, [1.0, 0.8])
    assert np.allclose(hold_score, [0.8, 0.8])
