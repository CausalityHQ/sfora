import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_l14_inshop_train_heads import clustered_recall_interval


def test_clustered_recall_interval_preserves_point_difference() -> None:
    labels = (3, 3, 4, 4)
    earlier = (1.0, 0.0, 1.0, 1.0)
    later = (1.0, 1.0, 0.0, 1.0)
    result = clustered_recall_interval(labels, earlier, later, seed=11, resamples=100)
    assert result["difference"] == 0.0
    assert result["identity_clusters"] == 2
    assert result["interval_95"][0] <= 0.0 <= result["interval_95"][1]


def test_clustered_recall_interval_detects_one_sided_loss() -> None:
    result = clustered_recall_interval(
        (3, 3, 4, 4), (1.0, 1.0, 1.0, 1.0), (1.0, 0.0, 1.0, 1.0),
        seed=11, resamples=100,
    )
    assert result["difference"] == -0.25
    assert np.isfinite(result["interval_95"]).all()
    assert result["interval_95"][1] <= 0.0
