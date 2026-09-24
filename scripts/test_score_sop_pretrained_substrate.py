"""Fit-only projection and row alignment for the substrate screen."""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from score_sop_pretrained_substrate import fit_project_pack, require_aligned_rows


def test_candidate_row_order_must_match_reference():
    ids = np.array([1, 2, 3], dtype=np.int64)
    labels = np.array([7, 7, 9], dtype=np.int64)
    paths = np.array(["one.jpg", "two.jpg", "three.jpg"])
    require_aligned_rows(ids, labels, paths, ids.copy(), labels.copy(), paths.copy())
    with pytest.raises(ValueError, match="row alignment"):
        require_aligned_rows(ids, labels, paths, ids[[1, 0, 2]], labels, paths)


def test_pca_fit_is_unchanged_when_only_holdout_descriptors_change(tmp_path):
    generator = np.random.default_rng(17)
    features = generator.normal(size=(9, 5)).astype(np.float32)
    fit_rows = np.array([0, 1, 2, 3, 4, 5], dtype=np.int64)
    first = fit_project_pack(features, fit_rows, dimensions=3)
    changed = features.copy()
    changed[6:] = generator.normal(size=(3, 5)).astype(np.float32) * 3
    second = fit_project_pack(changed, fit_rows, dimensions=3)
    torch.testing.assert_close(first.pca.mean, second.pca.mean, rtol=0, atol=0)
    torch.testing.assert_close(first.pca.components, second.pca.components, rtol=0, atol=0)
    assert not torch.equal(first.projected[6:], second.projected[6:])
    assert first.packed.codes.shape == (9, 3)
    file = tmp_path / "features.npy"
    np.save(file, features)
    mapped = fit_project_pack(np.load(file, mmap_mode="r"), fit_rows, dimensions=3)
    torch.testing.assert_close(first.packed.codes, mapped.packed.codes)
