from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "measure_fragmentation_partition_stability",
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "measure_fragmentation_partition_stability.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_adjusted_rand_is_permutation_invariant_and_penalizes_crossing() -> None:
    left = np.asarray([0, 0, 1, 1])
    assert _MODULE._adjusted_rand(left, np.asarray([7, 7, 3, 3])) == pytest.approx(1.0)
    assert _MODULE._adjusted_rand(left, np.asarray([0, 1, 0, 1])) < 0.0


def test_components_symmetrizes_directed_neighbours() -> None:
    similarities = np.asarray(
        [
            [1.0, 0.9, 0.1, 0.0],
            [0.9, 1.0, 0.2, 0.1],
            [0.1, 0.2, 1.0, 0.8],
            [0.0, 0.1, 0.8, 1.0],
        ]
    )
    labels = _MODULE._components(similarities, 1)
    assert labels[0] == labels[1]
    assert labels[2] == labels[3]
    assert labels[0] != labels[2]


def test_validate_refuses_duplicate_or_reordered_examples() -> None:
    pack = {
        "embeddings": np.eye(3),
        "labels": np.asarray([0, 0, 0]),
        "example_ids": np.asarray([10, 11, 12]),
    }
    reordered = {**pack, "example_ids": np.asarray([11, 10, 12])}
    with pytest.raises(ValueError, match="example_ids differ"):
        _MODULE._validate_packs([pack, reordered, pack])
    duplicated = {**pack, "example_ids": np.asarray([10, 10, 12])}
    with pytest.raises(ValueError, match="duplicate example_ids"):
        _MODULE._validate_packs([pack, duplicated, pack])
