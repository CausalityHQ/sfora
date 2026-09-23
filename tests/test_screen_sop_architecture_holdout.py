"""Paired pretrained architecture screens require identical train rows."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "screen_sop_architecture_holdout.py"
SPEC = importlib.util.spec_from_file_location("screen_sop_architecture_holdout", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)
sys.path.pop(0)


def _archive(ids=(1, 2, 3, 4)):
    return {
        "train_image_ids": np.asarray(ids, dtype=np.int64),
        "train_labels": np.asarray([1, 1, 2, 2], dtype=np.int64),
        "train_relative_paths": np.asarray(["a", "b", "c", "d"]),
        "train_embeddings": np.ones((4, 768), dtype=np.float32),
    }


def test_paired_architecture_screen_rejects_reordered_images():
    MODULE.assert_train_alignment(_archive(), _archive())
    with pytest.raises(ValueError, match="paired SOP train rows"):
        MODULE.assert_train_alignment(_archive(), _archive((2, 1, 3, 4)))
