"""The feature cache and training tower must share one pinned initialization."""

import hashlib

import pytest
import torch
from export_inshop_siglip2_train_features import load_vision_init
from train_inshop_siglip2_unseen_gallery import validate_cache_vision_init


def test_vision_init_rejects_wrong_digest_and_loads_strictly(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"vision": {"weight": torch.ones(2, 2)}}, path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    vision = torch.nn.Linear(2, 2, bias=False)
    with pytest.raises(ValueError, match="checkpoint SHA"):
        load_vision_init(vision, path, "0" * 64)
    load_vision_init(vision, path, digest)
    assert torch.equal(vision.weight, torch.ones(2, 2))
    with pytest.raises(RuntimeError, match="Missing key"):
        load_vision_init(torch.nn.Linear(2, 2), path, digest)


def test_cache_rejects_mismatched_vision_init():
    digest = "a" * 64
    validate_cache_vision_init({"vision_init_sha256": digest}, digest)
    validate_cache_vision_init({}, None)
    with pytest.raises(ValueError, match="cache vision initialization"):
        validate_cache_vision_init({}, digest)
    with pytest.raises(ValueError, match="cache vision initialization"):
        validate_cache_vision_init({"vision_init_sha256": digest}, None)
