"""Focused checks for the fit-only deployed-code rank screen."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings


def _module():
    path = Path(__file__).with_name("_scratch_deployed_code_rank_finish_f0.py")
    spec = importlib.util.spec_from_file_location("deployed_code_f0", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fake_quant_forward_matches_serving_code_and_backpropagates() -> None:
    module = _module()
    values = torch.randn(8, 128, generator=torch.Generator().manual_seed(7))
    values.requires_grad_(True)
    coded = module.fake_quantized_unit(values)
    wire = pack_int8_unit_embeddings(F.normalize(values.detach(), dim=1))
    expected = F.normalize(wire.codes.float(), dim=1)
    torch.testing.assert_close(coded.detach(), expected, rtol=0, atol=1e-7)
    coded[:, 0].sum().backward()
    assert values.grad is not None
    assert torch.isfinite(values.grad).all()
    assert torch.count_nonzero(values.grad) > 0


def test_packed_query_gallery_metrics_respects_map_at_r_and_ordinal_ties() -> None:
    module = _module()
    query = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    gallery = F.normalize(
        torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]]),
        dim=1,
    )
    result = module.packed_query_gallery_metrics(
        query, gallery, np.asarray(["a", "b"]), np.asarray(["a", "a", "b", "b"]),
        device=torch.device("cpu"),
    )
    assert result["map_at_r"] == 0.75
    assert result["recall_at_1"] == 1.0
    tied = module.packed_query_gallery_metrics(
        torch.tensor([[1.0, 0.0]]),
        F.normalize(torch.tensor([[1.0, 0.0], [1.0, 0.0]]), dim=1),
        np.asarray(["a"]), np.asarray(["b", "a"]), device=torch.device("cpu"),
    )
    assert tied["recall_at_1"] == 0.0


def test_result_is_published_once_and_validated(tmp_path: Path) -> None:
    module = _module()
    path = tmp_path / "result.json"
    payload = b'{"result":"ok"}\n'
    module.publish_result(path, payload)
    assert path.read_bytes() == payload
    with pytest.raises(FileExistsError):
        module.publish_result(path, payload)
