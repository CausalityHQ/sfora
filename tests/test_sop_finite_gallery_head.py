"""Oracle checks for the head screen's deployed score and mixed gallery."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from screen_sop_finite_gallery_head import (  # noqa: E402
    packed_scores,
    packed_unit_ste,
    score_heldout_against_all,
)

from sfora.joint_relational_compaction import pack_int8_unit_embeddings  # noqa: E402
from sfora.sop_evaluation import score_symmetric  # noqa: E402


def test_straight_through_forward_equals_deployed_packed_scores() -> None:
    torch.manual_seed(17)
    values = torch.randn(9, 128, requires_grad=True)
    fake = packed_unit_ste(values)
    actual = packed_scores(fake, fake)
    packed = pack_int8_unit_embeddings(F.normalize(values.detach(), dim=1).contiguous())
    code = packed.codes.float()
    inverse = packed.inverse_norms.float()
    expected = (code @ code.T) * inverse[:, None] * inverse[None, :]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.sum().backward()
    assert values.grad is not None
    assert bool(torch.isfinite(values.grad).all())


def test_mixed_gallery_scores_match_full_symmetric_oracle() -> None:
    torch.manual_seed(23)
    value = F.normalize(torch.randn(8, 128), dim=1).contiguous()
    packed = pack_int8_unit_embeddings(value)
    codes = packed.codes.float()
    inverse = packed.inverse_norms
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64)
    oracle = score_symmetric(codes, labels, inverse_norms=inverse)
    result = score_heldout_against_all(
        codes,
        inverse,
        labels,
        torch.tensor([0, 1, 2, 3]),
        torch.tensor([4, 5, 6, 7]),
        device=torch.device("cpu"),
        sample_size=1,
    )
    assert result["per_query_r1"] == oracle["per_query_r1"][:4]
    torch.testing.assert_close(
        torch.tensor(result["per_query_ap"]), torch.tensor(oracle["per_query_ap"][:4])
    )
