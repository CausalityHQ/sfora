"""Small exact checks for the cached teacher-tail training objective."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from screen_sop_cached_teacher_tail import arcface_term, relational_term  # noqa: E402

from sfora.deployed_code_rank import packed_cosine_ste  # noqa: E402


def test_relational_loss_matches_off_diagonal_packed_score_kl() -> None:
    torch.manual_seed(13)
    student = torch.randn(6, 128, requires_grad=True)
    teacher = torch.randn(6, 768)
    actual = relational_term(student, teacher)
    rows = torch.arange(6)
    other = rows[:, None] != rows[None, :]
    student_logits = packed_cosine_ste(student)[other].reshape(6, 5) / 0.1
    teacher_unit = F.normalize(teacher, dim=1)
    teacher_logits = (teacher_unit @ teacher_unit.T)[other].reshape(6, 5) / 0.1
    expected = F.kl_div(
        F.log_softmax(student_logits, dim=1),
        F.softmax(teacher_logits, dim=1),
        reduction="batchmean",
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    assert student.grad is not None
    assert bool(torch.isfinite(student.grad).all())


def test_arcface_term_backpropagates_to_embedding_and_proxy() -> None:
    torch.manual_seed(29)
    feature = torch.randn(8, 128, requires_grad=True)
    proxy = torch.randn(4, 128, requires_grad=True)
    target = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    loss = arcface_term(feature, proxy, target)
    loss.backward()
    assert bool(torch.isfinite(loss))
    assert feature.grad is not None and bool(torch.isfinite(feature.grad).all())
    assert proxy.grad is not None and bool(torch.isfinite(proxy.grad).all())
