from __future__ import annotations

import pytest
import torch

from sfora.pq_lookup_distillation import (
    LookupTableDistillationSpec,
    QueryDependentLookupScorer,
    fit_query_dependent_lookup_scorer,
    lookup_table_distillation_loss,
)


def _spec() -> LookupTableDistillationSpec:
    return LookupTableDistillationSpec(
        query_dimensions=3,
        blocks=2,
        codebook_size=2,
        hidden_dimensions=4,
        correction_dimensions=2,
    )


def test_zero_initialized_lookup_correction_is_exact_baseline_and_permutation_independent() -> None:
    torch.manual_seed(7)
    model = QueryDependentLookupScorer(_spec())
    queries = torch.randn((2, 3), dtype=torch.float32)
    codes = torch.tensor([[[0, 1], [1, 0], [1, 1]], [[1, 1], [0, 0], [0, 1]]], dtype=torch.uint8)
    baseline = torch.randn((2, 3), dtype=torch.float32)

    observed = model(queries, codes, baseline)
    torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)

    with torch.no_grad():
        model.code_embeddings.copy_(
            torch.tensor(
                [
                    [[0.2, -0.3], [-0.1, 0.4]],
                    [[0.5, 0.1], [-0.2, -0.4]],
                ],
                dtype=torch.float32,
            )
        )
    permutation = torch.tensor([2, 0, 1])
    permuted = model(queries, codes[:, permutation], baseline[:, permutation])
    torch.testing.assert_close(permuted, model(queries, codes, baseline)[:, permutation])


def test_lookup_tables_are_mean_centered_and_match_explicit_additive_score() -> None:
    torch.manual_seed(11)
    model = QueryDependentLookupScorer(_spec())
    with torch.no_grad():
        model.code_embeddings.normal_()
    queries = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float32)
    codes = torch.tensor([[[0, 0], [0, 1], [1, 0], [1, 1]]], dtype=torch.uint8)
    baseline = torch.tensor([[0.3, 0.2, -0.1, 0.6]], dtype=torch.float32)

    tables = model.correction_tables(queries)
    torch.testing.assert_close(
        tables.mean(dim=-1), torch.zeros((1, 2), dtype=torch.float32), atol=1e-7, rtol=0.0
    )
    expected = baseline.clone()
    for candidate in range(codes.shape[1]):
        expected[0, candidate] += sum(
            tables[0, block, int(codes[0, candidate, block])] for block in range(codes.shape[2])
        )
    torch.testing.assert_close(model(queries, codes, baseline), expected)


def test_gallery_scoring_matches_repeated_candidate_codes_without_materializing_them() -> None:
    torch.manual_seed(13)
    model = QueryDependentLookupScorer(_spec())
    with torch.no_grad():
        model.code_embeddings.normal_()
    queries = torch.randn((3, 3), dtype=torch.float32)
    gallery_codes = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.uint8)
    baseline = torch.randn((3, 4), dtype=torch.float32)

    observed = model.score_gallery(queries, gallery_codes, baseline)
    repeated = gallery_codes.unsqueeze(0).expand(3, -1, -1)

    torch.testing.assert_close(observed, model(queries, repeated, baseline))


def test_lookup_distillation_loss_matches_registered_terms() -> None:
    student = torch.tensor([[0.4, 0.1, -0.2, 0.0]], requires_grad=True)
    teacher = torch.tensor([[0.5, -0.1, 0.2, -0.3]])
    pairs = torch.tensor([[[0, 1], [0, 2], [2, 3]]], dtype=torch.int64)

    observed = lookup_table_distillation_loss(
        student,
        teacher,
        pairs,
        temperature=0.2,
        listwise_weight=1.0,
        pairwise_weight=0.25,
        score_weight=0.1,
    )
    probabilities = torch.softmax(teacher / 0.2, dim=-1)
    expected_kl = torch.nn.functional.kl_div(
        torch.log_softmax(student / 0.2, dim=-1), probabilities, reduction="batchmean"
    )
    student_delta = student[:, pairs[0, :, 0]] - student[:, pairs[0, :, 1]]
    teacher_delta = teacher[:, pairs[0, :, 0]] - teacher[:, pairs[0, :, 1]]
    expected_pairwise = torch.nn.functional.binary_cross_entropy_with_logits(
        student_delta / 0.2, torch.sigmoid(teacher_delta / 0.2)
    )
    expected_mse = torch.nn.functional.mse_loss(
        student - student.mean(dim=-1, keepdim=True),
        teacher - teacher.mean(dim=-1, keepdim=True),
    )
    torch.testing.assert_close(
        observed.total, expected_kl + 0.25 * expected_pairwise + 0.1 * expected_mse
    )
    torch.testing.assert_close(observed.listwise_kl, expected_kl)
    torch.testing.assert_close(observed.pairwise_bce, expected_pairwise)
    torch.testing.assert_close(observed.score_mse, expected_mse)
    observed.total.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())


@pytest.mark.parametrize(
    ("queries", "codes", "baseline"),
    (
        (torch.zeros((1, 2)), torch.zeros((1, 2, 2), dtype=torch.uint8), torch.zeros((1, 2))),
        (torch.zeros((1, 3)), torch.zeros((1, 2, 1), dtype=torch.uint8), torch.zeros((1, 2))),
        (torch.zeros((1, 3)), torch.zeros((1, 2, 2), dtype=torch.int64), torch.zeros((1, 2))),
        (
            torch.tensor([[float("nan"), 0.0, 0.0]]),
            torch.zeros((1, 2, 2), dtype=torch.uint8),
            torch.zeros((1, 2)),
        ),
    ),
)
def test_lookup_scorer_rejects_input_authority_drift(
    queries: torch.Tensor, codes: torch.Tensor, baseline: torch.Tensor
) -> None:
    with pytest.raises(ValueError, match="lookup scorer input authority"):
        QueryDependentLookupScorer(_spec())(queries, codes, baseline)


def test_lookup_loss_can_isolate_teacher_score_mse_control() -> None:
    student = torch.tensor([[0.4, 0.1, -0.2]], requires_grad=True)
    teacher = torch.tensor([[0.5, -0.1, 0.2]])
    pairs = torch.tensor([[[0, 1]]], dtype=torch.int64)

    observed = lookup_table_distillation_loss(
        student,
        teacher,
        pairs,
        temperature=0.2,
        listwise_weight=0.0,
        pairwise_weight=0.0,
        score_weight=1.0,
    )

    torch.testing.assert_close(observed.total, observed.score_mse)


def test_lookup_scorer_fit_is_seeded_and_reduces_teacher_ordering_loss() -> None:
    generator = torch.Generator().manual_seed(19)
    queries = torch.randn((32, 3), generator=generator)
    codes = torch.randint(0, 2, (32, 6, 2), generator=generator, dtype=torch.uint8)
    baseline = torch.randn((32, 6), generator=generator)
    teacher = baseline + 0.7 * queries[:, :1] * (codes[:, :, 0].float() - 0.5)
    pairs = torch.tensor([[[0, 1], [0, 2], [1, 3], [2, 4]]] * 32, dtype=torch.int64)

    first = fit_query_dependent_lookup_scorer(
        queries,
        codes,
        baseline,
        teacher,
        pairs,
        spec=_spec(),
        seed=23,
        updates=160,
        batch_size=8,
        learning_rate=3e-3,
        temperature=0.2,
        listwise_weight=1.0,
        pairwise_weight=0.25,
        score_weight=0.1,
    )
    second = fit_query_dependent_lookup_scorer(
        queries,
        codes,
        baseline,
        teacher,
        pairs,
        spec=_spec(),
        seed=23,
        updates=160,
        batch_size=8,
        learning_rate=3e-3,
        temperature=0.2,
        listwise_weight=1.0,
        pairwise_weight=0.25,
        score_weight=0.1,
    )
    before = lookup_table_distillation_loss(
        baseline,
        teacher,
        pairs,
        temperature=0.2,
        listwise_weight=1.0,
        pairwise_weight=0.25,
        score_weight=0.1,
    ).total
    after = lookup_table_distillation_loss(
        first(queries, codes, baseline),
        teacher,
        pairs,
        temperature=0.2,
        listwise_weight=1.0,
        pairwise_weight=0.25,
        score_weight=0.1,
    ).total
    assert float(after.detach()) < 0.7 * float(before.detach())
    for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_lookup_scorer_refreshes_student_candidates_at_registered_boundaries() -> None:
    queries = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]] * 4)
    codes = torch.zeros((8, 3, 2), dtype=torch.uint8)
    baseline = torch.zeros((8, 3))
    teacher = torch.zeros((8, 3))
    pairs = torch.tensor([[[0, 1]]] * 8, dtype=torch.int64)
    calls: list[int] = []

    def refresh(
        _model: QueryDependentLookupScorer, refresh_index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calls.append(refresh_index)
        refreshed_codes = codes.clone()
        refreshed_codes[:, 0, 0] = 1
        refreshed_teacher = teacher.clone()
        refreshed_teacher[:, 0] = queries[:, 0]
        return refreshed_codes, baseline, refreshed_teacher, pairs

    fit_query_dependent_lookup_scorer(
        queries,
        codes,
        baseline,
        teacher,
        pairs,
        spec=_spec(),
        seed=29,
        updates=6,
        batch_size=2,
        learning_rate=3e-3,
        temperature=0.2,
        listwise_weight=1.0,
        pairwise_weight=0.25,
        score_weight=0.1,
        refresh_interval=2,
        refresh=refresh,
    )

    assert calls == [1, 2]


def test_lookup_scorer_consumes_each_epoch_permutation_without_dropping_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sfora.pq_lookup_distillation as subject

    queries = torch.randn((10, 3), generator=torch.Generator().manual_seed(31))
    codes = torch.zeros((10, 3, 2), dtype=torch.uint8)
    baseline = torch.zeros((10, 3))
    teacher = torch.zeros((10, 3))
    pairs = torch.tensor([[[0, 1]]] * 10, dtype=torch.int64)
    observed_batch_sizes: list[int] = []
    original_loss = subject.lookup_table_distillation_loss

    def recording_loss(*args: object, **kwargs: object) -> subject.LookupTableDistillationLoss:
        scores = args[0]
        assert isinstance(scores, torch.Tensor)
        if len(scores) < len(queries):
            observed_batch_sizes.append(len(scores))
        return original_loss(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(subject, "lookup_table_distillation_loss", recording_loss)
    fit_query_dependent_lookup_scorer(
        queries,
        codes,
        baseline,
        teacher,
        pairs,
        spec=_spec(),
        seed=37,
        updates=3,
        batch_size=4,
        learning_rate=3e-3,
        temperature=0.2,
        listwise_weight=1.0,
        pairwise_weight=0.25,
        score_weight=0.1,
    )

    assert observed_batch_sizes == [4, 4, 2]
