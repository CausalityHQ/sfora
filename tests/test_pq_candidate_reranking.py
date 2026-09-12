from __future__ import annotations

import pytest
import torch

from sfora.pq_candidate_reranking import (
    CandidateSetReranker,
    ProductCodeResidualDecoder,
    ProductCodeResidualStatistics,
    aligned_product_squared_distances,
    candidate_set_distillation_loss,
    candidate_set_supervised_loss,
    exact_product_shortlists,
    fit_candidate_set_reranker,
    fit_product_code_residual_decoder,
    refine_supervised_candidate_set_reranker,
)
from sfora.product_quantization import ProductQuantizationSpec, ProductQuantizer


def _codec() -> tuple[ProductQuantizer, torch.Tensor, torch.Tensor]:
    spec = ProductQuantizationSpec(block_dimensions=(2, 1), codebook_size=2)
    codec = ProductQuantizer.from_codebooks(
        spec,
        (
            torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
            torch.tensor([[0.0], [1.0]], dtype=torch.float32),
        ),
    )
    values = torch.tensor(
        [[1.1, 0.2, 0.1], [0.8, -0.1, -0.2], [0.1, 1.2, 1.1], [-0.2, 0.9, 0.8]],
        dtype=torch.float32,
    )
    codes = codec.hard_encode(values)
    return codec, values, codes


def test_residual_statistics_and_candidate_features_match_explicit_math() -> None:
    codec, values, codes = _codec()
    statistics = ProductCodeResidualStatistics.fit(codec, values, codes)
    queries = torch.tensor([[0.6, 0.8, 0.2]], dtype=torch.float32)
    candidate_codes = codes.reshape(1, 4, 2)

    features = statistics.candidate_features(queries, candidate_codes)
    decoded = codec.hard_decode(codes)
    residuals = values - decoded

    expected_partial = torch.stack(
        (
            queries[:, :2] @ codec.codebooks[0].T,
            queries[:, 2:] @ codec.codebooks[1].T,
        ),
        dim=-1,
    )
    expected_partial = torch.stack(
        (
            expected_partial[:, :, 0][:, codes[:, 0].long()],
            expected_partial[:, :, 1][:, codes[:, 1].long()],
        ),
        dim=-1,
    )
    torch.testing.assert_close(features.partial_dot_scores, expected_partial)

    first_residuals = residuals[codes[:, 0] == 0, :2]
    expected_covariance = first_residuals.T @ first_residuals / len(first_residuals)
    expected_variance = queries[:, :2] @ expected_covariance @ queries[:, :2].T
    torch.testing.assert_close(features.directional_variances[0, 0, 0], expected_variance[0, 0])
    torch.testing.assert_close(
        features.pairwise_code_similarity,
        decoded.reshape(1, 4, 3) @ decoded.reshape(1, 4, 3).transpose(1, 2),
    )
    assert features.code_residual_energy.shape == (1, 4, 2)
    assert features.decoded_norm.shape == (1, 4, 1)

    restored = ProductCodeResidualStatistics(
        codec,
        tuple(
            statistics.state_dict()[f"covariance_{index}"]
            for index in range(codec.spec.bytes_per_vector)
        ),
        tuple(
            statistics.state_dict()[f"residual_energy_{index}"]
            for index in range(codec.spec.bytes_per_vector)
        ),
    )
    restored.load_state_dict(statistics.state_dict())
    restored_features = restored.candidate_features(queries, candidate_codes)
    torch.testing.assert_close(
        restored_features.directional_variances, features.directional_variances
    )


def test_residual_decoder_starts_at_exact_decode_and_preserves_selected_cells() -> None:
    codec, values, codes = _codec()
    decoder = ProductCodeResidualDecoder(codec, hidden_dimensions=8)

    corrected, step = decoder.decode(codes, codec)

    torch.testing.assert_close(corrected, codec.hard_decode(codes), rtol=0.0, atol=0.0)
    torch.testing.assert_close(step, torch.ones((len(codes), 2)), rtol=0.0, atol=0.0)
    assert torch.equal(codec.hard_encode(corrected), codes)


def test_residual_decoder_clips_corrections_at_product_cell_boundaries() -> None:
    codec, _values, codes = _codec()
    decoder = ProductCodeResidualDecoder(codec, hidden_dimensions=4)
    with torch.no_grad():
        decoder.network[-1].bias.copy_(torch.tensor([-2.0, 2.0, 2.0]))

    corrected, step = decoder.decode(codes[:1], codec)

    assert step.tolist() == [[0.25, 0.25]]
    torch.testing.assert_close(
        corrected,
        torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32),
        rtol=0.0,
        atol=1e-7,
    )


def test_residual_decoder_returns_normalized_candidate_pair_geometry() -> None:
    codec, _values, codes = _codec()
    decoder = ProductCodeResidualDecoder(codec, hidden_dimensions=8)

    pairwise, step = decoder.candidate_pairwise(codes.reshape(1, 4, 2), codec)

    torch.testing.assert_close(
        pairwise,
        torch.tensor(
            [
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [1.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 1.0, 1.0],
                ]
            ]
        ),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(step, torch.ones((1, 4, 2)), rtol=0.0, atol=0.0)


def test_residual_decoder_fit_is_seeded_and_reduces_pointwise_error() -> None:
    codec, values, codes = _codec()
    baseline = torch.nn.functional.mse_loss(codec.hard_decode(codes), values)

    first = fit_product_code_residual_decoder(
        codec,
        values,
        codes,
        hidden_dimensions=8,
        seed=11,
        updates=80,
        batch_size=4,
        learning_rate=0.01,
    )
    second = fit_product_code_residual_decoder(
        codec,
        values,
        codes,
        hidden_dimensions=8,
        seed=11,
        updates=80,
        batch_size=4,
        learning_rate=0.01,
    )
    corrected, _step = first.decode(codes, codec)

    assert torch.nn.functional.mse_loss(corrected, values) < baseline
    for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value[:, :, :1],
        lambda value: value.to(torch.int16),
        lambda value: value.reshape(-1, value.shape[-1]),
    ),
)
def test_residual_decoder_rejects_candidate_code_authority_drift(mutation: object) -> None:
    codec, _values, codes = _codec()
    decoder = ProductCodeResidualDecoder(codec, hidden_dimensions=8)

    with pytest.raises(ValueError, match="product-code residual decoder candidates"):
        decoder.candidate_pairwise(mutation(codes.reshape(1, 4, 2)), codec)  # type: ignore[operator]


def test_candidate_reranker_starts_at_baseline_and_is_permutation_equivariant() -> None:
    torch.manual_seed(3)
    model = CandidateSetReranker(blocks=2, model_dimensions=16, heads=4, layers=2)
    baseline = torch.randn((2, 5))
    feature_values = torch.randn((2, 5, 7))
    pairwise = torch.randn((2, 5, 5))
    pairwise = 0.5 * (pairwise + pairwise.transpose(1, 2))

    observed = model(baseline, feature_values, pairwise)
    torch.testing.assert_close(observed, baseline, rtol=0.0, atol=0.0)

    with torch.no_grad():
        model.residual_scale.fill_(0.5)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    permuted = model(
        baseline[:, permutation],
        feature_values[:, permutation],
        pairwise[:, permutation][:, :, permutation],
    )
    torch.testing.assert_close(permuted, model(baseline, feature_values, pairwise)[:, permutation])

    model(baseline, feature_values, pairwise).sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_candidate_set_distillation_loss_matches_kl_and_mse_and_backpropagates() -> None:
    student = torch.tensor([[0.3, 0.1, -0.2], [0.2, -0.1, 0.4]], requires_grad=True)
    teacher = torch.tensor([[0.5, 0.0, -0.1], [0.3, -0.2, 0.1]])

    observed = candidate_set_distillation_loss(student, teacher, temperature=0.05, score_weight=0.1)
    probabilities = torch.softmax(teacher / 0.05, dim=-1)
    expected_kl = torch.nn.functional.kl_div(
        torch.log_softmax(student / 0.05, dim=-1), probabilities, reduction="batchmean"
    )
    expected_mse = torch.nn.functional.mse_loss(
        student - student.mean(dim=-1, keepdim=True),
        teacher - teacher.mean(dim=-1, keepdim=True),
    )
    torch.testing.assert_close(observed.total, expected_kl + 0.1 * expected_mse)
    torch.testing.assert_close(observed.listwise_kl, expected_kl)
    torch.testing.assert_close(observed.score_mse, expected_mse)
    observed.total.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())


def test_candidate_set_supervised_loss_matches_uniform_positive_cross_entropy() -> None:
    student = torch.tensor([[0.3, 0.1, -0.2], [0.2, -0.1, 0.4]], requires_grad=True)
    teacher = torch.tensor([[0.5, 0.0, -0.1], [0.3, -0.2, 0.1]])
    positive = torch.tensor([[True, True, False], [False, True, False]])

    observed = candidate_set_supervised_loss(
        student,
        teacher,
        positive,
        temperature=0.05,
        score_weight=0.1,
        supervised_weight=0.2,
    )
    target = positive.float() / positive.sum(dim=1, keepdim=True)
    expected_positive = -(target * torch.log_softmax(student / 0.05, dim=1)).sum(dim=1).mean()
    expected_distillation = candidate_set_distillation_loss(
        student, teacher, temperature=0.05, score_weight=0.1
    ).total

    torch.testing.assert_close(observed.positive_listwise, expected_positive)
    torch.testing.assert_close(observed.total, expected_distillation + 0.2 * expected_positive)
    assert observed.positive_coverage == 1.0
    observed.total.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and bool(torch.isfinite(student.grad).all())


def test_candidate_set_supervised_loss_uses_distillation_for_all_negative_batch() -> None:
    student = torch.tensor([[0.3, 0.1, -0.2]], requires_grad=True)
    teacher = torch.tensor([[0.5, 0.0, -0.1]])
    positive = torch.zeros_like(student, dtype=torch.bool)

    observed = candidate_set_supervised_loss(
        student,
        teacher,
        positive,
        temperature=0.05,
        score_weight=0.1,
        supervised_weight=0.2,
    )
    expected = candidate_set_distillation_loss(
        student, teacher, temperature=0.05, score_weight=0.1
    ).total

    torch.testing.assert_close(observed.total, expected)
    assert observed.positive_listwise.item() == 0.0
    assert observed.positive_coverage == 0.0


@pytest.mark.parametrize(
    "mutation",
    (
        lambda value: value[:, :, :1],
        lambda value: value.double(),
        lambda value: value.masked_fill(torch.ones_like(value, dtype=torch.bool), float("nan")),
    ),
)
def test_candidate_reranker_rejects_feature_authority_drift(mutation: object) -> None:
    model = CandidateSetReranker(blocks=2, model_dimensions=16, heads=4, layers=1)
    baseline = torch.zeros((1, 3))
    features = torch.zeros((1, 3, 7))
    pairwise = torch.eye(3).reshape(1, 3, 3)

    with pytest.raises(ValueError, match="candidate reranker input authority"):
        model(baseline, mutation(features), pairwise)  # type: ignore[operator]


def test_candidate_reranker_rejects_asymmetric_pairwise_authority() -> None:
    model = CandidateSetReranker(blocks=2, model_dimensions=16, heads=4, layers=1)
    pairwise = torch.eye(3).reshape(1, 3, 3)
    pairwise[0, 0, 1] = 1.0

    with pytest.raises(ValueError, match="candidate reranker input authority"):
        model(torch.zeros((1, 3)), torch.zeros((1, 3, 7)), pairwise)


def test_exact_product_shortlists_exclude_self_and_use_lowest_row_ties() -> None:
    codec, values, codes = _codec()

    observed = exact_product_shortlists(values, codes, codec, width=2, query_block_size=2)
    distances = codec.asymmetric_squared_distances(values, codes)
    distances.fill_diagonal_(torch.inf)
    expected = torch.argsort(distances, dim=1, stable=True)[:, :2]

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)
    assert all(row not in selected for row, selected in enumerate(observed.tolist()))


def test_aligned_product_distances_are_bit_identical_to_deployment_adc() -> None:
    codec, values, codes = _codec()
    candidate_codes = torch.stack((codes.roll(1, 0), codes.roll(2, 0)), dim=1)

    observed = aligned_product_squared_distances(values, candidate_codes, codec)
    expected = torch.stack(
        [
            codec.asymmetric_squared_distances(values[row : row + 1], candidate_codes[row])[0]
            for row in range(len(values))
        ]
    )

    torch.testing.assert_close(observed, expected, rtol=0.0, atol=0.0)


def test_exact_product_shortlists_repair_ties_at_retention_boundary() -> None:
    codec, values, _codes = _codec()
    tied = values[:1].repeat(5, 1)
    codes = codec.hard_encode(tied)

    observed = exact_product_shortlists(tied, codes, codec, width=2, query_block_size=3)

    assert observed.tolist() == [[1, 2], [0, 2], [0, 1], [0, 1], [0, 1]]


@pytest.mark.parametrize("width", (0, 4, True))
def test_exact_product_shortlists_reject_invalid_width(width: object) -> None:
    codec, values, codes = _codec()
    with pytest.raises(ValueError, match="product shortlist authority"):
        exact_product_shortlists(values, codes, codec, width=width, query_block_size=2)  # type: ignore[arg-type]


def test_candidate_reranker_fit_is_seeded_and_reduces_teacher_list_loss() -> None:
    generator = torch.Generator().manual_seed(13)
    baseline = torch.randn((24, 4), generator=generator)
    features = torch.randn((24, 4, 7), generator=generator)
    raw_pairwise = torch.randn((24, 4, 4), generator=generator)
    pairwise = 0.5 * (raw_pairwise + raw_pairwise.transpose(1, 2))
    teacher = baseline + 0.4 * features[:, :, 0] - 0.2 * features[:, :, 3]

    first = fit_candidate_set_reranker(
        baseline,
        features,
        pairwise,
        teacher,
        blocks=2,
        model_dimensions=16,
        heads=4,
        layers=1,
        seed=7,
        updates=120,
        batch_size=8,
        learning_rate=3e-3,
        temperature=0.2,
        score_weight=0.1,
    )
    second = fit_candidate_set_reranker(
        baseline,
        features,
        pairwise,
        teacher,
        blocks=2,
        model_dimensions=16,
        heads=4,
        layers=1,
        seed=7,
        updates=120,
        batch_size=8,
        learning_rate=3e-3,
        temperature=0.2,
        score_weight=0.1,
    )

    before = candidate_set_distillation_loss(
        baseline, teacher, temperature=0.2, score_weight=0.1
    ).total
    after = candidate_set_distillation_loss(
        first(baseline, features, pairwise),
        teacher,
        temperature=0.2,
        score_weight=0.1,
    ).total
    assert float(after.detach()) < 0.35 * float(before.detach())
    for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_supervised_candidate_reranker_refinement_is_seeded_and_does_not_mutate_source() -> None:
    generator = torch.Generator().manual_seed(29)
    baseline = torch.randn((24, 4), generator=generator)
    features = torch.randn((24, 4, 7), generator=generator)
    raw_pairwise = torch.randn((24, 4, 4), generator=generator)
    pairwise = 0.5 * (raw_pairwise + raw_pairwise.transpose(1, 2))
    teacher = baseline + 0.2 * features[:, :, 1]
    positive = features[:, :, 0] > 0
    positive[~positive.any(dim=1), 0] = True
    source = CandidateSetReranker(blocks=2, model_dimensions=16, heads=4, layers=1)
    source_state = {name: value.detach().clone() for name, value in source.state_dict().items()}

    first = refine_supervised_candidate_set_reranker(
        source,
        baseline,
        features,
        pairwise,
        teacher,
        positive,
        seed=31,
        updates=120,
        batch_size=8,
        learning_rate=3e-3,
        temperature=0.2,
        score_weight=0.1,
        supervised_weight=1.0,
    )
    second = refine_supervised_candidate_set_reranker(
        source,
        baseline,
        features,
        pairwise,
        teacher,
        positive,
        seed=31,
        updates=120,
        batch_size=8,
        learning_rate=3e-3,
        temperature=0.2,
        score_weight=0.1,
        supervised_weight=1.0,
    )
    before = candidate_set_supervised_loss(
        source(baseline, features, pairwise),
        teacher,
        positive,
        temperature=0.2,
        score_weight=0.1,
        supervised_weight=1.0,
    ).total
    after = candidate_set_supervised_loss(
        first(baseline, features, pairwise),
        teacher,
        positive,
        temperature=0.2,
        score_weight=0.1,
        supervised_weight=1.0,
    ).total

    assert float(after.detach()) < float(before.detach())
    for name, value in source.state_dict().items():
        torch.testing.assert_close(value, source_state[name], rtol=0.0, atol=0.0)
    for left, right in zip(first.state_dict().values(), second.state_dict().values(), strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
