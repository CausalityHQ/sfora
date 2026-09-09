from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from torch import nn

import sfora
from sfora.teacher_anchored_distillation import (
    TeacherAnchoredConfig,
    TeacherAnchoredNumericalError,
    TeacherAnchorSchedule,
    TeacherNeighborBatches,
    TeacherNeighborRanking,
    embedding_geometry_diagnostics,
    teacher_anchor_schedule,
    teacher_anchored_forward,
    teacher_anchored_loss,
    teacher_neighbor_batches,
    teacher_neighbor_ranking,
)


def test_forward_normalizes_features_before_the_affine_head() -> None:
    encoder = nn.Linear(3, 3, bias=False)
    head = nn.Linear(3, 2)
    with torch.no_grad():
        encoder.weight.copy_(2.0 * torch.eye(3))
        head.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        head.bias.copy_(torch.tensor([0.25, -0.5]))
    images = torch.tensor([[3.0, 4.0, 0.0]], dtype=torch.float32)

    features, codes = teacher_anchored_forward(encoder, head, images)

    expected_features = torch.tensor([[0.6, 0.8, 0.0]], dtype=torch.float32)
    expected_codes = torch.nn.functional.normalize(
        torch.tensor([[0.85, 0.3]], dtype=torch.float32), dim=1
    )
    torch.testing.assert_close(features, expected_features, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(codes, expected_codes, rtol=0.0, atol=1e-7)
    assert not torch.equal(codes, torch.nn.functional.normalize(head(encoder(images)), dim=1))


def test_forward_fences_the_float32_path_from_ambient_autocast() -> None:
    encoder = nn.Linear(3, 3, bias=False)
    head = nn.Linear(3, 2)
    images = torch.tensor([[0.1, 0.2, 0.3], [0.7, -0.4, 0.9]], dtype=torch.float32)

    expected = teacher_anchored_forward(encoder, head, images)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actual = teacher_anchored_forward(encoder, head, images)

    torch.testing.assert_close(actual[0], expected[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("value", (float("nan"), 0.0))
def test_forward_classifies_nonfinite_or_zero_feature_norm_as_numerical(value: float) -> None:
    encoder = nn.Linear(3, 3, bias=False)
    head = nn.Linear(3, 2)
    images = torch.ones((2, 3), dtype=torch.float32)
    with torch.no_grad():
        encoder.weight.zero_()
        encoder.weight[0, 0] = value

    with pytest.raises(TeacherAnchoredNumericalError, match="forward numerical"):
        teacher_anchored_forward(encoder, head, images)


def test_forward_classifies_nonfinite_or_zero_code_norm_as_numerical() -> None:
    encoder = nn.Linear(3, 3, bias=False)
    head = nn.Linear(3, 2)
    images = torch.ones((2, 3), dtype=torch.float32)
    with torch.no_grad():
        encoder.weight.copy_(torch.eye(3))
        head.weight.zero_()
        head.bias.zero_()

    with pytest.raises(TeacherAnchoredNumericalError, match="forward numerical"):
        teacher_anchored_forward(encoder, head, images)


def test_forward_classifies_float32_normalization_overflow_as_numerical() -> None:
    encoder = nn.Identity()
    head = nn.Linear(3, 2)
    with torch.no_grad():
        head.weight.zero_()
        head.bias.copy_(torch.tensor([1.0, -1.0]))
    images = torch.tensor(
        [[1.0e20, -1.0e20, 1.0e20], [-1.0e20, 1.0e20, 1.0e20]],
        dtype=torch.float32,
    )

    with pytest.raises(TeacherAnchoredNumericalError, match="forward numerical"):
        teacher_anchored_forward(encoder, head, images)


def _unit_codes(rows: int, *, seed: int = 17) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(seed))
    values = generator.normal(size=(rows, 128)).astype(np.float32)
    values /= np.linalg.norm(values.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    return np.ascontiguousarray(values)


def test_teacher_anchored_config_is_the_frozen_generic_recipe() -> None:
    config = TeacherAnchoredConfig()

    assert config.dimensions == 128
    assert config.temperatures == (0.05, 0.20)
    assert config.nearest_anchor_count == 64
    assert config.middle_anchor_count == 64
    assert config.uniform_anchor_count == 384
    assert config.anchor_count == 512
    assert config.seed_rows_per_batch == 128
    assert config.batch_rows == 256
    assert config.anchor_weight == 1.0
    assert config.point_weight == 0.1
    assert config.symmetric_weight == 0.5
    assert config.drift_weight == 0.05
    assert config.covariance_weight == 0.01


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("dimensions", 127),
        ("dimensions", True),
        ("temperatures", (0.20, 0.05)),
        ("nearest_anchor_count", 63),
        ("uniform_anchor_count", 383),
        ("seed_rows_per_batch", 64),
        ("anchor_weight", 0.5),
        ("covariance_weight", float("nan")),
    ],
)
def test_teacher_anchored_config_rejects_recipe_drift(field: str, value: object) -> None:
    values = {
        "dimensions": 128,
        "temperatures": (0.05, 0.20),
        "nearest_anchor_count": 64,
        "middle_anchor_count": 64,
        "uniform_anchor_count": 384,
        "seed_rows_per_batch": 128,
        "anchor_weight": 1.0,
        "point_weight": 0.1,
        "symmetric_weight": 0.5,
        "drift_weight": 0.05,
        "covariance_weight": 0.01,
    }
    values[field] = value
    with pytest.raises(ValueError, match="teacher-anchored configuration"):
        TeacherAnchoredConfig(**values)  # type: ignore[arg-type]


def test_anchor_schedule_is_deterministic_disjoint_and_rank_bounded() -> None:
    codes = _unit_codes(897)
    original = codes.copy()
    sample_ids = tuple(f"sample-{index:04d}" for index in range(len(codes)))

    first = teacher_anchor_schedule(codes, sample_ids, seed=17)
    second = teacher_anchor_schedule(codes.copy(), sample_ids, seed=17)
    changed = teacher_anchor_schedule(codes, sample_ids, seed=1729)

    assert isinstance(first, TeacherAnchorSchedule)
    assert first.sha256 == second.sha256
    assert np.array_equal(first.row_indexes, second.row_indexes)
    assert first.row_indexes.dtype == np.int64
    assert first.row_indexes.shape == (897, 512)
    assert not first.row_indexes.flags.writeable
    assert np.array_equal(codes, original)
    assert not np.array_equal(first.row_indexes[:, 64:], changed.row_indexes[:, 64:])
    assert first.sha256 == "fb5a91b91da764fb7f7e062e1af68a3792bce9e8e1f32d99707ce41a0abbce80"
    assert first.row_indexes[0, :12].tolist() == [
        225,
        364,
        140,
        894,
        841,
        429,
        266,
        282,
        26,
        50,
        31,
        657,
    ]
    assert first.row_indexes[0, 64:72].tolist() == [322, 700, 533, 157, 753, 857, 791, 620]
    assert first.row_indexes[0, 128:136].tolist() == [208, 765, 553, 42, 611, 782, 815, 71]

    for query in (0, 113, 896):
        similarities = codes.astype(np.float64) @ codes[query].astype(np.float64)
        ranked = sorted(
            (row for row in range(len(codes)) if row != query),
            key=lambda row: (-float(similarities[row]), sample_ids[row]),
        )
        anchors = first.row_indexes[query].tolist()
        assert anchors[:64] == ranked[:64]
        assert set(anchors[64:128]) <= set(ranked[64:512])
        assert set(anchors[128:]).isdisjoint({query, *ranked[:512]})
        assert len(set(anchors)) == 512


def test_anchor_schedule_breaks_exact_similarity_ties_by_sample_identity() -> None:
    codes = _unit_codes(897)
    codes[1] = codes[2]
    codes[0] = codes[1]
    sample_ids = tuple(["query", "z-tie", "a-tie", *[f"row-{i}" for i in range(3, 897)]])

    schedule = teacher_anchor_schedule(codes, sample_ids, seed=17)

    assert schedule.row_indexes[0, :2].tolist() == [2, 1]


def test_anchor_schedule_uses_natural_identity_order_across_byte_boundaries() -> None:
    codes = _unit_codes(897)
    codes[:] = codes[0]
    integer_ids = tuple(range(897))
    string_ids = tuple(["query", "z", "aa", *[f"row-{i:04d}" for i in range(3, 897)]])

    integer_schedule = teacher_anchor_schedule(codes, integer_ids, seed=17)
    string_schedule = teacher_anchor_schedule(codes, string_ids, seed=17)

    assert integer_schedule.row_indexes[0, :64].tolist() == list(range(1, 65))
    assert string_schedule.row_indexes[0, :2].tolist() == [2, 3]


@pytest.mark.parametrize(
    ("mutation", "sample_ids", "seed"),
    [
        ("dtype", None, 17),
        ("nonfinite", None, 17),
        ("norm", None, 17),
        ("duplicate-id", tuple(["x", "x", *range(895)]), 17),
        ("bool-id", tuple([True, *range(1, 897)]), 17),
        ("seed", None, True),
    ],
)
def test_anchor_schedule_rejects_invalid_authority(
    mutation: str, sample_ids: tuple[object, ...] | None, seed: int
) -> None:
    codes = _unit_codes(897)
    if mutation == "dtype":
        codes = codes.astype(np.float64)
    elif mutation == "nonfinite":
        codes[0, 0] = np.nan
    elif mutation == "norm":
        codes[0] *= 0.5
    identities = sample_ids or tuple(range(897))
    with pytest.raises(ValueError, match="teacher anchor authority"):
        teacher_anchor_schedule(codes, identities, seed=seed)  # type: ignore[arg-type]


def test_anchor_schedule_rejects_insufficient_uniform_inventory() -> None:
    with pytest.raises(ValueError, match="teacher anchor authority"):
        teacher_anchor_schedule(_unit_codes(896), tuple(range(896)), seed=17)


def test_neighbor_batches_are_deterministic_collision_safe_and_epoch_sensitive() -> None:
    codes = _unit_codes(390)
    original = codes.copy()
    sample_ids = tuple(f"sample-{index:04d}" for index in range(len(codes)))

    first = teacher_neighbor_batches(codes, sample_ids, seed=17, epoch=1)
    second = teacher_neighbor_batches(codes.copy(), sample_ids, seed=17, epoch=1)
    changed = teacher_neighbor_batches(codes, sample_ids, seed=17, epoch=2)

    assert isinstance(first, TeacherNeighborBatches)
    assert first.sha256 == second.sha256
    assert np.array_equal(first.row_indexes, second.row_indexes)
    assert first.row_indexes.dtype == np.int64
    assert first.row_indexes.shape == (3, 256)
    assert not first.row_indexes.flags.writeable
    assert len(first.dropped_seed_row_indexes) == 6
    assert first.dropped_seed_row_indexes == (338, 285, 374, 34, 296, 86)
    assert first.repeated_identity_count == 289
    assert first.sha256 == "f52b31d942332dce7375ac771e2ba6f2a6d282d9ed41f781cdd14175b228a13d"
    assert first.row_indexes[0, :8].tolist() == [243, 246, 2, 330, 311, 220, 91, 3]
    assert first.row_indexes[0, 128:136].tolist() == [257, 338, 168, 43, 150, 78, 76, 266]
    assert np.array_equal(codes, original)
    assert not np.array_equal(first.row_indexes, changed.row_indexes)
    for batch in first.row_indexes:
        seeds = batch[:128]
        partners = batch[128:]
        assert len(set(seeds.tolist())) == 128
        assert len(set(partners.tolist())) == 128
        assert set(seeds.tolist()).isdisjoint(partners.tolist())


def test_neighbor_batches_choose_the_nearest_available_partner_in_seed_order() -> None:
    codes = _unit_codes(256)
    sample_ids = tuple(range(256))
    batches = teacher_neighbor_batches(codes, sample_ids, seed=17, epoch=1)
    batch = batches.row_indexes[0]
    selected = set(int(row) for row in batch[:128])
    expected = []
    similarities = codes.astype(np.float64) @ codes.astype(np.float64).T
    for seed_row in batch[:128]:
        candidates = sorted(
            (row for row in range(256) if row != int(seed_row) and row not in selected),
            key=lambda row: (-float(similarities[int(seed_row), row]), sample_ids[row]),
        )
        partner = candidates[0]
        expected.append(partner)
        selected.add(partner)
    assert batch[128:].tolist() == expected


def test_neighbor_rankings_are_reusable_across_epochs_and_bind_inputs() -> None:
    codes = _unit_codes(390)
    sample_ids = tuple(range(390))
    ranking = teacher_neighbor_ranking(codes, sample_ids)

    assert isinstance(ranking, TeacherNeighborRanking)
    assert ranking.row_indexes.shape == (390, 256)
    assert ranking.row_indexes.dtype == np.int64
    assert not ranking.row_indexes.flags.writeable
    first = teacher_neighbor_batches(codes, sample_ids, seed=17, epoch=1, ranking=ranking)
    second = teacher_neighbor_batches(codes, sample_ids, seed=17, epoch=2, ranking=ranking)
    assert not np.array_equal(first.row_indexes, second.row_indexes)
    with pytest.raises(ValueError, match="teacher batch authority"):
        teacher_neighbor_batches(
            _unit_codes(390, seed=18), sample_ids, seed=17, epoch=1, ranking=ranking
        )


def test_neighbor_batches_reject_insufficient_or_invalid_authority() -> None:
    with pytest.raises(ValueError, match="teacher batch authority"):
        teacher_neighbor_batches(_unit_codes(255), tuple(range(255)), seed=17, epoch=1)
    with pytest.raises(ValueError, match="teacher batch authority"):
        teacher_neighbor_batches(_unit_codes(256), tuple(range(256)), seed=True, epoch=1)
    with pytest.raises(ValueError, match="teacher batch authority"):
        teacher_neighbor_batches(_unit_codes(256), tuple(range(256)), seed=17, epoch=0)


def test_teacher_anchored_schedule_api_is_public() -> None:
    assert sfora.TeacherAnchoredConfig is TeacherAnchoredConfig
    assert sfora.TeacherAnchoredNumericalError is TeacherAnchoredNumericalError
    assert sfora.TeacherAnchorSchedule is TeacherAnchorSchedule
    assert sfora.TeacherNeighborBatches is TeacherNeighborBatches
    assert sfora.TeacherNeighborRanking is TeacherNeighborRanking
    assert sfora.teacher_anchor_schedule is teacher_anchor_schedule
    assert sfora.teacher_neighbor_batches is teacher_neighbor_batches
    assert sfora.teacher_neighbor_ranking is teacher_neighbor_ranking
    assert sfora.embedding_geometry_diagnostics is embedding_geometry_diagnostics
    assert sfora.teacher_anchored_loss is teacher_anchored_loss


def _normalized_tensor(rows: int, dimensions: int, *, seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(rows, dimensions, generator=generator, dtype=torch.float32)
    return torch.nn.functional.normalize(values, dim=-1)


def _scalar_forward_kl(reference: list[list[float]], candidate: list[list[float]]) -> float:
    total = 0.0
    for reference_row, candidate_row in zip(reference, candidate, strict=True):
        reference_max = max(reference_row)
        candidate_max = max(candidate_row)
        reference_exp = [math.exp(value - reference_max) for value in reference_row]
        candidate_exp = [math.exp(value - candidate_max) for value in candidate_row]
        reference_total = sum(reference_exp)
        reference_log_total = math.log(reference_total) + reference_max
        candidate_log_total = math.log(sum(candidate_exp)) + candidate_max
        for probability, reference_value, candidate_value in zip(
            (value / reference_total for value in reference_exp),
            reference_row,
            candidate_row,
            strict=True,
        ):
            total += probability * (
                reference_value - reference_log_total - candidate_value + candidate_log_total
            )
    return total / len(reference)


def test_teacher_anchored_loss_matches_independent_five_term_reference() -> None:
    student = _normalized_tensor(4, 3, seed=1).requires_grad_()
    teacher = _normalized_tensor(4, 3, seed=2)
    anchors = _normalized_tensor(20, 3, seed=3).reshape(4, 5, 3)
    adapted = _normalized_tensor(4, 6, seed=4).requires_grad_()
    original = _normalized_tensor(4, 6, seed=5)

    result = teacher_anchored_loss(
        student, teacher, anchors, adapted, original, TeacherAnchoredConfig()
    )

    anchor_terms = []
    symmetric_terms = []
    mask = ~torch.eye(4, dtype=torch.bool)
    for temperature in (0.05, 0.20):
        teacher_anchor = (torch.einsum("bd,bad->ba", teacher, anchors) / temperature).tolist()
        student_anchor = (torch.einsum("bd,bad->ba", student, anchors) / temperature).tolist()
        anchor_terms.append(_scalar_forward_kl(teacher_anchor, student_anchor))
        teacher_pairs = ((teacher @ teacher.T) / temperature)[mask].reshape(4, 3).tolist()
        student_pairs = ((student @ student.T) / temperature)[mask].reshape(4, 3).tolist()
        symmetric_terms.append(_scalar_forward_kl(teacher_pairs, student_pairs))
    expected_anchor = torch.tensor(sum(anchor_terms) / len(anchor_terms))
    expected_symmetric = torch.tensor(sum(symmetric_terms) / len(symmetric_terms))
    expected_point = (1.0 - torch.sum(student * teacher, dim=-1)).mean()
    expected_drift = torch.mean(torch.square(adapted @ adapted.T - original @ original.T))
    centered_student = student - student.mean(dim=0)
    centered_teacher = teacher - teacher.mean(dim=0)
    student_covariance = centered_student.T @ centered_student / 3
    teacher_covariance = centered_teacher.T @ centered_teacher / 3
    expected_covariance = torch.square(student_covariance - teacher_covariance).sum() / (
        torch.square(teacher_covariance).sum() + 1e-12
    )
    student_sigma = torch.sqrt(torch.diag(student_covariance).clamp_min(0) + 1e-4)
    teacher_sigma = torch.sqrt(torch.diag(teacher_covariance).clamp_min(0) + 1e-4)
    expected_covariance = (
        expected_covariance
        + torch.square(torch.relu(0.5 - student_sigma / (teacher_sigma + 1e-4))).mean()
    )
    expected_total = (
        expected_anchor
        + 0.1 * expected_point
        + 0.5 * expected_symmetric
        + 0.05 * expected_drift
        + 0.01 * expected_covariance
    )

    for actual, expected in (
        (result.anchor, expected_anchor),
        (result.point, expected_point),
        (result.symmetric, expected_symmetric),
        (result.drift, expected_drift),
        (result.covariance, expected_covariance),
        (result.total, expected_total),
    ):
        assert torch.allclose(actual, expected, rtol=1e-6, atol=1e-7)
    assert float(result.anchor.detach()) == pytest.approx(11.812127113342285, rel=1e-7)
    assert float(result.symmetric.detach()) == pytest.approx(8.88290786743164, rel=1e-7)
    assert float(result.total.detach()) == pytest.approx(16.406381607055664, rel=1e-7)

    result.total.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert adapted.grad is not None and torch.isfinite(adapted.grad).all()
    assert float(student.grad.abs().sum()) > 0
    assert float(adapted.grad.abs().sum()) > 0


def test_teacher_anchored_loss_and_diagnostics_stay_finite_for_constant_codes() -> None:
    student = torch.nn.functional.normalize(torch.ones(4, 3), dim=-1).requires_grad_()
    teacher = student.detach().clone()
    anchors = teacher.detach().clone()[:, None, :].expand(-1, 5, -1).contiguous()
    adapted = torch.nn.functional.normalize(torch.ones(4, 6), dim=-1).requires_grad_()
    original = adapted.detach().clone()

    result = teacher_anchored_loss(
        student, teacher, anchors, adapted, original, TeacherAnchoredConfig()
    )
    diagnostics = embedding_geometry_diagnostics(student)

    assert all(
        torch.isfinite(value)
        for value in (
            result.total,
            result.anchor,
            result.point,
            result.symmetric,
            result.drift,
            result.covariance,
        )
    )
    assert diagnostics.effective_rank == 0.0
    assert diagnostics.leading_eigenvalue_share == 1.0
    assert diagnostics.top_eight_eigenvalue_share == 1.0
    result.total.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and torch.isfinite(student.grad).all()
    assert adapted.grad is not None and torch.isfinite(adapted.grad).all()


def test_teacher_anchored_loss_supports_head_only_and_no_grad_diagnostics() -> None:
    student = _normalized_tensor(4, 3, seed=11).requires_grad_()
    teacher = _normalized_tensor(4, 3, seed=12)
    anchors = _normalized_tensor(20, 3, seed=13).reshape(4, 5, 3)
    frozen_features = _normalized_tensor(4, 6, seed=14)
    original = _normalized_tensor(4, 6, seed=15)

    head_only = teacher_anchored_loss(
        student,
        teacher,
        anchors,
        frozen_features,
        original,
        TeacherAnchoredConfig(),
    )
    head_only.total.backward()  # type: ignore[no-untyped-call]
    assert student.grad is not None and float(student.grad.abs().sum()) > 0

    diagnostic = teacher_anchored_loss(
        student.detach(),
        teacher,
        anchors,
        frozen_features,
        original,
        TeacherAnchoredConfig(),
    )
    assert not diagnostic.total.requires_grad
    assert torch.isfinite(diagnostic.total)


@pytest.mark.parametrize(
    "mutation", ["student-dtype", "teacher-grad", "zero-norm", "nan", "anchors", "feature-shape"]
)
def test_teacher_anchored_loss_rejects_invalid_authority(mutation: str) -> None:
    student = _normalized_tensor(4, 3, seed=1).requires_grad_()
    teacher = _normalized_tensor(4, 3, seed=2)
    anchors = _normalized_tensor(20, 3, seed=3).reshape(4, 5, 3)
    adapted = _normalized_tensor(4, 6, seed=4).requires_grad_()
    original = _normalized_tensor(4, 6, seed=5)
    if mutation == "student-dtype":
        student = student.double().requires_grad_()
    elif mutation == "teacher-grad":
        teacher.requires_grad_()
    elif mutation == "zero-norm":
        student = student.clone()
        student[0] = 0
    elif mutation == "nan":
        anchors = anchors.clone()
        anchors[0, 0, 0] = torch.nan
    elif mutation == "anchors":
        anchors = anchors[:, :, :2]
    elif mutation == "feature-shape":
        original = original[:, :5]
    expected = TeacherAnchoredNumericalError if mutation == "nan" else ValueError
    with pytest.raises(expected, match="teacher-anchored loss") as error:
        teacher_anchored_loss(student, teacher, anchors, adapted, original, TeacherAnchoredConfig())
    if mutation != "nan":
        assert type(error.value) is ValueError


def test_embedding_geometry_diagnostics_match_covariance_eigenvalues() -> None:
    codes = _normalized_tensor(10, 8, seed=9)
    diagnostics = embedding_geometry_diagnostics(codes)
    centered = codes - codes.mean(dim=0)
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered / 9).clamp_min(0)
    probabilities = eigenvalues / eigenvalues.sum()
    positive = probabilities > 0
    expected_rank = torch.exp(-(probabilities[positive] * probabilities[positive].log()).sum())

    assert diagnostics.effective_rank == pytest.approx(float(expected_rank), rel=1e-6)
    assert diagnostics.leading_eigenvalue_share == pytest.approx(float(probabilities[-1]), rel=1e-6)
    assert diagnostics.top_eight_eigenvalue_share == pytest.approx(1.0, rel=1e-6)


def test_embedding_geometry_diagnostics_known_rank_ten_spectrum() -> None:
    codes = torch.cat((torch.eye(10), -torch.eye(10))).to(torch.float32)
    diagnostics = embedding_geometry_diagnostics(codes)
    assert diagnostics.effective_rank == pytest.approx(10.0, rel=1e-6)
    assert diagnostics.leading_eigenvalue_share == pytest.approx(0.1, rel=1e-6)
    assert diagnostics.top_eight_eigenvalue_share == pytest.approx(0.8, rel=1e-6)


def test_free_student_codes_converge_toward_teacher_without_collapse() -> None:
    teacher = _normalized_tensor(16, 8, seed=41)
    anchor_rows = [[(query + offset) % 16 for offset in range(1, 9)] for query in range(16)]
    anchors = teacher[torch.tensor(anchor_rows)]
    raw_student = (teacher + 0.35 * _normalized_tensor(16, 8, seed=42)).requires_grad_()
    original = _normalized_tensor(16, 12, seed=43)
    adapted = original.clone().requires_grad_()
    optimizer = torch.optim.Adam([raw_student, adapted], lr=0.03)
    initial = float(
        torch.sum(
            torch.nn.functional.normalize(raw_student.detach(), dim=-1) * teacher, dim=-1
        ).mean()
    )

    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        student = torch.nn.functional.normalize(raw_student, dim=-1)
        features = torch.nn.functional.normalize(adapted, dim=-1)
        result = teacher_anchored_loss(
            student, teacher, anchors, features, original, TeacherAnchoredConfig()
        )
        result.total.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

    final_codes = torch.nn.functional.normalize(raw_student.detach(), dim=-1)
    final = float(torch.sum(final_codes * teacher, dim=-1).mean())
    diagnostics = embedding_geometry_diagnostics(final_codes)
    assert final > initial + 0.04
    assert final > 0.99
    assert diagnostics.effective_rank > 3.0
