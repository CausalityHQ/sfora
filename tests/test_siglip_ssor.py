from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from sfora.siglip_head_screen import build_feature_split_authority
from sfora.siglip_ssor import (
    SSOR_BETA_GRID,
    SSORDiagnosticEvidence,
    SSORProjectorEvidence,
    compose_restored_head,
    restore_descriptors,
    run_ssor_nested_diagnostic,
    seen_class_projector,
    ssor_recall_at_one_hits,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor]:
    descriptors = F.normalize(
        torch.tensor(
            [
                [3.0, 0.2, 0.1, 0.4, 0.0, 0.3],
                [2.7, 0.1, 0.3, 0.5, 0.2, 0.1],
                [0.2, 3.0, 0.1, 0.3, 0.4, 0.0],
                [0.1, 2.8, 0.4, 0.2, 0.3, 0.2],
                [0.1, 0.3, 3.0, 0.2, 0.1, 0.4],
                [0.3, 0.1, 2.7, 0.4, 0.2, 0.2],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    ).contiguous()
    labels = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.int64)
    return descriptors, labels


def test_seen_class_projector_has_exact_uncentered_mean_span_authority() -> None:
    descriptors, labels = _fixture()
    evidence = seen_class_projector(descriptors, labels, fit_labels=(0, 1, 2))

    assert type(evidence) is SSORProjectorEvidence
    assert evidence.fit_labels == (0, 1, 2)
    assert evidence.rank == 3
    assert evidence.projector.dtype == torch.float64
    assert evidence.projector.shape == (6, 6)
    assert torch.equal(evidence.projector, evidence.projector.T)
    assert torch.allclose(evidence.projector @ evidence.projector, evidence.projector, atol=1e-12)
    assert float(torch.trace(evidence.projector)) == pytest.approx(3.0, abs=1e-12)
    eigenvalues = torch.linalg.eigvalsh(evidence.projector)
    assert float(eigenvalues.min()) >= -1e-10
    assert float(eigenvalues.max()) <= 1.0 + 1e-10
    assert evidence.mean_span_energy > 0.0
    assert evidence.mean_complement_energy >= 0.0
    assert len(evidence.class_span_energy) == 3
    assert len(evidence.class_complement_energy) == 3
    class_means = torch.stack(
        [
            F.normalize(descriptors[labels == label].double().mean(dim=0), dim=0)
            for label in (0, 1, 2)
        ]
    )
    assert torch.allclose(class_means @ evidence.projector, class_means, atol=1e-10, rtol=0)
    assert evidence.orthogonal_probe_residual <= 1e-10


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("nonunit", "descriptor authority"),
        ("nonfinite", "descriptor authority"),
        ("wrong-label-dtype", "label authority"),
        ("missing-class", "fit-label authority"),
        ("duplicate-fit-label", "fit-label authority"),
        ("unsorted-fit-label", "fit-label authority"),
        ("deficient-rank", "rank authority"),
    ],
)
def test_seen_class_projector_rejects_authority_and_rank_drift(mutation: str, message: str) -> None:
    descriptors, labels = _fixture()
    fit_labels = (0, 1, 2)
    if mutation == "nonunit":
        descriptors = descriptors.clone()
        descriptors[0] *= 0.9
    elif mutation == "nonfinite":
        descriptors = descriptors.clone()
        descriptors[0, 0] = torch.nan
    elif mutation == "wrong-label-dtype":
        labels = labels.to(torch.int32)
    elif mutation == "missing-class":
        fit_labels = (0, 1, 3)
    elif mutation == "duplicate-fit-label":
        fit_labels = (0, 1, 1)
    elif mutation == "unsorted-fit-label":
        fit_labels = (2, 0)
    elif mutation == "deficient-rank":
        descriptors = descriptors.clone()
        descriptors[labels == 2] = descriptors[labels == 1]
    else:  # pragma: no cover
        raise AssertionError(mutation)

    with pytest.raises(ValueError, match=message):
        seen_class_projector(descriptors, labels, fit_labels=fit_labels)


def test_restoration_and_composed_head_are_the_same_linear_map() -> None:
    descriptors, labels = _fixture()
    evidence = seen_class_projector(descriptors, labels, fit_labels=(0, 1, 2))
    generator = torch.Generator().manual_seed(17)
    pooler = torch.randn(9, 11, generator=generator, dtype=torch.float32)
    head = torch.randn(6, 11, generator=generator, dtype=torch.float32)
    control = F.normalize(pooler @ head.T, dim=1)

    identity = restore_descriptors(control, evidence, beta=1.0)
    restored = restore_descriptors(control, evidence, beta=1.5)
    composed = compose_restored_head(head, evidence, beta=1.5)
    deployed = F.normalize(pooler @ composed.T, dim=1)

    assert identity.dtype == torch.float64
    assert torch.allclose(identity, control.double(), atol=2e-7, rtol=2e-7)
    assert composed.dtype == torch.float32
    assert composed.shape == head.shape
    assert torch.allclose(restored.float(), deployed, atol=2e-6, rtol=2e-6)


@pytest.mark.parametrize("beta", [0.0, -1.0, float("nan"), float("inf"), True])
def test_restoration_rejects_invalid_beta(beta: object) -> None:
    descriptors, labels = _fixture()
    evidence = seen_class_projector(descriptors, labels, fit_labels=(0, 1, 2))
    with pytest.raises(ValueError, match="beta authority"):
        restore_descriptors(descriptors, evidence, beta=beta)  # type: ignore[arg-type]


def test_composed_head_rejects_shape_and_finite_drift() -> None:
    descriptors, labels = _fixture()
    evidence = seen_class_projector(descriptors, labels, fit_labels=(0, 1, 2))
    with pytest.raises(ValueError, match="head authority"):
        compose_restored_head(torch.ones(7, 11), evidence, beta=1.5)
    head = torch.ones(6, 11)
    head[0, 0] = torch.inf
    with pytest.raises(ValueError, match="head authority"):
        compose_restored_head(head, evidence, beta=1.5)


def _nested_fixture() -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(2901)
    centers = F.normalize(torch.randn(12, 16, generator=generator), dim=1)
    rows: list[torch.Tensor] = []
    row_labels: list[int] = []
    for label, center in enumerate(centers):
        for _sample in range(4):
            noise = 0.015 * torch.randn(16, generator=generator)
            rows.append(F.normalize(center + noise, dim=0))
            row_labels.append(label)
    return torch.stack(rows).float().contiguous(), torch.tensor(row_labels, dtype=torch.int64)


def test_recall_at_one_has_lowest_row_ties_and_scalar_replay() -> None:
    descriptors = F.normalize(
        torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float32),
        dim=1,
    ).contiguous()
    labels = torch.tensor([0, 0, 1], dtype=torch.int64)

    vectorized = ssor_recall_at_one_hits(descriptors, labels, scalar=False)
    scalar = ssor_recall_at_one_hits(descriptors, labels, scalar=True)

    assert vectorized == scalar == (1, 3)


def test_nested_ssor_is_class_disjoint_deterministic_and_uses_registered_ties() -> None:
    descriptors, labels = _nested_fixture()
    authority = build_feature_split_authority(
        source_manifest_sha256="3" * 64,
        role="optimization-train",
        official_test_access=False,
        ordered_example_ids=tuple(f"ssor-row-{row:04d}" for row in range(labels.numel())),
        features=descriptors,
    )

    ordered_ids = tuple(f"ssor-row-{row:04d}" for row in range(labels.numel()))
    first = run_ssor_nested_diagnostic(
        descriptors,
        labels,
        ordered_example_ids=ordered_ids,
        split_authority=authority,
    )
    second = run_ssor_nested_diagnostic(
        descriptors,
        labels,
        ordered_example_ids=ordered_ids,
        split_authority=authority,
    )

    assert type(first) is SSORDiagnosticEvidence
    assert first == second
    assert first.beta_grid == SSOR_BETA_GRID == (1.0, 0.5, 0.75, 1.25, 1.5, 2.0)
    assert len(first.folds) == 4
    assert sorted(label for fold in first.folds for label in fold.validation_labels) == list(
        range(12)
    )
    assert all(set(fold.fit_labels).isdisjoint(fold.validation_labels) for fold in first.folds)
    assert all(fold.scalar_identity_hits == fold.identity_hits for fold in first.folds)
    assert all(fold.scalar_ssor_hits == fold.ssor_hits for fold in first.folds)
    assert all(fold.selected_beta in SSOR_BETA_GRID for fold in first.folds)
    assert all(len(fold.inner_folds) == 3 for fold in first.folds)
    assert first.deployment_projector_rank == 12
    assert first.deployment_mean_complement_energy >= 0.0
    for outer in first.folds:
        assert outer.projector_rank == len(outer.fit_labels)
        assert outer.mean_complement_energy >= 0.0
        assert sorted(
            label for inner in outer.inner_folds for label in inner.validation_labels
        ) == list(outer.fit_labels)
        assert all(
            set(inner.fit_labels).isdisjoint(inner.validation_labels)
            and set(inner.fit_labels) | set(inner.validation_labels) == set(outer.fit_labels)
            and set(inner.validation_labels).isdisjoint(outer.validation_labels)
            for inner in outer.inner_folds
        )
        assert all(
            inner.projector_rank == len(inner.fit_labels) and inner.mean_complement_energy >= 0.0
            for inner in outer.inner_folds
        )
        aggregate = tuple(
            sum(inner.beta_hits[index] for inner in outer.inner_folds)
            for index in range(len(SSOR_BETA_GRID))
        )
        expected = SSOR_BETA_GRID[
            min(range(len(SSOR_BETA_GRID)), key=lambda index: (-aggregate[index], index))
        ]
        assert outer.selected_beta == expected
        assert len(outer.all_beta_hits) == len(SSOR_BETA_GRID)
        assert outer.inner_fold_schedule_sha256 != first.fold_schedule_sha256
    assert first.query_count == labels.numel()
    assert first.identity_recall_ppm == first.identity_hits * 1_000_000 // labels.numel()
    assert first.ssor_recall_ppm == first.ssor_hits * 1_000_000 // labels.numel()
    assert first.identity_errors == first.query_count - first.identity_hits
    assert first.materiality_eligible == (first.identity_errors >= 40)
    assert first.valid
    counts = {beta: first.selected_betas.count(beta) for beta in SSOR_BETA_GRID}
    consensus = next((beta for beta in SSOR_BETA_GRID if counts[beta] >= 3), None)
    assert first.deployment_beta == consensus
    assert first.consensus_count == (0 if consensus is None else counts[consensus])
    consensus_index = 0 if consensus is None else SSOR_BETA_GRID.index(consensus)
    assert first.ssor_hits == sum(fold.all_beta_hits[consensus_index] for fold in first.folds)

    mutated_ids = list(ordered_ids)
    mutated_ids[0], mutated_ids[1] = mutated_ids[1], mutated_ids[0]
    with pytest.raises(ValueError, match="split authority"):
        run_ssor_nested_diagnostic(
            descriptors,
            labels,
            ordered_example_ids=tuple(mutated_ids),
            split_authority=authority,
        )
