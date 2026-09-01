from __future__ import annotations

import hashlib
import io
import json
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch.nn import functional as F

from sfora.siglip_head_screen import build_feature_split_authority
from sfora.siglip_ssor import (
    SSOR_BETA_GRID,
    SSORDiagnosticEvidence,
    SSORInnerFoldEvidence,
    SSOROuterFoldEvidence,
    SSORProjectorEvidence,
    canonical_ssor_result_bytes,
    compose_restored_head,
    restore_descriptors,
    run_ssor_nested_diagnostic,
    seen_class_projector,
    ssor_float_tensor_sha256,
    ssor_recall_at_one_hits,
    validate_ssor_result_bytes,
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


def _nested_result_fixture() -> tuple[bytes, dict[str, object]]:
    descriptors, labels = _nested_fixture()
    ordered_ids = tuple(f"ssor-row-{row:04d}" for row in range(labels.numel()))
    authority = build_feature_split_authority(
        source_manifest_sha256="3" * 64,
        role="optimization-train",
        official_test_access=False,
        ordered_example_ids=ordered_ids,
        features=descriptors,
    )
    evidence = run_ssor_nested_diagnostic(
        descriptors,
        labels,
        ordered_example_ids=ordered_ids,
        split_authority=authority,
    )
    control_head = torch.eye(descriptors.shape[1], dtype=torch.float32).contiguous()
    raw = canonical_ssor_result_bytes(
        evidence,
        source_manifest_sha256=authority.source_manifest_sha256,
        feature_cache_manifest_sha256="4" * 64,
        ordered_example_ids_sha256=authority.ordered_example_ids_sha256,
        feature_matrix_sha256=authority.feature_matrix_sha256,
        label_vector_sha256="5" * 64,
        control_head_weight=control_head,
        deployment_projector=None,
        deployment_head_artifact=None,
    )
    payload = json.loads(raw)
    identities: dict[str, object] = {
        "expected_source_manifest_sha256": authority.source_manifest_sha256,
        "expected_feature_cache_manifest_sha256": "4" * 64,
        "expected_ordered_example_ids_sha256": authority.ordered_example_ids_sha256,
        "expected_feature_matrix_sha256": authority.feature_matrix_sha256,
        "expected_label_vector_sha256": "5" * 64,
        "expected_control_head_sha256": payload["control_head_sha256"],
        "expected_deployment_projector_sha256": None,
        "expected_deployment_head_sha256": None,
        "expected_deployment_head_file_sha256": None,
    }
    return raw, identities


def _npy_bytes(tensor: torch.Tensor) -> bytes:
    stream = io.BytesIO()
    np.save(stream, tensor.numpy().astype("<f4", copy=False), allow_pickle=False)
    return stream.getvalue()


def _passing_evidence_fixture() -> tuple[
    SSORDiagnosticEvidence, SSORProjectorEvidence, torch.Tensor, bytes
]:
    descriptors, labels = _nested_fixture()
    projector = seen_class_projector(
        descriptors,
        labels,
        fit_labels=tuple(range(12)),
    )
    control_head = torch.eye(descriptors.shape[1], dtype=torch.float32).contiguous()
    beta_hits = (10, 10, 10, 10, 12, 11)
    all_beta_hits = (15, 16, 16, 16, 18, 17)
    folds: list[SSOROuterFoldEvidence] = []
    for ordinal in range(4):
        validation_labels = tuple(range(ordinal * 3, ordinal * 3 + 3))
        fit_labels = tuple(label for label in range(12) if label not in validation_labels)
        inner_folds: list[SSORInnerFoldEvidence] = []
        for inner_ordinal in range(3):
            inner_validation = fit_labels[inner_ordinal * 3 : inner_ordinal * 3 + 3]
            inner_fit = tuple(label for label in fit_labels if label not in inner_validation)
            inner_folds.append(
                SSORInnerFoldEvidence(
                    ordinal=inner_ordinal,
                    fit_labels=inner_fit,
                    validation_labels=inner_validation,
                    projector_rank=len(inner_fit),
                    mean_complement_energy=0.25,
                    query_count=20,
                    beta_hits=beta_hits,
                )
            )
        folds.append(
            SSOROuterFoldEvidence(
                ordinal=ordinal,
                fit_labels=fit_labels,
                validation_labels=validation_labels,
                projector_rank=len(fit_labels),
                mean_complement_energy=0.25,
                selected_beta=1.5,
                query_count=30,
                identity_hits=15,
                scalar_identity_hits=15,
                ssor_hits=18,
                scalar_ssor_hits=18,
                all_beta_hits=all_beta_hits,
                inner_fold_schedule_sha256=f"{ordinal + 1:x}" * 64,
                inner_folds=tuple(inner_folds),
            )
        )
    evidence = SSORDiagnosticEvidence(
        beta_grid=SSOR_BETA_GRID,
        fold_schedule_sha256="a" * 64,
        folds=tuple(folds),
        selected_betas=(1.5, 1.5, 1.5, 1.5),
        deployment_beta=1.5,
        consensus_count=4,
        deployment_projector_rank=projector.rank,
        deployment_mean_complement_energy=projector.mean_complement_energy,
        query_count=120,
        identity_hits=60,
        identity_errors=60,
        materiality_eligible=True,
        ssor_hits=72,
        identity_recall_ppm=500_000,
        ssor_recall_ppm=600_000,
        delta_ppm=100_000,
        fold_wins=4,
        minimum_fold_delta_ppm=100_000,
        valid=True,
        passed=True,
    )
    deployed = compose_restored_head(control_head, projector, beta=1.5)
    artifact = _npy_bytes(deployed)
    return evidence, projector, control_head, artifact


def _passing_result_fixture() -> tuple[bytes, bytes, dict[str, object]]:
    evidence, projector, control_head, artifact = _passing_evidence_fixture()
    raw = canonical_ssor_result_bytes(
        evidence,
        source_manifest_sha256="3" * 64,
        feature_cache_manifest_sha256="4" * 64,
        ordered_example_ids_sha256="5" * 64,
        feature_matrix_sha256="6" * 64,
        label_vector_sha256="7" * 64,
        control_head_weight=control_head,
        deployment_projector=projector,
        deployment_head_artifact=artifact,
    )
    payload = json.loads(raw)
    identities: dict[str, object] = {
        "expected_source_manifest_sha256": "3" * 64,
        "expected_feature_cache_manifest_sha256": "4" * 64,
        "expected_ordered_example_ids_sha256": "5" * 64,
        "expected_feature_matrix_sha256": "6" * 64,
        "expected_label_vector_sha256": "7" * 64,
        "expected_control_head_sha256": payload["control_head_sha256"],
        "expected_deployment_projector_sha256": payload["deployment_projector_sha256"],
        "expected_deployment_head_sha256": payload["deployment_head_sha256"],
        "expected_deployment_head_file_sha256": hashlib.sha256(artifact).hexdigest(),
    }
    return raw, artifact, identities


def test_ssor_passing_result_binds_nonidentity_beta_and_exact_head_artifact() -> None:
    raw, artifact, identities = _passing_result_fixture()

    result = validate_ssor_result_bytes(raw, **identities)

    assert result["passed"] is True
    assert result["deployment_beta"] == 1.5
    assert result["deployment_head_file_sha256"] == hashlib.sha256(artifact).hexdigest()
    deployed = torch.from_numpy(np.load(io.BytesIO(artifact), allow_pickle=False)).contiguous()
    assert result["deployment_head_sha256"] == ssor_float_tensor_sha256("deployment-head", deployed)
    assert result["ssor_hits"] == 72

    stale_artifact = artifact[:-1] + bytes([artifact[-1] ^ 1])
    evidence, projector, control_head, _artifact = _passing_evidence_fixture()
    with pytest.raises(ValueError, match="deployment artifact"):
        canonical_ssor_result_bytes(
            evidence,
            source_manifest_sha256="3" * 64,
            feature_cache_manifest_sha256="4" * 64,
            ordered_example_ids_sha256="5" * 64,
            feature_matrix_sha256="6" * 64,
            label_vector_sha256="7" * 64,
            control_head_weight=control_head,
            deployment_projector=projector,
            deployment_head_artifact=stale_artifact,
        )


def test_ssor_result_accepts_signed_fold_tolerance_and_negative_null_delta() -> None:
    evidence, projector, control_head, _artifact = _passing_evidence_fixture()
    tolerated_folds: list[SSOROuterFoldEvidence] = []
    for ordinal, fold in enumerate(evidence.folds):
        deployed_hits = 149 if ordinal == 3 else 190
        tolerated_folds.append(
            replace(
                fold,
                query_count=300,
                identity_hits=150,
                scalar_identity_hits=150,
                ssor_hits=deployed_hits,
                scalar_ssor_hits=deployed_hits,
                all_beta_hits=(150, 160, 160, 160, deployed_hits, 170),
            )
        )
    tolerated = replace(
        evidence,
        folds=tuple(tolerated_folds),
        query_count=1_200,
        identity_hits=600,
        identity_errors=600,
        ssor_hits=719,
        identity_recall_ppm=500_000,
        ssor_recall_ppm=599_166,
        delta_ppm=99_166,
        fold_wins=3,
        minimum_fold_delta_ppm=-3_334,
    )
    tolerated_artifact = _npy_bytes(compose_restored_head(control_head, projector, beta=1.5))
    tolerated_raw = canonical_ssor_result_bytes(
        tolerated,
        source_manifest_sha256="3" * 64,
        feature_cache_manifest_sha256="4" * 64,
        ordered_example_ids_sha256="5" * 64,
        feature_matrix_sha256="6" * 64,
        label_vector_sha256="7" * 64,
        control_head_weight=control_head,
        deployment_projector=projector,
        deployment_head_artifact=tolerated_artifact,
    )
    assert json.loads(tolerated_raw)["minimum_fold_delta_ppm"] == -3_334

    null_folds = tuple(
        replace(
            fold,
            identity_hits=20,
            scalar_identity_hits=20,
            ssor_hits=17,
            scalar_ssor_hits=17,
            all_beta_hits=(20, 19, 19, 19, 17, 18),
        )
        for fold in evidence.folds
    )
    null = replace(
        evidence,
        folds=null_folds,
        identity_hits=80,
        identity_errors=40,
        ssor_hits=68,
        identity_recall_ppm=666_666,
        ssor_recall_ppm=566_666,
        delta_ppm=-100_000,
        fold_wins=0,
        minimum_fold_delta_ppm=-100_000,
        passed=False,
    )
    null_raw = canonical_ssor_result_bytes(
        null,
        source_manifest_sha256="3" * 64,
        feature_cache_manifest_sha256="4" * 64,
        ordered_example_ids_sha256="5" * 64,
        feature_matrix_sha256="6" * 64,
        label_vector_sha256="7" * 64,
        control_head_weight=control_head,
        deployment_projector=None,
        deployment_head_artifact=None,
    )
    assert json.loads(null_raw)["delta_ppm"] == -100_000


@pytest.mark.parametrize(
    "mutation",
    ("deployed-hit", "deployment-beta", "logical-digest", "file-digest"),
)
def test_ssor_nonidentity_result_rejects_deployment_mutations(mutation: str) -> None:
    raw, _artifact, identities = _passing_result_fixture()
    payload = json.loads(raw)
    mutated_identities = identities.copy()
    if mutation == "deployed-hit":
        payload["folds"][0]["all_beta_hits"][4] -= 1
    elif mutation == "deployment-beta":
        payload["deployment_beta"] = 1.0
    elif mutation == "logical-digest":
        mutated_identities["expected_deployment_head_sha256"] = "0" * 64
    elif mutation == "file-digest":
        mutated_identities["expected_deployment_head_file_sha256"] = "0" * 64
    else:  # pragma: no cover
        raise AssertionError(mutation)
    if mutation in {"deployed-hit", "deployment-beta"}:
        unsigned = {key: value for key, value in payload.items() if key != "result_sha256"}
        payload["result_sha256"] = hashlib.sha256(
            json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        ).hexdigest()
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"

    with pytest.raises(ValueError, match="SSOR result"):
        validate_ssor_result_bytes(raw, **mutated_identities)


def test_ssor_result_is_canonical_and_validator_reconstructs_every_gate() -> None:
    raw, identities = _nested_result_fixture()

    result = validate_ssor_result_bytes(raw, **identities)

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert json.dumps(result, sort_keys=True, separators=(",", ":")).encode() + b"\n" == raw
    assert result["schema"] == "sfora-siglip-ssor-v1"
    assert result["claim_eligible"] is False
    assert result["official_test_access"] is False
    assert (
        result["result_sha256"]
        == __import__("hashlib")
        .sha256(
            json.dumps(
                {key: value for key, value in result.items() if key != "result_sha256"},
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )
        .hexdigest()
    )


@pytest.mark.parametrize(
    ("mutation", "value"),
    [
        ("schema", "sfora-siglip-ssor-v2"),
        ("claim_eligible", 0),
        ("query_count", 1),
        ("identity_errors", 999),
        ("deployment_beta", 2.0),
        ("passed", True),
        ("result_sha256", "0" * 64),
    ],
)
def test_ssor_result_rejects_schema_type_identity_and_gate_drift(
    mutation: str, value: object
) -> None:
    raw, identities = _nested_result_fixture()
    payload = json.loads(raw)
    payload[mutation] = value
    if mutation != "result_sha256":
        unsigned = {key: item for key, item in payload.items() if key != "result_sha256"}
        payload["result_sha256"] = (
            __import__("hashlib")
            .sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + b"\n")
            .hexdigest()
        )
    mutated = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="SSOR result"):
        validate_ssor_result_bytes(mutated, **identities)


def test_ssor_result_rejects_fold_and_inner_selection_drift() -> None:
    raw, identities = _nested_result_fixture()
    payload = json.loads(raw)
    payload["folds"][0]["inner_folds"][0]["beta_hits"][1] ^= 1
    unsigned = {key: value for key, value in payload.items() if key != "result_sha256"}
    payload["result_sha256"] = (
        __import__("hashlib")
        .sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        .hexdigest()
    )
    mutated = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="SSOR result"):
        validate_ssor_result_bytes(mutated, **identities)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("valid",), 1),
        (("materiality_eligible",), 1),
        (("query_count",), 48.0),
        (("deployment_projector_rank",), 12.0),
        (("beta_grid",), [1, 0.5, 0.75, 1.25, 1.5, 2]),
        (("folds", 0, "ordinal"), 0.0),
        (("folds", 0, "identity_hits"), 11.0),
    ],
)
def test_ssor_result_rejects_concrete_type_drift(path: tuple[object, ...], value: object) -> None:
    raw, identities = _nested_result_fixture()
    payload = json.loads(raw)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    unsigned = {key: item for key, item in payload.items() if key != "result_sha256"}
    payload["result_sha256"] = (
        __import__("hashlib")
        .sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        .hexdigest()
    )
    mutated = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="SSOR result"):
        validate_ssor_result_bytes(mutated, **identities)


def test_ssor_result_rejects_impossible_inner_partitions() -> None:
    raw, identities = _nested_result_fixture()
    payload = json.loads(raw)
    inner = payload["folds"][0]["inner_folds"][0]
    inner["validation_labels"] = sorted([*inner["fit_labels"], *inner["validation_labels"]])
    inner["fit_labels"] = []
    inner["projector_rank"] = 0
    unsigned = {key: item for key, item in payload.items() if key != "result_sha256"}
    payload["result_sha256"] = (
        __import__("hashlib")
        .sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        .hexdigest()
    )
    mutated = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="SSOR result inner partition"):
        validate_ssor_result_bytes(mutated, **identities)


def test_ssor_result_rejects_noncanonical_schema_and_identity_bytes() -> None:
    raw, identities = _nested_result_fixture()
    payload = json.loads(raw)
    for mutated in (
        json.dumps(payload, sort_keys=False, indent=1).encode() + b"\n",
        raw.replace(b'"schema":', b'"extra":0,"schema":', 1),
    ):
        with pytest.raises(ValueError, match="SSOR result"):
            validate_ssor_result_bytes(mutated, **identities)

    payload["source_manifest_sha256"] = "9" * 64
    unsigned = {key: item for key, item in payload.items() if key != "result_sha256"}
    payload["result_sha256"] = (
        __import__("hashlib")
        .sha256(json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + b"\n")
        .hexdigest()
    )
    mutated = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with pytest.raises(ValueError, match="SSOR result identity"):
        validate_ssor_result_bytes(mutated, **identities)


def test_ssor_result_requires_derived_deployment_artifacts_for_a_pass() -> None:
    descriptors, labels = _nested_fixture()
    ordered_ids = tuple(f"ssor-row-{row:04d}" for row in range(labels.numel()))
    authority = build_feature_split_authority(
        source_manifest_sha256="3" * 64,
        role="optimization-train",
        official_test_access=False,
        ordered_example_ids=ordered_ids,
        features=descriptors,
    )
    evidence = run_ssor_nested_diagnostic(
        descriptors,
        labels,
        ordered_example_ids=ordered_ids,
        split_authority=authority,
    )
    forged = replace(evidence, passed=True)
    with pytest.raises(ValueError, match="deployment identity"):
        canonical_ssor_result_bytes(
            forged,
            source_manifest_sha256=authority.source_manifest_sha256,
            feature_cache_manifest_sha256="4" * 64,
            ordered_example_ids_sha256=authority.ordered_example_ids_sha256,
            feature_matrix_sha256=authority.feature_matrix_sha256,
            label_vector_sha256="5" * 64,
            control_head_weight=torch.eye(descriptors.shape[1], dtype=torch.float32),
            deployment_projector=None,
            deployment_head_artifact=None,
        )
