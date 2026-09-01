from __future__ import annotations

import copy
import json
from dataclasses import replace

import numpy as np
import pytest

from sfora.twin_reachability import (
    TwinReachabilityAuthority,
    TwinReachabilityEvidence,
    build_twin_reachability,
    canonical_twin_reachability_artifact_bytes,
    canonical_twin_reachability_bytes,
    validate_twin_reachability_artifact_bytes,
    validate_twin_reachability_bytes,
)


def _separable() -> tuple[np.ndarray, np.ndarray]:
    negative = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float32), (20, 1))
    positive = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float32), (20, 1))
    return np.concatenate((negative, positive)), np.asarray(
        [82] * 20 + [83] * 20, dtype=np.int64
    )


def _authority() -> TwinReachabilityAuthority:
    return TwinReachabilityAuthority(
        plane="trained-raw",
        source_revision="1" * 40,
        source_tree_digest="2" * 64,
        dataset_revision="3" * 40,
        dataset_manifest_sha256="4" * 64,
        model_name="google/siglip-so400m-patch14-384",
        model_revision="5" * 40,
        producer_kind="trained-checkpoint",
        producer_identity="6" * 64,
        ordered_example_ids_sha256="7" * 64,
        label_vector_sha256="8" * 64,
        descriptor_sha256="9" * 64,
    )


def test_global_twin_cue_is_exact_scale_invariant_and_deterministic() -> None:
    descriptors, labels = _separable()

    evidence = build_twin_reachability("frozen-pooled", descriptors, labels)
    scaled = build_twin_reachability(
        "frozen-pooled",
        descriptors * np.linspace(1.0, 7.0, 40, dtype=np.float32)[:, None],
        labels,
    )

    assert isinstance(evidence, TwinReachabilityEvidence)
    assert evidence.plane == "frozen-pooled"
    assert evidence.source_count == 40
    assert evidence.class_82_count == 20
    assert evidence.class_83_count == 20
    assert evidence.signed_scores == (-1.0,) * 20 + (1.0,) * 20
    assert evidence.full_auc == 1.0
    assert evidence.cue_present is True
    assert evidence == scaled


@pytest.mark.parametrize(
    ("plane", "descriptors", "labels", "message"),
    (
        ("", np.ones((4, 2), dtype=np.float32), np.asarray([82, 82, 83, 83]), "plane"),
        ("x", np.ones(4, dtype=np.float32), np.asarray([82, 82, 83, 83]), "rank"),
        ("x", np.ones((4, 2), dtype=np.int64), np.asarray([82, 82, 83, 83]), "dtype"),
        ("x", np.ones((4, 2), dtype=np.float16), np.asarray([82, 82, 83, 83]), "dtype"),
        ("x", np.ones((4, 2), dtype=np.float32), np.asarray([[82, 82, 83, 83]]), "rank"),
        ("x", np.ones((3, 2), dtype=np.float32), np.asarray([82, 82, 83, 83]), "length"),
        (
            "x",
            np.asarray([[0.0, 0.0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
            np.asarray([82, 82, 83, 83]),
            "norm",
        ),
        (
            "x",
            np.asarray([[np.nan, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32),
            np.asarray([82, 82, 83, 83]),
            "finite",
        ),
        ("x", np.ones((3, 2), dtype=np.float32), np.asarray([82, 83, 83]), "class"),
        ("x", np.ones((4, 2), dtype=np.float32), np.asarray([82, 82, 83, 84]), "labels"),
    ),
)
def test_twin_authority_rejects_invalid_inputs(
    plane: str,
    descriptors: np.ndarray,
    labels: np.ndarray,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        build_twin_reachability(plane, descriptors, labels)


def test_twin_authority_rejects_concrete_container_and_boolean_labels() -> None:
    descriptors, labels = _separable()
    with pytest.raises(TypeError, match="array"):
        build_twin_reachability("x", descriptors.tolist(), labels)  # type: ignore[arg-type]
    mutated = labels.astype(object)
    mutated[0] = True
    with pytest.raises((TypeError, ValueError), match="dtype"):
        build_twin_reachability("x", descriptors, mutated)


def test_tied_scores_have_half_auc() -> None:
    descriptors = np.ones((40, 3), dtype=np.float64)
    labels = np.asarray([82] * 20 + [83] * 20, dtype=np.int64)

    evidence = build_twin_reachability("tied", descriptors, labels)

    assert evidence.signed_scores == (0.0,) * 40
    assert evidence.full_auc == 0.5
    assert evidence.cue_present is False


@pytest.mark.parametrize("dimension", (4, 64))
def test_shrinkage_lda_recovers_linear_cue_hidden_from_centroid_geometry(
    dimension: int,
) -> None:
    rng = np.random.default_rng(0)
    count = 31
    negative = np.column_stack(
        (
            -3.0 + 0.3 * rng.standard_normal(count),
            20.0 * rng.standard_normal((count, 3)),
        )
    )
    positive = np.column_stack(
        (
            3.0 + 0.3 * rng.standard_normal(count),
            20.0 * rng.standard_normal((count, 3)),
        )
    )
    descriptors = np.concatenate((negative, positive)).astype(np.float64)
    descriptors = np.pad(descriptors, ((0, 0), (0, dimension - 4)))
    labels = np.asarray([82] * count + [83] * count, dtype=np.int64)

    evidence = build_twin_reachability("nuisance", descriptors, labels)

    assert evidence.full_auc < 0.80
    assert evidence.centroid_cue_present is False
    assert evidence.lda_full_auc > 0.95
    assert evidence.cue_present is True


@pytest.mark.parametrize("dimension", (4, 64))
def test_dual_shrinkage_lda_matches_direct_primal_reference(dimension: int) -> None:
    rng = np.random.default_rng(37)
    descriptors = rng.standard_normal((40, dimension)).astype(np.float64)
    descriptors[20:, 0] += 1.5
    labels = np.asarray([82] * 20 + [83] * 20, dtype=np.int64)
    normalized = descriptors / np.linalg.vector_norm(descriptors, axis=1)[:, None]
    expected = []
    for query_index in range(labels.size):
        retained = np.arange(labels.size) != query_index
        training = normalized[retained]
        training_labels = labels[retained]
        mean_82 = training[training_labels == 82].mean(axis=0)
        mean_83 = training[training_labels == 83].mean(axis=0)
        centered = training.copy()
        centered[training_labels == 82] -= mean_82
        centered[training_labels == 83] -= mean_83
        covariance = centered.T @ centered / (training.shape[0] - 2)
        shrinkage = max(0.1 * np.trace(covariance) / dimension, 1e-8)
        direction = np.linalg.solve(
            covariance + shrinkage * np.eye(dimension),
            mean_83 - mean_82,
        )
        expected.append(
            normalized[query_index] @ direction
            - 0.5 * (mean_82 + mean_83) @ direction
        )

    evidence = build_twin_reachability("primal-reference", descriptors, labels)

    np.testing.assert_allclose(evidence.lda_signed_scores, expected, rtol=1e-10, atol=1e-10)


def test_reachability_statistics_are_invariant_to_registered_row_permutation() -> None:
    rng = np.random.default_rng(71)
    descriptors = rng.standard_normal((40, 12)).astype(np.float64)
    descriptors[20:, 0] += 2.0
    labels = np.asarray([82] * 20 + [83] * 20, dtype=np.int64)
    permutation = rng.permutation(40)
    inverse = np.argsort(permutation)

    original = build_twin_reachability("trained-raw", descriptors, labels)
    permuted = build_twin_reachability(
        "trained-raw", descriptors[permutation], labels[permutation]
    )

    assert original.full_auc == permuted.full_auc
    assert original.lda_full_auc == permuted.lda_full_auc
    assert original.cue_present == permuted.cue_present
    np.testing.assert_allclose(
        original.signed_scores,
        np.asarray(permuted.signed_scores)[inverse],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        original.lda_signed_scores,
        np.asarray(permuted.lda_signed_scores)[inverse],
        rtol=1e-10,
        atol=1e-10,
    )


def test_conditional_high_evidence_mode_is_detected() -> None:
    rng = np.random.default_rng(11)
    strong_82 = np.tile(np.asarray([[1.0, 0.0]], dtype=np.float64), (10, 1))
    weak_82 = np.tile(np.asarray([[1.0, 1.0]], dtype=np.float64), (10, 1))
    strong_83 = np.tile(np.asarray([[0.0, 1.0]], dtype=np.float64), (10, 1))
    weak_83 = np.tile(np.asarray([[1.0, 1.0]], dtype=np.float64), (10, 1))
    strong_82 += 0.05 * rng.standard_normal(strong_82.shape)
    weak_82 += 0.05 * rng.standard_normal(weak_82.shape)
    strong_83 += 0.05 * rng.standard_normal(strong_83.shape)
    weak_83 += 0.05 * rng.standard_normal(weak_83.shape)
    descriptors = np.concatenate((strong_82, weak_82, strong_83, weak_83))
    labels = np.asarray([82] * 20 + [83] * 20, dtype=np.int64)

    evidence = build_twin_reachability("conditional", descriptors, labels)

    assert evidence.bic_improvement >= 10.0
    assert evidence.high_evidence_fraction >= 0.25
    assert evidence.high_evidence_auc >= 0.80
    assert evidence.cue_present is True


def test_canonical_result_roundtrips_and_rejects_mutations() -> None:
    descriptors, labels = _separable()
    evidence = build_twin_reachability("frozen-pooled", descriptors, labels)
    raw = canonical_twin_reachability_bytes(evidence)

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert validate_twin_reachability_bytes(raw, expected_plane="frozen-pooled") == evidence
    assert canonical_twin_reachability_bytes(evidence) == raw
    payload = json.loads(raw)
    mutations = []
    for mutate in (
        lambda value: value.update(schema="wrong"),
        lambda value: value.update(plane="trained-projected"),
        lambda value: value.update(claim_eligible=True),
        lambda value: value.update(source_count=5),
        lambda value: value.update(class_82_count=3),
        lambda value: value.update(class_83_count=3),
        lambda value: value.update(full_auc=0.5),
        lambda value: value.update(bic_one=value["bic_one"] + 1.0),
        lambda value: value.update(bic_two=value["bic_two"] + 1.0),
        lambda value: value.update(bic_improvement=value["bic_improvement"] + 1.0),
        lambda value: value.update(high_evidence_count=value["high_evidence_count"] + 1),
        lambda value: value.update(
            high_evidence_fraction=value["high_evidence_fraction"] + 0.1
        ),
        lambda value: value.update(high_evidence_auc=0.25),
        lambda value: value.update(centroid_cue_present=False),
        lambda value: value.update(lda_full_auc=0.5),
        lambda value: value["lda_signed_scores"].__setitem__(0, 200_000_000.0),
        lambda value: value.update(cue_present=False),
        lambda value: value["labels"].__setitem__(0, 83),
        lambda value: value["signed_scores"].__setitem__(0, 0.0),
    ):
        changed = copy.deepcopy(payload)
        mutate(changed)
        mutations.append(changed)
    for changed in mutations:
        changed_raw = (
            json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        with pytest.raises((TypeError, ValueError)):
            validate_twin_reachability_bytes(changed_raw, expected_plane="frozen-pooled")

    with pytest.raises((TypeError, ValueError)):
        canonical_twin_reachability_bytes(replace(evidence, full_auc=True))


def test_canonical_result_rejects_insufficient_class_evidence_before_derivation() -> None:
    descriptors, labels = _separable()
    payload = json.loads(
        canonical_twin_reachability_bytes(
            build_twin_reachability("frozen-pooled", descriptors, labels)
        )
    )
    payload.update(
        source_count=39,
        class_82_count=19,
        class_83_count=20,
        labels=[82] * 19 + [83] * 20,
        signed_scores=[-1.0] * 19 + [1.0] * 20,
        lda_signed_scores=[-1.0] * 19 + [1.0] * 20,
    )
    raw = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()

    with pytest.raises(ValueError, match="at least twenty"):
        validate_twin_reachability_bytes(raw, expected_plane="frozen-pooled")


def test_artifact_binds_exact_caller_authority_and_rejects_mutations() -> None:
    descriptors, labels = _separable()
    authority = _authority()
    evidence = build_twin_reachability("trained-raw", descriptors, labels)
    raw = canonical_twin_reachability_artifact_bytes(authority, evidence)

    assert validate_twin_reachability_artifact_bytes(raw, expected=authority) == evidence
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    payload = json.loads(raw)
    for field in (
        "source_revision",
        "plane",
        "source_tree_digest",
        "dataset_revision",
        "dataset_manifest_sha256",
        "model_name",
        "model_revision",
        "producer_kind",
        "producer_identity",
        "ordered_example_ids_sha256",
        "label_vector_sha256",
        "descriptor_sha256",
    ):
        changed = copy.deepcopy(payload)
        changed["authority"][field] = "wrong"
        changed_raw = (
            json.dumps(changed, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()
        with pytest.raises((TypeError, ValueError)):
            validate_twin_reachability_artifact_bytes(changed_raw, expected=authority)

    with pytest.raises(ValueError, match="authority"):
        validate_twin_reachability_artifact_bytes(
            raw,
            expected=replace(authority, descriptor_sha256="0" * 64),
        )
