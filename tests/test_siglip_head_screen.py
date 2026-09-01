from __future__ import annotations

import torch
from torch import nn

from sfora.siglip_head_screen import (
    CotangentRankEvidence,
    FeatureSplitAuthority,
    build_feature_split_authority,
    cosine_subclass_assignments,
    cotangent_rank_evidence,
    initialize_spectral_projection_,
    principal_angles_degrees,
    subclass_proxy_anchor_loss,
    uncentered_spectral_projection,
)


def _train_authority(features: torch.Tensor, *, digest: str = "1" * 64) -> FeatureSplitAuthority:
    return build_feature_split_authority(
        source_manifest_sha256=digest,
        role="optimization-train",
        official_test_access=False,
        ordered_example_ids=tuple(f"train-{row:04d}" for row in range(features.shape[0])),
        features=features,
    )


def test_uncentered_spectral_projection_is_orthonormal_and_sign_canonical() -> None:
    features = torch.tensor(
        [
            [-3.0, 0.0, 0.0],
            [-3.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float64,
    )

    weight = uncentered_spectral_projection(
        features, output_dimensions=2, split_authority=_train_authority(features)
    )

    assert weight.dtype == torch.float32
    assert weight.shape == (2, 3)
    torch.testing.assert_close(weight @ weight.T, torch.eye(2), rtol=0.0, atol=1.0e-6)
    torch.testing.assert_close(weight[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(weight[1], torch.tensor([0.0, 1.0, 0.0]))


def test_uncentered_spectral_projection_does_not_silently_center_features() -> None:
    features = torch.tensor([[10.0, -1.0], [10.0, 0.0], [10.0, 1.0]], dtype=torch.float64)

    weight = uncentered_spectral_projection(
        features, output_dimensions=1, split_authority=_train_authority(features)
    )

    torch.testing.assert_close(weight, torch.tensor([[1.0, 0.0]]), rtol=0.0, atol=1.0e-6)


def test_cotangent_rank_evidence_exposes_registered_control_bottleneck() -> None:
    evidence = cotangent_rank_evidence(
        class_count=49,
        logical_batch_size=120,
        embedding_dimensions=512,
        tower_dimensions=1152,
    )

    assert evidence == CotangentRankEvidence(
        class_count=49,
        logical_batch_size=120,
        embedding_dimensions=512,
        tower_dimensions=1152,
        maximum_per_example_cotangent_rank=50,
        maximum_projection_gradient_rank=120,
        per_example_tower_fraction=50 / 1152,
        projection_gradient_fraction=120 / 512,
    )


def test_head_screen_rejects_nonfinite_or_rank_impossible_inputs() -> None:
    for features, dimensions in (
        (torch.tensor([[float("nan"), 0.0]]), 1),
        (torch.ones((2, 3)), 3),
        (torch.ones((3, 2)), 0),
    ):
        try:
            uncentered_spectral_projection(
                features,
                output_dimensions=dimensions,
                split_authority=_train_authority(features),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid spectral projection input accepted")

    try:
        cotangent_rank_evidence(
            class_count=1,
            logical_batch_size=120,
            embedding_dimensions=512,
            tower_dimensions=1152,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid cotangent contract accepted")


def test_subclass_proxy_anchor_expands_readout_without_sibling_repulsion() -> None:
    scores = torch.tensor(
        [[0.4, 0.3, -0.2, -0.1], [-0.2, -0.3, 0.5, 0.4]],
        dtype=torch.float64,
        requires_grad=True,
    )
    original_labels = torch.tensor([0, 1], dtype=torch.int64)
    subclass_assignments = torch.tensor([0, 2], dtype=torch.int64)
    subclass_parents = torch.tensor([0, 0, 1, 1], dtype=torch.int64)

    loss = subclass_proxy_anchor_loss(
        scores,
        original_labels,
        subclass_assignments,
        subclass_parents,
        alpha=32.0,
        delta=0.1,
    )
    (gradient,) = torch.autograd.grad(loss, scores)

    assert torch.isfinite(loss)
    assert gradient[0, 0] < 0
    assert gradient[0, 1] == 0
    assert gradient[0, 2] > 0
    assert gradient[0, 3] > 0
    assert gradient[1, 0] > 0
    assert gradient[1, 1] > 0
    assert gradient[1, 2] < 0
    assert gradient[1, 3] == 0


def test_cosine_subclass_assignments_are_deterministic_and_class_local() -> None:
    features = torch.tensor(
        [
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
            [-1.0, 0.0],
            [-0.99, 0.01],
            [0.0, -1.0],
            [0.01, -0.99],
        ],
        dtype=torch.float64,
    )
    labels = torch.tensor([7, 7, 7, 7, 11, 11, 11, 11], dtype=torch.int64)

    first = cosine_subclass_assignments(
        features,
        labels,
        subclasses_per_class=2,
        master_seed_sha256="1" * 64,
        iterations=5,
        split_authority=_train_authority(features, digest="2" * 64),
    )
    second = cosine_subclass_assignments(
        features,
        labels,
        subclasses_per_class=2,
        master_seed_sha256="1" * 64,
        iterations=5,
        split_authority=_train_authority(features, digest="2" * 64),
    )
    different_procedure = cosine_subclass_assignments(
        features,
        labels,
        subclasses_per_class=2,
        master_seed_sha256="1" * 64,
        iterations=6,
        split_authority=_train_authority(features, digest="2" * 64),
    )

    assert torch.equal(first.assignments, second.assignments)
    assert first.sha256 == "bbb90f500183871359f89175de3da444f50f6f823d82c1b1752cec3f20a77832"
    assert second.sha256 == first.sha256
    assert different_procedure.sha256 != first.sha256
    assert first.parents == (7, 7, 11, 11)
    assert set(first.assignments[:4].tolist()) == {0, 1}
    assert set(first.assignments[4:].tolist()) == {2, 3}
    assert first.assignments[0] == first.assignments[1]
    assert first.assignments[2] == first.assignments[3]
    assert first.assignments[4] == first.assignments[5]
    assert first.assignments[6] == first.assignments[7]


def test_principal_angles_measure_proxy_span_rotation() -> None:
    reference = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    rotated = torch.tensor([[1.0, 0.0, 0.0]])

    angles = principal_angles_degrees(reference, rotated)

    torch.testing.assert_close(angles, torch.tensor([0.0]), atol=1.0e-4, rtol=0.0)


def test_spectral_initializer_replaces_only_projection_weight() -> None:
    projection = nn.Linear(3, 2, bias=False)
    features = torch.tensor(
        [[-3.0, 0.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
        dtype=torch.float64,
    )

    result = initialize_spectral_projection_(
        projection, features, split_authority=_train_authority(features)
    )

    assert result is projection
    assert projection.weight.requires_grad is True
    torch.testing.assert_close(
        projection.weight,
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        rtol=0.0,
        atol=1.0e-6,
    )

    low_precision = nn.Linear(3, 2, bias=False, dtype=torch.bfloat16)
    try:
        initialize_spectral_projection_(
            low_precision, features, split_authority=_train_authority(features)
        )
    except ValueError:
        pass
    else:
        raise AssertionError("low-precision spectral parameter accepted")


def test_data_dependent_head_screen_rejects_nontraining_authority() -> None:
    features = torch.eye(2)
    leaked = build_feature_split_authority(
        source_manifest_sha256="3" * 64,
        role="evaluation-gallery",
        official_test_access=True,
        ordered_example_ids=("evaluation-0", "evaluation-1"),
        features=features,
    )
    try:
        uncentered_spectral_projection(
            features,
            output_dimensions=1,
            split_authority=leaked,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("evaluation features accepted")

    training = _train_authority(features)
    mutated = features.clone()
    mutated[0, 0] = 0.5
    try:
        uncentered_spectral_projection(
            mutated,
            output_dimensions=1,
            split_authority=training,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("feature content drift accepted")


def test_head_screen_rejects_low_precision_features_and_float_labels() -> None:
    low_precision = torch.eye(4, dtype=torch.bfloat16)
    with torch.no_grad():
        try:
            uncentered_spectral_projection(
                low_precision,
                output_dimensions=2,
                split_authority=_train_authority(low_precision),
            )
        except ValueError:
            pass
        else:
            raise AssertionError("low-precision spectral features accepted")

    try:
        subclass_proxy_anchor_loss(
            torch.eye(2),
            torch.tensor([0.0, 1.0]),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
            alpha=32.0,
            delta=0.1,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("floating subclass labels accepted")


def test_principal_angles_reject_empty_span_and_measure_rotation() -> None:
    reference = torch.tensor([[1.0, 0.0]])
    rotated = torch.tensor([[1.0, 1.0]])
    torch.testing.assert_close(
        principal_angles_degrees(reference, rotated),
        torch.tensor([45.0]),
        atol=1.0e-4,
        rtol=0.0,
    )
    try:
        principal_angles_degrees(torch.eye(2), torch.zeros((0, 2)))
    except ValueError:
        pass
    else:
        raise AssertionError("empty principal-angle span accepted")


def test_subclass_assignment_rejects_degenerate_empty_clusters() -> None:
    features = torch.ones((4, 2), dtype=torch.float32)
    labels = torch.zeros(4, dtype=torch.int64)
    try:
        cosine_subclass_assignments(
            features,
            labels,
            subclasses_per_class=2,
            master_seed_sha256="4" * 64,
            iterations=3,
            split_authority=_train_authority(features),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("degenerate subclass assignments accepted")
