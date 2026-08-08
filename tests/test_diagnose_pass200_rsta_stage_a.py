"""Tiny pure tests for the preregistered Pass200 RSTA Stage-A core."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "diagnose_pass200_rsta_stage_a.py"
_SPEC = importlib.util.spec_from_file_location("diagnose_pass200_rsta_stage_a", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


_SELECTED_LABELS = [
    69,
    7,
    45,
    64,
    34,
    20,
    12,
    48,
    55,
    47,
    66,
    35,
    43,
    24,
    40,
    13,
    61,
    16,
    9,
    42,
    57,
    17,
    0,
    21,
    33,
    46,
    28,
    31,
    52,
    60,
    29,
    27,
    49,
    8,
    25,
    67,
    68,
    54,
    41,
    38,
    53,
    32,
    58,
    51,
    23,
    6,
    50,
    36,
    5,
    18,
    22,
    19,
    37,
    39,
    4,
    59,
    56,
    63,
    65,
    2,
    26,
    10,
    14,
    11,
]
_SELECTED_RECEIVERS = [
    "identity-69-image-2",
    "identity-07-image-1",
    "identity-45-image-2",
    "identity-64-image-2",
    "identity-34-image-0",
    "identity-20-image-0",
    "identity-12-image-2",
    "identity-48-image-0",
    "identity-55-image-2",
    "identity-47-image-0",
    "identity-66-image-1",
    "identity-35-image-0",
    "identity-43-image-1",
    "identity-24-image-1",
    "identity-40-image-2",
    "identity-13-image-1",
    "identity-61-image-2",
    "identity-16-image-1",
    "identity-09-image-2",
    "identity-42-image-1",
    "identity-57-image-1",
    "identity-17-image-1",
    "identity-00-image-2",
    "identity-21-image-1",
    "identity-33-image-1",
    "identity-46-image-1",
    "identity-28-image-1",
    "identity-31-image-2",
    "identity-52-image-2",
    "identity-60-image-2",
    "identity-29-image-0",
    "identity-27-image-1",
    "identity-49-image-1",
    "identity-08-image-0",
    "identity-25-image-1",
    "identity-67-image-0",
    "identity-68-image-1",
    "identity-54-image-0",
    "identity-41-image-1",
    "identity-38-image-2",
    "identity-53-image-2",
    "identity-32-image-1",
    "identity-58-image-2",
    "identity-51-image-1",
    "identity-23-image-0",
    "identity-06-image-0",
    "identity-50-image-1",
    "identity-36-image-2",
    "identity-05-image-2",
    "identity-18-image-2",
    "identity-22-image-0",
    "identity-19-image-0",
    "identity-37-image-0",
    "identity-39-image-1",
    "identity-04-image-2",
    "identity-59-image-2",
    "identity-56-image-1",
    "identity-63-image-1",
    "identity-65-image-2",
    "identity-02-image-1",
    "identity-26-image-2",
    "identity-10-image-1",
    "identity-14-image-1",
    "identity-11-image-0",
]


def _selection_fixture() -> tuple[list[str], list[int]]:
    example_ids: list[str] = []
    labels: list[int] = []
    for label in range(70):
        for rank in range(3):
            example_ids.append(f"identity-{label:02d}-image-{rank}")
            labels.append(label)
    for index in range(2_000):
        example_ids.append(f"distractor-{index:04d}")
        labels.append(1_000 + index)
    return example_ids, labels


def test_domain_hash_uses_nul_separator_and_raw_big_endian_seed() -> None:
    digest = _MODULE.domain_hash("rsta-stage-a-v1|identity|", "17")

    assert digest.hex() == "4c4b69f9d182a6582c54c0c57b07159d361da0e89b2dcc1f316e11a7ef2bab5b"
    assert (
        _MODULE.domain_seed("rsta-stage-a-v1|random-target|", "3\0identity-17-image-2")
        == 17_455_142_975_560_980_662
    )


def test_selection_uses_canonical_identity_and_role_order() -> None:
    example_ids, labels = _selection_fixture()

    panel = _MODULE.select_primary_panel(example_ids, labels)

    assert panel["labels"] == _SELECTED_LABELS
    assert panel["receiver_ids"] == _SELECTED_RECEIVERS
    assert panel["support_ids_by_label"][17] == [
        "identity-17-image-2",
        "identity-17-image-0",
    ]


def test_selection_builds_eight_exact_disjoint_support_free_batches() -> None:
    example_ids, labels = _selection_fixture()

    panel = _MODULE.select_primary_panel(example_ids, labels)

    assert len(panel["groups"]) == 8
    assert all(len(group) == 8 for group in panel["groups"])
    assert len(panel["batches"]) == 8
    assert all(len(batch) == 180 for batch in panel["batches"])
    assert all(len(block) == 172 for block in panel["distractor_blocks"])
    flat_distractors = [value for block in panel["distractor_blocks"] for value in block]
    assert len(set(flat_distractors)) == 8 * 172
    selected_prefixes = {f"identity-{label:02d}-" for label in _SELECTED_LABELS}
    assert not any(
        any(value.startswith(prefix) for prefix in selected_prefixes) for value in flat_distractors
    )
    selected_supports = {
        value
        for label, values in panel["support_ids_by_label"].items()
        if label in _SELECTED_LABELS
        for value in values
    }
    assert selected_supports.isdisjoint(flat_distractors)
    assert [
        hashlib.sha256("\n".join(batch).encode("utf-8")).hexdigest() for batch in panel["batches"]
    ] == [
        "5d5857b8297cb1710e32fb391c8b116bf903b1174bda87517798ab5ebae435ac",
        "bfeeb106fe83bd4845ac886a99bb88f2124d7c9edf3d194ea0da84e4cd420f55",
        "f7003ce532d508daa60d009869bdfd1ffc5398f119d2eaadb7dd40aa0ec0496c",
        "96afdf22d3138e3022c63b15edee4814eb4b431fc818b561b5c818e1c8801567",
        "c0052dcfa17ee6c71d10ecaf8b24d69a75d47c661e8c352993fdd88b510dff82",
        "4223ba5b669d91edf05dd638a37b68d1e5abd6c6275e6491f7cc396b9fa46e07",
        "60b271fe2a9b3192142f8fff55ba132f672b7df118ff7b4bc2ee3fd193a3a9c1",
        "898a472c2b237d7cb7ccc64971871ce812cd63940d61e6c3c32acb00a612db0c",
    ]


def test_selection_alternate_regroups_positions_and_excludes_frozen_rows() -> None:
    example_ids, labels = _selection_fixture()
    primary = _MODULE.select_primary_panel(example_ids, labels)

    alternate = _MODULE.select_alternate_panel(example_ids, labels, primary)

    assert alternate["labels"] == [69, 7, 55, 47, 61, 16, 33, 46, 49, 8, 53, 32, 5, 18, 56, 63]
    assert alternate["receiver_ids"] == [
        _SELECTED_RECEIVERS[index] for index in range(64) if index % 8 in (0, 1)
    ]
    assert [len(group) for group in alternate["groups"]] == [8, 8]
    assert all(len(batch) == 180 for batch in alternate["batches"])
    alternate_distractors = {value for block in alternate["distractor_blocks"] for value in block}
    assert len(alternate_distractors) == 344
    assert alternate_distractors.isdisjoint(
        value for block in primary["distractor_blocks"] for value in block
    )
    all_eligible_supports = {
        value for values in primary["support_ids_by_label"].values() for value in values
    }
    assert alternate_distractors.isdisjoint(all_eligible_supports)
    assert [
        hashlib.sha256("\n".join(block).encode("utf-8")).hexdigest()
        for block in alternate["distractor_blocks"]
    ] == [
        "f923b403bc729b0d225396c00ad0d9539fb77e6f8525e5d6629a0cebcf53e2da",
        "8ba9877f24a7b784689c867f4561054cc062bd5035d4a6072804192d1f0d938a",
    ]
    assert (
        hashlib.sha256(
            "\0".join(value for block in alternate["distractor_blocks"] for value in block).encode(
                "utf-8"
            )
        ).hexdigest()
        == "9489780df65bd81b031f4f4daaee96a0d71c35ee0b7b470e7db476121d42bb10"
    )


@pytest.mark.parametrize("defect", ["identity_order", "primary_distractor"])
def test_selection_alternate_rejects_forged_primary_panel(defect: str) -> None:
    example_ids, labels = _selection_fixture()
    primary = _MODULE.select_primary_panel(example_ids, labels)
    forged = deepcopy(primary)
    if defect == "identity_order":
        forged["labels"].reverse()
        forged["receiver_ids"].reverse()
    else:
        used = {value for block in forged["distractor_blocks"] for value in block}
        replacement = next(value for value in example_ids if value not in used)
        forged["distractor_blocks"][0][0] = replacement

    with pytest.raises(ValueError, match="canonical primary"):
        _MODULE.select_alternate_panel(example_ids, labels, forged)


@pytest.mark.parametrize("defect", ["duplicate", "too_few_identities", "too_few_distractors"])
def test_selection_fails_closed_instead_of_replacing_rows(defect: str) -> None:
    example_ids, labels = _selection_fixture()
    if defect == "duplicate":
        example_ids[-1] = example_ids[-2]
    elif defect == "too_few_identities":
        labels = [0] * len(labels)
    else:
        example_ids = example_ids[:500]
        labels = labels[:500]

    with pytest.raises(ValueError):
        _MODULE.select_primary_panel(example_ids, labels)


def _unit_rows(rows: np.ndarray) -> np.ndarray:
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def test_tangent_projection_removes_radial_component_and_rejects_zero() -> None:
    receiver = np.asarray([0.6, 0.8, 0.0])

    projected = _MODULE.tangent_projection(np.asarray([3.0, 4.0, 5.0]), receiver)

    np.testing.assert_allclose(projected, np.asarray([0.0, 0.0, 5.0]), atol=1e-12)
    with pytest.raises(ValueError, match="zero"):
        _MODULE.tangent_projection(np.asarray([3.0, 4.0, 0.0]), receiver)


def test_unit_validation_rejects_absolute_norm_error_above_two_e_minus_five() -> None:
    with pytest.raises(ValueError, match="unit"):
        _MODULE.tangent_projection(np.asarray([0.0, 1.0]), np.asarray([1.00003, 0.0]))
    receivers = np.tile(np.asarray([[1.0, 0.0, 0.0]]), (8, 1))
    receivers[0, 0] = 1.00003
    targets = np.tile(np.asarray([[0.0, 1.0, 0.0]]), (8, 1))
    with pytest.raises(ValueError, match="unit"):
        _MODULE.deranged_tangent_targets(receivers, targets)


def test_smooth_margin_gradient_matches_independent_central_difference() -> None:
    receiver = _unit_rows(np.asarray([[1.0, 0.3, -0.2, 0.1]]))[0]
    positives = _unit_rows(np.asarray([[0.9, 0.2, 0.1, -0.1], [0.7, 0.4, -0.1, 0.2]]))
    angles = np.arange(1, 33, dtype=np.float64)
    foreign = _unit_rows(
        np.column_stack(
            (
                np.sin(angles),
                np.cos(angles),
                ((angles % 5.0) - 2.0) / 3.0,
                ((angles % 7.0) - 3.0) / 4.0,
            )
        )
    )

    actual = _MODULE.smooth_margin_gradient(receiver, positives, foreign, tau=0.05)

    def margin(value: np.ndarray) -> float:
        positive_logits = positives @ value / 0.05
        foreign_logits = foreign @ value / 0.05
        positive_lme = np.log(np.exp(positive_logits - positive_logits.max()).mean())
        positive_lme += positive_logits.max()
        foreign_lme = np.log(np.exp(foreign_logits - foreign_logits.max()).mean())
        foreign_lme += foreign_logits.max()
        return 0.05 * (positive_lme - foreign_lme)

    epsilon = 1.0e-6
    ambient = np.asarray(
        [
            (
                margin(receiver + epsilon * np.eye(4)[index])
                - margin(receiver - epsilon * np.eye(4)[index])
            )
            / (2.0 * epsilon)
            for index in range(4)
        ]
    )
    expected = ambient - receiver * float(np.dot(receiver, ambient))
    np.testing.assert_allclose(actual, expected, atol=2e-9, rtol=2e-9)
    assert float(np.dot(actual, receiver)) == pytest.approx(0.0, abs=1e-12)


def test_smooth_margin_requires_exact_frozen_support_counts() -> None:
    receiver = np.asarray([1.0, 0.0])
    with pytest.raises(ValueError, match="two positive"):
        _MODULE.smooth_margin_gradient(receiver, np.asarray([[0.0, 1.0]]), np.eye(2), tau=0.05)
    with pytest.raises(ValueError, match="32 foreign"):
        _MODULE.smooth_margin_gradient(
            receiver,
            np.asarray([[0.0, 1.0], [0.0, -1.0]]),
            np.tile(np.asarray([[0.0, 1.0]]), (31, 1)),
            tau=0.05,
        )


def test_foreign_support_selection_excludes_batch_and_breaks_cutoff_tie_by_role_hash() -> None:
    receiver = np.asarray([1.0, 0.0])
    support_ids = [f"support-{index}" for index in range(34)]
    support_labels = list(range(1, 35))
    cosines = np.linspace(0.99, 0.20, 34)
    cosines[32:] = 0.10
    supports = np.column_stack((cosines, np.sqrt(1.0 - cosines**2)))

    selected_ids, selected = _MODULE.select_foreign_supports(
        receiver,
        receiver_label=0,
        support_ids=support_ids,
        support_labels=support_labels,
        support_descriptors=supports,
        current_batch_ids={"support-0"},
    )

    assert len(selected_ids) == 32
    assert "support-0" not in selected_ids
    assert "support-33" in selected_ids
    assert "support-32" not in selected_ids
    np.testing.assert_allclose(
        selected[:, 0], supports[[support_ids.index(x) for x in selected_ids], 0]
    )


def test_deranged_targets_shift_forward_then_reproject_to_each_receiver() -> None:
    receivers = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    targets = np.asarray(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, -1.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, -1.0],
            [1.0, 1.0, 0.0],
        ]
    )

    deranged = _MODULE.deranged_tangent_targets(receivers, targets)

    np.testing.assert_allclose(deranged[0], np.asarray([0.0, 0.0, 1.0]), atol=1e-12)
    np.testing.assert_allclose(deranged[7], np.asarray([0.0, 1.0, 0.0]), atol=1e-12)
    np.testing.assert_allclose(np.linalg.norm(deranged, axis=1), np.ones(8), atol=1e-12)
    np.testing.assert_allclose(np.sum(deranged * receivers, axis=1), np.zeros(8), atol=1e-12)


def test_random_tangent_target_uses_fresh_domain_seed_and_norm_matches() -> None:
    target = _MODULE.random_tangent_target(
        np.asarray([0.6, 0.8, 0.0]), seed=3, example_id="receiver-17", target_norm=2.5
    )

    np.testing.assert_allclose(
        target,
        np.asarray([0.3611577494748494, -0.2708683121061372, -2.458901266316124]),
        atol=1e-14,
        rtol=1e-14,
    )
    assert np.linalg.norm(target) == pytest.approx(2.5, abs=1e-12)


def test_head_only_kernel_includes_cross_rows_and_has_positive_self_collinearity() -> None:
    prehead = np.asarray([[1.0, 0.0], [1.0, 2.0]])
    head_outputs = np.asarray([[2.0, 0.0, 0.0], [3.0, 3.0, 0.0]])
    cotangents = np.asarray([[0.0, 2.0, 0.0], [3.0, -3.0, 0.0]])

    batch_motion, self_motion = _MODULE.head_only_kernel_motion(prehead, head_outputs, cotangents)

    np.testing.assert_allclose(
        batch_motion,
        np.asarray(
            [
                [0.0, 0.2928932188134524, 0.0],
                [0.7642977396044841, -0.7642977396044842, 0.0],
            ]
        ),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        self_motion, np.asarray([[0.0, 1.0, 0.0], [1.0, -1.0, 0.0]]), atol=1e-12
    )
    assert _MODULE.cosine_similarity(self_motion[0], cotangents[0]) == pytest.approx(1.0)
    assert _MODULE.cosine_similarity(self_motion[1], cotangents[1]) == pytest.approx(1.0)


def test_head_only_kernel_preserves_registered_model_arithmetic_dtype() -> None:
    batch_motion, self_motion = _MODULE.head_only_kernel_motion(
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[2.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 1.0]], dtype=np.float32),
    )

    assert batch_motion.dtype == np.float32
    assert self_motion.dtype == np.float32


def test_cosine_rejects_either_tiny_norm_without_rejecting_small_valid_pair() -> None:
    with pytest.raises(ValueError, match="nonzero"):
        _MODULE.cosine_similarity(np.asarray([1.0e-13, 0.0]), np.asarray([1.0e13, 0.0]))

    assert _MODULE.cosine_similarity(
        np.asarray([1.0e-8, 0.0]), np.asarray([1.0e-8, 0.0])
    ) == pytest.approx(1.0)


def test_rotation_construction_fixes_qr_sign_and_checker_rejects_bad_rotation() -> None:
    rotation = _MODULE.construct_rotation(4)

    np.testing.assert_allclose(
        rotation,
        np.asarray(
            [
                [
                    0.22074921927657543,
                    -0.5386053982113567,
                    -0.7400458490721568,
                    -0.33690673557742834,
                ],
                [0.7121696678136576, 0.14669404556946844, 0.3691693505574429, -0.5787998029108538],
                [
                    -0.5454996011474486,
                    -0.49412392472309014,
                    0.4335827274294307,
                    -0.5198824392486062,
                ],
                [0.3827719054253348, -0.6665032850844014, 0.3578437513146246, 0.5302894389235343],
            ]
        ),
        atol=1e-14,
        rtol=1e-14,
    )
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(4), atol=1e-14)
    vectors = {name: np.asarray([1.0, 2.0, 3.0, 4.0]) for name in ("z", "dbar", "b", "s", "q")}
    statistics = {
        name: 0.2
        for name in ("A_self", "A_batch", "Delta", "A_desc", "rho", "log_ratio", "cos_b_s")
    }
    checked = _MODULE.check_rotation(
        vectors,
        {name: rotation @ value for name, value in vectors.items()},
        statistics,
        {name: value + 0.0001 for name, value in statistics.items()},
        rotation,
    )
    assert checked["vector_residuals"]["z"] < 1e-14
    assert checked["statistic_differences"]["Delta"] == pytest.approx(0.0001)
    bad_vectors = {name: rotation @ value for name, value in vectors.items()}
    bad_vectors["z"] = bad_vectors["z"] + 0.01
    with pytest.raises(ValueError, match="rotation"):
        _MODULE.check_rotation(
            vectors,
            bad_vectors,
            statistics,
            statistics,
            rotation,
        )


def test_rotation_checker_accepts_registered_model_dtype_cast() -> None:
    rotation = _MODULE.construct_rotation(16, dtype=np.float32)
    vector = np.arange(1.0, 17.0, dtype=np.float32)
    vectors = {name: vector for name in ("z", "dbar", "b", "s", "q")}
    statistics = {
        name: 0.25
        for name in ("A_self", "A_batch", "Delta", "A_desc", "rho", "log_ratio", "cos_b_s")
    }

    checked = _MODULE.check_rotation(
        vectors,
        {name: rotation @ value for name, value in vectors.items()},
        statistics,
        statistics,
        rotation,
    )

    assert checked["vector_residuals"]["z"] <= 5e-4


def test_rotation_checker_requires_every_named_nonzero_gate_value() -> None:
    rotation = np.eye(3)

    with pytest.raises(ValueError, match="registered names"):
        _MODULE.check_rotation({}, {}, {}, {}, rotation)


_ROTATION_VECTOR_NAMES = ("z", "dbar", "b", "s", "q")
_ROTATION_STATISTIC_NAMES = (
    "A_self",
    "A_batch",
    "Delta",
    "A_desc",
    "rho",
    "log_ratio",
    "cos_b_s",
)


def _rotation_gate_fixture() -> tuple[
    np.ndarray,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, float],
    dict[str, float],
]:
    rotation = np.eye(3)
    vectors = {name: np.asarray([1.0, 0.0, 0.0]) for name in _ROTATION_VECTOR_NAMES}
    rotated_vectors = {name: value.copy() for name, value in vectors.items()}
    statistics = {name: 0.25 for name in _ROTATION_STATISTIC_NAMES}
    return rotation, vectors, rotated_vectors, statistics, dict(statistics)


def test_rotation_checker_rejects_errors_just_above_registered_thresholds() -> None:
    rotation, vectors, rotated_vectors, statistics, rotated_statistics = _rotation_gate_fixture()
    rotated_vectors["z"] = np.asarray([1.0, 5.00001e-4, 0.0])
    with pytest.raises(ValueError, match="vector gate"):
        _MODULE.check_rotation(vectors, rotated_vectors, statistics, rotated_statistics, rotation)

    rotation, vectors, rotated_vectors, statistics, rotated_statistics = _rotation_gate_fixture()
    rotated_statistics["Delta"] += 2.00001e-4
    with pytest.raises(ValueError, match="statistic gate"):
        _MODULE.check_rotation(vectors, rotated_vectors, statistics, rotated_statistics, rotation)


@pytest.mark.parametrize(
    ("kind", "name"),
    [
        *(("vector", name) for name in _ROTATION_VECTOR_NAMES),
        *(("statistic", name) for name in _ROTATION_STATISTIC_NAMES),
    ],
)
def test_rotation_checker_rejects_each_missing_registered_name(kind: str, name: str) -> None:
    rotation, vectors, rotated_vectors, statistics, rotated_statistics = _rotation_gate_fixture()
    if kind == "vector":
        vectors.pop(name)
        rotated_vectors.pop(name)
    else:
        statistics.pop(name)
        rotated_statistics.pop(name)

    with pytest.raises(ValueError, match="registered names"):
        _MODULE.check_rotation(vectors, rotated_vectors, statistics, rotated_statistics, rotation)


@pytest.mark.parametrize("name", _ROTATION_VECTOR_NAMES)
def test_rotation_checker_rejects_each_named_zero_vector(name: str) -> None:
    rotation, vectors, rotated_vectors, statistics, rotated_statistics = _rotation_gate_fixture()
    vectors[name] = np.zeros(3)
    rotated_vectors[name] = np.zeros(3)

    with pytest.raises(ValueError, match="nonzero"):
        _MODULE.check_rotation(vectors, rotated_vectors, statistics, rotated_statistics, rotation)


def _verdict_rows(
    *,
    delta_by_seed: tuple[float, float, float, float] = (0.04, 0.035, 0.03, 0.02),
    self_desc: float = 0.02,
    rho: float = 0.25,
    log_ratio: float = float(np.log(1.2)),
    deranged_delta: float = 0.005,
) -> list[dict[str, float | int]]:
    return [
        {
            "seed": seed,
            "label": label,
            "delta": delta_by_seed[seed],
            "self_minus_desc": self_desc,
            "rho": rho,
            "log_ratio": log_ratio,
            "deranged_delta": deranged_delta,
        }
        for seed in range(4)
        for label in range(64)
    ]


def _alternate_rows(
    delta_by_seed: tuple[float, float, float, float] = (0.02, 0.01, 0.015, 0.005),
) -> list[dict[str, float | int]]:
    return [
        {"seed": seed, "label": label, "delta": delta_by_seed[seed]}
        for seed in range(4)
        for label in (7, 45, 35, 43, 42, 57, 31, 52, 54, 41, 6, 50, 39, 4, 14, 11)
    ]


def test_joint_bootstrap_reuses_each_identity_draw_across_all_seeds() -> None:
    values = np.asarray(
        [
            [1.0, 2.0, 9.0],
            [10.0, 20.0, 90.0],
            [100.0, 200.0, 900.0],
            [1_000.0, 2_000.0, 9_000.0],
        ]
    )

    distribution = _MODULE.joint_bootstrap(values, replicates=8)

    np.testing.assert_allclose(
        distribution,
        np.asarray(
            [
                370.3333333333333,
                462.9166666666667,
                1_111.0,
                555.5,
                555.5,
                1_111.0,
                1_203.5833333333333,
                1_111.0,
            ]
        ),
        atol=0.0,
        rtol=0.0,
    )
    assert _MODULE.float64_c_order_sha256(distribution) == (
        "45ca5cf79ea2c8fa48a44680e57be67b67e4b65e1e0d448d7f0d6c74226a7e9b"
    )


def test_decide_stage_a_passes_at_inclusive_registered_thresholds() -> None:
    rows = _verdict_rows(
        delta_by_seed=(0.03, 0.03, 0.03, 0.03),
        self_desc=0.001,
        rho=0.20,
        log_ratio=float(np.log(1.10)),
        deranged_delta=0.01,
    )
    alternate = _alternate_rows((0.001, 0.001, 0.001, 0.001))

    result = _MODULE.decide_stage_a(rows, alternate)

    assert result["stage_a"] == "PASS_ONWARD"
    assert result["first_decisive_clause"] == "all_pass_requirements"
    assert result["complete_identity_count"] == 64
    assert result["alternate_identity_count"] == 16
    assert result["pooled_delta"] == pytest.approx(0.03)
    assert result["pooled_median_rho"] == pytest.approx(0.20)
    assert result["pooled_median_abs_log_ratio"] == pytest.approx(np.log(1.10))
    assert len(result["bootstrap_delta_sha256"]) == 64
    assert len(result["bootstrap_self_desc_sha256"]) == 64
    assert all(result["criteria"].values())


@pytest.mark.parametrize(
    ("field", "high_value", "criterion"),
    [
        ("delta", 3.0, "bootstrap_delta_lower_positive"),
        (
            "self_minus_desc",
            1.0,
            "bootstrap_self_minus_desc_lower_positive",
        ),
    ],
)
def test_decide_stage_a_requires_each_bootstrap_lower_bound_independently(
    field: str, high_value: float, criterion: str
) -> None:
    rows = _verdict_rows(delta_by_seed=(0.04, 0.04, 0.04, 0.04), self_desc=0.02)
    for row in rows:
        row[field] = high_value if int(row["label"]) == 7 else -0.01

    result = _MODULE.decide_stage_a(rows, _alternate_rows())

    assert result["stage_a"] == "UNRESOLVED"
    assert result["criteria"][criterion] is False
    assert all(value for name, value in result["criteria"].items() if name != criterion)


def test_decide_stage_a_bootstrap_uses_canonical_identity_order() -> None:
    rows = _verdict_rows()
    for row in rows:
        row["delta"] = 0.01 + int(row["label"]) / 10_000.0
    rows.reverse()

    result = _MODULE.decide_stage_a(rows, _alternate_rows())

    assert result["bootstrap_delta_sha256"] == (
        "6636370af0752a7dd99a9195517a96bb435987cad3b55dfa9a3952239d919185"
    )


def test_decide_stage_a_rejects_nonregistered_alternate_subset() -> None:
    alternate = _alternate_rows()
    for row in alternate:
        if row["label"] == 7:
            row["label"] = 0

    with pytest.raises(ValueError, match="registered alternate"):
        _MODULE.decide_stage_a(_verdict_rows(), alternate)


@pytest.mark.parametrize(
    ("primary", "alternate", "clause"),
    [
        (
            _verdict_rows(delta_by_seed=(0.0, 0.0, 0.0, 0.0)),
            _alternate_rows(),
            "pooled_delta_nonpositive",
        ),
        (
            _verdict_rows(delta_by_seed=(-0.01, -0.01, -0.01, 0.20)),
            _alternate_rows(),
            "three_primary_seed_means_nonpositive",
        ),
        (_verdict_rows(rho=0.099), _alternate_rows(), "median_rho_below_0_10"),
        (
            _verdict_rows(),
            _alternate_rows((0.0, 0.0, 0.0, 0.0)),
            "alternate_pooled_delta_nonpositive",
        ),
        (
            _verdict_rows(),
            _alternate_rows((-0.01, -0.01, -0.01, 0.10)),
            "three_alternate_seed_means_nonpositive",
        ),
    ],
)
def test_decide_stage_a_fail_clauses_take_precedence_in_registered_order(
    primary: list[dict[str, float | int]],
    alternate: list[dict[str, float | int]],
    clause: str,
) -> None:
    result = _MODULE.decide_stage_a(primary, alternate)

    assert result["stage_a"] == "FAIL"
    assert result["first_decisive_clause"] == clause


@pytest.mark.parametrize(
    ("primary", "alternate", "clause"),
    [
        (
            _verdict_rows(delta_by_seed=(0.0, 0.0, 0.0, 0.0), rho=0.05),
            _alternate_rows((0.0, 0.0, 0.0, 0.0)),
            "pooled_delta_nonpositive",
        ),
        (
            _verdict_rows(delta_by_seed=(-0.01, -0.01, -0.01, 0.20), rho=0.05),
            _alternate_rows((0.0, 0.0, 0.0, 0.0)),
            "three_primary_seed_means_nonpositive",
        ),
        (
            _verdict_rows(rho=0.05),
            _alternate_rows((0.0, 0.0, 0.0, 0.0)),
            "median_rho_below_0_10",
        ),
        (
            _verdict_rows(),
            _alternate_rows((0.0, 0.0, 0.0, 0.0)),
            "alternate_pooled_delta_nonpositive",
        ),
        (
            _verdict_rows(),
            _alternate_rows((-0.01, -0.01, -0.01, 0.10)),
            "three_alternate_seed_means_nonpositive",
        ),
    ],
)
def test_decide_stage_a_uses_first_triggered_clause_when_later_failures_also_hold(
    primary: list[dict[str, float | int]],
    alternate: list[dict[str, float | int]],
    clause: str,
) -> None:
    result = _MODULE.decide_stage_a(primary, alternate)

    assert result["stage_a"] == "FAIL"
    assert result["first_decisive_clause"] == clause


@pytest.mark.parametrize(
    ("primary", "alternate"),
    [
        (_verdict_rows(delta_by_seed=(0.029999, 0.029999, 0.029999, 0.029999)), _alternate_rows()),
        (_verdict_rows(delta_by_seed=(0.05, 0.05, 0.019, 0.019)), _alternate_rows()),
        (_verdict_rows(self_desc=0.0), _alternate_rows()),
        (_verdict_rows(rho=0.10), _alternate_rows()),
        (_verdict_rows(log_ratio=float(np.log(1.10)) - 1e-9), _alternate_rows()),
        (_verdict_rows(deranged_delta=0.010001), _alternate_rows()),
        (_verdict_rows(), _alternate_rows((0.10, 0.10, 0.0, 0.0))),
    ],
)
def test_decide_stage_a_keeps_registered_gap_boundaries_unresolved(
    primary: list[dict[str, float | int]],
    alternate: list[dict[str, float | int]],
) -> None:
    result = _MODULE.decide_stage_a(primary, alternate)

    assert result["stage_a"] == "UNRESOLVED"
    assert result["first_decisive_clause"] == "no_pass_or_fail_rule"


@pytest.mark.parametrize("panel", ["primary", "alternate"])
def test_decide_stage_a_requires_all_64_and_all_16_rows_in_every_seed(panel: str) -> None:
    primary = _verdict_rows()
    alternate = _alternate_rows()
    if panel == "primary":
        primary.pop()
    else:
        alternate.pop()

    with pytest.raises(ValueError, match="complete"):
        _MODULE.decide_stage_a(primary, alternate)
