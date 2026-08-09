"""Tiny pure tests for the preregistered Pass200 RSTA Stage-A core."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import random
import subprocess
import sys
import weakref
from collections.abc import Callable
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

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
    with pytest.raises(ValueError, match="nonzero"):
        _MODULE.cosine_similarity(np.asarray([1.0e13, 0.0]), np.asarray([1.0e-13, 0.0]))

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


def test_rotation_checker_accepts_errors_exactly_at_registered_thresholds() -> None:
    rotation, vectors, rotated_vectors, statistics, rotated_statistics = _rotation_gate_fixture()
    rotated_vectors["z"] = np.asarray([1.0, 5.0e-4, 0.0])
    statistics["Delta"] = 0.0
    rotated_statistics["Delta"] = 2.0e-4

    checked = _MODULE.check_rotation(
        vectors, rotated_vectors, statistics, rotated_statistics, rotation
    )

    assert checked["vector_residuals"]["z"] == pytest.approx(5.0e-4)
    assert checked["statistic_differences"]["Delta"] == pytest.approx(2.0e-4)


@pytest.mark.parametrize(
    ("kind", "name", "side"),
    [
        *(
            ("vector", name, side)
            for name in _ROTATION_VECTOR_NAMES
            for side in ("original", "rotated")
        ),
        *(
            ("statistic", name, side)
            for name in _ROTATION_STATISTIC_NAMES
            for side in ("original", "rotated")
        ),
    ],
)
def test_rotation_checker_rejects_each_missing_registered_name(
    kind: str, name: str, side: str
) -> None:
    rotation, vectors, rotated_vectors, statistics, rotated_statistics = _rotation_gate_fixture()
    if kind == "vector":
        target = vectors if side == "original" else rotated_vectors
    else:
        target = statistics if side == "original" else rotated_statistics
    target.pop(name)

    with pytest.raises(ValueError, match="registered names"):
        _MODULE.check_rotation(vectors, rotated_vectors, statistics, rotated_statistics, rotation)


@pytest.mark.parametrize(
    ("name", "side"),
    [(name, side) for name in _ROTATION_VECTOR_NAMES for side in ("original", "rotated")],
)
def test_rotation_checker_rejects_each_named_zero_vector(name: str, side: str) -> None:
    rotation, vectors, rotated_vectors, statistics, rotated_statistics = _rotation_gate_fixture()
    target = vectors if side == "original" else rotated_vectors
    target[name] = np.zeros(3)

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
            "a_self": 0.4,
            "a_batch": 0.4 - delta_by_seed[seed],
            "a_desc": 0.4 - self_desc,
            "cos_b_s": 0.7,
            "random_a_self": 0.03,
            "random_a_batch": 0.02,
            "random_delta": 0.01,
            "head_a_self": 0.2,
            "head_a_batch": 0.1,
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


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_rsta_final_pack(
    path: Path,
    *,
    split: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    example_ids: list[str],
    source_paths: list[str],
    checkpoint_sha256: str,
    report_sha256: str,
) -> None:
    np.savez_compressed(
        path,
        embeddings=np.asarray(embeddings, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        example_ids=np.asarray(example_ids),
        source_paths=np.asarray(source_paths),
        artifact_selection=np.asarray("final_training_state"),
        split=np.asarray(split),
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        report_sha256=np.asarray(report_sha256),
    )


def _synthetic_rsta_bundle(
    root: Path,
    *,
    seed: int = 0,
    source_root: Path | None = None,
    batch_size: int = 180,
    proxy_dimension: int = 2,
    reported_r1: float = 1.0,
    embedded_checkpoint_digest: str | None = None,
    evaluation_model_source: str = "student",
) -> tuple[
    dict[str, dict[str, str]],
    Callable[..., dict[str, dict[str, np.ndarray]]],
]:
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "dataset_name": "inshop",
        "objectives": ["proxy_anchor"],
        "seed": seed,
        "proxy_anchor_alpha": 32.0,
        "proxy_anchor_delta": 0.1,
        "checkpoint_selection_interval": 0,
        "backbone_name": "bn_inception",
        "head_pooling": "avg_max",
        "batch_size": batch_size,
        "drop_last_train_batch": True,
        "freeze_batch_norm": False,
        "freeze_batch_norm_affine": False,
        "embedding_dimensions": 2,
    }
    prehead = {
        "train": np.asarray(
            [[1.0, 0.0], [0.9, 0.1], [0.8, 0.2], [0.0, 1.0], [0.1, 0.9], [0.2, 0.8]],
            dtype=np.float32,
        ),
        "query": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        "gallery": np.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=np.float32),
    }
    labels = {
        "train": np.asarray([10, 10, 10, 20, 20, 20], dtype=np.int64),
        "query": np.asarray([30, 40], dtype=np.int64),
        "gallery": np.asarray([30, 40], dtype=np.int64),
    }
    ids = {
        split: [f"{split}-{index}" for index in range(len(rows))] for split, rows in prehead.items()
    }
    sources: dict[str, list[str]] = {}
    image_root = root / "images" if source_root is None else source_root
    for split, split_ids in ids.items():
        sources[split] = []
        for example_id in split_ids:
            source = image_root / split / f"{example_id}.jpg"
            source.parent.mkdir(parents=True, exist_ok=True)
            source.write_bytes(example_id.encode("ascii"))
            sources[split].append(str(source.resolve()))

    def normalize(rows: np.ndarray) -> np.ndarray:
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    embeddings = {split: normalize(rows) for split, rows in prehead.items()}
    report = {
        "config": config,
        "methods": {
            "proxy_anchor_end_to_end:bn_inception": {
                "dimensions": 2,
                "recall_at_1": reported_r1,
            }
        },
    }
    report_path = root / "report.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    checkpoint_path = root / "checkpoint.pt"
    torch.save(
        {
            "state_dict": {
                "model.embedding.weight": torch.eye(2),
                "model.embedding.bias": torch.zeros(2),
                "metric_proxies": torch.eye(2, proxy_dimension),
                "metric_proxy_labels": torch.tensor([10, 20]),
            },
            "artifact_selection": "final_training_state",
            "evaluation_model_source": evaluation_model_source,
            "training_config": config,
            "training_step": 10,
        },
        checkpoint_path,
    )
    checkpoint_digest = _sha256_file(checkpoint_path)
    report_digest = _sha256_file(report_path)
    np.savez_compressed(
        root / "prehead.npz",
        train=prehead["train"],
        train_labels=labels["train"],
        query=prehead["query"],
        query_labels=labels["query"],
        gallery=prehead["gallery"],
        gallery_labels=labels["gallery"],
    )
    bound_checkpoint_digest = embedded_checkpoint_digest or checkpoint_digest
    for split in ("train", "query", "gallery"):
        _write_rsta_final_pack(
            root / f"{split}.npz",
            split=split,
            embeddings=embeddings[split],
            labels=labels[split],
            example_ids=ids[split],
            source_paths=sources[split],
            checkpoint_sha256=bound_checkpoint_digest,
            report_sha256=report_digest,
        )
    retrieval_path = root / "retrieval.json"
    retrieval_path.write_text(
        json.dumps(
            {
                "artifact_selection": "final_training_state",
                "checkpoint_sha256": checkpoint_digest,
                "report_sha256": report_digest,
                "resolved_training_steps": 10,
                "reported_final_recall_at_1": reported_r1,
                "independent_recall_at_1": 1.0,
                "canonical_float64_euclidean_recall_at_1": 1.0,
            }
        ),
        encoding="utf-8",
    )
    paths = {
        "prehead_npz": root / "prehead.npz",
        "checkpoint_pt": checkpoint_path,
        "report_json": report_path,
        "train_npz": root / "train.npz",
        "query_npz": root / "query.npz",
        "gallery_npz": root / "gallery.npz",
        "retrieval_json": retrieval_path,
    }
    entry = {
        name: {"path": str(path), "sha256": _sha256_file(path)} for name, path in paths.items()
    }

    def source_exporter(**_: Any) -> dict[str, dict[str, np.ndarray]]:
        return {
            split: {
                "embeddings": embeddings[split].copy(),
                "labels": labels[split].copy(),
                "example_ids": np.asarray(ids[split]),
                "source_paths": np.asarray(sources[split]),
                "row_indices": np.arange(len(ids[split]), dtype=np.int64),
            }
            for split in ("train", "query", "gallery")
        }

    return entry, source_exporter


_TINY_PARTITION = {"train": (6, 2), "query": (2, 2), "gallery": (2, 2)}


def test_load_and_bind_seed_accepts_exact_student_evaluation_source(tmp_path: Path) -> None:
    """Catches rejecting the production checkpoint source before its source export."""
    entry, base_exporter = _synthetic_rsta_bundle(tmp_path)
    export_calls = 0

    def source_exporter(**kwargs: Any) -> dict[str, dict[str, np.ndarray]]:
        nonlocal export_calls
        export_calls += 1
        return base_exporter(**kwargs)

    _MODULE.load_and_bind_seed(
        entry,
        seed=0,
        source_exporter=source_exporter,
        expected_partition=_TINY_PARTITION,
        expected_dimension=2,
    )

    assert export_calls == 1


@pytest.mark.parametrize("evaluation_model_source", ["trained_model", "ema_weight_average"])
def test_load_and_bind_seed_rejects_nonstudent_source_before_export(
    tmp_path: Path,
    evaluation_model_source: str,
) -> None:
    """Catches accepting a synthetic alias or EMA checkpoint for RSTA execution."""
    entry, _ = _synthetic_rsta_bundle(
        tmp_path,
        evaluation_model_source=evaluation_model_source,
    )
    export_calls = 0

    def forbidden_exporter(**_kwargs: Any) -> dict[str, dict[str, np.ndarray]]:
        nonlocal export_calls
        export_calls += 1
        raise AssertionError("invalid evaluation source reached source export")

    with pytest.raises(ValueError, match="evaluation_model_source is not student"):
        _MODULE.load_and_bind_seed(
            entry,
            seed=0,
            source_exporter=forbidden_exporter,
            expected_partition=_TINY_PARTITION,
            expected_dimension=2,
        )

    assert export_calls == 0


def test_load_and_bind_seed_returns_immutable_training_only_scientific_input(
    tmp_path: Path,
) -> None:
    entry, source_exporter = _synthetic_rsta_bundle(tmp_path)

    bound = _MODULE.load_and_bind_seed(
        entry,
        seed=0,
        source_exporter=source_exporter,
        expected_partition=_TINY_PARTITION,
        expected_dimension=2,
    )

    assert isinstance(bound, _MODULE.TrainingOnlySeedInput)
    assert bound.train_example_ids.tolist() == [f"train-{index}" for index in range(6)]
    assert bound.train_labels.tolist() == [10, 10, 10, 20, 20, 20]
    assert not hasattr(bound, "query_embeddings")
    assert not hasattr(bound, "gallery_embeddings")
    assert not hasattr(bound, "query_example_ids")
    assert not hasattr(bound, "gallery_example_ids")
    assert bound.train_embeddings.flags.writeable is False
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        bound.train_embeddings.setflags(write=True)
    with pytest.raises(TypeError):
        bound.config["objectives"][0] = "tampered"
    assert set(bound.artifact_binding["current_source_export"]) == {
        "train",
        "query",
        "gallery",
    }
    assert all(
        check["max_abs_descriptor_difference"] == 0.0
        for check in bound.artifact_binding["current_source_export"].values()
    )
    with pytest.raises((AttributeError, TypeError)):
        bound.seed = 9


def test_load_and_bind_seed_rejects_proxy_descriptor_dimension_mismatch(tmp_path: Path) -> None:
    entry, source_exporter = _synthetic_rsta_bundle(tmp_path, proxy_dimension=3)

    with pytest.raises(ValueError, match="proxy descriptor dimension"):
        _MODULE.load_and_bind_seed(
            entry,
            seed=0,
            source_exporter=source_exporter,
            expected_partition=_TINY_PARTITION,
            expected_dimension=2,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest_sha", "manifest SHA-256"),
        ("embedded_digest", "checkpoint digest"),
        ("config", "batch_size"),
        ("id_order", "example-ID order"),
        ("label_order", "label order"),
        ("fractional_label", "labels must use an integral dtype"),
        ("float_label_dtype", "labels must use an integral dtype"),
        ("duplicate_index", "row indices"),
        ("source_membership", "source-path order"),
        ("descriptor", "descriptors differ"),
        ("r1", "official R@1"),
    ],
)
def test_load_and_bind_seed_fails_closed_on_every_binding_mismatch(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    entry, base_exporter = _synthetic_rsta_bundle(
        tmp_path,
        batch_size=179 if mutation == "config" else 180,
        reported_r1=0.5 if mutation == "r1" else 1.0,
        embedded_checkpoint_digest="f" * 64 if mutation == "embedded_digest" else None,
    )
    if mutation == "manifest_sha":
        entry["prehead_npz"]["sha256"] = "0" * 64

    def source_exporter(**kwargs: Any) -> dict[str, dict[str, np.ndarray]]:
        exported = base_exporter(**kwargs)
        if mutation == "id_order":
            exported["train"]["example_ids"][[0, 1]] = exported["train"]["example_ids"][[1, 0]]
        elif mutation == "label_order":
            exported["train"]["labels"][[2, 3]] = exported["train"]["labels"][[3, 2]]
        elif mutation == "fractional_label":
            exported["train"]["labels"] = exported["train"]["labels"].astype(np.float64)
            exported["train"]["labels"][0] = 10.9
        elif mutation == "float_label_dtype":
            exported["train"]["labels"] = exported["train"]["labels"].astype(np.float64)
        elif mutation == "duplicate_index":
            exported["train"]["row_indices"][1] = 0
        elif mutation == "source_membership":
            exported["train"]["source_paths"][[0, 1]] = exported["train"]["source_paths"][[1, 0]]
        elif mutation == "descriptor":
            exported["gallery"]["embeddings"][0] = np.asarray([0.0, 1.0])
        return exported

    with pytest.raises(ValueError, match=message):
        _MODULE.load_and_bind_seed(
            entry,
            seed=0,
            source_exporter=source_exporter,
            expected_partition=_TINY_PARTITION,
            expected_dimension=2,
        )


def test_atomic_json_rejects_nonfinite_without_replacing_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "binding.json"
    output.write_text("sentinel\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Out of range float"):
        _MODULE.write_json_atomic(output, {"value": float("nan")})

    assert output.read_text(encoding="utf-8") == "sentinel\n"
    assert list(tmp_path.glob(".*.tmp")) == []

    with pytest.raises(FileExistsError):
        _MODULE.write_json_atomic(output, {"schema_version": 1, "finite": 2.0})
    assert output.read_text(encoding="utf-8") == "sentinel\n"
    fresh = tmp_path / "fresh.json"
    _MODULE.write_json_atomic(fresh, {"schema_version": 1, "finite": 2.0})
    assert json.loads(fresh.read_text(encoding="utf-8")) == {
        "finite": 2.0,
        "schema_version": 1,
    }


def test_atomic_json_rolls_back_link_when_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches leaving a requested result after publication durability failed."""
    output = tmp_path / "result.json"
    real_fsync = os.fsync
    calls = 0

    def fail_directory_fsync(fd: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("directory fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", fail_directory_fsync)

    with pytest.raises(OSError, match="directory fsync failed"):
        _MODULE.write_json_atomic(output, {"finite": 1.0})

    assert not output.exists()
    assert list(tmp_path.glob(".*.tmp")) == []


def test_atomic_json_does_not_delete_preexisting_exact_temporary_file(tmp_path: Path) -> None:
    """Catches cleanup claiming a PID temp file that this call did not create."""
    output = tmp_path / "result.json"
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    sentinel = b"preexisting unowned temporary bytes\n"
    temporary.write_bytes(sentinel)

    with pytest.raises(FileExistsError):
        _MODULE.write_json_atomic(output, {"finite": 1.0})

    assert not output.exists()
    assert temporary.read_bytes() == sentinel


def _numpy_rng_state_equal(left: tuple[Any, ...], right: tuple[Any, ...]) -> bool:
    return left[0] == right[0] and np.array_equal(left[1], right[1]) and left[2:] == right[2:]


def test_deterministic_transform_cache_is_order_invariant_and_restores_global_rngs() -> None:
    source_values = {"example-a": 1.0, "example-b": 2.0}
    materialized: list[str] = []

    def materialize(value: tuple[str, float]) -> float:
        materialized.append(value[0])
        return value[1]

    def transform(value: float) -> torch.Tensor:
        return torch.tensor(
            [value, random.random(), np.random.random(), torch.rand(()).item()],
            dtype=torch.float64,
        )

    random.seed(987)
    np.random.seed(654)
    torch.manual_seed(321)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    ordered_ids = ["example-b", "example-a"]

    cache = _MODULE.cache_deterministic_transforms(
        ordered_ids,
        {example_id: (example_id, source_values[example_id]) for example_id in ordered_ids},
        transform=transform,
        materialize=materialize,
    )

    assert random.getstate() == python_before
    assert _numpy_rng_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)

    assert materialized == ordered_ids
    assert cache.tensor_sha256 == {
        "example-b": "dbe836c839ad351099072a4ab66575ea9c2f5e8b82a85cfbeb06dc855e5fa72b",
        "example-a": "caef1ad2711c486e37f78fbdffc0db895987a9c765f9dc14bea6a9300760e9cd",
    }
    assert (
        cache.ordered_id_sha256
        == "57da9971203ef92ea031227063af78449810ddf937b48e3f7bb6f3067f76f5eb"
    )
    expected_batch = torch.stack(
        [
            torch.tensor(
                [2.0, 0.8618200168998673, 0.880922787539386, 0.8161659836769104],
                dtype=torch.float64,
            ),
            torch.tensor(
                [1.0, 0.38632329357109396, 0.5420103737595127, 0.11299240589141846],
                dtype=torch.float64,
            ),
        ]
    )
    assert torch.equal(cache.batch(ordered_ids), expected_batch)

    reversed_cache = _MODULE.cache_deterministic_transforms(
        list(reversed(ordered_ids)),
        {example_id: (example_id, source_values[example_id]) for example_id in ordered_ids},
        transform=transform,
        materialize=lambda value: value[1],
    )
    for example_id in ordered_ids:
        assert torch.equal(cache.tensors[example_id], reversed_cache.tensors[example_id])
        assert cache.tensor_sha256[example_id] == reversed_cache.tensor_sha256[example_id]
    assert random.getstate() == python_before
    assert _numpy_rng_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_training_only_input_is_constructed_after_query_gallery_arrays_are_released(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entry, base_exporter = _synthetic_rsta_bundle(tmp_path)
    binding_only_refs: list[weakref.ReferenceType[np.ndarray]] = []

    def source_exporter(**kwargs: Any) -> dict[str, dict[str, np.ndarray]]:
        exported = base_exporter(**kwargs)
        for split in ("query", "gallery"):
            for name in exported[split]:
                binding_only_refs.append(weakref.ref(exported[split][name]))
        return exported

    original_type = _MODULE.TrainingOnlySeedInput

    def guarded_constructor(**kwargs: Any) -> object:
        gc.collect()
        assert all(reference() is None for reference in binding_only_refs)
        return original_type(**kwargs)

    monkeypatch.setattr(_MODULE, "TrainingOnlySeedInput", guarded_constructor)

    bound = _MODULE.load_and_bind_seed(
        entry,
        seed=0,
        source_exporter=source_exporter,
        expected_partition=_TINY_PARTITION,
        expected_dimension=2,
    )

    assert bound.train_example_ids[0] == "train-0"


def test_current_source_exporter_preserves_literal_symlinked_loader_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sfora import image_end_to_end as image_module

    entry, _ = _synthetic_rsta_bundle(tmp_path / "artifacts")
    config = json.loads(Path(entry["report_json"]["path"]).read_text(encoding="utf-8"))["config"]
    dataset_root = tmp_path / "inshop"
    dataset_root.mkdir()
    physical_image_root = dataset_root / "img"
    physical_image_root.mkdir()
    (dataset_root / "Img").symlink_to(physical_image_root, target_is_directory=True)
    image_root = dataset_root / "Img" / "img"
    image_root.mkdir(parents=True)
    partition_rows = [
        ("img/train-0.jpg", "train-item-0", "train"),
        ("img/train-1.jpg", "train-item-1", "train"),
        ("img/query-0.jpg", "eval-item-0", "query"),
        ("img/query-1.jpg", "eval-item-1", "query"),
        ("img/gallery-0.jpg", "eval-item-0", "gallery"),
        ("img/gallery-1.jpg", "eval-item-1", "gallery"),
    ]
    for image_name, _, _ in partition_rows:
        path = dataset_root / "Img" / image_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not-materialized-by-forward-mock")
    partition = dataset_root / "Eval" / "list_eval_partition.txt"
    partition.parent.mkdir(parents=True)
    partition.write_text(
        "6\nimage_name item_id evaluation_status\n"
        + "\n".join(" ".join(row) for row in partition_rows)
        + "\n",
        encoding="utf-8",
    )
    config["dataset_root"] = str(dataset_root)
    checkpoint = torch.load(entry["checkpoint_pt"]["path"], map_location="cpu", weights_only=False)

    class FakeModel:
        def __init__(self) -> None:
            self.loaded: tuple[dict[str, torch.Tensor], bool] | None = None
            self.device: torch.device | None = None
            self.eval_called = False

        def load_state_dict(self, state: dict[str, torch.Tensor], *, strict: bool) -> None:
            self.loaded = (state, strict)

        def to(self, device: torch.device) -> FakeModel:
            self.device = device
            return self

        def eval(self) -> FakeModel:
            self.eval_called = True
            return self

    model = FakeModel()
    monkeypatch.setattr(image_module, "_torchvision_model_factory", lambda _: model)
    traversed: list[list[str]] = []

    def fake_encode(
        observed_model: FakeModel,
        loader: Any,
        device: torch.device,
        torch_module: Any,
    ) -> tuple[np.ndarray, np.ndarray]:
        assert observed_model is model
        assert observed_model.eval_called is True
        assert device == model.device
        assert torch_module is torch
        assert loader.batch_size == 128
        examples = loader.dataset._examples
        traversed.append([example.example_id for example in examples])
        rows = np.zeros((len(examples), 2), dtype=np.float32)
        rows[:, 0] = 1.0
        return rows, np.asarray([example.label for example in examples], dtype=np.int64)

    monkeypatch.setattr(image_module, "_encode_model", fake_encode)

    exported = _MODULE._export_current_source(
        paths={name: Path(item["path"]) for name, item in entry.items()},
        config=config,
        checkpoint=checkpoint,
    )

    assert model.eval_called is True
    assert model.device == torch.device("cpu")
    assert model.loaded is not None
    loaded_state, strict = model.loaded
    assert strict is True
    assert set(loaded_state) == {"model.embedding.weight", "model.embedding.bias"}
    assert traversed == [
        ["inshop-train-img/train-0.jpg", "inshop-train-img/train-1.jpg"],
        ["inshop-query-img/query-0.jpg", "inshop-query-img/query-1.jpg"],
        ["inshop-gallery-img/gallery-0.jpg", "inshop-gallery-img/gallery-1.jpg"],
    ]
    assert set(exported) == {"train", "query", "gallery"}
    expected_labels = {"train": [2, 3], "query": [0, 1], "gallery": [0, 1]}
    for split, expected_ids in zip(("train", "query", "gallery"), traversed, strict=True):
        assert exported[split]["example_ids"].tolist() == expected_ids
        assert exported[split]["labels"].tolist() == expected_labels[split]
        expected_literal_paths = [
            str(dataset_root / "Img" / example_id.split("-", 2)[2])
            for example_id in expected_ids
        ]
        assert expected_literal_paths != [
            str(Path(path).resolve()) for path in expected_literal_paths
        ]
        assert exported[split]["source_paths"].tolist() == expected_literal_paths
        assert exported[split]["row_indices"].tolist() == [0, 1]
        assert exported[split]["embeddings"].shape == (2, 2)


def _synthetic_rsta_manifest(
    root: Path,
) -> tuple[Path, dict[int, Callable[..., dict[str, dict[str, np.ndarray]]]]]:
    docs = root / "docs"
    docs.mkdir(parents=True)
    preregistration = docs / "pass200.md"
    preregistration.write_text("frozen synthetic preregistration\n", encoding="utf-8")
    source_paths = (
        "scripts/diagnose_pass159_cotangent_stage_a.py",
        "scripts/diagnose_pass200_rsta_stage_a.py",
        "scripts/export_final_inshop_embeddings.py",
        "src/sfora/bn_inception.py",
        "src/sfora/data.py",
        "src/sfora/image_end_to_end.py",
    )
    for index, path_text in enumerate(source_paths):
        source_file = root / path_text
        source_file.parent.mkdir(parents=True, exist_ok=True)
        source_file.write_text(f"SOURCE_SCHEMA = {index + 1}\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "rsta@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "RSTA Test"], cwd=root, check=True)
    subprocess.run(["git", "add", *source_paths], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "pin synthetic source"], cwd=root, check=True)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    seeds: dict[str, dict[str, dict[str, str]]] = {}
    exporters: dict[int, Callable[..., dict[str, dict[str, np.ndarray]]]] = {}
    common_sources = root / "images"
    for seed in range(4):
        entry, exporter = _synthetic_rsta_bundle(
            root / f"seed-{seed}",
            seed=seed,
            source_root=common_sources,
        )
        seeds[str(seed)] = entry
        exporters[seed] = exporter
    pass159_manifest = docs / "pass159_stage_a_manifest.json"
    pass159_manifest.write_text(
        json.dumps({"schema_version": 1, "seeds": seeds}),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "preregistration": {
            "path": "docs/pass200.md",
            "sha256": _sha256_file(preregistration),
        },
        "artifact_schema": {
            "path": "docs/pass159_stage_a_manifest.json",
            "sha256": _sha256_file(pass159_manifest),
        },
        "source": {
            "git_revision": revision,
            "files": {path_text: _sha256_file(root / path_text) for path_text in source_paths},
        },
        "seeds": seeds,
    }
    manifest_path = docs / "pass200_rsta_stage_a_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    subprocess.run(
        [
            "git",
            "add",
            "docs/pass200.md",
            "docs/pass159_stage_a_manifest.json",
            "docs/pass200_rsta_stage_a_manifest.json",
        ],
        cwd=root,
        check=True,
    )
    subprocess.run(
        ["git", "commit", "-q", "-m", "commit synthetic manifest"], cwd=root, check=True
    )
    return manifest_path, exporters


def _synthetic_execution_manifest(
    root: Path,
    *,
    seeds: dict[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    diagnostic_path = "scripts/diagnose_pass200_rsta_stage_a.py"
    diagnostic = root / diagnostic_path
    diagnostic.parent.mkdir(parents=True, exist_ok=True)
    diagnostic.write_text("EXECUTION_SCHEMA = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.email", "rsta@example.invalid"], cwd=root, check=True)
    subprocess.run(["git", "config", "user.name", "RSTA Test"], cwd=root, check=True)
    subprocess.run(["git", "add", diagnostic_path], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "freeze diagnostic"], cwd=root, check=True)
    frozen_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    manifest = {
        "source": {
            "git_revision": frozen_revision,
            "files": {diagnostic_path: _sha256_file(diagnostic)},
        },
        "seeds": seeds or {str(seed): {} for seed in range(4)},
    }
    manifest_path = root / "docs" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    subprocess.run(["git", "add", "docs/manifest.json"], cwd=root, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "commit manifest"], cwd=root, check=True)
    return manifest_path, manifest


def _bind_synthetic_executing_diagnostic(
    monkeypatch: pytest.MonkeyPatch,
    *,
    manifest_path: Path,
) -> None:
    repository = manifest_path.resolve().parent.parent
    monkeypatch.setattr(
        _MODULE,
        "__file__",
        str(repository / "scripts" / "diagnose_pass200_rsta_stage_a.py"),
    )


def test_binding_only_cli_is_no_longer_operational_after_receipt_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path, exporters = _synthetic_rsta_manifest(tmp_path)
    output = tmp_path / "binding.json"
    calls: list[int] = []

    def source_exporter(**kwargs: Any) -> dict[str, dict[str, np.ndarray]]:
        seed = int(kwargs["config"]["seed"])
        calls.append(seed)
        return exporters[seed](**kwargs)

    with pytest.raises(SystemExit):
        _MODULE.main(
            [
                "--manifest",
                str(manifest_path),
                "--binding-receipt",
                str(tmp_path / "receipt.json"),
                "--output",
                str(output),
                "--binding-only",
            ]
        )

    assert calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("train_example_ids", "example-ID order"),
        ("train_labels", "label order"),
        ("train_source_paths", "source membership"),
        ("train_row_indices", "row-index binding"),
    ],
)
def test_cross_seed_training_binding_rejects_every_row_metadata_mismatch(
    tmp_path: Path,
    field: str,
    message: str,
) -> None:
    entry, source_exporter = _synthetic_rsta_bundle(tmp_path)
    bound = _MODULE.load_and_bind_seed(
        entry,
        seed=0,
        source_exporter=source_exporter,
        expected_partition=_TINY_PARTITION,
        expected_dimension=2,
    )
    values = list(getattr(bound, field))
    if field == "train_labels":
        values[0] = 999
    elif field == "train_row_indices":
        values[0] = 99
    else:
        values[0] = f"changed-{values[0]}"
    changed = replace(bound, seed=1, **{field: tuple(values)})

    with pytest.raises(ValueError, match=message):
        _MODULE.validate_cross_seed_training_binding([bound, changed])


def test_manifest_validation_rejects_source_and_pass159_schema_drift(tmp_path: Path) -> None:
    manifest_path, _ = _synthetic_rsta_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = tmp_path / "src" / "sfora" / "data.py"
    source_path.write_text("SOURCE_SCHEMA = changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="source SHA-256"):
        _MODULE.validate_rsta_manifest(manifest, manifest_path=manifest_path)

    source_path.write_text("SOURCE_SCHEMA = 3\n", encoding="utf-8")
    manifest["seeds"]["0"]["report_json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Pass159 manifest seeds"):
        _MODULE.validate_rsta_manifest(manifest, manifest_path=manifest_path)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_manifest_source_requires_exact_frozen_file_set(tmp_path: Path, mutation: str) -> None:
    manifest_path, _ = _synthetic_rsta_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        manifest["source"]["files"].pop("src/sfora/data.py")
    else:
        extra = tmp_path / "src" / "unregistered.py"
        extra.write_text("UNREGISTERED = True\n", encoding="utf-8")
        manifest["source"]["files"]["src/unregistered.py"] = _sha256_file(extra)

    with pytest.raises(ValueError, match="source file keys"):
        _MODULE.validate_rsta_manifest(manifest, manifest_path=manifest_path)


@pytest.mark.parametrize(
    "path_text",
    ["scripts/diagnose_pass200_rsta_stage_a.py", "src/sfora/bn_inception.py"],
)
def test_manifest_source_requires_each_executing_source(tmp_path: Path, path_text: str) -> None:
    """Catches omission of either source that directly executes the scientific field."""
    manifest_path, _ = _synthetic_rsta_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["files"].pop(path_text)

    with pytest.raises(ValueError, match="source file keys"):
        _MODULE.validate_rsta_manifest(manifest, manifest_path=manifest_path)


@pytest.mark.parametrize(
    "path_text",
    ["scripts/diagnose_pass200_rsta_stage_a.py", "src/sfora/bn_inception.py"],
)
def test_manifest_source_rejects_executing_source_worktree_drift(
    tmp_path: Path, path_text: str
) -> None:
    """Catches executing a diagnostic/model source differing from the frozen digest."""
    manifest_path, _ = _synthetic_rsta_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    (tmp_path / path_text).write_text("EXECUTION_SOURCE = 'drifted'\n", encoding="utf-8")

    with pytest.raises(ValueError, match=f"source SHA-256 mismatch for {path_text}"):
        _MODULE.validate_rsta_manifest(manifest, manifest_path=manifest_path)


@pytest.mark.parametrize(
    "path_text",
    ["scripts/diagnose_pass200_rsta_stage_a.py", "src/sfora/bn_inception.py"],
)
def test_manifest_source_rejects_executing_source_revision_drift(
    tmp_path: Path, path_text: str
) -> None:
    """Catches a frozen revision whose executing source blob differs from the worktree."""
    manifest_path, _ = _synthetic_rsta_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = tmp_path / path_text
    original = source_path.read_text(encoding="utf-8")
    source_path.write_text("EXECUTION_SOURCE = 'revision-drift'\n", encoding="utf-8")
    subprocess.run(["git", "add", path_text], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "drift executing source"], cwd=tmp_path, check=True
    )
    drift_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_path.write_text(original, encoding="utf-8")
    manifest["source"]["git_revision"] = drift_revision

    with pytest.raises(
        ValueError, match=f"revision blob SHA-256 mismatch for {path_text}"
    ):
        _MODULE.validate_rsta_manifest(manifest, manifest_path=manifest_path)


def test_execution_audit_accepts_manifest_commit_descended_from_frozen_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches requiring executing HEAD to equal, rather than bind, the source revision."""
    manifest_path, manifest = _synthetic_execution_manifest(tmp_path)
    _bind_synthetic_executing_diagnostic(monkeypatch, manifest_path=manifest_path)

    audit = _MODULE.build_execution_audit(manifest, manifest_path=manifest_path)
    _MODULE.validate_execution_audit(
        audit,
        manifest_source=manifest["source"],
        manifest_path=manifest_path,
    )

    assert set(audit) == {
        "executing_git_commit",
        "diagnostic_path",
        "diagnostic_sha256",
        "frozen_source_revision",
    }
    assert audit["executing_git_commit"] != audit["frozen_source_revision"]
    assert len(audit["executing_git_commit"]) == 40
    assert audit["diagnostic_path"] == "scripts/diagnose_pass200_rsta_stage_a.py"
    assert audit["diagnostic_sha256"] == manifest["source"]["files"][
        audit["diagnostic_path"]
    ]


def test_execution_audit_rejects_manifest_repository_other_than_executing_module(
    tmp_path: Path,
) -> None:
    """Catches repo-A code falsely reporting the frozen diagnostic found in repo B."""
    manifest_path, manifest = _synthetic_execution_manifest(tmp_path)

    with pytest.raises(ValueError, match="executing diagnostic path"):
        _MODULE.build_execution_audit(manifest, manifest_path=manifest_path)


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_execution_audit_rejects_missing_or_extra_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Catches partial or caller-extended execution provenance."""
    manifest_path, manifest = _synthetic_execution_manifest(tmp_path)
    _bind_synthetic_executing_diagnostic(monkeypatch, manifest_path=manifest_path)
    audit = _MODULE.build_execution_audit(manifest, manifest_path=manifest_path)
    if mutation == "missing":
        audit.pop("diagnostic_sha256")
    else:
        audit["unchecked"] = True

    with pytest.raises(ValueError, match="execution audit fields"):
        _MODULE.validate_execution_audit(
            audit,
            manifest_source=manifest["source"],
            manifest_path=manifest_path,
        )


@pytest.mark.parametrize("commit", ["HEAD", "f" * 40])
def test_execution_audit_rejects_malformed_or_unresolvable_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, commit: str
) -> None:
    """Catches recording a symbolic, abbreviated, or nonexistent executing revision."""
    manifest_path, manifest = _synthetic_execution_manifest(tmp_path)
    _bind_synthetic_executing_diagnostic(monkeypatch, manifest_path=manifest_path)
    audit = _MODULE.build_execution_audit(manifest, manifest_path=manifest_path)
    audit["executing_git_commit"] = commit

    with pytest.raises(ValueError, match="executing Git commit"):
        _MODULE.validate_execution_audit(
            audit,
            manifest_source=manifest["source"],
            manifest_path=manifest_path,
        )


def test_execution_audit_rejects_wrong_diagnostic_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches binding a different frozen file in place of the executed diagnostic."""
    manifest_path, manifest = _synthetic_execution_manifest(tmp_path)
    _bind_synthetic_executing_diagnostic(monkeypatch, manifest_path=manifest_path)
    audit = _MODULE.build_execution_audit(manifest, manifest_path=manifest_path)
    audit["diagnostic_path"] = "scripts/other.py"

    with pytest.raises(ValueError, match="diagnostic path"):
        _MODULE.validate_execution_audit(
            audit,
            manifest_source=manifest["source"],
            manifest_path=manifest_path,
        )


def test_execution_audit_rejects_mismatched_script_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an observed diagnostic hash detached from the manifest's frozen file hash."""
    manifest_path, manifest = _synthetic_execution_manifest(tmp_path)
    _bind_synthetic_executing_diagnostic(monkeypatch, manifest_path=manifest_path)
    audit = _MODULE.build_execution_audit(manifest, manifest_path=manifest_path)
    audit["diagnostic_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="diagnostic SHA-256"):
        _MODULE.validate_execution_audit(
            audit,
            manifest_source=manifest["source"],
            manifest_path=manifest_path,
        )


def test_manifest_source_rejects_nonexistent_git_revision(tmp_path: Path) -> None:
    manifest_path, _ = _synthetic_rsta_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source"]["git_revision"] = "f" * 40

    with pytest.raises(ValueError, match="does not resolve to a commit"):
        _MODULE.validate_rsta_manifest(manifest, manifest_path=manifest_path)


def test_manifest_source_rejects_revision_blob_mismatch_even_when_worktree_matches(
    tmp_path: Path,
) -> None:
    manifest_path, _ = _synthetic_rsta_manifest(tmp_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    source_path = tmp_path / "src" / "sfora" / "data.py"
    original = source_path.read_text(encoding="utf-8")
    source_path.write_text("SOURCE_SCHEMA = revision-drift\n", encoding="utf-8")
    subprocess.run(["git", "add", "src/sfora/data.py"], cwd=tmp_path, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "drift synthetic source"], cwd=tmp_path, check=True
    )
    drift_revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_path.write_text(original, encoding="utf-8")
    manifest["source"]["git_revision"] = drift_revision

    with pytest.raises(ValueError, match="revision blob SHA-256 mismatch"):
        _MODULE.validate_rsta_manifest(manifest, manifest_path=manifest_path)


def test_deterministic_transform_cache_restores_rngs_when_transform_raises() -> None:
    random.seed(12)
    np.random.seed(34)
    torch.manual_seed(56)
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    def exploding_transform(_: object) -> torch.Tensor:
        random.random()
        np.random.random()
        torch.rand(())
        raise RuntimeError("synthetic transform failure")

    with pytest.raises(RuntimeError, match="synthetic transform failure"):
        _MODULE.cache_deterministic_transforms(
            ["example-a"],
            {"example-a": object()},
            transform=exploding_transform,
        )

    assert random.getstate() == python_before
    assert _numpy_rng_state_equal(np.random.get_state(), numpy_before)
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_deterministic_transform_cache_rejects_tensor_mutation_before_batch_use() -> None:
    cache = _MODULE.cache_deterministic_transforms(
        ["example-a"],
        {"example-a": 1.0},
        transform=lambda value: torch.tensor([value, 2.0]),
    )

    cache.tensors["example-a"].add_(5.0)

    with pytest.raises(ValueError, match="cached tensor SHA-256 mismatch"):
        cache.batch(["example-a"])


def test_training_only_seed_routes_selected_sources_through_deterministic_cache(
    tmp_path: Path,
) -> None:
    entry, source_exporter = _synthetic_rsta_bundle(tmp_path)
    bound = _MODULE.load_and_bind_seed(
        entry,
        seed=0,
        source_exporter=source_exporter,
        expected_partition=_TINY_PARTITION,
        expected_dimension=2,
    )
    selected = ["train-3", "train-0"]

    cache = _MODULE.cache_seed_training_tensors(
        bound,
        selected,
        transform=lambda value: torch.tensor([float(len(value))]),
        materialize=lambda path: Path(path).read_bytes(),
    )

    assert cache.example_ids == tuple(selected)
    assert torch.equal(cache.batch(selected), torch.tensor([[7.0], [7.0]]))
    with pytest.raises(ValueError, match="unknown training example IDs"):
        _MODULE.cache_seed_training_tensors(
            bound,
            ["not-a-training-row"],
            transform=lambda value: torch.tensor([float(len(value))]),
            materialize=lambda path: Path(path).read_bytes(),
        )


def test_training_only_seed_default_materializer_opens_bound_string_path(
    tmp_path: Path,
) -> None:
    """Catches passing a bound string path directly into the official image transform."""
    from PIL import Image

    entry, source_exporter = _synthetic_rsta_bundle(tmp_path / "artifacts")
    bound = _MODULE.load_and_bind_seed(
        entry,
        seed=0,
        source_exporter=source_exporter,
        expected_partition=_TINY_PARTITION,
        expected_dimension=2,
    )
    image_path = tmp_path / "pixel.png"
    Image.new("RGB", (2, 1), color=(17, 23, 31)).save(image_path)
    paths = bound.train_source_paths.astype(str).tolist()
    paths[0] = str(image_path)
    frozen_paths = _MODULE._readonly_array(np.asarray(paths))
    hashes = dict(bound.training_array_sha256)
    hashes["train_source_paths"] = _MODULE._framed_array_sha256(
        "train_source_paths", frozen_paths
    )
    rebound = replace(
        bound,
        train_source_paths=frozen_paths,
        training_array_sha256=hashes,
    )
    observed: list[tuple[str, tuple[int, int, int]]] = []

    def image_transform(image: Any) -> torch.Tensor:
        observed.append((image.mode, image.getpixel((0, 0))))
        return torch.tensor([float(image.width), float(image.height)])

    cache = _MODULE.cache_seed_training_tensors(
        rebound,
        ["train-0"],
        transform=image_transform,
    )

    assert observed == [("RGB", (17, 23, 31))]
    assert torch.equal(cache.batch(["train-0"]), torch.tensor([[2.0, 1.0]]))


class _TinyAffine(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 3, bias=True, dtype=torch.float64)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.tensor([[0.7, -0.2], [0.1, 0.9], [-0.4, 0.3]], dtype=torch.float64)
            )
            self.projection.bias.copy_(torch.tensor([0.2, -0.1, 0.4], dtype=torch.float64))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(values)


def _dense_affine_fixture() -> tuple[_TinyAffine, torch.Tensor, torch.Tensor]:
    model = _TinyAffine()
    images = torch.tensor([[0.2, -0.8], [1.1, 0.4], [-0.5, 0.7]], dtype=torch.float64)
    cotangents = torch.tensor(
        [[0.3, -0.5, 0.2], [-0.1, 0.4, 0.6], [0.7, 0.2, -0.3]],
        dtype=torch.float64,
    )
    return model, images, cotangents


def test_exact_kernel_fields_match_independent_dense_jacobian_and_central_difference() -> None:
    """Catches transposed VJPs, mixed receivers, or rescaling a JVP without undoing it."""
    model, images, cotangents = _dense_affine_fixture()
    initial = torch.cat(
        [model.projection.weight.detach().reshape(-1), model.projection.bias.detach()]
    ).requires_grad_(True)

    def literal_encoder(flat: torch.Tensor) -> torch.Tensor:
        weight = flat[:6].reshape(3, 2)
        bias = flat[6:]
        raw = images @ weight.T + bias
        return raw / torch.linalg.vector_norm(raw, dim=1, keepdim=True)

    dense = torch.autograd.functional.jacobian(literal_encoder, initial).reshape(9, 9)
    flat_cotangent = cotangents.reshape(-1)
    expected_g = dense.T @ flat_cotangent
    expected_b = (dense @ expected_g).reshape(3, 3)
    expected_s = []
    for receiver in range(3):
        block = dense[receiver * 3 : (receiver + 1) * 3]
        expected_s.append(block @ (block.T @ cotangents[receiver]))
    expected_s_tensor = torch.stack(expected_s)

    observed = _MODULE.exact_kernel_fields(
        model,
        images,
        cotangents,
        receiver_indices=(0, 1, 2),
        expected_batch_size=3,
        expected_dimension=3,
    )

    assert torch.allclose(observed["parameter_gradient_flat"], expected_g, atol=1e-8, rtol=1e-8)
    assert torch.allclose(observed["batch_motion"], expected_b, atol=1e-8, rtol=1e-8)
    assert torch.allclose(observed["self_motion"], expected_s_tensor, atol=1e-8, rtol=1e-8)
    epsilon = 1.0e-5
    positive = literal_encoder(initial + epsilon * expected_g)
    negative = literal_encoder(initial - epsilon * expected_g)
    finite_difference = (positive - negative) / (2.0 * epsilon)
    assert torch.allclose(observed["batch_motion"], finite_difference, atol=1e-6, rtol=1e-6)
    for receiver in range(3):
        block = dense[receiver * 3 : (receiver + 1) * 3]
        receiver_gradient = block.T @ cotangents[receiver]
        positive = literal_encoder(initial + epsilon * receiver_gradient)[receiver]
        negative = literal_encoder(initial - epsilon * receiver_gradient)[receiver]
        finite_difference = (positive - negative) / (2.0 * epsilon)
        assert torch.allclose(
            observed["self_motion"][receiver],
            finite_difference,
            atol=1e-6,
            rtol=1e-6,
        )


class _TinyBatchNorm(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(2, dtype=torch.float64)
        self.projection = torch.nn.Linear(2, 2, dtype=torch.float64)
        with torch.no_grad():
            self.bn.weight.copy_(torch.tensor([1.2, 0.7], dtype=torch.float64))
            self.bn.bias.copy_(torch.tensor([-0.1, 0.2], dtype=torch.float64))
            self.projection.weight.copy_(
                torch.tensor([[0.8, -0.4], [0.3, 0.9]], dtype=torch.float64)
            )
            self.projection.bias.copy_(torch.tensor([0.05, -0.02], dtype=torch.float64))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.projection(self.bn(values))


def test_bufferless_train_clone_matches_preupdate_bn_forward_and_gradients() -> None:
    """Catches eval-mode substitution, BN-buffer mutation, or dropping affine gradients."""
    original = _TinyBatchNorm().train()
    reference = deepcopy(original).train()
    before = {name: value.detach().clone() for name, value in original.named_buffers()}
    values = torch.tensor([[0.4, -0.3], [1.2, 0.8]], dtype=torch.float64)
    target = torch.tensor([[0.2, -0.5], [0.7, 0.1]], dtype=torch.float64)

    reference_output = reference(values)
    reference_gradient = torch.autograd.grad(
        (reference_output * target).sum(), tuple(reference.parameters())
    )
    clone = _MODULE.make_bufferless_train_clone(original)
    observed_output = clone(values)
    observed_gradient = torch.autograd.grad(
        (observed_output * target).sum(), tuple(clone.parameters())
    )

    assert torch.allclose(observed_output, reference_output, atol=1e-6, rtol=1e-6)
    for observed, expected in zip(observed_gradient, reference_gradient, strict=True):
        assert torch.allclose(observed, expected, atol=1e-6, rtol=1e-6)
    assert clone.training
    assert all(
        not module.track_running_stats
        for module in clone.modules()
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
    )
    for name, value in original.named_buffers():
        assert torch.equal(value, before[name])


class _ExactGlobalMaxRoot(torch.nn.Module):
    def __init__(self, *, output_size: Any = 1) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.gmp = torch.nn.AdaptiveMaxPool2d(output_size)


def test_diagnostic_clone_replaces_only_exact_global_max_and_preserves_original() -> None:
    original = _ExactGlobalMaxRoot()

    clone = _MODULE.make_rsta_diagnostic_clone(original)

    assert type(original.model.gmp) is torch.nn.AdaptiveMaxPool2d
    assert type(clone.model.gmp) is _MODULE._DeterministicGlobalMaxPool2d
    assert list(clone.model.gmp.parameters()) == []
    assert list(clone.model.gmp.buffers()) == []
    assert original.state_dict().keys() == clone.state_dict().keys()
    values = torch.tensor(
        [[[[100.0, 100.0, -1.0], [2.0, 3.0, 4.0]]]], requires_grad=True
    )
    expected_values = values.detach().clone().requires_grad_(True)
    observed = clone.model.gmp(values)
    expected = original.model.gmp(expected_values)
    observed.sum().backward()
    expected.sum().backward()
    assert torch.equal(observed, expected)
    assert torch.equal(values.grad, expected_values.grad)
    assert values.grad.flatten().tolist() == [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.mark.parametrize("defect", ["missing", "subclass", "tuple", "two"])
def test_diagnostic_clone_rejects_nonexact_global_max_contract(defect: str) -> None:
    model = _ExactGlobalMaxRoot()
    if defect == "missing":
        del model.model.gmp
    elif defect == "subclass":
        class PoolSubclass(torch.nn.AdaptiveMaxPool2d):
            pass

        model.model.gmp = PoolSubclass(1)
    elif defect == "tuple":
        model.model.gmp = torch.nn.AdaptiveMaxPool2d((1, 1))
    else:
        model.model.gmp = torch.nn.AdaptiveMaxPool2d(2)

    with pytest.raises(ValueError, match="model.gmp"):
        _MODULE.make_rsta_diagnostic_clone(model)


_GLOBAL_MAX_INPUT_SHA256 = {
    "random": "849f58506a8eabf18741d830a3d83e053d327786a8bfe731df0556b31d43389c",
    "relu": "5810fd957d263f60a15aff4c9a4cb3401a7ad99b165413eaa8503026582a8887",
    "zeros": "16d0edc8b7ad7705b23a14058f366ff1c0dfa16a0ad14f741924c308754cf8d1",
    "tie": "55688cd7f3585fc5402d755dde3f30ac70701bea80c44b8a2e13d5dfa394d5b5",
}

_ZERO_JACOBIAN_NAMES = [
    "model.last_linear.weight",
    "model.last_linear.bias",
]


def _valid_zero_jacobian_audit() -> dict[str, Any]:
    return {
        "audit_id": "pass200-zero-jacobian-last-linear-v1",
        "parameter_names": list(_ZERO_JACOBIAN_NAMES),
        "parameter_shapes": [[1000, 1024], [1000]],
        "parameter_dtypes": ["torch.float32", "torch.float32"],
        "pre_sha256": {name: "1" * 64 for name in _ZERO_JACOBIAN_NAMES},
        "restored_sha256": {name: "1" * 64 for name in _ZERO_JACOBIAN_NAMES},
        "gradients_none": [True, True],
        "mutated_output_equal": True,
        "frozen_requires_grad": [False, False],
    }


def _valid_global_max_audit() -> dict[str, Any]:
    cpu = _MODULE._audit_deterministic_global_max_cpu(
        _MODULE._deterministic_global_max_inputs()
    )
    cuda = {
        name: {
            "output_equal": True,
            "index_equal": True,
            "replacement_gradient_equal_expected": True,
            "max_abs_output_difference": 0.0,
            "max_abs_replacement_gradient_difference": 0.0,
        }
        for name in ("random", "relu", "zeros", "tie")
    }
    return {
        "replacement_id": "pass200-global-max-flatten-first-v1",
        "module_path": "model.gmp",
        "reference_type": "torch.nn.modules.pooling.AdaptiveMaxPool2d",
        "reference_output_size": 1,
        "fixture_seed": 200,
        "fixture_generator": "numpy.PCG64",
        "fixture_shape": [2, 3, 5, 7],
        "fixture_dtype": "float32",
        "derivative": "output.sum()",
        "input_sha256": dict(_GLOBAL_MAX_INPUT_SHA256),
        "cases": {"cpu": cpu, "cuda": cuda},
        "deterministic_cuda_backward": {
            "enabled": True,
            "warn_only": False,
            "completed": True,
        },
    }


def test_deterministic_global_max_fixture_has_exact_inputs_and_cpu_equivalence() -> None:
    cases = _MODULE._deterministic_global_max_inputs()

    assert list(cases) == ["random", "relu", "zeros", "tie"]
    assert all(value.dtype == np.float32 and value.flags.c_contiguous for value in cases.values())
    assert {
        name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
        for name, value in cases.items()
    } == _GLOBAL_MAX_INPUT_SHA256
    assert _MODULE._audit_deterministic_global_max_cpu(cases) == {
        name: {
            "output_equal": True,
            "gradient_equal": True,
            "max_abs_output_difference": 0.0,
            "max_abs_gradient_difference": 0.0,
        }
        for name in ("random", "relu", "zeros", "tie")
    }


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "false", "int_zero", "nonzero", "wrong_type"],
)
def test_deterministic_global_max_audit_rejects_exact_schema_drift(mutation: str) -> None:
    audit = _valid_global_max_audit()
    if mutation == "missing":
        audit.pop("derivative")
    elif mutation == "extra":
        audit["unchecked"] = True
    elif mutation == "false":
        audit["cases"]["cuda"]["tie"]["index_equal"] = False
    elif mutation == "int_zero":
        audit["cases"]["cpu"]["random"]["max_abs_output_difference"] = 0
    elif mutation == "nonzero":
        audit["cases"]["cpu"]["random"]["max_abs_output_difference"] = 1.0e-12
    else:
        audit["reference_output_size"] = 1.0

    with pytest.raises(ValueError, match="deterministic global max"):
        _MODULE._validate_deterministic_global_max_audit(audit)


def test_deterministic_global_max_audit_requires_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _MODULE.configure_deterministic_process()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(ValueError, match="CUDA"):
        _MODULE.audit_deterministic_global_max()


def test_zero_jacobian_classifier_audit_restores_and_freezes_only_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    _MODULE.configure_deterministic_process()
    class DisconnectedClassifier(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.last_linear = torch.nn.Linear(1024, 1000, dtype=torch.float32)
            self.model.embedding = torch.nn.Linear(4, 3, dtype=torch.float32)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return self.model.embedding(values)

    model = DisconnectedClassifier().train()
    before = {
        name: value.detach().clone()
        for name, value in model.named_parameters()
    }
    images = torch.arange(180 * 4, dtype=torch.float32).reshape(180, 4) / 100.0

    audit = _MODULE.audit_zero_jacobian_classifier(model, images)

    _MODULE._validate_zero_jacobian_classifier_audit(audit)
    assert audit["parameter_names"] == _ZERO_JACOBIAN_NAMES
    assert audit["gradients_none"] == [True, True]
    assert audit["mutated_output_equal"] is True
    assert audit["pre_sha256"] == audit["restored_sha256"]
    for name, value in model.named_parameters():
        assert torch.equal(value, before[name])
        assert value.requires_grad is (name not in _ZERO_JACOBIAN_NAMES)


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "name", "shape", "dtype", "sha", "gradient", "output", "frozen"],
)
def test_zero_jacobian_classifier_audit_rejects_exact_schema_drift(mutation: str) -> None:
    audit = _valid_zero_jacobian_audit()
    if mutation == "missing":
        audit.pop("audit_id")
    elif mutation == "extra":
        audit["unchecked"] = True
    elif mutation == "name":
        audit["parameter_names"].reverse()
    elif mutation == "shape":
        audit["parameter_shapes"][0] = [1000, 1023]
    elif mutation == "dtype":
        audit["parameter_dtypes"][0] = "torch.float64"
    elif mutation == "sha":
        audit["restored_sha256"][_ZERO_JACOBIAN_NAMES[0]] = "2" * 64
    elif mutation == "gradient":
        audit["gradients_none"][0] = False
    elif mutation == "output":
        audit["mutated_output_equal"] = False
    else:
        audit["frozen_requires_grad"][1] = True

    with pytest.raises(ValueError, match="zero-Jacobian classifier"):
        _MODULE._validate_zero_jacobian_classifier_audit(audit)


def test_capture_prehead_and_raw_uses_the_executed_final_affine_head() -> None:
    """Catches analytic-head controls fed normalized outputs or the wrong linear layer."""
    model, images, _ = _dense_affine_fixture()
    grad_modes: list[bool] = []
    handle = model.register_forward_pre_hook(
        lambda _module, _inputs: grad_modes.append(torch.is_grad_enabled())
    )
    prehead, raw, output = _MODULE.capture_prehead_and_raw(
        model,
        images,
        head_name="projection",
        expected_in_features=2,
        expected_out_features=3,
    )
    handle.remove()
    expected_raw = images @ model.projection.weight.T + model.projection.bias

    assert torch.equal(prehead, images)
    assert torch.equal(raw, expected_raw)
    assert torch.equal(output, expected_raw)
    assert not prehead.requires_grad and prehead.grad_fn is None
    assert not raw.requires_grad and raw.grad_fn is None
    assert not output.requires_grad and output.grad_fn is None
    assert grad_modes == [False]


def test_integrity_only_gate_short_circuits_on_adjoint_failure() -> None:
    events: list[str] = []

    def fail_adjoint() -> float:
        events.append("adjoint")
        raise ValueError("adjoint failed")

    with pytest.raises(ValueError, match="adjoint failed"):
        _MODULE._integrity_only(
            repeatability_runner=lambda: events.append("repeatability") or {},
            adjoint_runner=fail_adjoint,
            rotation_runner=lambda: events.append("rotation") or {},
        )

    assert events == ["repeatability", "adjoint"]


def test_contextual_pa_fields_use_full_batch_and_exclude_proxy_parameters() -> None:
    """Catches singleton cotangents or an accidental gradient path through proxies."""
    from sfora.image_end_to_end import _normalize, _proxy_anchor_loss

    model, images, _ = _dense_affine_fixture()
    labels = torch.tensor([0, 1, 0], dtype=torch.int64)
    proxies = torch.tensor(
        [[0.8, -0.2, 0.5], [-0.4, 0.7, 0.1]], dtype=torch.float64, requires_grad=True
    )
    proxy_labels = torch.tensor([0, 1], dtype=torch.int64)

    observed = _MODULE.exact_contextual_rsta_fields(
        model,
        images,
        labels,
        proxies,
        proxy_labels,
        alpha=32.0,
        delta=0.1,
        receiver_indices=(0, 1, 2),
        expected_batch_size=3,
        expected_dimension=3,
    )
    raw = model(images)
    z = _normalize(raw, torch)
    full_loss = _proxy_anchor_loss(
        z,
        labels,
        proxy_embeddings=proxies.detach(),
        proxy_labels=proxy_labels,
        alpha=32.0,
        delta=0.1,
        torch_module=torch,
    )
    expected = -torch.autograd.grad(full_loss, z)[0]
    singleton = []
    for row in range(3):
        one_loss = _proxy_anchor_loss(
            z[row : row + 1],
            labels[row : row + 1],
            proxy_embeddings=proxies.detach(),
            proxy_labels=proxy_labels,
            alpha=32.0,
            delta=0.1,
            torch_module=torch,
        )
        singleton.append(-torch.autograd.grad(one_loss, z, retain_graph=True)[0][row])

    assert torch.allclose(observed["dbar"], expected, atol=1e-12, rtol=1e-12)
    assert not torch.allclose(observed["dbar"], torch.stack(singleton))
    assert proxies.grad is None
    assert all("proxy" not in name for name in observed["parameter_names"])


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("eval", "train mode"),
        ("batch", "batch size 180"),
        ("dimension", "dimension 512"),
        ("alpha", "alpha must equal 32"),
        ("delta", "delta must equal 0.1"),
    ],
)
def test_contextual_derivative_boundary_rejects_nonofficial_execution(
    defect: str, message: str
) -> None:
    """Catches eval, microbatch, wrong descriptor, or altered Proxy Anchor constants."""
    model, images, _ = _dense_affine_fixture()
    labels = torch.tensor([0, 1, 0], dtype=torch.int64)
    proxies = torch.tensor([[0.8, -0.2, 0.5], [-0.4, 0.7, 0.1]], dtype=torch.float64)
    proxy_labels = torch.tensor([0, 1], dtype=torch.int64)
    kwargs = {
        "alpha": 32.0,
        "delta": 0.1,
        "receiver_indices": (0, 1, 2),
        "expected_batch_size": 3,
        "expected_dimension": 3,
    }
    if defect == "eval":
        model.eval()
    elif defect == "batch":
        kwargs["expected_batch_size"] = 180
    elif defect == "dimension":
        kwargs["expected_dimension"] = 512
    elif defect == "alpha":
        kwargs["alpha"] = 31.0
    else:
        kwargs["delta"] = 0.2

    with pytest.raises(ValueError, match=message):
        _MODULE.exact_contextual_rsta_fields(
            model,
            images,
            labels,
            proxies,
            proxy_labels,
            **kwargs,
        )


def test_contextual_derivative_boundary_requires_every_bn_train_and_bufferless() -> None:
    """Catches a mixed eval BN or running-stat mutation in the exact graph."""
    model = _TinyBatchNorm().train()
    images = torch.tensor([[0.4, -0.3], [1.2, 0.8]], dtype=torch.float64)
    labels = torch.tensor([0, 1], dtype=torch.int64)
    proxies = torch.eye(2, dtype=torch.float64)
    proxy_labels = torch.tensor([0, 1], dtype=torch.int64)
    for defect in ("tracked", "eval"):
        candidate = deepcopy(model)
        if defect == "tracked":
            candidate.bn.track_running_stats = True
        else:
            candidate.bn.track_running_stats = False
            candidate.bn.eval()
        with pytest.raises(ValueError, match="BatchNorm.*train.*bufferless"):
            _MODULE.exact_contextual_rsta_fields(
                candidate,
                images,
                labels,
                proxies,
                proxy_labels,
                alpha=32.0,
                delta=0.1,
                receiver_indices=(0, 1),
                expected_batch_size=2,
                expected_dimension=2,
            )


class _NamedHeadWithDecoy(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.embedding = torch.nn.Linear(2, 3, dtype=torch.float64)
        self.decoy = torch.nn.Linear(3, 3, dtype=torch.float64)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        embedded = self.model.embedding(values)
        self.decoy(embedded)
        return embedded


def test_capture_prehead_is_bound_to_exact_named_embedding_not_last_matching_linear() -> None:
    """Catches selecting an executed decoy merely because its output shape matches."""
    model = _NamedHeadWithDecoy()
    images = torch.tensor([[0.2, -0.8], [1.1, 0.4]], dtype=torch.float64)
    prehead, raw, output = _MODULE.capture_prehead_and_raw(
        model,
        images,
        head_name="model.embedding",
        expected_in_features=2,
        expected_out_features=3,
    )
    assert torch.equal(prehead, images)
    assert torch.equal(raw, output)


def test_adjoint_identity_and_repeatability_are_exact_on_tiny_model() -> None:
    """Catches inconsistent functional graphs or a nondeterministic derivative path."""
    model, images, cotangents = _dense_affine_fixture()
    direction = {
        name: torch.arange(value.numel(), dtype=value.dtype).reshape(value.shape) / 17.0
        for name, value in model.named_parameters()
    }
    first = _MODULE.exact_kernel_fields(
        model,
        images,
        cotangents,
        receiver_indices=(0, 2),
        expected_batch_size=3,
        expected_dimension=3,
    )
    second = _MODULE.exact_kernel_fields(
        model,
        images,
        cotangents,
        receiver_indices=(0, 2),
        expected_batch_size=3,
        expected_dimension=3,
    )
    error = _MODULE.adjoint_relative_error(
        model,
        images,
        cotangents,
        direction,
        expected_batch_size=3,
        expected_dimension=3,
    )

    assert error < 1.0e-12
    for name in ("z", "batch_motion", "self_motion", "dbar"):
        assert torch.equal(first[name], second[name])


def test_registered_adjoint_directions_use_separate_exact_pcg64_streams() -> None:
    """Catches shared RNG state, hex seeds, or non-C-order parameter filling."""
    model, images, cotangents = _dense_affine_fixture()
    output, parameters = _MODULE.registered_adjoint_directions(
        model, cotangents.shape, seed=2, dtype=torch.float64, device=torch.device("cpu")
    )
    u_seed = int.from_bytes(
        hashlib.sha256(b"rsta-stage-a-v1|adjoint-u|\0" + b"2").digest()[:8], "big"
    )
    v_seed = int.from_bytes(
        hashlib.sha256(b"rsta-stage-a-v1|adjoint-v|\0" + b"2").digest()[:8], "big"
    )
    expected_u = np.random.Generator(np.random.PCG64(u_seed)).standard_normal((3, 3))
    expected_v = np.random.Generator(np.random.PCG64(v_seed)).standard_normal(9)

    assert np.array_equal(output.numpy(), expected_u)
    assert np.array_equal(
        torch.cat([parameters[name].reshape(-1) for name, _ in model.named_parameters()]).numpy(),
        expected_v,
    )
    assert images.shape == (3, 2)


def test_configure_deterministic_process_requires_preexported_cublas_and_records_gates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches late CUBLAS setup, warn-only determinism, TF32, or cuDNN benchmarking."""
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="CUBLAS_WORKSPACE_CONFIG"):
        _MODULE.configure_deterministic_process()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = (
        torch.are_deterministic_algorithms_enabled(),
        torch.backends.cudnn.benchmark,
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    try:
        audit = _MODULE.configure_deterministic_process()
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.backends.cudnn.benchmark
        assert not torch.backends.cuda.matmul.allow_tf32
        assert not torch.backends.cudnn.allow_tf32
        assert audit["cublas_workspace_config"] == ":4096:8"
        assert audit["deterministic_warn_only"] is False
        assert audit["autocast"] is False
        assert audit["model_arithmetic"] == "float32"
    finally:
        torch.use_deterministic_algorithms(previous[0])
        torch.backends.cudnn.benchmark = previous[1]
        torch.backends.cuda.matmul.allow_tf32 = previous[2]
        torch.backends.cudnn.allow_tf32 = previous[3]


def test_default_integrity_fixtures_persist_measured_residuals_not_booleans() -> None:
    """Catches a scientific CLI that marks dense/BN fixtures passed without executing them."""
    audit = _MODULE._default_fixture_runner()
    assert audit["dense_fixture"]["passed"] is True
    assert audit["dense_fixture"]["max_jacobian_residual"] <= 1.0e-8
    assert audit["dense_fixture"]["max_finite_difference_residual"] <= 1.0e-6
    assert audit["bn_fixture"]["passed"] is True
    assert audit["bn_fixture"]["max_output_residual"] <= 1.0e-6
    assert audit["bn_fixture"]["max_gradient_residual"] <= 1.0e-6
    assert audit["bn_fixture"]["buffers_unchanged"] is True


class _UnusedParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.used = torch.nn.Linear(2, 3, dtype=torch.float64)
        self.unused = torch.nn.Parameter(torch.ones(2, dtype=torch.float64))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.used(values)


def test_exact_kernel_fields_fail_closed_on_disconnected_parameter() -> None:
    """Catches silent materialization of a missing encoder gradient as zero."""
    model = _UnusedParameterModel()
    images = torch.tensor([[0.2, 0.3], [0.7, -0.4]], dtype=torch.float64)
    cotangents = torch.tensor([[0.1, -0.2, 0.4], [0.5, 0.3, -0.1]], dtype=torch.float64)
    with pytest.raises(ValueError, match="missing gradient.*unused"):
        _MODULE.exact_kernel_fields(
            model,
            images,
            cotangents,
            receiver_indices=(0,),
            expected_batch_size=2,
            expected_dimension=3,
        )


@pytest.mark.parametrize("defect", ["zero", "radial", "nonfinite"])
def test_project_and_validate_fields_rejects_registered_vector_defects(defect: str) -> None:
    """Catches row dropping or accepting zero, radial, or nonfinite diagnostic vectors."""
    z = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    vectors = {
        "dbar": torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64),
        "b": torch.tensor([[0.0, 0.4, 0.2]], dtype=torch.float64),
        "s": torch.tensor([[0.0, 0.3, -0.1]], dtype=torch.float64),
    }
    if defect == "zero":
        vectors["s"].zero_()
    elif defect == "radial":
        vectors["b"] = torch.tensor([[0.01, 1.0e-6, 0.0]], dtype=torch.float64)
    else:
        vectors["dbar"][0, 1] = float("nan")

    with pytest.raises(ValueError, match="nonzero|radial|nonfinite"):
        _MODULE.project_and_validate_fields(z, **vectors)


def _valid_scientific_payload_arguments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    example_ids, labels = _selection_fixture()
    primary = _MODULE.select_primary_panel(example_ids, labels)
    alternate = _MODULE.select_alternate_panel(example_ids, labels, primary)
    tensor_hashes = {
        example_id: hashlib.sha256(example_id.encode()).hexdigest() for example_id in example_ids
    }
    rank_zero = {
        label: support_ids[0] for label, support_ids in primary["support_ids_by_label"].items()
    }

    def make_row(seed: int, panel: str, position: int) -> dict[str, Any]:
        selected = primary if panel == "primary" else alternate
        label = selected["labels"][position]
        receiver_id = selected["receiver_ids"][position]
        batch_index = position // 8
        batch_ids = selected["batches"][batch_index]
        receiver_index = batch_ids.index(receiver_id)
        foreign_ids = [
            rank_zero[other]
            for other in primary["eligible_labels"]
            if other != label and rank_zero[other] not in batch_ids
        ][:32]
        a_self = 0.20 if panel == "primary" else 0.19
        a_batch = 0.175 if panel == "primary" else 0.18
        a_desc = 0.18
        return {
            "panel": panel,
            "seed": seed,
            "label": label,
            "batch_index": batch_index,
            "receiver_index": receiver_index,
            "receiver_id": receiver_id,
            "support_ids": primary["support_ids_by_label"][label],
            "foreign_ids": foreign_ids,
            "batch_ids": batch_ids,
            "batch_tensor_sha256": [tensor_hashes[value] for value in batch_ids],
            "batch_id_order_sha256": _MODULE._ordered_text_sha256(batch_ids),
            "tensor_sha256": tensor_hashes[receiver_id],
            "a_self": a_self,
            "a_batch": a_batch,
            "delta": a_self - a_batch,
            "a_desc": a_desc,
            "self_minus_desc": a_self - a_desc,
            "rho": 0.15,
            "log_ratio": 0.05,
            "cos_b_s": 0.7,
            "random_a_self": 0.03,
            "random_a_batch": 0.02,
            "random_delta": 0.01,
            "deranged_a_self": 0.04,
            "deranged_a_batch": 0.04,
            "deranged_delta": 0.0,
            "norm_z": 1.0,
            "norm_dbar": 0.4,
            "norm_b": 0.3,
            "norm_s": 0.25,
            "norm_q": 0.2,
            "norm_random_target": 0.25,
            "norm_deranged_target": 1.0,
            "radial_fraction_dbar": 0.0,
            "radial_fraction_b": 0.0,
            "radial_fraction_s": 0.0,
            "head_a_batch": 0.1,
            "head_a_self": 0.2,
            "head_self_desc_gap": 0.0,
            "norm_b_head": 0.3,
            "norm_s_head": 0.2,
            "support_cosines": [0.25] * 34,
        }

    primary_rows = [
        make_row(seed, "primary", position) for seed in range(4) for position in range(64)
    ]
    alternate_rows = [
        make_row(seed, "alternate", position) for seed in range(4) for position in range(16)
    ]
    aggregation = _MODULE.decide_stage_a(primary_rows, alternate_rows)
    _, matrices = _MODULE._panel_matrices(
        primary_rows,
        identity_count=64,
        value_names=("delta", "self_minus_desc"),
        panel_name="primary",
    )
    delta_bootstrap = _MODULE.joint_bootstrap(matrices["delta"])
    self_desc_bootstrap = _MODULE.joint_bootstrap(matrices["self_minus_desc"])
    used_ids = {value for batch in [*primary["batches"], *alternate["batches"]] for value in batch}
    seed_audits = []
    for seed in range(4):
        seed_audits.append(
            {
                "seed": seed,
                "official_recall_at_1": 0.9,
                "artifact_binding": {"checkpoint_sha256": "d" * 64},
                "config": {
                    "batch_size": 180,
                    "embedding_dimensions": 512,
                    "proxy_anchor_alpha": 32.0,
                    "proxy_anchor_delta": 0.1,
                },
                "parameter_names": ["model.embedding.weight", "model.embedding.bias"],
                "parameter_count": 1024 * 512 + 512,
                "proxy_sha256": "e" * 64,
                "proxy_label_sha256": "f" * 64,
                "train_example_id_order_sha256": _MODULE._ordered_text_sha256(example_ids),
                "train_label_order_sha256": _MODULE._ordered_int64_sha256(labels),
                "train_source_order_sha256": "3" * 64,
                "transform_cache_order_sha256": _MODULE._ordered_text_sha256(sorted(used_ids)),
                "transform_tensor_sha256": {
                    value: tensor_hashes[value] for value in sorted(used_ids)
                },
                "primary_batch_ids": primary["batches"],
                "alternate_batch_ids": alternate["batches"],
            }
        )
    integrity = {
        "dense_fixture": {
            "passed": True,
            "max_jacobian_residual": 1.0e-10,
            "max_finite_difference_residual": 1.0e-8,
            "jacobian_tolerance": 1.0e-8,
            "finite_difference_tolerance": 1.0e-6,
        },
        "bn_fixture": {
            "passed": True,
            "max_output_residual": 1.0e-8,
            "max_gradient_residual": 1.0e-8,
            "tolerance": 1.0e-6,
            "buffers_unchanged": True,
        },
        "zero_jacobian_classifier": {
            str(seed): _valid_zero_jacobian_audit() for seed in range(4)
        },
        "seeds": [
            {
                "seed": seed,
                "repeatability": {
                    name: {"first_sha256": str(seed) * 64, "repeat_sha256": str(seed) * 64}
                    for name in ("z", "dbar", "b", "s")
                },
                "adjoint_relative_error": 0.0,
                "rotation": {
                    "vector_residuals": {name: 0.0 for name in ("z", "dbar", "b", "s", "q")},
                    "statistic_differences": {
                        name: 0.0
                        for name in (
                            "A_self",
                            "A_batch",
                            "Delta",
                            "A_desc",
                            "rho",
                            "log_ratio",
                            "cos_b_s",
                        )
                    },
                },
            }
            for seed in range(4)
        ],
    }
    manifest_path, manifest = _synthetic_execution_manifest(tmp_path / "execution-binding")
    _bind_synthetic_executing_diagnostic(monkeypatch, manifest_path=manifest_path)
    execution_audit = _MODULE.build_execution_audit(manifest, manifest_path=manifest_path)
    return {
        "manifest_audit": {
            "path": str(manifest_path),
            "sha256": _sha256_file(manifest_path),
            "source": manifest["source"],
        },
        "execution_audit": execution_audit,
        "environment": {
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "deterministic_warn_only": False,
            "cudnn_benchmark": False,
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "autocast": False,
            "model_arithmetic": "float32",
            "reduction_arithmetic": "float64",
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
        },
        "seed_audits": seed_audits,
        "primary_rows": primary_rows,
        "alternate_rows": alternate_rows,
        "integrity": integrity,
        "aggregation": aggregation,
        "bootstrap": {
            "delta_distribution": delta_bootstrap.tolist(),
            "delta_sha256": _MODULE.float64_c_order_sha256(delta_bootstrap),
            "self_minus_desc_distribution": self_desc_bootstrap.tolist(),
            "self_minus_desc_sha256": _MODULE.float64_c_order_sha256(self_desc_bootstrap),
        },
        "panel_binding": {
            "primary": primary,
            "alternate": alternate,
            "expected_dimension": 512,
            "tensor_sha256": tensor_hashes,
            "foreign_support_ids": sorted(rank_zero.values()),
        },
    }


def test_scientific_payload_requires_receiver_rows_and_persists_full_audit_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an aggregate-only result or omission of a registered audit field."""
    arguments = _valid_scientific_payload_arguments(tmp_path, monkeypatch)
    payload = _MODULE.scientific_payload(**arguments)
    assert payload["candidate_values_computed"] is True
    assert len(payload["rows"]["primary"]) == 4 * 64
    assert len(payload["rows"]["alternate"]) == 4 * 16
    assert "control_aggregates" in payload["aggregation"]
    assert payload["execution_audit"] == arguments["execution_audit"]


@pytest.mark.parametrize("mutation", ["missing_seed", "invalid_seed"])
def test_scientific_payload_rejects_zero_jacobian_audit_mapping_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    arguments = _valid_scientific_payload_arguments(tmp_path, monkeypatch)
    audits = arguments["integrity"]["zero_jacobian_classifier"]
    if mutation == "missing_seed":
        audits.pop("3")
    else:
        audits["2"]["gradients_none"][0] = False

    with pytest.raises(ValueError, match="zero-Jacobian classifier"):
        _MODULE.scientific_payload(**arguments)


@pytest.mark.parametrize("mutation", ["missing", "extra", "digest_relationship"])
def test_scientific_payload_independently_rejects_unbound_execution_audit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    """Catches scientific serialization of unchecked caller-supplied execution metadata."""
    arguments = _valid_scientific_payload_arguments(tmp_path, monkeypatch)
    if mutation == "missing":
        arguments["execution_audit"].pop("diagnostic_sha256")
    elif mutation == "extra":
        arguments["execution_audit"]["unchecked"] = True
    else:
        arguments["manifest_audit"]["source"]["files"][
            "scripts/diagnose_pass200_rsta_stage_a.py"
        ] = "0" * 64

    with pytest.raises(ValueError, match="execution audit fields|diagnostic SHA-256"):
        _MODULE.scientific_payload(**arguments)


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("primary_rows", "norm_b", 0.0, "norm"),
        ("primary_rows", "norm_z", 1.01, "unit"),
        ("primary_rows", "radial_fraction_b", 0.002, "radial"),
        ("primary_rows", "head_self_desc_gap", 0.001, "head"),
        ("primary_rows", "a_self", 2.0, "cosine"),
        ("primary_rows", "rho", 1.1, "rho"),
        ("primary_rows", "receiver_id", "forged", "receiver|registered"),
        ("primary_rows", "tensor_sha256", "not-a-sha", "SHA-256"),
        ("integrity", "rotation_vector", 0.001, "rotation"),
    ],
)
def test_scientific_payload_rejects_every_mandatory_invalid_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
    field: str,
    value: Any,
    message: str,
) -> None:
    """Catches accepting an INVALID row, forged role, hash, or failed rotation gate."""
    arguments = _valid_scientific_payload_arguments(tmp_path, monkeypatch)
    if target == "primary_rows":
        arguments[target][0][field] = value
    else:
        arguments["integrity"]["seeds"][0]["rotation"]["vector_residuals"]["z"] = value
    with pytest.raises(ValueError, match=message):
        _MODULE.scientific_payload(**arguments)


@pytest.mark.parametrize(
    "defect",
    [
        "duplicate_support",
        "duplicate_foreign",
        "support_in_batch",
        "receiver_index",
        "batch_tensor_hash",
        "alternate_tensor",
        "seed_batch_matrix",
        "seed_transform_hash",
        "parameter_order",
        "repeatability_hash",
        "repeatability_nonhex",
        "rotation_scalar",
        "rotation_negative",
        "dense_residual",
        "dense_tolerance",
        "bn_residual",
        "bn_buffers",
    ],
)
def test_scientific_payload_binds_ids_hashes_roles_and_integrity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, defect: str
) -> None:
    """Catches internally consistent-looking rows detached from the frozen panel/model audit."""
    arguments = _valid_scientific_payload_arguments(tmp_path, monkeypatch)
    row = arguments["primary_rows"][0]
    if defect == "duplicate_support":
        row["support_ids"][1] = row["support_ids"][0]
    elif defect == "duplicate_foreign":
        row["foreign_ids"][1] = row["foreign_ids"][0]
    elif defect == "support_in_batch":
        row["support_ids"][0] = row["batch_ids"][1]
    elif defect == "receiver_index":
        row["receiver_index"] = (row["receiver_index"] + 1) % 180
    elif defect == "batch_tensor_hash":
        row["batch_tensor_sha256"][row["receiver_index"]] = "0" * 64
    elif defect == "alternate_tensor":
        alternate = arguments["alternate_rows"][0]
        changed = "0" * 64
        alternate["tensor_sha256"] = changed
        alternate["batch_tensor_sha256"][alternate["receiver_index"]] = changed
    elif defect == "seed_batch_matrix":
        arguments["seed_audits"][0]["primary_batch_ids"][0] = list(reversed(row["batch_ids"]))
    elif defect == "seed_transform_hash":
        arguments["seed_audits"][0]["transform_tensor_sha256"][row["receiver_id"]] = "0" * 64
    elif defect == "parameter_order":
        arguments["seed_audits"][1]["parameter_names"].reverse()
    elif defect == "repeatability_hash":
        arguments["integrity"]["seeds"][0]["repeatability"]["z"]["repeat_sha256"] = "f" * 64
    elif defect == "repeatability_nonhex":
        hashes = arguments["integrity"]["seeds"][0]["repeatability"]["z"]
        hashes["first_sha256"] = "g" * 64
        hashes["repeat_sha256"] = "g" * 64
    elif defect == "rotation_negative":
        arguments["integrity"]["seeds"][0]["rotation"]["vector_residuals"]["z"] = -1.0e-9
    elif defect == "dense_residual":
        arguments["integrity"]["dense_fixture"]["max_jacobian_residual"] = 2.0e-8
    elif defect == "dense_tolerance":
        arguments["integrity"]["dense_fixture"]["jacobian_tolerance"] = 2.0e-8
    elif defect == "bn_residual":
        arguments["integrity"]["bn_fixture"]["max_gradient_residual"] = 2.0e-6
    elif defect == "bn_buffers":
        arguments["integrity"]["bn_fixture"]["buffers_unchanged"] = False
    else:
        arguments["integrity"]["seeds"][0]["rotation"]["statistic_differences"]["Delta"] = 3.0e-4
    with pytest.raises(ValueError):
        _MODULE.scientific_payload(**arguments)


def _unit_numpy(values: np.ndarray) -> np.ndarray:
    return values / np.linalg.norm(values, axis=1, keepdims=True)


def test_score_rsta_batch_computes_all_controls_and_complete_receiver_audits() -> None:
    """Catches a scientific loop that omits rows, controls, IDs, or head-kernel evidence."""
    generator = np.random.Generator(np.random.PCG64(314))
    batch_size = 180
    dimension = 3
    receiver_indices = tuple(range(8))
    z = _unit_numpy(generator.standard_normal((batch_size, dimension)))

    def tangent(values: np.ndarray) -> np.ndarray:
        return values - z * np.sum(z * values, axis=1, keepdims=True)

    dbar = tangent(generator.standard_normal((batch_size, dimension)))
    b = tangent(generator.standard_normal((batch_size, dimension)))
    s = tangent(generator.standard_normal((batch_size, dimension)))[:8]
    raw_norms = np.linspace(0.8, 1.5, batch_size)
    raw_head = z * raw_norms[:, None]
    prehead = generator.standard_normal((batch_size, 2))
    fields = {
        "z": torch.tensor(z, dtype=torch.float64),
        "dbar": torch.tensor(dbar, dtype=torch.float64),
        "batch_motion": torch.tensor(b, dtype=torch.float64),
        "self_motion": torch.tensor(s, dtype=torch.float64),
        "receiver_indices": receiver_indices,
    }
    labels = tuple(range(8))
    receiver_ids = tuple(f"receiver-{index}" for index in range(8))
    supports_by_label = {
        label: (
            (f"support-{label}-a", f"support-{label}-b"),
            _unit_numpy(generator.standard_normal((2, dimension))),
        )
        for label in labels
    }
    foreign_ids = tuple(f"foreign-{index}" for index in range(40))
    foreign_labels = tuple(range(100, 140))
    foreign_descriptors = _unit_numpy(generator.standard_normal((40, dimension)))
    batch_ids = tuple(receiver_ids) + tuple(f"distractor-{index}" for index in range(172))
    tensor_hashes = {example_id: f"{index:064x}" for index, example_id in enumerate(batch_ids)}

    rows = _MODULE.score_rsta_batch(
        seed=0,
        panel="primary",
        batch_index=0,
        receiver_indices=receiver_indices,
        receiver_ids=receiver_ids,
        receiver_labels=labels,
        batch_ids=batch_ids,
        tensor_hashes=tensor_hashes,
        fields=fields,
        supports_by_label=supports_by_label,
        foreign_ids=foreign_ids,
        foreign_labels=foreign_labels,
        foreign_descriptors=foreign_descriptors,
        prehead_features=prehead,
        raw_head_outputs=raw_head,
    )

    assert len(rows) == 8
    assert [row["receiver_id"] for row in rows] == list(receiver_ids)
    assert all(set(row) >= _MODULE.RECEIVER_AUDIT_FIELDS for row in rows)
    assert all(len(row["foreign_ids"]) == 32 for row in rows)
    assert all(len(row["support_cosines"]) == 34 for row in rows)
    assert all(abs(row["head_self_desc_gap"]) <= 1.0e-5 for row in rows)
    assert rows[0]["delta"] == pytest.approx(rows[0]["a_self"] - rows[0]["a_batch"])
    assert rows[0]["self_minus_desc"] == pytest.approx(rows[0]["a_self"] - rows[0]["a_desc"])


def test_exact_fields_and_registered_statistics_rotate_with_affine_head() -> None:
    """Catches a wrong Q convention for head weights or field cotangents."""
    model, images, _ = _dense_affine_fixture()
    rotation = torch.tensor(_MODULE.construct_rotation(3, seed=200), dtype=torch.float64)
    rotated_model = deepcopy(model)
    with torch.no_grad():
        rotated_model.projection.weight.copy_(rotation @ model.projection.weight)
        rotated_model.projection.bias.copy_(rotation @ model.projection.bias)
    labels = torch.tensor([0, 1, 0], dtype=torch.int64)
    proxy_labels = torch.tensor([0, 1], dtype=torch.int64)
    proxies = torch.tensor([[0.8, -0.2, 0.5], [-0.4, 0.7, 0.1]], dtype=torch.float64)
    rotated_proxies = proxies @ rotation.T
    original = _MODULE.exact_contextual_rsta_fields(
        model,
        images,
        labels,
        proxies,
        proxy_labels,
        alpha=32.0,
        delta=0.1,
        receiver_indices=(0, 1, 2),
        expected_batch_size=3,
        expected_dimension=3,
    )
    rotated = _MODULE.exact_contextual_rsta_fields(
        rotated_model,
        images,
        labels,
        rotated_proxies,
        proxy_labels,
        alpha=32.0,
        delta=0.1,
        receiver_indices=(0, 1, 2),
        expected_batch_size=3,
        expected_dimension=3,
    )
    receiver = 0
    generator = np.random.Generator(np.random.PCG64(18))
    positive_supports = _unit_numpy(generator.standard_normal((2, 3)))
    foreign_supports = _unit_numpy(generator.standard_normal((32, 3)))
    q = _MODULE.smooth_margin_gradient(
        original["z"][receiver].detach().numpy(), positive_supports, foreign_supports
    )
    rotated_q = _MODULE.smooth_margin_gradient(
        rotated["z"][receiver].detach().numpy(),
        positive_supports @ rotation.numpy().T,
        foreign_supports @ rotation.numpy().T,
    )

    def statistics(fields: dict[str, Any], target: np.ndarray) -> dict[str, float]:
        z_value = fields["z"][receiver].detach().numpy()
        dbar_value = _MODULE.tangent_projection(fields["dbar"][receiver].detach().numpy(), z_value)
        b_value = _MODULE.tangent_projection(
            fields["batch_motion"][receiver].detach().numpy(), z_value
        )
        s_value = _MODULE.tangent_projection(
            fields["self_motion"][receiver].detach().numpy(), z_value
        )
        a_self = _MODULE.cosine_similarity(s_value, target)
        a_batch = _MODULE.cosine_similarity(b_value, target)
        return {
            "A_self": a_self,
            "A_batch": a_batch,
            "Delta": a_self - a_batch,
            "A_desc": _MODULE.cosine_similarity(dbar_value, target),
            "rho": float(
                np.linalg.norm(
                    b_value / np.linalg.norm(b_value)
                    - s_value
                    / np.linalg.norm(s_value)
                    * np.dot(b_value / np.linalg.norm(b_value), s_value / np.linalg.norm(s_value))
                )
            ),
            "log_ratio": float(np.log(np.linalg.norm(b_value) / np.linalg.norm(s_value))),
            "cos_b_s": _MODULE.cosine_similarity(b_value, s_value),
        }

    original_vectors = {
        "z": original["z"][receiver].detach().numpy(),
        "dbar": original["dbar"][receiver].detach().numpy(),
        "b": original["batch_motion"][receiver].detach().numpy(),
        "s": original["self_motion"][receiver].detach().numpy(),
        "q": q,
    }
    rotated_vectors = {
        "z": rotated["z"][receiver].detach().numpy(),
        "dbar": rotated["dbar"][receiver].detach().numpy(),
        "b": rotated["batch_motion"][receiver].detach().numpy(),
        "s": rotated["self_motion"][receiver].detach().numpy(),
        "q": rotated_q,
    }
    audit = _MODULE.check_rotation(
        original_vectors,
        rotated_vectors,
        statistics(original, q),
        statistics(rotated, rotated_q),
        rotation.numpy(),
    )
    assert max(audit["vector_residuals"].values()) < 1.0e-10
    assert max(audit["statistic_differences"].values()) < 1.0e-10


def _tiny_scientific_bound(
    *,
    seed: int,
    clean: np.ndarray,
    labels: list[int],
    example_ids: list[str],
    source_paths: tuple[str, ...],
    proxies: np.ndarray,
    proxy_labels: tuple[int, ...],
) -> Any:
    arrays = {
        "train_embeddings": _MODULE._readonly_array(clean, dtype=np.float32),
        "train_labels": _MODULE._readonly_array(labels, dtype=np.int64),
        "train_example_ids": _MODULE._readonly_array(np.asarray(example_ids)),
        "train_source_paths": _MODULE._readonly_array(np.asarray(source_paths)),
        "train_row_indices": _MODULE._readonly_array(range(len(example_ids)), dtype=np.int64),
        "proxies": _MODULE._readonly_array(proxies, dtype=np.float32),
        "proxy_labels": _MODULE._readonly_array(proxy_labels, dtype=np.int64),
    }
    checkpoint_bytes = f"synthetic-checkpoint-{seed}".encode()
    return _MODULE.TrainingOnlySeedInput(
        seed=seed,
        **arrays,
        alpha=32.0,
        delta=0.1,
        official_recall_at_1=0.9,
        checkpoint_bytes=checkpoint_bytes,
        checkpoint_sha256=hashlib.sha256(checkpoint_bytes).hexdigest(),
        training_array_sha256={
            name: _MODULE._framed_array_sha256(name, value) for name, value in arrays.items()
        },
        config={
            "batch_size": 180,
            "embedding_dimensions": 3,
            "proxy_anchor_alpha": 32.0,
            "proxy_anchor_delta": 0.1,
        },
        artifact_binding={
            "artifacts": {"checkpoint_pt": {"sha256": f"{seed + 1}" * 64}}
        },
    )


def _tiny_validated_receipt() -> Any:
    return _MODULE.ValidatedBindingReceipt(
        sha256=_MODULE._HISTORICAL_RECEIPT_SHA256,
        producer_commit=_MODULE._HISTORICAL_PRODUCER_COMMIT,
        historical_manifest_sha256=_MODULE._HISTORICAL_MANIFEST_SHA256,
        historical_source_revision=_MODULE._HISTORICAL_SOURCE_REVISION,
        historical_diagnostic_sha256=_MODULE._HISTORICAL_DIAGNOSTIC_SHA256,
        seeds=tuple(
            _MODULE.ReceiptSeed(
                seed=seed,
                artifacts={},
                official_recall_at_1=0.9,
                train_row_count=0,
                train_identity_count=0,
                train_example_id_order_sha256="0" * 64,
                train_label_order_sha256="0" * 64,
                train_source_order_sha256="0" * 64,
                train_source_export_sha256="0" * 64,
            )
            for seed in range(4)
        ),
    )


def test_scientific_cli_executes_exact_four_seed_pipeline_and_writes_atomic_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a schema-only implementation with no executable four-seed scientific path."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    example_ids, raw_labels = _selection_fixture()
    labels = [
        label if index < 210 else 1_000 + (index - 210) // 2
        for index, label in enumerate(raw_labels)
    ]
    generator = np.random.Generator(np.random.PCG64(991))
    clean = _unit_numpy(generator.standard_normal((len(example_ids), 3))).astype(np.float32)
    proxy_labels = tuple(sorted(set(labels)))
    proxies = _unit_numpy(generator.standard_normal((len(proxy_labels), 3))).astype(np.float32)
    source_paths = tuple(f"/synthetic/{example_id}.jpg" for example_id in example_ids)

    def bound_for(seed: int) -> Any:
        return _tiny_scientific_bound(
            seed=seed,
            clean=clean,
            labels=labels,
            example_ids=example_ids,
            source_paths=source_paths,
            proxies=np.frombuffer(proxies.tobytes(), dtype=np.float32).reshape(proxies.shape),
            proxy_labels=proxy_labels,
        )

    bound_calls: list[int] = []

    def bound_loader(_entry: Any, receipt_seed: Any, **_kwargs: Any) -> Any:
        bound_calls.append(receipt_seed.seed)
        return bound_for(receipt_seed.seed)

    def cache_builder(bound: Any, ordered_ids: Any, **_kwargs: Any) -> Any:
        sources = {value: value for value in ordered_ids}

        def transform(example_id: str) -> torch.Tensor:
            seed = _MODULE.domain_seed("tiny-rsta-transform", example_id)
            values = np.random.Generator(np.random.PCG64(seed)).standard_normal(2)
            return torch.tensor(values, dtype=torch.float32)

        return _MODULE.cache_deterministic_transforms(ordered_ids, sources, transform=transform)

    class TinyOfficialHead(torch.nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            torch.manual_seed(seed + 20)
            self.model = torch.nn.Module()
            self.model.gmp = torch.nn.AdaptiveMaxPool2d(1)
            self.model.embedding = torch.nn.Linear(2, 3)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(self.model.embedding(values), dim=-1)

    rotation_calls: list[int] = []
    integrity_score_events: list[str] = []

    original_configure = _MODULE.configure_deterministic_process

    def configure_first() -> dict[str, Any]:
        integrity_score_events.append("configure")
        return original_configure()

    monkeypatch.setattr(_MODULE, "configure_deterministic_process", configure_first)

    def audit_before_scoring(path: Path) -> dict[str, Any]:
        integrity_score_events.append("execution-audit")
        manifest = _MODULE.load_strict_json(path)
        return _MODULE.build_execution_audit(manifest, manifest_path=path)

    def rotation_auditor(*args: Any, seed: int, **kwargs: Any) -> dict[str, Any]:
        rotation_calls.append(seed)
        integrity_score_events.append(f"rotation-{seed}")
        return _MODULE._default_rotation_auditor(*args, seed=seed, **kwargs)

    original_score = _MODULE.score_rsta_batch
    original_decide = _MODULE.decide_stage_a

    def score_after_integrity(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        integrity_score_events.append(f"score-{kwargs['seed']}")
        return original_score(*args, **kwargs)

    monkeypatch.setattr(_MODULE, "score_rsta_batch", score_after_integrity)

    def decide_after_execution(*args: Any, **kwargs: Any) -> dict[str, Any]:
        integrity_score_events.append("decision")
        return original_decide(*args, **kwargs)

    monkeypatch.setattr(_MODULE, "decide_stage_a", decide_after_execution)

    fixture_audit = {
        "dense_fixture": {
            "passed": True,
            "max_jacobian_residual": 0.0,
            "max_finite_difference_residual": 0.0,
            "jacobian_tolerance": 1.0e-8,
            "finite_difference_tolerance": 1.0e-6,
        },
        "bn_fixture": {
            "passed": True,
            "max_output_residual": 0.0,
            "max_gradient_residual": 0.0,
            "tolerance": 1.0e-6,
            "buffers_unchanged": True,
        },
    }
    manifest_path, _ = _synthetic_execution_manifest(
        tmp_path,
        seeds={str(seed): {} for seed in range(4)},
    )
    _bind_synthetic_executing_diagnostic(monkeypatch, manifest_path=manifest_path)
    output = tmp_path / "scientific.json"
    validated: list[Path] = []
    zero_jacobian_calls: list[int] = []

    def zero_jacobian_auditor(_model: Any, images: torch.Tensor) -> dict[str, Any]:
        zero_jacobian_calls.append(int(images.shape[0]))
        return _valid_zero_jacobian_audit()

    invalid_output = tmp_path / "invalid-scientific.json"
    with pytest.raises(ValueError, match="zero-Jacobian classifier"):
        _MODULE.main(
            [
                "--manifest",
                str(manifest_path),
                "--binding-receipt",
                str(tmp_path / "receipt.json"),
                "--output",
                str(invalid_output),
                "--scientific",
            ],
            expected_dimension=3,
            receipt_validator=lambda manifest_path, _receipt_path: validated.append(manifest_path)
            or _tiny_validated_receipt(),
            execution_source_validator=audit_before_scoring,
            bound_loader=bound_loader,
            cache_builder=cache_builder,
            model_loader=lambda bound: TinyOfficialHead(bound.seed).train(),
            fixture_runner=lambda: fixture_audit,
            deterministic_pool_auditor=_valid_global_max_audit,
            zero_jacobian_auditor=lambda _model, _images: {},
            rotation_auditor=rotation_auditor,
            head_name="model.embedding",
            expected_head_in_features=2,
        )
    assert not any(event.startswith("score-") for event in integrity_score_events)
    assert not invalid_output.exists()
    validated.clear()
    bound_calls.clear()
    integrity_score_events.clear()

    _MODULE.main(
        [
            "--manifest",
            str(manifest_path),
            "--binding-receipt",
            str(tmp_path / "receipt.json"),
            "--output",
            str(output),
            "--scientific",
        ],
        expected_dimension=3,
        receipt_validator=lambda manifest_path, _receipt_path: validated.append(manifest_path)
        or _tiny_validated_receipt(),
        execution_source_validator=audit_before_scoring,
        bound_loader=bound_loader,
        cache_builder=cache_builder,
        model_loader=lambda bound: TinyOfficialHead(bound.seed).train(),
        fixture_runner=lambda: fixture_audit,
        deterministic_pool_auditor=_valid_global_max_audit,
        zero_jacobian_auditor=zero_jacobian_auditor,
        rotation_auditor=rotation_auditor,
        head_name="model.embedding",
        expected_head_in_features=2,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert validated == [manifest_path]
    assert bound_calls == [0, 1, 2, 3]
    assert rotation_calls == [0, 1, 2, 3]
    assert zero_jacobian_calls == [180, 180, 180, 180]
    assert integrity_score_events[:2] == ["configure", "execution-audit"]
    assert integrity_score_events.index("execution-audit") < integrity_score_events.index(
        "decision"
    )
    for seed in range(4):
        assert integrity_score_events.index(f"rotation-{seed}") < integrity_score_events.index(
            f"score-{seed}"
        )
    assert result["mode"] == "scientific"
    assert len(result["rows"]["primary"]) == 4 * 64
    assert len(result["rows"]["alternate"]) == 4 * 16
    assert all(len(audit["primary_batch_ids"]) == 8 for audit in result["seed_audits"])
    assert result["exclusions"] == []
    assert result["integrity"]["deterministic_global_max"] == _valid_global_max_audit()
    assert result["integrity"]["zero_jacobian_classifier"] == {
        str(seed): _valid_zero_jacobian_audit() for seed in range(4)
    }
    assert result["execution_audit"]["diagnostic_sha256"] == result["manifest"]["source"][
        "files"
    ]["scripts/diagnose_pass200_rsta_stage_a.py"]


def test_scientific_invalid_global_max_audit_prevents_loading_scoring_and_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    output = tmp_path / "scientific.json"
    downstream_calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        downstream_calls.append("downstream")
        raise AssertionError("invalid deterministic pool audit reached science")

    monkeypatch.setattr(_MODULE, "score_rsta_batch", forbidden)
    with pytest.raises(ValueError, match="deterministic global max"):
        _MODULE.run_scientific_diagnostic(
            {"seeds": {str(seed): {} for seed in range(4)}},
            manifest_path=tmp_path / "manifest.json",
            receipt_path=tmp_path / "receipt.json",
            output_path=output,
            receipt_validator=lambda *_args: _tiny_validated_receipt(),
            execution_source_validator=lambda _path: {"validated": True},
            deterministic_pool_auditor=lambda: {},
            bound_loader=forbidden,
        )

    assert downstream_calls == []
    assert not output.exists()


@pytest.mark.parametrize(
    "modes",
    [
        (),
        ("--binding-only", "--smoke-only"),
        ("--binding-only", "--scientific"),
        ("--smoke-only", "--scientific"),
        ("--binding-only", "--smoke-only", "--scientific"),
    ],
)
def test_cli_requires_exactly_one_execution_mode(tmp_path: Path, modes: tuple[str, ...]) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit):
        _MODULE.main(
            [
                "--manifest",
                str(manifest),
                "--binding-receipt",
                str(tmp_path / "receipt.json"),
                "--output",
                str(tmp_path / "out.json"),
                *modes,
            ]
        )


def test_cli_rejects_removed_binding_mode_and_requires_exact_receipt_argument(
    tmp_path: Path,
) -> None:
    """Catches exposing receipt creation or permitting an implicit receipt fallback."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    output = tmp_path / "out.json"
    with pytest.raises(SystemExit):
        _MODULE.main(
            ["--manifest", str(manifest), "--output", str(output), "--binding-only"]
        )
    with pytest.raises(SystemExit):
        _MODULE.main(
            ["--manifest", str(manifest), "--output", str(output), "--smoke-only"]
        )
    assert not output.exists()


def test_smoke_cli_executes_only_first_batch_integrity_without_candidate_values(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    example_ids, raw_labels = _selection_fixture()
    labels = [
        label if index < 210 else 1_000 + (index - 210) // 2
        for index, label in enumerate(raw_labels)
    ]
    generator = np.random.Generator(np.random.PCG64(1200))
    clean = _unit_numpy(generator.standard_normal((len(example_ids), 3))).astype(np.float32)
    proxy_labels = tuple(sorted(set(labels)))
    proxies = _unit_numpy(generator.standard_normal((len(proxy_labels), 3))).astype(np.float32)
    source_paths = tuple(f"/synthetic/{example_id}.jpg" for example_id in example_ids)
    bound = _tiny_scientific_bound(
        seed=0,
        clean=clean,
        labels=labels,
        example_ids=example_ids,
        source_paths=source_paths,
        proxies=proxies,
        proxy_labels=proxy_labels,
    )
    calls: list[str] = []

    def bound_loader(_entry: Any, receipt_seed: Any, **_kwargs: Any) -> Any:
        calls.append(f"bind-{receipt_seed.seed}")
        assert receipt_seed.seed == 0
        return bound

    def cache_builder(_bound: Any, ordered_ids: Any, **_kwargs: Any) -> Any:
        calls.append("cache")
        sources = {value: value for value in ordered_ids}
        return _MODULE.cache_deterministic_transforms(
            ordered_ids,
            sources,
            transform=lambda example_id: torch.tensor(
                np.random.Generator(
                    np.random.PCG64(_MODULE.domain_seed("tiny-smoke-transform", example_id))
                ).standard_normal(2),
                dtype=torch.float32,
            ),
        )

    class TinyOfficialHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            torch.manual_seed(1201)
            self.model = torch.nn.Module()
            self.model.gmp = torch.nn.AdaptiveMaxPool2d(1)
            self.model.embedding = torch.nn.Linear(2, 3)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(self.model.embedding(values), dim=-1)

    fixture_audit = {
        "dense_fixture": {
            "passed": True,
            "max_jacobian_residual": 0.0,
            "max_finite_difference_residual": 0.0,
            "jacobian_tolerance": 1.0e-8,
            "finite_difference_tolerance": 1.0e-6,
        },
        "bn_fixture": {
            "passed": True,
            "max_output_residual": 0.0,
            "max_gradient_residual": 0.0,
            "tolerance": 1.0e-6,
            "buffers_unchanged": True,
        },
    }
    manifest_path, _ = _synthetic_execution_manifest(
        tmp_path,
        seeds={str(seed): {} for seed in range(4)},
    )
    _bind_synthetic_executing_diagnostic(monkeypatch, manifest_path=manifest_path)
    output = tmp_path / "smoke.json"

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("smoke touched candidate scoring")

    for name in ("score_rsta_batch", "decide_stage_a", "joint_bootstrap", "scientific_payload"):
        monkeypatch.setattr(_MODULE, name, forbidden)

    _MODULE.main(
        [
            "--manifest",
            str(manifest_path),
            "--binding-receipt",
            str(tmp_path / "receipt.json"),
            "--output",
            str(output),
            "--smoke-only",
        ],
        expected_dimension=3,
        receipt_validator=lambda _manifest_path, _receipt_path: calls.append("manifest")
        or _tiny_validated_receipt(),
        execution_source_validator=lambda path: _MODULE.build_execution_audit(
            _MODULE.load_strict_json(path), manifest_path=path
        ),
        bound_loader=bound_loader,
        cache_builder=cache_builder,
        model_loader=lambda _bound: TinyOfficialHead().train(),
        fixture_runner=lambda: fixture_audit,
        deterministic_pool_auditor=_valid_global_max_audit,
        zero_jacobian_auditor=lambda _model, _images: _valid_zero_jacobian_audit(),
        head_name="model.embedding",
        expected_head_in_features=2,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    assert calls == ["manifest", "bind-0", "cache"]
    assert result["mode"] == "integrity_smoke"
    assert result["candidate_values_computed"] is False
    assert result["stage_a_verdict"] == "NOT_COMPUTED"
    assert result["uses_test_data"] == "artifact_binding_only"
    assert "rows" not in result and "aggregation" not in result and "bootstrap" not in result
    assert result["integrity"]["seed"] == 0
    assert result["integrity"]["deterministic_global_max"] == _valid_global_max_audit()
    assert result["integrity"]["zero_jacobian_classifier"] == _valid_zero_jacobian_audit()
    assert set(result["integrity"]["repeatability"]) == {"z", "dbar", "b", "s"}
    assert result["integrity"]["adjoint_relative_error"] <= 5.0e-4
    assert max(result["integrity"]["rotation"]["vector_residuals"].values()) <= 5.0e-4
    assert len(result["binding"]["first_batch_id_sha256"]) == 64
    assert len(result["binding"]["transform_tensor_set_sha256"]) == 64
    assert result["binding"]["checkpoint_sha256"] == "1" * 64
    assert result["execution_audit"]["diagnostic_path"] == (
        "scripts/diagnose_pass200_rsta_stage_a.py"
    )


@pytest.mark.parametrize(
    "binding",
    [
        {},
        {"checkpoint_sha256": "1" * 64},
        {"artifacts": {}},
        {"artifacts": {"checkpoint_pt": {}}},
        {"artifacts": {"checkpoint_pt": {"sha256": "A" * 64}}},
        {"artifacts": {"checkpoint_pt": {"sha256": "1" * 63}}},
    ],
)
def test_smoke_checkpoint_digest_requires_exact_nested_lowercase_sha256(
    binding: dict[str, Any],
) -> None:
    with pytest.raises(ValueError, match="nested checkpoint SHA-256"):
        _MODULE._bound_checkpoint_sha256(binding)


@pytest.mark.parametrize(
    "contents",
    [
        '{"schema": 1, "schema": 1}',
        '{"schema": NaN}',
        '{"schema": Infinity}',
        '{"schema": 1e999}',
    ],
)
def test_strict_receipt_json_rejects_ambiguous_or_nonfinite_bytes(
    tmp_path: Path,
    contents: str,
) -> None:
    """Catches accepting JSON whose meaning is not byte-unique and finite."""
    path = tmp_path / "receipt.json"
    path.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate|nonfinite"):
        _MODULE.load_strict_json(path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("extra_top", "receipt fields"),
        ("boolean_batch", "source export batch size"),
        ("float_seed", "seed"),
        ("nonzero_difference", "descriptor difference"),
        ("integer_tolerance", "descriptor atol"),
        ("integer_difference", "descriptor difference"),
        ("integer_recall", "official recall"),
        ("integer_prehead_difference", "prehead difference"),
        ("missing_artifact_key", "artifact keys"),
    ],
)
def test_historical_receipt_schema_rejects_recursive_shape_and_exact_type_drift(
    mutation: str,
    message: str,
) -> None:
    """Catches schema drift that ordinary equality and isinstance(int) can miss."""
    payload = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "docs"
            / "pass200_rsta_binding_receipt_d6270a9.json"
        ).read_text(encoding="utf-8")
    )
    if mutation == "extra_top":
        payload["unchecked"] = True
    elif mutation == "boolean_batch":
        payload["binding"]["source_export_batch_size"] = True
    elif mutation == "float_seed":
        payload["binding"]["seeds"][0]["seed"] = 0.0
    elif mutation == "nonzero_difference":
        payload["binding"]["seeds"][0]["artifact_binding"]["current_source_export"][
            "train"
        ]["max_abs_descriptor_difference"] = 1.0e-12
    elif mutation == "integer_tolerance":
        payload["binding"]["descriptor_atol"] = 0
    elif mutation == "integer_difference":
        payload["binding"]["seeds"][0]["artifact_binding"]["current_source_export"][
            "train"
        ]["max_abs_descriptor_difference"] = 0
    elif mutation == "integer_recall":
        payload["binding"]["seeds"][0]["official_recall_at_1"] = 1
    elif mutation == "integer_prehead_difference":
        payload["binding"]["seeds"][0]["artifact_binding"]["prehead_reconstruction"][
            "train"
        ]["max_abs_difference"] = 0
    else:
        payload["binding"]["seeds"][0]["artifact_binding"]["artifacts"].pop(
            "gallery_npz"
        )

    with pytest.raises(ValueError, match=message):
        _MODULE._validate_historical_receipt_schema(payload)


def test_historical_receipt_schema_returns_hash_only_immutable_seed_records() -> None:
    """Catches leaking historical arrays or mutable receipt mappings into science."""
    root = Path(__file__).resolve().parents[1]
    payload = _MODULE.load_strict_json(
        root / "docs" / "pass200_rsta_binding_receipt_d6270a9.json"
    )

    receipt = _MODULE._validate_historical_receipt_schema(payload)

    assert tuple(seed.seed for seed in receipt.seeds) == (0, 1, 2, 3)
    assert receipt.sha256 == _MODULE._HISTORICAL_RECEIPT_SHA256
    assert receipt.producer_commit == "d6270a94f14f5e0b4f4a3eeaa23f3f66d9bfaa54"
    assert receipt.seeds[0].train_source_export_sha256 == (
        "5341f5a28fd6f105d8beaf022c336b16406f560e3db74fe79a0066da0ef409b7"
    )
    with pytest.raises(TypeError):
        receipt.seeds[0].artifacts["unchecked"] = {"path": "x", "sha256": "0" * 64}
    assert not any(
        "embedding" in name or "query" in name or "gallery" in name or "prehead" in name
        for name in receipt.seeds[0].__dataclass_fields__
    )


@pytest.mark.parametrize(
    "mutation",
    ["missing", "extra", "path", "sha256", "commit"],
)
def test_amended_manifest_requires_exact_deterministic_pool_amendment(
    mutation: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "docs" / "pass200_rsta_receipt_stage_a_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["deterministic_pool_amendment"] = {
        "path": "docs/pass200_rsta_deterministic_global_max_amendment_2026-08-09.md",
        "sha256": "6b2ffed724f0056b011831bb74997cb3e8d50f83304448805b119f6a3d78b361",
        "commit": "db29ab7bb6478cfef57eccbad142f93d2f805f7f",
    }
    manifest["zero_jacobian_classifier_amendment"] = {
        "path": "docs/pass200_rsta_zero_jacobian_classifier_amendment_2026-08-09.md",
        "sha256": "4b981efd3893436e1a4da09568c3cf167d7beeeb8fd637979b5869588c956ade",
        "commit": "85e8f983053f3839e5bbb2bb11563380e6b77919",
    }
    if mutation == "missing":
        manifest.pop("deterministic_pool_amendment")
    elif mutation == "extra":
        manifest["deterministic_pool_amendment"]["unchecked"] = True
    else:
        manifest["deterministic_pool_amendment"][mutation] = "0" * (
            64 if mutation == "sha256" else 40
        )

    with pytest.raises(ValueError, match="deterministic_pool_amendment|manifest fields"):
        _MODULE._validate_amended_manifest_schema(manifest)


@pytest.mark.parametrize("mutation", ["missing", "extra", "path", "sha256", "commit"])
def test_amended_manifest_requires_exact_zero_jacobian_classifier_amendment(
    mutation: str,
) -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads(
        (root / "docs" / "pass200_rsta_receipt_stage_a_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["deterministic_pool_amendment"] = {
        "path": "docs/pass200_rsta_deterministic_global_max_amendment_2026-08-09.md",
        "sha256": "6b2ffed724f0056b011831bb74997cb3e8d50f83304448805b119f6a3d78b361",
        "commit": "db29ab7bb6478cfef57eccbad142f93d2f805f7f",
    }
    manifest["zero_jacobian_classifier_amendment"] = {
        "path": "docs/pass200_rsta_zero_jacobian_classifier_amendment_2026-08-09.md",
        "sha256": "4b981efd3893436e1a4da09568c3cf167d7beeeb8fd637979b5869588c956ade",
        "commit": "85e8f983053f3839e5bbb2bb11563380e6b77919",
    }
    if mutation == "missing":
        manifest.pop("zero_jacobian_classifier_amendment")
    elif mutation == "extra":
        manifest["zero_jacobian_classifier_amendment"]["unchecked"] = True
    else:
        manifest["zero_jacobian_classifier_amendment"][mutation] = "0" * (
            64 if mutation == "sha256" else 40
        )

    with pytest.raises(ValueError, match="zero_jacobian_classifier_amendment|manifest fields"):
        _MODULE._validate_amended_manifest_schema(manifest)


def test_historical_receipt_rejects_nonliteral_path_before_receipt_or_semantic_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches a manifest-selected fallback receipt reaching parsing or model state."""
    calls: list[str] = []
    monkeypatch.setattr(_MODULE, "load_strict_json", lambda _path: calls.append("json") or {})
    monkeypatch.setattr(_MODULE, "_git_blob", lambda *_args: calls.append("git") or b"")
    wrong_receipt = tmp_path / "receipt.json"
    wrong_receipt.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="literal historical receipt path"):
        _MODULE.validate_historical_binding_receipt(
            tmp_path / "docs" / "manifest.json",
            wrong_receipt,
        )

    assert calls == []


def test_historical_receipt_rejects_digest_before_manifest_or_git_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches parsing a manifest before authenticating the sole receipt bytes."""
    docs = tmp_path / "docs"
    docs.mkdir()
    manifest = docs / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    receipt = docs / "pass200_rsta_binding_receipt_d6270a9.json"
    receipt.write_text("{}", encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(
        _MODULE,
        "sha256_file",
        lambda _path: calls.append("hash") or "0" * 64,
    )
    monkeypatch.setattr(_MODULE, "load_strict_json", lambda _path: calls.append("json") or {})
    monkeypatch.setattr(_MODULE, "_git_blob", lambda *_args: calls.append("git") or b"")

    with pytest.raises(ValueError, match="receipt SHA-256 mismatch"):
        _MODULE.validate_historical_binding_receipt(manifest, receipt)

    assert calls == ["hash"]


def test_invalid_amended_manifest_fails_before_receipt_hash_or_git_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches touching historical authority before strict current manifest parsing."""
    docs = tmp_path / "docs"
    docs.mkdir()
    manifest = docs / "manifest.json"
    manifest.write_text('{"schema_version": 1, "schema_version": 1}', encoding="utf-8")
    receipt = docs / "pass200_rsta_binding_receipt_d6270a9.json"
    receipt.write_text("{}", encoding="utf-8")
    accesses: list[str] = []
    monkeypatch.setattr(
        _MODULE,
        "sha256_file",
        lambda _path: accesses.append("hash") or _MODULE._HISTORICAL_RECEIPT_SHA256,
    )
    monkeypatch.setattr(
        _MODULE,
        "_git_blob",
        lambda *_args: accesses.append("git") or b"",
    )

    with pytest.raises(ValueError, match="duplicate"):
        _MODULE.validate_historical_binding_receipt(manifest, receipt)

    assert accesses == ["hash"]


def test_scientific_source_authenticates_deterministic_pool_amendment_bytes_and_blob(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository = tmp_path
    docs = repository / "docs"
    scripts = repository / "scripts"
    docs.mkdir()
    scripts.mkdir()
    manifest_path = docs / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    diagnostic = scripts / "diagnose_pass200_rsta_stage_a.py"
    diagnostic.write_text("BOUND = True\n", encoding="utf-8")
    references = {
        "base_preregistration": {"path": "docs/base.md", "sha256": "1" * 64},
        "amendment": {
            "path": _MODULE._AMENDMENT_PATH,
            "sha256": _MODULE._AMENDMENT_SHA256,
            "commit": _MODULE._AMENDMENT_COMMIT,
        },
        "deterministic_pool_amendment": {
            "path": _MODULE._DETERMINISTIC_POOL_AMENDMENT_PATH,
            "sha256": _MODULE._DETERMINISTIC_POOL_AMENDMENT_SHA256,
            "commit": _MODULE._DETERMINISTIC_POOL_AMENDMENT_COMMIT,
        },
        "zero_jacobian_classifier_amendment": {
            "path": "docs/pass200_rsta_zero_jacobian_classifier_amendment_2026-08-09.md",
            "sha256": "4b981efd3893436e1a4da09568c3cf167d7beeeb8fd637979b5869588c956ade",
            "commit": "85e8f983053f3839e5bbb2bb11563380e6b77919",
        },
        "artifact_schema": {"path": "docs/artifacts.json", "sha256": "2" * 64},
    }
    for reference in references.values():
        path = repository / reference["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("bound", encoding="utf-8")
    manifest = {
        **references,
        "current_scientific_source": {
            "git_revision": "3" * 40,
            "files": {
                "scripts/diagnose_pass200_rsta_stage_a.py": "4" * 64,
            },
        },
    }
    monkeypatch.setattr(_MODULE, "load_strict_json", lambda _path: manifest)
    monkeypatch.setattr(_MODULE, "_validate_amended_manifest_schema", lambda value: value)
    monkeypatch.setattr(_MODULE, "__file__", str(diagnostic))
    hashed_paths: list[Path] = []

    def fake_sha(path: Path) -> str:
        hashed_paths.append(path.resolve())
        if path.resolve() == diagnostic.resolve():
            return "4" * 64
        for reference in references.values():
            if path.resolve() == (repository / reference["path"]).resolve():
                return str(reference["sha256"])
        raise AssertionError(path)

    blobs: list[tuple[str, str]] = []

    def fake_blob(_repository: Path, revision: str, path_text: str) -> bytes:
        blobs.append((revision, path_text))
        digest = (
            "4b981efd3893436e1a4da09568c3cf167d7beeeb8fd637979b5869588c956ade"
            if revision == "85e8f983053f3839e5bbb2bb11563380e6b77919"
            else _MODULE._DETERMINISTIC_POOL_AMENDMENT_SHA256
            if revision == _MODULE._DETERMINISTIC_POOL_AMENDMENT_COMMIT
            else _MODULE._AMENDMENT_SHA256
            if revision == _MODULE._AMENDMENT_COMMIT
            else "4" * 64
        )
        return b"blob-" + digest.encode("ascii")

    monkeypatch.setattr(_MODULE, "sha256_file", fake_sha)
    monkeypatch.setattr(_MODULE, "_git_blob", fake_blob)
    monkeypatch.setattr(
        _MODULE.hashlib,
        "sha256",
        lambda value=b"": type("Digest", (), {"hexdigest": lambda self: value[5:].decode()})(),
    )

    class Result:
        returncode = 0
        stdout = "5" * 40 + "\n"

    monkeypatch.setattr(_MODULE.subprocess, "run", lambda *_args, **_kwargs: Result())

    _MODULE.validate_scientific_execution_source(manifest_path)

    assert (repository / _MODULE._DETERMINISTIC_POOL_AMENDMENT_PATH).resolve() in hashed_paths
    assert (
        _MODULE._DETERMINISTIC_POOL_AMENDMENT_COMMIT,
        _MODULE._DETERMINISTIC_POOL_AMENDMENT_PATH,
    ) in blobs
    assert (
        "85e8f983053f3839e5bbb2bb11563380e6b77919",
        "docs/pass200_rsta_zero_jacobian_classifier_amendment_2026-08-09.md",
    ) in blobs


def _tiny_receipt_seed(
    entry: dict[str, dict[str, str]],
    source_exporter: Callable[..., dict[str, dict[str, np.ndarray]]],
) -> Any:
    train = source_exporter()["train"]
    digest = hashlib.sha256()
    for name in ("row_indices", "labels", "example_ids", "source_paths", "embeddings"):
        value = np.asarray(train[name])
        digest.update(name.encode("ascii") + b"\0")
        if name in {"example_ids", "source_paths"}:
            for item in value.astype(str).tolist():
                digest.update(item.encode("utf-8") + b"\0")
        else:
            contiguous = np.ascontiguousarray(value)
            digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
            digest.update(str(contiguous.shape).encode("ascii") + b"\0")
            digest.update(contiguous.tobytes())
    ids = tuple(str(value) for value in train["example_ids"].tolist())
    labels = tuple(int(value) for value in train["labels"].tolist())
    sources = tuple(str(value) for value in train["source_paths"].tolist())
    return _MODULE.ReceiptSeed(
        seed=0,
        artifacts={name: dict(record) for name, record in entry.items()},
        official_recall_at_1=1.0,
        train_row_count=6,
        train_identity_count=2,
        train_example_id_order_sha256=hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        train_label_order_sha256=hashlib.sha256(
            np.asarray(labels, dtype=np.int64).tobytes(order="C")
        ).hexdigest(),
        train_source_order_sha256=hashlib.sha256("\n".join(sources).encode()).hexdigest(),
        train_source_export_sha256=digest.hexdigest(),
    )


def test_training_only_loader_hashes_every_artifact_and_materializes_only_train(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches query/gallery/prehead materialization or a skipped immutable artifact hash."""
    entry, source_exporter = _synthetic_rsta_bundle(tmp_path)
    receipt_seed = _tiny_receipt_seed(entry, source_exporter)
    hashed: list[Path] = []
    loaded_npz_sources: list[str] = []
    original_np_load = np.load

    def artifact_hasher(path: Path) -> str:
        hashed.append(path)
        return _sha256_file(path)

    def guarded_np_load(source: Any, *args: Any, **kwargs: Any) -> Any:
        loaded_npz_sources.append(type(source).__name__)
        assert not isinstance(source, (str, Path))
        return original_np_load(source, *args, **kwargs)

    monkeypatch.setattr(np, "load", guarded_np_load)
    for name in ("_export_current_source", "_load_digest_bound_packs", "load_bound_seed"):
        monkeypatch.setattr(
            _MODULE,
            name,
            lambda *_args, _name=name, **_kwargs: (_ for _ in ()).throw(
                AssertionError(f"legacy loader reached: {_name}")
            ),
        )

    bound = _MODULE.load_training_only_seed(
        entry,
        receipt_seed,
        artifact_hasher=artifact_hasher,
        checkpoint_loader=lambda data: torch.load(
            data,
            map_location="cpu",
            weights_only=False,
        ),
        expected_partition={"train": (6, 2)},
        expected_dimension=2,
    )

    assert set(hashed) == {Path(record["path"]) for record in entry.values()}
    assert loaded_npz_sources == ["BytesIO"]
    assert bound.train_embeddings.shape == (6, 2)
    assert bound.checkpoint_sha256 == entry["checkpoint_pt"]["sha256"]
    assert not any(
        hasattr(bound, name)
        for name in (
            "checkpoint_path",
            "query_embeddings",
            "gallery_embeddings",
            "prehead",
        )
    )
    for array in (
        bound.train_embeddings,
        bound.train_labels,
        bound.train_example_ids,
        bound.train_source_paths,
        bound.train_row_indices,
        bound.proxies,
    ):
        assert array.flags.writeable is False
        with pytest.raises(ValueError):
            array.setflags(write=True)


def test_training_only_digest_gate_rejects_forged_array_before_cache() -> None:
    """Catches a mutable or replaced retained array crossing into tensor caching."""
    values = np.asarray([[1.0, 0.0]], dtype=np.float32)
    hashes = {
        "train_embeddings": _MODULE._framed_array_sha256("train_embeddings", values)
    }
    forged = values.copy()
    forged[0, 0] = 0.5
    bound = object.__new__(_MODULE.TrainingOnlySeedInput)
    object.__setattr__(bound, "train_embeddings", forged)
    object.__setattr__(bound, "training_array_sha256", hashes)

    with pytest.raises(ValueError, match="retained training array SHA-256"):
        _MODULE.validate_retained_training_arrays(bound)
