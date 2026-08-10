"""Tiny pure tests for the preregistered Pass200 RSTA Stage-A core."""

from __future__ import annotations

import builtins
import gc
import hashlib
import importlib.util
import json
import os
import random
import struct
import subprocess
import sys
import types
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


def test_normwise_helper_preserves_diagnostic_import_without_torch_side_effect() -> None:
    """Catches importing torch before an explicit integrity or scientific entrypoint."""
    code = (
        "import importlib.util,sys;"
        f"p={str(_SCRIPT)!r};"
        "s=importlib.util.spec_from_file_location('isolated_rsta_diagnostic',p);"
        "m=importlib.util.module_from_spec(s);"
        "sys.modules[s.name]=m;"
        "s.loader.exec_module(m);"
        "print('torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.strip() == "False"


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


def test_atomic_json_unsorted_preserves_nonalphabetic_order_and_no_clobber(
    tmp_path: Path,
) -> None:
    output = tmp_path / "ordered.json"
    payload = {"zeta": 1, "alpha": 2, "middle": 3}
    _MODULE.write_json_atomic(output, payload, sort_keys=False)
    assert list(json.loads(output.read_text(encoding="utf-8"))) == list(payload)
    original = output.read_bytes()
    with pytest.raises(FileExistsError):
        _MODULE.write_json_atomic(output, payload, sort_keys=False)
    assert output.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


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
    helper = repository / "scripts" / "rsta_normwise_adjoint.py"
    helper.write_bytes(_SCRIPT.with_name("rsta_normwise_adjoint.py").read_bytes())
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


def _legacy_graphful_contextual_fields(
    model: torch.nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    proxies: torch.Tensor,
    proxy_labels: torch.Tensor,
    *,
    receiver_indices: tuple[int, ...],
) -> dict[str, torch.Tensor]:
    """Reproduce the pre-repair field schedule as an exact-value oracle."""
    from sfora.image_end_to_end import _proxy_anchor_loss

    expected_batch_size = int(images.shape[0])
    expected_dimension = int(proxies.shape[1])
    encoder, parameters, parameter_names = _MODULE._functional_encoder(
        model,
        images,
        expected_batch_size=expected_batch_size,
        expected_dimension=expected_dimension,
    )
    z, vjp_function = torch.func.vjp(encoder, parameters)
    loss = _proxy_anchor_loss(
        z,
        labels,
        proxy_embeddings=proxies.detach(),
        proxy_labels=proxy_labels,
        alpha=32.0,
        delta=0.1,
        torch_module=torch,
    )
    dbar = -torch.autograd.grad(loss, z, create_graph=True)[0]
    global_gradient = vjp_function(dbar)[0]
    _, batch_motion = torch.func.jvp(
        encoder,
        (parameters,),
        (global_gradient,),
    )
    self_rows = []
    for receiver in receiver_indices:
        receiver_cotangent = torch.zeros_like(dbar)
        receiver_cotangent[receiver] = dbar[receiver]
        receiver_gradient = vjp_function(receiver_cotangent)[0]
        _, receiver_motion = torch.func.jvp(
            encoder,
            (parameters,),
            (receiver_gradient,),
        )
        self_rows.append(receiver_motion[receiver])
    return {
        "z": z,
        "dbar": dbar,
        "batch_motion": batch_motion,
        "self_motion": torch.stack(self_rows),
        "parameter_gradient_flat": _MODULE._flatten_parameter_tree(
            global_gradient, parameter_names
        ),
        "loss": loss,
    }


@pytest.mark.parametrize("fixture", ["dense", "train_bn"])
def test_exact_contextual_fields_preserve_bits_while_releasing_autograd_graphs(
    fixture: str,
) -> None:
    """Catches detaching too early, numeric rewrites, or returning retained field graphs."""
    if fixture == "dense":
        model, images, _ = _dense_affine_fixture()
        labels = torch.tensor([0, 1, 0], dtype=torch.int64)
        proxies = torch.tensor(
            [[0.8, -0.2, 0.5], [-0.4, 0.7, 0.1]], dtype=torch.float64
        )
    else:
        model = _MODULE.make_bufferless_train_clone(_TinyBatchNorm().train())
        images = torch.tensor([[0.4, -0.3], [1.2, 0.8]], dtype=torch.float64)
        labels = torch.tensor([0, 1], dtype=torch.int64)
        proxies = torch.eye(2, dtype=torch.float64)
    proxy_labels = torch.tensor([0, 1], dtype=torch.int64)
    receivers = tuple(range(int(images.shape[0])))
    legacy = _legacy_graphful_contextual_fields(
        model,
        images,
        labels,
        proxies,
        proxy_labels,
        receiver_indices=receivers,
    )

    observed = _MODULE.exact_contextual_rsta_fields(
        model,
        images,
        labels,
        proxies,
        proxy_labels,
        alpha=32.0,
        delta=0.1,
        receiver_indices=receivers,
        expected_batch_size=int(images.shape[0]),
        expected_dimension=int(proxies.shape[1]),
    )

    for name, legacy_value in legacy.items():
        assert legacy_value.requires_grad
        assert legacy_value.grad_fn is not None
        assert torch.equal(observed[name], legacy_value)
        assert not observed[name].requires_grad
        assert observed[name].grad_fn is None


class _ForwardLifetimeModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 3, dtype=torch.float64)
        self.forward_outputs: list[weakref.ReferenceType[torch.Tensor]] = []

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.projection(values)
        self.forward_outputs.append(weakref.ref(output))
        return output


def test_dependency_audit_forward_is_released_before_vjp_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches reusing or retaining the allow-unused audit graph for field construction."""
    model = _ForwardLifetimeModel()
    images = torch.tensor([[0.2, 0.3], [0.7, -0.4]], dtype=torch.float64)
    cotangents = torch.tensor([[0.1, -0.2, 0.4], [0.5, 0.3, -0.1]], dtype=torch.float64)
    original_vjp = torch.func.vjp

    def checked_vjp(function: Callable[..., Any], *primals: Any, **kwargs: Any) -> Any:
        gc.collect()
        assert len(model.forward_outputs) == 1
        assert model.forward_outputs[0]() is None
        return original_vjp(function, *primals, **kwargs)

    monkeypatch.setattr(torch.func, "vjp", checked_vjp)

    observed = _MODULE.exact_kernel_fields(
        model,
        images,
        cotangents,
        receiver_indices=(0,),
        expected_batch_size=2,
        expected_dimension=3,
    )

    assert observed["parameter_names"] == ("projection.weight", "projection.bias")


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


def test_adjoint_direction_metadata_binds_registered_fp32_tensors_without_resampling() -> None:
    """Catches hashing regenerated directions or changing their dtype/order/topology."""
    model = torch.nn.Linear(2, 3, bias=True, dtype=torch.float32).train()
    parameter_names = tuple(name for name, _ in model.named_parameters())
    expected_name_hash = hashlib.sha256(
        "\n".join(parameter_names).encode("utf-8")
    ).hexdigest()

    for seed in range(4):
        output, parameters = _MODULE.registered_adjoint_directions(
            model,
            (3, 3),
            seed=seed,
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        output_seed = int.from_bytes(
            hashlib.sha256(
                b"rsta-stage-a-v1|adjoint-u|\0" + str(seed).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        parameter_seed = int.from_bytes(
            hashlib.sha256(
                b"rsta-stage-a-v1|adjoint-v|\0" + str(seed).encode("utf-8")
            ).digest()[:8],
            "big",
        )
        expected_output = np.random.Generator(np.random.PCG64(output_seed)).standard_normal(
            (3, 3)
        ).astype(np.float32)
        expected_parameter_flat = np.random.Generator(
            np.random.PCG64(parameter_seed)
        ).standard_normal(9).astype(np.float32)

        assert np.array_equal(output.numpy(), expected_output)
        assert np.array_equal(
            torch.cat([parameters[name].reshape(-1) for name in parameter_names]).numpy(),
            expected_parameter_flat,
        )
        metadata = _MODULE._adjoint_direction_metadata(
            model,
            output,
            parameters,
            output_direction_seed=output_seed,
            parameter_direction_seed=parameter_seed,
        )

        assert list(metadata) == [
            "direction_domain",
            "output_direction_seed",
            "parameter_direction_seed",
            "output_direction_sha256",
            "parameter_direction_sha256",
            "output_shape",
            "parameter_name_order_sha256",
            "parameter_count",
            "model_dtype",
            "reduction_dtype",
        ]
        assert metadata == {
            "direction_domain": "rsta-stage-a-v1",
            "output_direction_seed": output_seed,
            "parameter_direction_seed": parameter_seed,
            "output_direction_sha256": hashlib.sha256(
                expected_output.tobytes(order="C")
            ).hexdigest(),
            "parameter_direction_sha256": hashlib.sha256(
                expected_parameter_flat.tobytes(order="C")
            ).hexdigest(),
            "output_shape": [3, 3],
            "parameter_name_order_sha256": expected_name_hash,
            "parameter_count": 9,
            "model_dtype": "torch.float32",
            "reduction_dtype": "torch.float64",
        }


@pytest.mark.parametrize(
    ("rhs_value", "expected_passed"),
    [
        (np.nextafter(5.0e-16, 0.0), True),
        (5.0e-16, True),
        (np.nextafter(5.0e-16, np.inf), False),
    ],
)
def test_finalize_adjoint_scalars_uses_exact_float64_denominator_and_boundary(
    rhs_value: float,
    expected_passed: bool,
) -> None:
    """Catches a changed denominator, tolerance, boundary, dtype, or finite gate."""
    result = _MODULE._finalize_adjoint_scalars(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(rhs_value, dtype=torch.float64),
    )

    assert list(result) == [
        "lhs",
        "rhs",
        "absolute_error",
        "denominator",
        "relative_error",
        "tolerance",
        "passed",
    ]
    assert result["lhs"] == 0.0
    assert result["rhs"] == rhs_value
    assert result["absolute_error"] == abs(rhs_value)
    assert result["denominator"] == 1.0e-12
    assert result["relative_error"] == abs(rhs_value) / np.float64(1.0e-12)
    assert result["tolerance"] == 5.0e-4
    assert result["passed"] is expected_passed

    with pytest.raises(ValueError, match="float64"):
        _MODULE._finalize_adjoint_scalars(
            torch.tensor(0.0, dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float64),
        )
    for lhs, rhs in (
        (float("nan"), 0.0),
        (0.0, float("inf")),
        (np.finfo(np.float64).max, -np.finfo(np.float64).max),
    ):
        with pytest.raises(ValueError, match="nonfinite"):
            _MODULE._finalize_adjoint_scalars(
                torch.tensor(lhs, dtype=torch.float64),
                torch.tensor(rhs, dtype=torch.float64),
            )


def test_adjoint_inner_products_keep_fp32_operators_and_cast_before_multiply() -> None:
    """Catches casting after FP32 multiplication or returning FP32 reductions."""
    jv = torch.tensor([4097.0], dtype=torch.float32)
    output_direction = torch.tensor([4097.0], dtype=torch.float32)
    parameter_names = ("second", "first")
    tangents = {
        "second": torch.tensor([4097.0], dtype=torch.float32),
        "first": torch.tensor([-8193.0], dtype=torch.float32),
    }
    jtu = {
        "second": torch.tensor([4097.0], dtype=torch.float32),
        "first": torch.tensor([4097.0], dtype=torch.float32),
    }
    expected_lhs = np.sum(
        jv.numpy().astype(np.float64)
        * output_direction.numpy().astype(np.float64),
        dtype=np.float64,
    )
    expected_rhs = np.stack(
        [
            np.sum(
                tangents[name].numpy().astype(np.float64)
                * jtu[name].numpy().astype(np.float64),
                dtype=np.float64,
            )
            for name in parameter_names
        ]
    ).sum(dtype=np.float64)

    lhs, rhs = _MODULE._float64_adjoint_inner_products(
        jv,
        output_direction,
        tangents,
        jtu,
        parameter_names,
    )

    assert lhs.dtype == torch.float64
    assert rhs.dtype == torch.float64
    assert lhs.item() == expected_lhs
    assert rhs.item() == expected_rhs
    assert all(value.dtype == torch.float32 for value in (jv, output_direction))
    assert all(value.dtype == torch.float32 for value in (*tangents.values(), *jtu.values()))
    assert lhs.item() != float((jv * output_direction).sum(dtype=torch.float32))


def test_adjoint_float64_reduction_matches_independent_cancellation_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches FP32 cancellation and RHS reduction outside named-parameter order."""
    jv = torch.tensor([1.0e8, -1.0e8, 1.0, 3.0, 2.0], dtype=torch.float32)
    output_direction = torch.ones(5, dtype=torch.float32)
    parameter_names = ("zeta", "alpha", "mu", "beta", "gamma")
    tangents = {
        "zeta": torch.tensor([1.0e16], dtype=torch.float32),
        "alpha": torch.tensor([-1.0e16], dtype=torch.float32),
        "mu": torch.tensor([1.0], dtype=torch.float32),
        "beta": torch.tensor([3.0], dtype=torch.float32),
        "gamma": torch.tensor([2.0], dtype=torch.float32),
    }
    jtu = {name: torch.ones(1, dtype=torch.float32) for name in parameter_names}
    expected_lhs = np.sum(
        jv.numpy().astype(np.float64)
        * output_direction.numpy().astype(np.float64),
        dtype=np.float64,
    )
    expected_rhs_terms = [
        np.sum(
            tangents[name].numpy().astype(np.float64)
            * jtu[name].numpy().astype(np.float64),
            dtype=np.float64,
        )
        for name in parameter_names
    ]
    expected_rhs = np.stack(expected_rhs_terms).sum(dtype=np.float64)
    fp32_lhs = float((jv * output_direction).sum(dtype=torch.float32))
    fp32_relative_error = abs(fp32_lhs - expected_rhs) / max(
        abs(fp32_lhs), abs(expected_rhs), 1.0e-12
    )
    reference_relative_error = abs(expected_lhs - expected_rhs) / max(
        abs(expected_lhs), abs(expected_rhs), 1.0e-12
    )
    real_stack = torch.stack
    observed_rhs_terms: list[list[float]] = []

    def ordered_stack(values: list[torch.Tensor]) -> torch.Tensor:
        observed_rhs_terms.append([float(value) for value in values])
        return real_stack(values)

    monkeypatch.setattr(torch, "stack", ordered_stack)
    lhs, rhs = _MODULE._float64_adjoint_inner_products(
        jv,
        output_direction,
        tangents,
        jtu,
        parameter_names,
    )

    assert reference_relative_error <= 5.0e-4
    assert fp32_relative_error > 5.0e-4
    assert lhs.item() == expected_lhs == 6.0
    assert rhs.item() == expected_rhs == 6.0
    assert observed_rhs_terms == [expected_rhs_terms]


def test_normwise_adjoint_integrity_audit_composes_exact_extended_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches formula duplication, factor-two drift, or changed action bytes/order."""
    model = torch.nn.Linear(2, 3, bias=True, dtype=torch.float32).train()
    monkeypatch.delitem(sys.modules, "rsta_normwise_adjoint", raising=False)
    images = torch.tensor(
        [[0.2, -0.8], [1.1, 0.4], [-0.5, 0.7]], dtype=torch.float32
    )
    output, directions = _MODULE.registered_adjoint_directions(
        model,
        (3, 3),
        seed=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    output_seed = int.from_bytes(
        hashlib.sha256(b"rsta-stage-a-v1|adjoint-u|\0" + b"2").digest()[:8], "big"
    )
    parameter_seed = int.from_bytes(
        hashlib.sha256(b"rsta-stage-a-v1|adjoint-v|\0" + b"2").digest()[:8], "big"
    )
    parameter_names = tuple(name for name, _ in model.named_parameters())
    encoder, parameters, reference_names = _MODULE._functional_encoder(
        model, images, expected_batch_size=3, expected_dimension=3
    )
    assert tuple(reference_names) == parameter_names
    _, pullback = torch.func.vjp(encoder, parameters)
    _, reference_jvp = torch.func.jvp(encoder, (parameters,), (directions,))
    reference_vjp = pullback(output)[0]
    audit = _MODULE.adjoint_integrity_audit(
        model,
        images,
        output,
        directions,
        output_direction_seed=output_seed,
        parameter_direction_seed=parameter_seed,
        expected_batch_size=3,
        expected_dimension=3,
    )

    output64 = output.numpy().astype(np.float64)
    jvp64 = reference_jvp.detach().numpy().astype(np.float64)
    direction64 = {
        name: directions[name].numpy().astype(np.float64) for name in parameter_names
    }
    vjp64 = {
        name: reference_vjp[name].detach().numpy().astype(np.float64)
        for name in parameter_names
    }
    expected_lhs = float(np.sum(output64 * jvp64, dtype=np.float64))
    expected_rhs = float(
        np.stack(
            [
                np.sum(direction64[name] * vjp64[name], dtype=np.float64)
                for name in parameter_names
            ]
        ).sum(dtype=np.float64)
    )
    expected_error = abs(expected_lhs - expected_rhs)
    expected_denominator = max(abs(expected_lhs), abs(expected_rhs), np.float64(1.0e-12))
    expected_relative = expected_error / expected_denominator
    output_l2 = float(np.sqrt(np.sum(output64 * output64, dtype=np.float64)))
    parameter_l2 = float(
        np.sqrt(
            np.stack(
                [np.sum(direction64[name] ** 2, dtype=np.float64) for name in parameter_names]
            ).sum(dtype=np.float64)
        )
    )
    jvp_l2 = float(np.sqrt(np.sum(jvp64 * jvp64, dtype=np.float64)))
    vjp_l2 = float(
        np.sqrt(
            np.stack(
                [np.sum(vjp64[name] ** 2, dtype=np.float64) for name in parameter_names]
            ).sum(dtype=np.float64)
        )
    )
    normwise_denominator = output_l2 * jvp_l2 + parameter_l2 * vjp_l2
    eta_norm = expected_error / normwise_denominator
    beta_norm = 2.0 * eta_norm
    lhs_absolute = float(np.sum(np.abs(output64 * jvp64), dtype=np.float64))
    rhs_absolute = float(
        np.stack(
            [
                np.sum(np.abs(direction64[name] * vjp64[name]), dtype=np.float64)
                for name in parameter_names
            ]
        ).sum(dtype=np.float64)
    )
    expected_parameter_bytes = b"".join(
        directions[name].numpy().tobytes(order="C") for name in parameter_names
    )
    expected_jvp_hash = hashlib.sha256(
        reference_jvp.detach().contiguous().numpy().tobytes(order="C")
    ).hexdigest()
    expected_vjp_hash = hashlib.sha256(
        b"".join(
            reference_vjp[name].detach().contiguous().numpy().tobytes(order="C")
            for name in parameter_names
        )
    ).hexdigest()

    expected = {
        "direction_domain": "rsta-stage-a-v1",
        "output_direction_seed": output_seed,
        "parameter_direction_seed": parameter_seed,
        "output_direction_sha256": hashlib.sha256(
            output.numpy().tobytes(order="C")
        ).hexdigest(),
        "parameter_direction_sha256": hashlib.sha256(expected_parameter_bytes).hexdigest(),
        "output_shape": [3, 3],
        "parameter_name_order_sha256": hashlib.sha256(
            "\n".join(parameter_names).encode("utf-8")
        ).hexdigest(),
        "parameter_count": 9,
        "model_dtype": "torch.float32",
        "reduction_dtype": "torch.float64",
        "lhs": expected_lhs,
        "rhs": expected_rhs,
        "absolute_error": expected_error,
        "denominator": expected_denominator,
        "relative_error": expected_relative,
        "tolerance": 5.0e-4,
        "passed": expected_relative <= 5.0e-4,
        "output_direction_l2": output_l2,
        "parameter_direction_l2": parameter_l2,
        "jvp_l2": jvp_l2,
        "vjp_l2": vjp_l2,
        "normwise_denominator": normwise_denominator,
        "eta_norm": eta_norm,
        "beta_norm": beta_norm,
        "lhs_absolute_product_sum": lhs_absolute,
        "rhs_absolute_product_sum": rhs_absolute,
        "lhs_cancellation_factor": lhs_absolute / abs(expected_lhs),
        "rhs_cancellation_factor": rhs_absolute / abs(expected_rhs),
        "jvp_sha256": expected_jvp_hash,
        "vjp_sha256": expected_vjp_hash,
        "controls": {
            "rebuild": {
                "jvp_sha256": expected_jvp_hash,
                "vjp_sha256": expected_vjp_hash,
                "beta_norm": beta_norm,
                "exact_action_hash_match": True,
                "passed": beta_norm <= 5.0e-4,
            },
            "reversed_action_order": {
                "jvp_sha256": expected_jvp_hash,
                "vjp_sha256": expected_vjp_hash,
                "beta_norm": beta_norm,
                "exact_action_hash_match": True,
                "passed": beta_norm <= 5.0e-4,
            },
            "parameter_sign": {
                "jvp_sha256": hashlib.sha256(
                    (-reference_jvp).detach().contiguous().numpy().tobytes(order="C")
                ).hexdigest(),
                "vjp_sha256": expected_vjp_hash,
                "reference_jvp_sha256": expected_jvp_hash,
                "reference_vjp_sha256": expected_vjp_hash,
                "beta_norm": beta_norm,
                "reference_exact_action_hash_match": True,
                "exact_relation": True,
                "passed": beta_norm <= 5.0e-4,
            },
            "output_sign": {
                "jvp_sha256": expected_jvp_hash,
                "vjp_sha256": hashlib.sha256(
                    b"".join(
                        (-reference_vjp[name]).detach().contiguous().numpy().tobytes(order="C")
                        for name in parameter_names
                    )
                ).hexdigest(),
                "reference_jvp_sha256": expected_jvp_hash,
                "reference_vjp_sha256": expected_vjp_hash,
                "beta_norm": beta_norm,
                "reference_exact_action_hash_match": True,
                "exact_relation": True,
                "passed": beta_norm <= 5.0e-4,
            },
        },
        "normwise_tolerance": 5.0e-4,
        "normwise_passed": beta_norm <= 5.0e-4,
        "integrity_passed": beta_norm <= 5.0e-4,
    }
    assert list(audit) == list(_MODULE._ADJOINT_AUDIT_FIELDS)
    assert audit == expected
    assert audit["beta_norm"] == 2.0 * audit["eta_norm"]
    assert "rsta_normwise_adjoint" not in sys.modules
    assert not any(
        name.startswith("_pass200_rsta_normwise_adjoint_") for name in sys.modules
    )

    with pytest.raises(ValueError, match="every encoder parameter"):
        _MODULE.adjoint_integrity_audit(
            model,
            images,
            output,
            {"weight": directions["weight"]},
            output_direction_seed=output_seed,
            parameter_direction_seed=parameter_seed,
            expected_batch_size=3,
            expected_dimension=3,
        )
    wrong_shape = dict(directions)
    wrong_shape["bias"] = torch.zeros(4, dtype=torch.float32)
    with pytest.raises(ValueError, match="topology"):
        _MODULE.adjoint_integrity_audit(
            model,
            images,
            output,
            wrong_shape,
            output_direction_seed=output_seed,
            parameter_direction_seed=parameter_seed,
            expected_batch_size=3,
            expected_dimension=3,
        )
    with pytest.raises(ValueError, match="descriptor shape"):
        _MODULE.adjoint_integrity_audit(
            model,
            images,
            output[:2],
            directions,
            output_direction_seed=output_seed,
            parameter_direction_seed=parameter_seed,
            expected_batch_size=3,
            expected_dimension=3,
        )


def test_normwise_adjoint_preserves_device_legacy_bytes_when_cpu_metrics_are_perturbed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches replacing the pre-extension device reductions with helper CPU reductions."""
    model = torch.nn.Linear(2, 3, bias=True, dtype=torch.float32).train()
    images = torch.tensor(
        [[0.2, -0.8], [1.1, 0.4], [-0.5, 0.7]], dtype=torch.float32
    )
    output, directions = _MODULE.registered_adjoint_directions(
        model,
        (3, 3),
        seed=2,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    encoder, parameters, names = _MODULE._functional_encoder(
        model, images, expected_batch_size=3, expected_dimension=3
    )
    _, pullback = torch.func.vjp(encoder, parameters)
    _, jvp_action = torch.func.jvp(encoder, (parameters,), (directions,))
    vjp_action = pullback(output)[0]
    lhs_tensor, rhs_tensor = _MODULE._float64_adjoint_inner_products(
        jvp_action, output, directions, vjp_action, names
    )
    expected = _MODULE._finalize_adjoint_scalars(lhs_tensor, rhs_tensor)
    del pullback, encoder, parameters, jvp_action, vjp_action, lhs_tensor, rhs_tensor

    helper = importlib.import_module("rsta_normwise_adjoint")
    original_metrics = helper.normwise_adjoint_metrics
    observed_overrides: list[tuple[float, float]] = []

    def perturbed_cpu_metrics(
        u: torch.Tensor,
        a: torch.Tensor,
        parameter_direction: dict[str, torch.Tensor],
        transpose_action: dict[str, torch.Tensor],
        parameter_names: tuple[str, ...],
        *,
        legacy_lhs: float,
        legacy_rhs: float,
    ) -> dict[str, object]:
        observed_overrides.append((legacy_lhs, legacy_rhs))
        return original_metrics(
            u + torch.full_like(u, 37.0),
            a,
            parameter_direction,
            transpose_action,
            parameter_names,
            legacy_lhs=legacy_lhs,
            legacy_rhs=legacy_rhs,
        )

    monkeypatch.setattr(helper, "normwise_adjoint_metrics", perturbed_cpu_metrics)
    monkeypatch.setattr(
        _MODULE,
        "_load_authenticated_normwise_adjoint_helper",
        lambda *_args: helper,
    )
    audit = _MODULE.adjoint_integrity_audit(
        model,
        images,
        output,
        directions,
        output_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", "2"),
        parameter_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-v|", "2"),
        expected_batch_size=3,
        expected_dimension=3,
    )

    assert len(observed_overrides) == 5
    for field, expected_field in (
        ("lhs", "lhs"),
        ("rhs", "rhs"),
        ("absolute_error", "absolute_error"),
        ("denominator", "denominator"),
        ("relative_error", "relative_error"),
        ("tolerance", "tolerance"),
    ):
        assert struct.pack(">d", audit[field]) == struct.pack(">d", expected[expected_field])
    assert audit["passed"] is expected["passed"]


def _tiny_normwise_audit_inputs() -> tuple[
    torch.nn.Module, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
]:
    model = torch.nn.Linear(2, 2, bias=True, dtype=torch.float32).train()
    images = torch.tensor([[0.25, -0.5], [0.75, 0.125]], dtype=torch.float32)
    output, directions = _MODULE.registered_adjoint_directions(
        model, (2, 2), seed=7, dtype=torch.float32, device=torch.device("cpu")
    )
    return model, images, output, directions


def _run_tiny_normwise_audit(
    model: torch.nn.Module,
    images: torch.Tensor,
    output: torch.Tensor,
    directions: dict[str, torch.Tensor],
) -> dict[str, Any]:
    return _MODULE.adjoint_integrity_audit(
        model,
        images,
        output,
        directions,
        output_direction_seed=7,
        parameter_direction_seed=8,
        expected_batch_size=2,
        expected_dimension=2,
    )


def test_normwise_sign_controls_accept_dead_relu_signed_zero_by_direct_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 2, bias=True, dtype=torch.float32), torch.nn.ReLU()
    ).train()
    with torch.no_grad():
        model[0].weight.fill_(-1.0)
        model[0].bias.fill_(-1.0)
    images = torch.ones((2, 2), dtype=torch.float32)
    output, directions = _MODULE.registered_adjoint_directions(
        model, (2, 2), seed=9, dtype=torch.float32, device=torch.device("cpu")
    )
    helper = importlib.import_module("rsta_normwise_adjoint")
    real_equal = torch.equal
    labels: list[str] = []

    def comparator(
        control_name: str,
        target_jvp: torch.Tensor,
        target_vjp: dict[str, torch.Tensor],
        reference_jvp: torch.Tensor,
        reference_vjp: dict[str, torch.Tensor],
        parameter_names: tuple[str, ...],
        *,
        expected_device: torch.device,
    ) -> bool:
        assert expected_device == target_jvp.device
        if control_name == "parameter_sign":
            assert real_equal(target_jvp, -reference_jvp)
            target_hash = hashlib.sha256(target_jvp.detach().numpy().tobytes()).hexdigest()
            negated_hash = hashlib.sha256(
                (-reference_jvp).detach().numpy().tobytes()
            ).hexdigest()
            assert target_hash != negated_hash
        else:
            assert real_equal(target_jvp, reference_jvp)
        results = []
        right_jvp = -reference_jvp if control_name == "parameter_sign" else reference_jvp
        left_right = [(target_jvp, right_jvp)]
        left_right.extend(
            (
                target_vjp[name],
                reference_vjp[name] if control_name == "parameter_sign" else -reference_vjp[name],
            )
            for name in parameter_names
        )
        for index, (left, right) in enumerate(left_right):
            label = (
                f"{control_name}:jvp"
                if index == 0
                else f"{control_name}:vjp:{parameter_names[index - 1]}"
            )
            labels.append(label)
            results.append(torch.equal(left, right))
        return all(results)

    monkeypatch.setattr(helper, "exact_live_sign_control_relation", comparator, raising=False)
    monkeypatch.setattr(
        _MODULE, "_load_authenticated_normwise_adjoint_helper", lambda *_args: helper
    )
    audit = _MODULE.adjoint_integrity_audit(
        model,
        images,
        output,
        directions,
        output_direction_seed=9,
        parameter_direction_seed=10,
        expected_batch_size=2,
        expected_dimension=2,
    )
    assert audit["controls"]["parameter_sign"]["exact_relation"] is True
    assert audit["controls"]["output_sign"]["exact_relation"] is True
    names = tuple(name for name, _ in model.named_parameters())
    assert labels == [
        *["parameter_sign:jvp", *(f"parameter_sign:vjp:{name}" for name in names)],
        *["output_sign:jvp", *(f"output_sign:vjp:{name}" for name in names)],
    ]


def test_normwise_sign_comparator_receives_raw_torch_func_actions_before_copy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, images, output, directions = _tiny_normwise_audit_inputs()
    helper = importlib.import_module("rsta_normwise_adjoint")
    events: list[str] = []
    raw_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    metric_calls = 0
    original_metrics = helper.normwise_adjoint_metrics
    original_tensor = helper._tensor

    def comparator(
        control_name: str,
        target_jvp: torch.Tensor,
        target_vjp: dict[str, torch.Tensor],
        reference_jvp: torch.Tensor,
        reference_vjp: dict[str, torch.Tensor],
        parameter_names: tuple[str, ...],
        *,
        expected_device: torch.device,
    ) -> bool:
        events.append(control_name)
        raw_refs.extend((weakref.ref(target_jvp), weakref.ref(reference_jvp)))
        raw_refs.extend(weakref.ref(target_vjp[name]) for name in parameter_names)
        raw_refs.extend(weakref.ref(reference_vjp[name]) for name in parameter_names)
        assert all(value.device == expected_device for value in (target_jvp, reference_jvp))
        return True

    def guarded_tensor(value: Any, *, name: str) -> torch.Tensor:
        assert all(reference() is not value for reference in raw_refs)
        return original_tensor(value, name=name)

    def metrics(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal metric_calls
        if metric_calls >= 3:
            assert len(events) == metric_calls - 2
        metric_calls += 1
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(helper, "exact_live_sign_control_relation", comparator, raising=False)
    monkeypatch.setattr(helper, "_tensor", guarded_tensor)
    monkeypatch.setattr(helper, "normwise_adjoint_metrics", metrics)
    monkeypatch.setattr(
        _MODULE, "_load_authenticated_normwise_adjoint_helper", lambda *_args: helper
    )
    _run_tiny_normwise_audit(model, images, output, directions)
    assert events == ["parameter_sign", "output_sign"]


def test_normwise_sign_controls_use_exact_target_reference_call_count_and_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, images, output, directions = _tiny_normwise_audit_inputs()
    helper = importlib.import_module("rsta_normwise_adjoint")
    monkeypatch.setattr(
        helper,
        "exact_live_sign_control_relation",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        _MODULE, "_load_authenticated_normwise_adjoint_helper", lambda *_args: helper
    )
    real_vjp, real_jvp = torch.func.vjp, torch.func.jvp
    events: list[str] = []
    graph = -1
    jvp_count = 0
    vjp_count = 0
    graph_names = ("baseline", "rebuild", "reversed", "parameter_sign", "output_sign")

    def wrapped_vjp(*args: Any, **kwargs: Any) -> Any:
        nonlocal graph
        graph += 1
        name = graph_names[graph]
        events.append(f"{name}:vjp_construct")
        primal, closure = real_vjp(*args, **kwargs)
        local_calls = 0

        def wrapped_closure(*closure_args: Any, **closure_kwargs: Any) -> Any:
            nonlocal local_calls, vjp_count
            local_calls += 1
            vjp_count += 1
            suffix = "vjp" if graph < 3 else ("target_vjp" if local_calls == 1 else "reference_vjp")
            events.append(f"{name}:{suffix}")
            return closure(*closure_args, **closure_kwargs)

        return primal, wrapped_closure

    per_graph_jvp: dict[int, int] = {}

    def wrapped_jvp(*args: Any, **kwargs: Any) -> Any:
        nonlocal jvp_count
        jvp_count += 1
        per_graph_jvp[graph] = per_graph_jvp.get(graph, 0) + 1
        name = graph_names[graph]
        suffix = (
            "jvp"
            if graph < 3
            else ("target_jvp" if per_graph_jvp[graph] == 1 else "reference_jvp")
        )
        events.append(f"{name}:{suffix}")
        return real_jvp(*args, **kwargs)

    monkeypatch.setattr(torch.func, "vjp", wrapped_vjp)
    monkeypatch.setattr(torch.func, "jvp", wrapped_jvp)
    _run_tiny_normwise_audit(model, images, output, directions)
    assert events == [
        "baseline:vjp_construct", "baseline:jvp", "baseline:vjp",
        "rebuild:vjp_construct", "rebuild:jvp", "rebuild:vjp",
        "reversed:vjp_construct", "reversed:vjp", "reversed:jvp",
        "parameter_sign:vjp_construct", "parameter_sign:target_jvp",
        "parameter_sign:target_vjp", "parameter_sign:reference_jvp",
        "parameter_sign:reference_vjp", "output_sign:vjp_construct",
        "output_sign:target_jvp", "output_sign:target_vjp",
        "output_sign:reference_jvp", "output_sign:reference_vjp",
    ]
    assert (graph + 1, jvp_count, vjp_count) == (5, 7, 7)


def test_normwise_sign_controls_compute_metrics_for_targets_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, images, output, directions = _tiny_normwise_audit_inputs()
    helper = importlib.import_module("rsta_normwise_adjoint")
    calls: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    original = helper.normwise_adjoint_metrics
    comparator_calls: list[str] = []

    def metrics(
        u: torch.Tensor,
        a: torch.Tensor,
        parameter_direction: dict[str, torch.Tensor],
        vjp_action: dict[str, torch.Tensor],
        names: tuple[str, ...],
        **kwargs: Any,
    ) -> dict[str, object]:
        calls.append((u, parameter_direction))
        return original(u, a, parameter_direction, vjp_action, names, **kwargs)

    monkeypatch.setattr(helper, "normwise_adjoint_metrics", metrics)
    monkeypatch.setattr(
        helper,
        "exact_live_sign_control_relation",
        lambda control_name, *_args, **_kwargs: comparator_calls.append(control_name) or True,
        raising=False,
    )
    monkeypatch.setattr(
        _MODULE, "_load_authenticated_normwise_adjoint_helper", lambda *_args: helper
    )
    _run_tiny_normwise_audit(model, images, output, directions)
    assert len(calls) == 5
    assert comparator_calls == ["parameter_sign", "output_sign"]
    assert all(torch.equal(calls[3][1][name], -directions[name]) for name in directions)
    assert torch.equal(calls[4][0], -output)


def test_normwise_sign_control_reference_drift_fails_despite_target_consistency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, images, output, directions = _tiny_normwise_audit_inputs()
    helper = importlib.import_module("rsta_normwise_adjoint")
    real_jvp, real_vjp = torch.func.jvp, torch.func.vjp
    graph = -1
    jvp_calls: dict[int, int] = {}

    def wrapped_vjp(*args: Any, **kwargs: Any) -> Any:
        nonlocal graph
        graph += 1
        primal, closure = real_vjp(*args, **kwargs)
        local = 0

        def wrapped_closure(*closure_args: Any, **closure_kwargs: Any) -> Any:
            nonlocal local
            local += 1
            result = closure(*closure_args, **closure_kwargs)
            if graph == 3:
                return ({name: value + 0.25 for name, value in result[0].items()},)
            return result

        return primal, wrapped_closure

    def wrapped_jvp(*args: Any, **kwargs: Any) -> Any:
        jvp_calls[graph] = jvp_calls.get(graph, 0) + 1
        primal, action = real_jvp(*args, **kwargs)
        if graph == 3:
            delta = torch.full_like(action, 0.25)
            action = action - delta if jvp_calls[graph] == 1 else action + delta
        return primal, action

    monkeypatch.setattr(torch.func, "vjp", wrapped_vjp)
    monkeypatch.setattr(torch.func, "jvp", wrapped_jvp)
    monkeypatch.setattr(
        helper,
        "exact_live_sign_control_relation",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        _MODULE, "_load_authenticated_normwise_adjoint_helper", lambda *_args: helper
    )
    audit = _run_tiny_normwise_audit(model, images, output, directions)
    sign = audit["controls"]["parameter_sign"]
    assert sign["exact_relation"] is True
    assert sign["reference_jvp_sha256"] != audit["jvp_sha256"]
    assert sign["reference_vjp_sha256"] != audit["vjp_sha256"]
    assert sign["reference_exact_action_hash_match"] is False
    assert sign["passed"] is False
    assert audit["integrity_passed"] is False


def test_normwise_sign_control_schema_rejects_every_nested_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, images, output, directions = _tiny_normwise_audit_inputs()
    helper = importlib.import_module("rsta_normwise_adjoint")
    monkeypatch.setattr(
        helper,
        "exact_live_sign_control_relation",
        lambda *_args, **_kwargs: True,
        raising=False,
    )
    monkeypatch.setattr(
        _MODULE, "_load_authenticated_normwise_adjoint_helper", lambda *_args: helper
    )
    audit = _run_tiny_normwise_audit(model, images, output, directions)
    sign_keys = (
        "jvp_sha256",
        "vjp_sha256",
        "reference_jvp_sha256",
        "reference_vjp_sha256",
        "beta_norm",
        "reference_exact_action_hash_match",
        "exact_relation",
        "passed",
    )
    _MODULE._validate_adjoint_integrity_audit(audit, expected_output_shape=(2, 2))
    for control_name in ("parameter_sign", "output_sign"):
        record = audit["controls"][control_name]
        assert tuple(record) == sign_keys
        for key in sign_keys:
            changed = deepcopy(audit)
            changed["controls"][control_name].pop(key)
            with pytest.raises(ValueError):
                _MODULE._validate_adjoint_integrity_audit(
                    changed, expected_output_shape=(2, 2)
                )
        changed = deepcopy(audit)
        changed["controls"][control_name]["extra"] = True
        with pytest.raises(ValueError):
            _MODULE._validate_adjoint_integrity_audit(changed, expected_output_shape=(2, 2))
        for key in sign_keys:
            changed = deepcopy(audit)
            value = changed["controls"][control_name].pop(key)
            changed["controls"][control_name] = {key: value, **changed["controls"][control_name]}
            if tuple(changed["controls"][control_name]) != sign_keys:
                with pytest.raises(ValueError):
                    _MODULE._validate_adjoint_integrity_audit(
                        changed, expected_output_shape=(2, 2)
                    )
        for key in ("reference_jvp_sha256", "reference_vjp_sha256"):
            changed = deepcopy(audit)
            changed["controls"][control_name][key] = "f" * 64
            with pytest.raises(ValueError):
                _MODULE._validate_adjoint_integrity_audit(
                    changed, expected_output_shape=(2, 2)
                )
        for key in (
            "reference_exact_action_hash_match",
            "exact_relation",
            "passed",
        ):
            for replacement in (0, 1, np.bool_(True)):
                changed = deepcopy(audit)
                changed["controls"][control_name][key] = replacement
                with pytest.raises(ValueError):
                    _MODULE._validate_adjoint_integrity_audit(
                        changed, expected_output_shape=(2, 2)
                    )
        for beta, expected_passed in ((5.0e-4, True), (5.0000001e-4, False), ("infinity", False)):
            changed = deepcopy(audit)
            sign = changed["controls"][control_name]
            sign["beta_norm"] = beta
            sign["passed"] = expected_passed
            changed["integrity_passed"] = expected_passed and all(
                item["passed"] is True
                for name, item in changed["controls"].items()
                if name != control_name
            )
            _MODULE._validate_adjoint_integrity_audit(
                changed, expected_output_shape=(2, 2)
            )


@pytest.mark.parametrize("control_name", ("parameter_sign", "output_sign"))
def test_normwise_sign_control_infinity_requires_exact_builtin_string(
    control_name: str,
) -> None:
    model, images, output, directions = _tiny_normwise_audit_inputs()
    audit = _run_tiny_normwise_audit(model, images, output, directions)
    builtin = deepcopy(audit)
    builtin["controls"][control_name]["beta_norm"] = "infinity"
    builtin["controls"][control_name]["passed"] = False
    builtin["integrity_passed"] = False
    _MODULE._validate_adjoint_integrity_audit(builtin, expected_output_shape=(2, 2))

    mutated = deepcopy(builtin)
    mutated["controls"][control_name]["beta_norm"] = np.str_("infinity")
    with pytest.raises(ValueError, match="beta_norm|predicate|control|adjoint"):
        _MODULE._validate_adjoint_integrity_audit(
            mutated, expected_output_shape=(2, 2)
        )


def test_real_normwise_sign_comparator_provenance_order_and_lifetime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the authenticated helper on real torch.func target/reference actions."""
    model, images, output, directions = _tiny_normwise_audit_inputs()
    parameter_names = tuple(name for name, _ in model.named_parameters())
    real_encoder = _MODULE._functional_encoder
    real_jvp = torch.func.jvp
    real_vjp = torch.func.vjp
    real_neg = torch.Tensor.__neg__
    graph_index = -1
    sign_actions: dict[int, dict[str, Any]] = {}
    negated_by_source: dict[int, weakref.ReferenceType[torch.Tensor]] = {}
    compared_negated_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    comparison_labels: list[str] = []
    comparison_index = 0

    def tracked_encoder(*args: Any, **kwargs: Any) -> Any:
        gc.collect()
        for earlier, record in sign_actions.items():
            if earlier < graph_index + 1:
                assert all(reference() is None for reference in record["refs"])
        assert all(reference() is None for reference in compared_negated_refs)
        return real_encoder(*args, **kwargs)

    def tracked_vjp(*args: Any, **kwargs: Any) -> Any:
        nonlocal graph_index
        graph_index += 1
        primal, closure = real_vjp(*args, **kwargs)
        closure_calls = 0

        def tracked_closure(*closure_args: Any, **closure_kwargs: Any) -> Any:
            nonlocal closure_calls
            closure_calls += 1
            result = closure(*closure_args, **closure_kwargs)
            if graph_index >= 3:
                record = sign_actions.setdefault(
                    graph_index, {"jvps": [], "vjps": [], "refs": []}
                )
                tree = result[0]
                record["vjps"].append({name: id(tree[name]) for name in parameter_names})
                record["refs"].extend(weakref.ref(tree[name]) for name in parameter_names)
            return result

        return primal, tracked_closure

    def tracked_jvp(*args: Any, **kwargs: Any) -> Any:
        primal, action = real_jvp(*args, **kwargs)
        if graph_index >= 3:
            record = sign_actions.setdefault(
                graph_index, {"jvps": [], "vjps": [], "refs": []}
            )
            record["jvps"].append(id(action))
            record["refs"].append(weakref.ref(action))
        return primal, action

    def tracked_neg(value: torch.Tensor) -> torch.Tensor:
        result = real_neg(value)
        negated_by_source[id(value)] = weakref.ref(result)
        return result

    def equality_oracle(left: torch.Tensor, right: torch.Tensor) -> bool:
        nonlocal comparison_index
        per_control = len(parameter_names) + 1
        control_offset = comparison_index // per_control
        within = comparison_index % per_control
        graph = 3 + control_offset
        control_name = "parameter_sign" if graph == 3 else "output_sign"
        record = sign_actions[graph]
        if within == 0:
            assert id(left) == record["jvps"][0]
            if control_name == "parameter_sign":
                reference_id = record["jvps"][1]
                negated = negated_by_source[reference_id]
                assert negated() is right
                compared_negated_refs.append(weakref.ref(right))
            else:
                assert id(right) == record["jvps"][1]
            comparison_labels.append(f"{control_name}:jvp")
        else:
            name = parameter_names[within - 1]
            assert id(left) == record["vjps"][0][name]
            reference_id = record["vjps"][1][name]
            if control_name == "output_sign":
                negated = negated_by_source[reference_id]
                assert negated() is right
                compared_negated_refs.append(weakref.ref(right))
            else:
                assert id(right) == reference_id
            comparison_labels.append(f"{control_name}:vjp:{name}")
        comparison_index += 1
        return within != 0

    monkeypatch.setattr(_MODULE, "_functional_encoder", tracked_encoder)
    monkeypatch.setattr(torch.func, "vjp", tracked_vjp)
    monkeypatch.setattr(torch.func, "jvp", tracked_jvp)
    monkeypatch.setattr(torch.Tensor, "__neg__", tracked_neg)
    monkeypatch.setattr(torch, "equal", equality_oracle)
    audit = _run_tiny_normwise_audit(model, images, output, directions)

    expected_labels = [
        "parameter_sign:jvp",
        *(f"parameter_sign:vjp:{name}" for name in parameter_names),
        "output_sign:jvp",
        *(f"output_sign:vjp:{name}" for name in parameter_names),
    ]
    assert comparison_labels == expected_labels
    assert comparison_index == 2 * (len(parameter_names) + 1)
    assert audit["controls"]["parameter_sign"]["exact_relation"] is False
    assert audit["controls"]["output_sign"]["exact_relation"] is False
    assert len(compared_negated_refs) == 1 + len(parameter_names)
    gc.collect()
    assert all(
        reference() is None
        for record in sign_actions.values()
        for reference in record["refs"]
    )
    assert all(reference() is None for reference in compared_negated_refs)


def test_normwise_sign_controls_release_target_reference_actions_before_next_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, images, output, directions = _tiny_normwise_audit_inputs()
    helper = importlib.import_module("rsta_normwise_adjoint")
    original_encoder = _MODULE._functional_encoder
    original_metrics = helper.normwise_adjoint_metrics
    graph_count = 0
    metric_count = 0
    peak_live_graphs = 0
    graph_refs: list[weakref.ReferenceType[Any]] = []
    target_reference_action_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    negated_reference_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    comparison_count = 0

    def tracked_encoder(*args: Any, **kwargs: Any) -> Any:
        nonlocal graph_count, peak_live_graphs, graph_refs
        gc.collect()
        live = sum(reference() is not None for reference in graph_refs)
        peak_live_graphs = max(peak_live_graphs, live + 1)
        assert live == 0
        assert all(reference() is None for reference in target_reference_action_refs)
        encoder, parameters, names = original_encoder(*args, **kwargs)
        graph_count += 1
        graph_refs = [weakref.ref(encoder)]
        return encoder, parameters, names

    def comparator(
        control_name: str,
        target_jvp: torch.Tensor,
        target_vjp: dict[str, torch.Tensor],
        reference_jvp: torch.Tensor,
        reference_vjp: dict[str, torch.Tensor],
        names: tuple[str, ...],
        *,
        expected_device: torch.device,
    ) -> bool:
        nonlocal comparison_count
        del expected_device
        comparison_count += 1
        target_reference_action_refs.extend(
            [weakref.ref(target_jvp), weakref.ref(reference_jvp)]
            + [weakref.ref(target_vjp[name]) for name in names]
            + [weakref.ref(reference_vjp[name]) for name in names]
        )
        negative = -reference_jvp if control_name == "parameter_sign" else -reference_vjp[names[0]]
        negated_reference_refs.append(weakref.ref(negative))
        return True

    def metrics(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal metric_count
        result = original_metrics(*args, **kwargs)
        metric_count += 1
        return result

    monkeypatch.setattr(_MODULE, "_functional_encoder", tracked_encoder)
    monkeypatch.setattr(helper, "normwise_adjoint_metrics", metrics)
    monkeypatch.setattr(helper, "exact_live_sign_control_relation", comparator, raising=False)
    monkeypatch.setattr(
        _MODULE, "_load_authenticated_normwise_adjoint_helper", lambda *_args: helper
    )
    _run_tiny_normwise_audit(model, images, output, directions)
    gc.collect()
    assert graph_count == 5
    assert metric_count == 5
    assert comparison_count == 2
    assert peak_live_graphs == 1
    assert all(reference() is None for reference in graph_refs)
    assert all(reference() is None for reference in target_reference_action_refs)
    assert all(reference() is None for reference in negated_reference_refs)


def test_normwise_sign_control_reference_structure_fails_before_next_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, images, output, directions = _tiny_normwise_audit_inputs()
    helper = importlib.import_module("rsta_normwise_adjoint")
    real_vjp = torch.func.vjp
    graph_count = 0
    metric_count = 0
    original_encoder = _MODULE._functional_encoder
    original_metrics = helper.normwise_adjoint_metrics

    def encoder(*args: Any, **kwargs: Any) -> Any:
        nonlocal graph_count
        graph_count += 1
        return original_encoder(*args, **kwargs)

    def wrapped_vjp(*args: Any, **kwargs: Any) -> Any:
        primal, closure = real_vjp(*args, **kwargs)
        calls = 0

        def wrapped_closure(*closure_args: Any, **closure_kwargs: Any) -> Any:
            nonlocal calls
            calls += 1
            result = closure(*closure_args, **closure_kwargs)
            if graph_count == 4 and calls == 2:
                changed = dict(result[0])
                changed.pop(next(reversed(changed)))
                return (changed,)
            return result

        return primal, wrapped_closure

    def metrics(*args: Any, **kwargs: Any) -> dict[str, object]:
        nonlocal metric_count
        metric_count += 1
        return original_metrics(*args, **kwargs)

    monkeypatch.setattr(_MODULE, "_functional_encoder", encoder)
    monkeypatch.setattr(torch.func, "vjp", wrapped_vjp)
    monkeypatch.setattr(helper, "normwise_adjoint_metrics", metrics)
    monkeypatch.setattr(
        _MODULE, "_load_authenticated_normwise_adjoint_helper", lambda *_args: helper
    )
    with pytest.raises(ValueError, match="topology|order|VJP"):
        _run_tiny_normwise_audit(model, images, output, directions)
    assert graph_count == 4
    assert metric_count == 3


def test_normwise_adjoint_validator_accepts_only_authorized_zero_denominator_corner() -> None:
    """Catches rejecting the frozen infinity corner or accepting other string/nonfinite drift."""
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float32)
    images = torch.ones((1, 1), dtype=torch.float32)
    output, directions = _MODULE.registered_adjoint_directions(
        model, (1, 1), seed=0, dtype=torch.float32, device=torch.device("cpu")
    )
    audit = _MODULE.adjoint_integrity_audit(
        model,
        images,
        output,
        directions,
        output_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", "0"),
        parameter_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-v|", "0"),
        expected_batch_size=1,
        expected_dimension=1,
    )
    corner = deepcopy(audit)
    corner.update(
        {
            "lhs": 1.0,
            "rhs": 0.0,
            "absolute_error": 1.0,
            "denominator": 1.0,
            "relative_error": 1.0,
            "passed": False,
            "output_direction_l2": 0.0,
            "parameter_direction_l2": 0.0,
            "jvp_l2": 0.0,
            "vjp_l2": 0.0,
            "normwise_denominator": 0.0,
            "eta_norm": "infinity",
            "beta_norm": "infinity",
            "lhs_absolute_product_sum": 1.0,
            "rhs_absolute_product_sum": 0.0,
            "lhs_cancellation_factor": 1.0,
            "rhs_cancellation_factor": 1.0,
            "normwise_passed": False,
            "integrity_passed": False,
        }
    )
    for control in corner["controls"].values():
        control["beta_norm"] = "infinity"
        control["passed"] = False
    _MODULE._validate_adjoint_integrity_audit(corner, expected_output_shape=(1, 1))

    for field, replacement in (
        ("eta_norm", "Infinity"),
        ("beta_norm", float("inf")),
        ("normwise_passed", 0),
        ("integrity_passed", 1),
    ):
        changed = deepcopy(corner)
        changed[field] = replacement
        with pytest.raises(ValueError, match="adjoint|normwise|integrity"):
            _MODULE._validate_adjoint_integrity_audit(changed, expected_output_shape=(1, 1))


@pytest.mark.parametrize(
    ("control_name", "hash_name"),
    (
        ("parameter_sign", "reference_vjp_sha256"),
        ("output_sign", "reference_jvp_sha256"),
    ),
)
def test_normwise_adjoint_validator_rejects_forged_sign_relation_hash_consequences(
    control_name: str, hash_name: str
) -> None:
    """Catches trusting a reference-match boolean whose baseline action hash differs."""
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float32)
    images = torch.ones((1, 1), dtype=torch.float32)
    output, directions = _MODULE.registered_adjoint_directions(
        model, (1, 1), seed=0, dtype=torch.float32, device=torch.device("cpu")
    )
    audit = _task2_adjoint_auditor(
        model,
        images,
        output,
        directions,
        output_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", "0"),
        parameter_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-v|", "0"),
        expected_batch_size=1,
        expected_dimension=1,
    )
    _MODULE._validate_adjoint_integrity_audit(audit, expected_output_shape=(1, 1))
    audit["controls"][control_name][hash_name] = "f" * 64

    with pytest.raises(ValueError, match="sign|relation|hash|adjoint"):
        _MODULE._validate_adjoint_integrity_audit(audit, expected_output_shape=(1, 1))


@pytest.mark.parametrize(
    ("lhs", "rhs", "denominator", "relative_error", "legacy_passed"),
    (
        (0.0, 0.0, 1.0e-12, 0.0, True),
        (5.0e-13, 0.0, 1.0e-12, 0.5, False),
        (-5.0e-13, 0.0, 1.0e-12, 0.5, False),
        (1.0e-12, 0.0, 1.0e-12, 1.0, False),
    ),
)
def test_normwise_adjoint_validator_uses_python_float_legacy_floor_boundaries(
    lhs: float,
    rhs: float,
    denominator: float,
    relative_error: float,
    legacy_passed: bool,
) -> None:
    """Catches NumPy scalar/bool leakage at zero and both sides of the legacy floor."""
    model = torch.nn.Linear(1, 1, bias=False, dtype=torch.float32)
    output, directions = _MODULE.registered_adjoint_directions(
        model, (1, 1), seed=0, dtype=torch.float32, device=torch.device("cpu")
    )
    audit = _task2_adjoint_auditor(
        model,
        None,
        output,
        directions,
        output_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", "0"),
        parameter_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-v|", "0"),
        expected_batch_size=1,
        expected_dimension=1,
    )
    absolute_error = abs(lhs - rhs)
    audit.update(
        {
            "lhs": lhs,
            "rhs": rhs,
            "absolute_error": absolute_error,
            "denominator": denominator,
            "relative_error": relative_error,
            "passed": legacy_passed,
            "output_direction_l2": 1.0,
            "parameter_direction_l2": 0.0,
            "jvp_l2": 1.0,
            "vjp_l2": 0.0,
            "normwise_denominator": 1.0,
            "eta_norm": absolute_error,
            "beta_norm": 2.0 * absolute_error,
            "lhs_absolute_product_sum": abs(lhs),
            "rhs_absolute_product_sum": abs(rhs),
            "lhs_cancellation_factor": 1.0,
            "rhs_cancellation_factor": 1.0,
            "normwise_passed": True,
            "integrity_passed": True,
        }
    )
    _MODULE._validate_adjoint_integrity_audit(audit, expected_output_shape=(1, 1))

    for field, replacement in (
        ("lhs", np.float64(lhs)),
        ("denominator", np.float64(denominator)),
        ("passed", int(legacy_passed)),
    ):
        changed = deepcopy(audit)
        changed[field] = replacement
        with pytest.raises(ValueError, match="adjoint|scalar|contract"):
            _MODULE._validate_adjoint_integrity_audit(
                changed, expected_output_shape=(1, 1)
            )


def test_normwise_adjoint_releases_each_graph_and_cpu_action_tree_before_next_trial(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches retaining full graphs or detached action trees across trial boundaries."""
    model = torch.nn.Linear(2, 2, dtype=torch.float32)
    images = torch.ones((2, 2), dtype=torch.float32)
    output, directions = _MODULE.registered_adjoint_directions(
        model, (2, 2), seed=0, dtype=torch.float32, device=torch.device("cpu")
    )
    graph_count = 0
    metric_count = 0
    graph_refs: list[weakref.ReferenceType[Any]] = []
    cpu_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    original_functional_encoder = _MODULE._functional_encoder
    normwise_helper = importlib.import_module("rsta_normwise_adjoint")
    original_metrics = normwise_helper.normwise_adjoint_metrics

    def tracked_functional_encoder(*args: Any, **kwargs: Any) -> Any:
        nonlocal graph_count, graph_refs
        gc.collect()
        assert metric_count == graph_count
        assert all(reference() is None for reference in graph_refs)
        assert all(reference() is None for reference in cpu_refs)
        encoder, parameters, names = original_functional_encoder(*args, **kwargs)
        graph_count += 1
        graph_refs = [weakref.ref(encoder)]
        return encoder, parameters, names

    def tracked_metrics(
        u: torch.Tensor,
        a: torch.Tensor,
        parameter_direction: dict[str, torch.Tensor],
        vjp_action: dict[str, torch.Tensor],
        parameter_names: tuple[str, ...],
        **kwargs: Any,
    ) -> dict[str, object]:
        nonlocal metric_count, cpu_refs
        gc.collect()
        assert graph_count == metric_count + 1
        assert all(reference() is None for reference in graph_refs)
        cpu_refs = [
            weakref.ref(u),
            weakref.ref(a),
            *(weakref.ref(parameter_direction[name]) for name in parameter_names),
            *(weakref.ref(vjp_action[name]) for name in parameter_names),
        ]
        result = original_metrics(
            u,
            a,
            parameter_direction,
            vjp_action,
            parameter_names,
            **kwargs,
        )
        metric_count += 1
        return result

    monkeypatch.setattr(_MODULE, "_functional_encoder", tracked_functional_encoder)
    monkeypatch.setattr(normwise_helper, "normwise_adjoint_metrics", tracked_metrics)
    monkeypatch.setattr(
        _MODULE,
        "_load_authenticated_normwise_adjoint_helper",
        lambda *_args: normwise_helper,
    )
    _MODULE.adjoint_integrity_audit(
        model,
        images,
        output,
        directions,
        output_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", "0"),
        parameter_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-v|", "0"),
        expected_batch_size=2,
        expected_dimension=2,
    )

    gc.collect()
    assert graph_count == metric_count == 5
    assert all(reference() is None for reference in graph_refs)
    assert all(reference() is None for reference in cpu_refs)


def test_normwise_adjoint_nonfinite_first_trial_fails_before_constructing_control_graphs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches delaying first-trial structural failure until all five graphs exist."""
    model = torch.nn.Linear(2, 2, dtype=torch.float32)
    images = torch.ones((2, 2), dtype=torch.float32)
    output, directions = _MODULE.registered_adjoint_directions(
        model, (2, 2), seed=0, dtype=torch.float32, device=torch.device("cpu")
    )
    graph_count = 0
    original_functional_encoder = _MODULE._functional_encoder
    normwise_helper = importlib.import_module("rsta_normwise_adjoint")

    def tracked_functional_encoder(*args: Any, **kwargs: Any) -> Any:
        nonlocal graph_count
        graph_count += 1
        return original_functional_encoder(*args, **kwargs)

    def reject_nonfinite(*_args: Any, **_kwargs: Any) -> dict[str, object]:
        raise ValueError("normwise adjoint reduction is nonfinite")

    monkeypatch.setattr(_MODULE, "_functional_encoder", tracked_functional_encoder)
    monkeypatch.setattr(normwise_helper, "normwise_adjoint_metrics", reject_nonfinite)
    monkeypatch.setattr(
        _MODULE,
        "_load_authenticated_normwise_adjoint_helper",
        lambda *_args: normwise_helper,
    )
    with pytest.raises(ValueError, match="nonfinite"):
        _MODULE.adjoint_integrity_audit(
            model,
            images,
            output,
            directions,
            output_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", "0"),
            parameter_direction_seed=_MODULE.domain_seed(
                "rsta-stage-a-v1|adjoint-v|", "0"
            ),
            expected_batch_size=2,
            expected_dimension=2,
        )

    assert graph_count == 1


def test_normwise_adjoint_rejects_helper_source_before_model_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches inspecting or executing a model before helper authentication succeeds."""
    model_accesses: list[str] = []

    class ForbiddenModel:
        def named_parameters(self) -> Any:
            model_accesses.append("named_parameters")
            raise AssertionError("model accessed before helper rejection")

    def reject_helper(*_args: Any) -> Any:
        raise ValueError("normwise adjoint helper content SHA-256 differs")

    monkeypatch.setattr(
        _MODULE, "_load_authenticated_normwise_adjoint_helper", reject_helper
    )
    with pytest.raises(ValueError, match="helper content"):
        _MODULE.adjoint_integrity_audit(
            ForbiddenModel(),
            torch.ones((1, 1), dtype=torch.float32),
            torch.ones((1, 1), dtype=torch.float32),
            {},
            output_direction_seed=0,
            parameter_direction_seed=0,
            expected_batch_size=1,
            expected_dimension=1,
        )

    assert model_accesses == []


def test_normwise_adjoint_ignores_preloaded_bare_helper_module(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Catches resolving audit arithmetic through a fileless sys.modules forgery."""
    fake = types.ModuleType("rsta_normwise_adjoint")
    fake_calls: list[str] = []

    def forged(*_args: Any, **_kwargs: Any) -> Any:
        fake_calls.append("forged")
        raise AssertionError("preloaded bare helper executed")

    fake.normwise_adjoint_metrics = forged
    fake.tensor_sha256 = forged
    fake.parameter_tree_sha256 = forged
    monkeypatch.setitem(sys.modules, "rsta_normwise_adjoint", fake)
    model = torch.nn.Linear(2, 2, dtype=torch.float32)
    images = torch.ones((2, 2), dtype=torch.float32)
    output, directions = _MODULE.registered_adjoint_directions(
        model, (2, 2), seed=0, dtype=torch.float32, device=torch.device("cpu")
    )

    audit = _MODULE.adjoint_integrity_audit(
        model,
        images,
        output,
        directions,
        output_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", "0"),
        parameter_direction_seed=_MODULE.domain_seed("rsta-stage-a-v1|adjoint-v|", "0"),
        expected_batch_size=2,
        expected_dimension=2,
    )

    assert audit["integrity_passed"] is True
    assert fake_calls == []


def test_authenticated_normwise_helper_uses_literal_content_address_and_cleans_registration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches PYTHONPATH shadowing or leaked/preloaded private helper modules."""
    helper_path = _SCRIPT.with_name("rsta_normwise_adjoint.py").resolve()
    digest = _sha256_file(helper_path)
    private_name = f"_pass200_rsta_normwise_adjoint_{digest}"
    shadow = tmp_path / "rsta_normwise_adjoint.py"
    shadow.write_text("raise AssertionError('shadow helper executed')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))

    helper = _MODULE._load_authenticated_normwise_adjoint_helper(helper_path, digest)

    assert helper.__name__ == private_name
    assert Path(helper.__file__).resolve() == helper_path
    assert helper.__spec__.origin == str(helper_path)
    assert helper.THRESHOLD == 5.0e-4
    assert private_name not in sys.modules

    monkeypatch.setitem(sys.modules, private_name, types.ModuleType(private_name))
    with pytest.raises(ValueError, match="preexisting|registered|module"):
        _MODULE._load_authenticated_normwise_adjoint_helper(helper_path, digest)


def test_authenticated_normwise_helper_rejects_dirty_symlink_and_execution_race(
    tmp_path: Path,
) -> None:
    """Catches executing mismatched bytes, symlink substitution, or mutation during exec."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    helper_path = scripts / "rsta_normwise_adjoint.py"
    marker = tmp_path / "executed"
    dirty_source = f"from pathlib import Path\nPath({str(marker)!r}).write_text('yes')\n"
    helper_path.write_text(dirty_source, encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256|content|digest"):
        _MODULE._load_authenticated_normwise_adjoint_helper(helper_path, "0" * 64)
    assert not marker.exists()

    target = scripts / "target.py"
    target.write_text("VALUE = 1\n", encoding="utf-8")
    helper_path.unlink()
    helper_path.symlink_to(target)
    with pytest.raises(ValueError, match="path|symlink|regular"):
        _MODULE._load_authenticated_normwise_adjoint_helper(
            helper_path, _sha256_file(target)
        )

    helper_path.unlink()
    race_source = (
        "from pathlib import Path\n"
        "Path(__file__).write_text('VALUE = 2\\n', encoding='utf-8')\n"
    )
    helper_path.write_text(race_source, encoding="utf-8")
    race_digest = _sha256_file(helper_path)
    race_name = f"_pass200_rsta_normwise_adjoint_{race_digest}"
    with pytest.raises(ValueError, match="changed|execution|content"):
        _MODULE._load_authenticated_normwise_adjoint_helper(helper_path, race_digest)
    assert race_name not in sys.modules
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


def test_environment_versions_are_builtin_strings_and_validate_after_json_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class VersionLike(str):
        pass

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(torch, "__version__", VersionLike("torch-test-version"))
    monkeypatch.setattr(np, "__version__", VersionLike("numpy-test-version"))
    audit = _MODULE.configure_deterministic_process()
    assert type(audit["torch_version"]) is str
    assert type(audit["numpy_version"]) is str
    roundtrip = json.loads(json.dumps(audit))
    _MODULE._validate_environment_audit(roundtrip)

    for name, value in (
        ("torch_version", ""),
        ("numpy_version", "wrong"),
        ("torch_version", VersionLike("torch-test-version")),
    ):
        changed = dict(roundtrip)
        changed[name] = value
        with pytest.raises(ValueError, match="environment"):
            _MODULE._validate_environment_audit(changed)


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


class _FrozenClassifierParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.last_linear = torch.nn.Linear(2, 2, dtype=torch.float64)
        self.used = torch.nn.Linear(2, 3, dtype=torch.float64)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.used(values)


def test_exact_kernel_fields_supply_frozen_unused_parameters_as_constants() -> None:
    """Catches omission of frozen model state from strict functional calls."""
    model = _FrozenClassifierParameterModel()
    for parameter in model.model.last_linear.parameters():
        parameter.requires_grad_(False)
    images = torch.tensor([[0.2, 0.3], [0.7, -0.4]], dtype=torch.float64)
    cotangents = torch.tensor([[0.1, -0.2, 0.4], [0.5, 0.3, -0.1]], dtype=torch.float64)

    observed = _MODULE.exact_kernel_fields(
        model,
        images,
        cotangents,
        receiver_indices=(0,),
        expected_batch_size=2,
        expected_dimension=3,
    )

    assert observed["parameter_names"] == ("used.weight", "used.bias")
    assert torch.equal(observed["z"], torch.nn.functional.normalize(model(images), dim=-1))


def test_exact_kernel_fields_reject_unregistered_frozen_parameter() -> None:
    """Catches unrelated frozen parameters bypassing the missing-gradient gate."""
    model = _UnusedParameterModel()
    model.unused.requires_grad_(False)
    images = torch.tensor([[0.2, 0.3], [0.7, -0.4]], dtype=torch.float64)
    cotangents = torch.tensor([[0.1, -0.2, 0.4], [0.5, 0.3, -0.1]], dtype=torch.float64)

    with pytest.raises(ValueError, match="frozen encoder parameter.*unused"):
        _MODULE.exact_kernel_fields(
            model,
            images,
            cotangents,
            receiver_indices=(0,),
            expected_batch_size=2,
            expected_dimension=3,
        )


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


def _extended_normwise_adjoint(legacy: dict[str, Any]) -> dict[str, Any]:
    """Extend a valid frozen legacy adjoint fixture with passing normwise evidence."""
    result = dict(legacy)
    action_hash = "c" * 64
    result.update(
        {
            "output_direction_l2": 1.0,
            "parameter_direction_l2": 1.0,
            "jvp_l2": 1.0,
            "vjp_l2": 1.0,
            "normwise_denominator": 2.0,
            "eta_norm": 0.0,
            "beta_norm": 0.0,
            "lhs_absolute_product_sum": abs(result["lhs"]),
            "rhs_absolute_product_sum": abs(result["rhs"]),
            "lhs_cancellation_factor": 1.0,
            "rhs_cancellation_factor": 1.0,
            "jvp_sha256": action_hash,
            "vjp_sha256": action_hash,
            "controls": {
                "rebuild": {
                    "jvp_sha256": action_hash,
                    "vjp_sha256": action_hash,
                    "beta_norm": 0.0,
                    "exact_action_hash_match": True,
                    "passed": True,
                },
                "reversed_action_order": {
                    "jvp_sha256": action_hash,
                    "vjp_sha256": action_hash,
                    "beta_norm": 0.0,
                    "exact_action_hash_match": True,
                    "passed": True,
                },
                "parameter_sign": {
                    "jvp_sha256": "d" * 64,
                    "vjp_sha256": action_hash,
                    "reference_jvp_sha256": action_hash,
                    "reference_vjp_sha256": action_hash,
                    "beta_norm": 0.0,
                    "reference_exact_action_hash_match": True,
                    "exact_relation": True,
                    "passed": True,
                },
                "output_sign": {
                    "jvp_sha256": action_hash,
                    "vjp_sha256": "d" * 64,
                    "reference_jvp_sha256": action_hash,
                    "reference_vjp_sha256": action_hash,
                    "beta_norm": 0.0,
                    "reference_exact_action_hash_match": True,
                    "exact_relation": True,
                    "passed": True,
                },
            },
            "normwise_tolerance": 5.0e-4,
            "normwise_passed": True,
            "integrity_passed": True,
        }
    )
    return result


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
                "adjoint": _extended_normwise_adjoint({
                    "direction_domain": "rsta-stage-a-v1",
                    "output_direction_seed": _MODULE.domain_seed(
                        "rsta-stage-a-v1|adjoint-u|", str(seed)
                    ),
                    "parameter_direction_seed": _MODULE.domain_seed(
                        "rsta-stage-a-v1|adjoint-v|", str(seed)
                    ),
                    "output_direction_sha256": "a" * 64,
                    "parameter_direction_sha256": "b" * 64,
                    "output_shape": [180, 512],
                    "parameter_name_order_sha256": _MODULE._ordered_text_sha256(
                        ["model.embedding.weight", "model.embedding.bias"]
                    ),
                    "parameter_count": 1024 * 512 + 512,
                    "model_dtype": "torch.float32",
                    "reduction_dtype": "torch.float64",
                    "lhs": 1.0,
                    "rhs": 1.0,
                    "absolute_error": 0.0,
                    "denominator": 1.0,
                    "relative_error": 0.0,
                    "tolerance": 5.0e-4,
                    "passed": True,
                }),
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
            "primary": {
                **primary,
                "support_ids_by_label": {
                    str(label): ids
                    for label, ids in primary["support_ids_by_label"].items()
                },
            },
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


def test_roundtrip_recovery_live_scientific_payload_is_exact_and_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = _valid_scientific_payload_arguments(tmp_path, monkeypatch)
    arguments["environment"]["roundtrip_signed_zero_probe"] = -0.0
    verifier_path = _SCRIPT.parent / "verify_pass200_rsta_scientific_artifact.py"
    namespace: dict[str, Any] = {}
    exec(compile(verifier_path.read_bytes(), str(verifier_path), "exec"), namespace)
    first = _MODULE.scientific_payload(**arguments)
    first_path = tmp_path / "first-live-roundtrip.json"
    _MODULE.write_json_atomic(first_path, first, sort_keys=False)
    first_bytes = first_path.read_bytes()
    assert first_bytes == (
        json.dumps(first, indent=2, sort_keys=False, allow_nan=False) + "\n"
    ).encode("utf-8")
    persisted = namespace["strict_json_object"](first_bytes, name="live roundtrip")
    second = _MODULE.scientific_payload(
        manifest_audit=persisted["manifest"],
        execution_audit=persisted["execution_audit"],
        environment=persisted["environment"],
        seed_audits=persisted["seed_audits"],
        primary_rows=persisted["rows"]["primary"],
        alternate_rows=persisted["rows"]["alternate"],
        integrity=persisted["integrity"],
        aggregation=persisted["aggregation"],
        bootstrap=persisted["bootstrap"],
        panel_binding=persisted["panel_binding"],
    )
    assert namespace["exact_ordered_equal"](second, persisted)
    second_path = tmp_path / "second-live-roundtrip.json"
    _MODULE.write_json_atomic(second_path, second, sort_keys=False)
    assert second_path.read_bytes() == first_bytes
    signed_zero_mutant = deepcopy(second)
    signed_zero_mutant["environment"]["roundtrip_signed_zero_probe"] = 0.0
    assert not namespace["exact_ordered_equal"](signed_zero_mutant, persisted)


@pytest.mark.parametrize(
    "mutation",
    (
        "integer_keys",
        "boolean_key",
        "noncanonical_string",
        "missing",
        "extra",
        "order",
        "integer_alias_collision",
    ),
)
def test_scientific_payload_requires_canonical_persisted_support_label_keys(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    arguments = _valid_scientific_payload_arguments(tmp_path, monkeypatch)
    primary = arguments["panel_binding"]["primary"]
    support = primary["support_ids_by_label"]
    keys = list(support)
    first = keys[0]
    if mutation == "integer_keys":
        primary["support_ids_by_label"] = {
            int(label): ids for label, ids in support.items()
        }
    elif mutation == "boolean_key":
        support[True] = support[first]
    elif mutation == "noncanonical_string":
        primary["support_ids_by_label"] = {
            (f"0{label}" if label == first else label): ids
            for label, ids in support.items()
        }
    elif mutation == "missing":
        support.pop(first)
    elif mutation == "extra":
        support["999999"] = support[first]
    elif mutation == "order":
        primary["support_ids_by_label"] = dict(reversed(support.items()))
    else:
        support[int(first)] = support[first]

    with pytest.raises(ValueError, match="support label keys"):
        _MODULE.scientific_payload(**arguments)


def test_scientific_payload_requires_exact_structured_adjoint_audits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches scalar substitution or lossy/malformed structured adjoint persistence."""
    arguments = _valid_scientific_payload_arguments(tmp_path, monkeypatch)
    expected = deepcopy(arguments["integrity"]["seeds"])
    payload = _MODULE.scientific_payload(**arguments)
    assert payload["integrity"]["seeds"] == expected

    mutations = {
        "direction_domain": "wrong",
        "output_direction_seed": 0,
        "parameter_direction_seed": 0,
        "output_direction_sha256": "g" * 64,
        "parameter_direction_sha256": "g" * 64,
        "output_shape": [180, 511],
        "parameter_name_order_sha256": "g" * 64,
        "parameter_count": 0,
        "model_dtype": "torch.float64",
        "reduction_dtype": "torch.float32",
        "lhs": float("nan"),
        "rhs": float("nan"),
        "absolute_error": 1.0,
        "denominator": 2.0,
        "relative_error": 1.0,
        "tolerance": 1.0,
        "passed": False,
    }
    for mutation in ("missing", "extra", *mutations):
        changed = deepcopy(arguments)
        audit = changed["integrity"]["seeds"][2]["adjoint"]
        if mutation == "missing":
            audit.pop("lhs")
        elif mutation == "extra":
            audit["unexpected"] = 0
        else:
            audit[mutation] = mutations[mutation]
        with pytest.raises(ValueError, match="adjoint"):
            _MODULE.scientific_payload(**changed)

    for mutation in ("wrong_parameter_hash", "wrong_parameter_count", "duplicate", "permutation"):
        changed = deepcopy(arguments)
        if mutation == "wrong_parameter_hash":
            changed["integrity"]["seeds"][1]["adjoint"][
                "parameter_name_order_sha256"
            ] = "d" * 64
        elif mutation == "wrong_parameter_count":
            changed["integrity"]["seeds"][1]["adjoint"]["parameter_count"] += 1
        elif mutation == "duplicate":
            changed["integrity"]["seeds"][2] = deepcopy(
                changed["integrity"]["seeds"][1]
            )
        else:
            changed["integrity"]["seeds"][1:3] = reversed(
                changed["integrity"]["seeds"][1:3]
            )
        with pytest.raises(ValueError, match="seed|parameter|adjoint"):
            _MODULE.scientific_payload(**changed)
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
            "artifacts": {
                "checkpoint_pt": {"sha256": f"{seed + 1}" * 64},
                "train_npz": {"sha256": f"{seed + 5}" * 64},
            }
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


def _task2_fixture_audit() -> dict[str, Any]:
    return {
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


_NORMWISE_SOURCE_ORDER = (
    "scripts/diagnose_pass159_cotangent_stage_a.py",
    "scripts/diagnose_pass200_rsta_stage_a.py",
    "scripts/rsta_normwise_adjoint.py",
    "src/sfora/__init__.py",
    "src/sfora/ablation.py",
    "src/sfora/api.py",
    "src/sfora/arcg.py",
    "src/sfora/benchmark.py",
    "src/sfora/bn_inception.py",
    "src/sfora/catalog.py",
    "src/sfora/cea.py",
    "src/sfora/cem.py",
    "src/sfora/cli.py",
    "src/sfora/compose.py",
    "src/sfora/data.py",
    "src/sfora/encoder_ablation.py",
    "src/sfora/encoder_training.py",
    "src/sfora/evaluation.py",
    "src/sfora/experiments.py",
    "src/sfora/image_benchmark.py",
    "src/sfora/image_end_to_end.py",
    "src/sfora/image_recipes.py",
    "src/sfora/ipsr.py",
    "src/sfora/losses.py",
    "src/sfora/method.py",
    "src/sfora/oapf.py",
    "src/sfora/publication.py",
    "src/sfora/remote.py",
    "src/sfora/report.py",
    "src/sfora/text_baselines.py",
    "src/sfora/training.py",
)


def _future_normwise_manifest() -> dict[str, Any]:
    root = _SCRIPT.parents[1]
    prior = json.loads(
        (root / "docs" / "pass200_rsta_receipt_stage_a_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    source_files = dict(prior["current_scientific_source"]["files"])
    diagnostic_hash = source_files.pop("scripts/diagnose_pass200_rsta_stage_a.py")
    ordered_source = {
        "scripts/diagnose_pass159_cotangent_stage_a.py": source_files.pop(
            "scripts/diagnose_pass159_cotangent_stage_a.py"
        ),
        "scripts/diagnose_pass200_rsta_stage_a.py": diagnostic_hash,
        "scripts/rsta_normwise_adjoint.py": _sha256_file(
            root / "scripts" / "rsta_normwise_adjoint.py"
        ),
        "scripts/verify_pass200_rsta_scientific_artifact.py": _sha256_file(
            root / "scripts" / "verify_pass200_rsta_scientific_artifact.py"
        ),
        **source_files,
    }


    return {
        "schema_version": prior["schema_version"],
        "base_preregistration": prior["base_preregistration"],
        "amendment": prior["amendment"],
        "deterministic_pool_amendment": prior["deterministic_pool_amendment"],
        "zero_jacobian_classifier_amendment": prior[
            "zero_jacobian_classifier_amendment"
        ],
        "adjoint_integrity_amendment": prior["adjoint_integrity_amendment"],
        "normwise_adjoint_calibration_protocol": {
            "path": "docs/pass200_rsta_normwise_adjoint_calibration_protocol_2026-08-09.md",
            "sha256": "2f4d52fd6c69588248f1b27acbcd5503b0e53dc3c5bd6b5e0755564017dc21db",
            "commit": "171a3fe24386dbab4eb361c04cbf252da4f4e0bb",
        },
        "normwise_adjoint_calibration_result": {
            "path": (
                "reports/generated/pass200_rsta_receipt/"
                "0f5d1e2f626524f02c565a04f6fa0ae7127cd7e2-"
                "normwise-adjoint-calibration.json"
            ),
            "sha256": "5fcb09a1e3a6eedddd05ef49bd22bc9920656089aa401a5aae2c5704a9d9dc50",
            "commit": "95525af61d66b063983dc55a6015168d9aafd12b",
        },
        "normwise_adjoint_amendment": {
            "path": "docs/pass200_rsta_normwise_adjoint_amendment_2026-08-09.md",
            "sha256": "416fdd6af90fa2e54ace61fcd72721713aae84dc0dd2010bde91037bf0eccbd4",
            "commit": "6ddf1db20e75a47e40726d223827cd3f1a8968e3",
        },
        "normwise_adjoint_sign_control_amendment": {
            "path": "docs/pass200_rsta_sign_control_comparator_amendment_2026-08-10.md",
            "sha256": "a4e20431c47889796ff13c90347accce855067249fc8205769bf7c8c120dd020",
            "commit": "2d09a23994b8584d6726d737d7a3e4022b4a064e",
        },
        "scientific_artifact_roundtrip_recovery_amendment": {
            "path": (
                "docs/pass200_rsta_scientific_artifact_roundtrip_"
                "recovery_amendment_2026-08-10.md"
            ),
            "sha256": "6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591",
            "commit": "043121f8a414b91d7fb2e3d6a1635a6bd585676a",
        },
        "binding_receipt": prior["binding_receipt"],
        "historical": prior["historical"],
        "current_scientific_source": {
            "git_revision": prior["current_scientific_source"]["git_revision"],
            "files": ordered_source,
        },
        "artifact_schema": prior["artifact_schema"],
        "seeds": prior["seeds"],
    }


_ROUNDTRIP_SOURCE_ORDER = (
    "scripts/diagnose_pass159_cotangent_stage_a.py",
    "scripts/diagnose_pass200_rsta_stage_a.py",
    "scripts/rsta_normwise_adjoint.py",
    "scripts/verify_pass200_rsta_scientific_artifact.py",
    *_NORMWISE_SOURCE_ORDER[3:],
)


def _future_roundtrip_manifest() -> dict[str, Any]:
    prior = _future_normwise_manifest()
    old_files = prior["current_scientific_source"]["files"]
    source_files = {
        path: (
            "0" * 64
            if path == "scripts/verify_pass200_rsta_scientific_artifact.py"
            else old_files[path]
        )
        for path in _ROUNDTRIP_SOURCE_ORDER
    }
    return {
        **{
            key: value
            for key, value in prior.items()
            if key
            not in {
                "binding_receipt", "historical", "current_scientific_source",
                "artifact_schema", "seeds",
            }
        },
        "scientific_artifact_roundtrip_recovery_amendment": {
            "path": (
                "docs/pass200_rsta_scientific_artifact_roundtrip_"
                "recovery_amendment_2026-08-10.md"
            ),
            "sha256": "6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591",
            "commit": "043121f8a414b91d7fb2e3d6a1635a6bd585676a",
        },
        "binding_receipt": prior["binding_receipt"],
        "historical": prior["historical"],
        "current_scientific_source": {"git_revision": "a" * 40, "files": source_files},
        "artifact_schema": prior["artifact_schema"],
        "seeds": prior["seeds"],
    }


def test_roundtrip_recovery_manifest_authority_order_and_32_source_paths() -> None:
    validated = _MODULE._validate_amended_manifest_schema(_future_roundtrip_manifest())
    assert list(validated) == [
        "schema_version", "base_preregistration", "amendment",
        "deterministic_pool_amendment", "zero_jacobian_classifier_amendment",
        "adjoint_integrity_amendment", "normwise_adjoint_calibration_protocol",
        "normwise_adjoint_calibration_result", "normwise_adjoint_amendment",
        "normwise_adjoint_sign_control_amendment",
        "scientific_artifact_roundtrip_recovery_amendment", "binding_receipt",
        "historical", "current_scientific_source", "artifact_schema", "seeds",
    ]
    assert tuple(_MODULE._CURRENT_SCIENTIFIC_SOURCE_FILES) == _ROUNDTRIP_SOURCE_ORDER


def test_roundtrip_recovery_projection_rejects_every_nested_mutation() -> None:
    manifest = _future_roundtrip_manifest()
    _MODULE._validate_amended_manifest_schema(manifest)
    authority = "scientific_artifact_roundtrip_recovery_amendment"
    for field, replacement in (
        ("path", "docs/wrong.md"),
        ("sha256", "f" * 64),
        ("commit", "f" * 40),
    ):
        changed = deepcopy(manifest)
        changed[authority][field] = replacement
        with pytest.raises(ValueError):
            _MODULE._validate_amended_manifest_schema(changed)
    for mutation in ("missing", "extra", "reordered", "digest"):
        changed = deepcopy(manifest)
        files = changed["current_scientific_source"]["files"]
        if mutation == "missing":
            files.pop("scripts/verify_pass200_rsta_scientific_artifact.py")
        elif mutation == "extra":
            files["scripts/unreviewed.py"] = "0" * 64
        elif mutation == "reordered":
            value = files.pop("scripts/verify_pass200_rsta_scientific_artifact.py")
            files["scripts/verify_pass200_rsta_scientific_artifact.py"] = value
        else:
            files["scripts/verify_pass200_rsta_scientific_artifact.py"] = "g" * 64
        with pytest.raises(ValueError):
            _MODULE._validate_amended_manifest_schema(changed)


def test_normwise_manifest_freezes_exact_authorities_projection_and_source_order() -> None:
    """Catches omitted authorities, projection drift, or helper/source reordering."""
    manifest = _future_normwise_manifest()
    validated = _MODULE._validate_amended_manifest_schema(manifest)

    assert list(validated) == [
        "schema_version",
        "base_preregistration",
        "amendment",
        "deterministic_pool_amendment",
        "zero_jacobian_classifier_amendment",
        "adjoint_integrity_amendment",
        "normwise_adjoint_calibration_protocol",
        "normwise_adjoint_calibration_result",
        "normwise_adjoint_amendment",
        "normwise_adjoint_sign_control_amendment",
        "scientific_artifact_roundtrip_recovery_amendment",
        "binding_receipt",
        "historical",
        "current_scientific_source",
        "artifact_schema",
        "seeds",
    ]
    assert list(validated["current_scientific_source"]["files"]) == list(
        _ROUNDTRIP_SOURCE_ORDER
    )
    assert tuple(_MODULE._CURRENT_SCIENTIFIC_SOURCE_FILES) == _ROUNDTRIP_SOURCE_ORDER
    assert (
        _MODULE._NORMWISE_ADJOINT_CALIBRATION_PROTOCOL_SHA256
        == "2f4d52fd6c69588248f1b27acbcd5503b0e53dc3c5bd6b5e0755564017dc21db"
    )
    assert (
        _MODULE._NORMWISE_ADJOINT_CALIBRATION_RESULT_SHA256
        == "5fcb09a1e3a6eedddd05ef49bd22bc9920656089aa401a5aae2c5704a9d9dc50"
    )
    assert (
        _MODULE._NORMWISE_ADJOINT_AMENDMENT_SHA256
        == "416fdd6af90fa2e54ace61fcd72721713aae84dc0dd2010bde91037bf0eccbd4"
    )


def test_normwise_manifest_rejects_every_new_authority_leaf_and_order_mutation() -> None:
    """Catches partial authentication or order-insensitive future manifest validation."""
    authority_names = (
        "normwise_adjoint_calibration_protocol",
        "normwise_adjoint_calibration_result",
        "normwise_adjoint_amendment",
        "normwise_adjoint_sign_control_amendment",
        "scientific_artifact_roundtrip_recovery_amendment",
    )
    for authority in authority_names:
        for mutation in ("remove_object", "extra", "path", "sha256", "commit"):
            changed = deepcopy(_future_normwise_manifest())
            if mutation == "remove_object":
                changed.pop(authority)
            elif mutation == "extra":
                changed[authority]["unchecked"] = True
            else:
                changed[authority][mutation] = {
                    "path": "docs/substituted",
                    "sha256": "0" * 64,
                    "commit": "0" * 40,
                }[mutation]
            with pytest.raises(
                ValueError, match="normwise|manifest|calibration|roundtrip|recovery"
            ):
                _MODULE._validate_amended_manifest_schema(changed)

    changed = _future_normwise_manifest()
    changed["unchecked"] = True
    with pytest.raises(ValueError, match="manifest"):
        _MODULE._validate_amended_manifest_schema(changed)
    changed = _future_normwise_manifest()
    reordered = dict(changed)
    value = reordered.pop("normwise_adjoint_calibration_protocol")
    reordered["normwise_adjoint_calibration_protocol"] = value
    with pytest.raises(ValueError, match="order|manifest"):
        _MODULE._validate_amended_manifest_schema(reordered)
    for mutation in ("missing", "extra", "reordered"):
        changed = _future_normwise_manifest()
        files = changed["current_scientific_source"]["files"]
        if mutation == "missing":
            files.pop("scripts/rsta_normwise_adjoint.py")
        elif mutation == "extra":
            files["scripts/unreviewed.py"] = "0" * 64
        else:
            helper = files.pop("scripts/rsta_normwise_adjoint.py")
            files["scripts/rsta_normwise_adjoint.py"] = helper
        with pytest.raises(ValueError, match="source|order|files"):
            _MODULE._validate_amended_manifest_schema(changed)


def test_sign_control_manifest_authority_order_provenance_and_prior_domains(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches unvalidated authority blobs, a false calibration verdict, or ancestry drift."""
    source_root = _SCRIPT.parents[1]
    repository = tmp_path
    manifest = _future_normwise_manifest()
    prior = json.loads(
        (_SCRIPT.parents[1] / "docs" / "pass200_rsta_receipt_stage_a_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(_ROUNDTRIP_SOURCE_ORDER) == 32
    assert tuple(_MODULE._CURRENT_SCIENTIFIC_SOURCE_FILES) == _ROUNDTRIP_SOURCE_ORDER
    for name in (
        "schema_version",
        "base_preregistration",
        "amendment",
        "deterministic_pool_amendment",
        "zero_jacobian_classifier_amendment",
        "adjoint_integrity_amendment",
        "binding_receipt",
        "historical",
        "artifact_schema",
        "seeds",
    ):
        assert manifest[name] == prior[name]
    revision = "c" * 40
    executing = "d" * 40
    manifest["current_scientific_source"]["git_revision"] = revision
    for path_text in _ROUNDTRIP_SOURCE_ORDER:
        destination = repository / path_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / path_text).read_bytes())
        manifest["current_scientific_source"]["files"][path_text] = _sha256_file(
            destination
        )
    reference_names = (
        "base_preregistration",
        "amendment",
        "deterministic_pool_amendment",
        "zero_jacobian_classifier_amendment",
        "adjoint_integrity_amendment",
        "normwise_adjoint_calibration_protocol",
        "normwise_adjoint_calibration_result",
        "normwise_adjoint_amendment",
        "normwise_adjoint_sign_control_amendment",
        "scientific_artifact_roundtrip_recovery_amendment",
        "artifact_schema",
    )
    for name in reference_names:
        path_text = manifest[name]["path"]
        destination = repository / path_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((source_root / path_text).read_bytes())
    manifest_path = repository / "docs" / "future-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        _MODULE,
        "__file__",
        str(repository / "scripts" / "diagnose_pass200_rsta_stage_a.py"),
    )
    bad_blob: str | None = None

    def git_blob(_repository: Path, _revision: str, path_text: str) -> bytes:
        if path_text == bad_blob:
            return b"substituted Git blob"
        return (repository / path_text).read_bytes()

    monkeypatch.setattr(_MODULE, "_git_blob", git_blob)
    failed_ancestry_edge: tuple[str, str] | None = None
    ancestry_edges: list[tuple[str, str]] = []

    def fake_run(args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, stdout=executing + "\n", stderr="")
        if "merge-base" in args:
            edge = (args[-2], args[-1])
            ancestry_edges.append(edge)
            return subprocess.CompletedProcess(
                args, 1 if edge == failed_ancestry_edge else 0, stdout="", stderr=""
            )
        raise AssertionError(f"unexpected subprocess call: {args}")

    monkeypatch.setattr(subprocess, "run", fake_run)
    audit = _MODULE.validate_scientific_execution_source(manifest_path)
    assert audit["frozen_source_revision"] == revision
    expected_ancestry_edges = [
        (
            manifest["normwise_adjoint_calibration_protocol"]["commit"],
            manifest["normwise_adjoint_calibration_result"]["commit"],
        ),
        (
            manifest["normwise_adjoint_calibration_result"]["commit"],
            manifest["normwise_adjoint_amendment"]["commit"],
        ),
        (
            manifest["normwise_adjoint_amendment"]["commit"],
            manifest["normwise_adjoint_sign_control_amendment"]["commit"],
        ),
        (
            manifest["normwise_adjoint_sign_control_amendment"]["commit"],
            manifest["scientific_artifact_roundtrip_recovery_amendment"]["commit"],
        ),
        (
            manifest["scientific_artifact_roundtrip_recovery_amendment"]["commit"],
            revision,
        ),
        (revision, executing),
    ]
    assert ancestry_edges == expected_ancestry_edges

    for name in (
        "normwise_adjoint_calibration_protocol",
        "normwise_adjoint_calibration_result",
        "normwise_adjoint_amendment",
        "normwise_adjoint_sign_control_amendment",
        "scientific_artifact_roundtrip_recovery_amendment",
    ):
        path = repository / manifest[name]["path"]
        original = path.read_bytes()
        path.write_bytes(b"dirty normwise authority")
        with pytest.raises(ValueError, match="normwise|calibration|amendment|worktree"):
            _MODULE.validate_scientific_execution_source(manifest_path)
        path.write_bytes(original)
        bad_blob = manifest[name]["path"]
        with pytest.raises(ValueError, match="normwise|calibration|amendment|blob"):
            _MODULE.validate_scientific_execution_source(manifest_path)
        bad_blob = None

    bad_blob = "scripts/rsta_normwise_adjoint.py"
    with pytest.raises(ValueError, match="source|blob"):
        _MODULE.validate_scientific_execution_source(manifest_path)
    bad_blob = None
    for edge_to_fail in expected_ancestry_edges:
        failed_ancestry_edge = edge_to_fail
        with pytest.raises(ValueError, match="ancestor"):
            _MODULE.validate_scientific_execution_source(manifest_path)
    failed_ancestry_edge = None

    helper_path = repository / "scripts" / "rsta_normwise_adjoint.py"
    helper_bytes = helper_path.read_bytes()
    result_path = repository / manifest["normwise_adjoint_calibration_result"]["path"]
    calibration_loads: list[Path] = []
    original_load_strict_json = _MODULE.load_strict_json

    def track_calibration_load(path: Path) -> Any:
        if path.resolve() == result_path.resolve():
            calibration_loads.append(path.resolve())
        return original_load_strict_json(path)

    monkeypatch.setattr(_MODULE, "load_strict_json", track_calibration_load)
    helper_imports: list[str] = []
    original_import = builtins.__import__

    def tracked_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "rsta_normwise_adjoint":
            helper_imports.append(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", tracked_import)
    helper_path.write_bytes(b"dirty helper must not execute")
    with pytest.raises(ValueError, match="source|worktree"):
        _MODULE.validate_scientific_execution_source(manifest_path)
    assert helper_imports == []
    assert calibration_loads == []
    monkeypatch.setattr(_MODULE, "load_strict_json", original_load_strict_json)
    helper_path.write_bytes(helper_bytes)

    execution_marker = repository / "helper-executed"
    substituted_helper = (
        "from pathlib import Path\n"
        f"Path({str(execution_marker)!r}).write_text('executed')\n"
        "def normwise_adjoint_metrics(*args, **kwargs): return {}\n"
        "def parameter_tree_sha256(*args, **kwargs): return '0' * 64\n"
        "def tensor_sha256(*args, **kwargs): return '0' * 64\n"
        "def validate_calibration_result(value): return None\n"
    ).encode()
    helper_path.write_bytes(substituted_helper)
    substituted_digest = hashlib.sha256(substituted_helper).hexdigest()
    manifest["current_scientific_source"]["files"][
        "scripts/rsta_normwise_adjoint.py"
    ] = substituted_digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="helper|source|SHA-256"):
        _MODULE.validate_scientific_execution_source(manifest_path)
    assert not execution_marker.exists()
    helper_path.write_bytes(helper_bytes)
    manifest["current_scientific_source"]["files"][
        "scripts/rsta_normwise_adjoint.py"
    ] = hashlib.sha256(helper_bytes).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    failed_result = json.loads(result_path.read_text(encoding="utf-8"))
    failed_result["all_passed"] = False
    result_path.write_text(json.dumps(failed_result), encoding="utf-8")
    failed_digest = _sha256_file(result_path)
    manifest["normwise_adjoint_calibration_result"]["sha256"] = failed_digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        _MODULE, "_NORMWISE_ADJOINT_CALIBRATION_RESULT_SHA256", failed_digest
    )
    with pytest.raises(ValueError, match="calibration|passed|result"):
        _MODULE.validate_scientific_execution_source(manifest_path)

    semantically_failed = json.loads(
        (
            source_root
            / manifest["normwise_adjoint_calibration_result"]["path"]
        ).read_text(encoding="utf-8")
    )
    failed_fixture = semantically_failed["correct_fixtures"]["zero_corner"]
    failed_fixture["controls"]["parameter_sign"]["exact_relation"] = False
    failed_fixture["controls"]["parameter_sign"]["passed"] = False
    failed_fixture["passed"] = False
    semantically_failed["all_passed"] = False
    result_path.write_text(json.dumps(semantically_failed), encoding="utf-8")
    failed_digest = _sha256_file(result_path)
    manifest["normwise_adjoint_calibration_result"]["sha256"] = failed_digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(
        _MODULE, "_NORMWISE_ADJOINT_CALIBRATION_RESULT_SHA256", failed_digest
    )
    with pytest.raises(ValueError, match="calibration|passed|result"):
        _MODULE.validate_scientific_execution_source(manifest_path)


def _task2_manifest(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    manifest = _future_normwise_manifest()
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def _task2_inputs(
    manifest: dict[str, Any],
) -> tuple[list[Any], Callable[..., Any], type[torch.nn.Module]]:
    example_ids, raw_labels = _selection_fixture()
    labels = [
        label if index < 210 else 1_000 + (index - 210) // 2
        for index, label in enumerate(raw_labels)
    ]
    generator = np.random.Generator(np.random.PCG64(2200))
    clean = _unit_numpy(generator.standard_normal((len(example_ids), 3))).astype(np.float32)
    proxy_labels = tuple(sorted(set(labels)))
    proxies = _unit_numpy(generator.standard_normal((len(proxy_labels), 3))).astype(np.float32)
    source_paths = tuple(f"/synthetic/{example_id}.jpg" for example_id in example_ids)
    bounds = []
    for seed in range(4):
        bound = _tiny_scientific_bound(
            seed=seed,
            clean=clean,
            labels=labels,
            example_ids=example_ids,
            source_paths=source_paths,
            proxies=proxies,
            proxy_labels=proxy_labels,
        )
        bounds.append(
            replace(
                bound,
                artifact_binding={
                    "artifacts": {
                        "checkpoint_pt": manifest["seeds"][str(seed)]["checkpoint_pt"],
                        "train_npz": manifest["seeds"][str(seed)]["train_npz"],
                    }
                },
            )
        )

    def cache_builder(_bound: Any, ordered_ids: Any, **_kwargs: Any) -> Any:
        sources = {value: value for value in ordered_ids}
        return _MODULE.cache_deterministic_transforms(
            ordered_ids,
            sources,
            transform=lambda example_id: torch.tensor(
                np.random.Generator(
                    np.random.PCG64(_MODULE.domain_seed("task2-transform", example_id))
                ).standard_normal(2),
                dtype=torch.float32,
            ),
        )

    class TinyOfficialHead(torch.nn.Module):
        def __init__(self, seed: int) -> None:
            super().__init__()
            torch.manual_seed(seed + 2200)
            self.model = torch.nn.Module()
            self.model.gmp = torch.nn.AdaptiveMaxPool2d(1)
            self.model.embedding = torch.nn.Linear(2, 512)

        def forward(self, values: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(self.model.embedding(values), dim=-1)

    return bounds, cache_builder, TinyOfficialHead


def _task2_adjoint_auditor(
    model: Any,
    _images: Any,
    output_direction: Any,
    parameter_direction: Any,
    *,
    output_direction_seed: int,
    parameter_direction_seed: int,
    passed: bool = True,
    legacy_passed: bool | None = None,
    integrity_passed: bool | None = None,
    control_failure: str | None = None,
    **_kwargs: Any,
) -> dict[str, Any]:
    metadata = _MODULE._adjoint_direction_metadata(
        model,
        output_direction,
        parameter_direction,
        output_direction_seed=output_direction_seed,
        parameter_direction_seed=parameter_direction_seed,
    )
    legacy_ok = passed if legacy_passed is None else legacy_passed
    integrity_ok = passed if integrity_passed is None else integrity_passed
    rhs = 1.0 if legacy_ok and integrity_ok else (0.9999 if legacy_ok else 0.999)
    absolute_error = abs(1.0 - rhs)
    beta_norm = 0.0 if absolute_error == 0.0 else (2.0e-4 if integrity_ok else 1.0e-3)
    normwise_denominator = (
        2.0 if absolute_error == 0.0 else 2.0 * absolute_error / beta_norm
    )
    action_hash = "a" * 64
    controls: dict[str, dict[str, Any]] = {
        "rebuild": {
            "jvp_sha256": action_hash,
            "vjp_sha256": action_hash,
            "beta_norm": 0.0,
            "exact_action_hash_match": True,
            "passed": True,
        },
        "reversed_action_order": {
            "jvp_sha256": action_hash,
            "vjp_sha256": action_hash,
            "beta_norm": 0.0,
            "exact_action_hash_match": True,
            "passed": True,
        },
        "parameter_sign": {
            "jvp_sha256": "b" * 64,
            "vjp_sha256": action_hash,
            "reference_jvp_sha256": action_hash,
            "reference_vjp_sha256": action_hash,
            "beta_norm": 0.0,
            "reference_exact_action_hash_match": True,
            "exact_relation": True,
            "passed": True,
        },
        "output_sign": {
            "jvp_sha256": action_hash,
            "vjp_sha256": "b" * 64,
            "reference_jvp_sha256": action_hash,
            "reference_vjp_sha256": action_hash,
            "beta_norm": 0.0,
            "reference_exact_action_hash_match": True,
            "exact_relation": True,
            "passed": True,
        },
    }
    if control_failure in {"rebuild", "reversed_action_order"}:
        controls[control_failure]["jvp_sha256"] = "c" * 64
        controls[control_failure]["exact_action_hash_match"] = False
        controls[control_failure]["passed"] = False
    elif control_failure in {"parameter_sign", "output_sign"}:
        controls[control_failure]["exact_relation"] = False
        controls[control_failure]["passed"] = False
    control_passed = all(control["passed"] is True for control in controls.values())
    return {
        **metadata,
        "lhs": 1.0,
        "rhs": rhs,
        "absolute_error": absolute_error,
        "denominator": 1.0,
        "relative_error": absolute_error,
        "tolerance": 5.0e-4,
        "passed": legacy_ok,
        "output_direction_l2": 1.0,
        "parameter_direction_l2": 1.0,
        "jvp_l2": normwise_denominator / 2.0,
        "vjp_l2": normwise_denominator / 2.0,
        "normwise_denominator": normwise_denominator,
        "eta_norm": beta_norm / 2.0,
        "beta_norm": beta_norm,
        "lhs_absolute_product_sum": 1.0,
        "rhs_absolute_product_sum": abs(rhs),
        "lhs_cancellation_factor": 1.0,
        "rhs_cancellation_factor": 1.0,
        "jvp_sha256": action_hash,
        "vjp_sha256": action_hash,
        "controls": controls,
        "normwise_tolerance": 5.0e-4,
        "normwise_passed": integrity_ok,
        "integrity_passed": integrity_ok and control_passed,
    }


def _run_task2_cli(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    adjoint_auditor: Callable[..., dict[str, Any]] = _task2_adjoint_auditor,
    zero_jacobian_auditor: Callable[..., dict[str, Any]] | None = None,
    receipt_validator: Callable[..., Any] | None = None,
) -> tuple[Path, list[int]]:
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest_path, manifest = _task2_manifest(tmp_path)
    bounds, cache_builder, model_type = _task2_inputs(manifest)
    calls: list[int] = []

    def bound_loader(_entry: Any, receipt_seed: Any, **_kwargs: Any) -> Any:
        calls.append(receipt_seed.seed)
        return bounds[receipt_seed.seed]

    output = tmp_path / "all-seeds.json"
    source = manifest["current_scientific_source"]
    execution = {
        "executing_git_commit": "a" * 40,
        "diagnostic_path": "scripts/diagnose_pass200_rsta_stage_a.py",
        "diagnostic_sha256": source["files"]["scripts/diagnose_pass200_rsta_stage_a.py"],
        "frozen_source_revision": source["git_revision"],
    }
    monkeypatch.setattr(_MODULE, "validate_execution_audit", lambda *_args, **_kwargs: None)
    _MODULE.main(
        [
            "--manifest",
            str(manifest_path),
            "--binding-receipt",
            str(tmp_path / "receipt.json"),
            "--output",
            str(output),
            "--integrity-all-seeds-only",
        ],
        expected_dimension=512,
        receipt_validator=receipt_validator or (lambda *_args: _tiny_validated_receipt()),
        execution_source_validator=lambda _path: execution,
        bound_loader=bound_loader,
        cache_builder=cache_builder,
        model_loader=lambda bound: model_type(bound.seed).train(),
        fixture_runner=_task2_fixture_audit,
        deterministic_pool_auditor=_valid_global_max_audit,
        zero_jacobian_auditor=zero_jacobian_auditor
        or (lambda *_args: _valid_zero_jacobian_audit()),
        adjoint_auditor=adjoint_auditor,
    )
    return output, calls


def test_integrity_all_seeds_mode_is_candidate_free_and_exact_schema(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches candidate reachability or any widening of the frozen audit schema."""
    forbidden_calls: list[str] = []

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        forbidden_calls.append("candidate")
        raise AssertionError("candidate code reached from all-seed integrity")

    for name in (
        "exact_contextual_rsta_fields",
        "score_rsta_batch",
        "decide_stage_a",
        "joint_bootstrap",
        "scientific_payload",
        "_validate_receiver_audit_row",
    ):
        monkeypatch.setattr(_MODULE, name, forbidden)
    output, calls = _run_task2_cli(tmp_path, monkeypatch)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert calls == [0, 1, 2, 3]
    assert forbidden_calls == []
    assert list(payload) == [
        "schema_version", "diagnostic", "mode", "candidate_values_computed",
        "stage_a_verdict", "uses_test_data", "execution_audit", "manifest",
        "environment", "binding", "integrity",
    ]
    assert payload["schema_version"] == 1
    assert payload["diagnostic"] == "pass200-rsta-adjoint-integrity"
    assert payload["mode"] == "integrity_all_seeds"
    assert payload["candidate_values_computed"] is False
    assert payload["stage_a_verdict"] == "NOT_COMPUTED"
    assert payload["uses_test_data"] == "artifact_binding_only"
    assert set(payload["execution_audit"]) == _MODULE._EXECUTION_AUDIT_FIELDS
    assert list(payload["manifest"]) == [
        "path", "sha256", "base_preregistration", "amendment",
        "deterministic_pool_amendment", "zero_jacobian_classifier_amendment",
        "adjoint_integrity_amendment", "normwise_adjoint_calibration_protocol",
            "normwise_adjoint_calibration_result", "normwise_adjoint_amendment",
            "normwise_adjoint_sign_control_amendment",
            "scientific_artifact_roundtrip_recovery_amendment",
            "binding_receipt", "historical",
        "artifact_schema", "source",
    ]
    assert set(payload["environment"]) == _MODULE.ENVIRONMENT_AUDIT_FIELDS
    assert set(payload["binding"]) == {
        "receipt_sha256", "receipt_producer_commit", "historical_manifest_sha256", "seeds"
    }
    assert list(payload["binding"]["seeds"]) == ["0", "1", "2", "3"]
    assert all(
        set(value) == {
            "checkpoint_sha256", "train_pack_sha256", "first_batch_ordered_id_sha256",
            "transform_cache_order_sha256", "transform_tensor_set_sha256",
        }
        for value in payload["binding"]["seeds"].values()
    )
    assert set(payload["integrity"]) == {
        "dense_fixture", "bn_fixture", "deterministic_global_max", "seeds", "all_passed"
    }
    assert list(payload["integrity"]["seeds"]) == ["0", "1", "2", "3"]
    assert all(
        set(value) == {"zero_jacobian_classifier", "adjoint"}
        and list(value["adjoint"]) == list(_MODULE._ADJOINT_AUDIT_FIELDS)
        for value in payload["integrity"]["seeds"].values()
    )
    assert all(
        value["adjoint"]["output_shape"] == [180, 512]
        for value in payload["integrity"]["seeds"].values()
    )
    assert payload["integrity"]["all_passed"] is True

    def keys(value: Any) -> set[str]:
        if isinstance(value, dict):
            return set(value) | set().union(*(keys(item) for item in value.values()))
        if isinstance(value, list):
            return set().union(*(keys(item) for item in value))
        return set()

    assert not keys(payload) & {"rows", "fields", "scores", "decision", "aggregation", "bootstrap"}


def test_sign_control_candidate_free_projection_schema_rejects_every_authority_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, _calls = _run_task2_cli(tmp_path, monkeypatch)
    payload = json.loads(output.read_text(encoding="utf-8"))
    expected_projection = (
        "path", "sha256", "base_preregistration", "amendment",
        "deterministic_pool_amendment", "zero_jacobian_classifier_amendment",
        "adjoint_integrity_amendment", "normwise_adjoint_calibration_protocol",
        "normwise_adjoint_calibration_result", "normwise_adjoint_amendment",
        "normwise_adjoint_sign_control_amendment",
        "scientific_artifact_roundtrip_recovery_amendment", "binding_receipt",
        "historical", "artifact_schema", "source",
    )
    sign_keys = (
        "jvp_sha256", "vjp_sha256", "reference_jvp_sha256",
        "reference_vjp_sha256", "beta_norm",
        "reference_exact_action_hash_match", "exact_relation", "passed",
    )
    assert tuple(payload["manifest"]) == expected_projection
    for seed in payload["integrity"]["seeds"].values():
        for name in ("parameter_sign", "output_sign"):
            assert tuple(seed["adjoint"]["controls"][name]) == sign_keys
    authority = "normwise_adjoint_sign_control_amendment"
    for mutation in ("missing", "extra", "reordered", "path", "sha256", "commit"):
        changed = deepcopy(payload)
        record = changed["manifest"]
        if mutation == "missing":
            record.pop(authority)
        elif mutation == "extra":
            record[authority]["unchecked"] = True
        elif mutation == "reordered":
            value = record.pop(authority)
            record[authority] = value
        else:
            record[authority][mutation] = {
                "path": "docs/substituted.md",
                "sha256": "0" * 64,
                "commit": "0" * 40,
            }[mutation]
        with pytest.raises(ValueError, match="manifest|projection|sign|amendment"):
            _MODULE.validate_all_seed_adjoint_integrity_payload(changed)


def test_integrity_all_seeds_recursively_validates_execution_manifest_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches trusting copied provenance/environment objects or accepting nested drift."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest_path, manifest = _task2_manifest(tmp_path)
    environment = _MODULE.configure_deterministic_process()
    source = manifest["current_scientific_source"]
    execution = {
        "executing_git_commit": "a" * 40,
        "diagnostic_path": "scripts/diagnose_pass200_rsta_stage_a.py",
        "diagnostic_sha256": source["files"]["scripts/diagnose_pass200_rsta_stage_a.py"],
        "frozen_source_revision": source["git_revision"],
    }
    manifest_audit = {
        "path": str(manifest_path), "sha256": _sha256_file(manifest_path),
        "base_preregistration": manifest["base_preregistration"],
        "amendment": manifest["amendment"],
        "deterministic_pool_amendment": manifest["deterministic_pool_amendment"],
        "zero_jacobian_classifier_amendment": manifest["zero_jacobian_classifier_amendment"],
        "adjoint_integrity_amendment": manifest["adjoint_integrity_amendment"],
        "normwise_adjoint_calibration_protocol": manifest[
            "normwise_adjoint_calibration_protocol"
        ],
        "normwise_adjoint_calibration_result": manifest[
            "normwise_adjoint_calibration_result"
        ],
        "normwise_adjoint_amendment": manifest["normwise_adjoint_amendment"],
        "normwise_adjoint_sign_control_amendment": manifest[
            "normwise_adjoint_sign_control_amendment"
        ],
        "scientific_artifact_roundtrip_recovery_amendment": manifest[
            "scientific_artifact_roundtrip_recovery_amendment"
        ],
        "binding_receipt": manifest["binding_receipt"], "historical": manifest["historical"],
        "artifact_schema": manifest["artifact_schema"], "source": source,
    }
    adjoint = _extended_normwise_adjoint({
        "direction_domain": "rsta-stage-a-v1", "output_direction_seed": 1,
        "parameter_direction_seed": 2, "output_direction_sha256": "1" * 64,
        "parameter_direction_sha256": "2" * 64, "output_shape": [180, 512],
        "parameter_name_order_sha256": "3" * 64, "parameter_count": 9,
        "model_dtype": "torch.float32", "reduction_dtype": "torch.float64",
        "lhs": 1.0, "rhs": 1.0, "absolute_error": 0.0, "denominator": 1.0,
        "relative_error": 0.0, "tolerance": 0.0005, "passed": True,
    })
    payload = {
        "schema_version": 1, "diagnostic": "pass200-rsta-adjoint-integrity",
        "mode": "integrity_all_seeds", "candidate_values_computed": False,
        "stage_a_verdict": "NOT_COMPUTED", "uses_test_data": "artifact_binding_only",
        "execution_audit": execution, "manifest": manifest_audit, "environment": environment,
        "binding": {
            "receipt_sha256": _MODULE._HISTORICAL_RECEIPT_SHA256,
            "receipt_producer_commit": _MODULE._HISTORICAL_PRODUCER_COMMIT,
            "historical_manifest_sha256": _MODULE._HISTORICAL_MANIFEST_SHA256,
            "seeds": {str(seed): {
                "checkpoint_sha256": manifest["seeds"][str(seed)]["checkpoint_pt"]["sha256"],
                "train_pack_sha256": manifest["seeds"][str(seed)]["train_npz"]["sha256"],
                "first_batch_ordered_id_sha256": "9" * 64,
                "transform_cache_order_sha256": "a" * 64,
                "transform_tensor_set_sha256": "b" * 64,
            } for seed in range(4)},
        },
        "integrity": {
            **_task2_fixture_audit(),
            "deterministic_global_max": _valid_global_max_audit(),
            "seeds": {
                str(seed): {
                    "zero_jacobian_classifier": _valid_zero_jacobian_audit(),
                    "adjoint": {
                        **deepcopy(adjoint),
                        "output_direction_seed": _MODULE.domain_seed(
                            "rsta-stage-a-v1|adjoint-u|", str(seed)
                        ),
                        "parameter_direction_seed": _MODULE.domain_seed(
                            "rsta-stage-a-v1|adjoint-v|", str(seed)
                        ),
                    },
                }
                for seed in range(4)
            },
            "all_passed": True,
        },
    }
    calls: list[dict[str, Any]] = []

    def execution_validator(value: Any, *, manifest_source: Any, manifest_path: Any) -> None:
        calls.append(dict(value))
        if (
            value != execution
            or manifest_source != source
            or manifest_path != Path(manifest_audit["path"])
        ):
            raise ValueError("execution audit differs")

    monkeypatch.setattr(_MODULE, "validate_execution_audit", execution_validator)
    _MODULE.validate_all_seed_adjoint_integrity_payload(payload)
    assert calls and set(environment) == _MODULE.ENVIRONMENT_AUDIT_FIELDS

    def paths(value: Any, prefix: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
        found = [prefix]
        if isinstance(value, dict):
            for key, item in value.items():
                found.extend(paths(item, (*prefix, key)))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                found.extend(paths(item, (*prefix, index)))
        return found

    for root_name in ("execution_audit", "manifest", "environment"):
        root = payload[root_name]
        for path in paths(root):
            changed = deepcopy(payload)
            target = changed[root_name]
            for part in path[:-1]:
                target = target[part]
            if not path:
                changed[root_name] = None
            elif isinstance(target[path[-1]], dict):
                target[path[-1]]["unexpected"] = True
            elif isinstance(target[path[-1]], list):
                target[path[-1]].append(None)
            else:
                target[path[-1]] = None
            with pytest.raises((ValueError, TypeError, FileNotFoundError)):
                _MODULE.validate_all_seed_adjoint_integrity_payload(changed)

    authority_mutations = (
        ("receipt_sha256", "0" * 64),
        ("receipt_producer_commit", "0" * 40),
        ("historical_manifest_sha256", "0" * 64),
    )
    for name, replacement in authority_mutations:
        changed = deepcopy(payload)
        changed["binding"][name] = replacement
        with pytest.raises(ValueError, match="authority|binding|receipt|historical"):
            _MODULE.validate_all_seed_adjoint_integrity_payload(changed)

    fabricated = deepcopy(payload)
    fabricated["binding"]["receipt_sha256"] = "4" * 64
    fabricated["binding"]["receipt_producer_commit"] = "5" * 40
    fabricated["binding"]["historical_manifest_sha256"] = "6" * 64
    with pytest.raises(ValueError, match="authority|binding|receipt|historical"):
        _MODULE.validate_all_seed_adjoint_integrity_payload(fabricated)

    for seed in range(4):
        for field in ("checkpoint_sha256", "train_pack_sha256"):
            changed = deepcopy(payload)
            changed["binding"]["seeds"][str(seed)][field] = "f" * 64
            with pytest.raises(ValueError, match="manifest|binding"):
                _MODULE.validate_all_seed_adjoint_integrity_payload(changed)
    for field in (
        "first_batch_ordered_id_sha256",
        "transform_cache_order_sha256",
        "transform_tensor_set_sha256",
    ):
        changed = deepcopy(payload)
        changed["binding"]["seeds"]["2"][field] = "f" * 64
        with pytest.raises(ValueError, match="common|cache|binding"):
            _MODULE.validate_all_seed_adjoint_integrity_payload(changed)

    for seed in range(4):
        for field, domain in (
            ("output_direction_seed", "rsta-stage-a-v1|adjoint-u|"),
            ("parameter_direction_seed", "rsta-stage-a-v1|adjoint-v|"),
        ):
            changed = deepcopy(payload)
            changed["integrity"]["seeds"][str(seed)]["adjoint"][field] = (
                _MODULE.domain_seed(domain, str((seed + 1) % 4))
            )
            with pytest.raises(ValueError, match="seed|adjoint"):
                _MODULE.validate_all_seed_adjoint_integrity_payload(changed)
        for wrong_shape in ([180, 511], [179, 512], [180, 512, 1]):
            changed = deepcopy(payload)
            changed["integrity"]["seeds"][str(seed)]["adjoint"]["output_shape"] = wrong_shape
            with pytest.raises(ValueError, match="shape|adjoint"):
                _MODULE.validate_all_seed_adjoint_integrity_payload(changed)


def test_integrity_all_seeds_records_all_finite_adjoint_failures_without_candidate_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches fail-fast treatment of a finite tolerance failure in candidate-free mode."""
    seed_by_output_seed = {
        _MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", str(seed)): seed
        for seed in range(4)
    }
    adjoint_calls: list[int] = []
    candidate_calls: list[str] = []
    publications: list[Path] = []

    def adjoint(*args: Any, output_direction_seed: int, **kwargs: Any) -> dict[str, Any]:
        seed = seed_by_output_seed[output_direction_seed]
        adjoint_calls.append(seed)
        return _task2_adjoint_auditor(
            *args,
            output_direction_seed=output_direction_seed,
            passed=seed != 1,
            **kwargs,
        )

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        candidate_calls.append("candidate")
        raise AssertionError("candidate code reached")

    for name in (
        "exact_contextual_rsta_fields", "score_rsta_batch", "decide_stage_a",
        "joint_bootstrap", "scientific_payload", "_validate_receiver_audit_row",
    ):
        monkeypatch.setattr(_MODULE, name, forbidden)
    real_write = _MODULE.write_json_atomic

    def write_once(path: Path, payload: dict[str, Any], **kwargs: Any) -> None:
        publications.append(path)
        real_write(path, payload, **kwargs)

    monkeypatch.setattr(_MODULE, "write_json_atomic", write_once)
    output, _ = _run_task2_cli(tmp_path, monkeypatch, adjoint_auditor=adjoint)
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert adjoint_calls == [0, 1, 2, 3]
    assert list(payload["integrity"]["seeds"]) == ["0", "1", "2", "3"]
    assert payload["integrity"]["seeds"]["1"]["adjoint"]["passed"] is False
    assert payload["integrity"]["all_passed"] is False
    assert candidate_calls == []
    assert publications == [output]


def test_normwise_integrity_all_seeds_uses_integrity_verdict_not_legacy_passed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches retaining the legacy scalar verdict as the prospective global gate."""
    seed_by_output_seed = {
        _MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", str(seed)): seed
        for seed in range(4)
    }

    def adjoint(*args: Any, output_direction_seed: int, **kwargs: Any) -> dict[str, Any]:
        seed = seed_by_output_seed[output_direction_seed]
        return _task2_adjoint_auditor(
            *args,
            output_direction_seed=output_direction_seed,
            legacy_passed=seed != 1,
            integrity_passed=True,
            **kwargs,
        )

    output, _ = _run_task2_cli(tmp_path, monkeypatch, adjoint_auditor=adjoint)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["integrity"]["seeds"]["1"]["adjoint"]["passed"] is False
    assert payload["integrity"]["seeds"]["1"]["adjoint"]["integrity_passed"] is True
    assert payload["integrity"]["all_passed"] is True


@pytest.mark.parametrize(
    "failure",
    [
        "normwise",
        "rebuild",
        "reversed_action_order",
        "parameter_sign",
        "output_sign",
        "zero_denominator",
    ],
)
def test_normwise_candidate_free_records_complete_four_seed_finite_failure_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Catches fail-fast or candidate reachability for every authorized finite gate failure."""
    seed_by_output_seed = {
        _MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", str(seed)): seed
        for seed in range(4)
    }
    calls: list[int] = []

    def adjoint(*args: Any, output_direction_seed: int, **kwargs: Any) -> dict[str, Any]:
        seed = seed_by_output_seed[output_direction_seed]
        calls.append(seed)
        audit = _task2_adjoint_auditor(
            *args,
            output_direction_seed=output_direction_seed,
            integrity_passed=not (seed == 2 and failure in {"normwise", "zero_denominator"}),
            control_failure=(
                failure
                if seed == 2 and failure not in {"normwise", "zero_denominator"}
                else None
            ),
            **kwargs,
        )
        if seed == 2 and failure == "zero_denominator":
            audit.update(
                {
                    "lhs": 1.0,
                    "rhs": 0.0,
                    "absolute_error": 1.0,
                    "denominator": 1.0,
                    "relative_error": 1.0,
                    "passed": False,
                    "output_direction_l2": 0.0,
                    "parameter_direction_l2": 0.0,
                    "jvp_l2": 0.0,
                    "vjp_l2": 0.0,
                    "normwise_denominator": 0.0,
                    "eta_norm": "infinity",
                    "beta_norm": "infinity",
                    "lhs_absolute_product_sum": 1.0,
                    "rhs_absolute_product_sum": 0.0,
                    "normwise_passed": False,
                    "integrity_passed": False,
                }
            )
            for control in audit["controls"].values():
                control["beta_norm"] = "infinity"
                control["passed"] = False
        return audit

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("candidate code reached from candidate-free normwise failure")

    for name in (
        "exact_contextual_rsta_fields",
        "score_rsta_batch",
        "decide_stage_a",
        "joint_bootstrap",
        "scientific_payload",
        "_validate_receiver_audit_row",
    ):
        monkeypatch.setattr(_MODULE, name, forbidden)
    output, _ = _run_task2_cli(tmp_path, monkeypatch, adjoint_auditor=adjoint)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert calls == [0, 1, 2, 3]
    assert list(payload["integrity"]["seeds"]) == ["0", "1", "2", "3"]
    assert payload["integrity"]["seeds"]["2"]["adjoint"]["integrity_passed"] is False
    assert payload["integrity"]["all_passed"] is False


@pytest.mark.parametrize(
    "failure",
    [
        "structural_provenance",
        "nonfinite_direction",
        "nonfinite_reduction",
        "zero_jacobian",
        "serialization",
        "atomic_publication",
    ],
)
def test_integrity_all_seeds_fail_fast_without_destination_or_sibling_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    """Catches continuation/publication after any structural candidate-free failure."""
    adjoint_calls: list[int] = []
    seed_by_output_seed = {
        _MODULE.domain_seed("rsta-stage-a-v1|adjoint-u|", str(seed)): seed
        for seed in range(4)
    }

    def adjoint(*args: Any, output_direction_seed: int, **kwargs: Any) -> dict[str, Any]:
        seed = seed_by_output_seed[output_direction_seed]
        adjoint_calls.append(seed)
        if failure == "nonfinite_direction" and seed == 1:
            raise ValueError("adjoint direction is nonfinite")
        if failure == "nonfinite_reduction" and seed == 1:
            raise ValueError("adjoint reduction is nonfinite")
        return _task2_adjoint_auditor(
            *args, output_direction_seed=output_direction_seed, **kwargs
        )

    def fail_receipt(*_args: Any) -> Any:
        raise ValueError("structural provenance failed")

    receipt_validator = fail_receipt if failure == "structural_provenance" else None

    def invalid_zero_jacobian(*_args: Any) -> dict[str, Any]:
        return {}

    zero = invalid_zero_jacobian if failure == "zero_jacobian" else None
    if failure == "serialization":
        real_dumps = json.dumps

        def fail_payload_json(value: Any, *args: Any, **kwargs: Any) -> str:
            if isinstance(value, dict) and value.get("diagnostic") == (
                "pass200-rsta-adjoint-integrity"
            ):
                raise TypeError("serialization failed")
            return real_dumps(value, *args, **kwargs)

        monkeypatch.setattr(json, "dumps", fail_payload_json)
    if failure == "atomic_publication":
        monkeypatch.setattr(
            os,
            "link",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("publication failed")),
        )

    expected_message = {
        "structural_provenance": "structural provenance failed",
        "nonfinite_direction": "adjoint direction is nonfinite",
        "nonfinite_reduction": "adjoint reduction is nonfinite",
        "zero_jacobian": "zero-Jacobian classifier",
        "serialization": "serialization failed",
        "atomic_publication": "publication failed",
    }[failure]
    with pytest.raises((ValueError, TypeError, OSError), match=expected_message):
        _run_task2_cli(
            tmp_path,
            monkeypatch,
            adjoint_auditor=adjoint,
            zero_jacobian_auditor=zero,
            receipt_validator=receipt_validator,
        )

    output = tmp_path / "all-seeds.json"
    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []
    if failure == "structural_provenance":
        assert adjoint_calls == []
    elif failure in {"nonfinite_direction", "nonfinite_reduction"}:
        assert adjoint_calls == [0, 1]


def test_scientific_source_authenticates_adjoint_integrity_amendment_bytes_and_blob(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches an absent, substituted, dirty, or Git-unbound prospective amendment."""
    expected_path = "docs/pass200_rsta_adjoint_integrity_amendment_2026-08-09.md"
    expected_sha = "2187aa4ae343c77e50cee28d1a64d0f5e31464ea220e40b8b4b95abf0f183b2c"
    expected_commit = "4c6886997b4116dcdb4ee5057e9544852695b42d"
    assert expected_path == _MODULE._ADJOINT_INTEGRITY_AMENDMENT_PATH
    assert expected_sha == _MODULE._ADJOINT_INTEGRITY_AMENDMENT_SHA256
    assert expected_commit == _MODULE._ADJOINT_INTEGRITY_AMENDMENT_COMMIT

    repository = tmp_path
    source_amendment = (
        _SCRIPT.parents[1]
        / "docs"
        / "pass200_rsta_adjoint_integrity_amendment_2026-08-09.md"
    )
    amendment_path = repository / expected_path
    amendment_path.parent.mkdir(parents=True)
    amendment_bytes = source_amendment.read_bytes()
    amendment_path.write_bytes(amendment_bytes)
    references: dict[str, dict[str, str]] = {}
    old_bindings = (
        ("amendment", "_AMENDMENT_PATH", "_AMENDMENT_SHA256", "_AMENDMENT_COMMIT"),
        (
            "deterministic_pool_amendment",
            "_DETERMINISTIC_POOL_AMENDMENT_PATH",
            "_DETERMINISTIC_POOL_AMENDMENT_SHA256",
            "_DETERMINISTIC_POOL_AMENDMENT_COMMIT",
        ),
        (
            "zero_jacobian_classifier_amendment",
            "_ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_PATH",
            "_ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_SHA256",
            "_ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_COMMIT",
        ),
    )
    for index, (name, path_constant, sha_constant, commit_constant) in enumerate(
        old_bindings
    ):
        path_text = f"docs/{name}.md"
        data = f"{name}\n".encode()
        path = repository / path_text
        path.write_bytes(data)
        digest = hashlib.sha256(data).hexdigest()
        commit = f"{index + 1}" * 40
        monkeypatch.setattr(_MODULE, path_constant, path_text)
        monkeypatch.setattr(_MODULE, sha_constant, digest)
        monkeypatch.setattr(_MODULE, commit_constant, commit)
        references[name] = {"path": path_text, "sha256": digest, "commit": commit}
    for name in ("base_preregistration", "artifact_schema"):
        path_text = f"docs/{name}.json"
        data = f"{name}\n".encode()
        (repository / path_text).write_bytes(data)
        references[name] = {
            "path": path_text,
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    for name, path_text, digest, commit in (
        (
            "normwise_adjoint_calibration_protocol",
            _MODULE._NORMWISE_ADJOINT_CALIBRATION_PROTOCOL_PATH,
            _MODULE._NORMWISE_ADJOINT_CALIBRATION_PROTOCOL_SHA256,
            _MODULE._NORMWISE_ADJOINT_CALIBRATION_PROTOCOL_COMMIT,
        ),
        (
            "normwise_adjoint_calibration_result",
            _MODULE._NORMWISE_ADJOINT_CALIBRATION_RESULT_PATH,
            _MODULE._NORMWISE_ADJOINT_CALIBRATION_RESULT_SHA256,
            _MODULE._NORMWISE_ADJOINT_CALIBRATION_RESULT_COMMIT,
        ),
            (
                "normwise_adjoint_amendment",
                _MODULE._NORMWISE_ADJOINT_AMENDMENT_PATH,
                _MODULE._NORMWISE_ADJOINT_AMENDMENT_SHA256,
                _MODULE._NORMWISE_ADJOINT_AMENDMENT_COMMIT,
            ),
            (
                "normwise_adjoint_sign_control_amendment",
                _MODULE._NORMWISE_ADJOINT_SIGN_CONTROL_AMENDMENT_PATH,
                _MODULE._NORMWISE_ADJOINT_SIGN_CONTROL_AMENDMENT_SHA256,
                _MODULE._NORMWISE_ADJOINT_SIGN_CONTROL_AMENDMENT_COMMIT,
            ),
            (
                "scientific_artifact_roundtrip_recovery_amendment",
                _MODULE._SCIENTIFIC_ARTIFACT_ROUNDTRIP_RECOVERY_AMENDMENT_PATH,
                _MODULE._SCIENTIFIC_ARTIFACT_ROUNDTRIP_RECOVERY_AMENDMENT_SHA256,
                _MODULE._SCIENTIFIC_ARTIFACT_ROUNDTRIP_RECOVERY_AMENDMENT_COMMIT,
            ),
    ):
        destination = repository / path_text
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((_SCRIPT.parents[1] / path_text).read_bytes())
        references[name] = {"path": path_text, "sha256": digest, "commit": commit}
    diagnostic_path = repository / "scripts" / "diagnose_pass200_rsta_stage_a.py"
    diagnostic_path.parent.mkdir(parents=True)
    diagnostic_path.write_text("EXECUTION = 1\n", encoding="utf-8")
    helper_path = repository / "scripts" / "rsta_normwise_adjoint.py"
    helper_path.write_bytes((_SCRIPT.parent / "rsta_normwise_adjoint.py").read_bytes())
    revision = "c" * 40
    manifest = {
        **references,
        "adjoint_integrity_amendment": {
            "path": expected_path,
            "sha256": expected_sha,
            "commit": expected_commit,
        },
        "current_scientific_source": {
                "git_revision": revision,
                "files": {
                    "scripts/diagnose_pass200_rsta_stage_a.py": _sha256_file(diagnostic_path),
                    "scripts/rsta_normwise_adjoint.py": _sha256_file(helper_path),
                },
        },
    }
    manifest_path = repository / "docs" / "manifest.json"
    monkeypatch.setattr(_MODULE, "_validate_amended_manifest_schema", lambda value: value)
    monkeypatch.setattr(_MODULE, "__file__", str(diagnostic_path))
    bad_adjoint_blob = False

    def git_blob(_repository: Path, _revision: str, path_text: str) -> bytes:
        if path_text == expected_path and bad_adjoint_blob:
            return b"substituted amendment blob"
        return (repository / path_text).read_bytes()

    monkeypatch.setattr(_MODULE, "_git_blob", git_blob)

    def fake_run(args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[Any]:
        if "rev-parse" in args:
            return subprocess.CompletedProcess(args, 0, stdout=revision + "\n", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    def write(value: dict[str, Any]) -> None:
        manifest_path.write_text(json.dumps(value), encoding="utf-8")

    write(manifest)
    audit = _MODULE.validate_scientific_execution_source(manifest_path)
    assert audit["frozen_source_revision"] == revision

    for mutation in ("missing", "extra", "path", "sha256", "commit"):
        changed = deepcopy(manifest)
        if mutation == "missing":
            changed.pop("adjoint_integrity_amendment")
        elif mutation == "extra":
            changed["adjoint_integrity_amendment"]["extra"] = True
        else:
            changed["adjoint_integrity_amendment"][mutation] = "0" * (
                64 if mutation == "sha256" else 40
            )
        write(changed)
        with pytest.raises((KeyError, ValueError), match="adjoint|amendment"):
            _MODULE.validate_scientific_execution_source(manifest_path)

    write(manifest)
    amendment_path.write_bytes(b"dirty worktree amendment")
    with pytest.raises(ValueError, match="adjoint|amendment"):
        _MODULE.validate_scientific_execution_source(manifest_path)
    amendment_path.write_bytes(amendment_bytes)
    bad_adjoint_blob = True
    with pytest.raises(ValueError, match="adjoint|amendment"):
        _MODULE.validate_scientific_execution_source(manifest_path)


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

    cache_calls: list[tuple[str, ...]] = []

    def cache_builder(bound: Any, ordered_ids: Any, **_kwargs: Any) -> Any:
        cache_calls.append(tuple(ordered_ids))
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
    field_phases: list[str] = []
    integrity_field_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    scoring_tensor_refs: list[weakref.ReferenceType[torch.Tensor]] = []

    original_configure = _MODULE.configure_deterministic_process

    def configure_first() -> dict[str, Any]:
        integrity_score_events.append("configure")
        return original_configure()

    monkeypatch.setattr(_MODULE, "configure_deterministic_process", configure_first)
    original_fields = _MODULE.exact_contextual_rsta_fields

    def record_field_construction(*args: Any, **kwargs: Any) -> dict[str, Any]:
        phase = "scoring" if "integrity-3" in integrity_score_events else "integrity"
        field_phases.append(phase)
        result = original_fields(*args, **kwargs)
        if phase == "scoring":
            scoring_tensor_refs.extend(
                weakref.ref(value) for value in result.values() if torch.is_tensor(value)
            )
        return result

    monkeypatch.setattr(_MODULE, "exact_contextual_rsta_fields", record_field_construction)
    original_integrity = _MODULE._registered_first_batch_integrity

    def record_integrity_fields(*args: Any, **kwargs: Any) -> dict[str, Any]:
        gc.collect()
        assert all(reference() is None for reference in integrity_field_refs)
        result = original_integrity(*args, **kwargs)
        integrity_field_refs.append(weakref.ref(result["fields"]["z"]))
        return result

    monkeypatch.setattr(_MODULE, "_registered_first_batch_integrity", record_integrity_fields)

    def audit_before_scoring(path: Path) -> dict[str, Any]:
        integrity_score_events.append("execution-audit")
        manifest = _MODULE.load_strict_json(path)
        return _MODULE.build_execution_audit(manifest, manifest_path=path)

    def rotation_auditor(*args: Any, seed: int, **kwargs: Any) -> dict[str, Any]:
        rotation_calls.append(seed)
        integrity_score_events.append(f"integrity-{seed}")
        return _MODULE._default_rotation_auditor(*args, seed=seed, **kwargs)

    original_score = _MODULE.score_rsta_batch
    original_decide = _MODULE.decide_stage_a

    def score_after_integrity(*args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        integrity_score_events.append(f"score-{kwargs['seed']}")
        gc.collect()
        assert all(reference() is None for reference in integrity_field_refs)
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
    cache_calls.clear()
    integrity_score_events.clear()

    scoring_model_refs: list[weakref.ReferenceType[torch.nn.Module]] = []
    scoring_parameter_refs: list[weakref.ReferenceType[torch.Tensor]] = []
    loader_calls = 0
    clone_calls = 0
    original_clone = _MODULE.make_rsta_diagnostic_clone

    def track_clone(model: torch.nn.Module) -> torch.nn.Module:
        nonlocal clone_calls
        clone = original_clone(model)
        clone_calls += 1
        if clone_calls > 4:
            scoring_model_refs.append(weakref.ref(clone))
            scoring_parameter_refs.extend(weakref.ref(value) for value in clone.parameters())
        return clone

    def tracked_model_loader(bound: Any) -> torch.nn.Module:
        nonlocal loader_calls
        loader_calls += 1
        if loader_calls > 5:
            gc.collect()
            assert all(reference() is None for reference in scoring_model_refs)
            assert all(reference() is None for reference in scoring_parameter_refs)
            assert all(reference() is None for reference in scoring_tensor_refs)
        return TinyOfficialHead(bound.seed).train()

    monkeypatch.setattr(_MODULE, "make_rsta_diagnostic_clone", track_clone)

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
        model_loader=tracked_model_loader,
        fixture_runner=lambda: fixture_audit,
        deterministic_pool_auditor=_valid_global_max_audit,
        zero_jacobian_auditor=zero_jacobian_auditor,
        rotation_auditor=rotation_auditor,
        head_name="model.embedding",
        expected_head_in_features=2,
    )

    result = json.loads(output.read_text(encoding="utf-8"))
    for seed_audit in result["integrity"]["seeds"]:
        assert list(seed_audit["adjoint"]) == list(_MODULE._ADJOINT_AUDIT_FIELDS)
        _MODULE._validate_adjoint_integrity_audit(
            seed_audit["adjoint"], expected_output_shape=(180, 3)
        )
    assert validated == [manifest_path]
    assert bound_calls == [0, 1, 2, 3]
    primary = _MODULE.select_primary_panel(example_ids, labels)
    alternate = _MODULE.select_alternate_panel(example_ids, labels, primary)
    scoring_ids = tuple(
        sorted(
            {
                example_id
                for batch in [*primary["batches"], *alternate["batches"]]
                for example_id in batch
            }
        )
    )
    assert cache_calls == [tuple(primary["batches"][0]), scoring_ids]
    assert rotation_calls == [0, 1, 2, 3]
    assert zero_jacobian_calls == [180] * 8
    assert integrity_score_events[:2] == ["configure", "execution-audit"]
    assert integrity_score_events.index("execution-audit") < integrity_score_events.index(
        "decision"
    )
    integrity_events = [
        event for event in integrity_score_events if event.startswith("integrity-")
    ]
    score_events = [event for event in integrity_score_events if event.startswith("score-")]
    assert integrity_events == ["integrity-0", "integrity-1", "integrity-2", "integrity-3"]
    assert integrity_score_events.index("integrity-3") < integrity_score_events.index(
        score_events[0]
    )
    assert integrity_score_events.index(score_events[-1]) < integrity_score_events.index(
        "decision"
    )
    assert "integrity" in field_phases and "scoring" in field_phases
    assert result["mode"] == "scientific"
    assert len(result["rows"]["primary"]) == 4 * 64
    assert len(result["rows"]["alternate"]) == 4 * 16
    persisted_primary = result["panel_binding"]["primary"]
    assert list(persisted_primary["support_ids_by_label"]) == [
        str(label) for label in persisted_primary["eligible_labels"]
    ]
    assert all(type(row["label"]) is int for row in result["rows"]["primary"])
    assert all(type(row["label"]) is int for row in result["rows"]["alternate"])
    assert all(len(audit["primary_batch_ids"]) == 8 for audit in result["seed_audits"])
    assert result["exclusions"] == []
    assert result["integrity"]["deterministic_global_max"] == _valid_global_max_audit()
    assert result["integrity"]["zero_jacobian_classifier"] == {
        str(seed): _valid_zero_jacobian_audit() for seed in range(4)
    }
    assert result["execution_audit"]["diagnostic_sha256"] == result["manifest"]["source"][
        "files"
    ]["scripts/diagnose_pass200_rsta_stage_a.py"]


def test_scientific_integrity_graphs_are_released_before_next_seed_and_scoring_graphs_are_fresh(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    test_scientific_cli_executes_exact_four_seed_pipeline_and_writes_atomic_rows(
        tmp_path, monkeypatch
    )


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


@pytest.mark.parametrize("failing_seed", [1, 2, 3])
@pytest.mark.parametrize(
    "gate",
    [
        "zero_jacobian",
        "repeatability",
        "adjoint",
        "rotation",
        "normwise",
        "rebuild",
        "reversed_action_order",
        "parameter_sign",
        "output_sign",
    ],
)
def test_scientific_later_seed_integrity_failure_prevents_all_candidate_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failing_seed: int,
    gate: str,
) -> None:
    """Catches any candidate work before every later-seed integrity gate passes."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    manifest_path, manifest = _task2_manifest(tmp_path)
    bounds, cache_builder, model_type = _task2_inputs(manifest)
    assert hasattr(_MODULE, "_new_scientific_scoring_state")
    primary = _MODULE.select_primary_panel(
        bounds[0].train_example_ids, bounds[0].train_labels
    )
    expected_integrity_ids = tuple(primary["batches"][0])
    alternate_calls = 0
    cache_calls: list[tuple[str, ...]] = []
    scoring_state_calls = 0

    def forbidden_alternate(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal alternate_calls
        alternate_calls += 1
        raise AssertionError("alternate panel initialized before integrity prefix")

    def tracked_cache(bound: Any, ordered_ids: Any, **kwargs: Any) -> Any:
        cache_calls.append(tuple(ordered_ids))
        if len(cache_calls) > 1:
            raise AssertionError("full scoring cache initialized before integrity prefix")
        return cache_builder(bound, ordered_ids, **kwargs)

    def forbidden_scoring_state() -> Any:
        nonlocal scoring_state_calls
        scoring_state_calls += 1
        raise AssertionError("scoring state initialized before integrity prefix")

    monkeypatch.setattr(_MODULE, "select_alternate_panel", forbidden_alternate)
    monkeypatch.setattr(_MODULE, "_new_scientific_scoring_state", forbidden_scoring_state)
    candidate_calls = {name: 0 for name in (
        "score_rsta_batch", "decide_stage_a", "joint_bootstrap", "scientific_payload"
    )}

    def forbidden(name: str) -> Callable[..., Any]:
        def call(*_args: Any, **_kwargs: Any) -> Any:
            candidate_calls[name] += 1
            raise AssertionError(f"candidate {name} reached before {gate}-{failing_seed}")

        return call

    for name in candidate_calls:
        monkeypatch.setattr(_MODULE, name, forbidden(name))

    def model_loader(bound: Any) -> Any:
        model = model_type(bound.seed).train()
        model.audit_seed = bound.seed
        return model

    def zero_jacobian(model: Any, _images: Any) -> dict[str, Any]:
        if gate == "zero_jacobian" and model.audit_seed == failing_seed:
            raise ValueError(f"zero_jacobian-{failing_seed}")
        return _valid_zero_jacobian_audit()

    def integrity(model: Any, *_args: Any, seed: int, **_kwargs: Any) -> dict[str, Any]:
        assert model.audit_seed == seed
        if gate in {"repeatability", "adjoint", "rotation"} and seed == failing_seed:
            raise ValueError(f"{gate}-{failing_seed}")
        adjoint = _task2_adjoint_auditor(
            model,
            None,
            *_MODULE.registered_adjoint_directions(
                model,
                (180, 512),
                seed=seed,
                dtype=torch.float32,
                device=torch.device("cpu"),
            ),
            output_direction_seed=_MODULE.domain_seed(
                "rsta-stage-a-v1|adjoint-u|", str(seed)
            ),
            parameter_direction_seed=_MODULE.domain_seed(
                "rsta-stage-a-v1|adjoint-v|", str(seed)
            ),
            integrity_passed=not (seed == failing_seed and gate == "normwise"),
            control_failure=gate
            if seed == failing_seed
            and gate in {"rebuild", "reversed_action_order", "parameter_sign", "output_sign"}
            else None,
        )
        return {
            "fields": {"sentinel": torch.tensor(float(seed), requires_grad=True)},
            "prehead": torch.zeros(180, 2),
            "raw_head": torch.zeros(180, 512),
            "repeatability": {},
            "adjoint": adjoint,
            "adjoint_relative_error": 0.0,
            "rotation": {},
        }

    monkeypatch.setattr(_MODULE, "_registered_first_batch_integrity", integrity)
    output = tmp_path / "scientific-failure.json"
    expected_message = (
        f"{gate}-{failing_seed}"
        if gate in {"zero_jacobian", "repeatability", "adjoint", "rotation"}
        else "first-batch adjoint integrity failed"
    )
    with pytest.raises(ValueError, match=expected_message):
        _MODULE.run_scientific_diagnostic(
            manifest,
            manifest_path=manifest_path,
            receipt_path=tmp_path / "receipt.json",
            output_path=output,
            expected_dimension=512,
            receipt_validator=lambda *_args: _tiny_validated_receipt(),
            execution_source_validator=lambda _path: {"validated": True},
            bound_loader=lambda _entry, receipt_seed, **_kwargs: bounds[receipt_seed.seed],
            cache_builder=tracked_cache,
            model_loader=model_loader,
            fixture_runner=_task2_fixture_audit,
            deterministic_pool_auditor=_valid_global_max_audit,
            zero_jacobian_auditor=zero_jacobian,
        )

    assert candidate_calls == {name: 0 for name in candidate_calls}
    assert alternate_calls == 0
    assert cache_calls == [expected_integrity_ids]
    assert scoring_state_calls == 0
    assert not output.exists()
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


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
    assert result["integrity"]["adjoint"]["passed"] is True
    assert list(result["integrity"]["adjoint"]) == list(_MODULE._ADJOINT_AUDIT_FIELDS)
    _MODULE._validate_adjoint_integrity_audit(
        result["integrity"]["adjoint"], expected_output_shape=(180, 3)
    )
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
    helper_path = scripts / "rsta_normwise_adjoint.py"
    helper_path.write_text("BOUND = True\n", encoding="utf-8")
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
        "adjoint_integrity_amendment": {
            "path": _MODULE._ADJOINT_INTEGRITY_AMENDMENT_PATH,
            "sha256": _MODULE._ADJOINT_INTEGRITY_AMENDMENT_SHA256,
            "commit": _MODULE._ADJOINT_INTEGRITY_AMENDMENT_COMMIT,
        },
        "normwise_adjoint_calibration_protocol": {
            "path": _MODULE._NORMWISE_ADJOINT_CALIBRATION_PROTOCOL_PATH,
            "sha256": _MODULE._NORMWISE_ADJOINT_CALIBRATION_PROTOCOL_SHA256,
            "commit": _MODULE._NORMWISE_ADJOINT_CALIBRATION_PROTOCOL_COMMIT,
        },
        "normwise_adjoint_calibration_result": {
            "path": _MODULE._NORMWISE_ADJOINT_CALIBRATION_RESULT_PATH,
            "sha256": _MODULE._NORMWISE_ADJOINT_CALIBRATION_RESULT_SHA256,
            "commit": _MODULE._NORMWISE_ADJOINT_CALIBRATION_RESULT_COMMIT,
        },
            "normwise_adjoint_amendment": {
                "path": _MODULE._NORMWISE_ADJOINT_AMENDMENT_PATH,
                "sha256": _MODULE._NORMWISE_ADJOINT_AMENDMENT_SHA256,
                "commit": _MODULE._NORMWISE_ADJOINT_AMENDMENT_COMMIT,
            },
                "normwise_adjoint_sign_control_amendment": {
                    "path": _MODULE._NORMWISE_ADJOINT_SIGN_CONTROL_AMENDMENT_PATH,
                    "sha256": _MODULE._NORMWISE_ADJOINT_SIGN_CONTROL_AMENDMENT_SHA256,
                    "commit": _MODULE._NORMWISE_ADJOINT_SIGN_CONTROL_AMENDMENT_COMMIT,
                },
                "scientific_artifact_roundtrip_recovery_amendment": {
                    "path": _MODULE._SCIENTIFIC_ARTIFACT_ROUNDTRIP_RECOVERY_AMENDMENT_PATH,
                    "sha256": _MODULE._SCIENTIFIC_ARTIFACT_ROUNDTRIP_RECOVERY_AMENDMENT_SHA256,
                    "commit": _MODULE._SCIENTIFIC_ARTIFACT_ROUNDTRIP_RECOVERY_AMENDMENT_COMMIT,
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
                "scripts/rsta_normwise_adjoint.py": _MODULE._NORMWISE_ADJOINT_HELPER_SHA256,
            },
        },
    }
    calibration_result_path = (
        repository / references["normwise_adjoint_calibration_result"]["path"]
    ).resolve()

    def load_json(path: Path) -> dict[str, Any]:
        if path.resolve() == manifest_path.resolve():
            return manifest
        if path.resolve() == calibration_result_path:
            return {"all_passed": True}
        raise AssertionError(path)

    monkeypatch.setattr(_MODULE, "load_strict_json", load_json)
    normwise_helper = importlib.import_module("rsta_normwise_adjoint")
    monkeypatch.setattr(normwise_helper, "validate_calibration_result", lambda _value: None)
    monkeypatch.setattr(
        _MODULE,
        "_load_authenticated_normwise_adjoint_helper",
        lambda *_args: normwise_helper,
    )
    monkeypatch.setattr(_MODULE, "_validate_amended_manifest_schema", lambda value: value)
    monkeypatch.setattr(_MODULE, "__file__", str(diagnostic))
    hashed_paths: list[Path] = []

    def fake_sha(path: Path) -> str:
        hashed_paths.append(path.resolve())
        if path.resolve() == diagnostic.resolve():
            return "4" * 64
        if path.resolve() == helper_path.resolve():
            return _MODULE._NORMWISE_ADJOINT_HELPER_SHA256
        for reference in references.values():
            if path.resolve() == (repository / reference["path"]).resolve():
                return str(reference["sha256"])
        raise AssertionError(path)

    blobs: list[tuple[str, str]] = []

    def fake_blob(_repository: Path, revision: str, path_text: str) -> bytes:
        blobs.append((revision, path_text))
        if path_text == "scripts/rsta_normwise_adjoint.py":
            return b"blob-" + _MODULE._NORMWISE_ADJOINT_HELPER_SHA256.encode("ascii")
        digest_by_commit = {
            str(reference["commit"]): str(reference["sha256"])
            for reference in references.values()
            if "commit" in reference
        }
        digest = digest_by_commit.get(revision, "4" * 64)
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
