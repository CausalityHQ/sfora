from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from sfora.native_twin_probe import (
    NativeTwinAuthority,
    NativeTwinResult,
    _predictions,
    canonical_native_twin_result_bytes,
    native_descriptor_sha256,
    score_native_twin_probe,
    validate_canonical_native_twin_result_bytes,
    validate_native_twin_result,
)


def _planes() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return non-degenerate global/control/native planes with paired rescues."""

    rng = np.random.default_rng(20260901)
    global_plane = rng.standard_normal((40, 4)).astype(np.float32)
    control = np.empty((40, 9, 4), dtype=np.float32)
    native = np.empty_like(control)
    for index in range(20):
        angle = 0.13 + 0.11 * index
        base = np.asarray([np.cos(angle), np.sin(angle), 0.03, -0.02])
        peer = 20 + index
        for view in range(9):
            perturb = np.asarray([0.0, 0.0, 0.002 * view, -0.001 * view])
            if index < 8:
                control[index, view] = base + perturb
                control[peer, view] = base - perturb
            else:
                control[index, view] = np.asarray(
                    [1.0, 0.08 * index, 0.02 * view, 0.0]
                )
                control[peer, view] = np.asarray(
                    [-1.0, 0.08 * index, 0.02 * view, 0.0]
                )
            native[index, view] = np.asarray(
                [1.0, 0.04 * index, 0.02 * view, 0.01]
            )
            native[peer, view] = np.asarray(
                [-1.0, 0.04 * index, 0.02 * view, -0.01]
            )
    global_plane /= np.linalg.norm(global_plane, axis=1, keepdims=True)
    for plane in (control, native):
        plane /= np.linalg.norm(plane, axis=2, keepdims=True)
    return global_plane, control, native


def _authority(
    global_plane: np.ndarray,
    control: np.ndarray,
    native: np.ndarray,
) -> NativeTwinAuthority:
    return NativeTwinAuthority(
        source_identity="native-probe-source-v2",
        checkpoint_sha256="1" * 64,
        model_revision="2" * 40,
        probe_revision="3" * 40,
        probe_tree_digest="4" * 64,
        example_ids=tuple(f"example-{index:03d}" for index in range(40)),
        image_sha256=tuple(f"{1000 - index:064x}" for index in range(40)),
        labels=tuple([82] * 20 + [83] * 20),
        crop_long_edges=tuple((300,) * 9 for _index in range(40)),
        global_descriptor_sha256=native_descriptor_sha256(global_plane),
        control_descriptor_sha256=native_descriptor_sha256(control),
        native_descriptor_sha256=native_descriptor_sha256(native),
    )


def test_native_twin_probe_isolates_pixels_with_matched_control_and_exact_mcnemar() -> None:
    """Removing the matched plane or exact McNemar gate must fail."""

    global_plane, control, native = _planes()
    authority = _authority(global_plane, control, native)

    result = score_native_twin_probe(authority, global_plane, control, native)

    assert isinstance(result, NativeTwinResult)
    assert result.control_errors == 16
    assert result.native_errors == 0
    assert result.native_error_reduction == 1.0
    assert result.rescues == 16
    assert result.harms == 0
    assert result.mcnemar_p_value == 1.0 / 65_536.0
    assert result.swap_permutation_draws == 10_000
    assert result.swap_permutation_p_value <= 0.05
    assert len(result.swap_permutation_seed_sha256) == 64
    assert result.control_noop_crop_count == 0
    assert result.native_balanced_accuracy == 1.0
    assert result.classification == "native-pixel-cue-pass"
    assert result.passed is True
    assert result.control_candidate_ids != result.native_candidate_ids

    validate_native_twin_result(result, authority, global_plane, control, native)
    raw = canonical_native_twin_result_bytes(result)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    payload = json.loads(raw)
    assert payload["claim_eligible"] is False
    assert payload["schema"] == "sfora-native-twin-probe-v2"
    assert validate_canonical_native_twin_result_bytes(raw, expected=result) == result
    with pytest.raises(ValueError, match="derivation"):
        validate_native_twin_result(
            replace(result, mcnemar_p_value=0.5),
            authority,
            global_plane,
            control,
            native,
        )


def test_native_twin_probe_binds_all_descriptor_planes_and_uses_digest_ties() -> None:
    """Changing descriptor bytes or label-correlated IDs must not change custody/ties."""

    global_plane, control, native = _planes()
    authority = _authority(global_plane, control, native)
    original = score_native_twin_probe(authority, global_plane, control, native)

    changed_ids = score_native_twin_probe(
        replace(authority, example_ids=tuple(reversed(authority.example_ids))),
        global_plane,
        control,
        native,
    )
    assert original.control_candidate_labels == changed_ids.control_candidate_labels
    assert original.native_candidate_labels == changed_ids.native_candidate_labels

    tied = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.0, 1.0], [-1.0, 0.0]],
        dtype=np.float32,
    )
    selected = _predictions(tied, ("f" * 64, "e" * 64, "1" * 64, "d" * 64))
    assert int(selected[0]) == 2

    changed = native.copy()
    changed[0, 0] = changed[0, 1]
    with pytest.raises(ValueError, match="digest"):
        score_native_twin_probe(authority, global_plane, control, changed)
    with pytest.raises(ValueError, match="descriptor"):
        score_native_twin_probe(
            replace(authority, native_descriptor_sha256="0" * 64),
            global_plane,
            control,
            native,
        )


def test_native_twin_probe_rejects_view_score_gain_without_native_pixels() -> None:
    """Identical nine-view control/native planes must fail the causal gate."""

    global_plane, control, _native = _planes()
    native = control.copy()
    authority = _authority(global_plane, control, native)

    result = score_native_twin_probe(authority, global_plane, control, native)

    assert result.rescues == 0
    assert result.harms == 0
    assert result.mcnemar_p_value == 1.0
    assert result.swap_permutation_p_value > 0.05
    assert result.classification == "native-pixel-cue-fail"
    assert result.passed is False


@pytest.mark.parametrize(
    ("global_mutation", "control_mutation", "native_mutation", "message"),
    (
        (lambda value: value[:-1], lambda value: value, lambda value: value, "shape"),
        (lambda value: value, lambda value: value[:, :8], lambda value: value, "nine"),
        (lambda value: value, lambda value: value, lambda value: value[:, :8], "nine"),
        (
            lambda value: value,
            lambda value: value,
            lambda value: value.astype(np.float64),
            "fp32",
        ),
    ),
)
def test_native_twin_probe_rejects_plane_shape_and_type_drift(
    global_mutation: object,
    control_mutation: object,
    native_mutation: object,
    message: str,
) -> None:
    global_plane, control, native = _planes()
    authority = _authority(global_plane, control, native)
    with pytest.raises((TypeError, ValueError), match=message):
        score_native_twin_probe(
            authority,
            global_mutation(global_plane),  # type: ignore[operator]
            control_mutation(control),  # type: ignore[operator]
            native_mutation(native),  # type: ignore[operator]
        )
