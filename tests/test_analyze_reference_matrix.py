"""Regression tests for paired reference-matrix arm registration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "analyze_reference_matrix",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_reference_matrix.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


def test_bncorrect_ema_arms_are_registered_against_plain_proxy_anchor() -> None:
    """The In-Shop confirmation pairs each EMA arm with plain Proxy Anchor."""
    for arm in ("pa_ema_avg_bnfix", "pa_dual_ema_bnfix"):
        assert _module.ARMS[arm] == ("proxy_anchor", arm)
        assert _module.PAIRED_CONTROL[arm] == "proxy_anchor"
        assert _module.BASE_OF[arm] == "proxy_anchor"


def test_capacity_arms_keep_their_weakened_paired_controls() -> None:
    """Adding explicit EMA controls must not break the existing narrow-arm override."""
    assert _module.BASE_OF["narrow128_distill"] == "narrow128"
    assert _module.BASE_OF["narrow64_distill"] == "narrow64"


def test_fiedler_arm_pairs_with_ipc4_sampler_control() -> None:
    assert _module.ARMS["pa_ipc4"] == ("proxy_anchor", "pa_ipc4")
    assert _module.ARMS["pa_fiedler"] == ("proxy_anchor", "pa_fiedler")
    assert _module.BASE_OF["pa_fiedler"] == "pa_ipc4"


def test_rsatk_is_a_standalone_reference_arm() -> None:
    assert _module.ARMS["rsatk"] == ("recall_at_k_surrogate", "auto")
    assert "rsatk" not in _module.BASE_OF


def test_exact_sign_test_uses_counts_not_delta_magnitudes() -> None:
    five_positive_one_negative = [100.0, 1.0, 1.0, 1.0, 1.0, -99.0]

    assert _module.exact_sign_test_p(five_positive_one_negative) == pytest.approx(0.21875)
    assert _module.exact_sign_test_p([1.0] * 6) == pytest.approx(0.03125)
    assert _module.exact_sign_test_p([1.0, 1.0, 1.0, -1.0, -1.0, -1.0]) == 1.0
    assert _module.exact_sign_test_p([0.0, 0.0]) == 1.0


def test_paired_permutation_test_remains_separate_from_sign_test() -> None:
    deltas = [100.0, 1.0, 1.0, 1.0, 1.0, -99.0]

    assert _module.exact_paired_permutation_p(deltas) != _module.exact_sign_test_p(deltas)
