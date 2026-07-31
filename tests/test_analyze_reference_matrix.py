"""Regression tests for paired reference-matrix arm registration."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

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
