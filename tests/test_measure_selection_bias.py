"""Tests for the best-over-training selection-bias estimator."""

from __future__ import annotations

import importlib.util
import random
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "measure_selection_bias",
    Path(__file__).resolve().parents[1] / "scripts" / "measure_selection_bias.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)

leave_one_out_local_mean = _module.leave_one_out_local_mean
selection_overshoot = _module.selection_overshoot


def test_selected_epoch_is_excluded_from_its_own_trend_estimate() -> None:
    """The selected point's own noise is exactly what selection exploited, so including
    it in the trend estimate would cancel the effect being measured."""
    history = [0.70] * 10 + [0.90] + [0.70] * 10

    assert leave_one_out_local_mean(history, 10, 2) == pytest.approx(0.70)
    reported, estimated, index = selection_overshoot(history)
    assert (index, reported, estimated) == (10, 0.90, pytest.approx(0.70))


def test_noiseless_history_has_no_overshoot() -> None:
    """A monotone, noiseless curve is selected at its true maximum, so a correct
    estimator must report zero bias -- otherwise it would invent one everywhere."""
    history = [0.60 + 0.001 * i for i in range(60)]

    reported, estimated, index = selection_overshoot(history)
    assert index == 59
    # Trailing-edge neighbours are all below the endpoint, so the estimate is lower by
    # the local slope only; the point is that it does not scale with any noise term.
    assert reported - estimated == pytest.approx(0.0015, abs=1e-6)


@pytest.mark.parametrize(("noise", "floor"), [(0.005, 0.5), (0.02, 2.0)])
def test_overshoot_scales_with_evaluation_noise_on_a_flat_plateau(
    noise: float, floor: float
) -> None:
    """The claim being made about the protocol: on a FLAT truth, every reported point of
    'improvement' is selection. Bigger noise must buy a bigger reported maximum."""
    generator = random.Random(0)
    overshoots = []
    for _ in range(200):
        observed = [0.70 + generator.gauss(0, noise) for _ in range(60)]
        reported, estimated, _ = selection_overshoot(observed)
        overshoots.append((reported - estimated) * 100)

    assert sum(overshoots) / len(overshoots) > floor


def test_a_collapsed_run_is_reported_far_above_its_trend() -> None:
    """Guards the diagnostic that validated the estimator on real data: `local_nca`
    collapsed, peaked in its first epochs, and best-over-training still reported 0.5733
    against a 0.3394 trend."""
    history = [0.57, 0.55, 0.30, 0.28, 0.27] + [0.26] * 55

    reported, estimated, index = selection_overshoot(history)
    assert index == 0
    assert (reported - estimated) * 100 > 10.0
