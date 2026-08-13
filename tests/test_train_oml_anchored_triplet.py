from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts/train_oml_anchored_triplet.py"
_SPEC = importlib.util.spec_from_file_location("train_oml_anchored_triplet", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
registered_seed0_screen_decision = _MODULE.registered_seed0_screen_decision


def test_registered_seed0_screen_decision_binds_baseline_and_role() -> None:
    initial = {"recall_at_1": 0.961093585699264}
    final = {"recall_at_1": 0.9605}
    paired = {"map_at_r_delta": 0.0041, "map_at_r_delta_ci95_lower": 0.0001}

    assert (
        registered_seed0_screen_decision(
            seed=0, evaluation_role="screen", initial=initial, final=final, paired=paired
        )
        is True
    )
    assert (
        registered_seed0_screen_decision(
            seed=1, evaluation_role="screen", initial=initial, final=final, paired=paired
        )
        is None
    )
    assert (
        registered_seed0_screen_decision(
            seed=0, evaluation_role="holdout", initial=initial, final=final, paired=paired
        )
        is None
    )


def test_registered_seed0_screen_decision_rejects_baseline_drift() -> None:
    try:
        registered_seed0_screen_decision(
            seed=0,
            evaluation_role="screen",
            initial={"recall_at_1": 0.95},
            final={"recall_at_1": 0.96},
            paired={"map_at_r_delta": 0.01, "map_at_r_delta_ci95_lower": 0.005},
        )
    except ValueError as error:
        assert str(error) == "measured screen baseline differs"
    else:
        raise AssertionError("baseline drift was accepted")
