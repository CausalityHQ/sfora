from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_oml_interpolation.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("evaluate_oml_interpolation", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_interpolate_state_dict_blends_floats_and_preserves_exact_buffers() -> None:
    base = {"weight": torch.tensor([1.0, 3.0]), "count": torch.tensor(2)}
    trained = {"weight": torch.tensor([5.0, 7.0]), "count": torch.tensor(2)}

    result = _MODULE.interpolate_state_dict(base, trained, alpha=0.25)

    torch.testing.assert_close(result["weight"], torch.tensor([2.0, 4.0]))
    assert result["count"].item() == 2


def test_interpolate_state_dict_rejects_nonfloat_buffer_drift() -> None:
    try:
        _MODULE.interpolate_state_dict(
            {"count": torch.tensor(2)}, {"count": torch.tensor(3)}, alpha=0.5
        )
    except ValueError as error:
        assert "non-floating state differs" in str(error)
    else:
        raise AssertionError("non-floating drift was accepted")


def test_registered_interpolation_decision_uses_paired_map_and_recall_floor() -> None:
    initial = {"recall_at_1": 0.96}
    paired = {"map_at_r_delta": 0.0041, "map_at_r_delta_ci95_lower": 0.0002}

    assert _MODULE.registered_interpolation_decision(
        initial=initial,
        final={"recall_at_1": 0.9593},
        paired=paired,
    )
    assert not _MODULE.registered_interpolation_decision(
        initial=initial,
        final={"recall_at_1": 0.9592},
        paired=paired,
    )
    assert not _MODULE.registered_interpolation_decision(
        initial=initial,
        final={"recall_at_1": 0.96},
        paired={"map_at_r_delta": 0.0041, "map_at_r_delta_ci95_lower": -0.0001},
    )


def test_bootstrap_seed_is_stable_per_alpha_and_checkpoint() -> None:
    digest = "ab" * 32
    assert _MODULE.bootstrap_seed(alpha=0.75, checkpoint_sha256=digest) == _MODULE.bootstrap_seed(
        alpha=0.75, checkpoint_sha256=digest
    )
    assert _MODULE.bootstrap_seed(
        alpha=0.75, checkpoint_sha256=digest
    ) != _MODULE.bootstrap_seed(alpha=0.5, checkpoint_sha256=digest)


def test_registered_baseline_check_only_applies_to_the_fixed_screen() -> None:
    assert _MODULE.is_registered_screen(
        evaluation_fraction=0.2, evaluation_seed=17, evaluation_role="screen"
    )
    assert not _MODULE.is_registered_screen(
        evaluation_fraction=1.0, evaluation_seed=17, evaluation_role="screen"
    )
