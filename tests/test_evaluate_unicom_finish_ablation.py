from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_unicom_finish_ablation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_finish_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_contrast_clusters_by_identity_and_counts_discordance() -> None:
    control = [0.2, 0.4, 0.8, 0.6]
    candidate = [0.3, 0.4, 0.9, 0.5]
    labels = ["a", "a", "b", "c"]
    result = MODULE.paired_contrast(
        control,
        candidate,
        labels,
        control_top1=[False, True, True, True],
        candidate_top1=[True, True, False, True],
        bootstrap_samples=200,
    )
    assert math.isclose(result["delta_map_at_r"], 0.025, abs_tol=1e-15)
    assert result["win_tie_loss"] == {"win": 2, "tie": 1, "loss": 1}
    assert result["top1_discordant"] == {
        "candidate_only_correct": 1,
        "control_only_correct": 1,
    }
    assert len(result["identity_cluster_bootstrap_95"]) == 2


@pytest.mark.parametrize(
    ("ca", "cb", "r1a", "r1b", "expected"),
    (
        (0.003, 0.004, -0.001, 0.0, "GO"),
        (0.0029, 0.004, 0.0, 0.0, "CLOSE"),
        (0.004, 0.0029, 0.0, 0.0, "CLOSE"),
        (0.004, 0.004, -0.0011, 0.0, "CLOSE"),
    ),
)
def test_causal_gate_requires_both_matched_contrasts(ca, cb, r1a, r1b, expected) -> None:
    assert MODULE.classify_causal_panel(
        c_minus_a=ca,
        c_minus_b=cb,
        c_minus_a_recall1=r1a,
        c_minus_b_recall1=r1b,
        c_minus_a_recall10=0.0,
        c_minus_b_recall10=0.0,
        c_minus_parent_recall1=0.0,
        c_minus_parent_recall10=0.0,
    ) == expected
