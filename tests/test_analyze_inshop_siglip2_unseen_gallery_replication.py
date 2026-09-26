"""Seed replication keeps the frozen paired quality gate."""

import importlib.util
import sys
from pathlib import Path

import numpy as np

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "analyze_inshop_siglip2_unseen_gallery",
    SCRIPTS / "analyze_inshop_siglip2_unseen_gallery.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_paired_gate_requires_positive_r1_and_nonregressing_map() -> None:
    labels = np.asarray(["a", "a", "b", "b"])
    control_r1 = np.zeros(4)
    freeze_r1 = np.ones(4)
    control_ap = np.full(4, 0.5)
    assert MODULE.paired_gate(control_r1, freeze_r1, control_ap, control_ap, labels)["gate_pass"]
    assert not MODULE.paired_gate(control_r1, freeze_r1, control_ap, np.full(4, 0.49), labels)[
        "gate_pass"
    ]
    assert not MODULE.paired_gate(control_r1, control_r1, control_ap, control_ap, labels)[
        "gate_pass"
    ]
