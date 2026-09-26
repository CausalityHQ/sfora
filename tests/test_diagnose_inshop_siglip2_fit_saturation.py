"""Block drift keeps lower and upper encoder changes separate."""

import importlib.util
import sys
from pathlib import Path

import torch

SCRIPTS = Path(__file__).parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location(
    "diagnose_inshop_siglip2_fit_saturation", SCRIPTS / "diagnose_inshop_siglip2_fit_saturation.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_block_drift_separates_encoder_halves() -> None:
    original = {f"encoder.layers.{i}.weight": torch.ones(2) for i in range(24)}
    trained = {
        name: value * (2 if i < 12 else 1) for i, (name, value) in enumerate(original.items())
    }
    drift = MODULE.block_drift(original, trained)
    assert drift["lower_0_11_mean"] == 1
    assert drift["upper_12_23_mean"] == 0
