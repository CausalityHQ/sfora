from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


_SPEC = importlib.util.spec_from_file_location(
    "measure_frequency_band_acquisition",
    Path(__file__).resolve().parents[1] / "scripts" / "measure_frequency_band_acquisition.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_replace_amplitude_band_uses_donor_only_inside_band() -> None:
    rng = np.random.default_rng(11)
    source = torch.tensor(rng.uniform(20, 220, size=(3, 32, 32)), dtype=torch.float32)
    donor = torch.tensor(rng.uniform(20, 220, size=(3, 32, 32)), dtype=torch.float32)
    result = _MODULE.replace_amplitude_band(source, donor, (1.0 / 3.0, 2.0 / 3.0))
    assert result.shape == source.shape
    assert torch.isfinite(result).all()
    assert not torch.allclose(result, source)


def test_donor_indices_exclude_label_and_series() -> None:
    records = [
        _MODULE.Record(f"x{i}.jpg", i // 2, f"s{i % 3}", Path(f"/x/{i}")) for i in range(12)
    ]
    donors = _MODULE.donor_indices(records)
    for index, donor in enumerate(donors):
        assert records[index].label != records[int(donor)].label
        assert records[index].series != records[int(donor)].series
