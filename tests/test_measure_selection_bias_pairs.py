from __future__ import annotations

from pathlib import Path


def test_fiedler_selection_bias_pair_is_declared() -> None:
    source = (
        Path(__file__).resolve().parents[1] / "scripts" / "measure_selection_bias.py"
    ).read_text(encoding="utf-8")
    assert '("pa_fiedler", "pa_ipc4")' in source
