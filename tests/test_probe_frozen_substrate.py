from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT = Path(__file__).parents[1] / "scripts" / "probe_frozen_substrate.py"
_SPEC = importlib.util.spec_from_file_location("probe_frozen_substrate", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_gate_is_exact_single_substrate_threshold() -> None:
    assert _MODULE.substrate_passed(correct=1265)
    assert not _MODULE.substrate_passed(correct=1264)


def test_probe_pins_dinov2_and_never_reads_test_split() -> None:
    source = _SCRIPT.read_text()
    assert '"facebook/dinov2-large"' in source
    assert '"47b73eefe95e8d44ec3623f8890bd894b6ea2d6c"' in source
    assert 'split="train"' in source
    assert 'split="test"' not in source
    assert "torch.autocast" not in source
    assert "torch.backends.cudnn.allow_tf32 = False" in source
    assert 'len(holdout) != _EXPECTED_QUERIES' in source
    assert '"last_hidden_state_cls"' in source
    assert '"google/siglip2-so400m-patch14-384"' in source
    assert '"e8e487298228002f3d8a82e0cd5c8ea9c567f57f"' in source
    assert '"vision_pooler_output"' in source
