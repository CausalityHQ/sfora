"""Contracts for monitored SigLIP calibration preparation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_siglip_coverage_calibration.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("prepare_siglip_coverage_calibration", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_SUBJECT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_SUBJECT)
parse_args = _SUBJECT.parse_args


def test_prepare_requires_explicit_guard_and_exact_local_paths(tmp_path: Path) -> None:
    argv = [
        "--control",
        str(tmp_path / "control"),
        "--authority",
        str(tmp_path / "authority"),
        "--optimization-image-root",
        str(tmp_path / "optimization"),
        "--support-image-root",
        str(tmp_path / "support"),
        "--heldout-image-root",
        str(tmp_path / "heldout"),
    ]
    with pytest.raises(SystemExit):
        parse_args(argv)
    parsed = parse_args([*argv, "--execute-preparation"])
    assert parsed.control == tmp_path / "control"
    assert parsed.authority == tmp_path / "authority"
    with pytest.raises(SystemExit):
        parse_args([*argv, "--execute-preparation", "--network-uri", "s3://forbidden"])
