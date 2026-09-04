from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "build_unicom_finish_ablation_config.py"
)
SPEC = importlib.util.spec_from_file_location("build_finish_config", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_config_cli_requires_authority_and_explicit_execution(tmp_path: Path) -> None:
    required = [
        "--source-commit",
        "a" * 40,
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--partition-sha256",
        "b" * 64,
        "--parent-checkpoint-sha256",
        "c" * 64,
        "--parent-receipt-sha256",
        "d" * 64,
        "--official-checkpoint-sha256",
        "e" * 64,
        "--unicom-revision",
        "f" * 40,
        "--environment-sha256",
        "1" * 64,
        "--output",
        str(tmp_path / "config.json"),
    ]
    with pytest.raises(SystemExit):
        MODULE.parse_args(required)
    assert MODULE.parse_args([*required, "--execute-config"]).execute_config is True


def test_config_bytes_are_canonical_and_claim_ineligible() -> None:
    value = {
        "schema": "unicom-finish-ablation-config-v1",
        "claim_eligible": False,
    }
    payload = MODULE.canonical_config_bytes(value)
    assert payload == (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
