from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/audit_unicom_frozen_embeddings.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("audit_unicom_frozen_embeddings", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_accepts_exact_arguments_and_never_imports_torch() -> None:
    module = _load_script()
    assert module.parse_args(["--bundle", "in.npz", "--output", "out.json"]).bundle == Path(
        "in.npz"
    )
    assert "torch" not in module.__dict__
    with pytest.raises(SystemExit):
        module.parse_args(["--bundle", "in.npz"])


def test_cli_runs_e1_then_e2_publishes_once_and_no_clobbers(
    tmp_path: Path, monkeypatch
) -> None:
    module = _load_script()
    bundle = object()
    calls: list[str] = []
    output = tmp_path / "report.json"
    report = {
        "schema_version": 1,
        "input": {"bundle_path": "in.npz", "bundle_sha256": "a" * 64, "metadata": {}},
        "constants": {
            "selected_coordinates": 512,
            "random_masks": 32,
            "geometry_bootstrap_samples": 10_000,
            "panel_classes": 64,
            "examples_per_class": 4,
            "shard_trials": 32,
            "permutations_per_trial": 16,
        },
        "geometry": {},
        "sharding": {},
        "runtime": {"python": "3.12", "numpy": "2.5"},
        "warnings": [],
    }
    monkeypatch.setattr(module, "load_embedding_bundle", lambda path: bundle)
    monkeypatch.setattr(module, "run_geometry", lambda value: calls.append("E1") or object())
    monkeypatch.setattr(module, "run_sharding", lambda value: calls.append("E2") or object())
    monkeypatch.setattr(module, "build_audit_report", lambda *args: report)
    monkeypatch.setattr(module, "validate_audit_report", lambda value: None)

    def publish(path: Path, payload: dict[str, object]) -> None:
        path.open("x", encoding="utf-8").write(json.dumps(payload) + "\n")

    monkeypatch.setattr(module, "publish_json_no_clobber", publish)

    assert module.run(["--bundle", "in.npz", "--output", str(output)]) == 0
    assert calls == ["E1", "E2"]
    assert json.loads(output.read_text()) == report
    original = output.read_bytes()
    assert module.run(["--bundle", "in.npz", "--output", str(output)]) == 2
    assert output.read_bytes() == original
