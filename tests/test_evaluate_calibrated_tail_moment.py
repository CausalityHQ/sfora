from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sfora.calibrated_tail_moment_io import evaluate_bundle as evaluate_reduced_bundle
from sfora.unicom_audit_io import EmbeddingBundle

SCRIPT = Path(__file__).parents[1] / "scripts/evaluate_calibrated_tail_moment.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("evaluate_calibrated_tail_moment", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _evaluated_bundle(tmp_path: Path) -> EmbeddingBundle:
    values = np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.8, 0.0, 0.0, 0.6],
            [0.0, 0.8, 0.6, 0.0],
            [0.0, 0.8, 0.0, 0.6],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(["a", "b", "c", "d"])
    return EmbeddingBundle(
        path=(tmp_path / "evaluated.npz").resolve(),
        sha256="b" * 64,
        metadata={"embedding_dimension": 4},
        train_embeddings=np.ascontiguousarray(np.repeat(values, 13, axis=0)),
        train_labels=np.repeat(labels, 13),
        query_embeddings=values,
        query_labels=labels,
        gallery_embeddings=values,
        gallery_labels=labels,
    )


def _evaluated_report(tmp_path: Path) -> dict[str, object]:
    return evaluate_reduced_bundle(
        _evaluated_bundle(tmp_path),
        widths=(2,),
        neighbors=1,
        null_seeds=range(206, 210),
    )


def test_cli_accepts_only_bundle_and_output() -> None:
    module = _load_script()
    parsed = module.parse_args(["--bundle", "input.npz", "--output", "out.json"])
    assert parsed.bundle == Path("input.npz")
    assert parsed.output == Path("out.json")
    with pytest.raises(SystemExit):
        module.parse_args(["--bundle", "input.npz"])


def test_cli_rejects_existing_output_before_loading_bundle(tmp_path, monkeypatch) -> None:
    module = _load_script()
    output = tmp_path / "result.json"
    output.write_bytes(b"sentinel")
    monkeypatch.setattr(
        module,
        "load_embedding_bundle",
        lambda path: pytest.fail("bundle must not load after output preflight"),
    )

    assert module.run(["--bundle", "input.npz", "--output", str(output)]) == 2
    assert output.read_bytes() == b"sentinel"


def test_cli_runs_registered_grid_once_and_never_clobbers(tmp_path, monkeypatch) -> None:
    module = _load_script()
    output = tmp_path / "ctm.json"
    calls: list[str] = []
    monkeypatch.setattr(module, "REGISTERED_WIDTHS", (2,))
    monkeypatch.setattr(module, "REGISTERED_NEIGHBORS", 1)
    monkeypatch.setattr(module, "REGISTERED_NULL_SEEDS", range(206, 210))
    monkeypatch.setattr(module, "load_embedding_bundle", lambda path: _evaluated_bundle(tmp_path))
    monkeypatch.setattr(
        module,
        "evaluate_bundle",
        lambda bundle, **kwargs: calls.append("evaluate") or _evaluated_report(tmp_path),
    )

    assert module.run(["--bundle", "input.npz", "--output", str(output)]) == 0
    first = output.read_bytes()
    assert module.run(["--bundle", "input.npz", "--output", str(output)]) == 2
    assert calls == ["evaluate"]
    assert output.read_bytes() == first


def test_cli_rejects_owned_temp_before_loading_bundle(tmp_path, monkeypatch) -> None:
    module = _load_script()
    output = tmp_path / "ctm.json"
    temporary = tmp_path / f".{output.name}.123.deadbeef.tmp"
    temporary.write_bytes(b"foreign")
    monkeypatch.setattr(
        module,
        "load_embedding_bundle",
        lambda path: pytest.fail("bundle must not load after temp preflight"),
    )

    assert module.run(["--bundle", "input.npz", "--output", str(output)]) == 2
    assert temporary.read_bytes() == b"foreign"


def test_cli_import_is_candidate_free_and_does_not_load_training_stacks() -> None:
    root = SCRIPT.parents[1]
    code = f"""
import importlib.util
import sys
sys.path.insert(0, {str(root / 'src')!r})
spec = importlib.util.spec_from_file_location('ctm_cli', {str(SCRIPT)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
for forbidden in ('torch', 'faiss', 'train', 'trainer'):
    assert forbidden not in sys.modules
"""

    completed = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
