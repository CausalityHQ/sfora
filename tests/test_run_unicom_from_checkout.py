from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_unicom_from_checkout.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_run_unicom_from_checkout", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load checkout launcher")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_runs_trainer_from_its_own_checkout_under_isolated_python() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(SCRIPT),
            str(ROOT / "scripts" / "train_unicom_inshop.py"),
            "--help",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--evaluation-features" in completed.stdout


def test_launcher_places_checkout_src_first_and_rejects_unregistered_target(
    tmp_path: Path,
) -> None:
    module = _load_module()
    src, target = module.authenticate_paths(ROOT / "scripts" / "profile_unicom_training_step.py")
    assert src == (ROOT / "src").resolve()
    assert target == (ROOT / "scripts" / "profile_unicom_training_step.py").resolve()

    outside = tmp_path / "outside.py"
    outside.write_text("raise SystemExit(0)\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registered experiment script"):
        module.authenticate_paths(outside)

    linked_target = tmp_path / "linked-trainer.py"
    linked_target.symlink_to(ROOT / "scripts" / "train_unicom_inshop.py")
    with pytest.raises(ValueError, match="registered experiment script"):
        module.authenticate_paths(linked_target)


def test_launcher_checkout_source_beats_a_decoy_package(tmp_path: Path) -> None:
    decoy = tmp_path / "decoy"
    (decoy / "sfora").mkdir(parents=True)
    (decoy / "sfora" / "__init__.py").write_text(
        'raise RuntimeError("EXTERNAL_VENV_DECOY")\n', encoding="utf-8"
    )
    code = """
import importlib.util,json,sys
from pathlib import Path
spec=importlib.util.spec_from_file_location('_launcher',sys.argv[1])
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
sys.path[:]=[sys.argv[2],*sys.path]
src,target=module.prepare_checkout_imports(Path(sys.argv[3]))
import sfora.unicom_training as loaded
print(json.dumps({'src':str(src),'origin':str(Path(loaded.__file__).resolve())}))
"""
    completed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-c",
            code,
            str(SCRIPT),
            str(decoy),
            str(ROOT / "scripts" / "train_unicom_inshop.py"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    observed = json.loads(completed.stdout)
    assert Path(observed["src"]) == (ROOT / "src").resolve()
    assert Path(observed["origin"]) == (ROOT / "src" / "sfora" / "unicom_training.py").resolve()


def test_launcher_preserves_target_exception_and_traceback(tmp_path: Path) -> None:
    module = _load_module()
    target = ROOT / "scripts" / "profile_unicom_training_step.py"

    def explode(_path: str, *, run_name: str) -> None:
        assert run_name == "__main__"
        raise RuntimeError("target exploded")

    original = module.runpy.run_path
    module.runpy.run_path = explode
    try:
        with pytest.raises(RuntimeError, match="target exploded"):
            module.main([str(target)])
    finally:
        module.runpy.run_path = original
