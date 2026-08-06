"""Behavioral lock for the corrected In-Shop seed-2 reference runner."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


def test_seed2_runner_executes_locked_recipe_and_validates_final_artifacts(
    tmp_path: Path,
) -> None:
    project = tmp_path / "remote-project"
    (project / "src").mkdir(parents=True)
    dataset = tmp_path / "official-standard"
    dataset.mkdir()
    invocation = tmp_path / "invocation.json"

    # The runner's inline validator imports torch.  This tiny real module reads
    # the JSON checkpoint emitted by the fake trainer, so the validation code
    # itself executes instead of being mocked away.
    (project / "src" / "torch.py").write_text(
        "import json\n"
        "def load(path, map_location=None):\n"
        "    with open(path, encoding='utf-8') as handle:\n"
        "        return json.load(handle)\n",
        encoding="utf-8",
    )

    fake_sfora = tmp_path / "fake_sfora.py"
    fake_sfora.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "def value(flag): return args[args.index(flag) + 1]\n"
        "pathlib.Path(os.environ['INVOCATION']).write_text(json.dumps(args))\n"
        "config = {'seed': int(value('--seed')), 'recipe_id': value('--recipe')}\n"
        "report = pathlib.Path(value('--output'))\n"
        "checkpoint = pathlib.Path(value('--save-model-path'))\n"
        "report.write_text(json.dumps({'config': config}))\n"
        "checkpoint.write_text(json.dumps({\n"
        "  'artifact_selection': 'final_training_state',\n"
        "  'training_step': 8580,\n"
        "  'training_config': config,\n"
        "}))\n",
        encoding="utf-8",
    )
    fake_sfora.chmod(0o755)

    runner = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_inshop_corrected_reference_seed2.sh"
    )
    env = os.environ | {
        "PROJECT_DIR": str(project),
        "PYTHON_BIN": sys.executable,
        "SFORA_BIN": str(fake_sfora),
        "DATASET_ROOT": str(dataset),
        "INVOCATION": str(invocation),
    }
    completed = subprocess.run(
        ["bash", str(runner)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    args = json.loads(invocation.read_text(encoding="utf-8"))
    assert args == [
        "image-end-to-end",
        "--dataset-name",
        "inshop",
        "--dataset-root",
        str(dataset),
        "--objectives",
        "proxy_anchor",
        "--recipe",
        "proxy_anchor.inshop.official-51db570",
        "--num-workers",
        "8",
        "--seed",
        "2",
        "--save-model-path",
        "reports/checkpoints/inshop_corrected_pa_seed2.pt",
        "--output",
        "reports/generated/inshop_corrected_pa_seed2.json",
    ]
    assert (project / "reports/generated/inshop_corrected_pa_seed2.json").is_file()
    assert (project / "reports/checkpoints/inshop_corrected_pa_seed2.pt").is_file()
