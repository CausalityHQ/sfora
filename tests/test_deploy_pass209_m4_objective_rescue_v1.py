"""Contract tests for deployment of the Pass209 M4 DGX campaign."""

from __future__ import annotations

import subprocess
from pathlib import Path

_DEPLOY = Path(__file__).parents[1] / "scripts" / "deploy_pass209_m4_objective_rescue_v1.sh"


def test_deployer_uses_content_addressed_create_new_remote_source() -> None:
    text = _DEPLOY.read_text()

    subprocess.run(["bash", "-n", str(_DEPLOY)], check=True)
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "git status --porcelain" in text
    assert "git ls-files --error-unmatch" in text
    for required in (
        "scripts/run_pass209_m4_cell.py",
        "scripts/analyze_pass209_m4.py",
        "scripts/run_pass209_m4_objective_rescue_v1.sh",
        "scripts/deploy_pass209_m4_objective_rescue_v1.sh",
        "src/sfora/pass209_m4.py",
    ):
        assert required in text
    assert "git archive" in text
    assert "SOURCE_REVISION" in text
    assert "SOURCE_MANIFEST" in text
    assert "sha256sum --check --strict" in text
    assert "run_pass209_m4_objective_rescue_v1.sh" in text
    assert "run_siglip_proxy_control.py (train|aggregate)" in text
    assert "refusing to duplicate a live M4 campaign" in text
    assert "nohup" in text and "setsid" in text
    assert "rm -rf" not in text
    assert "git push" not in text
