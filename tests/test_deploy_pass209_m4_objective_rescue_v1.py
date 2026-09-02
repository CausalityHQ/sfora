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
    assert 'LC_ALL=C sort -k2,2 -o "$manifest" "$manifest"' in text
    assert "sha256sum --check --strict" in text
    assert "run_pass209_m4_objective_rescue_v1.sh" in text
    assert "[r]un_siglip_proxy_control.py" in text
    assert "refusing to duplicate a live M4 campaign" in text
    assert "nohup" in text and "setsid" in text
    assert 'case "$source_dir" in' in text
    assert 'rm -rf -- "$source_dir"' in text
    assert "git push" not in text


def test_deployer_preflights_every_dependency_and_gpu_stage_before_upload() -> None:
    text = _DEPLOY.read_text()

    scp_index = text.index('scp -o BatchMode=yes "$archive"')
    for variable in (
        "DGX_ERROR_MANIFEST",
        "DGX_DINOV2_PREREQUISITE",
        "DGX_SIGLIP2_PREREQUISITE",
        "DGX_SELECTING_PREREQUISITE",
        "DGX_M3_SEED_017",
        "DGX_M3_SEED_029",
        "DGX_M3_SEED_043",
        "DGX_M3_AGGREGATE",
    ):
        assert text.index(f'${{{variable}:?') < scp_index
    assert "for prerequisite in \"$@\"" in text
    assert "missing M4 prerequisite:" in text
    for process in (
        "[r]un_siglip_proxy_control.py",
        "[r]un_native_twin_probe.py",
        "[a]udit_siglip_control_checkpoint.py",
        "[p]robe_frozen_substrate.py",
        "[r]un_pass209_m4_(cell|objective_rescue)",
    ):
        assert process in text
        assert text.index(process) < scp_index
    assert "active GPU campaign has not released the DGX" in text
    assert "missing uv environment:" in text
    assert "unusable M4 output parent:" in text
    assert "M4 launch did not survive" in text
    assert 'cat "$output.launch.log" >&2' in text
    assert 'kill -0 "$pid"' in text
    assert text.count("-o BatchMode=yes") >= 3
    assert "printf 'root=%q\\n'" in text
    assert "printf 'prerequisites=(" in text
    assert 'case "$source_dir" in' in text
    assert '"$root"/*)' in text
    assert 'rm -rf -- "$source_dir"' in text
    assert 'test ! -e "$archive" || unlink "$archive"' in text
    assert "launch_committed=false" in text
    assert "cleanup_failed_launch()" in text
    assert 'test ! -e "$output.launch.log" || unlink "$output.launch.log"' in text
