"""Static contract tests for the guarded SigLIP RSTA Stage-A deployment."""

from __future__ import annotations

import os
from pathlib import Path

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "deploy_siglip_rsta_stage_a_v1.sh"


def test_rsta_deployment_is_executable_and_waits_for_complete_control() -> None:
    source = _SCRIPT.read_text()
    assert os.access(_SCRIPT, os.X_OK)
    assert "pgrep -f '[r]un_siglip_proxy_control.py'" in source
    assert "pgrep -f '[d]iagnose_siglip_rsta_stage_a.py'" in source
    assert "exit 75" in source
    assert "control.receipt.json" in source
    for seed in (17, 29, 43):
        assert f"seed-{seed:03d}.receipt.json" in source
        assert f"seed-{seed:03d}-epoch-060.pt" in source
    assert "run_siglip_rsta_stage_a.py" in source
    assert "--execute-controller" in source


def test_rsta_deployment_builds_only_the_optimization_image_namespace() -> None:
    source = _SCRIPT.read_text()
    assert "write_control_manifest_artifacts(" in source
    assert "bands=load_control_examples()" in source
    assert "--optimization-image-root" in source
    for forbidden in (
        "clean-validation.npy",
        "burned-diagnostic.npy",
        "official-test",
        "--test",
        "aws s3",
    ):
        assert forbidden not in source


def test_rsta_deployment_is_revision_scoped_and_preserves_one_terminal() -> None:
    source = _SCRIPT.read_text()
    assert "git diff --quiet HEAD --" in source
    assert "git diff --cached --quiet" in source
    assert "SOURCE_MANIFEST.sha256" in source
    assert "sha256sum --check --strict" in source
    assert "git bundle create" in source
    assert "git clone --quiet --no-checkout" in source
    assert 'git -C "$source_dir" rev-parse HEAD' in source
    assert "result.json" in source
    assert "terminal.json" in source
    assert 'test ! -e "$output"' in source
    assert "timeout --foreground --signal=TERM --kill-after=30s 7200s" in source
    assert 'kill -TERM -- "-$child"' in source
    assert 'test "$result_count" -eq 1' in source
    assert 'printf \'%s\\n\' "$controller_status" >"$staging/controller.exit"' in source
    assert 'controller_status=$(<"$scratch/controller.exit")' in source
    assert 'printf \'%s\\n\' "$control_source" >"$staging/control.source"' in source
    assert "034e66407c5de6e2ff1acf3d18455b10760d3509" in source
    assert 'test "$control_source" = "$expected_control_source"' in source
    assert 'value["authority"]["control_binding"]["source_commit"] == control_source' in source
    assert 'value["authority"]["optimization_manifest_sha256"]' in source
    assert "claim_eligible" in source
    assert "PASS_ONWARD" in source
    assert "UNRESOLVED" in source
    assert "INVALID" in source
