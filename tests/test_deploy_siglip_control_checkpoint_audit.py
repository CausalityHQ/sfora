"""Static contract for the guarded trained-checkpoint audit deployment."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "deploy_siglip_control_checkpoint_audit_v1.sh"
)


def test_deployment_binds_terminal_control_and_all_six_audit_artifacts() -> None:
    source = _SCRIPT.read_text()
    assert os.access(_SCRIPT, os.X_OK)
    assert "git diff --quiet HEAD --" in source
    assert "git diff --cached --quiet" in source
    assert "git bundle create" in source
    assert "sha256sum --check --strict SOURCE_MANIFEST.sha256" in source
    assert "control.receipt.json" in source
    for seed in (17, 29, 43):
        assert f"seed-{seed:03d}.receipt.json" in source
    assert "seed-017-epoch-060.pt" in source
    assert "seed-017-epoch-060.checkpoint.json" in source
    assert "--selected-seed 17" in source
    assert "audit_siglip_control_checkpoint.py" in source
    for flag in (
        "--initial-output",
        "--output",
        "--raw-twin-output",
        "--projected-twin-output",
        "--raw-twin-inference-output",
        "--projected-twin-inference-output",
    ):
        assert flag in source
    assert "--execute-checkpoint-audit" in source
    for basename in (
        "initial.json",
        "trained.json",
        "raw-twin.json",
        "projected-twin.json",
        "raw-twin-inference.json",
        "projected-twin-inference.json",
    ):
        assert source.count(basename) >= 2


def test_deployment_refuses_overlap_and_has_pressure_cleanup_fences() -> None:
    source = _SCRIPT.read_text()
    assert "[r]un_siglip_proxy_control.py|[r]un_native_twin_probe.py" in source
    assert "[a]udit_siglip_control_checkpoint.py" in source
    assert "control or native probe is still active" in source
    assert "checkpoint audit is already active" in source
    assert "HF_HUB_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "timeout --foreground --signal=TERM --kill-after=30s 14400s" in source
    assert 'kill -TERM -- "-$child"' in source
    assert 'ps -o rss= -g "$child" 2>/dev/null' in source
    assert "stop_reason=psi" in source
    assert "stop_reason=swap-delta" in source
    assert 'test ! -e "$output"' in source
    assert 'test ! -e "$staging"' in source
    assert "stale source clone:" in source
    assert "missing seed-017 terminal checkpoint" in source
    assert 'rm -rf -- "$source_dir"' in source
    assert 'rm -rf -- "$local_output"' in source
    assert 'mv "$staging" "$output"' in source
    assert 'rsync -a -- "$remote_host:$remote_output/" "$local_output/"' in source


def test_deployment_validates_canonical_outputs_before_publication() -> None:
    source = _SCRIPT.read_text()
    subprocess.run(["bash", "-n", str(_SCRIPT)], check=True)
    assert "canonical output differs" in source
    assert 'value["claim_eligible"] is False' in source
    assert "hashlib.sha256(raw).hexdigest()" in source
    assert "exactly six audit artifacts are required" in source
    assert "validate_siglip_initial_control_audit_bytes" in source
    assert "validate_siglip_checkpoint_audit_bytes" in source
    assert "validate_twin_reachability_artifact_bytes" in source
    assert "validate_twin_reachability_inference_artifact_bytes" in source
    assert 'value["authority"]["plane"]' in source
    assert 'value["official_test_access"] is False' in source
    assert "launcher receipt differs" in source
    assert source.index('mv "$staging" "$output"') > source.index(
        "canonical output differs"
    )
    for forbidden in ("--official-test", "--clean-validation", "aws s3"):
        assert forbidden not in source
