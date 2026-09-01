"""Static contract for the guarded intermediate-readout deployment."""

from __future__ import annotations

import os
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "deploy_siglip_intermediate_readout_v1.sh"
)


def test_deployment_is_revision_scoped_waits_for_control_and_uses_only_seed17() -> None:
    source = _SCRIPT.read_text()
    assert os.access(_SCRIPT, os.X_OK)
    assert "git diff --quiet HEAD --" in source
    assert "git diff --cached --quiet" in source
    assert "git bundle create" in source
    assert "sha256sum --check --strict SOURCE_MANIFEST.sha256" in source
    assert "pgrep -f '[r]un_siglip_proxy_control.py'" in source
    assert "control process is still active" in source
    assert "control.receipt.json" in source
    for seed in (17, 29, 43):
        assert f"seed-{seed:03d}.receipt.json" in source
        assert f"seed-{seed:03d}-epoch-060.pt" in source
    assert "project_stage_a_authority" in source
    assert "write_control_manifest_artifacts" in source
    assert "diagnose_siglip_intermediate_readout.py" in source
    assert '--checkpoint-seed17 "$control/seed-017/checkpoints/seed-017-epoch-060.pt"' in source
    assert "--checkpoint-seed29" not in source
    assert "--checkpoint-seed43" not in source


def test_deployment_has_offline_single_process_pressure_and_cleanup_fences() -> None:
    source = _SCRIPT.read_text()
    assert "HF_HUB_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source
    assert "timeout --foreground --signal=TERM --kill-after=30s 3600s" in source
    assert "pgrep -f '[d]iagnose_siglip_intermediate_readout.py'" in source
    assert "intermediate process is already active" in source
    assert 'kill -TERM -- "-$child"' in source
    assert 'test ! -e "$output"' in source
    assert 'test ! -e "$staging"' in source
    assert 'rm -rf -- "$images" "$staging/authority"' in source
    assert 'rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"' in source
    for forbidden in ("clean-validation", "burned-diagnostic", "official-test", "aws s3"):
        assert forbidden not in source
