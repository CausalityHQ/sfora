"""Static contract for guarded SigLIP spatial-tail deployment."""

from __future__ import annotations

import os
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "deploy_siglip_spatial_tail_recovery_v1.sh"
)


def test_deployment_stages_only_optimization_authority() -> None:
    source = _SCRIPT.read_text()
    assert os.access(_SCRIPT, os.X_OK)
    assert 'git diff --quiet HEAD -- "${source_files[@]}"' in source
    assert 'git diff --cached --quiet -- "${source_files[@]}"' in source
    assert "git bundle create" in source
    assert "sha256sum --check --strict SOURCE_MANIFEST.sha256" in source
    assert "write_control_manifest_artifacts" in source
    assert "png_compress_level=0" in source
    assert "project_stage_a_authority" in source
    assert "probe_siglip_spatial_tail_recovery.py" in source
    assert '--optimization-image-root "$optimization_images"' in source
    assert "--execute-spatial-tail" in source
    for forbidden in (
        "evaluation-manifest",
        "evaluation-image",
        "clean_validation",
        "official-test",
        "aws s3",
    ):
        assert forbidden not in source


def test_deployment_is_single_process_bounded_and_validates_result() -> None:
    source = _SCRIPT.read_text()
    assert "HF_HUB_OFFLINE=1" in source
    assert "HF_DATASETS_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "timeout --foreground --signal=TERM --kill-after=30s 5400s" in source
    assert "spatial tail process is already active" in source
    assert 'kill -TERM -- "-$child"' in source
    for reason in ("rss-cap", "psi-immediate", "psi-sustained", "swap-delta", "progress-gap"):
        assert f"stop_reason={reason}" in source
    assert "validate_spatial_tail_result_bytes" in source
    assert 'unlink "$staging/control-manifest.json"' in source
    assert 'rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"' in source

