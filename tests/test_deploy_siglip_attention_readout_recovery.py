"""Static contract for guarded attention-readout recovery deployment."""

from __future__ import annotations

import os
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "deploy_siglip_attention_readout_recovery_v1.sh"
)


def test_deployment_separates_optimization_and_evaluation_authority() -> None:
    source = _SCRIPT.read_text()
    assert os.access(_SCRIPT, os.X_OK)
    assert 'git diff --quiet HEAD -- "${source_files[@]}"' in source
    assert 'git diff --cached --quiet -- "${source_files[@]}"' in source
    assert 'git ls-files --error-unmatch "$path"' in source
    assert "git bundle create" in source
    assert "sha256sum --check --strict SOURCE_MANIFEST.sha256" in source
    assert "write_control_manifest_artifacts" in source
    assert "bands.clean_validation" in source
    assert "evaluation_image_root / _image_basename(example.example_id)" in source
    assert '"schema": "sfora-attention-readout-evaluation-v1"' in source
    assert "project_stage_a_authority" in source
    assert "probe_siglip_attention_readout_recovery.py" in source
    assert '--optimization-image-root "$optimization_images"' in source
    assert '--evaluation-image-root "$evaluation_images"' in source
    assert "--execute-attention-readout" in source


def test_deployment_is_single_process_offline_bounded_and_verifies_result() -> None:
    source = _SCRIPT.read_text()
    assert "HF_HUB_OFFLINE=1" in source
    assert "HF_DATASETS_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "timeout --foreground --signal=TERM --kill-after=30s 5400s" in source
    assert "attention readout process is already active" in source
    assert 'kill -TERM -- "-$child"' in source
    assert "stop_reason=rss-cap" in source
    assert "stop_reason=psi-immediate" in source
    assert "stop_reason=psi-sustained" in source
    assert "stop_reason=swap-delta" in source
    assert "stop_reason=progress-gap" in source
    assert '"cuda_memory_cap_enforced_in_process": True' in source
    assert "validate_attention_readout_result_bytes" in source
    assert 'unlink "$staging/control-manifest.json"' in source
    assert 'unlink "$staging/evaluation-manifest.json"' not in source
    assert 'rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"' in source
    for forbidden in ("official-test", "aws s3", "--model", "--student"):
        assert forbidden not in source
