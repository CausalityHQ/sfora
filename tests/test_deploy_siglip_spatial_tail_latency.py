"""Static contract for the guarded spatial-tail latency deployment."""

from __future__ import annotations

import os
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "deploy_siglip_spatial_tail_latency_v1.sh"
)


def test_deployment_uses_only_the_sealed_artifact_and_optimization_inputs() -> None:
    source = _SCRIPT.read_text()
    assert os.access(_SCRIPT, os.X_OK)
    assert 'git diff --quiet HEAD -- "${source_files[@]}"' in source
    assert 'git diff --cached --quiet -- "${source_files[@]}"' in source
    assert "git bundle create" in source
    assert "sha256sum --check --strict SOURCE_MANIFEST.sha256" in source
    assert "probe_siglip_spatial_tail_latency.py" in source
    assert "cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9" in source
    assert "67973944" in source
    assert "--execute-spatial-tail-latency" in source
    for forbidden in (
        "evaluation-manifest",
        "evaluation-image",
        "clean_validation",
        "official-test",
        "aws s3",
    ):
        assert forbidden not in source


def test_deployment_is_single_process_bounded_and_validates_the_receipt() -> None:
    source = _SCRIPT.read_text()
    assert "HF_HUB_OFFLINE=1" in source
    assert "HF_DATASETS_OFFLINE=1" in source
    assert "TRANSFORMERS_OFFLINE=1" in source
    assert "timeout --foreground --signal=TERM --kill-after=30s 3600s" in source
    assert "spatial tail latency process is already active" in source
    assert 'kill -TERM -- "-$child"' in source
    assert 'cpu=$(ps -o time= -g "$child" 2>/dev/null || true)' in source
    for reason in ("rss-cap", "psi-immediate", "psi-sustained", "swap-delta", "progress-gap"):
        assert f"stop_reason={reason}" in source
    assert 'value["quality_measured"] is False' in source
    assert 'value["external_evaluation_access"] is False' in source
    assert 'value["spatial_artifact_sha256"] == sys.argv[2]' in source
    assert 'rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"' in source
