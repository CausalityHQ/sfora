"""Static contract for the guarded gallery-alignment deployment."""

from __future__ import annotations

import os
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "deploy_siglip_gallery_compatibility_alignment_v1.sh"
)


def test_deployment_is_optimization_only_and_binds_the_sealed_tail() -> None:
    source = _SCRIPT.read_text()
    assert os.access(_SCRIPT, os.X_OK)
    assert "git bundle create" in source
    assert "git diff --quiet HEAD -- src/sfora scripts" in source
    assert "probe_siglip_gallery_compatibility_alignment.py" in source
    assert "cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9" in source
    assert "67973944" in source
    assert "--execute-gallery-alignment" in source
    assert "load_control_examples" not in source
    assert "materialize_registered_optimization_images" in source
    assert "landlock_exec.c" in source
    assert '"$staging/landlock-exec" --ro /usr' in source
    assert "datasets--tanganke--stanford_cars" not in source
    assert "models--google--siglip-so400m-patch14-384" in source
    assert "--traverse /home/riomus/.cache/huggingface/hub" in source
    assert "HF_HUB_OFFLINE=1" in source
    for forbidden in ("evaluation-manifest", "official-test", "aws s3"):
        assert forbidden not in source


def test_deployment_has_single_process_pressure_and_evidence_guards() -> None:
    source = _SCRIPT.read_text()
    assert "timeout --signal=TERM --kill-after=30s 3600s" in source
    assert "gallery alignment process is already active" in source
    assert 'kill -TERM -- "-$child"' in source
    assert 'pgrep -g "$group"' in source
    assert 'kill -KILL -- "-$group"' in source
    for reason in ("rss-cap", "psi-immediate", "psi-sustained", "swap-delta", "progress-gap"):
        assert f"stop_reason={reason}" in source
    assert "validate_alignment_result_bytes" in source
    assert "load_file" in source
    assert "orthogonality_error" in source
    assert 'payload["development_student"]' in source
    assert 'payload["development_aligned"]' in source
    assert 'score_delta == value["self_geometry_max_score_delta"]' in source
    assert "gpu-memory-cap" in source
    assert "nvidia-smi --query-compute-apps=used_memory" in source
    assert "((gpu_mib <= 49152))" in source
    assert "source_owned=0" in source
    assert "source_owned=1" in source
    assert "source_checkout_complete=0" in source
    assert "source_checkout_complete=1" in source
    assert 'test ! -L "$source_dir"' in source
    assert 'test "${source_dir##*/}" = "$revision"' in source
    assert 'git -C "$source_dir" rev-parse HEAD' in source
    assert 'rm -rf -- "$source_dir"' in source
    assert 'value["external_evaluation_access"] is False' in source
    assert 'private_tmp="$staging/private-tmp"' in source
    assert 'export HOME="$private_tmp" TMPDIR="$private_tmp"' in source
    assert 'rm -rf -- "$staging/private-tmp"' in source
    assert "--rw /tmp" not in source
    assert "--rw /dev " not in source
    assert "--rw /dev/shm" not in source
    assert '--traverse "$control"' in source
    assert '--ro "$checkpoint"' in source
    assert '--ro "$control"' not in source
    for device in (
        "/dev/null",
        "/dev/zero",
        "/dev/random",
        "/dev/urandom",
        "/dev/nvidiactl",
        "/dev/nvidia0",
        "/dev/nvidia-uvm",
        "/dev/nvidia-uvm-tools",
    ):
        assert f"--rw {device}" in source
    assert 'rsync -a -- "$remote_host:$remote_output/result.json" "$local_result"' in source
    assert (
        'rsync -a -- "$remote_host:$remote_output/alignment.safetensors" "$local_artifact"'
        in source
    )
