"""Static contract for the guarded compatibility-capacity deployment."""

from __future__ import annotations

import os
from pathlib import Path

_SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts" / "deploy_siglip_compatibility_capacity_v1.sh"
)


def test_capacity_deployment_is_burned_data_only_and_binds_the_sealed_tail() -> None:
    source = _SCRIPT.read_text()
    assert os.access(_SCRIPT, os.X_OK)
    assert "git bundle create" in source
    assert "git diff --quiet HEAD -- src/sfora scripts" in source
    assert "probe_siglip_compatibility_capacity.py" in source
    assert "cf12e5eced83f23327b15919bcfe7f0cd15e01b184c7945bac14c3f80d445fd9" in source
    assert "67973944" in source
    assert "--execute-capacity-diagnostic" in source
    assert "materialize_registered_optimization_images" in source
    assert '"$staging/landlock-exec" --ro /usr' in source
    assert "HF_HUB_OFFLINE=1" in source
    assert "models--google--siglip-so400m-patch14-384" in source
    for forbidden in (
        "evaluation-manifest",
        "official-test",
        "class-names",
        "aws s3",
    ):
        assert forbidden not in source


def test_capacity_deployment_has_single_process_resource_and_cleanup_guards() -> None:
    source = _SCRIPT.read_text()
    assert "timeout --signal=TERM --kill-after=30s 7200s" in source
    assert "compatibility capacity process is already active" in source
    assert 'kill -TERM -- "-$child"' in source
    assert 'pgrep -g "$group"' in source
    assert 'kill -KILL -- "-$group"' in source
    for reason in (
        "rss-cap",
        "gpu-memory-cap",
        "psi-immediate",
        "psi-sustained",
        "swap-delta",
        "progress-gap",
    ):
        assert f"stop_reason={reason}" in source
    assert "((rss <= 51539607552))" in source
    assert "((gpu_mib <= 49152))" in source
    assert "source_owned=0" in source and "source_owned=1" in source
    assert "source_checkout_complete=0" in source
    assert "source_checkout_complete=1" in source
    assert 'test ! -L "$source_dir"' in source
    assert 'git -C "$source_dir" rev-parse HEAD' in source
    assert 'rm -rf -- "$source_dir"' in source
    assert 'private_tmp="$staging/private-tmp"' in source
    assert 'export HOME="$private_tmp" TMPDIR="$private_tmp"' in source
    assert "--rw /tmp" not in source
    assert "--rw /dev " not in source
    assert "--rw /dev/shm" not in source


def test_capacity_deployment_revalidates_and_preserves_both_outputs() -> None:
    source = _SCRIPT.read_text()
    postrun_validation = source.split("artifact_sha=", 1)[1]
    validation_block = postrun_validation.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    compile(validation_block, "<capacity-postrun-validation>", "exec")
    assert "import hashlib, pathlib, sys" in postrun_validation
    assert "validate_compatibility_capacity_result_bytes" in source
    assert 'value["claim_eligible"] is False' in source
    assert 'value["external_evaluation_access"] is False' in source
    assert 'value["descriptor_artifact_sha256"] == sys.argv[2]' in source
    assert 'value["spatial_artifact_sha256"] == sys.argv[3]' in source
    assert 'value["control_binding_sha256"] == sys.argv[4]' in source
    assert 'value["optimization_manifest_sha256"] == sys.argv[5]' in source
    assert "assert metadata[name] == value[name]" in source
    assert 'b"sfora-compatibility-capacity-id-v1\\0"' in source
    assert "assert artifact_identities == result_identities" in source
    for name in (
        "checkpoint_sha256",
        "control_binding_sha256",
        "optimization_manifest_sha256",
        "spatial_artifact_sha256",
        "image_manifest_sha256",
        "preprocessing",
    ):
        assert f'    "{name}",' in source
    assert 'set(payload) == {"student", "teacher", "id_sha256", "labels"}' in source
    assert 'record["dimensions"] == payload["student"].shape[1]' in source
    assert 'rsync -a -- "$remote_host:$remote_output/result.json" "$local_result"' in source
    assert (
        'rsync -a -- "$remote_host:$remote_output/descriptors.safetensors" '
        '"$local_artifact"' in source
    )
