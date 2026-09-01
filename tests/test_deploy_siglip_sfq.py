"""Static contract for the revision-scoped post-control SFQ deployment."""

from pathlib import Path


def test_sfq_deploy_is_revision_scoped_and_removes_evaluation_features() -> None:
    """The launcher must not overlap control or expose evaluation feature rows to SFQ."""

    path = Path(__file__).resolve().parents[1] / "scripts" / "deploy_siglip_sfq_v1.sh"
    script = path.read_text()

    assert path.stat().st_mode & 0o111
    assert "git diff --quiet HEAD --" in script
    assert "git diff --cached --quiet" in script
    assert "SOURCE_REVISION" in script
    assert "SOURCE_MANIFEST.sha256" in script
    assert "sha256sum --check --strict" in script
    assert "[r]un_siglip_proxy_control.py" in script
    assert "control process is still active" in script
    assert "prepare_siglip_head_features.py" in script
    assert "diagnose_siglip_sfq.py" in script
    assert 'unlink "$features/clean-validation.npy"' in script
    assert 'unlink "$features/burned-diagnostic.npy"' in script
    assert '--feature-source-commit "$revision"' in script
    assert "--execute-sfq-folds" in script
    assert "--output-dimensions" not in script
    assert "PYTHONDONTWRITEBYTECODE=1" in script
    assert "HF_HUB_OFFLINE=1" in script
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script
    assert "timeout --signal=TERM --kill-after=30s 7200s" in script
    assert 'rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"' in script
