"""Static contract for the revision-scoped SigLIP head-screen deployment."""

from pathlib import Path


def test_head_screen_deploy_is_revision_scoped_and_waits_for_control() -> None:
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "deploy_siglip_head_screen_v1.sh"
    ).read_text()

    assert "git diff --quiet HEAD --" in script
    assert "git diff --cached --quiet" in script
    assert "git ls-files --others --exclude-standard -- src scripts sitecustomize.py" in script
    assert "SOURCE_REVISION" in script
    assert "SOURCE_MANIFEST.sha256" in script
    assert "sha256sum --check --strict" in script
    assert "[r]un_siglip_proxy_control.py" in script
    assert "control process is still active" in script
    assert "PYTHONDONTWRITEBYTECODE=1" in script
    assert '"$python" -B' in script
    assert "prepare_siglip_head_features.py" in script
    assert "diagnose_siglip_head_screen.py" in script
    assert "--device cuda" in script
    assert "--execute-feature-cache" in script
    assert "--execute-head-screen" in script
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in script
    assert "timeout --signal=TERM --kill-after=30s 7200s" in script
    assert 'rsync -a -- "$remote_host:$remote_output/result.json" "$local_output"' in script
