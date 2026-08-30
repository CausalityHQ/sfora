from pathlib import Path


def test_deploy_launcher_uses_revision_scoped_manifest_and_cuda_gate() -> None:
    text = Path("scripts/deploy_siglip_token_set_f0_v49.sh").read_text()

    assert "git ls-files -z" in text
    assert "git diff --quiet" in text
    assert "git diff --cached --quiet" in text
    assert "origin/devbox/emafactorial" in text
    assert "sha256sum" in text
    assert "sort -z" in text
    assert "LC_ALL=C sort -z" in text
    assert "rsync" in text
    assert "--files-from" in text
    assert "--from0" in text
    assert "--delete" not in text
    assert "git pull" not in text
    assert "git checkout" not in text
    assert "SOURCE_REVISION" in text
    assert "SOURCE_MANIFEST" in text
    assert "tests/test_set_maxsim_kernel.py" in text
    assert "-k triton" in text
    assert "CUDA parity gate cannot skip" in text
    assert "PYTHONDONTWRITEBYTECODE=1" in text
    assert "UV_PROJECT_ENVIRONMENT" in text
    assert text.rindex("tests/test_set_maxsim_kernel.py") < text.rindex(
        "bash scripts/run_siglip_token_set_f0_v49.sh"
    )
    assert "reports/generated/cars-token-set-f0-2026-08-30.json" in text
