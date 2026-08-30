from pathlib import Path


def test_deploy_is_revision_scoped_and_runs_only_train_band() -> None:
    root = Path(__file__).parents[1]
    deploy = (root / "scripts/deploy_frozen_substrate_f0_v1.sh").read_text()
    run = (root / "scripts/run_frozen_substrate_f0_v1.sh").read_text()
    assert "origin/devbox/emafactorial" in deploy
    assert "git diff --quiet" in deploy
    assert "SOURCE_MANIFEST.sha256" in deploy
    assert "--offline --locked" in deploy
    assert "facebook/dinov2-large" in run
    assert "PREFLIGHT_ONLY" in run
    assert "split=\"test\"" not in deploy + run
