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


def test_siglip2_successor_is_revision_scoped_and_offline() -> None:
    root = Path(__file__).parents[1]
    deploy = (root / "scripts/deploy_frozen_substrate_f0_siglip2_v1.sh").read_text()
    run = (root / "scripts/run_frozen_substrate_f0_siglip2_v1.sh").read_text()
    assert "origin/devbox/emafactorial" in deploy
    assert "git diff --quiet" in deploy
    assert "SOURCE_MANIFEST.sha256" in deploy
    assert "--offline --locked" in deploy + run
    assert "google/siglip2-so400m-patch14-384" in run
    assert "google/siglip-so400m-patch14-384" in run
    assert "CELL" in deploy + run
    assert "split=\"test\"" not in deploy + run


def test_final_siglip_cell_can_publish_the_sealed_pass209_error_manifest() -> None:
    root = Path(__file__).parents[1]
    deploy = (root / "scripts/deploy_frozen_substrate_f0_siglip2_v1.sh").read_text()
    run = (root / "scripts/run_frozen_substrate_f0_siglip2_v1.sh").read_text()

    assert "LOCAL_ERROR_MANIFEST" in deploy
    assert "REMOTE_ERROR_MANIFEST" in deploy
    assert '[[ $cell == siglip-so400m ]]' in deploy
    assert 'EXPECTED_CORRECT=1242' in deploy
    assert '"${remote_error_manifest:-none}"' in deploy
    assert '[[ $6 != none ]]' in deploy
    assert '"$remote_error_manifest" <<\'REMOTE\'' not in deploy
    assert '[[ $remote_error_manifest =~ ^reports/generated/' in deploy
    assert 'realpath -m -- "$local_error_manifest"' in deploy
    assert '!= "$(realpath -m -- "$local_output")"' in deploy
    assert "validate_pass209_m2_artifacts.py" in deploy + run
    assert "tests/test_validate_pass209_m2_artifacts.py" in deploy
    assert 'rsync -a -- "$host:$dir/$remote_error_manifest" "$local_error_manifest"' in deploy
    assert "ERROR_MANIFEST" in run and "EXPECTED_CORRECT" in run
    assert '--error-manifest "$error_manifest"' in run
    assert '--expected-correct "$expected_correct"' in run
