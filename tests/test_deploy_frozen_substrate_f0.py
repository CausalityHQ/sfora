import importlib.util
from pathlib import Path


def _probe_module() -> object:
    path = Path(__file__).parents[1] / "scripts" / "probe_frozen_substrate.py"
    spec = importlib.util.spec_from_file_location("probe_frozen_substrate_authority", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_every_registered_cell_can_publish_a_common_protocol_error_manifest() -> None:
    root = Path(__file__).parents[1]
    deploy = (root / "scripts/deploy_frozen_substrate_f0_siglip2_v1.sh").read_text()
    run = (root / "scripts/run_frozen_substrate_f0_siglip2_v1.sh").read_text()

    assert "LOCAL_ERROR_MANIFEST" in deploy
    assert "REMOTE_ERROR_MANIFEST" in deploy
    authorities = _probe_module()._ERROR_EVIDENCE_AUTHORITIES
    assert authorities == {
        "dinov2-large": (32, 32, 1196),
        "siglip2-so400m": (8, 32, 1227),
        "siglip-so400m": (8, 32, 1242),
    }
    for cell, (batch_size, query_block, correct) in authorities.items():
        assert f"{cell}) expected_correct={correct}" in deploy
        assert f"{cell}:{correct}" in run
        assert f"{cell}) batch_size={batch_size}" in run or (
            batch_size == 8 and "siglip2-so400m|siglip-so400m) batch_size=8" in run
        )
        assert query_block == 32
    assert (
        "[[ $cell == dinov2-large || $cell == siglip2-so400m || "
        "$cell == siglip-so400m ]]"
    ) in deploy
    assert "dinov2-large)" in run
    assert "batch_size=32" in run
    assert "siglip2-so400m|siglip-so400m) batch_size=8" in run
    assert '"${remote_error_manifest:-none}"' in deploy
    assert '"${expected_correct:-none}"' in deploy
    assert '[[ $6 != none ]]' in deploy
    assert 'EXPECTED_CORRECT="$7"' in deploy
    assert '"$remote_error_manifest" <<\'REMOTE\'' not in deploy
    assert '[[ $remote_error_manifest =~ ^reports/generated/' in deploy
    assert 'realpath -m -- "$local_error_manifest"' in deploy
    assert '!= "$(realpath -m -- "$local_output")"' in deploy
    assert "validate_pass209_m2_artifacts.py" in deploy + run
    assert "tests/test_validate_pass209_m2_artifacts.py" in deploy
    assert 'rsync -a -- "$host:$dir/$remote_error_manifest" "$local_error_manifest"' in deploy
    assert '[[ $local_manifest_sha256 == "$expected_manifest_sha256" ]]' in deploy
    assert "ERROR_MANIFEST" in run and "EXPECTED_CORRECT" in run
    assert '--error-manifest "$error_manifest"' in run
    assert '--expected-correct "$expected_correct"' in run
    assert 'if [[ $cell == siglip-so400m ]]; then' in deploy
    assert 'if [[ -n $error_manifest && $cell == siglip-so400m ]]; then' in run
