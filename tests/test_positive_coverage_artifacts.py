from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch

_SCRIPT = Path(__file__).parents[1] / "scripts" / "positive_coverage_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("positive_coverage_artifacts", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
artifacts = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = artifacts
_SPEC.loader.exec_module(artifacts)


def _state(offset: float) -> dict[str, torch.Tensor]:
    return {"weight": (torch.eye(3, dtype=torch.float32) + offset).contiguous()}


def _receipt(states: dict[str, dict[str, torch.Tensor]]) -> dict[str, object]:
    arms = {}
    for name, state in states.items():
        arm = {"parameter_sha256": artifacts.linear_weight_sha256(state["weight"])}
        if "base_head_weight" in state:
            arm["base_head_sha256"] = artifacts.affine_parameters_sha256(
                state["base_head_weight"], state["base_head_bias"]
            )
        arms[name] = arm
    return {
        "arms": arms,
        "claim_eligible": False,
        "schema": "fixture-positive-coverage-v1",
    }


def test_positive_coverage_artifacts_write_checkpoints_before_complete_receipt(
    tmp_path: Path,
) -> None:
    states = {"pooled": _state(0.0), "coverage": _state(0.1)}
    paths = artifacts.PositiveCoverageArtifactPaths(
        pooled_checkpoint=tmp_path / "pooled.pt",
        coverage_checkpoint=tmp_path / "coverage.pt",
        complete_receipt=tmp_path / "complete.json",
    )

    completed = artifacts.write_positive_coverage_artifacts(
        states=states,
        receipt=_receipt(states),
        paths=paths,
    )

    wire = paths.complete_receipt.read_bytes()
    assert wire.endswith(b"\n") and not wire.endswith(b"\n\n")
    assert wire == artifacts.canonical_positive_coverage_receipt_bytes(completed)
    parsed = json.loads(wire)
    for name, path in (
        ("pooled", paths.pooled_checkpoint),
        ("coverage", paths.coverage_checkpoint),
    ):
        checkpoint_wire = path.read_bytes()
        assert parsed["arms"][name]["checkpoint"] == {
            "bytes": len(checkpoint_wire),
            "sha256": hashlib.sha256(checkpoint_wire).hexdigest(),
        }
        restored = torch.load(path, map_location="cpu", weights_only=True)
        torch.testing.assert_close(restored["weight"], states[name]["weight"])
    assert completed == parsed


def test_positive_coverage_artifacts_preserve_self_contained_base_head(
    tmp_path: Path,
) -> None:
    head_weight = torch.arange(12, dtype=torch.float32).reshape(3, 4).contiguous()
    head_bias = torch.arange(3, dtype=torch.float32).contiguous()
    states = {
        name: {
            **_state(offset),
            "base_head_weight": head_weight,
            "base_head_bias": head_bias,
        }
        for name, offset in (("pooled", 0.0), ("coverage", 0.1))
    }
    paths = artifacts.PositiveCoverageArtifactPaths(
        pooled_checkpoint=tmp_path / "pooled.pt",
        coverage_checkpoint=tmp_path / "coverage.pt",
        complete_receipt=tmp_path / "complete.json",
    )

    artifacts.write_positive_coverage_artifacts(
        states=states, receipt=_receipt(states), paths=paths
    )

    restored = torch.load(paths.coverage_checkpoint, map_location="cpu", weights_only=True)
    assert tuple(restored) == ("weight", "base_head_weight", "base_head_bias")
    torch.testing.assert_close(restored["base_head_weight"], head_weight)
    torch.testing.assert_close(restored["base_head_bias"], head_bias)

    invalid_receipt = _receipt(states)
    invalid_receipt["arms"]["coverage"]["base_head_sha256"] = "0" * 64  # type: ignore[index]
    invalid_paths = artifacts.PositiveCoverageArtifactPaths(
        pooled_checkpoint=tmp_path / "invalid-pooled.pt",
        coverage_checkpoint=tmp_path / "invalid-coverage.pt",
        complete_receipt=tmp_path / "invalid-complete.json",
    )
    with pytest.raises(ValueError, match="positive-coverage artifact authority"):
        artifacts.write_positive_coverage_artifacts(
            states=states, receipt=invalid_receipt, paths=invalid_paths
        )
    assert not invalid_paths.pooled_checkpoint.exists()


def test_positive_coverage_artifacts_refuse_overwrite(tmp_path: Path) -> None:
    states = {"pooled": _state(0.0), "coverage": _state(0.1)}
    paths = artifacts.PositiveCoverageArtifactPaths(
        pooled_checkpoint=tmp_path / "pooled.pt",
        coverage_checkpoint=tmp_path / "coverage.pt",
        complete_receipt=tmp_path / "complete.json",
    )
    paths.pooled_checkpoint.write_bytes(b"owned")

    with pytest.raises(ValueError, match="artifact path authority"):
        artifacts.write_positive_coverage_artifacts(
            states=states,
            receipt=_receipt(states),
            paths=paths,
        )
    assert paths.pooled_checkpoint.read_bytes() == b"owned"
    assert not paths.coverage_checkpoint.exists()
    assert not paths.complete_receipt.exists()


def test_positive_coverage_artifacts_do_not_clobber_a_racing_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = {"pooled": _state(0.0), "coverage": _state(0.1)}
    paths = artifacts.PositiveCoverageArtifactPaths(
        pooled_checkpoint=tmp_path / "pooled.pt",
        coverage_checkpoint=tmp_path / "coverage.pt",
        complete_receipt=tmp_path / "complete.json",
    )
    original = artifacts._sync_save

    def racing_save(state: dict[str, torch.Tensor], temporary: Path) -> None:
        original(state, temporary)
        if temporary == paths.pooled_checkpoint.with_name("pooled.pt.partial"):
            paths.pooled_checkpoint.write_bytes(b"owned-by-racer")

    monkeypatch.setattr(artifacts, "_sync_save", racing_save)
    with pytest.raises(FileExistsError):
        artifacts.write_positive_coverage_artifacts(
            states=states,
            receipt=_receipt(states),
            paths=paths,
        )
    assert paths.pooled_checkpoint.read_bytes() == b"owned-by-racer"
    assert not paths.coverage_checkpoint.exists()
    assert not paths.complete_receipt.exists()


def test_positive_coverage_artifacts_do_not_delete_a_racing_partial_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    states = {"pooled": _state(0.0), "coverage": _state(0.1)}
    paths = artifacts.PositiveCoverageArtifactPaths(
        pooled_checkpoint=tmp_path / "pooled.pt",
        coverage_checkpoint=tmp_path / "coverage.pt",
        complete_receipt=tmp_path / "complete.json",
    )
    racing_partial = tmp_path / "pooled.pt.partial"
    original = artifacts._sync_save

    def racing_save(state: dict[str, torch.Tensor], temporary: Path) -> None:
        if temporary == racing_partial:
            temporary.write_bytes(b"owned-by-racer")
        original(state, temporary)

    monkeypatch.setattr(artifacts, "_sync_save", racing_save)
    with pytest.raises(FileExistsError):
        artifacts.write_positive_coverage_artifacts(
            states=states,
            receipt=_receipt(states),
            paths=paths,
        )

    assert racing_partial.read_bytes() == b"owned-by-racer"
    assert not paths.pooled_checkpoint.exists()
    assert not paths.coverage_checkpoint.exists()
    assert not paths.complete_receipt.exists()


def test_positive_coverage_artifact_paths_reject_generated_name_aliases(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="artifact path authority"):
        artifacts.PositiveCoverageArtifactPaths(
            pooled_checkpoint=tmp_path / "model.pt",
            coverage_checkpoint=tmp_path / "model.pt.partial",
            complete_receipt=tmp_path / "complete.json",
        )


def test_linear_replay_artifacts_are_atomic_authenticated_and_no_clobber(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {
        "weight": torch.arange(12, dtype=torch.float32).reshape(3, 4).contiguous(),
        "bias": torch.arange(3, dtype=torch.float32).contiguous(),
    }
    parameter_sha256 = artifacts.affine_parameters_sha256(state["weight"], state["bias"])
    receipt = {
        "claim_eligible": False,
        "final_head_sha256": parameter_sha256,
        "schema": "fixture-replay-v1",
    }
    checkpoint = tmp_path / "base.pt"
    complete = tmp_path / "base.json"

    completed = artifacts.write_linear_replay_artifacts(
        state=state,
        receipt=receipt,
        checkpoint=checkpoint,
        complete_receipt=complete,
    )

    assert json.loads(complete.read_bytes()) == completed
    assert completed["checkpoint"] == {
        "bytes": checkpoint.stat().st_size,
        "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    }
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    torch.testing.assert_close(restored["weight"], state["weight"])
    torch.testing.assert_close(restored["bias"], state["bias"])

    raced_checkpoint = tmp_path / "raced.pt"
    raced_receipt = tmp_path / "raced.json"
    original = artifacts._sync_save

    def racing_save(value: dict[str, torch.Tensor], temporary: Path) -> None:
        original(value, temporary)
        raced_checkpoint.write_bytes(b"owned-by-racer")

    monkeypatch.setattr(artifacts, "_sync_save", racing_save)
    with pytest.raises(FileExistsError):
        artifacts.write_linear_replay_artifacts(
            state=state,
            receipt=receipt,
            checkpoint=raced_checkpoint,
            complete_receipt=raced_receipt,
        )
    assert raced_checkpoint.read_bytes() == b"owned-by-racer"
    assert not raced_receipt.exists()


def test_canonical_receipt_no_clobber_preserves_racing_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "mismatch.json"
    original = artifacts._sync_write

    def racing_write(data: bytes, temporary: Path) -> None:
        original(data, temporary)
        destination.write_bytes(b"owned-by-racer")

    monkeypatch.setattr(artifacts, "_sync_write", racing_write)
    with pytest.raises(FileExistsError):
        artifacts.write_canonical_receipt_no_clobber(
            {"claim_eligible": False, "schema": "fixture-mismatch-v1"},
            destination,
        )
    assert destination.read_bytes() == b"owned-by-racer"


def test_mean_recent_loss_uses_the_observed_tail_length() -> None:
    assert artifacts.mean_recent_loss([1.0] * 23, window=100) == 1.0
    assert artifacts.mean_recent_loss([0.0, *([1.0] * 100)], window=100) == 1.0


def test_positive_coverage_source_identity_authenticates_driver(tmp_path: Path) -> None:
    driver = tmp_path / "driver.py"
    driver.write_bytes(b"print('frozen')\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "driver.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    digest = hashlib.sha256(driver.read_bytes()).hexdigest()
    revision = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()

    assert artifacts.positive_coverage_source_identity(
        driver=driver,
        driver_sha256=digest,
        source_revision=revision,
    ) == {"driver_sha256": digest, "source_revision": revision}


@pytest.mark.parametrize("mutation", ("driver", "digest", "revision", "checkout"))
def test_positive_coverage_source_identity_rejects_drift(tmp_path: Path, mutation: str) -> None:
    driver = tmp_path / "driver.py"
    driver.write_bytes(b"print('frozen')\n")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "add", "driver.py"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    digest = hashlib.sha256(driver.read_bytes()).hexdigest()
    revision = subprocess.check_output(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"], text=True
    ).strip()
    if mutation == "driver":
        driver = tmp_path / "missing.py"
    elif mutation == "digest":
        digest = "0" * 64
    elif mutation == "revision":
        revision = "A" * 40
    else:
        revision = "0" * 40
    with pytest.raises(ValueError, match="source identity authority"):
        artifacts.positive_coverage_source_identity(
            driver=driver,
            driver_sha256=digest,
            source_revision=revision,
        )


@pytest.mark.parametrize(
    ("script", "arguments"),
    (
        (
            "replay_sop_retrieval_local_rank.py",
            (
                "--source-snapshot",
                "source.npz",
                "--source-sha256",
                "0",
                "--teacher-snapshot",
                "teacher.npz",
                "--teacher-sha256",
                "0",
                "--checkpoint",
                "checkpoint.pt",
                "--receipt",
                "receipt.json",
                "--source-revision",
                "0",
                "--driver-sha256",
                "0",
            ),
        ),
        (
            "run_sop_positive_coverage_metric.py",
            (
                "--source-snapshot",
                "source.npz",
                "--source-sha256",
                "0",
                "--teacher-snapshot",
                "teacher.npz",
                "--teacher-sha256",
                "0",
                "--base-checkpoint",
                "base.pt",
                "--base-checkpoint-sha256",
                "0",
                "--receipt",
                "receipt.json",
                "--seed",
                "0",
                "--pooled-checkpoint",
                "pooled.pt",
                "--coverage-checkpoint",
                "coverage.pt",
                "--source-revision",
                "0",
                "--driver-sha256",
                "0",
            ),
        ),
        (
            "run_cub_positive_coverage_replication.py",
            (
                "--b16",
                "b16.npz",
                "--l14",
                "l14.npz",
                "--pooled-checkpoint",
                "pooled.pt",
                "--coverage-checkpoint",
                "coverage.pt",
                "--receipt",
                "receipt.json",
                "--source-revision",
                "0",
                "--driver-sha256",
                "0",
            ),
        ),
        (
            "run_inshop_positive_coverage_replication.py",
            (
                "--b16",
                "b16.npz",
                "--l14",
                "l14.npz",
                "--pooled-checkpoint",
                "pooled.pt",
                "--coverage-checkpoint",
                "coverage.pt",
                "--receipt",
                "receipt.json",
                "--seed",
                "0",
                "--source-revision",
                "0",
                "--driver-sha256",
                "0",
            ),
        ),
    ),
)
def test_positive_coverage_drivers_require_explicit_execution(
    script: str, arguments: tuple[str, ...]
) -> None:
    result = subprocess.run(
        [sys.executable, str(_SCRIPT.parent / script), *arguments],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--execute-positive-coverage" in result.stderr


@pytest.mark.parametrize("mutation", ("state-key", "nonfinite", "parameter", "claim"))
def test_positive_coverage_artifacts_reject_invalid_authority(
    tmp_path: Path, mutation: str
) -> None:
    states = {"pooled": _state(0.0), "coverage": _state(0.1)}
    receipt = _receipt(states)
    if mutation == "state-key":
        states["extra"] = _state(0.2)
    elif mutation == "nonfinite":
        states["coverage"]["weight"][0, 0] = torch.nan
    elif mutation == "parameter":
        receipt["arms"]["coverage"]["parameter_sha256"] = "0" * 64  # type: ignore[index]
    else:
        receipt["claim_eligible"] = True
    paths = artifacts.PositiveCoverageArtifactPaths(
        pooled_checkpoint=tmp_path / "pooled.pt",
        coverage_checkpoint=tmp_path / "coverage.pt",
        complete_receipt=tmp_path / "complete.json",
    )

    with pytest.raises(ValueError, match="positive-coverage artifact authority"):
        artifacts.write_positive_coverage_artifacts(
            states=states,
            receipt=receipt,
            paths=paths,
        )
    assert not any(tmp_path.iterdir())
