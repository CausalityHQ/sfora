from __future__ import annotations

import importlib.util
import json
import math
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_unicom_fepf_run_config", ROOT / "scripts/build_unicom_fepf_run_config.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _inference_structure() -> dict[str, object]:
    return {
        "schema": "unicom-fepf-structure-v1",
        "tensors": [{
            "name": "weight", "kind": "parameter", "shape": [1, 2],
            "dtype": "torch.float32", "numel": 2, "element_size": 4, "bytes": 8,
        }],
        "classifier": {
            "shape": [2, 2], "dtype": "torch.float32", "numel": 4,
            "element_size": 4, "bytes": 16,
        },
        "operations": [
            "official_forward", "full768_l2", "prefix512", "squared_euclidean"
        ],
    }


def _partition_inventory() -> dict[str, int]:
    return {
        "query_rows": 14_218, "gallery_rows": 12_612,
        "maximum_relevant_count": 64, "maximum_path_bytes": 120,
    }


def _canary_authority() -> dict[str, str]:
    return {"device_uuid": "GPU-registered", "environment_sha256": "d" * 64}


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "source"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "Test Operator")
    _git(repo, "config", "user.email", "operator@example.test")
    for relative in MODULE.REGISTERED_SOURCE_PATHS:
        destination = repo / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = ROOT / relative
        destination.write_bytes(source.read_bytes() if source.exists() else b"placeholder\n")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "source")
    return repo


def _build(source_repo: Path, tmp_path: Path) -> dict[str, object]:
    return MODULE.build_run_config(
        repo=source_repo,
        checkout_root_template=str(tmp_path / "checkout-{config_commit}"),
        artifact_root=tmp_path / "artifacts",
        inference_structure=_inference_structure(),
        partition_inventory=_partition_inventory(),
        cuda_canary_authority=_canary_authority(),
    )


def test_build_config_freezes_registered_protocol_and_commands(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    MODULE.validate_config_build(config, source_repo)
    assert config["schema"] == "unicom-fepf-run-config-v1"
    assert config["source_commit"] == _git(source_repo, "rev-parse", "HEAD")
    assert config["model"] == {
        "revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "checkpoint_sha256": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
        "partition_sha256": "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c",
    }
    assert config["inputs"] == {
        "unicom_checkout": "/home/riomus/unicom-d71992e",
        "checkpoint": "/home/riomus/.cache/unicom/FP16-ViT-L-14-336px.pt",
        "dataset_root": "/home/riomus/datasets/inshop_official_standard",
        "partition": "/home/riomus/datasets/inshop_official_standard/Eval/list_eval_partition.txt",
        "runtime_checkpoint": "/home/riomus/unicom-ema-imprinted-387d697-seed2-e16/epoch-0016.pt",
        "runtime_run_receipt": (
            "/home/riomus/unicom-ema-imprinted-387d697-seed2-e16/run-receipt.json"
        ),
    }
    assert config["runtime_order"] == [
        "current", "composed", "composed", "current",
        "current", "composed", "composed", "current",
    ]
    assert config["exploratory"]["arms"] == ["imprinted", "fepf_mean", "fepf_random"]
    assert config["confirmation_pairs"] == [
        [7, 20_260_828], [8, 271_828], [9, 314_159],
        [10, 1_618_033], [11, 57_721],
    ]
    assert config["thresholds"]["row_norm_rtol"] == 2e-6
    assert config["thresholds"]["row_norm_atol"] == 2e-7
    assert config["cuda_canary_command"] == [
        ".venv/bin/python", "-I", "-B", "scripts/run_unicom_fepf_cuda_canary.py",
        "--config", "docs/unicom_fepf_run_config.json",
    ]
    assert config["cuda_canary_receipt"] == "preflight/cuda_canary_v1.json"
    train = config["commands"]["train"]
    assert train[-8:] == [
        "--unicom-checkout", config["inputs"]["unicom_checkout"],
        "--checkpoint", config["inputs"]["checkpoint"],
        "--dataset-root", config["inputs"]["dataset_root"],
        "--run-config", "docs/unicom_fepf_run_config.json",
    ]
    assert config["artifact_budget_bytes"] > 52 * 64 * 1024 * 1024
    assert config["artifact_budget_inodes"] > 52


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(source_commit="f" * 40),
        lambda value: value["runtime_order"].__setitem__(0, "composed"),
        lambda value: value["confirmation_pairs"].__setitem__(0, [7, 271_828]),
        lambda value: value["thresholds"].update(row_norm_rtol=True),
        lambda value: value.update(artifact_budget_bytes=True),
        lambda value: value["commands"].pop("runtime"),
    ],
)
def test_build_validation_rejects_protocol_mutations(
    source_repo: Path, tmp_path: Path, mutation
) -> None:
    config = _build(source_repo, tmp_path)
    mutation(config)
    with pytest.raises(ValueError):
        MODULE.validate_config_build(config, source_repo)


def test_builder_requires_distinct_non_nested_absolute_roots(
    source_repo: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "runs"
    with pytest.raises(ValueError, match="distinct"):
        MODULE.build_run_config(
            repo=source_repo,
            checkout_root_template=str(artifact / "checkout-{config_commit}"),
            artifact_root=artifact,
            inference_structure=_inference_structure(),
            partition_inventory=_partition_inventory(),
            cuda_canary_authority=_canary_authority(),
        )
    with pytest.raises(ValueError, match="template"):
        MODULE.build_run_config(
            repo=source_repo,
            checkout_root_template=str(tmp_path / "checkout-{other}"),
            artifact_root=artifact,
            inference_structure=_inference_structure(),
            partition_inventory=_partition_inventory(),
            cuda_canary_authority=_canary_authority(),
        )


def test_build_requires_clean_committed_source_and_absent_destinations(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    (source_repo / MODULE.REGISTERED_SOURCE_PATHS[0]).write_text("dirty\n")
    with pytest.raises(ValueError, match="clean"):
        MODULE.validate_config_build(config, source_repo)
    _git(source_repo, "checkout", "--", MODULE.REGISTERED_SOURCE_PATHS[0])
    Path(config["artifact_root"]).mkdir()
    with pytest.raises(FileExistsError):
        MODULE.validate_config_build(config, source_repo)


def test_canonical_builder_writes_once_and_reloads_distinct_bytes(
    source_repo: Path, tmp_path: Path
) -> None:
    output = source_repo / "config.json"
    config = MODULE.build_and_write(
        repo=source_repo,
        checkout_root_template=str(tmp_path / "checkout-{config_commit}"),
        artifact_root=tmp_path / "artifacts",
        output=output,
        inference_structure=_inference_structure(),
        partition_inventory=_partition_inventory(),
        cuda_canary_authority=_canary_authority(),
    )
    assert output.read_bytes() == MODULE.canonical_json_bytes(config)
    with pytest.raises(FileExistsError):
        MODULE.build_and_write(
            repo=source_repo,
            checkout_root_template=str(tmp_path / "other-{config_commit}"),
            artifact_root=tmp_path / "other-artifacts",
            output=output,
            cuda_canary_authority=_canary_authority(),
        )


def test_handoff_requires_sole_config_child_clean_detached_checkout(
    source_repo: Path, tmp_path: Path
) -> None:
    config_path = source_repo / "docs" / "unicom_fepf_run_config.json"
    config_path.parent.mkdir()
    config = _build(source_repo, tmp_path)
    config_path.write_bytes(MODULE.canonical_json_bytes(config))
    _git(source_repo, "add", str(config_path.relative_to(source_repo)))
    _git(source_repo, "commit", "-qm", "config")
    commit = _git(source_repo, "rev-parse", "HEAD")
    _git(source_repo, "checkout", "--detach", "-q", commit)
    resolved = MODULE.validate_config_handoff(config_path, source_repo)
    assert resolved["config_commit"] == commit
    assert resolved["checkout_root"] == str(tmp_path / f"checkout-{commit}")


def test_prepare_artifact_root_checks_capacity_then_atomically_creates(
    tmp_path: Path
) -> None:
    root = tmp_path / "artifacts"
    observed: list[Path] = []

    def statvfs(path: Path):
        observed.append(path)
        return os.statvfs(path)

    MODULE.prepare_artifact_root(
        root, required_bytes=1, required_inodes=1, statvfs=statvfs
    )
    assert root.is_dir() and not root.is_symlink()
    assert observed == [tmp_path, root]


def test_prepare_artifact_root_rejects_absent_parent_capacity_and_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="parent"):
        MODULE.prepare_artifact_root(
            tmp_path / "absent" / "root", required_bytes=1, required_inodes=1
        )

    class Tiny:
        f_bavail = 0
        f_frsize = 4096
        f_favail = 0

    with pytest.raises(OSError, match="capacity"):
        MODULE.prepare_artifact_root(
            tmp_path / "small", required_bytes=1, required_inodes=1,
            statvfs=lambda _path: Tiny(),
        )
    assert not (tmp_path / "small").exists()

    root = tmp_path / "race"
    original = Path.mkdir

    def racing_mkdir(path: Path, *args, **kwargs):
        original(path)
        raise FileExistsError(path)

    monkeypatch.setattr(Path, "mkdir", racing_mkdir)
    with pytest.raises(FileExistsError):
        MODULE.prepare_artifact_root(root, required_bytes=1, required_inodes=1)


def test_remaining_capacity_uses_reserved_prior_bytes_and_inodes(tmp_path: Path) -> None:
    root = tmp_path / "artifacts"
    root.mkdir()
    terminal = root / "terminal.json"
    terminal.write_text(json.dumps({"schema": "terminal"}))
    MODULE.require_remaining_capacity(
        root, total_budget_bytes=terminal.stat().st_size + 1,
        total_budget_inodes=2, consumed_bytes=terminal.stat().st_size,
        consumed_inodes=1,
    )
    assert terminal.exists()
    with pytest.raises(OSError):
        MODULE.require_remaining_capacity(
            root, total_budget_bytes=10**30, total_budget_inodes=2,
            consumed_bytes=0, consumed_inodes=0,
        )


def test_config_contains_task2_task4_task5_authority_schemas(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    assert config["source"] == {"commit": config["source_commit"]}
    assert config["handoff"] == {
        "config_parent": config["source_commit"],
        "config_commit_paths": ["docs/unicom_fepf_run_config.json"],
        "execution_checkout": "config_commit_detached_clean",
    }
    assert config["parent_trainer_commit"] == MODULE.PARENT_TRAINER_COMMIT
    assert config["parent_trainer_path"] == MODULE.PARENT_TRAINER_PATH
    assert config["parent_trainer_sha256"] == MODULE.PARENT_TRAINER_SHA256
    assert config["fepf_inference_structure"] == _inference_structure()
    assert {
        key: config["artifact_budget_inputs"][key] for key in _partition_inventory()
    } == _partition_inventory()
    assert config["artifact_budget_inodes"] == math.ceil(
        1.25 * config["artifact_budget_inputs"]["planned_file_inodes"]
    )


def test_review2_config_is_task4_canonical_and_binds_live_profile_sources(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    assert MODULE.canonical_json_bytes(config) == (
        json.dumps(config, indent=2, allow_nan=False) + "\n"
    ).encode()
    source_hashes = {row["path"]: row["sha256"] for row in config["source_files"]}
    assert config["live_trainer_sha256"] == source_hashes["scripts/train_unicom_inshop.py"]
    assert config["profiler_sha256"] == source_hashes["scripts/profile_unicom_training_step.py"]


def test_review2_budget_has_typed_exact_publication_inventory(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    inventory = MODULE.registered_artifact_inventory(config)
    assert inventory
    assert list(inventory) == config["artifact_inventory"]
    assert sum(row["inodes"] for row in inventory) == config["artifact_budget_inputs"][
        "planned_file_inodes"
    ]
    assert all(tuple(row) == ("role", "count", "bytes_each", "inodes") for row in inventory)


def test_review2_resume_requires_paired_source_and_result_publication(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    root = Path(config["artifact_root"])
    root.mkdir()
    (root / "exploratory-decision-sources.json").write_text("{}\n")
    with pytest.raises(ValueError, match="incomplete"):
        MODULE.validate_campaign_resume(
            config, root, terminal_validator=lambda _path: None
        )


def test_resume_rejects_unknown_and_incomplete_stage_paths(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    root = Path(config["artifact_root"])
    root.mkdir()
    (root / "unknown-stage").mkdir()
    with pytest.raises(ValueError, match="registered"):
        MODULE.validate_campaign_resume(
            config, root, terminal_validator=lambda _path: None
        )
    (root / "unknown-stage").rmdir()
    (root / "runtime-00").mkdir()
    with pytest.raises(ValueError, match="terminal"):
        MODULE.validate_campaign_resume(
            config, root, terminal_validator=lambda _path: None
        )
