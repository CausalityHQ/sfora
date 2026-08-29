from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts/build_unicom_fepf_run_config.py"
SPEC = importlib.util.spec_from_file_location(
    "build_unicom_fepf_run_config", MODULE_PATH
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


def _runtime_inference_signature() -> dict[str, object]:
    return {
        "schema": "unicom-inference-signature-v1",
        "tensors": [
            {
                **_inference_structure()["tensors"][0],
                "sha256": "3" * 64,
            }
        ],
        "total_bytes": 8,
        "aggregate_sha256": "4" * 64,
        "descriptor_dtype": "torch.float32",
        "descriptor_dimension": 512,
        "descriptor_sha256": "5" * 64,
        "operations": _inference_structure()["operations"],
    }


def _partition_inventory() -> dict[str, int]:
    return {
        "query_rows": 14_218, "gallery_rows": 12_612,
        "maximum_relevant_count": 64, "maximum_path_bytes": 120,
    }


def _canary_authority() -> dict[str, str]:
    return {"device_uuid": "GPU-registered", "environment_sha256": "d" * 64}


def _cross_task_authorities(tmp_path: Path) -> dict[str, object]:
    return {
        "cuda_canary_environment": {
            "path": str(
                (tmp_path / "artifacts/preflight/cuda-environment.json").resolve()
            ),
            "sha256": "d" * 64,
            "bytes": 1024,
        },
        "publication_budget": {
            "path": str(
                (tmp_path / "artifacts/preflight/publication-budget.json").resolve()
            ),
            "sha256": "e" * 64,
            "bytes": 2048,
        },
        "runtime_inference_signature": _runtime_inference_signature(),
    }


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
        **_cross_task_authorities(tmp_path),
    )


def test_review5_config_emits_exact_cross_task_authorities_and_cli_vectors(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    assert config["cuda_canary_environment"]["sha256"] == "d" * 64
    assert config["publication_budget"]["schema"] == "unicom-fepf-publication-budget-v1"
    assert len(config["publication_budget_sha256"]) == 64
    assert config["runtime_inference_signature"] == _runtime_inference_signature()
    for command in config["commands"]["runtime"]:
        assert "--environment-authority" in command
        assert "--environment-sha256" in command
    quality = config["commands"]["profile_quality"]
    assert "--environment-authority" in quality
    assert "--environment-sha256" in quality
    train = config["commands"]["train"]
    for flag in (
        "--environment-authority",
        "--environment-sha256",
        "--publication-budget",
        "--publication-budget-sha256",
    ):
        assert flag in train


def test_review6_signature_structure_and_root_publications_are_exact(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    changed = json.loads(json.dumps(config))
    changed["runtime_inference_signature"]["tensors"][0]["kind"] = "buffer"
    with pytest.raises(ValueError, match="signature|structure"):
        MODULE.validate_config_build(changed, source_repo)

    root = Path(config["artifact_root"])
    root.mkdir()
    (root / "arbitrary-sources.json").write_text("{}\n")
    (root / "arbitrary-result.json").write_text("{}\n")
    with pytest.raises(ValueError, match="registered|root"):
        MODULE.validate_campaign_resume(
            config, root, terminal_validator=lambda _path: None
        )


def test_review6_builder_materializes_exact_budget_before_absent_campaign_root(
    source_repo: Path, tmp_path: Path
) -> None:
    output = source_repo / "config.json"
    config, budget = MODULE.build_and_write_with_authorities(
        repo=source_repo,
        checkout_root_template=str(tmp_path / "checkout-{config_commit}"),
        artifact_root=tmp_path / "artifacts",
        output=output,
        inference_structure=_inference_structure(),
        runtime_inference_signature=_runtime_inference_signature(),
        partition_inventory=_partition_inventory(),
        cuda_canary_authority=_canary_authority(),
    )
    assert not Path(config["artifact_root"]).exists()
    budget_path = Path(config["artifact_root"]) / config["publication_budget_path"]
    assert not budget_path.exists()
    assert MODULE._sha256(MODULE.canonical_json_bytes(budget)) == (
        config["publication_budget_sha256"]
    )
    MODULE.validate_exact_publication_budget(config, budget)


def test_review7_config_embeds_transferable_budget_and_external_legacy_root(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    assert config["publication_budget"]["schema"] == (
        "unicom-fepf-publication-budget-v1"
    )
    assert config["publication_budget_path"] == "preflight/publication-budget.json"
    assert len(config["publication_budget_sha256"]) == 64
    legacy = config["legacy_runtime_authority"]
    assert tuple(legacy) == ("run_receipt", "config", "history", "checkpoints")
    assert [row["epoch"] for row in legacy["checkpoints"]] == [4, 8, 12, 16]


def test_review7_budget_expands_every_training_stage_path(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    rows = config["publication_budget"]["publications"]
    paths = {row["path"] for row in rows}
    assert "exploratory-control-stage4/epoch-0004.pt" in paths
    assert "exploratory-control-stage16/epoch-0008.pt" in paths
    assert "exploratory-candidate-stage16/evaluation-epoch-0016.json" in paths
    assert "confirmation-4-control/initialization-receipt.json" in paths
    assert "confirmation-4-candidate/run-receipt.json" in paths
    assert len(paths) == len(rows)


def test_review7_registered_builder_rejects_caller_inference_overrides() -> None:
    with pytest.raises(SystemExit):
        MODULE.parse_args(
            [
                "--repo", "/tmp/repo",
                "--checkout-root-template", "/tmp/checkout-{config_commit}",
                "--artifact-root", "/tmp/artifacts",
                "--output", "/tmp/config.json",
                "--inference-structure", "/tmp/caller-structure.json",
            ]
        )


def test_review10_non_authentic_synthesized_cpu_builder_contract(
    source_repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        MODULE,
        "_checkpoint_inference_structure",
        lambda _path: _inference_structure(),
    )
    monkeypatch.setattr(
        MODULE,
        "_checkpoint_runtime_inference_signature",
        lambda _path: _runtime_inference_signature(),
        raising=False,
    )
    monkeypatch.setattr(
        MODULE, "_partition_inventory", lambda _path: _partition_inventory()
    )
    monkeypatch.setattr(
        MODULE,
        "_registered_legacy_runtime_authority",
        lambda: _build(source_repo, tmp_path)["legacy_runtime_authority"],
        raising=False,
    )
    output = source_repo / "run-config.json"
    arguments = [
        "--repo", str(source_repo),
        "--checkout-root-template", str(tmp_path / "checkout-{config_commit}"),
        "--artifact-root", str(tmp_path / "artifacts"),
        "--output", str(output),
        "--non-authentic-synthesized-authorities",
    ]
    assert MODULE.main(arguments) == 0
    config = json.loads(output.read_bytes())
    assert config["cuda_canary_environment"] == {
        "path": str((tmp_path / "artifacts/preflight/cuda-environment.json").resolve())
    }
    assert config["cuda_canary_authority"] == {}
    assert config["publication_budget"]["schema"] == (
        "unicom-fepf-publication-budget-v1"
    )

    output.unlink()
    (source_repo / "untracked.txt").write_text("must fail closed\n")
    assert MODULE.main(arguments) == 2
    assert not output.exists()


def test_review10_target_builder_uses_actual_registered_checkpoint_and_normal_cli(
    source_repo: Path, tmp_path: Path
) -> None:
    checkpoint = Path(MODULE._inputs()["runtime_checkpoint"])
    if not checkpoint.is_file():
        pytest.skip("target-local historical authority is exercised on DGX in Task 7")
    signature = MODULE._checkpoint_runtime_inference_signature(checkpoint)
    assert signature["descriptor_dimension"] == 512
    output = source_repo / "run-config.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(MODULE_PATH),
            "--repo", str(source_repo),
            "--checkout-root-template", str(tmp_path / "exec-{config_commit}"),
            "--artifact-root", str(tmp_path / "campaign"),
            "--output", str(output),
        ],
        cwd=source_repo,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert output.is_file()


def test_review10_budget_is_recomputed_not_accepted_from_coherent_config_mutation(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    changed = json.loads(json.dumps(config))
    changed["publication_budget"]["publications"][0]["persistent_bytes"] += 1
    budget_payload = MODULE.canonical_json_bytes(changed["publication_budget"])
    changed["publication_budget_sha256"] = MODULE._sha256(budget_payload)
    changed["commands"] = MODULE._commands(
        changed["cuda_canary_environment"],
        {
            "path": str(
                (
                    Path(changed["artifact_root"])
                    / changed["publication_budget_path"]
                ).resolve()
            ),
            "sha256": changed["publication_budget_sha256"],
            "bytes": len(budget_payload),
        },
    )
    with pytest.raises(ValueError, match="exact publication budget"):
        MODULE.validate_config_document(changed)


def test_review10_budget_inventory_covers_exact_paths_and_serialization_bounds(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    rows = config["publication_budget"]["publications"]
    by_name = {row["name"]: row for row in rows}
    assert len(by_name) == len(rows)
    assert len({row["path"] for row in rows}) == len(rows)
    assert by_name["campaign:publication-budget"]["path"] == (
        "preflight/publication-budget.json"
    )
    assert by_name["campaign:directory:preflight"]["persistent_inodes"] == 1
    query = by_name[
        "exploratory-control-stage4:evaluation-epoch-0004-query"
    ]
    raw_query_bytes = config["artifact_budget_inputs"]["query_rows"] * 768 * 4
    assert query["persistent_bytes"] > raw_query_bytes
    ranked = by_name[
        "exploratory-control-stage4:evaluation-epoch-0004-ranked-prefix"
    ]
    ranked_count = min(
        max(30, config["artifact_budget_inputs"]["maximum_relevant_count"]),
        config["artifact_budget_inputs"]["gallery_rows"],
    )
    minimum_ranked = (
        config["artifact_budget_inputs"]["query_rows"]
        * ranked_count
        * (2 * config["artifact_budget_inputs"]["maximum_path_bytes"] + 128)
    )
    assert ranked["persistent_bytes"] >= minimum_ranked
    assert all(
        row["temporary_inodes"] >= 1
        for row in rows
        if not row["name"].startswith("campaign:directory:")
        and row["name"] != "campaign:controller-status"
    )


def test_review9_membership_accepts_execution_checkout_but_transfer_requires_absence(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    execution = Path(config["checkout_root_template"].replace(
        "{config_commit}", "f" * 40
    ))
    execution.mkdir()
    MODULE.validate_config_document(config)
    MODULE.validate_config_membership_document(config, source_repo)
    with pytest.raises(FileExistsError):
        MODULE.validate_transfer_handoff(config, execution)


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
    assert train[4:12] == [
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
            **_cross_task_authorities(tmp_path),
        )
    with pytest.raises(ValueError, match="template"):
        MODULE.build_run_config(
            repo=source_repo,
            checkout_root_template=str(tmp_path / "checkout-{other}"),
            artifact_root=artifact,
            inference_structure=_inference_structure(),
            partition_inventory=_partition_inventory(),
            cuda_canary_authority=_canary_authority(),
            **_cross_task_authorities(tmp_path),
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
    config, _budget_value = MODULE.build_and_write_with_authorities(
        repo=source_repo,
        checkout_root_template=str(tmp_path / "checkout-{config_commit}"),
        artifact_root=tmp_path / "artifacts",
        output=output,
        inference_structure=_inference_structure(),
        partition_inventory=_partition_inventory(),
        cuda_canary_authority=_canary_authority(),
        runtime_inference_signature=_runtime_inference_signature(),
    )
    assert output.read_bytes() == MODULE.canonical_json_bytes(config)
    with pytest.raises(FileExistsError):
        MODULE.build_and_write_with_authorities(
            repo=source_repo,
            checkout_root_template=str(tmp_path / "other-{config_commit}"),
            artifact_root=tmp_path / "other-artifacts",
            output=output,
            inference_structure=_inference_structure(),
            partition_inventory=_partition_inventory(),
            cuda_canary_authority=_canary_authority(),
            runtime_inference_signature=_runtime_inference_signature(),
        )


def test_handoff_requires_sole_config_child_clean_detached_checkout(
    source_repo: Path, tmp_path: Path
) -> None:
    config_path = source_repo / "docs" / "unicom_fepf_run_config.json"
    config_path.parent.mkdir()
    config = _build(source_repo, tmp_path)
    legacy = config["legacy_runtime_authority"]
    non_authentic = tmp_path / "non-authentic"
    for name in ("run_receipt", "config", "history"):
        legacy[name]["path"] = str(
            (non_authentic / f"{name.replace('_', '-')}.json").resolve()
        )
    for row in legacy["checkpoints"]:
        row["path"] = str(
            (non_authentic / f"epoch-{row['epoch']:04d}.pt").resolve()
        )
    config_path.write_bytes(MODULE.canonical_json_bytes(config))
    _git(source_repo, "add", str(config_path.relative_to(source_repo)))
    _git(source_repo, "commit", "-qm", "config")
    commit = _git(source_repo, "rev-parse", "HEAD")
    _git(source_repo, "checkout", "--detach", "-q", commit)
    resolved = MODULE.validate_non_authentic_synthesized_handoff(
        config_path, source_repo
    )
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
    assert config["artifact_budget_inodes"] == sum(
        row["persistent_inodes"] + row["temporary_inodes"]
        for row in config["publication_budget"]["publications"]
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
    inventory = tuple(config["publication_budget"]["publications"])
    assert inventory
    assert list(inventory) == config["publication_budget"]["publications"]
    assert sum(
        row["persistent_inodes"] + row["temporary_inodes"] for row in inventory
    ) == config["artifact_budget_inodes"]
    assert all(
        tuple(row)
        == (
            "name", "path", "persistent_bytes", "temporary_bytes",
            "persistent_inodes", "temporary_inodes",
        )
        for row in inventory
    )


def test_review2_resume_requires_paired_source_and_result_publication(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    root = Path(config["artifact_root"])
    root.mkdir()
    (root / "exploratory-decision-sources.json").write_text("{}\n")
    assert MODULE.validate_campaign_resume(
        config, root, terminal_validator=lambda _path: None
    ) == ()


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


def test_review11_budget_omits_unused_named_evaluator_temps(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    rows = config["publication_budget"]["publications"]
    temporary_rows = [row for row in rows if row["path"].endswith(".tmp")]
    assert temporary_rows == [
        {
            "name": "campaign:controller-status-temporary",
            "path": ".controller-status.json.tmp",
            "persistent_bytes": 0,
            "temporary_bytes": 256 * 1024,
            "persistent_inodes": 0,
            "temporary_inodes": 1,
        }
    ]


def test_review11_non_auth_handoff_rejects_authentic_looking_authorities(
    source_repo: Path, tmp_path: Path
) -> None:
    config_path = source_repo / "docs/unicom_fepf_run_config.json"
    config_path.parent.mkdir()
    config = _build(source_repo, tmp_path)
    legacy = config["legacy_runtime_authority"]
    authentic_root = tmp_path / "historical-authorities"
    for name in ("run_receipt", "config", "history"):
        legacy[name]["path"] = str(
            (authentic_root / f"{name.replace('_', '-')}.json").resolve()
        )
    for checkpoint in legacy["checkpoints"]:
        checkpoint["path"] = str(
            (authentic_root / f"epoch-{checkpoint['epoch']:04d}.pt").resolve()
        )
    config_path.write_bytes(MODULE.canonical_json_bytes(config))
    _git(source_repo, "add", str(config_path.relative_to(source_repo)))
    _git(source_repo, "commit", "-qm", "config")
    commit = _git(source_repo, "rev-parse", "HEAD")
    _git(source_repo, "checkout", "--detach", "-q", commit)

    with pytest.raises(ValueError, match="non-authentic|synthesized"):
        MODULE.validate_non_authentic_synthesized_handoff(config_path, source_repo)


def test_review12_authentic_builder_cli_has_only_four_authorities_and_handoff_mode() -> None:
    exact = [
        "--repo", "/source", "--checkout-root-template", "/exec-{config_commit}",
        "--artifact-root", "/campaign", "--output", "/source/config.json",
    ]
    parsed = MODULE.parse_args(exact)
    assert parsed.repo == Path("/source")
    assert parsed.validate_handoff is False
    assert MODULE.parse_args([*exact, "--validate-handoff"]).validate_handoff is True
    for override in (
        "--cuda-device-uuid", "--cuda-environment-sha256",
        "--cuda-environment-path", "--cuda-environment-bytes",
        "--publication-budget-path", "--publication-budget-sha256",
        "--publication-budget-bytes", "--runtime-inference-signature",
        "--partition-inventory",
    ):
        with pytest.raises(SystemExit):
            MODULE.parse_args([*exact, override, "caller-owned"])


def test_review12_aggregate_inventory_is_derived_only_from_exact_rows(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    rows = config["publication_budget"]["publications"]
    assert config["artifact_budget_bytes"] == sum(
        row["persistent_bytes"] + row["temporary_bytes"] for row in rows
    )
    assert config["artifact_budget_inodes"] == sum(
        row["persistent_inodes"] + row["temporary_inodes"] for row in rows
    )
    assert "artifact_inventory" not in config


def test_review13_ranked_prefix_budget_includes_complete_query_envelopes(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    inputs = config["artifact_budget_inputs"]
    ranked_count = min(
        max(30, inputs["maximum_relevant_count"]), inputs["gallery_rows"]
    )
    maximum_path = "p" * inputs["maximum_path_bytes"]
    ranked = [
        {
            "gallery_index": inputs["gallery_rows"] - 1,
            "gallery_path": maximum_path,
            "gallery_label": maximum_path,
            "score": -1.7976931348623157e308,
            "correct": False,
        }
        for _index in range(ranked_count)
    ]
    query = {
        "query_path": maximum_path,
        "query_label": maximum_path,
        "relevant_gallery_count": inputs["maximum_relevant_count"],
        "ap_at_r": 0.12345678901234568,
        "query_sha256": "f" * 64,
        "complete_ranking_sha256": "f" * 64,
        "ranked_prefix": ranked,
    }
    one = (json.dumps([query], indent=2, allow_nan=False) + "\n").encode()
    two = (json.dumps([query, query], indent=2, allow_nan=False) + "\n").encode()
    exact_official_upper_bound = len(one) + (inputs["query_rows"] - 1) * (
        len(two) - len(one)
    )
    row = next(
        item
        for item in config["publication_budget"]["publications"]
        if item["name"]
        == "exploratory-control-stage4:evaluation-epoch-0004-ranked-prefix"
    )
    assert row["persistent_bytes"] >= exact_official_upper_bound


def test_review14_ranked_prefix_budget_uses_max_width_ap_at_r(
    source_repo: Path, tmp_path: Path
) -> None:
    config = _build(source_repo, tmp_path)
    inputs = config["artifact_budget_inputs"]
    count = min(max(30, inputs["maximum_relevant_count"]), inputs["gallery_rows"])
    maximum_path = "p" * inputs["maximum_path_bytes"]
    ranked_row = {
        "gallery_index": inputs["gallery_rows"] - 1,
        "gallery_path": maximum_path,
        "gallery_label": maximum_path,
        "score": -1.7976931348623157e308,
        "correct": False,
    }
    query = {
        "query_path": maximum_path,
        "query_label": maximum_path,
        "relevant_gallery_count": inputs["maximum_relevant_count"],
        "ap_at_r": -1.7976931348623157e308,
        "query_sha256": "f" * 64,
        "complete_ranking_sha256": "f" * 64,
        "ranked_prefix": [dict(ranked_row) for _index in range(count)],
    }
    one = MODULE.canonical_json_bytes([query])
    two = MODULE.canonical_json_bytes([query, query])
    required = len(one) + (inputs["query_rows"] - 1) * (len(two) - len(one))
    row = next(
        item for item in config["publication_budget"]["publications"]
        if item["name"]
        == "exploratory-control-stage4:evaluation-epoch-0004-ranked-prefix"
    )
    assert row["persistent_bytes"] >= required


@pytest.mark.parametrize("namespace", ("canary-evidence.staging", "canary-evidence"))
def test_review12_public_campaign_resume_accepts_registered_observation_namespace(
    source_repo: Path, tmp_path: Path, namespace: str
) -> None:
    config = _build(source_repo, tmp_path)
    root = Path(config["artifact_root"])
    evidence = root / "preflight" / namespace
    evidence.mkdir(parents=True)
    names = (
        ("observation.json",)
        if namespace.endswith(".staging")
        else (
            "observation.json", "initialization-receipt.json",
            "cache-inventory.json", "model-inventory.json", "rng-audit.json",
            "model-modes.json", "environment.json", "manifest.json",
        )
    )
    for name in names:
        (evidence / name).write_text("{}\n")
    MODULE.validate_campaign_resume(
        config, root, terminal_validator=lambda _path: None
    )
