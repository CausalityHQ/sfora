from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pass201_pa_source_v2_contract import (  # noqa: E402
    bind_external_file,
    bind_import_roots,
    bind_merkle,
    bind_repo_blob,
    canonical_json_bytes,
    load_strict_json_bytes,
    validate_authorization_topology,
    validate_complete_receipt,
    validate_prelaunch,
)

H64 = "a" * 64
G40 = "b" * 40


def file_binding(path: str = "scripts/controller.py") -> dict[str, Any]:
    return {"path": path, "git_mode": "100644", "bytes": 1, "sha256": H64, "git_blob": G40}


def external(path: str = "/usr/bin/python") -> dict[str, Any]:
    return {"path": path, "mode": 0o100755, "device": 1, "inode": 2, "bytes": 1, "sha256": H64}


def merkle(root: str = "src/sfora") -> dict[str, Any]:
    return {
        "root": root,
        "algorithm": "pass201-length-framed-merkle-v1",
        "count": 1,
        "bytes": 1,
        "root_sha256": H64,
    }


@pytest.fixture
def valid_prelaunch() -> dict[str, Any]:
    absence = {
        k: "ENOENT"
        for k in (
            "run_directory",
            "report",
            "checkpoint",
            "log",
            "resolved_config",
            "train_manifest",
            "receipt",
        )
    }
    run_directory = "reports/generated/pass201_source_v2/run-v2"
    output_names = {
        "report": "report.json",
        "checkpoint": "checkpoint.pt",
        "log": "training.log",
        "resolved_config": "resolved_config.json",
        "train_manifest": "train_manifest.json",
        "receipt": "receipt.json",
    }
    outputs = {
        key: {"path": f"{run_directory}/{name}", "required_absent": True}
        for key, name in output_names.items()
    }
    return {
        "schema_version": "pass201-pa-source-v2-prelaunch-v1",
        "status": "frozen",
        "purpose": "ordinary source",
        "source_commit": G40,
        "authorization": {
            "manifest_path": "docs/pass201_pa_source_v2_prelaunch.json",
            "required_parent_commit": G40,
            "required_diff_paths": ["docs/pass201_pa_source_v2_prelaunch.json"],
            "required_diff_status": ["A"],
            "required_diff_modes": ["100644"],
            "clean_policy": "empty-porcelain-v1-z",
            "frozen_absence_checked_utc": "now",
            "frozen_absence": absence,
        },
        "controller": file_binding(),
        "source": {
            "files": [file_binding()],
            "python_tree": merkle(),
            "pyproject": file_binding("pyproject.toml"),
            "lockfile": file_binding("uv.lock"),
            "equivalence_test_id": "eq",
        },
        "execution": {
            "checkout_root": "/checkout",
            "cwd": "/checkout",
            "python": external(),
            "python_realpath": "/usr/bin/python3",
            "python_version": "3.12",
            "git": external("/usr/bin/git"),
            "python_packages": {"bytes": 1, "sha256": H64},
            "python_import_roots": [{"entry": "/missing", "status": "nonexistent"}],
            "environment": {
                **{
                    k: "1"
                    for k in (
                        "HOME",
                        "PATH",
                        "PYTHONPATH",
                        "PYTHONNOUSERSITE",
                        "PYTHONDONTWRITEBYTECODE",
                        "LD_LIBRARY_PATH",
                        "CUDA_VISIBLE_DEVICES",
                        "CUBLAS_WORKSPACE_CONFIG",
                        "PYTHONHASHSEED",
                        "LC_ALL",
                        "LANG",
                        "TZ",
                        "OMP_NUM_THREADS",
                        "MKL_NUM_THREADS",
                        "XDG_CACHE_HOME",
                        "TORCH_HOME",
                    )
                },
                "PATH": "/usr/bin:/usr/bin:/bin",
                "PYTHONPATH": "/checkout/src",
            },
            "environment_policy": "replace",
            "argv": ["python", "-m", "sfora.cli"],
            "objective": "proxy_anchor",
            "seed": 0,
            "expected_config_json": "{}\n",
            "expected_config_sha256": hashlib.sha256(b"{}\n").hexdigest(),
            "recipe_id": "recipe",
            "recipe_digest": H64,
            "schedule": {"resolved_train_steps": 10, "steps_per_epoch": 2, "total_epochs": 5},
            "pretrained_checkpoint": external("/models/pretrained.pt"),
        },
        "dataset": {
            "root": "/home/riomus/datasets/inshop_official_standard",
            "partition": external("/data/partition.txt"),
            "partition_lines": 1,
            "bundle": {
                "train": 1,
                "query": 1,
                "gallery": 1,
                "protocol": "query_gallery",
                "protocol_name": "official",
            },
            "declared_image_root": "/home/riomus/datasets/inshop_official_standard/Img/img",
            "resolved_image_root": "/home/riomus/datasets/inshop_official_standard/img/img",
            "image_root_link": {
                "path": "/home/riomus/datasets/inshop_official_standard/Img",
                "target": "img",
                "lstat_mode": 0o120777,
            },
            "image_tree": merkle("/home/riomus/datasets/inshop_official_standard/img/img"),
            "image_tree_leaf_base": "resolved_image_root",
            "image_tree_leaf_schema": "relative_path,size,sha256",
            "selection_policy": "full_official_partition",
            "optimization_authority": {
                "algorithm_id": "pass201-production-invocation-capture-v1",
                "row_count": 1,
                "identity_count": 1,
                "ordered_row_sha256": H64,
                "resolved_membership_sha256": H64,
            },
        },
        "outputs": {
            "run_directory": run_directory,
            "run_directory_required_absent": True,
            **outputs,
        },
        "sidecars": {
            "config_algorithm": "pass201-resolved-config-v2",
            "manifest_algorithm": "pass201-inshop-benchmark-row-suffix-v2",
            "schedule_algorithm": "pass201-inshop-completed-epoch-v1",
            "config_schema": "canonical-json-object-v1",
            "manifest_schema": "pass201-train-manifest-v1",
        },
        "postconditions": {
            "required_exit_code": 0,
            "require_source_equal": True,
            "require_partition_equal": True,
            "require_image_tree_equal": True,
            "require_two_process_sidecar_identity": True,
            "require_restricted_checkpoint_metadata": True,
            "require_complete_receipt": True,
        },
    }


@pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}', b'{"x":Infinity}'])
def test_strict_json_rejects_ambiguous_bytes(raw: bytes) -> None:
    with pytest.raises(ValueError):
        load_strict_json_bytes(raw)


def test_strict_json_canonical_bytes() -> None:
    assert canonical_json_bytes({"é": 1, "a": [True]}) == '{"a":[true],"é":1}\n'.encode()


@pytest.mark.parametrize("replacement", [True, 1.0])
def test_schema_rejects_non_integer_schedule(
    valid_prelaunch: dict[str, Any], replacement: object
) -> None:
    valid_prelaunch["execution"]["schedule"]["resolved_train_steps"] = replacement
    with pytest.raises(ValueError, match="resolved_train_steps"):
        validate_prelaunch(valid_prelaunch)


def test_prelaunch_rejects_false_required_predicate(valid_prelaunch: dict[str, Any]) -> None:
    valid_prelaunch["postconditions"]["require_source_equal"] = False
    with pytest.raises(ValueError, match="require_source_equal"):
        validate_prelaunch(valid_prelaunch)


@pytest.mark.parametrize("bad", ["A" * 64, "a" * 63])
def test_schema_rejects_noncanonical_hash(valid_prelaunch: dict[str, Any], bad: str) -> None:
    valid_prelaunch["controller"]["sha256"] = bad
    with pytest.raises(ValueError, match="sha256"):
        validate_prelaunch(valid_prelaunch)


def test_schema_rejects_absolute_repo_path(valid_prelaunch: dict[str, Any]) -> None:
    valid_prelaunch["controller"]["path"] = "/scripts/controller.py"
    with pytest.raises(ValueError, match="path"):
        validate_prelaunch(valid_prelaunch)


def test_schema_rejects_relative_external_path(valid_prelaunch: dict[str, Any]) -> None:
    valid_prelaunch["execution"]["python"]["path"] = "venv/bin/python"
    with pytest.raises(ValueError, match="path"):
        validate_prelaunch(valid_prelaunch)


@pytest.mark.parametrize("operation", ["missing", "extra"])
def test_schema_rejects_nested_key_drift(valid_prelaunch: dict[str, Any], operation: str) -> None:
    if operation == "missing":
        del valid_prelaunch["source"]["python_tree"]["count"]
    else:
        valid_prelaunch["source"]["python_tree"]["surprise"] = 1
    with pytest.raises(ValueError):
        validate_prelaunch(valid_prelaunch)


def test_schema_rejects_open_ended_absence_keys(valid_prelaunch: dict[str, Any]) -> None:
    valid_prelaunch["authorization"]["frozen_absence"]["extra"] = "ENOENT"
    with pytest.raises(ValueError, match="frozen_absence"):
        validate_prelaunch(valid_prelaunch)


def test_schema_accepts_complete_prelaunch(valid_prelaunch: dict[str, Any]) -> None:
    authority = validate_prelaunch(valid_prelaunch)
    assert authority.expected_train_steps == 10
    assert authority.expected_config_bytes == b"{}\n"
    valid_prelaunch["status"] = "changed"
    assert authority.payload["status"] == "frozen"


def test_external_file_binds_open_descriptor(tmp_path: Path) -> None:
    path = tmp_path / "data"
    path.write_bytes(b"bound bytes")
    binding = bind_external_file(path)
    st = path.stat()
    assert (binding.path, binding.device, binding.inode, binding.byte_count) == (
        path.resolve(),
        st.st_dev,
        st.st_ino,
        11,
    )
    assert binding.sha256 == hashlib.sha256(b"bound bytes").hexdigest()


def test_external_file_rejects_symlink_and_directory(tmp_path: Path) -> None:
    target = tmp_path / "target"
    target.write_bytes(b"x")
    (tmp_path / "link").symlink_to(target)
    with pytest.raises(ValueError):
        bind_external_file(tmp_path / "link")
    with pytest.raises(ValueError):
        bind_external_file(tmp_path)


def test_external_file_detects_mutation_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pass201_pa_source_v2_contract as contract

    path = tmp_path / "data"
    path.write_bytes(b"old")
    real_read = os.read
    changed = False

    def mutating_read(fd: int, size: int) -> bytes:
        nonlocal changed
        data = real_read(fd, size)
        if data and not changed:
            changed = True
            path.write_bytes(b"new value")
        return data

    monkeypatch.setattr(contract.os, "read", mutating_read)
    with pytest.raises(ValueError, match="changed"):
        bind_external_file(path)


def test_ordinary_merkle_has_exact_framing_and_odd_duplication(tmp_path: Path) -> None:
    (tmp_path / "a").write_bytes(b"A")
    (tmp_path / "b").write_bytes(b"BB")
    (tmp_path / "é").write_bytes(b"C")
    binding = bind_merkle(tmp_path)
    assert (binding.count, binding.byte_count) == (3, 4)
    assert binding.root_sha256 == "2172ee9f531e19532fd146b740805da7cbd641ba47671aa93d0cfe3b81ce154e"


def test_ordinary_merkle_rejects_symlink(tmp_path: Path) -> None:
    (tmp_path / "file").write_bytes(b"x")
    (tmp_path / "link").symlink_to("file")
    with pytest.raises(ValueError, match="symlink"):
        bind_merkle(tmp_path)


def _path_emitter(tmp_path: Path) -> Path:
    script = tmp_path / "python"
    script.write_text("#!/usr/bin/python3\nimport json,os\nprint(os.environ['BOUND_PATHS'])\n")
    script.chmod(0o755)
    return script


def test_import_root_preserves_tags_and_binds_external_targets(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    (checkout / "src").mkdir(parents=True)
    root = tmp_path / "root"
    root.mkdir()
    (root / "module.py").write_bytes(b"x")
    (root / "cached.pyc").write_bytes(b"pyc")
    (root / "x.pth").write_text("extra\n")
    (root / "sitecustomize.py").write_text("x=1\n")
    (root / "internal").symlink_to("module.py")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "other.py").write_bytes(b"outside")
    external_file = tmp_path / "external.py"
    external_file.write_bytes(b"external")
    (root / "external-file").symlink_to(external_file)
    (root / "external-dir").symlink_to(outside)
    archive = tmp_path / "lib.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("m.py", "x=1")
    missing = tmp_path / "missing"
    entries = [str(checkout), str(checkout / "src"), str(root), str(archive), str(missing)]
    bound = bind_import_roots(
        _path_emitter(tmp_path), {"BOUND_PATHS": json.dumps(entries)}, checkout
    )
    assert [item.status for item in bound] == ["directory", "zip", "nonexistent"]
    directory = bound[0].directory  # type: ignore[union-attr]
    assert (directory.tree.regular_count, directory.tree.symlink_count) == (4, 3)
    assert [target.kind for target in directory.external_symlink_targets] == ["directory", "file"]
    assert directory.external_symlink_targets[1].target_text == str(external_file)


def test_import_root_rejects_external_symlink_cycle(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    one = tmp_path / "one"
    two = tmp_path / "two"
    one.mkdir()
    two.mkdir()
    (one / "to-two").symlink_to(two)
    (two / "to-one").symlink_to(one)
    with pytest.raises(ValueError, match="cycle"):
        bind_import_roots(
            _path_emitter(tmp_path), {"BOUND_PATHS": json.dumps([str(one)])}, checkout
        )


def output(path: str) -> dict[str, Any]:
    return {"path": path, "file_type": "regular", "mode": 0o100644, "bytes": 1, "sha256": H64}


def valid_receipt(authority: Any) -> dict[str, Any]:
    source_tree = merkle()
    partition = external("/data/partition.txt")
    images = merkle("/home/riomus/datasets/inshop_official_standard/img/img")
    pretrained = external("/models/pretrained.pt")
    return {
        "schema_version": "pass201-pa-source-v2-receipt-v1",
        "status": "complete",
        "candidate_values_computed": False,
        "authorization": {
            "authorization_commit": G40,
            "source_commit": authority.source_commit,
            "manifest_path": "docs/pass201_pa_source_v2_prelaunch.json",
            "manifest_bytes": 1,
            "manifest_sha256": H64,
            "manifest_git_blob": G40,
            "parent_verified": True,
            "single_addition_verified": True,
            "detached_head_verified": True,
            "clean_policy_verified": True,
        },
        "controller": {
            "file": file_binding(),
            "python": external(),
            "python_packages": {"bytes": 1, "sha256": H64},
            "source_tree": source_tree,
        },
        "command": {
            "cwd": "/checkout",
            "environment": copy.deepcopy(authority.payload["execution"]["environment"]),
            "argv": ["python", "-m", "sfora.cli"],
        },
        "preflight": {
            "started_utc": "now",
            "run_directory_absent": True,
            "source_tree": source_tree,
            "partition": partition,
            "image_tree": images,
            "pretrained_checkpoint": pretrained,
            "outputs_absent": {
                k: True
                for k in (
                    "report",
                    "checkpoint",
                    "log",
                    "resolved_config",
                    "train_manifest",
                    "receipt",
                )
            },
        },
        "process": {"pid": 1, "started_utc": "now", "ended_utc": "later", "exit_code": 0},
        "postflight": {
            "ended_utc": "later",
            "source_tree": source_tree,
            "partition": partition,
            "image_tree": images,
            "pretrained_checkpoint": pretrained,
            "source_equal": True,
            "partition_equal": True,
            "image_tree_equal": True,
            "pretrained_checkpoint_equal": True,
        },
        "outputs": {
            k: output(authority.payload["outputs"][k]["path"])
            for k in ("report", "checkpoint", "log", "resolved_config", "train_manifest")
        },
        "checkpoint_metadata": {
            "literal_top_keys": [
                "arch",
                "artifact_selection",
                "evaluation_model_source",
                "state_dict",
                "training_config",
                "training_step",
            ],
            "artifact_selection": "final_training_state",
            "evaluation_model_source": "student",
            "arch": {
                "backbone_name": "bn_inception",
                "pretrained_weights": "bn_inception_52deb4733",
                "head_pooling": "avg_max",
                "embedding_dimensions": 512,
                "embedding_head_init": "kaiming_normal",
                "embedding_layer_norm": False,
            },
            "training_step": 10,
            "training_config_sha256": authority.expected_config_sha256,
            "state_dict_storage_materialized": False,
        },
        "sidecar_derivation": {
            "config_algorithm": "pass201-resolved-config-v2",
            "manifest_algorithm": "pass201-inshop-benchmark-row-suffix-v2",
            "schedule_algorithm": "pass201-inshop-completed-epoch-v1",
            "source_files": [file_binding()],
            "input_hashes": {
                k: authority.expected_config_sha256 if k == "expected_config" else H64
                for k in (
                    "manifest",
                    "source_tree",
                    "partition",
                    "image_tree",
                    "pretrained_checkpoint",
                    "report",
                    "checkpoint",
                    "expected_config",
                )
            },
            "child_processes": [
                {"ordinal": 1, "pid": 11, "config_sha256": H64, "manifest_sha256": H64},
                {"ordinal": 2, "pid": 12, "config_sha256": H64, "manifest_sha256": H64},
            ],
            "row_count": 1,
            "identity_count": 1,
            "ordered_row_sha256": H64,
            "resolved_membership_count": 1,
            "resolved_membership_sha256": H64,
            "membership_covered_by_preflight": True,
            "membership_covered_by_postflight": True,
        },
        "scope": {
            "ordinary_source_uses_official_query_gallery": True,
            "uses_pass201_operator_data": False,
            "pass201_candidate_paths_read": False,
            "authorized_action": "source_binding_only",
        },
    }


def test_receipt_schema_rejects_third_child_and_open_hashes(
    valid_prelaunch: dict[str, Any],
) -> None:
    authority = validate_prelaunch(valid_prelaunch)
    receipt = valid_receipt(authority)
    receipt["sidecar_derivation"]["child_processes"].append(
        copy.deepcopy(receipt["sidecar_derivation"]["child_processes"][1])
    )
    with pytest.raises(ValueError, match="child_processes"):
        validate_complete_receipt(receipt, authority)
    receipt = valid_receipt(authority)
    receipt["sidecar_derivation"]["input_hashes"]["extra"] = H64
    with pytest.raises(ValueError, match="input_hashes"):
        validate_complete_receipt(receipt, authority)


def test_receipt_schema_rejects_false_success_and_duplicate_pid(
    valid_prelaunch: dict[str, Any],
) -> None:
    authority = validate_prelaunch(valid_prelaunch)
    receipt = valid_receipt(authority)
    receipt["postflight"]["source_equal"] = False
    with pytest.raises(ValueError, match="source_equal"):
        validate_complete_receipt(receipt, authority)
    receipt = valid_receipt(authority)
    receipt["sidecar_derivation"]["child_processes"][1]["pid"] = 11
    with pytest.raises(ValueError, match="distinct"):
        validate_complete_receipt(receipt, authority)


def test_receipt_schema_accepts_complete_receipt(valid_prelaunch: dict[str, Any]) -> None:
    authority = validate_prelaunch(valid_prelaunch)
    result = validate_complete_receipt(valid_receipt(authority), authority)
    assert result.authorization_commit == G40
    assert set(result.output_evidence) == {
        "report",
        "checkpoint",
        "log",
        "resolved_config",
        "train_manifest",
    }


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, stdout=subprocess.PIPE
    ).stdout.strip()


def authorization_repo(tmp_path: Path, valid_prelaunch: dict[str, Any]) -> tuple[Path, Any, str]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    (repo / "base").write_text("base")
    _git(repo, "add", "base")
    _git(repo, "commit", "-qm", "base")
    source = _git(repo, "rev-parse", "HEAD")
    valid_prelaunch["source_commit"] = source
    valid_prelaunch["authorization"]["required_parent_commit"] = source
    valid_prelaunch["execution"]["checkout_root"] = str(repo)
    valid_prelaunch["execution"]["cwd"] = str(repo)
    valid_prelaunch["execution"]["environment"]["PYTHONPATH"] = str(repo / "src")
    git_binding = bind_external_file(Path(shutil.which("git") or "").resolve())
    valid_prelaunch["execution"]["git"] = {
        "path": str(git_binding.path),
        "mode": git_binding.mode,
        "device": git_binding.device,
        "inode": git_binding.inode,
        "bytes": git_binding.byte_count,
        "sha256": git_binding.sha256,
    }
    authority = validate_prelaunch(valid_prelaunch)
    manifest = repo / "docs/pass201_pa_source_v2_prelaunch.json"
    manifest.parent.mkdir()
    manifest.write_bytes(canonical_json_bytes(valid_prelaunch))
    _git(repo, "add", str(manifest.relative_to(repo)))
    _git(repo, "commit", "-qm", "authorize")
    head = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "--detach", head)
    return repo, authority, head


def test_git_topology_accepts_exact_detached_single_addition(
    tmp_path: Path, valid_prelaunch: dict[str, Any]
) -> None:
    repo, authority, head = authorization_repo(tmp_path, valid_prelaunch)
    assert validate_authorization_topology(repo, authority) == head
    blob = bind_repo_blob(repo, head, PurePosixPath("docs/pass201_pa_source_v2_prelaunch.json"))
    assert blob.git_mode == "100644" and blob.byte_count > 0


@pytest.mark.parametrize("fault", ["symbolic", "dirty", "extra-change"])
def test_git_topology_rejects_noncanonical_state(
    tmp_path: Path, valid_prelaunch: dict[str, Any], fault: str
) -> None:
    repo, authority, head = authorization_repo(tmp_path, valid_prelaunch)
    if fault == "symbolic":
        _git(repo, "checkout", "-qb", "bad")
    elif fault == "dirty":
        (repo / "untracked").write_text("x")
    else:
        _git(repo, "checkout", "-q", authority.source_commit)
        manifest = repo / "docs/pass201_pa_source_v2_prelaunch.json"
        manifest.parent.mkdir(exist_ok=True)
        manifest.write_bytes(canonical_json_bytes(dict(authority.payload)))
        (repo / "extra").write_text("x")
        _git(repo, "add", "extra", str(manifest.relative_to(repo)))
        _git(repo, "commit", "-qm", "bad")
        _git(repo, "checkout", "-q", "--detach", "HEAD")
    with pytest.raises(ValueError):
        validate_authorization_topology(repo, authority)
