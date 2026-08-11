from __future__ import annotations

import builtins
import copy
import hashlib
import json
import os
import pickle
import pickletools
import shutil
import stat
import struct
import subprocess
import sys
import types
import zipfile
from collections import OrderedDict
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import pass201_pa_source_v2_contract as contract  # noqa: E402
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
    validate_train_manifest,
)

H64 = "a" * 64
G40 = "b" * 40


def mutable_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: mutable_json(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return [mutable_json(child) for child in value]
    return value


def frozen_argv(python: str, checkpoint: str, report: str) -> list[str]:
    return [
        python,
        "-m",
        "sfora.cli",
        "image-end-to-end",
        "--dataset-name",
        "inshop",
        "--dataset-root",
        "/home/riomus/datasets/inshop_official_standard",
        "--objectives",
        "proxy_anchor",
        "--recipe",
        "auto",
        "--num-workers",
        "8",
        "--seed",
        "0",
        "--save-model-path",
        checkpoint,
        "--output",
        report,
    ]


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
            "argv": frozen_argv(
                "/usr/bin/python",
                outputs["checkpoint"]["path"],
                outputs["report"]["path"],
            ),
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


def test_package_evidence_is_version_selected_and_historical_v2_remains_valid(
    valid_prelaunch: dict[str, Any],
) -> None:
    legacy = {"bytes": 1, "sha256": H64}
    current = {
        "algorithm": "importlib-metadata-v1",
        "bytes": 123,
        "distribution_count": 2,
        "sha256": H64,
    }
    assert contract._validate_python_package_evidence(
        legacy, "pass201-pa-source-v2-prelaunch-v1", "packages"
    ) == legacy
    assert contract._validate_python_package_evidence(
        current, "pass201-pa-source-v3-prelaunch-v1", "packages"
    ) == current
    assert contract._validate_python_package_evidence(
        dict(reversed(list(legacy.items()))),
        "pass201-pa-source-v2-receipt-v1",
        "packages",
    ) == dict(reversed(list(legacy.items())))
    assert contract._validate_python_package_evidence(
        current, "pass201-pa-source-v3-receipt-v1", "packages"
    ) == current
    assert validate_prelaunch(valid_prelaunch).payload["execution"]["python_packages"] == legacy
    with pytest.raises(ValueError):
        contract._validate_python_package_evidence(
            current, "pass201-pa-source-v2-prelaunch-v1", "packages"
        )
    with pytest.raises(ValueError):
        contract._validate_python_package_evidence(
            legacy, "pass201-pa-source-v3-prelaunch-v1", "packages"
        )
    with pytest.raises(ValueError, match="unknown"):
        contract._validate_python_package_evidence(
            current, "pass201-pa-source-v4-prelaunch-v1", "packages"
        )


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("algorithm", "pip-freeze-v1"),
        ("distribution_count", True),
        ("distribution_count", 0),
        ("bytes", True),
        ("bytes", 0),
        ("sha256", "A" * 64),
    ),
)
def test_source_v3_package_evidence_rejects_fixed_value_type_and_range_drift(
    field: str, replacement: object
) -> None:
    evidence: dict[str, object] = {
        "algorithm": "importlib-metadata-v1",
        "bytes": 123,
        "distribution_count": 2,
        "sha256": H64,
    }
    evidence[field] = replacement
    with pytest.raises(ValueError):
        contract._validate_python_package_evidence(
            evidence, "pass201-pa-source-v3-prelaunch-v1", "packages"
        )


@pytest.mark.parametrize("operation", ("missing", "extra", "reordered"))
def test_source_v3_package_evidence_rejects_recursive_key_drift(
    operation: str,
) -> None:
    evidence: dict[str, object] = {
        "algorithm": "importlib-metadata-v1",
        "bytes": 123,
        "distribution_count": 2,
        "sha256": H64,
    }
    if operation == "missing":
        del evidence["bytes"]
    elif operation == "extra":
        evidence["extra"] = None
    else:
        evidence = dict(reversed(list(evidence.items())))
    with pytest.raises(ValueError):
        contract._validate_python_package_evidence(
            evidence, "pass201-pa-source-v3-prelaunch-v1", "packages"
        )


def test_schema_payload_is_recursively_immutable(valid_prelaunch: dict[str, Any]) -> None:
    authority = validate_prelaunch(valid_prelaunch)
    with pytest.raises(TypeError):
        authority.payload["execution"]["schedule"]["resolved_train_steps"] = 11
    with pytest.raises(AttributeError):
        authority.payload["execution"]["argv"].append("--forbidden")


def _checkpoint_pickle(
    root_update: Mapping[str, Any] | None = None,
    *,
    persistent_id: object | None = None,
    rebuild_arguments: tuple[object, ...] | None = None,
    state_dict_state: Mapping[str, Any] | None = None,
    tensor_build_state: object | None = None,
    storage_size: int = 1,
    storage_offset: int = 0,
    tensor_size: tuple[int, ...] = (1,),
    tensor_stride: tuple[int, ...] = (1,),
) -> bytes:
    previous = {name: sys.modules.get(name) for name in ("torch", "torch._utils")}
    torch_module = types.ModuleType("torch")
    utils_module = types.ModuleType("torch._utils")

    def rebuild_tensor_v2(*args: object) -> object:
        return args

    rebuild_tensor_v2.__module__ = "torch._utils"
    rebuild_tensor_v2.__name__ = "_rebuild_tensor_v2"
    rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    storage_type = type("FloatStorage", (), {"__module__": "torch"})
    utils_module._rebuild_tensor_v2 = rebuild_tensor_v2  # type: ignore[attr-defined]
    torch_module._utils = utils_module  # type: ignore[attr-defined]
    torch_module.FloatStorage = storage_type  # type: ignore[attr-defined]
    sys.modules["torch"] = torch_module
    sys.modules["torch._utils"] = utils_module

    storage = storage_type()
    pid = (
        ("storage", storage_type, "0", "cpu", storage_size)
        if persistent_id is None
        else persistent_id
    )
    arguments = (
        (
            storage,
            storage_offset,
            tensor_size,
            tensor_stride,
            False,
            OrderedDict(),
        )
        if rebuild_arguments is None
        else (storage, 0, (1,), (1,))
        if rebuild_arguments == ("short",)
        else rebuild_arguments
    )

    class Tensor:
        def __reduce__(self) -> Any:
            reduction = (rebuild_tensor_v2, arguments)
            if tensor_build_state is None:
                return reduction
            return (*reduction, tensor_build_state)

    state_dict = OrderedDict((("weight", Tensor()),))
    # Module.state_dict() attaches exactly this OrderedDict instance state; keeping
    # the tensor fake makes the restricted-child fixture production-shaped but torch-free.
    attributes = (
        {"_metadata": OrderedDict((("", {"version": 1}),))}
        if state_dict_state is None
        else state_dict_state
    )
    for name, value in attributes.items():
        setattr(state_dict, name, value)
    root: dict[str, Any] = {
        "state_dict": state_dict,
        "arch": {
            "backbone_name": "bn_inception",
            "pretrained_weights": "bn_inception_52deb4733",
            "head_pooling": "avg_max",
            "embedding_dimensions": 512,
            "embedding_head_init": "kaiming_normal",
            "embedding_layer_norm": False,
        },
        "artifact_selection": "final_training_state",
        "training_step": 10,
        "evaluation_model_source": "student",
        "training_config": {},
    }
    if root_update:
        root.update(root_update)

    class TorchPickler(pickle.Pickler):
        def persistent_id(self, obj: object) -> object | None:
            return pid if obj is storage else None

    try:
        buffer = __import__("io").BytesIO()
        TorchPickler(buffer, protocol=2).dump(root)
        return buffer.getvalue()
    finally:
        for name, old in previous.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def _write_checkpoint_zip(
    tmp_path: Path,
    data_pickle: bytes,
    *,
    names: tuple[str, ...] = ("checkpoint/data.pkl", "checkpoint/data/0"),
) -> Path:
    path = tmp_path / "checkpoint.pt"
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(name, data_pickle if name.endswith("/data.pkl") else b"storage")
    return path


@pytest.fixture
def checkpoint_authority(valid_prelaunch: dict[str, Any]) -> Any:
    return validate_prelaunch(valid_prelaunch)


@pytest.fixture
def valid_checkpoint_zip(tmp_path: Path) -> Path:
    return _write_checkpoint_zip(tmp_path, _checkpoint_pickle())


def test_checkpoint_reader_accepts_metadata_without_opening_storage(
    valid_checkpoint_zip: Path,
    checkpoint_authority: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_pickle = _checkpoint_pickle()
    assert [opcode.name for opcode, _, _ in pickletools.genops(data_pickle)].count("BUILD") == 1
    opened: list[str] = []
    real_open = zipfile.ZipFile.open

    def recording_open(self: zipfile.ZipFile, name: Any, *args: Any, **kwargs: Any) -> Any:
        opened.append(name.filename if isinstance(name, zipfile.ZipInfo) else str(name))
        return real_open(self, name, *args, **kwargs)

    monkeypatch.setattr(zipfile.ZipFile, "open", recording_open)
    metadata = contract.read_restricted_checkpoint_metadata(
        valid_checkpoint_zip, checkpoint_authority
    )
    assert metadata.top_keys == (
        "arch",
        "artifact_selection",
        "evaluation_model_source",
        "state_dict",
        "training_config",
        "training_step",
    )
    assert metadata.state_dict_key_count == 1
    assert metadata.training_config_sha256 == hashlib.sha256(b"{}\n").hexdigest()
    assert metadata.state_dict_storage_materialized is False
    assert opened == ["checkpoint/data.pkl"]


def test_checkpoint_reader_does_not_import_torch(
    valid_checkpoint_zip: Path,
    checkpoint_authority: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def rejecting_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "torch" or name.startswith("torch."):
            raise AssertionError("torch import attempted")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", rejecting_import)
    contract.read_restricted_checkpoint_metadata(valid_checkpoint_zip, checkpoint_authority)


def test_restricted_metadata_child_imports_no_torch_or_sfora(
    tmp_path: Path,
    valid_checkpoint_zip: Path,
    checkpoint_authority: Any,
) -> None:
    guard_dir = tmp_path / "import-guard"
    guard_dir.mkdir()
    (guard_dir / "sitecustomize.py").write_text(
        """
import builtins
real_import = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == "torch" or name.startswith("torch.") or name == "sfora" or name.startswith("sfora."):
        raise RuntimeError(f"forbidden production import: {name}")
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded
""",
        encoding="utf-8",
    )
    request = contract.encode_checkpoint_metadata_request(checkpoint_authority)
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(guard_dir)

    result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "pass201_pa_source_v2_contract.py"),
            "restricted-metadata-child",
            "--checkpoint",
            str(valid_checkpoint_zip),
        ],
        cwd=repo,
        env=env,
        input=request,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr.decode()
    bound = contract.decode_checkpoint_metadata_response(
        result.stdout, checkpoint_authority, valid_checkpoint_zip
    )
    assert bound.binding == contract.bind_external_file(valid_checkpoint_zip)
    assert bound.metadata.training_step == checkpoint_authority.expected_train_steps
    assert bound.metadata.state_dict_storage_materialized is False

    binding_result = subprocess.run(
        [
            sys.executable,
            str(repo / "scripts" / "pass201_pa_source_v2_contract.py"),
            "restricted-binding-child",
            "--checkpoint",
            str(valid_checkpoint_zip),
        ],
        cwd=repo,
        env=env,
        input=contract.encode_checkpoint_binding_request(bound.binding),
        capture_output=True,
        check=False,
    )

    assert binding_result.returncode == 0, binding_result.stderr.decode()
    assert (
        contract.decode_checkpoint_binding_response(
            binding_result.stdout, bound.binding, valid_checkpoint_zip
        )
        == bound.binding
    )


@pytest.mark.parametrize(
    ("names", "message"),
    [
        (("checkpoint/data/0",), "exactly one data.pkl"),
        (("a/data.pkl", "b/data.pkl"), "exactly one data.pkl"),
        (("../escape", "checkpoint/data.pkl"), "unsafe ZIP member"),
    ],
)
def test_checkpoint_rejects_invalid_member_topology(
    tmp_path: Path,
    checkpoint_authority: Any,
    names: tuple[str, ...],
    message: str,
) -> None:
    path = _write_checkpoint_zip(tmp_path, _checkpoint_pickle(), names=names)
    with pytest.raises(ValueError, match=message):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


def test_checkpoint_rejects_duplicate_members(tmp_path: Path, checkpoint_authority: Any) -> None:
    path = tmp_path / "checkpoint.pt"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("checkpoint/data.pkl", _checkpoint_pickle())
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("checkpoint/data.pkl", _checkpoint_pickle())
    with pytest.raises(ValueError, match="duplicate ZIP member"):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


def test_checkpoint_rejects_encrypted_member_declaration(
    valid_checkpoint_zip: Path, checkpoint_authority: Any
) -> None:
    raw = bytearray(valid_checkpoint_zip.read_bytes())
    local = raw.index(b"PK\x03\x04")
    central = raw.index(b"PK\x01\x02")
    raw[local + 6 : local + 8] = (1).to_bytes(2, "little")
    raw[central + 8 : central + 10] = (1).to_bytes(2, "little")
    valid_checkpoint_zip.write_bytes(raw)
    with pytest.raises(ValueError, match="encrypted ZIP member"):
        contract.read_restricted_checkpoint_metadata(valid_checkpoint_zip, checkpoint_authority)


def test_checkpoint_rejects_member_count_declaration(
    valid_checkpoint_zip: Path,
    checkpoint_authority: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_infolist = zipfile.ZipFile.infolist

    def too_many(self: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        return real_infolist(self)[:1] * 100_001

    monkeypatch.setattr(zipfile.ZipFile, "infolist", too_many)
    with pytest.raises(ValueError, match="too many ZIP members"):
        contract.read_restricted_checkpoint_metadata(valid_checkpoint_zip, checkpoint_authority)


def test_checkpoint_rejects_oversized_archive_declaration(
    valid_checkpoint_zip: Path,
    checkpoint_authority: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_fstat = contract.os.fstat
    first = True

    def oversized(fd: int) -> Any:
        nonlocal first
        result = real_fstat(fd)
        if first:
            first = False
            values = {name: getattr(result, name) for name in dir(result) if name.startswith("st_")}
            values["st_size"] = (2 << 30) + 1
            return types.SimpleNamespace(**values)
        return result

    monkeypatch.setattr(contract.os, "fstat", oversized)
    with pytest.raises(ValueError, match="archive exceeds"):
        contract.read_restricted_checkpoint_metadata(valid_checkpoint_zip, checkpoint_authority)


def test_checkpoint_rejects_oversized_metadata_declaration(
    valid_checkpoint_zip: Path,
    checkpoint_authority: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_infolist = zipfile.ZipFile.infolist

    def oversized(self: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        result = real_infolist(self)
        next(info for info in result if info.filename.endswith("/data.pkl")).file_size = (
            64 << 20
        ) + 1
        return result

    monkeypatch.setattr(zipfile.ZipFile, "infolist", oversized)
    with pytest.raises(ValueError, match="data.pkl exceeds"):
        contract.read_restricted_checkpoint_metadata(valid_checkpoint_zip, checkpoint_authority)


def test_checkpoint_accepts_exact_100000_member_declaration(
    valid_checkpoint_zip: Path,
    checkpoint_authority: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_infolist = zipfile.ZipFile.infolist

    class DeclaredMembers(list[zipfile.ZipInfo]):
        def __len__(self) -> int:
            return 100_000

    def exact_maximum(self: zipfile.ZipFile) -> DeclaredMembers:
        return DeclaredMembers(real_infolist(self))

    monkeypatch.setattr(zipfile.ZipFile, "infolist", exact_maximum)
    metadata = contract.read_restricted_checkpoint_metadata(
        valid_checkpoint_zip, checkpoint_authority
    )
    assert metadata.state_dict_key_count == 1


def test_checkpoint_accepts_exact_2_gib_archive_declaration(
    valid_checkpoint_zip: Path,
    checkpoint_authority: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundaryReached(Exception):
        pass

    real_fstat = contract.os.fstat
    first = True

    def exact_maximum(fd: int) -> Any:
        nonlocal first
        result = real_fstat(fd)
        if first:
            first = False
            values = {name: getattr(result, name) for name in dir(result) if name.startswith("st_")}
            values["st_size"] = 2 << 30
            return types.SimpleNamespace(**values)
        return result

    def reached_zip_reader(*args: Any, **kwargs: Any) -> Any:
        raise BoundaryReached

    monkeypatch.setattr(contract.os, "fstat", exact_maximum)
    monkeypatch.setattr(contract.zipfile, "ZipFile", reached_zip_reader)
    with pytest.raises(BoundaryReached):
        contract.read_restricted_checkpoint_metadata(valid_checkpoint_zip, checkpoint_authority)


def test_checkpoint_accepts_exact_64_mib_metadata_declaration(
    valid_checkpoint_zip: Path,
    checkpoint_authority: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class BoundaryReached(Exception):
        pass

    real_infolist = zipfile.ZipFile.infolist

    def exact_maximum(self: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
        result = real_infolist(self)
        next(info for info in result if info.filename.endswith("/data.pkl")).file_size = 64 << 20
        return result

    def reached_metadata_reader(*args: Any, **kwargs: Any) -> Any:
        raise BoundaryReached

    monkeypatch.setattr(zipfile.ZipFile, "infolist", exact_maximum)
    monkeypatch.setattr(zipfile.ZipFile, "open", reached_metadata_reader)
    with pytest.raises(BoundaryReached):
        contract.read_restricted_checkpoint_metadata(valid_checkpoint_zip, checkpoint_authority)


@pytest.mark.parametrize(
    ("data_pickle", "message"),
    [
        (pickle.dumps(eval, protocol=2), "forbidden pickle global"),
        (b"\x80\x02\x82\x01.", "extension opcode"),
        (_checkpoint_pickle() + b"N.", "trailing pickle data"),
        (_checkpoint_pickle(persistent_id=("wrong",)), "persistent ID"),
        (_checkpoint_pickle(rebuild_arguments=(None,)), "tensor rebuild"),
    ],
)
def test_checkpoint_rejects_unsafe_pickle_constructs(
    tmp_path: Path,
    checkpoint_authority: Any,
    data_pickle: bytes,
    message: str,
) -> None:
    path = _write_checkpoint_zip(tmp_path, data_pickle, names=("checkpoint/data.pkl",))
    with pytest.raises(ValueError, match=message):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


@pytest.mark.parametrize(
    "state_dict_state",
    [
        {"unexpected": OrderedDict()},
        {"_metadata": {}},
        {"_metadata": OrderedDict()},
        {"_metadata": OrderedDict((("child", {"version": 1}),))},
        {"_metadata": OrderedDict(((0, {"version": 1}),))},
        {"_metadata": OrderedDict((("", {"version": True}),))},
        {"_metadata": OrderedDict((("", {"version": -1}),))},
        {"_metadata": OrderedDict((("", {"version": 1, "unexpected": None}),))},
        {
            "_metadata": OrderedDict((("", {"version": 1}),)),
            "unexpected": None,
        },
    ],
    ids=(
        "wrong-attribute",
        "plain-dict-metadata",
        "empty-metadata",
        "missing-root-metadata",
        "non-string-module-path",
        "boolean-version",
        "negative-version",
        "extra-local-metadata",
        "extra-instance-state",
    ),
)
def test_checkpoint_rejects_noncanonical_ordered_dict_build_state(
    tmp_path: Path,
    checkpoint_authority: Any,
    state_dict_state: Mapping[str, Any],
) -> None:
    data_pickle = _checkpoint_pickle(state_dict_state=state_dict_state)
    assert [opcode.name for opcode, _, _ in pickletools.genops(data_pickle)].count("BUILD") == 1
    path = _write_checkpoint_zip(tmp_path, data_pickle, names=("checkpoint/data.pkl",))
    with pytest.raises(ValueError, match="forbidden BUILD state"):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


def test_checkpoint_rejects_build_targeting_tensor_stub(
    tmp_path: Path, checkpoint_authority: Any
) -> None:
    data_pickle = _checkpoint_pickle(
        state_dict_state={},
        tensor_build_state={"storage_offset": 0},
    )
    assert [opcode.name for opcode, _, _ in pickletools.genops(data_pickle)].count("BUILD") == 1
    path = _write_checkpoint_zip(tmp_path, data_pickle, names=("checkpoint/data.pkl",))
    with pytest.raises(ValueError, match="forbidden BUILD state"):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


def test_checkpoint_rejects_metadata_build_on_non_state_dict_ordered_dict(
    tmp_path: Path, checkpoint_authority: Any
) -> None:
    arch = OrderedDict(
        (
            ("backbone_name", "bn_inception"),
            ("pretrained_weights", "bn_inception_52deb4733"),
            ("head_pooling", "avg_max"),
            ("embedding_dimensions", 512),
            ("embedding_head_init", "kaiming_normal"),
            ("embedding_layer_norm", False),
        )
    )
    arch._metadata = OrderedDict((("", {"version": 1}),))  # type: ignore[attr-defined]
    data_pickle = _checkpoint_pickle(
        {"arch": arch},
        state_dict_state={},
    )
    assert [opcode.name for opcode, _, _ in pickletools.genops(data_pickle)].count("BUILD") == 1
    path = _write_checkpoint_zip(tmp_path, data_pickle, names=("checkpoint/data.pkl",))
    with pytest.raises(ValueError, match="forbidden BUILD state"):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


def test_checkpoint_rejects_metadata_mutated_after_valid_build(
    tmp_path: Path, checkpoint_authority: Any
) -> None:
    data_pickle = _checkpoint_pickle()
    operations = list(pickletools.genops(data_pickle))
    metadata_key_index = next(
        index
        for index, (opcode, argument, _) in enumerate(operations)
        if opcode.name == "BINUNICODE" and argument == "_metadata"
    )
    metadata_reduce_index = next(
        index
        for index in range(metadata_key_index + 1, len(operations))
        if operations[index][0].name == "REDUCE"
    )
    memo_opcode, metadata_memo, _ = operations[metadata_reduce_index + 1]
    assert memo_opcode.name == "BINPUT"
    assert type(metadata_memo) is int and metadata_memo < 256
    build_end = next(
        position + 1 for opcode, _, position in operations if opcode.name == "BUILD"
    )
    mutation = b"h" + bytes((metadata_memo,)) + b"X\x05\x00\x00\x00child}s0"
    data_pickle = data_pickle[:build_end] + mutation + data_pickle[build_end:]
    path = _write_checkpoint_zip(tmp_path, data_pickle, names=("checkpoint/data.pkl",))
    with pytest.raises(ValueError, match="forbidden BUILD state"):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


@pytest.mark.parametrize(
    ("root_update", "message"),
    [
        ({"extra": None}, "checkpoint root"),
        ({"training_config": []}, "training_config"),
        ({"artifact_selection": "best"}, "artifact_selection"),
        ({"evaluation_model_source": "ema_weight_average"}, "evaluation_model_source"),
        ({"training_step": 9}, "training_step"),
        ({"arch": {}}, "arch"),
    ],
)
def test_checkpoint_rejects_unbound_metadata(
    tmp_path: Path,
    checkpoint_authority: Any,
    root_update: Mapping[str, Any],
    message: str,
) -> None:
    path = _write_checkpoint_zip(tmp_path, _checkpoint_pickle(root_update))
    with pytest.raises(ValueError, match=message):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


def test_checkpoint_rejects_recursive_training_config(
    tmp_path: Path, checkpoint_authority: Any
) -> None:
    recursive: dict[str, Any] = {}
    recursive["self"] = recursive
    path = _write_checkpoint_zip(tmp_path, _checkpoint_pickle({"training_config": recursive}))
    with pytest.raises(ValueError, match="training_config"):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


@pytest.mark.parametrize(
    ("storage_size", "storage_offset", "tensor_size", "tensor_stride"),
    [
        (1, -1, (1,), (1,)),
        (1, 0, (1,), (-1,)),
        (1, 0, (2,), (1,)),
        (sys.maxsize, sys.maxsize - 1, (sys.maxsize,), (sys.maxsize,)),
    ],
)
def test_checkpoint_rejects_tensor_storage_span_outside_bound_storage(
    tmp_path: Path,
    checkpoint_authority: Any,
    storage_size: int,
    storage_offset: int,
    tensor_size: tuple[int, ...],
    tensor_stride: tuple[int, ...],
) -> None:
    path = _write_checkpoint_zip(
        tmp_path,
        _checkpoint_pickle(
            storage_size=storage_size,
            storage_offset=storage_offset,
            tensor_size=tensor_size,
            tensor_stride=tensor_stride,
        ),
    )
    with pytest.raises(ValueError, match="tensor rebuild"):
        contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)


@pytest.mark.parametrize(
    ("storage_size", "storage_offset", "tensor_size", "tensor_stride"),
    [
        (0, 0, (0,), (1,)),
        (3, 1, (2,), (1,)),
    ],
)
def test_checkpoint_accepts_zero_size_and_exact_boundary_tensor_spans(
    tmp_path: Path,
    checkpoint_authority: Any,
    storage_size: int,
    storage_offset: int,
    tensor_size: tuple[int, ...],
    tensor_stride: tuple[int, ...],
) -> None:
    path = _write_checkpoint_zip(
        tmp_path,
        _checkpoint_pickle(
            storage_size=storage_size,
            storage_offset=storage_offset,
            tensor_size=tensor_size,
            tensor_stride=tensor_stride,
        ),
    )
    metadata = contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)
    assert metadata.state_dict_key_count == 1


def test_checkpoint_rejects_pickle_build_that_poisoned_shared_rebuild_stub(
    tmp_path: Path, checkpoint_authority: Any
) -> None:
    valid = _checkpoint_pickle(rebuild_arguments=("short",), state_dict_state={})
    poison = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n}X\x07\x00\x00\x00versionK\x01sb0"
    path = _write_checkpoint_zip(
        tmp_path,
        poison + valid[2:],
        names=("checkpoint/data.pkl",),
    )
    rebuild = contract._REBUILD_STUBS["torch._utils._rebuild_tensor_v2"]
    try:
        with pytest.raises(ValueError, match="pickle opcode"):
            contract.read_restricted_checkpoint_metadata(path, checkpoint_authority)
        assert rebuild.version == 2
    finally:
        rebuild.version = 2


def test_checkpoint_rejects_symlink_input(
    valid_checkpoint_zip: Path, checkpoint_authority: Any
) -> None:
    link = valid_checkpoint_zip.with_name("link.pt")
    link.symlink_to(valid_checkpoint_zip)
    with pytest.raises(ValueError, match="checkpoint"):
        contract.read_restricted_checkpoint_metadata(link, checkpoint_authority)


def test_publication_is_exclusive_fsynced_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    evidence = contract.publish_new_file(path, b"bound\n")
    assert path.read_bytes() == b"bound\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert evidence.path == PurePosixPath(path.as_posix())
    assert evidence.mode == path.stat().st_mode
    assert evidence.byte_count == 6
    assert evidence.sha256 == hashlib.sha256(b"bound\n").hexdigest()
    assert sorted(child.name for child in tmp_path.iterdir()) == ["result.json"]


@pytest.mark.parametrize("kind", ["regular", "symlink"])
def test_publication_rejects_preexisting_destination(tmp_path: Path, kind: str) -> None:
    path = tmp_path / "result"
    target = tmp_path / "target"
    target.write_bytes(b"target")
    if kind == "regular":
        path.write_bytes(b"old")
    else:
        path.symlink_to(target)
    with pytest.raises(ValueError, match="publish"):
        contract.publish_new_file(path, b"new")
    assert (target if kind == "symlink" else path).read_bytes() == (
        b"target" if kind == "symlink" else b"old"
    )


def test_publication_rejects_symlink_parent(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    link_parent = tmp_path / "link"
    link_parent.symlink_to(real_parent, target_is_directory=True)
    with pytest.raises(ValueError, match="directory"):
        contract.publish_new_file(link_parent / "result", b"new")
    assert not (real_parent / "result").exists()


@pytest.mark.parametrize("fault", ["write", "file_fsync", "link", "directory_fsync"])
def test_publication_failure_never_leaves_accepted_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    path = tmp_path / "result"
    real_fsync = contract.os.fsync

    if fault == "write":

        def fail_write(*args: Any, **kwargs: Any) -> int:
            raise OSError("write")

        monkeypatch.setattr(contract.os, "write", fail_write)
    elif fault == "file_fsync":

        def fail_file_fsync(fd: int) -> None:
            if stat.S_ISREG(contract.os.fstat(fd).st_mode):
                raise OSError("file fsync")
            real_fsync(fd)

        monkeypatch.setattr(contract.os, "fsync", fail_file_fsync)
    elif fault == "link":

        def fail_link(*args: Any, **kwargs: Any) -> None:
            raise OSError("link")

        monkeypatch.setattr(contract.os, "link", fail_link)
    else:

        def fail_directory_fsync(fd: int) -> None:
            if stat.S_ISDIR(contract.os.fstat(fd).st_mode):
                raise OSError("directory fsync")
            real_fsync(fd)

        monkeypatch.setattr(contract.os, "fsync", fail_directory_fsync)

    with pytest.raises(ValueError, match="publish"):
        contract.publish_new_file(path, b"new")
    assert not path.exists()
    assert not any(child.name.startswith(".result.") for child in tmp_path.iterdir())


def test_publication_loses_concurrent_creation_without_clobbering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result"
    real_link = contract.os.link

    def racing_link(*args: Any, **kwargs: Any) -> None:
        path.write_bytes(b"racer")
        real_link(*args, **kwargs)

    monkeypatch.setattr(contract.os, "link", racing_link)
    with pytest.raises(ValueError, match="publish"):
        contract.publish_new_file(path, b"ours")
    assert path.read_bytes() == b"racer"


def test_publication_rolls_back_when_post_link_stat_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result"
    real_stat = contract.os.stat
    failed = False

    def failing_stat(name: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal failed
        if name == path.name and kwargs.get("dir_fd") is not None and not failed:
            failed = True
            raise OSError("post-link stat")
        return real_stat(name, *args, **kwargs)

    monkeypatch.setattr(contract.os, "stat", failing_stat)
    with pytest.raises(ValueError, match="publish"):
        contract.publish_new_file(path, b"ours")
    assert not path.exists()


def test_publication_rollback_preserves_replacement_between_check_and_removal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result"
    real_stat = contract.os.stat
    real_rename = contract.os.rename
    real_fsync = contract.os.fsync
    destination_stats = 0
    replaced = False
    directory_fsync_failed = False

    def install_racer() -> None:
        nonlocal replaced
        if not replaced:
            path.unlink()
            path.write_bytes(b"racer")
            replaced = True

    def synchronizing_stat(name: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal destination_stats
        result = real_stat(name, *args, **kwargs)
        if name == path.name and kwargs.get("dir_fd") is not None:
            destination_stats += 1
            if destination_stats == 2:
                install_racer()
        return result

    def synchronizing_rename(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        if source == path.name:
            install_racer()
        real_rename(source, destination, *args, **kwargs)

    def failing_first_directory_fsync(fd: int) -> None:
        nonlocal directory_fsync_failed
        if stat.S_ISDIR(contract.os.fstat(fd).st_mode) and not directory_fsync_failed:
            directory_fsync_failed = True
            raise OSError("directory fsync")
        real_fsync(fd)

    monkeypatch.setattr(contract.os, "stat", synchronizing_stat)
    monkeypatch.setattr(contract.os, "rename", synchronizing_rename)
    monkeypatch.setattr(contract.os, "fsync", failing_first_directory_fsync)
    with pytest.raises(ValueError, match="publish"):
        contract.publish_new_file(path, b"ours")
    assert replaced
    assert path.read_bytes() == b"racer"


@pytest.mark.parametrize("fault", ["quarantine_stat", "restore_link"])
def test_publication_rollback_preserves_racer_when_quarantine_operation_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    path = tmp_path / "result"
    path.write_bytes(b"ours")
    os.link(path, tmp_path / "held-original")
    expected = path.stat()
    real_rename = contract.os.rename
    real_stat = contract.os.stat
    real_link = contract.os.link

    def racing_rename(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        if source == path.name:
            path.unlink()
            path.write_bytes(b"racer")
        real_rename(source, destination, *args, **kwargs)

    def failing_stat(name: Any, *args: Any, **kwargs: Any) -> Any:
        if fault == "quarantine_stat" and str(name).endswith(".rollback"):
            raise OSError("quarantine stat")
        return real_stat(name, *args, **kwargs)

    def failing_link(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        if fault == "restore_link" and str(source).endswith(".rollback"):
            raise OSError("restore link")
        real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(contract.os, "rename", racing_rename)
    monkeypatch.setattr(contract.os, "stat", failing_stat)
    monkeypatch.setattr(contract.os, "link", failing_link)
    directory_fd = os.open(tmp_path, os.O_RDONLY)
    try:
        contract._unlink_if_same(directory_fd, path.name, expected)
    finally:
        os.close(directory_fd)
    if fault == "quarantine_stat":
        assert path.read_bytes() == b"racer"
    else:
        assert not path.exists()
        quarantined = list(tmp_path.glob(".result.*.rollback"))
        assert len(quarantined) == 1
        assert quarantined[0].read_bytes() == b"racer"


def test_publication_rollback_never_overwrites_newer_creator_after_restore_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result"
    real_rename = contract.os.rename
    real_link = contract.os.link
    real_fsync = contract.os.fsync
    directory_fsync_failed = False

    def racing_rename(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        if source == path.name:
            path.unlink()
            path.write_bytes(b"racer")
        real_rename(source, destination, *args, **kwargs)

    def occupied_link(source: Any, destination: Any, *args: Any, **kwargs: Any) -> None:
        if str(source).endswith(".rollback"):
            path.write_bytes(b"newer")
            raise OSError("non-clobber restore failure")
        real_link(source, destination, *args, **kwargs)

    def failing_first_directory_fsync(fd: int) -> None:
        nonlocal directory_fsync_failed
        if stat.S_ISDIR(contract.os.fstat(fd).st_mode) and not directory_fsync_failed:
            directory_fsync_failed = True
            raise OSError("directory fsync")
        real_fsync(fd)

    monkeypatch.setattr(contract.os, "rename", racing_rename)
    monkeypatch.setattr(contract.os, "link", occupied_link)
    monkeypatch.setattr(contract.os, "fsync", failing_first_directory_fsync)
    with pytest.raises(ValueError, match="publish"):
        contract.publish_new_file(path, b"ours")
    assert path.read_bytes() == b"newer"
    quarantined = list(tmp_path.glob(".result.*.rollback"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"racer"


def test_publication_does_not_chmod_replacement_after_preliminary_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result"
    real_hash = contract._hash_regular_at
    swapped = False

    def swapping_hash(*args: Any, **kwargs: Any) -> Any:
        nonlocal swapped
        result = real_hash(*args, **kwargs)
        expected = args[3] if len(args) > 3 else kwargs.get("expected")
        if expected is not None and not swapped:
            swapped = True
            path.unlink()
            path.write_bytes(b"racer")
            path.chmod(0o600)
        return result

    monkeypatch.setattr(contract, "_hash_regular_at", swapping_hash)
    with pytest.raises(ValueError, match="publish"):
        contract.publish_new_file(path, b"ours")
    assert path.read_bytes() == b"racer"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_publication_rejects_parent_swap_before_final_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    parent.mkdir()
    path = parent / "result"
    parked = tmp_path / "parked"
    real_link = contract.os.link

    def swapping_link(*args: Any, **kwargs: Any) -> None:
        real_link(*args, **kwargs)
        parent.rename(parked)
        parent.mkdir()
        (parent / "result").write_bytes(b"ours")
        (parent / "result").chmod(0o444)

    monkeypatch.setattr(contract.os, "link", swapping_link)
    with pytest.raises(ValueError, match="publish"):
        contract.publish_new_file(path, b"ours")


def test_output_evidence_rejects_mutation_without_changing_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result"
    path.write_bytes(b"old")
    path.chmod(0o600)
    real_read = contract.os.read
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
        contract.hash_open_regular(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_output_evidence_makes_successfully_hashed_file_read_only(tmp_path: Path) -> None:
    path = tmp_path / "result"
    path.write_bytes(b"bound")
    path.chmod(0o600)
    evidence = contract.hash_open_regular(path)
    assert evidence.sha256 == hashlib.sha256(b"bound").hexdigest()
    assert stat.S_IMODE(evidence.mode) == 0o444
    assert stat.S_IMODE(path.stat().st_mode) == 0o444


def test_output_evidence_rejects_same_size_mtime_restored_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result"
    path.write_bytes(b"old")
    path.chmod(0o600)
    original = path.stat()
    real_read = contract.os.read
    changed = False

    def mutating_read(fd: int, size: int) -> bytes:
        nonlocal changed
        data = real_read(fd, size)
        if data and not changed:
            changed = True
            path.write_bytes(b"new")
            os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
        return data

    monkeypatch.setattr(contract.os, "read", mutating_read)
    with pytest.raises(ValueError, match="changed"):
        contract.hash_open_regular(path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_publication_mutation_is_not_accepted_or_made_read_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "result"
    real_read = contract.os.read
    changed = False

    def mutating_read(fd: int, size: int) -> bytes:
        nonlocal changed
        data = real_read(fd, size)
        if data and not changed:
            changed = True
            path.write_bytes(b"attacker")
        return data

    monkeypatch.setattr(contract.os, "read", mutating_read)
    with pytest.raises(ValueError, match="publish"):
        contract.publish_new_file(path, b"ours")
    assert not path.exists()


@pytest.mark.parametrize("index", range(20))
def test_schema_frozen_argv_rejects_every_element_substitution(
    valid_prelaunch: dict[str, Any], index: int
) -> None:
    valid_prelaunch["execution"]["argv"][index] += "-forbidden"
    with pytest.raises(ValueError, match="argv"):
        validate_prelaunch(valid_prelaunch)


def test_schema_frozen_argv_rejects_added_argument(valid_prelaunch: dict[str, Any]) -> None:
    valid_prelaunch["execution"]["argv"].append("--forbidden")
    with pytest.raises(ValueError, match="argv"):
        validate_prelaunch(valid_prelaunch)


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


def _swap_queued_directory_on_second_scan(
    monkeypatch: pytest.MonkeyPatch, root: Path, outside: Path
) -> None:
    import pass201_pa_source_v2_contract as contract

    real_scandir = os.scandir
    calls = 0

    def swapping_scandir(path: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            child = root / "child"
            child.rename(root / "parked")
            child.symlink_to(outside, target_is_directory=True)
        return real_scandir(path)

    monkeypatch.setattr(contract.os, "scandir", swapping_scandir)


def test_ordinary_merkle_rejects_queued_directory_swap_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    (root / "child").mkdir(parents=True)
    outside.mkdir()
    (root / "child" / "inside").write_bytes(b"inside")
    (outside / "secret").write_bytes(b"secret")
    _swap_queued_directory_on_second_scan(monkeypatch, root, outside)
    with pytest.raises(ValueError, match="changed|symlink"):
        bind_merkle(root)


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


def test_import_root_rejects_queued_directory_swap_escape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    root = tmp_path / "root"
    outside = tmp_path / "outside"
    (root / "child").mkdir(parents=True)
    outside.mkdir()
    (root / "child" / "inside.py").write_bytes(b"inside")
    (outside / "secret.py").write_bytes(b"secret")
    _swap_queued_directory_on_second_scan(monkeypatch, root, outside)
    with pytest.raises(ValueError, match="changed|symlink"):
        bind_import_roots(
            _path_emitter(tmp_path), {"BOUND_PATHS": json.dumps([str(root)])}, checkout
        )


def output(path: str) -> dict[str, Any]:
    return {"path": path, "file_type": "regular", "mode": 0o100644, "bytes": 1, "sha256": H64}


def valid_receipt(authority: Any) -> dict[str, Any]:
    manifest = canonical_json_bytes(mutable_json(authority.payload))
    manifest_blob = hashlib.sha1(
        b"blob " + str(len(manifest)).encode("ascii") + b"\0" + manifest
    ).hexdigest()
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
            "manifest_bytes": len(manifest),
            "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            "manifest_git_blob": manifest_blob,
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
            "environment": mutable_json(authority.payload["execution"]["environment"]),
            "argv": mutable_json(authority.payload["execution"]["argv"]),
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
                k: (
                    authority.expected_config_sha256
                    if k == "expected_config"
                    else hashlib.sha256(manifest).hexdigest()
                    if k == "manifest"
                    else H64
                )
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
    with pytest.raises(TypeError):
        result.payload["postflight"]["source_equal"] = False
    with pytest.raises(AttributeError):
        result.payload["sidecar_derivation"]["child_processes"].append({})


@pytest.mark.parametrize(
    ("field", "replacement"),
    [("manifest_bytes", 1), ("manifest_sha256", H64), ("manifest_git_blob", G40)],
)
def test_receipt_manifest_identity_rejects_unrelated_value(
    valid_prelaunch: dict[str, Any], field: str, replacement: object
) -> None:
    authority = validate_prelaunch(valid_prelaunch)
    receipt = valid_receipt(authority)
    assert receipt["authorization"][field] != replacement
    receipt["authorization"][field] = replacement
    with pytest.raises(ValueError, match=field):
        validate_complete_receipt(receipt, authority)


def _valid_train_manifest(authority: Any) -> dict[str, Any]:
    return {
        "schema_version": "pass201-train-manifest-v1",
        "algorithm_id": "pass201-inshop-benchmark-row-suffix-v2",
        "source_commit": authority.source_commit,
        "dataset_authority": {
            "root": authority.payload["dataset"]["root"],
            "partition_sha256": authority.payload["dataset"]["partition"]["sha256"],
            "resolved_image_root": authority.payload["dataset"]["resolved_image_root"],
            "image_tree_sha256": authority.payload["dataset"]["image_tree"]["root_sha256"],
            "bundle": mutable_json(authority.payload["dataset"]["bundle"]),
            "selection_policy": "full_official_partition",
        },
        "rows": [{"sample_index": 0, "example_id": "inshop-train-img/a.jpg", "label": 7}],
        "derivation": {
            "call_graph": [
                "sfora.cli._load_cli_image_retrieval_bundle",
                "sfora.image_recipes.resolve_recipe",
                "sfora.image_recipes.config_for_recipe",
                "sfora.image_recipes.mark_recipe_config_modified",
                "sfora.image_end_to_end._checkpoint_train_validation_split",
                "sfora.image_end_to_end._apply_training_label_noise",
                "sfora.image_end_to_end._resolve_training_schedule",
            ],
            "source_files": mutable_json(authority.payload["source"]["files"]),
            "resolved_config_sha256": authority.expected_config_sha256,
            "row_count": 1,
            "identity_count": 1,
            "ordered_row_sha256": authority.payload["dataset"]["optimization_authority"][
                "ordered_row_sha256"
            ],
            "resolved_membership_count": 1,
            "resolved_membership_sha256": authority.payload["dataset"][
                "optimization_authority"
            ]["resolved_membership_sha256"],
        },
    }


def _bind_valid_manifest_row_hash(valid_prelaunch: dict[str, Any]) -> None:
    row = {"sample_index": 0, "example_id": "inshop-train-img/a.jpg", "label": 7}
    encoded = canonical_json_bytes(row)
    valid_prelaunch["dataset"]["optimization_authority"]["ordered_row_sha256"] = (
        hashlib.sha256(struct.pack(">Q", len(encoded)) + encoded).hexdigest()
    )


def test_train_manifest_accepts_only_exact_nested_schema(valid_prelaunch: dict[str, Any]) -> None:
    _bind_valid_manifest_row_hash(valid_prelaunch)
    authority = validate_prelaunch(valid_prelaunch)
    payload = _valid_train_manifest(authority)

    validate_train_manifest(payload, authority)

    mutations = []
    for path, key in (
        ((), "extra"),
        (("dataset_authority",), "extra"),
        (("dataset_authority", "bundle"), "extra"),
        (("rows", 0), "extra"),
        (("derivation",), "extra"),
        (("derivation", "source_files", 0), "extra"),
    ):
        mutated = copy.deepcopy(payload)
        target: Any = mutated
        for part in path:
            target = target[part]
        target[key] = 1
        mutations.append(mutated)
    for mutated in mutations:
        with pytest.raises(ValueError):
            validate_train_manifest(mutated, authority)


@pytest.mark.parametrize("replacement", [True, 0.0, "0"])
def test_train_manifest_rejects_non_integer_row_fields(
    valid_prelaunch: dict[str, Any], replacement: object
) -> None:
    _bind_valid_manifest_row_hash(valid_prelaunch)
    authority = validate_prelaunch(valid_prelaunch)
    payload = _valid_train_manifest(authority)
    payload["rows"][0]["sample_index"] = replacement
    with pytest.raises(ValueError, match="sample_index"):
        validate_train_manifest(payload, authority)


def test_train_manifest_rejects_noncontiguous_rows_and_source_order(
    valid_prelaunch: dict[str, Any],
) -> None:
    _bind_valid_manifest_row_hash(valid_prelaunch)
    valid_prelaunch["source"]["files"] = [
        file_binding("scripts/z.py"),
        file_binding("scripts/a.py"),
    ]
    with pytest.raises(ValueError, match="source.files order"):
        validate_prelaunch(valid_prelaunch)

    valid_prelaunch["source"]["files"] = [
        file_binding("scripts/a.py"),
        file_binding("scripts/z.py"),
    ]
    authority = validate_prelaunch(valid_prelaunch)
    payload = _valid_train_manifest(authority)
    payload["rows"][0]["sample_index"] = 1
    with pytest.raises(ValueError, match="sample_index"):
        validate_train_manifest(payload, authority)


MANIFEST_OBJECT_KEYS = {
    (): (
        "schema_version",
        "algorithm_id",
        "source_commit",
        "dataset_authority",
        "rows",
        "derivation",
    ),
    ("dataset_authority",): (
        "root",
        "partition_sha256",
        "resolved_image_root",
        "image_tree_sha256",
        "bundle",
        "selection_policy",
    ),
    ("dataset_authority", "bundle"): (
        "train",
        "query",
        "gallery",
        "protocol",
        "protocol_name",
    ),
    ("rows", 0): ("sample_index", "example_id", "label"),
    ("derivation",): (
        "call_graph",
        "source_files",
        "resolved_config_sha256",
        "row_count",
        "identity_count",
        "ordered_row_sha256",
        "resolved_membership_count",
        "resolved_membership_sha256",
    ),
    ("derivation", "source_files", 0): (
        "path",
        "git_mode",
        "bytes",
        "sha256",
        "git_blob",
    ),
}
MANIFEST_MISSING_PATHS = tuple(
    prefix + (key,) for prefix, keys in MANIFEST_OBJECT_KEYS.items() for key in keys
)
MANIFEST_WRONG_TYPES = (
    (("schema_version",), 1),
    (("algorithm_id",), 1),
    (("source_commit",), 1),
    (("dataset_authority",), []),
    (("rows",), {}),
    (("derivation",), []),
    (("dataset_authority", "root"), 1),
    (("dataset_authority", "partition_sha256"), 1),
    (("dataset_authority", "resolved_image_root"), 1),
    (("dataset_authority", "image_tree_sha256"), 1),
    (("dataset_authority", "bundle"), []),
    (("dataset_authority", "selection_policy"), 1),
    (("dataset_authority", "bundle", "train"), True),
    (("dataset_authority", "bundle", "query"), 1.0),
    (("dataset_authority", "bundle", "gallery"), "1"),
    (("dataset_authority", "bundle", "protocol"), 1),
    (("dataset_authority", "bundle", "protocol_name"), 1),
    (("rows", 0), []),
    (("rows", 0, "sample_index"), True),
    (("rows", 0, "example_id"), 1),
    (("rows", 0, "label"), 7.0),
    (("derivation", "call_graph"), {}),
    (("derivation", "call_graph", 0), 1),
    (("derivation", "source_files"), {}),
    (("derivation", "resolved_config_sha256"), 1),
    (("derivation", "row_count"), True),
    (("derivation", "identity_count"), 1.0),
    (("derivation", "ordered_row_sha256"), 1),
    (("derivation", "resolved_membership_count"), False),
    (("derivation", "resolved_membership_sha256"), 1),
    (("derivation", "source_files", 0), []),
    (("derivation", "source_files", 0, "path"), 1),
    (("derivation", "source_files", 0, "git_mode"), 1),
    (("derivation", "source_files", 0, "bytes"), True),
    (("derivation", "source_files", 0, "sha256"), 1),
    (("derivation", "source_files", 0, "git_blob"), 1),
)


def _mutate_manifest_path(payload: dict[str, Any], path: tuple[object, ...], value: Any) -> None:
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    target[path[-1]] = value


@pytest.mark.parametrize("path", MANIFEST_MISSING_PATHS)
def test_train_manifest_rejects_every_nested_missing_key(
    valid_prelaunch: dict[str, Any], path: tuple[object, ...]
) -> None:
    _bind_valid_manifest_row_hash(valid_prelaunch)
    authority = validate_prelaunch(valid_prelaunch)
    payload = _valid_train_manifest(authority)
    target: Any = payload
    for part in path[:-1]:
        target = target[part]
    del target[path[-1]]
    with pytest.raises(ValueError):
        validate_train_manifest(payload, authority)


@pytest.mark.parametrize(("path", "replacement"), MANIFEST_WRONG_TYPES)
def test_train_manifest_rejects_every_nested_wrong_type(
    valid_prelaunch: dict[str, Any], path: tuple[object, ...], replacement: object
) -> None:
    _bind_valid_manifest_row_hash(valid_prelaunch)
    authority = validate_prelaunch(valid_prelaunch)
    payload = _valid_train_manifest(authority)
    _mutate_manifest_path(payload, path, replacement)
    with pytest.raises(ValueError):
        validate_train_manifest(payload, authority)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("embedding_dimensions", 512.0),
        ("embedding_dimensions", True),
        ("embedding_layer_norm", 0),
    ],
)
def test_receipt_checkpoint_arch_rejects_equal_cross_type_value(
    valid_prelaunch: dict[str, Any], field: str, replacement: object
) -> None:
    authority = validate_prelaunch(valid_prelaunch)
    receipt = valid_receipt(authority)
    receipt["checkpoint_metadata"]["arch"][field] = replacement
    with pytest.raises(ValueError, match=field):
        validate_complete_receipt(receipt, authority)


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
        manifest.write_bytes(canonical_json_bytes(mutable_json(authority.payload)))
        (repo / "extra").write_text("x")
        _git(repo, "add", "extra", str(manifest.relative_to(repo)))
        _git(repo, "commit", "-qm", "bad")
        _git(repo, "checkout", "-q", "--detach", "HEAD")
    with pytest.raises(ValueError):
        validate_authorization_topology(repo, authority)


def test_git_topology_uses_one_bound_git_despite_path_substitution(
    tmp_path: Path,
    valid_prelaunch: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import pass201_pa_source_v2_contract as contract

    repo, authority, head = authorization_repo(tmp_path, valid_prelaunch)
    real_git = shutil.which("git")
    assert real_git is not None
    marker = tmp_path / "malicious-ran"
    malicious = tmp_path / "git"
    malicious.write_text(f"#!/bin/sh\n: > {marker}\nexit 1\n")
    malicious.chmod(0o755)
    calls = 0

    def substituting_which(name: str) -> str:
        nonlocal calls
        assert name == "git"
        calls += 1
        return real_git if calls == 1 else str(malicious)

    monkeypatch.setattr(contract.shutil, "which", substituting_which)
    assert validate_authorization_topology(repo, authority) == head
    assert calls == 1
    assert not marker.exists()
