from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import py_compile
import struct
import subprocess
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import torch
from torch import nn

from sfora.nested_rank_protocol import ordered_training_records_sha256


def _load_subject() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "sop_teacher_anchored_runtime.py"
    spec = importlib.util.spec_from_file_location("sop_teacher_anchored_runtime", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SUBJECT = _load_subject()
_AUTHENTICATED_UNICOM_MODULE = "_sfora_authenticated_unicom"


def test_registered_snapshot_metadata_matches_the_sealed_sop_authority() -> None:
    source = SUBJECT._REGISTERED_SNAPSHOT_METADATA["source"]
    teacher = SUBJECT._REGISTERED_SNAPSHOT_METADATA["teacher"]

    assert source["source_archive_sha256"] == (
        "6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818"
    )
    assert teacher["source_archive_sha256"] == (
        "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a"
    )
    assert source["train_array_sha256"]["train_embeddings"] == (
        "baca47e3349b4d8cd2f3d52a85d2692fe8c539347daa54a49a351adef0a5f9df"
    )
    assert source["batch_size"] == 64
    assert source["runtime"] == {
        "blas_threads": 2,
        "cpu_threads": 2,
        "cublas_workspace_config": ":4096:8",
        "cuda_matmul_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cudnn_sdp_enabled": False,
        "cudnn_tf32": False,
        "cudnn_version": 92000,
        "cuda_device_capability": "12.1",
        "cuda_device_name": "NVIDIA GB10",
        "cuda_version": "13.0",
        "deterministic_algorithms": True,
        "flash_sdp_enabled": False,
        "float32_matmul_precision": "highest",
        "math_sdp_enabled": True,
        "memory_efficient_sdp_enabled": False,
        "seed": 17,
        "torch_version": "2.12.1+cu130",
    }
    assert teacher["train_array_sha256"]["train_embeddings"] == (
        "5a8629deee1adff92ac941a0f78f4fd55f1d4cb0db43b1459a0d6c92f84b46fc"
    )
    assert source["train_array_sha256"]["train_labels"] == (
        "d785d2eca417d91257195bf2c16d87b7805178984db82c0d0a0f5ca8546a216f"
    )
    assert (
        teacher["train_array_sha256"]["train_labels"]
        == source["train_array_sha256"]["train_labels"]
    )
    assert (
        source["model_revision"]
        == teacher["model_revision"]
        == ("d71992ed969e6c271436ac0a0ee1f3ca61474ac0")
    )


def _snapshot(
    path: Path,
    *,
    model: str,
    embeddings: np.ndarray,
    labels: np.ndarray,
    image_ids: np.ndarray,
    relative_paths: np.ndarray,
) -> str:
    arrays = {
        "train_embeddings": np.ascontiguousarray(embeddings, dtype=np.float32),
        "train_labels": np.ascontiguousarray(labels, dtype=np.int64),
        "train_image_ids": np.ascontiguousarray(image_ids, dtype=np.int64),
        "train_relative_paths": np.ascontiguousarray(relative_paths),
    }
    array_sha = {
        name: hashlib.sha256(value.tobytes(order="C")).hexdigest() for name, value in arrays.items()
    }
    metadata = {
        "schema": "sfora-nnrl-sop-train-snapshot-v1",
        "source_archive_sha256": "1" * 64 if model.endswith("B/16") else "2" * 64,
        "model_identifier": model,
        "model_revision": "3" * 40,
        "checkpoint_sha256": "4" * 64 if model.endswith("B/16") else "5" * 64,
        "embedding_dimension": embeddings.shape[1],
        "train_rows": len(labels),
        "train_classes": len(set(labels.tolist())),
        "train_array_sha256": array_sha,
        "excluded_test_array_sha256": {
            "test_embeddings": "6" * 64 if model.endswith("B/16") else "7" * 64,
            "test_labels": "8" * 64,
            "test_image_ids": "9" * 64,
            "test_relative_paths": "a" * 64,
        },
        "ordered_train_record_sha256": ordered_training_records_sha256(
            arrays["train_image_ids"],
            arrays["train_labels"],
            tuple(str(value) for value in arrays["train_relative_paths"]),
        ),
        "transform": "fixture-transform",
    }
    if model.endswith("B/16"):
        metadata["batch_size"] = 64
        metadata["runtime"] = SUBJECT._REGISTERED_SNAPSHOT_METADATA["source"]["runtime"]
    np.savez(
        path,
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        **arrays,
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_snapshots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch | None = None
) -> tuple[Path, str, Path, str]:
    labels = np.asarray([1, 1, 2, 2], dtype=np.int64)
    image_ids = np.asarray([11, 12, 21, 22], dtype=np.int64)
    paths = np.asarray(["train/a.jpg", "train/b.jpg", "train/c.jpg", "train/d.jpg"])
    source = tmp_path / "source.npz"
    teacher = tmp_path / "teacher.npz"
    source_sha = _snapshot(
        source,
        model="UNICOM-ViT-B/16",
        embeddings=np.arange(4 * 768, dtype=np.float32).reshape(4, 768) + 1.0,
        labels=labels,
        image_ids=image_ids,
        relative_paths=paths,
    )
    teacher_sha = _snapshot(
        teacher,
        model="UNICOM-ViT-L/14@336px",
        embeddings=np.arange(4 * 768, dtype=np.float32).reshape(4, 768) + 2.0,
        labels=labels,
        image_ids=image_ids,
        relative_paths=paths,
    )
    if monkeypatch is not None:
        registered = {}
        for role, path in (("source", source), ("teacher", teacher)):
            with np.load(path, allow_pickle=False) as archive:
                metadata = json.loads(str(archive["metadata_json"].item()))
            keys = (
                "checkpoint_sha256",
                "excluded_test_array_sha256",
                "model_identifier",
                "model_revision",
                "source_archive_sha256",
                "train_array_sha256",
            ) + (("batch_size", "runtime") if role == "source" else ())
            registered[role] = {key: metadata[key] for key in keys}
        monkeypatch.setattr(SUBJECT, "_REGISTERED_SNAPSHOT_METADATA", registered)
    return source, source_sha, teacher, teacher_sha


def test_load_pair_authenticates_train_only_roles_and_alignment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)

    pair = SUBJECT.load_authenticated_train_pair(
        source,
        source_sha,
        teacher,
        teacher_sha,
    )

    assert pair.labels.tolist() == [1, 1, 2, 2]
    assert pair.image_ids.tolist() == [11, 12, 21, 22]
    assert pair.relative_paths == (
        "train/a.jpg",
        "train/b.jpg",
        "train/c.jpg",
        "train/d.jpg",
    )
    assert pair.source_embeddings.shape == (4, 768)
    assert pair.teacher_embeddings.shape == (4, 768)


def test_load_pair_rejects_self_consistent_but_unregistered_arrays(tmp_path: Path) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path)

    with pytest.raises(ValueError, match="snapshot pair"):
        SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)


def test_load_pair_rejects_digest_role_and_row_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="snapshot pair"):
        SUBJECT.load_authenticated_train_pair(source, "0" * 64, teacher, teacher_sha)

    labels = np.asarray([1, 1, 2, 2], dtype=np.int64)
    image_ids = np.asarray([12, 11, 21, 22], dtype=np.int64)
    paths = np.asarray(["train/a.jpg", "train/b.jpg", "train/c.jpg", "train/d.jpg"])
    teacher_sha = _snapshot(
        teacher,
        model="UNICOM-ViT-L/14@336px",
        embeddings=np.arange(4 * 768, dtype=np.float32).reshape(4, 768) + 2.0,
        labels=labels,
        image_ids=image_ids,
        relative_paths=paths,
    )
    with pytest.raises(ValueError, match="snapshot pair"):
        SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)


def _image_digest(root: Path, relative_paths: tuple[str, ...]) -> str:
    digest = hashlib.sha256(b"sfora-teacher-anchored-image-tree-v1\x00")
    for relative in relative_paths:
        payload = (root / relative).read_bytes()
        encoded = relative.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(hashlib.sha256(payload).digest())
    return digest.hexdigest()


def test_image_binding_reads_only_ordered_train_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    pair = SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)
    root = tmp_path / "images"
    for index, relative in enumerate(pair.relative_paths):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"image-{index}".encode())
    forbidden = root / "test" / "never-open.jpg"
    forbidden.parent.mkdir(parents=True)
    forbidden.write_bytes(b"official-test")
    expected = _image_digest(root, pair.relative_paths)
    heartbeats: list[int] = []

    bound = SUBJECT.bind_authenticated_train_images(
        pair, root, expected, heartbeat=lambda: heartbeats.append(1)
    )

    assert bound.relative_paths == pair.relative_paths
    assert bound.image_paths == tuple(root / value for value in pair.relative_paths)
    assert bound.sha256 == expected
    assert heartbeats == [1]


def test_image_binding_rejects_path_escape_symlink_and_digest_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    pair = SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)
    root = tmp_path / "images"
    for relative in pair.relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"image")
    expected = _image_digest(root, pair.relative_paths)
    (root / pair.relative_paths[0]).unlink()
    (root / pair.relative_paths[0]).symlink_to(root / pair.relative_paths[1])

    with pytest.raises(ValueError, match="image manifest"):
        SUBJECT.bind_authenticated_train_images(pair, root, expected)
    with pytest.raises(ValueError, match="image manifest"):
        SUBJECT.bind_authenticated_train_images(pair, root, "0" * 64)


def test_image_binding_rejects_symlinked_parent_before_reading_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    pair = SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)
    root = tmp_path / "images"
    target = root / "test"
    target.mkdir(parents=True)
    for relative in pair.relative_paths:
        (target / Path(relative).name).write_bytes(b"official-test")
    (root / "train").symlink_to(target, target_is_directory=True)
    expected = _image_digest(root, pair.relative_paths)

    with pytest.raises(ValueError, match="image manifest"):
        SUBJECT.bind_authenticated_train_images(pair, root, expected)


def test_image_binding_rejects_empty_relative_path_uniformly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, source_sha, teacher, teacher_sha = _paired_snapshots(tmp_path, monkeypatch)
    pair = SUBJECT.load_authenticated_train_pair(source, source_sha, teacher, teacher_sha)
    root = tmp_path / "images"
    root.mkdir()
    invalid = replace(pair, relative_paths=(".",))

    with pytest.raises(ValueError, match="image manifest"):
        SUBJECT.bind_authenticated_train_images(invalid, root, "0" * 64)


def _fake_unicom_source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path, str]:
    checkout = tmp_path / "unicom-checkout"
    package = checkout / "unicom" / "unicom"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from .model import load\n",
        encoding="utf-8",
    )
    (package / "model.py").write_text(
        """
import torch

def load(*_args, **_kwargs):
    raise AssertionError("download-capable loader must not be called")

def load_model_and_transform(name):
    assert name == "ViT-B/16"
    return torch.nn.Linear(4, 4), lambda value: value
""".lstrip(),
        encoding="utf-8",
    )
    checkpoint = checkout / "checkpoints" / "FP16-ViT-B-16.pt"
    checkpoint.parent.mkdir()
    torch.save(nn.Linear(4, 4).state_dict(), checkpoint)
    checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    registered = {
        **SUBJECT._REGISTERED_SNAPSHOT_METADATA,
        "source": {
            **SUBJECT._REGISTERED_SNAPSHOT_METADATA["source"],
            "checkpoint_sha256": checkpoint_sha256,
            "model_revision": "1" * 40,
        },
    }
    monkeypatch.setattr(SUBJECT, "_REGISTERED_SNAPSHOT_METADATA", registered)
    monkeypatch.setattr(SUBJECT, "_git_root", lambda path: path.resolve(), raising=False)
    monkeypatch.setattr(SUBJECT, "_git_revision", lambda _path: "1" * 40, raising=False)
    monkeypatch.setattr(SUBJECT, "_git_status_porcelain", lambda _path: "", raising=False)
    monkeypatch.setattr(
        SUBJECT,
        "_git_source_bytes",
        lambda root, _revision, relative: (root / relative).read_bytes(),
        raising=False,
    )
    for name in tuple(sys.modules):
        if name in {"unicom", _AUTHENTICATED_UNICOM_MODULE} or name.startswith(
            ("unicom.", f"{_AUTHENTICATED_UNICOM_MODULE}.")
        ):
            monkeypatch.delitem(sys.modules, name, raising=False)
    return checkout, checkpoint, checkpoint_sha256


def test_source_model_loader_binds_checkout_checkpoint_and_local_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, checkpoint, checkpoint_sha256 = _fake_unicom_source(tmp_path, monkeypatch)
    original_sys_path = tuple(sys.path)

    loaded = SUBJECT.load_authenticated_source_model(checkout, checkpoint)

    assert isinstance(loaded.encoder, nn.Module)
    assert loaded.encoder.training is False
    assert isinstance(loaded.transform, Callable)
    assert loaded.revision == "1" * 40
    assert loaded.checkpoint_sha256 == checkpoint_sha256
    assert loaded.package_file == (checkout / "unicom" / "unicom" / "__init__.py").resolve()
    assert tuple(sys.path) == original_sys_path
    assert not any(
        name == _AUTHENTICATED_UNICOM_MODULE or name.startswith(f"{_AUTHENTICATED_UNICOM_MODULE}.")
        for name in sys.modules
    )
    assert all(
        not value.is_floating_point() or value.dtype == torch.float32
        for value in loaded.encoder.state_dict().values()
    )


def test_source_model_loader_rejects_revision_checkpoint_symlink_and_module_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, checkpoint, _checkpoint_sha256 = _fake_unicom_source(tmp_path, monkeypatch)
    monkeypatch.setattr(SUBJECT, "_git_revision", lambda _path: "2" * 40)
    with pytest.raises(ValueError, match="source model"):
        SUBJECT.load_authenticated_source_model(checkout, checkpoint)

    monkeypatch.setattr(SUBJECT, "_git_revision", lambda _path: "1" * 40)
    target = checkpoint.with_name("target.pt")
    checkpoint.rename(target)
    checkpoint.symlink_to(target)
    with pytest.raises(ValueError, match="source model"):
        SUBJECT.load_authenticated_source_model(checkout, checkpoint)

    checkpoint.unlink()
    target.rename(checkpoint)
    foreign = ModuleType(_AUTHENTICATED_UNICOM_MODULE)
    foreign.__file__ = str((checkout / "unicom" / "unicom" / "__init__.py").resolve())
    monkeypatch.setitem(sys.modules, _AUTHENTICATED_UNICOM_MODULE, foreign)
    with pytest.raises(ValueError, match="source model"):
        SUBJECT.load_authenticated_source_model(checkout, checkpoint)


def test_source_model_loader_rejects_dirty_checkout_and_never_uses_named_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, checkpoint, _checkpoint_sha256 = _fake_unicom_source(tmp_path, monkeypatch)
    monkeypatch.setattr(SUBJECT, "_git_status_porcelain", lambda _path: " M unicom/model.py")

    with pytest.raises(ValueError, match="source model"):
        SUBJECT.load_authenticated_source_model(checkout, checkpoint)


def test_source_model_loader_reads_the_authenticated_open_file_description(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, checkpoint, _checkpoint_sha256 = _fake_unicom_source(tmp_path, monkeypatch)
    original_load = torch.load

    def replace_path_then_load(stream: object, *args: object, **kwargs: object) -> object:
        replacement = checkpoint.with_name("replacement.pt")
        torch.save(nn.Linear(3, 3).state_dict(), replacement)
        replacement.replace(checkpoint)
        return original_load(stream, *args, **kwargs)

    monkeypatch.setattr(torch, "load", replace_path_then_load)

    with pytest.raises(ValueError, match="source model"):
        SUBJECT.load_authenticated_source_model(checkout, checkpoint)


def test_source_model_loader_rejects_in_place_checkpoint_mutation_after_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, checkpoint, _checkpoint_sha256 = _fake_unicom_source(tmp_path, monkeypatch)
    original_load = torch.load

    def mutate_after_load(stream: object, *args: object, **kwargs: object) -> object:
        state = original_load(stream, *args, **kwargs)
        with checkpoint.open("r+b") as mutable:
            mutable.seek(-1, 2)
            last = mutable.read(1)
            mutable.seek(-1, 2)
            mutable.write(bytes([last[0] ^ 1]))
        return state

    monkeypatch.setattr(torch, "load", mutate_after_load)

    with pytest.raises(ValueError, match="source model"):
        SUBJECT.load_authenticated_source_model(checkout, checkpoint)


def test_source_model_loader_deserializes_only_the_authenticated_immutable_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, checkpoint, _checkpoint_sha256 = _fake_unicom_source(tmp_path, monkeypatch)
    original_bytes = checkpoint.read_bytes()
    alternate_model = nn.Linear(4, 4)
    with torch.no_grad():
        alternate_model.weight.fill_(9.0)
        alternate_model.bias.fill_(9.0)
    torch.save(alternate_model.state_dict(), checkpoint)
    alternate_bytes = checkpoint.read_bytes()
    checkpoint.write_bytes(original_bytes)
    assert len(alternate_bytes) == len(original_bytes)
    original_load = torch.load

    def overwrite_load_restore(stream: object, *args: object, **kwargs: object) -> object:
        checkpoint.write_bytes(alternate_bytes)
        try:
            return original_load(stream, *args, **kwargs)
        finally:
            checkpoint.write_bytes(original_bytes)

    monkeypatch.setattr(torch, "load", overwrite_load_restore)

    loaded = SUBJECT.load_authenticated_source_model(checkout, checkpoint)

    assert not torch.equal(loaded.encoder.weight, alternate_model.weight)


def test_source_model_loader_ignores_unauthenticated_bytecode_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, checkpoint, _checkpoint_sha256 = _fake_unicom_source(tmp_path, monkeypatch)
    model_path = checkout / "unicom" / "unicom" / "model.py"
    trusted_source = model_path.read_text(encoding="utf-8")
    malicious_source = """
import torch
def load(*_args, **_kwargs): raise AssertionError
def load_model_and_transform(_name):
    raise AssertionError("UNAUTHENTICATED_BYTECODE_EXECUTED")
""".lstrip()
    assert len(malicious_source) <= len(trusted_source)
    malicious_source += "#" * (len(trusted_source) - len(malicious_source))
    fixed_time = 1_700_000_000
    model_path.write_text(malicious_source, encoding="utf-8")
    os.utime(model_path, (fixed_time, fixed_time))
    py_compile.compile(str(model_path), doraise=True)
    model_path.write_text(trusted_source, encoding="utf-8")
    os.utime(model_path, (fixed_time, fixed_time))

    loaded = SUBJECT.load_authenticated_source_model(checkout, checkpoint)

    assert isinstance(loaded.encoder, nn.Linear)


def test_source_model_loader_executes_registered_git_blobs_not_transient_worktree_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, checkpoint, _checkpoint_sha256 = _fake_unicom_source(tmp_path, monkeypatch)
    relative_files = (
        Path("unicom/unicom/__init__.py"),
        Path("unicom/unicom/model.py"),
    )
    registered_blobs = {relative: (checkout / relative).read_bytes() for relative in relative_files}
    monkeypatch.setattr(
        SUBJECT,
        "_git_source_bytes",
        lambda _root, _revision, relative: registered_blobs[relative],
        raising=False,
    )
    (checkout / "unicom/unicom/model.py").write_text(
        "raise AssertionError('TRANSIENT_WORKTREE_CODE_EXECUTED')\n",
        encoding="utf-8",
    )

    loaded = SUBJECT.load_authenticated_source_model(checkout, checkpoint)

    assert isinstance(loaded.encoder, nn.Linear)


def test_source_model_loader_never_falls_back_to_sourceless_bytecode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout, checkpoint, _checkpoint_sha256 = _fake_unicom_source(tmp_path, monkeypatch)
    model_path = checkout / "unicom/unicom/model.py"
    model_pyc = model_path.with_suffix(".pyc")
    malicious = model_path.with_name("malicious.py")
    malicious.write_text(
        "raise AssertionError('SOURCELESS_BYTECODE_EXECUTED')\n",
        encoding="utf-8",
    )
    py_compile.compile(str(malicious), cfile=str(model_pyc), doraise=True)
    model_path.unlink()

    with pytest.raises(ValueError, match="source model"):
        SUBJECT.load_authenticated_source_model(checkout, checkpoint)


def test_git_source_bytes_ignores_git_replacement_objects(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    subprocess.run(["git", "-C", str(checkout), "config", "user.name", "Fixture"], check=True)
    subprocess.run(
        ["git", "-C", str(checkout), "config", "user.email", "fixture@example.invalid"],
        check=True,
    )
    relative = Path("unicom/unicom/model.py")
    source = checkout / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"trusted source\n")
    subprocess.run(["git", "-C", str(checkout), "add", relative.as_posix()], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "fixture"], check=True)
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    original_blob = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", f"{revision}:{relative.as_posix()}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    replacement_blob = (
        subprocess.run(
            ["git", "-C", str(checkout), "hash-object", "-w", "--stdin"],
            input=b"replacement source\n",
            check=True,
            capture_output=True,
        )
        .stdout.decode()
        .strip()
    )
    subprocess.run(
        ["git", "-C", str(checkout), "replace", original_blob, replacement_blob],
        check=True,
    )

    assert SUBJECT._git_source_bytes(checkout, revision, relative) == b"trusted source\n"


def test_git_source_bytes_rejects_ambient_repository_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    subprocess.run(["git", "init", "-q", str(checkout)], check=True)
    relative = Path("unicom/unicom/model.py")
    source = checkout / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"trusted source\n")
    subprocess.run(["git", "-C", str(checkout), "add", relative.as_posix()], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(checkout),
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
    revision = subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.setenv("GIT_DIR", str(checkout / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(checkout))

    with pytest.raises(subprocess.CalledProcessError):
        SUBJECT._git_source_bytes(unrelated, revision, relative)
