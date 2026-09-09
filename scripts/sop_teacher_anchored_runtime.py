#!/usr/bin/env python3
"""Authenticated SOP-only runtime adapter for teacher-anchored training."""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import io
import os
import stat
import struct
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import CodeType, ModuleType
from typing import Any, cast

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from sfora.nested_rank_protocol import ordered_training_records_sha256

_COMMON_TRAIN_ARRAY_SHA256 = {
    "train_image_ids": "0d3165e8e29881062b85f0d83b68ef2d70d265f007a4d62263677a234a42473a",
    "train_labels": "d785d2eca417d91257195bf2c16d87b7805178984db82c0d0a0f5ca8546a216f",
    "train_relative_paths": "4795c07c5dce273dac358403435269a89ea68b2b0870733a7b152b2531f8c949",
}
_COMMON_EXCLUDED_TEST_ARRAY_SHA256 = {
    "test_image_ids": "8f2a940a1eb82997aed2cc685aa17b06e833db8f25aa9818fabe786158f5ada0",
    "test_labels": "2446a789df05e0e579ad69b03c9ce55d61dee714eb973aa14120d1ed6658e929",
    "test_relative_paths": "cb7054d0e026dc459b9f1b0a8a7cfcd351e7beff8bf2f9038d3f980fea4b789d",
}
_REGISTERED_SNAPSHOT_METADATA = {
    "source": {
        "checkpoint_sha256": "c04f324f7c3b4435667236ec6c0eca1cd62f9d64fbfc2d06f8e8e60e6497edef",
        "excluded_test_array_sha256": {
            **_COMMON_EXCLUDED_TEST_ARRAY_SHA256,
            "test_embeddings": "52b7b1fa8c2668468c9ac8983a8dd1d98d8c04ca0684bec79c2c31ee5666bfa9",
        },
        "model_identifier": "UNICOM-ViT-B/16",
        "model_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "source_archive_sha256": "6bc0d8383251685eaccd472eeda357861caffb3bfb4129f18e0124c3ddc72818",
        "train_array_sha256": {
            **_COMMON_TRAIN_ARRAY_SHA256,
            "train_embeddings": "d88e9d35f8419c7a661bd1358c901ecb2c64d4111ecd6ec311229d0d7c76dd74",
        },
    },
    "teacher": {
        "checkpoint_sha256": "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea",
        "excluded_test_array_sha256": {
            **_COMMON_EXCLUDED_TEST_ARRAY_SHA256,
            "test_embeddings": "626abb452f8305c6a3f687b61daacba096f1bcd61243cfde6abdedbf191354b8",
        },
        "model_identifier": "UNICOM-ViT-L/14@336px",
        "model_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "source_archive_sha256": "1ba27b2d6b9db39067aa6facd0ef8aafc303c4527f6feabed859b0512c7d921a",
        "train_array_sha256": {
            **_COMMON_TRAIN_ARRAY_SHA256,
            "train_embeddings": "5a8629deee1adff92ac941a0f78f4fd55f1d4cb0db43b1459a0d6c92f84b46fc",
        },
    },
}
_AUTHENTICATED_UNICOM_MODULE = "_sfora_authenticated_unicom"


@dataclass(frozen=True, slots=True)
class TeacherAnchoredTrainPair:
    """Aligned train-only source and teacher snapshot rows."""

    source_embeddings: NDArray[np.float32]
    teacher_embeddings: NDArray[np.float32]
    labels: NDArray[np.int64]
    image_ids: NDArray[np.int64]
    relative_paths: tuple[str, ...]
    source_metadata: dict[str, object]
    teacher_metadata: dict[str, object]
    source_snapshot_sha256: str
    teacher_snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class TeacherAnchoredImageManifest:
    """Ordered training-image capabilities bound without walking the dataset tree."""

    image_paths: tuple[Path, ...]
    relative_paths: tuple[str, ...]
    sha256: str


@dataclass(frozen=True, slots=True)
class TeacherAnchoredSourceModel:
    """Authenticated local B/16 encoder and its canonical image transform."""

    encoder: nn.Module
    transform: Callable[[Any], torch.Tensor]
    revision: str
    checkpoint_sha256: str
    package_file: Path


def _load_snapshot_module() -> Any:
    path = Path(__file__).resolve().parent / "train_sop_nested_neighborhood_rank.py"
    name = "sfora_teacher_anchored_snapshot_loader"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("teacher-anchored snapshot pair authority differs")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    if Path(cast(str, module.__file__)).resolve() != path:
        raise ValueError("teacher-anchored snapshot pair authority differs")
    return module


def _is_sha256(value: object) -> bool:
    return bool(
        type(value) is str and len(value) == 64 and not (set(value) - set("0123456789abcdef"))
    )


def _git_revision(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status_porcelain(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def _stream_sha256(stream: Any) -> str:
    digest = hashlib.sha256()
    while chunk := stream.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def _git_source_bytes(checkout: Path, revision: str, relative: Path) -> bytes:
    return subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-C",
            str(checkout),
            "show",
            f"{revision}:{relative.as_posix()}",
        ],
        check=True,
        capture_output=True,
    ).stdout


class _SourceOnlyLoader(importlib.machinery.SourceFileLoader):
    """Compile authenticated source directly, never an ignored bytecode cache."""

    def __init__(self, fullname: str, path: str, source: bytes) -> None:
        super().__init__(fullname, path)
        self._source = source

    def get_code(self, fullname: str) -> CodeType:
        source_path = self.get_filename(fullname)
        return self.source_to_code(self._source, source_path)


class _AuthenticatedPackageFinder(importlib.abc.MetaPathFinder):
    def __init__(self, prefix: str, checkout: Path, revision: str, package: Path) -> None:
        self._prefix = prefix
        self._checkout = checkout
        self._revision = revision
        self._package = package

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        del path, target
        if fullname == self._prefix:
            candidate = self._package / "__init__.py"
            package_locations = [str(self._package)]
        elif fullname.startswith(f"{self._prefix}."):
            module_parts = fullname[len(self._prefix) + 1 :].split(".")
            module_path = self._package.joinpath(*module_parts)
            package_file = module_path / "__init__.py"
            if package_file.is_file():
                candidate = package_file
                package_locations = [str(module_path)]
            else:
                candidate = module_path.with_suffix(".py")
                package_locations = None
        else:
            return None
        try:
            info = candidate.lstat()
        except OSError as error:
            raise ImportError("authenticated package source is absent") from error
        if candidate.is_symlink() or not stat.S_ISREG(info.st_mode):
            raise ImportError("authenticated package source authority differs")
        try:
            source_relative = candidate.relative_to(self._checkout)
            source = _git_source_bytes(self._checkout, self._revision, source_relative)
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            raise ImportError("authenticated package Git blob authority differs") from error
        loader = _SourceOnlyLoader(fullname, str(candidate), source)
        return importlib.util.spec_from_file_location(
            fullname,
            candidate,
            loader=loader,
            submodule_search_locations=package_locations,
        )


def _load_authenticated_unicom_builder(
    checkout: Path,
    revision: str,
    package: Path,
    package_file: Path,
) -> Callable[[str], tuple[nn.Module, Callable[[Any], torch.Tensor]]]:
    prefix = _AUTHENTICATED_UNICOM_MODULE
    if any(name == prefix or name.startswith(f"{prefix}.") for name in sys.modules):
        raise ValueError("teacher-anchored source model authority differs")
    finder = _AuthenticatedPackageFinder(prefix, checkout, revision, package)
    original_meta_path = tuple(sys.meta_path)
    sys.meta_path.insert(0, finder)
    try:
        importlib.import_module(prefix)
        model_module = sys.modules.get(f"{prefix}.model")
        builder = getattr(model_module, "load_model_and_transform", None)
        imported = {
            name: candidate
            for name, candidate in sys.modules.items()
            if name == prefix or name.startswith(f"{prefix}.")
        }
        if not callable(builder) or not imported:
            raise ValueError("teacher-anchored source model authority differs")
        for candidate in imported.values():
            candidate_file = getattr(candidate, "__file__", None)
            if type(candidate_file) is not str:
                raise ValueError("teacher-anchored source model authority differs")
            resolved = Path(candidate_file).resolve(strict=True)
            if package not in resolved.parents or resolved.is_symlink():
                raise ValueError("teacher-anchored source model authority differs")
        return cast(Callable[[str], tuple[nn.Module, Callable[[Any], torch.Tensor]]], builder)
    finally:
        sys.meta_path[:] = original_meta_path
        for name in tuple(sys.modules):
            if name == prefix or name.startswith(f"{prefix}."):
                sys.modules.pop(name, None)


def load_authenticated_source_model(
    checkout: Path,
    checkpoint: Path,
) -> TeacherAnchoredSourceModel:
    """Load the registered local B/16 graph without a download-capable input."""

    registered = _REGISTERED_SNAPSHOT_METADATA["source"]
    expected_revision = registered["model_revision"]
    expected_checkpoint = registered["checkpoint_sha256"]
    if (
        not isinstance(checkout, Path)
        or not isinstance(checkpoint, Path)
        or checkout.is_symlink()
        or not checkout.is_dir()
        or checkpoint.name != "FP16-ViT-B-16.pt"
        or checkpoint.is_symlink()
        or not checkpoint.is_file()
        or type(expected_revision) is not str
        or type(expected_checkpoint) is not str
    ):
        raise ValueError("teacher-anchored source model authority differs")
    try:
        checkpoint_info = checkpoint.lstat()
        if not stat.S_ISREG(checkpoint_info.st_mode):
            raise ValueError("teacher-anchored source model authority differs")
        revision = _git_revision(checkout)
        checkout_status = _git_status_porcelain(checkout)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("teacher-anchored source model authority differs") from error
    if revision != expected_revision or checkout_status:
        raise ValueError("teacher-anchored source model authority differs")

    package_parent = checkout / "unicom"
    package = package_parent / "unicom"
    package_file = package / "__init__.py"
    try:
        for component, expected_kind in (
            (package_parent, stat.S_ISDIR),
            (package, stat.S_ISDIR),
            (package_file, stat.S_ISREG),
        ):
            info = component.lstat()
            if component.is_symlink() or not expected_kind(info.st_mode):
                raise ValueError("teacher-anchored source model authority differs")
        expected_package_file = package_file.resolve(strict=True)
        flags = os.O_RDONLY | os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(checkpoint, flags)
        with os.fdopen(descriptor, "rb", buffering=0) as stream:
            opened_info = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened_info.st_mode):
                raise ValueError("teacher-anchored source model authority differs")
            checkpoint_bytes = stream.read()
            checkpoint_sha256 = hashlib.sha256(checkpoint_bytes).hexdigest()
            if checkpoint_sha256 != expected_checkpoint:
                raise ValueError("teacher-anchored source model authority differs")
            state_dict = torch.load(
                io.BytesIO(checkpoint_bytes),
                map_location="cpu",
                weights_only=True,
            )
            final_opened_info = os.fstat(stream.fileno())
            stream.seek(0)
            final_opened_sha256 = _stream_sha256(stream)
        if not isinstance(state_dict, dict) or not state_dict:
            raise ValueError("teacher-anchored source model authority differs")
        if any(
            type(key) is not str
            or not isinstance(value, torch.Tensor)
            or (value.is_floating_point() and not bool(torch.isfinite(value).all()))
            for key, value in state_dict.items()
        ):
            raise ValueError("teacher-anchored source model authority differs")
        builder = _load_authenticated_unicom_builder(
            checkout,
            revision,
            package,
            expected_package_file,
        )
        loaded = builder("ViT-B/16")
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError("teacher-anchored source model authority differs") from error
    if (
        type(loaded) is not tuple
        or len(loaded) != 2
        or not isinstance(loaded[0], nn.Module)
        or not callable(loaded[1])
    ):
        raise ValueError("teacher-anchored source model authority differs")
    encoder = loaded[0]
    try:
        encoder.load_state_dict(state_dict, strict=True)
    except RuntimeError as error:
        raise ValueError("teacher-anchored source model authority differs") from error
    encoder.eval()
    if any(
        not bool(torch.isfinite(value).all())
        or (value.is_floating_point() and value.dtype != torch.float32)
        for value in encoder.state_dict().values()
    ):
        raise ValueError("teacher-anchored source model authority differs")
    try:
        final_info = checkpoint.lstat()
        final_status = _git_status_porcelain(checkout)
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("teacher-anchored source model authority differs") from error
    if (
        checkpoint.is_symlink()
        or (final_info.st_dev, final_info.st_ino, final_info.st_size)
        != (opened_info.st_dev, opened_info.st_ino, opened_info.st_size)
        or (final_opened_info.st_dev, final_opened_info.st_ino, final_opened_info.st_size)
        != (opened_info.st_dev, opened_info.st_ino, opened_info.st_size)
        or final_opened_sha256 != checkpoint_sha256
        or final_status
    ):
        raise ValueError("teacher-anchored source model authority differs")
    return TeacherAnchoredSourceModel(
        encoder=encoder,
        transform=loaded[1],
        revision=revision,
        checkpoint_sha256=checkpoint_sha256,
        package_file=expected_package_file,
    )


def load_authenticated_train_pair(
    source_path: Path,
    source_sha256: str,
    teacher_path: Path,
    teacher_sha256: str,
) -> TeacherAnchoredTrainPair:
    """Authenticate and align the B/16 and L/14 official-training-only snapshots."""

    if (
        not isinstance(source_path, Path)
        or not isinstance(teacher_path, Path)
        or source_path == teacher_path
        or not _is_sha256(source_sha256)
        or not _is_sha256(teacher_sha256)
    ):
        raise ValueError("teacher-anchored snapshot pair authority differs")
    try:
        loader = _load_snapshot_module().load_train_snapshot
        source = loader(source_path, source_sha256)
        teacher = loader(teacher_path, teacher_sha256)
    except (AttributeError, OSError, ValueError) as error:
        raise ValueError("teacher-anchored snapshot pair authority differs") from error
    source_metadata = cast(dict[str, object], source.metadata)
    teacher_metadata = cast(dict[str, object], teacher.metadata)
    registered_source = _REGISTERED_SNAPSHOT_METADATA["source"]
    registered_teacher = _REGISTERED_SNAPSHOT_METADATA["teacher"]
    if (
        any(source_metadata.get(key) != value for key, value in registered_source.items())
        or any(teacher_metadata.get(key) != value for key, value in registered_teacher.items())
        or source_metadata.get("embedding_dimension") != 768
        or teacher_metadata.get("embedding_dimension") != 768
        or source_metadata.get("model_revision") != teacher_metadata.get("model_revision")
        or source_metadata.get("ordered_train_record_sha256")
        != teacher_metadata.get("ordered_train_record_sha256")
        or source_metadata.get("ordered_train_record_sha256")
        != ordered_training_records_sha256(
            source.image_ids,
            source.labels,
            tuple(str(value) for value in source.relative_paths),
        )
        or cast(dict[str, object], source_metadata["excluded_test_array_sha256"])["test_labels"]
        != cast(dict[str, object], teacher_metadata["excluded_test_array_sha256"])["test_labels"]
        or cast(dict[str, object], source_metadata["excluded_test_array_sha256"])["test_image_ids"]
        != cast(dict[str, object], teacher_metadata["excluded_test_array_sha256"])["test_image_ids"]
        or cast(dict[str, object], source_metadata["excluded_test_array_sha256"])[
            "test_relative_paths"
        ]
        != cast(dict[str, object], teacher_metadata["excluded_test_array_sha256"])[
            "test_relative_paths"
        ]
        or not np.array_equal(source.labels, teacher.labels)
        or not np.array_equal(source.image_ids, teacher.image_ids)
        or not np.array_equal(source.relative_paths, teacher.relative_paths)
    ):
        raise ValueError("teacher-anchored snapshot pair authority differs")
    return TeacherAnchoredTrainPair(
        source_embeddings=np.ascontiguousarray(source.embeddings, dtype=np.float32),
        teacher_embeddings=np.ascontiguousarray(teacher.embeddings, dtype=np.float32),
        labels=np.ascontiguousarray(source.labels, dtype=np.int64),
        image_ids=np.ascontiguousarray(source.image_ids, dtype=np.int64),
        relative_paths=tuple(str(value) for value in source.relative_paths),
        source_metadata=dict(source_metadata),
        teacher_metadata=dict(teacher_metadata),
        source_snapshot_sha256=source_sha256,
        teacher_snapshot_sha256=teacher_sha256,
    )


def bind_authenticated_train_images(
    pair: TeacherAnchoredTrainPair,
    image_root: Path,
    expected_sha256: str,
) -> TeacherAnchoredImageManifest:
    """Bind only the ordered training images named by the authenticated snapshots."""

    if (
        type(pair) is not TeacherAnchoredTrainPair
        or not isinstance(image_root, Path)
        or image_root.is_symlink()
        or not image_root.is_dir()
        or not _is_sha256(expected_sha256)
    ):
        raise ValueError("teacher-anchored image manifest authority differs")
    root = image_root.resolve()
    digest = hashlib.sha256(b"sfora-teacher-anchored-image-tree-v1\x00")
    image_paths: list[Path] = []
    for relative in pair.relative_paths:
        pure = PurePosixPath(relative)
        if (
            not relative
            or not pure.parts
            or pure.is_absolute()
            or ".." in pure.parts
            or "." in pure.parts
            or str(pure) != relative
        ):
            raise ValueError("teacher-anchored image manifest authority differs")
        path = image_root.joinpath(*pure.parts)
        try:
            current = image_root
            for index, component in enumerate(pure.parts):
                current = current / component
                info = current.lstat()
                if current.is_symlink() or (
                    index < len(pure.parts) - 1 and not stat.S_ISDIR(info.st_mode)
                ):
                    raise ValueError("teacher-anchored image manifest authority differs")
        except OSError as error:
            raise ValueError("teacher-anchored image manifest authority differs") from error
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("teacher-anchored image manifest authority differs")
        resolved = path.resolve()
        if not resolved.is_relative_to(root):
            raise ValueError("teacher-anchored image manifest authority differs")
        file_digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                size += len(chunk)
                file_digest.update(chunk)
        encoded = relative.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(struct.pack("<Q", size))
        digest.update(file_digest.digest())
        image_paths.append(path)
    observed = digest.hexdigest()
    if observed != expected_sha256:
        raise ValueError("teacher-anchored image manifest authority differs")
    return TeacherAnchoredImageManifest(
        image_paths=tuple(image_paths),
        relative_paths=pair.relative_paths,
        sha256=observed,
    )
