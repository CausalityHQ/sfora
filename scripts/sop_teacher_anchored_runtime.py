#!/usr/bin/env python3
"""Authenticated SOP-only runtime adapter for teacher-anchored training."""

from __future__ import annotations

import hashlib
import importlib.util
import stat
import struct
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

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
