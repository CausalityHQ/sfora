#!/usr/bin/env python3
"""Run the authenticated optimization-only SigLIP SSOR diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from sfora.siglip_head_screen import FeatureSplitAuthority, build_feature_split_authority
from sfora.siglip_ssor import (
    canonical_ssor_result_bytes,
    run_ssor_nested_diagnostic,
    seen_class_projector,
    ssor_deployment_head_artifact_bytes,
)

_POOLER_DIMENSIONS = 1152
_DESCRIPTOR_DIMENSIONS = 512
_OPTIMIZATION_CLASS_COUNT = 49
_RECONSTRUCTION_COSINE_TOLERANCE = 1.0e-5


@dataclass(frozen=True, slots=True)
class LoadedSSORCache:
    """Authenticated optimization-only tensors and immutable identities."""

    pooler: torch.Tensor
    descriptors: torch.Tensor
    labels: torch.Tensor
    head: torch.Tensor
    ordered_example_ids: tuple[str, ...]
    split_authority: FeatureSplitAuthority
    feature_cache_manifest_sha256: str
    label_vector_sha256: str
    control_head_sha256: str
    maximum_reconstruction_cosine_deviation: float


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hexadecimal characters")
    return value


def _source_commit(value: str) -> str:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError(
            "source commit must be 40 lowercase hexadecimal characters"
        )
    return value


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("paths must be normalized absolute paths")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-only SSOR execution boundary."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--feature-manifest", required=True, type=_absolute_path)
    parser.add_argument("--feature-manifest-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--source-commit", required=True, type=_source_commit)
    parser.add_argument("--checkpoint-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--result", required=True, type=_absolute_path)
    parser.add_argument("--deployment-head", required=True, type=_absolute_path)
    parser.add_argument("--execute-ssor", required=True, action="store_true")
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    parsed = parser.parse_args(effective)
    if parsed.result == parsed.deployment_head:
        parser.error("result and deployment-head paths must differ")
    return parsed


def _read_regular(path: Path, *, role: str) -> bytes:
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"SSOR {role} path differs")
    return path.read_bytes()


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _digest(value: object) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("SSOR manifest digest differs")
    return value


def _load_array(
    root: Path,
    row: object,
    *,
    role: str,
    dtype: np.dtype[object],
    shape: tuple[int, ...],
) -> tuple[np.ndarray, str]:
    if type(row) is not dict or set(row) != {"file", "sha256", "shape"}:
        raise ValueError("SSOR manifest file schema differs")
    filename = row["file"]
    expected_digest = _digest(row["sha256"])
    if type(filename) is not str or filename != f"{role}.npy" or row["shape"] != list(shape):
        raise ValueError("SSOR manifest file authority differs")
    raw = _read_regular(root / filename, role=role)
    if hashlib.sha256(raw).hexdigest() != expected_digest:
        raise ValueError("SSOR cache digest differs")
    try:
        array = np.load(io.BytesIO(raw), allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("SSOR cache array differs") from error
    if type(array) is not np.ndarray or array.dtype != dtype or array.shape != shape:
        raise ValueError("SSOR cache array authority differs")
    return array, expected_digest


def _tensor_sha256(role: str, tensor: torch.Tensor) -> str:
    payload = bytearray(b"sfora-ssor-tensor-v1\0")
    encoded = role.encode()
    payload.extend(len(encoded).to_bytes(8, "big"))
    payload.extend(encoded)
    payload.extend(tensor.ndim.to_bytes(8, "big"))
    for dimension in tensor.shape:
        payload.extend(dimension.to_bytes(8, "big"))
    if tensor.dtype == torch.int64:
        payload.extend(tensor.contiguous().numpy().astype("<i8", copy=False).tobytes())
    elif tensor.dtype == torch.float32:
        payload.extend(tensor.contiguous().numpy().astype("<f4", copy=False).tobytes())
    else:
        raise ValueError("SSOR tensor digest authority differs")
    return hashlib.sha256(payload).hexdigest()


def load_ssor_cache(
    manifest_path: Path,
    *,
    expected_sha256: str,
    expected_source_commit: str,
    expected_checkpoint_sha256: str,
    expected_pooler_dimensions: int = _POOLER_DIMENSIONS,
    expected_descriptor_dimensions: int = _DESCRIPTOR_DIMENSIONS,
    expected_class_count: int = _OPTIMIZATION_CLASS_COUNT,
) -> LoadedSSORCache:
    """Authenticate and reconstruct one optimization-only SSOR cache."""

    if (
        not isinstance(manifest_path, Path)
        or type(expected_pooler_dimensions) is not int
        or expected_pooler_dimensions < 2
        or type(expected_descriptor_dimensions) is not int
        or expected_descriptor_dimensions < 2
        or type(expected_class_count) is not int
        or expected_class_count < 8
    ):
        raise ValueError("SSOR loader authority differs")
    raw = _read_regular(manifest_path, role="manifest")
    if hashlib.sha256(raw).hexdigest() != _digest(expected_sha256):
        raise ValueError("SSOR manifest digest differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("SSOR manifest JSON differs") from error
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError("SSOR manifest canonical bytes differ")
    if set(value) != {
        "schema",
        "claim_eligible",
        "official_test_access",
        "role",
        "source_manifest_sha256",
        "source_commit",
        "checkpoint_sha256",
        "example_ids",
        "files",
    }:
        raise ValueError("SSOR manifest schema differs")
    example_ids_value = value["example_ids"]
    if (
        value["schema"] != "sfora-siglip-ssor-cache-v1"
        or type(value["claim_eligible"]) is not bool
        or value["claim_eligible"]
        or type(value["official_test_access"]) is not bool
        or value["official_test_access"]
        or value["role"] != "optimization-train"
        or value["source_commit"] != expected_source_commit
        or value["checkpoint_sha256"] != _digest(expected_checkpoint_sha256)
        or type(example_ids_value) is not list
        or len(example_ids_value) < 2
        or len(set(example_ids_value)) != len(example_ids_value)
        or any(type(item) is not str or not item for item in example_ids_value)
    ):
        raise ValueError("SSOR manifest authority differs")
    source_manifest_sha256 = _digest(value["source_manifest_sha256"])
    files = value["files"]
    if type(files) is not dict or set(files) != {"pooler", "descriptors", "labels", "head"}:
        raise ValueError("SSOR manifest files differ")
    count = len(example_ids_value)
    pooler_array, _pooler_digest = _load_array(
        manifest_path.parent,
        files["pooler"],
        role="pooler",
        dtype=np.dtype("<f4"),
        shape=(count, expected_pooler_dimensions),
    )
    descriptors_array, _descriptor_digest = _load_array(
        manifest_path.parent,
        files["descriptors"],
        role="descriptors",
        dtype=np.dtype("<f4"),
        shape=(count, expected_descriptor_dimensions),
    )
    labels_array, _labels_file_digest = _load_array(
        manifest_path.parent,
        files["labels"],
        role="labels",
        dtype=np.dtype("<i8"),
        shape=(count,),
    )
    head_array, head_file_digest = _load_array(
        manifest_path.parent,
        files["head"],
        role="head",
        dtype=np.dtype("<f4"),
        shape=(expected_descriptor_dimensions, expected_pooler_dimensions),
    )
    pooler = torch.from_numpy(pooler_array.copy()).contiguous()
    descriptors = torch.from_numpy(descriptors_array.copy()).contiguous()
    labels = torch.from_numpy(labels_array.copy()).contiguous()
    head = torch.from_numpy(head_array.copy()).contiguous()
    if (
        not bool(torch.isfinite(pooler).all())
        or not bool(torch.isfinite(descriptors).all())
        or not bool(torch.isfinite(head).all())
        or tuple(sorted(int(item) for item in torch.unique(labels).tolist()))
        != tuple(range(expected_class_count))
        or any(int((labels == label).sum()) < 2 for label in range(expected_class_count))
    ):
        raise ValueError("SSOR cache tensor authority differs")
    descriptor_norms = torch.linalg.vector_norm(descriptors.double(), dim=1)
    if not bool(torch.all(torch.abs(descriptor_norms - 1.0) <= 2.0e-6)):
        raise ValueError("SSOR cache descriptor norm differs")
    recomputed = F.normalize(pooler.double() @ head.double().T, dim=1)
    cosine_deviation = torch.abs(1.0 - torch.sum(recomputed * descriptors.double(), dim=1))
    maximum_deviation = float(cosine_deviation.max())
    if maximum_deviation > _RECONSTRUCTION_COSINE_TOLERANCE:
        raise ValueError("SSOR cache head reconstruction differs")
    ordered_example_ids = tuple(example_ids_value)
    split_authority = build_feature_split_authority(
        source_manifest_sha256=source_manifest_sha256,
        role="optimization-train",
        official_test_access=False,
        ordered_example_ids=ordered_example_ids,
        features=descriptors,
    )
    return LoadedSSORCache(
        pooler=pooler,
        descriptors=descriptors,
        labels=labels,
        head=head,
        ordered_example_ids=ordered_example_ids,
        split_authority=split_authority,
        feature_cache_manifest_sha256=hashlib.sha256(raw).hexdigest(),
        label_vector_sha256=_tensor_sha256("labels", labels),
        control_head_sha256=head_file_digest,
        maximum_reconstruction_cosine_deviation=maximum_deviation,
    )


def _write_exclusive(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    path.parent.mkdir(parents=True, exist_ok=True)
    installed = False
    try:
        with partial.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        partial.replace(path)
        installed = True
        directory_descriptor = os.open(
            path.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        if installed and path.is_file() and not path.is_symlink():
            path.unlink()
        if partial.is_file() and not partial.is_symlink():
            partial.unlink()
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Authenticate local cache bytes, run SSOR, and write one canonical result."""

    arguments = parse_args(argv)
    if arguments.result.exists() or arguments.deployment_head.exists():
        raise FileExistsError("SSOR output already exists")
    cache = load_ssor_cache(
        arguments.feature_manifest,
        expected_sha256=arguments.feature_manifest_sha256,
        expected_source_commit=arguments.source_commit,
        expected_checkpoint_sha256=arguments.checkpoint_sha256,
    )
    evidence = run_ssor_nested_diagnostic(
        cache.descriptors,
        cache.labels,
        ordered_example_ids=cache.ordered_example_ids,
        split_authority=cache.split_authority,
    )
    head_raw = None
    deployment_projector = None
    deployment_head_file_sha256 = None
    if evidence.passed:
        deployment_projector = seen_class_projector(
            cache.descriptors,
            cache.labels,
            fit_labels=tuple(range(_OPTIMIZATION_CLASS_COUNT)),
        )
        head_raw = ssor_deployment_head_artifact_bytes(
            cache.head,
            deployment_projector,
            beta=float(evidence.deployment_beta),
        )
        deployment_head_file_sha256 = hashlib.sha256(head_raw).hexdigest()
    result_raw = canonical_ssor_result_bytes(
        evidence,
        source_manifest_sha256=cache.split_authority.source_manifest_sha256,
        feature_cache_manifest_sha256=cache.feature_cache_manifest_sha256,
        ordered_example_ids_sha256=cache.split_authority.ordered_example_ids_sha256,
        feature_matrix_sha256=cache.split_authority.feature_matrix_sha256,
        label_vector_sha256=cache.label_vector_sha256,
        control_head_weight=cache.head,
        deployment_projector=deployment_projector,
        deployment_head_artifact=head_raw,
    )
    result_value = json.loads(result_raw)
    head_written = False
    result_written = False
    try:
        if head_raw is not None:
            _write_exclusive(arguments.deployment_head, head_raw)
            head_written = True
            written_head = _read_regular(arguments.deployment_head, role="deployment head")
            if (
                written_head != head_raw
                or hashlib.sha256(written_head).hexdigest()
                != result_value["deployment_head_file_sha256"]
            ):
                raise ValueError("SSOR deployment artifact differs after write")
        _write_exclusive(arguments.result, result_raw)
        result_written = True
        written_result = _read_regular(arguments.result, role="result artifact")
        if written_result != result_raw:
            raise ValueError("SSOR result artifact differs after write")
    except BaseException:
        if result_written and arguments.result.is_file() and not arguments.result.is_symlink():
            arguments.result.unlink()
        if (
            head_written
            and arguments.deployment_head.is_file()
            and not arguments.deployment_head.is_symlink()
        ):
            arguments.deployment_head.unlink()
        raise
    sys.stdout.write(
        json.dumps(
            {
                "passed": evidence.passed,
                "result": str(arguments.result),
                "result_file_sha256": hashlib.sha256(result_raw).hexdigest(),
                "deployment_head": (
                    str(arguments.deployment_head)
                    if deployment_head_file_sha256 is not None
                    else None
                ),
                "deployment_head_file_sha256": deployment_head_file_sha256,
                "deployment_head_tensor_sha256": result_value["deployment_head_sha256"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
