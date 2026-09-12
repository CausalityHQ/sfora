#!/usr/bin/env python3
"""Fail-closed artifacts for positive-coverage adaptation experiments."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch

_ARM_NAMES = ("pooled", "coverage")
_CONTROL_ARM_NAMES = ("pooled", "coverage", "mean_logit", "supcon", "multi_similarity")
_PROJECTION_ARM_NAMES = ("restricted_adapter", "factorized_adapter", "direct_projection")


def mean_recent_loss(values: list[float], *, window: int) -> float:
    """Return the finite mean of the observed tail, including short runs."""

    if (
        type(values) is not list
        or not values
        or type(window) is not int
        or window <= 0
        or any(type(value) is not float or not math.isfinite(value) for value in values)
    ):
        raise ValueError("training-loss authority differs")
    tail = values[-window:]
    return math.fsum(tail) / len(tail)


def _lower_hex(value: object, width: int) -> bool:
    return (
        type(value) is str
        and len(value) == width
        and all(character in "0123456789abcdef" for character in value)
    )


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def linear_weight_sha256(weight: torch.Tensor) -> str:
    """Hash one linear-map weight with explicit shape and float32 byte authority."""

    if (
        type(weight) is not torch.Tensor
        or weight.device.type != "cpu"
        or weight.dtype != torch.float32
        or weight.ndim != 2
        or weight.shape[0] < 2
        or weight.shape[0] != weight.shape[1]
        or not weight.is_contiguous()
        or not bool(torch.isfinite(weight).all())
    ):
        raise ValueError("positive-coverage artifact authority differs")
    return _tensor_sequence_sha256(weight)


def _tensor_sequence_sha256(*values: torch.Tensor) -> str:
    digest = hashlib.sha256()
    for value in values:
        array = value.detach().numpy().astype("<f4", copy=False)
        digest.update(struct.pack("<I", array.ndim))
        digest.update(struct.pack(f"<{array.ndim}Q", *array.shape))
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def affine_parameters_sha256(weight: torch.Tensor, bias: torch.Tensor) -> str:
    """Hash one float32 affine head with explicit shape and byte authority."""

    if (
        type(weight) is not torch.Tensor
        or type(bias) is not torch.Tensor
        or weight.device.type != "cpu"
        or bias.device.type != "cpu"
        or weight.dtype != torch.float32
        or bias.dtype != torch.float32
        or weight.ndim != 2
        or bias.ndim != 1
        or weight.shape[0] < 2
        or weight.shape[1] < 2
        or bias.shape[0] != weight.shape[0]
        or not weight.is_contiguous()
        or not bias.is_contiguous()
        or not bool(torch.isfinite(weight).all())
        or not bool(torch.isfinite(bias).all())
    ):
        raise ValueError("positive-coverage artifact authority differs")
    return _tensor_sequence_sha256(weight, bias)


def positive_coverage_source_identity(
    *, driver: Path, driver_sha256: str, source_revision: str
) -> dict[str, str]:
    """Authenticate the exact experiment driver and repository source revision."""

    if (
        not isinstance(driver, Path)
        or not driver.is_absolute()
        or not driver.is_file()
        or not _lower_hex(driver_sha256, 64)
        or not _lower_hex(source_revision, 40)
        or _file_sha256(driver) != driver_sha256
    ):
        raise ValueError("source identity authority differs")
    try:
        root_text = subprocess.check_output(
            ["git", "-C", str(driver.parent), "rev-parse", "--show-toplevel"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        head = subprocess.check_output(
            ["git", "-C", root_text, "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", root_text, "status", "--porcelain"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("source identity authority differs") from error
    root = Path(root_text).resolve()
    if head != source_revision or status or not driver.resolve().is_relative_to(root):
        raise ValueError("source identity authority differs")
    return {"driver_sha256": driver_sha256, "source_revision": source_revision}


def canonical_positive_coverage_receipt_bytes(value: dict[str, object]) -> bytes:
    """Encode one finite sorted compact JSON receipt with exactly one final LF."""

    if type(value) is not dict:
        raise ValueError("positive-coverage artifact authority differs")
    try:
        text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("positive-coverage artifact authority differs") from error
    return (text + "\n").encode()


@dataclass(frozen=True, slots=True)
class PositiveCoverageArtifactPaths:
    """Distinct no-clobber output paths for both arms and their complete receipt."""

    pooled_checkpoint: Path
    coverage_checkpoint: Path
    complete_receipt: Path

    def __post_init__(self) -> None:
        paths = (self.pooled_checkpoint, self.coverage_checkpoint, self.complete_receipt)
        all_paths = (*paths, *(_partial(path) for path in paths))
        if (
            any(not isinstance(path, Path) or not path.is_absolute() for path in paths)
            or len(set(all_paths)) != len(all_paths)
            or any(not path.parent.is_dir() for path in paths)
            or any(path.exists() or _partial(path).exists() for path in paths)
        ):
            raise ValueError("artifact path authority differs")


@dataclass(frozen=True, slots=True)
class MatchedLossPanelArtifactPaths:
    """Distinct no-clobber output paths for the five matched loss arms."""

    pooled_checkpoint: Path
    coverage_checkpoint: Path
    mean_logit_checkpoint: Path
    supcon_checkpoint: Path
    multi_similarity_checkpoint: Path
    complete_receipt: Path

    def __post_init__(self) -> None:
        paths = (
            self.pooled_checkpoint,
            self.coverage_checkpoint,
            self.mean_logit_checkpoint,
            self.supcon_checkpoint,
            self.multi_similarity_checkpoint,
            self.complete_receipt,
        )
        all_paths = (*paths, *(_partial(path) for path in paths))
        if (
            any(not isinstance(path, Path) or not path.is_absolute() for path in paths)
            or len(set(all_paths)) != len(all_paths)
            or any(not path.parent.is_dir() for path in paths)
            or any(path.exists() or _partial(path).exists() for path in paths)
        ):
            raise ValueError("artifact path authority differs")


@dataclass(frozen=True, slots=True)
class ProjectionParameterizationArtifactPaths:
    """Distinct no-clobber paths for the matched projection arms."""

    restricted_checkpoint: Path
    factorized_checkpoint: Path
    direct_checkpoint: Path
    complete_receipt: Path

    def __post_init__(self) -> None:
        paths = (
            self.restricted_checkpoint,
            self.factorized_checkpoint,
            self.direct_checkpoint,
            self.complete_receipt,
        )
        all_paths = (*paths, *(_partial(path) for path in paths))
        if (
            any(not isinstance(path, Path) or not path.is_absolute() for path in paths)
            or len(set(all_paths)) != len(all_paths)
            or any(not path.parent.is_dir() for path in paths)
            or any(path.exists() or _partial(path).exists() for path in paths)
        ):
            raise ValueError("artifact path authority differs")


def _validated_states(
    states: dict[str, dict[str, torch.Tensor]], receipt: dict[str, object]
) -> dict[str, dict[str, torch.Tensor]]:
    if (
        type(states) is not dict
        or tuple(states) != _ARM_NAMES
        or type(receipt) is not dict
        or receipt.get("claim_eligible") is not False
        or type(receipt.get("schema")) is not str
        or not receipt["schema"]
        or type(receipt.get("arms")) is not dict
        or tuple(cast(dict[str, object], receipt["arms"])) != _ARM_NAMES
    ):
        raise ValueError("positive-coverage artifact authority differs")
    arms = cast(dict[str, object], receipt["arms"])
    for name in _ARM_NAMES:
        state = states.get(name)
        arm = arms.get(name)
        state_schema = tuple(state) if type(state) is dict else ()
        self_contained = state_schema == (
            "weight",
            "base_head_weight",
            "base_head_bias",
        )
        if (
            type(state) is not dict
            or state_schema
            not in (
                ("weight",),
                ("weight", "base_head_weight", "base_head_bias"),
            )
            or type(arm) is not dict
            or cast(dict[str, object], arm).get("parameter_sha256")
            != linear_weight_sha256(state["weight"])
            or (
                self_contained
                and cast(dict[str, object], arm).get("base_head_sha256")
                != affine_parameters_sha256(state["base_head_weight"], state["base_head_bias"])
            )
            or (not self_contained and "base_head_sha256" in cast(dict[str, object], arm))
            or "checkpoint" in cast(dict[str, object], arm)
        ):
            raise ValueError("positive-coverage artifact authority differs")
    return states


def _validated_control_states(
    states: dict[str, dict[str, torch.Tensor]], receipt: dict[str, object]
) -> dict[str, dict[str, torch.Tensor]]:
    if (
        type(states) is not dict
        or set(states) != set(_CONTROL_ARM_NAMES)
        or type(receipt) is not dict
        or receipt.get("claim_eligible") is not False
        or receipt.get("schema") != "sfora-matched-loss-controls-v1"
        or type(receipt.get("arms")) is not dict
        or set(cast(dict[str, object], receipt["arms"])) != set(_CONTROL_ARM_NAMES)
    ):
        raise ValueError("matched-loss control artifact authority differs")
    arms = cast(dict[str, object], receipt["arms"])
    for name in _CONTROL_ARM_NAMES:
        state = states.get(name)
        arm = arms.get(name)
        if (
            type(state) is not dict
            or tuple(state) != ("weight",)
            or type(arm) is not dict
            or cast(dict[str, object], arm).get("parameter_sha256")
            != linear_weight_sha256(state["weight"])
            or "checkpoint" in cast(dict[str, object], arm)
        ):
            raise ValueError("matched-loss control artifact authority differs")
    return states


def _validated_projection_states(
    states: dict[str, dict[str, torch.Tensor]], receipt: dict[str, object]
) -> dict[str, dict[str, torch.Tensor]]:
    if (
        type(states) is not dict
        or set(states) != set(_PROJECTION_ARM_NAMES)
        or type(receipt) is not dict
        or receipt.get("claim_eligible") is not False
        or receipt.get("schema") != "sfora-projection-parameterizations-v2"
        or type(receipt.get("arms")) is not dict
        or set(cast(dict[str, object], receipt["arms"])) != set(_PROJECTION_ARM_NAMES)
    ):
        raise ValueError("projection-parameterization artifact authority differs")
    arms = cast(dict[str, object], receipt["arms"])
    restricted = states.get("restricted_adapter")
    direct = states.get("direct_projection")
    factorized = states.get("factorized_adapter")
    restricted_arm = arms.get("restricted_adapter")
    direct_arm = arms.get("direct_projection")
    factorized_arm = arms.get("factorized_adapter")
    if (
        type(restricted) is not dict
        or set(restricted) != {"weight", "base_head_weight", "base_head_bias"}
        or type(direct) is not dict
        or set(direct) != {"weight", "bias"}
        or type(factorized) is not dict
        or set(factorized) != {"weight", "bias"}
        or type(restricted_arm) is not dict
        or type(direct_arm) is not dict
        or type(factorized_arm) is not dict
        or cast(dict[str, object], restricted_arm).get("parameter_sha256")
        != linear_weight_sha256(restricted["weight"])
        or cast(dict[str, object], restricted_arm).get("base_head_sha256")
        != affine_parameters_sha256(restricted["base_head_weight"], restricted["base_head_bias"])
        or cast(dict[str, object], direct_arm).get("parameter_sha256")
        != affine_parameters_sha256(direct["weight"], direct["bias"])
        or cast(dict[str, object], factorized_arm).get("deployed_head_sha256")
        != affine_parameters_sha256(factorized["weight"], factorized["bias"])
        or "checkpoint" in cast(dict[str, object], restricted_arm)
        or "checkpoint" in cast(dict[str, object], direct_arm)
        or "checkpoint" in cast(dict[str, object], factorized_arm)
    ):
        raise ValueError("projection-parameterization artifact authority differs")
    return states


def _partial(path: Path) -> Path:
    return path.with_name(f"{path.name}.partial")


def _owned_partial(path: Path) -> tuple[Path, int, int]:
    status = path.stat(follow_symlinks=False)
    return path, status.st_dev, status.st_ino


def _unlink_owned_partial(owned: tuple[Path, int, int]) -> None:
    path, device, inode = owned
    try:
        status = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        return
    if (status.st_dev, status.st_ino) == (device, inode):
        path.unlink()


def _sync_save(state: dict[str, torch.Tensor], path: Path) -> None:
    try:
        with path.open("xb") as stream:
            try:
                torch.save(state, stream)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
    except FileExistsError:
        raise


def _sync_write(data: bytes, path: Path) -> None:
    try:
        with path.open("xb") as stream:
            try:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            except BaseException:
                path.unlink(missing_ok=True)
                raise
    except FileExistsError:
        raise


def _publish_no_clobber(temporary: Path, destination: Path) -> None:
    os.link(temporary, destination)
    temporary.unlink()
    directory = os.open(destination.parent, os.O_RDONLY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def write_canonical_receipt_no_clobber(receipt: dict[str, object], destination: Path) -> None:
    """Atomically publish one canonical receipt without replacing any owner."""

    if (
        not isinstance(destination, Path)
        or not destination.is_absolute()
        or not destination.parent.is_dir()
        or destination.exists()
        or _partial(destination).exists()
    ):
        raise ValueError("artifact path authority differs")
    temporary = _partial(destination)
    owned: tuple[Path, int, int] | None = None
    try:
        _sync_write(canonical_positive_coverage_receipt_bytes(receipt), temporary)
        owned = _owned_partial(temporary)
        _publish_no_clobber(temporary, destination)
    finally:
        if owned is not None:
            _unlink_owned_partial(owned)


def write_linear_replay_artifacts(
    *,
    state: dict[str, torch.Tensor],
    receipt: dict[str, object],
    checkpoint: Path,
    complete_receipt: Path,
) -> dict[str, object]:
    """Publish one authenticated affine checkpoint before its complete receipt."""

    if (
        type(state) is not dict
        or tuple(state) != ("weight", "bias")
        or type(receipt) is not dict
        or receipt.get("claim_eligible") is not False
        or type(receipt.get("schema")) is not str
        or not receipt["schema"]
        or "checkpoint" in receipt
        or not isinstance(checkpoint, Path)
        or not isinstance(complete_receipt, Path)
    ):
        raise ValueError("positive-coverage artifact authority differs")
    parameter_sha256 = affine_parameters_sha256(state["weight"], state["bias"])
    if receipt.get("final_head_sha256") != parameter_sha256:
        raise ValueError("positive-coverage artifact authority differs")
    outputs = (checkpoint, complete_receipt)
    partials = tuple(_partial(path) for path in outputs)
    if (
        any(not path.is_absolute() or not path.parent.is_dir() for path in outputs)
        or len(set((*outputs, *partials))) != 4
        or any(path.exists() for path in (*outputs, *partials))
    ):
        raise ValueError("artifact path authority differs")
    completed = copy.deepcopy(receipt)
    owned: list[tuple[Path, int, int]] = []
    try:
        checkpoint_partial = _partial(checkpoint)
        _sync_save(state, checkpoint_partial)
        checkpoint_owner = _owned_partial(checkpoint_partial)
        owned.append(checkpoint_owner)
        completed["checkpoint"] = {
            "bytes": checkpoint_partial.stat().st_size,
            "sha256": _file_sha256(checkpoint_partial),
        }
        _publish_no_clobber(checkpoint_partial, checkpoint)
        owned.remove(checkpoint_owner)
        receipt_partial = _partial(complete_receipt)
        _sync_write(canonical_positive_coverage_receipt_bytes(completed), receipt_partial)
        receipt_owner = _owned_partial(receipt_partial)
        owned.append(receipt_owner)
        _publish_no_clobber(receipt_partial, complete_receipt)
        owned.remove(receipt_owner)
    finally:
        for partial in owned:
            _unlink_owned_partial(partial)
    return completed


def write_positive_coverage_artifacts(
    *,
    states: dict[str, dict[str, torch.Tensor]],
    receipt: dict[str, object],
    paths: PositiveCoverageArtifactPaths,
) -> dict[str, object]:
    """Write both authenticated checkpoints before publishing the complete receipt."""

    if type(paths) is not PositiveCoverageArtifactPaths:
        raise ValueError("artifact path authority differs")
    validated = _validated_states(states, receipt)
    canonical_positive_coverage_receipt_bytes(receipt)
    outputs = (paths.pooled_checkpoint, paths.coverage_checkpoint, paths.complete_receipt)
    partials = tuple(_partial(path) for path in outputs)
    if any(path.exists() for path in (*outputs, *partials)):
        raise ValueError("artifact path authority differs")
    completed = copy.deepcopy(receipt)
    completed_arms = cast(dict[str, dict[str, object]], completed["arms"])
    owned_partials: list[tuple[Path, int, int]] = []
    try:
        for name, destination in zip(
            _ARM_NAMES,
            (paths.pooled_checkpoint, paths.coverage_checkpoint),
            strict=True,
        ):
            temporary = _partial(destination)
            _sync_save(validated[name], temporary)
            owner = _owned_partial(temporary)
            owned_partials.append(owner)
            checkpoint = {
                "bytes": temporary.stat().st_size,
                "sha256": _file_sha256(temporary),
            }
            _publish_no_clobber(temporary, destination)
            owned_partials.remove(owner)
            completed_arms[name]["checkpoint"] = checkpoint
        wire = canonical_positive_coverage_receipt_bytes(completed)
        temporary_receipt = _partial(paths.complete_receipt)
        _sync_write(wire, temporary_receipt)
        receipt_owner = _owned_partial(temporary_receipt)
        owned_partials.append(receipt_owner)
        _publish_no_clobber(temporary_receipt, paths.complete_receipt)
        owned_partials.remove(receipt_owner)
    finally:
        for partial in owned_partials:
            _unlink_owned_partial(partial)
    return completed


def write_matched_loss_panel_artifacts(
    *,
    states: dict[str, dict[str, torch.Tensor]],
    receipt: dict[str, object],
    paths: MatchedLossPanelArtifactPaths,
) -> dict[str, object]:
    """Write all matched-loss checkpoints before publishing the receipt."""

    if type(paths) is not MatchedLossPanelArtifactPaths:
        raise ValueError("artifact path authority differs")
    validated = _validated_control_states(states, receipt)
    canonical_positive_coverage_receipt_bytes(receipt)
    destinations = (
        paths.pooled_checkpoint,
        paths.coverage_checkpoint,
        paths.mean_logit_checkpoint,
        paths.supcon_checkpoint,
        paths.multi_similarity_checkpoint,
    )
    outputs = (*destinations, paths.complete_receipt)
    partials = tuple(_partial(path) for path in outputs)
    if any(path.exists() for path in (*outputs, *partials)):
        raise ValueError("artifact path authority differs")
    completed = copy.deepcopy(receipt)
    completed_arms = cast(dict[str, dict[str, object]], completed["arms"])
    owned_partials: list[tuple[Path, int, int]] = []
    try:
        for name, destination in zip(_CONTROL_ARM_NAMES, destinations, strict=True):
            temporary = _partial(destination)
            _sync_save(validated[name], temporary)
            owner = _owned_partial(temporary)
            owned_partials.append(owner)
            completed_arms[name]["checkpoint"] = {
                "bytes": temporary.stat().st_size,
                "sha256": _file_sha256(temporary),
            }
            _publish_no_clobber(temporary, destination)
            owned_partials.remove(owner)
        temporary_receipt = _partial(paths.complete_receipt)
        _sync_write(canonical_positive_coverage_receipt_bytes(completed), temporary_receipt)
        receipt_owner = _owned_partial(temporary_receipt)
        owned_partials.append(receipt_owner)
        _publish_no_clobber(temporary_receipt, paths.complete_receipt)
        owned_partials.remove(receipt_owner)
    finally:
        for partial in owned_partials:
            _unlink_owned_partial(partial)
    return completed


def write_projection_parameterization_artifacts(
    *,
    states: dict[str, dict[str, torch.Tensor]],
    receipt: dict[str, object],
    paths: ProjectionParameterizationArtifactPaths,
) -> dict[str, object]:
    """Publish all projection checkpoints before their complete receipt."""

    if type(paths) is not ProjectionParameterizationArtifactPaths:
        raise ValueError("artifact path authority differs")
    validated = _validated_projection_states(states, receipt)
    canonical_positive_coverage_receipt_bytes(receipt)
    destinations = (
        paths.restricted_checkpoint,
        paths.factorized_checkpoint,
        paths.direct_checkpoint,
    )
    outputs = (*destinations, paths.complete_receipt)
    partials = tuple(_partial(path) for path in outputs)
    if any(path.exists() for path in (*outputs, *partials)):
        raise ValueError("artifact path authority differs")
    completed = copy.deepcopy(receipt)
    completed_arms = cast(dict[str, dict[str, object]], completed["arms"])
    owned_partials: list[tuple[Path, int, int]] = []
    try:
        for name, destination in zip(_PROJECTION_ARM_NAMES, destinations, strict=True):
            temporary = _partial(destination)
            _sync_save(validated[name], temporary)
            owner = _owned_partial(temporary)
            owned_partials.append(owner)
            completed_arms[name]["checkpoint"] = {
                "bytes": temporary.stat().st_size,
                "sha256": _file_sha256(temporary),
            }
            _publish_no_clobber(temporary, destination)
            owned_partials.remove(owner)
        temporary_receipt = _partial(paths.complete_receipt)
        _sync_write(canonical_positive_coverage_receipt_bytes(completed), temporary_receipt)
        receipt_owner = _owned_partial(temporary_receipt)
        owned_partials.append(receipt_owner)
        _publish_no_clobber(temporary_receipt, paths.complete_receipt)
        owned_partials.remove(receipt_owner)
    finally:
        for partial in owned_partials:
            _unlink_owned_partial(partial)
    return completed
