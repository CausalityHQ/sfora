#!/usr/bin/env python3
"""Authenticated local-only SigLIP RSTA Stage-A scientific diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import sys
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sfora.siglip_rsta_stage_a import RstaCheckpointBinding, RstaControlBinding


def _lower_sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be 64 lowercase hexadecimal characters")
    return value


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or os.path.normpath(value) != value:
        raise argparse.ArgumentTypeError("scientific input paths must be normalized absolute paths")
    return path


def parse_stage_a_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the closed local capability surface without touching any input."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--control-binding", required=True, type=_absolute_path)
    parser.add_argument("--control-binding-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--checkpoint-seed17", required=True, type=_absolute_path)
    parser.add_argument("--checkpoint-seed29", required=True, type=_absolute_path)
    parser.add_argument("--checkpoint-seed43", required=True, type=_absolute_path)
    parser.add_argument("--optimization-manifest", required=True, type=_absolute_path)
    parser.add_argument("--optimization-manifest-sha256", required=True, type=_lower_sha256)
    parser.add_argument("--image-root", required=True, type=_absolute_path)
    parser.add_argument("--execute-stage-a", action="store_true", required=True)
    effective = list(sys.argv[1:] if argv is None else argv)
    flags = [value.split("=", 1)[0] for value in effective if value.startswith("--")]
    duplicates = sorted({flag for flag in flags if flags.count(flag) > 1})
    if duplicates:
        parser.error(f"duplicate arguments are forbidden: {duplicates!r}")
    return parser.parse_args(effective)


@dataclass(frozen=True)
class LoadedStageACheckpoint:
    """One authenticated final checkpoint projected to model state only."""

    seed: int
    model_state: Mapping[str, object]


@dataclass(frozen=True)
class LoadedStageAAuthority:
    """Outcome-blind local authority available to the scientific loop."""

    binding: RstaControlBinding
    checkpoints: tuple[LoadedStageACheckpoint, ...]
    example_ids: tuple[str, ...]
    labels: tuple[int, ...]
    image_paths: tuple[Path, ...]


def _read_regular(path: Path, *, role: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"{role} must be one regular file")
    return path.read_bytes()


def _canonical_json(value: dict[str, object]) -> bytes:
    from sfora.pass209_m4 import canonical_json_bytes

    return canonical_json_bytes(value)


def _image_basename(example_id: str) -> str:
    digest = hashlib.sha256(
        b"rsta-siglip-a-v1|image-path|\0" + example_id.encode("utf-8")
    ).hexdigest()
    return f"{digest}.image"


def _parse_control_binding(raw: bytes) -> RstaControlBinding:
    from sfora.siglip_rsta_stage_a import RstaCheckpointBinding, RstaControlBinding

    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RSTA control binding is not valid JSON") from error
    expected = {
        "schema",
        "claim_eligible",
        "control_complete",
        "source_commit",
        "config_sha256",
        "run_authority_sha256",
        "dataset_id",
        "dataset_revision",
        "environment_sha256",
        "optimization_manifest_sha256",
        "checkpoints",
    }
    if type(value) is not dict or set(value) != expected or raw != _canonical_json(value):
        raise ValueError("RSTA control binding authority differs")
    checkpoints = value["checkpoints"]
    if type(checkpoints) is not list or any(
        type(item) is not dict or set(item) != {"seed", "sha256", "byte_length"}
        for item in checkpoints
    ):
        raise ValueError("RSTA control binding checkpoint schema differs")
    try:
        return RstaControlBinding(
            schema=value["schema"],
            claim_eligible=value["claim_eligible"],
            control_complete=value["control_complete"],
            source_commit=value["source_commit"],
            config_sha256=value["config_sha256"],
            run_authority_sha256=value["run_authority_sha256"],
            dataset_id=value["dataset_id"],
            dataset_revision=value["dataset_revision"],
            environment_sha256=value["environment_sha256"],
            optimization_manifest_sha256=value["optimization_manifest_sha256"],
            checkpoints=tuple(
                RstaCheckpointBinding(
                    seed=item["seed"],
                    sha256=item["sha256"],
                    byte_length=item["byte_length"],
                )
                for item in checkpoints
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("RSTA control binding authority differs") from error


def _load_model_state_checkpoint(
    path: Path,
    authority: RstaCheckpointBinding,
    binding: RstaControlBinding,
) -> LoadedStageACheckpoint:
    import numpy as np
    import torch

    raw = _read_regular(path, role="RSTA checkpoint")
    if len(raw) != authority.byte_length or hashlib.sha256(raw).hexdigest() != authority.sha256:
        raise ValueError("RSTA checkpoint digest or length differs")

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cpu_state = torch.random.get_rng_state().clone()
    cuda_states = tuple(state.clone() for state in torch.cuda.get_rng_state_all())
    try:
        payload = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(list(cuda_states))

    expected = {
        "claim_eligible",
        "completed_epoch",
        "config_sha256",
        "cpu_rng_state",
        "cuda_rng_states",
        "final_objective",
        "initial_snapshot_sha256",
        "maximum_score_disagreement",
        "model_state",
        "optimizer_state",
        "run_authority_sha256",
        "sampler_cycles",
        "sampler_positions",
        "schema",
        "seed",
    }
    if type(payload) is not dict or set(payload) != expected:
        raise ValueError("RSTA checkpoint payload schema differs")
    model_state = payload["model_state"]
    cycles = payload["sampler_cycles"]
    positions = payload["sampler_positions"]
    objective = payload["final_objective"]
    disagreement = payload["maximum_score_disagreement"]
    cpu_rng_state = payload["cpu_rng_state"]
    checkpoint_cuda_states = payload["cuda_rng_states"]
    if (
        payload["schema"] != "sfora-siglip-proxy-checkpoint-payload-v1"
        or payload["claim_eligible"] is not False
        or type(payload["seed"]) is not int
        or payload["seed"] != authority.seed
        or type(payload["completed_epoch"]) is not int
        or payload["completed_epoch"] != 60
        or payload["config_sha256"] != binding.config_sha256
        or payload["run_authority_sha256"] != binding.run_authority_sha256
        or type(objective) is not float
        or not math.isfinite(objective)
        or type(disagreement) is not float
        or not math.isfinite(disagreement)
        or disagreement < 0.0
        or type(payload["initial_snapshot_sha256"]) is not str
        or len(payload["initial_snapshot_sha256"]) != 64
        or any(
            character not in "0123456789abcdef" for character in payload["initial_snapshot_sha256"]
        )
        or type(cycles) is not tuple
        or type(positions) is not tuple
        or len(cycles) != 49
        or len(positions) != 49
        or any(type(value) is not int or value < 0 for value in cycles + positions)
        or not isinstance(cpu_rng_state, torch.Tensor)
        or cpu_rng_state.dtype != torch.uint8
        or type(checkpoint_cuda_states) is not tuple
        or any(
            not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
            for value in checkpoint_cuda_states
        )
        or type(payload["optimizer_state"]) is not dict
        or type(model_state) is not OrderedDict
        or not model_state
        or any(type(name) is not str or not name for name in model_state)
        or any(not isinstance(tensor, torch.Tensor) for tensor in model_state.values())
    ):
        raise ValueError("RSTA checkpoint authority differs")
    return LoadedStageACheckpoint(
        seed=authority.seed,
        model_state=MappingProxyType(dict(model_state)),
    )


def _load_optimization_manifest(
    path: Path,
    expected_sha256: str,
    binding: RstaControlBinding,
    image_root: Path,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[Path, ...]]:
    raw = _read_regular(path, role="RSTA optimization manifest")
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("RSTA optimization manifest digest differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("RSTA optimization manifest is not valid JSON") from error
    if (
        type(value) is not dict
        or set(value) != {"schema", "claim_eligible", "dataset_id", "dataset_revision", "examples"}
        or raw != _canonical_json(value)
        or value["schema"] != "rsta-optimization-manifest-v1"
        or value["claim_eligible"] is not False
        or value["dataset_id"] != binding.dataset_id
        or value["dataset_revision"] != binding.dataset_revision
        or type(value["examples"]) is not list
        or not value["examples"]
    ):
        raise ValueError("RSTA optimization manifest authority differs")
    try:
        resolved_root = image_root.resolve(strict=True)
    except OSError as error:
        raise ValueError("RSTA image root must be one real directory") from error
    if image_root.is_symlink() or not image_root.is_dir() or resolved_root != image_root:
        raise ValueError("RSTA image root must be one real directory")
    example_ids: list[str] = []
    labels: list[int] = []
    image_paths: list[Path] = []
    expected_basenames: set[str] = set()
    for row in value["examples"]:
        if (
            type(row) is not dict
            or set(row) != {"example_id", "label"}
            or type(row["example_id"]) is not str
            or not row["example_id"]
            or type(row["label"]) is not int
            or not 0 <= row["label"] < 49
        ):
            raise ValueError("RSTA optimization manifest row differs")
        basename = _image_basename(row["example_id"])
        image_path = image_root / basename
        try:
            resolved_image = image_path.resolve(strict=True)
        except OSError as error:
            raise ValueError("RSTA optimization image must be one regular file") from error
        if not resolved_image.is_relative_to(resolved_root) or resolved_image != image_path:
            raise ValueError("RSTA optimization image path escapes authority")
        if image_path.is_symlink() or not image_path.is_file():
            raise ValueError("RSTA optimization image must be one regular file")
        example_ids.append(row["example_id"])
        labels.append(row["label"])
        image_paths.append(image_path)
        expected_basenames.add(basename)
    if len(set(example_ids)) != len(example_ids):
        raise ValueError("RSTA optimization example identities are duplicated")
    observed_basenames = {path.name for path in image_root.iterdir()}
    if observed_basenames != expected_basenames:
        raise ValueError("RSTA image namespace differs from optimization authority")
    return tuple(example_ids), tuple(labels), tuple(image_paths)


def load_stage_a_authority(arguments: argparse.Namespace) -> LoadedStageAAuthority:
    """Authenticate local evidence and retain no optimizer or outcome artifacts."""

    binding_raw = _read_regular(arguments.control_binding, role="RSTA control binding")
    if hashlib.sha256(binding_raw).hexdigest() != arguments.control_binding_sha256:
        raise ValueError("RSTA control binding digest differs")
    binding = _parse_control_binding(binding_raw)
    if arguments.optimization_manifest_sha256 != binding.optimization_manifest_sha256:
        raise ValueError("RSTA optimization manifest digest differs")
    checkpoint_paths = (
        arguments.checkpoint_seed17,
        arguments.checkpoint_seed29,
        arguments.checkpoint_seed43,
    )
    if len(set(checkpoint_paths)) != 3:
        raise ValueError("RSTA checkpoint paths must be distinct")
    checkpoints = tuple(
        _load_model_state_checkpoint(path, authority, binding)
        for path, authority in zip(checkpoint_paths, binding.checkpoints, strict=True)
    )
    example_ids, labels, image_paths = _load_optimization_manifest(
        arguments.optimization_manifest,
        arguments.optimization_manifest_sha256,
        binding,
        arguments.image_root,
    )
    return LoadedStageAAuthority(
        binding=binding,
        checkpoints=checkpoints,
        example_ids=example_ids,
        labels=labels,
        image_paths=image_paths,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Validate arguments; scientific execution is added behind this boundary."""

    parse_stage_a_args(argv)
    raise RuntimeError("SigLIP RSTA Stage-A scientific runner is not implemented")


if __name__ == "__main__":
    raise SystemExit(main())
