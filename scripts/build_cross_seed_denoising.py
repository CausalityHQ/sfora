#!/usr/bin/env python3
"""Build fixed cross-seed candidate towers from outcome-free local artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import cast

import torch

from sfora.cross_seed_denoising import (
    CandidateStates,
    build_cross_seed_candidates,
    read_tensor_artifact,
    write_tensor_artifact,
)

_SEEDS = (17, 29, 43)
_ROLES = ("tower-soup", "wiener-denoise", "spectral-denoise")
_RSS_CAP = 110 * 1024**3


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def _absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        raise argparse.ArgumentTypeError("path must be absolute")
    return path


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise argparse.ArgumentTypeError("digest must be lowercase SHA-256")
    return value


def _positive(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("byte length must be an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("byte length must be positive")
    return parsed


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the CPU-only, outcome-free candidate builder interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-root", required=True, type=_absolute_path)
    parser.add_argument("--prepared-manifest", required=True, type=_absolute_path)
    parser.add_argument("--prepared-manifest-sha256", required=True, type=_sha256)
    parser.add_argument("--prepared-manifest-bytes", required=True, type=_positive)
    parser.add_argument("--output", required=True, type=_absolute_path)
    parser.add_argument("--execute-cross-seed-builder", required=True, action="store_true")
    return parser.parse_args(argv)


def _read_manifest(prepared_root: Path, raw: bytes) -> dict[str, object]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("prepared manifest is not valid JSON") from exc
    keys = {"bindings", "claim_eligible", "initial_tower", "schema", "seeds"}
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError("prepared manifest is not canonical")
    if (
        set(value) != keys
        or value["schema"] != "sfora-cross-seed-prepared-inputs-v1"
        or value["claim_eligible"] is not False
    ):
        raise ValueError("prepared manifest schema differs")
    path = prepared_root / "manifest.json"
    if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
        raise ValueError("prepared manifest file differs")
    if type(value["bindings"]) is not dict or not value["bindings"]:
        raise ValueError("prepared manifest bindings differ")
    bindings = cast(dict[object, object], value["bindings"])
    if any(type(key) is not str or type(item) is not str for key, item in bindings.items()):
        raise ValueError("prepared manifest bindings differ")
    return cast(dict[str, object], value)


def _artifact(
    prepared_root: Path,
    row: object,
    *,
    directory_key: str,
    manifest_prefix: str,
    role: str,
) -> OrderedDict[str, torch.Tensor]:
    required = {
        directory_key,
        f"{manifest_prefix}_manifest_bytes",
        f"{manifest_prefix}_manifest_sha256",
        f"{manifest_prefix}_state_sha256",
    }
    if type(row) is not dict or not required.issubset(row):
        raise ValueError("prepared artifact row differs")
    typed = cast(dict[str, object], row)
    directory = typed[directory_key]
    if type(directory) is not str or Path(directory).name != directory:
        raise ValueError("prepared artifact path differs")
    root = prepared_root / directory
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("prepared artifact manifest differs")
    raw = manifest_path.read_bytes()
    if (
        type(typed[f"{manifest_prefix}_manifest_bytes"]) is not int
        or typed[f"{manifest_prefix}_manifest_bytes"] != len(raw)
        or typed[f"{manifest_prefix}_manifest_sha256"] != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("prepared artifact manifest identity differs")
    manifest = json.loads(raw)
    if typed[f"{manifest_prefix}_state_sha256"] != manifest.get("state_sha256"):
        raise ValueError("prepared artifact state binding differs")
    return read_tensor_artifact(root, raw, role=role)


def _load_inputs(
    prepared_root: Path, raw: bytes
) -> tuple[OrderedDict[str, torch.Tensor], dict[int, OrderedDict[str, torch.Tensor]]]:
    value = _read_manifest(prepared_root, raw)
    initial_row = value["initial_tower"]
    if type(initial_row) is not dict:
        raise ValueError("initial tower row differs")
    initial_typed = cast(dict[str, object], initial_row)
    initial = _artifact(
        prepared_root,
        {
            "directory": initial_typed.get("directory"),
            "initial_manifest_bytes": initial_typed.get("manifest_bytes"),
            "initial_manifest_sha256": initial_typed.get("manifest_sha256"),
            "initial_state_sha256": initial_typed.get("state_sha256"),
        },
        directory_key="directory",
        manifest_prefix="initial",
        role="initial-tower",
    )
    if type(value["seeds"]) is not list or len(value["seeds"]) != 3:
        raise ValueError("prepared seed rows differ")
    endpoints: dict[int, OrderedDict[str, torch.Tensor]] = {}
    expected_namespace = {"manifest.json", "initial-tower"}
    for raw_row, seed in zip(cast(list[object], value["seeds"]), _SEEDS, strict=True):
        if type(raw_row) is not dict or raw_row.get("seed") != seed:
            raise ValueError("prepared seed order differs")
        row = cast(dict[str, object], raw_row)
        tower = _artifact(
            prepared_root,
            row,
            directory_key="tower_directory",
            manifest_prefix="tower",
            role="trained-tower",
        )
        endpoints[seed] = tower
        tower_directory = row["tower_directory"]
        head_directory = row.get("head_directory")
        if type(tower_directory) is not str or type(head_directory) is not str:
            raise ValueError("prepared seed artifact paths differ")
        expected_namespace.update((tower_directory, head_directory))
    actual_namespace = {path.name for path in prepared_root.iterdir()}
    if actual_namespace != expected_namespace or any(
        path.is_symlink() for path in prepared_root.iterdir()
    ):
        raise ValueError("prepared namespace differs")
    return initial, endpoints


def project_builder_peak_bytes(
    initial: object,
    endpoints: object,
) -> int:
    """Project the registered seven-state plus one-tensor decomposition peak."""

    if not isinstance(initial, OrderedDict) or type(endpoints) is not dict:
        raise ValueError("builder projection states differ")
    total = 0
    largest_float64 = 0
    for tensor in initial.values():
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("builder projection tensor differs")
        byte_count = tensor.numel() * tensor.element_size()
        total += byte_count
        if tensor.is_floating_point():
            largest_float64 = max(largest_float64, tensor.numel() * 8)
    endpoint_map = cast(dict[object, object], endpoints)
    if set(endpoint_map) != set(_SEEDS):
        raise ValueError("builder projection seeds differ")
    # W0, three task vectors, and three accumulating candidate states.
    return 7 * total + 4 * largest_float64


def _states_equal(left: CandidateStates, right: CandidateStates) -> bool:
    if left.groups != right.groups or left.spectral != right.spectral:
        return False
    for role in _ROLES:
        attribute = role.replace("-", "_")
        left_state = cast(OrderedDict[str, torch.Tensor], getattr(left, attribute))
        right_state = cast(OrderedDict[str, torch.Tensor], getattr(right, attribute))
        if tuple(left_state) != tuple(right_state) or any(
            not torch.equal(left_state[name], right_state[name]) for name in left_state
        ):
            return False
    return True


def _evidence_payload(candidate: CandidateStates) -> dict[str, object]:
    return {
        "groups": [asdict(row) for row in candidate.groups],
        "spectral": [asdict(row) for row in candidate.spectral],
    }


def build_candidate_artifacts(
    *,
    prepared_root: Path,
    prepared_manifest_raw: bytes,
    output: Path,
) -> bytes:
    """Authenticate, build twice, and atomically publish all three candidates."""

    if not isinstance(prepared_root, Path) or not isinstance(output, Path):
        raise TypeError("builder paths differ")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    initial, endpoints = _load_inputs(prepared_root, prepared_manifest_raw)
    projected_peak = project_builder_peak_bytes(initial, endpoints)
    if projected_peak >= _RSS_CAP:
        raise ValueError("builder projected RSS exceeds authority")
    first = build_cross_seed_candidates(initial, endpoints)
    replay_initial, replay_endpoints = _load_inputs(prepared_root, prepared_manifest_raw)
    replay = build_cross_seed_candidates(replay_initial, replay_endpoints)
    if not _states_equal(first, replay):
        raise ValueError("candidate determinism replay differs")

    partial = output.with_name(f".{output.name}.partial-{os.getpid()}")
    if partial.exists() or partial.is_symlink():
        raise FileExistsError(partial)
    partial.mkdir()
    try:
        prepared_sha256 = hashlib.sha256(prepared_manifest_raw).hexdigest()
        candidate_rows: list[dict[str, object]] = []
        for role in _ROLES:
            directory = role
            state = cast(
                OrderedDict[str, torch.Tensor], getattr(first, role.replace("-", "_"))
            )
            raw = write_tensor_artifact(
                partial / directory,
                state,
                role=role,
                bindings={"prepared_manifest_sha256": prepared_sha256},
            )
            manifest = json.loads(raw)
            candidate_rows.append(
                {
                    "directory": directory,
                    "manifest_bytes": len(raw),
                    "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                    "role": role,
                    "state_sha256": manifest["state_sha256"],
                }
            )
        receipt = {
            "candidates": candidate_rows,
            "claim_eligible": False,
            "construction_evidence": _evidence_payload(first),
            "determinism_replay": True,
            "prepared_manifest_bytes": len(prepared_manifest_raw),
            "prepared_manifest_sha256": prepared_sha256,
            "projected_peak_rss_bytes": projected_peak,
            "schema": "sfora-cross-seed-candidate-receipt-v1",
        }
        raw = _canonical(receipt)
        with (partial / "receipt.json").open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        partial.rename(output)
        return raw
    except BaseException:
        if partial.is_dir() and not partial.is_symlink():
            shutil.rmtree(partial)
        raise


def main(argv: Sequence[str] | None = None) -> int:
    """Run the authenticated local CPU builder and publish one receipt."""

    arguments = parse_arguments(argv)
    if (
        arguments.prepared_manifest.is_symlink()
        or not arguments.prepared_manifest.is_file()
        or arguments.prepared_manifest.parent != arguments.prepared_root
    ):
        raise ValueError("prepared manifest path differs")
    raw = arguments.prepared_manifest.read_bytes()
    if (
        len(raw) != arguments.prepared_manifest_bytes
        or hashlib.sha256(raw).hexdigest() != arguments.prepared_manifest_sha256
    ):
        raise ValueError("prepared manifest identity differs")
    receipt = build_candidate_artifacts(
        prepared_root=arguments.prepared_root,
        prepared_manifest_raw=raw,
        output=arguments.output,
    )
    sys.stdout.buffer.write(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
