#!/usr/bin/env python3
"""Build and authenticate the immutable UniCOM FEPF campaign config."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

UNICOM_REVISION = "d71992ed969e6c271436ac0a0ee1f3ca61474ac0"
UNICOM_CHECKPOINT_SHA256 = "3916ab5aed3b522fc90345be8b4457fe5dad60801ad2af5a6871c0c096e8d7ea"
INSHOP_PARTITION_SHA256 = "cfada103c44df866db5e2ee9ecc2301ca691a4d0cdb3c875fe4051b62570894c"
RUNTIME_ORDER = ("current", "composed", "composed", "current") * 2
CONFIRMATION_PAIRS = (
    (7, 20_260_828),
    (8, 271_828),
    (9, 314_159),
    (10, 1_618_033),
    (11, 57_721),
)
REGISTERED_SOURCE_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "src/sfora/unicom_fepf.py",
    "src/sfora/unicom_retrieval_audit.py",
    "scripts/train_unicom_inshop.py",
    "scripts/profile_unicom_training_step.py",
    "scripts/evaluate_unicom_fepf.py",
    "scripts/build_unicom_fepf_run_config.py",
    "scripts/run_unicom_fepf_campaign.py",
    "scripts/run_unicom_fepf_cuda_canary.py",
)
CONFIG_KEYS = (
    "schema",
    "source_commit",
    "source_files",
    "model",
    "inputs",
    "checkout_root_template",
    "artifact_root",
    "runtime_order",
    "exploratory",
    "confirmation_pairs",
    "thresholds",
    "artifact_budget_inputs",
    "artifact_budget_bytes",
    "artifact_budget_inodes",
    "cuda_canary_command",
    "cuda_canary_receipt",
    "commands",
)
_LOWER_COMMIT = re.compile(r"[0-9a-f]{40}").fullmatch

# Frozen conservative sizing authorities. The eightfold multiplier covers the
# complete checkpoint payload rather than pretending state_dict tensor bytes are
# the serialized checkpoint size.
RAW_BACKBONE_STATE_BYTES = 1_218_000_000
CLASSIFIER_STATE_BYTES = 3_997 * 768 * 4
QUALITY_CHECKPOINTS = 13 * 4
QUERY_ROWS = 14_218
GALLERY_ROWS = 12_612
DESCRIPTOR_BYTES = (QUERY_ROWS + GALLERY_ROWS) * 768 * 4
ATOMIC_COPY_FACTOR = 2
PLANNED_NONCHECKPOINT_BYTES = 16 * 1024**3
PLANNED_FILE_INODES = 2_048


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")


def _git(repo: Path, *arguments: str, binary: bool = False) -> str | bytes:
    result = subprocess.run(
        ["git", *arguments], cwd=repo, check=True, capture_output=True,
        text=not binary,
    )
    return result.stdout


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_absolute_plain(path: Path, name: str) -> Path:
    if not path.is_absolute() or path != Path(os.path.normpath(path)):
        raise ValueError(f"{name} must be a normalized absolute path")
    return path


def _validate_template(template: str) -> None:
    if type(template) is not str or template.count("{config_commit}") != 1:
        raise ValueError("checkout root template differs")
    residual = template.replace("{config_commit}", "")
    if "{" in residual or "}" in residual:
        raise ValueError("checkout root template differs")
    expanded = Path(template.replace("{config_commit}", "0" * 40))
    _require_absolute_plain(expanded, "checkout root template")


def _paths_nested(left: Path, right: Path) -> bool:
    return left == right or left in right.parents or right in left.parents


def _source_inventory(repo: Path, commit: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for relative in REGISTERED_SOURCE_PATHS:
        payload = _git(repo, "show", f"{commit}:{relative}", binary=True)
        assert isinstance(payload, bytes)
        rows.append({"path": relative, "sha256": _sha256(payload), "bytes": len(payload)})
    return rows


def _budget() -> tuple[dict[str, int], int, int]:
    checkpoint_bound = 8 * (RAW_BACKBONE_STATE_BYTES + CLASSIFIER_STATE_BYTES) + 64 * 1024**2
    checkpoint_bytes = QUALITY_CHECKPOINTS * checkpoint_bound
    descriptor_bytes = 13 * 4 * DESCRIPTOR_BYTES
    subtotal = ATOMIC_COPY_FACTOR * (
        checkpoint_bytes + descriptor_bytes + PLANNED_NONCHECKPOINT_BYTES
    )
    inputs = {
        "quality_checkpoints": QUALITY_CHECKPOINTS,
        "raw_backbone_state_bytes": RAW_BACKBONE_STATE_BYTES,
        "classifier_state_bytes": CLASSIFIER_STATE_BYTES,
        "query_rows": QUERY_ROWS,
        "gallery_rows": GALLERY_ROWS,
        "descriptor_dimension": 768,
        "atomic_copy_factor": ATOMIC_COPY_FACTOR,
        "planned_file_inodes": PLANNED_FILE_INODES,
    }
    return inputs, math.ceil(1.25 * subtotal), math.ceil(1.25 * PLANNED_FILE_INODES)


def _inputs() -> dict[str, str]:
    return {
        "unicom_checkout": "/home/riomus/unicom-d71992e",
        "checkpoint": "/home/riomus/.cache/unicom/FP16-ViT-L-14-336px.pt",
        "dataset_root": "/home/riomus/datasets/inshop_official_standard",
        "partition": "/home/riomus/datasets/inshop_official_standard/Eval/list_eval_partition.txt",
        "runtime_checkpoint": "/home/riomus/unicom-ema-imprinted-387d697-seed2-e16/epoch-0016.pt",
        "runtime_run_receipt": (
            "/home/riomus/unicom-ema-imprinted-387d697-seed2-e16/run-receipt.json"
        ),
    }


def _commands() -> dict[str, object]:
    python = ".venv/bin/python"
    inputs = _inputs()
    return {
        "runtime": [
            [python, "-I", "-B", "scripts/profile_unicom_training_step.py",
             "--profile-kind", "runtime", "--runtime-mode", mode,
             "--checkpoint", inputs["runtime_checkpoint"], "--run-receipt",
             inputs["runtime_run_receipt"], "--config",
             "docs/unicom_fepf_run_config.json"]
            for mode in RUNTIME_ORDER
        ],
        "train": [
            python, "-I", "-B", "scripts/train_unicom_inshop.py",
            "--unicom-checkout", inputs["unicom_checkout"],
            "--checkpoint", inputs["checkpoint"],
            "--dataset-root", inputs["dataset_root"],
            "--run-config", "docs/unicom_fepf_run_config.json",
        ],
        "profile_quality": [
            python, "-I", "-B", "scripts/profile_unicom_training_step.py",
            "--profile-kind", "quality",
        ],
        "evaluate": [python, "-I", "-B", "scripts/evaluate_unicom_fepf.py"],
        "quality_profile_order": ["imprinted", "fepf_mean", "fepf_mean", "imprinted"],
    }


def build_run_config(
    *, repo: Path, checkout_root_template: str, artifact_root: Path
) -> dict[str, object]:
    repo = _require_absolute_plain(repo.resolve(), "repository")
    artifact_root = _require_absolute_plain(artifact_root, "artifact root")
    _validate_template(checkout_root_template)
    expanded = Path(checkout_root_template.replace("{config_commit}", "0" * 40))
    if _paths_nested(expanded, artifact_root):
        raise ValueError("checkout and artifact roots must be distinct and non-nested")
    source_commit = str(_git(repo, "rev-parse", "HEAD")).strip()
    if _LOWER_COMMIT(source_commit) is None:
        raise ValueError("source commit differs")
    budget_inputs, budget_bytes, budget_inodes = _budget()
    return {
        "schema": "unicom-fepf-run-config-v1",
        "source_commit": source_commit,
        "source_files": _source_inventory(repo, source_commit),
        "model": {
            "revision": UNICOM_REVISION,
            "checkpoint_sha256": UNICOM_CHECKPOINT_SHA256,
            "partition_sha256": INSHOP_PARTITION_SHA256,
        },
        "inputs": _inputs(),
        "checkout_root_template": checkout_root_template,
        "artifact_root": str(artifact_root),
        "runtime_order": list(RUNTIME_ORDER),
        "exploratory": {
            "training_seed": 0,
            "holdout_seed": 0,
            "holdout_fraction": 0.2,
            "arms": ["imprinted", "fepf_mean", "fepf_random"],
            "evaluation_epochs": [4, 8, 12, 16],
        },
        "confirmation_pairs": [list(pair) for pair in CONFIRMATION_PAIRS],
        "thresholds": {
            "epoch4_map_delta_min": 0.003,
            "exploratory_map_delta_min": 0.010,
            "confirmation_map_delta_min": 0.010,
            "confirmation_r1_delta_min": 0.005,
            "compute_ratio_max": 1.02,
            "row_norm_rtol": 2e-6,
            "row_norm_atol": 2e-7,
        },
        "artifact_budget_inputs": budget_inputs,
        "artifact_budget_bytes": budget_bytes,
        "artifact_budget_inodes": budget_inodes,
        "cuda_canary_command": [
            ".venv/bin/python", "-I", "-B", "scripts/run_unicom_fepf_cuda_canary.py",
            "--config", "docs/unicom_fepf_run_config.json",
        ],
        "cuda_canary_receipt": "preflight/cuda_canary_v1.json",
        "commands": _commands(),
    }


def _strict_shape(config: object) -> dict[str, object]:
    if type(config) is not dict or tuple(config) != CONFIG_KEYS:
        raise ValueError("FEPF config schema differs")
    return config


def _validate_config_values(config: dict[str, object]) -> None:
    if (
        config["schema"] != "unicom-fepf-run-config-v1"
        or _LOWER_COMMIT(config["source_commit"]) is None
        or config["model"] != {
            "revision": UNICOM_REVISION,
            "checkpoint_sha256": UNICOM_CHECKPOINT_SHA256,
            "partition_sha256": INSHOP_PARTITION_SHA256,
        }
        or config["inputs"] != _inputs()
        or config["runtime_order"] != list(RUNTIME_ORDER)
        or config["confirmation_pairs"] != [list(pair) for pair in CONFIRMATION_PAIRS]
        or type(config["artifact_budget_bytes"]) is not int
        or config["artifact_budget_bytes"] <= 0
        or type(config["artifact_budget_inodes"]) is not int
        or config["artifact_budget_inodes"] <= 0
    ):
        raise ValueError("FEPF config protocol differs")
    threshold = config["thresholds"]
    if type(threshold) is not dict or threshold != {
        "epoch4_map_delta_min": 0.003,
        "exploratory_map_delta_min": 0.010,
        "confirmation_map_delta_min": 0.010,
        "confirmation_r1_delta_min": 0.005,
        "compute_ratio_max": 1.02,
        "row_norm_rtol": 2e-6,
        "row_norm_atol": 2e-7,
    }:
        raise ValueError("FEPF config threshold differs")
    if config["exploratory"] != {
        "training_seed": 0, "holdout_seed": 0, "holdout_fraction": 0.2,
        "arms": ["imprinted", "fepf_mean", "fepf_random"],
        "evaluation_epochs": [4, 8, 12, 16],
    } or config["commands"] != _commands():
        raise ValueError("FEPF config commands differ")
    inputs, budget_bytes, budget_inodes = _budget()
    if (
        config["artifact_budget_inputs"] != inputs
        or config["artifact_budget_bytes"] != budget_bytes
        or config["artifact_budget_inodes"] != budget_inodes
        or config["cuda_canary_command"]
        != [".venv/bin/python", "-I", "-B", "scripts/run_unicom_fepf_cuda_canary.py",
            "--config", "docs/unicom_fepf_run_config.json"]
        or config["cuda_canary_receipt"] != "preflight/cuda_canary_v1.json"
    ):
        raise ValueError("FEPF config budget differs")
    _validate_template(config["checkout_root_template"])
    artifact = _require_absolute_plain(Path(config["artifact_root"]), "artifact root")
    checkout = Path(config["checkout_root_template"].replace("{config_commit}", "0" * 40))
    if _paths_nested(artifact, checkout):
        raise ValueError("checkout and artifact roots must be distinct and non-nested")


def _require_clean(repo: Path) -> None:
    status_text = str(_git(repo, "status", "--porcelain", "--untracked-files=no"))
    if status_text:
        raise ValueError("source checkout is not clean")


def validate_config_build(config: object, repo: Path) -> None:
    value = _strict_shape(config)
    _validate_config_values(value)
    repo = repo.resolve()
    _require_clean(repo)
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    if head != value["source_commit"]:
        raise ValueError("source commit differs")
    if value["source_files"] != _source_inventory(repo, head):
        raise ValueError("source files differ")
    artifact = Path(value["artifact_root"])
    checkout = Path(value["checkout_root_template"].replace("{config_commit}", "0" * 40))
    if os.path.lexists(artifact) or os.path.lexists(checkout):
        raise FileExistsError("campaign destination already exists")


def build_and_write(
    *, repo: Path, checkout_root_template: str, artifact_root: Path, output: Path
) -> dict[str, object]:
    if os.path.lexists(output):
        raise FileExistsError(output)
    config = build_run_config(
        repo=repo, checkout_root_template=checkout_root_template,
        artifact_root=artifact_root,
    )
    validate_config_build(config, repo)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as handle:
        handle.write(canonical_json_bytes(config))
        handle.flush()
        os.fsync(handle.fileno())
    reloaded = json.loads(output.read_bytes())
    if canonical_json_bytes(reloaded) != output.read_bytes() or reloaded != config:
        raise RuntimeError("persisted FEPF config differs")
    return config


def validate_config_handoff(config_path: Path, repo: Path) -> dict[str, str]:
    raw = config_path.read_bytes()
    config = _strict_shape(json.loads(raw))
    if canonical_json_bytes(config) != raw:
        raise ValueError("FEPF config is not canonical")
    _validate_config_values(config)
    repo = repo.resolve()
    _require_clean(repo)
    head = str(_git(repo, "rev-parse", "HEAD")).strip()
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"], cwd=repo,
        check=False, capture_output=True, text=True,
    ).stdout.strip()
    if _LOWER_COMMIT(head) is None or symbolic:
        raise ValueError("handoff checkout must be detached")
    parent = str(_git(repo, "rev-parse", f"{head}^")).strip()
    relative = config_path.resolve().relative_to(repo).as_posix()
    changed = str(
        _git(repo, "diff-tree", "--no-commit-id", "--name-only", "-r", head)
    ).splitlines()
    committed = _git(repo, "show", f"{head}:{relative}", binary=True)
    if (
        parent != config["source_commit"]
        or changed != [relative]
        or committed != raw
        or config["source_files"] != _source_inventory(repo, parent)
    ):
        raise ValueError("config handoff commit differs")
    checkout = config["checkout_root_template"].replace("{config_commit}", head)
    if "{" in checkout or "}" in checkout or os.path.lexists(checkout):
        raise ValueError("resolved checkout root differs")
    if os.path.lexists(config["artifact_root"]):
        raise FileExistsError(config["artifact_root"])
    return {"config_commit": head, "checkout_root": checkout}


def _available(stat_result: Any) -> tuple[int, int]:
    return stat_result.f_bavail * stat_result.f_frsize, stat_result.f_favail


def _plain_directory(path: Path, name: str) -> os.stat_result:
    info = path.lstat()
    if not stat.S_ISDIR(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise ValueError(f"{name} is not a plain directory")
    return info


def _require_capacity(
    path: Path, required_bytes: int, required_inodes: int,
    statvfs: Callable[[Path], Any],
) -> None:
    available_bytes, available_inodes = _available(statvfs(path))
    if available_bytes < required_bytes or available_inodes < required_inodes:
        raise OSError("FEPF artifact capacity is insufficient")


def prepare_artifact_root(
    root: Path, *, required_bytes: int, required_inodes: int,
    statvfs: Callable[[Path], Any] = os.statvfs,
) -> None:
    root = _require_absolute_plain(root, "artifact root")
    if type(required_bytes) is not int or type(required_inodes) is not int:
        raise ValueError("artifact budget differs")
    if os.path.lexists(root):
        raise FileExistsError(root)
    parent = root.parent
    try:
        parent_info = _plain_directory(parent, "artifact parent")
    except FileNotFoundError as error:
        raise ValueError("artifact parent is absent") from error
    _require_capacity(parent, required_bytes, required_inodes, statvfs)
    root.mkdir(mode=0o700)
    root_info = _plain_directory(root, "artifact root")
    owned = (root_info.st_dev, root_info.st_ino)
    try:
        if root_info.st_dev != parent_info.st_dev:
            raise OSError("artifact root device differs")
        _require_capacity(root, required_bytes, required_inodes, statvfs)
    except Exception:
        current = root.lstat()
        if (current.st_dev, current.st_ino) == owned and not any(root.iterdir()):
            root.rmdir()
        raise


def require_remaining_capacity(
    root: Path, *, total_budget_bytes: int, total_budget_inodes: int,
    consumed_bytes: int, consumed_inodes: int,
    statvfs: Callable[[Path], Any] = os.statvfs,
) -> None:
    _plain_directory(root, "artifact root")
    values = (total_budget_bytes, total_budget_inodes, consumed_bytes, consumed_inodes)
    if any(type(value) is not int or value < 0 for value in values):
        raise ValueError("artifact budget differs")
    _require_capacity(
        root, max(total_budget_bytes - consumed_bytes, 0),
        max(total_budget_inodes - consumed_inodes, 0), statvfs,
    )


def validate_campaign_resume(
    config: object, run_root: Path,
    *, terminal_validator: Callable[[Path], None],
) -> tuple[Path, ...]:
    value = _strict_shape(config)
    _validate_config_values(value)
    root = run_root.resolve()
    _plain_directory(root, "campaign root")
    terminals: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("campaign resume symlink differs")
        if path.is_file():
            if path.name.endswith(".tmp") or path.name.endswith(".partial"):
                raise ValueError("campaign partial artifact differs")
            if path.name in {"run-receipt.json", "profile-receipt.json", "result.json",
                             "cuda_canary_v1.json"}:
                terminal_validator(path)
                terminals.append(path)
    return tuple(terminals)


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--checkout-root-template", required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validate-handoff", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    args = parse_args(arguments)
    try:
        if args.validate_handoff:
            validate_config_handoff(args.output, args.repo)
        else:
            build_and_write(
                repo=args.repo, checkout_root_template=args.checkout_root_template,
                artifact_root=args.artifact_root, output=args.output,
            )
    except Exception as error:
        print(f"FEPF config failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
