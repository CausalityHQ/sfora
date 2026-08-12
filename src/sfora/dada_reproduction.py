"""Faithful, minimal adapter for the pinned upstream DADA In-Shop recipe."""

from __future__ import annotations

import hashlib
import math
import os
import re
import stat
import subprocess
from collections.abc import Iterable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DADA_REVISION = "726ee8b9c94371e37beeeeeb9a50e6a0fec1d1c8"
INSHOP_CONFIG_SHA256 = "2685672b2a42faef74d5ee3af0cecc035741728379ea147ef1197af777ff2160"

_EXPECTED_INSHOP_CONFIG: dict[str, object] = {
    "dataset": "inshop",
    "n_epochs": 200,
    "batch_size": 180,
    "fd2_bn": True,
    "fc2_bn": True,
    "pos_class_mix": True,
    "arch": "resnet50_layernorm_double",
    "d_dis_ratio": 0.01,
    "decay": 0.0001,
    "dis_decay": 0.00005,
    "fc_fc2_dim": 4096,
    "fd_fc1_dim": 512,
    "loss_oproxy_neg_alpha": 200,
    "loss_oproxy_pos_alpha": 40,
    "lr": 0.00012,
    "lr_reduce_rate": 0.25,
    "lr_reduce_step": 40,
    "oproxy_ratio": 0.0075,
    "mix_alpha": 3,
    "mix_beta": 3,
    "warmup": 5,
    "store_improvements": True,
}


@dataclass(frozen=True)
class DadaSource:
    checkout: Path
    revision: str
    config_path: Path
    config_sha256: str
    config: dict[str, object]


@dataclass(frozen=True)
class DadaSmokeRequest:
    python: Path
    source: DadaSource
    smoke_config: Path
    dataset_root: Path
    output_root: Path
    save_name: str
    gpu: int
    seed: int


@dataclass(frozen=True)
class DadaProgress:
    completed_epochs: int
    optimizer_steps: int
    last_loss: float
    last_recall_at_1: float


def _run_git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(checkout), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_regular_file(path: Path, name: str) -> None:
    try:
        mode = path.lstat().st_mode
    except FileNotFoundError as exc:
        raise ValueError(f"{name} is missing") from exc
    if not stat.S_ISREG(mode):
        raise ValueError(f"{name} is not a regular file")


def _require_literal_config(value: Any) -> dict[str, object]:
    if type(value) is not dict or list(value) != list(_EXPECTED_INSHOP_CONFIG):
        raise ValueError("DADA In-Shop config keys/order differ")
    for key, expected in _EXPECTED_INSHOP_CONFIG.items():
        actual = value[key]
        if type(actual) is not type(expected) or actual != expected:
            raise ValueError(f"DADA In-Shop config field differs: {key}")
    return dict(value)


def validate_dada_source(checkout: Path) -> DadaSource:
    checkout = checkout.resolve(strict=True)
    if _run_git(checkout, "status", "--porcelain", "--untracked-files=no"):
        raise ValueError("DADA checkout has dirty tracked bytes")
    revision = _run_git(checkout, "rev-parse", "HEAD")
    if revision != DADA_REVISION:
        raise ValueError("DADA revision differs")

    main_path = checkout / "main.py"
    config_path = checkout / "configs" / "inshop.yaml"
    _require_regular_file(main_path, "DADA main.py")
    _require_regular_file(config_path, "DADA In-Shop config")
    config_sha256 = _sha256_file(config_path)
    if config_sha256 != INSHOP_CONFIG_SHA256:
        raise ValueError("DADA In-Shop config SHA-256 differs")
    config = _require_literal_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    return DadaSource(
        checkout=checkout,
        revision=revision,
        config_path=config_path,
        config_sha256=config_sha256,
        config=config,
    )


def build_smoke_config(
    source: DadaSource,
    destination: Path,
    *,
    epochs: int = 6,
) -> str:
    if type(epochs) is not int or epochs != 6:
        raise ValueError("DADA smoke epoch count must be exactly 6")
    payload = dict(source.config)
    payload["n_epochs"] = epochs
    encoded = yaml.safe_dump(payload, sort_keys=False).encode("utf-8")
    descriptor = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with suppress(FileNotFoundError):
            destination.unlink()
        raise
    return _sha256_file(destination)


def build_dada_command(request: DadaSmokeRequest) -> tuple[str, ...]:
    return (
        str(request.python),
        "-I",
        "-B",
        str(request.source.checkout / "main.py"),
        "--source_path",
        str(request.dataset_root),
        "--save_path",
        str(request.output_root),
        "--save_name",
        request.save_name,
        "--config",
        str(request.smoke_config),
        "--gpu",
        str(request.gpu),
        "--seed",
        str(request.seed),
    )


_TRAIN_RE = re.compile(
    r"\[Train Epoch (?P<epoch>\d+)\].*\[(?P<done>\d+)/(?P<total>\d+).*"
    r"DML:(?P<loss>[-+A-Za-z0-9.eE]+)"
)
_RECALL_RE = re.compile(r"e_recall@1:\s*(?P<recall>[-+A-Za-z0-9.eE]+)")


def parse_dada_log(lines: Iterable[str]) -> DadaProgress:
    epochs: dict[int, int] = {}
    losses: list[float] = []
    recalls: list[float] = []
    for raw_line in lines:
        line = str(raw_line)
        if "Traceback (most recent call last):" in line:
            raise ValueError("DADA child traceback")
        if "CUDA out of memory" in line:
            raise ValueError("DADA child CUDA out of memory")
        for fragment in line.split("\r"):
            train_match = _TRAIN_RE.search(fragment)
            if train_match is not None:
                done = int(train_match.group("done"))
                total = int(train_match.group("total"))
                if done <= 0 or total <= 0 or done != total:
                    raise ValueError("DADA optimizer progress is incomplete")
                loss = float(train_match.group("loss"))
                if not math.isfinite(loss):
                    raise ValueError("DADA loss is non-finite")
                epochs[int(train_match.group("epoch"))] = done
                losses.append(loss)
            recall_match = _RECALL_RE.search(fragment)
            if recall_match is not None:
                recall = float(recall_match.group("recall"))
                if not math.isfinite(recall):
                    raise ValueError("DADA recall is non-finite")
                recalls.append(recall)
    if not epochs or not losses:
        raise ValueError("DADA optimizer progress is absent")
    if not recalls:
        raise ValueError("DADA evaluation R@1 is absent")
    return DadaProgress(
        completed_epochs=len(epochs),
        optimizer_steps=sum(epochs.values()),
        last_loss=losses[-1],
        last_recall_at_1=recalls[-1],
    )
