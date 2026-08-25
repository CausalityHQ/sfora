from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "build_unicom_full_width_pair_config.py"
ARMS = ("sampled_512", "full_768")
EPOCHS = (4, 8, 12, 16)


def _load_module():
    spec = importlib.util.spec_from_file_location("_build_pair_config", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load pair-config builder")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint_args(tmp_path: Path) -> tuple[list[str], list[tuple[str, int, Path]]]:
    args: list[str] = []
    fixtures: list[tuple[str, int, Path]] = []
    for epoch in EPOCHS:
        for arm in ARMS:
            payload = f"{epoch}:{arm}\n".encode()
            path = tmp_path / arm / f"epoch-{epoch:04d}.pt"
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(payload)
            args.extend(("--checkpoint", arm, str(epoch), str(path)))
            fixtures.append((arm, epoch, path))
    return args, fixtures


def test_builder_publishes_exact_epoch_major_inventory_once(tmp_path: Path) -> None:
    args, fixtures = _checkpoint_args(tmp_path)
    output = tmp_path / "pair-inventory.json"
    command = [
        sys.executable,
        "-I",
        "-B",
        str(SCRIPT),
        "--seed",
        "0",
        "--output",
        str(output),
        *args,
    ]

    first = subprocess.run(command, text=True, capture_output=True, check=False)
    assert first.returncode == 0, first.stderr
    assert output.stat().st_mode & 0o777 == 0o600
    observed = json.loads(output.read_text(encoding="utf-8"))
    assert tuple(observed) == ("schema_version", "seed", "inventory")
    assert observed["schema_version"] == "unicom-full-width-pair-config-v1"
    assert observed["seed"] == 0
    assert observed["inventory"] == [
        {
            "arm": arm,
            "epoch": epoch,
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "bytes": len(payload),
        }
        for arm, epoch, path in fixtures
        for payload in (path.read_bytes(),)
    ]
    evaluator_spec = importlib.util.spec_from_file_location(
        "_evaluate_full_width_for_pair_builder_test",
        ROOT / "scripts" / "evaluate_unicom_full_width_objective.py",
    )
    assert evaluator_spec is not None and evaluator_spec.loader is not None
    evaluator = importlib.util.module_from_spec(evaluator_spec)
    evaluator_spec.loader.exec_module(evaluator)
    assert len(evaluator._validate_pair_config(observed)) == 8

    original = output.read_bytes()
    second = subprocess.run(command, text=True, capture_output=True, check=False)
    assert second.returncode == 2
    assert "output already exists" in second.stderr
    assert output.read_bytes() == original


def test_builder_rejects_wrong_order_symlink_and_incomplete_rows(tmp_path: Path) -> None:
    module = _load_module()
    _, fixtures = _checkpoint_args(tmp_path)

    with pytest.raises(ValueError, match="epoch-major order"):
        module.build_inventory(0, tuple(reversed(fixtures)))

    linked = tmp_path / "linked.pt"
    linked.symlink_to(fixtures[0][2])
    mutation = list(fixtures)
    mutation[0] = (mutation[0][0], mutation[0][1], linked)
    with pytest.raises(ValueError, match="regular non-symlink"):
        module.build_inventory(0, tuple(mutation))

    with pytest.raises(ValueError, match="eight checkpoints"):
        module.build_inventory(0, tuple(fixtures[:-1]))

    repeated = tuple((arm, epoch, fixtures[0][2]) for epoch in EPOCHS for arm in ARMS)
    with pytest.raises(ValueError, match="distinct physical checkpoint"):
        module.build_inventory(0, repeated)

    identical_copy = tmp_path / "identical-copy.pt"
    identical_copy.write_bytes(fixtures[0][2].read_bytes())
    duplicate_hash_rows = list(fixtures)
    duplicate_hash_rows[1] = (
        duplicate_hash_rows[1][0],
        duplicate_hash_rows[1][1],
        identical_copy,
    )
    with pytest.raises(ValueError, match="distinct physical checkpoint"):
        module.build_inventory(0, tuple(duplicate_hash_rows))

    with pytest.raises(ValueError, match="registered seed"):
        module.build_inventory(1, tuple(fixtures))

    fifo = tmp_path / "checkpoint.fifo"
    os.mkfifo(fifo)
    fifo_rows = list(fixtures)
    fifo_rows[0] = (fifo_rows[0][0], fifo_rows[0][1], fifo)
    with pytest.raises(ValueError, match="regular non-symlink"):
        module.build_inventory(0, tuple(fifo_rows))
