#!/usr/bin/env python3
"""Build the registered UniCOM full-width checkpoint pair inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

ARMS = ("sampled_512", "full_768")
EPOCHS = (4, 8, 12, 16)
ROW_KEYS = ("arm", "epoch", "path", "sha256", "bytes")
INVENTORY_KEYS = ("schema_version", "seed", "inventory")


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    with os.fdopen(os.dup(descriptor), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inventory(
    seed: int, checkpoints: tuple[tuple[str, int, Path], ...]
) -> dict[str, object]:
    """Authenticate eight checkpoint files and return their ordered inventory."""

    if type(seed) is not int or isinstance(seed, bool) or seed not in (0, 2, 3, 4, 5, 6):
        raise ValueError("seed must be a registered seed")
    if type(checkpoints) is not tuple or len(checkpoints) != 8:
        raise ValueError("inventory requires exactly eight checkpoints")
    expected = tuple((arm, epoch) for epoch in EPOCHS for arm in ARMS)
    if any(type(row) is not tuple or len(row) != 3 for row in checkpoints):
        raise ValueError("checkpoint row schema differs")
    observed = tuple((row[0], row[1]) for row in checkpoints)
    if observed != expected:
        raise ValueError("checkpoints must use exact epoch-major order")
    rows: list[dict[str, object]] = []
    identities: set[tuple[int, int]] = set()
    paths: set[str] = set()
    hashes: set[str] = set()
    for arm, epoch, raw_path in checkpoints:
        if not isinstance(raw_path, Path):
            raise TypeError("checkpoint path must be a pathlib.Path")
        if (
            not raw_path.is_absolute()
            or raw_path.is_symlink()
            or raw_path.resolve() != raw_path
        ):
            raise ValueError("checkpoint must be a regular non-symlink file")
        before = raw_path.lstat()
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("checkpoint must be a regular non-symlink file")
        descriptor = os.open(raw_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            if (before.st_dev, before.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("checkpoint identity changed during authentication")
            identity = (info.st_dev, info.st_ino)
            digest = _sha256_descriptor(descriptor)
        finally:
            os.close(descriptor)
        path = str(raw_path)
        if (
            not stat.S_ISREG(info.st_mode)
            or identity in identities
            or path in paths
            or digest in hashes
        ):
            raise ValueError("inventory requires distinct physical checkpoint files and hashes")
        identities.add(identity)
        paths.add(path)
        hashes.add(digest)
        rows.append(
            {
                "arm": arm,
                "epoch": epoch,
                "path": path,
                "sha256": digest,
                "bytes": info.st_size,
            }
        )
    return {
        "schema_version": "unicom-full-width-pair-config-v1",
        "seed": seed,
        "inventory": rows,
    }


def validate_inventory(payload: object) -> None:
    if type(payload) is not dict or tuple(payload) != INVENTORY_KEYS:
        raise ValueError("pair inventory schema differs")
    if payload["schema_version"] != "unicom-full-width-pair-config-v1":
        raise ValueError("pair inventory version differs")
    seed = payload["seed"]
    rows = payload["inventory"]
    if type(seed) is not int or isinstance(seed, bool) or seed not in (0, 2, 3, 4, 5, 6):
        raise ValueError("pair inventory seed differs")
    if type(rows) is not list or len(rows) != 8:
        raise ValueError("pair inventory rows differ")
    expected = tuple((arm, epoch) for epoch in EPOCHS for arm in ARMS)
    for index, (row, identity) in enumerate(zip(rows, expected, strict=True)):
        if (
            type(row) is not dict
            or tuple(row) != ROW_KEYS
            or (row["arm"], row["epoch"]) != identity
            or type(row["path"]) is not str
            or not Path(row["path"]).is_absolute()
            or type(row["sha256"]) is not str
            or len(row["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in row["sha256"])
            or type(row["bytes"]) is not int
            or isinstance(row["bytes"], bool)
            or row["bytes"] <= 0
        ):
            raise ValueError(f"pair inventory row {index} differs")
    paths = [row["path"] for row in rows]
    hashes = [row["sha256"] for row in rows]
    if len(set(paths)) != 8 or len(set(hashes)) != 8:
        raise ValueError("pair inventory requires distinct checkpoint paths and hashes")


def _strict_json(path: Path) -> dict[str, object]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"nonfinite JSON constant: {value}")

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if type(key) is not str or key in result:
                raise ValueError("pair inventory JSON keys differ")
            result[key] = value
        return result

    value = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=pairs,
        parse_constant=reject_constant,
    )
    if type(value) is not dict:
        raise ValueError("pair inventory root differs")
    return value


def write_inventory_atomic(payload: dict[str, object], output: Path) -> None:
    validate_inventory(payload)
    if not output.is_absolute():
        raise ValueError("output must be absolute")
    if output.exists() or output.is_symlink():
        raise FileExistsError("output already exists")
    parent = output.parent
    if not parent.is_dir() or parent.is_symlink() or parent.resolve() != parent:
        raise ValueError("output parent must be a real directory")
    data = (json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=parent
    )
    temporary = Path(temporary_name)
    published = False
    owned: tuple[int, int] | None = None
    try:
        offset = 0
        while offset < len(data):
            offset += os.write(descriptor, data[offset:])
        os.fsync(descriptor)
        info = os.fstat(descriptor)
        owned = (info.st_dev, info.st_ino)
        os.close(descriptor)
        descriptor = -1
        os.link(temporary, output)
        published = True
        os.unlink(temporary)
        validate_inventory(_strict_json(output))
        if output.read_bytes() != data or output.stat().st_mode & 0o777 != 0o600:
            raise ValueError("published pair inventory differs")
        directory = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        if descriptor >= 0:
            os.close(descriptor)
        if temporary.exists() and not temporary.is_symlink():
            os.unlink(temporary)
        if published and owned is not None:
            try:
                info = output.lstat()
            except FileNotFoundError:
                pass
            else:
                if not output.is_symlink() and (info.st_dev, info.st_ino) == owned:
                    os.unlink(output)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=3,
        metavar=("ARM", "EPOCH", "PATH"),
        required=True,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        checkpoints = tuple(
            (arm, int(epoch), Path(path)) for arm, epoch, path in args.checkpoint
        )
        write_inventory_atomic(build_inventory(args.seed, checkpoints), args.output)
        return 0
    except (OSError, TypeError, ValueError) as error:
        print(f"structural failure: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
