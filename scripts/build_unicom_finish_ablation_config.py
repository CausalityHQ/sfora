#!/usr/bin/env python3
"""Build the authenticated UniCOM finish causal-panel config."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.unicom_finish_protocol import build_finish_config, validate_finish_config
from sfora.unicom_inshop import parse_inshop_partition
from sfora.unicom_training import identity_holdout


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_config_bytes(value: object) -> bytes:
    """Serialize the frozen config as sorted compact JSON plus one LF."""

    if (
        type(value) is not dict
        or value.get("schema") != "unicom-finish-ablation-config-v1"
        or value.get("claim_eligible") is not False
    ):
        raise ValueError("finish ablation config differs")
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--partition-sha256", required=True)
    parser.add_argument("--parent-checkpoint-sha256", required=True)
    parser.add_argument("--parent-receipt-sha256", required=True)
    parser.add_argument("--official-checkpoint-sha256", required=True)
    parser.add_argument("--unicom-revision", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-config", action="store_true", required=True)
    return parser.parse_args(arguments)


def run(args: argparse.Namespace) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    partition = args.dataset_root / "Eval" / "list_eval_partition.txt"
    if (
        source != args.source_commit
        or dirty
        or partition.is_symlink()
        or not partition.is_file()
        or _sha256_file(partition) != args.partition_sha256
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise ValueError("finish ablation config authority differs")
    records = parse_inshop_partition(args.dataset_root)
    training = tuple(record for record in records if record.split == "train")
    optimization, _query, _gallery, _labels = identity_holdout(
        training, fraction=0.2, seed=0
    )
    config = build_finish_config(
        tuple(record.label for record in optimization),
        partition_sha256=args.partition_sha256,
        parent_checkpoint_sha256=args.parent_checkpoint_sha256,
        parent_receipt_sha256=args.parent_receipt_sha256,
        source_commit=args.source_commit,
        official_checkpoint_sha256=args.official_checkpoint_sha256,
        unicom_revision=args.unicom_revision,
        environment_sha256=args.environment_sha256,
    )
    validate_finish_config(config)
    payload = canonical_config_bytes(config)
    publication = publish_bytes_noreplace(
        args.output,
        payload,
        validator=lambda observed: (
            None
            if observed == payload
            else (_ for _ in ()).throw(ValueError("finish config bytes differ"))
        ),
    )
    publication.close()
    return config


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(arguments)
        config = run(args)
    except Exception as error:
        print(f"finish config failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "arms": config["arms"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
