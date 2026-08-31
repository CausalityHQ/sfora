#!/usr/bin/env python3
"""Acquire and seal immutable local inputs for the SAGA GB10 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from collections.abc import Callable
from pathlib import Path

from sfora.pass209_m4 import canonical_json_bytes
from sfora.saga_feasibility import (
    ObjectAuthority,
    fixture_message_serialization_sha256,
    generated_fixture_image_bytes,
)

_ARCHITECTURE = "Qwen3VLForConditionalGeneration"
_PROMPT = "List the visible car attributes and relations."


def _lower_hex(value: str, width: int) -> bool:
    return len(value) == width and all(
        character in "0123456789abcdef" for character in value
    )


def _write_new(path: Path, payload: bytes) -> None:
    if path.exists():
        raise ValueError("SAGA preparation output already exists")
    path.write_bytes(payload)


def _snapshot_rows(model_root: Path) -> tuple[ObjectAuthority, ...]:
    rows = []
    for path in sorted(model_root.rglob("*")):
        if path.is_symlink():
            raise ValueError("SAGA prepared snapshot contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("SAGA prepared snapshot contains a special file")
        payload = path.read_bytes()
        rows.append(
            ObjectAuthority(
                role="model-file",
                relative_path=path.relative_to(model_root).as_posix(),
                byte_length=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            ).validated()
        )
    if not rows:
        raise ValueError("SAGA prepared snapshot is empty")
    return tuple(rows)


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    root.chmod(0o555)


def prepare_inputs(
    *,
    output_root: Path,
    repository_id: str,
    model_revision: str,
    source_commit: str,
    controller_commit: str,
    binary_sha256: str,
    environment_sha256: str,
    host: str,
    snapshot_resolver: Callable[..., str],
) -> Path:
    """Download one pinned snapshot and atomically seal all local authorities."""

    if output_root.exists() or output_root.is_symlink():
        raise ValueError("SAGA preparation output already exists")
    partial = output_root.with_name(f"{output_root.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError("SAGA preparation partial output already exists")
    if repository_id != "Qwen/Qwen3-VL-8B-Instruct":
        raise ValueError("SAGA preparation repository authority differs")
    for value, width, name in (
        (model_revision, 40, "model revision"),
        (source_commit, 40, "source commit"),
        (controller_commit, 40, "controller commit"),
        (binary_sha256, 64, "binary digest"),
        (environment_sha256, 64, "environment digest"),
    ):
        if not _lower_hex(value, width):
            raise ValueError(f"SAGA preparation {name} differs")
    if type(host) is not str or not host:
        raise ValueError("SAGA preparation host differs")

    model_root = partial / "model"
    try:
        partial.mkdir(mode=0o700)
        resolved = Path(
            snapshot_resolver(
                repository_id,
                revision=model_revision,
                local_dir=model_root,
            )
        ).resolve(strict=True)
        if resolved != model_root.resolve(strict=True):
            raise ValueError("SAGA preparation resolved snapshot differs")
        metadata = model_root / ".cache"
        if metadata.exists():
            shutil.rmtree(metadata)
        rows = _snapshot_rows(model_root)
        row_mappings = [row.to_mapping() for row in rows]
        tree_sha256 = hashlib.sha256(
            canonical_json_bytes({"files": row_mappings})
        ).hexdigest()
        _write_new(
            partial / "snapshot.json",
            canonical_json_bytes(
                {
                    "schema": "sfora-saga-snapshot-v1",
                    "repository_id": repository_id,
                    "model_revision": model_revision,
                    "processor_revision": model_revision,
                    "tokenizer_revision": model_revision,
                    "snapshot_tree_sha256": tree_sha256,
                    "architecture": _ARCHITECTURE,
                    "dtype": "bfloat16",
                    "attention_backend": "eager",
                    "trust_remote_code": False,
                    "files": row_mappings,
                }
            ),
        )
        image_sha256 = [
            hashlib.sha256(
                generated_fixture_image_bytes(source_commit, ordinal)
            ).hexdigest()
            for ordinal in range(64)
        ]
        _write_new(
            partial / "fixture.json",
            canonical_json_bytes(
                {
                    "schema": "sfora-saga-synthetic-fixture-v1",
                    "source_commit": source_commit,
                    "controller_commit": controller_commit,
                    "model_revision": model_revision,
                    "binary_sha256": binary_sha256,
                    "environment_sha256": environment_sha256,
                    "host": host,
                    "image_width": 224,
                    "image_height": 224,
                    "image_sha256": image_sha256,
                    "pair_ordinals": [0, 1],
                    "microbatch_ordinals": list(range(64)),
                    "prompt_utf8": _PROMPT,
                    "prompt_sha256": hashlib.sha256(_PROMPT.encode()).hexdigest(),
                    "message_serialization_sha256": (
                        fixture_message_serialization_sha256(_PROMPT, (0, 1))
                    ),
                    "group_size": 8,
                    "temperature_ppm": 700_000,
                    "top_p_ppm": 950_000,
                    "max_new_tokens": 1024,
                    "generation_seeds": list(range(8)),
                    "synthetic_rewards": [0, 1, 0, 1, 0, 1, 0, 1],
                    "attention_layer": 26,
                    "attribute_token_span": [0, 1],
                    "patch_tokens_per_image": 49,
                    "pseudo_labels": [ordinal % 2 for ordinal in range(64)],
                }
            ),
        )
        _make_read_only(partial)
        partial.rename(output_root)
    except Exception:
        if partial.exists():
            for path in partial.rglob("*"):
                if path.exists() and not path.is_symlink():
                    path.chmod(0o755 if path.is_dir() else 0o644)
            shutil.rmtree(partial)
        raise
    return output_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--repository-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--controller-commit", required=True)
    parser.add_argument("--binary-sha256", required=True)
    parser.add_argument("--environment-sha256", required=True)
    parser.add_argument("--host", required=True)
    parser.add_argument("--execute-acquisition", required=True, action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from huggingface_hub import snapshot_download

    prepare_inputs(
        output_root=args.output_root,
        repository_id=args.repository_id,
        model_revision=args.model_revision,
        source_commit=args.source_commit,
        controller_commit=args.controller_commit,
        binary_sha256=args.binary_sha256,
        environment_sha256=args.environment_sha256,
        host=args.host,
        snapshot_resolver=snapshot_download,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
