#!/usr/bin/env python3
"""Seal train-only Cars inputs for the ASG-CV P32 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from PIL import Image

from sfora.asgcv_pilot import derive_asgcv_p32_schedule_seed
from sfora.asgcv_protocol import (
    AsgcvCompletionProtocol,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    build_asgcv_pair_schedule,
)
from sfora.data import ImageExample, load_image_retrieval_examples, materialize_image
from sfora.saga_feasibility import (
    fixture_message_serialization_sha256,
    generated_fixture_image_bytes,
    load_snapshot_authority,
)

P32_PROMPT = (
    "Determine whether these two images show the SAME car model or DIFFERENT car models. "
    "Start with exactly SAME or DIFFERENT, then list the visible attributes supporting "
    "the decision."
)
_PARTITION_DOMAIN = b"sfora-asgcv-p32-class-partition-v1\0"
_EXAMPLE_DOMAIN = b"sfora-asgcv-p32-example-v1\0"


@dataclass(frozen=True, slots=True)
class P32PromptAuthority:
    """Tokenizer-derived prompt and completion boundaries."""

    same_prefix_ids: tuple[int, ...]
    different_prefix_ids: tuple[int, ...]
    terminal_token_ids: tuple[int, ...]
    attribute_token_span: tuple[int, int]
    patch_tokens_per_image: int

    def validated(self) -> P32PromptAuthority:
        AsgcvCompletionProtocol(
            self.same_prefix_ids,
            self.different_prefix_ids,
            self.terminal_token_ids,
        ).validated()
        if (
            type(self.attribute_token_span) is not tuple
            or len(self.attribute_token_span) != 2
            or any(type(value) is not int for value in self.attribute_token_span)
            or not 0 <= self.attribute_token_span[0] < self.attribute_token_span[1]
            or type(self.patch_tokens_per_image) is not int
            or self.patch_tokens_per_image <= 0
        ):
            raise ValueError("ASG-CV P32 prompt authority differs")
        return self


class PromptResolver(Protocol):
    """Resolve exact tokenizer and multimodal prompt boundaries."""

    def resolve(self, model_root: Path, prompt: str) -> P32PromptAuthority: ...


def _default_processor_loader(model_root: Path) -> object:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(  # type: ignore[no-untyped-call]
        model_root, local_files_only=True
    )


@dataclass(frozen=True, slots=True)
class TransformersPromptResolver:
    """Derive P32 token boundaries from the sealed local processor."""

    processor_loader: Callable[[Path], object] = _default_processor_loader

    def resolve(self, model_root: Path, prompt: str) -> P32PromptAuthority:
        processor = self.processor_loader(model_root)
        tokenizer = getattr(processor, "tokenizer", None)
        encode = getattr(tokenizer, "encode", None)
        apply_chat_template = getattr(processor, "apply_chat_template", None)
        if not callable(encode) or not callable(apply_chat_template):
            raise ValueError("ASG-CV P32 processor authority differs")

        same = tuple(encode("SAME", add_special_tokens=False))
        different = tuple(encode("DIFFERENT", add_special_tokens=False))
        prompt_ids = tuple(encode(prompt, add_special_tokens=False))
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if type(eos_token_id) is not int or not prompt_ids:
            raise ValueError("ASG-CV P32 tokenizer authority differs")

        images = [np.zeros((224, 224, 3), dtype=np.uint8) for _ in range(2)]
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": images[0]},
                    {"type": "image", "image": images[1]},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        raw = apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        required = {"input_ids", "mm_token_type_ids"}
        if not isinstance(raw, Mapping) or not required <= set(raw):
            raise ValueError("ASG-CV P32 processor output authority differs")
        input_ids = raw["input_ids"]
        token_types = raw["mm_token_type_ids"]
        if tuple(input_ids.shape[:1]) != (1,) or token_types.shape != input_ids.shape:
            raise ValueError("ASG-CV P32 processor sequence authority differs")
        sequence = tuple(int(value) for value in input_ids[0].tolist())
        mask = tuple(int(value) for value in token_types[0].tolist())

        prompt_starts = [
            start
            for start in range(len(sequence) - len(prompt_ids) + 1)
            if sequence[start : start + len(prompt_ids)] == prompt_ids
        ]
        if len(prompt_starts) != 1:
            raise ValueError("ASG-CV P32 prompt token span differs")
        prompt_start = prompt_starts[0]

        image_positions = [ordinal for ordinal, token_type in enumerate(mask) if token_type == 1]
        groups: list[tuple[int, int]] = []
        for position in image_positions:
            if not groups or position != groups[-1][1]:
                groups.append((position, position + 1))
            else:
                groups[-1] = (groups[-1][0], position + 1)
        if len(groups) != 2 or groups[0][1] - groups[0][0] != groups[1][1] - groups[1][0]:
            raise ValueError("ASG-CV P32 image token ranges differ")

        return P32PromptAuthority(
            same_prefix_ids=same,
            different_prefix_ids=different,
            terminal_token_ids=(eos_token_id,),
            attribute_token_span=(prompt_start, prompt_start + len(prompt_ids)),
            patch_tokens_per_image=groups[0][1] - groups[0][0],
        ).validated()


@dataclass(frozen=True, slots=True)
class PreparedP32Inputs:
    """Three canonical P32 inputs under one sealed directory."""

    train_manifest: Path
    launch_authority: Path
    fixture: Path


def _canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _lower_hex(value: str, width: int, *, name: str) -> None:
    if len(value) != width or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"ASG-CV P32 {name} differs")


def _class_bands(seed: str) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    seed_bytes = bytes.fromhex(seed)
    ranked = sorted(
        range(98),
        key=lambda label: (
            hashlib.sha256(_PARTITION_DOMAIN + seed_bytes + label.to_bytes(2, "big")).digest(),
            label,
        ),
    )
    return (
        tuple(sorted(ranked[:64])),
        tuple(sorted(ranked[64:81])),
        tuple(sorted(ranked[81:])),
    )


def _select_rows(
    examples: Sequence[ImageExample], classes: tuple[int, ...], seed: str
) -> tuple[ImageExample, ...]:
    seed_bytes = bytes.fromhex(seed)
    selected = []
    for label in classes:
        candidates = [example for example in examples if example.label == label]
        candidates.sort(
            key=lambda example: (
                hashlib.sha256(
                    _EXAMPLE_DOMAIN + seed_bytes + example.example_id.encode("utf-8")
                ).digest(),
                example.example_id,
            )
        )
        if len(candidates) < 4:
            raise ValueError("ASG-CV P32 class has fewer than four training examples")
        selected.extend(candidates[:4])
    return tuple(sorted(selected, key=lambda example: example.example_id))


def _rgb_224(value: object) -> np.ndarray:
    materialized = materialize_image(value)
    if isinstance(materialized, np.ndarray):
        if materialized.dtype != np.uint8 or materialized.ndim != 3 or materialized.shape[-1] != 3:
            raise ValueError("ASG-CV P32 image differs")
        image = Image.fromarray(materialized, mode="RGB")
    else:
        convert = getattr(materialized, "convert", None)
        if not callable(convert):
            raise ValueError("ASG-CV P32 image differs")
        image = convert("RGB")
    resized = image.resize((224, 224), resample=Image.Resampling.BICUBIC)
    result = np.asarray(resized, dtype=np.uint8)
    if result.shape != (224, 224, 3):
        raise ValueError("ASG-CV P32 resized image differs")
    return np.ascontiguousarray(result)


def _write_array(path: Path, value: np.ndarray) -> str:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture_bytes(
    *,
    source_commit: str,
    model_revision: str,
    prompt: P32PromptAuthority,
) -> bytes:
    repository = Path(__file__).resolve().parents[1]
    binary_sha256 = hashlib.sha256(
        (repository / "scripts/run_asgcv_p32.py").read_bytes()
    ).hexdigest()
    environment_sha256 = hashlib.sha256((repository / "uv.lock").read_bytes()).hexdigest()
    pair_ordinals = (0, 1)
    return _canonical_bytes(
        {
            "schema": "sfora-saga-synthetic-fixture-v1",
            "source_commit": source_commit,
            "controller_commit": source_commit,
            "model_revision": model_revision,
            "binary_sha256": binary_sha256,
            "environment_sha256": environment_sha256,
            "host": socket.gethostname(),
            "image_width": 224,
            "image_height": 224,
            "image_sha256": [
                hashlib.sha256(generated_fixture_image_bytes(source_commit, ordinal)).hexdigest()
                for ordinal in range(64)
            ],
            "pair_ordinals": list(pair_ordinals),
            "microbatch_ordinals": list(range(64)),
            "prompt_utf8": P32_PROMPT,
            "prompt_sha256": hashlib.sha256(P32_PROMPT.encode()).hexdigest(),
            "message_serialization_sha256": fixture_message_serialization_sha256(
                P32_PROMPT, pair_ordinals
            ),
            "group_size": 8,
            "temperature_ppm": 700_000,
            "top_p_ppm": 950_000,
            "max_new_tokens": 1024,
            "generation_seeds": list(range(8)),
            "synthetic_rewards": [0, 1, 0, 1, 0, 1, 0, 1],
            "attention_layer": 26,
            "attribute_token_span": [0, 1],
            "patch_tokens_per_image": prompt.patch_tokens_per_image,
            "pseudo_labels": [ordinal % 2 for ordinal in range(64)],
        }
    )


def prepare_p32_inputs(
    *,
    output_root: Path,
    model_root: Path,
    source_commit: str,
    model_revision: str,
    partition_seed_sha256: str,
    rollout_seed_sha256: str,
    predictor_initialization_seed_sha256: str,
    examples: Sequence[ImageExample],
    prompt_resolver: PromptResolver,
) -> PreparedP32Inputs:
    """Build one outcome-blind P32 manifest, fixture, and launch authority."""

    for value, width, name in (
        (source_commit, 40, "source commit"),
        (model_revision, 40, "model revision"),
        (partition_seed_sha256, 64, "partition seed"),
        (rollout_seed_sha256, 64, "rollout seed"),
        (predictor_initialization_seed_sha256, 64, "predictor seed"),
    ):
        _lower_hex(value, width, name=name)
    if output_root.exists() or output_root.is_symlink():
        raise ValueError("ASG-CV P32 output already exists")
    if type(examples) not in {list, tuple} or len(examples) < 98 * 4:
        raise ValueError("ASG-CV P32 Cars training rows differ")
    if any(type(example) is not ImageExample for example in examples):
        raise ValueError("ASG-CV P32 Cars training rows differ")
    if any(example.label < 0 or example.label >= 98 for example in examples):
        raise ValueError("ASG-CV P32 official test access is forbidden")
    if len({example.example_id for example in examples}) != len(examples):
        raise ValueError("ASG-CV P32 Cars example identities differ")
    if {example.label for example in examples} != set(range(98)):
        raise ValueError("ASG-CV P32 Cars training class authority differs")

    prompt = prompt_resolver.resolve(model_root, P32_PROMPT).validated()
    predictor_classes, validation_classes, optimization_classes = _class_bands(
        partition_seed_sha256
    )
    predictor = _select_rows(examples, predictor_classes, partition_seed_sha256)
    validation = _select_rows(examples, validation_classes, partition_seed_sha256)
    optimization = _select_rows(examples, optimization_classes, partition_seed_sha256)
    partial = output_root.with_name(f"{output_root.name}.partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError("ASG-CV P32 partial output already exists")
    partial.mkdir(mode=0o700)
    try:

        def materialized_rows(role: str, rows: tuple[ImageExample, ...]) -> list[dict[str, object]]:
            result = []
            for ordinal, example in enumerate(rows):
                basename = f"{role}-{ordinal:04d}.npy"
                digest = _write_array(partial / basename, _rgb_224(example.image))
                result.append(
                    {
                        "example_id": example.example_id,
                        "label": example.label,
                        "array_path": basename,
                        "array_sha256": digest,
                    }
                )
            return result

        manifest = {
            "schema": "sfora-cars-train-p32-manifest-v1",
            "official_test_access": False,
            "predictor_train": materialized_rows("predictor", predictor),
            "e0_validation": materialized_rows("e0-validation", validation),
            "e1_optimization": materialized_rows("e1-optimization", optimization),
        }
        manifest_bytes = _canonical_bytes(manifest)
        manifest_path = partial / "train-manifest.json"
        manifest_path.write_bytes(manifest_bytes)
        partition = AsgcvPartitionAuthority(
            source_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            partition_seed_sha256=partition_seed_sha256,
            predictor_train_class_ids=predictor_classes,
            e0_validation_class_ids=validation_classes,
            e1_optimization_class_ids=optimization_classes,
        ).validated()
        protocol = AsgcvCompletionProtocol(
            prompt.same_prefix_ids,
            prompt.different_prefix_ids,
            prompt.terminal_token_ids,
        ).validated()
        rollout = AsgcvRolloutAuthority(
            master_seed_sha256=rollout_seed_sha256,
            model_revision=model_revision,
            temperature=0.7,
            top_p=0.95,
            max_new_tokens=1024,
        ).validated()
        predictor_ids = tuple(example.example_id for example in predictor)
        predictor_labels = tuple(example.label for example in predictor)
        schedule = build_asgcv_pair_schedule(
            predictor_ids,
            predictor_labels,
            schedule_seed_sha256=derive_asgcv_p32_schedule_seed(
                partition_authority=partition,
                source_commit=source_commit,
            ),
            pair_count=32,
        )
        authority_path = partial / "p32-authority.json"
        authority_path.write_bytes(
            _canonical_bytes(
                {
                    "schema": "sfora-asgcv-p32-launch-v1",
                    "source_commit": source_commit,
                    "prompt_utf8": P32_PROMPT,
                    "attribute_token_span": list(prompt.attribute_token_span),
                    "patch_tokens_per_image": prompt.patch_tokens_per_image,
                    "predictor_initialization_seed_sha256": (predictor_initialization_seed_sha256),
                    "partition_authority": partition.to_mapping(),
                    "completion_protocol": protocol.to_mapping(),
                    "rollout_authority": rollout.to_mapping(),
                    "pilot_schedule": schedule.to_mapping(),
                }
            )
        )
        fixture_path = partial / "fixture.json"
        fixture_path.write_bytes(
            _fixture_bytes(
                source_commit=source_commit,
                model_revision=model_revision,
                prompt=prompt,
            )
        )
        partial.rename(output_root)
    except BaseException:
        if partial.exists():
            shutil.rmtree(partial)
        raise
    return PreparedP32Inputs(
        train_manifest=output_root / "train-manifest.json",
        launch_authority=output_root / "p32-authority.json",
        fixture=output_root / "fixture.json",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    arguments = list(sys.argv[1:] if argv is None else argv)
    option_names = [value.split("=", 1)[0] for value in arguments if value.startswith("--")]
    if len(option_names) != len(set(option_names)):
        raise SystemExit("ASG-CV P32 duplicate option")
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--partition-seed-sha256", required=True)
    parser.add_argument("--rollout-seed-sha256", required=True)
    parser.add_argument("--predictor-initialization-seed-sha256", required=True)
    parser.add_argument("--execute-preparation", required=True, action="store_true")
    return parser.parse_args(arguments)


def _current_clean_source_commit() -> str:
    repository = Path(__file__).resolve().parents[1]
    return _authenticated_source_commit(repository)


def _authenticated_source_commit(repository: Path) -> str:
    """Authenticate a clean Git checkout or its sealed frozen-revision export."""

    if not (repository / ".git").exists():
        revision_path = repository / "SOURCE_REVISION"
        manifest_path = repository / "SOURCE_MANIFEST.sha256"
        if (
            revision_path.is_symlink()
            or manifest_path.is_symlink()
            or not revision_path.is_file()
            or not manifest_path.is_file()
        ):
            raise ValueError("ASG-CV P32 frozen source authority differs")
        try:
            revision = revision_path.read_text(encoding="ascii").strip()
            manifest_lines = manifest_path.read_text(encoding="ascii").splitlines()
        except (OSError, UnicodeError) as error:
            raise ValueError("ASG-CV P32 frozen source authority differs") from error
        _lower_hex(revision, 40, name="source commit")
        previous = ""
        observed: set[str] = set()
        for line in manifest_lines:
            if len(line) < 67 or line[64:66] != "  ":
                raise ValueError("ASG-CV P32 source manifest differs")
            digest, relative = line[:64], line[66:]
            _lower_hex(digest, 64, name="source manifest digest")
            relative_path = Path(relative)
            if (
                not relative
                or "\\" in relative
                or relative_path.is_absolute()
                or any(part in ("", ".", "..") for part in relative_path.parts)
                or relative <= previous
                or relative in observed
            ):
                raise ValueError("ASG-CV P32 source manifest differs")
            path = repository / relative_path
            if path.is_symlink() or not path.is_file():
                raise ValueError("ASG-CV P32 source manifest differs")
            try:
                payload = path.read_bytes()
            except OSError as error:
                raise ValueError("ASG-CV P32 source manifest differs") from error
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("ASG-CV P32 source manifest differs")
            observed.add(relative)
            previous = relative
        if "SOURCE_REVISION" not in observed:
            raise ValueError("ASG-CV P32 source manifest differs")
        for path in repository.rglob("*"):
            if not (path.is_file() or path.is_symlink()):
                continue
            relative = path.relative_to(repository).as_posix()
            if relative not in observed and relative != "SOURCE_MANIFEST.sha256":
                raise ValueError("ASG-CV P32 unregistered source file differs")
        return revision

    for arguments in (("diff", "--quiet", "HEAD", "--"), ("diff", "--cached", "--quiet")):
        try:
            subprocess.run(
                ("git", *arguments),
                cwd=repository,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ValueError("ASG-CV P32 source checkout is not clean") from error
    try:
        untracked = subprocess.run(
            (
                "git",
                "ls-files",
                "--others",
                "--exclude-standard",
                "--",
                "src",
                "scripts",
                "sitecustomize.py",
            ),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("ASG-CV P32 source checkout is not clean") from error
    if untracked:
        raise ValueError("ASG-CV P32 unregistered source file differs")
    try:
        result = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("ASG-CV P32 source commit is unavailable") from error
    commit = result.stdout.strip()
    _lower_hex(commit, 40, name="source commit")
    return commit


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if _current_clean_source_commit() != args.source_commit:
        raise ValueError("ASG-CV P32 source commit differs")
    snapshot = load_snapshot_authority(
        root=args.model_root,
        manifest_path=args.snapshot_manifest,
    )
    if snapshot.model_revision != args.model_revision:
        raise ValueError("ASG-CV P32 snapshot revision differs")
    examples = load_image_retrieval_examples(
        dataset_name="cars",
        split="train",
        limit_per_class=None,
        min_per_class=None,
        max_classes=None,
        seed=0,
    )
    prepare_p32_inputs(
        output_root=args.output_root,
        model_root=args.model_root,
        source_commit=args.source_commit,
        model_revision=args.model_revision,
        partition_seed_sha256=args.partition_seed_sha256,
        rollout_seed_sha256=args.rollout_seed_sha256,
        predictor_initialization_seed_sha256=args.predictor_initialization_seed_sha256,
        examples=examples,
        prompt_resolver=TransformersPromptResolver(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
