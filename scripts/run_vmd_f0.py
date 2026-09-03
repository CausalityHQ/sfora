#!/usr/bin/env python3
"""Run the local-only, forward-only VMD F0 teacher-target screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
from collections.abc import Sequence
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol, cast

import numpy as np
import torch
from PIL import Image

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

from scripts.diagnose_saga_gb10_feasibility import (  # noqa: E402
    LoadedAuthority,
    TransformersFactory,
    load_qwen_adapter,
)
from scripts.prepare_asgcv_p32_inputs import _authenticated_source_commit  # noqa: E402
from scripts.run_asgcv_p32 import load_p32_local_authority  # noqa: E402
from sfora.data import load_image_retrieval_examples, materialize_image  # noqa: E402
from sfora.saga_feasibility import (  # noqa: E402
    load_fixture_authority,
    load_snapshot_authority,
)
from sfora.verbalizer_margin_f0 import (  # noqa: E402
    VMD_F0_M2_SHA256,
    VMD_F0_M4_QUERY_SHA256,
    VMD_F0_REPEAT_ORDINALS,
    VmdF0Candidate,
    VmdF0Observation,
    VmdF0Result,
    canonical_vmd_f0_observation_bytes,
    canonical_vmd_f0_result_bytes,
    load_vmd_f0_candidates,
)


class VerdictScoreAdapter(Protocol):
    """Only the forward-only model operations F0 can invoke."""

    def prepare_image_pair(
        self,
        images: object,
        prompt_utf8: object,
        attribute_token_span: object,
        patch_tokens_per_image: object,
    ) -> object: ...

    def score_verdict_pair(
        self,
        pair: object,
        *,
        same_completion_ids: tuple[int, ...],
        different_completion_ids: tuple[int, ...],
    ) -> tuple[float, float]: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the explicit local-file-only F0 execution boundary."""

    values = list(sys.argv[1:] if argv is None else argv)
    options = [token.split("=", 1)[0] for token in values if token.startswith("--")]
    if len(options) != len(set(options)):
        raise SystemExit("duplicate VMD F0 option")
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--p32-authority", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--m2-error-manifest", type=Path, required=True)
    parser.add_argument("--m4-query-evidence", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--fixture-source-commit", required=True)
    parser.add_argument("--execute-vmd-f0", action="store_true", required=True)
    args = parser.parse_args(values)
    for name in ("source_commit", "fixture_source_commit"):
        value = getattr(args, name)
        if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
            parser.error(f"{name.replace('_', ' ')} must be 40 lowercase hex")
    if not args.model_root.is_dir() or args.model_root.is_symlink():
        parser.error("model root must be an existing regular directory")
    for name in (
        "snapshot_manifest",
        "fixture",
        "p32_authority",
        "train_manifest",
        "m2_error_manifest",
        "m4_query_evidence",
    ):
        path = getattr(args, name)
        if not path.is_file() or path.is_symlink():
            parser.error(f"{name.replace('_', ' ')} must be an existing regular file")
    if not args.output_directory.is_dir() or args.output_directory.is_symlink():
        parser.error("output directory must be an existing regular directory")
    return args


def _atomic_write(path: Path, raw: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or path.exists():
        raise ValueError("VMD F0 output already exists")
    with partial.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _cuda_peak() -> int:
    return max(1, int(torch.cuda.max_memory_reserved())) if torch.cuda.is_available() else 1


def _rss_peak() -> int:
    return max(1, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)


def _configure_determinism() -> None:
    configured = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if configured not in {None, ":4096:8"}:
        raise ValueError("VMD F0 CUBLAS determinism authority differs")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _load_observation(path: Path) -> VmdF0Observation:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("VMD F0 observation bytes differ") from error
    observation = VmdF0Observation.from_mapping(value)
    if canonical_vmd_f0_observation_bytes(observation) != raw:
        raise ValueError("VMD F0 observation bytes differ")
    return observation


def _score_candidate(
    adapter: VerdictScoreAdapter,
    candidate: VmdF0Candidate,
    *,
    images: Sequence[object],
    prompt_utf8: str,
    attribute_token_span: tuple[int, int],
    patch_tokens_per_image: int,
    same_completion_ids: tuple[int, ...],
    different_completion_ids: tuple[int, ...],
    source_commit: str,
    fixture_source_commit: str,
    model_revision: str,
    launch_authority_sha256: str,
    fixture_sha256: str,
) -> VmdF0Observation:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    started = perf_counter_ns()

    def score(position: int) -> tuple[float, float]:
        pair = adapter.prepare_image_pair(
            (images[candidate.query_position], images[position]),
            prompt_utf8,
            attribute_token_span,
            patch_tokens_per_image,
        )
        return adapter.score_verdict_pair(
            pair,
            same_completion_ids=same_completion_ids,
            different_completion_ids=different_completion_ids,
        )

    true_scores = score(candidate.true_position)
    wrong_scores = score(candidate.wrong_position)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return VmdF0Observation(
        source_commit=source_commit,
        fixture_source_commit=fixture_source_commit,
        model_revision=model_revision,
        launch_authority_sha256=launch_authority_sha256,
        fixture_sha256=fixture_sha256,
        m2_manifest_sha256=VMD_F0_M2_SHA256,
        m4_query_sha256=VMD_F0_M4_QUERY_SHA256,
        ordinal=candidate.ordinal,
        query_position=candidate.query_position,
        query_example_id=candidate.query_example_id,
        query_label=candidate.query_label,
        true_position=candidate.true_position,
        true_example_id=candidate.true_example_id,
        wrong_position=candidate.wrong_position,
        wrong_example_id=candidate.wrong_example_id,
        is_caliber_block=candidate.is_caliber_block,
        true_same_score=float(true_scores[0]),
        true_different_score=float(true_scores[1]),
        wrong_same_score=float(wrong_scores[0]),
        wrong_different_score=float(wrong_scores[1]),
        elapsed_ns=max(1, perf_counter_ns() - started),
        peak_cuda_reserved_bytes=_cuda_peak(),
        peak_rss_bytes=_rss_peak(),
    ).validated()


def run_vmd_f0_campaign(
    adapter: VerdictScoreAdapter,
    *,
    candidates: tuple[VmdF0Candidate, ...],
    images: Sequence[object],
    prompt_utf8: str,
    attribute_token_span: tuple[int, int],
    patch_tokens_per_image: int,
    same_completion_ids: tuple[int, ...],
    different_completion_ids: tuple[int, ...],
    output_directory: Path,
    source_commit: str,
    fixture_source_commit: str,
    model_revision: str,
    launch_authority_sha256: str,
    fixture_sha256: str,
) -> bytes:
    """Score or resume the exact F0 schedule and publish one terminal result."""

    result_path = output_directory / "result.json"
    observations: list[VmdF0Observation] = []
    for candidate in candidates:
        path = output_directory / f"observation-{candidate.ordinal:03d}.json"
        if path.exists():
            observation = _load_observation(path)
        else:
            observation = _score_candidate(
                adapter,
                candidate,
                images=images,
                prompt_utf8=prompt_utf8,
                attribute_token_span=attribute_token_span,
                patch_tokens_per_image=patch_tokens_per_image,
                same_completion_ids=same_completion_ids,
                different_completion_ids=different_completion_ids,
                source_commit=source_commit,
                fixture_source_commit=fixture_source_commit,
                model_revision=model_revision,
                launch_authority_sha256=launch_authority_sha256,
                fixture_sha256=fixture_sha256,
            )
            _atomic_write(path, canonical_vmd_f0_observation_bytes(observation))
        if (
            observation.ordinal != candidate.ordinal
            or observation.query_position != candidate.query_position
            or observation.query_example_id != candidate.query_example_id
            or observation.query_label != candidate.query_label
            or observation.true_position != candidate.true_position
            or observation.true_example_id != candidate.true_example_id
            or observation.wrong_position != candidate.wrong_position
            or observation.wrong_example_id != candidate.wrong_example_id
            or observation.is_caliber_block != candidate.is_caliber_block
        ):
            raise ValueError("VMD F0 observation candidate binding differs")
        observations.append(observation)
    if result_path.exists():
        raw = result_path.read_bytes()
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("VMD F0 result bytes differ")
        repeat_scores = tuple(tuple(row) for row in value["repeat_branch_scores"])
        rebuilt = VmdF0Result.from_observations(
            tuple(observations),
            repeat_checked_ordinals=tuple(value["repeat_checked_ordinals"]),
            repeat_branch_scores=cast(tuple[tuple[float, float, float, float], ...], repeat_scores),
            total_elapsed_ns=value["total_elapsed_ns"],
        )
        if canonical_vmd_f0_result_bytes(rebuilt) != raw:
            raise ValueError("VMD F0 result bytes differ")
        return raw
    repeats = tuple(
        _score_candidate(
            adapter,
            candidates[ordinal],
            images=images,
            prompt_utf8=prompt_utf8,
            attribute_token_span=attribute_token_span,
            patch_tokens_per_image=patch_tokens_per_image,
            same_completion_ids=same_completion_ids,
            different_completion_ids=different_completion_ids,
            source_commit=source_commit,
            fixture_source_commit=fixture_source_commit,
            model_revision=model_revision,
            launch_authority_sha256=launch_authority_sha256,
            fixture_sha256=fixture_sha256,
        )
        for ordinal in VMD_F0_REPEAT_ORDINALS
    )
    result = VmdF0Result.from_observations(
        tuple(observations),
        repeat_checked_ordinals=VMD_F0_REPEAT_ORDINALS,
        repeat_branch_scores=tuple(row.branch_scores for row in repeats),
        total_elapsed_ns=sum(row.elapsed_ns for row in (*observations, *repeats)),
    )
    raw = canonical_vmd_f0_result_bytes(result)
    _atomic_write(result_path, raw)
    return raw


def _rgb(image: object) -> np.ndarray:
    converted = materialize_image(image).convert("RGB")
    resized = converted.resize((224, 224), resample=Image.Resampling.BICUBIC)
    value = np.asarray(resized, dtype=np.uint8)
    if value.shape != (224, 224, 3):
        raise ValueError("VMD F0 image authority differs")
    return np.ascontiguousarray(value)


def main(argv: list[str] | None = None) -> int:
    """Authenticate all local inputs, run F0, and emit canonical stdout."""

    args = parse_args(argv)
    if _authenticated_source_commit(_REPOSITORY_ROOT) != args.source_commit:
        raise ValueError("VMD F0 executing source commit differs")
    _configure_determinism()
    examples = tuple(
        row
        for row in load_image_retrieval_examples(dataset_name="cars", split="train")
        if int(row.label) in range(82, 98)
    )
    candidates = load_vmd_f0_candidates(
        args.m2_error_manifest,
        args.m4_query_evidence,
        examples,
    )
    local = load_p32_local_authority(
        args.p32_authority,
        args.train_manifest,
        source_commit=args.fixture_source_commit,
    )
    snapshot = load_snapshot_authority(
        root=args.model_root,
        manifest_path=args.snapshot_manifest,
    )
    fixture = load_fixture_authority(args.fixture)
    if (
        fixture.source_commit != args.fixture_source_commit
        or fixture.model_revision != snapshot.model_revision
        or fixture.prompt_utf8 != local.prompt_utf8
        or fixture.patch_tokens_per_image != local.patch_tokens_per_image
    ):
        raise ValueError("VMD F0 model fixture binding differs")
    adapter = load_qwen_adapter(
        LoadedAuthority(snapshot=snapshot, fixture=fixture),
        factory=TransformersFactory(),
    )
    raw = run_vmd_f0_campaign(
        adapter,
        candidates=candidates,
        images=tuple(_rgb(row.image) for row in examples),
        prompt_utf8=local.prompt_utf8,
        attribute_token_span=local.attribute_token_span,
        patch_tokens_per_image=local.patch_tokens_per_image,
        same_completion_ids=local.completion_protocol.same_prefix_ids,
        different_completion_ids=local.completion_protocol.different_prefix_ids,
        output_directory=args.output_directory,
        source_commit=args.source_commit,
        fixture_source_commit=args.fixture_source_commit,
        model_revision=snapshot.model_revision,
        launch_authority_sha256=local.authority_sha256,
        fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
    )
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
