#!/usr/bin/env python3
"""Run the train-only forced SAME-before-DIFFERENT SAGA pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter_ns
from typing import Protocol, cast

import numpy as np
import torch

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
from sfora.asgcv_forced_pilot import (  # noqa: E402
    ASGCV_FORCED_PAIR_COUNT,
    AsgcvForcedObservation,
    AsgcvForcedResult,
    canonical_asgcv_forced_observation_bytes,
    canonical_asgcv_forced_result_bytes,
)
from sfora.asgcv_pilot import ASGCV_P32_BOUNDARY_NAMES  # noqa: E402
from sfora.asgcv_protocol import AsgcvCompletionProtocol, AsgcvPairSchedule  # noqa: E402
from sfora.saga_feasibility import load_fixture_authority, load_snapshot_authority  # noqa: E402


class ForcedAdapter(Protocol):
    """Only the two model operations used by the forced-verdict pilot."""

    def prepare_image_pair(
        self,
        images: object,
        prompt_utf8: object,
        attribute_token_span: object,
        patch_tokens_per_image: object,
    ) -> object: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the local-only forced-verdict execution boundary."""

    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--p32-authority", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execute-forced-p32", required=True, action="store_true")
    parsed = parser.parse_args(argv)
    if (
        type(parsed.source_commit) is not str
        or len(parsed.source_commit) != 40
        or any(character not in "0123456789abcdef" for character in parsed.source_commit)
    ):
        parser.error("source commit must be 40 lowercase hex")
    if not parsed.model_root.is_dir() or parsed.model_root.is_symlink():
        parser.error("model root must be an existing regular directory")
    for name in ("snapshot_manifest", "fixture", "p32_authority", "train_manifest"):
        path = getattr(parsed, name)
        if path.is_symlink() or not path.is_file():
            parser.error(f"{name.replace('_', ' ')} must be an existing regular file")
    if not parsed.output_directory.is_dir() or parsed.output_directory.is_symlink():
        parser.error("output directory must be an existing regular directory")
    return parsed

    def collapsed_verdict_patch_gradient(
        self,
        pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> object: ...


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_candidate_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def _peak_cuda_reserved_bytes() -> int:
    return max(1, int(torch.cuda.max_memory_reserved())) if torch.cuda.is_available() else 1


def _process_peak_rss_bytes() -> int:
    return max(1, int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024)


def _timed[Value](operation: object) -> tuple[Value, int]:
    if not callable(operation):
        raise ValueError("ASG-CV forced timed operation differs")
    _synchronize()
    started = perf_counter_ns()
    value = operation()
    _synchronize()
    return cast(Value, value), max(1, perf_counter_ns() - started)


def _array(value: object, *, name: str) -> np.ndarray:
    if type(value) is torch.Tensor:
        value = cast(torch.Tensor, value).detach().float().cpu().contiguous().numpy()
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or value.size == 0
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError(f"ASG-CV forced {name} differs")
    return np.ascontiguousarray(value)


def _norm(value: np.ndarray) -> float:
    result = float(np.sqrt(np.square(value.astype(np.float64)).sum(dtype=np.float64)))
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("ASG-CV forced gradient norm differs")
    return result


def run_forced_candidate(
    adapter: ForcedAdapter,
    *,
    ordinal: int,
    images: tuple[np.ndarray, ...],
    prompt_utf8: str,
    attribute_token_span: tuple[int, int],
    patch_tokens_per_image: int,
    completion_protocol: AsgcvCompletionProtocol,
    pilot_schedule: AsgcvPairSchedule,
    source_commit: str,
    launch_authority_sha256: str,
    model_revision: str,
    fixture_sha256: str,
) -> AsgcvForcedObservation:
    """Capture one label-blind fixed-order verdict score gap and gradient."""

    completion_protocol.validated()
    pilot_schedule.validated()
    if type(ordinal) is not int or not 0 <= ordinal < pilot_schedule.pair_count:
        raise ValueError("ASG-CV forced candidate ordinal differs")
    row = pilot_schedule.pairs[ordinal]
    if max(row.left_index, row.right_index) >= len(images):
        raise ValueError("ASG-CV forced image schedule differs")
    same = completion_protocol.same_prefix_ids
    different = completion_protocol.different_prefix_ids
    _reset_candidate_peak()
    pair, prepare_elapsed_ns = _timed(
        lambda: adapter.prepare_image_pair(
            (images[row.left_index], images[row.right_index]),
            prompt_utf8,
            attribute_token_span,
            patch_tokens_per_image,
        )
    )
    target, replay_elapsed_ns = _timed(
        lambda: adapter.collapsed_verdict_patch_gradient(
            pair,
            correct_completion_ids=same,
            incorrect_completion_ids=different,
        )
    )
    gradient = _array(getattr(target, "predicted_gradient", None), name="gradient")
    boundary = _array(
        getattr(target, "boundary_predicted_gradient", None), name="boundary gradient"
    )
    scores = getattr(target, "branch_scores", None)
    if (
        getattr(target, "boundary_names", None) != ASGCV_P32_BOUNDARY_NAMES
        or boundary.shape[0] != 4
        or type(scores) is not tuple
        or len(scores) != 2
        or any(type(score) is not float or not math.isfinite(score) for score in scores)
    ):
        raise ValueError("ASG-CV forced target differs")
    return AsgcvForcedObservation(
        source_commit=source_commit,
        launch_authority_sha256=launch_authority_sha256,
        pilot_schedule_sha256=pilot_schedule.sha256(),
        model_revision=model_revision,
        fixture_sha256=fixture_sha256,
        candidate_pair_ordinal=ordinal,
        pair_ordinals=(row.left_index, row.right_index),
        relation_sign=row.relation_sign,
        same_score=scores[0],
        different_score=scores[1],
        gradient_sha256=hashlib.sha256(gradient.tobytes()).hexdigest(),
        gradient_norm=_norm(gradient),
        boundary_norms=cast(
            tuple[float, float, float, float], tuple(_norm(value) for value in boundary)
        ),
        prepare_elapsed_ns=prepare_elapsed_ns,
        replay_elapsed_ns=replay_elapsed_ns,
        peak_cuda_reserved_bytes=_peak_cuda_reserved_bytes(),
        peak_rss_bytes=_process_peak_rss_bytes(),
    ).validated()


def _write_atomic(path: Path, payload: bytes) -> None:
    partial = path.with_name(path.name + ".partial")
    if partial.exists():
        raise ValueError("ASG-CV forced partial output exists")
    with partial.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _read_observation(path: Path) -> AsgcvForcedObservation:
    raw = path.read_bytes()
    try:
        observation = AsgcvForcedObservation.from_mapping(json.loads(raw))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("ASG-CV forced observation bytes differ") from error
    if canonical_asgcv_forced_observation_bytes(observation) != raw:
        raise ValueError("ASG-CV forced observation bytes differ")
    return observation


def _validate_context(
    observation: AsgcvForcedObservation,
    *,
    pilot_schedule: AsgcvPairSchedule,
    source_commit: str,
    launch_authority_sha256: str,
    model_revision: str,
    fixture_sha256: str,
) -> None:
    row = pilot_schedule.pairs[observation.candidate_pair_ordinal]
    if (
        observation.source_commit != source_commit
        or observation.launch_authority_sha256 != launch_authority_sha256
        or observation.pilot_schedule_sha256 != pilot_schedule.sha256()
        or observation.model_revision != model_revision
        or observation.fixture_sha256 != fixture_sha256
        or observation.pair_ordinals != (row.left_index, row.right_index)
        or observation.relation_sign != row.relation_sign
    ):
        raise ValueError("ASG-CV forced observation context differs")


def run_forced_campaign(
    directory: Path,
    *,
    pilot_schedule: AsgcvPairSchedule,
    source_commit: str,
    launch_authority_sha256: str,
    model_revision: str,
    fixture_sha256: str,
    execute_one: Callable[[int], AsgcvForcedObservation],
) -> bytes:
    """Run or resume the fixed 32-pair pilot and repeat its edge observations."""

    pilot_schedule.validated()
    if (
        not directory.is_dir()
        or directory.is_symlink()
        or pilot_schedule.pair_count != ASGCV_FORCED_PAIR_COUNT
        or not callable(execute_one)
    ):
        raise ValueError("ASG-CV forced campaign context differs")
    result_path = directory / "result.json"
    observations: list[AsgcvForcedObservation] = []
    for ordinal in range(ASGCV_FORCED_PAIR_COUNT):
        path = directory / f"candidate-{ordinal:06d}.json"
        if path.exists():
            observation = _read_observation(path)
        else:
            observation = execute_one(ordinal)
            if type(observation) is not AsgcvForcedObservation:
                raise ValueError("ASG-CV forced candidate result differs")
            _validate_context(
                observation,
                pilot_schedule=pilot_schedule,
                source_commit=source_commit,
                launch_authority_sha256=launch_authority_sha256,
                model_revision=model_revision,
                fixture_sha256=fixture_sha256,
            )
            _write_atomic(path, canonical_asgcv_forced_observation_bytes(observation))
        _validate_context(
            observation,
            pilot_schedule=pilot_schedule,
            source_commit=source_commit,
            launch_authority_sha256=launch_authority_sha256,
            model_revision=model_revision,
            fixture_sha256=fixture_sha256,
        )
        observations.append(observation)
    if result_path.exists():
        raw = result_path.read_bytes()
        try:
            result = AsgcvForcedResult.from_mapping(json.loads(raw))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("ASG-CV forced result bytes differ") from error
        if canonical_asgcv_forced_result_bytes(result) != raw:
            raise ValueError("ASG-CV forced result bytes differ")
        return raw
    repeated = tuple(execute_one(ordinal) for ordinal in (0, 31))
    for repeat, original in zip(repeated, (observations[0], observations[31]), strict=True):
        _validate_context(
            repeat,
            pilot_schedule=pilot_schedule,
            source_commit=source_commit,
            launch_authority_sha256=launch_authority_sha256,
            model_revision=model_revision,
            fixture_sha256=fixture_sha256,
        )
        if (
            repeat.gradient_sha256 != original.gradient_sha256
            or repeat.same_score != original.same_score
            or repeat.different_score != original.different_score
            or repeat.boundary_norms != original.boundary_norms
        ):
            raise ValueError("ASG-CV forced repeat evidence differs")
    result = AsgcvForcedResult.from_observations(
        tuple(observations),
        repeat_checked_ordinals=(0, 31),
        repeat_gradient_sha256s=(repeated[0].gradient_sha256, repeated[1].gradient_sha256),
    )
    raw = canonical_asgcv_forced_result_bytes(result)
    _write_atomic(result_path, raw)
    return raw


def main(argv: list[str] | None = None) -> int:
    """Authenticate train-only inputs and run one resumable forced-verdict campaign."""

    args = parse_args(argv)
    if _authenticated_source_commit(_REPOSITORY_ROOT) != args.source_commit:
        raise ValueError("ASG-CV forced executing source commit differs")
    local = load_p32_local_authority(
        args.p32_authority,
        args.train_manifest,
        source_commit=args.source_commit,
    )
    snapshot = load_snapshot_authority(
        root=args.model_root,
        manifest_path=args.snapshot_manifest,
    )
    fixture = load_fixture_authority(args.fixture)
    if (
        fixture.source_commit != args.source_commit
        or fixture.model_revision != local.rollout_authority.model_revision
        or fixture.prompt_utf8 != local.prompt_utf8
        or fixture.patch_tokens_per_image != local.patch_tokens_per_image
    ):
        raise ValueError("ASG-CV forced model fixture binding differs")
    adapter = load_qwen_adapter(
        LoadedAuthority(snapshot=snapshot, fixture=fixture),
        factory=TransformersFactory(),
    )
    fixture_sha256 = hashlib.sha256(args.fixture.read_bytes()).hexdigest()

    def execute_one(ordinal: int) -> AsgcvForcedObservation:
        return run_forced_candidate(
            adapter,
            ordinal=ordinal,
            images=local.images,
            prompt_utf8=local.prompt_utf8,
            attribute_token_span=local.attribute_token_span,
            patch_tokens_per_image=local.patch_tokens_per_image,
            completion_protocol=local.completion_protocol,
            pilot_schedule=local.pilot_schedule,
            source_commit=args.source_commit,
            launch_authority_sha256=local.authority_sha256,
            model_revision=local.rollout_authority.model_revision,
            fixture_sha256=fixture_sha256,
        )

    raw = run_forced_campaign(
        args.output_directory,
        pilot_schedule=local.pilot_schedule,
        source_commit=args.source_commit,
        launch_authority_sha256=local.authority_sha256,
        model_revision=local.rollout_authority.model_revision,
        fixture_sha256=fixture_sha256,
        execute_one=execute_one,
    )
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
