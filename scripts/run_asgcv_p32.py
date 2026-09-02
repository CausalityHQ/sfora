#!/usr/bin/env python3
"""Training-only ASG-CV P32 scientific candidate runner."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import sys
from collections.abc import Callable
from dataclasses import dataclass
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
    PreparedPair,
    TransformersFactory,
    capture_asgcv_patch_gradient,
    load_qwen_adapter,
)
from scripts.prepare_asgcv_p32_inputs import _authenticated_source_commit  # noqa: E402
from scripts.run_asgcv_e0 import (  # noqa: E402
    ASGCV_P32_FAILURE_SCHEMA,
    run_p32_campaign_with_failure_terminal,
)
from sfora.asgcv_pilot import (  # noqa: E402
    ASGCV_P32_BOUNDARY_NAMES,
    AsgcvP32Candidate,
    asgcv_p32_branch_exchange_energy_ppm,
    asgcv_p32_collapsed_exact_cosine,
    asgcv_p32_field_authority,
    validate_asgcv_p32_pilot_schedule,
)
from sfora.asgcv_predictor import source_bound_predictor  # noqa: E402
from sfora.asgcv_protocol import (  # noqa: E402
    AsgcvCompletionGroup,
    AsgcvCompletionProtocol,
    AsgcvPairSchedule,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    classify_asgcv_completion_group,
    derive_asgcv_rollout_seeds,
)
from sfora.saga_feasibility import (  # noqa: E402
    load_fixture_authority,
    load_snapshot_authority,
)


@dataclass(frozen=True, slots=True)
class LoadedP32LocalAuthority:
    """Authenticated local-only P32 inputs after partition reconstruction."""

    images: tuple[np.ndarray, ...]
    validation_images: tuple[np.ndarray, ...]
    optimization_images: tuple[np.ndarray, ...]
    predictor_train: tuple[tuple[str, ...], tuple[int, ...]]
    e0_validation: tuple[tuple[str, ...], tuple[int, ...]]
    e1_optimization: tuple[tuple[str, ...], tuple[int, ...]]
    partition_authority: AsgcvPartitionAuthority
    completion_protocol: AsgcvCompletionProtocol
    rollout_authority: AsgcvRolloutAuthority
    pilot_schedule: AsgcvPairSchedule
    prompt_utf8: str
    attribute_token_span: tuple[int, int]
    patch_tokens_per_image: int
    predictor_initialization_seed_sha256: str
    authority_sha256: str
    official_test_access: bool = False


def _strict_json(path: Path, *, role: str) -> tuple[bytes, dict[str, object]]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"ASG-CV P32 {role} path differs")
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"ASG-CV P32 {role} JSON differs") from error
    if (
        type(value) is not dict
        or raw
        != (
            json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        ).encode()
    ):
        raise ValueError(f"ASG-CV P32 {role} bytes differ")
    return raw, value


def _partition_rows(
    value: object,
    *,
    role: str,
    with_arrays: bool,
    root: Path,
) -> tuple[tuple[str, ...], tuple[int, ...], tuple[np.ndarray, ...]]:
    if type(value) is not list or not value:
        raise ValueError(f"ASG-CV P32 {role} rows differ")
    expected = (
        {"example_id", "label", "array_path", "array_sha256"}
        if with_arrays
        else {
            "example_id",
            "label",
        }
    )
    ids: list[str] = []
    labels: list[int] = []
    images: list[np.ndarray] = []
    for row in value:
        if (
            type(row) is not dict
            or set(row) != expected
            or type(row["example_id"]) is not str
            or not row["example_id"]
            or type(row["label"]) is not int
            or row["label"] < 0
        ):
            raise ValueError(f"ASG-CV P32 {role} row differs")
        ids.append(row["example_id"])
        labels.append(row["label"])
        if with_arrays:
            relative = row["array_path"]
            digest = row["array_sha256"]
            if (
                type(relative) is not str
                or not relative
                or Path(relative).name != relative
                or type(digest) is not str
                or len(digest) != 64
                or any(character not in "0123456789abcdef" for character in digest)
            ):
                raise ValueError("ASG-CV P32 image authority differs")
            path = root / relative
            if path.is_symlink() or not path.is_file():
                raise ValueError("ASG-CV P32 image path differs")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("ASG-CV P32 image digest differs")
            try:
                with path.open("rb") as stream:
                    image = np.load(stream, allow_pickle=False)
            except (OSError, ValueError) as error:
                raise ValueError("ASG-CV P32 image payload differs") from error
            if (
                type(image) is not np.ndarray
                or image.dtype != np.dtype(np.uint8)
                or image.ndim != 3
                or image.shape[-1] != 3
                or any(size <= 0 for size in image.shape)
            ):
                raise ValueError("ASG-CV P32 image array differs")
            images.append(np.ascontiguousarray(image))
    if len(ids) != len(set(ids)):
        raise ValueError(f"ASG-CV P32 {role} identities differ")
    return tuple(ids), tuple(labels), tuple(images)


def load_p32_local_authority(
    authority_path: Path,
    train_manifest_path: Path,
    *,
    source_commit: str,
) -> LoadedP32LocalAuthority:
    """Authenticate train-only arrays and reconstruct every P32 authority relation."""

    manifest_raw, manifest = _strict_json(train_manifest_path, role="train manifest")
    if (
        set(manifest)
        != {
            "schema",
            "official_test_access",
            "predictor_train",
            "e0_validation",
            "e1_optimization",
        }
        or manifest["schema"] != "sfora-cars-train-p32-manifest-v1"
        or manifest["official_test_access"] is not False
    ):
        raise ValueError("ASG-CV P32 train manifest schema differs")
    predictor_ids, predictor_labels, images = _partition_rows(
        manifest["predictor_train"],
        role="predictor training",
        with_arrays=True,
        root=train_manifest_path.parent,
    )
    validation_ids, validation_labels, validation_images = _partition_rows(
        manifest["e0_validation"],
        role="E0 validation",
        with_arrays=True,
        root=train_manifest_path.parent,
    )
    optimization_ids, optimization_labels, optimization_images = _partition_rows(
        manifest["e1_optimization"],
        role="E1 optimization",
        with_arrays=True,
        root=train_manifest_path.parent,
    )
    authority_raw, authority = _strict_json(authority_path, role="launch authority")
    expected = {
        "schema",
        "source_commit",
        "prompt_utf8",
        "attribute_token_span",
        "patch_tokens_per_image",
        "predictor_initialization_seed_sha256",
        "partition_authority",
        "completion_protocol",
        "rollout_authority",
        "pilot_schedule",
    }
    if (
        set(authority) != expected
        or authority["schema"] != "sfora-asgcv-p32-launch-v1"
        or authority["source_commit"] != source_commit
        or type(source_commit) is not str
        or len(source_commit) != 40
        or any(character not in "0123456789abcdef" for character in source_commit)
        or type(authority["prompt_utf8"]) is not str
        or not authority["prompt_utf8"]
        or type(authority["attribute_token_span"]) is not list
        or len(authority["attribute_token_span"]) != 2
        or any(type(value) is not int for value in authority["attribute_token_span"])
        or not 0 <= authority["attribute_token_span"][0] < authority["attribute_token_span"][1]
        or type(authority["patch_tokens_per_image"]) is not int
        or authority["patch_tokens_per_image"] <= 0
    ):
        raise ValueError("ASG-CV P32 launch authority differs")
    seed = authority["predictor_initialization_seed_sha256"]
    if type(seed) is not str or len(seed) != 64 or any(c not in "0123456789abcdef" for c in seed):
        raise ValueError("ASG-CV P32 predictor initialization differs")
    partition = AsgcvPartitionAuthority.from_mapping(authority["partition_authority"])
    if partition.source_manifest_sha256 != hashlib.sha256(manifest_raw).hexdigest():
        raise ValueError("ASG-CV P32 source manifest binding differs")
    protocol = AsgcvCompletionProtocol.from_mapping(authority["completion_protocol"])
    rollout = AsgcvRolloutAuthority.from_mapping(authority["rollout_authority"])
    schedule = AsgcvPairSchedule.from_mapping(authority["pilot_schedule"])
    predictor_train = (predictor_ids, predictor_labels)
    e0_validation = (validation_ids, validation_labels)
    e1_optimization = (optimization_ids, optimization_labels)
    validate_asgcv_p32_pilot_schedule(
        schedule,
        partition_authority=partition,
        source_commit=source_commit,
        predictor_train=predictor_train,
        e0_validation=e0_validation,
        e1_optimization=e1_optimization,
    )
    return LoadedP32LocalAuthority(
        images=images,
        validation_images=validation_images,
        optimization_images=optimization_images,
        predictor_train=predictor_train,
        e0_validation=e0_validation,
        e1_optimization=e1_optimization,
        partition_authority=partition,
        completion_protocol=protocol,
        rollout_authority=rollout,
        pilot_schedule=schedule,
        prompt_utf8=authority["prompt_utf8"],
        attribute_token_span=tuple(authority["attribute_token_span"]),
        patch_tokens_per_image=authority["patch_tokens_per_image"],
        predictor_initialization_seed_sha256=seed,
        authority_sha256=hashlib.sha256(authority_raw).hexdigest(),
    )


class P32Adapter(Protocol):
    """Only the model operations needed by one P32 candidate."""

    def prepare_image_pair(
        self,
        images: object,
        prompt_utf8: object,
        attribute_token_span: object,
        patch_tokens_per_image: object,
    ) -> PreparedPair: ...

    def generate(
        self,
        pair: object,
        seed: int,
        *,
        temperature: float,
        top_p: float,
        max_new_tokens: int,
    ) -> tuple[int, ...]: ...

    def score_completions(
        self, pair: object, completion_ids: tuple[tuple[int, ...], ...]
    ) -> tuple[float, ...]: ...

    def collapsed_verdict_patch_gradient(
        self,
        pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> object: ...


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse a local-only P32 execution boundary."""

    values = list(argv) if argv is not None else sys.argv[1:]
    options = [token.split("=", 1)[0] for token in values if token.startswith("--")]
    if len(options) != len(set(options)):
        raise SystemExit("duplicate ASG-CV P32 option")
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--p32-authority", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execute-p32", required=True, action="store_true")
    parsed = parser.parse_args(values)
    if len(parsed.source_commit) != 40 or any(
        character not in "0123456789abcdef" for character in parsed.source_commit
    ):
        parser.error("source commit must be 40 lowercase hex")
    if not parsed.model_root.is_dir():
        parser.error("model root must be an existing directory")
    for name in ("snapshot_manifest", "fixture", "p32_authority", "train_manifest"):
        path = getattr(parsed, name)
        if path.is_symlink() or not path.is_file():
            parser.error(f"{name.replace('_', ' ')} must be an existing regular file")
    if not parsed.output_directory.is_dir() or parsed.output_directory.is_symlink():
        parser.error("output directory must be an existing regular directory")
    return parsed


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


def _timed[Value](operation: Callable[[], Value]) -> tuple[Value, int]:
    _synchronize()
    started = perf_counter_ns()
    value = operation()
    _synchronize()
    return value, max(1, perf_counter_ns() - started)


def _array(value: object, *, name: str) -> np.ndarray:
    if type(value) is torch.Tensor:
        tensor = cast(torch.Tensor, value)
        value = tensor.detach().cpu().contiguous().numpy()
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or value.size == 0
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError(f"ASG-CV P32 {name} array differs")
    return np.ascontiguousarray(value)


def _field_evidence(value: object, *, role: str) -> tuple[str, float, np.ndarray]:
    array = _array(value, name=role)
    digest, norm = asgcv_p32_field_authority(array, role=role)
    return digest, norm, array


def run_p32_candidate(
    adapter: P32Adapter,
    *,
    ordinal: int,
    images: tuple[np.ndarray, ...],
    prompt_utf8: str,
    attribute_token_span: tuple[int, int],
    patch_tokens_per_image: int,
    completion_protocol: AsgcvCompletionProtocol,
    rollout_authority: AsgcvRolloutAuthority,
    pilot_schedule: AsgcvPairSchedule,
    source_commit: str,
    fixture_sha256: str,
    launch_authority_sha256: str,
    predictor_initialization_seed_sha256: str,
    partition_authority_sha256: str,
    pooler_state_sha256: str,
    exact_diagnostic_available: bool,
    exact_capture: Callable[[PreparedPair, AsgcvCompletionGroup], object],
    predictor_probe: Callable[[object, int], object],
) -> tuple[AsgcvCompletionGroup, AsgcvP32Candidate]:
    """Execute one digest-only P32 row without resetting phase memory peaks."""

    if (
        type(ordinal) is not int
        or not 0 <= ordinal < pilot_schedule.pair_count
        or type(images) is not tuple
        or not callable(exact_capture)
        or not callable(predictor_probe)
        or type(exact_diagnostic_available) is not bool
    ):
        raise ValueError("ASG-CV P32 candidate execution context differs")
    completion_protocol.validated()
    rollout_authority.validated()
    pilot_schedule.validated()
    pair_row = pilot_schedule.pairs[ordinal]
    if max(pair_row.left_index, pair_row.right_index) >= len(images):
        raise ValueError("ASG-CV P32 image schedule differs")

    _reset_candidate_peak()
    _synchronize()
    total_started = perf_counter_ns()
    pair, prepare_elapsed_ns = _timed(
        lambda: adapter.prepare_image_pair(
            (images[pair_row.left_index], images[pair_row.right_index]),
            prompt_utf8,
            attribute_token_span,
            patch_tokens_per_image,
        )
    )
    generation_seeds = derive_asgcv_rollout_seeds(
        rollout_authority,
        candidate_pair_ordinal=ordinal,
    )
    completion_ids, generate_elapsed_ns = _timed(
        lambda: tuple(
            adapter.generate(
                pair,
                seed,
                temperature=rollout_authority.temperature,
                top_p=rollout_authority.top_p,
                max_new_tokens=rollout_authority.max_new_tokens,
            )
            for seed in generation_seeds
        )
    )
    group = classify_asgcv_completion_group(
        completion_ids,
        pair_row.relation_sign,
        completion_protocol,
        rollout_authority=rollout_authority,
        candidate_pair_ordinal=ordinal,
    )
    run_exact_diagnostic = group.nonzero_reward_variance and exact_diagnostic_available
    completion_scores, score_elapsed_ns = _timed(
        lambda: adapter.score_completions(pair, group.completion_ids)
    )
    if (
        type(completion_scores) is not tuple
        or len(completion_scores) != 8
        or any(type(score) is not float or not math.isfinite(score) for score in completion_scores)
    ):
        raise ValueError("ASG-CV P32 completion scores differ")

    correct = tuple(
        index
        for index, (valid, verdict) in enumerate(
            zip(group.valid_flags, group.verdict_relation_signs, strict=True)
        )
        if valid and verdict == pair_row.relation_sign
    )
    incorrect = tuple(
        index
        for index, (valid, verdict) in enumerate(
            zip(group.valid_flags, group.verdict_relation_signs, strict=True)
        )
        if valid and verdict == -pair_row.relation_sign
    )
    lowest_indices = (correct[0], incorrect[0]) if correct and incorrect else None
    branch_exchange_distinct = (
        len(correct) >= 2
        and len(incorrect) >= 2
        and group.completion_ids[correct[0]] != group.completion_ids[correct[-1]]
        and group.completion_ids[incorrect[0]] != group.completion_ids[incorrect[-1]]
    )
    highest_indices = (correct[-1], incorrect[-1]) if branch_exchange_distinct else None

    lowest_target = None
    lowest_digest = None
    lowest_norm = None
    lowest_field = None
    collapsed_scores = None
    backend_coefficient_ppm = None
    boundary_norms = None
    collapsed_elapsed_ns = 0
    predictor_elapsed_ns = 0
    if lowest_indices is not None:
        lowest_target, collapsed_elapsed_ns = _timed(
            lambda: adapter.collapsed_verdict_patch_gradient(
                pair,
                correct_completion_ids=group.completion_ids[lowest_indices[0]],
                incorrect_completion_ids=group.completion_ids[lowest_indices[1]],
            )
        )
        lowest_digest, lowest_norm, lowest_field = _field_evidence(
            getattr(lowest_target, "predicted_gradient", None),
            role="lowest-collapsed-gradient",
        )
        collapsed_scores = getattr(lowest_target, "branch_scores", None)
        coefficient = getattr(lowest_target, "coefficient", None)
        if (
            type(collapsed_scores) is not tuple
            or len(collapsed_scores) != 2
            or type(coefficient) is not float
            or not math.isfinite(coefficient)
        ):
            raise ValueError("ASG-CV P32 collapsed target differs")
        backend_coefficient_ppm = int(round(coefficient * 1_000_000.0))
        boundary_names = getattr(lowest_target, "boundary_names", None)
        boundary = _array(
            getattr(lowest_target, "boundary_predicted_gradient", None),
            name="boundary gradient",
        )
        if boundary_names != ASGCV_P32_BOUNDARY_NAMES or boundary.shape[0] != 4:
            raise ValueError("ASG-CV P32 boundary target differs")
        boundary_norms = tuple(
            float(np.sqrt(np.square(row.astype(np.float64)).sum(dtype=np.float64)))
            for row in boundary
        )
        _, predictor_elapsed_ns = _timed(
            lambda: predictor_probe(
                getattr(lowest_target, "patch_tokens", None),
                pair_row.relation_sign,
            )
        )
        del lowest_target

    highest_digest = None
    highest_norm = None
    highest_scores = None
    highest_backend_coefficient_ppm = None
    exchange_energy = None
    exchange_elapsed_ns = 0
    if highest_indices is not None:
        highest_target, exchange_elapsed_ns = _timed(
            lambda: adapter.collapsed_verdict_patch_gradient(
                pair,
                correct_completion_ids=group.completion_ids[highest_indices[0]],
                incorrect_completion_ids=group.completion_ids[highest_indices[1]],
            )
        )
        highest_digest, highest_norm, highest_field = _field_evidence(
            getattr(highest_target, "predicted_gradient", None),
            role="highest-collapsed-gradient",
        )
        highest_scores = getattr(highest_target, "branch_scores", None)
        highest_coefficient = getattr(highest_target, "coefficient", None)
        if (
            type(highest_scores) is not tuple
            or len(highest_scores) != 2
            or type(highest_coefficient) is not float
            or not math.isfinite(highest_coefficient)
        ):
            raise ValueError("ASG-CV P32 highest target differs")
        highest_backend_coefficient_ppm = int(round(highest_coefficient * 1_000_000.0))
        exchange_energy = asgcv_p32_branch_exchange_energy_ppm(
            lowest_field,
            highest_field,
        )
        del highest_target

    exact_digest = None
    exact_norm = None
    exact_tokens = None
    exact_cosine = None
    exact_elapsed_ns = 0
    if run_exact_diagnostic:
        exact_target, exact_elapsed_ns = _timed(lambda: exact_capture(pair, group))
        exact_digest, exact_norm, exact_field = _field_evidence(
            getattr(exact_target, "exact_gradient", None),
            role="exact-gradient",
        )
        replay = getattr(exact_target, "replay", None)
        exact_tokens = getattr(replay, "generated_tokens", None)
        if type(exact_tokens) is not int or exact_tokens <= 0:
            raise ValueError("ASG-CV P32 exact replay target differs")
        if (
            lowest_field is not None
            and lowest_norm is not None
            and exact_norm is not None
            and lowest_norm > 0.0
            and exact_norm > 0.0
        ):
            exact_cosine = asgcv_p32_collapsed_exact_cosine(lowest_field, exact_field)
        del exact_target
    _synchronize()
    total_elapsed_ns = max(1, perf_counter_ns() - total_started)
    candidate = AsgcvP32Candidate(
        source_commit=source_commit,
        model_revision=rollout_authority.model_revision,
        fixture_sha256=fixture_sha256,
        launch_authority_sha256=launch_authority_sha256,
        predictor_initialization_seed_sha256=predictor_initialization_seed_sha256,
        partition_authority_sha256=partition_authority_sha256,
        pilot_schedule_sha256=pilot_schedule.sha256(),
        completion_protocol_sha256=completion_protocol.sha256(),
        rollout_authority_sha256=rollout_authority.sha256(),
        completion_group_sha256=group.sha256(),
        pooler_state_sha256=pooler_state_sha256,
        candidate_pair_ordinal=ordinal,
        pair_ordinals=(pair_row.left_index, pair_row.right_index),
        relation_sign=pair_row.relation_sign,
        generation_seeds=generation_seeds,
        rewards=group.rewards,
        valid_flags=group.valid_flags,
        verdict_relation_signs=group.verdict_relation_signs,
        attribute_span_lengths=tuple(
            0 if span is None else span[1] - span[0] for span in group.attribute_spans
        ),
        generated_token_counts=tuple(len(completion) for completion in group.completion_ids),
        completion_scores=completion_scores,
        lowest_branch_indices=lowest_indices,
        highest_branch_indices=highest_indices,
        branch_exchange_distinct=branch_exchange_distinct,
        collapsed_branch_scores=collapsed_scores,
        collapsed_backend_coefficient_ppm=backend_coefficient_ppm,
        highest_branch_scores=highest_scores,
        highest_backend_coefficient_ppm=highest_backend_coefficient_ppm,
        lowest_gradient_sha256=lowest_digest,
        highest_gradient_sha256=highest_digest,
        lowest_gradient_norm=lowest_norm,
        highest_gradient_norm=highest_norm,
        branch_exchange_energy_ppm=exchange_energy,
        boundary_names=ASGCV_P32_BOUNDARY_NAMES,
        boundary_norms=boundary_norms,
        exact_gradient_sha256=exact_digest,
        exact_gradient_norm=exact_norm,
        exact_replay_generated_tokens=exact_tokens,
        collapsed_exact_cosine=exact_cosine,
        prepare_elapsed_ns=prepare_elapsed_ns,
        generate_elapsed_ns=generate_elapsed_ns,
        score_elapsed_ns=score_elapsed_ns,
        collapsed_replay_elapsed_ns=collapsed_elapsed_ns,
        branch_exchange_replay_elapsed_ns=exchange_elapsed_ns,
        exact_replay_elapsed_ns=exact_elapsed_ns,
        predictor_forward_elapsed_ns=predictor_elapsed_ns,
        candidate_total_elapsed_ns=total_elapsed_ns,
        peak_cuda_reserved_bytes=_peak_cuda_reserved_bytes(),
        peak_rss_bytes=_process_peak_rss_bytes(),
    ).validated()
    return group, candidate


def _predictor_probe(adapter: object, seed_sha256: str) -> Callable[[object, int], object]:
    channel_dimensions = getattr(adapter, "pooler_token_dim", None)
    pooler = getattr(adapter, "pooler", None)
    if type(channel_dimensions) is not int or not isinstance(pooler, torch.nn.Module):
        raise ValueError("ASG-CV P32 predictor runtime differs")
    try:
        device = next(pooler.parameters()).device
    except StopIteration as error:
        raise ValueError("ASG-CV P32 predictor device differs") from error
    predictor = source_bound_predictor(
        channel_dimensions=channel_dimensions,
        seed_sha256=seed_sha256,
    ).to(device)

    def probe(tokens: object, relation_sign: int) -> object:
        if type(tokens) is not torch.Tensor:
            raise ValueError("ASG-CV P32 predictor tokens differ")
        tensor = cast(torch.Tensor, tokens)
        if tensor.ndim != 3:
            raise ValueError("ASG-CV P32 predictor tokens differ")
        stopped = tensor.detach().float().unsqueeze(0)
        signs = torch.tensor([relation_sign], dtype=torch.int8, device=stopped.device)
        return predictor.predict_detached(stopped, signs)

    return probe


def main(argv: list[str] | None = None) -> int:
    """Authenticate local inputs, run one resumable P32 campaign, and emit its terminal."""

    args = parse_args(argv)
    if _authenticated_source_commit(_REPOSITORY_ROOT) != args.source_commit:
        raise ValueError("ASG-CV P32 executing source commit differs")
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
        or fixture.attention_layer != 26
    ):
        raise ValueError("ASG-CV P32 model fixture binding differs")
    adapter = load_qwen_adapter(
        LoadedAuthority(snapshot=snapshot, fixture=fixture),
        factory=TransformersFactory(),
    )
    predictor_probe = _predictor_probe(adapter, local.predictor_initialization_seed_sha256)
    fixture_sha256 = hashlib.sha256(args.fixture.read_bytes()).hexdigest()

    def execute_one(
        ordinal: int,
        exact_diagnostic_available: bool,
    ) -> tuple[AsgcvCompletionGroup, AsgcvP32Candidate]:
        return run_p32_candidate(
            adapter,
            ordinal=ordinal,
            images=local.images,
            prompt_utf8=local.prompt_utf8,
            attribute_token_span=local.attribute_token_span,
            patch_tokens_per_image=local.patch_tokens_per_image,
            completion_protocol=local.completion_protocol,
            rollout_authority=local.rollout_authority,
            pilot_schedule=local.pilot_schedule,
            source_commit=args.source_commit,
            fixture_sha256=fixture_sha256,
            launch_authority_sha256=local.authority_sha256,
            predictor_initialization_seed_sha256=local.predictor_initialization_seed_sha256,
            partition_authority_sha256=local.partition_authority.sha256(),
            pooler_state_sha256=adapter.pooler_sha256,
            exact_diagnostic_available=exact_diagnostic_available,
            exact_capture=lambda pair, group: capture_asgcv_patch_gradient(
                adapter,
                pair,
                group,
                attention_layer=fixture.attention_layer,
            ),
            predictor_probe=predictor_probe,
        )

    raw = run_p32_campaign_with_failure_terminal(
        args.output_directory,
        rollout_authority=local.rollout_authority,
        pilot_schedule=local.pilot_schedule,
        partition_authority=local.partition_authority,
        source_commit=args.source_commit,
        launch_authority_sha256=local.authority_sha256,
        predictor_initialization_seed_sha256=local.predictor_initialization_seed_sha256,
        predictor_train=local.predictor_train,
        e0_validation=local.e0_validation,
        e1_optimization=local.e1_optimization,
        execute_one=execute_one,
    )
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    terminal = json.loads(raw)
    return 3 if terminal.get("schema") == ASGCV_P32_FAILURE_SCHEMA else 0


if __name__ == "__main__":
    raise SystemExit(main())
