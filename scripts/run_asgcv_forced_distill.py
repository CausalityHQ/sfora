#!/usr/bin/env python3
"""Capture and fit the local-only ASG-CV forced-gradient student."""

from __future__ import annotations

import argparse
import gc
import hashlib
import os
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

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
from sfora.asgcv import AsgcvSrhtAuthority  # noqa: E402
from sfora.asgcv_forced_distill import (  # noqa: E402
    ASGCV_FORCED_DISTILL_SHAPE,
    ASGCV_FORCED_DISTILL_TRAIN_PAIRS,
    ASGCV_FORCED_DISTILL_VALIDATION_PAIRS,
    ForcedDistillCapture,
    ForcedDistillResult,
    build_forced_distill_schedule,
    canonical_forced_distill_capture_bytes,
    canonical_forced_distill_result_bytes,
    dense_gradient_cosine_with_liveness,
    relation_correct_gradient,
    validate_forced_distill_capture_bytes,
    validate_forced_distill_result_bytes,
)
from sfora.asgcv_predictor import (  # noqa: E402
    AsgcvPatchGradientPredictor,
    canonical_predictor_state_bytes,
    predictor_state_sha256,
    predictor_training_loss,
    source_bound_predictor,
)
from sfora.asgcv_protocol import (  # noqa: E402
    AsgcvCompletionProtocol,
    AsgcvPair,
    AsgcvPairSchedule,
)
from sfora.saga_feasibility import (  # noqa: E402
    load_fixture_authority,
    load_snapshot_authority,
)

_BOUNDARY_NAMES = ("merger", "deepstack-0", "deepstack-1", "deepstack-2")
ASGCV_FORCED_DISTILL_EPOCHS = 20
ASGCV_FORCED_DISTILL_LEARNING_RATE = 1e-3
ASGCV_FORCED_DISTILL_WEIGHT_DECAY = 1e-4
ASGCV_FORCED_DISTILL_SRHT_DIMENSIONS = 256
_SRHT_DOMAIN = b"sfora-asgcv-forced-distill-srht-v1\0"


class ForcedDistillAdapter(Protocol):
    """Only Qwen operations needed to capture one forced target."""

    def prepare_image_pair(
        self,
        images: object,
        prompt_utf8: object,
        attribute_token_span: object,
        patch_tokens_per_image: object,
    ) -> object: ...

    def collapsed_verdict_patch_gradient(
        self,
        pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> object: ...


def _array(value: object, *, name: str) -> np.ndarray:
    try:
        import torch
    except ImportError:
        torch = None  # type: ignore[assignment]
    if torch is not None and type(value) is torch.Tensor:
        value = value.detach().float().cpu().contiguous().numpy()
    if (
        type(value) is not np.ndarray
        or value.dtype != np.dtype(np.float32)
        or value.shape != ASGCV_FORCED_DISTILL_SHAPE
        or not bool(np.isfinite(value).all())
    ):
        raise ValueError(f"ASG-CV forced distill {name} differs")
    return np.ascontiguousarray(value)


def capture_forced_distill_pair(
    adapter: ForcedDistillAdapter,
    *,
    role: str,
    pair: AsgcvPair,
    schedule: AsgcvPairSchedule,
    images: tuple[np.ndarray, ...],
    prompt_utf8: str,
    attribute_token_span: tuple[int, int],
    patch_tokens_per_image: int,
    completion_protocol: AsgcvCompletionProtocol,
    source_commit: str,
    launch_authority_sha256: str,
) -> tuple[bytes, np.ndarray, np.ndarray]:
    """Capture one fixed-order target and orient it to the registered relation."""

    schedule.validated()
    completion_protocol.validated()
    if (
        type(pair) is not AsgcvPair
        or pair.ordinal >= schedule.pair_count
        or schedule.pairs[pair.ordinal] != pair
        or max(pair.left_index, pair.right_index) >= len(images)
    ):
        raise ValueError("ASG-CV forced distill pair differs")
    prepared = adapter.prepare_image_pair(
        (images[pair.left_index], images[pair.right_index]),
        prompt_utf8,
        attribute_token_span,
        patch_tokens_per_image,
    )
    target = adapter.collapsed_verdict_patch_gradient(
        prepared,
        correct_completion_ids=completion_protocol.same_prefix_ids,
        incorrect_completion_ids=completion_protocol.different_prefix_ids,
    )
    if getattr(target, "boundary_names", None) != _BOUNDARY_NAMES:
        raise ValueError("ASG-CV forced distill boundary differs")
    patches = _array(getattr(target, "patch_tokens", None), name="patch tokens")
    fixed_gradient = _array(getattr(target, "predicted_gradient", None), name="gradient")
    gradient = relation_correct_gradient(fixed_gradient, pair.relation_sign)
    capture = ForcedDistillCapture(
        source_commit=source_commit,
        launch_authority_sha256=launch_authority_sha256,
        schedule_sha256=schedule.sha256(),
        role=role,
        pair_ordinal=pair.ordinal,
        pair_indices=(pair.left_index, pair.right_index),
        relation_sign=pair.relation_sign,
        patch_sha256=hashlib.sha256(patches.tobytes()).hexdigest(),
        gradient_sha256=hashlib.sha256(gradient.tobytes()).hexdigest(),
        array_shape=ASGCV_FORCED_DISTILL_SHAPE,
    ).validated()
    return canonical_forced_distill_capture_bytes(capture), patches, gradient


def _paths(directory: Path, role: str, ordinal: int) -> tuple[Path, Path, Path]:
    stem = f"{role}-{ordinal:06d}"
    return (
        directory / f"{stem}.json",
        directory / f"{stem}-patch.npy",
        directory / f"{stem}-gradient.npy",
    )


def _load_array(path: Path) -> np.ndarray:
    try:
        with path.open("rb") as stream:
            value = np.load(stream, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError("ASG-CV forced distill array file differs") from error
    return _array(value, name="array")


def _validate_triple(
    directory: Path,
    *,
    role: str,
    schedule: AsgcvPairSchedule,
    ordinal: int,
) -> None:
    receipt_path, patch_path, gradient_path = _paths(directory, role, ordinal)
    capture = validate_forced_distill_capture_bytes(
        receipt_path.read_bytes(),
        patch_tokens=_load_array(patch_path),
        exact_gradient=_load_array(gradient_path),
    )
    pair = schedule.pairs[ordinal]
    if (
        capture.role != role
        or capture.pair_ordinal != ordinal
        or capture.schedule_sha256 != schedule.sha256()
        or capture.pair_indices != (pair.left_index, pair.right_index)
        or capture.relation_sign != pair.relation_sign
    ):
        raise ValueError("ASG-CV forced distill capture context differs")


def _read_triple(
    directory: Path,
    *,
    role: str,
    schedule: AsgcvPairSchedule,
    ordinal: int,
) -> tuple[ForcedDistillCapture, np.ndarray, np.ndarray]:
    _validate_triple(directory, role=role, schedule=schedule, ordinal=ordinal)
    receipt_path, patch_path, gradient_path = _paths(directory, role, ordinal)
    patches = _load_array(patch_path)
    gradient = _load_array(gradient_path)
    capture = validate_forced_distill_capture_bytes(
        receipt_path.read_bytes(),
        patch_tokens=patches,
        exact_gradient=gradient,
    )
    return capture, patches, gradient


def _write_array(path: Path, value: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())


def _write_triple(
    directory: Path,
    *,
    role: str,
    ordinal: int,
    receipt: bytes,
    patches: np.ndarray,
    gradient: np.ndarray,
) -> None:
    final = _paths(directory, role, ordinal)
    partial = tuple(path.with_name(path.name + ".partial") for path in final)
    if any(path.exists() for path in (*final, *partial)):
        raise ValueError("ASG-CV forced distill output exists")
    try:
        with partial[0].open("xb") as stream:
            stream.write(receipt)
            stream.flush()
            os.fsync(stream.fileno())
        _write_array(partial[1], patches)
        _write_array(partial[2], gradient)
        for source, destination in zip(partial, final, strict=True):
            os.replace(source, destination)
        directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        for path in partial:
            if path.exists():
                path.unlink()


def run_capture_phase(
    directory: Path,
    *,
    role: str,
    schedule: AsgcvPairSchedule,
    execute_one: Callable[[int], tuple[bytes, np.ndarray, np.ndarray]],
    maximum_new_rows: int | None = None,
) -> int:
    """Validate a contiguous prefix and append a bounded number of captures."""

    if (
        not isinstance(directory, Path)
        or not directory.is_dir()
        or role not in {"train", "validation"}
        or not callable(execute_one)
        or (maximum_new_rows is not None and maximum_new_rows <= 0)
        or tuple(directory.glob("*.partial"))
    ):
        raise ValueError("ASG-CV forced distill capture phase differs")
    schedule.validated()
    prefix = 0
    for ordinal in range(schedule.pair_count):
        existing = tuple(path.exists() for path in _paths(directory, role, ordinal))
        if all(existing):
            _validate_triple(directory, role=role, schedule=schedule, ordinal=ordinal)
            prefix += 1
            continue
        if any(existing):
            raise ValueError("ASG-CV forced distill partial triple differs")
        if any(
            any(path.exists() for path in _paths(directory, role, later))
            for later in range(ordinal + 1, schedule.pair_count)
        ):
            raise ValueError("ASG-CV forced distill ordinal gap differs")
        break
    stop = schedule.pair_count
    if maximum_new_rows is not None:
        stop = min(stop, prefix + maximum_new_rows)
    for ordinal in range(prefix, stop):
        receipt, patches, gradient = execute_one(ordinal)
        validate_forced_distill_capture_bytes(
            receipt,
            patch_tokens=patches,
            exact_gradient=gradient,
        )
        _write_triple(
            directory,
            role=role,
            ordinal=ordinal,
            receipt=receipt,
            patches=patches,
            gradient=gradient,
        )
        _validate_triple(directory, role=role, schedule=schedule, ordinal=ordinal)
    return stop


def forced_distill_srht_authority(initialization_seed_sha256: str) -> AsgcvSrhtAuthority:
    """Derive the fixed 256-dimensional predictor-loss projection."""

    if (
        type(initialization_seed_sha256) is not str
        or len(initialization_seed_sha256) != 64
        or any(character not in "0123456789abcdef" for character in initialization_seed_sha256)
    ):
        raise ValueError("ASG-CV forced distill initialization seed differs")
    return AsgcvSrhtAuthority(
        input_dimensions=ASGCV_FORCED_DISTILL_SHAPE[-1],
        padded_dimensions=ASGCV_FORCED_DISTILL_SHAPE[-1],
        output_dimensions=ASGCV_FORCED_DISTILL_SRHT_DIMENSIONS,
        seed_sha256=hashlib.sha256(
            _SRHT_DOMAIN + bytes.fromhex(initialization_seed_sha256)
        ).hexdigest(),
    ).validated()


def train_predictor_epoch(
    predictor: AsgcvPatchGradientPredictor,
    optimizer: torch.optim.Optimizer,
    rows: tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], ...],
    *,
    srht_authority: AsgcvSrhtAuthority,
) -> float:
    """Apply one fixed-order epoch and return its mean normalized loss."""

    if (
        type(predictor) is not AsgcvPatchGradientPredictor
        or not isinstance(optimizer, torch.optim.Optimizer)
        or type(rows) is not tuple
        or not rows
    ):
        raise ValueError("ASG-CV forced distill training epoch differs")
    losses: list[float] = []
    predictor.train()
    for tokens, signs, exact in rows:
        if exact.requires_grad:
            raise ValueError("ASG-CV forced distill exact target differs")
        optimizer.zero_grad(set_to_none=True)
        predicted = predictor(tokens, signs)
        loss = predictor_training_loss(predicted, exact, srht_authority)
        loss.backward()
        optimizer.step()
        value = float(loss.detach())
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("ASG-CV forced distill training loss differs")
        losses.append(value)
    return float(sum(losses) / len(losses))


def fit_forced_distill_predictor(
    directory: Path,
    *,
    train_schedule: AsgcvPairSchedule,
    validation_schedule: AsgcvPairSchedule,
    initialization_seed_sha256: str,
    source_commit: str,
    launch_authority_sha256: str,
) -> tuple[AsgcvPatchGradientPredictor, ForcedDistillResult]:
    """Fit the frozen student recipe and evaluate the disjoint validation band once."""

    if (
        train_schedule.pair_count != ASGCV_FORCED_DISTILL_TRAIN_PAIRS
        or validation_schedule.pair_count != ASGCV_FORCED_DISTILL_VALIDATION_PAIRS
    ):
        raise ValueError("ASG-CV forced distill fit schedule differs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = source_bound_predictor(
        channel_dimensions=ASGCV_FORCED_DISTILL_SHAPE[-1],
        seed_sha256=initialization_seed_sha256,
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=ASGCV_FORCED_DISTILL_LEARNING_RATE,
        weight_decay=ASGCV_FORCED_DISTILL_WEIGHT_DECAY,
    )
    srht = forced_distill_srht_authority(initialization_seed_sha256)
    for _epoch in range(ASGCV_FORCED_DISTILL_EPOCHS):
        for ordinal in range(train_schedule.pair_count):
            capture, patches, gradient = _read_triple(
                directory,
                role="train",
                schedule=train_schedule,
                ordinal=ordinal,
            )
            rows = (
                (
                    torch.from_numpy(patches).unsqueeze(0).to(device),
                    torch.tensor([capture.relation_sign], dtype=torch.int8, device=device),
                    torch.from_numpy(gradient).unsqueeze(0).to(device),
                ),
            )
            train_predictor_epoch(predictor, optimizer, rows, srht_authority=srht)
    predictor.eval()
    cosines: list[float] = []
    prediction_nonzero: list[bool] = []
    with torch.no_grad():
        for ordinal in range(validation_schedule.pair_count):
            capture, patches, gradient = _read_triple(
                directory,
                role="validation",
                schedule=validation_schedule,
                ordinal=ordinal,
            )
            predicted = predictor(
                torch.from_numpy(patches).unsqueeze(0).to(device),
                torch.tensor([capture.relation_sign], dtype=torch.int8, device=device),
            )[0]
            predicted_array = predicted.detach().float().cpu().contiguous().numpy()
            cosine, live = dense_gradient_cosine_with_liveness(gradient, predicted_array)
            cosines.append(cosine)
            prediction_nonzero.append(live)
    result = ForcedDistillResult.from_cosines(
        source_commit=source_commit,
        launch_authority_sha256=launch_authority_sha256,
        train_schedule_sha256=train_schedule.sha256(),
        validation_schedule_sha256=validation_schedule.sha256(),
        predictor_state_sha256=predictor_state_sha256(predictor.cpu()),
        validation_cosines=tuple(cosines),
        prediction_nonzero_flags=tuple(prediction_nonzero),
    )
    return predictor, result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the strict local-only distillation boundary."""

    values = list(sys.argv[1:] if argv is None else argv)
    option_names = [value.split("=", 1)[0] for value in values if value.startswith("--")]
    if len(option_names) != len(set(option_names)):
        raise SystemExit("duplicate ASG-CV forced distill option")
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--p32-authority", required=True, type=Path)
    parser.add_argument("--train-manifest", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--execute-forced-distill", required=True, action="store_true")
    parsed = parser.parse_args(values)
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


def _write_atomic_bytes(path: Path, payload: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise ValueError("ASG-CV forced distill result exists")
    partial = path.with_name(path.name + ".partial")
    if partial.exists() or partial.is_symlink():
        raise ValueError("ASG-CV forced distill result partial exists")
    try:
        with partial.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(partial, path)
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if partial.exists():
            partial.unlink()


def main(argv: list[str] | None = None) -> int:
    """Run one authenticated capture, frozen fit, and validation campaign."""

    args = parse_args(argv)
    if _authenticated_source_commit(_REPOSITORY_ROOT) != args.source_commit:
        raise ValueError("ASG-CV forced distill executing source commit differs")
    result_path = args.output_directory / "result.json"
    predictor_path = args.output_directory / "predictor-state.bin"
    if result_path.exists():
        result = validate_forced_distill_result_bytes(result_path.read_bytes())
        if (
            predictor_path.is_symlink()
            or not predictor_path.is_file()
            or hashlib.sha256(predictor_path.read_bytes()).hexdigest()
            != result.predictor_state_sha256
        ):
            raise ValueError("ASG-CV forced distill predictor result binding differs")
        raw = canonical_forced_distill_result_bytes(result)
        sys.stdout.buffer.write(raw)
        sys.stdout.buffer.flush()
        return 0
    if predictor_path.exists() or tuple(args.output_directory.glob("*.partial")):
        raise ValueError("ASG-CV forced distill terminal output differs")

    local = load_p32_local_authority(
        args.p32_authority,
        args.train_manifest,
        source_commit=args.source_commit,
    )
    train_schedule = build_forced_distill_schedule(
        local.predictor_train[0],
        local.predictor_train[1],
        source_commit=args.source_commit,
        launch_authority_sha256=local.authority_sha256,
        role="train",
    )
    validation_schedule = build_forced_distill_schedule(
        local.e0_validation[0],
        local.e0_validation[1],
        source_commit=args.source_commit,
        launch_authority_sha256=local.authority_sha256,
        role="validation",
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
        raise ValueError("ASG-CV forced distill fixture binding differs")
    adapter = load_qwen_adapter(
        LoadedAuthority(snapshot=snapshot, fixture=fixture),
        factory=TransformersFactory(),
    )

    def capture_all(
        adapter_value: ForcedDistillAdapter,
    ) -> None:
        def capture_role(
            role: str,
            schedule: AsgcvPairSchedule,
            images: tuple[np.ndarray, ...],
        ) -> None:
            run_capture_phase(
                args.output_directory,
                role=role,
                schedule=schedule,
                execute_one=lambda ordinal: capture_forced_distill_pair(
                    adapter_value,
                    role=role,
                    pair=schedule.pairs[ordinal],
                    schedule=schedule,
                    images=images,
                    prompt_utf8=local.prompt_utf8,
                    attribute_token_span=local.attribute_token_span,
                    patch_tokens_per_image=local.patch_tokens_per_image,
                    completion_protocol=local.completion_protocol,
                    source_commit=args.source_commit,
                    launch_authority_sha256=local.authority_sha256,
                ),
            )

        capture_role("train", train_schedule, local.images)
        capture_role("validation", validation_schedule, local.validation_images)

    capture_all(adapter)
    del capture_all
    del adapter
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    predictor, result = fit_forced_distill_predictor(
        args.output_directory,
        train_schedule=train_schedule,
        validation_schedule=validation_schedule,
        initialization_seed_sha256=local.predictor_initialization_seed_sha256,
        source_commit=args.source_commit,
        launch_authority_sha256=local.authority_sha256,
    )
    predictor_bytes = canonical_predictor_state_bytes(predictor)
    if hashlib.sha256(predictor_bytes).hexdigest() != result.predictor_state_sha256:
        raise ValueError("ASG-CV forced distill predictor digest differs")
    raw = canonical_forced_distill_result_bytes(result)
    _write_atomic_bytes(predictor_path, predictor_bytes)
    _write_atomic_bytes(result_path, raw)
    sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
