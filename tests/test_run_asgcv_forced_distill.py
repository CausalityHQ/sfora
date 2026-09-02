from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import torch

from sfora.asgcv import AsgcvSrhtAuthority
from sfora.asgcv_forced_distill import (
    ASGCV_FORCED_DISTILL_SHAPE,
    build_forced_distill_schedule,
    validate_forced_distill_capture_bytes,
)
from sfora.asgcv_predictor import predictor_state_sha256, source_bound_predictor
from sfora.asgcv_protocol import AsgcvCompletionProtocol

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_asgcv_forced_distill.py"
_SPEC = importlib.util.spec_from_file_location("run_asgcv_forced_distill_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
capture_forced_distill_pair = _MODULE.capture_forced_distill_pair
parse_args = _MODULE.parse_args
run_capture_phase = _MODULE.run_capture_phase
train_predictor_epoch = _MODULE.train_predictor_epoch


SOURCE_COMMIT = "12" * 20
LAUNCH_SHA256 = "34" * 32


@dataclass
class _Target:
    patch_tokens: np.ndarray
    predicted_gradient: np.ndarray
    branch_scores: tuple[float, float] = (-0.25, -3.0)
    boundary_names: tuple[str, ...] = ("merger", "deepstack-0", "deepstack-1", "deepstack-2")


class _Adapter:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def prepare_image_pair(self, images: object, *args: object) -> object:
        return images

    def collapsed_verdict_patch_gradient(
        self,
        pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> _Target:
        self.calls.append((correct_completion_ids, incorrect_completion_ids))
        return _Target(
            patch_tokens=np.ones(ASGCV_FORCED_DISTILL_SHAPE, dtype=np.float32),
            predicted_gradient=np.full(ASGCV_FORCED_DISTILL_SHAPE, 2.0, dtype=np.float32),
        )


def _schedule():
    ids = tuple(f"c{label}/i{image}" for label in range(64) for image in range(4))
    labels = tuple(label for label in range(64) for _ in range(4))
    return build_forced_distill_schedule(
        ids,
        labels,
        source_commit=SOURCE_COMMIT,
        launch_authority_sha256=LAUNCH_SHA256,
        role="train",
    )


def test_capture_forced_pair_uses_prefix_only_and_orients_negative_target() -> None:
    adapter = _Adapter()
    schedule = _schedule()
    negative = next(pair for pair in schedule.pairs if pair.relation_sign == -1)
    images = tuple(np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(256))
    protocol = AsgcvCompletionProtocol((11,), (21, 22, 23), (99,)).validated()
    receipt, patches, gradient = capture_forced_distill_pair(
        adapter,
        role="train",
        pair=negative,
        schedule=schedule,
        images=images,
        prompt_utf8="prompt",
        attribute_token_span=(2, 3),
        patch_tokens_per_image=49,
        completion_protocol=protocol,
        source_commit=SOURCE_COMMIT,
        launch_authority_sha256=LAUNCH_SHA256,
    )
    assert adapter.calls == [((11,), (21, 22, 23))]
    assert np.array_equal(patches, np.ones_like(patches))
    assert np.array_equal(gradient, np.full_like(gradient, -2.0))
    validate_forced_distill_capture_bytes(
        receipt,
        patch_tokens=patches,
        exact_gradient=gradient,
    )


def test_capture_phase_writes_atomic_triple_and_resumes(tmp_path: Path) -> None:
    schedule = _schedule()
    patches = np.ones(ASGCV_FORCED_DISTILL_SHAPE, dtype=np.float32)
    gradient = np.full(ASGCV_FORCED_DISTILL_SHAPE, 2.0, dtype=np.float32)
    calls: list[int] = []

    def execute(pair_ordinal: int):
        calls.append(pair_ordinal)
        pair = schedule.pairs[pair_ordinal]
        adapter = _Adapter()
        return capture_forced_distill_pair(
            adapter,
            role="train",
            pair=pair,
            schedule=schedule,
            images=tuple(np.zeros((1, 1, 3), dtype=np.uint8) for _ in range(256)),
            prompt_utf8="prompt",
            attribute_token_span=(2, 3),
            patch_tokens_per_image=49,
            completion_protocol=AsgcvCompletionProtocol((11,), (21, 22), (99,)).validated(),
            source_commit=SOURCE_COMMIT,
            launch_authority_sha256=LAUNCH_SHA256,
        )

    assert (
        run_capture_phase(
            tmp_path,
            role="train",
            schedule=schedule,
            execute_one=execute,
            maximum_new_rows=1,
        )
        == 1
    )
    assert calls == [0]
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "train-000000-gradient.npy",
        "train-000000-patch.npy",
        "train-000000.json",
    ]
    assert not tuple(tmp_path.glob("*.partial"))
    assert (
        run_capture_phase(
            tmp_path,
            role="train",
            schedule=schedule,
            execute_one=execute,
            maximum_new_rows=1,
        )
        == 2
    )
    assert calls == [0, 1]
    with (tmp_path / "train-000000-patch.npy").open("rb") as stream:
        assert np.array_equal(np.load(stream, allow_pickle=False), patches)
    with (tmp_path / "train-000000-gradient.npy").open("rb") as stream:
        observed = np.load(stream, allow_pickle=False)
    expected_sign = schedule.pairs[0].relation_sign
    assert np.array_equal(observed, gradient * expected_sign)


def test_forced_distill_cli_rejects_network_and_official_test_flags(tmp_path: Path) -> None:
    required = [
        "--model-root",
        str(tmp_path),
        "--snapshot-manifest",
        __file__,
        "--fixture",
        __file__,
        "--p32-authority",
        __file__,
        "--train-manifest",
        __file__,
        "--output-directory",
        str(tmp_path),
        "--source-commit",
        SOURCE_COMMIT,
        "--execute-forced-distill",
    ]
    for forbidden in ("--official-test", "--model-uri", "--aws-profile"):
        with pytest.raises(SystemExit):
            parse_args([*required, forbidden, "x"])


def test_train_predictor_epoch_updates_student_without_mutating_exact_target() -> None:
    predictor = source_bound_predictor(channel_dimensions=16, seed_sha256="ab" * 32)
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=1e-3, weight_decay=1e-4)
    tokens = torch.arange(64, dtype=torch.float32).reshape(1, 2, 2, 16) / 64.0
    signs = torch.tensor([1], dtype=torch.int8)
    exact = torch.flip(tokens, dims=(-1,)).detach()
    preserved = exact.clone()
    before = predictor_state_sha256(predictor)
    loss = train_predictor_epoch(
        predictor,
        optimizer,
        ((tokens, signs, exact),),
        srht_authority=AsgcvSrhtAuthority(
            input_dimensions=16,
            padded_dimensions=16,
            output_dimensions=8,
            seed_sha256="cd" * 32,
        ).validated(),
    )
    assert np.isfinite(loss) and loss >= 0.0
    assert predictor_state_sha256(predictor) != before
    assert torch.equal(exact, preserved)
