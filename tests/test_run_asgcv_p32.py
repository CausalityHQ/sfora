from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import sys
import weakref
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sfora.asgcv_pilot import ASGCV_P32_BOUNDARY_NAMES, derive_asgcv_p32_schedule_seed
from sfora.asgcv_protocol import (
    AsgcvCompletionProtocol,
    AsgcvPartitionAuthority,
    AsgcvRolloutAuthority,
    build_asgcv_pair_schedule,
    derive_asgcv_rollout_seeds,
)

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_asgcv_p32.py"
_SPEC = importlib.util.spec_from_file_location("run_asgcv_p32_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def test_p32_cli_accepts_only_local_authority_and_refuses_network_or_test_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    files = {}
    for name in ("snapshot", "fixture", "p32", "train"):
        path = tmp_path / f"{name}.json"
        path.write_bytes(b"{}\n")
        files[name] = path
    argv = [
        "--model-root",
        str(model_root),
        "--snapshot-manifest",
        str(files["snapshot"]),
        "--fixture",
        str(files["fixture"]),
        "--p32-authority",
        str(files["p32"]),
        "--train-manifest",
        str(files["train"]),
        "--output-directory",
        str(output),
        "--source-commit",
        "1" * 40,
        "--execute-p32",
    ]
    parsed = _MODULE.parse_args(argv)
    assert parsed.model_root == model_root
    assert parsed.output_directory == output
    for forbidden in ("--dataset-split", "--test-manifest", "--url", "--aws-profile"):
        with pytest.raises(SystemExit):
            _MODULE.parse_args([*argv, forbidden, "forbidden"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*argv, "--fixture", str(files["fixture"])])
    with pytest.raises(SystemExit):
        _MODULE.parse_args([*argv, f"--source-commit={'2' * 40}"])
    monkeypatch.setattr(sys, "argv", [str(_SCRIPT), *argv, f"--source-commit={'2' * 40}"])
    with pytest.raises(SystemExit):
        _MODULE.parse_args()


def _write_json(path: Path, value: object) -> bytes:
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    return raw


def test_p32_local_authority_authenticates_train_only_arrays_and_rebuilds_schedule(
    tmp_path: Path,
) -> None:
    source_commit = "1" * 40
    predictor_rows = []
    for ordinal in range(64):
        image_path = tmp_path / f"image-{ordinal:03d}.npy"
        with image_path.open("wb") as stream:
            np.save(stream, np.full((2, 3, 3), ordinal, dtype=np.uint8), allow_pickle=False)
        payload = image_path.read_bytes()
        predictor_rows.append(
            {
                "array_path": image_path.name,
                "array_sha256": hashlib.sha256(payload).hexdigest(),
                "example_id": f"train-{ordinal:03d}",
                "label": ordinal // 4,
            }
        )
    train_manifest_path = tmp_path / "train.json"
    manifest_raw = _write_json(
        train_manifest_path,
        {
            "schema": "sfora-cars-train-p32-manifest-v1",
            "official_test_access": False,
            "predictor_train": predictor_rows,
            "e0_validation": [
                {"example_id": f"valid-{index:03d}", "label": 16 + index // 4}
                for index in range(16)
            ],
            "e1_optimization": [
                {"example_id": f"optim-{index:03d}", "label": 20 + index // 4}
                for index in range(16)
            ],
        },
    )
    partition = AsgcvPartitionAuthority(
        source_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(),
        partition_seed_sha256="2" * 64,
        predictor_train_class_ids=tuple(range(16)),
        e0_validation_class_ids=tuple(range(16, 20)),
        e1_optimization_class_ids=tuple(range(20, 24)),
    ).validated()
    predictor = (
        tuple(row["example_id"] for row in predictor_rows),
        tuple(row["label"] for row in predictor_rows),
    )
    schedule = build_asgcv_pair_schedule(
        *predictor,
        schedule_seed_sha256=derive_asgcv_p32_schedule_seed(
            partition_authority=partition,
            source_commit=source_commit,
        ),
        pair_count=32,
    )
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11,), different_prefix_ids=(21,), terminal_token_ids=(99,)
    ).validated()
    rollout = AsgcvRolloutAuthority(
        master_seed_sha256="3" * 64,
        model_revision="4" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128,
    ).validated()
    authority_path = tmp_path / "p32.json"
    _write_json(
        authority_path,
        {
            "schema": "sfora-asgcv-p32-launch-v1",
            "source_commit": source_commit,
            "prompt_utf8": "compare the two cars",
            "attribute_token_span": [1, 2],
            "patch_tokens_per_image": 1,
            "predictor_initialization_seed_sha256": "5" * 64,
            "partition_authority": partition.to_mapping(),
            "completion_protocol": protocol.to_mapping(),
            "rollout_authority": rollout.to_mapping(),
            "pilot_schedule": schedule.to_mapping(),
        },
    )

    loaded = _MODULE.load_p32_local_authority(
        authority_path,
        train_manifest_path,
        source_commit=source_commit,
    )
    assert loaded.partition_authority == partition
    assert loaded.pilot_schedule == schedule
    assert loaded.predictor_train == predictor
    assert len(loaded.images) == 64
    assert loaded.images[63].dtype == np.dtype(np.uint8)
    assert int(loaded.images[63][0, 0, 0]) == 63
    assert loaded.official_test_access is False

    drift = json.loads(train_manifest_path.read_bytes())
    drift["official_test_access"] = True
    _write_json(train_manifest_path, drift)
    with pytest.raises(ValueError):
        _MODULE.load_p32_local_authority(
            authority_path,
            train_manifest_path,
            source_commit=source_commit,
        )


def test_p32_main_wires_one_local_model_and_one_campaign(
    tmp_path: Path,
    monkeypatch,
    capsysbinary,
) -> None:
    model_root = tmp_path / "model"
    model_root.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    paths = {}
    for name in ("snapshot", "fixture", "p32", "train"):
        path = tmp_path / f"{name}.json"
        path.write_bytes(b"{}\n")
        paths[name] = path
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11,), different_prefix_ids=(21,), terminal_token_ids=(99,)
    ).validated()
    rollout = AsgcvRolloutAuthority(
        master_seed_sha256="3" * 64,
        model_revision="4" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128,
    ).validated()
    partition = AsgcvPartitionAuthority(
        source_manifest_sha256="5" * 64,
        partition_seed_sha256="6" * 64,
        predictor_train_class_ids=(0,),
        e0_validation_class_ids=(1,),
        e1_optimization_class_ids=(2,),
    ).validated()
    schedule = SimpleNamespace()
    local = SimpleNamespace(
        images=(np.zeros((2, 2, 3), dtype=np.uint8),),
        predictor_train=(("train",), (0,)),
        e0_validation=(("valid",), (1,)),
        e1_optimization=(("optim",), (2,)),
        partition_authority=partition,
        completion_protocol=protocol,
        rollout_authority=rollout,
        pilot_schedule=schedule,
        prompt_utf8="compare",
        attribute_token_span=(0, 1),
        patch_tokens_per_image=1,
        predictor_initialization_seed_sha256="7" * 64,
        authority_sha256="8" * 64,
    )
    fixture = SimpleNamespace(
        source_commit="1" * 40,
        model_revision=rollout.model_revision,
        attention_layer=26,
        prompt_utf8="compare",
        attribute_token_span=(0, 1),
        patch_tokens_per_image=1,
    )
    adapter = SimpleNamespace(
        pooler_sha256="9" * 64,
        pooler_token_dim=32,
        pooler=type("Pooler", (), {"parameters": lambda self: iter(())})(),
    )
    monkeypatch.setattr(_MODULE, "load_p32_local_authority", lambda *_args, **_kwargs: local)
    monkeypatch.setattr(_MODULE, "load_snapshot_authority", lambda **_kwargs: object())
    monkeypatch.setattr(_MODULE, "load_fixture_authority", lambda _path: fixture)
    monkeypatch.setattr(_MODULE, "load_qwen_adapter", lambda *_args, **_kwargs: adapter)
    monkeypatch.setattr(_MODULE, "TransformersFactory", lambda: object())
    monkeypatch.setattr(_MODULE, "_predictor_probe", lambda *_args, **_kwargs: lambda *_: None)
    calls = []

    def campaign(directory: Path, **kwargs: object) -> bytes:
        calls.append((directory, kwargs))
        assert callable(kwargs["execute_one"])
        return b'{"claim_eligible":false}\n'

    monkeypatch.setattr(_MODULE, "run_p32_campaign_with_failure_terminal", campaign)
    argv = [
        "--model-root",
        str(model_root),
        "--snapshot-manifest",
        str(paths["snapshot"]),
        "--fixture",
        str(paths["fixture"]),
        "--p32-authority",
        str(paths["p32"]),
        "--train-manifest",
        str(paths["train"]),
        "--output-directory",
        str(output),
        "--source-commit",
        "1" * 40,
        "--execute-p32",
    ]
    assert _MODULE.main(argv) == 0
    assert len(calls) == 1
    assert calls[0][0] == output
    assert calls[0][1]["partition_authority"] == partition
    assert capsysbinary.readouterr().out == b'{"claim_eligible":false}\n'

    monkeypatch.setattr(
        _MODULE,
        "run_p32_campaign_with_failure_terminal",
        lambda *_args, **_kwargs: b'{"schema":"sfora-asgcv-p32-failure-v1"}\n',
    )
    assert _MODULE.main(argv) == 3
    assert capsysbinary.readouterr().out == b'{"schema":"sfora-asgcv-p32-failure-v1"}\n'


@dataclass(frozen=True)
class _Target:
    patch_tokens: np.ndarray
    predicted_gradient: np.ndarray
    boundary_names: tuple[str, ...]
    boundary_predicted_gradient: np.ndarray
    branch_scores: tuple[float, float]
    coefficient: float
    generated_tokens: int = 0


@dataclass(frozen=True)
class _ExactTarget:
    patch_tokens: np.ndarray
    exact_gradient: np.ndarray
    replay: object


class _Adapter:
    def __init__(self, completions: tuple[tuple[int, ...], ...]) -> None:
        self.completions = completions
        self.branch_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
        self.last_target: weakref.ReferenceType[_Target] | None = None

    def prepare_image_pair(self, *args: object) -> object:
        return object()

    def generate(self, _pair: object, seed: int, **_kwargs: object) -> tuple[int, ...]:
        return self.completions[self.seeds.index(seed)]

    def score_completions(
        self, _pair: object, _completions: tuple[tuple[int, ...], ...]
    ) -> tuple[float, ...]:
        return (-0.10, -0.11, -0.09, -0.10, -1.10, -1.11, -1.09, -1.10)

    def collapsed_verdict_patch_gradient(
        self,
        _pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> _Target:
        self.branch_calls.append((correct_completion_ids, incorrect_completion_ids))
        index = len(self.branch_calls)
        if self.last_target is not None:
            gc.collect()
            assert self.last_target() is None
        field = np.full((2, 4, 4), float(index), dtype=np.float32)
        boundary = np.full((4, 2, 1, 4), float(index), dtype=np.float32)
        scores = (-0.10, -1.10) if index == 1 else (-0.10, -1.10)
        target = _Target(
            patch_tokens=np.ones_like(field),
            predicted_gradient=field,
            boundary_names=ASGCV_P32_BOUNDARY_NAMES,
            boundary_predicted_gradient=boundary,
            branch_scores=scores,
            coefficient=0.393256,
        )
        self.last_target = weakref.ref(target)
        return target


def test_p32_candidate_runner_binds_branch_order_fields_exact_replay_and_resources(
    monkeypatch,
) -> None:
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11, 12),
        different_prefix_ids=(21, 22),
        terminal_token_ids=(99,),
    ).validated()
    rollout = AsgcvRolloutAuthority(
        master_seed_sha256="8" * 64,
        model_revision="2" * 40,
        temperature=0.7,
        top_p=0.9,
        max_new_tokens=128,
    ).validated()
    example_ids = tuple(f"train-{index:03d}" for index in range(64))
    labels = tuple(index // 4 for index in range(64))
    schedule = build_asgcv_pair_schedule(
        example_ids,
        labels,
        schedule_seed_sha256="a" * 64,
        pair_count=32,
    )
    pair_row = schedule.pairs[0]
    completions = tuple(
        (
            *((11, 12) if pair_row.relation_sign == 1 else (21, 22)),
            30 + index,
            99,
        )
        if index < 4
        else (
            *((21, 22) if pair_row.relation_sign == 1 else (11, 12)),
            30 + index,
            99,
        )
        for index in range(8)
    )
    adapter = _Adapter(completions)
    adapter.seeds = derive_asgcv_rollout_seeds(rollout, candidate_pair_ordinal=0)
    reset_calls: list[str] = []
    monkeypatch.setattr(_MODULE, "_reset_candidate_peak", lambda: reset_calls.append("reset"))
    monkeypatch.setattr(_MODULE, "_peak_cuda_reserved_bytes", lambda: 1234)
    monkeypatch.setattr(_MODULE, "_process_peak_rss_bytes", lambda: 5678)

    exact_field = np.full((2, 4, 4), 3.0, dtype=np.float32)
    exact = _ExactTarget(
        patch_tokens=np.ones_like(exact_field),
        exact_gradient=exact_field,
        replay=type("Replay", (), {"generated_tokens": 32})(),
    )
    predictor_calls: list[int] = []

    def exact_capture(*_args: object, **_kwargs: object) -> _ExactTarget:
        gc.collect()
        assert adapter.last_target is not None and adapter.last_target() is None
        return exact

    group, candidate = _MODULE.run_p32_candidate(
        adapter,
        ordinal=0,
        images=(np.zeros((2, 2, 3), dtype=np.uint8),) * 64,
        prompt_utf8="compare",
        attribute_token_span=(0, 1),
        patch_tokens_per_image=1,
        completion_protocol=protocol,
        rollout_authority=rollout,
        pilot_schedule=schedule,
        source_commit="1" * 40,
        fixture_sha256="3" * 64,
        launch_authority_sha256="4" * 64,
        predictor_initialization_seed_sha256="5" * 64,
        partition_authority_sha256="4" * 64,
        pooler_state_sha256="6" * 64,
        exact_diagnostic_available=True,
        exact_capture=exact_capture,
        predictor_probe=lambda _tokens, sign: predictor_calls.append(sign),
    )

    assert reset_calls == ["reset"]
    assert adapter.branch_calls == [
        (completions[0], completions[4]),
        (completions[3], completions[7]),
    ]
    assert predictor_calls == [pair_row.relation_sign]
    assert candidate.collapsed_branch_scores == (-0.10, -1.10)
    assert candidate.collapsed_backend_coefficient_ppm == 393_256
    assert candidate.highest_branch_scores == (-0.10, -1.10)
    assert candidate.highest_backend_coefficient_ppm == 393_256
    assert candidate.boundary_names == ASGCV_P32_BOUNDARY_NAMES
    assert candidate.branch_exchange_energy_ppm == 200_000
    assert candidate.exact_diagnostic is True
    assert candidate.peak_cuda_reserved_bytes == 1234
    assert candidate.peak_rss_bytes == 5678
    assert group.completion_ids == completions

    duplicate_branches = (
        completions[0],
        completions[1],
        completions[2],
        completions[0],
        completions[4],
        completions[5],
        completions[6],
        completions[4],
    )
    duplicate_adapter = _Adapter(duplicate_branches)
    duplicate_adapter.seeds = adapter.seeds
    _, duplicate_candidate = _MODULE.run_p32_candidate(
        duplicate_adapter,
        ordinal=0,
        images=(np.zeros((2, 2, 3), dtype=np.uint8),) * 64,
        prompt_utf8="compare",
        attribute_token_span=(0, 1),
        patch_tokens_per_image=1,
        completion_protocol=protocol,
        rollout_authority=rollout,
        pilot_schedule=schedule,
        source_commit="1" * 40,
        fixture_sha256="3" * 64,
        launch_authority_sha256="4" * 64,
        predictor_initialization_seed_sha256="5" * 64,
        partition_authority_sha256="4" * 64,
        pooler_state_sha256="6" * 64,
        exact_diagnostic_available=False,
        exact_capture=lambda *_args, **_kwargs: pytest.fail("exact capture must remain disabled"),
        predictor_probe=lambda *_args: None,
    )
    assert duplicate_candidate.branch_exchange_distinct is False
    assert duplicate_candidate.exchange_evaluable is False
    assert duplicate_candidate.highest_branch_indices is None
    assert len(duplicate_adapter.branch_calls) == 1

    all_correct = tuple(
        (
            *((11, 12) if pair_row.relation_sign == 1 else (21, 22)),
            40 + index,
            99,
        )
        for index in range(8)
    )
    degenerate = _Adapter(all_correct)
    degenerate.seeds = adapter.seeds
    degenerate_group, degenerate_candidate = _MODULE.run_p32_candidate(
        degenerate,
        ordinal=0,
        images=(np.zeros((2, 2, 3), dtype=np.uint8),) * 64,
        prompt_utf8="compare",
        attribute_token_span=(0, 1),
        patch_tokens_per_image=1,
        completion_protocol=protocol,
        rollout_authority=rollout,
        pilot_schedule=schedule,
        source_commit="1" * 40,
        fixture_sha256="3" * 64,
        launch_authority_sha256="4" * 64,
        predictor_initialization_seed_sha256="5" * 64,
        partition_authority_sha256="4" * 64,
        pooler_state_sha256="6" * 64,
        exact_diagnostic_available=True,
        exact_capture=lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("zero-variance group must not consume an exact slot")
        ),
        predictor_probe=lambda *_args: (_ for _ in ()).throw(
            AssertionError("branch-ineligible group must not time the predictor")
        ),
    )
    assert degenerate_group.nonzero_reward_variance is False
    assert degenerate_candidate.exact_diagnostic is False
