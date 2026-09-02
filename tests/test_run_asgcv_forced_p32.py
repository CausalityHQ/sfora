from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sfora.asgcv_protocol import AsgcvCompletionProtocol, build_asgcv_pair_schedule

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_asgcv_forced_p32.py"
_SPEC = importlib.util.spec_from_file_location("run_asgcv_forced_p32_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@dataclass
class _Target:
    predicted_gradient: np.ndarray
    boundary_predicted_gradient: np.ndarray
    boundary_names: tuple[str, ...]
    branch_scores: tuple[float, float]


class _Adapter:
    def __init__(self) -> None:
        self.branch_calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def prepare_image_pair(self, *_args: object) -> object:
        return object()

    def collapsed_verdict_patch_gradient(
        self,
        _pair: object,
        *,
        correct_completion_ids: tuple[int, ...],
        incorrect_completion_ids: tuple[int, ...],
    ) -> _Target:
        self.branch_calls.append((correct_completion_ids, incorrect_completion_ids))
        return _Target(
            predicted_gradient=np.ones((2, 4, 8), dtype=np.float32),
            boundary_predicted_gradient=np.ones((4, 2, 1, 8), dtype=np.float32),
            boundary_names=("merger", "deepstack-0", "deepstack-1", "deepstack-2"),
            branch_scores=(-0.1, -1.1),
        )


def test_forced_candidate_uses_fixed_same_then_different_without_sampling(monkeypatch) -> None:
    ids = tuple(f"train-{index:03d}" for index in range(64))
    labels = tuple(index // 4 for index in range(64))
    schedule = build_asgcv_pair_schedule(
        ids,
        labels,
        schedule_seed_sha256="a" * 64,
        pair_count=32,
    )
    protocol = AsgcvCompletionProtocol(
        same_prefix_ids=(11, 12),
        different_prefix_ids=(21, 22),
        terminal_token_ids=(99,),
    ).validated()
    adapter = _Adapter()
    monkeypatch.setattr(_MODULE, "_peak_cuda_reserved_bytes", lambda: 123)
    monkeypatch.setattr(_MODULE, "_process_peak_rss_bytes", lambda: 456)

    observation = _MODULE.run_forced_candidate(
        adapter,
        ordinal=0,
        images=(np.zeros((2, 2, 3), dtype=np.uint8),) * 64,
        prompt_utf8="compare",
        attribute_token_span=(0, 1),
        patch_tokens_per_image=1,
        completion_protocol=protocol,
        pilot_schedule=schedule,
        source_commit="1" * 40,
        launch_authority_sha256="2" * 64,
        model_revision="3" * 40,
        fixture_sha256="4" * 64,
    )

    assert adapter.branch_calls == [((11, 12), (21, 22))]
    assert observation.candidate_pair_ordinal == 0
    assert observation.same_score == -0.1
    assert observation.different_score == -1.1
    assert observation.gradient_norm == 8.0
    assert observation.boundary_norms == (4.0, 4.0, 4.0, 4.0)
    assert observation.peak_cuda_reserved_bytes == 123
    assert observation.peak_rss_bytes == 456


def test_forced_campaign_writes_resumable_candidates_and_repeats_edges(tmp_path) -> None:
    ids = tuple(f"train-{index:03d}" for index in range(64))
    labels = tuple(index // 4 for index in range(64))
    schedule = build_asgcv_pair_schedule(
        ids,
        labels,
        schedule_seed_sha256="a" * 64,
        pair_count=32,
    )
    observations = tuple(
        _MODULE.AsgcvForcedObservation(
            source_commit="1" * 40,
            launch_authority_sha256="2" * 64,
            pilot_schedule_sha256=schedule.sha256(),
            model_revision="4" * 40,
            fixture_sha256="5" * 64,
            candidate_pair_ordinal=ordinal,
            pair_ordinals=(schedule.pairs[ordinal].left_index, schedule.pairs[ordinal].right_index),
            relation_sign=schedule.pairs[ordinal].relation_sign,
            same_score=0.0 if schedule.pairs[ordinal].relation_sign == 1 else -1.0,
            different_score=-1.0 if schedule.pairs[ordinal].relation_sign == 1 else 0.0,
            gradient_sha256=f"{ordinal + 1:064x}",
            gradient_norm=1.0,
            boundary_norms=(1.0, 1.0, 1.0, 1.0),
            prepare_elapsed_ns=1,
            replay_elapsed_ns=1,
            peak_cuda_reserved_bytes=1,
            peak_rss_bytes=1,
        ).validated()
        for ordinal in range(32)
    )
    calls: list[int] = []

    def execute(ordinal: int) -> object:
        calls.append(ordinal)
        return observations[ordinal]

    raw = _MODULE.run_forced_campaign(
        tmp_path,
        pilot_schedule=schedule,
        source_commit="1" * 40,
        launch_authority_sha256="2" * 64,
        model_revision="4" * 40,
        fixture_sha256="5" * 64,
        execute_one=execute,
    )
    assert calls == [*range(32), 0, 31]
    assert json.loads(raw)["passed"] is True
    assert len(tuple(tmp_path.glob("candidate-*.json"))) == 32
    assert (tmp_path / "result.json").read_bytes() == raw

    calls.clear()
    assert (
        _MODULE.run_forced_campaign(
            tmp_path,
            pilot_schedule=schedule,
            source_commit="1" * 40,
            launch_authority_sha256="2" * 64,
            model_revision="4" * 40,
            fixture_sha256="5" * 64,
            execute_one=execute,
        )
        == raw
    )
    assert calls == []


def test_forced_cli_is_local_train_only_and_explicit(tmp_path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    output = tmp_path / "output"
    output.mkdir()
    files = []
    for name in ("snapshot.json", "fixture.json", "authority.json", "train.json"):
        path = tmp_path / name
        path.write_text("{}\n")
        files.append(path)
    args = _MODULE.parse_args(
        [
            "--model-root",
            str(model),
            "--snapshot-manifest",
            str(files[0]),
            "--fixture",
            str(files[1]),
            "--p32-authority",
            str(files[2]),
            "--train-manifest",
            str(files[3]),
            "--output-directory",
            str(output),
            "--source-commit",
            "1" * 40,
            "--execute-forced-p32",
        ]
    )
    assert args.execute_forced_p32 is True
    with pytest.raises(SystemExit):
        _MODULE.parse_args(
            [
                "--model-root",
                str(model),
                "--official-test",
                str(files[0]),
                "--execute-forced-p32",
            ]
        )


def test_forced_main_authenticates_local_inputs_and_runs_one_campaign(
    monkeypatch, tmp_path, capsysbinary
) -> None:
    ids = tuple(f"train-{index:03d}" for index in range(64))
    labels = tuple(index // 4 for index in range(64))
    schedule = build_asgcv_pair_schedule(
        ids,
        labels,
        schedule_seed_sha256="a" * 64,
        pair_count=32,
    )
    protocol = AsgcvCompletionProtocol((11,), (21,), (99,)).validated()
    fixture_path = tmp_path / "fixture.json"
    fixture_path.write_bytes(b"{}\n")
    args = Namespace(
        model_root=tmp_path,
        snapshot_manifest=tmp_path / "snapshot.json",
        fixture=fixture_path,
        p32_authority=tmp_path / "authority.json",
        train_manifest=tmp_path / "train.json",
        output_directory=tmp_path,
        source_commit="1" * 40,
        execute_forced_p32=True,
    )
    local = SimpleNamespace(
        images=(np.zeros((2, 2, 3), dtype=np.uint8),) * 64,
        prompt_utf8="compare",
        attribute_token_span=(0, 1),
        patch_tokens_per_image=1,
        completion_protocol=protocol,
        pilot_schedule=schedule,
        rollout_authority=SimpleNamespace(model_revision="4" * 40),
        authority_sha256="2" * 64,
    )
    fixture = SimpleNamespace(
        source_commit="1" * 40,
        model_revision="4" * 40,
        prompt_utf8="compare",
        patch_tokens_per_image=1,
    )
    adapter = _Adapter()
    monkeypatch.setattr(_MODULE, "parse_args", lambda _argv=None: args)
    monkeypatch.setattr(_MODULE, "_authenticated_source_commit", lambda _root: "1" * 40)
    monkeypatch.setattr(_MODULE, "load_p32_local_authority", lambda *_args, **_kwargs: local)
    monkeypatch.setattr(_MODULE, "load_snapshot_authority", lambda **_kwargs: object())
    monkeypatch.setattr(_MODULE, "load_fixture_authority", lambda _path: fixture)
    monkeypatch.setattr(_MODULE, "load_qwen_adapter", lambda *_args, **_kwargs: adapter)

    def campaign(_directory: Path, **kwargs: object) -> bytes:
        execute = kwargs["execute_one"]
        assert callable(execute)
        observation = execute(0)
        assert observation.candidate_pair_ordinal == 0
        return b'{"passed":true}\n'

    monkeypatch.setattr(_MODULE, "run_forced_campaign", campaign)
    assert _MODULE.main([]) == 0
    assert capsysbinary.readouterr().out == b'{"passed":true}\n'
