from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest


def _load_subject() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "run_sop_teacher_anchored_panel.py"
    spec = importlib.util.spec_from_file_location("run_sop_teacher_anchored_panel", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_trainer() -> ModuleType:
    path = Path(__file__).parents[1] / "scripts" / "train_sop_teacher_anchored_distillation.py"
    spec = importlib.util.spec_from_file_location("train_sop_teacher_anchored_distillation", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SUBJECT = _load_subject()


def _completed_arm_result(launch: object, *, stopped_reason: str | None = None) -> bytes:
    checkpoint = launch.output.with_suffix(".pt")
    checkpoint.write_bytes(b"fixture-checkpoint")
    checkpoint_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    launch_value = json.loads(launch.receipt.read_bytes())
    authority = {
        "anchor_schedule_sha256": "1" * 64,
        "arm": launch.arm,
        "batch_schedule_sha256": "2" * 64,
        "fitting_probe_sha256": "3" * 64,
        "head_replay_sha256": "4" * 64,
        "inputs_sha256": {**launch_value["inputs_sha256"], "launch_receipt": launch.receipt_sha256},
        "model_mode_sha256": "5" * 64,
        "objective_sha256": "6" * 64,
        "ridge_sha256": "7" * 64,
        "runtime_sha256": "8" * 64,
        "seed": launch_value["seed"],
        "snapshot_replay_sha256": "9" * 64,
        "source_revision": launch_value["source_revision"],
        "split_sha256": "a" * 64,
        "teacher_pca_sha256": "b" * 64,
        "trainable_inventory_sha256": "c" * 64,
    }
    complete = stopped_reason is None
    value = {
        "arm": launch.arm,
        "attempted_updates": 10,
        "authority": authority,
        "candidate_epoch": 10 if complete else None,
        "checkpoint_path": checkpoint.name,
        "checkpoint_sha256": checkpoint_sha,
        "claim_eligible": False,
        "completed_epochs": list(range(1, 11)) if complete else [1],
        "diagnostics": [
            {
                "epoch": epoch,
                "fitting_effective_rank": 32.0,
                "fitting_leading_eigenvalue_share": 0.1,
                "fitting_map_at_r": 0.5,
                "validation_map_at_r": 0.5,
            }
            for epoch in range(11)
        ],
        "final_encoder_sha256": "d" * 64,
        "final_frozen_sha256": "e" * 64,
        "final_head_sha256": "f" * 64,
        "initial_encoder_sha256": "1" * 64,
        "initial_frozen_sha256": "2" * 64,
        "initial_head_sha256": "3" * 64,
        "optimizer_reset_epochs": [1, 2],
        "schedule_sha256": "2" * 64,
        "schema": "sfora-teacher-anchored-arm-v1",
        "stopped_reason": stopped_reason,
        "successful_updates": 10 if complete else 9,
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _panel_arguments(tmp_path: Path) -> list[str]:
    paths: dict[str, Path] = {}
    for name in (
        "source-checkpoint",
        "teacher-checkpoint",
        "source-snapshot",
        "teacher-snapshot",
        "schedule",
        "ceiling-receipt",
    ):
        path = tmp_path / name
        path.write_bytes(name.encode())
        paths[name] = path
    checkout = tmp_path / "unicom"
    checkout.mkdir(exist_ok=True)
    image_root = tmp_path / "images"
    image_root.mkdir(exist_ok=True)
    output = tmp_path / "panel"
    output.mkdir(exist_ok=True)
    values = {
        "source-checkpoint-sha256": "1" * 64,
        "teacher-checkpoint-sha256": "2" * 64,
        "source-snapshot-sha256": "3" * 64,
        "teacher-snapshot-sha256": "4" * 64,
        "schedule-sha256": "5" * 64,
        "image-tree-sha256": "6" * 64,
        "teacher-pca-sha256": "7" * 64,
        "ceiling-receipt-sha256": "8" * 64,
        "source-revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
        "seed": "17",
    }
    arguments: list[str] = []
    for name, path in paths.items():
        arguments.extend((f"--{name}", str(path.resolve())))
    arguments.extend(("--unicom-checkout", str(checkout.resolve())))
    arguments.extend(("--image-root", str(image_root.resolve())))
    arguments.extend(("--output-directory", str(output.resolve())))
    for name, value in values.items():
        arguments.extend((f"--{name}", value))
    arguments.append("--execute-teacher-anchored-panel")
    return arguments


def _sample(
    *,
    rss: int = 1024,
    psi: float = 0.0,
    swap_delta: int = 0,
    progress_age: float = 1.0,
    wall: float = 1.0,
) -> object:
    return SUBJECT.TeacherAnchoredPressureSample(
        rss_bytes=rss,
        psi_full_avg10=psi,
        swap_delta_bytes=swap_delta,
        progress_age_seconds=progress_age,
        wall_seconds=wall,
    )


@pytest.mark.parametrize(
    ("samples", "expected"),
    [
        ((_sample(rss=32 * 1024**3 + 1),), "rss-cap"),
        ((_sample(psi=0.79),), "psi-immediate"),
        ((_sample(psi=0.5), _sample(psi=0.51), _sample(psi=0.5)), "psi-sustained"),
        ((_sample(swap_delta=2 * 1024**3 + 1),), "swap-delta"),
        ((_sample(progress_age=900.0),), "progress-timeout"),
        ((_sample(wall=float(18 * 3600)),), "wall-timeout"),
        ((_sample(psi=0.5), _sample(psi=0.0), _sample(psi=0.5)), None),
    ],
)
def test_stop_classification_is_exact_and_sustained(
    samples: tuple[object, ...], expected: str | None
) -> None:
    assert SUBJECT.classify_teacher_anchored_stop(samples) == expected


def test_canonical_stop_receipt_is_outcome_blind_and_hash_stable() -> None:
    launch = "a" * 64
    progress = "b" * 64
    samples = (_sample(psi=0.8),)
    raw = SUBJECT.canonical_teacher_anchored_stop_bytes(
        launch_receipt_sha256=launch,
        last_progress_sha256=progress,
        last_arm="complete",
        last_epoch=4,
        last_update=71,
        samples=samples,
        reason="psi-immediate",
    )
    value = json.loads(raw)
    assert raw == json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert value["claim_eligible"] is False
    assert value["reason"] == "psi-immediate"
    assert "map" not in raw.decode().lower()
    assert len(hashlib.sha256(raw).hexdigest()) == 64
    with pytest.raises(ValueError, match="stop receipt"):
        SUBJECT.canonical_teacher_anchored_stop_bytes(
            launch_receipt_sha256=launch,
            last_progress_sha256=progress,
            last_arm="complete",
            last_epoch=4,
            last_update=71,
            samples=samples,
            reason="quality-failed",
        )


def test_progress_cursor_advances_only_on_strict_authenticated_extension() -> None:
    launch = "a" * 64
    first = (
        json.dumps(
            {
                "arm": "complete",
                "epoch": 0,
                "launch_receipt_sha256": launch,
                "previous_line_sha256": "0" * 64,
                "schema": "sfora-teacher-anchored-progress-v1",
                "sequence": 1,
                "stage": "initialized",
                "update": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    second = (
        json.dumps(
            {
                "arm": "complete",
                "epoch": 1,
                "launch_receipt_sha256": launch,
                "previous_line_sha256": hashlib.sha256(first).hexdigest(),
                "schema": "sfora-teacher-anchored-progress-v1",
                "sequence": 2,
                "stage": "update",
                "update": 1,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    state = SUBJECT.advance_teacher_anchored_progress(None, (first,), launch)
    assert state.sequence == 1
    state = SUBJECT.advance_teacher_anchored_progress(state, (first, second), launch)
    assert state.sequence == 2
    with pytest.raises(ValueError, match="progress replay"):
        SUBJECT.advance_teacher_anchored_progress(state, (first, second), launch)


def test_progress_validator_is_loaded_once_per_launcher_process() -> None:
    first = SUBJECT._load_progress_validator()
    second = SUBJECT._load_progress_validator()

    assert first is second


def test_live_progress_tolerates_partial_tail_but_terminal_requires_framing(
    tmp_path: Path,
) -> None:
    launch = "a" * 64
    first = (
        json.dumps(
            {
                "arm": "complete",
                "epoch": 0,
                "launch_receipt_sha256": launch,
                "previous_line_sha256": "0" * 64,
                "schema": "sfora-teacher-anchored-progress-v1",
                "sequence": 1,
                "stage": "initialized",
                "update": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    path = tmp_path / "progress.jsonl"
    path.write_bytes(first + b'{"arm":"complete"')

    state = SUBJECT.replay_teacher_anchored_progress_file(path, None, launch, terminal=False)

    assert state is not None and state.sequence == 1
    with pytest.raises(ValueError, match="progress replay"):
        SUBJECT.replay_teacher_anchored_progress_file(path, state, launch, terminal=True)


def test_terminal_progress_must_preserve_last_authenticated_prefix(tmp_path: Path) -> None:
    launch = "a" * 64

    def line(arm: str) -> bytes:
        return (
            json.dumps(
                {
                    "arm": arm,
                    "epoch": 0,
                    "launch_receipt_sha256": launch,
                    "previous_line_sha256": "0" * 64,
                    "schema": "sfora-teacher-anchored-progress-v1",
                    "sequence": 1,
                    "stage": "initialized",
                    "update": 0,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            + b"\n"
        )

    path = tmp_path / "progress.jsonl"
    path.write_bytes(line("complete"))
    state = SUBJECT.replay_teacher_anchored_progress_file(path, None, launch, terminal=False)
    path.write_bytes(line("base"))

    with pytest.raises(ValueError, match="progress replay"):
        SUBJECT.replay_teacher_anchored_progress_file(path, state, launch, terminal=True)


def test_panel_cli_is_local_only_and_binds_all_five_arms(tmp_path: Path) -> None:
    parsed = SUBJECT.parse_teacher_anchored_panel_args(_panel_arguments(tmp_path))

    assert parsed.output_directory == (tmp_path / "panel").resolve()
    assert parsed.seed == 17
    assert SUBJECT.TEACHER_ANCHORED_ARMS == (
        "head-only",
        "base",
        "anchor",
        "symmetric",
        "complete",
    )
    for forbidden in ("--bucket", "--s3-uri", "--official-test", "--arm"):
        with pytest.raises(ValueError, match="unsupported argument"):
            SUBJECT.parse_teacher_anchored_panel_args(
                [*_panel_arguments(tmp_path), forbidden, "forbidden"]
            )

    checkout = tmp_path / "unicom"
    checkout.rmdir()
    checkout.symlink_to(tmp_path / "images", target_is_directory=True)
    symlink_arguments = _panel_arguments(tmp_path)
    symlink_arguments[symlink_arguments.index("--unicom-checkout") + 1] = str(checkout)
    with pytest.raises(ValueError, match="input authority"):
        SUBJECT.parse_teacher_anchored_panel_args(symlink_arguments)


def test_child_environment_drops_import_and_network_override_capabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("PYTHONPATH", "/tmp/shadow")
    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid")
    monkeypatch.setenv("AWS_PROFILE", "forbidden")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")

    environment = SUBJECT.teacher_anchored_child_environment()

    assert "PYTHONPATH" not in environment
    assert "HTTP_PROXY" not in environment
    assert "AWS_PROFILE" not in environment
    assert environment["CUDA_VISIBLE_DEVICES"] == "2"
    assert environment["PYTHONNOUSERSITE"] == "1"


def test_launch_receipt_is_exact_canonical_and_arm_specific(tmp_path: Path) -> None:
    parsed = SUBJECT.parse_teacher_anchored_panel_args(_panel_arguments(tmp_path))
    output = (tmp_path / "panel" / "complete.result.json").resolve()

    raw = SUBJECT.canonical_teacher_anchored_launch_bytes(parsed, "complete", output)

    expected = {
        "arm": "complete",
        "claim_eligible": False,
        "inputs_sha256": {
            "ceiling_receipt": "8" * 64,
            "image_tree": "6" * 64,
            "schedule": "5" * 64,
            "source_checkpoint": "1" * 64,
            "source_snapshot": "3" * 64,
            "teacher_checkpoint": "2" * 64,
            "teacher_snapshot": "4" * 64,
        },
        "output": str(output),
        "schema": "sfora-teacher-anchored-launch-v1",
        "seed": 17,
        "source_revision": "d71992ed969e6c271436ac0a0ee1f3ca61474ac0",
    }
    assert raw == json.dumps(expected, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert SUBJECT.canonical_teacher_anchored_launch_bytes(parsed, "base", output) != raw
    with pytest.raises(ValueError, match="launch receipt"):
        SUBJECT.canonical_teacher_anchored_launch_bytes(parsed, "unknown", output)


def test_prepared_arm_is_accepted_by_real_trainer_parser_and_cannot_overwrite(
    tmp_path: Path,
) -> None:
    parsed = SUBJECT.parse_teacher_anchored_panel_args(_panel_arguments(tmp_path))
    launch = SUBJECT.prepare_teacher_anchored_arm(parsed, "anchor")

    assert launch.receipt == (tmp_path / "panel" / "anchor.launch.json").resolve()
    assert launch.output == (tmp_path / "panel" / "anchor.result.json").resolve()
    assert launch.progress == (tmp_path / "panel" / "anchor.result.progress.jsonl").resolve()
    assert hashlib.sha256(launch.receipt.read_bytes()).hexdigest() == launch.receipt_sha256
    trainer = _load_trainer()
    consumed = trainer.parse_teacher_anchored_args(list(launch.command[2:]))
    assert consumed.arm == "anchor"
    assert consumed.launch_receipt == launch.receipt
    assert consumed.launch_receipt_sha256 == launch.receipt_sha256
    assert consumed.output == launch.output
    assert trainer.authenticate_teacher_anchored_launch_receipt(consumed) == launch.receipt_sha256
    with pytest.raises(ValueError, match="already exists"):
        SUBJECT.prepare_teacher_anchored_arm(parsed, "anchor")


def test_process_group_runner_accepts_only_authenticated_progress_and_matching_output(
    tmp_path: Path,
) -> None:
    launch_sha = "a" * 64
    output = (tmp_path / "complete.result.json").resolve()
    progress = output.with_suffix(".progress.jsonl")
    receipt = (tmp_path / "complete.launch.json").resolve()
    receipt.write_bytes(b"{}\n")
    child = tmp_path / "child.py"
    child.write_text(
        """\
import hashlib
import json
import pathlib
import sys

progress = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
launch = sys.argv[3]
line = json.dumps({
    "arm": "complete",
    "epoch": 0,
    "launch_receipt_sha256": launch,
    "previous_line_sha256": "0" * 64,
    "schema": "sfora-teacher-anchored-progress-v1",
    "sequence": 1,
    "stage": "initialized",
    "update": 0,
}, sort_keys=True, separators=(",", ":")).encode() + b"\\n"
progress.write_bytes(line)
payload = b'{"claim_eligible":false,"schema":"fixture-result-v1"}\\n'
output.write_bytes(payload)
sys.stdout.buffer.write(payload)
""",
        encoding="utf-8",
    )
    launch = SUBJECT.TeacherAnchoredArmLaunch(
        arm="complete",
        receipt=receipt,
        receipt_sha256=launch_sha,
        output=output,
        progress=progress,
        command=(sys.executable, str(child), str(progress), str(output), launch_sha),
    )

    terminal = SUBJECT.run_teacher_anchored_arm_process(launch, poll_seconds=0.01)

    assert terminal.returncode == 0
    assert terminal.result_bytes == output.read_bytes()
    assert terminal.progress.sequence == 1
    assert terminal.completed is False
    assert terminal.stop_receipt is None
    assert not output.with_suffix(".stdout.partial").exists()
    assert output.with_suffix(".stdout.json").read_bytes() == terminal.result_bytes


def test_process_group_runner_terminates_on_registered_pressure_and_emits_stop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_sha = "a" * 64
    output = (tmp_path / "complete.result.json").resolve()
    progress = output.with_suffix(".progress.jsonl")
    line = (
        json.dumps(
            {
                "arm": "complete",
                "epoch": 0,
                "launch_receipt_sha256": launch_sha,
                "previous_line_sha256": "0" * 64,
                "schema": "sfora-teacher-anchored-progress-v1",
                "sequence": 1,
                "stage": "initialized",
                "update": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    progress.write_bytes(line)
    receipt = (tmp_path / "complete.launch.json").resolve()
    receipt.write_bytes(b"{}\n")
    child = tmp_path / "sleep.py"
    child.write_text("import time\ntime.sleep(60)\n", encoding="utf-8")
    launch = SUBJECT.TeacherAnchoredArmLaunch(
        arm="complete",
        receipt=receipt,
        receipt_sha256=launch_sha,
        output=output,
        progress=progress,
        command=(sys.executable, str(child)),
    )
    monkeypatch.setattr(
        SUBJECT,
        "_sample_teacher_anchored_pressure",
        lambda *_args: _sample(rss=32 * 1024**3 + 1),
    )

    terminal = SUBJECT.run_teacher_anchored_arm_process(launch, poll_seconds=0.01)

    assert terminal.returncode != 0
    assert terminal.result_bytes == b""
    assert terminal.progress.sequence == 1
    assert terminal.completed is False
    assert terminal.stop_receipt is not None
    stop = json.loads(terminal.stop_receipt)
    assert stop["reason"] == "rss-cap"
    assert stop["last_progress_sha256"] == hashlib.sha256(line).hexdigest()
    assert (tmp_path / "complete.result.stop.json").read_bytes() == terminal.stop_receipt


def test_pressure_stop_escalates_when_group_child_ignores_term(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launch_sha = "a" * 64
    output = (tmp_path / "complete.result.json").resolve()
    progress = output.with_suffix(".progress.jsonl")
    progress.write_bytes(
        json.dumps(
            {
                "arm": "complete",
                "epoch": 0,
                "launch_receipt_sha256": launch_sha,
                "previous_line_sha256": "0" * 64,
                "schema": "sfora-teacher-anchored-progress-v1",
                "sequence": 1,
                "stage": "initialized",
                "update": 0,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        + b"\n"
    )
    receipt = (tmp_path / "complete.launch.json").resolve()
    receipt.write_bytes(b"{}\n")
    grandchild_pid = tmp_path / "grandchild.pid"
    child = tmp_path / "fork.py"
    child.write_text(
        """\
import os
import pathlib
import signal
import sys
import time

pid = os.fork()
if pid == 0:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    pathlib.Path(sys.argv[1]).write_text(str(os.getpid()))
    time.sleep(60)
else:
    while not pathlib.Path(sys.argv[1]).exists():
        time.sleep(0.001)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    time.sleep(60)
""",
        encoding="utf-8",
    )
    launch = SUBJECT.TeacherAnchoredArmLaunch(
        arm="complete",
        receipt=receipt,
        receipt_sha256=launch_sha,
        output=output,
        progress=progress,
        command=(sys.executable, str(child), str(grandchild_pid)),
    )
    monkeypatch.setattr(SUBJECT, "_TERMINATION_GRACE_SECONDS", 0.05, raising=False)
    calls = 0

    def pressure(*_args: object) -> object:
        nonlocal calls
        calls += 1
        if not grandchild_pid.exists() and calls < 100:
            return _sample()
        return _sample(rss=32 * 1024**3 + 1)

    monkeypatch.setattr(SUBJECT, "_sample_teacher_anchored_pressure", pressure)

    SUBJECT.run_teacher_anchored_arm_process(launch, poll_seconds=0.01)

    pid = int(grandchild_pid.read_text())
    deadline = time.monotonic() + 1.0
    while Path(f"/proc/{pid}").exists() and time.monotonic() < deadline:
        state = Path(f"/proc/{pid}/stat").read_text().split()[2]
        if state == "Z":
            break
        time.sleep(0.01)
    assert not Path(f"/proc/{pid}").exists() or Path(f"/proc/{pid}/stat").read_text().split()[
        2
    ] == "Z"


def test_panel_receipt_binds_exact_serial_arm_launches_and_results(tmp_path: Path) -> None:
    parsed = SUBJECT.parse_teacher_anchored_panel_args(_panel_arguments(tmp_path))
    launches = SUBJECT.prepare_teacher_anchored_panel(parsed)
    terminals = tuple(
        SUBJECT.TeacherAnchoredArmTerminal(
            returncode=0,
            result_bytes=_completed_arm_result(launch),
            progress=SUBJECT.TeacherAnchoredProgressState(
                sequence=1,
                arm=launch.arm,
                epoch=10,
                update=123,
                line_sha256=(str(index + 1) * 64),
            ),
            completed=True,
            stop_receipt=None,
        )
        for index, launch in enumerate(launches)
    )

    raw = SUBJECT.canonical_teacher_anchored_panel_bytes(parsed, launches, terminals)
    value = json.loads(raw)

    assert tuple(launch.arm for launch in launches) == SUBJECT.TEACHER_ANCHORED_ARMS
    assert raw == json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    assert value["schema"] == "sfora-teacher-anchored-panel-v1"
    assert value["claim_eligible"] is False
    assert list(value["arms"]) == sorted(SUBJECT.TEACHER_ANCHORED_ARMS)
    for launch, terminal in zip(launches, terminals, strict=True):
        assert value["arms"][launch.arm] == {
            "launch_receipt_sha256": launch.receipt_sha256,
            "result_sha256": hashlib.sha256(terminal.result_bytes).hexdigest(),
        }
    with pytest.raises(ValueError, match="panel receipt"):
        SUBJECT.canonical_teacher_anchored_panel_bytes(parsed, launches, terminals[:-1])


def test_panel_executor_runs_each_arm_once_in_registered_order(tmp_path: Path) -> None:
    parsed = SUBJECT.parse_teacher_anchored_panel_args(_panel_arguments(tmp_path))
    observed: list[str] = []

    def run_arm(launch: object) -> object:
        observed.append(launch.arm)
        result = _completed_arm_result(launch)
        launch.output.write_bytes(result)
        return SUBJECT.TeacherAnchoredArmTerminal(
            returncode=0,
            result_bytes=result,
            progress=SUBJECT.TeacherAnchoredProgressState(
                sequence=1,
                arm=launch.arm,
                epoch=10,
                update=1,
                line_sha256="a" * 64,
            ),
            completed=True,
            stop_receipt=None,
        )

    panel = SUBJECT.execute_teacher_anchored_panel(parsed, run_arm=run_arm)

    assert tuple(observed) == SUBJECT.TEACHER_ANCHORED_ARMS
    assert panel == (tmp_path / "panel" / "panel.result.json").read_bytes()
    assert json.loads(panel)["schema"] == "sfora-teacher-anchored-panel-v1"
    with pytest.raises(ValueError, match="already exists"):
        SUBJECT.execute_teacher_anchored_panel(parsed, run_arm=run_arm)


def test_panel_preloads_progress_validator_before_starting_any_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parsed = SUBJECT.parse_teacher_anchored_panel_args(_panel_arguments(tmp_path))
    loaded = False

    def load_validator() -> object:
        nonlocal loaded
        loaded = True
        return object()

    def run_arm(launch: object) -> object:
        assert loaded
        result = _completed_arm_result(launch)
        launch.output.write_bytes(result)
        return SUBJECT.TeacherAnchoredArmTerminal(
            returncode=0,
            result_bytes=result,
            progress=SUBJECT.TeacherAnchoredProgressState(1, launch.arm, 10, 1, "a" * 64),
            completed=True,
            stop_receipt=None,
        )

    monkeypatch.setattr(SUBJECT, "_load_progress_validator", load_validator)

    SUBJECT.execute_teacher_anchored_panel(parsed, run_arm=run_arm)

    assert loaded


def test_panel_rejects_scientifically_stopped_arm_even_when_process_exits_zero(
    tmp_path: Path,
) -> None:
    parsed = SUBJECT.parse_teacher_anchored_panel_args(_panel_arguments(tmp_path))
    launches = SUBJECT.prepare_teacher_anchored_panel(parsed)
    terminals = tuple(
        SUBJECT.TeacherAnchoredArmTerminal(
            returncode=0,
            result_bytes=_completed_arm_result(
                launch,
                stopped_reason=(
                    "epoch-one-fitting-map-regression" if launch.arm == "complete" else None
                ),
            ),
            progress=SUBJECT.TeacherAnchoredProgressState(
                sequence=2,
                arm=launch.arm,
                epoch=10 if launch.arm != "complete" else 1,
                update=10,
                line_sha256="a" * 64,
            ),
            completed=True,
            stop_receipt=None,
        )
        for launch in launches
    )

    with pytest.raises(ValueError, match="arm result"):
        SUBJECT.canonical_teacher_anchored_panel_bytes(parsed, launches, terminals)
