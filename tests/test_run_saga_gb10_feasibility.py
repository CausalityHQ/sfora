from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from sfora.pass209_m4 import canonical_json_bytes
from sfora.saga_feasibility import (
    FeasibilityEvidence,
    ObjectAuthority,
    PhaseMeasurement,
    ResourceEnvelope,
    canonical_feasibility_result_bytes,
)

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_saga_gb10_feasibility.py"
)
_SPEC = importlib.util.spec_from_file_location("run_saga_gb10_feasibility", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _result_bytes() -> bytes:
    def phase(name: str, elapsed_ns: int) -> PhaseMeasurement:
        return PhaseMeasurement(name, True, elapsed_ns, 1, 1)

    return canonical_feasibility_result_bytes(
        FeasibilityEvidence(
            source_commit="a" * 40,
            controller_commit="b" * 40,
            binary_sha256="c" * 64,
            environment_sha256="d" * 64,
            host="spark-fixture",
            model=ObjectAuthority("model", "snapshot.json", 1, "e" * 64),
            fixture=ObjectAuthority("fixture", "fixture.json", 1, "f" * 64),
            envelope=ResourceEnvelope(
                103_079_215_104,
                118_111_600_640,
                7_200_000_000_000,
                300_000_000_000,
            ),
            load=phase("load", 1),
            rollout=phase("rollout", 2),
            replay=phase("replay", 3),
            attention=phase("attention", 4),
            dml=phase("dml", 5),
            deterministic=True,
            attention_available=True,
            backend_valid=True,
            authority_valid=True,
            memory_within_envelope=True,
            time_within_envelope=True,
            dataset_reads=0,
            label_reads=0,
            evaluation_reads=0,
            optimizer_steps=0,
        )
    )


class _FakeRunner:
    def __init__(
        self,
        observations: list[object],
        *,
        child_result: bytes,
        exit_code: int = 0,
    ) -> None:
        self.observations = observations
        self.child_result = child_result
        self.exit_code = exit_code
        self.spawn_count = 0
        self.terminate_count = 0
        self.restart_count = 0
        self.environment: dict[str, str] = {}
        self.argv: tuple[str, ...] = ()
        self.alive = False

    def spawn(
        self,
        argv: tuple[str, ...],
        environment: dict[str, str],
        child_result_path: Path,
    ) -> object:
        self.spawn_count += 1
        self.argv = argv
        self.environment = environment
        self.alive = True
        child_result_path.write_bytes(self.child_result)
        return object()

    def observe(self, _process: object) -> object:
        observation = self.observations.pop(0)
        if not observation.process_alive:
            self.alive = False
        return observation

    def terminate(self, _process: object) -> None:
        self.terminate_count += 1
        self.alive = False

    def wait(self, _process: object) -> int:
        self.alive = False
        return self.exit_code

    def is_alive(self, _process: object) -> bool:
        return self.alive


def _observation(**updates: object) -> object:
    values = {
        "process_alive": True,
        "elapsed_ns": 1,
        "progress_age_ns": 1,
        "cuda_reserved_bytes": 1,
        "process_rss_bytes": 1,
        "psi_full_avg10_ppm": 0,
        "swap_growth_bytes": 0,
    }
    values.update(updates)
    return _MODULE.ResourceObservation(**values)


def _controller(tmp_path: Path) -> object:
    model = tmp_path / "model"
    model.mkdir()
    snapshot = tmp_path / "snapshot.json"
    fixture = tmp_path / "fixture.json"
    scientific_cli = tmp_path / "diagnostic.py"
    for path in (snapshot, fixture, scientific_cli):
        path.write_bytes(b"{}\n")
    paths = _MODULE.ControllerPaths(
        model_root=model,
        snapshot_manifest=snapshot,
        fixture=fixture,
        scientific_cli=scientific_cli,
        scratch_root=tmp_path / "scratch",
        result_output=tmp_path / "result.json",
        terminal_output=tmp_path / "terminal.json",
    )
    identity = _MODULE.ControllerIdentity(
        source_commit="a" * 40,
        controller_commit="b" * 40,
        binary_sha256="c" * 64,
        environment_sha256="d" * 64,
        host="spark-fixture",
    )
    return _MODULE.FeasibilityController(paths=paths, identity=identity)


def test_controller_launches_one_offline_process_group_and_publishes_result(
    tmp_path: Path,
) -> None:
    result = _result_bytes()
    runner = _FakeRunner(
        [_observation(), _observation(process_alive=False)], child_result=result
    )
    terminal = _controller(tmp_path).run(runner=runner)
    assert terminal.outcome == "FITS"
    assert terminal.result_published is True
    assert runner.spawn_count == 1
    assert runner.terminate_count == 0
    assert runner.environment["HF_HUB_OFFLINE"] == "1"
    assert runner.environment["TRANSFORMERS_OFFLINE"] == "1"
    assert not any("dataset" in argument for argument in runner.argv)
    assert "--progress-output" in runner.argv
    assert (tmp_path / "result.json").read_bytes() == result
    assert not (tmp_path / "terminal.json").exists()
    assert not (tmp_path / "scratch").exists()


def _controller_argv(tmp_path: Path) -> list[str]:
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    files = []
    for name in ("snapshot.json", "fixture.json", "diagnostic.py"):
        path = tmp_path / name
        path.write_bytes(b"{}\n")
        files.append(path)
    return [
        "--model-root",
        str(model),
        "--snapshot-manifest",
        str(files[0]),
        "--fixture",
        str(files[1]),
        "--scientific-cli",
        str(files[2]),
        "--scratch-root",
        str(tmp_path / "scratch"),
        "--result-output",
        str(tmp_path / "result.json"),
        "--terminal-output",
        str(tmp_path / "terminal.json"),
        "--source-commit",
        "a" * 40,
        "--controller-commit",
        "b" * 40,
        "--binary-sha256",
        "c" * 64,
        "--environment-sha256",
        "d" * 64,
        "--host",
        "spark-fixture",
        "--execute-controller",
    ]


def test_controller_cli_is_strict_and_local_only(tmp_path: Path) -> None:
    args = _MODULE.parse_controller_args(_controller_argv(tmp_path))
    assert args.execute_controller is True
    assert args.model_root == tmp_path / "model"
    for forbidden in ("--dataset", "--aws-profile", "--model-uri", "--train"):
        with pytest.raises(SystemExit):
            _MODULE.parse_controller_args(
                [*_controller_argv(tmp_path), forbidden, "x"]
            )


@pytest.mark.parametrize(
    ("fault", "updates", "expected"),
    [
        ("cuda", {"cuda_reserved_bytes": 103_079_215_105}, "MEMORY_FAIL"),
        ("rss", {"process_rss_bytes": 118_111_600_641}, "MEMORY_FAIL"),
        ("psi-immediate", {"psi_full_avg10_ppm": 790_000}, "MEMORY_FAIL"),
        ("swap", {"swap_growth_bytes": 268_435_457}, "MEMORY_FAIL"),
        ("progress", {"progress_age_ns": 300_000_000_001}, "TIME_BUDGET_FAIL"),
        ("wall", {"elapsed_ns": 7_200_000_000_001}, "TIME_BUDGET_FAIL"),
    ],
)
def test_controller_stops_once_preserves_terminal_and_cleans(
    tmp_path: Path, fault: str, updates: dict[str, object], expected: str
) -> None:
    del fault
    runner = _FakeRunner(
        [_observation(**updates)],
        child_result=_result_bytes(),
    )
    terminal = _controller(tmp_path).run(runner=runner)
    assert terminal.outcome == expected
    assert terminal.restart_count == 0
    assert terminal.process_cleared is True
    assert terminal.scratch_cleared is True
    assert runner.spawn_count == 1
    assert runner.terminate_count == 1
    assert not (tmp_path / "result.json").exists()
    assert (tmp_path / "terminal.json").read_bytes().endswith(b"\n")
    assert not (tmp_path / "scratch").exists()


def test_controller_stops_on_sustained_psi_only_after_three_samples(
    tmp_path: Path,
) -> None:
    runner = _FakeRunner(
        [
            _observation(psi_full_avg10_ppm=500_000),
            _observation(psi_full_avg10_ppm=500_000),
            _observation(psi_full_avg10_ppm=500_000),
        ],
        child_result=_result_bytes(),
    )
    terminal = _controller(tmp_path).run(runner=runner)
    assert terminal.outcome == "MEMORY_FAIL"
    assert runner.terminate_count == 1


@pytest.mark.parametrize(
    "fault", ["child-exit", "noncanonical-result", "canonical-forged-result"]
)
def test_controller_rejects_child_or_result_failure(
    tmp_path: Path, fault: str
) -> None:
    if fault == "noncanonical-result":
        child_result = b'{"claim_eligible": false, "outcome": "FITS"}\n'
    elif fault == "canonical-forged-result":
        child_result = canonical_json_bytes(
            {"outcome": "FITS", "claim_eligible": False}
        )
    else:
        child_result = _result_bytes()
    runner = _FakeRunner(
        [_observation(process_alive=False)],
        child_result=child_result,
        exit_code=1 if fault == "child-exit" else 0,
    )
    terminal = _controller(tmp_path).run(runner=runner)
    assert terminal.outcome == "BACKEND_INVALID"
    assert terminal.result_published is False
    assert not (tmp_path / "result.json").exists()
    assert not (tmp_path / "scratch").exists()
