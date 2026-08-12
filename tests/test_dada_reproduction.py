from __future__ import annotations

import copy
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from sfora import dada_reproduction as dada

_INSHOP_YAML = """# for inshop
dataset: "inshop"
n_epochs: 200
batch_size: 180
fd2_bn: true
fc2_bn: true
pos_class_mix: true
arch: resnet50_layernorm_double
d_dis_ratio: 0.01
decay: 0.0001
dis_decay: 0.00005
fc_fc2_dim: 4096
fd_fc1_dim: 512
loss_oproxy_neg_alpha: 200
loss_oproxy_pos_alpha: 40
lr: 0.00012
lr_reduce_rate: 0.25
lr_reduce_step: 40
oproxy_ratio: 0.0075
mix_alpha: 3
mix_beta: 3
warmup: 5
store_improvements: true
"""


def _git(checkout: Path, *args: str) -> str:
    result = subprocess.run(
        ("git", "-C", str(checkout), *args),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "DADA"
    (checkout / "configs").mkdir(parents=True)
    (checkout / "configs" / "inshop.yaml").write_text(_INSHOP_YAML)
    (checkout / "main.py").write_text("raise SystemExit(0)\n")
    _git(checkout, "init", "-q")
    _git(checkout, "config", "user.email", "test@example.com")
    _git(checkout, "config", "user.name", "Test")
    _git(checkout, "add", "configs/inshop.yaml", "main.py")
    _git(checkout, "commit", "-qm", "fixture")
    return checkout


def test_validate_dada_source_accepts_exact_revision_and_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(dada, "DADA_REVISION", _git(checkout, "rev-parse", "HEAD"))
    monkeypatch.setattr(
        dada,
        "INSHOP_CONFIG_SHA256",
        hashlib.sha256(_INSHOP_YAML.encode()).hexdigest(),
    )

    source = dada.validate_dada_source(checkout)

    assert source.revision == dada.DADA_REVISION
    assert source.config["n_epochs"] == 200
    assert source.config["batch_size"] == 180
    assert source.config["arch"] == "resnet50_layernorm_double"


def test_build_smoke_config_changes_only_epoch_count(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(dada, "DADA_REVISION", _git(checkout, "rev-parse", "HEAD"))
    monkeypatch.setattr(
        dada,
        "INSHOP_CONFIG_SHA256",
        hashlib.sha256(_INSHOP_YAML.encode()).hexdigest(),
    )
    source = dada.validate_dada_source(checkout)
    destination = tmp_path / "smoke.yaml"

    digest = dada.build_smoke_config(source, destination)

    written = yaml.safe_load(destination.read_text())
    expected = dict(source.config)
    expected["n_epochs"] = 6
    assert written == expected
    assert digest == hashlib.sha256(destination.read_bytes()).hexdigest()


@pytest.mark.parametrize("epochs", [0, 1, 5, 7, 200, True])
def test_build_smoke_config_rejects_any_other_epoch_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    epochs: object,
) -> None:
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(dada, "DADA_REVISION", _git(checkout, "rev-parse", "HEAD"))
    monkeypatch.setattr(
        dada,
        "INSHOP_CONFIG_SHA256",
        hashlib.sha256(_INSHOP_YAML.encode()).hexdigest(),
    )
    source = dada.validate_dada_source(checkout)

    with pytest.raises(ValueError, match="exactly 6"):
        dada.build_smoke_config(source, tmp_path / "smoke.yaml", epochs=epochs)  # type: ignore[arg-type]


def test_validate_dada_source_rejects_dirty_tracked_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(dada, "DADA_REVISION", _git(checkout, "rev-parse", "HEAD"))
    monkeypatch.setattr(
        dada,
        "INSHOP_CONFIG_SHA256",
        hashlib.sha256(_INSHOP_YAML.encode()).hexdigest(),
    )
    (checkout / "main.py").write_text("print('dirty')\n")

    with pytest.raises(ValueError, match="dirty"):
        dada.validate_dada_source(checkout)


def test_build_smoke_config_never_clobbers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkout = _make_checkout(tmp_path)
    monkeypatch.setattr(dada, "DADA_REVISION", _git(checkout, "rev-parse", "HEAD"))
    monkeypatch.setattr(
        dada,
        "INSHOP_CONFIG_SHA256",
        hashlib.sha256(_INSHOP_YAML.encode()).hexdigest(),
    )
    source = dada.validate_dada_source(checkout)
    destination = tmp_path / "smoke.yaml"
    destination.write_bytes(b"sentinel")

    with pytest.raises(FileExistsError):
        dada.build_smoke_config(source, destination)

    assert destination.read_bytes() == b"sentinel"


def test_inshop_dataset_preflight_requires_real_registered_subdirectory(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "data"
    dataset_root.mkdir()
    with pytest.raises(ValueError, match="In-Shop dataset"):
        dada.validate_inshop_dataset_root(dataset_root)

    inshop = dataset_root / "inshop"
    inshop.mkdir()
    with pytest.raises(ValueError, match="partition"):
        dada.validate_inshop_dataset_root(dataset_root)

    (inshop / "list_eval_partition.txt").write_text(
        "1\nimage_name item_id evaluation_status\nimg/example.jpg id_1 train\n"
    )
    with pytest.raises(ValueError, match="registered image"):
        dada.validate_inshop_dataset_root(dataset_root)

    (inshop / "img").mkdir()
    (inshop / "img" / "example.jpg").write_bytes(b"image")
    assert dada.validate_inshop_dataset_root(dataset_root) == dataset_root.resolve()


def test_command_preserves_official_cli_contract(tmp_path: Path) -> None:
    request = dada.DadaSmokeRequest(
        python=Path("/opt/dada/bin/python"),
        source=dada.DadaSource(
            checkout=Path("/work/DADA"),
            revision=dada.DADA_REVISION,
            config_path=Path("/work/DADA/configs/inshop.yaml"),
            config_sha256=dada.INSHOP_CONFIG_SHA256,
            config={},
        ),
        smoke_config=Path("/work/smoke.yaml"),
        dataset_root=Path("/data"),
        output_root=tmp_path / "results",
        save_name="dada-inshop-smoke-seed0",
        gpu=0,
        seed=0,
    )

    command = dada.build_dada_command(request)
    assert command[:4] == ("/opt/dada/bin/python", "-I", "-B", "-u")
    assert command[4] == "-c"
    assert "runpy.run_path" in command[5]
    assert "/work/DADA" in command[5]
    assert command[6:] == (
        "--source_path",
        "/data",
        "--save_path",
        str(tmp_path / "results"),
        "--save_name",
        "dada-inshop-smoke-seed0",
        "--config",
        "/work/smoke.yaml",
        "--gpu",
        "0",
        "--seed",
        "0",
    )


def test_isolated_command_loads_only_the_pinned_upstream_checkout(tmp_path: Path) -> None:
    checkout = tmp_path / "DADA"
    checkout.mkdir()
    (checkout / "pandas.py").write_text("def read_table(*args,**kwargs): return None\n")
    (checkout / "helper.py").write_text("VALUE = 'from-pinned-checkout'\n")
    (checkout / "main.py").write_text("import helper\nprint(helper.VALUE)\n")
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text("n_epochs: 6\n")
    request = dada.DadaSmokeRequest(
        python=Path(sys.executable),
        source=dada.DadaSource(
            checkout=checkout,
            revision=dada.DADA_REVISION,
            config_path=checkout / "configs" / "inshop.yaml",
            config_sha256=dada.INSHOP_CONFIG_SHA256,
            config={},
        ),
        smoke_config=smoke,
        dataset_root=tmp_path,
        output_root=tmp_path / "results",
        save_name="smoke",
        gpu=0,
        seed=0,
    )

    completed = subprocess.run(
        dada.build_dada_command(request),
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "from-pinned-checkout"


def test_isolated_command_translates_removed_pandas_delim_whitespace(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "DADA"
    checkout.mkdir()
    dependency = tmp_path / "deps"
    dependency.mkdir()
    (dependency / "pandas.py").write_text(
        "LAST=None\n"
        "def read_table(path,**kwargs):\n"
        " global LAST\n"
        " if 'delim_whitespace' in kwargs: raise TypeError('removed')\n"
        " LAST=kwargs\n"
    )
    table = tmp_path / "table.txt"
    table.write_text("a b\n1 2\n")
    (checkout / "main.py").write_text(
        "import pandas as pd,sys\n"
        "pd.read_table(sys.argv[1],delim_whitespace=True)\n"
        "print(pd.LAST['sep'])\n"
    )
    smoke = tmp_path / "smoke.yaml"
    smoke.write_text("n_epochs: 6\n")
    request = dada.DadaSmokeRequest(
        python=Path(sys.executable),
        source=dada.DadaSource(
            checkout=checkout,
            revision=dada.DADA_REVISION,
            config_path=checkout / "configs" / "inshop.yaml",
            config_sha256=dada.INSHOP_CONFIG_SHA256,
            config={},
        ),
        smoke_config=smoke,
        dataset_root=tmp_path,
        output_root=tmp_path / "results",
        save_name="smoke",
        gpu=0,
        seed=0,
        dependency_root=dependency,
    )
    command = (*dada.build_dada_command(request), str(table))

    completed = subprocess.run(command, check=False, capture_output=True, text=True)

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == r"\s+"


def test_checkpoint_reload_command_inserts_pinned_checkout(tmp_path: Path) -> None:
    request = _request(tmp_path)
    checkpoint = tmp_path / "checkpoint.pth.tar"
    command = dada._build_checkpoint_reload_command(request, checkpoint)

    assert command[:4] == (str(request.python), "-I", "-B", "-c")
    assert str(request.source.checkout) in command[4]
    assert "sys.path.insert" in command[4]
    assert command[-1] == str(checkpoint)


def test_active_python_executable_preserves_virtual_environment_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = tmp_path / "python3"
    base.write_bytes(b"base")
    virtual = tmp_path / "venv" / "bin" / "python"
    virtual.parent.mkdir(parents=True)
    virtual.symlink_to(base)
    monkeypatch.setattr(dada.sys, "executable", str(virtual))

    assert dada.active_python_executable() == virtual
    assert dada.active_python_executable() != virtual.resolve()


def test_environment_probe_imports_exact_upstream_graph_through_registered_paths(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    dependency_root = tmp_path / "deps"
    dependency_root.mkdir()
    request = dada.DadaSmokeRequest(**{**request.__dict__, "dependency_root": dependency_root})

    command = dada._build_environment_probe_command(request)

    assert command[:4] == (str(request.python), "-I", "-B", "-c")
    assert command[4].index(str(dependency_root)) < command[4].index(
        str(request.source.checkout)
    )
    for module in (
        "faiss",
        "sklearn",
        "pandas",
        "pretrainedmodels",
        "termcolor",
        "tqdm",
        "yaml",
        "torchvision",
        "parameters",
        "metrics",
        "evaluation",
        "architectures",
        "datasampler",
        "datasets",
        "criteria",
    ):
        assert module in command[4]


def test_command_inserts_dedicated_dependency_root_without_changing_torch(
    tmp_path: Path,
) -> None:
    request = _request(tmp_path)
    dependency_root = tmp_path / "deps"
    dependency_root.mkdir()
    request = dada.DadaSmokeRequest(**{**request.__dict__, "dependency_root": dependency_root})

    command = dada.build_dada_command(request)

    assert str(dependency_root) in command[5]
    assert command[5].index(str(dependency_root)) < command[5].index(str(request.source.checkout))


@pytest.mark.parametrize(("field", "value"), [("seed", 1), ("gpu", 1)])
def test_command_rejects_unregistered_seed_or_gpu(
    tmp_path: Path, field: str, value: int
) -> None:
    request = _request(tmp_path)
    mutated = dada.DadaSmokeRequest(
        **{**request.__dict__, field: value},
    )

    with pytest.raises(ValueError, match=f"{field} must be exactly 0"):
        dada.build_dada_command(mutated)


def test_log_parser_requires_finite_loss_and_optimizer_progress() -> None:
    lines = []
    for epoch in range(6):
        lines.extend(
            (
                f"[Train Epoch {epoch}]: 100.00% [143/143, 00:10<00:00, "
                f"DisL:0.5000, DML:{1.0 / (epoch + 1):.4f}]",
                "Embed-Type: embeds:",
                f"e_recall@1: {0.80 + epoch / 100:.4f}",
                "Total Epoch Runtime: 12.50s",
            )
        )

    progress = dada.parse_dada_log(lines)

    assert progress.completed_epochs == 6
    assert progress.optimizer_steps == 5 * 143
    assert math.isfinite(progress.last_loss)
    assert progress.last_recall_at_1 == pytest.approx(0.85)
    assert progress.best_recall_at_1 == pytest.approx(0.85)
    assert progress.last_epoch_seconds == pytest.approx(12.5)


def test_log_parser_accepts_real_tqdm_intermediate_frames() -> None:
    frames = []
    for epoch in range(6):
        frames.extend(
            (
                f"[Train Epoch {epoch}]: 0.00% [0/8, 00:00/?, DisL:0.5, DML:0.4]",
                f"[Train Epoch {epoch}]: 50.00% [4/8, 00:01/00:01, DisL:0.4, DML:0.3]",
                f"[Train Epoch {epoch}]: 100.00% [8/8, 00:02/00:00, DisL:0.3, DML:0.2]",
                "e_recall@1: 0.8",
                "Total Epoch Runtime: 12.50s",
            )
        )

    progress = dada.parse_dada_log("\r".join(frames).splitlines())

    assert progress.completed_epochs == 6
    assert progress.optimizer_steps == 5 * 8
    assert progress.last_loss == pytest.approx(0.2)


def test_log_parser_rejects_incomplete_last_tqdm_frame() -> None:
    with pytest.raises(ValueError, match="optimizer progress"):
        dada.parse_dada_log(
            (
                "[Train Epoch 0]: 0.00% [0/8, 00:00/?, DisL:0.5, DML:0.4]",
                "[Train Epoch 0]: 50.00% [4/8, 00:01/00:01, DisL:0.4, DML:0.3]",
                "e_recall@1: 0.8",
                "Total Epoch Runtime: 12.50s",
            )
        )


def test_gpu_memory_parser_selects_only_the_training_pid() -> None:
    output = "111, 1024\n222, 35396\n333, 2048\n"
    assert dada._parse_gpu_memory(output, pid=222) == 35_396
    assert dada._parse_gpu_memory(output, pid=444) is None


@pytest.mark.parametrize(
    "failure",
    (FileNotFoundError("nvidia-smi"), subprocess.TimeoutExpired("nvidia-smi", 5)),
)
def test_gpu_memory_telemetry_failure_is_nonfatal(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    def fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise failure

    monkeypatch.setattr(dada.subprocess, "run", fail)

    assert dada._query_gpu_memory(pid=123) is None


def test_child_is_terminated_and_reaped_when_monitoring_is_interrupted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    request.output_root.mkdir()

    class Process:
        pid = 321
        terminated = False
        waited = False

        def poll(self) -> None:
            return None

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            self.waited = True
            return -15

    process = Process()
    monkeypatch.setattr(dada.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(dada, "_query_gpu_memory", lambda **_kwargs: None)

    def interrupt(_seconds: float) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(dada.time, "sleep", interrupt)

    with pytest.raises(KeyboardInterrupt):
        dada._execute_dada_child(request)

    assert process.terminated is True
    assert process.waited is True


@pytest.mark.parametrize(
    ("replacement", "message"),
    [
        ("DML:nan", "non-finite"),
        ("DML:inf", "non-finite"),
        ("[0/143", "optimizer progress"),
        ("Traceback (most recent call last):", "traceback"),
        ("CUDA out of memory", "CUDA out of memory"),
    ],
)
def test_log_parser_rejects_structural_training_failures(
    replacement: str,
    message: str,
) -> None:
    baseline = (
        "[Train Epoch 0]: 100.00% [143/143, 00:10<00:00, DisL:0.5, DML:0.4]",
        "e_recall@1: 0.8",
        "Total Epoch Runtime: 12.5s",
    )
    if replacement.startswith("DML:"):
        lines = (baseline[0].replace("DML:0.4", replacement), *baseline[1:])
    elif replacement.startswith("["):
        lines = (baseline[0].replace("[143/143", replacement), *baseline[1:])
    else:
        lines = (*baseline, replacement)

    with pytest.raises(ValueError, match=message):
        dada.parse_dada_log(lines)


def _request(tmp_path: Path) -> dada.DadaSmokeRequest:
    smoke_config = tmp_path / "smoke.yaml"
    smoke_config.write_text("n_epochs: 6\n")
    return dada.DadaSmokeRequest(
        python=Path(sys.executable),
        source=dada.DadaSource(
            checkout=tmp_path / "DADA",
            revision=dada.DADA_REVISION,
            config_path=tmp_path / "DADA" / "configs" / "inshop.yaml",
            config_sha256=dada.INSHOP_CONFIG_SHA256,
            config={"n_epochs": 200},
        ),
        smoke_config=smoke_config,
        dataset_root=tmp_path / "data",
        output_root=tmp_path / "results",
        save_name="dada-inshop-smoke-seed0",
        gpu=0,
        seed=0,
    )


def _successful_log() -> str:
    rows = []
    for epoch in range(6):
        rows.extend(
            (
                f"[Train Epoch {epoch}]: 100.00% [143/143, 00:10<00:00, "
                f"DisL:0.5000, DML:{1.0 / (epoch + 1):.4f}]",
                f"e_recall@1: {0.80 + epoch / 100:.4f}",
                "Total Epoch Runtime: 12.50s",
            )
        )
    return "\n".join(rows)


def test_success_report_recomputes_runtime_and_budget_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    checkpoint = request.output_root / "inshop" / "checkpoint.pth.tar"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        dada,
        "_execute_dada_child",
        lambda _request: dada.DadaChildResult(
            returncode=0,
            stdout=_successful_log(),
            elapsed_seconds=90.0,
            peak_gpu_memory_mib=12_345,
        ),
    )
    monkeypatch.setattr(dada, "_checkpoint_is_reloadable", lambda *_: True)
    monkeypatch.setattr(
        dada,
        "_probe_environment",
        lambda _request: {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "cuda": "13.0",
            "device": "NVIDIA GB10",
        },
    )

    report = dada.run_dada_smoke(request)
    persisted = json.loads(json.dumps(report))

    dada.validate_dada_smoke_report(persisted)
    assert report["status"] == "PASS"
    assert report["projection"]["projected_full_run_seconds"] == pytest.approx(2500.0)
    assert report["progress"]["optimizer_steps"] == 715
    assert report["evaluation"]["last_recall_at_1"] == pytest.approx(0.85)
    assert report["evaluation"]["best_recall_at_1"] == pytest.approx(0.85)
    assert report["evaluation"]["best_checkpoint"] == str(checkpoint)
    assert report["config"]["smoke_config_sha256"] == hashlib.sha256(
        request.smoke_config.read_bytes()
    ).hexdigest()
    assert report["progress"]["last_epoch_seconds"] == pytest.approx(12.5)
    missing_peak = copy.deepcopy(report)
    missing_peak["resources"]["peak_gpu_memory_mib"] = None
    dada.validate_dada_smoke_report(missing_peak)


@pytest.mark.parametrize(
    ("stdout", "expected"),
    (
        ("CUDA out of memory", "CUDA out of memory"),
        ("Traceback (most recent call last):\nRuntimeError: boom", "traceback"),
        ("plain failure", "plain failure"),
    ),
)
def test_nonzero_child_report_preserves_bounded_diagnosis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    expected: str,
) -> None:
    request = _request(tmp_path)
    request.output_root.mkdir()
    monkeypatch.setattr(
        dada,
        "_execute_dada_child",
        lambda _request: dada.DadaChildResult(1, stdout, 3.0, None),
    )
    monkeypatch.setattr(
        dada,
        "_probe_environment",
        lambda _request: {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "cuda": "13.0",
            "device": "NVIDIA GB10",
        },
    )

    report = dada.run_dada_smoke(request)

    assert report["status"] == "INCOMPATIBLE"
    assert expected in report["failure"]
    assert len(report["failure"]) <= 1024
    dada.validate_dada_smoke_report(report)


def test_runtime_parse_failure_returns_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    request.output_root.mkdir()
    monkeypatch.setattr(
        dada,
        "_execute_dada_child",
        lambda _request: dada.DadaChildResult(
            returncode=0,
            stdout="unexpected upstream output",
            elapsed_seconds=3.0,
            peak_gpu_memory_mib=512,
        ),
    )
    monkeypatch.setattr(
        dada,
        "_probe_environment",
        lambda _request: {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "cuda": "13.0",
            "device": "NVIDIA GB10",
        },
    )

    report = dada.run_dada_smoke(request)

    assert report["status"] == "INVALID"
    assert "optimizer progress" in report["failure"]
    dada.validate_dada_smoke_report(report)


def test_missing_checkpoint_returns_invalid_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request(tmp_path)
    request.output_root.mkdir()
    monkeypatch.setattr(
        dada,
        "_execute_dada_child",
        lambda _request: dada.DadaChildResult(
            returncode=0,
            stdout=_successful_log(),
            elapsed_seconds=75.0,
            peak_gpu_memory_mib=12_345,
        ),
    )
    monkeypatch.setattr(
        dada,
        "_probe_environment",
        lambda _request: {
            "python": "3.12.3",
            "torch": "2.12.1+cu130",
            "cuda": "13.0",
            "device": "NVIDIA GB10",
        },
    )

    report = dada.run_dada_smoke(request)

    assert report["status"] == "INVALID"
    assert report["failure"] == "DADA smoke produced no checkpoint"
    dada.validate_dada_smoke_report(report)


def test_publish_report_never_clobbers_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    destination.write_bytes(b"sentinel")
    payload = dada.invalid_dada_smoke_report("fixture failure")

    with pytest.raises(FileExistsError):
        dada.publish_dada_smoke_report(destination, payload)

    assert destination.read_bytes() == b"sentinel"
    assert not list(tmp_path.glob(".*.tmp"))


def test_publish_report_reloads_mode_0600_and_preserves_foreign_temp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "report.json"
    foreign_temp = tmp_path / ".report.json.foreign.tmp"
    foreign_temp.write_bytes(b"foreign")
    payload = dada.invalid_dada_smoke_report("fixture failure")
    original = dada.validate_dada_smoke_report
    validations = 0

    def count_validations(value: object) -> None:
        nonlocal validations
        validations += 1
        original(value)

    monkeypatch.setattr(dada, "validate_dada_smoke_report", count_validations)

    dada.publish_dada_smoke_report(destination, payload)

    assert validations == 2
    assert destination.stat().st_mode & 0o777 == 0o600
    assert dada._strict_json_object(destination.read_bytes()) == payload
    assert foreign_temp.read_bytes() == b"foreign"


@pytest.mark.parametrize(
    ("path", "replacement"),
    [
        (("config", "seed"), True),
        (("environment", "torch"), 2121),
        (("process", "returncode"), True),
        (("progress", "completed_epochs"), False),
        (("resources", "peak_gpu_memory_mib"), -1),
        (("command",), "python main.py"),
    ],
)
def test_report_rejects_wrong_concrete_types_and_impossible_resources(
    path: tuple[str, ...], replacement: object
) -> None:
    payload = dada.invalid_dada_smoke_report("fixture failure")
    mutated = copy.deepcopy(payload)
    target: object = mutated
    for key in path[:-1]:
        assert isinstance(target, dict)
        target = target[key]
    assert isinstance(target, dict)
    target[path[-1]] = replacement

    with pytest.raises((TypeError, ValueError)):
        dada.validate_dada_smoke_report(mutated)


def test_cli_refuses_existing_report_before_reading_source(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    destination.write_bytes(b"sentinel")
    script = Path(__file__).parents[1] / "scripts" / "run_dada_inshop_smoke.py"
    assert script.is_file()

    completed = subprocess.run(
        (
            sys.executable,
            str(script),
            "--dada-checkout",
            str(tmp_path / "missing-dada"),
            "--dataset-root",
            str(tmp_path / "missing-data"),
            "--work-root",
            str(tmp_path / "work"),
            "--dependency-root",
            str(tmp_path / "missing-deps"),
            "--output",
            str(destination),
            "--gpu",
            "0",
            "--seed",
            "0",
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 2
    assert "output already exists" in completed.stderr
    assert destination.read_bytes() == b"sentinel"
