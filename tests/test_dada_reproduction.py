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
    assert command[:3] == ("/opt/dada/bin/python", "-I", "-B")
    assert command[3] == "-c"
    assert "runpy.run_path" in command[4]
    assert "/work/DADA" in command[4]
    assert command[5:] == (
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


def test_checkpoint_reload_command_inserts_pinned_checkout(tmp_path: Path) -> None:
    request = _request(tmp_path)
    checkpoint = tmp_path / "checkpoint.pth.tar"
    command = dada._build_checkpoint_reload_command(request, checkpoint)

    assert command[:4] == (str(request.python), "-I", "-B", "-c")
    assert str(request.source.checkout) in command[4]
    assert "sys.path.insert" in command[4]
    assert command[-1] == str(checkpoint)


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


def test_gpu_memory_parser_selects_only_the_training_pid() -> None:
    output = "111, 1024\n222, 35396\n333, 2048\n"
    assert dada._parse_gpu_memory(output, pid=222) == 35_396
    assert dada._parse_gpu_memory(output, pid=444) is None


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
            elapsed_seconds=75.0,
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
    missing_peak = copy.deepcopy(report)
    missing_peak["resources"]["peak_gpu_memory_mib"] = None
    with pytest.raises(ValueError, match="peak GPU memory"):
        dada.validate_dada_smoke_report(missing_peak)


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


def test_publish_report_never_clobbers_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "report.json"
    destination.write_bytes(b"sentinel")
    payload = dada.invalid_dada_smoke_report("fixture failure")

    with pytest.raises(FileExistsError):
        dada.publish_dada_smoke_report(destination, payload)

    assert destination.read_bytes() == b"sentinel"
    assert not list(tmp_path.glob(".*.tmp"))


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
