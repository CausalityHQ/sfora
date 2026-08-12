from __future__ import annotations

import hashlib
import math
import subprocess
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

    assert dada.build_dada_command(request) == (
        "/opt/dada/bin/python",
        "-I",
        "-B",
        "/work/DADA/main.py",
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
    assert progress.optimizer_steps == 6 * 143
    assert math.isfinite(progress.last_loss)
    assert progress.last_recall_at_1 == pytest.approx(0.85)


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
