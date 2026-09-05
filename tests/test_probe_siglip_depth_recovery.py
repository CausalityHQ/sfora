"""Authority, image role and matched timing tests for the inference-only probe."""

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn

from sfora.data import ImageExample

_SPEC = importlib.util.spec_from_file_location(
    "probe_siglip_depth_recovery",
    Path(__file__).parents[1] / "scripts/probe_siglip_depth_recovery.py",
)
assert _SPEC and _SPEC.loader
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def _raw(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _authority_files(tmp_path, monkeypatch):
    # The byte-root is replaced for this local fixture; no tensor loader is mocked.
    payload = b"authenticated-but-not-yet-parsed-tensor-bytes"
    digest = hashlib.sha256(payload).hexdigest()
    receipt = {
        "schema": "sfora-siglip-proxy-control-seed-v1",
        "claim_eligible": False,
        "seed": 17,
        "config_sha256": probe.control._config_sha256(probe.SiglipProxyControlConfig()),
        "checkpoint": {
            "basename": "seed-017-epoch-060.pt",
            "bytes": len(payload),
            "epoch": 60,
            "sha256": digest,
        },
        "dataset": {"manifest_sha256": "a" * 64},
    }
    raw = _raw(receipt)
    (tmp_path / "seed-017.receipt.json").write_bytes(raw)
    checkpoint = tmp_path / "seed-017/checkpoints/seed-017-epoch-060.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(payload)
    monkeypatch.setattr(probe, "SEED_RECEIPT_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(probe, "CHECKPOINT_SHA256", digest)
    monkeypatch.setattr(probe, "CHECKPOINT_BYTES", len(payload))
    monkeypatch.setattr(probe, "MANIFEST_SHA256", "a" * 64)
    return receipt, checkpoint


def test_authenticate_exact_receipt_and_checkpoint_before_parsing(tmp_path, monkeypatch):
    expected, checkpoint = _authority_files(tmp_path, monkeypatch)
    receipt, resolved = probe.authenticate_control(tmp_path)
    assert receipt == expected and resolved == checkpoint
    checkpoint.write_bytes(checkpoint.read_bytes()[:-1] + b"!")
    with pytest.raises(ValueError):
        probe.authenticate_control(tmp_path)


@pytest.mark.parametrize("mutation", ["receipt", "length", "symlink"])
def test_bad_authority_fails_without_tensor_access(tmp_path, monkeypatch, mutation):
    _, checkpoint = _authority_files(tmp_path, monkeypatch)
    if mutation == "receipt":
        (tmp_path / "seed-017.receipt.json").write_bytes(b"{}\n")
    elif mutation == "length":
        checkpoint.write_bytes(b"x")
    else:
        target = checkpoint.with_suffix(".target")
        checkpoint.rename(target)
        checkpoint.symlink_to(target)
    with pytest.raises(ValueError):
        probe.authenticate_control(tmp_path)


def test_only_first_sorted_optimization_images_are_materialized(monkeypatch):
    opt = tuple(
        ImageExample(
            f"x{i:05}", Image.new("RGB", (3, 4), (i % 256, 1, 2)) if i < 128 else None, i % 49
        )
        for i in range(3963)
    )
    evaluation = tuple(ImageExample(f"x{i + 3963:05}", None, 49 + i % 33) for i in range(2746))
    burned = tuple(ImageExample(f"x{i + 6709:05}", None, 82 + i % 16) for i in range(1345))
    bands = probe.control.ControlExampleBands(opt, evaluation, burned, opt + evaluation + burned)
    digest = probe.control.control_manifest_sha256(bands.ordered_manifest)
    monkeypatch.setattr(probe, "MANIFEST_SHA256", digest)
    images, rows, pixels_sha = probe.select_speed_images(bands)
    assert len(images) == 128 and len(rows) == 128 and len(pixels_sha) == 64
    assert [row["example_id"] for row in rows] == [f"x{i:05}" for i in range(128)]
    assert all(row["label"] < 49 for row in rows)
    assert images[127].getpixel((0, 0)) == (127, 1, 2)
    drift = list(bands.ordered_manifest)
    drift[0], drift[1] = drift[1], drift[0]
    bad = probe.control.ControlExampleBands(opt, evaluation, burned, tuple(drift))
    with pytest.raises(ValueError):
        probe.select_speed_images(bad)


def test_round_pairing_uses_same_batch_order_and_excludes_warmups():
    calls, sync_calls = [], []
    layer = nn.Linear(2, 2)

    def forward(name, values):
        calls.append((name, values[:, 0].tolist(), torch.is_grad_enabled()))
        return layer(values)

    clock_value = 0

    def clock():
        nonlocal clock_value
        clock_value += 20
        return clock_value

    bank = torch.arange(256, dtype=torch.float32).reshape(128, 2)
    result = probe.measure_pair(
        {name: lambda values, n=name: forward(n, values) for name in ("full", "student")},
        bank,
        window=0,
        synchronize=lambda: sync_calls.append(1),
        clock=clock,
    )
    assert result == {"full": [20] * 100, "student": [20] * 100}
    assert len(calls) == 220 and len(sync_calls) == 440
    for i in range(110):
        left, right = calls[2 * i : 2 * i + 2]
        assert (left[0], right[0]) == (("full", "student") if i % 2 == 0 else ("student", "full"))
        assert left[1] == right[1] and len(left[1]) == 8
        assert left[1][0] == 2 * ((i * 8) % 121)
        assert left[2] is False and right[2] is False


def test_cli_requires_explicit_execution_and_refuses_training_surfaces():
    valid = ["--control-root", "/inputs", "--output", "/result.json", "--execute-speed-preflight"]
    assert probe.parse_args(valid).control_root == Path("/inputs")
    for args in (valid[:-1], valid + ["--epochs", "6"], valid + ["--eval-split", "test"]):
        with pytest.raises(SystemExit):
            probe.parse_args(args)
