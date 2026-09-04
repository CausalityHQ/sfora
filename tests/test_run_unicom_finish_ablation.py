from __future__ import annotations

import hashlib
import importlib.util
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sfora.unicom_finish_protocol import FinishArm

SCRIPT = Path(__file__).parents[1] / "scripts" / "run_unicom_finish_ablation.py"
SPEC = importlib.util.spec_from_file_location("run_unicom_finish_ablation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _required(tmp_path: Path) -> list[str]:
    return [
        "--source-commit",
        "a" * 40,
        "--arm",
        FinishArm.SMOOTH_AP_PK.value,
        "--config",
        str(tmp_path / "config.json"),
        "--config-sha256",
        "f" * 64,
        "--unicom-checkout",
        str(tmp_path / "unicom"),
        "--official-checkpoint",
        str(tmp_path / "official.pt"),
        "--official-checkpoint-sha256",
        "b" * 64,
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--partition-sha256",
        "c" * 64,
        "--resume-checkpoint",
        str(tmp_path / "epoch-0004.pt"),
        "--resume-checkpoint-sha256",
        "d" * 64,
        "--resume-run-receipt",
        str(tmp_path / "run-receipt.json"),
        "--resume-run-receipt-sha256",
        "e" * 64,
        "--evidence-root",
        str(tmp_path / "evidence"),
        "--output",
        str(tmp_path / "result.json"),
    ]


def test_cli_requires_all_authority_and_explicit_execution(tmp_path: Path) -> None:
    required = _required(tmp_path)
    with pytest.raises(SystemExit):
        MODULE.parse_args(required)
    parsed = MODULE.parse_args([*required, "--execute-finish-ablation"])
    assert parsed.arm is FinishArm.SMOOTH_AP_PK
    with pytest.raises(SystemExit):
        MODULE.parse_args([*required, "--execute-finish-ablation", "--unknown"])


def test_classification_loss_uses_original_eight_mask_arcface() -> None:
    embeddings = torch.randn(4, 8, requires_grad=True)
    classifier = torch.nn.Parameter(torch.randn(3, 8))
    labels = torch.tensor([0, 0, 1, 1])
    observed: dict[str, object] = {}

    def masks(objective, **kwargs):
        observed["objective"] = objective
        observed.update(kwargs)
        return torch.arange(8)[None]

    def loss(emb, head, target, mask, **kwargs):
        observed["loss"] = (emb, head, target, mask, kwargs)
        return emb.square().mean() + head.square().mean()

    trainer = SimpleNamespace(objective_masks=masks, sharded_mask_arcface_loss=loss)
    value = MODULE.finish_loss(
        trainer,
        arm=FinishArm.CLASSIFICATION_PK,
        embeddings=embeddings,
        classifier=classifier,
        labels=labels,
        mask_generator=torch.Generator(),
        margin=0.25,
        scale=32.0,
    )

    assert value.ndim == 0
    assert observed["objective"] == "official-eight-mask"
    assert observed["loss"][-1] == {"margin": 0.25, "scale": 32.0}


def test_smooth_ap_loss_leaves_classifier_gradient_absent(monkeypatch) -> None:
    embeddings = torch.randn(8, 16, requires_grad=True)
    classifier = torch.nn.Parameter(torch.randn(4, 16))
    labels = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    monkeypatch.setattr(
        MODULE,
        "smooth_ap_finish_loss",
        lambda values, batch_labels: values.square().mean()
        + 0.0 * len(batch_labels),
    )

    value = MODULE.finish_loss(
        SimpleNamespace(),
        arm=FinishArm.SMOOTH_AP_PK,
        embeddings=embeddings,
        classifier=classifier,
        labels=labels,
        mask_generator=torch.Generator(),
        margin=0.25,
        scale=32.0,
    )
    value.backward()

    assert embeddings.grad is not None
    assert classifier.grad is None


def test_evaluation_restores_all_rng_streams() -> None:
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    expected_python = random.random()
    expected_numpy = float(np.random.rand())
    expected_torch = torch.rand(1)
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)

    def evaluation():
        random.random()
        np.random.rand()
        torch.rand(1)
        return {"map_at_r": 0.5}

    assert MODULE.isolated_evaluation(evaluation) == {"map_at_r": 0.5}
    assert random.random() == expected_python
    assert float(np.random.rand()) == expected_numpy
    assert torch.equal(torch.rand(1), expected_torch)


def test_parent_receipt_binds_exact_checkpoint_bytes(tmp_path: Path) -> None:
    receipt_path = tmp_path / "run-receipt.json"
    checkpoint = tmp_path / "epoch-0004.pt"
    checkpoint.write_bytes(b"parent")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    receipt = {
        "checkpoints": [
            {
                "epoch": 4,
                "root": "current",
                "path": checkpoint.name,
                "sha256": digest,
                "bytes": checkpoint.stat().st_size,
            }
        ]
    }

    MODULE.require_parent_checkpoint_binding(
        receipt, receipt_path=receipt_path, checkpoint=checkpoint, expected_sha256=digest
    )
    other = tmp_path / "other.pt"
    other.write_bytes(b"other")
    with pytest.raises(ValueError):
        MODULE.require_parent_checkpoint_binding(
            receipt,
            receipt_path=receipt_path,
            checkpoint=other,
            expected_sha256=hashlib.sha256(other.read_bytes()).hexdigest(),
        )


def test_scaled_step_requires_one_real_optimizer_update() -> None:
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _step: 1.0)
    scaler = torch.amp.GradScaler("cpu")
    scaler.scale(parameter.square()).backward()

    assert MODULE.execute_scaled_step(scaler, optimizer, scheduler) == 1
    assert scheduler.last_epoch == 1

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(parameter * torch.tensor(float("inf"))).backward()
    with pytest.raises(ValueError, match="skipped"):
        MODULE.execute_scaled_step(scaler, optimizer, scheduler)
    assert scheduler.last_epoch == 1
