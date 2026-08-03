from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from sfora.image_end_to_end import ImageEndToEndConfig

_spec = importlib.util.spec_from_file_location(
    "diagnose_rspg_graph",
    Path(__file__).resolve().parents[1] / "scripts" / "diagnose_rspg_graph.py",
)
assert _spec is not None and _spec.loader is not None
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)


def test_scalar_rejects_vector() -> None:
    with pytest.raises(ValueError, match="must be scalar"):
        _module._scalar(np.asarray([10]), "epoch")


def test_scalar_returns_item() -> None:
    assert _module._scalar(np.asarray("final"), "selection") == "final"


def test_operating_point_verifier_binds_pack_report_and_checkpoint(tmp_path: Path) -> None:
    pack = tmp_path / "train.npz"
    report = tmp_path / "report.json"
    checkpoint = tmp_path / "checkpoint.pt"
    np.savez_compressed(
        pack,
        embeddings=np.asarray([[1.0, 0.0]], dtype=np.float32),
        labels=np.asarray([0]),
        example_ids=np.asarray(["only"]),
        artifact_selection=np.asarray("final_no_periodic_test_evaluation"),
        artifact_epoch=np.asarray(10),
    )
    config = ImageEndToEndConfig(
        dataset_name="inshop",
        objectives=("proxy_anchor",),
        recipe_id="proxy_anchor.inshop.official-51db570",
        train_epochs=10,
        train_steps=1440,
        eval_test_interval_epochs=0,
        checkpoint_selection_interval=0,
        save_train_embeddings=str(pack),
        save_model_path=str(checkpoint),
    )
    report.write_text(
        json.dumps({"config": config.model_dump(mode="json")}),
        encoding="utf-8",
    )
    torch.save(
        {
            "artifact_selection": "final_training_state",
            "training_step": 1440,
            "training_config": config.model_dump(mode="json"),
        },
        checkpoint,
    )
    with np.load(pack, allow_pickle=False) as payload:
        _module._verify_operating_point_artifacts(
            payload=payload,
            pack_path=pack,
            report_path=report,
            checkpoint_path=checkpoint,
        )

    wrong = tmp_path / "wrong.npz"
    wrong.write_bytes(pack.read_bytes())
    with (
        np.load(wrong, allow_pickle=False) as payload,
        pytest.raises(ValueError, match="not bound"),
    ):
        _module._verify_operating_point_artifacts(
            payload=payload,
            pack_path=wrong,
            report_path=report,
            checkpoint_path=checkpoint,
        )
