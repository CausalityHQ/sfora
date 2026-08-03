#!/usr/bin/env python3
"""Run RSPG's preregistered graph gate on an existing training embedding pack.

This command is deliberately CPU-only and refuses to run if its tensors leave
the CPU. It reads no test examples or retrieval scores.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from sfora.image_end_to_end import ImageEndToEndConfig, _build_rspg_state


def _scalar(value: np.ndarray, name: str) -> Any:
    if value.shape != ():
        raise ValueError(f"RSPG pack {name} must be scalar")
    return value.item()


def _verify_operating_point_artifacts(
    *,
    payload: Any,
    pack_path: Path,
    report_path: Path,
    checkpoint_path: Path,
) -> None:
    required = {"example_ids", "artifact_selection", "artifact_epoch"}
    missing = required - set(payload.files)
    if missing:
        raise ValueError(f"RSPG operating pack is missing {sorted(missing)}")
    if _scalar(payload["artifact_selection"], "artifact_selection") != (
        "final_no_periodic_test_evaluation"
    ):
        raise ValueError("RSPG operating pack is not a final no-test-evaluation artifact")
    if int(_scalar(payload["artifact_epoch"], "artifact_epoch")) != 10:
        raise ValueError("RSPG operating pack is not from epoch 10")
    example_ids = np.asarray(payload["example_ids"])
    if len(np.unique(example_ids)) != len(example_ids):
        raise ValueError("RSPG operating pack has duplicate example IDs")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    config = ImageEndToEndConfig.model_validate(report["config"])
    expected = {
        "dataset_name": "inshop",
        "objectives": ("proxy_anchor",),
        "recipe_id": "proxy_anchor.inshop.official-51db570",
        "train_epochs": 10,
        # This is the recipe's nominal scheduler horizon.  The checkpoint's
        # actual optimizer-step count is checked separately below.
        "train_steps": 1440,
        "eval_test_interval_epochs": 0,
        "checkpoint_selection_interval": 0,
    }
    for name, value in expected.items():
        if getattr(config, name) != value:
            raise ValueError(
                f"RSPG operating report {name}={getattr(config, name)!r}, expected {value!r}"
            )
    if Path(config.save_train_embeddings or "").resolve() != pack_path.resolve():
        raise ValueError("RSPG report is not bound to the supplied training pack path")
    if Path(config.save_model_path or "").resolve() != checkpoint_path.resolve():
        raise ValueError("RSPG report is not bound to the supplied checkpoint path")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("RSPG operating checkpoint is not a final training state")
    # The official partition has 25,882 images.  With batch size 180 and
    # drop_last=True this is 143 optimizer updates per epoch, hence 1,430
    # actual updates at the registered epoch-10 operating point.  This differs
    # intentionally from the recipe's nominal 1,440-step scheduler horizon.
    expected_checkpoint_step = 1430
    if checkpoint.get("training_step") != expected_checkpoint_step:
        raise ValueError(
            "RSPG operating checkpoint is not from step "
            f"{expected_checkpoint_step}"
        )
    if checkpoint.get("training_config") != config.model_dump(mode="json"):
        raise ValueError("RSPG checkpoint configuration differs from its report")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    with np.load(args.pack, allow_pickle=False) as payload:
        missing = {"embeddings", "labels"} - set(payload.files)
        if missing:
            raise ValueError(f"{args.pack}: missing {sorted(missing)}")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int64)
        if (args.report is None) != (args.checkpoint is None):
            raise ValueError("--report and --checkpoint must be supplied together")
        if args.report is not None and args.checkpoint is not None:
            _verify_operating_point_artifacts(
                payload=payload,
                pack_path=args.pack,
                report_path=args.report,
                checkpoint_path=args.checkpoint,
            )

    device = torch.device("cpu")
    config = ImageEndToEndConfig(rspg_weight=1.0)
    state = _build_rspg_state(
        embeddings,
        labels,
        config=config,
        device=device,
        torch_module=torch,
        enforce_diagnostic=True,
    )
    if state.target_embeddings.device.type != "cpu":
        raise RuntimeError("RSPG diagnostic left the CPU")
    print(
        "RSPG_CPU_GATE=PASS "
        f"density={state.edge_density:.4f} "
        f"multi_component_fraction={state.multi_component_fraction:.4f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
