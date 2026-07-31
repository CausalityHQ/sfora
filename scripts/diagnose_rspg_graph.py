#!/usr/bin/env python3
"""Run RSPG's preregistered graph gate on an existing training embedding pack.

This command is deliberately CPU-only and refuses to run if its tensors leave
the CPU. It reads no test examples or retrieval scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from sfora.image_end_to_end import ImageEndToEndConfig, _build_rspg_state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pack", type=Path)
    args = parser.parse_args()
    with np.load(args.pack, allow_pickle=False) as payload:
        missing = {"embeddings", "labels"} - set(payload.files)
        if missing:
            raise ValueError(f"{args.pack}: missing {sorted(missing)}")
        embeddings = np.asarray(payload["embeddings"], dtype=np.float64)
        labels = np.asarray(payload["labels"], dtype=np.int64)

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
