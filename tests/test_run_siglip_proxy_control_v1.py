"""Deployment-wrapper contract for the pooled SigLIP control."""

from __future__ import annotations

import subprocess
from pathlib import Path

_WRAPPER = Path(__file__).parents[1] / "scripts" / "run_siglip_proxy_control_v1.sh"


def test_wrapper_is_source_bound_offline_smoke_first_and_fixed_three_seed() -> None:
    text = _WRAPPER.read_text()

    subprocess.run(["bash", "-n", str(_WRAPPER)], check=True)
    assert text.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "SOURCE_REVISION" in text and "SOURCE_MANIFEST" in text
    assert "sha256sum --check --strict" in text
    assert "HF_HUB_OFFLINE=1" in text and "TRANSFORMERS_OFFLINE=1" in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    smoke_position = text.index("run_siglip_proxy_control.py smoke")
    train_position = text.index("run_siglip_proxy_control.py train")
    aggregate_position = text.index("run_siglip_proxy_control.py aggregate")
    assert smoke_position < train_position < aggregate_position
    assert "for seed in 17 29 43" in text
    assert "--maximum-checkpoint-bytes" in text
    assert "--evaluation-batch-size 32" in text
    assert "--query-block 128" in text
    assert 'dataset_name="cars", split="test"' not in text
    assert "rm -rf" not in text
