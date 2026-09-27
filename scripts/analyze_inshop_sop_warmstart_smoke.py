#!/usr/bin/env python3
"""Replay the frozen 17-update SOP-to-In-Shop integrity gate on DGX."""

import hashlib
import json
import math
from pathlib import Path

import torch
from safetensors import safe_open

BASE = Path("/home/riomus/runs")
RUN = BASE / "sfora-inshop-sop-warmstart-smoke-v1"
SNAPSHOT = Path(
    "/home/riomus/.cache/huggingface/hub/models--google--siglip2-large-patch16-256/snapshots/787800c8990e6f058423089178e718139608408c/model.safetensors"
)
SOP = BASE / "sfora-sop-true-freeze-freeze-179024-1000-v1/checkpoint.pt"
ARM = {
    "control": BASE / "sfora-inshop-sop-warmstart-control-179024-17-v1",
    "treatment": BASE / "sfora-inshop-sop-warmstart-treatment-179024-17-v1",
}
CACHE = BASE / "sfora-inshop-sop-warmstart-cache-v1"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def frozen(name: str) -> bool:
    return name.startswith("embeddings.") or any(
        name.startswith(f"encoder.layers.{index}.") for index in range(12)
    )


def main() -> None:
    receipts = {name: json.loads((path / "receipt.json").read_text()) for name, path in ARM.items()}
    control, treatment = receipts.values()
    cache = json.loads((CACHE / "receipt.json").read_text())
    checks = {
        "cache_checkpoint_bound": cache["vision_init_sha256"]
        == treatment["vision_init_sha256"]
        == sha256(SOP),
        "cache_content_bound": cache["features_sha256"]
        == treatment["features_sha256"]
        == sha256(CACHE / "train_features.npy"),
        "cache_receipt_bound": treatment["feature_receipt_sha256"]
        == sha256(CACHE / "receipt.json"),
        "control_cache_bound": control["vision_init_sha256"] is None,
        "paired_inputs": control["first_input_batch_sha256"]
        == treatment["first_input_batch_sha256"]
        and len(control["first_input_batch_sha256"]) == 10,
        "paired_schedule": control["executed_schedule_sha256"]
        == treatment["executed_schedule_sha256"],
    }
    for name, path in ARM.items():
        receipt = receipts[name]
        checks[f"{name}_finite_17"] = (
            receipt["updates"] == 17
            and len(receipt["step_seconds"]) == 17
            and len(receipt["preclip_grad_norms"]) == 17
            and all(math.isfinite(value) and value > 0 for value in receipt["preclip_grad_norms"])
            and math.isfinite(receipt["first_loss"])
            and math.isfinite(receipt["last_loss"])
            and receipt["checkpoint_sha256"] == sha256(path / "checkpoint.pt")
        )
    sop_vision = torch.load(SOP, map_location="cpu", weights_only=True)["vision"]
    with safe_open(SNAPSHOT, framework="pt", device="cpu") as pretrained:
        for name, path in ARM.items():
            checkpoint = torch.load(path / "checkpoint.pt", map_location="cpu", weights_only=True)
            if name == "control":

                def source(key: str) -> torch.Tensor:
                    return pretrained.get_tensor("vision_model." + key)
            else:
                source = sop_vision.__getitem__
            checks[f"{name}_frozen_equal_init"] = all(
                torch.equal(value, source(key).half().float())
                for key, value in checkpoint["vision"].items()
                if frozen(key)
            )
            checks[f"{name}_upper_changed"] = any(
                not torch.equal(value, source(key).half().float())
                for key, value in checkpoint["vision"].items()
                if key.startswith("encoder.layers.") and not frozen(key)
            )
            del checkpoint, source
    wall_ratio = treatment["training_wall_seconds"] / control["training_wall_seconds"]
    cuda_ratio = (
        treatment["training_peak_cuda_allocated_bytes"]
        / control["training_peak_cuda_allocated_bytes"]
    )
    checks["wall_at_most_1p20"] = wall_ratio <= 1.2
    checks["cuda_at_most_1p20"] = cuda_ratio <= 1.2
    report = {
        "schema": "sfora-inshop-sop-warmstart-smoke-analysis-v1",
        "claim_eligible": False,
        "split": "official In-Shop TRAIN product-disjoint fit; 17-update integrity only",
        "source_sha256": sha256(Path(__file__)),
        "receipt_sha256": {name: sha256(path / "receipt.json") for name, path in ARM.items()},
        "cache_receipt_sha256": sha256(CACHE / "receipt.json"),
        "training_wall_seconds": {
            name: receipt["training_wall_seconds"] for name, receipt in receipts.items()
        },
        "peak_cuda_allocated_bytes": {
            name: receipt["training_peak_cuda_allocated_bytes"]
            for name, receipt in receipts.items()
        },
        "cache_export_wall_seconds": cache["encode_wall_seconds"],
        "wall_ratio": wall_ratio,
        "cuda_ratio": cuda_ratio,
        "checks": checks,
        "gate_passed": all(checks.values()),
    }
    output = RUN / "analysis.json"
    with output.open("x") as stream:
        json.dump(report, stream, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True), flush=True)
    if not report["gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
