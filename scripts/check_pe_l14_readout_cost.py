#!/usr/bin/env python3
"""Stdlib replay of the terminal frozen L14 readout cost evidence."""

import hashlib
import json
import math
import statistics
from pathlib import Path


def main():
    base = (
        Path(__file__).resolve().parents[1]
        / "docs/evidence/compact_metric/sop-siglip2-substrate-v1"
    )
    root = base / "pe-l14-readout-v1"

    def load(path):
        return json.loads(path.read_text())

    receipt = load(root / "mechanics/receipt.json")
    train = load(root / "mechanics/training.json")
    cpu = load(root / "cpu-preflight.json")
    control = load(base / "pe-augmented-100-v1/pe.json")
    inputs = load(root / "input-profile/readout-input-profile-v1.json")
    median = statistics.median(train["step_seconds"][2:])
    assert (
        len(train["step_seconds"]) == 17
        and median == receipt["median_step_3_17_seconds"]
    )
    assert median > 0.71769696 and not receipt["advance"]
    assert train["scales"] == [128] * 17
    assert train["rgb_sha256"] == control["rgb_sha256"][:17] == inputs["rgb_sha256"]
    assert train["frozen_sha256"] == cpu["whole_source_frozen_sha256"]
    assert all(
        train["initial_group_sha256"][n] != final
        for n, final in train["final_group_sha256"].items()
    )
    assert [row["step"] for row in train["diagnostics"]] == [1, 17]
    assert all(
        math.isfinite(v) and v > 0
        for row in train["diagnostics"]
        for v in row["data_gradient_norms"].values()
    )
    assert all(
        math.isfinite(v) for v in train["losses"] + train["preclip_gradient_norms"]
    )
    assert train["training_state_discarded"] and receipt["discard_training_state"]
    assert not train["quality_read"] and not receipt["quality_read"]
    assert (
        train["peak_cuda_allocated_bytes"]
        == receipt["peak_cuda_allocated_bytes"]
        < 10_000_000_000
    )
    assert (
        hashlib.sha256((root / "cpu-preflight.json").read_bytes()).hexdigest()
        == receipt["preflight_sha256"]
    )
    assert inputs["pixels_sha256"][0] == cpu["first_pixels_sha256"]
    assert (
        not inputs["cuda"]
        and not inputs["quality_read"]
        and inputs["optimizer_updates"] == 0
    )
    print(
        f"PASS terminal cost KILL replay: {median:.12f}s/update; exact controls/roles/resources, no quality"
    )


if __name__ == "__main__":
    main()
