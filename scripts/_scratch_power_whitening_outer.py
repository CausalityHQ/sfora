#!/usr/bin/env python3
"""Outer scorer for cells sealed by the fit-only power-whitening screen."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path

import numpy as np
import torch

from sfora.compact_metric import _score_compact_metric_codes
from sfora.deterministic_similarity_runtime import configure_deterministic_similarity_runtime


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--screen-script", type=Path, required=True)
    parser.add_argument("--screen-script-sha256", required=True)
    parser.add_argument("--screen-result", type=Path, required=True)
    parser.add_argument("--screen-result-sha256", required=True)
    parser.add_argument("--dataset", action="append", nargs=3, metavar=("NAME", "PATH", "SHA256"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or sha256(args.screen_script) != args.screen_script_sha256
        or sha256(args.screen_result) != args.screen_result_sha256
        or not args.dataset
    ):
        raise ValueError("power whitening outer authority differs")
    screen = json.loads(args.screen_result.read_text())
    if screen["schema"] != "sfora-power-whitening-screen-v1" or not screen["decision"]["passed"]:
        raise ValueError("power whitening screen did not pass")
    spec = importlib.util.spec_from_file_location("power_whitening_screen", args.screen_script)
    if spec is None or spec.loader is None:
        raise ImportError(args.screen_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    supplied = {name: (Path(path), digest) for name, path, digest in args.dataset}
    if set(supplied) != set(screen["datasets"]):
        raise ValueError("power whitening outer dataset authority differs")
    for path, digest in supplied.values():
        if sha256(path) != digest:
            raise ValueError("power whitening outer dataset authority differs")

    configure_deterministic_similarity_runtime(17, cpu_threads=2)
    device = torch.device("cuda")
    started = time.monotonic()
    datasets: dict[str, object] = {}
    for name, (path, _digest) in sorted(supplied.items()):
        with np.load(path, allow_pickle=False) as archive:
            fit = torch.from_numpy(
                np.ascontiguousarray(archive["fit_embeddings"], dtype=np.float32)
            )
            fit_labels = torch.from_numpy(
                np.ascontiguousarray(archive["fit_labels"], dtype=np.int64)
            )
            evaluation = torch.from_numpy(
                np.ascontiguousarray(archive["evaluation_embeddings"], dtype=np.float32)
            )
            evaluation_labels = torch.from_numpy(
                np.ascontiguousarray(archive["evaluation_labels"], dtype=np.int64)
            )
        selected_key = screen["datasets"][name]["selected"]
        alpha, regularization = (float(value) for value in selected_key.split(":"))
        encoders = module.fit_cells(fit, fit_labels)
        selected = encoders[(alpha, regularization)]
        pca = encoders[(0.0, 0.0)]
        arms: dict[str, object] = {}
        for arm_name, encoder in (("pca", pca), ("selected", selected)):
            float_score = _score_compact_metric_codes(
                encoder.transform(evaluation), evaluation_labels, device=device
            )
            int8_score = _score_compact_metric_codes(
                encoder.encode(evaluation), evaluation_labels, device=device
            )
            arms[arm_name] = {
                "float_map_at_r": float_score[0],
                "float_recall_at_1": float_score[1],
                "int8_map_at_r": int8_score[0],
                "int8_recall_at_1": int8_score[1],
                "encoder_sha256": encoder.sha256,
            }
        datasets[name] = {"selected_cell": selected_key, "arms": arms}
    result = {
        "schema": "sfora-power-whitening-outer-v1",
        "claim_eligible": False,
        "screen_result_sha256": args.screen_result_sha256,
        "screen_script_sha256": args.screen_script_sha256,
        "datasets": datasets,
        "elapsed_seconds": time.monotonic() - started,
    }
    payload = json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n"
    args.output.write_text(payload)
    print(payload, end="")


if __name__ == "__main__":
    main()
