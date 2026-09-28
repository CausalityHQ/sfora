#!/usr/bin/env python3
"""Bounded CPU-only native weight-delta rank diagnostic, no model forward."""

import argparse
import hashlib
import json
import re
import signal
import time
from pathlib import Path

import torch
from safetensors import safe_open


def captured_energy(matrix, rank=32):
    singular = torch.linalg.svdvals(matrix.float()).double()
    energy = singular.square()
    if not bool(torch.isfinite(energy).all()) or float(energy.sum()) <= 0:
        raise ValueError("native delta spectrum undefined")
    return float(energy[:rank].sum() / energy.sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in ("checkpoint", "pretrained", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("native rank output already exists")
    started = time.perf_counter()
    result = {
        "schema": "sfora-native-delta-rank32-v1",
        "rank": 32,
        "claim_eligible": False,
        "matrices": [],
        "images_or_features_read": 0,
        "encoder_training": False,
        "serving_latency_measured": False,
    }

    def deadline(_signal, _frame):
        raise TimeoutError("120-second native rank CPU budget exceeded")

    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(120)
    try:
        torch.set_num_threads(8)
        assert captured_energy(torch.eye(64)) == 0.5
        assert captured_energy(torch.diag(torch.cat([torch.ones(16), torch.zeros(48)]))) == 1.0
        expected = (
            "ad58838e2492cd664a447308317b11a53f362a4c5a36a99f28978f32a3b71089",
            "fa34f822f016dbb167d8d0e3a8af99b5e199aa28573360d4295252b9ec418e2a",
        )
        for path, wanted in zip((args.checkpoint, args.pretrained), expected, strict=True):
            with path.open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == wanted
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)[
            "vision"
        ]
        pattern = re.compile(
            r"encoder\.layers\.(1[2-9]|2[0-3])\.(self_attn\.(q|k|v|out)_proj|mlp\.fc[12])\.weight"
        )
        names = sorted(key for key in state if pattern.fullmatch(key))
        if len(names) != 72:
            raise ValueError(f"native rank tensor inventory differs: {len(names)}")
        result.update(
            {
                "checkpoint_sha256": expected[0],
                "pretrained_sha256": expected[1],
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "planned_matrix_count": len(names),
                "minimum_required_capture": 0.8,
            }
        )
        with safe_open(args.pretrained, framework="pt", device="cpu") as source:
            for key in names:
                trained = state[key]
                original = source.get_tensor("vision_model." + key)
                if trained.shape != original.shape or trained.dtype != torch.float32:
                    raise ValueError("native rank matrix geometry differs")
                delta = trained.float() - original.float()
                fraction = captured_energy(delta)
                record = {
                    "name": key,
                    "shape": list(delta.shape),
                    "captured_frobenius_energy_rank32": fraction,
                    "native_parameters": delta.numel(),
                    "adapter_parameters_rank32": 32 * sum(delta.shape),
                    "delta_frobenius_norm": float(delta.double().norm()),
                    "relative_delta_norm": float(delta.double().norm() / original.double().norm()),
                }
                result["matrices"].append(record)
                print(json.dumps(record), flush=True)
                if fraction < 0.8:
                    result.update(
                        {
                            "decision": "KILL_NATIVE_UPDATE_APPROXIMATION_RANK32",
                            "failed_matrix": key,
                            "unrun_matrices": len(names) - len(result["matrices"]),
                        }
                    )
                    break
            else:
                result.update({"decision": "GO_ADAPTER_CAPACITY_SMOKE_DESIGN", "unrun_matrices": 0})
    except TimeoutError as error:
        result.update({"decision": "KILL_CPU_FEASIBILITY", "error": str(error)})
    except Exception as error:
        result.update({"decision": "KILL_INSTRUMENT", "error": f"{type(error).__name__}: {error}"})
        raise
    finally:
        signal.alarm(0)
        result["cpu_wall_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "matrices"}), flush=True)


if __name__ == "__main__":
    main()
