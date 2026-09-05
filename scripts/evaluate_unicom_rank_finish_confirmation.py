#!/usr/bin/env python3
"""Authenticate discovery seed 0 and evaluate seeds 1/2 confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from sfora.atomic_publication import publish_bytes_noreplace

BASELINE = {
    "map_at_r": 0.8975116742477199,
    "recall_at_1": 0.986198243412798,
    "recall_at_10": 0.9974905897114178,
}
SEED2_PRECEDENCE_DEFECT_COMMIT = "d833ec6ecd71738b45e3285607594f9774de001f"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(value: object, name: str) -> float:
    if type(value) is not float or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"rank-finish {name} differs")
    return value


def _seed_observation(result: object, expected_seed: int) -> dict[str, object]:
    if (
        type(result) is not dict
        or result.get("schema") != "unicom-rank-finish-screen-v1"
        or result.get("claim_eligible") is not False
        or type(result.get("source_commit")) is not str
        or len(result["source_commit"]) != 40
        or result.get("baseline") != BASELINE
        or type(result.get("history")) is not list
        or type(result.get("decision")) is not dict
    ):
        raise ValueError("rank-finish seed result differs")
    observed_seed = result.get("finish_seed", 0)
    if type(observed_seed) is not int or observed_seed != expected_seed:
        raise ValueError("rank-finish seed order differs")
    if expected_seed == 1:
        artifact = result.get("model_artifact")
        if (
            type(artifact) is not dict
            or set(artifact) != {"path", "sha256", "bytes"}
            or type(artifact["path"]) is not str
            or type(artifact["sha256"]) is not str
            or len(artifact["sha256"]) != 64
            or type(artifact["bytes"]) is not int
            or artifact["bytes"] <= 0
        ):
            raise ValueError("rank-finish model authority differs")
    elif expected_seed == 2 and result.get("model_artifact") is not None:
        artifact = result["model_artifact"]
        if (
            type(artifact) is not dict
            or set(artifact) != {"path", "sha256", "bytes"}
            or type(artifact["path"]) is not str
            or type(artifact["sha256"]) is not str
            or len(artifact["sha256"]) != 64
            or type(artifact["bytes"]) is not int
            or artifact["bytes"] <= 0
        ):
            raise ValueError("rank-finish model authority differs")
    rows = [row for row in result["history"] if type(row) is dict and row.get("epoch") == 8]
    if len(rows) != 1 or type(rows[0].get("metrics")) is not dict:
        raise ValueError("rank-finish epoch-8 evidence differs")
    metrics: Mapping[str, object] = rows[0]["metrics"]
    deltas = {
        key: _metric(metrics.get(key), key) - baseline
        for key, baseline in BASELINE.items()
    }
    registered = result["decision"].get("epoch8_deltas")
    registered_matches = (
        type(registered) is dict
        and tuple(registered) == tuple(BASELINE)
        and all(
            type(registered[key]) is float
            and math.isclose(registered[key], delta, rel_tol=0.0, abs_tol=1e-15)
            for key, delta in deltas.items()
        )
    )
    if not registered_matches:
        epoch6_rows = [
            row
            for row in result["history"]
            if type(row) is dict and row.get("epoch") == 6
        ]
        legacy = result["decision"]
        if (
            expected_seed != 2
            or result["source_commit"] != SEED2_PRECEDENCE_DEFECT_COMMIT
            or result.get("status") != "ABORT_EPOCH6"
            or len(epoch6_rows) != 1
            or type(epoch6_rows[0].get("metrics")) is not dict
            or type(legacy) is not dict
            or set(legacy) != {"status", "epoch6_delta_map"}
            or legacy["status"] != "ABORT_EPOCH6"
            or type(legacy["epoch6_delta_map"]) is not float
            or not math.isclose(
                legacy["epoch6_delta_map"],
                _metric(epoch6_rows[0]["metrics"].get("map_at_r"), "map_at_r")
                - BASELINE["map_at_r"],
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError("rank-finish seed decision differs")
    passes = (
        deltas["map_at_r"] >= 0.003
        and deltas["recall_at_1"] >= -0.001
        and deltas["recall_at_10"] >= -0.001
    )
    return {
        "finish_seed": expected_seed,
        "source_commit": result["source_commit"],
        "metrics": {key: metrics[key] for key in BASELINE},
        "deltas": deltas,
        "passes": passes,
        "model_artifact": result.get("model_artifact"),
    }


def evaluate_confirmation(results: Sequence[object]) -> dict[str, object]:
    """Recompute all per-seed gates and the seeds 1/2 confirmation mean."""

    if type(results) not in {list, tuple} or len(results) != 3:
        raise ValueError("rank-finish confirmation inventory differs")
    seeds = [
        _seed_observation(result, expected_seed)
        for expected_seed, result in enumerate(results)
    ]
    confirmation_mean_delta = math.fsum(
        row["deltas"]["map_at_r"] for row in seeds[1:]
    ) / 2
    status = (
        "CONFIRM"
        if all(row["passes"] for row in seeds) and confirmation_mean_delta >= 0.010
        else "REJECT"
    )
    return {
        "status": status,
        "confirmation_mean_delta_map_at_r": confirmation_mean_delta,
        "all_seed_quality_pass": all(row["passes"] for row in seeds),
        "seeds": seeds,
    }


def canonical_result_bytes(result: object) -> bytes:
    if (
        type(result) is not dict
        or result.get("schema") != "unicom-rank-finish-confirmation-v1"
        or result.get("claim_eligible") is not False
    ):
        raise ValueError("rank-finish confirmation result differs")
    return (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    for seed in range(3):
        parser.add_argument(f"--seed-{seed}-result", required=True, type=Path)
        parser.add_argument(f"--seed-{seed}-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-confirmation", action="store_true", required=True)
    return parser.parse_args(arguments)


def run(args: argparse.Namespace) -> dict[str, object]:
    repository = Path(__file__).resolve().parents[1]
    source = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if source != args.source_commit or dirty:
        raise ValueError("rank-finish confirmation source differs")
    results = []
    authorities = []
    for seed in range(3):
        path = getattr(args, f"seed_{seed}_result")
        expected = getattr(args, f"seed_{seed}_sha256")
        if (
            type(expected) is not str
            or len(expected) != 64
            or path.is_symlink()
            or not path.is_file()
            or _sha256_file(path) != expected
        ):
            raise ValueError("rank-finish confirmation input differs")
        results.append(json.loads(path.read_bytes()))
        authorities.append(
            {"finish_seed": seed, "path": str(path.resolve()), "sha256": expected}
        )
    decision = evaluate_confirmation(results)
    result = {
        "schema": "unicom-rank-finish-confirmation-v1",
        "claim_eligible": False,
        "source_commit": source,
        "inputs": authorities,
        "gates": {
            "per_seed_delta_map_at_r_min": 0.003,
            "per_seed_recall_at_1_delta_min": -0.001,
            "per_seed_recall_at_10_delta_min": -0.001,
            "confirmation_mean_delta_map_at_r_min": 0.010,
        },
        "decision": decision,
        "status": decision["status"],
    }
    payload = canonical_result_bytes(result)
    published = publish_bytes_noreplace(
        args.output,
        payload,
        validator=lambda observed: (
            None
            if observed == payload
            else (_ for _ in ()).throw(ValueError("confirmation bytes differ"))
        ),
    )
    published.close()
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        args = parse_args(arguments)
        result = run(args)
    except Exception as error:
        print(f"rank-finish confirmation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output": str(args.output), "status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
