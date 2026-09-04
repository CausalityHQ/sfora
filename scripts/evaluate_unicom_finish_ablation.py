#!/usr/bin/env python3
"""Evaluate the matched UniCOM finish A/B/C causal panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.unicom_finish_evidence import validate_finish_evidence
from sfora.unicom_finish_protocol import FinishArm, validate_finish_config

PARENT_RECALL_AT_1 = 0.986198243412798
PARENT_RECALL_AT_10 = 0.9974905897114178


def paired_contrast(
    control_ap: Sequence[float],
    candidate_ap: Sequence[float],
    labels: Sequence[str],
    *,
    control_top1: Sequence[bool],
    candidate_top1: Sequence[bool],
    bootstrap_samples: int = 10_000,
) -> dict[str, object]:
    """Compute one paired contrast with identity-clustered uncertainty."""

    size = len(control_ap)
    if (
        size == 0
        or len(candidate_ap) != size
        or len(labels) != size
        or len(control_top1) != size
        or len(candidate_top1) != size
        or type(bootstrap_samples) is not int
        or bootstrap_samples <= 0
    ):
        raise ValueError("finish paired evidence inventory differs")
    if any(type(label) is not str or not label for label in labels):
        raise ValueError("finish paired identity differs")
    left = np.asarray(control_ap, dtype=np.float64)
    right = np.asarray(candidate_ap, dtype=np.float64)
    if (
        not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or np.any(left < 0.0)
        or np.any(left > 1.0)
        or np.any(right < 0.0)
        or np.any(right > 1.0)
        or any(type(value) is not bool for value in control_top1)
        or any(type(value) is not bool for value in candidate_top1)
    ):
        raise ValueError("finish paired evidence differs")
    delta = right - left
    grouped: dict[str, list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        grouped[label].append(index)
    identities = tuple(sorted(grouped))
    generator = np.random.Generator(np.random.PCG64(20260904))
    replicates = np.empty(bootstrap_samples, dtype=np.float64)
    for iteration in range(bootstrap_samples):
        sampled = generator.integers(0, len(identities), size=len(identities))
        values = np.concatenate(
            [delta[grouped[identities[int(index)]]] for index in sampled]
        )
        replicates[iteration] = float(np.mean(values))
    lower, upper = np.quantile(replicates, [0.025, 0.975])
    wins = int(np.count_nonzero(delta > 0.0))
    losses = int(np.count_nonzero(delta < 0.0))
    control_top = np.asarray(control_top1, dtype=np.bool_)
    candidate_top = np.asarray(candidate_top1, dtype=np.bool_)
    return {
        "delta_map_at_r": float(np.mean(delta)),
        "identity_cluster_bootstrap_seed": 20260904,
        "identity_cluster_bootstrap_samples": bootstrap_samples,
        "identity_cluster_bootstrap_95": [float(lower), float(upper)],
        "win_tie_loss": {"win": wins, "tie": size - wins - losses, "loss": losses},
        "top1_discordant": {
            "candidate_only_correct": int(np.count_nonzero(candidate_top & ~control_top)),
            "control_only_correct": int(np.count_nonzero(control_top & ~candidate_top)),
        },
    }


def classify_causal_panel(
    *,
    c_minus_a: float,
    c_minus_b: float,
    c_minus_a_recall1: float,
    c_minus_b_recall1: float,
    c_minus_a_recall10: float,
    c_minus_b_recall10: float,
    c_minus_parent_recall1: float,
    c_minus_parent_recall10: float,
) -> str:
    """Apply the preregistered causal gain and recall noninferiority gates."""

    values = (
        c_minus_a,
        c_minus_b,
        c_minus_a_recall1,
        c_minus_b_recall1,
        c_minus_a_recall10,
        c_minus_b_recall10,
        c_minus_parent_recall1,
        c_minus_parent_recall10,
    )
    if any(type(value) is not float or not math.isfinite(value) for value in values):
        raise ValueError("finish causal contrast differs")
    return (
        "GO"
        if c_minus_a >= 0.003
        and c_minus_b >= 0.003
        and min(
            c_minus_a_recall1,
            c_minus_b_recall1,
            c_minus_a_recall10,
            c_minus_b_recall10,
            c_minus_parent_recall1,
            c_minus_parent_recall10,
        )
        >= -0.001
        else "CLOSE"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_arm(
    *,
    path: Path,
    expected_sha256: str,
    evidence_root: Path,
    expected_arm: FinishArm,
    source_commit: str,
    config: dict[str, object],
    config_sha256: str,
) -> tuple[
    dict[str, object], list[float], list[bool], list[str], list[str], list[dict[str, object]]
]:
    if (
        path.is_symlink()
        or not path.is_file()
        or _sha256_file(path) != expected_sha256
    ):
        raise ValueError("finish arm result bytes differ")
    payload = path.read_bytes()
    result = json.loads(payload)
    if payload != (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode():
        raise ValueError("finish arm result encoding differs")
    if (
        type(result) is not dict
        or result.get("schema") != "unicom-finish-ablation-result-v1"
        or result.get("claim_eligible") is not False
        or result.get("source_commit") != source_commit
        or result.get("arm") != expected_arm.value
        or result.get("finish_seed") != 3
        or result.get("config_sha256") != config_sha256
        or result.get("parent_checkpoint_sha256")
        != config["parent_checkpoint_sha256"]
        or result.get("parent_receipt_sha256") != config["parent_receipt_sha256"]
        or result.get("partition_sha256") != config["partition_sha256"]
        or result.get("schedule_sha256")
        != config["schedule_sha256"][expected_arm.value]
        or type(result.get("evidence")) is not dict
        or result["evidence"].get("arm") != expected_arm.value
        or result["evidence"].get("finish_seed") != 3
        or type(result.get("updates")) is not dict
        or result["updates"].get("attempted") != 644
        or result["updates"].get("successful") != 644
        or result["updates"].get("skipped") != 0
    ):
        raise ValueError("finish arm result authority differs")
    checkpoint = result.get("terminal_checkpoint")
    if (
        type(checkpoint) is not dict
        or set(checkpoint) != {"path", "sha256", "bytes"}
        or type(checkpoint["path"]) is not str
        or type(checkpoint["sha256"]) is not str
        or type(checkpoint["bytes"]) is not int
        or checkpoint["bytes"] <= 0
        or Path(checkpoint["path"]).is_symlink()
        or not Path(checkpoint["path"]).is_file()
        or Path(checkpoint["path"]).stat().st_size != checkpoint["bytes"]
        or _sha256_file(Path(checkpoint["path"])) != checkpoint["sha256"]
    ):
        raise ValueError("finish terminal checkpoint differs")
    validate_finish_evidence(
        result["evidence"],
        evidence_root,
        expected_schedule_sha256=result["schedule_sha256"],
    )
    rows = json.loads(
        (evidence_root / "evaluation-epoch-0008-ranked-prefix.json").read_bytes()
    )
    records = result["evidence"]
    receipt = json.loads(
        (evidence_root / records["evaluation_receipt"]["path"]).read_bytes()
    )
    if type(rows) is not list or len(rows) != len(receipt["query_records"]):
        raise ValueError("finish paired row inventory differs")
    paths = [row["query_path"] for row in rows]
    labels = [row["query_label"] for row in rows]
    if paths != [row["image_name"] for row in receipt["query_records"]]:
        raise ValueError("finish paired query order differs")
    ap = [row["ap_at_r"] for row in rows]
    top1 = [bool(row["ranked_prefix"][0]["correct"]) for row in rows]
    return result, ap, top1, labels, paths, receipt["gallery_records"]


def canonical_result_bytes(result: object) -> bytes:
    if (
        type(result) is not dict
        or result.get("schema") != "unicom-finish-causal-panel-v1"
        or result.get("claim_eligible") is not False
    ):
        raise ValueError("finish causal result differs")
    return (
        json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode()


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--config-sha256", required=True)
    for name in ("a", "b", "c"):
        parser.add_argument(f"--{name}-result", required=True, type=Path)
        parser.add_argument(f"--{name}-sha256", required=True)
        parser.add_argument(f"--{name}-evidence-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute-causal-evaluation", action="store_true", required=True)
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
    if (
        source != args.source_commit
        or dirty
        or args.config.is_symlink()
        or not args.config.is_file()
        or _sha256_file(args.config) != args.config_sha256
        or args.output.exists()
        or args.output.is_symlink()
    ):
        raise ValueError("finish causal authority differs")
    config = validate_finish_config(json.loads(args.config.read_bytes()))
    arms = {}
    for name, arm in zip(("a", "b", "c"), FinishArm, strict=True):
        arms[name] = _load_arm(
            path=getattr(args, f"{name}_result"),
            expected_sha256=getattr(args, f"{name}_sha256"),
            evidence_root=getattr(args, f"{name}_evidence_root"),
            expected_arm=arm,
            source_commit=source,
            config=config,
            config_sha256=args.config_sha256,
        )
    a_result, a_ap, a_top1, labels, paths, gallery = arms["a"]
    b_result, b_ap, b_top1, b_labels, b_paths, b_gallery = arms["b"]
    c_result, c_ap, c_top1, c_labels, c_paths, c_gallery = arms["c"]
    if (
        labels != b_labels
        or labels != c_labels
        or paths != b_paths
        or paths != c_paths
        or gallery != b_gallery
        or gallery != c_gallery
        or a_result["state_sha256"]["initial_model"]
        != b_result["state_sha256"]["initial_model"]
        or a_result["state_sha256"]["initial_model"]
        != c_result["state_sha256"]["initial_model"]
        or a_result["state_sha256"]["initial_classifier_optimizer"]
        != b_result["state_sha256"]["initial_classifier_optimizer"]
        or a_result["state_sha256"]["initial_classifier_optimizer"]
        != c_result["state_sha256"]["initial_classifier_optimizer"]
        or a_result["state_sha256"]["initial_classifier_optimizer"]
        == a_result["state_sha256"]["final_classifier_optimizer"]
        or b_result["state_sha256"]["initial_classifier_optimizer"]
        == b_result["state_sha256"]["final_classifier_optimizer"]
        or c_result["state_sha256"]["initial_classifier_optimizer"]
        != c_result["state_sha256"]["final_classifier_optimizer"]
        or any(
            result["updates"]["final_ema"] - result["updates"]["initial_ema"]
            != 644
            or result["updates"]["final_scheduler_step"]
            - result["updates"]["initial_scheduler_step"]
            != 644
            for result in (a_result, b_result, c_result)
        )
    ):
        raise ValueError("finish paired query identity differs")
    ca = paired_contrast(
        a_ap,
        c_ap,
        labels,
        control_top1=a_top1,
        candidate_top1=c_top1,
    )
    cb = paired_contrast(
        b_ap,
        c_ap,
        labels,
        control_top1=b_top1,
        candidate_top1=c_top1,
    )
    metrics = {name: arms[name][0]["evidence"]["metrics"] for name in arms}
    status = classify_causal_panel(
        c_minus_a=ca["delta_map_at_r"],
        c_minus_b=cb["delta_map_at_r"],
        c_minus_a_recall1=metrics["c"]["recall_at_1"] - metrics["a"]["recall_at_1"],
        c_minus_b_recall1=metrics["c"]["recall_at_1"] - metrics["b"]["recall_at_1"],
        c_minus_a_recall10=metrics["c"]["recall_at_10"] - metrics["a"]["recall_at_10"],
        c_minus_b_recall10=metrics["c"]["recall_at_10"] - metrics["b"]["recall_at_10"],
        c_minus_parent_recall1=metrics["c"]["recall_at_1"] - PARENT_RECALL_AT_1,
        c_minus_parent_recall10=metrics["c"]["recall_at_10"] - PARENT_RECALL_AT_10,
    )
    result = {
        "schema": "unicom-finish-causal-panel-v1",
        "claim_eligible": False,
        "source_commit": source,
        "config_sha256": args.config_sha256,
        "inputs": {
            name: {
                "sha256": getattr(args, f"{name}_sha256"),
                "path": str(getattr(args, f"{name}_result").resolve()),
            }
            for name in ("a", "b", "c")
        },
        "gates": {
            "delta_map_at_r_min": 0.003,
            "recall_at_1_delta_min": -0.001,
            "recall_at_10_delta_min": -0.001,
        },
        "metrics": metrics,
        "contrasts": {"c_minus_a": ca, "c_minus_b": cb},
        "status": status,
    }
    payload = canonical_result_bytes(result)
    publication = publish_bytes_noreplace(
        args.output,
        payload,
        validator=lambda observed: (
            None
            if observed == payload
            else (_ for _ in ()).throw(ValueError("finish causal bytes differ"))
        ),
    )
    publication.close()
    return result


def main(arguments: Sequence[str] | None = None) -> int:
    try:
        result = run(parse_args(arguments))
    except Exception as error:
        print(f"finish causal evaluation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
