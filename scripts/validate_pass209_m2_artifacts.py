#!/usr/bin/env python3
"""Validate the sealed Pass209 M2 receipt/error-manifest pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

_CLASSES = list(range(82, 98))
_CLASS_NAMES = [
    {"id": 82, "name": "Dodge Caliber Wagon 2012"},
    {"id": 83, "name": "Dodge Caliber Wagon 2007"},
    {"id": 84, "name": "Dodge Caravan Minivan 1997"},
    {"id": 85, "name": "Dodge Ram Pickup 3500 Crew Cab 2010"},
    {"id": 86, "name": "Dodge Ram Pickup 3500 Quad Cab 2009"},
    {"id": 87, "name": "Dodge Sprinter Cargo Van 2009"},
    {"id": 88, "name": "Dodge Journey SUV 2012"},
    {"id": 89, "name": "Dodge Dakota Crew Cab 2010"},
    {"id": 90, "name": "Dodge Dakota Club Cab 2007"},
    {"id": 91, "name": "Dodge Magnum Wagon 2008"},
    {"id": 92, "name": "Dodge Challenger SRT8 2011"},
    {"id": 93, "name": "Dodge Durango SUV 2012"},
    {"id": 94, "name": "Dodge Durango SUV 2007"},
    {"id": 95, "name": "Dodge Charger Sedan 2012"},
    {"id": 96, "name": "Dodge Charger SRT-8 2009"},
    {"id": 97, "name": "Eagle Talon Hatchback 1998"},
]
_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
_MODEL_NAME = "google/siglip-so400m-patch14-384"
_MODEL_REVISION = "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"
_RECEIPT_KEYS = {
    "schema",
    "claim_eligible",
    "source_revision",
    "source_tree_digest",
    "dataset",
    "dataset_revision",
    "dataset_examples_sha256",
    "split",
    "holdout_classes",
    "cell",
    "model_name",
    "model_revision",
    "readout",
    "compute_dtype",
    "processor_image_shape",
    "descriptors_validated",
    "norm_tolerance",
    "metrics",
    "gates",
    "passed",
    "batch_size",
    "query_block",
    "descriptor_shape",
    "descriptor_sha256",
    "error_manifest_sha256",
}
_MANIFEST_KEYS = {
    "schema",
    "claim_eligible",
    "source_revision",
    "source_tree_digest",
    "dataset",
    "dataset_revision",
    "dataset_examples_sha256",
    "descriptor_sha256",
    "batch_size",
    "query_block",
    "split",
    "holdout_classes",
    "class_names",
    "cell",
    "model_name",
    "model_revision",
    "error_count",
    "errors",
}
_ERROR_KEYS = {
    "query_position",
    "query_example_id",
    "query_label",
    "nearest_position",
    "nearest_example_id",
    "nearest_label",
}


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _read_canonical(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} is not valid JSON") from error
    if not isinstance(value, dict) or raw != _canonical(value):
        raise ValueError(f"{path.name} is not canonical JSON")
    return raw, value


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def validate_pass209_m2_artifacts(
    receipt_path: Path, manifest_path: Path
) -> dict[str, int | str]:
    """Authenticate the exact M2 pair raters are allowed to inspect."""

    receipt_bytes, receipt = _read_canonical(receipt_path)
    manifest_bytes, manifest = _read_canonical(manifest_path)
    del receipt_bytes
    if set(receipt) != _RECEIPT_KEYS or set(manifest) != _MANIFEST_KEYS:
        raise ValueError("Pass209 M2 artifact schema differs")
    common = {
        "claim_eligible": False,
        "dataset": "cars",
        "dataset_revision": _DATASET_REVISION,
        "split": "train",
        "holdout_classes": _CLASSES,
        "cell": "siglip-so400m",
        "model_name": _MODEL_NAME,
        "model_revision": _MODEL_REVISION,
    }
    for key, expected in common.items():
        if receipt[key] != expected or manifest[key] != expected:
            raise ValueError(f"Pass209 M2 {key} authority differs")
    for field, length in (
        ("source_revision", 40),
        ("source_tree_digest", 64),
        ("dataset_examples_sha256", 64),
        ("descriptor_sha256", 64),
    ):
        if not _is_hex(receipt[field], length) or receipt[field] != manifest[field]:
            raise ValueError(f"Pass209 M2 {field} binding differs")
    if (
        receipt["schema"] != "sfora-frozen-substrate-screen-v2"
        or manifest["schema"] != "sfora-frozen-substrate-errors-v1"
        or receipt["readout"] != "vision_pooler_output"
        or receipt["compute_dtype"] != "float32"
        or receipt["processor_image_shape"] != [384, 384]
        or receipt["descriptors_validated"] is not True
        or receipt["norm_tolerance"] != 1.0e-6
        or receipt["batch_size"] != 8
        or manifest["batch_size"] != 8
        or receipt["query_block"] != 32
        or manifest["query_block"] != 32
        or receipt["descriptor_shape"] != [1345, 1152]
        or manifest["class_names"] != _CLASS_NAMES
    ):
        raise ValueError("Pass209 M2 execution authority differs")
    metrics = receipt["metrics"]
    if not isinstance(metrics, dict) or set(metrics) != {"correct", "queries", "recall_at_1"}:
        raise ValueError("Pass209 M2 metric schema differs")
    if (
        not _is_int(metrics["correct"])
        or not _is_int(metrics["queries"])
        or metrics["correct"] != 1242
        or metrics["queries"] != 1345
        or metrics["recall_at_1"] != 1242 / 1345
        or receipt["gates"]
        != {"expected_queries": 1345, "recall_at_1_minimum": 0.94}
        or receipt["passed"] is not False
    ):
        raise ValueError("Pass209 M2 metric authority differs")
    if (
        not _is_hex(receipt["error_manifest_sha256"], 64)
        or hashlib.sha256(manifest_bytes).hexdigest()
        != receipt["error_manifest_sha256"]
    ):
        raise ValueError("Pass209 M2 manifest digest binding differs")
    rows = manifest["errors"]
    if (
        not _is_int(manifest["error_count"])
        or manifest["error_count"] != 103
        or not isinstance(rows, list)
        or len(rows) != 103
    ):
        raise ValueError("Pass209 M2 error cardinality differs")
    previous = -1
    for row in rows:
        if not isinstance(row, dict) or set(row) != _ERROR_KEYS:
            raise ValueError("Pass209 M2 error row schema differs")
        integer_fields = (
            "query_position",
            "query_label",
            "nearest_position",
            "nearest_label",
        )
        if any(not _is_int(row[field]) for field in integer_fields):
            raise ValueError("Pass209 M2 error row types differ")
        if (
            row["query_position"] <= previous
            or not 0 <= row["query_position"] < 1345
            or not 0 <= row["nearest_position"] < 1345
            or row["query_label"] not in _CLASSES
            or row["nearest_label"] not in _CLASSES
            or not isinstance(row["query_example_id"], str)
            or not row["query_example_id"]
            or not isinstance(row["nearest_example_id"], str)
            or not row["nearest_example_id"]
        ):
            raise ValueError("Pass209 M2 error row authority differs")
        previous = row["query_position"]
    return {
        "cell": "siglip-so400m",
        "correct": 1242,
        "error_count": 103,
        "queries": 1345,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--error-manifest", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_pass209_m2_artifacts(args.receipt, args.error_manifest),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
