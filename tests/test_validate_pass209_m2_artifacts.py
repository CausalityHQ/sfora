from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_pass209_m2_artifacts.py"
_SPEC = importlib.util.spec_from_file_location("validate_pass209_m2_artifacts", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _artifacts(tmp_path: Path) -> tuple[Path, Path, dict[str, Any], dict[str, Any]]:
    source_revision = "1" * 40
    tree = "2" * 64
    examples = "3" * 64
    descriptors = "4" * 64
    class_names = deepcopy(_MODULE._CLASS_NAMES)
    errors = [
        {
            "query_position": index,
            "query_example_id": f"cars/train/query-{index}.jpg",
            "query_label": 82 + index % 16,
            "nearest_position": 103 + index,
            "nearest_example_id": f"cars/train/nearest-{index}.jpg",
            "nearest_label": 82 + (index + 1) % 16,
        }
        for index in range(103)
    ]
    manifest: dict[str, Any] = {
        "schema": "sfora-frozen-substrate-errors-v1",
        "claim_eligible": False,
        "source_revision": source_revision,
        "source_tree_digest": tree,
        "dataset": "cars",
        "dataset_revision": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        "dataset_examples_sha256": examples,
        "descriptor_sha256": descriptors,
        "batch_size": 8,
        "query_block": 32,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "class_names": class_names,
        "cell": "siglip-so400m",
        "model_name": "google/siglip-so400m-patch14-384",
        "model_revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        "error_count": 103,
        "errors": errors,
    }
    manifest_bytes = _canonical(manifest)
    receipt: dict[str, Any] = {
        "schema": "sfora-frozen-substrate-screen-v2",
        "claim_eligible": False,
        "source_revision": source_revision,
        "source_tree_digest": tree,
        "dataset": "cars",
        "dataset_revision": manifest["dataset_revision"],
        "dataset_examples_sha256": examples,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "cell": "siglip-so400m",
        "model_name": manifest["model_name"],
        "model_revision": manifest["model_revision"],
        "readout": "vision_pooler_output",
        "compute_dtype": "float32",
        "processor_image_shape": [384, 384],
        "descriptors_validated": True,
        "norm_tolerance": 1.0e-6,
        "metrics": {"correct": 1242, "queries": 1345, "recall_at_1": 1242 / 1345},
        "gates": {"expected_queries": 1345, "recall_at_1_minimum": 0.94},
        "passed": False,
        "batch_size": 8,
        "query_block": 32,
        "descriptor_shape": [1345, 1152],
        "descriptor_sha256": descriptors,
        "error_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    receipt_path = tmp_path / "receipt.json"
    manifest_path = tmp_path / "errors.json"
    receipt_path.write_bytes(_canonical(receipt))
    manifest_path.write_bytes(manifest_bytes)
    return receipt_path, manifest_path, receipt, manifest


def test_validator_accepts_only_the_cross_bound_m2_pair(tmp_path: Path) -> None:
    receipt_path, manifest_path, _, _ = _artifacts(tmp_path)
    summary = _MODULE.validate_pass209_m2_artifacts(receipt_path, manifest_path)
    assert summary == {
        "cell": "siglip-so400m",
        "correct": 1242,
        "error_count": 103,
        "queries": 1345,
    }


def test_protocol_class_table_matches_the_executable_authority() -> None:
    protocol = (
        Path(__file__).parents[1]
        / "docs"
        / "pass209_error_taxonomy_protocol_2026-08-30.md"
    ).read_text()
    rows = re.findall(r"^\| (\d+) \| (.+) \|$", protocol, flags=re.MULTILINE)
    assert [
        {"id": int(label), "name": name}
        for label, name in rows
        if 82 <= int(label) <= 97
    ] == _MODULE._CLASS_NAMES


@pytest.mark.parametrize(
    ("role", "mutation"),
    [
        ("receipt", lambda value: value.update(source_tree_digest="5" * 64)),
        ("receipt", lambda value: value["metrics"].update(correct=1243)),
        ("manifest", lambda value: value.update(batch_size=True)),
        ("manifest", lambda value: value.update(error_count=102)),
        ("manifest", lambda value: value["class_names"][0].update(name="wrong")),
        ("manifest", lambda value: value["errors"][1].update(query_position=0)),
    ],
)
def test_validator_rejects_authority_and_cross_binding_drift(
    tmp_path: Path, role: str, mutation: Any
) -> None:
    receipt_path, manifest_path, receipt, manifest = _artifacts(tmp_path)
    mutated = deepcopy(receipt if role == "receipt" else manifest)
    mutation(mutated)
    if role == "receipt":
        receipt_path.write_bytes(_canonical(mutated))
    else:
        manifest_path.write_bytes(_canonical(mutated))
        receipt["error_manifest_sha256"] = hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest()
        receipt_path.write_bytes(_canonical(receipt))

    with pytest.raises(ValueError):
        _MODULE.validate_pass209_m2_artifacts(receipt_path, manifest_path)
