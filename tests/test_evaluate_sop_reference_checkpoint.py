"""The official evaluator must load only a selected, matching checkpoint."""

from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from PIL import Image

from sfora.sop_reference_selection import ReferenceSelection

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_sop_reference_checkpoint.py"
SPEC = importlib.util.spec_from_file_location("evaluate_sop_reference_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
sys.path.insert(0, str(SCRIPT.parent))
SPEC.loader.exec_module(MODULE)
sys.path.pop(0)


def _selection(width: int = 128):
    return ReferenceSelection(
        checkpoint=Path("selected.pt"),
        receipt={
            "seed": 179019,
            "embedding_width": width,
            "schedule_sha256": "c" * 64,
        },
        step=8000,
        packed_map_at_r=0.8,
        packed_recall_at_1=0.9,
    )


def _payload(width: int = 128):
    return {
        "recipe": "reference",
        "arm": "arcface",
        "seed": 179019,
        "updates": 8000,
        "embedding_width": width,
        "schedule_sha256": "c" * 64,
        "model": {},
        "head": {
            "weight": torch.zeros(width, 768),
            "bias": torch.zeros(width),
        },
        "classifier": torch.zeros(10_186, width),
    }


@pytest.mark.parametrize("width", (128, 768))
def test_selected_payload_width_and_receipt_must_match(width: int):
    selection = _selection(width)
    assert MODULE.validate_selected_payload(_payload(width), selection) == width
    wrong = _payload(width)
    wrong["head"]["weight"] = torch.zeros(width - 1, 768)
    with pytest.raises(ValueError, match="selected checkpoint"):
        MODULE.validate_selected_payload(wrong, selection)
    wrong = _payload(width)
    wrong["seed"] = 1
    with pytest.raises(ValueError, match="selected checkpoint"):
        MODULE.validate_selected_payload(wrong, selection)


def test_b8_selected_payload_without_width_is_128():
    selection = _selection()
    selection.receipt.pop("embedding_width")
    payload = _payload()
    payload.pop("embedding_width")
    assert MODULE.validate_selected_payload(payload, selection) == 128


def test_evaluator_manifest_binds_loaded_scorer_and_selector():
    manifest = MODULE.evaluator_source_manifest()
    assert "src/sfora/sop_evaluation.py" in manifest
    assert "src/sfora/sop_reference_selection.py" in manifest
    assert "scripts/evaluate_sop_reference_checkpoint.py" in manifest
    assert all(len(value) == 64 for value in manifest.values())


def test_final_receipt_claim_cannot_be_reused_with_another_output(tmp_path: Path):
    final_receipt = tmp_path / "final.json"
    final_receipt.write_text('{"completed":true}\n')
    checkpoint_sha256 = "a" * 64
    claim = MODULE.claim_official_test_once(
        final_receipt, checkpoint_sha256, tmp_path, tmp_path / "first.json"
    )
    assert claim.is_file()
    copied_receipt = tmp_path / "copy.json"
    copied_receipt.write_text('{ "completed" : true }\n')
    with pytest.raises(ValueError, match="already claimed"):
        MODULE.claim_official_test_once(
            copied_receipt, checkpoint_sha256, tmp_path, tmp_path / "second.json"
        )


def test_verified_images_decode_the_same_bytes_that_match_manifest(tmp_path: Path):
    image_path = tmp_path / "product.png"
    Image.new("RGB", (4, 4), "red").save(image_path)
    manifest = hashlib.sha256(image_path.read_bytes()).digest()
    record = SimpleNamespace(image_path=image_path, label=7)
    dataset = MODULE.VerifiedImages((record,), lambda image: image.size, manifest)
    assert dataset[0] == ((4, 4), 7)
    Image.new("RGB", (4, 4), "blue").save(image_path)
    with pytest.raises(ValueError, match="image content"):
        dataset[0]
    with pytest.raises(ValueError, match="image manifest"):
        MODULE.VerifiedImages((record,), lambda image: image.size, b"short")


def test_image_manifest_requires_pinned_file_digest(tmp_path: Path):
    manifest_path = tmp_path / "test-images.bin"
    manifest_path.write_bytes(hashlib.sha256(b"one image").digest())
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert MODULE.load_test_image_manifest(manifest_path, digest, image_count=1) == (
        manifest_path.read_bytes()
    )
    with pytest.raises(ValueError, match="image manifest digest"):
        MODULE.load_test_image_manifest(manifest_path, "f" * 64, image_count=1)
