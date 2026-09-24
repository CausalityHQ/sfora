"""Authority checks for the selected trained-encoder timing replay."""

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _subject():
    script = ROOT / "scripts/benchmark_sop_trained_image_to_topk_pair.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_sop_trained_image_to_topk_pair", script
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


def test_selected_training_matches_official_receipt() -> None:
    subject = _subject()
    digest = "a" * 64
    official = {
        "schema": "sfora-sop-reference-official-test-v1",
        "seed": 179019,
        "selection_step": 16000,
        "embedding_width": 128,
        "inputs": {"selected_checkpoint_sha256": digest},
    }
    trained = {
        "arm": "arcface",
        "recipe": "reference",
        "seed": 179019,
        "updates": 16000,
        "embedding_width": 128,
    }
    subject.validate_selected_training(official, trained, digest)
    for bad_digest, bad_trained, bad_official in (
        ("b" * 64, trained, official),
        (digest, {**trained, "updates": 8000}, official),
        (digest, {**trained, "seed": 1}, official),
        (digest, {**trained, "arm": "packed_rank"}, official),
        (digest, trained, {**official, "embedding_width": 768}),
    ):
        with pytest.raises(ValueError, match="selected training authority differs"):
            subject.validate_selected_training(bad_official, bad_trained, bad_digest)


def test_train_ids_and_output_are_frozen(tmp_path: Path) -> None:
    subject = _subject()
    subject.validate_train_ids([10, 11, 12], [10, 11, 12], expected_rows=3)
    with pytest.raises(ValueError, match="SOP train row identities differ"):
        subject.validate_train_ids([10, 11, 12], [10, 12, 11], expected_rows=3)
    with pytest.raises(ValueError, match="SOP train row identities differ"):
        subject.validate_train_ids([10, 11], [10, 11], expected_rows=3)

    output = tmp_path / "result.json"
    subject.validate_output_absent(output)
    output.write_text("existing")
    with pytest.raises(ValueError, match="SOP trained timing output already exists"):
        subject.validate_output_absent(output)
    with pytest.raises(ValueError, match="output directory differs"):
        subject.validate_output_absent(Path("relative-result.json"))


def test_invocation_is_reserved_once(tmp_path: Path) -> None:
    subject = _subject()
    output = tmp_path / "result.json"
    marker = subject.reserve_invocation(output)
    assert marker.is_dir()
    with pytest.raises(ValueError, match="invocation already reserved"):
        subject.reserve_invocation(output)


def test_official_receipt_is_bound_to_claim_and_expected_digest(tmp_path: Path) -> None:
    import hashlib
    import json

    subject = _subject()
    claim = tmp_path / "claim.json"
    claim.write_text('{"claimed":true}')
    output = tmp_path / "official.json"
    official = {
        "schema": "sfora-sop-reference-official-test-v1",
        "claim_eligible": False,
        "test_images": 60_502,
        "gallery_bytes_per_item": 130,
        "packed": {"recall_at_1": 0.87, "map_at_r": 0.7},
        "inputs": {
            "evaluator_sha256": subject.EVALUATOR_SHA256,
            "official_test_claim_path": str(claim),
            "official_test_claim_sha256": hashlib.sha256(claim.read_bytes()).hexdigest(),
            "test_image_manifest_sha256": subject.TEST_IMAGE_MANIFEST_SHA256,
        },
    }
    output.write_text(json.dumps(official))
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    subject.validate_official_quality(output, official, digest)
    with pytest.raises(ValueError, match="official quality authority differs"):
        subject.validate_official_quality(output, official, "b" * 64)
    claim.write_text('{"claimed":false}')
    with pytest.raises(ValueError, match="official quality authority differs"):
        subject.validate_official_quality(output, official, digest)


def test_gpu_process_inventory_fails_closed() -> None:
    subject = _subject()
    assert subject.parse_gpu_pids("") == set()
    assert subject.parse_gpu_pids("123\n456\n") == {123, 456}
    with pytest.raises(ValueError, match="GPU process inventory differs"):
        subject.parse_gpu_pids("[N/A]\n")


def test_selected_train_holdout_parity_rejects_wrong_score() -> None:
    subject = _subject()
    expected = {"packed_map_at_r": 0.74, "packed_recall_at_1": 0.92}
    measured = {"map_at_r": 0.7401, "recall_at_1": 0.9199}
    subject.validate_train_holdout_parity(measured, expected)
    with pytest.raises(ValueError, match="train holdout parity differs"):
        subject.validate_train_holdout_parity({**measured, "map_at_r": 0.70}, expected)
