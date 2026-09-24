"""Keep the paired image benchmark on the accepted scorer and frozen screen."""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_benchmark_uses_released_native_library_and_pinned_quality_screen() -> None:
    source = ast.parse((ROOT / "scripts/benchmark_sop_image_to_topk_pair.py").read_text())
    constants = {
        target.id: node.value.value
        for node in source.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    decision = json.loads(
        (ROOT / "docs/evidence/packed_topk_rc5_pointer_decision_v1.json").read_text()
    )
    screen = ROOT / "docs/evidence/compact_metric/unicom-b16-sop-pretrained-screen-v2.json"

    assert decision["release_gate_passed"] is True
    assert constants["NATIVE_LIBRARY_SHA256"] == decision["candidate_library_sha256"]
    assert constants["NATIVE_LIBRARY_SHA256"] != decision["baseline_library_sha256"]
    assert constants["NATIVE_API_SHA256"] == decision["candidate_api_sha256"]
    assert (
        constants["NATIVE_API_SHA256"]
        == hashlib.sha256((ROOT / "src/sfora/cutile_int8.py").read_bytes()).hexdigest()
    )
    assert constants["QUALITY_SCREEN_SHA256"] == hashlib.sha256(screen.read_bytes()).hexdigest()
    assert (
        constants["TEST_IMAGE_MANIFEST_SHA256"]
        == hashlib.sha256(
            (ROOT / "docs/evidence/compact_metric/sop-test-image-sha256-v1.bin").read_bytes()
        ).hexdigest()
    )


def test_benchmark_rejects_changed_query_image_bytes(tmp_path: Path) -> None:
    script = ROOT / "scripts/benchmark_sop_image_to_topk_pair.py"
    spec = importlib.util.spec_from_file_location("benchmark_sop_image_to_topk_pair", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    paths = tuple(tmp_path / f"image-{index}.jpg" for index in range(32))
    for index, path in enumerate(paths):
        path.write_bytes(f"query image {index}".encode())
    manifest = b"".join(hashlib.sha256(path.read_bytes()).digest() for path in paths)
    manifest += b"\0" * ((60_502 - 32) * 32)
    module.verify_query_images(paths, manifest)

    paths[17].write_bytes(b"changed image")
    with pytest.raises(ValueError, match="query image content differs"):
        module.verify_query_images(paths, manifest)


def test_live_query_features_must_match_cached_gallery_source() -> None:
    script = ROOT / "scripts/benchmark_sop_image_to_topk_pair.py"
    spec = importlib.util.spec_from_file_location("benchmark_sop_image_to_topk_pair", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    cached = torch.eye(32, dtype=torch.float32)
    matching = cached * 2
    parity = module.verify_live_query_features(matching, cached, arm="oml")
    assert parity["min_cosine"] == pytest.approx(1.0)
    assert parity["queries"] == 32

    swapped = cached[[1, 0, *range(2, 32)]]
    with pytest.raises(ValueError, match="live query feature parity differs"):
        module.verify_live_query_features(swapped, cached, arm="oml")
    with pytest.raises(ValueError, match="live query feature parity differs"):
        module.verify_live_query_features(cached[:2], cached, arm="oml")
