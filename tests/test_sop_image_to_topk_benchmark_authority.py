"""Keep the paired image benchmark on the accepted scorer and frozen screen."""

import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import ToTensor

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


def test_decode_encode_separates_host_preprocess_from_device_transfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    script = ROOT / "scripts/benchmark_sop_image_to_topk_pair.py"
    spec = importlib.util.spec_from_file_location("benchmark_sop_image_to_topk_pair", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(script.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    path = tmp_path / "query.png"
    Image.new("RGB", (2, 2), color=(255, 0, 0)).save(path)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, non_blocking=False: self)
    arm = {"transform": ToTensor(), "model": torch.nn.Flatten()}

    features, started, host_ready, device_ready, encoded = module._decode_encode(arm, (path,))

    assert features.shape == (1, 12)
    assert started < host_ready <= device_ready <= encoded

    class FixedModel(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.nn.functional.normalize(
                torch.ones((images.shape[0], 128), dtype=images.dtype), dim=1
            )

    class IdentityHead:
        def transform(self, values: torch.Tensor) -> torch.Tensor:
            return values

    class FixedGallery:
        def search(self, codes: object, inverse_norms: object) -> tuple[np.ndarray, np.ndarray]:
            del codes, inverse_norms
            return np.array([[0]], dtype=np.int32), np.array([[1.0]], dtype=np.float32)

    arm.update({"name": "oml", "model": FixedModel(), "head": IdentityHead()})
    durations, _ = module._call(arm, (path,), FixedGallery())
    assert durations["host_decode_preprocess_ns"] > 0
    assert durations["host_to_device_ns"] >= 0
    assert durations["decode_preprocess_ns"] == (
        durations["host_decode_preprocess_ns"] + durations["host_to_device_ns"]
    )
    assert (
        sum(
            durations[name]
            for name in (
                "host_decode_preprocess_ns",
                "host_to_device_ns",
                "encoder_transfer_ns",
                "project_pack_ns",
                "native_search_ns",
            )
        )
        == durations["image_to_topk_ns"]
    )
