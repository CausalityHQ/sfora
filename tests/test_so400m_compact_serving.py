"""CPU checks for sfora.so400m_compact_serving with a tiny random SigLIP vision tower."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F

from sfora.joint_relational_compaction import pack_int8_unit_embeddings
from sfora.so400m_compact_serving import So400mCompactEncoder, So400mCompactIndex

# Same content as the So400m snapshot preprocessor_config.json; size is overwritten per resolution.
_PREPROCESSOR = {
    "do_normalize": True, "do_rescale": True, "do_resize": True, "image_mean": [0.5, 0.5, 0.5],
    "image_processor_type": "SiglipImageProcessor", "image_std": [0.5, 0.5, 0.5],
    "processor_class": "SiglipProcessor", "resample": 2, "rescale_factor": 0.00392156862745098,
    "size": {"height": 384, "width": 384},
}


def _encoder(resolution: int, output_dim: int = 128) -> So400mCompactEncoder:
    from transformers import AutoImageProcessor
    from transformers.models.siglip.modeling_siglip import SiglipVisionModel
    from transformers.models.siglip.configuration_siglip import SiglipVisionConfig

    torch.manual_seed(0)
    config = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                num_attention_heads=2, image_size=384, patch_size=14)
    vision = SiglipVisionModel(config).float()  # what SiglipModel.vision_model is in transformers 5.12
    head = nn.Linear(32, output_dim)
    with tempfile.TemporaryDirectory() as root:
        (Path(root) / "preprocessor_config.json").write_text(json.dumps(_PREPROCESSOR))
        processor = AutoImageProcessor.from_pretrained(root, local_files_only=True, backend="torchvision")
    processor.size = {"height": resolution, "width": resolution}
    return So400mCompactEncoder(processor, vision, head, resolution=resolution, device=torch.device("cpu"))


def _images(count: int, seed: int = 1) -> list[Image.Image]:
    rng = np.random.default_rng(seed)
    return [
        Image.fromarray(rng.integers(0, 256, size=(int(rng.integers(40, 90)), int(rng.integers(40, 90)), 3),
                                     dtype=np.uint8))
        for _ in range(count)
    ]


def test_encode_order_and_batching_invariance() -> None:
    encoder = _encoder(280)
    images = _images(70)
    whole = encoder.encode_images(images)  # 64 + 6
    assert whole.dtype == torch.float32 and whole.shape == (70, 128)
    assert torch.allclose(torch.linalg.vector_norm(whole, dim=1), torch.ones(70), atol=1e-5)
    pieces = torch.cat([encoder.encode_images(images[:64]), encoder.encode_images(images[64:])])
    assert torch.equal(whole, pieces)
    reversed_rows = encoder.encode_images(images[64:][::-1])
    assert torch.equal(reversed_rows, whole[64:].flip(0))
    packed = encoder.encode_packed(images)
    reference = pack_int8_unit_embeddings(whole)
    assert torch.equal(packed.codes, reference.codes)
    assert torch.equal(packed.inverse_norms, reference.inverse_norms)


def test_reference_math_and_interpolation() -> None:
    encoder = _encoder(280)
    assert encoder.interpolate is True and _encoder(384).interpolate is False
    images = _images(3)
    pixels = encoder.processor(images=images, return_tensors="pt")["pixel_values"]
    assert pixels.shape == (3, 3, 280, 280)
    with torch.inference_mode():
        with pytest.raises(Exception):  # 400 patches vs 729 positions: needs interpolation
            encoder.vision(pixel_values=pixels)
        with torch.autocast(device_type="cpu", dtype=torch.float16):
            pooled = encoder.vision(pixel_values=pixels, interpolate_pos_encoding=True).pooler_output
        with torch.autocast(device_type="cpu", enabled=False):
            expected = F.normalize(encoder.head(F.normalize(pooled.float(), dim=1)), dim=1)
    assert torch.equal(encoder.encode_images(images), expected)


def _brute(queries, gallery, k, exclude):
    q = queries.codes.numpy().astype(np.float64)
    g = gallery.codes.numpy().astype(np.float64)
    scores = (q @ g.T).astype(np.float32) * queries.inverse_norms.numpy().astype(np.float32)[:, None]
    scores = scores * gallery.inverse_norms.numpy().astype(np.float32)[None, :]
    if exclude is not None:
        scores[np.arange(len(q)), exclude] = -np.inf
    order = np.argsort(-scores, axis=1, kind="stable")[:, :k]
    return order, np.take_along_axis(scores, order, axis=1)


def test_search_matches_brute_force_with_ties_and_self_exclusion() -> None:
    rng = np.random.default_rng(3)
    base = F.normalize(torch.from_numpy(rng.standard_normal((150, 16)).astype(np.float32)), dim=1)
    values = torch.cat([base, base[:40]])  # duplicates -> exact score ties
    gallery = pack_int8_unit_embeddings(values)
    index = So400mCompactIndex(gallery, device=torch.device("cpu"))
    exclude = np.arange(len(values))
    for k, excl in ((1, exclude), (7, exclude), (len(values), None)):
        ordinals, scores = index.search(gallery, k, exclude=excl)
        want_ord, want_scores = _brute(gallery, gallery, k, excl)
        assert np.array_equal(ordinals, want_ord)
        assert np.array_equal(scores, want_scores)
    ordinals, _ = index.search(gallery, 2, exclude=None)
    assert np.array_equal(ordinals[150:, :2], np.stack([np.arange(40), np.arange(150, 190)], axis=1))
    ordinals, _ = index.search(gallery, 1, exclude=exclude)
    assert np.array_equal(ordinals[150:, 0], np.arange(40)) and np.array_equal(ordinals[:40, 0], np.arange(150, 190))
    with pytest.raises(ValueError):
        index.search(gallery, 0)
    with pytest.raises(ValueError):
        index.search(gallery, 1, exclude=exclude[:-1])


def test_bytes_round_trip() -> None:
    values = F.normalize(torch.randn(33, 128, generator=torch.Generator().manual_seed(4)), dim=1)
    index = So400mCompactIndex(pack_int8_unit_embeddings(values), device=torch.device("cpu"))
    wire = index.to_bytes()
    restored = So400mCompactIndex.from_bytes(wire, device=torch.device("cpu"))
    assert torch.equal(restored.gallery.codes, index.gallery.codes)
    assert torch.equal(restored.gallery.inverse_norms, index.gallery.inverse_norms)
    assert restored.to_bytes() == wire
    with pytest.raises(ValueError):
        So400mCompactIndex.from_bytes(wire[:-1] + bytes([wire[-1] ^ 1]))


def test_bad_output_dim_and_resolution_rejected() -> None:
    with pytest.raises(ValueError):
        _encoder(336)
    with pytest.raises(ValueError):
        _encoder(280, output_dim=256)
    for kwargs in ({"resolution": 336, "output_dim": 512}, {"resolution": 384, "output_dim": 768}):
        with pytest.raises(ValueError):
            So400mCompactEncoder.from_checkpoint(Path("/nonexistent"), Path("/nonexistent"),
                                                 device=torch.device("cpu"), **kwargs)
    with tempfile.TemporaryDirectory() as root:
        checkpoint = Path(root) / "checkpoint.pt"
        torch.save({"vision": {}, "head": {"weight": torch.zeros(512, 1152), "bias": torch.zeros(512)}}, checkpoint)
        with pytest.raises(ValueError, match="head differs"):
            So400mCompactEncoder.from_checkpoint(Path(root), checkpoint, resolution=384, output_dim=128,
                                                 device=torch.device("cpu"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
