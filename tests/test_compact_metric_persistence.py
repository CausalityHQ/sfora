from pathlib import Path

import pytest
import torch

from sfora import CompactMetricEncoder


def test_compact_metric_encoder_save_load_is_exact_and_rejects_corruption(tmp_path: Path) -> None:
    """Catch deployment persistence changing codes or accepting unauthenticated bytes."""

    generator = torch.Generator().manual_seed(23)
    embeddings = torch.nn.functional.normalize(
        torch.randn(8, 6, generator=generator), dim=1
    ).contiguous()
    encoder = CompactMetricEncoder(
        weight=torch.arange(18, dtype=torch.float32).reshape(3, 6).contiguous() / 19.0,
        bias=torch.tensor([0.25, -0.5, 0.75], dtype=torch.float32),
    )
    artifact = tmp_path / "compact-metric.sfora"

    encoder.save(artifact)
    restored = CompactMetricEncoder.load(artifact)

    assert restored.sha256 == encoder.sha256
    torch.testing.assert_close(restored.weight, encoder.weight, atol=0, rtol=0)
    torch.testing.assert_close(restored.bias, encoder.bias, atol=0, rtol=0)
    torch.testing.assert_close(
        restored.encode(embeddings), encoder.encode(embeddings), atol=0, rtol=0
    )

    corrupted = bytearray(artifact.read_bytes())
    corrupted[-1] ^= 1
    artifact.write_bytes(corrupted)
    with pytest.raises(ValueError, match="compact metric artifact differs"):
        CompactMetricEncoder.load(artifact)
