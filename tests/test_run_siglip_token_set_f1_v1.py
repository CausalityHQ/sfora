from pathlib import Path


def test_f1_launcher_reuses_revision_seal_and_freezes_determinism() -> None:
    text = Path("scripts/run_siglip_token_set_f1_v1.sh").read_text()

    assert "SOURCE_REVISION" in text
    assert "SOURCE_MANIFEST" in text
    assert "F0_RECEIPT" in text
    assert "F0_RECEIPT_SHA256" in text
    assert "PREFLIGHT_ONLY=1" in text
    assert "run_siglip_token_set_f0_v49.sh" in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    assert "--f0-receipt-sha256" in text
    assert "run_siglip_token_set_f1.py" in text
    assert 'payload["train_classes"] == list(range(49))' in text
    assert 'payload["validation_classes"] == list(range(49, 82))' in text
    assert 'payload["seeds"] == [17, 29, 43]' in text
    assert 'payload["claim_eligible"] is False' in text
