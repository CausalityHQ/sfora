from pathlib import Path


def test_f0_launcher_freezes_train_only_protocol_and_authority() -> None:
    text = Path("scripts/run_siglip_token_set_f0_v49.sh").read_text()

    assert "SOURCE_REVISION" in text
    assert "SOURCE_MANIFEST" in text
    assert "UV_PROJECT_ENVIRONMENT" in text
    assert "must be outside the deployed tree" in text
    assert '"SOURCE_REVISION"' in text
    assert "manifest-bound revision" in text
    assert "unmanifested or missing files" in text
    assert "/usr/bin/python3 -I -S" in text
    assert '\npython - "$source_manifest"' not in text
    assert "source_tree_digest" in text
    assert "sha256sum --check --strict" in text
    assert "git rev-parse" not in text
    assert "git status" not in text
    assert "src/sfora/token_set_proxy_anchor.py" in text
    assert "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed" in text
    assert "--top-k 32" in text
    assert "--set-weight 0.25" in text
    assert 'payload["split"] == "train"' in text
    assert 'payload["holdout_classes"] == list(range(82, 98))' in text
    assert '"test_split_reads" not in payload' in text
    assert 'split="test"' not in text
    assert "--test" not in text
