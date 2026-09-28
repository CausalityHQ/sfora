import sys

import pytest
import train_inshop_siglip2_unseen_gallery as trainer


def test_centroid_100_cannot_run_without_positive_qualification(monkeypatch):
    argv = ["trainer"]
    for key in ("dataset-root", "model-snapshot", "features-dir", "preflight", "output-dir"):
        argv += [f"--{key}", "/missing"]
    argv += [
        "--arm",
        "freeze_emb",
        "--updates",
        "100",
        "--seed",
        "179024",
        "--centroid-pca-smoke",
        "products",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(ValueError, match="centroid PCA smoke authority differs"):
        trainer.main()


def test_centroid_100_rejects_unqualified_receipt_before_gpu(monkeypatch):
    argv = ["trainer"]
    for key in ("dataset-root", "model-snapshot", "features-dir", "preflight", "output-dir"):
        argv += [f"--{key}", "/missing"]
    argv += [
        "--arm",
        "freeze_emb",
        "--updates",
        "100",
        "--seed",
        "179024",
        "--centroid-pca-smoke",
        "products",
        "--centroid-pca-qualification",
        "/missing",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(trainer, "sha256", lambda _: "invalid")
    with pytest.raises(ValueError, match="centroid PCA100 qualification differs"):
        trainer.main()


def test_fresh_seed_confirmation_rejects_unqualified_receipt_before_gpu(monkeypatch):
    argv = ["trainer"]
    for key in ("dataset-root", "model-snapshot", "features-dir", "preflight", "output-dir"):
        argv += [f"--{key}", "/missing"]
    argv += [
        "--arm",
        "freeze_emb",
        "--updates",
        "1000",
        "--seed",
        "179028",
        "--centroid-pca-smoke",
        "products",
        "--centroid-pca-confirmation",
        "/missing",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    monkeypatch.setattr(trainer, "sha256", lambda _: "invalid")
    with pytest.raises(ValueError, match="centroid PCA seed confirmation differs"):
        trainer.main()
