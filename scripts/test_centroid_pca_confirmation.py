import copy
import json
import subprocess
import sys

import numpy as np
import pytest
import run_inshop_centroid_pca_confirmation as campaign


def test_confirmation_rejects_truncated_nonfinite_and_empty_parity():
    receipt = {
        "seed": 179028,
        "updates": 1000,
        "all_step_losses": [1.0] * 1000,
        "step_seconds": [0.8] * 1000,
        "preclip_grad_norms": [1.0] * 1000,
        "first_input_batch_sha256": ["a" * 64] * 1000,
        "width_history": [{"group_gradient_norms": dict(vision=1.0, head=1.0, classifier=1.0)}]
        * 1000,
        "matched_public32_parity": dict.fromkeys(campaign.PARITY, True),
    }
    campaign.validate_history(receipt, 179028)
    for key, value in (
        ("all_step_losses", [1.0] * 999),
        ("preclip_grad_norms", [np.nan] * 1000),
        ("matched_public32_parity", {}),
    ):
        bad = copy.deepcopy(receipt)
        bad[key] = value
        with pytest.raises(ValueError):
            campaign.validate_history(bad, 179028)
    with pytest.raises(ValueError, match="seed/update"):
        campaign.validate_history(receipt, 179029)


def test_confirmation_averages_aligned_queries_before_cluster_bootstrap(monkeypatch):
    seen = []

    def lower(values, labels):
        seen.append(np.asarray(values).copy())
        assert list(labels) == ["p", "p", "q"]
        return float(np.min(values))

    monkeypatch.setattr(campaign, "bootstrap_lower", lower)
    pairs = [
        {
            "products": {"per_query_r1": [effect] * 3, "per_query_ap": [effect] * 3},
            "control": {"per_query_ap": [0.0] * 3, "per_query_r1": [0.0] * 3},
        }
        for effect in (0.01, 0.02, 0.03)
    ]
    report = campaign.aggregate(pairs, np.asarray(["p", "p", "q"]))
    assert seen[0].shape == (3,)
    np.testing.assert_allclose(seen[0], [0.02] * 3)
    radius = 4.3026527299 * 0.01 / np.sqrt(3)
    np.testing.assert_allclose(report["map"]["seed_t95"], [0.02 - radius, 0.02 + radius])
    assert not report["go"]  # conditional product interval alone cannot pass the seed gate
    with pytest.raises(ValueError, match="all three"):
        campaign.aggregate(pairs[:2], np.asarray(["p", "p", "q"]))


def test_campaign_failure_preserves_unrun_seed_status(tmp_path, monkeypatch):
    preflights = tmp_path / "preflights"
    preflights.mkdir()
    for seed in campaign.SEEDS:
        (preflights / f"{seed}.json").write_text(json.dumps({"seed": seed}))
    qualification = tmp_path / "qualified.json"
    qualification.write_text("{}")
    gate = tmp_path / "gate.md"
    gate.write_text("frozen")
    argv = ["campaign"]
    for key in ("dataset-root", "model-snapshot", "features-dir", "cached-gate", "native-control"):
        argv += [f"--{key}", str(tmp_path / key)]
    output = tmp_path / "result"
    argv += [
        "--preflight-dir",
        str(preflights),
        "--qualification",
        str(qualification),
        "--gate",
        str(gate),
        "--output-dir",
        str(output),
    ]
    original_sha = campaign.sha256
    monkeypatch.setattr(
        campaign,
        "sha256",
        lambda path: campaign.QUALIFIED_SHA if path == qualification else original_sha(path),
    )
    monkeypatch.setattr(sys, "argv", argv)

    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(7, args[0])

    monkeypatch.setattr(campaign.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        campaign.main()
    result = json.loads((output / "receipt.json").read_text())
    assert result["decision"] == "KILL" and result["failed_stage"] == "smoke"
    assert all(state["status"] == "unrun" for state in result["seeds"].values())
    assert "aggregate" not in result
