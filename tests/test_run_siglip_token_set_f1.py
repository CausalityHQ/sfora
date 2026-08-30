from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sfora.token_set_training import F1ArmResult

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_siglip_token_set_f1.py"
_SPEC = importlib.util.spec_from_file_location("run_siglip_token_set_f1", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _result(arm: str, seed: int, recall: float, *, collapse: bool | None = None) -> F1ArmResult:
    return F1ArmResult(
        arm=arm,  # type: ignore[arg-type]
        seed=seed,
        final_training_objective=1.0,
        objective_kind=(
            "proxy-anchor" if arm == "pooled" else "proxy-anchor-plus-proxy-diversity"
        ),
        validation_recall_at_1=recall,
        mean_token_proxy_cosine=None if arm == "pooled" else 0.1,
        collapse_exceeded=collapse,
    )


def test_summarize_f1_results_applies_paired_method_and_shuffle_gates() -> None:
    rows = []
    for seed in (17, 29, 43):
        rows.extend(
            [
                _result("pooled", seed, 0.90),
                _result("tspa", seed, 0.91, collapse=False),
                _result("token-shuffled-tspa", seed, 0.90, collapse=False),
            ]
        )

    summary = _MODULE.summarize_f1_results(rows)

    assert summary["mean_tspa_gain_over_pooled"] == pytest.approx(0.01)
    assert summary["mean_tspa_gain_over_shuffled"] == pytest.approx(0.01)
    assert summary["passed"] is True


def test_summarize_f1_results_rejects_collapse_or_incomplete_pairing() -> None:
    rows = [
        _result("pooled", seed, 0.90)
        for seed in (17, 29, 43)
    ] + [
        _result("tspa", seed, 0.91, collapse=seed == 29)
        for seed in (17, 29, 43)
    ] + [
        _result("token-shuffled-tspa", seed, 0.90, collapse=False)
        for seed in (17, 29, 43)
    ]
    assert _MODULE.summarize_f1_results(rows)["passed"] is False

    with pytest.raises(ValueError, match="paired seeds"):
        _MODULE.summarize_f1_results(rows[:-1])

    rows[0] = _result("pooled", 17, 0.92)
    assert _MODULE.summarize_f1_results(rows)["passed"] is False


def test_validate_f0_receipt_requires_passed_train_only_screen() -> None:
    receipt = {
        "schema": "sfora-siglip-token-set-screen-v1",
        "claim_eligible": False,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "model_revision": "7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed",
        "model_name": "google/siglip-base-patch16-224",
        "dataset": "cars",
        "dataset_revision": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        "dataset_examples_sha256": "1" * 64,
        "source_revision": "2" * 40,
        "source_tree_digest": "3" * 64,
        "top_k": 32,
        "set_weight": 0.25,
        "passed": True,
    }
    _MODULE.validate_f0_receipt(receipt)

    receipt["passed"] = False
    with pytest.raises(ValueError, match="did not pass"):
        _MODULE.validate_f0_receipt(receipt)

    receipt["passed"] = True
    receipt["top_k"] = 8
    with pytest.raises(ValueError, match="authority differs"):
        _MODULE.validate_f0_receipt(receipt)
