from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from sfora.image_recipes import recipe_digest, reference_recipe


_SPEC = importlib.util.spec_from_file_location(
    "analyze_rsatk_reference",
    Path(__file__).resolve().parents[1] / "scripts" / "analyze_rsatk_reference.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _payload(*, epochs: int = 170) -> dict[str, object]:
    recipe = reference_recipe("recall_at_k_surrogate", "cars")
    history = [0.70 + i / 10000 for i in range(epochs)]
    history[99] = 0.807
    retrieval = {f"recall_at_{k}": 0.80 + k / 1000 for k in (1, 2, 4, 8)}
    best = {f"recall_at_{k}": 0.81 + k / 1000 for k in (1, 2, 4, 8)}
    best["recall_at_1"] = 0.807
    return {
        "config": {
            "dataset_name": "cars",
            "objectives": ["recall_at_k_surrogate"],
            "recipe_id": recipe.recipe_id,
            "recipe_digest": recipe_digest(recipe),
            "train_epochs": 170,
            "seed": 0,
        },
        "methods": {
            "rsatk": {
                "best_test_epoch": 100,
                "best_test_recall_at_1": 0.807,
                "best_test_retrieval": best,
                "retrieval": retrieval,
                "test_recall_history": history,
            }
        },
    }


def test_analyze_reports_preregistered_and_corrected_values() -> None:
    result = _MODULE.analyze(_payload())
    assert result["verdict"] == "WITHIN_PREREGISTERED_RANGE"
    assert result["raw_best_r@1"] == pytest.approx(0.807)
    assert result["best_epoch"] == 100
    assert result["selection_corrected_r@1"] < result["raw_best_r@1"]
    assert result["best_checkpoint_curve"]["r@8"] == pytest.approx(0.818)


def test_analyze_refuses_partial_history() -> None:
    with pytest.raises(ValueError, match="170 completed evaluations"):
        _MODULE.analyze(_payload(epochs=169))


def test_analyze_refuses_wrong_digest() -> None:
    payload = _payload()
    payload["config"]["recipe_digest"] = "wrong"  # type: ignore[index]
    with pytest.raises(ValueError, match="recipe_digest"):
        _MODULE.analyze(payload)
