from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np

_SPEC = spec_from_file_location(
    "audit_static_checkpoint_limits",
    Path(__file__).resolve().parents[1] / "scripts" / "audit_static_checkpoint_limits.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def test_compute_observables_recovers_margins_and_agreement() -> None:
    embeddings = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=float)
    labels = np.array([0, 0, 1, 1])
    proxies = np.array([[1, 0], [0, 1]], dtype=float)
    result = _MODULE.compute_observables(embeddings, labels, proxies, np.array([0, 1]))
    assert not result["error"].any()
    assert np.all(result["image_margin"] > 0)
    assert result["foreign_index"].tolist() == [3, 3, 1, 1]


def test_margin_sufficiency_detects_added_agreement_signal() -> None:
    rng = np.random.default_rng(4)
    count = 4000
    proxy = rng.normal(size=count)
    image = rng.normal(size=count)
    agreement = rng.random(count) < 0.3
    logits = -3.0 - proxy - image + 2.0 * agreement
    error = rng.random(count) < 1.0 / (1.0 + np.exp(-logits))
    result = _MODULE.margin_sufficiency(
        {
            "proxy_margin": proxy,
            "image_margin": image,
            "agreement": agreement,
            "error": error,
        }
    )
    assert result["agreement_coefficient"] > 1.0
    assert result["p_value_chi2_df1"] < 0.01


def test_margin_sufficiency_excludes_undefined_singleton_margin() -> None:
    rng = np.random.default_rng(8)
    count = 2000
    proxy = rng.normal(size=count)
    image = rng.normal(size=count)
    agreement = rng.random(count) < 0.2
    logits = -3.0 - proxy - image
    error = rng.random(count) < 1.0 / (1.0 + np.exp(-logits))
    image[0] = -np.inf
    error[0] = True
    result = _MODULE.margin_sufficiency(
        {
            "proxy_margin": proxy,
            "image_margin": image,
            "agreement": agreement,
            "error": error,
        }
    )
    assert result["excluded_nonfinite_rows"] == 1
    assert result["excluded_nonfinite_events"] == 1
    assert np.isfinite(result["base_log_likelihood"])


def test_margin_sufficiency_reports_definitional_complete_separation() -> None:
    rng = np.random.default_rng(12)
    count = 1000
    image = rng.normal(size=count)
    result = _MODULE.margin_sufficiency(
        {
            "proxy_margin": rng.normal(size=count),
            "image_margin": image,
            "agreement": rng.random(count) < 0.2,
            "error": image < 0,
        }
    )
    assert result["model_status"] == "complete_separation_by_definition"
    assert result["image_margin_sign_identity_mismatches"] == 0
    assert result["likelihood_ratio_valid"] is False
