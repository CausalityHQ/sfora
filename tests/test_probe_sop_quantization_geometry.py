from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts/probe_sop_quantization_geometry.py"
sys.path.insert(0, str(_SCRIPT.parent))
_SPEC = importlib.util.spec_from_file_location("probe_sop_quantization_geometry", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _unit(rows: int, dimensions: int) -> torch.Tensor:
    values = torch.arange(1, rows * dimensions + 1, dtype=torch.float32).reshape(rows, dimensions)
    return F.normalize(values, dim=1).contiguous()


def test_clipped_int4_is_exact_and_ratio_is_selected_train_only() -> None:
    values = F.normalize(
        torch.tensor(
            [
                [9.0, 1.0, 1.0, 1.0],
                [1.0, 9.0, 1.0, 1.0],
                [1.0, 1.0, 9.0, 1.0],
                [1.0, 1.0, 1.0, 9.0],
            ],
            dtype=torch.float32,
        ),
        dim=1,
    ).contiguous()

    ratio, evidence = _MODULE.select_clipping_ratio(values, ratios=(0.5, 1.0))
    packed = _MODULE.pack_clipped_int4_unit_embeddings(values, scale_ratio=ratio)

    assert ratio in (0.5, 1.0)
    assert set(evidence) == {"0.5", "1.0"}
    assert evidence[str(ratio)] == min(evidence.values())
    assert packed.bytes_per_vector == 4
    with pytest.raises(ValueError, match="clipping authority"):
        _MODULE.pack_clipped_int4_unit_embeddings(values, scale_ratio=0.0)


def test_asymmetric_score_uses_float_queries_and_exact_int4_gallery() -> None:
    values = F.normalize(
        torch.tensor(
            [[1.0, 0.1], [1.0, 0.1], [0.1, 1.0], [0.1, 1.0]],
            dtype=torch.float32,
        ),
        dim=1,
    ).contiguous()
    gallery = _MODULE.pack_clipped_int4_unit_embeddings(values, scale_ratio=1.0)

    score = _MODULE.score_asymmetric_int4(
        values,
        gallery,
        (1, 1, 2, 2),
        candidate_width=2,
        device=torch.device("cpu"),
    )

    assert score["map_at_r"] == 1.0
    assert score["r1"] == 1.0
    assert score["per_query_ap"] == (1.0, 1.0, 1.0, 1.0)

    challenging = torch.tensor(
        [
            [
                0.15659668,
                -0.47528854,
                -0.18951826,
                0.52923125,
                0.09509413,
                0.16080928,
                0.24744396,
                0.58089960,
            ],
            [
                0.50087386,
                0.14729896,
                0.48620504,
                0.13882582,
                0.05193219,
                0.16049665,
                0.19607909,
                0.63628393,
            ],
            [
                -0.05302174,
                -0.60945630,
                0.32725704,
                0.11774431,
                -0.13368745,
                -0.09903065,
                0.63094592,
                -0.28110278,
            ],
            [
                0.31774649,
                -0.66898692,
                0.04530496,
                -0.25937417,
                0.23649243,
                -0.41531593,
                -0.35937023,
                -0.15685400,
            ],
            [
                0.04472185,
                0.25639027,
                -0.52161556,
                0.05461674,
                -0.61107665,
                0.06600674,
                0.00607694,
                0.52857316,
            ],
            [
                0.01458514,
                0.11636167,
                0.69988889,
                0.10584597,
                0.29252610,
                -0.23219010,
                -0.51652646,
                0.28091988,
            ],
        ],
        dtype=torch.float32,
    ).contiguous()
    challenging_gallery = _MODULE.pack_clipped_int4_unit_embeddings(challenging, scale_ratio=1.0)
    challenging_labels = (1, 1, 2, 2, 3, 3)
    asymmetric = _MODULE.score_asymmetric_int4(
        challenging,
        challenging_gallery,
        challenging_labels,
        candidate_width=2,
        device=torch.device("cpu"),
    )
    symmetric = _MODULE.score_int4_symmetric(
        challenging_gallery,
        challenging_labels,
        candidate_width=2,
        device=torch.device("cpu"),
    )

    assert asymmetric["map_at_r"] == pytest.approx(1.0 / 3.0)
    assert symmetric["map_at_r"] == pytest.approx(0.5)


def test_asymmetric_score_decodes_gallery_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values = _unit(520, 8)
    gallery = _MODULE.pack_clipped_int4_unit_embeddings(values, scale_ratio=1.0)
    labels = tuple(index // 2 + 1 for index in range(520))
    original = _MODULE.PackedInt4Embeddings.signed_codes
    calls = 0

    def counted(instance: object) -> torch.Tensor:
        nonlocal calls
        calls += 1
        return original(instance)

    monkeypatch.setattr(_MODULE.PackedInt4Embeddings, "signed_codes", counted)
    _MODULE.score_asymmetric_int4(
        values,
        gallery,
        labels,
        candidate_width=1,
        device=torch.device("cpu"),
    )

    assert calls == 1


def test_output_rotation_folds_into_linear_model_without_runtime_stage() -> None:
    model = _MODULE.RelationalLinearEncoder(_unit(4, 8))
    rotation = _MODULE.fixed_random_rotation(4, seed=17)
    folded = _MODULE.fold_output_rotation(model, rotation)
    values = _unit(6, 8)

    expected = F.normalize(model(values) @ rotation.T, dim=1)

    torch.testing.assert_close(folded(values), expected, atol=2e-6, rtol=2e-6)
    assert folded.to_bytes() != model.to_bytes()


def test_quantization_decision_is_sequential_and_fail_closed() -> None:
    passing = _MODULE.quantization_decision(
        baseline_symmetric_map_at_r=0.407,
        baseline_asymmetric_map_at_r=0.411,
        rotated_clipped_asymmetric_map_at_r=0.412,
        asymmetric_map_lower_bound=0.001,
        composed_map_lower_bound=0.0001,
        pca_float_to_int4_loss=0.014,
        rotated_recovery=0.008,
    )
    failing = _MODULE.quantization_decision(
        baseline_symmetric_map_at_r=0.407,
        baseline_asymmetric_map_at_r=0.409,
        rotated_clipped_asymmetric_map_at_r=0.414,
        asymmetric_map_lower_bound=0.001,
        composed_map_lower_bound=0.001,
        pca_float_to_int4_loss=0.014,
        rotated_recovery=0.006,
    )
    regressing_composition = _MODULE.quantization_decision(
        baseline_symmetric_map_at_r=0.407,
        baseline_asymmetric_map_at_r=0.411,
        rotated_clipped_asymmetric_map_at_r=0.390,
        asymmetric_map_lower_bound=0.001,
        composed_map_lower_bound=-0.001,
        pca_float_to_int4_loss=0.014,
        rotated_recovery=0.008,
    )

    assert passing == {
        "asymmetric": True,
        "asymmetric_gain": pytest.approx(0.004),
        "asymmetric_map_lower_bound": 0.001,
        "composed_gain": pytest.approx(0.001),
        "composed_map_lower_bound": 0.0001,
        "rotation_clipping": True,
        "winner": "rotated-clipped-asymmetric-int4",
    }
    assert failing == {
        "asymmetric": False,
        "asymmetric_gain": pytest.approx(0.002),
        "asymmetric_map_lower_bound": 0.001,
        "composed_gain": pytest.approx(0.005),
        "composed_map_lower_bound": 0.001,
        "rotation_clipping": False,
        "winner": "baseline-symmetric-int4",
    }
    assert regressing_composition == {
        "asymmetric": True,
        "asymmetric_gain": pytest.approx(0.004),
        "asymmetric_map_lower_bound": 0.001,
        "composed_gain": pytest.approx(-0.021),
        "composed_map_lower_bound": -0.001,
        "rotation_clipping": False,
        "winner": "asymmetric-int4",
    }
    with pytest.raises(ValueError, match="decision authority"):
        _MODULE.quantization_decision(
            baseline_symmetric_map_at_r=float("nan"),
            baseline_asymmetric_map_at_r=0.411,
            rotated_clipped_asymmetric_map_at_r=0.412,
            asymmetric_map_lower_bound=0.001,
            composed_map_lower_bound=0.0001,
            pca_float_to_int4_loss=0.014,
            rotated_recovery=0.008,
        )


def test_sealed_pca_reproduction_fails_closed_on_drift() -> None:
    _MODULE.validate_sealed_pca_reproduction(
        float_map_at_r=_MODULE.SEALED_PCA_FLOAT_MAP_AT_R,
    )
    with pytest.raises(ValueError, match="sealed PCA reproduction"):
        _MODULE.validate_sealed_pca_reproduction(
            float_map_at_r=_MODULE.SEALED_PCA_FLOAT_MAP_AT_R + 2e-6,
        )


def test_real_small_quantization_pipeline_preserves_equal_byte_arms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_MODULE, "DIMENSIONS", 4)
    monkeypatch.setattr(_MODULE, "BATCH", 4)
    monkeypatch.setattr(_MODULE, "EPOCHS", 1)
    monkeypatch.setattr(_MODULE, "CANDIDATE_WIDTH", 2)
    monkeypatch.setattr(_MODULE, "validate_sealed_pca_reproduction", lambda **_kwargs: None)
    pair = {
        "source_metadata": {},
        "teacher_metadata": {},
        "source_train": _unit(8, 8),
        "teacher_train": _unit(8, 10).roll(1, dims=1),
        "source_test": _unit(8, 8).roll(2, dims=1),
        "teacher_test": _unit(8, 10).roll(3, dims=1),
        "train_labels": (1, 1, 2, 2, 3, 3, 4, 4),
        "test_labels": (5, 5, 6, 6, 7, 7, 8, 8),
    }

    result = _MODULE.run_burned_sop_quantization(pair)

    assert result["schema"] == "sfora-sop-quantization-geometry-v1"
    assert result["claim_eligible"] is False
    assert result["clipping_pair_samples"] == _MODULE.CLIPPING_PAIR_SAMPLES
    assert result["clipping_ratios"] == _MODULE.CLIPPING_RATIOS
    assert result["gates"] == {
        "asymmetric_gain": _MODULE.ASYMMETRIC_GAIN_GATE,
        "rotation_recovery_fraction": _MODULE.ROTATION_RECOVERY_FRACTION,
    }
    assert result["rotation"] == {"kind": "seed-fixed-orthogonal", "seed": 17}
    assert result["pca"]["persistent_bytes_per_item"] == 4
    assert result["relational"]["persistent_bytes_per_item"] == 4
    assert result["relational"]["query_bytes_per_item"] == 16
    assert {
        "baseline_asymmetric",
        "baseline_symmetric",
        "clipped_only_asymmetric",
        "clipped_only_symmetric",
        "rotation_only_asymmetric",
        "rotation_only_symmetric",
        "rotated_clipped_asymmetric",
        "rotated_clipped_symmetric",
    } <= set(result["relational"])
    assert len(result["relational"]["per_query_evidence"]["baseline_asymmetric"]["ap"]) == 8
    assert len(result["selected_model_sha256"]) == 64
    if result["decision"]["rotation_clipping"]:
        assert result["selected_model_sha256"] == result["rotated_relational_model_sha256"]
    else:
        assert result["selected_model_sha256"] == result["relational_model_sha256"]
    assert len(result["training_losses"]) == 1


def test_quantization_source_must_match_registered_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    checked: list[str] = []
    monkeypatch.setattr(
        _MODULE,
        "verify_source_commit",
        lambda commit: checked.append(commit),
    )
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=_SCRIPT.read_bytes()
        ),
    )

    _MODULE.verify_quantization_source_commit("12" * 20)

    assert checked == ["12" * 20]
    monkeypatch.setattr(
        _MODULE.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            args=[], returncode=0, stdout=b"different"
        ),
    )
    with pytest.raises(ValueError, match="registered commit"):
        _MODULE.verify_quantization_source_commit("12" * 20)
