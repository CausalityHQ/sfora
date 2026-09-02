import json
from collections import OrderedDict
from dataclasses import replace

import pytest
import torch

from sfora.weight_space_transfer import (
    INTERPOLATION_ALPHAS,
    AlphaEvaluation,
    FoldedInferenceState,
    SeedInterpolationCurve,
    canonical_interpolation_result_bytes,
    classify_interpolation_curves,
    interpolate_inference_state,
    model_state_sha256,
)


def _states() -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, torch.Tensor]]:
    initial = OrderedDict(
        {
            "tower.weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "projection.weight": torch.tensor([[2.0, -2.0]]),
            "tower.position_ids": torch.tensor([0, 1], dtype=torch.int64),
            "proxies": torch.tensor([[9.0, 8.0]]),
        }
    )
    trained = OrderedDict(
        {
            "tower.weight": torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            "projection.weight": torch.tensor([[6.0, 2.0]]),
            "tower.position_ids": torch.tensor([0, 1], dtype=torch.int64),
            "proxies": torch.tensor([[1.0, 2.0]]),
        }
    )
    return initial, trained


def test_fixed_alpha_authority_and_exact_endpoints() -> None:
    assert INTERPOLATION_ALPHAS == (0.0, 0.25, 0.5, 0.75, 1.0)
    initial, trained = _states()

    start = interpolate_inference_state(initial, trained, alpha=0.0)
    middle = interpolate_inference_state(initial, trained, alpha=0.5)
    end = interpolate_inference_state(initial, trained, alpha=1.0)

    assert type(start) is FoldedInferenceState
    assert tuple(start.state) == (
        "tower.weight",
        "projection.weight",
        "tower.position_ids",
        "proxies",
    )
    assert torch.equal(start.state["tower.weight"], initial["tower.weight"])
    assert torch.equal(end.state["tower.weight"], trained["tower.weight"])
    assert torch.equal(middle.state["tower.weight"], torch.tensor([[3.0, 4.0], [5.0, 6.0]]))
    assert torch.equal(start.state["projection.weight"], trained["projection.weight"])
    assert torch.equal(middle.state["projection.weight"], trained["projection.weight"])
    assert torch.equal(middle.state["proxies"], trained["proxies"])
    assert middle.tower_squared_displacement == pytest.approx(16.0)
    assert start.sha256 == model_state_sha256(start.state)
    assert end.sha256 == model_state_sha256(end.state)
    replay = interpolate_inference_state(initial, trained, alpha=0.5)
    assert replay.sha256 == middle.sha256
    assert replay.tower_squared_displacement == middle.tower_squared_displacement
    assert all(torch.equal(replay.state[name], value) for name, value in middle.state.items())


@pytest.mark.parametrize("alpha", [-0.1, 0.1, 1.1, True, 1])
def test_interpolation_rejects_unregistered_or_concrete_type_alpha(alpha: object) -> None:
    initial, trained = _states()
    with pytest.raises(ValueError, match="alpha"):
        interpolate_inference_state(initial, trained, alpha=alpha)  # type: ignore[arg-type]


def test_interpolation_rejects_schema_shape_dtype_and_nonfinite_drift() -> None:
    initial, trained = _states()
    mutations = []

    missing = trained.copy()
    del missing["tower.weight"]
    mutations.append(missing)

    shape = trained.copy()
    shape["tower.weight"] = torch.zeros(3)
    mutations.append(shape)

    dtype = trained.copy()
    dtype["tower.weight"] = trained["tower.weight"].double()
    mutations.append(dtype)

    nonfinite = trained.copy()
    nonfinite["tower.weight"] = trained["tower.weight"].clone()
    nonfinite["tower.weight"][0, 0] = float("nan")
    mutations.append(nonfinite)

    integer = trained.copy()
    integer["tower.position_ids"] = torch.tensor([0, 2], dtype=torch.int64)
    mutations.append(integer)

    for mutation in mutations:
        with pytest.raises(ValueError):
            interpolate_inference_state(initial, mutation, alpha=0.5)


def test_interpolation_rejects_invalid_mapping_names_and_proxy_shape() -> None:
    initial, trained = _states()

    with pytest.raises(ValueError):
        interpolate_inference_state(dict(initial), trained, alpha=0.5)  # type: ignore[arg-type]

    bad_name = initial.copy()
    bad_name[""] = bad_name.pop("tower.weight")
    with pytest.raises(ValueError):
        interpolate_inference_state(bad_name, trained, alpha=0.5)

    proxy_shape = trained.copy()
    proxy_shape["proxies"] = torch.zeros(2, 2)
    with pytest.raises(ValueError):
        interpolate_inference_state(initial, proxy_shape, alpha=0.5)


def test_state_digest_binds_order_metadata_and_bytes() -> None:
    initial, _ = _states()
    state = initial
    same = OrderedDict(reversed(tuple(state.items())))
    changed = state.copy()
    changed["tower.weight"] = changed["tower.weight"].clone()
    changed["tower.weight"][0, 0] += 1.0

    assert model_state_sha256(state) == model_state_sha256(same)
    assert model_state_sha256(state) != model_state_sha256(changed)
    with pytest.raises(ValueError):
        model_state_sha256({"tower.weight": state["tower.weight"]})  # type: ignore[arg-type]


def _row(seed: int, alpha: float, correct: int, margin: float) -> AlphaEvaluation:
    queries = 1_000
    return AlphaEvaluation(
        seed=seed,
        alpha=alpha,
        correct=correct,
        queries=queries,
        recall_ppm=correct * 1_000_000 // queries,
        mean_margin=margin,
        correctness=(True,) * correct + (False,) * (queries - correct),
        folded_state_sha256=f"{seed + round(alpha * 100):064x}",
        tower_squared_displacement=float(alpha),
    )


def _curve(
    seed: int,
    *,
    endpoint: int = 900,
    quarter: int = 901,
    half: int = 902,
    three_quarters: int = 904,
) -> SeedInterpolationCurve:
    return SeedInterpolationCurve(
        seed=seed,
        rows=(
            _row(seed, 0.0, 880, 0.10),
            _row(seed, 0.25, quarter, 0.15),
            _row(seed, 0.5, half, 0.17),
            _row(seed, 0.75, three_quarters, 0.20),
            _row(seed, 1.0, endpoint, 0.18),
        ),
    )


def test_two_seed_decisions_are_symmetric_and_provisional() -> None:
    positive = classify_interpolation_curves((_curve(17), _curve(29, three_quarters=905)))
    assert positive.terminal_class == "provisional-interior-benefit"
    assert positive.selected_alpha == 0.75
    assert positive.aggregate_delta_ppm == 4_500

    negative = classify_interpolation_curves(
        (_curve(17, three_quarters=901), _curve(29, three_quarters=900))
    )
    assert negative.terminal_class == "provisional-no-interior-benefit"
    assert negative.selected_alpha is None


def test_three_seed_common_alpha_gate_and_tie_order() -> None:
    passing = classify_interpolation_curves(
        (
            _curve(17, half=904, three_quarters=904),
            _curve(29, half=904, three_quarters=904),
            _curve(43, half=904, three_quarters=904),
        )
    )
    assert passing.terminal_class == "interior-benefit"
    assert passing.selected_alpha == 0.75
    assert passing.aggregate_delta_ppm == 4_000
    assert tuple(item.seed for item in passing.paired_evidence) == (17, 29, 43)
    assert all(item.candidate_only == 4 for item in passing.paired_evidence)
    assert all(item.endpoint_only == 0 for item in passing.paired_evidence)

    failing = classify_interpolation_curves(
        (
            _curve(17, three_quarters=904),
            _curve(29, three_quarters=904),
            _curve(43, endpoint=905, three_quarters=903),
        )
    )
    assert failing.terminal_class == "no-interior-benefit"
    assert failing.selected_alpha is None


def test_result_authority_rejects_order_arithmetic_and_concrete_type_drift() -> None:
    row = _row(17, 0.0, 880, 0.1)
    with pytest.raises(ValueError):
        AlphaEvaluation(**{**row.__dict__, "recall_ppm": row.recall_ppm + 1})
    with pytest.raises(ValueError):
        AlphaEvaluation(**{**row.__dict__, "correct": True})
    with pytest.raises(ValueError):
        AlphaEvaluation(**{**row.__dict__, "mean_margin": float("nan")})
    with pytest.raises(ValueError):
        AlphaEvaluation(**{**row.__dict__, "correctness": row.correctness[:-1]})
    with pytest.raises(ValueError):
        SeedInterpolationCurve(seed=17, rows=tuple(reversed(_curve(17).rows)))
    with pytest.raises(ValueError):
        classify_interpolation_curves((_curve(29), _curve(17)))


def test_one_query_non_regression_is_independent_from_mean_gate() -> None:
    decision = classify_interpolation_curves(
        (
            _curve(17, three_quarters=910),
            _curve(29, three_quarters=910),
            _curve(43, endpoint=900, three_quarters=898),
        )
    )
    assert decision.terminal_class == "no-interior-benefit"


def test_canonical_result_recomputes_decision_and_binds_rows() -> None:
    curves = (_curve(17), _curve(29, three_quarters=905), _curve(43, three_quarters=904))
    decision = classify_interpolation_curves(curves)
    payload = canonical_interpolation_result_bytes(curves, decision)
    assert payload.endswith(b"\n") and not payload.endswith(b"\n\n")
    value = json.loads(payload)
    assert value["schema"] == "sfora-weight-space-transfer-result-v1"
    assert value["claim_eligible"] is False
    assert value["decision"]["terminal_class"] == "interior-benefit"
    assert value["decision"]["selected_alpha"] == 0.75
    assert len(value["curves"]) == 3
    assert len(value["curves"][0]["rows"]) == 5
    assert len(value["curves"][0]["rows"][0]["correctness_sha256"]) == 64
    assert payload == canonical_interpolation_result_bytes(curves, decision)

    with pytest.raises(ValueError, match="decision"):
        canonical_interpolation_result_bytes(
            curves,
            replace(decision, aggregate_delta_ppm=decision.aggregate_delta_ppm + 1),
        )
