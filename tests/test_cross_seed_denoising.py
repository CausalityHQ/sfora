from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from sfora.cross_seed_denoising import (
    CandidateEvaluation,
    CandidateStates,
    DenoisingDecision,
    HeadSwapEvaluation,
    ProjectedEvaluation,
    build_cross_seed_candidates,
    canonical_denoising_result_bytes,
    classify_denoising_result,
    read_denoising_result,
    read_tensor_artifact,
    wiener_gain,
    write_tensor_artifact,
)
from sfora.weight_space_transfer import AlphaEvaluation, SeedInterpolationCurve

BINDINGS = {
    "checkpoint_sha256": "11" * 32,
    "source_commit": "22" * 20,
}


def _state() -> OrderedDict[str, torch.Tensor]:
    return OrderedDict(
        (
            ("tower.a", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
            ("tower.count", torch.tensor([3], dtype=torch.int64)),
        )
    )


def _assert_state_equal(
    actual: OrderedDict[str, torch.Tensor],
    expected: OrderedDict[str, torch.Tensor],
) -> None:
    assert tuple(actual) == tuple(expected)
    for name in expected:
        assert actual[name].dtype == expected[name].dtype
        assert actual[name].shape == expected[name].shape
        assert torch.equal(actual[name], expected[name])


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()


def test_tensor_artifact_round_trip_is_byte_deterministic(tmp_path: Path) -> None:
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"

    left = write_tensor_artifact(left_root, _state(), role="tower", bindings=BINDINGS)
    right = write_tensor_artifact(right_root, _state(), role="tower", bindings=BINDINGS)

    assert left == right
    assert left_root.joinpath("manifest.json").read_bytes() == left
    _assert_state_equal(read_tensor_artifact(left_root, left, role="tower"), _state())


def test_tensor_artifact_rejects_nonfinite_and_concrete_type_drift(tmp_path: Path) -> None:
    nonfinite = _state()
    nonfinite["tower.a"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        write_tensor_artifact(tmp_path / "nan", nonfinite, role="tower", bindings=BINDINGS)

    with pytest.raises((TypeError, ValueError), match="bindings"):
        write_tensor_artifact(
            tmp_path / "bool-binding",
            _state(),
            role="tower",
            bindings={"checkpoint_sha256": False},  # type: ignore[dict-item]
        )


def test_tensor_artifact_rejects_payload_digest_drift(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    manifest = json.loads(manifest_bytes)
    tensor_path = root / manifest["tensors"][0]["file"]
    payload = bytearray(tensor_path.read_bytes())
    payload[0] ^= 1
    tensor_path.write_bytes(payload)

    with pytest.raises(ValueError, match="digest"):
        read_tensor_artifact(root, manifest_bytes, role="tower")


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        (lambda value: value.update({"role": "head"}), "role"),
        (lambda value: value["bindings"].update({"source_commit": "33" * 20}), "bindings"),
        (lambda value: value["tensors"][0].update({"dtype": "torch.complex64"}), "dtype"),
        (lambda value: value["tensors"][0].update({"shape": [3]}), "shape"),
        (lambda value: value["tensors"][0].update({"bytes": 7}), "length"),
        (lambda value: value["tensors"][0].update({"file": "../escape.bin"}), "path"),
    ),
)
def test_tensor_artifact_rejects_manifest_semantic_drift(
    tmp_path: Path,
    mutation: object,
    message: str,
) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    value = json.loads(manifest_bytes)
    mutation(value)  # type: ignore[operator]

    with pytest.raises(ValueError, match=message):
        read_tensor_artifact(root, _canonical(value), role="tower")


def test_tensor_artifact_rejects_noncanonical_manifest(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    noncanonical = json.dumps(json.loads(manifest_bytes), indent=2).encode() + b"\n"

    with pytest.raises(ValueError, match="canonical"):
        read_tensor_artifact(root, noncanonical, role="tower")


def test_tensor_artifact_rejects_symlinked_payload(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    manifest = json.loads(manifest_bytes)
    tensor_path = root / manifest["tensors"][0]["file"]
    original = tensor_path.with_suffix(".original")
    tensor_path.rename(original)
    tensor_path.symlink_to(original.name)

    with pytest.raises(ValueError, match="symlink"):
        read_tensor_artifact(root, manifest_bytes, role="tower")


def test_tensor_artifact_manifest_binds_complete_payload_digest(tmp_path: Path) -> None:
    root = tmp_path / "artifact"
    manifest_bytes = write_tensor_artifact(root, _state(), role="tower", bindings=BINDINGS)
    manifest = json.loads(manifest_bytes)
    payload_digests = tuple(record["sha256"] for record in manifest["tensors"])

    assert payload_digests == tuple(
        hashlib.sha256((root / record["file"]).read_bytes()).hexdigest()
        for record in manifest["tensors"]
    )
    assert len(manifest["state_sha256"]) == 64


def _candidate_fixture(
    updates: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    name: str = "tower.weight",
) -> tuple[OrderedDict[str, torch.Tensor], dict[int, OrderedDict[str, torch.Tensor]]]:
    initial = OrderedDict(((name, torch.zeros_like(updates[0])),))
    endpoints = {
        seed: OrderedDict(((name, update.clone()),))
        for seed, update in zip((17, 29, 43), updates, strict=True)
    }
    return initial, endpoints


def test_wiener_gain_has_registered_closed_form_and_domain() -> None:
    assert wiener_gain(0.0) == 0.0
    assert wiener_gain(0.5) == 0.75
    assert wiener_gain(1.0) == 1.0
    with pytest.raises(ValueError, match="rho"):
        wiener_gain(-0.01)
    with pytest.raises(ValueError, match="rho"):
        wiener_gain(float("nan"))


def test_wiener_candidate_uses_one_group_per_named_tensor_and_reports_gjs() -> None:
    shared = (
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
        torch.tensor([1.0, 2.0]),
    )
    orthogonal = (
        torch.tensor([1.0, 0.0]),
        torch.tensor([0.0, 1.0]),
        torch.tensor([-1.0, 0.0]),
    )
    initial = OrderedDict(
        (
            ("tower.a", torch.zeros(2)),
            ("tower.b", torch.zeros(2)),
            ("tower.counter", torch.tensor([7], dtype=torch.int64)),
        )
    )
    endpoints = {
        seed: OrderedDict(
            (
                ("tower.a", shared[index]),
                ("tower.b", orthogonal[index]),
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
            )
        )
        for index, seed in enumerate((17, 29, 43))
    }

    result = build_cross_seed_candidates(initial, endpoints)

    assert isinstance(result, CandidateStates)
    assert tuple(row.name for row in result.groups) == ("tower.a", "tower.b")
    assert result.groups[0].rho == 1.0
    assert result.groups[0].beta == 1.0
    assert result.groups[0].g_js == 1.0
    assert result.groups[1].rho == 0.0
    assert result.groups[1].beta == 0.0
    assert 0.0 <= result.groups[1].g_js <= 1.0
    assert torch.equal(result.wiener_denoise["tower.a"], shared[0])
    assert torch.equal(result.wiener_denoise["tower.b"], torch.zeros(2))
    assert torch.equal(result.tower_soup["tower.counter"], initial["tower.counter"])
    assert result.tower_soup["tower.a"].dtype == torch.float32


def test_wiener_zero_norm_forces_zero_rho_and_candidate_update() -> None:
    initial, endpoints = _candidate_fixture(
        (
            torch.zeros(3),
            torch.tensor([1.0, 0.0, 0.0]),
            torch.tensor([1.0, 0.0, 0.0]),
        )
    )
    result = build_cross_seed_candidates(initial, endpoints)
    assert result.groups[0].cosines == (0.0, 0.0, 1.0)
    assert result.groups[0].rho == 0.0
    assert torch.equal(result.wiener_denoise["tower.weight"], torch.zeros(3))


def test_candidate_construction_is_invariant_to_all_seed_mapping_permutations() -> None:
    initial, endpoints = _candidate_fixture(
        (
            torch.tensor([[3.0, 0.4], [0.0, 0.2]]),
            torch.tensor([[3.0, -0.4], [0.0, 0.2]]),
            torch.tensor([[3.0, 0.0], [0.0, 0.2]]),
        )
    )
    authority = build_cross_seed_candidates(initial, endpoints)
    for order in itertools.permutations((17, 29, 43)):
        permuted = {seed: endpoints[seed] for seed in order}
        candidate = build_cross_seed_candidates(initial, permuted)
        assert candidate.groups == authority.groups
        assert candidate.spectral == authority.spectral
        for role in ("tower_soup", "wiener_denoise", "spectral_denoise"):
            assert torch.equal(
                getattr(candidate, role)["tower.weight"],
                getattr(authority, role)["tower.weight"],
            )


def test_spectral_candidate_uses_symmetric_contrast_edge_and_hard_rank_cut() -> None:
    mean = torch.diag(torch.tensor([3.0, 0.1]))
    noise = torch.diag(torch.tensor([0.0, 0.5]))
    initial, endpoints = _candidate_fixture((mean + noise, mean - noise, mean))

    result = build_cross_seed_candidates(initial, endpoints)

    evidence = result.spectral[0]
    assert evidence.name == "tower.weight"
    assert evidence.kept_rank == 1
    assert evidence.total_rank == 2
    assert evidence.edge == pytest.approx(math.sqrt(2.0 / 3.0) * 0.5)
    assert evidence.retained_energy == pytest.approx(9.0)
    assert evidence.total_energy == pytest.approx(9.01)
    assert torch.equal(
        result.spectral_denoise["tower.weight"],
        torch.tensor([[3.0, 0.0], [0.0, 0.0]]),
    )


def test_spectral_vector_and_scalar_tensors_use_wiener_fallback() -> None:
    initial = OrderedDict(
        (
            ("tower.scalar", torch.tensor(0.0)),
            ("tower.vector", torch.zeros(2)),
        )
    )
    endpoints = {
        seed: OrderedDict(
            (
                ("tower.scalar", torch.tensor(value)),
                ("tower.vector", torch.tensor([value, value * 2])),
            )
        )
        for seed, value in zip((17, 29, 43), (1.0, 1.1, 0.9), strict=True)
    }
    result = build_cross_seed_candidates(initial, endpoints)
    assert result.spectral == ()
    assert torch.equal(
        result.spectral_denoise["tower.scalar"], result.wiener_denoise["tower.scalar"]
    )
    assert torch.equal(
        result.spectral_denoise["tower.vector"], result.wiener_denoise["tower.vector"]
    )


def test_spectral_convolution_and_rectangular_updates_preserve_shape() -> None:
    mean = torch.arange(24, dtype=torch.float32).reshape(2, 3, 2, 2) / 10
    noise = torch.zeros_like(mean)
    noise[0, 0, 0, 0] = 0.05
    initial, endpoints = _candidate_fixture((mean + noise, mean - noise, mean))
    result = build_cross_seed_candidates(initial, endpoints)
    assert result.spectral_denoise["tower.weight"].shape == (2, 3, 2, 2)
    assert result.spectral[0].total_rank == 2


def test_spectral_rejects_singular_value_at_registered_edge() -> None:
    mean = torch.diag(torch.tensor([3.0, 1.0], dtype=torch.float64))
    noise = torch.diag(torch.tensor([0.0, math.sqrt(1.5)], dtype=torch.float64))
    initial, endpoints = _candidate_fixture((mean + noise, mean - noise, mean))
    with pytest.raises(ValueError, match="spectral edge"):
        build_cross_seed_candidates(initial, endpoints)


def test_candidate_construction_rejects_state_and_nonfloating_drift() -> None:
    initial = OrderedDict(
        (
            ("tower.counter", torch.tensor([7], dtype=torch.int64)),
            ("tower.weight", torch.zeros(2)),
        )
    )
    endpoints = {
        seed: OrderedDict(
            (
                ("tower.counter", torch.tensor([7], dtype=torch.int64)),
                ("tower.weight", torch.ones(2)),
            )
        )
        for seed in (17, 29, 43)
    }
    endpoints[29]["tower.counter"][0] = 8
    with pytest.raises(ValueError, match="non-floating"):
        build_cross_seed_candidates(initial, endpoints)

    with pytest.raises(ValueError, match="seeds"):
        build_cross_seed_candidates(initial, {17: endpoints[17], 29: endpoints[29]})


_QUERIES = 1345
_CANDIDATES = ("tower-soup", "wiener-denoise", "spectral-denoise")


def _correctness(correct: int) -> tuple[bool, ...]:
    return (True,) * correct + (False,) * (_QUERIES - correct)


def _scalar_curves(correct: int = 1258, margin: float = 0.20) -> tuple[SeedInterpolationCurve, ...]:
    curves: list[SeedInterpolationCurve] = []
    for seed in (17, 29, 43):
        rows = []
        for alpha, delta in zip((0.0, 0.25, 0.5, 0.75, 1.0), (-10, -5, -2, 0, -1), strict=True):
            hits = correct + delta
            rows.append(
                AlphaEvaluation(
                    seed=seed,
                    alpha=alpha,
                    correct=hits,
                    queries=_QUERIES,
                    recall_ppm=hits * 1_000_000 // _QUERIES,
                    mean_nearest_positive_cosine=0.5,
                    mean_nearest_negative_cosine=0.5 - (margin + delta / 10_000),
                    mean_margin=margin + delta / 10_000,
                    correctness=_correctness(hits),
                    folded_state_sha256=f"{seed:02x}" * 32,
                    tower_squared_displacement=float(alpha),
                    wall_time_ns=1,
                    peak_cuda_bytes=0,
                    peak_rss_bytes=1,
                )
            )
        curves.append(SeedInterpolationCurve(seed=seed, rows=tuple(rows)))
    return tuple(curves)


def _projected(seed: int, correct: int, margin: float, *, digest_byte: str) -> ProjectedEvaluation:
    return ProjectedEvaluation(
        seed=seed,
        correctness=_correctness(correct),
        mean_nearest_positive_cosine=0.5,
        mean_nearest_negative_cosine=0.5 - margin,
        mean_margin=margin,
        folded_state_sha256=digest_byte * 64,
        wall_time_ns=10,
        peak_cuda_bytes=100,
        peak_rss_bytes=200,
        determinism_replay=True,
    )


def _candidate(
    role: str,
    correct: int,
    margin: float,
    *,
    digest_byte: str,
) -> CandidateEvaluation:
    return CandidateEvaluation(
        role=role,
        raw_correctness=_correctness(correct - 5),
        raw_mean_nearest_positive_cosine=0.5,
        raw_mean_nearest_negative_cosine=0.5 - (margin - 0.01),
        raw_mean_margin=margin - 0.01,
        raw_wall_time_ns=11,
        raw_peak_cuda_bytes=101,
        raw_peak_rss_bytes=201,
        raw_determinism_replay=True,
        projected=tuple(
            _projected(seed, correct, margin, digest_byte=digest_byte)
            for seed in (17, 29, 43)
        ),
        tower_state_sha256=digest_byte * 64,
        construction_evidence_sha256=("f" if digest_byte != "f" else "e") * 64,
    )


def _candidates(
    *,
    soup: tuple[int, float] = (1261, 0.21),
    wiener: tuple[int, float] = (1263, 0.22),
    spectral: tuple[int, float] = (1265, 0.23),
) -> tuple[CandidateEvaluation, ...]:
    return (
        _candidate("tower-soup", *soup, digest_byte="a"),
        _candidate("wiener-denoise", *wiener, digest_byte="b"),
        _candidate("spectral-denoise", *spectral, digest_byte="c"),
    )


def _swaps(*, coadapted: bool = True) -> tuple[HeadSwapEvaluation, ...]:
    rows = []
    for source in (17, 29, 43):
        for target in (17, 29, 43):
            if source == target:
                continue
            own = 1260
            swapped = 1259 if coadapted else own
            rows.append(
                HeadSwapEvaluation(
                    source_seed=source,
                    target_seed=target,
                    own_correctness=_correctness(own),
                    swapped_correctness=_correctness(swapped),
                    own_mean_margin=0.20,
                    swapped_mean_margin=0.19 if coadapted else 0.20,
                )
            )
    return tuple(rows)


def test_evaluation_rows_recompute_counts_ppm_and_per_seed_mcnemar() -> None:
    decision = classify_denoising_result(_scalar_curves(), _candidates(), _swaps())
    assert decision.terminal_class == "spectral-denoise-benefit"
    assert decision.selected_candidate == "spectral-denoise"
    assert decision.best_scalar_alpha == 0.75
    assert decision.head_coadaptation_observed is True
    assert decision.candidate_passes == (True, True, True)
    assert decision.reaches_95_percent == (False, False, False)
    spectral_pairs = decision.paired_evidence[2]
    assert tuple(row.seed for row in spectral_pairs) == (17, 29, 43)
    assert all(row.candidate_only == 7 and row.scalar_only == 0 for row in spectral_pairs)
    assert all(row.mcnemar_p_value == pytest.approx(1.0 / 64.0) for row in spectral_pairs)


@pytest.mark.parametrize(
    ("candidates", "terminal", "selected"),
    (
        (_candidates(spectral=(1263, 0.22)), "wiener-denoise-benefit", "wiener-denoise"),
        (
            _candidates(wiener=(1261, 0.21), spectral=(1261, 0.21)),
            "tower-soup-only-benefit",
            "tower-soup",
        ),
        (
            _candidates(soup=(1258, 0.20), wiener=(1258, 0.20), spectral=(1258, 0.20)),
            "no-cross-seed-benefit-with-head-coadaptation",
            None,
        ),
    ),
)
def test_decision_uses_fixed_candidate_priority_and_complete_gates(
    candidates: tuple[CandidateEvaluation, ...],
    terminal: str,
    selected: str | None,
) -> None:
    decision = classify_denoising_result(_scalar_curves(), candidates, _swaps())
    assert decision.terminal_class == terminal
    assert decision.selected_candidate == selected


def test_decision_distinguishes_no_benefit_without_head_coadaptation() -> None:
    candidates = _candidates(
        soup=(1258, 0.20), wiener=(1258, 0.20), spectral=(1258, 0.20)
    )
    decision = classify_denoising_result(_scalar_curves(), candidates, _swaps(coadapted=False))
    assert decision.terminal_class == "no-cross-seed-benefit"


@pytest.mark.parametrize(
    "failure",
    ("authority-failure", "numerical-failure", "resource-failure"),
)
def test_decision_failure_precedence_is_fail_closed(failure: str) -> None:
    decision = classify_denoising_result(
        _scalar_curves(), _candidates(), _swaps(), failure=failure
    )
    assert decision.terminal_class == failure
    assert decision.selected_candidate is None
    assert decision.candidate_passes == (False, False, False)


def test_evaluation_rejects_wrong_order_cardinality_types_and_nonfinite_values() -> None:
    projected = _candidate("tower-soup", 1261, 0.21, digest_byte="a").projected
    with pytest.raises(ValueError, match="seed order"):
        CandidateEvaluation(
            role="tower-soup",
            raw_correctness=_correctness(1250),
            raw_mean_nearest_positive_cosine=0.5,
            raw_mean_nearest_negative_cosine=0.3,
            raw_mean_margin=0.2,
            raw_wall_time_ns=1,
            raw_peak_cuda_bytes=0,
            raw_peak_rss_bytes=1,
            raw_determinism_replay=True,
            projected=(projected[1], projected[0], projected[2]),
            tower_state_sha256="a" * 64,
            construction_evidence_sha256="f" * 64,
        )
    with pytest.raises(ValueError, match="finite"):
        _projected(17, 1261, float("nan"), digest_byte="a")
    with pytest.raises(ValueError, match="determinism"):
        ProjectedEvaluation(
            seed=17,
            correctness=_correctness(1261),
            mean_nearest_positive_cosine=0.5,
            mean_nearest_negative_cosine=0.29,
            mean_margin=0.21,
            folded_state_sha256="a" * 64,
            wall_time_ns=1,
            peak_cuda_bytes=0,
            peak_rss_bytes=1,
            determinism_replay=False,
        )
    with pytest.raises(ValueError, match="correctness"):
        ProjectedEvaluation(
            seed=17,
            correctness=(True,) * (_QUERIES - 1),
            mean_nearest_positive_cosine=0.5,
            mean_nearest_negative_cosine=0.2,
            mean_margin=0.3,
            folded_state_sha256="a" * 64,
            wall_time_ns=1,
            peak_cuda_bytes=0,
            peak_rss_bytes=1,
            determinism_replay=True,
        )


def test_head_swap_requires_all_six_ordered_pairs() -> None:
    with pytest.raises(ValueError, match="swap order"):
        classify_denoising_result(_scalar_curves(), _candidates(), _swaps()[:-1])


def test_canonical_result_round_trips_and_rejects_stored_mutations() -> None:
    scalar = _scalar_curves()
    candidates = _candidates()
    swaps = _swaps()
    decision = classify_denoising_result(scalar, candidates, swaps)
    raw = canonical_denoising_result_bytes(scalar, candidates, swaps, decision)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert read_denoising_result(raw) == decision
    first = json.loads(raw)["candidates"][0]
    assert first["raw_wall_time_ns"] == 11
    assert first["raw_peak_cuda_bytes"] == 101
    assert first["raw_peak_rss_bytes"] == 201
    assert first["raw_determinism_replay"] is True

    for mutation, message in (
        (lambda item: item.update({"claim_eligible": True}), "claim"),
        (lambda item: item["decision"].update({"terminal_class": "resource-failure"}), "decision"),
        (lambda item: item["candidates"][0].update({"aggregate_correct": 0}), "aggregate"),
        (lambda item: item["candidates"][0].update({"role": "spectral-denoise"}), "order"),
    ):
        changed = json.loads(raw)
        mutation(changed)
        with pytest.raises(ValueError, match=message):
            read_denoising_result(_canonical(changed))


def test_canonical_result_rejects_stale_decision_object() -> None:
    scalar = _scalar_curves()
    candidates = _candidates()
    swaps = _swaps()
    decision = classify_denoising_result(scalar, candidates, swaps)
    stale = DenoisingDecision(
        terminal_class="resource-failure",
        selected_candidate=None,
        best_scalar_alpha=decision.best_scalar_alpha,
        candidate_passes=(False, False, False),
        reaches_95_percent=decision.reaches_95_percent,
        head_coadaptation_observed=decision.head_coadaptation_observed,
        paired_evidence=decision.paired_evidence,
    )
    with pytest.raises(ValueError, match="decision"):
        canonical_denoising_result_bytes(scalar, candidates, swaps, stale)
