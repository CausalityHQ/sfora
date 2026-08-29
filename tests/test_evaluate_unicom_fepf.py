from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
import math
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).parents[1] / "scripts" / "evaluate_unicom_fepf.py"
SPEC = importlib.util.spec_from_file_location("evaluate_unicom_fepf", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("mutation", (None, "path", "hash"))
def test_review13_public_evaluator_loads_separate_ranked_prefix_authority(
    tmp_path: Path, mutation: str | None
) -> None:
    ranked = [{
        "query_path": "query/a.jpg",
        "query_label": "item-a",
        "relevant_gallery_count": 1,
        "ap_at_r": 1.0,
        "query_sha256": "1" * 64,
        "complete_ranking_sha256": "2" * 64,
        "ranked_prefix": [{
            "gallery_index": 0,
            "gallery_path": "gallery/a.jpg",
            "gallery_label": "item-a",
            "score": 0.0,
            "correct": True,
        }],
    }]
    ranked_path = tmp_path / "evaluation-epoch-0004-ranked-prefix.json"
    ranked_payload = (json.dumps(ranked, indent=2, allow_nan=False) + "\n").encode()
    ranked_path.write_bytes(ranked_payload)
    binding = {
        "path": ranked_path.name,
        "sha256": hashlib.sha256(ranked_payload).hexdigest(),
        "bytes": len(ranked_payload),
    }
    if mutation == "path":
        binding["path"] = "foreign-ranked-prefix.json"
    elif mutation == "hash":
        binding["sha256"] = "0" * 64
    evaluation = {
        "epoch": 4,
        "ranked_prefix_evidence": binding,
        "query_records": [{"image_name": "query/a.jpg", "label": "item-a"}],
        "gallery_records": [{"image_name": "gallery/a.jpg", "label": "item-a"}],
        "geometry": {
            "input_dimension": 768,
            "coordinates": list(range(512)),
            "normalize_before": True,
            "ranking": "ascending_squared_euclidean",
        },
    }
    if mutation is None:
        observed = MODULE.load_ranked_query_observation(
            evaluation, evidence_root=tmp_path, expected_epoch=4
        )
        assert observed["query_evidence"][0]["top1_correct"] is True
    else:
        with pytest.raises(ValueError, match="ranked|query evidence"):
            MODULE.load_ranked_query_observation(
                evaluation, evidence_root=tmp_path, expected_epoch=4
            )


def _query_rows(top1: tuple[bool, ...]) -> list[dict[str, object]]:
    return [
        {"query_path": f"query/{index}.jpg", "top1_correct": correct, "ap_at_r": 0.5}
        for index, correct in enumerate(top1)
    ]


def _exploratory_arm(
    *,
    mode: str,
    maps: tuple[float, float, float, float],
    recalls: tuple[float, float, float, float],
    top1: tuple[bool, ...],
    initialization_seconds: float,
    profiled_step_wall: float,
) -> dict[str, object]:
    return {
        "mode": mode,
        "training_seed": 0,
        "holdout_seed": 0,
        "history": [
            {
                "epoch": epoch,
                "metrics": {"map_at_r": map_at_r, "recall_at_1": recall_at_1},
            }
            for epoch, map_at_r, recall_at_1 in zip(
                MODULE.EVALUATION_EPOCHS, maps, recalls, strict=True
            )
        ],
        "query_evidence": _query_rows(top1),
        "initialization_seconds": initialization_seconds,
        "optimizer_steps_per_epoch": 100,
        "profiled_step_wall": profiled_step_wall,
    }


def _promoting_exploratory_pair() -> tuple[dict[str, object], dict[str, object]]:
    control = _exploratory_arm(
        mode="imprinted",
        maps=(0.70, 0.72, 0.74, 0.75),
        recalls=(0.80, 0.81, 0.82, 0.83),
        top1=(False, False, False, False, False, True),
        initialization_seconds=100.0,
        profiled_step_wall=1.0,
    )
    candidate = _exploratory_arm(
        mode="fepf_mean",
        maps=(0.703, 0.74, 0.755, 0.761),
        recalls=(0.805, 0.82, 0.835, 0.841),
        top1=(True, True, True, True, True, False),
        initialization_seconds=130.0,
        profiled_step_wall=0.8,
    )
    return control, candidate


def test_review11_publication_stage_is_required_before_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments = [
        "--phase", "epoch4",
        "--sources", str(tmp_path / "sources.json"),
        "--sources-sha256", "1" * 64,
        "--sources-bytes", "1",
        "--evidence-root", str(tmp_path),
        "--output", str(tmp_path / "result.json"),
        "--temporary", str(tmp_path / ".result.json.tmp"),
        "--config", str(tmp_path / "config.json"),
        "--campaign-root", str(tmp_path),
    ]
    reached: list[Path] = []

    def forbidden(path: Path) -> object:
        reached.append(path)
        raise AssertionError("missing publication stage reached config/work")

    monkeypatch.setattr(MODULE, "_strict_json_file", forbidden)
    assert MODULE.main(arguments) == 2
    assert reached == []


def test_exploratory_promotes_only_when_every_registered_predicate_passes() -> None:
    control, candidate = _promoting_exploratory_pair()

    result = MODULE.evaluate_exploratory(control, candidate, structural_all=True)

    assert result["decision"] == "PROMOTE"
    assert result["clause"] == "PROMOTE"
    assert result["epoch4_delta_map"] == pytest.approx(0.003)
    assert result["epoch4_pass"] is True
    assert result["endpoint_delta_map"] == pytest.approx(0.011)
    assert result["endpoint_delta_r1"] == pytest.approx(0.011)
    assert result["gains"] == 5
    assert result["losses"] == 1
    assert result["control_first_attainment_epoch"] == 16
    assert result["candidate_first_attainment_epoch"] == 12
    assert result["control_profiled_compute"] == 1_700.0
    assert result["candidate_profiled_compute"] == 1_090.0
    assert all(result["predicates"].values())


@pytest.mark.parametrize(
    ("epoch4_delta", "decision", "passes"),
    ((0.002999, "CLOSE_EPOCH4", False), (0.003, "PROMOTE", True)),
)
def test_exploratory_preserves_separate_epoch4_controller_predicate(
    epoch4_delta: float, decision: str, passes: bool
) -> None:
    control, candidate = _promoting_exploratory_pair()
    candidate["history"][0]["metrics"]["map_at_r"] = 0.70 + epoch4_delta

    result = MODULE.evaluate_exploratory(control, candidate, structural_all=True)

    assert result["epoch4_pass"] is passes
    assert result["decision"] == decision
    assert result["clause"] == decision


@pytest.mark.parametrize("endpoint_delta", (0.0, 0.009999))
def test_exploratory_closes_marginal_endpoint_before_pareto_claim(
    endpoint_delta: float,
) -> None:
    control, candidate = _promoting_exploratory_pair()
    candidate["history"][-1]["metrics"]["map_at_r"] = 0.75 + endpoint_delta

    result = MODULE.evaluate_exploratory(control, candidate, structural_all=True)

    assert result["decision"] == "CLOSE_MARGINAL"
    assert result["clause"] == "CLOSE_MARGINAL"


@pytest.mark.parametrize("failure", ("recall", "gain_loss", "structural"))
def test_exploratory_closes_nonpareto_for_each_noncompute_predicate(
    failure: str,
) -> None:
    control, candidate = _promoting_exploratory_pair()
    structural_all = True
    if failure == "recall":
        candidate["history"][-1]["metrics"]["recall_at_1"] = 0.83
    elif failure == "gain_loss":
        candidate["query_evidence"] = _query_rows(
            (True, True, True, True, False, False)
        )
    else:
        structural_all = False

    result = MODULE.evaluate_exploratory(
        control, candidate, structural_all=structural_all
    )

    assert result["decision"] == "CLOSE_NONPARETO"
    assert result["clause"] == "CLOSE_NONPARETO"


def test_exploratory_first_attainment_is_discrete_and_right_censored() -> None:
    control, candidate = _promoting_exploratory_pair()
    control["history"][-1]["metrics"]["map_at_r"] = 0.80
    candidate["history"][-1]["metrics"]["map_at_r"] = 0.79

    result = MODULE.evaluate_exploratory(control, candidate, structural_all=True)

    assert result["candidate_first_attainment_epoch"] is None
    assert result["candidate_right_censored"] is True
    assert result["candidate_profiled_compute"] is None
    assert result["decision"] == "CLOSE_MARGINAL"


def test_exploratory_compute_includes_full_initialization_and_exact_tolerance() -> None:
    control, candidate = _promoting_exploratory_pair()
    control["initialization_seconds"] = 100.0
    control["profiled_step_wall"] = 1.0
    candidate["initialization_seconds"] = 80.0
    candidate["profiled_step_wall"] = (1.02 * 1_700.0 - 80.0) / (12 * 100)

    boundary = MODULE.evaluate_exploratory(
        control, candidate, structural_all=True
    )
    assert boundary["candidate_profiled_compute"] == pytest.approx(1_734.0)
    assert boundary["predicates"]["compute_within_1_02"] is True
    assert boundary["decision"] == "PROMOTE"

    candidate["profiled_step_wall"] += 1e-9
    outside = MODULE.evaluate_exploratory(control, candidate, structural_all=True)
    assert outside["predicates"]["compute_within_1_02"] is False
    assert outside["decision"] == "CLOSE_NONPARETO"


def test_exploratory_rejects_history_or_query_order_and_nonfinite_drift() -> None:
    control, candidate = _promoting_exploratory_pair()
    wrong_history = copy.deepcopy(candidate)
    wrong_history["history"][1], wrong_history["history"][2] = (
        wrong_history["history"][2],
        wrong_history["history"][1],
    )
    with pytest.raises(ValueError, match="history"):
        MODULE.evaluate_exploratory(control, wrong_history, structural_all=True)

    wrong_queries = copy.deepcopy(candidate)
    wrong_queries["query_evidence"].reverse()
    with pytest.raises(ValueError, match="query"):
        MODULE.evaluate_exploratory(control, wrong_queries, structural_all=True)

    nonfinite = copy.deepcopy(candidate)
    nonfinite["profiled_step_wall"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        MODULE.evaluate_exploratory(control, nonfinite, structural_all=True)


def _confirmation_arm(
    *,
    mode: str,
    training_seed: int,
    holdout_seed: int,
    endpoint_map: float,
    endpoint_r1: float,
    query_aps: tuple[float, ...],
    initialization_seconds: float = 100.0,
    profiled_step_wall: float = 1.0,
    peak_allocated_bytes: int = 1_000,
    peak_reserved_bytes: int = 2_000,
) -> dict[str, object]:
    return {
        "mode": mode,
        "training_seed": training_seed,
        "holdout_seed": holdout_seed,
        "history": [
            {
                "epoch": epoch,
                "metrics": {
                    "map_at_r": value,
                    "recall_at_1": endpoint_r1 - (16 - epoch) * 0.001,
                },
            }
            for epoch, value in zip(
                MODULE.EVALUATION_EPOCHS,
                (endpoint_map - 0.04, endpoint_map - 0.02, endpoint_map, endpoint_map),
                strict=True,
            )
        ],
        "query_evidence": [
            {
                "query_path": f"query/{index}.jpg",
                "top1_correct": True,
                "ap_at_r": value,
            }
            for index, value in enumerate(query_aps)
        ],
        "initialization_seconds": initialization_seconds,
        "optimizer_steps_per_epoch": 100,
        "profiled_step_wall": profiled_step_wall,
        "peak_allocated_bytes": peak_allocated_bytes,
        "peak_reserved_bytes": peak_reserved_bytes,
    }


def _passing_confirmation_pairs() -> tuple[dict[str, object], ...]:
    map_deltas = (0.012, 0.011, 0.010, 0.013, 0.014)
    recall_deltas = (0.006, 0.007, 0.008, 0.009, 0.010)
    rows = []
    for index, ((training_seed, holdout_seed), map_delta, recall_delta) in enumerate(
        zip(MODULE.CONFIRMATION_PAIRS, map_deltas, recall_deltas, strict=True)
    ):
        control_aps = (0.30 + index * 0.01, 0.50, 0.70 - index * 0.01)
        candidate_aps = tuple(value + 0.01 + index * 0.001 for value in control_aps)
        rows.append(
            {
                "training_seed": training_seed,
                "holdout_seed": holdout_seed,
                "control": _confirmation_arm(
                    mode="imprinted",
                    training_seed=training_seed,
                    holdout_seed=holdout_seed,
                    endpoint_map=0.75,
                    endpoint_r1=0.83,
                    query_aps=control_aps,
                ),
                "candidate": _confirmation_arm(
                    mode="fepf_mean",
                    training_seed=training_seed,
                    holdout_seed=holdout_seed,
                    endpoint_map=0.75 + map_delta,
                    endpoint_r1=0.83 + recall_delta,
                    query_aps=candidate_aps,
                ),
                "structural_equal": True,
            }
        )
    return tuple(rows)


def _independent_query_bootstrap(
    pair_query_deltas: tuple[tuple[float, ...], ...],
) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(MODULE.QUERY_BOOTSTRAP_SEED))
    result = np.empty(MODULE.QUERY_BOOTSTRAP_REPLICATES, dtype=np.float64)
    for replicate in range(MODULE.QUERY_BOOTSTRAP_REPLICATES):
        selected_pairs = generator.integers(0, 5, size=5)
        selected_means = []
        for pair_index in selected_pairs:
            values = pair_query_deltas[int(pair_index)]
            selected_queries = generator.integers(0, len(values), size=len(values))
            selected_means.append(
                math.fsum(values[int(index)] for index in selected_queries)
                / len(values)
            )
        result[replicate] = math.fsum(selected_means) / 5
    return result


def test_t_bound_uses_literal_and_float64_sample_standard_deviation() -> None:
    values = (0.012, 0.011, 0.010, 0.013, 0.014)
    mean = math.fsum(values) / 5
    sample_std = math.sqrt(math.fsum((value - mean) ** 2 for value in values) / 4)
    expected = mean - 2.131846786326649 * sample_std / math.sqrt(5)

    observed = MODULE.one_sided_t_lower_bound(values)

    assert observed == {
        "mean": mean,
        "sample_std": sample_std,
        "lower_bound": expected,
    }


def test_bootstrap_uses_exact_pcg64_nested_order_array_hash_and_linear_quantiles() -> None:
    pairs = (
        (0.01, 0.02, -0.01),
        (0.03, 0.00),
        (0.015, 0.025, 0.035, -0.005),
        (0.01,),
        (0.02, 0.04, 0.00),
    )
    expected = _independent_query_bootstrap(pairs)

    observed = MODULE.query_bootstrap(pairs)

    assert observed["bit_generator"] == "PCG64"
    assert observed["seed"] == 20_260_829
    assert observed["replicates"] == 10_000
    assert observed["quantile_method"] == "linear"
    assert np.array_equal(np.asarray(observed["values"], dtype=np.float64), expected)
    assert observed["values_sha256"] == hashlib.sha256(
        expected.tobytes(order="C")
    ).hexdigest()
    assert observed["interval"] == list(
        np.quantile(expected, (0.025, 0.975), method="linear")
    )


def test_confirmation_requires_exact_five_pairs_and_passes_all_frozen_gates() -> None:
    pairs = _passing_confirmation_pairs()

    result = MODULE.evaluate_confirmation(pairs)

    assert result["decision"] == "CONFIRM"
    assert result["clause"] == "CONFIRM"
    assert [(row["training_seed"], row["holdout_seed"]) for row in result["pairs"]] == list(
        MODULE.CONFIRMATION_PAIRS
    )
    assert result["statistics"]["map"]["mean"] == pytest.approx(0.012)
    assert result["statistics"]["map"]["median"] == pytest.approx(0.012)
    assert result["statistics"]["map"]["leave_one_out_means"] == pytest.approx(
        (0.012, 0.01225, 0.0125, 0.01175, 0.0115)
    )
    assert result["statistics"]["recall_at_1"]["mean"] == pytest.approx(0.008)
    assert all(result["predicates"].values())
    assert len(result["bootstrap"]["values"]) == 10_000


@pytest.mark.parametrize("mutation", ("missing", "reordered", "wrong_identity"))
def test_confirmation_rejects_missing_reordered_or_substituted_pairs(
    mutation: str,
) -> None:
    pairs = list(_passing_confirmation_pairs())
    if mutation == "missing":
        pairs.pop()
    elif mutation == "reordered":
        pairs[0], pairs[1] = pairs[1], pairs[0]
    else:
        pairs[0]["training_seed"] = 12

    with pytest.raises(ValueError, match="pair order"):
        MODULE.evaluate_confirmation(tuple(pairs))


def test_confirmation_closes_when_primary_statistics_fail() -> None:
    pairs = list(_passing_confirmation_pairs())
    deltas = (0.001, 0.001, 0.001, 0.001, 0.060)
    for pair, delta in zip(pairs, deltas, strict=True):
        pair["candidate"]["history"][-1]["metrics"]["map_at_r"] = 0.75 + delta

    result = MODULE.evaluate_confirmation(tuple(pairs))

    assert result["statistics"]["map"]["mean"] >= 0.010
    assert result["statistics"]["map"]["t_lower_bound"] <= 0.0
    assert result["predicates"]["map_t_lower_positive"] is False
    assert result["decision"] == "CLOSE_CONFIRMATION"


@pytest.mark.parametrize(
    "failure", ("recall", "compute", "step", "allocated", "reserved", "structural")
)
def test_confirmation_closes_for_resource_recall_compute_or_structure_failure(
    failure: str,
) -> None:
    pairs = list(_passing_confirmation_pairs())
    if failure == "recall":
        pairs[0]["candidate"]["history"][-1]["metrics"]["recall_at_1"] = 0.83
    elif failure == "compute":
        pairs[0]["candidate"]["initialization_seconds"] = 2_000.0
    elif failure == "structural":
        pairs[0]["structural_equal"] = False
    else:
        key = {
            "step": "profiled_step_wall",
            "allocated": "peak_allocated_bytes",
            "reserved": "peak_reserved_bytes",
        }[failure]
        for pair in pairs:
            changed = pair["control"][key] * 1.03
            pair["candidate"][key] = (
                int(changed) if failure in {"allocated", "reserved"} else changed
            )

    result = MODULE.evaluate_confirmation(tuple(pairs))

    assert not all(result["predicates"].values())
    assert result["decision"] == "CLOSE_CONFIRMATION"


def test_confirmation_allows_different_authenticated_values_when_structure_matches() -> None:
    pairs = list(_passing_confirmation_pairs())
    for index, pair in enumerate(pairs):
        pair["control_value_sha256"] = f"{index + 1:064x}"
        pair["candidate_value_sha256"] = f"{index + 11:064x}"

    result = MODULE.evaluate_confirmation(tuple(pairs))

    assert result["predicates"]["cross_arm_structure_equal"] is True
    assert result["decision"] == "CONFIRM"


def _write_canonical_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sources_authority(
    tmp_path: Path, sources: list[dict[str, object]]
) -> dict[str, object]:
    path = tmp_path / "sources.json"
    payload = (json.dumps(sources, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("test sources authority already has different bytes")
    if not path.exists():
        path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }


def _strict_reload_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[dict[str, object]], dict[str, object], dict[str, object]]:
    sources = []
    for index, pair in enumerate(_passing_confirmation_pairs()):
        relative = f"pair-{index}.json"
        _write_canonical_json(tmp_path / relative, pair)
        sources.append(
            {
                "training_seed": pair["training_seed"],
                "holdout_seed": pair["holdout_seed"],
                "control_root": relative,
                "candidate_root": f"candidate-{index}",
                "quality_profiles": [
                    f"profile-{index}-control-0.json",
                    f"profile-{index}-candidate-0.json",
                    f"profile-{index}-candidate-1.json",
                    f"profile-{index}-control-1.json",
                ],
            }
        )
    config = {
        "path": str((tmp_path / "config.json").resolve()),
        "sha256": "a" * 64,
        "bytes": 10,
    }

    def reload_pairs(*, observed_sources, evidence_root, phase):
        assert phase == "confirmation"
        assert evidence_root == tmp_path
        pairs = []
        entries = []
        for source in observed_sources:
            path = evidence_root / source["control_root"]
            payload = path.read_bytes()
            pairs.append(json.loads(payload))
            entries.append(
                {
                    "role": "test.pair",
                    "identity": source["control_root"],
                    "path": path,
                }
            )
        return tuple(pairs), config, MODULE.build_evidence_manifest(entries)

    monkeypatch.setattr(MODULE, "_reload_registered_pairs", reload_pairs, raising=False)
    return sources, config, _sources_authority(tmp_path, sources)


def test_strict_reload_recomputes_from_external_bytes_not_in_memory_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    persisted = json.loads((tmp_path / "pair-0.json").read_bytes())
    persisted["candidate"]["initialization_seconds"] += 1.0
    _write_canonical_json(tmp_path / "pair-0.json", persisted)

    with pytest.raises(ValueError, match="recomputation"):
        MODULE.validate_fepf_result(
            result, tmp_path, sources_authority=sources_authority
        )


def test_strict_reload_rejects_exploratory_query_prefix_gain_loss_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control, candidate = _promoting_exploratory_pair()
    pair = {
        "training_seed": 0,
        "holdout_seed": 0,
        "control": control,
        "candidate": candidate,
        "structural_equal": True,
    }
    pair_path = tmp_path / "exploratory-pair.json"
    _write_canonical_json(pair_path, pair)
    sources = [
        {
            "training_seed": 0,
            "holdout_seed": 0,
            "control_root": pair_path.name,
            "candidate_root": "candidate",
            "quality_profiles": ["c0.json", "f0.json", "f1.json", "c1.json"],
        }
    ]
    config = {
        "path": str((tmp_path / "config.json").resolve()),
        "sha256": "a" * 64,
        "bytes": 10,
    }

    def reload_pairs(*, observed_sources, evidence_root, phase):
        assert phase == "exploratory"
        path = evidence_root / observed_sources[0]["control_root"]
        payload = path.read_bytes()
        manifest = MODULE.build_evidence_manifest(
            [
                {
                    "role": "test.exploratory_pair",
                    "identity": observed_sources[0]["control_root"],
                    "path": path,
                }
            ]
        )
        return (json.loads(payload),), config, manifest

    monkeypatch.setattr(MODULE, "_reload_registered_pairs", reload_pairs)
    sources_authority = _sources_authority(tmp_path, sources)
    result = MODULE.build_fepf_result(
        phase="exploratory", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    changed = json.loads(pair_path.read_bytes())
    changed["candidate"]["query_evidence"][4]["top1_correct"] = False
    _write_canonical_json(pair_path, changed)

    with pytest.raises(ValueError, match="recomputation"):
        MODULE.validate_fepf_result(
            result, tmp_path, sources_authority=sources_authority
        )


@pytest.mark.parametrize(
    "mutation",
    (
        "status",
        "clause",
        "sample_std",
        "t_bound",
        "bootstrap_array_with_rehashed_bytes",
        "bootstrap_hash",
        "pair_order",
        "source_binding",
        "config_binding",
    ),
)
def test_strict_reload_rejects_recursive_result_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    sources, _config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    changed = copy.deepcopy(result)
    if mutation == "status":
        changed["status"] = "CLOSE_CONFIRMATION"
    elif mutation == "clause":
        changed["clause"] = "CLOSE_CONFIRMATION"
    elif mutation == "sample_std":
        changed["decision"]["statistics"]["map"]["sample_std"] += 1e-6
    elif mutation == "t_bound":
        changed["decision"]["statistics"]["map"]["t_lower_bound"] += 1e-6
    elif mutation == "bootstrap_array_with_rehashed_bytes":
        changed["decision"]["bootstrap"]["values"][0] += 1e-6
        values = np.asarray(changed["decision"]["bootstrap"]["values"], dtype=np.float64)
        changed["decision"]["bootstrap"]["values_sha256"] = hashlib.sha256(
            values.tobytes(order="C")
        ).hexdigest()
    elif mutation == "bootstrap_hash":
        changed["decision"]["bootstrap"]["values_sha256"] = "b" * 64
    elif mutation == "pair_order":
        changed["sources"][0], changed["sources"][1] = (
            changed["sources"][1],
            changed["sources"][0],
        )
    elif mutation == "source_binding":
        changed["evaluator_sha256"] = "c" * 64
    else:
        changed["config"]["sha256"] = "d" * 64

    with pytest.raises(ValueError):
        MODULE.validate_fepf_result(
            changed, tmp_path, sources_authority=sources_authority
        )


@pytest.mark.parametrize(
    "mutation", ("query_prefix", "descriptor_preimage", "profile_step", "memory", "structure")
)
def test_strict_reload_rejects_mutated_external_scientific_or_structural_input(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    sources, _config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    pair = json.loads((tmp_path / "pair-0.json").read_bytes())
    if mutation == "query_prefix":
        pair["candidate"]["query_evidence"][0]["top1_correct"] = False
    elif mutation == "descriptor_preimage":
        pair["candidate"]["query_evidence"][0]["ap_at_r"] += 1e-4
    elif mutation == "profile_step":
        pair["candidate"]["profiled_step_wall"] += 1e-4
    elif mutation == "memory":
        pair["candidate"]["peak_reserved_bytes"] += 1
    else:
        pair["structural_equal"] = False
    _write_canonical_json(tmp_path / "pair-0.json", pair)

    with pytest.raises(ValueError, match="recomputation"):
        MODULE.validate_fepf_result(
            result, tmp_path, sources_authority=sources_authority
        )


def test_atomic_publication_rejects_preexisting_output_or_temp_without_modification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    output = tmp_path / "result.json"
    temporary = tmp_path / ".result.json.tmp"
    output.write_bytes(b"owner\n")
    with pytest.raises(FileExistsError):
        MODULE.write_fepf_result_atomic(
            result, output, temporary, tmp_path, sources_authority=sources_authority
        )
    assert output.read_bytes() == b"owner\n"
    assert not temporary.exists()

    output.unlink()
    temporary.write_bytes(b"temp-owner\n")
    with pytest.raises(FileExistsError):
        MODULE.write_fepf_result_atomic(
            result, output, temporary, tmp_path, sources_authority=sources_authority
        )
    assert temporary.read_bytes() == b"temp-owner\n"
    assert not output.exists()


def test_atomic_publication_uses_link_no_replace_against_racing_writer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    output = tmp_path / "result.json"
    temporary = tmp_path / ".result.json.tmp"
    publication = importlib.import_module("sfora.atomic_publication")
    real_link = publication._link_fd_noreplace

    def racing_link(descriptor: int, directory: int, name: str) -> None:
        output.write_bytes(b"racing-owner\n")
        real_link(descriptor, directory, name)

    monkeypatch.setattr(publication, "_link_fd_noreplace", racing_link)
    with pytest.raises(FileExistsError):
        MODULE.write_fepf_result_atomic(
            result, output, temporary, tmp_path, sources_authority=sources_authority
        )

    assert output.read_bytes() == b"racing-owner\n"
    assert not temporary.exists()


def test_atomic_publication_fsyncs_both_directory_transitions_and_distinct_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    output = tmp_path / "result.json"
    temporary = tmp_path / ".result.json.tmp"
    real_fsync = os.fsync
    directory_fsyncs = 0

    def observed_fsync(descriptor: int) -> None:
        nonlocal directory_fsyncs
        if os.path.isdir(f"/proc/self/fd/{descriptor}"):
            directory_fsyncs += 1
        real_fsync(descriptor)

    monkeypatch.setattr(MODULE.os, "fsync", observed_fsync)
    published = MODULE.write_fepf_result_atomic(
        result, output, temporary, tmp_path, sources_authority=sources_authority
    )

    assert published == result
    assert directory_fsyncs == 2
    assert not temporary.exists()
    MODULE.validate_fepf_result(
        json.loads(output.read_bytes()),
        tmp_path,
        sources_authority=sources_authority,
    )


def _independent_checkpoint_signature(
    state: dict[str, torch.Tensor], parameter_names: set[str], descriptor_sha256: str
) -> dict[str, object]:
    aggregate = hashlib.sha256()
    rows = []
    total = 0
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        payload = tensor.numpy().tobytes(order="C")
        row = {
            "name": name,
            "kind": "parameter" if name in parameter_names else "buffer",
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "element_size": tensor.element_size(),
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        metadata = json.dumps(
            {key: row[key] for key in tuple(row)[:-1]}, separators=(",", ":")
        ).encode()
        aggregate.update(len(metadata).to_bytes(8, "big"))
        aggregate.update(metadata)
        aggregate.update(len(payload).to_bytes(8, "big"))
        aggregate.update(payload)
        rows.append(row)
        total += len(payload)
    return {
        "schema": "unicom-inference-signature-v1",
        "tensors": rows,
        "total_bytes": total,
        "aggregate_sha256": aggregate.hexdigest(),
        "descriptor_dtype": "torch.float32",
        "descriptor_dimension": 512,
        "descriptor_sha256": descriptor_sha256,
        "operations": list(MODULE.INFERENCE_OPERATIONS),
    }


def test_checkpoint_signature_rebuilds_parameter_and_buffer_values_not_receipt_hashes() -> None:
    state = {
        "layer.running": torch.tensor([3], dtype=torch.int64),
        "layer.weight": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
    }
    descriptor_sha256 = "d" * 64
    expected = _independent_checkpoint_signature(
        state, {"layer.weight"}, descriptor_sha256
    )
    structure = {
        "schema": "unicom-fepf-structure-v1",
        "tensors": [
            {
                key: row[key]
                for key in (
                    "name",
                    "kind",
                    "shape",
                    "dtype",
                    "numel",
                    "element_size",
                    "bytes",
                )
            }
            for row in expected["tensors"]
        ],
        "classifier": {
            "shape": [2, 2],
            "dtype": "torch.float32",
            "numel": 4,
            "element_size": 4,
            "bytes": 16,
        },
        "operations": list(MODULE.INFERENCE_OPERATIONS),
    }
    checkpoint = {
        "model": state,
        "classifier": torch.zeros((2, 2), dtype=torch.float32),
        "ema": None,
    }

    rebuilt = MODULE.checkpoint_inference_signature(
        checkpoint,
        structural_inventory=structure,
        descriptor_sha256=descriptor_sha256,
    )

    assert rebuilt == expected
    forged = copy.deepcopy(expected)
    forged["tensors"][0]["sha256"] = "e" * 64
    forged["aggregate_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="checkpoint inference"):
        MODULE.require_same_arm_checkpoint_signature(
            checkpoint,
            recorded=forged,
            structural_inventory=structure,
            descriptor_sha256=descriptor_sha256,
        )


def test_checkpoint_signature_accepts_ordered_state_with_scalar_buffer() -> None:
    state = OrderedDict(
        (
            ("layer.num_batches_tracked", torch.tensor(560_388, dtype=torch.int64)),
            ("layer.weight", torch.tensor([[1.0, 2.0]], dtype=torch.float32)),
        )
    )
    structure = {
        "schema": "unicom-fepf-structure-v1",
        "tensors": [
            {
                "name": name,
                "kind": "parameter" if name == "layer.weight" else "buffer",
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "numel": value.numel(),
                "element_size": value.element_size(),
                "bytes": value.numel() * value.element_size(),
            }
            for name, value in state.items()
        ],
        "classifier": {
            "shape": [2, 2],
            "dtype": "torch.float32",
            "numel": 4,
            "element_size": 4,
            "bytes": 16,
        },
        "operations": list(MODULE.INFERENCE_OPERATIONS),
    }
    checkpoint = {
        "model": state,
        "classifier": torch.zeros((2, 2), dtype=torch.float32),
        "ema": None,
    }

    signature = MODULE.checkpoint_inference_signature(
        checkpoint,
        structural_inventory=structure,
        descriptor_sha256="d" * 64,
    )

    scalar = signature["tensors"][0]
    expected = torch.tensor(560_388, dtype=torch.int64).numpy().tobytes(order="C")
    assert scalar["shape"] == []
    assert scalar["bytes"] == len(expected) == 8
    assert scalar["sha256"] == hashlib.sha256(expected).hexdigest()


def _complete_query_observation(
    *, label: str = "id-a", relevant: int = 2, gallery_label: str = "id-a"
) -> dict[str, object]:
    return {
        "history": [
            {"epoch": epoch, "metrics": {"map_at_r": 0.7, "recall_at_1": 0.8}}
            for epoch in MODULE.EVALUATION_EPOCHS
        ],
        "query_evidence": [
            {
                "query_path": "query/a.jpg",
                "query_label": label,
                "relevant_gallery_count": relevant,
                "top1_correct": True,
                "ap_at_r": 0.5,
            }
        ],
        "query_inventory": [["query/a.jpg", label, relevant]],
        "gallery_inventory": [["gallery/a.jpg", gallery_label]],
        "gallery_inventory_sha256": "a" * 64,
        "geometry": {
            "input_dimension": 768,
            "coordinates": list(range(512)),
            "normalize_before": True,
            "ranking": "ascending_squared_euclidean",
        },
        "initialization_seconds": 1.0,
        "optimizer_steps_per_epoch": 10,
        "profiled_step_wall": 1.0,
        "peak_allocated_bytes": 100,
        "peak_reserved_bytes": 200,
    }


@pytest.mark.parametrize("drift", ("label", "relevant", "gallery", "geometry"))
def test_paired_query_deltas_require_complete_query_gallery_geometry_unit(
    drift: str,
) -> None:
    control = _complete_query_observation()
    candidate = copy.deepcopy(control)
    if drift == "label":
        candidate["query_evidence"][0]["query_label"] = "id-b"
        candidate["query_inventory"][0][1] = "id-b"
    elif drift == "relevant":
        candidate["query_evidence"][0]["relevant_gallery_count"] = 1
        candidate["query_inventory"][0][2] = 1
    elif drift == "gallery":
        candidate["gallery_inventory"][0][1] = "id-b"
        candidate["gallery_inventory_sha256"] = "b" * 64
    else:
        candidate["geometry"]["normalize_before"] = False

    with pytest.raises(ValueError, match="paired query unit"):
        MODULE.paired_query_deltas(control, candidate)


@pytest.mark.parametrize(
    ("delta", "decision"),
    ((0.002999, "CLOSE_EPOCH4"), (0.003, "PASS_TO_RESUME")),
)
def test_epoch4_external_phase_has_no_profiles_and_explicit_resume_decision(
    delta: float, decision: str
) -> None:
    control, candidate = _promoting_exploratory_pair()
    control["history"] = control["history"][:1]
    candidate["history"] = candidate["history"][:1]
    candidate["history"][0]["metrics"]["map_at_r"] = 0.70 + delta

    observed = MODULE.evaluate_epoch4(control, candidate, structural_all=True)

    assert observed["decision"] == decision
    assert observed["clause"] == decision
    sources = [
        {
            "training_seed": 0,
            "holdout_seed": 0,
            "control_root": "control-stop4",
            "candidate_root": "candidate-stop4",
            "quality_profiles": [],
            "config": {
                "path": "/registered/config.json",
                "sha256": "a" * 64,
                "bytes": 1,
            },
        }
    ]
    assert MODULE._validate_source_inventory("epoch4", sources) == sources


def test_evidence_manifest_is_domain_separated_ordered_transitive_and_unique(
    tmp_path: Path,
) -> None:
    first = tmp_path / "run.json"
    second = tmp_path / "descriptor.npy"
    first.write_bytes(b"run\n")
    second.write_bytes(b"descriptor\n")
    entries = [
        {"role": "control.run", "identity": "seed7", "path": first},
        {"role": "control.query_descriptor", "identity": "seed7", "path": second},
    ]

    manifest = MODULE.build_evidence_manifest(entries)

    assert [row["role"] for row in manifest["entries"]] == [
        "control.run",
        "control.query_descriptor",
    ]
    assert manifest["entries"][0]["bytes"] == len(b"run\n")
    assert manifest["entries"][1]["sha256"] == hashlib.sha256(
        b"descriptor\n"
    ).hexdigest()
    assert len(manifest["sha256"]) == 64
    with pytest.raises(ValueError, match="duplicate evidence"):
        MODULE.build_evidence_manifest(entries + [dict(entries[0])])


def _profile_process(started: int, finished: int, *, token: str) -> dict[str, object]:
    return {
        "started_unix_ns": started,
        "finished_unix_ns": finished,
        "checkpoint": {"sha256": token * 64},
        "run_receipt": {"sha256": ("f" if token == "a" else token) * 64},
    }


def test_quality_profiles_are_four_fresh_nonoverlapping_processes_in_order() -> None:
    profiles = tuple(
        _profile_process(start, start + 5, token=token)
        for start, token in zip((10, 20, 30, 40), "abcd", strict=True)
    )
    MODULE.validate_profile_process_order(profiles)

    copied = list(copy.deepcopy(profiles))
    copied[3] = copy.deepcopy(copied[0])
    with pytest.raises(ValueError, match="fresh profile"):
        MODULE.validate_profile_process_order(tuple(copied))

    overlap = list(copy.deepcopy(profiles))
    overlap[1]["started_unix_ns"] = 14
    with pytest.raises(ValueError, match="chronology"):
        MODULE.validate_profile_process_order(tuple(overlap))


def test_fully_observed_structural_failure_builds_schema_valid_invalid_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    original = MODULE._reload_registered_pairs

    def structural_reload(**kwargs):
        pairs, observed_config, evidence_sha256 = original(**kwargs)
        changed = list(copy.deepcopy(pairs))
        changed[0]["structural_equal"] = False
        return tuple(changed), observed_config, evidence_sha256

    monkeypatch.setattr(MODULE, "_reload_registered_pairs", structural_reload)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )

    assert result["status"] == "INVALID"
    assert result["clause"] == "INVALID_STRUCTURAL_PANEL"
    assert result["config"] == config
    MODULE.validate_fepf_result(
        result, tmp_path, sources_authority=sources_authority
    )


def test_atomic_cleanup_preserves_substituted_temp_after_link_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    output = tmp_path / "result.json"
    temporary = tmp_path / ".result.json.tmp"

    publication = importlib.import_module("sfora.atomic_publication")

    def substitute_then_fail(_descriptor: int, _directory: int, _name: str) -> None:
        temporary.write_bytes(b"racer-temp\n")
        raise OSError("injected link failure")

    monkeypatch.setattr(publication, "_link_fd_noreplace", substitute_then_fail)
    with pytest.raises(OSError, match="injected link failure"):
        MODULE.write_fepf_result_atomic(
            result, output, temporary, tmp_path, sources_authority=sources_authority
        )

    assert temporary.read_bytes() == b"racer-temp\n"
    assert not output.exists()


def test_atomic_cleanup_preserves_substituted_temp_before_persisted_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    output = tmp_path / "result.json"
    temporary = tmp_path / ".result.json.tmp"
    publication = importlib.import_module("sfora.atomic_publication")
    real_read = publication._pread_all
    reads = 0

    def substitute_before_read(descriptor: int) -> bytes:
        nonlocal reads
        reads += 1
        if reads == 1:
            temporary.write_bytes(b"racer-before-validation\n")
            raise ValueError("strict JSON injected failure")
        return real_read(descriptor)

    monkeypatch.setattr(publication, "_pread_all", substitute_before_read)
    with pytest.raises(ValueError, match="strict JSON"):
        MODULE.write_fepf_result_atomic(
            result, output, temporary, tmp_path, sources_authority=sources_authority
        )

    assert temporary.read_bytes() == b"racer-before-validation\n"
    assert not output.exists()


def test_atomic_cleanup_removes_owned_output_when_link_succeeds_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources, _config, sources_authority = _strict_reload_fixture(tmp_path, monkeypatch)
    result = MODULE.build_fepf_result(
        phase="confirmation", sources=sources, sources_authority=sources_authority,
        evidence_root=tmp_path
    )
    output = tmp_path / "result.json"
    temporary = tmp_path / ".result.json.tmp"
    publication = importlib.import_module("sfora.atomic_publication")
    real_link = publication._link_fd_noreplace

    def link_then_fail(descriptor: int, directory: int, name: str) -> None:
        real_link(descriptor, directory, name)
        raise OSError("injected post-link failure")

    monkeypatch.setattr(publication, "_link_fd_noreplace", link_then_fail)
    with pytest.raises(OSError, match="post-link failure"):
        MODULE.write_fepf_result_atomic(
            result, output, temporary, tmp_path, sources_authority=sources_authority
        )

    assert not temporary.exists()
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation", ("buffer_value", "ema_kind_rebind", "config_substitution", "selector")
)
def test_real_reload_chain_rejects_recursive_authority_rebinding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    class TrainerAuthority:
        @staticmethod
        def validate_training_run_receipt_v2(_receipt, *, evidence_root):
            assert evidence_root.is_dir()

        @staticmethod
        def validate_fepf_result(_history, evidence_root):
            assert evidence_root.is_dir()

        @staticmethod
        def require_cross_arm_inference_equality(left, right):
            left_structure = [
                {key: row[key] for key in ("name", "kind", "shape", "dtype", "numel")}
                for row in left["tensors"]
            ]
            right_structure = [
                {key: row[key] for key in ("name", "kind", "shape", "dtype", "numel")}
                for row in right["tensors"]
            ]
            if left_structure != right_structure:
                raise ValueError("structure differs")

    monkeypatch.setattr(
        MODULE, "_authority_modules", lambda: (TrainerAuthority, object())
    )
    descriptor_sha256 = "d" * 64
    structural_inventory = {
        "schema": "unicom-fepf-structure-v1",
        "tensors": [
            {
                "name": "running",
                "kind": "buffer",
                "shape": [1],
                "dtype": "torch.int64",
                "numel": 1,
                "element_size": 8,
                "bytes": 8,
            },
            {
                "name": "weight",
                "kind": "parameter",
                "shape": [1, 2],
                "dtype": "torch.float32",
                "numel": 2,
                "element_size": 4,
                "bytes": 8,
            },
        ],
        "classifier": {
            "shape": [2, 2],
            "dtype": "torch.float32",
            "numel": 4,
            "element_size": 4,
            "bytes": 16,
        },
        "operations": list(MODULE.INFERENCE_OPERATIONS),
    }
    config_path = tmp_path / "config.json"
    _write_canonical_json(
        config_path,
        {"registered": True, "fepf_inference_structure": structural_inventory},
    )
    config = {
        "path": str(config_path.resolve()),
        "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "bytes": config_path.stat().st_size,
    }

    def binding(path: Path) -> dict[str, object]:
        return {
            "root": "current",
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }

    def arm_root(name: str, mode: str, value: float, map_at_r: float) -> Path:
        root = tmp_path / name
        root.mkdir()
        query_descriptor = root / "query.npy"
        gallery_descriptor = root / "gallery.npy"
        query_descriptor.write_bytes(b"query-descriptor\n")
        gallery_descriptor.write_bytes(b"gallery-descriptor\n")
        evaluation_path = root / "evaluation-epoch-0004.json"
        ranked_rows = [
            {
                "query_path": "query/a.jpg",
                "query_label": "id-a",
                "relevant_gallery_count": 1,
                "ap_at_r": map_at_r,
                "query_sha256": "1" * 64,
                "complete_ranking_sha256": "2" * 64,
                "ranked_prefix": [{
                    "gallery_index": 0,
                    "gallery_path": "gallery/a.jpg",
                    "gallery_label": "id-a",
                    "score": 0.0,
                    "correct": True,
                }],
            }
        ]
        ranked_path = root / "evaluation-epoch-0004-ranked-prefix.json"
        _write_canonical_json(ranked_path, ranked_rows)
        evaluation = {
            "epoch": 4,
            "geometry": {
                "input_dimension": 768,
                "coordinates": list(range(512)),
                "normalize_before": True,
                "ranking": "ascending_squared_euclidean",
            },
            "query_descriptors": {
                "path": query_descriptor.name,
                "sha256": hashlib.sha256(query_descriptor.read_bytes()).hexdigest(),
                "bytes": query_descriptor.stat().st_size,
            },
            "gallery_descriptors": {
                "path": gallery_descriptor.name,
                "sha256": hashlib.sha256(gallery_descriptor.read_bytes()).hexdigest(),
                "bytes": gallery_descriptor.stat().st_size,
            },
            "query_records": [{"image_name": "query/a.jpg", "label": "id-a"}],
            "gallery_records": [
                {"image_name": "gallery/a.jpg", "label": "id-a"}
            ],
            "ranked_prefix_evidence": {
                "path": ranked_path.name,
                "sha256": hashlib.sha256(ranked_path.read_bytes()).hexdigest(),
                "bytes": ranked_path.stat().st_size,
            },
            "evaluation_signature": {"descriptor_sha256": descriptor_sha256},
        }
        _write_canonical_json(evaluation_path, evaluation)
        checkpoint_path = root / "epoch-0004.pt"
        model = {
            "running": torch.tensor([4], dtype=torch.int64),
            "weight": torch.tensor([[value, value + 1.0]], dtype=torch.float32),
        }
        checkpoint = {
            "model": model,
            "classifier": torch.zeros((2, 2), dtype=torch.float32),
            "ema": {
                "backbone": {"weight": model["weight"].clone()},
                "classifier": torch.zeros((2, 2), dtype=torch.float32),
                "decay": 0.999,
                "updates": 1,
            },
        }
        torch.save(checkpoint, checkpoint_path)
        signature = MODULE.checkpoint_inference_signature(
            checkpoint,
            structural_inventory=structural_inventory,
            descriptor_sha256=descriptor_sha256,
        )
        history_path = root / "history.json"
        history = [
            {
                "epoch": epoch,
                "train": {"steps": 2},
                "metrics": (
                    {"map_at_r": map_at_r, "recall_at_1": 0.8}
                    if epoch == 4
                    else None
                ),
            }
            for epoch in range(1, 5)
        ]
        _write_canonical_json(history_path, history)
        protocol = {
            "epochs": 16,
            "batch_size": 2,
            "workers": 0,
            "learning_rate": 1e-5,
            "classifier_learning_rate": 1e-4,
            "margin": 0.3,
            "scale": 32.0,
            "objective": "arcface",
            "selected_features": 512,
            "evaluation_features": 512,
            "eval_every": 4,
            "checkpoint_every": 4,
            "max_steps": None,
            "bf16": False,
            "trainer_sha256": "a" * 64,
            "initial_checkpoint_sha256": "b" * 64,
        }
        initialization_path = root / "initialization-receipt.json"
        initialization = {
            "schema": "initialization-receipt-v2",
            "mode": mode,
            "training_seed": 0,
            "holdout_fraction": 0.2,
            "holdout_seed": 0,
            "source_sha256": protocol["trainer_sha256"],
            "checkpoint_sha256": protocol["initial_checkpoint_sha256"],
            "config_sha256": config["sha256"],
            "schedule_sha256": MODULE._schedule_sha256(protocol),
            "row_norm_rtol": 2e-6,
            "row_norm_atol": 2e-7,
            "initialization_seconds": 1.0,
        }
        _write_canonical_json(initialization_path, initialization)
        run_receipt = {
            "mode": mode,
            "training_seed": 0,
            "holdout_fraction": 0.2,
            "holdout_seed": 0,
            "stop_after_epoch": 4,
            "training_protocol": protocol,
            "parent_evidence_root": None,
            "parent_run_receipt": None,
            "initialization_receipt": binding(initialization_path),
            "history": binding(history_path),
            "checkpoints": [{"epoch": 4, **binding(checkpoint_path)}],
            "evaluations": [{"epoch": 4, **binding(evaluation_path)}],
            "inference_signature": signature,
        }
        _write_canonical_json(root / "run-receipt.json", run_receipt)
        return root

    control_root = arm_root("control-stop4", "imprinted", 1.0, 0.70)
    candidate_root = arm_root("candidate-stop4", "fepf_mean", 2.0, 0.71)
    sources = [
        {
            "training_seed": 0,
            "holdout_seed": 0,
            "control_root": control_root.name,
            "candidate_root": candidate_root.name,
            "quality_profiles": [],
            "config": config,
        }
    ]
    sources_path = tmp_path / "sources.json"
    _write_canonical_json(sources_path, sources)
    sources_authority = {
        "path": str(sources_path.resolve()),
        "sha256": hashlib.sha256(sources_path.read_bytes()).hexdigest(),
        "bytes": sources_path.stat().st_size,
    }

    result = MODULE.build_fepf_result(
        phase="epoch4",
        sources=sources,
        sources_authority=sources_authority,
        evidence_root=tmp_path,
    )

    roles = [entry["role"] for entry in result["evidence_manifest"]["entries"]]
    assert result["status"] == "PASS_TO_RESUME"
    assert any("checkpoint.epoch4" in role for role in roles)
    assert any("query_descriptors.epoch4" in role for role in roles)
    assert any(role.endswith(".config") for role in roles)

    if mutation in {"buffer_value", "ema_kind_rebind"}:
        checkpoint_path = candidate_root / "epoch-0004.pt"
        changed = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        run_path = candidate_root / "run-receipt.json"
        run_receipt = json.loads(run_path.read_bytes())
        if mutation == "buffer_value":
            changed["model"]["running"] += 1
        else:
            changed["ema"] = None
            rebound_structure = copy.deepcopy(structural_inventory)
            rebound_structure["tensors"][0]["kind"] = "parameter"
            run_receipt["inference_signature"] = MODULE.checkpoint_inference_signature(
                changed,
                structural_inventory=rebound_structure,
                descriptor_sha256=descriptor_sha256,
            )
        torch.save(changed, checkpoint_path)
        run_receipt["checkpoints"][0].update(binding(checkpoint_path))
        _write_canonical_json(run_path, run_receipt)
        expected_match = "checkpoint inference"
        observed_authority = sources_authority
    elif mutation == "config_substitution":
        _write_canonical_json(
            config_path,
            {"registered": False, "fepf_inference_structure": structural_inventory},
        )
        sources[0]["config"] = {
            "path": str(config_path.resolve()),
            "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "bytes": config_path.stat().st_size,
        }
        _write_canonical_json(sources_path, sources)
        observed_authority = {
            "path": str(sources_path.resolve()),
            "sha256": hashlib.sha256(sources_path.read_bytes()).hexdigest(),
            "bytes": sources_path.stat().st_size,
        }
        expected_match = "initialization provenance"
    else:
        alternate_candidate = tmp_path / "candidate-stop4-alternate"
        shutil.copytree(candidate_root, alternate_candidate)
        sources[0]["candidate_root"] = alternate_candidate.name
        _write_canonical_json(sources_path, sources)
        rebound_authority = {
            "path": str(sources_path.resolve()),
            "sha256": hashlib.sha256(sources_path.read_bytes()).hexdigest(),
            "bytes": sources_path.stat().st_size,
        }
        rebound = MODULE.build_fepf_result(
            phase="epoch4",
            sources=sources,
            sources_authority=rebound_authority,
            evidence_root=tmp_path,
        )
        with pytest.raises(ValueError, match="sources authority"):
            MODULE.validate_fepf_result(
                rebound,
                tmp_path,
                sources_authority=sources_authority,
            )
        return

    with pytest.raises(ValueError, match=expected_match):
        MODULE.build_fepf_result(
            phase="epoch4",
            sources=sources,
            sources_authority=observed_authority,
            evidence_root=tmp_path,
        )


def test_review7_evaluator_parser_requires_config_rooted_publication_budget() -> None:
    with pytest.raises(SystemExit):
        MODULE.parse_args(
            [
                "--phase", "epoch4",
                "--sources", "/tmp/sources.json",
                "--sources-sha256", "1" * 64,
                "--sources-bytes", "1",
                "--evidence-root", "/tmp/evidence",
                "--output", "/tmp/result.json",
                "--temporary", "/tmp/result.tmp",
            ]
        )


def test_review8_evaluator_main_requires_stage_budget_before_sources(
    tmp_path: Path,
) -> None:
    root = tmp_path / "campaign"
    root.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({
        "artifact_root": str(root),
        "publication_budget": {
            "schema": "unicom-fepf-publication-budget-v1", "publications": []
        },
        "publication_budget_sha256": "0" * 64,
    }, indent=2) + "\n")
    assert MODULE.main([
        "--phase", "epoch4",
        "--sources", str(tmp_path / "absent-sources.json"),
        "--sources-sha256", "1" * 64,
        "--sources-bytes", "1",
        "--evidence-root", str(root),
        "--output", str(root / "exploratory-epoch4-decision-result.json"),
        "--temporary", str(root / ".retired.tmp"),
        "--config", str(config),
        "--publication-stage", "exploratory-epoch4-decision",
        "--campaign-root", str(root),
        "--authority-preflight-only",
        ]) == 2


def test_review9_normal_evaluator_enforces_actual_serialized_byte_bound(
    tmp_path: Path,
) -> None:
    row = {
        "name": "exploratory-epoch4-decision:result",
        "path": "exploratory-epoch4-decision-result.json",
        "persistent_bytes": 1,
        "temporary_bytes": 1,
        "persistent_inodes": 1,
        "temporary_inodes": 1,
    }
    with pytest.raises(OSError, match="budget|bytes|capacity"):
        MODULE.validate_publication_payload_bound(
            row,
            destination=tmp_path / row["path"],
            payload=b"too large",
            campaign_root=tmp_path,
        )
