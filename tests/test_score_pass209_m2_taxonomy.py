from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "score_pass209_m2_taxonomy.py"
_SPEC = importlib.util.spec_from_file_location("score_pass209_m2_taxonomy", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def _canonical(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _m2_artifacts(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    errors = [
        {
            "query_position": index,
            "query_example_id": f"cars-train-{82 + index % 16}-{index}",
            "query_label": 82 + index % 16,
            "nearest_position": 200 + index,
            "nearest_example_id": f"cars-train-{82 + (index + 1) % 16}-{200 + index}",
            "nearest_label": 82 + (index + 1) % 16,
        }
        for index in range(103)
    ]
    class_names = [
        {"id": 82, "name": "Dodge Caliber Wagon 2012"},
        {"id": 83, "name": "Dodge Caliber Wagon 2007"},
        {"id": 84, "name": "Dodge Caravan Minivan 1997"},
        {"id": 85, "name": "Dodge Ram Pickup 3500 Crew Cab 2010"},
        {"id": 86, "name": "Dodge Ram Pickup 3500 Quad Cab 2009"},
        {"id": 87, "name": "Dodge Sprinter Cargo Van 2009"},
        {"id": 88, "name": "Dodge Journey SUV 2012"},
        {"id": 89, "name": "Dodge Dakota Crew Cab 2010"},
        {"id": 90, "name": "Dodge Dakota Club Cab 2007"},
        {"id": 91, "name": "Dodge Magnum Wagon 2008"},
        {"id": 92, "name": "Dodge Challenger SRT8 2011"},
        {"id": 93, "name": "Dodge Durango SUV 2012"},
        {"id": 94, "name": "Dodge Durango SUV 2007"},
        {"id": 95, "name": "Dodge Charger Sedan 2012"},
        {"id": 96, "name": "Dodge Charger SRT-8 2009"},
        {"id": 97, "name": "Eagle Talon Hatchback 1998"},
    ]
    manifest: dict[str, Any] = {
        "schema": "sfora-frozen-substrate-errors-v1",
        "claim_eligible": False,
        "source_revision": "1" * 40,
        "source_tree_digest": "2" * 64,
        "dataset": "cars",
        "dataset_revision": "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        "dataset_examples_sha256": "3" * 64,
        "descriptor_sha256": "4" * 64,
        "batch_size": 8,
        "query_block": 32,
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "class_names": class_names,
        "cell": "siglip-so400m",
        "model_name": "google/siglip-so400m-patch14-384",
        "model_revision": "9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        "error_count": 103,
        "errors": errors,
    }
    manifest_bytes = _canonical(manifest)
    receipt: dict[str, Any] = {
        "schema": "sfora-frozen-substrate-screen-v2",
        "claim_eligible": False,
        "source_revision": manifest["source_revision"],
        "source_tree_digest": manifest["source_tree_digest"],
        "dataset": "cars",
        "dataset_revision": manifest["dataset_revision"],
        "dataset_examples_sha256": manifest["dataset_examples_sha256"],
        "split": "train",
        "holdout_classes": list(range(82, 98)),
        "cell": "siglip-so400m",
        "model_name": manifest["model_name"],
        "model_revision": manifest["model_revision"],
        "readout": "vision_pooler_output",
        "compute_dtype": "float32",
        "processor_image_shape": [384, 384],
        "descriptors_validated": True,
        "norm_tolerance": 1.0e-6,
        "metrics": {"correct": 1242, "queries": 1345, "recall_at_1": 1242 / 1345},
        "gates": {"expected_queries": 1345, "recall_at_1_minimum": 0.94},
        "passed": False,
        "batch_size": 8,
        "query_block": 32,
        "descriptor_shape": [1345, 1152],
        "descriptor_sha256": manifest["descriptor_sha256"],
        "error_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
    }
    receipt_path = tmp_path / "receipt.json"
    manifest_path = tmp_path / "manifest.json"
    receipt_path.write_bytes(_canonical(receipt))
    manifest_path.write_bytes(manifest_bytes)
    return receipt_path, manifest_path, manifest


def _image() -> dict[str, Any]:
    return {
        "viewpoint": "front",
        "dominant_color": "red",
        "background": "paved-road-lot",
        "degradation": {
            "vehicle_crop": False,
            "occlusion_above_25_percent": False,
            "strong_blur": False,
            "watermark_over_vehicle": False,
            "rendering_not_photo": False,
            "multiple_vehicles": False,
        },
        "badge_text": "no",
    }


def _submission(
    manifest: dict[str, Any], rater_number: int, manifest_sha256: str
) -> dict[str, Any]:
    order_key = f"pass209-m2-rater-{rater_number}-v1"
    ordered = sorted(
        enumerate(manifest["errors"]),
        key=lambda item: (
            hashlib.sha256(
                f"{order_key}\0{item[1]['query_example_id']}".encode()
            ).hexdigest(),
            item[1]["query_example_id"],
        ),
    )
    rows = []
    for sequence, (error_ordinal, error) in enumerate(ordered):
        rows.append(
            {
                "sequence": sequence,
                "error_ordinal": error_ordinal,
                "query_example_id": error["query_example_id"],
                "nearest_example_id": error["nearest_example_id"],
                "query_image": _image(),
                "nearest_image": _image(),
                "primary_account": "localized-cue-visible",
                "localized_region": "front grille",
                "special_evidence": None,
                "cannot_judge_reason_kind": None,
                "visible_evidence": "The front grille geometry visibly differs.",
            }
        )
    return {
        "schema": "sfora-pass209-m2-rater-submission-v1",
        "claim_eligible": False,
        "rater": {
            "id": f"rater-{rater_number}",
            "identity": f"fixture-rater-{rater_number}",
            "expertise": "general visual reasoning",
            "calibration_completed": True,
        },
        "order_key": order_key,
        "error_manifest_sha256": manifest_sha256,
        "rows": rows,
    }


def _coherent_inputs(
    tmp_path: Path,
) -> tuple[Path, Path, list[Path], list[dict[str, Any]]]:
    receipt_path, manifest_path, manifest = _m2_artifacts(tmp_path)
    manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    paths = []
    submissions = []
    for number in (1, 2):
        submission = _submission(manifest, number, manifest_sha256)
        path = tmp_path / f"rater-{number}.json"
        path.write_bytes(_canonical(submission))
        paths.append(path)
        submissions.append(submission)
    return receipt_path, manifest_path, paths, submissions


def test_load_taxonomy_inputs_authenticates_complete_independent_orders(
    tmp_path: Path,
) -> None:
    receipt_path, manifest_path, paths, _ = _coherent_inputs(tmp_path)
    expected_submission_sha256 = [
        hashlib.sha256(path.read_bytes()).hexdigest() for path in paths
    ]

    inputs = _MODULE.load_taxonomy_inputs(
        receipt_path, manifest_path, (paths[0], paths[1])
    )

    assert inputs.raw_submission_sha256 == tuple(expected_submission_sha256)
    assert tuple(row.error_ordinal for row in inputs.rows) == tuple(range(103))
    assert inputs.order_sequences[0] != inputs.order_sequences[1]
    assert sorted(inputs.order_sequences[0]) == list(range(103))
    assert sorted(inputs.order_sequences[1]) == list(range(103))
    assert inputs.rows[0].ratings[0].rater_id == "rater-1"
    assert inputs.rows[0].ratings[1].rater_id == "rater-2"


def test_load_taxonomy_inputs_rejects_nonstring_categorical_values(
    tmp_path: Path,
) -> None:
    receipt_path, manifest_path, paths, submissions = _coherent_inputs(tmp_path)
    mutated = deepcopy(submissions[0])
    mutated["rows"][0]["query_image"]["viewpoint"] = ["front"]
    paths[0].write_bytes(_canonical(mutated))

    with pytest.raises(ValueError, match="image category"):
        _MODULE.load_taxonomy_inputs(
            receipt_path, manifest_path, (paths[0], paths[1])
        )


def _mutate_submission(value: dict[str, Any], case: str) -> None:
    row = value["rows"][0]
    if case == "extra-top-level":
        value["extra"] = None
    elif case == "schema":
        value["schema"] = "wrong"
    elif case == "claim":
        value["claim_eligible"] = 0
    elif case == "rater-id":
        value["rater"]["id"] = "rater-2"
    elif case == "identity":
        value["rater"]["identity"] = ""
    elif case == "expertise":
        value["rater"]["expertise"] = " "
    elif case == "calibration":
        value["rater"]["calibration_completed"] = 1
    elif case == "order-key":
        value["order_key"] = "pass209-m2-rater-2-v1"
    elif case == "manifest-digest":
        value["error_manifest_sha256"] = "0" * 64
    elif case == "sequence-bool":
        row["sequence"] = False
    elif case == "sequence-order":
        value["rows"][0], value["rows"][1] = value["rows"][1], value["rows"][0]
    elif case == "query-id":
        row["query_example_id"] = "wrong"
    elif case == "nearest-id":
        row["nearest_example_id"] = "wrong"
    elif case == "degradation-keys":
        del row["query_image"]["degradation"]["strong_blur"]
    elif case == "degradation-int":
        row["query_image"]["degradation"]["strong_blur"] = 0
    elif case == "primary-type":
        row["primary_account"] = ["localized-cue-visible"]
    elif case == "localized-region":
        row["localized_region"] = ""
    elif case == "label-integrity-evidence":
        row.update(
            primary_account="suspected-label-integrity",
            localized_region=None,
            special_evidence=None,
        )
    elif case == "indistinguishable-attestation":
        row.update(
            primary_account="visually-indistinguishable",
            localized_region=None,
            special_evidence="cannot tell",
        )
    elif case == "cannot-judge-reason":
        row.update(
            primary_account="cannot-judge",
            localized_region=None,
            cannot_judge_reason_kind="unclear",
        )
    elif case == "inapplicable-field":
        row.update(primary_account="duplicate", localized_region="grille")
    elif case == "visible-evidence":
        row["visible_evidence"] = ""
    else:
        raise AssertionError(case)


@pytest.mark.parametrize(
    "case",
    [
        "extra-top-level",
        "schema",
        "claim",
        "rater-id",
        "identity",
        "expertise",
        "calibration",
        "order-key",
        "manifest-digest",
        "sequence-bool",
        "sequence-order",
        "query-id",
        "nearest-id",
        "degradation-keys",
        "degradation-int",
        "primary-type",
        "localized-region",
        "label-integrity-evidence",
        "indistinguishable-attestation",
        "cannot-judge-reason",
        "inapplicable-field",
        "visible-evidence",
    ],
)
def test_load_taxonomy_inputs_rejects_submission_authority_drift(
    tmp_path: Path, case: str
) -> None:
    receipt_path, manifest_path, paths, submissions = _coherent_inputs(tmp_path)
    mutated = deepcopy(submissions[0])
    _mutate_submission(mutated, case)
    paths[0].write_bytes(_canonical(mutated))

    with pytest.raises(ValueError):
        _MODULE.load_taxonomy_inputs(
            receipt_path, manifest_path, (paths[0], paths[1])
        )


def _row_for_ordinal(submission: dict[str, Any], ordinal: int) -> dict[str, Any]:
    return next(row for row in submission["rows"] if row["error_ordinal"] == ordinal)


def test_score_agreement_matches_hand_derived_primary_and_checklist_tables(
    tmp_path: Path,
) -> None:
    receipt_path, manifest_path, paths, submissions = _coherent_inputs(tmp_path)
    rater_two = deepcopy(submissions[1])
    for ordinal in range(23):
        row = _row_for_ordinal(rater_two, ordinal)
        row.update(primary_account="semantic-overlap", localized_region=None)
    _row_for_ordinal(rater_two, 0)["query_image"]["viewpoint"] = "side"
    paths[1].write_bytes(_canonical(rater_two))
    inputs = _MODULE.load_taxonomy_inputs(
        receipt_path, manifest_path, (paths[0], paths[1])
    )

    evidence = _MODULE.score_agreement(inputs)

    assert evidence.primary.total == 103
    assert evidence.primary.matches == 80
    assert evidence.primary.raw_agreement == 80 / 103
    assert evidence.primary.kappa == 0.0
    assert evidence.primary.pabak == 2 * (80 / 103) - 1
    assert evidence.primary.rater_prevalence == (
        {"localized-cue-visible": 103},
        {"localized-cue-visible": 80, "semantic-overlap": 23},
    )
    assert set(evidence.checklist) == {
        "viewpoint",
        "dominant_color",
        "background",
        "badge_text",
        "vehicle_crop",
        "occlusion_above_25_percent",
        "strong_blur",
        "watermark_over_vehicle",
        "rendering_not_photo",
        "multiple_vehicles",
    }
    assert evidence.checklist["viewpoint"].total == 206
    assert evidence.checklist["viewpoint"].matches == 205
    assert evidence.checklist["viewpoint"].raw_agreement == 205 / 206
    assert evidence.checklist["viewpoint"].kappa == 0.0
    assert evidence.checklist["viewpoint"].pabak == 2 * (205 / 206) - 1
    assert evidence.checklist["dominant_color"].kappa == 1.0


def test_adjudication_preserves_matches_and_marks_disagreements_unresolved(
    tmp_path: Path,
) -> None:
    receipt_path, manifest_path, paths, submissions = _coherent_inputs(tmp_path)
    rater_one = deepcopy(submissions[0])
    rater_two = deepcopy(submissions[1])
    rater_two_row = _row_for_ordinal(rater_two, 0)
    rater_two_row.update(primary_account="semantic-overlap", localized_region=None)
    for submission in (rater_one, rater_two):
        row = _row_for_ordinal(submission, 1)
        row.update(
            primary_account="cannot-judge",
            localized_region=None,
            cannot_judge_reason_kind="knowledge",
            visible_evidence="The visible details do not resolve the model year.",
        )
    paths[0].write_bytes(_canonical(rater_one))
    paths[1].write_bytes(_canonical(rater_two))
    inputs = _MODULE.load_taxonomy_inputs(
        receipt_path, manifest_path, (paths[0], paths[1])
    )

    rows = _MODULE.adjudicate_without_override(inputs)

    assert len(rows) == 103
    assert rows[0].primary_account == "unresolved"
    assert rows[0].judgeable is False
    assert rows[0].original_accounts == (
        "localized-cue-visible",
        "semantic-overlap",
    )
    assert rows[1].primary_account == "cannot-judge"
    assert rows[1].judgeable is False
    assert rows[2].primary_account == "localized-cue-visible"
    assert rows[2].judgeable is True


def test_taxonomy_eligibility_mutation_locks_registered_thresholds() -> None:
    assert _MODULE.taxonomy_eligibility(0.80, 0.0, 15) is True
    assert _MODULE.taxonomy_eligibility(0.0, 0.60, 15) is True
    assert _MODULE.taxonomy_eligibility(math.nextafter(0.80, 0.0), 0.0, 15) is False
    assert _MODULE.taxonomy_eligibility(0.0, math.nextafter(0.60, 0.0), 15) is False
    assert _MODULE.taxonomy_eligibility(0.80, 0.60, 16) is False
