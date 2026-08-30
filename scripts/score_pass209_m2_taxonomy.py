#!/usr/bin/env python3
"""Score the sealed Pass209 M2 blinded error taxonomy."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

_TOP_LEVEL_KEYS = {
    "schema",
    "claim_eligible",
    "rater",
    "order_key",
    "error_manifest_sha256",
    "rows",
}
_RATER_KEYS = {"id", "identity", "expertise", "calibration_completed"}
_ROW_KEYS = {
    "sequence",
    "error_ordinal",
    "query_example_id",
    "nearest_example_id",
    "query_image",
    "nearest_image",
    "primary_account",
    "localized_region",
    "special_evidence",
    "cannot_judge_reason_kind",
    "visible_evidence",
}
_IMAGE_KEYS = {
    "viewpoint",
    "dominant_color",
    "background",
    "degradation",
    "badge_text",
}
_DEGRADATION_KEYS = {
    "vehicle_crop",
    "occlusion_above_25_percent",
    "strong_blur",
    "watermark_over_vehicle",
    "rendering_not_photo",
    "multiple_vehicles",
}
_VIEWPOINTS = {
    "front",
    "front-three-quarter",
    "side",
    "rear-three-quarter",
    "rear",
    "interior/detail",
    "unclear",
}
_COLORS = {
    "white",
    "black",
    "silver-grey",
    "red",
    "blue",
    "green",
    "yellow-gold",
    "orange",
    "brown-beige",
    "purple",
    "multi",
    "other",
    "unclear",
}
_BACKGROUNDS = {
    "studio-white",
    "indoor-showroom",
    "paved-road-lot",
    "grass-nature",
    "other",
    "unclear",
}
_BADGE_TEXT = {"yes", "no", "unclear"}
_PRIMARY_ACCOUNTS = {
    "duplicate",
    "suspected-label-integrity",
    "semantic-overlap",
    "degraded-observation",
    "visually-indistinguishable",
    "localized-cue-visible",
    "global-shape-overridden",
    "unexplained-global",
    "cannot-judge",
}
_CANNOT_JUDGE_REASONS = {
    "image-quality",
    "knowledge",
    "view-combination",
    "other",
}


class Rating(NamedTuple):
    """One authenticated rater row aligned to the manifest ordinal."""

    rater_id: str
    value: dict[str, Any]


class AlignedRow(NamedTuple):
    """The two independent ratings for one manifest error."""

    error_ordinal: int
    manifest_error: dict[str, Any]
    ratings: tuple[Rating, Rating]


class TaxonomyInputs(NamedTuple):
    """Authenticated M2 artifacts and aligned blinded submissions."""

    receipt_sha256: str
    manifest_sha256: str
    manifest: dict[str, Any]
    raw_submission_sha256: tuple[str, str]
    order_sequences: tuple[tuple[int, ...], tuple[int, ...]]
    rows: tuple[AlignedRow, ...]


class AxisAgreement(NamedTuple):
    """Agreement statistics and each rater's categorical prevalence."""

    total: int
    matches: int
    raw_agreement: float
    kappa: float
    pabak: float
    rater_prevalence: tuple[dict[str, int], dict[str, int]]


class AgreementEvidence(NamedTuple):
    """Primary-account and observable-checklist agreement."""

    primary: AxisAgreement
    checklist: dict[str, AxisAgreement]


class AdjudicatedRow(NamedTuple):
    """A no-third-rater primary-account outcome."""

    error_ordinal: int
    query_label: int
    primary_account: str
    judgeable: bool
    original_accounts: tuple[str, str]


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _object_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _read_json_object(path: Path) -> tuple[bytes, dict[str, Any]]:
    raw = path.read_bytes()
    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{path.name} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must contain one JSON object")
    return raw, value


def _artifact_validator() -> Any:
    path = Path(__file__).with_name("validate_pass209_m2_artifacts.py")
    spec = importlib.util.spec_from_file_location("validate_pass209_m2_artifacts", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Pass209 M2 validator cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.validate_pass209_m2_artifacts


def _nonempty_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Pass209 M2 {field} must be a nonempty string")
    return value


def _validate_image(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _IMAGE_KEYS:
        raise ValueError("Pass209 M2 image schema differs")
    degradation = value["degradation"]
    if not isinstance(degradation, dict) or set(degradation) != _DEGRADATION_KEYS:
        raise ValueError("Pass209 M2 degradation schema differs")
    if any(type(degradation[key]) is not bool for key in _DEGRADATION_KEYS):
        raise ValueError("Pass209 M2 degradation values must be concrete booleans")
    categorical_values = (
        value["viewpoint"],
        value["dominant_color"],
        value["background"],
        value["badge_text"],
    )
    if any(not isinstance(item, str) for item in categorical_values) or (
        value["viewpoint"] not in _VIEWPOINTS
        or value["dominant_color"] not in _COLORS
        or value["background"] not in _BACKGROUNDS
        or value["badge_text"] not in _BADGE_TEXT
    ):
        raise ValueError("Pass209 M2 image category differs")
    return value


def _validate_dependent_fields(row: dict[str, Any]) -> None:
    account = row["primary_account"]
    localized = row["localized_region"]
    special = row["special_evidence"]
    reason = row["cannot_judge_reason_kind"]
    if account == "localized-cue-visible":
        _nonempty_string(localized, "localized region")
        if special is not None or reason is not None:
            raise ValueError("Pass209 M2 localized-cue fields differ")
    elif account == "suspected-label-integrity":
        _nonempty_string(special, "label-integrity evidence")
        if localized is not None or reason is not None:
            raise ValueError("Pass209 M2 label-integrity fields differ")
    elif account == "visually-indistinguishable":
        if special != "no visible discriminative region found":
            raise ValueError("Pass209 M2 indistinguishable attestation differs")
        if localized is not None or reason is not None:
            raise ValueError("Pass209 M2 indistinguishable fields differ")
    elif account == "cannot-judge":
        if reason not in _CANNOT_JUDGE_REASONS:
            raise ValueError("Pass209 M2 cannot-judge reason differs")
        if localized is not None or special is not None:
            raise ValueError("Pass209 M2 cannot-judge fields differ")
    elif localized is not None or special is not None or reason is not None:
        raise ValueError("Pass209 M2 inapplicable dependent field is populated")


def _validate_submission(
    value: dict[str, Any],
    raw: bytes,
    manifest: dict[str, Any],
    manifest_sha256: str,
    rater_number: int,
) -> tuple[str, tuple[int, ...], dict[int, Rating]]:
    if set(value) != _TOP_LEVEL_KEYS:
        raise ValueError("Pass209 M2 rater submission schema differs")
    expected_rater_id = f"rater-{rater_number}"
    expected_order_key = f"pass209-m2-rater-{rater_number}-v1"
    if (
        value["schema"] != "sfora-pass209-m2-rater-submission-v1"
        or value["claim_eligible"] is not False
        or value["order_key"] != expected_order_key
        or value["error_manifest_sha256"] != manifest_sha256
    ):
        raise ValueError("Pass209 M2 rater submission authority differs")
    rater = value["rater"]
    if not isinstance(rater, dict) or set(rater) != _RATER_KEYS:
        raise ValueError("Pass209 M2 rater identity schema differs")
    if rater["id"] != expected_rater_id or rater["calibration_completed"] is not True:
        raise ValueError("Pass209 M2 rater identity authority differs")
    _nonempty_string(rater["identity"], "rater identity")
    _nonempty_string(rater["expertise"], "rater expertise")
    rows = value["rows"]
    errors = manifest["errors"]
    if not isinstance(rows, list) or len(rows) != 103:
        raise ValueError("Pass209 M2 rater row cardinality differs")
    expected_order = sorted(
        range(103),
        key=lambda ordinal: (
            hashlib.sha256(
                f"{expected_order_key}\0{errors[ordinal]['query_example_id']}".encode()
            ).hexdigest(),
            errors[ordinal]["query_example_id"],
        ),
    )
    ratings: dict[int, Rating] = {}
    for sequence, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != _ROW_KEYS:
            raise ValueError("Pass209 M2 rater row schema differs")
        ordinal = row["error_ordinal"]
        if (
            not _is_int(row["sequence"])
            or row["sequence"] != sequence
            or not _is_int(ordinal)
            or ordinal != expected_order[sequence]
            or ordinal in ratings
        ):
            raise ValueError("Pass209 M2 rater viewing order differs")
        error = errors[ordinal]
        if (
            row["query_example_id"] != error["query_example_id"]
            or row["nearest_example_id"] != error["nearest_example_id"]
        ):
            raise ValueError("Pass209 M2 rater example identity differs")
        _validate_image(row["query_image"])
        _validate_image(row["nearest_image"])
        if (
            not isinstance(row["primary_account"], str)
            or row["primary_account"] not in _PRIMARY_ACCOUNTS
        ):
            raise ValueError("Pass209 M2 primary account differs")
        _validate_dependent_fields(row)
        _nonempty_string(row["visible_evidence"], "visible evidence")
        ratings[ordinal] = Rating(expected_rater_id, row)
    return hashlib.sha256(raw).hexdigest(), tuple(expected_order), ratings


def load_taxonomy_inputs(
    receipt_path: Path,
    manifest_path: Path,
    rater_paths: tuple[Path, Path],
) -> TaxonomyInputs:
    """Authenticate and align two complete blinded Pass209 M2 submissions."""

    _artifact_validator()(receipt_path, manifest_path)
    receipt_raw = receipt_path.read_bytes()
    manifest_raw, manifest = _read_json_object(manifest_path)
    manifest_sha256 = hashlib.sha256(manifest_raw).hexdigest()
    submission_sha256: list[str] = []
    order_sequences: list[tuple[int, ...]] = []
    rating_maps: list[dict[int, Rating]] = []
    for rater_number, path in enumerate(rater_paths, start=1):
        raw, value = _read_json_object(path)
        digest, order, ratings = _validate_submission(
            value, raw, manifest, manifest_sha256, rater_number
        )
        submission_sha256.append(digest)
        order_sequences.append(order)
        rating_maps.append(ratings)
    aligned = tuple(
        AlignedRow(
            error_ordinal=ordinal,
            manifest_error=manifest["errors"][ordinal],
            ratings=(rating_maps[0][ordinal], rating_maps[1][ordinal]),
        )
        for ordinal in range(103)
    )
    return TaxonomyInputs(
        receipt_sha256=hashlib.sha256(receipt_raw).hexdigest(),
        manifest_sha256=manifest_sha256,
        manifest=manifest,
        raw_submission_sha256=(submission_sha256[0], submission_sha256[1]),
        order_sequences=(order_sequences[0], order_sequences[1]),
        rows=aligned,
    )


def _category_name(value: str | bool) -> str:
    return str(value).lower() if isinstance(value, bool) else value


def _axis_agreement(
    rater_one: list[str | bool], rater_two: list[str | bool]
) -> AxisAgreement:
    if not rater_one or len(rater_one) != len(rater_two):
        raise ValueError("Pass209 M2 agreement vectors differ")
    total = len(rater_one)
    matches = sum(left == right for left, right in zip(rater_one, rater_two, strict=True))
    first = Counter(_category_name(value) for value in rater_one)
    second = Counter(_category_name(value) for value in rater_two)
    observed = matches / total
    expected = sum(
        first[category] * second[category] / (total * total)
        for category in set(first) | set(second)
    )
    if expected == 1.0:
        if observed != 1.0:
            raise ValueError("Pass209 M2 agreement table is impossible")
        kappa = 1.0
    else:
        kappa = (observed - expected) / (1.0 - expected)
    return AxisAgreement(
        total=total,
        matches=matches,
        raw_agreement=observed,
        kappa=kappa,
        pabak=2.0 * observed - 1.0,
        rater_prevalence=(dict(sorted(first.items())), dict(sorted(second.items()))),
    )


def score_agreement(inputs: TaxonomyInputs) -> AgreementEvidence:
    """Compute registered agreement evidence from aligned raw submissions."""

    primary_values: tuple[list[str | bool], list[str | bool]] = ([], [])
    checklist_values: dict[str, tuple[list[str | bool], list[str | bool]]] = {
        name: ([], [])
        for name in (
            "viewpoint",
            "dominant_color",
            "background",
            "badge_text",
            *_DEGRADATION_KEYS,
        )
    }
    for aligned in inputs.rows:
        for rater_index, rating in enumerate(aligned.ratings):
            row = rating.value
            primary_values[rater_index].append(row["primary_account"])
            for image_role in ("query_image", "nearest_image"):
                image = row[image_role]
                for name in ("viewpoint", "dominant_color", "background", "badge_text"):
                    checklist_values[name][rater_index].append(image[name])
                for name in _DEGRADATION_KEYS:
                    checklist_values[name][rater_index].append(
                        image["degradation"][name]
                    )
    checklist = {
        name: _axis_agreement(values[0], values[1])
        for name, values in sorted(checklist_values.items())
    }
    return AgreementEvidence(
        primary=_axis_agreement(primary_values[0], primary_values[1]),
        checklist=checklist,
    )


def adjudicate_without_override(
    inputs: TaxonomyInputs,
) -> tuple[AdjudicatedRow, ...]:
    """Keep matching accounts and mark every disagreement unresolved."""

    outcomes = []
    for aligned in inputs.rows:
        accounts = (
            aligned.ratings[0].value["primary_account"],
            aligned.ratings[1].value["primary_account"],
        )
        account = accounts[0] if accounts[0] == accounts[1] else "unresolved"
        outcomes.append(
            AdjudicatedRow(
                error_ordinal=aligned.error_ordinal,
                query_label=aligned.manifest_error["query_label"],
                primary_account=account,
                judgeable=account not in {"cannot-judge", "unresolved"},
                original_accounts=accounts,
            )
        )
    return tuple(outcomes)


def taxonomy_eligibility(
    primary_raw_agreement: float,
    primary_kappa: float,
    unjudgeable_count: int,
) -> bool:
    """Apply the frozen reliability and unresolved-cardinality gates."""

    if (
        type(primary_raw_agreement) is not float
        or type(primary_kappa) is not float
        or not math.isfinite(primary_raw_agreement)
        or not math.isfinite(primary_kappa)
        or not _is_int(unjudgeable_count)
        or not 0 <= unjudgeable_count <= 103
    ):
        raise ValueError("Pass209 M2 eligibility inputs differ")
    return (
        primary_kappa >= 0.60 or primary_raw_agreement >= 0.80
    ) and unjudgeable_count <= 15
