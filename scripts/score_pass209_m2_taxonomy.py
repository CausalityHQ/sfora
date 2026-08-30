#!/usr/bin/env python3
"""Score the sealed Pass209 M2 blinded error taxonomy."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

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
_PRIMARY_ACCOUNT_ORDER = (
    "duplicate",
    "suspected-label-integrity",
    "semantic-overlap",
    "degraded-observation",
    "visually-indistinguishable",
    "localized-cue-visible",
    "global-shape-overridden",
    "unexplained-global",
    "cannot-judge",
)
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
    submissions: tuple[dict[str, Any], dict[str, Any]]
    raw_submission_sha256: tuple[str, str]
    order_sequences: tuple[tuple[int, ...], tuple[int, ...]]
    rows: tuple[AlignedRow, ...]


class AxisAgreement(NamedTuple):
    """Agreement statistics and each rater's categorical prevalence."""

    total: int
    matches: int
    category_count: int
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
    invoked_rule: str | None = None
    changed_from_raw: bool = False


class ConsensusOutcome(NamedTuple):
    """One sealed two-rater outcome for a raw primary-account disagreement."""

    primary_account: str
    invoked_rule: str | None


class ConsensusRecord(NamedTuple):
    """Authenticated post-sealing consensus evidence."""

    raw_sha256: str
    value: dict[str, Any]
    outcomes: dict[int, ConsensusOutcome]


class BootstrapEvidence(NamedTuple):
    """One registered query-class clustered-bootstrap distribution summary."""

    resamples: int
    observed_share: float
    mean: float
    p025: float
    p10: float
    p975: float
    values_little_endian_f64_sha256: str


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
    submissions: list[dict[str, Any]] = []
    order_sequences: list[tuple[int, ...]] = []
    rating_maps: list[dict[int, Rating]] = []
    for rater_number, path in enumerate(rater_paths, start=1):
        raw, value = _read_json_object(path)
        digest, order, ratings = _validate_submission(
            value, raw, manifest, manifest_sha256, rater_number
        )
        submission_sha256.append(digest)
        submissions.append(value)
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
        submissions=(submissions[0], submissions[1]),
        raw_submission_sha256=(submission_sha256[0], submission_sha256[1]),
        order_sequences=(order_sequences[0], order_sequences[1]),
        rows=aligned,
    )


def _category_name(value: str | bool) -> str:
    return str(value).lower() if isinstance(value, bool) else value


def _axis_agreement(
    rater_one: list[str | bool],
    rater_two: list[str | bool],
    category_count: int,
) -> AxisAgreement:
    if not rater_one or len(rater_one) != len(rater_two) or category_count < 2:
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
        category_count=category_count,
        raw_agreement=observed,
        kappa=kappa,
        pabak=(category_count * observed - 1.0) / (category_count - 1.0),
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
        name: _axis_agreement(
            values[0],
            values[1],
            {
                "viewpoint": len(_VIEWPOINTS),
                "dominant_color": len(_COLORS),
                "background": len(_BACKGROUNDS),
                "badge_text": len(_BADGE_TEXT),
            }.get(name, 2),
        )
        for name, values in sorted(checklist_values.items())
    }
    return AgreementEvidence(
        primary=_axis_agreement(
            primary_values[0], primary_values[1], len(_PRIMARY_ACCOUNTS)
        ),
        checklist=checklist,
    )


def load_consensus_record(
    inputs: TaxonomyInputs, consensus_path: Path
) -> ConsensusRecord:
    """Authenticate an exact two-rater consensus record for every disagreement."""

    raw, value = _read_json_object(consensus_path)
    if raw != _canonical_json_bytes(value):
        raise ValueError("Pass209 M2 consensus record is not canonical")
    expected_keys = {
        "schema",
        "claim_eligible",
        "error_manifest_sha256",
        "rater_submission_sha256",
        "rows",
    }
    if set(value) != expected_keys:
        raise ValueError("Pass209 M2 consensus schema differs")
    if (
        value["schema"] != "sfora-pass209-m2-consensus-v1"
        or value["claim_eligible"] is not False
        or value["error_manifest_sha256"] != inputs.manifest_sha256
        or value["rater_submission_sha256"] != list(inputs.raw_submission_sha256)
    ):
        raise ValueError("Pass209 M2 consensus binding differs")
    disagreements = {
        row.error_ordinal: (
            row.ratings[0].value["primary_account"],
            row.ratings[1].value["primary_account"],
        )
        for row in inputs.rows
        if row.ratings[0].value["primary_account"]
        != row.ratings[1].value["primary_account"]
    }
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != len(disagreements):
        raise ValueError("Pass209 M2 consensus cardinality differs")
    outcomes: dict[int, ConsensusOutcome] = {}
    previous = -1
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "error_ordinal",
            "primary_account",
            "invoked_rule",
        }:
            raise ValueError("Pass209 M2 consensus row schema differs")
        ordinal = row["error_ordinal"]
        account = row["primary_account"]
        invoked_rule = row["invoked_rule"]
        if (
            not _is_int(ordinal)
            or ordinal <= previous
            or ordinal not in disagreements
            or not isinstance(account, str)
        ):
            raise ValueError("Pass209 M2 consensus row authority differs")
        if account == "unresolved":
            if invoked_rule is not None:
                raise ValueError("Pass209 M2 unresolved consensus rule differs")
        elif account not in disagreements[ordinal]:
            raise ValueError("Pass209 M2 consensus cannot introduce a third label")
        else:
            _nonempty_string(invoked_rule, "consensus invoked rule")
        outcomes[ordinal] = ConsensusOutcome(account, invoked_rule)
        previous = ordinal
    if set(outcomes) != set(disagreements):
        raise ValueError("Pass209 M2 consensus coverage differs")
    return ConsensusRecord(hashlib.sha256(raw).hexdigest(), value, outcomes)


def adjudicate_with_consensus(
    inputs: TaxonomyInputs, consensus: ConsensusRecord
) -> tuple[AdjudicatedRow, ...]:
    """Apply matching raw labels and the sealed two-rater consensus outcomes."""

    outcomes = []
    for aligned in inputs.rows:
        accounts = (
            aligned.ratings[0].value["primary_account"],
            aligned.ratings[1].value["primary_account"],
        )
        if accounts[0] == accounts[1]:
            account = accounts[0]
            invoked_rule = None
            changed = False
        else:
            outcome = consensus.outcomes.get(aligned.error_ordinal)
            if outcome is None:
                raise ValueError("Pass209 M2 consensus omits a disagreement")
            account = outcome.primary_account
            invoked_rule = outcome.invoked_rule
            changed = True
        outcomes.append(
            AdjudicatedRow(
                error_ordinal=aligned.error_ordinal,
                query_label=aligned.manifest_error["query_label"],
                primary_account=account,
                judgeable=account not in {"cannot-judge", "unresolved"},
                original_accounts=accounts,
                invoked_rule=invoked_rule,
                changed_from_raw=changed,
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


def _bootstrap_metric(
    class_rows: dict[int, tuple[AdjudicatedRow, ...]],
    numerator_accounts: frozenset[str],
    sampled_classes: tuple[tuple[int, ...], ...],
) -> BootstrapEvidence:
    class_denominators = {
        label: sum(row.judgeable for row in rows) for label, rows in class_rows.items()
    }
    class_numerators = {
        label: sum(
            row.judgeable and row.primary_account in numerator_accounts for row in rows
        )
        for label, rows in class_rows.items()
    }
    total_denominator = sum(class_denominators.values())
    observed = (
        sum(class_numerators.values()) / total_denominator
        if total_denominator
        else 0.0
    )
    values = []
    for sample in sampled_classes:
        denominator = sum(class_denominators[label] for label in sample)
        values.append(
            sum(class_numerators[label] for label in sample) / denominator
            if denominator
            else 0.0
        )
    array = np.asarray(values, dtype="<f8")
    percentiles = np.percentile(array, [2.5, 10.0, 97.5], method="inverted_cdf")
    return BootstrapEvidence(
        resamples=10_000,
        observed_share=observed,
        mean=float(array.mean()),
        p025=float(percentiles[0]),
        p10=float(percentiles[1]),
        p975=float(percentiles[2]),
        values_little_endian_f64_sha256=hashlib.sha256(array.tobytes()).hexdigest(),
    )


def bootstrap_primary_shares(
    rows: tuple[AdjudicatedRow, ...],
) -> dict[str, BootstrapEvidence]:
    """Bootstrap registered primary-account shares by whole query class."""

    if any(row.query_label not in range(82, 98) for row in rows):
        raise ValueError("Pass209 M2 bootstrap query class differs")
    class_rows = {
        label: tuple(row for row in rows if row.query_label == label)
        for label in range(82, 98)
    }
    seed = int.from_bytes(
        hashlib.sha256(b"pass209-m2-bootstrap-v1").digest()[:16], "big"
    )
    generator = np.random.Generator(np.random.PCG64(seed))
    sampled_classes = tuple(
        tuple(int(index) + 82 for index in generator.integers(0, 16, size=16))
        for _ in range(10_000)
    )
    account_sets: dict[str, frozenset[str]] = {
        account: frozenset({account}) for account in _PRIMARY_ACCOUNT_ORDER
    }
    account_sets.update(
        {
            "data_sum": frozenset(
                {"duplicate", "suspected-label-integrity", "semantic-overlap"}
            ),
            "ceiling_sum": frozenset(
                {
                    "duplicate",
                    "suspected-label-integrity",
                    "semantic-overlap",
                    "visually-indistinguishable",
                }
            ),
            "localized_cue_visible": frozenset({"localized-cue-visible"}),
            "global_shape_overridden": frozenset({"global-shape-overridden"}),
        }
    )
    return {
        name: _bootstrap_metric(class_rows, accounts, sampled_classes)
        for name, accounts in account_sets.items()
    }


_SAME_LINE_PAIRS = {
    (82, 83),
    (85, 86),
    (89, 90),
    (93, 94),
    (95, 96),
}
_BODY_GROUP = {
    82: "Wagon",
    83: "Wagon",
    84: "Van",
    85: "Cab",
    86: "Cab",
    87: "Van",
    88: "SUV",
    89: "Cab",
    90: "Cab",
    91: "Wagon",
    92: "SRT8",
    93: "SUV",
    94: "SUV",
    95: "Sedan",
    96: "SRT8",
    97: "Hatchback",
}


def _semantic_relation(query_label: int, nearest_label: int) -> str:
    pair = tuple(sorted((query_label, nearest_label)))
    if (
        query_label == nearest_label
        or query_label not in _BODY_GROUP
        or nearest_label not in _BODY_GROUP
    ):
        raise ValueError("Pass209 M2 semantic relation labels differ")
    if pair in _SAME_LINE_PAIRS:
        return "same-line"
    if 97 in pair:
        return "cross-make"
    if _BODY_GROUP[query_label] == _BODY_GROUP[nearest_label]:
        return "same-make-same-body"
    return "same-make-cross-body"


def manifest_error_tables(manifest: dict[str, Any]) -> dict[str, object]:
    """Compute deterministic non-visual error and gallery-pathology tables."""

    errors = manifest.get("errors")
    if not isinstance(errors, list) or len(errors) != 103:
        raise ValueError("Pass209 M2 manifest table cardinality differs")
    query_counts: Counter[int] = Counter()
    directed_counts: Counter[tuple[int, int]] = Counter()
    unordered_counts: Counter[tuple[int, int]] = Counter()
    relation_counts: Counter[str] = Counter()
    nearest_counts: Counter[str] = Counter()
    for row in errors:
        if not isinstance(row, dict):
            raise ValueError("Pass209 M2 manifest table row differs")
        query_label = row["query_label"]
        nearest_label = row["nearest_label"]
        nearest_id = row["nearest_example_id"]
        if (
            not _is_int(query_label)
            or not _is_int(nearest_label)
            or not isinstance(nearest_id, str)
            or not nearest_id
        ):
            raise ValueError("Pass209 M2 manifest table value differs")
        query_counts[query_label] += 1
        directed_counts[(query_label, nearest_label)] += 1
        unordered_counts[tuple(sorted((query_label, nearest_label)))] += 1
        relation_counts[_semantic_relation(query_label, nearest_label)] += 1
        nearest_counts[nearest_id] += 1
    multiplicity = [
        {"nearest_example_id": example_id, "count": count}
        for example_id, count in sorted(
            nearest_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ]
    maximum = max(nearest_counts.values())
    return {
        "query_class_counts": [
            {"query_label": label, "count": query_counts[label]}
            for label in range(82, 98)
        ],
        "directed_pair_counts": [
            {"query_label": pair[0], "nearest_label": pair[1], "count": count}
            for pair, count in sorted(directed_counts.items())
        ],
        "unordered_pair_counts": [
            {"labels": [pair[0], pair[1]], "count": count}
            for pair, count in sorted(unordered_counts.items())
        ],
        "semantic_relation_counts": [
            {"relation": relation, "count": relation_counts[relation]}
            for relation in (
                "cross-make",
                "same-line",
                "same-make-cross-body",
                "same-make-same-body",
            )
        ],
        "nearest_example_multiplicity": multiplicity,
        "maximum_nearest_example_multiplicity": maximum,
        "gallery_pathology": maximum >= 16,
    }


def _axis_value(evidence: AxisAgreement) -> dict[str, object]:
    return {
        "total": evidence.total,
        "matches": evidence.matches,
        "category_count": evidence.category_count,
        "raw_agreement": evidence.raw_agreement,
        "kappa": evidence.kappa,
        "pabak": evidence.pabak,
        "rater_prevalence": [
            evidence.rater_prevalence[0],
            evidence.rater_prevalence[1],
        ],
    }


def _agreement_value(evidence: AgreementEvidence) -> dict[str, object]:
    return {
        "primary": _axis_value(evidence.primary),
        "checklist": {
            name: _axis_value(axis) for name, axis in sorted(evidence.checklist.items())
        },
    }


def _bootstrap_value(evidence: BootstrapEvidence) -> dict[str, object]:
    return {
        "resamples": evidence.resamples,
        "observed_share": evidence.observed_share,
        "mean": evidence.mean,
        "p025": evidence.p025,
        "p10": evidence.p10,
        "p975": evidence.p975,
        "values_little_endian_f64_sha256": (
            evidence.values_little_endian_f64_sha256
        ),
    }


def _adjudicated_value(row: AdjudicatedRow) -> dict[str, object]:
    return {
        "error_ordinal": row.error_ordinal,
        "query_label": row.query_label,
        "primary_account": row.primary_account,
        "judgeable": row.judgeable,
        "original_accounts": list(row.original_accounts),
        "invoked_rule": row.invoked_rule,
        "changed_from_raw": row.changed_from_raw,
    }


def _derived_taxonomy_values(
    inputs: TaxonomyInputs, consensus: ConsensusRecord
) -> tuple[
    dict[str, object],
    list[dict[str, object]],
    dict[str, dict[str, object]],
    dict[str, object],
    dict[str, object],
    str,
]:
    agreement = score_agreement(inputs)
    adjudicated = adjudicate_with_consensus(inputs, consensus)
    bootstrap = bootstrap_primary_shares(adjudicated)
    unjudgeable = sum(not row.judgeable for row in adjudicated)
    eligible = taxonomy_eligibility(
        agreement.primary.raw_agreement,
        agreement.primary.kappa,
        unjudgeable,
    )
    eligibility: dict[str, object] = {
        "eligible": eligible,
        "primary_kappa_minimum": 0.60,
        "primary_raw_agreement_minimum": 0.80,
        "maximum_cannot_judge_or_unresolved": 15,
        "cannot_judge_or_unresolved": unjudgeable,
    }
    return (
        _agreement_value(agreement),
        [_adjudicated_value(row) for row in adjudicated],
        {name: _bootstrap_value(value) for name, value in sorted(bootstrap.items())},
        manifest_error_tables(inputs.manifest),
        eligibility,
        "pending-m3" if eligible else "F-NONE",
    )


def _taxonomy_receipt_value(
    inputs: TaxonomyInputs, consensus: ConsensusRecord
) -> dict[str, object]:
    agreement, adjudicated, bootstrap, tables, eligibility, decision = (
        _derived_taxonomy_values(inputs, consensus)
    )
    receipt: dict[str, object] = {
        "schema": "sfora-pass209-m2-taxonomy-v1",
        "claim_eligible": False,
        "source": {
            "receipt_sha256": inputs.receipt_sha256,
            "manifest_sha256": inputs.manifest_sha256,
            "rater_submission_sha256": list(inputs.raw_submission_sha256),
            "consensus_sha256": consensus.raw_sha256,
        },
        "source_manifest": inputs.manifest,
        "raw_submissions": list(inputs.submissions),
        "agreement": agreement,
        "manifest_tables": tables,
        "eligibility": eligibility,
        "family_decision": decision,
    }
    if eligibility["eligible"] is True:
        receipt.update(
            raw_consensus=consensus.value,
            adjudicated_rows=adjudicated,
            bootstrap=bootstrap,
        )
    return receipt


def _canonical_json_bytes(value: dict[str, object]) -> bytes:
    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("Pass209 M2 taxonomy contains a noncanonical value") from error
    return (text + "\n").encode()


def taxonomy_receipt_bytes(
    inputs: TaxonomyInputs, consensus: ConsensusRecord
) -> bytes:
    """Return the canonical, claim-ineligible M2 taxonomy receipt."""

    return _canonical_json_bytes(_taxonomy_receipt_value(inputs, consensus))


def validate_taxonomy_receipt_bytes(
    raw: bytes, inputs: TaxonomyInputs, consensus: ConsensusRecord
) -> dict[str, Any]:
    """Independently rederive and validate a canonical taxonomy receipt."""

    try:
        value = json.loads(raw, object_pairs_hook=_object_without_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("Pass209 M2 taxonomy receipt is not JSON") from error
    if not isinstance(value, dict) or raw != _canonical_json_bytes(value):
        raise ValueError("Pass209 M2 taxonomy receipt is not canonical")
    agreement, adjudicated, bootstrap, tables, eligibility, decision = (
        _derived_taxonomy_values(inputs, consensus)
    )
    expected_keys = {
        "schema",
        "claim_eligible",
        "source",
        "source_manifest",
        "raw_submissions",
        "agreement",
        "manifest_tables",
        "eligibility",
        "family_decision",
    }
    if eligibility["eligible"] is True:
        expected_keys.update({"raw_consensus", "adjudicated_rows", "bootstrap"})
    if set(value) != expected_keys:
        raise ValueError("Pass209 M2 taxonomy receipt schema differs")
    if value["schema"] != "sfora-pass209-m2-taxonomy-v1" or value[
        "claim_eligible"
    ] is not False:
        raise ValueError("Pass209 M2 taxonomy receipt authority differs")
    expected_source = {
        "receipt_sha256": inputs.receipt_sha256,
        "manifest_sha256": inputs.manifest_sha256,
        "rater_submission_sha256": list(inputs.raw_submission_sha256),
        "consensus_sha256": consensus.raw_sha256,
    }
    if (
        value["source"] != expected_source
        or value["source_manifest"] != inputs.manifest
        or value["raw_submissions"] != list(inputs.submissions)
    ):
        raise ValueError("Pass209 M2 taxonomy source binding differs")
    if eligibility["eligible"] is True and value["raw_consensus"] != consensus.value:
        raise ValueError("Pass209 M2 taxonomy consensus binding differs")
    derived: list[tuple[str, object]] = [
        ("agreement", agreement),
        ("manifest_tables", tables),
        ("eligibility", eligibility),
        ("family_decision", decision),
    ]
    if eligibility["eligible"] is True:
        derived.extend(
            [("adjudicated_rows", adjudicated), ("bootstrap", bootstrap)]
        )
    for key, expected in derived:
        if value[key] != expected:
            raise ValueError(f"Pass209 M2 taxonomy {key} differs")
    return value


def publish_create_new(output: Path, raw: bytes) -> None:
    """Atomically publish bytes without replacing any existing path."""

    if output.exists():
        raise FileExistsError(f"refusing to replace {output}")
    partial = output.with_name(f".{output.name}.partial")
    descriptor: int | None = None
    owned_partial = False
    try:
        descriptor = os.open(partial, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
        owned_partial = True
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            descriptor = None
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, output)
        directory_descriptor = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if owned_partial and partial.exists():
            partial.unlink()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--error-manifest", type=Path, required=True)
    parser.add_argument("--rater-one", type=Path, required=True)
    parser.add_argument("--rater-two", type=Path, required=True)
    parser.add_argument("--consensus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    inputs = load_taxonomy_inputs(
        arguments.receipt,
        arguments.error_manifest,
        (arguments.rater_one, arguments.rater_two),
    )
    consensus = load_consensus_record(inputs, arguments.consensus)
    raw = taxonomy_receipt_bytes(inputs, consensus)
    validate_taxonomy_receipt_bytes(raw, inputs, consensus)
    publish_create_new(arguments.output, raw)


if __name__ == "__main__":
    main()
