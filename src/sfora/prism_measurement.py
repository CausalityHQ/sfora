"""Pure, capability-separated PRISM cue-measurement evidence."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from dataclasses import asdict, dataclass

import numpy as np

PRISM_CHANNELS = (
    "grille-fascia",
    "lamps",
    "wheels",
    "silhouette-roofline",
    "trim-badging",
    "stance-proportions",
    "interior-dashboard",
    "model-year-evidence",
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class PrismExample:
    """One source-bound image identity used only during schedule construction."""

    example_id: str
    label: int
    image_sha256: str


@dataclass(frozen=True, slots=True)
class PrismObservationRow:
    """One anonymous cue request with no scoring truth capability."""

    pair_ordinal: int
    fold: int
    channel: str
    left_payload_sha256: str
    right_payload_sha256: str
    left_first: bool
    generation_seed: int


@dataclass(frozen=True, slots=True)
class PrismScoringRow:
    """One pair's scoring truth with no image or model capability."""

    pair_ordinal: int
    fold: int
    left_example_id: str
    right_example_id: str
    left_payload_sha256: str
    right_payload_sha256: str
    left_label: int
    right_label: int
    relation: str


@dataclass(frozen=True, slots=True)
class _Pair:
    fold: int
    left: PrismExample
    right: PrismExample
    relation: str
    stratum: str


@dataclass(frozen=True, slots=True)
class PrismObservationCapabilityRow:
    """The fold-free capability released to one observation process."""

    pair_handle: str
    channel: str
    left_payload_sha256: str
    right_payload_sha256: str
    left_first: bool
    generation_seed: int


@dataclass(frozen=True, slots=True)
class PrismTokenProtocol:
    """Exact token sequences for one tokenizer- and prompt-bound completion."""

    channel_prefixes: tuple[tuple[int, ...], ...]
    visibility_prefixes: tuple[tuple[int, ...], ...]
    relation_prefixes: tuple[tuple[int, ...], ...]
    confidence_prefixes: tuple[tuple[int, ...], ...]
    evidence_separator: tuple[int, ...]
    terminal_tokens: tuple[int, ...]
    max_evidence_tokens: int


@dataclass(frozen=True, slots=True)
class PrismObservation:
    """One typed completion, without decoded text or scoring truth."""

    pair_ordinal: int
    fold: int
    channel: str
    left_first: bool
    left_payload_sha256: str
    right_payload_sha256: str
    generation_seed: int
    protocol_valid: bool
    left_visible: bool
    right_visible: bool
    relation: str
    confidence: str
    evidence_left_token_ids: tuple[int, ...]
    evidence_right_token_ids: tuple[int, ...]
    completion_sha256: str


@dataclass(frozen=True, slots=True)
class PrismChannelCalibration:
    """One fixed-channel calibration from optimization folds one through three."""

    channel: str
    counts: tuple[tuple[int, int], ...]
    visibility_ppm: int
    loo_log_loss_improvement: float
    fold_log_loss_improvements: tuple[float, float, float]
    eligible: bool


@dataclass(frozen=True, slots=True)
class PrismCueResult:
    """Fixed-gate evidence from the sealed 32-pair Caliber panel."""

    bootstrap_draws: int
    bootstrap_seed_sha256: str
    calibration_receipt_sha256: str
    pair_scores: tuple[float, ...]
    pair_truth: tuple[int, ...]
    mean_log_loss_improvement: float
    mean_log_loss_improvement_lower_95: float
    auc: float
    auc_lower_95: float
    valid_orientation_ppm: tuple[int, int]
    orientation_auc_gap: float
    eligible_channels: tuple[str, ...]
    conditional_agreement: tuple[tuple[str, str, str, int, float | None], ...]
    log_loss_gate_passed: bool
    auc_gate_passed: bool
    channel_gate_passed: bool
    orientation_gate_passed: bool
    cue_classification: str
    passed: bool


def _rank(source_identity: str, domain: str, *values: object) -> bytes:
    digest = hashlib.sha256()
    for value in ("sfora-prism-schedule-v1", source_identity, domain, *values):
        encoded = str(value).encode()
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.digest()


def _validate_examples(
    optimization_examples: tuple[PrismExample, ...],
    caliber_examples: tuple[PrismExample, ...],
    source_identity: str,
) -> None:
    if (
        type(optimization_examples) is not tuple
        or type(caliber_examples) is not tuple
        or type(source_identity) is not str
        or not source_identity
    ):
        raise TypeError("PRISM schedule authority has the wrong concrete type")
    combined = optimization_examples + caliber_examples
    if any(type(example) is not PrismExample for example in combined):
        raise TypeError("PRISM examples have the wrong concrete type")
    if any(
        type(example.example_id) is not str
        or not example.example_id
        or type(example.label) is not int
        or _SHA256.fullmatch(example.image_sha256) is None
        for example in combined
    ):
        raise ValueError("PRISM example authority differs")
    if (
        any(not 0 <= example.label <= 48 for example in optimization_examples)
        or any(example.label not in {82, 83} for example in caliber_examples)
        or len({example.example_id for example in combined}) != len(combined)
        or len({example.image_sha256 for example in combined}) != len(combined)
    ):
        raise ValueError("PRISM example population differs")


def _same_candidates(
    examples: tuple[PrismExample, ...],
    source_identity: str,
) -> list[tuple[PrismExample, PrismExample]]:
    by_label: dict[int, list[PrismExample]] = defaultdict(list)
    for example in examples:
        by_label[example.label].append(example)
    pairs: list[tuple[PrismExample, PrismExample]] = []
    for label, rows in by_label.items():
        ordered = sorted(
            rows,
            key=lambda row: _rank(
                source_identity,
                "same-example",
                label,
                row.example_id,
                row.image_sha256,
            ),
        )
        paired_count = (len(ordered) // 2) * 2
        pairs.extend(
            zip(
                ordered[:paired_count:2],
                ordered[1:paired_count:2],
                strict=True,
            )
        )
    pairs.sort(
        key=lambda pair: _rank(
            source_identity,
            "same-pair",
            pair[0].label,
            pair[0].example_id,
            pair[1].example_id,
        )
    )
    return pairs


def _different_pairs(
    examples: tuple[PrismExample, ...],
    used_ids: set[str],
    source_identity: str,
    count: int,
) -> list[tuple[PrismExample, PrismExample]]:
    remaining = sorted(
        (example for example in examples if example.example_id not in used_ids),
        key=lambda row: _rank(
            source_identity,
            "different-example",
            row.label,
            row.example_id,
            row.image_sha256,
        ),
    )
    pairs: list[tuple[PrismExample, PrismExample]] = []
    while len(pairs) < count and remaining:
        left = remaining.pop(0)
        right_index = next(
            (index for index, row in enumerate(remaining) if row.label != left.label),
            None,
        )
        if right_index is None:
            break
        right = remaining.pop(right_index)
        pairs.append((left, right))
    if len(pairs) != count:
        raise ValueError("PRISM optimization population cannot form balanced pairs")
    return pairs


def _optimization_pairs(
    examples: tuple[PrismExample, ...],
    source_identity: str,
) -> list[_Pair]:
    candidates = _same_candidates(examples, source_identity)
    if len(candidates) < 64:
        raise ValueError("PRISM optimization population has too few same pairs")
    by_label: dict[int, list[tuple[PrismExample, PrismExample]]] = defaultdict(list)
    for pair in candidates:
        by_label[pair[0].label].append(pair)
    same: list[tuple[PrismExample, PrismExample]] = []
    round_ordinal = 0
    while len(same) < 64:
        labels = sorted(
            (label for label, rows in by_label.items() if len(rows) > round_ordinal),
            key=lambda label: _rank(
                source_identity,
                "same-class-round",
                round_ordinal,
                label,
            ),
        )
        if not labels:
            break
        for label in labels:
            same.append(by_label[label][round_ordinal])
            if len(same) == 64:
                break
        round_ordinal += 1
    if len(same) != 64:
        raise ValueError("PRISM optimization population cannot stratify same pairs")
    used_ids = {row.example_id for pair in same for row in pair}
    different = _different_pairs(examples, used_ids, source_identity, 64)
    pairs: list[_Pair] = []
    fold_same_counts = [0] * 4
    fold_same_labels = [set() for _fold in range(4)]
    selected_by_label: dict[int, list[tuple[PrismExample, PrismExample]]] = defaultdict(list)
    for pair in same:
        selected_by_label[pair[0].label].append(pair)
    label_order = sorted(
        selected_by_label,
        key=lambda label: (
            -len(selected_by_label[label]),
            _rank(source_identity, "same-fold-label", label),
        ),
    )
    for label in label_order:
        for left, right in selected_by_label[label]:
            choices = [
                fold
                for fold in range(4)
                if fold_same_counts[fold] < 16 and label not in fold_same_labels[fold]
            ]
            if not choices:
                raise ValueError("PRISM same-pair fold stratification differs")
            fold = min(
                choices,
                key=lambda candidate: (
                    fold_same_counts[candidate],
                    _rank(source_identity, "same-fold", label, candidate),
                ),
            )
            fold_same_counts[fold] += 1
            fold_same_labels[fold].add(label)
            pairs.append(_Pair(fold, left, right, "same", "same"))
    fold_different_counts = [0] * 4
    fold_label_counts: list[dict[int, int]] = [defaultdict(int) for _fold in range(4)]
    for left, right in different:
        choices = [fold for fold in range(4) if fold_different_counts[fold] < 16]
        fold = min(
            choices,
            key=lambda candidate: (
                fold_label_counts[candidate][left.label]
                + fold_label_counts[candidate][right.label],
                fold_different_counts[candidate],
                _rank(
                    source_identity,
                    "different-fold",
                    left.label,
                    right.label,
                    candidate,
                ),
            ),
        )
        fold_different_counts[fold] += 1
        fold_label_counts[fold][left.label] += 1
        fold_label_counts[fold][right.label] += 1
        pairs.append(_Pair(fold, left, right, "different", "different"))
    return pairs


def _caliber_pairs(
    examples: tuple[PrismExample, ...],
    source_identity: str,
) -> list[_Pair]:
    selected: dict[int, list[PrismExample]] = {}
    for label in (82, 83):
        rows = sorted(
            (example for example in examples if example.label == label),
            key=lambda row: _rank(
                source_identity,
                "caliber-example",
                label,
                row.example_id,
                row.image_sha256,
            ),
        )
        if len(rows) < 32:
            raise ValueError("PRISM Caliber population has too few examples")
        selected[label] = rows[:32]
    pairs = [
        *(
            _Pair(4, left, right, "same", "same-82")
            for left, right in zip(selected[82][:16:2], selected[82][1:16:2], strict=True)
        ),
        *(
            _Pair(4, left, right, "same", "same-83")
            for left, right in zip(selected[83][:16:2], selected[83][1:16:2], strict=True)
        ),
        *(
            _Pair(4, left, right, "different", "different")
            for left, right in zip(selected[82][16:], selected[83][16:], strict=True)
        ),
    ]
    return pairs


def _orientations(pairs: list[_Pair], source_identity: str) -> dict[_Pair, bool]:
    grouped: dict[tuple[int, str], list[_Pair]] = defaultdict(list)
    for pair in pairs:
        grouped[(pair.fold, pair.stratum)].append(pair)
    orientations: dict[_Pair, bool] = {}
    for (fold, stratum), rows in grouped.items():
        ordered = sorted(
            rows,
            key=lambda pair: _rank(
                source_identity,
                "orientation",
                fold,
                stratum,
                pair.left.example_id,
                pair.right.example_id,
            ),
        )
        if len(ordered) % 2:
            raise ValueError("PRISM orientation stratum differs")
        for index, pair in enumerate(ordered):
            orientations[pair] = index >= len(ordered) // 2
    return orientations


def _generation_seed(
    source_identity: str,
    pair_ordinal: int,
    channel: str,
    left_payload_sha256: str,
    right_payload_sha256: str,
    left_first: bool,
) -> int:
    return int.from_bytes(
        _rank(
            source_identity,
            "generation-seed",
            pair_ordinal,
            channel,
            left_payload_sha256,
            right_payload_sha256,
            left_first,
        )[:8],
        "little",
    )


def build_prism_schedules(
    optimization_examples: tuple[PrismExample, ...],
    caliber_examples: tuple[PrismExample, ...],
    *,
    source_identity: str,
) -> tuple[tuple[PrismObservationRow, ...], tuple[PrismScoringRow, ...]]:
    """Build fixed truth-separated optimization and Caliber pair schedules."""

    _validate_examples(optimization_examples, caliber_examples, source_identity)
    pairs = _optimization_pairs(optimization_examples, source_identity)
    pairs.extend(_caliber_pairs(caliber_examples, source_identity))
    orientations = _orientations(pairs, source_identity)
    ordered: list[_Pair] = []
    for fold in range(5):
        ordered.extend(
            sorted(
                (pair for pair in pairs if pair.fold == fold),
                key=lambda pair: _rank(
                    source_identity,
                    "pair-order",
                    fold,
                    pair.stratum,
                    pair.left.example_id,
                    pair.right.example_id,
                ),
            )
        )
    if len(ordered) != 160:
        raise ValueError("PRISM pair cardinality differs")

    observations: list[PrismObservationRow] = []
    scoring: list[PrismScoringRow] = []
    seeds: set[int] = set()
    for pair_ordinal, pair in enumerate(ordered):
        scoring.append(
            PrismScoringRow(
                pair_ordinal=pair_ordinal,
                fold=pair.fold,
                left_example_id=pair.left.example_id,
                right_example_id=pair.right.example_id,
                left_payload_sha256=pair.left.image_sha256,
                right_payload_sha256=pair.right.image_sha256,
                left_label=pair.left.label,
                right_label=pair.right.label,
                relation=pair.relation,
            )
        )
        for channel in PRISM_CHANNELS:
            seed = _generation_seed(
                source_identity,
                pair_ordinal,
                channel,
                pair.left.image_sha256,
                pair.right.image_sha256,
                orientations[pair],
            )
            if seed in seeds:
                raise ValueError("PRISM generation seed collision")
            seeds.add(seed)
            observations.append(
                PrismObservationRow(
                    pair_ordinal=pair_ordinal,
                    fold=pair.fold,
                    channel=channel,
                    left_payload_sha256=pair.left.image_sha256,
                    right_payload_sha256=pair.right.image_sha256,
                    left_first=orientations[pair],
                    generation_seed=seed,
                )
            )
    used_ids = {
        example_id for row in scoring for example_id in (row.left_example_id, row.right_example_id)
    }
    if len(used_ids) != 320:
        raise ValueError("PRISM pair image reuse differs")
    result = tuple(observations), tuple(scoring)
    validate_prism_schedules(*result, source_identity=source_identity)
    return result


def release_prism_observation_capability(
    observations: tuple[PrismObservationRow, ...],
    scoring_rows: tuple[PrismScoringRow, ...],
    *,
    source_identity: str,
    phase: str,
    calibration_receipt_sha256: str | None = None,
    calibrations: tuple[PrismChannelCalibration, ...] | None = None,
    pilot_observations: tuple[PrismObservation, ...] | None = None,
    pilot_completion_ids: tuple[tuple[int, ...], ...] | None = None,
    protocol: PrismTokenProtocol | None = None,
) -> tuple[PrismObservationCapabilityRow, ...]:
    """Release calibration now or diagnostic rows only after a sealed receipt."""

    if type(observations) is not tuple or any(
        type(row) is not PrismObservationRow for row in observations
    ):
        raise TypeError("PRISM observation capability differs")
    validate_prism_schedules(observations, scoring_rows, source_identity=source_identity)
    if phase == "calibration":
        if any(
            value is not None
            for value in (
                calibration_receipt_sha256,
                calibrations,
                pilot_observations,
                pilot_completion_ids,
                protocol,
            )
        ):
            raise ValueError("PRISM calibration must precede its receipt")
        selected = (row for row in observations if row.fold < 4)
    elif phase == "diagnostic":
        if (
            type(calibration_receipt_sha256) is not str
            or _SHA256.fullmatch(calibration_receipt_sha256) is None
            or type(calibrations) is not tuple
            or type(protocol) is not PrismTokenProtocol
        ):
            raise ValueError("PRISM diagnostic requires a calibration receipt")
        if calibration_receipt_sha256 != prism_calibration_receipt_sha256(
            calibrations, protocol
        ):
            raise ValueError("PRISM diagnostic requires a calibration receipt")
        expected_pilot_rows = tuple(row for row in observations if row.fold == 0)
        if (
            len(expected_pilot_rows) != 32 * len(PRISM_CHANNELS)
            or not expected_pilot_rows
            or
            type(pilot_observations) is not tuple
            or len(pilot_observations) != len(expected_pilot_rows)
            or any(type(row) is not PrismObservation for row in pilot_observations)
            or type(pilot_completion_ids) is not tuple
            or len(pilot_completion_ids) != len(expected_pilot_rows)
        ):
            raise ValueError("PRISM diagnostic requires authenticated pilot observations")
        for observed, expected, completion_ids in zip(
            pilot_observations,
            expected_pilot_rows,
            pilot_completion_ids,
            strict=True,
        ):
            if (
                observed.pair_ordinal != expected.pair_ordinal
                or observed.fold != 0
                or observed.channel != expected.channel
                or observed.left_first is not expected.left_first
                or observed.left_payload_sha256 != expected.left_payload_sha256
                or observed.right_payload_sha256 != expected.right_payload_sha256
                or observed.generation_seed != expected.generation_seed
                or type(observed.protocol_valid) is not bool
            ):
                raise ValueError("PRISM pilot observation binding differs")
            validate_prism_observation(
                observed,
                expected,
                completion_ids,
                protocol,
            )
        valid = sum(row.protocol_valid for row in pilot_observations)
        if valid * 4 < len(pilot_observations) * 3:
            raise ValueError("PRISM pilot protocol-validity gate failed")
        selected = (row for row in observations if row.fold == 4)
    else:
        raise ValueError("PRISM observation phase differs")
    return tuple(
        PrismObservationCapabilityRow(
            pair_handle=_prism_capability_handle(source_identity, row.pair_ordinal),
            channel=row.channel,
            left_payload_sha256=row.left_payload_sha256,
            right_payload_sha256=row.right_payload_sha256,
            left_first=row.left_first,
            generation_seed=row.generation_seed,
        )
        for row in selected
    )


def _prism_capability_handle(source_identity: str, pair_ordinal: int) -> str:
    digest = hashlib.sha256()
    digest.update(b"sfora-prism-capability-handle-v1\0")
    encoded = source_identity.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "little"))
    digest.update(encoded)
    digest.update(pair_ordinal.to_bytes(8, "little"))
    return digest.hexdigest()


def _validate_token_sequence(sequence: tuple[int, ...], field: str) -> None:
    if (
        type(sequence) is not tuple
        or not sequence
        or any(type(token) is not int or not 0 <= token <= 0xFFFF_FFFF for token in sequence)
    ):
        raise ValueError(f"PRISM token protocol {field} differs")


def _validate_token_protocol(protocol: PrismTokenProtocol) -> None:
    if type(protocol) is not PrismTokenProtocol:
        raise TypeError("PRISM token protocol has the wrong concrete type")
    groups = (
        ("channel", protocol.channel_prefixes, len(PRISM_CHANNELS)),
        ("visibility", protocol.visibility_prefixes, 4),
        ("relation", protocol.relation_prefixes, 3),
        ("confidence", protocol.confidence_prefixes, 3),
    )
    for name, sequences, cardinality in groups:
        if type(sequences) is not tuple or len(sequences) != cardinality:
            raise ValueError(f"PRISM token protocol {name} cardinality differs")
        for sequence in sequences:
            _validate_token_sequence(sequence, name)
        for left_index, left in enumerate(sequences):
            for right in sequences[left_index + 1 :]:
                shared = min(len(left), len(right))
                if left[:shared] == right[:shared]:
                    raise ValueError(f"PRISM token protocol {name} overlap differs")
    _validate_token_sequence(protocol.evidence_separator, "separator")
    _validate_token_sequence(protocol.terminal_tokens, "terminal")
    if type(protocol.max_evidence_tokens) is not int or protocol.max_evidence_tokens < 0:
        raise ValueError("PRISM token protocol evidence bound differs")


def _match_prefix(
    completion: tuple[int, ...],
    cursor: int,
    sequences: tuple[tuple[int, ...], ...],
    field: str,
) -> tuple[int, int]:
    matches = [
        (index, sequence)
        for index, sequence in enumerate(sequences)
        if completion[cursor : cursor + len(sequence)] == sequence
    ]
    if len(matches) != 1:
        raise ValueError(f"PRISM completion {field} differs")
    index, sequence = matches[0]
    return index, cursor + len(sequence)


def _completion_sha256(completion_ids: tuple[int, ...]) -> str:
    digest = hashlib.sha256()
    digest.update(len(completion_ids).to_bytes(8, "little"))
    for token in completion_ids:
        digest.update(token.to_bytes(4, "little"))
    return digest.hexdigest()


def _validate_observation_row(row: PrismObservationRow) -> None:
    if (
        type(row) is not PrismObservationRow
        or row.channel not in PRISM_CHANNELS
        or type(row.pair_ordinal) is not int
        or type(row.fold) is not int
        or _SHA256.fullmatch(row.left_payload_sha256) is None
        or _SHA256.fullmatch(row.right_payload_sha256) is None
        or type(row.left_first) is not bool
        or type(row.generation_seed) is not int
    ):
        raise ValueError("PRISM observation row authority differs")


def _validate_completion_ids(completion_ids: tuple[int, ...]) -> None:
    if type(completion_ids) is not tuple or any(
        type(token) is not int or not 0 <= token <= 0xFFFF_FFFF
        for token in completion_ids
    ):
        raise ValueError("PRISM completion token authority differs")


def parse_prism_completion(
    row: PrismObservationRow,
    completion_ids: tuple[int, ...],
    protocol: PrismTokenProtocol,
) -> PrismObservation:
    """Parse one completion by exact token IDs without decoding text."""

    _validate_observation_row(row)
    _validate_token_protocol(protocol)
    _validate_completion_ids(completion_ids)
    if not completion_ids:
        raise ValueError("PRISM completion token authority differs")
    terminal = protocol.terminal_tokens
    if len(completion_ids) <= len(terminal) or completion_ids[-len(terminal) :] != terminal:
        raise ValueError("PRISM completion terminal differs")
    body = completion_ids[: -len(terminal)]
    cursor = 0
    channel_index, cursor = _match_prefix(body, cursor, protocol.channel_prefixes, "channel")
    if PRISM_CHANNELS[channel_index] != row.channel:
        raise ValueError("PRISM completion channel differs")
    visibility_index, cursor = _match_prefix(
        body, cursor, protocol.visibility_prefixes, "visibility"
    )
    relation_index, cursor = _match_prefix(body, cursor, protocol.relation_prefixes, "relation")
    confidence_index, cursor = _match_prefix(
        body, cursor, protocol.confidence_prefixes, "confidence"
    )
    separator = protocol.evidence_separator
    separator_positions = [
        position
        for position in range(cursor, len(body) - len(separator) + 1)
        if body[position : position + len(separator)] == separator
    ]
    if len(separator_positions) != 1:
        raise ValueError("PRISM completion separator differs")
    separator_position = separator_positions[0]
    left_evidence = body[cursor:separator_position]
    right_evidence = body[separator_position + len(separator) :]
    if (
        len(left_evidence) > protocol.max_evidence_tokens
        or len(right_evidence) > protocol.max_evidence_tokens
    ):
        raise ValueError("PRISM completion evidence differs")
    visibility = ((True, True), (True, False), (False, True), (False, False))[visibility_index]
    return PrismObservation(
        pair_ordinal=row.pair_ordinal,
        fold=row.fold,
        channel=row.channel,
        left_first=row.left_first,
        left_payload_sha256=row.left_payload_sha256,
        right_payload_sha256=row.right_payload_sha256,
        generation_seed=row.generation_seed,
        protocol_valid=True,
        left_visible=visibility[0],
        right_visible=visibility[1],
        relation=("same", "different", "indeterminate")[relation_index],
        confidence=("low", "medium", "high")[confidence_index],
        evidence_left_token_ids=left_evidence,
        evidence_right_token_ids=right_evidence,
        completion_sha256=_completion_sha256(completion_ids),
    )


def invalid_prism_observation(
    row: PrismObservationRow,
    completion_ids: tuple[int, ...],
) -> PrismObservation:
    """Bind one syntactically invalid completion without inventing a verdict."""

    _validate_observation_row(row)
    _validate_completion_ids(completion_ids)
    return PrismObservation(
        pair_ordinal=row.pair_ordinal,
        fold=row.fold,
        channel=row.channel,
        left_first=row.left_first,
        left_payload_sha256=row.left_payload_sha256,
        right_payload_sha256=row.right_payload_sha256,
        generation_seed=row.generation_seed,
        protocol_valid=False,
        left_visible=False,
        right_visible=False,
        relation="indeterminate",
        confidence="low",
        evidence_left_token_ids=(),
        evidence_right_token_ids=(),
        completion_sha256=_completion_sha256(completion_ids),
    )


def validate_prism_observation(
    observation: PrismObservation,
    row: PrismObservationRow,
    completion_ids: tuple[int, ...],
    protocol: PrismTokenProtocol,
) -> None:
    """Reparse exact completion IDs and reject any observation-field drift."""

    if type(observation) is not PrismObservation:
        raise TypeError("PRISM observation has the wrong concrete type")
    _validate_observation_row(row)
    _validate_token_protocol(protocol)
    _validate_completion_ids(completion_ids)
    try:
        expected = parse_prism_completion(row, completion_ids, protocol)
    except ValueError:
        expected = invalid_prism_observation(row, completion_ids)
    if observation.completion_sha256 != expected.completion_sha256:
        raise ValueError("PRISM observation completion digest differs")
    if observation != expected:
        raise ValueError("PRISM observation derivation differs")


def validate_prism_schedules(
    observations: tuple[PrismObservationRow, ...],
    scoring: tuple[PrismScoringRow, ...],
    *,
    source_identity: str,
) -> None:
    """Fail closed on schedule shape, truth separation, order, and seed drift."""

    if (
        type(observations) is not tuple
        or type(scoring) is not tuple
        or type(source_identity) is not str
        or not source_identity
    ):
        raise TypeError("PRISM schedule has the wrong concrete type")
    if (
        len(scoring) != 160
        or len(observations) != 160 * len(PRISM_CHANNELS)
        or any(type(row) is not PrismScoringRow for row in scoring)
        or any(type(row) is not PrismObservationRow for row in observations)
    ):
        raise ValueError("PRISM schedule cardinality differs")
    if tuple(row.pair_ordinal for row in scoring) != tuple(range(160)):
        raise ValueError("PRISM scoring order differs")

    example_ids: list[str] = []
    for row in scoring:
        if (
            type(row.pair_ordinal) is not int
            or type(row.fold) is not int
            or type(row.left_example_id) is not str
            or not row.left_example_id
            or type(row.right_example_id) is not str
            or not row.right_example_id
            or _SHA256.fullmatch(row.left_payload_sha256) is None
            or _SHA256.fullmatch(row.right_payload_sha256) is None
            or type(row.left_label) is not int
            or type(row.right_label) is not int
            or type(row.relation) is not str
            or row.relation not in {"same", "different"}
            or row.relation != ("same" if row.left_label == row.right_label else "different")
        ):
            raise ValueError("PRISM scoring authority differs")
        if row.fold in range(4):
            if not 0 <= row.left_label <= 48 or not 0 <= row.right_label <= 48:
                raise ValueError("PRISM optimization scoring authority differs")
        elif row.fold == 4:
            if row.left_label not in {82, 83} or row.right_label not in {82, 83}:
                raise ValueError("PRISM Caliber scoring authority differs")
        else:
            raise ValueError("PRISM scoring fold differs")
        example_ids.extend((row.left_example_id, row.right_example_id))
    if len(set(example_ids)) != 320:
        raise ValueError("PRISM scoring image reuse differs")

    for fold in range(4):
        rows = [row for row in scoring if row.fold == fold]
        if (
            len(rows) != 32
            or sum(row.relation == "same" for row in rows) != 16
            or sum(row.relation == "different" for row in rows) != 16
        ):
            raise ValueError("PRISM optimization fold balance differs")
    caliber = [row for row in scoring if row.fold == 4]
    if (
        len(caliber) != 32
        or sum((row.left_label, row.right_label) == (82, 82) for row in caliber) != 8
        or sum((row.left_label, row.right_label) == (83, 83) for row in caliber) != 8
        or sum(row.relation == "different" for row in caliber) != 16
    ):
        raise ValueError("PRISM Caliber balance differs")

    seeds: set[int] = set()
    payload_pairs: list[tuple[str, str]] = []
    orientations: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for pair_ordinal, score_row in enumerate(scoring):
        start = pair_ordinal * len(PRISM_CHANNELS)
        pair_rows = observations[start : start + len(PRISM_CHANNELS)]
        if (
            tuple(row.pair_ordinal for row in pair_rows) != (pair_ordinal,) * len(PRISM_CHANNELS)
            or tuple(row.fold for row in pair_rows) != (score_row.fold,) * len(PRISM_CHANNELS)
            or tuple(row.channel for row in pair_rows) != PRISM_CHANNELS
            or len({row.left_first for row in pair_rows}) != 1
            or len({(row.left_payload_sha256, row.right_payload_sha256) for row in pair_rows}) != 1
        ):
            raise ValueError("PRISM observation pair authority differs")
        for row in pair_rows:
            if (
                type(row.pair_ordinal) is not int
                or type(row.fold) is not int
                or type(row.channel) is not str
                or _SHA256.fullmatch(row.left_payload_sha256) is None
                or _SHA256.fullmatch(row.right_payload_sha256) is None
                or type(row.left_first) is not bool
                or type(row.generation_seed) is not int
            ):
                raise ValueError("PRISM observation concrete authority differs")
            expected_seed = _generation_seed(
                source_identity,
                pair_ordinal,
                row.channel,
                row.left_payload_sha256,
                row.right_payload_sha256,
                row.left_first,
            )
            if row.generation_seed != expected_seed:
                raise ValueError("PRISM observation seed authority differs")
            if row.generation_seed in seeds:
                raise ValueError("PRISM observation seed collision")
            seeds.add(row.generation_seed)
        payload_pairs.append((pair_rows[0].left_payload_sha256, pair_rows[0].right_payload_sha256))
        if payload_pairs[-1] != (
            score_row.left_payload_sha256,
            score_row.right_payload_sha256,
        ):
            raise ValueError("PRISM observation-to-scoring payload binding differs")
        if score_row.fold < 4:
            stratum = score_row.relation
        elif score_row.relation == "different":
            stratum = "different"
        else:
            stratum = f"same-{score_row.left_label}"
        orientations[(score_row.fold, stratum)].append(pair_rows[0].left_first)
    payloads = [digest for pair in payload_pairs for digest in pair]
    if len(set(payloads)) != 320:
        raise ValueError("PRISM observation payload reuse differs")
    if any(
        len(values) % 2
        or values.count(False) != len(values) // 2
        or values.count(True) != len(values) // 2
        for values in orientations.values()
    ):
        raise ValueError("PRISM observation orientation balance differs")


def _validate_panel_inputs(
    observations: tuple[PrismObservation, ...],
    scoring_rows: tuple[PrismScoringRow, ...],
    *,
    source_identity: str,
) -> None:
    if (
        type(observations) is not tuple
        or type(scoring_rows) is not tuple
        or len(observations) != 160 * len(PRISM_CHANNELS)
        or len(scoring_rows) != 160
        or any(type(row) is not PrismObservation for row in observations)
        or any(type(row) is not PrismScoringRow for row in scoring_rows)
        or type(source_identity) is not str
        or not source_identity
    ):
        raise ValueError("PRISM panel cardinality differs")
    if tuple(row.pair_ordinal for row in scoring_rows) != tuple(range(160)):
        raise ValueError("PRISM panel scoring order differs")
    seeds: set[int] = set()
    orientations: dict[tuple[int, str], list[bool]] = defaultdict(list)
    for pair_ordinal, score_row in enumerate(scoring_rows):
        if (
            type(score_row.fold) is not int
            or score_row.fold != (pair_ordinal // 32)
            or type(score_row.left_example_id) is not str
            or not score_row.left_example_id
            or type(score_row.right_example_id) is not str
            or not score_row.right_example_id
            or _SHA256.fullmatch(score_row.left_payload_sha256) is None
            or _SHA256.fullmatch(score_row.right_payload_sha256) is None
            or type(score_row.left_label) is not int
            or type(score_row.right_label) is not int
            or (
                score_row.fold < 4
                and not (
                    0 <= score_row.left_label <= 48
                    and 0 <= score_row.right_label <= 48
                )
            )
            or (
                score_row.fold == 4
                and (
                    score_row.left_label not in {82, 83}
                    or score_row.right_label not in {82, 83}
                )
            )
            or score_row.relation
            != ("same" if score_row.left_label == score_row.right_label else "different")
        ):
            raise ValueError("PRISM panel scoring authority differs")
        start = pair_ordinal * len(PRISM_CHANNELS)
        pair_rows = observations[start : start + len(PRISM_CHANNELS)]
        if (
            tuple(row.pair_ordinal for row in pair_rows) != (pair_ordinal,) * len(PRISM_CHANNELS)
            or tuple(row.fold for row in pair_rows) != (score_row.fold,) * len(PRISM_CHANNELS)
            or tuple(row.channel for row in pair_rows) != PRISM_CHANNELS
            or len({row.left_first for row in pair_rows}) != 1
            or any(
                (
                    row.left_payload_sha256,
                    row.right_payload_sha256,
                )
                != (
                    score_row.left_payload_sha256,
                    score_row.right_payload_sha256,
                )
                for row in pair_rows
            )
        ):
            raise ValueError("PRISM panel observation order or payload binding differs")
        for row in pair_rows:
            if (
                type(row.left_first) is not bool
                or _SHA256.fullmatch(row.left_payload_sha256) is None
                or _SHA256.fullmatch(row.right_payload_sha256) is None
                or type(row.generation_seed) is not int
                or type(row.protocol_valid) is not bool
                or type(row.left_visible) is not bool
                or type(row.right_visible) is not bool
                or row.relation not in {"same", "different", "indeterminate"}
                or row.confidence not in {"low", "medium", "high"}
                or type(row.evidence_left_token_ids) is not tuple
                or type(row.evidence_right_token_ids) is not tuple
                or any(
                    type(token) is not int or not 0 <= token <= 0xFFFF_FFFF
                    for token in (row.evidence_left_token_ids + row.evidence_right_token_ids)
                )
                or _SHA256.fullmatch(row.completion_sha256) is None
            ):
                raise ValueError("PRISM panel observation authority differs")
            if row.generation_seed in seeds:
                raise ValueError("PRISM panel observation seed collision")
            seeds.add(row.generation_seed)
            expected_seed = _generation_seed(
                source_identity,
                pair_ordinal,
                row.channel,
                score_row.left_payload_sha256,
                score_row.right_payload_sha256,
                row.left_first,
            )
            if row.generation_seed != expected_seed:
                raise ValueError("PRISM panel observation seed authority differs")
            if not row.protocol_valid and (
                row.left_visible
                or row.right_visible
                or row.relation != "indeterminate"
                or row.confidence != "low"
                or row.evidence_left_token_ids
                or row.evidence_right_token_ids
            ):
                raise ValueError("PRISM invalid observation derivation differs")
        if score_row.fold < 4:
            stratum = score_row.relation
        elif score_row.relation == "different":
            stratum = "different"
        else:
            stratum = f"same-{score_row.left_label}"
        orientations[(score_row.fold, stratum)].append(pair_rows[0].left_first)
    for fold in range(5):
        rows = [row for row in scoring_rows if row.fold == fold]
        if (
            len(rows) != 32
            or sum(row.relation == "same" for row in rows) != 16
            or sum(row.relation == "different" for row in rows) != 16
        ):
            raise ValueError("PRISM panel truth balance differs")
    caliber_rows = [row for row in scoring_rows if row.fold == 4]
    if (
        sum((row.left_label, row.right_label) == (82, 82) for row in caliber_rows) != 8
        or sum((row.left_label, row.right_label) == (83, 83) for row in caliber_rows) != 8
    ):
        raise ValueError("PRISM Caliber truth authority differs")
    if any(
        len(values) % 2
        or values.count(False) != len(values) // 2
        or values.count(True) != len(values) // 2
        for values in orientations.values()
    ):
        raise ValueError("PRISM panel orientation balance differs")


def _auc(scores: np.ndarray, truth: np.ndarray) -> float:
    negative = scores[truth == 0]
    positive = scores[truth == 1]
    if negative.size == 0 or positive.size == 0:
        raise ValueError("PRISM AUC requires both truth strata")
    comparisons = positive[:, None] - negative[None, :]
    return float(
        (np.count_nonzero(comparisons > 0) + 0.5 * np.count_nonzero(comparisons == 0))
        / comparisons.size
    )


def _relation_index(relation: str) -> int:
    return {"same": 0, "different": 1, "indeterminate": 2}[relation]


def _probability_different(counts: np.ndarray, relation_index: int) -> float:
    likelihood_same = (counts[relation_index, 0] + 0.5) / (float(counts[:, 0].sum()) + 1.5)
    likelihood_different = (counts[relation_index, 1] + 0.5) / (float(counts[:, 1].sum()) + 1.5)
    return float(likelihood_different / (likelihood_same + likelihood_different))


def _binary_log_loss(probability: float, truth: int) -> float:
    clipped = min(max(probability, 1e-15), 1.0 - 1e-15)
    return -math.log(clipped if truth else 1.0 - clipped)


def _calibration_counts(
    rows: list[tuple[PrismObservation, int]],
) -> np.ndarray:
    counts = np.zeros((3, 2), dtype=np.int64)
    for observation, truth in rows:
        if observation.protocol_valid:
            counts[_relation_index(observation.relation), truth] += 1
    return counts


def calibrate_prism_channels(
    observations: tuple[PrismObservation, ...],
    scoring_rows: tuple[PrismScoringRow, ...],
    *,
    source_identity: str,
) -> tuple[PrismChannelCalibration, ...]:
    """Calibrate eight fixed channels without using pilot or Caliber truth."""

    _validate_panel_inputs(
        observations, scoring_rows, source_identity=source_identity
    )
    result: list[PrismChannelCalibration] = []
    for channel in PRISM_CHANNELS:
        rows = [
            (observation, int(scoring_rows[observation.pair_ordinal].relation == "different"))
            for observation in observations
            if observation.channel == channel and observation.fold in {1, 2, 3}
        ]
        if len(rows) != 96:
            raise ValueError("PRISM calibration channel cardinality differs")
        counts = _calibration_counts(rows)
        visible = sum(
            observation.protocol_valid and observation.left_visible and observation.right_visible
            for observation, _truth in rows
        )
        loo_losses: list[float] = []
        for observation, truth in rows:
            if not observation.protocol_valid:
                probability = 0.5
            else:
                remaining = counts.copy()
                remaining[_relation_index(observation.relation), truth] -= 1
                probability = _probability_different(
                    remaining, _relation_index(observation.relation)
                )
            loo_losses.append(_binary_log_loss(probability, truth))
        loo_improvement = math.log(2.0) - math.fsum(loo_losses) / len(loo_losses)
        fold_improvements: list[float] = []
        for fold in (1, 2, 3):
            training = [row for row in rows if row[0].fold != fold]
            evaluation = [row for row in rows if row[0].fold == fold]
            training_counts = _calibration_counts(training)
            losses = [
                _binary_log_loss(
                    _probability_different(training_counts, _relation_index(observation.relation))
                    if observation.protocol_valid
                    else 0.5,
                    truth,
                )
                for observation, truth in evaluation
            ]
            fold_improvements.append(math.log(2.0) - math.fsum(losses) / len(losses))
        visibility_ppm = visible * 1_000_000 // len(rows)
        eligible = (
            visibility_ppm >= 500_000
            and loo_improvement >= 0.02
            and all(value > 0.0 for value in fold_improvements)
        )
        result.append(
            PrismChannelCalibration(
                channel=channel,
                counts=tuple((int(counts[index, 0]), int(counts[index, 1])) for index in range(3)),
                visibility_ppm=visibility_ppm,
                loo_log_loss_improvement=float(loo_improvement),
                fold_log_loss_improvements=tuple(fold_improvements),
                eligible=eligible,
            )
        )
    return tuple(result)


def _validate_calibrations(
    calibrations: tuple[PrismChannelCalibration, ...],
) -> None:
    if (
        type(calibrations) is not tuple
        or len(calibrations) != len(PRISM_CHANNELS)
        or any(type(row) is not PrismChannelCalibration for row in calibrations)
        or tuple(row.channel for row in calibrations) != PRISM_CHANNELS
    ):
        raise ValueError("PRISM calibration authority differs")
    for row in calibrations:
        if (
            type(row.counts) is not tuple
            or len(row.counts) != 3
            or any(
                type(cell) is not tuple
                or len(cell) != 2
                or any(type(value) is not int or value < 0 for value in cell)
                for cell in row.counts
            )
            or type(row.visibility_ppm) is not int
            or not 0 <= row.visibility_ppm <= 1_000_000
            or type(row.loo_log_loss_improvement) is not float
            or not math.isfinite(row.loo_log_loss_improvement)
            or type(row.fold_log_loss_improvements) is not tuple
            or len(row.fold_log_loss_improvements) != 3
            or any(
                type(value) is not float or not math.isfinite(value)
                for value in row.fold_log_loss_improvements
            )
            or type(row.eligible) is not bool
        ):
            raise ValueError("PRISM calibration concrete authority differs")
        expected_eligible = (
            row.visibility_ppm >= 500_000
            and row.loo_log_loss_improvement >= 0.02
            and all(value > 0.0 for value in row.fold_log_loss_improvements)
        )
        if row.eligible is not expected_eligible:
            raise ValueError("PRISM calibration eligibility derivation differs")


def canonical_prism_calibration_bytes(
    calibrations: tuple[PrismChannelCalibration, ...],
    protocol: PrismTokenProtocol,
) -> bytes:
    """Serialize the exact immutable calibration receipt."""

    _validate_calibrations(calibrations)
    _validate_token_protocol(protocol)
    value = {
        "channels": [asdict(row) for row in calibrations],
        "protocol_sha256": hashlib.sha256(
            json.dumps(
                asdict(protocol),
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest(),
        "schema": "sfora-prism-channel-calibration-v1",
    }
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def prism_calibration_receipt_sha256(
    calibrations: tuple[PrismChannelCalibration, ...],
    protocol: PrismTokenProtocol,
) -> str:
    """Return the content identity required to release diagnostic capability."""

    return hashlib.sha256(
        canonical_prism_calibration_bytes(calibrations, protocol)
    ).hexdigest()


def _cue_contribution(
    calibration: PrismChannelCalibration,
    observation: PrismObservation,
) -> float:
    if not calibration.eligible or not observation.protocol_valid:
        return 0.0
    counts = np.asarray(calibration.counts, dtype=np.float64)
    relation_index = _relation_index(observation.relation)
    same = (counts[relation_index, 0] + 0.5) / (float(counts[:, 0].sum()) + 1.5)
    different = (counts[relation_index, 1] + 0.5) / (float(counts[:, 1].sum()) + 1.5)
    return math.log(different) - math.log(same)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _valid_orientation_ppm(
    observations: tuple[PrismObservation, ...],
    fold: int,
) -> tuple[int, int]:
    return tuple(
        sum(
            row.protocol_valid and row.left_first is orientation
            for row in observations
            if row.fold == fold
        )
        * 1_000_000
        // sum(row.left_first is orientation for row in observations if row.fold == fold)
        for orientation in (False, True)
    )


def _passes_prism_cue_gates(
    *,
    improvement_lower: float,
    auc_lower: float,
    eligible_channels: tuple[str, ...],
    valid_orientation_ppm: tuple[int, int],
    orientation_gap: float,
) -> bool:
    return all(
        _prism_cue_gate_components(
            improvement_lower=improvement_lower,
            auc_lower=auc_lower,
            eligible_channels=eligible_channels,
            valid_orientation_ppm=valid_orientation_ppm,
            orientation_gap=orientation_gap,
        )
    )


def _prism_cue_gate_components(
    *,
    improvement_lower: float,
    auc_lower: float,
    eligible_channels: tuple[str, ...],
    valid_orientation_ppm: tuple[int, int],
    orientation_gap: float,
) -> tuple[bool, bool, bool, bool]:
    log_loss_gate_passed = improvement_lower >= 0.05
    auc_gate_passed = auc_lower >= 0.80
    channel_gate_passed = len(eligible_channels) >= 4 and len(
        set(eligible_channels)
        & {PRISM_CHANNELS[0], PRISM_CHANNELS[1], PRISM_CHANNELS[7]}
    ) >= 2
    orientation_gate_passed = (
        min(valid_orientation_ppm) >= 750_000 and orientation_gap <= 0.10
    )
    return (
        log_loss_gate_passed,
        auc_gate_passed,
        channel_gate_passed,
        orientation_gate_passed,
    )


def _build_cue_result(
    calibrations: tuple[PrismChannelCalibration, ...],
    observations: tuple[PrismObservation, ...],
    scoring_rows: tuple[PrismScoringRow, ...],
    bootstrap_seed: bytes,
    source_identity: str,
    calibration_receipt_sha256: str,
    protocol: PrismTokenProtocol,
) -> PrismCueResult:
    _validate_panel_inputs(
        observations, scoring_rows, source_identity=source_identity
    )
    _validate_calibrations(calibrations)
    expected_calibrations = calibrate_prism_channels(
        observations, scoring_rows, source_identity=source_identity
    )
    if calibrations != expected_calibrations:
        raise ValueError("PRISM calibration evidence differs")
    if (
        type(calibration_receipt_sha256) is not str
        or _SHA256.fullmatch(calibration_receipt_sha256) is None
        or calibration_receipt_sha256
        != prism_calibration_receipt_sha256(calibrations, protocol)
    ):
        raise ValueError("PRISM calibration receipt differs")
    if type(bootstrap_seed) is not bytes or not bootstrap_seed:
        raise TypeError("PRISM cue bootstrap seed differs")
    pilot_rows = tuple(row for row in observations if row.fold == 0)
    if (
        not pilot_rows
        or sum(row.protocol_valid for row in pilot_rows) * 4 < len(pilot_rows) * 3
    ):
        raise ValueError("PRISM pilot protocol-validity gate failed")
    calibration_by_channel = {row.channel: row for row in calibrations}
    eligible_channels = tuple(row.channel for row in calibrations if row.eligible)
    eligible_count = len(eligible_channels)
    pair_scores: list[float] = []
    pair_truth: list[int] = []
    pair_orientations: list[bool] = []
    contributions: dict[str, list[float]] = {channel: [] for channel in PRISM_CHANNELS}
    for pair_ordinal in range(128, 160):
        start = pair_ordinal * len(PRISM_CHANNELS)
        rows = observations[start : start + len(PRISM_CHANNELS)]
        channel_values = [
            _cue_contribution(calibration_by_channel[row.channel], row) for row in rows
        ]
        for channel, value in zip(PRISM_CHANNELS, channel_values, strict=True):
            contributions[channel].append(value)
        pair_scores.append(
            math.fsum(channel_values) / eligible_count if eligible_count else 0.0
        )
        pair_truth.append(int(scoring_rows[pair_ordinal].relation == "different"))
        pair_orientations.append(rows[0].left_first)
    scores = np.asarray(pair_scores, dtype=np.float64)
    truth = np.asarray(pair_truth, dtype=np.int64)
    losses = np.asarray(
        [
            _binary_log_loss(_sigmoid(score), int(label))
            for score, label in zip(scores, truth, strict=True)
        ],
        dtype=np.float64,
    )
    improvement = float(math.log(2.0) - float(losses.mean()))
    auc = _auc(scores, truth)
    seed_material = (
        b"sfora-prism-cue-bootstrap-v1\0"
        + len(bootstrap_seed).to_bytes(8, "little")
        + bootstrap_seed
    )
    generator = np.random.Generator(
        np.random.PCG64(int.from_bytes(hashlib.sha256(seed_material).digest()[:16], "little"))
    )
    bootstrap_improvement = np.empty(10_000, dtype=np.float64)
    bootstrap_auc = np.empty(10_000, dtype=np.float64)
    completed = 0
    while completed < 10_000:
        indexes = generator.integers(0, 32, size=32)
        sampled_truth = truth[indexes]
        if set(sampled_truth.tolist()) != {0, 1}:
            continue
        bootstrap_improvement[completed] = math.log(2.0) - float(losses[indexes].mean())
        bootstrap_auc[completed] = _auc(scores[indexes], sampled_truth)
        completed += 1
    bootstrap_improvement.sort(kind="stable")
    bootstrap_auc.sort(kind="stable")
    lower_index = math.floor(0.05 * (10_000 - 1))
    valid_orientation_ppm = _valid_orientation_ppm(observations, 4)
    orientation_aucs = [
        _auc(
            scores[np.asarray(pair_orientations) == orientation],
            truth[np.asarray(pair_orientations) == orientation],
        )
        for orientation in (False, True)
    ]
    agreements: list[tuple[str, str, str, int, float | None]] = []
    for left_index, left_channel in enumerate(eligible_channels):
        for right_channel in eligible_channels[left_index + 1 :]:
            left_values = np.sign(np.asarray(contributions[left_channel]))
            right_values = np.sign(np.asarray(contributions[right_channel]))
            for truth_value, truth_name in ((0, "same"), (1, "different")):
                selected = (
                    (truth == truth_value)
                    & (left_values != 0.0)
                    & (right_values != 0.0)
                )
                agreements.append(
                    (
                        left_channel,
                        right_channel,
                        truth_name,
                        int(np.count_nonzero(selected)),
                        float(np.mean(left_values[selected] == right_values[selected]))
                        if np.any(selected)
                        else None,
                    )
                )
    improvement_lower = float(bootstrap_improvement[lower_index])
    auc_lower = float(bootstrap_auc[lower_index])
    orientation_gap = abs(orientation_aucs[0] - orientation_aucs[1])
    (
        log_loss_gate_passed,
        auc_gate_passed,
        channel_gate_passed,
        orientation_gate_passed,
    ) = _prism_cue_gate_components(
        improvement_lower=improvement_lower,
        auc_lower=auc_lower,
        eligible_channels=eligible_channels,
        valid_orientation_ppm=valid_orientation_ppm,
        orientation_gap=orientation_gap,
    )
    passed = _passes_prism_cue_gates(
        improvement_lower=improvement_lower,
        auc_lower=auc_lower,
        eligible_channels=eligible_channels,
        valid_orientation_ppm=valid_orientation_ppm,
        orientation_gap=orientation_gap,
    )
    return PrismCueResult(
        bootstrap_draws=10_000,
        bootstrap_seed_sha256=hashlib.sha256(bootstrap_seed).hexdigest(),
        calibration_receipt_sha256=calibration_receipt_sha256,
        pair_scores=tuple(pair_scores),
        pair_truth=tuple(pair_truth),
        mean_log_loss_improvement=improvement,
        mean_log_loss_improvement_lower_95=improvement_lower,
        auc=auc,
        auc_lower_95=auc_lower,
        valid_orientation_ppm=valid_orientation_ppm,
        orientation_auc_gap=orientation_gap,
        eligible_channels=eligible_channels,
        conditional_agreement=tuple(agreements),
        log_loss_gate_passed=log_loss_gate_passed,
        auc_gate_passed=auc_gate_passed,
        channel_gate_passed=channel_gate_passed,
        orientation_gate_passed=orientation_gate_passed,
        cue_classification=(
            "cue-pass"
            if passed
            else "rank-cue-only"
            if auc_gate_passed and channel_gate_passed and orientation_gate_passed
            else "probability-cue-only"
            if log_loss_gate_passed and channel_gate_passed and orientation_gate_passed
            else "cue-fail"
        ),
        passed=passed,
    )


def score_prism_cue_panel(
    calibrations: tuple[PrismChannelCalibration, ...],
    observations: tuple[PrismObservation, ...],
    scoring_rows: tuple[PrismScoringRow, ...],
    *,
    bootstrap_seed: bytes,
    source_identity: str,
    calibration_receipt_sha256: str,
    protocol: PrismTokenProtocol,
) -> PrismCueResult:
    """Score the sealed diagnostic once with literal preregistered gates."""

    return _build_cue_result(
        calibrations,
        observations,
        scoring_rows,
        bootstrap_seed,
        source_identity,
        calibration_receipt_sha256,
        protocol,
    )


def validate_prism_cue_result(
    result: PrismCueResult,
    calibrations: tuple[PrismChannelCalibration, ...],
    observations: tuple[PrismObservation, ...],
    scoring_rows: tuple[PrismScoringRow, ...],
    *,
    bootstrap_seed: bytes,
    source_identity: str,
    calibration_receipt_sha256: str,
    protocol: PrismTokenProtocol,
) -> None:
    """Recompute every diagnostic statistic and gate from authenticated rows."""

    if type(result) is not PrismCueResult:
        raise TypeError("PRISM cue result has the wrong concrete type")
    expected = _build_cue_result(
        calibrations,
        observations,
        scoring_rows,
        bootstrap_seed,
        source_identity,
        calibration_receipt_sha256,
        protocol,
    )
    if result != expected:
        raise ValueError("PRISM cue result derivation differs")
