"""Pure, capability-separated PRISM cue-measurement evidence."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass

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

    pair_ordinal: int
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
    left_visible: bool
    right_visible: bool
    relation: str
    confidence: str
    evidence_left_token_ids: tuple[int, ...]
    evidence_right_token_ids: tuple[int, ...]
    completion_sha256: str


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
    *,
    phase: str,
    calibration_receipt_sha256: str | None = None,
) -> tuple[PrismObservationCapabilityRow, ...]:
    """Release calibration now or diagnostic rows only after a sealed receipt."""

    if type(observations) is not tuple or any(
        type(row) is not PrismObservationRow for row in observations
    ):
        raise TypeError("PRISM observation capability differs")
    if phase == "calibration":
        if calibration_receipt_sha256 is not None:
            raise ValueError("PRISM calibration must precede its receipt")
        selected = (row for row in observations if row.fold < 4)
    elif phase == "diagnostic":
        if (
            type(calibration_receipt_sha256) is not str
            or _SHA256.fullmatch(calibration_receipt_sha256) is None
        ):
            raise ValueError("PRISM diagnostic requires a calibration receipt")
        selected = (row for row in observations if row.fold == 4)
    else:
        raise ValueError("PRISM observation phase differs")
    return tuple(
        PrismObservationCapabilityRow(
            pair_ordinal=row.pair_ordinal,
            channel=row.channel,
            left_payload_sha256=row.left_payload_sha256,
            right_payload_sha256=row.right_payload_sha256,
            left_first=row.left_first,
            generation_seed=row.generation_seed,
        )
        for row in selected
    )


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


def parse_prism_completion(
    row: PrismObservationRow,
    completion_ids: tuple[int, ...],
    protocol: PrismTokenProtocol,
) -> PrismObservation:
    """Parse one completion by exact token IDs without decoding text."""

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
    _validate_token_protocol(protocol)
    if (
        type(completion_ids) is not tuple
        or not completion_ids
        or any(type(token) is not int or not 0 <= token <= 0xFFFF_FFFF for token in completion_ids)
    ):
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
        left_visible=visibility[0],
        right_visible=visibility[1],
        relation=("same", "different", "indeterminate")[relation_index],
        confidence=("low", "medium", "high")[confidence_index],
        evidence_left_token_ids=left_evidence,
        evidence_right_token_ids=right_evidence,
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
    expected = parse_prism_completion(row, completion_ids, protocol)
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
