#!/usr/bin/env python3
"""Score authenticated local PRISM evidence without model or data capability."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import fields
from pathlib import Path

from sfora.prism_measurement import (
    PrismMeasurementAuthority,
    PrismMeasurementEvidence,
    PrismObservationRow,
    PrismScoringRow,
    PrismTokenProtocol,
    calibrate_prism_channels,
    canonical_prism_cue_result_bytes,
    invalid_prism_observation,
    parse_prism_completion,
    prism_calibration_receipt_sha256,
    score_prism_cue_panel,
    validate_prism_schedules,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False, description=__doc__)
    parser.add_argument("--observation", type=Path, required=True)
    parser.add_argument("--scoring", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute-score", action="store_true", required=True)
    return parser


def _strict_json(raw: bytes, *, role: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        value: dict[str, object] = {}
        for key, item in items:
            if key in value:
                raise ValueError(f"{role} has duplicate keys")
            value[key] = item
        return value

    try:
        value = json.loads(raw, object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{role} is not valid JSON") from error
    canonical = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )
    if type(value) is not dict or raw != canonical:
        raise ValueError(f"{role} is not canonical JSON")
    return value


def _exact_keys(value: dict[str, object], expected: set[str], *, role: str) -> None:
    if set(value) != expected:
        raise ValueError(f"{role} schema differs")


def _read(path: Path, *, role: str) -> tuple[bytes, dict[str, object]]:
    if not isinstance(path, Path) or not path.is_file() or path.is_symlink():
        raise ValueError(f"{role} path must be an existing regular file")
    raw = path.read_bytes()
    return raw, _strict_json(raw, role=role)


def _authority(value: dict[str, object]) -> tuple[PrismMeasurementAuthority, str, bytes]:
    _exact_keys(
        value,
        {"authority", "bootstrap_seed_hex", "schema", "source_identity"},
        role="authority",
    )
    if value["schema"] != "sfora-prism-measurement-authority-v1":
        raise ValueError("authority schema differs")
    payload = value["authority"]
    if type(payload) is not dict:
        raise ValueError("authority payload differs")
    names = {field.name for field in fields(PrismMeasurementAuthority)}
    _exact_keys(payload, names, role="authority payload")
    authority = PrismMeasurementAuthority(**payload)
    source_identity = value["source_identity"]
    seed_hex = value["bootstrap_seed_hex"]
    if type(source_identity) is not str or not source_identity:
        raise ValueError("authority source identity differs")
    if type(seed_hex) is not str or not seed_hex or len(seed_hex) % 2:
        raise ValueError("authority bootstrap seed differs")
    try:
        seed = bytes.fromhex(seed_hex)
    except ValueError as error:
        raise ValueError("authority bootstrap seed differs") from error
    if not seed:
        raise ValueError("authority bootstrap seed differs")
    return authority, source_identity, seed


def _rows(
    value: dict[str, object],
    *,
    schema: str,
    row_type: type[PrismObservationRow] | type[PrismScoringRow],
    role: str,
) -> tuple[PrismObservationRow, ...] | tuple[PrismScoringRow, ...]:
    _exact_keys(value, {"rows", "schema"}, role=role)
    if value["schema"] != schema or type(value["rows"]) is not list:
        raise ValueError(f"{role} schema differs")
    names = {field.name for field in fields(row_type)}
    output = []
    for item in value["rows"]:
        if type(item) is not dict:
            raise ValueError(f"{role} row differs")
        _exact_keys(item, names, role=f"{role} row")
        output.append(row_type(**item))
    return tuple(output)


def _protocol(value: dict[str, object]) -> PrismTokenProtocol:
    _exact_keys(value, {"protocol", "schema"}, role="protocol")
    if value["schema"] != "sfora-prism-token-protocol-v1":
        raise ValueError("protocol schema differs")
    payload = value["protocol"]
    if type(payload) is not dict:
        raise ValueError("protocol payload differs")
    _exact_keys(
        payload,
        {field.name for field in fields(PrismTokenProtocol)},
        role="protocol payload",
    )
    return PrismTokenProtocol(
        channel_prefixes=tuple(tuple(row) for row in payload["channel_prefixes"]),
        visibility_prefixes=tuple(tuple(row) for row in payload["visibility_prefixes"]),
        relation_prefixes=tuple(tuple(row) for row in payload["relation_prefixes"]),
        confidence_prefixes=tuple(tuple(row) for row in payload["confidence_prefixes"]),
        evidence_separator=tuple(payload["evidence_separator"]),
        terminal_tokens=tuple(payload["terminal_tokens"]),
        max_evidence_tokens=payload["max_evidence_tokens"],
    )


def _observations(
    schedules: tuple[PrismObservationRow, ...],
    value: dict[str, object],
    protocol: PrismTokenProtocol,
) -> tuple:
    _exact_keys(value, {"rows", "schema"}, role="completion bundle")
    if value["schema"] != "sfora-prism-completion-bundle-v1" or type(value["rows"]) is not list:
        raise ValueError("completion bundle schema differs")
    if len(value["rows"]) != len(schedules):
        raise ValueError("completion bundle cardinality differs")
    output = []
    for schedule, item in zip(schedules, value["rows"], strict=True):
        if type(item) is not dict:
            raise ValueError("completion row differs")
        _exact_keys(
            item,
            {"channel", "completion_ids", "pair_ordinal"},
            role="completion row",
        )
        if (
            item["channel"] != schedule.channel
            or item["pair_ordinal"] != schedule.pair_ordinal
            or type(item["completion_ids"]) is not list
        ):
            raise ValueError("completion row binding differs")
        completion_ids = tuple(item["completion_ids"])
        try:
            output.append(parse_prism_completion(schedule, completion_ids, protocol))
        except ValueError:
            output.append(invalid_prism_observation(schedule, completion_ids))
    return tuple(output)


def _authenticate(raw: bytes, expected: str, *, role: str) -> None:
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"{role} digest differs")


def _publish_new(path: Path, raw: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    descriptor = os.open(
        partial,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


def main(argv: list[str] | None = None) -> int:
    tokens = list(sys.argv[1:] if argv is None else argv)
    parser = _parser()
    known_options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option.startswith("--")
    }
    seen: set[str] = set()
    for token in tokens:
        option = token.partition("=")[0]
        if option in known_options:
            if option in seen:
                parser.error(f"duplicate option: {option}")
            seen.add(option)
    arguments = parser.parse_args(tokens)
    if arguments.output.exists() or arguments.output.is_symlink():
        raise FileExistsError(f"output already exists: {arguments.output}")

    _authority_raw, authority_value = _read(arguments.authority, role="authority")
    authority, source_identity, bootstrap_seed = _authority(authority_value)

    observation_raw, observation_value = _read(arguments.observation, role="observation")
    _authenticate(
        observation_raw,
        authority.observation_manifest_sha256,
        role="observation",
    )
    protocol_raw, protocol_value = _read(arguments.protocol, role="protocol")
    _authenticate(protocol_raw, authority.token_protocol_sha256, role="protocol")
    completion_raw, completion_value = _read(arguments.completion, role="completion")
    _authenticate(completion_raw, authority.completion_bundle_sha256, role="completion")

    schedules = _rows(
        observation_value,
        schema="sfora-prism-observation-manifest-v1",
        row_type=PrismObservationRow,
        role="observation",
    )
    protocol = _protocol(protocol_value)
    observations = _observations(schedules, completion_value, protocol)

    scoring_raw, scoring_value = _read(arguments.scoring, role="scoring")
    _authenticate(scoring_raw, authority.scoring_manifest_sha256, role="scoring")
    scoring_rows = _rows(
        scoring_value,
        schema="sfora-prism-scoring-manifest-v1",
        row_type=PrismScoringRow,
        role="scoring",
    )
    validate_prism_schedules(
        schedules,
        scoring_rows,
        source_identity=source_identity,
    )
    evidence = PrismMeasurementEvidence(
        authority=authority,
        observations=observations,
        scoring_rows=scoring_rows,
        protocol=protocol,
        bootstrap_seed=bootstrap_seed,
        source_identity=source_identity,
    )
    calibrations = calibrate_prism_channels(
        observations,
        scoring_rows,
        source_identity=source_identity,
    )
    result = score_prism_cue_panel(
        calibrations,
        observations,
        scoring_rows,
        bootstrap_seed=bootstrap_seed,
        source_identity=source_identity,
        calibration_receipt_sha256=prism_calibration_receipt_sha256(
            calibrations, protocol
        ),
        protocol=protocol,
    )
    raw = canonical_prism_cue_result_bytes(evidence, calibrations, result)
    _publish_new(arguments.output, raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
