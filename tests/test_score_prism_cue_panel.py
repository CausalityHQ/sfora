from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

from sfora.prism_measurement import (
    PRISM_CHANNELS,
    PrismExample,
    PrismMeasurementAuthority,
    PrismTokenProtocol,
    build_prism_schedules,
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        + b"\n"
    )


def _protocol() -> PrismTokenProtocol:
    return PrismTokenProtocol(
        channel_prefixes=tuple((100 + index,) for index in range(8)),
        visibility_prefixes=((200,), (201,), (202,), (203,)),
        relation_prefixes=((300,), (301,), (302,)),
        confidence_prefixes=((400,), (401,), (402,)),
        evidence_separator=(500,),
        terminal_tokens=(600, 601),
        max_evidence_tokens=4,
    )


def _examples() -> tuple[tuple[PrismExample, ...], tuple[PrismExample, ...]]:
    optimization = tuple(
        PrismExample(
            example_id=f"optimization-{label}-{ordinal}",
            label=label,
            image_sha256=f"{1 + label * 8 + ordinal:064x}",
        )
        for label in range(49)
        for ordinal in range(8)
    )
    caliber = tuple(
        PrismExample(
            example_id=f"caliber-{label}-{ordinal}",
            label=label,
            image_sha256=f"{10_000 + (label - 82) * 32 + ordinal:064x}",
        )
        for label in (82, 83)
        for ordinal in range(32)
    )
    return optimization, caliber


def _completion(channel: str, relation: str) -> list[int]:
    return [
        100 + PRISM_CHANNELS.index(channel),
        200,
        300 if relation == "same" else 301,
        402,
        9,
        500,
        10,
        600,
        601,
    ]


def _write_fixture(directory: Path) -> dict[str, Path]:
    source_identity = "offline-prism-fixture"
    observation_rows, scoring_rows = build_prism_schedules(
        *_examples(), source_identity=source_identity
    )
    protocol = _protocol()
    values: dict[str, object] = {
        "observation": {
            "rows": [asdict(row) for row in observation_rows],
            "schema": "sfora-prism-observation-manifest-v1",
        },
        "scoring": {
            "rows": [asdict(row) for row in scoring_rows],
            "schema": "sfora-prism-scoring-manifest-v1",
        },
        "protocol": {
            "protocol": asdict(protocol),
            "schema": "sfora-prism-token-protocol-v1",
        },
        "completion": {
            "rows": [
                {
                    "channel": row.channel,
                    "completion_ids": _completion(
                        row.channel, scoring_rows[row.pair_ordinal].relation
                    ),
                    "pair_ordinal": row.pair_ordinal,
                }
                for row in observation_rows
            ],
            "schema": "sfora-prism-completion-bundle-v1",
        },
    }
    paths: dict[str, Path] = {}
    for role, value in values.items():
        path = directory / f"{role}.json"
        path.write_bytes(_canonical(value))
        paths[role] = path
    authority = PrismMeasurementAuthority(
        source_commit="1" * 40,
        source_tree_sha256="2" * 64,
        dataset_revision="dataset-revision",
        dataset_manifest_sha256="3" * 64,
        model_revision="model-revision",
        processor_revision="processor-revision",
        tokenizer_revision="tokenizer-revision",
        prompt_bundle_sha256="4" * 64,
        token_protocol_sha256=hashlib.sha256(paths["protocol"].read_bytes()).hexdigest(),
        observation_manifest_sha256=hashlib.sha256(
            paths["observation"].read_bytes()
        ).hexdigest(),
        scoring_manifest_sha256=hashlib.sha256(paths["scoring"].read_bytes()).hexdigest(),
        completion_bundle_sha256=hashlib.sha256(
            paths["completion"].read_bytes()
        ).hexdigest(),
    )
    authority_path = directory / "authority.json"
    authority_path.write_bytes(
        _canonical(
            {
                "authority": asdict(authority),
                "bootstrap_seed_hex": b"offline-bootstrap-seed".hex(),
                "schema": "sfora-prism-measurement-authority-v1",
                "source_identity": source_identity,
            }
        )
    )
    paths["authority"] = authority_path
    paths["output"] = directory / "result.json"
    return paths


def _command(paths: dict[str, Path]) -> list[str]:
    return [
        sys.executable,
        "scripts/score_prism_cue_panel.py",
        "--observation",
        str(paths["observation"]),
        "--scoring",
        str(paths["scoring"]),
        "--protocol",
        str(paths["protocol"]),
        "--completion",
        str(paths["completion"]),
        "--authority",
        str(paths["authority"]),
        "--output",
        str(paths["output"]),
        "--execute-score",
    ]


def _refresh_authority_digest(
    paths: dict[str, Path], *, role: str, field: str
) -> None:
    authority = json.loads(paths["authority"].read_bytes())
    authority["authority"][field] = hashlib.sha256(paths[role].read_bytes()).hexdigest()
    paths["authority"].write_bytes(_canonical(authority))


def test_offline_prism_scorer_authenticates_inputs_and_writes_once(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    completed = subprocess.run(
        _command(paths),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == ""
    raw = paths["output"].read_bytes()
    value = json.loads(raw)
    assert value["schema"] == "sfora-prism-cue-result-artifact-v1"
    assert value["result"]["cue_classification"] == "cue-pass"
    assert value["result"]["passed"] is True
    assert raw == _canonical(value)

    repeated = subprocess.run(
        _command(paths),
        check=False,
        capture_output=True,
        text=True,
    )
    assert repeated.returncode != 0
    assert "exists" in repeated.stderr
    assert paths["output"].read_bytes() == raw


def test_offline_prism_scorer_refuses_nonmeasurement_capabilities(tmp_path: Path) -> None:
    paths = _write_fixture(tmp_path)
    base = _command(paths)
    for extra in (
        ["--model", "checkpoint.pt"],
        ["--image-root", "/images"],
        ["--dataset-root", "/dataset"],
        ["--network", "https://example.invalid"],
        ["--dgx"],
        ["--clean"],
        ["--test"],
        ["--training"],
    ):
        completed = subprocess.run(
            [*base, *extra],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode != 0
        assert not paths["output"].exists()

    without_execute = [argument for argument in base if argument != "--execute-score"]
    completed = subprocess.run(
        without_execute,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert not paths["output"].exists()

    duplicate = [*base, "--observation", str(paths["observation"])]
    completed = subprocess.run(
        duplicate,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "duplicate" in completed.stderr
    assert not paths["output"].exists()


def test_offline_prism_scorer_authenticates_observation_before_truth_opens(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    observation = json.loads(paths["observation"].read_bytes())
    observation["rows"][0]["generation_seed"] ^= 1
    paths["observation"].write_bytes(_canonical(observation))
    command = _command(paths)
    scoring_index = command.index("--scoring") + 1
    command[scoring_index] = str(tmp_path / "truth-must-not-open.json")

    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "observation digest differs" in completed.stderr
    assert "scoring path" not in completed.stderr
    assert not paths["output"].exists()


def test_offline_prism_scorer_rejects_image_reuse_before_bootstrap(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    scoring = json.loads(paths["scoring"].read_bytes())
    scoring["rows"][1]["left_example_id"] = scoring["rows"][0]["left_example_id"]
    scoring["rows"][1]["right_example_id"] = scoring["rows"][0]["right_example_id"]
    paths["scoring"].write_bytes(_canonical(scoring))
    _refresh_authority_digest(
        paths,
        role="scoring",
        field="scoring_manifest_sha256",
    )

    completed = subprocess.run(
        _command(paths),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "image reuse" in completed.stderr
    assert not paths["output"].exists()


def test_offline_prism_scorer_records_malformed_completion_as_invalid(
    tmp_path: Path,
) -> None:
    paths = _write_fixture(tmp_path)
    completion = json.loads(paths["completion"].read_bytes())
    completion["rows"][128 * len(PRISM_CHANNELS)]["completion_ids"] = [999, 600, 601]
    paths["completion"].write_bytes(_canonical(completion))
    _refresh_authority_digest(
        paths,
        role="completion",
        field="completion_bundle_sha256",
    )

    completed = subprocess.run(
        _command(paths),
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    value = json.loads(paths["output"].read_bytes())
    assert value["result"]["passed"] is True
    assert value["result"]["valid_orientation_ppm"] != [1_000_000, 1_000_000]
