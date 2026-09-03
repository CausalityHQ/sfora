"""Authority and result algebra for the claim-ineligible VMD F0 screen."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, cast

VMD_F0_ERROR_COUNT = 103
VMD_F0_CALIBER_COUNT = 63
VMD_F0_OTHER_COUNT = 40
VMD_F0_OVERALL_WINS_GATE = 62
VMD_F0_CALIBER_WINS_GATE = 38
VMD_F0_OTHER_WINS_GATE = 24
VMD_F0_REPEAT_ORDINALS = (0, 102)
VMD_F0_MAX_CUDA_BYTES = 56 * 1024**3
VMD_F0_MAX_RSS_BYTES = 64 * 1024**3
VMD_F0_MAX_ELAPSED_NS = 900_000_000_000
VMD_F0_M2_SHA256 = "64d491607d4dac144b31edac3a182130e6f94f994a272f612c195a7a72d55611"
VMD_F0_M4_QUERY_SHA256 = "b2fc9baf52feb3917554241b5aba205a7a10799ef6e3742e128e7aa173b33c67"
VMD_F0_DATASET_REVISION = "9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40"
VMD_F0_EXAMPLES_SHA256 = "83a7800ee948a816e2fb9a2c9163027d9e90f167abc90052bf220619fa32240f"
VMD_F0_DESCRIPTOR_SHA256 = "4031dc2da90588dcc39005eab92c6c519f3058c581222421ca917501dd3df071"
VMD_F0_M4_DESCRIPTOR_FILE_SHA256 = (
    "2cb7c25e803ec66fca879f08f77813a4e98c7bb3e52b510e75014c66c203e214"
)


class ExampleIdentity(Protocol):
    """Minimum ordered Cars example identity used by the authority loader."""

    example_id: str
    label: int


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _read_registered_json(path: Path, digest: str, *, name: str) -> dict[str, object]:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != digest:
        raise ValueError(f"VMD F0 {name} digest differs")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"VMD F0 {name} JSON differs") from error
    if type(value) is not dict or raw != _canonical(value):
        raise ValueError(f"VMD F0 {name} canonical bytes differ")
    return cast(dict[str, object], value)


def _integer(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise ValueError(f"VMD F0 {name} differs")
    return value


def _string(value: object, *, name: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(f"VMD F0 {name} differs")
    return value


def _finite(value: object, *, name: str) -> float:
    if type(value) is not float or not math.isfinite(value):
        raise ValueError(f"VMD F0 {name} differs")
    return value


@dataclass(frozen=True, slots=True)
class VmdF0Candidate:
    """One outcome-blind true-versus-wrong candidate comparison."""

    ordinal: int
    query_position: int
    query_example_id: str
    query_label: int
    true_position: int
    true_example_id: str
    wrong_position: int
    wrong_example_id: str
    wrong_label: int
    is_caliber_block: bool


def build_vmd_f0_candidates(
    m2: object,
    m4: object,
    examples: tuple[ExampleIdentity, ...],
) -> tuple[VmdF0Candidate, ...]:
    """Cross-bind M2 errors to M4's frozen nearest same-class candidates."""

    if type(m2) is not dict or type(m4) is not dict or len(examples) != 1345:
        raise ValueError("VMD F0 authority differs")
    m2 = cast(dict[str, object], m2)
    m4 = cast(dict[str, object], m4)
    if (
        m2.get("schema") != "sfora-frozen-substrate-errors-v1"
        or m2.get("claim_eligible") is not False
        or m2.get("dataset") != "cars"
        or m2.get("dataset_revision") != VMD_F0_DATASET_REVISION
        or m2.get("dataset_examples_sha256") != VMD_F0_EXAMPLES_SHA256
        or m2.get("descriptor_sha256") != VMD_F0_DESCRIPTOR_SHA256
        or m2.get("cell") != "siglip-so400m"
        or m2.get("split") != "train"
        or m2.get("error_count") != VMD_F0_ERROR_COUNT
        or m4.get("schema") != "sfora-pass209-m4-query-evidence-v1"
        or m4.get("claim_eligible") is not False
        or m4.get("cell") != "siglip-so400m"
        or m4.get("dataset_examples_sha256") != VMD_F0_EXAMPLES_SHA256
        or m4.get("descriptor_file_sha256") != VMD_F0_M4_DESCRIPTOR_FILE_SHA256
        or m4.get("query_block") != 32
    ):
        raise ValueError("VMD F0 authority differs")
    errors = m2.get("errors")
    rows = m4.get("historical_cuda_rows")
    if (
        type(errors) is not list
        or len(errors) != VMD_F0_ERROR_COUNT
        or type(rows) is not list
        or len(rows) != len(examples)
    ):
        raise ValueError("VMD F0 row authority differs")
    candidates: list[VmdF0Candidate] = []
    previous_query = -1
    for ordinal, raw_error in enumerate(errors):
        if type(raw_error) is not dict:
            raise ValueError("VMD F0 error row differs")
        error = cast(dict[str, object], raw_error)
        query_position = _integer(error.get("query_position"), name="query position")
        wrong_position = _integer(error.get("nearest_position"), name="wrong position")
        if (
            query_position <= previous_query
            or not 0 <= query_position < len(examples)
            or not 0 <= wrong_position < len(examples)
            or type(rows[query_position]) is not dict
        ):
            raise ValueError("VMD F0 error order differs")
        previous_query = query_position
        row = cast(dict[str, object], rows[query_position])
        true_position = _integer(row.get("best_same_position"), name="true position")
        query = examples[query_position]
        wrong = examples[wrong_position]
        if (
            row.get("query_position") != query_position
            or row.get("query_example_id") != query.example_id
            or row.get("query_label") != query.label
            or error.get("query_example_id") != query.example_id
            or error.get("query_label") != query.label
            or row.get("nearest_position") != wrong_position
            or row.get("nearest_example_id") != wrong.example_id
            or row.get("nearest_label") != wrong.label
            or error.get("nearest_example_id") != wrong.example_id
            or error.get("nearest_label") != wrong.label
        ):
            raise ValueError("VMD F0 wrong neighbor authority differs")
        if (
            not 0 <= true_position < len(examples)
            or true_position == query_position
            or wrong_position == query_position
            or examples[true_position].label != query.label
            or wrong.label == query.label
        ):
            raise ValueError("VMD F0 true neighbor authority differs")
        caliber = {query.label, wrong.label} == {82, 83}
        candidates.append(
            VmdF0Candidate(
                ordinal=ordinal,
                query_position=query_position,
                query_example_id=query.example_id,
                query_label=query.label,
                true_position=true_position,
                true_example_id=examples[true_position].example_id,
                wrong_position=wrong_position,
                wrong_example_id=wrong.example_id,
                wrong_label=wrong.label,
                is_caliber_block=caliber,
            )
        )
    if sum(row.is_caliber_block for row in candidates) != VMD_F0_CALIBER_COUNT:
        raise ValueError("VMD F0 Caliber block authority differs")
    return tuple(candidates)


def load_vmd_f0_candidates(
    m2_path: Path,
    m4_query_path: Path,
    examples: tuple[ExampleIdentity, ...],
) -> tuple[VmdF0Candidate, ...]:
    """Authenticate exact registered evidence and return frozen candidates."""

    return build_vmd_f0_candidates(
        _read_registered_json(m2_path, VMD_F0_M2_SHA256, name="M2 manifest"),
        _read_registered_json(m4_query_path, VMD_F0_M4_QUERY_SHA256, name="M4 query"),
        examples,
    )


@dataclass(frozen=True, slots=True)
class VmdF0Observation:
    """One forward-only teacher comparison for a frozen retrieval error."""

    source_commit: str
    fixture_source_commit: str
    model_revision: str
    launch_authority_sha256: str
    fixture_sha256: str
    m2_manifest_sha256: str
    m4_query_sha256: str
    ordinal: int
    query_position: int
    query_example_id: str
    query_label: int
    true_position: int
    true_example_id: str
    wrong_position: int
    wrong_example_id: str
    is_caliber_block: bool
    true_same_score: float
    true_different_score: float
    wrong_same_score: float
    wrong_different_score: float
    elapsed_ns: int
    peak_cuda_reserved_bytes: int
    peak_rss_bytes: int

    @property
    def preference_margin(self) -> float:
        return (self.true_same_score - self.true_different_score) - (
            self.wrong_same_score - self.wrong_different_score
        )

    @property
    def win(self) -> bool:
        return self.preference_margin > 0.0

    @property
    def branch_scores(self) -> tuple[float, float, float, float]:
        return (
            self.true_same_score,
            self.true_different_score,
            self.wrong_same_score,
            self.wrong_different_score,
        )

    def validated(self) -> VmdF0Observation:
        for name, value, width in (
            ("source commit", self.source_commit, 40),
            ("fixture source commit", self.fixture_source_commit, 40),
            ("model revision", self.model_revision, 40),
            ("launch authority", self.launch_authority_sha256, 64),
            ("fixture", self.fixture_sha256, 64),
            ("M2 manifest", self.m2_manifest_sha256, 64),
            ("M4 query", self.m4_query_sha256, 64),
        ):
            if (
                type(value) is not str
                or len(value) != width
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"VMD F0 {name} identity differs")
        if (
            self.m2_manifest_sha256 != VMD_F0_M2_SHA256
            or self.m4_query_sha256 != VMD_F0_M4_QUERY_SHA256
        ):
            raise ValueError("VMD F0 evidence identity differs")
        if (
            type(self.ordinal) is not int
            or not 0 <= self.ordinal < VMD_F0_ERROR_COUNT
            or any(
                type(value) is not int or value < 0
                for value in (self.query_position, self.true_position, self.wrong_position)
            )
            or len({self.query_position, self.true_position, self.wrong_position}) != 3
            or any(
                not _string(value, name="example id")
                for value in (
                    self.query_example_id,
                    self.true_example_id,
                    self.wrong_example_id,
                )
            )
            or type(self.query_label) is not int
            or type(self.is_caliber_block) is not bool
            or any(
                type(value) is not int or value <= 0
                for value in (
                    self.elapsed_ns,
                    self.peak_cuda_reserved_bytes,
                    self.peak_rss_bytes,
                )
            )
        ):
            raise ValueError("VMD F0 observation authority differs")
        for value in self.branch_scores:
            _finite(value, name="branch score")
        _finite(self.preference_margin, name="preference margin")
        return self

    @classmethod
    def from_mapping(cls, value: object) -> VmdF0Observation:
        if type(value) is not dict:
            raise ValueError("VMD F0 observation schema differs")
        expected = set(cls.__dataclass_fields__) | {
            "schema",
            "claim_eligible",
            "official_test_access",
            "generated_tokens",
            "language_model_gradients",
        }
        raw = cast(dict[str, object], value)
        if (
            set(raw) != expected
            or raw["schema"] != "sfora-vmd-f0-observation-v1"
            or raw["claim_eligible"] is not False
            or raw["official_test_access"] is not False
            or raw["generated_tokens"] != 0
            or raw["language_model_gradients"] != 0
        ):
            raise ValueError("VMD F0 observation schema differs")
        try:
            return cls(**{name: raw[name] for name in cls.__dataclass_fields__}).validated()
        except TypeError as error:
            raise ValueError("VMD F0 observation differs") from error


def canonical_vmd_f0_observation_bytes(observation: VmdF0Observation) -> bytes:
    """Serialize one validated forward-only observation."""

    observation.validated()
    return _canonical(
        {
            "schema": "sfora-vmd-f0-observation-v1",
            "claim_eligible": False,
            "official_test_access": False,
            "generated_tokens": 0,
            "language_model_gradients": 0,
            **asdict(observation),
        }
    )


@dataclass(frozen=True, slots=True)
class VmdF0Result:
    """Recomputed terminal result for all 103 frozen retrieval errors."""

    observations: tuple[VmdF0Observation, ...]
    repeat_checked_ordinals: tuple[int, int]
    repeat_branch_scores: tuple[tuple[float, float, float, float], ...]
    overall_wins: int
    caliber_wins: int
    other_wins: int
    overall_win_ppm: int
    caliber_win_ppm: int
    other_win_ppm: int
    total_elapsed_ns: int
    peak_cuda_reserved_bytes: int
    peak_rss_bytes: int
    outcome: str
    passed: bool

    @classmethod
    def from_observations(
        cls,
        observations: tuple[VmdF0Observation, ...],
        *,
        repeat_checked_ordinals: tuple[int, int],
        repeat_branch_scores: tuple[tuple[float, float, float, float], ...],
        total_elapsed_ns: int,
    ) -> VmdF0Result:
        if (
            type(observations) is not tuple
            or len(observations) != VMD_F0_ERROR_COUNT
            or tuple(row.ordinal for row in observations) != tuple(range(VMD_F0_ERROR_COUNT))
            or sum(row.is_caliber_block for row in observations) != VMD_F0_CALIBER_COUNT
        ):
            raise ValueError("VMD F0 observation cardinality differs")
        for row in observations:
            row.validated()
        identities = {
            (
                row.source_commit,
                row.fixture_source_commit,
                row.model_revision,
                row.launch_authority_sha256,
                row.fixture_sha256,
                row.m2_manifest_sha256,
                row.m4_query_sha256,
            )
            for row in observations
        }
        if len(identities) != 1:
            raise ValueError("VMD F0 observation identity differs")
        if (
            repeat_checked_ordinals != VMD_F0_REPEAT_ORDINALS
            or type(repeat_branch_scores) is not tuple
            or len(repeat_branch_scores) != 2
            or repeat_branch_scores
            != tuple(observations[ordinal].branch_scores for ordinal in VMD_F0_REPEAT_ORDINALS)
        ):
            raise ValueError("VMD F0 replay authority differs")
        if type(total_elapsed_ns) is not int or not 0 < total_elapsed_ns <= VMD_F0_MAX_ELAPSED_NS:
            raise ValueError("VMD F0 resource authority differs")
        peak_cuda = max(row.peak_cuda_reserved_bytes for row in observations)
        peak_rss = max(row.peak_rss_bytes for row in observations)
        if peak_cuda > VMD_F0_MAX_CUDA_BYTES or peak_rss > VMD_F0_MAX_RSS_BYTES:
            raise ValueError("VMD F0 resource authority differs")
        caliber = tuple(row for row in observations if row.is_caliber_block)
        other = tuple(row for row in observations if not row.is_caliber_block)
        overall_wins = sum(row.win for row in observations)
        caliber_wins = sum(row.win for row in caliber)
        other_wins = sum(row.win for row in other)
        passed = (
            overall_wins >= VMD_F0_OVERALL_WINS_GATE
            and caliber_wins >= VMD_F0_CALIBER_WINS_GATE
            and other_wins >= VMD_F0_OTHER_WINS_GATE
        )
        return cls(
            observations=observations,
            repeat_checked_ordinals=repeat_checked_ordinals,
            repeat_branch_scores=repeat_branch_scores,
            overall_wins=overall_wins,
            caliber_wins=caliber_wins,
            other_wins=other_wins,
            overall_win_ppm=overall_wins * 1_000_000 // VMD_F0_ERROR_COUNT,
            caliber_win_ppm=caliber_wins * 1_000_000 // VMD_F0_CALIBER_COUNT,
            other_win_ppm=other_wins * 1_000_000 // VMD_F0_OTHER_COUNT,
            total_elapsed_ns=total_elapsed_ns,
            peak_cuda_reserved_bytes=peak_cuda,
            peak_rss_bytes=peak_rss,
            outcome="teacher-target-supported" if passed else "teacher-target-rejected",
            passed=passed,
        )


def canonical_vmd_f0_result_bytes(result: VmdF0Result) -> bytes:
    """Serialize and revalidate one canonical claim-ineligible result."""

    if type(result) is not VmdF0Result:
        raise ValueError("VMD F0 result differs")
    rebuilt = VmdF0Result.from_observations(
        result.observations,
        repeat_checked_ordinals=result.repeat_checked_ordinals,
        repeat_branch_scores=result.repeat_branch_scores,
        total_elapsed_ns=result.total_elapsed_ns,
    )
    if rebuilt != result:
        raise ValueError("VMD F0 result recomputation differs")
    return _canonical(
        {
            "schema": "sfora-vmd-f0-result-v1",
            "claim_eligible": False,
            "official_test_access": False,
            "generated_tokens": 0,
            "language_model_gradients": 0,
            "gates": {
                "overall_wins": VMD_F0_OVERALL_WINS_GATE,
                "caliber_wins": VMD_F0_CALIBER_WINS_GATE,
                "other_wins": VMD_F0_OTHER_WINS_GATE,
                "maximum_cuda_reserved_bytes": VMD_F0_MAX_CUDA_BYTES,
                "maximum_rss_bytes": VMD_F0_MAX_RSS_BYTES,
                "maximum_total_elapsed_ns": VMD_F0_MAX_ELAPSED_NS,
            },
            **asdict(result),
        }
    )
