"""Leakage-safe native-pixel reachability evidence for the Cars twin pair."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import numpy as np

_HEX = frozenset("0123456789abcdef")
_VIEW_COUNT = 9
_ERROR_REDUCTION_GATE = 0.25
_BALANCED_ACCURACY_GATE = 0.75
_SWAP_PERMUTATION_DRAWS = 10_000
_SWAP_PERMUTATION_GATE = 0.05


@dataclass(frozen=True, slots=True)
class NativeTwinAuthority:
    """Exact source, model, image, label, and descriptor identities."""

    source_identity: str
    checkpoint_sha256: str
    model_revision: str
    probe_revision: str
    probe_tree_digest: str
    example_ids: tuple[str, ...]
    image_sha256: tuple[str, ...]
    labels: tuple[int, ...]
    crop_long_edges: tuple[tuple[int, ...], ...]
    global_descriptor_sha256: str
    control_descriptor_sha256: str
    native_descriptor_sha256: str


@dataclass(frozen=True, slots=True)
class NativeTwinResult:
    """Paired evidence isolating native pixels from view-count and scorer effects."""

    source_identity: str
    authority_sha256: str
    checkpoint_sha256: str
    model_revision: str
    global_descriptor_sha256: str
    control_descriptor_sha256: str
    native_descriptor_sha256: str
    example_count: int
    view_count: int
    control_noop_crop_count: int
    swap_permutation_draws: int
    swap_permutation_seed_sha256: str
    swap_permutation_p_value: float
    global_candidate_ids: tuple[str, ...]
    global_candidate_labels: tuple[int, ...]
    control_candidate_ids: tuple[str, ...]
    control_candidate_labels: tuple[int, ...]
    native_candidate_ids: tuple[str, ...]
    native_candidate_labels: tuple[int, ...]
    global_errors: int
    control_errors: int
    native_errors: int
    global_balanced_accuracy: float
    control_balanced_accuracy: float
    native_balanced_accuracy: float
    native_error_reduction: float
    rescues: int
    harms: int
    unchanged: int
    mcnemar_discordant: int
    mcnemar_p_value: float
    classification: str
    passed: bool


def _lower_hex(value: object, length: int) -> bool:
    return type(value) is str and len(value) == length and set(value).issubset(_HEX)


def native_descriptor_sha256(descriptors: np.ndarray) -> str:
    """Hash one concrete fp32 descriptor tensor with shape and byte framing."""

    if type(descriptors) is not np.ndarray or descriptors.dtype != np.float32:
        raise TypeError("native descriptor digest requires one fp32 array")
    if descriptors.ndim not in (2, 3):
        raise ValueError("native descriptor digest rank differs")
    if not bool(np.isfinite(descriptors).all()):
        raise ValueError("native descriptor digest requires finite values")
    concrete = np.ascontiguousarray(descriptors.astype("<f4", copy=False))
    digest = hashlib.sha256()
    digest.update(b"sfora-native-descriptor-v2\0")
    digest.update(concrete.ndim.to_bytes(1, "little"))
    for extent in concrete.shape:
        digest.update(int(extent).to_bytes(8, "little"))
    digest.update(concrete.tobytes(order="C"))
    return digest.hexdigest()


def _validate_authority(authority: NativeTwinAuthority) -> None:
    if type(authority) is not NativeTwinAuthority:
        raise TypeError("native twin authority has the wrong concrete type")
    if type(authority.source_identity) is not str or not authority.source_identity:
        raise ValueError("native twin source identity differs")
    if not _lower_hex(authority.checkpoint_sha256, 64) or not _lower_hex(
        authority.model_revision, 40
    ) or not _lower_hex(authority.probe_revision, 40) or not _lower_hex(
        authority.probe_tree_digest, 64
    ):
        raise ValueError("native twin model identity differs")
    descriptor_digests = (
        authority.global_descriptor_sha256,
        authority.control_descriptor_sha256,
        authority.native_descriptor_sha256,
    )
    if any(not _lower_hex(value, 64) for value in descriptor_digests):
        raise ValueError("native twin descriptor authority differs")
    count = len(authority.example_ids)
    if (
        type(authority.example_ids) is not tuple
        or type(authority.image_sha256) is not tuple
        or type(authority.labels) is not tuple
        or type(authority.crop_long_edges) is not tuple
        or count != len(authority.image_sha256)
        or count != len(authority.labels)
        or count != len(authority.crop_long_edges)
        or count < 40
    ):
        raise ValueError("native twin authority shape differs")
    if (
        any(type(value) is not str or not value for value in authority.example_ids)
        or len(set(authority.example_ids)) != count
        or any(not _lower_hex(value, 64) for value in authority.image_sha256)
        or len(set(authority.image_sha256)) != count
    ):
        raise ValueError("native twin image identity differs")
    if any(type(label) is not int for label in authority.labels) or set(
        authority.labels
    ) != {82, 83}:
        raise ValueError("native twin labels differ")
    if any(
        type(edges) is not tuple
        or len(edges) != _VIEW_COUNT
        or any(type(value) is not int or value < 2 for value in edges)
        for edges in authority.crop_long_edges
    ):
        raise ValueError("native twin crop authority differs")
    if min(authority.labels.count(82), authority.labels.count(83)) < 20:
        raise ValueError("native twin labels have insufficient support")


def _authority_bytes(authority: NativeTwinAuthority) -> bytes:
    _validate_authority(authority)
    return (
        json.dumps(
            asdict(authority),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _authority_sha256(authority: NativeTwinAuthority) -> str:
    return hashlib.sha256(_authority_bytes(authority)).hexdigest()


def _validate_descriptors(
    authority: NativeTwinAuthority,
    global_plane: np.ndarray,
    control: np.ndarray,
    native: np.ndarray,
) -> None:
    planes = (global_plane, control, native)
    if any(type(plane) is not np.ndarray for plane in planes):
        raise TypeError("native twin descriptors must be concrete arrays")
    if any(plane.dtype != np.float32 for plane in planes):
        raise TypeError("native twin descriptors must be fp32")
    count = len(authority.example_ids)
    if (
        global_plane.ndim != 2
        or global_plane.shape[0] != count
        or global_plane.shape[1] < 2
    ):
        raise ValueError("native twin global shape differs")
    expected = (count, _VIEW_COUNT, global_plane.shape[1])
    if control.ndim != 3 or control.shape != expected:
        raise ValueError("native twin control requires exactly nine views")
    if native.ndim != 3 or native.shape != expected:
        raise ValueError("native twin native plane requires exactly nine views")
    if not all(bool(np.isfinite(plane).all()) for plane in planes):
        raise ValueError("native twin descriptors must be finite")
    norms = (
        np.linalg.vector_norm(global_plane.astype(np.float64), axis=1),
        np.linalg.vector_norm(control.astype(np.float64), axis=2),
        np.linalg.vector_norm(native.astype(np.float64), axis=2),
    )
    if any(not bool(np.all(np.abs(value - 1.0) <= 1e-5)) for value in norms):
        raise ValueError("native twin descriptors must be unit normalized")
    observed = tuple(native_descriptor_sha256(plane) for plane in planes)
    registered = (
        authority.global_descriptor_sha256,
        authority.control_descriptor_sha256,
        authority.native_descriptor_sha256,
    )
    if observed != registered:
        raise ValueError("native twin descriptor digest differs")


def _normalized_rows(plane: np.ndarray) -> np.ndarray:
    flat = plane.reshape(plane.shape[0], -1).astype(np.float64)
    norms = np.linalg.vector_norm(flat, axis=1)
    if bool((norms <= 0.0).any()) or not bool(np.isfinite(norms).all()):
        raise ValueError("native twin flattened descriptor norm differs")
    return flat / norms[:, None]


def _predictions(plane: np.ndarray, tie_keys: tuple[str, ...]) -> np.ndarray:
    normalized = _normalized_rows(plane)
    scores = normalized @ normalized.T
    return _candidates_from_scores(scores, tie_keys)


def _candidates_from_scores(
    scores: np.ndarray, tie_keys: tuple[str, ...]
) -> np.ndarray:
    scores = scores.copy()
    np.fill_diagonal(scores, -math.inf)
    candidates = np.empty(scores.shape[0], dtype=np.int64)
    for query, row in enumerate(scores):
        maximum = float(np.max(row))
        tied = np.flatnonzero(row == maximum).tolist()
        candidates[query] = min(tied, key=lambda index: tie_keys[index])
    return candidates


def _balanced_accuracy(correct: np.ndarray, labels: np.ndarray) -> float:
    return 0.5 * math.fsum(
        float(np.mean(correct[labels == label])) for label in (82, 83)
    )


def _mcnemar_p_value(rescues: int, harms: int) -> float:
    discordant = rescues + harms
    if discordant == 0:
        return 1.0
    numerator = sum(math.comb(discordant, value) for value in range(rescues, discordant + 1))
    return math.ldexp(float(numerator), -discordant)


def _swap_permutation_p_value(
    authority: NativeTwinAuthority,
    control: np.ndarray,
    native: np.ndarray,
    labels: np.ndarray,
    observed_improvement: int,
) -> tuple[float, str]:
    control_rows = _normalized_rows(control)
    native_rows = _normalized_rows(native)
    control_control = control_rows @ control_rows.T
    control_native = control_rows @ native_rows.T
    native_native = native_rows @ native_rows.T
    seed = b"sfora-native-pixel-swap-v2\0" + _authority_bytes(authority)
    seed_digest = hashlib.sha256(seed).hexdigest()
    generator = np.random.Generator(
        np.random.PCG64(int.from_bytes(hashlib.sha256(seed).digest()[:16], "little"))
    )
    extreme = 0
    for _draw in range(_SWAP_PERMUTATION_DRAWS):
        swapped = generator.integers(0, 2, size=labels.size, dtype=np.int8).astype(bool)
        left = swapped[:, None]
        right = swapped[None, :]
        first = np.where(
            left,
            np.where(right, native_native, control_native.T),
            np.where(right, control_native, control_control),
        )
        second = np.where(
            left,
            np.where(right, control_control, control_native),
            np.where(right, control_native.T, native_native),
        )
        first_candidates = _candidates_from_scores(first, authority.image_sha256)
        second_candidates = _candidates_from_scores(second, authority.image_sha256)
        first_errors = int(np.count_nonzero(labels[first_candidates] != labels))
        second_errors = int(np.count_nonzero(labels[second_candidates] != labels))
        extreme += int(first_errors - second_errors >= observed_improvement)
    return (
        (extreme + 1) / float(_SWAP_PERMUTATION_DRAWS + 1),
        seed_digest,
    )


def _build_result(
    authority: NativeTwinAuthority,
    global_plane: np.ndarray,
    control: np.ndarray,
    native: np.ndarray,
) -> NativeTwinResult:
    _validate_authority(authority)
    _validate_descriptors(authority, global_plane, control, native)
    labels = np.asarray(authority.labels, dtype=np.int64)
    tie_keys = authority.image_sha256
    global_candidates = _predictions(global_plane, tie_keys)
    control_candidates = _predictions(control, tie_keys)
    native_candidates = _predictions(native, tie_keys)
    global_correct = labels[global_candidates] == labels
    control_correct = labels[control_candidates] == labels
    native_correct = labels[native_candidates] == labels
    global_errors = int(np.count_nonzero(~global_correct))
    control_errors = int(np.count_nonzero(~control_correct))
    native_errors = int(np.count_nonzero(~native_correct))
    rescues = int(np.count_nonzero(~control_correct & native_correct))
    harms = int(np.count_nonzero(control_correct & ~native_correct))
    reduction = (
        (control_errors - native_errors) / control_errors if control_errors else 0.0
    )

    p_value = _mcnemar_p_value(rescues, harms)
    swap_p_value, swap_seed_sha256 = _swap_permutation_p_value(
        authority,
        control,
        native,
        labels,
        control_errors - native_errors,
    )
    native_balanced_accuracy = _balanced_accuracy(native_correct, labels)
    passed = (
        control_errors > 0
        and reduction >= _ERROR_REDUCTION_GATE
        and native_balanced_accuracy >= _BALANCED_ACCURACY_GATE
        and rescues > harms
        and swap_p_value <= _SWAP_PERMUTATION_GATE
    )

    def candidate_ids(candidates: np.ndarray) -> tuple[str, ...]:
        return tuple(authority.example_ids[int(index)] for index in candidates)

    def candidate_labels(candidates: np.ndarray) -> tuple[int, ...]:
        return tuple(int(labels[int(index)]) for index in candidates)

    return NativeTwinResult(
        source_identity=authority.source_identity,
        authority_sha256=_authority_sha256(authority),
        checkpoint_sha256=authority.checkpoint_sha256,
        model_revision=authority.model_revision,
        global_descriptor_sha256=authority.global_descriptor_sha256,
        control_descriptor_sha256=authority.control_descriptor_sha256,
        native_descriptor_sha256=authority.native_descriptor_sha256,
        example_count=len(labels),
        view_count=_VIEW_COUNT,
        control_noop_crop_count=sum(
            value <= 256 for edges in authority.crop_long_edges for value in edges
        ),
        swap_permutation_draws=_SWAP_PERMUTATION_DRAWS,
        swap_permutation_seed_sha256=swap_seed_sha256,
        swap_permutation_p_value=swap_p_value,
        global_candidate_ids=candidate_ids(global_candidates),
        global_candidate_labels=candidate_labels(global_candidates),
        control_candidate_ids=candidate_ids(control_candidates),
        control_candidate_labels=candidate_labels(control_candidates),
        native_candidate_ids=candidate_ids(native_candidates),
        native_candidate_labels=candidate_labels(native_candidates),
        global_errors=global_errors,
        control_errors=control_errors,
        native_errors=native_errors,
        global_balanced_accuracy=_balanced_accuracy(global_correct, labels),
        control_balanced_accuracy=_balanced_accuracy(control_correct, labels),
        native_balanced_accuracy=native_balanced_accuracy,
        native_error_reduction=reduction,
        rescues=rescues,
        harms=harms,
        unchanged=len(labels) - rescues - harms,
        mcnemar_discordant=rescues + harms,
        mcnemar_p_value=p_value,
        classification=(
            "native-pixel-cue-pass" if passed else "native-pixel-cue-fail"
        ),
        passed=passed,
    )


def score_native_twin_probe(
    authority: NativeTwinAuthority,
    global_plane: np.ndarray,
    control: np.ndarray,
    native: np.ndarray,
) -> NativeTwinResult:
    """Score the frozen matched-control native-pixel probe."""

    return _build_result(authority, global_plane, control, native)


def validate_native_twin_result(
    result: NativeTwinResult,
    authority: NativeTwinAuthority,
    global_plane: np.ndarray,
    control: np.ndarray,
    native: np.ndarray,
) -> None:
    """Recompute every prediction, statistic, inference result, and gate."""

    if type(result) is not NativeTwinResult:
        raise TypeError("native twin result has the wrong concrete type")
    if result != _build_result(authority, global_plane, control, native):
        raise ValueError("native twin result derivation differs")


def canonical_native_twin_result_bytes(result: NativeTwinResult) -> bytes:
    """Serialize one strict claim-ineligible result with a trailing newline."""

    if type(result) is not NativeTwinResult:
        raise TypeError("native twin result has the wrong concrete type")
    value = {
        "claim_eligible": False,
        "result": asdict(result),
        "schema": "sfora-native-twin-probe-v2",
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


def validate_canonical_native_twin_result_bytes(
    raw: bytes,
    *,
    expected: NativeTwinResult,
) -> NativeTwinResult:
    """Require exact canonical bytes for an independently recomputed result."""

    if type(raw) is not bytes or type(expected) is not NativeTwinResult:
        raise TypeError("native twin canonical artifact types differ")
    if raw != canonical_native_twin_result_bytes(expected):
        raise ValueError("native twin canonical artifact differs")
    return expected
