"""Deterministic representation reachability evidence for the Cars twin pair."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, cast

import numpy as np

_SCHEMA = "sfora-cars-twin-reachability-v1"
_LABEL_82 = 82
_LABEL_83 = 83
_AUC_GATE = 0.80
_BIC_GATE = 10.0
_HIGH_FRACTION_GATE = 0.25
_HIGH_AUC_GATE = 0.80
_VARIANCE_FLOOR = 1e-8
_EM_ITERATIONS = 128
_MIN_CLASS_COUNT = 20
_BOOTSTRAP_DRAWS = 10_000
_PERMUTATION_DRAWS = 64
_INFERENCE_KEYS = {
    "bootstrap_draws",
    "permutation_draws",
    "bootstrap_seed_sha256",
    "permutation_seed_sha256",
    "lda_auc_lower_95",
    "permutation_extreme_count",
    "permutation_p_value",
    "passed",
}
_COMMIT = re.compile(r"[0-9a-f]{40}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True, slots=True)
class TwinReachabilityAuthority:
    """Exact caller-authenticated identity for one descriptor plane."""

    plane: str
    source_revision: str
    source_tree_digest: str
    dataset_revision: str
    dataset_manifest_sha256: str
    model_name: str
    model_revision: str
    producer_kind: str
    producer_identity: str
    ordered_example_ids_sha256: str
    label_vector_sha256: str
    descriptor_sha256: str


@dataclass(frozen=True, slots=True)
class TwinReachabilityEvidence:
    """Complete deterministic evidence for one descriptor plane."""

    plane: str
    source_count: int
    class_82_count: int
    class_83_count: int
    labels: tuple[int, ...]
    signed_scores: tuple[float, ...]
    full_auc: float
    bic_one: float
    bic_two: float
    bic_improvement: float
    high_evidence_count: int
    high_evidence_fraction: float
    high_evidence_auc: float
    centroid_cue_present: bool
    lda_signed_scores: tuple[float, ...]
    lda_full_auc: float
    cue_present: bool


@dataclass(frozen=True, slots=True)
class TwinReachabilityInference:
    """Uncertainty and refitted-null evidence for one reachability plane."""

    bootstrap_draws: int
    permutation_draws: int
    bootstrap_seed_sha256: str
    permutation_seed_sha256: str
    lda_auc_lower_95: float
    permutation_extreme_count: int
    permutation_p_value: float
    passed: bool


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    negative = scores[labels == _LABEL_82]
    positive = scores[labels == _LABEL_83]
    wins = 0.0
    for positive_score in positive:
        wins += float(np.count_nonzero(positive_score > negative))
        wins += 0.5 * float(np.count_nonzero(positive_score == negative))
    return wins / float(positive.size * negative.size)


def _log_normal(values: np.ndarray, mean: float, variance: float) -> np.ndarray:
    return -0.5 * (
        math.log(2.0 * math.pi * variance) + np.square(values - mean) / variance
    )


def _logsumexp_pair(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    maximum = np.maximum(left, right)
    return maximum + np.log(np.exp(left - maximum) + np.exp(right - maximum))


def _mixture_statistics(
    scores: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, int, float, float]:
    margins = np.abs(scores)
    count = margins.size
    one_mean = float(np.mean(margins, dtype=np.float64))
    one_variance = max(float(np.mean(np.square(margins - one_mean))), _VARIANCE_FLOOR)
    one_log_likelihood = float(np.sum(_log_normal(margins, one_mean, one_variance)))
    bic_one = 2.0 * math.log(count) - 2.0 * one_log_likelihood

    sorted_margins = np.sort(margins, kind="stable")
    lower_index = int(math.floor(0.25 * (count - 1)))
    upper_index = int(math.floor(0.75 * (count - 1)))
    means = np.asarray(
        [sorted_margins[lower_index], sorted_margins[upper_index]], dtype=np.float64
    )
    variances = np.asarray([one_variance, one_variance], dtype=np.float64)
    weights = np.asarray([0.5, 0.5], dtype=np.float64)
    for _iteration in range(_EM_ITERATIONS):
        log_zero = math.log(float(weights[0])) + _log_normal(
            margins, float(means[0]), float(variances[0])
        )
        log_one = math.log(float(weights[1])) + _log_normal(
            margins, float(means[1]), float(variances[1])
        )
        denominator = _logsumexp_pair(log_zero, log_one)
        responsibilities = np.stack(
            (np.exp(log_zero - denominator), np.exp(log_one - denominator)), axis=1
        )
        masses = np.sum(responsibilities, axis=0, dtype=np.float64)
        if bool((masses <= 0.0).any()):
            raise ValueError("twin mixture component mass is zero")
        weights = masses / float(count)
        means = np.sum(responsibilities * margins[:, None], axis=0) / masses
        differences = margins[:, None] - means[None, :]
        variances = np.sum(responsibilities * np.square(differences), axis=0) / masses
        variances = np.maximum(variances, _VARIANCE_FLOOR)
        order = np.lexsort((weights, variances, means))
        weights = weights[order]
        means = means[order]
        variances = variances[order]

    log_zero = math.log(float(weights[0])) + _log_normal(
        margins, float(means[0]), float(variances[0])
    )
    log_one = math.log(float(weights[1])) + _log_normal(
        margins, float(means[1]), float(variances[1])
    )
    two_log_likelihood = float(np.sum(_logsumexp_pair(log_zero, log_one)))
    bic_two = 5.0 * math.log(count) - 2.0 * two_log_likelihood
    assignments = log_one > log_zero
    high_count = int(np.count_nonzero(assignments))
    high_fraction = high_count / float(count)
    high_labels = labels[assignments]
    high_auc = (
        _auc(scores[assignments], high_labels)
        if {_LABEL_82, _LABEL_83} == set(high_labels.tolist())
        else 0.5
    )
    return (
        bic_one,
        bic_two,
        bic_one - bic_two,
        high_count,
        high_fraction,
        high_auc,
    )


def _evidence_from_scores(
    plane: str,
    scores: np.ndarray,
    labels: np.ndarray,
    lda_scores: np.ndarray,
) -> TwinReachabilityEvidence:
    full_auc = _auc(scores, labels)
    (
        bic_one,
        bic_two,
        bic_improvement,
        high_count,
        high_fraction,
        high_auc,
    ) = _mixture_statistics(scores, labels)
    values = (
        full_auc,
        bic_one,
        bic_two,
        bic_improvement,
        high_fraction,
        high_auc,
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("twin reachability result must be finite")
    centroid_cue_present = full_auc >= _AUC_GATE or (
        bic_improvement >= _BIC_GATE
        and high_fraction >= _HIGH_FRACTION_GATE
        and high_auc >= _HIGH_AUC_GATE
    )
    lda_full_auc = _auc(lda_scores, labels)
    if not math.isfinite(lda_full_auc):
        raise ValueError("twin LDA result must be finite")
    cue_present = centroid_cue_present or lda_full_auc >= _AUC_GATE
    return TwinReachabilityEvidence(
        plane=plane,
        source_count=int(labels.size),
        class_82_count=int(np.count_nonzero(labels == _LABEL_82)),
        class_83_count=int(np.count_nonzero(labels == _LABEL_83)),
        labels=tuple(int(value) for value in labels),
        signed_scores=tuple(float(value) for value in scores),
        full_auc=full_auc,
        bic_one=bic_one,
        bic_two=bic_two,
        bic_improvement=bic_improvement,
        high_evidence_count=high_count,
        high_evidence_fraction=high_fraction,
        high_evidence_auc=high_auc,
        centroid_cue_present=centroid_cue_present,
        lda_signed_scores=tuple(float(value) for value in lda_scores),
        lda_full_auc=lda_full_auc,
        cue_present=cue_present,
    )


def _loo_shrinkage_lda_scores(
    normalized: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    scores = np.empty(labels.size, dtype=np.float64)
    row_ordinals = np.arange(labels.size)
    dimension = normalized.shape[1]
    for query_index in range(labels.size):
        retained = row_ordinals != query_index
        training = normalized[retained]
        training_labels = labels[retained]
        means = {
            label: np.mean(training[training_labels == label], axis=0, dtype=np.float64)
            for label in (_LABEL_82, _LABEL_83)
        }
        centered = training.copy()
        for row_index, label in enumerate(training_labels):
            centered[row_index] -= means[int(label)]
        degrees_of_freedom = training.shape[0] - 2
        covariance_trace = float(
            np.sum(np.square(centered), dtype=np.float64) / degrees_of_freedom
        )
        shrinkage = max(0.1 * covariance_trace / dimension, _VARIANCE_FLOOR)
        scaled = centered / math.sqrt(degrees_of_freedom)
        delta = means[_LABEL_83] - means[_LABEL_82]
        dual = np.eye(training.shape[0], dtype=np.float64)
        dual += (scaled @ scaled.T) / shrinkage
        try:
            solution = np.linalg.solve(dual, scaled @ delta)
        except np.linalg.LinAlgError as error:
            raise ValueError("twin shrinkage LDA solve failed") from error
        direction = delta / shrinkage - (scaled.T @ solution) / (shrinkage * shrinkage)
        boundary = 0.5 * float((means[_LABEL_82] + means[_LABEL_83]) @ direction)
        scores[query_index] = float(normalized[query_index] @ direction - boundary)
    if not bool(np.isfinite(scores).all()):
        raise ValueError("twin shrinkage LDA scores must be finite")
    return scores


def build_twin_reachability(
    plane: str,
    descriptors: np.ndarray,
    labels: np.ndarray,
) -> TwinReachabilityEvidence:
    """Compute the registered leave-one-out Caliber reachability evidence."""

    if type(plane) is not str or not plane:
        raise ValueError("twin plane must be a nonempty string")
    if type(descriptors) is not np.ndarray or type(labels) is not np.ndarray:
        raise TypeError("twin descriptors and labels must be concrete arrays")
    if descriptors.ndim != 2 or labels.ndim != 1:
        raise ValueError("twin descriptors must be rank two and labels rank one")
    if descriptors.shape[0] != labels.shape[0]:
        raise ValueError("twin descriptor and label length differs")
    if descriptors.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise TypeError("twin descriptor dtype must be float32 or float64")
    if not np.issubdtype(labels.dtype, np.integer) or np.issubdtype(labels.dtype, np.bool_):
        raise TypeError("twin label dtype must be a concrete integer")
    if not bool(np.isfinite(descriptors).all()):
        raise ValueError("twin descriptors must be finite")
    if set(labels.tolist()) != {_LABEL_82, _LABEL_83}:
        raise ValueError("twin labels must be exactly 82 and 83")
    counts = {label: int(np.count_nonzero(labels == label)) for label in (_LABEL_82, _LABEL_83)}

    normalized = descriptors.astype(np.float64, copy=True)
    norms = np.linalg.vector_norm(normalized, axis=1)
    if bool((norms <= 0.0).any()) or not bool(np.isfinite(norms).all()):
        raise ValueError("twin descriptor norm must be finite and positive")
    if min(counts.values()) < _MIN_CLASS_COUNT:
        raise ValueError("twin class requires at least twenty rows")
    normalized /= norms[:, None]
    class_sums = {
        label: np.sum(normalized[labels == label], axis=0, dtype=np.float64)
        for label in (_LABEL_82, _LABEL_83)
    }
    scores = np.empty(labels.size, dtype=np.float64)
    for index in range(labels.size):
        own_label = int(labels[index])
        means: dict[int, np.ndarray] = {}
        for label in (_LABEL_82, _LABEL_83):
            total = (
                class_sums[label] - normalized[index]
                if label == own_label
                else class_sums[label]
            )
            denominator = counts[label] - 1 if label == own_label else counts[label]
            mean = total / float(denominator)
            norm = float(np.linalg.vector_norm(mean))
            if not math.isfinite(norm) or norm <= 0.0:
                raise ValueError("twin leave-one-out centroid norm must be positive")
            means[label] = mean / norm
        scores[index] = float(
            np.dot(normalized[index], means[_LABEL_83])
            - np.dot(normalized[index], means[_LABEL_82])
        )
    if not bool(np.isfinite(scores).all()):
        raise ValueError("twin signed scores must be finite")
    concrete_labels = labels.astype(np.int64, copy=False)
    lda_scores = _loo_shrinkage_lda_scores(normalized, concrete_labels)
    return _evidence_from_scores(plane, scores, concrete_labels, lda_scores)


def _seeded_generator(domain: bytes, seed: bytes) -> np.random.Generator:
    if type(seed) is not bytes or not seed:
        raise TypeError("twin inference seed must be nonempty bytes")
    framed = domain + len(seed).to_bytes(8, "little") + seed
    integer = int.from_bytes(hashlib.sha256(framed).digest()[:16], "little")
    return np.random.Generator(np.random.PCG64(integer))


def twin_reachability_inference_seeds(
    authority: TwinReachabilityAuthority,
) -> tuple[bytes, bytes]:
    """Derive both inference seeds from one canonical source authority."""

    _validate_authority(authority)
    canonical = (
        json.dumps(asdict(authority), sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    return b"bootstrap\0" + canonical, b"permutation\0" + canonical


def _bootstrap_auc_lower(
    evidence: TwinReachabilityEvidence,
    bootstrap_seed: bytes,
) -> float:
    labels = np.asarray(evidence.labels, dtype=np.int64)
    scores = np.asarray(evidence.lda_signed_scores, dtype=np.float64)
    bootstrap = _seeded_generator(b"prism-twin-bootstrap-v1\0", bootstrap_seed)
    aucs = np.empty(_BOOTSTRAP_DRAWS, dtype=np.float64)
    completed = 0
    while completed < _BOOTSTRAP_DRAWS:
        indexes = bootstrap.integers(0, labels.size, size=labels.size)
        sampled_labels = labels[indexes]
        if set(sampled_labels.tolist()) != {_LABEL_82, _LABEL_83}:
            continue
        aucs[completed] = _auc(scores[indexes], sampled_labels)
        completed += 1
    aucs.sort(kind="stable")
    return float(aucs[math.floor(0.05 * (_BOOTSTRAP_DRAWS - 1))])


def build_twin_reachability_inference(
    plane: str,
    descriptors: np.ndarray,
    labels: np.ndarray,
    *,
    bootstrap_seed: bytes,
    permutation_seed: bytes,
    expected_evidence: TwinReachabilityEvidence,
) -> TwinReachabilityInference:
    """Compute the fixed bootstrap bound and a fully refitted permutation null."""

    evidence = build_twin_reachability(plane, descriptors, labels)
    if evidence != expected_evidence:
        raise ValueError("twin inference evidence differs")
    concrete_labels = labels.astype(np.int64, copy=False)
    lower = _bootstrap_auc_lower(evidence, bootstrap_seed)

    permutation = _seeded_generator(
        b"prism-twin-permutation-v1\0", permutation_seed
    )
    extreme = 0
    for _draw in range(_PERMUTATION_DRAWS):
        shuffled = permutation.permutation(concrete_labels)
        permuted = build_twin_reachability(plane, descriptors, shuffled)
        extreme += int(permuted.lda_full_auc >= evidence.lda_full_auc)
    p_value = (extreme + 1) / float(_PERMUTATION_DRAWS + 1)
    return TwinReachabilityInference(
        bootstrap_draws=_BOOTSTRAP_DRAWS,
        permutation_draws=_PERMUTATION_DRAWS,
        bootstrap_seed_sha256=hashlib.sha256(bootstrap_seed).hexdigest(),
        permutation_seed_sha256=hashlib.sha256(permutation_seed).hexdigest(),
        lda_auc_lower_95=lower,
        permutation_extreme_count=extreme,
        permutation_p_value=p_value,
        passed=(
            evidence.lda_full_auc >= _AUC_GATE
            and lower > 0.50
            and p_value <= 0.05
        ),
    )


def _require_evidence_types(evidence: TwinReachabilityEvidence) -> None:
    if type(evidence) is not TwinReachabilityEvidence:
        raise TypeError("twin evidence has the wrong concrete type")
    if (
        type(evidence.plane) is not str
        or not evidence.plane
        or type(evidence.source_count) is not int
        or type(evidence.class_82_count) is not int
        or type(evidence.class_83_count) is not int
        or type(evidence.labels) is not tuple
        or any(type(value) is not int for value in evidence.labels)
        or type(evidence.signed_scores) is not tuple
        or any(type(value) is not float for value in evidence.signed_scores)
        or type(evidence.full_auc) is not float
        or type(evidence.bic_one) is not float
        or type(evidence.bic_two) is not float
        or type(evidence.bic_improvement) is not float
        or type(evidence.high_evidence_count) is not int
        or type(evidence.high_evidence_fraction) is not float
        or type(evidence.high_evidence_auc) is not float
        or type(evidence.centroid_cue_present) is not bool
        or type(evidence.lda_signed_scores) is not tuple
        or any(type(value) is not float for value in evidence.lda_signed_scores)
        or type(evidence.lda_full_auc) is not float
        or type(evidence.cue_present) is not bool
    ):
        raise TypeError("twin evidence concrete types differ")


def _payload(evidence: TwinReachabilityEvidence) -> dict[str, Any]:
    value = asdict(evidence)
    value["labels"] = list(evidence.labels)
    value["signed_scores"] = list(evidence.signed_scores)
    value["lda_signed_scores"] = list(evidence.lda_signed_scores)
    value["schema"] = _SCHEMA
    value["claim_eligible"] = False
    return value


def _validate_authority(authority: TwinReachabilityAuthority) -> None:
    if type(authority) is not TwinReachabilityAuthority:
        raise TypeError("twin authority has the wrong concrete type")
    if (
        authority.plane not in {"frozen-pooled", "trained-raw", "trained-projected"}
        or _COMMIT.fullmatch(authority.source_revision) is None
        or _SHA256.fullmatch(authority.source_tree_digest) is None
        or _COMMIT.fullmatch(authority.dataset_revision) is None
        or _SHA256.fullmatch(authority.dataset_manifest_sha256) is None
        or type(authority.model_name) is not str
        or not authority.model_name
        or _COMMIT.fullmatch(authority.model_revision) is None
        or authority.producer_kind not in {"frozen-model", "trained-checkpoint"}
        or (
            authority.producer_kind == "frozen-model"
            and _COMMIT.fullmatch(authority.producer_identity) is None
        )
        or (
            authority.producer_kind == "trained-checkpoint"
            and _SHA256.fullmatch(authority.producer_identity) is None
        )
        or _SHA256.fullmatch(authority.ordered_example_ids_sha256) is None
        or _SHA256.fullmatch(authority.label_vector_sha256) is None
        or _SHA256.fullmatch(authority.descriptor_sha256) is None
    ):
        raise ValueError("twin authority differs")


def canonical_twin_reachability_artifact_bytes(
    authority: TwinReachabilityAuthority,
    evidence: TwinReachabilityEvidence,
) -> bytes:
    """Bind one reachability result to exact caller-authenticated identities."""

    _validate_authority(authority)
    if authority.plane != evidence.plane:
        raise ValueError("twin artifact plane authority differs")
    canonical_twin_reachability_bytes(evidence)
    value = {
        "schema": "sfora-cars-twin-reachability-artifact-v1",
        "claim_eligible": False,
        "authority": asdict(authority),
        "evidence": _payload(evidence),
    }
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if validate_twin_reachability_artifact_bytes(raw, expected=authority) != evidence:
        raise ValueError("twin artifact failed canonical self-validation")
    return raw


def validate_twin_reachability_artifact_bytes(
    raw: bytes,
    *,
    expected: TwinReachabilityAuthority,
) -> TwinReachabilityEvidence:
    """Validate one canonical artifact against independently supplied authority."""

    _validate_authority(expected)
    if type(raw) is not bytes or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("twin artifact bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("twin artifact JSON differs") from error
    if type(value) is not dict or set(value) != {
        "schema",
        "claim_eligible",
        "authority",
        "evidence",
    }:
        raise ValueError("twin artifact schema differs")
    if (
        value["schema"] != "sfora-cars-twin-reachability-artifact-v1"
        or value["claim_eligible"] is not False
        or type(value["authority"]) is not dict
        or set(value["authority"]) != set(asdict(expected))
    ):
        raise ValueError("twin artifact schema or authority differs")
    try:
        observed = TwinReachabilityAuthority(**value["authority"])
    except TypeError as error:
        raise ValueError("twin artifact authority differs") from error
    _validate_authority(observed)
    if observed != expected:
        raise ValueError("twin artifact authority differs")
    evidence_raw = (
        json.dumps(value["evidence"], sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    evidence = validate_twin_reachability_bytes(evidence_raw, expected_plane=expected.plane)
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if raw != canonical:
        raise ValueError("twin artifact is not canonical")
    return evidence


def _validate_inference(
    inference: TwinReachabilityInference,
    evidence: TwinReachabilityEvidence,
    authority: TwinReachabilityAuthority,
) -> None:
    if type(inference) is not TwinReachabilityInference:
        raise TypeError("twin inference has the wrong concrete type")
    if (
        type(inference.bootstrap_draws) is not int
        or inference.bootstrap_draws != _BOOTSTRAP_DRAWS
        or type(inference.permutation_draws) is not int
        or inference.permutation_draws != _PERMUTATION_DRAWS
        or _SHA256.fullmatch(inference.bootstrap_seed_sha256) is None
        or _SHA256.fullmatch(inference.permutation_seed_sha256) is None
        or type(inference.lda_auc_lower_95) is not float
        or not math.isfinite(inference.lda_auc_lower_95)
        or not 0.0 <= inference.lda_auc_lower_95 <= 1.0
        or type(inference.permutation_extreme_count) is not int
        or not 0 <= inference.permutation_extreme_count <= _PERMUTATION_DRAWS
        or type(inference.permutation_p_value) is not float
        or not math.isfinite(inference.permutation_p_value)
        or type(inference.passed) is not bool
    ):
        raise ValueError("twin inference authority differs")
    expected_p = (inference.permutation_extreme_count + 1) / float(
        _PERMUTATION_DRAWS + 1
    )
    bootstrap_seed, permutation_seed = twin_reachability_inference_seeds(authority)
    if (
        inference.bootstrap_seed_sha256 != hashlib.sha256(bootstrap_seed).hexdigest()
        or inference.permutation_seed_sha256
        != hashlib.sha256(permutation_seed).hexdigest()
    ):
        raise ValueError("twin inference seed authority differs")
    if inference.lda_auc_lower_95 != _bootstrap_auc_lower(evidence, bootstrap_seed):
        raise ValueError("twin inference bootstrap derivation differs")
    expected_passed = (
        evidence.lda_full_auc >= _AUC_GATE
        and inference.lda_auc_lower_95 > 0.50
        and expected_p <= 0.05
    )
    if inference.permutation_p_value != expected_p or inference.passed is not expected_passed:
        raise ValueError("twin inference derivation differs")


def canonical_twin_reachability_inference_artifact_bytes(
    authority: TwinReachabilityAuthority,
    evidence: TwinReachabilityEvidence,
    inference: TwinReachabilityInference,
) -> bytes:
    """Bind uncertainty evidence to the exact descriptor and v1 evidence."""

    _validate_authority(authority)
    if authority.plane != evidence.plane:
        raise ValueError("twin inference plane authority differs")
    canonical_twin_reachability_bytes(evidence)
    _validate_inference(inference, evidence, authority)
    value = {
        "schema": "sfora-cars-twin-reachability-inference-artifact-v1",
        "claim_eligible": False,
        "authority": asdict(authority),
        "evidence": _payload(evidence),
        "inference": asdict(inference),
    }
    raw = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if validate_twin_reachability_inference_artifact_bytes(
        raw,
        expected=authority,
    ) != (evidence, inference):
        raise ValueError("twin inference artifact failed canonical self-validation")
    return raw


def validate_twin_reachability_inference_artifact_bytes(
    raw: bytes,
    *,
    expected: TwinReachabilityAuthority,
) -> tuple[TwinReachabilityEvidence, TwinReachabilityInference]:
    """Validate canonical uncertainty evidence against independent authority."""

    _validate_authority(expected)
    if type(raw) is not bytes or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("twin inference artifact bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("twin inference artifact JSON differs") from error
    if type(value) is not dict or set(value) != {
        "schema",
        "claim_eligible",
        "authority",
        "evidence",
        "inference",
    }:
        raise ValueError("twin inference artifact schema differs")
    if (
        value["schema"] != "sfora-cars-twin-reachability-inference-artifact-v1"
        or value["claim_eligible"] is not False
        or type(value["authority"]) is not dict
        or set(value["authority"]) != set(asdict(expected))
        or type(value["evidence"]) is not dict
        or type(value["inference"]) is not dict
        or set(value["inference"]) != _INFERENCE_KEYS
    ):
        raise ValueError("twin inference artifact schema differs")
    try:
        observed = TwinReachabilityAuthority(**value["authority"])
        inference = TwinReachabilityInference(**value["inference"])
    except TypeError as error:
        raise ValueError("twin inference artifact concrete types differ") from error
    _validate_authority(observed)
    if observed != expected:
        raise ValueError("twin inference artifact authority differs")
    evidence_raw = (
        json.dumps(value["evidence"], sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    evidence = validate_twin_reachability_bytes(
        evidence_raw,
        expected_plane=expected.plane,
    )
    _validate_inference(inference, evidence, expected)
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if canonical != raw:
        raise ValueError("twin inference artifact is not canonical")
    return evidence, inference


def canonical_twin_reachability_bytes(evidence: TwinReachabilityEvidence) -> bytes:
    """Serialize and independently self-validate one canonical result."""

    _require_evidence_types(evidence)
    raw = (json.dumps(_payload(evidence), sort_keys=True, separators=(",", ":")) + "\n").encode()
    if validate_twin_reachability_bytes(raw, expected_plane=evidence.plane) != evidence:
        raise ValueError("twin evidence failed canonical self-validation")
    return raw


def validate_twin_reachability_bytes(
    raw: bytes,
    *,
    expected_plane: str,
) -> TwinReachabilityEvidence:
    """Validate canonical bytes and recompute every derived statistic."""

    if type(expected_plane) is not str or expected_plane not in {
        "frozen-pooled",
        "trained-raw",
        "trained-projected",
    }:
        raise ValueError("twin expected plane differs")
    if type(raw) is not bytes or not raw.endswith(b"\n") or raw.endswith(b"\n\n"):
        raise ValueError("twin evidence bytes differ")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("twin evidence JSON differs") from error
    expected_keys = {
        "schema",
        "claim_eligible",
        "plane",
        "source_count",
        "class_82_count",
        "class_83_count",
        "labels",
        "signed_scores",
        "full_auc",
        "bic_one",
        "bic_two",
        "bic_improvement",
        "high_evidence_count",
        "high_evidence_fraction",
        "high_evidence_auc",
        "centroid_cue_present",
        "lda_signed_scores",
        "lda_full_auc",
        "cue_present",
    }
    if type(value) is not dict or set(value) != expected_keys:
        raise ValueError("twin evidence schema differs")
    value = cast(dict[str, Any], value)
    if value["schema"] != _SCHEMA or value["claim_eligible"] is not False:
        raise ValueError("twin evidence schema or claim differs")
    if value["plane"] != expected_plane:
        raise ValueError("twin evidence plane differs")
    if (
        type(value["labels"]) is not list
        or type(value["signed_scores"]) is not list
        or type(value["lda_signed_scores"]) is not list
    ):
        raise TypeError("twin evidence vectors differ")
    if any(type(item) is not int for item in value["labels"]) or any(
        type(item) is not float for item in value["signed_scores"]
    ) or any(
        type(item) is not float for item in value["lda_signed_scores"]
    ):
        raise TypeError("twin evidence vector concrete types differ")
    labels = np.asarray(value["labels"], dtype=np.int64)
    scores = np.asarray(value["signed_scores"], dtype=np.float64)
    lda_scores = np.asarray(value["lda_signed_scores"], dtype=np.float64)
    if (
        labels.ndim != 1
        or scores.ndim != 1
        or lda_scores.ndim != 1
        or labels.size != scores.size
        or labels.size != lda_scores.size
    ):
        raise ValueError("twin evidence vector shapes differ")
    if (
        set(labels.tolist()) != {_LABEL_82, _LABEL_83}
        or not bool(np.isfinite(scores).all())
        or not bool(np.isfinite(lda_scores).all())
    ):
        raise ValueError("twin evidence labels or scores differ")
    if min(
        int(np.count_nonzero(labels == _LABEL_82)),
        int(np.count_nonzero(labels == _LABEL_83)),
    ) < _MIN_CLASS_COUNT:
        raise ValueError("twin evidence requires at least twenty rows per class")
    recomputed = _evidence_from_scores(value["plane"], scores, labels, lda_scores)
    supplied = TwinReachabilityEvidence(
        plane=value["plane"],
        source_count=value["source_count"],
        class_82_count=value["class_82_count"],
        class_83_count=value["class_83_count"],
        labels=tuple(value["labels"]),
        signed_scores=tuple(value["signed_scores"]),
        full_auc=value["full_auc"],
        bic_one=value["bic_one"],
        bic_two=value["bic_two"],
        bic_improvement=value["bic_improvement"],
        high_evidence_count=value["high_evidence_count"],
        high_evidence_fraction=value["high_evidence_fraction"],
        high_evidence_auc=value["high_evidence_auc"],
        centroid_cue_present=value["centroid_cue_present"],
        lda_signed_scores=tuple(value["lda_signed_scores"]),
        lda_full_auc=value["lda_full_auc"],
        cue_present=value["cue_present"],
    )
    _require_evidence_types(supplied)
    if supplied != recomputed:
        raise ValueError("twin evidence derivation differs")
    canonical = (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
    if raw != canonical:
        raise ValueError("twin evidence is not canonical")
    return supplied
