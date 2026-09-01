"""Tests for the frozen SigLIP manufacturer-band audit."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest
import torch
from torch.nn import functional as F

from sfora.siglip_band_audit import (
    SIGLIP_AUDIT_BANDS,
    SIGLIP_AUDIT_TWIN_GROUPS,
    SiglipAuditBand,
    SiglipBandAuditAuthority,
    SiglipBandAuditEvidence,
    SiglipBandConfusion,
    SiglipBandEvidence,
    canonical_siglip_band_audit_bytes,
    score_siglip_frozen_bands,
    siglip_band_nearest_rows,
    twin_representative,
    validate_siglip_band_audit_bytes,
    validate_siglip_band_definition,
    validate_siglip_band_inputs,
)


def _valid_inputs() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    labels = torch.arange(98, dtype=torch.int64).repeat_interleave(2)
    generator = torch.Generator().manual_seed(901)
    descriptors = F.normalize(
        torch.randn(labels.numel(), 12, generator=generator, dtype=torch.float32),
        dim=1,
    )
    class_names = tuple(f"class-{index:03d}" for index in range(196))
    return descriptors, labels, class_names


def test_siglip_band_and_twin_authority_is_exact_and_disjoint() -> None:
    assert (
        SiglipAuditBand("optimization", 0, 48),
        SiglipAuditBand("clean", 49, 81),
        SiglipAuditBand("burned", 82, 97),
    ) == SIGLIP_AUDIT_BANDS
    assert SIGLIP_AUDIT_TWIN_GROUPS == (
        (7, 8),
        (9, 10),
        (16, 17),
        (20, 21),
        (22, 23),
        (26, 27),
        (28, 29),
        (41, 42),
        (44, 45),
        (53, 68, 69, 73, 74),
        (54, 55, 56),
        (63, 70),
        (66, 72),
        (82, 83),
        (85, 86),
        (89, 90),
        (93, 94),
        (95, 96),
    )
    assert twin_representative(68) == 53
    assert twin_representative(96) == 95
    assert twin_representative(0) == 0

    validate_siglip_band_definition(SIGLIP_AUDIT_BANDS, SIGLIP_AUDIT_TWIN_GROUPS)

    with pytest.raises(ValueError, match="overlap"):
        validate_siglip_band_definition(
            SIGLIP_AUDIT_BANDS,
            SIGLIP_AUDIT_TWIN_GROUPS + ((8, 11),),
        )
    with pytest.raises(ValueError, match="one band"):
        validate_siglip_band_definition(SIGLIP_AUDIT_BANDS, ((48, 49),))
    with pytest.raises(ValueError, match="partition"):
        validate_siglip_band_definition(
            (replace(SIGLIP_AUDIT_BANDS[0], last_label=47),) + SIGLIP_AUDIT_BANDS[1:],
            SIGLIP_AUDIT_TWIN_GROUPS,
        )


def test_siglip_band_inputs_require_complete_finite_unit_train_authority() -> None:
    descriptors, labels, class_names = _valid_inputs()
    validate_siglip_band_inputs(descriptors, labels, class_names)

    cases: list[tuple[str, torch.Tensor, torch.Tensor, tuple[str, ...]]] = []
    cases.append(("row", descriptors[:-1], labels, class_names))
    cases.append(("integer", descriptors, labels.float(), class_names))
    cases.append(("exactly", descriptors[:-2], labels[:-2], class_names))
    cases.append(("at least two", descriptors[1:], labels[1:], class_names))
    nonfinite = descriptors.clone()
    nonfinite[0, 0] = torch.nan
    cases.append(("finite", nonfinite, labels, class_names))
    nonunit = descriptors.clone()
    nonunit[0] *= 0.5
    cases.append(("unit", nonunit, labels, class_names))
    cases.append(("196", descriptors, labels, class_names[:-1]))

    for message, bad_descriptors, bad_labels, bad_names in cases:
        with pytest.raises(ValueError, match=message):
            validate_siglip_band_inputs(bad_descriptors, bad_labels, bad_names)

    with pytest.raises(TypeError, match="strings"):
        validate_siglip_band_inputs(
            descriptors,
            labels,
            class_names[:-1] + (3,),  # type: ignore[arg-type]
        )

    for bad_label in (-1, 98, True):
        with pytest.raises((TypeError, ValueError), match="label"):
            twin_representative(bad_label)  # type: ignore[arg-type]


def _hand_derived_scoring_inputs() -> tuple[torch.Tensor, torch.Tensor, tuple[str, ...]]:
    labels = torch.arange(98, dtype=torch.int64).repeat_interleave(2)
    descriptors = torch.zeros(196, 98, dtype=torch.float32)
    descriptors[torch.arange(196), labels] = 1.0
    descriptors[0] = F.normalize(0.6 * descriptors[0] + 0.8 * descriptors[4], dim=0)
    descriptors[2] = F.normalize(descriptors[2] + descriptors[6], dim=0)
    descriptors[14] = F.normalize(0.6 * descriptors[14] + 0.8 * descriptors[16], dim=0)
    return descriptors, labels, tuple(f"class-{index:03d}" for index in range(196))


def _scalar_nearest_rows(
    descriptors: torch.Tensor,
    labels: torch.Tensor,
    band: SiglipAuditBand,
) -> tuple[int, ...]:
    rows = [
        row
        for row, label in enumerate(labels.tolist())
        if band.first_label <= label <= band.last_label
    ]
    nearest: list[int] = []
    for query in rows:
        best_row = -1
        best_score = -torch.inf
        for candidate in rows:
            if candidate == query:
                continue
            score = torch.dot(descriptors[query], descriptors[candidate])
            if bool(score > best_score):
                best_score = score
                best_row = candidate
        nearest.append(best_row)
    return tuple(nearest)


def test_siglip_band_scorer_records_strict_twin_and_confusion_evidence() -> None:
    descriptors, labels, class_names = _hand_derived_scoring_inputs()

    evidence = score_siglip_frozen_bands(
        descriptors,
        labels,
        class_names,
        query_block=7,
    )

    assert evidence == SiglipBandAuditEvidence(
        bands=(
            SiglipBandEvidence(
                role="optimization",
                first_label=0,
                last_label=48,
                query_count=98,
                strict_hits=96,
                strict_recall_ppm=979_591,
                twin_hits=97,
                twin_recall_ppm=989_795,
                twin_rescued_errors=1,
                confusions=(
                    SiglipBandConfusion(query_label=0, nearest_label=2, count=1),
                    SiglipBandConfusion(query_label=7, nearest_label=8, count=1),
                ),
            ),
            SiglipBandEvidence(
                role="clean",
                first_label=49,
                last_label=81,
                query_count=66,
                strict_hits=66,
                strict_recall_ppm=1_000_000,
                twin_hits=66,
                twin_recall_ppm=1_000_000,
                twin_rescued_errors=0,
                confusions=(),
            ),
            SiglipBandEvidence(
                role="burned",
                first_label=82,
                last_label=97,
                query_count=32,
                strict_hits=32,
                strict_recall_ppm=1_000_000,
                twin_hits=32,
                twin_recall_ppm=1_000_000,
                twin_rescued_errors=0,
                confusions=(),
            ),
        ),
        query_count=196,
        strict_hits=194,
        strict_recall_ppm=989_795,
        twin_hits=195,
        twin_recall_ppm=994_897,
        twin_rescued_errors=1,
    )


def test_siglip_blocked_neighbours_match_scalar_lowest_row_ties() -> None:
    descriptors, labels, _ = _hand_derived_scoring_inputs()
    for band in SIGLIP_AUDIT_BANDS:
        expected = _scalar_nearest_rows(descriptors, labels, band)
        band_size = band.last_label - band.first_label + 1
        for query_block in (1, 2, band_size * 2):
            assert (
                siglip_band_nearest_rows(
                    descriptors,
                    labels,
                    band,
                    query_block=query_block,
                )
                == expected
            )

    with pytest.raises(ValueError, match="query block"):
        score_siglip_frozen_bands(descriptors, labels, _valid_inputs()[2], query_block=0)


def _authority() -> SiglipBandAuditAuthority:
    return SiglipBandAuditAuthority(
        source_commit="1" * 40,
        source_tree_digest="2" * 64,
        dataset_revision="9abf6cf7d6dfa7b95152a0d6e791ea9435b47a40",
        dataset_examples_sha256="3" * 64,
        ordered_example_ids_sha256="4" * 64,
        descriptor_sha256="5" * 64,
        label_vector_sha256="6" * 64,
        class_names_sha256="9da9ec6333105a7a2f0d50d7a5a6afe18b1ec3ede7dd8f1df298e59eb859ce35",
        model_name="google/siglip-so400m-patch14-384",
        model_revision="9fdffc58afc957d1a03a25b10dba0329ab15c2a3",
        readout="vision_pooler_output",
        split="train",
        batch_size=8,
        query_block=32,
    )


def _evidence() -> SiglipBandAuditEvidence:
    descriptors, labels, class_names = _hand_derived_scoring_inputs()
    return score_siglip_frozen_bands(
        descriptors,
        labels,
        class_names,
        query_block=32,
    )


def _resign(value: dict[str, object]) -> bytes:
    unsigned = dict(value)
    unsigned.pop("result_sha256", None)
    unsigned_raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    value["result_sha256"] = hashlib.sha256(unsigned_raw).hexdigest()
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def test_siglip_band_result_is_canonical_claim_ineligible_and_recomputable() -> None:
    authority = _authority()
    raw = canonical_siglip_band_audit_bytes(_evidence(), authority=authority)

    parsed = validate_siglip_band_audit_bytes(raw, expected_authority=authority)

    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert json.dumps(parsed, sort_keys=True, separators=(",", ":")).encode() + b"\n" == raw
    assert parsed["schema"] == "sfora-siglip-band-audit-v1"
    assert parsed["claim_eligible"] is False
    assert parsed["official_test_access"] is False
    assert "passed" not in parsed
    assert parsed["bands"][0]["strict_hits"] == 96
    assert parsed["twin_rescued_errors"] == 1


def test_siglip_band_result_rejects_schema_authority_and_metric_mutations() -> None:
    authority = _authority()
    baseline = json.loads(canonical_siglip_band_audit_bytes(_evidence(), authority=authority))

    mutations: list[tuple[str, dict[str, object]]] = []
    extra = json.loads(json.dumps(baseline))
    extra["extra"] = 1
    mutations.append(("schema", extra))
    boolean = json.loads(json.dumps(baseline))
    boolean["claim_eligible"] = 0
    mutations.append(("claim", boolean))
    official = json.loads(json.dumps(baseline))
    official["official_test_access"] = True
    mutations.append(("official", official))
    authority_drift = json.loads(json.dumps(baseline))
    authority_drift["authority"]["descriptor_sha256"] = "7" * 64
    mutations.append(("authority", authority_drift))
    hits = json.loads(json.dumps(baseline))
    hits["bands"][0]["strict_hits"] -= 1
    mutations.append(("band", hits))
    confusion = json.loads(json.dumps(baseline))
    confusion["bands"][0]["confusions"][0]["count"] += 1
    mutations.append(("confusion", confusion))
    aggregate = json.loads(json.dumps(baseline))
    aggregate["strict_recall_ppm"] -= 1
    mutations.append(("aggregate", aggregate))
    concrete = json.loads(json.dumps(baseline))
    concrete["query_count"] = 196.0
    mutations.append(("integer", concrete))
    role = json.loads(json.dumps(baseline))
    role["bands"][0]["role"] = "burned"
    mutations.append(("band", role))

    for message, value in mutations:
        with pytest.raises(ValueError, match=message):
            validate_siglip_band_audit_bytes(_resign(value), expected_authority=authority)

    with pytest.raises(ValueError, match="digest"):
        validate_siglip_band_audit_bytes(
            canonical_siglip_band_audit_bytes(_evidence(), authority=authority).replace(
                b"979591",
                b"979590",
                1,
            ),
            expected_authority=authority,
        )
