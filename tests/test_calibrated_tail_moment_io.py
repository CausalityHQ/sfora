from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from sfora.calibrated_tail_moment import (
    EQUAL_TOTAL_STORAGE_CONTROLS,
    MATCHED_ROW_WIDTH_CONTROLS,
    ctm_decision,
)
from sfora.calibrated_tail_moment_io import (
    evaluate_bundle,
    publish_ctm_report,
    validate_ctm_report,
)
from sfora.unicom_audit_io import EmbeddingBundle


def _fit_closed_report() -> dict[str, object]:
    return {
        "schema_version": 1,
        "phase": "FIT_CLOSED",
        "input": {
            "bundle_path": "/tmp/input.npz",
            "bundle_sha256": "a" * 64,
            "metadata": {"embedding_dimension": 768},
            "array_sha256": {
                "train_embeddings": "b" * 64,
                "train_labels": "c" * 64,
                "query_embeddings": "d" * 64,
                "query_labels": "e" * 64,
                "gallery_embeddings": "f" * 64,
                "gallery_labels": "1" * 64,
            },
            "embedding_dimension": 768,
            "query_count": 4,
            "gallery_count": 4,
        },
        "constants": {
            "widths": [64, 128, 256, 512],
            "neighbors": 50,
            "bootstrap_samples": 10_000,
            "bootstrap_seed": 205,
            "null_seeds": list(range(206, 238)),
        },
        "fits": {
            "primary_width": 128,
            "native": {
                str(width): {
                    "dimension": 768,
                    "basis_kind": "native",
                    "width": width,
                    "neighbors": 50,
                    "lambda_raw": -0.1,
                    "lambda_value": 0.0,
                }
                for width in (64, 128, 256, 512)
            },
            "pca": {
                str(width): {
                    "dimension": 768,
                    "basis_kind": "pca",
                    "width": width,
                    "neighbors": 50,
                    "lambda_raw": -0.1,
                    "lambda_value": 0.0,
                }
                for width in (64, 128, 256, 512)
            },
            "primary_scalar_nonzero": False,
            "null": {
                "seeds": list(range(206, 238)),
                "lambda_raw_values": [0.0] * 32,
                "p_value": 1.0,
            },
        },
        "views": None,
        "primary": None,
        "timing": {
            "normalization_seconds": 0.1,
            "basis_fit_seconds": 0.2,
            "coefficient_fit_seconds": 0.3,
            "falsifier_seconds": 0.4,
            "evaluation_seconds": 0.0,
        },
        "storage": None,
        "runtime": {"python": "3.12.3", "numpy": "2.5.0"},
        "decision": {"status": "CLOSE", "reason": "primary coefficient clipped to zero"},
    }


def test_fit_closed_report_roundtrips_with_exact_schema() -> None:
    report = _fit_closed_report()
    validate_ctm_report(report)
    persisted = json.loads(json.dumps(report, allow_nan=False, separators=(",", ":")))

    validate_ctm_report(persisted)

    assert tuple(persisted) == (
        "schema_version",
        "phase",
        "input",
        "constants",
        "fits",
        "views",
        "primary",
        "timing",
        "storage",
        "runtime",
        "decision",
    )


def test_fit_closed_report_rejects_evaluation_payload_or_false_decision() -> None:
    report = _fit_closed_report()
    report["views"] = {}
    with pytest.raises(ValueError, match="FIT_CLOSED"):
        validate_ctm_report(report)
    report = _fit_closed_report()
    report["decision"]["status"] = "REPLICATE"
    with pytest.raises(ValueError, match="decision"):
        validate_ctm_report(report)


def test_publication_strictly_reloads_and_never_clobbers(tmp_path: Path) -> None:
    output = tmp_path / "ctm.json"
    report = _fit_closed_report()

    publish_ctm_report(output, report)

    first = output.read_bytes()
    assert output.stat().st_mode & 0o777 == 0o600
    assert json.loads(first)["phase"] == "FIT_CLOSED"
    with pytest.raises(FileExistsError):
        publish_ctm_report(output, report)
    assert output.read_bytes() == first
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_publication_fsyncs_directory_after_publish_and_temp_cleanup(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "ctm.json"
    directory_fsyncs = 0
    original = os.fsync

    def observed(fd: int) -> None:
        nonlocal directory_fsyncs
        if os.fstat(fd).st_mode & 0o170000 == 0o040000:
            directory_fsyncs += 1
        original(fd)

    monkeypatch.setattr(os, "fsync", observed)
    publish_ctm_report(output, _fit_closed_report())

    assert directory_fsyncs == 2


def test_publication_hard_link_race_preserves_foreign_destination(
    tmp_path: Path, monkeypatch
) -> None:
    output = tmp_path / "ctm.json"
    original = os.link

    def race(source, destination) -> None:
        output.write_bytes(b"foreign")
        original(source, destination)

    monkeypatch.setattr(os, "link", race)

    with pytest.raises(FileExistsError):
        publish_ctm_report(output, _fit_closed_report())
    assert output.read_bytes() == b"foreign"
    assert not list(tmp_path.glob(f".{output.name}.*.tmp"))


def test_evaluate_bundle_builds_a_strict_reduced_report(tmp_path: Path) -> None:
    labels = np.asarray(["a", "b", "c", "d"])
    bundle = EmbeddingBundle(
        path=(tmp_path / "synthetic.npz").resolve(),
        sha256="a" * 64,
        metadata={"embedding_dimension": 4},
        train_embeddings=np.ascontiguousarray(np.tile(np.eye(4, dtype=np.float32), (13, 1))),
        train_labels=np.tile(labels, 13),
        query_embeddings=np.eye(4, dtype=np.float32),
        query_labels=labels,
        gallery_embeddings=np.eye(4, dtype=np.float32),
        gallery_labels=labels,
    )

    report = evaluate_bundle(bundle, widths=(2,), neighbors=1, null_seeds=range(206, 210))

    validate_ctm_report(report, expected_widths=(2,), expected_neighbors=1, expected_null_count=4)
    assert report["phase"] in {"FIT_CLOSED", "EVALUATED"}
    assert report["input"]["bundle_sha256"] == "a" * 64
    assert report["constants"]["widths"] == [2]
    assert tuple(report["timing"]) == (
        "normalization_seconds",
        "basis_fit_seconds",
        "coefficient_fit_seconds",
        "falsifier_seconds",
        "evaluation_seconds",
    )


def _evaluated_bundle(tmp_path: Path) -> EmbeddingBundle:
    values = np.asarray(
        [
            [0.8, 0.0, 0.6, 0.0],
            [0.8, 0.0, 0.0, 0.6],
            [0.0, 0.8, 0.6, 0.0],
            [0.0, 0.8, 0.0, 0.6],
        ],
        dtype=np.float32,
    )
    labels = np.asarray(["a", "b", "c", "d"])
    return EmbeddingBundle(
        path=(tmp_path / "evaluated.npz").resolve(),
        sha256="b" * 64,
        metadata={"embedding_dimension": 4},
        train_embeddings=np.ascontiguousarray(np.repeat(values, 13, axis=0)),
        train_labels=np.repeat(labels, 13),
        query_embeddings=values,
        query_labels=labels,
        gallery_embeddings=values,
        gallery_labels=labels,
    )


def _evaluated_report(tmp_path: Path) -> dict[str, object]:
    report = evaluate_bundle(
        _evaluated_bundle(tmp_path),
        widths=(2,),
        neighbors=1,
        null_seeds=range(206, 210),
    )
    assert report["phase"] == "EVALUATED"
    return report


def _refresh_decision(report: dict[str, object]) -> None:
    width = str(report["fits"]["primary_width"])
    views = report["views"][width]
    paired = report["primary"]["paired"]
    interval = report["primary"]["lambda_interval"]
    decision = ctm_decision(
        ctm_r1=views["native_ctm"]["recall"]["1"],
        ctm_map_at_r=views["native_ctm"]["map_at_r"],
        renormalized_r1=views["renormalized_prefix_plus_zero"]["recall"]["1"],
        renormalized_map_at_r=views["renormalized_prefix_plus_zero"]["map_at_r"],
        official_512_r1=views["official_512"]["recall"]["1"],
        paired_lower=paired["lower"],
        control_r1={key: views[key]["recall"]["1"] for key in MATCHED_ROW_WIDTH_CONTROLS},
        ctm_total_bytes=views["native_ctm"]["total_bytes"],
        control_total_bytes={
            key: views[key]["total_bytes"] for key in EQUAL_TOTAL_STORAGE_CONTROLS
        },
        lambda_raw=report["fits"]["native"][width]["lambda_raw"],
        lambda_lower=interval["lower"],
        null_p_value=report["primary"]["null_p_value"],
        width_gains=tuple(tuple(item) for item in report["primary"]["width_gains"]),
        replication_status=report["primary"]["replication_status"],
    )
    report["decision"] = {"status": decision.status, "reason": decision.reason}


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["views"]["2"].__setitem__(
            "native_ctm", report["views"]["2"].pop("pca_ctm")
        ),
        lambda report: report["views"]["2"]["native_ctm"]["recall"].__setitem__("1", 0.125),
        lambda report: report["views"]["2"]["native_ctm"].__setitem__("total_bytes", 999),
        lambda report: report["primary"]["paired"].__setitem__("point", 0.125),
        lambda report: report["primary"]["width_gains"][0].__setitem__(1, 0.125),
        lambda report: report["decision"].__setitem__("status", "REPLICATE"),
    ],
)
def test_evaluated_report_rejects_relational_mutations(tmp_path: Path, mutate) -> None:
    report = _evaluated_report(tmp_path)
    mutate(report)

    with pytest.raises((TypeError, ValueError, KeyError)):
        validate_ctm_report(
            report, expected_widths=(2,), expected_neighbors=1, expected_null_count=4
        )


def test_paired_point_recomputes_from_persisted_correctness_order(tmp_path: Path) -> None:
    report = _evaluated_report(tmp_path)
    report["input"]["query_count"] = 7
    baseline = [True, True, True, True, True, False, False]
    candidate = [True, True, False, False, False, False, False]
    for view in report["views"]["2"].values():
        view["top1_indices"] = [0] * 7
        view["top1_correct"] = list(baseline)
        view["recall"]["1"] = float(np.mean(baseline))
    report["views"]["2"]["native_ctm"]["top1_correct"] = candidate
    report["views"]["2"]["native_ctm"]["recall"]["1"] = float(np.mean(candidate))
    difference = np.asarray(candidate, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    point = float(np.mean(difference))
    assert point != float(np.mean(candidate)) - float(np.mean(baseline))
    report["primary"]["paired"].update({"point": point, "lower": point - 0.1, "upper": point + 0.1})
    report["primary"]["width_gains"] = [[2, float(np.mean(candidate)) - float(np.mean(baseline))]]
    _refresh_decision(report)

    validate_ctm_report(report, expected_widths=(2,), expected_neighbors=1, expected_null_count=4)


def test_lambda_interval_need_not_contain_unweighted_point(tmp_path: Path) -> None:
    report = _evaluated_report(tmp_path)
    raw = report["primary"]["lambda_interval"]["point"]
    report["primary"]["lambda_interval"].update({"lower": raw + 0.01, "upper": raw + 0.02})
    _refresh_decision(report)

    validate_ctm_report(report, expected_widths=(2,), expected_neighbors=1, expected_null_count=4)
