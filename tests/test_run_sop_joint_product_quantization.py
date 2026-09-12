from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from sfora.additive_quantization import AdditiveQuantizationSpec, AdditiveQuantizer

_SCRIPT = Path(__file__).parents[1] / "scripts" / "run_sop_joint_product_quantization.py"
sys.path.insert(0, str(_SCRIPT.parent))


def _subject() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_sop_joint_product_quantization", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_exact_additive_rankings_exclude_self_and_break_ties_by_row() -> None:
    subject = _subject()
    quantizer = AdditiveQuantizer.from_codebooks(
        AdditiveQuantizationSpec(dimensions=2, stages=1, codebook_size=2),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
    )
    queries = torch.tensor([[1.0, 0.0]] * 4, dtype=torch.float32)
    codes = torch.tensor([[0], [0], [1], [1]], dtype=torch.uint8)

    observed = subject.exact_additive_rankings(
        queries, codes, quantizer, width=2, query_block_size=3
    )

    assert observed.tolist() == [[1, 2], [0, 2], [0, 1], [0, 1]]


def test_exact_additive_rankings_accept_inference_mode_embeddings() -> None:
    quantizer = AdditiveQuantizer.from_codebooks(
        AdditiveQuantizationSpec(dimensions=2, stages=1, codebook_size=2),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
    )
    with torch.inference_mode():
        values = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float32)
        codes = torch.tensor([[0], [1], [0]], dtype=torch.uint8)

    observed = _subject().exact_additive_rankings(
        values, codes, quantizer, width=1, query_block_size=2
    )

    assert observed.shape == (3, 1)


def test_metric_rankings_materialize_on_cpu() -> None:
    subject = _subject()
    source = torch.tensor([[2, 1]], dtype=torch.int64)
    if torch.cuda.is_available():
        source = source.cuda()

    observed = subject.metric_rankings(source)

    assert observed.device.type == "cpu"
    assert observed.is_contiguous()
    assert observed.tolist() == [[2, 1]]


def test_joint_codec_decision_uses_pq32_screen_target_and_r1_guardrail() -> None:
    subject = _subject()

    assert (
        subject.joint_codec_decision(
            best_map_at_r=0.5780,
            best_r1=0.840,
            pq32_map_at_r=0.578923,
            matched_pq24_dot_r1=0.8222,
            padded_pq24_dot_map_at_r=0.560,
        )["classification"]
        == "additive-codec-failed-pq32"
    )
    assert (
        subject.joint_codec_decision(
            best_map_at_r=0.5800,
            best_r1=0.840,
            pq32_map_at_r=0.578923,
            matched_pq24_dot_r1=0.8222,
            padded_pq24_dot_map_at_r=0.560,
        )["classification"]
        == "additive-codec-insufficient-recovery"
    )
    assert (
        subject.joint_codec_decision(
            best_map_at_r=0.5820,
            best_r1=0.819,
            pq32_map_at_r=0.578923,
            matched_pq24_dot_r1=0.8222,
            padded_pq24_dot_map_at_r=0.560,
        )["classification"]
        == "additive-codec-r1-regressed"
    )
    assert (
        subject.joint_codec_decision(
            best_map_at_r=0.5820,
            best_r1=0.821,
            pq32_map_at_r=0.578923,
            matched_pq24_dot_r1=0.8222,
            padded_pq24_dot_map_at_r=0.560,
        )["classification"]
        == "additive-codec-seed0-continue"
    )
    assert (
        subject.joint_codec_decision(
            best_map_at_r=0.5860,
            best_r1=0.819,
            pq32_map_at_r=0.578923,
            matched_pq24_dot_r1=0.8222,
            padded_pq24_dot_map_at_r=0.560,
        )["classification"]
        == "additive-codec-r1-regressed"
    )
    passed = subject.joint_codec_decision(
        best_map_at_r=0.5860,
        best_r1=0.821,
        pq32_map_at_r=0.578923,
        matched_pq24_dot_r1=0.8222,
        padded_pq24_dot_map_at_r=0.560,
    )
    assert passed["classification"] == "additive-codec-target-screen"
    assert passed["gain_over_padded_pq24_dot"] == pytest.approx(0.026)

    with pytest.raises(ValueError, match="joint codec decision authority differs"):
        subject.joint_codec_decision(
            best_map_at_r=True,
            best_r1=0.821,
            pq32_map_at_r=0.578923,
            matched_pq24_dot_r1=0.8222,
            padded_pq24_dot_map_at_r=0.560,
        )


def test_joint_codec_selection_prefers_an_arm_that_passes_both_gates() -> None:
    subject = _subject()

    selected, decisions = subject.select_joint_codec_arm(
        {
            "higher-map-low-r1": {"map_at_r": 0.5860, "r1": 0.8190},
            "qualifying": {"map_at_r": 0.5858, "r1": 0.8220},
        },
        pq32_map_at_r=0.578923,
        matched_pq24_dot_r1=0.8222,
        padded_pq24_dot_map_at_r=0.560,
    )

    assert selected == "qualifying"
    assert decisions["higher-map-low-r1"]["classification"] == "additive-codec-r1-regressed"
    assert decisions["qualifying"]["passed"] is True

    selected, decisions = subject.select_joint_codec_arm(
        {
            "higher-map-low-r1": {"map_at_r": 0.5840, "r1": 0.8190},
            "continuation": {"map_at_r": 0.5820, "r1": 0.8220},
        },
        pq32_map_at_r=0.578923,
        matched_pq24_dot_r1=0.8222,
        padded_pq24_dot_map_at_r=0.560,
    )
    assert selected == "continuation"
    assert decisions["continuation"]["continuation_eligible"] is True
    with pytest.raises(ValueError, match="joint codec decision authority differs"):
        subject.joint_codec_decision(
            best_map_at_r=1.01,
            best_r1=0.821,
            pq32_map_at_r=0.578923,
            matched_pq24_dot_r1=0.8222,
            padded_pq24_dot_map_at_r=0.560,
        )


def test_float_rerank_matches_stable_scalar_candidate_ordering() -> None:
    subject = _subject()
    values = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=torch.float32)
    candidates = torch.tensor([[3, 2, 1], [3, 2, 0], [3, 1, 0], [2, 1, 0]])

    observed = subject._float_rerank_within_candidates(values, candidates)

    assert observed.tolist() == [[1, 2, 3], [0, 2, 3], [0, 1, 3], [2, 0, 1]]


def test_checkpoint_publication_recovers_only_an_identical_existing_file(tmp_path: Path) -> None:
    subject = _subject()
    path = tmp_path / "model.pt"

    subject.publish_checkpoint_or_validate_existing(path, b"registered-checkpoint")
    subject.publish_checkpoint_or_validate_existing(path, b"registered-checkpoint")

    assert path.read_bytes() == b"registered-checkpoint"
    with pytest.raises(ValueError, match="joint codec checkpoint differs"):
        subject.publish_checkpoint_or_validate_existing(path, b"different-checkpoint")


def test_failed_lookup_receipt_is_bound_to_the_same_parent_artifacts() -> None:
    subject = _subject()
    receipt = {
        "claim_eligible": False,
        "decision": {"classification": "fixed-code-lookup-r1-regressed"},
        "inputs": {
            "direct_checkpoint_sha256": "1" * 64,
            "parent_codec_checkpoint_sha256": "2" * 64,
            "parent_codec_receipt_sha256": "3" * 64,
            "prior_reranker_receipt_sha256": "4" * 64,
            "source_snapshot_sha256": "5" * 64,
            "teacher_snapshot_sha256": "6" * 64,
        },
        "official_test_touched": False,
        "schema": "sfora-pq-lookup-distillation-v1",
    }

    subject.validate_failed_lookup_receipt(
        receipt,
        direct_checkpoint_sha256="1" * 64,
        parent_codec_checkpoint_sha256="2" * 64,
        parent_codec_receipt_sha256="3" * 64,
        source_snapshot_sha256="5" * 64,
        teacher_snapshot_sha256="6" * 64,
    )
    receipt["inputs"]["teacher_snapshot_sha256"] = "7" * 64
    with pytest.raises(ValueError, match="failed lookup receipt authority differs"):
        subject.validate_failed_lookup_receipt(
            receipt,
            direct_checkpoint_sha256="1" * 64,
            parent_codec_checkpoint_sha256="2" * 64,
            parent_codec_receipt_sha256="3" * 64,
            source_snapshot_sha256="5" * 64,
            teacher_snapshot_sha256="6" * 64,
        )
    receipt["inputs"]["teacher_snapshot_sha256"] = "6" * 64
    receipt["inputs"]["parent_codec_receipt_sha256"] = "8" * 64
    with pytest.raises(ValueError, match="failed lookup receipt authority differs"):
        subject.validate_failed_lookup_receipt(
            receipt,
            direct_checkpoint_sha256="1" * 64,
            parent_codec_checkpoint_sha256="2" * 64,
            parent_codec_receipt_sha256="3" * 64,
            source_snapshot_sha256="5" * 64,
            teacher_snapshot_sha256="6" * 64,
        )


def test_ordering_diagnostics_exclude_self_from_positive_error_tail() -> None:
    subject = _subject()
    rows = 130
    angles = torch.linspace(0.0, 1.0, rows)
    values = torch.stack((torch.cos(angles), torch.sin(angles)), dim=1).float()
    quantizer = AdditiveQuantizer.from_codebooks(
        AdditiveQuantizationSpec(dimensions=2, stages=1, codebook_size=2),
        torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
    )
    codes = (torch.arange(rows) % 2).to(torch.uint8).unsqueeze(1)
    rankings = subject.exact_additive_rankings(
        values, codes, quantizer, width=128, query_block_size=64
    )
    teacher_scores = values @ values.T
    teacher_scores.fill_diagonal_(-torch.inf)
    teacher_rankings = torch.argsort(teacher_scores, dim=1, descending=True, stable=True)[:, :128]

    observed = subject._ordering_diagnostics(values, codes, quantizer, rankings, teacher_rankings)

    assert math.isfinite(observed["score_positive_error_maximum"])
    assert math.isfinite(observed["score_positive_error_per_query_maximum_p99"])


def test_direct_execution_refuses_to_run_without_explicit_science_flag() -> None:
    completed = subprocess.run(
        [sys.executable, str(_SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert "--execute-joint-product-quantization" in completed.stderr
