from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).parents[1] / "scripts" / "diagnose_pass201_cis_operator.py"
SPEC = importlib.util.spec_from_file_location("diagnose_pass201_cis_operator", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

LITERAL_ROWS = [
    {"example_id": "7-a", "sample_index": 70, "label": 7},
    {"example_id": "3-a", "sample_index": 31, "label": 3},
    {"example_id": "7-d", "sample_index": 72, "label": 7},
    {"example_id": "5-a", "sample_index": 50, "label": 5},
]
LITERAL_MANIFEST = LITERAL_ROWS + [
    {"example_id": "7-b", "sample_index": 71, "label": 7},
    {"example_id": "7-c", "sample_index": 73, "label": 7},
    {"example_id": "3-b", "sample_index": 30, "label": 3},
    {"example_id": "5-b", "sample_index": 51, "label": 5},
]

HEX_A = "a" * 64
HEX_B = "b" * 64
OPERATORS = (
    "proxy_anchor",
    "atomic_one_hot",
    "atomic_complementary",
    "atomic_full_union",
    "summed_union",
    "summed_dropout",
)
PANELS = ("network_only", "joint_including_proxies")
REGIMES = ("configured_loss_stateless", "equal_norm")
METRICS = ("R_F", "Delta_M", "D_F", "D_M")
THRESHOLDS = {
    "shared_confuser_excess": 0.010,
    "network_equal_union_advantage_foreign": 0.001,
    "network_equal_union_advantage_margin": 0.001,
    "network_equal_union_foreign_suppression": 0.001,
    "network_equal_union_margin_change": 0.000,
    "network_equal_union_predicted_suppression": 0.001,
    "network_equal_union_predicted_margin_change": 0.000,
    "joint_equal_union_advantage_foreign": 0.000,
    "joint_equal_union_advantage_margin": 0.000,
    "joint_equal_union_foreign_suppression": 0.000,
    "joint_equal_union_margin_change": 0.000,
}


def _literal_tensor_frame(array: np.ndarray) -> bytes:
    contiguous = np.ascontiguousarray(array)
    dtype_bytes = contiguous.dtype.str.encode("utf-8")
    payload = contiguous.tobytes(order="C")
    return b"".join(
        [
            struct.pack("<I", len(dtype_bytes)),
            dtype_bytes,
            struct.pack("<I", contiguous.ndim),
            *(struct.pack("<q", dimension) for dimension in contiguous.shape),
            struct.pack("<Q", len(payload)),
            payload,
        ]
    )


def test_canonical_json_bytes_uses_frozen_encoding():
    value = {"z": "café", "a": [1, True, None]}
    assert MODULE.canonical_json_bytes(value) == b'{"a":[1,true,null],"z":"caf\xc3\xa9"}'
    with pytest.raises(ValueError):
        MODULE.canonical_json_bytes({"bad": float("nan")})


def test_tensor_digest_frames_dtype_shape_and_payload():
    base = np.array([[1.0, 2.0]], dtype="<f8")
    expected = hashlib.sha256(_literal_tensor_frame(base)).hexdigest()
    assert MODULE.sha256_tensor_frame(base) == expected
    assert MODULE.sha256_tensor_frame(base) != MODULE.sha256_tensor_frame(base.astype("<f4"))
    assert MODULE.sha256_tensor_frame(base) != MODULE.sha256_tensor_frame(base.reshape(2, 1))


def test_named_tensor_digest_frames_names_shapes_and_lengths():
    left = [("ab", np.array([1.0], dtype="<f8"))]
    right = [("a", np.array([98.0, 1.0], dtype="<f8"))]
    assert MODULE.sha256_named_tensors(left) != MODULE.sha256_named_tensors(right)


def test_named_tensor_digest_has_literal_framing_and_little_endian_float64():
    array = np.array([1.0, -2.0], dtype=">f4")
    name = "w.β".encode()
    normalized = np.asarray(array, dtype="<f8")
    frame = b"".join(
        [
            struct.pack("<I", len(name)),
            name,
            struct.pack("<I", 1),
            struct.pack("<q", 2),
            struct.pack("<Q", normalized.nbytes),
            normalized.tobytes(order="C"),
        ]
    )
    assert MODULE.sha256_named_tensors([("w.β", array)]) == hashlib.sha256(frame).hexdigest()


def test_s_prime_is_disjoint_and_preserves_literal_label_sequence():
    context = MODULE.construct_one_context(
        rows=LITERAL_ROWS, train_manifest=LITERAL_MANIFEST, context_index=0
    )
    assert context["row_labels"] == [7, 3, 7, 5]
    assert context["s_prime_example_ids"] == ["7-b", "3-b", "7-c", "5-b"]
    assert set(context["row_example_ids"]).isdisjoint(context["s_prime_example_ids"])


def test_representatives_use_sorted_labels_and_minimum_stable_index():
    context = MODULE.construct_one_context(
        rows=LITERAL_ROWS, train_manifest=LITERAL_MANIFEST, context_index=0
    )
    assert context["representative_row_indices"] == [1, 3, 0]
    assert context["representative_sample_indices"] == [31, 50, 70]


def test_cross_context_reuse_is_causal_prefix_only():
    first = MODULE.construct_one_context(
        rows=LITERAL_ROWS, train_manifest=LITERAL_MANIFEST, context_index=0
    )
    second_rows = [
        {"example_id": "7-a", "sample_index": 70, "label": 7},
        {"example_id": "8-a", "sample_index": 80, "label": 8},
    ]
    manifest = second_rows + [
        {"example_id": "7-e", "sample_index": 74, "label": 7},
        {"example_id": "8-b", "sample_index": 81, "label": 8},
    ]
    second = MODULE.construct_one_context(
        rows=second_rows,
        train_manifest=manifest,
        context_index=1,
        prior_contexts=[first],
    )
    assert first["cross_context_reuse"] == {
        "prior_context_indices_sharing_s_ids": [],
        "prior_context_indices_sharing_s_prime_ids": [],
        "prior_context_indices_sharing_any_ids": [],
        "reused_s_image_count": 0,
        "reused_s_prime_image_count": 0,
        "reused_any_image_count": 0,
        "reused_label_count": 0,
    }
    assert second["cross_context_reuse"] == {
        "prior_context_indices_sharing_s_ids": [0],
        "prior_context_indices_sharing_s_prime_ids": [],
        "prior_context_indices_sharing_any_ids": [0],
        "reused_s_image_count": 1,
        "reused_s_prime_image_count": 0,
        "reused_any_image_count": 1,
        "reused_label_count": 1,
    }


def test_rejected_batch_remains_in_partial_audit():
    infeasible = [
        {"example_id": "9-a", "sample_index": 90, "label": 9},
        {"example_id": "9-b", "sample_index": 91, "label": 9},
    ]
    manifest = (
        LITERAL_MANIFEST + infeasible + [{"example_id": "9-c", "sample_index": 92, "label": 9}]
    )
    accepted, audit = MODULE.construct_context_audit(
        batches=[LITERAL_ROWS, infeasible], train_manifest=manifest, target_count=2
    )
    assert len(accepted) == 1
    assert [entry["status"] for entry in audit] == ["accepted", "rejected"]
    rejected = audit[1]
    assert rejected["context_index"] == 1
    assert rejected["rejection_code"] == "INSUFFICIENT_DISJOINT_S_PRIME"
    assert rejected["row_example_ids"] == ["9-a", "9-b"]
    assert rejected["s_prime_example_ids"] == []
    assert rejected["s_prime_sample_indices"] == []


def test_input_context_digest_excludes_process_metadata():
    context = MODULE.construct_one_context(
        rows=LITERAL_ROWS, train_manifest=LITERAL_MANIFEST, context_index=0
    )
    context["s_tensor"] = np.arange(8, dtype="<f4").reshape(2, 4)
    context["s_prime_tensor"] = np.arange(8, 16, dtype="<f4").reshape(2, 4)
    context["pid"] = 111
    left = MODULE.build_input_context_digest(context)
    context["pid"] = 222
    right = MODULE.build_input_context_digest(context)
    assert left == right
    assert set(left) == {
        "context_index",
        "s_tensor_sha256",
        "s_prime_tensor_sha256",
        "metadata_sha256",
        "combined_sha256",
    }
    combined = {key: value for key, value in left.items() if key != "combined_sha256"}
    assert (
        left["combined_sha256"]
        == hashlib.sha256(
            json.dumps(
                combined,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
    )


def _summary(*, lcb: float = 0.02, ucb: float = 0.03) -> dict:
    return {
        "n": 32,
        "mean": 0.025,
        "median": 0.025,
        "sample_sd": 0.001,
        "q25": 0.024,
        "q75": 0.026,
        "lcb_0_005": lcb,
        "ucb_0_995": ucb,
    }


def _update(*, equal_norm: bool) -> dict:
    return {
        "update_sha256": HEX_B,
        "parameter_update_norm": 0.1,
        "R_F": 0.02,
        "Delta_M": 0.02,
        "D_F": 0.02,
        "D_M": 0.02,
        "reference_pa_norm": 0.1 if equal_norm else None,
        "norm_match_absolute_error": 0.0 if equal_norm else None,
    }


def _operator(name: str) -> dict:
    panels = {}
    for panel in PANELS:
        panels[panel] = {
            "parameter_count": 2,
            "gradient_sha256": HEX_A,
            "raw_gradient_norm": 0.1,
            "update_space_norm": 0.1,
            "auxiliary_to_pa_norm_ratio": 1.0,
            "cosine_with_pa": 1.0,
            "cosine_with_atomic_full_union": 1.0,
            "cosine_with_summed_dropout": 1.0,
            "scale_residual_to_summed_union": (
                1.0
                if name in ("atomic_one_hot", "atomic_complementary", "atomic_full_union")
                else None
            ),
            "updates": {
                "configured_loss_stateless": _update(equal_norm=False),
                "equal_norm": _update(equal_norm=True),
            },
        }
    return {
        "name": name,
        "loss": 0.5,
        "representative_count": 2,
        "panels": panels,
    }


def _context(context_index: int) -> dict:
    row_ids = [f"{context_index}-s-{index}" for index in range(180)]
    s_prime_ids = [f"{context_index}-p-{index}" for index in range(180)]
    return {
        "context_index": context_index,
        "production_epoch": 0,
        "production_batch_index": context_index,
        "batch_size": 180,
        "m_unique": 2,
        "row_example_ids": row_ids,
        "row_sample_indices": list(range(180)),
        "row_labels": [1, 2] * 90,
        "class_multiplicities": {"1": 90, "2": 90},
        "representative_row_indices": [0, 1],
        "representative_sample_indices": [0, 1],
        "s_tensor_sha256": HEX_A,
        "s_prime_example_ids": s_prime_ids,
        "s_prime_sample_indices": list(range(180, 360)),
        "s_prime_tensor_sha256": HEX_B,
        "cross_context_reuse": {
            "prior_context_indices_sharing_s_ids": [],
            "prior_context_indices_sharing_s_prime_ids": [],
            "prior_context_indices_sharing_any_ids": [],
            "reused_s_image_count": 0,
            "reused_s_prime_image_count": 0,
            "reused_any_image_count": 0,
            "reused_label_count": 0,
        },
        "foreign_proxy_rows": 3,
        "shared_confuser": {
            "A_aligned": 0.2,
            "null_mean": 0.1,
            "E_shared": 1.0,
            "null_distribution_sha256": HEX_A,
        },
        "operators": {name: _operator(name) for name in OPERATORS},
    }


def _digest_record(context: dict) -> dict:
    metadata_keys = (
        "context_index",
        "production_epoch",
        "production_batch_index",
        "row_example_ids",
        "row_sample_indices",
        "row_labels",
        "class_multiplicities",
        "representative_row_indices",
        "representative_sample_indices",
        "s_prime_example_ids",
        "s_prime_sample_indices",
        "cross_context_reuse",
    )
    record = {
        "context_index": context["context_index"],
        "s_tensor_sha256": context["s_tensor_sha256"],
        "s_prime_tensor_sha256": context["s_prime_tensor_sha256"],
        "metadata_sha256": hashlib.sha256(
            json.dumps(
                {key: context[key] for key in metadata_keys},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }
    record["combined_sha256"] = hashlib.sha256(
        json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    return record


def test_named_tensor_digest_enforces_lexicographic_utf8_name_order():
    tensors = [
        ("z.weight", np.array([2.0])),
        ("ä.weight", np.array([3.0])),
        ("a.weight", np.array([1.0])),
    ]
    expected_order = [tensors[2], tensors[0], tensors[1]]
    assert MODULE.sha256_named_tensors(tensors) == MODULE.sha256_named_tensors(expected_order)


def _metric_paths() -> list[str]:
    paths = ["m_unique", "shared_confuser.E_shared"]
    for regime in REGIMES:
        for panel in PANELS:
            for operator in OPERATORS:
                for metric in METRICS:
                    paths.append(f"{regime}.{panel}.operators.{operator}.{metric}")
            for metric in ("A_F", "A_M"):
                paths.append(f"{regime}.{panel}.paired_advantages.{metric}")
    return paths


def _aggregates() -> dict:
    aggregates = {"m_unique": _summary(), "shared_confuser": _summary(lcb=0.01)}
    for regime in REGIMES:
        aggregates[regime] = {}
        for panel in PANELS:
            aggregates[regime][panel] = {
                "operators": {
                    operator: {metric: _summary() for metric in METRICS} for operator in OPERATORS
                },
                "paired_advantages": {
                    "A_F": _summary(),
                    "A_M": _summary(),
                },
            }
    aggregates["bootstrap"] = {
        "seed": 2010811,
        "replicates": 20000,
        "quantile_method": "linear",
        "joint_context_index_sha256": HEX_A,
        "distribution_sha256_by_metric": {path: HEX_B for path in _metric_paths()},
    }
    return aggregates


def _source() -> dict:
    return {
        "prelaunch_source_manifest_path": "docs/pass201_pa_source_prelaunch_manifest.json",
        "prelaunch_source_manifest_sha256": (
            "37644551f99976a7982589c1574effa00a9c77aa4a690117b5a8cd84244cc803"
        ),
        "source_report_path": "reports/source.json",
        "source_report_sha256": HEX_A,
        "source_revision": "f" * 40,
        "checkpoint_path": "checkpoints/source.pt",
        "checkpoint_sha256": HEX_A,
        "checkpoint_bytes": 123,
        "checkpoint_epoch": 2,
        "resolved_config_path": "reports/config.json",
        "resolved_config_sha256": HEX_A,
        "train_manifest_path": "reports/train.json",
        "train_manifest_sha256": HEX_A,
        "diagnostic_source_sha256": HEX_A,
        "activated_preregistration_sha256": HEX_A,
        "python_version": "3.11.0",
        "torch_version": "2.0.0",
        "numpy_version": "2.0.0",
        "cuda_version": "12.0",
        "cudnn_version": "9",
    }


def _constants() -> dict:
    return {
        "batch_size": 180,
        "context_pairs": 32,
        "null_replicates": 256,
        "bootstrap_replicates": 20000,
        "s_prime_rank_seed": 2010809,
        "null_seed": 2010810,
        "bootstrap_seed": 2010811,
        "model_forward_seed": 2010812,
        "learning_rate": 0.001,
        "coalition_weight": 0.2,
        "proxy_learning_rate_multiplier": 10.0,
        "owner_margin_temperature": 0.05,
    }


def _process_record(role: str, contexts: list[dict]) -> dict:
    return {
        "role": role,
        "pid": 100,
        "accelerator": "synthetic",
        "python_version": "3.11.0",
        "torch_version": "2.0.0",
        "cuda_version": "12.0",
        "cudnn_version": "9",
        "visible_cuda_devices": ["0000:00:01.0"],
        "initial_python_rng_sha256": HEX_A,
        "initial_numpy_rng_sha256": HEX_A,
        "initial_torch_cpu_rng_sha256": HEX_A,
        "initial_torch_cuda_rng_sha256_by_device": {"0": HEX_A},
        "prepared_context_count": 32,
        "input_context_digest_records": [_digest_record(context) for context in contexts],
        "context0_record_sha256": hashlib.sha256(
            json.dumps(
                contexts[0],
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    }


def literal_valid_scored_payload() -> dict:
    contexts = [_context(index) for index in range(32)]
    records = [
        _process_record(role, contexts)
        for role in ("integrity_replay_a", "integrity_replay_b", "scientific")
    ]
    return {
        "schema_version": "pass201-cis-operator-v1",
        "status": "PASS",
        "reason_codes": [],
        "candidate_values_computed": True,
        "uses_test_data": "artifact_binding_only",
        "source": _source(),
        "constants": _constants(),
        "contexts": contexts,
        "aggregates": _aggregates(),
        "decision": {
            "thresholds": dict(THRESHOLDS),
            "component_decisions": {key: "PASS" for key in THRESHOLDS},
            "overall": "PASS",
            "authorized_next_action": "write_separate_gpu_preregistration",
        },
        "integrity": {
            "accepted_context_count": 32,
            "rejected_context_count": 0,
            "invalid_context_count": 0,
            "input_replay_verified": True,
            "parameter_hash_before": HEX_A,
            "parameter_hash_after": HEX_A,
            "buffer_hash_before": HEX_B,
            "buffer_hash_after": HEX_B,
            "training_flags_restored": True,
            "deterministic_process_verified": True,
            "first_context_operator_replay_verified": True,
            "deterministic_settings": {
                "cublas_workspace_config": ":4096:8",
                "deterministic_algorithms": True,
                "cudnn_benchmark": False,
                "cudnn_deterministic": True,
                "matmul_tf32": False,
                "cudnn_tf32": False,
                "autocast": False,
                "dtype": "float32",
            },
            "process_records": records,
            "replay_residuals": {
                "pair_count": 3,
                "tensor_max_absolute": 2e-6,
                "scalar_max_relative": 1e-5,
                "tensor_tolerance": 2e-6,
                "scalar_tolerance": 1e-5,
                "scalar_denominator": "max(abs(a),abs(b),1e-12)",
            },
            "all_finite": True,
        },
    }


def _failure_digest(status: str, reason_codes: list[str], integrity: dict) -> str:
    process_records = integrity["process_records"]
    evidence = {
        "status": status,
        "reason_codes": sorted(reason_codes),
        "stage": integrity["stage"],
        "accepted_context_count": integrity["accepted_context_count"],
        "rejected_context_count": integrity["rejected_context_count"],
        "invalid_context_count": integrity["invalid_context_count"],
        "last_process_record": process_records[-1] if process_records else None,
    }
    return hashlib.sha256(
        json.dumps(
            evidence,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def literal_valid_blocked_payload() -> dict:
    reason_codes = ["BLOCKED_SOURCE_UNAVAILABLE"]
    integrity = {
        "stage": "source_activation",
        "accepted_context_count": 0,
        "rejected_context_count": 0,
        "invalid_context_count": 0,
        "input_replay_verified": False,
        "deterministic_process_verified": False,
        "process_records": [],
        "failure_evidence_sha256": "",
        "all_finite": False,
    }
    integrity["failure_evidence_sha256"] = _failure_digest("BLOCKED", reason_codes, integrity)
    source = _source()
    for key in source.keys() - {
        "prelaunch_source_manifest_path",
        "prelaunch_source_manifest_sha256",
    }:
        source[key] = None
    constants = _constants()
    for key in (
        "learning_rate",
        "coalition_weight",
        "proxy_learning_rate_multiplier",
    ):
        constants[key] = None
    return {
        "schema_version": "pass201-cis-operator-v1",
        "status": "BLOCKED",
        "reason_codes": reason_codes,
        "candidate_values_computed": False,
        "uses_test_data": "artifact_binding_only",
        "source": source,
        "constants": constants,
        "decision": {
            "thresholds": dict(THRESHOLDS),
            "overall": "BLOCKED",
            "authorized_next_action": "none",
        },
        "integrity": integrity,
    }


def literal_valid_early_unresolved_payload() -> dict:
    full = _context(0)
    contexts = []
    for index in range(32):
        entry = {
            key: deepcopy(value)
            for key, value in full.items()
            if key
            in {
                "context_index",
                "production_epoch",
                "production_batch_index",
                "row_example_ids",
                "row_sample_indices",
                "row_labels",
                "class_multiplicities",
                "representative_row_indices",
                "representative_sample_indices",
                "s_prime_example_ids",
                "s_prime_sample_indices",
            }
        }
        entry["context_index"] = index
        entry["production_batch_index"] = index
        entry["status"] = "accepted" if index < 31 else "rejected"
        entry["rejection_code"] = None if index < 31 else "INSUFFICIENT_DISJOINT_S_PRIME"
        if index == 31:
            entry["s_prime_example_ids"] = []
            entry["s_prime_sample_indices"] = []
        contexts.append(entry)
    process = _process_record("integrity_replay_a", [_context(index) for index in range(32)])
    process["prepared_context_count"] = 31
    process["input_context_digest_records"] = process["input_context_digest_records"][:31]
    process["context0_record_sha256"] = None
    reason_codes = ["UNRESOLVED_INSUFFICIENT_DISJOINT_CONTEXTS"]
    integrity = {
        "stage": "context_construction",
        "accepted_context_count": 31,
        "rejected_context_count": 1,
        "invalid_context_count": 0,
        "input_replay_verified": False,
        "deterministic_process_verified": False,
        "process_records": [process],
        "failure_evidence_sha256": "",
        "all_finite": True,
    }
    integrity["failure_evidence_sha256"] = _failure_digest("UNRESOLVED", reason_codes, integrity)
    return {
        "schema_version": "pass201-cis-operator-v1",
        "status": "UNRESOLVED",
        "reason_codes": reason_codes,
        "candidate_values_computed": False,
        "uses_test_data": "artifact_binding_only",
        "source": _source(),
        "constants": _constants(),
        "contexts": contexts,
        "decision": {
            "thresholds": dict(THRESHOLDS),
            "overall": "UNRESOLVED",
            "authorized_next_action": "none",
        },
        "integrity": integrity,
    }


def test_bootstrap_indices_are_the_frozen_shared_matrix_and_digest():
    indices = MODULE.bootstrap_indices()
    expected = np.random.Generator(np.random.PCG64(2010811)).integers(
        0, 32, size=(20000, 32), dtype=np.int64
    )
    assert indices.dtype.str == "<i8"
    assert indices.flags.c_contiguous
    assert np.array_equal(indices, expected)
    assert hashlib.sha256(indices.tobytes(order="C")).hexdigest() == (
        MODULE.sha256_bootstrap_indices(indices)
    )


def test_summary_uses_sample_sd_and_frozen_bootstrap_bounds():
    values = np.arange(32, dtype=np.float64)
    indices = np.tile(np.arange(32, dtype="<i8"), (20000, 1))
    summary = MODULE.summarize_metric(values, indices)
    assert summary == {
        "n": 32,
        "mean": 15.5,
        "median": 15.5,
        "sample_sd": pytest.approx(np.std(values, ddof=1)),
        "q25": 7.75,
        "q75": 23.25,
        "lcb_0_005": 15.5,
        "ucb_0_995": 15.5,
    }
    distribution = MODULE.bootstrap_mean_distribution(values, indices)
    assert distribution.dtype.str == "<f8"
    assert hashlib.sha256(distribution.tobytes(order="C")).hexdigest() == (
        MODULE.sha256_bootstrap_distribution(distribution)
    )


@pytest.mark.parametrize(
    "payload_factory",
    [
        literal_valid_scored_payload,
        literal_valid_early_unresolved_payload,
        literal_valid_blocked_payload,
    ],
)
def test_payload_validator_accepts_each_conditional_family(payload_factory):
    MODULE.validate_payload_structure(payload_factory())


def test_early_payload_rejects_internally_wrong_partial_audit_metadata():
    payload = literal_valid_early_unresolved_payload()
    payload["contexts"][0]["class_multiplicities"] = {"3": 2, "5": 1, "7": 1}
    with pytest.raises(ValueError, match="class_multiplicities"):
        MODULE.validate_payload_structure(payload)


def _extra_key(payload):
    payload["unexpected"] = True


def _missing_key(payload):
    del payload["source"]


def _wrong_null(payload):
    payload["contexts"][0]["operators"]["proxy_anchor"]["panels"]["network_only"][
        "scale_residual_to_summed_union"
    ] = 1.0


def _nonfinite(payload):
    payload["contexts"][0]["shared_confuser"]["E_shared"] = float("inf")


def _status_mismatch(payload):
    payload["status"] = "FAIL"


def _wrong_process_prefix(payload):
    payload["integrity"]["process_records"][0]["role"] = "scientific"


def _unsorted_visible_devices(payload):
    record = payload["integrity"]["process_records"][0]
    record["visible_cuda_devices"] = ["0000:00:02.0", "0000:00:01.0"]
    record["initial_torch_cuda_rng_sha256_by_device"] = {"0": HEX_A, "1": HEX_A}


def _source_activation_invalid_count(payload):
    blocked = literal_valid_blocked_payload()
    payload.clear()
    payload.update(blocked)
    payload["status"] = "INVALID"
    payload["reason_codes"] = ["INVALID_SOURCE_BINDING"]
    payload["decision"]["overall"] = "INVALID"
    payload["integrity"]["invalid_context_count"] = 1
    payload["integrity"]["failure_evidence_sha256"] = _failure_digest(
        "INVALID", payload["reason_codes"], payload["integrity"]
    )


def _component_reason_mixing(payload):
    payload["reason_codes"] = ["FAIL_NO_SHARED_CONFOUNDER"]


def _malformed_digest(payload):
    payload["source"]["checkpoint_sha256"] = "ABC"


def _wrong_prelaunch_digest(payload):
    payload["source"]["prelaunch_source_manifest_sha256"] = "0" * 64


def _context0_hash_includes_process_metadata(payload):
    record = payload["integrity"]["process_records"][0]
    record["context0_record_sha256"] = hashlib.sha256(
        json.dumps(
            {**payload["contexts"][0], "pid": record["pid"]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _self_consistent_wrong_class_multiplicities(payload):
    context = payload["contexts"][0]
    context["class_multiplicities"] = {"1": 89, "2": 91}
    digest_record = _digest_record(context)
    context_sha256 = hashlib.sha256(
        json.dumps(
            context,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    for record in payload["integrity"]["process_records"]:
        record["input_context_digest_records"][0] = deepcopy(digest_record)
        record["context0_record_sha256"] = context_sha256


@pytest.mark.parametrize(
    "mutation",
    [
        _extra_key,
        _missing_key,
        _wrong_null,
        _nonfinite,
        _status_mismatch,
        _wrong_process_prefix,
        _unsorted_visible_devices,
        _source_activation_invalid_count,
        _component_reason_mixing,
        _malformed_digest,
        _wrong_prelaunch_digest,
        _context0_hash_includes_process_metadata,
        _self_consistent_wrong_class_multiplicities,
    ],
)
def test_payload_validator_fails_closed(mutation):
    payload = literal_valid_scored_payload()
    mutation(payload)
    with pytest.raises(ValueError):
        MODULE.validate_payload_structure(payload)


def test_threshold_boundaries_are_inclusive_and_failure_reasons_have_precedence():
    payload = literal_valid_scored_payload()
    aggregates = payload["aggregates"]
    aggregates["shared_confuser"]["lcb_0_005"] = 0.010
    aggregates["equal_norm"]["network_only"]["paired_advantages"]["A_F"]["lcb_0_005"] = 0.001
    MODULE.validate_payload_structure(payload)

    aggregates["shared_confuser"]["lcb_0_005"] = -0.1
    aggregates["shared_confuser"]["ucb_0_995"] = 0.0
    aggregates["equal_norm"]["network_only"]["paired_advantages"]["A_F"]["lcb_0_005"] = -0.1
    aggregates["equal_norm"]["network_only"]["paired_advantages"]["A_F"]["ucb_0_995"] = 0.0
    payload["status"] = "FAIL"
    payload["reason_codes"] = [
        "FAIL_NO_SHARED_CONFOUNDER",
        "FAIL_NO_COALITION_SPECIFIC_ACTION",
        "FAIL_PROXY_ONLY",
    ]
    payload["decision"]["overall"] = "FAIL"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["shared_confuser_excess"] = "FAIL"
    payload["decision"]["component_decisions"]["network_equal_union_advantage_foreign"] = "FAIL"
    MODULE.validate_payload_structure(payload)


def test_any_failed_component_makes_overall_fail_even_without_a_reason_predicate():
    payload = literal_valid_scored_payload()
    equal_joint = payload["aggregates"]["equal_norm"]["joint_including_proxies"]
    equal_joint["paired_advantages"]["A_F"]["lcb_0_005"] = -0.1
    equal_joint["paired_advantages"]["A_F"]["ucb_0_995"] = 0.0
    configured_joint = payload["aggregates"]["configured_loss_stateless"]["joint_including_proxies"]
    configured_joint["paired_advantages"]["A_F"]["lcb_0_005"] = -0.1
    payload["status"] = "FAIL"
    payload["decision"]["overall"] = "FAIL"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["joint_equal_union_advantage_foreign"] = "FAIL"
    MODULE.validate_payload_structure(payload)


def _component_summaries(payload: dict) -> dict[str, dict]:
    aggregates = payload["aggregates"]
    network = aggregates["equal_norm"]["network_only"]
    joint = aggregates["equal_norm"]["joint_including_proxies"]
    return {
        "shared_confuser_excess": aggregates["shared_confuser"],
        "network_equal_union_advantage_foreign": network["paired_advantages"]["A_F"],
        "network_equal_union_advantage_margin": network["paired_advantages"]["A_M"],
        "network_equal_union_foreign_suppression": network["operators"]["summed_union"]["R_F"],
        "network_equal_union_margin_change": network["operators"]["summed_union"]["Delta_M"],
        "network_equal_union_predicted_suppression": network["operators"]["summed_union"]["D_F"],
        "network_equal_union_predicted_margin_change": network["operators"]["summed_union"]["D_M"],
        "joint_equal_union_advantage_foreign": joint["paired_advantages"]["A_F"],
        "joint_equal_union_advantage_margin": joint["paired_advantages"]["A_M"],
        "joint_equal_union_foreign_suppression": joint["operators"]["summed_union"]["R_F"],
        "joint_equal_union_margin_change": joint["operators"]["summed_union"]["Delta_M"],
    }


def test_every_pass_lcb_boundary_is_inclusive():
    payload = literal_valid_scored_payload()
    for key, summary in _component_summaries(payload).items():
        threshold = THRESHOLDS[key]
        summary["lcb_0_005"] = threshold
    MODULE.validate_payload_structure(payload)


def test_owner_margin_zero_ucb_is_unresolved_not_fail():
    payload = literal_valid_scored_payload()
    summary = payload["aggregates"]["equal_norm"]["network_only"]["operators"]["summed_union"][
        "Delta_M"
    ]
    summary["lcb_0_005"] = -0.1
    summary["ucb_0_995"] = 0.0
    payload["status"] = "UNRESOLVED"
    payload["decision"]["overall"] = "UNRESOLVED"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["network_equal_union_margin_change"] = "UNRESOLVED"
    MODULE.validate_payload_structure(payload)


def test_owner_margin_next_float_below_zero_is_fail():
    payload = literal_valid_scored_payload()
    summary = payload["aggregates"]["equal_norm"]["network_only"]["operators"]["summed_union"][
        "D_M"
    ]
    summary["lcb_0_005"] = -0.1
    summary["ucb_0_995"] = float(np.nextafter(0.0, -np.inf))
    payload["status"] = "FAIL"
    payload["reason_codes"] = ["FAIL_OWNER_DAMAGE"]
    payload["decision"]["overall"] = "FAIL"
    payload["decision"]["authorized_next_action"] = "none"
    payload["decision"]["component_decisions"]["network_equal_union_predicted_margin_change"] = (
        "FAIL"
    )
    MODULE.validate_payload_structure(payload)


def literal_valid_scored_payload_and_evidence() -> tuple[dict, dict]:
    payload = literal_valid_scored_payload()
    gradient_tensors = {}
    update_tensors = {}
    null_distributions = {}
    for context in payload["contexts"]:
        context_index = context["context_index"]
        for operator in OPERATORS:
            for panel in PANELS:
                key = f"{context_index}.{operator}.{panel}"
                tensors = [("weight", np.array([1.0, 2.0]))]
                gradient_tensors[key] = tensors
                context["operators"][operator]["panels"][panel]["gradient_sha256"] = (
                    MODULE.sha256_named_tensors(tensors)
                )
                for regime in REGIMES:
                    update_key = f"{key}.{regime}"
                    update_tensors[update_key] = [("weight", np.array([1.0, 2.0]))]
                    context["operators"][operator]["panels"][panel]["updates"][regime][
                        "update_sha256"
                    ] = MODULE.sha256_named_tensors(update_tensors[update_key])
        null = np.linspace(0.0, 1.0, 256, dtype="<f8")
        null_distributions[str(context_index)] = null
        context["shared_confuser"]["null_distribution_sha256"] = hashlib.sha256(
            null.tobytes(order="C")
        ).hexdigest()
    indices = MODULE.bootstrap_indices()
    payload["aggregates"]["bootstrap"]["joint_context_index_sha256"] = hashlib.sha256(
        indices.tobytes(order="C")
    ).hexdigest()
    bootstrap_distributions = {}
    for metric_path in _metric_paths():
        distribution = np.linspace(0.0, 1.0, 20000, dtype="<f8")
        bootstrap_distributions[metric_path] = distribution
        payload["aggregates"]["bootstrap"]["distribution_sha256_by_metric"][metric_path] = (
            hashlib.sha256(distribution.tobytes(order="C")).hexdigest()
        )
    context0_sha256 = hashlib.sha256(
        json.dumps(
            payload["contexts"][0],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()
    for record in payload["integrity"]["process_records"]:
        record["context0_record_sha256"] = context0_sha256
    return payload, {
        "gradient_tensors": gradient_tensors,
        "update_tensors": update_tensors,
        "null_distributions": null_distributions,
        "bootstrap_indices": indices,
        "bootstrap_distributions": bootstrap_distributions,
    }


def test_construction_validation_rejects_digest_raw_evidence_mismatch():
    payload, evidence = literal_valid_scored_payload_and_evidence()
    MODULE.validate_construction_evidence(payload, evidence)
    evidence["gradient_tensors"]["0.proxy_anchor.network_only"][0][1][0] += 1.0
    with pytest.raises(ValueError, match="gradient_sha256"):
        MODULE.validate_construction_evidence(payload, evidence)


def _mutate_update_evidence(evidence):
    evidence["update_tensors"]["0.proxy_anchor.network_only.configured_loss_stateless"][0][1][
        0
    ] += 1.0


def _mutate_null_evidence(evidence):
    evidence["null_distributions"]["0"][0] += 1.0


def _mutate_bootstrap_indices(evidence):
    evidence["bootstrap_indices"][0, 0] = (evidence["bootstrap_indices"][0, 0] + 1) % 32


def _mutate_bootstrap_distribution(evidence):
    evidence["bootstrap_distributions"]["m_unique"][0] += 1.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (_mutate_update_evidence, "update_sha256"),
        (_mutate_null_evidence, "null_distribution_sha256"),
        (_mutate_bootstrap_indices, "bootstrap indices"),
        (_mutate_bootstrap_distribution, "distribution_sha256"),
    ],
)
def test_construction_validation_checks_every_live_evidence_family(mutation, message):
    payload, evidence = literal_valid_scored_payload_and_evidence()
    mutation(evidence)
    with pytest.raises(ValueError, match=message):
        MODULE.validate_construction_evidence(payload, evidence)
