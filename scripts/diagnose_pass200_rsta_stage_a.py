#!/usr/bin/env python3
"""Preregistered Pass200 RSTA Stage-A diagnostic.

Imports remain side-effect free: artifact, dataset, model, and torch work occurs
only through explicit binding/cache/CLI calls.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import os
import random
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from diagnose_pass159_cotangent_stage_a import (  # noqa: E402
    _canonical_query_gallery_recall_at_1,
    _load_final_pack,
    _manifest_paths,
    load_bound_seed,
    sha256_file,
)

_ROLE_DOMAIN = "rsta-stage-a-v1|role|"
_IDENTITY_DOMAIN = "rsta-stage-a-v1|identity|"
_DISTRACTOR_DOMAIN = "rsta-stage-a-v1|distractor|"
_ALTERNATE_DISTRACTOR_DOMAIN = "rsta-stage-a-v1|alternate-distractor|"
_BATCH_ORDER_PREFIX = "rsta-stage-a-v1|batch-order|"
_PRIMARY_IDENTITIES = 64
_RECEIVERS_PER_BATCH = 8
_DISTRACTORS_PER_BATCH = 172
_VECTOR_EPS = 1.0e-12
_ROTATION_VECTOR_NAMES = frozenset(("z", "dbar", "b", "s", "q"))
_ROTATION_STATISTIC_NAMES = frozenset(
    ("A_self", "A_batch", "Delta", "A_desc", "rho", "log_ratio", "cos_b_s")
)
_OFFICIAL_PARTITION = {
    "train": (25_882, 3_997),
    "query": (14_218, 3_985),
    "gallery": (12_612, 3_985),
}
_SOURCE_SPLITS = ("train", "query", "gallery")
_SOURCE_EXPORT_KEYS = {
    "embeddings",
    "labels",
    "example_ids",
    "source_paths",
    "row_indices",
}
_FROZEN_SOURCE_FILES = frozenset(
    {
        "scripts/diagnose_pass159_cotangent_stage_a.py",
        "scripts/diagnose_pass200_rsta_stage_a.py",
        "scripts/export_final_inshop_embeddings.py",
        "src/sfora/bn_inception.py",
        "src/sfora/data.py",
        "src/sfora/image_end_to_end.py",
    }
)
_DIAGNOSTIC_PATH = "scripts/diagnose_pass200_rsta_stage_a.py"
_EXECUTION_AUDIT_FIELDS = frozenset(
    {
        "executing_git_commit",
        "diagnostic_path",
        "diagnostic_sha256",
        "frozen_source_revision",
    }
)
_HISTORICAL_RECEIPT_PATH = "docs/pass200_rsta_binding_receipt_d6270a9.json"
_HISTORICAL_RECEIPT_SHA256 = (
    "e75944aed5af0fbe53af9febbc9a9a5d30045357eb6b1f086c4ba61e10f82300"
)
_HISTORICAL_PRODUCER_COMMIT = "d6270a94f14f5e0b4f4a3eeaa23f3f66d9bfaa54"
_HISTORICAL_MANIFEST_PATH = "docs/pass200_rsta_stage_a_manifest.json"
_HISTORICAL_MANIFEST_SHA256 = (
    "aafab355a06667a9ca513cddeceb2a0129ea8ee09ce3dec0a19b6839fe15ffb1"
)
_BASE_PREREGISTRATION_PATH = "docs/pass200_rsta_candidate_2026-08-09.md"
_BASE_PREREGISTRATION_SHA256 = (
    "a35cd3469d5561ce59202030dd3c3050e018dbfc537cb0ee0401a1d0340f5857"
)
_HISTORICAL_SOURCE_REVISION = "0146f2d1200fec26fcd483005804dbe71ec72786"
_HISTORICAL_DIAGNOSTIC_SHA256 = (
    "78eeb3d0d3f92ad1a0b7e76708851e940a36a1ef260a2618dc58bf7f3fab7f1a"
)
_ARTIFACT_NAMES = frozenset(
    {
        "checkpoint_pt",
        "gallery_npz",
        "prehead_npz",
        "query_npz",
        "report_json",
        "retrieval_json",
        "train_npz",
    }
)
_AMENDMENT_PATH = "docs/pass200_rsta_binding_receipt_amendment_2026-08-09.md"
_AMENDMENT_SHA256 = "691d786942c33cf8a943159280287bea08570114242854cdb7111795dc79e019"
_AMENDMENT_COMMIT = "d1aeed63ade0e15d5f5a44be5981a4312e9a8df2"
_DETERMINISTIC_POOL_AMENDMENT_PATH = (
    "docs/pass200_rsta_deterministic_global_max_amendment_2026-08-09.md"
)
_DETERMINISTIC_POOL_AMENDMENT_SHA256 = (
    "6b2ffed724f0056b011831bb74997cb3e8d50f83304448805b119f6a3d78b361"
)
_DETERMINISTIC_POOL_AMENDMENT_COMMIT = "db29ab7bb6478cfef57eccbad142f93d2f805f7f"
_ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_PATH = (
    "docs/pass200_rsta_zero_jacobian_classifier_amendment_2026-08-09.md"
)
_ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_SHA256 = (
    "4b981efd3893436e1a4da09568c3cf167d7beeeb8fd637979b5869588c956ade"
)
_ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_COMMIT = (
    "85e8f983053f3839e5bbb2bb11563380e6b77919"
)
_DETERMINISTIC_GLOBAL_MAX_INPUT_SHA256 = {
    "random": "849f58506a8eabf18741d830a3d83e053d327786a8bfe731df0556b31d43389c",
    "relu": "5810fd957d263f60a15aff4c9a4cb3401a7ad99b165413eaa8503026582a8887",
    "zeros": "16d0edc8b7ad7705b23a14058f366ff1c0dfa16a0ad14f741924c308754cf8d1",
    "tie": "55688cd7f3585fc5402d755dde3f30ac70701bea80c44b8a2e13d5dfa394d5b5",
}
_CURRENT_SCIENTIFIC_SOURCE_FILES = frozenset(
    {
        "scripts/diagnose_pass159_cotangent_stage_a.py",
        "scripts/diagnose_pass200_rsta_stage_a.py",
        "src/sfora/__init__.py",
        "src/sfora/ablation.py",
        "src/sfora/api.py",
        "src/sfora/arcg.py",
        "src/sfora/benchmark.py",
        "src/sfora/bn_inception.py",
        "src/sfora/catalog.py",
        "src/sfora/cea.py",
        "src/sfora/cem.py",
        "src/sfora/cli.py",
        "src/sfora/compose.py",
        "src/sfora/data.py",
        "src/sfora/encoder_ablation.py",
        "src/sfora/encoder_training.py",
        "src/sfora/evaluation.py",
        "src/sfora/experiments.py",
        "src/sfora/image_benchmark.py",
        "src/sfora/image_end_to_end.py",
        "src/sfora/image_recipes.py",
        "src/sfora/ipsr.py",
        "src/sfora/losses.py",
        "src/sfora/method.py",
        "src/sfora/oapf.py",
        "src/sfora/publication.py",
        "src/sfora/remote.py",
        "src/sfora/report.py",
        "src/sfora/text_baselines.py",
        "src/sfora/training.py",
    }
)
RECEIVER_AUDIT_FIELDS = frozenset(
    {
        "panel",
        "seed",
        "label",
        "batch_index",
        "receiver_index",
        "receiver_id",
        "support_ids",
        "foreign_ids",
        "batch_ids",
        "batch_tensor_sha256",
        "batch_id_order_sha256",
        "tensor_sha256",
        "a_self",
        "a_batch",
        "delta",
        "a_desc",
        "self_minus_desc",
        "rho",
        "log_ratio",
        "cos_b_s",
        "random_a_self",
        "random_a_batch",
        "random_delta",
        "deranged_a_self",
        "deranged_a_batch",
        "deranged_delta",
        "norm_z",
        "norm_dbar",
        "norm_b",
        "norm_s",
        "norm_q",
        "norm_random_target",
        "norm_deranged_target",
        "radial_fraction_dbar",
        "radial_fraction_b",
        "radial_fraction_s",
        "head_a_batch",
        "head_a_self",
        "head_self_desc_gap",
        "norm_b_head",
        "norm_s_head",
        "support_cosines",
    }
)
SEED_AUDIT_FIELDS = frozenset(
    {
        "seed",
        "official_recall_at_1",
        "artifact_binding",
        "config",
        "parameter_names",
        "parameter_count",
        "proxy_sha256",
        "proxy_label_sha256",
        "train_example_id_order_sha256",
        "train_label_order_sha256",
        "train_source_order_sha256",
        "transform_cache_order_sha256",
        "transform_tensor_sha256",
        "primary_batch_ids",
        "alternate_batch_ids",
    }
)
ENVIRONMENT_AUDIT_FIELDS = frozenset(
    {
        "cublas_workspace_config",
        "deterministic_algorithms",
        "deterministic_warn_only",
        "cudnn_benchmark",
        "cuda_matmul_tf32",
        "cudnn_tf32",
        "autocast",
        "model_arithmetic",
        "reduction_arithmetic",
        "torch_version",
        "numpy_version",
    }
)


def _readonly_array(values: Any, *, dtype: Any | None = None) -> np.ndarray:
    source = np.asarray(values, dtype=dtype)
    return np.frombuffer(source.tobytes(order="C"), dtype=source.dtype).reshape(source.shape)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({str(key): _deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_json_ready(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_strict_json_bytes(data: bytes, *, name: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"nonfinite JSON constant: {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data,
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid strict JSON: {name}") from error
    if not isinstance(value, dict):
        raise ValueError("strict JSON root must be an object")

    def require_finite(item: Any) -> None:
        if isinstance(item, float) and not np.isfinite(item):
            raise ValueError("nonfinite JSON number")
        if isinstance(item, dict):
            for nested in item.values():
                require_finite(nested)
        elif isinstance(item, list):
            for nested in item:
                require_finite(nested)

    require_finite(value)
    return value


def load_strict_json(path: Path) -> dict[str, Any]:
    """Load finite JSON while rejecting duplicate object keys."""
    return _load_strict_json_bytes(path.read_bytes(), name=str(path))


def _require_exact_keys(
    value: Any,
    expected: set[str] | frozenset[str],
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != set(expected):
        observed = set(value) if isinstance(value, dict) else None
        raise ValueError(f"{name} fields differ: {observed}")
    return value


def _require_exact_int(value: Any, expected: int, *, name: str) -> None:
    if type(value) is not int or value != expected:
        raise ValueError(f"{name} must equal integer {expected}")


def _require_exact_number(value: Any, expected: float, *, name: str) -> None:
    if type(value) is not float or not np.isfinite(value) or value != expected:
        raise ValueError(f"{name} must equal {expected}")


def _require_sha256(value: Any, *, name: str) -> str:
    if not _is_lowercase_hex(value, length=64):
        raise ValueError(f"{name} must be an exact lowercase SHA-256")
    return str(value)


def _validate_historical_receipt_schema(payload: dict[str, Any]) -> ValidatedBindingReceipt:
    """Validate the immutable historical receipt without opening any registered artifact."""
    top = _require_exact_keys(
        payload,
        {
            "schema_version",
            "diagnostic",
            "mode",
            "candidate_values_computed",
            "stage_a_verdict",
            "uses_test_data",
            "execution_audit",
            "manifest",
            "binding",
        },
        name="historical receipt",
    )
    _require_exact_int(top["schema_version"], 1, name="receipt schema version")
    required_literals = {
        "diagnostic": "pass200_rsta_stage_a",
        "mode": "binding_only",
        "stage_a_verdict": "NOT_COMPUTED",
        "uses_test_data": "artifact_binding_only",
    }
    for name, expected in required_literals.items():
        if top[name] != expected or type(top[name]) is not str:
            raise ValueError(f"receipt {name} differs")
    if top["candidate_values_computed"] is not False:
        raise ValueError("receipt candidate_values_computed must be false")

    execution = _require_exact_keys(
        top["execution_audit"], _EXECUTION_AUDIT_FIELDS, name="receipt execution audit"
    )
    expected_execution = {
        "executing_git_commit": _HISTORICAL_PRODUCER_COMMIT,
        "diagnostic_path": _DIAGNOSTIC_PATH,
        "diagnostic_sha256": _HISTORICAL_DIAGNOSTIC_SHA256,
        "frozen_source_revision": _HISTORICAL_SOURCE_REVISION,
    }
    if execution != expected_execution:
        raise ValueError("receipt execution audit provenance differs")

    manifest = _require_exact_keys(
        top["manifest"],
        {"path", "sha256", "preregistration", "artifact_schema", "source"},
        name="receipt historical manifest",
    )
    if manifest["sha256"] != _HISTORICAL_MANIFEST_SHA256:
        raise ValueError("receipt historical manifest SHA-256 differs")
    preregistration = _require_exact_keys(
        manifest["preregistration"], {"path", "sha256"}, name="receipt preregistration"
    )
    if preregistration != {
        "path": _BASE_PREREGISTRATION_PATH,
        "sha256": _BASE_PREREGISTRATION_SHA256,
    }:
        raise ValueError("receipt base preregistration differs")
    artifact_schema = _require_exact_keys(
        manifest["artifact_schema"], {"path", "sha256"}, name="receipt artifact schema"
    )
    if artifact_schema != {
        "path": "docs/pass159_stage_a_manifest.json",
        "sha256": "8323d0543979edd5331a3294601c508ea0b9787b7a8c0b111cef65562a7225da",
    }:
        raise ValueError("receipt artifact schema differs")
    historical_source = _require_exact_keys(
        manifest["source"], {"git_revision", "files"}, name="receipt historical source"
    )
    if historical_source["git_revision"] != _HISTORICAL_SOURCE_REVISION:
        raise ValueError("receipt historical source revision differs")
    source_files = _require_exact_keys(
        historical_source["files"], _FROZEN_SOURCE_FILES, name="receipt historical source files"
    )
    for path_text, digest in source_files.items():
        _require_sha256(digest, name=f"historical source {path_text}")
    if source_files.get(_DIAGNOSTIC_PATH) != _HISTORICAL_DIAGNOSTIC_SHA256:
        raise ValueError("receipt historical diagnostic SHA-256 differs")

    binding = _require_exact_keys(
        top["binding"],
        {
            "cross_seed_training_rows_identical",
            "query_gallery_released_before_scientific_input",
            "source_export_batch_size",
            "descriptor_atol",
            "descriptor_rtol",
            "seeds",
        },
        name="receipt binding",
    )
    if binding["cross_seed_training_rows_identical"] is not True:
        raise ValueError("receipt cross-seed training identity flag differs")
    if binding["query_gallery_released_before_scientific_input"] is not True:
        raise ValueError("receipt query/gallery release flag differs")
    _require_exact_int(
        binding["source_export_batch_size"], 128, name="receipt source export batch size"
    )
    _require_exact_number(binding["descriptor_atol"], 2.0e-5, name="receipt descriptor atol")
    _require_exact_number(binding["descriptor_rtol"], 2.0e-5, name="receipt descriptor rtol")
    raw_seeds = binding["seeds"]
    if not isinstance(raw_seeds, list) or len(raw_seeds) != 4:
        raise ValueError("receipt seeds must contain exactly four entries")
    seeds: list[ReceiptSeed] = []
    common_orders: tuple[str, str, str] | None = None
    for expected_seed, raw_seed in enumerate(raw_seeds):
        seed_value = _require_exact_keys(
            raw_seed,
            {
                "seed",
                "train_row_count",
                "train_identity_count",
                "train_example_id_order_sha256",
                "train_label_order_sha256",
                "train_source_order_sha256",
                "official_recall_at_1",
                "artifact_binding",
            },
            name="receipt seed",
        )
        _require_exact_int(seed_value["seed"], expected_seed, name="receipt seed")
        _require_exact_int(seed_value["train_row_count"], 25_882, name="train row count")
        _require_exact_int(
            seed_value["train_identity_count"], 3_997, name="train identity count"
        )
        if type(seed_value["official_recall_at_1"]) is not float or not np.isfinite(
            seed_value["official_recall_at_1"]
        ):
            raise ValueError("receipt official recall must be finite")
        orders = tuple(
            _require_sha256(seed_value[name], name=f"receipt {name}")
            for name in (
                "train_example_id_order_sha256",
                "train_label_order_sha256",
                "train_source_order_sha256",
            )
        )
        if common_orders is None:
            common_orders = orders
        elif orders != common_orders:
            raise ValueError("receipt cross-seed training row hashes differ")
        artifact_binding = _require_exact_keys(
            seed_value["artifact_binding"],
            {
                "artifacts",
                "current_source_export",
                "descriptor_atol",
                "descriptor_rtol",
                "official_r1_source",
                "prehead_reconstruction",
                "source_export_batch_size",
            },
            name="receipt artifact binding",
        )
        _require_exact_int(
            artifact_binding["source_export_batch_size"],
            128,
            name="receipt source export batch size",
        )
        _require_exact_number(
            artifact_binding["descriptor_atol"], 2.0e-5, name="receipt descriptor atol"
        )
        _require_exact_number(
            artifact_binding["descriptor_rtol"], 2.0e-5, name="receipt descriptor rtol"
        )
        if artifact_binding["official_r1_source"] != (
            "current_source_all_rows_and_digest_bound_final_packs"
        ):
            raise ValueError("receipt official R@1 source differs")
        artifacts = _require_exact_keys(
            artifact_binding["artifacts"], _ARTIFACT_NAMES, name="receipt artifact keys"
        )
        frozen_artifacts: dict[str, Mapping[str, str]] = {}
        for name, raw_artifact in artifacts.items():
            artifact = _require_exact_keys(
                raw_artifact, {"path", "sha256"}, name=f"receipt artifact {name}"
            )
            if type(artifact["path"]) is not str or not artifact["path"]:
                raise ValueError(f"receipt artifact {name} path differs")
            _require_sha256(artifact["sha256"], name=f"receipt artifact {name}")
            frozen_artifacts[name] = MappingProxyType(dict(artifact))
        exports = _require_exact_keys(
            artifact_binding["current_source_export"],
            set(_SOURCE_SPLITS),
            name="receipt current-source export",
        )
        for split in _SOURCE_SPLITS:
            split_value = _require_exact_keys(
                exports[split],
                {
                    "row_count",
                    "identity_count",
                    "max_abs_descriptor_difference",
                    "atol",
                    "rtol",
                    "source_export_sha256",
                },
                name=f"receipt {split} export",
            )
            rows, identities = _OFFICIAL_PARTITION[split]
            _require_exact_int(split_value["row_count"], rows, name=f"{split} row count")
            _require_exact_int(
                split_value["identity_count"], identities, name=f"{split} identity count"
            )
            _require_exact_number(
                split_value["max_abs_descriptor_difference"],
                0.0,
                name=f"{split} descriptor difference",
            )
            _require_exact_number(split_value["atol"], 2.0e-5, name=f"{split} atol")
            _require_exact_number(split_value["rtol"], 2.0e-5, name=f"{split} rtol")
            _require_sha256(
                split_value["source_export_sha256"], name=f"{split} source export"
            )
        prehead = _require_exact_keys(
            artifact_binding["prehead_reconstruction"],
            set(_SOURCE_SPLITS),
            name="receipt prehead reconstruction",
        )
        for split in _SOURCE_SPLITS:
            prehead_split = _require_exact_keys(
                prehead[split],
                {
                    "max_abs_difference",
                    "rows_above_2e_5",
                    "used_for_official_r1",
                    "within_tolerance",
                },
                name=f"receipt {split} prehead reconstruction",
            )
            if type(prehead_split["max_abs_difference"]) is not float or not np.isfinite(
                prehead_split["max_abs_difference"]
            ):
                raise ValueError(f"receipt {split} prehead difference must be finite")
            if type(prehead_split["rows_above_2e_5"]) is not int:
                raise ValueError(f"receipt {split} prehead row count must be an integer")
            if type(prehead_split["used_for_official_r1"]) is not bool or type(
                prehead_split["within_tolerance"]
            ) is not bool:
                raise ValueError(f"receipt {split} prehead flags must be booleans")
        seeds.append(
            ReceiptSeed(
                seed=expected_seed,
                artifacts=MappingProxyType(frozen_artifacts),
                official_recall_at_1=float(seed_value["official_recall_at_1"]),
                train_row_count=25_882,
                train_identity_count=3_997,
                train_example_id_order_sha256=orders[0],
                train_label_order_sha256=orders[1],
                train_source_order_sha256=orders[2],
                train_source_export_sha256=str(exports["train"]["source_export_sha256"]),
            )
        )
    return ValidatedBindingReceipt(
        sha256=_HISTORICAL_RECEIPT_SHA256,
        producer_commit=_HISTORICAL_PRODUCER_COMMIT,
        historical_manifest_sha256=_HISTORICAL_MANIFEST_SHA256,
        historical_source_revision=_HISTORICAL_SOURCE_REVISION,
        historical_diagnostic_sha256=_HISTORICAL_DIAGNOSTIC_SHA256,
        seeds=tuple(seeds),
    )


def _validate_amended_manifest_schema(manifest: dict[str, Any]) -> dict[str, Any]:
    value = _require_exact_keys(
        manifest,
        {
            "schema_version",
            "base_preregistration",
            "amendment",
            "deterministic_pool_amendment",
            "zero_jacobian_classifier_amendment",
            "binding_receipt",
            "historical",
            "current_scientific_source",
            "artifact_schema",
            "seeds",
        },
        name="amended RSTA manifest",
    )
    if value["schema_version"] != "pass200-rsta-receipt-manifest-v1":
        raise ValueError("amended RSTA manifest schema version differs")
    expected_references = {
        "base_preregistration": {
            "path": _BASE_PREREGISTRATION_PATH,
            "sha256": _BASE_PREREGISTRATION_SHA256,
        },
        "binding_receipt": {
            "path": _HISTORICAL_RECEIPT_PATH,
            "sha256": _HISTORICAL_RECEIPT_SHA256,
        },
        "artifact_schema": {
            "path": "docs/pass159_stage_a_manifest.json",
            "sha256": "8323d0543979edd5331a3294601c508ea0b9787b7a8c0b111cef65562a7225da",
        },
    }
    for name, expected in expected_references.items():
        reference = _require_exact_keys(value[name], {"path", "sha256"}, name=name)
        if reference != expected:
            raise ValueError(f"amended RSTA manifest {name} differs")
    amendment = _require_exact_keys(
        value["amendment"], {"path", "sha256", "commit"}, name="amendment"
    )
    if amendment != {
        "path": _AMENDMENT_PATH,
        "sha256": _AMENDMENT_SHA256,
        "commit": _AMENDMENT_COMMIT,
    }:
        raise ValueError("amended RSTA manifest amendment differs")
    deterministic_pool_amendment = _require_exact_keys(
        value["deterministic_pool_amendment"],
        {"path", "sha256", "commit"},
        name="deterministic_pool_amendment",
    )
    if deterministic_pool_amendment != {
        "path": _DETERMINISTIC_POOL_AMENDMENT_PATH,
        "sha256": _DETERMINISTIC_POOL_AMENDMENT_SHA256,
        "commit": _DETERMINISTIC_POOL_AMENDMENT_COMMIT,
    }:
        raise ValueError("amended RSTA manifest deterministic_pool_amendment differs")
    zero_jacobian_amendment = _require_exact_keys(
        value["zero_jacobian_classifier_amendment"],
        {"path", "sha256", "commit"},
        name="zero_jacobian_classifier_amendment",
    )
    if zero_jacobian_amendment != {
        "path": _ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_PATH,
        "sha256": _ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_SHA256,
        "commit": _ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_COMMIT,
    }:
        raise ValueError(
            "amended RSTA manifest zero_jacobian_classifier_amendment differs"
        )
    historical = _require_exact_keys(
        value["historical"], {"producer_commit", "manifest", "source"}, name="historical"
    )
    if historical["producer_commit"] != _HISTORICAL_PRODUCER_COMMIT:
        raise ValueError("historical producer commit differs")
    historical_manifest = _require_exact_keys(
        historical["manifest"], {"path", "sha256"}, name="historical manifest"
    )
    if historical_manifest != {
        "path": _HISTORICAL_MANIFEST_PATH,
        "sha256": _HISTORICAL_MANIFEST_SHA256,
    }:
        raise ValueError("historical manifest reference differs")
    historical_source = _require_exact_keys(
        historical["source"], {"git_revision", "files"}, name="historical source"
    )
    if historical_source["git_revision"] != _HISTORICAL_SOURCE_REVISION:
        raise ValueError("historical source revision differs")
    historical_files = _require_exact_keys(
        historical_source["files"], _FROZEN_SOURCE_FILES, name="historical source files"
    )
    for path_text, digest in historical_files.items():
        _require_sha256(digest, name=f"historical source {path_text}")
    current = _require_exact_keys(
        value["current_scientific_source"],
        {"git_revision", "files"},
        name="current scientific source",
    )
    revision = current["git_revision"]
    if not _is_lowercase_hex(revision, length=40):
        raise ValueError("current scientific source revision must be a full commit")
    current_files = _require_exact_keys(
        current["files"], _CURRENT_SCIENTIFIC_SOURCE_FILES, name="current scientific source files"
    )
    for path_text, digest in current_files.items():
        _require_sha256(digest, name=f"current scientific source {path_text}")
    seeds = value["seeds"]
    if not isinstance(seeds, dict) or list(seeds) != ["0", "1", "2", "3"]:
        raise ValueError("amended RSTA manifest requires ordered seeds 0-3")
    for seed, entry in seeds.items():
        artifacts = _require_exact_keys(entry, _ARTIFACT_NAMES, name=f"manifest seed {seed}")
        for name, artifact in artifacts.items():
            record = _require_exact_keys(
                artifact, {"path", "sha256"}, name=f"manifest seed {seed} artifact {name}"
            )
            if type(record["path"]) is not str or not record["path"]:
                raise ValueError(f"manifest seed {seed} artifact {name} path differs")
            _require_sha256(record["sha256"], name=f"manifest seed {seed} artifact {name}")
    return value


def _git_blob(repository: Path, revision: str, path_text: str) -> bytes:
    result = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "blob", f"{revision}:{path_text}"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"Git provenance lacks blob {revision}:{path_text}")
    return result.stdout


def validate_historical_binding_receipt(
    manifest_path: Path,
    receipt_path: Path,
) -> ValidatedBindingReceipt:
    """Authenticate the sole historical receipt and its independent Git provenance."""
    repository = manifest_path.resolve().parent.parent
    literal_receipt = (repository / _HISTORICAL_RECEIPT_PATH).resolve()
    if receipt_path.resolve() != literal_receipt:
        raise ValueError("binding receipt must use the literal historical receipt path")
    if not literal_receipt.is_file():
        raise ValueError("literal historical receipt is missing")
    observed_receipt_digest = sha256_file(literal_receipt)
    if observed_receipt_digest != _HISTORICAL_RECEIPT_SHA256:
        raise ValueError("literal historical receipt SHA-256 mismatch")
    manifest = _validate_amended_manifest_schema(load_strict_json(manifest_path))
    if manifest["binding_receipt"] != {
        "path": _HISTORICAL_RECEIPT_PATH,
        "sha256": _HISTORICAL_RECEIPT_SHA256,
    }:
        raise ValueError("manifest binding receipt differs from literal authority")
    receipt_payload = load_strict_json(literal_receipt)
    receipt = _validate_historical_receipt_schema(receipt_payload)
    if manifest["historical"]["source"] != receipt_payload["manifest"]["source"]:
        raise ValueError("manifest historical source differs from receipt")
    for receipt_seed in receipt.seeds:
        if manifest["seeds"][str(receipt_seed.seed)] != {
            name: dict(record) for name, record in receipt_seed.artifacts.items()
        }:
            raise ValueError(f"manifest seed {receipt_seed.seed} artifacts differ from receipt")
    historical_manifest_blob = _git_blob(
        repository, _HISTORICAL_PRODUCER_COMMIT, _HISTORICAL_MANIFEST_PATH
    )
    if hashlib.sha256(historical_manifest_blob).hexdigest() != _HISTORICAL_MANIFEST_SHA256:
        raise ValueError("historical manifest Git blob SHA-256 mismatch")
    historical_manifest = _load_strict_json_bytes(
        historical_manifest_blob,
        name=f"{_HISTORICAL_PRODUCER_COMMIT}:{_HISTORICAL_MANIFEST_PATH}",
    )
    historical_manifest = _require_exact_keys(
        historical_manifest,
        {"schema_version", "preregistration", "artifact_schema", "source", "seeds"},
        name="historical manifest blob",
    )
    _require_exact_int(
        historical_manifest["schema_version"], 1, name="historical manifest schema version"
    )
    if historical_manifest["preregistration"] != manifest["base_preregistration"]:
        raise ValueError("historical manifest preregistration differs")
    if historical_manifest["artifact_schema"] != manifest["artifact_schema"]:
        raise ValueError("historical manifest artifact schema differs")
    if historical_manifest["source"] != manifest["historical"]["source"]:
        raise ValueError("historical manifest source differs")
    if historical_manifest["seeds"] != manifest["seeds"]:
        raise ValueError("historical manifest seed artifacts differ")
    base_blob = _git_blob(
        repository, _HISTORICAL_PRODUCER_COMMIT, _BASE_PREREGISTRATION_PATH
    )
    if hashlib.sha256(base_blob).hexdigest() != _BASE_PREREGISTRATION_SHA256:
        raise ValueError("historical preregistration Git blob SHA-256 mismatch")
    diagnostic_blob = _git_blob(repository, _HISTORICAL_PRODUCER_COMMIT, _DIAGNOSTIC_PATH)
    if hashlib.sha256(diagnostic_blob).hexdigest() != _HISTORICAL_DIAGNOSTIC_SHA256:
        raise ValueError("historical producer diagnostic Git blob SHA-256 mismatch")
    for path_text, expected_digest in manifest["historical"]["source"]["files"].items():
        blob = _git_blob(repository, _HISTORICAL_SOURCE_REVISION, path_text)
        if hashlib.sha256(blob).hexdigest() != expected_digest:
            raise ValueError(f"historical source Git blob SHA-256 mismatch for {path_text}")
    return receipt


def validate_scientific_execution_source(manifest_path: Path) -> dict[str, Any]:
    """Validate the current source domain separately from historical receipt provenance."""
    manifest = _validate_amended_manifest_schema(load_strict_json(manifest_path))
    repository = manifest_path.resolve().parent.parent
    for name in (
        "base_preregistration",
        "amendment",
        "deterministic_pool_amendment",
        "zero_jacobian_classifier_amendment",
        "artifact_schema",
    ):
        reference = manifest[name]
        path = (repository / reference["path"]).resolve()
        if not path.is_file() or sha256_file(path) != reference["sha256"]:
            raise ValueError(f"current scientific {name} worktree SHA-256 differs")
    amendment_blob = _git_blob(repository, _AMENDMENT_COMMIT, _AMENDMENT_PATH)
    if hashlib.sha256(amendment_blob).hexdigest() != _AMENDMENT_SHA256:
        raise ValueError("prospective amendment Git blob SHA-256 mismatch")
    pool_amendment_blob = _git_blob(
        repository,
        _DETERMINISTIC_POOL_AMENDMENT_COMMIT,
        _DETERMINISTIC_POOL_AMENDMENT_PATH,
    )
    if (
        hashlib.sha256(pool_amendment_blob).hexdigest()
        != _DETERMINISTIC_POOL_AMENDMENT_SHA256
    ):
        raise ValueError("deterministic pool amendment Git blob SHA-256 mismatch")
    zero_jacobian_blob = _git_blob(
        repository,
        _ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_COMMIT,
        _ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_PATH,
    )
    if (
        hashlib.sha256(zero_jacobian_blob).hexdigest()
        != _ZERO_JACOBIAN_CLASSIFIER_AMENDMENT_SHA256
    ):
        raise ValueError("zero-Jacobian classifier amendment Git blob SHA-256 mismatch")
    current = manifest["current_scientific_source"]
    revision = current["git_revision"]
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"],
        capture_output=True,
        check=False,
        text=True,
    )
    executing_commit = head.stdout.strip()
    if head.returncode != 0 or not _is_lowercase_hex(executing_commit, length=40):
        raise ValueError("current scientific repository HEAD does not resolve")
    ancestry = subprocess.run(
        ["git", "-C", str(repository), "merge-base", "--is-ancestor", revision, executing_commit],
        capture_output=True,
        check=False,
    )
    if ancestry.returncode != 0:
        raise ValueError("current scientific source revision is not an ancestor of HEAD")
    for path_text, expected_digest in current["files"].items():
        blob = _git_blob(repository, revision, path_text)
        if hashlib.sha256(blob).hexdigest() != expected_digest:
            raise ValueError(f"current scientific source Git blob differs for {path_text}")
        path = (repository / path_text).resolve()
        if not path.is_file() or sha256_file(path) != expected_digest:
            raise ValueError(f"current scientific source worktree differs for {path_text}")
    executing_diagnostic = Path(__file__).resolve()
    if executing_diagnostic != (repository / _DIAGNOSTIC_PATH).resolve():
        raise ValueError("current scientific executing diagnostic path differs")
    return {
        "executing_git_commit": executing_commit,
        "diagnostic_path": _DIAGNOSTIC_PATH,
        "diagnostic_sha256": current["files"][_DIAGNOSTIC_PATH],
        "frozen_source_revision": revision,
    }


@dataclass(frozen=True)
class TrainingOnlySeedInput:
    """Digest/source-bound seed state with binding-only splits permanently absent."""

    seed: int
    train_embeddings: np.ndarray
    train_labels: np.ndarray
    train_example_ids: np.ndarray
    train_source_paths: np.ndarray
    train_row_indices: np.ndarray
    proxies: np.ndarray
    proxy_labels: np.ndarray
    alpha: float
    delta: float
    official_recall_at_1: float
    checkpoint_bytes: bytes
    checkpoint_sha256: str
    training_array_sha256: Mapping[str, str]
    config: Mapping[str, Any]
    artifact_binding: Mapping[str, Any]


@dataclass(frozen=True)
class ReceiptSeed:
    """Immutable scalar and digest authority for one historical seed."""

    seed: int
    artifacts: Mapping[str, Mapping[str, str]]
    official_recall_at_1: float
    train_row_count: int
    train_identity_count: int
    train_example_id_order_sha256: str
    train_label_order_sha256: str
    train_source_order_sha256: str
    train_source_export_sha256: str


@dataclass(frozen=True)
class ValidatedBindingReceipt:
    """Hash-only historical authority; never contains tensors or descriptor arrays."""

    sha256: str
    producer_commit: str
    historical_manifest_sha256: str
    historical_source_revision: str
    historical_diagnostic_sha256: str
    seeds: tuple[ReceiptSeed, ...]


@dataclass(frozen=True)
class DeterministicTransformCache:
    """One immutable lookup of exactly-once, per-example diagnostic augmentations."""

    example_ids: tuple[str, ...]
    tensors: Mapping[str, Any]
    tensor_sha256: Mapping[str, str]
    ordered_id_sha256: str

    def batch(self, ordered_ids: Sequence[str]) -> Any:
        import torch

        ids = tuple(str(value) for value in ordered_ids)
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("cached batch IDs must be nonempty and unique")
        missing = [value for value in ids if value not in self.tensors]
        if missing:
            raise ValueError(f"cached batch references unknown IDs: {missing}")
        for value in ids:
            observed = _tensor_sha256(self.tensors[value])
            if observed != self.tensor_sha256[value]:
                raise ValueError(
                    f"cached tensor SHA-256 mismatch for {value}: "
                    f"{observed} != {self.tensor_sha256[value]}"
                )
        return torch.stack([self.tensors[value] for value in ids], dim=0)


def _tensor_sha256(tensor: Any) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def cache_deterministic_transforms(
    example_ids: Sequence[str],
    sources_by_id: Mapping[str, Any],
    *,
    transform: Callable[[Any], Any],
    materialize: Callable[[Any], Any] | None = None,
) -> DeterministicTransformCache:
    """Apply the official transform once per row under isolated global RNG states."""
    import torch

    ids = tuple(str(value) for value in example_ids)
    if not ids or any(not value for value in ids) or len(set(ids)) != len(ids):
        raise ValueError("transform-cache example IDs must be nonempty and unique")
    if set(sources_by_id) != set(ids):
        raise ValueError("transform-cache sources must match the ordered example IDs exactly")
    load = (lambda value: value) if materialize is None else materialize
    tensors: dict[str, Any] = {}
    hashes: dict[str, str] = {}
    for example_id in ids:
        source = load(sources_by_id[example_id])
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.get_rng_state().clone()
        try:
            random.seed(domain_seed("rsta-stage-a-v1|augment-python|", example_id))
            np.random.seed(domain_seed("rsta-stage-a-v1|augment-numpy|", example_id) % (2**32))
            torch.manual_seed(domain_seed("rsta-stage-a-v1|augment-torch|", example_id))
            transformed = transform(source)
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.set_rng_state(torch_state)
        if not isinstance(transformed, torch.Tensor):
            raise ValueError("official transform must return a torch tensor")
        cached = transformed.detach().cpu().contiguous().clone()
        if cached.numel() == 0 or not bool(torch.isfinite(cached).all()):
            raise ValueError("official transform returned an empty or nonfinite tensor")
        tensors[example_id] = cached
        hashes[example_id] = _tensor_sha256(cached)
    return DeterministicTransformCache(
        example_ids=ids,
        tensors=MappingProxyType(tensors),
        tensor_sha256=MappingProxyType(hashes),
        ordered_id_sha256=hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
    )


def cache_seed_training_tensors(
    bound: TrainingOnlySeedInput,
    ordered_ids: Sequence[str],
    *,
    transform: Callable[[Any], Any] | None = None,
    materialize: Callable[[Any], Any] | None = None,
) -> DeterministicTransformCache:
    """Cache selected training rows through the current official training transform."""
    validate_retained_training_arrays(bound)
    ids = tuple(str(value) for value in ordered_ids)
    source_by_id = dict(zip(bound.train_example_ids, bound.train_source_paths, strict=True))
    unknown = [value for value in ids if value not in source_by_id]
    if unknown:
        raise ValueError(f"unknown training example IDs requested for caching: {unknown}")
    official_transform = transform
    if official_transform is None:
        from sfora.image_end_to_end import ImageEndToEndConfig, _default_transform_factory

        validated = ImageEndToEndConfig.model_validate(_json_ready(bound.config))
        official_transform = _default_transform_factory(validated, True)
    image_materializer = materialize
    if image_materializer is None:
        from sfora.data import materialize_image

        def materialize_bound_path(value: Any) -> Any:
            return materialize_image(Path(str(value)))

        image_materializer = materialize_bound_path
    selected_sources = {value: source_by_id[value] for value in ids}
    return cache_deterministic_transforms(
        ids,
        selected_sources,
        transform=official_transform,
        materialize=image_materializer,
    )


def _validate_rsta_config(
    config: dict[str, Any],
    report: dict[str, Any],
    *,
    seed: int,
    expected_dimension: int,
) -> None:
    required = {
        "dataset_name": "inshop",
        "objectives": ["proxy_anchor"],
        "seed": int(seed),
        "proxy_anchor_alpha": 32.0,
        "proxy_anchor_delta": 0.1,
        "checkpoint_selection_interval": 0,
        "backbone_name": "bn_inception",
        "head_pooling": "avg_max",
        "batch_size": 180,
        "drop_last_train_batch": True,
        "freeze_batch_norm": False,
        "freeze_batch_norm_affine": False,
        "embedding_dimensions": int(expected_dimension),
    }
    for name, expected in required.items():
        if config.get(name) != expected:
            raise ValueError(f"report config {name}={config.get(name)!r} != {expected!r}")
    methods = report.get("methods")
    if not isinstance(methods, dict) or len(methods) != 1:
        raise ValueError("report must contain exactly one method")
    method = next(iter(methods.values()))
    if not isinstance(method, dict) or method.get("dimensions") != expected_dimension:
        raise ValueError(f"report method dimension must equal {expected_dimension}")


def _load_digest_bound_packs(
    paths: dict[str, Path],
    *,
    checkpoint_digest: str,
    report_digest: str,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        split: _load_final_pack(
            paths[f"{split}_npz"],
            split=split,
            checkpoint_digest=checkpoint_digest,
            report_digest=report_digest,
        )
        for split in _SOURCE_SPLITS
    }


def _source_export_hash(split: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in ("row_indices", "labels", "example_ids", "source_paths", "embeddings"):
        value = np.asarray(split[name])
        digest.update(name.encode("ascii") + b"\0")
        if name in {"example_ids", "source_paths"}:
            for item in value.astype(str).tolist():
                digest.update(item.encode("utf-8") + b"\0")
        else:
            contiguous = np.ascontiguousarray(value)
            digest.update(contiguous.dtype.str.encode("ascii") + b"\0")
            digest.update(str(contiguous.shape).encode("ascii") + b"\0")
            digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _validate_source_export(
    exported: dict[str, dict[str, np.ndarray]],
    packs: dict[str, dict[str, np.ndarray]],
    *,
    expected_partition: dict[str, tuple[int, int]],
    expected_dimension: int,
) -> dict[str, dict[str, Any]]:
    if set(exported) != set(_SOURCE_SPLITS):
        raise ValueError("current-source export must contain train, query, and gallery")
    checks: dict[str, dict[str, Any]] = {}
    seen_ids: list[set[str]] = []
    seen_sources: list[set[str]] = []
    for split in _SOURCE_SPLITS:
        current = exported[split]
        if set(current) != _SOURCE_EXPORT_KEYS:
            raise ValueError(f"{split} current-source export keys differ from schema")
        embeddings = np.asarray(current["embeddings"], dtype=np.float32)
        raw_labels = np.asarray(current["labels"])
        if not np.issubdtype(raw_labels.dtype, np.integer):
            raise ValueError(f"{split} current-source labels must use an integral dtype")
        if raw_labels.size and (
            np.any(raw_labels < 0) or int(raw_labels.max()) > int(np.iinfo(np.int64).max)
        ):
            raise ValueError(f"{split} current-source labels must be exact unsigned int64 values")
        labels = raw_labels.astype(np.int64, copy=False)
        ids = np.asarray(current["example_ids"]).astype(str)
        sources = np.asarray(current["source_paths"]).astype(str)
        indices = np.asarray(current["row_indices"])
        row_count, identity_count = expected_partition[split]
        if embeddings.shape != (row_count, expected_dimension):
            raise ValueError(f"{split} current-source descriptor shape differs")
        if labels.shape != (row_count,) or len(np.unique(labels)) != identity_count:
            raise ValueError(f"{split} current-source label partition differs")
        if ids.shape != (row_count,) or sources.shape != (row_count,):
            raise ValueError(f"{split} current-source row metadata shape differs")
        if indices.shape != (row_count,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{split} current-source row indices must be integral")
        if not np.array_equal(indices.astype(np.int64), np.arange(row_count, dtype=np.int64)):
            raise ValueError(f"{split} current-source row indices are duplicate or out of order")
        if len(set(ids.tolist())) != row_count:
            raise ValueError(f"{split} current-source example IDs contain duplicates")
        if len(set(sources.tolist())) != row_count:
            raise ValueError(f"{split} current-source paths contain duplicates")
        if not np.isfinite(embeddings).all():
            raise ValueError(f"{split} current-source descriptors are nonfinite")
        norms = np.linalg.norm(embeddings, axis=1)
        if np.any(np.abs(norms - 1.0) > 2.0e-5):
            raise ValueError(f"{split} current-source descriptors are not unit rows")
        pack_ids = np.asarray(packs[split]["example_ids"]).astype(str)
        pack_labels = np.asarray(packs[split]["labels"], dtype=np.int64)
        pack_sources = np.asarray(packs[split]["source_paths"]).astype(str)
        pack_embeddings = np.asarray(packs[split]["embeddings"], dtype=np.float32)
        if not np.array_equal(ids, pack_ids):
            raise ValueError(f"{split} current-source example-ID order differs from final pack")
        if not np.array_equal(labels, pack_labels):
            raise ValueError(f"{split} current-source label order differs from final pack")
        if not np.array_equal(sources, pack_sources):
            raise ValueError(f"{split} current-source source-path order differs from final pack")
        if not np.allclose(embeddings, pack_embeddings, atol=2.0e-5, rtol=2.0e-5):
            raise ValueError(f"{split} current-source descriptors differ from final pack")
        difference = np.abs(embeddings - pack_embeddings)
        checks[split] = {
            "row_count": row_count,
            "identity_count": identity_count,
            "max_abs_descriptor_difference": float(difference.max(initial=0.0)),
            "atol": 2.0e-5,
            "rtol": 2.0e-5,
            "source_export_sha256": _source_export_hash(current),
        }
        seen_ids.append(set(ids.tolist()))
        seen_sources.append(set(sources.tolist()))
    for left, right in ((0, 1), (0, 2), (1, 2)):
        if seen_ids[left] & seen_ids[right]:
            raise ValueError("current-source example IDs overlap across official splits")
        if seen_sources[left] & seen_sources[right]:
            raise ValueError("current-source paths overlap across official splits")
    return checks


def _export_current_source(
    *,
    paths: dict[str, Path],
    config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, dict[str, np.ndarray]]:
    """Re-export every official row using the diagnostic's current production source."""
    import torch
    from torch.utils.data import DataLoader

    from sfora.data import load_image_retrieval_bundle
    from sfora.image_end_to_end import (
        ImageEndToEndConfig,
        _default_transform_factory,
        _encode_model,
        _TorchImageDataset,
        _torchvision_model_factory,
    )

    validated = ImageEndToEndConfig.model_validate(config)
    bundle = load_image_retrieval_bundle(
        dataset_name="inshop",
        dataset_root=validated.dataset_root,
        seed=validated.seed,
    )
    if bundle.gallery is None:
        raise ValueError("current In-Shop source lacks an official gallery")
    examples_by_split = {
        "train": bundle.train,
        "query": bundle.query,
        "gallery": bundle.gallery,
    }
    transform = _default_transform_factory(validated, False)

    def loader(examples: list[Any]) -> Any:
        return DataLoader(
            _TorchImageDataset(examples, transform),
            batch_size=128,
            shuffle=False,
            num_workers=4,
            pin_memory=torch.cuda.is_available(),
        )

    model: Any = _torchvision_model_factory(validated)
    state = {
        name: value
        for name, value in checkpoint["state_dict"].items()
        if name not in {"metric_proxies", "metric_proxy_labels"}
    }
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    result: dict[str, dict[str, np.ndarray]] = {}
    for split in _SOURCE_SPLITS:
        examples = examples_by_split[split]
        embeddings, labels = _encode_model(model, loader(examples), device, torch)
        result[split] = {
            "embeddings": np.asarray(embeddings, dtype=np.float32),
            "labels": np.asarray(labels, dtype=np.int64),
            "example_ids": np.asarray([str(example.example_id) for example in examples]),
            "source_paths": np.asarray([str(Path(example.image)) for example in examples]),
            "row_indices": np.arange(len(examples), dtype=np.int64),
        }
    return result


def load_and_bind_seed(
    entry: dict[str, dict[str, str]],
    *,
    seed: int,
    source_exporter: Callable[..., dict[str, dict[str, np.ndarray]]] | None = None,
    expected_partition: dict[str, tuple[int, int]] | None = None,
    expected_dimension: int = 512,
) -> TrainingOnlySeedInput:
    """Bind immutable artifacts and current source before exposing training-only state."""
    import torch

    expected = _OFFICIAL_PARTITION if expected_partition is None else expected_partition
    paths = _manifest_paths(entry)
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    config = report.get("config")
    if not isinstance(config, dict):
        raise ValueError("report lacks a config object")
    _validate_rsta_config(config, report, seed=seed, expected_dimension=expected_dimension)
    checkpoint = torch.load(paths["checkpoint_pt"], map_location="cpu", weights_only=False)
    if checkpoint.get("evaluation_model_source") != "student":
        raise ValueError("checkpoint evaluation_model_source is not student")
    bound = load_bound_seed(entry, seed=seed, expected_partition=expected)
    if bound.train_embeddings.shape[1] != expected_dimension:
        raise ValueError(f"train descriptor dimension must equal {expected_dimension}")
    train_identity_labels = set(int(value) for value in bound.train_labels.tolist())
    proxy_labels = tuple(int(value) for value in bound.proxy_labels.tolist())
    if (
        len(proxy_labels) != len(train_identity_labels)
        or set(proxy_labels) != train_identity_labels
    ):
        raise ValueError("checkpoint must contain exactly one proxy for every train identity")
    if bound.proxies.shape != (len(proxy_labels), expected_dimension):
        raise ValueError(f"proxy descriptor dimension must equal {expected_dimension}")
    packs = _load_digest_bound_packs(
        paths,
        checkpoint_digest=entry["checkpoint_pt"]["sha256"],
        report_digest=entry["report_json"]["sha256"],
    )
    exporter = _export_current_source if source_exporter is None else source_exporter
    exported = exporter(paths=paths, config=config, checkpoint=checkpoint)
    source_checks = _validate_source_export(
        exported,
        packs,
        expected_partition=expected,
        expected_dimension=expected_dimension,
    )
    source_r1 = _canonical_query_gallery_recall_at_1(
        exported["query"]["embeddings"],
        exported["query"]["labels"],
        exported["gallery"]["embeddings"],
        exported["gallery"]["labels"],
    )
    if source_r1 != bound.official_recall_at_1:
        raise ValueError(
            f"official R@1 mismatch: current-source={source_r1}, bound={bound.official_recall_at_1}"
        )
    train = exported.pop("train")
    del exported["query"], exported["gallery"]
    del exported, packs
    train_embeddings = _readonly_array(train["embeddings"], dtype=np.float32)
    proxies = _readonly_array(bound.proxies, dtype=np.float32)
    metadata = {
        **bound.artifact_binding,
        "current_source_export": source_checks,
        "official_r1_source": "current_source_all_rows_and_digest_bound_final_packs",
        "source_export_batch_size": 128,
        "descriptor_atol": 2.0e-5,
        "descriptor_rtol": 2.0e-5,
    }
    immutable_arrays = {
        "train_embeddings": train_embeddings,
        "train_labels": _readonly_array(train["labels"], dtype=np.int64),
        "train_example_ids": _readonly_array(np.asarray(train["example_ids"]).astype(str)),
        "train_source_paths": _readonly_array(np.asarray(train["source_paths"]).astype(str)),
        "train_row_indices": _readonly_array(train["row_indices"], dtype=np.int64),
        "proxies": proxies,
        "proxy_labels": _readonly_array(proxy_labels, dtype=np.int64),
    }
    return TrainingOnlySeedInput(
        seed=int(seed),
        **immutable_arrays,
        alpha=float(bound.alpha),
        delta=float(bound.delta),
        official_recall_at_1=float(bound.official_recall_at_1),
        checkpoint_bytes=paths["checkpoint_pt"].read_bytes(),
        checkpoint_sha256=entry["checkpoint_pt"]["sha256"],
        training_array_sha256=MappingProxyType(
            {
                name: _framed_array_sha256(name, value)
                for name, value in immutable_arrays.items()
            }
        ),
        config=_deep_freeze(config),
        artifact_binding=_deep_freeze(metadata),
    )


def _framed_array_sha256(name: str, value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(name.encode("ascii") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(str(array.shape).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def validate_retained_training_arrays(bound: TrainingOnlySeedInput) -> None:
    """Re-authenticate immutable retained arrays immediately before scientific use."""
    hashes = bound.training_array_sha256
    expected_names = {
        "train_embeddings",
        "train_labels",
        "train_example_ids",
        "train_source_paths",
        "train_row_indices",
        "proxies",
        "proxy_labels",
    }
    if not isinstance(hashes, Mapping) or set(hashes) != expected_names:
        raise ValueError("retained training array SHA-256 registry is absent")
    for name, expected in hashes.items():
        value = getattr(bound, name, None)
        if not isinstance(value, np.ndarray) or value.flags.writeable:
            raise ValueError(f"retained training array SHA-256 boundary failed for {name}")
        if _framed_array_sha256(name, value) != expected:
            raise ValueError(f"retained training array SHA-256 mismatch for {name}")


def _np_scalar(value: Any, *, name: str) -> Any:
    array = np.asarray(value)
    if array.shape != ():
        raise ValueError(f"{name} must be a scalar")
    return array.item()


def _validate_checkpoint_mapping(
    checkpoint: Any,
    *,
    config: dict[str, Any],
    train_labels: np.ndarray,
    expected_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(checkpoint, dict):
        raise ValueError("checkpoint root must be a mapping")
    if checkpoint.get("artifact_selection") != "final_training_state":
        raise ValueError("checkpoint is not a final training state")
    if checkpoint.get("evaluation_model_source") != "student":
        raise ValueError("checkpoint evaluation_model_source is not student")
    if checkpoint.get("training_config") != config:
        raise ValueError("checkpoint training_config differs from report config")
    state = checkpoint.get("state_dict")
    if not isinstance(state, dict):
        raise ValueError("checkpoint lacks a state_dict")
    required = {
        "model.embedding.weight",
        "model.embedding.bias",
        "metric_proxies",
        "metric_proxy_labels",
    }
    if not required.issubset(state):
        raise ValueError("checkpoint lacks embedding head or proxy tensors")
    proxies = np.asarray(state["metric_proxies"].detach().cpu().numpy(), dtype=np.float32)
    raw_proxy_labels = np.asarray(state["metric_proxy_labels"].detach().cpu().numpy())
    if not np.issubdtype(raw_proxy_labels.dtype, np.integer):
        raise ValueError("checkpoint proxy labels must be integral")
    proxy_labels = raw_proxy_labels.astype(np.int64, copy=False)
    identity_labels = set(int(value) for value in train_labels.tolist())
    if (
        proxies.shape != (len(identity_labels), expected_dimension)
        or proxy_labels.shape != (len(identity_labels),)
        or len(np.unique(proxy_labels)) != len(identity_labels)
        or set(int(value) for value in proxy_labels.tolist()) != identity_labels
    ):
        raise ValueError("checkpoint must contain exactly one aligned proxy per train identity")
    if not np.isfinite(proxies).all():
        raise ValueError("checkpoint proxies are nonfinite")
    norms = np.linalg.norm(proxies, axis=1)
    if np.any(norms <= _VECTOR_EPS):
        raise ValueError("checkpoint proxies contain a zero row")
    proxies = proxies / norms[:, None]
    return proxies, proxy_labels


def load_training_only_seed(
    entry: dict[str, Any],
    receipt_seed: ReceiptSeed,
    *,
    artifact_hasher: Callable[[Path], str] = sha256_file,
    checkpoint_loader: Callable[[io.BytesIO], Any] | None = None,
    expected_partition: dict[str, tuple[int, int]] | None = None,
    expected_dimension: int = 512,
) -> TrainingOnlySeedInput:
    """Load only receipt-bound training state after hashing every immutable artifact."""
    if receipt_seed.seed < 0 or receipt_seed.seed > 3:
        raise ValueError("receipt seed is outside the frozen set")
    if not isinstance(entry, dict) or set(entry) != _ARTIFACT_NAMES:
        raise ValueError("training-only manifest artifact keys differ")
    if {
        name: dict(record) for name, record in receipt_seed.artifacts.items()
    } != entry:
        raise ValueError("training-only manifest artifacts differ from receipt")
    paths: dict[str, Path] = {}
    for name in sorted(_ARTIFACT_NAMES):
        record = _require_exact_keys(entry[name], {"path", "sha256"}, name=f"artifact {name}")
        path = Path(record["path"])
        observed = artifact_hasher(path)
        if observed != record["sha256"]:
            raise ValueError(f"artifact SHA-256 mismatch for {name}")
        paths[name] = path

    captured: dict[str, bytes] = {}
    for name in ("report_json", "retrieval_json", "checkpoint_pt", "train_npz"):
        data = paths[name].read_bytes()
        if hashlib.sha256(data).hexdigest() != entry[name]["sha256"]:
            raise ValueError(f"captured artifact SHA-256 mismatch for {name}")
        captured[name] = data
    report = _load_strict_json_bytes(captured["report_json"], name="receipt-bound report")
    config = report.get("config")
    if not isinstance(config, dict):
        raise ValueError("report lacks a config object")
    _validate_rsta_config(
        config,
        report,
        seed=receipt_seed.seed,
        expected_dimension=expected_dimension,
    )
    retrieval = _load_strict_json_bytes(
        captured["retrieval_json"], name="receipt-bound retrieval audit"
    )
    if retrieval.get("artifact_selection") != "final_training_state":
        raise ValueError("retrieval audit is not a final training state")
    if (
        retrieval.get("checkpoint_sha256") != entry["checkpoint_pt"]["sha256"]
        or retrieval.get("report_sha256") != entry["report_json"]["sha256"]
    ):
        raise ValueError("retrieval audit artifact digests differ")
    methods = report.get("methods")
    method = (
        next(iter(methods.values()))
        if isinstance(methods, dict) and len(methods) == 1
        else None
    )
    if not isinstance(method, dict):
        raise ValueError("report must contain exactly one method")
    recall_values = (
        method.get("recall_at_1"),
        retrieval.get("reported_final_recall_at_1"),
        retrieval.get("independent_recall_at_1"),
        retrieval.get("canonical_float64_euclidean_recall_at_1"),
    )
    if any(
        type(value) is not float or value != receipt_seed.official_recall_at_1
        for value in recall_values
    ):
        raise ValueError("report/retrieval recall differs from receipt")

    with np.load(io.BytesIO(captured["train_npz"]), allow_pickle=False) as archive:
        required = {
            "embeddings",
            "labels",
            "example_ids",
            "source_paths",
            "artifact_selection",
            "split",
            "checkpoint_sha256",
            "report_sha256",
        }
        if set(archive.files) != required:
            raise ValueError("train final-pack keys differ from the frozen schema")
        pack = {name: np.asarray(archive[name]) for name in archive.files}
    if (
        _np_scalar(pack["artifact_selection"], name="train artifact_selection")
        != "final_training_state"
    ):
        raise ValueError("train final pack is not a final training state")
    if _np_scalar(pack["split"], name="train split") != "train":
        raise ValueError("train final pack has the wrong split marker")
    if _np_scalar(pack["checkpoint_sha256"], name="train checkpoint_sha256") != entry[
        "checkpoint_pt"
    ]["sha256"]:
        raise ValueError("train final pack checkpoint digest differs")
    if _np_scalar(pack["report_sha256"], name="train report_sha256") != entry["report_json"][
        "sha256"
    ]:
        raise ValueError("train final pack report digest differs")
    raw_labels = np.asarray(pack["labels"])
    if not np.issubdtype(raw_labels.dtype, np.integer):
        raise ValueError("train labels must use an integral dtype")
    if raw_labels.size and (
        np.any(raw_labels < 0) or int(raw_labels.max()) > int(np.iinfo(np.int64).max)
    ):
        raise ValueError("train labels must be exact unsigned int64 values")
    labels = raw_labels.astype(np.int64, copy=False)
    embeddings = np.asarray(pack["embeddings"], dtype=np.float32)
    example_ids = np.asarray(pack["example_ids"]).astype(str)
    source_paths = np.asarray(pack["source_paths"]).astype(str)
    expected = (
        _OFFICIAL_PARTITION["train"]
        if expected_partition is None
        else expected_partition["train"]
    )
    if embeddings.shape != (expected[0], expected_dimension):
        raise ValueError("train descriptor shape differs")
    if labels.shape != (expected[0],) or len(np.unique(labels)) != expected[1]:
        raise ValueError("train label partition differs")
    if example_ids.shape != (expected[0],) or len(set(example_ids.tolist())) != expected[0]:
        raise ValueError("train example-ID order contains duplicates or differs")
    if source_paths.shape != (expected[0],) or len(set(source_paths.tolist())) != expected[0]:
        raise ValueError("train source-path order contains duplicates or differs")
    if not np.isfinite(embeddings).all() or np.any(
        np.abs(np.linalg.norm(embeddings, axis=1) - 1.0) > 2.0e-5
    ):
        raise ValueError("train descriptors must contain finite unit rows")
    row_indices = np.arange(expected[0], dtype=np.int64)
    if _ordered_text_sha256(example_ids.tolist()) != receipt_seed.train_example_id_order_sha256:
        raise ValueError("train example-ID order differs from receipt")
    if _ordered_int64_sha256(labels.tolist()) != receipt_seed.train_label_order_sha256:
        raise ValueError("train label order differs from receipt")
    if _ordered_text_sha256(source_paths.tolist()) != receipt_seed.train_source_order_sha256:
        raise ValueError("train source-path order differs from receipt")
    source_export = {
        "embeddings": embeddings,
        "labels": labels,
        "example_ids": example_ids,
        "source_paths": source_paths,
        "row_indices": row_indices,
    }
    if _source_export_hash(source_export) != receipt_seed.train_source_export_sha256:
        raise ValueError("train source-export SHA-256 differs from receipt")
    loader = checkpoint_loader
    if loader is None:
        _assert_deterministic_tf32_off()
        import torch

        def load_checkpoint(data: io.BytesIO) -> Any:
            return torch.load(data, map_location="cpu", weights_only=False)

        loader = load_checkpoint
    checkpoint = loader(io.BytesIO(captured["checkpoint_pt"]))
    if retrieval.get("resolved_training_steps") != checkpoint.get("training_step"):
        raise ValueError("retrieval audit training step differs from checkpoint")
    proxies, proxy_labels = _validate_checkpoint_mapping(
        checkpoint,
        config=config,
        train_labels=labels,
        expected_dimension=expected_dimension,
    )
    immutable_arrays = {
        "train_embeddings": _readonly_array(embeddings, dtype=np.float32),
        "train_labels": _readonly_array(labels, dtype=np.int64),
        "train_example_ids": _readonly_array(example_ids),
        "train_source_paths": _readonly_array(source_paths),
        "train_row_indices": _readonly_array(row_indices, dtype=np.int64),
        "proxies": _readonly_array(proxies, dtype=np.float32),
        "proxy_labels": _readonly_array(proxy_labels, dtype=np.int64),
    }
    array_hashes = MappingProxyType(
        {name: _framed_array_sha256(name, value) for name, value in immutable_arrays.items()}
    )
    artifact_binding = MappingProxyType(
        {
            "receipt_sha256": _HISTORICAL_RECEIPT_SHA256,
            "producer_commit": _HISTORICAL_PRODUCER_COMMIT,
            "historical_manifest_sha256": _HISTORICAL_MANIFEST_SHA256,
            "artifacts": MappingProxyType(
                {name: MappingProxyType(dict(record)) for name, record in entry.items()}
            ),
            "train_source_export_sha256": receipt_seed.train_source_export_sha256,
        }
    )
    return TrainingOnlySeedInput(
        seed=receipt_seed.seed,
        **immutable_arrays,
        alpha=float(config["proxy_anchor_alpha"]),
        delta=float(config["proxy_anchor_delta"]),
        official_recall_at_1=receipt_seed.official_recall_at_1,
        checkpoint_bytes=bytes(captured["checkpoint_pt"]),
        checkpoint_sha256=entry["checkpoint_pt"]["sha256"],
        training_array_sha256=array_hashes,
        config=_deep_freeze(config),
        artifact_binding=artifact_binding,
    )


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Publish finite JSON once without ever replacing an existing destination."""
    encoded = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    if not path.parent.is_dir():
        raise FileNotFoundError(f"output parent directory is missing: {path.parent}")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_owned = False
    linked = False
    published = False
    try:
        with temporary.open("xb") as stream:
            temporary_owned = True
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path, follow_symlinks=False)
        linked = True
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
            temporary.unlink()
            os.fsync(directory_fd)
            published = True
        finally:
            os.close(directory_fd)
    finally:
        if temporary_owned:
            temporary.unlink(missing_ok=True)
        if linked and not published:
            path.unlink(missing_ok=True)


def _manifest_reference_path(path_text: str, *, manifest_path: Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else manifest_path.resolve().parent.parent / path


def _validated_reference(
    value: Any,
    *,
    name: str,
    manifest_path: Path,
) -> Path:
    if not isinstance(value, dict) or set(value) != {"path", "sha256"}:
        raise ValueError(f"RSTA manifest {name} must contain path and sha256")
    path = _manifest_reference_path(str(value["path"]), manifest_path=manifest_path)
    if not path.is_file():
        raise ValueError(f"RSTA manifest {name} is missing: {path}")
    observed = sha256_file(path)
    if observed != value["sha256"]:
        raise ValueError(f"RSTA manifest {name} SHA-256 mismatch: {observed}")
    return path


def validate_rsta_manifest(manifest: dict[str, Any], *, manifest_path: Path) -> None:
    """Validate frozen preregistration, Pass159 artifacts, and production source."""
    required = {"schema_version", "preregistration", "artifact_schema", "source", "seeds"}
    if set(manifest) != required or manifest.get("schema_version") != 1:
        raise ValueError("RSTA manifest must match schema version 1 exactly")
    _validated_reference(
        manifest["preregistration"],
        name="preregistration",
        manifest_path=manifest_path,
    )
    pass159_path = _validated_reference(
        manifest["artifact_schema"],
        name="artifact schema",
        manifest_path=manifest_path,
    )
    pass159 = json.loads(pass159_path.read_text(encoding="utf-8"))
    if pass159.get("schema_version") != 1 or pass159.get("seeds") != manifest["seeds"]:
        raise ValueError("RSTA manifest seeds differ from the frozen Pass159 manifest seeds")
    if not isinstance(manifest["seeds"], dict) or set(manifest["seeds"]) != {
        "0",
        "1",
        "2",
        "3",
    }:
        raise ValueError("RSTA manifest requires exactly Pass159 seeds 0-3")
    source = manifest["source"]
    if not isinstance(source, dict) or set(source) != {"git_revision", "files"}:
        raise ValueError("RSTA manifest source must contain git_revision and files")
    revision = source["git_revision"]
    if (
        not isinstance(revision, str)
        or len(revision) != 40
        or any(character not in "0123456789abcdef" for character in revision)
    ):
        raise ValueError("RSTA manifest source git_revision must be a full lowercase hash")
    files = source["files"]
    if not isinstance(files, dict) or set(files) != _FROZEN_SOURCE_FILES:
        observed_keys = set(files) if isinstance(files, dict) else files
        raise ValueError(f"RSTA manifest source file keys differ: {observed_keys}")
    repository = manifest_path.resolve().parent.parent
    commit_check = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if commit_check.returncode != 0:
        raise ValueError(f"RSTA source git_revision does not resolve to a commit: {revision}")
    for path_text, expected_digest in sorted(files.items()):
        path = _manifest_reference_path(str(path_text), manifest_path=manifest_path)
        if not path.is_file():
            raise ValueError(f"RSTA source file is missing: {path}")
        observed = sha256_file(path)
        if observed != expected_digest:
            raise ValueError(f"RSTA source SHA-256 mismatch for {path_text}: {observed}")
        blob = subprocess.run(
            ["git", "-C", str(repository), "cat-file", "blob", f"{revision}:{path_text}"],
            capture_output=True,
            check=False,
        )
        if blob.returncode != 0:
            raise ValueError(f"RSTA source revision lacks frozen blob: {revision}:{path_text}")
        blob_digest = hashlib.sha256(blob.stdout).hexdigest()
        if blob_digest != expected_digest:
            raise ValueError(
                f"RSTA source revision blob SHA-256 mismatch for {path_text}: {blob_digest}"
            )


def _is_lowercase_hex(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def validate_execution_audit(
    execution_audit: Mapping[str, Any],
    *,
    manifest_source: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    """Fail closed unless execution is bound to the frozen diagnostic source."""
    if not isinstance(execution_audit, Mapping) or set(execution_audit) != (
        _EXECUTION_AUDIT_FIELDS
    ):
        observed = set(execution_audit) if isinstance(execution_audit, Mapping) else None
        raise ValueError(f"execution audit fields differ: {observed}")
    executing_commit = execution_audit["executing_git_commit"]
    if not _is_lowercase_hex(executing_commit, length=40):
        raise ValueError("execution audit executing Git commit must be a full lowercase hash")
    diagnostic_path = execution_audit["diagnostic_path"]
    if diagnostic_path != _DIAGNOSTIC_PATH:
        raise ValueError(f"execution audit diagnostic path differs: {diagnostic_path}")
    diagnostic_sha256 = execution_audit["diagnostic_sha256"]
    if not _is_lowercase_hex(diagnostic_sha256, length=64):
        raise ValueError("execution audit diagnostic SHA-256 must be lowercase hex")
    frozen_revision = execution_audit["frozen_source_revision"]
    if not _is_lowercase_hex(frozen_revision, length=40):
        raise ValueError("execution audit frozen source revision must be a full lowercase hash")
    if not isinstance(manifest_source, Mapping):
        raise ValueError("execution audit requires manifest source metadata")
    files = manifest_source.get("files")
    if not isinstance(files, Mapping) or diagnostic_path not in files:
        raise ValueError("execution audit diagnostic path is not a frozen manifest file")
    if frozen_revision != manifest_source.get("git_revision"):
        raise ValueError("execution audit frozen source revision differs from manifest")
    if diagnostic_sha256 != files[diagnostic_path]:
        raise ValueError("execution audit diagnostic SHA-256 differs from frozen manifest")
    repository = manifest_path.resolve().parent.parent
    diagnostic = (repository / diagnostic_path).resolve()
    executing_diagnostic = Path(__file__).resolve()
    if executing_diagnostic != diagnostic:
        raise ValueError(
            "execution audit executing diagnostic path differs from manifest repository: "
            f"{executing_diagnostic} != {diagnostic}"
        )
    if not executing_diagnostic.is_file():
        raise ValueError(
            f"execution audit executing diagnostic path is missing: {executing_diagnostic}"
        )
    observed_sha256 = sha256_file(executing_diagnostic)
    if observed_sha256 != diagnostic_sha256:
        raise ValueError(
            "execution audit diagnostic SHA-256 differs from observed worktree: "
            f"{observed_sha256}"
        )
    commit_check = subprocess.run(
        ["git", "-C", str(repository), "cat-file", "-e", f"{executing_commit}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    if commit_check.returncode != 0:
        raise ValueError(
            f"execution audit executing Git commit does not resolve: {executing_commit}"
        )
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"],
        capture_output=True,
        check=False,
        text=True,
    )
    observed_commit = head.stdout.strip()
    if head.returncode != 0 or not _is_lowercase_hex(observed_commit, length=40):
        raise ValueError("execution audit repository HEAD does not resolve to a commit")
    if executing_commit != observed_commit:
        raise ValueError(
            "execution audit executing Git commit differs from repository HEAD: "
            f"{observed_commit}"
        )


def build_execution_audit(
    manifest: Mapping[str, Any],
    *,
    manifest_path: Path,
) -> dict[str, Any]:
    """Derive deterministic executing-source provenance from the manifest repository."""
    source = manifest.get("source")
    files = source.get("files") if isinstance(source, Mapping) else None
    if not isinstance(source, Mapping) or not isinstance(files, Mapping):
        raise ValueError("execution audit requires manifest source metadata")
    if _DIAGNOSTIC_PATH not in files:
        raise ValueError("execution audit diagnostic path is not a frozen manifest file")
    repository = manifest_path.resolve().parent.parent
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "--verify", "HEAD^{commit}"],
        capture_output=True,
        check=False,
        text=True,
    )
    executing_commit = head.stdout.strip()
    if head.returncode != 0 or not _is_lowercase_hex(executing_commit, length=40):
        raise ValueError("execution audit repository HEAD does not resolve to a commit")
    diagnostic = (repository / _DIAGNOSTIC_PATH).resolve()
    executing_diagnostic = Path(__file__).resolve()
    if executing_diagnostic != diagnostic:
        raise ValueError(
            "execution audit executing diagnostic path differs from manifest repository: "
            f"{executing_diagnostic} != {diagnostic}"
        )
    if not executing_diagnostic.is_file():
        raise ValueError(
            f"execution audit executing diagnostic path is missing: {executing_diagnostic}"
        )
    audit = {
        "executing_git_commit": executing_commit,
        "diagnostic_path": _DIAGNOSTIC_PATH,
        "diagnostic_sha256": sha256_file(executing_diagnostic),
        "frozen_source_revision": source.get("git_revision"),
    }
    validate_execution_audit(
        audit,
        manifest_source=source,
        manifest_path=manifest_path,
    )
    return audit


def _ordered_text_sha256(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(str(value) for value in values).encode("utf-8")).hexdigest()


def _ordered_int64_sha256(values: Sequence[int]) -> str:
    return hashlib.sha256(np.asarray(values, dtype=np.int64).tobytes(order="C")).hexdigest()


def validate_cross_seed_training_binding(bounds: Sequence[TrainingOnlySeedInput]) -> None:
    """Require exact training row identity, label, source, and index binding across seeds."""
    if not bounds:
        raise ValueError("cross-seed training binding requires at least one seed")
    reference = bounds[0]
    for bound in bounds[1:]:
        if not np.array_equal(bound.train_example_ids, reference.train_example_ids):
            raise ValueError("training example-ID order differs across seeds")
        if not np.array_equal(bound.train_labels, reference.train_labels):
            raise ValueError("training label order differs across seeds")
        if not np.array_equal(bound.train_source_paths, reference.train_source_paths):
            raise ValueError("training source membership differs across seeds")
        if not np.array_equal(bound.train_row_indices, reference.train_row_indices):
            raise ValueError("training row-index binding differs across seeds")


def binding_only_payload(
    manifest: dict[str, Any],
    *,
    manifest_path: Path,
    source_exporter: Callable[..., dict[str, dict[str, np.ndarray]]] | None = None,
    expected_partition: dict[str, tuple[int, int]] | None = None,
    expected_dimension: int = 512,
) -> dict[str, Any]:
    """Run only immutable artifact/source gates and return a non-scientific audit."""
    validate_rsta_manifest(manifest, manifest_path=manifest_path)
    execution_audit = build_execution_audit(manifest, manifest_path=manifest_path)
    bounds = [
        load_and_bind_seed(
            manifest["seeds"][str(seed)],
            seed=seed,
            source_exporter=source_exporter,
            expected_partition=expected_partition,
            expected_dimension=expected_dimension,
        )
        for seed in range(4)
    ]
    validate_cross_seed_training_binding(bounds)
    seed_results = [
        {
            "seed": bound.seed,
            "train_row_count": len(bound.train_example_ids),
            "train_identity_count": len(set(bound.train_labels)),
            "train_example_id_order_sha256": _ordered_text_sha256(bound.train_example_ids),
            "train_label_order_sha256": _ordered_int64_sha256(bound.train_labels),
            "train_source_order_sha256": _ordered_text_sha256(bound.train_source_paths),
            "official_recall_at_1": bound.official_recall_at_1,
            "artifact_binding": _json_ready(bound.artifact_binding),
        }
        for bound in bounds
    ]
    return {
        "schema_version": 1,
        "diagnostic": "pass200_rsta_stage_a",
        "mode": "binding_only",
        "candidate_values_computed": False,
        "stage_a_verdict": "NOT_COMPUTED",
        "uses_test_data": "artifact_binding_only",
        "execution_audit": execution_audit,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "preregistration": manifest["preregistration"],
            "artifact_schema": manifest["artifact_schema"],
            "source": manifest["source"],
        },
        "binding": {
            "cross_seed_training_rows_identical": True,
            "query_gallery_released_before_scientific_input": True,
            "source_export_batch_size": 128,
            "descriptor_atol": 2.0e-5,
            "descriptor_rtol": 2.0e-5,
            "seeds": seed_results,
        },
    }


def make_bufferless_train_clone(model: Any) -> Any:
    """Clone a train-mode model while making BN use batch statistics without mutation."""
    import torch

    clone = copy.deepcopy(model)
    clone.train()
    for module in clone.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.track_running_stats = False
    return clone


def _new_deterministic_global_max_pool() -> Any:
    import torch

    pool_type = globals().get("_DeterministicGlobalMaxPool2d")
    if pool_type is None:
        class _DeterministicGlobalMaxPool2d(torch.nn.Module):
            def forward(self, x: Any) -> Any:
                return x.flatten(-2).max(dim=-1, keepdim=True).values.unsqueeze(-1)

        globals()["_DeterministicGlobalMaxPool2d"] = _DeterministicGlobalMaxPool2d
        pool_type = _DeterministicGlobalMaxPool2d
    return pool_type()


def make_rsta_diagnostic_clone(model: Any) -> Any:
    """Clone the bound model and replace only its exact final global-max operator."""
    import torch

    clone = make_bufferless_train_clone(model)
    try:
        pool = clone.get_submodule("model.gmp")
        parent = clone.get_submodule("model")
    except (AttributeError, KeyError) as error:
        raise ValueError("diagnostic model lacks exact model.gmp") from error
    if type(pool) is not torch.nn.AdaptiveMaxPool2d or type(pool.output_size) is not int:
        raise ValueError("diagnostic model.gmp must be exact AdaptiveMaxPool2d integer 1")
    if pool.output_size != 1:
        raise ValueError("diagnostic model.gmp output_size must equal integer 1")

    parent.gmp = _new_deterministic_global_max_pool()
    return clone


def _deterministic_global_max_inputs() -> dict[str, np.ndarray]:
    generator = np.random.Generator(np.random.PCG64(200))
    random_values = np.ascontiguousarray(
        generator.standard_normal((2, 3, 5, 7)).astype(np.float32)
    )
    tie = random_values.copy(order="C")
    tie[:, :, 0, 0] = np.float32(100.0)
    tie[:, :, 0, 1] = np.float32(100.0)
    return {
        "random": random_values,
        "relu": np.ascontiguousarray(np.maximum(random_values, np.float32(0.0))),
        "zeros": np.zeros_like(random_values, dtype=np.float32, order="C"),
        "tie": tie,
    }


def _max_abs_difference(left: Any, right: Any) -> float:
    import torch

    return float(torch.max(torch.abs(left - right)).detach().cpu())


def _audit_deterministic_global_max_cpu(
    cases: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    import torch

    results: dict[str, dict[str, Any]] = {}
    for name in ("random", "relu", "zeros", "tie"):
        raw = np.ascontiguousarray(cases[name], dtype=np.float32).tobytes(order="C")
        reference_input = torch.from_numpy(
            np.frombuffer(raw, dtype=np.float32).copy().reshape(2, 3, 5, 7)
        ).requires_grad_(True)
        replacement_input = torch.from_numpy(
            np.frombuffer(raw, dtype=np.float32).copy().reshape(2, 3, 5, 7)
        ).requires_grad_(True)
        reference_output = torch.nn.AdaptiveMaxPool2d(1)(reference_input)
        replacement_output = _new_deterministic_global_max_pool()(replacement_input)
        reference_output.sum().backward()
        replacement_output.sum().backward()
        output_equal = torch.equal(reference_output, replacement_output)
        gradient_equal = torch.equal(reference_input.grad, replacement_input.grad)
        results[name] = {
            "output_equal": output_equal,
            "gradient_equal": gradient_equal,
            "max_abs_output_difference": _max_abs_difference(
                reference_output, replacement_output
            ),
            "max_abs_gradient_difference": _max_abs_difference(
                reference_input.grad, replacement_input.grad
            ),
        }
    return results


def _audit_deterministic_global_max_cuda(
    cases: Mapping[str, np.ndarray],
) -> dict[str, dict[str, Any]]:
    import torch

    results: dict[str, dict[str, Any]] = {}
    for name in ("random", "relu", "zeros", "tie"):
        raw = np.ascontiguousarray(cases[name], dtype=np.float32).tobytes(order="C")
        replacement_input = torch.from_numpy(
            np.frombuffer(raw, dtype=np.float32).copy().reshape(2, 3, 5, 7)
        ).to(device="cuda").requires_grad_(True)
        reference_output, reference_indices = torch.nn.functional.adaptive_max_pool2d(
            replacement_input.detach(), (1, 1), return_indices=True
        )
        flattened = replacement_input.flatten(-2)
        replacement_indices = flattened.max(dim=-1, keepdim=True).indices
        replacement_output = _new_deterministic_global_max_pool()(replacement_input)
        expected_gradient = torch.zeros_like(flattened)
        expected_gradient.scatter_(-1, reference_indices.flatten(-2), 1.0)
        replacement_output.sum().backward()
        actual_gradient = replacement_input.grad.flatten(-2)
        results[name] = {
            "output_equal": torch.equal(reference_output, replacement_output),
            "index_equal": torch.equal(
                reference_indices.flatten(-2), replacement_indices
            ),
            "replacement_gradient_equal_expected": torch.equal(
                actual_gradient, expected_gradient
            ),
            "max_abs_output_difference": _max_abs_difference(
                reference_output, replacement_output
            ),
            "max_abs_replacement_gradient_difference": _max_abs_difference(
                actual_gradient, expected_gradient
            ),
        }
    return results


def _validate_deterministic_global_max_audit(audit: Mapping[str, Any]) -> None:
    top = _require_exact_keys(
        audit,
        {
            "replacement_id",
            "module_path",
            "reference_type",
            "reference_output_size",
            "fixture_seed",
            "fixture_generator",
            "fixture_shape",
            "fixture_dtype",
            "derivative",
            "input_sha256",
            "cases",
            "deterministic_cuda_backward",
        },
        name="deterministic global max audit",
    )
    scalar_expected = {
        "replacement_id": "pass200-global-max-flatten-first-v1",
        "module_path": "model.gmp",
        "reference_type": "torch.nn.modules.pooling.AdaptiveMaxPool2d",
        "fixture_generator": "numpy.PCG64",
        "fixture_dtype": "float32",
        "derivative": "output.sum()",
    }
    if any(
        type(top[name]) is not str or top[name] != expected
        for name, expected in scalar_expected.items()
    ):
        raise ValueError("deterministic global max scalar contract differs")
    if type(top["reference_output_size"]) is not int or top["reference_output_size"] != 1:
        raise ValueError("deterministic global max reference output size differs")
    if type(top["fixture_seed"]) is not int or top["fixture_seed"] != 200:
        raise ValueError("deterministic global max fixture seed differs")
    if top["fixture_shape"] != [2, 3, 5, 7] or any(
        type(value) is not int for value in top["fixture_shape"]
    ):
        raise ValueError("deterministic global max fixture shape differs")
    input_sha = _require_exact_keys(
        top["input_sha256"],
        _DETERMINISTIC_GLOBAL_MAX_INPUT_SHA256,
        name="deterministic global max input SHA-256",
    )
    if input_sha != _DETERMINISTIC_GLOBAL_MAX_INPUT_SHA256:
        raise ValueError("deterministic global max input SHA-256 differs")
    cases = _require_exact_keys(
        top["cases"], {"cpu", "cuda"}, name="deterministic global max devices"
    )
    case_names = {"random", "relu", "zeros", "tie"}
    schemas = {
        "cpu": {
            "output_equal",
            "gradient_equal",
            "max_abs_output_difference",
            "max_abs_gradient_difference",
        },
        "cuda": {
            "output_equal",
            "index_equal",
            "replacement_gradient_equal_expected",
            "max_abs_output_difference",
            "max_abs_replacement_gradient_difference",
        },
    }
    for device, schema in schemas.items():
        device_cases = _require_exact_keys(
            cases[device], case_names, name=f"deterministic global max {device} cases"
        )
        for name, result in device_cases.items():
            result = _require_exact_keys(
                result, schema, name=f"deterministic global max {device} {name}"
            )
            for key, value in result.items():
                if key.endswith("equal") or key.endswith("expected"):
                    if type(value) is not bool or value is not True:
                        raise ValueError("deterministic global max equality gate failed")
                elif type(value) is not float or value != 0.0:
                    raise ValueError("deterministic global max difference must be float 0.0")
    deterministic = _require_exact_keys(
        top["deterministic_cuda_backward"],
        {"enabled", "warn_only", "completed"},
        name="deterministic global max CUDA backward",
    )
    if deterministic != {"enabled": True, "warn_only": False, "completed": True} or any(
        type(value) is not bool for value in deterministic.values()
    ):
        raise ValueError("deterministic global max CUDA backward audit differs")


def audit_deterministic_global_max() -> dict[str, Any]:
    import torch

    _assert_deterministic_tf32_off()
    if not torch.cuda.is_available():
        raise ValueError("deterministic global max fixture requires CUDA")
    cases = _deterministic_global_max_inputs()
    input_sha256 = {
        name: hashlib.sha256(value.tobytes(order="C")).hexdigest()
        for name, value in cases.items()
    }
    audit = {
        "replacement_id": "pass200-global-max-flatten-first-v1",
        "module_path": "model.gmp",
        "reference_type": "torch.nn.modules.pooling.AdaptiveMaxPool2d",
        "reference_output_size": 1,
        "fixture_seed": 200,
        "fixture_generator": "numpy.PCG64",
        "fixture_shape": [2, 3, 5, 7],
        "fixture_dtype": "float32",
        "derivative": "output.sum()",
        "input_sha256": input_sha256,
        "cases": {
            "cpu": _audit_deterministic_global_max_cpu(cases),
            "cuda": _audit_deterministic_global_max_cuda(cases),
        },
        "deterministic_cuda_backward": {
            "enabled": torch.are_deterministic_algorithms_enabled(),
            "warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
            "completed": True,
        },
    }
    _assert_deterministic_tf32_off()
    _validate_deterministic_global_max_audit(audit)
    return audit


_ZERO_JACOBIAN_CLASSIFIER_NAMES = (
    "model.last_linear.weight",
    "model.last_linear.bias",
)


def _validate_zero_jacobian_classifier_audit(audit: Mapping[str, Any]) -> None:
    value = _require_exact_keys(
        audit,
        {
            "audit_id",
            "parameter_names",
            "parameter_shapes",
            "parameter_dtypes",
            "pre_sha256",
            "restored_sha256",
            "gradients_none",
            "mutated_output_equal",
            "frozen_requires_grad",
        },
        name="zero-Jacobian classifier audit",
    )
    if value["audit_id"] != "pass200-zero-jacobian-last-linear-v1" or type(
        value["audit_id"]
    ) is not str:
        raise ValueError("zero-Jacobian classifier audit identifier differs")
    if value["parameter_names"] != list(_ZERO_JACOBIAN_CLASSIFIER_NAMES) or any(
        type(name) is not str for name in value["parameter_names"]
    ):
        raise ValueError("zero-Jacobian classifier parameter names differ")
    if value["parameter_shapes"] != [[1000, 1024], [1000]] or any(
        type(dimension) is not int
        for shape in value["parameter_shapes"]
        if isinstance(shape, list)
        for dimension in shape
    ):
        raise ValueError("zero-Jacobian classifier parameter shapes differ")
    if value["parameter_dtypes"] != ["torch.float32", "torch.float32"] or any(
        type(dtype) is not str for dtype in value["parameter_dtypes"]
    ):
        raise ValueError("zero-Jacobian classifier parameter dtypes differ")
    for field in ("pre_sha256", "restored_sha256"):
        hashes = _require_exact_keys(
            value[field],
            set(_ZERO_JACOBIAN_CLASSIFIER_NAMES),
            name=f"zero-Jacobian classifier {field}",
        )
        for name, digest in hashes.items():
            _require_sha256(digest, name=f"zero-Jacobian classifier {field} {name}")
    if value["restored_sha256"] != value["pre_sha256"]:
        raise ValueError("zero-Jacobian classifier restoration SHA-256 differs")
    if value["gradients_none"] != [True, True] or any(
        type(flag) is not bool for flag in value["gradients_none"]
    ):
        raise ValueError("zero-Jacobian classifier gradient audit differs")
    if type(value["mutated_output_equal"]) is not bool or not value[
        "mutated_output_equal"
    ]:
        raise ValueError("zero-Jacobian classifier mutated output differs")
    if value["frozen_requires_grad"] != [False, False] or any(
        type(flag) is not bool for flag in value["frozen_requires_grad"]
    ):
        raise ValueError("zero-Jacobian classifier frozen state differs")


def audit_zero_jacobian_classifier(model: Any, images: Any) -> dict[str, Any]:
    import torch

    _assert_deterministic_tf32_off()
    if type(images.shape[0]) is not int or images.shape[0] != 180:
        raise ValueError("zero-Jacobian classifier audit requires exact B=180")
    named = list(model.named_parameters())
    selected = [
        (name, parameter)
        for name, parameter in named
        if name in _ZERO_JACOBIAN_CLASSIFIER_NAMES
    ]
    if [name for name, _ in selected] != list(_ZERO_JACOBIAN_CLASSIFIER_NAMES):
        raise ValueError("zero-Jacobian classifier exact parameter names/order differ")
    parameters = [parameter for _, parameter in selected]
    if [list(parameter.shape) for parameter in parameters] != [[1000, 1024], [1000]]:
        raise ValueError("zero-Jacobian classifier exact parameter shapes differ")
    if [parameter.dtype for parameter in parameters] != [torch.float32, torch.float32]:
        raise ValueError("zero-Jacobian classifier exact parameter dtypes differ")
    if [parameter.requires_grad for parameter in parameters] != [True, True]:
        raise ValueError("zero-Jacobian classifier parameters must initially require gradients")
    _assert_deterministic_tf32_off()
    output = model(images)
    gradients = torch.autograd.grad(
        output.sum(), tuple(parameters), allow_unused=True
    )
    gradients_none = [gradient is None for gradient in gradients]
    if gradients_none != [True, True]:
        raise ValueError("zero-Jacobian classifier has a nonzero graph dependency")
    originals = [parameter.detach().clone() for parameter in parameters]
    pre_sha256 = {
        name: _torch_tensor_sha256(parameter)
        for (name, parameter) in selected
    }
    mutated_output: Any | None = None
    try:
        with torch.no_grad():
            parameters[0].fill_(0.125)
            parameters[1].fill_(-0.25)
        _assert_deterministic_tf32_off()
        mutated_output = model(images)
    finally:
        with torch.no_grad():
            for parameter, original in zip(parameters, originals, strict=True):
                parameter.copy_(original)
    restored_sha256 = {
        name: _torch_tensor_sha256(parameter)
        for (name, parameter) in selected
    }
    if restored_sha256 != pre_sha256:
        raise ValueError("zero-Jacobian classifier restoration SHA-256 differs")
    mutated_output_equal = torch.equal(output, mutated_output)
    if not mutated_output_equal:
        raise ValueError("zero-Jacobian classifier mutation changed model output")
    for parameter in parameters:
        parameter.requires_grad_(False)
    audit = {
        "audit_id": "pass200-zero-jacobian-last-linear-v1",
        "parameter_names": list(_ZERO_JACOBIAN_CLASSIFIER_NAMES),
        "parameter_shapes": [[1000, 1024], [1000]],
        "parameter_dtypes": ["torch.float32", "torch.float32"],
        "pre_sha256": pre_sha256,
        "restored_sha256": restored_sha256,
        "gradients_none": gradients_none,
        "mutated_output_equal": mutated_output_equal,
        "frozen_requires_grad": [parameter.requires_grad for parameter in parameters],
    }
    _assert_deterministic_tf32_off()
    _validate_zero_jacobian_classifier_audit(audit)
    return audit


def capture_prehead_and_raw(
    model: Any,
    images: Any,
    *,
    head_name: str = "model.embedding",
    expected_in_features: int = 1024,
    expected_out_features: int = 512,
) -> tuple[Any, Any, Any]:
    """Capture the exact artifact-bound affine embedding head in one full forward."""
    import torch

    try:
        head = model.get_submodule(head_name)
    except (AttributeError, KeyError) as error:
        raise ValueError(f"encoder lacks exact affine head {head_name}") from error
    if (
        not isinstance(head, torch.nn.Linear)
        or int(head.in_features) != int(expected_in_features)
        or int(head.out_features) != int(expected_out_features)
    ):
        raise ValueError(
            f"exact affine head {head_name} must be {expected_in_features}->{expected_out_features}"
        )
    captures: list[tuple[Any, Any]] = []

    def capture(_module: Any, inputs: tuple[Any, ...], output: Any) -> None:
        if len(inputs) != 1:
            raise ValueError("affine head must receive exactly one tensor")
        captures.append((inputs[0], output))

    handle = head.register_forward_hook(capture)
    try:
        with torch.no_grad():
            output = model(images)
    finally:
        handle.remove()
    if len(captures) != 1:
        raise ValueError(f"exact affine head {head_name} must execute exactly once")
    prehead, raw_head = captures[0]
    if (
        not isinstance(prehead, torch.Tensor)
        or not isinstance(raw_head, torch.Tensor)
        or prehead.shape != (images.shape[0], expected_in_features)
        or raw_head.shape != (images.shape[0], expected_out_features)
        or output.shape != raw_head.shape
    ):
        raise ValueError("exact affine head tensors differ from the artifact contract")
    return prehead.detach().clone(), raw_head.detach().clone(), output.detach().clone()


def _functional_encoder(
    model: Any,
    images: Any,
    *,
    expected_batch_size: int,
    expected_dimension: int,
) -> tuple[Callable[[dict[str, Any]], Any], dict[str, Any], tuple[str, ...]]:
    import torch

    if not model.training:
        raise ValueError("exact derivative encoder must be in train mode")
    if (
        not isinstance(images, torch.Tensor)
        or images.ndim < 2
        or images.shape[0] != int(expected_batch_size)
    ):
        raise ValueError(f"exact derivative batch size {expected_batch_size} is required")
    if not bool(torch.isfinite(images).all()):
        raise ValueError("encoder images must be finite")
    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm) and (
            not module.training or module.track_running_stats
        ):
            raise ValueError("every BatchNorm must be train mode and bufferless")
    parameters = {
        name: parameter for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    constant_parameters = {
        name: parameter
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    constant_parameter_names = tuple(constant_parameters)
    if constant_parameter_names and constant_parameter_names != _ZERO_JACOBIAN_CLASSIFIER_NAMES:
        raise ValueError(
            "frozen encoder parameter names differ: " + ", ".join(constant_parameter_names)
        )
    if not parameters:
        raise ValueError("encoder has no trainable parameters")
    parameter_names = tuple(parameters)
    buffers = {name: buffer for name, buffer in model.named_buffers()}

    def encoder(current_parameters: dict[str, Any]) -> Any:
        raw = torch.func.functional_call(
            model,
            (current_parameters, constant_parameters, buffers),
            (images,),
            strict=True,
        )
        if not isinstance(raw, torch.Tensor) or raw.shape != (
            int(expected_batch_size),
            int(expected_dimension),
        ):
            raise ValueError(
                f"exact derivative descriptor dimension {expected_dimension} is required"
            )
        return torch.nn.functional.normalize(raw, p=2, dim=-1)

    return encoder, parameters, parameter_names


def _flatten_parameter_tree(tree: Mapping[str, Any], names: Sequence[str]) -> Any:
    import torch

    return torch.cat([tree[name].reshape(-1) for name in names])


def exact_kernel_fields(
    model: Any,
    images: Any,
    cotangents: Any,
    *,
    receiver_indices: Sequence[int],
    expected_batch_size: int = 180,
    expected_dimension: int = 512,
    _graph: tuple[Any, ...] | None = None,
) -> dict[str, Any]:
    """Compute exact ``J J^T dbar`` and serial ``J_i J_i^T dbar_i`` motions."""
    import torch

    if _graph is None:
        encoder, parameters, parameter_names = _functional_encoder(
            model,
            images,
            expected_batch_size=expected_batch_size,
            expected_dimension=expected_dimension,
        )
        z, vjp_function = torch.func.vjp(encoder, parameters)
        dependencies = torch.autograd.grad(
            z,
            tuple(parameters.values()),
            grad_outputs=torch.ones_like(z),
            allow_unused=True,
            retain_graph=True,
            create_graph=False,
        )
        missing = [
            name
            for name, gradient in zip(parameter_names, dependencies, strict=True)
            if gradient is None
        ]
        if missing:
            raise ValueError(f"missing gradient for encoder parameter(s): {', '.join(missing)}")
    else:
        encoder, parameters, parameter_names, z, vjp_function = _graph
    directions = torch.as_tensor(cotangents, device=z.device, dtype=z.dtype)
    if directions.shape != z.shape or not bool(torch.isfinite(directions).all()):
        raise ValueError("cotangents must be a finite tensor aligned with descriptors")
    receivers = tuple(int(value) for value in receiver_indices)
    if (
        not receivers
        or len(set(receivers)) != len(receivers)
        or any(value < 0 or value >= int(z.shape[0]) for value in receivers)
    ):
        raise ValueError("receiver indices must be unique in-range batch rows")
    global_gradient = vjp_function(directions)[0]
    if set(global_gradient) != set(parameter_names):
        raise ValueError("VJP parameter tree differs from the encoder parameter tree")
    if any(not bool(torch.isfinite(value).all()) for value in global_gradient.values()):
        raise ValueError("global parameter gradient is nonfinite")
    _, batch_motion = torch.func.jvp(
        encoder,
        (parameters,),
        (global_gradient,),
    )
    self_rows: list[Any] = []
    self_parameter_norms: list[float] = []
    for receiver in receivers:
        receiver_cotangent = torch.zeros_like(directions)
        receiver_cotangent[receiver] = directions[receiver]
        receiver_gradient = vjp_function(receiver_cotangent)[0]
        if any(not bool(torch.isfinite(value).all()) for value in receiver_gradient.values()):
            raise ValueError(f"receiver {receiver} parameter gradient is nonfinite")
        _, receiver_motion = torch.func.jvp(
            encoder,
            (parameters,),
            (receiver_gradient,),
        )
        self_rows.append(receiver_motion[receiver])
        self_parameter_norms.append(
            float(
                torch.linalg.vector_norm(
                    _flatten_parameter_tree(receiver_gradient, parameter_names)
                )
                .detach()
                .cpu()
            )
        )
    return {
        "z": z,
        "dbar": directions,
        "batch_motion": batch_motion,
        "self_motion": torch.stack(self_rows),
        "receiver_indices": receivers,
        "parameter_names": parameter_names,
        "parameter_count": int(sum(value.numel() for value in parameters.values())),
        "parameter_gradient_flat": _flatten_parameter_tree(global_gradient, parameter_names),
        "self_parameter_norms": tuple(self_parameter_norms),
    }


def exact_contextual_rsta_fields(
    model: Any,
    images: Any,
    labels: Any,
    proxies: Any,
    proxy_labels: Any,
    *,
    alpha: float,
    delta: float,
    receiver_indices: Sequence[int],
    expected_batch_size: int = 180,
    expected_dimension: int = 512,
) -> dict[str, Any]:
    """Construct the exact same-batch PA cotangent, then apply the encoder tangent kernel."""
    import torch

    from sfora.image_end_to_end import _proxy_anchor_loss

    if float(alpha) != 32.0:
        raise ValueError("Proxy Anchor alpha must equal 32")
    if float(delta) != 0.1:
        raise ValueError("Proxy Anchor delta must equal 0.1")
    encoder, parameters, parameter_names = _functional_encoder(
        model,
        images,
        expected_batch_size=expected_batch_size,
        expected_dimension=expected_dimension,
    )
    z, vjp_function = torch.func.vjp(encoder, parameters)
    dependencies = torch.autograd.grad(
        z,
        tuple(parameters.values()),
        grad_outputs=torch.ones_like(z),
        allow_unused=True,
        retain_graph=True,
        create_graph=False,
    )
    missing = [
        name
        for name, gradient in zip(parameter_names, dependencies, strict=True)
        if gradient is None
    ]
    if missing:
        raise ValueError(f"missing gradient for encoder parameter(s): {', '.join(missing)}")
    if labels.shape != (expected_batch_size,):
        raise ValueError("Proxy Anchor labels differ from exact batch size")
    if proxies.ndim != 2 or proxies.shape[1] != expected_dimension:
        raise ValueError("Proxy Anchor proxies differ from exact descriptor dimension")
    loss = _proxy_anchor_loss(
        z,
        labels,
        proxy_embeddings=proxies.detach(),
        proxy_labels=proxy_labels,
        alpha=float(alpha),
        delta=float(delta),
        torch_module=torch,
    )
    dbar = -torch.autograd.grad(loss, z, create_graph=True)[0]
    fields = exact_kernel_fields(
        model,
        images,
        dbar,
        receiver_indices=receiver_indices,
        expected_batch_size=expected_batch_size,
        expected_dimension=expected_dimension,
        _graph=(encoder, parameters, parameter_names, z, vjp_function),
    )
    fields["loss"] = loss
    fields["dbar"] = dbar
    return fields


def adjoint_relative_error(
    model: Any,
    images: Any,
    output_direction: Any,
    parameter_direction: Mapping[str, Any],
    *,
    expected_batch_size: int = 180,
    expected_dimension: int = 512,
) -> float:
    """Return the registered relative error in ``<Jv,u> = <v,J^T u>``."""
    import torch

    encoder, parameters, parameter_names = _functional_encoder(
        model,
        images,
        expected_batch_size=expected_batch_size,
        expected_dimension=expected_dimension,
    )
    if set(parameter_direction) != set(parameter_names):
        raise ValueError("adjoint direction must contain every encoder parameter exactly")
    tangents = {
        name: torch.as_tensor(parameter_direction[name], device=value.device, dtype=value.dtype)
        for name, value in parameters.items()
    }
    if any(tangents[name].shape != parameters[name].shape for name in parameter_names):
        raise ValueError("adjoint parameter direction shape differs")
    z, vjp_function = torch.func.vjp(encoder, parameters)
    u = torch.as_tensor(output_direction, device=z.device, dtype=z.dtype)
    if u.shape != z.shape or not bool(torch.isfinite(u).all()):
        raise ValueError("adjoint output direction differs from descriptor shape")
    _, jv = torch.func.jvp(encoder, (parameters,), (tangents,))
    jtu = vjp_function(u)[0]
    left = torch.sum(jv * u)
    right = sum(torch.sum(tangents[name] * jtu[name]) for name in parameter_names)
    denominator = torch.maximum(
        torch.maximum(torch.abs(left), torch.abs(right)),
        torch.as_tensor(1.0e-12, dtype=left.dtype, device=left.device),
    )
    return float((torch.abs(left - right) / denominator).detach().cpu())


def registered_adjoint_directions(
    model: Any,
    output_shape: Sequence[int],
    *,
    seed: int,
    dtype: Any,
    device: Any,
) -> tuple[Any, dict[str, Any]]:
    """Draw registered C-order output/parameter directions from independent PCG64 streams."""
    import torch

    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("adjoint seed must be an unsigned integer")
    shape = tuple(int(value) for value in output_shape)
    if not shape or any(value <= 0 for value in shape):
        raise ValueError("adjoint output shape must be nonempty and positive")
    parameter_items = [
        (name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not parameter_items:
        raise ValueError("adjoint model has no trainable parameters")
    u_rng = np.random.Generator(
        np.random.PCG64(domain_seed("rsta-stage-a-v1|adjoint-u|", str(int(seed))))
    )
    v_rng = np.random.Generator(
        np.random.PCG64(domain_seed("rsta-stage-a-v1|adjoint-v|", str(int(seed))))
    )
    output_array = np.ascontiguousarray(u_rng.standard_normal(shape), dtype=np.float64)
    total = sum(parameter.numel() for _, parameter in parameter_items)
    flat_parameter = np.ascontiguousarray(v_rng.standard_normal(total), dtype=np.float64)
    output = torch.as_tensor(output_array, dtype=dtype, device=device)
    parameters: dict[str, Any] = {}
    start = 0
    for name, parameter in parameter_items:
        end = start + parameter.numel()
        parameters[name] = torch.as_tensor(
            flat_parameter[start:end].reshape(tuple(parameter.shape), order="C"),
            dtype=dtype,
            device=device,
        )
        start = end
    return output, parameters


def configure_deterministic_process() -> dict[str, Any]:
    """Activate and report every frozen deterministic-process gate before CUDA work."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CUBLAS_WORKSPACE_CONFIG=:4096:8 must be exported before CUDA init")
    import torch

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "autocast": False,
        "model_arithmetic": "float32",
        "reduction_arithmetic": "float64",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def _assert_deterministic_tf32_off() -> None:
    """Fail closed if any fresh-process arithmetic gate changed after configuration."""
    import torch

    if (
        not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.backends.cudnn.benchmark
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.is_autocast_enabled()
        or torch.is_autocast_enabled("cpu")
    ):
        raise ValueError("deterministic TF32-off runtime boundary failed")


def project_and_validate_fields(
    z: Any,
    *,
    dbar: Any,
    b: Any,
    s: Any,
) -> dict[str, Any]:
    """Project measured vectors once into the receiver tangent and enforce INVALID gates."""
    import torch

    descriptors = torch.as_tensor(z)
    if descriptors.ndim != 2 or descriptors.shape[0] == 0:
        raise ValueError("descriptors must be a nonempty matrix")
    if not bool(torch.isfinite(descriptors).all()):
        raise ValueError("descriptors are nonfinite")
    unit_error = torch.abs(torch.linalg.vector_norm(descriptors, dim=1) - 1.0)
    if bool(torch.any(unit_error > 2.0e-5)):
        raise ValueError("descriptor unit-row error exceeds 2e-5")
    projected: dict[str, Any] = {}
    radial_fractions: dict[str, Any] = {}
    for name, raw in (("dbar", dbar), ("b", b), ("s", s)):
        value = torch.as_tensor(raw, device=descriptors.device, dtype=descriptors.dtype)
        if value.shape != descriptors.shape:
            raise ValueError(f"{name} shape differs from descriptors")
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"{name} is nonfinite")
        norms = torch.linalg.vector_norm(value, dim=1)
        if bool(torch.any(norms <= _VECTOR_EPS)):
            raise ValueError(f"{name} must have nonzero row norms")
        radial = descriptors * torch.sum(descriptors * value, dim=1, keepdim=True)
        tangent = value - radial
        tangent_norms = torch.linalg.vector_norm(tangent, dim=1)
        radial_fraction = torch.linalg.vector_norm(radial, dim=1) / norms
        if bool(torch.any(radial_fraction > 1.0e-3)):
            raise ValueError(f"{name} radial fraction exceeds 1e-3")
        if bool(torch.any(tangent_norms <= _VECTOR_EPS)):
            raise ValueError(f"{name} tangent projection must have nonzero row norms")
        projected[name] = tangent
        radial_fractions[name] = radial_fraction
    return {
        "z": descriptors,
        **projected,
        "unit_error": unit_error,
        "radial_fractions": radial_fractions,
    }


def _validate_receiver_audit_row(row: Mapping[str, Any], *, expected_panel: str) -> None:
    missing = RECEIVER_AUDIT_FIELDS - set(row)
    if missing:
        raise ValueError(f"receiver audit fields missing: {sorted(missing)}")
    if row["panel"] != expected_panel:
        raise ValueError(f"receiver audit panel must equal {expected_panel}")
    if len(row["support_ids"]) != 2:
        raise ValueError("receiver audit requires exactly two support IDs")
    if len(row["foreign_ids"]) != 32:
        raise ValueError("receiver audit requires exactly 32 foreign IDs")
    if (
        len(set(row["support_ids"])) != 2
        or len(set(row["foreign_ids"])) != 32
        or set(row["support_ids"]) & set(row["foreign_ids"])
    ):
        raise ValueError("receiver support and foreign IDs must be unique and disjoint")
    if len(row["batch_ids"]) != 180 or len(set(row["batch_ids"])) != 180:
        raise ValueError("receiver audit requires 180 unique ordered batch IDs")
    receiver_index = row["receiver_index"]
    if (
        isinstance(receiver_index, bool)
        or not isinstance(receiver_index, (int, np.integer))
        or int(receiver_index) < 0
        or int(receiver_index) >= 180
        or row["batch_ids"][int(receiver_index)] != row["receiver_id"]
    ):
        raise ValueError("receiver audit receiver index and batch membership differ")
    if (
        row["receiver_id"] in row["support_ids"]
        or row["receiver_id"] in row["foreign_ids"]
        or set(row["support_ids"]) & set(row["batch_ids"])
        or set(row["foreign_ids"]) & set(row["batch_ids"])
    ):
        raise ValueError("receiver supports and foreign IDs must be absent from the batch")
    if len(row["batch_tensor_sha256"]) != 180:
        raise ValueError("receiver audit requires every ordered batch tensor hash")
    if row["batch_id_order_sha256"] != _ordered_text_sha256(row["batch_ids"]):
        raise ValueError("receiver audit batch-ID order hash differs")
    if len(row["support_cosines"]) != 34:
        raise ValueError("receiver audit requires every support cosine")
    sha_fields = [row["tensor_sha256"], *row["batch_tensor_sha256"]]
    if any(
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
        for value in sha_fields
    ):
        raise ValueError("receiver audit SHA-256 values must be lowercase hex")
    if row["batch_tensor_sha256"][int(receiver_index)] != row["tensor_sha256"]:
        raise ValueError("receiver tensor SHA-256 differs from its batch tensor hash")
    scalar_names = RECEIVER_AUDIT_FIELDS - {
        "panel",
        "seed",
        "label",
        "batch_index",
        "receiver_index",
        "receiver_id",
        "support_ids",
        "foreign_ids",
        "batch_ids",
        "batch_tensor_sha256",
        "batch_id_order_sha256",
        "tensor_sha256",
        "support_cosines",
    }
    if any(not np.isfinite(float(row[name])) for name in scalar_names) or any(
        not np.isfinite(float(value)) for value in row["support_cosines"]
    ):
        raise ValueError("receiver audit scalar is nonfinite")
    norm_names = (
        "norm_dbar",
        "norm_b",
        "norm_s",
        "norm_q",
        "norm_random_target",
        "norm_deranged_target",
        "norm_b_head",
        "norm_s_head",
    )
    if any(float(row[name]) <= _VECTOR_EPS for name in norm_names):
        raise ValueError("receiver audit norm must be finite and positive")
    if abs(float(row["norm_z"]) - 1.0) > 2.0e-5:
        raise ValueError("receiver audit unit norm exceeds tolerance")
    if any(
        float(row[name]) < 0.0 or float(row[name]) > 1.0e-3
        for name in ("radial_fraction_dbar", "radial_fraction_b", "radial_fraction_s")
    ):
        raise ValueError("receiver audit radial fraction exceeds tolerance")
    if abs(float(row["head_self_desc_gap"])) > 1.0e-5:
        raise ValueError("receiver audit head self-descriptor gap exceeds tolerance")
    cosine_names = (
        "a_self",
        "a_batch",
        "a_desc",
        "cos_b_s",
        "random_a_self",
        "random_a_batch",
        "deranged_a_self",
        "deranged_a_batch",
        "head_a_batch",
        "head_a_self",
    )
    if any(abs(float(row[name])) > 1.0 + 1.0e-12 for name in cosine_names) or any(
        abs(float(value)) > 1.0 + 1.0e-12 for value in row["support_cosines"]
    ):
        raise ValueError("receiver audit cosine lies outside [-1,1]")
    if float(row["rho"]) < 0.0 or float(row["rho"]) > 1.0 + 1.0e-12:
        raise ValueError("receiver audit rho lies outside [0,1]")
    identities = (
        ("delta", "a_self", "a_batch"),
        ("self_minus_desc", "a_self", "a_desc"),
        ("random_delta", "random_a_self", "random_a_batch"),
        ("deranged_delta", "deranged_a_self", "deranged_a_batch"),
    )
    if any(
        abs(float(row[result]) - (float(row[left]) - float(row[right]))) > 1.0e-12
        for result, left, right in identities
    ):
        raise ValueError("receiver audit signed difference is inconsistent")
    json.dumps(_json_ready(dict(row)), allow_nan=False)


def _validate_registered_rows(
    primary_rows: Sequence[Mapping[str, Any]],
    alternate_rows: Sequence[Mapping[str, Any]],
    seed_audits: Sequence[Mapping[str, Any]],
    panel_binding: Mapping[str, Any],
) -> None:
    primary = panel_binding.get("primary")
    alternate = panel_binding.get("alternate")
    tensor_hashes = panel_binding.get("tensor_sha256")
    expected_dimension = panel_binding.get("expected_dimension")
    foreign_support_ids = set(panel_binding.get("foreign_support_ids", ()))
    if not all(isinstance(value, Mapping) for value in (primary, alternate, tensor_hashes)):
        raise ValueError("registered panel binding is incomplete")
    expected_panels = {"primary": primary, "alternate": alternate}
    rows_by_panel = {"primary": primary_rows, "alternate": alternate_rows}
    primary_tensors: dict[tuple[int, str], str] = {}
    for panel_name, rows in rows_by_panel.items():
        selected = expected_panels[panel_name]
        expected_count = 64 if panel_name == "primary" else 16
        by_seed = {seed: [] for seed in range(4)}
        for row in rows:
            by_seed[int(row["seed"])].append(row)
        for seed in range(4):
            if len(by_seed[seed]) != expected_count:
                raise ValueError("registered panel row count differs")
            ordered = sorted(by_seed[seed], key=lambda row: selected["labels"].index(row["label"]))
            for position, row in enumerate(ordered):
                expected_id = selected["receiver_ids"][position]
                batch_index = position // 8
                expected_batch = selected["batches"][batch_index]
                if (
                    row["receiver_id"] != expected_id
                    or row["batch_index"] != batch_index
                    or row["batch_ids"] != expected_batch
                    or row["support_ids"] != primary["support_ids_by_label"][row["label"]]
                ):
                    raise ValueError("receiver row differs from registered roles or batches")
                if not set(row["foreign_ids"]) <= foreign_support_ids:
                    raise ValueError("receiver foreign IDs differ from registered rank-0 supports")
                if row["tensor_sha256"] != tensor_hashes[expected_id]:
                    raise ValueError("receiver SHA-256 differs from registered tensor")
                if row["batch_tensor_sha256"] != [tensor_hashes[value] for value in expected_batch]:
                    raise ValueError("batch tensor SHA-256 list differs from registered tensors")
                if panel_name == "primary":
                    primary_tensors[(seed, expected_id)] = row["tensor_sha256"]
                elif row["tensor_sha256"] != primary_tensors[(seed, expected_id)]:
                    raise ValueError("alternate receiver tensor differs from its primary tensor")
    for audit in seed_audits:
        config = audit["config"]
        if (
            config.get("batch_size") != 180
            or config.get("embedding_dimensions") != expected_dimension
            or config.get("proxy_anchor_alpha") != 32.0
            or config.get("proxy_anchor_delta") != 0.1
        ):
            raise ValueError("seed audit config differs from registered scientific execution")
        audit_hashes = (
            "proxy_sha256",
            "proxy_label_sha256",
            "train_example_id_order_sha256",
            "train_label_order_sha256",
            "train_source_order_sha256",
            "transform_cache_order_sha256",
        )
        if any(
            not isinstance(audit[name], str)
            or len(audit[name]) != 64
            or any(character not in "0123456789abcdef" for character in audit[name])
            for name in audit_hashes
        ):
            raise ValueError("seed audit SHA-256 must be lowercase hex")
        if (
            audit["primary_batch_ids"] != primary["batches"]
            or audit["alternate_batch_ids"] != alternate["batches"]
        ):
            raise ValueError("seed audit batch matrices differ from registered panels")
        observed = audit["transform_tensor_sha256"]
        for batch in [*primary["batches"], *alternate["batches"]]:
            for example_id in batch:
                if observed.get(example_id) != tensor_hashes[example_id]:
                    raise ValueError("seed audit transform/tensor SHA-256 differs")


def _validate_fixture_integrity(integrity: Mapping[str, Any]) -> None:
    """Enforce the frozen dense-Jacobian and bufferless-BN fixture gates."""
    dense = integrity.get("dense_fixture")
    if not isinstance(dense, Mapping) or dense.get("passed") is not True:
        raise ValueError("integrity gate failed: dense_fixture")
    if (
        float(dense.get("jacobian_tolerance", float("nan"))) != 1.0e-8
        or float(dense.get("finite_difference_tolerance", float("nan"))) != 1.0e-6
    ):
        raise ValueError("dense fixture tolerances differ from the frozen contract")
    dense_residuals = (
        (dense.get("max_jacobian_residual"), 1.0e-8),
        (dense.get("max_finite_difference_residual"), 1.0e-6),
    )
    if any(
        value is None
        or not np.isfinite(float(value))
        or float(value) < 0.0
        or float(value) > tolerance
        for value, tolerance in dense_residuals
    ):
        raise ValueError("dense fixture residual exceeds the frozen tolerance")
    bn = integrity.get("bn_fixture")
    if not isinstance(bn, Mapping) or bn.get("passed") is not True:
        raise ValueError("integrity gate failed: bn_fixture")
    if float(bn.get("tolerance", float("nan"))) != 1.0e-6:
        raise ValueError("BN fixture tolerance differs from the frozen contract")
    if bn.get("buffers_unchanged") is not True or any(
        value is None
        or not np.isfinite(float(value))
        or float(value) < 0.0
        or float(value) > 1.0e-6
        for value in (bn.get("max_output_residual"), bn.get("max_gradient_residual"))
    ):
        raise ValueError("BN fixture residual or buffer audit failed")


def scientific_payload(
    *,
    manifest_audit: Mapping[str, Any],
    execution_audit: Mapping[str, Any],
    environment: Mapping[str, Any],
    seed_audits: Sequence[Mapping[str, Any]],
    primary_rows: Sequence[Mapping[str, Any]],
    alternate_rows: Sequence[Mapping[str, Any]],
    integrity: Mapping[str, Any],
    aggregation: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    panel_binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the non-lossy scientific result contract; aggregate-only output is forbidden."""
    manifest_path = manifest_audit.get("path")
    manifest_source = manifest_audit.get("source")
    if not isinstance(manifest_path, str) or not isinstance(manifest_source, Mapping):
        raise ValueError("scientific manifest audit lacks execution-binding metadata")
    validate_execution_audit(
        execution_audit,
        manifest_source=manifest_source,
        manifest_path=Path(manifest_path),
    )
    if ENVIRONMENT_AUDIT_FIELDS - set(environment):
        raise ValueError("scientific environment audit is incomplete")
    if (
        environment.get("cublas_workspace_config") != ":4096:8"
        or environment.get("deterministic_algorithms") is not True
        or environment.get("deterministic_warn_only") is not False
        or environment.get("cudnn_benchmark") is not False
        or environment.get("cuda_matmul_tf32") is not False
        or environment.get("cudnn_tf32") is not False
        or environment.get("autocast") is not False
        or environment.get("model_arithmetic") != "float32"
        or environment.get("reduction_arithmetic") != "float64"
    ):
        raise ValueError("scientific environment differs from the frozen deterministic gates")
    if not primary_rows or not alternate_rows:
        raise ValueError("scientific result requires primary and alternate receiver rows")
    for row in primary_rows:
        _validate_receiver_audit_row(row, expected_panel="primary")
    for row in alternate_rows:
        _validate_receiver_audit_row(row, expected_panel="alternate")
    _validate_fixture_integrity(integrity)
    zero_jacobian = _require_exact_keys(
        integrity.get("zero_jacobian_classifier"),
        {"0", "1", "2", "3"},
        name="zero-Jacobian classifier scientific audits",
    )
    for seed in ("0", "1", "2", "3"):
        _validate_zero_jacobian_classifier_audit(zero_jacobian[seed])
    integrity_seeds = integrity.get("seeds")
    if not isinstance(integrity_seeds, Sequence) or {
        audit.get("seed") for audit in integrity_seeds if isinstance(audit, Mapping)
    } != {0, 1, 2, 3}:
        raise ValueError("integrity requires exactly seeds 0-3")
    for audit in integrity_seeds:
        repeatability = audit.get("repeatability") if isinstance(audit, Mapping) else None
        if not isinstance(repeatability, Mapping) or set(repeatability) != {"z", "dbar", "b", "s"}:
            raise ValueError("integrity repeatability gate failed")
        for hashes in repeatability.values():
            if (
                not isinstance(hashes, Mapping)
                or hashes.get("first_sha256") != hashes.get("repeat_sha256")
                or not isinstance(hashes.get("first_sha256"), str)
                or len(hashes["first_sha256"]) != 64
                or any(character not in "0123456789abcdef" for character in hashes["first_sha256"])
            ):
                raise ValueError("integrity repeatability hashes differ")
        adjoint = float(audit.get("adjoint_relative_error", float("inf")))
        if not np.isfinite(adjoint) or adjoint > 5.0e-4:
            raise ValueError("integrity gate failed: adjoint relative error")
        if not isinstance(audit.get("rotation"), Mapping):
            raise ValueError("integrity rotation gate is missing")
        rotation = audit["rotation"]
        if (
            set(rotation.get("vector_residuals", {})) != _ROTATION_VECTOR_NAMES
            or set(rotation.get("statistic_differences", {})) != _ROTATION_STATISTIC_NAMES
        ):
            raise ValueError("integrity rotation audit lacks every registered name")
        if any(
            not np.isfinite(float(value)) or float(value) < 0.0 or float(value) > 5.0e-4
            for value in rotation["vector_residuals"].values()
        ) or any(
            not np.isfinite(float(value)) or float(value) < 0.0 or float(value) > 2.0e-4
            for value in rotation["statistic_differences"].values()
        ):
            raise ValueError("integrity rotation residual exceeds registered tolerance")
    if {audit.get("seed") for audit in seed_audits} != {0, 1, 2, 3} or any(
        SEED_AUDIT_FIELDS - set(audit)
        or not audit.get("parameter_names")
        or int(audit.get("parameter_count", 0)) <= 0
        or len(audit.get("primary_batch_ids", ())) != 8
        or any(len(batch) != 180 for batch in audit.get("primary_batch_ids", ()))
        or len(audit.get("alternate_batch_ids", ())) != 2
        or any(len(batch) != 180 for batch in audit.get("alternate_batch_ids", ()))
        for audit in seed_audits
    ):
        raise ValueError("scientific result requires ordered parameter names for every seed")
    reference_names = tuple(seed_audits[0]["parameter_names"])
    reference_count = int(seed_audits[0]["parameter_count"])
    if any(
        tuple(audit["parameter_names"]) != reference_names
        or int(audit["parameter_count"]) != reference_count
        for audit in seed_audits[1:]
    ):
        raise ValueError("ordered encoder parameter names/count differ across seeds")
    _validate_registered_rows(primary_rows, alternate_rows, seed_audits, panel_binding)
    recomputed = decide_stage_a(primary_rows, alternate_rows)
    if _json_ready(aggregation) != _json_ready(recomputed):
        raise ValueError("scientific aggregation differs from the persisted receiver rows")
    delta_distribution = np.asarray(bootstrap.get("delta_distribution"), dtype=np.float64)
    self_desc_distribution = np.asarray(
        bootstrap.get("self_minus_desc_distribution"), dtype=np.float64
    )
    if delta_distribution.shape != (10_000,) or self_desc_distribution.shape != (10_000,):
        raise ValueError("scientific bootstrap must persist both 10,000-replicate distributions")
    if (
        float64_c_order_sha256(delta_distribution) != bootstrap.get("delta_sha256")
        or float64_c_order_sha256(self_desc_distribution) != bootstrap.get("self_minus_desc_sha256")
        or bootstrap.get("delta_sha256") != recomputed["bootstrap_delta_sha256"]
        or bootstrap.get("self_minus_desc_sha256") != recomputed["bootstrap_self_desc_sha256"]
    ):
        raise ValueError("scientific bootstrap distribution hash differs from aggregation")
    stage_a = recomputed["stage_a"]
    clause = recomputed["first_decisive_clause"]
    payload = {
        "schema_version": 1,
        "diagnostic": "pass200_rsta_stage_a",
        "mode": "scientific",
        "candidate_values_computed": True,
        "stage_a_verdict": stage_a,
        "first_decisive_clause": clause,
        "uses_test_data": False,
        "scope_limitation": (
            "Euclidean tangent kernel; invariant only to the registered common descriptor "
            "rotation, not hidden-layer rescaling or AdamW preconditioning"
        ),
        "manifest": _json_ready(manifest_audit),
        "execution_audit": _json_ready(execution_audit),
        "environment": _json_ready(environment),
        "integrity": _json_ready(integrity),
        "seed_audits": _json_ready(seed_audits),
        "panel_binding": _json_ready(panel_binding),
        "rows": {
            "primary": _json_ready(primary_rows),
            "alternate": _json_ready(alternate_rows),
        },
        "exclusions": [],
        "bootstrap": _json_ready(bootstrap),
        "aggregation": _json_ready(aggregation),
    }
    json.dumps(payload, allow_nan=False)
    return payload


def domain_hash(domain: str, text: str) -> bytes:
    """Return ``SHA256(domain.encode('ascii') + NUL + text.encode('utf-8'))``."""
    if not isinstance(domain, str) or not isinstance(text, str):
        raise TypeError("domain and text must be strings")
    return hashlib.sha256(domain.encode("ascii") + b"\0" + text.encode("utf-8")).digest()


def domain_seed(domain: str, text: str) -> int:
    """Extract the registered unsigned big-endian seed from a domain hash."""
    return int.from_bytes(domain_hash(domain, text)[:8], byteorder="big", signed=False)


def _canonical_inputs(
    example_ids: Sequence[str], labels: Sequence[int]
) -> tuple[list[str], list[int]]:
    ids = list(example_ids)
    raw_labels = list(labels)
    if not ids or len(ids) != len(raw_labels):
        raise ValueError("example IDs and labels must be nonempty and aligned")
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("every example ID must be a nonempty string")
    if len(set(ids)) != len(ids):
        raise ValueError("example IDs must be unique")
    canonical_labels: list[int] = []
    for value in raw_labels:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
            raise ValueError("identity labels must be unsigned integers")
        label = int(value)
        if label < 0:
            raise ValueError("identity labels must be unsigned integers")
        canonical_labels.append(label)
    return ids, canonical_labels


def _identity_roles(
    example_ids: list[str], labels: list[int]
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    rows_by_label: dict[int, list[str]] = {}
    for example_id, label in zip(example_ids, labels, strict=True):
        rows_by_label.setdefault(label, []).append(example_id)
    eligible_roles: dict[int, list[str]] = {}
    for label, row_ids in rows_by_label.items():
        if len(row_ids) >= 3:
            eligible_roles[label] = sorted(
                row_ids, key=lambda value: (domain_hash(_ROLE_DOMAIN, value), value)
            )
    return rows_by_label, eligible_roles


def _ordered_batch(receiver_ids: list[str], distractor_ids: list[str], batch: int) -> list[str]:
    if len(receiver_ids) != _RECEIVERS_PER_BATCH:
        raise ValueError("each diagnostic batch requires exactly eight receivers")
    if len(distractor_ids) != _DISTRACTORS_PER_BATCH:
        raise ValueError("each diagnostic batch requires exactly 172 distractors")
    domain = f"{_BATCH_ORDER_PREFIX}{batch}|"
    combined = receiver_ids + distractor_ids
    if len(set(combined)) != _RECEIVERS_PER_BATCH + _DISTRACTORS_PER_BATCH:
        raise ValueError("diagnostic batch rows must be unique")
    return sorted(combined, key=lambda value: (domain_hash(domain, value), value))


def select_primary_panel(example_ids: Sequence[str], labels: Sequence[int]) -> dict[str, Any]:
    """Select frozen roles, 64 receivers, and eight official-size batches."""
    ids, canonical_labels = _canonical_inputs(example_ids, labels)
    rows_by_label, roles = _identity_roles(ids, canonical_labels)
    ordered_eligible = sorted(
        roles,
        key=lambda label: (domain_hash(_IDENTITY_DOMAIN, str(label)), label),
    )
    if len(ordered_eligible) < _PRIMARY_IDENTITIES:
        raise ValueError("at least 64 identities with three rows are required")
    selected_labels = ordered_eligible[:_PRIMARY_IDENTITIES]
    selected_set = set(selected_labels)
    receiver_ids = [roles[label][2] for label in selected_labels]
    groups = [
        selected_labels[start : start + _RECEIVERS_PER_BATCH]
        for start in range(0, _PRIMARY_IDENTITIES, _RECEIVERS_PER_BATCH)
    ]
    distractor_ids = [
        example_id
        for example_id, label in zip(ids, canonical_labels, strict=True)
        if label not in selected_set
    ]
    distractor_ids.sort(key=lambda value: (domain_hash(_DISTRACTOR_DOMAIN, value), value))
    needed = len(groups) * _DISTRACTORS_PER_BATCH
    if len(distractor_ids) < needed:
        raise ValueError("not enough nonselected rows for primary distractors")
    distractor_blocks = [
        distractor_ids[start : start + _DISTRACTORS_PER_BATCH]
        for start in range(0, needed, _DISTRACTORS_PER_BATCH)
    ]
    batches = [
        _ordered_batch(receiver_ids[batch * 8 : (batch + 1) * 8], block, batch)
        for batch, block in enumerate(distractor_blocks)
    ]
    support_ids_by_label = {label: roles[label][:2] for label in ordered_eligible}
    return {
        "eligible_labels": ordered_eligible,
        "labels": selected_labels,
        "receiver_ids": receiver_ids,
        "groups": groups,
        "support_ids_by_label": support_ids_by_label,
        "distractor_blocks": distractor_blocks,
        "batches": batches,
        "rows_by_label": rows_by_label,
    }


def select_alternate_panel(
    example_ids: Sequence[str],
    labels: Sequence[int],
    primary_panel: dict[str, Any],
) -> dict[str, Any]:
    """Build the frozen two-batch alternate contexts from a primary panel."""
    ids, canonical_labels = _canonical_inputs(example_ids, labels)
    canonical_primary = select_primary_panel(ids, canonical_labels)
    for name in ("labels", "receiver_ids", "distractor_blocks", "batches"):
        if primary_panel.get(name) != canonical_primary[name]:
            raise ValueError("alternate selection requires the canonical primary panel")
    _, roles = _identity_roles(ids, canonical_labels)
    primary_labels = list(primary_panel.get("labels", []))
    primary_receivers = list(primary_panel.get("receiver_ids", []))
    primary_blocks = list(primary_panel.get("distractor_blocks", []))
    if len(primary_labels) != 64 or len(primary_receivers) != 64 or len(primary_blocks) != 8:
        raise ValueError("alternate selection requires a complete primary panel")
    if len(set(primary_labels)) != 64 or len(set(primary_receivers)) != 64:
        raise ValueError("primary receivers and labels must be unique")
    for label, receiver_id in zip(primary_labels, primary_receivers, strict=True):
        if label not in roles or roles[label][2] != receiver_id:
            raise ValueError("primary receiver roles do not match the supplied rows")
    selected_positions = [index for index in range(64) if index % 8 in (0, 1)]
    alternate_labels = [primary_labels[index] for index in selected_positions]
    alternate_receivers = [primary_receivers[index] for index in selected_positions]
    alternate_set = set(alternate_labels)
    excluded = {value for block in primary_blocks for value in block}
    if len(excluded) != 8 * _DISTRACTORS_PER_BATCH:
        raise ValueError("primary distractors must be complete and nonoverlapping")
    excluded.update(value for role_ids in roles.values() for value in role_ids[:2])
    excluded.update(
        example_id
        for example_id, label in zip(ids, canonical_labels, strict=True)
        if label in alternate_set
    )
    candidates = [value for value in ids if value not in excluded]
    candidates.sort(key=lambda value: (domain_hash(_ALTERNATE_DISTRACTOR_DOMAIN, value), value))
    needed = 2 * _DISTRACTORS_PER_BATCH
    if len(candidates) < needed:
        raise ValueError("not enough rows for alternate distractors")
    blocks = [
        candidates[start : start + _DISTRACTORS_PER_BATCH]
        for start in range(0, needed, _DISTRACTORS_PER_BATCH)
    ]
    groups = [alternate_labels[:8], alternate_labels[8:]]
    batches = [
        _ordered_batch(alternate_receivers[batch * 8 : (batch + 1) * 8], block, batch)
        for batch, block in enumerate(blocks)
    ]
    return {
        "labels": alternate_labels,
        "receiver_ids": alternate_receivers,
        "groups": groups,
        "distractor_blocks": blocks,
        "batches": batches,
    }


def _unit_vector(vector: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite unit vector")
    norm = float(np.linalg.norm(value))
    if norm <= _VECTOR_EPS or abs(norm - 1.0) > 2.0e-5:
        raise ValueError(f"{name} must be a finite unit vector")
    return value / norm


def _unit_matrix(rows: np.ndarray, *, name: str) -> np.ndarray:
    value = np.asarray(rows, dtype=np.float64)
    if value.ndim != 2 or value.shape[0] == 0 or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a nonempty finite matrix")
    norms = np.linalg.norm(value, axis=1)
    if np.any(norms <= _VECTOR_EPS) or np.any(np.abs(norms - 1.0) > 2.0e-5):
        raise ValueError(f"{name} must contain finite unit rows")
    return value / norms[:, None]


def tangent_projection(vector: np.ndarray, receiver: np.ndarray) -> np.ndarray:
    """Project a vector into the receiver descriptor tangent, rejecting zero."""
    z = _unit_vector(receiver, name="receiver")
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != z.shape or not np.isfinite(value).all():
        raise ValueError("vector must be finite and aligned with receiver")
    projected = value - z * float(np.dot(z, value))
    if float(np.linalg.norm(projected)) <= _VECTOR_EPS:
        raise ValueError("tangent projection has zero norm")
    return projected


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    weights = np.exp(shifted)
    return weights / float(weights.sum())


def smooth_margin_gradient(
    receiver: np.ndarray,
    positive_supports: np.ndarray,
    frozen_foreign_supports: np.ndarray,
    *,
    tau: float = 0.05,
) -> np.ndarray:
    """Return the tangent ascent of the registered frozen smooth margin."""
    z = _unit_vector(receiver, name="receiver")
    positives = _unit_matrix(positive_supports, name="positive supports")
    foreign = _unit_matrix(frozen_foreign_supports, name="foreign supports")
    if positives.shape != (2, z.size):
        raise ValueError("smooth margin requires exactly two positive supports")
    if foreign.shape != (32, z.size):
        raise ValueError("smooth margin requires exactly 32 foreign supports")
    if not np.isfinite(tau) or tau <= 0.0:
        raise ValueError("tau must be finite and positive")
    ambient = _softmax((positives @ z) / tau) @ positives
    ambient -= _softmax((foreign @ z) / tau) @ foreign
    return tangent_projection(ambient, z)


def select_foreign_supports(
    receiver: np.ndarray,
    *,
    receiver_label: int,
    support_ids: Sequence[str],
    support_labels: Sequence[int],
    support_descriptors: np.ndarray,
    current_batch_ids: set[str] | frozenset[str],
) -> tuple[list[str], np.ndarray]:
    """Freeze the 32 largest foreign receiver-view cosines with registered ties."""
    ids, labels = _canonical_inputs(support_ids, support_labels)
    if len(set(labels)) != len(labels):
        raise ValueError("foreign pool requires exactly one rank-0 support per identity")
    if (
        isinstance(receiver_label, bool)
        or not isinstance(receiver_label, (int, np.integer))
        or int(receiver_label) < 0
    ):
        raise ValueError("receiver label must be an unsigned integer")
    if not isinstance(current_batch_ids, (set, frozenset)) or any(
        not isinstance(value, str) for value in current_batch_ids
    ):
        raise ValueError("current batch IDs must be a set of strings")
    z = _unit_vector(receiver, name="receiver")
    supports = _unit_matrix(support_descriptors, name="foreign support candidates")
    if supports.shape != (len(ids), z.size):
        raise ValueError("foreign support IDs, labels, and descriptors must align")
    eligible = [
        index
        for index, (example_id, label) in enumerate(zip(ids, labels, strict=True))
        if label != int(receiver_label) and example_id not in current_batch_ids
    ]
    if len(eligible) < 32:
        raise ValueError("foreign pool must contain at least 32 eligible supports")
    similarities = supports @ z
    ordered = sorted(
        eligible,
        key=lambda index: (
            -float(similarities[index]),
            domain_hash(_ROLE_DOMAIN, ids[index]),
            ids[index],
        ),
    )[:32]
    return [ids[index] for index in ordered], supports[ordered]


def deranged_tangent_targets(receivers: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Shift targets by +1 within one eight-receiver group and reproject."""
    z = _unit_matrix(receivers, name="receivers")
    values = np.asarray(targets, dtype=np.float64)
    if z.shape[0] != 8 or values.shape != z.shape or not np.isfinite(values).all():
        raise ValueError("derangement requires aligned arrays for eight receivers")
    result = np.empty_like(values)
    for index in range(8):
        projected = tangent_projection(values[(index + 1) % 8], z[index])
        result[index] = projected / float(np.linalg.norm(projected))
    return result


def random_tangent_target(
    receiver: np.ndarray,
    *,
    seed: int,
    example_id: str,
    target_norm: float,
) -> np.ndarray:
    """Draw the registered fresh PCG64 tangent-random negative control."""
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or int(seed) < 0:
        raise ValueError("seed must be an unsigned integer")
    if not isinstance(example_id, str) or not example_id:
        raise ValueError("example_id must be a nonempty string")
    if not np.isfinite(target_norm) or target_norm <= _VECTOR_EPS:
        raise ValueError("target_norm must be finite and positive")
    z = _unit_vector(receiver, name="receiver")
    random_seed = domain_seed("rsta-stage-a-v1|random-target|", f"{int(seed)}\0{example_id}")
    random = np.random.Generator(np.random.PCG64(random_seed)).standard_normal(z.size)
    projected = tangent_projection(random, z)
    return projected * (float(target_norm) / float(np.linalg.norm(projected)))


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Return a signed cosine, rejecting nonfinite or zero inputs."""
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.ndim != 1 or second.shape != first.shape:
        raise ValueError("cosine inputs must be aligned vectors")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise ValueError("cosine inputs must be finite")
    first_norm = float(np.linalg.norm(first))
    second_norm = float(np.linalg.norm(second))
    if first_norm <= _VECTOR_EPS or second_norm <= _VECTOR_EPS:
        raise ValueError("cosine inputs must have nonzero norm")
    denominator = first_norm * second_norm
    return float(np.dot(first, second) / denominator)


def head_only_kernel_motion(
    prehead_features: np.ndarray,
    head_outputs: np.ndarray,
    cotangents: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the analytic normalized affine-head kernel to all cotangents."""
    features = np.asarray(prehead_features)
    outputs = np.asarray(head_outputs)
    directions = np.asarray(cotangents)
    if (
        not all(
            np.issubdtype(value.dtype, np.floating) for value in (features, outputs, directions)
        )
        or len({features.dtype, outputs.dtype, directions.dtype}) != 1
    ):
        raise ValueError("head arrays must share one floating model dtype")
    if features.ndim != 2 or outputs.ndim != 2 or directions.shape != outputs.shape:
        raise ValueError("head arrays must be aligned nonempty matrices")
    if features.shape[0] == 0 or features.shape[0] != outputs.shape[0]:
        raise ValueError("head arrays must have the same nonzero batch dimension")
    if not all(np.isfinite(value).all() for value in (features, outputs, directions)):
        raise ValueError("head arrays must be finite")
    raw_norms = np.linalg.norm(outputs, axis=1)
    if np.any(raw_norms <= _VECTOR_EPS):
        raise ValueError("head outputs must have nonzero norms")
    descriptors = outputs / raw_norms[:, None]
    projected_directions = directions - descriptors * np.sum(
        descriptors * directions, axis=1, keepdims=True
    )
    first_actions = projected_directions / raw_norms[:, None]
    coefficients = features @ features.T + 1.0
    batch_motion = np.empty_like(outputs)
    self_motion = np.empty_like(outputs)
    for receiver in range(features.shape[0]):
        after_receiver_projection = (
            first_actions - descriptors[receiver] * (first_actions @ descriptors[receiver])[:, None]
        )
        contributions = after_receiver_projection / raw_norms[receiver]
        batch_motion[receiver] = coefficients[receiver] @ contributions
        self_motion[receiver] = coefficients[receiver, receiver] * contributions[receiver]
        if cosine_similarity(self_motion[receiver], directions[receiver]) < 1.0 - 1.0e-5:
            raise ValueError("head-self motion is not positively collinear with cotangent")
    return batch_motion, self_motion


def score_rsta_batch(
    *,
    seed: int,
    panel: str,
    batch_index: int,
    receiver_indices: Sequence[int],
    receiver_ids: Sequence[str],
    receiver_labels: Sequence[int],
    batch_ids: Sequence[str],
    tensor_hashes: Mapping[str, str],
    fields: Mapping[str, Any],
    supports_by_label: Mapping[int, tuple[Sequence[str], np.ndarray]],
    foreign_ids: Sequence[str],
    foreign_labels: Sequence[int],
    foreign_descriptors: np.ndarray,
    prehead_features: np.ndarray,
    raw_head_outputs: np.ndarray,
) -> list[dict[str, Any]]:
    """Score one registered eight-receiver context and retain every audit input/output."""
    if panel not in {"primary", "alternate"}:
        raise ValueError("RSTA panel must be primary or alternate")
    receivers = tuple(int(value) for value in receiver_indices)
    ids = tuple(str(value) for value in receiver_ids)
    labels = tuple(int(value) for value in receiver_labels)
    ordered_batch_ids = tuple(str(value) for value in batch_ids)
    if len(receivers) != 8 or len(ids) != 8 or len(labels) != 8:
        raise ValueError("RSTA scoring requires exactly eight aligned receivers")
    if fields.get("receiver_indices") != receivers:
        raise ValueError("field receiver order differs from registered receiver order")
    if len(ordered_batch_ids) != 180 or len(set(ordered_batch_ids)) != 180:
        raise ValueError("RSTA scoring requires one unique ordered B=180 batch")
    z_all = np.asarray(fields["z"].detach().cpu(), dtype=np.float64)
    dbar_all = np.asarray(fields["dbar"].detach().cpu(), dtype=np.float64)
    b_all = np.asarray(fields["batch_motion"].detach().cpu(), dtype=np.float64)
    s_rows = np.asarray(fields["self_motion"].detach().cpu(), dtype=np.float64)
    if s_rows.shape != (8, z_all.shape[1]):
        raise ValueError("self-motion rows differ from the registered receivers")
    selected = np.asarray(receivers, dtype=np.int64)
    validated = project_and_validate_fields(
        z_all[selected],
        dbar=dbar_all[selected],
        b=b_all[selected],
        s=s_rows,
    )
    z_rows = np.asarray(validated["z"].detach().cpu(), dtype=np.float64)
    dbar_rows = np.asarray(validated["dbar"].detach().cpu(), dtype=np.float64)
    b_rows = np.asarray(validated["b"].detach().cpu(), dtype=np.float64)
    s_rows = np.asarray(validated["s"].detach().cpu(), dtype=np.float64)
    radial = {
        name: np.asarray(value.detach().cpu(), dtype=np.float64)
        for name, value in validated["radial_fractions"].items()
    }
    q_rows: list[np.ndarray] = []
    chosen_foreign_ids: list[list[str]] = []
    chosen_foreign_rows: list[np.ndarray] = []
    positive_ids: list[tuple[str, str]] = []
    positive_rows: list[np.ndarray] = []
    current_batch = set(ordered_batch_ids)
    for position, label in enumerate(labels):
        if label not in supports_by_label:
            raise ValueError(f"missing clean supports for receiver label {label}")
        raw_support_ids, raw_positive = supports_by_label[label]
        support_ids = tuple(str(value) for value in raw_support_ids)
        positives = _unit_matrix(raw_positive, name="positive supports")
        if len(support_ids) != 2 or positives.shape != (2, z_rows.shape[1]):
            raise ValueError("each receiver requires exactly two aligned clean supports")
        selected_ids, selected_foreign = select_foreign_supports(
            z_rows[position],
            receiver_label=label,
            support_ids=foreign_ids,
            support_labels=foreign_labels,
            support_descriptors=foreign_descriptors,
            current_batch_ids=current_batch,
        )
        q_rows.append(smooth_margin_gradient(z_rows[position], positives, selected_foreign))
        positive_ids.append((support_ids[0], support_ids[1]))
        positive_rows.append(positives)
        chosen_foreign_ids.append(selected_ids)
        chosen_foreign_rows.append(selected_foreign)
    q_matrix = np.stack(q_rows)
    deranged = deranged_tangent_targets(z_rows, q_matrix)
    head_batch, head_self = head_only_kernel_motion(
        np.asarray(prehead_features),
        np.asarray(raw_head_outputs),
        np.asarray(fields["dbar"].detach().cpu()),
    )
    rows: list[dict[str, Any]] = []
    for position, receiver in enumerate(receivers):
        q = q_matrix[position]
        z = z_rows[position]
        dbar = dbar_rows[position]
        b = b_rows[position]
        s = s_rows[position]
        b_norm = float(np.linalg.norm(b))
        s_norm = float(np.linalg.norm(s))
        b_unit = b / b_norm
        s_unit = s / s_norm
        a_self = cosine_similarity(s, q)
        a_batch = cosine_similarity(b, q)
        a_desc = cosine_similarity(dbar, q)
        random_target = random_tangent_target(
            z,
            seed=int(seed),
            example_id=ids[position],
            target_norm=s_norm,
        )
        random_a_self = cosine_similarity(s, random_target)
        random_a_batch = cosine_similarity(b, random_target)
        deranged_a_self = cosine_similarity(s, deranged[position])
        deranged_a_batch = cosine_similarity(b, deranged[position])
        b_head = head_batch[receiver]
        s_head = head_self[receiver]
        head_a_self = cosine_similarity(s_head, q)
        head_a_batch = cosine_similarity(b_head, q)
        head_self_desc_gap = 1.0 - cosine_similarity(s_head, dbar)
        if head_self_desc_gap > 1.0e-5:
            raise ValueError("head-only self motion failed positive-collinearity gate")
        supports = np.concatenate((positive_rows[position], chosen_foreign_rows[position]), axis=0)
        row = {
            "panel": panel,
            "seed": int(seed),
            "label": labels[position],
            "batch_index": int(batch_index),
            "receiver_index": receiver,
            "receiver_id": ids[position],
            "support_ids": list(positive_ids[position]),
            "foreign_ids": chosen_foreign_ids[position],
            "batch_ids": list(ordered_batch_ids),
            "batch_tensor_sha256": [str(tensor_hashes[value]) for value in ordered_batch_ids],
            "batch_id_order_sha256": _ordered_text_sha256(ordered_batch_ids),
            "tensor_sha256": str(tensor_hashes[ids[position]]),
            "a_self": a_self,
            "a_batch": a_batch,
            "delta": a_self - a_batch,
            "a_desc": a_desc,
            "self_minus_desc": a_self - a_desc,
            "rho": float(np.linalg.norm(b_unit - s_unit * float(np.dot(s_unit, b_unit)))),
            "log_ratio": float(np.log((b_norm + _VECTOR_EPS) / (s_norm + _VECTOR_EPS))),
            "cos_b_s": cosine_similarity(b, s),
            "random_a_self": random_a_self,
            "random_a_batch": random_a_batch,
            "random_delta": random_a_self - random_a_batch,
            "deranged_a_self": deranged_a_self,
            "deranged_a_batch": deranged_a_batch,
            "deranged_delta": deranged_a_self - deranged_a_batch,
            "norm_z": float(np.linalg.norm(z)),
            "norm_dbar": float(np.linalg.norm(dbar)),
            "norm_b": b_norm,
            "norm_s": s_norm,
            "norm_q": float(np.linalg.norm(q)),
            "norm_random_target": float(np.linalg.norm(random_target)),
            "norm_deranged_target": float(np.linalg.norm(deranged[position])),
            "radial_fraction_dbar": float(radial["dbar"][position]),
            "radial_fraction_b": float(radial["b"][position]),
            "radial_fraction_s": float(radial["s"][position]),
            "head_a_batch": head_a_batch,
            "head_a_self": head_a_self,
            "head_self_desc_gap": head_self_desc_gap,
            "norm_b_head": float(np.linalg.norm(b_head)),
            "norm_s_head": float(np.linalg.norm(s_head)),
            "support_cosines": [float(value) for value in supports @ z],
        }
        _validate_receiver_audit_row(row, expected_panel=panel)
        rows.append(row)
    return rows


def construct_rotation(
    dimension: int = 512,
    *,
    seed: int = 200,
    dtype: np.dtype[Any] | type[np.floating[Any]] = np.float64,
) -> np.ndarray:
    """Construct the registered dense orthogonal descriptor rotation."""
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
        raise ValueError("rotation dimension must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("rotation seed must be an unsigned integer")
    matrix = np.ascontiguousarray(
        np.random.Generator(np.random.PCG64(seed)).standard_normal((dimension, dimension))
    )
    rotation, triangular = np.linalg.qr(matrix)
    signs = np.where(np.diag(triangular) < 0.0, -1.0, 1.0)
    rotation = rotation * signs
    return rotation.astype(dtype, copy=False)


def check_rotation(
    vectors: dict[str, np.ndarray],
    rotated_vectors: dict[str, np.ndarray],
    statistics: dict[str, float],
    rotated_statistics: dict[str, float],
    rotation: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Check the registered vector and scalar rotation-equivariance gates."""
    raw_rotation = np.asarray(rotation)
    q = np.asarray(raw_rotation, dtype=np.float64)
    if q.ndim != 2 or q.shape[0] != q.shape[1] or not np.isfinite(q).all():
        raise ValueError("rotation must be a finite square matrix")
    orthogonality_tolerance = 5.0e-5 if raw_rotation.dtype.itemsize <= 4 else 1.0e-10
    if not np.allclose(
        q.T @ q,
        np.eye(q.shape[0]),
        atol=orthogonality_tolerance,
        rtol=orthogonality_tolerance,
    ):
        raise ValueError("rotation must be orthogonal")
    if (
        set(vectors) != _ROTATION_VECTOR_NAMES
        or set(rotated_vectors) != _ROTATION_VECTOR_NAMES
        or set(statistics) != _ROTATION_STATISTIC_NAMES
        or set(rotated_statistics) != _ROTATION_STATISTIC_NAMES
    ):
        raise ValueError("rotation check requires all registered names")
    vector_residuals: dict[str, float] = {}
    for name, vector in vectors.items():
        original = np.asarray(vector, dtype=np.float64)
        observed = np.asarray(rotated_vectors[name], dtype=np.float64)
        if original.shape != (q.shape[0],) or observed.shape != original.shape:
            raise ValueError("rotation vectors have incompatible shapes")
        if (
            not np.isfinite(original).all()
            or not np.isfinite(observed).all()
            or float(np.linalg.norm(original)) <= _VECTOR_EPS
            or float(np.linalg.norm(observed)) <= _VECTOR_EPS
        ):
            raise ValueError(f"rotation vectors must be finite and nonzero for {name}")
        residual = float(np.linalg.norm(observed - q @ original)) / max(
            float(np.linalg.norm(original)), _VECTOR_EPS
        )
        if not np.isfinite(residual) or residual > 5.0e-4:
            raise ValueError(f"rotation vector gate failed for {name}")
        vector_residuals[name] = residual
    statistic_differences: dict[str, float] = {}
    for name, value in statistics.items():
        difference = abs(float(rotated_statistics[name]) - float(value))
        if not np.isfinite(difference) or difference > 2.0e-4:
            raise ValueError(f"rotation statistic gate failed for {name}")
        statistic_differences[name] = difference
    return {
        "vector_residuals": vector_residuals,
        "statistic_differences": statistic_differences,
    }


def joint_bootstrap(
    values_by_seed: np.ndarray,
    *,
    replicates: int = 10_000,
    seed: int = 200,
) -> np.ndarray:
    """Jointly resample identity columns, retaining all four seed pairings."""
    values = np.asarray(values_by_seed, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 4 or values.shape[1] == 0:
        raise ValueError("bootstrap values must have shape (4, identity_count)")
    if not np.isfinite(values).all():
        raise ValueError("bootstrap values must be finite")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be an unsigned integer")
    generator = np.random.Generator(np.random.PCG64(seed))
    identity_count = values.shape[1]
    distribution = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sampled = generator.integers(0, identity_count, size=identity_count)
        distribution[replicate] = values[:, sampled].mean(axis=1).mean()
    return distribution


def float64_c_order_sha256(values: np.ndarray) -> str:
    """Hash the exact float64 C-order bytes required by the output contract."""
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError("hashed values must be finite")
    return hashlib.sha256(np.ascontiguousarray(array).tobytes(order="C")).hexdigest()


def _panel_matrices(
    rows: Sequence[dict[str, Any]],
    *,
    identity_count: int,
    value_names: tuple[str, ...],
    panel_name: str,
    label_order: Sequence[int] | None = None,
) -> tuple[list[int], dict[str, np.ndarray]]:
    by_seed: dict[int, dict[int, dict[str, Any]]] = {seed: {} for seed in range(4)}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError(f"{panel_name} rows must be mappings")
        raw_seed = row.get("seed")
        raw_label = row.get("label")
        if (
            isinstance(raw_seed, bool)
            or not isinstance(raw_seed, (int, np.integer))
            or int(raw_seed) not in by_seed
        ):
            raise ValueError(f"{panel_name} seed must be one of 0, 1, 2, 3")
        if (
            isinstance(raw_label, bool)
            or not isinstance(raw_label, (int, np.integer))
            or int(raw_label) < 0
        ):
            raise ValueError(f"{panel_name} labels must be unsigned integers")
        seed = int(raw_seed)
        label = int(raw_label)
        if label in by_seed[seed]:
            raise ValueError(f"duplicate {panel_name} row for seed={seed}, label={label}")
        for name in value_names:
            if name not in row or not np.isfinite(float(row[name])):
                raise ValueError(f"{panel_name} row field {name} must be finite")
        by_seed[seed][label] = row
    label_sets = [set(by_seed[seed]) for seed in range(4)]
    if any(len(labels) != identity_count for labels in label_sets) or any(
        labels != label_sets[0] for labels in label_sets[1:]
    ):
        raise ValueError(
            f"{panel_name} panel must contain a complete shared set of "
            f"{identity_count} identities in every seed"
        )
    if label_order is None:
        labels = sorted(
            label_sets[0],
            key=lambda label: (domain_hash(_IDENTITY_DOMAIN, str(label)), label),
        )
    else:
        labels = list(label_order)
        if len(labels) != identity_count or set(labels) != label_sets[0]:
            raise ValueError(f"{panel_name} labels differ from the registered alternate subset")
    matrices = {
        name: np.asarray(
            [[float(by_seed[seed][label][name]) for label in labels] for seed in range(4)],
            dtype=np.float64,
        )
        for name in value_names
    }
    return labels, matrices


def decide_stage_a(
    rows: Sequence[dict[str, Any]], alternate_rows: Sequence[dict[str, Any]]
) -> dict[str, object]:
    """Aggregate complete panels and apply the frozen Pass200 Stage-A verdict."""
    labels, primary = _panel_matrices(
        rows,
        identity_count=64,
        value_names=(
            "delta",
            "self_minus_desc",
            "rho",
            "log_ratio",
            "deranged_delta",
            "a_self",
            "a_batch",
            "a_desc",
            "cos_b_s",
            "random_a_self",
            "random_a_batch",
            "random_delta",
            "head_a_self",
            "head_a_batch",
        ),
        panel_name="primary",
    )
    registered_alternate_labels = [
        label for index, label in enumerate(labels) if index % 8 in (0, 1)
    ]
    alternate_labels, alternate = _panel_matrices(
        alternate_rows,
        identity_count=16,
        value_names=("delta",),
        panel_name="alternate",
        label_order=registered_alternate_labels,
    )

    seed_deltas_array = primary["delta"].mean(axis=1)
    seed_deltas = {str(seed): float(seed_deltas_array[seed]) for seed in range(4)}
    pooled_delta = float(seed_deltas_array.mean())
    seed_self_desc_array = primary["self_minus_desc"].mean(axis=1)
    seed_self_desc = {str(seed): float(seed_self_desc_array[seed]) for seed in range(4)}
    pooled_self_desc = float(seed_self_desc_array.mean())
    pooled_median_rho = float(np.median(primary["rho"]))
    pooled_median_abs_log_ratio = float(np.median(np.abs(primary["log_ratio"])))
    pooled_deranged_delta = float(primary["deranged_delta"].mean(axis=1).mean())
    alternate_seed_array = alternate["delta"].mean(axis=1)
    alternate_seed_deltas = {str(seed): float(alternate_seed_array[seed]) for seed in range(4)}
    alternate_pooled_delta = float(alternate_seed_array.mean())

    bootstrap_delta = joint_bootstrap(primary["delta"])
    bootstrap_self_desc = joint_bootstrap(primary["self_minus_desc"])
    bootstrap_delta_lower = float(np.percentile(bootstrap_delta, 2.5))
    bootstrap_self_desc_lower = float(np.percentile(bootstrap_self_desc, 2.5))
    primary_seed_ge = int(np.count_nonzero(seed_deltas_array >= 0.02))
    primary_seed_nonpositive = int(np.count_nonzero(seed_deltas_array <= 0.0))
    alternate_seed_positive = int(np.count_nonzero(alternate_seed_array > 0.0))
    alternate_seed_nonpositive = int(np.count_nonzero(alternate_seed_array <= 0.0))
    criteria = {
        "pooled_delta_ge_0_03": pooled_delta >= 0.03,
        "bootstrap_delta_lower_positive": bootstrap_delta_lower > 0.0,
        "three_primary_seed_means_ge_0_02": primary_seed_ge >= 3,
        "pooled_self_minus_desc_positive": pooled_self_desc > 0.0,
        "bootstrap_self_minus_desc_lower_positive": bootstrap_self_desc_lower > 0.0,
        "median_rho_ge_0_20": pooled_median_rho >= 0.20,
        "median_abs_log_ratio_ge_log_1_10": pooled_median_abs_log_ratio >= float(np.log(1.10)),
        "absolute_deranged_delta_le_0_01": abs(pooled_deranged_delta) <= 0.01,
        "alternate_pooled_delta_positive": alternate_pooled_delta > 0.0,
        "three_alternate_seed_means_positive": alternate_seed_positive >= 3,
    }
    fail_clauses = (
        (pooled_delta <= 0.0, "pooled_delta_nonpositive"),
        (primary_seed_nonpositive >= 3, "three_primary_seed_means_nonpositive"),
        (pooled_median_rho < 0.10, "median_rho_below_0_10"),
        (alternate_pooled_delta <= 0.0, "alternate_pooled_delta_nonpositive"),
        (
            alternate_seed_nonpositive >= 3,
            "three_alternate_seed_means_nonpositive",
        ),
    )
    first_fail = next((name for triggered, name in fail_clauses if triggered), None)
    if first_fail is not None:
        stage_a = "FAIL"
        first_decisive_clause = first_fail
    elif all(criteria.values()):
        stage_a = "PASS_ONWARD"
        first_decisive_clause = "all_pass_requirements"
    else:
        stage_a = "UNRESOLVED"
        first_decisive_clause = "no_pass_or_fail_rule"

    control_names = (
        "a_self",
        "a_batch",
        "a_desc",
        "cos_b_s",
        "random_a_self",
        "random_a_batch",
        "random_delta",
        "deranged_delta",
        "head_a_self",
        "head_a_batch",
    )
    control_aggregates = {
        name: {
            "seed_means": {str(seed): float(primary[name][seed].mean()) for seed in range(4)},
            "pooled_mean": float(primary[name].mean(axis=1).mean()),
            "pooled_median": float(np.median(primary[name])),
        }
        for name in control_names
    }

    return {
        "stage_a": stage_a,
        "first_decisive_clause": first_decisive_clause,
        "criteria": criteria,
        "complete_identity_count": len(labels),
        "alternate_identity_count": len(alternate_labels),
        "complete_labels": labels,
        "alternate_labels": alternate_labels,
        "pooled_delta": pooled_delta,
        "seed_deltas": seed_deltas,
        "seed_deltas_ge_0_02": primary_seed_ge,
        "seed_deltas_nonpositive": primary_seed_nonpositive,
        "pooled_self_minus_desc": pooled_self_desc,
        "seed_self_minus_desc": seed_self_desc,
        "pooled_median_rho": pooled_median_rho,
        "pooled_median_abs_log_ratio": pooled_median_abs_log_ratio,
        "pooled_deranged_delta": pooled_deranged_delta,
        "alternate_pooled_delta": alternate_pooled_delta,
        "alternate_seed_deltas": alternate_seed_deltas,
        "alternate_seed_deltas_positive": alternate_seed_positive,
        "alternate_seed_deltas_nonpositive": alternate_seed_nonpositive,
        "bootstrap_seed": 200,
        "bootstrap_replicates": 10_000,
        "bootstrap_delta_lower_bound": bootstrap_delta_lower,
        "bootstrap_self_desc_lower_bound": bootstrap_self_desc_lower,
        "bootstrap_delta_sha256": float64_c_order_sha256(bootstrap_delta),
        "bootstrap_self_desc_sha256": float64_c_order_sha256(bootstrap_self_desc),
        "numpy_version": np.__version__,
        "control_aggregates": control_aggregates,
    }


def _torch_tensor_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _numpy_array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(value)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def _load_scientific_model(bound: TrainingOnlySeedInput) -> Any:
    import torch

    from sfora.image_end_to_end import ImageEndToEndConfig, _torchvision_model_factory

    _assert_deterministic_tf32_off()
    validate_retained_training_arrays(bound)
    if hashlib.sha256(bound.checkpoint_bytes).hexdigest() != bound.checkpoint_sha256:
        raise ValueError("retained checkpoint SHA-256 mismatch")
    config_dict = _json_ready(bound.config)
    checkpoint = torch.load(
        io.BytesIO(bound.checkpoint_bytes), map_location="cpu", weights_only=False
    )
    _validate_checkpoint_mapping(
        checkpoint,
        config=config_dict,
        train_labels=bound.train_labels,
        expected_dimension=int(config_dict["embedding_dimensions"]),
    )
    config = ImageEndToEndConfig.model_validate(config_dict)
    model = _torchvision_model_factory(config)
    state = {
        name: value
        for name, value in checkpoint["state_dict"].items()
        if name not in {"metric_proxies", "metric_proxy_labels"}
    }
    model.load_state_dict(state, strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device=device, dtype=torch.float32).train()


def _default_fixture_runner() -> dict[str, Any]:
    """Execute the registered dense-Jacobian and two-sample BN fixtures."""
    import torch

    torch.manual_seed(200)
    model = torch.nn.Linear(2, 3, dtype=torch.float64).train()
    images = torch.tensor([[0.2, -0.8], [1.1, 0.4], [-0.5, 0.7]], dtype=torch.float64)
    cotangents = torch.tensor(
        [[0.3, -0.5, 0.2], [-0.1, 0.4, 0.6], [0.7, 0.2, -0.3]],
        dtype=torch.float64,
    )
    fields = exact_kernel_fields(
        model,
        images,
        cotangents,
        receiver_indices=(0, 1, 2),
        expected_batch_size=3,
        expected_dimension=3,
    )
    parameters = torch.cat(
        [value.detach().reshape(-1) for value in model.parameters()]
    ).requires_grad_(True)

    def dense_encoder(flat: Any) -> Any:
        weight = flat[:6].reshape(3, 2)
        bias = flat[6:]
        raw = images @ weight.T + bias
        return torch.nn.functional.normalize(raw, dim=-1)

    dense = torch.autograd.functional.jacobian(dense_encoder, parameters).reshape(9, 9)
    expected_g = dense.T @ cotangents.reshape(-1)
    expected_b = (dense @ expected_g).reshape(3, 3)
    expected_self = []
    for receiver in range(3):
        block = dense[receiver * 3 : (receiver + 1) * 3]
        expected_self.append(block @ (block.T @ cotangents[receiver]))
    expected_self_tensor = torch.stack(expected_self)
    dense_residual = max(
        float(torch.max(torch.abs(fields["parameter_gradient_flat"] - expected_g)).detach().cpu()),
        float(torch.max(torch.abs(fields["batch_motion"] - expected_b)).detach().cpu()),
        float(torch.max(torch.abs(fields["self_motion"] - expected_self_tensor)).detach().cpu()),
    )
    epsilon = 1.0e-5
    finite = (
        dense_encoder(parameters + epsilon * expected_g)
        - dense_encoder(parameters - epsilon * expected_g)
    ) / (2.0 * epsilon)
    finite_residuals = [float(torch.max(torch.abs(fields["batch_motion"] - finite)).detach().cpu())]
    for receiver in range(3):
        block = dense[receiver * 3 : (receiver + 1) * 3]
        receiver_gradient = block.T @ cotangents[receiver]
        positive = dense_encoder(parameters + epsilon * receiver_gradient)[receiver]
        negative = dense_encoder(parameters - epsilon * receiver_gradient)[receiver]
        finite_self = (positive - negative) / (2.0 * epsilon)
        finite_residuals.append(
            float(
                torch.max(torch.abs(fields["self_motion"][receiver] - finite_self)).detach().cpu()
            )
        )
    finite_residual = max(finite_residuals)
    if dense_residual > 1.0e-8 or finite_residual > 1.0e-6:
        raise ValueError("registered dense-Jacobian fixture failed")

    class TinyBN(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bn = torch.nn.BatchNorm1d(2, dtype=torch.float64)
            self.head = torch.nn.Linear(2, 2, dtype=torch.float64)

        def forward(self, values: Any) -> Any:
            return self.head(self.bn(values))

    original = TinyBN().train()
    reference = copy.deepcopy(original).train()
    clone = make_bufferless_train_clone(original)
    bn_images = torch.tensor([[0.4, -0.3], [1.2, 0.8]], dtype=torch.float64)
    target = torch.tensor([[0.2, -0.5], [0.7, 0.1]], dtype=torch.float64)
    before = {name: value.detach().clone() for name, value in original.named_buffers()}
    reference_output = reference(bn_images)
    reference_gradients = torch.autograd.grad(
        (reference_output * target).sum(), tuple(reference.parameters())
    )
    clone_output = clone(bn_images)
    clone_gradients = torch.autograd.grad((clone_output * target).sum(), tuple(clone.parameters()))
    output_residual = float(torch.max(torch.abs(reference_output - clone_output)).detach().cpu())
    gradient_residual = max(
        float(torch.max(torch.abs(left - right)).detach().cpu())
        for left, right in zip(reference_gradients, clone_gradients, strict=True)
    )
    buffers_unchanged = all(
        torch.equal(value, before[name]) for name, value in original.named_buffers()
    )
    if output_residual > 1.0e-6 or gradient_residual > 1.0e-6 or not buffers_unchanged:
        raise ValueError("registered train-BN fixture failed")
    return {
        "dense_fixture": {
            "passed": True,
            "max_jacobian_residual": dense_residual,
            "max_finite_difference_residual": finite_residual,
            "jacobian_tolerance": 1.0e-8,
            "finite_difference_tolerance": 1.0e-6,
        },
        "bn_fixture": {
            "passed": True,
            "max_output_residual": output_residual,
            "max_gradient_residual": gradient_residual,
            "tolerance": 1.0e-6,
            "buffers_unchanged": buffers_unchanged,
        },
    }


def _rotation_vectors_and_statistics(
    fields: Mapping[str, Any],
    *,
    receiver_position: int,
    receiver_index: int,
    q: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    z = np.asarray(fields["z"][receiver_index].detach().cpu(), dtype=np.float64)
    dbar = tangent_projection(
        np.asarray(fields["dbar"][receiver_index].detach().cpu(), dtype=np.float64), z
    )
    b = tangent_projection(
        np.asarray(fields["batch_motion"][receiver_index].detach().cpu(), dtype=np.float64), z
    )
    s = tangent_projection(
        np.asarray(fields["self_motion"][receiver_position].detach().cpu(), dtype=np.float64), z
    )
    a_self = cosine_similarity(s, q)
    a_batch = cosine_similarity(b, q)
    b_unit = b / np.linalg.norm(b)
    s_unit = s / np.linalg.norm(s)
    return (
        {"z": z, "dbar": dbar, "b": b, "s": s, "q": q},
        {
            "A_self": a_self,
            "A_batch": a_batch,
            "Delta": a_self - a_batch,
            "A_desc": cosine_similarity(dbar, q),
            "rho": float(np.linalg.norm(b_unit - s_unit * np.dot(s_unit, b_unit))),
            "log_ratio": float(
                np.log((np.linalg.norm(b) + _VECTOR_EPS) / (np.linalg.norm(s) + _VECTOR_EPS))
            ),
            "cos_b_s": cosine_similarity(b, s),
        },
    )


def _default_rotation_auditor(
    model: Any,
    context: Mapping[str, Any],
    fields: Mapping[str, Any],
    proxies: Any,
    proxy_labels: Any,
    *,
    seed: int,
    expected_dimension: int,
    head_name: str,
    expected_head_in_features: int,
) -> dict[str, Any]:
    """Execute the registered common descriptor rotation on the first batch/receiver."""
    import torch

    rotated_model = copy.deepcopy(model)
    try:
        head = rotated_model.get_submodule(head_name)
    except (AttributeError, KeyError) as error:
        raise ValueError(f"rotation model lacks exact affine head {head_name}") from error
    if (
        not isinstance(head, torch.nn.Linear)
        or head.in_features != expected_head_in_features
        or head.out_features != expected_dimension
        or head.bias is None
    ):
        raise ValueError("rotation requires the exact registered affine embedding head")
    rotation_numpy = construct_rotation(expected_dimension, seed=200, dtype=np.float64)
    rotation = torch.as_tensor(rotation_numpy, dtype=head.weight.dtype, device=head.weight.device)
    rotation_rows = np.asarray(rotation.detach().cpu())
    with torch.no_grad():
        head.weight.copy_(rotation @ head.weight)
        head.bias.copy_(rotation @ head.bias)
    rotated_proxies = proxies @ rotation.T
    receiver_indices = tuple(context["receiver_indices"])
    rotated_fields = exact_contextual_rsta_fields(
        rotated_model,
        context["images"],
        context["labels"],
        rotated_proxies,
        proxy_labels,
        alpha=32.0,
        delta=0.1,
        receiver_indices=receiver_indices,
        expected_batch_size=180,
        expected_dimension=expected_dimension,
    )
    receiver_index = receiver_indices[0]
    receiver_label = int(context["labels"][receiver_index].detach().cpu())
    support_ids, supports = context["support_map"][receiver_label]
    original_z = np.asarray(fields["z"][receiver_index].detach().cpu(), dtype=np.float64)
    selected_ids, selected_foreign = select_foreign_supports(
        original_z,
        receiver_label=receiver_label,
        support_ids=context["foreign_ids"],
        support_labels=context["foreign_labels"],
        support_descriptors=context["foreign_descriptors"],
        current_batch_ids=set(context["batch_ids"]),
    )
    q = smooth_margin_gradient(original_z, supports, selected_foreign)
    rotated_z = np.asarray(rotated_fields["z"][receiver_index].detach().cpu(), dtype=np.float64)
    rotated_supports = np.asarray(supports, dtype=np.float64) @ rotation_rows.T
    rotated_foreign_pool = (
        np.asarray(context["foreign_descriptors"], dtype=np.float64) @ rotation_rows.T
    )
    rotated_ids, rotated_selected_foreign = select_foreign_supports(
        rotated_z,
        receiver_label=receiver_label,
        support_ids=context["foreign_ids"],
        support_labels=context["foreign_labels"],
        support_descriptors=rotated_foreign_pool,
        current_batch_ids=set(context["batch_ids"]),
    )
    if selected_ids != rotated_ids or len(support_ids) != 2:
        raise ValueError("rotation changed frozen foreign-support selection")
    rotated_q = smooth_margin_gradient(rotated_z, rotated_supports, rotated_selected_foreign)
    vectors, statistics = _rotation_vectors_and_statistics(
        fields,
        receiver_position=0,
        receiver_index=receiver_index,
        q=q,
    )
    rotated_vectors, rotated_statistics = _rotation_vectors_and_statistics(
        rotated_fields,
        receiver_position=0,
        receiver_index=receiver_index,
        q=rotated_q,
    )
    audit = check_rotation(
        vectors,
        rotated_vectors,
        statistics,
        rotated_statistics,
        rotation_rows,
    )
    return {"seed": int(seed), **audit}


def _integrity_only(
    *,
    repeatability_runner: Callable[[], Any],
    adjoint_runner: Callable[[], Any],
    rotation_runner: Callable[[], Any],
) -> tuple[Any, Any, Any]:
    """Run the ordered first-batch integrity gates without candidate scoring."""
    repeatability = repeatability_runner()
    adjoint = adjoint_runner()
    rotation = rotation_runner()
    return repeatability, adjoint, rotation


def _registered_first_batch_integrity(
    model: Any,
    images: Any,
    labels: Any,
    proxies: Any,
    proxy_labels: Any,
    *,
    seed: int,
    alpha: float,
    delta: float,
    receiver_indices: Sequence[int],
    context: Mapping[str, Any],
    expected_dimension: int,
    rotation_auditor: Callable[..., dict[str, Any]],
    head_name: str,
    expected_head_in_features: int,
) -> dict[str, Any]:
    """Run the exact registered first batch once, repeat it, and audit its operators."""
    import torch

    _assert_deterministic_tf32_off()
    prehead, raw_head, captured = capture_prehead_and_raw(
        model,
        images,
        head_name=head_name,
        expected_in_features=expected_head_in_features,
        expected_out_features=expected_dimension,
    )
    fields = exact_contextual_rsta_fields(
        model,
        images,
        labels,
        proxies,
        proxy_labels,
        alpha=alpha,
        delta=delta,
        receiver_indices=receiver_indices,
        expected_batch_size=180,
        expected_dimension=expected_dimension,
    )
    if not torch.equal(torch.nn.functional.normalize(captured, dim=-1), fields["z"]):
        raise ValueError("prehead control forward differs from exact field graph")
    names = {"z": "z", "dbar": "dbar", "b": "batch_motion", "s": "self_motion"}
    detached = {key: fields[key].detach().clone() for key in names.values()}
    detached["receiver_indices"] = fields["receiver_indices"]
    first_hashes = {name: _torch_tensor_sha256(detached[key]) for name, key in names.items()}
    del fields

    def run_repeatability() -> dict[str, dict[str, str]]:
        _assert_deterministic_tf32_off()
        repeated = exact_contextual_rsta_fields(
            model,
            images,
            labels,
            proxies,
            proxy_labels,
            alpha=alpha,
            delta=delta,
            receiver_indices=receiver_indices,
            expected_batch_size=180,
            expected_dimension=expected_dimension,
        )
        result = {
            name: {
                "first_sha256": first_hashes[name],
                "repeat_sha256": _torch_tensor_sha256(repeated[key]),
            }
            for name, key in names.items()
        }
        del repeated
        if any(
            hashes["first_sha256"] != hashes["repeat_sha256"] for hashes in result.values()
        ):
            raise ValueError("first-batch exact field repeatability failed")
        return result

    def run_adjoint() -> float:
        _assert_deterministic_tf32_off()
        parameter = next(value for value in model.parameters() if value.requires_grad)
        u, v = registered_adjoint_directions(
            model,
            detached["z"].shape,
            seed=seed,
            dtype=parameter.dtype,
            device=parameter.device,
        )
        result = adjoint_relative_error(
            model,
            images,
            u,
            v,
            expected_batch_size=180,
            expected_dimension=expected_dimension,
        )
        if not np.isfinite(result) or result < 0.0 or result > 5.0e-4:
            raise ValueError("first-batch adjoint integrity failed")
        return result

    def run_rotation() -> dict[str, Any]:
        _assert_deterministic_tf32_off()
        result = rotation_auditor(
            model,
            context,
            detached,
            proxies,
            proxy_labels,
            seed=seed,
            expected_dimension=expected_dimension,
            head_name=head_name,
            expected_head_in_features=expected_head_in_features,
        )
        vectors = result.get("vector_residuals", {})
        scalars = result.get("statistic_differences", {})
        if (
            set(vectors) != _ROTATION_VECTOR_NAMES
            or set(scalars) != _ROTATION_STATISTIC_NAMES
            or any(
                not np.isfinite(float(value)) or float(value) < 0.0 or float(value) > 5.0e-4
                for value in vectors.values()
            )
            or any(
                not np.isfinite(float(value)) or float(value) < 0.0 or float(value) > 2.0e-4
                for value in scalars.values()
            )
        ):
            raise ValueError("first-batch rotation integrity failed")
        return result

    repeatability, adjoint, rotation = _integrity_only(
        repeatability_runner=run_repeatability,
        adjoint_runner=run_adjoint,
        rotation_runner=run_rotation,
    )
    return {
        "fields": detached,
        "prehead": prehead,
        "raw_head": raw_head,
        "repeatability": repeatability,
        "adjoint_relative_error": adjoint,
        "rotation": rotation,
    }


def run_scientific_diagnostic(
    manifest: Mapping[str, Any] | None,
    *,
    manifest_path: Path,
    receipt_path: Path,
    output_path: Path,
    expected_dimension: int = 512,
    expected_partition: dict[str, tuple[int, int]] | None = None,
    receipt_validator: Callable[[Path, Path], ValidatedBindingReceipt] = (
        validate_historical_binding_receipt
    ),
    execution_source_validator: Callable[[Path], dict[str, Any]] = (
        validate_scientific_execution_source
    ),
    bound_loader: Callable[..., TrainingOnlySeedInput] = load_training_only_seed,
    cache_builder: Callable[..., DeterministicTransformCache] = cache_seed_training_tensors,
    model_loader: Callable[[TrainingOnlySeedInput], Any] = _load_scientific_model,
    fixture_runner: Callable[[], dict[str, Any]] = _default_fixture_runner,
    deterministic_pool_auditor: Callable[[], dict[str, Any]] = (
        audit_deterministic_global_max
    ),
    zero_jacobian_auditor: Callable[[Any, Any], dict[str, Any]] = (
        audit_zero_jacobian_classifier
    ),
    rotation_auditor: Callable[..., dict[str, Any]] | None = None,
    head_name: str = "model.embedding",
    expected_head_in_features: int = 1024,
) -> dict[str, Any]:
    """Execute the complete frozen four-seed Stage-A path and atomically persist rows."""
    environment = configure_deterministic_process()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"output already exists: {output_path}")
    receipt = receipt_validator(manifest_path, receipt_path)
    execution_audit = execution_source_validator(manifest_path)
    if manifest is None:
        manifest = load_strict_json(manifest_path)
    deterministic_global_max = deterministic_pool_auditor()
    _validate_deterministic_global_max_audit(deterministic_global_max)
    bounds = [
        bound_loader(
            manifest["seeds"][str(seed)],
            receipt.seeds[seed],
            expected_partition=expected_partition,
            expected_dimension=expected_dimension,
        )
        for seed in range(4)
    ]
    import torch

    validate_cross_seed_training_binding(bounds)
    for bound in bounds:
        validate_retained_training_arrays(bound)
    reference = bounds[0]
    primary = select_primary_panel(reference.train_example_ids, reference.train_labels)
    alternate = select_alternate_panel(reference.train_example_ids, reference.train_labels, primary)
    used_ids = sorted(
        {
            example_id
            for batch in [*primary["batches"], *alternate["batches"]]
            for example_id in batch
        }
    )
    cache = cache_builder(reference, used_ids)
    if set(cache.tensor_sha256) != set(used_ids):
        raise ValueError("scientific transform cache differs from registered batch rows")
    label_by_id = dict(zip(reference.train_example_ids, reference.train_labels, strict=True))
    index_by_id = {value: index for index, value in enumerate(reference.train_example_ids)}
    tensor_hashes = dict(cache.tensor_sha256)
    fixture_integrity = fixture_runner()
    primary_rows: list[dict[str, Any]] = []
    alternate_rows: list[dict[str, Any]] = []
    seed_audits: list[dict[str, Any]] = []
    seed_integrity: list[dict[str, Any]] = []
    zero_jacobian_integrity: dict[str, dict[str, Any]] = {}
    rotate = _default_rotation_auditor if rotation_auditor is None else rotation_auditor
    for bound in bounds:
        _assert_deterministic_tf32_off()
        model = make_rsta_diagnostic_clone(model_loader(bound))
        first_parameter = next(iter(model.parameters()), None)
        if first_parameter is None:
            raise ValueError("scientific encoder has no parameters")
        device, dtype = first_parameter.device, first_parameter.dtype
        audit_images = cache.batch(primary["batches"][0]).to(device=device, dtype=dtype)
        zero_jacobian = zero_jacobian_auditor(model, audit_images)
        _validate_zero_jacobian_classifier_audit(zero_jacobian)
        zero_jacobian_integrity[str(bound.seed)] = zero_jacobian
        parameter_items = [
            (name, value) for name, value in model.named_parameters() if value.requires_grad
        ]
        if not parameter_items:
            raise ValueError("scientific encoder has no trainable parameters")
        proxies = torch.tensor(np.array(bound.proxies, copy=True), device=device, dtype=dtype)
        proxy_labels = torch.tensor(
            np.array(bound.proxy_labels, copy=True), device=device, dtype=torch.long
        )
        support_map = {
            label: (
                tuple(primary["support_ids_by_label"][label]),
                np.asarray(
                    [
                        bound.train_embeddings[index_by_id[value]]
                        for value in primary["support_ids_by_label"][label]
                    ],
                    dtype=np.float32,
                ),
            )
            for label in primary["eligible_labels"]
        }
        foreign_ids = tuple(
            primary["support_ids_by_label"][label][0] for label in primary["eligible_labels"]
        )
        foreign_labels = tuple(primary["eligible_labels"])
        foreign_descriptors = np.asarray(
            [bound.train_embeddings[index_by_id[value]] for value in foreign_ids],
            dtype=np.float32,
        )
        first_integrity: dict[str, Any] | None = None
        for panel_name, selected in (("primary", primary), ("alternate", alternate)):
            destination = primary_rows if panel_name == "primary" else alternate_rows
            for batch_index, batch_ids in enumerate(selected["batches"]):
                receiver_ids = tuple(
                    selected["receiver_ids"][batch_index * 8 : (batch_index + 1) * 8]
                )
                receiver_labels = tuple(selected["labels"][batch_index * 8 : (batch_index + 1) * 8])
                receiver_indices = tuple(batch_ids.index(value) for value in receiver_ids)
                images = cache.batch(batch_ids).to(device=device, dtype=dtype)
                labels = torch.as_tensor(
                    [label_by_id[value] for value in batch_ids], device=device, dtype=torch.long
                )
                context = {
                    "images": images,
                    "labels": labels,
                    "receiver_indices": receiver_indices,
                    "batch_ids": tuple(batch_ids),
                    "support_map": support_map,
                    "foreign_ids": foreign_ids,
                    "foreign_labels": foreign_labels,
                    "foreign_descriptors": foreign_descriptors,
                }
                if panel_name == "primary" and batch_index == 0:
                    first = _registered_first_batch_integrity(
                        model,
                        images,
                        labels,
                        proxies,
                        proxy_labels,
                        seed=bound.seed,
                        alpha=bound.alpha,
                        delta=bound.delta,
                        receiver_indices=receiver_indices,
                        context=context,
                        expected_dimension=expected_dimension,
                        rotation_auditor=rotate,
                        head_name=head_name,
                        expected_head_in_features=expected_head_in_features,
                    )
                    fields = first["fields"]
                    prehead, raw_head = first["prehead"], first["raw_head"]
                    first_integrity = {
                        "seed": bound.seed,
                        "repeatability": first["repeatability"],
                        "adjoint_relative_error": first["adjoint_relative_error"],
                        "rotation": first["rotation"],
                    }
                else:
                    prehead, raw_head, captured = capture_prehead_and_raw(
                        model,
                        images,
                        head_name=head_name,
                        expected_in_features=expected_head_in_features,
                        expected_out_features=expected_dimension,
                    )
                    fields = exact_contextual_rsta_fields(
                        model,
                        images,
                        labels,
                        proxies,
                        proxy_labels,
                        alpha=bound.alpha,
                        delta=bound.delta,
                        receiver_indices=receiver_indices,
                        expected_batch_size=180,
                        expected_dimension=expected_dimension,
                    )
                    if not torch.equal(
                        torch.nn.functional.normalize(captured, dim=-1), fields["z"]
                    ):
                        raise ValueError("prehead control forward differs from exact field graph")
                destination.extend(
                    score_rsta_batch(
                        seed=bound.seed,
                        panel=panel_name,
                        batch_index=batch_index,
                        receiver_indices=receiver_indices,
                        receiver_ids=receiver_ids,
                        receiver_labels=receiver_labels,
                        batch_ids=batch_ids,
                        tensor_hashes=tensor_hashes,
                        fields=fields,
                        supports_by_label=support_map,
                        foreign_ids=foreign_ids,
                        foreign_labels=foreign_labels,
                        foreign_descriptors=foreign_descriptors,
                        prehead_features=np.asarray(prehead.cpu()),
                        raw_head_outputs=np.asarray(raw_head.cpu()),
                    )
                )
                del fields
        if first_integrity is None:
            raise ValueError("scientific primary first-batch integrity context is missing")
        seed_integrity.append(first_integrity)
        seed_audits.append(
            {
                "seed": bound.seed,
                "official_recall_at_1": bound.official_recall_at_1,
                "artifact_binding": _json_ready(bound.artifact_binding),
                "config": _json_ready(bound.config),
                "parameter_names": [name for name, _ in parameter_items],
                "parameter_count": int(sum(value.numel() for _, value in parameter_items)),
                "proxy_sha256": _numpy_array_sha256(bound.proxies),
                "proxy_label_sha256": _ordered_int64_sha256(bound.proxy_labels),
                "train_example_id_order_sha256": _ordered_text_sha256(bound.train_example_ids),
                "train_label_order_sha256": _ordered_int64_sha256(bound.train_labels),
                "train_source_order_sha256": _ordered_text_sha256(bound.train_source_paths),
                "transform_cache_order_sha256": cache.ordered_id_sha256,
                "transform_tensor_sha256": tensor_hashes,
                "primary_batch_ids": primary["batches"],
                "alternate_batch_ids": alternate["batches"],
            }
        )
    aggregation = decide_stage_a(primary_rows, alternate_rows)
    _, matrices = _panel_matrices(
        primary_rows,
        identity_count=64,
        value_names=("delta", "self_minus_desc"),
        panel_name="primary",
    )
    delta_distribution = joint_bootstrap(matrices["delta"])
    self_desc_distribution = joint_bootstrap(matrices["self_minus_desc"])
    integrity = {
        **fixture_integrity,
        "deterministic_global_max": deterministic_global_max,
        "zero_jacobian_classifier": zero_jacobian_integrity,
        "seeds": seed_integrity,
    }
    panel_binding = {
        "primary": primary,
        "alternate": alternate,
        "expected_dimension": int(expected_dimension),
        "tensor_sha256": tensor_hashes,
        "foreign_support_ids": sorted(
            primary["support_ids_by_label"][label][0] for label in primary["eligible_labels"]
        ),
    }
    payload = scientific_payload(
        manifest_audit={
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "base_preregistration": manifest.get(
                "base_preregistration", manifest.get("preregistration")
            ),
            "amendment": manifest.get("amendment"),
            "deterministic_pool_amendment": manifest.get(
                "deterministic_pool_amendment"
            ),
            "zero_jacobian_classifier_amendment": manifest.get(
                "zero_jacobian_classifier_amendment"
            ),
            "binding_receipt": manifest.get("binding_receipt"),
            "historical": manifest.get("historical"),
            "artifact_schema": manifest.get("artifact_schema"),
            "source": manifest.get("current_scientific_source", manifest.get("source")),
        },
        execution_audit=execution_audit,
        environment=environment,
        seed_audits=seed_audits,
        primary_rows=primary_rows,
        alternate_rows=alternate_rows,
        integrity=integrity,
        aggregation=aggregation,
        bootstrap={
            "delta_distribution": delta_distribution.tolist(),
            "delta_sha256": float64_c_order_sha256(delta_distribution),
            "self_minus_desc_distribution": self_desc_distribution.tolist(),
            "self_minus_desc_sha256": float64_c_order_sha256(self_desc_distribution),
        },
        panel_binding=panel_binding,
    )
    write_json_atomic(output_path, payload)
    return payload


def _bound_checkpoint_sha256(artifact_binding: Mapping[str, Any]) -> str:
    """Return the exact digest-bound checkpoint hash from the real artifact schema."""
    artifacts = artifact_binding.get("artifacts")
    checkpoint = artifacts.get("checkpoint_pt") if isinstance(artifacts, Mapping) else None
    digest = checkpoint.get("sha256") if isinstance(checkpoint, Mapping) else None
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError("artifact binding lacks exact nested checkpoint SHA-256")
    return digest


def run_integrity_smoke(
    manifest: Mapping[str, Any] | None,
    *,
    manifest_path: Path,
    receipt_path: Path,
    output_path: Path,
    expected_dimension: int = 512,
    expected_partition: dict[str, tuple[int, int]] | None = None,
    receipt_validator: Callable[[Path, Path], ValidatedBindingReceipt] = (
        validate_historical_binding_receipt
    ),
    execution_source_validator: Callable[[Path], dict[str, Any]] = (
        validate_scientific_execution_source
    ),
    bound_loader: Callable[..., TrainingOnlySeedInput] = load_training_only_seed,
    cache_builder: Callable[..., DeterministicTransformCache] = cache_seed_training_tensors,
    model_loader: Callable[[TrainingOnlySeedInput], Any] = _load_scientific_model,
    fixture_runner: Callable[[], dict[str, Any]] = _default_fixture_runner,
    deterministic_pool_auditor: Callable[[], dict[str, Any]] = (
        audit_deterministic_global_max
    ),
    zero_jacobian_auditor: Callable[[Any, Any], dict[str, Any]] = (
        audit_zero_jacobian_classifier
    ),
    rotation_auditor: Callable[..., dict[str, Any]] | None = None,
    head_name: str = "model.embedding",
    expected_head_in_features: int = 1024,
) -> dict[str, Any]:
    """Run the registered Step-7 first-batch integrity gates without candidate scoring."""
    environment = configure_deterministic_process()
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(f"output already exists: {output_path}")
    receipt = receipt_validator(manifest_path, receipt_path)
    execution_audit = execution_source_validator(manifest_path)
    if manifest is None:
        manifest = load_strict_json(manifest_path)
    seed = 0
    bound = bound_loader(
        manifest["seeds"][str(seed)],
        receipt.seeds[seed],
        expected_partition=expected_partition,
        expected_dimension=expected_dimension,
    )
    import torch

    validate_retained_training_arrays(bound)
    fixtures = fixture_runner()
    _validate_fixture_integrity(fixtures)
    primary = select_primary_panel(bound.train_example_ids, bound.train_labels)
    batch_ids = tuple(primary["batches"][0])
    receiver_ids = tuple(primary["receiver_ids"][:8])
    receiver_indices = tuple(batch_ids.index(value) for value in receiver_ids)
    cache = cache_builder(bound, batch_ids)
    if set(cache.tensor_sha256) != set(batch_ids):
        raise ValueError("smoke transform cache differs from registered first batch")
    label_by_id = dict(zip(bound.train_example_ids, bound.train_labels, strict=True))
    index_by_id = {value: index for index, value in enumerate(bound.train_example_ids)}
    _assert_deterministic_tf32_off()
    model = make_rsta_diagnostic_clone(model_loader(bound))
    deterministic_global_max = deterministic_pool_auditor()
    _validate_deterministic_global_max_audit(deterministic_global_max)
    first_parameter = next(iter(model.parameters()), None)
    if first_parameter is None:
        raise ValueError("smoke encoder has no parameters")
    device, dtype = first_parameter.device, first_parameter.dtype
    images = cache.batch(batch_ids).to(device=device, dtype=dtype)
    zero_jacobian = zero_jacobian_auditor(model, images)
    _validate_zero_jacobian_classifier_audit(zero_jacobian)
    parameter_items = [
        (name, value) for name, value in model.named_parameters() if value.requires_grad
    ]
    if not parameter_items:
        raise ValueError("smoke encoder has no trainable parameters")
    proxies = torch.tensor(np.array(bound.proxies, copy=True), device=device, dtype=dtype)
    proxy_labels = torch.tensor(
        np.array(bound.proxy_labels, copy=True), device=device, dtype=torch.long
    )
    support_map = {
        label: (
            tuple(primary["support_ids_by_label"][label]),
            np.asarray(
                [
                    bound.train_embeddings[index_by_id[value]]
                    for value in primary["support_ids_by_label"][label]
                ],
                dtype=np.float32,
            ),
        )
        for label in primary["eligible_labels"]
    }
    foreign_ids = tuple(
        primary["support_ids_by_label"][label][0] for label in primary["eligible_labels"]
    )
    foreign_labels = tuple(primary["eligible_labels"])
    foreign_descriptors = np.asarray(
        [bound.train_embeddings[index_by_id[value]] for value in foreign_ids], dtype=np.float32
    )
    labels = torch.as_tensor(
        [label_by_id[value] for value in batch_ids], device=device, dtype=torch.long
    )
    context = {
        "images": images,
        "labels": labels,
        "receiver_indices": receiver_indices,
        "batch_ids": batch_ids,
        "support_map": support_map,
        "foreign_ids": foreign_ids,
        "foreign_labels": foreign_labels,
        "foreign_descriptors": foreign_descriptors,
    }
    rotate = _default_rotation_auditor if rotation_auditor is None else rotation_auditor
    first = _registered_first_batch_integrity(
        model,
        images,
        labels,
        proxies,
        proxy_labels,
        seed=seed,
        alpha=bound.alpha,
        delta=bound.delta,
        receiver_indices=receiver_indices,
        context=context,
        expected_dimension=expected_dimension,
        rotation_auditor=rotate,
        head_name=head_name,
        expected_head_in_features=expected_head_in_features,
    )
    del first["fields"], first["prehead"], first["raw_head"]

    def json_sha256(value: Any) -> str:
        encoded = json.dumps(
            _json_ready(value), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    tensor_hashes = dict(cache.tensor_sha256)
    payload = {
        "schema_version": 1,
        "diagnostic": "pass200_rsta_stage_a",
        "mode": "integrity_smoke",
        "candidate_values_computed": False,
        "stage_a_verdict": "NOT_COMPUTED",
        "uses_test_data": "artifact_binding_only",
        "execution_audit": execution_audit,
        "manifest": {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
            "base_preregistration": manifest.get(
                "base_preregistration", manifest.get("preregistration")
            ),
            "amendment": manifest.get("amendment"),
            "deterministic_pool_amendment": manifest.get(
                "deterministic_pool_amendment"
            ),
            "zero_jacobian_classifier_amendment": manifest.get(
                "zero_jacobian_classifier_amendment"
            ),
            "binding_receipt": manifest.get("binding_receipt"),
            "historical": manifest.get("historical"),
            "artifact_schema": manifest.get("artifact_schema"),
            "source": manifest.get("current_scientific_source", manifest.get("source")),
        },
        "environment": environment,
        "binding": {
            "seed": seed,
            "receipt_sha256": receipt.sha256,
            "receipt_producer_commit": receipt.producer_commit,
            "historical_manifest_sha256": receipt.historical_manifest_sha256,
            "train_source_export_sha256": receipt.seeds[0].train_source_export_sha256,
            "artifact_binding_sha256": json_sha256(bound.artifact_binding),
            "config_sha256": json_sha256(bound.config),
            "checkpoint_sha256": _bound_checkpoint_sha256(bound.artifact_binding),
            "train_example_id_order_sha256": _ordered_text_sha256(bound.train_example_ids),
            "train_label_order_sha256": _ordered_int64_sha256(bound.train_labels),
            "train_source_order_sha256": _ordered_text_sha256(bound.train_source_paths),
            "proxy_sha256": _numpy_array_sha256(bound.proxies),
            "proxy_label_sha256": _ordered_int64_sha256(bound.proxy_labels),
            "parameter_name_order_sha256": _ordered_text_sha256(
                [name for name, _ in parameter_items]
            ),
            "parameter_count": int(sum(value.numel() for _, value in parameter_items)),
            "first_batch_size": len(batch_ids),
            "first_batch_id_sha256": _ordered_text_sha256(batch_ids),
            "receiver_id_sha256": _ordered_text_sha256(receiver_ids),
            "transform_cache_order_sha256": cache.ordered_id_sha256,
            "transform_tensor_set_sha256": _ordered_text_sha256(
                [f"{example_id}\0{tensor_hashes[example_id]}" for example_id in batch_ids]
            ),
        },
        "integrity": {
            "seed": seed,
            **fixtures,
            "deterministic_global_max": deterministic_global_max,
            "zero_jacobian_classifier": zero_jacobian,
            **first,
        },
    }
    write_json_atomic(output_path, payload)
    return payload


def main(
    argv: Sequence[str] | None = None,
    *,
    expected_partition: dict[str, tuple[int, int]] | None = None,
    expected_dimension: int = 512,
    receipt_validator: Callable[[Path, Path], ValidatedBindingReceipt] = (
        validate_historical_binding_receipt
    ),
    execution_source_validator: Callable[[Path], dict[str, Any]] = (
        validate_scientific_execution_source
    ),
    bound_loader: Callable[..., TrainingOnlySeedInput] = load_training_only_seed,
    cache_builder: Callable[..., DeterministicTransformCache] = cache_seed_training_tensors,
    model_loader: Callable[[TrainingOnlySeedInput], Any] = _load_scientific_model,
    fixture_runner: Callable[[], dict[str, Any]] = _default_fixture_runner,
    deterministic_pool_auditor: Callable[[], dict[str, Any]] = (
        audit_deterministic_global_max
    ),
    zero_jacobian_auditor: Callable[[Any, Any], dict[str, Any]] = (
        audit_zero_jacobian_classifier
    ),
    rotation_auditor: Callable[..., dict[str, Any]] | None = None,
    head_name: str = "model.embedding",
    expected_head_in_features: int = 1024,
) -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--binding-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--smoke-only", action="store_true")
    modes.add_argument("--scientific", action="store_true")
    args = parser.parse_args(argv)
    if args.smoke_only:
        run_integrity_smoke(
            None,
            manifest_path=args.manifest,
            receipt_path=args.binding_receipt,
            output_path=args.output,
            expected_dimension=expected_dimension,
            expected_partition=expected_partition,
            receipt_validator=receipt_validator,
            execution_source_validator=execution_source_validator,
            bound_loader=bound_loader,
            cache_builder=cache_builder,
            model_loader=model_loader,
            fixture_runner=fixture_runner,
            deterministic_pool_auditor=deterministic_pool_auditor,
            zero_jacobian_auditor=zero_jacobian_auditor,
            rotation_auditor=rotation_auditor,
            head_name=head_name,
            expected_head_in_features=expected_head_in_features,
        )
    else:
        run_scientific_diagnostic(
            None,
            manifest_path=args.manifest,
            receipt_path=args.binding_receipt,
            output_path=args.output,
            expected_dimension=expected_dimension,
            expected_partition=expected_partition,
            receipt_validator=receipt_validator,
            execution_source_validator=execution_source_validator,
            bound_loader=bound_loader,
            cache_builder=cache_builder,
            model_loader=model_loader,
            fixture_runner=fixture_runner,
            deterministic_pool_auditor=deterministic_pool_auditor,
            zero_jacobian_auditor=zero_jacobian_auditor,
            rotation_auditor=rotation_auditor,
            head_name=head_name,
            expected_head_in_features=expected_head_in_features,
        )


if __name__ == "__main__":
    main()
