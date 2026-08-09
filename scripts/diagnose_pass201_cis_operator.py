"""Prospectively frozen Pass201 CIS operator diagnostic.

The module stays import-side-effect-free.  Dataset/model activation belongs to
the command layer added by later implementation tasks; this foundation contains
only deterministic codecs and pure record construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

S_PRIME_RANK_SEED = 2010809
BOOTSTRAP_SEED = 2010811
BOOTSTRAP_REPLICATES = 20_000
CONTEXT_PAIRS = 32

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
OUTCOME_METRICS = ("R_F", "Delta_M", "D_F", "D_M")
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


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON value with the frozen canonical JSON settings."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _tensor_frame(array: Any) -> bytes:
    contiguous = np.ascontiguousarray(np.asarray(array))
    dtype_bytes = contiguous.dtype.str.encode("utf-8")
    payload = contiguous.tobytes(order="C")
    return b"".join(
        (
            struct.pack("<I", len(dtype_bytes)),
            dtype_bytes,
            struct.pack("<I", contiguous.ndim),
            *(struct.pack("<q", dimension) for dimension in contiguous.shape),
            struct.pack("<Q", len(payload)),
            payload,
        )
    )


def sha256_tensor_frame(array: Any) -> str:
    """Hash the frozen dtype/shape/length-framed C-order tensor bytes."""

    return hashlib.sha256(_tensor_frame(array)).hexdigest()


def sha256_named_tensors(named_tensors: Iterable[tuple[str, Any]]) -> str:
    """Hash ordered named tensors in canonical little-endian float64 form."""

    digest = hashlib.sha256()
    tensors = sorted(named_tensors, key=lambda item: item[0].encode())
    if len({name for name, _ in tensors}) != len(tensors):
        raise ValueError("tensor names must be unique")
    for name, value in tensors:
        name_bytes = name.encode("utf-8")
        array = np.ascontiguousarray(np.asarray(value, dtype="<f8"))
        payload = array.tobytes(order="C")
        digest.update(struct.pack("<I", len(name_bytes)))
        digest.update(name_bytes)
        digest.update(struct.pack("<I", array.ndim))
        for dimension in array.shape:
            digest.update(struct.pack("<q", dimension))
        digest.update(struct.pack("<Q", len(payload)))
        digest.update(payload)
    return digest.hexdigest()


def _metadata_for_digest(context: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
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
    return {key: context[key] for key in keys}


def build_input_context_digest(context: Mapping[str, Any]) -> dict[str, Any]:
    """Build exactly one ``input-context-digest-v1`` record."""

    record: dict[str, Any] = {
        "context_index": context["context_index"],
        "s_tensor_sha256": sha256_tensor_frame(context["s_tensor"]),
        "s_prime_tensor_sha256": sha256_tensor_frame(context["s_prime_tensor"]),
        "metadata_sha256": hashlib.sha256(
            canonical_json_bytes(_metadata_for_digest(context))
        ).hexdigest(),
    }
    record["combined_sha256"] = hashlib.sha256(canonical_json_bytes(record)).hexdigest()
    return record


def _representatives(rows: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    representative_rows: list[int] = []
    representative_samples: list[int] = []
    for label in sorted({int(row["label"]) for row in rows}):
        row_index, row = min(
            ((row_index, row) for row_index, row in enumerate(rows) if int(row["label"]) == label),
            key=lambda item: (int(item[1]["sample_index"]), item[0]),
        )
        representative_rows.append(row_index)
        representative_samples.append(int(row["sample_index"]))
    return representative_rows, representative_samples


def _s_prime_rows(
    rows: Sequence[Mapping[str, Any]], train_manifest: Sequence[Mapping[str, Any]]
) -> list[Mapping[str, Any]] | None:
    excluded_ids = {str(row["example_id"]) for row in rows}
    candidates: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in train_manifest:
        if str(row["example_id"]) not in excluded_ids:
            candidates[int(row["label"])].append(row)

    for label_candidates in candidates.values():
        label_candidates.sort(
            key=lambda row: (
                hashlib.sha256(
                    (f"pass201-sprime|{S_PRIME_RANK_SEED}|{row['example_id']}").encode()
                ).digest(),
                str(row["example_id"]).encode("utf-8"),
            )
        )

    needed = Counter(int(row["label"]) for row in rows)
    if any(len(candidates[label]) < count for label, count in needed.items()):
        return None

    next_offset: Counter[int] = Counter()
    selected: list[Mapping[str, Any]] = []
    for row in rows:
        label = int(row["label"])
        selected.append(candidates[label][next_offset[label]])
        next_offset[label] += 1
    if len({str(row["example_id"]) for row in selected}) != len(selected):
        return None
    return selected


def _cross_context_reuse(
    row_ids: set[str],
    s_prime_ids: set[str],
    labels: set[int],
    prior_contexts: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    current_any = row_ids | s_prime_ids
    sharing_s: list[int] = []
    sharing_s_prime: list[int] = []
    sharing_any: list[int] = []
    prior_ids: set[str] = set()
    prior_labels: set[int] = set()
    for prior in prior_contexts:
        prior_s = set(prior["row_example_ids"])
        prior_s_prime = set(prior["s_prime_example_ids"])
        prior_any = prior_s | prior_s_prime
        prior_index = int(prior["context_index"])
        if row_ids & prior_s:
            sharing_s.append(prior_index)
        if s_prime_ids & prior_s_prime:
            sharing_s_prime.append(prior_index)
        if current_any & prior_any:
            sharing_any.append(prior_index)
        prior_ids.update(prior_any)
        prior_labels.update(int(label) for label in prior["row_labels"])
    return {
        "prior_context_indices_sharing_s_ids": sorted(sharing_s),
        "prior_context_indices_sharing_s_prime_ids": sorted(sharing_s_prime),
        "prior_context_indices_sharing_any_ids": sorted(sharing_any),
        "reused_s_image_count": len(row_ids & prior_ids),
        "reused_s_prime_image_count": len(s_prime_ids & prior_ids),
        "reused_any_image_count": len(current_any & prior_ids),
        "reused_label_count": len(labels & prior_labels),
    }


def construct_one_context(
    *,
    rows: Sequence[Mapping[str, Any]],
    train_manifest: Sequence[Mapping[str, Any]],
    context_index: int,
    production_epoch: int = 0,
    production_batch_index: int | None = None,
    prior_contexts: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    """Construct one feasible context from literal row metadata.

    ``ValueError`` indicates that the candidate lacks enough disjoint
    same-class alternatives; callers that traverse an epoch retain it as a
    rejected partial-audit record.
    """

    rows = list(rows)
    representative_rows, representative_samples = _representatives(rows)
    s_prime = _s_prime_rows(rows, train_manifest)
    if s_prime is None:
        raise ValueError("INSUFFICIENT_DISJOINT_S_PRIME")
    row_ids = [str(row["example_id"]) for row in rows]
    row_labels = [int(row["label"]) for row in rows]
    s_prime_ids = [str(row["example_id"]) for row in s_prime]
    return {
        "context_index": context_index,
        "production_epoch": production_epoch,
        "production_batch_index": (
            context_index if production_batch_index is None else production_batch_index
        ),
        "row_example_ids": row_ids,
        "row_sample_indices": [int(row["sample_index"]) for row in rows],
        "row_labels": row_labels,
        "class_multiplicities": dict(Counter(row_labels)),
        "representative_row_indices": representative_rows,
        "representative_sample_indices": representative_samples,
        "s_prime_example_ids": s_prime_ids,
        "s_prime_sample_indices": [int(row["sample_index"]) for row in s_prime],
        "cross_context_reuse": _cross_context_reuse(
            set(row_ids), set(s_prime_ids), set(row_labels), prior_contexts
        ),
    }


def _partial_audit_record(
    context: Mapping[str, Any], *, status: str, rejection_code: str | None
) -> dict[str, Any]:
    keys = (
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
    )
    record = {key: context[key] for key in keys}
    record.update(status=status, rejection_code=rejection_code)
    return record


def construct_context_audit(
    *,
    batches: Iterable[Sequence[Mapping[str, Any]] | Mapping[str, Any]],
    train_manifest: Sequence[Mapping[str, Any]],
    target_count: int = 32,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Traverse candidate batches and retain accepted and rejected audit rows."""

    accepted: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []
    for consumed_index, batch in enumerate(batches):
        if len(accepted) == target_count:
            break
        if isinstance(batch, Mapping):
            rows = batch["rows"]
            production_epoch = int(batch.get("production_epoch", 0))
            production_batch_index = int(batch.get("production_batch_index", consumed_index))
        else:
            rows = batch
            production_epoch = 0
            production_batch_index = consumed_index
        representative_rows, representative_samples = _representatives(rows)
        base: dict[str, Any] = {
            "context_index": consumed_index,
            "production_epoch": production_epoch,
            "production_batch_index": production_batch_index,
            "row_example_ids": [str(row["example_id"]) for row in rows],
            "row_sample_indices": [int(row["sample_index"]) for row in rows],
            "row_labels": [int(row["label"]) for row in rows],
            "class_multiplicities": dict(Counter(int(row["label"]) for row in rows)),
            "representative_row_indices": representative_rows,
            "representative_sample_indices": representative_samples,
            "s_prime_example_ids": [],
            "s_prime_sample_indices": [],
        }
        try:
            context = construct_one_context(
                rows=rows,
                train_manifest=train_manifest,
                context_index=len(accepted),
                production_epoch=production_epoch,
                production_batch_index=production_batch_index,
                prior_contexts=accepted,
            )
        except ValueError as error:
            if str(error) != "INSUFFICIENT_DISJOINT_S_PRIME":
                raise
            audit.append(
                _partial_audit_record(
                    base,
                    status="rejected",
                    rejection_code="INSUFFICIENT_DISJOINT_S_PRIME",
                )
            )
            continue
        accepted.append(context)
        audit_context = dict(context)
        audit_context["context_index"] = consumed_index
        audit.append(_partial_audit_record(audit_context, status="accepted", rejection_code=None))
    return accepted, audit


def bootstrap_indices() -> np.ndarray:
    """Return the sole frozen paired-bootstrap resample matrix."""

    indices = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED)).integers(
        0,
        CONTEXT_PAIRS,
        size=(BOOTSTRAP_REPLICATES, CONTEXT_PAIRS),
        dtype=np.int64,
    )
    return np.ascontiguousarray(indices, dtype="<i8")


def bootstrap_mean_distribution(values: Any, indices: Any) -> np.ndarray:
    """Compute bootstrap means in the supplied fixed replicate order."""

    value_array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    index_array = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    if value_array.shape != (CONTEXT_PAIRS,):
        raise ValueError("values must contain exactly 32 context metrics")
    if index_array.shape != (BOOTSTRAP_REPLICATES, CONTEXT_PAIRS):
        raise ValueError("bootstrap indices must have shape (20000, 32)")
    if np.any(index_array < 0) or np.any(index_array >= CONTEXT_PAIRS):
        raise ValueError("bootstrap index out of range")
    if not np.isfinite(value_array).all():
        raise ValueError("metric values must be finite")
    return np.ascontiguousarray(value_array[index_array].mean(axis=1), dtype="<f8")


def sha256_bootstrap_indices(indices: Any) -> str:
    array = np.ascontiguousarray(np.asarray(indices, dtype="<i8"))
    if array.shape != (BOOTSTRAP_REPLICATES, CONTEXT_PAIRS):
        raise ValueError("bootstrap indices must have shape (20000, 32)")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def sha256_bootstrap_distribution(distribution: Any) -> str:
    array = np.ascontiguousarray(np.asarray(distribution, dtype="<f8"))
    if array.shape != (BOOTSTRAP_REPLICATES,) or not np.isfinite(array).all():
        raise ValueError("bootstrap distribution must contain 20000 finite values")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def summarize_metric(values: Any, bootstrap_indices: Any) -> dict[str, Any]:
    """Summarize one 32-context metric with the frozen paired bootstrap."""

    value_array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    distribution = bootstrap_mean_distribution(value_array, bootstrap_indices)
    return {
        "n": CONTEXT_PAIRS,
        "mean": float(np.mean(value_array)),
        "median": float(np.median(value_array)),
        "sample_sd": float(np.std(value_array, ddof=1)),
        "q25": float(np.quantile(value_array, 0.25, method="linear")),
        "q75": float(np.quantile(value_array, 0.75, method="linear")),
        "lcb_0_005": float(np.quantile(distribution, 0.005, method="linear")),
        "ucb_0_995": float(np.quantile(distribution, 0.995, method="linear")),
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _exact_keys(value: Any, expected: Iterable[str], path: str) -> None:
    _require(isinstance(value, dict), f"{path} must be an object")
    expected_set = set(expected)
    _require(set(value) == expected_set, f"{path} has wrong keys")


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _finite(value: Any, path: str) -> None:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value),
        f"{path} must be finite",
    )


def _digest(value: Any, path: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{path} must be a lowercase SHA-256",
    )


def _metric_paths() -> set[str]:
    paths = {"m_unique", "shared_confuser.E_shared"}
    for regime in REGIMES:
        for panel in PANELS:
            for operator in OPERATORS:
                for metric in OUTCOME_METRICS:
                    paths.add(f"{regime}.{panel}.operators.{operator}.{metric}")
            for advantage in ("A_F", "A_M"):
                paths.add(f"{regime}.{panel}.paired_advantages.{advantage}")
    return paths


def _validate_source(source: Any, *, activated: bool) -> None:
    keys = (
        "prelaunch_source_manifest_path",
        "prelaunch_source_manifest_sha256",
        "source_report_path",
        "source_report_sha256",
        "source_revision",
        "checkpoint_path",
        "checkpoint_sha256",
        "checkpoint_bytes",
        "checkpoint_epoch",
        "resolved_config_path",
        "resolved_config_sha256",
        "train_manifest_path",
        "train_manifest_sha256",
        "diagnostic_source_sha256",
        "activated_preregistration_sha256",
        "python_version",
        "torch_version",
        "numpy_version",
        "cuda_version",
        "cudnn_version",
    )
    _exact_keys(source, keys, "source")
    _require(
        source["prelaunch_source_manifest_path"]
        == "docs/pass201_pa_source_prelaunch_manifest.json",
        "wrong prelaunch source path",
    )
    _require(
        source["prelaunch_source_manifest_sha256"]
        == "37644551f99976a7982589c1574effa00a9c77aa4a690117b5a8cd84244cc803",
        "wrong prelaunch source digest",
    )
    for key in (
        "prelaunch_source_manifest_sha256",
        "source_report_sha256",
        "checkpoint_sha256",
        "resolved_config_sha256",
        "train_manifest_sha256",
        "diagnostic_source_sha256",
        "activated_preregistration_sha256",
    ):
        if not activated and source[key] is None:
            continue
        _digest(source[key], f"source.{key}")
    for key in (
        "source_report_path",
        "source_revision",
        "checkpoint_path",
        "resolved_config_path",
        "train_manifest_path",
        "python_version",
        "torch_version",
        "numpy_version",
        "cuda_version",
        "cudnn_version",
    ):
        if not activated and source[key] is None:
            continue
        _require(isinstance(source[key], str) and bool(source[key]), f"source.{key}")
    if activated or source["checkpoint_bytes"] is not None:
        _require(
            _is_int(source["checkpoint_bytes"]) and source["checkpoint_bytes"] > 0,
            "checkpoint_bytes",
        )
    if activated or source["checkpoint_epoch"] is not None:
        _require(
            _is_int(source["checkpoint_epoch"]) and source["checkpoint_epoch"] >= 0,
            "checkpoint_epoch",
        )


def _validate_constants(constants: Any, *, activated: bool) -> None:
    expected = {
        "batch_size": 180,
        "context_pairs": 32,
        "null_replicates": 256,
        "bootstrap_replicates": 20000,
        "s_prime_rank_seed": 2010809,
        "null_seed": 2010810,
        "bootstrap_seed": 2010811,
        "model_forward_seed": 2010812,
        "owner_margin_temperature": 0.05,
    }
    _exact_keys(
        constants,
        (*expected, "learning_rate", "coalition_weight", "proxy_learning_rate_multiplier"),
        "constants",
    )
    for key, expected_value in expected.items():
        _require(constants[key] == expected_value, f"constants.{key}")
    for key in ("learning_rate", "coalition_weight", "proxy_learning_rate_multiplier"):
        if not activated and constants[key] is None:
            continue
        _finite(constants[key], f"constants.{key}")


SUMMARY_KEYS = {
    "n",
    "mean",
    "median",
    "sample_sd",
    "q25",
    "q75",
    "lcb_0_005",
    "ucb_0_995",
}


def _validate_summary(summary: Any, path: str) -> None:
    _exact_keys(summary, SUMMARY_KEYS, path)
    _require(_is_int(summary["n"]) and summary["n"] == 32, f"{path}.n")
    for key in SUMMARY_KEYS - {"n"}:
        _finite(summary[key], f"{path}.{key}")
    _require(summary["lcb_0_005"] <= summary["ucb_0_995"], f"{path} bounds")


def _validate_update(update: Any, *, equal_norm: bool, path: str) -> None:
    keys = {
        "update_sha256",
        "parameter_update_norm",
        "R_F",
        "Delta_M",
        "D_F",
        "D_M",
        "reference_pa_norm",
        "norm_match_absolute_error",
    }
    _exact_keys(update, keys, path)
    _digest(update["update_sha256"], f"{path}.update_sha256")
    for key in ("parameter_update_norm", *OUTCOME_METRICS):
        _finite(update[key], f"{path}.{key}")
    if equal_norm:
        _finite(update["reference_pa_norm"], f"{path}.reference_pa_norm")
        _finite(update["norm_match_absolute_error"], f"{path}.norm_match_absolute_error")
        _require(update["reference_pa_norm"] >= 0, f"{path}.reference_pa_norm")
        _require(
            0
            <= update["norm_match_absolute_error"]
            <= 1e-10 * max(update["reference_pa_norm"], 1e-12),
            f"{path}.norm_match_absolute_error",
        )
    else:
        _require(update["reference_pa_norm"] is None, f"{path}.reference_pa_norm")
        _require(update["norm_match_absolute_error"] is None, f"{path}.norm_match_absolute_error")


def _validate_operator(operator: Any, name: str, representative_count: int, path: str) -> None:
    _exact_keys(operator, {"name", "loss", "representative_count", "panels"}, path)
    _require(operator["name"] == name, f"{path}.name")
    _finite(operator["loss"], f"{path}.loss")
    _require(
        operator["representative_count"] == representative_count, f"{path}.representative_count"
    )
    _exact_keys(operator["panels"], PANELS, f"{path}.panels")
    for panel_name, panel in operator["panels"].items():
        panel_path = f"{path}.panels.{panel_name}"
        keys = {
            "parameter_count",
            "gradient_sha256",
            "raw_gradient_norm",
            "update_space_norm",
            "auxiliary_to_pa_norm_ratio",
            "cosine_with_pa",
            "cosine_with_atomic_full_union",
            "cosine_with_summed_dropout",
            "scale_residual_to_summed_union",
            "updates",
        }
        _exact_keys(panel, keys, panel_path)
        _require(
            _is_int(panel["parameter_count"]) and panel["parameter_count"] > 0,
            f"{panel_path}.parameter_count",
        )
        _digest(panel["gradient_sha256"], f"{panel_path}.gradient_sha256")
        for key in (
            "raw_gradient_norm",
            "update_space_norm",
            "auxiliary_to_pa_norm_ratio",
            "cosine_with_pa",
            "cosine_with_atomic_full_union",
            "cosine_with_summed_dropout",
        ):
            _finite(panel[key], f"{panel_path}.{key}")
        if name in {"atomic_one_hot", "atomic_complementary", "atomic_full_union"}:
            _finite(
                panel["scale_residual_to_summed_union"],
                f"{panel_path}.scale_residual_to_summed_union",
            )
        else:
            _require(
                panel["scale_residual_to_summed_union"] is None,
                f"{panel_path}.scale_residual_to_summed_union",
            )
        _exact_keys(panel["updates"], REGIMES, f"{panel_path}.updates")
        for regime in REGIMES:
            _validate_update(
                panel["updates"][regime],
                equal_norm=regime == "equal_norm",
                path=f"{panel_path}.updates.{regime}",
            )


CROSS_REUSE_KEYS = {
    "prior_context_indices_sharing_s_ids",
    "prior_context_indices_sharing_s_prime_ids",
    "prior_context_indices_sharing_any_ids",
    "reused_s_image_count",
    "reused_s_prime_image_count",
    "reused_any_image_count",
    "reused_label_count",
}


def _validate_context(context: Any, expected_index: int) -> None:
    keys = {
        "context_index",
        "production_epoch",
        "production_batch_index",
        "batch_size",
        "m_unique",
        "row_example_ids",
        "row_sample_indices",
        "row_labels",
        "class_multiplicities",
        "representative_row_indices",
        "representative_sample_indices",
        "s_tensor_sha256",
        "s_prime_example_ids",
        "s_prime_sample_indices",
        "s_prime_tensor_sha256",
        "cross_context_reuse",
        "foreign_proxy_rows",
        "shared_confuser",
        "operators",
    }
    path = f"contexts[{expected_index}]"
    _exact_keys(context, keys, path)
    _require(context["context_index"] == expected_index, f"{path}.context_index")
    _require(context["production_epoch"] == 0, f"{path}.production_epoch")
    _require(
        _is_int(context["production_batch_index"])
        and context["production_batch_index"] >= expected_index,
        f"{path}.production_batch_index",
    )
    _require(context["batch_size"] == 180, f"{path}.batch_size")
    _require(_is_int(context["m_unique"]) and context["m_unique"] > 0, f"{path}.m_unique")
    for key in (
        "row_example_ids",
        "row_sample_indices",
        "row_labels",
        "s_prime_example_ids",
        "s_prime_sample_indices",
    ):
        _require(isinstance(context[key], list) and len(context[key]) == 180, f"{path}.{key}")
    _require(len(set(context["s_prime_example_ids"])) == 180, f"{path}.s_prime_example_ids")
    _require(
        set(context["row_example_ids"]).isdisjoint(context["s_prime_example_ids"]),
        f"{path} disjointness",
    )
    _require(context["m_unique"] == len(set(context["row_labels"])), f"{path}.m_unique")
    expected_multiplicities = {
        str(label): count for label, count in Counter(context["row_labels"]).items()
    }
    actual_multiplicities = {
        str(label): count for label, count in context["class_multiplicities"].items()
    }
    _require(
        len(actual_multiplicities) == len(context["class_multiplicities"])
        and actual_multiplicities == expected_multiplicities,
        f"{path}.class_multiplicities",
    )
    _require(
        len(context["representative_row_indices"]) == context["m_unique"],
        f"{path}.representative_row_indices",
    )
    _require(
        len(context["representative_sample_indices"]) == context["m_unique"],
        f"{path}.representative_sample_indices",
    )
    expected_representative_rows = []
    expected_representative_samples = []
    for label in sorted(set(context["row_labels"])):
        row_index = min(
            (index for index, row_label in enumerate(context["row_labels"]) if row_label == label),
            key=lambda index: (context["row_sample_indices"][index], index),
        )
        expected_representative_rows.append(row_index)
        expected_representative_samples.append(context["row_sample_indices"][row_index])
    _require(
        context["representative_row_indices"] == expected_representative_rows,
        f"{path}.representative_row_indices",
    )
    _require(
        context["representative_sample_indices"] == expected_representative_samples,
        f"{path}.representative_sample_indices",
    )
    _digest(context["s_tensor_sha256"], f"{path}.s_tensor_sha256")
    _digest(context["s_prime_tensor_sha256"], f"{path}.s_prime_tensor_sha256")
    _exact_keys(context["cross_context_reuse"], CROSS_REUSE_KEYS, f"{path}.cross_context_reuse")
    for key in CROSS_REUSE_KEYS:
        value = context["cross_context_reuse"][key]
        if key.startswith("prior_"):
            _require(
                isinstance(value, list)
                and value == sorted(set(value))
                and all(_is_int(item) and 0 <= item < expected_index for item in value),
                f"{path}.cross_context_reuse.{key}",
            )
        else:
            _require(_is_int(value) and value >= 0, f"{path}.cross_context_reuse.{key}")
    _require(
        _is_int(context["foreign_proxy_rows"]) and context["foreign_proxy_rows"] > 0,
        f"{path}.foreign_proxy_rows",
    )
    shared = context["shared_confuser"]
    _exact_keys(
        shared,
        {"A_aligned", "null_mean", "E_shared", "null_distribution_sha256"},
        f"{path}.shared_confuser",
    )
    for key in ("A_aligned", "null_mean", "E_shared"):
        _finite(shared[key], f"{path}.shared_confuser.{key}")
    _digest(shared["null_distribution_sha256"], f"{path}.shared_confuser.null_distribution_sha256")
    _exact_keys(context["operators"], OPERATORS, f"{path}.operators")
    for name in OPERATORS:
        _validate_operator(
            context["operators"][name], name, context["m_unique"], f"{path}.operators.{name}"
        )


def _validate_aggregate_regime(regime: Any, path: str) -> None:
    _exact_keys(regime, PANELS, path)
    for panel_name, panel in regime.items():
        panel_path = f"{path}.{panel_name}"
        _exact_keys(panel, {"operators", "paired_advantages"}, panel_path)
        _exact_keys(panel["operators"], OPERATORS, f"{panel_path}.operators")
        for operator, metrics in panel["operators"].items():
            _exact_keys(metrics, OUTCOME_METRICS, f"{panel_path}.operators.{operator}")
            for metric, summary in metrics.items():
                _validate_summary(summary, f"{panel_path}.operators.{operator}.{metric}")
        _exact_keys(panel["paired_advantages"], {"A_F", "A_M"}, f"{panel_path}.paired_advantages")
        for metric, summary in panel["paired_advantages"].items():
            _validate_summary(summary, f"{panel_path}.paired_advantages.{metric}")


def _validate_aggregates(aggregates: Any) -> None:
    _exact_keys(aggregates, {"m_unique", *REGIMES, "shared_confuser", "bootstrap"}, "aggregates")
    _validate_summary(aggregates["m_unique"], "aggregates.m_unique")
    _validate_summary(aggregates["shared_confuser"], "aggregates.shared_confuser")
    for regime in REGIMES:
        _validate_aggregate_regime(aggregates[regime], f"aggregates.{regime}")
    bootstrap = aggregates["bootstrap"]
    _exact_keys(
        bootstrap,
        {
            "seed",
            "replicates",
            "quantile_method",
            "joint_context_index_sha256",
            "distribution_sha256_by_metric",
        },
        "aggregates.bootstrap",
    )
    _require(bootstrap["seed"] == BOOTSTRAP_SEED, "aggregates.bootstrap.seed")
    _require(bootstrap["replicates"] == BOOTSTRAP_REPLICATES, "aggregates.bootstrap.replicates")
    _require(bootstrap["quantile_method"] == "linear", "aggregates.bootstrap.quantile_method")
    _digest(
        bootstrap["joint_context_index_sha256"], "aggregates.bootstrap.joint_context_index_sha256"
    )
    distributions = bootstrap["distribution_sha256_by_metric"]
    _exact_keys(
        distributions, _metric_paths(), "aggregates.bootstrap.distribution_sha256_by_metric"
    )
    for path, digest in distributions.items():
        _digest(digest, f"aggregates.bootstrap.distribution_sha256_by_metric.{path}")


PROCESS_KEYS = {
    "role",
    "pid",
    "accelerator",
    "python_version",
    "torch_version",
    "cuda_version",
    "cudnn_version",
    "visible_cuda_devices",
    "initial_python_rng_sha256",
    "initial_numpy_rng_sha256",
    "initial_torch_cpu_rng_sha256",
    "initial_torch_cuda_rng_sha256_by_device",
    "prepared_context_count",
    "input_context_digest_records",
    "context0_record_sha256",
}
INPUT_DIGEST_KEYS = {
    "context_index",
    "s_tensor_sha256",
    "s_prime_tensor_sha256",
    "metadata_sha256",
    "combined_sha256",
}


def _validate_input_digest(record: Any, expected_index: int, path: str) -> None:
    _exact_keys(record, INPUT_DIGEST_KEYS, path)
    _require(record["context_index"] == expected_index, f"{path}.context_index")
    for key in INPUT_DIGEST_KEYS - {"context_index"}:
        _digest(record[key], f"{path}.{key}")
    combined = {key: record[key] for key in INPUT_DIGEST_KEYS - {"combined_sha256"}}
    _require(
        hashlib.sha256(canonical_json_bytes(combined)).hexdigest() == record["combined_sha256"],
        f"{path}.combined_sha256",
    )


def _validate_process_record(record: Any, role: str, path: str, *, context0_required: bool) -> None:
    _exact_keys(record, PROCESS_KEYS, path)
    _require(record["role"] == role, f"{path}.role")
    _require(_is_int(record["pid"]) and record["pid"] > 0, f"{path}.pid")
    for key in ("accelerator", "python_version", "torch_version", "cuda_version", "cudnn_version"):
        _require(isinstance(record[key], str) and bool(record[key]), f"{path}.{key}")
    _require(
        isinstance(record["visible_cuda_devices"], list)
        and bool(record["visible_cuda_devices"])
        and all(isinstance(item, str) and item for item in record["visible_cuda_devices"]),
        f"{path}.visible_cuda_devices",
    )
    _require(
        record["visible_cuda_devices"] == sorted(record["visible_cuda_devices"]),
        f"{path}.visible_cuda_devices order",
    )
    for key in (
        "initial_python_rng_sha256",
        "initial_numpy_rng_sha256",
        "initial_torch_cpu_rng_sha256",
    ):
        _digest(record[key], f"{path}.{key}")
    cuda_hashes = record["initial_torch_cuda_rng_sha256_by_device"]
    _require(
        isinstance(cuda_hashes, dict) and bool(cuda_hashes),
        f"{path}.initial_torch_cuda_rng_sha256_by_device",
    )
    _require(
        list(cuda_hashes) == [str(index) for index in range(len(record["visible_cuda_devices"]))],
        f"{path}.initial_torch_cuda_rng_sha256_by_device order",
    )
    for key, value in cuda_hashes.items():
        _require(str(int(key)) == key, f"{path}.CUDA device index")
        _digest(value, f"{path}.initial_torch_cuda_rng_sha256_by_device.{key}")
    _require(
        _is_int(record["prepared_context_count"]) and record["prepared_context_count"] >= 0,
        f"{path}.prepared_context_count",
    )
    digest_records = record["input_context_digest_records"]
    _require(
        isinstance(digest_records, list)
        and len(digest_records) == record["prepared_context_count"],
        f"{path}.input_context_digest_records",
    )
    for index, digest_record in enumerate(digest_records):
        _validate_input_digest(
            digest_record, index, f"{path}.input_context_digest_records[{index}]"
        )
    if context0_required:
        _digest(record["context0_record_sha256"], f"{path}.context0_record_sha256")
    else:
        _require(
            record["context0_record_sha256"] is None
            or isinstance(record["context0_record_sha256"], str),
            f"{path}.context0_record_sha256",
        )
        if record["context0_record_sha256"] is not None:
            _digest(record["context0_record_sha256"], f"{path}.context0_record_sha256")


def _validate_scored_integrity(integrity: Any, contexts: list[dict[str, Any]]) -> None:
    keys = {
        "accepted_context_count",
        "rejected_context_count",
        "invalid_context_count",
        "input_replay_verified",
        "parameter_hash_before",
        "parameter_hash_after",
        "buffer_hash_before",
        "buffer_hash_after",
        "training_flags_restored",
        "deterministic_process_verified",
        "first_context_operator_replay_verified",
        "deterministic_settings",
        "process_records",
        "replay_residuals",
        "all_finite",
    }
    _exact_keys(integrity, keys, "integrity")
    _require(integrity["accepted_context_count"] == 32, "integrity.accepted_context_count")
    _require(
        _is_int(integrity["rejected_context_count"]) and integrity["rejected_context_count"] >= 0,
        "integrity.rejected_context_count",
    )
    _require(integrity["invalid_context_count"] == 0, "integrity.invalid_context_count")
    for key in (
        "input_replay_verified",
        "training_flags_restored",
        "deterministic_process_verified",
        "first_context_operator_replay_verified",
        "all_finite",
    ):
        _require(integrity[key] is True, f"integrity.{key}")
    for key in (
        "parameter_hash_before",
        "parameter_hash_after",
        "buffer_hash_before",
        "buffer_hash_after",
    ):
        _digest(integrity[key], f"integrity.{key}")
    _require(
        integrity["parameter_hash_before"] == integrity["parameter_hash_after"],
        "parameter hashes differ",
    )
    _require(
        integrity["buffer_hash_before"] == integrity["buffer_hash_after"], "buffer hashes differ"
    )
    settings = integrity["deterministic_settings"]
    expected_settings = {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "autocast": False,
        "dtype": "float32",
    }
    _exact_keys(settings, expected_settings, "integrity.deterministic_settings")
    _require(settings == expected_settings, "deterministic settings mismatch")
    records = integrity["process_records"]
    roles = ("integrity_replay_a", "integrity_replay_b", "scientific")
    _require(isinstance(records, list) and len(records) == 3, "integrity.process_records")
    context_zero_hash = hashlib.sha256(canonical_json_bytes(contexts[0])).hexdigest()
    canonical_digests: list[dict[str, Any]] | None = None
    for index, (record, role) in enumerate(zip(records, roles, strict=True)):
        _validate_process_record(
            record, role, f"integrity.process_records[{index}]", context0_required=True
        )
        _require(
            record["prepared_context_count"] == 32,
            f"integrity.process_records[{index}].prepared_context_count",
        )
        _require(
            record["context0_record_sha256"] == context_zero_hash,
            f"integrity.process_records[{index}].context0_record_sha256",
        )
        if canonical_digests is None:
            canonical_digests = record["input_context_digest_records"]
        else:
            _require(
                record["input_context_digest_records"] == canonical_digests,
                "input context replays differ",
            )
        for context, digest_record in zip(
            contexts, record["input_context_digest_records"], strict=True
        ):
            _require(
                digest_record["s_tensor_sha256"] == context["s_tensor_sha256"],
                "contained S tensor hash mismatch",
            )
            _require(
                digest_record["s_prime_tensor_sha256"] == context["s_prime_tensor_sha256"],
                "contained S-prime tensor hash mismatch",
            )
            expected_metadata = hashlib.sha256(
                canonical_json_bytes(_metadata_for_digest(context))
            ).hexdigest()
            _require(
                digest_record["metadata_sha256"] == expected_metadata,
                "contained metadata hash mismatch",
            )
    residuals = integrity["replay_residuals"]
    _exact_keys(
        residuals,
        {
            "pair_count",
            "tensor_max_absolute",
            "scalar_max_relative",
            "tensor_tolerance",
            "scalar_tolerance",
            "scalar_denominator",
        },
        "integrity.replay_residuals",
    )
    _require(residuals["pair_count"] == 3, "integrity.replay_residuals.pair_count")
    for key in (
        "tensor_max_absolute",
        "scalar_max_relative",
        "tensor_tolerance",
        "scalar_tolerance",
    ):
        _finite(residuals[key], f"integrity.replay_residuals.{key}")
    _require(
        residuals["tensor_tolerance"] == 2e-6 and residuals["tensor_max_absolute"] <= 2e-6,
        "tensor replay tolerance",
    )
    _require(
        residuals["scalar_tolerance"] == 1e-5 and residuals["scalar_max_relative"] <= 1e-5,
        "scalar replay tolerance",
    )
    _require(residuals["scalar_denominator"] == "max(abs(a),abs(b),1e-12)", "scalar denominator")


def _validate_partial_context(context: Any, expected_index: int) -> None:
    keys = {
        "context_index",
        "production_epoch",
        "production_batch_index",
        "status",
        "rejection_code",
        "row_example_ids",
        "row_sample_indices",
        "row_labels",
        "class_multiplicities",
        "representative_row_indices",
        "representative_sample_indices",
        "s_prime_example_ids",
        "s_prime_sample_indices",
    }
    path = f"contexts[{expected_index}]"
    _exact_keys(context, keys, path)
    _require(context["context_index"] == expected_index, f"{path}.context_index")
    _require(context["production_epoch"] == 0, f"{path}.production_epoch")
    for key in ("row_example_ids", "row_sample_indices", "row_labels"):
        _require(
            isinstance(context[key], list) and len(context[key]) == 180,
            f"{path}.{key}",
        )
    expected_multiplicities = {
        str(label): count for label, count in Counter(context["row_labels"]).items()
    }
    actual_multiplicities = {
        str(label): count for label, count in context["class_multiplicities"].items()
    }
    _require(
        len(actual_multiplicities) == len(context["class_multiplicities"])
        and actual_multiplicities == expected_multiplicities,
        f"{path}.class_multiplicities",
    )
    expected_rows = []
    expected_samples = []
    for label in sorted(set(context["row_labels"])):
        row_index = min(
            (index for index, row_label in enumerate(context["row_labels"]) if row_label == label),
            key=lambda index: (context["row_sample_indices"][index], index),
        )
        expected_rows.append(row_index)
        expected_samples.append(context["row_sample_indices"][row_index])
    _require(
        context["representative_row_indices"] == expected_rows,
        f"{path}.representative_row_indices",
    )
    _require(
        context["representative_sample_indices"] == expected_samples,
        f"{path}.representative_sample_indices",
    )
    _require(context["status"] in {"accepted", "rejected"}, f"{path}.status")
    if context["status"] == "accepted":
        _require(context["rejection_code"] is None, f"{path}.rejection_code")
        _require(
            len(context["s_prime_example_ids"]) == len(context["row_example_ids"]),
            f"{path}.s_prime_example_ids",
        )
    else:
        _require(
            context["rejection_code"] == "INSUFFICIENT_DISJOINT_S_PRIME", f"{path}.rejection_code"
        )
        _require(
            context["s_prime_example_ids"] == [] and context["s_prime_sample_indices"] == [],
            f"{path}.s_prime",
        )


def _failure_evidence_digest(
    status: str, reason_codes: list[str], integrity: Mapping[str, Any]
) -> str:
    records = integrity["process_records"]
    evidence = {
        "status": status,
        "reason_codes": sorted(reason_codes),
        "stage": integrity["stage"],
        "accepted_context_count": integrity["accepted_context_count"],
        "rejected_context_count": integrity["rejected_context_count"],
        "invalid_context_count": integrity["invalid_context_count"],
        "last_process_record": records[-1] if records else None,
    }
    return hashlib.sha256(canonical_json_bytes(evidence)).hexdigest()


def _validate_reduced_integrity(
    integrity: Any, status: str, reason_codes: list[str], *, early: bool
) -> None:
    keys = {
        "stage",
        "accepted_context_count",
        "rejected_context_count",
        "invalid_context_count",
        "input_replay_verified",
        "deterministic_process_verified",
        "process_records",
        "failure_evidence_sha256",
        "all_finite",
    }
    _exact_keys(integrity, keys, "integrity")
    stages = (
        "source_activation",
        "context_construction",
        "integrity_replay_a",
        "integrity_replay_b",
        "scientific",
    )
    _require(integrity["stage"] in stages, "integrity.stage")
    for key in ("accepted_context_count", "rejected_context_count", "invalid_context_count"):
        _require(_is_int(integrity[key]) and integrity[key] >= 0, f"integrity.{key}")
    for key in ("input_replay_verified", "deterministic_process_verified", "all_finite"):
        _require(isinstance(integrity[key], bool), f"integrity.{key}")
    records = integrity["process_records"]
    expected_count_by_stage = {
        "source_activation": 0,
        "context_construction": 1,
        "integrity_replay_a": 1,
        "integrity_replay_b": 2,
        "scientific": 3,
    }
    _require(
        isinstance(records, list) and len(records) == expected_count_by_stage[integrity["stage"]],
        "integrity.process_records prefix",
    )
    roles = ("integrity_replay_a", "integrity_replay_b", "scientific")
    for index, record in enumerate(records):
        _validate_process_record(
            record, roles[index], f"integrity.process_records[{index}]", context0_required=False
        )
    _digest(integrity["failure_evidence_sha256"], "integrity.failure_evidence_sha256")
    _require(
        integrity["failure_evidence_sha256"]
        == _failure_evidence_digest(status, reason_codes, integrity),
        "failure_evidence_sha256 mismatch",
    )
    if early:
        _require(integrity["stage"] == "context_construction", "early unresolved stage")
        _require(0 <= integrity["accepted_context_count"] <= 31, "early accepted count")
        _require(
            integrity["rejected_context_count"] >= 1 and integrity["invalid_context_count"] == 0,
            "early context counts",
        )
        _require(
            integrity["input_replay_verified"] is False
            and integrity["deterministic_process_verified"] is False
            and integrity["all_finite"] is True,
            "early integrity flags",
        )
        _require(
            records[0]["prepared_context_count"] == integrity["accepted_context_count"]
            and records[0]["context0_record_sha256"] is None,
            "early process record",
        )
    elif status == "BLOCKED":
        _require(
            integrity["stage"] == "source_activation"
            and all(
                integrity[key] == 0
                for key in (
                    "accepted_context_count",
                    "rejected_context_count",
                    "invalid_context_count",
                )
            ),
            "blocked source activation",
        )
        _require(
            integrity["input_replay_verified"] is False
            and integrity["deterministic_process_verified"] is False
            and integrity["all_finite"] is False,
            "blocked integrity flags",
        )
    else:
        if integrity["stage"] == "source_activation":
            _require(
                integrity["invalid_context_count"] == 0
                and integrity["accepted_context_count"] == 0
                and integrity["rejected_context_count"] == 0,
                "source activation counts",
            )
        else:
            _require(integrity["invalid_context_count"] >= 1, "invalid context count")


def _summary_at(aggregates: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    mapping = {
        "shared_confuser_excess": aggregates["shared_confuser"],
        "network_equal_union_advantage_foreign": aggregates["equal_norm"]["network_only"][
            "paired_advantages"
        ]["A_F"],
        "network_equal_union_advantage_margin": aggregates["equal_norm"]["network_only"][
            "paired_advantages"
        ]["A_M"],
        "network_equal_union_foreign_suppression": aggregates["equal_norm"]["network_only"][
            "operators"
        ]["summed_union"]["R_F"],
        "network_equal_union_margin_change": aggregates["equal_norm"]["network_only"]["operators"][
            "summed_union"
        ]["Delta_M"],
        "network_equal_union_predicted_suppression": aggregates["equal_norm"]["network_only"][
            "operators"
        ]["summed_union"]["D_F"],
        "network_equal_union_predicted_margin_change": aggregates["equal_norm"]["network_only"][
            "operators"
        ]["summed_union"]["D_M"],
        "joint_equal_union_advantage_foreign": aggregates["equal_norm"]["joint_including_proxies"][
            "paired_advantages"
        ]["A_F"],
        "joint_equal_union_advantage_margin": aggregates["equal_norm"]["joint_including_proxies"][
            "paired_advantages"
        ]["A_M"],
        "joint_equal_union_foreign_suppression": aggregates["equal_norm"][
            "joint_including_proxies"
        ]["operators"]["summed_union"]["R_F"],
        "joint_equal_union_margin_change": aggregates["equal_norm"]["joint_including_proxies"][
            "operators"
        ]["summed_union"]["Delta_M"],
    }
    return mapping[key]


def _component_decisions(aggregates: Mapping[str, Any]) -> dict[str, str]:
    decisions = {}
    owner_margin = {
        "network_equal_union_margin_change",
        "network_equal_union_predicted_margin_change",
        "joint_equal_union_margin_change",
    }
    for key, threshold in THRESHOLDS.items():
        summary = _summary_at(aggregates, key)
        if summary["lcb_0_005"] >= threshold:
            decisions[key] = "PASS"
        elif summary["ucb_0_995"] < 0 if key in owner_margin else summary["ucb_0_995"] <= 0:
            decisions[key] = "FAIL"
        else:
            decisions[key] = "UNRESOLVED"
    return decisions


def _failure_reasons(aggregates: Mapping[str, Any], decisions: Mapping[str, str]) -> list[str]:
    reasons = []
    if decisions["shared_confuser_excess"] == "FAIL":
        reasons.append("FAIL_NO_SHARED_CONFOUNDER")
    if any(
        decisions[key] == "FAIL"
        for key in ("network_equal_union_advantage_foreign", "network_equal_union_advantage_margin")
    ):
        reasons.append("FAIL_NO_COALITION_SPECIFIC_ACTION")
    if any(
        decisions[key] == "FAIL"
        for key in (
            "network_equal_union_foreign_suppression",
            "network_equal_union_margin_change",
            "network_equal_union_predicted_suppression",
        )
    ):
        reasons.append("FAIL_NOT_VIABLE")
    configured_joint = aggregates["configured_loss_stateless"]["joint_including_proxies"][
        "paired_advantages"
    ]
    equal_joint = aggregates["equal_norm"]["joint_including_proxies"]["paired_advantages"]
    if all(configured_joint[key]["lcb_0_005"] >= 0 for key in ("A_F", "A_M")) and any(
        equal_joint[key]["ucb_0_995"] <= 0 for key in ("A_F", "A_M")
    ):
        reasons.append("FAIL_SCALE_SUFFICIENT")
    joint_keys = (
        "joint_equal_union_advantage_foreign",
        "joint_equal_union_advantage_margin",
        "joint_equal_union_foreign_suppression",
        "joint_equal_union_margin_change",
    )
    network_keys = (
        "network_equal_union_advantage_foreign",
        "network_equal_union_advantage_margin",
        "network_equal_union_foreign_suppression",
        "network_equal_union_margin_change",
    )
    if all(decisions[key] == "PASS" for key in joint_keys) and any(
        decisions[key] == "FAIL" for key in network_keys
    ):
        reasons.append("FAIL_PROXY_ONLY")
    union = aggregates["equal_norm"]["network_only"]["operators"]["summed_union"]
    if union["D_F"]["lcb_0_005"] > 0 and union["D_M"]["ucb_0_995"] < 0:
        reasons.append("FAIL_OWNER_DAMAGE")
    return reasons


def _validate_decision(decision: Any, status: str, aggregates: Mapping[str, Any] | None) -> None:
    if aggregates is None:
        _exact_keys(decision, {"thresholds", "overall", "authorized_next_action"}, "decision")
        _require(decision["overall"] == status, "decision.overall")
        _require(decision["authorized_next_action"] == "none", "decision.authorized_next_action")
    else:
        _exact_keys(
            decision,
            {"thresholds", "component_decisions", "overall", "authorized_next_action"},
            "decision",
        )
        expected_components = _component_decisions(aggregates)
        _exact_keys(decision["component_decisions"], THRESHOLDS, "decision.component_decisions")
        _require(
            decision["component_decisions"] == expected_components, "component decisions mismatch"
        )
        if any(value == "FAIL" for value in expected_components.values()):
            expected_status = "FAIL"
        elif all(value == "PASS" for value in expected_components.values()):
            expected_status = "PASS"
        else:
            expected_status = "UNRESOLVED"
        _require(
            status == expected_status and decision["overall"] == expected_status,
            "status/decision mismatch",
        )
        expected_action = "write_separate_gpu_preregistration" if status == "PASS" else "none"
        _require(
            decision["authorized_next_action"] == expected_action, "decision.authorized_next_action"
        )
    _exact_keys(decision["thresholds"], THRESHOLDS, "decision.thresholds")
    _require(decision["thresholds"] == THRESHOLDS, "decision.thresholds mismatch")


def validate_payload_structure(payload: Mapping[str, Any]) -> None:
    """Fail closed on any violation of a frozen conditional result schema."""

    _require(isinstance(payload, dict), "payload must be an object")
    status = payload.get("status")
    _require(status in {"PASS", "FAIL", "UNRESOLVED", "BLOCKED", "INVALID"}, "status")
    common = {
        "schema_version",
        "status",
        "reason_codes",
        "candidate_values_computed",
        "uses_test_data",
        "source",
        "constants",
        "decision",
        "integrity",
    }
    scored = payload.get("candidate_values_computed") is True
    early = status == "UNRESOLVED" and payload.get("candidate_values_computed") is False
    expected_keys = common | (
        {"contexts", "aggregates"} if scored else ({"contexts"} if early else set())
    )
    _exact_keys(payload, expected_keys, "payload")
    _require(payload["schema_version"] == "pass201-cis-operator-v1", "schema_version")
    _require(payload["uses_test_data"] == "artifact_binding_only", "uses_test_data")
    _require(
        isinstance(payload["reason_codes"], list)
        and len(payload["reason_codes"]) == len(set(payload["reason_codes"]))
        and all(isinstance(code, str) and code for code in payload["reason_codes"]),
        "reason_codes",
    )
    activated = not (
        not scored and not early and payload["integrity"].get("stage") == "source_activation"
    )
    _validate_source(payload["source"], activated=activated)
    _validate_constants(payload["constants"], activated=activated)
    if scored:
        _require(status in {"PASS", "FAIL", "UNRESOLVED"}, "scored status")
        contexts = payload["contexts"]
        _require(isinstance(contexts, list) and len(contexts) == 32, "contexts")
        for index, context in enumerate(contexts):
            _validate_context(context, index)
        _validate_aggregates(payload["aggregates"])
        _validate_decision(payload["decision"], status, payload["aggregates"])
        expected_reasons = _failure_reasons(
            payload["aggregates"], _component_decisions(payload["aggregates"])
        )
        _require(payload["reason_codes"] == expected_reasons, "reason codes mismatch")
        _validate_scored_integrity(payload["integrity"], contexts)
    elif early:
        _require(
            payload["reason_codes"] == ["UNRESOLVED_INSUFFICIENT_DISJOINT_CONTEXTS"],
            "early unresolved reason",
        )
        contexts = payload["contexts"]
        _require(isinstance(contexts, list) and bool(contexts), "partial contexts")
        for index, context in enumerate(contexts):
            _validate_partial_context(context, index)
        _require(
            sum(context["status"] == "accepted" for context in contexts)
            == payload["integrity"]["accepted_context_count"],
            "partial accepted count",
        )
        _require(
            sum(context["status"] == "rejected" for context in contexts)
            == payload["integrity"]["rejected_context_count"],
            "partial rejected count",
        )
        _validate_decision(payload["decision"], status, None)
        _validate_reduced_integrity(
            payload["integrity"], status, payload["reason_codes"], early=True
        )
    else:
        _require(status in {"BLOCKED", "INVALID"}, "non-scored status")
        _validate_decision(payload["decision"], status, None)
        _validate_reduced_integrity(
            payload["integrity"], status, payload["reason_codes"], early=False
        )


def _sha256_float64_vector(values: Any, expected_length: int, path: str) -> str:
    array = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    _require(array.shape == (expected_length,) and np.isfinite(array).all(), path)
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def validate_construction_evidence(
    payload: Mapping[str, Any], raw_evidence: Mapping[str, Any]
) -> None:
    """Recompute all digests that require still-live raw scored evidence."""

    validate_payload_structure(payload)
    _require(
        payload["candidate_values_computed"] is True,
        "construction evidence requires a scored payload",
    )
    keys = {
        "gradient_tensors",
        "update_tensors",
        "null_distributions",
        "bootstrap_indices",
        "bootstrap_distributions",
    }
    _exact_keys(raw_evidence, keys, "raw_evidence")
    gradients = raw_evidence["gradient_tensors"]
    updates = raw_evidence["update_tensors"]
    nulls = raw_evidence["null_distributions"]
    expected_gradient_keys = set()
    expected_update_keys = set()
    for context in payload["contexts"]:
        context_index = context["context_index"]
        for operator in OPERATORS:
            for panel in PANELS:
                key = f"{context_index}.{operator}.{panel}"
                expected_gradient_keys.add(key)
                actual = sha256_named_tensors(gradients.get(key, ()))
                expected = context["operators"][operator]["panels"][panel]["gradient_sha256"]
                _require(actual == expected, f"gradient_sha256 mismatch at {key}")
                for regime in REGIMES:
                    update_key = f"{key}.{regime}"
                    expected_update_keys.add(update_key)
                    actual_update = sha256_named_tensors(updates.get(update_key, ()))
                    expected_update = context["operators"][operator]["panels"][panel]["updates"][
                        regime
                    ]["update_sha256"]
                    _require(
                        actual_update == expected_update, f"update_sha256 mismatch at {update_key}"
                    )
        null_key = str(context_index)
        actual_null = _sha256_float64_vector(
            nulls.get(null_key, ()), 256, f"null distribution {null_key}"
        )
        _require(
            actual_null == context["shared_confuser"]["null_distribution_sha256"],
            f"null_distribution_sha256 mismatch at {null_key}",
        )
    _exact_keys(gradients, expected_gradient_keys, "raw_evidence.gradient_tensors")
    _exact_keys(updates, expected_update_keys, "raw_evidence.update_tensors")
    _exact_keys(nulls, {str(index) for index in range(32)}, "raw_evidence.null_distributions")
    indices = np.asarray(raw_evidence["bootstrap_indices"])
    _require(indices.dtype.str == "<i8" and indices.flags.c_contiguous, "bootstrap index encoding")
    _require(
        np.array_equal(indices, bootstrap_indices()), "bootstrap indices differ from frozen matrix"
    )
    expected_index_digest = payload["aggregates"]["bootstrap"]["joint_context_index_sha256"]
    _require(
        sha256_bootstrap_indices(indices) == expected_index_digest,
        "joint_context_index_sha256 mismatch",
    )
    distributions = raw_evidence["bootstrap_distributions"]
    expected_distribution_digests = payload["aggregates"]["bootstrap"][
        "distribution_sha256_by_metric"
    ]
    _exact_keys(
        distributions, expected_distribution_digests, "raw_evidence.bootstrap_distributions"
    )
    for metric_path, expected_digest in expected_distribution_digests.items():
        actual = _sha256_float64_vector(distributions[metric_path], 20000, metric_path)
        _require(actual == expected_digest, f"distribution_sha256 mismatch at {metric_path}")
