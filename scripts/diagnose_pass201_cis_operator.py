"""Prospectively frozen, source-bound Pass201 CIS operator diagnostic.

The module stays import-side-effect-free; torch and model/data code load only in
an authenticated child process.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import math
import os
import pickle
import platform
import random
import re
import stat
import struct
import subprocess
import sys
import tempfile
import zipfile
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

S_PRIME_RANK_SEED = 2010809
BOOTSTRAP_SEED = 2010811
BOOTSTRAP_REPLICATES = 20_000
CONTEXT_PAIRS = 32
PRELAUNCH_SOURCE_MANIFEST_PATH = "docs/pass201_pa_source_prelaunch_manifest.json"
SOURCE_V3_AUTHORIZATION_MANIFEST_PATH = "docs/pass201_pa_source_v3_authorization_manifest.json"
SOURCE_V4_AUTHORIZATION_MANIFEST_PATH = "docs/pass201_pa_source_v4_authorization_manifest.json"
SOURCE_V5_AUTHORIZATION_MANIFEST_PATH = "docs/pass201_pa_source_v5_authorization_manifest.json"
SOURCE_V3_I3_COMMIT = "23e7ff5c82fb28cd9fdd8e9e819e34b8fc9aacde"
SOURCE_V3_I3A_COMMIT = "757d0672fc4409d7bc5076004bc3797d0c7b3cde"
SOURCE_V3_V3_COMMIT = "03d0ed509fe7b65aee0162941a9f6a3b6fea228f"
SOURCE_V3_H3_COMMIT = "183fa5b9cf99b7f860e954c9be38c06a477b3912"
PROCESS_ENTRY_DRAFT_AMENDMENT_COMMIT = "967a02d5d1535dd2a019f3b34039f0a706796310"
PROCESS_ENTRY_DRAFT_PLAN_COMMIT = "6067219a3a312053cadfaeb4cfa8d8d5fb907b9c"
PROCESS_ENTRY_F4_COMMIT = "463985d86afd1ea54ff021df1cf8624b6aa013d1"
PROCESS_ENTRY_F5_COMMIT = "ed743aff23792b13209f5974f2a74f26c0104f74"
SOURCE_V3_CHANGED_PATHS = tuple(
    sorted(
        (
            "scripts/run_pass201_pa_source_v2.py",
            "scripts/pass201_pa_source_v2_contract.py",
            "scripts/diagnose_pass201_cis_operator.py",
            "tests/test_run_pass201_pa_source_v2.py",
            "tests/test_pass201_pa_source_v2_contract.py",
            "tests/test_diagnose_pass201_cis_operator.py",
        ),
        key=lambda value: value.encode("utf-8"),
    )
)
SOURCE_V3_STATIC_AUTHORITIES = {
    "protocol": {
        "path": "docs/pass201_pa_source_v3_protocol_2026-08-11.md",
        "sha256": "716460eda8664a4c37b5f14332244a8dae4f921b393b7e4c085ff0b4e26a7426",
        "commit": "9782eb44f4a087682563d8a1f4e075f4fcdd165b",
    },
    "plan": {
        "path": "docs/superpowers/plans/2026-08-11-pass201-pa-source-v3.md",
        "sha256": "351abb720c7526ce71f5ccea85e5ee16385b8e1d79df9073309ef5b4321ba3ae",
        "commit": "f38af4465333f4e50c08b1c30c10aa9f06829f43",
    },
}
PROCESS_ENTRY_STATIC_AUTHORITIES = {
    "process_entry_amendment": {
        "commit": PROCESS_ENTRY_F5_COMMIT,
        "path": "docs/pass201_pa_source_v3_process_entry_amendment_2026-08-11.md",
        "sha256": "9d751d0a9cd215438d150cffc62fe61baca62eb57addb62a4414412378144003",
    },
    "process_entry_plan": {
        "commit": PROCESS_ENTRY_F5_COMMIT,
        "path": "docs/superpowers/plans/2026-08-11-pass201-source-v3-process-entry-repair.md",
        "sha256": "621ae3939f54a3f7d3944d2c6a63ff533fd2205f4121169dacec910bc52e7edd",
    },
    "process_entry_evidence": {
        "commit": PROCESS_ENTRY_F4_COMMIT,
        "path": "docs/pass201_pa_source_v3_process_entry_evidence_2026-08-11.json",
        "sha256": "dd05361cb3630bf37b0bfbde79df7019afa6db9dee9f78b7343cd385acd77549",
    },
}
PRELAUNCH_SOURCE_MANIFEST_SHA256 = (
    "37644551f99976a7982589c1574effa00a9c77aa4a690117b5a8cd84244cc803"
)
FROZEN_DRAFT_PATH = "docs/pass201_cis_operator_diagnostic_draft_2026-08-09.md"
FROZEN_DRAFT_SHA256 = "310f194ee28727caa5908e877338afed82c7ac8be5f2f446affb08f402ef8066"
RESULT_PATH = "reports/generated/pass201_cis_operator/pass201_inshop_seed0.json"
SMOKE_RESULT_PATH = "reports/generated/pass201_cis_operator/pass201_inshop_seed0_smoke.json"
SOURCE_V5_REPAIR_PLAN_COMMIT = "20785ba6be243a9b7e95fcb25647ee3cdc55cc9e"
SOURCE_V4_HISTORICAL_SOURCE_COMMIT = "53a9db9e9dbe54fcebb33769b915c3f33699d522"
SOURCE_V4_HISTORICAL_HANDOFF_COMMIT = "32c4d39322fca2a5a906f785bdb612dcd7008647"
SOURCE_V4_HISTORICAL_TAG = "pass201-source-v4-handoff-32c4d39"
SOURCE_V5_DOCS_CHAIN = (
    (
        "d4a2df313a1f4fb708d9d30c5bce70abf232fa10",
        SOURCE_V4_HISTORICAL_SOURCE_COMMIT,
        "A",
        "docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md",
    ),
    (
        "04f423baae919318855297ebec9fc3d4cdf6b1ab",
        "d4a2df313a1f4fb708d9d30c5bce70abf232fa10",
        "A",
        "docs/superpowers/plans/2026-08-11-pass201-pa-source-v4-dispatch-repair.md",
    ),
    (
        "1f5bb121ed6062b00f904395715dcd89bb28fb6f",
        "04f423baae919318855297ebec9fc3d4cdf6b1ab",
        "M",
        "docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md",
    ),
    (
        "4afbda4886bf7e71eb72359904c838d8db87ff4c",
        "1f5bb121ed6062b00f904395715dcd89bb28fb6f",
        "M",
        "docs/superpowers/plans/2026-08-11-pass201-pa-source-v4-dispatch-repair.md",
    ),
    (
        "12003f535d1dcfa91274c895af830df690856a2c",
        "4afbda4886bf7e71eb72359904c838d8db87ff4c",
        "M",
        "docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md",
    ),
    (
        "c66a5af736d33f4743039031f81776bc3a6ada0a",
        "12003f535d1dcfa91274c895af830df690856a2c",
        "M",
        "docs/superpowers/plans/2026-08-11-pass201-pa-source-v4-dispatch-repair.md",
    ),
    (
        "d392a1c028ac7c8f918acc6c7c69c78bffbbacf4",
        "c66a5af736d33f4743039031f81776bc3a6ada0a",
        "M",
        "docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md",
    ),
    (
        "d3ad7701ed000dcd941e21d0a58577a70b626f3f",
        "d392a1c028ac7c8f918acc6c7c69c78bffbbacf4",
        "M",
        "docs/superpowers/plans/2026-08-11-pass201-pa-source-v4-dispatch-repair.md",
    ),
    (
        "b8a6e567db1352707afe5f38a364c342d006b9c8",
        "d3ad7701ed000dcd941e21d0a58577a70b626f3f",
        "M",
        "docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md",
    ),
    (
        "886c979433b94ce00420679668569ef83e2969b0",
        "b8a6e567db1352707afe5f38a364c342d006b9c8",
        "M",
        "docs/superpowers/plans/2026-08-11-pass201-pa-source-v4-dispatch-repair.md",
    ),
    (
        "3436bc7fdf09465667635fe89b6229707370c5a2",
        "886c979433b94ce00420679668569ef83e2969b0",
        "M",
        "docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md",
    ),
    (
        "61c2bd3dd62ad4e81f91c39a77e584e67a5532a2",
        "3436bc7fdf09465667635fe89b6229707370c5a2",
        "M",
        "docs/superpowers/plans/2026-08-11-pass201-pa-source-v4-dispatch-repair.md",
    ),
    (
        "8d7938a4e66abad0a8351422fcfa6f741ea76b00",
        "61c2bd3dd62ad4e81f91c39a77e584e67a5532a2",
        "M",
        "docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md",
    ),
    (
        SOURCE_V5_REPAIR_PLAN_COMMIT,
        "8d7938a4e66abad0a8351422fcfa6f741ea76b00",
        "M",
        "docs/superpowers/plans/2026-08-11-pass201-pa-source-v4-dispatch-repair.md",
    ),
)
ACTIVATED_PREREGISTRATION_PATH = "docs/pass201_cis_operator_activated_preregistration.json"
SOURCE_MANIFEST_PATH = "docs/pass201_cis_operator_source_manifest.json"

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


class PanelGradient(NamedTuple):
    parameter_names: tuple[str, ...]
    named_gradients: tuple[tuple[str, Any], ...]
    named_update_gradients: tuple[tuple[str, Any], ...]
    parameter_count: int
    gradient_sha256: str
    raw_gradient_norm: float
    update_space_norm: float
    auxiliary_to_pa_norm_ratio: float
    cosine_with_pa: float
    cosine_with_atomic_full_union: float
    cosine_with_summed_dropout: float
    scale_residual_to_summed_union: float | None


class StatelessUpdate(NamedTuple):
    parameter_names: tuple[str, ...]
    named_updates: tuple[tuple[str, Any], ...]
    update_sha256: str
    parameter_update_norm: float
    reference_pa_norm: float | None
    norm_match_absolute_error: float | None


class OutcomeFields(NamedTuple):
    R_F: float
    Delta_M: float
    D_F: float
    D_M: float


class TrainGraph(NamedTuple):
    embeddings: Any
    disposable_buffers_before: tuple[tuple[str, Any], ...]
    disposable_buffers_after: tuple[tuple[str, Any], ...]
    changed_buffer_names: tuple[str, ...]


class _OutcomeBaseline(NamedTuple):
    parameters: dict[str, Any]
    buffers: dict[str, Any]
    before_f: float
    before_m: float
    foreign_gradients: dict[str, Any]
    margin_gradients: dict[str, Any]


class SourceV3GitHandoff(NamedTuple):
    handoff_commit: str
    source_commit: str
    manifest_bytes: bytes
    manifest_sha256: str
    manifest_git_blob: str


class SourceV3Authority(NamedTuple):
    handoff: SourceV3GitHandoff
    contract: Any
    authority: Any
    receipt: Any


def _git_command_bytes(root: Path, *arguments: str, check: bool = True) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        capture_output=True,
    )
    if check and completed.returncode != 0:
        raise ValueError(f"git {' '.join(arguments)} failed with exit {completed.returncode}")
    return completed.stdout


def _git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(
        b"blob " + str(len(data)).encode("ascii") + b"\0" + data,
        usedforsecurity=False,
    ).hexdigest()


def _authenticate_source_v3_git_handoff(
    *, root: Path, git_root: Path, manifest_path: Path
) -> SourceV3GitHandoff:
    checkout = root.resolve(strict=True)
    _require(git_root.resolve(strict=True) == checkout, "git root differs from checkout")
    relative_manifest = manifest_path.relative_to(checkout).as_posix()
    _require(
        relative_manifest
        in (
            SOURCE_V3_AUTHORIZATION_MANIFEST_PATH,
            SOURCE_V4_AUTHORIZATION_MANIFEST_PATH,
            SOURCE_V5_AUTHORIZATION_MANIFEST_PATH,
        ),
        "source authorization manifest path differs",
    )
    expected_manifest = checkout / relative_manifest
    _require(
        manifest_path == expected_manifest,
        "source-v3 authorization manifest path differs",
    )
    _require(
        _git_command_bytes(checkout, "symbolic-ref", "-q", "HEAD", check=False) == b"",
        "source-v3 handoff checkout must be detached",
    )
    _require(
        _git_command_bytes(checkout, "status", "--porcelain=v1", "-z") == b"",
        "source-v3 handoff checkout must be clean",
    )
    parent_fields = (
        _git_command_bytes(checkout, "rev-list", "--parents", "-n", "1", "HEAD")
        .decode("ascii")
        .strip()
        .split()
    )
    _require(len(parent_fields) == 2, "source-v3 handoff must have one parent")
    handoff_commit, source_commit = parent_fields
    diff = _git_command_bytes(
        checkout,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        handoff_commit,
    ).split(b"\0")
    _require(
        diff == [b"A", relative_manifest.encode(), b""],
        "source handoff must add only the authorization manifest",
    )
    tree_line = _git_command_bytes(
        checkout,
        "ls-tree",
        handoff_commit,
        "--",
        relative_manifest,
    ).decode("utf-8")
    _require(
        tree_line.startswith("100644 blob ") and tree_line.endswith(f"\t{relative_manifest}\n"),
        "source handoff manifest mode differs",
    )
    committed = _git_command_bytes(
        checkout,
        "show",
        f"{handoff_commit}:{relative_manifest}",
    )
    _require(manifest_path.is_file(), "source-v3 authorization manifest unavailable")
    worktree = manifest_path.read_bytes()
    _require(committed == worktree, "source-v3 manifest Git/worktree bytes differ")
    return SourceV3GitHandoff(
        handoff_commit=handoff_commit,
        source_commit=source_commit,
        manifest_bytes=committed,
        manifest_sha256=hashlib.sha256(committed).hexdigest(),
        manifest_git_blob=_git_blob_sha1(committed),
    )


def _authenticate_source_v3_static_authorities(root: Path, source_commit: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, reference in SOURCE_V3_STATIC_AUTHORITIES.items():
        commit = reference["commit"]
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, source_commit],
            cwd=root,
            check=False,
            capture_output=True,
        )
        _require(ancestry.returncode == 0, f"{name} is not an ancestor of source-v3")
        data = _git_command_bytes(root, "show", f"{commit}:{reference['path']}")
        _require(
            hashlib.sha256(data).hexdigest() == reference["sha256"],
            f"{name} Git bytes differ",
        )
        path = root / reference["path"]
        _require(path.is_file() and path.read_bytes() == data, f"{name} worktree bytes differ")
        result[name] = commit
    return result


def _authenticate_process_entry_static_authorities(
    root: Path, source_commit: str
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, reference in PROCESS_ENTRY_STATIC_AUTHORITIES.items():
        commit = reference["commit"]
        ancestry = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, source_commit],
            cwd=root,
            check=False,
            capture_output=True,
        )
        _require(ancestry.returncode == 0, f"{name} is not an ancestor of source-v4")
        data = _git_command_bytes(root, "show", f"{commit}:{reference['path']}")
        _require(
            hashlib.sha256(data).hexdigest() == reference["sha256"],
            f"{name} Git bytes differ",
        )
        path = root / reference["path"]
        _require(path.is_file() and path.read_bytes() == data, f"{name} worktree bytes differ")
        result[name] = commit
    return result


def _authenticate_source_v3_source_chain(root: Path, source_commit: str) -> None:
    parent_fields = (
        _git_command_bytes(root, "rev-list", "--parents", "-n", "1", source_commit)
        .decode("ascii")
        .strip()
        .split()
    )
    _require(
        parent_fields == [source_commit, SOURCE_V3_I3_COMMIT],
        "source-v3 source must be the sole child of reviewed I3",
    )
    changed = _git_command_bytes(
        root,
        "diff-tree",
        "--no-commit-id",
        "--name-status",
        "-r",
        "-z",
        source_commit,
    ).split(b"\0")
    expected: list[bytes] = []
    for path in SOURCE_V3_CHANGED_PATHS:
        expected.extend((b"M", path.encode("utf-8")))
    expected.append(b"")
    _require(changed == expected, "source-v3 source edge differs from exact six paths")


def _authenticate_source_v4_source_chain(root: Path, source_commit: str) -> None:
    expected_edges = {
        SOURCE_V3_H3_COMMIT: SOURCE_V3_V3_COMMIT,
        SOURCE_V3_V3_COMMIT: SOURCE_V3_I3_COMMIT,
        SOURCE_V3_I3_COMMIT: SOURCE_V3_I3A_COMMIT,
        PROCESS_ENTRY_DRAFT_AMENDMENT_COMMIT: SOURCE_V3_H3_COMMIT,
        PROCESS_ENTRY_DRAFT_PLAN_COMMIT: PROCESS_ENTRY_DRAFT_AMENDMENT_COMMIT,
        PROCESS_ENTRY_F4_COMMIT: PROCESS_ENTRY_DRAFT_PLAN_COMMIT,
        PROCESS_ENTRY_F5_COMMIT: PROCESS_ENTRY_F4_COMMIT,
    }
    for child, parent in expected_edges.items():
        fields = (
            _git_command_bytes(root, "rev-list", "--parents", "-n", "1", child)
            .decode("ascii")
            .strip()
            .split()
        )
        _require(fields == [child, parent], f"source-v4 historical edge differs: {child}")
    current = source_commit
    aggregate: set[str] = set()
    while current != PROCESS_ENTRY_F5_COMMIT:
        fields = (
            _git_command_bytes(root, "rev-list", "--parents", "-n", "1", current)
            .decode("ascii")
            .strip()
            .split()
        )
        _require(len(fields) == 2, "source-v4 source chain must be merge-free")
        raw_changed = _git_command_bytes(
            root, "diff-tree", "--no-commit-id", "--name-status", "-r", "-z", current
        ).split(b"\0")
        _require(
            raw_changed[-1:] == [b""] and len(raw_changed) > 1,
            "source-v4 source commit must not be empty",
        )
        entries = raw_changed[:-1]
        _require(
            len(entries) % 2 == 0,
            "source-v4 source edge status differs from exact modifications",
        )
        changed: list[str] = []
        for index in range(0, len(entries), 2):
            _require(
                entries[index] == b"M",
                "source-v4 source edge status differs from exact modifications",
            )
            try:
                changed.append(entries[index + 1].decode("utf-8"))
            except UnicodeDecodeError as exc:
                raise ValueError("source-v4 source path is not UTF-8") from exc
        _require(
            set(changed) <= set(SOURCE_V3_CHANGED_PATHS),
            "source-v4 source commit changes an unauthorized path",
        )
        aggregate.update(changed)
        current = fields[1]
    _require(
        aggregate == set(SOURCE_V3_CHANGED_PATHS),
        "source-v4 aggregate source scope differs from exact six paths",
    )


def _authenticate_source_v5_docs_chain(root: Path) -> None:
    for commit, parent, status, path in SOURCE_V5_DOCS_CHAIN:
        fields = (
            _git_command_bytes(root, "rev-list", "--parents", "-n", "1", commit)
            .decode("ascii")
            .strip()
            .split()
        )
        _require(fields == [commit, parent], f"source-v5 docs parent differs: {commit}")
        changed = _git_command_bytes(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            commit,
        ).split(b"\0")
        _require(
            changed == [status.encode("ascii"), path.encode("utf-8"), b""],
            f"source-v5 docs scope differs: {commit}",
        )
        tree = _git_command_bytes(root, "ls-tree", commit, "--", path).decode("utf-8")
        _require(
            tree.startswith("100644 blob ") and tree.endswith(f"\t{path}\n"),
            f"source-v5 docs mode differs: {commit}",
        )


def _authenticate_source_v5_source_chain(root: Path, source_commit: str) -> None:
    _authenticate_source_v5_docs_chain(root)
    current = source_commit
    aggregate: set[str] = set()
    allowed = {
        "scripts/diagnose_pass201_cis_operator.py",
        "scripts/pass201_pa_source_v2_contract.py",
        "tests/test_diagnose_pass201_cis_operator.py",
    }
    while current != SOURCE_V5_REPAIR_PLAN_COMMIT:
        fields = (
            _git_command_bytes(root, "rev-list", "--parents", "-n", "1", current)
            .decode("ascii")
            .strip()
            .split()
        )
        _require(len(fields) == 2, "source-v5 source chain must be merge-free")
        entries = _git_command_bytes(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            current,
        ).split(b"\0")
        _require(entries[-1:] == [b""] and len(entries) > 1, "source-v5 source commit empty")
        entries = entries[:-1]
        _require(len(entries) % 2 == 0, "source-v5 source edge malformed")
        changed: set[str] = set()
        for index in range(0, len(entries), 2):
            _require(entries[index] == b"M", "source-v5 source edge status differs")
            try:
                changed.add(entries[index + 1].decode("utf-8"))
            except UnicodeDecodeError as error:
                raise ValueError("source-v5 source path is not UTF-8") from error
        _require(changed <= allowed, "source-v5 source commit changes unauthorized path")
        aggregate.update(changed)
        current = fields[1]
    _require(aggregate == allowed, "source-v5 aggregate source scope differs")


def _validate_source_v3_receipt_relations(
    authority: Any,
    receipt: Mapping[str, Any],
    handoff: SourceV3GitHandoff,
) -> None:
    payload = authority.payload
    is_v4 = payload["schema_version"] == "pass201-pa-source-v4-prelaunch-v1"
    _require(
        is_v4 or payload["schema_version"] == "pass201-pa-source-v3-prelaunch-v1",
        "source authority schema differs",
    )
    manifest_path = (
        SOURCE_V4_AUTHORIZATION_MANIFEST_PATH if is_v4 else SOURCE_V3_AUTHORIZATION_MANIFEST_PATH
    )
    _require(
        payload["source_commit"]
        == payload["authorization"]["required_parent_commit"]
        == handoff.source_commit,
        "source-v3 source commit differs",
    )
    _require(
        payload["authorization"]["manifest_path"] == manifest_path
        and payload["authorization"]["required_diff_paths"] == (manifest_path,),
        "source-v3 manifest authority differs",
    )
    _require(
        payload["protocol"] == SOURCE_V3_STATIC_AUTHORITIES["protocol"]
        and payload["plan"] == SOURCE_V3_STATIC_AUTHORITIES["plan"],
        "source-v3 static authority differs",
    )
    _require(
        receipt["schema_version"]
        == ("pass201-pa-source-v4-receipt-v1" if is_v4 else "pass201-pa-source-v3-receipt-v1")
        and receipt["candidate_values_computed"] is False,
        "source-v3 receipt schema or scope differs",
    )
    authorization = receipt["authorization"]
    _require(
        authorization["authorization_commit"] == handoff.handoff_commit
        and authorization["source_commit"] == handoff.source_commit
        and authorization["manifest_path"] == manifest_path
        and authorization["manifest_sha256"] == handoff.manifest_sha256
        and authorization["manifest_git_blob"] == handoff.manifest_git_blob,
        "source-v3 receipt handoff binding differs",
    )
    _require(
        authorization["protocol"] == payload["protocol"]
        and authorization["plan"] == payload["plan"],
        "source-v3 receipt static authority differs",
    )
    if is_v4:
        for key in PROCESS_ENTRY_STATIC_AUTHORITIES:
            _require(
                payload[key] == PROCESS_ENTRY_STATIC_AUTHORITIES[key]
                and authorization[key] == payload[key],
                f"source-v4 receipt {key} differs",
            )


def _authenticate_source_v3_repo_row(
    root: Path, source_commit: str, row: Mapping[str, Any]
) -> bytes:
    _require(
        isinstance(row, Mapping)
        and set(row) == {"path", "git_mode", "bytes", "sha256", "git_blob"},
        "source row keys differ",
    )
    relative = row["path"]
    _require(isinstance(relative, str) and relative, "source row path")
    relative_path = Path(relative)
    _require(
        not relative_path.is_absolute()
        and relative_path.as_posix() == relative
        and ".." not in relative_path.parts,
        "source row path is not normalized",
    )
    _require(row["git_mode"] == "100644", "source row Git mode differs")
    _require(type(row["bytes"]) is int and row["bytes"] > 0, "source row byte count")
    _digest(row["sha256"], "source row sha256")
    _require(
        isinstance(row["git_blob"], str)
        and len(row["git_blob"]) == 40
        and all(character in "0123456789abcdef" for character in row["git_blob"]),
        "source row Git blob",
    )
    data = _git_command_bytes(root, "show", f"{source_commit}:{relative}")
    blob = (
        _git_command_bytes(root, "rev-parse", f"{source_commit}:{relative}").decode("ascii").strip()
    )
    tree_line = _git_command_bytes(root, "ls-tree", source_commit, "--", relative).decode("utf-8")
    _require(
        tree_line.startswith("100644 blob ") and tree_line.endswith(f"\t{relative}\n"),
        "source row committed mode differs",
    )
    _require(
        len(data) == row["bytes"]
        and hashlib.sha256(data).hexdigest() == row["sha256"]
        and blob == row["git_blob"],
        "source row Git identity differs",
    )
    path = root / relative_path
    _require(
        path.resolve(strict=True) == path.absolute()
        and stat.S_ISREG(os.lstat(path).st_mode)
        and path.read_bytes() == data,
        "source row worktree bytes differ",
    )
    return data


def _load_authenticated_source_v3_contract(
    root: Path, source_commit: str, row: Mapping[str, Any]
) -> Any:
    _require(
        row.get("path") == "scripts/pass201_pa_source_v2_contract.py",
        "source-v3 contract path differs",
    )
    authenticated = _authenticate_source_v3_repo_row(root, source_commit, row)
    path = root / "scripts/pass201_pa_source_v2_contract.py"
    module_name = f"_pass201_source_v3_contract_{hashlib.sha256(authenticated).hexdigest()}"
    _require(module_name not in sys.modules, "source-v3 private contract already loaded")
    specification = importlib.util.spec_from_file_location(module_name, path)
    _require(
        specification is not None and specification.loader is not None,
        "unable to load source-v3 contract",
    )
    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    try:
        specification.loader.exec_module(module)
    finally:
        sys.modules.pop(module_name, None)
    _require(
        _authenticate_source_v3_repo_row(root, source_commit, row) == authenticated,
        "source-v3 contract changed during import",
    )
    _require(
        Path(module.__file__).resolve(strict=True) == path.resolve(strict=True),
        "source-v3 contract module path differs",
    )
    return module


def _bootstrap_strict_json_object(data: bytes, where: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{where}: nonfinite JSON constant {value}")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{where}: duplicate JSON key {key}")
            result[key] = value
        return result

    try:
        value = json.loads(
            data.decode("utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=unique_object,
        )
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{where}: invalid JSON") from error
    _require(type(value) is dict, f"{where}: expected object")
    return value


def _load_source_v3_authority(
    *, root: Path, git_root: Path, manifest_path: Path, receipt_path: Path
) -> SourceV3Authority:
    handoff = _authenticate_source_v3_git_handoff(
        root=root, git_root=git_root, manifest_path=manifest_path
    )
    is_v4 = manifest_path.name == Path(SOURCE_V4_AUTHORIZATION_MANIFEST_PATH).name
    if is_v4:
        _authenticate_source_v4_source_chain(root, handoff.source_commit)
    else:
        _authenticate_source_v3_source_chain(root, handoff.source_commit)
    bootstrap = _bootstrap_strict_json_object(
        handoff.manifest_bytes, "source-v3 authorization manifest"
    )
    _require(
        bootstrap.get("schema_version")
        == ("pass201-pa-source-v4-prelaunch-v1" if is_v4 else "pass201-pa-source-v3-prelaunch-v1")
        and bootstrap.get("source_commit") == handoff.source_commit,
        "source-v3 bootstrap authority differs",
    )
    source = bootstrap.get("source")
    _require(type(source) is dict, "source-v3 source mapping")
    files = source.get("files")
    _require(type(files) is list and len(files) >= 2, "source-v3 source files")
    _require(
        files[0].get("path") == "scripts/diagnose_pass201_cis_operator.py"
        and files[1].get("path") == "scripts/pass201_pa_source_v2_contract.py",
        "source-v3 bootstrap source prefix differs",
    )
    contract = _load_authenticated_source_v3_contract(root, handoff.source_commit, files[1])
    manifest = contract.load_strict_json_bytes(handoff.manifest_bytes)
    _require(
        contract.canonical_json_bytes(manifest) == handoff.manifest_bytes,
        "source-v3 manifest bytes are not canonical",
    )
    authority = contract.validate_prelaunch(manifest)
    _authenticate_source_v3_static_authorities(root, handoff.source_commit)
    if is_v4:
        _authenticate_process_entry_static_authorities(root, handoff.source_commit)
    payload = authority.payload
    _require(
        tuple(row["path"] for row in payload["source"]["files"]) == contract.SOURCE_V3_PATHS,
        "source-v3 source path order differs",
    )
    _require(
        payload["controller"]["path"] == "scripts/run_pass201_pa_source_v2.py",
        "source-v3 controller path differs",
    )
    for row in (
        payload["controller"],
        *payload["source"]["files"],
        payload["source"]["pyproject"],
        payload["source"]["lockfile"],
    ):
        _authenticate_source_v3_repo_row(root, handoff.source_commit, row)
    diagnostic_path = root / "scripts/diagnose_pass201_cis_operator.py"
    _require(
        Path(__file__).resolve(strict=True) == diagnostic_path.resolve(strict=True),
        "executing diagnostic is not the source-v3 worktree path",
    )
    expected_receipt = root / payload["outputs"]["receipt"]["path"]
    _require(receipt_path == expected_receipt, "source-v3 receipt path differs")
    receipt_bytes = receipt_path.read_bytes()
    receipt = contract.load_strict_json_bytes(receipt_bytes)
    _require(
        contract.canonical_json_bytes(receipt) == receipt_bytes,
        "source-v3 receipt bytes are not canonical",
    )
    contract.validate_complete_receipt(receipt, authority)
    _validate_source_v3_receipt_relations(authority, receipt, handoff)
    return SourceV3Authority(handoff, contract, authority, receipt)


def _load_source_v5_authority(
    *, root: Path, git_root: Path, manifest_path: Path, receipt_path: Path
) -> SourceV3Authority:
    handoff = _authenticate_source_v3_git_handoff(
        root=root, git_root=git_root, manifest_path=manifest_path
    )
    _require(
        manifest_path == root / SOURCE_V5_AUTHORIZATION_MANIFEST_PATH,
        "source-v5 authorization manifest path differs",
    )
    _authenticate_source_v5_source_chain(root, handoff.source_commit)
    bootstrap = _bootstrap_strict_json_object(
        handoff.manifest_bytes, "source-v5 authorization manifest"
    )
    _require(
        bootstrap.get("schema_version") == "pass201-pa-source-v5-activation-v1"
        and bootstrap.get("source_commit") == handoff.source_commit,
        "source-v5 bootstrap authority differs",
    )
    source = bootstrap.get("source")
    _require(type(source) is dict, "source-v5 source mapping")
    files = source.get("files")
    _require(type(files) is list and len(files) == 30, "source-v5 source files")
    _require(
        [row.get("path") for row in files[:2]]
        == [
            "scripts/diagnose_pass201_cis_operator.py",
            "scripts/pass201_pa_source_v2_contract.py",
        ],
        "source-v5 bootstrap source prefix differs",
    )
    contract = _load_authenticated_source_v3_contract(root, handoff.source_commit, files[1])
    manifest = contract.load_strict_json_bytes(handoff.manifest_bytes)
    _require(
        contract.canonical_ordered_json_bytes(manifest) == handoff.manifest_bytes,
        "source-v5 manifest bytes are not canonical",
    )
    authority = contract.validate_prelaunch(manifest)
    _require(
        tuple(row["path"] for row in authority.payload["source"]["files"])
        == contract.SOURCE_V3_PATHS,
        "source-v5 source path order differs",
    )
    for row in (
        authority.payload["controller"],
        *authority.payload["source"]["files"],
        authority.payload["source"]["pyproject"],
        authority.payload["source"]["lockfile"],
    ):
        _authenticate_source_v3_repo_row(root, handoff.source_commit, row)
    _require(
        Path(__file__).resolve(strict=True)
        == (root / "scripts/diagnose_pass201_cis_operator.py").resolve(strict=True),
        "executing diagnostic is not the source-v5 worktree path",
    )
    _authenticate_source_v3_static_authorities(root, handoff.source_commit)
    _authenticate_process_entry_static_authorities(root, handoff.source_commit)

    _require(
        _git_command_bytes(root, "rev-parse", SOURCE_V4_HISTORICAL_TAG).decode("ascii").strip()
        == SOURCE_V4_HISTORICAL_HANDOFF_COMMIT,
        "source-v4 preservation tag differs",
    )
    historical_fields = (
        _git_command_bytes(
            root,
            "rev-list",
            "--parents",
            "-n",
            "1",
            SOURCE_V4_HISTORICAL_HANDOFF_COMMIT,
        )
        .decode("ascii")
        .strip()
        .split()
    )
    _require(
        historical_fields
        == [SOURCE_V4_HISTORICAL_HANDOFF_COMMIT, SOURCE_V4_HISTORICAL_SOURCE_COMMIT],
        "source-v4 historical parent differs",
    )
    historical_bytes = _git_command_bytes(
        root,
        "show",
        f"{SOURCE_V4_HISTORICAL_HANDOFF_COMMIT}:{SOURCE_V4_AUTHORIZATION_MANIFEST_PATH}",
    )
    historical = contract.load_strict_json_bytes(historical_bytes)
    historical_manifest = authority.payload["historical_producer"]["manifest"]
    _require(
        len(historical_bytes) == historical_manifest["bytes"]
        and hashlib.sha256(historical_bytes).hexdigest() == historical_manifest["sha256"]
        and _git_blob_sha1(historical_bytes) == historical_manifest["git_blob"],
        "source-v4 historical manifest identity differs",
    )
    _require(
        contract.canonical_json_bytes(historical) == historical_bytes,
        "source-v4 historical manifest bytes are not canonical",
    )
    historical_authority = contract.validate_prelaunch(historical)
    for key in (
        "controller",
        "dataset",
        "execution",
        "plan",
        "postconditions",
        "process_entry_amendment",
        "process_entry_evidence",
        "process_entry_plan",
        "protocol",
        "sidecars",
        "status",
    ):
        _require(
            contract.canonical_ordered_json_bytes(manifest[key])
            == contract.canonical_ordered_json_bytes(historical[key]),
            f"source-v5 retained historical domain differs: {key}",
        )
    expected_receipt = root / authority.payload["outputs"]["receipt"]["path"]
    _require(receipt_path == expected_receipt, "source-v5 receipt path differs")
    receipt_bytes = receipt_path.read_bytes()
    historical_receipt = authority.payload["historical_producer"]["receipt"]
    _require(
        len(receipt_bytes) == historical_receipt["bytes"]
        and hashlib.sha256(receipt_bytes).hexdigest() == historical_receipt["sha256"],
        "source-v5 historical receipt identity differs",
    )
    receipt = contract.load_strict_json_bytes(receipt_bytes)
    _require(
        contract.canonical_json_bytes(receipt) == receipt_bytes,
        "source-v4 receipt bytes are not canonical",
    )
    contract.validate_complete_receipt(receipt, historical_authority)
    historical_handoff = SourceV3GitHandoff(
        handoff_commit=SOURCE_V4_HISTORICAL_HANDOFF_COMMIT,
        source_commit=SOURCE_V4_HISTORICAL_SOURCE_COMMIT,
        manifest_bytes=historical_bytes,
        manifest_sha256=hashlib.sha256(historical_bytes).hexdigest(),
        manifest_git_blob=_git_blob_sha1(historical_bytes),
    )
    _validate_source_v3_receipt_relations(historical_authority, receipt, historical_handoff)
    return SourceV3Authority(handoff, contract, authority, receipt)


def _representative_tensor_indices(labels: Any, sample_indices: Any, torch: Any) -> Any:
    unique_labels = torch.unique(labels, sorted=True)
    if int(unique_labels.numel()) == 0:
        raise ValueError("at least one represented class is required")
    positions = []
    for label in unique_labels:
        candidates = torch.nonzero(labels.eq(label), as_tuple=False)[:, 0]
        local_position = torch.argmin(sample_indices[candidates].to(torch.long))
        positions.append(candidates[local_position])
    return unique_labels, torch.stack(positions)


def _named_tensor_norm(named_tensors: Iterable[tuple[str, Any]]) -> float:
    import torch

    total = torch.zeros((), dtype=torch.float64)
    for _, value in named_tensors:
        tensor = value.detach().to(device="cpu", dtype=torch.float64)
        total = total + torch.sum(tensor * tensor)
    return float(torch.sqrt(total).item())


def _named_tensor_cosine(
    left: tuple[tuple[str, Any], ...], right: tuple[tuple[str, Any], ...]
) -> float:
    import torch

    if tuple(name for name, _ in left) != tuple(name for name, _ in right):
        raise ValueError("cosine parameter membership mismatch")
    dot = torch.zeros((), dtype=torch.float64)
    for (_, left_value), (_, right_value) in zip(left, right, strict=True):
        dot = dot + torch.sum(
            left_value.detach().to(device="cpu", dtype=torch.float64)
            * right_value.detach().to(device="cpu", dtype=torch.float64)
        )
    denominator = _named_tensor_norm(left) * _named_tensor_norm(right)
    if denominator == 0.0:
        raise ValueError("cosine requires nonzero vectors")
    return float(dot.item() / denominator)


def coalition_losses(
    embeddings: Any,
    labels: Any,
    sample_indices: Any,
    proxies: Any,
    proxy_labels: Any,
    *,
    alpha: float,
    delta: float,
) -> dict[str, Any]:
    """Build the six frozen operator losses from one shared embedding graph."""

    import torch

    from sfora.image_end_to_end import _normalize, _proxy_anchor_loss

    if embeddings.ndim != 2 or proxies.ndim != 2 or embeddings.shape[1] != proxies.shape[1]:
        raise ValueError("embeddings and proxies must be aligned matrices")
    if labels.ndim != 1 or sample_indices.ndim != 1:
        raise ValueError("labels and sample_indices must be vectors")
    if labels.shape[0] != embeddings.shape[0] or sample_indices.shape[0] != labels.shape[0]:
        raise ValueError("row metadata must align with embeddings")
    if proxy_labels.ndim != 1 or proxy_labels.shape[0] != proxies.shape[0]:
        raise ValueError("proxy_labels must align with proxies")

    unique_labels, representative_indices = _representative_tensor_indices(
        labels, sample_indices, torch
    )

    members = _normalize(embeddings[representative_indices], torch)
    normalized_proxies = _normalize(proxies, torch)
    atomic_logits = members @ normalized_proxies.T
    bundle_target = proxy_labels.unsqueeze(0).eq(unique_labels.unsqueeze(1)).any(dim=0)
    one_hot_target = unique_labels.unsqueeze(1).eq(proxy_labels.unsqueeze(0))
    complementary_target = bundle_target.unsqueeze(0) & ~one_hot_target
    full_union_target = bundle_target.unsqueeze(0).expand_as(atomic_logits)
    bce = torch.nn.functional.binary_cross_entropy_with_logits

    coalition = members.sum(dim=0) / math.sqrt(int(unique_labels.numel()))
    summed_logits = (coalition @ normalized_proxies.T).unsqueeze(0)
    dropout_labels = unique_labels[:-1]
    dropout_target = proxy_labels.unsqueeze(0).eq(dropout_labels.unsqueeze(1)).any(dim=0)

    losses = {
        "proxy_anchor": _proxy_anchor_loss(
            embeddings,
            labels,
            proxy_embeddings=proxies,
            proxy_labels=proxy_labels,
            alpha=alpha,
            delta=delta,
            torch_module=torch,
        ),
        "atomic_one_hot": bce(
            atomic_logits, one_hot_target.to(atomic_logits.dtype), reduction="mean"
        ),
        "atomic_complementary": bce(
            atomic_logits, complementary_target.to(atomic_logits.dtype), reduction="mean"
        ),
        "atomic_full_union": bce(
            atomic_logits, full_union_target.to(atomic_logits.dtype), reduction="mean"
        ),
        "summed_union": bce(
            summed_logits, bundle_target.to(summed_logits.dtype).unsqueeze(0), reduction="mean"
        ),
        "summed_dropout": bce(
            summed_logits, dropout_target.to(summed_logits.dtype).unsqueeze(0), reduction="mean"
        ),
    }
    return {name: losses[name] for name in OPERATORS}


def operator_gradients(
    losses: Mapping[str, Any],
    named_parameters: Mapping[str, Any],
    *,
    expected_trainable_parameter_names: Sequence[str],
    proxy_parameter_name: str,
    proxy_learning_rate_multiplier: float,
    representative_count: int | None = None,
) -> dict[str, PanelGradient]:
    """Differentiate each operator once and derive the two frozen panels."""

    import torch

    expected_names = tuple(expected_trainable_parameter_names)
    if len(expected_names) != len(set(expected_names)):
        raise ValueError("duplicate expected trainable parameter name")
    lexical_names = tuple(sorted(expected_names, key=lambda name: name.encode("utf-8")))
    if expected_names != lexical_names:
        raise ValueError("expected trainable parameter names must be lexicographic")
    actual = {name: value for name, value in named_parameters.items() if value.requires_grad}
    actual_names = set(actual)
    expected_set = set(expected_names)
    missing = expected_set - actual_names
    unexpected = actual_names - expected_set
    if missing:
        raise ValueError(f"missing trainable parameter: {min(missing)}")
    if unexpected:
        raise ValueError(f"unexpected trainable parameter: {min(unexpected)}")
    if proxy_parameter_name not in expected_set:
        raise ValueError("missing trainable proxy parameter")
    if tuple(losses) != OPERATORS:
        raise ValueError("losses must contain the exact ordered operator panel")
    if not math.isfinite(proxy_learning_rate_multiplier):
        raise ValueError("proxy learning-rate multiplier must be finite")
    if representative_count is not None and representative_count <= 0:
        raise ValueError("representative_count must be positive")

    parameters = tuple(actual[name] for name in expected_names)
    output: dict[str, PanelGradient] = {}
    for operator in OPERATORS:
        values = torch.autograd.grad(
            losses[operator], parameters, retain_graph=True, allow_unused=True
        )
        disconnected = [
            name for name, value in zip(expected_names, values, strict=True) if value is None
        ]
        if disconnected:
            raise ValueError(f"disconnected required parameter: {disconnected[0]}")
        gradients_by_name = {
            name: value.detach().clone() for name, value in zip(expected_names, values, strict=True)
        }
        for panel in PANELS:
            panel_names = tuple(
                name
                for name in expected_names
                if panel == "joint_including_proxies" or name != proxy_parameter_name
            )
            if not panel_names:
                raise ValueError(f"{panel} has no parameters")
            named_gradients = tuple((name, gradients_by_name[name]) for name in panel_names)
            named_update_gradients = tuple(
                (
                    name,
                    gradients_by_name[name]
                    * (
                        float(proxy_learning_rate_multiplier)
                        if name == proxy_parameter_name
                        else 1.0
                    ),
                )
                for name in panel_names
            )
            raw_norm = _named_tensor_norm(named_gradients)
            update_norm = _named_tensor_norm(named_update_gradients)
            if not math.isfinite(raw_norm) or raw_norm == 0.0:
                raise ValueError(f"zero or nonfinite required gradient: {operator}.{panel}")
            if not math.isfinite(update_norm) or update_norm == 0.0:
                raise ValueError(f"zero or nonfinite update-space gradient: {operator}.{panel}")
            output[f"{operator}.{panel}"] = PanelGradient(
                parameter_names=panel_names,
                named_gradients=named_gradients,
                named_update_gradients=named_update_gradients,
                parameter_count=sum(int(grad.numel()) for _, grad in named_gradients),
                gradient_sha256=sha256_named_tensors(named_gradients),
                raw_gradient_norm=raw_norm,
                update_space_norm=update_norm,
                auxiliary_to_pa_norm_ratio=0.0,
                cosine_with_pa=0.0,
                cosine_with_atomic_full_union=0.0,
                cosine_with_summed_dropout=0.0,
                scale_residual_to_summed_union=None,
            )
    for panel in PANELS:
        pa = output[f"proxy_anchor.{panel}"]
        full_union = output[f"atomic_full_union.{panel}"]
        dropout = output[f"summed_dropout.{panel}"]
        summed_union = output[f"summed_union.{panel}"]
        for operator in OPERATORS:
            key = f"{operator}.{panel}"
            record = output[key]
            scale_residual = None
            if representative_count is not None and operator.startswith("atomic_"):
                scale_residual = summed_union.update_space_norm / (
                    math.sqrt(representative_count) * record.update_space_norm
                )
            output[key] = record._replace(
                auxiliary_to_pa_norm_ratio=record.update_space_norm / pa.update_space_norm,
                cosine_with_pa=_named_tensor_cosine(
                    record.named_update_gradients, pa.named_update_gradients
                ),
                cosine_with_atomic_full_union=_named_tensor_cosine(
                    record.named_update_gradients, full_union.named_update_gradients
                ),
                cosine_with_summed_dropout=_named_tensor_cosine(
                    record.named_update_gradients, dropout.named_update_gradients
                ),
                scale_residual_to_summed_union=scale_residual,
            )
    return output


def _scaled_named_tensors(
    named_tensors: tuple[tuple[str, Any], ...], scale: float
) -> tuple[tuple[str, Any], ...]:
    return tuple((name, value * scale) for name, value in named_tensors)


def _combine_named_tensors(
    left: tuple[tuple[str, Any], ...],
    right: tuple[tuple[str, Any], ...],
    right_scale: float,
) -> tuple[tuple[str, Any], ...]:
    if tuple(name for name, _ in left) != tuple(name for name, _ in right):
        raise ValueError("gradient panel membership mismatch")
    return tuple(
        (left_name, left_value + right_scale * right_value)
        for (left_name, left_value), (_, right_value) in zip(left, right, strict=True)
    )


def make_stateless_updates(
    gradients: Mapping[str, PanelGradient],
    *,
    learning_rate: float,
    coalition_weight: float,
) -> dict[str, Any]:
    """Construct configured and panel-specific equal-norm virtual updates."""

    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(coalition_weight):
        raise ValueError("coalition_weight must be finite")
    expected_keys = {f"{operator}.{panel}" for operator in OPERATORS for panel in PANELS}
    if set(gradients) != expected_keys:
        raise ValueError("gradient records must contain the exact operator panels")

    updates: dict[str, Any] = {
        regime: {operator: {} for operator in OPERATORS} for regime in REGIMES
    }
    for panel in PANELS:
        pa = gradients[f"proxy_anchor.{panel}"]
        reference_pa_norm = float(learning_rate) * pa.update_space_norm
        for operator in OPERATORS:
            pure = gradients[f"{operator}.{panel}"]
            configured_direction = (
                pa.named_update_gradients
                if operator == "proxy_anchor"
                else _combine_named_tensors(
                    pa.named_update_gradients,
                    pure.named_update_gradients,
                    float(coalition_weight),
                )
            )
            configured_update = _scaled_named_tensors(configured_direction, -float(learning_rate))
            updates["configured_loss_stateless"][operator][panel] = StatelessUpdate(
                parameter_names=pure.parameter_names,
                named_updates=configured_update,
                update_sha256=sha256_named_tensors(configured_update),
                parameter_update_norm=_named_tensor_norm(configured_update),
                reference_pa_norm=None,
                norm_match_absolute_error=None,
            )

            equal_scale = -reference_pa_norm / pure.update_space_norm
            equal_update = _scaled_named_tensors(pure.named_update_gradients, equal_scale)
            equal_norm = _named_tensor_norm(equal_update)
            error = abs(equal_norm - reference_pa_norm)
            if error > 1e-10 * max(reference_pa_norm, 1e-12):
                raise ValueError(f"equal-norm mismatch: {operator}.{panel}")
            updates["equal_norm"][operator][panel] = StatelessUpdate(
                parameter_names=pure.parameter_names,
                named_updates=equal_update,
                update_sha256=sha256_named_tensors(equal_update),
                parameter_update_norm=equal_norm,
                reference_pa_norm=reference_pa_norm,
                norm_match_absolute_error=error,
            )
    return updates


def _outcome_scalars(
    embeddings: Any,
    labels: Any,
    proxies: Any,
    proxy_labels: Any,
    *,
    temperature: float,
    torch: Any,
) -> tuple[Any, Any]:
    from sfora.image_end_to_end import _normalize

    normalized_embeddings = _normalize(embeddings, torch)
    normalized_proxies = _normalize(proxies, torch)
    logits = normalized_embeddings @ normalized_proxies.T
    bundle_labels = torch.unique(labels, sorted=True)
    foreign_mask = ~torch.isin(proxy_labels, bundle_labels)
    if not bool(foreign_mask.any()):
        raise ValueError("at least one foreign proxy row is required")
    foreign_mass = torch.sigmoid(logits[:, foreign_mask]).mean()
    margins = []
    for row, label in enumerate(labels):
        owner_mask = proxy_labels.eq(label)
        if not bool(owner_mask.any()):
            raise ValueError(f"represented class lacks owning proxy: {int(label)}")
        hard_mask = ~owner_mask
        if not bool(hard_mask.any()):
            raise ValueError("owner margin requires a non-owner proxy")
        owner_logits = logits[row, owner_mask] / float(temperature)
        hard_logits = logits[row, hard_mask] / float(temperature)
        owner = float(temperature) * (
            torch.logsumexp(owner_logits, dim=0) - math.log(int(owner_logits.numel()))
        )
        hard = float(temperature) * (
            torch.logsumexp(hard_logits, dim=0) - math.log(int(hard_logits.numel()))
        )
        margins.append(owner - hard)
    return foreign_mass, torch.stack(margins).mean()


def _module_training_flags(model: Any) -> tuple[tuple[Any, bool], ...]:
    return tuple((module, bool(module.training)) for module in model.modules())


def _restore_module_training_flags(flags: Iterable[tuple[Any, bool]]) -> None:
    for module, training in flags:
        module.training = training


def _tensor_byte_identity(value: Any) -> tuple[str, tuple[int, ...], bytes]:
    array = value.detach().cpu().contiguous().numpy()
    return str(value.dtype), tuple(value.shape), array.tobytes(order="C")


def bufferless_train_embeddings(model: Any, inputs: Any) -> TrainGraph:
    """Run one train-mode functional forward with cloned disposable buffers."""

    import torch

    parameters = dict(model.named_parameters())
    parameter_values = {name: value.detach().clone() for name, value in parameters.items()}
    parameter_identities = {
        name: _tensor_byte_identity(value) for name, value in parameters.items()
    }
    before = {name: value.detach().clone() for name, value in model.named_buffers()}
    buffer_identities = {name: _tensor_byte_identity(value) for name, value in before.items()}
    disposable = {name: value.clone() for name, value in before.items()}
    flags = _module_training_flags(model)
    try:
        model.train()
        embeddings = torch.func.functional_call(
            model, (parameters, disposable), (inputs,), strict=True
        )
        after = {name: value.detach().clone() for name, value in disposable.items()}
    finally:
        _restore_module_training_flags(flags)
    current_parameters = dict(model.named_parameters())
    current_buffers = dict(model.named_buffers())
    changed_parameters = tuple(
        name
        for name, identity in parameter_identities.items()
        if name not in current_parameters
        or _tensor_byte_identity(current_parameters[name]) != identity
    )
    changed_persistent_buffers = tuple(
        name
        for name, identity in buffer_identities.items()
        if name not in current_buffers or _tensor_byte_identity(current_buffers[name]) != identity
    )
    if changed_parameters or changed_persistent_buffers:
        with torch.no_grad():
            for name, original in parameter_values.items():
                current_parameters[name].copy_(original)
            for name, original in before.items():
                current_buffers[name].copy_(original)
        if changed_parameters:
            raise ValueError(
                f"checkpoint parameter is not byte-identical after train graph: "
                f"{changed_parameters[0]}"
            )
        raise ValueError(
            f"checkpoint buffer is not byte-identical after train graph: "
            f"{changed_persistent_buffers[0]}"
        )
    names = tuple(sorted(before, key=lambda name: name.encode("utf-8")))
    changed = tuple(name for name in names if not torch.equal(before[name], after[name]))
    return TrainGraph(
        embeddings=embeddings,
        disposable_buffers_before=tuple((name, before[name]) for name in names),
        disposable_buffers_after=tuple((name, after[name]) for name in names),
        changed_buffer_names=changed,
    )


def _prepare_outcome_baseline(
    *,
    model: Any,
    clean_inputs: Any,
    labels: Any,
    proxy_labels: Any,
    parameter_names: Sequence[str],
    proxy_parameter_name: str,
    temperature: float,
) -> _OutcomeBaseline:
    import torch

    parameters = dict(model.named_parameters())
    missing = set(parameter_names) - set(parameters)
    if missing:
        raise ValueError(f"outcome parameter is absent from model: {min(missing)}")
    if proxy_parameter_name not in parameters:
        raise ValueError("proxy parameter is absent from model")
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    flags = _module_training_flags(model)
    try:
        model.eval()
        embeddings = torch.func.functional_call(
            model,
            (parameters, {name: value.clone() for name, value in buffers.items()}),
            (clean_inputs,),
            strict=True,
        )
        before_f, before_m = _outcome_scalars(
            embeddings,
            labels,
            parameters[proxy_parameter_name],
            proxy_labels,
            temperature=temperature,
            torch=torch,
        )
        selected_parameters = tuple(parameters[name] for name in parameter_names)
        foreign_values = torch.autograd.grad(
            before_f, selected_parameters, retain_graph=True, allow_unused=True
        )
        margin_values = torch.autograd.grad(
            before_m, selected_parameters, retain_graph=False, allow_unused=True
        )
    finally:
        _restore_module_training_flags(flags)
    if any(value is None for value in foreign_values + margin_values):
        raise ValueError("outcome gradient is disconnected from a required parameter")
    return _OutcomeBaseline(
        parameters=parameters,
        buffers=buffers,
        before_f=float(before_f.detach().to(torch.float64).item()),
        before_m=float(before_m.detach().to(torch.float64).item()),
        foreign_gradients={
            name: value.detach().clone()
            for name, value in zip(parameter_names, foreign_values, strict=True)
        },
        margin_gradients={
            name: value.detach().clone()
            for name, value in zip(parameter_names, margin_values, strict=True)
        },
    )


def _outcome_after_update(
    *,
    baseline: _OutcomeBaseline,
    model: Any,
    clean_inputs: Any,
    labels: Any,
    proxy_labels: Any,
    named_updates: Mapping[str, Any],
    proxy_parameter_name: str,
    temperature: float,
) -> OutcomeFields:
    import torch

    update_names = tuple(sorted(named_updates, key=lambda name: name.encode("utf-8")))
    missing_gradients = set(update_names) - set(baseline.foreign_gradients)
    if missing_gradients:
        raise ValueError(f"outcome gradient missing parameter: {min(missing_gradients)}")
    foreign_dot = sum(
        torch.sum(
            baseline.foreign_gradients[name].to(torch.float64)
            * named_updates[name].to(torch.float64)
        )
        for name in update_names
    )
    margin_dot = sum(
        torch.sum(
            baseline.margin_gradients[name].to(torch.float64)
            * named_updates[name].to(torch.float64)
        )
        for name in update_names
    )
    after_parameters = dict(baseline.parameters)
    after_parameters.update(
        {name: baseline.parameters[name] + named_updates[name] for name in update_names}
    )
    flags = _module_training_flags(model)
    try:
        model.eval()
        with torch.no_grad():
            after_embeddings = torch.func.functional_call(
                model,
                (
                    after_parameters,
                    {name: value.clone() for name, value in baseline.buffers.items()},
                ),
                (clean_inputs,),
                strict=True,
            )
            after_f, after_m = _outcome_scalars(
                after_embeddings,
                labels,
                after_parameters[proxy_parameter_name],
                proxy_labels,
                temperature=temperature,
                torch=torch,
            )
    finally:
        _restore_module_training_flags(flags)
    return OutcomeFields(
        R_F=float((baseline.before_f - float(after_f.item())) / max(baseline.before_f, 1e-6)),
        Delta_M=float(after_m.item()) - baseline.before_m,
        D_F=float((-foreign_dot / max(baseline.before_f, 1e-6)).item()),
        D_M=float(margin_dot.item()),
    )


def owner_outcomes(
    *,
    model: Any,
    clean_inputs: Any,
    labels: Any,
    proxy_labels: Any,
    named_updates: Mapping[str, Any],
    proxy_parameter_name: str,
    temperature: float = 0.05,
) -> OutcomeFields:
    """Evaluate an immutable stateless update on clean eval-mode support rows."""

    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("temperature must be positive and finite")
    update_names = tuple(sorted(named_updates, key=lambda name: name.encode("utf-8")))
    baseline = _prepare_outcome_baseline(
        model=model,
        clean_inputs=clean_inputs,
        labels=labels,
        proxy_labels=proxy_labels,
        parameter_names=update_names,
        proxy_parameter_name=proxy_parameter_name,
        temperature=temperature,
    )
    return _outcome_after_update(
        baseline=baseline,
        model=model,
        clean_inputs=clean_inputs,
        labels=labels,
        proxy_labels=proxy_labels,
        named_updates=named_updates,
        proxy_parameter_name=proxy_parameter_name,
        temperature=temperature,
    )


def shared_confuser_statistic(
    embeddings: Any,
    labels: Any,
    sample_indices: Any,
    proxies: Any,
    proxy_labels: Any,
    *,
    context_index: int,
    null_replicates: int = 256,
    null_seed: int = 2010810,
) -> dict[str, Any]:
    """Compute the frozen aligned geometric mean and streamed row-wise null."""

    import torch

    from sfora.image_end_to_end import _normalize

    if context_index < 0 or null_replicates <= 0:
        raise ValueError("context index and null replicate count must be valid")
    unique_labels, representative_indices = _representative_tensor_indices(
        labels, sample_indices, torch
    )
    members = _normalize(embeddings[representative_indices], torch)
    normalized_proxies = _normalize(proxies, torch)
    foreign_mask = ~torch.isin(proxy_labels, unique_labels)
    if not bool(foreign_mask.any()):
        raise ValueError("at least one proxy outside the bundle is required")
    q = (
        torch.sigmoid(members @ normalized_proxies[foreign_mask].T)
        .clamp(min=1e-12, max=1.0)
        .detach()
        .to(device="cpu", dtype=torch.float64)
        .numpy()
    )
    aligned = float(np.exp(np.log(q).mean(axis=0)).mean())
    null_values = np.empty(null_replicates, dtype="<f8")
    generator = np.random.Generator(np.random.PCG64(null_seed))
    for replicate in range(null_replicates):
        permuted = np.stack([row[generator.permutation(q.shape[1])] for row in q], axis=0)
        null_values[replicate] = np.exp(np.log(permuted).mean(axis=0)).mean()
    null_mean = float(null_values.mean())
    return {
        "foreign_proxy_rows": int(q.shape[1]),
        "A_aligned": aligned,
        "null_mean": null_mean,
        "E_shared": (aligned - null_mean) / max(null_mean, 1e-12),
        "null_distribution": null_values,
        "null_distribution_sha256": hashlib.sha256(null_values.tobytes(order="C")).hexdigest(),
    }


def score_context(
    *,
    model: Any,
    train_inputs: Any,
    clean_inputs: Any,
    labels: Any,
    sample_indices: Any,
    proxy_labels: Any,
    expected_trainable_parameter_names: Sequence[str],
    proxy_parameter_name: str,
    alpha: float,
    delta: float,
    learning_rate: float,
    coalition_weight: float,
    proxy_learning_rate_multiplier: float,
    context_index: int,
    temperature: float = 0.05,
) -> dict[str, Any]:
    """Score one context while preserving checkpoint state and shared graphs."""

    parameters = dict(model.named_parameters())
    if proxy_parameter_name not in parameters:
        raise ValueError("proxy parameter is absent from model")
    train_graph = bufferless_train_embeddings(model, train_inputs)
    losses = coalition_losses(
        train_graph.embeddings,
        labels,
        sample_indices,
        parameters[proxy_parameter_name],
        proxy_labels,
        alpha=alpha,
        delta=delta,
    )
    gradients = operator_gradients(
        losses,
        parameters,
        expected_trainable_parameter_names=expected_trainable_parameter_names,
        proxy_parameter_name=proxy_parameter_name,
        proxy_learning_rate_multiplier=proxy_learning_rate_multiplier,
        representative_count=int(labels.unique().numel()),
    )
    updates = make_stateless_updates(
        gradients,
        learning_rate=learning_rate,
        coalition_weight=coalition_weight,
    )
    shared_confuser = shared_confuser_statistic(
        train_graph.embeddings,
        labels,
        sample_indices,
        parameters[proxy_parameter_name],
        proxy_labels,
        context_index=context_index,
    )
    baseline = _prepare_outcome_baseline(
        model=model,
        clean_inputs=clean_inputs,
        labels=labels,
        proxy_labels=proxy_labels,
        parameter_names=tuple(expected_trainable_parameter_names),
        proxy_parameter_name=proxy_parameter_name,
        temperature=temperature,
    )
    outcomes: dict[str, Any] = {
        regime: {operator: {} for operator in OPERATORS} for regime in REGIMES
    }
    for regime in REGIMES:
        for operator in OPERATORS:
            for panel in PANELS:
                update = updates[regime][operator][panel]
                outcomes[regime][operator][panel] = _outcome_after_update(
                    baseline=baseline,
                    model=model,
                    clean_inputs=clean_inputs,
                    labels=labels,
                    proxy_labels=proxy_labels,
                    named_updates=dict(update.named_updates),
                    proxy_parameter_name=proxy_parameter_name,
                    temperature=temperature,
                )
    detached_train_graph = train_graph._replace(embeddings=train_graph.embeddings.detach())
    return {
        "losses": {name: loss.detach() for name, loss in losses.items()},
        "gradients": gradients,
        "updates": updates,
        "outcomes": outcomes,
        "shared_confuser": shared_confuser,
        "train_graph": detached_train_graph,
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
        if hasattr(value, "detach") and hasattr(value, "cpu"):
            value = value.detach().cpu().numpy()
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


def _finite_float(value: Any, path: str) -> None:
    _require(
        type(value) is float and math.isfinite(value),
        f"{path} must be a finite float",
    )


def _digest(value: Any, path: str) -> None:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{path} must be a lowercase SHA-256",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_object(path: Path, name: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"unable to read {name}") from error
    _require(isinstance(value, dict), f"{name} must be an object")
    return value


class _CheckpointMetadataUnpickler(pickle.Unpickler):
    """Decode torch-save metadata without importing torch or materializing tensors."""

    def find_class(self, module: str, name: str) -> Any:
        if module == "collections" and name == "OrderedDict":
            return OrderedDict
        if module == "torch._utils" and name.startswith("_rebuild"):
            return lambda *args: None
        if module == "torch" and name.endswith("Storage"):
            return object
        raise pickle.UnpicklingError(f"forbidden checkpoint global: {module}.{name}")

    def persistent_load(self, persistent_id: Any) -> None:
        return None


def _read_checkpoint_metadata(path: Path) -> dict[str, Any]:
    try:
        return _read_json_object(path, "checkpoint binding metadata")
    except ValueError:
        try:
            with zipfile.ZipFile(path) as archive:
                pickle_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
                _require(len(pickle_names) == 1, "checkpoint metadata pickle")
                value = _CheckpointMetadataUnpickler(
                    io.BytesIO(archive.read(pickle_names[0]))
                ).load()
        except (OSError, ValueError, zipfile.BadZipFile, pickle.UnpicklingError) as error:
            raise ValueError("unable to read checkpoint binding metadata") from error
        _require(isinstance(value, dict), "checkpoint binding metadata must be an object")
        return value


def _repo_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"bound path is outside source root: {path}") from error


def _path_tree_digest(root: Path, relative_root: str, *, python_only: bool) -> tuple[int, int, str]:
    tree_root = root / relative_root
    _require(tree_root.is_dir(), f"missing bound tree: {relative_root}")
    paths = sorted(
        (
            path
            for path in tree_root.rglob("*")
            if path.is_file() and (not python_only or path.suffix == ".py")
        ),
        key=lambda path: path.relative_to(root).as_posix().encode("utf-8"),
    )
    framed = b"".join(
        f"{_sha256_file(path)}  {path.relative_to(root).as_posix()}\n".encode() for path in paths
    )
    return (
        len(paths),
        sum(path.stat().st_size for path in paths),
        hashlib.sha256(framed).hexdigest(),
    )


def _git_output(root: Path, *arguments: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *arguments],
            cwd=root,
            check=True,
            capture_output=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("unable to authenticate executing source revision") from error


def _git_python_tree_digest(root: Path, revision: str, relative_root: str) -> tuple[int, str]:
    names = (
        _git_output(root, "ls-tree", "-r", "--name-only", revision, "--", relative_root)
        .decode("utf-8")
        .splitlines()
    )
    paths = sorted(
        (name for name in names if name.endswith(".py")),
        key=lambda name: name.encode("utf-8"),
    )
    framed = b"".join(
        (
            f"{hashlib.sha256(_git_output(root, 'show', f'{revision}:{path}')).hexdigest()}"
            f"  {path}\n"
        ).encode()
        for path in paths
    )
    return len(paths), hashlib.sha256(framed).hexdigest()


def _validate_prelaunch_source(
    prelaunch: Mapping[str, Any], *, git_root: Path, dataset_root: Path
) -> tuple[str, str]:
    _require(
        prelaunch.get("schema_version") == "pass201-pa-source-prelaunch-v1",
        "prelaunch schema_version",
    )
    _require(prelaunch.get("status") == "frozen_before_training", "prelaunch status")
    revision = prelaunch.get("local_source_revision")
    _require(
        isinstance(revision, str)
        and len(revision) == 40
        and all(character in "0123456789abcdef" for character in revision),
        "prelaunch source revision",
    )
    source = prelaunch.get("source")
    _require(isinstance(source, dict), "prelaunch source")
    relative_source_root = source.get("python_tree_root")
    _require(isinstance(relative_source_root, str) and relative_source_root, "python tree root")
    source_count, _, source_digest = _path_tree_digest(
        git_root, relative_source_root, python_only=True
    )
    _require(source_count == source.get("python_file_count"), "python file count mismatch")
    _require(
        source_digest == source.get("python_tree_merkle_sha256"),
        "executing source tree mismatch",
    )
    committed_revision = _git_output(git_root, "rev-parse", revision).decode("ascii").strip()
    _require(committed_revision == revision, "executing source revision mismatch")
    revision_count, revision_digest = _git_python_tree_digest(
        git_root, revision, relative_source_root
    )
    _require(
        revision_count == source_count and revision_digest == source_digest,
        "executing source revision tree mismatch",
    )
    for relative_path, expected in source.get("files", {}).items():
        _require(isinstance(relative_path, str) and isinstance(expected, dict), "source files")
        path = git_root / relative_path
        _require(path.is_file(), f"missing bound source file: {relative_path}")
        _require(
            path.stat().st_size == expected.get("bytes")
            and _sha256_file(path) == expected.get("sha256"),
            f"bound source file mismatch: {relative_path}",
        )

    dataset = prelaunch.get("dataset")
    _require(isinstance(dataset, dict), "prelaunch dataset")
    _require(Path(str(dataset.get("root"))).resolve() == dataset_root.resolve(), "dataset root")
    partition_path = dataset_root / str(dataset.get("partition_path"))
    _require(partition_path.is_file(), "dataset partition path")
    partition_bytes = partition_path.read_bytes()
    _require(len(partition_bytes) == dataset.get("partition_bytes"), "partition bytes")
    _require(
        hashlib.sha256(partition_bytes).hexdigest() == dataset.get("partition_sha256"),
        "partition digest",
    )
    _require(
        len(partition_bytes.splitlines()) == dataset.get("partition_line_count"),
        "partition line count",
    )
    image_root = dataset.get("image_root")
    _require(isinstance(image_root, str) and image_root, "dataset image root")
    image_count, image_bytes, image_digest = _path_tree_digest(
        dataset_root, image_root, python_only=False
    )
    _require(image_count == dataset.get("image_file_count"), "image file count")
    _require(image_bytes == dataset.get("image_total_bytes"), "image total bytes")
    _require(image_digest == dataset.get("image_tree_merkle_sha256"), "dataset tree mismatch")
    return source_digest, image_digest


def _validate_activation_config(config: Mapping[str, Any]) -> None:
    expected = {
        "dataset_name": "inshop",
        "objectives": ["proxy_anchor"],
        "seed": 0,
        "batch_size": 180,
        "samples_per_class": 0,
        "hard_class_fraction": 0.0,
        "drop_last_train_batch": True,
        "freeze_batch_norm": False,
        "freeze_batch_norm_affine": False,
        "checkpoint_selection_interval": 0,
        "proxy_count_per_class": 1,
    }
    for key, expected_value in expected.items():
        _require(
            key in config
            and type(config[key]) is type(expected_value)
            and config[key] == expected_value,
            f"resolved config {key} mismatch",
        )
    for key in ("learning_rate", "coalition_weight", "proxy_learning_rate_multiplier"):
        _finite_float(config.get(key), f"resolved config {key}")
        _require(config[key] > 0.0, f"resolved config {key} must be positive")


def _validate_train_manifest(manifest: Mapping[str, Any]) -> None:
    _require(manifest.get("schema_version") == "pass201-train-manifest-v1", "train manifest")
    rows = manifest.get("rows")
    _require(isinstance(rows, list) and bool(rows), "train manifest rows")
    ids: list[str] = []
    indices: list[int] = []
    for row in rows:
        _require(isinstance(row, dict), "train manifest row")
        _exact_keys(row, {"example_id", "sample_index", "label"}, "train manifest row")
        _require(isinstance(row["example_id"], str) and row["example_id"], "train example id")
        _require(_is_int(row["sample_index"]) and row["sample_index"] >= 0, "train index")
        _require(_is_int(row["label"]), "train label")
        ids.append(row["example_id"])
        indices.append(row["sample_index"])
    _require(len(ids) == len(set(ids)), "duplicate train example id")
    _require(len(indices) == len(set(indices)), "duplicate train sample index")


def _validate_source_v3_output(
    root: Path,
    path: Path,
    evidence: Mapping[str, Any],
    expected_relative: str,
    contract: Any,
) -> None:
    _require(path == root / expected_relative, "source-v3 output path differs")
    _require(
        type(evidence) is dict
        and set(evidence) == {"bytes", "file_type", "mode", "path", "sha256"}
        and type(evidence["bytes"]) is int
        and evidence["bytes"] >= 0
        and type(evidence["file_type"]) is str
        and evidence["file_type"] == "regular"
        and type(evidence["mode"]) is int
        and evidence["mode"] == 0o100444
        and type(evidence["path"]) is str
        and evidence["path"] == expected_relative
        and type(evidence["sha256"]) is str
        and re.fullmatch(r"[0-9a-f]{64}", evidence["sha256"]) is not None,
        "source-v3 output evidence differs",
    )
    observed = contract.verify_existing_regular_file(
        path,
        expected_mode=evidence["mode"],
        expected_bytes=evidence["bytes"],
        expected_sha256=evidence["sha256"],
    )
    _require(
        observed.file_type == "regular"
        and evidence["file_type"] == "regular"
        and observed.mode == evidence["mode"]
        and observed.byte_count == evidence["bytes"]
        and observed.sha256 == evidence["sha256"],
        "source-v3 output evidence differs",
    )


def _validate_source_v3_binding(
    args: Any, *, bound: SourceV3Authority | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(args.root).resolve(strict=True)
    if bound is None:
        bound = _load_source_v3_authority(
            root=root,
            git_root=Path(args.git_root),
            manifest_path=Path(args.prelaunch_manifest),
            receipt_path=Path(args.source_receipt),
        )
    payload = bound.authority.payload
    receipt = bound.receipt
    output_paths = {
        "report": Path(args.source_report),
        "checkpoint": Path(args.checkpoint),
        "resolved_config": Path(args.resolved_config),
        "train_manifest": Path(args.train_manifest),
    }
    for key, path in output_paths.items():
        _validate_source_v3_output(
            root,
            path,
            receipt["outputs"][key],
            payload["outputs"][key]["path"],
            bound.contract,
        )
    report = _read_json_object(output_paths["report"], "source report")
    checkpoint = _read_checkpoint_metadata(output_paths["checkpoint"])
    config = _read_json_object(output_paths["resolved_config"], "resolved config")
    train_manifest = _read_json_object(output_paths["train_manifest"], "train manifest")
    _validate_activation_config(config)
    _validate_train_manifest(train_manifest)
    _require(report.get("config") == config, "source-v3 report config mismatch")
    methods = report.get("methods")
    _require(isinstance(methods, dict) and len(methods) == 1, "ordinary PA report methods")
    method = next(iter(methods.values()))
    _require(
        isinstance(method, dict)
        and method.get("objective") == "proxy_anchor"
        and method.get("executed_train_steps") == bound.authority.expected_train_steps,
        "ordinary PA report identity",
    )
    _require(checkpoint.get("training_config") == config, "checkpoint config mismatch")
    _require(
        checkpoint.get("artifact_selection") == "final_training_state"
        and checkpoint.get("evaluation_model_source") == "student"
        and checkpoint.get("training_step") == bound.authority.expected_train_steps,
        "ordinary PA checkpoint identity",
    )
    checkpoint_epoch = checkpoint.get("checkpoint_epoch")
    _require(_is_int(checkpoint_epoch) and checkpoint_epoch >= 0, "checkpoint epoch")
    diagnostic_row = payload["source"]["files"][0]
    source_manifest = {
        "schema_version": "pass201-source-v1",
        "status": "frozen",
        "prelaunch_source_manifest_path": payload["authorization"]["manifest_path"],
        "prelaunch_source_manifest_sha256": bound.handoff.manifest_sha256,
        "source_report_path": payload["outputs"]["report"]["path"],
        "source_report_sha256": receipt["outputs"]["report"]["sha256"],
        "source_revision": bound.handoff.source_commit,
        "checkpoint_path": payload["outputs"]["checkpoint"]["path"],
        "checkpoint_sha256": receipt["outputs"]["checkpoint"]["sha256"],
        "checkpoint_bytes": receipt["outputs"]["checkpoint"]["bytes"],
        "checkpoint_epoch": checkpoint_epoch,
        "objective": "proxy_anchor",
        "seed": 0,
        "resolved_config_path": payload["outputs"]["resolved_config"]["path"],
        "resolved_config_sha256": receipt["outputs"]["resolved_config"]["sha256"],
        "train_manifest_path": payload["outputs"]["train_manifest"]["path"],
        "train_manifest_sha256": receipt["outputs"]["train_manifest"]["sha256"],
        "diagnostic_source_sha256": diagnostic_row["sha256"],
        "activated_preregistration_sha256": "",
        "torch_version": "2.12.1+cu130",
        "numpy_version": "2.5.0",
    }
    constants = {
        "batch_size": 180,
        "context_pairs": 32,
        "null_replicates": 256,
        "bootstrap_replicates": 20000,
        "s_prime_rank_seed": 2010809,
        "null_seed": 2010810,
        "bootstrap_seed": 2010811,
        "model_forward_seed": 2010812,
        "learning_rate": float(config["learning_rate"]),
        "coalition_weight": float(config["coalition_weight"]),
        "proxy_learning_rate_multiplier": float(config["proxy_learning_rate_multiplier"]),
        "owner_margin_temperature": 0.05,
    }
    return source_manifest, constants


def _validate_source_v5_binding(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    root = Path(args.root).resolve(strict=True)
    bound = _load_source_v5_authority(
        root=root,
        git_root=Path(args.git_root),
        manifest_path=Path(args.prelaunch_manifest),
        receipt_path=Path(args.source_receipt),
    )
    source_manifest, constants = _validate_source_v3_binding(args, bound=bound)
    payload = bound.authority.payload
    historical = payload["historical_producer"]
    source_manifest["schema_version"] = "pass201-source-v2"
    source_manifest["source_revision"] = SOURCE_V4_HISTORICAL_SOURCE_COMMIT
    source_manifest["activation_repair"] = {
        "historical_authorization_commit": SOURCE_V4_HISTORICAL_HANDOFF_COMMIT,
        "historical_source_commit": SOURCE_V4_HISTORICAL_SOURCE_COMMIT,
        "historical_manifest_path": historical["manifest"]["path"],
        "historical_manifest_sha256": historical["manifest"]["sha256"],
        "historical_receipt_path": historical["receipt"]["path"],
        "historical_receipt_sha256": historical["receipt"]["sha256"],
        "executor_authorization_commit": bound.handoff.handoff_commit,
        "executor_source_commit": bound.handoff.source_commit,
        "executor_manifest_path": payload["authorization"]["manifest_path"],
        "executor_manifest_sha256": bound.handoff.manifest_sha256,
        "executor_diagnostic_sha256": source_manifest["diagnostic_source_sha256"],
    }
    return source_manifest, constants


def _validate_source_binding(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    prelaunch_bytes = Path(args.prelaunch_manifest).read_bytes()
    prelaunch_bootstrap = _bootstrap_strict_json_object(
        prelaunch_bytes, "source authorization manifest"
    )
    if prelaunch_bootstrap.get("schema_version") == "pass201-pa-source-v3-prelaunch-v1":
        return _validate_source_v3_binding(args)
    if prelaunch_bootstrap.get("schema_version") == "pass201-pa-source-v5-activation-v1":
        return _validate_source_v5_binding(args)
    if prelaunch_bootstrap.get("schema_version") == "pass201-pa-source-v4-prelaunch-v1":
        raise ValueError("source-v4 authorization is repair-required")
    root = Path(args.root)
    git_root = Path(args.git_root)
    prelaunch_path = Path(args.prelaunch_manifest)
    report_path = Path(args.source_report)
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.resolved_config)
    train_manifest_path = Path(args.train_manifest)
    dataset_root = Path(args.dataset_root)
    diagnostic_path = Path(args.diagnostic_path)
    paths = (prelaunch_path, report_path, checkpoint_path, config_path, train_manifest_path)
    _require(all(path.is_file() for path in paths), "source artifact unavailable")
    expected_prelaunch = getattr(
        args, "expected_prelaunch_sha256", PRELAUNCH_SOURCE_MANIFEST_SHA256
    )
    prelaunch_digest = _sha256_file(prelaunch_path)
    _require(prelaunch_digest == expected_prelaunch, "prelaunch manifest SHA-256 mismatch")
    prelaunch = _read_json_object(prelaunch_path, "prelaunch source manifest")
    outputs = prelaunch.get("outputs")
    _require(isinstance(outputs, dict), "prelaunch outputs")
    _require(
        outputs.get("report") == _repo_relative(report_path, root)
        and outputs.get("checkpoint") == _repo_relative(checkpoint_path, root),
        "source artifact path differs from prelaunch output",
    )
    source_tree_digest, data_tree_digest = _validate_prelaunch_source(
        prelaunch, git_root=git_root, dataset_root=dataset_root
    )
    report = _read_json_object(report_path, "source report")
    checkpoint = _read_checkpoint_metadata(checkpoint_path)
    config = _read_json_object(config_path, "resolved config")
    train_manifest = _read_json_object(train_manifest_path, "train manifest")
    _validate_activation_config(config)
    _validate_train_manifest(train_manifest)

    binding = report.get("source_binding")
    _require(isinstance(binding, dict), "report source binding")
    execution = prelaunch.get("execution")
    _require(isinstance(execution, dict), "prelaunch execution")
    expected_argv = execution.get("argv")
    _require(
        isinstance(expected_argv, list)
        and all(isinstance(value, str) for value in expected_argv)
        and binding.get("argv") == expected_argv,
        "run command mismatch",
    )
    revision = prelaunch["local_source_revision"]
    expected_binding = {
        "prelaunch_source_manifest_sha256": prelaunch_digest,
        "source_revision": revision,
        "source_python_tree_sha256_before": source_tree_digest,
        "source_python_tree_sha256_after": source_tree_digest,
        "dataset_image_tree_sha256_before": data_tree_digest,
        "dataset_image_tree_sha256_after": data_tree_digest,
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "resolved_config_sha256": _sha256_file(config_path),
        "train_manifest_sha256": _sha256_file(train_manifest_path),
    }
    for key, expected_value in expected_binding.items():
        _require(binding.get(key) == expected_value, f"report binding mismatch: {key}")
    _require(report.get("config") == config, "report resolved config mismatch")
    methods = report.get("methods")
    _require(isinstance(methods, dict) and len(methods) == 1, "ordinary PA report methods")
    method = next(iter(methods.values()))
    _require(
        isinstance(method, dict)
        and method.get("objective") == "proxy_anchor"
        and method.get("executed_train_steps") == 8580,
        "ordinary PA report identity",
    )
    _require(checkpoint.get("training_config") == config, "checkpoint config mismatch")
    _require(
        checkpoint.get("artifact_selection") == "final_training_state"
        and checkpoint.get("evaluation_model_source") == "student"
        and checkpoint.get("training_step") == 8580
        and checkpoint.get("objective") == "proxy_anchor"
        and checkpoint.get("seed") == 0,
        "ordinary PA checkpoint identity",
    )
    _require(
        checkpoint.get("resolved_config_sha256") == expected_binding["resolved_config_sha256"]
        and checkpoint.get("train_manifest_sha256") == expected_binding["train_manifest_sha256"],
        "checkpoint embedded digest mismatch",
    )
    checkpoint_epoch = checkpoint.get("checkpoint_epoch")
    _require(_is_int(checkpoint_epoch) and checkpoint_epoch >= 0, "checkpoint epoch")
    _require(diagnostic_path.is_file(), "diagnostic source unavailable")
    environment = prelaunch.get("environment")
    _require(isinstance(environment, dict), "prelaunch environment")
    for key in ("python", "torch", "numpy"):
        _require(isinstance(environment.get(key), str) and environment[key], f"environment {key}")

    source_manifest = {
        "schema_version": "pass201-source-v1",
        "status": "frozen",
        "prelaunch_source_manifest_path": PRELAUNCH_SOURCE_MANIFEST_PATH,
        "prelaunch_source_manifest_sha256": prelaunch_digest,
        "source_report_path": _repo_relative(report_path, root),
        "source_report_sha256": _sha256_file(report_path),
        "source_revision": revision,
        "checkpoint_path": _repo_relative(checkpoint_path, root),
        "checkpoint_sha256": expected_binding["checkpoint_sha256"],
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "checkpoint_epoch": checkpoint_epoch,
        "objective": "proxy_anchor",
        "seed": 0,
        "resolved_config_path": _repo_relative(config_path, root),
        "resolved_config_sha256": expected_binding["resolved_config_sha256"],
        "train_manifest_path": _repo_relative(train_manifest_path, root),
        "train_manifest_sha256": expected_binding["train_manifest_sha256"],
        "diagnostic_source_sha256": _sha256_file(diagnostic_path),
        "activated_preregistration_sha256": "",
        "torch_version": environment["torch"],
        "numpy_version": environment["numpy"],
    }
    constants = {
        "batch_size": 180,
        "context_pairs": 32,
        "null_replicates": 256,
        "bootstrap_replicates": 20000,
        "s_prime_rank_seed": 2010809,
        "null_seed": 2010810,
        "bootstrap_seed": 2010811,
        "model_forward_seed": 2010812,
        "learning_rate": float(config["learning_rate"]),
        "coalition_weight": float(config["coalition_weight"]),
        "proxy_learning_rate_multiplier": float(config["proxy_learning_rate_multiplier"]),
        "owner_margin_temperature": 0.05,
    }
    return source_manifest, constants


SOURCE_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "prelaunch_source_manifest_path",
    "prelaunch_source_manifest_sha256",
    "source_report_path",
    "source_report_sha256",
    "source_revision",
    "checkpoint_path",
    "checkpoint_sha256",
    "checkpoint_bytes",
    "checkpoint_epoch",
    "objective",
    "seed",
    "resolved_config_path",
    "resolved_config_sha256",
    "train_manifest_path",
    "train_manifest_sha256",
    "diagnostic_source_sha256",
    "activated_preregistration_sha256",
    "torch_version",
    "numpy_version",
}
ACTIVATION_REPAIR_KEYS = (
    "historical_authorization_commit",
    "historical_source_commit",
    "historical_manifest_path",
    "historical_manifest_sha256",
    "historical_receipt_path",
    "historical_receipt_sha256",
    "executor_authorization_commit",
    "executor_source_commit",
    "executor_manifest_path",
    "executor_manifest_sha256",
    "executor_diagnostic_sha256",
)


def _validate_activation_repair(
    repair: Any,
    *,
    prelaunch_path: Any,
    prelaunch_sha256: Any,
    source_revision: Any,
    diagnostic_sha256: Any,
) -> None:
    _require(
        type(repair) is dict and list(repair) == list(ACTIVATION_REPAIR_KEYS),
        "source manifest activation_repair keys/order",
    )
    _require(
        repair["historical_authorization_commit"] == SOURCE_V4_HISTORICAL_HANDOFF_COMMIT
        and repair["historical_source_commit"] == SOURCE_V4_HISTORICAL_SOURCE_COMMIT
        and source_revision == SOURCE_V4_HISTORICAL_SOURCE_COMMIT,
        "source manifest historical commits differ",
    )
    _require(
        repair["historical_manifest_path"] == SOURCE_V4_AUTHORIZATION_MANIFEST_PATH
        and repair["historical_manifest_sha256"]
        == "080adaeaaa5c7bf9c87ed93761d6e4c517b958bb60c49af68a880109f5abce1f"
        and repair["historical_receipt_path"]
        == "reports/generated/pass201_source_v3/run-v3/receipt.json"
        and repair["historical_receipt_sha256"]
        == "a494ead85167539670f8c5d1481f8d9eabc274776727df06d7362e99e9b7cdf9",
        "source manifest historical artifacts differ",
    )
    for key in ("executor_authorization_commit", "executor_source_commit"):
        _require(
            type(repair[key]) is str
            and len(repair[key]) == 40
            and all(character in "0123456789abcdef" for character in repair[key]),
            f"source manifest {key}",
        )
    _require(
        prelaunch_path == SOURCE_V5_AUTHORIZATION_MANIFEST_PATH
        and repair["executor_manifest_path"] == SOURCE_V5_AUTHORIZATION_MANIFEST_PATH
        and repair["executor_manifest_sha256"] == prelaunch_sha256
        and repair["executor_diagnostic_sha256"] == diagnostic_sha256,
        "source manifest executor provenance differs",
    )


def _validate_source_manifest_artifact(manifest: Any) -> None:
    _require(type(manifest) is dict, "source manifest must be an object")
    is_v2 = manifest.get("schema_version") == "pass201-source-v2"
    expected_keys = SOURCE_MANIFEST_KEYS | ({"activation_repair"} if is_v2 else set())
    _exact_keys(manifest, expected_keys, "source manifest")
    _require(
        is_v2 or manifest["schema_version"] == "pass201-source-v1",
        "source manifest schema",
    )
    _require(manifest["status"] == "frozen", "source manifest status")
    if is_v2:
        _require(
            manifest["prelaunch_source_manifest_path"] == SOURCE_V5_AUTHORIZATION_MANIFEST_PATH,
            "source manifest prelaunch path",
        )
    else:
        _require(
            manifest["prelaunch_source_manifest_path"]
            in (
                PRELAUNCH_SOURCE_MANIFEST_PATH,
                SOURCE_V3_AUTHORIZATION_MANIFEST_PATH,
                SOURCE_V4_AUTHORIZATION_MANIFEST_PATH,
            ),
            "source manifest prelaunch path",
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
        _digest(manifest[key], f"source manifest {key}")
    for key in (
        "source_report_path",
        "source_revision",
        "checkpoint_path",
        "resolved_config_path",
        "train_manifest_path",
        "torch_version",
        "numpy_version",
    ):
        _require(isinstance(manifest[key], str) and manifest[key], f"source manifest {key}")
    _require(
        _is_int(manifest["checkpoint_bytes"]) and manifest["checkpoint_bytes"] > 0,
        "source manifest checkpoint_bytes",
    )
    _require(
        _is_int(manifest["checkpoint_epoch"]) and manifest["checkpoint_epoch"] >= 0,
        "source manifest checkpoint_epoch",
    )
    _require(manifest["objective"] == "proxy_anchor", "source manifest objective")
    _require(manifest["seed"] == 0 and _is_int(manifest["seed"]), "source manifest seed")
    if is_v2:
        _validate_activation_repair(
            manifest["activation_repair"],
            prelaunch_path=manifest["prelaunch_source_manifest_path"],
            prelaunch_sha256=manifest["prelaunch_source_manifest_sha256"],
            source_revision=manifest["source_revision"],
            diagnostic_sha256=manifest["diagnostic_source_sha256"],
        )


def _validate_activated_preregistration(payload: Any) -> None:
    keys = {
        "schema_version",
        "frozen_draft_path",
        "frozen_draft_sha256",
        "result_path",
        "source",
        "constants",
        "thresholds",
        "authorized_action",
    }
    _exact_keys(payload, keys, "activated preregistration")
    _require(
        payload["schema_version"] == "pass201-cis-activated-preregistration-v1",
        "activated preregistration schema",
    )
    _require(payload["frozen_draft_path"] == FROZEN_DRAFT_PATH, "frozen draft path")
    _require(payload["frozen_draft_sha256"] == FROZEN_DRAFT_SHA256, "frozen draft digest")
    _require(payload["result_path"] == RESULT_PATH, "activated result path")
    _require(
        payload["authorized_action"] == "binding_and_integrity_smoke_then_scientific_if_green",
        "activated authorized action",
    )
    is_v2 = "activation_repair" in payload["source"]
    source_keys = (SOURCE_MANIFEST_KEYS | ({"activation_repair"} if is_v2 else set())) - {
        "schema_version",
        "status",
        "activated_preregistration_sha256",
    }
    _exact_keys(payload["source"], source_keys, "activated source")
    source_for_validation = {
        **payload["source"],
        "schema_version": "pass201-source-v2" if is_v2 else "pass201-source-v1",
        "status": "frozen",
        "activated_preregistration_sha256": "0" * 64,
    }
    _validate_source_manifest_artifact(source_for_validation)
    _validate_constants(payload["constants"], activated=True)
    _exact_keys(payload["thresholds"], THRESHOLDS, "activated thresholds")
    _require(payload["thresholds"] == THRESHOLDS, "activated thresholds mismatch")


def _publish_source_v5_candidate(path: Path, data: bytes) -> None:
    if not path.parent.is_dir() or path.parent.is_symlink():
        raise ValueError("source-v5 candidate parent must be a real directory")
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    if any(path.parent.glob(f".{path.name}.tmp-*")):
        raise ValueError("source-v5 candidate temporary path already exists")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    published = False

    def fsync_directory() -> None:
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)

    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary_identity = (temporary.stat().st_dev, temporary.stat().st_ino)
        os.link(temporary, path, follow_symlinks=False)
        published = True
        fsync_directory()
        temporary.unlink()
        fsync_directory()
    except BaseException:
        if published:
            with contextlib.suppress(FileNotFoundError):
                current = path.stat(follow_symlinks=False)
                if (current.st_dev, current.st_ino) == temporary_identity:
                    path.unlink()
        temporary.unlink(missing_ok=True)
        with contextlib.suppress(OSError):
            fsync_directory()
        raise


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
        raise


def _atomic_write_json(path: Path, payload: object, *, sort_keys: bool = True) -> bytes:
    data = (
        canonical_json_bytes(payload)
        if sort_keys
        else json.dumps(
            payload,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ) + b"\n"
    _atomic_write_bytes(path, data)
    return data


def _publish_json_no_replace(path: Path, payload: object, *, sort_keys: bool = True) -> bytes:
    data = (
        canonical_json_bytes(payload)
        if sort_keys
        else json.dumps(
            payload,
            sort_keys=False,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    ) + b"\n"
    _publish_source_v5_candidate(path, data)
    return data


def _restore_atomic_file(path: Path, prior: bytes | None) -> None:
    if prior is None:
        try:
            path.unlink()
        except FileNotFoundError:
            return
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    else:
        _atomic_write_bytes(path, prior)


def activate_source(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Authenticate source artifacts before importing torch and freeze activation."""

    source_manifest, constants = _validate_source_binding(args)
    preregistration_source = {
        key: value
        for key, value in source_manifest.items()
        if key not in {"schema_version", "status", "activated_preregistration_sha256"}
    }
    preregistration = {
        "schema_version": "pass201-cis-activated-preregistration-v1",
        "frozen_draft_path": FROZEN_DRAFT_PATH,
        "frozen_draft_sha256": FROZEN_DRAFT_SHA256,
        "result_path": RESULT_PATH,
        "source": preregistration_source,
        "constants": constants,
        "thresholds": dict(THRESHOLDS),
        "authorized_action": "binding_and_integrity_smoke_then_scientific_if_green",
    }
    _validate_activated_preregistration(preregistration)
    preregistration_path = Path(
        getattr(
            args,
            "activated_preregistration",
            Path(args.root) / ACTIVATED_PREREGISTRATION_PATH,
        )
    )
    source_manifest_path = Path(
        getattr(args, "source_manifest", Path(args.root) / SOURCE_MANIFEST_PATH)
    )
    prior_preregistration = (
        preregistration_path.read_bytes() if preregistration_path.exists() else None
    )
    prior_manifest = source_manifest_path.read_bytes() if source_manifest_path.exists() else None
    try:
        is_v2 = source_manifest["schema_version"] == "pass201-source-v2"
        preregistration_bytes = _atomic_write_json(
            preregistration_path,
            preregistration,
            sort_keys=not is_v2,
        )
        source_manifest["activated_preregistration_sha256"] = hashlib.sha256(
            preregistration_bytes
        ).hexdigest()
        _validate_source_manifest_artifact(source_manifest)
        _atomic_write_json(source_manifest_path, source_manifest, sort_keys=not is_v2)
        persisted_preregistration = _read_json_object(
            preregistration_path, "activated preregistration"
        )
        persisted_manifest = _read_json_object(source_manifest_path, "source manifest")
        _validate_activated_preregistration(persisted_preregistration)
        _validate_source_manifest_artifact(persisted_manifest)
        _require(
            persisted_manifest["activated_preregistration_sha256"]
            == hashlib.sha256(preregistration_path.read_bytes()).hexdigest(),
            "activated preregistration byte digest mismatch",
        )
    except BaseException:
        _restore_atomic_file(preregistration_path, prior_preregistration)
        _restore_atomic_file(source_manifest_path, prior_manifest)
        raise
    return preregistration, source_manifest


def _import_torch() -> Any:
    import torch

    return torch


def _load_process_runtime(source_manifest: Mapping[str, Any], torch: Any) -> Any:
    factory_path = os.environ.get("PASS201_RUNTIME_FACTORY")
    if not factory_path:
        return _ProductionRuntime(source_manifest, torch)
    path = Path(factory_path)
    _require(path.is_file(), "PASS201 runtime factory is unavailable")
    module_name = f"_pass201_runtime_{hashlib.sha256(str(path).encode()).hexdigest()}"
    specification = importlib.util.spec_from_file_location(module_name, path)
    _require(
        specification is not None and specification.loader is not None,
        "unable to load PASS201 runtime factory",
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    factory = getattr(module, "build_runtime", None)
    _require(callable(factory), "PASS201 runtime factory lacks build_runtime")
    return factory(dict(source_manifest), torch)


def _select_production_cpu_device(torch: Any) -> Any:
    cuda_available = torch.cuda.is_available()
    _require(
        type(cuda_available) is bool and cuda_available is False,
        "Pass201 production runtime requires a CPU runtime",
    )
    return torch.device("cpu")


class _ProductionRuntime:
    """Bound production adapter; imports model/data code only inside a child."""

    def __init__(self, source_manifest: Mapping[str, Any], torch: Any):
        from sfora.data import ImageExample
        from sfora.image_end_to_end import (
            ImageEndToEndConfig,
            _default_transform_factory,
            _metric_proxy_labels,
            _torchvision_model_factory,
        )

        self.torch = torch
        self.root = Path(os.environ.get("PASS201_SOURCE_ROOT", ".")).resolve()
        config_path = self.root / str(source_manifest["resolved_config_path"])
        train_manifest_path = self.root / str(source_manifest["train_manifest_path"])
        checkpoint_path = self.root / str(source_manifest["checkpoint_path"])
        config_payload = _read_json_object(config_path, "resolved config")
        manifest_payload = _read_json_object(train_manifest_path, "train manifest")
        _validate_activation_config(config_payload)
        _validate_train_manifest(manifest_payload)
        self.config = ImageEndToEndConfig(**config_payload)
        dataset_root = Path(self.config.dataset_root or "")
        _require(dataset_root.is_dir(), "activated dataset root is unavailable")
        rows = manifest_payload["rows"]
        _require(
            [row["sample_index"] for row in rows] == list(range(len(rows))),
            "train manifest stable indices must be contiguous and ordered",
        )
        self.train_manifest = rows
        self.examples = []
        for row in rows:
            relative = Path(row["example_id"])
            image_path = dataset_root / relative
            if not image_path.is_file():
                image_path = dataset_root / "img" / relative
            _require(image_path.is_file(), f"missing train image: {row['example_id']}")
            self.examples.append(
                ImageExample(
                    example_id=row["example_id"],
                    image=image_path,
                    label=row["label"],
                )
            )
        self.train_transform = _default_transform_factory(self.config, True)
        self.clean_transform = _default_transform_factory(self.config, False)
        device = _select_production_cpu_device(torch)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        _require(
            isinstance(checkpoint, dict) and checkpoint.get("training_config") == config_payload,
            "checkpoint training config replay mismatch",
        )
        self.model = _torchvision_model_factory(self.config).to(device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)
        proxy_names = [
            name for name, _ in self.model.named_parameters() if name.endswith("metric_proxies")
        ]
        _require(len(proxy_names) == 1, "checkpoint must expose one metric proxy parameter")
        self.proxy_parameter_name = proxy_names[0]
        proxy_labels = _metric_proxy_labels(self.model)
        _require(proxy_labels is not None, "checkpoint metric proxy labels unavailable")
        self.proxy_labels = proxy_labels.to(device=device)
        self.parameter_names = tuple(
            sorted(
                (
                    name
                    for name, parameter in self.model.named_parameters()
                    if parameter.requires_grad
                ),
                key=lambda name: name.encode("utf-8"),
            )
        )
        self.parameter_hash_before = sha256_named_tensors(self.model.named_parameters())
        self.buffer_hash_before = sha256_named_tensors(self.model.named_buffers())
        self.prepared: list[dict[str, Any]] = []
        self.scored: list[dict[str, Any]] = []
        self.rejected_context_count = 0

    def prepare_contexts(self) -> list[dict[str, Any]]:
        from torch.utils.data import DataLoader

        from sfora.image_end_to_end import _IndexedTorchImageDataset, _TorchImageDataset

        train_dataset = _IndexedTorchImageDataset(self.examples, self.train_transform)
        clean_dataset = _TorchImageDataset(self.examples, self.clean_transform)
        generator = self.torch.Generator()
        generator.manual_seed(int(self.config.seed))
        loader = DataLoader(
            train_dataset,
            batch_size=int(self.config.batch_size),
            shuffle=True,
            generator=generator,
            num_workers=int(self.config.num_workers),
            pin_memory=self.torch.cuda.is_available(),
            drop_last=bool(self.config.drop_last_train_batch),
        )
        accepted: list[dict[str, Any]] = []
        for production_batch_index, (train_tensor, labels, sample_indices) in enumerate(loader):
            rows = [self.train_manifest[int(index)] for index in sample_indices.tolist()]
            try:
                metadata = construct_one_context(
                    rows=rows,
                    train_manifest=self.train_manifest,
                    context_index=len(accepted),
                    production_epoch=0,
                    production_batch_index=production_batch_index,
                    prior_contexts=[entry["context"] for entry in accepted],
                )
            except ValueError as error:
                if str(error) == "INSUFFICIENT_DISJOINT_S_PRIME":
                    self.rejected_context_count += 1
                    continue
                raise
            clean_rows = [clean_dataset[index][0] for index in metadata["s_prime_sample_indices"]]
            clean_tensor = self.torch.stack(clean_rows, dim=0)
            context = {
                **metadata,
                "batch_size": 180,
                "m_unique": len(metadata["class_multiplicities"]),
                "s_tensor_sha256": sha256_tensor_frame(train_tensor),
                "s_prime_tensor_sha256": sha256_tensor_frame(clean_tensor),
            }
            digest_source = {
                **context,
                "s_tensor": train_tensor,
                "s_prime_tensor": clean_tensor,
            }
            accepted.append(
                {
                    "digest_record": build_input_context_digest(digest_source),
                    "context": context,
                    "train_tensor": train_tensor,
                    "clean_tensor": clean_tensor,
                    "labels": labels,
                    "sample_indices": sample_indices,
                }
            )
            if len(accepted) == 32:
                break
        _require(len(accepted) == 32, "UNRESOLVED_INSUFFICIENT_DISJOINT_CONTEXTS")
        self.prepared = accepted
        return [
            {"digest_record": entry["digest_record"], "context": entry["context"]}
            for entry in accepted
        ]

    def score_context(self, index: int, prepared: Mapping[str, Any]) -> dict[str, Any]:
        _require(prepared["digest_record"] == self.prepared[index]["digest_record"], "context")
        entry = self.prepared[index]
        device = next(self.model.parameters()).device
        score = score_context(
            model=self.model,
            train_inputs=entry["train_tensor"].to(device=device, dtype=self.torch.float32),
            clean_inputs=entry["clean_tensor"].to(device=device, dtype=self.torch.float32),
            labels=entry["labels"].to(device=device, dtype=self.torch.long),
            sample_indices=entry["sample_indices"].to(device=device, dtype=self.torch.long),
            proxy_labels=self.proxy_labels,
            expected_trainable_parameter_names=self.parameter_names,
            proxy_parameter_name=self.proxy_parameter_name,
            alpha=float(self.config.proxy_anchor_alpha),
            delta=float(self.config.proxy_anchor_delta),
            learning_rate=float(self.config.learning_rate),
            coalition_weight=float(self.config.coalition_weight),
            proxy_learning_rate_multiplier=float(self.config.proxy_learning_rate_multiplier),
            context_index=index,
        )
        context = materialize_scored_context(entry["context"], score)
        scalar_replay: dict[str, float] = {}
        for operator in OPERATORS:
            scalar_replay[f"{operator}.loss"] = context["operators"][operator]["loss"]
            for panel in PANELS:
                panel_record = context["operators"][operator]["panels"][panel]
                for key in (
                    "raw_gradient_norm",
                    "update_space_norm",
                    "auxiliary_to_pa_norm_ratio",
                    "cosine_with_pa",
                    "cosine_with_atomic_full_union",
                    "cosine_with_summed_dropout",
                ):
                    scalar_replay[f"{operator}.{panel}.{key}"] = panel_record[key]
                for regime in REGIMES:
                    update = panel_record["updates"][regime]
                    for metric in ("parameter_update_norm", *OUTCOME_METRICS):
                        scalar_replay[f"{operator}.{panel}.{regime}.{metric}"] = update[metric]
        self.scored.append({"context": context, "score": score})
        return {
            "context": context,
            "replay_tensors": {},
            "replay_scalars": scalar_replay,
        }

    def finalize(self, contexts: list[dict[str, Any]]) -> dict[str, Any]:
        aggregates, _ = aggregate_scored_contexts(contexts)
        parameter_hash_after = sha256_named_tensors(self.model.named_parameters())
        buffer_hash_after = sha256_named_tensors(self.model.named_buffers())
        _require(
            parameter_hash_after == self.parameter_hash_before,
            "scientific process mutated checkpoint parameters",
        )
        _require(
            buffer_hash_after == self.buffer_hash_before,
            "scientific process mutated checkpoint buffers",
        )
        return {
            "schema_version": "pass201-scientific-core-v1",
            "contexts": contexts,
            "aggregates": aggregates,
            "parameter_hash_before": self.parameter_hash_before,
            "parameter_hash_after": parameter_hash_after,
            "buffer_hash_before": self.buffer_hash_before,
            "buffer_hash_after": buffer_hash_after,
            "training_flags_restored": True,
            "rejected_context_count": self.rejected_context_count,
            "all_finite": True,
        }


PROCESS_ROLES = ("integrity_replay_a", "integrity_replay_b", "scientific")
CPU_THREAD_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
}
PROCESS_OUTPUT_KEYS = {
    "schema_version",
    "status",
    "role",
    "process_record",
    "contexts",
    "aggregate_context_indices",
    "replay_tensors",
    "replay_scalars",
    "result",
}


def _rng_sha256(value: Any) -> str:
    if hasattr(value, "detach"):
        data = value.detach().to(device="cpu").contiguous().numpy().tobytes(order="C")
    else:
        array = np.ascontiguousarray(np.asarray(value))
        data = array.tobytes(order="C")
    return hashlib.sha256(data).hexdigest()


def _require_cpu_process_environment() -> None:
    for key, expected in CPU_THREAD_ENVIRONMENT.items():
        _require(
            os.environ.get(key) == expected,
            f"{key} must equal {expected!r} before torch import",
        )


def _configure_cpu_threading(torch: Any) -> tuple[int, int]:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    num_threads = torch.get_num_threads()
    num_interop_threads = torch.get_num_interop_threads()
    _require(
        type(num_threads) is int and num_threads == 1,
        "torch intra-op thread count differs",
    )
    _require(
        type(num_interop_threads) is int and num_interop_threads == 1,
        "torch inter-op thread count differs",
    )
    return num_threads, num_interop_threads


def _initialize_deterministic_process(
    torch: Any, *, thread_counts: tuple[int, int] | None = None
) -> dict[str, Any]:
    _require(
        os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8",
        "CUBLAS_WORKSPACE_CONFIG must be set before torch import",
    )
    cuda_count = int(torch.cuda.device_count())
    _require(cuda_count == 0, "visible CUDA devices are forbidden for Pass201 replay")
    if thread_counts is None:
        thread_counts = _configure_cpu_threading(torch)
    num_threads, num_interop_threads = thread_counts
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    random.seed(2010812)
    np.random.seed(2010812)
    torch.manual_seed(2010812)
    torch.cuda.manual_seed_all(2010812)
    python_digest = hashlib.sha256(pickle.dumps(random.getstate(), protocol=4)).hexdigest()
    numpy_digest = hashlib.sha256(pickle.dumps(np.random.get_state(), protocol=4)).hexdigest()
    torch_cpu_digest = _rng_sha256(torch.get_rng_state())
    visible_devices = ["cpu"]
    cuda_digests = {"0": torch_cpu_digest}
    cudnn_version_function = getattr(torch.backends.cudnn, "version", None)
    cudnn_version = cudnn_version_function() if callable(cudnn_version_function) else None
    cuda_version = str(getattr(torch.version, "cuda", None) or "unavailable")
    cudnn_version_text = str(cudnn_version or "unavailable")
    _require(
        cuda_version == "13.0" and cudnn_version_text == "92000",
        "Pass201 CPU build version differs from the registered runtime",
    )
    return {
        "accelerator": "cpu",
        "python_version": platform.python_version(),
        "torch_version": str(torch.__version__),
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version_text,
        "visible_cuda_devices": visible_devices,
        "initial_python_rng_sha256": python_digest,
        "initial_numpy_rng_sha256": numpy_digest,
        "initial_torch_cpu_rng_sha256": torch_cpu_digest,
        "initial_torch_cuda_rng_sha256_by_device": cuda_digests,
        "deterministic_settings": {
            "cublas_workspace_config": ":4096:8",
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "autocast": False,
            "dtype": "float32",
            "torch_num_threads": num_threads,
            "torch_num_interop_threads": num_interop_threads,
        },
    }


def _validate_process_output(payload: Any, role: str) -> None:
    _exact_keys(payload, PROCESS_OUTPUT_KEYS, "process output")
    _require(payload["schema_version"] == "pass201-process-v1", "process schema")
    _require(payload["status"] == "ok", "process status")
    _require(payload["role"] == role, "process role")
    _validate_process_record(
        payload["process_record"], role, "process_record", context0_required=True
    )
    expected_contexts = 32 if role == "scientific" else 1
    _require(
        isinstance(payload["contexts"], list) and len(payload["contexts"]) == expected_contexts,
        "process scored context count",
    )
    _require(
        payload["aggregate_context_indices"] == (list(range(32)) if role == "scientific" else []),
        "process aggregate ownership",
    )
    _require(
        payload["process_record"]["context0_record_sha256"]
        == hashlib.sha256(canonical_json_bytes(payload["contexts"][0])).hexdigest(),
        "context-0 record digest mismatch",
    )
    for key in ("replay_tensors", "replay_scalars"):
        _require(isinstance(payload[key], dict), f"process {key}")
    if role != "scientific":
        _require(payload["result"] is None, "integrity process cannot emit aggregate result")


def run_process_role(role: str, source_manifest: Mapping[str, Any], output_path: Path) -> None:
    """Run one fresh deterministic replay role and atomically publish its record."""

    _require(role in PROCESS_ROLES, "unknown process role")
    _require(
        os.environ.get("PASS201_PROCESS_ROLE", role) == role,
        "process role environment mismatch",
    )
    _require_cpu_process_environment()
    torch = _import_torch()
    thread_counts = _configure_cpu_threading(torch)
    environment = _initialize_deterministic_process(torch, thread_counts=thread_counts)
    runtime = _load_process_runtime(source_manifest, torch)
    prepared_contexts = runtime.prepare_contexts()
    _require(
        isinstance(prepared_contexts, list) and len(prepared_contexts) == 32,
        "process must prepare exactly 32 contexts",
    )
    digest_records = []
    for index, prepared in enumerate(prepared_contexts):
        _require(isinstance(prepared, dict), "prepared context")
        digest_record = prepared.get("digest_record")
        _validate_input_digest(digest_record, index, f"prepared_contexts[{index}].digest_record")
        digest_records.append(digest_record)

    score_indices = range(32) if role == "scientific" else (0,)
    contexts = []
    context_zero_tensors: dict[str, Any] | None = None
    context_zero_scalars: dict[str, Any] | None = None
    for index in score_indices:
        score = runtime.score_context(index, prepared_contexts[index])
        _require(isinstance(score, dict), "runtime score record")
        _exact_keys(
            score,
            {"context", "replay_tensors", "replay_scalars"},
            "runtime score record",
        )
        context = score["context"]
        _require(isinstance(context, dict), "scored context")
        contexts.append(context)
        if index == 0:
            context_zero_tensors = score["replay_tensors"]
            context_zero_scalars = score["replay_scalars"]
    _require(context_zero_tensors is not None and context_zero_scalars is not None, "context 0")
    context_zero_sha256 = hashlib.sha256(canonical_json_bytes(contexts[0])).hexdigest()
    process_record = {
        "role": role,
        "pid": os.getpid(),
        **environment,
        "prepared_context_count": 32,
        "input_context_digest_records": digest_records,
        "context0_record_sha256": context_zero_sha256,
    }
    result = runtime.finalize(contexts) if role == "scientific" else None
    payload = {
        "schema_version": "pass201-process-v1",
        "status": "ok",
        "role": role,
        "process_record": process_record,
        "contexts": contexts,
        "aggregate_context_indices": list(range(32)) if role == "scientific" else [],
        "replay_tensors": context_zero_tensors,
        "replay_scalars": context_zero_scalars,
        "result": result,
    }
    _validate_process_output(payload, role)
    _atomic_write_json(Path(output_path), payload)


def materialize_scored_context(
    base_context: Mapping[str, Any], score: Mapping[str, Any]
) -> dict[str, Any]:
    """Convert Task 2's pure scientific records to the frozen context schema."""

    required = {
        "losses",
        "gradients",
        "updates",
        "outcomes",
        "shared_confuser",
        "train_graph",
    }
    _exact_keys(score, required, "Task 2 score")
    representative_count = base_context.get("m_unique")
    _require(
        _is_int(representative_count) and representative_count > 0,
        "scored context m_unique",
    )
    operators: dict[str, Any] = {}
    for operator in OPERATORS:
        panels: dict[str, Any] = {}
        for panel in PANELS:
            gradient = score["gradients"][f"{operator}.{panel}"]
            update_records: dict[str, Any] = {}
            for regime in REGIMES:
                update = score["updates"][regime][operator][panel]
                outcome = score["outcomes"][regime][operator][panel]
                update_records[regime] = {
                    "update_sha256": update.update_sha256,
                    "parameter_update_norm": float(update.parameter_update_norm),
                    "R_F": float(outcome.R_F),
                    "Delta_M": float(outcome.Delta_M),
                    "D_F": float(outcome.D_F),
                    "D_M": float(outcome.D_M),
                    "reference_pa_norm": (
                        None
                        if update.reference_pa_norm is None
                        else float(update.reference_pa_norm)
                    ),
                    "norm_match_absolute_error": (
                        None
                        if update.norm_match_absolute_error is None
                        else float(update.norm_match_absolute_error)
                    ),
                }
            panels[panel] = {
                "parameter_count": int(gradient.parameter_count),
                "gradient_sha256": gradient.gradient_sha256,
                "raw_gradient_norm": float(gradient.raw_gradient_norm),
                "update_space_norm": float(gradient.update_space_norm),
                "auxiliary_to_pa_norm_ratio": float(gradient.auxiliary_to_pa_norm_ratio),
                "cosine_with_pa": float(gradient.cosine_with_pa),
                "cosine_with_atomic_full_union": float(gradient.cosine_with_atomic_full_union),
                "cosine_with_summed_dropout": float(gradient.cosine_with_summed_dropout),
                "scale_residual_to_summed_union": (
                    None
                    if gradient.scale_residual_to_summed_union is None
                    else float(gradient.scale_residual_to_summed_union)
                ),
                "updates": update_records,
            }
        loss = score["losses"][operator]
        loss_value = loss.item() if hasattr(loss, "item") else loss
        operators[operator] = {
            "name": operator,
            "loss": float(loss_value),
            "representative_count": representative_count,
            "panels": panels,
        }
    shared = score["shared_confuser"]
    context = {
        **dict(base_context),
        "foreign_proxy_rows": int(shared["foreign_proxy_rows"]),
        "shared_confuser": {
            "A_aligned": float(shared["A_aligned"]),
            "null_mean": float(shared["null_mean"]),
            "E_shared": float(shared["E_shared"]),
            "null_distribution_sha256": shared["null_distribution_sha256"],
        },
        "operators": operators,
    }
    return context


def aggregate_scored_contexts(
    contexts: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Aggregate only the scientific process's 32 retained context records."""

    _require(len(contexts) == 32, "scientific aggregate requires 32 contexts")
    indices = bootstrap_indices()
    distributions: dict[str, np.ndarray] = {}

    def add_summary(path: str, values: Sequence[float | int]) -> dict[str, Any]:
        vector = np.asarray(values, dtype="<f8")
        summary = summarize_metric(vector, indices)
        distributions[path] = bootstrap_mean_distribution(vector, indices)
        return summary

    aggregates: dict[str, Any] = {
        "m_unique": add_summary("m_unique", [context["m_unique"] for context in contexts]),
        "shared_confuser": add_summary(
            "shared_confuser.E_shared",
            [context["shared_confuser"]["E_shared"] for context in contexts],
        ),
    }
    for regime in REGIMES:
        regime_record: dict[str, Any] = {}
        for panel in PANELS:
            operator_records: dict[str, Any] = {}
            for operator in OPERATORS:
                metrics: dict[str, Any] = {}
                for metric in OUTCOME_METRICS:
                    path = f"{regime}.{panel}.operators.{operator}.{metric}"
                    values = [
                        context["operators"][operator]["panels"][panel]["updates"][regime][metric]
                        for context in contexts
                    ]
                    metrics[metric] = add_summary(path, values)
                operator_records[operator] = metrics
            advantages: dict[str, Any] = {}
            for advantage, metric in (("A_F", "R_F"), ("A_M", "Delta_M")):
                path = f"{regime}.{panel}.paired_advantages.{advantage}"
                values = [
                    context["operators"]["summed_union"]["panels"][panel]["updates"][regime][metric]
                    - context["operators"]["atomic_full_union"]["panels"][panel]["updates"][regime][
                        metric
                    ]
                    for context in contexts
                ]
                advantages[advantage] = add_summary(path, values)
            regime_record[panel] = {
                "operators": operator_records,
                "paired_advantages": advantages,
            }
        aggregates[regime] = regime_record
    aggregates["bootstrap"] = {
        "seed": BOOTSTRAP_SEED,
        "replicates": BOOTSTRAP_REPLICATES,
        "quantile_method": "linear",
        "joint_context_index_sha256": sha256_bootstrap_indices(indices),
        "distribution_sha256_by_metric": {
            path: sha256_bootstrap_distribution(distributions[path])
            for path in sorted(distributions, key=lambda value: value.encode("utf-8"))
        },
    }
    return aggregates, distributions


def _flatten_finite_numbers(value: Any, path: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path} must be numeric") from error
    _require(array.size > 0 and np.isfinite(array).all(), f"{path} must be finite")
    return array.reshape(-1)


def _replay_maxima(
    a: Mapping[str, Any], b: Mapping[str, Any], scientific: Mapping[str, Any]
) -> tuple[float, float]:
    payloads = (a, b, scientific)
    tensor_keys = set(a["replay_tensors"])
    scalar_keys = set(a["replay_scalars"])
    _require(
        all(set(payload["replay_tensors"]) == tensor_keys for payload in payloads),
        "tensor replay keys",
    )
    _require(
        all(set(payload["replay_scalars"]) == scalar_keys for payload in payloads),
        "scalar replay keys",
    )
    tensor_max = 0.0
    scalar_max = 0.0
    pairs = ((a, b), (a, scientific), (b, scientific))
    for left, right in pairs:
        for key in tensor_keys:
            left_values = _flatten_finite_numbers(left["replay_tensors"][key], key)
            right_values = _flatten_finite_numbers(right["replay_tensors"][key], key)
            _require(left_values.shape == right_values.shape, "tensor replay shape")
            tensor_max = max(
                tensor_max, float(np.max(np.abs(left_values - right_values), initial=0.0))
            )
        for key in scalar_keys:
            left_value = float(left["replay_scalars"][key])
            right_value = float(right["replay_scalars"][key])
            _require(math.isfinite(left_value) and math.isfinite(right_value), "scalar replay")
            residual = abs(left_value - right_value) / max(abs(left_value), abs(right_value), 1e-12)
            scalar_max = max(scalar_max, residual)
    return tensor_max, scalar_max


def compare_integrity_records(
    a: Mapping[str, Any], b: Mapping[str, Any], scientific: Mapping[str, Any]
) -> None:
    """Reject any input, context-0, tensor, or scalar replay mismatch."""

    for payload, role in zip((a, b, scientific), PROCESS_ROLES, strict=True):
        _validate_process_output(payload, role)
    records = [payload["process_record"] for payload in (a, b, scientific)]
    environment_keys = (
        "accelerator",
        "python_version",
        "torch_version",
        "cuda_version",
        "cudnn_version",
        "visible_cuda_devices",
        "deterministic_settings",
    )
    _require(
        all(
            all(record[key] == records[0][key] for key in environment_keys)
            for record in records[1:]
        ),
        "environment replay mismatch",
    )
    rng_keys = (
        "initial_python_rng_sha256",
        "initial_numpy_rng_sha256",
        "initial_torch_cpu_rng_sha256",
        "initial_torch_cuda_rng_sha256_by_device",
    )
    _require(
        all(all(record[key] == records[0][key] for key in rng_keys) for record in records[1:]),
        "RNG replay mismatch",
    )
    _require(
        records[0]["input_context_digest_records"]
        == records[1]["input_context_digest_records"]
        == records[2]["input_context_digest_records"],
        "input context replay mismatch",
    )
    _require(
        records[0]["context0_record_sha256"]
        == records[1]["context0_record_sha256"]
        == records[2]["context0_record_sha256"],
        "context-0 replay mismatch",
    )
    tensor_max, scalar_max = _replay_maxima(a, b, scientific)
    _require(tensor_max <= 2e-6, "tensor replay tolerance exceeded")
    _require(scalar_max <= 1e-5, "scalar replay tolerance exceeded")


def _validate_controller_binding(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    source_manifest, constants = _validate_source_binding(args)
    preregistration_path = Path(args.activated_preregistration)
    source_manifest_path = Path(args.source_manifest)
    persisted_preregistration = _read_json_object(preregistration_path, "activated preregistration")
    persisted_manifest = _read_json_object(source_manifest_path, "source manifest")
    _validate_activated_preregistration(persisted_preregistration)
    _validate_source_manifest_artifact(persisted_manifest)
    source_manifest["activated_preregistration_sha256"] = hashlib.sha256(
        preregistration_path.read_bytes()
    ).hexdigest()
    _require(source_manifest == persisted_manifest, "activated source binding replay mismatch")
    expected_source = {
        key: value
        for key, value in source_manifest.items()
        if key not in {"schema_version", "status", "activated_preregistration_sha256"}
    }
    _require(persisted_preregistration["source"] == expected_source, "activated source mismatch")
    _require(persisted_preregistration["constants"] == constants, "activated constants mismatch")
    return persisted_manifest, constants


def _spawn_process_role(
    role: str,
    *,
    source_manifest_path: Path,
    output_path: Path,
    environment: Mapping[str, str],
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--process-role",
        role,
        "--source-manifest",
        str(source_manifest_path),
        "--process-output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"{role} failed with exit {completed.returncode}: {completed.stderr.strip()}"
        )
    payload = _read_json_object(output_path, f"{role} output")
    _validate_process_output(payload, role)
    return payload


def _result_source_from_manifest(
    source_manifest: Mapping[str, Any], process_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    last_record = process_records[-1]
    result = {
        "prelaunch_source_manifest_path": source_manifest["prelaunch_source_manifest_path"],
        "prelaunch_source_manifest_sha256": source_manifest["prelaunch_source_manifest_sha256"],
        "source_report_path": source_manifest["source_report_path"],
        "source_report_sha256": source_manifest["source_report_sha256"],
        "source_revision": source_manifest["source_revision"],
        "checkpoint_path": source_manifest["checkpoint_path"],
        "checkpoint_sha256": source_manifest["checkpoint_sha256"],
        "checkpoint_bytes": source_manifest["checkpoint_bytes"],
        "checkpoint_epoch": source_manifest["checkpoint_epoch"],
        "resolved_config_path": source_manifest["resolved_config_path"],
        "resolved_config_sha256": source_manifest["resolved_config_sha256"],
        "train_manifest_path": source_manifest["train_manifest_path"],
        "train_manifest_sha256": source_manifest["train_manifest_sha256"],
        "diagnostic_source_sha256": source_manifest["diagnostic_source_sha256"],
        "activated_preregistration_sha256": source_manifest["activated_preregistration_sha256"],
        "python_version": last_record["python_version"],
        "torch_version": source_manifest["torch_version"],
        "numpy_version": source_manifest["numpy_version"],
        "cuda_version": last_record["cuda_version"],
        "cudnn_version": last_record["cudnn_version"],
    }
    if source_manifest["schema_version"] == "pass201-source-v2":
        result["activation_repair"] = copy.deepcopy(source_manifest["activation_repair"])
    return result


def _reduced_replay_failure(
    source_manifest: Mapping[str, Any],
    constants: Mapping[str, Any],
    process_payloads: Sequence[Mapping[str, Any]],
    error: Exception,
) -> dict[str, Any]:
    records = [dict(payload["process_record"]) for payload in process_payloads]
    _require(records, "replay failure requires a launched process record")
    stage = PROCESS_ROLES[len(records) - 1]
    reason = (
        "INVALID_NONDETERMINISTIC_TRAIN_INPUT"
        if "input context" in str(error)
        else "INVALID_NONDETERMINISTIC_OPERATOR_REPLAY"
    )
    integrity = {
        "stage": stage,
        "accepted_context_count": min(record["prepared_context_count"] for record in records),
        "rejected_context_count": 0,
        "invalid_context_count": 1,
        "input_replay_verified": False,
        "deterministic_process_verified": False,
        "process_records": records,
        "failure_evidence_sha256": "",
        "all_finite": False,
    }
    integrity["failure_evidence_sha256"] = _failure_evidence_digest("INVALID", [reason], integrity)
    payload = {
        "schema_version": "pass201-cis-operator-v1",
        "status": "INVALID",
        "reason_codes": [reason],
        "candidate_values_computed": False,
        "uses_test_data": "artifact_binding_only",
        "source": _result_source_from_manifest(source_manifest, records),
        "constants": dict(constants),
        "decision": {
            "thresholds": dict(THRESHOLDS),
            "overall": "INVALID",
            "authorized_next_action": "none",
        },
        "integrity": integrity,
    }
    validate_payload_structure(payload)
    return payload


def _finalize_scientific_payload(
    source_manifest: Mapping[str, Any],
    constants: Mapping[str, Any],
    processes: Sequence[Mapping[str, Any]],
    tensor_max: float,
    scalar_max: float,
) -> dict[str, Any]:
    core = processes[2]["result"]
    _exact_keys(
        core,
        {
            "schema_version",
            "contexts",
            "aggregates",
            "parameter_hash_before",
            "parameter_hash_after",
            "buffer_hash_before",
            "buffer_hash_after",
            "training_flags_restored",
            "rejected_context_count",
            "all_finite",
        },
        "scientific core",
    )
    _require(core["schema_version"] == "pass201-scientific-core-v1", "scientific core")
    decisions = _component_decisions(core["aggregates"])
    reasons = _failure_reasons(core["aggregates"], decisions)
    if reasons or "FAIL" in decisions.values():
        status = "FAIL"
    elif all(value == "PASS" for value in decisions.values()):
        status = "PASS"
    else:
        status = "UNRESOLVED"
    records = [process["process_record"] for process in processes]
    payload = {
        "schema_version": "pass201-cis-operator-v1",
        "status": status,
        "reason_codes": reasons,
        "candidate_values_computed": True,
        "uses_test_data": "artifact_binding_only",
        "source": _result_source_from_manifest(source_manifest, records),
        "constants": dict(constants),
        "contexts": core["contexts"],
        "aggregates": core["aggregates"],
        "decision": {
            "thresholds": dict(THRESHOLDS),
            "component_decisions": decisions,
            "overall": status,
            "authorized_next_action": (
                "write_separate_gpu_preregistration" if status == "PASS" else "none"
            ),
        },
        "integrity": {
            "accepted_context_count": 32,
            "rejected_context_count": core["rejected_context_count"],
            "invalid_context_count": 0,
            "input_replay_verified": True,
            "parameter_hash_before": core["parameter_hash_before"],
            "parameter_hash_after": core["parameter_hash_after"],
            "buffer_hash_before": core["buffer_hash_before"],
            "buffer_hash_after": core["buffer_hash_after"],
            "training_flags_restored": core["training_flags_restored"],
            "deterministic_process_verified": True,
            "first_context_operator_replay_verified": True,
            "deterministic_settings": dict(records[0]["deterministic_settings"]),
            "process_records": records,
            "replay_residuals": {
                "pair_count": 3,
                "tensor_max_absolute": float(tensor_max),
                "scalar_max_relative": float(scalar_max),
                "tensor_tolerance": 2e-6,
                "scalar_tolerance": 1e-5,
                "scalar_denominator": "max(abs(a),abs(b),1e-12)",
            },
            "all_finite": core["all_finite"],
        },
    }
    validate_payload_structure(payload)
    return payload


def run_controller(args: Any) -> None:
    """Validate bindings, then launch exactly A, B, and scientific in order."""

    source_manifest, constants = _validate_controller_binding(args)
    if bool(getattr(args, "binding_only", False)):
        return
    mode = "scientific" if bool(getattr(args, "scientific", False)) else "smoke"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _require(
        output_path.parent.is_dir() and not output_path.parent.is_symlink(),
        "controller output parent must be a real directory",
    )
    if output_path.exists() or output_path.is_symlink():
        raise FileExistsError(output_path)
    if any(output_path.parent.glob(f".{output_path.name}.tmp-*")):
        raise ValueError("controller output temporary path already exists")
    child_environment = dict(os.environ)
    child_environment.pop("PASS201_RUNTIME_FACTORY", None)
    child_environment.update(CPU_THREAD_ENVIRONMENT)
    child_environment["PASS201_PROCESS_MODE"] = mode
    child_environment["PASS201_SOURCE_ROOT"] = str(Path(args.root).resolve())
    child_environment["PASS201_BINDING_AUTHORIZED_SHA256"] = _sha256_file(
        Path(args.source_manifest)
    )
    runtime_factory = getattr(args, "runtime_factory", None)
    if runtime_factory is not None:
        _require(
            getattr(args, "expected_prelaunch_sha256", PRELAUNCH_SOURCE_MANIFEST_SHA256)
            != PRELAUNCH_SOURCE_MANIFEST_SHA256,
            "external runtime factories are forbidden for the frozen source",
        )
        child_environment["PASS201_RUNTIME_FACTORY"] = str(Path(runtime_factory).resolve())
    processes = []
    try:
        with tempfile.TemporaryDirectory(
            prefix=".pass201-process-", dir=output_path.parent
        ) as name:
            temporary_root = Path(name)
            for role in PROCESS_ROLES[:2]:
                child_environment["PASS201_PROCESS_ROLE"] = role
                process = _spawn_process_role(
                    role,
                    source_manifest_path=Path(args.source_manifest),
                    output_path=temporary_root / f"{role}.json",
                    environment=child_environment,
                )
                processes.append(process)
                if len(processes) == 2:
                    _require(
                        processes[0]["process_record"]["input_context_digest_records"]
                        == processes[1]["process_record"]["input_context_digest_records"],
                        "input context replay mismatch before scientific",
                    )
                    _require(
                        processes[0]["process_record"]["context0_record_sha256"]
                        == processes[1]["process_record"]["context0_record_sha256"],
                        "context-0 replay mismatch before scientific",
                    )
            child_environment["PASS201_REPLAY_AUTHORIZED_SHA256"] = hashlib.sha256(
                canonical_json_bytes([process["process_record"] for process in processes])
            ).hexdigest()
            child_environment["PASS201_PROCESS_ROLE"] = "scientific"
            scientific = _spawn_process_role(
                "scientific",
                source_manifest_path=Path(args.source_manifest),
                output_path=temporary_root / "scientific.json",
                environment=child_environment,
            )
            processes.append(scientific)
            compare_integrity_records(*processes)
    except (RuntimeError, ValueError) as error:
        if processes:
            invalid = _reduced_replay_failure(source_manifest, constants, processes, error)
            _publish_json_no_replace(
                output_path,
                invalid,
                sort_keys=source_manifest["schema_version"] != "pass201-source-v2",
            )
            return
        raise
    tensor_max, scalar_max = _replay_maxima(*processes)
    if mode == "smoke":
        payload = {
            "schema_version": "pass201-controller-smoke-v1",
            "source_manifest_sha256": _sha256_file(Path(args.source_manifest)),
            "processes": processes,
            "replay_residuals": {
                "pair_count": 3,
                "tensor_max_absolute": tensor_max,
                "scalar_max_relative": scalar_max,
                "tensor_tolerance": 2e-6,
                "scalar_tolerance": 1e-5,
                "scalar_denominator": "max(abs(a),abs(b),1e-12)",
            },
        }
    else:
        payload = _finalize_scientific_payload(
            source_manifest,
            constants,
            processes,
            tensor_max,
            scalar_max,
        )
    _publish_json_no_replace(
        output_path,
        payload,
        sort_keys=source_manifest["schema_version"] != "pass201-source-v2",
    )


def _build_source_v5_authority(
    *,
    historical_manifest: Mapping[str, Any],
    historical_manifest_bytes: bytes,
    source_commit: str,
    source_files: Sequence[Mapping[str, Any]],
    frozen_absence_checked_utc: str,
) -> dict[str, Any]:
    run_dir = "reports/generated/pass201_source_v3/run-v3"
    artifacts = {
        "checkpoint": (
            "checkpoint.pt",
            55760186,
            "e42d25b4e8e98f1d619aada2215ecbfeca579327dabaf6f09f02151183220696",
        ),
        "log": (
            "training.log",
            8757,
            "053a7dc0b447f6bfeabf7dac347d80b0b889e94db7688eeebaf94e02f5f4d1d2",
        ),
        "receipt": (
            "receipt.json",
            15179,
            "a494ead85167539670f8c5d1481f8d9eabc274776727df06d7362e99e9b7cdf9",
        ),
        "report": (
            "report.json",
            250481,
            "a74a2a1c08beee2a0b4d67adcf378421309077c3ce39c93711f782f0589cc843",
        ),
        "resolved_config": (
            "resolved_config.json",
            5981,
            "bd5d1f0216b8c55d70a7ac4bd528fb5545d66472e964abd321bba3877079584e",
        ),
        "train_manifest": (
            "train_manifest.json",
            2895656,
            "c60e998e802f2b1050fc3c934ce9960c27c7ec4e598b0d88acd8062295c3def9",
        ),
    }

    def completed(name: str, *, execution: bool = True) -> dict[str, Any]:
        filename, size, digest = artifacts[name]
        item: dict[str, Any] = {
            "path": f"{run_dir}/{filename}",
            "bytes": size,
            "sha256": digest,
        }
        if execution:
            item["required_present_at_execution"] = True
        return item

    def future(path: str) -> dict[str, Any]:
        return {"path": path, "required_absent_when_frozen": True}

    source = copy.deepcopy(historical_manifest["source"])
    source["files"] = [copy.deepcopy(dict(row)) for row in source_files]
    retained = {
        key: copy.deepcopy(historical_manifest[key])
        for key in (
            "controller",
            "dataset",
            "execution",
            "plan",
            "postconditions",
            "process_entry_amendment",
            "process_entry_evidence",
            "process_entry_plan",
            "protocol",
            "sidecars",
            "status",
        )
    }
    return {
        "authorization": {
            "clean_policy": "empty-porcelain-v1-z",
            "frozen_absence": {
                "activated_preregistration": "ENOENT",
                "result": "ENOENT",
                "smoke": "ENOENT",
                "source_manifest": "ENOENT",
            },
            "frozen_absence_checked_utc": frozen_absence_checked_utc,
            "manifest_path": SOURCE_V5_AUTHORIZATION_MANIFEST_PATH,
            "required_diff_modes": ["100644"],
            "required_diff_paths": [SOURCE_V5_AUTHORIZATION_MANIFEST_PATH],
            "required_diff_status": ["A"],
            "required_parent_commit": source_commit,
        },
        "controller": retained["controller"],
        "dataset": retained["dataset"],
        "execution": retained["execution"],
        "historical_producer": {
            "authorization_commit": "32c4d39322fca2a5a906f785bdb612dcd7008647",
            "source_commit": "53a9db9e9dbe54fcebb33769b915c3f33699d522",
            "manifest": {
                "path": SOURCE_V4_AUTHORIZATION_MANIFEST_PATH,
                "bytes": len(historical_manifest_bytes),
                "sha256": hashlib.sha256(historical_manifest_bytes).hexdigest(),
                "git_blob": "430f340a17cc32c5fd239083b1a0dba98e09ad7c",
            },
            "receipt": {
                **completed("receipt", execution=False),
                "schema_version": "pass201-pa-source-v4-receipt-v1",
                "candidate_values_computed": False,
            },
            "outputs": {
                name: completed(name, execution=False)
                for name in ("checkpoint", "log", "report", "resolved_config", "train_manifest")
            },
        },
        "outputs": {
            "activated_preregistration": future(ACTIVATED_PREREGISTRATION_PATH),
            "checkpoint": completed("checkpoint"),
            "log": completed("log"),
            "receipt": completed("receipt"),
            "report": completed("report"),
            "resolved_config": completed("resolved_config"),
            "result": future(RESULT_PATH),
            "run_directory": {"path": run_dir, "required_present_at_execution": True},
            "smoke": future(SMOKE_RESULT_PATH),
            "source_manifest": future(SOURCE_MANIFEST_PATH),
            "train_manifest": completed("train_manifest"),
        },
        "plan": retained["plan"],
        "postconditions": retained["postconditions"],
        "process_entry_amendment": retained["process_entry_amendment"],
        "process_entry_evidence": retained["process_entry_evidence"],
        "process_entry_plan": retained["process_entry_plan"],
        "protocol": retained["protocol"],
        "purpose": "activate_completed_source_v4_then_run_cpu_diagnostic",
        "repair_amendment": {
            "path": "docs/pass201_pa_source_v4_dispatch_repair_amendment_2026-08-11.md",
            "sha256": "a265f7d2a5f55c52eb63263909aa387a55c74820b9644bf92a3cadda965da716",
            "commit": "8d7938a4e66abad0a8351422fcfa6f741ea76b00",
        },
        "repair_plan": {
            "path": "docs/superpowers/plans/2026-08-11-pass201-pa-source-v4-dispatch-repair.md",
            "sha256": "470f3e08ad138cb48b4802f251b63b5c4387b28f3fd023e8492b0fd98950732e",
            "commit": "20785ba6be243a9b7e95fcb25647ee3cdc55cc9e",
        },
        "schema_version": "pass201-pa-source-v5-activation-v1",
        "sidecars": retained["sidecars"],
        "source": source,
        "source_commit": source_commit,
        "status": retained["status"],
    }


def _source_v5_current_rows(
    root: Path, source_commit: str, paths: Sequence[str]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for relative in paths:
        committed = _git_command_bytes(root, "show", f"{source_commit}:{relative}")
        worktree_path = root / relative
        _require(
            worktree_path.is_file() and not worktree_path.is_symlink(),
            "source-v5 worktree file unavailable",
        )
        worktree = worktree_path.read_bytes()
        _require(worktree == committed, f"source-v5 worktree bytes differ: {relative}")
        tree = _git_command_bytes(root, "ls-tree", source_commit, "--", relative).decode("utf-8")
        _require(
            tree.startswith("100644 blob ") and tree.endswith(f"\t{relative}\n"),
            f"source-v5 source mode differs: {relative}",
        )
        rows.append(
            {
                "bytes": len(committed),
                "git_blob": tree.split()[2],
                "git_mode": "100644",
                "path": relative,
                "sha256": hashlib.sha256(committed).hexdigest(),
            }
        )
    return rows


def _authenticate_source_v5_freezer_root(root: Path) -> tuple[str, bytes]:
    _require(
        Path(__file__).resolve(strict=True).parents[1] == root,
        "source-v5 root differs from executing checkout",
    )
    symbolic = subprocess.run(
        ["git", "symbolic-ref", "-q", "HEAD"],
        cwd=root,
        capture_output=True,
    )
    _require(symbolic.returncode == 1, "source-v5 freezer requires detached HEAD")
    _require(
        _git_command_bytes(root, "status", "--porcelain", "--untracked-files=no", "-z") == b"",
        "source-v5 freezer requires tracked-clean checkout",
    )
    head = _git_command_bytes(root, "rev-parse", "HEAD").decode("ascii").strip()
    _require(
        _git_command_bytes(root, "rev-parse", SOURCE_V4_HISTORICAL_TAG).decode("ascii").strip()
        == SOURCE_V4_HISTORICAL_HANDOFF_COMMIT,
        "source-v4 preservation tag differs",
    )
    parents = (
        _git_command_bytes(
            root,
            "rev-list",
            "--parents",
            "-n",
            "1",
            SOURCE_V4_HISTORICAL_HANDOFF_COMMIT,
        )
        .decode("ascii")
        .strip()
        .split()
    )
    _require(
        parents == [SOURCE_V4_HISTORICAL_HANDOFF_COMMIT, SOURCE_V4_HISTORICAL_SOURCE_COMMIT],
        "source-v4 historical parent differs",
    )
    h4_path = SOURCE_V4_AUTHORIZATION_MANIFEST_PATH
    _require(
        _git_command_bytes(
            root,
            "diff-tree",
            "--no-commit-id",
            "--name-status",
            "-r",
            "-z",
            SOURCE_V4_HISTORICAL_HANDOFF_COMMIT,
        ).split(b"\0")
        == [b"A", h4_path.encode("utf-8"), b""],
        "source-v4 historical handoff scope differs",
    )
    h4_bytes = _git_command_bytes(
        root,
        "show",
        f"{SOURCE_V4_HISTORICAL_HANDOFF_COMMIT}:{h4_path}",
    )
    _require(
        hashlib.sha256(h4_bytes).hexdigest()
        == "080adaeaaa5c7bf9c87ed93761d6e4c517b958bb60c49af68a880109f5abce1f",
        "source-v4 historical manifest digest differs",
    )
    _require(
        _git_command_bytes(
            root,
            "rev-parse",
            f"{SOURCE_V4_HISTORICAL_HANDOFF_COMMIT}:{h4_path}",
        )
        .decode("ascii")
        .strip()
        == "430f340a17cc32c5fd239083b1a0dba98e09ad7c",
        "source-v4 historical manifest blob differs",
    )
    _authenticate_source_v5_source_chain(root, head)
    return head, h4_bytes


def freeze_source_v5_authority(
    *, root: Path, output: Path, frozen_absence_checked_utc: str
) -> bytes:
    root = root.resolve(strict=True)
    _require(output.is_absolute(), "source-v5 output must be absolute")
    _require(output == output.resolve(strict=False), "source-v5 output must be normalized")
    try:
        parsed_utc = datetime.strptime(frozen_absence_checked_utc, "%Y-%m-%dT%H:%M:%SZ")
    except (TypeError, ValueError) as error:
        raise ValueError("source-v5 UTC evidence differs") from error
    _require(
        type(frozen_absence_checked_utc) is str
        and parsed_utc.strftime("%Y-%m-%dT%H:%M:%SZ") == frozen_absence_checked_utc,
        "source-v5 UTC evidence differs",
    )
    source_commit, h4_bytes = _authenticate_source_v5_freezer_root(root)
    for relative in (
        ACTIVATED_PREREGISTRATION_PATH,
        SOURCE_MANIFEST_PATH,
        SMOKE_RESULT_PATH,
        RESULT_PATH,
    ):
        _require(
            not (root / relative).exists() and not (root / relative).is_symlink(),
            f"source-v5 frozen output exists: {relative}",
        )
    bootstrap = _bootstrap_strict_json_object(h4_bytes, "source-v4 historical authority")
    paths = [row["path"] for row in bootstrap["source"]["files"]]
    rows = _source_v5_current_rows(root, source_commit, paths)
    contract = _load_authenticated_source_v3_contract(root, source_commit, rows[1])
    historical = contract.load_strict_json_bytes(h4_bytes)
    contract.validate_prelaunch(historical)
    payload = _build_source_v5_authority(
        historical_manifest=historical,
        historical_manifest_bytes=h4_bytes,
        source_commit=source_commit,
        source_files=rows,
        frozen_absence_checked_utc=frozen_absence_checked_utc,
    )
    contract.validate_prelaunch(payload)
    data = contract.canonical_ordered_json_bytes(payload)
    reloaded = contract.load_strict_json_bytes(data)
    contract.validate_prelaunch(reloaded)
    _require(
        reloaded == payload and contract.canonical_ordered_json_bytes(reloaded) == data,
        "source-v5 persisted authority differs",
    )
    _publish_source_v5_candidate(output, data)
    return data


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bound Pass201 CIS operator diagnostic")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--activate-source", action="store_true")
    modes.add_argument("--process-role", choices=PROCESS_ROLES)
    modes.add_argument("--binding-only", action="store_true")
    modes.add_argument("--smoke-only", action="store_true")
    modes.add_argument("--scientific", action="store_true")
    modes.add_argument("--freeze-v5-authority", action="store_true")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--git-root", type=Path)
    parser.add_argument("--prelaunch-manifest", type=Path)
    parser.add_argument("--source-report", type=Path)
    parser.add_argument("--source-receipt", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--resolved-config", type=Path)
    parser.add_argument("--train-manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--diagnostic-path", type=Path)
    parser.add_argument("--activated-preregistration", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--process-output", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-factory", type=Path)
    parser.add_argument("--frozen-absence-checked-utc")
    return parser


def _default_cli_paths(args: Any) -> None:
    root = Path(args.root).resolve(strict=True)
    args.root = root
    defaults = {
        "git_root": root,
        "prelaunch_manifest": root / SOURCE_V5_AUTHORIZATION_MANIFEST_PATH,
        "source_report": root / "reports/generated/pass201_source_v3/run-v3/report.json",
        "source_receipt": root / "reports/generated/pass201_source_v3/run-v3/receipt.json",
        "checkpoint": root / "reports/generated/pass201_source_v3/run-v3/checkpoint.pt",
        "resolved_config": root / "reports/generated/pass201_source_v3/run-v3/resolved_config.json",
        "train_manifest": root / "reports/generated/pass201_source_v3/run-v3/train_manifest.json",
        "activated_preregistration": root / ACTIVATED_PREREGISTRATION_PATH,
        "source_manifest": root / SOURCE_MANIFEST_PATH,
        "diagnostic_path": Path(__file__).resolve(),
        "output": root / (SMOKE_RESULT_PATH if args.smoke_only else RESULT_PATH),
    }
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)


def _validate_source_v3_public_controller_args(args: Any) -> None:
    root = Path(args.root).resolve(strict=True)
    expected = root / SOURCE_V5_AUTHORIZATION_MANIFEST_PATH
    _require(
        args.runtime_factory is None and "PASS201_RUNTIME_FACTORY" not in os.environ,
        "source-v5 runtime factories are forbidden",
    )
    _require(
        Path(args.prelaunch_manifest) == expected,
        "source-v4 authorization is repair-required; public controller requires the "
        "literal source-v5 authorization manifest path",
    )
    executing_root = Path(__file__).resolve(strict=True).parents[1]
    _require(root == executing_root, "public controller root differs from executing checkout")
    expected_paths = {
        "git_root": root,
        "diagnostic_path": root / "scripts/diagnose_pass201_cis_operator.py",
        "activated_preregistration": root / ACTIVATED_PREREGISTRATION_PATH,
        "source_manifest": root / SOURCE_MANIFEST_PATH,
        "source_receipt": root / "reports/generated/pass201_source_v3/run-v3/receipt.json",
        "output": root / (SMOKE_RESULT_PATH if args.smoke_only else RESULT_PATH),
    }
    for name, value in expected_paths.items():
        _require(Path(getattr(args, name)) == value, f"public controller {name} differs")


def _dispatch_source_v5_freezer(args: Any) -> None:
    incompatible = (
        "git_root",
        "prelaunch_manifest",
        "source_report",
        "source_receipt",
        "checkpoint",
        "resolved_config",
        "train_manifest",
        "dataset_root",
        "diagnostic_path",
        "activated_preregistration",
        "source_manifest",
        "process_output",
        "runtime_factory",
    )
    supplied = [name for name in incompatible if getattr(args, name) is not None]
    if supplied:
        raise ValueError(f"source-v5 freezer incompatible argument: {supplied[0]}")
    _require(args.output is not None, "source-v5 freezer requires --output")
    _require(
        args.frozen_absence_checked_utc is not None,
        "source-v5 freezer requires --frozen-absence-checked-utc",
    )
    raw_root = Path(args.root)
    _require(raw_root.is_absolute(), "source-v5 freezer root must be absolute")
    _require(
        raw_root == raw_root.resolve(strict=True),
        "source-v5 freezer root must be normalized",
    )
    freeze_source_v5_authority(
        root=raw_root,
        output=Path(args.output),
        frozen_absence_checked_utc=args.frozen_absence_checked_utc,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_cli_parser().parse_args(argv)
    if args.freeze_v5_authority:
        _dispatch_source_v5_freezer(args)
        return
    _default_cli_paths(args)
    if not args.process_role:
        _validate_source_v3_public_controller_args(args)
    if args.activate_source:
        activate_source(args)
    elif args.process_role:
        _require(args.process_output is not None, "--process-output is required")
        manifest = _read_json_object(args.source_manifest, "source manifest")
        if manifest.get("prelaunch_source_manifest_path") in (
            SOURCE_V3_AUTHORIZATION_MANIFEST_PATH,
            SOURCE_V4_AUTHORIZATION_MANIFEST_PATH,
            SOURCE_V5_AUTHORIZATION_MANIFEST_PATH,
        ):
            _require(
                args.runtime_factory is None and "PASS201_RUNTIME_FACTORY" not in os.environ,
                "source-v3 runtime factories are forbidden",
            )
        _require(
            os.environ.get("PASS201_BINDING_AUTHORIZED_SHA256")
            == _sha256_file(Path(args.source_manifest)),
            "process source manifest differs from the binding-authorized source manifest",
        )
        _validate_source_manifest_artifact(manifest)
        _require(
            _sha256_file(Path(__file__).resolve()) == manifest["diagnostic_source_sha256"],
            "executing diagnostic source mismatch",
        )
        if args.runtime_factory is not None:
            os.environ["PASS201_RUNTIME_FACTORY"] = str(args.runtime_factory.resolve())
        run_process_role(args.process_role, manifest, args.process_output)
    else:
        run_controller(args)


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
    keys = {
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
    }
    has_repair = type(source) is dict and "activation_repair" in source
    if has_repair:
        keys.add("activation_repair")
    _exact_keys(source, keys, "source")
    if has_repair:
        _validate_activation_repair(
            source["activation_repair"],
            prelaunch_path=source["prelaunch_source_manifest_path"],
            prelaunch_sha256=source["prelaunch_source_manifest_sha256"],
            source_revision=source["source_revision"],
            diagnostic_sha256=source["diagnostic_source_sha256"],
        )
    else:
        _require(
            source["prelaunch_source_manifest_path"]
            in (SOURCE_V3_AUTHORIZATION_MANIFEST_PATH, SOURCE_V4_AUTHORIZATION_MANIFEST_PATH),
            "wrong prelaunch source path",
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
    if source["cuda_version"] is not None or source["cudnn_version"] is not None:
        _require(
            source["cuda_version"] == "13.0" and source["cudnn_version"] == "92000",
            "source build version",
        )
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
        type_is_exact = (
            _is_int(constants[key])
            if type(expected_value) is int
            else type(constants[key]) is float
        )
        _require(type_is_exact and constants[key] == expected_value, f"constants.{key}")
    for key in ("learning_rate", "coalition_weight", "proxy_learning_rate_multiplier"):
        if not activated and constants[key] is None:
            continue
        _finite_float(constants[key], f"constants.{key}")


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
        _finite_float(summary[key], f"{path}.{key}")
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
        _finite_float(update[key], f"{path}.{key}")
    if equal_norm:
        _finite_float(update["reference_pa_norm"], f"{path}.reference_pa_norm")
        _finite_float(update["norm_match_absolute_error"], f"{path}.norm_match_absolute_error")
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
    _finite_float(operator["loss"], f"{path}.loss")
    _require(
        _is_int(operator["representative_count"])
        and operator["representative_count"] == representative_count,
        f"{path}.representative_count",
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
            _finite_float(panel[key], f"{panel_path}.{key}")
        if name in {"atomic_one_hot", "atomic_complementary", "atomic_full_union"}:
            _finite_float(
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


def _validate_context(
    context: Any, expected_index: int, prior_contexts: Sequence[Mapping[str, Any]]
) -> None:
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
    _require(
        _is_int(context["context_index"]) and context["context_index"] == expected_index,
        f"{path}.context_index",
    )
    _require(
        _is_int(context["production_epoch"]) and context["production_epoch"] == 0,
        f"{path}.production_epoch",
    )
    _require(
        _is_int(context["production_batch_index"])
        and context["production_batch_index"] >= expected_index,
        f"{path}.production_batch_index",
    )
    if prior_contexts:
        _require(
            context["production_batch_index"] > prior_contexts[-1]["production_batch_index"],
            f"{path}.production_batch_index must be strictly increasing",
        )
    _require(
        _is_int(context["batch_size"]) and context["batch_size"] == 180,
        f"{path}.batch_size",
    )
    _require(_is_int(context["m_unique"]) and context["m_unique"] > 0, f"{path}.m_unique")
    for key in ("row_example_ids", "s_prime_example_ids"):
        _require(
            isinstance(context[key], list)
            and len(context[key]) == 180
            and all(isinstance(value, str) and value for value in context[key]),
            f"{path}.{key}",
        )
    for key in ("row_sample_indices", "row_labels", "s_prime_sample_indices"):
        _require(
            isinstance(context[key], list)
            and len(context[key]) == 180
            and all(_is_int(value) for value in context[key]),
            f"{path}.{key}",
        )
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
        all(_is_int(count) and count > 0 for count in context["class_multiplicities"].values())
        and len(actual_multiplicities) == len(context["class_multiplicities"])
        and actual_multiplicities == expected_multiplicities,
        f"{path}.class_multiplicities",
    )
    _require(
        isinstance(context["representative_row_indices"], list)
        and len(context["representative_row_indices"]) == context["m_unique"]
        and all(_is_int(value) for value in context["representative_row_indices"]),
        f"{path}.representative_row_indices",
    )
    _require(
        isinstance(context["representative_sample_indices"], list)
        and len(context["representative_sample_indices"]) == context["m_unique"]
        and all(_is_int(value) for value in context["representative_sample_indices"]),
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
    expected_reuse = _cross_context_reuse(
        set(context["row_example_ids"]),
        set(context["s_prime_example_ids"]),
        set(context["row_labels"]),
        prior_contexts,
    )
    _require(
        context["cross_context_reuse"] == expected_reuse,
        f"{path}.cross_context_reuse does not match causal prefix",
    )
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
        _finite_float(shared[key], f"{path}.shared_confuser.{key}")
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
    _require(
        _is_int(bootstrap["seed"]) and bootstrap["seed"] == BOOTSTRAP_SEED,
        "aggregates.bootstrap.seed",
    )
    _require(
        _is_int(bootstrap["replicates"]) and bootstrap["replicates"] == BOOTSTRAP_REPLICATES,
        "aggregates.bootstrap.replicates",
    )
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
    "deterministic_settings",
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
    _require(
        _is_int(record["context_index"]) and record["context_index"] == expected_index,
        f"{path}.context_index",
    )
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
    _require(record["accelerator"] == "cpu", f"{path}.accelerator")
    _require(
        record["cuda_version"] == "13.0" and record["cudnn_version"] == "92000",
        f"{path}.build version",
    )
    _require(
        record["visible_cuda_devices"] == ["cpu"],
        f"{path}.visible_cuda_devices",
    )
    for key in (
        "initial_python_rng_sha256",
        "initial_numpy_rng_sha256",
        "initial_torch_cpu_rng_sha256",
    ):
        _digest(record[key], f"{path}.{key}")
    cuda_hashes = record["initial_torch_cuda_rng_sha256_by_device"]
    _require(
        isinstance(cuda_hashes, dict) and list(cuda_hashes) == ["0"],
        f"{path}.initial_torch_cuda_rng_sha256_by_device",
    )
    _require(
        cuda_hashes["0"] == record["initial_torch_cpu_rng_sha256"],
        f"{path}.initial_torch_cuda_rng_sha256_by_device CPU placeholder",
    )
    for key, value in cuda_hashes.items():
        _require(str(int(key)) == key, f"{path}.CUDA device index")
        _digest(value, f"{path}.initial_torch_cuda_rng_sha256_by_device.{key}")
    expected_settings = {
        "cublas_workspace_config": ":4096:8",
        "deterministic_algorithms": True,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "autocast": False,
        "dtype": "float32",
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    _exact_keys(
        record["deterministic_settings"],
        expected_settings,
        f"{path}.deterministic_settings",
    )
    _require(
        all(
            type(record["deterministic_settings"][key]) is type(expected)
            and record["deterministic_settings"][key] == expected
            for key, expected in expected_settings.items()
        ),
        f"{path}.deterministic_settings",
    )
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
    _require(
        _is_int(integrity["accepted_context_count"]) and integrity["accepted_context_count"] == 32,
        "integrity.accepted_context_count",
    )
    _require(
        _is_int(integrity["rejected_context_count"]) and integrity["rejected_context_count"] >= 0,
        "integrity.rejected_context_count",
    )
    _require(
        _is_int(integrity["invalid_context_count"]) and integrity["invalid_context_count"] == 0,
        "integrity.invalid_context_count",
    )
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
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
    }
    _exact_keys(settings, expected_settings, "integrity.deterministic_settings")
    _require(
        all(
            type(settings[key]) is type(expected_value) and settings[key] == expected_value
            for key, expected_value in expected_settings.items()
        ),
        "deterministic settings mismatch",
    )
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
    _require(
        _is_int(residuals["pair_count"]) and residuals["pair_count"] == 3,
        "integrity.replay_residuals.pair_count",
    )
    for key in (
        "tensor_max_absolute",
        "scalar_max_relative",
        "tensor_tolerance",
        "scalar_tolerance",
    ):
        _finite_float(residuals[key], f"integrity.replay_residuals.{key}")
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
    _require(
        _is_int(context["context_index"]) and context["context_index"] == expected_index,
        f"{path}.context_index",
    )
    _require(
        _is_int(context["production_epoch"]) and context["production_epoch"] == 0,
        f"{path}.production_epoch",
    )
    _require(
        _is_int(context["production_batch_index"])
        and context["production_batch_index"] == expected_index,
        f"{path}.production_batch_index",
    )
    _require(
        isinstance(context["row_example_ids"], list)
        and len(context["row_example_ids"]) == 180
        and all(isinstance(value, str) and value for value in context["row_example_ids"]),
        f"{path}.row_example_ids",
    )
    for key in ("row_sample_indices", "row_labels"):
        _require(
            isinstance(context[key], list)
            and len(context[key]) == 180
            and all(_is_int(value) for value in context[key]),
            f"{path}.{key}",
        )
    expected_multiplicities = {
        str(label): count for label, count in Counter(context["row_labels"]).items()
    }
    actual_multiplicities = {
        str(label): count for label, count in context["class_multiplicities"].items()
    }
    _require(
        all(_is_int(count) and count > 0 for count in context["class_multiplicities"].values())
        and len(actual_multiplicities) == len(context["class_multiplicities"])
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
        isinstance(context["representative_row_indices"], list)
        and all(_is_int(value) for value in context["representative_row_indices"])
        and context["representative_row_indices"] == expected_rows,
        f"{path}.representative_row_indices",
    )
    _require(
        isinstance(context["representative_sample_indices"], list)
        and all(_is_int(value) for value in context["representative_sample_indices"])
        and context["representative_sample_indices"] == expected_samples,
        f"{path}.representative_sample_indices",
    )
    _require(context["status"] in {"accepted", "rejected"}, f"{path}.status")
    if context["status"] == "accepted":
        _require(context["rejection_code"] is None, f"{path}.rejection_code")
        _require(
            isinstance(context["s_prime_example_ids"], list)
            and len(context["s_prime_example_ids"]) == 180
            and all(isinstance(value, str) and value for value in context["s_prime_example_ids"])
            and len(set(context["s_prime_example_ids"])) == 180
            and set(context["row_example_ids"]).isdisjoint(context["s_prime_example_ids"]),
            f"{path}.s_prime_example_ids",
        )
        _require(
            isinstance(context["s_prime_sample_indices"], list)
            and len(context["s_prime_sample_indices"]) == 180
            and all(_is_int(value) for value in context["s_prime_sample_indices"])
            and len(set(context["s_prime_sample_indices"])) == 180,
            f"{path}.s_prime_sample_indices",
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
        "network_equal_union_predicted_suppression",
        "network_equal_union_predicted_margin_change",
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
    _require(
        all(type(value) is float for value in decision["thresholds"].values())
        and decision["thresholds"] == THRESHOLDS,
        "decision.thresholds mismatch",
    )


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
        payload["candidate_values_computed"] is True
        or payload["candidate_values_computed"] is False,
        "candidate_values_computed must be a literal boolean",
    )
    if not scored:
        _require(
            payload["candidate_values_computed"] is False,
            "candidate_values_computed must be literal false",
        )
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
            _validate_context(context, index, contexts[:index])
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
        if status == "BLOCKED":
            _require(
                payload["reason_codes"] == ["BLOCKED_SOURCE_ARTIFACT_UNAVAILABLE"],
                "BLOCKED reason_codes",
            )
        else:
            invalid_codes = {
                "INVALID_OPERATING_POINT_MISMATCH",
                "INVALID_NONDETERMINISTIC_TRAIN_INPUT",
                "INVALID_NONDETERMINISTIC_OPERATOR_REPLAY",
            }
            _require(
                bool(payload["reason_codes"])
                and payload["reason_codes"] == sorted(payload["reason_codes"])
                and set(payload["reason_codes"]) <= invalid_codes,
                "INVALID reason_codes",
            )
            if payload["integrity"].get("stage") == "source_activation":
                _require(
                    payload["reason_codes"] == ["INVALID_OPERATING_POINT_MISMATCH"],
                    "source_activation INVALID reason_codes",
                )
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


if __name__ == "__main__":
    main()
