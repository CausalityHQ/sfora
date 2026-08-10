"""Synthetic, outcome-blind tests for the immutable RSTA artifact verifier."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import subprocess
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest

_REPOSITORY = Path(__file__).resolve().parents[1]
_SCRIPT = _REPOSITORY / "scripts" / "verify_pass200_rsta_scientific_artifact.py"
_NAMESPACE: dict[str, object] = {}
exec(compile(_SCRIPT.read_bytes(), str(_SCRIPT), "exec"), _NAMESPACE)


def _call(function_name: str, *args: object, **kwargs: object) -> object:
    return _NAMESPACE[function_name](*args, **kwargs)


def _exception(name: str) -> type[Exception]:
    value = _NAMESPACE[name]
    assert isinstance(value, type) and issubclass(value, Exception)
    return value


def _raw_payload() -> dict[str, object]:
    return {
        "manifest": {"path": "manifest.json", "source": {"revision": "old"}},
        "execution_audit": {"bound": True},
        "environment": {"numpy_version": "2.1.3", "signed_zero": -0.0},
        "seed_audits": [{"seed": 0}],
        "rows": {"primary": [{"label": 0}], "alternate": [{"label": 7}]},
        "integrity": {"all_passed": True},
        "aggregation": {"count": 1},
        "bootstrap": {"replicates": 10_000},
        "panel_binding": {
            "primary": {
                "eligible_labels": [0, 7, 42],
                "support_ids_by_label": {
                    "0": ["support-zero"],
                    "7": ["support-seven"],
                    "42": ["support-forty-two"],
                },
            }
        },
    }


def _encoded(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=False, allow_nan=False) + "\n").encode()


def _legacy_module(raw: dict[str, object], calls: list[dict[str, object]]) -> types.ModuleType:
    module = types.ModuleType("authenticated_legacy")
    module.np = sys.modules.get("numpy")

    def scientific_payload(**kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        result = deepcopy(raw)
        adapted_support = kwargs["panel_binding"]["primary"]["support_ids_by_label"]
        result["panel_binding"]["primary"]["support_ids_by_label"] = {
            str(key): value for key, value in adapted_support.items()
        }
        return result

    module.scientific_payload = scientific_payload
    return module


def _receipt(*, status: str = "VALID") -> dict[str, object]:
    exit_code = 0 if status == "VALID" else 1
    return {
        "schema_version": 1,
        "validation": "pass200-rsta-scientific-artifact-roundtrip",
        "mode": "offline_immutable_artifact",
        "attempt": 1,
        "status": status,
        "outcome_disclosed": False,
        "artifact": {
            "path": (
                "reports/generated/pass200_rsta_receipt/"
                "c04574e2bb751c3229bce673408577cfedc00a88-stage-a.json"
            ),
            "sha256": "e9bcd77c6e372e9c3bab4a420b97ff56f8ea164cbca56f53ec9c99a3b3c527ae",
            "producer_pid": 1002393,
            "producer_exit_code": 0,
            "immutable": True,
        },
        "legacy_provenance": {
            "handoff_commit": "c04574e2bb751c3229bce673408577cfedc00a88",
            "source_commit": "15234a529a181c39c1c8b6477ad7eb7823fd0798",
            "manifest_path": "docs/pass200_rsta_receipt_stage_a_manifest.json",
            "manifest_sha256": "9260329a0f9ad45257f51292d40c3a6d70c9494ea3e8fd185afcf8484f9378fe",
            "diagnostic_path": "scripts/diagnose_pass200_rsta_stage_a.py",
            "diagnostic_sha256": "85958a940c5a4c9f0ae27f3342e436a8a37e49d94fe9515b22db0340d597ef6e",
        },
        "verifier_provenance": {
            "source_commit": "a" * 40,
            "handoff_commit": "b" * 40,
            "manifest_path": "docs/pass200_rsta_receipt_stage_a_manifest.json",
            "manifest_sha256": "c" * 64,
            "verifier_path": "scripts/verify_pass200_rsta_scientific_artifact.py",
            "verifier_sha256": "d" * 64,
            "amendment": {
                "path": (
                    "docs/pass200_rsta_scientific_artifact_roundtrip_"
                    "recovery_amendment_2026-08-10.md"
                ),
                "sha256": "6e1767e802295fcfbf29e7151ac05991a016994ca92b99bf2e2cbcd46e4e9591",
                "commit": "043121f8a414b91d7fb2e3d6a1635a6bd585676a",
            },
        },
        "process": {
            "parent_pid": 123,
            "child_pid": 456,
            "child_exit_code": exit_code,
            "python_executable": ".venv/bin/python",
            "python_version": "3.12.3",
            "numpy_version": "2.1.3",
            "isolated": True,
            "child_head_commit": "c04574e2bb751c3229bce673408577cfedc00a88",
            "cuda_visible_devices": "",
        },
    }


@pytest.mark.parametrize(
    "payload",
    (b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}', b"[]", b"null"),
)
def test_strict_json_object_rejects_duplicate_nonfinite_and_nonobject(payload: bytes) -> None:
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call("strict_json_object", payload, name="synthetic")


@pytest.mark.parametrize(
    "mutation",
    ("integer", "boolean", "00", "+7", "042", "missing", "extra", "reordered", "collision"),
)
def test_legacy_adapter_requires_canonical_ordered_string_keys(mutation: str) -> None:
    raw = _raw_payload()
    support = raw["panel_binding"]["primary"]["support_ids_by_label"]
    if mutation == "integer":
        raw["panel_binding"]["primary"]["support_ids_by_label"] = {
            int(key): value for key, value in support.items()
        }
    elif mutation == "boolean":
        raw["panel_binding"]["primary"]["support_ids_by_label"] = {
            True: support["0"],
            "7": support["7"],
            "42": support["42"],
        }
    elif mutation in {"00", "+7", "042"}:
        key = {"00": "0", "+7": "7", "042": "42"}[mutation]
        raw["panel_binding"]["primary"]["support_ids_by_label"] = {
            (mutation if observed == key else observed): value
            for observed, value in support.items()
        }
    elif mutation == "missing":
        support.pop("7")
    elif mutation == "extra":
        support["99"] = ["extra"]
    elif mutation == "reordered":
        raw["panel_binding"]["primary"]["support_ids_by_label"] = dict(reversed(support.items()))
    else:
        support[7] = support["7"]
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call("adapt_legacy_support_keys", raw)


def test_legacy_adapter_changes_only_support_key_types() -> None:
    raw = _raw_payload()
    before = deepcopy(raw)
    adapted, ledger = _call("adapt_legacy_support_keys", raw)
    assert raw == before
    assert ledger == (
        ("panel_binding", "primary", "support_ids_by_label", "0", "str", "int"),
        ("panel_binding", "primary", "support_ids_by_label", "7", "str", "int"),
        ("panel_binding", "primary", "support_ids_by_label", "42", "str", "int"),
    )
    assert list(adapted["panel_binding"]["primary"]["support_ids_by_label"]) == [0, 7, 42]
    restored = deepcopy(adapted)
    restored["panel_binding"]["primary"]["support_ids_by_label"] = {
        str(key): value
        for key, value in adapted["panel_binding"]["primary"]["support_ids_by_label"].items()
    }
    assert _call("exact_ordered_equal", restored, raw) is True


def test_exact_ordered_equal_rejects_key_order_scalar_type_and_signed_zero_drift() -> None:
    assert _call("exact_ordered_equal", {"a": [1, -0.0]}, {"a": [1, -0.0]}) is True
    assert _call("exact_ordered_equal", {"a": 1}, {"a": True}) is False
    assert _call("exact_ordered_equal", {"a": 1, "b": 2}, {"b": 2, "a": 1}) is False
    assert struct.pack(">d", 0.0) != struct.pack(">d", -0.0)
    assert _call("exact_ordered_equal", {"z": 0.0}, {"z": -0.0}) is False
    assert _call("exact_ordered_equal", (1,), (1,)) is False


def test_legacy_roundtrip_calls_old_scientific_payload_with_exact_components() -> None:
    raw = _raw_payload()
    calls: list[dict[str, object]] = []
    _call("validate_legacy_roundtrip", _encoded(raw), _legacy_module(raw, calls))
    assert len(calls) == 1
    assert tuple(calls[0]) == (
        "manifest_audit",
        "execution_audit",
        "environment",
        "seed_audits",
        "primary_rows",
        "alternate_rows",
        "integrity",
        "aggregation",
        "bootstrap",
        "panel_binding",
    )
    assert list(calls[0]["panel_binding"]["primary"]["support_ids_by_label"]) == [0, 7, 42]


def test_legacy_roundtrip_requires_full_ordered_equality_and_exact_writer_bytes() -> None:
    raw = _raw_payload()
    calls: list[dict[str, object]] = []
    module = _legacy_module(raw, calls)
    _call("validate_legacy_roundtrip", _encoded(raw), module)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call(
            "validate_legacy_roundtrip",
            _encoded(raw).replace(b'  "manifest"', b' "manifest"', 1),
            module,
        )


@pytest.mark.parametrize("mutation", ("field", "source", "order", "signed_zero"))
def test_legacy_roundtrip_rejects_selected_field_current_source_and_canonicalizing_mutants(
    mutation: str,
) -> None:
    raw = _raw_payload()
    calls: list[dict[str, object]] = []
    module = _legacy_module(raw, calls)
    original = module.scientific_payload

    def mutant(**kwargs: object) -> dict[str, object]:
        result = original(**kwargs)
        if mutation == "field":
            result["aggregation"]["count"] = 2
        elif mutation == "source":
            result["manifest"]["source"] = {"revision": "current"}
        elif mutation == "order":
            result = dict(reversed(result.items()))
        else:
            result["environment"]["signed_zero"] = 0.0
        return result

    module.scientific_payload = mutant
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call("validate_legacy_roundtrip", _encoded(raw), module)


def test_legacy_manifest_projection_is_exact_ordered_ten_keys_derived_from_h(
    tmp_path: Path,
) -> None:
    manifest = {
        "base_preregistration": {"path": "base", "sha256": "a" * 64},
        "amendment": {"path": "a", "sha256": "b" * 64, "commit": "1" * 40},
        "deterministic_pool_amendment": {"path": "d", "sha256": "c" * 64, "commit": "2" * 40},
        "zero_jacobian_classifier_amendment": {"path": "z", "sha256": "d" * 64, "commit": "3" * 40},
        "binding_receipt": {"path": "r", "sha256": "e" * 64},
        "historical": {"producer_commit": "4" * 40},
        "artifact_schema": {"path": "schema", "sha256": "f" * 64},
        "current_scientific_source": {"git_revision": "5" * 40, "files": {}},
    }
    projection = _call("legacy_manifest_projection", tmp_path, manifest)
    assert tuple(projection) == (
        "path",
        "sha256",
        "base_preregistration",
        "amendment",
        "deterministic_pool_amendment",
        "zero_jacobian_classifier_amendment",
        "binding_receipt",
        "historical",
        "artifact_schema",
        "source",
    )
    assert projection["source"] is manifest["current_scientific_source"]


@pytest.mark.parametrize(
    "later",
    (
        "adjoint_integrity_amendment",
        "normwise_adjoint_calibration_protocol",
        "normwise_adjoint_calibration_result",
        "normwise_adjoint_amendment",
        "normwise_adjoint_sign_control_amendment",
    ),
)
def test_legacy_manifest_projection_rejects_later_authority_and_current_projection_mutants(
    later: str,
) -> None:
    raw = _raw_payload()
    raw["manifest"][later] = {"forbidden": True}
    calls: list[dict[str, object]] = []
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call("validate_legacy_roundtrip", _encoded(raw), _legacy_module(raw, calls))


def _synthetic_old_manifest() -> dict[str, object]:
    return {
        "base_preregistration": {"path": "base", "sha256": "a" * 64},
        "amendment": {"path": "amendment", "sha256": "b" * 64},
        "deterministic_pool_amendment": {"path": "pool", "sha256": "c" * 64},
        "zero_jacobian_classifier_amendment": {"path": "zero", "sha256": "d" * 64},
        "binding_receipt": {"path": "receipt", "sha256": "e" * 64},
        "historical": {"producer_commit": "f" * 40},
        "artifact_schema": {"path": "schema", "sha256": "1" * 64},
        "current_scientific_source": {"git_revision": "2" * 40, "files": {}},
    }


def _run_synthetic_legacy_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    *,
    raised: Exception | None = None,
    persisted_numpy_version: object | None = None,
    wrong_numpy_module: bool = False,
    forked: bool = False,
) -> tuple[int, bytes, list[dict[str, object]]]:
    import numpy

    old = tmp_path / "old"
    manifest_path = old / _NAMESPACE["LEGACY_MANIFEST_PATH"]
    manifest_path.parent.mkdir(parents=True)
    old_manifest = _synthetic_old_manifest()
    manifest_path.write_text(json.dumps(old_manifest), encoding="utf-8")
    raw = _raw_payload()
    raw["environment"]["numpy_version"] = (
        numpy.__version__
        if persisted_numpy_version is None
        else persisted_numpy_version
    )
    raw["manifest"] = _call("legacy_manifest_projection", old, old_manifest)
    raw_bytes = _encoded(raw)
    artifact = tmp_path / "artifact.json"
    artifact.write_bytes(raw_bytes)
    descriptor = os.open(artifact, os.O_RDONLY)
    calls: list[dict[str, object]] = []
    module = _legacy_module(raw, calls)
    if wrong_numpy_module:
        module.np = object()
    if raised is not None:
        def fail(**_kwargs: object) -> dict[str, object]:
            raise raised

        module.scientific_payload = fail
    runtime = {
        "python_executable": ".venv/bin/python",
        "python_version": "3.12.3",
        "numpy_version": numpy.__version__,
    }
    monkeypatch.setitem(_NAMESPACE, "ARTIFACT_SHA256", hashlib.sha256(raw_bytes).hexdigest())
    monkeypatch.setitem(_NAMESPACE, "authenticate_runtime", lambda _repository: runtime)
    monkeypatch.setitem(
        _NAMESPACE,
        "authenticate_verifier_provenance",
        lambda _repository, _manifest: {
            "source_commit": "a" * 40,
            "handoff_commit": "b" * 40,
        },
    )
    monkeypatch.setitem(_NAMESPACE, "authenticate_legacy_provenance", lambda _repo: {})
    monkeypatch.setitem(_NAMESPACE, "_load_legacy_module", lambda _checkout: module)

    def git(_repository: Path, *arguments: str, **_kwargs: object) -> str:
        return _NAMESPACE["LEGACY_HANDOFF_COMMIT"] if "rev-parse" in arguments else ""

    monkeypatch.setitem(_NAMESPACE, "_git", git)
    arguments = types.SimpleNamespace(
        live_repository=str(tmp_path),
        old_checkout=str(old),
        artifact_fd=descriptor,
        verifier_source_commit="a" * 40,
        verifier_handoff_commit="b" * 40,
        expected_numpy_version=numpy.__version__,
    )
    if forked:
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            try:
                os.close(read_fd)
                os.dup2(write_fd, 1)
                os.close(write_fd)
                os._exit(_call("_legacy_child", arguments))
            except BaseException:
                os._exit(127)
        os.close(write_fd)
        os.close(descriptor)
        token = os.read(read_fd, 65)
        os.close(read_fd)
        _, status = os.waitpid(pid, 0)
        exit_code = os.waitstatus_to_exitcode(status)
    else:
        try:
            exit_code = _call("_legacy_child", arguments)
        finally:
            os.close(descriptor)
        token = capfd.readouterr().out.encode("ascii")
    return exit_code, token, calls


def test_real_h_scientific_payload_roundtrips_a_synthetic_artifact_in_isolated_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    exit_code, token, _calls = _run_synthetic_legacy_child(
        tmp_path, monkeypatch, capfd, forked=True
    )
    assert (exit_code, token) == (0, _NAMESPACE["VALID_TOKEN"])


def test_legacy_provenance_binds_h_s_old_manifest_and_all_31_blobs() -> None:
    provenance = _call("authenticate_legacy_provenance", _REPOSITORY)
    assert tuple(provenance) == (
        "handoff_commit",
        "source_commit",
        "manifest_path",
        "manifest_sha256",
        "diagnostic_path",
        "diagnostic_sha256",
    )
    assert _NAMESPACE["LEGACY_SOURCE_ORDER"][1] == "scripts/diagnose_pass200_rsta_stage_a.py"
    assert len(_NAMESPACE["LEGACY_SOURCE_ORDER"]) == 31


def test_legacy_child_uses_old_h_cwd_diagnostic_file_and_callable() -> None:
    temporary = _call("_create_legacy_checkout", _REPOSITORY)
    try:
        module = _call("_load_legacy_module", temporary.checkout)
        expected = temporary.checkout / _NAMESPACE["LEGACY_DIAGNOSTIC_PATH"]
        assert Path(module.__file__).resolve() == expected.resolve()
        assert callable(module.scientific_payload)
    finally:
        temporary.cleanup()


@pytest.mark.parametrize(
    ("raised", "expected_exit", "expected_token"),
    (
        (ValueError("registered validation"), 1, "INVALID_TOKEN"),
        (RuntimeError("runtime failure"), 2, "STRUCTURAL_TOKEN"),
    ),
)
def test_legacy_child_distinguishes_artifact_invalid_from_structural_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    raised: Exception,
    expected_exit: int,
    expected_token: str,
) -> None:
    exit_code, token, calls = _run_synthetic_legacy_child(
        tmp_path, monkeypatch, capfd, raised=raised
    )
    assert (exit_code, token) == (expected_exit, _NAMESPACE[expected_token])
    assert calls == []


@pytest.mark.parametrize(
    ("mutation", "expected_exit", "expected_token"),
    (
        ("version_type", 2, "STRUCTURAL_TOKEN"),
        ("module_identity", 2, "STRUCTURAL_TOKEN"),
        ("persisted_version", 1, "INVALID_TOKEN"),
    ),
)
def test_legacy_child_rejects_numpy_type_module_and_persisted_drift_before_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capfd: pytest.CaptureFixture[str],
    mutation: str,
    expected_exit: int,
    expected_token: str,
) -> None:
    import numpy

    if mutation == "version_type":
        monkeypatch.setattr(
            numpy, "__version__", type("Version", (str,), {})(numpy.__version__)
        )
    exit_code, token, calls = _run_synthetic_legacy_child(
        tmp_path,
        monkeypatch,
        capfd,
        persisted_numpy_version="wrong" if mutation == "persisted_version" else None,
        wrong_numpy_module=mutation == "module_identity",
    )
    assert (exit_code, token) == (expected_exit, _NAMESPACE[expected_token])
    assert calls == []


def test_verifier_provenance_binds_v_hv_manifest_file_and_all_32_blobs() -> None:
    assert (
        _NAMESPACE["ROUNDTRIP_SOURCE_ORDER"][3]
        == "scripts/verify_pass200_rsta_scientific_artifact.py"
    )
    assert len(_NAMESPACE["ROUNDTRIP_SOURCE_ORDER"]) == 32
    assert callable(_NAMESPACE["authenticate_verifier_provenance"])


def _git_run(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _commit_all(repository: Path, message: str) -> str:
    _git_run(repository, "add", "--all")
    _git_run(repository, "commit", "-m", message)
    return _git_run(repository, "rev-parse", "HEAD")


def _synthetic_verifier_repository(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    extra_between_plan_and_source: bool,
) -> tuple[Path, Path]:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git_run(repository, "init", "-q")
    _git_run(repository, "config", "user.email", "rsta@example.invalid")
    _git_run(repository, "config", "user.name", "RSTA Test")
    for path_text in _NAMESPACE["ROUNDTRIP_SOURCE_ORDER"]:
        path = repository / path_text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"initial {path_text}\n", encoding="utf-8")
    amendment_path_text = "docs/recovery.md"
    amendment_path = repository / amendment_path_text
    amendment_path.parent.mkdir(parents=True, exist_ok=True)
    amendment_path.write_text("recovery\n", encoding="utf-8")
    recovery = _commit_all(repository, "recovery")
    plan_path = repository / "docs/superpowers/plans/recovery.md"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text("plan\n", encoding="utf-8")
    plan = _commit_all(repository, "plan")
    if extra_between_plan_and_source:
        extra_path = repository / "docs/unregistered-intermediate.md"
        extra_path.write_text("intermediate\n", encoding="utf-8")
        _commit_all(repository, "unregistered intermediate")
    source_paths = (
        "scripts/diagnose_pass200_rsta_stage_a.py",
        "scripts/verify_pass200_rsta_scientific_artifact.py",
        "tests/test_diagnose_pass200_rsta_stage_a.py",
        "tests/test_verify_pass200_rsta_scientific_artifact.py",
    )
    for path_text in source_paths:
        path = repository / path_text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"reviewed {path_text}\n", encoding="utf-8")
    source = _commit_all(repository, "source")
    files = {
        path_text: hashlib.sha256((repository / path_text).read_bytes()).hexdigest()
        for path_text in _NAMESPACE["ROUNDTRIP_SOURCE_ORDER"]
    }
    manifest = {
        "scientific_artifact_roundtrip_recovery_amendment": {
            "path": amendment_path_text,
            "sha256": hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
            "commit": recovery,
        },
        "current_scientific_source": {"git_revision": source, "files": files},
    }
    manifest_path = repository / _NAMESPACE["LEGACY_MANIFEST_PATH"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _commit_all(repository, "handoff")
    _git_run(repository, "checkout", "--detach", "HEAD")
    monkeypatch.setitem(
        _NAMESPACE, "__file__", str(repository / _NAMESPACE["VERIFIER_PATH"])
    )
    monkeypatch.setitem(_NAMESPACE, "RECOVERY_AMENDMENT_PATH", amendment_path_text)
    monkeypatch.setitem(
        _NAMESPACE,
        "RECOVERY_AMENDMENT_SHA256",
        hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
    )
    monkeypatch.setitem(_NAMESPACE, "RECOVERY_AMENDMENT_COMMIT", recovery)
    monkeypatch.setitem(_NAMESPACE, "RECOVERY_PLAN_COMMIT", plan)
    return repository, manifest_path


def test_verifier_provenance_requires_exact_recovery_plan_source_handoff_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest_path = _synthetic_verifier_repository(
        tmp_path, monkeypatch, extra_between_plan_and_source=True
    )
    with pytest.raises(_exception("StructuralFailure"), match="plan|parent|chain"):
        _call("authenticate_verifier_provenance", repository, manifest_path)


def test_verifier_provenance_accepts_exact_recovery_plan_source_handoff_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest_path = _synthetic_verifier_repository(
        tmp_path, monkeypatch, extra_between_plan_and_source=False
    )
    provenance = _call("authenticate_verifier_provenance", repository, manifest_path)
    assert provenance["source_commit"] == _git_run(repository, "rev-parse", "HEAD^")
    assert provenance["handoff_commit"] == _git_run(repository, "rev-parse", "HEAD")


def test_verifier_provenance_rejects_merge_handoff_with_exact_source_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository, manifest_path = _synthetic_verifier_repository(
        tmp_path, monkeypatch, extra_between_plan_and_source=False
    )
    handoff = _git_run(repository, "rev-parse", "HEAD")
    source = _git_run(repository, "rev-parse", "HEAD^")
    _git_run(repository, "checkout", "-q", "-b", "side", source)
    side_path = repository / "docs/side.md"
    side_path.write_text("side\n", encoding="utf-8")
    _commit_all(repository, "side")
    _git_run(repository, "checkout", "-q", "--detach", handoff)
    _git_run(repository, "merge", "--no-ff", "side", "-m", "merge handoff")
    with pytest.raises(_exception("StructuralFailure"), match="handoff parent"):
        _call("authenticate_verifier_provenance", repository, manifest_path)


@pytest.mark.parametrize("version", (type("Version", (str,), {})("2.1.3"), object()))
def test_runtime_authentication_requires_exact_builtin_numpy_version(
    monkeypatch: pytest.MonkeyPatch, version: object
) -> None:
    import numpy

    monkeypatch.setattr(numpy, "__version__", version)
    with pytest.raises(_exception("StructuralFailure"), match="NumPy|version|runtime"):
        _call("authenticate_runtime", _REPOSITORY)


def test_runtime_authentication_rejects_python_version_tuple_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "version_info", (3, 12, 4))
    with pytest.raises(_exception("StructuralFailure"), match="Python|runtime"):
        _call("authenticate_runtime", _REPOSITORY)


@pytest.mark.parametrize(
    ("raised", "expected"),
    (
        (ValueError("artifact validation"), "ArtifactInvalid"),
        (RuntimeError("runtime defect"), "RuntimeError"),
        (ImportError("import defect"), "ImportError"),
        (OSError("process defect"), "OSError"),
        (MemoryError("memory defect"), "MemoryError"),
    ),
)
def test_legacy_roundtrip_classifies_only_registered_validation_failures_as_invalid(
    raised: Exception, expected: str
) -> None:
    raw = _raw_payload()
    module = types.ModuleType("legacy_exception")

    def scientific_payload(**_kwargs: object) -> dict[str, object]:
        raise raised

    module.scientific_payload = scientific_payload
    builtin_types = {
        "RuntimeError": RuntimeError,
        "ImportError": ImportError,
        "OSError": OSError,
        "MemoryError": MemoryError,
    }
    expected_type = _exception(expected) if expected in _NAMESPACE else builtin_types[expected]
    with pytest.raises(expected_type):
        _call("validate_legacy_roundtrip", _encoded(raw), module)


def test_isolated_child_uses_exact_command_fd_environment_tokens_and_timeout(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    class Completed:
        returncode = 0
        pid = 789

        def __init__(self, command: list[str], **kwargs: object) -> None:
            observed["command"] = command
            observed.update(kwargs)

        def communicate(self, *, timeout: int | None = None) -> tuple[bytes, bytes]:
            observed["timeout"] = timeout
            return b"RSTA_LEGACY_VALID\n", b""

    monkeypatch.setattr(subprocess, "Popen", Completed)
    monkeypatch.setitem(_NAMESPACE, "_create_legacy_checkout", lambda repository: tmp_path)
    code, exit_code = _call(
        "run_isolated_legacy_child",
        _REPOSITORY,
        17,
        verifier_source_commit="a" * 40,
        verifier_handoff_commit="b" * 40,
        python_executable=Path(sys.executable),
        expected_numpy_version="2.1.3",
    )
    assert (code, exit_code) == (789, 0)
    command = observed["command"]
    assert command[:3] == [sys.executable, "-I", "-B"]
    assert observed["cwd"] == tmp_path
    assert observed["close_fds"] is True
    assert observed["pass_fds"] == (17,)
    assert observed["start_new_session"] is True
    assert observed["timeout"] == 600
    assert observed["stdin"] == subprocess.DEVNULL
    assert observed["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_verifier_rejects_wrong_parent_import_path_dirty_checkout_and_blob(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(_NAMESPACE, "__file__", str(tmp_path / "shadow.py"))
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call(
            "authenticate_verifier_provenance",
            _REPOSITORY,
            _REPOSITORY / "docs/pass200_rsta_receipt_stage_a_manifest.json",
        )


def test_runtime_authentication_rejects_wrong_sys_executable_python_and_numpy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(sys, "executable", str(tmp_path / "python"))
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call("authenticate_runtime", _REPOSITORY)


def test_legacy_child_requires_parent_child_persisted_numpy_before_call() -> None:
    raw = _raw_payload()
    raw["environment"]["numpy_version"] = "wrong"
    calls: list[dict[str, object]] = []
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call(
            "validate_legacy_roundtrip",
            _encoded(raw),
            _legacy_module(raw, calls),
            expected_numpy_version="2.1.3",
        )
    assert calls == []


def test_roundtrip_receipt_runtime_fields_are_observed_authenticated_values() -> None:
    receipt = _receipt()
    _call("validate_roundtrip_receipt", receipt)
    assert tuple(receipt["process"])[3:6] == (
        "python_executable",
        "python_version",
        "numpy_version",
    )


def test_roundtrip_receipt_exact_schema_predicates_and_every_nested_mutation() -> None:
    valid = _receipt()
    _call("validate_roundtrip_receipt", valid)
    mutations = []
    missing = deepcopy(valid)
    missing.pop("attempt")
    mutations.append(missing)
    extra = deepcopy(valid)
    extra["extra"] = None
    mutations.append(extra)
    reordered = dict(reversed(valid.items()))
    mutations.append(reordered)
    wrong_type = deepcopy(valid)
    wrong_type["attempt"] = True
    mutations.append(wrong_type)
    relation = deepcopy(valid)
    relation["process"]["child_exit_code"] = 1
    mutations.append(relation)
    nested_extra = deepcopy(valid)
    nested_extra["artifact"]["row"] = {}
    mutations.append(nested_extra)
    for mutant in mutations:
        with pytest.raises((ValueError, RuntimeError, OSError)):
            _call("validate_roundtrip_receipt", mutant)


def test_roundtrip_receipt_contains_no_scientific_content() -> None:
    forbidden = {
        "verdict",
        "decisive_clause",
        "candidate",
        "field",
        "row",
        "score",
        "metric",
        "aggregate",
        "bootstrap",
        "criterion",
        "exclusion",
        "excerpt",
    }
    receipt = _receipt()
    assert not forbidden.intersection(
        key.lower()
        for mapping in (
            receipt,
            receipt["artifact"],
            receipt["legacy_provenance"],
            receipt["verifier_provenance"],
            receipt["process"],
        )
        for key in mapping
    )
    _call("validate_roundtrip_receipt", receipt)


def test_receipt_path_is_derived_only_from_authenticated_hv(tmp_path: Path) -> None:
    expected = (
        tmp_path
        / "reports/generated/pass200_rsta_receipt"
        / (f"{'b' * 40}-scientific-artifact-roundtrip-validation.json")
    )
    assert _call("receipt_path", tmp_path, "b" * 40) == expected
    for invalid in ("B" * 40, "b" * 39, "../escape"):
        with pytest.raises((ValueError, RuntimeError, OSError)):
            _call("receipt_path", tmp_path, invalid)


def test_atomic_receipt_never_replaces_or_follows_a_path(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.parent.mkdir(exist_ok=True)
    value = _receipt()
    _call("write_validation_receipt_atomic", path, value)
    first = path.read_bytes()
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call("write_validation_receipt_atomic", path, value)
    assert path.read_bytes() == first
    target = tmp_path / "target.json"
    target.write_text("protected", encoding="utf-8")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises((ValueError, RuntimeError, OSError)):
        _call("write_validation_receipt_atomic", link, value)
    assert target.read_text(encoding="utf-8") == "protected"


def test_cli_consumes_one_attempt_and_never_reaches_science_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    manifest_path = repository / _NAMESPACE["LEGACY_MANIFEST_PATH"]
    artifact_path = repository / _NAMESPACE["ARTIFACT_PATH"]
    output_parent = repository / "reports/generated/pass200_rsta_receipt"
    manifest_path.parent.mkdir(parents=True)
    artifact_path.parent.mkdir(parents=True)
    output_parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{}", encoding="utf-8")
    artifact_path.write_bytes(b"opaque synthetic artifact")
    monkeypatch.chdir(repository)
    receipt = _receipt()
    runtime = {
        "python_executable": ".venv/bin/python",
        "python_version": "3.12.3",
        "numpy_version": receipt["process"]["numpy_version"],
    }
    verifier = deepcopy(receipt["verifier_provenance"])
    legacy = deepcopy(receipt["legacy_provenance"])
    child_calls: list[int] = []
    forbidden_calls: list[str] = []
    monkeypatch.setitem(_NAMESPACE, "authenticate_runtime", lambda _repo: runtime)
    monkeypatch.setitem(
        _NAMESPACE,
        "authenticate_verifier_provenance",
        lambda _repo, _manifest: verifier,
    )
    monkeypatch.setitem(_NAMESPACE, "authenticate_legacy_provenance", lambda _repo: legacy)

    def child(_repo: Path, descriptor: int, **_kwargs: object) -> tuple[int, int]:
        child_calls.append(descriptor)
        return 456, 0

    monkeypatch.setitem(_NAMESPACE, "run_isolated_legacy_child", child)
    for name in (
        "load_dataset", "load_model", "run_scientific_diagnostic",
        "score_rsta_batch", "joint_bootstrap", "decide_stage_a",
    ):
        monkeypatch.setitem(
            _NAMESPACE,
            name,
            lambda *_args, _name=name, **_kwargs: forbidden_calls.append(_name),
        )
    output = _call("receipt_path", repository, verifier["handoff_commit"])
    argv = [
        "--manifest", _NAMESPACE["LEGACY_MANIFEST_PATH"],
        "--artifact", _NAMESPACE["ARTIFACT_PATH"],
        "--output", str(output),
        "--validate-immutable-artifact-once",
    ]
    assert _call("main", argv) == 0
    first_bytes = output.read_bytes()
    persisted = _call("strict_json_object", first_bytes, name="synthetic receipt")
    _call("validate_roundtrip_receipt", persisted)
    assert forbidden_calls == []
    assert len(child_calls) == 1
    assert _call("main", argv) == 2
    assert output.read_bytes() == first_bytes
    assert len(child_calls) == 1
