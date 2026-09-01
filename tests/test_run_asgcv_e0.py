from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from sfora.asgcv import canonical_gradient_sample_bytes

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_asgcv_e0.py"
_SPEC = importlib.util.spec_from_file_location("run_asgcv_e0_subject", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


def _sample(ordinal: int) -> tuple[bytes, np.ndarray, np.ndarray]:
    patch = np.full((2, 49, 4), ordinal + 1, dtype=np.float32)
    gradient = patch * np.float32(0.25)
    receipt = canonical_gradient_sample_bytes(
        source_commit="1" * 40,
        model_revision="2" * 40,
        fixture_sha256="3" * 64,
        completion_group_sha256=f"{ordinal + 1:064x}",
        completion_protocol_sha256="4" * 64,
        eligible_schedule_sha256="5" * 64,
        pooler_state_sha256="6" * 64,
        predictor_state_sha256="7" * 64,
        eligible_pair_ordinal=ordinal,
        candidate_pair_ordinal=ordinal,
        pair_ordinals=(ordinal * 2, ordinal * 2 + 1),
        relation_sign=1 if ordinal % 2 == 0 else -1,
        grpo_loss=0.0,
        attention_kl=0.0,
        generated_tokens=8,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    return receipt, patch, gradient


def test_capture_triples_are_atomic_idempotent_and_resume_from_first_absent(tmp_path: Path) -> None:
    for ordinal in range(2):
        receipt, patch, gradient = _sample(ordinal)
        assert _MODULE.write_capture_triple(
            tmp_path,
            ordinal=ordinal,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        ) == ("written" if ordinal == 0 else "written")
        assert not tuple(tmp_path.glob("*.partial"))
    assert _MODULE.validated_capture_prefix(tmp_path, expected_count=4) == 2

    receipt, patch, gradient = _sample(1)
    assert (
        _MODULE.write_capture_triple(
            tmp_path,
            ordinal=1,
            receipt=receipt,
            patch_tokens=patch,
            exact_gradient=gradient,
        )
        == "reused"
    )


def test_capture_resume_rejects_partial_gap_corruption_and_shape_drift(tmp_path: Path) -> None:
    receipt, patch, gradient = _sample(0)
    _MODULE.write_capture_triple(
        tmp_path,
        ordinal=0,
        receipt=receipt,
        patch_tokens=patch,
        exact_gradient=gradient,
    )
    np.save(tmp_path / "patch-000001.npy", patch, allow_pickle=False)
    with pytest.raises(ValueError, match="partial"):
        _MODULE.validated_capture_prefix(tmp_path, expected_count=4)

    (tmp_path / "patch-000001.npy").unlink()
    np.save(tmp_path / "gradient-000000.npy", gradient + np.float32(1.0), allow_pickle=False)
    with pytest.raises(ValueError):
        _MODULE.validated_capture_prefix(tmp_path, expected_count=4)

    other = tmp_path / "other"
    other.mkdir()
    bad_patch = np.zeros((2, 48, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        _MODULE.write_capture_triple(
            other,
            ordinal=0,
            receipt=receipt,
            patch_tokens=bad_patch,
            exact_gradient=np.zeros_like(bad_patch),
        )

    receipt_one, patch_one, gradient_one = _sample(1)
    with pytest.raises(ValueError, match="ordinal"):
        _MODULE.write_capture_triple(
            other,
            ordinal=0,
            receipt=receipt_one,
            patch_tokens=patch_one,
            exact_gradient=gradient_one,
        )
    assert not tuple(other.iterdir())
