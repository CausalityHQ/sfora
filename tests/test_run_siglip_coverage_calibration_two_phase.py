"""Static and structural contracts for the two-phase local supervisor."""

from __future__ import annotations

import os
from pathlib import Path

_RUNNER = (
    Path(__file__).resolve().parents[1] / "scripts" / "run_siglip_coverage_calibration_two_phase.sh"
)


def test_runner_uses_two_sequential_phase_specific_landlock_children() -> None:
    source = _RUNNER.read_text()
    assert os.access(_RUNNER, os.X_OK)
    assert source.count('\n  "$landlock"') == 2
    assert source.count("scripts/probe_siglip_coverage_calibration.py") == 2
    fit, evaluate = source.split('\n  "$landlock"')[1:]
    fit = fit.split('\n  "$landlock"', 1)[0]
    assert "--phase fit" in fit
    assert '--ro "$optimization_images"' in fit
    assert '--ro "$support_images"' in fit
    assert '--ro "$heldout_images"' not in fit
    assert '--rw "$phase1"' in fit
    assert '--rw "$staging"' not in fit
    assert "--phase evaluate" in evaluate
    assert '--ro "$heldout_images"' in evaluate
    assert '--ro "$phase1"' in evaluate
    assert '--rw "$phase1"' not in evaluate
    assert '--rw "$phase2"' in evaluate
    assert "fit_receipt_sha=$(sha256sum" in source
    assert source.index("--phase fit") < source.index("fit_receipt_sha=$(sha256sum")
    assert source.index("fit_receipt_sha=$(sha256sum") < source.index("--phase evaluate")
    for forbidden in ("setsid", "timeout", "aws s3", "--storage-uri"):
        assert forbidden not in source
    assert '"$python" -B scripts/prepare_siglip_coverage_calibration.py' in source
    assert source.index("scripts/prepare_siglip_coverage_calibration.py") < source.index(
        '"$landlock"'
    )
    assert ': >"$staging/preparation.complete"' in source
    assert 'test -d "$staging"' in source


def test_runner_passes_only_receipt_digest_across_the_phase_boundary() -> None:
    source = _RUNNER.read_text()
    assert "fit_receipt_sha=$(sha256sum" in source
    boundary = source.split("fit_receipt_sha=$(sha256sum", 1)[1].split('"$landlock"', 1)[0]
    assert "python" not in boundary
    assert "image" not in boundary
    assert "map_sha" not in boundary
    assert '--fit-receipt-sha256 "$fit_receipt_sha"' in source
