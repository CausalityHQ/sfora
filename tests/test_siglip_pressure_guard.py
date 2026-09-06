"""Behavioral contracts for the SigLIP deployment pressure fence."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

_GUARD = Path(__file__).resolve().parents[1] / "scripts" / "siglip_pressure_guard.sh"


def _classify(psi_percent: str, consecutive_samples: int) -> tuple[int, str]:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; sfora_pressure_update "$2" "$3"',
            "pressure-test",
            str(_GUARD),
            psi_percent,
            str(consecutive_samples),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    hits, reason = completed.stdout.strip().split()
    return int(hits), reason


@pytest.mark.parametrize(
    ("psi_percent", "prior_hits", "expected"),
    [
        ("0.79", 2, (0, "none")),
        ("49.99", 2, (0, "none")),
        ("50.00", 0, (1, "none")),
        ("50.00", 1, (2, "none")),
        ("50.00", 2, (3, "psi-sustained")),
        ("78.99", 0, (1, "none")),
        ("79.00", 0, (1, "psi-immediate")),
    ],
)
def test_pressure_guard_interprets_linux_psi_avg10_as_percent(
    psi_percent: str, prior_hits: int, expected: tuple[int, str]
) -> None:
    """Catch treating Linux's 0..100 PSI percentage as a 0..1 fraction."""

    assert _classify(psi_percent, prior_hits) == expected


@pytest.mark.parametrize("psi_percent", ["", "nan", "inf", "-1", "100.01"])
def test_pressure_guard_rejects_invalid_kernel_samples(psi_percent: str) -> None:
    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; sfora_pressure_update "$2" 0',
            "pressure-test",
            str(_GUARD),
            psi_percent,
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
