"""Regression checks for RC5 release-decision evidence rejection."""

from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_ARCHIVE = _ROOT / "docs/evidence/packed_topk_rc5_pointer_raw_v1.tar.gz"
_COLLECTOR = _ROOT / "scripts/collect_packed_topk_rc5_pointer_decision.py"


def _changed_archive(target: Path, replacements: dict[str, bytes]) -> None:
    with tarfile.open(_ARCHIVE, "r:gz") as source, tarfile.open(target, "w:gz") as changed:
        for member in source.getmembers():
            if not member.isfile():
                changed.addfile(member)
                continue
            stream = source.extractfile(member)
            assert stream is not None
            contents = replacements.get(member.name, stream.read())
            member.size = len(contents)
            changed.addfile(member, io.BytesIO(contents))


@pytest.mark.parametrize("mutation", ["substituted_api", "false_pool", "failed_rss"])
def test_collector_rejects_substituted_or_failed_evidence(tmp_path: Path, mutation: str) -> None:
    replacements: dict[str, bytes] = {}
    with tarfile.open(_ARCHIVE, "r:gz") as source:

        def read(name: str) -> dict:
            stream = source.extractfile(name)
            assert stream is not None
            return json.load(stream)

        if mutation == "substituted_api":
            for name in (
                "raw/candidate_gc_objects.json",
                "raw/pair1_candidate.json",
                "raw/pair2_candidate.json",
                "raw/combined_memory_profile_fixed.json",
            ):
                receipt = read(name)
                receipt["api_sha256"] = "0" * 64
                replacements[name] = json.dumps(receipt).encode()
        elif mutation == "false_pool":
            receipt = read("raw/combined_pool_peak.json")
            receipt["tracked_pool_peak_bytes"] = 1
            replacements["raw/combined_pool_peak.json"] = json.dumps(receipt).encode()
        else:
            receipt = read("raw/pair1_candidate.json")
            receipt["process_peak_rss_bytes"] = 2 * 1024**3
            replacements["raw/pair1_candidate.json"] = json.dumps(receipt).encode()

        order = read("raw/run_order.json")
        for item in order["files"]:
            name = f"raw/{item['name']}"
            if name in replacements:
                item["sha256"] = hashlib.sha256(replacements[name]).hexdigest()
        replacements["raw/run_order.json"] = json.dumps(order).encode()

    archive = tmp_path / "changed.tar.gz"
    _changed_archive(archive, replacements)
    result = subprocess.run(
        [sys.executable, str(_COLLECTOR), str(archive), str(tmp_path / "decision.json")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0
    if mutation == "failed_rss":
        decision = json.loads((tmp_path / "decision.json").read_text())
        assert decision["release_gate_passed"] is False
