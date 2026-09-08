from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/benchmark_packed_int4_resident.py"
_ROOT = _SCRIPT.parents[1]
_SPEC = importlib.util.spec_from_file_location("benchmark_packed_int4_resident", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.skipif(
    not _MODULE.packed_int4._cpu_int_mm_available(),
    reason="exact CPU int8 matrix kernel unavailable",
)
def test_benchmark_records_three_exact_arms_and_memory_authority() -> None:
    original_threads = _MODULE.torch.get_num_threads()
    alternate_threads = 1 if original_threads != 1 else 2
    receipt = _MODULE.benchmark_resident_int4(
        gallery_count=19,
        dimensions=128,
        query_count=3,
        samples=4,
        warmups=2,
        threads=alternate_threads,
        seed=17,
    )

    assert _MODULE.torch.get_num_threads() == original_threads
    assert receipt["schema"] == "sfora-packed-int4-resident-benchmark-v2"
    assert receipt["claim_eligible"] is False
    assert receipt["integer_backend_required"] is True
    assert len(receipt["script_sha256"]) == 64
    assert len(receipt["packed_int4_source_sha256"]) == 64
    assert receipt["execution_backend"] == "torch-private-int-mm-cpu-observed"
    assert isinstance(receipt["cpu_model"], str) and receipt["cpu_model"]
    assert receipt["resident_construction_ns"] > 0
    assert receipt["process_peak_rss_bytes"] > 0
    assert receipt["scores_bit_exact"] is True
    assert set(receipt["arms"]) == {
        "packed_decode_float",
        "resident_float32",
        "resident_int8",
    }
    assert all(len(arm["samples_ns"]) == 4 for arm in receipt["arms"].values())
    assert receipt["memory_bytes_per_vector"] == {
        "packed_persistent": 66,
        "resident_float32_layout": 516,
        "resident_int8_layout": 130,
        "resident_int8_plus_packed_if_coexisting": 196,
    }
    raw = _MODULE.canonical_benchmark_bytes(receipt)
    assert raw.endswith(b"\n") and not raw.endswith(b"\n\n")
    assert json.loads(raw) == receipt


def test_committed_benchmark_receipt_authenticates_released_sources() -> None:
    receipt_path = _ROOT / "docs/evidence/packed_int4_resident/resident-int4-dgx-v1.json"
    raw = receipt_path.read_bytes()
    receipt = json.loads(raw)

    assert hashlib.sha256(raw).hexdigest() == (
        "2b2804acaa720ee7f39520d918d4b20096ae1c65b7d7cca7caeb34e29b8adc48"
    )
    assert receipt["schema"] == "sfora-packed-int4-resident-benchmark-v2"
    assert receipt["claim_eligible"] is False
    assert (
        receipt["packed_int4_source_sha256"]
        == hashlib.sha256((_ROOT / "src/sfora/packed_int4.py").read_bytes()).hexdigest()
    )
    assert receipt["script_sha256"] == hashlib.sha256(_SCRIPT.read_bytes()).hexdigest()
    assert _MODULE.canonical_benchmark_bytes(receipt) == raw


def test_benchmark_rejects_unobserved_thread_count(monkeypatch: pytest.MonkeyPatch) -> None:
    observed_threads = _MODULE.torch.get_num_threads()
    requested_threads = 1 if observed_threads != 1 else 2
    monkeypatch.setattr(_MODULE.torch, "set_num_threads", lambda _threads: None)

    with pytest.raises(RuntimeError, match="thread authority differs"):
        _MODULE.benchmark_resident_int4(
            gallery_count=3,
            dimensions=2,
            query_count=1,
            samples=1,
            warmups=0,
            threads=requested_threads,
            seed=17,
        )


def test_benchmark_cli_requires_explicit_execution(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        _MODULE.parse_args([])
    parsed = _MODULE.parse_args(
        [
            "--output",
            str(tmp_path / "receipt.json"),
            "--execute-resident-int4-benchmark",
        ]
    )
    assert parsed.output == tmp_path / "receipt.json"
