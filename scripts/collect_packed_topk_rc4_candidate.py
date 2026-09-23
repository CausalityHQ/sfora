"""Recompute the RC4 decision receipt from archived DGX replay artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
RAW = EVIDENCE / "packed_topk_rc4_candidate_raw_v1.tar.gz"
STAGE_RAW = EVIDENCE / "packed_topk_rc4_stage_raw_v1.tar.gz"
OUTPUT = EVIDENCE / "packed_topk_rc4_candidate_v1.json"
RC3_LIBRARY = "a9e583881201760088f3d99c180367e587ca68219c79dfe1760a3467f06b8dea"
CANDIDATE_LIBRARY = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    return digest(path.read_bytes())


def rank(samples: list[int], percent: int) -> int:
    assert len(samples) == 50 and all(isinstance(item, int) and item > 0 for item in samples)
    return sorted(samples)[math.ceil(len(samples) * percent / 100) - 1]


def main() -> None:
    with tarfile.open(RAW, "r:gz") as archive, tarfile.open(STAGE_RAW, "r:gz") as old:

        def read(path: str) -> bytes:
            item = archive.extractfile(path)
            assert item is not None
            return item.read()

        def old_read(path: str) -> bytes:
            item = old.extractfile(path)
            assert item is not None
            return item.read()

        def replay(name: str) -> dict:
            return json.loads(read(f"evidence-outputs/{name}.json").splitlines()[0])

        def previous(name: str) -> dict:
            return json.loads(old_read(f"evidence-outputs/{name}.json").splitlines()[0])

        source_hash = digest(read("rust/sfora-cutile-int8-score/src/topk.rs"))
        script_hash = digest(read("profile_packed_topk_rc4_library.py"))
        manifest_hash = digest(read("evidence-inputs/manifest.json"))
        assert source_hash == file_digest(ROOT / "rust/sfora-cutile-int8-score/src/topk.rs")
        assert script_hash == file_digest(ROOT / "scripts/profile_packed_topk_rc4_library.py")

        exactness = {}
        for rows in (1_000_000, 1_000_003):
            for batch in (1, 32):
                new = replay(f"selector_exact_{rows}_{batch}")
                baseline = previous(f"exact_{rows}_{batch}")
                assert new["exact_ordinals"] and new["exact_score_bits"]
                assert new["fused_ordinals"] == baseline["fused_ordinals"]
                assert new["fused_score_bits"] == baseline["fused_score_bits"]
                exactness[f"{rows}_{batch}"] = {
                    "exact_ordinals": True,
                    "exact_score_bits": True,
                    "matched_frozen_fused": True,
                    "process_peak_rss_bytes": new["process_peak_rss_bytes"],
                }

        pairs = {}
        for pair in (1, 2):
            arms = {}
            for label, expected_hash in (("rc3", RC3_LIBRARY), ("candidate", CANDIDATE_LIBRARY)):
                item = replay(f"selector_pair{pair}_{label}")
                assert item["library_sha256"] == expected_hash
                assert item["fixture_manifest_sha256"] == manifest_hash
                assert item["process_peak_rss_bytes"] < 2 * 1024**3
                batches = {}
                for batch in (1, 32):
                    for rows in (1_000_000, 1_000_003):
                        check = item["batches"][f"{rows}_{batch}"]
                        assert check["exact_ordinals"] and check["exact_score_bits"]
                    result = item["batches"][f"1000000_{batch}"]
                    samples = result["end_to_end_ns"]
                    assert result["warmups"] == 5
                    for percent in (50, 95, 99):
                        assert result[f"p{percent}_ns"] == rank(samples, percent)
                    batches[str(batch)] = {
                        "p50_ns": result["p50_ns"],
                        "p95_ns": result["p95_ns"],
                        "p99_ns": result["p99_ns"],
                        "queries_per_second": result["queries_per_second"],
                        "process_peak_rss_bytes": item["process_peak_rss_bytes"],
                    }
                arms[label] = {"library_sha256": expected_hash, "batches": batches}
            batch_1_regression = (
                arms["candidate"]["batches"]["1"]["p99_ns"] / arms["rc3"]["batches"]["1"]["p99_ns"]
                - 1
            )
            batch_32_gain = 1 - (
                arms["candidate"]["batches"]["32"]["p99_ns"]
                / arms["rc3"]["batches"]["32"]["p99_ns"]
            )
            assert batch_1_regression <= 0.05 and batch_32_gain >= 0.20
            pairs[str(pair)] = {
                **arms,
                "batch_1_p99_regression_fraction": batch_1_regression,
                "batch_32_p99_gain_fraction": batch_32_gain,
            }

        rust_replay = {}
        gpu_memory = {}
        for batch in (1, 32):
            item = replay(f"selector_fused_{batch}")
            baseline = previous(f"fused_{batch}")
            assert item["ordinals"] == baseline["ordinals"]
            assert item["score_bits"] == baseline["score_bits"]
            samples = item["raw_end_to_end_ns"]
            assert item["warmups"] == 5
            rust_replay[str(batch)] = {
                "p50_ns": rank(samples, 50),
                "p95_ns": rank(samples, 95),
                "p99_ns": rank(samples, 99),
                "queries_per_second": batch * 1e9 / (sum(samples) / len(samples)),
                "process_peak_rss_bytes": item["process_peak_rss_bytes"],
            }
            connection = sqlite3.connect(":memory:")
            connection.deserialize(read(f"evidence-outputs/selector_mem_{batch}_profile.sqlite"))
            peak = connection.execute(
                "select max(localMemoryPoolUtilizedSize) from CUDA_GPU_MEMORY_USAGE_EVENTS"
            ).fetchone()[0]
            connection.close()
            assert isinstance(peak, int) and peak > 0
            gpu_memory[str(batch)] = {
                "tracked_cuda_pool_peak_bytes": peak,
                "nsys_report_sha256": digest(
                    read(f"evidence-outputs/selector_mem_{batch}_profile.nsys-rep")
                ),
                "scope": "maximum CUDA memory-pool utilized bytes in separate memory-enabled trace",
            }

        smoke = json.loads(read("evidence-outputs/clean_wheel_smoke.json"))
        smoke_path = EVIDENCE / "packed_topk_rc4_clean_wheel_smoke_v1.json"
        assert smoke == json.loads(smoke_path.read_text())
        assert smoke["library_sha256"] == CANDIDATE_LIBRARY

    receipt = {
        "schema": "sfora-packed-topk-rc4-candidate-v1",
        "source_commit": "086eb6c29ddc759c336aa9863367a3d001b11d42",
        "kernel_commit": "8dd89c8b02c3b88c0a5b05ea6ae1e4e44b9d785e",
        "kernel_source_sha256": source_hash,
        "ffi_replay_script_sha256": script_hash,
        "candidate_library_sha256": CANDIDATE_LIBRARY,
        "candidate_diagnostic_binary_sha256": (
            "ee68c1e7684298f3845138ce15c71a27462fe674800559bb4cbd7706a41ef112"
        ),
        "rc3_library_sha256": RC3_LIBRARY,
        "fixture_manifest_sha256": manifest_hash,
        "stage_receipt_sha256": file_digest(EVIDENCE / "packed_topk_rc4_stage_v1.json"),
        "flat_pilot_raw_sha256": file_digest(
            EVIDENCE / "packed_topk_rc4_pilot_width512_raw_v1.tar.gz"
        ),
        "candidate_raw_sha256": file_digest(RAW),
        "quality_revalidation_sha256": file_digest(
            EVIDENCE / "packed_topk_rc4_quality_revalidation_v1.json"
        ),
        "clean_wheel_smoke_sha256": file_digest(smoke_path),
        "wheel_sha256": smoke["wheel_sha256"],
        "sdist_sha256": "c08e5f1626d4de64082938888f48c46ae9dd8bc3db7489a449e9fb019913f988",
        "gallery_rows": 1_000_000,
        "tail_rows": 1_000_003,
        "dimensions": 128,
        "top_k": 10,
        "persistent_bytes_per_item": 130,
        "device": "NVIDIA GB10",
        "tileiras": "NVIDIA CUDA TileIR 13.4.92",
        "profiler": "NVIDIA Nsight Systems 2025.3.2",
        "exactness": exactness,
        "rust_replay": rust_replay,
        "paired_ffi_replays": pairs,
        "gpu_memory": gpu_memory,
        "release_gate": {
            "batch_32_p99_gain_at_least": 0.20,
            "batch_1_p99_regression_at_most": 0.05,
            "process_rss_below_bytes": 2 * 1024**3,
            "passed": True,
        },
        "claim_eligible": False,
    }
    OUTPUT.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(OUTPUT)
    for pair, item in pairs.items():
        print(pair, item["batch_1_p99_regression_fraction"], item["batch_32_p99_gain_fraction"])


if __name__ == "__main__":
    main()
