"""Verify failed RC4 public-call pilots and write the finite-negative receipt."""

from __future__ import annotations

import hashlib
import json
import math
import tarfile
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence"
OUTPUT = EVIDENCE / "packed_topk_rc4_decision_v2.json"
LIBRARIES = {
    "rc3": "a9e583881201760088f3d99c180367e587ca68219c79dfe1760a3467f06b8dea",
    "candidate": "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c",
}
MANIFEST = "2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799"
PILOTS = {
    "rc4_api": {
        "archive": "packed_topk_rc4_rc4api_failed_raw_v1.tar.gz",
        "prefix": "selector_rc4api",
        "api_path": "clean-wheel-venv/lib/python3.12/site-packages/sfora/cutile_int8.py",
        "api_sha256": "7e585fa716ad79b0ad9f998ac6eb63a4f9c7a89409eefee2d9367885686c3818",
        "script_path": "profile_packed_topk_rc4_library.py",
        "script_sha256": "6ee62e2f569281515f54c843e1f7f90c0822a30a14c3444a1f4cebe00de44a3d",
        "fixture_file_authentication": "manifest_hash_only",
    },
    "direct_native_api": {
        "archive": "packed_topk_rc4_api_fast_failed_raw_v1.tar.gz",
        "prefix": "api_fast",
        "api_path": "api-pilot-venv/lib/python3.12/site-packages/sfora/cutile_int8.py",
        "api_sha256": "634d5edd161323924d8234fbbe6a780f56f6baf0ff9e404401ab9b951de172e5",
        "script_path": "api-pilot/profile_packed_topk_rc4_library.py",
        "script_sha256": "f7fc06bb42ec706662fa4ef336b0f26a8a4723a739cac814664a32043eac4071",
        "fixture_file_authentication": "consumed_files_sha256_verified",
    },
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def percentile(values: list[int], percent: int) -> int:
    assert len(values) == 50 and all(type(value) is int and value > 0 for value in values)
    return sorted(values)[math.ceil(len(values) * percent / 100) - 1]


def read(archive: tarfile.TarFile, path: str) -> bytes:
    item = archive.extractfile(path)
    assert item is not None
    return item.read()


def main() -> None:
    pilots = {}
    for name, authority in PILOTS.items():
        archive_path = EVIDENCE / authority["archive"]
        with tarfile.open(archive_path, "r:gz") as archive:
            assert sha256(read(archive, authority["api_path"])) == authority["api_sha256"]
            assert sha256(read(archive, authority["script_path"])) == authority["script_sha256"]
            assert sha256(read(archive, "evidence-inputs/manifest.json")) == MANIFEST
            pairs = {}
            for pair in (1, 2):
                arms = {}
                for label, library_hash in LIBRARIES.items():
                    item = json.loads(
                        read(
                            archive,
                            f"evidence-outputs/{authority['prefix']}_pair{pair}_{label}.json",
                        )
                    )
                    assert item["api_sha256"] == authority["api_sha256"]
                    assert item["library_sha256"] == library_hash
                    assert item["fixture_manifest_sha256"] == MANIFEST
                    batches = {}
                    for batch in (1, 32):
                        for rows in (1_000_000, 1_000_003):
                            check = item["batches"][f"{rows}_{batch}"]
                            assert check["exact_score_bits"] and check["exact_ordinals"]
                        measured = item["batches"][f"1000000_{batch}"]
                        samples = measured["end_to_end_ns"]
                        assert measured["warmups"] == 5
                        for percent in (50, 95, 99):
                            assert percentile(samples, percent) == measured[f"p{percent}_ns"]
                        batches[str(batch)] = {
                            "p50_ns": measured["p50_ns"],
                            "p95_ns": measured["p95_ns"],
                            "p99_ns": measured["p99_ns"],
                            "queries_per_second": measured["queries_per_second"],
                        }
                    assert item["process_peak_rss_bytes"] < 2 * 1024**3
                    arms[label] = {
                        "library_sha256": library_hash,
                        "process_peak_rss_bytes": item["process_peak_rss_bytes"],
                        "batches": batches,
                    }
                regression = (
                    arms["candidate"]["batches"]["1"]["p99_ns"]
                    / arms["rc3"]["batches"]["1"]["p99_ns"]
                    - 1
                )
                gain = 1 - (
                    arms["candidate"]["batches"]["32"]["p99_ns"]
                    / arms["rc3"]["batches"]["32"]["p99_ns"]
                )
                pairs[str(pair)] = {
                    **arms,
                    "batch_1_p99_regression_fraction": regression,
                    "batch_32_p99_gain_fraction": gain,
                    "passes_p99_gate": regression <= 0.05 and gain >= 0.20,
                }
        pilots[name] = {
            "archive_sha256": sha256(archive_path.read_bytes()),
            "api_sha256": authority["api_sha256"],
            "replay_script_sha256": authority["script_sha256"],
            "fixture_file_authentication": authority["fixture_file_authentication"],
            "pairs": pairs,
        }
    assert all(
        not pair["passes_p99_gate"] for pilot in pilots.values() for pair in pilot["pairs"].values()
    )
    retained_kernel_sha256 = sha256(
        (ROOT / "rust/sfora-cutile-int8-score/src/topk.rs").read_bytes()
    )
    retained_api_sha256 = sha256((ROOT / "src/sfora/cutile_int8.py").read_bytes())
    assert (
        retained_kernel_sha256 == "5a8643a6303918d0e59b114cd65b9f61659290c5d709c21e48af56100fa382b9"
    )
    assert retained_api_sha256 == PILOTS["rc4_api"]["api_sha256"]
    assert tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"] == "0.3.0rc3"
    retained_smoke_path = EVIDENCE / "packed_topk_rc4_retained_clean_wheel_smoke_v1.json"
    retained_smoke = json.loads(retained_smoke_path.read_text())
    assert retained_smoke["package_version"] == "0.3.0rc3"
    assert retained_smoke["api_sha256"] == retained_api_sha256
    assert retained_smoke["library_sha256"] == LIBRARIES["rc3"]
    assert retained_smoke["wheel_sha256"] == (
        "6f85355d5a4d847250e190cd3f38a90ceb2e26d53fd55897bba45c2364c58bd8"
    )
    receipt = {
        "schema": "sfora-packed-topk-rc4-decision-v2",
        "decision": "reject_merge_width_change_and_retain_rc3_scorer",
        "release_ready_performance_increment": False,
        "source_candidate_commit": "b4f352575eba7307666c1f770994fcecd2100032",
        "retained_source_commit": "60fcd2358915aed8f91a9049e1c8b8fc8da3f434",
        "rc3_library_sha256": LIBRARIES["rc3"],
        "rejected_library_sha256": LIBRARIES["candidate"],
        "retained_kernel_source_sha256": retained_kernel_sha256,
        "retained_python_api_sha256": retained_api_sha256,
        "retained_package_version": "0.3.0rc3",
        "retained_clean_wheel_smoke_sha256": sha256(retained_smoke_path.read_bytes()),
        "retained_wheel_sha256": retained_smoke["wheel_sha256"],
        "retained_sdist_sha256": "a312db1187dd7a13dc7358f772ce347142739653ef2a7fd8ceaa13e35abd5c8a",
        "fixture_manifest_sha256": MANIFEST,
        "stage_receipt_sha256": sha256((EVIDENCE / "packed_topk_rc4_stage_v1.json").read_bytes()),
        "historical_misqualified_receipt_sha256": sha256(
            (EVIDENCE / "packed_topk_rc4_candidate_v1.json").read_bytes()
        ),
        "quality_revalidation_sha256": sha256(
            (EVIDENCE / "packed_topk_rc4_quality_revalidation_v1.json").read_bytes()
        ),
        "gallery_rows": 1_000_000,
        "tail_rows": 1_000_003,
        "persistent_bytes_per_item": 130,
        "pilots": pilots,
        "next_bottleneck": (
            "public Python FFI tail: rare 14–19 ms calls; localize garbage "
            "collection, ctypes dispatch, and CUDA synchronization under "
            "mixed-batch traffic"
        ),
        "gc_attribution": "unverified_no_archived_gc_event_log",
        "claim_eligible": False,
    }
    OUTPUT.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(OUTPUT)


if __name__ == "__main__":
    main()
