#!/usr/bin/env python3
"""Paired public-call 1M search replay for the OML SOP compact profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from sfora.atomic_publication import publish_bytes_noreplace
from sfora.cutile_int8 import CutilePackedInt8Gallery
from sfora.model_profiles import load_oml_sop_compact_encoder

FEATURE_SHA256 = "8f565027b20e55923826a5240d97c564171428a5f1cc7290e680d2223fd28da4"
FIXTURE_MANIFEST_SHA256 = "2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799"
LIBRARY_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
API_SHA256 = "b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409"
ROWS = 1_000_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def percentile(samples: list[int], percent: int) -> int:
    return sorted(samples)[math.ceil(len(samples) * percent / 100) - 1]


def fixture_arrays(
    root: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]]:
    if sha256(root / "manifest.json") != FIXTURE_MANIFEST_SHA256:
        raise ValueError("RC4 fixture manifest differs")
    manifest = json.loads((root / "manifest.json").read_text())
    needed = [f"gallery_{ROWS}_{kind}.bin" for kind in ("codes", "norms")]
    needed += [f"query_{batch}_{kind}.bin" for batch in (1, 32) for kind in ("codes", "norms")]
    for name in needed:
        path = root / name
        record = manifest["files"][name]
        if sha256(path) != record["sha256"] or path.stat().st_size != record["bytes"]:
            raise ValueError("RC4 fixture input differs")
    codes = np.fromfile(root / f"gallery_{ROWS}_codes.bin", dtype=np.int8).reshape(ROWS, 128)
    norms = np.fromfile(root / f"gallery_{ROWS}_norms.bin", dtype="<f2")
    queries = {
        batch: (
            np.fromfile(root / f"query_{batch}_codes.bin", dtype=np.int8).reshape(batch, 128),
            np.fromfile(root / f"query_{batch}_norms.bin", dtype="<f2"),
        )
        for batch in (1, 32)
    }
    return codes, norms, queries


def profile_arrays(
    features: Path,
) -> tuple[np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]], dict[str, str]]:
    if sha256(features) != FEATURE_SHA256:
        raise ValueError("OML SOP features differ")
    with np.load(features, allow_pickle=False) as archive:
        values = torch.from_numpy(np.ascontiguousarray(archive["test_features"]))
    if values.shape != (60_502, 384):
        raise ValueError("OML SOP feature inventory differs")
    encoder = load_oml_sop_compact_encoder()
    packed = encoder.encode_packed(values)
    repeats = math.ceil(ROWS / len(packed.codes))
    codes = np.ascontiguousarray(np.tile(packed.codes.numpy(), (repeats, 1))[:ROWS])
    norms = np.ascontiguousarray(np.tile(packed.inverse_norms.numpy(), repeats)[:ROWS])
    queries = {
        batch: (
            np.ascontiguousarray(packed.codes.numpy()[:batch]),
            np.ascontiguousarray(packed.inverse_norms.numpy()[:batch]),
        )
        for batch in (1, 32)
    }
    return (
        codes,
        norms,
        queries,
        {
            "feature_sha256": FEATURE_SHA256,
            "encoder_sha256": encoder.sha256,
            "gallery_code_sha256": hashlib.sha256(codes.tobytes()).hexdigest(),
            "gallery_norm_sha256": hashlib.sha256(norms.tobytes()).hexdigest(),
            "construction": "tile 60502 SOP test codes in original order to 1000000 rows",
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or sha256(args.library) != LIBRARY_SHA256:
        raise ValueError("SOP packed benchmark authority differs")
    import sfora.cutile_int8 as api

    if sha256(Path(api.__file__)) != API_SHA256:
        raise ValueError("SOP packed API differs")
    fixture_codes, fixture_norms, fixture_queries = fixture_arrays(args.fixture)
    profile_codes, profile_norms, profile_queries, profile_meta = profile_arrays(args.features)
    all_data = {
        "fixture": (fixture_codes, fixture_norms, fixture_queries),
        "profile": (profile_codes, profile_norms, profile_queries),
    }
    with ExitStack() as stack:
        galleries = {
            name: stack.enter_context(CutilePackedInt8Gallery.open(args.library, data[0], data[1]))
            for name, data in all_data.items()
        }
        results = []
        for pair, order in enumerate((("fixture", "profile"), ("profile", "fixture")), start=1):
            for name in order:
                for batch in (1, 32):
                    qcodes, qnorms = all_data[name][2][batch]
                    gallery = galleries[name]
                    first_ordinals, first_scores = gallery.search(qcodes, qnorms, k=10)
                    first_hash = hashlib.sha256(
                        first_ordinals.tobytes() + first_scores.tobytes()
                    ).hexdigest()
                    for _ in range(5):
                        gallery.search(qcodes, qnorms, k=10)
                    samples = []
                    for _ in range(50):
                        started = time.perf_counter_ns()
                        ordinals, scores = gallery.search(qcodes, qnorms, k=10)
                        samples.append(time.perf_counter_ns() - started)
                    if (
                        hashlib.sha256(ordinals.tobytes() + scores.tobytes()).hexdigest()
                        != first_hash
                    ):
                        raise ValueError("SOP packed result changed during replay")
                    results.append(
                        {
                            "pair": pair,
                            "name": name,
                            "batch": batch,
                            "warmups": 5,
                            "samples_ns": samples,
                            "p50_ns": percentile(samples, 50),
                            "p95_ns": percentile(samples, 95),
                            "p99_ns": percentile(samples, 99),
                            "queries_per_second": batch * 1e9 / (sum(samples) / len(samples)),
                            "first_result_sha256": first_hash,
                        }
                    )
    receipt = {
        "schema": "sfora-oml-sop-packed-1m-profile-v1",
        "claim_eligible": False,
        "source_commit": args.source_commit,
        "script_sha256": sha256(Path(__file__)),
        "library_sha256": LIBRARY_SHA256,
        "api_sha256": API_SHA256,
        "fixture_manifest_sha256": FIXTURE_MANIFEST_SHA256,
        "profile": profile_meta,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "rows": results,
    }
    payload = (
        json.dumps(receipt, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()

    def validate(persisted: bytes) -> None:
        if persisted != payload:
            raise ValueError("SOP packed benchmark publication differs")

    published = publish_bytes_noreplace(args.output, payload, validator=validate)
    published.close()
    print(json.dumps([{k: row[k] for k in ("pair", "name", "batch", "p99_ns")} for row in results]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
