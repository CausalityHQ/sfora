"""Verify a clean RC4 wheel installation against the frozen packed-search fixture."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import resource
import sys
import zipfile
from pathlib import Path

import numpy as np

_VERSION = "0.3.0rc4"
_API_SHA256 = "b7c57022a836774d641a829e6aac71c1d547e3f71136f716d3c9aeeedad11409"
_LIBRARY_SHA256 = "39602d0e4e8b0d5ec441be460ad7f18e288241bef19fb6e6c5df14f4033ac73c"
_MANIFEST_SHA256 = "2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799"
_REFERENCE_SHA256 = {
    "1000000_1": "ca1e391d3c942d127d3f4cc3ba42b888b1e37710e42bc0ac3f1a62e57cb004a4",
    "1000000_32": "afbb2fe2421ca9b147cfc135dafce6fd016ec356bdc58e8c6c24e47ee4200e62",
    "1000003_1": "e915b606fec42a77f7641fd5c482857033980ef0974627d51f0be10e80cbf9fc",
    "1000003_32": "6d2379b579b503732acf002875675b23ba5d707409aa4e2dd6c15649284a741c",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)

    from sfora import cutile_int8

    api_path = Path(cutile_int8.__file__).resolve()
    if not api_path.is_relative_to(Path(sys.prefix).resolve()):
        raise AssertionError("API was not imported from the isolated wheel environment")
    if importlib.metadata.version("sfora") != _VERSION or sha256(api_path) != _API_SHA256:
        raise AssertionError("installed package identity differs")
    with zipfile.ZipFile(args.wheel) as wheel:
        if hashlib.sha256(wheel.read("sfora/cutile_int8.py")).hexdigest() != _API_SHA256:
            raise AssertionError("wheel API bytes differ from installed API")
    if sha256(args.library) != _LIBRARY_SHA256:
        raise AssertionError("native library identity differs")

    manifest_path = args.fixture / "manifest.json"
    if sha256(manifest_path) != _MANIFEST_SHA256:
        raise AssertionError("fixture manifest differs")
    manifest = json.loads(manifest_path.read_text())
    expected_files = {
        f"gallery_{rows}_{kind}.bin"
        for rows in (1_000_000, 1_000_003)
        for kind in ("codes", "norms")
    } | {f"query_{batch}_{kind}.bin" for batch in (1, 32) for kind in ("codes", "norms")}
    if set(manifest["files"]) != expected_files:
        raise AssertionError("fixture file set differs")
    for name, authority in manifest["files"].items():
        path = args.fixture / name
        if path.stat().st_size != authority["bytes"] or sha256(path) != authority["sha256"]:
            raise AssertionError(f"fixture bytes differ: {name}")

    checks = {}
    for rows in (1_000_000, 1_000_003):
        gallery_codes = np.fromfile(
            args.fixture / f"gallery_{rows}_codes.bin", dtype=np.int8
        ).reshape(rows, 128)
        gallery_norms = np.fromfile(args.fixture / f"gallery_{rows}_norms.bin", dtype="<f2")
        with cutile_int8.CutilePackedInt8Gallery.open(
            args.library.resolve(), gallery_codes, gallery_norms
        ) as gallery:
            for batch in (1, 32):
                key = f"{rows}_{batch}"
                query_codes = np.fromfile(
                    args.fixture / f"query_{batch}_codes.bin", dtype=np.int8
                ).reshape(batch, 128)
                query_norms = np.fromfile(args.fixture / f"query_{batch}_norms.bin", dtype="<f2")
                ordinals, scores = gallery.search(query_codes, query_norms, k=10)
                reference_path = args.reference / f"exact_{key}.json"
                if sha256(reference_path) != _REFERENCE_SHA256[key]:
                    raise AssertionError(f"exact reference differs: {key}")
                reference = json.loads(reference_path.read_text())
                exact_ordinals = ordinals.reshape(-1).tolist() == reference["fused_ordinals"]
                exact_scores = (
                    scores.reshape(-1).view("<u4").tolist() == reference["fused_score_bits"]
                )
                if not exact_ordinals or not exact_scores:
                    raise AssertionError(f"installed wheel exactness differs: {key}")
                checks[key] = {"exact_ordinals": exact_ordinals, "exact_score_bits": exact_scores}

    receipt = {
        "schema": "sfora-rc4-clean-install-v1",
        "version": _VERSION,
        "wheel_filename": args.wheel.name,
        "wheel_sha256": sha256(args.wheel),
        "installed_api_path": str(api_path),
        "installed_api_sha256": sha256(api_path),
        "native_library_sha256": _LIBRARY_SHA256,
        "fixture_manifest_sha256": _MANIFEST_SHA256,
        "python": sys.version,
        "python_prefix": sys.prefix,
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "checks": checks,
    }
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"wheel_sha256": receipt["wheel_sha256"], "checks": checks}))


if __name__ == "__main__":
    main()
