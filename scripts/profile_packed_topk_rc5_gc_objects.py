"""Count retained GC-tracked objects across authenticated packed-search calls."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import resource
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_MANIFEST_SHA256 = "2d34832dfe22378b767f9cbedc0911e04ff2e06add2d1fdea66e6d0d55497799"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def _verify_fixture(path: Path) -> None:
    manifest_path = path / "manifest.json"
    if _sha256(manifest_path) != _MANIFEST_SHA256:
        raise ValueError("fixture manifest hash differs")
    files = json.loads(manifest_path.read_text())["files"]
    expected = {
        f"gallery_{rows}_{kind}.bin"
        for rows in (1_000_000, 1_000_003)
        for kind in ("codes", "norms")
    } | {f"query_{batch}_{kind}.bin" for batch in (1, 32) for kind in ("codes", "norms")}
    if set(files) != expected:
        raise ValueError("fixture file set differs")
    for name in expected:
        authority = files[name]
        source = path / name
        if source.stat().st_size != authority["bytes"] or _sha256(source) != authority["sha256"]:
            raise ValueError(f"fixture bytes differ: {name}")


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--library", required=True, type=Path)
    parser.add_argument("--library-sha256", required=True)
    parser.add_argument("--api-root", required=True, type=Path)
    parser.add_argument("--api-sha256", required=True)
    parser.add_argument("--fixture", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _verify_fixture(args.fixture)
    api_path = args.api_root / "sfora/cutile_int8.py"
    if _sha256(api_path) != args.api_sha256:
        raise ValueError("Python API hash differs")
    if _sha256(args.library) != args.library_sha256:
        raise ValueError("native library hash differs")
    reference = json.loads(args.reference.read_text())
    sys.path.insert(0, str(args.api_root.resolve()))
    from sfora.cutile_int8 import CutilePackedInt8Gallery

    rows = 1_000_000
    gallery_codes = np.fromfile(args.fixture / "gallery_1000000_codes.bin", dtype=np.int8).reshape(
        rows, 128
    )
    gallery_norms = np.fromfile(args.fixture / "gallery_1000000_norms.bin", dtype="<f2")
    query_codes = np.fromfile(args.fixture / "query_32_codes.bin", dtype=np.int8).reshape(32, 128)
    query_norms = np.fromfile(args.fixture / "query_32_norms.bin", dtype="<f2")
    with CutilePackedInt8Gallery.open(
        args.library.resolve(), gallery_codes, gallery_norms
    ) as gallery:
        ordinals, scores = gallery.search(query_codes, query_norms)
        if ordinals.reshape(-1).tolist() != reference["fused_ordinals"]:
            raise AssertionError("ordinal authority differs")
        if scores.reshape(-1).view("<u4").tolist() != reference["fused_score_bits"]:
            raise AssertionError("score-bit authority differs")
        for _ in range(5):
            gallery.search(query_codes, query_norms)
        gc.collect(2)
        was_enabled = gc.isenabled()
        gc.disable()
        try:
            before_ids = {id(obj) for obj in gc.get_objects()}
            before_count = gc.get_count()
            for _ in range(50):
                gallery.search(query_codes, query_norms)
            after_count = gc.get_count()
            new_objects = [obj for obj in gc.get_objects() if id(obj) not in before_ids]
        finally:
            if was_enabled:
                gc.enable()
    types = Counter(type(obj).__module__ + "." + type(obj).__qualname__ for obj in new_objects)
    receipt = {
        "schema": "sfora-packed-topk-rc5-gc-objects-v1",
        "api_sha256": args.api_sha256,
        "library_sha256": args.library_sha256,
        "fixture_manifest_sha256": _MANIFEST_SHA256,
        "reference_sha256": _sha256(args.reference),
        "rows": rows,
        "batch": 32,
        "warmups": 5,
        "calls": 50,
        "gc_count_before": before_count,
        "gc_count_after": after_count,
        "new_tracked_object_types": sorted(types.items()),
        "process_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
    }
    args.output.write_text(json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n")
    print(json.dumps({"output": str(args.output), "types": types.most_common(8)}, sort_keys=True))


if __name__ == "__main__":
    main()
