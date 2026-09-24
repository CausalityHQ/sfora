#!/usr/bin/env python3
"""Time one cold native packed top-10 call and verify deterministic ties."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

import numpy as np

from sfora.cutile_int8 import CutilePackedInt8Gallery


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--gallery-rows", type=int, default=59_551)
    parser.add_argument("--batch", type=int, choices=(1, 32), default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tileiras = os.environ.get("CUTILE_TILEIRAS_PATH")
    if (
        args.gallery_rows < 10
        or args.output.exists()
        or args.output.is_symlink()
        or not args.library.is_absolute()
        or not args.library.is_file()
        or not tileiras
        or not Path(tileiras).is_file()
    ):
        raise ValueError("cuTile cold-start source authority differs")

    codes = np.ones((args.gallery_rows, 128), dtype=np.int8)
    norms = np.full(args.gallery_rows, 1.0 / np.sqrt(128.0), dtype=np.float16)
    queries = np.ones((args.batch, 128), dtype=np.int8)
    query_norms = np.full(args.batch, 1.0 / np.sqrt(128.0), dtype=np.float16)
    started = time.perf_counter_ns()
    with CutilePackedInt8Gallery.open(args.library, codes, norms) as gallery:
        created = time.perf_counter_ns()
        ordinals, scores = gallery.search(queries, query_norms)
        searched = time.perf_counter_ns()
    expected = np.broadcast_to(np.arange(10, dtype=np.int64), (args.batch, 10))
    if (
        ordinals.shape != (args.batch, 10)
        or not np.array_equal(ordinals, expected)
        or scores.shape != (args.batch, 10)
        or not np.isfinite(scores).all()
        or not np.all(scores == scores[:, :1])
    ):
        raise ValueError("cuTile cold-start top-10 tie parity differs")

    receipt = {
        "schema": "sfora-cutile-cold-start-v1",
        "source_sha256": sha256(Path(__file__)),
        "library_sha256": sha256(args.library),
        "tileiras_sha256": sha256(Path(tileiras)),
        "gallery_rows": args.gallery_rows,
        "batch": args.batch,
        "stored_gallery_bytes": args.gallery_rows * 130,
        "create_ns": created - started,
        "first_search_ns": searched - created,
        "total_ns": searched - started,
        "exact_ordinal_ties": True,
        "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "platform": platform.platform(),
    }
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    temporary.replace(args.output)
    print(json.dumps(receipt, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
