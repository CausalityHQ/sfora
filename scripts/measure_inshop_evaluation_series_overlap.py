#!/usr/bin/env python3
"""Measure filename-series overlap between official In-Shop queries and gallery."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from measure_fragmentation_acquisition_alignment import _parse


def measure(lines: list[str]) -> dict[str, Any]:
    if len(lines) < 3:
        raise ValueError("evaluation partition is truncated")
    expected = int(lines[0].strip())
    rows: list[tuple[str, str, str]] = []
    for line in lines[2:]:
        fields = line.split()
        if len(fields) != 3:
            raise ValueError(f"malformed evaluation row: {line!r}")
        rows.append((fields[0], fields[1], fields[2]))
    if len(rows) != expected:
        raise ValueError(f"declared {expected} rows but parsed {len(rows)}")

    gallery: dict[str, list[str]] = defaultdict(list)
    queries: list[tuple[str, str]] = []
    for path, item, status in rows:
        series = _parse(path)[0]
        if status == "gallery":
            gallery[item].append(series)
        elif status == "query":
            queries.append((item, series))
        elif status != "train":
            raise ValueError(f"unknown evaluation status: {status!r}")

    counts = {"both": 0, "same_only": 0, "cross_only": 0, "none": 0}
    same_pairs = 0
    all_pairs = 0
    for item, query_series in queries:
        positive_series = gallery.get(item, [])
        same = any(value == query_series for value in positive_series)
        cross = any(value != query_series for value in positive_series)
        if same and cross:
            counts["both"] += 1
        elif same:
            counts["same_only"] += 1
        elif cross:
            counts["cross_only"] += 1
        else:
            counts["none"] += 1
        same_pairs += sum(value == query_series for value in positive_series)
        all_pairs += len(positive_series)

    total = len(queries)
    return {
        "declared_images": expected,
        "queries": total,
        "query_category_counts": counts,
        "query_category_fractions": {key: value / total for key, value in counts.items()},
        "queries_with_same_series_gallery_positive_fraction": (counts["both"] + counts["same_only"])
        / total,
        "queries_with_cross_series_gallery_positive_fraction": (
            counts["both"] + counts["cross_only"]
        )
        / total,
        "query_positive_gallery_pairs": all_pairs,
        "same_series_query_positive_gallery_pairs": same_pairs,
        "same_series_query_positive_gallery_pair_fraction": same_pairs / all_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("partition", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = measure(args.partition.read_text(encoding="utf-8").splitlines())
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
