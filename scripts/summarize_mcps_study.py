#!/usr/bin/env python3
"""Summarize the fixed PA/MCPS/compactness three-seed study."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sfora.mcps_study import summarize_mcps_study, write_summary_no_clobber


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = summarize_mcps_study(args.study_root)
        write_summary_no_clobber(args.output, summary)
    except Exception as error:
        print(f"summary failed: {error}", file=sys.stderr)
        return 2
    print(f"summary complete: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
