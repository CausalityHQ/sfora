#!/usr/bin/env python3
"""Run one pinned six-epoch PA+DADA In-Shop compatibility smoke."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sfora.dada_reproduction import (
    DadaSmokeRequest,
    build_smoke_config,
    publish_dada_smoke_report,
    run_dada_smoke,
    validate_dada_source,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dada-checkout", required=True, type=Path)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--work-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gpu", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    if output.exists() or output.is_symlink():
        print(f"structural failure: output already exists: {output}", file=sys.stderr)
        return 2
    try:
        source = validate_dada_source(args.dada_checkout)
        work_root = args.work_root.resolve()
        work_root.mkdir(parents=True, exist_ok=False)
        result_root = work_root / "results"
        result_root.mkdir()
        smoke_config = work_root / "inshop-smoke.yaml"
        build_smoke_config(source, smoke_config)
        request = DadaSmokeRequest(
            python=Path(sys.executable).resolve(),
            source=source,
            smoke_config=smoke_config,
            dataset_root=args.dataset_root.resolve(strict=True),
            output_root=result_root,
            save_name=f"dada-inshop-smoke-seed{args.seed}",
            gpu=args.gpu,
            seed=args.seed,
        )
        report = run_dada_smoke(request)
        publish_dada_smoke_report(output, report)
    except Exception as exc:
        print(f"structural failure: {exc}", file=sys.stderr)
        return 2
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
