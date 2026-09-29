#!/usr/bin/env python3
"""Exercise actual CPU startup with changed qualified-code/mechanics fixtures."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import torch
import train_pe_large_pool as driver


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--preflight-sha256", required=True)
    parser.add_argument("--execution-sha256", required=True)
    parser.add_argument("--mechanics-receipt-sha256")
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    root = Path(__file__).resolve().parent
    argv = [
        str(root / "train_pe_large_pool.py"),
        "--check-startup-only",
        "--updates",
        "100" if args.mechanics_receipt_sha256 else "17",
        "--output",
        "/unused-large-pool-cpu-startup",
        "--preflight-sha256",
        args.preflight_sha256,
        "--execution-sha256",
        args.execution_sha256,
    ]
    if args.mechanics_receipt_sha256:
        argv += ["--mechanics-receipt-sha256", args.mechanics_receipt_sha256]
    with patch.object(sys, "argv", argv):
        driver.main()
        original_sha = driver.pair.sha

        def changed_sha(path):
            return (
                "changed-code"
                if Path(path) == root / "check_pe_large_pool.py"
                else original_sha(path)
            )

        with patch.object(driver.pair, "sha", changed_sha):
            try:
                driver.main()
            except AssertionError as error:
                assert str(error) == "GPU-qualified code differs"
            else:
                raise AssertionError("changed qualified code accepted")
        if args.mechanics_receipt_sha256:
            original_loads = json.loads

            def changed_mechanics(value, *positional, **keywords):
                result = original_loads(value, *positional, **keywords)
                if (
                    isinstance(result, dict)
                    and result.get("updates") == 17
                    and result.get("advance")
                ):
                    result["execution_sha256"] = "other-mechanics-execution"
                return result

            with patch.object(json, "loads", changed_mechanics):
                try:
                    driver.main()
                except AssertionError as error:
                    assert str(error) == "mechanics execution differs"
                else:
                    raise AssertionError("different mechanics execution accepted")
        driver.main()
    print(
        "PASS actual CPU startup; changed GPU-qualified code and supplied mechanics identity rejected without modifying files"
    )


if __name__ == "__main__":
    main()
