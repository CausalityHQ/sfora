#!/usr/bin/env python3
"""Actual no-CUDA validation startup and changed-code authority rejection."""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch
import validate_pe_large_pool_checkpoint as driver


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    root = Path(__file__).resolve().parent
    driver.startup(root, args.execution_sha256)
    original = driver.native.pair.sha

    def changed(path):
        return "changed-code" if Path(path) == root / "check_pe_large_pool.py" else original(path)

    with patch.object(driver.native.pair, "sha", changed):
        try:
            driver.startup(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == "validation code differs"
        else:
            raise AssertionError("changed validation code accepted")
    print("PASS fresh validation CPU startup and changed-code rejection; no CUDA/images/scores")


if __name__ == "__main__":
    main()
