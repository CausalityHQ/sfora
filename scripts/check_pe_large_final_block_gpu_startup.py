#!/usr/bin/env python3
"""Authenticate actual GPU startup on CPU and reject changed qualified code."""

import argparse
from pathlib import Path
from unittest.mock import patch

import torch

import qualify_pe_large_final_block_gpu as driver


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--execution-sha256", required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    root = Path(__file__).resolve().parent
    driver.startup(root, args.execution_sha256)
    original = driver.native.base.pair.sha

    def changed(path):
        return "changed-code" if Path(path) == root / "pe_large_final_block.py" else original(path)

    with patch.object(driver.native.base.pair, "sha", changed):
        try:
            driver.startup(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == "GPU execution code differs"
        else:
            raise AssertionError("changed native GPU code accepted")
    print("PASS authenticated native GPU startup and changed-qualified-code rejection; no CUDA/images/quality")


if __name__ == "__main__":
    main()
