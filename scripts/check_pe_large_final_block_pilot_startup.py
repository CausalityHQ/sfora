#!/usr/bin/env python3
"""Real CPU startup authority and changed-pilot-code rejection."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
import train_pe_large_final_block_pilot as driver


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    root = Path(__file__).resolve().parent
    driver.startup(root, args.execution_sha256)
    original = driver.pair.sha
    def changed(path):
        return 'changed-code' if Path(path) == root / 'train_pe_large_final_block_pilot.py' else original(path)
    with patch.object(driver.pair, 'sha', changed):
        try:
            driver.startup(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == 'pilot execution code differs'
        else:
            raise AssertionError('changed pilot code accepted')
    def changed_receipt(path):
        return 'changed-receipt' if Path(path) == driver.MECHANICS_DIR / 'receipt.json' else original(path)
    with patch.object(driver.pair, 'sha', changed_receipt):
        try:
            driver.startup(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == 'discarded mechanics authority differs'
        else:
            raise AssertionError('changed mechanics receipt accepted')
    print('PASS actual pilot startup and changed-code rejection; no CUDA/images/quality')


if __name__ == '__main__':
    main()
