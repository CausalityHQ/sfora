#!/usr/bin/env python3
"""Real CPU startup authority and changed-training-code rejection."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
import train_pe_large_final_two as driver


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    root = Path(__file__).resolve().parent
    driver.startup(root, args.execution_sha256)
    original = driver.pair.sha
    def changed(path):
        return 'changed-code' if Path(path) == root / 'train_pe_large_final_two.py' else original(path)
    with patch.object(driver.pair, 'sha', changed):
        try:
            driver.startup(root, args.execution_sha256)
        except AssertionError as error:
            assert str(error) == 'training execution code differs'
        else:
            raise AssertionError('changed training code accepted')
    print('PASS actual training startup and changed-code rejection; no CUDA/images/quality')


if __name__ == '__main__':
    main()
