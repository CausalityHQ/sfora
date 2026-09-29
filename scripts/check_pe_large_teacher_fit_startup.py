#!/usr/bin/env python3
"""Actual CPU startup and altered code/teacher authority rejection."""
import argparse
from pathlib import Path
from unittest.mock import patch
import torch
import export_pe_large_teacher_fit as driver


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    args = parser.parse_args()
    assert not torch.cuda.is_available()
    root = Path(__file__).resolve().parent
    driver.startup(root, args.execution_sha256)
    original = driver.pair.sha
    for target, message in ((root / 'export_pe_large_teacher_fit.py', 'teacher execution code differs'), (driver.TEACHER, 'teacher checkpoint authority differs')):
        def changed(path):
            return 'changed-authority' if Path(path) == target else original(path)
        with patch.object(driver.pair, 'sha', changed):
            try:
                driver.startup(root, args.execution_sha256)
            except AssertionError as error:
                assert str(error) == message
            else:
                raise AssertionError('altered authority accepted')
    print('PASS actual teacher startup and changed-code/checkpoint rejection; no CUDA/images/quality')


if __name__ == '__main__':
    main()
