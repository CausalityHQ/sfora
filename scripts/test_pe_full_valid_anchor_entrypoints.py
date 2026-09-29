#!/usr/bin/env python3
"""Scientific entrypoints must reject optimized Python before model imports."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    for name in ('qualify_pe_full_valid_anchor_checkpoint.py', 'compare_pe_full_valid_anchor_checkpoint.py', 'audit_pe_full_valid_anchor_states.py', 'audit_pe_full_valid_anchor_processor.py'):
        for flag in ('-O', '-OO'):
            result = subprocess.run([sys.executable, flag, str(root / name), '--help'], capture_output=True, text=True, timeout=10)
            assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr and 'ModuleNotFoundError' not in result.stderr
        result = subprocess.run([sys.executable, str(root / name), '--help'], env={**os.environ, 'PYTHONOPTIMIZE': '1'}, capture_output=True, text=True, timeout=10)
        assert result.returncode != 0 and 'optimized mode is forbidden' in result.stderr and 'ModuleNotFoundError' not in result.stderr
    print('PASS all4 authority entrypoints reject -O/-OO/PYTHONOPTIMIZE before model imports')


if __name__ == '__main__':
    main()
