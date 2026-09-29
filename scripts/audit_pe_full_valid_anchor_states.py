#!/usr/bin/env python3
"""Recheck physical preservation of all20 original fixed candidate resume states."""
if not __debug__:
    raise SystemExit('State audit requires Python assertions; optimized mode is forbidden')

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--source-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert sha(Path(__file__)) == args.source_sha256
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '' and not args.output.exists()
    root = Path('/home/riomus/runs/sfora-full-valid-anchor-training-v1')
    manifest = root / 'full-valid-anchor-training-execution.json'
    code = '0da63376f73c9bc55c4c5f6e8c0f3f80fab7efe211ef6234a72af2ce4ed3ac36'
    assert sha(manifest) == code
    sources = json.loads(manifest.read_text())
    assert all(sha(root / n) == h for n, h in sources.items())
    previous_receipt = previous_checkpoint = None
    rows, total_bytes = [], 0
    for end in range(100, 2001, 100):
        run = Path(f'/home/riomus/runs/sfora-full-valid-anchor-{end}-v1')
        receipt = run / 'receipt.json'
        value = json.loads(receipt.read_text())
        assert value['pass'] and value['completed_step'] == end and value['execution_sha256'] == code and not value['quality_read']
        assert value['previous_receipt_sha256'] == previous_receipt and value['previous_checkpoint_sha256'] == previous_checkpoint
        assert value['frozen_source_code_environment_rng_preserved'] and value['all_input_hashes_equal_archived_control']
        checkpoint = run / 'resume.pt'
        actual = sha(checkpoint)
        assert actual == value['checkpoint_sha256']
        log = (root / f'full-valid-anchor-{end}.log').read_text()
        assert 'Finished with result: success' in log and 'code=exited/status=0' in log and 'Memory swap peak: 0B' in log
        previous_receipt, previous_checkpoint = sha(receipt), actual
        total_bytes += checkpoint.stat().st_size
        rows.append({'completed_step': end, 'receipt_sha256': previous_receipt, 'checkpoint_sha256': actual, 'bytes': checkpoint.stat().st_size})
        print(f'PASS physical resume state {end}', flush=True)
    assert previous_receipt == 'd718e4dee3971b13398a43ec56d02057491059d7e62f99362ef384d71b3a036e'
    assert all(sha(root / n) == h for n, h in sources.items()) and sha(Path(__file__)) == args.source_sha256
    args.output.write_text(json.dumps({'pass': True, 'source_sha256': args.source_sha256, 'training_execution_sha256': code, 'all20_physical_resume_files_hash_exact': True, 'states': rows, 'total_checkpoint_bytes': total_bytes, 'python_assertions_enabled': __debug__, 'optimizer_updates': 0, 'quality_read': False}, indent=2) + '\n')


if __name__ == '__main__':
    main()
