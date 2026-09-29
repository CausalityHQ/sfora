#!/usr/bin/env python3
"""Bind the existing public serving qualifier to first-declared native100 arms."""
import argparse
import json
import sys
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import qualify_pe_native_valid_anchor_held as held
import qualify_pe_large_public_serving as public

HELD_CODE = '7484cb290fb8ec8bde8050fe2d66a1bfcac26dac81a686580dadf70cdbc4d365'
DECISION = Path('/home/riomus/runs/sfora-native-valid-anchor-decision-v1/decision.json')
DECISION_SHA = '3b54c2206a531de5f5020117025d55c1c9b0fcd447ec597cbc268ebf9744f7ad'


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--execution-sha256', required=True)
    parser.add_argument('--arm', choices=('control', 'treatment'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-startup-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    pair = held.driver.pair
    manifest = root / 'native-valid-anchor-serving-execution.json'
    assert pair.sha(manifest) == args.execution_sha256
    code = json.loads(manifest.read_text())
    old = json.loads((root / 'native-valid-anchor-held-execution.json').read_text())
    assert len(code) == len(old) + 1 and all(code[n] == h for n, h in old.items())
    assert pair.sha(DECISION) == DECISION_SHA
    decision = json.loads(DECISION.read_text())
    assert decision['decision'] == 'GO' and decision['official_read'] is False
    entry = next(v for v in decision['inputs'] if v['seed'] == 179041 and v['arm'] == args.arm)
    directory = Path(f'/home/riomus/runs/sfora-native-valid-anchor-held-179041-{args.arm}-v1')
    source = Path('/home/riomus/runs/sfora-native-valid-anchor-held-v3') / f'source-179041-{args.arm}-proof.json'
    assert pair.sha(source) == entry['source_sha256']
    proof = json.loads(source.read_text())
    assert proof['code'] == old and proof['native_training_seed'] == 179041
    helpers = held.cpu.qualified.confirmation.selected.helpers
    assert all(pair.sha(root / n) == h for n, h in code.items()), 'public serving code differs'
    # Authenticate the historical chain before overriding its shared public module.
    with patch.object(held.cpu.qualified.confirmation.selected, 'helpers', lambda r, _: helpers(r, code)):
        control, frozen, _, run, terminal, original = held.authority(root, HELD_CODE, 179041, args.arm, proof['native_training_receipt_sha256'])
    assert original == old and pair.sha(run / 'native.pt') == proof['teacher_checkpoint_sha256']
    assert terminal['updated_whole_sha256'] == proof['teacher_whole_sha256'] and terminal['updated_head_sha256'] == proof['teacher_head_sha256']

    def startup(r, execution):
        assert r == root and execution == args.execution_sha256
        assert all(pair.sha(root / n) == h for n, h in code.items()), 'public serving code differs'
        assert pair.sha(directory / 'receipt.json') == entry['receipt_sha256'] and pair.sha(directory / 'cpu-audit.json') == entry['audit_sha256']
        receipt = json.loads((directory / 'receipt.json').read_text())
        audit = json.loads((directory / 'cpu-audit.json').read_text())
        assert audit['pass'] and audit['receipt_sha256'] == entry['receipt_sha256']
        assert receipt['code'] == old and receipt['cpu_authority_sha256'] == entry['source_sha256']
        assert receipt['held_manifest'] == frozen['held_manifest'] and receipt['query'] == frozen['query'] and receipt['gallery'] == frozen['gallery']
        assert receipt['source_head_rng_environment_unchanged'] and receipt['independent_two_block_suffix_head_packed_exact']
        assert pair.sha(directory / 'large.held.npy') == receipt['held_sha256']
        assert pair.sha(public.LIBRARY) == public.LIBRARY_SHA
        return control, frozen, proof, receipt, code

    checkpoint = Path(f'/home/riomus/runs/sfora-native-valid-anchor-179041-{args.arm}-100-v1/native.pt')
    with ExitStack() as scope:
        scope.enter_context(patch.object(public, 'startup', startup))
        scope.enter_context(patch.object(public, 'HELD', directory))
        scope.enter_context(patch.object(public, 'HELD_SHA', entry['receipt_sha256']))
        scope.enter_context(patch.object(public, 'AUDIT_SHA', entry['audit_sha256']))
        scope.enter_context(patch.object(public.teacher, 'TEACHER', checkpoint))
        scope.enter_context(patch.object(public.teacher, 'TEACHER_SHA', proof['teacher_checkpoint_sha256']))
        argv = ['updated-native-public', '--execution-sha256', args.execution_sha256, '--output', str(args.output)]
        if args.check_startup_only:
            argv.append('--check-startup-only')
        scope.enter_context(patch.object(sys, 'argv', argv))
        public.main()
    assert pair.sha(DECISION) == DECISION_SHA


if __name__ == '__main__':
    main()
