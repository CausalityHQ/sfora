#!/usr/bin/env python3
"""Offline, parent-run exactness falsifier; never trains or reads official data.

NOT RUN as part of source qualification. Native checkpoint hashing imports torch
on CPU only, using the original pilot's typed fingerprint. Pin both original
terminal.json SHAs on the command line. The v1 endpoint is chunk-0500, whose
receipt chain supplies steps 1..500 across chunk-0100 through chunk-0500.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import train_siglip2_substrate_full as v1
import train_siglip2_substrate_full_v2 as v2
from score_siglip2_substrate_official_v2 import validate_source_pin

STEP_FIELDS = ('step', 'loss', 'ce', 'rank', 'preclip_norm', 'gradient_norms', 'batch',
               'augmentation_seed', 'pixels_sha256', 'rgb_sha256', 'schedule_sha256', 'scale')
STATE_FIELDS = ('config', 'vision', 'buffers', 'head', 'classifier', 'bank', 'target', 'pca',
                'schedules', 'positive', 'optimizer', 'optimizer_defaults', 'scaler',
                'counter', 'seed', 'cpu_rng', 'cuda_rng', 'numerical_flags')


def compare_evidence(old_rows, new_rows, old_receipt, new_receipt, old_state, new_state):
    """Exact numerical equality; exclude only pre-registered version/code metadata."""
    v2.require(old_receipt['schema'] == v1.SCHEMA and new_receipt['schema'] == v2.SCHEMA and
               new_receipt.get('chunk_length') == 500, 'v1->v2 schema/chunk length differs')
    identities = []
    for record in (old_receipt, new_receipt):
        identity = v2.canonical(record['identity'])
        v2.require({'schema', 'code'} <= identity.keys(), 'identity schema/code required')
        del identity['schema'], identity['code']
        identities.append(identity)
    v2.require(v2.json_sha(identities[0]) == v2.json_sha(identities[1]),
               'identity differs outside schema/code')
    for rows in (old_rows, new_rows):
        v2.require([row['step'] for row in rows] == list(range(1, 501)), 'steps 1..500 required')
    for old, new in zip(old_rows, new_rows, strict=True):
        for field in STEP_FIELDS:
            v2.require(field in old and field in new and v2.json_sha(old[field]) == v2.json_sha(new[field]),
                       f"step {old['step']} {field} differs")
    v2.require(set(old_state) == set(new_state) == set(STATE_FIELDS), 'state fingerprint inventory differs')
    for field in STATE_FIELDS:
        v2.require(field in old_state and field in new_state and old_state[field] == new_state[field],
                   f'state fingerprint {field} differs')
    # Per-step state_sha256 also hashes schema/identity; stable step fields are compared above.
    return {'raw_terminal_hash_equal': old_receipt['terminal_state_sha256'] == new_receipt['terminal_state_sha256']}


def validate_payload_fields(disk, pilot):
    metadata = {'schema', 'identity'}
    fields = set(pilot.PAYLOAD_KEYS) - metadata
    v2.require(fields == set(STATE_FIELDS) and set(pilot.PAYLOAD_KEYS) - set(STATE_FIELDS) == metadata,
               'pilot payload numerical/metadata inventory differs')
    v2.require(set(disk) == fields | metadata, 'checkpoint payload key inventory differs')


def admitted_endpoint(module, folder, terminal_sha):
    terminal_ref = module.descriptor(folder / 'terminal.json', terminal_sha)
    terminal = module.read_json(terminal_ref)
    v2.require(Path(terminal['receipt']['path']) == folder / 'receipt.json', 'endpoint receipt path differs')
    record = module.read_json(terminal['receipt'])
    v2.require(record['arm'] == 'large' and record['completed_step'] == 500, 'Large step500 required')
    v2.require(record['code'] == {'trainer': module.sha(Path(module.__file__)),
                                'partition_parser': module.PARSER_SHA}, 'reviewed endpoint source required')
    endpoint = module.check_chain(terminal['receipt'], 500, 'large', record['source'], record['code'], terminal_ref)
    rows, previous = [], terminal['receipt']
    while previous:
        row = module.read_json(previous)
        rows[:0] = row['steps']
        previous = row['previous']
    return endpoint, rows, terminal_ref


def checkpoint_fingerprints(record, pilot):
    # The existing typed fingerprint is the numerical authority; no tensor conversion or tolerance.
    import torch
    path = Path(record['checkpoint']['path'])
    v2.descriptor(path, record['checkpoint']['sha256'])
    disk = torch.load(path, weights_only=True, map_location='cpu', mmap=True)
    validate_payload_fields(disk, pilot)
    v2.require(disk['counter'] == 500 and disk['seed'] == v2.SEED and
               v2.canonical(disk['identity']) == record['identity'] and disk['schema'] == record['schema'],
               'checkpoint endpoint metadata differs')
    with path.open('rb') as stream:
        pages = pilot.CheckpointPages(stream)
        v2.require(pilot.fingerprint(disk, consumed=pages.consume) == record['terminal_state_sha256'],
                   'checkpoint complete-state fingerprint differs from receipt')
        result = {field: pilot.fingerprint(disk[field], consumed=pages.consume) for field in STATE_FIELDS}
    del disk
    gc.collect()
    return result


def run(args):
    old, old_rows, old_terminal = admitted_endpoint(v1, args.v1_chunk500, args.v1_terminal_sha256)
    new, new_rows, new_terminal = admitted_endpoint(v2, args.v2_chunk500, args.v2_terminal_sha256)
    v2.require(old['source'] == new['source'], 'same original pilot checkpoint/receipt required')
    source = v2.read_json(new['source']['receipt'])
    validate_source_pin(new['source'], 'large', source)
    v2.descriptor(new['source']['checkpoint']['path'], new['source']['checkpoint']['sha256'])
    for record in (old, new):
        v2.require(record['identity']['full_rows_sha256'] == new['identity']['full_rows_sha256'] and
                   record['identity']['schedule_sha256'] == new['identity']['schedule_sha256'],
                   'matched full rows/schedule required')
    pilot_path = Path(source['invocation']['argv'][0])
    pilot = v2.load_module('_chunk500_exactness_pilot', pilot_path, v2.PILOT_SHA)
    old_state = checkpoint_fingerprints(old, pilot)
    new_state = checkpoint_fingerprints(new, pilot)
    comparison = compare_evidence(old_rows, new_rows, old, new, old_state, new_state)
    # Re-admit the original references after checkpoint I/O; never rewrite campaign bytes.
    admitted_endpoint(v1, args.v1_chunk500, old_terminal['sha256'])
    admitted_endpoint(v2, args.v2_chunk500, new_terminal['sha256'])
    v2.descriptor(old['checkpoint']['path'], old['checkpoint']['sha256'])
    v2.descriptor(new['checkpoint']['path'], new['checkpoint']['sha256'])
    return {'pass': True, 'steps': 500, **comparison,
            'v1_terminal_state_sha256': old['terminal_state_sha256'],
            'v2_terminal_state_sha256': new['terminal_state_sha256'],
            'state_fingerprints': old_state, 'v1_terminal': old_terminal, 'v2_terminal': new_terminal,
            'quality_read': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--v1-chunk500', type=Path, required=True)
    parser.add_argument('--v1-terminal-sha256', required=True)
    parser.add_argument('--v2-chunk500', type=Path, required=True)
    parser.add_argument('--v2-terminal-sha256', required=True)
    args = parser.parse_args()
    try:
        result = run(args)
    except (ValueError, KeyError, OSError, RuntimeError) as error:
        print(json.dumps({'pass': False, 'error': str(error), 'quality_read': False}), flush=True)
        raise SystemExit(1)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
