#!/usr/bin/env python3
"""Stdlib-only source/inventory/authority negatives; never run extraction."""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import copy
import hashlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch


def rejects(call, message):
    try:
        call()
    except (ValueError, FileExistsError) as error:
        assert message in str(error), str(error)
        return
    raise AssertionError('invalid input accepted: ' + message)


def main():
    path = Path(__file__).with_name('extract_siglip2_vision_source.py')
    # This check catches omitted required authority before any third-party import.
    result = subprocess.run([sys.executable, '-B', '-S', str(path)], capture_output=True, text=True)
    assert result.returncode == 2 and 'required' in result.stderr, result.stderr
    import extract_siglip2_vision_source as extract

    metadata = Path(__file__).resolve().parents[1] / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1/so400-native256-upstream-metadata-v1.json'
    upstream = json.loads(metadata.read_text())
    config = upstream['config']
    expected, resolved = extract.expected_vision(config, upstream['model'])
    assert len(expected) == 448
    assert expected['vision_model.embeddings.patch_embedding.weight'] == [1152, 3, 16, 16]
    assert expected['vision_model.encoder.layers.26.mlp.fc1.weight'] == [4304, 1152]
    assert expected['vision_model.head.attention.in_proj_weight'] == [3456, 1152]
    assert expected['vision_model.head.probe'] == [1, 1, 1152]
    assert resolved['num_hidden_layers'] == 27
    assert not any('position_ids' in key for key in expected)
    for field, value in (('hidden_size', 1024), ('num_hidden_layers', 24),
                         ('intermediate_size', 4096), ('patch_size', 14),
                         ('image_size', 384), ('num_channels', 1),
                         ('num_attention_heads', 12), ('vision_use_head', False)):
        changed = copy.deepcopy(config)
        changed['vision_config'][field] = value
        rejects(lambda: extract.expected_vision(changed, upstream['model']), 'profile')
    rejects(lambda: extract.expected_vision(config, 'google/siglip2-large-patch16-256'), 'profile')
    rejects(lambda: extract.expected_vision(config, 'unrecognized'), 'source model')
    large = copy.deepcopy(config)
    large['vision_config'].update(hidden_size=1024, num_hidden_layers=24, intermediate_size=4096)
    large_expected, _ = extract.expected_vision(large, 'google/siglip2-large-patch16-256')
    assert len(large_expected) == 400
    assert large_expected['vision_model.encoder.layers.23.mlp.fc2.weight'] == [1024, 4096]
    rejects(lambda: extract.expected_vision(large, upstream['model']), 'profile')

    # Header fixtures describe real shapes without allocating source weights.
    inventory = {key: {'dtype': 'F32', 'shape': shape, 'data_offsets': [0, 0]}
                 for key, shape in expected.items()}
    inventory.update({'text_model.embeddings.token_embedding.weight':
                      {'dtype': 'F32', 'shape': [2, 2], 'data_offsets': [0, 16]},
                      'logit_bias': {'dtype': 'F32', 'shape': [], 'data_offsets': [0, 4]},
                      'logit_scale': {'dtype': 'F32', 'shape': [], 'data_offsets': [0, 4]}})
    mapping = extract.validate_vision(inventory, expected)
    assert len(set(mapping.values())) == 448
    assert mapping['vision_model.embeddings.patch_embedding.weight'] == 'embeddings.patch_embedding.weight'
    assert mapping['vision_model.head.probe'] == 'head.probe'
    assert all(not key.startswith('vision_model.') for key in mapping.values())
    for key in ('vision_model.head.probe', 'vision_model.encoder.layers.26.layer_norm2.bias'):
        changed = copy.deepcopy(inventory)
        del changed[key]
        rejects(lambda: extract.validate_vision(changed, expected), 'vision inventory')
    for key in ('vision_model.embeddings.position_ids', 'vision_model.extra',
                'vision_model.vision_model.head.probe'):
        changed = copy.deepcopy(inventory)
        changed[key] = {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}
        rejects(lambda: extract.validate_vision(changed, expected), 'vision inventory')
    for change in ({'shape': [1]}, {'dtype': 'I32'}):
        changed = copy.deepcopy(inventory)
        changed['vision_model.head.probe'].update(change)
        rejects(lambda: extract.validate_vision(changed, expected), 'vision tensor')
    for key in ('logit_scale', 'logit_bias', 'text_model.embeddings.token_embedding.weight'):
        changed = copy.deepcopy(inventory)
        del changed[key]
        rejects(lambda: extract.validate_vision(changed, expected), 'full source')
    changed = {**inventory, 'optimizer.weight': inventory['logit_bias']}
    rejects(lambda: extract.validate_vision(changed, expected), 'source key')

    with TemporaryDirectory() as temporary:
        base = Path(temporary)
        source = base / 'source.safetensors'
        source.write_bytes(b'not safetensors')
        output, provenance = base / 'vision.safetensors', base / 'provenance.json'
        config_path, processor_path = base / 'config.json', base / 'preprocessor_config.json'
        config_path.write_text(json.dumps(config))
        processor_path.write_text(json.dumps(upstream['preprocessor']))
        digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        args = extract.parser().parse_args([
            '--source', str(source), '--source-sha256', digest(source), '--source-size', str(source.stat().st_size),
            '--config', str(config_path), '--config-sha256', digest(config_path),
            '--preprocessor', str(processor_path), '--preprocessor-sha256', digest(processor_path),
            '--source-model', upstream['model'], '--revision', upstream['revision'],
            '--output', str(output), '--provenance', str(provenance)])
        extract.authority(args)
        for field, value, message in (
            ('source_sha256', '0'*64, 'source SHA256'), ('source_size', 1, 'source size'),
            ('source_sha256', 'bad', 'SHA256'), ('source_size', 0, 'source size'),
            ('config_sha256', '0'*64, 'config SHA256'),
            ('preprocessor_sha256', '0'*64, 'preprocessor SHA256'),
            ('revision', 'main', 'revision'), ('source_model', 'google/wrong', 'source model')):
            changed = copy.copy(args)
            setattr(changed, field, value)
            rejects(lambda: extract.authority(changed), message)
        def header_check():
            with source.open('rb') as stream:
                return extract.read_header(stream)
        rejects(header_check, 'header')
        # These catch byte-count, overlap, hole, malformed-offset and duplicate
        # tensor bugs without importing safetensors or copying any weights.
        def put_header(header, payload=b'\0'*8):
            encoded = json.dumps(header).encode() if isinstance(header, dict) else header
            source.write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
        small = {'text_model.a': {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]},
                 'text_model.b': {'dtype': 'F32', 'shape': [1], 'data_offsets': [4, 8]}}
        put_header(small)
        assert header_check()[0] == small
        for change in ({'shape': [2]}, {'shape': [-1]}, {'shape': [True]},
                       {'dtype': 'UNKNOWN'}, {'data_offsets': [0]},
                       {'data_offsets': []}, {'data_offsets': [False, 4]},
                       {'data_offsets': [0, 3]}, {'data_offsets': 'bad'},
                       {'data_offsets': [None, 4]}, {'unexpected': 1}):
            changed = copy.deepcopy(small)
            changed['text_model.a'].update(change)
            put_header(changed)
            rejects(header_check, 'tensor')
        for offsets in ([0, 4], [5, 9], [-4, 0]):
            changed = copy.deepcopy(small)
            changed['text_model.b']['data_offsets'] = offsets
            put_header(changed)
            rejects(header_check, 'tensor')
        put_header(small, b'\0'*9)
        rejects(header_check, 'payload')
        put_header({'__metadata__': {'format': 1}, **small})
        rejects(header_check, 'metadata')
        put_header(b'{"a":{},"a":{}}')
        rejects(header_check, 'duplicate')
        put_header(small)
        alias = base / 'alias'
        alias.symlink_to(source)
        changed = copy.copy(args)
        changed.source = alias
        rejects(lambda: extract.authority(changed), 'canonical')
        assert not output.exists() and not provenance.exists()
        for target in (output, provenance):
            target.write_bytes(b'preserved')
            rejects(lambda: extract.authority(args), 'output already exists')
            assert target.read_bytes() == b'preserved'
            target.unlink()
            target.symlink_to(base / 'missing')
            rejects(lambda: extract.authority(args), 'output already exists')
            target.unlink()
        changed = copy.copy(args)
        changed.provenance = changed.output
        rejects(lambda: extract.authority(changed), 'distinct')
        changed = copy.copy(args)
        changed.output = changed.source
        rejects(lambda: extract.authority(changed), 'distinct')
        output.write_bytes(b'preserved')
        rejects(lambda: extract.exclusive(output), 'exists')
        # Simulate another writer winning after the precheck. The real open
        # must still refuse and preserve that writer's bytes.
        with patch.object(extract, 'new_output', lambda _: None):
            rejects(lambda: extract.exclusive(output), 'exists')
        assert output.read_bytes() == b'preserved'

        # Reject duplicate JSON keys before a parser can silently discard them.
        config_path.write_text('{"vision_config":{},"vision_config":{}}')
        rejects(lambda: extract.authenticated_json(config_path, digest(config_path), 'config'), 'duplicate')
        processor_path.write_text('{"bad":NaN}')
        rejects(lambda: extract.authenticated_json(processor_path, digest(processor_path), 'preprocessor'), 'JSON')
        rejects(lambda: extract.validate_preprocessor({'size': {'height': 384, 'width': 384}}), 'preprocessor')

    result = subprocess.run([sys.executable, '-B', '-S', '-O', str(path), '--help'], capture_output=True, text=True)
    assert result.returncode != 0 and 'optimized' in result.stderr
    assert 'torch' not in sys.modules and 'safetensors' not in sys.modules
    print('PASS: stdlib source, inventory, authority, exclusive-output and -O negatives; no extraction')


if __name__ == '__main__':
    main()
