#!/usr/bin/env python3
"""Extract LOCAL full SigLIP2 bytes; extraction is not model qualification.

Required CLI hashes/size/revision are the caller's frozen source authority.
Output keys strip exactly ``vision_model.`` for the parent's installed
SiglipVisionModel with direct embeddings/encoder/post_layernorm/head modules.
Other Transformers layouts require separate qualification, never auto-remap.

safe_open (imported only after stdlib preflight) validates metadata using its
full-file mmap. get_slice only inspects shape/dtype; get_tensor, Torch, model
construction and save_file/serialize_file's full-state dictionary are avoided.
Copy/hash uses one 1MiB buffer, never even one complete tensor. Header JSON is
capped at 8MiB. mmap virtual size can equal the whole source; these APIs give
NO hard RSS/cgroup ceiling. Consumed-range POSIX_FADV_DONTNEED is advisory;
the parent's 120s/8GiB/noSwap unit, both locks and whole-unit peaks remain
mandatory. Parent owns final input rehash and source/model qualification.
Numeric finiteness/runtime/nonpersistent buffers are later model checks;
here tensor validity means dtype/shape/byte-range and exact byte provenance.

Provenance schema siglip2-vision-source-extraction-v1 records input paths,
hashes/size, source model/revision, raw config/processor objects, resolved
vision defaults, every source tensor's original/derived key, shape, dtype,
offsets and serialized-byte SHA256, plus output whole-file SHA256/size.
No success receipt is written until output readback and input guards pass.
Failure may leave incomplete NEW files; they must never be reused/overwritten.
"""
if not __debug__:
    raise SystemExit('Extraction requires assertions; optimized mode is forbidden')

import argparse
import hashlib
import json
import math
import os
import re
import resource
import stat
import struct
import time
from pathlib import Path

SCHEMA = 'siglip2-vision-source-extraction-v1'
PREFIX = 'vision_model.'
PROFILES = {'google/siglip2-large-patch16-256': (1024, 24, 4096),
            'google/siglip2-so400m-patch16-256': (1152, 27, 4304)}
DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'U16': 2, 'I16': 2,
               'U32': 4, 'I32': 4, 'U64': 8, 'I64': 8,
               'F16': 2, 'BF16': 2, 'F32': 4, 'F64': 8}
CHUNK = 1024**2
HEADER_CAP = 8 * CHUNK


def strict_json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('duplicate JSON key: ' + key)
            result[key] = value
        return result

    def constant(value):
        raise ValueError('invalid JSON constant: ' + value)

    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(value, dict):
        raise ValueError('JSON object required')
    return value


def digest_string(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('lowercase SHA256 required')
    return value


def stream_range(stream, count, whole=None, output=None):
    """Hash/copy exactly count bytes and advise away only consumed file ranges."""
    digest, buffer = hashlib.sha256(), bytearray(CHUNK)
    while count:
        offset = stream.tell()
        read = stream.readinto(memoryview(buffer)[:min(count, CHUNK)])
        if not read:
            raise ValueError('truncated tensor/file bytes')
        view = memoryview(buffer)[:read]
        digest.update(view)
        if whole is not None:
            whole.update(view)
        if output is not None:
            output.write(view)
        os.posix_fadvise(stream.fileno(), offset, read, os.POSIX_FADV_DONTNEED)
        count -= read
    return digest.hexdigest()


def sha(path):
    with Path(path).open('rb') as stream:
        return stream_range(stream, os.fstat(stream.fileno()).st_size)


def authenticated_json(path, expected, label):
    digest_string(expected)
    with path.open('rb') as stream:
        raw = stream.read(CHUNK + 1)
    if len(raw) > CHUNK:
        raise ValueError(label + ' JSON too large')
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(label + ' SHA256 differs')
    return strict_json(raw)


def expected_vision(config, model):
    if model not in PROFILES:
        raise ValueError('unsupported source model')
    vision = config.get('vision_config')
    if config.get('model_type') != 'siglip' or not isinstance(vision, dict):
        raise ValueError('wrong source config profile')
    width, layers, intermediate = PROFILES[model]
    required = {'hidden_size': width, 'num_hidden_layers': layers,
                'intermediate_size': intermediate, 'image_size': 256}
    defaults = {'patch_size': 16, 'num_channels': 3, 'num_attention_heads': 16}
    if any(type(vision.get(k)) is not int or vision[k] != v for k, v in required.items()) or any(
            type(vision.get(k, v)) is not int or vision.get(k, v) != v for k, v in defaults.items()) or (
            vision.get('model_type', 'siglip_vision_model') != 'siglip_vision_model' or
            vision.get('vision_use_head', True) is not True):
        raise ValueError('source vision config profile differs')
    resolved = {**required, **defaults, 'model_type': 'siglip_vision_model', 'vision_use_head': True}
    expected = {}

    def linear(name, rows, columns):
        expected[PREFIX + name + '.weight'] = [rows, columns]
        expected[PREFIX + name + '.bias'] = [rows]

    def norm(name):
        expected[PREFIX + name + '.weight'] = [width]
        expected[PREFIX + name + '.bias'] = [width]

    expected[PREFIX + 'embeddings.patch_embedding.weight'] = [width, 3, 16, 16]
    expected[PREFIX + 'embeddings.patch_embedding.bias'] = [width]
    expected[PREFIX + 'embeddings.position_embedding.weight'] = [256, width]
    for index in range(layers):
        layer = f'encoder.layers.{index}.'
        for name in ('q_proj', 'k_proj', 'v_proj', 'out_proj'):
            linear(layer + 'self_attn.' + name, width, width)
        norm(layer + 'layer_norm1')
        norm(layer + 'layer_norm2')
        linear(layer + 'mlp.fc1', intermediate, width)
        linear(layer + 'mlp.fc2', width, intermediate)
    norm('post_layernorm')
    expected[PREFIX + 'head.probe'] = [1, 1, width]
    expected[PREFIX + 'head.attention.in_proj_weight'] = [3 * width, width]
    expected[PREFIX + 'head.attention.in_proj_bias'] = [3 * width]
    linear('head.attention.out_proj', width, width)
    norm('head.layernorm')
    linear('head.mlp.fc1', intermediate, width)
    linear('head.mlp.fc2', width, intermediate)
    return expected, resolved


def validate_preprocessor(value):
    required = {'image_processor_type': 'SiglipImageProcessor', 'do_resize': True,
                'size': {'height': 256, 'width': 256}, 'resample': 2,
                'do_rescale': True, 'rescale_factor': 1 / 255,
                'do_normalize': True, 'image_mean': [0.5]*3, 'image_std': [0.5]*3}
    if any(value.get(k) != v for k, v in required.items()):
        raise ValueError('preprocessor profile differs')


def validate_vision(inventory, expected):
    selected = {k for k in inventory if k.startswith(PREFIX)}
    if selected != expected.keys():
        missing, extra = sorted(expected.keys() - selected), sorted(selected - expected.keys())
        raise ValueError(f'vision inventory differs: missing={missing}, extra={extra}')
    for key in selected:
        entry = inventory[key]
        if entry['shape'] != expected[key] or entry['dtype'] not in ('F16', 'BF16', 'F32'):
            raise ValueError('vision tensor differs: ' + key)
    if not {'logit_scale', 'logit_bias'} <= inventory.keys() or not any(
            key.startswith('text_model.') for key in inventory):
        raise ValueError('full source text/logit tensors required')
    if any(not (k.startswith((PREFIX, 'text_model.')) or k in ('logit_scale', 'logit_bias')) for k in inventory):
        raise ValueError('unexpected source key')
    return {key: key[len(PREFIX):] for key in selected}


def read_header(stream):
    size = os.fstat(stream.fileno()).st_size
    stream.seek(0)
    length_bytes = stream.read(8)
    if len(length_bytes) != 8:
        raise ValueError('truncated safetensors header')
    length = struct.unpack('<Q', length_bytes)[0]
    if not 2 <= length <= min(HEADER_CAP, size - 8):
        raise ValueError('invalid safetensors header length')
    raw = stream.read(length)
    if len(raw) != length or not raw.startswith(b'{'):
        raise ValueError('invalid safetensors header')
    header = strict_json(raw)
    metadata = header.pop('__metadata__', {})
    if not isinstance(metadata, dict) or any(not isinstance(v, str) for v in metadata.values()):
        raise ValueError('invalid safetensors metadata')
    for key, entry in header.items():
        if not key or not isinstance(entry, dict) or entry.keys() != {'dtype', 'shape', 'data_offsets'}:
            raise ValueError('invalid tensor entry: ' + key)
        shape, offsets, dtype = entry['shape'], entry['data_offsets'], entry['dtype']
        if not isinstance(shape, list) or any(type(n) is not int or n <= 0 for n in shape) or (
                not isinstance(dtype, str) or dtype not in DTYPE_BYTES) or (
                not isinstance(offsets, list) or len(offsets) != 2 or any(type(n) is not int for n in offsets)):
            raise ValueError('invalid tensor shape/dtype/offsets: ' + key)
        start, end = offsets
        if start < 0 or end < start or end - start != math.prod(shape) * DTYPE_BYTES[dtype]:
            raise ValueError('invalid tensor byte range: ' + key)
    cursor = 0
    for key, entry in sorted(header.items(), key=lambda item: item[1]['data_offsets'][0]):
        start, end = entry['data_offsets']
        if start != cursor:
            raise ValueError('tensor overlap/hole: ' + key)
        cursor = end
    if not header or cursor != size - 8 - length:
        raise ValueError('tensor ranges do not cover complete payload')
    return header, metadata, length_bytes + raw


def new_output(path):
    if not path.is_absolute() or path.parent.resolve() != path.parent or not path.parent.is_dir():
        raise ValueError('output requires an existing canonical absolute parent')
    if path.exists() or path.is_symlink():
        raise FileExistsError('output already exists: ' + str(path))


def exclusive(path):
    new_output(path)
    return path.open('x+b')


def authority(args):
    paths = [args.source, args.config, args.preprocessor, args.output, args.provenance]
    if len({str(p) for p in paths}) != len(paths):
        raise ValueError('source/config/preprocessor/output/provenance paths must be distinct')
    new_output(args.output)
    new_output(args.provenance)
    if args.source_model not in PROFILES:
        raise ValueError('unsupported source model')
    if not re.fullmatch('[0-9a-f]{40}', args.revision):
        raise ValueError('immutable 40-hex revision required')
    for path in paths[:3]:
        if not path.is_absolute() or path.resolve() != path or not path.is_file():
            raise ValueError('local canonical regular file required: ' + str(path))
    if type(args.source_size) is not int or args.source_size <= 0 or args.source.stat().st_size != args.source_size:
        raise ValueError('source size differs')
    digest_string(args.source_sha256)
    if sha(args.source) != args.source_sha256:
        raise ValueError('source SHA256 differs')
    config = authenticated_json(args.config, args.config_sha256, 'config')
    processor = authenticated_json(args.preprocessor, args.preprocessor_sha256, 'preprocessor')
    expected, resolved = expected_vision(config, args.source_model)
    validate_preprocessor(processor)
    return config, processor, expected, resolved


def native_metadata(path, inventory):
    # Import only here: help and all stdlib preflight negatives need no packages.
    import safetensors
    with safetensors.safe_open(str(path), framework='np') as source:
        if set(source.keys()) != inventory.keys():
            raise ValueError('safetensors key inventory differs')
        for key, entry in inventory.items():
            tensor = source.get_slice(key)
            if tensor.get_shape() != entry['shape'] or tensor.get_dtype() != entry['dtype']:
                raise ValueError('safetensors tensor metadata differs: ' + key)
    return safetensors.__version__


def signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def extract(args):
    started = time.perf_counter()
    config, processor, expected, resolved = authority(args)
    with args.source.open('rb') as source:
        initial = signature(os.fstat(source.fileno()))
        if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
            raise ValueError('source must be a regular file')
        inventory, metadata, raw = read_header(source)
        mapping = validate_vision(inventory, expected)
        # Hash every tensor and full archive together before writing anything.
        whole = hashlib.sha256(raw)
        rows = []
        for key, entry in sorted(inventory.items(), key=lambda item: item[1]['data_offsets'][0]):
            start, end = entry['data_offsets']
            digest = stream_range(source, end - start, whole=whole)
            rows.append({'original_key': key, 'derived_key': mapping.get(key),
                         **entry, 'sha256': digest, 'derived_data_offsets': None})
        if whole.hexdigest() != args.source_sha256 or signature(os.fstat(source.fileno())) != initial:
            raise ValueError('source SHA256/identity changed during inventory')
        version = native_metadata(Path(f'/proc/self/fd/{source.fileno()}'), inventory)
        output_header, cursor = {'__metadata__': {'format': 'pt'}}, 0
        for row in rows:
            if row['derived_key'] is not None:
                length = row['data_offsets'][1] - row['data_offsets'][0]
                row['derived_data_offsets'] = [cursor, cursor + length]
                output_header[row['derived_key']] = {'dtype': row['dtype'], 'shape': row['shape'],
                                                      'data_offsets': row['derived_data_offsets']}
                cursor += length
        encoded = json.dumps(output_header, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        encoded += b' ' * (-len(encoded) % 8)
        with exclusive(args.output) as output, exclusive(args.provenance) as receipt:
            output.write(struct.pack('<Q', len(encoded)) + encoded)
            for row in rows:
                if row['derived_key'] is not None:
                    start, end = row['data_offsets']
                    source.seek(len(raw) + start)
                    if stream_range(source, end - start, output=output) != row['sha256']:
                        raise ValueError('source tensor changed during copy: ' + row['original_key'])
            output.flush()
            os.fsync(output.fileno())
            derived, _, output_raw = read_header(output)
            if derived != {k: v for k, v in output_header.items() if k != '__metadata__'}:
                raise ValueError('output inventory differs')
            output_whole = hashlib.sha256(output_raw)
            for row in rows:
                if row['derived_key'] is not None:
                    start, end = row['derived_data_offsets']
                    if stream_range(output, end - start, whole=output_whole) != row['sha256']:
                        raise ValueError('output tensor readback differs: ' + row['derived_key'])
            native_metadata(Path(f'/proc/self/fd/{output.fileno()}'), derived)
            if signature(os.fstat(source.fileno())) != initial or signature(args.source.stat()) != initial:
                raise ValueError('source identity changed at exit')
            if sha(args.config) != args.config_sha256 or sha(args.preprocessor) != args.preprocessor_sha256:
                raise ValueError('config/preprocessor changed at exit')
            value = {'schema': SCHEMA, 'pass': True, 'model_qualified': False, 'source_model': args.source_model,
                     'revision': args.revision, 'source': {'path': str(args.source), 'sha256': args.source_sha256,
                                                         'size_bytes': args.source_size, 'metadata': metadata},
                     'config': {'path': str(args.config), 'sha256': args.config_sha256, 'value': config},
                     'preprocessor': {'path': str(args.preprocessor), 'sha256': args.preprocessor_sha256, 'value': processor},
                     'resolved_vision_inventory_config': resolved,
                     'output': {'path': str(args.output), 'sha256': output_whole.hexdigest(),
                                'size_bytes': os.fstat(output.fileno()).st_size, 'tensor_count': len(expected),
                                'payload_bytes': cursor, 'key_mapping': 'strip exactly vision_model. for installed direct SiglipVisionModel'},
                     'source_tensor_count': len(rows), 'tensors': rows, 'safetensors_version': version,
                     'cache_advice': 'consumed-range POSIX_FADV_DONTNEED; advisory, no RSS guarantee',
                     'copy_chunk_bytes': CHUNK, 'header_cap_bytes': HEADER_CAP,
                     'main_wall_seconds': time.perf_counter() - started,
                     'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
            receipt.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
            receipt.flush()
            os.fsync(receipt.fileno())
    return value


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'config', 'preprocessor', 'output', 'provenance'):
        result.add_argument('--' + name, type=Path, required=True)
    for name in ('source-sha256', 'config-sha256', 'preprocessor-sha256', 'revision'):
        result.add_argument('--' + name, required=True)
    result.add_argument('--source-size', type=int, required=True)
    result.add_argument('--source-model', choices=sorted(PROFILES), required=True)
    return result


def main():
    args = parser().parse_args()
    try:
        value = extract(args)
    except (OSError, ValueError, ImportError) as error:
        raise SystemExit('extraction rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'output': value['output'], 'provenance': str(args.provenance)}, sort_keys=True))


if __name__ == '__main__':
    main()
