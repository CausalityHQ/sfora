#!/usr/bin/env python3
"""Descriptive control-vs-candidate layer-26 MLP displacement read; stdlib only.

Reads exactly the four absolute FP32 ``root['encoder']`` storages of the control and
candidate portable ``endpoint.pt`` files through the pinned pass201 restricted checkpoint
metadata helper (loaded from SHA-pinned bytes, never modified or broadened). No Torch, NumPy,
model, image, GPU or vision.pt read. Reports norms only: no threshold, verdict or GO.

Order: authority FILE -> authority schema -> own/helper source pins -> helper API -> inputs
(whole-file streamed fresh SHA first, then only the selected storage members) -> complete
uncached rehash of every source/input -> exclusive atomic publication. Any error: no result.

Terminal contract: only exit status 0 plus the single stdout line {"output","sha256","bytes"} that matches
the published file is a normal terminal. An output file left behind by a nonzero exit (for example an
fsync/cleanup failure after the hard link, when best-effort retraction also fails) is diagnostic-only and
must be rejected. The result carries measurements only: no threshold, verdict or GO.
"""
import argparse
import array
import dataclasses
import hashlib
import inspect
import json
import math
import operator
import os
from pathlib import Path
import re
import secrets
import stat
import struct
import sys
import time
import types
import zipfile

HELPER_MODULE = '_pinned_pass201_checkpoint_helper'
AUTHORITY_SCHEMA = 'sfora-connected-mlp-displacement-authority-v1'
RESULT_SCHEMA = 'sfora-connected-mlp-displacement-result-v1'
BUNDLE_SCHEMA = 'siglip2-connected-mlp-bundle-v1'
INFERENCE_SCHEMA = 'siglip2-connected-mlp-inference-v1'
ARMS = ('control', 'candidate')
MLP = tuple('encoder.layers.26.mlp.' + layer + '.' + field for layer in ('fc1', 'fc2') for field in ('weight', 'bias'))
SHAPES = dict(zip(MLP, ((4304, 1152), (4304,), (1152, 4304), (1152,))))
INFERENCE_KEYS = {'schema', 'source', 'arm', 'config', 'buffers', 'processor', 'head', 'A', 'C', 'means',
                  'mu_train', 'mu_train_provenance', 'scope', 'common_statistics', 'numerical_flags',
                  'base_vision', 'encoder', 'encoder_identity', 'vision_sha256', 'fixed_sha256'}
BUNDLE_KEYS = {'schema', 'code', 'files', 'endpoint_state_sha256', 'environment', 'encoder_identity',
               'base_vision_sha256', 'vision_sha256', 'scope'}
BUNDLE_FILES = {'vision.pt', 'endpoint.pt', 'processor.json'}
HEX64 = re.compile('[0-9a-f]{64}')
SECONDS = 120
MAX_AUTHORITY, MAX_SOURCE, MAX_BUNDLE, MAX_ENDPOINT, MAX_INTERPRETER = 1 << 20, 8 << 20, 8 << 20, 1 << 30, 1 << 30
HELPER_API = {'_read_checkpoint_data_pickle': ['path'], '_load_restricted_pickle': ['data'],
              '_safe_zip_member_name': ['name']}
HELPER_FIELDS = {'_TensorStub': ('storage', 'storage_offset', 'size', 'stride'),
                 '_StorageStub': ('storage_class', 'key', 'location', 'size'),
                 '_StorageClassStub': ('global_name',)}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def identity(st):
    return (st.st_dev, st.st_ino, st.st_mode, st.st_size, st.st_mtime_ns, st.st_ctime_ns)


def canonical(value, what):
    require(type(value) is str and value.startswith('/'), what + ': absolute path required')
    path = Path(value)
    try:
        require(path.resolve(strict=True) == path, what + ': canonical path required (no symlink/..)')
    except OSError as error:
        raise ValueError(what + ': cannot resolve path') from error
    return path


def read_regular(path, limit, keep=False):
    """Fresh open, streamed SHA-256, identity before/after; no cache. -> (identity, sha256, bytes|None)."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError as error:
        raise ValueError('cannot open regular file: ' + str(path)) from error
    before = os.fstat(fd)
    if not (stat.S_ISREG(before.st_mode) and before.st_size <= limit):
        os.close(fd)
        raise ValueError('regular file within size limit required: ' + str(path))
    with os.fdopen(fd, 'rb') as stream:
        digest, chunks, count = hashlib.sha256(), [], 0
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
            count += len(chunk)
            if keep:
                chunks.append(chunk)
        require(identity(os.fstat(fd)) == identity(before) and count == before.st_size, 'file changed during read: ' + str(path))
    require(identity(os.stat(path, follow_symlinks=False)) == identity(before), 'file changed during read: ' + str(path))
    return identity(before), digest.hexdigest(), b''.join(chunks) if keep else None


def strict_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def constant(name):
        raise ValueError('non-finite JSON constant')
    try:
        return json.loads(data.decode('utf-8'), object_pairs_hook=pairs, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError('invalid JSON') from error


def keys(value, expected, what):
    require(type(value) is dict and set(value) == set(expected), what + ': exact keys required')
    return value


def sha(value, what):
    require(type(value) is str and HEX64.fullmatch(value), what + ': lowercase SHA-256 required')
    return value


def pin(value, what):
    keys(value, {'path', 'sha256'}, what)
    return {'path': canonical(value['path'], what + '.path'), 'sha256': sha(value['sha256'], what + '.sha256')}


def parse_authority(obj):
    keys(obj, {'schema', 'own_source_sha256', 'helper_source', 'arms'}, 'authority')
    require(obj['schema'] == AUTHORITY_SCHEMA, 'authority schema differs')
    keys(obj['arms'], ARMS, 'authority.arms')
    arms = {}
    for arm in ARMS:
        keys(obj['arms'][arm], {'bundle', 'endpoint'}, 'authority.arms.' + arm)
        arms[arm] = {kind: pin(obj['arms'][arm][kind], 'authority.arms.%s.%s' % (arm, kind)) for kind in ('bundle', 'endpoint')}
    paths = [arms[a][k]['path'] for a in ARMS for k in ('bundle', 'endpoint')]
    require(len(set(paths)) == 4, 'control/candidate bundle and endpoint files must be four distinct paths')
    return {'own_source_sha256': sha(obj['own_source_sha256'], 'authority.own_source_sha256'),
            'helper': pin(obj['helper_source'], 'authority.helper_source'), 'arms': arms}


def release_helper(module):
    """Remove only the registry entry this run created; a foreign replacement is preserved and rejected."""
    current = sys.modules.get(HELPER_MODULE)
    if current is module:
        del sys.modules[HELPER_MODULE]
    elif current is not None:
        raise ValueError('helper module registry entry was replaced by a foreign module; left in place')


def load_helper(path, data):
    """Execute the already hash-verified helper bytes; the file itself is never imported."""
    require(HELPER_MODULE not in sys.modules, 'helper module name is already registered; nothing overwritten')
    module = types.ModuleType(HELPER_MODULE)
    module.__file__ = str(path)
    sys.modules[HELPER_MODULE] = module
    try:
        exec(compile(data, str(path), 'exec'), module.__dict__)
    except BaseException:
        release_helper(module)
        raise
    return module


def check_api(helper):
    for name, parameters in HELPER_API.items():
        function = getattr(helper, name, None)
        require(callable(function) and list(inspect.signature(function).parameters) == parameters, 'helper API differs: ' + name)
    for name, fields in HELPER_FIELDS.items():
        cls = getattr(helper, name, None)
        require(isinstance(cls, type) and dataclasses.is_dataclass(cls) and
                tuple(f.name for f in dataclasses.fields(cls)) == fields, 'helper API differs: ' + name)


def require_little_endian():
    probe = array.array('f', struct.pack('<2f', 1.5, -2.0))
    require(sys.byteorder == 'little' and array.array('f').itemsize == 4 and list(probe) == [1.5, -2.0],
            'little-endian IEEE-754 float32 host required')


def typed_fingerprint(shape, raw_sha256):
    """Pinned substrate-adaptation fingerprint of one CPU float32 tensor, from its raw facts."""
    def frame(text):
        raw = text.encode()
        return str(len(raw)).encode() + b':' + raw
    parts = [frame('Tensor'), frame('tuple'), frame('3'), frame('str'), frame(repr('torch.float32')),
             frame('tuple'), frame(str(len(shape)))]
    for dimension in shape:
        parts += [frame('int'), frame(repr(dimension))]
    parts += [frame('str'), frame(repr(raw_sha256))]
    return hashlib.sha256(b''.join(parts)).hexdigest()


def contiguous(shape):
    strides, step = [], 1
    for dimension in reversed(shape):
        strides.append(step)
        step *= dimension
    return tuple(reversed(strides))


def parse_bundle(obj, arm, endpoint_pin):
    keys(obj, BUNDLE_KEYS, arm + ' bundle')
    require(obj['schema'] == BUNDLE_SCHEMA, arm + ' bundle schema differs')
    files = keys(obj['files'], BUNDLE_FILES, arm + ' bundle.files')
    for name in BUNDLE_FILES:
        sha(files[name], arm + ' bundle.files.' + name)
    require(files['endpoint.pt'] == endpoint_pin, arm + ' bundle endpoint.pt SHA differs from authority pin')
    ident = obj['encoder_identity']
    require(type(ident) is dict and type(ident.get('initial_four_sha256')) is dict and type(ident.get('inventory')) is dict,
            arm + ' bundle encoder_identity differs')
    initial = keys(ident['initial_four_sha256'], MLP, arm + ' bundle initial_four_sha256')
    for name in MLP:
        sha(initial[name], arm + ' bundle initial_four_sha256')
        require(ident['inventory'].get(name) == list(SHAPES[name]), arm + ' bundle inventory shape differs')
    return {'initial': dict(initial), 'base_vision_sha256': sha(obj['base_vision_sha256'], arm + ' base_vision_sha256'),
            'vision_sha256': sha(obj['vision_sha256'], arm + ' vision_sha256'), 'vision_pt_sha256': files['vision.pt']}


class Endpoint:
    """One endpoint.pt: whole-file fresh SHA first, then only data.pkl, byteorder and the four selected storages."""

    def __init__(self, helper, path, pin_sha, arm, bundle):
        self.path, self.arm, self._zip, self._file = path, arm, None, None
        self.identity, self.sha256, _ = read_regular(path, MAX_ENDPOINT)
        require(self.sha256 == pin_sha, arm + ' endpoint SHA-256 differs from pin')
        self.size = self.identity[3]
        try:
            self._file = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC), 'rb')
            require(identity(os.fstat(self._file.fileno())) == self.identity, arm + ' endpoint changed after hash')
            self._zip = zipfile.ZipFile(self._file)
            infos = self._zip.infolist()
        except (zipfile.BadZipFile, zipfile.LargeZipFile, OSError, EOFError, NotImplementedError, RuntimeError) as error:
            self.close()
            raise ValueError(arm + ' endpoint is not a readable ZIP') from error
        try:
            self._open(helper, infos, bundle)
        except BaseException:
            self.close()
            raise

    def close(self):
        for resource in (self._zip, self._file):
            if resource is not None:
                resource.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def _open(self, helper, infos, bundle):
        arm = self.arm
        names = [info.filename for info in infos]
        require(len(names) <= 100_000 and len(set(names)) == len(names), arm + ' ZIP has duplicate or too many members')
        require(all(helper._safe_zip_member_name(n) for n in names), arm + ' ZIP unsafe member name')
        require(not any(info.flag_bits & 1 for info in infos), arm + ' ZIP encrypted member')
        require(all(stat.S_IFMT(info.external_attr >> 16) in (0, stat.S_IFREG) for info in infos), arm + ' ZIP non-regular member')
        self._by_name = dict(zip(names, infos))
        pickles = [n for n in names if n == 'data.pkl' or n.endswith('/data.pkl')]
        require(len(pickles) == 1 and pickles[0].count('/') == 1, arm + ' ZIP needs exactly one <archive>/data.pkl')
        self.prefix = pickles[0].split('/')[0]
        require(self._read(self.prefix + '/byteorder', 6) == b'little', arm + ' checkpoint byteorder must be little')
        data = helper._read_checkpoint_data_pickle(self.path)
        require(data == self._read(self.prefix + '/data.pkl'), arm + ' metadata pickle differs between helper and hashed file')
        root = helper._load_restricted_pickle(data)
        require(type(root) is dict and set(root) == INFERENCE_KEYS, arm + ' endpoint root keys differ')
        require(root['schema'] == INFERENCE_SCHEMA and root['arm'] == arm, arm + ' endpoint schema/arm differs')
        require(root['vision_sha256'] == bundle['vision_sha256'], arm + ' endpoint vision_sha256 differs from bundle')
        require(type(root['base_vision']) is dict and root['base_vision'].get('sha256') == bundle['base_vision_sha256'],
                arm + ' endpoint base_vision differs from bundle')
        require(type(root['encoder_identity']) is dict and root['encoder_identity'].get('initial_four_sha256') == bundle['initial'],
                arm + ' endpoint initial_four_sha256 differs from bundle')
        encoder = root['encoder']
        require(type(encoder) is dict and set(encoder) == set(MLP) and len(encoder) == 4, arm + ' encoder must be exactly the four selected tensors')
        self.members = {}
        for name in MLP:
            tensor, shape = encoder[name], SHAPES[name]
            require(type(tensor) is helper._TensorStub and type(tensor.storage) is helper._StorageStub, arm + ' ' + name + ': tensor metadata differs')
            storage = tensor.storage
            require(storage.storage_class.global_name == 'torch.FloatStorage' and storage.location == 'cpu', arm + ' ' + name + ': CPU FloatStorage required')
            require(tensor.size == shape and tensor.stride == contiguous(shape), arm + ' ' + name + ': shape/stride differ')
            # offset 0 is implied by the span check plus the helper's bounds check; kept explicit for the contract
            require(tensor.storage_offset == 0 and storage.size == math.prod(shape), arm + ' ' + name + ': storage span is not exactly the tensor')
            require(re.fullmatch('[0-9]{1,9}', storage.key), arm + ' ' + name + ': storage key differs')
            self.members[name] = self.prefix + '/data/' + storage.key
        require(len(set(self.members.values())) == 4, arm + ' selected tensors share a storage')

    def _read(self, member, expected=None):
        info = self._by_name.get(member)
        require(info is not None, self.arm + ' ZIP member missing: ' + member)
        size = info.file_size
        if expected is None:
            require(size <= 64 << 20, self.arm + ' ZIP member too large: ' + member)
        else:
            require(size == expected and info.compress_type == zipfile.ZIP_STORED and info.compress_size == size,
                    self.arm + ' ZIP member size/compression differs: ' + member)
        try:
            with self._zip.open(info) as source:
                raw = source.read(size + 1)
        except (zipfile.BadZipFile, EOFError, OSError, NotImplementedError, RuntimeError) as error:
            raise ValueError(self.arm + ' ZIP member unreadable or CRC differs: ' + member) from error
        require(len(raw) == size, self.arm + ' ZIP member size differs: ' + member)
        return raw

    def storage(self, name):
        return self._read(self.members[name], 4 * math.prod(SHAPES[name]))


def sum_squares(values, what):
    try:
        total = math.fsum(map(operator.mul, values, values))
    except (ArithmeticError, ValueError) as error:
        raise ValueError(what + ': non-finite or overflowing values') from error
    require(math.isfinite(total), what + ': non-finite values')
    return total


def norm_row(control_ss, candidate_ss, difference_ss):
    control = math.sqrt(control_ss)
    difference = math.sqrt(difference_ss)
    return {'control_norm': control, 'candidate_norm': math.sqrt(candidate_ss), 'difference_norm': difference,
            'relative_l2': None if control_ss == 0.0 else difference / control, 'baseline_zero': control_ss == 0.0}


def measure(control, candidate, initial, deadline):
    tensors, totals = {}, ([], [], [])
    for name in MLP:
        raw = {'control': control.storage(name), 'candidate': candidate.storage(name)}
        values = {arm: array.array('f', raw[arm]) for arm in ARMS}
        sums = (sum_squares(values['control'], 'control ' + name), sum_squares(values['candidate'], 'candidate ' + name),
                sum_squares(array.array('d', map(operator.sub, values['candidate'], values['control'])), 'difference ' + name))
        raw_sha = {arm: hashlib.sha256(raw[arm]).hexdigest() for arm in ARMS}
        fingerprints = {arm: typed_fingerprint(SHAPES[name], raw_sha[arm]) for arm in ARMS}
        require(fingerprints['control'] == initial[name], 'control ' + name + ' differs from the guarded initial typed hash')
        for total, value in zip(totals, sums):
            total.append(value)
        tensors[name] = {'shape': list(SHAPES[name]), 'numel': math.prod(SHAPES[name]),
                         'control_raw_sha256': raw_sha['control'], 'candidate_raw_sha256': raw_sha['candidate'],
                         'control_matches_initial': True, 'candidate_matches_initial': fingerprints['candidate'] == initial[name],
                         **norm_row(*sums)}
        require(time.monotonic() < deadline, 'reader exceeded the %ss budget' % SECONDS)
    return tensors, {'numel': sum(t['numel'] for t in tensors.values()), **norm_row(*(math.fsum(t) for t in totals))}


def early_output(value):
    require(type(value) is str and value.startswith('/') and value.endswith('.json'), '--output: absolute .json path required')
    path = Path(value)
    require(path.parent.resolve(strict=True) == path.parent and path.parent.is_dir() and not os.path.lexists(path),
            '--output: new file in an existing canonical directory required')
    return path


def retract(output, linked):
    """Best effort: remove the just-linked output iff it is still our inode; a survivor is diagnostic-only."""
    try:
        if linked is not None:
            now = os.stat(output, follow_symlinks=False)
            if (now.st_dev, now.st_ino) == linked:
                os.unlink(output)
    except OSError:
        pass


def publish(output, payload):
    """Exclusive atomic publication: fully written + fsynced temp, hard-linked into place.

    Any failure after the link (directory fsync, temp cleanup) retracts the output best effort and still
    fails the run, so only exit status 0 plus the stdout line is a normal terminal."""
    temp = output.parent / ('.%s.%d.%s.tmp' % (output.name, os.getpid(), secrets.token_hex(8)))
    directory = os.open(output.parent, os.O_RDONLY | os.O_DIRECTORY)
    linked = None
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o444)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        mine = os.stat(temp, follow_symlinks=False)
        try:
            os.link(temp, output)
        except FileExistsError as error:
            raise ValueError('--output already exists') from error
        linked = (mine.st_dev, mine.st_ino)
        os.fsync(directory)
        os.unlink(temp)
    except BaseException:
        retract(output, linked)
        raise
    finally:
        try:
            os.unlink(temp)
        except OSError:
            pass
        os.close(directory)


def parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], allow_abbrev=False)
    parser.add_argument('--authority', required=True)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--output', required=True)
    return parser.parse_args(argv)


def run(argv, own, interpreter):
    args = parse_args(argv)
    deadline = time.monotonic() + SECONDS
    output = early_output(args.output)
    # 1. authority FILE, 2. schema -- before any other code runs.
    authority_path = canonical(args.authority, '--authority')
    sha(args.authority_sha256, '--authority-sha256')
    authority_id, authority_sha, authority_bytes = read_regular(authority_path, MAX_AUTHORITY, keep=True)
    require(authority_sha == args.authority_sha256, 'authority SHA-256 differs from --authority-sha256')
    authority = parse_authority(strict_json(authority_bytes))
    # 3. own/helper source pins, 4. helper API -- helper code only runs once its bytes match the pin.
    # Own source, interpreter and helper are admitted only as canonical regular files (no symlink component).
    own = canonical(str(own), 'own source')
    own_id, own_sha, _ = read_regular(own, MAX_SOURCE)
    require(own_sha == authority['own_source_sha256'], 'own source SHA-256 differs from authority pin')
    interpreter = canonical(str(interpreter), 'interpreter (launch with the canonical python path)')
    interpreter_id, interpreter_sha, _ = read_regular(interpreter, MAX_INTERPRETER)
    helper_path = authority['helper']['path']
    helper_id, helper_sha, helper_bytes = read_regular(helper_path, MAX_SOURCE, keep=True)
    require(helper_sha == authority['helper']['sha256'], 'helper source SHA-256 differs from authority pin')
    require_little_endian()
    helper = load_helper(helper_path, helper_bytes)
    try:
        check_api(helper)
        records = {'authority': (authority_path, authority_id, authority_sha), 'own_source': (own, own_id, own_sha),
                   'interpreter': (interpreter, interpreter_id, interpreter_sha), 'helper_source': (helper_path, helper_id, helper_sha)}
        bundles = {}
        for arm in ARMS:
            pins = authority['arms'][arm]
            bundle_id, bundle_sha, bundle_bytes = read_regular(pins['bundle']['path'], MAX_BUNDLE, keep=True)
            require(bundle_sha == pins['bundle']['sha256'], arm + ' bundle SHA-256 differs from pin')
            records[arm + '_bundle'] = (pins['bundle']['path'], bundle_id, bundle_sha)
            bundles[arm] = parse_bundle(strict_json(bundle_bytes), arm, pins['endpoint']['sha256'])
        for field in ('initial', 'base_vision_sha256', 'vision_pt_sha256'):
            require(bundles['control'][field] == bundles['candidate'][field], 'control/candidate bundles differ in ' + field)
        with Endpoint(helper, authority['arms']['control']['endpoint']['path'], authority['arms']['control']['endpoint']['sha256'],
                      'control', bundles['control']) as control, \
                Endpoint(helper, authority['arms']['candidate']['endpoint']['path'], authority['arms']['candidate']['endpoint']['sha256'],
                         'candidate', bundles['candidate']) as candidate:
            for endpoint in (control, candidate):
                records[endpoint.arm + '_endpoint'] = (endpoint.path, endpoint.identity, endpoint.sha256)
            tensors, combined = measure(control, candidate, bundles['control']['initial'], deadline)
    finally:
        release_helper(helper)
    # Complete uncached rehash of every source/input before anything is published.
    for label, (path, before, digest) in records.items():
        limit = MAX_ENDPOINT if label.endswith('_endpoint') else MAX_INTERPRETER if label == 'interpreter' else MAX_SOURCE
        again = read_regular(path, limit)
        require(again[:2] == (before, digest), label + ' changed before publication')
    require(time.monotonic() < deadline, 'reader exceeded the %ss budget' % SECONDS)
    result = {'schema': RESULT_SCHEMA, 'descriptive_only': True,
              'inputs': {label: {'path': str(path), 'sha256': digest, 'bytes': before[3]}
                         for label, (path, before, digest) in records.items()},
              'base_vision_sha256': bundles['control']['base_vision_sha256'], 'tensors': tensors, 'combined': combined}
    payload = (json.dumps(result, sort_keys=True, indent=1, allow_nan=False) + '\n').encode()
    publish(output, payload)
    return {'output': str(output), 'sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)}


def main(argv=None, *, own=None, interpreter=None):
    try:
        terminal = run(sys.argv[1:] if argv is None else argv, Path(own or __file__).absolute(), Path(interpreter or sys.executable))
        print(json.dumps(terminal, sort_keys=True), flush=True)
    except (ValueError, OSError) as error:
        print('error: %s' % error, file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (SECONDS, SECONDS))
    raise SystemExit(main())
