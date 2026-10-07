#!/usr/bin/env python3
"""Diagnostic-only coarse timing of the exact frozen connected CPUv2 source.

python -B observe_connected_mlp_phases.py --manifest FILE --manifest-sha256 SHA
Manifest connected-mlp-phase-observer-v1 has exactly schema, wrapper, python,
trainer, execution, authority (FILE={path:canonical_absolute_file,sha256:actual64}),
argv, output, unit, stdio={stdout:absolute_path,stderr:absolute_path}, and
qualification_eligible=false/state_reuse_eligible=false. Parent supplies actual
wrapper/manifest hashes and fresh output/unit/stdio, freezes the enclosing CLI,
and owns both locks, CPU300/8GiB/no swap/CUDA hidden, terminal and cleanup.
Phase times are inclusive, overlapping and not additive. END means a Python
return event, including exception unwind; only SOURCE_END means normal source
completion. Timeout/error may leave SOURCE_BEGIN without SOURCE_END. Copied
bundle functions remain untraced; no file opens occur in the profile callback.
"""
import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import re
import sys
import time
from types import BuiltinFunctionType, ModuleType

SCHEMA = 'connected-mlp-phase-observer-v1'
ROOT = '/home/riomus/runs/sfora-connected-mlp-train-source-v2/'
TRAINER = {'path': ROOT + 'train_siglip2_connected_mlp.py',
           'sha256': 'cd68f9b109ca48dfc488a248b66c6e2f4a1e58e884aa3c598f4325d4c912edb2'}
EXECUTION = {'path': ROOT + 'execution.json',
             'sha256': 'cb18b71dad340f49964679bf47b14449e546ab654e67d4d823256e4f212234b0'}
AUTHORITY = {'path': ROOT + 'authority-cpu-v2.json',
             'sha256': '56f37121ce44dee2f305961d540ce592b32d0e938c229b65f62d8c627f47c779'}
PYTHON = {'path': '/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13',
          'sha256': '9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b'}
PYTHON_VERSION = '3.13.9 (main, Oct 14 2025, 21:26:54) [Clang 20.1.4 ]'
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora', '_lsprof'}
PHASES = frozenset(('authority', 'fresh', 'load_initializer', 'construct_encoder', 'update',
    'integrity', 'encoder_facts', 'fingerprint', 'save', 'restore', 'export_bundle', 'qualify_bundle',
    'load_inference', 'inference_outputs', 'portable_mutants', 'release', 'serving_environment', 'run'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    require(isinstance(value, str), 'path must be a string')
    path = Path(value)
    require(path.is_absolute() and str(path) == value and path.resolve() == path and not path.is_symlink(),
            'canonical absolute non-symlink path required: ' + value)
    return path


def file_bytes(binding):
    require(isinstance(binding, dict) and binding.keys() == {'path', 'sha256'} and
            isinstance(binding['sha256'], str) and re.fullmatch('[0-9a-f]{64}', binding['sha256']),
            'exact FILE binding required')
    path = canonical(binding['path'])
    require(path.is_file(), 'regular FILE required: ' + str(path))
    with path.open('rb') as stream:
        raw = stream.read(2 * 1024**2 + 1)
    require(len(raw) <= 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == binding['sha256'],
            'FILE size/SHA256 differs: ' + str(path))
    return raw


def unique_pairs(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), 'duplicate JSON key')
    return result


def builtin_origins():
    for module, names in ((sys, ('setprofile', 'getprofile')), (time, ('perf_counter',))):
        spec = getattr(module, '__spec__', None)
        require(sys.modules.get(module.__name__) is module and getattr(module, '__file__', None) is None and
                spec is not None and spec.origin == 'built-in' and
                spec.loader is importlib.machinery.BuiltinImporter, 'builtin module origin differs')
        for name in names:
            function = getattr(module, name)
            require(isinstance(function, BuiltinFunctionType) and function.__module__ == module.__name__ and
                    function.__self__ is module, 'builtin observer function origin differs')


def prepare(manifest_path, manifest_sha):
    manifest = json.loads(file_bytes({'path': str(manifest_path), 'sha256': manifest_sha}),
        object_pairs_hook=unique_pairs, parse_constant=lambda value: require(False, 'nonfinite JSON: ' + value))
    require(isinstance(manifest, dict) and manifest.keys() == {'schema', 'wrapper', 'python', 'trainer',
            'execution', 'authority', 'argv', 'output', 'unit', 'stdio', 'qualification_eligible',
            'state_reuse_eligible'} and manifest['schema'] == SCHEMA and
            manifest['qualification_eligible'] is False and manifest['state_reuse_eligible'] is False,
            'exact diagnostic-only manifest required')
    require(all(manifest[name] == pinned for name, pinned in
            (('python', PYTHON), ('trainer', TRAINER), ('execution', EXECUTION), ('authority', AUTHORITY))),
            'frozen interpreter/trainer/execution/authority differs')
    file_bytes(manifest['wrapper'])
    wrapper, target = canonical(manifest['wrapper']['path']), canonical(TRAINER['path'])
    require(wrapper == Path(__file__).absolute() and prepare.__code__.co_filename == str(wrapper) and
            getattr(__loader__, 'path', None) == str(wrapper) and
            (__spec__ is None or __spec__.origin == str(wrapper)) and
            not wrapper.is_relative_to(target.parent), 'current wrapper source/module origin differs')
    argv, output = manifest['argv'], canonical(manifest['output'])
    require(isinstance(argv, list) and all(isinstance(a, str) for a in argv) and argv == [str(target),
            '--execution-sha256', EXECUTION['sha256'], '--authority', AUTHORITY['path'],
            '--authority-sha256', AUTHORITY['sha256'], '--phase', 'cpu', '--arm', 'control',
            '--seed', '179061', '--output', str(output)], 'canonical unchanged CPUv2 argv required')
    require(not output.exists() and not output.is_relative_to(target.parent) and
            not output.is_relative_to(wrapper.parent), 'unused separate output required')
    require(isinstance(manifest['stdio'], dict) and manifest['stdio'].keys() == {'stdout', 'stderr'},
            'exact parent stdio required')
    bound = {str(wrapper), str(target), str(manifest_path), PYTHON['path'], AUTHORITY['path'], EXECUTION['path']}
    require(not any(Path(value).is_relative_to(output) for value in bound), 'output overlaps bound FILE')
    for value in manifest['stdio'].values():
        log = canonical(value)
        require(str(log) not in bound and log != output and not log.is_relative_to(output) and
                not log.is_relative_to(target.parent), 'separate parent stdio required')
    builtin_origins()
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None and
            not any(name.split('.')[0] in NATIVE for name in sys.modules) and
            'cProfile' not in sys.modules and 'profile' not in sys.modules,
            'unoptimized -B source-only startup without profiler required')
    require(str(Path(sys.executable).resolve()) == PYTHON['path'] and sys.version == PYTHON_VERSION,
            'native interpreter path/version differs')
    with canonical(PYTHON['path']).open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == PYTHON['sha256'], 'interpreter SHA256 differs')
    unit = manifest['unit']
    require(isinstance(unit, str) and re.fullmatch('[A-Za-z0-9_.@-]+', unit) and
            not unit.endswith('.service') and unit != 'sfora-connected-mlp-cpu-v2', 'new parent unit required')
    groups = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    require(len(groups) == 1 and Path(groups[0]).name == unit + '.service', 'actual parent unit differs')
    invocation = os.environ.get('INVOCATION_ID', '')
    require(re.fullmatch('[0-9a-f]{32}', invocation), 'actual runtime INVOCATION_ID required')
    file_bytes(EXECUTION)
    file_bytes(AUTHORITY)
    raw = file_bytes(TRAINER)
    return manifest, raw, invocation


class PhaseObserver:
    """Only integer frame IDs and clock values survive a callback."""
    def __init__(self, filename, stream):
        self.filename, self.stream = filename, stream
        self.active = {}
        self.count = self.bytes = 0
        self.max_events, self.max_bytes = 2000, 1024**2
        self.stopped = False
        self.marker = self.line({'event': 'TRUNCATED'})

    @staticmethod
    def line(event):
        return json.dumps({'schema': 'connected-mlp-phase-observer-event-v1', 'qualification_eligible': False,
            'state_reuse_eligible': False, **event}, separators=(',', ':'), ensure_ascii=True) + '\n'

    def emit(self, event):
        if self.stopped:
            return
        line = self.line(event)
        size = len(line.encode('utf-8'))
        if self.count >= self.max_events - 1 or self.bytes + size + len(self.marker) > self.max_bytes:
            line, size = self.marker, len(self.marker)
            self.stopped = True
            self.active.clear()
        try:
            self.stream.write(line)
            self.stream.flush()
            self.count += 1
            self.bytes += size
        except (OSError, ValueError):
            self.stopped = True
            self.active.clear()  # Diagnostic output failure must not replace a target exception.

    def __call__(self, frame, event, arg):
        if self.stopped or event not in ('call', 'return'):
            return
        code = frame.f_code
        if code.co_filename != self.filename or code.co_name not in PHASES:
            return
        frame_id, tick = id(frame), time.perf_counter()
        row = {'event': 'PHASE', 'filename': code.co_filename, 'function': code.co_name,
               'line': code.co_firstlineno, 'frame_id': frame_id, 'seconds': tick}
        if event == 'call':
            self.active[frame_id] = tick
            self.emit({**row, 'boundary': 'BEGIN'})
        else:
            started = self.active.pop(frame_id, None)
            if started is not None:
                self.emit({**row, 'boundary': 'END', 'inclusive_seconds': tick - started,
                           'end_semantics': 'return_or_unwind'})


def observe_target(observer, raw, argv, context):
    target = Path(argv[0])
    code = compile(raw, str(target), 'exec', dont_inherit=True)
    del raw
    main = ModuleType('__main__')
    main.__file__, main.__package__, main.__spec__, main.__cached__ = str(target), None, None, None
    main.__loader__ = importlib.machinery.SourceFileLoader('__main__', str(target))
    saved = sys.argv, sys.path, sys.modules['__main__'], sys.getprofile()
    setprofile = sys.setprofile
    tick = time.perf_counter()
    observer.emit({**context, 'event': 'SOURCE_BEGIN', 'seconds': tick})
    sys.argv, sys.path, sys.modules['__main__'] = list(argv), [str(target.parent), *sys.path[1:]], main
    try:
        setprofile(observer)
        exec(code, vars(main), vars(main))
    finally:
        setprofile(saved[3])
        sys.argv, sys.path, sys.modules['__main__'] = saved[:3]
        observer.active.clear()
    ended = time.perf_counter()
    observer.emit({'event': 'SOURCE_END', 'seconds': ended, 'inclusive_seconds': ended - tick})


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    manifest, raw, invocation = prepare(args.manifest, args.manifest_sha256)
    context = {'manifest_sha256': args.manifest_sha256, 'invocation_id': invocation, 'unit': manifest['unit'],
        'trainer_sha256': TRAINER['sha256'], 'execution_sha256': EXECUTION['sha256'],
        'authority_sha256': AUTHORITY['sha256'], 'wrapper_sha256': manifest['wrapper']['sha256'],
        'output': manifest['output']}
    observe_target(PhaseObserver(TRAINER['path'], sys.stdout), raw, manifest['argv'], context)


if __name__ == '__main__':
    main()
