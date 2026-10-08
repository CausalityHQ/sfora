#!/usr/bin/env python3
"""Engineering-only observation of ONE genuine frozen v6 full-CPU authority call.

CLI: python -B observe_connected_full_authority.py --manifest FILE --manifest-sha256 SHA
Exact manifest connected-full-authority-observation-v1: schema, observer, test,
python, evaluator, execution, authority, command (each FILE={path,sha256}),
original_argv, argv, output, diagnostic, unit, resource_policy, both_locks_held,
qualification_eligible=false, state_reuse_eligible=false. The fixed original argv
may change ONLY its output to the declared never-created exclusive output path.
The parent supplies actual observer/test/manifest hashes and fresh paths/unit.
resource_policy={seconds:900,host_bytes:8589934592,swap_bytes:0,cuda_visible_devices:''}
is a declaration: the parent's original terminal enforces/proves it and both locks.

JSONL connected-full-authority-observation-event-v1 records START, SAMPLE,
AUTHORITY_RETURN or AUTHORITY_ERROR, OBSERVATION_ERROR on observer failure, END
only after clean observation. Every row has scalar monotonic `seconds` and false
qualification_eligible/state_reuse_eligible. SAMPLE has ordinal, classification
(innermost original evaluator function, or outside_original_evaluator), frames
({filename,function,line}, innermost first), frames_truncated. These are samples,
never exact phase boundaries or CPU attribution. Return adds only guard_count
and invocation_count, not a context. Errors have bounded scalar exception chains.
Limits: interval .25s, 3600 samples, 64 frames/sample, 16MiB JSONL, 2s thread join.
Sampler failures invalidate observation at unwind; they never replace an original
authority exception. Timeout may leave partial JSONL: END is not a terminal
normal-exit claim. No qualification receipt, output directory, native start,
model/image/quality work, repeated authority call or repeated guard scan.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import threading
import time

SCHEMA = 'connected-full-authority-observation-v1'
ROOT = '/home/riomus/runs/sfora-connected-mlp-evaluation-source-v6/'
EVALUATOR = {'path': ROOT + 'evaluate_siglip2_connected_mlp.py',
             'sha256': '2390c60fe5e87e82ab122c5c0101476b378792541bc454470c37d6c7f0410d40'}
EVALUATOR_TEST = {'path': ROOT + 'test_connected_mlp_evaluation.py',
                  'sha256': '28d0297cebd6a7bab395a44bc60a4aa1bcaacb9160a8093044a54fa270b104d9'}
EXECUTION = {'path': ROOT + 'execution.json',
             'sha256': 'a0c1f27db4da404e7777d89518dfc83e2e20b607fcc1031776978e9ca8ec9f3c'}
AUTHORITY = {'path': ROOT + 'authority-full-cpu-v2.json',
             'sha256': '3444df504430a2ee92f33fac04cfed609c512de542d8993f08069dd4bd24e19d'}
COMMAND = {'path': ROOT + 'full-cpu-v2-command.sh',
           'sha256': '8998b7abb9ad2cbbb119aff15b722578845b39d42ae179c8a183c3000fa0e1f7'}
PYTHON = {'path': '/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13',
          'sha256': '9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b'}
PYTHON_VERSION = '3.13.9 (main, Oct 14 2025, 21:26:54) [Clang 20.1.4 ]'
OLD_OUTPUT = '/home/riomus/runs/sfora-connected-mlp-evaluation-full-cpu-v2'
POLICY = {'seconds': 900, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0, 'cuda_visible_devices': ''}
NATIVE = frozenset(('torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'))
MODULE = '_connected_full_authority_v6'
INTERVAL, MAX_SAMPLES, MAX_BYTES = 0.25, 3600, 16 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    require(type(value) is str, 'path string required')
    path = Path(value)
    require(path.is_absolute() and str(path) == value and path.resolve() == path and not path.is_symlink(),
            'canonical absolute nonsymlink path required: ' + value)
    return path


def file_bytes(fact, *, keep=True):
    require(type(fact) is dict and fact.keys() == {'path', 'sha256'} and type(fact['sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', fact['sha256']), 'exact FILE pin required')
    path = canonical(fact['path'])
    require(path.is_file(), 'regular FILE required')
    with path.open('rb') as stream:
        raw = stream.read(2 * 1024**2 + 1) if keep else None
        digest = hashlib.sha256(raw).hexdigest() if keep else hashlib.file_digest(stream, 'sha256').hexdigest()
    require((raw is None or len(raw) <= 2 * 1024**2) and digest == fact['sha256'], 'FILE size/SHA256 differs: ' + str(path))
    return raw


def strict_json(raw):
    def pairs(items):
        result = dict(items)
        require(len(result) == len(items), 'duplicate JSON key')
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def original_argv():
    return [EVALUATOR['path'], '--execution-sha256', EXECUTION['sha256'], '--authority', AUTHORITY['path'],
            '--authority-sha256', AUTHORITY['sha256'], '--phase', 'cpu', '--output', OLD_OUTPUT]


def no_native():
    require(not any(name.split('.')[0] in NATIVE for name in sys.modules), 'native root imported during source observation')


def prepare(path, digest):
    manifest = strict_json(file_bytes({'path': str(path), 'sha256': digest}))
    require(type(manifest) is dict and manifest.keys() == {'schema', 'observer', 'test', 'python', 'evaluator',
        'execution', 'authority', 'command', 'original_argv', 'argv', 'output', 'diagnostic', 'unit',
        'resource_policy', 'both_locks_held', 'qualification_eligible', 'state_reuse_eligible'} and
        manifest['schema'] == SCHEMA and manifest['qualification_eligible'] is False and
        manifest['state_reuse_eligible'] is False and manifest['both_locks_held'] is True,
        'exact engineering-only manifest required')
    require(manifest['resource_policy'] == POLICY and
            all(type(manifest['resource_policy'][k]) is type(v) for k, v in POLICY.items()), 'fixed diagnostic policy required')
    for key, pin in (('python', PYTHON), ('evaluator', EVALUATOR), ('execution', EXECUTION),
                     ('authority', AUTHORITY), ('command', COMMAND)):
        require(manifest[key] == pin, 'original ' + key + ' pin differs')
    require(canonical(manifest['observer']['path']) == Path(__file__).absolute() and
            prepare.__code__.co_filename == str(Path(__file__).absolute()) and
            getattr(__loader__, 'path', None) == str(Path(__file__).absolute()) and
            (__spec__ is None or __spec__.origin == str(Path(__file__).absolute())), 'observer source origin differs')
    require(canonical(manifest['test']['path']) == Path(__file__).absolute().with_name('test_connected_full_authority_observation.py'),
            'observer test sibling required')
    for key in ('observer', 'test', 'command'):
        file_bytes(manifest[key])
    require(str(Path(sys.executable).resolve()) == PYTHON['path'] and sys.version == PYTHON_VERSION,
            'original interpreter path/version differs')
    file_bytes(PYTHON, keep=False)
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.gettrace() is None and
            sys.getprofile() is None and os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'untraced -B CPU-only startup required')
    no_native()
    require(strict_json(file_bytes(EXECUTION)) == {Path(p['path']).name: p['sha256'] for p in (EVALUATOR, EVALUATOR_TEST)},
            'exact original evaluator2 closure required')
    file_bytes(EVALUATOR_TEST)
    launch = strict_json(file_bytes(AUTHORITY))
    require(launch['phase'] == 'cpu' and launch['stage'] == 'full' and launch['panel'] == 'selection', 'original full CPU required')
    output, diagnostic = canonical(manifest['output']), canonical(manifest['diagnostic'])
    require(manifest['original_argv'] == original_argv() and
            manifest['argv'] == original_argv()[:-1] + [str(output)], 'exact original argv with only fresh output required')
    roots = [Path(EVALUATOR['path']).parent, Path(OLD_OUTPUT)] + [canonical(launch[k]['root']) for k in
             ('training', 'evaluator_reference', 'nearest_evaluator', 'genuine_evaluator', 'reference')]
    bound = [canonical(str(path)), *[canonical(manifest[k]['path']) for k in
             ('observer', 'test', 'python', 'evaluator', 'execution', 'authority', 'command')], canonical(EVALUATOR_TEST['path'])]
    for new in (output, diagnostic):
        require(not new.exists() and new.parent.is_dir() and all(not new.is_relative_to(p) and not p.is_relative_to(new)
                for p in roots + bound), 'unused exclusive path outside original sources/output required')
    require(not output.is_relative_to(diagnostic) and not diagnostic.is_relative_to(output), 'diagnostic/output overlap')
    unit = manifest['unit']
    require(type(unit) is str and re.fullmatch('[A-Za-z0-9_.@-]+', unit) and not unit.endswith('.service') and
            unit != Path(OLD_OUTPUT).name, 'fresh parent unit required')
    groups = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    require(len(groups) == 1 and Path(groups[0]).name == unit + '.service' and
            re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'parent unit/invocation differs')
    return manifest, file_bytes(EVALUATOR)


def exception_chain(error):
    """Scalar copies only; original exception/traceback is still re-raised unchanged."""
    result, seen = [], set()
    while error is not None and id(error) not in seen and len(result) < 8:
        seen.add(id(error))
        frames, tb = [], error.__traceback__
        while tb is not None and len(frames) < 16:
            frames.append({'filename': tb.tb_frame.f_code.co_filename[:512],
                           'function': tb.tb_frame.f_code.co_name[:128], 'line': tb.tb_lineno})
            tb = tb.tb_next
        result.append({'type': type(error).__module__ + '.' + type(error).__name__, 'message': str(error)[:2048],
                       'frames': frames, 'frames_truncated': tb is not None,
                       'link': 'cause' if error.__cause__ is not None else 'context',
                       'context_suppressed': error.__suppress_context__})
        error = error.__cause__ if error.__cause__ is not None else (None if error.__suppress_context__ else error.__context__)
    return {'chain': result, 'truncated': error is not None}


class Sampler:
    def __init__(self, filename, stream):
        self.filename, self.stream = filename, stream
        self.stop = threading.Event()
        self.thread = None
        self.failure = None
        self.count = self.bytes = 0

    def emit(self, event, **fields):
        raw = (json.dumps({'schema': 'connected-full-authority-observation-event-v1', 'event': event,
            'seconds': time.monotonic(), 'qualification_eligible': False, 'state_reuse_eligible': False,
            **fields}, separators=(',', ':'), allow_nan=False) + '\n').encode()
        limit = MAX_BYTES - 65536 if event == 'SAMPLE' else MAX_BYTES
        require(self.bytes + len(raw) <= limit, 'diagnostic byte limit reached')
        self.bytes += len(raw)
        require(self.stream.write(raw) == len(raw), 'short diagnostic write')
        self.stream.flush()

    def sample(self, ident):
        frame = None
        try:
            frame = sys._current_frames().get(ident)
            require(frame is not None, 'authority thread frame unavailable')
            frames, depth = [], 0
            while frame is not None and depth < 64:
                if frame.f_code.co_filename == self.filename:
                    frames.append({'filename': self.filename, 'function': frame.f_code.co_name, 'line': frame.f_lineno})
                frame, depth = frame.f_back, depth + 1
            self.emit('SAMPLE', ordinal=self.count + 1, frames=frames, frames_truncated=frame is not None,
                      classification=frames[0]['function'] if frames else 'outside_original_evaluator')
            self.count += 1
        finally:
            frame = None

    def loop(self, ident):
        try:
            while not self.stop.wait(INTERVAL):
                require(self.count < MAX_SAMPLES, 'sample limit reached')
                self.sample(ident)
        except BaseException as error:
            self.failure = exception_chain(error)
            self.stop.set()

    def start(self):
        self.thread = threading.Thread(target=self.loop, args=(threading.get_ident(),),
                                       name='connected-authority-sampler', daemon=True)
        self.thread.start()

    def finish(self):
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2)
            require(not self.thread.is_alive(), 'sampler did not join')
        require(self.failure is None, 'sampler failed: ' + json.dumps(self.failure))


class DenyNative:
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in NATIVE:
            # Original package_origins queries specs without importing packages.
            for finder in sys.meta_path:
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    spec.loader = NativeExecutionDenied(fullname)
                    return spec
        return None


class NativeExecutionDenied:
    def __init__(self, fullname):
        self.fullname = fullname

    def create_module(self, spec):
        raise ImportError('native import forbidden in authority observation: ' + self.fullname)

    def exec_module(self, module):
        raise ImportError('native import forbidden in authority observation: ' + self.fullname)


def observe(raw, argv, sampler):
    """Load unchanged bytes at their original filename; call authority exactly once."""
    no_native()
    require(MODULE not in sys.modules, 'original evaluator module already loaded')
    saved = sys.argv, sys.path, list(sys.meta_path)
    module = context = original_guard = None
    error = None
    counts = None
    ended = None
    problems = []
    try:
        sampler.emit('START', argv=argv)
        sys.meta_path.insert(0, DenyNative())
        sys.argv, sys.path = list(argv), [str(Path(argv[0]).parent), *sys.path[1:]]
        spec = importlib.util.spec_from_file_location(MODULE, argv[0])
        module = importlib.util.module_from_spec(spec)
        sys.modules[MODULE] = module
        exec(compile(raw, argv[0], 'exec', dont_inherit=True), vars(module))
        no_native()
        args = module.parser().parse_args(argv[1:])
        require(module.cli(args) == argv and module.authority.__code__.co_filename == argv[0] and
                module.__file__ == module.__spec__.origin == argv[0], 'original source/CLI identity differs')
        sampler.start()
        try:
            context, original_guard = module.authority(args)
        finally:
            ended = time.monotonic()
            sampler.stop.set()
        counts = {'guard_count': len(context['guards']),
                  'invocation_count': len(context['training_context']['legacy']['invocations'])}
    except BaseException as original:
        error = original
        raise
    finally:
        context = original_guard = None
        try:
            sampler.finish()
        except BaseException as secondary:
            problems.append(exception_chain(secondary))
        try:
            no_native()
            require(not Path(argv[-1]).exists() and not Path(argv[-1]).is_symlink(), 'authority created forbidden output')
        except BaseException as secondary:
            problems.append(exception_chain(secondary))
        sys.argv, sys.path = saved[:2]
        sys.meta_path[:] = saved[2]
        if module is not None:
            sys.modules.pop(MODULE, None)
        module = None
        try:
            if error is not None:
                sampler.emit('AUTHORITY_ERROR', authority_stopped_seconds=ended, exception=exception_chain(error))
            if problems:
                sampler.emit('OBSERVATION_ERROR', failures=problems)
            if error is None and not problems:
                sampler.emit('AUTHORITY_RETURN', authority_stopped_seconds=ended, **counts)
                sampler.emit('END', sample_count=sampler.count, native_roots_imported=[])
        except BaseException as secondary:
            problems.append(exception_chain(secondary))
        if problems:
            if error is not None:
                error.add_note('Observation also failed: ' + json.dumps(problems))
            else:
                raise RuntimeError('Observation failed: ' + json.dumps(problems))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    manifest, raw = prepare(args.manifest, args.manifest_sha256)
    # Exclusive creation preserves every prior diagnostic, including partial failures.
    with canonical(manifest['diagnostic']).open('xb', buffering=0) as stream:
        observe(raw, manifest['argv'], Sampler(EVALUATOR['path'], stream))


if __name__ == '__main__':
    main()
