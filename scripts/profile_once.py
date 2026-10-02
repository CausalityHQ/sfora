#!/usr/bin/env python3
"""One diagnostic profile of unchanged control061 mechanics; parent owns launch/admission.

python -B profile_once.py --manifest FILE --manifest-sha256 SHA
Exact manifest schema quadratic-profile-once-v1: wrapper/python/trainer FILEs,
stdlib={cProfile.py:FILE,profile.py:FILE}, argv, profile, unit,
stdio={stdout:absolute_path,stderr:absolute_path}, qualification_eligible=false,
state_reuse_eligible=false. FILE={path:canonical_absolute_path,sha256:lowercase64}.
The parent separately freezes the command/manifest/footers/locks/resources and
collects the actual runtime INVOCATION_ID, terminal, log and profile hashes.
Neither this diagnostic unit nor its receipt/state is a qualification input.
"""
import argparse
import hashlib
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from types import ModuleType


SCHEMA = 'quadratic-profile-once-v1'
EXECUTION_SHA = 'a12ac8fe3ec0363db2c89243e633666f2485a3097cd9c249b4ec27c9d5db86b3'
AUTHORITY_SHA = 'eb87cdc10ca73d0418e0feb759c9204ffbcbcffa34c196a73891c6e94fccd05e'
PYTHON = {'path': '/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/bin/python3.13',
          'sha256': '9258c53dcfde55ba0d0ba9dfdb03bd3f0f30328dc1950f0275f32929fa879b6b'}
PYTHON_VERSION = '3.13.9 (main, Oct 14 2025, 21:26:54) [Clang 20.1.4 ]'
STDLIB_ROOT = '/home/riomus/.local/share/uv/python/cpython-3.13.9-linux-aarch64-gnu/lib/python3.13/'
STDLIB = {'cProfile.py': {'path': STDLIB_ROOT + 'cProfile.py',
                        'sha256': '7834ef85dff0d2563a968da9d1a61d9c36687820b47a557fcb8ad577b71aef52'},
          'profile.py': {'path': STDLIB_ROOT + 'profile.py',
                        'sha256': '20de245bd1ac2702ee67d064584ed9153b2ea6df8597fb696a02952e34173877'}}
TRAINER = {'path': '/home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py',
           'sha256': 'f5759274d4c8ba75d69f6728b9b309e0cafeaba80ed7127fce6393099730e17b'}
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.parent.resolve() == path.parent and not path.is_symlink(),
            'canonical absolute path required: ' + str(path))
    return path


def file_bytes(binding):
    require(isinstance(binding, dict) and binding.keys() == {'path', 'sha256'} and
            isinstance(binding['sha256'], str) and re.fullmatch('[0-9a-f]{64}', binding['sha256']),
            'exact FILE binding required')
    path = canonical(binding['path'])
    with path.open('rb') as stream:
        raw = stream.read(2 * 1024**2 + 1)
    require(len(raw) <= 2 * 1024**2 and hashlib.sha256(raw).hexdigest() == binding['sha256'],
            'source size/SHA256 differs: ' + str(path))
    return raw


def unique_pairs(pairs):
    result = dict(pairs)
    require(len(result) == len(pairs), 'duplicate JSON key')
    return result


def require_builtin_profiler():
    require('_lsprof' in sys.builtin_module_names, 'builtin _lsprof required; no extension admission')
    spec = importlib.util.find_spec('_lsprof')
    require(spec is not None and spec.origin == 'built-in' and spec.loader is importlib.machinery.BuiltinImporter,
            'builtin _lsprof provenance differs')
    module = sys.modules.get('_lsprof')
    require(module is None or (getattr(module, '__file__', None) is None and
            getattr(getattr(module, '__spec__', None), 'origin', None) == 'built-in'),
            'loaded _lsprof provenance differs')
    mappings = Path('/proc/self/maps').read_text()
    require(not any('_lsprof' in line and '.so' in line for line in mappings.splitlines()),
            'mapped _lsprof extension forbidden')


def prepare(manifest_path, manifest_sha):
    manifest = json.loads(file_bytes({'path': str(manifest_path), 'sha256': manifest_sha}),
                          object_pairs_hook=unique_pairs,
                          parse_constant=lambda value: require(False, 'nonfinite JSON: ' + value))
    require(isinstance(manifest, dict) and manifest.keys() == {'schema', 'wrapper', 'python', 'stdlib', 'trainer',
            'argv', 'profile', 'unit', 'stdio', 'qualification_eligible', 'state_reuse_eligible'} and
            manifest['schema'] == SCHEMA and manifest['qualification_eligible'] is False and
            manifest['state_reuse_eligible'] is False, 'exact diagnostic manifest required')
    require(manifest['python'] == PYTHON and manifest['stdlib'] == STDLIB and manifest['trainer'] == TRAINER,
            'pinned native interpreter/stdlib/trainer provenance differs')
    wrapper = canonical(manifest['wrapper']['path'])
    target = canonical(TRAINER['path'])
    require(wrapper == Path(__file__).absolute() and wrapper.parent != target.parent and
            not wrapper.is_relative_to(target.parent), 'separate authenticated wrapper required')
    file_bytes(manifest['wrapper'])
    argv = manifest['argv']
    require(isinstance(argv, list) and len(argv) == 15 and all(isinstance(a, str) for a in argv),
            'canonical trainer argv required')
    authority, output = canonical(argv[4]), canonical(argv[14])
    require(argv == [str(target), '--execution-sha256', EXECUTION_SHA, '--authority', str(authority),
            '--authority-sha256', AUTHORITY_SHA, '--phase', 'mechanics', '--arm', 'control',
            '--seed', '179061', '--output', str(output)], 'unchanged canonical control061 mechanics CLI required')
    require(not authority.is_relative_to(target.parent) and not output.is_relative_to(target.parent) and
            not output.is_relative_to(wrapper.parent) and not output.exists(), 'new separate authority/output required')
    file_bytes({'path': str(authority), 'sha256': AUTHORITY_SHA})
    stats = canonical(manifest['profile'])
    require(not stats.exists() and not stats.is_relative_to(target.parent) and not stats.is_relative_to(output),
            'unused separate profile required')
    require(isinstance(manifest['stdio'], dict) and manifest['stdio'].keys() == {'stdout', 'stderr'},
            'parent stdio paths required')
    bound_paths = {str(wrapper), str(target), str(authority), str(stats), str(manifest_path),
                   PYTHON['path'], *(entry['path'] for entry in STDLIB.values())}
    for value in manifest['stdio'].values():
        log = canonical(value)
        require(str(log) not in bound_paths and not log.is_relative_to(target.parent) and
                not log.is_relative_to(output), 'separate parent stdio required')
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None and
            not any(n.split('.')[0] in NATIVE for n in sys.modules) and
            'profile' not in sys.modules and 'cProfile' not in sys.modules,
            'unoptimized -B source-only startup without profiling required')
    require(str(Path(sys.executable).resolve()) == PYTHON['path'] and sys.version == PYTHON_VERSION,
            'native interpreter path/version differs')
    with canonical(PYTHON['path']).open('rb') as stream:
        require(hashlib.file_digest(stream, 'sha256').hexdigest() == PYTHON['sha256'], 'native interpreter SHA256 differs')
    require(isinstance(manifest['unit'], str) and re.fullmatch('[A-Za-z0-9_.@-]+', manifest['unit']) and
            not manifest['unit'].endswith('.service'), 'fresh parent unit name required')
    groups = [line[3:] for line in Path('/proc/self/cgroup').read_text().splitlines() if line.startswith('0::')]
    require(len(groups) == 1 and Path(groups[0]).name == manifest['unit'] + '.service', 'actual parent unit differs')
    invocation = os.environ.get('INVOCATION_ID', '')
    require(re.fullmatch('[0-9a-f]{32}', invocation), 'actual runtime INVOCATION_ID required')
    require_builtin_profiler()
    raw = file_bytes(TRAINER)
    sources = {name: file_bytes(binding) for name, binding in STDLIB.items()}
    return manifest, raw, sources, invocation


def load_profiler(sources):
    for name in ('profile', 'cProfile'):
        path = STDLIB[name + '.py']['path']
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        # Execute the bytes just authenticated; neither read nor write a pyc.
        exec(compile(sources[name + '.py'], path, 'exec'), vars(module))
    return module.Profile()


def profile_target(profiler, raw, argv, stats):
    target = Path(argv[0])
    require(not stats.exists() and not stats.is_symlink(), 'profile already exists')
    code = compile(raw, str(target), 'exec')
    main = ModuleType('__main__')
    main.__file__, main.__package__, main.__spec__, main.__cached__ = str(target), None, None, None
    main.__loader__ = importlib.machinery.SourceFileLoader('__main__', str(target))
    saved = sys.argv, sys.path, sys.modules['__main__']
    sys.argv, sys.path, sys.modules['__main__'] = list(argv), [str(target.parent), *sys.path[1:]], main
    try:
        # Direct method only: the cProfile convenience _Utils path swallows SystemExit.
        profiler.runctx(code, vars(main), vars(main))
    finally:
        target_error = sys.exception()
        sys.argv, sys.path, sys.modules['__main__'] = saved
        try:
            with tempfile.TemporaryDirectory(prefix='.profile-once-', dir=stats.parent) as directory:
                temporary = Path(directory) / 'stats'
                profiler.dump_stats(str(temporary))
                os.link(temporary, stats)  # Atomic publication; never replace an existing artifact.
        except BaseException as cleanup_error:
            if target_error is None:
                raise
            try:
                target_error.add_note('Profile cleanup failed: ' + str(cleanup_error))
            except BaseException:
                pass  # Even a broken annotation must retain the original target error.


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--manifest-sha256', required=True)
    args = parser.parse_args()
    manifest, raw, sources, invocation = prepare(args.manifest, args.manifest_sha256)
    event = {'schema': 'quadratic-profile-once-runtime-v1', 'manifest_sha256': args.manifest_sha256,
             'invocation_id': invocation, 'unit': manifest['unit'], 'profile': manifest['profile'],
             'qualification_eligible': False, 'state_reuse_eligible': False}
    profiler = load_profiler(sources)
    print(json.dumps({**event, 'event': 'start'}), flush=True)
    profile_target(profiler, raw, manifest['argv'], Path(manifest['profile']))
    print(json.dumps({**event, 'event': 'dumped'}), flush=True)


if __name__ == '__main__':
    main()
