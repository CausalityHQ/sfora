#!/usr/bin/env python3
"""Discarded source-only public-request observer; native qualification is UNRUN.

CLI: python -B observe_connected_serving.py --authority FILE
--authority-sha256 SHA --output NEWFILE. It authenticates a prospective plan,
never launches native work. FILE is {path:canonical_absolute_file,sha256:actual64}.
The parent supplies the exact authority below, authenticates a quality survivor
with the original full terminal reader, and owns native read/decode/synchronize,
both locks, resource admission, whole-process cap and full uncached exit.

Parent API: RequestObserver(original.fingerprint, sources), then
measure_request(index, read_images, synchronize, pinned_paths, observer=probe).
The index is the unchanged ConnectedCompactIndex. No native imports, globals,
functions or registry entries are replaced. Removing the observer argument
recovers the same public request. No tensor/hash snapshot cache exists.
Successful instrumented PUBLIC requests require completed fingerprint, fresh
tensor and captured raw/unit/codes/inverse_norms/wire witnesses; a scalar-only
standalone observe_call does not authorize a public observation.

Proposed22calls: original owner B1/B32 each2warm+8timed; genuine release;
observation owner B1/B32 each1instrumented; genuine release. Charge both
admissions/releases and full exit separately. Body120s stays inside the parent's
unchanged cap, 8GiB/noSwap/CUDA<10GB/bothlocks; never extend a cap. The timer
includes image read/decode through synchronized native top10, with image cleanup
separate. Stop parity/source/predicate/ownership/resource/cleanup failures.

Baseline raw/unit/codes/inverse_norms/wire are absent from search_images' public
return. A root-pinned same-group original oracle is REQUIRED for full parity;
without it they remain UNMEASURED. Native ties/resources/terminal/full exit are
also root qualification seams, never inferred from this source check or FILE
metadata. Missing attribution forbids choosing an optimization target.

Host event intervals are inclusive/exclusive, with measured callback overhead
removed from their clocks. Dispatcher overhead remains unknown. Inspection and
capture overhead are subsets of callback time, not additional costs to sum.
CPU calls are copy-plus-wait. CUDA intervals and opaque native subdivisions
are UNMEASURED; overlapping host/CUDA intervals must never be added. Return
events can include unwind and do not establish successful source completion.
Mismatched/residual profile stacks fail incomplete. Process cancellation escapes
the callback unchanged; ordinary inspection errors only invalidate observation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from types import CodeType, FunctionType

SCHEMA = 'connected-serving-attribution-authority-v1'
LIMITS = {'body_seconds':120, 'host_bytes':8 * 1024**3, 'swap_bytes':0,
          'cuda_allocated_bytes_exclusive':10_000_000_000}
OUTPUT_KEYS = {'raw', 'unit', 'codes', 'inverse_norms', 'wire'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    require(type(value) is str, 'canonical path string required')
    path = Path(value)
    require(path.is_absolute() and str(path) == value and path.resolve() == path and
            not path.is_symlink(), 'canonical absolute nonsymlink path required')
    return path


def file_bytes(fact, *, keep=False):
    require(type(fact) is dict and fact.keys() == {'path', 'sha256'} and
            type(fact['sha256']) is str and re.fullmatch('[0-9a-f]{64}', fact['sha256']),
            'explicit actual FILE pin required')
    path = canonical(fact['path'])
    require(path.is_file(), 'regular FILE required')
    with path.open('rb') as stream:
        if keep:
            raw = stream.read(2 * 1024**2 + 1)
            require(len(raw) <= 2 * 1024**2, 'source/JSON FILE exceeds2MiB')
            digest = hashlib.sha256(raw).hexdigest()
        else:
            raw = None
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    require(digest == fact['sha256'], 'current FILE SHA256 differs: ' + str(path))
    return raw


def file_fact(fact):
    require(type(fact) is dict and fact.keys() == {'path','sha256'}, 'exact FILE binding required')
    canonical(fact['path'])
    require(type(fact['sha256']) is str and re.fullmatch('[0-9a-f]{64}', fact['sha256']), 'actual SHA256 required')
    return fact


def pairs(items):
    result = dict(items)
    require(len(result) == len(items), 'duplicate JSON key')
    return result


def prepare(path, digest):
    """Pinning inputs is preparation only, never terminal/native admission."""
    raw = file_bytes({'path':str(path), 'sha256':digest}, keep=True)
    authority = json.loads(raw, object_pairs_hook=pairs,
        parse_constant=lambda value: require(False, 'nonfinite JSON'))
    keys = {'schema', 'sources', 'bundle', 'gallery', 'native', 'train_images',
            'qualified_terminal', 'cache_conditions', 'resource_policy', 'both_locks_held',
            'qualification_eligible', 'state_reuse_eligible'}
    require(type(authority) is dict and authority.keys() == keys and authority['schema'] == SCHEMA and
            authority['qualification_eligible'] is False and authority['state_reuse_eligible'] is False,
            'exact source-only attribution authority required')
    sources = authority['sources']
    require(type(sources) is dict and sources.keys() == {'observer','test','bridge','trainer','serializer'},
            'complete observation source pins required')
    for fact in sources.values(): file_bytes(fact)
    require(canonical(sources['observer']['path']) == Path(__file__).absolute(), 'current observer FILE differs')
    bundle, gallery = authority['bundle'], authority['gallery']
    require(type(bundle) is dict and bundle.keys() == {'directory','manifest'} and
            canonical(bundle['directory']).is_dir() and
            file_fact(bundle['manifest'])['path'] == str(Path(bundle['directory']) / 'bundle.json'), 'exact bundle binding required')
    require(type(gallery) is dict and gallery.keys() == {'file','count'} and
            type(gallery['count']) is int and gallery['count'] >= 10, 'exact gallery binding required')
    images = authority['train_images']
    require(type(images) is list and len(images) == 32 and
            len({file_fact(fact)['path'] for fact in images}) == 32, 'fixed32 distinct canonical TRAIN FILEs required')
    require(type(authority['cache_conditions']) is str and authority['cache_conditions'].strip() and
            authority['both_locks_held'] is True, 'parent cache/lock declaration required')
    policy = authority['resource_policy']
    require(type(policy) is dict and policy.keys() == LIMITS.keys() | {'whole_process_seconds','exit_reserve_seconds'} and
            all(type(v) is int and v >= 0 for v in policy.values()) and
            all(policy[k] == v for k,v in LIMITS.items()) and
            policy['exit_reserve_seconds'] > 0 and
            policy['whole_process_seconds'] > 120 + policy['exit_reserve_seconds'], 'unchanged parent cap/reserve required')
    for fact in [bundle['manifest'], gallery['file'], authority['native'], authority['qualified_terminal'], *images]:
        file_bytes(fact)
    return authority


def tensor_snapshot(value):
    """Copy returned HOST bytes only; do not retain outputs or tensor references."""
    tensor_type = getattr(sys.modules.get('torch'), 'Tensor', None)
    require(tensor_type is not None and isinstance(value, tensor_type) and value.device.type == 'cpu',
            'genuine host output tensor required')
    raw = value.detach().cpu().contiguous().reshape(-1).view(sys.modules['torch'].uint8).numpy()
    return {'dtype':str(value.dtype), 'shape':list(value.shape), 'hex':memoryview(raw).tobytes().hex()}


def native_snapshot(result):
    require(isinstance(result, tuple) and len(result) == 2, 'native ID/score pair required')
    return [{'format':memoryview(value).format, 'shape':list(memoryview(value).shape),
             'hex':memoryview(value).tobytes().hex()} for value in result]


class RequestObserver:
    """No frames, tensors, outputs or endpoint owners are saved by the profiler."""

    def __init__(self, fingerprint, sources):
        require(sys.getprofile() is None and sys.flags.optimize == 0, 'unoptimized unprofiled observer required')
        require(type(fingerprint) is FunctionType and 'serializer' in sources, 'original serializer required')
        raw = file_bytes(sources['serializer'], keep=True)
        code = compile(raw, sources['serializer']['path'], 'exec', dont_inherit=True)
        expected = next(c for c in code.co_consts if isinstance(c, CodeType) and c.co_name == 'fingerprint')
        module = sys.modules.get(fingerprint.__module__)
        require(module is not None and vars(module) is fingerprint.__globals__ and
                getattr(module, 'fingerprint', None) is fingerprint and
                module.__file__ == sources['serializer']['path'] and fingerprint.__code__ == expected,
                'original serializer code/module identity differs')
        self.fingerprint_code = fingerprint.__code__
        self.visit_code = next(c for c in self.fingerprint_code.co_consts
            if isinstance(c, CodeType) and c.co_name == 'visit')
        self.files = {}
        for role, fact in sources.items():
            file_bytes(fact)
            self.files[fact['path']] = role
        self.stack, self.events, self.leaves, self.fingerprints = [], {}, [], []
        self.overhead = self.inspection = self.capture = 0
        self.failures, self.output = [], None
        self.calls = 0
        self.started = False
        self.target_completed = False
        self.target_error = None
        self.owner_frame_id = None

    def phase(self, name, filename):
        ancestors = {row['name'] for row in self.stack}
        if name in ('openssl_sha256', 'sha256', 'update', 'hexdigest') and 'fingerprint' in ancestors:
            return 'tensor-sha' if self.stack[-1]['name'] == 'visit' else 'fingerprint/framing'
        if name == 'cpu':
            return 'copy-plus-wait' if 'fingerprint' in ancestors else 'output-transfer/copy-plus-wait'
        if name == 'to' and self.stack and self.stack[-1]['name'] == 'inference_outputs':
            return 'pixel-transfer/copy-plus-wait'
        if name in ('contiguous', 'reshape', 'view', 'numpy') and 'fingerprint' in ancestors:
            return 'layout'
        if name in ('fingerprint', 'visit', 'frame') or 'fingerprint' in ancestors:
            return 'fingerprint/traversal'
        if name in ('_check_current', '_checked_file', '_read_checked', 'bound_file', 'encoder_facts',
                    'model_structure', 'module_origin', '_processor_cache', 'numerical_flags'):
            return 'integrity'
        if name in ('preprocess', '_preprocess') or 'image_processing' in filename:
            return 'preprocess'
        if 'fullfeature_raw_features' in ancestors or name == 'fullfeature_raw_features': return 'readout'
        if name == 'forward' or 'forward' in ancestors: return 'forward'
        if 'pack_int8_unit_embeddings' in ancestors or name in ('pack_int8_unit_embeddings','to_bytes','from_bytes'):
            return 'packing/wire'
        if name == 'search_packed' or 'search_packed' in ancestors: return 'native-search'
        if self.stack and self.stack[-1]['phase'] in ('integrity','preprocess','readout','forward','packing/wire','native-search'):
            return self.stack[-1]['phase']
        return 'other/unresolved'

    def __call__(self, frame, event, arg):
        entered = time.perf_counter_ns()
        try:
            if not self.failures:
                self._event(frame, event, arg, entered - self.overhead)
        except Exception as error:
            # Ordinary inspection errors must not replace a target error; cancellation propagates.
            if len(self.failures) < 8: self.failures.append(type(error).__name__ + ': ' + str(error))
        finally:
            self.overhead += time.perf_counter_ns() - entered

    def _event(self, frame, event, arg, tick):
        # Only the exact install/uninstall calls and predating caller are outside the stack.
        if id(frame) == self.owner_frame_id and ((event in ('c_call','c_return','c_exception') and
                arg is sys.setprofile) or (event == 'return' and not self.stack)):
            return
        if event in ('call', 'c_call'):
            self.calls += 1
            require(self.calls <= 2_000_000 and len(self.stack) < 512, 'profile resource bound exceeded')
            code = frame.f_code
            if event == 'call':
                name, filename = code.co_name, code.co_filename
                key = ('python', filename, code.co_qualname, code.co_firstlineno)
            else:
                name = getattr(arg, '__name__', type(arg).__name__)
                filename = str(getattr(arg, '__module__', None))
                key = ('c', filename, str(getattr(arg, '__qualname__', name)), 0)
            row = {'id':id(frame), 'kind':event, 'key':key, 'name':name, 'start':tick,
                   'children':0, 'phase':self.phase(name, filename), 'c_id':id(arg) if event == 'c_call' else None}
            if event == 'call' and code is self.fingerprint_code:
                require(frame.f_locals.get('frozen') is None, 'serializer hash cache is forbidden')
                row['fingerprint'] = len(self.fingerprints)
                self.fingerprints.append({'sha256':None, 'occurrences':0, 'bytes':0})
            if event == 'call' and code is self.visit_code:
                started = time.perf_counter_ns()
                tensor_type = getattr(sys.modules.get('torch'), 'Tensor', None)
                item = frame.f_locals['item']
                if tensor_type is not None and isinstance(item, tensor_type):
                    require(len(self.leaves) < 4096, 'tensor occurrence bound exceeded')
                    fp = next(r['fingerprint'] for r in reversed(self.stack) if 'fingerprint' in r)
                    row['leaf'] = len(self.leaves)
                    self.leaves.append({'fingerprint':fp, 'dtype':str(item.dtype), 'shape':list(item.shape),
                        'bytes':int(item.numel()) * int(item.element_size()), 'sha256':None})
                self.inspection += time.perf_counter_ns() - started
            self.stack.append(row)
        elif event in ('return', 'c_return', 'c_exception'):
            require(self.stack, 'profile return stack empty')
            row = self.stack[-1]
            require(row['id'] == id(frame) and row['kind'] == ('call' if event == 'return' else 'c_call') and
                    (event == 'return' or row['c_id'] == id(arg)), 'profile return stack mismatch')
            self.stack.pop()
            elapsed = max(0, tick - row['start'])
            if self.stack: self.stack[-1]['children'] += elapsed
            key = (*row['key'], row['phase'])
            require(key in self.events or len(self.events) < 4096, 'host event bound exceeded')
            stats = self.events.setdefault(key, [0,0,0,0])
            stats[0] += 1; stats[1] += elapsed; stats[2] += max(0, elapsed - row['children'])
            stats[3] += event == 'c_exception'
            if 'leaf' in row:
                started = time.perf_counter_ns()
                leaf = self.leaves[row['leaf']]
                local = frame.f_locals
                require('raw' in local and memoryview(local['raw']).nbytes == leaf['bytes'],
                        'fresh original tensor bytes missing/differ')
                fact = local['fact']
                require((str(fact[0]), list(fact[1])) == (leaf['dtype'], leaf['shape']) and
                        re.fullmatch('[0-9a-f]{64}', fact[2]), 'original typed tensor fact differs')
                leaf['sha256'] = fact[2]
                fp = self.fingerprints[leaf['fingerprint']]
                fp['occurrences'] += 1; fp['bytes'] += leaf['bytes']
                self.inspection += time.perf_counter_ns() - started
            if 'fingerprint' in row:
                require(type(arg) is str and re.fullmatch('[0-9a-f]{64}', arg), 'fingerprint return/unwind incomplete')
                self.fingerprints[row['fingerprint']]['sha256'] = arg
            if event == 'return' and frame.f_code.co_name == 'inference_outputs' and self.files.get(frame.f_code.co_filename) == 'trainer':
                started = time.perf_counter_ns()
                require(type(arg) is dict and arg.keys() == OUTPUT_KEYS and type(arg['wire']) is bytes,
                        'inference return/unwind incomplete')
                self.output = {name:tensor_snapshot(arg[name]) for name in OUTPUT_KEYS - {'wire'}}
                self.output['wire_hex'] = arg['wire'].hex()
                self.capture += time.perf_counter_ns() - started

    def report(self):
        phases = {}
        for key,value in self.events.items():
            phases[key[4]] = phases.get(key[4], 0) + value[2] / 1e9
        return {'complete':self.target_completed and not self.failures and
                all(r['sha256'] is not None for r in [*self.leaves,*self.fingerprints]),
            'failures':list(self.failures), 'target_error':self.target_error,
            'tensor_occurrences':self.leaves, 'fingerprints':self.fingerprints,
            'output':self.output, 'callback_seconds':self.overhead / 1e9,
            'counter_inspection_seconds':self.inspection / 1e9, 'output_capture_seconds':self.capture / 1e9,
            'overhead_semantics':'inspection/capture are overlapping subsets of callback time; dispatch unmeasured',
            'host_events':[{'kind':key[0], 'source':key[1], 'function':key[2], 'line':key[3], 'phase':key[4],
                'calls':value[0], 'inclusive_seconds':value[1] / 1e9, 'exclusive_seconds':value[2] / 1e9,
                'c_exceptions':value[3], 'end_semantics':'return_or_unwind'} for key,value in self.events.items()],
            'exclusive_host_phase_seconds':phases,
            'phase_semantics':'event-name attribution; unresolved/opaque subdivisions stay UNMEASURED; inclusive rows overlap',
            'cuda_seconds':None, 'opaque_native_subdivisions':'UNMEASURED',
            'baseline_raw_unit_packed_wire_parity':'UNMEASURED; parent authenticated same-group oracle required',
            'optimization_eligible':False, 'qualification_eligible':False, 'state_reuse_eligible':False}


def observe_call(observer, call):
    require(sys.getprofile() is None and not observer.started, 'fresh unprofiled observer required')
    previous = sys.getprofile()
    observer.started = True
    observer.owner_frame_id = id(sys._getframe())
    try:
        sys.setprofile(observer)
        result = call()
        observer.target_completed = True
    except BaseException as error:
        observer.target_error = type(error).__name__
        raise
    finally:
        sys.setprofile(previous)
        if observer.stack and not observer.failures:
            observer.failures.append('incomplete profile stack at stop')
        observer.stack.clear()
        observer.owner_frame_id = None
    require(not observer.failures and all(row['sha256'] is not None for row in [*observer.leaves,*observer.fingerprints]),
            'incomplete observation: ' + '; '.join(observer.failures))
    return result


def measure_request(index, read_images, synchronize, paths, *, observer=None):
    """Parent-owned decoder/sync; original public search and genuine cleanup."""
    require(isinstance(paths, (list,tuple)) and 1 <= len(paths) <= 32, 'fixed1..32 image paths required')
    images = []
    try:
        synchronize()
        started = time.perf_counter()
        decoded_images = read_images(paths)  # Reader owns cleanup if it fails part-way.
        decoded = time.perf_counter()
        require(isinstance(decoded_images, (list,tuple)), 'decoder image batch differs')
        images = decoded_images
        require(len(images) == len(paths), 'decoder image count differs')
        call = lambda: index.search_images(images)
        result = call() if observer is None else observe_call(observer, call)
        if observer is not None and not (observer.fingerprints and observer.leaves and
                type(observer.output) is dict and observer.output.keys() == (OUTPUT_KEYS - {'wire'}) | {'wire_hex'}):
            observer.failures.append('incomplete public witness: fingerprint/tensor/output required')
            raise ValueError(observer.failures[-1])
        returned = time.perf_counter()
        synchronize()
        completed = time.perf_counter()
        native = native_snapshot(result)
        captured = time.perf_counter()
    except BaseException as error:
        try: index.close()
        except BaseException as cleanup: error.add_note('connected cleanup also failed: ' + repr(cleanup))
        raise
    finally:
        closing = time.perf_counter()
        error = sys.exception()
        failures = []
        for image in images:
            try: image.close()
            except BaseException as failure: failures.append(failure)
        if failures:
            if error is not None:
                for failure in failures: error.add_note('image cleanup also failed: ' + repr(failure))
            else:
                try: index.close()
                except BaseException as failure: failures[0].add_note('connected cleanup also failed: ' + repr(failure))
                raise failures[0]
        cleanup = time.perf_counter() - closing
    return result, {'seconds':completed - started, 'read_decode_seconds':decoded - started,
                    'public_call_seconds':returned - decoded, 'completion_sync_seconds':completed - returned,
                    'native_capture_seconds':captured - completed, 'image_cleanup_seconds':cleanup, 'native':native,
                    'instrumented':observer is not None, 'qualification_eligible':False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--authority-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None,
            'unoptimized -B unprofiled startup required')
    authority = prepare(args.authority, args.authority_sha256)
    output = canonical(args.output)
    result = {'schema':'connected-serving-attribution-preparation-v1', 'authority_sha256':args.authority_sha256,
        'status':'SOURCE_ONLY_PREPARED; native gate UNRUN', 'authority':authority,
        'qualification_eligible':False, 'state_reuse_eligible':False, 'optimization_eligible':False,
        'required_root_seams':['qualified survivor and original terminal reader','actual TRAIN membership/grouping',
            'same-group original raw/unit/packed/wire oracle','both locks and unchanged whole-process/resource cap',
            'serial owners/admissions/genuine releases','native tied-score parity','full uncached exit'],
        'attribution':'UNMEASURED'}
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    main()
