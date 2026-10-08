#!/usr/bin/env python3
"""Prospective discarded public-request driver; source alone is UNQUALIFIED.

CLI: python -B qualify_connected_serving_requests.py --authority FILE
--authority-sha256 SHA --output NEWDIR. The parent freezes actual FILE/CODE/UNIT
bindings, the whole-process cap and exit reserve before any native execution.
Missing/full-selection KILL never admits native imports. No p99, product speed,
adoption or quality claim follows from the diagnostic's eight timed calls.

FILE={path:canonical_absolute_regular_file,sha256:actual64hex}; CODE and UNIT
are the unchanged evaluator descriptors. Authority keys are exactly KEYS.
observation is the existing observer authority FILE. evaluation_authority is
an original full-stage export launch FILE for survivor. selection is its
same-four full-selection GO UNIT; exports is the exact original four UNITs.
sources pins this driver/test, observer/test, bridge, native_wrapper, packing.
locks contains two {path,fd} canonical inherited lifetime lock descriptors,
verified exclusive without waiting and retained through process exit.

One ORIGINAL warmup per B1/B32 uses the existing observer to capture the
same-group original byte oracle. This is explicit prospective instrumentation,
never primary timing. Original owner: each2warm+8unprofiled timed; release;
fresh observation owner: each1instrumented; release. Total22 PUBLIC image calls,
four instrumented. No extra inference request, function/global substitution,
hash cache, trainer edit, image grouping change or state reuse is introduced.
Removing observer arguments recovers the unchanged public search_images calls.
The parent must freeze this warmup-oracle contract before a native launch.
"""
import argparse
import copy
import dataclasses
import fcntl
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time
from types import CodeType, FunctionType, SimpleNamespace
import uuid

SCHEMA = 'connected-serving-requests-authority-v2'
KEYS = {'schema', 'sources', 'observation', 'evaluator', 'evaluation_authority',
        'selection', 'exports', 'survivor', 'locks'}
STARTED = time.perf_counter()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_file(fact):
    require(type(fact) is dict and fact.keys() == {'path', 'sha256'} and
            type(fact['path']) is str and type(fact['sha256']) is str and
            re.fullmatch('[0-9a-f]{64}', fact['sha256']), 'actual FILE required')
    path = Path(fact['path'])
    require(path.is_absolute() and str(path) == fact['path'] and path.resolve() == path and
            path.is_file() and not path.is_symlink(), 'canonical regular FILE required')
    with path.open('rb') as stream:
        raw = stream.read(64*1024**2 + 1)
    require(len(raw) <= 64*1024**2 and hashlib.sha256(raw).hexdigest() == fact['sha256'],
            'current FILE size/SHA256 differs')
    return raw


def strict_json(raw):
    def pairs(items):
        value = dict(items)
        require(len(value) == len(items), 'duplicate JSON key')
        return value
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'nonfinite JSON'))


def read_go(unit):
    """Early rejection only; metadata GO still needs original accept_unit below."""
    require(type(unit) is dict and 'receipt' in unit, 'original full selection GO required')
    record = strict_json(read_file(unit['receipt']))
    require(type(record) is dict and record.get('decision') == 'GO' and
            record.get('stage') == 'full' and record.get('panel') == 'selection',
            'original full selection GO required; missing/KILL is ineligible')
    return record


class Source:
    """Fresh byte reads and authenticated code/live identities, never a hash cache."""
    def __init__(self, module, fact, *, packed_source=None):
        raw = read_file(fact)
        require(len(raw) <= 2*1024**2, 'source exceeds2MiB')
        require(module.__file__ == fact['path'] and sys.modules.get(module.__name__) is module and
                (module.__spec__ is None and module.__name__ == '__main__' or
                 module.__spec__ is not None and module.__spec__.origin == fact['path'] and
                 module.__spec__.name == module.__name__),
                'live source module origin differs')
        expected = {}
        def visit(code):
            expected[code.co_qualname] = code
            for value in code.co_consts:
                if isinstance(value, CodeType): visit(value)
        visit(compile(raw, fact['path'], 'exec', dont_inherit=True))
        self.module, self.fact = module, dict(fact)
        self.spec_values = dict(vars(module.__spec__)) if module.__spec__ is not None else None
        self.values = dict(vars(module))
        self.literals = {k:copy.deepcopy(v) for k,v in self.values.items() if k != '__builtins__' and
                         type(v) in (dict,list,tuple,set,frozenset)}
        legacy_packing = ()
        if module.__name__ == 'sfora.packed_int8':
            # These four canonical definitions retain historical pickle metadata.
            names = ('_unit_rows','PackedInt8Embeddings','fixed_int8_unit_codes','pack_int8_unit_embeddings')
            legacy_packing = tuple(self.values.get(name) for name in names)
            require(all(getattr(value,'__module__',None) == 'sfora.joint_relational_compaction' and
                getattr(value,'__name__',None) == name and
                getattr(value,'__qualname__',None) == name for name,value in zip(names,legacy_packing,strict=True)) and
                isinstance(legacy_packing[1],type), 'canonical packing declarations differ')
            require(isinstance(expected.get('PackedInt8Embeddings'),CodeType) and
                {'__post_init__','bytes_per_vector','restore','cosine_similarity','to_bytes','save','load','from_bytes'} <=
                vars(legacy_packing[1]).keys(), 'canonical packing methods missing')
            for fn in (legacy_packing[0],legacy_packing[2],legacy_packing[3]):
                require(type(fn) is FunctionType and fn.__globals__ is vars(module) and
                    fn.__code__ == expected.get(fn.__qualname__), 'live source code differs')
        self.packed_source = packed_source
        reexports = ()
        self.packing_declarations = ()
        if packed_source is not None:
            require(type(packed_source) is Source and module.__name__ == 'sfora.joint_relational_compaction' and
                packed_source.module.__name__ == 'sfora.packed_int8', 'exact packing source composition required')
            packed_source.check()
            names = ('_PACKED_INT8_ARTIFACT_MAGIC','_SHA256_BYTES','_unit_rows','PackedInt8Embeddings',
                'fixed_int8_unit_codes','pack_int8_unit_embeddings')
            require(all(name in packed_source.values and self.values.get(name) is packed_source.values[name]
                for name in names), 'canonical packing reexports differ')
            reexports = tuple(packed_source.values[name] for name in names)
            self.packing_declarations = tuple(zip(names[2:],reexports[2:],strict=True))
        self.classes, self.functions = [], []
        for value in self.values.values():
            if any(value is owned for owned in reexports): continue
            members = [value]
            generated = False
            if isinstance(value, type) and (value.__module__ == module.__name__ or value in legacy_packing):
                self.classes.append((value, dict(vars(value))))
                generated = dataclasses.is_dataclass(value)
                members = [v.__func__ if isinstance(v,(classmethod,staticmethod)) else v
                           for v in vars(value).values()]
                if legacy_packing and value is legacy_packing[1]:
                    members += [v.fget for v in vars(value).values() if isinstance(v,property)]
            for fn in members:
                if isinstance(fn, FunctionType) and (fn.__module__ == module.__name__ or fn in legacy_packing or
                        legacy_packing and value is legacy_packing[1]):
                    if legacy_packing and value is legacy_packing[1] and fn.__module__ == 'dataclasses':
                        require(fn is getattr(dataclasses,fn.__name__,None), 'canonical dataclass helper differs')
                        self.functions.append((fn,fn.__code__,fn.__defaults__,copy.deepcopy(fn.__kwdefaults__)))
                        continue
                    original = getattr(fn, '__wrapped__', fn)
                    if generated and original.__qualname__ not in expected:
                        # Only fresh admitted stdlib dataclass-generated methods lack source code.
                        require(original.__code__.co_filename == '<string>', 'live generated source differs')
                        self.functions.append((fn,fn.__code__,fn.__defaults__,copy.deepcopy(fn.__kwdefaults__)))
                        continue
                    if original is not fn:
                        wrapper_code = next(c for c in contextmanager.__code__.co_consts
                                            if isinstance(c, CodeType) and c.co_name == 'helper')
                        require(fn.__code__ is wrapper_code and fn.__globals__ is contextmanager.__globals__ and
                                fn.__closure__ is not None and len(fn.__closure__) == 1 and
                                fn.__closure__[0].cell_contents is original, 'live source decorator differs')
                        self.functions.append((fn,fn.__code__,fn.__defaults__,copy.deepcopy(fn.__kwdefaults__)))
                    require(original.__globals__ is vars(module) and original.__code__ == expected.get(original.__qualname__),
                            'live source code differs')
                    self.functions.append((original,original.__code__,original.__defaults__,copy.deepcopy(original.__kwdefaults__)))
        self.function_state = [(fn,dict(vars(fn)),tuple(c.cell_contents for c in fn.__closure__ or ()))
                               for fn,*rest in self.functions]
        if packed_source is not None: self.check()

    @classmethod
    def load(cls, fact):
        raw = read_file(fact)
        name = '_connected_requests_' + uuid.uuid4().hex
        spec = importlib.util.spec_from_file_location(name, fact['path'])
        require(spec is not None and spec.loader is not None, 'source loader required')
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        try:
            exec(compile(raw, fact['path'], 'exec', dont_inherit=True), vars(module))
            return cls(module, fact)
        except BaseException:
            if sys.modules.get(name) is module: del sys.modules[name]
            raise

    def check(self):
        if self.packed_source is not None: self.packed_source.check()
        read_file(self.fact)
        module = self.module
        require(sys.modules.get(module.__name__) is module and vars(module).keys() == self.values.keys() and
                all(vars(module)[k] is v for k,v in self.values.items()) and
                all(vars(module)[k] == v for k,v in self.literals.items()) and
                all(value.__module__ == module.__name__ and value.__name__ == name and value.__qualname__ == name
                    for name,value in self.packing_declarations) and
                all(vars(cls).keys() == values.keys() and all(vars(cls)[k] is v for k,v in values.items())
                    for cls,values in self.classes) and
                all(fn.__code__ is code and fn.__defaults__ == defaults and
                    fn.__kwdefaults__ == kw for fn,code,defaults,kw in self.functions) and
                all(vars(fn).keys() == values.keys() and all(vars(fn)[k] is v for k,v in values.items()) and
                    len(fn.__closure__ or ()) == len(cells) and
                    all(c.cell_contents is value for c,value in zip(fn.__closure__ or (),cells,strict=True))
                    for fn,values,cells in self.function_state) and
                module.__file__ == self.fact['path'] and
                (module.__spec__ is None and module.__name__ == '__main__' or
                 module.__spec__ is not None and module.__spec__.origin == self.fact['path'] and
                 module.__spec__.name == module.__name__ and vars(module.__spec__) == self.spec_values),
                'authenticated live source changed')


def raise_failures(failures):
    primary = next((e for e in failures if not isinstance(e, Exception)), failures[0])
    others = [e for e in failures if e is not primary]
    for error in others: primary.add_note('request driver also failed: ' + repr(error))
    if others:
        if primary.__cause__ is not None and all(primary.__cause__ is not e for e in others):
            others.append(primary.__cause__)
        raise primary from BaseExceptionGroup('other request driver failures', others)
    raise primary


@contextmanager
def owner(factory, guard, charges):
    index, failures = None, []
    started = time.perf_counter()
    try:
        guard()
        index = factory()
        charges['admission_seconds'] = time.perf_counter() - started
        guard()
        yield index
    except BaseException as error:
        failures.append(error)
    finally:
        started = time.perf_counter()
        if index is not None:
            try: index.close()
            except BaseException as error: failures.append(error)
        charges['release_seconds'] = time.perf_counter() - started
        try: guard()
        except BaseException as error: failures.append(error)
        index = None
        if failures: raise_failures(failures)



def validate_instrumentation(report):
    policy = {'schema':'connected-serving-retained-resources-v1',
        'total_calls':'diagnostic_only','live_depth':512,'aggregate_keys':4096,
        'tensor_occurrences':4096,'fingerprint_records':4096,'encoded_bytes':8*1024**2,
        'byte_semantics':'compact ASCII JSON upper-bound reservation; not Python RSS',
        'output_batch_max':32,'cuda_timing':'UNMEASURED'}
    require(type(report.get('instrumentation_policy')) is dict and
        report['instrumentation_policy'] == policy and
        all(type(report['instrumentation_policy'][k]) is type(v) for k,v in policy.items()) and
        report.get('first_failure','missing') is None, 'exact successful instrumentation policy required')
    usage = report.get('resource_usage')
    require(type(usage) is dict and usage.keys() == {'total_calls','max_depth','aggregate_keys',
        'fingerprint_records','tensor_occurrences','encoded_bytes'} and
        all(type(v) is int and v > 0 for k,v in usage.items() if k != 'tensor_occurrences') and
        type(usage['tensor_occurrences']) is int and usage['tensor_occurrences'] >= 0 and
        usage['max_depth'] <= 512 and
        0 < usage['aggregate_keys'] == len(report['host_events']) <= 4096 and
        0 < usage['fingerprint_records'] == len(report['fingerprints']) <= 4096 and
        usage['tensor_occurrences'] == len(report['tensor_occurrences']) <= 4096 and
        usage['total_calls'] == sum(r['calls'] for r in report['host_events']) and
        usage['encoded_bytes'] <= 8*1024**2, 'complete retained resource accounting required')
    encoded = sum(len(part) for part in json.JSONEncoder(ensure_ascii=True,separators=(',',':'),allow_nan=False).iterencode(report))
    require(encoded <= usage['encoded_bytes'], 'encoded report exceeds reserved bytes')

def witness(report, count):
    validate_instrumentation(report)
    require(report['complete'] is True and not report['failures'] and report['target_error'] is None and
            report['fingerprints'] and report['tensor_occurrences'], 'complete original public witness required')
    output = report['output']
    require(type(output) is dict and output.keys() == {'raw','unit','codes','inverse_norms','wire_hex'},
            'complete original raw/unit/packed/wire witness required')
    for name,dtype,shape,width in (('raw','torch.float32',[count,128],4),
            ('unit','torch.float32',[count,128],4), ('codes','torch.int8',[count,128],1),
            ('inverse_norms','torch.float16',[count],2)):
        value = output[name]
        require(type(value) is dict and value.keys() == {'dtype','shape','hex'} and value['dtype'] == dtype and
                value['shape'] == shape and type(value['hex']) is str and
                re.fullmatch('[0-9a-f]*',value['hex']) and
                len(value['hex']) == count * (128 if name != 'inverse_norms' else 1) * width * 2,
                'typed original output bytes differ')
    require(type(output['wire_hex']) is str and re.fullmatch('[0-9a-f]*',output['wire_hex']) and
            len(output['wire_hex']) == count*130*2, 'complete original wire differs')
    codes, norms = bytes.fromhex(output['codes']['hex']), bytes.fromhex(output['inverse_norms']['hex'])
    require(bytes.fromhex(output['wire_hex']) == b''.join(codes[i*128:(i+1)*128] + norms[i*2:(i+1)*2]
                for i in range(count)), 'original typed codes/inverse norms/wire parity differs')
    return output


def request_body(factory, observer, read_images, synchronize, paths, sources, guard):
    """Reuse the existing read/decode→synchronized native timer without changing owners."""
    require(len(paths) == 32 and len(set(paths)) == 32, 'fixed32 distinct image paths required')
    started = time.perf_counter()
    calls, originals, observations, charges = [], {}, {}, [{},{}]
    def bounded():
        guard()
        require(time.perf_counter()-started < 120, 'diagnostic body120 cap exceeded')
    def request(index, count, kind, *, probe=None):
        bounded()
        result, row = observer.measure_request(index, read_images, synchronize, paths[:count], observer=probe)
        result = None  # Do not retain a native result or any live tensor in the runner.
        bounded()
        require(all(type(row[k]) in (int,float) and 0 <= row[k] < float('inf') for k in
            ('seconds','read_decode_seconds','public_call_seconds','completion_sync_seconds',
             'native_capture_seconds','image_cleanup_seconds')) and row['seconds'] > 0 and
            abs(row['seconds'] - row['read_decode_seconds'] - row['public_call_seconds'] -
                row['completion_sync_seconds']) < 1e-6, 'complete public request timing differs')
        row.update(batch=count, kind=kind)
        calls.append(row)
        return row
    with owner(factory, bounded, charges[0]) as index:
        for count in (1,32):
            probe = observer.RequestObserver.from_index(index, sources)
            row = request(index,count,'warm_oracle',probe=probe)
            report = probe.report()
            originals[str(count)] = {'output':witness(report,count), 'native':row['native']}
            probe = report = None
            for kind in ['warm'] + ['timed']*8:
                row = request(index,count,kind)
                require(row['native'] == originals[str(count)]['native'], 'original native ID/score parity differs')
    index = None
    with owner(factory, bounded, charges[1]) as index:
        for count in (1,32):
            probe = observer.RequestObserver.from_index(index, sources)
            row = request(index,count,'observation',probe=probe)
            report = probe.report()
            require({'output':witness(report,count), 'native':row['native']} == originals[str(count)],
                    'same-group original raw/unit/codes/inverse norms/wire/native parity differs')
            observations[str(count)] = report
            probe = report = None
    index = None
    bounded()
    timed = {}
    for count in (1,32):
        seconds = [r['seconds'] for r in calls if r['batch'] == count and r['kind'] == 'timed']
        require(len(seconds) == 8, 'eight unprofiled original timed requests required')
        timed[str(count)] = {'seconds':seconds, 'measured_images_per_second':count*8/sum(seconds)}
    return {'calls':calls, 'timed':timed, 'original_warmup_oracles':originals,
        'observations':observations, 'owners':charges, 'body_seconds':time.perf_counter()-started,
        'same_group_original_byte_native_parity':True,
        'oracle_semantics':'instrumented first original warmup; second warmup and all8timed are unprofiled',
        'timing_semantics':'read/decode through synchronized native top10; capture/cleanup separate; observer overhead retained',
        'qualification_eligible':False, 'state_reuse_eligible':False, 'optimization_eligible':False,
        'product_p99':'UNQUALIFIED; requires10000interleaved paired calls and confidence interval'}


class Locks:
    """Verify the parent's two inherited lifetime descriptors; never release them."""
    def __init__(self, rows):
        require(type(rows) is list and len(rows) == 2 and
                all(type(r) is dict and r.keys() == {'path','fd'} and type(r['path']) is str and
                    type(r['fd']) is int and r['fd'] >= 3 for r in rows) and
                len({r['path'] for r in rows}) == len({r['fd'] for r in rows}) == 2,
                'two original inherited lifetime lock descriptors required')
        self.rows = rows
        self.check()

    def check(self):
        for row in self.rows:
            path, fd = Path(row['path']), row['fd']
            require(path.is_absolute() and str(path) == row['path'] and path.resolve() == path and
                    not path.is_symlink() and Path('/proc/self/fd/'+str(fd)).resolve() == path,
                    'original lifetime lock path/descriptor differs')
            actual, current = os.fstat(fd), path.stat()
            require(stat.S_ISREG(actual.st_mode) and (actual.st_dev,actual.st_ino) ==
                    (current.st_dev,current.st_ino), 'original lifetime lock inode differs')
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with path.open('rb') as independent:
                try: fcntl.flock(independent.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError: pass
                else: raise ValueError('original exclusive lifetime lock not held')


def check_resources(facts, policy, *, reserve):
    remaining = policy['exit_reserve_seconds'] if reserve else 0
    require(type(facts['wall_seconds']) in (int,float) and math.isfinite(facts['wall_seconds']) and
            0 <= facts['wall_seconds'] < policy['whole_process_seconds']-remaining and
            type(facts['process_peak_rss_kib']) in (int,float) and
            0 < facts['process_peak_rss_kib'] <= 8*1024**2 and
            type(facts['peak_cuda_allocated_bytes']) is int and
            0 <= facts['peak_cuda_allocated_bytes'] < 10_000_000_000,
            'whole-unit resource cap/exit headroom differs')


def native_ties(packed, gallery_type, native, observer):
    """Independent discarded gallery: duplicate exact e1 vectors, exact ordinal ties."""
    observer.file_bytes(native)
    row = b'\x01' + bytes(127) + struct.pack('<e',1.)
    gallery = gallery_type.open_packed(Path(native['path']), packed.from_bytes(row*32,count=32,dimensions=128))
    failures, rows = [], []
    try:
        for count in (1,32):
            result = gallery.search_packed(packed.from_bytes(row*count,count=count,dimensions=128),k=10)
            snapshot = observer.native_snapshot(result)
            require(snapshot[0]['shape'] == snapshot[1]['shape'] == [count,10] and
                    snapshot[0]['hex'] == (struct.pack('<10q',*range(10))*count).hex() and
                    snapshot[1]['hex'] == (struct.pack('<10f',*([1.]*10))*count).hex(),
                    'original native tied-score ordinal/score bits differ')
            rows.append(snapshot)
            result = None
    except BaseException as error: failures.append(error)
    finally:
        try: gallery.close()
        except BaseException as error: failures.append(error)
        gallery = None
        if failures: raise_failures(failures)
    return {'ascending_ordinal_score_bits_exact':True, 'native':rows,
            'gallery':'separate discarded duplicate e1; resident public gallery unchanged'}


def decode_images(observer, image_api, facts, paths):
    images, errors = [], []
    for path in paths:
        opened = None
        try:
            fact = next(f for f in facts if f['path'] == str(path))
            observer.file_bytes(fact)
            opened = image_api.open(path)
            images.append(opened.convert('RGB'))
        except BaseException as error: errors.append(error)
        finally:
            if opened is not None:
                try: opened.close()
                except BaseException as error: errors.append(error)
            opened = None
        if errors: break
    if errors:
        for image in images:
            try: image.close()
            except BaseException as error: errors.append(error)
        images.clear()
        raise_failures(errors)
    return images


def run(args):
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None,
            'unoptimized -B unprofiled startup required')
    require(not any(n.split('.')[0] in {'torch','numpy','PIL','sfora','transformers','torchvision','safetensors'}
                    for n in sys.modules), 'native import preceded GO admission')
    authority_fact = {'path':str(args.authority), 'sha256':args.authority_sha256}
    authority = strict_json(read_file(authority_fact))
    require(type(authority) is dict and authority.keys() == KEYS and authority['schema'] == SCHEMA,
            'exact prospective request authority required')
    go = read_go(authority['selection'])
    sources = authority['sources']
    require(type(sources) is dict and sources.keys() ==
        {'driver','test','observer','observer_test','bridge','native_wrapper','packing','runtime','ledger','packed'}, 'complete request source pins required')
    require(sources['driver']['path'] == str(Path(__file__).absolute()) and
            sources['test']['path'] == str(Path(__file__).with_name('test_connected_serving_requests.py').absolute()),
            'current driver/test FILE required')
    output = args.output
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and
            not output.is_symlink(), 'exclusive canonical new output required')
    owned, failures, context, exit_guard, before, guard = [], [], None, None, None, None
    try:
        observer_source = Source.load(sources['observer']); owned.append(observer_source)
        observer = observer_source.module
        observation = observer.prepare(Path(authority['observation']['path']),authority['observation']['sha256'])
        require(observation['sources']['observer'] == sources['observer'] and
                observation['sources']['test'] == sources['observer_test'] and
                observation['sources']['bridge'] == sources['bridge'] and
                all(observation['sources'][role] == sources[role] for role in ('runtime','ledger','packed')) and
                observation['qualified_terminal'] == authority['selection']['receipt'], 'observation source/GO binding differs')
        for fact in sources.values(): observer.file_bytes(fact)
        self_source = Source(sys.modules[__name__], sources['driver'])
        evaluator_fact = authority['evaluator']
        evaluator_source = Source.load({'path':str(Path(evaluator_fact['root'])/'evaluate_siglip2_connected_mlp.py'),
            'sha256':evaluator_fact['code']['evaluate_siglip2_connected_mlp.py']}); owned.append(evaluator_source)
        evaluator = evaluator_source.module
        evaluator.check_code(evaluator_fact,evaluator.FILES)
        require(evaluator.closure(evaluator_fact['root'],evaluator_fact['execution_sha256'],evaluator.FILES,{}) ==
                evaluator_fact['code'], 'complete original evaluator CODE differs')
        original = strict_json(read_file(authority['evaluation_authority']))
        survivor = authority['survivor']
        require(type(survivor) is dict and survivor.keys() == {'seed','arm'} and survivor['arm'] == 'candidate' and
                type(survivor['seed']) is int and survivor['seed'] in evaluator.SEEDS and
                original['stage'] == 'full' and original['panel'] == 'selection', 'exact full candidate survivor required')
        eargs = SimpleNamespace(execution_sha256=evaluator_fact['execution_sha256'],
            authority=Path(authority['evaluation_authority']['path']),authority_sha256=authority['evaluation_authority']['sha256'],
            phase='export',arm=survivor['arm'],seed=survivor['seed'],output=output)
        locks = Locks(authority['locks'])
        context, exit_guard = evaluator.authority(eargs)
        evaluator.merge_guards(context['guards'], {f['path']:f['sha256'] for f in
            [authority_fact,authority['observation'],authority['evaluation_authority'],*sources.values(),
             *observation['sources'].values(),observation['bundle']['manifest'],
             observation['gallery']['file'],observation['native'],*observation['train_images']]})
        require(go['launch']['selected_cpu'] == context['launch']['selected_cpu'] and
                go['launch']['endpoints'] == context['launch']['endpoints'] and
                authority['exports'] == go['launch']['exports'] and
                authority['exports'].keys() == {evaluator.label(e) for e in context['launch']['endpoints']},
                'same-four original CPU/export/selection authority required')
        export_records = {}
        for endpoint in context['launch']['endpoints']:
            export_records[evaluator.label(endpoint)] = evaluator.accept_unit(context,
                authority['exports'][evaluator.label(endpoint)],'export',endpoint['arm'],endpoint['seed'],
                stage='full',panel='selection')
        admitted = evaluator.accept_unit(context,authority['selection'],'score',stage='full',panel='selection')
        require(admitted == go and admitted['decision'] == 'GO' and
                admitted['selection_go_admits_validation_only'] is True, 'original full terminal GO required')
        endpoint = next(e for e in context['launch']['endpoints'] if (e['seed'],e['arm']) ==
                        (survivor['seed'],survivor['arm']))
        require(observation['bundle']['manifest'] == endpoint['bundle'], 'same survivor bundle FILE required')
        policy = observation['resource_policy']
        require(policy['whole_process_seconds'] <= evaluator.policy('export')['seconds'] and
                time.perf_counter()-STARTED + 120 + policy['exit_reserve_seconds'] < policy['whole_process_seconds'],
                'insufficient frozen whole-process admission/body/exit headroom')
        before = evaluator.native_start(context)
        import torch
        from PIL import Image
        from sfora import cutile_int8, joint_relational_compaction, packed_int8
        native_source = Source(cutile_int8,sources['native_wrapper'])
        packed_source = Source(packed_int8,sources['packed'])
        packing_source = Source(joint_relational_compaction,sources['packing'],packed_source=packed_source)
        bridge_source = Source.load(sources['bridge']); owned.append(bridge_source)
        torch.random.default_generator.manual_seed(survivor['seed'])
        torch.cuda.manual_seed_all(survivor['seed'])
        rng, cuda_rng = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
        def guard(*, reserve=True):
            locks.check()
            for source in (self_source,observer_source,evaluator_source,bridge_source,native_source,packed_source,packing_source): source.check()
            for fact in [*sources.values(),*observation['sources'].values(),observation['bundle']['manifest'],
                         authority_fact,authority['observation'],
                         authority['evaluation_authority'],observation['native'],observation['gallery']['file']]:
                observer.file_bytes(fact)
            observer.check_runtime_sources(observation['sources'],observation['bundle'])
            evaluator.guard_helpers(context)
            resources = evaluator.resources(context,before)
            resources['wall_seconds'] = time.perf_counter()-STARTED
            check_resources(resources,policy,reserve=reserve)
            require(torch.equal(rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in
                    zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)), 'whole-unit RNG changed')
            return resources
        final_resources = guard()
        ties = native_ties(joint_relational_compaction.PackedInt8Embeddings,
                          cutile_int8.CutilePackedInt8Gallery,observation['native'],observer)
        guard()
        def read_images(paths):
            return decode_images(observer,Image,observation['train_images'],paths)
        with evaluator.endpoint_scope(context,endpoint):
            facts = evaluator.authenticate_payloads(context,endpoint)
            require(facts == context['cpu']['payload_facts'][evaluator.label(endpoint)], 'CPU-qualified updated payload differs')
            rows,mapping = context['nearest_evaluator'].image_rows(context,'selection')
            selected = {r['path']:r for r in rows}
            require(all(f['path'] in selected and selected[f['path']]['image_sha256'] == f['sha256']
                        for f in observation['train_images']), 'actual ordered TRAIN membership/image bytes differ')
            exported = export_records[evaluator.label(endpoint)]
            name = evaluator.label(endpoint)+'.packed.bin'
            wire = read_file({'path':str(Path(exported['output'])/name),'sha256':exported['files'][name]})
            gallery_wire = b''.join(wire[i*130:(i+1)*130] for i in mapping['gallery'])
            require(len(wire) == len(rows)*130 and observation['gallery']['count'] == len(mapping['gallery']) and
                    hashlib.sha256(gallery_wire).hexdigest() == observation['gallery']['file']['sha256'],
                    'original admitted TRAIN gallery row/wire binding differs')
            wire = gallery_wire = None
            def factory():
                index = bridge_source.module.ConnectedCompactIndex.from_bundle(
                    bundle_dir=Path(observation['bundle']['directory']),expected_bundle_sha256=endpoint['bundle']['sha256'],
                    gallery_path=Path(observation['gallery']['file']['path']),
                    expected_gallery_sha256=observation['gallery']['file']['sha256'],gallery_count=observation['gallery']['count'],
                    native_library_path=Path(observation['native']['path']),expected_native_library_sha256=observation['native']['sha256'])
                return index
            diagnostic = request_body(factory,observer,read_images,torch.cuda.synchronize,
                [Path(f['path']) for f in observation['train_images']],observation['sources'],guard)
        guard()
        record = {'schema':'connected-serving-requests-diagnostic-v1','status':'DISCARDED_DIAGNOSTIC',
            'authority':authority_fact,'sources':sources,'selection':authority['selection'],'survivor':survivor,
            'ties':ties,**diagnostic,'qualification_eligible':False,'state_reuse_eligible':False,
            'resource_policy':policy,'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),
                'python_sha256':context['training_context']['legacy']['selected']['source_cpu']['invocation']['python_sha256'],
                'python_version':sys.version,'pid':os.getpid(),'invocation_id':os.environ['INVOCATION_ID'],
                'optimize':sys.flags.optimize,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
    except BaseException as error:
        failures.append(error)
    finally:
        if context is not None:
            try:
                evaluator.exit_rehash(context,exit_guard)
                if guard is not None:
                    final_resources = guard(reserve=False)
            except BaseException as error: failures.append(error)
        for source in reversed(owned):
            if sys.modules.get(source.module.__name__) is source.module:
                del sys.modules[source.module.__name__]
            else: failures.append(ValueError('owned source registry changed at exit'))
        if failures: raise_failures(failures)
    record['full_uncached_exit_pass'] = True
    record['resources'] = final_resources
    record['input_guards'] = dict(context['guards'])
    record['whole_process_seconds'] = time.perf_counter()-STARTED
    check_resources({'wall_seconds':record['whole_process_seconds'],
        **{k:final_resources[k] for k in ('process_peak_rss_kib','peak_cuda_allocated_bytes')}},policy,reserve=False)
    output.mkdir()
    context['helper'].publish(output/'receipt.json',record)
    require(time.perf_counter()-STARTED < policy['whole_process_seconds'], 'receipt included whole-process cap exceeded')
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--authority',type=Path,required=True)
    parser.add_argument('--authority-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    return run(parser.parse_args(argv))


if __name__ == '__main__':
    main()
