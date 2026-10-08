#!/usr/bin/env python3
"""Stdlib-only control batch observer seams; native qualification stays root-owned."""
import ast
from contextlib import contextmanager, nullcontext, ExitStack, redirect_stderr
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
DRIVER = HERE / 'observe_connected_control_batch_execution.py'
EVIDENCE = HERE.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(path.read_bytes(), str(path), 'exec', dont_inherit=True), vars(module))
    return module


class Tensor:
    """Ordered rows, distinct bytes; implements only native boundary operations."""
    def __init__(self, rows, width=1152, dtype='f32'):
        self.rows = list(rows)
        self.width = width
        self.shape = (len(rows), width) if width != 1 else (len(rows),)
        self.dtype = dtype
        self.device = SimpleNamespace(type='cuda')
    def __getitem__(self, index):
        return Tensor(self.rows[index] if isinstance(index, slice) else [self.rows[i] for i in index], self.width, self.dtype)
    def detach(self): return self
    def cpu(self): return self
    def contiguous(self): return self
    def reshape(self, *args): return self
    def view(self, *args): return self
    def numpy(self):
        code = {'f32': 'f', 'f16': 'e', 'i8': 'b'}[self.dtype]
        return b''.join(struct.pack('<'+code, int(x) if code == 'b' else x)*self.width for x in self.rows)
    def to(self, *args): return self
    def float(self): return self
    def clone(self): return Tensor(self.rows, self.width, self.dtype)
    def norm(self, dim): return self
    def __gt__(self, value): return self
    def all(self): return self
    def item(self): return True


def outputs(rows):
    return {'raw': Tensor(rows, 128), 'unit': Tensor(rows, 128),
            'codes': Tensor([int(x) for x in rows], 128, 'i8'), 'inverse_norms': Tensor([.5]*len(rows), 1, 'f16'),
            'wire': b''.join(struct.pack('<128be', *([int(x)]*128), .5) for x in rows)}


def fingerprint(value):
    if isinstance(value, Tensor):
        return hashlib.sha256(repr(value.shape).encode()+value.numpy()).hexdigest()
    if isinstance(value, dict):
        return hashlib.sha256(b''.join(k.encode()+(v if isinstance(v, bytes) else v.numpy()) for k, v in value.items())).hexdigest()
    return 'fixed'


@contextmanager
def native_standins(observer, *, drift=False, normalization=False, pixel_mutant=False, repeat_mutant=False):
    calls = []
    images = []
    mode = {'forwards': 0}
    class Image:
        size = (256, 256)
        def __init__(self, row): self.row = row; self.closed = False; images.append(self)
        def __enter__(self): return self
        def __exit__(self, *args): self.close()
        def close(self): self.closed = True
        def convert(self, mode): return Image(self.row)
        def tobytes(self): return struct.pack('<i', self.row)
    def processor(*, images, return_tensors):
        rows = [i.row for i in images]
        if pixel_mutant and len(rows) == 6:
            rows = rows[::-1]
        result = Tensor(rows)
        result.shape = (len(rows), 3, 256, 256)
        return {'pixel_values':result}
    def model(*, pixel_values):
        calls.append(('encoder', len(pixel_values.rows)))
        mode['forwards'] += 1
        delta = .125 if len(pixel_values.rows) == 6 and drift else 0
        if repeat_mutant and mode['forwards'] == 3: delta += .125
        return SimpleNamespace(pooler_output=Tensor([r+delta for r in pixel_values.rows]))
    def raw(features, *args):
        calls.append(('readout', len(features.rows)))
        return Tensor(features.rows, 128)
    def pack(unit):
        values = outputs(unit.rows)
        return SimpleNamespace(codes=values['codes'], inverse_norms=values['inverse_norms'], to_bytes=lambda: values['wire'])
    def normalize(value, dim):
        if normalization and value.width == 1152 and len(value.rows) == 6:
            return Tensor([r+.0625 for r in value.rows])
        return value
    torch = SimpleNamespace(uint8='u8', float32='f32', float16='f16',
        isfinite=lambda x:x, no_grad=nullcontext, autocast=lambda *a, **k:nullcontext(),
        random=SimpleNamespace(get_rng_state=lambda:Tensor([1])),
        cuda=SimpleNamespace(get_rng_state_all=lambda:[Tensor([2])]),
        equal=lambda a,b:a.numpy() == b.numpy())
    flags = {'cudnn_allow_tf32': True, 'matmul_allow_tf32': False}
    original = SimpleNamespace(fingerprint=lambda value:'fixed')
    source = SimpleNamespace(numerical_flags=lambda:flags)
    module = SimpleNamespace(__file__='/source.py')
    state = {'modules': {'train_siglip2_substrate_adaptation.py':original,
        'qualify_siglip2_substrate_cpu.py':source, 'quadratic_readout.py':module,
        'prototype_residual_readout.py':module,
        'joint_relational_compaction.py':SimpleNamespace(pack_int8_unit_embeddings=pack)},
        'device':'cuda', 'guards':{}, 'flags':flags,
        'encoder_identity':'fixed', 'vision_sha256':'fixed', 'readout_sha256':'fixed',
        'manifest':{'encoder_identity':'fixed', 'vision_sha256':'fixed', 'environment':{'packages':{}}},
        'head_object':SimpleNamespace(parameters=lambda:[], modules=lambda:[]), 'model':model,
        'processor_object':processor, 'A':None, 'C':None, 'means':None, 'mu_train':None, 'arm':'control'}
    for m in state['modules'].values():
        m.__file__ = '/source.py'
    state['guards']['/source.py'] = 'x'
    connected = SimpleNamespace(bound_file=lambda *a:None, encoder_facts=lambda *a,**k:{'vision_sha256':'fixed'},
        inference_readout_tree=lambda state:None, fullfeature_raw_features=raw, mapping_absent=lambda path:None)
    owner = SimpleNamespace(packed_outputs=lambda c,r:outputs(r.rows))
    d = SimpleNamespace(label=lambda e:observer.KEY, fresh_values=lambda *a:outputs(list(range(1,33))),
        output_bytes=lambda values:{k:v if k == 'wire' else v.numpy() for k,v in values.items()})
    sources = SimpleNamespace(modules={'connected':connected, 'original':SimpleNamespace(fingerprint=fingerprint),
        'quadratic_owner':owner, 'baseline':SimpleNamespace(cache_rows=lambda c,r:Tensor(list(range(1,33))))})
    rows = [{'path':str(i), 'image_sha256':'x', 'panel_ordinal':i-1, 'original_row':i} for i in range(1,33)]
    rgb = hashlib.sha256()
    for row in rows:
        rgb.update(str((256,256)).encode()); rgb.update(struct.pack('<i', row['original_row']))
    pixels = processor(images=[Image(i) for i in range(1,33)], return_tensors='pt')['pixel_values']
    for image in images: image.close()
    batch = {'rows':rows, 'rgb_sha256':rgb.hexdigest(), 'pixels_sha256':fingerprint(pixels),
             'outputs_sha256':fingerprint(outputs(list(range(1,33))))}
    context = {'guards':{}, 'launch':{'original_cache':{'path':'cache'}},
               'score':{'launch':{'endpoints':[{}]}}}
    fake = {'torch':torch, 'torch.nn':SimpleNamespace(functional=SimpleNamespace(normalize=normalize)),
            'PIL':SimpleNamespace(Image=SimpleNamespace(open=lambda path:Image(int(path))))}
    with patch.dict(sys.modules, fake), patch.object(observer, 'authenticated', lambda *a,**k:None):
        yield SimpleNamespace(d=d, sources=sources, context=context, state=state, batch=batch,
                              calls=calls, images=images, torch=torch)


class ObserverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.o = load(DRIVER, '_control_observer_test') if DRIVER.is_file() else None
        cls.d = load(HERE/'diagnose_connected_gallery_freshness.py', '_control_historical_test')
        cls.old = load(HERE/'test_connected_gallery_freshness.py', '_control_old_fixtures')

    def test_implementation_and_original_math_correspondence(self):
        self.assertTrue(DRIVER.is_file(), 'control batch observer missing')
        observer = self.o
        observer.check_capture_ast((HERE / 'train_siglip2_connected_mlp.py').read_bytes())

    def test_math_mutants_fail_whole_body_ast(self):
        raw = DRIVER.read_bytes()
        original = (HERE/'train_siglip2_connected_mlp.py').read_bytes()
        for a,b in [(b'F.normalize(pooled.float(),dim=1)', b'F.normalize(pooled,dim=1)'),
                    (b'pixels.to(device)', b'pixels'),
                    (b'enabled=device == \'cuda\'', b'enabled=False'),
                    (b'require_pixels(pixels, expected_pixels)', b'require_pixels(pixels, None)'),
                    (b'def capture_inference_outputs(connected, endpoint, images, expected_pixels=None):',
                     b'def capture_inference_outputs(connected, endpoint, images, expected_pixels=None, *, extra=False):')]:
            with self.subTest(a=a), self.assertRaises(ValueError):
                self.o.check_capture_ast(original, b.join(raw.rsplit(a,1)))
        late=raw.replace(b'    require_pixels(pixels, expected_pixels)\n',b'')
        late=late.replace(b"    return ({'raw':", b"    require_pixels(pixels, expected_pixels)\n    return ({'raw':")
        with self.assertRaisesRegex(ValueError,'pre-forward'):self.o.check_capture_ast(original,late)

    def test_real_historical_prepare_then_fixed_selection_and_denials(self):
        with self.old.metadata_fixture(self.d) as f, patch.object(self.d, 'file_bytes', f.reader):
            context = self.d.prepare(f.args)
            context['partition']['panels']['validation'] = {'original_rows':[13216,13217,13218,13219,13270]}
            batch = self.o.fixed_batch(context)
            self.assertEqual([r['original_row'] for r in batch['rows'][26:]], self.o.SUBSET_FIT)
            self.assertEqual(len(batch['rows']),32)
            self.assertEqual([r['panel_ordinal'] for r in batch['rows']], context['partition']['panels']['selection']['query'][1696:1728])
            for panel in ('train','validation'):
                bad = copy.deepcopy(context)
                bad['partition']['panels'][panel]['original_rows'].append(batch['rows'][0]['original_row'])
                with self.assertRaisesRegex(ValueError,'TRAIN/VAL'):self.o.fixed_batch(bad)
            bad = copy.deepcopy(context)
            bad['exports'][self.o.KEY]['images'][53]['rows'].reverse()
            with self.assertRaisesRegex(ValueError,'roles'):self.o.fixed_batch(bad)
            self.assertTrue(any('candidate' in p and p.endswith('endpoint.pt') for p in f.reads))
            self.assertFalse(any(p.endswith('.jpg') for p in f.reads))

            endpoint=context['score']['launch']['endpoints'][0]
            prospective={'guards':{'/prospective.py':'actual'},'launch':{
                'control':{'bundle':endpoint['bundle'],'export':context['score']['launch']['exports'][self.o.KEY]['receipt']},
                'images':[{'path':r['path'],'sha256':r['image_sha256']} for r in batch['rows']]}}
            seen=[]
            def admit(fact,guards):seen.append(fact['path']);guards[fact['path']]=fact['sha256']
            with patch.object(self.o,'authenticated',admit):
                self.o.bind_control(prospective,context)
                self.assertEqual(seen,[r['path'] for r in batch['rows']])
                for mutate in (lambda p:p['launch']['images'].reverse(),
                    lambda p:p['launch']['control'].__setitem__('bundle',context['score']['launch']['endpoints'][1]['bundle']),
                    lambda p:p['launch']['images'].append({'path':'VAL.jpg','sha256':'x'})):
                    bad=copy.deepcopy(prospective);mutate(bad);seen.clear()
                    with self.assertRaises(ValueError):self.o.bind_control(bad,context)
                    self.assertEqual(seen,[])

    def test_actual_prospective_authority_and_exact_two_closure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); frozen=EVIDENCE/'connected-gallery-freshness-v3-freeze'
            for name in self.o.FILES:(root/name).write_bytes((HERE/name).read_bytes())
            code={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in self.o.FILES}
            (root/'execution.json').write_text(json.dumps(code))
            execution=hashlib.sha256((root/'execution.json').read_bytes()).hexdigest()
            history={key:{'path':str(frozen/name),'sha256':self.o.HISTORICAL[key]} for key,name in
                [('source','diagnose_connected_gallery_freshness.py'),('execution','execution.json'),('authority','authority.json')]}
            launch={'schema':'connected-control-batch-execution-launch-v1','execution_sha256':execution,
                'historical':history,'control':{},'images':[],'output':str(root/'result'),
                'resource_policy':self.o.LIMITS,'both_locks_held':True,'candidate_status':'KILL',
                'qualification_eligible':False,'state_reuse_eligible':False}
            module=load(root/DRIVER.name,'_observer_authority_test')
            args=SimpleNamespace(execution_sha256=execution,authority=root/'authority.json',output=root/'result')
            def prepare(value):
                args.authority.write_text(json.dumps(value))
                args.authority_sha256=hashlib.sha256(args.authority.read_bytes()).hexdigest()
                return module.prepare(args)
            self.assertEqual(prepare(launch)['code'],code)
            for mutate in (lambda v:v.__setitem__('extra',True),lambda v:v.__setitem__('output',str(root/'wrong')),
                lambda v:v['resource_policy'].__setitem__('seconds',701),lambda v:v.__setitem__('state_reuse_eligible',True),
                lambda v:v['historical']['source'].__setitem__('sha256','0'*64)):
                bad=copy.deepcopy(launch);mutate(bad)
                with self.assertRaises(ValueError):prepare(bad)
            (root/next(iter(self.o.FILES))).write_text('mutated')
            with self.assertRaisesRegex(ValueError,'SHA'):prepare(launch)
            self.o.remove_source(module)

    def test_genuine_historical_and_own_source_guards(self):
        sources = self.d.Sources({'launch':{'sources':self.old.source_facts(self.d)},'guards':{}})
        try:
            sources.admit()
            guard = sources.modules['evaluator'].source_live_guard
            own = guard(self.o, hashlib.sha256(DRIVER.read_bytes()).hexdigest(), {})
            historical = guard(self.d, hashlib.sha256(Path(self.d.__file__).read_bytes()).hexdigest(), {}, class_name='Sources')
            for fn,check in ((self.o.capture_inference_outputs,own),(self.d.Sources.admit,historical),
                             (sources.modules['connected'].load_inference,sources.guard)):
                old = fn.__code__
                try:
                    fn.__code__ = (lambda *a,**k:None).__code__
                    with self.assertRaisesRegex(ValueError,'source|function'):check()
                finally:fn.__code__ = old
            own();historical();sources.guard()
        finally:sources.close()

    def test_exact_file_rehash_rejects_mutation_and_duplicate_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'input.json'; p.write_text('{"a":1}')
            fact={'path':str(p), 'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}; guards={}
            self.assertEqual(self.o.read_json(fact,guards),{'a':1})
            self.o.rehash(guards)
            p.write_text('{"a":2}')
            with self.assertRaisesRegex(ValueError,'SHA'):self.o.rehash(guards)
            p.write_text('{"a":1,"a":2}')
            fact['sha256']=hashlib.sha256(p.read_bytes()).hexdigest()
            with self.assertRaisesRegex(ValueError,'duplicate'):self.o.read_json(fact,{})

    def test_bulk_advice_covers_consumed_bytes_and_tail_after_hashing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bulk'; tail=b'tail\x00\xff'
            raw=b'a'*1024**2+b'b'*1024**2+tail; path.write_bytes(raw)
            fact={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}; guards={}; calls=[]
            digest=hashlib.sha256()
            prefixes=[hashlib.sha256(b'a'*1048576).hexdigest(),
                hashlib.sha256(b'a'*1048576+b'b'*1048576).hexdigest(),fact['sha256']]
            advice=os.posix_fadvise
            def advise(fd,offset,count,kind):
                self.assertEqual(digest.hexdigest(),prefixes[len(calls)])
                calls.append((offset,count,kind))
                advice(fd,offset,count,kind)
            with patch.object(self.o.hashlib,'sha256',return_value=digest),patch.object(self.o.os,'posix_fadvise',advise):
                self.assertEqual(self.o.authenticated(fact,guards),path)
            self.assertEqual(calls,[(0,1048576,os.POSIX_FADV_DONTNEED),
                (1048576,1048576,os.POSIX_FADV_DONTNEED),(2097152,6,os.POSIX_FADV_DONTNEED)])
            self.assertEqual(guards,{str(path):fact['sha256']})
            with patch.object(self.o.os,'posix_fadvise',side_effect=OSError('advice failed')):
                self.assertEqual(self.o.authenticated(fact,{},keep=True),raw)
                denied={}
                with self.assertRaisesRegex(OSError,'advice failed'):self.o.authenticated(fact,denied)
                self.assertEqual(denied,{})

    def test_bulk_mutation_during_advice_and_restored_mtime_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bulk'; raw=b'a'*1024**2+b'original tail'; path.write_bytes(raw)
            fact={'path':str(path),'sha256':hashlib.sha256(raw).hexdigest()}; before=path.stat()
            advice=os.posix_fadvise
            def mutate(fd,offset,count,kind):
                advice(fd,offset,count,kind)
                if offset==0:
                    with path.open('r+b') as stream:
                        stream.seek(1024**2);stream.write(b'mutated! tail')
                    os.utime(path,ns=(before.st_atime_ns,before.st_mtime_ns))
            guards={}
            with patch.object(self.o.os,'posix_fadvise',mutate):
                with self.assertRaisesRegex(ValueError,'changed during authentication|SHA'):
                    self.o.authenticated(fact,guards)
            self.assertEqual(guards,{})
            self.assertEqual(path.stat().st_mtime_ns,before.st_mtime_ns)
            with self.assertRaisesRegex(ValueError,'SHA'):self.o.rehash({str(path):fact['sha256']})

    def test_memory_snapshot_is_bounded_scalar_json_from_admitted_path(self):
        self.assertTrue(callable(getattr(self.o,'memory_snapshot',None)),'memory snapshot missing')
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); values={'memory.current':'111','memory.peak':'222',
                'memory.stat':'anon 11\nfile 22\nkernel 33\nfile_mapped 44\n',
                'memory.events':'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n',
                'memory.swap.current':'0','memory.swap.peak':'0'}
            for name,value in values.items():(root/name).write_text(value)
            stderr=io.StringIO()
            with redirect_stderr(stderr):self.o.memory_snapshot(str(root),'after_admission')
            expected={'diagnostic':'memory_snapshot','phase':'after_admission','path':str(root),
                'memory.current':111,'memory.peak':222,'memory.stat.anon':11,'memory.stat.file':22,
                'memory.stat.kernel':33,'memory.swap.current':0,'memory.swap.peak':0,
                **{'memory.events.'+k:0 for k in ('low','high','max','oom','oom_kill','oom_group_kill')}}
            self.assertEqual(json.loads(stderr.getvalue()),expected)
            self.assertLess(len(stderr.getvalue()),2048)
            self.assertTrue(all(type(v) in (str,int) for v in expected.values()))
            for name,bad in [('memory.current','-1'),('memory.stat','anon 11\nfile 22\n'),
                ('memory.events','low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\nmax 1\n'),
                ('memory.peak','0'*16385)]:
                with self.subTest(name=name),redirect_stderr(io.StringIO()):
                    (root/name).write_text(bad)
                    with self.assertRaises((ValueError,KeyError)):self.o.memory_snapshot(str(root),'failure')
                    (root/name).write_text(values[name])

    def test_only_advice_and_diagnostic_seams_invert_complete_production_ast(self):
        phases=('after_admission','before_load_inference','after_load_inference',
                'before_observe','after_observe','after_release_inference')
        summary="""print(json.dumps({'diagnostic': 'observation_summary', 'status': 'UNACCEPTED',
            'pending_final_checks': True,
            **{k: observation[k] for k in ('decision', 'b32', 'b6_repeat_exact',
                'same_b6_cache_gather_exact', 'differences', 'capture_sha256',
                'image_forwards', 'original_tail_cause_established',
                'serving_correction_authorized')}},
            sort_keys=True, allow_nan=False), file=sys.stderr, flush=True)"""
        snippets=['admitted = False','admitted = True',
            'os.posix_fadvise(stream.fileno(), stream.tell() - len(block), len(block), os.POSIX_FADV_DONTNEED)',
            summary, *[f"memory_snapshot(before['path'], '{phase}')" for phase in phases],
            *[f"actions.append(lambda: memory_snapshot(before['path'], '{side}_rehash_{owner}'))"
              for owner in ('context','prospective') for side in ('before','after')]]
        seams={ast.dump(ast.parse(s).body[0],include_attributes=False):0 for s in snippets}
        counts={'helper':0,'conditions':0}
        class Restore(ast.NodeTransformer):
            def visit_FunctionDef(self,node):
                if node.name=='memory_snapshot':
                    counts['helper']+=1;return None
                return self.generic_visit(node)
            def visit_Expr(self,node):
                key=ast.dump(node,include_attributes=False)
                if key in seams:seams[key]+=1;return None
                return self.generic_visit(node)
            visit_Assign=visit_Expr
            def visit_If(self,node):
                diagnostic=ast.unparse(node.test)=='admitted'
                node=self.generic_visit(node)
                if diagnostic and not node.body and not node.orelse:
                    counts['conditions']+=1;return None
                return node
        restored=Restore().visit(ast.parse(DRIVER.read_bytes()))
        self.assertEqual(counts,{'helper':1,'conditions':4})
        self.assertEqual(list(seams.values()),[1]*len(seams))
        digest=hashlib.sha256(ast.dump(restored,include_attributes=False).encode()).hexdigest()
        self.assertEqual(digest,'3416f669a3cee9bbf941334bcbc8d05ab65eae864d81d8ab1afe9c773892346b')

    def test_fixed_sequence_falsification_encoder_and_normalization_drift(self):
        for kwargs,decision in [({},'FIXED_WITNESS_FALSIFIED'),
            ({'drift':True},'FIXED_WITNESS_ENCODER_BATCH_DRIFT'),
            ({'normalization':True},'FIXED_WITNESS_NORMALIZATION_BATCH_DRIFT')]:
            with self.subTest(decision=decision), native_standins(self.o,**kwargs) as f:
                # Match real pixel tensor slicing/clone dimensions at the native boundary.
                getter=Tensor.__getitem__;cloner=Tensor.clone
                def get(t,i):
                    result=getter(t,i)
                    if len(t.shape)==4:result.shape=(len(result.rows),3,256,256)
                    return result
                def clone(t):
                    result=cloner(t);result.shape=t.shape;return result
                with patch.object(Tensor,'__getitem__',get),patch.object(Tensor,'clone',clone):
                    result=self.o.observe(f.d,f.context,f.sources,f.state,f.batch,lambda:None)
                self.assertEqual(result['decision'],decision)
                self.assertEqual([n for k,n in f.calls if k=='encoder'],[32,6,6])
                self.assertEqual([n for k,n in f.calls if k=='readout'],[32,32,6,6,6,6,6,6])
                self.assertTrue(all(i.closed for i in f.images))
                self.assertFalse(result['original_tail_cause_established'])

    def test_unmatched_rgb_output_cache_and_pixel_stop_before_more_forwards(self):
        for mutant in ('rgb','output','cache','pixel','repeat'):
            with self.subTest(mutant=mutant), native_standins(self.o,pixel_mutant=mutant=='pixel',repeat_mutant=mutant=='repeat') as f:
                if mutant=='rgb':f.batch['rgb_sha256']='bad'
                if mutant=='output':f.batch['outputs_sha256']='bad'
                if mutant=='cache':f.sources.modules['baseline'].cache_rows=lambda *a:Tensor(list(range(32)))
                getter=Tensor.__getitem__;cloner=Tensor.clone
                def get(t,i):
                    result=getter(t,i)
                    if len(t.shape)==4:result.shape=(len(result.rows),3,256,256)
                    return result
                def clone(t):
                    result=cloner(t);result.shape=t.shape;return result
                with patch.object(Tensor,'__getitem__',get),patch.object(Tensor,'clone',clone):
                    message={'rgb':'RGB','output':'pixel/output proof','cache':'cache features','pixel':'B6 pixels','repeat':'repeatability'}[mutant]
                    with self.assertRaisesRegex(ValueError,message):self.o.observe(f.d,f.context,f.sources,f.state,f.batch,lambda:None)
                expected=[] if mutant=='rgb' else [32,6,6] if mutant=='repeat' else [32]
                self.assertEqual([n for k,n in f.calls if k=='encoder'],expected)
                self.assertTrue(all(i.closed for i in f.images))

    def test_frozen_readout_and_flag_mutations_stop_at_real_capture_guard(self):
        for key,value in [('vision_sha256','mutant'),('readout_sha256','mutant'),('flags',{})]:
            with self.subTest(key=key),native_standins(self.o) as f:
                f.state[key]=value
                with self.assertRaises(ValueError):
                    self.o.capture_inference_outputs(f.sources.modules['connected'],f.state,f.images[:32])
                self.assertEqual(f.calls,[])
        with native_standins(self.o) as f:
            f.state['head_object'].parameters=lambda:[SimpleNamespace(grad=None,requires_grad=True,dtype='f32',device=SimpleNamespace(type='cuda'))]
            with self.assertRaisesRegex(ValueError,'roles/hooks'):
                self.o.capture_inference_outputs(f.sources.modules['connected'],f.state,f.images[:32])
            self.assertEqual(f.calls,[])

    def test_genuine_release_clears_lru_and_rejects_retained_model(self):
        connected=load(HERE/'train_siglip2_connected_mlp.py','_control_release_test')
        class Owned:
            def parameters(self):return []
            def buffers(self):return []
        class Cache:
            count=1
            def cache_clear(self):self.count=0
            def cache_info(self):return SimpleNamespace(currsize=self.count)
        for retained in (False,True):
            module=SimpleNamespace(__name__='_owned_release_fixture');sys.modules[module.__name__]=module
            cache=Cache();state={k:Owned() for k in ('model','processor_object','head_object','A','C','mu_train')}
            state.update(modules={'one':module},processor_cache=cache,guards={})
            ref=state['model'] if retained else None
            with patch.object(connected,'_processor_cache',lambda *a:cache):
                if retained:
                    with self.assertRaisesRegex(ValueError,'lifetime'):connected.release_inference(state)
                else:connected.release_inference(state)
            self.assertEqual(cache.count,0);self.assertEqual(state,{})
            self.assertNotIn(module.__name__,sys.modules)
            ref=None

    def test_exit_cap_cleanup_and_byte_decisions(self):
        with patch.object(self.o.time,'perf_counter',lambda:700):
            with self.assertRaisesRegex(ValueError,'700'):self.o.elapsed_cap(0)
        events=[]
        def fail():events.append('guard');raise ValueError('source failure')
        with self.assertRaisesRegex(ValueError,'source failure'):
            self.d.cleanup_error(None,[fail,lambda:events.append('release'),lambda:events.append('rehash')])
        self.assertEqual(events,['guard','release','rehash'])
        with self.assertRaisesRegex(ValueError,'same features'):
            self.o.interpretation(False,False,True)
        for a,b in [(b'\x00',b'\x01'),(b'',b'\x00'),(b'\x00\x80',b'\x00\x00')]:
            with self.assertRaises(ValueError):self.o.exact({'x':a},{'x':b},'byte mutant')

    def test_exact_four_native_origins_reject_missing_extra_or_mutant(self):
        expected={f'/site/library{i}.so':str(i) for i in range(4)}
        proof={'authority':{'installed_site_root':'/site'},'comparison':{'selected_members':{
            Path(k).name:{'sha256':v} for k,v in expected.items()}}}
        context={'launch':{'runtime':{'native_authority':{}}},'guards':{},
                 'source_cpu':{'origins':{'files':{'/old.so':'old'}}},'warm':{'origins':{'files':{}}}}
        sources=SimpleNamespace(modules={'nearest':SimpleNamespace(NATIVE_PROOF_PINS={'proof':'pin'})})
        def read(fact,guards):return proof if fact else {'proof':{'sha256':'pin'}}
        with patch.object(self.o,'read_json',read):
            origins={'files':{'/old.so':'old',**expected},'native_files':list(expected)}
            self.o.exact_four(context,sources,origins)
            for mutate in (lambda v:v['files'].pop('/site/library0.so'),
                           lambda v:v['files'].__setitem__('/extra.so','x'),
                           lambda v:v['files'].__setitem__('/site/library0.so','x'),
                           lambda v:v['native_files'].pop()):
                bad=copy.deepcopy(origins);mutate(bad)
                with self.assertRaisesRegex(ValueError,'exact original four'):self.o.exact_four(context,sources,bad)

    def test_run_disposes_before_full_exit_and_never_publishes_failed_exit(self):
        phases=('after_admission','before_load_inference','after_load_inference',
            'before_observe','after_observe','after_release_inference',
            'before_rehash_context','after_rehash_context',
            'before_rehash_prospective','after_rehash_prospective')
        for failure in (None,'admission','observation','exit_rehash','final_cap','live_guard',
                        'observation_with_sample',*[f'sample_{p}' for p in phases]):
            with self.subTest(failure=failure),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                events=[]; guards_live={'mutated':False}; root=Path(tmp)
                flags={'threads':8,'interop_threads':20,'cudnn_allow_tf32':True,'matmul_allow_tf32':False}
                torch=SimpleNamespace(set_num_threads=lambda n:None,get_num_interop_threads=lambda:20,
                    random=SimpleNamespace(get_rng_state=lambda:Tensor([1])),equal=lambda a,b:a.numpy()==b.numpy(),
                    cuda=SimpleNamespace(is_initialized=lambda:False,get_rng_state_all=lambda:[Tensor([1])],
                                         device_count=lambda:1,max_memory_allocated=lambda:0))
                endpoint={'seed':179061,'arm':'control','bundle':{'path':'/control/bundle.json','sha256':'bundle'}}
                cgroup=root/'unit.service';cgroup.mkdir()
                for name,value in {'memory.current':'111','memory.peak':'222',
                    'memory.stat':'anon 11\nfile 22\nkernel 33\n',
                    'memory.events':'low 0\nhigh 0\nmax 0\noom 0\noom_kill 0\noom_group_kill 0\n',
                    'memory.swap.current':'0','memory.swap.peak':'0'}.items():
                    (cgroup/name).write_text(value)
                source=SimpleNamespace(numerical_flags=lambda:flags,cgroup_memory=lambda:{'path':str(cgroup)})
                def admit_cgroup(*a):
                    events.append('cgroup')
                    if failure=='admission':raise ValueError('admission')
                initializer=SimpleNamespace(admit_cgroup=admit_cgroup)
                def release(state):events.append('release');state.clear()
                def load_inference(*a):events.append('load');return {'modules':{},'guards':{}}
                connected=SimpleNamespace(load_inference=load_inference,
                    release_inference=release,mapping_absent=lambda p:events.append('maps'))
                def source_guard(*a,**k):
                    def check():
                        if guards_live['mutated']:raise ValueError('live_guard')
                    return check
                modules={'source_driver':source,'extract':SimpleNamespace(sha=lambda p:'python'),
                    'initializer':initializer,'connected':connected,'evaluator':SimpleNamespace(source_live_guard=source_guard)}
                context={'guards':{'historical':'pin'},'launch':{'sources':{'connected':{},'packing':{'sha256':'packing'}},'original_cache':{'path':'cache'}},
                    'score':{'invocation':{'python':str(Path(sys.executable).resolve()),'python_sha256':'python',
                        'python_version':sys.version},'launch':{'endpoints':[endpoint]}},
                    'exports':{self.o.KEY:{'numerical_flags':flags}},'cpu':{'numerical_flags':flags}}
                prospective={'launch':{'historical':{'source':{'sha256':'source'},'execution':{'sha256':'execution'},
                    'authority':{'path':'/authority.json','sha256':'authority'}},'control':{'bundle':endpoint['bundle']}},
                    'code':{DRIVER.name:'self'},'guards':{'prospective':'pin'},'output':root/'output'}
                def admit():events.append('historical_admit');sys.modules['torch']=torch
                def close():events.append('sources_close');modules.clear()
                def load_packing(name):
                    self.assertEqual(name,'packing');events.append('packing');modules[name]=SimpleNamespace();return modules[name]
                sources=SimpleNamespace(modules=modules,admit=admit,guard=lambda:None,close=close,load=load_packing,checks=[])
                def observe(*a):
                    events.append('observation')
                    if failure in ('observation','observation_with_sample'):raise ValueError('observation')
                    if failure=='live_guard':guards_live['mutated']=True
                    return {'decision':'FIXED_WITNESS_FALSIFIED','b32':{'cache_features_exact':True},
                        'b6_repeat_exact':True,'same_b6_cache_gather_exact':True,
                        'differences':{'pooled':{'exact':True,'different_bytes':0}},
                        'capture_sha256':{'pooled':'a'*64},'image_forwards':44,
                        'original_tail_cause_established':False,'serving_correction_authorized':False}
                def rehash(value):
                    events.append('rehash')
                    if failure=='exit_rehash':raise ValueError(failure)
                def final(budget,src,init,before,rng,fl,receipt):
                    self.assertIs(src,source);self.assertIs(init,initializer)
                    self.assertEqual(modules,{})
                    events.append('final')
                    if failure=='final_cap':raise ValueError(failure)
                diagnostic=SimpleNamespace(prepare=lambda args:context,Sources=lambda c:sources,
                    terminal_admission=lambda *a:set(range(6)),origin_audit=lambda *a:lambda:{},
                    cleanup_error=self.d.cleanup_error,final_state=final,write_json=self.d.write_json)
                memory_snapshot=self.o.memory_snapshot
                def sample(path,phase):
                    self.assertEqual(path,str(cgroup));events.append('snapshot:'+phase)
                    broken=failure=='sample_'+phase or (failure=='observation_with_sample' and phase=='after_release_inference')
                    if broken:(cgroup/'memory.current').write_text('-1')
                    try:memory_snapshot(path,phase)
                    finally:(cgroup/'memory.current').write_text('111')
                patches={'prepare':lambda a:prospective,'load_source':lambda *a:diagnostic,
                    'bind_control':lambda *a:(events.append('prospective_bind') or endpoint,{}),
                    'authenticated':lambda *a,**k:b'', 'check_capture_ast':lambda *a:None,
                    'check_endpoint':lambda *a:None,'observe':observe,'exact_four':lambda *a:None,
                    'rehash':rehash,'remove_source':lambda m:events.append('remove'),'memory_snapshot':sample}
                for name,value in patches.items():stack.enter_context(patch.object(self.o,name,value))
                stack.enter_context(patch.dict('os.environ',{'CUDA_VISIBLE_DEVICES':'0','CUBLAS_WORKSPACE_CONFIG':':4096:8',
                                                         'INVOCATION_ID':'a'*32}))
                stack.enter_context(patch.dict(sys.modules,{}))
                stderr=stack.enter_context(redirect_stderr(io.StringIO()))
                args=SimpleNamespace(authority=root/'authority.json',authority_sha256='authority',output=root/'output')
                if failure:
                    message='memory diagnostic' if failure.startswith('sample_') else 'observation' if failure=='observation_with_sample' else failure
                    with self.assertRaisesRegex(ValueError,message) as raised:self.o.run(args)
                    if failure=='observation_with_sample':
                        self.assertTrue(any('memory diagnostic' in note for note in getattr(raised.exception,'__notes__',[])))
                    self.assertFalse((root/'output/receipt.json').exists())
                else:
                    self.o.run(args)
                    self.assertTrue((root/'output/receipt.json').is_file())
                self.assertLess(events.index('historical_admit'),events.index('prospective_bind'))
                if 'observation' in events:self.assertLess(events.index('packing'),events.index('observation'))
                if 'release' in events:self.assertLess(events.index('release'),events.index('rehash'))
                self.assertEqual(events.count('rehash'),2)
                self.assertLess(events.index('rehash'),events.index('sources_close'))
                if 'final' in events:self.assertLess(events.index('sources_close'),events.index('final'))
                self.assertEqual(events[-1],'remove')
                logs=[json.loads(line) for line in stderr.getvalue().splitlines()]
                if failure=='admission':self.assertEqual(logs,[],'unadmitted cgroup was sampled')
                summaries=[v for v in logs if v['diagnostic']=='observation_summary']
                returned='observation' in events and failure not in ('observation','observation_with_sample')
                self.assertEqual(len(summaries),int(returned))
                if summaries:
                    self.assertEqual(summaries[0]['status'],'UNACCEPTED')
                    self.assertIs(summaries[0]['pending_final_checks'],True)
                    self.assertEqual(summaries[0]['image_forwards'],44)
                if failure is None:
                    self.assertEqual([v['phase'] for v in logs if v['diagnostic']=='memory_snapshot'],list(phases))
                    self.assertLess(events.index('snapshot:before_load_inference'),events.index('load'))
                    self.assertLess(events.index('load'),events.index('snapshot:after_load_inference'))
                    self.assertLess(events.index('snapshot:before_observe'),events.index('observation'))
                    self.assertLess(events.index('observation'),events.index('snapshot:after_observe'))
                    self.assertLess(events.index('release'),events.index('snapshot:after_release_inference'))
                    rehashes=[i for i,event in enumerate(events) if event=='rehash']
                    for i,owner in enumerate(('context','prospective')):
                        self.assertLess(events.index('snapshot:before_rehash_'+owner),rehashes[i])
                        self.assertLess(rehashes[i],events.index('snapshot:after_rehash_'+owner))

    def test_cli_and_no_native_imports(self):
        self.assertFalse({'torch','numpy','PIL','transformers','torchvision','sfora'}.intersection(sys.modules))
        result=subprocess.run([sys.executable,'-B',str(DRIVER),'--help'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        result=subprocess.run([sys.executable,'-B','-O',str(DRIVER),'--help'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        with patch.dict(sys.modules,{'torch':SimpleNamespace()}):
            with self.assertRaisesRegex(ValueError,'native imports'):self.o.run(SimpleNamespace())


if __name__ == '__main__':
    unittest.main()
