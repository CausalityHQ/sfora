"""Run with python3 -B -S; authority/ownership/dispatch checks, no native runtime."""
import ast
import copy
import gc
import hashlib
import json
import math
import sys
import unittest
import weakref
from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import train_role_matched_adaptation as driver

PATH = Path(driver.__file__)
PARENT = Path('/home/rb/worktrees/sfora-positive-causality')
EVIDENCE = Path('docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1')


def function(name, namespace):
    node = next(n for n in ast.parse(PATH.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(PATH), 'exec'), namespace)
    return namespace[name]


class Tensor:
    """Only layout/scalar authority, never a simulated model or update."""
    def __init__(self, shape=(), dtype='float32', value=0.):
        self.shape, self.dtype, self.value = shape, dtype, value
    def __float__(self): return float(self.value)
    def numel(self): return math.prod(self.shape)
    def __repr__(self): return repr((self.shape,self.dtype,self.value))


def resume_fixture(step=8, cpu=False):
    fingerprint = lambda value: hashlib.sha256(repr(value).encode()).hexdigest()
    namespace = {'torch':SimpleNamespace(Tensor=Tensor, float32='float32',
        isfinite=lambda v:SimpleNamespace(all=lambda:math.isfinite(v.value)), equal=lambda a,b:repr(a) == repr(b)),
        'old':SimpleNamespace(fingerprint=fingerprint), 'RESUME_KEYS':driver.RESUME_KEYS,
        'RESUME_SCHEMA':driver.RESUME_SCHEMA, 'METHOD':driver.METHOD,
        'MASK':driver.MASK, 'objective_name':driver.objective_name}
    function('tensor_layout',namespace)
    check = function('validate_resume',namespace)
    members = [(str(i),Tensor((2,))) for i in range(208)]
    groups = [{'params':list(range(205)), 'lr':1e-5, 'weight_decay':.05},
              {'params':[205,206], 'lr':1e-4, 'weight_decay':.05},
              {'params':[207], 'lr':1e-4, 'weight_decay':.05}]
    base = {'schema':driver.RESUME_SCHEMA, 'intervention':driver.METHOD, 'arm':'role_matched',
            'objective':driver.METHOD, 'role_positive_sha256':driver.MASK,
            'source_checkpoint_sha256':driver.SOURCE, 'objective_sha256':'objective',
            'cpu_authority_sha256':driver.CPU_PROOF, 'cpu_log_sha256':driver.CPU_LOG,
            'initial_rng_sha256':'initial-rng', 'code':{'trainer':'code'},
            'parameter_names':[n for n,_ in members],
            'precision':'cpu_float32' if cpu else 'native_float32_fp16_autocast',
            'frozen_names':['v0','embeddings.position_ids']}
    saved = {'vision':{f'v{i}':Tensor((2,)) for i in range(400)},
             'buffers':{'embeddings.position_ids':Tensor((1,256),'int64')},
             'head':{'weight':Tensor((128,1024)), 'bias':Tensor((128,))},
             'classifier':Tensor((2004,128)), 'bank':Tensor((13283,128)),
             'optimizer':{'param_groups':groups, 'state':{i:{'step':Tensor(value=step),
                 'exp_avg':Tensor(p.shape), 'exp_avg_sq':Tensor(p.shape)} for i,(_,p) in enumerate(members)} if step else {}},
             'scaler':None if cpu else {'scale':128., 'growth_factor':2., 'backoff_factor':.5, 'growth_interval':2000, '_growth_tracker':step},
             'cpu_rng':Tensor((32,),'uint8'), 'cuda_rng':[] if cpu else [Tensor((32,),'uint8')]}
    base['buffers_sha256'] = fingerprint(saved['buffers'])
    named = {**saved['vision'], **saved['buffers']}
    base['frozen_sha256'] = fingerprint({n:named[n] for n in base['frozen_names']})
    saved['identity'] = {**base,'global_step':step}
    return saved,base,groups,members,check


class Authorities(unittest.TestCase):
    def setUp(self):
        root = PATH.parents[1] if (PATH.parents[1] / EVIDENCE).exists() else PARENT
        self.cpu = driver.read_authority(root / EVIDENCE / 'role-matched-cpu-v3.json', driver.CPU_PROOF)
        self.log = driver.read_authority(root / EVIDENCE / 'role-matched-cpu-v3.log', driver.CPU_LOG, log=True)
        self.code = {**self.cpu['code'], **dict.fromkeys(driver.ADDED, 'new')}

    def tearDown(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('numpy', sys.modules)

    def test_actual_cpu_and_original_log(self):
        driver.validate_cpu(self.cpu, driver.CPU_EXECUTION, self.code, self.cpu['code'])
        driver.validate_log(self.log, self.cpu, 120)

    def test_source_mask_and_closure_negatives(self):
        for key, value in (('source_checkpoint_sha256', 'foreign'), ('initial_state_sha256', 'foreign'),
                           ('role_positive_sha256', 'foreign'), ('module_sha256', 'foreign'),
                           ('boundary', 10), ('optimizer_members', 209), ('optimizer_updates', 1),
                           ('quality_read', True), ('RNG_preserved', False)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                driver.validate_cpu({**self.cpu, key: value}, driver.CPU_EXECUTION, self.code, self.cpu['code'])
        for name in self.cpu['code']:
            with self.subTest(name=name), self.assertRaises(AssertionError):
                driver.validate_cpu(self.cpu, driver.CPU_EXECUTION, {**self.code, name: 'changed'}, self.cpu['code'])
        with self.assertRaises(AssertionError):
            driver.validate_cpu(self.cpu, driver.CPU_EXECUTION, {**self.code, 'extra': 'extra'}, self.cpu['code'])

    def test_log_binds_unit_locks_native_resources(self):
        for old, new in (('status=0', 'status=1'), ('4133df57f4c6435382e3c0ac00416f67', 'foreign'),
                         ('5198156', '9000000'), ('27.251s', '121s'), ('Memory swap peak: 0B', 'Memory swap peak: 1B'),
                         ('flock -n /home/riomus/runs/.sfora-siglip2-gpu.lock', 'missing')):
            with self.subTest(old=old), self.assertRaises(AssertionError):
                driver.validate_log(self.log.replace(old, new), self.cpu, 120)

    def test_phase_limits_and_complete_authority_arguments(self):
        options = dict(phase='startup', arm='control', seed=179032, boundary=12,
                       cpu_execution_sha256=driver.CPU_EXECUTION, cpu_sha256=driver.CPU_PROOF,
                       cpu_log_sha256=driver.CPU_LOG)
        for name in ('startup', 'mechanics'):
            options.update({name+'_'+suffix: None for suffix in ('proof', 'sha256', 'log', 'log_sha256')})
        driver.validate_arguments(SimpleNamespace(**options))
        for changes in ({'phase': 'mechanics', 'arm': 'control'}, {'boundary': 10}, {'seed': 179039},
                        {'phase': 'train'}, {'cpu_sha256': 'foreign'}):
            with self.subTest(changes=changes), self.assertRaises(AssertionError):
                driver.validate_arguments(SimpleNamespace(**{**options, **changes}))

    def test_exclusive_publish_preserves_existing_bytes(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'receipt.json'
            driver.atomic_write(path, lambda f: f.write(b'first'))
            with self.assertRaises(AssertionError):
                driver.atomic_write(path, lambda f: f.write(b'second'))
            self.assertEqual(path.read_bytes(), b'first')
            self.assertFalse(path.with_name(path.name+'.part').exists())

    def test_complete_resume_rejects_arm_mask_objective_source_foreign_state(self):
        for cpu,step in ((True,0), (False,8), (False,17), (False,100)):
            saved,base,groups,members,check = resume_fixture(step,cpu)
            check(saved,base,step,groups,saved,members)
            for key in driver.RESUME_KEYS:
                bad = dict(saved); del bad[key]
                with self.subTest(cpu=cpu,step=step,key=key), self.assertRaises(AssertionError):
                    check(bad,base,step,groups,saved,members)
            for key in ('arm','role_positive_sha256','objective','objective_sha256','source_checkpoint_sha256',
                        'schema','cpu_authority_sha256','cpu_log_sha256','code','initial_rng_sha256','global_step'):
                bad = {**saved,'identity':{**saved['identity'],key:'foreign'}}
                with self.subTest(key=key), self.assertRaises(AssertionError):
                    check(bad,base,step,groups,saved,members)
            for extra in ('residual','roles','positive','foreign'):
                with self.subTest(extra=extra), self.assertRaises(AssertionError):
                    check({**saved,extra:Tensor()},base,step,groups,saved,members)

    def test_saved_optimizer_layout_rng_scaler_and_frozen_negatives(self):
        saved,base,groups,members,check = resume_fixture()
        for case in ('member','duplicate','options','step','moment','shape','dtype','nonfinite','counter',
                     'scaler','rng','frozen','buffer','head','bank','classifier'):
            bad = copy.deepcopy(saved)
            if case == 'member': del bad['optimizer']['state'][207]
            elif case == 'duplicate': bad['optimizer']['param_groups'][0]['params'][-1] = 0
            elif case == 'options': bad['optimizer']['param_groups'][0]['lr'] = 1e-3
            elif case == 'step': bad['optimizer']['state'][207]['step'].value = 8.5
            elif case == 'moment': del bad['optimizer']['state'][207]['exp_avg']
            elif case == 'shape': bad['optimizer']['state'][207]['exp_avg'].shape = (3,)
            elif case == 'dtype': bad['optimizer']['state'][207]['exp_avg'].dtype = 'float16'
            elif case == 'nonfinite': bad['optimizer']['state'][207]['exp_avg'].value = float('nan')
            elif case == 'counter': bad['identity']['global_step'] = 1000
            elif case == 'scaler': bad['scaler']['_growth_tracker'] = 7
            elif case == 'rng': bad['cuda_rng'][0].dtype = 'float32'
            elif case == 'frozen': bad['vision']['v0'].value = 1
            elif case == 'buffer': bad['buffers']['embeddings.position_ids'].value = 1
            elif case == 'head': bad['head']['bias'].shape = (127,)
            else: bad[case].shape = (1,128)
            with self.subTest(case=case), self.assertRaises(AssertionError):
                check(bad,base,8,groups,saved,members)

    def test_startup_and_mechanics_reject_foreign_admission(self):
        args = SimpleNamespace(execution_sha256='execution',cpu_sha256=driver.CPU_PROOF,
                               cpu_log_sha256=driver.CPU_LOG,startup_sha256='startup',startup_log_sha256='startup-log')
        schedules = {str(seed):{'input_authority_sha256':digest,'schedule_sha256':str(seed),
                               'class_sequence_sha256':'classes'} for seed,digest in driver.INPUT_AUTHORITIES.items()}
        bases = {}
        for arm in ('control','role_matched'):
            bases[arm] = {'schema':driver.RESUME_SCHEMA,'intervention':driver.METHOD,'arm':arm,
                'precision':'cpu_float32','boundary':12,'width':128,'objective':driver.objective_name(arm),
                'role_positive_sha256':driver.MASK,'code':self.code,
                'module_sha256':self.code['role_matched_bank_rank.py'],'execution_sha256':'execution',
                'source_checkpoint_sha256':driver.SOURCE,'initial_state_sha256':driver.INITIAL,
                'cpu_authority_sha256':driver.CPU_PROOF,'cpu_log_sha256':driver.CPU_LOG,
                'cpu_execution_sha256':driver.CPU_EXECUTION,'parameter_names':list(range(208)), 'optimizer_groups':[{}, {}, {}]}
        startup = {**{k:self.cpu[k] for k in ('unit_invocation_id','unit_cgroup','unit_memory_max_bytes',
                                             'unit_memory_swap_max_bytes','host_max_rss_kib','host_swap_kib','total_seconds')},
            'schema':driver.SCHEMA,'intervention':driver.METHOD,'pass':True,'phase':'startup','arm':'control',
            'seed':179032,'boundary':12,'execution_sha256':'execution','cpu_authority_sha256':driver.CPU_PROOF,
            'cpu_log_sha256':driver.CPU_LOG,'cpu_execution_sha256':driver.CPU_EXECUTION,
            'source_checkpoint_sha256':driver.SOURCE,'initial_state_sha256':driver.INITIAL,'role_positive_sha256':driver.MASK,
            'optimizer_members':{'control':208,'role_matched':208},'optimizer_updates':0,'quality_read':False,
            'checkpoint_sha256':None,'schedules':schedules,'resume_identities':bases,'code':self.code,
            'peak_cuda_allocated_bytes':0,**dict.fromkeys(driver.STARTUP_FACTS,True)}
        driver.validate_startup(startup,args,self.cpu)
        rows = [{'step':i,'optimizer_counter':i,'augmentation_step':1000+i,'image_ids':list(range(64)),
                 'ce':1.,'rank':.5,'loss':5.,'scale':128.,'preclip_norm':1.,'seconds':1.,
                 'gradient_norms':dict.fromkeys((str(j) for j in range(12,24)),1.)} for i in range(1,18)]
        mechanics = {**startup,'phase':'mechanics','arm':'role_matched','optimizer_members':208,
            'startup_authority_sha256':'startup','startup_log_sha256':'startup-log','updates':17,'completed_step':17,
            'chunk100_admission_seconds':200.,'schedule_sha256':'179032','steps':rows,'resumed_steps':rows[8:],
            'resume_identity':{**bases['role_matched'],'schedule_sha256':'179032','class_sequence_sha256':'classes'},
            'peak_cuda_allocated_bytes':1000,**dict.fromkeys(driver.MECHANICS_FACTS,True)}
        driver.validate_mechanics(mechanics,args,self.cpu,startup)
        for key,value in (('seed',179039),('arm','control'),('boundary',10),('updates',16),
                          ('role_positive_sha256','foreign'),('startup_authority_sha256','foreign'),
                          ('checkpoint_sha256','retained'),('chunk100_admission_seconds',270.),
                          ('chunk100_admission_seconds',float('nan')),('host_swap_kib',1),
                          ('peak_cuda_allocated_bytes',10_000_000_000),('total_seconds',120.)):
            with self.subTest(key=key), self.assertRaises(AssertionError):
                driver.validate_mechanics({**mechanics,key:value},args,self.cpu,startup)
        for arm in bases:
            bad = copy.deepcopy(startup); bad['resume_identities'][arm]['objective'] = 'old-objective'
            with self.assertRaises(AssertionError): driver.validate_startup(bad,args,self.cpu)


class OwnershipAndDispatch(unittest.TestCase):
    def tearDown(self):
        self.assertNotIn('torch', sys.modules)

    def test_restore_clones_only_steps_releases_mapping_before_live_fingerprint(self):
        class Allocation: pass
        class Step:
            def __init__(self, owner=None): self.owner = owner
            def clone(self): return Step()
        mapped = []
        def load(*a, **kw):
            owner = Allocation(); mapped.append(weakref.ref(owner))
            return {'optimizer': {'param_groups': [], 'state': {0: {'step': Step(owner), 'exp_avg': owner, 'exp_avg_sq': owner}}},
                    'vision': {}, 'head': {}, 'buffers': {}, 'cpu_rng': 0, 'cuda_rng': [], 'classifier': 0, 'bank': 0}
        class Optimizer:
            def load_state_dict(self, value): self.step = value['state'][0]['step']
        optimizer = Optimizer()
        model = SimpleNamespace(load_state_dict=lambda *a, **kw: None, named_buffers=lambda: [])
        state = {'optimizer': optimizer, 'model': model, 'head': model, 'scaler': None, 'params': []}
        state.update({k: SimpleNamespace(copy_=lambda _: None) for k in ('classifier', 'bank')})
        calls = []
        def fingerprint(value):
            if calls: self.assertIsNone(mapped[0](), 'mmap alive at live fingerprint')
            calls.append(1); return 'fingerprint'
        namespace = {'torch': SimpleNamespace(load=load, no_grad=nullcontext,
                         random=SimpleNamespace(set_rng_state=lambda _: None)),
                     'verify': lambda *a: None, 'sha': lambda _: 'sha', 'gc': gc,
                     'payload': lambda *a: {'optimizer': {'param_groups': []}},
                     'validate_resume': lambda *a: None, 'old': SimpleNamespace(fingerprint=fingerprint)}
        function('own_optimizer_steps', namespace)
        function('restore', namespace)(state, {}, 'path', 'sha', 'fingerprint', 8)
        self.assertIsNone(mapped[0]()); self.assertIsNone(optimizer.step.owner)
        self.assertEqual(state['counter'], 8)

    def test_candidate_routes_unconditionally_and_control_is_original(self):
        calls = []
        class Table:
            def __getitem__(self, index): return ('role-positive', index)
        def original(raw, bank, head, positive, index):
            calls.append(('control', positive)); return 3
        lane = SimpleNamespace(valid_rank=original)
        # Exercise the actual corrected terms source, with arithmetic stubbed only.
        source = PATH.with_name('train_pe_teacher_retained256.py')
        if not source.exists(): source = PARENT / 'scripts/train_pe_teacher_retained256.py'
        tree = ast.parse(source.read_text())
        terms = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'terms')
        objective_ns = {'lane': lane, 'compact_head_features': lambda *a, **kw: 'raw',
                        'pair': SimpleNamespace(smoke=SimpleNamespace(sharded_mask_arcface_loss=lambda *a, **kw: 7)),
                        'torch': SimpleNamespace(arange=lambda *a, **kw: SimpleNamespace(unsqueeze=lambda _: None))}
        exec(compile(ast.Module(body=[terms], type_ignores=[]), str(source), 'exec'), objective_ns)
        def rank(*args, **kw): calls.append(('role_matched', args[2], kw['roles'])); return 5
        coverage = SimpleNamespace(terms='old-closure')
        state = {'head': SimpleNamespace(out_features=128), 'bank': SimpleNamespace(shape=(2,128)),
                 'classifier': SimpleNamespace(shape=(2,128)), 'target': {'index': 0},
                 'positive': {'index': 'original-positive'}, 'role_positive': Table(), 'role_roles': 'roles'}
        def step(s, pixels, ids, active, micro):
            return coverage.terms(SimpleNamespace(device='cpu'), s['head'], s['classifier'], s['bank'],
                                  s['target'], s['positive'], 'index', active)
        namespace = {'patch': patch, 'native': SimpleNamespace(objective=SimpleNamespace(terms=objective_ns['terms'], lane=lane)),
                     'old': SimpleNamespace(coverage=coverage, step=step), 'role': SimpleNamespace(bank_rank_loss=rank)}
        run = function('corrected_step', namespace)
        for active in (False, True):
            self.assertEqual(run(state, None, None, active, 'role_matched'), (7, 5, 'raw'))
            self.assertEqual(run(state, None, None, active, 'control'), (7, 3, 'raw'))
            self.assertIs(lane.valid_rank, original); self.assertEqual(coverage.terms, 'old-closure')
        self.assertEqual([c[0] for c in calls], ['role_matched', 'control']*2)
        with patch.object(namespace['old'], 'step', side_effect=RuntimeError('stop')):
            with self.assertRaises(RuntimeError): run(state, None, None, True, 'role_matched')
        self.assertIs(lane.valid_rank, original); self.assertEqual(coverage.terms, 'old-closure')

    def test_independent_cuda_reload_releases_mmap_before_live_checks(self):
        events = []
        class Value:
            def copy_(self, _): pass
        class Model:
            config = {}
            def __init__(self, *a): pass
            def eval(self): return self
            def float(self): return self
            def state_dict(self): return {str(i):Value() for i in range(400)}
            def named_buffers(self): return [('position',Value())]
            def load_state_dict(self, value, strict): self.strict = strict
        class Disk(dict): pass
        mapped = [None]
        def load(*a, **kw):
            self.assertTrue(kw['mmap']); self.assertEqual(kw['map_location'],'cpu')
            value = Disk(vision={str(i):Value() for i in range(400)},head={},buffers={'position':Value()})
            mapped[0] = weakref.ref(value); events.append('map'); return value
        def device(name):
            self.assertEqual(name,'cuda'); events.append('cuda'); return nullcontext()
        calls = []
        def fingerprint(value):
            if len(calls) >= 3: self.assertIsNone(mapped[0](),'mmap alive during live fingerprint')
            calls.append(1); return 'same'
        def forward(*a): raise RuntimeError('live-forward')
        namespace = {'torch':SimpleNamespace(load=load,device=device,no_grad=nullcontext,
                         random=SimpleNamespace(fork_rng=lambda **kw:nullcontext()),nn=SimpleNamespace(Linear=lambda *a:Model())),
                     'copy':copy,'gc':gc,'rng_fingerprint':lambda _: 'rng',
                     'old':SimpleNamespace(fingerprint=fingerprint,previous=SimpleNamespace(training=SimpleNamespace(fp16=forward))),
                     'qualified':SimpleNamespace(late=SimpleNamespace(frozen_state=lambda *a:{}))}
        with self.assertRaisesRegex(RuntimeError,'live-forward'):
            function('strict_reload',namespace)({'model':Model(),'head':Model()},'checkpoint',None,
                                                {'buffers_sha256':'same','frozen_sha256':'same'})
        self.assertLess(events.index('cuda'),events.index('map'))
        self.assertGreater(len(calls),3)

    def test_rebuild_preserves_initial_rng_identity_without_resetting_live_rng(self):
        current = {'rng':'initial'}
        namespace = {'native':SimpleNamespace(identity=lambda *a: {}), 'RESUME_SCHEMA':driver.RESUME_SCHEMA,
                     'METHOD':driver.METHOD,'MASK':driver.MASK,'objective_name':driver.objective_name,
                     'INPUT_AUTHORITIES':driver.INPUT_AUTHORITIES, 'rng_fingerprint':lambda _: current['rng'],
                     'old':SimpleNamespace(fingerprint=repr,coverage=SimpleNamespace(
                         teacher=SimpleNamespace(qualified=SimpleNamespace(numerical_flags=lambda: {}))))}
        state = {'scaler':None,'head':SimpleNamespace(named_parameters=lambda: []),
                 'classifier':SimpleNamespace(requires_grad=True)}
        args = SimpleNamespace(arm='role_matched',startup_sha256='startup',startup_log_sha256='log')
        code = dict.fromkeys(('role_matched_bank_rank.py','train_pe_teacher_retained256.py',
                              'pe_native_valid_anchor.py','src/sfora/deployed_code_rank.py'),'code')
        fn = function('identity',namespace)
        chosen = {'schedule_sha256':'schedule','class_sequence_sha256':'classes'}
        original = fn(state,args,'initial',chosen,code)
        current['rng'] = 'after-second-factory'
        self.assertNotEqual(fn(state,args,'initial',chosen,code),original)
        self.assertEqual(fn(state,args,'initial',chosen,code,initial_rng_sha256=original['initial_rng_sha256']),original)
        self.assertEqual(current['rng'],'after-second-factory')


if __name__ == '__main__':
    unittest.main()
