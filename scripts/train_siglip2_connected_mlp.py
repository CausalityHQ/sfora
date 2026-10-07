#!/usr/bin/env python3
"""Frozen connected last-MLP method. Native qualification is UNRUN.

Durable implementation/launch contract (the parent freezes actual hashes):
 execution.json: exactly FILES; CLI --execution-sha256 SHA --authority FILE
 --authority-sha256 SHA --phase cpu|mechanics|train --arm control|candidate
 --seed 179061|179069 --output NEWDIR. LAUNCH_KEYS is the exact launch schema.
 original_cpu={authority:FILE,terminal:UNIT} is unchanged source-v5 CPU500;
 actual_gradient={authority:FILE,terminal:UNIT} is admitted actual-v2 CPU300.
 witness={root:absolute canonical directory,files:exact WITNESS_FILES hashes}.
 FILE={path:absolute canonical regular file,sha256:actual SHA256}; UNIT has
 receipt:FILE,log:FILE,unit,invocation_id,service_seconds,native_peak_rss_kib,
 both_locks_held:true. No inferred/future hashes. Historical policies unchanged.
 selected_cpu is null only for cpu, otherwise one NEW CPU300 UNIT. CPU launch
 is control061; it validates both CONTROL initializers and discards a genuine
 candidate061 B64/two-view update, independently restores it and qualifies its
 nonzero-four public bundle. selected_mechanics is null except TRAIN, where
 {control:UNIT,candidate:UNIT} selects NEW mechanics300 for the requested seed.
 Each mechanics arm runs uninterrupted17 vs independent8+save+release+restore+9.
 Fresh TRAIN128 replays the selected same-arm/same-seed first17 diagnostics.
 Both arms always use CONTROL1008/6355 accepted seed-specific CPU-v5 state.
 Only candidate adds four absolute layer26 MLP parameters; no loss changes.
 Core is the complete arm work window, conservatively including initializer,
 construction, pixels, optimizer/integrity, serialization/restore and portable
 qualification. Whole service also includes admission/full exit and terminal
 serialization. Individual phase timings are retained, with no subtraction.
 TRAIN candidate additionally names a fresh same-seed control UNIT in
 fresh_control, null in other phases and for control. Both ratios <=1.50;
 final whole-service ratio is checked from normal-exit UNITs by the parent
 and by admit_terminal, never from a historical cached-feature threshold.
 Payload schema retains all original STATIC_KEYS and complete typed identity,
 optimizer/scaler/counter/numerics/RNG, adds original base_vision FILE+typed
 digest, exact four absolute FP32 encoder tensors and full updated448 digest.
 Original source/source_proof always describes the original base separately.
 Serving closure: FILES + SERVING_FILES + joint_relational_compaction.py;
 no witness/historical trainer imports or TRAIN files in the public loader.
 APIs: load_initializer, fresh, payload, check_payload, integrity, save,
 restore, update, export_bundle, load_inference, inference_outputs, release,
 release_inference. State/checkpoint/bundle storage and processor lifetimes
 are independent and audited. Gates: ownCPU300/mechanics300/TRAIN600, 8GiB,
 no swap/events, CUDA allocation <10GB, both locks, original uncached exit.
 Source tests authorize no native/model-fit/quality/cost/state reuse claims.
"""
import argparse
import copy
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import gc
import hashlib
import importlib.util
import inspect
import json
import math
import os
from pathlib import Path
import re
import resource
import shutil
import sys
import time
from types import CodeType, FunctionType, SimpleNamespace
import weakref

if not __debug__:
    raise SystemExit('qualification requires assertions')

STARTED = time.perf_counter()
HERE = Path(__file__).absolute().parent
SCHEMA = 'siglip2-connected-mlp-v1'
AUTHORITY_SCHEMA = 'siglip2-connected-mlp-launch-v1'
INFERENCE_SCHEMA = 'siglip2-connected-mlp-inference-v1'
BUNDLE_SCHEMA = 'siglip2-connected-mlp-bundle-v1'
FILES = {'train_siglip2_connected_mlp.py', 'test_siglip2_connected_mlp.py'}
SERVING_FILES = {'qualify_siglip2_substrate_cpu.py', 'extract_siglip2_vision_source.py',
                 'train_siglip2_cached_readout.py', 'train_siglip2_substrate_adaptation.py',
                 'prototype_residual_readout.py', 'quadratic_readout.py'}
WITNESS_FILES = {'train_siglip2_cached_readout.py', 'connected_residual_readout.py',
                 'prototype_residual_readout.py', 'quadratic_readout.py',
                 'qualify_connected_encoder_gradients.py',
                 'qualify_actual_objective_encoder_gradients.py', 'test_actual_objective_encoder_gradients.py'}
NATIVE = {'torch','numpy','PIL','transformers','safetensors','torchvision','sfora'}
ARMS, SEEDS, VIEWS = ('control','candidate'), (179061,179069), ('canonical','augmented')
MLP = tuple('encoder.layers.26.mlp.'+layer+'.'+field for layer in ('fc1','fc2') for field in ('weight','bias'))
MLP_SHAPES = [[4304,1152],[4304],[1152,4304],[1152]]
ADAM = {'lr':1e-4,'betas':(.9,.999),'eps':1e-8,'weight_decay':.05,'amsgrad':False,
        'maximize':False,'foreach':False,'capturable':False,'differentiable':False,'fused':False}
RECIPE = {'adamw':{**ADAM,'betas':list(ADAM['betas'])},'encoder_lr':1e-5,'batch':64,
          'microbatch':16,'updates':128,'seeds':list(SEEDS),'views':list(VIEWS),
          'classes':{'control':1008,'candidate':1008},'rows':6355,'clip':1.,'scaler':128.,
          'training':'FP32 autocast disabled','serving':'CUDA FP16 autocast / CPU FP32',
          'objective':'unchanged original CONTROL complete-gallery SmoothAP and P[label]',
          'encoder_names':list(MLP),'encoder_shapes':MLP_SHAPES,'cost_ratio':1.50}
LAUNCH_KEYS = {'schema','execution_sha256','phase','arm','seed','recipe','resource_policy',
               'both_locks_held','original_cpu','actual_gradient','witness','selected_cpu',
               'selected_mechanics','fresh_control'}
BACKEND_SHA = '250394884a9f90845cf87b6fc0cf3341337b193428556c6ff61ed2b04bedb692'
SCOPE_SHA256 = '55cde4ef9de3c3636da2215c706115f36b90436f47fc23b345f72cc168874726'
CONTROL_SHA256 = '1f3ad34bbd20a3b375ccb4908f9a3da05b63b514395cb553e1c81925789b2280'
STATIC_KEYS = ('provenance','config','buffers','processor','head','classifier','means',
               'partition','original_rows','target','schedules','views','teachers','mu_train',
               'mu_train_provenance','scope','common_statistics','cache_provenance','masks','schedule_provenance')
PAYLOAD_KEYS = {'schema','identity','source','A','C',*STATIC_KEYS,'optimizer','scaler','counter',
                'cpu_rng','cuda_rng','numerical_flags','base_vision','encoder','vision_sha256'}
IDENTITY_KEYS = {'method','source','arm','seed','device','scope','initializer','validator_identity','encoder_identity',
                 'base_vision','parameter_names','parameter_shapes','optimizer_defaults','optimizer_groups',
                 'initial_scaler','numerical_flags'}
INFERENCE_KEYS = {'schema','source','arm','config','buffers','processor','head','A','C','means',
                  'mu_train','mu_train_provenance','scope','common_statistics','numerical_flags',
                  'base_vision','encoder','encoder_identity','vision_sha256','fixed_sha256'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy(phase):
    require(phase in ('cpu','mechanics','train'), 'fixed phase required')
    return {'seconds':600 if phase == 'train' else 300,'host_bytes':8*1024**3,
            'swap_bytes':0,'cuda_allocated_bytes_exclusive':10_000_000_000}


def parameter_roles(arm):
    require(type(arm) is str and arm in ARMS, 'fixed arm required')
    names, shapes = ['A','C'], [[128,160],[128,1152]]
    if arm == 'candidate':
        names += list(MLP)
        shapes += copy.deepcopy(MLP_SHAPES)
    return names, shapes, sum(math.prod(shape) for shape in shapes)


def check_launch(launch, args):
    require(isinstance(launch,dict) and launch.keys() == LAUNCH_KEYS and
            launch['schema'] == AUTHORITY_SCHEMA and launch['execution_sha256'] == args.execution_sha256 and
            launch['phase'] == args.phase and launch['arm'] == args.arm and args.arm in ARMS and
            type(launch['seed']) is int and type(args.seed) is int and launch['seed'] == args.seed in SEEDS and
            launch['recipe'] == RECIPE and json.dumps(launch['recipe'],sort_keys=True,allow_nan=False) ==
            json.dumps(RECIPE,sort_keys=True,allow_nan=False) and launch['resource_policy'] == policy(args.phase) and
            all(type(v) is int for v in launch['resource_policy'].values()) and
            launch['both_locks_held'] is True and
            (args.phase != 'cpu' or (args.arm == 'control' and args.seed == SEEDS[0])) and
            (launch['selected_cpu'] is None) == (args.phase == 'cpu') and
            (launch['selected_mechanics'] is None) == (args.phase != 'train') and
            (launch['fresh_control'] is None) == (args.phase != 'train' or args.arm == 'control'),
            'frozen connected launch differs')
    for name in ('original_cpu','actual_gradient'):
        value = launch[name]
        require(isinstance(value,dict) and value.keys() == {'authority','terminal'}, 'launch admission differs')
        file_fact(value['authority'])
        check_unit(value['terminal'])
    witness = launch['witness']
    require(isinstance(witness,dict) and witness.keys() == {'root','files'} and
            isinstance(witness['root'],str) and Path(witness['root']).is_absolute() and
            isinstance(witness['files'],dict) and witness['files'].keys() == WITNESS_FILES and
            all(isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h) for h in witness['files'].values()),
            'launch witness closure differs')
    if args.phase != 'cpu':
        check_unit(launch['selected_cpu'])
    if args.phase == 'train':
        require(isinstance(launch['selected_mechanics'],dict) and launch['selected_mechanics'].keys() == set(ARMS),
                'launch paired mechanics required')
        for unit in launch['selected_mechanics'].values():
            check_unit(unit)
        if args.arm == 'candidate':
            check_unit(launch['fresh_control'])


def select_initializer(record, seed):
    require(type(seed) is int and seed in SEEDS, 'initializer seed differs')
    chosen = [q for q in record['qualifications'] if q.get('arm') == 'control' and q.get('seed') == seed and
              q.get('identity',{}).get('device') == 'cpu' and q['identity'].get('scope',{}).get('arm') == 'control']
    require(len(chosen) == 1, 'unique accepted seed-specific CONTROL CPU initializer required')
    file_fact(chosen[0]['checkpoint'])
    require(re.fullmatch('[0-9a-f]{64}',chosen[0]['terminal_state_sha256']), 'initializer typed digest differs')
    return chosen[0]


def method(launch):
    return {k:launch[k] for k in ('execution_sha256','recipe','original_cpu','actual_gradient','witness')}


def cli(root, authority, sha, execution, phase, arm, seed, output):
    return [str(Path(root)/'train_siglip2_connected_mlp.py'),'--execution-sha256',execution,
            '--authority',str(authority),'--authority-sha256',sha,'--phase',phase,
            '--arm',arm,'--seed',str(seed),'--output',str(output)]


def fingerprint(context, value, **kwargs):
    return context['trainer'].fingerprint(context,value,**kwargs)


@contextmanager
def timed(context, name):
    tick = time.perf_counter()
    try:
        yield
    finally:
        delta = time.perf_counter()-tick
        context['phase_seconds'][name] = context['phase_seconds'].get(name,0.)+delta
        print(json.dumps({'event':'CONNECTED_PHASE','phase':name,'seconds':delta}),flush=True)


def mapping_absent(path):
    stat = Path(path).stat()
    for line in Path('/proc/self/maps').read_text().splitlines():
        _,_,_,device,inode,*_ = line.split(maxsplit=5)
        major,minor = (int(v,16) for v in device.split(':'))
        require((major,minor,int(inode)) != (os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino),
                'checkpoint mapping survived independent copy/release')


def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded admission')
    guards = {}
    code = closure(HERE,args.execution_sha256,FILES,guards)
    launch = read_json({'path':str(args.authority),'sha256':args.authority_sha256},guards)
    check_launch(launch,args)
    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and
            not args.output.exists() and not args.output.is_symlink(), 'exclusive canonical output required')
    root = Path(launch['witness']['root'])
    require(root.resolve() == root and root.is_dir() and root != HERE and
            not args.output.is_relative_to(root) and not root.is_relative_to(args.output), 'separate witness closure required')
    for name,digest in launch['witness']['files'].items():
        bound_file(guards,root/name,digest)
    witness = load_authenticated('_connected_admitted_witness',root/'qualify_actual_objective_encoder_gradients.py',
                                 launch['witness']['files']['qualify_actual_objective_encoder_gradients.py'],guards)
    require(witness.FILES == WITNESS_FILES and all(launch['witness']['files'][n] == h for n,h in witness.HELPERS.items()),
            'admitted witness dependencies differ')
    original = launch['original_cpu']
    require(original['authority'] == {'path':str(witness.TRAIN_ROOT/'authority-cpu-v5.json'),
                                      'sha256':witness.CPU_AUTHORITY_SHA} and original['terminal'] == witness.CPU_UNIT,
            'unchanged original CPU-v5 authority/UNIT required')
    require(closure(witness.TRAIN_ROOT,witness.TRAIN_EXECUTION,witness.TRAIN_CODE,guards) == witness.TRAIN_CODE,
            'original exact2 source-v5 differs')
    trainer = load_authenticated('_connected_original_trainer',witness.TRAIN_ROOT/'train_siglip2_identity_diversity.py',
                                 witness.TRAIN_CODE['train_siglip2_identity_diversity.py'],guards)
    historical = SimpleNamespace(execution_sha256=witness.TRAIN_EXECUTION,
        authority=Path(original['authority']['path']),authority_sha256=original['authority']['sha256'],
        output=args.output,phase='cpu',arm='control',seed=SEEDS[0])
    context = trainer.authority(historical)
    record = trainer.admit_terminal(context,original['terminal'],'cpu','control',SEEDS[0])
    # No historical policy/schema or module global is rebound.
    for path,digest in guards.items():
        require(context['guards'].setdefault(path,digest) == digest, 'prospective source guard conflict')
    context.update(trainer=trainer,witness=witness,connected_args=args,connected_launch=launch,
                   connected_code=code,connected_root=HERE,original_cpu_record=record,started=STARTED)
    context['fit_context']['unit_started'] = STARTED
    admit_actual_gradient(context)
    if args.phase != 'cpu':
        admit_terminal(context,launch['selected_cpu'],'cpu','control',SEEDS[0])
    if args.phase == 'train':
        for arm in ARMS:
            admit_terminal(context,launch['selected_mechanics'][arm],'mechanics',arm,args.seed)
        if args.arm == 'candidate':
            control = admit_terminal(context,launch['fresh_control'],'train','control',args.seed)
            context['fresh_control_record'] = control
    context['connected_required_guards'] = dict(context['guards'])
    return context


def admit_actual_gradient(context):
    launch,trainer = context['connected_launch'],context['trainer']
    selected = launch['actual_gradient']
    record = read_json(selected['terminal']['receipt'],context['guards'])
    auth = read_json(selected['authority'],context['guards'])
    require(record['schema'] == 'actual-objective-encoder-gradients-v1' and record['pass'] is True and
            record['engineering_only'] is True and record['training_updates'] == 0 and
            all(record[k] is False for k in ('model_fit_qualified','cost_qualified','quality_read','state_reuse','state_reuse_eligible')) and
            all(record[k] is True for k in ('exit_rehash_pass','all_temporary_references_released')) and
            record['frozen_encoder_tensors'] == 444 and record['unchanged_encoder_tensors'] == 448 and
            record['gradient_parameters'] == list(MLP) and record['resource_policy'] == context['witness'].POLICY and
            selected['terminal']['invocation_id'] == record['invocation_id'] == '1a0ef321f35748ec9b88abf7115d7fba' and
            selected['terminal']['receipt']['sha256'] == 'e7da19ba546f9d5375697d3d0347b3c96f9cfe0ee5183f9341d473114461d896' and
            selected['terminal']['log']['sha256'] == '3dd0843123c576c91869286f1f69320ae2f4fa50c32aef7c714bc51e1f305b1a' and
            record['authority_sha256'] == selected['authority']['sha256'] and
            auth['schema'] == context['witness'].AUTHORITY_SCHEMA and auth['files'] == launch['witness']['files'] and
            auth['both_locks_held'] is True and auth['resource_policy'] == context['witness'].POLICY,
            'original actual-gradient engineering admission differs')
    require([v['view'] for v in record['views']] == list(VIEWS), 'both actual pixel views required')
    for view in record['views']:
        require(all(view[k] is True for k in ('detached_forward_identical','detached_encoder_gradients_absent',
                'detached_A_C_gradients_match','GPU_readout_unchanged','global_RNG_unchanged','temporary_references_released')) and
                view['frozen_encoder_gradients_absent'] == 444, 'actual gradient/lifetime/RNG predicates differ')
        for term in ('ranking','total'):
            require(all(view['gradients'][term][n]['norm'] > 0 and view['gradients'][term][n]['nonzero'] > 0
                        for n in MLP), 'all-four actual ranking/total gradients required')
    for path,digest in record['input_guards'].items():
        bound_file(context['guards'],path,digest)
    require(all(record['input_guards'].get(str(Path(launch['witness']['root'])/n)) == h
                for n,h in launch['witness']['files'].items()), 'admitted witness closure differs')
    # Only field names differ in the discarded witness receipt. The original
    # terminal reader retains every log, cgroup, normal-exit and lock predicate.
    projected = {**record,'invocation':{'invocation_id':record['invocation_id'],'optimize':0},
                 'wall_seconds':record['whole_seconds'],'process_peak_rss_kib':selected['terminal']['native_peak_rss_kib']}
    final = context['fitter'].original_terminal_reader(context['fit_context'])(
        context['legacy']['original'].FlatAdmission(),projected,selected['terminal'],context['witness'].POLICY['seconds'],context['guards'])
    for cgroup in (record['cgroup_before'],record['cgroup_after'],final):
        context['old'].zero_events(cgroup)


def load_initializer(context, qualification):
    """Model-free admission of complete original typed CPU-v5 state."""
    import torch
    trainer = context['trainer']
    fact,digest = qualification['checkpoint'],qualification['terminal_state_sha256']
    path = bound_file(context['guards'],fact['path'],fact['sha256'])
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    try:
        require(disk.keys() == trainer.PAYLOAD_KEYS and type(disk['counter']) is int and disk['counter'] == 0 and
                disk['optimizer']['state'] == {} and
                json.loads(json.dumps(disk['identity'],allow_nan=False)) == qualification['identity'],
                'typed original initializer identity differs')
        ident = copy.deepcopy(disk['identity'])
        context['flags'],context['initial_static_sha256'] = ident['numerical_flags'],ident['static_sha256']
        for key in ('common_initial_sha256','common_statistics_sha256','initial_A_sha256','initial_C_sha256',
                    'mu_train_sha256','mu_train_provenance_sha256'):
            context[key] = ident[key]
        context['initial'] = {k:trainer.clone(context,disk[k]) for k in ('provenance','scope')}
        with path.open('rb') as stream:
            pages = context['legacy']['original'].CheckpointPages(stream)
            trainer.check_payload(context,disk,ident,0)
            require(trainer.fingerprint(context,disk,consumed=pages.consume) == digest, 'complete typed original initializer differs')
        torch.random.set_rng_state(disk['cpu_rng'].clone())
        del pages
    finally:
        del disk
        gc.collect()
        mapping_absent(path)
    state = trainer.restore(context,path,fact['sha256'],digest,ident,0)
    mapping_absent(path)
    trainer.integrity(context,state,ident)
    canonical = trainer.canonical_initial_witness(context,state,ident)
    require(canonical == qualification['canonical_initial'], 'original canonical initializer differs')
    context['initializer_fact'] = {'checkpoint':copy.deepcopy(fact),'payload_sha256':digest,'identity':copy.deepcopy(ident)}
    return state,ident


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, 'nonfinite JSON: ' + v))


def file_fact(value):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'} and
            isinstance(value['path'], str) and Path(value['path']).is_absolute() and
            isinstance(value['sha256'], str) and re.fullmatch('[0-9a-f]{64}', value['sha256']), 'exact FILE required')


def bound_file(guards, path, expected):
    file_fact({'path': str(path), 'sha256': expected})
    path = Path(path)
    require(path.resolve() == path and path.is_file(), 'canonical regular FILE required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        buffer = bytearray(1024**2)
        while count := stream.readinto(buffer):
            digest.update(memoryview(buffer)[:count])
            os.posix_fadvise(stream.fileno(), stream.tell() - count, count, os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'current FILE bytes differ: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting FILE authority')
    return path


def batch_bound_files(guards, items):
    """Fresh per occurrence; publish on the owner only after complete success."""
    items = list(items)
    for path, expected in items:
        file_fact({'path': str(path), 'sha256': expected})
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(bound_file, {}, path, expected) for path, expected in items]
        paths = [future.result() for future in futures]
    staged = dict(guards)
    for path, (_, expected) in zip(paths, items):
        require(staged.setdefault(str(path), expected) == expected, 'conflicting FILE authority')
    guards.update(staged)
    return paths


def read_json(fact, guards):
    file_fact(fact)
    path = bound_file(guards, fact['path'], fact['sha256'])
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == fact['sha256'], 'JSON size/current bytes differ')
    return strict_json(raw)


def closure(root, sha, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure required')
    code = read_json({'path': str(root / 'execution.json'), 'sha256': sha}, guards)
    require(isinstance(code, dict) and code.keys() == set(names) and
            all(Path(n).name == n for n in code), 'exact code closure required')
    for name, digest in code.items():
        bound_file(guards, root / name, digest)
    return code


def load_authenticated(name, path, sha, guards):
    require(name not in sys.modules, 'fresh helper namespace required')
    path = bound_file(guards, path, sha)
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == sha, 'helper changed before execution')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None, 'helper origin required')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), vars(module))
    return module


def check_unit(unit):
    require(isinstance(unit, dict) and unit.keys() == {'receipt', 'log', 'unit', 'invocation_id',
            'service_seconds', 'native_peak_rss_kib', 'both_locks_held'} and unit['both_locks_held'] is True,
            'complete UNIT required')
    for key in ('receipt', 'log'):
        file_fact(unit[key])
    require(isinstance(unit['unit'], str) and re.fullmatch('[A-Za-z0-9_.@-]+', unit['unit']) and
            isinstance(unit['invocation_id'], str) and re.fullmatch('[0-9a-f]{32}', unit['invocation_id']) and
            all(type(unit[k]) in (int, float) and math.isfinite(unit[k]) and unit[k] > 0
                for k in ('service_seconds', 'native_peak_rss_kib')), 'actual UNIT identity/resources required')


def fullfeature_raw_features(features, head, A, means, C, mu_train, arm, primitive, readout):
    """Public FP32 readout over the actual normalized input, with one concat call."""
    import torch
    from torch.nn import functional as F
    parameter_roles(arm)
    primitive._check_tensor(C, (128, 1152), features.device)
    primitive._check_tensor(mu_train, (1152,), features.device, frozen=True)
    require(torch.isfinite(C).all().item() and torch.isfinite(mu_train).all().item(),
            'finite residual/mean required')
    with torch.autocast(features.device.type, enabled=False):
        raw = readout.raw_features(features, head, A, means, 'concat', primitive)
        raw = raw + F.linear(features.detach().float() - mu_train, C)
        require(torch.isfinite(raw).all().item(), 'finite fullfeature raw required')
    return raw


def _processor_cache(processor, guards, *, empty=False):
    """Authenticate the one original self-keyed cache; never adopt prior entries."""
    backend = sys.modules.get('transformers.image_processing_backends')
    siglip = sys.modules.get('transformers.models.siglip.image_processing_siglip')
    require(backend is not None and siglip is not None and
            type(processor) is getattr(siglip, 'SiglipImageProcessor', None) and
            type(processor).__bases__ == (getattr(backend, 'TorchvisionBackend', None),),
            'processor cache owner differs')
    path = Path(backend.__file__)
    require(path.is_absolute() and path.resolve() == path and
            guards.get(str(path)) == BACKEND_SHA, 'processor cache source guard differs')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BACKEND_SHA, 'processor cache source differs')
    name = '_fuse_mean_std_and_rescale_factor'
    wrapper = vars(backend.TorchvisionBackend).get(name)
    require(type(wrapper) is type(lru_cache(maxsize=10)(lambda: None)) and
            inspect.getattr_static(processor, name) is wrapper and
            wrapper.cache_parameters() == {'maxsize': 10, 'typed': False} and
            all(getattr(wrapper, key) == getattr(type(wrapper), key).__get__(wrapper)
                for key in ('cache_info', 'cache_clear')) and wrapper.cache_info().maxsize == 10,
            'processor cache wrapper differs')
    function = getattr(wrapper, '__wrapped__', None)
    # Read-only GC edges bind __wrapped__ to the LRU's actual callable, not a decoy.
    require(type(function) is FunctionType and
            [obj for obj in gc.get_referents(wrapper) if type(obj) is FunctionType] == [function],
            'processor cache wrapper callable differs')
    # Compile only: no re-execution of the backend or replacement of its globals/math.
    module_code = compile(raw, str(path), 'exec', dont_inherit=True)
    class_code = next(c for c in module_code.co_consts if isinstance(c, CodeType) and c.co_name == 'TorchvisionBackend')
    original = next(c for c in class_code.co_consts if isinstance(c, CodeType) and c.co_name == name)
    require(function.__code__ == original and function.__globals__ is vars(backend) and
            function.__defaults__ == (None,) * 6 and function.__kwdefaults__ is None and
            function.__closure__ is None, 'processor cache live code differs')
    require(not empty or wrapper.cache_info().currsize == 0, 'processor cache must initially be empty')
    return wrapper


def apply_overlay(model, encoder):
    """Copy absolute parameters into the exact existing objects; never rebind."""
    import torch
    require(isinstance(encoder,dict) and encoder.keys() == set(MLP), 'exact four overlay names required')
    params = dict(model.named_parameters())
    require(len(params) == 448 and params.keys() == model.state_dict().keys() and set(MLP) <= params.keys(),
            'complete original 448 parameter inventory required')
    for name,shape in zip(MLP,MLP_SHAPES,strict=True):
        value,parameter = encoder[name],params[name]
        require(isinstance(value,torch.Tensor) and list(value.shape) == shape == list(parameter.shape) and
                value.dtype == parameter.dtype == torch.float32 and not value.requires_grad and value.grad_fn is None and
                torch.isfinite(value).all().item(), 'absolute finite FP32 overlay shape/dtype/role differs: '+name)
    with torch.no_grad():
        for name in MLP:
            params[name].copy_(encoder[name])


def model_structure(model, source, packages):
    """Original runtime module facts, separate from updated bytes and roles."""
    import torch
    modules = []
    for name,module in model.named_modules():
        require(not module.training and not module._forward_hooks and not module._forward_pre_hooks and
                not module._backward_hooks and not getattr(module,'gradient_checkpointing',False), 'encoder mode/hooks differ')
        row = {'name':name,**source.module_origin(type(module),packages),'training':module.training}
        row['attributes'] = {key:value for key,value in vars(module).items() if not key.startswith('_') and
            (value is None or isinstance(value,(bool,int,float,str)) or isinstance(value,(tuple,list)) and
             all(isinstance(item,(bool,int,float,str)) for item in value))}
        if hasattr(module,'config'):
            row['attn_implementation'] = module.config._attn_implementation
        if isinstance(module,torch.nn.LayerNorm):
            require(module.eps == model.config.layer_norm_eps, 'LayerNorm epsilon differs')
        modules.append(row)
    require(model.config._attn_implementation == 'sdpa', 'original SDPA implementation required')
    return json.loads(json.dumps({'config':model.config.to_dict(),'attn_implementation':model.config._attn_implementation,
                                  'modules':modules},allow_nan=False))


def construct_encoder(source, original, construct_context, config, buffers, processor_config, base,
                      overlay=None, source_runtime=None):
    """One genuine CPU factory/strict base load, then exact overlay and transfer."""
    import torch
    from transformers import AutoImageProcessor
    guards = construct_context['guards']
    fact = base['checkpoint']
    path = bound_file(guards,fact['path'],fact['sha256'])
    model = source.construct(config,construct_context).eval()
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    try:
        with path.open('rb') as stream:
            pages = original.CheckpointPages(stream)
            require(disk.keys() == {'vision','buffers','config','runtime','cpu_rng'} and disk['config'] == config and
                    original.fingerprint(disk['buffers']) == original.fingerprint(buffers) and
                    len(disk['vision']) == 448, 'complete authenticated original base differs')
            digest = original.fingerprint(disk['vision'],consumed=pages.consume)
            require(base.get('sha256',digest) == digest, 'original typed base vision identity differs')
            original.load_vision(model,disk['vision'],pages)
            require(dict(model.named_buffers()).keys() == buffers.keys() == {'embeddings.position_ids'},
                    'complete nonpersistent buffer inventory differs')
            with torch.no_grad():
                for name,value in model.named_buffers():
                    require(value.shape == disk['buffers'][name].shape and value.dtype == disk['buffers'][name].dtype,
                            'buffer shape/dtype differs')
                    value.copy_(disk['buffers'][name])
                    pages.consume(disk['buffers'][name])
            del value
            require(original.fingerprint(model.state_dict()) == digest, 'genuine strict copied original448 differs')
        processor = AutoImageProcessor.from_pretrained(processor_config,local_files_only=True,backend='torchvision')
        cache = _processor_cache(processor,guards,empty=True)
        if source_runtime is not None:
            roles = source_runtime['roles']
            for row,(name,p) in zip(roles,model.named_parameters(),strict=True):
                require(row['name'] == name, 'original role inventory differs')
                p.requires_grad_(row['role'] == 'trainable')
            del p
            require(source.model_facts(model,processor,roles,construct_context['packages']) == source_runtime == disk['runtime'],
                    'original source proof/config/runtime/buffers/processor differs')
        require('position_ids' in model.embeddings._non_persistent_buffers_set and
                torch.equal(dict(model.named_buffers())['embeddings.position_ids'],torch.arange(256).expand(1,-1)),
                'original nonpersistent position buffer differs')
        if overlay is not None:
            apply_overlay(model,overlay)
        structure = model_structure(model,source,construct_context['packages'])
        require(source_runtime is None or structure == {k:source_runtime[k] for k in ('config','attn_implementation','modules')},
                'adapted original runtime structure differs')
        base = {'checkpoint':copy.deepcopy(fact),'sha256':digest}
        del pages
    finally:
        del disk
        gc.collect()
        mapping_absent(path)
    return model,processor,cache,base,structure


def bind_optimizer(state):
    import torch
    groups = [{'params':[state['A'],state['C']],**ADAM}]
    if state['arm'] == 'candidate':
        params = dict(state['model'].named_parameters())
        groups.append({'params':[params[n] for n in MLP],**ADAM,'lr':1e-5})
    optimizer = torch.optim.AdamW(groups,**ADAM)
    expected = dict(ADAM)
    if 'decoupled_weight_decay' in optimizer.defaults:
        expected['decoupled_weight_decay'] = True
    require(optimizer.defaults == expected, 'original AdamW defaults differ')
    for i,group in enumerate(optimizer.param_groups):
        require({k:v for k,v in group.items() if k != 'params'} == {**expected,'lr':1e-5 if i else 1e-4},
                'fixed ordered group defaults differ')
    state['optimizer_object'] = optimizer
    state['scaler_object'] = torch.amp.GradScaler(state['device'],init_scale=128.,enabled=state['device'] == 'cuda')


def encoder_facts(state, original, source, packages, *, serving=False):
    import torch
    model = state['model']
    params = dict(model.named_parameters())
    ident = state['encoder_identity']
    require(len(params) == 448 and params.keys() == model.state_dict().keys() == ident['inventory'].keys() and
            sum(n not in MLP for n in params) == 444, 'exact four/frozen444 encoder inventory differs')
    for name,p in params.items():
        require(list(p.shape) == ident['inventory'][name] and p.dtype == torch.float32 and p.device.type == state['device'] and
                p.grad is None and p.is_leaf and p.grad_fn is None and
                p.requires_grad is (not serving and state['arm'] == 'candidate' and name in MLP),
                'encoder parameter shape/dtype/device/role/gradient differs: '+name)
    nonpersistent = {n:sorted(m._non_persistent_buffers_set) for n,m in model.named_modules() if m._non_persistent_buffers_set}
    require(nonpersistent == ident['nonpersistent'] == {'embeddings':['position_ids']}, 'nonpersistent registration differs')
    buffers = dict(model.named_buffers())
    require(buffers.keys() == {'embeddings.position_ids'} and
            original.fingerprint(buffers) == ident['buffers_sha256'], 'complete current buffers differ')
    require(model_structure(model,source,packages) == ident['runtime'], 'current config/runtime/module origin differs')
    processor = state['processor_object']
    require(json.loads(processor.to_json_string()) == state['processor']['config'] and
            processor.backend == state['processor']['backend'] and
            source.module_origin(type(processor),packages) == state['processor']['origin'], 'current processor origin/config differs')
    require(_processor_cache(processor,state['guards']) is state['processor_cache'], 'authenticated processor cache changed')
    frozen = original.fingerprint({n:p for n,p in params.items() if n not in MLP})
    require(frozen == ident['frozen_sha256'], 'current frozen444 bytes differ')
    return {'vision_sha256':original.fingerprint(model.state_dict()),
            'encoder':{n:original.fingerprint(params[n]) for n in MLP}}


def fresh(context, arm, seed, device):
    import torch
    trainer,legacy = context['trainer'],context['legacy']
    trainer.require_no_training(context)
    qualification = select_initializer(context['original_cpu_record'],seed)
    state,initial_ident = load_initializer(context,qualification)
    context.get('A_owners',{}).pop(id(state),None)
    context.get('C_owners',{}).pop(id(state),None)
    for key in ('teachers','target','means','classifier','mu_train'):
        state[key] = trainer.clone(context,state[key],device)
    state['head_object'].to(device)
    state['A'] = torch.nn.Parameter(state['A'].detach().to(device,copy=True))
    state['C'] = torch.nn.Parameter(state['C'].detach().to(device,copy=True))
    state.update(arm=arm,device=device,guards=context['guards'])
    context['live_training'],context['live_residual'] = weakref.ref(state['A']),weakref.ref(state['C'])
    base = {'checkpoint':copy.deepcopy(state['provenance']['encoder']['checkpoint'])}
    config_path = legacy['prior']['entry']['input']['preprocessor']['path']
    factory = {**legacy['prior'],'guards':context['guards']}
    model,processor,cache,base,structure = construct_encoder(legacy['source_driver'],legacy['original'],factory,
        state['config'],state['buffers'],config_path,base,source_runtime=state['provenance']['encoder']['source_proof']['runtime'])
    state['base_vision'] = base
    state['encoder_identity'] = {'inventory':{n:list(p.shape) for n,p in model.named_parameters()},
        'initial_four_sha256':{n:fingerprint(context,dict(model.named_parameters())[n]) for n in MLP},
        'frozen_sha256':fingerprint(context,{n:p for n,p in model.named_parameters() if n not in MLP}),
        'buffers_sha256':fingerprint(context,dict(model.named_buffers())),
        'nonpersistent':{'embeddings':['position_ids']},'runtime':structure}
    for name,p in model.named_parameters():
        p.requires_grad_(arm == 'candidate' and name in MLP)
    del p
    model.to(device).eval()
    state.update(model=model,processor_object=processor,processor_cache=cache)
    context['live_model'] = weakref.ref(model)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
    bind_optimizer(state)
    trainer.own_A(context,state,admit=True)
    state['current_encoder'] = encoder_facts(state,legacy['original'],legacy['source_driver'],legacy['selected']['packages'])
    validator = copy.deepcopy(initial_ident)
    validator.update(device=device,initial_scaler=state['scaler_object'].state_dict(),
        initial_cpu_rng_sha256=fingerprint(context,torch.random.get_rng_state()),
        initial_cuda_rng_sha256=fingerprint(context,torch.cuda.get_rng_state_all()) if device == 'cuda' else None)
    state['identity'] = {'method':method(context['connected_launch']),'source':copy.deepcopy(context['source']),
        'arm':arm,'seed':seed,'device':device,'scope':copy.deepcopy(initial_ident['scope']),
        'initializer':copy.deepcopy(context['initializer_fact']),'validator_identity':validator,
        'encoder_identity':copy.deepcopy(state['encoder_identity']),'base_vision':copy.deepcopy(base),
        'parameter_names':parameter_roles(arm)[0],'parameter_shapes':parameter_roles(arm)[1],
        'optimizer_defaults':copy.deepcopy(state['optimizer_object'].defaults),
        'optimizer_groups':[{k:v for k,v in g.items() if k != 'params'} for g in state['optimizer_object'].param_groups],
        'initial_scaler':copy.deepcopy(state['scaler_object'].state_dict()),'numerical_flags':copy.deepcopy(context['flags'])}
    integrity(context,state,state['identity'])
    return state


def payload(context, state, identity):
    trainer = context['trainer']
    saved = trainer.payload(context,state,identity)
    saved.update(schema=SCHEMA,base_vision=copy.deepcopy(state['base_vision']),
        encoder={n:dict(state['model'].named_parameters())[n].detach() for n in MLP},
        vision_sha256=fingerprint(context,state['model'].state_dict()))
    return saved


def legacy_payload(context, saved, identity):
    trainer = context['trainer']
    projected = {k:saved[k] for k in trainer.PAYLOAD_KEYS}
    projected.update(schema=trainer.SCHEMA,identity=identity['validator_identity'],
        optimizer={'state':{i:v for i,v in saved['optimizer']['state'].items() if i in (0,1)},
                   'param_groups':[saved['optimizer']['param_groups'][0]]})
    return projected


def check_optimizer(saved, identity, step):
    import torch
    names,shapes,_ = parameter_roles(identity['arm'])
    optimizer = saved['optimizer']
    defaults = {**ADAM,**({'decoupled_weight_decay':True} if 'decoupled_weight_decay' in identity['optimizer_defaults'] else {})}
    expected = [defaults]+([{**defaults,'lr':1e-5}] if identity['arm'] == 'candidate' else [])
    require(identity['optimizer_defaults'] == defaults and identity['optimizer_groups'] == expected,
            'frozen AdamW defaults/rates differ')
    require(all(type(v) is type(defaults[k]) for k,v in identity['optimizer_defaults'].items()) and
            all(type(v) is type(expected[i][k]) for i,g in enumerate(identity['optimizer_groups']) for k,v in g.items()),
            'typed original AdamW defaults/groups differ')
    groups = [list(range(2))]+([list(range(2,6))] if identity['arm'] == 'candidate' else [])
    require(optimizer.keys() == {'state','param_groups'} and len(optimizer['param_groups']) == len(groups) and
            identity['parameter_names'] == names and identity['parameter_shapes'] == shapes and
            optimizer['state'].keys() == (set(range(len(names))) if step else set()), 'ordered optimizer ownership differs')
    for g,ids,expected in zip(optimizer['param_groups'],groups,identity['optimizer_groups'],strict=True):
        require(type(g['params']) is list and all(type(p) is int for p in g['params']) and
                g['params'] == ids and {k:v for k,v in g.items() if k != 'params'} == expected and
                all(type(v) is type(expected[k]) for k,v in g.items() if k != 'params'),
                'ordered optimizer groups differ')
    require(all(type(i) is int for i in optimizer['state']), 'typed named optimizer indices differ')
    for i,member in optimizer['state'].items():
        require(member.keys() == {'step','exp_avg','exp_avg_sq'} and member['step'].shape == () and
                member['step'].dtype == torch.float32 and member['step'].device.type == 'cpu' and
                not member['step'].requires_grad and float(member['step']) == step, 'independent Adam CPU step differs')
        for key in ('exp_avg','exp_avg_sq'):
            value = member[key]
            require(list(value.shape) == shapes[i] and value.dtype == torch.float32 and not value.requires_grad and
                    value.grad_fn is None and torch.isfinite(value).all().item(), 'complete finite named moment differs')


def check_payload(context, saved, identity, step):
    import torch
    trainer = context['trainer']
    require(saved.keys() == PAYLOAD_KEYS and identity.keys() == IDENTITY_KEYS and saved['schema'] == SCHEMA and saved['identity'] == identity and
            identity['method'] == method(context['connected_launch']) and
            identity['source'] == saved['source'] == context['source'] and identity['arm'] in ARMS and
            identity['seed'] in SEEDS and identity['device'] in ('cpu','cuda') and
            identity['scope']['arm'] == 'control' and identity['scope']['manifest_sha256'] == SCOPE_SHA256 and
            identity['scope']['arm_sha256'] == CONTROL_SHA256 and saved['base_vision'] == identity['base_vision'] and
            type(saved['counter']) is int and saved['counter'] == step, 'complete connected payload identity differs')
    qualification = select_initializer(context['original_cpu_record'],identity['seed'])
    require(identity['initializer']['checkpoint'] == qualification['checkpoint'] and
            identity['initializer']['payload_sha256'] == qualification['terminal_state_sha256'] and
            json.loads(json.dumps(identity['initializer']['identity'],allow_nan=False)) == qualification['identity'],
            'seed-specific original typed initializer identity differs')
    original,validator = identity['initializer']['identity'],identity['validator_identity']
    allowed = {'device','initial_scaler','initial_cuda_rng_sha256'}
    require(validator.keys() == original.keys() and all(validator[k] == original[k] for k in original.keys()-allowed) and
            validator['device'] == identity['device'] and validator['initial_scaler'] == identity['initial_scaler'] and
            identity['optimizer_defaults'] == original['optimizer_defaults'] and
            validator['optimizer_groups'] == original['optimizer_groups'], 'preserved typed original validator identity differs')
    trainer.check_payload(context,legacy_payload(context,saved,identity),identity['validator_identity'],step)
    check_optimizer(saved,identity,step)
    require(saved['encoder'].keys() == set(MLP) and re.fullmatch('[0-9a-f]{64}',saved['vision_sha256']),
            'exact four/full448 payload required')
    for name,shape in zip(MLP,MLP_SHAPES,strict=True):
        value = saved['encoder'][name]
        require(isinstance(value,torch.Tensor) and list(value.shape) == shape and value.dtype == torch.float32 and
                not value.requires_grad and value.grad_fn is None and torch.isfinite(value).all().item(), 'four typed payload differs')
        initial = identity['encoder_identity']['initial_four_sha256'][name]
        require((fingerprint(context,value) == initial) if step == 0 or identity['arm'] == 'control' else
                (fingerprint(context,value) != initial), 'original/updated absolute encoder substitution differs')


def resource_check(context):
    import torch
    p = policy(context['connected_args'].phase)
    cgroup = context['legacy']['source_driver'].cgroup_memory()
    context['old'].zero_events(cgroup)
    unit = Path(cgroup['path']).name.removesuffix('.service')
    context['legacy']['selected']['genuine']['reference'].admit_cgroup(cgroup,unit)
    require(time.perf_counter()-context['started'] < p['seconds'] and
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= p['host_bytes']//1024 and
            (not torch.cuda.is_initialized() or torch.cuda.max_memory_allocated() < p['cuda_allocated_bytes_exclusive']),
            'whole new phase resource/deadline exceeded')
    return cgroup


def integrity(context, state, identity):
    trainer,legacy = context['trainer'],context['legacy']
    trainer.helper_guard(context)
    trainer.own_A(context,state)
    require(state['identity'] == identity and state['encoder_identity'] == identity['encoder_identity'] and
            state['ranking_bank'] == trainer.ranking_bank(state['target'].tolist(),state['original_rows'].tolist()) and
            state['target_list'] == state['target'].tolist() and state['row_list'] == state['original_rows'].tolist(),
            'live complete identity/ranking bank differs')
    require(encoder_facts(state,legacy['original'],legacy['source_driver'],legacy['selected']['packages']) == state['current_encoder'],
            'authorized current full448/overlay bytes differ')
    optimizer = state['optimizer_object']
    params = [state['A'],state['C']]+([dict(state['model'].named_parameters())[n] for n in MLP] if state['arm'] == 'candidate' else [])
    owned = [p for g in optimizer.param_groups for p in g['params']]
    require(len(owned) == len(params) and all(a is b for a,b in zip(owned,params,strict=True)) and
            all(p.grad is None and p.requires_grad and p.is_leaf and p.device.type == state['device'] for p in params) and
            all(p.grad is None and not p.requires_grad and p.device.type == state['device'] for p in state['head_object'].parameters()) and
            all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in state['head_object'].modules()) and optimizer.defaults == identity['optimizer_defaults'],
            'active/frozen runtime ownership/roles/hooks differ')
    if state['counter']:
        require(all(optimizer.state[p][k].device == p.device for p in params for k in ('exp_avg','exp_avg_sq')),
                'moments must follow active parameter devices')
    require(legacy['source_driver'].numerical_flags() == context['flags'], 'original numerical flags changed')
    check_payload(context,payload(context,state,identity),identity,state['counter'])
    resource_check(context)


def release(context, state):
    """End only this owned state; prove tensors/model/processor/optimizer died."""
    trainer = context['trainer']
    refs = trainer.tensor_weakrefs(context,{k:state[k] for k in STATIC_KEYS+('A','C')})
    refs += trainer.tensor_weakrefs(context,state['optimizer_object'].state_dict())
    refs += [weakref.ref(v) for v in (state['model'],state['processor_object'],state['head_object'],
                                    state['optimizer_object'],state['scaler_object'])]
    refs += [weakref.ref(p) for p in (*state['model'].parameters(),*state['model'].buffers(),
                                    *state['head_object'].parameters(),*state['head_object'].buffers())]
    cache = state['processor_cache']
    require(_processor_cache(state['processor_object'],context['guards']) is cache, 'processor teardown authority differs')
    cache.cache_clear()
    require(cache.cache_info().currsize == 0, 'processor cache teardown failed')
    trainer.release(context,state)
    gc.collect()
    require(all(ref() is None for ref in refs), 'training tensor/model/graph/processor lifetime survived release')


def save(context, state, identity, path):
    import torch
    with timed(context,'save'):
        integrity(context,state,identity)
        saved = payload(context,state,identity)
        digest = fingerprint(context,saved)
        with context['legacy']['extract'].exclusive(path) as stream:
            writer = context['legacy']['original'].CheckpointWriter(stream)
            torch.save(saved,writer)
            writer.flush()
        fact = context['legacy']['extract'].sha(path)
        bound_file(context['guards'],path,fact)
    return fact,digest


def restore(context, path, file_sha256, payload_sha256, identity, step):
    """Release-first; genuine CPU construction, final transfer, independent moments/RNG."""
    import torch
    trainer,legacy = context['trainer'],context['legacy']
    trainer.require_no_training(context)
    path = bound_file(context['guards'],path,file_sha256)
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    try:
        with path.open('rb') as stream:
            pages = legacy['original'].CheckpointPages(stream)
            check_payload(context,disk,identity,step)
            require(fingerprint(context,disk,consumed=pages.consume) == payload_sha256, 'complete typed checkpoint differs')
            state = trainer.fresh(context,'control',identity['seed'],identity['device'],initial=disk)
            state.update(arm=identity['arm'],identity=copy.deepcopy(identity),guards=context['guards'],
                         base_vision=copy.deepcopy(identity['base_vision']),encoder_identity=copy.deepcopy(identity['encoder_identity']))
            factory = {**legacy['prior'],'guards':context['guards']}
            model,processor,cache,base,structure = construct_encoder(legacy['source_driver'],legacy['original'],factory,
                disk['config'],disk['buffers'],legacy['prior']['entry']['input']['preprocessor']['path'],
                identity['base_vision'],disk['encoder'],disk['provenance']['encoder']['source_proof']['runtime'])
            require(structure == identity['encoder_identity']['runtime'] and base == identity['base_vision'], 'restored base/runtime differs')
            for name,p in model.named_parameters():
                p.requires_grad_(identity['arm'] == 'candidate' and name in MLP)
            del p
            model.to(identity['device']).eval()
            state.update(model=model,processor_object=processor,processor_cache=cache)
            context['live_model'] = weakref.ref(model)
            bind_optimizer(state)
            optimizer = {'state':{},'param_groups':copy.deepcopy(disk['optimizer']['param_groups'])}
            for index,member in disk['optimizer']['state'].items():
                optimizer['state'][index] = {k:pages.copy(v,'cpu' if k == 'step' else identity['device']) for k,v in member.items()}
            member = None
            state['optimizer_object'].load_state_dict(optimizer)
            state['scaler_object'].load_state_dict(disk['scaler'])
            state['counter'] = step
            state['current_encoder'] = encoder_facts(state,legacy['original'],legacy['source_driver'],legacy['selected']['packages'])
            require(state['current_encoder']['vision_sha256'] == disk['vision_sha256'], 'complete updated448 restoration differs')
            trainer.own_A(context,state,admit=True)
            torch.random.set_rng_state(pages.copy(disk['cpu_rng'],'cpu'))
            if identity['device'] == 'cuda':
                torch.cuda.set_rng_state_all([pages.copy(v,'cpu') for v in disk['cuda_rng']])
            del optimizer,pages,model,processor
    finally:
        del disk
        gc.collect()
        mapping_absent(path)
    integrity(context,state,identity)
    require(fingerprint(context,payload(context,state,identity)) == payload_sha256,
            'complete independent static/optimizer/scaler/counter/RNG/updated448 payload differs')
    return state


def update(context, state, identity, step):
    import torch
    from torch.nn import functional as F
    trainer,connected = context['trainer'],context['connected']
    owned_function,owned_code = context['connected_function']
    require(connected.raw_features is owned_function and owned_function.__code__ is owned_code,
            'admitted connected helper live function changed')
    bound_file({},connected.__file__,context['guards'][connected.__file__])
    require(type(step) is int and state['counter'] == step-1 and 1 <= step <= 128, 'fixed complete update required')
    integrity(context,state,identity)
    if state['device'] == 'cuda':
        torch.cuda.synchronize()
    tick = time.perf_counter()
    batch = state['schedules'][str(state['seed'])][step-1].tolist()
    full = trainer.ranking_membership(state['ranking_bank'],batch)
    K = full['valid']
    optimizer,scaler = state['optimizer_object'],state['scaler_object']
    members = [p for g in optimizer.param_groups for p in g['params']]
    names = identity['parameter_names']
    before = {n:fingerprint(context,p) for n,p in zip(names,members,strict=True)}
    view_gradients,membership = [],[]
    mse_sum = rank_sum = 0.
    optimizer.zero_grad(set_to_none=True)
    ranking_total = [torch.zeros_like(p) for p in members] if step == 1 else None
    for view in VIEWS:
        ranking = [torch.zeros_like(p) for p in members] if step == 1 else None
        for offset in range(0,64,16):
            anchors = batch[offset:offset+16]
            cpu_pixels,facts = context['witness'].pixels_for(trainer,context,state,state['processor_object'],anchors,view)
            with torch.autocast(state['device'],enabled=False):
                pixels = cpu_pixels.to(state['device'])
                features = F.normalize(state['model'](pixel_values=pixels).pooler_output.float(),dim=1)
                if state['arm'] == 'candidate':
                    require(features.requires_grad, 'live encoder features disconnected')
                    raw = connected.raw_features(features,state['head_object'],state['A'],state['means'],state['C'],
                        state['mu_train'],context['legacy']['quadratic'],trainer.helper_guard(context))
                else:
                    raw = trainer.raw_features(context,state,features)
                require(raw.requires_grad, 'live objective graph detached')
                mse,rank,selected = trainer.loss_terms(context,state,raw,anchors,K)
                if step == 1:
                    gradients = torch.autograd.grad(rank,members,retain_graph=True,allow_unused=True)
                    for accumulator,gradient in zip(ranking,gradients,strict=True):
                        require(gradient is not None and gradient.dtype == torch.float32 and torch.isfinite(gradient).all().item(),
                                'actual SmoothAP member gradient absent/nonfinite')
                        accumulator.add_(gradient.detach())
                    del accumulator,gradient,gradients
                loss = mse+rank
                scaler.scale(loss).backward()
            mse_sum += float(mse.detach())
            rank_sum += float(rank.detach())
            membership.append({'view':view,'batch':anchors,**selected})
            del pixels,cpu_pixels,features,raw,mse,rank,loss,facts
        if step == 1:
            norms = {n:float(g.double().norm()) for n,g in zip(names,ranking,strict=True)}
            require(all(math.isfinite(v) and v > 0 for v in norms.values()), 'both-view actual SmoothAP gradients must be nonzero')
            view_gradients.append({'view':view,'ranking_gradient_norms':norms})
            for total,part in zip(ranking_total,ranking,strict=True):
                total.add_(part)
            del total,part
            del ranking
    scaler.unscale_(optimizer)
    gradient_norms = {n:float(p.grad.double().norm()) if p.grad is not None else 0. for n,p in zip(names,members,strict=True)}
    require(all(p.grad is not None and p.grad.dtype == torch.float32 and torch.isfinite(p.grad).all().item()
                for p in members) and all(math.isfinite(v) and v > 0 for v in gradient_norms.values()),
            'finite nonzero total gradients for all active members required')
    norm = torch.nn.utils.clip_grad_norm_(members,1.,error_if_nonfinite=True)
    scale = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    require(scaler.get_scale() == scale == (128. if state['device'] == 'cuda' else 1.) and
            all(float(optimizer.state[p]['step']) == step for p in members), 'skipped/rescaled optimizer step forbidden')
    state['counter'] = step
    after = {n:fingerprint(context,p) for n,p in zip(names,members,strict=True)}
    require(all(before[n] != after[n] for n in names), 'genuine all-active parameter update required')
    trainer.own_A(context,state,advanced=True)
    optimizer.zero_grad(set_to_none=True)
    state['current_encoder'] = encoder_facts(state,context['legacy']['original'],context['legacy']['source_driver'],
                                            context['legacy']['selected']['packages'])
    integrity(context,state,identity)
    digest = fingerprint(context,payload(context,state,identity))
    if state['device'] == 'cuda':
        torch.cuda.synchronize()
    seconds = time.perf_counter()-tick
    ranking_gradient_norm = float(ranking_total[0].double().norm()) if step == 1 else None
    ranking_C_gradient_norm = float(ranking_total[1].double().norm()) if step == 1 else None
    del ranking_total
    row = {'step':step,'batch':batch,'full_membership_sha256':trainer.json_sha256(full),'full_valid':K,
        'membership':membership,'arm':state['arm'],'mse':mse_sum,'rank':rank_sum,'loss':mse_sum+rank_sum,
        'gradient_norms':gradient_norms,'view_gradients':view_gradients,'preclip_norm':float(norm),'scale':scale,
        'ranking_gradient_norm':ranking_gradient_norm,'ranking_C_gradient_norm':ranking_C_gradient_norm,
        'before_sha256':before,'after_sha256':after,'vision_sha256':state['current_encoder']['vision_sha256'],
        'state_sha256':digest,'core_seconds':seconds,'seconds':seconds}
    print(json.dumps({'event':'CONNECTED_UPDATE',**row},sort_keys=True,allow_nan=False),flush=True)
    return row


def diagnostic(row):
    return {k:v for k,v in row.items() if k not in ('seconds','core_seconds')}


def inference_members(context, state):
    trainer = context['trainer']
    keys = ('config','buffers','processor','means','mu_train','mu_train_provenance','scope','common_statistics')
    members = {k:trainer.clone(context,state[k]) for k in keys}
    members.update(schema=INFERENCE_SCHEMA,source=copy.deepcopy(context['source']),arm=state['arm'],
        head=trainer.clone(context,dict(state['head_object'].state_dict())),
        A=trainer.clone(context,state['A'].detach()),C=trainer.clone(context,state['C'].detach()),
        numerical_flags=copy.deepcopy(context['flags']),base_vision=copy.deepcopy(state['base_vision']),
        encoder={n:trainer.clone(context,dict(state['model'].named_parameters())[n].detach()) for n in MLP},
        encoder_identity=copy.deepcopy(state['encoder_identity']),vision_sha256=state['current_encoder']['vision_sha256'])
    members['fixed_sha256'] = fingerprint(context,members)
    require(members.keys() == INFERENCE_KEYS, 'complete portable inference state required')
    return members


def serving_environment(context):
    trainer,legacy = context['trainer'],context['legacy']
    api = context['nearest'].native_source_api(context)
    trainer.audit_origin_diagnostics(context,api,admission=legacy['original'].FlatAdmission(),
                                     require_exact=context['connected_args'].phase != 'cpu')
    origins = legacy['origins']
    for path,digest in origins['files'].items():
        bound_file(context['guards'],path,digest)
    roots = [Path(v['root']) for v in origins['packages'].values()]
    native = set(origins['native_files']) | {p for p in context['guards'] if Path(p).name in context['nearest'].NATIVE_MEMBERS}
    files = {p:h for p,h in context['guards'].items() if any(Path(p).is_relative_to(r) for r in roots) or p in native}
    constructor = legacy['prior']['sources']['native_environment']['vision_constructor']['path']
    require(constructor in files, 'owned serving constructor missing')
    return {'packages':origins['packages'],'files':files,'native_files':{p:files[p] for p in native},
            'vision_constructor':constructor}


def export_bundle(context, members, directory):
    """Own base bytes/absolute overlay and serving code; no optimizer/TRAIN dependency."""
    import torch
    trainer,legacy = context['trainer'],context['legacy']
    trainer.require_no_training(context)
    require(directory.is_absolute() and directory.parent.resolve() == directory.parent and
            not directory.exists() and not directory.is_symlink(), 'exclusive portable bundle required')
    directory.mkdir()
    source = bound_file(context['guards'],members['base_vision']['checkpoint']['path'],members['base_vision']['checkpoint']['sha256'])
    with source.open('rb') as reader,legacy['extract'].exclusive(directory/'vision.pt') as stream:
        writer = legacy['original'].CheckpointWriter(stream)
        buffer = bytearray(1024**2)
        while count := reader.readinto(buffer):
            writer.write(memoryview(buffer)[:count])
            os.posix_fadvise(reader.fileno(),reader.tell()-count,count,os.POSIX_FADV_DONTNEED)
        writer.flush()
    require((directory/'vision.pt').stat().st_nlink == 1, 'bundle must exclusively own original base')
    bound_file(context['guards'],directory/'vision.pt',members['base_vision']['checkpoint']['sha256'])
    with legacy['extract'].exclusive(directory/'endpoint.pt') as stream:
        writer = legacy['original'].CheckpointWriter(stream)
        torch.save(members,writer)
        writer.flush()
    trainer.write_json(context,directory/'processor.json',members['processor']['config'])
    modules = (legacy['source_driver'],legacy['extract'],legacy['selected']['cached'],legacy['original'],
               trainer.helper_guard(context),legacy['quadratic'],legacy['packing'])
    paths = {Path(m.__file__).name:Path(m.__file__) for m in modules}
    require(paths.keys() == SERVING_FILES | {'joint_relational_compaction.py'}, 'exact admitted serving closure required')
    paths.update({n:context['connected_root']/n for n in FILES})
    code = {}
    for name,path in paths.items():
        digest = context['guards'][str(path)]
        bound_file({},path,digest)
        shutil.copyfile(path,directory/name)
        bound_file(context['guards'],directory/name,digest)
        code[name] = digest
    files = {n:legacy['extract'].sha(directory/n) for n in ('vision.pt','endpoint.pt','processor.json')}
    for name,digest in files.items():
        bound_file(context['guards'],directory/name,digest)
    manifest = {'schema':BUNDLE_SCHEMA,'code':code,'files':files,'endpoint_state_sha256':fingerprint(context,members),
                'environment':serving_environment(context),'encoder_identity':copy.deepcopy(members['encoder_identity']),
                'base_vision_sha256':members['base_vision']['sha256'],'vision_sha256':members['vision_sha256'],
                'scope':{'arm':'control','manifest_sha256':SCOPE_SHA256,'arm_sha256':CONTROL_SHA256}}
    trainer.write_json(context,directory/'bundle.json',manifest)
    digest = legacy['extract'].sha(directory/'bundle.json')
    bound_file(context['guards'],directory/'bundle.json',digest)
    fd = os.open(directory,os.O_RDONLY|os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    return {'path':str(directory/'bundle.json'),'sha256':digest}


def admit_bundle(directory, digest):
    directory,guards = Path(directory),{}
    require(directory.is_absolute() and directory.resolve() == directory and directory.is_dir(), 'canonical owned bundle required')
    manifest = read_json({'path':str(directory/'bundle.json'),'sha256':digest},guards)
    require(manifest.keys() == {'schema','code','files','endpoint_state_sha256','environment','encoder_identity',
                               'base_vision_sha256','vision_sha256','scope'} and manifest['schema'] == BUNDLE_SCHEMA and
            manifest['code'].keys() == FILES | SERVING_FILES | {'joint_relational_compaction.py'} and
            manifest['files'].keys() == {'vision.pt','endpoint.pt','processor.json'} and
            manifest['scope'] == {'arm':'control','manifest_sha256':SCOPE_SHA256,'arm_sha256':CONTROL_SHA256} and
            all(re.fullmatch('[0-9a-f]{64}',manifest[k]) for k in ('endpoint_state_sha256','base_vision_sha256','vision_sha256')),
            'exact updated owned bundle schema/closure differs')
    for name,digest in {**manifest['code'],**manifest['files']}.items():
        path = bound_file(guards,directory/name,digest)
        require(path.stat().st_nlink == 1 and not path.is_symlink(), 'regular single-link owned bundle required')
    require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == manifest['code']['train_siglip2_connected_mlp.py'],
            'current copied public loader source differs')
    env = manifest['environment']
    require(env.keys() == {'packages','files','native_files','vision_constructor'} and env['vision_constructor'] in env['files'] and
            env['packages'].keys() == NATIVE-{'sfora'} and set(env['native_files']) <= set(env['files']),
            'admitted serving environment differs')
    roots = [Path(v['root']) for v in env['packages'].values()]
    require(all(any(Path(p).is_relative_to(r) for r in roots) or env['native_files'].get(p) == h for p,h in env['files'].items()),
            'serving environment contains TRAIN data')
    batch_bound_files(guards,env['files'].items())
    return manifest,guards


def inference_readout_tree(endpoint):
    return {'arm':endpoint['arm'],'scope':endpoint['scope'],'common_statistics':endpoint['common_statistics'],
            'A':endpoint['A'].detach(),'C':endpoint['C'].detach(),'means':endpoint['means'],'mu_train':endpoint['mu_train'],
            'mu_train_provenance':endpoint['mu_train_provenance'],'head':dict(endpoint['head_object'].state_dict()),
            'A_trainable':endpoint['A'].requires_grad,'C_trainable':endpoint['C'].requires_grad}


def load_inference(directory, bundle_sha256, device):
    """Copied public loader uses only owned bundle files and installed packages."""
    directory = Path(directory)
    manifest,guards = admit_bundle(directory,bundle_sha256)
    require(device in ('cpu','cuda'), 'fixed serving device required')
    modules,prefix = {},'_connected_serving_'+str(time.time_ns())+'_'
    for filename in SERVING_FILES | {'joint_relational_compaction.py'}:
        modules[filename] = load_authenticated(prefix+filename.removesuffix('.py'),directory/filename,manifest['code'][filename],guards)
    source,original,extract = (modules[n] for n in ('qualify_siglip2_substrate_cpu.py','train_siglip2_substrate_adaptation.py',
                                                   'extract_siglip2_vision_source.py'))
    import torch
    path = directory/'endpoint.pt'
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    try:
        require(disk.keys() == INFERENCE_KEYS and disk['schema'] == INFERENCE_SCHEMA and disk['arm'] in ARMS and
                original.fingerprint(disk) == manifest['endpoint_state_sha256'] and
                original.fingerprint({k:v for k,v in disk.items() if k != 'fixed_sha256'}) == disk['fixed_sha256'] and
                source.numerical_flags() == disk['numerical_flags'] and
                disk['vision_sha256'] == manifest['vision_sha256'] and disk['encoder_identity'] == manifest['encoder_identity'] and
                disk['base_vision']['sha256'] == manifest['base_vision_sha256'] and
                disk['scope']['arm'] == 'control' and disk['scope']['payload']['scope_sha256'] == CONTROL_SHA256 and
                len(disk['scope']['payload']['class_names']) == 1008,
                'complete original-scope/updated inference identity differs')
        env = manifest['environment']
        construct_context = {'packages':env['packages'],'guards':guards,'extract':extract,
            'sources':{'native_environment':{'vision_constructor':{'path':env['vision_constructor']}}}}
        owned_base = {'checkpoint':{'path':str(directory/'vision.pt'),'sha256':manifest['files']['vision.pt']},
                      'sha256':disk['base_vision']['sha256']}
        model,processor,cache,_,structure = construct_encoder(source,original,construct_context,disk['config'],disk['buffers'],
            directory/'processor.json',owned_base,disk['encoder'])
        # The overlay is copied by apply_overlay within the shared strict CPU
        # constructor before device transfer; the full updated identity is checked.
        require(original.fingerprint(model.state_dict()) == disk['vision_sha256'] and
                structure == disk['encoder_identity']['runtime'], 'updated full448 portable reload differs')
        model.requires_grad_(False).eval().to(device)
        with path.open('rb') as stream:
            pages = original.CheckpointPages(stream)
            copied = {k:owned_copy(disk[k],pages,device) for k in ('head','A','C','means','mu_train','common_statistics')}
            head = modules['train_siglip2_cached_readout.py'].head_from('control',tensors=copied.pop('head')).to(device).requires_grad_(False).train()
            endpoint = {**copied,'A':torch.nn.Parameter(copied['A'],requires_grad=True),
                'C':torch.nn.Parameter(copied['C'],requires_grad=True),'head_object':head,'model':model,
                'processor_object':processor,'processor_cache':cache,'processor':copy.deepcopy(disk['processor']),
                'arm':disk['arm'],'scope':copy.deepcopy(disk['scope']),
                'mu_train_provenance':copy.deepcopy(disk['mu_train_provenance']),
                'encoder_identity':copy.deepcopy(disk['encoder_identity']),'vision_sha256':disk['vision_sha256'],
                'flags':copy.deepcopy(disk['numerical_flags']),'device':device,'modules':modules,'guards':guards,'manifest':manifest}
            endpoint['readout_sha256'] = original.fingerprint(inference_readout_tree(endpoint))
            del copied,pages,model,processor,head
    finally:
        del disk
        gc.collect()
        mapping_absent(path)
    require(encoder_facts(endpoint,original,source,manifest['environment']['packages'],serving=True)['vision_sha256'] ==
            endpoint['vision_sha256'], 'public updated encoder identity differs')
    return endpoint


def owned_copy(value, pages, device='cpu'):
    import torch
    if isinstance(value,torch.Tensor):
        return pages.copy(value,device)
    if isinstance(value,dict):
        return {k:owned_copy(v,pages,device) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return type(value)(owned_copy(v,pages,device) for v in value)
    return copy.deepcopy(value)


def inference_outputs(endpoint, images):
    import torch
    from torch.nn import functional as F
    modules,device = endpoint['modules'],endpoint['device']
    original,source = modules['train_siglip2_substrate_adaptation.py'],modules['qualify_siglip2_substrate_cpu.py']
    for module in modules.values():
        bound_file({},module.__file__,endpoint['guards'][module.__file__])
    require(0 < len(images) <= 32 and source.numerical_flags() == endpoint['flags'], 'serving batch/numerics differ')
    require(endpoint['encoder_identity'] == endpoint['manifest']['encoder_identity'] and
            endpoint['vision_sha256'] == endpoint['manifest']['vision_sha256'] and
            all(p.grad is None and not p.requires_grad and p.dtype == torch.float32 and p.device.type == device
                for p in endpoint['head_object'].parameters()) and
            all(m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in endpoint['head_object'].modules()), 'serving authenticated encoder/frozen head roles/hooks differ')
    require(encoder_facts(endpoint,original,source,endpoint['manifest']['environment']['packages'],serving=True)['vision_sha256'] ==
            endpoint['vision_sha256'] and original.fingerprint(inference_readout_tree(endpoint)) == endpoint['readout_sha256'],
            'current .data updated encoder/readout/role substitution rejected')
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if device == 'cuda' else []
    pixels = endpoint['processor_object'](images=images,return_tensors='pt')['pixel_values']
    require(pixels.shape == (len(images),3,256,256) and pixels.dtype == torch.float32 and torch.isfinite(pixels).all().item(),
            'owned processor pixels differ')
    with torch.no_grad():
        with torch.autocast(device,dtype=torch.float16,enabled=device == 'cuda'):
            pooled = endpoint['model'](pixel_values=pixels.to(device)).pooler_output
        with torch.autocast(device,enabled=False):
            features = F.normalize(pooled.float(),dim=1)
            raw = fullfeature_raw_features(features,endpoint['head_object'],endpoint['A'],endpoint['means'],endpoint['C'],
                endpoint['mu_train'],endpoint['arm'],modules['quadratic_readout.py'],modules['prototype_residual_readout.py'])
            require((raw.norm(dim=1) > 0).all().item(), 'nonzero portable raw required')
            unit = F.normalize(raw,dim=1)
            packed = modules['joint_relational_compaction.py'].pack_int8_unit_embeddings(unit.cpu())
    require(torch.equal(cpu_rng,torch.random.get_rng_state()) and
            all(torch.equal(a,b) for a,b in zip(cuda_rng,torch.cuda.get_rng_state_all() if device == 'cuda' else [],strict=True)),
            'serving complete RNG changed')
    return {'raw':raw.cpu(),'unit':unit.cpu(),'codes':packed.codes.cpu(),'inverse_norms':packed.inverse_norms.cpu(),'wire':packed.to_bytes()}


def release_inference(endpoint):
    modules = tuple(endpoint['modules'].values())
    refs = [weakref.ref(endpoint[n]) for n in ('model','processor_object','head_object','A','C','mu_train')]
    refs += [weakref.ref(p) for p in (*endpoint['model'].parameters(),*endpoint['model'].buffers(),
                                    *endpoint['head_object'].parameters(),*endpoint['head_object'].buffers())]
    cache = endpoint['processor_cache']
    require(_processor_cache(endpoint['processor_object'],endpoint['guards']) is cache, 'serving processor teardown authority differs')
    cache.cache_clear()
    require(cache.cache_info().currsize == 0, 'serving processor teardown failed')
    endpoint.clear()
    for module in modules:
        require(sys.modules.pop(module.__name__,None) is module, 'owned serving registry changed')
    gc.collect()
    require(all(ref() is None for ref in refs), 'serving model/processor/readout lifetime survived release')
    if 'torch' in sys.modules and sys.modules['torch'].cuda.is_initialized():
        sys.modules['torch'].cuda.empty_cache()


def read_images(context, state, batch):
    from PIL import Image
    images = []
    try:
        for ordinal in batch:
            row,path,_ = context['trainer'].canonical_row(context,state,ordinal)
            bound_file(context['guards'],path,row['image_sha256'])
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
    except BaseException:
        for image in images:
            image.close()
        raise
    return images


def image_witness(context, state):
    """Independent live-state oracle with exactly the public serving arithmetic role."""
    import torch
    from torch.nn import functional as F
    integrity(context,state,state['identity'])
    batch = state['schedules'][str(state['seed'])][0].tolist()[:2 if state['device'] == 'cpu' else 16]
    images = read_images(context,state,batch)
    try:
        rng = torch.random.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state_all() if state['device'] == 'cuda' else []
        pixels = state['processor_object'](images=images,return_tensors='pt')['pixel_values']
        with torch.no_grad():
            with torch.autocast(state['device'],dtype=torch.float16,enabled=state['device'] == 'cuda'):
                pooled = state['model'](pixel_values=pixels.to(state['device'])).pooler_output
            with torch.autocast(state['device'],enabled=False):
                features = F.normalize(pooled.float(),dim=1)
                raw = context['trainer'].raw_features(context,state,features)
                output = context['old'].packed_outputs(context['legacy'],raw)
        require(torch.equal(rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in
                zip(cuda_rng,torch.cuda.get_rng_state_all() if state['device'] == 'cuda' else [],strict=True)),
                'independent oracle complete RNG changed')
        digest = fingerprint(context,output)
        del pixels,pooled,features,raw,output
        integrity(context,state,state['identity'])
        return {'batch':batch,'raw_unit_packed_sha256':digest,'vision_sha256':state['current_encoder']['vision_sha256'],
                'seed':state['seed'],'device':state['device'],'arm':state['arm']}
    finally:
        for image in images:
            image.close()


def expect_rejection(call, message):
    try:
        call()
    except ValueError:
        return
    raise ValueError(message)


def tamper_witness(context, state, identity):
    """Run real public/training byte, role, source, configuration and ownership mutants."""
    import torch
    params = dict(state['model'].named_parameters())
    values = [params[n] for n in MLP]+[next(p for n,p in params.items() if n not in MLP),
              state['A'],state['C'],next(state['head_object'].parameters()),
              dict(state['model'].named_buffers())['embeddings.position_ids']]
    for value in values:
        backup,version = value.detach().clone(),value._version
        try:
            value.data.reshape(-1)[0] += 1
            require(value._version == version, 'mutant must bypass tensor version')
            expect_rejection(lambda:integrity(context,state,identity), 'current .data mutant accepted')
        finally:
            value.data.copy_(backup)
    del value,backup,values
    registrations = state['model'].embeddings._non_persistent_buffers_set
    registrations.remove('position_ids')
    try:
        expect_rejection(lambda:integrity(context,state,identity), 'nonpersistent registration mutant accepted')
    finally:
        registrations.add('position_ids')
    config = state['model'].config
    previous = config._attn_implementation
    try:
        config._attn_implementation = 'eager'
        expect_rejection(lambda:integrity(context,state,identity), 'configuration mutant accepted')
    finally:
        config._attn_implementation = previous
    group = state['optimizer_object'].param_groups[0]['params']
    group.reverse()
    try:
        expect_rejection(lambda:integrity(context,state,identity), 'reordered optimizer ownership accepted')
    finally:
        group.reverse()
    if state['arm'] == 'candidate':
        parameter = params[MLP[0]]
        parameter.requires_grad_(False)
        try:
            expect_rejection(lambda:integrity(context,state,identity), 'candidate role mutant accepted')
        finally:
            parameter.requires_grad_(True)
    integrity(context,state,identity)
    return True


def portable_mutants(portable, endpoint, images, directory):
    import torch
    model = endpoint['model']
    params = dict(model.named_parameters())
    original = endpoint['modules']['train_siglip2_substrate_adaptation.py']
    for value in [*tuple(params[n] for n in MLP),next(p for n,p in params.items() if n not in MLP),
                  next(endpoint['head_object'].parameters()),endpoint['C'],endpoint['mu_train']]:
        backup,version = value.detach().clone(),value._version
        try:
            value.data.reshape(-1)[0] += .25
            require(value._version == version, 'public .data mutant must bypass versions')
            expect_rejection(lambda:portable.inference_outputs(endpoint,images), 'public current-byte mutant accepted')
        finally:
            value.data.copy_(backup)
    del value,backup
    config = model.config
    prior = config._attn_implementation
    try:
        config._attn_implementation = 'eager'
        expect_rejection(lambda:portable.inference_outputs(endpoint,images), 'public config mutant accepted')
    finally:
        config._attn_implementation = prior
    registration = model.embeddings._non_persistent_buffers_set
    registration.remove('position_ids')
    try:
        expect_rejection(lambda:portable.inference_outputs(endpoint,images), 'public buffer registration mutant accepted')
    finally:
        registration.add('position_ids')
    # Exact original-four substitution is stronger than an arbitrary mutation.
    if endpoint['arm'] == 'candidate':
        path = directory/'vision.pt'
        vision = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
        backups = {n:params[n].detach().clone() for n in MLP}
        try:
            with torch.no_grad():
                for name in MLP:
                    params[name].copy_(vision['vision'][name])
            expect_rejection(lambda:portable.inference_outputs(endpoint,images), 'original encoder served instead of updated overlay')
        finally:
            with torch.no_grad():
                for name in MLP:
                    params[name].copy_(backups[name])
            del vision,backups
            gc.collect()
            mapping_absent(path)
    # A pooling-head member can share the nominated fc2.bias shape; strict names
    # and updated complete identity must reject the substitution as well.
    nominated = MLP[-1]
    wrong = next(p for n,p in params.items() if n.startswith('head.') and p.shape == params[nominated].shape)
    backup = params[nominated].detach().clone()
    try:
        params[nominated].data.copy_(wrong)
        expect_rejection(lambda:portable.inference_outputs(endpoint,images), 'same-shaped pooling-head substitution accepted')
    finally:
        params[nominated].data.copy_(backup)
    return True


def qualify_bundle(context, directory, digest, witness):
    """Two sequential copied public loads under original dependency-denial guard."""
    trainer = context['trainer']
    # Retain only the small authenticated row namespace after training release.
    scope = context['initial']['scope']
    shell = {'scope':scope,'original_rows':scope['payload']['original_rows'],'target':scope['payload']['targets']}
    images = read_images(context,shell,witness['batch'])
    hashes = []
    try:
        for _ in range(2):
            name = '_connected_owned_entry_'+str(time.time_ns())
            portable = load_authenticated(name,directory/'train_siglip2_connected_mlp.py',
                context['connected_code']['train_siglip2_connected_mlp.py'],{})
            manifest = read_json({'path':str(directory/'bundle.json'),'sha256':digest},context['guards'])
            trainer.authenticate_bundle_environment(context,manifest['environment'])
            endpoint = None
            try:
                with trainer.deny_training_dependencies(context,directory,manifest['environment']):
                    endpoint = portable.load_inference(directory,digest,witness['device'])
                    context['live_model'] = weakref.ref(endpoint['model'])
                    output = portable.inference_outputs(endpoint,images)
                    require(fingerprint(context,output) == witness['raw_unit_packed_sha256'] and
                            endpoint['vision_sha256'] == witness['vision_sha256'], 'independent same-role raw/unit/packed updated reload differs')
                    hashes.append(fingerprint(context,output))
                    del output
                    portable_mutants(portable,endpoint,images,directory)
                    require(fingerprint(context,portable.inference_outputs(endpoint,images)) == witness['raw_unit_packed_sha256'],
                            'public parity changed after restored mutants')
            finally:
                if endpoint is not None:
                    portable.release_inference(endpoint)
                require(sys.modules.pop(name,None) is portable, 'copied loader registry changed')
            trainer.require_no_training(context)
            resource_check(context)
    finally:
        for image in images:
            image.close()
    require(hashes == [witness['raw_unit_packed_sha256']]*2, 'sequential independent public parity differs')
    return {'strict_reload_exact':True,'native_training_inference_exact':True,'inference_artifact_independent':True,
            'bundle_original_dependencies_denied':True,'updated_source_mutants_rejected':True,
            'raw_unit_packed_sha256':hashes[0],'vision_sha256':witness['vision_sha256'],'bundle':{'path':str(directory/'bundle.json'),'sha256':digest}}


def arm_run(context, arm, seed, device, *, discarded_update=False):
    tick = time.perf_counter()
    args = context['connected_args']
    state = fresh(context,arm,seed,device)
    identity = copy.deepcopy(state['identity'])
    prefix = context['connected_args'].output/(arm+'-'+str(seed))
    total = 1 if discarded_update else 17 if args.phase == 'mechanics' else 128
    rows = []
    try:
        for step in range(1,total+1):
            row = update(context,state,identity,step)
            if args.phase == 'train' and step <= 17:
                mechanics = context['connected_terminals'][f'mechanics:{seed}:{arm}']
                require(diagnostic(row) == diagnostic(mechanics['steps'][step-1]), 'fresh first17 mechanics replay differs')
            rows.append(row)
        tamper_witness(context,state,identity)
        witness = image_witness(context,state)
        checkpoint = Path(str(prefix)+'-terminal.pt')
        sha,digest = save(context,state,identity,checkpoint)
        release(context,state)
        if args.phase == 'mechanics':
            state = fresh(context,arm,seed,device)
            require(state['identity'] == identity, 'independent fresh mechanics identity differs')
            first8 = [update(context,state,identity,step) for step in range(1, 9)]
            require([diagnostic(r) for r in first8] == [diagnostic(r) for r in rows[:8]], 'independent first8 replay differs')
            checkpoint8 = Path(str(prefix)+'-step8.pt')
            sha8,digest8 = save(context,state,identity,checkpoint8)
            release(context,state)
            state = restore(context,checkpoint8,sha8,digest8,identity,8)
            resumed = [update(context,state,identity,step) for step in range(9, 18)]
            require([diagnostic(r) for r in resumed] == [diagnostic(r) for r in rows[8:]] and
                    fingerprint(context,payload(context,state,identity)) == digest,
                    'uninterrupted17 vs independent serialized8+9 complete payload differs')
            release(context,state)
        state = restore(context,checkpoint,sha,digest,identity,total)
        require(image_witness(context,state) == witness, 'independent state image/raw/unit/packed restore differs')
        members = inference_members(context,state)
        release(context,state)
        directory = Path(str(prefix)+'-bundle')
        bundle = export_bundle(context,members,directory)
        del members
        parity = qualify_bundle(context,directory,bundle['sha256'],witness)
        result = {'identity':identity,'steps':rows,'checkpoint':{'path':str(checkpoint),'sha256':sha},
                  'terminal_state_sha256':digest,'parity':parity,'independent_17_vs_8_plus_9':args.phase == 'mechanics',
                  'training_updates':total,'total_training_core_seconds':time.perf_counter()-tick}
        return result
    finally:
        if state:
            release(context,state)


def check_steps(context, rows, identity):
    """Retain all original complete B64/two-view/membership/arithmetic predicates."""
    trainer = context['trainer']
    qualification = select_initializer(context['original_cpu_record'],identity['seed'])
    bank = qualification['ranking_bank']
    projected = []
    for row in rows:
        require(row['arm'] == identity['arm'] and row['batch'] == qualification['scope_schedule'][row['step']-1] and
                row['gradient_norms'].keys() == row['before_sha256'].keys() == row['after_sha256'].keys() ==
                set(identity['parameter_names']) and all(math.isfinite(v) and v > 0 for v in row['gradient_norms'].values()) and
                all(row['before_sha256'][n] != row['after_sha256'][n] for n in identity['parameter_names']),
                'all-active named update/schedule predicates differ')
        require(([v['view'] for v in row['view_gradients']] == list(VIEWS) and
                 all(v['ranking_gradient_norms'].keys() == set(identity['parameter_names']) and
                     all(math.isfinite(n) and n > 0 for n in v['ranking_gradient_norms'].values()) for v in row['view_gradients']))
                if row['step'] == 1 else row['view_gradients'] == [], 'both-view original SmoothAP connectivity differs')
        projected.append({k:row[k] for k in ('step','batch','membership','full_membership_sha256','full_valid','mse','rank',
            'loss','preclip_norm','state_sha256','core_seconds','seconds','arm','ranking_gradient_norm','ranking_C_gradient_norm')})
        projected[-1].update(active_anchors=sum(m['active'] for m in row['membership']),scale=128,
            gradient_norm=row['gradient_norms']['A'],C_gradient_norm=row['gradient_norms']['C'],
            A_before_sha256=row['before_sha256']['A'],A_after_sha256=row['after_sha256']['A'],
            C_before_sha256=row['before_sha256']['C'],C_after_sha256=row['after_sha256']['C'])
        require(row['scale'] == (128. if identity['device'] == 'cuda' else 1.), 'original numerical-role scaler differs')
    trainer.check_steps(projected,1,len(rows),bank)


def check_terminal(context, record, phase, arm, seed):
    launch = record['launch']
    check_launch(launch,SimpleNamespace(execution_sha256=context['connected_args'].execution_sha256,
                                      phase=phase,arm=arm,seed=seed))
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
            record['seed'] == seed and method(launch) == method(context['connected_launch']) and
            record['execution_sha256'] == context['connected_args'].execution_sha256 and
            record['code'] == context['connected_code'] and record['source'] == context['source'] and
            record['resource_policy'] == policy(phase) and record['quality_read'] is False and
            record['state_reuse_eligible'] is False and record['cost_qualified'] is False and
            all(record[k] is True for k in ('pass','exit_rehash_pass','sequential_model_ownership',
                'strict_reload_exact','native_training_inference_exact','inference_artifact_independent',
                'bundle_original_dependencies_denied','updated_source_mutants_rejected','both_locks_held_in_parent_authority')) and
            0 < record['total_training_core_seconds'] < record['wall_seconds'] < policy(phase)['seconds'] and
            0 < record['process_peak_rss_kib'] <= 8*1024**2 and
            record['invocation']['optimize'] == 0 and record['numerical_flags'] == context['original_cpu_record']['numerical_flags'],
            'complete prospective terminal contract differs')
    results = record['qualifications'] if phase == 'cpu' else [record['result']]
    require(len(results) == 1, 'one discarded CPU update/full arm result required')
    result = results[0]
    identity = result['identity']
    require(identity['arm'] == ('candidate' if phase == 'cpu' else arm) and identity['seed'] == seed and
            identity['device'] == ('cpu' if phase == 'cpu' else 'cuda') and identity['method'] == method(launch) and
            result['training_updates'] == (1 if phase == 'cpu' else 17 if phase == 'mechanics' else 128) and
            result['independent_17_vs_8_plus_9'] is (phase == 'mechanics') and
            identity['scope'] == {'arm':'control','manifest_sha256':SCOPE_SHA256,'arm_sha256':CONTROL_SHA256} and
            (identity['parameter_names'],identity['parameter_shapes']) == parameter_roles(identity['arm'])[:2],
            'terminal restored original-scope/active group/complete step identity differs')
    check_steps(context,result['steps'],identity)
    if phase == 'cpu':
        require([q['seed'] for q in record['accepted_initializers']] == list(SEEDS) and
                all(q['checkpoint'] == select_initializer(context['original_cpu_record'],q['seed'])['checkpoint'] and
                    q['canonical_initial'] == select_initializer(context['original_cpu_record'],q['seed'])['canonical_initial']
                    for q in record['accepted_initializers']), 'both original seed-specific initializers required')
    if phase == 'train':
        require(result['fresh_first17_replay'] is True, 'TRAIN first17 mechanics replay required')


def admit_terminal(context, unit, phase, arm, seed):
    """NEW terminal predicates plus the unchanged original uncached UNIT reader."""
    trainer,guards = context['trainer'],context['guards']
    check_unit(unit)
    reader = context['legacy']['original'].FlatAdmission()
    record = context['nearest'].read_json(unit['receipt'],guards,admission=reader)
    check_terminal(context,record,phase,arm,seed)
    require(record['authority']['sha256'] == record['authority_sha256'] and
            context['nearest'].read_json(record['authority'],guards,admission=reader) == record['launch'] and
            record['invocation']['argv'] == cli(context['connected_root'],record['authority']['path'],record['authority']['sha256'],
                context['connected_args'].execution_sha256,phase,arm,seed,Path(unit['receipt']['path']).parent) and
            all(record['invocation'][k] == context['original_cpu_record']['invocation'][k]
                for k in ('python','python_sha256','python_version')), 'actual terminal authority/CLI/interpreter differs')
    for path,digest in record['input_guards'].items():
        reader.bound_file(guards,path,digest)
    expected = {str(context['connected_root']/'execution.json'):context['connected_args'].execution_sha256,
                **{str(context['connected_root']/n):h for n,h in context['connected_code'].items()},
                **{str(Path(context['connected_launch']['witness']['root'])/n):h
                   for n,h in context['connected_launch']['witness']['files'].items()}}
    require(all(record['input_guards'].get(p) == h for p,h in expected.items()), 'complete new source guards differ')
    final = context['fitter'].original_terminal_reader(context['fit_context'])(reader,record,unit,policy(phase)['seconds'],guards)
    for value in (record['cgroup_before'],record['cgroup_after'],final):
        context['old'].zero_events(value)
    require(unit['invocation_id'] not in context['legacy']['invocations'], 'reused terminal invocation forbidden')
    context['legacy']['invocations'].add(unit['invocation_id'])
    context.setdefault('connected_terminals',{})[f'{phase}:{seed}:{arm}'] = record
    context.setdefault('connected_terminal_cgroups',{})[f'{phase}:{seed}:{arm}'] = final
    if phase == 'train' and arm == 'candidate':
        control_unit = record['launch']['fresh_control']
        control = read_json(control_unit['receipt'],guards)
        check_terminal(context,control,'train','control',seed)
        context['fitter'].original_terminal_reader(context['fit_context'])(reader,control,control_unit,600,guards)
        require(record['total_training_core_seconds']/control['total_training_core_seconds'] <= 1.50 and
                unit['service_seconds']/control_unit['service_seconds'] <= 1.50,
                'fresh live-control whole-service/core cost ratio failed')
    return record


def cpu_run(context):
    tick = time.perf_counter()
    trainer = context['trainer']
    result = arm_run(context,'candidate',SEEDS[0],'cpu',discarded_update=True)
    accepted = []
    for seed in SEEDS:
        qualification = select_initializer(context['original_cpu_record'],seed)
        state,identity = load_initializer(context,qualification)
        refs = trainer.tensor_weakrefs(context,{k:state[k] for k in STATIC_KEYS+('A','C')})
        accepted.append({'seed':seed,'checkpoint':qualification['checkpoint'],'canonical_initial':qualification['canonical_initial']})
        trainer.release(context,state)
        require(all(ref() is None for ref in refs), 'model-free original initializer lifetime survived release')
        resource_check(context)
    return {'qualifications':[result],'accepted_initializers':accepted,
            'total_training_core_seconds':time.perf_counter()-tick}


def run(args):
    require(not sys.flags.optimize and re.fullmatch('[0-9a-f]{32}',os.environ.get('INVOCATION_ID','')) and
            (os.environ.get('CUDA_VISIBLE_DEVICES') == '' if args.phase == 'cpu' else
             os.environ.get('CUDA_VISIBLE_DEVICES') == '0' and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8'),
            'original enclosing CPU/single-GPU unit/numerics required')
    require(sys.argv == cli(HERE,args.authority,args.authority_sha256,args.execution_sha256,args.phase,args.arm,args.seed,args.output),
            'fixed canonical CLI required')
    context = authority(args)
    trainer,legacy = context['trainer'],context['legacy']
    source = legacy['source_driver']
    context['phase_seconds']['authority'] = time.perf_counter()-STARTED
    prior = context['original_cpu_record']['invocation']
    python = Path(sys.executable).resolve()
    require(str(python) == prior['python'] and legacy['extract'].sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'], 'accepted original interpreter required')
    packages = source.package_origins(legacy['prior'])
    require(packages == legacy['selected']['packages'], 'original package preflight differs')
    legacy['prior']['packages'] = packages
    import torch
    require(not torch.cuda.is_initialized() and torch.is_grad_enabled() and not torch.is_inference_mode_enabled(),
            'native construction must follow original admission')
    flags = context['original_cpu_record']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'original numerical flags differ')
    context['flags'] = legacy['flags'] = flags
    packing = legacy['selected']['launch']['helpers']['packing']
    legacy['packing'] = legacy['genuine'].load_helper('_quadratic_packing',packing['path'],packing['sha256'],context['guards'])
    connected_path = Path(context['connected_launch']['witness']['root'])/'connected_residual_readout.py'
    context['connected'] = load_authenticated('_connected_live_readout',connected_path,
        context['connected_launch']['witness']['files']['connected_residual_readout.py'],context['guards'])
    context['connected_function'] = (context['connected'].raw_features,context['connected'].raw_features.__code__)
    before = resource_check(context)
    args.output.mkdir()
    try:
        with timed(context,'complete_arm_work'):
            if args.phase == 'cpu':
                result = cpu_run(context)
            else:
                require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one genuine CUDA encoder required')
                arm = arm_run(context,args.arm,args.seed,'cuda')
                if args.phase == 'train':
                    arm['fresh_first17_replay'] = True
                result = {'result':arm,'total_training_core_seconds':arm['total_training_core_seconds']}
        context['initial'].clear()
        trainer.require_no_training(context)
        serving_environment(context)
    finally:
        # Original complete uncached source/extraction/package/UNIT exit reader,
        # plus prospective guards. Never reconstruct the historical warm model.
        trainer.exit_rehash(context)
        if args.phase != 'cpu':
            trainer.audit_origin_diagnostics(context,context['nearest'].native_source_api(context),
                admission=legacy['original'].FlatAdmission(),require_exact=True)
        require(closure(HERE,args.execution_sha256,FILES,{}) == context['connected_code'], 'new exact2 exit closure differs')
        for path,digest in context['guards'].items():
            bound_file({},path,digest)
    after = resource_check(context)
    require(before['path'] == after['path'], 'whole enclosing cgroup changed')
    wall = time.perf_counter()-STARTED
    if args.phase == 'train' and args.arm == 'candidate':
        control = context['fresh_control_record']
        require(result['total_training_core_seconds']/control['total_training_core_seconds'] <= 1.50 and
                wall/context['connected_launch']['fresh_control']['service_seconds'] <= 1.50,
                'fresh live-control core/whole cost floor failed')
    parity = (result['qualifications'][0] if args.phase == 'cpu' else result['result'])['parity']
    receipt = {'schema':SCHEMA,'phase':args.phase,'arm':args.arm,'seed':args.seed,'pass':True,'quality_read':False,
        'state_reuse_eligible':False,'cost_qualified':False,'cost_requires_parent_normal_exit_units':True,
        'model_fit_qualified':args.phase == 'train','exit_rehash_pass':True,'sequential_model_ownership':True,
        'strict_reload_exact':parity['strict_reload_exact'],'native_training_inference_exact':parity['native_training_inference_exact'],
        'inference_artifact_independent':parity['inference_artifact_independent'],
        'bundle_original_dependencies_denied':parity['bundle_original_dependencies_denied'],
        'updated_source_mutants_rejected':parity['updated_source_mutants_rejected'],
        'source':copy.deepcopy(context['source']),'launch':context['connected_launch'],'code':context['connected_code'],
        'execution_sha256':args.execution_sha256,'authority':{'path':str(args.authority),'sha256':args.authority_sha256},
        'authority_sha256':args.authority_sha256,'resource_policy':policy(args.phase),'numerical_flags':flags,
        'wall_seconds':wall,'process_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated() if args.phase != 'cpu' else 0,
        'cgroup_before':before,'cgroup_after':after,'input_guards':dict(context['guards']),
        'phase_seconds':context['phase_seconds'],'both_locks_held_in_parent_authority':True,
        'terminal_exit_and_both_locks_require_parent_receipt':True,
        'invocation':{'argv':sys.argv,'python':str(python),'python_sha256':prior['python_sha256'],
            'python_version':sys.version,'optimize':sys.flags.optimize,'invocation_id':os.environ['INVOCATION_ID'],
            'pid':os.getpid(),'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
            'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES']},**result}
    check_terminal(context,receipt,args.phase,args.arm,args.seed)
    trainer.write_json(context,args.output/'receipt.json',receipt)
    resource_check(context)
    return receipt


def parser():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--execution-sha256',required=True)
    p.add_argument('--authority',type=Path,required=True)
    p.add_argument('--authority-sha256',required=True)
    p.add_argument('--phase',choices=('cpu','mechanics','train'),required=True)
    p.add_argument('--arm',choices=ARMS,required=True)
    p.add_argument('--seed',type=int,choices=SEEDS,required=True)
    p.add_argument('--output',type=Path,required=True)
    return p


if __name__ == '__main__':
    run(parser().parse_args())
