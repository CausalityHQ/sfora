#!/usr/bin/env python3
"""Independent compact-ranking CPU admission, native bundle export and paired scoring.

FILE={path: canonical absolute regular file, sha256: actual lowercase SHA256}.
UNIT={receipt:FILE, log:FILE, unit, invocation_id, service_seconds,
      native_peak_rss_kib, both_locks_held:true} describes a complete exited unit.
Endpoint={seed,arm,launch:FILE,terminal:UNIT,checkpoint:FILE,
          terminal_state_sha256,bundle:FILE,inference_state_sha256}.
The launch schema and exact keys are LAUNCH_KEYS. CPU is metadata/synthetic
only, export is one seed/arm, score first admits only CONTINUE or KILL without
CIs, full applies equal-seed paired gates. Selection GO admits validation only.
No native qualification or product-quality result is claimed by this source.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import argparse
import base64
import copy
from contextlib import contextmanager
import csv
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import sysconfig
import time
from types import FunctionType, SimpleNamespace
import weakref

UNIT_STARTED = time.perf_counter()
SCHEMA = 'siglip2-compact-ranking-evaluation-v1'
AUTHORITY_SCHEMA = 'siglip2-compact-ranking-evaluation-launch-v1'
FILES = {'evaluate_siglip2_compact_ranking.py', 'test_compact_ranking_evaluation.py'}
TRAIN_FILES = {'train_siglip2_compact_ranking.py', 'test_siglip2_compact_ranking.py'}
TRAINING = {'root': '/home/riomus/runs/sfora-so400-compact-ranking-train-source-v7', 'execution_sha256': 'ee9f77da0bac2a90b09f87fdcd7e7935de9c8dd38b663d84ca8c57271f43ce26', 'code': {'train_siglip2_compact_ranking.py': '880a8e40a1b32e97786d6dcf2449e4ecbdc98a0a5d9e5a35d9c6a2519d7cb53c', 'test_siglip2_compact_ranking.py': '544dc5fc63adb53f312e9d0b7eae2419958adef71cc8318092bd733941654274'}}
ARMS = ('control', 'candidate')
SEEDS = (179061, 179069)
ORDER = ((179061,'control'),(179061,'candidate'),(179069,'candidate'),(179069,'control'))
METRICS = ('per_query_r1','per_query_ap')
PANELS = {'selection':(3449,1734,1715,498),'validation':(3479,1749,1730,498)}
NATIVE = {'torch','numpy','PIL','transformers','safetensors','torchvision','sfora'}
PACKAGING_SOURCES = {'__init__.py','_elffile.py','_manylinux.py','_musllinux.py','_parser.py','_structures.py',
 '_tokenizer.py','dependency_groups.py','direct_url.py','errors.py','licenses/__init__.py','licenses/_spdx.py',
 'markers.py','metadata.py','pylock.py','requirements.py','specifiers.py','tags.py','utils.py','version.py'}
RUNTIME_SOURCES = {'packaging':{'packaging/'+n for n in PACKAGING_SOURCES}|{'packaging-26.2.dist-info/METADATA'},
 'regex':{'regex/__init__.py','regex/_main.py','regex/_regex_core.py','regex-2026.6.28.dist-info/METADATA'},
 'tqdm':set(('tqdm/__init__.py tqdm/_monitor.py tqdm/_tqdm_pandas.py tqdm/cli.py tqdm/gui.py tqdm/std.py '
  'tqdm/utils.py tqdm/version.py tqdm/auto.py tqdm/autonotebook.py tqdm/asyncio.py '
  'tqdm-4.68.3.dist-info/METADATA').split()),
 'anyio':set(('anyio/__init__.py anyio/_core/__init__.py anyio/_core/_contextmanagers.py anyio/_core/_eventloop.py '
  'anyio/_core/_exceptions.py anyio/_core/_fileio.py anyio/_core/_resources.py anyio/_core/_signals.py '
  'anyio/_core/_sockets.py anyio/_core/_streams.py anyio/_core/_subprocesses.py '
  'anyio/_core/_synchronization.py anyio/_core/_tasks.py anyio/_core/_tempfile.py anyio/_core/_testing.py '
  'anyio/_core/_typedattr.py anyio/abc/__init__.py anyio/abc/_eventloop.py anyio/abc/_resources.py '
  'anyio/abc/_sockets.py anyio/abc/_streams.py anyio/abc/_subprocesses.py anyio/abc/_tasks.py '
  'anyio/abc/_testing.py anyio/from_thread.py anyio/lowlevel.py anyio/streams/__init__.py '
  'anyio/streams/memory.py anyio/streams/stapled.py anyio/streams/tls.py anyio/to_thread.py').split()),
 'certifi':set(('certifi/__init__.py certifi/core.py').split()),
 'h11':set(('h11/__init__.py h11/_abnf.py h11/_connection.py h11/_events.py h11/_headers.py h11/_readers.py '
  'h11/_receivebuffer.py h11/_state.py h11/_util.py h11/_version.py h11/_writers.py').split()),
 'httpcore':set(('httpcore/__init__.py httpcore/_api.py httpcore/_async/__init__.py httpcore/_async/connection.py '
  'httpcore/_async/connection_pool.py httpcore/_async/http11.py httpcore/_async/http2.py '
  'httpcore/_async/http_proxy.py httpcore/_async/interfaces.py httpcore/_async/socks_proxy.py '
  'httpcore/_backends/__init__.py httpcore/_backends/anyio.py httpcore/_backends/auto.py '
  'httpcore/_backends/base.py httpcore/_backends/mock.py httpcore/_backends/sync.py '
  'httpcore/_backends/trio.py httpcore/_exceptions.py httpcore/_models.py httpcore/_ssl.py '
  'httpcore/_sync/__init__.py httpcore/_sync/connection.py httpcore/_sync/connection_pool.py '
  'httpcore/_sync/http11.py httpcore/_sync/http2.py httpcore/_sync/http_proxy.py '
  'httpcore/_sync/interfaces.py httpcore/_sync/socks_proxy.py httpcore/_synchronization.py '
  'httpcore/_trace.py httpcore/_utils.py').split()),
 'httpx':set(('httpx/__init__.py httpx/__version__.py httpx/_api.py httpx/_auth.py httpx/_client.py httpx/_config.py '
  'httpx/_content.py httpx/_decoders.py httpx/_exceptions.py httpx/_main.py httpx/_models.py '
  'httpx/_multipart.py httpx/_status_codes.py httpx/_transports/__init__.py httpx/_transports/asgi.py '
  'httpx/_transports/base.py httpx/_transports/default.py httpx/_transports/mock.py '
  'httpx/_transports/wsgi.py httpx/_types.py httpx/_urlparse.py httpx/_urls.py httpx/_utils.py '
  'httpx-0.28.1.dist-info/METADATA').split()),
 'huggingface_hub':set(('huggingface_hub/__init__.py huggingface_hub/constants.py huggingface_hub/dataclasses.py '
  'huggingface_hub/errors.py huggingface_hub/serialization/__init__.py huggingface_hub/serialization/_base.py '
  'huggingface_hub/serialization/_torch.py huggingface_hub/utils/__init__.py huggingface_hub/utils/_auth.py '
  'huggingface_hub/utils/_cache_assets.py huggingface_hub/utils/_cache_manager.py huggingface_hub/utils/_chunk_utils.py '
  'huggingface_hub/utils/_datetime.py huggingface_hub/utils/_detect_agent.py huggingface_hub/utils/_experimental.py '
  'huggingface_hub/utils/_fixes.py huggingface_hub/utils/_git_credential.py huggingface_hub/utils/_headers.py '
  'huggingface_hub/utils/_hf_uris.py huggingface_hub/utils/_http.py huggingface_hub/utils/_lfs.py '
  'huggingface_hub/utils/_pagination.py huggingface_hub/utils/_parsing.py huggingface_hub/utils/_paths.py '
  'huggingface_hub/utils/_runtime.py huggingface_hub/utils/_safetensors.py huggingface_hub/utils/_subprocess.py '
  'huggingface_hub/utils/_telemetry.py huggingface_hub/utils/_terminal.py huggingface_hub/utils/_typing.py '
  'huggingface_hub/utils/_validators.py huggingface_hub/utils/_xet.py huggingface_hub/utils/logging.py '
  'huggingface_hub/utils/tqdm.py huggingface_hub-1.16.1.dist-info/METADATA').split()),
 'idna':set(('idna/__init__.py idna/core.py idna/idnadata.py idna/intranges.py idna/package_data.py').split()),
 'jinja2':set(('jinja2/__init__.py jinja2/_identifier.py jinja2/async_utils.py jinja2/bccache.py jinja2/compiler.py '
  'jinja2/defaults.py jinja2/environment.py jinja2/exceptions.py jinja2/ext.py jinja2/filters.py '
  'jinja2/idtracking.py jinja2/lexer.py jinja2/loaders.py jinja2/meta.py jinja2/nodes.py jinja2/optimizer.py '
  'jinja2/parser.py jinja2/runtime.py jinja2/sandbox.py jinja2/tests.py jinja2/utils.py jinja2/visitor.py '
  'jinja2-3.1.6.dist-info/METADATA').split()),
 'markupsafe':set(('markupsafe/__init__.py markupsafe/_native.py').split()),
 'tokenizers':set(('tokenizers/__init__.py tokenizers/decoders/__init__.py tokenizers/implementations/__init__.py '
  'tokenizers/implementations/base_tokenizer.py tokenizers/implementations/bert_wordpiece.py '
  'tokenizers/implementations/byte_level_bpe.py tokenizers/implementations/char_level_bpe.py '
  'tokenizers/implementations/sentencepiece_bpe.py tokenizers/implementations/sentencepiece_unigram.py '
  'tokenizers/models/__init__.py tokenizers/normalizers/__init__.py tokenizers/pre_tokenizers/__init__.py '
  'tokenizers/processors/__init__.py tokenizers/trainers/__init__.py tokenizers-0.22.2.dist-info/METADATA').split()),
 'typing_extensions':set(('typing_extensions.py').split()),
 # Definition-time version checks read identity bytes, without granting package code.
 'accelerate':{'accelerate-1.14.0.dist-info/METADATA'},
 'aiohttp':{'aiohttp-3.14.1.dist-info/METADATA'},
 'filelock':{'filelock-3.29.4.dist-info/METADATA'},
 'hf_xet':{'hf_xet-1.5.1.dist-info/METADATA'},
 'numpy':{'numpy-2.5.0.dist-info/METADATA'},
 'pillow':{'pillow-12.2.0.dist-info/METADATA'},
 'pydantic':{'pydantic-2.13.4.dist-info/METADATA'},
 'pyyaml':{'pyyaml-6.0.3.dist-info/METADATA'},
 'safetensors':{'safetensors-0.8.0.dist-info/METADATA'},
 'torch':{'torch-2.12.1.dist-info/METADATA'},
}
NEAREST_EVALUATOR = {'root':'/home/riomus/runs/sfora-so400-compact-ranking-evaluation-reference-v1',
 'execution_sha256':'5c24fe113c03ae26d4ab68f21caf8e3a8d19c1a082e696fdf54abdd8bb73ab59',
 'code':{'evaluate_siglip2_nearest_ranking.py':'73e4386256c576329438da805cf6ff71ce67af7b4eae5b1074f2258d7d7029be',
         'test_nearest_ranking_evaluation.py':'a0e42560d1cce4b7e416e48a09ea27cf531e5e01c035ead5a387fea5cf48d90e'}}
GENUINE_PINS = {'evaluate_siglip2_genuine_views.py':'85fd39e08bb676cbfc827576bf5c5bfda7fb5ff0a9da800aa4d35a4d30a2c6aa',
                'test_siglip2_genuine_view_evaluation.py':'ca558a7d43f753a345e2fe67262e0880a9b82aaa778decc7993bc2a622bd0c35'}
REFERENCE = {'root': '/home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2',
 'execution_sha256': 'c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970',
 'code': {'evaluate_siglip2_prototype_residual.py': 'e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb',
          'test_siglip2_prototype_residual_evaluation.py': '5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49'}}
GENUINE_EXECUTION_SHA = '82e4e71362a474e58643214380736da606ca5b67a9cd1b50c6dc5fa997a3224f'
COST_POLICY = {'whole_service_ratio_max':1.50,'total_training_core_ratio_max':1.50,
 'core':'both-cache preparation + teacher construction + mining + paired forwards/backward + optimizer',
 'denominator':'fresh matched control128 per seed; complete cold whole units',
 'shared_export_seconds':283.636,'shared_export_in_ratios':False,
 'optimization_throughput_is_image_training_throughput':False}
LAUNCH_KEYS = {'schema','execution_sha256','training','nearest_evaluator','genuine_evaluator','reference',
 'phase','arm','seed','stage','panel','endpoints','selected_cpu','exports','first_selection','selection_go',
 'resource_policies','cost_policy','both_locks_held','selection_previously_exposed'}
READINESS = ('training_units','matched_costs','cpu_qualification','source_replay','concat_replay',
             'updated_state','train_witnesses','wire_readbacks','bundle_portability')

def require(condition, message):
    if not condition:
        raise ValueError(message)

def strict_json(raw):
    def pairs(items):
        result = {}
        for k, v in items:
            require(k not in result, 'duplicate JSON key')
            result[k] = v
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda v: require(False, 'nonfinite JSON: '+v))

def sha(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None

def check_file(value):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'} and
            isinstance(value['path'], str) and Path(value['path']).is_absolute() and sha(value['sha256']),
            'exact actual FILE required')

def bound_file(guards, path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file() and sha(expected), 'canonical FILE/SHA required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while raw := stream.read(1024**2):
            digest.update(raw)
            os.posix_fadvise(stream.fileno(), stream.tell()-len(raw), len(raw), os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'file SHA256 differs: '+str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting FILE authority')
    return path

def read_json(value, guards):
    check_file(value)
    path = bound_file(guards, value['path'], value['sha256'])
    with path.open('rb') as stream:
        raw = stream.read(64*1024**2+1)
    require(len(raw) <= 64*1024**2 and hashlib.sha256(raw).hexdigest() == value['sha256'], 'JSON size/hash differs')
    return strict_json(raw)

def closure(root, expected, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    code = read_json({'path':str(root/'execution.json'), 'sha256':expected}, guards)
    require(code.keys() == set(names) and all(sha(v) for v in code.values()), 'exact execution closure required')
    for name, digest in code.items():
        bound_file(guards, root/name, digest)
    return code

def merge_guards(target, values):
    for p, h in values.items():
        require(target.setdefault(p,h) == h, 'conflicting source authority: '+p)

def load_authenticated(name, path, digest, guards):
    require(name not in sys.modules, 'preloaded helper forbidden')
    raw = bound_file(guards, path, digest).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'helper bytes changed before execution')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path, 'helper origin differs')
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), vars(module))
    require(Path(module.__file__) == Path(module.__spec__.origin) == path, 'loaded helper origin differs')
    return module

def policy(phase):
    require(phase in ('cpu','export','score'), 'fixed evaluation phase required')
    return {'seconds': 300 if phase == 'export' else 500, 'host_bytes':8*1024**3,
            'swap_bytes':0, 'cuda_visible_devices':'0' if phase == 'export' else ''}

def check_unit(unit):
    require(isinstance(unit,dict) and unit.keys() == {'receipt','log','unit','invocation_id',
        'service_seconds','native_peak_rss_kib','both_locks_held'} and unit['both_locks_held'] is True and
        re.fullmatch('[A-Za-z0-9_.@-]+',unit['unit']) and re.fullmatch('[0-9a-f]{32}',unit['invocation_id']) and
        all(type(unit[k]) in (int,float) and math.isfinite(unit[k]) and unit[k]>0
            for k in ('service_seconds','native_peak_rss_kib')), 'complete actual UNIT required')
    check_file(unit['receipt']); check_file(unit['log'])

def batch_sizes(count):
    require(type(count) is int and count>0,'positive image population required')
    return [min(32,count-start) for start in range(0,count,32)]

def check_quality(value,count):
    require(value.keys() == {'recall_at_1','map_at_r',*METRICS} and
        len(value['per_query_r1']) == len(value['per_query_ap']) == count and
        all(type(v) in (int,float) and math.isfinite(v) and v in (0,1) for v in value['per_query_r1']) and
        all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1 for v in value['per_query_ap']) and
        all(type(value[k]) in (int,float) and math.isfinite(value[k]) and
            math.isclose(value[k],statistics.mean(value[m]),rel_tol=0,abs_tol=1e-12)
            for k,m in zip(('recall_at_1','map_at_r'),METRICS,strict=True)), 'complete finite per-query quality required')

def metric_deltas(quality,source,concat,count):
    require(quality.keys() == set(ARMS),'complete matched quality required')
    for q in (source,concat,*quality.values()):
        check_quality(q,count)
    return {n:{m:[b-a for a,b in zip(l[m],r[m],strict=True)] for m in METRICS}
            for n,l,r in (('candidate_minus_control',quality['control'],quality['candidate']),
                ('candidate_minus_source',source,quality['candidate']),('candidate_minus_concat',concat,quality['candidate']))}

def immediate_quality_pass(quality,source,concat,count):
    d = metric_deltas(quality,source,concat,count)
    return statistics.mean(d['candidate_minus_control'][METRICS[0]])>0 and \
        statistics.mean(d['candidate_minus_control'][METRICS[1]])>=0 and \
        all(statistics.mean(d['candidate_minus_source'][m])>=0 for m in METRICS) and \
        statistics.mean(d['candidate_minus_concat'][METRICS[0]])>0 and \
        statistics.mean(d['candidate_minus_concat'][METRICS[1]])>=0

def diagnostic_result(control,candidate):
    require(len(control) == len(candidate)<=128 and all(type(x) in (int,float) and math.isfinite(x)
        for x in (*control,*candidate)), 'finite complete fixed TRAIN diagnostic required')
    cv, av = sum(v<.05 for v in control), sum(v<.05 for v in candidate)
    delta = statistics.median(b-a for a,b in zip(control,candidate,strict=True)) if control else None
    demonstrated = bool(control and av<cv and delta>0)
    return {'triples':len(control),'control_violations':cv,'candidate_violations':av,'median_margin_delta':delta,
        'mechanism_demonstrated':demonstrated,'interpretation':'mechanism demonstrated on this panel' if demonstrated
        else 'mechanism not demonstrated on this panel','diagnostic_only':True,'utility_veto':False,'remine':False}

def quality_after_readiness(readiness,score):
    require(readiness.keys() == set(READINESS) and all(v is True for v in readiness.values()),
            'complete qualification/source/cost/wire admission must precede candidate quality')
    return score()

def check_resource_facts(record,phase):
    require(record['resource_policy'] == policy(phase) and
        type(record['wall_seconds']) in (int,float) and math.isfinite(record['wall_seconds']) and
        0 < record['wall_seconds'] < policy(phase)['seconds'] and
        type(record['process_peak_rss_kib']) in (int,float) and math.isfinite(record['process_peak_rss_kib']) and
        0 < record['process_peak_rss_kib'] <= 8*1024**2 and
        type(record['peak_cuda_allocated_bytes']) is int and
        (0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000 if phase == 'export' else
         record['peak_cuda_allocated_bytes'] == 0) and record['cuda_initialized'] is (phase == 'export'),
        'whole-unit endpoint resources/partial native receipt differ')

def json_form(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))

def seeds(stage):
    require(stage in ('first','full'), 'fixed prospective stage required')
    return SEEDS[:1] if stage == 'first' else SEEDS


def endpoint_order(stage):
    seeds(stage)
    return ORDER[:2] if stage == 'first' else ORDER


def label(endpoint):
    return endpoint['arm']+'-'+str(endpoint['seed'])


def check_code(value, names, pins=None):
    require(isinstance(value,dict) and value.keys() == {'root','execution_sha256','code'} and
        isinstance(value['root'],str) and Path(value['root']).is_absolute() and sha(value['execution_sha256']) and
        value['code'].keys() == set(names) and all(sha(h) for h in value['code'].values()) and
        (pins is None or value['code'] == pins), 'actual exact separate code descriptor required')


def check_endpoint(endpoint):
    require(isinstance(endpoint,dict) and endpoint.keys() == {'seed','arm','launch','terminal','checkpoint',
        'terminal_state_sha256','bundle','inference_state_sha256'} and endpoint['arm'] in ARMS and
        type(endpoint['seed']) is int and endpoint['seed'] in SEEDS and sha(endpoint['terminal_state_sha256']) and
        sha(endpoint['inference_state_sha256']), 'complete trained compact endpoint required')
    for k in ('launch','checkpoint','bundle'):
        check_file(endpoint[k])
    check_unit(endpoint['terminal'])
    root = Path(endpoint['terminal']['receipt']['path']).parent
    require(Path(endpoint['checkpoint']['path']) == root/'resume.pt' and
        Path(endpoint['bundle']['path']) == root/'bundle'/'bundle.json', 'trained endpoint artifact roles differ')


def check_launch(launch,args):
    require(isinstance(launch,dict) and launch.keys() == LAUNCH_KEYS and
        launch['schema'] == AUTHORITY_SCHEMA and launch['execution_sha256'] == args.execution_sha256 and
        launch['phase'] == args.phase and launch['arm'] == args.arm and launch['seed'] == args.seed and
        launch['both_locks_held'] is True and launch['selection_previously_exposed'] is True and
        launch['resource_policies'] == {p:policy(p) for p in ('cpu','export','score')} and
        launch['cost_policy'] == COST_POLICY and launch['nearest_evaluator'] == NEAREST_EVALUATOR and launch['reference'] == REFERENCE and
        launch['panel'] in PANELS, 'frozen compact evaluator launch differs')
    check_code(launch['training'],TRAIN_FILES)
    require(launch['training'] == TRAINING, 'actual parent-frozen trainer2 differs')
    check_code(launch['genuine_evaluator'],GENUINE_PINS,GENUINE_PINS)
    require(launch['genuine_evaluator']['root'] == '/home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4' and
        launch['genuine_evaluator']['execution_sha256'] == GENUINE_EXECUTION_SHA,
        'original paired-seed evaluator origin differs')
    require([(e['seed'],e['arm']) for e in launch['endpoints']] == list(endpoint_order(launch['stage'])),
        'prospective ordered endpoints required')
    for endpoint in launch['endpoints']:
        check_endpoint(endpoint)
    require((args.phase == 'export') == (args.arm in ARMS and type(args.seed) is int and
        args.seed in seeds(launch['stage'])) and (args.phase == 'export' or args.arm is args.seed is None),
        'export owns exactly one admitted seed/arm')
    require((launch['selected_cpu'] is None) == (args.phase == 'cpu') and
        (args.phase != 'cpu' or launch['panel'] == 'selection'), 'fresh stage CPU metadata qualification required')
    require(launch['exports'].keys() == ({label(e) for e in launch['endpoints']} if args.phase == 'score' else set()),
        'complete admitted export set required before score')
    require((launch['first_selection'] is None) == (launch['stage'] == 'first') and
        (launch['selection_go'] is None) == (launch['panel'] == 'selection') and
        (launch['panel'] != 'validation' or launch['stage'] == 'full'), 'first continuation/full selection GO required')
    for u in (launch['selected_cpu'],launch['first_selection'],launch['selection_go'],*launch['exports'].values()):
        if u is not None:
            check_unit(u)
    units = [e['terminal'] for e in launch['endpoints']]+list(launch['exports'].values())
    for k in ('selected_cpu','first_selection','selection_go'):
        if launch[k] is not None:
            units.append(launch[k])
    require(len({u['unit'] for u in units}) == len(units) and
        len({u['invocation_id'] for u in units}) == len(units), 'distinct complete evaluation/training units required')


def paired_cost(records,stage):
    require(records.keys() == set(endpoint_order(stage)), 'fresh complete paired-seed training costs required')
    costs = {}
    for seed in seeds(stage):
        pair = {a:records[seed,a] for a in ARMS}
        for r in pair.values():
            require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>0 for k in
                ('service_seconds','total_training_core_seconds')) and
                r['total_training_core_seconds'] <= r['service_seconds'], 'complete charged training core required')
        ratios = {n:pair['candidate'][k]/pair['control'][k] for n,k in
            (('whole_service_ratio','service_seconds'),('total_training_core_ratio','total_training_core_seconds'))}
        costs[str(seed)] = {**ratios,'pass':all(v<=1.50 for v in ratios.values()),
            **{a:{k:pair[a][k] for k in ('service_seconds','total_training_core_seconds')} for a in ARMS}}
    return costs


def decide(math_helper,quality,source,concat,stage,panel,intervals,costs):
    count = PANELS[panel][1]
    require(quality.keys() == {str(s) for s in seeds(stage)} and costs.keys() == quality.keys(), 'complete stage seeds required')
    all_deltas = {str(s):metric_deltas(quality[str(s)],source,concat,count) for s in seeds(stage)}
    deltas,average = math_helper.averaged_deltas(quality,stage,panel)
    immediate = all(immediate_quality_pass(quality[str(s)],source,concat,count) for s in seeds(stage))
    require(intervals.keys() == (set(METRICS) if stage == 'full' and immediate else set()),
        'first/immediate KILL has no confidence intervals')
    each_seed = all(statistics.mean(deltas[str(s)][METRICS[0]])>0 and
                    statistics.mean(deltas[str(s)][METRICS[1]])>=0 for s in seeds(stage))
    if stage == 'first':
        passed = immediate and math_helper.first_gate(deltas)
    else:
        passed = False
        if immediate:
            for m,v in intervals.items():
                require(v.keys() == {'mean_delta','product_lower95','product_upper95','query_lower95','query_upper95'} and
                    all(type(x) in (int,float) and math.isfinite(x) and -1<=x<=1 for x in v.values()) and
                    all(v[k+'_lower95']<=v[k+'_upper95'] for k in ('product','query')) and
                    math.isclose(v['mean_delta'],statistics.mean(average[m]),rel_tol=0,abs_tol=1e-12),
                    'finite paired interval/mean differs')
            _,passed = math_helper.quality_gate(deltas,intervals)
    reconstructed = paired_cost({(s,a):costs[str(s)][a] for s in seeds(stage) for a in ARMS},stage)
    require(costs == reconstructed, 'fresh matched per-seed cost decision differs')
    cost_pass = all(v['pass'] for v in costs.values())
    decision = ('CONTINUE' if stage == 'first' else 'GO') if passed and cost_pass else 'KILL'
    return {'decision':decision,'quality_pass':bool(passed),'each_seed_quality_pass':each_seed,
        'immediate_quality_pass':immediate,'cost_pass':cost_pass,'deltas':all_deltas,
        'mean_deltas':{m:statistics.mean(average[m]) for m in METRICS},
        'source_floor_pass':all(all(statistics.mean(all_deltas[str(s)]['candidate_minus_source'][m])>=0 for m in METRICS)
                              for s in seeds(stage)),
        'concat_floor_pass':all(statistics.mean(all_deltas[str(s)]['candidate_minus_concat'][METRICS[0]])>0 and
                               statistics.mean(all_deltas[str(s)]['candidate_minus_concat'][METRICS[1]])>=0 for s in seeds(stage)),
        'selection_go_admits_validation_only':decision == 'GO' and panel == 'selection',
        'global_production_goal_met':False,'product_go':False}


def parser():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--execution-sha256',required=True); p.add_argument('--authority',type=Path,required=True)
    p.add_argument('--authority-sha256',required=True); p.add_argument('--phase',choices=('cpu','export','score'),required=True)
    p.add_argument('--seed',type=int,choices=SEEDS); p.add_argument('--arm',choices=ARMS)
    p.add_argument('--output',type=Path,required=True)
    return p


def cli(args):
    result = [str(Path(__file__).absolute()),'--execution-sha256',args.execution_sha256,
        '--authority',str(args.authority),'--authority-sha256',args.authority_sha256,'--phase',args.phase]
    if args.seed is not None:
        result += ['--seed',str(args.seed),'--arm',args.arm]
    return result+['--output',str(args.output)]


def binding(context):
    return {k:context['launch'][k] for k in ('training','nearest_evaluator','genuine_evaluator','reference',
        'stage','panel','endpoints','cost_policy','both_locks_held','selection_previously_exposed')}


def guard_helpers(context):
    snapshots = context.setdefault('helper_snapshots',[])
    if not snapshots:
        for key in ('trainer','nearest_evaluator','math','reference','helper','baseline'):
            module = context[key]
            values = dict(vars(module))
            functions = [(f,f.__code__,f.__defaults__,copy.deepcopy(f.__kwdefaults__))
                         for f in values.values() if isinstance(f,FunctionType)]
            literals = {k:copy.deepcopy(v) for k,v in values.items() if k != '__builtins__' and
                        isinstance(v,(dict,list,tuple,set,frozenset))}
            # Mutable owned runtime caches are checked by the trainer's authenticated API.
            snapshots.append((module,Path(module.__file__),module.__spec__,values,functions,literals))
    for module,path,spec,values,functions,literals in snapshots:
        require(module.__spec__ is spec and Path(spec.origin) == Path(module.__file__) == path and
            sys.modules.get(module.__name__) is module and vars(module).keys() == values.keys() and
            all(vars(module)[k] is v for k,v in values.items()) and
            all(f.__code__ is code and f.__defaults__ == defaults and f.__kwdefaults__ == kw
                for f,code,defaults,kw in functions) and all(vars(module)[k] == v for k,v in literals.items()),
            'authenticated helper live source/global binding changed')
        bound_file({},path,context['guards'][str(path)])
def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded source admission')
    root,guards = Path(__file__).absolute().parent,{}
    code = closure(root,args.execution_sha256,FILES,guards)
    launch = read_json({'path':str(args.authority),'sha256':args.authority_sha256},guards)
    check_launch(launch,args)
    roots = [root]+[Path(launch[k]['root']) for k in ('training','nearest_evaluator','genuine_evaluator','reference')]
    require(args.output.is_absolute() and args.output.parent.resolve() == args.output.parent and
        not args.output.exists() and not args.output.is_symlink() and
        all(not a.is_relative_to(b) and not b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]) and
        all(not args.output.is_relative_to(p) and not p.is_relative_to(args.output) for p in roots),
        'separate immutable evaluator/trainer/reference closures and exclusive output required')
    loaded = {}
    for key,filename in (('training','train_siglip2_compact_ranking.py'),
        ('nearest_evaluator','evaluate_siglip2_nearest_ranking.py'),
        ('genuine_evaluator','evaluate_siglip2_genuine_views.py'),
        ('reference','evaluate_siglip2_prototype_residual.py')):
        fact = launch[key]
        require(closure(fact['root'],fact['execution_sha256'],fact['code'],guards) == fact['code'],
            'complete authenticated source closure differs: '+key)
        loaded[key] = load_authenticated('_compact_eval_'+key,Path(fact['root'])/filename,fact['code'][filename],guards)
    trainer,native,math_helper,reference = (loaded[k] for k in ('training','nearest_evaluator','genuine_evaluator','reference'))
    require(launch['reference'] == native.REFERENCE and trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and
        tuple(trainer.SEEDS) == SEEDS and trainer.SCHEMA == 'siglip2-compact-ranking-v1' and
        trainer.BUNDLE_SCHEMA == 'siglip2-compact-ranking-bundle-v1' and math_helper.ORDER == ORDER and
        math_helper.METRICS == METRICS and math_helper.PANELS == PANELS, 'owned trainer/paired-seed math contract differs')
    first = launch['endpoints'][0]; training = launch['training']
    targs = SimpleNamespace(execution_sha256=training['execution_sha256'],authority=Path(first['launch']['path']),
        authority_sha256=first['launch']['sha256'],phase='train',arm='control',seed=SEEDS[0],output=args.output)
    t = trainer.authority(targs)
    common_guards = {p:h for p,h in {**guards,**t['guards']}.items()
                     if p not in (str(args.authority),first['launch']['path'])}
    records = {}
    for endpoint in launch['endpoints']:
        seed,arm = endpoint['seed'],endpoint['arm']
        el = read_json(endpoint['launch'],guards)
        branch_args = SimpleNamespace(**{**vars(targs),'seed':seed,'arm':arm,
            'authority':Path(endpoint['launch']['path']),'authority_sha256':endpoint['launch']['sha256']})
        trainer.check_launch(el,branch_args)
        require(trainer.method(el) == trainer.method(t['launch']) and el['selected_cpu'] == t['launch']['selected_cpu'],
            'same frozen source/recipe and native CPU qualification required')
        for a in ARMS:
            require(el['selected_mechanics'] == t['launch']['selected_mechanics'], 'same two discarded mechanics061 receipts required')
            token='mechanics:'+str(SEEDS[0])+':'+a
            if token not in t['terminals']:
                trainer.admit_terminal(t,el['selected_mechanics'][a],'mechanics',a,SEEDS[0])
        record = trainer.admit_terminal(t,endpoint['terminal'],'train',arm,seed)
        require(record['authority'] == endpoint['launch'] and record['checkpoint'] == endpoint['checkpoint'] and
            record['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and
            record['bundle'] == endpoint['bundle'] and record['inference_state_sha256'] == endpoint['inference_state_sha256'] and
            record['completed_step'] == 128, 'complete TRAIN128/bundle endpoint differs')
        bound_file(guards,endpoint['checkpoint']['path'],endpoint['checkpoint']['sha256'])
        manifest,bundle_guards = trainer.admit_bundle(Path(endpoint['bundle']['path']).parent,endpoint['bundle']['sha256'])
        require(manifest['endpoint_state_sha256'] == endpoint['inference_state_sha256'] and
            all(manifest['code'][n] == training['code'][n] for n in TRAIN_FILES), 'bundle serving closure/endpoint differs')
        merge_guards(guards,bundle_guards)
        records[seed,arm] = {**record,'service_seconds':endpoint['terminal']['service_seconds']}
    for seed in seeds(launch['stage']):
        c,a = (records[seed,arm] for arm in ARMS)
        require(all(c[k] == a[k] for k in ('source','initial_A_sha256','initial_raw_unit_packed_sha256','numerical_flags')) and
            all(c['identity'][k] == a['identity'][k] for k in ('static_sha256','initial_A_sha256','parameter_names',
                'source','numerical_flags','initial_cpu_rng_sha256','initial_cuda_rng_sha256')) and
            [s['batch'] for s in c['steps']] == [s['batch'] for s in a['steps']],
            'fresh same-seed complete initialization/RNG/schedule differs')
    require(len({e['checkpoint']['sha256'] for e in launch['endpoints']}) == len(launch['endpoints']),
        'trained state reused between endpoints')
    costs = paired_cost(records,launch['stage'])
    # No held images or candidate quality are admitted after any cost failure.
    require(all(v['pass'] for v in costs.values()), 'fresh core/whole cost gate failed before held quality')
    archived = read_json(native.CONCAT_TERMINAL['receipt'],guards)
    require(archived['invocation']['invocation_id'] == native.CONCAT_TERMINAL['invocation_id'] and
        archived['schema'] == reference.SCHEMA and archived['phase'] == 'score' and archived['pass'] is True and
        archived['source_code'] == launch['reference']['code'] and
        archived['execution_sha256'] == launch['reference']['execution_sha256'] and
        archived['quality']['concat']['recall_at_1'] == 0.9648212226066898 and
        archived['quality']['concat']['map_at_r'] == 0.8177754035543956 and
        archived['spec']['panel'] == 'selection' and archived['validation_quality_exposed'] is False,
        'preserved accepted concat source/floors differ')
    spec = archived['spec']
    for item,pins in ((spec['original_evaluator'],reference.EVALUATOR_PINS),
        (reference.EVALUATION_REFERENCE,reference.REFERENCE_PINS),(reference.ORIGINAL_REFERENCE,reference.ORIGINAL_PINS)):
        require(closure(item['root'],item['execution_sha256'],pins,guards) == pins, 'immutable scoring/source closure differs')
    baseline = load_authenticated('_compact_eval_baseline',Path(spec['original_evaluator']['root'])/'evaluate_siglip2_quadratic_readout.py',
        reference.EVALUATOR_PINS['evaluate_siglip2_quadratic_readout.py'],guards)
    helper = load_authenticated('_compact_eval_helper',Path(reference.REFERENCE_ROOT)/'export_siglip2_substrate_adaptation.py',
        reference.REFERENCE_PINS['export_siglip2_substrate_adaptation.py'],guards)
    legacy = t['legacy']; terminal_reader = reference.original_terminal_reader(t['fit_context'])
    final = terminal_reader(legacy['admission'],archived,native.CONCAT_TERMINAL,500,guards)
    for value in (archived['cgroup_before'],archived['cgroup_after'],final):
        helper.zero_events(value)
    for p,h in archived['input_guards'].items():
        bound_file(guards,p,h)
    for name,h in archived['files'].items():
        bound_file(guards,Path(archived['output'])/name,h)
    merge_guards(guards,t['guards'])
    s = {'args':args,'guards':guards,'spec':spec,'selected':legacy,'admission':legacy['admission'],
        'helper':helper,'baseline':baseline,'fit':legacy['prior']['fit'],'partition':legacy['selected']['partition'],
        'origin_records':[archived],'terminals':[native.CONCAT_TERMINAL]}
    require(s['partition'] == read_json(spec['partition'],guards) and
        spec['partition'] == t['fit_context']['launch']['partition'] and
        s['partition']['original_cache']['sha256'] == reference.FIT_SHA, 'original ordered panel partition differs')
    reference.source_selection_adapter(baseline,t['fit_context'])(s)
    source_record = s['source_record']
    require(source_record['source_code'] == launch['genuine_evaluator']['code'] and
        source_record['execution_sha256'] == launch['genuine_evaluator']['execution_sha256'], 'original paired-seed math source differs')
    context = {'args':args,'root':root,'guards':guards,'code':code,'launch':launch,'trainer':trainer,
        'training_context':t,'nearest_evaluator':native,'math':math_helper,'reference':reference,
        'score_context':s,'records':records,'costs':costs,'concat_record':archived,'terminal_reader':terminal_reader,
        'helper':helper,'baseline':baseline,'accepted_units':[],'required_guards':dict(guards),'common_guards':common_guards}
    guard_helpers(context)
    if launch['stage'] == 'full':
        first_receipt = accept_unit(context,launch['first_selection'],'score',stage='first',panel='selection')
        require(first_receipt['decision'] == 'CONTINUE' and
            first_receipt['launch']['endpoints'] == launch['endpoints'][:2], 'eligible original first pair required before second seed')
        context['first_receipt'] = first_receipt
    if launch['panel'] == 'validation':
        selection = accept_unit(context,launch['selection_go'],'score',stage='full',panel='selection')
        require(selection['decision'] == 'GO' and selection['selection_go_admits_validation_only'] is True and
            selection['launch']['endpoints'] == launch['endpoints'], 'same four frozen selection GO endpoints required')
    if args.phase != 'cpu':
        context['cpu'] = accept_unit(context,launch['selected_cpu'],'cpu',panel='selection')
    if args.phase == 'score':
        context['export_records'] = {label(e):accept_unit(context,launch['exports'][label(e)],'export',e['arm'],e['seed'])
                                     for e in launch['endpoints']}
    return context
def check_receipt(context,record,phase,arm=None,seed=None,stage=None,panel=None):
    stage = stage or context['launch']['stage']; panel = panel or context['launch']['panel']
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and record['seed'] == seed and
        record['stage'] == stage and record['panel'] == panel and record['execution_sha256'] == context['args'].execution_sha256 and
        record['source_code'] == context['code'] and record['source'] == context['training_context']['source'] and
        record['cost_policy'] == COST_POLICY and all(record[k] is True for k in
            ('pass','engineering_admission_pass','integrity_pass','resources_pass','exit_rehash_pass',
             'sequential_model_ownership','rng_flags_preserved','both_locks_held_in_parent_authority')) and
        all(record[k] is False for k in ('official_read','global_production_goal_met','public_latency_measured','product_go')),
        'complete source/resource evaluator receipt required')
    check_resource_facts(record,phase)
    rargs = SimpleNamespace(execution_sha256=record['execution_sha256'],authority=Path(record['authority']['path']),
        authority_sha256=record['authority']['sha256'],phase=phase,arm=arm,seed=seed,output=Path(record['output']))
    check_launch(record['launch'],rargs)
    require(read_json(record['authority'],context['guards']) == record['launch'] and
        record['launch']['stage'] == stage and record['launch']['panel'] == panel and
        record['numerical_flags'] == context['training_context']['legacy']['selected']['source_cpu']['numerical_flags'] and
        record['invocation']['cublas_workspace_config'] == ':4096:8' and
        record['authority_sha256'] == record['authority']['sha256'] and
        record['binding'] == binding({'launch':record['launch']}) and record['invocation']['argv'] == cli(rargs) and
        record['invocation']['optimize'] == 0 and record['invocation']['cuda_visible_devices'] == policy(phase)['cuda_visible_devices'],
        'actual launch/CLI/binding differs')
    for k in ('training','nearest_evaluator','genuine_evaluator','reference','cost_policy','selection_previously_exposed'):
        require(record['launch'][k] == context['launch'][k], 'receipt changes frozen procedure: '+k)
    endpoints = context['launch']['endpoints'][:2] if stage == 'first' else context['launch']['endpoints']
    require(record['launch']['endpoints'] == endpoints and
        record['cost'] == {str(s):context['costs'][str(s)] for s in seeds(stage)}, 'receipt changes endpoint/cost binding')
    if stage == context['launch']['stage']:
        require(record['launch']['first_selection'] == context['launch']['first_selection'], 'same eligible first selection required')
        if phase != 'cpu':
            require(record['launch']['selected_cpu'] == context['launch']['selected_cpu'], 'same complete stage CPU required')
    if panel == 'validation':
        require(record['launch']['selection_go'] == context['launch']['selection_go'], 'same sealed selection GO required')
    if phase == 'cpu':
        context['reference'].check_synthetic_bootstrap(record['synthetic_bootstrap'])
        require(record['files'] == {} and record['quality_read'] is False and record['metadata_only'] is True and
            record['updated_payloads_authenticated'] is True and record['malformed_inference_rejected'] is True and
            record['payload_facts'].keys() == {label(e) for e in endpoints} and
            record['calibration']['same_role_forward_exact'] is True and record['calibration']['raw_unit_packed_exact'] is True,
            'CPU full-payload/bundle/synthetic metadata qualification differs')
    elif phase == 'export':
        endpoint = next(e for e in endpoints if (e['seed'],e['arm']) == (seed,arm)); key=label(endpoint)
        require(record['launch']['selected_cpu'] == context['launch']['selected_cpu'] and
            record['files'].keys() == {key+s for s in ('.raw.npy','.unit.npy','.packed.bin')} and
            record['batch_sizes'] == {r:batch_sizes(PANELS[panel][i]) for r,i in (('query',1),('gallery',2))} and
            record['payload_facts'] == context['cpu']['payload_facts'][key] and
            record['inference_state_sha256'] == endpoint['inference_state_sha256'] and
            all(record[k] is True for k in ('strict_independent_reload_exact','full_updated_state_exact',
                'raw_unit_packed_readback_exact','train_native_witness_exact','bundle_dependency_boundary_enforced',
                'same_role_oracle_exact','native_exact_four_post_calibration')),
            'two complete B32 bundle-only export/readback witnesses differ')
        context['reference'].check_value_facts(record['panel_facts'],PANELS[panel][0])
        require(len(record['train_witness']['batch']) == 16, 'actual TRAIN micro16 witness required')
    else:
        decision = decide(context['math'],record['quality'],record['source_quality'],record['concat_quality'],stage,panel,
            record['paired_seed_average_intervals'],record['cost'])
        require(all(record[k] == v for k,v in decision.items()) and
            record['readiness'] == dict.fromkeys(READINESS,True) and record['quality_read'] is True and record['files'] == {} and
            record['bootstrap_seed'] == 179019 and record['bootstrap_draws'] ==
            (5000 if stage == 'full' and decision['immediate_quality_pass'] else 0) and
            all(record[k] is True for k in ('source_archived_perquery_exact','concat_archived_perquery_exact',
                'all_export_wires_readback_before_quality','persisted_wire_scoring_replay_exact')),
            'first/full paired scoring gate/readiness differs')
        if panel == 'selection':
            context['reference'].replay_equal(context['score_context']['source_record']['quality']['179061']['control'],record['source_quality'])
            context['reference'].replay_equal(context['concat_record']['quality']['concat'],record['concat_quality'])


def accept_unit(context,unit,phase,arm=None,seed=None,stage=None,panel=None):
    check_unit(unit)
    record = read_json(unit['receipt'],context['guards'])
    check_receipt(context,record,phase,arm,seed,stage,panel)
    require(Path(unit['receipt']['path']) == Path(record['output'])/'receipt.json', 'complete receipt output role differs')
    t=context['training_context']; legacy=t['legacy']
    final=context['terminal_reader'](legacy['admission'],record,unit,policy(phase)['seconds'],context['guards'])
    for value in (record['cgroup_before'],record['cgroup_after'],final):
        context['helper'].zero_events(value)
    require(unit['invocation_id'] not in legacy['invocations'], 'reused terminal invocation')
    legacy['invocations'].add(unit['invocation_id'])
    for p,h in record['input_guards'].items():
        bound_file(context['guards'],p,h)
    for n,h in record['files'].items():
        bound_file(context['guards'],Path(record['output'])/n,h)
    require(all(record['input_guards'].get(p) == h for p,h in context['common_guards'].items()),
        'foreign complete receipt omits frozen original source guards')
    t['nearest'].native_source_api(t).audit_origins(legacy,initial=True)
    require(record['origins']['packages'] == legacy['selected']['packages'] and
        all(record['input_guards'].get(p) == h for p,h in record['origins']['files'].items()), 'receipt original origins differ')
    if phase == 'export':
        known=set(legacy['selected']['source_cpu']['origins']['files'])|set(legacy['warm_record']['origins']['files'])
        actual=set(record['origins']['files'])-known
        authority=read_json(t['launch']['native_authority'],context['guards'])
        require(authority['proof']['sha256'] == t['nearest'].NATIVE_PROOF_PINS['proof'], 'native proof pin differs')
        proof=read_json(authority['proof'],context['guards'])
        site=Path(proof['authority']['installed_site_root'])
        expected={str(site/name):fact['sha256'] for name,fact in proof['comparison']['selected_members'].items()}
        require(actual == set(expected) and actual <= set(record['origins']['native_files']) and
            all(record['origins']['files'][p] == h for p,h in expected.items()),
            'export receipt observed supplemental membership/bytes must be exact original four')
    prior=legacy['selected']['source_cpu']['invocation']
    require(all(record['invocation'][k] == prior[k] for k in ('python','python_sha256','python_version')),
        'qualified original interpreter differs')
    context['accepted_units'].append(unit)
    return record


def native_start(context):
    args,t=context['args'],context['training_context']; legacy=t['legacy']; source=legacy['source_driver']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == policy(args.phase)['cuda_visible_devices'] and
        os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and sys.flags.optimize == 0 and
        re.fullmatch('[0-9a-f]{32}',os.environ.get('INVOCATION_ID','')), 'deterministic complete-unit environment required')
    prior=legacy['selected']['source_cpu']['invocation']; python=Path(sys.executable).resolve()
    require(str(python) == prior['python'] and bound_file(context['guards'],python,prior['python_sha256']) == python and
        sys.version == prior['python_version'], 'qualified original interpreter required')
    before=source.cgroup_memory(); unit=Path(before['path']).name.removesuffix('.service')
    legacy['selected']['genuine']['reference'].admit_cgroup(before,unit); context['helper'].zero_events(before)
    units=[e['terminal'] for e in context['launch']['endpoints']]+context['accepted_units']+context['score_context']['terminals']
    require(unit not in {u['unit'] for u in units} and os.environ['INVOCATION_ID'] not in
        {u['invocation_id'] for u in units}|legacy['invocations'], 'new distinct complete unit required')
    import torch
    require(not torch.cuda.is_initialized(), 'initial source admission must precede CUDA')
    flags=legacy['selected']['source_cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'source numerical flags differ')
    if args.phase in ('cpu','export') or context['launch']['panel'] == 'validation':
        context['trainer'].prepare_native(t)
    else:
        t['fitter'].prepare_original(t['fit_context']); t['flags']=legacy['flags']
    context['score_context']['packing']=legacy['packing']; context['flags']=flags
    context['trainer'].helper_guard(t); guard_helpers(context)
    merge_guards(context['guards'],t['guards'])
    if args.phase == 'export':
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one admitted CUDA device required')
    else:
        require(not torch.cuda.is_available() and not torch.cuda.is_initialized(), 'CPU phase must hide CUDA')
    return before


def authenticate_payloads(context,endpoint):
    """Read the complete new payload and complete bundle endpoint sequentially."""
    import torch
    trainer,t=context['trainer'],context['training_context']; key=label(endpoint)
    path=bound_file(context['guards'],endpoint['checkpoint']['path'],endpoint['checkpoint']['sha256'])
    disk=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    with path.open('rb') as stream:
        pages=t['legacy']['original'].CheckpointPages(stream); ident=disk['identity']
        trainer.check_payload(t,disk,ident,128)
        require(json_form(ident) == context['records'][endpoint['seed'],endpoint['arm']]['identity'] and
            trainer.fingerprint(t,disk,consumed=pages.consume) == endpoint['terminal_state_sha256'],
            'complete TRAIN128 current bytes/actual identity differ')
        members={k:trainer.fingerprint(t,disk[k]) for k in ('config','buffers','processor','head','A','means')}
        ident=trainer.clone(t,ident)
    del disk,pages; gc.collect()
    directory=Path(endpoint['bundle']['path']).parent
    manifest,guards=trainer.admit_bundle(directory,endpoint['bundle']['sha256']); merge_guards(context['guards'],guards)
    path=directory/'endpoint.pt'; disk=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    require(disk.keys() == trainer.INFERENCE_KEYS and disk['schema'] == trainer.INFERENCE_SCHEMA and
        trainer.fingerprint(t,disk) == manifest['endpoint_state_sha256'] == endpoint['inference_state_sha256'] and
        trainer.fingerprint(t,{k:v for k,v in disk.items() if k != 'fixed_sha256'}) == disk['fixed_sha256'] and
        all(trainer.fingerprint(t,disk[k]) == h for k,h in members.items()) and
        disk['numerical_flags'] == t['flags'], 'bundle substitutes trained A or complete fixed endpoint members')
    require(disk['source']['accepted_A_sha256'] == t['initial_A_sha256'] and
        disk['source']['encoder_checkpoint_sha256'] == t['initial']['provenance']['encoder']['checkpoint']['sha256'] and
        disk['source']['readout_sha256'] == trainer.READOUT['sha256'] and
        manifest['files']['vision.pt'] == disk['source']['encoder_checkpoint_sha256'] and
        manifest['vision_inventory'] == t['initial']['provenance']['encoder']['inventory'], 'bundle original source identity differs')
    t['old'].finite_tree(disk)
    require(disk['schema'] != 'foreign', 'inference schema differs')
    # Explicit malformed full endpoint boundary, without constructing vision or held pixels.
    try:
        check_inference_members(trainer,{**disk,'schema':'foreign'})
    except ValueError:
        pass
    else:
        require(False,'malformed inference accepted')
    facts={'identity':json_form(ident),'members':members,'vision_sha256':disk['vision_sha256'],
        'fixed_sha256':disk['fixed_sha256'],'processor_config_sha256':trainer.fingerprint(t,disk['processor']['config']),
        'terminal_state_sha256':endpoint['terminal_state_sha256'],
        'inference_state_sha256':endpoint['inference_state_sha256'],'bundle':endpoint['bundle']}
    del disk; gc.collect()
    return facts


def check_inference_members(trainer,disk):
    require(disk.keys() == trainer.INFERENCE_KEYS and disk['schema'] == trainer.INFERENCE_SCHEMA,
        'complete compact inference schema differs')


def cpu_calibration(context):
    """Updated bundle readouts versus the accepted helper, synthetic features only."""
    import torch
    from torch.nn import functional as F
    t=context['training_context'];trainer=context['trainer'];legacy=t['legacy'];facts={}
    features=F.normalize(torch.linspace(-1,1,32*1152,dtype=torch.float32).reshape(32,1152),dim=1)
    for endpoint in context['launch']['endpoints']:
        directory=Path(endpoint['bundle']['path']).parent
        manifest,guards=trainer.admit_bundle(directory,endpoint['bundle']['sha256'])
        merge_guards(context['guards'],guards)
        disk=torch.load(directory/'endpoint.pt',map_location='cpu',weights_only=True,mmap=True)
        check_inference_members(trainer,disk)
        name='_compact_cpu_readout_'+label(endpoint).replace('-','_')
        serving=load_authenticated(name,directory/'prototype_residual_readout.py',
            manifest['code']['prototype_residual_readout.py'],context['guards'])
        head=legacy['selected']['cached'].head_from('control',tensors=disk['head']).requires_grad_(False).train()
        A=torch.nn.Parameter(disk['A'].clone())
        with torch.no_grad(),torch.autocast('cpu',enabled=False):
            raw=serving.raw_features(features,head,A,disk['means'],'concat',legacy['quadratic'])
            oracle=trainer.helper_guard(t).raw_features(features.clone(),head,torch.nn.Parameter(A.detach().clone()),
                disk['means'],'concat',legacy['quadratic'])
            first,second=packed_outputs(context,raw),packed_outputs(context,oracle)
            context['helper'].exact(tuple_outputs(first),tuple_outputs(second))
            require(trainer.fingerprint(t,first) == trainer.fingerprint(t,second), 'synthetic updated raw/unit/pack/wire parity differs')
        facts[label(endpoint)]=trainer.fingerprint(t,first)
        require(sys.modules.pop(name,None) is serving, 'synthetic helper registry changed')
        del head,A,raw,oracle,first,second,disk,serving;gc.collect()
    del features
    return {'role':'synthetic CPU FP32 updated readout','rows':32,'same_role_forward_exact':True,
        'raw_unit_packed_exact':True,'outputs_sha256':facts}


def cpu_qualification(context):
    facts={}
    for e in context['launch']['endpoints']:
        first=authenticate_payloads(context,e); second=authenticate_payloads(context,e)
        require(first == second, 'independent full payload/bundle current-byte authentication differs')
        facts[label(e)]=first
    proof=context['reference'].qualify_bootstrap(context['score_context'])
    context['trainer'].helper_guard(context['training_context']); guard_helpers(context)
    return {'payload_facts':facts,'synthetic_bootstrap':proof,'updated_payloads_authenticated':True,
        'malformed_inference_rejected':True,'metadata_only':True,'calibration':cpu_calibration(context),
        'files':{},'quality_read':False}


def tuple_outputs(values):
    return tuple(values[k] for k in ('raw','unit','codes','inverse_norms'))


def packed_outputs(context,raw):
    return context['training_context']['old'].packed_outputs(context['training_context']['legacy'],raw)


def lazy_runtime_files(context,environment):
    """Exact sources from admitted RECORDs; native files stay original CPU origins."""
    sites={Path(v['root']).parent for v in environment['packages'].values()}
    require(len(sites) == 1, 'one qualified runtime site required')
    site=sites.pop()
    require(site.is_absolute() and site.resolve() == site and site.is_dir(), 'canonical runtime site required')
    original=context['training_context']['legacy']['selected']['source_cpu']['origins']
    files={}
    for distribution,sources in RUNTIME_SOURCES.items():
        records=[Path(p) for p in context['required_guards'] if Path(p).parent.parent == site and
            Path(p).name == 'RECORD' and re.fullmatch(re.escape(distribution)+r'-[^/]+\.dist-info',Path(p).parent.name)]
        require(len(records) == 1, 'exact previously admitted runtime RECORD required: '+distribution)
        record=records[0];digest=context['required_guards'][str(record)]
        raw=bound_file(context['guards'],record,digest).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == digest, 'runtime RECORD changed before parsing')
        files[record]=digest;seen=set();wanted=set(sources)
        packages={n.split('/')[0] for n in sources if '/' in n}
        natives={Path(p):original['files'][p] for p in original['native_files'] if
            Path(p).is_relative_to(site) and Path(p).relative_to(site).parts[0] in packages}
        for path,h in natives.items():
            require(context['required_guards'].get(str(path)) == h, 'native runtime original FILE authority differs')
            wanted.add(str(path.relative_to(site)))
        for row in csv.reader(raw.decode('utf-8').splitlines()):
            require(len(row) == 3 and row[0] not in seen, 'invalid/duplicate runtime RECORD row')
            seen.add(row[0])
            if row[0] not in wanted:
                continue
            name,encoded,size=row
            require(re.fullmatch(r'sha256=[A-Za-z0-9_-]{43}',encoded) and
                re.fullmatch(r'0|[1-9][0-9]*',size), 'runtime RECORD hash/size required')
            value=base64.urlsafe_b64decode(encoded[7:]+'=')
            require(base64.urlsafe_b64encode(value).decode().rstrip('=') == encoded[7:], 'noncanonical RECORD hash')
            path=site/name
            require(path not in natives or natives[path] == value.hex(), 'native runtime RECORD differs from original CPU origin')
            bound_file(context['guards'],path,value.hex())
            require(path.stat().st_size == int(size), 'runtime RECORD size differs')
            files[path]=value.hex()
        require(wanted <= seen, 'complete pinned runtime inventory required: '+distribution)
    return files


@contextmanager
def bundle_reads_only(context,endpoint):
    """Independently deny historical file dependencies during copied loading/forward.

One inert-after-use audit hook per bundle avoids rehashing vision weights for
individual image batches. The public loader and exit still reauthenticate all
bundle bytes. Native-origin admission remains the original owned API.
"""
    directory=Path(endpoint['bundle']['path']).parent
    cached=context.setdefault('portable_audits',{})
    identity=(str(directory),endpoint['bundle']['sha256'])
    if identity not in cached:
        manifest,_=context['trainer'].admit_bundle(directory,endpoint['bundle']['sha256'])
        runtime=lazy_runtime_files(context,manifest['environment'])
        sources={p for p in runtime if p.suffix == '.py'}
        site=next(p.parent.parent for p in runtime if p.name == 'RECORD')
        origins={str(p.relative_to(site)).removesuffix('.py').replace('/','.').removesuffix('.__init__'):str(p)
            for p in sources}
        origins.update({str(p.relative_to(site)).split('.',1)[0].replace('/','.'):str(p)
            for p in runtime if p.suffix == '.so'})
        # Imports enumerate these exact directories; no external file roots.
        directories={p.parent for p in sources}|{site}
        bytecode={Path(importlib.util.cache_from_source(str(p))).resolve() for p in sources}
        stdlib=Path(sysconfig.get_path('stdlib')).resolve()
        roots=[directory]
        roots += [Path(v['root']).resolve() for v in manifest['environment']['packages'].values()]
        exact={Path(p).resolve() for p in manifest['environment']['files']}|{p for p in runtime if p.name != 'RECORD'}
        active=[False]
        def audit(event,args):
            if not active[0] or event not in ('open','os.listdir','os.scandir'):
                return
            value=args[0]
            if type(value) is int:
                return
            require(isinstance(value,(str,bytes,os.PathLike)), 'unrecognized serving filesystem access')
            path=Path(os.fsdecode(value)).absolute().resolve()
            if event == 'open':
                mode,flags=args[1:3]
                require((mode is None or not any(c in mode for c in 'wax+')) and
                    not flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND), 'serving loader attempted write')
                if path in bytecode:
                    # SourceFileLoader catches OSError and compiles pinned .py.
                    # A timestamp-valid but unpinned cache must never execute.
                    raise FileNotFoundError('unpinned runtime bytecode: '+str(path))
            require(path in exact or any(path.is_relative_to(p) for p in roots) or
                (path.is_relative_to(stdlib) and not {'site-packages','dist-packages'}.intersection(path.parts)) or
                (event != 'open' and path in directories),
                'bundle-only loader attempted external dependency: '+str(path))
        sys.addaudithook(audit);cached[identity]=(active,runtime,origins)
    active,runtime,origins=cached[identity]
    require(active[0] is False, 'nested serving dependency boundary forbidden')
    original=context['training_context']['legacy']['selected']['source_cpu']['origins']
    for path,digest in runtime.items():
        if path.name == 'RECORD' or path.suffix == '.so':
            require(context['required_guards'].get(str(path)) == digest, 'runtime original FILE authority changed')
        if path.suffix == '.so':
            require(str(path) in original['native_files'] and original['files'].get(str(path)) == digest,
                'runtime original native origin changed')
        bound_file(context['guards'],path,digest)
    for name,module in tuple(sys.modules.items()):
        # The finite contract checks its exact modules. Existing packaging
        # admission also continues to reject every unknown packaging module.
        if name in origins or name.split('.')[0] in ('packaging','regex'):
            require(name in origins and getattr(module,'__file__',None) == origins[name] and
                getattr(getattr(module,'__spec__',None),'origin',None) == origins[name],
                'loaded lazy runtime origin differs: '+name)
    previous=sys.dont_write_bytecode;sys.dont_write_bytecode=True;active[0]=True
    try:
        yield
    finally:
        active[0]=False;sys.dont_write_bytecode=previous


def endpoint_facts(context,state):
    trainer,t=context['trainer'],context['training_context']
    return {'vision_sha256':trainer.fingerprint(t,state['model'].state_dict()),
        'members':{k:trainer.fingerprint(t,state['model'].config.to_dict() if k == 'config' else
            dict(state['model'].named_buffers()) if k == 'buffers' else
            json.loads(state['processor_object'].to_json_string()) if k == 'processor_config' else
            dict(state['head_object'].state_dict()) if k == 'head' else state[k])
            for k in ('config','buffers','processor_config','head','A','means')}}


def images_outputs(context,state,rows,*,oracle=False):
    """Canonical original images; true B32 serving order, no held cache path."""
    import torch
    from PIL import Image
    from torch.nn import functional as F
    trainer,t=context['trainer'],context['training_context']
    require(0<len(rows)<=32, 'actual native B32 image boundary required')
    images=[]; rgb=hashlib.sha256(); rng=torch.random.get_rng_state().clone()
    try:
        for row in rows:
            path=bound_file(context['guards'],row['path'],row['image_sha256'])
            with Image.open(path) as opened:
                image=opened.convert('RGB')
            images.append(image); rgb.update(str(image.size).encode()); rgb.update(image.tobytes())
        portable,endpoint=context['portable_entry']
        with bundle_reads_only(context,endpoint):
            values=portable.inference_outputs(state,images)
        pixels=state['processor_object'](images=images,return_tensors='pt')['pixel_values']
        require(pixels.dtype == torch.float32 and pixels.shape == (len(rows),3,256,256) and
            torch.isfinite(pixels).all().item() and torch.equal(rng,torch.random.get_rng_state()), 'native canonical pixels/RNG differ')
        if oracle:
            with torch.no_grad():
                with torch.autocast('cuda',dtype=torch.float16):
                    pooled=state['model'](pixel_values=pixels.to('cuda')).pooler_output
                with torch.autocast('cuda',enabled=False):
                    features=F.normalize(pooled.float(),dim=1)
                    A=torch.nn.Parameter(state['A'].detach().clone())
                    raw=trainer.helper_guard(t).raw_features(features,state['head_object'],A,state['means'],
                        'concat',t['legacy']['quadratic'])
                    expected=packed_outputs(context,raw)
            context['helper'].exact(tuple_outputs(values),tuple_outputs(expected))
            require(trainer.fingerprint(t,values) == trainer.fingerprint(t,expected), 'same-role original readout/pack/wire differs')
            del pooled,features,A,raw,expected
        fact={'rows':rows,'rgb_sha256':rgb.hexdigest(),'pixels_sha256':trainer.fingerprint(t,pixels),
            'outputs_sha256':trainer.fingerprint(t,values)}
        del pixels
        return values,fact
    finally:
        for image in images:
            image.close()


def train_rows(context,ids):
    t=context['training_context']; legacy=t['legacy']; fit=legacy['prior']['fit']; rows=[]
    for ordinal in ids:
        row,path,_=t['nearest'].canonical_row(t,t['initial'],ordinal)
        rows.append({'train_ordinal':ordinal,'original_row':int(t['initial']['original_rows'][ordinal]),
            'path':str(path),'image_sha256':row['image_sha256']})
    return rows


def native_train_witness(context,state,seed):
    import torch
    from torch.nn import functional as F
    t=context['training_context']; trainer=context['trainer']
    ids=t['initial']['schedules'][str(seed)][0].tolist()[:16]
    rows=train_rows(context,ids); values,fact=images_outputs(context,state,rows,oracle=True)
    with torch.no_grad(),torch.autocast('cpu',enabled=False):
        cache=t['initial']['views']['canonical'][ids]
        head=t['legacy']['selected']['cached'].head_from('control',tensors=t['initial']['head']).requires_grad_(False).train()
        raw=trainer.helper_guard(t).raw_features(cache,head,torch.nn.Parameter(state['A'].detach().cpu().clone()),
            {k:v.cpu() for k,v in state['means'].items()},'concat',t['legacy']['quadratic'])
    difference=values['raw']-raw
    result={'batch':ids,**fact,'cache_native_drift_max_abs':float(difference.abs().max()),
        'cache_native_drift_l2':float(difference.double().norm()),'arithmetic_role':'CUDA FP16 vision / FP32 readout micro16',
        'role_drift_is_diagnostic':True}
    del values,raw,cache,head,difference; gc.collect()
    return result


def diagnostic_triples(context,seed):
    """Freeze first128 valid teacher triples; never remine after an endpoint update."""
    import torch
    t=context['training_context']; initial=t['initial']; teachers=initial['teachers']; triples=[]
    with torch.no_grad():
        for row in initial['schedules'][str(seed)][0].tolist():
            scores=(teachers['V'][row]@teachers['V'].T).tolist()
            p,n=t['nearest'].select_nearest(scores,initial['target'].tolist(),initial['original_rows'].tolist(),row)
            if p>=0:
                triples.append([row,p,n])
    return triples[:128]


def train_diagnostic(context,state,triples):
    import torch
    ids=sorted({i for triple in triples for i in triple}); encoded={}
    with torch.no_grad():
        for start in range(0,len(ids),16):
            batch=ids[start:start+16]
            values,_=images_outputs(context,state,train_rows(context,batch))
            encoded.update({i:values['unit'][j].clone() for j,i in enumerate(batch)})
            del values
    margins=[float((encoded[a]*(encoded[p]-encoded[n])).sum()) for a,p,n in triples]
    require(all(math.isfinite(v) for v in margins), 'nonfinite fixed TRAIN diagnostic')
    return {'triples':triples,'triples_sha256':context['reference'].json_digest(triples),'margins':margins,
        'diagnostic_only':True,'utility_veto':False,'remine':False}


def export_pass(context,state,rows,mapping):
    import torch
    t=context['training_context']; raw=torch.empty((len(rows),128),dtype=torch.float32); unit=torch.empty_like(raw)
    images=[]; sizes={}
    with torch.no_grad():
        for role in ('query','gallery'):
            indices=mapping[role]; sizes[role]=[]
            for start in range(0,len(indices),32):
                batch=indices[start:start+32]
                values,fact=images_outputs(context,state,[rows[i] for i in batch],oracle=start == 0)
                raw[batch]=values['raw']; unit[batch]=values['unit']; fact['role']=role
                images.append(fact); sizes[role].append(len(batch)); del values
    require(sizes == {r:batch_sizes(len(mapping[r])) for r in ('query','gallery')}, 'complete actual B32 roles/tails differ')
    packed=t['legacy']['packing'].pack_int8_unit_embeddings(unit)
    return (raw,unit,packed.codes,packed.inverse_norms),images,sizes


def native_export(context):
    import torch
    trainer,t=context['trainer'],context['training_context']; args=context['args']
    endpoint=next(e for e in context['launch']['endpoints'] if (e['seed'],e['arm']) == (args.seed,args.arm)); key=label(endpoint)
    facts=authenticate_payloads(context,endpoint)
    require(facts == context['cpu']['payload_facts'][key], 'new CPU-qualified complete endpoint differs')
    rows,mapping=context['nearest_evaluator'].image_rows(context,context['launch']['panel'])
    triples=diagnostic_triples(context,args.seed); directory=Path(endpoint['bundle']['path']).parent
    values=None; first_facts=None; first_witness=None; first_diagnostic=None; images=None; sizes=None; files=None
    for pass_index in range(2):
        # TRAIN caches, teacher tensors and warm payloads are forbidden loader dependencies.
        portable_name='_compact_export_entry_'+str(pass_index)+'_'+key.replace('-','_')
        with bundle_reads_only(context,endpoint):
            portable=load_authenticated(portable_name,directory/'train_siglip2_compact_ranking.py',
                context['launch']['training']['code']['train_siglip2_compact_ranking.py'],context['guards'])
            state=portable.load_inference(directory,endpoint['bundle']['sha256'],'cuda')
        context['portable_entry']=(portable,endpoint)
        t['live_model']=weakref.ref(state['model'])
        try:
            model_facts=endpoint_facts(context,state)
            require(model_facts['vision_sha256'] == facts['vision_sha256'] and
                all(model_facts['members'][k] == facts['members'][k] for k in ('config','buffers','head','A','means')),
                'independent complete frozen vision/updated A differs')
            require(model_facts['members']['processor_config'] == facts['processor_config_sha256'],
                'bundle-owned processor config differs')
            witness=native_train_witness(context,state,args.seed)
            diagnostic=train_diagnostic(context,state,triples)
            current,current_images,current_sizes=export_pass(context,state,rows,mapping)
            require(endpoint_facts(context,state) == model_facts, 'native forward mutated complete endpoint')
            t['nearest'].native_source_api(t).audit_origins(t['legacy'],require_exact=True)
            if pass_index == 0:
                values=current; first_facts=model_facts; first_witness=witness; first_diagnostic=diagnostic
                images=current_images; sizes=current_sizes
                files=context['baseline'].write_wires(context['score_context'],key,values)
            else:
                context['helper'].exact(values,current)
                require(trainer.fingerprint(t,values) == trainer.fingerprint(t,current) and
                    model_facts == first_facts and witness == first_witness and diagnostic == first_diagnostic and
                    current_images == images and current_sizes == sizes, 'independent complete B32/TRAIN/parity replay differs')
                context['baseline'].readback_wires(context['score_context'],key,files,current)
                panel_facts=context['baseline'].value_facts(context['score_context'],current)
                del current
        finally:
            portable.release_inference(state);trainer.require_no_training(t)
            require(sys.modules.pop(portable_name,None) is portable, 'portable entry registry changed')
            del context['portable_entry']
    del values; gc.collect()
    return {'payload_facts':facts,'inference_state_sha256':endpoint['inference_state_sha256'],
        'train_witness':first_witness,'panel_facts':panel_facts,'images':images,
        'ordered_images_sha256':context['reference'].json_digest(rows),'batch_sizes':sizes,'diagnostic':first_diagnostic,
        'strict_independent_reload_exact':True,'full_updated_state_exact':True,'raw_unit_packed_readback_exact':True,
        'train_native_witness_exact':True,'bundle_dependency_boundary_enforced':True,'same_role_oracle_exact':True,
        'native_exact_four_post_calibration':True,'files':files,'quality_read':False}
def preserved_validation(context,fixed):
    import torch
    t=context['training_context']; s=context['score_context']; trainer=context['trainer']
    panel=s['partition']['panels']['validation']; cache=s['baseline'].cache_rows(s,panel['original_rows'])
    labels=tuple(s['fit']['class_names'][s['fit']['targets'][r]] for r in panel['original_rows'])
    head=t['legacy']['selected']['cached'].head_from('control',tensors=t['initial']['head']).requires_grad_(False).train()
    with torch.no_grad(),torch.autocast('cpu',enabled=False):
        source=tuple_outputs(packed_outputs(context,head(cache)))
        raw=trainer.helper_guard(t).raw_features(cache,head,torch.nn.Parameter(t['initial']['A'].clone()),
            t['initial']['means'],'concat',t['legacy']['quadratic'])
        concat=tuple_outputs(packed_outputs(context,raw))
    result=[fixed.packed_quality(v[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu')) for v in (source,concat)]
    del head,cache,source,concat,raw; gc.collect()
    return result


def score_exports(context):
    import numpy as np
    import torch
    s=context['score_context']; launch=context['launch']; panel_name=launch['panel']; panel=s['partition']['panels'][panel_name]
    native=context['nearest_evaluator']; fixed=s['baseline'].scoring_math(s)
    # Both archived panels are replayed with exact original per-query arrays first.
    source,concat=native.archived_replay(context,fixed)
    if panel_name == 'validation':
        require(launch['selection_go'] is not None, 'sealed validation requires same-four selection GO')
        source,concat=preserved_validation(context,fixed)
    labels=tuple(s['fit']['class_names'][s['fit']['targets'][r]] for r in panel['original_rows'])
    rows,_=native.image_rows(context,panel_name); held={}; diagnostics={}
    for endpoint in launch['endpoints']:
        key=label(endpoint); record=context['export_records'][key]
        require(record['payload_facts'] == context['cpu']['payload_facts'][key] and
            record['ordered_images_sha256'] == context['reference'].json_digest(rows), 'CPU/bundle/original ordered-image mapping differs')
        values=native.read_wires(context,record['output'],key,record['files'],PANELS[panel_name][0])
        require(s['baseline'].value_facts(s,values) == record['panel_facts'], 'full export wire facts differ before any candidate metric')
        # Independent readback for ALL arms finishes before the first candidate metric.
        repeated=native.read_wires(context,record['output'],key,record['files'],PANELS[panel_name][0])
        context['helper'].exact(values,repeated); held[key]=values; del repeated
    for seed in seeds(launch['stage']):
        c,a=(context['export_records'][arm+'-'+str(seed)]['diagnostic'] for arm in ARMS)
        require(c['triples'] == a['triples'] and c['triples_sha256'] == a['triples_sha256'] and
            all(v['diagnostic_only'] is True and v['utility_veto'] is False and v['remine'] is False for v in (c,a)),
            'fixed TRAIN diagnostic cannot remine or veto utility')
        diagnostics[str(seed)]=diagnostic_result(c['margins'],a['margins'])
    readiness=dict.fromkeys(READINESS,True)
    def compute_quality():
        quality={}
        for endpoint in launch['endpoints']:
            key=label(endpoint); record=context['export_records'][key]; values=held.pop(key)
            first=fixed.packed_quality(values[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu'))
            second=native.read_wires(context,record['output'],key,record['files'],PANELS[panel_name][0])
            context['helper'].exact(values,second)
            replay=fixed.packed_quality(second[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu'))
            context['reference'].replay_equal(first,replay)
            quality.setdefault(str(endpoint['seed']),{})[endpoint['arm']]=first
            del values,second
        return quality
    quality=quality_after_readiness(readiness,compute_quality)
    _,average=context['math'].averaged_deltas(quality,launch['stage'],panel_name)
    immediate=all(immediate_quality_pass(quality[str(seed)],source,concat,PANELS[panel_name][1]) for seed in seeds(launch['stage']))
    intervals={}
    if launch['stage'] == 'full' and immediate:
        # Equal-seed deltas first; original helper resets SAME PCG64 draws5000/179019
        # for every metric and both product/query interval signs.
        intervals=context['reference'].paired_intervals(fixed,average,np.asarray(labels)[panel['query']])
    decision=decide(context['math'],quality,source,concat,launch['stage'],panel_name,intervals,context['costs'])
    return {**decision,'readiness':readiness,'quality':quality,'source_quality':source,'concat_quality':concat,
        'paired_seed_average_intervals':intervals,'diagnostic':diagnostics,'bootstrap_seed':179019,
        'bootstrap_draws':5000 if launch['stage'] == 'full' and immediate else 0,'quality_read':True,'files':{},
        'source_archived_perquery_exact':True,'concat_archived_perquery_exact':True,
        'all_export_wires_readback_before_quality':True,'persisted_wire_scoring_replay_exact':True,
        'updated_descriptors_from_image_encoders':True,'query_images':PANELS[panel_name][1],
        'gallery_images':PANELS[panel_name][2],'products':PANELS[panel_name][3],
        'metric_units':'fractions; multiply by100 for percentage points',
        'interval_scope':'equal-seed paired product/query deltas; shared5000/179019 conditional draws',
        'selection_previously_exposed':True,'validation_quality_exposed':panel_name == 'validation',
        'public_encoder_qualified':False,'optimization_throughput_is_image_training_throughput':False,
        'preparation_costs':{'shared_genuine_export_seconds':283.636,'shared_bundle_preparation_separate':True,
            'shared_qualification_separate':True,'both_cache_target_preparation_charged_to_core':True}}


def resources(context,before):
    import resource
    import torch
    args=context['args']; legacy=context['training_context']['legacy']; source=legacy['source_driver']
    after=source.cgroup_memory(); unit=Path(after['path']).name.removesuffix('.service')
    legacy['selected']['genuine']['reference'].admit_cgroup(after,unit)
    for value in (before,after):
        context['helper'].zero_events(value)
    peak=torch.cuda.max_memory_allocated() if args.phase == 'export' else 0
    wall=time.perf_counter()-UNIT_STARTED; rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    swap=next(v for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('VmSwap:'))
    require(before['path'] == after['path'] and wall<policy(args.phase)['seconds'] and 0<rss<=8*1024**2 and
        int(swap.split()[1]) == 0 and peak<10_000_000_000 and
        (args.phase == 'export' or not torch.cuda.is_initialized()), 'whole-unit resource/zero swap cap differs')
    return {'resource_policy':policy(args.phase),'wall_seconds':wall,'process_peak_rss_kib':rss,
        'peak_cuda_allocated_bytes':peak,'cgroup_before':before,'cgroup_after':after,
        'both_locks_held_in_parent_authority':True,'terminal_exit_and_both_locks_require_parent_receipt':True}


def exit_rehash(context):
    trainer,t=context['trainer'],context['training_context']
    trainer.require_no_training(t); trainer.helper_guard(t); guard_helpers(context)
    api=t['nearest'].native_source_api(t)
    api.audit_origins(t['legacy'],require_exact=context['args'].phase == 'export')
    api.exit_rehash(t['fit_context'])
    merge_guards(context['guards'],t['guards']); merge_guards(context['guards'],t['legacy']['guards'])
    # Fresh uncached complete source/payload/bundle files, including restored-mtime mutations.
    for p,h in context['guards'].items():
        bound_file({},p,h)
    for descriptor,names,pins in (({'root':str(context['root']),'execution_sha256':context['args'].execution_sha256},FILES,context['code']),
        (context['launch']['training'],TRAIN_FILES,context['launch']['training']['code']),
        (context['launch']['nearest_evaluator'],NEAREST_EVALUATOR['code'],NEAREST_EVALUATOR['code']),
        (context['launch']['genuine_evaluator'],GENUINE_PINS,GENUINE_PINS),
        (context['launch']['reference'],context['launch']['reference']['code'],context['launch']['reference']['code'])):
        require(closure(descriptor['root'],descriptor['execution_sha256'],names,{}) == pins,
            'fresh complete source closure changed at exit')
    for endpoint in context['launch']['endpoints']:
        _,guards=trainer.admit_bundle(Path(endpoint['bundle']['path']).parent,endpoint['bundle']['sha256'])
        merge_guards(context['guards'],guards)
    guard_helpers(context)
    api.audit_origins(t['legacy'],require_exact=context['args'].phase == 'export')
    merge_guards(context['guards'],t['legacy']['origins']['files'])
    return t['legacy']['origins']


def run(args):
    require(sys.argv == cli(args), 'fixed canonical CLI order required')
    context=authority(args)
    context['training_context']['fit_context']['unit_started']=UNIT_STARTED
    before=native_start(context)
    import torch
    t=context['training_context']; source=t['legacy']['source_driver']; seed=args.seed or SEEDS[0]
    torch.random.default_generator.manual_seed(seed)
    if args.phase == 'export':
        torch.cuda.manual_seed_all(seed)
    rng=torch.random.get_rng_state().clone(); flags=source.numerical_flags()
    cuda_rng=torch.cuda.get_rng_state_all() if args.phase == 'export' else []
    args.output.mkdir()
    print(json.dumps({'progress':'admitted','phase':args.phase,'stage':context['launch']['stage'],
        'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
    result=cpu_qualification(context) if args.phase == 'cpu' else native_export(context) if args.phase == 'export' else score_exports(context)
    require(torch.equal(rng,torch.random.get_rng_state()) and source.numerical_flags() == flags,
        'whole-unit CPU RNG/numerical flags differ')
    require(args.phase != 'export' or all(torch.equal(a,b) for a,b in
        zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)), 'whole-unit CUDA RNG differs')
    print(json.dumps({'progress':'exit_rehash','seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
    origins=exit_rehash(context); prior=t['legacy']['selected']['source_cpu']['invocation']
    record={'schema':SCHEMA,'phase':args.phase,'arm':args.arm,'seed':args.seed,'stage':context['launch']['stage'],
        'panel':context['launch']['panel'],'binding':binding(context),'source_code':context['code'],
        'execution_sha256':args.execution_sha256,'source':t['source'],'launch':context['launch'],
        'authority':{'path':str(args.authority),'sha256':args.authority_sha256},'authority_sha256':args.authority_sha256,
        'output':str(args.output),'cost':context['costs'],'cost_policy':COST_POLICY,'pass':True,
        'engineering_admission_pass':True,'integrity_pass':True,'resources_pass':True,'exit_rehash_pass':True,
        'sequential_model_ownership':True,'rng_flags_preserved':True,'cuda_initialized':torch.cuda.is_initialized(),
        'official_read':False,'global_production_goal_met':False,'public_latency_measured':False,'product_go':False,
        'numerical_flags':flags,'origins':origins,'input_guards':context['guards'],
        'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),'python_sha256':prior['python_sha256'],
            'python_version':sys.version,'optimize':sys.flags.optimize,'pid':os.getpid(),
            'invocation_id':os.environ['INVOCATION_ID'],'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
            'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')},**result,**resources(context,before)}
    check_receipt(context,record,args.phase,args.arm,args.seed)
    context['helper'].publish(args.output/'receipt.json',record)
    require(time.perf_counter()-UNIT_STARTED<policy(args.phase)['seconds'], 'receipt included whole-unit cap differs')
    return record


def main():
    args=parser().parse_args()
    try:
        result=run(args)
    except (OSError,ValueError,ImportError,KeyError,TypeError,AttributeError,RuntimeError,SyntaxError) as error:
        raise SystemExit('Compact-ranking evaluation rejected: '+str(error)) from error
    print(json.dumps({'schema':SCHEMA,'phase':args.phase,'output':str(args.output),'decision':result.get('decision')}),flush=True)


if __name__ == '__main__':
    main()
