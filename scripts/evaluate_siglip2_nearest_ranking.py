#!/usr/bin/env python3
"""Procedure-owned nearest-ranking CPU qualification, image export and scoring.

Native execution is DGX-root-owned and UNRUN. execution.json owns exactly two
files; launch training owns the separate actual trainer/test/readout closure.
No endpoint hashes are anticipated. Updated image encoders produce every new
selection descriptor. Archived source/concat replay and all wire admissions
precede candidate metrics. A selection GO admits sealed validation only.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import argparse
import ast
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
import time
from types import SimpleNamespace

UNIT_STARTED = time.perf_counter()
SCHEMA = 'siglip2-nearest-ranking-evaluation-v1'
AUTHORITY_SCHEMA = 'siglip2-nearest-ranking-evaluation-launch-v1'
FILES = {'evaluate_siglip2_nearest_ranking.py', 'test_nearest_ranking_evaluation.py'}
TRAIN_FILES = {'train_siglip2_nearest_ranking.py', 'test_siglip2_nearest_ranking.py', 'nearest_ranking_readout.py'}
ARMS = ('control', 'candidate')
METRICS = ('per_query_r1', 'per_query_ap')
PANELS = {'selection': (3449, 1734, 1715, 498), 'validation': (3479, 1749, 1730, 498)}
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
REFERENCE = {'root': '/home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2',
 'execution_sha256': 'c456456c83472526313177fa289ca52228b04f9dff0edf44575933ab45f62970',
 'code': {'evaluate_siglip2_prototype_residual.py': 'e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb',
          'test_siglip2_prototype_residual_evaluation.py': '5c46839ec6a437387d1309dd72167411cfb8a815ceab837ea21dca81aa43dc49'}}
CONCAT_TERMINAL = {'receipt': {'path': '/home/riomus/runs/sfora-so400-signed-concat-evaluation-selection-score-v1/receipt.json',
 'sha256': '74af97389df8ecedde38dcdfbdc44b23c80e471a616ea1bcdcf33cb3c4abf5b5'},
 'log': {'path': '/home/riomus/runs/sfora-so400-signed-concat-evaluation-source-v2/score-selection-v1.log',
         'sha256': '4e12d4a0507571941d10f6ee79a935c448b77bce5e976b8d05893a729aaafcf9'},
 'unit': 'sfora-so400-signed-concat-evaluation-selection-score-v1',
 'invocation_id': 'bcdf75d1a60f45c59ce7ca71e26f87e1',
 'service_seconds': 316.522, 'native_peak_rss_kib': 2902116, 'both_locks_held': True}
# Historical receipt/log and invocation are committed, observed pins.
COST_POLICY = {'whole_service_ratio_max': 1.50, 'total_training_core_ratio_max': 1.50,
 'core': 'target construction + mining + decode/preprocess + forward/backward + optimizer',
 'denominator': 'fresh control32; both complete cold whole units; shared preparation separate'}
LAUNCH_KEYS = {'schema', 'execution_sha256', 'training', 'reference', 'phase', 'arm', 'panel',
 'endpoints', 'selected_cpu', 'exports', 'selection_go', 'resource_policies', 'cost_policy', 'both_locks_held'}
READINESS = ('training_units', 'matched_costs', 'cpu_qualification', 'source_replay',
             'concat_replay', 'updated_state', 'train_witnesses', 'wire_readbacks')


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


def check_endpoint(endpoint):
    require(endpoint.keys() == {'arm','launch','terminal','checkpoint','terminal_state_sha256',
                                'inference','inference_state_sha256'} and endpoint['arm'] in ARMS and
            sha(endpoint['terminal_state_sha256']) and sha(endpoint['inference_state_sha256']), 'complete updated endpoint required')
    for k in ('launch','checkpoint','inference'):
        check_file(endpoint[k])
    check_unit(endpoint['terminal'])
    root = Path(endpoint['terminal']['receipt']['path']).parent
    require(Path(endpoint['checkpoint']['path']) == root/'resume.pt' and
            Path(endpoint['inference']['path']) == root/'inference.pt', 'complete updated endpoint roles differ')


def check_launch(launch,args):
    require(launch.keys() == LAUNCH_KEYS and launch['schema'] == AUTHORITY_SCHEMA and
        launch['execution_sha256'] == args.execution_sha256 and launch['phase'] == args.phase and
        launch['arm'] == args.arm and launch['both_locks_held'] is True and
        launch['resource_policies'] == {p:policy(p) for p in ('cpu','export','score')} and
        launch['cost_policy'] == COST_POLICY and launch['reference'] == REFERENCE and
        launch['panel'] in PANELS, 'fixed evaluation launch differs')
    t = launch['training']
    require(t.keys() == {'root','execution_sha256','code'} and Path(t['root']).is_absolute() and
        sha(t['execution_sha256']) and t['code'].keys() == TRAIN_FILES and all(sha(v) for v in t['code'].values()),
        'actual separate trainer3 inventory required')
    require([e['arm'] for e in launch['endpoints']] == list(ARMS), 'control then candidate required')
    for endpoint in launch['endpoints']:
        check_endpoint(endpoint)
    require(args.phase != 'cpu' or launch['panel'] == 'selection', 'CPU qualification is selection-bound metadata only')
    require((args.phase == 'export') == (args.arm in ARMS), 'one arm only for export')
    require((launch['selected_cpu'] is None) == (args.phase == 'cpu'), 'new evaluator CPU qualification required')
    if launch['selected_cpu'] is not None:
        check_unit(launch['selected_cpu'])
    require(launch['exports'].keys() == (set(ARMS) if args.phase == 'score' else set()), 'paired exports required before score')
    for u in launch['exports'].values():
        check_unit(u)
    require((launch['selection_go'] is None) == (launch['panel'] == 'selection'), 'sealed validation requires selection GO')
    if launch['selection_go'] is not None:
        check_unit(launch['selection_go'])


def batch_sizes(count):
    require(type(count) is int and count>0,'positive image population required')
    return [min(32,count-start) for start in range(0,count,32)]


def paired_cost(records):
    require(records.keys() == set(ARMS),'fresh complete matched training costs required')
    for r in records.values():
        require(all(type(r[k]) in (int,float) and math.isfinite(r[k]) and r[k]>0 for k in
                ('service_seconds','total_training_core_seconds')) and
                r['total_training_core_seconds'] <= r['service_seconds'], 'positive complete training core required')
    ratios = {n:records['candidate'][k]/records['control'][k] for n,k in
              (('whole_service_ratio','service_seconds'),('total_training_core_ratio','total_training_core_seconds'))}
    return {**ratios,'pass':all(v<=1.50 for v in ratios.values()),
            **{a:{k:records[a][k] for k in ('service_seconds','total_training_core_seconds')} for a in ARMS}}


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


def decide(quality,source,concat,count,intervals,costs):
    d = metric_deltas(quality,source,concat,count)
    immediate = immediate_quality_pass(quality,source,concat,count)
    require(intervals.keys() == (set(METRICS) if immediate else set()), 'complete survivor intervals; immediate KILL has none')
    for m,v in intervals.items():
        require(v.keys() == {'mean_delta','product_lower95','product_upper95','query_lower95','query_upper95'} and
            all(type(x) in (int,float) and math.isfinite(x) and -1<=x<=1 for x in v.values()) and
            all(v[k+'_lower95']<=v[k+'_upper95'] for k in ('product','query')) and
            math.isclose(v['mean_delta'],statistics.mean(d['candidate_minus_control'][m]),rel_tol=0,abs_tol=1e-12),
            'finite intervals/mean differ')
    require(costs == paired_cost({a:costs[a] for a in ARMS}), 'matched training cost decision differs')
    passed = immediate and all(v['mean_delta']>=.002 and v['product_lower95']>0 for v in intervals.values())
    return {'decision':'GO' if passed and costs['pass'] else 'KILL','quality_pass':bool(passed),
        'cost_pass':costs['pass'],'immediate_quality_pass':immediate,'deltas':d,
        'source_floor_pass':all(statistics.mean(d['candidate_minus_source'][m])>=0 for m in METRICS),
        'concat_floor_pass':statistics.mean(d['candidate_minus_concat'][METRICS[0]])>0 and
                            statistics.mean(d['candidate_minus_concat'][METRICS[1]])>=0,
        'selection_go_admits_validation_only':bool(passed and costs['pass']),
        'global_production_goal_met':False,'product_go':False}


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


def parser():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--execution-sha256',required=True); p.add_argument('--authority',type=Path,required=True)
    p.add_argument('--authority-sha256',required=True); p.add_argument('--phase',choices=('cpu','export','score'),required=True)
    p.add_argument('--arm',choices=ARMS); p.add_argument('--output',type=Path,required=True)
    return p


def cli(args):
    result = [str(Path(__file__).absolute()),'--execution-sha256',args.execution_sha256,
              '--authority',str(args.authority),'--authority-sha256',args.authority_sha256,'--phase',args.phase]
    if args.arm is not None:
        result += ['--arm',args.arm]
    return result+['--output',str(args.output)]


def binding(context):
    launch = context['launch']
    return {k:launch[k] for k in ('training','reference','panel','endpoints','cost_policy','both_locks_held')}


def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root,args.execution_sha256,FILES,guards)
    launch = read_json({'path':str(args.authority),'sha256':args.authority_sha256},guards)
    check_launch(launch,args)
    output = args.output
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and
            not output.is_symlink(), 'exclusive canonical output required')
    roots = [root,Path(launch['training']['root']),Path(REFERENCE['root'])]
    require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for i,a in enumerate(roots) for b in roots[i+1:]) and
            all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in roots), 'separate evaluator2/trainer3/reference2 required')
    training = launch['training']
    require(closure(training['root'],training['execution_sha256'],TRAIN_FILES,guards) == training['code'] and
            closure(REFERENCE['root'],REFERENCE['execution_sha256'],REFERENCE['code'],guards) == REFERENCE['code'],
            'complete trainer/reference closure differs')
    trainer = load_authenticated('_nearest_eval_trainer',Path(training['root'])/'train_siglip2_nearest_ranking.py',
                                 training['code']['train_siglip2_nearest_ranking.py'],guards)
    reference = load_authenticated('_nearest_eval_reference',Path(REFERENCE['root'])/'evaluate_siglip2_prototype_residual.py',
                                   REFERENCE['code']['evaluate_siglip2_prototype_residual.py'],guards)
    require(trainer.FILES == TRAIN_FILES and trainer.ARMS == ARMS and trainer.SEED == 179061 and
            trainer.SCHEMA == 'siglip2-nearest-ranking-v1' and trainer.INFERENCE_SCHEMA == 'siglip2-nearest-ranking-inference-v1',
            'actual trainer contract differs')
    first = launch['endpoints'][0]
    targs = SimpleNamespace(execution_sha256=training['execution_sha256'],authority=Path(first['launch']['path']),
        authority_sha256=first['launch']['sha256'],phase='train',arm='control',seed=179061,output=output)
    tcontext = trainer.authority(targs)
    records = {}
    for endpoint in launch['endpoints']:
        arm = endpoint['arm']
        el = read_json(endpoint['launch'],guards)
        branch = {**tcontext,'args':SimpleNamespace(**{**vars(targs),'arm':arm,
            'authority':Path(endpoint['launch']['path']),'authority_sha256':endpoint['launch']['sha256']}),
            'launch':el,'guards':dict(tcontext['guards']),'terminals':dict(tcontext['terminals']),
            'terminal_cgroups':dict(tcontext['terminal_cgroups'])}
        trainer.check_launch(el,branch['args'])
        require(el['selected_cpu'] == tcontext['launch']['selected_cpu'] and
                el['selected_mechanics'] == tcontext['launch']['selected_mechanics'], 'same paired mechanics/CPU units required')
        record = trainer.admit_terminal(branch,endpoint['terminal'],'train',arm)
        require(record['authority'] == endpoint['launch'] and record['checkpoint'] == endpoint['checkpoint'] and
                record['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and
                record['inference_checkpoint'] == endpoint['inference'] and
                record['inference_state_sha256'] == endpoint['inference_state_sha256'] and record['completed_step'] == 32,
                'complete updated TRAIN32 endpoint binding differs')
        for k in ('checkpoint','inference'):
            bound_file(branch['guards'],endpoint[k]['path'],endpoint[k]['sha256'])
        records[arm] = {**record,'service_seconds':endpoint['terminal']['service_seconds']}
        merge_guards(tcontext['guards'],branch['guards'])
    c,a = (records[arm] for arm in ARMS)
    require(all(c[k] == a[k] for k in ('source','initial_model_sha256',
            'initial_raw_unit_packed_sha256','numerical_flags')) and
            all(c['identity'][k] == a['identity'][k] for k in ('static_sha256','parameter_names','source','numerical_flags')) and
            [s['batch'] for s in c['steps']] == [s['batch'] for s in a['steps']], 'fresh matched complete initialization/schedule differs')
    costs = paired_cost(records)
    require(costs['pass'], 'fresh matched training cost gate failed before held images/quality')
    archived = read_json(CONCAT_TERMINAL['receipt'],guards)
    require(archived['invocation']['invocation_id'] == CONCAT_TERMINAL['invocation_id'] and
            archived['schema'] == reference.SCHEMA and archived['phase'] == 'score' and archived['pass'] is True and
            archived['source_code'] == REFERENCE['code'] and archived['execution_sha256'] == REFERENCE['execution_sha256'] and
            archived['quality']['concat']['recall_at_1'] == 0.9648212226066898 and
            archived['quality']['concat']['map_at_r'] == 0.8177754035543956 and
            archived['spec']['panel'] == 'selection' and archived['validation_quality_exposed'] is False,
            'preserved accepted concat receipt differs')
    spec = archived['spec']
    for item,names,expected in ((spec['original_evaluator'],reference.EVALUATOR_PINS,reference.EVALUATOR_PINS),
        (reference.EVALUATION_REFERENCE,reference.REFERENCE_PINS,reference.REFERENCE_PINS),
        (reference.ORIGINAL_REFERENCE,reference.ORIGINAL_PINS,reference.ORIGINAL_PINS)):
        require(closure(item['root'],item['execution_sha256'],names,guards) == expected, 'immutable scoring/source closure differs')
    baseline = load_authenticated('_nearest_eval_baseline',Path(spec['original_evaluator']['root'])/'evaluate_siglip2_quadratic_readout.py',
                                  reference.EVALUATOR_PINS['evaluate_siglip2_quadratic_readout.py'],guards)
    helper = load_authenticated('_nearest_eval_helper',Path(reference.REFERENCE_ROOT)/'export_siglip2_substrate_adaptation.py',
                                reference.REFERENCE_PINS['export_siglip2_substrate_adaptation.py'],guards)
    legacy, fit_context = tcontext['legacy'], tcontext['fit_context']
    terminal_reader = reference.original_terminal_reader(fit_context)
    final = terminal_reader(legacy['admission'],archived,CONCAT_TERMINAL,500,guards)
    for value in (archived['cgroup_before'],archived['cgroup_after'],final):
        helper.zero_events(value)
    for p,h in archived['input_guards'].items():
        bound_file(guards,p,h)
    for name,h in archived['files'].items():
        bound_file(guards,Path(archived['output'])/name,h)
    merge_guards(guards,tcontext['guards'])
    score_context = {'args':args,'guards':guards,'spec':spec,'selected':legacy,'admission':legacy['admission'],
        'helper':helper,'baseline':baseline,'fit':legacy['prior']['fit'],'partition':legacy['partition'],
        'origin_records':[archived],'terminals':[CONCAT_TERMINAL]}
    require(score_context['partition'] == read_json(spec['partition'],guards), 'original ordered panel partition differs')
    reference.source_selection_adapter(baseline,fit_context)(score_context)
    context = {'args':args,'root':root,'guards':guards,'code':code,'launch':launch,'trainer':trainer,
        'training_context':tcontext,'reference':reference,'score_context':score_context,'records':records,
        'costs':costs,'concat_record':archived,'terminal_reader':terminal_reader,'helper':helper,
        'phase_seconds':{},'accepted_units':[],'required_guards':dict(guards)}
    if launch['panel'] == 'validation':
        selection = accept_unit(context,launch['selection_go'],'score',panel='selection')
        require(selection['decision'] == 'GO' and selection['selection_go_admits_validation_only'] is True,
                'new selection GO required before sealed validation')
    if args.phase != 'cpu':
        context['cpu'] = accept_unit(context,launch['selected_cpu'],'cpu',panel='selection')
    if args.phase == 'score':
        context['export_records'] = {arm:accept_unit(context,launch['exports'][arm],'export',arm) for arm in ARMS}
    return context


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


def check_receipt(context,record,phase,arm=None,panel=None):
    panel = ('selection' if phase == 'cpu' else context['launch']['panel']) if panel is None else panel
    check_resource_facts(record,phase)
    require(record['schema'] == SCHEMA and record['phase'] == phase and record['arm'] == arm and
        record['panel'] == panel and record['binding'] == {**binding(context),'panel':panel} and
        record['source_code'] == context['code'] and record['execution_sha256'] == context['args'].execution_sha256 and
        record['source'] == context['training_context']['source'] and record['cost'] == context['costs'] and
        record['resource_policy'] == policy(phase) and record['quality_read'] is (phase == 'score') and
        all(record[k] is True for k in ('pass','engineering_admission_pass','integrity_pass','resources_pass',
            'exit_rehash_pass','sequential_model_ownership','both_locks_held_in_parent_authority','rng_flags_preserved')) and
        all(record[k] is False for k in ('official_read','global_production_goal_met','public_latency_measured','product_go')),
        'new evaluator receipt source/resource/role differs')
    require(record['invocation']['optimize'] == 0 and record['invocation']['cuda_visible_devices'] == policy(phase)['cuda_visible_devices'] and
        record['cuda_initialized'] is (phase == 'export') and
        0<=record['peak_cuda_allocated_bytes']<10_000_000_000 and
        (phase == 'export' or record['peak_cuda_allocated_bytes'] == 0), 'original numerical/resource role differs')
    require(read_json(record['authority'],context['guards']) == record['launch'] and
        record['authority']['sha256'] == record['authority_sha256'] and
        (phase == 'cpu' or panel != context['launch']['panel'] or
         record['launch']['selected_cpu'] == context['launch']['selected_cpu']), 'receipt launch/actual CPU binding differs')
    rargs = SimpleNamespace(execution_sha256=record['execution_sha256'],authority=Path(record['authority']['path']),
        authority_sha256=record['authority']['sha256'],phase=phase,arm=arm,output=Path(record['output']))
    check_launch(record['launch'],rargs)
    require(record['invocation']['argv'] == cli(rargs),'original evaluator CLI differs')
    if phase == 'cpu':
        context['reference'].check_synthetic_bootstrap(record['synthetic_bootstrap'])
        require(record['files'] == {} and record['updated_payloads_authenticated'] is True and
                record['malformed_inference_rejected'] is True and record['metadata_only'] is True and
                record['calibration']['same_role_forward_exact'] is True and
                record['calibration']['raw_unit_packed_exact'] is True,
                'CPU qualification cannot expose held images/new quality')
    elif phase == 'export':
        require(record['files'].keys() == {arm+s for s in ('.raw.npy','.unit.npy','.packed.bin')} and
            record['batch_sizes'] == {r:batch_sizes(PANELS[panel][i]) for r,i in (('query',1),('gallery',2))} and
            all(record[k] is True for k in ('strict_independent_reload_exact','full_updated_state_exact',
                'raw_unit_packed_readback_exact','train_native_witness_exact')) and
            record['inference_state_sha256'] == context['launch']['endpoints'][ARMS.index(arm)]['inference_state_sha256'],
            'complete updated native export/witness differs')
        context['reference'].check_value_facts(record['panel_facts'],PANELS[panel][0])
    else:
        for q in (record['source_quality'],record['concat_quality'],*record['quality'].values()):
            check_quality(q,PANELS[panel][1])
        if panel == 'selection':
            context['reference'].replay_equal(context['score_context']['source_record']['quality']['179061']['control'],
                                               record['source_quality'])
            context['reference'].replay_equal(context['concat_record']['quality']['concat'],record['concat_quality'])
        decision = decide(record['quality'],record['source_quality'],record['concat_quality'],PANELS[panel][1],
                          record['paired_intervals'],record['cost'])
        decision['selection_go_admits_validation_only'] &= panel == 'selection'
        require(all(record[k] == v for k,v in decision.items()) and
            record['readiness'] == dict.fromkeys(READINESS,True) and
            record['bootstrap_seed'] == 179019 and record['bootstrap_draws'] == (5000 if decision['immediate_quality_pass'] else 0),
            'complete original quality gate differs')


def accept_unit(context,unit,phase,arm=None,panel=None):
    check_unit(unit)
    record = read_json(unit['receipt'],context['guards'])
    check_receipt(context,record,phase,arm,panel)
    require(Path(unit['receipt']['path']) == Path(record['output'])/'receipt.json', 'receipt role differs')
    t = context['training_context']; legacy = t['legacy']
    final = context['terminal_reader'](legacy['admission'],record,unit,policy(phase)['seconds'],context['guards'])
    for v in (record['cgroup_before'],record['cgroup_after'],final):
        context['helper'].zero_events(v)
    require(unit['invocation_id'] not in legacy['invocations'],'reused original UNIT')
    legacy['invocations'].add(unit['invocation_id'])
    expected = {p:h for p,h in context['required_guards'].items() if p != str(context['args'].authority)}
    require(all(record['input_guards'].get(p) == h for p,h in expected.items()),'foreign CPU/export omits actual original guards')
    t['old'].audit_origins(legacy,initial=True)
    require(record['origins']['packages'] == legacy['selected']['packages'] and
        all(record['input_guards'].get(p) == h for p,h in record['origins']['files'].items()),
        'actual admitted original origin files differ')
    prior = legacy['selected']['source_cpu']['invocation']
    require(all(record['invocation'][k] == prior[k] for k in ('python','python_sha256','python_version')),
            'qualified evaluator interpreter differs')
    for p,h in record['input_guards'].items():
        bound_file(context['guards'],p,h)
    for n,h in record['files'].items():
        bound_file(context['guards'],Path(record['output'])/n,h)
    context['accepted_units'].append(unit)
    return record


def json_form(value):
    return json.loads(json.dumps(value,sort_keys=True,allow_nan=False))


def native_start(context):
    args,t = context['args'],context['training_context']
    legacy,source = t['legacy'],t['legacy']['source_driver']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == policy(args.phase)['cuda_visible_devices'] and
            os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and sys.flags.optimize == 0 and
            re.fullmatch('[0-9a-f]{32}',os.environ.get('INVOCATION_ID','')), 'original deterministic unit environment required')
    prior = legacy['selected']['source_cpu']['invocation']
    python = Path(sys.executable).resolve()
    require(str(python) == prior['python'] and bound_file(context['guards'],python,prior['python_sha256']) == python and
            sys.version == prior['python_version'], 'qualified original interpreter differs')
    before = source.cgroup_memory(); unit = Path(before['path']).name.removesuffix('.service')
    legacy['selected']['genuine']['reference'].admit_cgroup(before,unit)
    context['helper'].zero_events(before)
    require(os.environ['INVOCATION_ID'] not in legacy['invocations'] and
            unit not in {e['terminal']['unit'] for e in context['launch']['endpoints']} and
            unit not in {u['unit'] for u in context['score_context']['terminals']} and
            os.environ['INVOCATION_ID'] not in {u['invocation_id'] for u in context['score_context']['terminals']},
            'distinct original evaluator unit required')
    import torch
    require(not torch.cuda.is_initialized(), 'native admission must precede CUDA')
    flags = legacy['selected']['source_cpu']['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags, 'source numerical flags differ')
    context['trainer'].helper_guard(t)
    if args.phase in ('cpu','export') or context['launch']['panel'] == 'validation':
        context['trainer'].prepare_native(t)
    else:
        # Scoring does not construct a vision or read new TRAIN teachers.
        t['fitter'].prepare_original(t['fit_context']); t['flags'] = legacy['flags']
    context['score_context']['packing'] = legacy['packing']
    context['flags'] = flags
    merge_guards(context['guards'],t['guards'])
    if args.phase != 'export':
        require(not torch.cuda.is_initialized() and not torch.cuda.is_available(), 'CPU unit must hide CUDA')
    else:
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'one authenticated DGX CUDA device required')
    return before


def authenticate_payloads(context,endpoint):
    """Consume complete updated resume then inference, never simultaneous archives."""
    import torch
    trainer,t = context['trainer'],context['training_context']
    arm = endpoint['arm']; original = t['legacy']['original']
    path = bound_file(context['guards'],endpoint['checkpoint']['path'],endpoint['checkpoint']['sha256'])
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    with path.open('rb') as stream:
        pages = original.CheckpointPages(stream)
        ident = disk['identity']
        trainer.check_payload(t,disk,ident,32)
        require(json_form(ident) == context['records'][arm]['identity'] and
            trainer.fingerprint(t,disk,consumed=pages.consume) == endpoint['terminal_state_sha256'],
            'complete updated training payload/actual identity differs')
        members = {k:trainer.fingerprint(t,disk[k]) for k in
                   ('vision','config','buffers','head','classifier','A','means','processor')}
        require(all(trainer.fingerprint(t,disk[k]) == trainer.fingerprint(t,t['initial'][k])
            for k in ('teachers','panel','schedule','original_rows','target','partition')),
            'saved immutable teachers/diagnostic/original rows differ from accepted construction')
        ident = trainer.clone(t,ident)
    del disk,pages
    gc.collect()
    path = bound_file(context['guards'],endpoint['inference']['path'],endpoint['inference']['sha256'])
    disk = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    with path.open('rb') as stream:
        pages = original.CheckpointPages(stream)
        trainer.check_inference(t,disk,endpoint['inference_state_sha256'])
        require(trainer.fingerprint(t,disk,consumed=pages.consume) == endpoint['inference_state_sha256'] and
            disk['identity'] == ident and all(trainer.fingerprint(t,disk[k]) == v for k,v in members.items()),
            'inference substitutes original/foreign vision or fixed members')
        facts = {'identity':json_form(ident),'vision_sha256':disk['vision_sha256'],
                 'fixed_sha256':disk['fixed_sha256'],'terminal_state_sha256':endpoint['terminal_state_sha256'],
                 'inference_state_sha256':endpoint['inference_state_sha256']}
        malformed = {**disk,'schema':'foreign'}
        try:
            trainer.check_inference(t,malformed,endpoint['inference_state_sha256'])
        except ValueError:
            pass
        else:
            require(False,'malformed inference schema accepted')
        del malformed
    del disk,pages
    gc.collect()
    return facts,ident


def cpu_calibration(context):
    """Same-role synthetic FP32 readout/packing oracle; no image or quality read."""
    import torch
    trainer,t = context['trainer'],context['training_context']
    head = t['legacy']['selected']['cached'].head_from('control',tensors=t['initial']['head']).requires_grad_(False).train()
    features = torch.linspace(-1,1,32*1152,dtype=torch.float32).reshape(32,1152)
    A = t['initial']['A']; isolated = torch.nn.Parameter(A.clone(),requires_grad=True)
    with torch.no_grad(),torch.autocast('cpu',enabled=False):
        raw = t['readout'].raw_features(features,head,A,t['initial']['means'],t['legacy']['quadratic'])
        oracle = t['fitter'].prepare_readout(t['fit_context']).raw_features(
            features,head,isolated,t['initial']['means'],'concat',t['legacy']['quadratic'])
        context['helper'].exact((raw,),(oracle,))
        first,second = trainer.packed_outputs(t,raw),trainer.packed_outputs(t,oracle)
        require(trainer.fingerprint(t,first) == trainer.fingerprint(t,second),'synthetic CPU raw/unit/packed parity differs')
    result = {'role':'synthetic FP32 readout only','rows':32,'same_role_forward_exact':True,
              'raw_unit_packed_exact':True,'outputs_sha256':trainer.fingerprint(t,first)}
    del features,head,isolated,raw,oracle,first,second
    gc.collect()
    return result


def cpu_qualification(context):
    t = context['training_context']; trainer = context['trainer']
    facts = {}
    for endpoint in context['launch']['endpoints']:
        first,_ = authenticate_payloads(context,endpoint)
        second,_ = authenticate_payloads(context,endpoint)
        require(first == second,'independent full updated payload authentication differs')
        facts[endpoint['arm']] = first
    # Frozen role cardinalities, not candidate metrics or held image pixels.
    bootstrap = context['reference'].qualify_bootstrap(context['score_context'])
    trainer.helper_guard(t)
    return {'payload_facts':facts,'synthetic_bootstrap':bootstrap,'updated_payloads_authenticated':True,
        'malformed_inference_rejected':True,'metadata_only':True,'calibration':cpu_calibration(context),
        'files':{},'quality_read':False}


def endpoint_for(context,arm):
    return context['launch']['endpoints'][ARMS.index(arm)]


def updated_inference_facts(context,state):
    trainer,t = context['trainer'],context['training_context']
    return {'vision_sha256':trainer.fingerprint(t,state['model'].state_dict()),
        'fixed_members_sha256':trainer.fingerprint(t,{k: state['model'].config.to_dict() if k == 'config' else
            trainer.current_buffers(state) if k == 'buffers' else dict(state['head_object'].state_dict()) if k == 'head'
            else state[k] for k in ('config','buffers','head','classifier','A','means','processor')})}


def image_rows(context,panel):
    t = context['training_context']; legacy = t['legacy']; fit = legacy['prior']['fit']
    mapping = context['score_context']['partition']['panels'][panel]
    require(tuple(map(len,(mapping['original_rows'],mapping['query'],mapping['gallery'],mapping['original_class_ids']))) == PANELS[panel],
            'complete original query/gallery/product population required')
    q,g = mapping['query'],mapping['gallery']
    require(len(set(q)) == len(q) and len(set(g)) == len(g) and not set(q)&set(g) and
            set(q)|set(g) == set(range(len(mapping['original_rows']))), 'complete ordered disjoint image roles required')
    rows=[]
    for i,r in enumerate(mapping['original_rows']):
        row = fit['rows'][r]; path = Path(legacy['prior']['all_images'][r])
        require((Path(fit['dataset_root'])/row['relative_path']).resolve() == path and
                path.is_relative_to(Path(fit['dataset_root'])) and
                row['product'] == fit['class_names'][fit['targets'][r]], 'original panel image/path/product mapping differs')
        bound_file(context['guards'],path,row['image_sha256'])
        rows.append({'panel_ordinal':i,'original_row':r,'role':'query' if i in set(q) else 'gallery',
            'path':str(path),'relative_path':row['relative_path'],'image_sha256':row['image_sha256'],
            'train_row':row['train_row'],'product':row['product'],'target':fit['targets'][r]})
    return rows,mapping


def image_pixels(context,state,rows):
    import torch
    from PIL import Image
    require(0<len(rows)<=32,'native B32 image boundary required')
    images=[]; rgb=hashlib.sha256(); rng=torch.random.get_rng_state().clone()
    try:
        for row in rows:
            path=bound_file(context['guards'],row['path'],row['image_sha256'])
            with Image.open(path) as opened:
                image=opened.convert('RGB')
            images.append(image); rgb.update(str(image.size).encode()); rgb.update(image.tobytes())
        pixels=state['processor_object'](images=images,return_tensors='pt')['pixel_values']
    finally:
        for image in images:
            image.close()
    require(pixels.dtype == torch.float32 and pixels.shape == (len(rows),3,256,256) and
        not pixels.requires_grad and pixels.grad_fn is None and torch.isfinite(pixels).all().item() and
        torch.equal(rng,torch.random.get_rng_state()), 'pinned canonical pixels/RNG differ')
    return pixels,{'rows':rows,'rgb_sha256':rgb.hexdigest(),
                   'pixels_sha256':context['trainer'].fingerprint(context['training_context'],pixels)}


def tuple_outputs(values):
    return tuple(values[k] for k in ('raw','unit','codes','inverse_norms'))


def native_train_witness(context,state):
    """Same scheduled actual TRAIN micro16; no cache-based encoder evaluation."""
    import torch
    trainer,t = context['trainer'],context['training_context']
    mapped = {**state,**{k:t['initial'][k] for k in ('original_rows','target','partition')}}
    batch=t['initial']['schedule'][0].tolist()[:16]
    pixels,rgb,mapping=trainer.canonical_pixels(t,mapped,batch)
    with torch.no_grad():
        raw=trainer.native_raw(t,state,pixels,oracle=True)
        outputs=trainer.packed_outputs(t,raw)
    result={'batch':batch,'rgb_sha256':rgb,'row_mapping_sha256':mapping,
            'pixels_sha256':trainer.fingerprint(t,pixels),'outputs_sha256':trainer.fingerprint(t,outputs),
            'facts':context['score_context']['baseline'].value_facts(context['score_context'],tuple_outputs(outputs))}
    del mapped,pixels,raw,outputs
    return result


def train_diagnostic(context,state):
    import torch
    trainer,t = context['trainer'],context['training_context']
    triples=t['initial']['panel'].tolist()
    mapped = {**state,**{k:t['initial'][k] for k in ('original_rows','target','partition')}}
    ids=sorted({i for triple in triples for i in triple}); encoded={}
    with torch.no_grad():
        for start in range(0,len(ids),16):
            batch=ids[start:start+16]; pixels,_,_=trainer.canonical_pixels(t,mapped,batch)
            raw=trainer.native_raw(t,state,pixels)
            values=trainer.packed_outputs(t,raw)['unit']
            encoded.update({i:values[j].clone() for j,i in enumerate(batch)})
            del pixels,raw,values
    margins=[float((encoded[a]*(encoded[p]-encoded[n])).sum()) for a,p,n in triples]
    require(all(math.isfinite(v) for v in margins),'nonfinite fixed diagnostic margin')
    return {'triples':triples,'triples_sha256':trainer.fingerprint(t,t['initial']['panel']),
            'margins':margins,'diagnostic_only':True,'utility_veto':False,'remine':False}


def export_pass(context,state,rows,mapping):
    import torch
    trainer,t = context['trainer'],context['training_context']
    raw=torch.empty((len(rows),128),dtype=torch.float32); unit=torch.empty_like(raw)
    image_facts=[]; sizes={}
    with torch.no_grad():
        for role in ('query','gallery'):
            indices=mapping[role]; sizes[role]=[]
            for start in range(0,len(indices),32):
                batch=indices[start:start+32]; selected=[rows[i] for i in batch]
                pixels,fact=image_pixels(context,state,selected)
                outputs=trainer.packed_outputs(t,trainer.native_raw(t,state,pixels,oracle=start == 0))
                raw[batch]=outputs['raw']; unit[batch]=outputs['unit']
                fact['role']=role; fact['outputs_sha256']=trainer.fingerprint(t,outputs); image_facts.append(fact)
                sizes[role].append(len(batch))
                del pixels,outputs
    require(sizes == {r:batch_sizes(len(mapping[r])) for r in ('query','gallery')},'exact B32/tail roles differ')
    packed=t['legacy']['packing'].pack_int8_unit_embeddings(unit)
    return (raw,unit,packed.codes,packed.inverse_norms),image_facts,sizes


def native_export(context):
    import torch
    trainer,t = context['trainer'],context['training_context']; arm=context['args'].arm
    endpoint=endpoint_for(context,arm)
    facts,_=authenticate_payloads(context,endpoint)
    require(facts == context['cpu']['payload_facts'][arm],'CPU-qualified updated endpoint differs')
    # Full saved TRAIN ownership was consumed above and cross-bound to inference.
    # Construct directly from the updated artifact, avoiding an original vision copy.
    first=trainer.load_inference(t,endpoint['inference']['path'],endpoint['inference']['sha256'],
                                 endpoint['inference_state_sha256'],'cuda')
    model_facts=updated_inference_facts(context,first)
    require(model_facts['vision_sha256'] == facts['vision_sha256'],'updated inference vision differs')
    witness=native_train_witness(context,first)
    diagnostic=train_diagnostic(context,first)
    rows,mapping=image_rows(context,context['launch']['panel'])
    values,images,sizes=export_pass(context,first,rows,mapping)
    require(updated_inference_facts(context,first) == model_facts,'native images mutated updated complete state')
    files=context['score_context']['baseline'].write_wires(context['score_context'],arm,values)
    trainer.release(t,first)
    second=trainer.load_inference(t,endpoint['inference']['path'],endpoint['inference']['sha256'],
                                  endpoint['inference_state_sha256'],'cuda')
    require(updated_inference_facts(context,second) == model_facts and native_train_witness(context,second) == witness,
            'independent updated state/TRAIN witness differs')
    require(train_diagnostic(context,second) == diagnostic,'independent fixed TRAIN diagnostic differs')
    repeated,second_images,second_sizes=export_pass(context,second,rows,mapping)
    context['helper'].exact(values,repeated)
    require(trainer.fingerprint(t,values) == trainer.fingerprint(t,repeated) and images == second_images and sizes == second_sizes,
            'independent full updated image/pixel/raw/unit/packed export differs')
    context['score_context']['baseline'].readback_wires(context['score_context'],arm,files,repeated)
    panel_facts=context['score_context']['baseline'].value_facts(context['score_context'],repeated)
    require(updated_inference_facts(context,second) == model_facts,'second export mutated updated state')
    trainer.release(t,second)
    del values,repeated
    gc.collect()
    return {'payload_facts':facts,'updated_vision_sha256':facts['vision_sha256'],
        'inference_state_sha256':endpoint['inference_state_sha256'],'train_witness':witness,
        'panel_facts':panel_facts,'images':images,'ordered_images_sha256':context['reference'].json_digest(rows),
        'batch_sizes':sizes,'diagnostic':diagnostic,'strict_independent_reload_exact':True,
        'full_updated_state_exact':True,'raw_unit_packed_readback_exact':True,'train_native_witness_exact':True,
        'files':files,'quality_read':False}


def read_wires(context,root,key,files,count):
    """Read saved wires independently; scored packing equals actual saved bytes."""
    import numpy as np
    import torch
    s=context['score_context']; arrays=[]
    for suffix in ('.raw.npy','.unit.npy'):
        name=key+suffix; path=bound_file(context['guards'],Path(root)/name,files[name])
        array=np.load(path,allow_pickle=False)
        require(array.dtype == np.float32 and array.shape == (count,128) and np.isfinite(array).all() and
                np.all(np.linalg.norm(array,axis=1)>0),'complete finite raw/unit wire layout differs')
        arrays.append(torch.from_numpy(array))
    require(np.allclose(np.linalg.norm(arrays[1].numpy(),axis=1),1,atol=1e-5,rtol=0),'saved unit row norms differ')
    packed=s['packing'].pack_int8_unit_embeddings(arrays[1])
    wire=bound_file(context['guards'],Path(root)/(key+'.packed.bin'),files[key+'.packed.bin']).read_bytes()
    require(wire == packed.to_bytes(),'actual packed codes/inverse bits differ')
    return *arrays,packed.codes,packed.inverse_norms


def archived_replay(context,fixed):
    import torch
    s=context['score_context']; baseline=s['baseline']; archived=context['concat_record']
    source=baseline.replay_archived_source(s,fixed)
    concat=read_wires(context,archived['output'],'concat',archived['files'],3449)
    require(baseline.value_facts(s,concat) == archived['panel_replay_facts']['concat'],
            'preserved concat full wire facts differ')
    panel=s['partition']['panels']['selection']
    labels=tuple(s['fit']['class_names'][s['fit']['targets'][r]] for r in panel['original_rows'])
    cq=fixed.packed_quality(concat[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu'))
    context['reference'].replay_equal(archived['quality']['concat'],cq)
    check_quality(cq,1734)
    return source,cq


def preserved_validation(context,fixed):
    """Frozen original cached source/concat floors only; updated arms use wires."""
    import torch
    t=context['training_context']; trainer=context['trainer']; s=context['score_context']
    panel=s['partition']['panels']['validation']; cache=s['baseline'].cache_rows(s,panel['original_rows'])
    labels=tuple(s['fit']['class_names'][s['fit']['targets'][r]] for r in panel['original_rows'])
    head=t['legacy']['selected']['cached'].head_from('control',tensors=t['initial']['head']).requires_grad_(False).train()
    with torch.no_grad(),torch.autocast('cpu',enabled=False):
        source=tuple_outputs(trainer.packed_outputs(t,head(cache)))
        raw=t['readout'].raw_features(cache,head,t['initial']['A'],
                                                t['initial']['means'],t['legacy']['quadratic'])
        concat=tuple_outputs(trainer.packed_outputs(t,raw))
    result=[fixed.packed_quality(v[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu'))
            for v in (source,concat)]
    del head,cache,source,concat,raw
    gc.collect()
    return result


def score_exports(context):
    import numpy as np
    import torch
    s=context['score_context']; panel_name=context['launch']['panel']; panel=s['partition']['panels'][panel_name]
    fixed=s['baseline'].scoring_math(s)
    source,concat=archived_replay(context,fixed)
    if panel_name == 'validation':
        require(context['launch']['selection_go'] is not None,'sealed validation requires authenticated selection GO')
        source,concat=preserved_validation(context,fixed)
    labels=tuple(s['fit']['class_names'][s['fit']['targets'][r]] for r in panel['original_rows'])
    held={}; exports=context['export_records']; rows,_=image_rows(context,panel_name)
    for arm in ARMS:
        record=exports[arm]
        require(record['payload_facts'] == context['cpu']['payload_facts'][arm] and
            record['ordered_images_sha256'] == context['reference'].json_digest(rows),
            'foreign CPU/model/original ordered-image mapping differs')
        values=read_wires(context,record['output'],arm,record['files'],PANELS[panel_name][0])
        require(s['baseline'].value_facts(s,values) == record['panel_facts'],'export wire facts differ before quality')
        held[arm]=values
    c,a=(exports[arm]['diagnostic'] for arm in ARMS)
    require(c['triples'] == a['triples'] and c['triples_sha256'] == a['triples_sha256'] and
        all(d['diagnostic_only'] is True and d['utility_veto'] is False and d['remine'] is False for d in (c,a)),
        'fixed immutable no-remine TRAIN diagnostic differs')
    diagnostic=diagnostic_result(c['margins'],a['margins'])
    readiness=dict.fromkeys(READINESS,True)
    def compute_quality():
        quality={}
        for arm in ARMS:
            record=exports[arm]; values=held.pop(arm)
            first=fixed.packed_quality(values[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu'))
            second=read_wires(context,record['output'],arm,record['files'],PANELS[panel_name][0])
            context['helper'].exact(values,second)
            replay=fixed.packed_quality(second[1].numpy(),labels,panel['query'],panel['gallery'],device=torch.device('cpu'))
            context['reference'].replay_equal(first,replay); quality[arm]=first
            del values,second
        return quality
    quality=quality_after_readiness(readiness,compute_quality)
    d=metric_deltas(quality,source,concat,PANELS[panel_name][1])
    immediate=immediate_quality_pass(quality,source,concat,PANELS[panel_name][1])
    intervals=context['reference'].paired_intervals(fixed,d['candidate_minus_control'],np.asarray(labels)[panel['query']]) if immediate else {}
    decision=decide(quality,source,concat,PANELS[panel_name][1],intervals,context['costs'])
    decision['selection_go_admits_validation_only'] &= panel_name == 'selection'
    return {**decision,'readiness':readiness,'quality':quality,'source_quality':source,'concat_quality':concat,
        'paired_intervals':intervals,'diagnostic':diagnostic,'bootstrap_seed':179019,
        'bootstrap_draws':5000 if immediate else 0,'quality_read':True,'files':{},
        'source_archived_perquery_exact':True,'concat_archived_perquery_exact':True,
        'all_export_wires_readback_before_quality':True,'persisted_wire_scoring_replay_exact':True,
        'updated_descriptors_from_image_encoders':True,'query_images':PANELS[panel_name][1],
        'gallery_images':PANELS[panel_name][2],'products':PANELS[panel_name][3],
        'metric_units':'fractions; multiply by100 for percentage points',
        'interval_scope':'conditional paired product/query intervals; one frozen source',
        'validation_quality_exposed':panel_name == 'validation','public_encoder_qualified':False}


def resources(context,before):
    import resource
    import torch
    args=context['args']; t=context['training_context']; legacy=t['legacy']; source=legacy['source_driver']
    after=source.cgroup_memory(); unit=Path(after['path']).name.removesuffix('.service')
    legacy['selected']['genuine']['reference'].admit_cgroup(after,unit)
    for value in (before,after):
        context['helper'].zero_events(value)
    peak=torch.cuda.max_memory_allocated() if args.phase == 'export' else 0
    wall=time.perf_counter()-UNIT_STARTED; rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    swap=next(v for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('VmSwap:'))
    require(before['path'] == after['path'] and wall<policy(args.phase)['seconds'] and
        0<rss<=8*1024**2 and int(swap.split()[1]) == 0 and peak<10_000_000_000 and
        (args.phase == 'export' or not torch.cuda.is_initialized()),'whole-unit resource/zero swap cap differs')
    return {'resource_policy':policy(args.phase),'wall_seconds':wall,'process_peak_rss_kib':rss,
        'peak_cuda_allocated_bytes':peak,'cgroup_before':before,'cgroup_after':after,
        'both_locks_held_in_parent_authority':True,'terminal_exit_and_both_locks_require_parent_receipt':True}


def exit_rehash(context):
    trainer,t=context['trainer'],context['training_context']
    trainer.require_no_model(t); trainer.exit_rehash(t)
    merge_guards(context['guards'],t['guards'])
    merge_guards(context['guards'],t['legacy']['guards'])
    # Empty exit-owned map: full current bytes, no startup hash/stat cache.
    for p,h in context['guards'].items():
        bound_file({},p,h)
    for descriptor,names,code in (({'root':str(context['root']),'execution_sha256':context['args'].execution_sha256},FILES,context['code']),
        (context['launch']['training'],TRAIN_FILES,context['launch']['training']['code']),
        (REFERENCE,REFERENCE['code'],REFERENCE['code'])):
        require(closure(descriptor['root'],descriptor['execution_sha256'],names,{}) == code,
                'fresh complete source closure changed at exit')
    trainer.helper_guard(t)
    trainer.native_source_api(t).audit_origins(t['legacy'])
    return t['legacy']['origins']


def run(args):
    require(sys.argv == cli(args),'fixed canonical CLI order required')
    context=authority(args)
    before=native_start(context)
    import torch
    t=context['training_context']; source=t['legacy']['source_driver']
    torch.random.default_generator.manual_seed(179061)
    if args.phase == 'export':
        torch.cuda.manual_seed_all(179061)
    rng=torch.random.get_rng_state().clone(); flags=source.numerical_flags()
    cuda_rng=torch.cuda.get_rng_state_all() if args.phase == 'export' else []
    args.output.mkdir()
    print(json.dumps({'progress':'admitted','phase':args.phase,'seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
    if args.phase == 'cpu':
        result=cpu_qualification(context)
    elif args.phase == 'export':
        result=native_export(context)
    else:
        result=score_exports(context)
    require(torch.equal(rng,torch.random.get_rng_state()) and source.numerical_flags() == flags,
            'whole-unit CPU RNG/numerical flags differ')
    require(args.phase != 'export' or all(torch.equal(a,b) for a,b in
        zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)), 'whole-unit CUDA RNG differs')
    print(json.dumps({'progress':'exit_rehash','seconds':time.perf_counter()-UNIT_STARTED}),flush=True)
    origins=exit_rehash(context)
    prior=t['legacy']['selected']['source_cpu']['invocation']
    record={'schema':SCHEMA,'phase':args.phase,'arm':args.arm,'panel':context['launch']['panel'],
        'binding':binding(context),'source_code':context['code'],'execution_sha256':args.execution_sha256,
        'source':t['source'],'launch':context['launch'],'authority':{'path':str(args.authority),'sha256':args.authority_sha256},
        'authority_sha256':args.authority_sha256,'output':str(args.output),'cost':context['costs'],
        'pass':True,'engineering_admission_pass':True,'integrity_pass':True,'resources_pass':True,
        'exit_rehash_pass':True,'sequential_model_ownership':True,'rng_flags_preserved':True,
        'cuda_initialized':torch.cuda.is_initialized(),'official_read':False,'global_production_goal_met':False,
        'public_latency_measured':False,'product_go':False,'numerical_flags':flags,'origins':origins,
        'input_guards':context['guards'],'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),
            'python_sha256':prior['python_sha256'],'python_version':sys.version,'optimize':sys.flags.optimize,
            'pid':os.getpid(),'invocation_id':os.environ['INVOCATION_ID'],
            'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        **result,**resources(context,before)}
    check_receipt(context,record,args.phase,args.arm)
    context['helper'].publish(args.output/'receipt.json',record)
    require(time.perf_counter()-UNIT_STARTED<policy(args.phase)['seconds'],'receipt included whole-unit cap differs')
    return record


def main():
    args=parser().parse_args()
    try:
        result=run(args)
    except (OSError,ValueError,ImportError,KeyError,TypeError,AttributeError,RuntimeError,SyntaxError) as error:
        raise SystemExit('Nearest-ranking evaluation rejected: '+str(error)) from error
    print(json.dumps({'schema':SCHEMA,'phase':args.phase,'output':str(args.output),'decision':result.get('decision')}),flush=True)


if __name__ == '__main__':
    main()
