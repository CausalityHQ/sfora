#!/usr/bin/env python3
"""Prospective fixed-basis quadratic cached readout evaluator; native gates UNRUN.

Own execution.json contains exactly FILES. Parent pins the actual future
trainer3 root/execution/code (exact TRAIN_FILES) in this hashed authority;
there are no guessed future execution, endpoint, receipt or log hashes.
Separate immutable evaluator6 reference is pinned REFERENCE_EXECUTION_SHA.

Authority siglip2-quadratic-readout-evaluation-authority-v1 has exactly SPEC_KEYS:
execution_sha256; training={root:absolute,execution_sha256:SHA,code:{three files:SHA}};
evaluation_reference={root:REFERENCE_ROOT,execution_sha256:REFERENCE_EXECUTION_SHA};
partition=FILE pinned PARTITION_SHA; source_selection={inventory:FILE pinned
SOURCE_INVENTORY_SHA,terminal:SOURCE_SCORE_TERMINAL}; stage=first|full;
panel=selection|validation; endpoints exactly endpoint_order(stage), each
{seed,arm,launch:FILE,terminal:TERMINAL,checkpoint:FILE,terminal_state_sha256:SHA};
first_selection=null for first, otherwise original accepted first CONTINUE
TERMINAL; selection_go=null for selection, otherwise original full selection GO
TERMINAL; resource_policies={cpu:policy('cpu'),score:policy('score')};
cost_policy=COST_POLICY; both_locks_held=true;
selection_previously_exposed=true; validation_previously_exposed=false.
FILE={path:canonical absolute regular file,sha256:actual lowercase SHA256}.
TERMINAL={receipt:FILE,log:FILE,unit,invocation_id,service_seconds,
native_peak_rss_kib,both_locks_held:true}; whole original normal service exit,
unit limits, log and zero events must authenticate, not a numeric cost summary.

Exact CLI under both parent-owned lifetime locks and unchanged systemd limits:
CUDA_VISIBLE_DEVICES='' python -B ROOT/evaluate_siglip2_quadratic_readout.py
 --execution-sha256 SHA --authority FILE --authority-sha256 SHA
 --phase cpu|score --output NEW_ABSOLUTE_DIRECTORY
Score additionally requires --prerequisite CPU_TERMINAL_JSON
 --prerequisite-sha256 SHA; CPU_TERMINAL_JSON is TERMINAL for this SAME authority.
CPU120/score300 seconds include admission/complete independent reload/fresh
uncached exit/receipt; 8GiB/noSwap/events0/CUDA hidden; no cap rescue.

CPU reloads each complete updated terminal twice and checks actual fresh
canonical TRAIN-normalized witnesses raw/unit/int8/FP16 inverse bits, typed
full state, RNG, flags and frozen source composition. Both arms control factory;
A/means remain explicit; no fitting or encoder factories. Score indexes ONLY
the authorized panel of original FIT cache FIT_SHA; direct FIT head input is
separate from TRAIN normalization. Save/read back/reload complete raw/unit/packed
wires and replay every R1/AP before applying a decision. Source061 selection
receipt914f7a...f06d plus all three exact inventory wires must replay FIRST.
Original reference uses exact per-query equality (including AP); never loosen it.
Validation floor is source061 on THAT panel, only after exact four-checkpoint
selection GO. Source-training uncertainty remains conditional.

First C061,A061 KILL for deltaR1<=0/AP<0 or either candidate source regression;
otherwise CONTINUE only with original whole-service and median-update ratios
<=1.50. No first CI. Full adds A069,C069; eachseed R1>0/AP>=0, equal-seed
both means>=.002, shared5000draw179019 paired-product95 lower both>0 plus query
intervals, source floor eachseed and same cost gates. Only full selection GO
admits sealed validation1749q/1730g/498; no retuning/refit/projections.
Shared actual preparation283.636s remains separate. Private cache updates/wires
are not image training throughput, official/SOTA quality or public latency.
Certificate: updated cached readout composed with qualified immutable encoder.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import time
UNIT_STARTED = time.perf_counter()

import __future__
import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

SCHEMA = 'siglip2-quadratic-readout-evaluation-v1'
AUTHORITY_SCHEMA = 'siglip2-quadratic-readout-evaluation-authority-v1'
FILES = {'evaluate_siglip2_quadratic_readout.py', 'test_siglip2_quadratic_readout_evaluation.py'}
TRAIN_FILES = {'train_siglip2_quadratic_readout.py', 'test_siglip2_quadratic_readout.py', 'quadratic_readout.py'}
SOURCE_INVENTORY_SHA = 'ed0cd43dbbf7f84066e3ab31a28cf249bd0798cf083e6939fe662e7fa8eb8985'


REFERENCE_ROOT = '/home/riomus/runs/sfora-so400-cached-readout-evaluation-source-v3'

REFERENCE_EXECUTION_SHA = 'aec32dd2f9dfecdcc9aa1de22d466503373d84fcac9cd8306969f8cbd54853c6'

REFERENCE_PINS = {
    'evaluate_siglip2_cached_readout.py': '31edb6d0bb4f40b5af938953bca7e558512698079bf55ef3962c51d9e2067e7b',
    'test_siglip2_cached_readout_evaluation.py': 'c0b03b08d7db6594066f436e75cbaeaa7e041b5d4b4fb673e0517ce72a061c57',
    'export_siglip2_substrate_adaptation.py': '89eed215795492d8237d736077b8ee0d0b821ba07dfd3535c65f6643533b9863',
    'score_siglip2_substrate_adaptation.py': '5be3d922b198bd2f6470523cfdcde95884fd03600b9eec0dba876da6df27683f',
    'reference_compare_inshop_sop_warmstart_100.py': '8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250',
    'reference_score_inshop_crop_view_pair.py': '16e27ccaa7325b9ef7efdb3512cb95791a682afdd5ae59bd1f0847d63b87f5ed'}

PARTITION_SHA = '702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c'

FIT_SHA = 'c418df0354408c8b33dca07f21e7b3cbf9d5089f7f278b93003a614f21085716'

SEEDS, ARMS = (179061, 179069), ('control', 'candidate')

ORDER = ((179061, 'control'), (179061, 'candidate'), (179069, 'candidate'), (179069, 'control'))

PANELS = {'selection': (3449, 1734, 1715, 498), 'validation': (3479, 1749, 1730, 498)}

METRICS = ('per_query_r1', 'per_query_ap')

NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}

COST_POLICY = {'whole_service_ratio_max': 1.50, 'median_update_ratio_max': 1.50,
               'training_wall_ratio': 'report_only'}

SOURCE_INVENTORY = {'baseline_endpoint': {'arm': 'control',
                       'checkpoint': {'path': '/home/riomus/runs/sfora-so400-genuine-view-train-control-179061-v1/resume.pt',
                                      'sha256': '97db53ab934edbef2656c6e9509e4518421eaba02d3864bdc336eedd7787c7d2'},
                       'launch': {'path': '/home/riomus/runs/sfora-so400-genuine-view-train-source-v1/authority-train-control-179061-v1.json',
                                  'sha256': '84ceaa5b819d9c9db89514bed13b140dab898e842ec6cbf85eb17dce123e6655'},
                       'seed': 179061,
                       'terminal': {'both_locks_held': True,
                                    'invocation_id': 'e76277bcc16c4a7b9b3c3643834cb840',
                                    'log': {'path': '/home/riomus/runs/sfora-so400-genuine-view-train-source-v1/train-control-179061.log',
                                            'sha256': '5c2c7ca41fe489d8ba56787c32ac6be5cc8b4fe5e842f223e6981c32de0b0bc8'},
                                    'native_peak_rss_kib': 2295028,
                                    'receipt': {'path': '/home/riomus/runs/sfora-so400-genuine-view-train-control-179061-v1/receipt.json',
                                                'sha256': 'a85d7ef29909d677a12510bdafce2e1edbcd8e62d768918af572bdd3cfa87a5f'},
                                    'service_seconds': 182.39,
                                    'unit': 'sfora-so400-genuine-view-train-control-179061-v1'},
                       'terminal_state_sha256': 'b78bd945256438bfd24873347e26d62c970bf9d85048663301d4fe1236ac35a9'},
 'control_quality': {'map_at_r': 0.8057229533585297,
                     'per_query_count': 1734,
                     'recall_at_1': 0.9630911188004614},
 'files': {'control-179061.packed.bin': {'bytes': 448370,
                                         'path': '/home/riomus/runs/sfora-so400-genuine-view-evaluation-first-score-v1/control-179061.packed.bin',
                                         'sha256': '9c8d7af9fa5eb30893b2019da40c66e17d96efbc6822e12a87ddcc962f82095f'},
           'control-179061.raw.npy': {'bytes': 1766016,
                                      'path': '/home/riomus/runs/sfora-so400-genuine-view-evaluation-first-score-v1/control-179061.raw.npy',
                                      'sha256': 'c4db77dee9bdcc1fde22a012a5671b1cc5f4c4ee2853eff05a1d0188e04765fb'},
           'control-179061.unit.npy': {'bytes': 1766016,
                                       'path': '/home/riomus/runs/sfora-so400-genuine-view-evaluation-first-score-v1/control-179061.unit.npy',
                                       'sha256': '09ad04f736886b97fe2ad0a8dafb5be2b7eb0c3d20e873f94ae7ad29c339bdb4'}},
 'gallery_images': 1715,
 'metadata_and_bytes_verified_only': True,
 'new_metric_replay': False,
 'panel': 'selection',
 'partition': {'path': '/home/riomus/runs/sfora-so400-genuine-view-export-source-v2/partition.json',
               'sha256': '702f763eab7138491450eb6a0c58a2aeabc57b194ed8edbb4db2a5abd95c676c'},
 'products': 498,
 'query_images': 1734,
 'receipt': {'path': '/home/riomus/runs/sfora-so400-genuine-view-evaluation-first-score-v1/receipt.json',
             'sha256': '914f7a082570da853a404e8dd1960f41bbbdd634e92c01c9dd92aac8abd3f06d'},
 'schema': 'quadratic-source-selection-inventory-v1'}

SOURCE_SCORE_TERMINAL = {'receipt': SOURCE_INVENTORY['receipt'], 'log': {'path': '/home/riomus/runs/sfora-so400-genuine-view-evaluation-source-v4/first-score.log', 'sha256': 'b74fd184eeafe42fe15b0aaf1af4d4226bf3c2847f7d61436e2e64348ee66650'}, 'unit': 'sfora-so400-genuine-view-evaluation-first-score-v1', 'invocation_id': 'f8a6278997014c35b274021dcf1f438e', 'service_seconds': 120.835, 'native_peak_rss_kib': 996056, 'both_locks_held': True}

SPEC_KEYS = {'schema', 'execution_sha256', 'training', 'evaluation_reference', 'partition', 'source_selection', 'stage', 'panel', 'endpoints', 'first_selection', 'selection_go', 'resource_policies', 'cost_policy', 'both_locks_held', 'selection_previously_exposed', 'validation_previously_exposed'}

def require(condition, message):
    if not condition:
        raise ValueError(message)

def policy(phase):
    require(phase in ('cpu', 'score'), 'fixed evaluation phase required')
    return {'seconds': 120 if phase == 'cpu' else 300, 'host_bytes': 8 * 1024**3,
            'swap_bytes': 0, 'cuda_visible_devices': ''}

def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'nonfinite JSON: ' + value))

def bound_file(guards, path, expected):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path and path.is_file(), 'canonical file required')
    require(isinstance(expected, str) and re.fullmatch('[0-9a-f]{64}', expected), 'SHA256 required')
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        while raw := stream.read(1024**2):
            digest.update(raw)
            os.posix_fadvise(stream.fileno(), stream.tell() - len(raw), len(raw), os.POSIX_FADV_DONTNEED)
    require(digest.hexdigest() == expected, 'file SHA256 differs: ' + str(path))
    require(guards.setdefault(str(path), expected) == expected, 'conflicting file authority')
    return path

def read_json(value, guards):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'}, 'exact FILE descriptor required')
    with bound_file(guards, value['path'], value['sha256']).open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == value['sha256'], 'JSON size/hash differs')
    return strict_json(raw)

def closure(root, expected, names, guards):
    root = Path(root)
    require(root.is_absolute() and root.resolve() == root and root.is_dir(), 'canonical closure root required')
    code = read_json({'path': str(root / 'execution.json'), 'sha256': expected}, guards)
    require(code.keys() == names, 'exact execution closure required')
    for name, digest in code.items():
        bound_file(guards, root / name, digest)
    return code

def load_bare(name, path, digest):
    require(name not in sys.modules, 'preloaded helper forbidden')
    raw = bound_file({}, path, digest).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'helper changed before execution')
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path, 'helper origin differs')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), vars(module))
    require(Path(module.__file__) == Path(module.__spec__.origin) == path, 'loaded helper origin differs')
    return module

def seeds(stage):
    require(stage in ('first', 'full'), 'fixed prospective stage required')
    return SEEDS[:1] if stage == 'first' else SEEDS

def endpoint_order(stage):
    seeds(stage)
    return ORDER[:2] if stage == 'first' else ORDER

def check_output(output):
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and
            not output.is_symlink(), 'exclusive canonical output required')

def check_terminal_descriptor(terminal):
    require(terminal.keys() == {'receipt', 'log', 'unit', 'invocation_id', 'service_seconds',
                               'native_peak_rss_kib', 'both_locks_held'} and terminal['both_locks_held'] is True and
            all(terminal[k].keys() == {'path', 'sha256'} and Path(terminal[k]['path']).is_absolute() and
                re.fullmatch('[0-9a-f]{64}', terminal[k]['sha256']) for k in ('receipt', 'log')) and
            re.fullmatch('[A-Za-z0-9_.@-]+', terminal['unit']) and
            re.fullmatch('[0-9a-f]{32}', terminal['invocation_id']) and
            all(type(terminal[k]) in (int, float) and math.isfinite(terminal[k]) and terminal[k] > 0 for k in
                ('service_seconds', 'native_peak_rss_kib')),
            'complete actual terminal descriptor required')

def first_gate(deltas):
    require(deltas.keys() == {str(SEEDS[0])}, 'first pair only required')
    return statistics.mean(deltas[str(SEEDS[0])][METRICS[0]]) > 0 and statistics.mean(deltas[str(SEEDS[0])][METRICS[1]]) >= 0

def quality_gate(deltas, intervals):
    require(deltas.keys() == {str(s) for s in SEEDS} and intervals.keys() == set(METRICS), 'full paired intervals required')
    require(all(math.isclose(intervals[m]['mean_delta'], statistics.mean(statistics.mean(deltas[str(s)][m]) for s in SEEDS),
                             rel_tol=0, abs_tol=1e-12) for m in METRICS), 'interval mean differs from per-query replay')
    each_seed = all(statistics.mean(deltas[str(s)][METRICS[0]]) > 0 and statistics.mean(deltas[str(s)][METRICS[1]]) >= 0 for s in SEEDS)
    bounds = all(v.keys() == {'mean_delta', 'product_lower95', 'product_upper95', 'query_lower95', 'query_upper95'} and
                 all(type(x) in (int, float) and math.isfinite(x) for x in v.values()) and
                 v['mean_delta'] >= .002 and v['product_lower95'] > 0 for v in intervals.values())
    return each_seed, bool(each_seed and bounds)

def cli_argv(authority, authority_sha, execution_sha, phase, output, prerequisite):
    value = [str(Path(__file__).absolute()), '--execution-sha256', execution_sha,
             '--authority', str(authority), '--authority-sha256', authority_sha,
             '--phase', phase, '--output', str(output)]
    if prerequisite is not None:
        value += ['--prerequisite', prerequisite['path'], '--prerequisite-sha256', prerequisite['sha256']]
    return value

def unique_terminals(terminals):
    unique = {t['receipt']['path']: t for t in terminals}
    require(all(unique[t['receipt']['path']] == t for t in terminals) and
            len({t['unit'] for t in unique.values()}) == len(unique) and
            len({t['invocation_id'] for t in unique.values()}) == len(unique), 'distinct original whole units required')

def label(endpoint):
    return endpoint['arm'] + '-' + str(endpoint['seed'])

def file_names(spec):
    return {label(e) + suffix for e in spec['endpoints'] for suffix in ('.raw.npy', '.unit.npy', '.packed.bin')}

def replay_equal(expected, actual):
    require(actual == expected, 'independent per-query packed quality replay differs')

def packed_readback(guards, path, digest, expected):
    require(bound_file(guards, path, digest).read_bytes() == expected, 'independent packed wire differs')

def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--execution-sha256', required=True)
    result.add_argument('--authority', type=Path, required=True)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--phase', choices=('cpu', 'score'), required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--prerequisite', type=Path)
    result.add_argument('--prerequisite-sha256')
    return result


def check_file_descriptor(value):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'} and
            type(value['path']) is str and Path(value['path']).is_absolute() and
            type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
            'exact actual FILE descriptor required')


def check_spec(spec, args):
    require(spec.keys() == SPEC_KEYS and spec['schema'] == AUTHORITY_SCHEMA and
            spec['execution_sha256'] == args.execution_sha256 and spec['both_locks_held'] is True and
            spec['resource_policies'] == {p: policy(p) for p in ('cpu', 'score')} and
            spec['cost_policy'] == COST_POLICY and spec['selection_previously_exposed'] is True and
            spec['validation_previously_exposed'] is False, 'evaluation authority profile differs')
    training = spec['training']
    require(training.keys() == {'root', 'execution_sha256', 'code'} and
            type(training['root']) is str and Path(training['root']).is_absolute() and
            re.fullmatch('[0-9a-f]{64}', training['execution_sha256']) and
            training['code'].keys() == TRAIN_FILES and
            all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v) for v in training['code'].values()),
            'parent-frozen actual complete trainer3 closure required')
    require(spec['evaluation_reference'] == {'root': REFERENCE_ROOT, 'execution_sha256': REFERENCE_EXECUTION_SHA},
            'original immutable evaluator6 required')
    check_file_descriptor(spec['partition'])
    source = spec['source_selection']
    require(source.keys() == {'inventory', 'terminal'}, 'source selection authority required')
    check_file_descriptor(source['inventory'])
    require(spec['partition']['sha256'] == PARTITION_SHA and
            source['inventory']['sha256'] == SOURCE_INVENTORY_SHA and
            source['terminal'] == SOURCE_SCORE_TERMINAL, 'original partition/source inventory/terminal differs')
    require(spec['panel'] in PANELS and (spec['panel'] != 'validation' or spec['stage'] == 'full') and
            (spec['first_selection'] is None) == (spec['stage'] == 'first') and
            (spec['selection_go'] is None) == (spec['panel'] == 'selection'),
            'validation requires selection GO; full requires first continuation')
    require([(e['seed'], e['arm']) for e in spec['endpoints']] == list(endpoint_order(spec['stage'])) and
            all(type(e['seed']) is int for e in spec['endpoints']), 'prospective ordered endpoints required')
    for endpoint in spec['endpoints']:
        require(endpoint.keys() == {'seed', 'arm', 'launch', 'terminal', 'checkpoint', 'terminal_state_sha256'} and
                re.fullmatch('[0-9a-f]{64}', endpoint['terminal_state_sha256']), 'exact endpoint required')
        for key in ('launch', 'checkpoint'):
            check_file_descriptor(endpoint[key])
        check_terminal_descriptor(endpoint['terminal'])
    for terminal in (spec['first_selection'], spec['selection_go']):
        if terminal is not None:
            check_terminal_descriptor(terminal)


def check_prior_binding(current, prior, stage):
    require(prior['stage'] == stage and prior['panel'] == 'selection' and
            all(prior[k] == current[k] for k in ('execution_sha256', 'training', 'evaluation_reference', 'partition',
                'source_selection', 'resource_policies', 'cost_policy', 'selection_previously_exposed',
                'validation_previously_exposed')) and
            prior['endpoints'] == current['endpoints'][:len(endpoint_order(stage))] and
            (stage != 'full' or prior['first_selection'] == current['first_selection']),
            'original selection authority/checkpoints differ')


def check_quality(value, count):
    require(value.keys() == {'recall_at_1', 'map_at_r', *METRICS} and
            len(value['per_query_r1']) == len(value['per_query_ap']) == count and
            all(type(v) in (int, float) and math.isfinite(v) and v in (0, 1) for v in value['per_query_r1']) and
            all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in value['per_query_ap']) and
            all(type(value[k]) in (int, float) and math.isfinite(value[k]) and
                math.isclose(value[k], statistics.mean(value[m]), rel_tol=0, abs_tol=1e-12)
                for k, m in zip(('recall_at_1', 'map_at_r'), METRICS, strict=True)),
            'complete finite per-query metrics/aggregate required')


def averaged_deltas(quality, stage, panel):
    require(panel in PANELS and quality.keys() == {str(s) for s in seeds(stage)}, 'complete panel seeds required')
    count, deltas = PANELS[panel][1], {}
    for seed in seeds(stage):
        pair = quality[str(seed)]
        require(pair.keys() == set(ARMS), 'complete paired panel arms required')
        for value in pair.values():
            check_quality(value, count)
        deltas[str(seed)] = {m: [b - a for a, b in zip(pair['control'][m], pair['candidate'][m], strict=True)] for m in METRICS}
    average = {m: [statistics.mean(row) for row in zip(*(deltas[str(s)][m] for s in seeds(stage)), strict=True)] for m in METRICS}
    return deltas, average


def source_floor(quality, source_quality, stage, panel):
    check_quality(source_quality, PANELS[panel][1])
    return {str(s): all(statistics.mean(quality[str(s)]['candidate'][m]) >= statistics.mean(source_quality[m])
                        for m in METRICS) for s in seeds(stage)}


def decide(quality, source_quality, stage, panel, intervals, costs):
    deltas, average = averaged_deltas(quality, stage, panel)
    floor = source_floor(quality, source_quality, stage, panel)
    require(costs.keys() == {str(s) for s in seeds(stage)} and
            all(type(c['pass']) is bool for c in costs.values()), 'complete paired cost decisions required')
    if stage == 'first':
        require(intervals == {} and panel == 'selection', 'first selection has no CI')
        each_seed = paired = first_gate(deltas)
    else:
        each_seed, paired = quality_gate(deltas, intervals)
    quality_pass, cost_pass = bool(paired and all(floor.values())), all(c['pass'] for c in costs.values())
    return {'decision': ('CONTINUE' if stage == 'first' else 'GO') if quality_pass and cost_pass else 'KILL',
            'each_seed_quality_pass': each_seed, 'source_floor_per_seed': floor,
            'source_floor_pass': all(floor.values()), 'quality_pass': quality_pass, 'cost_pass': cost_pass,
            'mean_deltas': {m: statistics.mean(average[m]) for m in METRICS}}


def scoring_math(context):
    """Execute unchanged pinned definitions in a fresh namespace; no rebinding."""
    import numpy as np
    import torch
    helper = context['helper']
    namespace = {'__name__': '_quadratic_fixed_scoring_math', 'np': np, 'torch': torch,
                 'pack_int8_unit_embeddings': context['packing'].pack_int8_unit_embeddings}
    for name, pin in helper.REFERENCES.items():
        require(REFERENCE_PINS[name] == pin['source'], 'fixed reference source differs')
        path = bound_file(context['guards'], Path(REFERENCE_ROOT) / name, pin['source'])
        raw = path.read_bytes()
        require(hashlib.sha256(raw).hexdigest() == pin['source'], 'scoring source changed before use')
        tree = ast.parse(raw, filename=str(path))
        nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == pin['name']]
        selected = ast.Module(body=nodes, type_ignores=[])
        require(len(nodes) == 1 and hashlib.sha256(ast.dump(selected, include_attributes=False).encode()).hexdigest() == pin['ast'],
                'fixed scoring reference AST differs')
        exec(compile(selected, str(path), 'exec', flags=__future__.annotations.compiler_flag, dont_inherit=True), namespace)
    require({'packed_quality', 'bootstrap_lower'} <= namespace.keys(), 'complete scoring math required')
    return SimpleNamespace(**namespace)


def paired_cost(records, stage):
    require(set(records) == set(endpoint_order(stage)), 'complete paired costs required')
    costs = {}
    for seed in seeds(stage):
        control, candidate = (records[seed, arm] for arm in ARMS)
        require(all(control['identity'][k] == candidate['identity'][k] for k in
                    ('encoder_sha256', 'complement_sha256', 'buffers_sha256', 'head_buffers_sha256',
                     'means_sha256', 'static_sha256', 'warm_members_sha256', 'schedule_sha256', 'full_schedule_sha256')) and
                all(all(a[k] == b[k] for k in ('step', 'batch', 'schedule_sha256', 'feature_rows_sha256', 'mask_sha256'))
                    for a, b in zip(control['steps'], candidate['steps'], strict=True)), 'paired TRAIN inputs differ')
        for record in (control, candidate):
            require(all(type(record[k]) in (int, float) and math.isfinite(record[k]) and record[k] > 0 for k in
                        ('service_seconds', 'median_update_seconds', 'training_wall_seconds')), 'finite original endpoint cost required')
        ratios = {name: candidate[key] / control[key] for name, key in
                  (('whole_service_ratio', 'service_seconds'), ('median_update_ratio', 'median_update_seconds'),
                   ('training_wall_ratio', 'training_wall_seconds'))}
        require(all(math.isfinite(v) and v > 0 for v in ratios.values()), 'finite cost ratio required')
        costs[str(seed)] = {**ratios, 'pass': ratios['whole_service_ratio'] <= 1.50 and ratios['median_update_ratio'] <= 1.50,
                            'training_wall_ratio_gate': False,
                            **{arm: {k: records[seed, arm][k] for k in ('service_seconds', 'median_update_seconds',
                                                                     'training_wall_seconds')} for arm in ARMS}}
    return costs


def preparation_costs(selected):
    terminal, record = selected['launch']['selected_export'], selected['export_record']
    require(terminal['service_seconds'] == 283.636 and
            0 < record['extraction_seconds'] <= record['wall_seconds'] < terminal['service_seconds'],
            'actual shared preparation cost differs')
    return {'shared': {'terminal': terminal, 'whole_service_seconds': terminal['service_seconds'],
                       'extraction_seconds': record['extraction_seconds'], 'views': ['canonical', 'augmented']},
            'arm_attributed': {a: {'shared_whole_service_seconds': terminal['service_seconds'],
                                  'additional_preparation_jobs': 0} for a in ARMS},
            'shared_counted_once': True, 'included_in_training_cost_ratios': False,
            'image_training_throughput': False}


def check_endpoint(trainer, record, endpoint, selected):
    launch, arm, seed = record['launch'], endpoint['arm'], endpoint['seed']
    trainer.check_terminal_record(record, selected['launch'], 'train', arm)
    require((record['seed'], record['arm']) == (seed, arm) and record['training_qualified'] is True and
            record['replay_exact'] is False and
            record['source'] == selected['source'] and record['code'] == selected['code'] and
            record['terminal_cgroups'] == selected['terminal_cgroups'] and
            record['numerical_flags'] == selected['selected']['source_cpu']['numerical_flags'] and
            record['partition_sha256'] == PARTITION_SHA and
            record['checkpoint'] == endpoint['checkpoint'] and
            record['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and
            launch['selected_cpu'] == selected['launch']['selected_cpu'] and
            launch['selected_mechanics'] == selected['launch']['selected_mechanics'],
            'complete fresh TRAIN1000 endpoint differs')
    ident = record['identity']
    cpu_ident = selected['terminals']['cpu:control']['arms'][arm]['identity']
    require(ident == {**cpu_ident, 'device': 'cuda', 'seed': seed,
                     'schedule_sha256': ident['schedule_sha256'], 'full_schedule_sha256': ident['full_schedule_sha256']},
            'complete TRAIN identity differs from new CPU qualification')
    if seed == SEEDS[0]:
        mechanics = selected['terminals']['mechanics:' + arm]
        require(record['initial_state_sha256'] == mechanics['initial_state_sha256'] and
                record['initial_raw_unit_packed_sha256'] == mechanics['initial_raw_unit_packed_sha256'] and
                [trainer.diagnostic(r) for r in record['steps'][:17]] ==
                [trainer.diagnostic(r) for r in mechanics['steps']], 'fresh first17 mechanics replay differs')


def check_source_record(record, selected):
    spec = record['spec']
    require(record['schema'] == 'siglip2-genuine-view-evaluation-v1' and
            record['phase'] == 'score' and record['pass'] is True and record['engineering_admission_pass'] is True and
            record['source'] == selected['selected']['source'] and
            spec['panel'] == 'selection' and spec['stage'] == 'first' and
            spec['endpoints'][0] == SOURCE_INVENTORY['baseline_endpoint'] and
            spec['partition'] == SOURCE_INVENTORY['partition'] and
            spec['evaluation_reference'] == {'root': REFERENCE_ROOT, 'execution_sha256': REFERENCE_EXECUTION_SHA} and
            record['resource_policy'] == policy('score') and
            (record['query_images'], record['gallery_images'], record['panel_products']) == (1734, 1715, 498) and
            all(record[k] is True for k in ('strict_independent_head_reload_exact', 'train_raw_unit_cpu_packed_exact',
                'rng_flags_preserved', 'exit_rehash_pass', 'full_panel_raw_unit_packed_replay_exact', 'per_query_replay_exact')) and
            all(record[k] is False for k in ('official_read', 'public_latency_measured', 'global_production_goal_met', 'cuda_initialized')) and
            record['peak_cuda_allocated_bytes'] == 0, 'archived source selection engineering evidence differs')
    for name, descriptor in SOURCE_INVENTORY['files'].items():
        require(record['files'][name] == descriptor['sha256'] and
                Path(record['output']) / name == Path(descriptor['path']), 'source selection wire role differs')
    quality = record['quality']['179061']['control']
    check_quality(quality, 1734)
    require(quality['recall_at_1'] == SOURCE_INVENTORY['control_quality']['recall_at_1'] and
            quality['map_at_r'] == SOURCE_INVENTORY['control_quality']['map_at_r'], 'source observed selection point differs')


def admit_source_selection(context):
    guards, selected = context['guards'], context['selected']
    inventory = read_json(context['spec']['source_selection']['inventory'], guards)
    require(inventory == SOURCE_INVENTORY, 'exact archived source selection inventory required')
    record = read_json(inventory['receipt'], guards)
    check_source_record(record, selected)
    final = context['admission'].admit_terminal(record, SOURCE_SCORE_TERMINAL, 300, guards)
    for value in (record['cgroup_before'], record['cgroup_after'], final):
        context['helper'].zero_events(value)
    for path, digest in record['input_guards'].items():
        context['admission'].bound_file(guards, path, digest)
    for descriptor in inventory['files'].values():
        path = context['admission'].bound_file(guards, descriptor['path'], descriptor['sha256'])
        require(path.stat().st_size == descriptor['bytes'], 'archived source wire length differs')
    context['source_record'] = record
    context['source_guards'] = {**record['input_guards'],
        context['spec']['source_selection']['inventory']['path']: SOURCE_INVENTORY_SHA,
        inventory['receipt']['path']: inventory['receipt']['sha256'],
        SOURCE_SCORE_TERMINAL['log']['path']: SOURCE_SCORE_TERMINAL['log']['sha256'],
        **{d['path']: d['sha256'] for d in inventory['files'].values()}}
    context['origin_records'].append(record)
    context['terminals'].append(SOURCE_SCORE_TERMINAL)


def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native imports preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    spec = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_spec(spec, args); check_output(args.output)
    train_root, reference_root = Path(spec['training']['root']), Path(REFERENCE_ROOT)
    roots = (root, train_root, reference_root)
    require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]) and
            all(not args.output.is_relative_to(p) and not p.is_relative_to(args.output) for p in roots),
            'separate immutable evaluator2/trainer3/reference6 required')
    train_code = closure(train_root, spec['training']['execution_sha256'], TRAIN_FILES, guards)
    require(train_code == spec['training']['code'], 'parent-frozen actual trainer3 differs')
    require(closure(reference_root, REFERENCE_EXECUTION_SHA, REFERENCE_PINS.keys(), guards) == REFERENCE_PINS,
            'pinned original evaluator reference differs')
    trainer = load_bare('_quadratic_evaluation_trainer', train_root / 'train_siglip2_quadratic_readout.py',
                        train_code['train_siglip2_quadratic_readout.py'])
    require(trainer.FILES == TRAIN_FILES and trainer.SCHEMA == 'siglip2-quadratic-readout-v1' and
            trainer.SEEDS == SEEDS and trainer.ARMS == ARMS and trainer.PARTITION_SHA == PARTITION_SHA,
            'new quadratic trainer profile differs')
    legacy = load_bare('_quadratic_evaluation_reference', reference_root / 'evaluate_siglip2_cached_readout.py',
                       REFERENCE_PINS['evaluate_siglip2_cached_readout.py'])
    helper = load_bare('_quadratic_evaluation_helper', reference_root / 'export_siglip2_substrate_adaptation.py',
                       REFERENCE_PINS['export_siglip2_substrate_adaptation.py'])
    first = spec['endpoints'][0]
    selected = trainer.authority(SimpleNamespace(execution_sha256=spec['training']['execution_sha256'],
        authority=Path(first['launch']['path']), authority_sha256=first['launch']['sha256'],
        phase='train', arm='control', seed=SEEDS[0], output=args.output))
    admission, old = selected['admission'], selected['selected']
    required = {p: h for p, h in selected['guards'].items() if p != first['launch']['path']}
    partition = read_json(spec['partition'], selected['guards'])
    require(partition == old['partition'] and spec['partition'] == selected['launch']['partition'] and
            partition['original_cache']['sha256'] == FIT_SHA, 'original frozen partition/FIT cache differs')
    admission.bound_file(selected['guards'], partition['original_cache']['path'], FIT_SHA)
    source_cpu = old['genuine']['prior']['launch']['source_cpu']['so400']
    startup = old['exporter'].file_json(old['export_record']['startup_terminal'], selected['guards'])
    terminals = [selected['launch']['warm_start']['terminal'], selected['launch']['selected_cpu'],
                 *selected['launch']['selected_mechanics'].values(), old['launch']['selected_cpu'],
                 *old['launch']['selected_mechanics'].values()]
    for proof in (source_cpu, startup, old['launch']['selected_export']):
        terminals.append({('receipt' if k == 'proof' else k): v for k, v in proof.items()})
    base_guards = {**selected['guards'], **{p: h for p, h in guards.items() if p != str(args.authority)}}
    records = {}
    for endpoint in spec['endpoints']:
        launch = read_json(endpoint['launch'], selected['guards'])
        record = read_json(endpoint['terminal']['receipt'], selected['guards'])
        require(record['launch'] == launch and record['authority'] == endpoint['launch'] and
                record['authority_sha256'] == endpoint['launch']['sha256'] and
                record['execution_sha256'] == spec['training']['execution_sha256'], 'original endpoint authority differs')
        check_endpoint(trainer, record, endpoint, selected)
        terminal, invocation, prior = endpoint['terminal'], record['invocation'], old['source_cpu']['invocation']
        require(invocation['argv'] == trainer.cli(train_root, endpoint['launch']['path'], endpoint['launch']['sha256'],
                spec['training']['execution_sha256'], 'train', endpoint['arm'], endpoint['seed'],
                Path(terminal['receipt']['path']).parent) and
                all(invocation[k] == prior[k] for k in ('python', 'python_sha256', 'python_version')) and
                record['output'] == str(Path(terminal['receipt']['path']).parent) and
                Path(terminal['receipt']['path']).name == 'receipt.json' and
                Path(endpoint['checkpoint']['path']) == Path(terminal['receipt']['path']).parent / 'resume.pt' and
                all(record['input_guards'].get(p) == h for p, h in required.items()), 'original TRAIN invocation/path/guards differs')
        final = admission.admit_terminal(record, terminal, 300, selected['guards'])
        for value in (record['cgroup_before'], record['cgroup_after'], final):
            helper.zero_events(value)
        helper.logged_steps(Path(terminal['log']['path']), record['steps'])
        for path, digest in record['input_guards'].items():
            admission.bound_file(selected['guards'], path, digest)
        admission.bound_file(selected['guards'], endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])
        records[endpoint['seed'], endpoint['arm']] = {**record, 'service_seconds': terminal['service_seconds']}
        terminals.append(terminal)
    for path, digest in guards.items():
        admission.bound_file(selected['guards'], path, digest)
    exported = old['export_record']
    context = {'args': args, 'root': root, 'code': code, 'spec': spec, 'trainer': trainer, 'legacy': legacy,
               'selected': selected, 'guards': selected['guards'], 'helper': helper, 'admission': admission,
               'records': records, 'fit': old['genuine']['prior']['fit'], 'partition': partition,
               'terminals': terminals, 'origin_records': [old['source_cpu'], selected['warm_record'],
                   {**exported, 'input_guards': {**exported['original_input_guards'], **exported['input_guards']}},
                   *old['terminals'].values(), *selected['terminals'].values(), *records.values()],
               'selected_context': {'source': selected['source_driver'], 'initialized': {'init': admission.init}},
               'unit_started': UNIT_STARTED, 'base_guards': base_guards}
    context['costs'] = paired_cost(records, spec['stage']); preparation_costs(old)
    admit_source_selection(context)
    context['base_guards'].update(context['source_guards'])
    context['required_guards'] = context['guards'].copy()
    panel_authority(context)
    context['required_guards'] = context['guards'].copy()
    require(all(not args.output.is_relative_to(Path(t['receipt']['path']).parent) and
                not Path(t['receipt']['path']).parent.is_relative_to(args.output) for t in context['terminals']),
            'immutable original terminal/output overlap')
    unique_terminals(context['terminals'])
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import during endpoint/panel admission')
    return context


def bind(context):
    return {'authority_sha256': context['args'].authority_sha256,
            'execution_sha256': context['args'].execution_sha256, 'spec': context['spec'],
            'selection_previously_exposed': True, 'validation_previously_exposed': False,
            'source': context['selected']['source'], 'source_code': context['code']}


def check_value_facts(facts, count):
    require(facts.keys() == {'raw', 'unit', 'codes', 'inverse_norms'} and all(
        facts[n].keys() == {'shape', 'dtype', 'sha256'} and facts[n]['shape'] == shape and
        facts[n]['dtype'] == dtype and re.fullmatch('[0-9a-f]{64}', facts[n]['sha256']) for n, shape, dtype in
        (('raw', [count, 128], 'torch.float32'), ('unit', [count, 128], 'torch.float32'),
         ('codes', [count, 128], 'torch.int8'), ('inverse_norms', [count], 'torch.float16'))),
        'complete raw/unit/int8/inverse-bit facts required')


def check_receipt(context, record, phase):
    spec, old = context['spec'], context['selected']['selected']
    require(all(record[k] == v for k, v in bind(context).items()) and record['schema'] == SCHEMA and
            record['phase'] == phase and record['resource_policy'] == policy(phase) and record['pass'] is True and
            record['engineering_admission_pass'] is True and record['optimizer_updates'] == 0 and record['optimizer_members'] == 1 and
            record['certificate'] == 'updated cached readout composed with qualified immutable encoder' and
            all(record[k] is True for k in ('strict_independent_head_reload_exact', 'train_raw_unit_cpu_packed_exact',
                'complete_typed_terminal_state_exact', 'first_heads_released_before_reload', 'rng_flags_preserved', 'exit_rehash_pass')) and
            all(record[k] is False for k in ('official_read', 'global_production_goal_met', 'public_latency_measured',
                                           'public_encoder_qualified', 'cuda_initialized')) and
            record['peak_cuda_allocated_bytes'] == 0, 'accepted evaluator engineering receipt differs')
    prior, invocation = old['source_cpu'], record['invocation']
    require(record['numerical_flags'] == prior['numerical_flags'] and invocation['argv'] ==
            cli_argv(Path(record['authority']['path']), record['authority']['sha256'], record['execution_sha256'],
                     phase, Path(record['output']), record['prerequisite']) and
            all(invocation[k] == prior['invocation'][k] for k in ('python', 'python_sha256', 'python_version')) and
            invocation['cuda_visible_devices'] == '' and invocation['optimize'] == 0 and
            record['authority'] == {'path': str(context['args'].authority), 'sha256': context['args'].authority_sha256} and
            record['output'] == str(Path(record['output']).resolve()) and
            record['head_facts'].keys() == record['train_witnesses'].keys() == {label(e) for e in spec['endpoints']},
            'original evaluator invocation/readout qualification differs')
    for endpoint in spec['endpoints']:
        key, fact = label(endpoint), record['head_facts'][label(endpoint)]
        ident = context['records'][endpoint['seed'], endpoint['arm']]['identity']
        require(fact.keys() == {'payload_sha256', 'readout_sha256', 'encoder_sha256', 'complement_sha256', 'means_sha256', 'features_sha256'} and
                fact['payload_sha256'] == endpoint['terminal_state_sha256'] and
                all(fact[k] == ident[k] for k in ('encoder_sha256', 'complement_sha256', 'means_sha256', 'features_sha256')) and
                re.fullmatch('[0-9a-f]{64}', fact['readout_sha256']), 'complete terminal/readout facts differ')
        check_value_facts(record['train_witnesses'][key], 64)
    if phase == 'cpu':
        require(record['quality_read'] is False and record['prerequisite'] is None and record['cpu_terminal'] is None and
                record['files'] == {} and record['validation_quality_exposed'] is False, 'TRAIN-only CPU qualification differs')
        return
    require(record['quality_read'] is True and record['full_panel_raw_unit_packed_replay_exact'] is True and
            record['per_query_replay_exact'] is True and record['archived_selection_per_query_exact'] is True and
            record['files'].keys() == file_names(spec) | source_file_names() and
            record['cost'] == paired_cost(context['records'], spec['stage']) and record['cost_policy'] == COST_POLICY and
            record['bootstrap_draws'] == (0 if spec['stage'] == 'first' else 5000) and record['bootstrap_seed'] == 179019 and
            record['source_quality_panel'] == spec['panel'] and
            record['source_selection_receipt'] == SOURCE_INVENTORY['receipt'] and
            record['source_checkpoint'] == SOURCE_INVENTORY['baseline_endpoint']['checkpoint'] and
            record['source_terminal_state_sha256'] == SOURCE_INVENTORY['baseline_endpoint']['terminal_state_sha256'] and
            record['canonical_panel_input_only'] is True and record['serving_view_averaging'] is False and
            record['validation_quality_exposed'] is (spec['panel'] == 'validation') and
            record['preparation_costs'] == preparation_costs(old), 'accepted panel/source replay/cost/scope differs')
    require((record['panel_images'], record['query_images'], record['gallery_images'], record['panel_products']) == PANELS[spec['panel']],
            'complete authorized panel geometry differs')
    check_value_facts(record['source_panel_facts'], record['panel_images'])
    require(record['panel_replay_facts'].keys() == record['head_facts'].keys(), 'complete panel endpoint facts required')
    for fact in record['panel_replay_facts'].values():
        check_value_facts(fact, record['panel_images'])
    if spec['panel'] == 'selection':
        replay_equal(context['source_record']['quality']['179061']['control'], record['source_quality'])
    decision = decide(record['quality'], record['source_quality'], spec['stage'], spec['panel'],
                      record['paired_seed_average_intervals'], record['cost'])
    require(all(record[k] == v for k, v in decision.items()), 'terminal quality decision differs')


def accept_terminal(context, terminal, phase):
    check_terminal_descriptor(terminal)
    record = read_json(terminal['receipt'], context['guards'])
    check_receipt(context, record, phase)
    require(Path(terminal['receipt']['path']) == Path(record['output']) / 'receipt.json', 'original receipt role differs')
    final = context['admission'].admit_terminal(record, terminal, policy(phase)['seconds'], context['guards'])
    for value in (record['cgroup_before'], record['cgroup_after'], final):
        context['helper'].zero_events(value)
    for path, digest in record['input_guards'].items():
        context['admission'].bound_file(context['guards'], path, digest)
    require(all(record['input_guards'].get(p) == h for p, h in context['required_guards'].items()),
            'complete evaluator input guards differ')
    for name, digest in record['files'].items():
        context['admission'].bound_file(context['guards'], Path(record['output']) / name, digest)
    context['terminals'].append(terminal); context['origin_records'].append(record)
    return record


def accept_selection(context, terminal, stage):
    record = read_json(terminal['receipt'], context['guards'])
    prior_spec = read_json(record['authority'], context['guards'])
    check_spec(prior_spec, context['args']); check_prior_binding(context['spec'], prior_spec, stage)
    args = SimpleNamespace(**vars(context['args']))
    args.authority, args.authority_sha256 = Path(record['authority']['path']), record['authority']['sha256']
    subset = {k: context['records'][k] for k in endpoint_order(stage)}
    prior_context = {**context, 'args': args, 'spec': prior_spec, 'records': subset, 'required_guards': context['base_guards']}
    accepted = accept_terminal(prior_context, terminal, 'score')
    require(accepted['decision'] == ('CONTINUE' if stage == 'first' else 'GO') and
            accepted['quality_pass'] is True and accepted['cost_pass'] is True, 'selection continuation/GO required')
    cpu_terminal = read_json(accepted['prerequisite'], context['guards'])
    require(cpu_terminal == accepted['cpu_terminal'], 'original selection CPU terminal differs')
    cpu = accept_terminal(prior_context, cpu_terminal, 'cpu')
    require(all(cpu[k] == accepted[k] for k in ('head_facts', 'train_witnesses')), 'selection CPU witnesses differ')
    return accepted


def panel_authority(context):
    """Close selection decisions before importing native code or panel inputs."""
    check_spec(context['spec'], context['args'])
    for key, stage in (('first_selection', 'first'), ('selection_go', 'full')):
        if context['spec'][key] is not None:
            context[key] = accept_selection(context, context['spec'][key], stage)


def prerequisites(context):
    args = context['args']
    require((args.prerequisite is None) == (args.phase == 'cpu') and
            (args.prerequisite_sha256 is None) == (args.phase == 'cpu'), 'phase prerequisite differs')
    if args.phase == 'cpu':
        context['cpu_terminal'] = None
        return None
    terminal = read_json({'path': str(args.prerequisite), 'sha256': args.prerequisite_sha256}, context['guards'])
    context['cpu_terminal'] = terminal
    cpu = accept_terminal(context, terminal, 'cpu'); unique_terminals(context['terminals'])
    return cpu


def qualified_origins(context):
    packages, qualified = context['selected']['selected']['packages'], {}
    for record in context['origin_records']:
        require(record['origins']['packages'] == packages, 'qualified origin packages differ')
        for path, digest in record['origins']['files'].items():
            require(record['input_guards'].get(path) == digest, 'qualified origin lacks authenticated guard: ' + path)
            require(qualified.setdefault(path, digest) == digest, 'conflicting qualified origin: ' + path)
    return qualified


def check_origins(context, origins):
    require(origins['packages'] == context['selected']['selected']['packages'], 'actual origin packages differ')
    qualified = qualified_origins(context)
    require(all(qualified.get(p) == h for p, h in origins['files'].items()), 'actual origin outside admitted qualified union')


def native_start(context):
    selected, trainer = context['selected'], context['trainer']
    source, prior = selected['source_driver'], selected['selected']['source_cpu']
    qualified_origins(context)
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sys.flags.optimize == 0 and
            re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'original CUDA-hidden unoptimized unit required')
    python = Path(sys.executable).resolve()
    require(str(python) == prior['invocation']['python'] and
            bound_file(context['guards'], python, prior['invocation']['python_sha256']) == python and
            sys.version == prior['invocation']['python_version'], 'qualified interpreter differs')
    before = source.cgroup_memory(); unit = Path(before['path']).name.removesuffix('.service')
    require(unit not in {t['unit'] for t in context['terminals']} and
            os.environ['INVOCATION_ID'] not in {t['invocation_id'] for t in context['terminals']}, 'distinct evaluator unit required')
    context['admission'].init.admit_cgroup(before, unit); context['helper'].zero_events(before)
    trainer.prepare_native(selected)
    import torch
    require(not torch.cuda.is_initialized() and not torch.cuda.is_available(), 'CPU evaluator initialized CUDA')
    raw, context['features'] = trainer.canonical_features(selected)
    del raw
    context['feature_state_sha256'] = selected['original'].fingerprint(context['features'])
    context['typed_encoder'] = trainer.encoder_metadata(selected)
    context['packing'] = selected['packing']
    check_origins(context, source.imported_origins(selected['extract'], selected['selected']['packages']))
    return before


def check_saved_metadata(context, saved, endpoint):
    """The complete typed payload must pass before any readout member is used."""
    trainer, selected = context['trainer'], context['selected']
    original, ident = selected['original'], saved['identity']
    record = context['records'][endpoint['seed'], endpoint['arm']]
    require(ident == record['identity'], 'typed payload identity/receipt differs')
    trainer.check_payload(saved, ident, 1000)
    require(saved['encoder'] == selected['encoder'] and
            saved['partition'] == context['partition'] and
            original.fingerprint({k: saved[k] for k in trainer.STATIC_KEYS}) == ident['static_sha256'] == selected['static_sha256'] and
            original.fingerprint({k: saved[k] for k in ('head', 'classifier')}) == ident['complement_sha256'] ==
                original.fingerprint({k: selected['initial'][k] for k in ('head', 'classifier')}) and
            original.fingerprint(saved['means']) == ident['means_sha256'] ==
                selected['terminals']['cpu:control']['arms'][endpoint['arm']]['identity']['means_sha256'] and
            original.fingerprint(saved['encoder']) == ident['encoder_sha256'] and
            original.fingerprint(saved['buffers']) == ident['buffers_sha256'] and
            original.fingerprint({k: saved['head'][k] for k in ('center', 'preactivation_std')}) == ident['head_buffers_sha256'] and
            context['feature_state_sha256'] == ident['features_sha256'], 'complete terminal frozen/means/source/TRAIN binding differs')
    config, buffers = context['typed_encoder']
    require(original.fingerprint(saved['config']) == original.fingerprint(config) and
            original.fingerprint(saved['buffers']) == original.fingerprint(buffers), 'typed config/nonpersistent buffers differ')
    frozen = {k: saved[k] for k in ('encoder', 'config', 'buffers', 'head', 'classifier', 'means', *trainer.STATIC_KEYS)}
    frozen['features'] = context['features']
    require(original.fingerprint(frozen) == ident['frozen_sha256'], 'complete frozen typed state differs')


def check_complete_payload(context, saved, endpoint, consumed=None):
    check_saved_metadata(context, saved, endpoint)
    original = context['selected']['original']
    require(original.fingerprint(saved, consumed=consumed) == endpoint['terminal_state_sha256'],
            'complete serialized typed state differs')


def load_head(context, endpoint):
    import torch
    selected, trainer = context['selected'], context['trainer']
    trainer.require_no_model(selected)
    original = selected['original']
    path = bound_file(context['guards'], endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    with path.open('rb') as stream:
        pages = original.CheckpointPages(stream)
        check_complete_payload(context, saved, endpoint, consumed=pages.consume)
        trainer.finite_tree(saved)
        head = selected['selected']['cached'].head_from('control', tensors=saved['head']).eval().requires_grad_(False)
        A = torch.nn.Parameter(pages.copy(saved['A']), requires_grad=True)
        means = {k: pages.copy(v) for k, v in saved['means'].items()}
        require(A.detach().count_nonzero().item() > 0, 'updated A must be nonzero')
        readout = {'head': head, 'A': A, 'means': means, 'arm': endpoint['arm']}
        selected['quadratic']._check_base(head, 'cpu')
        trainer.claim_model(selected, head)
        digest = readout_digest(context, readout)
        facts = {'payload_sha256': endpoint['terminal_state_sha256'], 'readout_sha256': digest,
                 **{k: saved['identity'][k] for k in ('encoder_sha256', 'complement_sha256', 'means_sha256', 'features_sha256')}}
    del saved, head, A, means
    gc.collect()
    return readout, facts


def readout_digest(context, readout):
    return context['selected']['original'].fingerprint({'head': dict(readout['head'].state_dict()),
                                                       'A': readout['A'], 'means': readout['means']})


def head_values(context, readout, cache):
    import torch
    head = readout['head']
    require(all(not m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in head.modules()) and readout['A'].grad is None, 'independent readout modes/hooks/grad differ')
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        raw = context['selected']['quadratic'].raw_features(cache, head, readout['A'], readout['means'], readout['arm'])
        values = context['trainer'].packed_outputs(context['selected'], raw)
    return tuple(values[k] for k in ('raw', 'unit', 'codes', 'inverse_norms'))


def release_head(context, readout):
    readout.clear(); gc.collect()
    context['trainer'].require_no_model(context['selected'])


def value_facts(context, values):
    return {k: context['selected']['source_driver'].tensor_fact(v) for k, v in
            zip(('raw', 'unit', 'codes', 'inverse_norms'), values, strict=True)}


def qualify_heads(context):
    cache, original = context['features'][:64], context['selected']['original']
    witnesses, facts = {}, {}
    for endpoint in context['spec']['endpoints']:
        key = label(endpoint)
        readout, facts[key] = load_head(context, endpoint)
        values = head_values(context, readout, cache); witnesses[key] = value_facts(context, values)
        require(readout_digest(context, readout) == facts[key]['readout_sha256'], 'TRAIN witness changed readout')
        release_head(context, readout)
        readout, second_facts = load_head(context, endpoint)
        second = head_values(context, readout, cache)
        context['helper'].exact(values, second)
        require(original.fingerprint(values) == original.fingerprint(second) and second_facts == facts[key] and
                readout_digest(context, readout) == second_facts['readout_sha256'], 'independent TRAIN readout/wire bits differ')
        release_head(context, readout); del values, second
        gc.collect()
    return {'head_facts': facts, 'train_witnesses': witnesses}


def source_file_names():
    return {'source-179061' + suffix for suffix in ('.raw.npy', '.unit.npy', '.packed.bin')}


def source_values(context, cache):
    selected = context['selected']
    context['trainer'].require_no_model(selected)
    head = selected['selected']['cached'].head_from('control', tensors=selected['initial']['head']).eval().requires_grad_(False)
    digest = selected['original'].fingerprint(dict(head.state_dict()))
    values = context['legacy'].head_values(context, head, cache)
    require(selected['original'].fingerprint(dict(head.state_dict())) == digest, 'source forward changed frozen head')
    del head
    gc.collect()
    return values


def archived_source_wires(context):
    import numpy as np
    import torch
    from torch.nn import functional as F
    files, arrays = SOURCE_INVENTORY['files'], []
    for suffix in ('.raw.npy', '.unit.npy'):
        descriptor = files['control-179061' + suffix]
        path = bound_file(context['guards'], descriptor['path'], descriptor['sha256'])
        require(path.stat().st_size == descriptor['bytes'], 'archived source wire size differs')
        array = np.load(path, allow_pickle=False)
        require(array.shape == (3449, 128) and array.dtype == np.float32 and np.isfinite(array).all(),
                'complete archived raw/unit layout differs')
        arrays.append(torch.from_numpy(array))
    raw, unit = arrays
    require((raw.norm(dim=1) > 0).all().item(), 'source raw descriptors must be nonzero')
    context['helper'].exact((F.normalize(raw, dim=1),), (unit,))
    require(context['selected']['original'].fingerprint(F.normalize(raw, dim=1)) ==
            context['selected']['original'].fingerprint(unit), 'source raw/unit normalization bits differ')
    packed = context['packing'].pack_int8_unit_embeddings(unit)
    descriptor = files['control-179061.packed.bin']
    require(Path(descriptor['path']).stat().st_size == descriptor['bytes'], 'archived packed source size differs')
    packed_readback(context['guards'], descriptor['path'], descriptor['sha256'], packed.to_bytes())
    return raw, unit, packed.codes, packed.inverse_norms


def replay_archived_source(context, fixed):
    import torch
    panel = context['partition']['panels']['selection']
    require(tuple(map(len, (panel['original_rows'], panel['query'], panel['gallery'], panel['original_class_ids']))) == PANELS['selection'],
            'complete source selection panel mapping differs')
    labels = tuple(context['fit']['class_names'][context['fit']['targets'][r]] for r in panel['original_rows'])
    values = archived_source_wires(context)
    quality = fixed.packed_quality(values[1].numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu'))
    replay_equal(context['source_record']['quality']['179061']['control'], quality)
    check_quality(quality, 1734)
    return quality


def write_wires(context, key, values):
    import numpy as np
    files = {}
    for suffix, value in (('.raw.npy', values[0]), ('.unit.npy', values[1])):
        path = context['args'].output / (key + suffix)
        with path.open('xb') as stream:
            np.save(stream, value.numpy(), allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = context['args'].output / (key + '.packed.bin')
    with path.open('xb') as stream:
        stream.write(context['packing'].pack_int8_unit_embeddings(values[1]).to_bytes()); stream.flush(); os.fsync(stream.fileno())
    files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return files


def readback_wires(context, key, files, values):
    import numpy as np
    import torch
    for suffix, value in (('.raw.npy', values[0]), ('.unit.npy', values[1])):
        name = key + suffix
        path = bound_file(context['guards'], context['args'].output / name, files[name])
        loaded = torch.from_numpy(np.load(path, allow_pickle=False))
        context['helper'].exact((loaded,), (value,))
        require(context['selected']['original'].fingerprint(loaded) == context['selected']['original'].fingerprint(value),
                'serialized raw/unit wire bits differ')
    name = key + '.packed.bin'
    packed_readback(context['guards'], context['args'].output / name, files[name],
                    context['packing'].pack_int8_unit_embeddings(values[1]).to_bytes())


def score_panel(context, cpu):
    import numpy as np
    import torch
    spec, original = context['spec'], context['selected']['original']
    fixed = scoring_math(context)
    replay_archived_source(context, fixed)  # Original per-query replay closes BEFORE any new decision.
    panel = context['partition']['panels'][spec['panel']]
    require(tuple(map(len, (panel['original_rows'], panel['query'], panel['gallery'], panel['original_class_ids']))) == PANELS[spec['panel']],
            'complete authorized panel mapping differs')
    cache = cache_rows(context, panel['original_rows'])  # Validation only after panel_authority accepted exact selection GO.
    labels = tuple(context['fit']['class_names'][context['fit']['targets'][r]] for r in panel['original_rows'])
    source_first = source_values(context, cache)
    source_second = source_values(context, cache)
    context['helper'].exact(source_first, source_second)
    require(original.fingerprint(source_first) == original.fingerprint(source_second), 'independent source raw/unit/packed bits differ')
    if spec['panel'] == 'selection':
        archived = archived_source_wires(context)
        context['helper'].exact(source_second, archived)
        require(original.fingerprint(source_second) == original.fingerprint(archived), 'fresh source061/archived selection arithmetic differs')
        del archived
    source_quality = fixed.packed_quality(source_second[1].numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu'))
    if spec['panel'] == 'selection':
        replay_equal(context['source_record']['quality']['179061']['control'], source_quality)
    files = write_wires(context, 'source-179061', source_first)
    readback_wires(context, 'source-179061', files, source_second)
    source_facts = value_facts(context, source_second)
    del source_first, source_second
    quality, replay_facts = {}, {}
    for endpoint in spec['endpoints']:
        key, seed, arm = label(endpoint), str(endpoint['seed']), endpoint['arm']
        readout, facts = load_head(context, endpoint)
        require(facts == cpu['head_facts'][key], 'CPU-qualified complete scoring readout differs')
        values = head_values(context, readout, cache)
        require(readout_digest(context, readout) == facts['readout_sha256'], 'panel forward changed readout')
        files.update(write_wires(context, key, values))
        first = fixed.packed_quality(values[1].numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu'))
        release_head(context, readout)
        readout, second_facts = load_head(context, endpoint)
        second = head_values(context, readout, cache)
        context['helper'].exact(values, second)
        require(original.fingerprint(values) == original.fingerprint(second) and second_facts == facts and
                readout_digest(context, readout) == facts['readout_sha256'], 'independent full panel readout/wire bits differ')
        readback_wires(context, key, files, second)
        second_quality = fixed.packed_quality(second[1].numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu'))
        replay_equal(first, second_quality)
        quality.setdefault(seed, {})[arm] = first; replay_facts[key] = value_facts(context, second)
        release_head(context, readout); del values, second
        gc.collect()
    _, average = averaged_deltas(quality, spec['stage'], spec['panel'])
    intervals = {}
    if spec['stage'] == 'full':
        for metric in METRICS:
            delta = np.asarray(average[metric]); intervals[metric] = {'mean_delta': float(delta.mean())}
            for kind, groups in (('product', np.asarray(labels)[panel['query']]), ('query', np.arange(len(panel['query'])))):
                # Pinned helper resets seed179019 every call: the SAME5000 draws across metrics/signs.
                intervals[metric][kind + '_lower95'] = fixed.bootstrap_lower(delta, groups)
                intervals[metric][kind + '_upper95'] = -fixed.bootstrap_lower(-delta, groups)
    decision = decide(quality, source_quality, spec['stage'], spec['panel'], intervals, context['costs'])
    return {**decision, 'quality': quality, 'source_quality': source_quality, 'source_quality_panel': spec['panel'],
            'source_selection_receipt': SOURCE_INVENTORY['receipt'],
            'source_checkpoint': SOURCE_INVENTORY['baseline_endpoint']['checkpoint'],
            'source_terminal_state_sha256': SOURCE_INVENTORY['baseline_endpoint']['terminal_state_sha256'],
            'archived_selection_per_query_exact': True, 'source_panel_facts': source_facts,
            'cost': context['costs'], 'cost_policy': COST_POLICY, 'paired_seed_average_intervals': intervals,
            'bootstrap_draws': 0 if spec['stage'] == 'first' else 5000, 'bootstrap_seed': 179019,
            'panel_images': len(panel['original_rows']), 'query_images': len(panel['query']), 'gallery_images': len(panel['gallery']),
            'panel_products': len(panel['original_class_ids']), 'fit_images': 6355, 'fit_products': 1008,
            'quality_read': True, 'files': files, 'panel_replay_facts': replay_facts,
            'full_panel_raw_unit_packed_replay_exact': True, 'per_query_replay_exact': True,
            'metric_units': 'fractions; multiply deltas by100 for percentage points',
            'interval_scope': 'equal-seed per-query deltas; shared paired product/query draws conditional on frozen source',
            'cost_denominator': 'fresh original control TRAIN1000 whole service and median update, eachseed',
            'optimization_throughput_is_image_training_throughput': False, 'independent_pretraining_seeds': False,
            'intermediate_checkpoint_selection': False, 'previous_official_and_pareto_preserved': True,
            'validation_quality_exposed': spec['panel'] == 'validation',
            'canonical_panel_input_only': True, 'serving_view_averaging': False,
            'preparation_costs': preparation_costs(context['selected']['selected'])}



def cache_rows(context, rows):
    import numpy as np
    import torch
    descriptor = context['partition']['original_cache']
    cache = np.load(bound_file(context['guards'], descriptor['path'], descriptor['sha256']), allow_pickle=False, mmap_mode='r')
    require(cache.shape == (13283, 1152) and cache.dtype == np.float32, 'original normalized F32 cache layout differs')
    values = cache[rows].copy()
    del cache
    require(np.isfinite(values).all() and np.allclose(np.linalg.norm(values, axis=1), 1, atol=1e-5, rtol=0), 'finite normalized cache rows required')
    return torch.from_numpy(values)

def exit_rehash(context):
    """Fresh conflict-checked union; retain every nested metadata predicate."""
    selected = context['selected']['selected']
    source, exporter = selected['source_driver'], selected['exporter']
    genuine, packages = selected['genuine'], selected['packages']
    prior = genuine['prior']
    qualified = qualified_origins(context)
    modules, origin_paths, native = {}, set(), set()
    for name, module in tuple(sys.modules.items()):
        if name.split('.')[0] not in packages:
            continue
        path = source.loaded_module_origin(name, module, packages)
        if path is not None:
            modules[name] = str(path)
            origin_paths.add(str(path))
    for line in Path('/proc/self/maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/') and '.so' in fields[5]:
            native.add(str(source.canonical(Path(fields[5]).resolve())))
    origin_paths.update(native)
    for path in origin_paths:
        require(path in qualified, 'actual native origin outside admitted qualified union: ' + path)

    union = {}
    for guards in (prior['guards'], genuine['guards'], context['guards']):
        for path, digest in guards.items():
            require(union.setdefault(path, digest) == digest, 'conflicting exit file authority: ' + path)
    for path in origin_paths:
        require(union.setdefault(path, qualified[path]) == qualified[path], 'conflicting qualified origin: ' + path)
    require(source.fit_rows(prior['extract'], prior['fit']) == prior['images'], 'FIT image resolution changed')
    root = Path(prior['fit']['dataset_root'])
    images = [(root / row['relative_path']).resolve() for row in prior['fit']['rows']]
    require(all(path.is_relative_to(root) for path in images), 'FIT image escaped dataset root')
    require(len(set(images)) == 13283, 'FIT resolved image aliases collide')
    require(images == prior['all_images'], 'FIT image resolution changed')
    for path, row in zip(images, prior['fit']['rows']):
        require(union.setdefault(str(path), row['image_sha256']) == row['image_sha256'],
                'conflicting FIT file authority: ' + str(path))

    # These small authenticated rereads preserve the nested exit predicates.
    require(source.bootstrap(prior['root'], prior['args'].execution_sha256)[1] == prior['code'],
            'exit closure differs')
    require(genuine['reference'].bootstrap(prior['own_root'], prior['export_args'].execution_sha256) == prior['own_code'],
            'exit exporter closure differs')
    require(exporter.closure(genuine['root'], genuine['args'].execution_sha256, exporter.FILES, {}) == genuine['code'],
            'exit genuine closure differs')
    manifest = exporter.selected_manifest(exporter.file_json(genuine['launch']['partition'], {}), prior['fit'])
    manifest['resolved_paths'] = [str(prior['all_images'][r]) for r in manifest['original_rows']]
    require(manifest == genuine['selected'], 'exit TRAIN mapping changed')
    exporter.image_rows_node(genuine['launch']['image_rows']['path'])
    for path, digest in sorted(union.items()):
        bound_file({}, path, digest)
    return {'packages': packages, 'modules': modules, 'native_files': sorted(native),
            'files': {path: union[path] for path in sorted(origin_paths)}}


def run(args):
    prior = None if args.prerequisite is None else {'path': str(args.prerequisite), 'sha256': args.prerequisite_sha256}
    require(sys.argv == cli_argv(args.authority, args.authority_sha256, args.execution_sha256, args.phase, args.output, prior),
            'fixed canonical CLI order required')
    print(json.dumps({'progress': 'authority_start', 'seconds': time.perf_counter() - UNIT_STARTED}), flush=True)
    context = authority(args); cpu = prerequisites(context)
    print(json.dumps({'progress': 'authority_end', 'seconds': time.perf_counter() - UNIT_STARTED}), flush=True)
    before = native_start(context)
    import torch
    source = context['selected']['source_driver']
    rng, flags = torch.random.get_rng_state().clone(), source.numerical_flags()
    args.output.mkdir()
    facts = qualify_heads(context)
    if cpu is not None:
        require(all(facts[k] == cpu[k] for k in facts), 'accepted evaluator CPU witnesses differ')
    result = score_panel(context, cpu) if args.phase == 'score' else {'quality_read': False, 'files': {}, 'validation_quality_exposed': False}
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags and not torch.cuda.is_initialized(),
            'whole-unit RNG/flags/CUDA differs')
    context['trainer'].require_no_model(context['selected'])
    print(json.dumps({'progress': 'exit_rehash_start', 'seconds': time.perf_counter() - UNIT_STARTED}), flush=True)
    origins = exit_rehash(context); check_origins(context, origins)
    require(context['trainer'].composition(context['selected']) == context['selected']['encoder'], 'exit encoder composition differs')
    require(closure(context['root'], args.execution_sha256, FILES, {}) == context['code'] and
            closure(Path(context['spec']['training']['root']), context['spec']['training']['execution_sha256'], TRAIN_FILES, {}) == context['spec']['training']['code'] and
            closure(Path(REFERENCE_ROOT), REFERENCE_EXECUTION_SHA, REFERENCE_PINS.keys(), {}) == REFERENCE_PINS,
            'exit separate evaluator/trainer/reference closure differs')
    resources = context['helper'].resources(context, args.phase, before)
    receipt = {**bind(context), 'schema': SCHEMA, 'phase': args.phase, 'pass': True, 'engineering_admission_pass': True,
        'certificate': 'updated cached readout composed with qualified immutable encoder', 'public_encoder_qualified': False,
        'authority': {'path': str(args.authority), 'sha256': args.authority_sha256}, 'output': str(args.output),
        'prerequisite': prior, 'cpu_terminal': context['cpu_terminal'], 'optimizer_updates': 0, 'optimizer_members': 1,
        'numerical_flags': flags, 'strict_independent_head_reload_exact': True, 'train_raw_unit_cpu_packed_exact': True,
        'complete_typed_terminal_state_exact': True, 'first_heads_released_before_reload': True,
        'rng_flags_preserved': True, 'cuda_initialized': False, 'official_read': False,
        'global_production_goal_met': False, 'public_latency_measured': False,
        'input_guards': context['guards'], 'origins': origins, 'exit_rehash_pass': True,
        'invocation': {'argv': sys.argv, 'python': str(Path(sys.executable).resolve()),
            'python_sha256': context['selected']['selected']['source_cpu']['invocation']['python_sha256'],
            'python_version': sys.version, 'optimize': sys.flags.optimize, 'pid': os.getpid(),
            'invocation_id': os.environ['INVOCATION_ID'], 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
            'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}, **facts, **result, **resources}
    check_receipt(context, receipt, args.phase)
    context['helper'].publish(args.output / 'receipt.json', receipt)
    require(time.perf_counter() - UNIT_STARTED < policy(args.phase)['seconds'], 'receipt included whole-unit cap differs')
    return receipt

def main():
    args = parser().parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Quadratic readout evaluation rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output), 'decision': result.get('decision')}))


if __name__ == '__main__':
    main()
