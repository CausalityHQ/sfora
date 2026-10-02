#!/usr/bin/env python3
"""Prototype residual ridge evaluator; all native gates remain UNRUN.

Own execution.json contains exactly FILES; parent supplies actual training2
and original evaluator2 roots/execution hashes. Immutable original source-v7
has exactly four files, and the pinned scoring reference has exactly six.
Authority keys are SPEC_KEYS, with FILE/UNIT descriptors identical to the
original complete original-service admission. Endpoint order is linear then
quadratic; neither endpoint carries a seed, optimizer or update schedule.
CPU120 independently reloads each complete fitted payload twice without fitting.
Score300 first authenticates engineering/paired costs, then replays every source
selection query before scoring the authorized direct-FIT panel and its wires.
GO permits sealed validation only; all inference is conditional on one frozen
trained source. Image training throughput/public quality/speed remain unmet.
Both locks, CUDA hidden, 8GiB/noSwap/zero events and original footers are required.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import time
UNIT_STARTED = time.perf_counter()

import argparse
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

SCHEMA = 'siglip2-prototype-residual-evaluation-v1'
AUTHORITY_SCHEMA = 'siglip2-prototype-residual-evaluation-authority-v1'
FILES = {'evaluate_siglip2_prototype_residual.py', 'test_siglip2_prototype_residual_evaluation.py'}
TRAIN_FILES = {'fit_siglip2_prototype_residual.py', 'test_siglip2_prototype_residual.py'}
ARMS = ('linear', 'quadratic')
ORIGINAL_REFERENCE = {'root': '/home/riomus/runs/sfora-so400-quadratic-readout-source-v7',
                      'execution_sha256': '84414302a6749842cae20e2608e932066334916218df68f91f10fd3c589ca382'}
ORIGINAL_PINS = {
    'train_siglip2_quadratic_readout.py': '17cf2dac93a45aae3ba929ad5c36e73f4c33576eae9bb5f9dec670c272a1731f',
    'test_siglip2_quadratic_readout.py': '1410060ac38065f39aa8e3f0a43fe47fb42321331bb00c9d43264381c4a23ba8',
    'quadratic_readout.py': '12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6',
    'quadratic_encoder_frames.py': '1928b2882b771f5c57437420755c1dcdb96fd119981b02077b275c2ba9b2e8db'}
EVALUATOR_PINS = {
    'evaluate_siglip2_quadratic_readout.py': '7bfaabeb855abecbfa87664c4dfc9381c1213196ffc5a40fc1bf60b2caacc1f8',
    'test_siglip2_quadratic_readout_evaluation.py': '1ea5312edddfc162bedf94859839e3b02d50433b67a7a250fa054e442b9155d9'}
COST_POLICY = {'whole_service_ratio_max': 1.50, 'total_fit_core_ratio_max': 1.50,
               'shared_export_seconds': 283.636, 'shared_export_in_fit_ratios': False}
SPEC_KEYS = {'schema', 'execution_sha256', 'training', 'original_reference', 'original_evaluator',
             'evaluation_reference', 'partition', 'source_selection', 'panel', 'endpoints',
             'selection_go', 'resource_policies', 'cost_policy', 'both_locks_held',
             'selection_previously_exposed', 'validation_previously_exposed'}

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

PANELS = {'selection': (3449, 1734, 1715, 498), 'validation': (3479, 1749, 1730, 498)}

METRICS = ('per_query_r1', 'per_query_ap')

NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}

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

EVALUATION_REFERENCE = {'root': REFERENCE_ROOT, 'execution_sha256': REFERENCE_EXECUTION_SHA}



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



def unique_terminals(terminals):
    unique = {t['receipt']['path']: t for t in terminals}
    require(all(unique[t['receipt']['path']] == t for t in terminals) and
            len({t['unit'] for t in unique.values()}) == len(unique) and
            len({t['invocation_id'] for t in unique.values()}) == len(unique), 'distinct original whole units required')



def replay_equal(expected, actual):
    require(actual == expected, 'independent per-query packed quality replay differs')



def packed_readback(guards, path, digest, expected):
    require(bound_file(guards, path, digest).read_bytes() == expected, 'independent packed wire differs')



def check_file_descriptor(value):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'} and
            type(value['path']) is str and Path(value['path']).is_absolute() and
            type(value['sha256']) is str and re.fullmatch('[0-9a-f]{64}', value['sha256']),
            'exact actual FILE descriptor required')



def check_quality(value, count):
    require(value.keys() == {'recall_at_1', 'map_at_r', *METRICS} and
            len(value['per_query_r1']) == len(value['per_query_ap']) == count and
            all(type(v) in (int, float) and math.isfinite(v) and v in (0, 1) for v in value['per_query_r1']) and
            all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in value['per_query_ap']) and
            all(type(value[k]) in (int, float) and math.isfinite(value[k]) and
                math.isclose(value[k], statistics.mean(value[m]), rel_tol=0, abs_tol=1e-12)
                for k, m in zip(('recall_at_1', 'map_at_r'), METRICS, strict=True)),
            'complete finite per-query metrics/aggregate required')



def check_value_facts(facts, count):
    require(facts.keys() == {'raw', 'unit', 'codes', 'inverse_norms'} and all(
        facts[n].keys() == {'shape', 'dtype', 'sha256'} and facts[n]['shape'] == shape and
        facts[n]['dtype'] == dtype and re.fullmatch('[0-9a-f]{64}', facts[n]['sha256']) for n, shape, dtype in
        (('raw', [count, 128], 'torch.float32'), ('unit', [count, 128], 'torch.float32'),
         ('codes', [count, 128], 'torch.int8'), ('inverse_norms', [count], 'torch.float16'))),
        'complete raw/unit/int8/inverse-bit facts required')


def check_code_descriptor(value, names, pins=None):
    require(isinstance(value, dict) and value.keys() == {'root', 'execution_sha256', 'code'} and
            type(value['root']) is str and Path(value['root']).is_absolute() and
            re.fullmatch('[0-9a-f]{64}', value['execution_sha256']) and
            value['code'].keys() == set(names) and all(type(v) is str and re.fullmatch('[0-9a-f]{64}', v)
            for v in value['code'].values()) and (pins is None or value['code'] == pins),
            'complete actual code closure required')


def check_spec(spec, args):
    require(spec.keys() == SPEC_KEYS and spec['schema'] == AUTHORITY_SCHEMA and
            spec['execution_sha256'] == args.execution_sha256 and spec['both_locks_held'] is True and
            spec['resource_policies'] == {p: policy(p) for p in ('cpu', 'score')} and
            spec['cost_policy'] == COST_POLICY and spec['selection_previously_exposed'] is True and
            spec['validation_previously_exposed'] is False, 'fixed evaluation authority differs')
    check_code_descriptor(spec['training'], TRAIN_FILES)
    check_code_descriptor(spec['original_evaluator'], EVALUATOR_PINS, EVALUATOR_PINS)
    require(spec['original_reference'] == ORIGINAL_REFERENCE and
            spec['evaluation_reference'] == EVALUATION_REFERENCE, 'immutable source/reference differs')
    check_file_descriptor(spec['partition'])
    source = spec['source_selection']
    require(source.keys() == {'inventory', 'terminal'}, 'complete source selection required')
    check_file_descriptor(source['inventory'])
    require(spec['partition']['sha256'] == PARTITION_SHA and
            source['inventory']['sha256'] == SOURCE_INVENTORY_SHA and
            source['terminal'] == SOURCE_SCORE_TERMINAL, 'original source selection differs')
    require(spec['panel'] in PANELS and (spec['selection_go'] is None) == (spec['panel'] == 'selection'),
            'validation requires original selection GO')
    require([e['arm'] for e in spec['endpoints']] == list(ARMS), 'linear then quadratic endpoints required')
    for endpoint in spec['endpoints']:
        require(endpoint.keys() == {'arm', 'launch', 'terminal', 'checkpoint', 'terminal_state_sha256'} and
                type(endpoint['terminal_state_sha256']) is str and
                re.fullmatch('[0-9a-f]{64}', endpoint['terminal_state_sha256']), 'exact new ridge endpoint required')
        for key in ('launch', 'checkpoint'):
            check_file_descriptor(endpoint[key])
        check_terminal_descriptor(endpoint['terminal'])
        require(Path(endpoint['checkpoint']['path']) == Path(endpoint['terminal']['receipt']['path']).parent / 'resume.pt',
                'complete fitted checkpoint role differs')
    unique_terminals([e['terminal'] for e in spec['endpoints']])
    if spec['selection_go'] is not None:
        check_terminal_descriptor(spec['selection_go'])


def check_prior_binding(current, prior):
    require(prior['panel'] == 'selection' and prior['selection_go'] is None and
            all(prior[k] == current[k] for k in SPEC_KEYS - {'panel', 'selection_go'}),
            'original selection authority/checkpoints differ')


def metric_deltas(quality, source, panel):
    require(panel in PANELS and quality.keys() == set(ARMS), 'complete matched panel arms required')
    count = PANELS[panel][1]
    for value in (source, *quality.values()):
        check_quality(value, count)
    return {name: {metric: [b - a for a, b in zip(left[metric], right[metric], strict=True)] for metric in METRICS}
            for name, left, right in (
                ('quadratic_minus_linear', quality['linear'], quality['quadratic']),
                ('quadratic_minus_source', source, quality['quadratic']),
                ('linear_minus_source', source, quality['linear']))}


def immediate_quality_pass(quality, source, panel):
    deltas = metric_deltas(quality, source, panel)
    pair = deltas['quadratic_minus_linear']
    return statistics.mean(pair[METRICS[0]]) > 0 and statistics.mean(pair[METRICS[1]]) >= 0 and all(
        statistics.mean(quality['quadratic'][m]) >= statistics.mean(source[m]) for m in METRICS)


def paired_cost(records):
    require(records.keys() == set(ARMS), 'complete matched original fit costs required')
    for record in records.values():
        require(all(type(record[k]) in (int, float) and math.isfinite(record[k]) and record[k] > 0
                    for k in ('service_seconds', 'total_fit_core_seconds')) and
                record['total_fit_core_seconds'] <= record['service_seconds'], 'positive complete original fit cost required')
    ratios = {name: records['quadratic'][key] / records['linear'][key] for name, key in (
        ('whole_service_ratio', 'service_seconds'), ('total_fit_core_ratio', 'total_fit_core_seconds'))}
    require(all(math.isfinite(v) and v > 0 for v in ratios.values()), 'finite paired cost ratios required')
    return {**ratios, 'pass': all(v <= 1.50 for v in ratios.values()),
            **{arm: {k: records[arm][k] for k in ('service_seconds', 'total_fit_core_seconds')} for arm in ARMS}}


def decide(quality, source, panel, intervals, costs):
    deltas = metric_deltas(quality, source, panel)
    immediate = immediate_quality_pass(quality, source, panel)
    require(intervals.keys() == (set(METRICS) if immediate else set()),
            'complete survivor intervals; immediate KILL has no intervals')
    for metric, value in intervals.items():
        require(value.keys() == {'mean_delta', 'product_lower95', 'product_upper95', 'query_lower95', 'query_upper95'} and
                all(type(v) in (int, float) and math.isfinite(v) and -1 <= v <= 1 for v in value.values()) and
                all(value[k + '_lower95'] <= value[k + '_upper95'] for k in ('product', 'query')) and
                math.isclose(value['mean_delta'], statistics.mean(deltas['quadratic_minus_linear'][metric]),
                             rel_tol=0, abs_tol=1e-12), 'interval mean/finite bounds differ from per-query replay')
    require(costs == paired_cost({a: costs[a] for a in ARMS}), 'original paired cost decision differs')
    quality_pass = immediate and all(v['mean_delta'] >= .002 and v['product_lower95'] > 0 for v in intervals.values())
    return {'decision': 'GO' if quality_pass and costs['pass'] else 'KILL',
            'immediate_quality_pass': immediate, 'quality_pass': bool(quality_pass), 'cost_pass': costs['pass'],
            'source_floor_pass': all(statistics.mean(quality['quadratic'][m]) >= statistics.mean(source[m]) for m in METRICS),
            'quadratic_source_gain_both': all(statistics.mean(deltas['quadratic_minus_source'][m]) > 0 for m in METRICS),
            'deltas': deltas,
            'aggregate_deltas': {name: {m: statistics.mean(values[m]) for m in METRICS} for name, values in deltas.items()},
            'selection_go_admits_validation_only': panel == 'selection' and quality_pass and costs['pass'],
            'global_production_goal_met': False, 'product_go': False}


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


def cli_argv(authority, authority_sha, execution_sha, phase, output, prerequisite):
    value = [str(Path(__file__).absolute()), '--execution-sha256', execution_sha,
             '--authority', str(authority), '--authority-sha256', authority_sha,
             '--phase', phase, '--output', str(output)]
    if prerequisite is not None:
        value += ['--prerequisite', prerequisite['path'], '--prerequisite-sha256', prerequisite['sha256']]
    return value


def file_names():
    return {arm + suffix for arm in (*ARMS, 'source-179061') for suffix in ('.raw.npy', '.unit.npy', '.packed.bin')}


def preparation_costs(context):
    original = context['baseline'].preparation_costs(context['selected']['selected'])
    shared = original['shared']
    require(shared['whole_service_seconds'] == COST_POLICY['shared_export_seconds'], 'original shared export cost differs')
    return {**original, 'arm_attributed': {arm: {'shared_whole_service_seconds': shared['whole_service_seconds'],
            'additional_preparation_jobs': 0} for arm in ARMS}, 'included_in_fit_cost_ratios': False}


def bind(context):
    return {'authority_sha256': context['args'].authority_sha256,
            'execution_sha256': context['args'].execution_sha256, 'spec': context['spec'],
            'selection_previously_exposed': True, 'validation_previously_exposed': False,
            'source': context['fitting']['source'], 'source_code': context['code']}


def merge_guards(target, values):
    for path, digest in values.items():
        require(target.setdefault(path, digest) == digest, 'conflicting admitted file authority: ' + path)


def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native imports preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    spec = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_spec(spec, args); check_output(args.output)
    training, old_evaluator = spec['training'], spec['original_evaluator']
    roots = tuple(map(Path, (str(root), training['root'], old_evaluator['root'], ORIGINAL_REFERENCE['root'], REFERENCE_ROOT)))
    require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]) and
            all(not args.output.is_relative_to(p) and not p.is_relative_to(args.output) for p in roots),
            'separate immutable evaluator2/fitter2/original4/evaluator2/reference6 required')
    for item, names, expected in (
        (training, TRAIN_FILES, training['code']), (old_evaluator, EVALUATOR_PINS, EVALUATOR_PINS),
        (ORIGINAL_REFERENCE, ORIGINAL_PINS, ORIGINAL_PINS), (EVALUATION_REFERENCE, REFERENCE_PINS, REFERENCE_PINS)):
        require(closure(Path(item['root']), item['execution_sha256'], names, guards) == expected,
                'complete actual pinned source closure differs')
    fitter = load_bare('_prototype_evaluation_fitter', Path(training['root']) / 'fit_siglip2_prototype_residual.py',
                       training['code']['fit_siglip2_prototype_residual.py'])
    require(fitter.FILES == TRAIN_FILES and fitter.SCHEMA == 'siglip2-prototype-residual-ridge-v1' and
            fitter.ARMS == ARMS and fitter.ORIGINAL_REFERENCE == ORIGINAL_REFERENCE and
            fitter.ORIGINAL_CODE == ORIGINAL_PINS and fitter.PARTITION_SHA == PARTITION_SHA and
            fitter.RECIPE['intercept'] is False and fitter.RECIPE['fit_passes'] == 2,
            'new deterministic prototype fitter profile differs')
    baseline = load_bare('_prototype_original_evaluator', Path(old_evaluator['root']) / 'evaluate_siglip2_quadratic_readout.py',
                         EVALUATOR_PINS['evaluate_siglip2_quadratic_readout.py'])
    require(baseline.REFERENCE_PINS == REFERENCE_PINS and baseline.SOURCE_INVENTORY == SOURCE_INVENTORY and
            baseline.SOURCE_SCORE_TERMINAL == SOURCE_SCORE_TERMINAL, 'original scoring/source profile differs')
    reference = load_bare('_prototype_evaluation_reference', Path(REFERENCE_ROOT) / 'evaluate_siglip2_cached_readout.py',
                          REFERENCE_PINS['evaluate_siglip2_cached_readout.py'])
    helper = load_bare('_prototype_evaluation_helper', Path(REFERENCE_ROOT) / 'export_siglip2_substrate_adaptation.py',
                       REFERENCE_PINS['export_siglip2_substrate_adaptation.py'])
    first = spec['endpoints'][0]
    fitting = fitter.authority(SimpleNamespace(execution_sha256=training['execution_sha256'],
        authority=Path(first['launch']['path']), authority_sha256=first['launch']['sha256'],
        phase='fit', arm='linear', output=args.output))
    selected, old = fitting['legacy'], fitting['legacy']['selected']
    require(selected['code'] == ORIGINAL_PINS and fitting['code'] == training['code'], 'admitted actual source/fitter differs')
    partition = read_json(spec['partition'], fitting['guards'])
    require(partition == old['partition'] and spec['partition'] == fitting['launch']['partition'] and
            partition['original_cache']['sha256'] == FIT_SHA, 'original partition/direct-FIT cache differs')
    bound_file(fitting['guards'], partition['original_cache']['path'], FIT_SHA)
    new_cpu = fitting['terminals']['cpu:linear']
    records, branches = {}, []
    for endpoint in spec['endpoints']:
        launch = read_json(endpoint['launch'], guards)
        record = read_json(endpoint['terminal']['receipt'], guards)
        require(record['launch'] == launch and record['authority'] == endpoint['launch'] and
                record['authority_sha256'] == endpoint['launch']['sha256'] and
                record['checkpoint'] == endpoint['checkpoint'] and
                record['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and
                record['output'] == str(Path(endpoint['terminal']['receipt']['path']).parent) and
                Path(endpoint['terminal']['receipt']['path']).name == 'receipt.json' and
                launch['selected_cpu'] == fitting['launch']['selected_cpu'], 'original fit endpoint binding differs')
        # Admission branches share immutable source frames, with independent metadata
        # maps so a sibling's checkpoint is never silently required by its launch.
        legacy = {**selected, 'guards': dict(selected['guards']), 'invocations': set(selected['invocations']),
                  'terminal_cgroups': dict(selected['terminal_cgroups'])}
        branch = {**fitting, 'legacy': legacy, 'guards': dict(fitting['guards']),
                  'terminals': dict(fitting['terminals']), 'terminal_cgroups': dict(fitting['terminal_cgroups'])}
        accepted = fitter.admit_terminal(branch, endpoint['terminal'], 'fit', endpoint['arm'])
        require(accepted == record and all(record[k] == new_cpu['arms'][endpoint['arm']][k] for k in
                ('identity', 'fit_witness', 'terminal_state_sha256', 'output_witness_sha256')),
                'new CPU qualified exact fitted identity/state differs')
        cores = record['fit_core_seconds']
        require(type(cores) is list and len(cores) == 2 and
                all(type(v) in (int, float) and math.isfinite(v) and v > 0 for v in cores) and
                record['total_fit_core_seconds'] == sum(cores), 'both complete fit-core passes required')
        bound_file(branch['guards'], endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])
        records[endpoint['arm']] = {**record, 'service_seconds': endpoint['terminal']['service_seconds']}
        branches.append(branch)
    require(all(records['linear']['identity'][k] == records['quadratic']['identity'][k] for k in
                ('source', 'method', 'device', 'native_inventory', 'numerical_flags',
                 'warm_payload_sha256', 'frozen_sha256', 'zero_source_sha256')),
            'matched complete frozen TRAIN/source inputs differ')
    for branch in branches:
        merge_guards(fitting['guards'], branch['guards'])
        merge_guards(selected['guards'], branch['legacy']['guards'])
        selected['invocations'].update(branch['legacy']['invocations'])
        fitting['terminal_cgroups'].update(branch['terminal_cgroups'])
        fitting['terminals'].update(branch['terminals'])
    merge_guards(selected['guards'], fitting['guards']); merge_guards(selected['guards'], guards)
    source_cpu = old['genuine']['prior']['launch']['source_cpu']['so400']
    startup = old['exporter'].file_json(old['export_record']['startup_terminal'], selected['guards'])
    terminals = [fitting['launch']['warm_start']['terminal'], fitter.ORIGINAL_CPU['terminal'],
                 fitting['launch']['selected_cpu'], old['launch']['selected_cpu'], *old['launch']['selected_mechanics'].values(),
                 *[e['terminal'] for e in spec['endpoints']]]
    for proof in (source_cpu, startup, old['launch']['selected_export']):
        terminals.append({('receipt' if k == 'proof' else k): v for k, v in proof.items()})
    exported = old['export_record']
    original_cpu = read_json(fitter.ORIGINAL_CPU['terminal']['receipt'], selected['guards'])
    context = {'args': args, 'root': root, 'code': code, 'spec': spec, 'fitter': fitter, 'fitting': fitting,
               'baseline': baseline, 'trainer': fitting['old'], 'legacy': reference, 'selected': selected,
               'guards': selected['guards'], 'helper': helper, 'admission': selected['admission'], 'records': records,
               'fit': old['genuine']['prior']['fit'], 'partition': partition, 'terminals': terminals,
               'origin_records': [old['source_cpu'], selected['warm_record'], original_cpu,
                   {**exported, 'input_guards': {**exported['original_input_guards'], **exported['input_guards']}},
                   *old['terminals'].values(), *selected['terminals'].values(), *fitting['terminals'].values()],
               'selected_context': {'source': selected['source_driver'], 'initialized': {'init': selected['admission'].init}},
               'unit_started': UNIT_STARTED, 'costs': paired_cost(records)}
    preparation_costs(context)
    baseline.admit_source_selection(context)  # Unchanged original full source selection admission, before native.
    merge_guards(fitting['guards'], context['guards'])
    context['base_guards'] = {p: h for p, h in context['guards'].items() if p != str(args.authority)}
    if spec['panel'] == 'validation':
        context['selection_go'] = accept_selection(context, spec['selection_go'])
    context['required_guards'] = dict(context['guards'])
    require(all(not args.output.is_relative_to(Path(t['receipt']['path']).parent) and
                not Path(t['receipt']['path']).parent.is_relative_to(args.output) for t in context['terminals']),
            'immutable original terminal/output overlap')
    unique_terminals(context['terminals'])
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import during complete admission')
    return context


def native_start(context):
    selected, fitter, baseline = context['selected'], context['fitter'], context['baseline']
    source, prior = selected['source_driver'], selected['selected']['source_cpu']
    baseline.qualified_origins(context)
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import preceded complete admission')
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
    fitter.prepare_native(context['fitting'])
    import torch
    require(not torch.cuda.is_initialized() and not torch.cuda.is_available(), 'CPU evaluator initialized CUDA')
    raw, context['features'] = context['trainer'].canonical_features(selected)
    del raw
    context['feature_state_sha256'] = selected['original'].fingerprint(context['features'])
    context['typed_encoder'] = context['trainer'].encoder_metadata(selected)
    context['packing'] = selected['packing']
    merge_guards(context['guards'], context['fitting']['guards'])
    baseline.check_origins(context, source.imported_origins(selected['extract'], selected['selected']['packages']))
    return before


def load_head(context, endpoint):
    fitter, fitting = context['fitter'], context['fitting']
    context['trainer'].require_no_model(context['selected'])
    ident = context['records'][endpoint['arm']]['identity']
    state = fitter.reload(fitting, Path(endpoint['checkpoint']['path']), endpoint['checkpoint']['sha256'],
                          endpoint['terminal_state_sha256'], ident)
    fitter.integrity(fitting, state, ident)
    require(context['selected']['original'].fingerprint(state['features']) == context['feature_state_sha256'],
            'independently reconstructed canonical TRAIN normalization differs')
    return state, {'payload_sha256': endpoint['terminal_state_sha256'], 'identity': ident,
                   'readout_sha256': readout_digest(context, state)}


def readout_digest(context, state):
    return context['selected']['original'].fingerprint({'head': dict(state['head'].state_dict()),
        **{k: state[k] for k in context['fitter'].FITTED_KEYS}})


def head_values(context, state, cache):
    import torch
    fitter, fitting = context['fitter'], context['fitting']
    ident = context['records'][state['arm']]['identity']
    fitter.integrity(fitting, state, ident)
    with torch.no_grad(), torch.autocast('cpu', enabled=False):
        raw = fitter.raw_features(fitting, state, cache)
        values = context['trainer'].packed_outputs(context['selected'], raw)
    fitter.integrity(fitting, state, ident)
    return tuple(values[k] for k in ('raw', 'unit', 'codes', 'inverse_norms'))


def release_head(context, state):
    context['fitter'].release(context['fitting'], state)
    gc.collect()
    context['trainer'].require_no_model(context['selected'])


def qualify_heads(context):
    cache, original = context['features'][:64], context['selected']['original']
    witnesses, facts = {}, {}
    for endpoint in context['spec']['endpoints']:
        arm = endpoint['arm']
        state, facts[arm] = load_head(context, endpoint)
        values = head_values(context, state, cache)
        witnesses[arm] = context['baseline'].value_facts(context, values)
        require(readout_digest(context, state) == facts[arm]['readout_sha256'], 'TRAIN witness changed fitted readout')
        release_head(context, state)
        state, second_facts = load_head(context, endpoint)
        second = head_values(context, state, cache)
        context['helper'].exact(values, second)
        require(original.fingerprint(values) == original.fingerprint(second) and second_facts == facts[arm] and
                readout_digest(context, state) == second_facts['readout_sha256'], 'independent TRAIN readout/wire bits differ')
        release_head(context, state)
        del values, second
        gc.collect()
    return {'head_facts': facts, 'train_witnesses': witnesses}


def check_receipt(context, record, phase):
    spec, prior = context['spec'], context['selected']['selected']['source_cpu']
    require(all(record[k] == v for k, v in bind(context).items()) and record['schema'] == SCHEMA and
            record['phase'] == phase and record['resource_policy'] == policy(phase) and record['pass'] is True and
            all(record[k] is True for k in ('engineering_admission_pass', 'integrity_pass', 'resources_pass',
                'strict_independent_head_reload_exact', 'train_raw_unit_cpu_packed_exact',
                'complete_typed_terminal_state_exact', 'first_heads_released_before_reload', 'rng_flags_preserved',
                'exit_rehash_pass', 'both_locks_held_in_parent_authority')) and
            all(record[k] is False for k in ('official_read', 'global_production_goal_met', 'public_latency_measured',
                'public_encoder_qualified', 'cuda_initialized', 'fitted_statistics_recomputed', 'product_go')) and
            record['optimizer_updates'] == record['optimizer_members'] == record['fit_calls'] == 0 and
            record['peak_cuda_allocated_bytes'] == 0 and
            record['certificate'] == 'updated cached readout composed with qualified immutable encoder',
            'accepted new evaluator integrity/resource/scope differs')
    invocation = record['invocation']
    require(record['numerical_flags'] == prior['numerical_flags'] and invocation['argv'] ==
            cli_argv(Path(record['authority']['path']), record['authority']['sha256'], record['execution_sha256'],
                     phase, Path(record['output']), record['prerequisite']) and
            all(invocation[k] == prior['invocation'][k] for k in ('python', 'python_sha256', 'python_version')) and
            invocation['cuda_visible_devices'] == '' and invocation['optimize'] == 0 and
            record['authority'] == {'path': str(context['args'].authority), 'sha256': context['args'].authority_sha256} and
            record['output'] == str(Path(record['output']).resolve()) and
            record['head_facts'].keys() == record['train_witnesses'].keys() == set(ARMS) and
            record['cost'] == context['costs'] and record['cost_pass'] is context['costs']['pass'] and
            record['cost_policy'] == COST_POLICY and record['preparation_costs'] == preparation_costs(context),
            'original evaluator invocation/cost attribution differs')
    for endpoint in spec['endpoints']:
        arm, fact = endpoint['arm'], record['head_facts'][endpoint['arm']]
        require(fact.keys() == {'payload_sha256', 'identity', 'readout_sha256'} and
                fact['payload_sha256'] == endpoint['terminal_state_sha256'] and
                fact['identity'] == context['records'][arm]['identity'] and
                re.fullmatch('[0-9a-f]{64}', fact['readout_sha256']), 'complete accepted state facts differ')
        check_value_facts(record['train_witnesses'][arm], 64)
    if phase == 'cpu':
        require(record['quality_read'] is False and record['prerequisite'] is None and record['cpu_terminal'] is None and
                record['files'] == {} and record['quality_pass'] is None and record['decision'] is None and
                record['validation_quality_exposed'] is False, 'TRAIN-only evaluator CPU qualification differs')
        return
    require(record['prerequisite'] is not None and record['cpu_terminal'] is not None, 'original evaluator CPU proof required')
    if context['costs']['pass'] is False:
        require(record['decision'] == 'KILL' and record['quality_pass'] is None and record['quality_read'] is False and
                record['files'] == {} and record['validation_quality_exposed'] is False and
                record['selection_go_admits_validation_only'] is False, 'cost KILL must precede panel-cache access')
        return
    require(record['quality_read'] is True and
            all(record[k] is True for k in ('full_panel_raw_unit_packed_replay_exact', 'per_query_replay_exact',
                'persisted_wire_scoring_replay_exact', 'archived_selection_per_query_exact', 'canonical_panel_input_only')) and
            record['files'].keys() == file_names() and record['bootstrap_seed'] == 179019 and
            record['bootstrap_draws'] == (5000 if record['immediate_quality_pass'] else 0) and
            record['intervals_computed'] is record['immediate_quality_pass'] and
            record['source_quality_panel'] == spec['panel'] and
            record['source_selection_receipt'] == SOURCE_INVENTORY['receipt'] and
            record['source_checkpoint'] == SOURCE_INVENTORY['baseline_endpoint']['checkpoint'] and
            record['source_terminal_state_sha256'] == SOURCE_INVENTORY['baseline_endpoint']['terminal_state_sha256'] and
            record['validation_quality_exposed'] is (spec['panel'] == 'validation') and
            record['serving_view_averaging'] is False and
            record['interval_scope'] == 'single frozen trained source conditional paired product/query intervals' and
            (record['panel_images'], record['query_images'], record['gallery_images'], record['panel_products']) == PANELS[spec['panel']],
            'accepted new panel/source wire replay/scope differs')
    check_value_facts(record['source_panel_facts'], record['panel_images'])
    require(record['panel_replay_facts'].keys() == set(ARMS), 'complete paired panel witnesses required')
    for facts in record['panel_replay_facts'].values():
        check_value_facts(facts, record['panel_images'])
    if spec['panel'] == 'selection':
        replay_equal(context['source_record']['quality']['179061']['control'], record['source_quality'])
    decision = decide(record['quality'], record['source_quality'], spec['panel'], record['paired_intervals'], record['cost'])
    require(all(record[k] == v for k, v in decision.items()), 'terminal quality/floor/cost decision differs')


def accept_terminal(context, terminal, phase):
    check_terminal_descriptor(terminal)
    record = read_json(terminal['receipt'], context['guards'])
    check_receipt(context, record, phase)
    require(Path(terminal['receipt']['path']) == Path(record['output']) / 'receipt.json', 'original receipt role differs')
    final = context['admission'].admit_terminal(record, terminal, policy(phase)['seconds'], context['guards'])
    for value in (record['cgroup_before'], record['cgroup_after'], final):
        context['helper'].zero_events(value)
    excluded = {terminal['receipt']['path'], terminal['log']['path']}
    if context['args'].prerequisite is not None:
        excluded.add(str(context['args'].prerequisite))
    require(all(record['input_guards'].get(p) == h for p, h in context['required_guards'].items() if p not in excluded),
            'complete evaluator original input guards differ')
    for path, digest in record['input_guards'].items():
        bound_file(context['guards'], path, digest)
    for name, digest in record['files'].items():
        bound_file(context['guards'], Path(record['output']) / name, digest)
    context['baseline'].check_origins(context, record['origins'])
    context['terminals'].append(terminal); context['origin_records'].append(record)
    return record


def accept_selection(context, terminal):
    record = read_json(terminal['receipt'], context['guards'])
    prior_spec = read_json(record['authority'], context['guards'])
    check_spec(prior_spec, context['args']); check_prior_binding(context['spec'], prior_spec)
    args = SimpleNamespace(**vars(context['args']))
    args.authority, args.authority_sha256 = Path(record['authority']['path']), record['authority']['sha256']
    args.prerequisite = None
    required = {**context['base_guards'], str(args.authority): args.authority_sha256}
    previous = {**context, 'args': args, 'spec': prior_spec, 'required_guards': required}
    accepted = accept_terminal(previous, terminal, 'score')
    require(accepted['decision'] == 'GO' and accepted['quality_pass'] is True and accepted['cost_pass'] is True and
            accepted['selection_go_admits_validation_only'] is True, 'selection GO required before validation')
    cpu_terminal = read_json(accepted['prerequisite'], context['guards'])
    require(cpu_terminal == accepted['cpu_terminal'], 'original selection CPU terminal differs')
    cpu = accept_terminal(previous, cpu_terminal, 'cpu')
    require(all(cpu[k] == accepted[k] for k in ('head_facts', 'train_witnesses')), 'selection CPU witnesses differ')
    return accepted


def prerequisites(context):
    args = context['args']
    require((args.prerequisite is None) == (args.phase == 'cpu') and
            (args.prerequisite_sha256 is None) == (args.phase == 'cpu'), 'phase prerequisite differs')
    context['cpu_terminal'] = None
    if args.phase == 'cpu':
        return None
    terminal = read_json({'path': str(args.prerequisite), 'sha256': args.prerequisite_sha256}, context['guards'])
    context['cpu_terminal'] = terminal
    cpu = accept_terminal(context, terminal, 'cpu')
    unique_terminals(context['terminals'])
    return cpu


def score_saved_wires(context, key, files, fixed, labels, panel, expected):
    """Score independently read persisted raw/unit/packed wires, including actual packed bytes."""
    import numpy as np
    import torch
    values = []
    for suffix in ('.raw.npy', '.unit.npy'):
        name = key + suffix
        path = bound_file(context['guards'], context['args'].output / name, files[name])
        array = np.load(path, allow_pickle=False)
        require(array.dtype == np.float32 and array.shape == (len(labels), 128) and np.isfinite(array).all(),
                'persisted complete panel layout differs')
        values.append(torch.from_numpy(array))
    packed = context['packing'].pack_int8_unit_embeddings(values[1])
    values.extend((packed.codes, packed.inverse_norms))
    name = key + '.packed.bin'
    raw = bound_file(context['guards'], context['args'].output / name, files[name]).read_bytes()
    require(raw == packed.to_bytes(), 'persisted scoring packed bytes differ')
    context['helper'].exact(expected, tuple(values))
    require(context['selected']['original'].fingerprint(expected) ==
            context['selected']['original'].fingerprint(tuple(values)), 'persisted scoring witness differs')
    # The unchanged pinned scorer packs these read-back unit rows itself. Equality above
    # proves those actual scored codes and inverse bits are precisely the saved wire.
    return fixed.packed_quality(values[1].numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu'))


def score_panel(context, cpu):
    import numpy as np
    import torch
    spec, baseline, original = context['spec'], context['baseline'], context['selected']['original']
    require(context['costs']['pass'] is True, 'paired original service/fit-core cost gate precedes panel access')
    fixed = baseline.scoring_math(context)
    baseline.replay_archived_source(context, fixed)
    panel = context['partition']['panels'][spec['panel']]
    require(tuple(map(len, (panel['original_rows'], panel['query'], panel['gallery'], panel['original_class_ids']))) ==
            PANELS[spec['panel']], 'complete authorized panel mapping differs')
    if spec['panel'] == 'validation':
        require(context['selection_go']['decision'] == 'GO' and
                context['selection_go']['selection_go_admits_validation_only'] is True,
                'sealed validation cannot precede original selection GO')
    cache = baseline.cache_rows(context, panel['original_rows'])
    labels = tuple(context['fit']['class_names'][context['fit']['targets'][r]] for r in panel['original_rows'])
    source_first = baseline.source_values(context, cache)
    source_second = baseline.source_values(context, cache)
    context['helper'].exact(source_first, source_second)
    require(original.fingerprint(source_first) == original.fingerprint(source_second), 'fresh source wire bits differ')
    if spec['panel'] == 'selection':
        archived = baseline.archived_source_wires(context)
        context['helper'].exact(source_second, archived)
        require(original.fingerprint(source_second) == original.fingerprint(archived), 'fresh source/archived direct-FIT bits differ')
        del archived
    source_quality = fixed.packed_quality(source_second[1].numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu'))
    if spec['panel'] == 'selection':
        replay_equal(context['source_record']['quality']['179061']['control'], source_quality)
    check_quality(source_quality, PANELS[spec['panel']][1])
    files = baseline.write_wires(context, 'source-179061', source_first)
    baseline.readback_wires(context, 'source-179061', files, source_second)
    replay_equal(source_quality, score_saved_wires(context, 'source-179061', files, fixed, labels, panel, source_second))
    source_facts = baseline.value_facts(context, source_second)
    del source_first, source_second
    quality, replay_facts = {}, {}
    for endpoint in spec['endpoints']:
        arm = endpoint['arm']
        state, facts = load_head(context, endpoint)
        require(facts == cpu['head_facts'][arm], 'CPU qualified complete scoring state differs')
        values = head_values(context, state, cache)
        files.update(baseline.write_wires(context, arm, values))
        first = fixed.packed_quality(values[1].numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu'))
        release_head(context, state)
        state, second_facts = load_head(context, endpoint)
        second = head_values(context, state, cache)
        context['helper'].exact(values, second)
        require(original.fingerprint(values) == original.fingerprint(second) and second_facts == facts,
                'independent full panel fitted state/wire bits differ')
        baseline.readback_wires(context, arm, files, second)
        replay_equal(first, fixed.packed_quality(second[1].numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu')))
        replay_equal(first, score_saved_wires(context, arm, files, fixed, labels, panel, second))
        quality[arm], replay_facts[arm] = first, baseline.value_facts(context, second)
        release_head(context, state)
        del values, second
        gc.collect()
    pair = metric_deltas(quality, source_quality, spec['panel'])['quadratic_minus_linear']
    intervals = {}
    immediate = immediate_quality_pass(quality, source_quality, spec['panel'])
    for metric in METRICS if immediate else ():
        delta = np.asarray(pair[metric]); intervals[metric] = {'mean_delta': float(delta.mean())}
        for kind, groups in (('product', np.asarray(labels)[panel['query']]), ('query', np.arange(len(panel['query'])))):
            # The pinned helper resets 179019 per call: identical 5000 draws for both metrics/signs.
            intervals[metric][kind + '_lower95'] = fixed.bootstrap_lower(delta, groups)
            intervals[metric][kind + '_upper95'] = -fixed.bootstrap_lower(-delta, groups)
    decision = decide(quality, source_quality, spec['panel'], intervals, context['costs'])
    return {**decision, 'quality': quality, 'source_quality': source_quality, 'source_quality_panel': spec['panel'],
            'source_selection_receipt': SOURCE_INVENTORY['receipt'],
            'source_checkpoint': SOURCE_INVENTORY['baseline_endpoint']['checkpoint'],
            'source_terminal_state_sha256': SOURCE_INVENTORY['baseline_endpoint']['terminal_state_sha256'],
            'archived_selection_per_query_exact': True, 'source_panel_facts': source_facts,
            'cost': context['costs'], 'cost_policy': COST_POLICY, 'paired_intervals': intervals,
            'bootstrap_draws': 5000 if immediate else 0, 'bootstrap_seed': 179019, 'intervals_computed': immediate,
            'panel_images': len(panel['original_rows']), 'query_images': len(panel['query']),
            'gallery_images': len(panel['gallery']), 'panel_products': len(panel['original_class_ids']),
            'fit_images': 6355, 'fit_products': 1008, 'quality_read': True, 'files': files,
            'panel_replay_facts': replay_facts, 'full_panel_raw_unit_packed_replay_exact': True,
            'per_query_replay_exact': True, 'persisted_wire_scoring_replay_exact': True,
            'metric_units': 'fractions; multiply deltas by100 for percentage points',
            'interval_scope': 'single frozen trained source conditional paired product/query intervals',
            'cost_denominator': 'original linear whole service and both complete fit-core passes incl prototypes',
            'optimization_throughput_is_image_training_throughput': False, 'independent_pretraining_seeds': False,
            'intermediate_checkpoint_selection': False, 'validation_quality_exposed': spec['panel'] == 'validation',
            'canonical_panel_input_only': True, 'serving_view_averaging': False,
            'preparation_costs': preparation_costs(context)}


def exit_rehash(context):
    """Use the unchanged complete source exit, with the new guard union included."""
    merge_guards(context['guards'], context['fitting']['guards'])
    merge_guards(context['guards'], context['fitting']['legacy']['guards'])
    context['trainer'].require_no_model(context['selected'])
    origins = context['baseline'].exit_rehash(context)
    context['baseline'].check_origins(context, origins)
    context['baseline'].check_exit_encoder(context)
    for item, names, expected in (
        ({'root': str(context['root']), 'execution_sha256': context['args'].execution_sha256}, FILES, context['code']),
        (context['spec']['training'], TRAIN_FILES, context['spec']['training']['code']),
        (context['spec']['original_evaluator'], EVALUATOR_PINS, EVALUATOR_PINS),
        (ORIGINAL_REFERENCE, ORIGINAL_PINS, ORIGINAL_PINS), (EVALUATION_REFERENCE, REFERENCE_PINS, REFERENCE_PINS)):
        require(closure(Path(item['root']), item['execution_sha256'], names, {}) == expected,
                'fresh uncached exit closure differs')
    return origins


def run(args):
    prior = None if args.prerequisite is None else {'path': str(args.prerequisite), 'sha256': args.prerequisite_sha256}
    require(sys.argv == cli_argv(args.authority, args.authority_sha256, args.execution_sha256, args.phase, args.output, prior),
            'fixed canonical CLI order required')
    print(json.dumps({'progress': 'authority_start', 'seconds': time.perf_counter() - UNIT_STARTED}), flush=True)
    context = authority(args); cpu = prerequisites(context)
    print(json.dumps({'progress': 'authority_end', 'cost_pass': context['costs']['pass'],
                      'seconds': time.perf_counter() - UNIT_STARTED}), flush=True)
    before = native_start(context)
    import torch
    source = context['selected']['source_driver']
    rng, flags = torch.random.get_rng_state().clone(), source.numerical_flags()
    args.output.mkdir()
    facts = qualify_heads(context)
    if cpu is not None:
        require(all(facts[k] == cpu[k] for k in facts), 'accepted evaluator CPU witnesses differ')
    result = {'decision': None if args.phase == 'cpu' else 'KILL', 'quality_pass': None,
              'quality_read': False, 'files': {}, 'validation_quality_exposed': False,
              'selection_go_admits_validation_only': False}
    if args.phase == 'score' and context['costs']['pass']:
        result = score_panel(context, cpu)
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags and
            not torch.cuda.is_initialized(), 'whole-unit RNG/flags/CUDA differs')
    print(json.dumps({'progress': 'exit_rehash_start', 'seconds': time.perf_counter() - UNIT_STARTED}), flush=True)
    origins = exit_rehash(context)
    resources = context['helper'].resources(context, args.phase, before)
    receipt = {**bind(context), 'schema': SCHEMA, 'phase': args.phase, 'pass': True,
        'engineering_admission_pass': True, 'integrity_pass': True, 'resources_pass': True,
        'certificate': 'updated cached readout composed with qualified immutable encoder', 'public_encoder_qualified': False,
        'authority': {'path': str(args.authority), 'sha256': args.authority_sha256}, 'output': str(args.output),
        'prerequisite': prior, 'cpu_terminal': context['cpu_terminal'], 'optimizer_updates': 0, 'optimizer_members': 0,
        'fit_calls': 0, 'fitted_statistics_recomputed': False, 'numerical_flags': flags,
        'strict_independent_head_reload_exact': True, 'train_raw_unit_cpu_packed_exact': True,
        'complete_typed_terminal_state_exact': True, 'first_heads_released_before_reload': True,
        'rng_flags_preserved': True, 'cuda_initialized': False, 'official_read': False,
        'global_production_goal_met': False, 'product_go': False, 'public_latency_measured': False,
        'cost': context['costs'], 'cost_pass': context['costs']['pass'], 'cost_policy': COST_POLICY,
        'preparation_costs': preparation_costs(context), 'input_guards': context['guards'],
        'origins': origins, 'exit_rehash_pass': True,
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
        raise SystemExit('Prototype residual evaluation rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output),
                      'decision': result.get('decision')}), flush=True)


if __name__ == '__main__':
    main()
