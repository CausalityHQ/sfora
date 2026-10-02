#!/usr/bin/env python3
"""Cache-only prospective genuine mild-view evaluator; native gates UNRUN.

Freeze execution.json with exactly FILES (this evaluator + its stdlib check).
Keep training's immutable trainer2 root separate, exactly TRAIN_FILES. Keep the
unchanged previous evaluator6 root separate, pinned REFERENCE_EXECUTION_SHA.
Parent supplies actual final trainer2 execution and file SHAs in the hashed
authority; trainer source-v1 is now pinned to its completed final source.
Future checkpoint/native evidence is never guessed.
No encoder, image decoding, export, fitting, official quality or old head reuse.

Authority siglip2-genuine-view-evaluation-authority-v1 has exactly SPEC_KEYS:
execution_sha256; training={root,execution_sha256,code}; evaluation_reference=
{root,execution_sha256}; partition=FILE pinned PARTITION_SHA; stage=first|full;
panel=selection|validation; endpoints in endpoint_order(stage), each exactly
{seed,arm,launch:FILE,terminal:TERMINAL,checkpoint:FILE,terminal_state_sha256};
first_selection=null for first, otherwise accepted original first score TERMINAL;
selection_go=null for selection, otherwise original full selection GO TERMINAL;
resource_policies={cpu:policy('cpu'),score:policy('score')}; cost_policy=COST_POLICY;
both_locks_held=true; selection_previously_exposed=true.
FILE={path:canonical absolute file,sha256:actual lowercase
SHA256}; TERMINAL={receipt:FILE,log:FILE,unit,invocation_id,service_seconds,
native_peak_rss_kib,both_locks_held:true}. Parent supplies only completed actual
CPU/both mechanics/TRAIN/log/checkpoint/normal-exit authorities, never future SHAs.

Exact CLI, both lifetime locks and original systemd limits supplied by parent:
CUDA_VISIBLE_DEVICES='' python -B ROOT/evaluate_siglip2_genuine_views.py
 --execution-sha256 SHA --authority FILE --authority-sha256 SHA
 --phase cpu --output NEW_DIRECTORY
CUDA_VISIBLE_DEVICES='' python -B ROOT/evaluate_siglip2_genuine_views.py
 --execution-sha256 SHA --authority FILE --authority-sha256 SHA
 --phase score --output NEW_DIRECTORY --prerequisite CPU_TERMINAL_JSON
 --prerequisite-sha256 SHA
CPU_TERMINAL_JSON is TERMINAL for the CPU receipt under the SAME authority.
CPU120/score300 include full admission/independent strict reload/uncached exit,
8GiB/noSwap/zero disallowed memory events, no CUDA initialization/peak reset.

CPU qualifies BOTH affine head arms on NEW canonical TRAIN-only cache witnesses,
using the trainer's exact shared normalization, and actual
complete terminal checkpoint reloads, raw/unit/CPU-packed bit equality, RNG and
flags. Score consumes normalized F32[13283,1152] SHA FIT_SHA, indexes ONLY the
authorized panel, and independently reloads/replays all raw/unit/packed and
per-query R1/AP before decision. receipt.json and raw/unit/packed wires bind all
input authorities. Engineering admission PASS is separate from quality/cost.
First selection C061,A061: KILL if deltaR1<=0 or deltaAP<0; otherwise CONTINUE
only with cost PASS. Full C061,A061,A069,C069 requires that authenticated first
continuation, eachseed R1>0/AP>=0, equal-seed mean BOTH>=.002, paired-product95%
lower BOTH>0. Unchanged stable packed mAP@R, shared5000drawseed179019 product
and query intervals. Original whole-service/median ratios<=1.50 eachseed;
optimization wall is reported separately. Validation requires original full
selection GO and identical four checkpoints, same gates. Any quality failure
is terminal KILL, no refit/retune/seed selection; production goal stays unmet.
Shared actual two-cache preparation is disclosed separately and attributed to
both arms as a shared prerequisite; it is never image-training throughput.
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

SCHEMA = 'siglip2-genuine-view-evaluation-v1'
AUTHORITY_SCHEMA = 'siglip2-genuine-view-evaluation-authority-v1'
FILES = {'evaluate_siglip2_genuine_views.py', 'test_siglip2_genuine_view_evaluation.py'}
TRAIN_FILES = {'train_siglip2_genuine_views.py', 'test_siglip2_genuine_view_training.py'}
TRAIN_ROOT = '/home/riomus/runs/sfora-so400-genuine-view-train-source-v1'
TRAIN_EXECUTION_SHA = '2003e9a6adba30c8f90fa17f6437cbba1f4b6cd2d12e75906195d16229846e08'
TRAIN_PINS = {'train_siglip2_genuine_views.py': '788d672bca384c478881fc8f0b59090efbaf6f84e1c9f27469828bc951b7ec96',
              'test_siglip2_genuine_view_training.py': '1effe6e87b9d667d29bc7b8904023634e7913f53c4b0a17e360728f83e1cd77b'}
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
SPEC_KEYS = {'schema', 'execution_sha256', 'training', 'evaluation_reference', 'partition',
             'stage', 'panel', 'endpoints', 'first_selection', 'selection_go',
             'resource_policies', 'cost_policy', 'both_locks_held', 'selection_previously_exposed'}


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


def check_spec(spec, args):
    require(spec.keys() == SPEC_KEYS and spec['schema'] == AUTHORITY_SCHEMA and
            spec['execution_sha256'] == args.execution_sha256 and spec['both_locks_held'] is True and
            spec['resource_policies'] == {p: policy(p) for p in ('cpu', 'score')} and
            spec['cost_policy'] == COST_POLICY and spec['selection_previously_exposed'] is True, 'evaluation authority profile differs')
    require(spec['training'].keys() == {'root', 'execution_sha256', 'code'} and
            spec['training']['root'] == TRAIN_ROOT and spec['training']['execution_sha256'] == TRAIN_EXECUTION_SHA and
            spec['training']['code'] == TRAIN_PINS and
            spec['evaluation_reference'].keys() == {'root', 'execution_sha256'} and
            spec['evaluation_reference']['root'] == REFERENCE_ROOT and
            spec['evaluation_reference']['execution_sha256'] == REFERENCE_EXECUTION_SHA and
            spec['partition'].keys() == {'path', 'sha256'} and spec['partition']['sha256'] == PARTITION_SHA,
            'separate pinned training/reference/partition required')
    require(spec['panel'] in PANELS and (spec['panel'] != 'validation' or spec['stage'] == 'full') and
            (spec['first_selection'] is None) == (spec['stage'] == 'first') and
            (spec['selection_go'] is None) == (spec['panel'] == 'selection'),
            'validation requires selection GO; full requires first continuation')
    require([(e['seed'], e['arm']) for e in spec['endpoints']] == list(endpoint_order(spec['stage'])) and
            all(type(e['seed']) is int for e in spec['endpoints']), 'prospective ordered endpoints required')
    for endpoint in spec['endpoints']:
        require(endpoint.keys() == {'seed', 'arm', 'launch', 'terminal', 'checkpoint', 'terminal_state_sha256'} and
                all(endpoint[k].keys() == {'path', 'sha256'} for k in ('launch', 'checkpoint')) and
                re.fullmatch('[0-9a-f]{64}', endpoint['terminal_state_sha256']), 'exact endpoint descriptor required')
        check_terminal_descriptor(endpoint['terminal'])
    for terminal in (spec['first_selection'], spec['selection_go']):
        if terminal is not None:
            check_terminal_descriptor(terminal)


def check_endpoint(trainer, record, endpoint, selected):
    seed, arm, launch = endpoint['seed'], endpoint['arm'], record['launch']
    trainer.check_launch(launch, SimpleNamespace(phase='train', arm=arm, seed=seed,
                                               execution_sha256=selected['launch']['execution_sha256']))
    require(record['schema'] == trainer.SCHEMA and record['phase'] == 'train' and
            (record['seed'], record['arm']) == (seed, arm) and record['completed_step'] == 1000 and
            record['optimizer_members'] == 5 and record['source'] == selected['source'] and
            trainer.method(launch) == trainer.method(selected['launch']) and
            launch['selected_cpu'] == selected['launch']['selected_cpu'] and
            launch['selected_mechanics'] == selected['launch']['selected_mechanics'] and
            record['terminal_cgroups'] == selected['terminal_cgroups'] and record['checkpoint'] == endpoint['checkpoint'] and
            record['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and
            record['partition_sha256'] == PARTITION_SHA and record['code'] == selected['code'] and
            record['resource_policy'] == trainer.policy('train') and
            record['numerical_flags'] == selected['source_cpu']['numerical_flags'] and
            record['head_scalars'] == 188544 and record['trainable_scalars'] == 317568 and
            record['encoder_updates'] == 0 and record['resumed_steps'] == [], 'complete TRAIN1000 endpoint differs')
    require(all(record[k] is True for k in ('pass', 'training_qualified', 'strict_reload_exact', 'exit_rehash_pass')) and
            all(record[k] is False for k in ('quality_read', 'trained_state_reused', 'training_state_discarded', 'replay_exact')) and
            0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000, 'fresh complete TRAIN1000 acceptance differs')
    ident = record['identity']
    trainer.check_views({k: ident['source'][k] for k in
                         ('caches', 'ordered_input_sha256', 'ordered_view_sha256')}, selected['source'])
    defaults = checked_defaults(ident)
    groups = [dict(defaults, lr=1e-4)] * 2
    require(ident['method'] == trainer.method(launch) and (ident['seed'], ident['arm']) == (seed, arm) and
            ident['source'] == selected['source'] and ident['selected_cpu'] == launch['selected_cpu'] and
            ident['device'] == 'cuda' and ident['parameter_names'] == trainer.PARAMETERS and
            ident['numerical_flags'] == record['numerical_flags'] and ident['optimizer_defaults'] == defaults and
            ident['optimizer_groups'] == groups and ident['optimizer_serial_groups'] ==
            [dict(groups[0], params=[0, 1, 2, 3]), dict(groups[1], params=[4])], 'five-member endpoint identity differs')
    trainer.check_steps(record['steps'], 1, 1000)
    require(record['steps'][-1]['state_sha256'] == endpoint['terminal_state_sha256'] and
            all(row['schedule_sha256'] == ident['schedule_sha256'] for row in record['steps']) and
            record['median_update_seconds'] == statistics.median(row['seconds'] for row in record['steps'][2:]) and
            0 < sum(row['seconds'] for row in record['steps']) <= record['training_wall_seconds'] <= record['wall_seconds'],
            'terminal state/schedule/update cost differs')
    if seed == SEEDS[0]:
        mechanics = selected['terminals']['mechanics:' + arm]
        require(record['initial_state_sha256'] == mechanics['initial_state_sha256'] and
                all(trainer.diagnostic(a) == trainer.diagnostic(b) for a, b in
                    zip(record['steps'][:17], mechanics['steps'], strict=True)), 'fresh first17 mechanics replay differs')


def checked_defaults(ident):
    defaults = {'lr': .001, 'betas': [.9, .999], 'eps': 1e-8, 'weight_decay': .05,
                'amsgrad': False, 'maximize': False, 'foreach': None, 'capturable': False,
                'differentiable': False, 'fused': None}
    if 'decoupled_weight_decay' in ident['optimizer_defaults']:
        defaults['decoupled_weight_decay'] = True
    require(ident['optimizer_defaults'] == defaults, 'original AdamW defaults differ')
    return defaults


def preparation_costs(selected):
    terminal = selected['launch']['selected_export']
    record = selected['export_record']
    require(all(type(terminal[k]) in (int, float) and math.isfinite(terminal[k]) and terminal[k] > 0
                for k in ('service_seconds', 'native_peak_rss_kib')) and
            0 < record['extraction_seconds'] <= record['wall_seconds'] < terminal['service_seconds'],
            'actual shared preparation cost differs')
    return {'shared': {'terminal': terminal, 'whole_service_seconds': terminal['service_seconds'],
                       'extraction_seconds': record['extraction_seconds'], 'views': ['canonical', 'augmented']},
            'arm_attributed': {a: {'shared_whole_service_seconds': terminal['service_seconds'],
                                  'additional_preparation_jobs': 0} for a in ARMS},
            'shared_counted_once': True, 'included_in_training_cost_ratios': False,
            'image_training_throughput': False}


def paired_cost(records, stage):
    require(set(records) == set(endpoint_order(stage)), 'complete paired costs required')
    costs = {}
    for seed in seeds(stage):
        control, candidate = (records[seed, arm] for arm in ARMS)
        require(all(control['identity'][k] == candidate['identity'][k] for k in
                    ('static_sha256', 'buffers_sha256', 'feature_state_sha256', 'schedule_sha256', 'mask_sha256')) and
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


def averaged_deltas(quality, stage, panel):
    wanted, count = {str(s) for s in seeds(stage)}, PANELS[panel][1]
    require(quality.keys() == wanted, 'complete panel seeds required')
    deltas = {}
    for seed in wanted:
        require(quality[seed].keys() == set(ARMS), 'complete paired panel arms required')
        deltas[seed] = {}
        for metric in METRICS:
            a, b = (quality[seed][arm][metric] for arm in ARMS)
            require(len(a) == len(b) == count and all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1
                                                    for v in (*a, *b)), 'complete finite per-query panel metrics required')
            deltas[seed][metric] = [y - x for x, y in zip(a, b, strict=True)]
    average = {m: [statistics.mean(row) for row in zip(*(deltas[str(s)][m] for s in seeds(stage)), strict=True)]
               for m in METRICS}
    return deltas, average


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


def bind(context):
    return {'authority_sha256': context['args'].authority_sha256,
            'execution_sha256': context['args'].execution_sha256,
            'spec': context['spec'], 'selection_previously_exposed': True, 'source': context['selected']['source'], 'source_code': context['code']}


def cli_argv(authority, authority_sha, execution_sha, phase, output, prerequisite):
    value = [str(Path(__file__).absolute()), '--execution-sha256', execution_sha,
             '--authority', str(authority), '--authority-sha256', authority_sha,
             '--phase', phase, '--output', str(output)]
    if prerequisite is not None:
        value += ['--prerequisite', prerequisite['path'], '--prerequisite-sha256', prerequisite['sha256']]
    return value


def check_receipt(context, record, phase):
    require(all(record[k] == v for k, v in bind(context).items()) and record['schema'] == SCHEMA and
            record['phase'] == phase and record['resource_policy'] == policy(phase) and record['pass'] is True and
            record['engineering_admission_pass'] is True and record['optimizer_updates'] == 0 and record['optimizer_members'] == 5 and
            all(record[k] is True for k in ('strict_independent_head_reload_exact', 'train_raw_unit_cpu_packed_exact',
                                          'first_heads_released_before_reload', 'rng_flags_preserved', 'exit_rehash_pass')) and
            all(record[k] is False for k in ('official_read', 'global_production_goal_met', 'public_latency_measured',
                                           'cuda_initialized')) and record['peak_cuda_allocated_bytes'] == 0,
            'accepted evaluator engineering receipt differs')
    prior = context['selected']['source_cpu']
    invocation = record['invocation']
    require(record['numerical_flags'] == prior['numerical_flags'] and invocation['argv'] ==
            cli_argv(Path(record['authority']['path']), record['authority']['sha256'], record['execution_sha256'],
                     phase, Path(record['output']), record['prerequisite']) and
            all(invocation[k] == prior['invocation'][k] for k in ('python', 'python_sha256', 'python_version')) and
            invocation['cuda_visible_devices'] == '' and invocation['optimize'] == 0 and
            record['authority'] == {'path': str(context['args'].authority), 'sha256': context['args'].authority_sha256} and
            record['output'] == str(Path(record['output']).resolve()) and
            record['head_facts'].keys() == record['train_witnesses'].keys() == {label(e) for e in context['spec']['endpoints']},
            'original evaluator invocation/head qualification differs')
    if phase == 'cpu':
        require(record['quality_read'] is False and record['prerequisite'] is None and record['cpu_terminal'] is None and
                record['files'] == {}, 'TRAIN-only CPU qualification differs')
    else:
        require(record['quality_read'] is True and record['full_panel_raw_unit_packed_replay_exact'] is True and
                record['per_query_replay_exact'] is True and record['files'].keys() == file_names(context['spec']) and
                record['cost'] == paired_cost(context['records'], context['spec']['stage']) and
                record['cost_policy'] == COST_POLICY and record['bootstrap_draws'] == (0 if context['spec']['stage'] == 'first' else 5000) and
                record['bootstrap_seed'] == 179019, 'accepted panel score replay/cost differs')
        require(record['canonical_panel_input_only'] is True and record['serving_view_averaging'] is False and
                record['preparation_costs'] == preparation_costs(context['selected']) and
                (context['spec']['stage'] != 'first' or record['paired_seed_average_intervals'] == {}),
                'canonical serving/preparation/first-stage no-CI binding differs')
        deltas, average = averaged_deltas(record['quality'], context['spec']['stage'], context['spec']['panel'])
        require(record['mean_deltas'] == {m: statistics.mean(average[m]) for m in METRICS},
                'mean deltas differ from per-query replay')
        quality_pass = first_gate(deltas) if context['spec']['stage'] == 'first' else quality_gate(deltas, record['paired_seed_average_intervals'])[1]
        cost_pass = all(v['pass'] for v in record['cost'].values())
        decision = ('CONTINUE' if context['spec']['stage'] == 'first' else 'GO') if quality_pass and cost_pass else 'KILL'
        require(record['quality_pass'] is quality_pass and record['cost_pass'] is cost_pass and record['decision'] == decision,
                'terminal quality decision differs')


def accept_terminal(context, terminal, phase):
    record = read_json(terminal['receipt'], context['guards'])
    check_receipt(context, record, phase)
    require(Path(terminal['receipt']['path']) == Path(record['output']) / 'receipt.json', 'original receipt role differs')
    final = context['admission'].admit_terminal(record, terminal, policy(phase)['seconds'], context['guards'])
    for value in (record['cgroup_before'], record['cgroup_after'], final):
        context['helper'].zero_events(value)
    for path, digest in record['input_guards'].items():
        bound_file(context['guards'], path, digest)
    require(all(record['input_guards'].get(p) == h for p, h in context['required_guards'].items()), 'complete evaluator input guards differ')
    for name, digest in record['files'].items():
        bound_file(context['guards'], Path(record['output']) / name, digest)
    context['terminals'].append(terminal)
    context['origin_records'].append(record)
    return record


def check_prior_binding(current, prior, stage):
    require(prior['stage'] == stage and prior['panel'] == 'selection' and
            all(prior[k] == current[k] for k in ('execution_sha256', 'training', 'evaluation_reference', 'partition',
                                               'resource_policies', 'cost_policy', 'selection_previously_exposed')) and
            prior['endpoints'] == current['endpoints'][:len(endpoint_order(stage))] and
            (stage != 'full' or prior['first_selection'] == current['first_selection']),
            'original selection authority/checkpoints differ')


def accept_selection(context, terminal, stage):
    record = read_json(terminal['receipt'], context['guards'])
    prior_spec = read_json(record['authority'], context['guards'])
    check_spec(prior_spec, context['args'])
    check_prior_binding(context['spec'], prior_spec, stage)
    subset = {k: context['records'][k] for k in endpoint_order(stage)}
    args = SimpleNamespace(**vars(context['args']))
    args.authority, args.authority_sha256 = Path(record['authority']['path']), record['authority']['sha256']
    prior_context = {**context, 'args': args, 'spec': prior_spec, 'records': subset,
                     'required_guards': context['base_guards']}
    accepted = accept_terminal(prior_context, terminal, 'score')
    require(accepted['decision'] == ('CONTINUE' if stage == 'first' else 'GO') and
            accepted['quality_pass'] is True and accepted['cost_pass'] is True, 'selection continuation/GO required')
    cpu_terminal = read_json(accepted['prerequisite'], context['guards'])
    require(cpu_terminal == accepted['cpu_terminal'], 'original selection CPU terminal differs')
    cpu = accept_terminal(prior_context, cpu_terminal, 'cpu')
    require(all(cpu[k] == accepted[k] for k in ('head_facts', 'train_witnesses')), 'selection CPU witnesses differ')
    return accepted


def authority(args):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native imports preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    spec = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_spec(spec, args)
    output, train_root, reference_root = args.output, Path(spec['training']['root']), Path(spec['evaluation_reference']['root'])
    check_output(output)
    roots = (root, train_root, reference_root)
    require(all(not a.is_relative_to(b) and not b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]) and
            all(not output.is_relative_to(p) and not p.is_relative_to(output) for p in roots),
            'separate immutable evaluator2/trainer2/reference required')
    train_code = closure(train_root, spec['training']['execution_sha256'], TRAIN_FILES, guards)
    require(train_code == spec['training']['code'], 'parent-frozen new trainer2 differs')
    require(closure(reference_root, REFERENCE_EXECUTION_SHA, REFERENCE_PINS.keys(), guards) == REFERENCE_PINS,
            'pinned original evaluator reference differs')
    trainer = load_bare('_genuine_view_evaluation_trainer', train_root / 'train_siglip2_genuine_views.py',
                        train_code['train_siglip2_genuine_views.py'])
    legacy = load_bare('_genuine_view_evaluation_previous', reference_root / 'evaluate_siglip2_cached_readout.py', REFERENCE_PINS['evaluate_siglip2_cached_readout.py'])
    helper = load_bare('_genuine_view_evaluation_helpers', reference_root / 'export_siglip2_substrate_adaptation.py', REFERENCE_PINS['export_siglip2_substrate_adaptation.py'])
    scoring = load_bare('_genuine_view_evaluation_scoring', reference_root / 'score_siglip2_substrate_adaptation.py', REFERENCE_PINS['score_siglip2_substrate_adaptation.py'])
    require(trainer.FILES == TRAIN_FILES and trainer.SEEDS == SEEDS and trainer.ARMS == ARMS and
            trainer.SCHEMA == 'siglip2-genuine-view-trainer-v1' and
            trainer.PARTITION_SHA == PARTITION_SHA and trainer.FIT_SHA == FIT_SHA and
            trainer.EXPORT_EXECUTION_SHA == '8ec7f2687f7d1e7962de4f9753cefae989d0b6310d325b5e19da33a9961af3dc' and
            trainer.EXPORT_AUTHORITY_SHA == 'a0072203febf1ef3d16ea329deb0a52dccef34a63223d7dc4b5a430d43d8b6a4',
            'new trainer/export profile differs')
    first = spec['endpoints'][0]
    selected = trainer.authority(SimpleNamespace(execution_sha256=spec['training']['execution_sha256'],
        authority=Path(first['launch']['path']), authority_sha256=first['launch']['sha256'],
        phase='train', arm='control', seed=SEEDS[0], output=output))
    admission = selected['original'].FlatAdmission()
    admission.init = selected['genuine']['reference']
    required = {p: h for p, h in selected['guards'].items() if p != first['launch']['path']}
    fit = selected['genuine']['prior']['fit']
    partition = read_json(spec['partition'], selected['guards'])
    train_rows = selected['exporter'].selected_manifest(partition, fit)
    require(partition == selected['partition'] and spec['partition'] == selected['launch']['partition'] and
            train_rows['targets'] == selected['target'], 'original frozen partition differs')
    cache = partition['original_cache']
    require(cache.keys() == {'path', 'sha256'} and cache['sha256'] == FIT_SHA,
            'original canonical panel cache required')
    bound_file(selected['guards'], cache['path'], cache['sha256'])
    terminals = [selected['launch']['selected_cpu'], *selected['launch']['selected_mechanics'].values()]
    export = selected['launch']['selected_export']
    startup = selected['exporter'].file_json(selected['export_record']['startup_terminal'], selected['guards'])
    source_cpu = selected['genuine']['prior']['launch']['source_cpu']['so400']
    preparation = [dict(t, receipt=t['proof']) for t in (source_cpu, startup, export)]
    for t in preparation:
        del t['proof']
    terminals += preparation
    for key, final in selected['terminal_cgroups'].items():
        proof = selected['terminals'][key]
        for value in (proof['cgroup_before'], proof['cgroup_after'], final):
            helper.zero_events(value)
    for arm in ARMS:
        helper.logged_steps(Path(selected['launch']['selected_mechanics'][arm]['log']['path']),
                            selected['terminals']['mechanics:' + arm]['steps'] + selected['terminals']['mechanics:' + arm]['resumed_steps'])
    base_guards = {**selected['guards'], **{p: h for p, h in guards.items() if p != str(args.authority)}}
    records = {}
    for endpoint in spec['endpoints']:
        launch = read_json(endpoint['launch'], selected['guards'])
        record = read_json(endpoint['terminal']['receipt'], selected['guards'])
        require(record['launch'] == launch and record['authority'] == endpoint['launch'] and
                record['authority_sha256'] == endpoint['launch']['sha256'] and
                record['execution_sha256'] == spec['training']['execution_sha256'], 'original endpoint authority differs')
        check_endpoint(trainer, record, endpoint, selected)
        terminal, invocation, prior = endpoint['terminal'], record['invocation'], selected['source_cpu']['invocation']
        require(invocation['argv'] == [str(train_root / 'train_siglip2_genuine_views.py'), '--execution-sha256',
                spec['training']['execution_sha256'], '--authority', endpoint['launch']['path'], '--authority-sha256',
                endpoint['launch']['sha256'], '--phase', 'train', '--arm', endpoint['arm'], '--seed', str(endpoint['seed']),
                '--output', str(Path(terminal['receipt']['path']).parent)] and
                all(invocation[k] == prior[k] for k in ('python', 'python_sha256', 'python_version')) and
                invocation['cuda_visible_devices'] not in (None, '') and invocation['cublas_workspace_config'] == ':4096:8' and
                Path(terminal['receipt']['path']).name == 'receipt.json' and
                Path(endpoint['checkpoint']['path']) == Path(terminal['receipt']['path']).parent / 'resume.pt', 'original TRAIN invocation/path differs')
        require(all(record['input_guards'].get(p) == h for p, h in required.items()), 'complete TRAIN prerequisite guards differ')
        final = admission.admit_terminal(record, terminal, 300, selected['guards'])
        for value in (record['cgroup_before'], record['cgroup_after'], final):
            helper.zero_events(value)
        helper.logged_steps(Path(terminal['log']['path']), record['steps'])
        for path, digest in record['input_guards'].items():
            bound_file(selected['guards'], path, digest)
        bound_file(selected['guards'], endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])
        records[endpoint['seed'], endpoint['arm']] = {**record, 'service_seconds': terminal['service_seconds']}
        terminals.append(terminal)
    for path, digest in guards.items():
        bound_file(selected['guards'], path, digest)
    context = {'args': args, 'root': root, 'code': code, 'spec': spec, 'trainer': trainer, 'legacy': legacy,
               'selected': selected, 'guards': selected['guards'], 'helper': helper, 'scoring': scoring,
               'admission': admission, 'records': records, 'fit': fit, 'partition': partition, 'target': selected['target'],
               'terminals': terminals, 'training': selected['original'],
               'selected_context': {'root': selected['math_context']['root'], 'source': selected['source_driver'],
                                    'initialized': {'init': selected['genuine']['reference']}},
               'origin_records': [selected['source_cpu'],
                   {**selected['export_record'], 'input_guards': {**selected['export_record']['original_input_guards'], **selected['export_record']['input_guards']}},
                   *selected['terminals'].values(), *records.values()],
               'unit_started': UNIT_STARTED, 'base_guards': base_guards,
               'required_guards': selected['guards'].copy()}
    context['costs'] = paired_cost(records, spec['stage'])
    preparation_costs(selected)
    panel_authority(context)
    prior = selected['genuine']['prior']
    external = [Path(selected['launch'][k]['root']) for k in ('cached_reference', 'training_reference', 'export_reference')]
    external += [prior['root'], prior['own_root']]
    external += [Path(v['path']).parent for v in selected['launch']['helpers'].values()]
    require(all(not root.is_relative_to(p) and not p.is_relative_to(root) and
                not output.is_relative_to(p) and not p.is_relative_to(output) for p in external) and
            all(not output.is_relative_to(Path(t['receipt']['path']).parent) and
                not Path(t['receipt']['path']).parent.is_relative_to(output) for t in terminals),
            'immutable dependencies/output overlap')
    unique_terminals(terminals)
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import during endpoint/panel admission')
    return context


def panel_authority(context):
    """Original selection decisions close before any native or panel feature read."""
    check_spec(context['spec'], context['args'])
    spec = context['spec']
    if spec['first_selection'] is not None:
        context['first_selection'] = accept_selection(context, spec['first_selection'], 'first')
    if spec['selection_go'] is not None:
        context['selection_go'] = accept_selection(context, spec['selection_go'], 'full')


def unique_terminals(terminals):
    unique = {t['receipt']['path']: t for t in terminals}
    require(all(unique[t['receipt']['path']] == t for t in terminals) and
            len({t['unit'] for t in unique.values()}) == len(unique) and
            len({t['invocation_id'] for t in unique.values()}) == len(unique), 'distinct original whole units required')


def qualified_origins(context):
    packages, qualified = context['selected']['packages'], {}
    for record in context['origin_records']:
        require(record['origins']['packages'] == packages, 'qualified origin packages differ')
        for path, digest in record['origins']['files'].items():
            require(record['input_guards'].get(path) == digest, 'qualified origin lacks authenticated guard: ' + path)
            require(qualified.setdefault(path, digest) == digest, 'conflicting qualified origin: ' + path)
    return qualified


def check_origins(context, origins):
    require(origins['packages'] == context['selected']['packages'], 'actual origin packages differ')
    qualified = qualified_origins(context)
    for path, digest in origins['files'].items():
        require(qualified.get(path) == digest, 'actual native origin outside admitted qualified union: ' + path)


def prerequisites(context):
    args = context['args']
    require((args.prerequisite is None) == (args.phase == 'cpu') and
            (args.prerequisite_sha256 is None) == (args.phase == 'cpu'), 'phase prerequisite differs')
    if args.phase == 'cpu':
        context['cpu_terminal'] = None
        return None
    terminal = read_json({'path': str(args.prerequisite), 'sha256': args.prerequisite_sha256}, context['guards'])
    context['cpu_terminal'] = terminal
    cpu = accept_terminal(context, terminal, 'cpu')
    unique_terminals(context['terminals'])
    return cpu


def native_start(context):
    qualified_origins(context)
    selected, source, trainer = context['selected'], context['selected']['source_driver'], context['trainer']
    prior = selected['source_cpu']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' and sys.flags.optimize == 0 and
            re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'unoptimized original CUDA-hidden unit required')
    python = Path(sys.executable).resolve()
    require(str(python) == prior['invocation']['python'] and
            bound_file(context['guards'], python, prior['invocation']['python_sha256']) == python and
            sys.version == prior['invocation']['python_version'], 'qualified interpreter differs')
    before = source.cgroup_memory()
    unit = Path(before['path']).name.removesuffix('.service')
    require(unit not in {t['unit'] for t in context['terminals']} and
            os.environ['INVOCATION_ID'] not in {t['invocation_id'] for t in context['terminals']},
            'distinct original evaluator unit required')
    selected['genuine']['reference'].admit_cgroup(before, unit)
    context['helper'].zero_events(before)
    import torch
    require(not torch.cuda.is_initialized(), 'CPU admission must precede CUDA')
    flags = prior['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(source.numerical_flags() == flags and not torch.cuda.is_available(), 'qualified flags/CUDA differs')
    value = selected['launch']['helpers']['packing']
    context['packing'] = trainer.load_helper('_genuine_view_evaluation_packing', value['path'], value['sha256'], context['guards'])
    context['reference_math'] = context['training'].reference_math(selected['math_context'])
    check_origins(context, source.imported_origins(selected['extract'], selected['packages']))
    context['features'] = trainer.training_features(selected)
    context['feature_state_sha256'] = context['training'].fingerprint(context['features'])
    return before


def label(endpoint):
    return endpoint['arm'] + '-' + str(endpoint['seed'])


def file_names(spec):
    return {label(e) + suffix for e in spec['endpoints'] for suffix in ('.raw.npy', '.unit.npy', '.packed.bin')}


def load_head(context, endpoint):
    """Complete NEW terminal payload and initializer validation before head-only use."""
    import torch
    trainer, selected, original = context['trainer'], context['selected'], context['training']
    record = context['records'][endpoint['seed'], endpoint['arm']]
    path = bound_file(context['guards'], endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    ident = saved['identity']
    require(context['legacy'].canonical(ident) == record['identity'], 'typed payload identity/receipt differs')
    trainer.check_payload(saved, ident, 1000)
    require(original.fingerprint(saved) == endpoint['terminal_state_sha256'] and saved['partition'] == context['partition'] and
            saved['original_rows'].tolist() == context['partition']['panels']['train']['original_rows'] and
            saved['target'].tolist() == context['target'] and
            original.fingerprint({n: saved[n] for n in trainer.STATIC_KEYS}) == ident['static_sha256'] and
            original.fingerprint({n: saved['head'][n] for n in ('center', 'preactivation_std')}) == ident['buffers_sha256'],
            'complete typed TRAIN partition/static/buffer binding differs')
    proof = selected['terminals']['cpu:control']
    initial_path = bound_file(context['guards'], proof['checkpoint']['path'], proof['checkpoint']['sha256'])
    initial = torch.load(initial_path, map_location='cpu', weights_only=True, mmap=True)
    require(initial['schema'] == trainer.SCHEMA and initial['source'] == selected['source'] and
            initial['partition'] == context['partition'] and initial['counter'] == 0 and
            initial['numerical_flags'] == record['numerical_flags'] and
            original.fingerprint(initial) == proof['cpu_state_sha256'] and
            original.fingerprint(initial['initial']) == proof['initial_state_sha256'] == original.fingerprint(saved['initializer']),
            'independent NEW TRAIN-only initializer differs')
    require(initial.keys() == {'schema', 'source', 'partition', 'initial', 'cpu_rng', 'numerical_flags', 'counter', 'fixtures'} and
            initial['fixtures'].keys() == {'calibration', 'gradients', 'resumes'} and
            original.fingerprint(initial['fixtures']) == proof['cpu_witnesses_sha256'] and
            saved['views'] == selected['views'] == initial['initial']['views'] and
            ident['feature_versions'] == {'canonical': 0, 'augmented': 0} and
            ident['feature_state_sha256'] == context['feature_state_sha256'],
            'complete CPU witnesses/paired TRAIN cache provenance differs')
    for seed in SEEDS:
        anchors, masks = trainer.schedule_and_masks(context['target'], seed)
        require(torch.equal(saved['schedules'][str(seed)], torch.from_numpy(anchors)) and
                torch.equal(saved['masks'][str(seed)], torch.from_numpy(masks)),
                'complete class-balanced schedules/independent masks differ')
    expected_positive = context['reference_math'].member_bank_positive_ordinals(saved['target'].numpy(), allow_singletons=True)
    schedule, masks = saved['schedules'][str(endpoint['seed'])], saved['masks'][str(endpoint['seed'])]
    require(torch.equal(saved['positive'], expected_positive) and
            original.fingerprint(schedule) == ident['schedule_sha256'] and
            original.fingerprint(masks) == ident['mask_sha256'] and
            float(saved['head']['preactivation_std']) > 0, 'positive ordinals/schedule/mask/buffer differs')
    for i, row in enumerate(record['steps']):
        batch = schedule[i].tolist()
        require(row['batch'] == batch and original.fingerprint(masks[i]) == row['mask_sha256'] and
                original.fingerprint({v: features[batch] for v, features in context['features'].items()}) == row['feature_rows_sha256'],
                'actual ordered update/cache/mask provenance differs')
    def finite(value):
        if isinstance(value, torch.Tensor):
            require(value.device.type == 'cpu' and not value.requires_grad and torch.isfinite(value).all().item(), 'finite CPU payload tensor required')
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                finite(item)
    finite(saved); finite(initial)
    digest = original.fingerprint(saved['head'])
    # BOTH intervention arms have phi=.5*z; never use old GELU candidate factory.
    head = selected['cached'].head_from('control', tensors=saved['head']).eval().requires_grad_(False)
    require(original.fingerprint(dict(head.state_dict())) == digest and
            [('compact_head.' + n, tuple(p.shape)) for n, p in head.named_parameters()] ==
            list(zip(trainer.PARAMETERS[:4], trainer.SHAPES[:4], strict=True)), 'strict independent FP32 head roles differ')
    del saved, initial, ident, anchors, masks, schedule, expected_positive
    gc.collect()
    return head, digest


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


def qualify_heads(context):
    """Two independent complete terminal reloads, TRAIN-only witnesses, no fitting."""
    source, original = context['selected']['source_driver'], context['training']
    cache = context['features']['canonical'][:64]  # Exact new TRAIN cache normalization, never historical FIT rows.
    witnesses, facts = {}, {}
    for endpoint in context['spec']['endpoints']:
        head, digest = load_head(context, endpoint)
        key = label(endpoint)
        values = context['legacy'].head_values(context, head, cache)
        witnesses[key] = {name: source.tensor_fact(value) for name, value in
                          zip(('raw', 'unit', 'codes', 'inverse_norms'), values, strict=True)}
        facts[key] = digest
        require(original.fingerprint(dict(head.state_dict())) == digest, 'TRAIN witness changed head')
        del head
        gc.collect()
        head, second_digest = load_head(context, endpoint)
        second = context['legacy'].head_values(context, head, cache)
        context['helper'].exact(values, second)
        require(original.fingerprint(values) == original.fingerprint(second), 'TRAIN raw/unit/packed bytes differ')
        require(second_digest == digest == original.fingerprint(dict(head.state_dict())), 'independent TRAIN witness/head reload differs')
        del head, values, second
        gc.collect()
    return {'head_facts': facts, 'train_witnesses': witnesses}


def replay_equal(expected, actual):
    require(actual == expected, 'independent per-query packed quality replay differs')


def packed_readback(guards, path, digest, expected):
    require(bound_file(guards, path, digest).read_bytes() == expected, 'independent packed wire differs')


def score_panel(context, cpu):
    import numpy as np
    import torch
    spec, original = context['spec'], context['training']
    panel = context['partition']['panels'][spec['panel']]
    cache = cache_rows(context, panel['original_rows'])  # Validation reached ONLY after authenticated selection GO.
    labels = tuple(context['fit']['class_names'][context['fit']['targets'][r]] for r in panel['original_rows'])
    context['scoring'].export = context['helper']
    fixed = context['scoring'].scoring_math({**context, 'root': Path(spec['evaluation_reference']['root'])})
    quality, files, replay_facts = {}, {}, {}
    for endpoint in spec['endpoints']:
        key, seed, arm = label(endpoint), str(endpoint['seed']), endpoint['arm']
        head, digest = load_head(context, endpoint)
        require(digest == cpu['head_facts'][key], 'CPU-qualified terminal scoring head differs')
        values = context['legacy'].head_values(context, head, cache)
        require(original.fingerprint(dict(head.state_dict())) == digest, 'panel forward changed head')
        raw, unit, codes, inverse = values
        for suffix, value in (('.raw.npy', raw), ('.unit.npy', unit)):
            path = context['args'].output / (key + suffix)
            with path.open('xb') as stream:
                np.save(stream, value.numpy(), allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        path = context['args'].output / (key + '.packed.bin')
        with path.open('xb') as stream:
            stream.write(context['packing'].pack_int8_unit_embeddings(unit).to_bytes()); stream.flush(); os.fsync(stream.fileno())
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        first = fixed.packed_quality(unit.numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu'))
        del head
        gc.collect()
        head, second_digest = load_head(context, endpoint)
        second = context['legacy'].head_values(context, head, cache)
        context['helper'].exact(values, second)
        require(original.fingerprint(values) == original.fingerprint(second), 'panel raw/unit/packed bytes differ')
        require(second_digest == digest == original.fingerprint(dict(head.state_dict())), 'independent full panel head reload differs')
        # Read back the saved raw/unit/packed wires, then independently rerank the second head.
        for suffix, value in (('.raw.npy', second[0]), ('.unit.npy', second[1])):
            path = bound_file(context['guards'], context['args'].output / (key + suffix), files[key + suffix])
            loaded = torch.from_numpy(np.load(path, allow_pickle=False))
            context['helper'].exact((loaded,), (value,))
            require(original.fingerprint(loaded) == original.fingerprint(value), 'serialized raw/unit bytes differ')
        packed_readback(context['guards'], context['args'].output / (key + '.packed.bin'), files[key + '.packed.bin'],
                        context['packing'].pack_int8_unit_embeddings(second[1]).to_bytes())
        second_quality = fixed.packed_quality(second[1].numpy(), labels, panel['query'], panel['gallery'], device=torch.device('cpu'))
        replay_equal(first, second_quality)
        quality.setdefault(seed, {})[arm] = first
        replay_facts[key] = {n: context['selected']['source_driver'].tensor_fact(v) for n, v in
                             zip(('raw', 'unit', 'codes', 'inverse_norms'), second, strict=True)}
        del head, values, second, raw, unit, codes, inverse
        gc.collect()
    deltas, average = averaged_deltas(quality, spec['stage'], spec['panel'])
    intervals = {}
    if spec['stage'] == 'first':
        each_seed = quality_pass = first_gate(deltas)
    else:
        for metric in METRICS:
            delta = np.asarray(average[metric])
            intervals[metric] = {'mean_delta': float(delta.mean())}
            for kind, groups in (('product', np.asarray(labels)[panel['query']]), ('query', np.arange(len(panel['query'])))):
                # Pinned helper resets PCG64 seed179019 each call: SAME5000 draws across metrics/signs.
                intervals[metric][kind + '_lower95'] = fixed.bootstrap_lower(delta, groups)
                intervals[metric][kind + '_upper95'] = -fixed.bootstrap_lower(-delta, groups)
        each_seed, quality_pass = quality_gate(deltas, intervals)
    cost_pass = all(v['pass'] for v in context['costs'].values())
    decision = ('CONTINUE' if spec['stage'] == 'first' else 'GO') if quality_pass and cost_pass else 'KILL'
    return {'decision': decision, 'quality': quality, 'each_seed_quality_pass': each_seed, 'quality_pass': quality_pass,
            'cost_pass': cost_pass, 'cost': context['costs'], 'cost_policy': COST_POLICY,
            'paired_seed_average_intervals': intervals, 'mean_deltas': {m: statistics.mean(average[m]) for m in METRICS},
            'bootstrap_draws': 0 if spec['stage'] == 'first' else 5000, 'bootstrap_seed': 179019,
            'panel_images': len(panel['original_rows']), 'query_images': len(panel['query']), 'gallery_images': len(panel['gallery']),
            'panel_products': len(panel['original_class_ids']), 'fit_images': 6355, 'fit_products': 1008,
            'quality_read': True, 'files': files, 'panel_replay_facts': replay_facts,
            'full_panel_raw_unit_packed_replay_exact': True, 'per_query_replay_exact': True,
            'metric_units': 'fractions; multiply deltas by100 for percentage points',
            'interval_scope': 'equal-seed per-query deltas; shared paired product/query draws conditional on frozen source',
            'cost_denominator': 'fresh original control TRAIN1000 whole service and median update, eachseed',
            'optimization_throughput_is_image_training_throughput': False, 'independent_pretraining_seeds': False,
            'intermediate_checkpoint_selection': False, 'previous_official_and_pareto_preserved': True, 'selection_previously_exposed': True,
            'canonical_panel_input_only': True, 'serving_view_averaging': False,
            'preparation_costs': preparation_costs(context['selected'])}


def exit_rehash(context):
    selected = context['selected']
    source = selected['source_driver']
    origins = source.imported_origins(selected['extract'], selected['packages'])
    check_origins(context, origins)
    selected['exporter'].rehash(selected['genuine'])
    for path, digest in context['guards'].items():
        bound_file({}, path, digest)
    return origins


def run(args):
    prior = None if args.prerequisite is None else {'path': str(args.prerequisite), 'sha256': args.prerequisite_sha256}
    require(sys.argv == cli_argv(args.authority, args.authority_sha256, args.execution_sha256, args.phase, args.output, prior),
            'fixed canonical CLI order required')
    context = authority(args)
    cpu = prerequisites(context)
    before = native_start(context)
    import torch
    source = context['selected']['source_driver']
    rng, flags = torch.random.get_rng_state().clone(), source.numerical_flags()
    args.output.mkdir()
    facts = qualify_heads(context)
    if cpu is not None:
        require(all(facts[k] == cpu[k] for k in facts), 'accepted evaluator CPU witnesses differ')
    result = score_panel(context, cpu) if args.phase == 'score' else {'quality_read': False, 'files': {}}
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags and not torch.cuda.is_initialized(),
            'whole-unit RNG/flags/CUDA differs')
    # Reject new foreign/conflicting origins before the original uncached exit reader registers anything.
    origins = source.imported_origins(context['selected']['extract'],
                                     context['selected']['packages'])
    check_origins(context, origins)
    origins = exit_rehash(context)
    check_origins(context, origins)
    require(closure(context['root'], args.execution_sha256, FILES, {}) == context['code'] and
            closure(Path(context['spec']['training']['root']), context['spec']['training']['execution_sha256'], TRAIN_FILES, {}) == context['spec']['training']['code'] and
            closure(Path(context['spec']['evaluation_reference']['root']), REFERENCE_EXECUTION_SHA, REFERENCE_PINS.keys(), {}) == REFERENCE_PINS,
            'exit separate evaluator/trainer/reference closure differs')
    resources = context['helper'].resources(context, args.phase, before)
    receipt = {**bind(context), 'schema': SCHEMA, 'phase': args.phase, 'pass': True, 'engineering_admission_pass': True,
        'authority': {'path': str(args.authority), 'sha256': args.authority_sha256}, 'output': str(args.output),
        'prerequisite': prior, 'cpu_terminal': context['cpu_terminal'], 'optimizer_updates': 0, 'optimizer_members': 5,
        'numerical_flags': flags, 'strict_independent_head_reload_exact': True, 'train_raw_unit_cpu_packed_exact': True,
        'first_heads_released_before_reload': True, 'rng_flags_preserved': True, 'cuda_initialized': False,
        'official_read': False, 'global_production_goal_met': False, 'public_latency_measured': False,
        'input_guards': context['guards'], 'origins': origins, 'exit_rehash_pass': True,
        'invocation': {'argv': sys.argv, 'python': str(Path(sys.executable).resolve()),
            'python_sha256': context['selected']['source_cpu']['invocation']['python_sha256'], 'python_version': sys.version,
            'optimize': sys.flags.optimize, 'pid': os.getpid(), 'invocation_id': os.environ['INVOCATION_ID'],
            'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        **facts, **result, **resources}
    check_receipt(context, receipt, args.phase)
    context['helper'].publish(args.output / 'receipt.json', receipt)
    require(time.perf_counter() - UNIT_STARTED < policy(args.phase)['seconds'], 'receipt included whole-unit cap differs')
    return receipt


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


def main():
    args = parser().parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Genuine view evaluation rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output), 'decision': result.get('decision')}))


if __name__ == '__main__':
    main()
