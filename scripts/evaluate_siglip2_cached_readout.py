#!/usr/bin/env python3
"""Prospective So400 cached-readout CPU120/export300/score300 evaluator.

Evaluator gates are UNRUN. The root owns source freeze, native units, final terminal
descriptors and decisions. No official/Pareto result is changed by this driver.

Freeze NEW execution.json with exactly FILES below. Its two scoring reference
files are byte-identical copies of the pinned previous held references. Keep
the cached trainer's original two-file execution root separate, unchanged;
its own original five-file trainer and transitive preparation remain separate.
training.execution_sha256 must be TRAIN_EXECUTION_SHA: the corrected v2
trainer2 closure (source commit148ab7d8), never the failed old v1 closure.
Do not invent hashes for future receipts, logs, checkpoints or authorities.

The common siglip2-cached-readout-evaluation-authority-v1 JSON has exactly
SPEC_KEYS. training is {root: canonical absolute directory, execution_sha256}.
endpoints is ordered control032,candidate032,candidate041,control041, each
exactly {seed, arm, launch: FILE, terminal: TERMINAL, checkpoint: FILE,
terminal_state_sha256}. launch is the ORIGINAL accepted TRAIN1000 authority.
frozen_split is the pinned original held preflight FILE, independently bound
from the native256 FIT authority's FIT-only teacher receipt. Policies
are {cpu: policy('cpu'), export: policy('export'), score: policy('score')},
cost_policy is COST_POLICY, both_locks_held is true. FILE is exactly
{path: canonical absolute file, sha256: actual lowercase SHA256}. TERMINAL is
exactly {receipt: FILE, log: FILE, unit, invocation_id, service_seconds,
native_peak_rss_kib, both_locks_held: true}, with the original FINAL_CGROUP
footer and complete normal-exit log. All four TRAIN1000 terminals and their
CPU/both mechanics/logs/checkpoints/full guards close before native admission.

Exact CLI (root supplies enclosing limits and BOTH lifetime locks):
CUDA_VISIBLE_DEVICES='' python -B ROOT/evaluate_siglip2_cached_readout.py
 --execution-sha256 SHA --authority JSON --authority-sha256 SHA
 --phase cpu --output NEW_DIRECTORY
CUDA_VISIBLE_DEVICES=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 python -B ROOT/evaluate_siglip2_cached_readout.py
 --execution-sha256 SHA --authority JSON --authority-sha256 SHA
 --phase export --output NEW_DIRECTORY --prerequisite CPU_TERMINAL_JSON
 --prerequisite-sha256 SHA
CUDA_VISIBLE_DEVICES='' python -B ROOT/evaluate_siglip2_cached_readout.py
 --execution-sha256 SHA --authority JSON --authority-sha256 SHA
 --phase score --output NEW_DIRECTORY --prerequisite COLLECTION_JSON
 --prerequisite-sha256 SHA

CPU_TERMINAL_JSON is a TERMINAL for the updated four-head CPU receipt.json.
COLLECTION_JSON has exactly {schema: 'siglip2-cached-readout-wires-v1',
authority_sha256, execution_sha256, cpu_terminal: TERMINAL,
export_terminal: TERMINAL, both_locks_held: true}. The single shared export
unit accepts ALL four wires together. Score admits BOTH originals before
loading arrays or reading quality. Every new phase includes full admission,
independent strict reload and uncached exit; no peak reset, partial reuse,
quality preview, tuning or best-seed selection. Failure leaves inadmissible
output. Parent finalizes terminal descriptors only after normal unit exit.

Encoder stays FP32/frozen1152: B32 FP16 autocast native256 pooled output is
normalized in FP32, then the FP32 rank32 head normalizes again. CPU reload
uses actual NEW five-member payloads, never the original208 optimizer.
Export saves held1152.npy plus each endpoint's raw/unit/packed files, with
full per-image parity from independently reloaded heads over the same cache.
This measures TRAIN-held only, not public latency or image-training throughput.
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

SCHEMA = 'siglip2-cached-readout-evaluation-v1'
AUTHORITY_SCHEMA = 'siglip2-cached-readout-evaluation-authority-v1'
SEEDS = (179032, 179041)
ARMS = ('control', 'candidate')
ORDER = ((179032, 'control'), (179032, 'candidate'), (179041, 'candidate'), (179041, 'control'))
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
PINS = {
    'export_siglip2_substrate_adaptation.py': '89eed215795492d8237d736077b8ee0d0b821ba07dfd3535c65f6643533b9863',
    'score_siglip2_substrate_adaptation.py': '5be3d922b198bd2f6470523cfdcde95884fd03600b9eec0dba876da6df27683f',
    'reference_compare_inshop_sop_warmstart_100.py': '8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250',
    'reference_score_inshop_crop_view_pair.py': '16e27ccaa7325b9ef7efdb3512cb95791a682afdd5ae59bd1f0847d63b87f5ed'}
FILES = {'evaluate_siglip2_cached_readout.py', 'test_siglip2_cached_readout_evaluation.py'} | PINS.keys()
TRAIN_PINS = {
    'train_siglip2_cached_readout.py': 'a687a62b78eeb4c122491f23394954ad192acc02394e29b66efb257d3a6f338c',
    'test_siglip2_cached_readout.py': '9fd8a780c412f9423e9e75192a1e8d91ead520d46f6269e77482965e85a28500'}
TRAIN_EXECUTION_SHA = '907dfed63ec7678b2ef930463640b098ea2e38ad1c5fc1ad312cceb628151598'
# Original split authenticated by pe_large_pool.control/train_inshop_pe_pair.check_startup.
FROZEN_SPLIT = {'path': '/home/riomus/runs/sfora-pe-augmented-100-v2/preflight.json',
                'sha256': '41b5fe09448163d755d278131427a1d5bc4663855d544501487489fbe0813293'}
SPEC_KEYS = {'schema', 'execution_sha256', 'training', 'endpoints', 'frozen_split',
             'resource_policies', 'cost_policy', 'both_locks_held'}
COST_POLICY = {'whole_service_ratio_max': 1.50, 'median_update_ratio_max': 1.50,
               'training_wall_ratio': 'report_only'}
PRECISION = 'FP32-frozen-vision/FP16-autocast/normalized-pooled/FP32-second-normalized-head'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def policy(phase):
    require(phase in ('cpu', 'export', 'score'), 'fixed evaluation phase required')
    value = {'seconds': 120 if phase == 'cpu' else 300, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0}
    value.update({'cuda_allocated_bytes_exclusive': 10_000_000_000} if phase == 'export'
                 else {'cuda_visible_devices': ''})
    return value


def canonical(value):
    """Compare JSON receipts only; the original typed fingerprint stays intact."""
    return json.loads(json.dumps(value, allow_nan=False))


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result
    def constant(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


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
    path = bound_file(guards, value['path'], value['sha256'])
    with path.open('rb') as stream:
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


def check_spec(spec, args):
    require(spec.keys() == SPEC_KEYS and spec['schema'] == AUTHORITY_SCHEMA and
            spec['execution_sha256'] == args.execution_sha256 and spec['both_locks_held'] is True and
            spec['resource_policies'] == {p: policy(p) for p in ('cpu', 'export', 'score')} and
            spec['cost_policy'] == COST_POLICY, 'evaluation authority profile differs')
    require(spec['training'].keys() == {'root', 'execution_sha256'} and
            spec['training']['execution_sha256'] == TRAIN_EXECUTION_SHA, 'exact corrected training REFERENCE required')
    require(spec['frozen_split'] == FROZEN_SPLIT, 'original frozen split authority differs')
    require([(e['seed'], e['arm']) for e in spec['endpoints']] == list(ORDER), 'four ordered endpoints required')
    for endpoint in spec['endpoints']:
        require(endpoint.keys() == {'seed', 'arm', 'launch', 'terminal', 'checkpoint', 'terminal_state_sha256'} and
                all(endpoint[k].keys() == {'path', 'sha256'} for k in ('launch', 'checkpoint')) and
                re.fullmatch('[0-9a-f]{64}', endpoint['terminal_state_sha256']), 'exact endpoint descriptor required')


def check_endpoint(cached, record, endpoint, context):
    seed, arm, launch = endpoint['seed'], endpoint['arm'], record['launch']
    cached.check_launch(launch, SimpleNamespace(phase='train', arm=arm, seed=seed,
                        execution_sha256=context['launch']['execution_sha256']))
    require(record['schema'] == cached.SCHEMA and record['phase'] == 'train' and record['arm'] == arm and
            record['seed'] == seed and record['completed_step'] == 1000 and record['optimizer_members'] == 5 and
            record['source'] == context['source'] and cached.method(launch) == cached.method(context['launch']) and
            launch['selected_cpu'] == context['launch']['selected_cpu'] and
            launch['selected_mechanics'] == context['launch']['selected_mechanics'] and
            record['terminal_cgroups'] == context['terminal_cgroups'] and record['checkpoint'] == endpoint['checkpoint'] and
            record['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and
            record['resource_policy'] == cached.policy('train') and record['code'] == context['code'] and
            record['numerical_flags'] == context['old_cpu']['numerical_flags'] and
            record['head_scalars'] == 188544 and record['trainable_scalars'] == 445056 and
            record['encoder_updates'] == 0 and record['resumed_steps'] == [], 'complete TRAIN1000 endpoint differs')
    require(all(record[k] is True for k in ('pass', 'training_qualified', 'strict_reload_exact', 'exit_rehash_pass', 'fixed_cached_views')) and
            all(record[k] is False for k in ('quality_read', 'trained_state_reused', 'training_state_discarded', 'image_augmentation', 'replay_exact')) and
            0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000, 'fresh complete TRAIN1000 acceptance differs')
    ident = record['identity']
    require(ident['method'] == cached.method(launch) and ident['arm'] == arm and ident['seed'] == seed and
            ident['source'] == context['source'] and ident['selected_cpu'] == launch['selected_cpu'] and
            ident['parameter_names'] == cached.PARAMETERS and ident['numerical_flags'] == record['numerical_flags'] and
            ident['optimizer_defaults'] == context['old_cpu']['state']['optimizer_defaults'] and
            ident['optimizer_groups'] == [dict(ident['optimizer_defaults'], lr=1e-4)] * 2 and
            ident['optimizer_serial_groups'] == [dict(ident['optimizer_groups'][0], params=[0, 1, 2, 3]),
                                                  dict(ident['optimizer_groups'][1], params=[4])],
            'five-member endpoint identity differs')
    cached.check_steps(record['steps'], 1, 1000)
    require(record['steps'][-1]['state_sha256'] == endpoint['terminal_state_sha256'] and
            all(row['schedule_sha256'] == ident['schedule_sha256'] for row in record['steps']) and
            record['median_update_seconds'] == statistics.median(row['seconds'] for row in record['steps'][2:]) and
            0 < sum(row['seconds'] for row in record['steps']) <= record['training_wall_seconds'] <= record['wall_seconds'],
            'terminal state/schedule/update cost differs')
    if seed == SEEDS[0]:
        mechanics = context['terminals']['mechanics:' + arm]
        require(record['initial_state_sha256'] == mechanics['initial_state_sha256'] and
                all(cached.diagnostic(a) == cached.diagnostic(b) for a, b in
                    zip(record['steps'][:17], mechanics['steps'], strict=True)), 'fresh original mechanics replay differs')
    reconstruction = record['affine_reconstruction']
    require(reconstruction.keys() == {'residual_energy', 'fit_unexplained_energy_fraction'} and
            type(reconstruction['residual_energy']) in (int, float) and math.isfinite(reconstruction['residual_energy']) and
            reconstruction['residual_energy'] >= 0 and
            (reconstruction['fit_unexplained_energy_fraction'] is None if reconstruction['residual_energy'] == 0 else
             type(reconstruction['fit_unexplained_energy_fraction']) in (int, float) and
             math.isfinite(reconstruction['fit_unexplained_energy_fraction']) and reconstruction['fit_unexplained_energy_fraction'] >= 0),
            'FIT affine reconstruction diagnostic differs')


def paired_cost(records):
    require(set(records) == set(ORDER), 'all four original endpoint costs required')
    result = {}
    for seed in SEEDS:
        control, candidate = (records[seed, arm] for arm in ARMS)
        require(all(control['identity'][k] == candidate['identity'][k] for k in
                    ('static_sha256', 'buffers_sha256', 'feature_state_sha256', 'schedule_sha256')) and
                all(all(a[k] == b[k] for k in ('step', 'batch', 'schedule_sha256', 'feature_rows_sha256'))
                    for a, b in zip(control['steps'], candidate['steps'], strict=True)), 'paired cached inputs differ')
        for value in (control, candidate):
            require(all(type(value[k]) in (int, float) and math.isfinite(value[k]) and value[k] > 0
                        for k in ('service_seconds', 'median_update_seconds', 'training_wall_seconds')), 'finite original endpoint cost required')
        ratios = {name: candidate[key] / control[key] for name, key in
                  (('whole_service_ratio', 'service_seconds'), ('median_update_ratio', 'median_update_seconds'),
                   ('training_wall_ratio', 'training_wall_seconds'))}
        require(all(math.isfinite(v) and v > 0 for v in ratios.values()) and
                ratios['whole_service_ratio'] <= 1.50 and ratios['median_update_ratio'] <= 1.50,
                'fresh whole-service/median cost gate failed')
        result[str(seed)] = {**ratios, 'training_wall_ratio_gate': False,
                            **{arm: {k: records[seed, arm][k] for k in
                               ('service_seconds', 'median_update_seconds', 'training_wall_seconds')} for arm in ARMS}}
    return result


def authority(args):
    """Metadata/bytes for every endpoint close before native imports or held decode."""
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native imports preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    code = closure(root, args.execution_sha256, FILES, guards)
    require(all(code[n] == h for n, h in PINS.items()), 'pinned previous evaluator/reference bytes differ')
    spec = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    check_spec(spec, args)
    output, train_root = args.output, Path(spec['training']['root'])
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and
            not output.is_symlink(), 'exclusive canonical output required')
    train_code = closure(train_root, spec['training']['execution_sha256'], TRAIN_PINS.keys(), guards)
    require(train_code == TRAIN_PINS and not root.is_relative_to(train_root) and not train_root.is_relative_to(root),
            'separate original cached trainer2 closure required')
    cached = load_bare('_cached_evaluation_trainer', train_root / 'train_siglip2_cached_readout.py', TRAIN_PINS['train_siglip2_cached_readout.py'])
    require(cached.FILES == TRAIN_PINS.keys() and cached.SEEDS == SEEDS and cached.ARMS == ARMS, 'original cached trainer profile differs')
    first = spec['endpoints'][0]
    selected_args = SimpleNamespace(execution_sha256=spec['training']['execution_sha256'],
        authority=Path(first['launch']['path']), authority_sha256=first['launch']['sha256'],
        phase='train', arm='control', seed=SEEDS[0], output=output)
    selected = cached.authority(selected_args)  # Genuine CPU, BOTH mechanics and all original preparation predicates.
    original = selected['original']
    admission = original.FlatAdmission()
    admission.init = selected['initialized']['init']
    helper = load_bare('_cached_evaluation_held_helpers', root / 'export_siglip2_substrate_adaptation.py', PINS['export_siglip2_substrate_adaptation.py'])
    scoring = load_bare('_cached_evaluation_score_helpers', root / 'score_siglip2_substrate_adaptation.py', PINS['score_siglip2_substrate_adaptation.py'])
    require(all(code[n] == pin['source'] for n, pin in helper.REFERENCES.items()), 'fixed packed scoring reference differs')
    required_guards = {p: h for p, h in selected['guards'].items() if p != str(selected_args.authority)}
    records, terminals = {}, [selected['launch']['selected_cpu'], *selected['launch']['selected_mechanics'].values()]
    for final in selected['terminal_cgroups'].values():
        helper.zero_events(final)
    for arm in ARMS:
        helper.logged_steps(Path(selected['launch']['selected_mechanics'][arm]['log']['path']),
                            selected['terminals']['mechanics:' + arm]['steps'] + selected['terminals']['mechanics:' + arm]['resumed_steps'])
    for endpoint in spec['endpoints']:
        launch = admission.descriptor_json(endpoint['launch'], selected['guards'])
        record = admission.descriptor_json(endpoint['terminal']['receipt'], selected['guards'])
        require(record['launch'] == launch and record['authority'] == endpoint['launch'] and
                record['authority_sha256'] == endpoint['launch']['sha256'] and
                record['execution_sha256'] == spec['training']['execution_sha256'], 'original endpoint authority differs')
        check_endpoint(cached, record, endpoint, selected)
        terminal = endpoint['terminal']
        invocation, prior = record['invocation'], selected['old_cpu']['invocation']
        require(invocation['argv'] == [str(train_root / 'train_siglip2_cached_readout.py'),
                '--execution-sha256', spec['training']['execution_sha256'], '--authority', endpoint['launch']['path'],
                '--authority-sha256', endpoint['launch']['sha256'], '--phase', 'train', '--arm', endpoint['arm'],
                '--seed', str(endpoint['seed']), '--output', str(Path(terminal['receipt']['path']).parent)] and
                all(invocation[k] == prior[k] for k in ('python', 'python_sha256', 'python_version')) and
                invocation['cuda_visible_devices'] not in (None, '') and invocation['cublas_workspace_config'] == ':4096:8' and
                Path(terminal['receipt']['path']).name == 'receipt.json' and
                Path(endpoint['checkpoint']['path']) == Path(terminal['receipt']['path']).parent / 'resume.pt', 'original TRAIN invocation/path differs')
        require(all(record['input_guards'].get(p) == h for p, h in required_guards.items()), 'complete prerequisite input guards differ')
        helper.zero_events(admission.admit_terminal(record, terminal, 300, selected['guards']))
        helper.logged_steps(Path(terminal['log']['path']), record['steps'])
        for path, digest in record['input_guards'].items():
            admission.bound_file(selected['guards'], path, digest)
        admission.bound_file(selected['guards'], endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])
        records[endpoint['seed'], endpoint['arm']] = {**record, 'service_seconds': terminal['service_seconds']}
        terminals.append(terminal)
    require(len({t['unit'] for t in terminals}) == len(terminals) and
            len({t['invocation_id'] for t in terminals}) == len(terminals) and
            len({e['checkpoint']['path'] for e in spec['endpoints']}) == 4, 'distinct original whole units/checkpoints required')
    costs = paired_cost(records)
    fit = selected['initialized']['source_context']['fit']
    frozen = admission.descriptor_json(spec['frozen_split'], selected['guards'])
    helper.validate_split(frozen, fit)
    for path, digest in guards.items():
        admission.bound_file(selected['guards'], path, digest)
    external = (train_root, selected['old']['root'], selected['initialized']['root'],
                selected['initialized']['pca']['root'], selected['initialized']['source_context']['root'],
                selected['initialized']['source_context']['own_root'])
    require(all(not root.is_relative_to(p) and not p.is_relative_to(root) and not output.is_relative_to(p) for p in external) and
            not output.is_relative_to(root) and all(not output.is_relative_to(Path(t['receipt']['path']).parent) for t in terminals),
            'separate immutable closures/output required')
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import during endpoint admission')
    return {'args': args, 'root': root, 'code': code, 'spec': spec, 'cached': cached, 'selected': selected,
            'guards': selected['guards'], 'helper': helper, 'scoring': scoring, 'admission': admission,
            'records': records, 'costs': costs, 'fit': fit, 'frozen': frozen, 'terminals': terminals,
            'training': original, 'selected_context': selected['old'], 'unit_started': UNIT_STARTED}


def bind(context):
    return {'authority_sha256': context['args'].authority_sha256, 'execution_sha256': context['args'].execution_sha256,
            'endpoints': context['spec']['endpoints'], 'source': context['selected']['source'],
            'frozen_split': context['spec']['frozen_split'], 'source_code': context['code']}


def check_receipt(context, record, phase):
    require(all(record[k] == v for k, v in bind(context).items()) and record['schema'] == SCHEMA and
            record['phase'] == phase and record['resource_policy'] == policy(phase) and
            record['optimizer_updates'] == 0 and record['optimizer_members'] == 5 and
            all(record[k] is True for k in ('pass', 'strict_independent_head_reload_exact', 'source_reload_exact',
                'fit_raw_unit_packed_exact', 'first_heads_released_before_reload', 'source_head_rng_flags_preserved', 'exit_rehash_pass')) and
            all(record[k] is False for k in ('quality_read', 'official_read', 'claim_eligible', 'public_serving_qualified', 'public_latency_measured')),
            'accepted updated CPU/export receipt differs')
    require(record['numerical_flags'] == context['selected']['old_cpu']['numerical_flags'] and
            record['invocation']['argv'] == cli_argv(Path(record['authority']['path']), record['authority']['sha256'],
                record['execution_sha256'], phase, Path(record['output']), record['prerequisite']) and
            all(record['invocation'][k] == context['selected']['old_cpu']['invocation'][k]
                for k in ('python', 'python_sha256', 'python_version')) and
            record['invocation']['cuda_visible_devices'] == ('' if phase == 'cpu' else record['invocation']['cuda_visible_devices']) and
            (phase != 'export' or record['invocation']['cuda_visible_devices'] not in (None, '') and
             record['invocation']['cublas_workspace_config'] == ':4096:8'), 'original evaluator invocation/flags differ')
    require(record['authority'] == {'path': str(context['args'].authority), 'sha256': context['args'].authority_sha256} and
            record['output'] == str(Path(record['output']).resolve()), 'original evaluator authority/output differs')
    if phase == 'cpu':
        require(record['cuda_initialized'] is False and record['peak_cuda_allocated_bytes'] == 0 and
                record['prerequisite'] is None and record['files'] == {}, 'CPU-only qualification differs')
    else:
        require(record['full_held_independent_raw_unit_packed_exact'] is True and record['batch'] == 32 and
                record['precision'] == PRECISION and record['cache_shape'] == [12599, 1152] and
                (record['query_images'], record['gallery_images'], record['held_products']) == (6354, 6245, 1993) and
                record['batch_sizes'] == batch_sizes(context) and set(record['files']) == file_names() and
                0 < record['peak_cuda_allocated_bytes'] < 10_000_000_000, 'all four full held export wires differ')


def accept_terminal(context, terminal, phase):
    record = read_json(terminal['receipt'], context['guards'])
    check_receipt(context, record, phase)
    require(Path(terminal['receipt']['path']) == Path(record['output']) / 'receipt.json', 'original evaluator receipt role differs')
    context['helper'].zero_events(context['admission'].admit_terminal(record, terminal, policy(phase)['seconds'], context['guards']))
    for path, digest in record['input_guards'].items():
        context['admission'].bound_file(context['guards'], path, digest)
    require(all(record['input_guards'].get(p) == h for p, h in context['evaluation_prereq_guards'].items()),
            'CPU/export complete input guards differ')
    return record


def prerequisites(context):
    args = context['args']
    context['evaluation_prereq_guards'] = context['guards'].copy()
    require((args.prerequisite is None) == (args.phase == 'cpu') and
            (args.prerequisite_sha256 is None) == (args.phase == 'cpu'), 'phase prerequisite differs')
    if args.phase == 'cpu':
        return None, None
    prior = read_json({'path': str(args.prerequisite), 'sha256': args.prerequisite_sha256}, context['guards'])
    if args.phase == 'export':
        cpu_terminal, export_terminal = prior, None
    else:
        require(prior.keys() == {'schema', 'authority_sha256', 'execution_sha256', 'cpu_terminal', 'export_terminal', 'both_locks_held'} and
                prior['schema'] == 'siglip2-cached-readout-wires-v1' and prior['both_locks_held'] is True and
                prior['authority_sha256'] == args.authority_sha256 and prior['execution_sha256'] == args.execution_sha256,
                'original four-wire collection differs')
        cpu_terminal, export_terminal = prior['cpu_terminal'], prior['export_terminal']
    cpu = accept_terminal(context, cpu_terminal, 'cpu')
    export = None
    if export_terminal is not None:
        export = accept_terminal(context, export_terminal, 'export')
        require(export['cpu_terminal'] == cpu_terminal and
                read_json(export['prerequisite'], context['guards']) == cpu_terminal and
                all(export[k] == cpu[k] for k in ('head_facts', 'source_facts', 'fit_witnesses', 'fit_pixels')),
                'export differs from original updated CPU qualification')
        for name, digest in export['files'].items():
            bound_file(context['guards'], Path(export['output']) / name, digest)
    terminals = context['terminals'] + [cpu_terminal] + ([export_terminal] if export_terminal else [])
    require(len({t['unit'] for t in terminals}) == len(terminals) and
            len({t['invocation_id'] for t in terminals}) == len(terminals), 'distinct original CPU/export units required')
    context['cpu_terminal'] = cpu_terminal
    return cpu, export


def native_start(context):
    """Packages/interpreter/limits/flags qualified before any native execution."""
    helper, selected = context['helper'], context['selected']
    context['cpus'] = {'so400': selected['old_cpu']}
    context['args'].arm = 'so400'
    before = helper.native_start(context, context['args'].phase)
    helper.zero_events(before)
    source = selected['initialized']['source']
    origins = source.imported_origins(selected['initialized']['source_context']['extract'], selected['initialized']['packages'])
    require(all(selected['old_cpu']['origins']['files'].get(p) == h for p, h in origins['files'].items()),
            'actual native imports differ from original qualified origins')
    return before


def load_head(context, endpoint):
    """Validate the NEW complete five-member payload, then own a strict head only."""
    import torch
    cached, selected, original = context['cached'], context['selected'], context['training']
    record = context['records'][endpoint['seed'], endpoint['arm']]
    path = bound_file(context['guards'], endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    ident = saved['identity']  # Keep tuples and integer optimizer keys for the typed hash.
    require(canonical(ident) == record['identity'], 'typed payload identity/receipt differs')
    cached.check_payload(saved, ident, 1000)
    require(original.fingerprint(saved) == endpoint['terminal_state_sha256'], 'complete typed TRAIN1000 state differs')
    require(ident['parameter_names'] == cached.PARAMETERS and ident['source'] == selected['source'] and
            saved['target'].tolist() == context['fit']['targets'] and
            original.fingerprint({n: saved[n] for n in ('pca', 'target', 'positive', 'schedules')}) == ident['static_sha256'] and
            original.fingerprint({n: saved['head'][n] for n in ('center', 'preactivation_std')}) == ident['buffers_sha256'],
            'complete tensor roles/static/buffer binding differs')
    source = selected['initialized']['source']
    for name, value in saved['pca'].items():
        require(source.tensor_fact(value) == selected['old_cpu']['state']['arrays'][name], 'original PCA tensor differs')
    require(source.tensor_fact(saved['target']) == selected['old_cpu']['state']['arrays']['target'], 'original FIT target differs')
    for seed in SEEDS:
        expected = torch.from_numpy(cached.schedule(context['fit']['targets'], seed))
        require(torch.equal(saved['schedules'][str(seed)], expected) and
                source.tensor_fact(expected[:100]) == selected['old_cpu']['state']['schedules'][str(seed)],
                'complete1000/original100 schedules differ')
    expected_positive = context['reference_math'].member_bank_positive_ordinals(saved['target'].numpy(), allow_singletons=True)
    require(torch.equal(saved['positive'], expected_positive) and
            original.fingerprint(saved['schedules'][str(endpoint['seed'])]) == ident['schedule_sha256'] and
            float(saved['head']['preactivation_std']) > 0, 'positive ordinals/schedule/init buffer differs')
    def finite(value):
        if isinstance(value, torch.Tensor):
            require(value.device.type == 'cpu' and not value.requires_grad and
                    torch.isfinite(value).all().item(), 'finite independent CPU payload tensor required')
        elif isinstance(value, dict):
            for item in value.values():
                finite(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                finite(item)
    finite(saved)
    digest = original.fingerprint(saved['head'])
    head = cached.head_from(endpoint['arm'], tensors=saved['head']).eval().requires_grad_(False)
    require(original.fingerprint(dict(head.state_dict())) == digest and
            [('compact_head.' + n, tuple(p.shape)) for n, p in head.named_parameters()] ==
            list(zip(cached.PARAMETERS[:4], cached.SHAPES[:4], strict=True)), 'strict independent FP32 head roles differ')
    del saved, ident, expected, expected_positive, value
    gc.collect()
    return head, digest


def source_facts(context, model):
    return context['training'].fingerprint({'vision': model.state_dict(), 'buffers': dict(model.named_buffers())})


def frozen_source(context, model):
    import torch
    require(len(list(model.named_parameters())) == 448 and all(not p.requires_grad and p.grad is None and
            p.dtype == torch.float32 for p in model.parameters()) and all(not m.training and
            not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks for m in model.modules()) and
            model.config._attn_implementation == 'sdpa' and not getattr(model.encoder, 'gradient_checkpointing', False) and
            dict(model.named_buffers()).keys() == {'embeddings.position_ids'} and
            'position_ids' in model.embeddings._non_persistent_buffers_set and
            torch.equal(model.embeddings.position_ids.cpu(), torch.arange(256).expand(1, -1)), 'frozen FP32 So400448/nonpersistent source differs')


def load_source(context, fresh=False):
    """Original derived source versus independently loaded source checkpoint only."""
    import torch
    selected, original = context['selected'], context['training']
    source, src = selected['initialized']['source'], selected['initialized']['source_context']
    if fresh:
        model, processor, roles = source.fresh_source(src)
    else:
        path = bound_file(context['guards'], selected['source']['checkpoint']['path'], selected['source']['checkpoint']['sha256'])
        disk = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        require(disk.keys() == {'vision', 'buffers', 'config', 'runtime', 'cpu_rng'} and
                disk['runtime'] == src['proof']['runtime'] and disk['config'] == disk['runtime']['config'] and
                source.tensor_fact(disk['cpu_rng']) == src['proof']['cpu_rng'] and
                disk['vision'].keys() == src['expected'].keys() and len(disk['vision']) == 448 and
                disk['buffers'].keys() == {'embeddings.position_ids'}, 'complete original source checkpoint differs')
        model = source.construct(disk['config'], src)
        with path.open('rb') as stream:
            pages = original.CheckpointPages(stream)
            original.load_vision(model, disk['vision'], pages)
            with torch.no_grad():
                for name, value in model.named_buffers():
                    require(source.tensor_fact(value) == source.tensor_fact(disk['buffers'][name]), 'original nonpersistent buffer bytes differ')
                    value.copy_(disk['buffers'][name]); pages.consume(disk['buffers'][name])
        del disk, value
        gc.collect()
        roles = source.configure_roles(model, src['expected'], model.config.num_hidden_layers)
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained(src['entry']['input']['preprocessor']['path'], local_files_only=True, backend='torchvision')
    require(source.model_facts(model, processor, roles, selected['initialized']['packages']) == src['proof']['runtime'],
            'actual original config/runtime/processor/roles differ')
    pixels, pooled, sample = source.pixels_and_raw(src, model, processor)
    require(sample == src['proof']['sample'], 'original source FIT witness differs')
    model.requires_grad_(False).eval()
    frozen_source(context, model)
    facts = source_facts(context, model)
    require(facts == selected['terminals']['cpu:control']['source_state_sha256'], 'original frozen source state differs')
    return model, processor, pixels, pooled, facts


def head_values(context, head, cache):
    import torch
    from torch.nn import functional as F
    require(all(not p.requires_grad and p.grad is None and p.dtype == torch.float32 for p in head.parameters()) and
            all(not m.training and not m._forward_hooks and not m._forward_pre_hooks and not m._backward_hooks
                for m in head.modules()), 'independent frozen FP32 head modes/roles differ')
    with torch.no_grad(), torch.autocast(device_type=cache.device.type, enabled=False):
        raw = head(cache.float())
        unit = F.normalize(raw, dim=1)
    require(raw.dtype == unit.dtype == torch.float32 and tuple(raw.shape) == (len(cache), 128) and
            torch.isfinite(raw).all().item() and torch.all(torch.linalg.vector_norm(raw, dim=1) > 0).item(),
            'finite nonzero FP32 second-normalized head output required')
    packed = context['packing'].pack_int8_unit_embeddings(unit.cpu())
    return raw.cpu(), unit.cpu(), packed.codes.cpu(), packed.inverse_norms.cpu()


def qualify_heads(context):
    """All updated heads and ORIGINAL source qualify on FIT only, without updates."""
    import torch
    from torch.nn import functional as F
    source = context['selected']['initialized']['source']
    original = context['training']
    model, processor, pixels, pooled, facts = load_source(context, fresh=True)
    cache = F.normalize(pooled.float(), dim=1)
    witnesses, head_facts = {}, {}
    for endpoint in context['spec']['endpoints']:
        head, digest = load_head(context, endpoint)
        key = label(endpoint)
        witnesses[key] = head_values(context, head, cache)
        head_facts[key] = digest
        require(original.fingerprint(dict(head.state_dict())) == digest, 'FIT forward changed updated head')
        del head
    require(source_facts(context, model) == facts, 'FIT forward changed frozen source')
    del model, processor, pooled, cache
    gc.collect()
    model, processor, second_pixels, pooled, second_facts = load_source(context)
    require(facts == second_facts and torch.equal(pixels, second_pixels), 'independent original source reload differs')
    cache = F.normalize(pooled.float(), dim=1)
    for endpoint in context['spec']['endpoints']:
        head, digest = load_head(context, endpoint)
        require(digest == head_facts[label(endpoint)], 'strict updated head reload differs')
        context['helper'].exact(witnesses[label(endpoint)], head_values(context, head, cache))
        require(original.fingerprint(dict(head.state_dict())) == digest, 'independent FIT forward changed head')
        del head
    require(source_facts(context, model) == facts, 'independent FIT forward changed source')
    witness_facts = {key: {name: source.tensor_fact(value) for name, value in
                     zip(('raw', 'unit', 'codes', 'inverse_norms'), values, strict=True)} for key, values in witnesses.items()}
    result = {'head_facts': head_facts, 'source_facts': facts, 'fit_witnesses': witness_facts,
              'fit_pixels': source.tensor_fact(pixels)}
    del second_pixels, pooled, cache, witnesses
    gc.collect()
    return model, processor, result


def label(endpoint):
    return endpoint['arm'] + '-' + str(endpoint['seed'])


def file_names():
    return {'held1152.npy'} | {arm + '-' + str(seed) + suffix for seed, arm in ORDER
                              for suffix in ('.raw.npy', '.unit.npy', '.packed.bin')}


def batch_sizes(context):
    return {name: [min(32, len(context['frozen'][name]) - start) for start in range(0, len(context['frozen'][name]), 32)]
            for name in ('query', 'gallery')}


def batches(context):
    for name in ('query', 'gallery'):
        rows = context['frozen'][name]
        for start in range(0, len(rows), 32):
            yield rows[start:start + 32]


def export_wires(context, model, processor, cpu_facts):
    """One shared image encoder pass, then independent four-head cache replay."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    output, original = context['args'].output, context['training']
    cache = np.empty((12599, 1152), dtype=np.float32)
    dataset = Path(context['fit']['dataset_root'])
    held_paths = [(dataset / r['relative_path']).resolve() for r in context['frozen']['held_manifest']]
    require(len(set(held_paths)) == 12599 and all(p.is_relative_to(dataset) for p in held_paths),
            'held image canonical roles/aliases differ')
    values = {label(e): (np.empty((12599, 128), dtype=np.float32), np.empty((12599, 128), dtype=np.float32),
                       np.empty((12599, 128), dtype=np.int8), np.empty(12599, dtype=np.float16)) for e in context['spec']['endpoints']}
    heads = {}
    for endpoint in context['spec']['endpoints']:
        head, digest = load_head(context, endpoint)
        require(digest == cpu_facts['head_facts'][label(endpoint)], 'CPU-qualified head differs before held decode')
        heads[label(endpoint)] = head.cuda()
    del head
    model.cuda()  # FP32 storage remains unchanged; only the encoder forward autocasts.
    cuda_rng = [v.clone() for v in torch.cuda.get_rng_state_all()]
    frozen_source(context, model)
    require(source_facts(context, model) == cpu_facts['source_facts'], 'CUDA move changed frozen source')
    for rows in batches(context):
        pixels = context['helper'].decode(context, processor, [context['frozen']['held_manifest'][i] for i in rows]).cuda()
        require(all((dataset / context['frozen']['held_manifest'][i]['relative_path']).resolve() == held_paths[i] for i in rows),
                'held image resolution changed during export')
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16):
            pooled = model(pixel_values=pixels).pooler_output
        with torch.no_grad(), torch.autocast(device_type='cuda', enabled=False):
            cached = F.normalize(pooled.float(), dim=1)
        require(cached.shape == (len(rows), 1152) and torch.isfinite(cached).all().item() and
                torch.all(torch.linalg.vector_norm(cached, dim=1) > 0).item(), 'native256 normalized held1152 output differs')
        cache[rows] = cached.cpu().numpy()
        for key, head in heads.items():
            for array, value in zip(values[key], head_values(context, head, cached), strict=True):
                array[rows] = value.numpy()
        del pixels, pooled, cached, value, array, head
    for key, head in heads.items():
        require(original.fingerprint(dict(head.state_dict())) == cpu_facts['head_facts'][key], 'held forward changed head')
    del heads, head
    gc.collect(); torch.cuda.empty_cache()
    for endpoint in context['spec']['endpoints']:
        head, digest = load_head(context, endpoint)
        key = label(endpoint)
        require(digest == cpu_facts['head_facts'][key], 'independent held head reload differs')
        head.cuda()
        for rows in batches(context):
            actual = head_values(context, head, torch.from_numpy(cache[rows]).cuda())
            context['helper'].exact(tuple(torch.from_numpy(array[rows].copy()) for array in values[key]), actual)
        require(original.fingerprint(dict(head.state_dict())) == digest, 'independent full held replay changed head')
        del head, actual
        gc.collect(); torch.cuda.empty_cache()
    frozen_source(context, model)
    require(source_facts(context, model) == cpu_facts['source_facts'], 'complete held pass changed frozen source')
    require(all(torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True)),
            'complete held forward/reload changed CUDA RNG')
    require(all((dataset / row['relative_path']).resolve() == path for row, path in
                zip(context['frozen']['held_manifest'], held_paths, strict=True)), 'exit held image resolution changed')
    def save_array(path, value):
        with path.open('xb') as stream:
            np.save(stream, value, allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
    save_array(output / 'held1152.npy', cache)
    for key, (raw, unit, codes, inverse) in values.items():
        packed = context['packing'].pack_int8_unit_embeddings(torch.from_numpy(unit))
        require(np.array_equal(codes, packed.codes.numpy()) and
                np.array_equal(inverse.view(np.uint16), packed.inverse_norms.numpy().view(np.uint16)), 'full packed wire inverse bits differ')
        save_array(output / (key + '.raw.npy'), raw)
        save_array(output / (key + '.unit.npy'), unit)
        with (output / (key + '.packed.bin')).open('xb') as stream:
            stream.write(packed.to_bytes()); stream.flush(); os.fsync(stream.fileno())
    files = {name: context['helper'].sha(output / name) for name in sorted(file_names())}
    for name, digest in files.items():
        bound_file(context['guards'], output / name, digest)
    return {'files': files, 'cpu_terminal': context['cpu_terminal'], 'cache_shape': [12599, 1152],
            'query_images': 6354, 'gallery_images': 6245, 'held_products': 1993, 'batch': 32,
            'batch_sizes': batch_sizes(context), 'precision': PRECISION,
            'full_held_independent_raw_unit_packed_exact': True}


def score_wires(context, receipt):
    """Only accepted full-export bytes enter pinned packed R1/AP/tie/bootstrap math."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    run = Path(receipt['output'])
    cache = np.load(run / 'held1152.npy', allow_pickle=False)
    require(cache.shape == (12599, 1152) and cache.dtype == np.float32 and np.isfinite(cache).all() and
            np.allclose(np.linalg.norm(cache, axis=1), 1, atol=1e-5, rtol=0), 'authenticated held1152 cache layout differs')
    arrays = {}
    for endpoint in context['spec']['endpoints']:
        key = label(endpoint)
        raw = np.load(run / (key + '.raw.npy'), allow_pickle=False)
        unit = np.load(run / (key + '.unit.npy'), allow_pickle=False)
        require(raw.shape == unit.shape == (12599, 128) and raw.dtype == unit.dtype == np.float32 and
                np.isfinite(raw).all() and np.isfinite(unit).all() and np.all(np.linalg.norm(raw, axis=1) > 0) and
                np.allclose(F.normalize(torch.from_numpy(raw), dim=1).numpy(), unit, atol=1e-6, rtol=0), 'complete raw/unit wire layout differs')
        packed = context['packing'].pack_int8_unit_embeddings(torch.from_numpy(unit))
        require(packed.to_bytes() == (run / (key + '.packed.bin')).read_bytes(), 'authenticated exact packed wire differs')
        arrays[endpoint['seed'], endpoint['arm']] = unit
    scoring = context['scoring']
    scoring.export = context['helper']
    fixed = scoring.scoring_math(context)  # Original decorated scorer and AST-pinned bootstrap, unchanged.
    labels = tuple(r['product'] for r in context['frozen']['held_manifest'])
    quality = {seed: {arm: fixed.packed_quality(arrays[seed, arm], labels, context['frozen']['query'],
                 context['frozen']['gallery'], device=torch.device('cpu')) for arm in ARMS} for seed in SEEDS}
    deltas, average = scoring.averaged_deltas(quality)
    intervals = {}
    for metric in scoring.METRICS:
        delta = np.asarray(average[metric])
        intervals[metric] = {'mean_delta': float(delta.mean())}
        for kind, groups in (('product', np.asarray(labels)[context['frozen']['query']]), ('query', np.arange(6354))):
            # The original helper resets RNG179019, sharing the exact5000 draws across metrics/signs.
            intervals[metric][kind + '_lower95'] = fixed.bootstrap_lower(delta, groups)
            intervals[metric][kind + '_upper95'] = -fixed.bootstrap_lower(-delta, groups)
    each_seed, quality_go = scoring.quality_gate(deltas, intervals)
    return {'decision': 'GO' if quality_go else 'KILL', 'quality': quality,
            'each_seed_quality_pass': each_seed, 'quality_pass': quality_go, 'cost_pass': True,
            'paired_seed_average_intervals': intervals, 'cost': context['costs'], 'cost_policy': COST_POLICY,
            'affine_reconstruction': {label(e): context['records'][e['seed'], e['arm']]['affine_reconstruction']
                                      for e in context['spec']['endpoints']},
            'nonlinear_mechanism_claim': False, 'bootstrap_draws': 5000, 'bootstrap_seed': 179019,
            'query_images': 6354, 'gallery_images': 6245, 'held_products': 1993, 'fit_images': 13283,
            'fit_products': 2004, 'updates_per_endpoint': 1000, 'schedule_seeds': list(SEEDS),
            'metric_units': 'fractions; multiply deltas by100 for percentage points',
            'interval_scope': 'equal-seed per-query deltas; paired product/query resampling conditional on frozen source',
            'independent_pretraining_seeds': False, 'intermediate_checkpoint_selection': False,
            'cost_denominator': 'fresh contemporaneous control TRAIN1000 for each seed; whole original service and median update',
            'optimization_throughput_is_image_training_throughput': False,
            'public_latency_measured': False, 'global_production_goal_met': False,
            'previous_official_and_pareto_preserved': True, 'quality_read': 'previously observed In-Shop TRAIN-held only',
            'export_receipt': {'path': str(run / 'receipt.json'), 'sha256': context['guards'][str(run / 'receipt.json')]}}


def cli_argv(authority, authority_sha, execution_sha, phase, output, prerequisite):
    value = [str(Path(__file__).absolute()), '--execution-sha256', execution_sha, '--authority', str(authority),
             '--authority-sha256', authority_sha, '--phase', phase, '--output', str(output)]
    if prerequisite is not None:
        value += ['--prerequisite', prerequisite['path'], '--prerequisite-sha256', prerequisite['sha256']]
    return value


def run(args):
    prior = None if args.prerequisite is None else {'path': str(args.prerequisite), 'sha256': args.prerequisite_sha256}
    require(sys.argv == cli_argv(args.authority, args.authority_sha256, args.execution_sha256, args.phase, args.output, prior), 'fixed canonical CLI order required')
    context = authority(args)
    cpu, export = prerequisites(context)
    before = native_start(context)
    import torch
    selected, source = context['selected'], context['selected']['initialized']['source']
    rng, flags = torch.random.get_rng_state().clone(), source.numerical_flags()
    args.output.mkdir()
    if args.phase == 'score':
        # Mandatory NEW five-member complete payloads even for the CUDA-hidden scorer.
        for endpoint in context['spec']['endpoints']:
            head, digest = load_head(context, endpoint)
            require(digest == cpu['head_facts'][label(endpoint)], 'updated CPU-qualified scoring head differs')
            del head
        gc.collect()
        facts = {k: cpu[k] for k in ('head_facts', 'source_facts', 'fit_witnesses', 'fit_pixels')}
        result = score_wires(context, export)
    else:
        model, processor, facts = qualify_heads(context)
        if cpu is not None:
            require(all(facts[k] == cpu[k] for k in facts), 'original updated CPU witnesses differ')
        result = export_wires(context, model, processor, facts) if args.phase == 'export' else {'files': {}}
        del model, processor
        gc.collect()
    require(torch.equal(rng, torch.random.get_rng_state()) and source.numerical_flags() == flags and
            (args.phase == 'export' or not torch.cuda.is_initialized()), 'whole-unit source/head CPU RNG/flags/CUDA differ')
    origins = context['training'].exit_rehash(selected['old'])  # Original full uncached dependencies/packages/loaded origins.
    require(closure(context['root'], args.execution_sha256, FILES, {}) == context['code'] and
            closure(Path(context['spec']['training']['root']), context['spec']['training']['execution_sha256'], TRAIN_PINS.keys(), {}) == TRAIN_PINS,
            'exit separate evaluator/cached trainer closures differ')
    resources = context['helper'].resources(context, args.phase, before)
    old_init = selected['initialized']
    preparation = {'source_cpu_service_seconds': old_init['record']['source_binding']['source_cpu']['service_seconds'],
                   'fit_export_service_seconds': old_init['record']['selected_export']['service_seconds'],
                   'pca_service_seconds': old_init['launch']['selected_initializer']['service_seconds'],
                   'initialized_cpu_service_seconds': selected['old']['launch']['selected_cpu']['service_seconds']}
    receipt = {**bind(context), 'schema': SCHEMA, 'phase': args.phase, 'pass': True,
        'authority': {'path': str(args.authority), 'sha256': args.authority_sha256}, 'output': str(args.output),
        'prerequisite': prior, 'optimizer_updates': 0, 'optimizer_members': 5, 'numerical_flags': flags,
        'strict_independent_head_reload_exact': True, 'source_reload_exact': True, 'fit_raw_unit_packed_exact': True,
        'first_heads_released_before_reload': True, 'source_head_rng_flags_preserved': True,
        'cuda_initialized': torch.cuda.is_initialized(), 'quality_read': False, 'official_read': False,
        'claim_eligible': False, 'public_serving_qualified': False, 'public_latency_measured': False,
        'prior_reusable_preparation': preparation,
        'new_method_prerequisite_services': {'cpu': selected['launch']['selected_cpu']['service_seconds'],
            'mechanics': {a: selected['launch']['selected_mechanics'][a]['service_seconds'] for a in ARMS}},
        'input_guards': context['guards'], 'origins': origins, 'exit_rehash_pass': True,
        'invocation': {'argv': sys.argv, 'python': str(Path(sys.executable).resolve()),
            'python_sha256': selected['old_cpu']['invocation']['python_sha256'], 'python_version': sys.version,
            'optimize': sys.flags.optimize, 'pid': os.getpid(), 'invocation_id': os.environ['INVOCATION_ID'],
            'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'], 'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        **facts, **result, **resources}
    context['helper'].publish(args.output / 'receipt.json', receipt)
    return receipt


def parser():
    result = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    result.add_argument('--execution-sha256', required=True)
    result.add_argument('--authority', type=Path, required=True)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--phase', choices=('cpu', 'export', 'score'), required=True)
    result.add_argument('--output', type=Path, required=True)
    result.add_argument('--prerequisite', type=Path)
    result.add_argument('--prerequisite-sha256')
    return result


def main():
    args = parser().parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Cached readout evaluation rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': args.phase, 'output': str(args.output), 'decision': result.get('decision')}))


if __name__ == '__main__':
    main()
