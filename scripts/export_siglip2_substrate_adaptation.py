#!/usr/bin/env python3
"""Frozen fresh Large/So400 TRAIN100 endpoints: FIT CPU proof and held export.

API: authority(args) -> dict admits all metadata/bytes before native imports;
validate_endpoint(record, endpoint, launch, cpu, mechanics, batches) -> None;
paired_cost(endpoints) -> dict; load_inference(context, endpoint, *, device)
-> (vision, FP32 head, processor, facts). No CUDA training restore is used.

Parent stages a NEW execution.json: exact original trainer5 + these three held
scripts + byte-identical reference_compare_inshop_sop_warmstart_100.py and
reference_score_inshop_crop_view_pair.py. The original trainer5 root is separate.
Schemas and exact CLI are documented in the native256 review boundary. All
terminal descriptors are original initializer TERMINALs, including full logs;
compact mechanics summaries cannot substitute for them. Hashes are actual pins.
Local help/admission tests use python3 -B -S; native imports occur only in run().
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import time
UNIT_STARTED = time.perf_counter()

import argparse
import ast
import gc
import hashlib
import importlib.util
import json
import math
import os
import re
import resource
import statistics
import sys
from pathlib import Path
from types import SimpleNamespace

SCHEMA = 'siglip2-substrate-held-v1'
AUTHORITY_SCHEMA = 'native256-held-authority-v1'
TRAIN_FILES = {'train_siglip2_substrate_adaptation.py', 'test_siglip2_substrate_adaptation.py',
               'deployed_code_rank.py', 'reference_train_sop_siglip2_compact.py', 'reference_unicom_training.py'}
ADDED = {'export_siglip2_substrate_adaptation.py', 'score_siglip2_substrate_adaptation.py',
         'test_siglip2_substrate_held.py'}
REFERENCES = {
    'reference_compare_inshop_sop_warmstart_100.py': {
        'source': '8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250',
        'ast': '717489188a008ceba1b930d9e7dff32b90cd5347302835538eb171fe23f4b33e',
        'name': 'packed_quality'},
    'reference_score_inshop_crop_view_pair.py': {
        'source': '16e27ccaa7325b9ef7efdb3512cb95791a682afdd5ae59bd1f0847d63b87f5ed',
        'ast': '76971d1089c8494e16450d56057a5b2fa2c71e40cf360d4ede0562fc53d9c720',
        'name': 'bootstrap_lower'}}
FILES = TRAIN_FILES | ADDED | REFERENCES.keys()
SEEDS = (179032, 179041)
WIDTHS = {'large': 1024, 'so400': 1152}
ORDER = ((179032, 'large'), (179032, 'so400'), (179041, 'so400'), (179041, 'large'))
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
AUGMENTATION = {'seed': 179032, 'formula': '179032*100000+step', 'steps': [1, 100]}
PRECISION = 'private_native_fp16'
COST_POLICY = {'whole_service_ratio_max': 1.50, 'median_update_ratio_max': 1.50,
               'training_wall_ratio': 'report_only'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    """Receipt comparisons only; never feed this to the TRAIN fingerprint."""
    return json.loads(json.dumps(value, allow_nan=False))


def policy(phase):
    require(phase in ('cpu', 'export', 'score'), 'fixed held phase required')
    value = {'seconds': 120 if phase == 'cpu' else 300, 'host_bytes': 8 * 1024**3, 'swap_bytes': 0}
    value.update({'cuda_allocated_bytes_exclusive': 10_000_000_000} if phase == 'export'
                 else {'cuda_visible_devices': ''})
    return value


def validate_resource_policies(policies):
    require(policies == {phase: policy(phase) for phase in ('cpu', 'export', 'score')}, 'held resource policies differ')


def sha(path):
    with Path(path).open('rb') as stream:
        digest, buffer = hashlib.sha256(), bytearray(1024**2)
        while count := stream.readinto(buffer):
            digest.update(memoryview(buffer)[:count])
            os.posix_fadvise(stream.fileno(), stream.tell() - count, count, os.POSIX_FADV_DONTNEED)
    return digest.hexdigest()


def descriptor(value, guards):
    require(value.keys() == {'path', 'sha256'}, 'exact file descriptor required')
    path = Path(value['path'])
    require(path.is_absolute() and path.resolve() == path and path.is_file() and
            re.fullmatch('[0-9a-f]{64}', value['sha256']) is not None and
            sha(path) == value['sha256'], 'canonical file/SHA differs: ' + str(path))
    require(guards.setdefault(str(path), value['sha256']) == value['sha256'], 'conflicting file authority')
    return path


def read_json(value, guards):
    path = descriptor(value, guards)
    with path.open('rb') as stream:
        raw = stream.read(64 * 1024**2 + 1)
    require(len(raw) <= 64 * 1024**2 and hashlib.sha256(raw).hexdigest() == value['sha256'], 'JSON size/SHA differs')
    def pairs(items):
        result = {}
        for key, item in items:
            require(key not in result, 'duplicate JSON key')
            result[key] = item
        return result
    def nonfinite(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite)


def load_bare(name, path, digest):
    descriptor({'path': str(path), 'sha256': digest}, {})
    require(name not in sys.modules, 'helper already loaded: ' + name)
    spec = importlib.util.spec_from_file_location(name, path)
    require(spec is not None and spec.loader is not None and Path(spec.origin) == path, 'helper origin differs')
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == digest, 'helper changed before execution')
    exec(compile(raw, str(path), 'exec'), vars(module))
    require(Path(module.__file__) == Path(module.__spec__.origin) == path and sha(path) == digest, 'helper origin/SHA differs')
    return module


def validate_closure(code, original):
    require(original.keys() == TRAIN_FILES and code.keys() == FILES and
            all(code[name] == digest for name, digest in original.items()) and
            all(code[name] == pin['source'] for name, pin in REFERENCES.items()), 'exact trainer5/held10 closure differs')


def validate_order(endpoints):
    require([(e['seed'], e['arm']) for e in endpoints] == list(ORDER), 'four distinct ordered endpoints required')


def validate_endpoint(record, endpoint, launch, cpu, mechanics, batches):
    """Pure metadata admission; approved exact draws are supplied in batches.

    batches has exactly batches (100x64), fact (original CPU tensor fact) and
    sha256 (TRAIN typed tensor fingerprint). Native admission regenerates both
    approved schedules with the genuine qualifier.schedule before any inference.
    """
    seed, arm = endpoint['seed'], endpoint['arm']
    require(seed in SEEDS and arm in WIDTHS, 'endpoint seed/arm differs')
    require(record['schema'] == 'siglip2-substrate-adaptation-v1' and record['phase'] == 'train' and
            record['arm'] == arm and record['seed'] == seed and record['width'] == WIDTHS[arm] and
            record['output_dim'] == 128 and record['completed_step'] == 100 and
            record['optimizer_members'] == 208, 'fresh TRAIN100 endpoint/width differs')
    require(all(record[k] is True for k in ('pass', 'training_qualified', 'fresh_source',
            'strict_independent_whole_head_buffers_raw_packed_reload_exact',
            'first_references_released_before_reload', 'constructor_rng_preserved', 'exit_rehash_pass',
            'both_locks_held_in_parent_authority', 'terminal_exit_and_both_locks_require_parent_receipt')) and
            all(record[k] is False for k in ('quality_read', 'quality_qualified', 'trained_state_reused',
                                           'training_state_discarded')), 'fresh TRAIN integrity/profile differs')
    require(launch['schema'] == 'native256-substrate-adaptation-launch-v1' and launch['phase'] == 'train' and
            launch['arm'] == arm and launch['seed'] == seed and launch['both_locks_held'] is True and
            record['authority_sha256'] == endpoint['launch']['sha256'] and
            record['execution_sha256'] == launch['execution_sha256'] and
            record['selected_cpu'] == launch['selected_cpu'] and
            record['selected_mechanics'] == launch['selected_mechanics'] and
            record['qualifier_authority'] == launch['qualifier_authority'] and
            record['checkpoint'] == endpoint['checkpoint'] and
            record['terminal_state_sha256'] == endpoint['terminal_state_sha256'], 'original launch/checkpoint lineage differs')
    require(cpu['schema'] == 'siglip2-substrate-initialized-cpu-v1' and cpu['arm'] == arm and
            cpu['phase'] == 'initialized-cpu' and cpu['width'] == WIDTHS[arm] and cpu['output_dim'] == 128 and
            cpu['source_binding']['arm'] == arm and
            cpu['authority_sha256'] == launch['qualifier_authority']['sha256'] and
            cpu['execution_sha256'] == launch['qualifier_execution_sha256'] and
            cpu['pass'] is True and cpu['fresh_source'] is True and cpu['reload_exact'] is True and
            cpu['quality_read'] is False and cpu['updates'] == cpu['head_updates'] == cpu['optimizer_state_entries'] == 0 and
            cpu['state']['counter'] == 0 and cpu['state']['seed'] == SEEDS[0] and cpu['state']['optimizer_state']['state'] == {} and
            all(cpu[k] is True for k in ('initializer_qualified', 'source_qualified', 'first_model_released_before_independent_clone',
                'source_cpu_runtime_and_first2_exact', 'constructor_rng_preserved', 'exit_rehash_pass', 'optimizer_created')) and
            all(cpu[k] is False for k in ('training_qualified', 'quality_qualified', 'cuda_initialized', 'gradients_created',
                                         'pca_rerun', 'teacher_state_reused', 'trained_state_reused')), 'own initialized CPU/source differs')
    require(mechanics['schema'] == record['schema'] and mechanics['phase'] == 'mechanics' and
            mechanics['arm'] == arm and mechanics['seed'] == SEEDS[0] and mechanics['completed_step'] == 17 and
            mechanics['selected_cpu'] == launch['selected_cpu'] and
            mechanics['qualifier_authority'] == launch['qualifier_authority'] and
            mechanics['execution_sha256'] == record['execution_sha256'] and mechanics['code'] == record['code'] and
            mechanics['pass'] is True and mechanics['quality_read'] is False and mechanics['training_state_discarded'] is True and
            mechanics['checkpoint'] is None and mechanics['native17_equals_serialized8_plus9_exact'] is True and
            mechanics['strict_independent_whole_head_buffers_raw_packed_reload_exact'] is True and
            mechanics['exit_rehash_pass'] is True, 'own discarded mechanics17 differs')
    require(record['augmentation'] == mechanics['augmentation'] == AUGMENTATION and
            record['reference_pins'] == mechanics['reference_pins'] and
            record['rank_helper_sha256'] == mechanics['rank_helper_sha256'], 'original augmentation/reference math differs')
    identity, facts = record['identity'], cpu['state']
    require(identity.keys() == {'arm', 'seed', 'execution_sha256', 'qualifier_authority', 'selected_cpu', 'config',
            'roles', 'parameter_names', 'optimizer_defaults', 'optimizer_groups', 'optimizer_serial_groups', 'runtime',
            'buffers_sha256', 'schedule_sha256', 'schedule_facts', 'augmentation'}, 'complete identity keys differ')
    runtime = facts['runtime']
    require(identity['arm'] == arm and identity['seed'] == seed and
            identity['execution_sha256'] == record['execution_sha256'] and
            identity['qualifier_authority'] == launch['qualifier_authority'] and
            identity['selected_cpu'] == launch['selected_cpu'] and identity['config'] == runtime['config'] and
            identity['roles'] == runtime['roles'] and identity['parameter_names'] == facts['parameter_names'] and
            len(set(identity['parameter_names'])) == 208 and
            identity['optimizer_defaults'] == facts['optimizer_defaults'] and
            identity['optimizer_groups'] == facts['optimizer_groups'] and
            identity['optimizer_serial_groups'] == facts['optimizer_state']['param_groups'] and
            identity['schedule_facts'] == facts['schedules'] and identity['augmentation'] == AUGMENTATION,
            'complete original CPU identity/config/optimizer differs')
    training_runtime = {'modules': [{**r, 'training': True, 'attributes': {**r['attributes'], 'training': True}}
                                   for r in runtime['modules']], 'processor': runtime['processor']}
    require(identity['runtime'] == training_runtime and identity['config']['hidden_size'] == WIDTHS[arm] and
            len(identity['roles']) == (400 if arm == 'large' else 448) and
            cpu['state']['arrays']['head.weight']['shape'] == [128, WIDTHS[arm]] and
            cpu['state']['arrays']['head.bias']['shape'] == [128], 'complete native source/head runtime differs')
    require(batches.keys() == {'batches', 'fact', 'sha256'} and
            batches['fact'] == facts['schedules'][str(seed)] and
            identity['schedule_sha256'] == batches['sha256'] and len(batches['batches']) == 100 and
            all(len(b) == 64 and all(type(n) is int and 0 <= n < 13283 for n in b) for b in batches['batches']),
            'approved complete seeded schedule differs')
    require(len(record['steps']) == 100, 'complete 100 logged steps required')
    require(record['steps'][-1]['state_sha256'] == endpoint['terminal_state_sha256'], 'last logged complete state differs from retained checkpoint')
    for step, (row, batch) in enumerate(zip(record['steps'], batches['batches'], strict=True), 1):
        require(row['step'] == step and row['batch'] == batch and row['schedule_sha256'] == batches['sha256'] and
                row['augmentation_seed'] == 179032 * 100000 + step and
                math.isfinite(row['seconds']) and row['seconds'] > 0, 'exact draws/augmentation/update differs')
        for key in ('rgb_sha256', 'pixels_sha256', 'state_sha256'):
            require(re.fullmatch('[0-9a-f]{64}', row[key]) is not None, 'step digest differs')
        require(all(math.isfinite(row[k]) for k in ('ce', 'rank', 'loss', 'scale', 'preclip_norm')) and
                row['scale'] >= 128 and math.isclose(row['loss'], row['ce'] + 8 * row['rank'], rel_tol=0, abs_tol=1e-12),
                'original objective diagnostics differ')
    diag = lambda row: {k: v for k, v in row.items() if k != 'seconds'}
    require(len(mechanics['steps']) == 17 and len(mechanics['resumed_steps']) == 9 and
            [r['step'] for r in mechanics['steps']] == list(range(1, 18)) and
            list(map(diag, mechanics['steps'][8:])) == list(map(diag, mechanics['resumed_steps'])), 'complete mechanics replay differs')
    if seed == SEEDS[0]:
        require(record['first17_mechanics_replay_exact'] is True and
                identity == mechanics['identity'] and record['initial_state_sha256'] == mechanics['initial_state_sha256'] and
                list(map(diag, record['steps'][:17])) == list(map(diag, mechanics['steps'])), 'fresh first17 mechanics replay differs')
    else:
        require(record['first17_mechanics_replay_exact'] is False, '041 replay claim differs')
    require(record['median_update_seconds'] == statistics.median(r['seconds'] for r in record['steps'][2:]) and
            all(math.isfinite(record[k]) and record[k] > 0 for k in ('training_wall_seconds', 'median_update_seconds')),
            'original update costs differ')


def paired_cost(endpoints):
    require(set(endpoints) == set(ORDER), 'four paired endpoints required')
    result = {}
    for seed in SEEDS:
        large, so400 = (endpoints[seed, arm] for arm in WIDTHS)
        require(large['identity']['schedule_facts'] == so400['identity']['schedule_facts'] and
                all(a[k] == b[k] for a, b in zip(large['steps'], so400['steps'], strict=True)
                    for k in ('step', 'batch', 'schedule_sha256', 'augmentation_seed', 'rgb_sha256', 'pixels_sha256')),
                'paired seeded RGB/pixel inputs differ')
        ratios = {name: so400[key] / large[key] for name, key in
                  (('whole_service_ratio', 'service_seconds'), ('median_update_ratio', 'median_update_seconds'),
                   ('training_wall_ratio', 'training_wall_seconds'))}
        require(all(math.isfinite(v) and v > 0 for v in ratios.values()) and
                ratios['whole_service_ratio'] <= 1.50 and ratios['median_update_ratio'] <= 1.50,
                'fresh whole-service/median cost gate failed')
        result[str(seed)] = {**ratios, 'training_wall_ratio_gate': False,
                            'large_service_seconds': large['service_seconds'], 'so400_service_seconds': so400['service_seconds']}
    return result


def logged_steps(path, steps):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith('{'):
            row = json.loads(line)
            if 'batch' in row and 'step' in row:
                rows.append(row)
    require(rows == steps, 'original full logged steps differ')


def zero_events(value):
    events = dict(line.split() for line in value['values']['memory.events'].splitlines())
    require(events and all(int(count) == 0 for count in events.values()), 'whole-unit memory events must be zero')


def validate_split(frozen, fit):
    rows, query, gallery = frozen['held_manifest'], frozen['query'], frozen['gallery']
    require(frozen['fit_manifest'] == fit['rows'] and len(rows) == 12599 and len(query) == 6354 and len(gallery) == 6245 and
            len(set(query)) == 6354 and len(set(gallery)) == 6245 and
            set(query).isdisjoint(gallery) and set(query) | set(gallery) == set(range(12599)) and
            all(type(i) is int for i in (*query, *gallery)) and len({r['product'] for r in rows}) == 1993 and
            {r['product'] for r in rows}.isdisjoint(fit['class_names']), 'original TRAIN-held split differs')
    require({rows[i]['product'] for i in query} == {rows[i]['product'] for i in gallery}, 'held positive inventory differs')
    for row in rows:
        path = Path(row['relative_path'])
        require(row.keys() == {'relative_path', 'image_sha256', 'product'} and not path.is_absolute() and
                '..' not in path.parts and str(path).startswith('Img/img/') and
                re.fullmatch('[0-9a-f]{64}', row['image_sha256']) is not None, 'held manifest row differs')


def authority(args):
    """One original trainer bootstrap; all four prior CPU/mechanics/logs admitted.

    Other arms' accepted prerequisite records are authenticated via their complete
    inventoried byte guards and original terminal footers. Only the selected
    arm's genuine factory context is loaded; no repeated fixed-name bootstrap.
    Native complete-payload admission of ALL endpoints follows, before FIT/held
    inference. Source-only fixtures do not establish native admission.
    """
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native packages preceded admission')
    root, guards = Path(__file__).absolute().parent, {}
    spec = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    validate_resource_policies(spec['resource_policies'])
    require(spec.keys() == {'schema', 'execution_sha256', 'training_root', 'training_execution_sha256',
            'prerequisites', 'endpoints', 'schedules', 'frozen_split', 'resource_policies', 'cost_policy', 'both_locks_held'} and
            spec['schema'] == AUTHORITY_SCHEMA and spec['execution_sha256'] == args.execution_sha256 and
            spec['cost_policy'] == COST_POLICY and spec['both_locks_held'] is True and
            spec['prerequisites'].keys() == WIDTHS.keys() and spec['schedules'].keys() == {'179032', '179041'},
            'held authority/profile differs')
    train_root = Path(spec['training_root'])
    require(train_root.is_absolute() and train_root.resolve() == train_root and
            not root.is_relative_to(train_root) and not train_root.is_relative_to(root), 'separate original trainer root required')
    original = read_json({'path': str(train_root / 'execution.json'), 'sha256': spec['training_execution_sha256']}, guards)
    code = read_json({'path': str(root / 'execution.json'), 'sha256': args.execution_sha256}, guards)
    validate_closure(code, original)
    for name, digest in code.items():
        descriptor({'path': str(root / name), 'sha256': digest}, guards)
    for name, digest in original.items():
        descriptor({'path': str(train_root / name), 'sha256': digest}, guards)
    training = load_bare('_native256_held_original_trainer', train_root / 'train_siglip2_substrate_adaptation.py',
                         original['train_siglip2_substrate_adaptation.py'])
    require(training.FILES == TRAIN_FILES and training.WIDTHS == WIDTHS and training.SEEDS == SEEDS, 'original trainer profile differs')
    validate_order(spec['endpoints'])
    require(args.seed in SEEDS and args.arm in WIDTHS and args.output.is_absolute() and
            args.output.parent.resolve() == args.output.parent and not args.output.exists() and
            not args.output.is_symlink() and not any(args.output.is_relative_to(p) for p in (root, train_root)),
            'exclusive separate canonical output required')
    selected = next(e for e in spec['endpoints'] if (e['seed'], e['arm']) == (args.seed, args.arm))
    selected_launch = read_json(selected['launch'], guards)
    train_args = SimpleNamespace(execution_sha256=spec['training_execution_sha256'],
        authority=Path(selected['launch']['path']), authority_sha256=selected['launch']['sha256'],
        phase='train', arm=args.arm, seed=args.seed, output=args.output)
    selected_context = training.authority(train_args)
    require(selected_context['launch'] == selected_launch, 'selected original launch changed')
    admission = training.FlatAdmission()
    admission.init = selected_context['initialized']['init']
    admission.exporter = selected_context['initialized']['pca']['exporter']
    source_base = selected_context['initialized']['source_context']
    source_launch = admission.descriptor_json(source_base['launch']['source_cpu_authority'], guards)
    records, cpus, mechanics, preparation = {}, {}, {}, {}
    cpu_finals, mechanics_finals = {}, {}
    units = []
    for arm, prior in spec['prerequisites'].items():
        require(prior.keys() == {'qualifier_root', 'qualifier_execution_sha256', 'qualifier_authority', 'cpu', 'mechanics'},
                'exact per-source prerequisite descriptor required')
        qcode = admission.closure(Path(prior['qualifier_root']), prior['qualifier_execution_sha256'], training.QUALIFIER_FILES, guards)
        qlaunch = admission.descriptor_json(prior['qualifier_authority'], guards)
        cpu = admission.descriptor_json(prior['cpu']['receipt'], guards)
        mech = admission.descriptor_json(prior['mechanics']['receipt'], guards)
        initializer = admission.descriptor_json(qlaunch['selected_initializer']['receipt'], guards)
        source_args = SimpleNamespace(**{**vars(source_base['args']), 'arm': arm, 'output': args.output})
        source_context = admission.source_authority(source_base['source_driver'], source_base['extract'],
            source_base['root'], source_base['code'], source_args)
        source_context['source_driver'] = source_base['source_driver']
        source_terminal = source_base['launch']['source_cpu'][arm]
        source_proof = admission.source_cpu(source_context, source_terminal, source_launch)
        for path, digest in source_context['guards'].items():
            require(guards.setdefault(path, digest) == digest, 'conflicting complete source authority')
        expected_binding = {**selected_context['initialized']['record']['source_binding'], 'arm': arm, 'source_cpu': source_terminal}
        require(cpu['code'] == qcode and cpu['authority_sha256'] == prior['qualifier_authority']['sha256'] and
                cpu['initializers'] == initializer['artifact'] and cpu['source_binding'] == initializer['source_binding'] and
                cpu['source_binding'] == expected_binding and cpu['state']['runtime'] == source_proof['runtime'] and
                initializer['arm'] == arm and initializer['width'] == WIDTHS[arm] and
                cpu['state']['arrays'] == initializer['arrays'] and cpu['resource_policy'] == selected_context['qualifier'].POLICY,
                'original CPU initializer lineage differs')
        preparation[arm] = {'initialized_cpu_service_seconds': prior['cpu']['service_seconds'],
            'mechanics_service_seconds': prior['mechanics']['service_seconds'],
            'pca_service_seconds': qlaunch['selected_initializer']['service_seconds'],
            'fit_export_service_seconds': initializer['selected_export']['service_seconds'],
            'source_cpu_service_seconds': cpu['source_binding']['source_cpu']['service_seconds']}
        for value, terminal, cap in ((cpu, prior['cpu'], 120), (mech, prior['mechanics'], training.policy('mechanics')['seconds'])):
            final = admission.admit_terminal(value, terminal, cap, guards)
            zero_events(final)
            (cpu_finals if value is cpu else mechanics_finals)[arm] = final
            require(value['exit_rehash_pass'] is True and value['quality_read'] is False, 'original prerequisite exit/profile differs')
            for path, digest in value['input_guards'].items():
                admission.bound_file(guards, path, digest)
            units.append(terminal)
        require(mech['cpu_final_cgroup'] == cpu_finals[arm] and mech['resource_policy'] == training.policy('mechanics') and
                cpu['input_guards'].get(cpu['checkpoint']['path']) == cpu['checkpoint']['sha256'], 'original prerequisite complete footer/checkpoint differs')
        mech_argv = mech['invocation']['argv']
        mech_launch = admission.read_json(mech_argv[4], mech['authority_sha256'], guards)
        require(mech_launch == {**selected_launch, 'arm': arm, 'seed': SEEDS[0], 'phase': 'mechanics',
                'qualifier_root': prior['qualifier_root'], 'qualifier_execution_sha256': prior['qualifier_execution_sha256'],
                'qualifier_authority': prior['qualifier_authority'], 'selected_cpu': prior['cpu'],
                'selected_mechanics': None, 'resource_policy': training.policy('mechanics')}, 'original mechanics launch differs')
        logged_steps(Path(prior['mechanics']['log']['path']), mech['steps'] + mech['resumed_steps'])
        cpus[arm], mechanics[arm] = cpu, mech
    for endpoint in spec['endpoints']:
        require(endpoint.keys() == {'seed', 'arm', 'launch', 'terminal', 'checkpoint', 'terminal_state_sha256'}, 'exact endpoint descriptor required')
        seed, arm = endpoint['seed'], endpoint['arm']
        launch = admission.descriptor_json(endpoint['launch'], guards)
        prior = spec['prerequisites'][arm]
        require(launch == {**selected_launch, 'arm': arm, 'seed': seed,
                'qualifier_root': prior['qualifier_root'], 'qualifier_execution_sha256': prior['qualifier_execution_sha256'],
                'qualifier_authority': prior['qualifier_authority'], 'selected_cpu': prior['cpu'],
                'selected_mechanics': prior['mechanics']}, 'own original TRAIN launch differs')
        value = admission.descriptor_json(endpoint['terminal']['receipt'], guards)
        require(value['code'] == original and value['reference_pins'] == canonical(training.REFERENCES) and
                value['rank_helper_sha256'] == training.RANK_SHA256 and value['resource_policy'] == training.policy('train'),
                'original frozen TRAIN closure/policy differs')
        validate_endpoint(value, endpoint, launch, cpus[arm], mechanics[arm], spec['schedules'][str(seed)])
        require(value['cpu_final_cgroup'] == cpu_finals[arm] and value['mechanics_final_cgroup'] == mechanics_finals[arm],
                'original TRAIN prerequisite final cgroup bindings differ')
        require(Path(endpoint['terminal']['receipt']['path']).name == 'receipt.json' and
                Path(endpoint['checkpoint']['path']) == Path(endpoint['terminal']['receipt']['path']).parent / 'resume.pt', 'original endpoint path roles differ')
        prior_argv = [str(train_root / 'train_siglip2_substrate_adaptation.py'), '--execution-sha256', spec['training_execution_sha256'],
            '--authority', endpoint['launch']['path'], '--authority-sha256', endpoint['launch']['sha256'], '--phase', 'train',
            '--arm', arm, '--seed', str(seed), '--output', str(Path(endpoint['terminal']['receipt']['path']).parent)]
        require(value['invocation']['argv'] == prior_argv and
                all(value['invocation'][k] == cpus[arm]['invocation'][k] for k in ('python', 'python_sha256', 'python_version')) and
                value['peak_cuda_allocated_bytes'] < 10_000_000_000, 'original TRAIN interpreter/argv/complete CUDA peak differs')
        zero_events(admission.admit_terminal(value, endpoint['terminal'], training.policy('train')['seconds'], guards))
        logged_steps(Path(endpoint['terminal']['log']['path']), value['steps'])
        for path, digest in value['input_guards'].items():
            admission.bound_file(guards, path, digest)
        admission.bound_file(guards, endpoint['checkpoint']['path'], endpoint['checkpoint']['sha256'])
        records[seed, arm] = {**value, 'service_seconds': endpoint['terminal']['service_seconds']}
        units.append(endpoint['terminal'])
    require(len({u['invocation_id'] for u in units}) == len(units) and len({u['unit'] for u in units}) == len(units) and
            len({e['checkpoint']['path'] for e in spec['endpoints']}) == 4, 'distinct original endpoints/units required')
    for key in ('ordered_input_sha256', 'ordered_rgb_sha256'):
        require(cpus['large'][key] == cpus['so400'][key], 'fresh paired FIT authority differs')
    for path, digest in selected_context['guards'].items():
        require(guards.setdefault(path, digest) == digest, 'conflicting selected byte guard')
    fit = selected_context['initialized']['source_context']['fit']
    require(spec['frozen_split'] == fit['original_receipt'], 'original frozen split authority differs')
    frozen = admission.descriptor_json(spec['frozen_split'], guards)
    validate_split(frozen, fit)
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import during authority')
    return {'args': args, 'spec': spec, 'root': root, 'code': code, 'guards': guards, 'training': training,
            'selected': selected, 'selected_context': selected_context, 'records': records, 'cpus': cpus,
            'mechanics': mechanics, 'frozen': frozen, 'fit': fit, 'costs': paired_cost(records), 'preparation': preparation}


def admit_checkpoint(context, endpoint, *, load=False):
    """Original complete typed fingerprint, exact source inventory and both draws.

    Returns owned inference modules only when load=True; mapped tensors never
    escape this function. Each hash/copy releases consumed complete mmap pages.
    """
    import torch
    from transformers import AutoImageProcessor
    training, arm, seed = context['training'], endpoint['arm'], endpoint['seed']
    record, cpu = context['records'][seed, arm], context['cpus'][arm]
    path = descriptor(endpoint['checkpoint'], context['guards'])
    saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    with path.open('rb') as stream:
        pages = training.CheckpointPages(stream)
        identity = saved['identity']  # Preserve tuples and integer config keys.
        require(canonical(identity) == record['identity'], 'serialized typed identity/receipt differs')
        training.check_payload(saved, identity, 100, identity['optimizer_defaults'], identity['optimizer_serial_groups'],
                               identity['parameter_names'], cpu['numerical_flags'])
        require(saved['config'] == identity['config'] and canonical(saved['config']) == cpu['state']['runtime']['config'],
                'complete serialized source config differs')
        roles = identity['roles']
        require(len(roles) == (400 if arm == 'large' else 448) and
                saved['vision'].keys() == {r['name'] for r in roles} and saved['head'].keys() == {'weight', 'bias'} and
                saved['buffers'].keys() == cpu['state']['runtime']['buffers'].keys() == {'embeddings.position_ids'},
                'complete vision/head/nonpersistent buffer inventory differs')
        shapes = {r['name']: tuple(r['shape']) for r in roles}
        shapes.update({'compact_head.weight': (128, WIDTHS[arm]), 'compact_head.bias': (128,), 'classifier': (2004, 128)})
        expected_arrays = {'classifier': (2004, 128), 'bank': (13283, 128), 'target': (13283,)}
        for name, shape in expected_arrays.items():
            require(tuple(saved[name].shape) == shape and saved[name].dtype == (torch.int64 if name == 'target' else torch.float32),
                    'complete classifier/bank/target differs')
        require(saved['pca'].keys() == {'mean', 'components'} and saved['schedules'].keys() == {'179032', '179041'} and
                saved['head']['weight'].shape == (128, WIDTHS[arm]) and saved['head']['bias'].shape == (128,), 'PCA/schedules/head width differs')
        source = context['selected_context']['source']
        for name, value in saved['vision'].items():
            require(tuple(value.shape) == shapes[name] and value.dtype == torch.float32 and torch.isfinite(value).all().item(), 'vision shape/dtype/finite differs')
            if next(r['role'] for r in roles if r['name'] == name) == 'frozen':
                require(source.tensor_fact(value) == cpu['state']['runtime']['vision'][name], 'fresh source frozen prefix differs')
            pages.consume(value)
        for name, value in saved['buffers'].items():
            fact = cpu['state']['runtime']['buffers'][name]
            require({k: v for k, v in fact.items() if k != 'persistent'} == source.tensor_fact(value) and
                    fact['persistent'] is False and torch.equal(value, torch.arange(256).expand(1, -1)), 'original nonpersistent buffer differs')
            pages.consume(value)
        for name, value in {**saved['head'], 'classifier': saved['classifier'], 'bank': saved['bank']}.items():
            require(value.dtype == torch.float32 and torch.isfinite(value).all().item(), 'finite FP32 head/proxy/bank required')
            pages.consume(value)
        for name, value in saved['pca'].items():
            require(source.tensor_fact(value) == cpu['state']['arrays'][name], 'original complete PCA differs')
            pages.consume(value)
        require(source.tensor_fact(saved['target']) == cpu['state']['arrays']['target'] and
                saved['target'].tolist() == context['fit']['targets'], 'original FIT targets differ')
        pages.consume(saved['target'])
        q = context['selected_context']['qualifier']
        for scheduled_seed in SEEDS:
            key = str(scheduled_seed)
            expected = torch.from_numpy(q.schedule(context['fit']['targets'], scheduled_seed))
            approved = context['spec']['schedules'][key]
            require(saved['schedules'][key].dtype == torch.int64 and torch.equal(saved['schedules'][key], expected) and
                    expected.tolist() == approved['batches'] and source.tensor_fact(expected) == approved['fact'] and
                    training.fingerprint(expected) == approved['sha256'], 'exact original seeded draws differ')
            pages.consume(saved['schedules'][key])
        import numpy as np
        positives = context['reference_math'].member_bank_positive_ordinals(
            np.asarray(context['fit']['targets'], dtype=np.int64), allow_singletons=True)
        require(saved['positive'].dtype == torch.int64 and torch.equal(saved['positive'], positives),
                'original singleton positive table differs')
        pages.consume(saved['positive'])
        for index, name in enumerate(identity['parameter_names']):
            moments = saved['optimizer']['state'][index]
            require(all(v.dtype == torch.float32 and tuple(v.shape) == shapes[name] and torch.isfinite(v).all().item()
                        for key, v in moments.items() if key != 'step'), 'complete208 optimizer moments differ')
            for value in moments.values():
                pages.consume(value)
        require(training.fingerprint(saved, consumed=pages.consume) == endpoint['terminal_state_sha256'], 'complete TRAIN typed fingerprint differs')
        facts = {name + '_sha256': training.fingerprint(saved[key], consumed=pages.consume)
                 for name, key in (('vision', 'vision'), ('head', 'head'), ('buffers', 'buffers'))}
        require(facts['buffers_sha256'] == identity['buffers_sha256'], 'complete identity buffer fingerprint differs')
        if load:
            selected = context['selected_context']
            require(arm == context['args'].arm and seed == context['args'].seed, 'selected source factory required')
            model = selected['source'].construct(saved['config'], selected['initialized']['source_context'])
            training.load_vision(model, saved['vision'], pages)
            inventory = selected['source'].configure_roles(model, selected['initialized']['source_context']['expected'], model.config.num_hidden_layers)
            require(canonical(inventory) == record['identity']['roles'] and model.config.to_dict() == saved['config'], 'strict source roles/config differs')
            buffers = dict(model.named_buffers())
            require(buffers.keys() == saved['buffers'].keys() and 'position_ids' in model.embeddings._non_persistent_buffers_set,
                    'independent complete nonpersistent buffers differ')
            with torch.no_grad():
                for name, value in buffers.items():
                    require(value.shape == saved['buffers'][name].shape and value.dtype == saved['buffers'][name].dtype, 'strict buffer shape/dtype differs')
                    value.copy_(saved['buffers'][name]); pages.consume(saved['buffers'][name])
            head = q.head_from({'head.weight': saved['head']['weight'], 'head.bias': saved['head']['bias']}, WIDTHS[arm])
            for value in saved['head'].values():
                pages.consume(value)
            processor = AutoImageProcessor.from_pretrained(selected['initialized']['source_context']['entry']['input']['preprocessor']['path'],
                                                          local_files_only=True, backend='torchvision')
            model.eval(); head.eval()
            runtime = training.runtime(selected, {'model': model, 'processor': processor})
            require(runtime == {'modules': cpu['state']['runtime']['modules'], 'processor': cpu['state']['runtime']['processor']},
                    'independent actual runtime/processor differs')
            require(training.fingerprint(dict(model.state_dict())) == facts['vision_sha256'] and
                    training.fingerprint(dict(head.state_dict())) == facts['head_sha256'] and
                    training.fingerprint(buffers) == facts['buffers_sha256'], 'strict complete vision/head/buffer reload differs')
            model.requires_grad_(False); head.requires_grad_(False)
    del saved, identity, value, moments
    gc.collect()
    return (model, head, processor, facts) if load else facts


def load_inference(context, endpoint, *, device):
    require(device in ('cpu', 'cuda'), 'fixed inference device required')
    model, head, processor, facts = admit_checkpoint(context, endpoint, load=True)
    if device == 'cuda':
        model.half().to(device); head.to(device)
    return model, head, processor, facts


def inference_facts(context, model, head):
    t = context['training']
    return {'vision_sha256': t.fingerprint(dict(model.state_dict())), 'head_sha256': t.fingerprint(dict(head.state_dict())),
            'buffers_sha256': t.fingerprint(dict(model.named_buffers()))}


def decode(context, processor, rows):
    from PIL import Image
    images = []
    root = Path(context['fit']['dataset_root'])
    try:
        for row in rows:
            path = (root / row['relative_path']).resolve()
            require(path.is_relative_to(root), 'image escaped dataset root')
            descriptor({'path': str(path), 'sha256': row['image_sha256']}, context['guards'])
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        pixels = processor(images=images, return_tensors='pt')['pixel_values']
        require(str(pixels.dtype) == 'torch.float32' and tuple(pixels.shape) == (len(rows), 3, 256, 256), 'stock native256 pixels differ')
        return pixels
    finally:
        for image in images:
            image.close()


def features(context, model, head, pixels):
    import torch
    from torch.nn import functional as F
    with torch.inference_mode(), torch.autocast(device_type=pixels.device.type, enabled=False):
        raw = context['selected_context']['qualifier'].raw_features(model, head, pixels)
        values = F.normalize(raw, dim=1)
    packed = context['packing'].pack_int8_unit_embeddings(values.cpu())
    return raw.cpu(), values.cpu(), packed.codes.cpu(), packed.inverse_norms.cpu()


def exact(a, b):
    import torch
    require(len(a) == len(b) and all(torch.equal(x.view(torch.int16) if x.dtype == torch.float16 else x,
                                             y.view(torch.int16) if y.dtype == torch.float16 else y)
                                   for x, y in zip(a, b, strict=True)), 'independent raw/unit/packed inverse bits differ')


def bind_endpoint(context, endpoint):
    return {'authority_sha256': context['args'].authority_sha256, 'execution_sha256': context['args'].execution_sha256,
            'seed': endpoint['seed'], 'arm': endpoint['arm'], 'width': WIDTHS[endpoint['arm']], 'output_dim': 128,
            'training_receipt': endpoint['terminal']['receipt'], 'checkpoint': endpoint['checkpoint'],
            'terminal_state_sha256': endpoint['terminal_state_sha256']}


def validate_cpu(proof, context, endpoint):
    require(all(proof[k] == v for k, v in bind_endpoint(context, endpoint).items()) and
            proof['schema'] == SCHEMA and proof['phase'] == 'cpu' and proof['source_code'] == context['code'] and
            proof['resource_policy'] == policy('cpu') and proof['optimizer_updates'] == 0 and
            all(proof[k] is True for k in ('pass', 'strict_complete_inference_reload_exact', 'fit_raw_packed_reload_exact',
                                          'first_references_released_before_reload', 'cpu_rng_preserved', 'exit_rehash_pass')) and
            proof['quality_read'] is False and proof['official_read'] is False and
            proof['invocation']['cuda_visible_devices'] == '' and proof['peak_cuda_allocated_bytes'] == 0,
            'original FIT-only inference CPU proof differs')


def rehash(context):
    """Full uncached exit pass, including all endpoints and consumed held bytes."""
    for path, digest in context['guards'].items():
        descriptor({'path': path, 'sha256': digest}, {})


def resources(context, phase, before):
    selected = context['selected_context']
    after = selected['source'].cgroup_memory()
    unit = Path(after['path']).name.removesuffix('.service')
    selected['initialized']['init'].admit_cgroup(after, unit)
    zero_events(after)
    elapsed, peak = time.perf_counter() - context.get('unit_started', UNIT_STARTED), 0
    if phase == 'export':
        import torch
        peak = torch.cuda.max_memory_allocated()
    require(elapsed < policy(phase)['seconds'] and peak < 10_000_000_000 and
            0 < resource.getrusage(resource.RUSAGE_SELF).ru_maxrss <= 8 * 1024**2 and
            after['path'] == before['path'], 'complete-unit resources differ')
    swap = next(v for v in Path('/proc/self/status').read_text().splitlines() if v.startswith('VmSwap:'))
    require(int(swap.split()[1]) == 0, 'native swap differs')
    return {'resource_policy': policy(phase), 'wall_seconds': elapsed,
            'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'peak_cuda_allocated_bytes': peak, 'cgroup_before': before, 'cgroup_after': after,
            'both_locks_held_in_parent_authority': True, 'terminal_exit_and_both_locks_require_parent_receipt': True}


def publish(path, value):
    with path.open('xb') as stream:
        stream.write((json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode())
        stream.flush(); os.fsync(stream.fileno())


def native_start(context, phase):
    """Called only after authority; flags/origins precede independent construction."""
    prior = context['cpus'][context['args'].arm]
    selected = context['selected_context']
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '' if phase != 'export'
            else os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, ''), 'explicit CPU-hidden/export CUDA required')
    require(re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')) is not None and sys.flags.optimize == 0,
            'unoptimized original systemd invocation required')
    python = Path(sys.executable).resolve()
    require(str(python) == prior['invocation']['python'] and sha(python) == prior['invocation']['python_sha256'] and
            sys.version == prior['invocation']['python_version'], 'qualified interpreter differs')
    before = selected['source'].cgroup_memory()
    selected['initialized']['init'].admit_cgroup(before, Path(before['path']).name.removesuffix('.service'))
    import torch
    require(not torch.cuda.is_initialized(), 'CPU admission must precede CUDA')
    flags = prior['numerical_flags']
    torch.set_num_threads(flags['threads'])
    if torch.get_num_interop_threads() != flags['interop_threads']:
        torch.set_num_interop_threads(flags['interop_threads'])
    require(selected['source'].numerical_flags() == flags, 'qualified numerical flags differ')
    if phase == 'export':
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8' and torch.cuda.is_available() and torch.cuda.device_count() == 1,
                'one visible CUDA/CUBLAS required')
    else:
        require(not torch.cuda.is_available(), 'CPU proof/scorer must hide CUDA')
    qroot = Path(prior['invocation']['argv'][0]).parent
    qcode = prior['code']
    packing = load_bare('_native256_held_packing', qroot / 'joint_relational_compaction.py', qcode['joint_relational_compaction.py'])
    context['packing'] = packing
    context['reference_math'] = context['training'].reference_math(selected)
    return before


def run(args):
    context = authority(args)
    before = native_start(context, args.phase)
    import numpy as np
    import torch
    endpoint, training = context['selected'], context['training']
    for e in context['spec']['endpoints']:
        admit_checkpoint(context, e)
    binding = bind_endpoint(context, endpoint)
    cpu_terminal = None
    if args.phase == 'export':
        require(args.cpu_terminal is not None and args.cpu_terminal_sha256 is not None, 'export needs original CPU terminal descriptor')
        cpu_terminal = read_json({'path': str(args.cpu_terminal), 'sha256': args.cpu_terminal_sha256}, context['guards'])
        cpu_proof = read_json(cpu_terminal['receipt'], context['guards'])
        validate_cpu(cpu_proof, context, endpoint)
        zero_events(context['selected_context']['initialized']['init'].admit_terminal(cpu_proof, cpu_terminal, 120, context['guards']))
    else:
        require(args.cpu_terminal is None and args.cpu_terminal_sha256 is None, 'CPU phase cannot supply prior proof')
    rng = torch.random.get_rng_state().clone()
    model, head, processor, fp32_facts = load_inference(context, endpoint, device='cpu')
    pixels = decode(context, processor, context['fit']['rows'][:2])
    witness = features(context, model, head, pixels)
    require(inference_facts(context, model, head) == fp32_facts, 'first CPU forward changed whole/head/buffers')
    witness_facts = {name: context['selected_context']['source'].tensor_fact(v) for name, v in
                     zip(('raw', 'unit', 'codes', 'inverse_norms'), witness, strict=True)}
    pixel_fact = context['selected_context']['source'].tensor_fact(pixels)
    model.half()
    f16_facts = inference_facts(context, model, head)
    del model, head, processor
    gc.collect()
    model, head, processor, reloaded_facts = load_inference(context, endpoint, device='cpu')
    require(reloaded_facts == fp32_facts, 'sequential strict complete model reload differs')
    second_pixels = decode(context, processor, context['fit']['rows'][:2])
    require(torch.equal(pixels, second_pixels), 'independent FIT pixels differ')
    exact(witness, features(context, model, head, second_pixels))
    require(inference_facts(context, model, head) == fp32_facts, 'independent CPU forward changed whole/head/buffers')
    del second_pixels
    model.half()
    require(inference_facts(context, model, head) == f16_facts and torch.equal(rng, torch.random.get_rng_state()),
            'independent nativeFP16/RNG differs')
    common = {**binding, 'schema': SCHEMA, 'source_code': context['code'], 'pass': True,
              'fp32_facts': fp32_facts, 'native_fp16_facts': f16_facts, 'fit_pixels': pixel_fact, 'fit_witness': witness_facts,
              'strict_complete_inference_reload_exact': True, 'fit_raw_packed_reload_exact': True,
              'first_references_released_before_reload': True, 'cpu_rng_preserved': True,
              'optimizer_updates': 0, 'quality_read': False, 'official_read': False,
              'claim_eligible': False, 'public_serving_qualified': False, 'public_latency_measured': False}
    if args.phase == 'export':
        require(all(cpu_proof[k] == common[k] for k in common), 'authenticated CPU inference witnesses differ')
        model.cuda(); head.cuda()
        clone, clone_head, _, clone_facts = load_inference(context, endpoint, device='cuda')
        require(clone_facts == fp32_facts and inference_facts(context, clone, clone_head) == f16_facts, 'independent CUDA clone differs')
        exact(features(context, model, head, pixels.half().cuda()), features(context, clone, clone_head, pixels.half().cuda()))
        require(all(p.dtype == torch.float16 for m in (model, clone) for p in m.parameters()) and
                all(p.dtype == torch.float32 for h in (head, clone_head) for p in h.parameters()), 'nativeFP16 vision/FP32 head required')
        device_rng = training.fingerprint({'cpu': torch.random.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()})
        values = torch.empty((12599, 128), dtype=torch.float32)
        frozen = context['frozen']
        batch_sizes = {}
        # First held decode follows every endpoint/payload, CPU proof and FIT parity.
        for role in ('query', 'gallery'):
            batch_sizes[role] = []
            for offset in range(0, len(frozen[role]), 32):
                indices = frozen[role][offset:offset + 32]
                batch_sizes[role].append(len(indices))
                batch_pixels = decode(context, processor, [frozen['held_manifest'][i] for i in indices]).half().cuda()
                a = features(context, model, head, batch_pixels)
                b = features(context, clone, clone_head, batch_pixels)
                exact(a, b)
                values[indices] = a[1]
                del a, b, batch_pixels
        torch.cuda.synchronize()
        require(training.fingerprint({'cpu': torch.random.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}) == device_rng and
                inference_facts(context, model, head) == inference_facts(context, clone, clone_head) == f16_facts,
                'complete held inference state/RNG changed')
        packed = context['packing'].pack_int8_unit_embeddings(values)
        args.output.mkdir()
        files = {}
        for name, value in (('held.npy', values), ('held.codes.npy', packed.codes), ('held.inverse.npy', packed.inverse_norms)):
            path = args.output / name
            with path.open('xb') as stream:
                np.save(stream, value.numpy(), allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
            files[name] = sha(path)
        common.update(cpu_terminal={'path': str(args.cpu_terminal), 'sha256': args.cpu_terminal_sha256},
            full_held_independent_raw_unit_packed_exact=True, source_head_rng_flags_preserved=True,
            files=files, batch=32, batch_sizes=batch_sizes, precision=PRECISION,
            frozen_split=context['spec']['frozen_split'], held_images=12599, query_images=6354, gallery_images=6245, held_products=1993)
    require(torch.equal(rng, torch.random.get_rng_state()) and
            context['selected_context']['source'].numerical_flags() == context['cpus'][args.arm]['numerical_flags'] and
            all(p.grad is None for m in (model, head) for p in m.parameters()), 'inference RNG/flags/gradients changed')
    origins = context['selected_context']['source'].imported_origins(context['selected_context']['initialized']['source_context']['extract'],
                                                                  context['selected_context']['initialized']['packages'])
    for path, digest in origins['files'].items():
        descriptor({'path': path, 'sha256': digest}, context['guards'])
    rehash(context)
    if args.phase == 'cpu':
        args.output.mkdir()
    receipt = {**common, 'phase': args.phase, 'exit_rehash_pass': True, 'origins': origins,
               'input_guards': context['guards'], **resources(context, args.phase, before),
               'invocation': {'argv': sys.argv, 'python': str(Path(sys.executable).resolve()),
                   'python_sha256': sha(Path(sys.executable).resolve()), 'python_version': sys.version,
                   'optimize': sys.flags.optimize, 'invocation_id': os.environ['INVOCATION_ID'],
                   'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES']}}
    publish(args.output / ('proof.json' if args.phase == 'cpu' else 'receipt.json'), receipt)
    return receipt


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--authority', type=Path, required=True)
    p.add_argument('--authority-sha256', required=True)
    p.add_argument('--phase', choices=('cpu', 'export'), required=True)
    p.add_argument('--seed', type=int, choices=SEEDS, required=True)
    p.add_argument('--arm', choices=WIDTHS, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cpu-terminal', type=Path)
    p.add_argument('--cpu-terminal-sha256')
    return p


def main():
    args = parser().parse_args()
    try:
        result = run(args)
    except (OSError, ValueError, ImportError, KeyError, TypeError, AttributeError, RuntimeError, SyntaxError) as error:
        raise SystemExit('Substrate held export rejected: ' + str(error)) from error
    print(json.dumps({'schema': SCHEMA, 'phase': result['phase'], 'output': str(args.output)}, sort_keys=True))


if __name__ == '__main__':
    main()
