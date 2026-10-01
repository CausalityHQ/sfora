#!/usr/bin/env python3
"""One stdlib falsifier for prospective native256 held tooling (no native admission).

Run: python3 -B -S scripts/test_siglip2_substrate_held.py
Synthetic records check exact metadata boundaries only. They never read quality
data, construct a model, load a checkpoint, or qualify native/resource parity.
"""
if not __debug__:
    raise SystemExit('Qualification requires assertions; optimized mode is forbidden')

import ast
import copy
import hashlib
import sys
import time
from pathlib import Path

import export_siglip2_substrate_adaptation as held
import score_siglip2_substrate_adaptation as score


def fixture(seed, arm):
    width = held.WIDTHS[arm]
    cpu_descriptor = {'receipt': {'path': '/synthetic/' + arm + '/proof.json', 'sha256': '1' * 64}}
    mechanics_descriptor = {'receipt': {'path': '/synthetic/' + arm + '/receipt.json', 'sha256': '2' * 64}}
    qualifier = {'path': '/synthetic/' + arm + '/authority.json', 'sha256': '3' * 64}
    approved = {'batches': [list(range(i * 64 % 2004, i * 64 % 2004 + 64)) for i in range(100)],
                'fact': {'dtype': 'torch.int64', 'shape': [100, 64], 'sha256': '4' * 64}, 'sha256': '5' * 64}
    roles = [{'name': 'synthetic.' + str(i), 'shape': [1], 'dtype': 'torch.float32', 'role': 'trainable' if i < 205 else 'frozen'}
             for i in range(400 if arm == 'large' else 448)]
    names = [r['name'] for r in roles if r['role'] == 'trainable'] + ['compact_head.weight', 'compact_head.bias', 'classifier']
    runtime = {'config': {'hidden_size': width, 'num_hidden_layers': 24 if arm == 'large' else 27},
               'roles': roles, 'modules': [{'name': '', 'training': False, 'attributes': {'training': False}}],
               'processor': {'backend': 'torchvision'}}
    defaults = {'lr': .001, 'betas': [.9, .999], 'weight_decay': .05}
    groups = [{'lr': 1e-5}, {'lr': 1e-4}, {'lr': 1e-4}]
    serial = [{'params': list(range(205))}, {'params': [205, 206]}, {'params': [207]}]
    facts = {'runtime': runtime, 'parameter_names': names, 'optimizer_defaults': defaults, 'optimizer_groups': groups,
             'optimizer_state': {'param_groups': serial, 'state': {}}, 'counter': 0, 'seed': 179032,
             'schedules': {str(s): approved['fact'] for s in held.SEEDS},
             'arrays': {'head.weight': {'shape': [128, width]}, 'head.bias': {'shape': [128]}}}
    cpu = {'schema': 'siglip2-substrate-initialized-cpu-v1', 'phase': 'initialized-cpu', 'arm': arm, 'width': width, 'output_dim': 128,
           'source_binding': {'arm': arm}, 'authority_sha256': qualifier['sha256'], 'execution_sha256': '6' * 64,
           'pass': True, 'fresh_source': True, 'reload_exact': True, 'quality_read': False, 'updates': 0, 'state': facts,
           'head_updates': 0, 'optimizer_state_entries': 0, 'initializer_qualified': True, 'source_qualified': True,
           'first_model_released_before_independent_clone': True, 'source_cpu_runtime_and_first2_exact': True,
           'constructor_rng_preserved': True, 'exit_rehash_pass': True, 'optimizer_created': True,
           'training_qualified': False, 'quality_qualified': False, 'cuda_initialized': False, 'gradients_created': False,
           'pca_rerun': False, 'teacher_state_reused': False, 'trained_state_reused': False}
    launch = {'schema': 'native256-substrate-adaptation-launch-v1', 'phase': 'train', 'arm': arm, 'seed': seed,
              'both_locks_held': True, 'execution_sha256': '7' * 64, 'selected_cpu': cpu_descriptor,
              'selected_mechanics': mechanics_descriptor, 'qualifier_authority': qualifier, 'qualifier_execution_sha256': '6' * 64}
    endpoint = {'seed': seed, 'arm': arm, 'launch': {'path': '/synthetic/train-launch.json', 'sha256': '8' * 64},
                'checkpoint': {'path': '/synthetic/' + arm + '/resume.pt', 'sha256': '9' * 64}, 'terminal_state_sha256': 'a' * 64}
    identity = {'arm': arm, 'seed': seed, 'execution_sha256': '7' * 64, 'qualifier_authority': qualifier,
                'selected_cpu': cpu_descriptor, 'config': runtime['config'], 'roles': roles, 'parameter_names': names,
                'optimizer_defaults': defaults, 'optimizer_groups': groups, 'optimizer_serial_groups': serial,
                'buffers_sha256': '0' * 64,
                'schedule_facts': facts['schedules'], 'augmentation': held.AUGMENTATION, 'schedule_sha256': approved['sha256'],
                'runtime': {'modules': [{**r, 'training': True, 'attributes': {'training': True}} for r in runtime['modules']],
                            'processor': runtime['processor']}}
    steps = [{'step': step, 'batch': batch, 'schedule_sha256': approved['sha256'],
              'augmentation_seed': 179032 * 100000 + step, 'seconds': 1., 'rgb_sha256': 'b' * 64,
              'pixels_sha256': 'c' * 64, 'state_sha256': 'd' * 64, 'ce': 1., 'rank': .25, 'loss': 3., 'scale': 128., 'preclip_norm': 1.}
             for step, batch in enumerate(approved['batches'], 1)]
    steps[-1]['state_sha256'] = endpoint['terminal_state_sha256']
    record = {'schema': 'siglip2-substrate-adaptation-v1', 'phase': 'train', 'arm': arm, 'seed': seed,
              'width': width, 'output_dim': 128, 'completed_step': 100, 'optimizer_members': 208,
              'pass': True, 'training_qualified': True, 'fresh_source': True,
              'strict_independent_whole_head_buffers_raw_packed_reload_exact': True,
              'first_references_released_before_reload': True, 'constructor_rng_preserved': True, 'exit_rehash_pass': True,
              'both_locks_held_in_parent_authority': True, 'terminal_exit_and_both_locks_require_parent_receipt': True,
              'quality_read': False, 'quality_qualified': False, 'trained_state_reused': False, 'training_state_discarded': False,
              'authority_sha256': endpoint['launch']['sha256'], 'execution_sha256': launch['execution_sha256'],
              'selected_cpu': cpu_descriptor, 'selected_mechanics': mechanics_descriptor, 'qualifier_authority': qualifier,
              'checkpoint': endpoint['checkpoint'], 'terminal_state_sha256': endpoint['terminal_state_sha256'],
              'augmentation': held.AUGMENTATION, 'reference_pins': {}, 'rank_helper_sha256': 'e' * 64, 'code': {},
              'identity': identity, 'steps': steps, 'first17_mechanics_replay_exact': seed == 179032,
              'initial_state_sha256': 'f' * 64, 'median_update_seconds': 1., 'training_wall_seconds': 100., 'service_seconds': 150.}
    mech_identity = copy.deepcopy(identity)
    mech_identity['seed'] = 179032
    mechanics = {**record, 'phase': 'mechanics', 'seed': 179032, 'completed_step': 17, 'checkpoint': None,
                 'training_state_discarded': True, 'native17_equals_serialized8_plus9_exact': True,
                 'steps': steps[:17], 'resumed_steps': steps[8:17], 'identity': mech_identity}
    return record, endpoint, launch, cpu, mechanics, approved


def rejected(label, call):
    try:
        call()
    except (ValueError, AssertionError):
        return
    raise AssertionError('mutation was admitted: ' + label)


def main():
    started = time.perf_counter()
    endpoints = {}
    for seed, arm in held.ORDER:
        values = fixture(seed, arm)
        held.validate_endpoint(*values)
        endpoints[seed, arm] = values[0]
    for label, mutate in (
        ('mechanics17 masquerades as TRAIN100', lambda v: v[0].update(phase='mechanics', completed_step=17)),
        ('wrong endpoint', lambda v: v[0].update(arm='large')),
        ('swapped seed', lambda v: v[1].update(seed=179041)),
        ('So4001024', lambda v: v[0].update(width=1024)),
        ('head width', lambda v: v[3]['state']['arrays']['head.weight'].update(shape=[128, 1024])),
        ('foreign source', lambda v: v[3]['source_binding'].update(arm='large')),
        ('foreign config', lambda v: v[0]['identity'].update(config={'hidden_size': 1024})),
        ('changed schedule digest', lambda v: v[0]['identity'].update(schedule_sha256='0' * 64)),
        ('different valid draws', lambda v: v[0]['steps'][42].update(batch=[v[0]['steps'][42]['batch'][0] + 2004] + v[0]['steps'][42]['batch'][1:])),
        ('continuation augmentation1000', lambda v: v[0]['steps'][42].update(augmentation_seed=179032 * 100000 + 1043)),
        ('foreign CPU descriptor', lambda v: v[0].update(selected_cpu={})),
        ('retained mechanics state', lambda v: v[4].update(training_state_discarded=False)),
    ):
        values = copy.deepcopy(fixture(179032, 'so400'))
        mutate(values)
        rejected(label, lambda: held.validate_endpoint(*values))
    ordered = [{'seed': seed, 'arm': arm} for seed, arm in held.ORDER]
    held.validate_order(ordered)
    duplicate = copy.deepcopy(ordered); duplicate[3] = duplicate[0]
    rejected('duplicate endpoint', lambda: held.validate_order(duplicate))
    swapped = copy.deepcopy(ordered); swapped[0], swapped[1] = swapped[1], swapped[0]
    rejected('wrong endpoint order', lambda: held.validate_order(swapped))
    original = {name: '1' * 64 for name in held.TRAIN_FILES}
    code = {**original, **{name: '2' * 64 for name in held.ADDED},
            **{name: pin['source'] for name, pin in held.REFERENCES.items()}}
    held.validate_closure(code, original)
    for label, mutate in (
        ('changed trainer prefix', lambda c: c.update({'train_siglip2_substrate_adaptation.py': '0' * 64})),
        ('changed member', lambda c: c.update({'deployed_code_rank.py': '0' * 64})),
        ('missing member', lambda c: c.pop('test_siglip2_substrate_held.py')),
        ('extra member', lambda c: c.update({'unexpected.py': '3' * 64})),
        ('changed fixed reference', lambda c: c.update({'reference_score_inshop_crop_view_pair.py': '0' * 64})),
    ):
        changed = code.copy(); mutate(changed)
        rejected(label, lambda: held.validate_closure(changed, original))
    # Unequal pretrained source identities are expected; training wall is report-only.
    endpoints[179032, 'so400']['training_wall_seconds'] = 201.
    costs = held.paired_cost(endpoints)
    assert costs['179032']['training_wall_ratio'] == 2.01 and not costs['179032']['training_wall_ratio_gate']
    endpoints[179041, 'so400']['service_seconds'] = 226.
    rejected('whole-service cost', lambda: held.paired_cost(endpoints))
    endpoints[179041, 'so400']['service_seconds'] = 150.
    endpoints[179041, 'so400']['steps'][20]['pixels_sha256'] = '0' * 64
    rejected('paired pixels', lambda: held.paired_cost(endpoints))
    policies = {phase: held.policy(phase) for phase in ('cpu', 'export', 'score')}
    assert policies['cpu']['seconds'] == 120 and policies['export']['seconds'] == policies['score']['seconds'] == 300
    assert policies['score']['cuda_visible_devices'] == ''
    held.validate_resource_policies(policies)
    wrong_score_cap = copy.deepcopy(policies); wrong_score_cap['score']['seconds'] = 120
    rejected('score authority retains old120 cap', lambda: held.validate_resource_policies(wrong_score_cap))
    # Freeze unchanged gate definitions without importing the old native scorer.
    root = Path(__file__).absolute().parent
    def definitions(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_bytes()).body
                if isinstance(n, ast.FunctionDef) and n.name in ('averaged_deltas', 'quality_gate')}
    actual_definitions = definitions(root / 'score_siglip2_substrate_adaptation.py')
    assert {name: hashlib.sha256(raw.encode()).hexdigest() for name, raw in actual_definitions.items()} == {
        'averaged_deltas': '9955f2d0d5bcf7ed75f10eca86aaa72f1f3e447f0dc091f6fa252650596ef600',
        'quality_gate': '36276cd9d0616926fd50495686517237c3e9cda85b5a782889a759befdd90a2f'}
    if (root / 'score_late_dense_adaptation.py').exists():
        assert actual_definitions == definitions(root / 'score_late_dense_adaptation.py')
    for name, pin in held.REFERENCES.items():
        source = root / name
        if not source.exists():
            source = root / name.removeprefix('reference_')
        if source.exists():
            raw = source.read_bytes()
            assert hashlib.sha256(raw).hexdigest() == pin['source']
            nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == pin['name']]
            assert hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()).hexdigest() == pin['ast']
            if pin['name'] == 'packed_quality':
                assert len(nodes[0].decorator_list) == 1
    assert not any(n.split('.')[0] in held.NATIVE for n in sys.modules), 'native package entered stdlib falsifier'
    assert time.perf_counter() - started < 5
    print('PASS: four synthetic endpoints; metadata/order/draw/augmentation/closure/cost tampering rejected; no native imports')


if __name__ == '__main__':
    main()
