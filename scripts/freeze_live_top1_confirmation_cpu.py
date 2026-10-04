#!/usr/bin/env python3
"""Source-only live-top1 full CPU freezer; never launch or compute new quality.

Exact CLI: --inputs CANONICAL_JSON --output CANONICAL_NEW_DIRECTORY.
The old eight input roles are retained with schema live-top1-confirmation-cpu-inputs-v1
and one additional key frozen={evaluator:CODE,first_cpu:TEMPLATES,first_selection:FIRST}.
CODE={root,execution_sha256,code}, with the actual pinned evaluator2 pair below.
TEMPLATES={authority:FILE,command:FILE,launch:FILE}; FIRST={terminal:UNIT,verification:FILE}.
The supplied roles must exactly equal these parent-frozen actual descriptors.
FILE and UNIT retain the historical/evaluator definitions. No future hash defaults.
Copies contain the historical exact roles plus the first score authority read by
check_receipt. Four ordered original endpoints and verifications remain mandatory.

Only the three fullCPU files are written. stdout is a source-only receipt with
an exact reversible source correspondence to the untouched historical helper.
The helper and metadata validators stay outside native closure. bash -n alone
is invoked; checkpoints/bundles/data are never read, nor native work admitted.
"""

import argparse
import copy
import difflib
import inspect
import json
from pathlib import Path
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import freeze_fullfeature_confirmation_cpu as historical
from freeze_fullfeature_confirmation_cpu import (
    require, strict_json, digest, local_file, read_bytes, file_fact, add_guard,
    load_module, cgroup, terminal_log, hash_blocks, NATIVE)

ROOT = '/home/riomus/runs/sfora-so400-live-top1-evaluation-source-v1'
FIRST_UNIT = 'sfora-so400-live-top1-evaluation-first-cpu-v1'
UNIT = 'sfora-so400-live-top1-evaluation-full-cpu-v1'
OUTPUT = '/home/riomus/runs/' + UNIT
SCHEMA = 'live-top1-confirmation-cpu-inputs-v1'
INPUT_KEYS = historical.INPUT_KEYS | {'frozen'}
EVALUATOR_PINS = {
    'evaluate_siglip2_compact_ranking.py': '8ecaa206d0343dde117cec97fa92a0179cb8a5991127d4c18dc7917a7c535563',
    'test_compact_ranking_evaluation.py': 'bd49b89abdf6a29a1b18ebe058ffe0b265a41b2736a0e819d17f1b7f1d674e63',
}
TEMPLATE_NAMES = {'authority': 'authority-first-cpu-v1.json',
                  'command': 'first-cpu-v1-command.sh', 'launch': 'first-cpu-v1-launch.sh'}
SHARED = ['require', 'strict_json', 'digest', 'local_file', 'read_bytes',
          'file_fact', 'add_guard', 'load_module', 'cgroup', 'terminal_log', 'hash_blocks']


def frozen_descriptors(value):
    frozen = value['frozen']
    require(isinstance(frozen, dict) and frozen.keys() == {'evaluator', 'first_cpu', 'first_selection'},
            'exact parent-frozen descriptors required')
    code = frozen['evaluator']
    require(isinstance(code, dict) and code.keys() == {'root', 'execution_sha256', 'code'} and
            code['root'] == ROOT and code['code'] == EVALUATOR_PINS,
            'actual strict live-top1 evaluator2 source descriptor required')
    require(value['evaluator'] == {'path': ROOT + '/evaluate_siglip2_compact_ranking.py',
                                 'sha256': code['code']['evaluate_siglip2_compact_ranking.py']} and
            file_fact(value['execution']) == {'path': ROOT + '/execution.json', 'sha256': code['execution_sha256']} and
            value['first_cpu'] == frozen['first_cpu'] and value['first_selection'] == frozen['first_selection'],
            'actual source/templates/first CONTINUE differ from parent-frozen descriptors')
    for key, name in TEMPLATE_NAMES.items():
        require(file_fact(frozen['first_cpu'][key])['path'] == ROOT + '/' + name,
                'fixed live-top1 first CPU template role required')
    first = frozen['first_selection']
    require(isinstance(first, dict) and first.keys() == {'terminal', 'verification'}, 'exact first CONTINUE descriptor required')
    unit = first['terminal']
    file_fact(first['verification'])
    # Complete UNIT validation is performed by the authenticated evaluator before endpoint reads.
    require(unit['unit'] == 'sfora-so400-live-top1-evaluation-first-selection-score-v1' and
            file_fact(unit['receipt'])['path'] == '/home/riomus/runs/' + unit['unit'] + '/receipt.json' and
            file_fact(unit['log'])['path'] == ROOT + '/first-selection-score-v1.log',
            'original live-top1 first selection receipt/log role required')
    return frozen


def check_first_continue(evaluator, first, launch, execution, source):
    """Validate the already recorded score with its original checker; never score data.

    read_json is redirected only to authenticated metadata copies. Replay equality
    checks retain the frozen original receipt's admitted source/concat fields;
    their native replay remains the original evaluator's responsibility.
    """
    helpers = {}
    for key, filename in (('math', 'evaluate_siglip2_genuine_views.py'),
                          ('reference', 'evaluate_siglip2_prototype_residual.py')):
        descriptor = launch['genuine_evaluator' if key == 'math' else 'reference']
        path = Path(__file__).resolve().parent / filename
        helpers[key] = load_module(path, read_bytes(path, descriptor['code'][filename]), '_freeze_' + key)
    require(first['decision'] == 'CONTINUE' and first['concat_quality']['recall_at_1'] == .9648212226066898 and
            first['concat_quality']['map_at_r'] == .8177754035543956,
            'original first CONTINUE and fixed retained concat floors required')
    context = {'args': SimpleNamespace(execution_sha256=launch['execution_sha256']),
               'code': execution, 'launch': launch, 'guards': {}, 'costs': first['cost'],
               'training_context': {'source': first['source'], 'legacy': {'selected': {
                   'source_cpu': {'numerical_flags': first['numerical_flags']}}}},
               'math': helpers['math'], 'reference': helpers['reference'],
               'score_context': {'source_record': {'quality': {'179061': {'control': first['source_quality']}}}},
               'concat_record': {'quality': {'concat': first['concat_quality']}}}
    original_reader = evaluator.read_json
    def metadata_reader(fact, guards):
        add_guard(guards, fact)
        return strict_json(source(fact))
    evaluator.read_json = metadata_reader
    try:
        evaluator.check_receipt(context, first, 'score', stage='first', panel='selection')
    finally:
        evaluator.read_json = original_reader
    for filename, sha in {**execution, 'execution.json': launch['execution_sha256']}.items():
        require(first['input_guards'].get(ROOT + '/' + filename) == sha, 'first evaluator2 source guard differs')
    for endpoint in launch['endpoints'][:2]:
        for fact in (endpoint['launch'], endpoint['terminal']['receipt'], endpoint['terminal']['log'],
                     endpoint['checkpoint'], endpoint['bundle']):
            require(first['input_guards'].get(fact['path']) == fact['sha256'], 'first endpoint/source guard differs')


def correspondence():
    """Record exact inverse edits for the two adapted historical entry points."""
    path = Path(historical.__file__).resolve()
    result = {'historical_helper': {'path': str(path), 'sha256': digest(read_bytes(path))},
              'unchanged': SHARED, 'adapted': {}, 'inverse_exact': True, 'metadata_helpers': {}}
    for filename in ('evaluate_siglip2_genuine_views.py', 'evaluate_siglip2_prototype_residual.py'):
        helper = Path(__file__).resolve().parent / filename
        result['metadata_helpers'][filename] = digest(read_bytes(helper))
    for name in SHARED:
        require(globals()[name] is getattr(historical, name), 'historical guard helper changed: ' + name)
    for name in ('prepare', 'generate'):
        before = inspect.getsource(getattr(historical, name))
        after = inspect.getsource(globals()[name])
        before_lines, after_lines = before.splitlines(keepends=True), after.splitlines(keepends=True)
        offsets = [0]
        for line in after_lines:
            offsets.append(offsets[-1] + len(line))
        edits = [{'start': offsets[j1], 'end': offsets[j2], 'old': ''.join(before_lines[i1:i2])}
                 for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, before_lines, after_lines, autojunk=False).get_opcodes()
                 if tag != 'equal']
        inverse = after
        for edit in reversed(edits):
            inverse = inverse[:edit['start']] + edit['old'] + inverse[edit['end']:]
        require(inverse == before, 'historical source inverse differs: ' + name)
        result['adapted'][name] = {'historical_sha256': digest(before.encode()),
                                  'live_top1_sha256': digest(after.encode()), 'inverse_edits': edits}
    return result


def prepare(inputs_path):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native imports forbidden')
    input_raw = read_bytes(inputs_path)
    value = strict_json(input_raw)
    require(isinstance(value, dict) and value.keys() == INPUT_KEYS and value['schema'] == SCHEMA,
            'exact input schema/keys required')
    require(isinstance(value['first_cpu'], dict) and value['first_cpu'].keys() == {'authority', 'command', 'launch'} and
            isinstance(value['first_selection'], dict) and value['first_selection'].keys() == {'terminal', 'verification'},
            'exact template/first_selection keys required')
    frozen = frozen_descriptors(value)
    pins = {**frozen['evaluator']['code'], 'execution.json': frozen['evaluator']['execution_sha256'],
            **{name: frozen['first_cpu'][key]['sha256'] for key, name in TEMPLATE_NAMES.items()}}
    first_selection = frozen['first_selection']['terminal']
    scientific = {}
    for fact, name in ((value['evaluator'], 'evaluate_siglip2_compact_ranking.py'),
                       (value['execution'], 'execution.json'),
                       *((value['first_cpu'][key], name) for key, name in
                         (('authority', 'authority-first-cpu-v1.json'), ('command', 'first-cpu-v1-command.sh'),
                          ('launch', 'first-cpu-v1-launch.sh')))):
        require(file_fact(fact) == {'path': ROOT + '/' + name, 'sha256': pins[name]}, 'frozen source/template differs: ' + name)
        add_guard(scientific, fact)
    add_guard(scientific, {'path': ROOT + '/test_compact_ranking_evaluation.py',
                           'sha256': pins['test_compact_ranking_evaluation.py']})
    require(isinstance(value['copies'], dict), 'exact local copies map required')
    copies = value['copies']

    def source(fact):
        add_guard(scientific, fact)
        require(fact['path'] in copies, 'missing local copy: ' + fact['path'])
        return read_bytes(copies[fact['path']], fact['sha256'])

    evaluator_raw = source(value['evaluator'])
    evaluator = load_module(ROOT + '/evaluate_siglip2_compact_ranking.py', evaluator_raw, '_freeze_evaluator')
    require(evaluator.SCHEMA == 'siglip2-compact-live-top1-evaluation-v1' and
            evaluator.AUTHORITY_SCHEMA == 'siglip2-compact-live-top1-evaluation-launch-v1',
            'strict live-top1 evaluator schema required')
    execution = strict_json(source(value['execution']))
    require(execution == {n: pins[n] for n in evaluator.FILES}, 'exact evaluator2 execution closure required')
    source({'path': ROOT + '/test_compact_ranking_evaluation.py', 'sha256': execution['test_compact_ranking_evaluation.py']})
    first_cpu = strict_json(source(value['first_cpu']['authority']))
    args = SimpleNamespace(execution_sha256=pins['execution.json'], phase='cpu', arm=None, seed=None)
    evaluator.check_launch(first_cpu, args)
    require(first_cpu['stage'] == 'first' and first_cpu['panel'] == 'selection', 'original first CPU required')
    require(isinstance(value['endpoints'], list) and len(value['endpoints']) == 4 and
            value['endpoints'][:2] == first_cpu['endpoints'], 'unchanged original 061 endpoints required')
    require(value['first_selection'] == frozen['first_selection'], 'parent-frozen first CONTINUE differs')
    launch = copy.deepcopy(first_cpu)
    launch.update(stage='full', panel='selection', phase='cpu', arm=None, seed=None, selected_cpu=None,
                  exports={}, selection_go=None, first_selection=value['first_selection']['terminal'], endpoints=value['endpoints'])
    evaluator.check_launch(launch, args)
    first = strict_json(source(first_selection['receipt']))
    first_verification_fact = value['first_selection']['verification']
    first_verification = strict_json(read_bytes(first_verification_fact['path'], first_verification_fact['sha256']))
    check_first_continue(evaluator, first, launch, execution, source)
    require(first_verification['decision'] == 'CONTINUE' and first_verification['cost'] == first['cost'] and
            first_verification['deltas'] == first['mean_deltas'] and first_verification['quality'] == {
                arm: {metric: first['quality']['179061'][arm][metric] for metric in ('recall_at_1', 'map_at_r')}
                for arm in evaluator.ARMS}, 'complete original first score verification differs')
    terminal_log(first, first_selection, source(first_selection['log']), first_verification, 500)
    require(len({e['checkpoint']['sha256'] for e in launch['endpoints']}) == 4,
            'distinct actual checkpoint hashes required')
    require(isinstance(value['verifications'], list) and len(value['verifications']) == 4,
            'four ordered actual verification FILEs required')
    trainer_path = Path(__file__).resolve().parent / 'train_siglip2_compact_ranking.py'
    trainer = load_module(trainer_path, read_bytes(trainer_path, launch['training']['code'][trainer_path.name]), '_freeze_trainer')
    trainer_test = trainer_path.with_name('test_siglip2_compact_ranking.py')
    read_bytes(trainer_test, launch['training']['code'][trainer_test.name])
    records, original_launch = {}, None
    for endpoint, verification_fact in zip(launch['endpoints'], value['verifications'], strict=True):
        seed, arm, unit = endpoint['seed'], endpoint['arm'], endpoint['terminal']
        record = strict_json(source(unit['receipt']))
        endpoint_launch = strict_json(source(endpoint['launch']))
        trainer.check_launch(endpoint_launch, SimpleNamespace(execution_sha256=launch['training']['execution_sha256'],
                                                              phase='train', arm=arm, seed=seed))
        if original_launch is None:
            original_launch = endpoint_launch
        require(trainer.method(endpoint_launch) == trainer.method(original_launch) and
                all(endpoint_launch[k] == original_launch[k] for k in ('selected_cpu', 'selected_mechanics')),
                'same frozen recipe/CPU/both discarded mechanics required')
        require(record['schema'] == trainer.SCHEMA and record['phase'] == 'train' and
                record['seed'] == seed and record['arm'] == arm and record['launch'] == endpoint_launch and
                record['authority'] == endpoint['launch'] and record['authority_sha256'] == endpoint['launch']['sha256'] and
                record['code'] == launch['training']['code'] and record['execution_sha256'] == launch['training']['execution_sha256'] and
                record['completed_step'] == 128 and record['checkpoint'] == endpoint['checkpoint'] and
                record['bundle'] == endpoint['bundle'] and record['terminal_state_sha256'] == endpoint['terminal_state_sha256'] and
                record['inference_state_sha256'] == endpoint['inference_state_sha256'] and
                record['output'] == str(Path(unit['receipt']['path']).parent), 'complete TRAIN128/artifact role binding differs')
        expected_argv = trainer.cli(launch['training']['root'], endpoint['launch']['path'], endpoint['launch']['sha256'],
                                    launch['training']['execution_sha256'], 'train', arm, seed, record['output'])
        require(record['invocation']['argv'] == expected_argv and record['invocation']['optimize'] == 0 and
                record['invocation']['cuda_visible_devices'] == '0' and record['invocation']['cublas_workspace_config'] == ':4096:8' and
                record['resource_policy'] == trainer.policy('train') and record['quality_read'] is False and
                record['training_state_discarded'] is False and record['resumed_steps'] == [] and
                all(record[k] is True for k in ('pass', 'strict_reload_exact', 'exit_rehash_pass', 'sequential_model_ownership',
                    'forward_oracle_exact', 'native_training_inference_exact', 'inference_artifact_independent',
                    'bundle_original_dependencies_denied', 'both_locks_held_in_parent_authority',
                    'exact_four_native_membership', 'source_substitution_rejected', 'omitted_C_mutant_rejected', 'wrong_mu_mutant_rejected')) and
                record['C_exact_zero'] is False and record['C_trainable'] is True and
                record['residual_nonzero_witness'] is True and
                record['identity']['seed'] == seed and record['identity']['source'] == record['source'] and
                record['identity']['method'] == trainer.method(endpoint_launch) and
                record['identity']['numerical_flags'] == record['numerical_flags'] and
                record['current_C_sha256'] != record['initial_C_sha256'] and
                0 < record['total_training_core_seconds'] < record['wall_seconds'] and
                len(record['steps']) == 128 and [s['step'] for s in record['steps']] == list(range(1, 129)) and
                all(s['arm'] == arm for s in record['steps']), 'complete recorded TRAIN proof/CLI differs')
        for fact in (endpoint['checkpoint'], endpoint['bundle'], endpoint['launch']):
            require(record['input_guards'].get(fact['path']) == fact['sha256'], 'TRAIN artifact/source guard differs')
        for name, sha in {**launch['training']['code'], 'execution.json': launch['training']['execution_sha256']}.items():
            require(record['input_guards'].get(launch['training']['root'] + '/' + name) == sha,
                    'frozen trainer2 source guard differs')
        verification = strict_json(read_bytes(file_fact(verification_fact)['path'], verification_fact['sha256']))
        require(verification['phase'] == 'train' and verification['arm'] == arm and verification['unit'] == unit['unit'] and
                verification['quality_read'] is False and verification['rss_kib'] == unit['native_peak_rss_kib'],
                'ordered TRAIN verification role differs')
        terminal_log(record, unit, source(unit['log']), verification, 600)
        records[seed, arm] = {**record, 'service_seconds': unit['service_seconds']}
    for seed in evaluator.SEEDS:
        evaluator.check_paired_initialization(records[seed, 'control'], records[seed, 'candidate'])
    require(all(r['source'] == records[179061, 'control']['source'] for r in records.values()), 'common frozen TRAIN source differs')
    costs = evaluator.paired_cost(records, 'full')
    require(all(v['pass'] for v in costs.values()), 'actual paired whole/core costs exceed 1.50')
    require(first['decision'] == 'CONTINUE' and first['launch']['endpoints'] == launch['endpoints'][:2] and
            first['cost'] == {'179061': costs['179061']} and first['source_code'] == execution and
            first['source'] == records[179061, 'control']['source'] and
            first['numerical_flags'] == records[179061, 'control']['numerical_flags'],
            'original first CONTINUE/source/pair binding differs')
    command = source(value['first_cpu']['command']).decode('utf-8')
    shell_launch = source(value['first_cpu']['launch']).decode('utf-8')
    require(copies.keys() == scientific.keys(), 'missing/extra local copy roles')
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import occurred during source admission')
    return value, launch, execution, costs, command, shell_launch, digest(input_raw), scientific


def generate(inputs_path, output):
    helper_path = Path(__file__).resolve()
    ledger = correspondence()
    helper_sha = digest(read_bytes(helper_path))
    output = Path(output)
    require(output.is_absolute() and str(output) == str(output.absolute()) and output.parent.resolve() == output.parent and
            output.parent.is_dir() and not output.exists() and not output.is_symlink(), 'canonical exclusive NEWDIR required')
    value, launch, execution, costs, command, shell_launch, input_sha, scientific = prepare(inputs_path)
    pins = {**value['frozen']['evaluator']['code'], 'execution.json': value['execution']['sha256'],
            **{name: value['first_cpu'][key]['sha256'] for key, name in TEMPLATE_NAMES.items()}}
    local_inputs = [local_file(str(inputs_path)), *(local_file(p) for p in value['copies'].values()),
                    *(local_file(v['path']) for v in value['verifications']),
                    local_file(value['first_selection']['verification']['path'])]
    require(all(not p.is_relative_to(output) for p in local_inputs), 'output overlaps immutable inputs')
    authority_raw = (json.dumps(launch, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    authority_path = ROOT + '/authority-full-cpu-v1.json'
    _, guards = hash_blocks(command)
    for p, h in scientific.items():
        add_guard(guards, {'path': p, 'sha256': h})
    for endpoint in launch['endpoints']:
        for fact in (endpoint['launch'], endpoint['terminal']['receipt'], endpoint['terminal']['log'],
                     endpoint['checkpoint'], endpoint['bundle']):
            add_guard(guards, fact)
    add_guard(guards, {'path': authority_path, 'sha256': digest(authority_raw)})
    historical = hash_blocks(command)[1]
    rows = ''.join(h + '  ' + p + '\n' for p, h in sorted(guards.items()) if p not in historical)
    # Append by full path. Historical first-authority/prerequisite rows remain intact.
    command = command.replace("\nHASHES\n", '\n' + rows + 'HASHES\n')
    old_lines = [line for line in command.splitlines() if line.startswith('/home/riomus/group-learning/.venv/bin/python -B ')]
    require(len(old_lines) == 1, 'one frozen evaluator argv required')
    old_argv = shlex.split(old_lines[0])
    expected = ['/home/riomus/group-learning/.venv/bin/python', '-B', ROOT + '/evaluate_siglip2_compact_ranking.py',
                '--execution-sha256', pins['execution.json'], '--authority', ROOT + '/authority-first-cpu-v1.json',
                '--authority-sha256', pins['authority-first-cpu-v1.json'], '--phase', 'cpu',
                '--output', '/home/riomus/runs/' + FIRST_UNIT]
    require(old_argv == expected, 'historical evaluator argv differs')
    argv = expected[:]
    argv[6], argv[8], argv[12] = authority_path, digest(authority_raw), OUTPUT
    command = command.replace(old_lines[0] + '\n', shlex.join(argv) + '\n', 1)
    command_raw = command.encode()
    old_command, new_command = ROOT + '/first-cpu-v1-command.sh', ROOT + '/full-cpu-v1-command.sh'
    replacements = [('/home/riomus/runs/' + FIRST_UNIT, OUTPUT, 1),
                    ('--unit=' + FIRST_UNIT + ' ', '--unit=' + UNIT + ' ', 1),
                    (old_command, new_command, 3),
                    ("printf '%s  %s\\n' " + pins['first-cpu-v1-command.sh'] + ' ' + new_command,
                     "printf '%s  %s\\n' " + digest(command_raw) + ' ' + new_command, 1)]
    for before, after, count in replacements:
        require(shell_launch.count(before) == count, 'exact launcher substitution differs: ' + before)
        shell_launch = shell_launch.replace(before, after)
    files = {'authority-full-cpu-v1.json': authority_raw, 'full-cpu-v1-command.sh': command_raw,
             'full-cpu-v1-launch.sh': shell_launch.encode()}
    with TemporaryDirectory(prefix='sfora-full-cpu-syntax-') as temporary:
        for name, raw in files.items():
            p = Path(temporary) / name
            p.write_bytes(raw)
            if name.endswith('.sh'):
                subprocess.run(['/bin/bash', '-n', str(p)], check=True, capture_output=True, timeout=5)
    # Recheck every admitted metadata byte immediately before exclusive publication.
    for p, h in scientific.items():
        read_bytes(value['copies'][p], h)
    read_bytes(inputs_path, input_sha)
    for fact in [*value['verifications'], value['first_selection']['verification']]:
        read_bytes(fact['path'], fact['sha256'])
    read_bytes(helper_path, helper_sha)
    for name, sha in launch['training']['code'].items():
        read_bytes(helper_path.parent / name, sha)
    require(correspondence() == ledger, 'helper/source correspondence changed before publication')
    output.mkdir()
    try:
        for name, raw in files.items():
            p = output / name
            with p.open('xb') as stream:
                stream.write(raw)
            p.chmod(0o755 if name.endswith('.sh') else 0o644)
    except BaseException:
        for name in files:
            (output / name).unlink(missing_ok=True)
        output.rmdir()
        raise
    return {'schema': 'live-top1-confirmation-cpu-freeze-receipt-v1', 'source_only': True, 'native_pass': False,
            'inputs': {'path': str(inputs_path), 'sha256': input_sha},
            'helper': {'path': str(helper_path), 'sha256': helper_sha},
            'execution': value['execution'], 'source_code': execution, 'scientific_inputs': scientific,
            'native_argv': argv, 'unit': UNIT, 'native_output': OUTPUT, 'costs': costs,
            'outputs': {n: {'path': str(output / n), 'sha256': digest(raw)} for n, raw in files.items()},
            'source_correspondence': ledger, 'bash_syntax_pass': True, 'helper_in_scientific_closure': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        require(__debug__, 'optimized mode forbidden')
        require(sys.argv[1:] == ['--inputs', args.inputs, '--output', args.output],
                'exact CLI order required: --inputs JSON --output NEWDIR')
        require(str(Path(args.output)) == args.output, 'canonical output path spelling required')
        receipt = generate(local_file(args.inputs), Path(args.output))
        receipt['generator_argv'] = sys.orig_argv
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    except (ValueError, KeyError, TypeError, OSError, subprocess.SubprocessError) as error:
        parser.exit(1, 'freeze rejected: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
