#!/usr/bin/env python3
"""Generate a prospective full-stage CPU launcher; never execute a native job.

CLI: --inputs CANONICAL_JSON --output CANONICAL_NEW_DIRECTORY.
Exact input keys: schema,evaluator,execution,first_cpu,first_selection,endpoints,
verifications,copies. schema is fullfeature-confirmation-cpu-inputs-v1.
FILE={path,sha256}; UNIT and endpoint objects use the unchanged evaluator schema.
first_cpu={authority:FILE,command:FILE,launch:FILE}; first_selection={terminal:UNIT,
verification:FILE}. verifications contains four ordered local FILEs. copies maps
exact scientific paths to canonical local copies of evaluator/test/execution,
the three first_cpu templates, first_selection receipt/log, and each endpoint's
launch/receipt/log. No extra copies. Checkpoint/bundle descriptors are bound to
the actual receipt; remote payload hashing/admission remains the parent's job.

Only three files are written. stdout is a source-only freeze receipt. This helper
is outside the scientific execution closure; bash -n is the only shell action.
"""

import argparse
import copy
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace


ROOT = '/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-source-v1'
FIRST_UNIT = 'sfora-so400-fullfeature-residual-evaluation-first-cpu-v1'
UNIT = 'sfora-so400-fullfeature-residual-evaluation-full-cpu-v1'
OUTPUT = '/home/riomus/runs/' + UNIT
SCHEMA = 'fullfeature-confirmation-cpu-inputs-v1'
INPUT_KEYS = {'schema', 'evaluator', 'execution', 'first_cpu', 'first_selection',
              'endpoints', 'verifications', 'copies'}
PINS = {
    'evaluate_siglip2_compact_ranking.py': '3684268825e8a03592ad355a0752b582245f1d05fd0bcb3ed084a01ef68140b0',
    'test_compact_ranking_evaluation.py': '426f0fe67434db451a50440b3a2be7b342afea9d2ad2da09616f76ff88e43167',
    'execution.json': 'c7b5916d70432dcd5bd990f007b2dc6a41c4c1912582a6838daf7e4554fa80a4',
    'authority-first-cpu-v1.json': '64deae60016adc18def2606339cfc8dc9c90f3000ea253b9da5c04e9433ffc91',
    'first-cpu-v1-command.sh': '25123e1f6f539804318cce05efdba4bc148b11207544dd0c4c42aa18e02c7eb7',
    'first-cpu-v1-launch.sh': '71ad2782ca1dd5ffe4516f4dae0333cbc76684b25ea56cea24cbf602bf7ec876',
}
FIRST_VERIFICATION_SHA = '9d1df306e9c28603b8fa0558f7ad7d27f2ebdbe3a9c464ee141357d8187b5410'
FIRST_SELECTION = {
    'receipt': {'path': '/home/riomus/runs/sfora-so400-fullfeature-residual-evaluation-first-selection-score-v1/receipt.json',
                'sha256': '02f79083cecd1ddf4d9f8c28996df305c517be8af65abbfd47e1e8d866e44392'},
    'log': {'path': ROOT + '/first-selection-score-v1.log',
            'sha256': 'a14e340b59c6cdb704ee86e26c2d607db84746a4dafe25fb0b7f883263941c87'},
    'unit': 'sfora-so400-fullfeature-residual-evaluation-first-selection-score-v1',
    'invocation_id': 'ba81beb6caaa4c72bb766774e867181e',
    'service_seconds': 303.282, 'native_peak_rss_kib': 993452, 'both_locks_held': True,
}
NATIVE = {'torch', 'numpy', 'PIL', 'transformers', 'safetensors', 'torchvision', 'sfora'}
MAX_BYTES = 64 * 1024**2


def require(condition, message):
    if not condition:
        raise ValueError(message)


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON key: ' + key)
            result[key] = value
        return result
    return json.loads(raw, object_pairs_hook=pairs,
                      parse_constant=lambda value: require(False, 'nonfinite JSON: ' + value))


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def local_file(path):
    require(isinstance(path, str), 'local path must be a string')
    p = Path(path)
    require(p.is_absolute() and str(p) == path and p.resolve() == p and p.is_file(),
            'canonical local regular file required: ' + path)
    return p


def read_bytes(path, expected=None):
    with local_file(str(path)).open('rb') as stream:
        raw = stream.read(MAX_BYTES + 1)
    require(len(raw) <= MAX_BYTES, 'metadata/source exceeds 64MiB')
    require(expected is None or digest(raw) == expected, 'SHA256 differs: ' + str(path))
    return raw


def file_fact(value):
    require(isinstance(value, dict) and value.keys() == {'path', 'sha256'} and
            isinstance(value['path'], str) and re.fullmatch(r'/[A-Za-z0-9_./-]+', value['path']) and
            str(Path(value['path'])) == value['path'] and '..' not in Path(value['path']).parts and
            isinstance(value['sha256'], str) and re.fullmatch('[0-9a-f]{64}', value['sha256']),
            'exact canonical FILE required')
    return value


def add_guard(guards, fact):
    file_fact(fact)
    require(guards.setdefault(fact['path'], fact['sha256']) == fact['sha256'],
            'conflicting path/SHA256: ' + fact['path'])


def load_module(path, raw, name):
    module = ModuleType(name)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), vars(module))
    return module


def cgroup(value, unit):
    require(Path(value['path']).name == unit + '.service', 'terminal cgroup/unit differs')
    values = value['values']
    require(int(values['memory.max']) == 8 * 1024**3 and
            0 < int(values['memory.peak']) <= 8 * 1024**3 and
            all(int(values[k]) == 0 for k in ('memory.swap.current', 'memory.swap.max', 'memory.swap.peak')),
            'whole-unit memory/swap cap differs')
    events = dict(line.split() for line in values['memory.events'].splitlines())
    require(events.keys() == {'low', 'high', 'max', 'oom', 'oom_kill', 'oom_group_kill'} and
            all(int(v) == 0 for v in events.values()), 'whole-unit memory events differ')


def terminal_log(record, unit, raw, verification, seconds):
    """Bind original normal-exit evidence, including both complete cgroup footers."""
    identity, name = unit['invocation_id'], unit['unit']
    require(record['invocation']['invocation_id'] == identity and
            0 < record['wall_seconds'] <= unit['service_seconds'] <= seconds and
            0 < record['process_peak_rss_kib'] <= unit['native_peak_rss_kib'] <= 8 * 1024**2,
            'whole-service invocation/duration/RSS binding differs')
    lines = raw.decode('utf-8').splitlines()
    required = [f'Running as unit: {name}.service; invocation ID: {identity}',
                '\tExit status: 0', 'Finished with result: success',
                'Main processes terminated with: code=exited/status=0', '\tSwaps: 0',
                'Memory swap peak: 0B',
                f"\tMaximum resident set size (kbytes): {unit['native_peak_rss_kib']}"]
    require(all(lines.count(line) == 1 for line in required), 'original normal-exit log differs')
    runtimes = [line.removeprefix('Service runtime: ') for line in lines if line.startswith('Service runtime: ')]
    require(len(runtimes) == 1, 'one original service runtime required')
    match = re.fullmatch(r'(?:(\d+)min )?(\d+(?:\.\d+)?)(s|ms)', runtimes[0])
    require(match is not None, 'service runtime format differs')
    minutes, native_seconds, suffix = match.groups()
    native_seconds = Decimal(native_seconds) / (1000 if suffix == 'ms' else 1)
    require((minutes is None or native_seconds < 60) and
            Decimal(minutes or '0') * 60 + native_seconds == Decimal(str(unit['service_seconds'])),
            'service runtime numeric binding differs')
    footers = {}
    for tag in ('FINAL_CGROUP', 'STOP_CGROUP'):
        found = [strict_json(line.removeprefix(tag + ' ')) for line in lines if line.startswith(tag + ' ')]
        require(len(found) == 1 and found[0]['invocation_id'] == identity, 'one original ' + tag + ' required')
        footers[tag] = found[0]
    final, stop = footers['FINAL_CGROUP'], footers['STOP_CGROUP']
    require(type(final['command_exit_status']) is int and final['command_exit_status'] == 0 and
            stop['exit_code'] == 'exited' and stop['exit_status'] == '0' and stop['service_result'] == 'success',
            'original command/service footer failed')
    values = [record['cgroup_before'], record['cgroup_after'], final, stop]
    for value in values:
        cgroup(value, name)
        require(value['path'] == final['path'], 'enclosing terminal cgroup changed')
    peaks = [int(value['values']['memory.peak']) for value in values]
    require(peaks == sorted(peaks) and verification['host_peak_bytes'] == peaks[-1] and
            verification['pass'] is True and verification['invocation_id'] == identity and
            verification['service_seconds'] == unit['service_seconds'] and
            verification['receipt_sha256'] == unit['receipt']['sha256'] and
            verification['log_sha256'] == unit['log']['sha256'], 'actual verification/terminal binding differs')


def hash_blocks(command):
    blocks = re.findall(r"(?m)^sha256sum -c <<'HASHES'\n(.*?)^HASHES\n", command, re.DOTALL)
    require(len(blocks) == 2 and blocks[0] == blocks[1], 'two unchanged pre/post hash blocks required')
    guards = {}
    for line in blocks[0].splitlines():
        match = re.fullmatch(r'([0-9a-f]{64})  (/[A-Za-z0-9_./-]+)', line)
        require(match is not None, 'exact historical hash row required')
        add_guard(guards, {'path': match[2], 'sha256': match[1]})
    return blocks, guards


def prepare(inputs_path):
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native imports forbidden')
    input_raw = read_bytes(inputs_path)
    value = strict_json(input_raw)
    require(isinstance(value, dict) and value.keys() == INPUT_KEYS and value['schema'] == SCHEMA,
            'exact input schema/keys required')
    require(isinstance(value['first_cpu'], dict) and value['first_cpu'].keys() == {'authority', 'command', 'launch'} and
            isinstance(value['first_selection'], dict) and value['first_selection'].keys() == {'terminal', 'verification'},
            'exact template/first_selection keys required')
    scientific = {}
    for fact, name in ((value['evaluator'], 'evaluate_siglip2_compact_ranking.py'),
                       (value['execution'], 'execution.json'),
                       *((value['first_cpu'][key], name) for key, name in
                         (('authority', 'authority-first-cpu-v1.json'), ('command', 'first-cpu-v1-command.sh'),
                          ('launch', 'first-cpu-v1-launch.sh')))):
        require(file_fact(fact) == {'path': ROOT + '/' + name, 'sha256': PINS[name]}, 'frozen source/template differs: ' + name)
        add_guard(scientific, fact)
    add_guard(scientific, {'path': ROOT + '/test_compact_ranking_evaluation.py',
                           'sha256': PINS['test_compact_ranking_evaluation.py']})
    require(isinstance(value['copies'], dict), 'exact local copies map required')
    copies = value['copies']

    def source(fact):
        add_guard(scientific, fact)
        require(fact['path'] in copies, 'missing local copy: ' + fact['path'])
        return read_bytes(copies[fact['path']], fact['sha256'])

    evaluator_raw = source(value['evaluator'])
    evaluator = load_module(copies[value['evaluator']['path']], evaluator_raw, '_freeze_evaluator')
    execution = strict_json(source(value['execution']))
    require(execution == {n: PINS[n] for n in evaluator.FILES}, 'exact evaluator2 execution closure required')
    source({'path': ROOT + '/test_compact_ranking_evaluation.py', 'sha256': execution['test_compact_ranking_evaluation.py']})
    first_cpu = strict_json(source(value['first_cpu']['authority']))
    args = SimpleNamespace(execution_sha256=PINS['execution.json'], phase='cpu', arm=None, seed=None)
    evaluator.check_launch(first_cpu, args)
    require(first_cpu['stage'] == 'first' and first_cpu['panel'] == 'selection', 'original first CPU required')
    require(isinstance(value['endpoints'], list) and len(value['endpoints']) == 4 and
            value['endpoints'][:2] == first_cpu['endpoints'], 'unchanged original 061 endpoints required')
    require(value['first_selection']['terminal'] == FIRST_SELECTION and
            file_fact(value['first_selection']['verification'])['sha256'] == FIRST_VERIFICATION_SHA,
            'original first CONTINUE UNIT/verification required')
    launch = copy.deepcopy(first_cpu)
    launch.update(stage='full', panel='selection', phase='cpu', arm=None, seed=None, selected_cpu=None,
                  exports={}, selection_go=None, first_selection=value['first_selection']['terminal'], endpoints=value['endpoints'])
    evaluator.check_launch(launch, args)
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
                record['C_exact_zero'] is (arm == 'control') and record['C_trainable'] is (arm == 'candidate') and
                record['residual_nonzero_witness'] is (arm == 'candidate') and
                record['identity']['seed'] == seed and record['identity']['source'] == record['source'] and
                record['identity']['method'] == trainer.method(endpoint_launch) and
                record['identity']['numerical_flags'] == record['numerical_flags'] and
                ((record['current_C_sha256'] == record['initial_C_sha256']) is (arm == 'control')) and
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
    first = strict_json(source(FIRST_SELECTION['receipt']))
    first_verification = strict_json(read_bytes(value['first_selection']['verification']['path'], FIRST_VERIFICATION_SHA))
    require(first['decision'] == 'CONTINUE' and first['launch']['endpoints'] == launch['endpoints'][:2] and
            first['cost'] == {'179061': costs['179061']} and first['source_code'] == execution and
            first['source'] == records[179061, 'control']['source'], 'original first CONTINUE/source/pair binding differs')
    terminal_log(first, FIRST_SELECTION, source(FIRST_SELECTION['log']), first_verification, 500)
    command = source(value['first_cpu']['command']).decode('utf-8')
    shell_launch = source(value['first_cpu']['launch']).decode('utf-8')
    require(copies.keys() == scientific.keys(), 'missing/extra local copy roles')
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native import occurred during source admission')
    return value, launch, execution, costs, command, shell_launch, digest(input_raw), scientific


def generate(inputs_path, output):
    helper_path = Path(__file__).resolve()
    helper_sha = digest(read_bytes(helper_path))
    output = Path(output)
    require(output.is_absolute() and str(output) == str(output.absolute()) and output.parent.resolve() == output.parent and
            output.parent.is_dir() and not output.exists() and not output.is_symlink(), 'canonical exclusive NEWDIR required')
    value, launch, execution, costs, command, shell_launch, input_sha, scientific = prepare(inputs_path)
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
                '--execution-sha256', PINS['execution.json'], '--authority', ROOT + '/authority-first-cpu-v1.json',
                '--authority-sha256', PINS['authority-first-cpu-v1.json'], '--phase', 'cpu',
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
                    ("printf '%s  %s\\n' " + PINS['first-cpu-v1-command.sh'] + ' ' + new_command,
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
    return {'schema': 'fullfeature-confirmation-cpu-freeze-receipt-v1', 'source_only': True, 'native_pass': False,
            'inputs': {'path': str(inputs_path), 'sha256': input_sha},
            'helper': {'path': str(helper_path), 'sha256': helper_sha},
            'execution': value['execution'], 'source_code': execution, 'scientific_inputs': scientific,
            'native_argv': argv, 'unit': UNIT, 'native_output': OUTPUT, 'costs': costs,
            'outputs': {n: {'path': str(output / n), 'sha256': digest(raw)} for n, raw in files.items()},
            'bash_syntax_pass': True, 'helper_in_scientific_closure': False}


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
