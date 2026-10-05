#!/usr/bin/env python3
"""Pinned observation only. --check is stdlib-only; --native belongs to root."""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import ModuleType

SOURCE_SHA = 'd62c2dbf7a58e0b4af03efccdb386ee84e38b9acc5ace6bbefcb29ddca915b38'
TEST_SHA = '39a36691b777052bd44554e73011aaa936400efea1cca46b817c3de380a915c1'
EXECUTION_SHA = '7f609e6076033e7227d6f196e7d2ebcf8b3d48129bf004df862252e6764b5938'
BASE = 'bb3022321af961c75b5735693e3e2de73e5571e3'
MESSAGE = 'qualified per-scope/seed CPU source/teachers/schedule/initialization differs'
STOP = 'Observation only: original admission held; updates remain forbidden'
REMOTE_SOURCE = '/home/riomus/runs/sfora-so400-identity-diversity-train-source-v4/train_siglip2_identity_diversity.py'

# The report owns only plain JSON metadata, never tensor storage or an extra witness.
REPORT = '''
try:
    def __sfora_structure(value, depth=0):
        kind = type(value).__module__ + '.' + type(value).__qualname__
        if isinstance(value, torch.Tensor):
            return {'type': kind, 'shape': list(value.shape), 'dtype': str(value.dtype),
                    'device': str(value.device), 'layout': str(value.layout),
                    'stride': list(value.stride()), 'requires_grad': value.requires_grad}
        if type(value) is dict:
            keys = sorted(value, key=repr)
            result = {'type': kind, 'length': len(value),
                      'keys': [str(k)[:120] for k in keys[:32]],
                      'keys_truncated': len(keys) > 32}
            if depth < 3:
                result['members'] = {str(k)[:120]: __sfora_structure(value[k], depth+1)
                                     for k in keys[:32]}
            return result
        if type(value) in (tuple, list):
            return {'type': kind, 'length': len(value),
                    'first_type': (type(value[0]).__module__ + '.' +
                                   type(value[0]).__qualname__) if value else None}
        return {'type': kind}
    __sfora_pairs = [
        ('static_sha256', ident['static_sha256'], qualified['identity']['static_sha256']),
        ('scope', ident['scope'], qualified['identity']['scope']),
        ('schedule_provenance_sha256', ident['schedule_provenance_sha256'],
         qualified['identity']['schedule_provenance_sha256']),
        ('initial_raw_unit_packed_sha256', initial_witness,
         qualified['initial_raw_unit_packed_sha256']),
        ('common_input_raw_unit_packed_sha256', context['common_input_raw_unit_packed_sha256'],
         qualified['common_input_raw_unit_packed_sha256']),
        ('common_statistics_sha256', context['common_statistics_sha256'],
         qualified['common_statistics_sha256'])]
    __sfora_report = {
        'event': 'INIT_ADMISSION_OBSERVATION_V1', 'production_pass': False,
        'arm': args.arm, 'seed': args.seed, 'device': state['device'], 'counter': state['counter'],
        'conjuncts': [{'index': i+1, 'name': name, 'equal': current == cpu,
                       'current': current, 'qualified_cpu': cpu,
                       'current_type': type(current).__name__, 'cpu_type': type(cpu).__name__}
                      for i, (name, current, cpu) in enumerate(__sfora_pairs)],
        'existing_hashes': {k: context[k] for k in
            ('initial_static_sha256', 'common_initial_sha256', 'common_statistics_sha256',
             'initial_A_sha256', 'initial_C_sha256', 'mu_train_sha256',
             'mu_train_provenance_sha256')},
        'static_metadata': {k: __sfora_structure(state[k]) for k in STATIC_KEYS},
        'initial_static_metadata': {k: __sfora_structure(context['initial'][k]) for k in STATIC_KEYS},
        'head_note': 'Metadata uses admitted head tensors; original static hash uses actual head_object.state_dict.',
        'limits': 'No per-member byte hashes or extra CPU/CUDA witness; no numerical cause inferred.'}
    print(json.dumps(__sfora_report, sort_keys=True, allow_nan=False), flush=True)
except Exception as __sfora_report_error:
    try:
        print(json.dumps({'event': 'INIT_ADMISSION_OBSERVATION_ERROR_V1',
                          'error_type': type(__sfora_report_error).__name__,
                          'error': str(__sfora_report_error)[:1024],
                          'original_require_still_mandatory': True}), flush=True)
    except Exception:
        pass
'''


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def dump(node, attributes=False):
    return ast.dump(node, include_attributes=attributes)


def transform(raw, filename):
    if digest(raw) != SOURCE_SHA:
        raise ValueError('pinned original source differs')
    original = ast.parse(raw, filename=filename)
    tree = copy.deepcopy(original)
    gpu = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'gpu_run')
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    gates = [(i, n) for i, n in enumerate(gpu.body) if isinstance(n, ast.Expr)
             and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
             and n.value.func.id == 'require' and len(n.value.args) == 2
             and isinstance(n.value.args[1], ast.Constant) and n.value.args[1].value == MESSAGE]
    if len(gates) != 1:
        raise ValueError('unique original admission gate required')
    gate_index, gate = gates[0]
    if not isinstance(gate.value.args[0], ast.BoolOp) or not isinstance(gate.value.args[0].op, ast.And) or len(gate.value.args[0].values) != 6:
        raise ValueError('unchanged original six conjunctions required')
    report = ast.parse(REPORT).body
    stop = ast.parse('raise ValueError(' + repr(STOP) + ')').body[0]
    disposal = ast.parse('release(context, state)').body
    gpu.body[gate_index] = ast.copy_location(ast.Try(
        body=[*report, gate, stop], handlers=[], orelse=[], finalbody=disposal), gate)

    # These are the exact original statements, regrouped, not recreated from text.
    result_index = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == 'result' for t in n.targets))
    receipt_index = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                         and any(isinstance(t, ast.Name) and t.id == 'receipt' for t in n.targets))
    lifecycle = run.body[result_index+1:receipt_index]
    if len(lifecycle) != 12 or ast.unparse(lifecycle[0]) != "if args.phase != 'cpu':\n    release_scope(context)":
        raise ValueError('exact original post-run lifecycle required')
    call_count = sum(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                     and n.func.id == 'exit_rehash' for node in lifecycle for n in ast.walk(node))
    if call_count != 1:
        raise ValueError('one original uncached exit required')
    run.body[result_index:receipt_index] = [ast.copy_location(ast.Try(
        body=[run.body[result_index]], handlers=[], orelse=[], finalbody=lifecycle), run.body[result_index])]
    ast.fix_missing_locations(tree)

    restored = copy.deepcopy(tree)
    inverse_gpu = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == 'gpu_run')
    inverse_run = next(n for n in restored.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    inserted = inverse_gpu.body[gate_index]
    if len(inserted.body) != 3 or dump(inserted.body[1], True) != dump(gate, True):
        raise ValueError('original gate AST changed')
    inverse_gpu.body[gate_index] = inserted.body[1]
    regrouped = inverse_run.body[result_index]
    inverse_run.body[result_index:result_index+1] = [*regrouped.body, *regrouped.finalbody]
    if dump(restored, True) != dump(original, True):
        raise ValueError('exact complete source AST inverse failed')
    proof = {'schema': 'sfora-init-observation-ast-proof-v1', 'source_sha256': SOURCE_SHA,
             'source_ast_sha256': digest(dump(original, True).encode()),
             'diagnostic_ast_sha256': digest(dump(tree, True).encode()),
             'restored_ast_sha256': digest(dump(restored, True).encode()),
             'exact_inverse_including_attributes': True,
             'original_gate_ast_sha256': digest(dump(gate, True).encode()),
             'gate_lines': [gate.lineno, gate.end_lineno], 'original_lifecycle_statements': len(lifecycle),
             'original_exit_calls': call_count, 'injected_report_blocks': len(report),
             'native_execution': False, 'python_version': sys.version,
             'no_original_function_or_global_rebound': True}
    return tree, proof


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--native', action='store_true')
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--freeze', type=Path, required=True)
    parser.add_argument('--proof', type=Path, required=True)
    parser.add_argument('trainer_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_bytes())
    expected = {'schema': 'sfora-init-observation-freeze-v1', 'base_head': BASE,
                'trainer_sha256': SOURCE_SHA, 'test_sha256': TEST_SHA, 'execution_sha256': EXECUTION_SHA,
                'wrapper_sha256': digest(Path(__file__).read_bytes()),
                'contract_sha256': digest((args.freeze.parent / 'diagnostic-contract.json').read_bytes())}
    if freeze != expected:
        raise ValueError('separately frozen observer/contract/source pins differ')
    if not args.source.is_absolute() or args.source.resolve() != args.source or args.source.name != 'train_siglip2_identity_diversity.py':
        raise ValueError('canonical original source required')
    if digest(args.source.with_name('test_siglip2_identity_diversity.py').read_bytes()) != TEST_SHA:
        raise ValueError('original exact-two test differs')
    tree, proof = transform(args.source.read_bytes(), str(args.source))
    if args.native:
        if str(args.source) != REMOTE_SOURCE:
            raise ValueError('root original native source path required')
        execution = args.source.with_name('execution.json').read_bytes()
        if digest(execution) != EXECUTION_SHA or json.loads(execution) != {
                args.source.name: SOURCE_SHA, 'test_siglip2_identity_diversity.py': TEST_SHA}:
            raise ValueError('original exact-two execution closure differs')
        tail = args.trainer_args
        if not tail or tail[0] != '--':
            raise ValueError('original trainer CLI required after --')
        tail = tail[1:]
        # Exact original CLI validation and full authority admission still run.
        if len(tail) != 14 or tail[0:2] != ['--execution-sha256', EXECUTION_SHA] or tail[6:12] != [
                '--phase', 'mechanics', '--arm', 'control', '--seed', '179061'] or tail[12] != '--output':
            raise ValueError('bounded control179061 mechanics observation only')
        proof['native_execution'] = 'root-startup-pending; never qualification PASS'
    elif args.trainer_args:
        raise ValueError('--check never executes trainer CLI')
    with args.proof.open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(proof, sort_keys=True, indent=2) + '\n')
    print(json.dumps({'event': 'INIT_ADMISSION_SOURCE_PROOF_V1', **proof}, sort_keys=True), flush=True)
    if args.check:
        return
    sys.argv = [str(args.source), *tail]
    sys.path[0] = str(args.source.parent)
    module = ModuleType('__main__')
    module.__file__ = str(args.source)
    module.__package__ = None
    module.__spec__ = None
    sys.modules['__main__'] = module
    exec(compile(tree, str(args.source), 'exec'), vars(module))


if __name__ == '__main__':
    main()
