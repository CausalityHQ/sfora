#!/usr/bin/env python3
"""Exact-Torch recovery of the descriptive four-endpoint core census; native UNRUN.

The original scalar census failed exact replay by one FP32 ULP and stays
immutable (so does the paired candidate KILL). This driver replays the four
stored packed wires through the ORIGINAL Torch CPU scorer arithmetic: the
authenticated body of compare_inshop_sop_warmstart_100.packed_quality with only
an exactly invertible packed-input / pre-batch-budget / read-only-capture / no-aggregate adapter.
All 1734 queries x four endpoints must reproduce the accepted per-query R1/AP
bit-for-bit before any geometry is derived or published. No tolerance, scalar
surrogate, packing stub, int32 scorer, CUDA, image, checkpoint or held read.

CLI: python -B census_connected_core_errors_torch.py --authority FILE.path
--authority-sha256 SHA --output NEWFILE. FILE={path:canonical absolute regular
file,sha256:actual lowercase SHA256}. The authority keys are LAUNCH_KEYS; its
sources are exactly SOURCE_ROLES (all but command sit beside this driver) and
source_cpu is the existing admit_cpu UNIT descriptor, validated by the same log
and cgroup predicates WITHOUT admit_cpu's image/checkpoint rehash. Runtime
membership is the original source-CPU proof only (existing package_origins,
audit_origins, numerical_flags and cgroup primitives; no new whitelist).

The running driver is itself live code: after the helpers load, a genuine serving.Source over this module
(FILE, registry, functions, defaults, globals, literals) is checked before native import, before every endpoint
replay and in the full exit (where check_interpreter's original predicate is repeated uncached); LIMITS needs
exact builtin value types (False/900.0 cannot alias 0/900). The adapter AST declarations are bound to their
import-time dumps before any adaptation (a forged batch/capture statement is rejected before exec).

Engineering-only caps: 900s total, the last 120s reserved for the uncached exit
(no metric work starts inside it), 8GiB cgroup, zero swap, CUDA hidden, both
inherited lifetime locks. Root owns the single native job, log, exit status and
terminal cgroup evidence. No retry, no fallback. The original publish writes into a private staging directory; the
exclusive link promoting it is the commit point, bracketed by the final whole-process cap, and a failure after it
removes only that staged inode+content (never a foreign replacement). The published terminal still needs the
parent's exit/lock receipt.
"""
if not __debug__:
    raise SystemExit('optimized mode forbidden; original assertions required')

import argparse
import ast
from collections import Counter
import copy
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import resource
import stat
import struct
import sys
import tempfile
import time
from types import SimpleNamespace

STARTED = time.perf_counter()
LAUNCH_SCHEMA = 'connected-exact-torch-census-launch-v1'
RESULT_SCHEMA = 'sfora-connected-core-error-census-torch-v1'
LAUNCH_KEYS = {'schema', 'inputs', 'sources', 'native_sources', 'source_cpu', 'locks', 'resource_policy',
               'both_locks_held', 'candidate_status', 'qualification_eligible', 'state_reuse_eligible'}
LIMITS = {'whole_process_seconds': 900, 'exit_reserve_seconds': 120, 'host_bytes': 8 * 1024**3,
          'swap_bytes': 0, 'cuda_visible_devices': ''}
LIMIT_TYPES = {'whole_process_seconds': int, 'exit_reserve_seconds': int, 'host_bytes': int, 'swap_bytes': int,
               'cuda_visible_devices': str}
# role -> (basename beside the driver, accepted pin). Pins are the sources of the accepted
# full-selection score / source-CPU proof and the frozen original census, never a future guess.
SOURCE_ROLES = {
    'driver': ('census_connected_core_errors_torch.py', None),
    'test': ('test_connected_core_errors_torch.py', None),
    'command': (None, None),
    'census': ('census_connected_core_errors.py',
               '0fdc170019de1137ac9d82b55868ec8c8f8b21867d2a71c8c31a29f9a5b333d2'),
    'scorer': ('compare_inshop_sop_warmstart_100.py',
               '8827bed4bc90dfdcba36fd2a90bbd686b05a188f6356c1a5c9f965dea5080250'),
    'replay': ('evaluate_siglip2_prototype_residual.py',
               'e74b944dee26565029a4ec7e6ea2948d0f23e8564e5ee60ffd2f876453f529eb'),
    'serving': ('qualify_connected_serving_requests.py', None),
    'source_driver': ('qualify_siglip2_substrate_cpu.py',
                      'eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38'),
    'extract': ('extract_siglip2_vision_source.py',
                'a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d'),
    'quadratic_owner': ('train_siglip2_quadratic_readout.py',
                        '17cf2dac93a45aae3ba929ad5c36e73f4c33576eae9bb5f9dec670c272a1731f'),
    'original': ('train_siglip2_substrate_adaptation.py',
                 'a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543'),
    'initializer': ('export_siglip2_substrate_fit.py',
                    '163bee8b62bc90792ee848a4830a06e1416a3a34903ae4e1ba546dd93e9ebaa8'),
}
HELPERS = ('census', 'source_driver', 'extract', 'quadratic_owner', 'original', 'initializer')
UNIT_KEYS = {'proof', 'log', 'unit', 'invocation_id', 'service_seconds', 'native_peak_rss_kib', 'both_locks_held'}
CPU_PROOF_SHA = 'e8a1e02e27ed4e4726682c0a61e547d097912b98e3535ddb3da8c1bb4e144cbf'
CPU_EXECUTION_SHA = '3eabe74c62aadafae5441bcded6e85d489f17409ce3b77116f61ab75c4014f2b'
NATIVE_SOURCES_SHA = '8cd70a9b706e0a8ff6dea3092310083ba3f7cd2fffd58cac5b8f6e3842a7ad3b'
SCORER_AST = '717489188a008ceba1b930d9e7dff32b90cd5347302835538eb171fe23f4b33e'
NATIVE = {'torch', 'numpy', 'PIL', 'sfora', 'transformers', 'torchvision', 'safetensors'}
WIDTH = 81
BATCHES = (128,) * 13 + (70,)
FALSIFIER = {'endpoint': 'candidate-179061', 'query_index': 747, 'expected_ap': 0.290910005569458}
PROOF_TRUE = ('pass', 'source_qualified', 'reload_exact', 'exit_rehash_pass', 'constructor_rng_preserved')
PROOF_FALSE = ('model_qualified', 'initializer_qualified', 'training_qualified', 'quality_qualified',
               'quality_read', 'gradients_created', 'optimizer_created')
# The pinned original statements the adapter may touch; everything else must match verbatim.
VALUES_ARG = ast.parse('def f(values: np.ndarray): pass').body[0].args.args[0]
PACK_STATEMENT = ast.parse('packed = pack_int8_unit_embeddings(torch.from_numpy(values.copy()))').body[0]
CAPTURE_STATEMENT = ast.parse('census_capture(start, rows, scores)').body[0]
BATCH_STATEMENT = ast.parse('census_batch(start)').body[0]
AGGREGATES = ast.parse("{'recall_at_1': float(np.mean(hits)), 'map_at_r': float(np.mean(aps))}", mode='eval').body
# The four declarations are mutable ASTs and adapt/unadapt compare the adapter against themselves, so their dumps are
# frozen here, at the authenticated source's import, as immutable strings (a forged capture would invert cleanly).
ADAPTER_DUMPS = tuple(ast.dump(node, include_attributes=False)
                      for node in (VALUES_ARG, PACK_STATEMENT, CAPTURE_STATEMENT, BATCH_STATEMENT, AGGREGATES))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_policy(policy):
    require(type(policy) is dict and policy == LIMITS and {k: type(v) for k, v in policy.items()} == LIMIT_TYPES,
            'exact builtin-typed resource policy required')


def file_fact(fact):
    require(type(fact) is dict and fact.keys() == {'path', 'sha256'} and type(fact['path']) is str and
            type(fact['sha256']) is str and re.fullmatch('[0-9a-f]{64}', fact['sha256']), 'actual exact FILE required')
    path = Path(fact['path'])
    require(path.is_absolute() and str(path) == fact['path'] and path.resolve() == path and
            path.is_file() and not path.is_symlink(), 'canonical regular FILE required')
    return path


def file_sha(path):
    """Stream the current bytes; reject a file that changes while it is read."""
    digest, before = hashlib.sha256(), os.stat(path)
    with open(path, 'rb') as stream:
        while block := stream.read(1024**2):
            digest.update(block)
        after = os.fstat(stream.fileno())
    key = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    require(all(getattr(before, k) == getattr(after, k) for k in key), f'FILE changed while reading: {path}')
    return digest.hexdigest()


def read_file(fact, guards):
    path = file_fact(fact)
    require(path.stat().st_size <= 64 * 1024**2, 'metadata/source FILE exceeds 64MiB')
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == fact['sha256'], f'current FILE SHA256 differs: {path}')
    require(guards.setdefault(str(path), fact['sha256']) == fact['sha256'], 'conflicting FILE authority')
    return raw


def strict_json(raw):
    def pairs(items):
        value = dict(items)
        require(len(value) == len(items), 'duplicate JSON key')
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda v: require(False, 'nonfinite JSON'))


def read_json(fact, guards):
    return strict_json(read_file(fact, guards))


def rehash_files(guards):
    for path, digest in tuple(guards.items()):
        file_fact({'path': path, 'sha256': digest})
        require(file_sha(path) == digest, f'exit authority SHA256 differs: {path}')


class Budget:
    """Whole-process wall cap; metric work never starts inside the exit reserve."""
    def __init__(self, started=STARTED, clock=time.perf_counter):
        self.started, self.clock = started, clock

    def check(self, reserve=True):
        limit = LIMITS['whole_process_seconds'] - (LIMITS['exit_reserve_seconds'] if reserve else 0)
        require(self.clock() - self.started < limit,
                'exit reserve reached; no further body work' if reserve else 'whole-process cap reached')


def cleanup_error(error, callbacks):
    failures = []
    for callback in callbacks:
        try:
            callback()
        except BaseException as failure:
            failures.append(failure)
    if error is not None:
        for failure in failures:
            error.add_note('cleanup: ' + repr(failure))
        raise error
    if failures:
        for failure in failures[1:]:
            failures[0].add_note('cleanup: ' + repr(failure))
        raise failures[0]


def read_authority(path, digest):
    """Strict root authority: exact keys, exact FILE roles, accepted pins, current bytes."""
    guards = {}
    authority = read_json({'path': str(path), 'sha256': digest}, guards)
    require(type(authority) is dict and authority.keys() == LAUNCH_KEYS and authority['schema'] == LAUNCH_SCHEMA and
            authority['resource_policy'] == LIMITS and authority['both_locks_held'] is True and
            authority['candidate_status'] == 'KILL' and authority['qualification_eligible'] is False and
            authority['state_reuse_eligible'] is False, 'exact diagnostic-only KILL authority and limits required')
    exact_policy(authority['resource_policy'])
    sources, here = authority['sources'], Path(__file__).absolute().parent
    require(type(sources) is dict and sources.keys() == SOURCE_ROLES.keys(), 'exact source FILE roles required')
    for role, (name, pin) in SOURCE_ROLES.items():
        read_file(sources[role], guards)
        require(name is None or Path(sources[role]['path']) == here / name, f'{role} FILE must be {name} beside the driver')
        require(pin is None or sources[role]['sha256'] == pin, f'{role} FILE differs from its accepted pin')
    require(Path(sources['driver']['path']) == Path(__file__).absolute(), 'running driver FILE required')
    read_file(authority['inputs'], guards)
    read_file(authority['native_sources'], guards)
    require(authority['native_sources']['sha256'] == NATIVE_SOURCES_SHA, 'original native sources.json pin differs')
    return authority, guards


def bootstrap_serving(fact, guards):
    """Load the serving helper from authenticated bytes so its own Source can load the rest."""
    raw, name = read_file(fact, guards), '_exact_census_serving'
    require(name not in sys.modules, 'serving helper namespace already owned')
    module = importlib.util.module_from_spec(importlib.util.spec_from_file_location(name, fact['path']))
    sys.modules[name] = module
    try:
        exec(compile(raw, fact['path'], 'exec', dont_inherit=True), vars(module))
        return module, module.Source(module, fact)
    except BaseException:
        sys.modules.pop(name, None)
        raise


def load_helpers(authority, guards):
    """Authenticated stdlib-only helper modules, each guarded by the serving Source."""
    serving, serving_source = bootstrap_serving(authority['sources']['serving'], guards)
    owned, modules = [serving_source], {}
    try:
        for role in HELPERS:
            owned.append(serving.Source.load(authority['sources'][role]))
            modules[role] = owned[-1].module
    except BaseException:
        close_sources(owned)
        raise
    return serving, owned, modules


def own_source(serving, authority):
    """Authenticate the running driver (never part of the owned set, so never removed from sys.modules)."""
    return serving.Source(sys.modules[__name__], authority['sources']['driver'])


def check_own(source):
    source.check()
    exact_policy(LIMITS)


def guard(ctx):
    ctx.locks.check()
    for source in ctx.owned:
        source.check()
    check_own(ctx.own)


def close_sources(owned):
    def remove(source):
        require(sys.modules.get(source.module.__name__) is source.module, 'owned source registry changed')
        del sys.modules[source.module.__name__]
    cleanup_error(None, [lambda s=s: remove(s) for s in reversed(owned)])
    owned.clear()


def check_interpreter(proof, extract):
    invocation, python = proof['invocation'], Path(sys.executable).resolve()
    require(str(python) == invocation['python'] and extract.sha(python) == invocation['python_sha256'] and
            sys.version == invocation['python_version'], 'original admitted interpreter required')


def check_accepted_environment(receipt, proof):
    """The accepted score ran in this exact CPU environment; its origins are a subset of the proof's."""
    invocation, original = receipt['invocation'], proof['invocation']
    require(all(invocation[k] == original[k] for k in ('python', 'python_sha256', 'python_version')) and
            invocation['cuda_visible_devices'] == '' and invocation['optimize'] == 0 and
            receipt['numerical_flags'] == proof['numerical_flags'] and
            receipt['origins']['packages'] == proof['origins']['packages'], 'accepted score environment differs')
    ours, theirs = proof['origins'], receipt['origins']
    require(all(ours['files'].get(p) == h for p, h in theirs['files'].items()) and
            all(ours['modules'].get(n) == p for n, p in theirs['modules'].items()) and
            set(theirs['native_files']) <= set(ours['native_files']),
            'accepted score origins exceed the original source-CPU authority')


def admit_source_cpu(descriptor, guards, modules):
    """Existing admit_cpu UNIT predicates for the proof, log and cgroups; never its input rehash."""
    source, initializer = modules['source_driver'], modules['initializer']
    require(type(descriptor) is dict and descriptor.keys() == UNIT_KEYS and descriptor['both_locks_held'] is True,
            'source CPU descriptor/locks differ')
    require(descriptor['proof']['sha256'] == CPU_PROOF_SHA, 'original source-CPU proof pin differs')
    proof = read_json(descriptor['proof'], guards)
    require(proof['schema'] == source.SCHEMA and proof['arm'] == 'so400' and proof['updates'] == 0 and
            all(proof[k] is True for k in PROOF_TRUE) and all(proof[k] is False for k in PROOF_FALSE),
            'actual source-only CPU proof required')
    require(proof['execution_sha256'] == CPU_EXECUTION_SHA and
            all(proof['code'][SOURCE_ROLES[r][0]] == SOURCE_ROLES[r][1] for r in ('extract', 'source_driver')),
            'source CPU authority/source binding differs')
    seconds, rss = descriptor['service_seconds'], descriptor['native_peak_rss_kib']
    require(proof['resource_policy'] == source.POLICY and type(seconds) in (int, float) and type(rss) is int and
            0 < proof['wall_seconds'] <= seconds <= 120 and 0 < proof['process_peak_rss_kib'] <= rss <= 8 * 1024**2,
            'source CPU service/RSS/caps differ')
    invocation = proof['invocation']
    require(re.fullmatch('[0-9a-f]{32}', descriptor['invocation_id']) and
            re.fullmatch('[A-Za-z0-9_.@-]+', descriptor['unit']) and
            invocation['invocation_id'] == descriptor['invocation_id'] and
            invocation['cuda_visible_devices'] == '' and invocation['optimize'] == 0,
            'source CPU original invocation differs')
    require(all(proof['input_guards'].get(p) == h for p, h in proof['origins']['files'].items()),
            'source CPU origin guard differs')
    lines = read_file(descriptor['log'], guards).decode().splitlines()
    required = [f"Running as unit: {descriptor['unit']}.service; invocation ID: {descriptor['invocation_id']}",
                '\tExit status: 0', 'Finished with result: success',
                'Main processes terminated with: code=exited/status=0', '\tSwaps: 0', 'Memory swap peak: 0B',
                f'Service runtime: {seconds}s', f'\tMaximum resident set size (kbytes): {rss}']
    require(all(lines.count(line) == 1 for line in required), 'source CPU original normal-exit log differs')
    footers = [strict_json(line[len('FINAL_CGROUP '):]) for line in lines if line.startswith('FINAL_CGROUP ')]
    require(len(footers) == 1 and footers[0]['invocation_id'] == descriptor['invocation_id'],
            'source CPU final cgroup footer differs')
    final = footers[0]
    for value in (proof['cgroup_before'], proof['cgroup_after'], final):
        initializer.admit_cgroup(value, descriptor['unit'])
        require(value['path'] == final['path'], 'source CPU cgroup path changed')
    peaks = [int(v['values']['memory.peak']) for v in (proof['cgroup_before'], proof['cgroup_after'], final)]
    require(peaks == sorted(peaks), 'source CPU complete peak differs')
    return proof


def check_packages(modules, authority, proof, guards):
    """Existing pre-import package discovery; the package __init__ bytes must be the proof's."""
    record = read_json(authority['native_sources'], guards)
    require(proof['input_guards'].get(authority['native_sources']['path']) == NATIVE_SOURCES_SHA and
            record['schema'] == 'paired-native256-vision-sources-v1', 'original native sources.json role differs')
    packages = modules['source_driver'].package_origins(
        {'sources': record, 'extract': modules['extract'], 'guards': guards})
    require(packages == proof['origins']['packages'], 'installed package origins differ from the original proof')
    for package in packages.values():
        require(proof['origins']['files'].get(package['origin']) == guards.get(package['origin']),
                'installed package __init__ bytes differ from the original proof')


def make_audit(modules, proof, guards):
    """Narrow original audit_origins context: the CPU proof is the sole authority (no supplement)."""
    context = {'source_driver': modules['source_driver'], 'extract': modules['extract'],
               'prior': {'guards': guards}, 'guards': guards,
               'selected': {'packages': proof['origins']['packages'], 'source_cpu': proof},
               'warm_record': {'origins': proof['origins']}}
    def audit():
        modules['quadratic_owner'].audit_origins(context, initial=True, admission=modules['original'].FlatAdmission())
        return {k: len(context['origins'][k]) for k in ('modules', 'files', 'native_files')}
    return audit


def dump(node):
    return ast.dump(node, include_attributes=False)


def scorer_function(raw):
    """The unchanged original packed scorer, authenticated by its function AST."""
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'packed_quality']
    tree = ast.Module(body=nodes, type_ignores=[])
    require(len(nodes) == 1 and hashlib.sha256(dump(tree).encode()).hexdigest() == SCORER_AST,
            'unchanged original packed scorer AST differs')
    return nodes[0]


def batch_loop(function):
    loops = [n for n in function.body if isinstance(n, ast.For)]
    require(len(loops) == 1 and isinstance(loops[0].target, ast.Name) and loops[0].target.id == 'start',
            'single original 128-query batch loop required')
    return loops[0]


def check_declarations():
    declarations = (VALUES_ARG, PACK_STATEMENT, CAPTURE_STATEMENT, BATCH_STATEMENT, AGGREGATES)
    require(tuple(dump(n) for n in declarations) == ADAPTER_DUMPS,
            'adapter AST declarations differ from the authenticated driver source')


def adapt(original):
    """Packed input, pre-batch budget hook, read-only capture, no np aggregates; all else verbatim."""
    check_declarations()
    function = copy.deepcopy(original)
    require(dump(function.args.args[0]) == dump(VALUES_ARG) and dump(function.body[1]) == dump(PACK_STATEMENT),
            'original packing statement differs')
    function.args.args[0] = ast.arg(arg='packed')
    del function.body[1]
    loop = batch_loop(function)
    loop.body.insert(0, copy.deepcopy(BATCH_STATEMENT))
    loop.body.append(copy.deepcopy(CAPTURE_STATEMENT))
    result = function.body[-1]
    require(isinstance(result, ast.Return) and isinstance(result.value, ast.Dict) and
            dump(ast.Dict(keys=result.value.keys[:2], values=result.value.values[:2])) == dump(AGGREGATES),
            'original aggregate return differs')
    result.value.keys, result.value.values = result.value.keys[2:], result.value.values[2:]
    return ast.fix_missing_locations(function)


def unadapt(adapted):
    check_declarations()
    function = copy.deepcopy(adapted)
    function.args.args[0] = copy.deepcopy(VALUES_ARG)
    function.body.insert(1, copy.deepcopy(PACK_STATEMENT))
    loop = batch_loop(function)
    require(dump(loop.body[-1]) == dump(CAPTURE_STATEMENT), 'capture is not the last batch statement')
    loop.body.pop()
    require(dump(loop.body[0]) == dump(BATCH_STATEMENT), 'budget hook is not the first batch statement')
    loop.body.pop(0)
    result = function.body[-1]
    result.value.keys = copy.deepcopy(AGGREGATES.keys) + result.value.keys
    result.value.values = copy.deepcopy(AGGREGATES.values) + result.value.values
    return function


def verify_adapter(original, adapted):
    require(dump(unadapt(adapted)) == dump(original) and dump(adapt(original)) == dump(adapted),
            'packed adapter is not the exact inverse of the pinned original scorer')


def compile_scorer(torch, raw, path, capture, batch):
    original = scorer_function(raw)
    adapted = adapt(original)
    verify_adapter(original, adapted)
    namespace = {'torch': torch, 'census_capture': capture, 'census_batch': batch}
    exec(compile(ast.Module(body=[adapted], type_ignores=[]), path, 'exec', dont_inherit=True), namespace)
    return namespace['packed_quality']


def packed_input(torch, rows):
    """Decoded wire as the original PackedInt8Embeddings fields; .float() promotion stays in the scorer."""
    packed = SimpleNamespace(codes=torch.tensor([r[0] for r in rows], dtype=torch.int8),
                             inverse_norms=torch.tensor([r[1] for r in rows], dtype=torch.float16))
    require([tuple(c) for c in packed.codes.tolist()] == [tuple(r[0]) for r in rows] and
            packed.inverse_norms.tolist() == [r[1] for r in rows], 'packed wire tensors differ from the stored wire')
    return packed


def make_capture(core, retained, budget):
    """Read-only: keep only the 44 core score rows (full gallery) for later geometry."""
    def capture(start, rows, scores):
        budget.check()
        for offset in range(len(rows)):
            if start + offset in core:
                require(start + offset not in retained, 'core score row captured twice')
                retained[start + offset] = [float(v) for v in scores[offset].tolist()]
    return capture


def replay_exact(results, expected):
    """All endpoints, all queries, bit-exact or the whole census is rejected."""
    bad = {}
    for key, want in expected.items():
        got = results[key]
        require(got.keys() == {'per_query_r1', 'per_query_ap'} and all(len(got[k]) == len(want[k]) for k in got),
                f'{key}: replay arrays have the wrong shape')
        bad[key] = [(k, i, a, e) for k in got for i, (a, e) in enumerate(zip(got[k], want[k])) if a != e]
    if any(bad.values()):
        shown = '; '.join(f'{key}: {len(rows)} differ, first (array, query index, actual, expected) {rows[:3]}'
                          for key, rows in bad.items() if rows)
        raise ValueError('exact original per-query replay differs: ' + shown)


def describe_scores(scores, labels, query, gallery, replayed):
    require(len(scores) == len(gallery) and all(math.isfinite(v) for v in scores), 'complete finite score row required')
    order = sorted(range(len(gallery)), key=lambda i: -scores[i])
    positive = [labels[i] == labels[query] for i in gallery]
    relevant = sum(positive)
    require(0 < relevant <= WIDTH <= len(gallery), 'positive/AP width inventory differs')
    best, impostor = next(i for i in order if positive[i]), next((i for i in order if not positive[i]), None)
    require(impostor is not None, 'top impostor inventory absent')
    require(int(positive[order[0]]) == replayed['per_query_r1'], 'stable order top-1 differs from exact replayed R1')
    return {'best_positive_rank': order.index(best) + 1, 'best_positive_gallery_index': best,
            'best_positive_panel_ordinal': gallery[best], 'top_impostor_rank': order.index(impostor) + 1,
            'top_impostor_gallery_index': impostor, 'top_impostor_panel_ordinal': gallery[impostor],
            'best_positive_score': scores[best], 'top_impostor_score': scores[impostor],
            'positive_minus_impostor_margin': scores[best] - scores[impostor],
            'positive_count': relevant, **replayed}


def score_digest(retained, core):
    digest = hashlib.sha256()
    for index in core:
        digest.update(struct.pack('<q', index) + struct.pack(f'<{len(retained[index])}f', *retained[index]))
    return digest.hexdigest()


def build_census(state, retained, results, scorer_facts):
    """Geometry from retained native rows; only reachable after complete exact replay."""
    census = state['census']
    panel = state['partition']['panels']['selection']
    query, gallery, rows, core = panel['query'], panel['gallery'], state['rows'], state['core']
    labels = [r['product'] for r in rows]
    queries = []
    for index in core:
        endpoints = {}
        for seed, arm in census.ENDPOINTS:
            key = f'{arm}-{seed}'
            replayed = {k: results[key][k][index] for k in ('per_query_r1', 'per_query_ap')}
            values = describe_scores(retained[key][index], labels, query[index], gallery, replayed)
            values['best_positive'] = rows[values['best_positive_panel_ordinal']]
            values['top_impostor'] = rows[values['top_impostor_panel_ordinal']]
            endpoints[key] = values
        queries.append({'query_index': index, 'query': rows[query[index]], 'endpoints': endpoints})
    core_counts = Counter(labels[query[i]] for i in core)
    query_counts, gallery_counts = Counter(labels[i] for i in query), Counter(labels[i] for i in gallery)
    products = [{'product': p, 'core_queries': n, 'panel_queries': query_counts[p], 'panel_gallery': gallery_counts[p]}
                for p, n in sorted(core_counts.items())]
    focus = results[FALSIFIER['endpoint']]['per_query_ap'][FALSIFIER['query_index']]
    return {'schema': RESULT_SCHEMA,
            'scope': 'descriptive previously exposed TRAIN-selection four-endpoint wrong intersection; '
                     'exact original Torch CPU scorer arithmetic',
            'original_decision': state['receipt']['decision'], 'candidate_status': 'KILL unchanged',
            'scientific_gate_changed': False, 'qualification_eligible': False, 'state_reuse_eligible': False,
            'core_query_indices': core, 'queries': queries, 'per_product_core_counts': products,
            'scored_queries': len(queries), 'gallery_rows': len(gallery), 'endpoints': len(census.ENDPOINTS),
            'scored_pairs': len(queries) * len(gallery) * len(census.ENDPOINTS),
            'replay': {'queries': len(query), 'endpoints': len(census.ENDPOINTS),
                       'per_query_pairs': len(query) * len(census.ENDPOINTS), 'exact': True, 'tolerance': None,
                       'batch_sizes': list(BATCHES), 'width': WIDTH,
                       'falsifier': {**FALSIFIER, 'actual_ap': focus, 'exact': focus == FALSIFIER['expected_ap']}},
            'core_score_rows_sha256': {key: score_digest(rows_, core) for key, rows_ in retained.items()},
            'scorer': scorer_facts}


def admit(ctx):
    """Everything up to and including the genuine origin audit; native code runs only after admission."""
    authority, guards = ctx.authority, ctx.guards
    serving, ctx.owned, modules = load_helpers(authority, guards)
    ctx.modules = modules
    ctx.own = own_source(serving, authority)
    ctx.locks = serving.Locks(authority['locks'])
    inputs = authority['inputs']
    ctx.state = state = modules['census'].admit_inputs(inputs['path'], inputs['sha256'])
    state['census'] = modules['census']
    ctx.proof = proof = admit_source_cpu(authority['source_cpu'], guards, modules)
    check_accepted_environment(state['receipt'], proof)
    check_interpreter(proof, modules['extract'])
    require(os.environ['INVOCATION_ID'] != proof['invocation']['invocation_id'], 'fresh enclosing unit required')
    ctx.before = modules['source_driver'].cgroup_memory()
    ctx.unit = Path(ctx.before['path']).name.removesuffix('.service')
    modules['initializer'].admit_cgroup(ctx.before, ctx.unit)
    check_packages(modules, authority, proof, guards)
    ctx.budget.check()
    ctx.flags = copy.deepcopy(proof['numerical_flags'])
    check_own(ctx.own)
    import torch
    ctx.torch = torch
    require(not torch.cuda.is_initialized(), 'CUDA initialization must not occur')
    torch.set_num_threads(ctx.flags['threads'])
    if torch.get_num_interop_threads() != ctx.flags['interop_threads']:
        torch.set_num_interop_threads(ctx.flags['interop_threads'])
    require(modules['source_driver'].numerical_flags() == ctx.flags, 'original numerical flags differ')
    ctx.audit = make_audit(modules, proof, guards)
    ctx.origins = {'initial': ctx.audit()}


def replay(ctx):
    """Original scorer arithmetic on all four stored wires; complete exact replay before geometry."""
    torch, state, budget, census = ctx.torch, ctx.state, ctx.budget, ctx.state['census']
    scorer = ctx.authority['sources']['scorer']
    raw = read_file(scorer, ctx.guards)
    panel = state['partition']['panels']['selection']
    query, gallery, labels = panel['query'], panel['gallery'], tuple(r['product'] for r in state['rows'])
    gallery_counts = Counter(labels[i] for i in gallery)
    require(max(gallery_counts[labels[i]] for i in query) == WIDTH, 'original panel-wide AP width differs')
    core, results, retained = set(state['core']), {}, {}
    for seed, arm in census.ENDPOINTS:
        key = f'{arm}-{seed}'
        budget.check()
        ctx.guard()
        retained[key] = {}
        fn = compile_scorer(torch, raw, scorer['path'], make_capture(core, retained[key], budget),
                            lambda start: budget.check())
        packed = packed_input(torch, state['wires'][key])
        budget.check()
        results[key] = fn(packed, labels, query, gallery, device=torch.device('cpu'))
        require(retained[key].keys() == core and all(len(r) == len(gallery) for r in retained[key].values()),
                'core score rows were not all captured')
    expected = {f'{a}-{s}': state['receipt']['quality'][s][a] for s, a in census.ENDPOINTS}
    require(expected[FALSIFIER['endpoint']]['per_query_ap'][FALSIFIER['query_index']] == FALSIFIER['expected_ap'],
            'accepted falsifier AP differs from the frozen plan')
    replay_exact(results, expected)
    ctx.budget.check()
    facts = {'source_sha256': scorer['sha256'], 'ast_sha256': SCORER_AST, 'device': 'cpu', 'width': WIDTH,
             'torch_version': torch.__version__, 'numerical_flags': ctx.flags,
             'adapter': 'packed input, read-only capture, no np aggregates; exact AST inverse verified'}
    ctx.payload = build_census(state, retained, results, facts)


def exit_checks(ctx):
    """Genuine uncached exit: fresh origins, every guard, live sources, locks, cgroup, then disposal."""
    def origins():
        ctx.origins['final'] = ctx.audit()

    def state_after():
        ctx.budget.check(reserve=False)
        torch = ctx.torch
        require(not torch.cuda.is_initialized(), 'final CUDA initialization differs')
        require(ctx.modules['source_driver'].numerical_flags() == ctx.flags, 'final numerical flags changed')
        after = ctx.modules['source_driver'].cgroup_memory()
        ctx.modules['initializer'].admit_cgroup(after, ctx.unit)
        require(after['path'] == ctx.before['path'], 'enclosing diagnostic unit changed')
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        require(0 < peak <= LIMITS['host_bytes'] // 1024, 'process peak RSS cap differs')
        ctx.after, ctx.peak = after, peak

    callbacks = []
    if ctx.audit is not None:
        callbacks.append(origins)
    callbacks.append(lambda: rehash_files(ctx.guards))
    if ctx.state is not None:
        callbacks.append(lambda: ctx.state['census'].rehash(ctx.state['guards']))
    if ctx.locks is not None:
        callbacks.append(ctx.locks.check)
    callbacks.append(lambda: [source.check() for source in ctx.owned])
    if ctx.own is not None:
        callbacks.append(lambda: check_own(ctx.own))
    if ctx.proof is not None:
        callbacks.append(lambda: check_interpreter(ctx.proof, ctx.modules['extract']))
    if ctx.torch is not None:
        callbacks.append(state_after)
    callbacks += [gc.collect, lambda: close_sources(ctx.owned)]
    return callbacks


def owned_output(output):
    """Stat identity and content hash of the regular file at output (a symlink or other file is never ours)."""
    info = os.lstat(output)
    require(stat.S_ISREG(info.st_mode), 'output is not a regular file; foreign file left in place')
    return ((info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns),
            hashlib.sha256(Path(output).read_bytes()).hexdigest())


def publish_census(census, output, payload, guards, budget):
    """Original publish into a private staging directory, then an exclusive-link promotion bracketed by the final cap.

    The owner is captured from the staged file before promotion, so a failure afterwards removes the final output
    only while it is still that very inode and content; a foreign replacement is never adopted or removed."""
    with tempfile.TemporaryDirectory(dir=Path(output).parent) as staging:
        staged = str(Path(staging) / Path(output).name)
        census.publish(staged, payload, guards)
        owner = owned_output(staged)
        budget.check(reserve=False)
        try:
            os.link(staged, output, follow_symlinks=False)
            budget.check(reserve=False)
        except BaseException as error:
            try:
                require(owned_output(output) == owner, 'output is not the staged file; foreign file left in place')
                os.unlink(output)
            except BaseException as failure:
                error.add_note('output removal: ' + repr(failure))
            raise


def finalize(args, ctx):
    state, payload = ctx.state, ctx.payload
    payload.update(
        full_uncached_exit_pass=True, exit_rehash_pass=True, cleanup_pass=True,
        authority={'path': str(args.authority), 'sha256': args.authority_sha256}, launch=ctx.authority,
        resource_policy=LIMITS, cgroup_before=ctx.before, cgroup_after=ctx.after, origins={
            **ctx.origins, 'authority': 'original source-CPU proof only; no vendor supplement or new whitelist'},
        invocation={'argv': sys.argv, 'python': str(Path(sys.executable).resolve()),
                    'python_sha256': ctx.proof['invocation']['python_sha256'], 'python_version': sys.version,
                    'pid': os.getpid(), 'invocation_id': os.environ['INVOCATION_ID'],
                    'optimize': sys.flags.optimize, 'cuda_visible_devices': os.environ['CUDA_VISIBLE_DEVICES'],
                    'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')},
        resources={'wall_seconds_before_publication': ctx.budget.clock() - ctx.budget.started,
                   'process_peak_rss_kib': ctx.peak},
        provenance={'inputs': state['guards'], 'fetch_record': state['record'], 'authority_inputs': dict(ctx.guards),
                    'ordered_selection_mapping_sha256': hashlib.sha256(json.dumps(
                        state['rows'], sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
                    'images_read': 0, 'model_executions': 0},
        terminal_exit_and_both_locks_require_parent_receipt=True)


def run(args):
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None,
            'unoptimized -B unprofiled startup required')
    require(not any(n.split('.')[0] in NATIVE for n in sys.modules), 'native imports must follow explicit admission')
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == '', 'CUDA must be explicitly hidden')
    require(re.fullmatch('[0-9a-f]{32}', os.environ.get('INVOCATION_ID', '')), 'enclosing systemd invocation required')
    output = Path(args.output)
    require(output.is_absolute() and str(output) == args.output and output.parent.resolve() == output.parent and
            not os.path.lexists(output), 'exclusive canonical NEWFILE required')
    ctx = SimpleNamespace(budget=Budget(), owned=[], own=None, modules=None, locks=None, state=None, proof=None,
                          torch=None, audit=None, payload=None, origins={}, flags=None, before=None, unit=None)
    ctx.authority, ctx.guards = read_authority(args.authority, args.authority_sha256)
    ctx.guard = lambda: guard(ctx)
    error = None
    try:
        admit(ctx)
        replay(ctx)
    except BaseException as failure:
        error = failure
    cleanup_error(error, exit_checks(ctx))
    finalize(args, ctx)
    publish_census(ctx.state['census'], args.output, ctx.payload, ctx.state['guards'], ctx.budget)
    return ctx.payload


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--authority', required=True)
    p.add_argument('--authority-sha256', required=True)
    p.add_argument('--output', required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        payload = run(args)
    except Exception as error:
        print(f'exact Torch core census rejected: {type(error).__name__}: {error}', file=sys.stderr)
        return 1
    print(json.dumps({'output': args.output, 'sha256': hashlib.sha256(Path(args.output).read_bytes()).hexdigest(),
                      'wall_seconds': time.perf_counter() - STARTED,
                      'exact_replay_pairs': payload['replay']['per_query_pairs']}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
