#!/usr/bin/env python3
"""Source-only FS asymmetric cache-gallery falsifier on the old MLP seed179061 endpoint; native gate UNRUN.

Root freezes exactly this file and test_connected_asymmetric_cache.py in execution.json, then supplies
connected-asymmetric-cache-launch-v1. CLI (absolute paths, python -B):
--execution-sha256 SHA --authority FILE --authority-sha256 SHA --output NEWDIR.
FILE={path:canonical absolute regular file,sha256:actual lowercase SHA256}. LAUNCH_KEYS are exact; the
fixed historical pins below are existing committed facts, every other FILE is root supplied and checked.

Method: ONE zero-training FS cell. Query F is the accepted archived candidate-179061 live export wire;
gallery S is the original FIT-cache selection-gallery rows (CUDA FP32, original B32/B19 batches) run through
the SAME candidate head/A/means/C/mu. FF is the accepted symmetric KILL replayed unchanged. No SS/SF cell,
no training, no VAL/official read, no causal-freshness claim, no decision key; the root replays FIRST decide().

Reuse, nothing replaced: the pinned v3 diagnostic (prepare, Sources, terminal_admission, fresh_values,
scorer, stale_values, control_byte_differences, compose, output_bytes, exact_wire, final_state), the pinned
v5 observer (capture_inference_outputs, readout, check_capture_ast, capture_workspace_owner), and the exact
MLP evaluator chain that already ran natively in qualify_connected_control_serving.run: unchanged
load_evaluator_source, evaluator.authority(export control179061 full launch), CombinedAuthority over the
genuine training_context with the actual Cutile authority, install, native_start. The independent candidate
readout is that evaluator's fullfeature_oracle(context,state,features) (AST identical to the probe copy).

Order is fixed: pure admission, native start, ALL archived replays, cache read, control tap (only the frozen
six query-tail rows may differ), accepted batch-of-6 replay (only those six images), candidate readout and
oracle, FS build and persisted read-back, THEN the first Cutile mapping: native ties and top10 parity against
independent persisted-wire arithmetic, THEN quality. v3 origin_audit/observer exact_four run only before the
Cutile mapping; the exit is the CombinedAuthority H-union-S exact audit through api.evaluator_exit.
700s/8GiB/zero swap+events/CUDA<10GB/both locks/uncached full exit. Serving and index cost are UNMEASURED.
"""
if not __debug__:
    raise SystemExit('optimized mode forbidden; original assertions required')

import argparse
import ast
import copy
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import struct
import sys
import time
import traceback
from types import FunctionType, SimpleNamespace

STARTED = time.perf_counter()
FILES = {'diagnose_connected_asymmetric_cache.py', 'test_connected_asymmetric_cache.py'}
SCHEMA = 'connected-asymmetric-cache-launch-v1'
RECEIPT_SCHEMA = 'connected-asymmetric-cache-diagnostic-v1'
LAUNCH_KEYS = {'schema', 'execution_sha256', 'stage', 'seeds', 'output', 'historical', 'observer', 'evaluator',
               'evaluation_authority', 'native', 'serving', 'controls', 'tail_oracle', 'resource_policy', 'locks',
               'both_locks_held', 'candidate_status', 'qualification_eligible', 'state_reuse_eligible'}
LIMITS = {'seconds': 700, 'host_bytes': 8*1024**3, 'swap_bytes': 0, 'cuda_visible_devices': '0',
          'cuda_allocated_bytes_exclusive': 10_000_000_000}
NATIVE_PACKAGES = {'torch', 'numpy', 'PIL', 'transformers', 'torchvision', 'safetensors', 'sfora'}
SEED = 179061
KEY = 'control-179061'
CANDIDATE = 'candidate-179061'
OUTPUT_NAMES = ('raw', 'unit', 'codes', 'inverse_norms', 'wire')
WIDTHS = {'raw': 512, 'unit': 512, 'codes': 128, 'inverse_norms': 2, 'wire': 130}
PANEL_ROWS = 3449
TOP_K = 10
# Accepted v3 diagnostic and its original launch (same pins as the v5 observer).
V3 = {'source': '3b756bf94da91f1e66b03db1e25c96a21105ff784cac4833e40e70122de6a434',
      'execution': '6ef0482ddd758a6a5fb76cc9911c57c164fa3cec33ec71db9ad3baa8245d7cf4',
      'authority': '179275e426dd7b2c2f0ab0c8d7323b55fbdf5c11bbb12d1f96056d53c0b2941d'}
# Frozen v5 observer (git a20c1f35), not the later repository revision.
V5 = {'source': '62e69a04bf4df55df5185366231d0237b5991c912f2a0e23e6af31f41c5dd24f',
      'test': '6d08228ab87cc1cf9aba5159a4b72ad671087462ef335c03c659d64b0a750c8c'}
V5_NAMES = {'source': 'observe_connected_control_batch_execution.py', 'test': 'test_connected_control_batch_execution.py'}
# Accepted old MLP evaluator (full score and full control export) and its control export launch.
EVALUATOR_NAME = 'evaluate_siglip2_connected_mlp.py'
EVALUATOR = {'root': '/home/riomus/runs/sfora-connected-mlp-evaluation-source-v9',
             'execution_sha256': '3b9ade347801291b2cd4f4a3849be9eb2811b4237ddd562303d46cc3af505029',
             'code': {EVALUATOR_NAME: 'b651f6a02666ab9eafb4104eafbf464118b804baf94004db00d9263a7e3224e4',
                      'test_connected_mlp_evaluation.py': '48b47ea04f0cdf5f1980583d2a219c8d6803d0215293628956bc3b408139b89b'}}
EVALUATION_AUTHORITY = {'name': 'authority-full-export-control-179061-v2.json',
    'sha256': '0875aec90fe11c1523b66584bbec8164be5dc886d3da81131a4c220a1f624c8b'}
ORACLE_SEGMENT_SHA = '061700ae58189a27cdfb0c710fe523b0a430cd2337a8e21d530976de2ec4ad32'
ORACLE_AST_SHA = '08af8551a995c8e01e091d65c5b2af8281aa7b3912deadaa47a071f609ca7e5b'
# Actual Cutile authority (separate from the old cuDNN-four supplement) and its archived binary/build.
NATIVE_AUTHORITY_SHA = 'cf085f68dc2cfd10d41ab87e44256f21aec059e925f3b646cf7159aa054a17b1'
BINARY_SHA = '3d1ec7968713aa0f069f742b9454976c77ad77d115cf39c0844b6f14d6b6b526'
ARCHIVE_SHA = 'c2d6ff677c5c533f576774d8268c536483d27ff21e93c3efa8cf2319cacc7740'
CONTROL_NATIVE_SHA = 'fc8795be3cca792aa328f087b1908c8fc12ff75362802042a35e874c02732442'
# Root-frozen tested versions of the serving helpers (probe-serving-v3 freeze set).
REQUEST_DRIVER_SHA = '6a4d310d4eb883b7bf3d2ea74a1e96223cee096d0bdb9bc6c2a651eecf9ff163'
SERVING_OBSERVER_SHA = 'b255c6e835ad3d66b1143f2ca2192e500958fe8ffd6f62ec135a0326a6ad48de'
SERVING_NAMES = {'request_driver': 'qualify_connected_serving_requests.py', 'observer': 'observe_connected_serving.py',
                 'control_native': 'connected_control_native_authority.py', 'native_wrapper': 'cutile_int8.py',
                 'packed': 'packed_int8.py'}
# Installed wrapper/packer: sfora code runs at import, so their exact bytes are pinned BEFORE any import.
WRAPPER_SHA = '173eb393ed82d01281557b2f76edb1d50ffcc29faf85b9f9deb07d0df3c90d87'
PACKED_SHA = 'ac605a9fd7f412fc50ad158472cfc8cffc0d70e2da124f0459f97ec6d6bb7bc4'
CONNECTED_SHA = '79efb320da6fa59bcae7f5dbe19ccc33be8c961bfdf2a210cbc77b1925d4135b'
EXPORT_SHA = 'db63db8270f2bf9a75448863e7ff701b7894b0c6aaf45c748730b6ea52393407'
CONTROL_BUNDLE_SHA = 'e12429efd5cf1a6bd43d4bbf1c55fc163548fffb415813e5c3315658c144f153'
# Accepted control061 export batch54: the only B6 allowed to be recomputed from images.
# The ONLY images this job may ever open (accepted control061 batch54 query tail), frozen from committed evidence.
TAIL_IMAGES = [
    {'path': '/home/riomus/datasets/inshop_official_standard/img/img/WOMEN/Dresses/id_00007940/02_3_back.jpg',
     'sha256': '4651007318b8c63f8b5e7f9d3766c0d0c99e8160bf77766f9a6c082d38652854'},
    {'path': '/home/riomus/datasets/inshop_official_standard/img/img/WOMEN/Dresses/id_00007940/02_4_full.jpg',
     'sha256': '2d864b6c55e77f87632c58e586afdfd4a392e373d614dc3c9cf77c8ab55acd08'},
    {'path': '/home/riomus/datasets/inshop_official_standard/img/img/WOMEN/Jackets_Coats/id_00007941/02_1_front.jpg',
     'sha256': '54c5c487334801a7459491eee95277f6377524c4149db2b26921c33ffed12a49'},
    {'path': '/home/riomus/datasets/inshop_official_standard/img/img/WOMEN/Jackets_Coats/id_00007941/02_3_back.jpg',
     'sha256': '0aa9b680eaa44835e1d4f4a1ca5a65d8ee9c42773ac68bf4804de54fec552885'},
    {'path': '/home/riomus/datasets/inshop_official_standard/img/img/WOMEN/Jackets_Coats/id_00007953/01_1_front.jpg',
     'sha256': 'b831a71ef0368927c2a5314f2ce557ebf81ed5991b5a949b6dd971d316d34b1e'},
    {'path': '/home/riomus/datasets/inshop_official_standard/img/img/WOMEN/Jackets_Coats/id_00007953/01_7_additional.jpg',
     'sha256': 'dd5335b2fa0796132fdd5fae6a004cd4221f18e80efa28d67bfccdfc7886b054'},
]
TAIL = {'batch_index': 54, 'role': 'query', 'selection_rows': [3439, 3440, 3441, 3442, 3445, 3448],
        'role_indices': [1728, 1729, 1730, 1731, 1732, 1733],
        'fit_rows': [13239, 13240, 13241, 13242, 13254, 13257]}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha_ok(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def fact_ok(value):
    return (type(value) is dict and value.keys() == {'path', 'sha256'} and type(value['path']) is str and
            sha_ok(value['sha256']) and Path(value['path']).is_absolute() and str(Path(value['path'])) == value['path'])


def authenticated(fact, guards, *, keep=False):
    """Fresh byte read of one prospective FILE; images are admitted only by their own allowlist."""
    require(fact_ok(fact), 'exact FILE required')
    path = Path(fact['path'])
    require(path.resolve() == path and path.is_file() and not path.is_symlink(), 'canonical regular FILE required')
    before, digest, raw = path.stat(), hashlib.sha256(), None
    with path.open('rb') as stream:
        if keep:
            raw = stream.read(64*1024**2+1)
            require(len(raw) <= 64*1024**2, 'metadata/source FILE exceeds64MiB')
            digest.update(raw)
        else:
            while block := stream.read(1024**2):
                digest.update(block)
        after = os.fstat(stream.fileno())
    require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) ==
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns), 'FILE changed while reading')
    require(digest.hexdigest() == fact['sha256'], 'FILE SHA differs: '+fact['path'])
    require(guards.setdefault(fact['path'], fact['sha256']) == fact['sha256'], 'conflicting FILE authority')
    return raw if keep else path


def strict_json(raw):
    def pairs(items):
        value = dict(items)
        require(len(value) == len(items), 'duplicate JSON key')
        return value
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=lambda value: require(False, 'nonfinite JSON'))


def read_json(fact, guards):
    return strict_json(authenticated(fact, guards, keep=True))


def rehash(guards, *, phase='driver_rehash'):
    with ExitPhase(phase):
        for path, digest in tuple(guards.items()):
            authenticated({'path': path, 'sha256': digest}, {})


def load_module(name, fact, guards):
    """Execute authenticated bytes under a private name; the registry entry is owned until removal."""
    raw = authenticated(fact, guards, keep=True)
    require(name not in sys.modules, 'source namespace already owned')
    spec = importlib.util.spec_from_file_location(name, fact['path'])
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, fact['path'], 'exec', dont_inherit=True), vars(module))
    except BaseException:
        del sys.modules[name]
        raise
    return module


def remove_module(module):
    require(sys.modules.get(module.__name__) is module, 'source registry ownership changed')
    del sys.modules[module.__name__]


def check_oracle_source(raw):
    """The unchanged admitted fullfeature_oracle: exact source segment and AST of the accepted MLP evaluator."""
    tree = ast.parse(raw)
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fullfeature_oracle']
    require(len(nodes) == 1, 'sole fullfeature_oracle required')
    text = ast.get_source_segment(raw.decode(), nodes[0])
    require(hashlib.sha256(text.encode()).hexdigest() == ORACLE_SEGMENT_SHA and
            hashlib.sha256(ast.dump(nodes[0], include_attributes=False).encode()).hexdigest() == ORACLE_AST_SHA,
            'admitted fullfeature_oracle source/AST differs')


def check_launch(launch, execution_sha256, output):
    """Pure shape, fixed-value and historical-pin admission; every dynamic fact is checked at its use."""
    require(type(launch) is dict and launch.keys() == LAUNCH_KEYS and launch['schema'] == SCHEMA and
            launch['execution_sha256'] == execution_sha256, 'exact asymmetric cache launch required')
    require(launch['stage'] == 'first' and launch['seeds'] == [SEED], 'first-stage seed179061 only')
    require(launch['resource_policy'] == LIMITS and launch['both_locks_held'] is True and
            launch['candidate_status'] == 'KILL' and launch['qualification_eligible'] is False and
            launch['state_reuse_eligible'] is False, 'exact diagnostic-only KILL authority/caps required')
    require(launch['output'] == str(output) and Path(launch['output']).is_absolute(), 'bound NEWDIR required')
    locks = launch['locks']
    require(type(locks) is list and len(locks) == 2 and all(type(r) is dict and r.keys() == {'path', 'fd'} and
            type(r['path']) is str and type(r['fd']) is int and r['fd'] >= 3 for r in locks), 'two lifetime locks required')
    history = launch['historical']
    require(type(history) is dict and history.keys() == V3.keys() and
            all(fact_ok(history[k]) and history[k]['sha256'] == V3[k] for k in V3), 'pinned original v3 admission required')
    root = Path(history['source']['path']).parent
    require(Path(history['source']['path']).name == 'diagnose_connected_gallery_freshness.py' and
            Path(history['execution']['path']) == root/'execution.json' and
            Path(history['authority']['path']) == root/'authority.json', 'separate original v3 roles required')
    observer = launch['observer']
    require(type(observer) is dict and observer.keys() == {'source', 'execution', 'test'} and
            all(fact_ok(observer[k]) for k in observer), 'exact v5 observer FILEs required')
    root = Path(observer['source']['path']).parent
    require(observer['source']['sha256'] == V5['source'] and observer['test']['sha256'] == V5['test'] and
            Path(observer['source']['path']).name == V5_NAMES['source'] and
            Path(observer['test']['path']) == root/V5_NAMES['test'] and
            Path(observer['execution']['path']) == root/'execution.json', 'frozen v5 observer pins required')
    evaluator = launch['evaluator']
    require(evaluator == EVALUATOR, 'accepted old MLP evaluator CODE required')
    require(fact_ok(launch['evaluation_authority']) and
            launch['evaluation_authority'] == {'path': EVALUATOR['root']+'/'+EVALUATION_AUTHORITY['name'],
                                               'sha256': EVALUATION_AUTHORITY['sha256']},
            'accepted full control179061 export launch FILE required')
    native = launch['native']
    require(type(native) is dict and native.keys() == {'authority', 'library', 'archived_control_binary_ack'} and
            fact_ok(native['authority']) and fact_ok(native['library']) and native['archived_control_binary_ack'] is True and
            native['authority']['sha256'] == NATIVE_AUTHORITY_SHA and native['library']['sha256'] == BINARY_SHA,
            'actual Cutile authority and archived binary required')
    serving = launch['serving']
    require(type(serving) is dict and serving.keys() == SERVING_NAMES.keys() and
            all(fact_ok(serving[r]) and Path(serving[r]['path']).name == n for r, n in SERVING_NAMES.items()) and
            serving['control_native']['sha256'] == CONTROL_NATIVE_SHA and
            serving['request_driver']['sha256'] == REQUEST_DRIVER_SHA and
            serving['observer']['sha256'] == SERVING_OBSERVER_SHA and
            serving['native_wrapper']['sha256'] == WRAPPER_SHA and serving['packed']['sha256'] == PACKED_SHA and
            Path(serving['native_wrapper']['path']).parent == Path(serving['packed']['path']).parent,
            'actual wrapper/packer/requests/observer/native sources required')
    controls = launch['controls']
    require(type(controls) is dict and controls.keys() == {KEY} and type(controls[KEY]) is dict and
            controls[KEY].keys() == {'bundle', 'export'} and fact_ok(controls[KEY]['bundle']) and
            fact_ok(controls[KEY]['export']) and controls[KEY]['export']['sha256'] == EXPORT_SHA and
            controls[KEY]['bundle']['sha256'] == CONTROL_BUNDLE_SHA and
            Path(controls[KEY]['bundle']['path']).name == 'bundle.json', 'accepted control061 bundle/export required')
    tail = launch['tail_oracle']
    require(type(tail) is dict and tail.keys() == TAIL.keys() | {'images'} and
            all(tail[k] == v for k, v in TAIL.items()) and tail['images'] == TAIL_IMAGES,
            'exact frozen six-image tail allowlist required (checked before any image is opened)')


def prepare(args):
    guards = {}
    root = Path(__file__).absolute().parent
    code = read_json({'path': str(root/'execution.json'), 'sha256': args.execution_sha256}, guards)
    require(type(code) is dict and code.keys() == FILES, 'exact2 asymmetric closure required')
    for name, digest in code.items():
        authenticated({'path': str(root/name), 'sha256': digest}, guards)
    launch = read_json({'path': str(args.authority), 'sha256': args.authority_sha256}, guards)
    output = Path(args.output)
    check_launch(launch, args.execution_sha256, output)
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and
            not output.is_symlink(), 'exclusive canonical NEWDIR required')
    facts = [launch['historical'][k] for k in V3] + [launch['observer'][k] for k in ('source', 'test', 'execution')]
    facts += [launch['evaluation_authority'], launch['native']['authority'], launch['native']['library']]
    facts += list(launch['serving'].values()) + list(launch['controls'][KEY].values())   # never the tail images
    for fact in facts:
        authenticated(fact, guards)
    execution = read_json(launch['observer']['execution'], guards)
    require(execution == {V5_NAMES[k]: V5[k] for k in V5}, 'frozen v5 observer closure differs')
    return {'launch': launch, 'guards': guards, 'code': code, 'output': output}


# ---- pure falsifier predicates (stdlib; exercised without native modules) --------------------------------------

def require_tail_oracle_tap(differences):
    """Accept control-cache vs archived-export differences ONLY in the frozen six query-tail rows.

    Every one of the five outputs must be complete and localizable; any gallery row (including the final B19),
    any other query row, any truncation or unattributed byte stops the job. Never a tolerance."""
    require(type(differences) is dict and differences.keys() >= set(OUTPUT_NAMES), 'complete five-output tap required')
    rows_by_output = {}
    for name in OUTPUT_NAMES:
        item = differences[name]
        location = item['row_localization']
        require(location['status'] == 'AVAILABLE' and location['row_bytes'] == WIDTHS[name] and
                location['actual_bytes'] == location['expected_bytes'] == PANEL_ROWS*WIDTHS[name],
                name+': complete localized tap required')
        rows = location['rows']
        require(sum(r['different_bytes'] for r in rows) == item['different_bytes'] and
                len({r['selection_row'] for r in rows}) == len(rows), name+': unattributed or duplicate difference')
        for row in rows:
            require(row['selection_row'] in TAIL['selection_rows'], name+': difference outside the frozen six rows')
            i = TAIL['selection_rows'].index(row['selection_row'])
            require(row.get('mapping_status') == 'AVAILABLE' and row['role'] == TAIL['role'] and
                    row['role_index'] == TAIL['role_indices'][i] and
                    row['live_encoder_batch_index'] == TAIL['batch_index'] and row['live_encoder_batch_size'] == 6 and
                    row['live_encoder_tail'] is True and row['original_fit_index'] == TAIL['fit_rows'][i],
                    name+': difference is not the accepted B6 query tail')
        rows_by_output[name] = sorted(r['selection_row'] for r in rows)
    return rows_by_output


def deny_group_images(rows):
    """Only the six accepted selection images may be decoded; FIT groups 13216/13248 hold sealed VAL images."""
    require(list(rows) == TAIL['fit_rows'], 'only the accepted six tail images may be decoded')


def check_fs_composition(fs_wire, f_wire, s_wire, query, gallery):
    """FS bytes: every query row is the archived F row, every gallery row is the cache-built S row."""
    rows = len(query)+len(gallery)
    require(type(fs_wire) is bytes and len(fs_wire) == len(f_wire) == len(s_wire) == rows*130 and
            sorted(query+gallery) == list(range(rows)), 'complete FS/F/S wires required')
    for i in query:
        require(fs_wire[i*130:(i+1)*130] == f_wire[i*130:(i+1)*130], 'FS query row is not the archived F row')
    for i in gallery:
        require(fs_wire[i*130:(i+1)*130] == s_wire[i*130:(i+1)*130], 'FS gallery row is not the cache-built S row')


def compare_native(native_ids, native_scores, reference_ids, reference_scores, rows):
    """Exact int64 ordinal and FP32 score bit bytes for <=32 queries; ties are fixed by identical ordinals."""
    require(type(rows) is int and 1 <= rows <= 32, 'native search is at most 32 queries per call')
    require(all(type(v) is bytes for v in (native_ids, native_scores, reference_ids, reference_scores)) and
            len(native_ids) == len(reference_ids) == rows*TOP_K*8 and
            len(native_scores) == len(reference_scores) == rows*TOP_K*4, 'complete top10 native/reference bytes required')
    require(native_ids == reference_ids, 'native top10 ordinals/ties differ')
    require(native_scores == reference_scores, 'native top10 FP32 score bits differ')


def transitions(fs, ff):
    """Descriptive per-query FS minus FF changes; never a gate or a decision."""
    require(len(fs['per_query_r1']) == len(ff['per_query_r1']) and len(fs['per_query_ap']) == len(ff['per_query_ap']),
            'paired per-query vectors required')
    up = sum(a > b for a, b in zip(fs['per_query_r1'], ff['per_query_r1'], strict=True))
    down = sum(a < b for a, b in zip(fs['per_query_r1'], ff['per_query_r1'], strict=True))
    return {'r1_up': up, 'r1_down': down, 'r1_net': up-down,
            'recall_at_1_delta': fs['recall_at_1']-ff['recall_at_1'], 'map_at_r_delta': fs['map_at_r']-ff['map_at_r'],
            'descriptive_only': True, 'gate': False}


def write_exclusive(path, data):
    with Path(path).open('xb') as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    with Path(path).open('rb') as stream:
        require(stream.read() == data, 'persisted bytes read-back differs')


def persist_fs(output, wire, roles):
    """Persist the mixed wire and its separate query/gallery provenance, then read both back."""
    wire_path, roles_path = Path(output)/'fs-179061.packed.bin', Path(output)/'fs-179061.roles.json'
    write_exclusive(wire_path, wire)
    encoded = json.dumps(roles, sort_keys=True, allow_nan=False).encode()+b'\n'
    write_exclusive(roles_path, encoded)
    require(strict_json(roles_path.read_bytes()) == roles, 'FS roles read-back differs')
    return {'wire': {'path': str(wire_path), 'sha256': hashlib.sha256(wire).hexdigest()},
            'roles': {'path': str(roles_path), 'sha256': hashlib.sha256(encoded).hexdigest()}}


class ExitPhase:
    """Synchronous scalar markers at finite call sites; no sampler or retained owner/exception/frame.

    Missing/partial output makes observation inconclusive, never changes the original acceptance or cleanup.
    """
    __slots__ = ('phase',)

    def __init__(self, phase):
        self.phase = phase

    def mark(self, boundary):
        try:
            print(json.dumps({'phase': self.phase, 'boundary': boundary, 'monotonic_seconds': time.monotonic()}),
                  flush=True)
        except BaseException:
            pass

    def __enter__(self):
        self.mark('entry')

    def __exit__(self, kind, error, trace):
        self.mark('exit' if kind is None else 'error')


def attempt(callbacks):
    """Run every callback; return the failures in order (never raises)."""
    failures = []
    for callback in callbacks:
        try:
            callback()
        except BaseException as failure:
            failures.append(failure)
    return failures


def raise_primary(error, failures):
    """The primary error (or else the first failure) is raised; every other failure rides along as a note."""
    describe = lambda failure: 'cleanup: '+repr(failure)+''.join('; '+n for n in getattr(failure, '__notes__', []))
    if error is not None:
        for failure in failures:
            error.add_note(describe(failure))
        raise error
    if failures:
        for failure in failures[1:]:
            failures[0].add_note(describe(failure))
        raise failures[0]


def cleanup_error(error, callbacks):
    """Attempt every cleanup; the primary failure (or the first cleanup failure) is raised with the rest noted."""
    raise_primary(error, attempt(callbacks))


def report(failure, limit=64*1024):
    """One bounded stderr report of the original error graph, made before any frame is detached."""
    if getattr(failure, '_asymmetric_reported', False):
        return
    try:
        text = ''.join(traceback.format_exception(failure))
    except Exception:
        text = type(failure).__name__+'\n'
    print(text[:limit]+('\n[report truncated]' if len(text) > limit else ''), file=sys.stderr, flush=True)
    try:
        failure._asymmetric_reported = True
    except Exception:
        pass


def clear_frames(error):
    """Drop locals of FINISHED frames pinned by a retained exception graph (causes, contexts, group members).

    Tracebacks, identities and messages stay; still-active frames are untouched. Needed so a weakref lifetime check
    in a release helper can pass while the primary error is still in flight."""
    seen, pending = set(), [error]
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen:
            continue
        seen.add(id(error))
        trace = error.__traceback__
        while trace is not None:
            try:
                trace.tb_frame.clear()
            except RuntimeError:
                pass
            trace = trace.tb_next
        pending += [error.__cause__, error.__context__, *(error.exceptions if isinstance(error, BaseExceptionGroup) else ())]


def protected(phase, closers):
    """Run one phase. A failure stays the PRIMARY error: report it once, detach retained frames, THEN attempt every
    release (so owner lifetime checks can pass), re-raising the primary with every secondary failure noted."""
    error = result = None
    try:
        result = phase()
    except BaseException as failure:
        error = failure
        report(failure)
        clear_frames(failure)
    raise_primary(error, attempt(closers))
    return result


def write_json(path, value):
    """Exclusive, fsynced, read-back JSON written by this guarded module (not by the historical v3 writer)."""
    write_exclusive(path, json.dumps(value, sort_keys=True, allow_nan=False).encode()+b'\n')


def pin(fn):
    """Live identity of ONE runtime-created callable: code, __defaults__ tuple, __kwdefaults__ dict and its shallow
    key/value bindings, globals dict, closure cell identities and function attributes. Source guards module-level
    definitions only; the origin audit, workspace disposal and scorer are closures/exec'd functions created at run
    time. Identity comparisons only: no ==, no deepcopy, so opaque defaults (tensors, registries) cannot raise or pass
    ambiguously, and a rebound-but-equal default is detected.
    ponytail: the returned checker is covered by the guarded module's code, not by another pin."""
    require(type(fn) is FunctionType, 'plain function required')
    code, defaults, kwdefaults, namespace = fn.__code__, fn.__defaults__, fn.__kwdefaults__, fn.__globals__
    bindings = None if kwdefaults is None else tuple(kwdefaults.items())
    cells = tuple(c.cell_contents for c in fn.__closure__ or ())
    attributes = dict(vars(fn))
    same = lambda current, pinned: len(current) == len(pinned) and all(x is y for x, y in zip(current, pinned, strict=True))

    def check():
        require(fn.__code__ is code and fn.__defaults__ is defaults and fn.__kwdefaults__ is kwdefaults and
                fn.__globals__ is namespace and
                (bindings is None or (list(kwdefaults) == [k for k, _ in bindings] and
                                      same([kwdefaults[k] for k, _ in bindings], [v for _, v in bindings]))) and
                same([c.cell_contents for c in fn.__closure__ or ()], cells) and
                vars(fn).keys() == attributes.keys() and all(vars(fn)[k] is v for k, v in attributes.items()),
                'pinned runtime callable changed')
    return check


def live(S):
    """Everything this driver calls into that no Sources/evaluator/api guard covers: the pinned v3 and v5 modules
    (code, defaults, class dicts, globals, literals, registry) and every pinned runtime-created callable."""
    for source in (S.diagnostic_source, S.v5_source):
        if source is not None:
            source.check()
    for check in S.checks:
        check()


def gallery_difference(s_wire, f_wire, gallery):
    """Descriptive count of gallery rows/bytes where the cache-built S differs from the archived live F."""
    rows = nbytes = 0
    for g in gallery:
        a, b = s_wire[g*130:(g+1)*130], f_wire[g*130:(g+1)*130]
        if a != b:
            rows += 1
            nbytes += sum(x != y for x, y in zip(a, b, strict=True))
    return {'rows': rows, 'bytes': nbytes, 'of_rows': len(gallery), 'descriptive_only': True}


def registry_plan(S, disposed=False):
    """Exact ownership of sys.modules entries created since the pre-admission snapshot.

    tracked  = modules this driver loaded itself (S.modules / S.owned): judged by IDENTITY ONLY. The same name holding
               another object is a conflict and that object is foreign, whatever __file__/spec.origin it claims;
    owned    = other new entries loaded from an authenticated guarded FILE or from the control/candidate bundle
               directories (genuine evaluator/trainer loaders), excluding native/third-party top-level packages;
    foreign  = every other new entry (stdlib lazy imports, site packages, torch/PIL/sfora): reported, never unloaded;
    conflict = a preexisting entry replaced or removed, or a tracked name that no longer holds the tracked object."""
    guarded = set()
    for holder in (S.prospective, S.context, S.context_e):
        guarded.update((holder or {}).get('guards', {}))
    roots = [Path(e['bundle']['path']).parent for e in (S.control, S.candidate) if e]
    tracked = {module.__name__: module for module in (*S.modules, *(s.module for s in S.owned))}
    owned, foreign, conflicts = {}, [], []
    for name, module in list(sys.modules.items()):
        if S.registry.get(name) is module or name in tracked:
            continue
        if name in S.registry:
            conflicts.append(name+': preexisting module replaced')
            continue
        origin = getattr(module, '__file__', None) or getattr(getattr(module, '__spec__', None), 'origin', None)
        if (name.split('.')[0] not in NATIVE_PACKAGES and type(origin) is str and
                (origin in guarded or any(Path(origin).is_relative_to(r) for r in roots))):
            owned[name] = module
        else:
            foreign.append(name)
    for name, module in tracked.items():
        current = sys.modules.get(name)
        if current is module:
            owned[name] = module
        elif current is not None:
            conflicts.append(name+': tracked owned name now holds a different (foreign) module; retained')
        elif not disposed:
            conflicts.append(name+': tracked owned module removed before its final guards')
    conflicts += [name+': preexisting module removed' for name in S.registry if name not in sys.modules]
    return owned, sorted(foreign), sorted(set(conflicts))


def registry_dispose(S):
    """Delete exactly the owned entries, each only while it is STILL the captured object (compare-and-delete); a
    replaced entry is retained untouched and reported. A conflict never stops the other disposals; it rejects at the end."""
    with ExitPhase('cleanup.registry_dispose'):
        owned, foreign, conflicts = registry_plan(S)
        for name, module in owned.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
            else:
                conflicts.append(name+': owned entry replaced before deletion; retained')
        again, _, more = registry_plan(S, disposed=True)
        require(not (conflicts or again or more),
                'owned source registry not restored: '+repr({'conflicts': sorted(set(conflicts+more)), 'left': sorted(again)}))


def terminal_checks(S):
    """Mandatory terminal checks attempted independently of the historical final_state (which skips the allocator
    check without a receipt and everything after an expired cap) and of the whole cap (its own retained failure)."""
    with ExitPhase('cleanup.terminal_checks'):
        torch, checks = S.torch, []
        if torch is not None and torch.cuda.is_initialized():
            def allocator():
                torch.cuda.synchronize()
                require(torch.cuda.memory_allocated() == 0, 'terminal residual CUDA allocation')
                require(torch.cuda.max_memory_allocated() < LIMITS['cuda_allocated_bytes_exclusive'], 'terminal CUDA cap differs')
            checks.append(allocator)
        if S.source is not None and S.flags is not None:
            checks.append(lambda: require(S.source.numerical_flags() == S.flags, 'terminal numerical flags changed'))
        if torch is not None and S.rng is not None:
            checks.append(lambda: require(torch.equal(torch.random.get_rng_state(), S.rng[0]) and
                len(torch.cuda.get_rng_state_all()) == len(S.rng[1]) and
                all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), S.rng[1], strict=True)),
                'terminal RNG changed'))
        if S.source is not None and S.v3_before is not None and S.initializer is not None:
            def cgroup():
                after = S.source.cgroup_memory()
                S.initializer.admit_cgroup(after, Path(S.v3_before['path']).name.removesuffix('.service'))
                require(after['path'] == S.v3_before['path'], 'terminal enclosing cgroup changed')
            checks.append(cgroup)
        cleanup_error(None, checks)


def independent_exit(S):
    """Uncached context_e guard/closure/bundle checks from the exact admitted descriptors: the part of the genuine
    exit_rehash that follows its native audit. Run only after the combined exit REJECTED (e.g. S never mapped); the
    original rejection stays the raised error, so this waives nothing."""
    e, c = S.evaluator, S.context_e
    t, trainer = c['training_context'], c['trainer']
    t['trainer'].require_no_training(t)
    t['trainer'].helper_guard(t)
    e.guard_helpers(c)
    e.merge_guards(c['guards'], t['guards'])
    e.merge_guards(c['guards'], t['legacy']['guards'])
    trainer.batch_bound_files({}, c['guards'].items())
    for descriptor, names, pins in (({'root': str(c['root']), 'execution_sha256': c['args'].execution_sha256}, e.FILES, c['code']),
        *(((e.ORIGINAL_EXPORT_OWNER, e.FILES, e.ORIGINAL_EXPORT_OWNER['code']),) if 'original_evaluator' in c else ()),
        (c['launch']['training'], e.TRAIN_FILES, c['launch']['training']['code']),
        (c['launch']['evaluator_reference'], e.EVALUATOR_PINS, e.EVALUATOR_PINS),
        (c['launch']['nearest_evaluator'], e.NEAREST_EVALUATOR['code'], e.NEAREST_EVALUATOR['code']),
        (c['launch']['genuine_evaluator'], e.GENUINE_PINS, e.GENUINE_PINS),
        (c['launch']['reference'], c['launch']['reference']['code'], c['launch']['reference']['code'])):
        require(e.closure(descriptor['root'], descriptor['execution_sha256'], names, {}) == pins,
                'fresh complete source closure changed at exit')
    for endpoint in c['launch']['endpoints']:
        _, guards = trainer.admit_bundle(Path(endpoint['bundle']['path']).parent, endpoint['bundle']['sha256'])
        e.merge_guards(c['guards'], guards)
    e.guard_helpers(c)


def final_guard(S):
    """Full live-source/lock/resource guard of every owner, pin and both locks. Runs BEFORE any registry or Sources
    removal (guards need the registered modules); publication repeats only locks, resources and the cap afterwards."""
    with ExitPhase('final_guard'):
        S.locks.check()
        if S.sources is not None:
            S.sources.guard()
        for source in (S.self_source, S.request_source, S.observer_source, S.native_source, S.evaluator_source,
                       S.diagnostic_source, S.v5_source, S.wrapper_source, S.packed_source):
            if source is not None:
                source.check()
        live(S)
        with ExitPhase('final_guard.authenticate'):
            S.api.authenticate()
        with ExitPhase('final_guard.helpers'):
            S.evaluator.guard_helpers(S.context_e)
        with ExitPhase('final_guard.resources'):
            S.evaluator.resources(S.context_e, S.before)
        S.budget.check()


def stage_receipt(S):
    """Stage the receipt under a name the parent never accepts; the registry plan must be conflict free."""
    with ExitPhase('cleanup.stage_receipt'):
        owned, foreign, conflicts = registry_plan(S)
        require(not conflicts, 'preexisting module replaced or removed: '+repr(conflicts))
        S.receipt.update(exit_rehash_pass=True, integrity_pass=True, cleanup_pass=True,
            wall_seconds=time.perf_counter()-STARTED,
            registry={'owned_disposed': sorted(owned), 'foreign_new': foreign, 'conflicts': conflicts},
            terminal_binding={'invocation_id': os.environ['INVOCATION_ID'], 'normal_terminal_required': True,
                'publication': 'receipt.json is linked last; a nonzero exit or expired cap after the link invalidates it'})
        write_json(S.prospective['output']/'receipt.json.partial', S.receipt)


def publish_receipt(S):
    """Final locks + resources + cap after the registry disposal, then an exclusive link (never replaces); one more
    cap check after the link."""
    with ExitPhase('cleanup.publish_receipt'):
        out = S.prospective['output']
        S.locks.check()
        S.evaluator.resources(S.context_e, S.before)
        S.budget.check()
        os.link(out/'receipt.json.partial', out/'receipt.json')
        os.unlink(out/'receipt.json.partial')
        S.budget.check()


# ---- admission (pure python until native_start) ------------------------------------------------------------------

def bind_endpoints(context, launch):
    ordered = {(e['seed'], e['arm']): e for e in context['score']['launch']['endpoints']}
    control, candidate = ordered[SEED, 'control'], ordered[SEED, 'candidate']
    require(launch['controls'][KEY] == {'bundle': control['bundle'],
            'export': context['score']['launch']['exports'][KEY]['receipt']} and
            context['launch']['sources']['connected']['sha256'] == CONNECTED_SHA, 'accepted control061 only')
    return control, candidate


def bind_tail(context, launch):
    """The accepted control061 batch54 query tail: exactly six selection images, never an original FIT group."""
    panel = context['partition']['panels']['selection']
    batch = context['exports'][KEY]['images'][TAIL['batch_index']]
    rows, first, last = batch['rows'], TAIL['role_indices'][0], TAIL['role_indices'][-1]
    require(batch['role'] == TAIL['role'] and len(rows) == 6 and
            panel['query'][first:last+1] == TAIL['selection_rows'] and
            [r['panel_ordinal'] for r in rows] == TAIL['selection_rows'] and
            [r['original_row'] for r in rows] == TAIL['fit_rows'] and
            [panel['original_rows'][i] for i in TAIL['selection_rows']] == TAIL['fit_rows'], 'accepted B6 tail roles differ')
    deny_group_images([r['original_row'] for r in rows])
    forbidden = set(context['partition']['panels']['train']['original_rows'])
    forbidden.update(context['partition']['panels']['validation']['original_rows'])
    require(not forbidden.intersection(TAIL['fit_rows']), 'TRAIN/VAL image denied')
    for row in rows:
        original = context['fit']['rows'][row['original_row']]
        require(row['role'] == 'query' and all(row[k] == original[k] for k in ('train_row', 'relative_path', 'image_sha256')),
                'canonical FIT image binding differs')
    require(launch['tail_oracle']['images'] == [{'path': r['path'], 'sha256': r['image_sha256']} for r in rows],
            'exact ordered six tail image FILEs required')
    return batch


def admit(S, args):
    """Authenticate prospective FILEs, load every pinned source and build the genuine MLP evaluator context."""
    S.prospective = prepare(args)
    L, guards, serving = S.prospective['launch'], S.prospective['guards'], S.prospective['launch']['serving']
    S.requests = requests = load_module('_asymmetric_requests', serving['request_driver'], guards)
    S.modules.append(requests)
    S.request_source = requests.Source(requests, serving['request_driver'])
    S.self_source = requests.Source(sys.modules[__name__], {'path': str(Path(__file__).absolute()),
                                                           'sha256': S.prospective['code'][Path(__file__).name]})
    S.locks = requests.Locks(L['locks'])
    S.observer_source = requests.Source.load(serving['observer'])
    S.owned.append(S.observer_source)
    S.observer = S.observer_source.module
    S.native_source = requests.Source.load(serving['control_native'])
    S.owned.append(S.native_source)
    native = S.native_source.module
    require(native.BINARY_SHA == BINARY_SHA == L['native']['library']['sha256'] and native.ARCHIVE_SHA == ARCHIVE_SHA,
            'archived control binary/build pins differ; a new native authority module is required')
    runtime = requests.strict_json(S.observer.file_bytes(L['native']['authority'], keep=True))
    require(runtime['library'] == L['native']['library'], 'same frozen native FILE required')
    native.validate_runtime_compiler(runtime, S.observer)
    fact = L['evaluator']
    evaluator_file = {'path': str(Path(fact['root'])/EVALUATOR_NAME), 'sha256': fact['code'][EVALUATOR_NAME]}
    check_oracle_source(authenticated(evaluator_file, guards, keep=True))
    S.evaluator_source = native.load_evaluator_source(evaluator_file, requests)
    S.owned.append(S.evaluator_source)
    S.evaluator = evaluator = S.evaluator_source.module
    evaluator.check_code(fact, evaluator.FILES)
    require(evaluator.closure(fact['root'], fact['execution_sha256'], evaluator.FILES, {}) == fact['code'],
            'complete original evaluator CODE differs')
    # Historical v3 admission and the frozen v5 capture seam; nothing here imports native modules.
    history = L['historical']
    S.diagnostic = d = load_module('_asymmetric_v3', history['source'], guards)
    S.modules.append(d)
    S.diagnostic_source = requests.Source(d, history['source'])    # no Sources/evaluator guard covers the v3 module itself
    S.v5 = v5 = load_module('_asymmetric_v5', L['observer']['source'], guards)
    S.modules.append(v5)
    S.v5_source = requests.Source(v5, L['observer']['source'])
    v3_args = SimpleNamespace(execution_sha256=history['execution']['sha256'], authority=Path(history['authority']['path']),
                              authority_sha256=history['authority']['sha256'], output=S.prospective['output'])
    S.context = context = d.prepare(v3_args)
    S.sources = sources = d.Sources(context)
    sources.admit()
    S.invocations = d.terminal_admission(context, sources)
    require(len(S.invocations) == 6, 'six distinct original UNITs required')
    S.control, S.candidate = bind_endpoints(context, L)
    S.batch = bind_tail(context, L)
    for image in L['tail_oracle']['images']:
        v5.authenticated(image, context['guards'])
    v5.check_capture_ast(v5.authenticated(context['launch']['sources']['connected'], context['guards'], keep=True),
                         observer_raw=authenticated(L['observer']['source'], guards, keep=True))
    S.panel = context['partition']['panels']['selection']
    # The genuine inherited training context, then the actual Cutile authority over it (as control serving v5).
    eargs = SimpleNamespace(execution_sha256=fact['execution_sha256'], authority=Path(L['evaluation_authority']['path']),
                            authority_sha256=L['evaluation_authority']['sha256'], phase='export', arm='control', seed=SEED,
                            output=S.prospective['output'])
    S.context_e, S.exit_guard = evaluator.authority(eargs)
    S.context_e['training_context']['fit_context']['unit_started'] = STARTED
    S.frozen = [{'path': str(args.authority), 'sha256': args.authority_sha256}, L['native']['authority'],
                L['native']['library'], L['evaluation_authority'], *serving.values(), *history.values(),
                *L['observer'].values()]
    evaluator.merge_guards(S.context_e['guards'], {f['path']: f['sha256'] for f in S.frozen})
    combined = native.CombinedAuthority(S.context_e['training_context'], L['native']['authority'], S.observer, requests)
    evaluator.merge_guards(S.context_e['guards'], {f['path']: f['sha256'] for f in combined.provenance_facts()})
    S.api = combined.install(S.evaluator_source, S.context_e)
    S.budget.check()
    S.prospective['output'].mkdir()


def start_native(S):
    """First torch import, original interpreter/cgroup/flags/RNG, v5 workspace owner, archived packing and scorer."""
    evaluator, d, v5, context, m = S.evaluator, S.diagnostic, S.v5, S.context, S.sources.modules
    S.before = evaluator.native_start(S.context_e)
    import torch
    S.torch = torch
    prior, python = context['score']['invocation'], Path(sys.executable).resolve()
    require(str(python) == prior['python'] and m['extract'].sha(python) == prior['python_sha256'] and
            sys.version == prior['python_version'] and os.environ['INVOCATION_ID'] not in S.invocations,
            'original interpreter/fresh enclosing UNIT required')
    S.source = m['source_driver']
    S.v3_before = S.source.cgroup_memory()
    m['initializer'].admit_cgroup(S.v3_before, Path(S.v3_before['path']).name.removesuffix('.service'))
    S.workspace_dispose = v5.capture_workspace_owner(torch, context)
    S.dispose_check = pin(S.workspace_dispose)   # not in S.checks: its 'used' cell flips when it is (once) invoked
    S.initializer = m['initializer']
    S.flags = copy.deepcopy(context['exports'][KEY]['numerical_flags'])
    require(S.flags == context['cpu']['numerical_flags'] and S.flags['cudnn_allow_tf32'] is True and
            S.flags['matmul_allow_tf32'] is False and S.source.numerical_flags() == S.flags, 'accepted live flags required')
    S.rng = (torch.random.get_rng_state().clone(), [v.clone() for v in torch.cuda.get_rng_state_all()])
    require(torch.cuda.device_count() == 1, 'one admitted CUDA device required')
    S.packing = S.sources.load('packing')
    S.sources.checks.append(m['evaluator'].source_live_guard(S.packing, context['launch']['sources']['packing']['sha256'],
                                                              context['guards']))
    S.fixed = d.scorer(S.sources, context)
    S.checks.append(pin(S.fixed))

    def guard(*, deep=False):
        S.budget.check()
        S.locks.check()
        for source in (S.self_source, S.request_source, S.observer_source, S.native_source, S.evaluator_source,
                       S.diagnostic_source, S.v5_source, S.wrapper_source, S.packed_source):
            if source is not None:
                source.check()
        S.sources.guard()
        for check in S.checks+S.serving_checks:
            check()
        S.api.authenticate()
        evaluator.guard_helpers(S.context_e)
        evaluator.resources(S.context_e, S.before)
        require(S.source.numerical_flags() == S.flags and torch.equal(torch.random.get_rng_state(), S.rng[0]) and
                len(torch.cuda.get_rng_state_all()) == len(S.rng[1]) and
                all(torch.equal(a, b) for a, b in zip(torch.cuda.get_rng_state_all(), S.rng[1], strict=True)),
                'complete RNG/flags changed')
        require(torch.cuda.max_memory_allocated() < LIMITS['cuda_allocated_bytes_exclusive'], 'CUDA cap reached')
        if deep:   # the large v3/serving guards are rehashed once, at the uncached exit
            rehash(S.prospective['guards'])
            for fact in S.frozen:
                S.observer.file_bytes(fact)
    S.guard = guard
    S.light = lambda: (S.budget.check(), S.sources.guard(), live(S))   # per-batch loops; full guard() at phase boundaries
    guard(deep=True)
    snap(S, 'after_admission')


# ---- body phases ---------------------------------------------------------------------------------------------------

def snap(S, phase):
    """Bounded scalar cgroup diagnostics from the admitted unit (v5 observer); never a cap decision."""
    S.v5.memory_snapshot(S.v3_before['path'], phase)


def archive_replays(S):
    """All four accepted wires/descriptors and per-query R1/AP replay (as v3) before any new readout."""
    d, torch, context = S.diagnostic, S.torch, S.context
    labels, query, gallery = context['labels'], S.panel['query'], S.panel['gallery']
    S.fresh, S.archived = {}, {}
    for endpoint in context['score']['launch']['endpoints']:
        S.budget.check()
        key = d.label(endpoint)
        S.fresh[key] = d.fresh_values(context, S.sources, endpoint)
        quality = S.fixed(S.fresh[key]['unit'].numpy(), labels, query, gallery, device=torch.device('cpu'))
        d.replay(context['score']['quality'][str(endpoint['seed'])][endpoint['arm']], quality)
        d.replay(quality, d.native_wire_quality(S.fresh[key]['wire'], labels, query, gallery))
        S.archived[key] = quality
    m = S.sources.modules
    S.features = m['baseline'].cache_rows({'partition': context['partition'], 'guards': context['guards']},
                                          S.panel['original_rows'])
    require(S.features.shape == (PANEL_ROWS, 1152) and S.features.dtype == torch.float32,
            'direct original selection cache tap differs')
    m['connected'].mapping_absent(context['launch']['original_cache']['path'])


def negatives(S, endpoint, first_bytes):
    for mutant in ('omitted_C', 'wrong_mu'):
        mutated = S.diagnostic.stale_values(S.context, S.sources, endpoint, S.features, S.budget, mutant=mutant)
        require(first_bytes['raw'] != S.diagnostic.output_bytes(mutated)['raw'], mutant+' readout negative failed')
        mutated = None


def control_phase(S):
    """Full 3449-row control comparison: only the six frozen query-tail rows may differ from the accepted export."""
    d, context = S.diagnostic, S.context
    query, gallery = S.panel['query'], S.panel['gallery']
    first = d.stale_values(context, S.sources, S.control, S.features, S.budget)
    first_bytes, fresh_bytes = d.output_bytes(first), d.output_bytes(S.fresh[KEY])
    differences = d.control_byte_differences(first_bytes, fresh_bytes, S.panel, context['fit'])
    write_json(S.prospective['output']/(KEY+'-tap.json'), differences)
    rows = require_tail_oracle_tap(differences)
    second = d.stale_values(context, S.sources, S.control, S.features, S.budget)
    d.require_control_tap(d.byte_differences(first_bytes, d.output_bytes(second)))
    second = None
    negatives(S, S.control, first_bytes)
    composed = d.compose(first['unit'], S.fresh[KEY]['unit'], query, gallery, 'FS')
    d.exact_wire(S.packing.pack_int8_unit_embeddings(composed).to_bytes(), S.fresh[KEY]['wire'])
    S.control_S = first
    return {'differing_rows': rows, 'tail_selection_rows': TAIL['selection_rows'], 'rows_exact_outside_tail': PANEL_ROWS-6,
            'rerun_exact': True, 'negatives_rejected': ['omitted_C', 'wrong_mu'], 'composition_wire_exact': True,
            'tap': {n: {'exact': differences[n]['exact'], 'different_bytes': differences[n]['different_bytes']}
                    for n in OUTPUT_NAMES}}


def tail_replay(S):
    """Independent accepted B6: only the six frozen images, accepted RGB/pixels/output hashes, exact repeat.

    Runs under protected(): a failure stays the primary error, retained frames are detached before the owner
    lifetime check inside release_inference, and every release is attempted."""
    torch, d, v5, m, context = S.torch, S.diagnostic, S.v5, S.sources.modules, S.context
    from PIL import Image
    connected, original, batch, ordinals = m['connected'], m['original'], S.batch, TAIL['selection_rows']
    control_dir = Path(S.control['bundle']['path']).parent
    images = []

    def phase():
        S.state = connected.load_inference(control_dir, S.control['bundle']['sha256'], 'cuda')
        for module in S.state['modules'].values():
            S.serving_checks.append(m['evaluator'].source_live_guard(module, S.state['guards'][module.__file__],
                context['guards'], class_name=('FlatAdmission' if Path(module.__file__).name ==
                                               'train_siglip2_substrate_adaptation.py' else None)))
        for path, digest in S.state['guards'].items():
            require(context['guards'].setdefault(path, digest) == digest, 'serving guard conflict')
        S.guard()
        runs, live6 = [], None
        rgb = hashlib.sha256()
        for row in batch['rows']:
            S.light()
            v5.authenticated({'path': row['path'], 'sha256': row['image_sha256']}, context['guards'])
            with Image.open(row['path']) as opened:
                image = opened.convert('RGB')
            images.append(image)
            require(image.size == (256, 256), 'fixed image dimensions differ')
            rgb.update(str(image.size).encode())
            rgb.update(image.tobytes())
        require(rgb.hexdigest() == batch['rgb_sha256'], 'accepted B6 RGB proof differs')
        expected = {k: S.fresh[KEY][k][ordinals] for k in ('raw', 'unit', 'codes', 'inverse_norms')}
        expected['wire'] = b''.join(S.fresh[KEY]['wire'][i*130:(i+1)*130] for i in ordinals)
        expected = d.output_bytes(expected)
        for repeat in range(2):
            S.guard()
            values, pixels, pooled, features6 = v5.capture_inference_outputs(connected, S.state, images)
            require(torch.isfinite(pooled).all().item() and torch.isfinite(features6).all().item(), 'finite B6 captures required')
            require(original.fingerprint(pixels) == batch['pixels_sha256'] and
                    original.fingerprint(values) == batch['outputs_sha256'], 'accepted B6 pixel/output proof differs')
            actual = d.output_bytes(values)
            v5.exact(actual, expected, 'accepted B6 live descriptors differ from the archived control rows')
            same6 = d.output_bytes(v5.readout(connected, m['quadratic_owner'], S.state, features6))
            v5.exact(actual, same6, 'public vs original same B6 readout differs')
            current = {'pooled': v5.tensor_bytes(pooled), 'features': v5.tensor_bytes(features6), **same6}
            if repeat:
                v5.exact(current, runs[0], 'B6 repeatability differs')
            else:
                live6 = features6.clone()
            runs.append(current)
            values = pixels = pooled = features6 = None
        cache6 = S.features[ordinals]
        cache_out = d.output_bytes(v5.readout(connected, m['quadratic_owner'], S.state, cache6))
        cache_tail = d.output_bytes({k: S.control_S[k][ordinals] if k != 'wire' else
            b''.join(S.control_S['wire'][i*130:(i+1)*130] for i in ordinals) for k in OUTPUT_NAMES})
        v5.exact(cache_out, cache_tail, 'cache B6 readout differs from the control S tail rows')
        return {'batch_index': TAIL['batch_index'], 'selection_rows': ordinals, 'rgb_sha256': rgb.hexdigest(),
                'pixels_sha256': batch['pixels_sha256'], 'outputs_sha256': batch['outputs_sha256'],
                'live_equals_archived_control_rows': True, 'repeat_exact': True, 'cache_readout_equals_control_S_tail': True,
                'live_vs_cache_features_exact': v5.tensor_bytes(live6) == v5.tensor_bytes(cache6),
                'live_vs_cache_differing_elements': int((live6 != cache6).sum().item()),
                'description_only': True, 'image_forwards': 2, 'images_decoded': 6}

    def close_images():
        for image in images:
            image.close()
        images.clear()

    def release_state():
        S.serving_checks.clear()
        state, S.state = S.state, None     # one attempt only: never re-release an endpoint a failed release cleared
        if state is not None:
            connected.release_inference(state)

    return protected(phase, [close_images, release_state, lambda: connected.mapping_absent(control_dir/'vision.pt'),
                             gc.collect])


def gallery_oracle(S, values, first):
    """The unchanged admitted fullfeature_oracle over every original gallery B32/B19 batch."""
    torch, v5 = S.torch, S.v5
    state = {'arm': values['arm'], 'head_object': values['head'], 'A': values['A'], 'means': values['means'],
             'C': values['C'], 'mu_train': values['mu_train']}
    gallery, proofs = S.panel['gallery'], []
    for start in range(0, len(gallery), 32):
        S.light()
        indices = gallery[start:start+32]
        batch = S.features[indices].to('cuda')
        with torch.no_grad(), torch.autocast('cuda', enabled=False):
            output, proof = S.evaluator.fullfeature_oracle(S.context_e, state, batch)
        for name in ('raw', 'unit', 'codes', 'inverse_norms'):
            require(v5.tensor_bytes(output[name]) == v5.tensor_bytes(first[name][indices]),
                    'independent oracle differs from cache-built S gallery: '+name)
        require(proof['C_exact_zero'] is False and proof['residual_nonzero_witness'] and
                proof['omitted_C_mutant_rejected'] and proof['wrong_mu_mutant_rejected'], 'oracle negatives not rejected')
        proofs.append(proof)
        del batch, output
    return {'batches': len(proofs), 'gallery_rows': len(gallery), 'oracle_equals_S_gallery': True,
            'residual_oracle_first': proofs[0], 'residual_oracle_last': proofs[-1]}


def candidate_phase(S):
    """Candidate S with the same arithmetic as control, independent oracle, and the NO_ASYMMETRY vacuity check."""
    d, context = S.diagnostic, S.context
    values = d.endpoint_payload(context, S.sources, S.candidate)
    try:
        first = d.stale_values(context, S.sources, S.candidate, S.features, S.budget)
        first_bytes = d.output_bytes(first)
        second = d.stale_values(context, S.sources, S.candidate, S.features, S.budget)
        d.require_control_tap(d.byte_differences(first_bytes, d.output_bytes(second)))
        second = None
        negatives(S, S.candidate, first_bytes)
        oracle = gallery_oracle(S, values, first)
    finally:
        values = None
        gc.collect()
    S.candidate_S = first
    difference = gallery_difference(first['wire'], S.fresh[CANDIDATE]['wire'], S.panel['gallery'])
    vacuous = difference['rows'] == 0
    return {'rerun_exact': True, 'negatives_rejected': ['omitted_C', 'wrong_mu'], 'oracle': oracle,
            'S_gallery_equals_archived_F_gallery': vacuous, 'gallery_difference': difference}, vacuous


def fs_phase(S):
    """Mixed wire: archived F query rows + cache-built S gallery rows; persisted with provenance and read back."""
    d, context, L = S.diagnostic, S.context, S.prospective['launch']
    query, gallery = S.panel['query'], S.panel['gallery']
    S.cell = d.compose(S.candidate_S['unit'], S.fresh[CANDIDATE]['unit'], query, gallery, 'FS')
    wire = S.packing.pack_int8_unit_embeddings(S.cell).to_bytes()
    check_fs_composition(wire, S.fresh[CANDIDATE]['wire'], S.candidate_S['wire'], query, gallery)
    export = context['score']['launch']['exports'][CANDIDATE]
    roles = {'schema': 'connected-asymmetric-cache-roles-v1', 'cell': 'FS', 'seed': SEED, 'rows': PANEL_ROWS,
             'query': {'ordinals': query, 'provenance': 'archived candidate-179061 live export F', 'export': export['receipt']},
             'gallery': {'ordinals': gallery, 'provenance': 'original FIT-cache selection rows through the same candidate '
                         'head/A/means/C/mu (CUDA FP32 B32/B19), cache-built S', 'cache': context['launch']['original_cache'],
                         'endpoint': {'bundle': S.candidate['bundle'], 'inference_state_sha256': S.candidate['inference_state_sha256']}},
             'causal_freshness_claim': False}
    persisted = persist_fs(S.prospective['output'], wire, roles)
    back = Path(persisted['wire']['path']).read_bytes()
    require(back == wire, 'FS wire read-back differs')
    d.decode_wire(back, PANEL_ROWS)
    check_fs_composition(back, S.fresh[CANDIDATE]['wire'], S.candidate_S['wire'], query, gallery)
    return back, persisted


def reference_topk(S, wire):
    """Independent persisted-wire arithmetic: int32 dot, two FP32 multiplies, stable descending top10 per <=32 queries."""
    torch, v5 = S.torch, S.v5
    codes, inverse = S.diagnostic.decode_wire(wire, PANEL_ROWS)
    code, inv = torch.tensor(codes, dtype=torch.int32), torch.tensor(inverse, dtype=torch.float32)
    query, gallery = S.panel['query'], S.panel['gallery']
    ids, bits = [], []
    for start in range(0, len(query), 32):
        rows = query[start:start+32]
        scores = (code[rows] @ code[gallery].T).float()*inv[rows, None]*inv[None, gallery]
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :TOP_K]
        ids.append(v5.tensor_bytes(order.to(torch.int64)))
        bits.append(v5.tensor_bytes(torch.gather(scores, 1, order).contiguous()))
    return ids, bits


def load_native(S):
    """The ONLY place sfora/Cutile is imported and mapped: pinned package dir, source guards, tie witness, origin audit.

    CombinedAuthority.install makes the evaluator exit require the complete frozen S, so every successful
    terminal path (including NO_ASYMMETRY) must pass here exactly once, after all encoder work is released."""
    from sfora import cutile_int8, packed_int8
    L, serving = S.prospective['launch'], S.prospective['launch']['serving']
    wrapper_dir = Path(serving['native_wrapper']['path']).parent
    sfora = sys.modules['sfora']
    require(Path(sfora.__file__).resolve() == wrapper_dir/'__init__.py' and
            all(Path(p).resolve() == wrapper_dir for p in sfora.__path__), 'sfora is not the pinned installed package')
    S.wrapper_source = S.requests.Source(cutile_int8, serving['native_wrapper'])
    S.packed_source = S.requests.Source(packed_int8, serving['packed'])
    S.guard()
    ties = S.requests.native_ties(packed_int8.PackedInt8Embeddings, cutile_int8.CutilePackedInt8Gallery,
                                  L['native']['library'], S.observer)
    S.api.audit_origins(S.context_e['training_context']['legacy'])
    S.guard()
    return cutile_int8, packed_int8, ties


def native_stage(S, wire):
    """First Cutile mapping: tie witness, then real top10 IDs/FP32 bits against the persisted-wire reference."""
    L = S.prospective['launch']
    reference_ids, reference_scores = reference_topk(S, wire)
    cutile_int8, packed_int8, ties = load_native(S)
    query, gallery = S.panel['query'], S.panel['gallery']
    embeddings = packed_int8.PackedInt8Embeddings.from_bytes(b''.join(wire[g*130:(g+1)*130] for g in gallery),
                                                             count=len(gallery), dimensions=128)
    native = cutile_int8.CutilePackedInt8Gallery.open_packed(Path(L['native']['library']['path']), embeddings)
    failures, top1, embedded = [], [], None
    digest = {'ids': hashlib.sha256(), 'scores': hashlib.sha256()}
    try:
        for chunk, start in enumerate(range(0, len(query), 32)):
            S.light()
            rows = query[start:start+32]
            embedded = packed_int8.PackedInt8Embeddings.from_bytes(b''.join(wire[q*130:(q+1)*130] for q in rows),
                                                                   count=len(rows), dimensions=128)
            snapshot = S.observer.native_snapshot(native.search_packed(embedded, k=TOP_K))
            ids, scores = bytes.fromhex(snapshot[0]['hex']), bytes.fromhex(snapshot[1]['hex'])
            compare_native(ids, scores, reference_ids[chunk], reference_scores[chunk], len(rows))
            digest['ids'].update(ids)
            digest['scores'].update(scores)
            top1.extend(struct.unpack_from('<q', ids, r*TOP_K*8)[0] for r in range(len(rows)))
    except BaseException as failure:
        failures.append(failure)
    finally:
        try:
            native.close()
        except BaseException as failure:
            failures.append(failure)
        native = embedded = embeddings = None
    if failures:
        S.requests.raise_failures(failures)
    S.guard()
    return {'ties': ties, 'queries': len(query), 'chunks': len(reference_ids), 'top_k': TOP_K,
            'native_ids_sha256': digest['ids'].hexdigest(), 'native_scores_sha256': digest['scores'].hexdigest(),
            'ids_scores_ties_exact_vs_persisted_wire_reference': True}, top1


def quality_stage(S, wire, top1):
    """Only after native parity: the original scorer on FS, replayed on the persisted wire and native top-1."""
    d, torch, labels = S.diagnostic, S.torch, S.context['labels']
    query, gallery = S.panel['query'], S.panel['gallery']
    fs = S.fixed(S.cell.numpy(), labels, query, gallery, device=torch.device('cpu'))
    d.replay(fs, d.native_wire_quality(wire, labels, query, gallery))
    hits = [int(labels[q] == labels[gallery[t]]) for q, t in zip(query, top1, strict=True)]
    require(hits == fs['per_query_r1'], 'native top-1 differs from scored per-query R1')
    ff = S.archived[CANDIDATE]
    return {'FF': ff, 'FS': fs}, transitions(fs, ff)


def body(S):
    """Fixed order; every earlier gate precedes the Cutile mapping, which precedes every quality number."""
    result = {}
    archive_replays(S)
    S.guard(deep=True)
    result['control'] = control_phase(S)
    S.guard()
    result['tail_oracle'] = tail_replay(S)
    S.guard()
    snap(S, 'after_tail_replay')
    result['candidate'], vacuous = candidate_phase(S)
    S.guard(deep=True)
    origins = S.audit()
    S.v5.exact_four(S.context, S.sources, origins)
    result['status'] = 'NO_ASYMMETRY' if vacuous else 'MEASURED'
    if vacuous:
        result['native_ties'] = load_native(S)[2]
        return result
    wire, result['fs_wire'] = fs_phase(S)
    S.guard()
    snap(S, 'before_native')
    result['native_parity'], top1 = native_stage(S, wire)
    S.guard(deep=True)
    result['cells'], result['fs_minus_ff_description'] = quality_stage(S, wire, top1)
    return result


# ---- lifecycle -----------------------------------------------------------------------------------------------------

def run(args):
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None,
            'unoptimized -B unprofiled startup required')
    require(not NATIVE_PACKAGES.intersection(n.split('.')[0] for n in sys.modules),
            'native import preceded explicit admission')
    S = SimpleNamespace(budget=SimpleNamespace(check=lambda: require(time.perf_counter()-STARTED < LIMITS['seconds'],
                                                                       'whole700-second asymmetric cap reached')),
                        registry=dict(sys.modules), modules=[], owned=[], checks=[], serving_checks=[], prospective=None,
                        requests=None, request_source=None, self_source=None, observer_source=None, native_source=None,
                        evaluator_source=None, wrapper_source=None, packed_source=None, diagnostic=None, v5=None,
                        context=None, sources=None, context_e=None, exit_guard=None, api=None, state=None, audit=None,
                        workspace_dispose=None, torch=None, receipt=None, combined=None, source=None, v3_before=None,
                        rng=None, flags=None, control=None, candidate=None, guard=None, light=None, locks=None,
                        diagnostic_source=None, v5_source=None, dispose_check=None, initializer=None, before=None,
                        evaluator=None)
    error = None
    try:
        admit(S, args)
        start_native(S)
        S.audit = S.diagnostic.origin_audit(S.context, S.sources)
        S.checks.append(pin(S.audit))
        S.audit()
        result = body(S)
        L = S.prospective['launch']
        prior = S.context['score']['invocation']
        S.receipt = {'schema': RECEIPT_SCHEMA, **result, 'candidate_status': 'KILL unchanged',
            'qualification_eligible': False, 'state_reuse_eligible': False, 'official_read': False,
            'validation_read': False, 'causal_freshness_claim': False, 'descriptive_only': True,
            'unmeasured': ['SS', 'SF', 'serving_index_cost'],
            'remaining_confound': 'FIT export batch composition/preprocessing and frozen-encoder cache versus the updated '
                                  'live encoder; control parity does not prove all candidate arithmetic equivalence',
            'authority': {'path': str(args.authority), 'sha256': args.authority_sha256}, 'launch': L,
            'cgroup_before': S.v3_before, 'numerical_flags': S.flags, 'rng_flags_preserved': True,
            'resource_policy': LIMITS, 'invocation': {'invocation_id': os.environ['INVOCATION_ID'],
                'python': str(Path(sys.executable).resolve()), 'python_sha256': prior['python_sha256'],
                'python_version': sys.version, 'optimize': 0},
            'terminal_exit_and_both_locks_require_parent_receipt': True}
    except BaseException as failure:
        error = failure
        report(failure)
        clear_frames(failure)
    finally:
        cleanup(S, error)
    return S.receipt


def cleanup(S, error):
    """Release every owner, run the combined H-union-S uncached exit, the independent terminal checks and the final
    source/lock/resource guard; ONLY with no error and no failure stage the receipt, dispose the registry and link it.

    Every step is attempted whatever failed before it; the primary error (else the first failure) is raised."""
    actions = []

    def checked(action, phase='cleanup.checked'):
        """Never execute a tampered v3/v5/Sources function during cleanup: guard first, the failure stays secondary."""
        def run_checked():
            with ExitPhase(phase):
                if S.locks is not None:
                    S.locks.check()
                live(S)
                if S.sources is not None:
                    S.sources.guard()
                action()
        return run_checked
    if S.v5 is not None and S.v3_before is not None:
        actions.append(checked(lambda: snap(S, 'before_exit'), phase='cleanup.snapshot'))
    m = S.sources.modules if S.sources is not None else {}
    connected, initializer = m.get('connected'), m.get('initializer')  # Sources.close empties m before final_state
    if S.state is not None and connected is not None:
        def release():
            S.serving_checks.clear()
            state, S.state = S.state, None
            connected.release_inference(state)
        actions.append(checked(release, phase='cleanup.release'))
    if S.context is not None and connected is not None:
        def maps():
            connected.mapping_absent(S.context['launch']['original_cache']['path'])
            for endpoint in S.context['score']['launch']['endpoints']:
                connected.mapping_absent(Path(endpoint['bundle']['path']).parent/'endpoint.pt')
            connected.mapping_absent(Path(S.control['bundle']['path']).parent/'vision.pt')
        actions.append(checked(maps, phase='cleanup.maps'))
    if S.context_e is not None:
        def exit_evaluator():
            if S.api is None:
                with ExitPhase('evaluator.exit_rehash'):
                    S.evaluator.exit_rehash(S.context_e, S.exit_guard)
                return
            try:
                with ExitPhase('combined.evaluator_exit'):
                    S.api.evaluator_exit(S.context_e, S.exit_guard)
            except BaseException as combined:
                try:
                    with ExitPhase('cleanup.independent_exit'):
                        independent_exit(S)
                    combined.add_note('independent uncached context/closure/bundle checks completed')
                except BaseException as extra:
                    combined.add_note('independent uncached exit checks also failed: '+repr(extra))
                raise
            with ExitPhase('combined.evidence'):
                S.combined = S.api.evidence()
            if S.receipt is not None:
                S.receipt['combined_native'] = S.combined
        actions.append(exit_evaluator)
    if S.context is not None:
        actions.append(lambda: rehash(S.context['guards'], phase='rehash.context'))
    if S.prospective is not None:
        actions.append(lambda: rehash(S.prospective['guards'], phase='rehash.prospective'))
    actions.append(S.budget.check)
    if S.api is not None and S.locks is not None and S.context_e is not None:
        actions.append(lambda: final_guard(S))      # full live guards while every registered module is still in place
    if S.sources is not None:
        actions.append(checked(S.sources.close, phase='cleanup.sources_close'))
    if S.workspace_dispose is not None:
        def final_resources():
            failure = None
            try:
                with ExitPhase('cleanup.workspace_dispose'):
                    S.dispose_check()
                    S.workspace_dispose()
            except BaseException as caught:
                failure = caught
                caught.__traceback__ = None
            try:
                with ExitPhase('cleanup.final_state'):
                    S.diagnostic.final_state(S.budget, S.source, initializer, S.v3_before, S.rng, S.flags, S.receipt)
            except BaseException as final_failure:
                if failure is not None:
                    final_failure.add_note('workspace cleanup: '+repr(failure))
                raise
            if failure is not None:
                raise failure
        actions.append(checked(final_resources, phase='cleanup.final_resources'))
    actions.append(lambda: terminal_checks(S))
    failures = attempt(actions)
    staged = False
    if error is None and not failures and S.receipt is not None:
        failures += attempt([lambda: stage_receipt(S)])
        staged = not failures
    failures += attempt([lambda: registry_dispose(S)])
    if staged and not failures:
        failures += attempt([lambda: publish_receipt(S)])
    if staged and failures:
        failures += attempt([lambda: (S.prospective['output']/'receipt.json.partial').unlink(missing_ok=True)])
    raise_primary(error, failures)


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--execution-sha256', required=True)
    p.add_argument('--authority', required=True, type=Path)
    p.add_argument('--authority-sha256', required=True)
    p.add_argument('--output', required=True, type=Path)
    return p


if __name__ == '__main__':
    published = run(parser().parse_args())
    print(json.dumps({'schema': RECEIPT_SCHEMA, 'status': published['status'], 'output': published['launch']['output']}),
          flush=True)
