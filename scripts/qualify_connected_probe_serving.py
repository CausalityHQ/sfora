#!/usr/bin/env python3
"""Separate engineering installed-probe public-parity gate; native launch UNRUN.

CLI: python -B qualify_connected_probe_serving.py --authority FILE
--authority-sha256 SHA --output NEWDIR. FILE={path:canonical_absolute_regular_file,
sha256:actual64hex}; CODE={root,execution_sha256,code}; UNIT is the unchanged
evaluator UNIT. Authority keys are exactly KEYS (schema SCHEMA); sources has
exactly SOURCES. Nothing future is hashed or invented here: bundle, gallery,
native, wheel, images, UNITs and evaluator CODE are root-supplied dynamic facts.
The literal pins below are ONLY the reviewed probe runtime/ledger/bridge/packed
bytes, the accepted probe trainer/test bytes, its nine-file historical vector and
the archived control cutile binary/build named by the root.

Engineering-only: no quality, speed, release, deployment or state-reuse claim and
a source pass does not authorize a native launch. Passing this gate is necessary
but not sufficient: the chosen artifact's own acceptance gates remain required.

One encoder owner at a time: (R) the authenticated copied probe public loader
produces same-batch raw/unit/codes/inverse_norms/wire for B1, B2, B32 and the
actual export tail and is completely released; then (A) the genuine installed
ConnectedCompactIndex.from_probe_bundle (installed non-editable files) is
compared exactly on the same batches through the public search_images, with a
return-value witness of the exact installed inference_outputs code object (no
rebinding), repeated calls, reversible live mutants, factory mutants, close,
double close and post-close rejection; then (B) a fresh reload owner repeats the
parity. No cross-batch floating-point equality is demanded. The export tail wire
is also compared with the persisted export packed.bin rows.

Reused unchanged: requests.Source/Locks/owner/check_resources/native_ties/
decode_images, observer.file_bytes/native_snapshot/tensor_snapshot, the probe
evaluator authority/accept_unit/native_start/endpoint_scope/authenticate_payloads/
resources/exit_rehash and the control CombinedAuthority H-union-S exit. SEAMS NOT
REUSABLE (OWN2 stop evidence): observer.check_runtime_sources/from_index and
requests.request_body are MLP-bound; control_native.load_evaluator_source hard-codes
one evaluator filename Constant (derived here by exact AST substitution/inverse
from the authenticated original bytes); CombinedAuthority pins the archived control
binary/build, so only that exact binary (explicit root ACK) is admitted and any
other binary STOPS (needs a new native-authority module outside OWN2).
MLP regression remains the unchanged control gate and is not duplicated here.

UNRESOLVED (source-only, NOT executed by any source pass): the live probe
training_context + CombinedAuthority composition. It first runs after the
parent-owned native admission: evaluator.authority(eargs) accepting the real frozen
probe export launch, then admit_probe, then CombinedAuthority.__init__
(nearest.native_source_api/bind_native_authority on the live context), install()
(derives on context['old'].audit_origins and context['fitter'].exit_rehash, the
evaluator exit_rehash substitution), api.evaluator_exit and evaluator.native_start.
Source evidence is only: key coverage by the probe flows, the sole exit_rehash
substitution target with exact inverse, the real evaluator loaded through the
derived loader, and the real evaluator.authority() rejecting a junk launch.

Limits are exact: whole1500/body300/exitreserve300/8GiB/0swap and zero events/
CUDA<1e10/both locks. Admission, every fresh hash, all owners, mutants, cleanup,
nested exit and publication count; failure closes the original attempt.
accept_unit(context,UNIT,authority_FILE) is the parent's original normal-terminal
analogue (receipt, exact CLI, normal exit, log, cgroups, resources, both locks).
"""
import argparse
import ast
import base64
import copy
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys
import time
from types import SimpleNamespace
import uuid
import weakref

import qualify_connected_serving_requests as requests

SCHEMA = 'connected-probe-serving-authority-v1'
RECEIPT = 'connected-probe-serving-diagnostic-v1'
KEYS = {'schema','sources','evaluator','evaluation_authority','endpoint','train_export','bundle',
        'gallery','images','native','wheel','locks','resource_policy'}
INSTALLED = ('bridge','runtime','ledger','packed','native_wrapper','packing')
SOURCES = {'probe_driver','probe_test','request_driver','request_test','observer','observer_test',
           'control_native', *INSTALLED}
INSTALLED_NAMES = {'bridge':'connected_compact_serving.py','runtime':'connected_probe_inference.py',
    'ledger':'_connected_probe_inference_authority.py','packed':'packed_int8.py',
    'native_wrapper':'cutile_int8.py','packing':'joint_relational_compaction.py'}
EVALUATOR_FILES = {'evaluate_siglip2_connected_probe.py','test_connected_probe_evaluation.py'}
KINDS = ('B1','B2','B32','TAIL')
OUTPUT_KEYS = {'raw','unit','codes','inverse_norms','wire'}
POLICY = {'body_seconds':300,'host_bytes':8*1024**3,'swap_bytes':0,
          'cuda_allocated_bytes_exclusive':10_000_000_000,'whole_process_seconds':1500,'exit_reserve_seconds':300}
STARTED = time.perf_counter()

# Reviewed SOURCE_GO bytes (connected-probe-library-source-v1); installed files must equal them.
RUNTIME_SHA = 'ea49b80d2d8c80aea54f71a1f01a58053f04858b9ffe03d168e0bd28c94fac44'
LEDGER_SHA = '0e989dd087614499096512a22948f4e8d3f9bb2840a487f2e9fe7e2d9f0371ab'
BRIDGE_SHA = 'd3ebfd9a575d7fb77ffbf3a3e1c2d01793edaa03eef0da63faf543596c1d68b3'
PACKED_SHA = 'ac605a9fd7f412fc50ad158472cfc8cffc0d70e2da124f0459f97ec6d6bb7bc4'
LEDGER_SCHEMA = 'sfora-connected-probe-inference-extraction-v1'
BUNDLE_SCHEMA = 'siglip2-connected-probe-bundle-v1'
TRAINER = 'train_siglip2_connected_probe.py'
HISTORICAL = (
    ('extract_siglip2_vision_source.py','a184b5382a2c0c4b0a24804648fbc53c1d10317a08dfe12afb9a2b00258afa2d'),
    ('joint_relational_compaction.py','4ca0de1b0579ea6165c81e9057e9afe77e6dd4141f0b4a0adb281de25300de67'),
    ('prototype_residual_readout.py','2bf9af74d2e59aed94898da4d54004ba47036a928d86c863939fa7188039de68'),
    ('quadratic_readout.py','12f0d8d4f799bdd620f827431d51f5ce011e5f54f7bc7efa24530d7e787eb3c6'),
    ('qualify_siglip2_substrate_cpu.py','eacd32d2ef551414906ae067c188f94d524562d3d031ac68bbd66c38b56f9e38'),
    ('test_siglip2_connected_probe.py','4accfd6c276ef2ca3d85b25c98ead7d013972d260656266b1227dd59648f0120'),
    ('train_siglip2_cached_readout.py','a687a62b78eeb4c122491f23394954ad192acc02394e29b66efb257d3a6f338c'),
    ('train_siglip2_connected_probe.py','e2496033a958cc83bf8d3204c87b4d3b8284528f03f44c1cb64517df04c8cc1e'),
    ('train_siglip2_substrate_adaptation.py','a168491758481a10d59469116b8ea5318eea733b7d9445a99a174afd6f74b543'))
# Archived control cutile binary/build (root-named; engineering reuse needs explicit provenance FILEs + ACK).
ARCHIVED_BINARY_SHA = '3d1ec7968713aa0f069f742b9454976c77ad77d115cf39c0844b6f14d6b6b526'
ARCHIVED_BUILD_SHA = 'c2d6ff677c5c533f576774d8268c536483d27ff21e93c3efa8cf2319cacc7740'
NATIVE_SCOPE = 'archived-control-binary-explicitly-acknowledged-engineering-only'
MLP_EVALUATOR_NAME = 'evaluate_siglip2_connected_mlp.py'
PROBE_EVALUATOR_NAME = 'evaluate_siglip2_connected_probe.py'
OWNED_PREFIXES = ('_sfora_connected_compact_','_connected_serving_','_connected_probe_gate_')


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def sha_ok(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}',value) is not None


def fact_ok(value):
    return (type(value) is dict and value.keys() == {'path','sha256'} and type(value['path']) is str and
        sha_ok(value['sha256']) and Path(value['path']).is_absolute() and str(Path(value['path'])) == value['path'])


def unit_ok(unit):
    return (type(unit) is dict and unit.keys() == {'receipt','log','unit','invocation_id','service_seconds',
        'native_peak_rss_kib','both_locks_held'} and fact_ok(unit['receipt']) and fact_ok(unit['log']) and
        unit['both_locks_held'] is True and type(unit['unit']) is str and
        re.fullmatch('[A-Za-z0-9_.@-]+',unit['unit']) is not None and type(unit['invocation_id']) is str and
        re.fullmatch('[0-9a-f]{32}',unit['invocation_id']) is not None and
        all(type(unit[k]) in (int,float) and math.isfinite(unit[k]) and unit[k] > 0 for k in ('service_seconds','native_peak_rss_kib')))


def policy_ok(policy):
    return type(policy) is dict and policy == POLICY and all(type(v) is int for v in policy.values())


def check_shape(authority):
    """Pure shape admission; every future fact is root-supplied and checked elsewhere."""
    require(type(authority) is dict and authority.keys() == KEYS and authority['schema'] == SCHEMA,
        'exact probe serving authority required')
    sources = authority['sources']
    require(type(sources) is dict and sources.keys() == SOURCES and all(fact_ok(v) for v in sources.values()),
        'complete actual probe source FILE pins required')
    evaluator = authority['evaluator']
    require(type(evaluator) is dict and evaluator.keys() == {'root','execution_sha256','code'} and
        type(evaluator['root']) is str and Path(evaluator['root']).is_absolute() and sha_ok(evaluator['execution_sha256']) and
        type(evaluator['code']) is dict and evaluator['code'].keys() == EVALUATOR_FILES and
        all(sha_ok(v) for v in evaluator['code'].values()), 'exact probe evaluator CODE required')
    require(fact_ok(authority['evaluation_authority']), 'actual probe export launch FILE required')
    endpoint = authority['endpoint']
    require(type(endpoint) is dict and endpoint.keys() == {'arm','seed','stage','panel'} and
        type(endpoint['arm']) is str and type(endpoint['seed']) is int and
        endpoint['stage'] in ('first','full') and endpoint['panel'] in ('selection','validation'),
        'exact same-arm/seed endpoint role required')
    require(unit_ok(authority['train_export']), 'complete actual probe export UNIT required')
    bundle = authority['bundle']
    require(type(bundle) is dict and bundle.keys() == {'directory','manifest','code','files'} and
        type(bundle['directory']) is str and Path(bundle['directory']).is_absolute() and
        fact_ok(bundle['manifest']) and bundle['manifest']['path'] == str(Path(bundle['directory'])/'bundle.json') and
        type(bundle['code']) is dict and bundle['code'].keys() == dict(HISTORICAL).keys() and
        all(fact_ok(f) and f['path'] == str(Path(bundle['directory'])/n) and f['sha256'] == dict(HISTORICAL)[n]
            for n,f in bundle['code'].items()) and
        type(bundle['files']) is dict and bundle['files'].keys() == {'vision.pt','endpoint.pt','processor.json'} and
        all(fact_ok(f) and f['path'] == str(Path(bundle['directory'])/n) for n,f in bundle['files'].items()),
        'exact copied probe bundle and reviewed nine-file vector required')
    gallery = authority['gallery']
    require(type(gallery) is dict and gallery.keys() == {'file','count'} and fact_ok(gallery['file']) and
        type(gallery['count']) is int and gallery['count'] >= 10, 'exact gallery binding required')
    images = authority['images']
    require(type(images) is dict and images.keys() == {'fixed32','tail'} and type(images['fixed32']) is list and
        len(images['fixed32']) == 32 and all(fact_ok(f) for f in images['fixed32']) and
        len({f['path'] for f in images['fixed32']}) == 32 and type(images['tail']) is dict and
        images['tail'].keys() == {'role','files'} and images['tail']['role'] in ('query','gallery') and
        type(images['tail']['files']) is list and 1 <= len(images['tail']['files']) <= 32 and
        all(fact_ok(f) for f in images['tail']['files']) and
        len({f['path'] for f in images['tail']['files']}) == len(images['tail']['files']),
        'fixed32 distinct ordered FILEs and an exact export-tail binding required')
    native = authority['native']
    require(type(native) is dict and native.keys() == {'authority','library','archived_control_binary_ack'} and
        fact_ok(native['authority']) and fact_ok(native['library']) and native['archived_control_binary_ack'] is True and
        native['library']['sha256'] == ARCHIVED_BINARY_SHA,
        'explicit root ACK of the archived control binary and its original provenance FILEs required')
    wheel = authority['wheel']
    require(type(wheel) is dict and wheel.keys() == {'site_root','distribution','version','record','direct_url'} and
        type(wheel['site_root']) is str and Path(wheel['site_root']).is_absolute() and
        type(wheel['distribution']) is str and wheel['distribution'] and type(wheel['version']) is str and wheel['version'] and
        fact_ok(wheel['record']) and (wheel['direct_url'] is None or fact_ok(wheel['direct_url'])),
        'exact non-editable installed wheel descriptor required')
    require(type(authority['locks']) is list, 'two original inherited lifetime lock descriptors required')
    require(policy_ok(authority['resource_policy']), 'exact whole1500/body300/exitreserve300 typed policy required')


def literal_record(raw):
    record = {}
    for node in ast.parse(raw).body:
        if isinstance(node,ast.Expr) and isinstance(node.value,ast.Constant) and type(node.value.value) is str:
            continue
        require(isinstance(node,ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0],ast.Name),
            'literal installed probe authority required')
        name = node.targets[0].id
        require(name not in record,'duplicate installed probe authority assignment')
        record[name] = ast.literal_eval(node.value)
    return record


def check_installed(sources, bundle, observer):
    """Authenticate installed execution and copied historical evidence without executing it."""
    for role in INSTALLED: observer.file_bytes(sources[role])
    directory = observer.canonical(sources['bridge']['path']).parent
    require(all(sources[r]['path'] == str(directory/n) for r,n in INSTALLED_NAMES.items()),
        'installed source sibling differs')
    require((sources['bridge']['sha256'],sources['runtime']['sha256'],sources['ledger']['sha256'],sources['packed']['sha256']) ==
        (BRIDGE_SHA,RUNTIME_SHA,LEDGER_SHA,PACKED_SHA), 'reviewed installed probe pins differ')
    record = literal_record(observer.file_bytes(sources['ledger'],keep=True))
    require(record.keys() == {'SCHEMA','HISTORICAL_CODE','SOURCE_SYMBOLS','PACKED_SOURCE_SYMBOLS','SUBSTITUTIONS',
            'RUNTIME_SHA256','PACKED_SHA256'} and record['SCHEMA'] == LEDGER_SCHEMA and
        record['RUNTIME_SHA256'] == RUNTIME_SHA and record['PACKED_SHA256'] == PACKED_SHA and
        record['HISTORICAL_CODE'] == HISTORICAL, 'exact installed probe ledger required')
    functions = [n for n in ast.parse(observer.file_bytes(sources['bridge'],keep=True)).body
        if isinstance(n,ast.FunctionDef) and n.name == '_installed_probe_authority']
    require(len(functions) == 1 and len(functions[0].body) == 1 and isinstance(functions[0].body[0],ast.Return) and
        isinstance(functions[0].body[0].value,ast.Tuple) and len(functions[0].body[0].value.elts) == 2 and
        isinstance(functions[0].body[0].value.elts[1],ast.Constant) and
        functions[0].body[0].value.elts[1].value == sources['ledger']['sha256'], 'bridge probe ledger identity differs')
    manifest = json.loads(observer.file_bytes(bundle['manifest'],keep=True),object_pairs_hook=observer.pairs,
        parse_constant=lambda value: require(False,'nonfinite JSON'))
    require(type(manifest) is dict and manifest.get('schema') == BUNDLE_SCHEMA and type(manifest.get('code')) is dict and
        manifest['code'] == dict(HISTORICAL) and type(manifest.get('files')) is dict and
        manifest['files'].keys() == bundle['files'].keys() and
        all(bundle['files'][n]['sha256'] == h for n,h in manifest['files'].items()),
        'exact probe bundle closure required (no MLP or mixed schema)')
    for fact in [*bundle['code'].values(),*bundle['files'].values()]:
        observer.file_bytes(fact)
        require(Path(fact['path']).stat().st_nlink == 1,'single-link owned bundle required')
    return manifest


def check_wheel(wheel, sources, observer, checkout):
    """Stdlib proof that the installed files are a non-editable regular install outside the checkout."""
    site = observer.canonical(wheel['site_root'])
    checkout = Path(checkout).resolve()
    require(site.is_dir() and not site.is_relative_to(checkout) and not checkout.is_relative_to(site) and
        not any((p/'.git').exists() for p in (site,*site.parents)), 'installed wheel root must be outside any checkout')
    record = observer.canonical(wheel['record']['path'])
    observer.file_bytes(wheel['record'])
    dist = site/(wheel['distribution']+'-'+wheel['version']+'.dist-info')
    require(record == dist/'RECORD', 'exact installed distribution RECORD required')
    rows = {r[0]:r for r in csv.reader(observer.file_bytes(wheel['record'],keep=True).decode().splitlines())
        if len(r) == 3}
    evidence = {}
    for role in INSTALLED:
        path = observer.canonical(sources[role]['path'])
        require(path.is_relative_to(site) and path.parent == site/'sfora', 'installed role FILE outside wheel root: '+role)
        row = rows.get(path.relative_to(site).as_posix())
        want = 'sha256='+base64.urlsafe_b64encode(bytes.fromhex(sources[role]['sha256'])).rstrip(b'=').decode()
        require(row is not None and row[1] == want and row[2] == str(path.stat().st_size),
            'installed RECORD digest/size differs: '+role)
        evidence[role] = row
    direct = dist/'direct_url.json'
    if wheel['direct_url'] is None:
        require(not direct.exists(),'undeclared direct_url.json')
    else:
        require(wheel['direct_url']['path'] == str(direct),'exact direct_url.json required')
        value = json.loads(observer.file_bytes(wheel['direct_url'],keep=True),object_pairs_hook=observer.pairs)
        require(type(value) is dict and not (value.get('dir_info') or {}).get('editable',False),'editable install rejected')
    for entry in site.iterdir():
        require(not entry.name.startswith('__editable__') or wheel['distribution'] not in entry.name.lower(),
            'editable finder rejected')
        if entry.suffix == '.pth' and entry.is_file():
            text = entry.read_bytes()[:64*1024].decode(errors='replace')
            require(wheel['distribution'] not in text.lower() and str(checkout) not in text,'editable .pth rejected')
    return {'site_root':str(site),'distribution':wheel['distribution'],'version':wheel['version'],
        'record':wheel['record'],'direct_url':wheel['direct_url'],'rows':evidence}


def check_loaded(site, sfora):
    site = Path(site)
    require(Path(sfora.__file__).resolve() == site/'sfora'/'__init__.py' and
        all(Path(p).resolve().is_relative_to(site) for p in sfora.__path__), 'sfora is not the installed wheel')


def derive_evaluator_loader(raw, namespace, path):
    """Exact one-Constant AST substitution of the original helper; inverse proves all else unchanged."""
    node = [n for n in ast.parse(raw).body if isinstance(n,ast.FunctionDef) and n.name == 'load_evaluator_source']
    require(len(node) == 1, 'original load_evaluator_source required')
    original, derived = node[0], copy.deepcopy(node[0])
    dump = lambda value: ast.dump(value,include_attributes=False)
    def swap(tree, before, after):
        hits = [n for n in ast.walk(tree) if isinstance(n,ast.Constant) and n.value == before]
        require(len(hits) == 1,'sole evaluator filename Constant required')
        hits[0].value = after
    swap(derived,MLP_EVALUATOR_NAME,PROBE_EVALUATOR_NAME)
    inverse = copy.deepcopy(derived)
    swap(inverse,PROBE_EVALUATOR_NAME,MLP_EVALUATOR_NAME)
    require(dump(inverse) == dump(original) and dump(derived) != dump(original),'derived evaluator loader inverse differs')
    space = dict(namespace)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[derived],type_ignores=[])),path,'exec'),space)
    return space['load_evaluator_source']


class Capture:
    """Return-value witness of one exact code object; keeps no frame, tensor or model reference."""
    def __init__(self, code, namespace, snapshot):
        self.code, self.namespace, self.snapshot = code, namespace, snapshot
        self.output = self.failure = None
        self.count = 0

    def __call__(self, frame, event, arg):
        try:
            if event == 'return' and frame.f_code is self.code and frame.f_globals is self.namespace:
                self.count += 1
                if arg is not None and self.output is None:
                    try: self.output = self.snapshot(arg)
                    except Exception as error:
                        # Bounded scalar facts only: a kept exception would pin arg, tensors and frames.
                        self.failure = (type(error).__name__[:64],str(error)[:256])
        finally:
            frame = arg = None


def captured_search(index, images, snapshot, native_snapshot):
    """Public search_images with a restored-identity profile witness of the installed outputs."""
    code, namespace = index._apis['inference_outputs'][1], vars(index._module)
    previous = sys.getprofile()
    require(previous is None,'unprofiled capture required')
    capture, result = Capture(code,namespace,snapshot), None
    try:
        sys.setprofile(capture)
        result = index.search_images(images)
    finally:
        sys.setprofile(previous)
        require(sys.getprofile() is previous,'profile callback identity not restored')
    failure, output, count = capture.failure, capture.output, capture.count
    capture = None
    if failure is not None: raise ValueError('installed output snapshot failed: %s: %s' % failure)
    require(count == 1 and output is not None,'exactly one installed inference_outputs return witness required')
    native = native_snapshot(result)
    result = None
    return output, native


def snapshot_outputs(arg, tensor_snapshot):
    require(type(arg) is dict and arg.keys() == OUTPUT_KEYS and type(arg['wire']) is bytes,'inference return incomplete')
    count = arg['raw'].shape[0]
    require(type(count) is int and 1 <= count <= 32 and len(arg['wire']) == count*130,'bounded output batch/wire required')
    specs = (('raw','torch.float32',[count,128],count*512),('unit','torch.float32',[count,128],count*512),
        ('codes','torch.int8',[count,128],count*128),('inverse_norms','torch.float16',[count],count*2))
    result = {name:tensor_snapshot(arg[name],dtype,shape,width) for name,dtype,shape,width in specs}
    result['wire_hex'] = arg['wire'].hex()
    return result


def check_outputs(output, count):
    require(type(output) is dict and output.keys() == {'raw','unit','codes','inverse_norms','wire_hex'},
        'complete raw/unit/codes/inverse_norms/wire witness required')
    for name,dtype,shape,width in (('raw','torch.float32',[count,128],4),('unit','torch.float32',[count,128],4),
            ('codes','torch.int8',[count,128],1),('inverse_norms','torch.float16',[count],2)):
        value = output[name]
        require(type(value) is dict and value.keys() == {'dtype','shape','hex'} and value['dtype'] == dtype and
            value['shape'] == shape and type(value['hex']) is str and re.fullmatch('[0-9a-f]*',value['hex']) is not None and
            len(value['hex']) == count*(128 if name != 'inverse_norms' else 1)*width*2,'typed output bytes differ')
    wire = output['wire_hex']
    require(type(wire) is str and re.fullmatch('[0-9a-f]*',wire) is not None and len(wire) == count*130*2,'wire differs')
    codes, norms = bytes.fromhex(output['codes']['hex']), bytes.fromhex(output['inverse_norms']['hex'])
    require(bytes.fromhex(wire) == b''.join(codes[i*128:(i+1)*128]+norms[i*2:(i+1)*2] for i in range(count)),
        'codes/inverse_norms/wire parity differs')


def check_native(value, count, gallery=None):
    require(type(value) is list and len(value) == 2,'complete native ID/score witness required')
    for row,fmt,width in zip(value,('q','f'),(8,4),strict=True):
        formats = ('q','l') if fmt == 'q' and struct.calcsize('l') == 8 else (fmt,)
        require(type(row) is dict and row.keys() == {'format','shape','hex'} and row['format'] in formats and
            row['shape'] == [count,10] and type(row['hex']) is str and re.fullmatch('[0-9a-f]+',row['hex']) is not None and
            len(row['hex']) == count*10*width*2,'typed native ID/score bytes differ')
    ids = [v[0] for v in struct.iter_unpack('<q',bytes.fromhex(value[0]['hex']))]
    require(all(v >= 0 and (gallery is None or v < gallery) for v in ids) and
        all(math.isfinite(v[0]) for v in struct.iter_unpack('<f',bytes.fromhex(value[1]['hex']))),
        'finite native score/in-gallery nonnegative ID required')


def rgb_digest(images):
    h = hashlib.sha256()
    for image in images:
        h.update(str(image.size).encode())
        h.update(image.tobytes())
    return h.hexdigest()


def clear_frames(error):
    """Drop locals of FINISHED frames pinned by a retained exception graph (causes, contexts and
    exception-group members); active frames are untouched."""
    seen, pending = set(), [error]
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen: continue
        seen.add(id(error))
        trace = error.__traceback__
        while trace is not None:
            try: trace.tb_frame.clear()
            except RuntimeError: pass
            trace = trace.tb_next
        pending += [error.__cause__,error.__context__,*(error.exceptions if isinstance(error,BaseExceptionGroup) else ())]


def close_all(images, failures):
    for image in images:
        try: image.close()
        except BaseException as error: failures.append(error)
    images.clear()


def owned_names():
    return sorted(n for n in sys.modules if n.startswith(OWNED_PREFIXES))


def select_batches(facts, rows, mapping, exported_sizes, batch_sizes):
    """Ordered fixed32 membership/bytes plus the exact last export batch of one role."""
    selected = {r['path']:r for r in rows}
    fixed = facts['fixed32']
    require(all(f['path'] in selected and selected[f['path']]['image_sha256'] == f['sha256'] for f in fixed),
        'fixed32 membership/image bytes differ from the admitted panel rows')
    role, tail = facts['tail']['role'], facts['tail']['files']
    size = batch_sizes(len(mapping[role]))[-1]
    require(exported_sizes[role][-1] == size == len(tail),'export tail size differs')
    ordinals = mapping[role][-size:]
    require([(f['path'],f['sha256']) for f in tail] == [(rows[i]['path'],rows[i]['image_sha256']) for i in ordinals],
        'ordered export-tail membership/bytes differ')
    return [('B1',fixed[:1]),('B2',fixed[:2]),('B32',fixed),('TAIL',tail)],ordinals


def bind_gallery(wire, rows, mapping, gallery):
    require(type(wire) is bytes and len(wire) == len(rows)*130 and gallery['count'] == len(mapping['gallery']),
        'export wire/gallery row count differs')
    data = b''.join(wire[i*130:(i+1)*130] for i in mapping['gallery'])
    require(hashlib.sha256(data).hexdigest() == gallery['file']['sha256'],'gallery row/wire binding differs')
    return data


def reference_outputs(*, name, directory, bundle_sha, loader, pin, guards, reads_only, batches, decode, rgb,
        snapshot, loaded, after, deadline, registry=None):
    """Mirror the evaluator's copied-loader ownership and fully release it before returning."""
    registry = sys.modules if registry is None else registry
    portable = state = None
    failures, rows = [], {}
    try:
        deadline()
        with reads_only():
            portable = loader(name,Path(directory)/TRAINER,pin,guards)
            observation = _LoaderObservation(portable,name,directory,pin,guards,registry)
            try:
                observation.start()
                state = portable.load_inference(Path(directory),bundle_sha,'cuda')
            finally:
                observation.close()
                observation = None
        deadline()
        loaded(state)
        for kind,paths in batches:
            images = decode(paths)
            try:
                pixels = rgb(images)
                with reads_only():
                    first = snapshot(portable.inference_outputs(state,images))
                    deadline()
                    second = snapshot(portable.inference_outputs(state,images))
                deadline()
            finally: close_all(images,failures)
            require(first == second,'reference same-batch repeat differs: '+kind)
            check_outputs(first,len(paths))
            rows[kind] = {'rgb_sha256':pixels,'outputs':first}
    except BaseException as error:
        _observation_emit({'event':'reference_catch_before_clear_frames','errors':_observation_errors(error)})
        clear_frames(error)
        failures.append(error)
    finally:
        # Each cleanup step is attempted independently, even after an earlier one (or the deadline) failed.
        if state is not None:
            try: portable.release_inference(state)
            except BaseException as error: failures.append(error)
        state = None
        try: after()
        except BaseException as error: failures.append(error)
        if portable is not None:
            try: require(registry.pop(name,None) is portable,'owned loader registry changed')
            except BaseException as error: failures.append(error)
        portable = None
        for error in failures: clear_frames(error)
    if failures: requests.raise_failures(failures)
    return rows


def reference_native(packed, gallery_type, native, gallery_wire, count, reference, observer, deadline):
    """Independent resident gallery searches the reference wires; closed before any installed owner."""
    deadline()
    observer.file_bytes(native)
    gallery = gallery_type.open_packed(Path(native['path']),packed.from_bytes(gallery_wire,count=count,dimensions=128))
    failures, rows = [], {}
    try:
        for kind,row in reference.items():
            deadline()
            wire = bytes.fromhex(row['outputs']['wire_hex'])
            result = gallery.search_packed(packed.from_bytes(wire,count=len(wire)//130,dimensions=128),k=10)
            rows[kind] = observer.native_snapshot(result)
            result = None
        deadline()
    except BaseException as error: failures.append(error)
    finally:
        try: gallery.close()
        except BaseException as error: failures.append(error)
        gallery = None
        if failures: requests.raise_failures(failures)
    return rows


def installed_pass(index, batches, reference, native, decode, rgb, capture, owner, repeats, deadline):
    rows, failures = [], []
    for kind,paths in batches:
        for call in range(repeats):
            deadline()
            images = decode(paths)
            try:
                require(rgb(images) == reference[kind]['rgb_sha256'],'identical image membership/order/preprocessing differs: '+kind)
                output, found = capture(index,images)
            finally: close_all(images,failures)
            if failures: requests.raise_failures(failures)
            check_outputs(output,len(paths))
            require(output == reference[kind]['outputs'],'installed raw/unit/codes/inverse_norms/wire differs: '+kind)
            require(found == native[kind],'installed native top10 ID/score bits differ: '+kind)
            rows.append({'owner':owner,'kind':kind,'call':call,'outputs_sha256':digest(output),'native_sha256':digest(found)})
    deadline()
    return rows


def flip(sha):
    return ('0' if sha[0] != '0' else '1')+sha[1:]


def rejection(error):
    """Bounded scalar evidence of an expected genuine ValueError rejection; keeps no exception or frame."""
    return {'type':type(error).__name__,'message':str(error)[:256]}


def factory_mutants(make, deadline):
    """Cheap hash/count/schema failures; none may leak an owned registry entry.

    Only an exact ValueError (the bridge/runtime/packed rejection type) counts; any other error fails the gate."""
    base = make.base
    cases = (('bundle_sha256','from_probe_bundle',{'expected_bundle_sha256':flip(base['expected_bundle_sha256'])}),
        ('gallery_sha256','from_probe_bundle',{'expected_gallery_sha256':flip(base['expected_gallery_sha256'])}),
        ('native_sha256','from_probe_bundle',{'expected_native_library_sha256':flip(base['expected_native_library_sha256'])}),
        ('gallery_count','from_probe_bundle',{'gallery_count':base['gallery_count']+1}),
        ('mlp_factory_on_probe_bundle','from_bundle',{}))
    results = {}
    for name,method,overrides in cases:
        deadline()
        before = owned_names()
        try: index = make(method=method,**overrides)
        except ValueError as error:
            if type(error) is not ValueError: raise
            results[name] = rejection(error)
        else:
            index.close()
            raise ValueError('factory mutant accepted: '+name)
        require(owned_names() == before,'factory mutant leaked owned registry entries: '+name)
    deadline()
    return results


def run_mutants(mutants, detect, deadline):
    """Apply, require rejection, restore: every restore is attempted and reported.

    The body deadline runs only between mutants (never while mutated, never inside the rejection
    handler); only an exact ValueError counts as rejection, anything else fails the gate."""
    results = {}
    for name,apply,restore,mode in mutants:
        deadline()
        rejected = None
        failures = []
        try:
            # Ownership first: a partial apply that raises must still be restored.
            try:
                apply()
                try: detect(mode)
                except ValueError as error:
                    if type(error) is not ValueError: raise
                    rejected = rejection(error)
            except BaseException as error: failures.append(error)
        finally:
            try: restore()
            except BaseException as error: failures.append(error)
        if failures: requests.raise_failures(failures)
        require(rejected is not None,'live mutant accepted: '+name)
        results[name] = rejected
    deadline()
    return results


def leaf_mutant(torch, name, value):
    """Direct zero-tuple .data mutation: the real storage changes and the version counter does not."""
    backup = value.detach().clone()
    where = (0,)*value.dim()
    def apply():
        version = value._version
        if where: value.data[where] += .25
        else: value.data.add_(.25)
        require(not torch.equal(value.detach(),backup) and value._version == version,
            'leaf mutation must change the real storage and bypass versions: '+name)
    def restore():
        value.data.copy_(backup)
        require(torch.equal(value.detach(),backup),'leaf mutation restoration differs: '+name)
    return (name,apply,restore,'outputs')


def native_live_mutants(index, torch):
    """Reversible live mutants of the installed owner; detection is the genuine runtime/bridge, except
    cpu_rng/cuda_rng, which the qualifier's own whole-unit RNG guard detects."""
    endpoint, module = index._endpoint, index._module
    model, head = endpoint['model'], endpoint['head_object']
    params = dict(model.named_parameters())
    leaves = [*[('probe_leaf:'+n,params[n]) for n in module.PROBE],
        ('frozen_leaf',next(p for n,p in params.items() if n not in module.PROBE)),('head_parameter',next(head.parameters())),
        ('readout_C',endpoint['C']),('readout_mu_train',endpoint['mu_train'])]
    # No nested closure owns the list: a failed backup clone leaves no earlier backup reachable from its traceback.
    mutants = [leaf_mutant(torch,name,value) for name,value in leaves]
    config = model.config
    prior = config._attn_implementation
    other = 'eager' if prior != 'eager' else 'sdpa'
    mutants.append(('config_attn_implementation',lambda: setattr(config,'_attn_implementation',other),
        lambda: setattr(config,'_attn_implementation',prior),'outputs'))
    registration = model.embeddings._non_persistent_buffers_set
    mutants.append(('buffer_registration',lambda: registration.remove('position_ids'),
        lambda: registration.add('position_ids'),'outputs'))
    nominated = params[module.PROBE[0]]
    wrong = next(p for n,p in params.items() if n != module.PROBE[0] and p.is_leaf and p.numel() == nominated.numel() and
        not torch.equal(p.detach().reshape(nominated.shape),nominated.detach()))
    backup = nominated.detach().clone()
    mutants.append(('same_numel_leaf_substitution',lambda: nominated.data.copy_(wrong.detach().reshape(nominated.shape)),
        lambda: nominated.data.copy_(backup),'outputs'))
    first = next(head.parameters())
    mutants.append(('head_requires_grad',lambda: first.requires_grad_(True),lambda: first.requires_grad_(False),'outputs'))
    processor = endpoint['processor_object']
    mean = processor.image_mean  # restore the original object itself, not a copy
    mutants.append(('processor_config',lambda: setattr(processor,'image_mean',[v+.125 for v in mean]),
        lambda: setattr(processor,'image_mean',mean),'outputs'))
    precision = torch.get_float32_matmul_precision()
    changed = 'medium' if precision != 'medium' else 'high'
    mutants.append(('numerical_flags',lambda: torch.set_float32_matmul_precision(changed),
        lambda: torch.set_float32_matmul_precision(precision),'outputs'))
    cpu, cuda = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state_all()
    mutants.append(('cpu_rng',lambda: torch.rand(1),lambda: torch.random.set_rng_state(cpu),'guard'))
    mutants.append(('cuda_rng',lambda: torch.rand(1,device='cuda'),lambda: torch.cuda.set_rng_state_all(cuda),'guard'))
    fn = index._apis['inference_outputs'][0]
    code, replacement = fn.__code__, (lambda endpoint, images: None).__code__
    mutants.append(('runtime_function_code',lambda: setattr(fn,'__code__',replacement),lambda: setattr(fn,'__code__',code),'current'))
    helper = module.owned_copy
    defaults = helper.__defaults__
    mutants.append(('runtime_function_defaults',lambda: setattr(helper,'__defaults__',('cuda',)),
        lambda: setattr(helper,'__defaults__',defaults),'current'))
    space, arms = vars(module), module.ARMS
    mutants.append(('runtime_global_rebind',lambda: space.__setitem__('ARMS',('control',)),
        lambda: space.__setitem__('ARMS',arms),'current'))
    mutants.append(('runtime_global_added',lambda: space.__setitem__('_gate_extra',1),
        lambda: space.pop('_gate_extra',None),'current'))
    return mutants


def live_mutants(index, torch, images, guard, deadline):
    def detect(mode):
        if mode == 'outputs': index._apis['inference_outputs'][0](index._endpoint,images)
        elif mode == 'current': index._check_current()
        else: guard()
    mutants = None
    try:
        mutants = native_live_mutants(index,torch)
        result = run_mutants(mutants,detect,deadline)
    except BaseException as error:
        # Drop closures/tensor backups and finished gate frames BEFORE requests.owner tears the index down.
        # A retained traceback keeps each failed apply/restore frame and, through it, the function and its
        # closure, so the gate-owned cells (parameters, backups) are emptied, not merely unreferenced.
        for _,apply,restore,_ in mutants or ():
            for cell in (*(apply.__closure__ or ()),*(restore.__closure__ or ())): cell.cell_contents = None
        mutants = detect = None
        clear_frames(error)
        raise
    mutants = detect = None
    index._check_current()
    return result


def mappings_absent(directory, text):
    return all(str(directory) not in line for line in text.splitlines())


def lifecycle(index, decode, paths, directory, maps, deadline):
    """Close/double-close/post-close plus finished-reference, registry and mapping cleanup."""
    deadline()
    endpoint = index._endpoint
    refs = [weakref.ref(endpoint[k]) for k in ('model','processor_object','head_object','A','C','mu_train')]
    cache = endpoint.get('processor_cache')
    endpoint = None
    require(owned_names(),'live installed owner registry entry missing')
    index.close()
    index.close()
    failures, images = [], decode(paths)
    try:
        try: index.search_images(images)
        except RuntimeError: pass
        else: raise ValueError('post-close search accepted')
    finally: close_all(images,failures)
    if failures: requests.raise_failures(failures)
    gc.collect()
    require(all(ref() is None for ref in refs),'closed index retained model/processor/readout references')
    require(owned_names() == [],'closed index retained owned registry entries')
    require(cache is not None and cache.cache_info().currsize == 0,'closed index retained processor cache entries')
    cache = None
    require(mappings_absent(directory,maps()),'closed index retained bundle mappings')
    deadline()
    return {'double_close_safe':True,'post_close_rejected':True,'weakrefs_released':True,
        'processor_cache_empty':True,'registry_clean':True,'mappings_absent':True}


class Denial:
    """One inert-after-use audit hook: installed inference must never execute the copied historical files."""
    def __init__(self, directory):
        self.paths = {str(Path(directory)/n) for n,_ in HISTORICAL}
        self.names = {n.removesuffix('.py') for n,_ in HISTORICAL}
        self.active, self.checks = False, 0
        sys.addaudithook(self)

    def __call__(self, event, args):
        if not self.active: return
        name = None
        if event == 'compile' and len(args) > 1: name = args[1]
        elif event == 'exec' and args and hasattr(args[0],'co_filename'): name = args[0].co_filename
        elif event == 'import' and len(args) > 1: name = args[1]
        if isinstance(name,(str,bytes,os.PathLike)) and os.fsdecode(name) in self.paths:
            raise ValueError('installed inference attempted to execute a historical source')

    def check(self):
        self.checks += 1
        for name,module in list(sys.modules.items()):
            file = getattr(module,'__file__',None)
            require(name.split('.')[0] not in self.names and
                not (isinstance(file,str) and file in self.paths),'historical source module is live: '+name)
        return {'checks':self.checks,'historical_modules_absent':True}


def parity_body(*, batches, ordinals, persisted_tail, reference, native, factory, decode, rgb, capture, mutants,
        lifecycle_check, denial, guard, started, mutant_factory):
    """Sequential owners R, A (parity+mutants+lifecycle), B (reload); full costs counted."""
    def deadline():
        # Lightweight exact body300 check between expensive calls: no guard/source/deep work.
        require(time.perf_counter()-started < POLICY['body_seconds'],'diagnostic body300 cap exceeded')
    def bounded():
        guard()
        deadline()
    charges, installed = [{},{}], []
    bounded()
    ref = reference(batches,deadline)
    require(ref.keys() == set(KINDS) and owned_names() == [],'one encoder owner at a time: reference registry survived')
    bounded()
    found = native(ref,deadline)
    bounded()
    tail = bytes.fromhex(ref['TAIL']['outputs']['wire_hex'])
    require(persisted_tail(tail),'reference export tail differs from the persisted export wire rows')
    denial.active = True
    try:
        rejected = mutant_factory(deadline)
        bounded()
        with requests.owner(factory,bounded,charges[0]) as index:
            denial.check()
            installed += installed_pass(index,batches,ref,found,decode,rgb,capture,'A',2,deadline)
            live = mutants(index,deadline)
            bounded()
            installed += installed_pass(index,[batches[0],batches[-1]],ref,found,decode,rgb,capture,'A_restored',1,deadline)
            life = lifecycle_check(index,deadline)
        index = None
        require(owned_names() == [],'second encoder owner preceded complete release')
        bounded()
        with requests.owner(factory,bounded,charges[1]) as index:
            denial.check()
            installed += installed_pass(index,batches,ref,found,decode,rgb,capture,'B',1,deadline)
        index = None
        require(owned_names() == [],'reload owner registry survived')
        bounded()
        proof = denial.check()
    finally:
        denial.active = False
    return {'batches':[{'kind':k,'count':len(p),'ordinals':ordinals if k == 'TAIL' else None,
            'rgb_sha256':ref[k]['rgb_sha256'],'outputs':ref[k]['outputs'],'native':found[k]} for k,p in batches],
        'installed':installed,'factory_mutants':rejected,'live_mutants':live,'lifecycle':life,'denial':proof,
        'persisted_tail_exact':True,'owners':charges,'body_seconds':time.perf_counter()-started,
        'owner_order':['reference','installed_A','installed_B'],'same_batch_exact_only':True,
        'timing_semantics':'unmeasured engineering charges only; no speed claim',
        'product_p99':'UNQUALIFIED; requires10000interleaved paired calls and confidence interval'}


MUTANT_NAMES = {'probe_leaf:head.probe','frozen_leaf','head_parameter','readout_C','readout_mu_train',
    'config_attn_implementation','buffer_registration','same_numel_leaf_substitution','head_requires_grad',
    'processor_config','numerical_flags','cpu_rng','cuda_rng','runtime_function_code','runtime_function_defaults',
    'runtime_global_rebind','runtime_global_added'}
FACTORY_NAMES = {'bundle_sha256','gallery_sha256','native_sha256','gallery_count','mlp_factory_on_probe_bundle'}
FLAGS = ('quality_read','quality_eligible','speed_eligible','release_eligible','qualification_eligible',
    'deployment_eligible','state_reuse_eligible','optimization_eligible','product_go','native_launch_authorized_by_source_pass')
RECEIPT_KEYS = {'schema','status','engineering_only','authority','sources','evaluator','evaluation_authority','endpoint',
    'train_export','bundle','gallery','images','native','wheel','wheel_evidence','native_runtime','native_scope','output','ties',
    'batches','installed','factory_mutants','live_mutants','lifecycle','denial','persisted_tail_exact','owners','body_seconds',
    'owner_order','same_batch_exact_only','timing_semantics','product_p99','installed_public_parity_pass',*FLAGS,
    'resource_policy','normal_terminal_required','owned_cleanup_requires_terminal','invocation','combined_native',
    'full_uncached_exit_pass','resources','input_guards','whole_process_seconds'}
INVOCATION_KEYS = {'argv','python','python_sha256','python_version','pid','invocation_id','optimize','cuda_visible_devices',
    'cublas_workspace_config'}


def cli(authority, output, driver):
    return [driver,'--authority',authority['path'],'--authority-sha256',authority['sha256'],'--output',output]


def validate_receipt(record, authority, authority_fact):
    def seconds(value):
        require(type(value) in (int,float) and math.isfinite(value) and value >= 0,'finite nonnegative measurement required')
        return value
    require(type(record) is dict and record.keys() == RECEIPT_KEYS and type(record['invocation']) is dict and
        record['invocation'].keys() == INVOCATION_KEYS,'exact probe receipt/invocation keyset required')
    require(record['schema'] == RECEIPT and record['status'] == 'ENGINEERING_PARITY_DIAGNOSTIC' and
        record['engineering_only'] is True and record['authority'] == authority_fact and
        record['sources'] == authority['sources'] and
        all(record[k] == authority[k] for k in ('evaluator','evaluation_authority','endpoint','train_export','bundle',
            'gallery','images','native','wheel')) and all(record[k] is False for k in FLAGS) and
        record['installed_public_parity_pass'] is True and record['full_uncached_exit_pass'] is True and
        record['native_scope'] == NATIVE_SCOPE and record['same_batch_exact_only'] is True and
        record['persisted_tail_exact'] is True, 'complete engineering-only probe receipt required')
    require(record['normal_terminal_required'] is True and record['owned_cleanup_requires_terminal'] is True,
        'publication is provisional until normal terminal after owned cleanup')
    require(record['invocation']['argv'] == cli(authority_fact,record['output'],authority['sources']['probe_driver']['path']) and
        record['invocation']['optimize'] == 0 and record['invocation']['cuda_visible_devices'] == '0' and
        record['invocation']['cublas_workspace_config'] == ':4096:8','exact new probe diagnostic CLI/environment required')
    require(policy_ok(record['resource_policy']) and record['resource_policy'] == authority['resource_policy'] and
        record['owner_order'] == ['reference','installed_A','installed_B'],'exact policy/owner order required')
    batches = record['batches']
    require(type(batches) is list and [b['kind'] for b in batches] == list(KINDS) and
        [b['count'] for b in batches][:3] == [1,2,32] and 1 <= batches[3]['count'] <= 32 and
        batches[3]['count'] == len(authority['images']['tail']['files']),'ordered B1/B2/B32/TAIL batches required')
    gallery = authority['gallery']['count']
    for b in batches:
        require(b.keys() == {'kind','count','ordinals','rgb_sha256','outputs','native'} and sha_ok(b['rgb_sha256']) and
            (b['ordinals'] is None) == (b['kind'] != 'TAIL'),'exact batch witness required')
        check_outputs(b['outputs'],b['count'])
        check_native(b['native'],b['count'],gallery)
    by_kind = {b['kind']:b for b in batches}
    shape = [('A',k,c) for k in KINDS for c in (0,1)]+[('A_restored','B1',0),('A_restored','TAIL',0)]+[('B',k,0) for k in KINDS]
    rows = record['installed']
    require(type(rows) is list and [(r['owner'],r['kind'],r['call']) for r in rows] == shape and
        all(r.keys() == {'owner','kind','call','outputs_sha256','native_sha256'} and
            r['outputs_sha256'] == digest(by_kind[r['kind']]['outputs']) and
            r['native_sha256'] == digest(by_kind[r['kind']]['native']) for r in rows),
        'every installed call must equal the same-batch reference exactly')
    def rejected(value, names):
        return type(value) is dict and value.keys() == names and all(type(v) is dict and v.keys() == {'type','message'} and
            v['type'] == 'ValueError' and type(v['message']) is str and 0 < len(v['message']) <= 256 for v in value.values())
    require(rejected(record['factory_mutants'],FACTORY_NAMES) and rejected(record['live_mutants'],MUTANT_NAMES) and
        all(record['live_mutants'][n]['message'] == 'whole-unit RNG changed' for n in ('cpu_rng','cuda_rng')),
        'complete bounded ValueError mutant rejection matrix required')
    evidence, wheel = record['wheel_evidence'], authority['wheel']
    require(type(evidence) is dict and evidence.keys() == {'site_root','distribution','version','record','direct_url','rows'} and
        all(evidence[k] == wheel[k] for k in ('site_root','distribution','version','record','direct_url')) and
        type(evidence['rows']) is dict and evidence['rows'].keys() == set(INSTALLED),'exact installed wheel evidence required')
    require(record['lifecycle'] == {'double_close_safe':True,'post_close_rejected':True,'weakrefs_released':True,
        'processor_cache_empty':True,'registry_clean':True,'mappings_absent':True},'complete lifecycle evidence required')
    denial = record['denial']
    require(type(denial) is dict and denial.keys() == {'checks','historical_modules_absent'} and
        denial['historical_modules_absent'] is True and type(denial['checks']) is int and denial['checks'] >= 3,
        'dependency-denial evidence required')
    require(type(record['owners']) is list and len(record['owners']) == 2,'two sequential installed owner charges required')
    for owner in record['owners']:
        require(owner.keys() == {'admission_seconds','release_seconds'},'exact owner charge required')
        for value in owner.values(): seconds(value)
    require(0 < seconds(record['body_seconds']) <= POLICY['body_seconds'] and
        sum(sum(o.values()) for o in record['owners']) <= record['body_seconds']+1e-6,'owner charges exceed body')
    ties = record['ties']
    require(type(ties) is dict and ties.keys() == {'ascending_ordinal_score_bits_exact','native','gallery'} and
        ties['ascending_ordinal_score_bits_exact'] is True and type(ties['native']) is list and len(ties['native']) == 2,
        'complete native ties required')
    for count,value in zip((1,32),ties['native'],strict=True):
        check_native(value,count)
        require(value[0]['hex'] == (struct.pack('<10q',*range(10))*count).hex() and
            value[1]['hex'] == (struct.pack('<10f',*([1.]*10))*count).hex(),'native tied ordinal/score bits differ')


def admit_probe(evaluator, context, authority):
    """Original export launch/UNIT admission for exactly this arm/seed; nothing historical is reused.

    evaluator.accept_unit CONSUMES the export UNIT invocation (legacy['invocations']), so the context must be
    a fresh export-phase context that has not already accepted that export UNIT ('reused terminal invocation')."""
    ep, launch = authority['endpoint'], context['launch']
    require(ep['arm'] in evaluator.ARMS and ep['seed'] in evaluator.SEEDS and launch['phase'] == 'export' and
        (launch['stage'],launch['panel'],launch['arm'],launch['seed']) == (ep['stage'],ep['panel'],ep['arm'],ep['seed']),
        'original probe export launch authority required')
    unit = authority['train_export']
    exported = evaluator.accept_unit(context,unit,'export',ep['arm'],ep['seed'],stage=ep['stage'],panel=ep['panel'])
    endpoint = next(e for e in launch['endpoints'] if (e['arm'],e['seed']) == (ep['arm'],ep['seed']))
    require(authority['bundle']['manifest'] == endpoint['bundle'],'same original TRAIN/export bundle FILE required')
    return endpoint,exported


def accept_unit(context, unit, authority_fact):
    """Parent's original normal-terminal analogue; prospective pins are the authority's dynamic facts.

    The parent context must be a FRESH export-phase evaluator context: admit_probe re-runs evaluator.accept_unit
    on the export UNIT, which consumes its invocation, so a context that already accepted that export UNIT
    fails closed. The installed wheel (including direct_url presence/absence) is re-checked fresh here."""
    authority = requests.strict_json(requests.read_file(authority_fact))
    check_shape(authority)
    record = requests.strict_json(requests.read_file(unit['receipt']))
    validate_receipt(record,authority,authority_fact)
    require(unit['receipt']['path'] == str(Path(record['output'])/'receipt.json') and
        record['native_runtime'] == authority['native']['authority'] and
        record['invocation']['invocation_id'] == unit['invocation_id'],'diagnostic receipt FILE/output/invocation roles differ')
    sources = authority['sources']
    owned,failures = [],[]
    try:
        observer_source = requests.Source.load(sources['observer']); owned.append(observer_source)
        observer = observer_source.module
        require(record['wheel_evidence'] == check_wheel(authority['wheel'],sources,observer,
            Path(sources['probe_driver']['path']).parent),'installed wheel evidence changed before parent acceptance')
        native_source = requests.Source.load(sources['control_native']); owned.append(native_source)
        native = native_source.module.CombinedAuthority(context['training_context'],authority['native']['authority'],observer,requests)
        require(native.record['library'] == authority['native']['library'],'terminal native FILE differs')
        combined = record['combined_native']; inventory = combined['inventory']
        require(combined.keys() == {'authority','inventory','supplemental_files','historical_projection','mapped_identities'} and
            combined['authority'] == authority['native']['authority'] and combined['supplemental_files'] == native.files and
            combined['mapped_identities'] == {p:list(v) for p,v in native.identities.items()} and
            inventory.keys() == {'files','modules','native_files','packages'} and
            inventory['packages'] == context['training_context']['legacy']['selected']['packages'] and
            type(inventory['native_files']) is list and len(inventory['native_files']) == len(set(inventory['native_files'])) and
            set(inventory['native_files']) <= inventory['files'].keys() and
            native.files.keys() <= set(inventory['native_files']) and
            all({**native.historical['files'],**native.files}.get(p) == h for p,h in inventory['files'].items()) and
            all(native.historical['modules'].get(n) == p for n,p in inventory['modules'].items()),
            'complete combined terminal native evidence differs')
        projected = {**inventory,'files':{p:h for p,h in inventory['files'].items() if p not in native.files},
            'native_files':[p for p in inventory['native_files'] if p not in native.files]}
        legacy = context['training_context']['legacy']
        historical_files = legacy['selected']['source_cpu']['origins']['files'].keys() | legacy['warm_record']['origins']['files'].keys()
        require(combined['historical_projection'] == projected and
            projected['files'].keys()-historical_files == native.supplement['files'].keys() and
            native.supplement['files'].keys() <= set(projected['native_files']), 'original exact-four terminal projection differs')
        required = [authority_fact,authority['native']['authority'],authority['evaluation_authority'],
            *sources.values(),*native.provenance_facts(),authority['bundle']['manifest'],*authority['bundle']['code'].values(),
            *authority['bundle']['files'].values(),authority['gallery']['file'],authority['train_export']['receipt'],
            *authority['images']['fixed32'],*authority['images']['tail']['files'],authority['wheel']['record'],
            *[f for f in (authority['wheel']['direct_url'],) if f is not None]]
        for file in required:
            observer.file_bytes(file)
            require(record['input_guards'].get(file['path']) == file['sha256'], 'terminal omits frozen probe FILE guards')
        fact = authority['evaluator']
        loader = derive_evaluator_loader(requests.read_file(sources['control_native']),vars(native_source.module),sources['control_native']['path'])
        evaluator_source = loader({'path':str(Path(fact['root'])/PROBE_EVALUATOR_NAME),
            'sha256':fact['code'][PROBE_EVALUATOR_NAME]},requests); owned.append(evaluator_source)
        evaluator = evaluator_source.module
        evaluator.check_code(fact,evaluator.FILES)
        require(evaluator.closure(fact['root'],fact['execution_sha256'],evaluator.FILES,{}) == fact['code'] and
            str(context['root']) == fact['root'] and context['args'].execution_sha256 == fact['execution_sha256'] and
            context['code'] == fact['code'] and str(context['args'].authority) == authority['evaluation_authority']['path'] and
            context['args'].authority_sha256 == authority['evaluation_authority']['sha256'], 'terminal original evaluator authority differs')
        admit_probe(evaluator,context,authority)
    except BaseException as error:
        failures.append(error)
    finally:
        for source in reversed(owned):
            if sys.modules.get(source.module.__name__) is source.module:
                del sys.modules[source.module.__name__]
            else: failures.append(ValueError('owned terminal source registry changed'))
        if failures: requests.raise_failures(failures)
    resources = record['resources']; policy = record['resource_policy']
    require(policy_ok(policy),'frozen diagnostic resource policy differs')
    requests.check_resources({**resources,'wall_seconds':record['whole_process_seconds']},policy,reserve=False)
    legacy = context['training_context']['legacy']
    prior = legacy['selected']['source_cpu']['invocation']
    require(all(record['invocation'][k] == prior[k] for k in ('python','python_sha256','python_version')) and
        unit['invocation_id'] not in legacy['invocations'], 'original interpreter/fresh diagnostic invocation required')
    terminal_record = {**record,'wall_seconds':record['whole_process_seconds'],**{k:resources[k] for k in
        ('process_peak_rss_kib','cgroup_before','cgroup_after')}}
    final = context['terminal_reader'](legacy['admission'],terminal_record,unit,policy['whole_process_seconds'],context['guards'])
    for value in (resources['cgroup_before'],resources['cgroup_after'],final): context['helper'].zero_events(value)
    for p,h in record['input_guards'].items():
        require(legacy['extract'].sha(Path(p)) == h, 'current diagnostic input SHA256 differs')
    require(all(record['input_guards'].get(p) == h for p,h in context['common_guards'].items()), 'diagnostic omits original guards')
    legacy['invocations'].add(unit['invocation_id'])
    return record


def run(args):
    require(sys.flags.optimize == 0 and sys.dont_write_bytecode and sys.getprofile() is None,
        'unoptimized -B unprofiled startup required')
    require(not any(n.split('.')[0] in {'torch','numpy','PIL','sfora','transformers','torchvision','safetensors'} for n in sys.modules),
        'native import preceded probe admission')
    authority_fact = {'path':str(args.authority),'sha256':args.authority_sha256}
    authority = requests.strict_json(requests.read_file(authority_fact))
    check_shape(authority)
    sources, output = authority['sources'], args.output
    here = Path(__file__).absolute()
    for role,name in (('probe_driver','qualify_connected_probe_serving.py'),('probe_test','test_connected_probe_serving.py'),
            ('request_driver','qualify_connected_serving_requests.py'),('request_test','test_connected_serving_requests.py'),
            ('observer','observe_connected_serving.py'),('observer_test','test_observe_connected_serving.py'),
            ('control_native','connected_control_native_authority.py')):
        require(sources[role]['path'] == str(here.with_name(name)),'current source role differs: '+role)
    require(sys.argv == cli(authority_fact,str(output),sources['probe_driver']['path']),'fixed canonical probe CLI order required')
    request_source = requests.Source(requests,sources['request_driver'])
    self_source = requests.Source(sys.modules[__name__],sources['probe_driver'])
    require(output.is_absolute() and output.parent.resolve() == output.parent and not output.exists() and not output.is_symlink(),
        'exclusive canonical new output required')
    owned,failures,context,exit_guard,api,before,guard = [],[],None,None,None,None,None
    try:
        locks = requests.Locks(authority['locks'])
        observer_source = requests.Source.load(sources['observer']); owned.append(observer_source)
        observer = observer_source.module
        for fact in sources.values(): observer.file_bytes(fact)
        bundle, native_fact = authority['bundle'], authority['native']
        check_installed(sources,bundle,observer)
        wheel = check_wheel(authority['wheel'],sources,observer,here.parent)
        native_source = requests.Source.load(sources['control_native']); owned.append(native_source)
        native_module = native_source.module
        require(native_module.BINARY_SHA == ARCHIVED_BINARY_SHA == native_fact['library']['sha256'] and
            native_module.ARCHIVE_SHA == ARCHIVED_BUILD_SHA, 'archived control binary/build pins differ; new native authority required')
        runtime = requests.strict_json(observer.file_bytes(native_fact['authority'],keep=True))
        require(runtime['library'] == native_fact['library'],'same frozen native FILE required')
        native_module.validate_runtime_compiler(runtime,observer)
        fact = authority['evaluator']
        loader = derive_evaluator_loader(requests.read_file(sources['control_native']),vars(native_module),sources['control_native']['path'])
        evaluator_source = loader({'path':str(Path(fact['root'])/PROBE_EVALUATOR_NAME),
            'sha256':fact['code'][PROBE_EVALUATOR_NAME]},requests); owned.append(evaluator_source)
        evaluator = evaluator_source.module
        require(evaluator.FILES == EVALUATOR_FILES,'probe evaluator file set differs')
        evaluator.check_code(fact,evaluator.FILES)
        require(evaluator.closure(fact['root'],fact['execution_sha256'],evaluator.FILES,{}) == fact['code'],
            'complete probe evaluator CODE differs')
        ep = authority['endpoint']
        eargs = SimpleNamespace(execution_sha256=fact['execution_sha256'],authority=Path(authority['evaluation_authority']['path']),
            authority_sha256=authority['evaluation_authority']['sha256'],phase='export',arm=ep['arm'],seed=ep['seed'],output=output)
        context,exit_guard = evaluator.authority(eargs)
        context['training_context']['fit_context']['unit_started'] = STARTED
        endpoint,exported = admit_probe(evaluator,context,authority)
        frozen = [authority_fact,native_fact['authority'],native_fact['library'],authority['evaluation_authority'],
            *sources.values(),bundle['manifest'],*bundle['code'].values(),authority['gallery']['file'],
            authority['train_export']['receipt'],authority['wheel']['record'],
            *[f for f in (authority['wheel']['direct_url'],) if f is not None],
            *authority['images']['fixed32'],*authority['images']['tail']['files']]
        evaluator.merge_guards(context['guards'],{f['path']:f['sha256'] for f in frozen})
        runtime_authority = native_module.CombinedAuthority(context['training_context'],native_fact['authority'],observer,requests)
        evaluator.merge_guards(context['guards'],{f['path']:f['sha256'] for f in runtime_authority.provenance_facts()})
        api = runtime_authority.install(evaluator_source,context)
        policy = authority['resource_policy']
        require(policy['whole_process_seconds'] <= evaluator.policy('export')['seconds'] and
            time.perf_counter()-STARTED+policy['body_seconds']+policy['exit_reserve_seconds'] < policy['whole_process_seconds'],
            'insufficient frozen admission/body/exit headroom')
        before = evaluator.native_start(context)
        import torch
        from PIL import Image
        import sfora
        from sfora import cutile_int8,joint_relational_compaction,packed_int8
        check_loaded(wheel['site_root'],sfora)
        wrapper_source = requests.Source(cutile_int8,sources['native_wrapper'])
        packed_source = requests.Source(packed_int8,sources['packed'])
        packing_source = requests.Source(joint_relational_compaction,sources['packing'],packed_source=packed_source)
        bridge_source = requests.Source.load(sources['bridge']); owned.append(bridge_source)
        torch.random.default_generator.manual_seed(ep['seed']); torch.cuda.manual_seed_all(ep['seed'])
        rng,cuda_rng = torch.random.get_rng_state().clone(),torch.cuda.get_rng_state_all()
        def guard(*, reserve=True, deep=False):
            locks.check()
            for source in (self_source,request_source,observer_source,native_source,evaluator_source,
                bridge_source,wrapper_source,packed_source,packing_source): source.check()
            for file in frozen: observer.file_bytes(file)
            if deep:
                for file in bundle['files'].values(): observer.file_bytes(file)
            api.authenticate()
            evaluator.guard_helpers(context)
            resources = evaluator.resources(context,before)
            resources['wall_seconds'] = time.perf_counter()-STARTED
            requests.check_resources(resources,policy,reserve=reserve)
            require(torch.equal(rng,torch.random.get_rng_state()) and all(torch.equal(a,b) for a,b in
                zip(cuda_rng,torch.cuda.get_rng_state_all(),strict=True)),'whole-unit RNG changed')
            return resources
        guard(deep=True)
        ties = requests.native_ties(joint_relational_compaction.PackedInt8Embeddings,
            cutile_int8.CutilePackedInt8Gallery,native_fact['library'],observer)
        api.audit_origins(context['training_context']['legacy'])
        guard()
        directory = Path(bundle['directory'])
        facts = [*authority['images']['fixed32'],*authority['images']['tail']['files']]
        decode = lambda paths: requests.decode_images(observer,Image,facts,paths)
        with evaluator.endpoint_scope(context,endpoint):
            payloads = evaluator.authenticate_payloads(context,endpoint)
            require(payloads == context['cpu']['payload_facts'][evaluator.label(endpoint)],'CPU-qualified probe payload differs')
            rows,mapping = context['nearest_evaluator'].image_rows(context,ep['panel'])
            batches,ordinals = select_batches(authority['images'],rows,mapping,exported['batch_sizes'],evaluator.batch_sizes)
            name = evaluator.label(endpoint)+'.packed.bin'
            wire = requests.read_file({'path':str(Path(exported['output'])/name),'sha256':exported['files'][name]})
            gallery_wire = bind_gallery(wire,rows,mapping,authority['gallery'])
            tail_wire = b''.join(wire[i*130:(i+1)*130] for i in ordinals)
            wire = None
            batches = [(k,[Path(f['path']) for f in v]) for k,v in batches]
            t = context['training_context']
            def reads_only(): return context['evaluator_reference'].bundle_reads_only(context,endpoint)
            def loaded(state): t['live_model'] = weakref.ref(state['model'])
            def after(): t['trainer'].require_no_training(t)
            def reference(items, deadline):
                return reference_outputs(name='_connected_probe_gate_reference_'+uuid.uuid4().hex,directory=directory,
                    bundle_sha=endpoint['bundle']['sha256'],loader=evaluator.load_authenticated,
                    pin=bundle['code'][TRAINER]['sha256'],guards=context['guards'],reads_only=reads_only,batches=items,
                    decode=decode,rgb=rgb_digest,snapshot=lambda v: snapshot_outputs(v,observer.tensor_snapshot),
                    loaded=loaded,after=after,deadline=deadline)
            def native_search(ref, deadline):
                return reference_native(joint_relational_compaction.PackedInt8Embeddings,cutile_int8.CutilePackedInt8Gallery,
                    native_fact['library'],gallery_wire,authority['gallery']['count'],ref,observer,deadline)
            def factory(*, method='from_probe_bundle', **overrides):
                values = dict(factory.base,**overrides)
                return getattr(bridge_source.module.ConnectedCompactIndex,method)(**values)
            factory.base = dict(bundle_dir=directory,expected_bundle_sha256=endpoint['bundle']['sha256'],
                gallery_path=Path(authority['gallery']['file']['path']),expected_gallery_sha256=authority['gallery']['file']['sha256'],
                gallery_count=authority['gallery']['count'],native_library_path=Path(native_fact['library']['path']),
                expected_native_library_sha256=native_fact['library']['sha256'])
            denial = Denial(directory)
            def capture(index, images):
                return captured_search(index,images,lambda v: snapshot_outputs(v,observer.tensor_snapshot),observer.native_snapshot)
            def mutate(index, deadline):
                images, closing = decode(batches[0][1]), []
                try: return live_mutants(index,torch,images,guard,deadline)
                finally:
                    close_all(images,closing)
                    if closing: requests.raise_failures(closing)
            def life(index, deadline):
                return lifecycle(index,decode,batches[0][1],directory,lambda: Path('/proc/self/maps').read_text(),deadline)
            require(time.perf_counter()-STARTED+policy['body_seconds']+policy['exit_reserve_seconds'] < policy['whole_process_seconds'],
                'insufficient whole-process body/exit headroom')
            diagnostic = parity_body(batches=batches,ordinals=ordinals,persisted_tail=lambda w: w == tail_wire,
                reference=reference,native=native_search,factory=factory,decode=decode,rgb=rgb_digest,capture=capture,
                mutants=mutate,
                lifecycle_check=life,denial=denial,guard=guard,started=time.perf_counter(),
                mutant_factory=lambda deadline: factory_mutants(factory,deadline))
        guard(deep=True)
        prior = context['training_context']['legacy']['selected']['source_cpu']['invocation']
        record = {'schema':RECEIPT,'status':'ENGINEERING_PARITY_DIAGNOSTIC','engineering_only':True,'authority':authority_fact,
            'sources':sources,**{k:authority[k] for k in ('evaluator','evaluation_authority','endpoint','train_export','bundle',
                'gallery','images','native','wheel')},'wheel_evidence':wheel,'native_runtime':native_fact['authority'],
            'native_scope':NATIVE_SCOPE,'output':str(output),'ties':ties,**diagnostic,'installed_public_parity_pass':True,
            **dict.fromkeys(FLAGS,False),'resource_policy':policy,
            'normal_terminal_required':True,'owned_cleanup_requires_terminal':True,
            'invocation':{'argv':sys.argv,'python':str(Path(sys.executable).resolve()),'python_sha256':prior['python_sha256'],
                'python_version':sys.version,'pid':os.getpid(),'invocation_id':os.environ['INVOCATION_ID'],
                'optimize':sys.flags.optimize,'cuda_visible_devices':os.environ['CUDA_VISIBLE_DEVICES'],
                'cublas_workspace_config':os.environ.get('CUBLAS_WORKSPACE_CONFIG')}}
    except BaseException as error:
        failures.append(error)
    finally:
        if context is not None:
            try:
                if api is None: evaluator.exit_rehash(context,exit_guard)
                else: api.evaluator_exit(context,exit_guard)
                if guard is not None: final_resources = guard(reserve=False,deep=True)
            except BaseException as error: failures.append(error)
        if not failures:
            try:
                record['combined_native'] = api.evidence()
                final_resources = guard(reserve=False,deep=True)
                require(check_wheel(authority['wheel'],sources,observer,here.parent) == wheel,
                    'installed wheel evidence changed before exit')
                record['full_uncached_exit_pass'] = True
                record['resources'] = final_resources
                record['input_guards'] = dict(context['guards'])
                record['whole_process_seconds'] = time.perf_counter()-STARTED
                validate_receipt(record,authority,authority_fact)
                requests.check_resources({**final_resources,'wall_seconds':record['whole_process_seconds']},policy,reserve=False)
                locks.check()
                output.mkdir()
                context['helper'].publish(output/'receipt.json',record)
            except BaseException as error: failures.append(error)
        for source in reversed(owned):
            if sys.modules.get(source.module.__name__) is source.module: del sys.modules[source.module.__name__]
            else: failures.append(ValueError('owned source registry changed at exit'))
        if not failures:
            try:
                locks.check()
                require(time.perf_counter()-STARTED < policy['whole_process_seconds'],'publication/cleanup included whole-process cap exceeded')
            except BaseException as error: failures.append(error)
        if failures: requests.raise_failures(failures)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--authority',type=Path,required=True)
    parser.add_argument('--authority-sha256',required=True)
    parser.add_argument('--output',type=Path,required=True)
    return run(parser.parse_args(argv))


# BEGIN bounded loader observation
import types as _observation_types
import functools as _observation_functools


def _observation_json(value, limit=65536):
    chunks, size = [], 0
    for chunk in json.JSONEncoder(ensure_ascii=True, separators=(',',':')).iterencode(value):
        size += len(chunk)
        if size > limit: raise ValueError('encoded observation quota')
        chunks.append(chunk)
    return ''.join(chunks)


def _observation_emit(record):
    try:
        print('LOADER_OBSERVATION '+_observation_json(record,196608),flush=True)
    except BaseException:
        # A broken diagnostic sink must not replace the original computation failure.
        pass


def _observation_text(value):
    value = str(value)
    if len(value) > 65536: raise ValueError('observation text quota')
    return value


def _observation_type(value):
    cls = type(value)
    module,name = (type.__getattribute__(cls,key) for key in ('__module__','__qualname__'))
    if type(module) is not str or type(name) is not str or len(module)+len(name) > 1024:
        raise ValueError('type metadata quota')
    return module+'.'+name


def _observation_message(error):
    # Calling custom __str__, or repr on an arbitrary exception argument, can change the workload.
    method = type.__getattribute__(type(error),'__str__')
    if isinstance(error,BaseExceptionGroup):
        if method is not BaseExceptionGroup.__str__: raise ValueError('unsupported group formatter')
        return _observation_text(error)
    if method not in (BaseException.__str__,OSError.__str__,KeyError.__str__):
        raise ValueError('unsupported exception formatter')
    args = BaseException.args.__get__(error)
    if len(args) > 64 or any(type(v) not in (str,int,float,bool,type(None)) for v in args):
        raise ValueError('unsupported exception arguments')
    if any(type(v) is str and len(v) > 65536 for v in args): raise ValueError('message quota')
    if method is OSError.__str__:
        for name in ('errno','strerror','filename','filename2'):
            value = OSError.__dict__[name].__get__(error)
            if type(value) not in (str,int,type(None)) or (type(value) is str and len(value) > 65536):
                raise ValueError('unsupported OS error metadata')
    return _observation_text(error)


def _observation_errors(error):
    record = {'diagnostic_complete':True,'root':None,'nodes':[],'edges':[]}
    pending, seen, traces = [], {}, 0
    current = trace = child = None
    try:
        def number(value):
            if id(value) not in seen:
                if len(seen) == 64: raise ValueError('exception node quota')
                seen[id(value)] = len(seen)
                pending.append(value)
            return seen[id(value)]
        if error is not None: record['root'] = number(error)
        while pending:
            current = pending.pop(0)
            notes = BaseException.__dict__['__dict__'].__get__(current).get('__notes__',[])
            if type(notes) is not list or len(notes) > 64 or any(type(n) is not str for n in notes):
                raise ValueError('exception notes quota/type')
            node = {'id':seen[id(current)],'type':_observation_type(current),
                'message':_observation_message(current),
                'notes':[_observation_text(n) for n in notes],
                'suppress_context':BaseException.__suppress_context__.__get__(current),'traceback':[]}
            trace = BaseException.__traceback__.__get__(current)
            while trace is not None:
                traces += 1
                if traces > 512: raise ValueError('exception traceback quota')
                node['traceback'].append({'file':trace.tb_frame.f_code.co_filename,
                    'function':trace.tb_frame.f_code.co_name,'line':trace.tb_lineno})
                trace = trace.tb_next
            record['nodes'].append(node)
            for kind,child in (('cause',BaseException.__cause__.__get__(current)),
                               ('context',BaseException.__context__.__get__(current))):
                if child is not None:
                    record['edges'].append({'from':node['id'],'to':number(child),'kind':kind})
            if isinstance(current,BaseExceptionGroup):
                members = BaseExceptionGroup.exceptions.__get__(current)
                if len(members) > 64: raise ValueError('exception member quota')
                for child in members:
                    record['edges'].append({'from':node['id'],'to':number(child),'kind':'member'})
            _observation_json(record)
        return record
    except BaseException:
        return {'diagnostic_complete':False,'reason':'exception inspection or quota failure'}
    finally:
        pending.clear()
        error = current = trace = child = None


def _observation_maps(path, identity):
    stat = path.stat()
    if (stat.st_dev,stat.st_ino) != identity: raise ValueError('endpoint inode changed')
    result = []
    with Path('/proc/self/maps').open() as stream:
        for count,line in enumerate(stream):
            if count >= 32768 or len(line) > 8192: raise ValueError('maps quota')
            address,mode,offset,device,inode,*_ = line.split(maxsplit=5)
            major,minor = (int(v,16) for v in device.split(':'))
            if (major,minor,int(inode)) == (os.major(stat.st_dev),os.minor(stat.st_dev),stat.st_ino):
                if len(result) == 64: raise ValueError('mapping quota')
                start,end = (int(v,16) for v in address.split('-'))
                result.append({'start':start,'end':end,'offset':int(offset,16),'mode':mode})
    return result


def _observation_dict(value):
    if type(value) is dict: return value
    if type(value) is type(sys._getframe().f_locals): return value
    if type(value) is _observation_types.ModuleType: return value.__dict__
    # Only CPython's instance-dictionary descriptor; never user properties or __getattribute__.
    for cls in type.__getattribute__(type(value),'__mro__'):
        descriptor = type.__getattribute__(cls,'__dict__').get('__dict__')
        if descriptor is not None:
            if type(descriptor) is _observation_types.GetSetDescriptorType:
                result = descriptor.__get__(value,type(value))
                return result if type(result) is dict else None
            return None
    return None


def _observation_member(cls, name):
    for base in type.__getattribute__(cls,'__mro__'):
        namespace = type.__getattribute__(base,'__dict__')
        if name in namespace: return namespace[name]
    raise ValueError('unsupported native metadata member')


def _observation_storage(value, torch):
    tensor,storage = torch.__dict__['Tensor'],torch.__dict__['UntypedStorage']
    if (not isinstance(tensor,type) or not isinstance(storage,type) or
            type.__getattribute__(tensor,'__module__') != 'torch' or
            type.__getattribute__(storage,'__module__') != 'torch.storage'):
        raise ValueError('unsupported native tensor descriptors')
    untyped,device_get,shape_get,dtype_get = (_observation_member(tensor,key)
        for key in ('untyped_storage','device','shape','dtype'))
    pointer,nbytes,storage_device = (_observation_member(storage,key) for key in ('data_ptr','nbytes','device'))
    if (any(type(method) is not _observation_types.MethodDescriptorType for method in (untyped,pointer,nbytes)) or
            any(type(member) is not _observation_types.GetSetDescriptorType
                for member in (device_get,shape_get,dtype_get,storage_device))):
        raise ValueError('unsupported native metadata descriptors')
    parameter_module = sys.modules.get('torch.nn.parameter')
    parameter = (parameter_module.__dict__.get('Parameter')
        if type(parameter_module) is _observation_types.ModuleType else None)
    if type(value) not in (tensor,storage,parameter): return None
    owned = None
    try:
        if type(value) is storage:
            owned = value
            shape,dtype,device = None,None,storage_device.__get__(owned)
        else:
            device = device_get.__get__(value)
            if device.type != 'cpu': return None
            owned = untyped(value)
            if type(owned) is not storage: raise ValueError('unsupported storage type')
            shape = list(shape_get.__get__(value))
            if len(shape) > 32: raise ValueError('shape quota')
            dtype = str(dtype_get.__get__(value))
        if device.type != 'cpu': return None
        address,size = pointer(owned),nbytes(owned)
        if type(address) is not int or type(size) is not int or address < 0 or size < 0:
            raise ValueError('unsupported storage range')
        return {'object_id':id(value),'type':_observation_type(value),'storage_id':id(owned),
            'address':address,'end':address+size,'bytes':size,'shape':shape,'dtype':dtype,'device':'cpu'}
    finally:
        owned = value = torch = parameter_module = None


def _observation_owners(frame, error, mappings):
    record = {'diagnostic_complete':False,'candidates':[],'roots':[],
        'limitations':['GC enumeration has no bounded streaming API; global snapshot not allocated',
                      'native/C++ ownership is not observable'], 'owner':'unresolved'}
    seen,errors,trace_roots = set(),[],[]
    value = current = trace = parent = torch = namespace = source_frame = None
    edges = 0
    try:
        torch = sys.modules.get('torch')
        if type(torch) is not _observation_types.ModuleType:
            raise ValueError('loaded torch module unavailable')
        # The loader has already authenticated these package files, without a diagnostic rehash.
        guards = frame.f_locals.get('guards',{})
        torch_path = torch.__dict__.get('__file__')
        if type(guards) is not dict or type(torch_path) is not str or not sha_ok(guards.get(torch_path)):
            raise ValueError('torch origin not in authenticated loader guards')

        def walk(value, path, depth=0):
            nonlocal edges
            edges += 1
            if edges > 512: raise ValueError('owner edge quota')
            if depth > 8: raise ValueError('owner depth quota')
            if type(value) in (str,bytes,int,float,bool,type(None)): return
            if isinstance(value,type) or type(value) in (_observation_types.FunctionType,
                    _observation_types.CodeType,_observation_types.BuiltinFunctionType,_observation_types.ModuleType):
                return
            candidate = _observation_storage(value,torch)
            if candidate is not None:
                if any(candidate['address'] < row['end'] and candidate['end'] > row['start'] for row in mappings):
                    if len(record['candidates']) == 64: raise ValueError('candidate quota')
                    record['candidates'].append({**candidate,'path':path})
                    record['owner'] = 'observed Python retaining path; other owners unresolved'
                return
            if id(value) in seen: return
            seen.add(id(value))
            if type(value) in (tuple,list):
                for index,item in enumerate(value): walk(item,path+'['+str(index)+']',depth+1)
                return
            namespace = _observation_dict(value)
            if namespace is not None:
                for key,item in namespace.items():
                    if type(key) is not str or len(key) > 256: raise ValueError('owner key unsupported')
                    if key == '__builtins__': continue
                    walk(item,path+'.'+key,depth+1)
            else:
                if 'unsupported object type' not in record['limitations']:
                    record['limitations'].append('unsupported object type')

        def root(value, label):
            record['roots'].append(label)
            seen.clear()
            walk(value,label)

        if error is not None: errors.append(error)
        error_seen,trace_count = set(),0
        while errors:
            current = errors.pop()
            if id(current) in error_seen: continue
            if len(error_seen) == 64: raise ValueError('owner exception quota')
            error_seen.add(id(current))
            trace = BaseException.__traceback__.__get__(current)
            while trace is not None:
                trace_count += 1
                if trace_count > 512: raise ValueError('owner traceback quota')
                trace_roots.append((trace.tb_frame,'traceback:'+trace.tb_frame.f_code.co_name+':'+str(trace.tb_lineno)))
                trace = trace.tb_next
            for value in (BaseException.__cause__.__get__(current),BaseException.__context__.__get__(current)):
                if value is not None: errors.append(value)
            if isinstance(current,BaseExceptionGroup):
                members = BaseExceptionGroup.exceptions.__get__(current)
                if len(members) > 64: raise ValueError('owner member quota')
                errors.extend(members)
        # Mapped arguments precede large environment/guard dictionaries in the finite traversal budget.
        for source_frame,label in trace_roots:
            for name in ('value','overlay','buffers','tensors','disk'):
                if name in source_frame.f_locals: root(source_frame.f_locals[name],label+'.'+name)
        for source_frame,label in trace_roots: root(source_frame.f_locals,label)
        root(frame.f_locals,'loader_local')
        modules = frame.f_locals.get('modules',{})
        if type(modules) is dict and len(modules) <= 16:
            for name,value in modules.items():
                if type(value) is _observation_types.ModuleType and sys.modules.get(value.__name__) is value:
                    source = value.__dict__.get('__file__')
                    if type(source) is str and sha_ok(guards.get(source)):
                        root(value.__dict__,'copied_module:'+name)
        parent = frame.f_back
        for _ in range(16):
            if parent is None: break
            for name in ('t','context'):
                if name in parent.f_locals: root(parent.f_locals[name],'parent:'+name)
            parent = parent.f_back
        cache = frame.f_locals.get('cache')
        if type(cache) is _observation_functools._lru_cache_wrapper:
            info = _observation_functools._lru_cache_wrapper.cache_info(cache)
            record['processor_cache'] = {'hits':info.hits,'misses':info.misses,'maxsize':info.maxsize,'currsize':info.currsize}
        record['edges_visited'] = edges
        _observation_json(record)
        return record
    except BaseException:
        record.update(reason='owner inspection or quota failure',edges_visited=edges)
        try:
            _observation_json(record)
            return record
        except BaseException:
            return {'diagnostic_complete':False,'owner':'unresolved','reason':'owner encoded quota failure'}
    finally:
        errors.clear(); trace_roots.clear()
        frame = error = value = current = trace = parent = torch = namespace = source_frame = None


class _LoaderObservation:
    """Scalar-only witness of authenticated original code; never owns loader tensors or frames."""
    def __init__(self, portable, name, directory, pin, guards, registry):
        self.active = self.eligible = False
        self.previous = self.profile = None
        self.points, self.seen = {},set()
        try:
            path = Path(directory)/TRAINER
            if (type(portable) is not _observation_types.ModuleType or type(registry) is not dict or
                    registry.get(name) is not portable or sys.modules.get(name) is not portable or
                    portable.__dict__.get('__name__') != name or portable.__dict__.get('__file__') != str(path) or
                    portable.__dict__.get('__spec__') is None or portable.__spec__.origin != str(path) or
                    pin != dict(HISTORICAL)[TRAINER] or guards.get(str(path)) != pin):
                raise ValueError('copied source authentication binding')
            with path.open('rb') as stream: raw = stream.read(262145)
            if len(raw) > 262144: raise ValueError('source quota')
            if hashlib.sha256(raw).hexdigest() != pin: raise ValueError('copied source bytes changed')
            tree = ast.parse(raw,filename=str(path))
            compiled = compile(raw,str(path),'exec')
            for function,filename in (('load_inference','endpoint.pt'),('construct_encoder','vision.pt')):
                nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name == function]
                if len(nodes) != 1: raise ValueError('source function cardinality')
                node = nodes[0]
                targets = [n for n in ast.walk(node) if isinstance(n,ast.Try) and
                    [ast.unparse(v) for v in n.finalbody] == ['del disk','gc.collect()','mapping_absent(path)']]
                if len(targets) != 1: raise ValueError('original mapping seam differs')
                code = [c for c in compiled.co_consts if type(c) is _observation_types.CodeType and c.co_name == function]
                method = portable.__dict__.get(function)
                if (len(code) != 1 or type(method) is not _observation_types.FunctionType or
                        method.__globals__ is not portable.__dict__ or method.__code__ != code[0] or
                        method.__code__.co_filename != str(path)):
                    raise ValueError('original code/global/line binding')
                checkpoint = Path(directory)/filename
                stat = checkpoint.stat()
                sha = guards.get(str(checkpoint))
                if not sha_ok(sha): raise ValueError('checkpoint authentication binding')
                self.points[id(method.__code__)] = {'function':function,'line':targets[0].finalbody[-1].lineno,
                    'path':str(checkpoint),'identity':(stat.st_dev,stat.st_ino),'sha256':sha}
            self.module_id,self.globals_id,self.name = id(portable),id(portable.__dict__),name
            self.source,self.pin = str(path),pin
            self.eligible = True
        except BaseException:
            _observation_emit({'event':'observation_ineligible','diagnostic_complete':False,
                'reason':'source/type/auth/AST contract failed'})

    def start(self):
        self.previous,self.profile = sys.gettrace(),sys.getprofile()
        if not self.eligible or self.previous is not None:
            _observation_emit({'event':'observation_ineligible','diagnostic_complete':False,
                'reason':'source contract or existing trace'})
            return
        try:
            self.active = True
            sys.settrace(self.trace)
        except BaseException:
            self.close()

    def trace(self, frame, event, arg):
        try:
            point = self.points.get(id(frame.f_code))
            if point is None or id(frame.f_globals) != self.globals_id: return None
            if event != 'line' or frame.f_lineno != point['line']: return self.trace
            if point['function'] in self.seen: return self.trace
            self.seen.add(point['function'])
            error = sys.exception()
            record = {'event':'before_original_mapping_guard','function':point['function'],
                'source':self.source,'source_sha256':self.pin,'line':point['line'],'checkpoint':point['path'],
                'code_id':id(frame.f_code),'globals_id':id(frame.f_globals),
                'errors':_observation_errors(error),'diagnostic_complete':False}
            try:
                module = sys.modules.get(self.name)
                if (id(module) != self.module_id or id(module.__dict__) != self.globals_id or
                        module.__dict__[point['function']].__code__ is not frame.f_code or
                        str(frame.f_locals['path']) != point['path'] or
                        frame.f_locals['guards'].get(point['path']) != point['sha256']):
                    raise ValueError('live registry/path/auth binding changed')
                mappings = _observation_maps(Path(point['path']),point['identity'])
                owners = _observation_owners(frame,error,mappings)
                record.update({'binding_complete':True,'device':point['identity'][0],'inode':point['identity'][1],
                    'mappings':mappings,'locals_present':{k:k in frame.f_locals for k in
                        ('disk','endpoint','copied','pages','model','head','cache')},
                    'owners':owners})
                record['diagnostic_complete'] = record['errors']['diagnostic_complete'] and owners['diagnostic_complete']
            except BaseException:
                record['reason'] = 'live binding/maps inspection or quota failure'
            _observation_emit(record)
            return self.trace
        except BaseException:
            _observation_emit({'event':'observation_incomplete','diagnostic_complete':False})
            return self.trace
        finally:
            frame = arg = None

    def close(self):
        try:
            if self.active:
                sys.settrace(self.previous)
                if sys.getprofile() is not self.profile: sys.setprofile(self.profile)
        except BaseException:
            _observation_emit({'event':'observation_incomplete','diagnostic_complete':False,'reason':'hook restoration failed'})
        finally:
            self.active = False
            self.previous = self.profile = None
# END bounded loader observation


if __name__ == '__main__':
    main()
