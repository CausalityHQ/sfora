#!/usr/bin/env python3
"""Deterministic source-only extraction of the accepted connected probe loader."""

import argparse
import ast
import builtins
import difflib
import hashlib
import json
from pathlib import Path
import pprint
import symtable

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / 'src/sfora'
TRAINER = 'scripts/train_siglip2_connected_probe.py'
EXTRA = ('_probe_decorators', '_probe_vision_forward', 'probe_source')
PINS = {
    'src/sfora/connected_inference.py': 'ea73430a5cc1a7f7ee769dbd37d27dc9379b03896bcab7e9ae3a7b771a42b9eb',
    'src/sfora/_connected_inference_authority.py': 'd1e23c4527794e9a2940a919547512fa779f3dc20ce8ff5ee807a124f623791b',
    'scripts/train_siglip2_connected_probe.py': 'e2496033a958cc83bf8d3204c87b4d3b8284528f03f44c1cb64517df04c8cc1e',
    'scripts/test_siglip2_connected_probe.py': '4accfd6c276ef2ca3d85b25c98ead7d013972d260656266b1227dd59648f0120',
    'docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-probe-cpu-v2-freeze/source-receipt.json':
        '2e933cabd22f5d1a1fd47b413f37ff94040d0167d1e65fc07870708c7bbd78f5',
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def astsha(source):
    return sha(ast.dump(ast.parse(source), include_attributes=False).encode())


def literals(source):
    return {n.targets[0].id: ast.literal_eval(n.value) for n in ast.parse(source).body
            if isinstance(n, ast.Assign)}


def definitions(source):
    nodes = [n for n in ast.parse(source).body if isinstance(n, (ast.FunctionDef, ast.ClassDef))]
    assert len({n.name for n in nodes}) == len(nodes), 'duplicate source definition'
    return {n.name: ast.get_source_segment(source, n) for n in nodes}


def extract(source, name, plumbing):
    """The original finite extraction, with the accepted probe self-source name."""
    counts = []
    def replace(before, after, count=None):
        nonlocal source
        actual = source.count(before)
        assert count is None or actual == count, (name, before, actual, count)
        counts.append((before, after, actual))
        source = source.replace(before, after)
    if name == 'construct_encoder':
        start = source.index('        if source_runtime is not None:')
        end = source.index("        require('position_ids'", start)
        replace(source[start:end], '', 1)
        start = source.index('        require(source_runtime is None or structure')
        end = source.index('        base =', start)
        replace(source[start:end], '', 1)
    if name == 'construct':
        replace("path in context['guards'] and context['extract'].sha(path) == context['guards'][path]",
                "path in context['guards']", 1)
        replace("                'loaded vision/config source hash differs')",
                "                'loaded vision/config source hash differs')\n        bound_file({},path,context['guards'][path])", 1)
    if name == 'admit_bundle':
        replace('    directory,guards = Path(directory),{}',
                '    _check_runtime()\n    directory,guards = Path(directory),{}', 1)
        replace("require(hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == manifest['code']['train_siglip2_connected_probe.py'],\n            'current copied public loader source differs')",
                "require(tuple(sorted(manifest['code'].items())) == _binding[0],\n            'unsupported historical inference closure')", 1)
        replace('    return manifest,guards',
                '    for path,sha in _binding[1]:\n        bound_file(guards,path,sha)\n    return manifest,guards', 1)
    if name == 'load_inference':
        start = source.index('    modules,prefix =')
        end = source.index('    import torch', start)
        replace(source[start:end], "    modules = {'runtime':sys.modules[__name__]}\n", 1)
        replace("{'packages':env['packages'],'guards':guards,'extract':extract,",
                "{'packages':env['packages'],'guards':guards,", 1)
        replace("modules['train_siglip2_cached_readout.py'].head_from", 'head_from', 1)
        replace("'manifest':manifest}", "'manifest':manifest,'directory':directory}", 1)
        replace('Copied public loader uses only owned bundle files and installed packages.',
                'Installed public loader uses authenticated historical evidence without executing it.', 1)
    if name == 'inference_outputs':
        replace("    original,source = modules['train_siglip2_substrate_adaptation.py'],modules['qualify_siglip2_substrate_cpu.py']\n",
                '    _check_runtime()\n', 1)
        replace('    require(0 < len(images)',
                "    for filename in SERVING_FILES | {'joint_relational_compaction.py'}:\n        path = endpoint['directory']/filename\n        bound_file({},path,endpoint['guards'][str(path)])\n    require(0 < len(images)", 1)
        replace("endpoint['mu_train'],endpoint['arm'],modules['quadratic_readout.py'],modules['prototype_residual_readout.py'])",
                "endpoint['mu_train'],endpoint['arm'])", 1)
        replace("modules['joint_relational_compaction.py'].pack_int8_unit_embeddings(unit.cpu())", '_pack(unit.cpu())', 1)
    if name == '_check_base':
        replace('Path(method.__code__.co_filename).name == "train_siglip2_cached_readout.py"',
                'method.__code__ == _head_method_code(name) and method.__globals__ is globals() and\n                 method.__defaults__ is None and method.__kwdefaults__ is None and\n                 Path(method.__code__.co_filename) == Path(__file__)', 1)
    for before, after in plumbing:
        replace(before, after)
    return source, tuple(counts)


def closed(source, expected):
    tree = ast.parse(source)
    imports = {'copy', 'gc', 'hashlib', 'inspect', 'json', 'math', 'os', 're', 'sys',
               'weakref', 'concurrent.futures', 'functools', 'pathlib', 'types'}
    assert definitions(source).keys() == expected, 'unexpected or missing runtime definitions'
    for node in tree.body:
        assert type(node) in {ast.Import, ast.ImportFrom, ast.Assign, ast.FunctionDef, ast.ClassDef, ast.Expr}
        if isinstance(node, ast.Import):
            assert all(a.name in imports for a in node.names)
        if isinstance(node, ast.ImportFrom):
            assert node.module in imports
        if isinstance(node, ast.Expr):
            assert type(node.value) is ast.Constant and type(node.value.value) is str
    table = symtable.symtable(source, 'connected_probe_inference.py', 'exec')
    defined = set(table.get_identifiers()) | {'__file__', '__name__'} | set(vars(builtins))
    def walk(scope):
        for symbol in scope.get_symbols():
            assert not symbol.is_referenced() or not symbol.is_global() or symbol.get_name() in defined, symbol.get_name()
        for child in scope.get_children():
            walk(child)
    walk(table)


def generate():
    for path, digest in PINS.items():
        assert sha((ROOT / path).read_bytes()) == digest, 'unaccepted source bytes: ' + path
    authority = literals((PACKAGE / '_connected_inference_authority.py').read_text())
    oracle = ast.parse((ROOT / 'scripts/test_connected_inference_extraction.py').read_text())
    inventory = {n.targets[0].id: ast.literal_eval(n.value) for n in oracle.body
                 if isinstance(n, ast.Assign) and n.targets[0].id in {'CLOSURE', 'PLUMBING'}}
    assert {(path, name) for path, names in inventory['CLOSURE'].items() for name in names} == {
        (path, name) for path, digest, name, *_ in authority['SOURCE_SYMBOLS']}, 'unadmitted extraction inventory'
    closure = {TRAINER if path == 'scripts/train_siglip2_connected_mlp.py' else path: names
               for path, names in inventory['CLOSURE'].items()}
    closure[TRAINER] += EXTRA
    assert sum(map(len, closure.values())) == 41
    # Common source pins are authenticated by the unchanged installed MLP authority.
    for path, digest, *_ in authority['SOURCE_SYMBOLS']:
        assert sha((ROOT / path).read_bytes()) == digest, 'common source differs: ' + path
    assert sha((PACKAGE / 'packed_int8.py').read_bytes()) == authority['PACKED_SHA256']
    receipt = json.loads((ROOT / next(p for p in PINS if p.endswith('source-receipt.json'))).read_text())
    assert receipt['structured_contract']['files'] == {Path(p).name: PINS[p] for p in (TRAINER, 'scripts/test_siglip2_connected_probe.py')}
    historical = tuple(sorted([(name, digest) for name, digest in authority['HISTORICAL_CODE']
                               if name not in {'train_siglip2_connected_mlp.py', 'test_siglip2_connected_mlp.py'}]
                              + list(receipt['structured_contract']['files'].items())))
    current = (PACKAGE / 'connected_inference.py').read_text()
    current_defs = definitions(current)
    header = current[:current.index('def _bind_runtime(')]
    header = header.replace('siglip2-connected-mlp-', 'siglip2-connected-probe-')
    header = header.replace('train_siglip2_connected_mlp.py', 'train_siglip2_connected_probe.py').replace(
        'test_siglip2_connected_mlp.py', 'test_siglip2_connected_probe.py')
    start = header.index('MLP =')
    end = header.index('BACKEND_SHA =', start)
    trainer_defs = definitions((ROOT / TRAINER).read_text())
    trainer_tree = ast.parse((ROOT / TRAINER).read_text())
    constants = [ast.get_source_segment((ROOT / TRAINER).read_text(), n) for n in trainer_tree.body
                 if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
                 and n.targets[0].id in {'PROBE', 'PROBE_SHAPES', 'PROBE_SOURCE_SHA'}]
    header = header[:start] + '\n\n\n'.join(constants) + '\n\n\n' + header[end:]
    sources, records, differences, counts = {}, [], [], []
    for path, names in closure.items():
        raw = (ROOT / path).read_text()
        defs = trainer_defs if path == TRAINER else definitions(raw)
        for name in names:
            original = defs[name]
            extracted, counted = extract(original, name, inventory['PLUMBING'])
            sources[name] = extracted
            records.append((path, sha(raw.encode()), name, sha(original.encode()), astsha(original),
                            sha(extracted.encode()), astsha(extracted)))
            counts.append((name, counted))
            if original != extracted:
                differences.append((name, ''.join(difflib.unified_diff(original.splitlines(True), extracted.splitlines(True),
                                   fromfile=path+':'+name, tofile='connected_probe_inference.py:'+name))))
    helper_names = ('_bind_runtime', '_check_runtime', '_head_method_code', '_pack')
    helpers = {name: current_defs[name] for name in helper_names}
    helpers['_bind_runtime'] = helpers['_bind_runtime'].replace('"connected_inference.py"', '"connected_probe_inference.py"').replace(
        '"_connected_inference_authority.py"', '"_connected_probe_inference_authority.py"')
    order = [n for n in current_defs if n not in {'_sha_cpu_bytes', '_fingerprint_cuda_dict'}]
    order[order.index('model_structure'):order.index('model_structure')] = EXTRA
    base = header + '\n\n\n'.join((helpers | sources)[name] for name in order) + '\n'
    encoder_before = sources['encoder_facts']
    replacements = (
        ('    frozen = fingerprint({n:p for n,p in params.items() if n not in PROBE})',
         '    tensor_hash = _fingerprint_cuda_dict if serving and state["device"] == "cuda" else fingerprint\n    frozen = tensor_hash({n:p for n,p in params.items() if n not in PROBE})'),
        ("return {'vision_sha256':fingerprint(model.state_dict()),", "return {'vision_sha256':tensor_hash(model.state_dict()),"),
    )
    for before, after in replacements:
        assert sources['encoder_facts'].count(before) == 1
        sources['encoder_facts'] = sources['encoder_facts'].replace(before, after)
    hash_helpers = ('_sha_cpu_bytes', '_fingerprint_cuda_dict')
    helpers.update({name: current_defs[name] for name in hash_helpers})
    order[order.index('fingerprint'):order.index('fingerprint')] = hash_helpers
    runtime = header + '\n\n\n'.join((helpers | sources)[name] for name in order) + '\n'
    closed(runtime, {n for names in closure.values() for n in names} | set(helpers))
    for name, byte_sha, ast_sha in authority['SUBSTITUTIONS'][-1][1]['helpers']:
        assert (sha(helpers[name].encode()), astsha(helpers[name])) == (byte_sha, ast_sha)
    pipeline = {'base_runtime_sha256': sha(base.encode()), 'replacements': replacements,
                'helpers': authority['SUBSTITUTIONS'][-1][1]['helpers'],
                'encoder': (sha(sources['encoder_facts'].encode()), astsha(sources['encoder_facts'])),
                'encoder_diff': ''.join(difflib.unified_diff(encoder_before.splitlines(True), sources['encoder_facts'].splitlines(True),
                                       fromfile='original:encoder_facts', tofile='fresh-sha:encoder_facts'))}
    record = {'SCHEMA': 'sfora-connected-probe-inference-extraction-v1', 'HISTORICAL_CODE': historical,
              'SOURCE_SYMBOLS': tuple(records), 'PACKED_SOURCE_SYMBOLS': authority['PACKED_SOURCE_SYMBOLS'],
              'SUBSTITUTIONS': (*differences, ('_counted_plumbing', tuple(counts)), ('_fresh_cpu_sha_pipeline', pipeline)),
              'RUNTIME_SHA256': sha(runtime.encode()), 'PACKED_SHA256': authority['PACKED_SHA256']}
    ledger = '"Literal probe extraction ledger; installed/native qualification remains parent-owned."\n\n' + '\n\n'.join(
        key + ' = ' + pprint.pformat(value, width=100, sort_dicts=False) for key, value in record.items()) + '\n'
    return {'connected_probe_inference.py': runtime, '_connected_probe_inference_authority.py': ledger}


def main():
    if not __debug__:
        raise SystemExit('extraction requires assertions')
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    for name, source in generate().items():
        path = PACKAGE / name
        if args.check:
            assert path.is_file() and path.read_bytes() == source.encode(), 'generated bytes differ: ' + name
        else:
            path.write_bytes(source.encode())
    print('connected probe extraction ' + ('matches' if args.check else 'written'))


if __name__ == '__main__':
    main()
