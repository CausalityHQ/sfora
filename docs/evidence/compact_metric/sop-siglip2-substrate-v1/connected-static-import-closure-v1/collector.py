"""Read-only candidate import graph; RECORD membership is not execution permission."""
import ast
import base64
import csv
import hashlib
import io
import json
import stat
import time
from pathlib import Path


def read_exact(path, expected, size=None):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError('canonical regular file required')
    before = path.stat()
    if before.st_size > 16 * 1024 * 1024:
        raise ValueError('source/RECORD exceeds byte bound')
    raw = path.read_bytes()
    after = path.stat()
    identity = lambda s: (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns)
    if identity(before) != identity(after) or len(raw) != before.st_size:
        raise ValueError('file changed during read')
    if hashlib.sha256(raw).hexdigest() != expected or (size is not None and len(raw) != size):
        raise ValueError('original RECORD byte identity differs')
    return raw


def module_name(relative):
    if not relative.endswith('.py'):
        return None
    parts = relative[:-3].split('/')
    if parts[-1] == '__init__':
        parts.pop()
    return '.'.join(parts) if parts and all(p.isidentifier() for p in parts) else None


def imports(raw, module, package):
    result, dynamic = set(), []
    tree = ast.parse(raw)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            result.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ''
            if node.level:
                parts = (module if package else module.rpartition('.')[0]).split('.')
                if not parts[0] or node.level > len(parts):
                    raise ValueError('relative import escapes package')
                base = '.'.join(parts[:len(parts) - node.level + 1] + ([base] if base else []))
            if base:
                result.add(base)
                result.update(base + '.' + a.name for a in node.names if a.name != '*')
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name) and node.func.id == '__import__' or isinstance(node.func, ast.Attribute) and node.func.attr == 'import_module':
                dynamic.append({'line': node.lineno, 'expression': ast.unparse(node)[:256]})
    return sorted(result), dynamic


def collect(settings):
    tick = time.monotonic()
    site = Path(settings['site'])
    if not site.is_absolute() or site.resolve() != site:
        raise ValueError('canonical original site required')
    files, modules, extensions, records = {}, {}, {}, {}
    for record, profile in sorted(settings['records'].items()):
        p = Path(record)
        if not p.is_relative_to(site) or p.name != 'RECORD' or not p.parent.name.endswith('.dist-info'):
            raise ValueError('original RECORD path differs')
        raw = read_exact(p, profile['sha256'])
        records[record] = profile['sha256']
        seen = set()
        for row in csv.reader(io.StringIO(raw.decode()), strict=True):
            if len(row) != 3 or row[0] in seen:
                raise ValueError('duplicate/malformed RECORD row')
            seen.add(row[0])
            rel = Path(row[0])
            if rel.is_absolute() or any(x in {'', '.', '..'} for x in row[0].split('/')):
                continue
            if not (row[0].endswith('.py') or row[0].endswith('.so')):
                continue
            if not row[1].startswith('sha256=') or not row[2].isdigit():
                raise ValueError('hashed source/native RECORD row required')
            digest = base64.urlsafe_b64decode(row[1][7:] + '=' * (-len(row[1][7:]) % 4)).hex()
            if len(digest) != 64:
                raise ValueError('SHA256 RECORD digest required')
            path = str(site / rel)
            if path in files:
                raise ValueError('duplicate source/native ownership')
            fact = {'sha256': digest, 'bytes': int(row[2]), 'distribution': profile['name'], 'record': record}
            files[path] = fact
            name = module_name(row[0])
            if name is not None:
                if name in modules:
                    raise ValueError('ambiguous Python module ownership')
                modules[name] = path
            elif row[0].endswith('.so'):
                pieces = row[0].split('/')
                pieces[-1] = pieces[-1].split('.')[0]
                if all(x.isidentifier() for x in pieces):
                    extensions['.'.join(pieces)] = path
    source_names = {path: name for name, path in modules.items()}
    pending = []
    for path, expected in settings['seeds'].items():
        if path not in source_names or files[path]['sha256'] != expected:
            raise ValueError('original source seed/RECORD identity differs')
        pending.append(source_names[path])
    for relative in settings.get('additional_seed_paths', []):
        path = str(site / relative)
        if path not in source_names:
            raise ValueError('finite historical source seed absent from RECORD')
        pending.append(source_names[path])
    consumed, edges, outside, external, native, dynamic = {}, set(), set(), set(), {}, []
    visited = set()
    while pending:
        module = pending.pop()
        if module in visited:
            continue
        visited.add(module)
        path = modules[module]
        fact = files[path]
        if fact['distribution'] not in settings['candidate_distributions']:
            outside.add((module, fact['distribution']))
            continue
        raw = read_exact(path, fact['sha256'], fact['bytes'])
        consumed[path] = fact
        targets, calls = imports(raw, module, Path(path).name == '__init__.py')
        dynamic.extend(dict(source=path, **call) for call in calls)
        for target in targets:
            if target in extensions:
                native[target] = {'path': extensions[target], **files[extensions[target]]}
            elif target not in modules:
                external.add(target)
            names = target.split('.')
            for length in range(1, len(names) + 1):
                name = '.'.join(names[:length])
                if name in modules:
                    edges.add((module, name))
                    pending.append(name)
        names = module.split('.')
        pending.extend('.'.join(names[:n]) for n in range(1, len(names)) if '.'.join(names[:n]) in modules)
    for path, fact in consumed.items():
        read_exact(path, fact['sha256'], fact['bytes'])
    for path, digest in records.items():
        read_exact(path, digest)
    return {'schema': 'record-backed-static-import-candidates-v1', 'sources': consumed,
            'edges': sorted(edges), 'outside_candidate_distributions': sorted(outside),
            'unindexed_import_candidates': sorted(external), 'native_import_candidates': native,
            'dynamic_import_sites': dynamic, 'records': records,
            'elapsed_seconds': time.monotonic() - tick,
            'classification': 'conservative AST candidate graph; not eager import closure or execution permission',
            'target_modified': False, 'native_imports': False, 'native_qualified': False, 'product_go': False}
