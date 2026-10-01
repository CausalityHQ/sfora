#!/usr/bin/env python3
"""Stdlib admission/math/layout checks; all real tensor work belongs to parent."""
if not __debug__:
    raise SystemExit('Checks require assertions; optimized mode is forbidden')

import ast
import builtins
import copy
import ctypes
from contextlib import nullcontext
from collections import Counter
import hashlib
import importlib.util
import io
import json
import math
import mmap
import os
import random
import shutil
import subprocess
import struct
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from types import FunctionType, SimpleNamespace
from unittest.mock import patch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + '\n')
    return sha(path)


def rejects(call, message):
    try:
        call()
    except (ValueError, OSError, KeyError, TypeError) as error:
        assert message in str(error), str(error)
        return
    raise AssertionError('invalid input accepted: ' + message)


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class Flags(list):
    def any(self):
        return any(self)

    def sum(self):
        return sum(self)


class Positives(list):
    def __ge__(self, other):
        return SimpleNamespace(any=lambda dim: Flags(any(n >= other for n in row) for row in self))

    def __getitem__(self, key):
        return Positives(v for v, flag in zip(self, key, strict=True) if flag)


class Rows(list):
    def __getitem__(self, key):
        return Rows(v for v, flag in zip(self, key, strict=True) if flag)

    def sum(self):
        return sum(self)


def checkpoint_restore_checks(driver, root):
    # A full fingerprint/copy must release its consumed mmap pages, and Adam
    # step scalars must own bytes independently even when their device stays CPU.
    assert hasattr(driver, 'CheckpointPages'), 'missing consumed checkpoint page lifetime'
    path = root / 'mapped-checkpoint.zip'
    block = bytes(range(256)) * 8192 + b'partial tail'
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
        archive.writestr('archive/data/0', block)
        archive.writestr('archive/data/1', b'1234')
    original_sha = sha(path)
    with path.open('rb') as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_COPY) as mapped:
        address = ctypes.addressof(ctypes.c_char.from_buffer(mapped))
        pages = driver.CheckpointPages(stream)
        events = []
        real_madvise, real_advice = pages.madvise, os.posix_fadvise
        def madvise(start, size, advice):
            assert advice == mmap.MADV_DONTNEED
            events.append(('mapped', start - address, size))
            return real_madvise(start, size, advice)
        def fadvise(fd, offset, size, advice):
            assert fd == stream.fileno() and advice == os.POSIX_FADV_DONTNEED
            assert events and events[-1] == ('mapped', offset, size), 'file advice preceded unmapping'
            events.append(('file', offset, size))
            real_advice(fd, offset, size, advice)
        pages.madvise = madvise
        page = os.sysconf('SC_PAGESIZE')
        with patch.object(os, 'posix_fadvise', fadvise):
            assert mapped[44:44 + len(block)] == block
            pages.release(address + 44, len(block))
            end = (44 + len(block)) // page * page
            assert events == [('mapped', page, end - page), ('file', page, end - page)]
            events.clear()
            pages.release(address + 44, 4)
            pages.release(0, 0)
            assert events == []  # No zero-length advice (which would mean to EOF).
            # Refaulting later aliases must preserve the exact serialized bytes.
            assert mapped[44:44 + len(block)] == block and sha(path) == original_sha
            for start, count in ((address - 1, page), (address, len(mapped) + 1), (address, -1)):
                rejects(lambda: pages.release(start, count), 'checkpoint mapping range')
            with patch.object(pages, 'madvise', return_value=-1):
                ctypes.set_errno(5)
                rejects(lambda: pages.release(address, page), 'Input/output error')
                assert events == [], 'file advice followed failed madvise'
            with patch.object(os, 'posix_fadvise', side_effect=OSError('advice failed')):
                rejects(lambda: pages.release(address, page), 'advice failed')

        class Tensor:
            dtype, shape, _version = 'torch.uint8', (len(block),), 0
            device = SimpleNamespace(type='cpu')
            def __init__(self, start=44, count=len(block), data=None):
                self.start, self.count, self.data = start, count, data
                self.shape = (count,)
            def data_ptr(self):
                return address + self.start if self.data is None else id(self.data)
            def numel(self):
                return self.count
            def element_size(self):
                return 1
            def is_contiguous(self):
                return True
            def detach(self):
                return self
            cpu = contiguous = detach
            def reshape(self, *args):
                return self
            def view(self, *args):
                return self
            def numpy(self):
                return memoryview(mapped)[self.start:self.start + self.count] if self.data is None else self.data
            def to(self, device, *, copy):
                assert copy is True
                events.append(('copy', device, self.count))
                return Tensor(count=self.count, data=bytes(self.numpy()))
        fake_torch = SimpleNamespace(Tensor=Tensor, uint8='torch.uint8')
        tensor = Tensor()
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(os, 'posix_fadvise', fadvise):
            original = driver.fingerprint({'vision': tensor, 'optimizer': {'step': Tensor(44, 4)}})
            events.clear()
            real_sha = hashlib.sha256
            def tensor_sha(data=b''):
                if isinstance(data, memoryview):
                    events.append(('hashed', len(data)))
                return real_sha(data)
            with patch.object(driver.hashlib, 'sha256', tensor_sha):
                assert driver.fingerprint({'vision': tensor, 'optimizer': {'step': Tensor(44, 4)}},
                                          consumed=pages.consume) == original
            assert events == [('hashed', 4), ('hashed', len(block)),
                              ('mapped', page, end - page), ('file', page, end - page)]
            events.clear()
            owned = pages.copy(tensor)
            assert owned.data == block and events[0] == ('copy', 'cpu', len(block))
            assert events[1:] == [('mapped', page, end - page), ('file', page, end - page)]
            step = pages.copy(Tensor(44, 4))
            assert step.data == b'\x00\x01\x02\x03' and step.data_ptr() != address + 44
            with patch.object(tensor, 'to', side_effect=OSError('copy failed')):
                events.clear()
                rejects(lambda: pages.copy(tensor), 'copy failed')
                assert events == []
            def failed_sha(data=b''):
                if isinstance(data, memoryview):
                    raise OSError('hash failed')
                return real_sha(data)
            with patch.object(driver.hashlib, 'sha256', failed_sha):
                rejects(lambda: driver.fingerprint(tensor, consumed=pages.consume), 'hash failed')
                assert events == []
            with patch.object(tensor, 'device', SimpleNamespace(type='cuda')):
                rejects(lambda: pages.consume(tensor), 'CPU tensor')
            with patch.object(tensor, 'is_contiguous', return_value=False):
                pages.consume(tensor)
                assert events == []  # Do not advise unread strided storage gaps.
            assert pages.copy(tensor, 'cuda').data == block
            assert events[0] == ('copy', 'cuda', len(block))

        # Scoped post-load hooks cannot survive success, strict-load errors or
        # advice failures. Native Torch load itself remains the parent's check.
        copied, order = {}, []
        class Module:
            def __init__(self, name):
                self.name, self.hooks = name, []
            def named_parameters(self, *, recurse):
                assert recurse is False
                return [('weight', None)] if self.name else []
            def named_buffers(self, *, recurse):
                assert recurse is False
                return [('running', None), ('nonpersistent', None)] if self.name else []
            def register_load_state_dict_post_hook(self, hook):
                self.hooks.append(hook)
                return SimpleNamespace(remove=lambda: self.hooks.remove(hook))
        class Model(Module):
            def __init__(self):
                super().__init__('')
                self.children = [Module('first'), Module('second')]
            def named_modules(self):
                return [('', self)] + [(m.name, m) for m in self.children]
            def load_state_dict(self, values, *, strict):
                assert strict is True
                for child in self.children:
                    for local in ('weight', 'running'):
                        name = child.name + '.' + local
                        if name not in values:
                            raise ValueError('strict missing weight/buffer')
                        copied[name] = bytes(values[name].numpy())
                        order.append(('copy', name))
                    for hook in child.hooks:
                        assert hook(child, None) is None
                for hook in self.hooks:
                    assert hook(self, None) is None
        model = Model()
        vision = {name + '.' + local: Tensor() for name in ('first', 'second') for local in ('weight', 'running')}
        real_consume = pages.consume
        def consumed(value):
            key = next(k for k, v in vision.items() if v is value)
            assert copied[key] == block
            order.append(('consume', key))
            real_consume(value)
        with patch.object(pages, 'consume', consumed):
            driver.load_vision(model, vision, pages)
        assert order == [('copy', 'first.weight'), ('copy', 'first.running'),
                         ('consume', 'first.weight'), ('consume', 'first.running'),
                         ('copy', 'second.weight'), ('copy', 'second.running'),
                         ('consume', 'second.weight'), ('consume', 'second.running')]
        assert not any(m.hooks for _, m in model.named_modules())
        with patch.object(pages, 'consume', side_effect=OSError('load advice failed')):
            rejects(lambda: driver.load_vision(model, vision, pages), 'load advice failed')
        assert not any(m.hooks for _, m in model.named_modules())
        rejects(lambda: driver.load_vision(model, {k: v for k, v in vision.items() if k.startswith('first.')}, pages),
                'strict missing weight')
        assert not any(m.hooks for _, m in model.named_modules())
        with patch.object(model.children[1], 'register_load_state_dict_post_hook',
                          side_effect=OSError('registration failed')):
            rejects(lambda: driver.load_vision(model, vision, pages), 'registration failed')
        assert not any(m.hooks for _, m in model.named_modules())
        # Only the exact private mapping of the opened inode may be advised.
        maps_path = driver.Path('/proc/self/maps')
        actual_maps = maps_path.read_text()
        checkpoint_line = next(line for line in actual_maps.splitlines() if str(path) in line)
        fields = checkpoint_line.split(maxsplit=5)
        for index, replacement in ((0, f'{address:x}-{address + page:x}'),
                                   (1, 'rw-s'), (2, '00001000'), (3, 'ff:ff'), (4, '0')):
            changed = fields.copy(); changed[index] = replacement
            with patch.object(driver.Path, 'read_text', return_value=' '.join(changed)):
                rejects(lambda: driver.CheckpointPages(stream), 'checkpoint mapping identity')
        for text in ('', checkpoint_line + '\n' + checkpoint_line):
            with patch.object(driver.Path, 'read_text', return_value=text):
                rejects(lambda: driver.CheckpointPages(stream), 'checkpoint mapping identity')
    assert sha(path) == original_sha
    with zipfile.ZipFile(path) as archive:
        assert archive.read('archive/data/0') == block and archive.read('archive/data/1') == b'1234'


def native_contract_checks(driver):
    # Frozen base 8bdaf7d6: only fingerprint/restore and their page helpers may
    # change. Includes ALL admission, writer, math, freshCPU,17/8+9, final raw/
    # packed reload, locks/caps, and uncached exit logic, not just named constants.
    tree = ast.parse(Path(driver.__file__).read_bytes())
    fingerprint = copy.deepcopy(next(node for node in tree.body if getattr(node, 'name', None) == 'fingerprint'))
    assert fingerprint.args.args[-1].arg == 'consumed'
    fingerprint.args.args.pop(); fingerprint.args.defaults.pop()
    class WithoutAdvice(ast.NodeTransformer):
        def visit_If(self, node):
            if any(isinstance(part, ast.Name) and part.id == 'consumed' for part in ast.walk(node.test)):
                return None
            return self.generic_visit(node)
    assert hashlib.sha256(ast.dump(WithoutAdvice().visit(fingerprint), include_attributes=False).encode()).hexdigest() == (
        '9f9e4a507c623fab8d907210db8626f597e37ecb45a9abd8f3fe5acc165d7aaa')
    restore = next(node for node in tree.body if getattr(node, 'name', None) == 'restore_independent')
    predicates = [node for node in ast.walk(restore) if isinstance(node, ast.Call) and
                  isinstance(node.func, ast.Name) and node.func.id == 'require']
    for predicate in predicates:
        for node in ast.walk(predicate):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'fingerprint':
                node.keywords = [kw for kw in node.keywords if kw.arg != 'consumed']
    assert hashlib.sha256('\n'.join(sorted(ast.dump(node, include_attributes=False)
                                           for node in predicates)).encode()).hexdigest() == (
        'bf2f4c8aa1174c86bac0218f34306da28c995cb8289ebd7abb7d436e37b9bb79')
    changed = {'fingerprint', 'restore_independent', 'CheckpointPages', 'load_vision'}
    tree.body = [node for node in tree.body if getattr(node, 'name', None) not in changed]
    assert hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest() == (
        '6325c182e27eda4e1aaa26f461adeb0c221f357502052644abfedbb38558e7c6')


def checkpoint_write_checks(driver, root):
    # Removing the writer from save must fail on oversized writes, before final SHA.
    events, lifecycle = [], []
    page_size = driver.os.sysconf('SC_PAGESIZE')
    real_io = True
    raw = None
    class Stream:
        def write(self, data):
            events.append(('write', raw.tell(), len(data)))
            assert len(data) <= 1024**2, 'checkpoint write exceeds 1MiB'
            return raw.write(data)
        def tell(self):
            return raw.tell()
        def fileno(self):
            return raw.fileno()
        def flush(self):
            events.append(('flush', raw.tell()))
            raw.flush()
    def exclusive(path):
        nonlocal raw
        raw = path.open('x+b')
        class Exclusive:
            def __enter__(self):
                return Stream()
            def __exit__(self, *args):
                raw.close()
        return Exclusive()
    real_fsync, real_advice = driver.os.fsync, driver.os.posix_fadvise
    def fsync(fd):
        assert fd == raw.fileno()
        events.append(('fsync', raw.tell()))
        if real_io:
            real_fsync(fd)
    def advice(fd, offset, count, hint):
        assert fd == raw.fileno() and hint == driver.os.POSIX_FADV_DONTNEED
        assert offset % page_size == count % page_size == 0 and 0 < count <= 64 * 1024**2 + page_size
        assert offset + count <= raw.tell() < offset + count + page_size
        assert events[-2:] == [('flush', raw.tell()), ('fsync', raw.tell())], events[-2:]
        events.append(('advice', offset, count))
        if real_io:
            real_advice(fd, offset, count, hint)
    saved, identity, flags = {'complete': ['original', 8]}, {'seed': 179032}, {'threads': 1}
    state = {'optimizer': SimpleNamespace(zero_grad=lambda **kw: lifecycle.append(('zero_grad', kw))),
             'frozen_cache': {}}
    pieces = ()
    def serialize(value, stream):
        assert value is saved
        lifecycle.append('serialize')
        for piece in pieces:
            assert stream.write(piece) == len(piece)
        stream.flush()  # Deployed zip writer's exit flush.
    def imported(name, *args):
        assert name == 'torch'
        return SimpleNamespace(save=serialize)
    def integrity(actual, binding):
        assert actual is state and binding is identity
        lifecycle.append('integrity')
    def payload(actual, binding, numerical):
        assert actual is state and binding is identity and numerical is flags
        lifecycle.append('payload')
        return saved
    def fingerprint(value, frozen):
        assert value is saved and frozen is state['frozen_cache']
        lifecycle.append('fingerprint')
        return 'complete-state-fingerprint'
    def full_sha(path):
        assert raw.closed  # Preserve the original full-file hash after exclusive close.
        lifecycle.append('sha')
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    # Run the real save code with a serializer standin; no Torch import/patch or source-helper mutation.
    save = FunctionType(driver.save.__code__, {**vars(driver), 'integrity': integrity, 'payload': payload,
        'fingerprint': fingerprint, '__builtins__': {**vars(builtins), '__import__': imported}})
    context = {'initialized': {'source_context': {'extract': SimpleNamespace(exclusive=exclusive)},
                               'init': SimpleNamespace(sha=full_sha)}}
    block = bytes(range(256)) * 8192 + b'boundary'
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_STORED) as zipped:
        for name, data in (('state', block), ('empty', b''), ('tail', b'tail!')):
            zipped.writestr(zipfile.ZipInfo(name), data)
    zip_bytes = archive.getvalue()
    # Deterministic irregular serializer calls, including >1MiB, empty and a partial page.
    zip_pieces = (zip_bytes[:17], zip_bytes[17:1048610], b'', zip_bytes[1048610:1048683],
                  zip_bytes[1048683:-3], zip_bytes[-3:])
    for index, pieces in enumerate(((), (b'',), (b'header', block, b'', b'tail!'),
                                    zip_pieces)):
        events.clear(); lifecycle.clear()
        checkpoint = root / f'checkpoint-{index}.bin'
        expected = hashlib.sha256()
        for piece in pieces:
            expected.update(piece)
        with patch.object(driver.os, 'fsync', fsync), patch.object(driver.os, 'posix_fadvise', advice):
            assert save(context, state, identity, flags, checkpoint) == (
                expected.hexdigest(), 'complete-state-fingerprint')
        with checkpoint.open('rb') as stream:
            for piece in pieces:
                assert stream.read(len(piece)) == piece
            assert stream.read(1) == b''
        assert lifecycle == [('zero_grad', {'set_to_none': True}), 'integrity', 'payload',
                             'serialize', 'sha', 'fingerprint']
        if index == 3:
            with zipfile.ZipFile(checkpoint) as zipped:
                assert zipped.namelist() == ['state', 'empty', 'tail']
                assert [zipped.read(name) for name in zipped.namelist()] == [block, b'', b'tail!']
        total = sum(map(len, pieces))
        expected_ranges = [('advice', 0, total - total % page_size)] if total >= page_size else []
        assert [event for event in events if event[0] == 'advice'] == expected_ranges
        assert events[-2:] == [('flush', checkpoint.stat().st_size), ('fsync', checkpoint.stat().st_size)]
        rejects(lambda: save(context, state, identity, flags, checkpoint), 'File exists')
    # Exercise both real 64MiB boundaries without writing a large regression artifact.
    class CountingRaw:
        position = 0
        def write(self, data):
            self.position += len(data)
            return len(data)
        def tell(self):
            return self.position
        def fileno(self):
            return 123
        def flush(self):
            pass
    raw, real_io = CountingRaw(), False
    events.clear()
    with patch.object(driver.os, 'fsync', fsync), patch.object(driver.os, 'posix_fadvise', advice):
        writer = driver.CheckpointWriter(Stream())
        for _ in range(64):
            assert writer.write(block) == len(block)
            assert writer.write(b'') == 0
        assert writer.write(b'tail' * page_size + b'!') == 4 * page_size + 1
        writer.flush()
    assert [event for event in events if event[0] == 'advice'] == [
        ('advice', 0, 67108864), ('advice', 67108864, 67108864), ('advice', 134217728, 4 * page_size)]
    # Every low-level failure and short write is fatal, including final-tail flush/advice.
    with (root / 'writer-errors.bin').open('x+b') as raw:
        stream = Stream()
        writer = driver.CheckpointWriter(stream)
        for operation in ('write', 'flush'):
            with patch.object(stream, operation, side_effect=OSError('injected ' + operation)):
                rejects(lambda: writer.write(b'x') if operation == 'write' else writer.flush(), 'injected')
        for result in (0, None):
            with patch.object(stream, 'write', return_value=result):
                rejects(lambda: writer.write(b'x'), 'short checkpoint write')
        assert writer.write(b'tail' * page_size + b'!') == 4 * page_size + 1
        for operation in ('fsync', 'posix_fadvise'):
            with patch.object(driver.os, operation, side_effect=OSError('injected ' + operation)):
                rejects(writer.flush, 'injected')
        writer.flush()
        raw.seek(0)
        rejects(lambda: writer.write(b'wrong offset'), 'checkpoint offset disorder')
        rejects(writer.flush, 'checkpoint offset disorder')
        raw.seek(1)
        rejects(lambda: driver.CheckpointWriter(stream), 'checkpoint offset disorder')
    with (root / 'short-write.bin').open('x+b') as raw:
        stream = Stream()
        writer = driver.CheckpointWriter(stream)
        with patch.object(stream, 'write', side_effect=lambda data: raw.write(data[:-1])):
            rejects(lambda: writer.write(b'bytes'), 'short checkpoint write')
        assert raw.tell() == 4
        rejects(writer.flush, 'checkpoint offset disorder')



def predicate_correspondence_checks(driver):
    """Every original require predicate survives with only explicit IO/name
    substitutions. Runtime fixtures additionally exercise their control flow.
    """
    repo = Path(__file__).parent
    tree = ast.parse(Path(driver.__file__).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FlatAdmission')
    ledger = (
        ('qualify_siglip2_substrate_cpu', 'original_extraction', 'original_extraction', {'source.POLICY': 'POLICY'}),
        ('qualify_siglip2_substrate_cpu', 'authority', 'source_authority', {'source_driver.ARMS': 'ARMS'}),
        ('qualify_siglip2_substrate_cpu', 'package_origins', 'package_origins', {"context['source_driver'].PACKAGES": 'PACKAGES'}),
        ('export_siglip2_substrate_fit', 'admit_cpu', 'source_cpu', {'self.exporter.STARTUP_POLICY': 'STARTUP_POLICY'}),
        ('export_siglip2_substrate_fit', 'all_fit_images', 'all_fit_images', {}),
        ('export_siglip2_substrate_fit', 'authority', 'export_authority', {
            'exporter.' + name: name for name in ('AUTHORITY_SCHEMA', 'STARTUP_POLICY', 'EXPORT_POLICY', 'SOURCE_ROOT')}),
        ('initialize_siglip2_substrate_fit', 'admit_export', 'admit_export', {'self.init.EXPORT_ARITHMETIC': 'EXPORT_ARITHMETIC'}),
        ('initialize_siglip2_substrate_fit', 'admit_terminal', 'admit_terminal', {}),
        ('initialize_siglip2_substrate_fit', 'cache_facts', 'cache_facts', {'digest.hexdigest()': 'sha(path)'}),
        ('initialize_siglip2_substrate_fit', 'authority', 'pca_authority', {
            **{'init.' + name: name for name in ('AUTHORITY_SCHEMA', 'POLICY', 'EXPORT_FILES', 'SOURCE_FILES')},
            'NATIVE': 'NATIVE_PACKAGES'}),
        ('qualify_siglip2_initialized_cpu', 'authority', 'initialized_authority', {'NATIVE': 'NATIVE_PACKAGES'}))
    def predicates(function, renames):
        class Normalize(ast.NodeTransformer):
            def visit(self, node):
                text = ast.unparse(node)
                if text in renames:
                    return ast.parse(renames[text], mode='eval').body
                return super().visit(node)
        return Counter(ast.dump(Normalize().visit(copy.deepcopy(node.args[0])), include_attributes=False)
            for node in ast.walk(function) if isinstance(node, ast.Call) and
            isinstance(node.func, ast.Name) and node.func.id == 'require')
    for file, original, replacement, renames in ledger:
        old = next(n for n in ast.parse((repo / (file + '.py')).read_text()).body
                   if isinstance(n, ast.FunctionDef) and n.name == original)
        new = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == replacement)
        assert not predicates(old, {}) - predicates(new, renames), (file, original, 'missing predicate')


def flat_chain_checks(driver, base):
    """Full stdlib authority chain; mock only the multi-GB safetensors payload
    header boundary. Native modules stay unloaded; package metadata is synthetic.
    Compare stage dictionaries with the original authority, not self-generated
    expectations. NPY/NPZ bodies are real stdlib bytes at their required shapes.
    """
    repo = Path(__file__).parent.resolve()
    exporter = module(repo / 'export_siglip2_substrate_fit.py', 'flat_fixture_exporter')
    fixtures = module(repo / 'test_siglip2_substrate_fit.py', 'flat_source_fixtures')
    export_args, export_launch, proof, source, extract, inventory, export_root, source_root = fixtures.authority_fixture(base, exporter)
    # A synthetic pinned closure gets its own declared root before hashing.
    export_file = export_root / 'export_siglip2_substrate_fit.py'
    export_file.write_text(export_file.read_text().replace(str(exporter.SOURCE_ROOT), str(source_root)))
    exporter = module(export_file, 'flat_fixture_exporter_local')
    export_code = {name: sha(export_root / name) for name in exporter.FILES}
    export_args.execution_sha256 = write(export_root / 'execution.json', export_code)
    export_launch['execution_sha256'] = export_args.execution_sha256
    site = base / 'site'
    site.mkdir()
    packages, versions, files = {}, {}, {}
    for package, distribution in source.PACKAGES.items():
        package_root = site / package
        package_root.mkdir()
        origin = package_root / '__init__.py'
        origin.write_text('# never imported\n')
        meta = site / (distribution + '-1.0.dist-info')
        meta.mkdir()
        (meta / 'METADATA').write_text('Metadata-Version: 2.1\nName: ' + distribution + '\nVersion: 1.0\n')
        versions[distribution.lower()] = '1.0'
        packages[package] = {'root': str(package_root), 'origin': str(origin), 'version': '1.0'}
        files[str(origin)] = {'sha256': sha(origin), 'bytes': origin.stat().st_size}
    constructor = site / 'transformers' / 'vision.py'
    constructor.write_text('# synthetic observed source\n')
    files[str(constructor)] = {'sha256': sha(constructor), 'bytes': constructor.stat().st_size}
    sources = json.loads((source_root / 'sources.json').read_bytes())
    sources['native_environment'] = {'schema': 'native256-installed-source-observation-v1',
        'native_imported': False, 'model_executed': False, 'quality_read': False,
        'site_packages': str(site), 'versions': versions, 'files': files,
        'vision_constructor': {'direct_bare_state_keys_source_observed': True, 'path': str(constructor),
            'assigned_self_attributes': ['config', 'embeddings', 'encoder', 'head', 'post_layernorm', 'use_head']}}
    sources_sha = write(source_root / 'sources.json', sources)
    export_launch['sources']['sha256'] = sources_sha
    original_path = Path(export_launch['source_cpu_authority']['path'])
    original = json.loads(original_path.read_bytes())
    original['sources_sha256'] = sources_sha
    export_launch['source_cpu_authority']['sha256'] = write(original_path, original)
    proof['sources_sha256'] = sources_sha
    proof['invocation']['argv'][6] = sources_sha
    proof['input_guards'][str(source_root / 'sources.json')] = sources_sha
    proof['origins']['packages'] = packages
    proof['origins']['files'] = {p: v['sha256'] for p, v in files.items()}
    proof['input_guards'].update(proof['origins']['files'])
    proof_descriptor = export_launch['source_cpu']['large']['proof']
    proof_descriptor['sha256'] = write(Path(proof_descriptor['path']), proof)
    export_args.authority_sha256 = write(export_args.authority, export_launch)
    evidence = repo.parent / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1/late-dense-v1'
    def receipt(name):
        return json.loads((evidence / (name + '.json')).read_bytes())
    def terminal(record, path, argv, cuda=''):
        unit, invocation = 'fixture-' + path.parent.name + '-' + path.stem, 'b'*32
        memory = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': {
            'memory.max': str(8 * 1024**3), 'memory.current': '128', 'memory.peak': '256',
            'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
            'memory.events': 'max 0\noom 0\noom_kill 0'}}
        record.update(wall_seconds=1., process_peak_rss_kib=1, cgroup_before=memory, cgroup_after=memory,
            invocation={**proof['invocation'], 'argv': argv, 'invocation_id': invocation,
                        'cuda_visible_devices': cuda})
        final = {**memory, 'invocation_id': invocation}
        log = path.with_suffix('.log')
        log.write_text(f'Running as unit: {unit}.service; invocation ID: {invocation}\n'
            '\tExit status: 0\nFinished with result: success\nMain processes terminated with: code=exited/status=0\n'
            '\tSwaps: 0\nMemory swap peak: 0B\nService runtime: 2s\n'
            '\tMaximum resident set size (kbytes): 2\nFINAL_CGROUP ' + json.dumps(final) + '\n')
        return {'receipt': {'path': str(path), 'sha256': write(path, record)},
                'log': {'path': str(log), 'sha256': sha(log)}, 'unit': unit, 'invocation_id': invocation,
                'service_seconds': 2., 'native_peak_rss_kib': 2, 'both_locks_held': True}, final
    def closure(root, names):
        root.mkdir()
        for name in names:
            local = repo / name
            if not local.exists():
                local = repo.parent / 'src/sfora' / name
            shutil.copyfile(local, root / name)
        code = {n: sha(root / n) for n in names}
        return code, write(root / 'execution.json', code)
    init_root, qroot, own = base / 'pca-code', base / 'q-code', base / 'trainer'
    init = module(repo / 'initialize_siglip2_substrate_fit.py', 'flat_fixture_init')
    q = module(repo / 'qualify_siglip2_initialized_cpu.py', 'flat_fixture_q')
    init_code, init_pin = closure(init_root, init.FILES)
    qcode, qpin = closure(qroot, q.FILES)
    init = module(init_root / 'initialize_siglip2_substrate_fit.py', 'flat_fixture_init_local')
    q = module(qroot / 'qualify_siglip2_initialized_cpu.py', 'flat_fixture_q_local')
    own.mkdir()
    originals = {'deployed_code_rank.py': repo.parent / 'src/sfora/deployed_code_rank.py',
                 'reference_train_sop_siglip2_compact.py': repo / 'train_sop_siglip2_compact.py',
                 'reference_unicom_training.py': repo.parent / 'src/sfora/unicom_training.py'}
    for name in driver.FILES:
        shutil.copyfile(originals.get(name, repo / name), own / name)
    own_code = {n: sha(own / n) for n in driver.FILES}
    own_pin = write(own / 'execution.json', own_code)
    # Original recursive admission is the independent stage-inventory oracle.
    names = ('extract_siglip2_vision_source', 'qualify_siglip2_substrate_cpu',
             '_siglip2_pinned_fit_export', '_siglip2_pinned_initialized_pca', '_siglip2_pinned_adaptation_qualifier')
    def clear():
        for name in names:
            sys.modules.pop(name, None)
    real_load = driver.load_bare
    def load_with_fixture_header(name, path, digest):
        value = real_load(name, path, digest)
        if name == 'extract_siglip2_vision_source':
            # The real parser is tested independently below; omit 1.26GB allocation.
            value.read_header = lambda stream: (inventory, {}, b'')
        return value
    prior_path = sys.path[:]
    sys.path.insert(0, str(site))
    try:
        with patch.object(extract, 'read_header', return_value=(inventory, {}, b'')):
            source_context = exporter.authority(export_args)
        startup_guards = source_context['guards'].copy()
        startup = receipt('native256-fit-startup-large-v1')
        startup.update(binding=exporter.binding(source_context), input_guards=startup_guards, packages=packages)
        startup_path = base / 'startup.json'
        prefix = [str(export_file), '--execution-sha256', export_args.execution_sha256,
                  '--authority', str(export_args.authority), '--authority-sha256', export_args.authority_sha256,
                  '--arm', 'large']
        startup_descriptor, startup_final = terminal(startup, startup_path, prefix + ['--check-startup-only', '--output', str(startup_path)])
        export_dir = base / 'fit-export'
        export_dir.mkdir()
        cache = export_dir / 'fit.npy'
        header = repr({'descr': '<f4', 'fortran_order': False, 'shape': (13283, 1024)}).encode()
        row = struct.pack('<1024f', 1., *([0.]*1023))
        with cache.open('wb') as stream:
            stream.write(b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header)
            for _ in range(13283):
                stream.write(row)
        exported = receipt('native256-fit-export-large-v1')
        rgb = []
        for ordinal, (row, target, image) in enumerate(zip(source_context['fit']['rows'], source_context['fit']['targets'], source_context['all_images'])):
            rgb.append({'ordinal': ordinal, 'train_row': row['train_row'], 'target': target,
                'relative_path': row['relative_path'], 'path': str(image), 'image_sha256': row['image_sha256'],
                'mode': 'RGB', 'size': [256, 256], 'rgb_sha256': 'a'*64})
        for row, sample in zip(rgb, proof['sample']['images']):
            row.update({k: sample[k] for k in ('mode', 'size', 'rgb_sha256')})
        ordered = {'rows': source_context['fit']['rows'], 'targets': source_context['fit']['targets'],
            'class_names': source_context['fit']['class_names'], 'resolved_paths': list(map(str, source_context['all_images']))}
        exported.update(binding=exporter.binding(source_context), startup=startup_descriptor['receipt'],
            source_checkpoint_metadata_only=proof['checkpoint'], cpu_numerical_flags=proof['numerical_flags'],
            input_guards={**startup_guards, str(startup_path): startup_descriptor['receipt']['sha256']},
            origins=proof['origins'], rgb_manifest=rgb, ordered_rgb_sha256=exporter.object_sha(rgb),
            ordered_input_sha256=exporter.object_sha(ordered), export_seconds=.5,
            cache={**exported['cache'], 'path': str(cache), 'sha256': sha(cache)})
        export_descriptor, export_final = terminal(exported, export_dir / 'receipt.json', prefix + [
            '--startup', str(startup_path), '--startup-sha256', startup_descriptor['receipt']['sha256'],
            '--output', str(export_dir)], '0')
        pca_launch = {'schema': init.AUTHORITY_SCHEMA, 'execution_sha256': init_pin,
            'pca_helper_sha256': init_code['representation_ceiling.py'], 'export_root': str(export_root),
            'export_execution_sha256': export_args.execution_sha256,
            'export_authority': {'path': str(export_args.authority), 'sha256': export_args.authority_sha256},
            'selected_export': export_descriptor, 'startup': startup_descriptor,
            'resource_policy': init.POLICY, 'both_locks_held': True}
        pca_args = SimpleNamespace(execution_sha256=init_pin, authority=base / 'pca-launch.json',
            authority_sha256=write(base / 'pca-launch.json', pca_launch), arm='large', output=base / 'NEW')
        # Original helpers load the same bytes; patch only their loader's header
        # boundary for the synthetic tiny derived source.
        clear()
        original_load = init.load_bare
        def init_load(name, path, digest):
            value = original_load(name, path, digest)
            if name == 'extract_siglip2_vision_source':
                value.read_header = lambda stream: (inventory, {}, b'')
            return value
        with patch.object(init, 'load_bare', init_load):
            pca = init.authority(pca_args)
        pca_dir = base / 'pca-proof'
        pca_dir.mkdir()
        arrays, facts = {}, {}
        for name, shape in q.array_shapes(1024).items():
            values = (struct.pack('<13283q', *source_context['fit']['targets']) if name == 'target'
                      else b'\0' * (4 * math.prod(shape)))
            arrays[name] = values
            facts[name] = {'dtype': 'torch.int64' if name == 'target' else 'torch.float32',
                           'shape': shape, 'sha256': hashlib.sha256(values).hexdigest()}
        archive = pca_dir / 'initializers.npz'
        with zipfile.ZipFile(archive, 'w') as stream:
            for name, raw in arrays.items():
                header = repr({'descr': '<i8' if name == 'target' else '<f4', 'fortran_order': False,
                               'shape': tuple(facts[name]['shape'])}).encode()
                stream.writestr(name + '.npy', b'\x93NUMPY\x01\x00' + len(header).to_bytes(2, 'little') + header + raw)
        pca_record = receipt('native256-fit-initializer-large-v2')
        pca_record.update(authority_sha256=pca_args.authority_sha256, execution_sha256=init_pin, code=init_code,
            pca_helper_sha256=pca_launch['pca_helper_sha256'], source_binding=exported['binding'],
            source_checkpoint_metadata_only=proof['checkpoint'], source_roles=proof['runtime']['roles'],
            source_roles_sha256=exporter.object_sha(proof['runtime']['roles']), cache=exported['cache'],
            cache_facts_before_native=pca['cache_facts_before_native'], ordered_input_sha256=exported['ordered_input_sha256'],
            ordered_rgb_sha256=exported['ordered_rgb_sha256'], class_names=source_context['fit']['class_names'],
            numerical_flags=proof['numerical_flags'], export_final_cgroup=export_final, startup_final_cgroup=startup_final,
            class_counts=[source_context['fit']['targets'].count(n) for n in range(2004)], arrays=facts,
            artifact={'path': str(archive), 'sha256': sha(archive)}, input_guards=pca['guards'].copy(),
            origins=proof['origins'], pca_seconds=.5,
            pca_sha256=hashlib.sha256(arrays['mean'] + arrays['components']).hexdigest())
        for key in ('export_root', 'export_execution_sha256', 'export_authority', 'selected_export', 'startup'):
            pca_record[key] = pca_launch[key]
        pca_descriptor, pca_final = terminal(pca_record, pca_dir / 'receipt.json', [
            str(init_root / 'initialize_siglip2_substrate_fit.py'), '--execution-sha256', init_pin,
            '--authority', str(pca_args.authority), '--authority-sha256', pca_args.authority_sha256,
            '--arm', 'large', '--output', str(pca_dir)])
        qlaunch = {'schema': q.AUTHORITY_SCHEMA, 'execution_sha256': qpin,
            'packing_helper_sha256': qcode['joint_relational_compaction.py'], 'initializer_root': str(init_root),
            'initializer_execution_sha256': init_pin,
            'initializer_authority': {'path': str(pca_args.authority), 'sha256': pca_args.authority_sha256},
            'selected_initializer': pca_descriptor, 'resource_policy': q.POLICY, 'both_locks_held': True}
        qargs = SimpleNamespace(execution_sha256=qpin, authority=base / 'q-launch.json',
            authority_sha256=write(base / 'q-launch.json', qlaunch), arm='large', output=base / 'NEW')
        # Oracle q.authority uses its genuine initializer; intercept only its
        # stdlib loader to provide the same synthetic derived-header boundary.
        clear()
        qload = q.load_bare
        def q_load(name, path, digest):
            value = qload(name, path, digest)
            value.load_bare = init_load
            return value
        with patch.object(q, 'load_bare', q_load):
            initialized = q.authority(qargs)
        prereq = initialized['guards'].copy()
        cpu_dir = base / 'cpu-proof'
        cpu_dir.mkdir()
        checkpoint = cpu_dir / 'initialized.pt'
        checkpoint.write_bytes(b'archived checkpoint; never initialization')
        cpu = receipt('native256-initialized-cpu-large-v3')
        cpu.update(authority_sha256=qargs.authority_sha256, execution_sha256=qpin, code=qcode,
            initializer_authority=qlaunch['initializer_authority'], selected_initializer=pca_descriptor,
            pca_final_cgroup=pca_final, source_binding=exported['binding'], source_sample=proof['sample'],
            ordered_input_sha256=exported['ordered_input_sha256'], ordered_rgb_sha256=exported['ordered_rgb_sha256'],
            initializers=pca_record['artifact'], numerical_flags=proof['numerical_flags'],
            checkpoint={'path': str(checkpoint), 'sha256': sha(checkpoint)}, origins=proof['origins'],
            input_guards={**prereq, str(checkpoint): sha(checkpoint)})
        cpu['state'].update(arrays=facts, runtime=proof['runtime'])
        cpu_descriptor, _ = terminal(cpu, cpu_dir / 'proof.json', [
            str(qroot / 'qualify_siglip2_initialized_cpu.py'), '--execution-sha256', qpin,
            '--authority', str(qargs.authority), '--authority-sha256', qargs.authority_sha256,
            '--arm', 'large', '--output', str(cpu_dir)])
        launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': own_pin,
            'phase': 'mechanics', 'arm': 'large', 'seed': 179032, 'qualifier_root': str(qroot),
            'qualifier_execution_sha256': qpin,
            'qualifier_authority': {'path': str(qargs.authority), 'sha256': qargs.authority_sha256},
            'selected_cpu': cpu_descriptor, 'selected_mechanics': None,
            'resource_policy': driver.policy('mechanics'), 'both_locks_held': True}
        args = SimpleNamespace(execution_sha256=own_pin, authority=base / 'trainer-launch.json',
            authority_sha256=write(base / 'trainer-launch.json', launch), phase='mechanics',
            arm='large', seed=179032, output=base / 'NEW')
        def run():
            clear()
            with patch.object(driver, '__file__', str(own / 'train_siglip2_substrate_adaptation.py')), \
                 patch.object(driver, 'load_bare', load_with_fixture_header):
                return driver.authority(args)
        with patch.object(driver, 'bound_file', wraps=driver.bound_file) as bounded:
            context = run()
        bulk = [Path(proof['checkpoint']['path']), checkpoint, archive, *source_context['all_images']]
        calls = [Path(call.args[1]) for call in bounded.call_args_list]
        counts = Counter(calls)
        assert all(counts[path] == 1 for path in bulk), 'repeated bulk admission SHA'
        assert counts[cache] == 0, 'NPY must hash during its semantic scan'
        assert context['initialized_prereq_guards'] == prereq
        initialized = context['initialized']
        assert context['guards'] is initialized['guards'] is initialized['pca']['guards'] is initialized['source_context']['guards']
        assert set(cpu['input_guards']) <= context['guards'].keys()
        assert initialized['source_context']['proof'] == proof
        assert context['cpu'] == cpu and initialized['record'] == pca_record
        # Missing source proof, changed flags/roles/stage inventories must reject
        # before any native import, even when the containing JSON is re-pinned.
        source_bad = copy.deepcopy(proof)
        source_bad['runtime']['roles'][0]['role'] = 'trainable'
        source_bad['runtime']['roles'][1]['role'] = 'frozen'
        proof_descriptor['sha256'] = write(Path(proof_descriptor['path']), source_bad)
        reader = driver.FlatAdmission()
        reader.exporter = exporter
        rejects(lambda: reader.source_cpu({**source_context, 'guards': {}},
            export_launch['source_cpu']['large'], original), 'role')
        proof_descriptor['sha256'] = write(Path(proof_descriptor['path']), proof)
        bad_startup = {**startup, 'input_guards': context['guards']}
        rejects(lambda: exporter.admit_startup({**source_context, 'guards': startup_guards}, bad_startup), 'startup admission')
        rejects(lambda: exporter.admit_startup({**source_context, 'guards': startup_guards}, {**startup, 'schema': 'bad'}), 'startup admission')
        # The unchanged complete exit chain must detect restored-mtime tampering.
        stamp = checkpoint.stat()
        checkpoint.write_bytes(b'X' + checkpoint.read_bytes()[1:])
        os.utime(checkpoint, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        rejects(lambda: context['qualifier'].rehash(initialized), 'exit authority SHA256')
    finally:
        sys.path[:] = prior_path
        clear()

def main():
    path = Path(__file__).with_name('train_siglip2_substrate_adaptation.py').resolve()
    driver = module(path, 'adaptation_under_test')
    native_contract_checks(driver)
    with TemporaryDirectory() as directory:
        checkpoint_restore_checks(driver, Path(directory))
    predicate_correspondence_checks(driver)
    assert hasattr(driver, 'FlatAdmission'), 'trainer-owned flat admission is missing'
    with TemporaryDirectory() as directory:
        root = Path(directory).resolve()
        artifact = root / 'bulk.bin'
        artifact.write_bytes(b'original bytes')
        digest = sha(artifact)
        admission = driver.FlatAdmission()
        first, second = {}, {}
        with patch.object(driver.os, 'posix_fadvise', wraps=driver.os.posix_fadvise) as advice:
            admission.bound_file(first, artifact, digest, artifact.stat().st_size)
            admission.bound_file(second, artifact, digest)
            assert advice.call_count == 1, 'duplicate bulk admission read'
        assert first == second == {str(artifact): digest}
        rejects(lambda: admission.bound_file({}, artifact, '0'*64), 'conflicting')
        rejects(lambda: admission.bound_file({}, artifact, digest, 99), 'size')
        stamp = artifact.stat()
        artifact.write_bytes(b'tampered bytes')
        os.utime(artifact, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        rejects(lambda: driver.bound_file({}, artifact, digest), 'SHA256')
        record = root / 'record.json'
        pin = write(record, {'valid': True})
        assert admission.read_json(record, pin, {}) == {'valid': True}
        record.write_text('{"valid":true,"valid":false}')
        rejects(lambda: driver.FlatAdmission().read_json(record, sha(record), {}), 'duplicate')
    assert not any(n.split('.')[0] in driver.NATIVE for n in sys.modules)
    tree = ast.parse(path.read_bytes())
    # No inherited startup/package initialization, peak reset, quality API or rescue.
    banned = {'reset_peak_memory_stats', 'evaluate', 'from_pretrained_model', 'set_default_dtype'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            assert node.func.attr not in banned
    for flags, arguments, code, marker in (([], ['--help'], 0, '--authority-sha256'),
                                          ([], [], 2, 'required'),
                                          (['-O'], ['--help'], 1, 'optimized'),
                                          (['-OO'], ['--help'], 1, 'optimized')):
        result = subprocess.run([sys.executable, '-B', '-S', *flags, str(path), *arguments],
                                capture_output=True, text=True)
        assert result.returncode == code and marker in result.stdout + result.stderr
    assert driver.policy('mechanics')['seconds'] == 300 and driver.policy('train')['seconds'] == 300
    rejects(lambda: driver.policy('quality'), 'fixed phase')
    native_facts = {'seed': 179041, 'counter': 0, 'parameter_names': ['original'],
                    'optimizer_defaults': {'betas': (.9, .999), 'eps': 1e-8},
                    'optimizer_groups': [{'betas': (.9, .999), 'lr': 1e-5}],
                    'optimizer_state': {'param_groups': [{'betas': (.9, .999), 'params': [0, 1]}], 'state': {}},
                    'runtime': {'config': {'id2label': {0: 'zero', 1: 'one'}}}}
    persisted_facts = json.loads(json.dumps({**native_facts, 'seed': 179032}))
    assert {**native_facts, 'seed': 179032} != persisted_facts  # Reproduces the actual JSON boundary defect.
    driver.check_cpu_state(native_facts, persisted_facts)
    for key in native_facts:
        if key != 'seed':
            rejects(lambda key=key: driver.check_cpu_state({**native_facts, key: 'tampered'}, persisted_facts), 'CPU state')
    targets = [n % 2004 for n in range(13283)]
    batches = [[(step * 64 + offset) % 2004 for offset in range(64)] for step in range(100)]
    for seed in driver.SEEDS:
        driver.check_schedule(batches, targets, seed)
    for changed, altered_targets, seed in ((batches[:-1], targets, 179032),
                                           ([[0]*64] + batches[1:], targets, 179032),
                                           ([[-1] + batches[0][1:]] + batches[1:], targets, 179032),
                                           (batches, targets[:-1], 179032),
                                           (batches, targets, 179019)):
        rejects(lambda: driver.check_schedule(changed, altered_targets, seed), 'schedule')
    calls = []
    def rank(raw, bank, head, positives, ordinals, *, live_head):
        calls.append((list(raw), list(positives), list(ordinals), live_head))
        assert live_head is False and all(any(n >= 0 for n in row) for row in positives)
        return 3.
    ref = SimpleNamespace(member_bank_rank_loss=rank)
    # Valid anchors retain the ORIGINAL microbatch denominator, including zero-valid.
    assert driver.valid_rank(ref, Rows([1, 2, 3, 4]), None, None,
                             Positives([[2, -1], [-1, -1], [0, 1], [-1, -1]]), Rows([0, 1, 2, 3])) == 1.5
    assert calls == [([1, 3], [[2, -1], [0, 1]], [0, 2], False)]
    assert driver.valid_rank(ref, Rows([1, 2]), None, None, Positives([[-1], [-1]]), Rows([0, 1])) == 0
    assert len(calls) == 1
    identity, defaults, groups, names, flags = {'seed': 179032}, {'lr': .001}, [{'params': list(range(208))}], list(range(208)), {'threads': 1}
    saved = {key: None for key in driver.PAYLOAD_KEYS}
    saved.update(schema=driver.SCHEMA, identity=identity, counter=8, seed=179032, numerical_flags=flags,
                 optimizer_defaults=defaults, optimizer={'param_groups': groups,
                    'state': {n: {'step': 8, 'exp_avg': 0., 'exp_avg_sq': 0.} for n in range(208)}},
                 cuda_rng=['CUDA'], scaler={'scale': 128., '_growth_tracker': 8})
    driver.check_payload(saved, identity, 8, defaults, groups, names, flags)
    for key in driver.PAYLOAD_KEYS:
        rejects(lambda key=key: driver.check_payload({k: v for k, v in saved.items() if k != key},
            identity, 8, defaults, groups, names, flags), 'complete state')
    for key, value in (('counter', 7), ('seed', 179041), ('cuda_rng', []), ('optimizer_defaults', {'lr': .1}),
                       ('numerical_flags', {}), ('scaler', {'scale': 64, '_growth_tracker': 8})):
        rejects(lambda: driver.check_payload({**saved, key: value}, identity, 8, defaults, groups, names, flags), 'complete state')
    bad = copy.deepcopy(saved)
    bad['optimizer']['state'][0]['step'] = 7
    rejects(lambda: driver.check_payload(bad, identity, 8, defaults, groups, names, flags), 'complete state')
    bad = copy.deepcopy(saved)
    bad['optimizer']['param_groups'][0]['params'].reverse()
    rejects(lambda: driver.check_payload(bad, identity, 8, defaults, groups, names, flags), 'complete state')
    with patch.dict(sys.modules, {'torch': SimpleNamespace(Tensor=type('NoRealTensor', (), {}))}):
        assert driver.fingerprint({'a': [1, 2], 'b': 'x'}) == driver.fingerprint({'b': 'x', 'a': [1, 2]})
        assert driver.fingerprint({'a': [1, 2]}) != driver.fingerprint({'a': (1, 2)})
        assert driver.fingerprint({'a': 1}) != driver.fingerprint({'a': True})
    with TemporaryDirectory() as directory, patch.dict(sys.modules):
        root = Path(directory).resolve()
        checkpoint_write_checks(driver, root)
        # Phase logs must flush live original-unit facts without entering replay state.
        assert hasattr(driver, 'phase_diagnostic'), 'missing flushed phase diagnostics'
        proc = root / 'cgroup'
        proc.write_text('0::/original.service\n')
        memory = root / 'memory'
        memory.mkdir()
        values = {'memory.current': '128', 'memory.peak': '256', 'memory.max': '8589934592',
                  'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
                  'memory.events': 'max 0\noom 0\noom_kill 0',
                  'memory.stat': 'anon 64\nfile 32\nfile_dirty 16\nfile_writeback 0'}
        for name, value in values.items():
            (memory / name).write_text(value + '\n')
        def diagnostic_path(value):
            if value == '/proc/self/cgroup':
                return proc
            path = Path(value)
            assert path == Path('/sys/fs/cgroup')
            return root
        (root / 'original.service').symlink_to(memory, target_is_directory=True)
        flushed = []
        class PhaseLog(io.StringIO):
            def flush(self):
                flushed.append(self.getvalue())
        output = PhaseLog()
        rng, replay = random.getstate(), {'step': 8, 'state_sha256': 'fixed', 'seconds': 1.}
        original_replay = driver.diagnostic(replay)
        with patch.object(driver, 'Path', side_effect=diagnostic_path), \
                patch.object(driver.time, 'monotonic', side_effect=(102.5, 104., 105.)), \
                patch.dict(driver.os.environ, {'INVOCATION_ID': 'a'*32}), patch.object(sys, 'stdout', output):
            assert driver.phase_diagnostic('save8.begin', 100.) is None
            (memory / 'memory.current').write_text('512\n')
            (memory / 'memory.peak').write_text('8589934592\n')
            (memory / 'memory.events').write_text('max 1\noom 0\noom_kill 0\n')
            (memory / 'memory.stat').write_text('anon 64\nfile 448\nfile_dirty 400\nfile_writeback 16\n')
            expected_bytes = {p.name: p.read_bytes() for p in memory.iterdir()}
            assert driver.phase_diagnostic('save8.end', 100.) is None
            assert {p.name: p.read_bytes() for p in memory.iterdir()} == expected_bytes
            (memory / 'memory.peak').unlink()
            expected_bytes.pop('memory.peak')
            assert driver.phase_diagnostic('unavailable', 100.) is None
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        assert [r['elapsed_seconds'] for r in rows] == [2.5, 4., 5.]
        assert [r['phase'] for r in rows] == ['save8.begin', 'save8.end', 'unavailable']
        assert all(r['diagnostic'] == 'phase' and r['invocation_id'] == 'a'*32 for r in rows)
        assert rows[0]['cgroup']['values'] == values
        assert rows[1]['cgroup']['values']['memory.current'] == '512'
        assert rows[1]['cgroup']['values']['memory.peak'] == '8589934592'
        assert rows[1]['cgroup']['values']['memory.events'].startswith('max 1\n')
        assert rows[1]['cgroup']['values']['memory.stat'] == 'anon 64\nfile 448\nfile_dirty 400\nfile_writeback 16'
        assert 'memory.peak' in rows[2]['cgroup']['error'] and 'values' not in rows[2]['cgroup']
        assert len(flushed) == 3 and flushed[0] == output.getvalue().splitlines(keepends=True)[0]
        assert {p.name: p.read_bytes() for p in memory.iterdir()} == expected_bytes
        assert random.getstate() == rng and driver.diagnostic(replay) == original_replay
        assert not any(n.split('.')[0] in driver.NATIVE for n in sys.modules)
        # Exact SHA including empty EOF/tail, advising ONLY consumed 1MiB ranges.
        hashed = root / 'hashed.bin'
        multichunk = bytes(range(256)) * 8192
        for data in (b'', multichunk, multichunk + b'tail' * 9 + b'!'):
            hashed.write_bytes(data)
            with patch.object(driver.os, 'posix_fadvise', wraps=driver.os.posix_fadvise) as advice:
                assert driver.bound_file({}, hashed, hashlib.sha256(data).hexdigest()) == hashed
            ranges = [(offset, count, hint) for fd, offset, count, hint in
                      (call.args for call in advice.call_args_list)]
            assert ranges == [(offset, min(1024**2, len(data) - offset), driver.os.POSIX_FADV_DONTNEED)
                              for offset in range(0, len(data), 1024**2)], ranges
        dataset_root = root / 'images'
        dataset_root.mkdir()
        paths, image_rows = [], []
        for ordinal in range(64):
            image_path = dataset_root / f'{ordinal}.bin'
            image_path.write_bytes(bytes([ordinal]))
            paths.append(image_path)
            image_rows.append({'relative_path': image_path.name, 'image_sha256': sha(image_path)})
        verified, closed, seeds = [], [], []
        class FakeRGB:
            size = (256, 256)
            def __init__(self, ordinal):
                self.ordinal = ordinal
            def tobytes(self):
                return bytes([self.ordinal]) * 3
            def close(self):
                closed.append(self.ordinal)
        class ImageRows:
            def __init__(self, source_paths, labels, *, augment):
                assert source_paths == tuple(paths) and labels == tuple(range(64)) and augment is True
            def __getitem__(self, ordinal):
                assert verified[-1] == paths[ordinal]  # Actual SHA guard must precede decoding.
                return FakeRGB(ordinal), ordinal
        rng = SimpleNamespace(clone=lambda: 'RNG')
        fake_torch = SimpleNamespace(float32='FP32', equal=lambda a, b: (a == 'RNG' and b is rng),
            random=SimpleNamespace(get_rng_state=lambda: rng, fork_rng=lambda devices: nullcontext(),
                                   default_generator=SimpleNamespace(manual_seed=seeds.append)))
        augmentation_context = {'guards': {}, 'initialized': {'source_context': {'fit': {
            'dataset_root': str(dataset_root), 'targets': list(range(64)), 'rows': image_rows}},
            'pca': {'source_context': {'all_images': paths}}}}
        augmentation_state = {'target': list(range(64)), 'seed': 179041,
            'processor': lambda *, images, return_tensors: {'pixel_values': SimpleNamespace(
                dtype='FP32', shape=(len(images), 3, 256, 256))}}
        real_bound = driver.bound_file
        def verified_bound(guards, source_path, expected):
            result = real_bound(guards, source_path, expected)
            verified.append(result)
            return result
        with patch.dict(sys.modules, {'torch': fake_torch}), patch.object(driver, 'bound_file', verified_bound):
            _, rgb_sha = driver.augmented_pixels(augmentation_context, SimpleNamespace(ImageRows=ImageRows),
                                                augmentation_state, tuple(range(64)), 17)
            original_pair_digest = hashlib.sha256()
            for ordinal in range(64):
                original_pair_digest.update(str((256, 256)).encode())
                original_pair_digest.update(bytes([ordinal]) * 3)
            assert rgb_sha == original_pair_digest.hexdigest() and closed == list(range(64))
            assert seeds == [179032 * 100000 + 17]  # Independent of schedule seed179041.
            paths[-1].write_bytes(b'changed')
            rejects(lambda: driver.augmented_pixels(augmentation_context, SimpleNamespace(ImageRows=ImageRows),
                augmentation_state, tuple(range(64)), 18), 'SHA256')
            image_rows[0]['relative_path'] = '../escape.bin'
            rejects(lambda: driver.augmented_pixels(augmentation_context, SimpleNamespace(ImageRows=ImageRows),
                augmentation_state, tuple(range(64)), 18), 'resolution/containment')
        own, qualified = root / 'own', root / 'qualified'
        own.mkdir(); qualified.mkdir()
        originals = {'deployed_code_rank.py': path.parent.parent / 'src/sfora/deployed_code_rank.py',
                     'reference_train_sop_siglip2_compact.py': path.with_name('train_sop_siglip2_compact.py'),
                     'reference_unicom_training.py': path.parent.parent / 'src/sfora/unicom_training.py'}
        for name in driver.FILES:
            shutil.copyfile(originals.get(name, path.parent / name), own / name)
        for name in driver.QUALIFIER_FILES:
            source = path.parent / name if name != 'joint_relational_compaction.py' else path.parent.parent / 'src/sfora' / name
            shutil.copyfile(source, qualified / name)
        code = {name: sha(own / name) for name in driver.FILES}
        qualifier_code = {name: sha(qualified / name) for name in driver.QUALIFIER_FILES}
        execution = write(own / 'execution.json', code)
        qualifier_execution = write(qualified / 'execution.json', qualifier_code)
        args = SimpleNamespace(execution_sha256=execution, authority=root / 'launch.json', authority_sha256=None,
                               arm='large', seed=179032, phase='mechanics', output=root / 'NEW')
        launch = {'schema': driver.AUTHORITY_SCHEMA, 'execution_sha256': execution, 'phase': 'mechanics',
                  'arm': 'large', 'seed': 179032, 'qualifier_root': str(qualified),
                  'qualifier_execution_sha256': qualifier_execution,
                  'qualifier_authority': {'path': str(root / 'qualified-launch.json'), 'sha256': 'a'*64},
                  'selected_cpu': {}, 'selected_mechanics': None, 'resource_policy': driver.policy('mechanics'),
                  'both_locks_held': True}
        args.authority_sha256 = write(args.authority, launch)
        with patch.object(driver, '__file__', str(own / path.name)):
            _, actual_code, _, actual_launch, q = driver.bootstrap(args)
            assert actual_code == code and actual_launch == launch and q.FILES == driver.QUALIFIER_FILES
            sys.modules.pop('_siglip2_pinned_adaptation_qualifier')
            for key, value in (('schema', 'wrong'), ('execution_sha256', '0'*64), ('arm', 'so400'),
                               ('seed', 179041), ('phase', 'train'), ('both_locks_held', False),
                               ('selected_mechanics', {}), ('resource_policy', {**driver.policy('mechanics'), 'seconds': 301})):
                args.authority_sha256 = write(args.authority, {**launch, key: value})
                rejects(lambda: driver.bootstrap(args), 'launch authority')
            args.authority_sha256 = write(args.authority, launch)
            for broken in ({**code, 'extra.py': '0'*64}, {n: h for n, h in code.items() if n != path.name}):
                args.execution_sha256 = write(own / 'execution.json', broken)
                rejects(lambda: driver.bootstrap(args), 'exactly declared')
            args.execution_sha256 = write(own / 'execution.json', code)
            raw = (own / path.name).read_bytes()
            (own / path.name).write_bytes(raw + b'\n# tamper\n')
            rejects(lambda: driver.bootstrap(args), 'SHA256')
            (own / path.name).write_bytes(raw)
            with patch.dict(sys.modules, {'torch.fake': SimpleNamespace()}):
                rejects(lambda: driver.authority(args), 'preceded admission')
            rejects(lambda: driver.descriptor_json({'path': str(root / 'missing'), 'sha256': 'a'*64}, {}), 'canonical file')
        for name, pin in driver.REFERENCES.items():
            namespace = {'Dataset': object}
            exec(driver.selected_ast(own / name, pin), namespace)
            assert set(namespace) == {'Dataset', '__builtins__', *pin['names']}
            for symbol in pin['names']:
                obj = namespace[symbol]
                if isinstance(obj, type):
                    obj = obj.__init__
                assert obj.__code__.co_filename == str(own / name)
            if 'member_bank_refresh_rows' in namespace:
                assert namespace['member_bank_refresh_rows']((3, 1, 3, 1, 2)) == ((1, 2, 3), (3, 4, 2))
                rejects(lambda: namespace['member_bank_refresh_rows'](()), 'refresh batch')
            else:
                assert [(s.start, s.stop) for s in namespace['_class_slices'](7, 3)] == [(0, 3), (3, 5), (5, 7)]
            rejects(lambda: driver.selected_ast(own / name, {**pin, 'ast': '0'*64}), 'AST differs')
            rejects(lambda: driver.selected_ast(own / name, {**pin, 'names': (*pin['names'], 'missing')}), 'selected reference')
        selected = root / 'selected.py'
        selected.write_text('raise RuntimeError("top-level executed")\ndef wanted():\n    return 7\n')
        node = ast.parse(selected.read_bytes()).body[1]
        pin = {'source': sha(selected), 'names': ('wanted',), 'ast': hashlib.sha256(
            ast.dump(ast.Module(body=[node], type_ignores=[]), include_attributes=False).encode()).hexdigest()}
        namespace = {}
        exec(driver.selected_ast(selected, pin), namespace)
        assert namespace['wanted']() == 7
        selected.write_text(selected.read_text() + '\ndef wanted():\n    return 8\n')
        rejects(lambda: driver.selected_ast(selected, {**pin, 'source': sha(selected)}), 'selected reference')
        rejects(lambda: driver.load_bare('unpinned', selected, pin['source']), 'SHA256')
        rejects(lambda: driver.strict_json('{"a":1,"a":2}'), 'duplicate')
        rejects(lambda: driver.strict_json('{"a":NaN}'), 'nonfinite')
        # Reuse the actual original terminal parser, not a permissive replacement.
        init = module(path.with_name('initialize_siglip2_substrate_fit.py'), 'terminal_under_test')
        unit, invocation = 'fixture-initialized', 'a'*32
        memory = {'path': '/sys/fs/cgroup/' + unit + '.service', 'values': {
            'memory.max': str(8 * 1024**3), 'memory.current': '128', 'memory.peak': '256',
            'memory.swap.current': '0', 'memory.swap.peak': '0', 'memory.swap.max': '0',
            'memory.events': 'max 0\noom 0\noom_kill 0'}}
        record = {'wall_seconds': 1, 'process_peak_rss_kib': 1, 'cgroup_before': memory, 'cgroup_after': memory,
                  'invocation': {'invocation_id': invocation, 'optimize': 0}}
        log = root / 'original.log'
        logtext = (f'Running as unit: {unit}.service; invocation ID: {invocation}\n'
                   '\tExit status: 0\nFinished with result: success\n'
                   'Main processes terminated with: code=exited/status=0\n'
                   '\tSwaps: 0\nMemory swap peak: 0B\nService runtime: 111.519s\n'
                   '\tMaximum resident set size (kbytes): 2\nFINAL_CGROUP ' +
                   json.dumps({**memory, 'invocation_id': invocation}) + '\n')
        log.write_text(logtext)
        descriptor = {'receipt': {'path': str(root / 'proof.json'), 'sha256': 'b'*64},
                      'log': {'path': str(log), 'sha256': sha(log)}, 'unit': unit, 'invocation_id': invocation,
                      'service_seconds': 111.519, 'native_peak_rss_kib': 2, 'both_locks_held': True}
        assert init.admit_terminal(record, descriptor, 120, {})['invocation_id'] == invocation
        for badtext, marker in ((logtext.replace('status=0', 'status=1'), 'normal-exit'),
                                (logtext.replace('max 0', 'max 1'), 'memory failure'),
                                (logtext.replace('swap.peak": "0', 'swap.peak": "1'), 'memory/swap')):
            log.write_text(badtext)
            descriptor['log']['sha256'] = sha(log)
            rejects(lambda: init.admit_terminal(record, descriptor, 120, {}), marker)
        log.write_text(logtext)
        descriptor['log']['sha256'] = sha(log)
        rejects(lambda: init.admit_terminal(record, {**descriptor, 'service_seconds': 120.001}, 120, {}), 'duration/RSS')
        initialized = {'code': {'qualifier': 'fixed'}, 'launch': {
            'initializer_authority': {'path': '/original/pca.json', 'sha256': 'c'*64},
            'selected_initializer': {'original': True}}, 'pca_final_cgroup': memory,
            'record': {'source_binding': {'original': True}, 'ordered_input_sha256': 'd'*64,
                       'ordered_rgb_sha256': 'e'*64, 'artifact': {'path': '/original/initializers.npz', 'sha256': 'f'*64},
                       'arrays': {'original': True}},
            'source_context': {'proof': {'sample': {'original': True}, 'runtime': {'original': True},
                                       'numerical_flags': {'threads': 1}}},
            'packages': {}, 'pca': {'startup': {'invocation': {
                'python': '/original/python', 'python_sha256': '1'*64, 'python_version': 'original'}}}, 'init': init}
        launch['selected_cpu'] = descriptor
        cpu = {**record, 'schema': q.SCHEMA, 'phase': 'initialized-cpu', 'arm': 'large', 'width': 1024,
               'output_dim': 128, 'authority_sha256': launch['qualifier_authority']['sha256'],
               'execution_sha256': launch['qualifier_execution_sha256'], 'code': initialized['code'],
               'initializer_authority': initialized['launch']['initializer_authority'],
               'selected_initializer': initialized['launch']['selected_initializer'], 'pca_final_cgroup': memory,
               'source_binding': initialized['record']['source_binding'],
               'source_sample': initialized['source_context']['proof']['sample'],
               'ordered_input_sha256': 'd'*64, 'ordered_rgb_sha256': 'e'*64,
               'initializers': initialized['record']['artifact'], 'resource_policy': q.POLICY,
               'numerical_flags': {'threads': 1}, 'augmentation': q.AUGMENTATION,
               'state': {'counter': 0, 'seed': 179032, 'arrays': initialized['record']['arrays'],
                         'runtime': initialized['source_context']['proof']['runtime'],
                         'parameter_names': list(range(208))},
               'updates': 0, 'head_updates': 0, 'optimizer_state_entries': 0,
               'checkpoint': {'path': str(root / 'initialized.pt'), 'sha256': 'a'*64},
               'input_guards': {'/original/input': '2'*64}, 'origins': {'packages': {}, 'files': {}}}
        for key in ('pass', 'initializer_qualified', 'source_qualified', 'reload_exact', 'fresh_source',
                    'first_model_released_before_independent_clone', 'source_cpu_runtime_and_first2_exact',
                    'constructor_rng_preserved', 'exit_rehash_pass', 'optimizer_created',
                    'both_locks_held_in_parent_authority', 'terminal_exit_and_both_locks_require_parent_receipt'):
            cpu[key] = True
        for key in ('training_qualified', 'quality_qualified', 'quality_read', 'cuda_initialized', 'gradients_created',
                    'pca_rerun', 'teacher_state_reused', 'trained_state_reused'):
            cpu[key] = False
        cpu['invocation'] = {**cpu['invocation'], **initialized['pca']['startup']['invocation'],
            'cuda_visible_devices': '', 'argv': [str(qualified / 'qualify_siglip2_initialized_cpu.py'),
                '--execution-sha256', qualifier_execution, '--authority', launch['qualifier_authority']['path'],
                '--authority-sha256', launch['qualifier_authority']['sha256'], '--arm', 'large', '--output', str(root)]}
        context = {'qualifier': q, 'launch': launch, 'initialized': initialized, 'args': args, 'guards': {},
                   'initialized_prereq_guards': {'/original/input': '2'*64}}
        descriptor['receipt']['sha256'] = write(root / 'proof.json', cpu)
        mechanics_root = root / 'mechanics'
        mechanics_root.mkdir()
        mechanics_descriptor = {**descriptor, 'receipt': {'path': str(mechanics_root / 'receipt.json'), 'sha256': None}}
        train_launch = {**launch, 'phase': 'train', 'seed': 179041, 'selected_mechanics': mechanics_descriptor,
                        'resource_policy': driver.policy('train')}
        original_launch_path = root / 'mechanics-launch.json'
        original_launch_sha = write(original_launch_path, launch)
        mechanics = {**record, 'schema': driver.SCHEMA, 'phase': 'mechanics', 'arm': 'large', 'seed': 179032,
            'execution_sha256': execution, 'code': code, 'selected_cpu': descriptor,
            'reference_pins': json.loads(json.dumps(driver.REFERENCES)), 'rank_helper_sha256': driver.RANK_SHA256,
            'qualifier_authority': launch['qualifier_authority'], 'resource_policy': driver.policy('mechanics'),
            'augmentation': driver.AUGMENTATION, 'pass': True, 'quality_read': False, 'completed_step': 17,
            'training_state_discarded': True, 'native17_equals_serialized8_plus9_exact': True,
            'strict_independent_whole_head_buffers_raw_packed_reload_exact': True,
            'first_references_released_before_reload': True, 'exit_rehash_pass': True,
            'peak_cuda_allocated_bytes': 1024,
            'steps': [{'step': step, 'seconds': 1, 'loss': float(step)} for step in range(1, 18)],
            'resumed_steps': [{'step': step, 'seconds': 2, 'loss': float(step)} for step in range(9, 18)],
            'authority_sha256': original_launch_sha,
            'invocation': {**cpu['invocation'], 'cuda_visible_devices': '0', 'argv': [str(own / path.name),
                '--execution-sha256', execution, '--authority', str(original_launch_path),
                '--authority-sha256', original_launch_sha, '--phase', 'mechanics', '--arm', 'large',
                '--seed', '179032', '--output', str(mechanics_root)]}}
        mechanics_descriptor['receipt']['sha256'] = write(mechanics_root / 'receipt.json', mechanics)
        context.update(launch=train_launch, root=own, code=code, cpu=cpu, guards={})
        # The same seed179032 mechanics proof admits BOTH separately CPU-pinned schedules.
        for seed in driver.SEEDS:
            args.seed = seed
            assert driver.admit_mechanics(context)[0] == mechanics
        for key, value in (('arm', 'so400'), ('seed', 179041), ('training_state_discarded', False),
                           ('native17_equals_serialized8_plus9_exact', False),
                           ('strict_independent_whole_head_buffers_raw_packed_reload_exact', False),
                           ('peak_cuda_allocated_bytes', 10_000_000_000), ('resumed_steps', [])):
            context['guards'] = {}
            mechanics_descriptor['receipt']['sha256'] = write(mechanics_root / 'receipt.json', {**mechanics, key: value})
            rejects(lambda: driver.admit_mechanics(context), 'mechanics proof')
        context['guards'] = {}
        broken = copy.deepcopy(mechanics)
        broken['resumed_steps'][0]['loss'] += 1
        mechanics_descriptor['receipt']['sha256'] = write(mechanics_root / 'receipt.json', broken)
        rejects(lambda: driver.admit_mechanics(context), 'mechanics proof')
        assert driver.admit_cpu(context)[0] == cpu
        for key, value, marker in (('arm', 'so400', 'CPU binding'), ('pass', False, 'CPU qualification'),
                                   ('updates', 1, 'CPU qualification'), ('training_qualified', True, 'CPU qualification'),
                                   ('quality_read', True, 'CPU qualification'), ('input_guards', {}, 'input guards'),
                                   ('state', {**cpu['state'], 'parameter_names': []}, 'CPU qualification')):
            descriptor['receipt']['sha256'] = write(root / 'proof.json', {**cpu, key: value})
            context['guards'] = {}
            rejects(lambda: driver.admit_cpu(context), marker)
        descriptor['receipt']['sha256'] = write(root / 'proof.json', cpu)
    with TemporaryDirectory() as directory, patch.dict(sys.modules):
        flat_chain_checks(driver, Path(directory).resolve())
    assert not any(n.split('.')[0] in driver.NATIVE for n in sys.modules)
    print('PASS stdlib checkpoint-mmap/consumed-order/ownership/failures/native-contract/flat-chain/predicate-ledger/stage-guards/unique-bulk-SHA/uncached-exit-tamper/authority/reference/math/schedule/full-state/JSON/RGB/source-byte/checkpoint-write/ZIP/phase-log checks; native unrun')


if __name__ == '__main__':
    main()
