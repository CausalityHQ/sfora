#!/usr/bin/env python3
"""Bounded stdlib falsifier (<=15s, AS1GiB): the actual caller/serializer/authority seams run
against a FAKE torch and FAKE native ABI. Real Torch, NumPy, CUDA, compiler, GPU, SSH and dlopen
are UNRUN; a pass here is a source falsifier, not a native, timing or product result."""
import ast
import contextlib
import ctypes
import gc
import hashlib
import importlib.util
import io
import itertools
import json
import os
from pathlib import Path
import py_compile
import re
import struct
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import weakref

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SCRIPT = HERE / 'qualify_cuda_sha256_native.py'
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
NATIVE = {'torch', 'numpy', 'PIL', 'sfora', 'transformers', 'torchvision', 'safetensors'}
if NATIVE & {name.split('.')[0] for name in sys.modules}:
    raise SystemExit('native package already imported; refusing to run')
SERIALIZERS = {'probe': ROOT / 'src/sfora/connected_probe_inference.py',
               'mlp': ROOT / 'src/sfora/connected_inference.py'}
CUDA = ROOT / 'rust/sfora-cuda-sha256/sha256_occurrences.cu'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fact(path):
    return {'path': str(path), 'sha256': sha(Path(path).read_bytes())}


def load(name):
    spec = importlib.util.spec_from_file_location(name, str(SCRIPT))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


mod = load('qualify_cuda_sha256_native_under_test')
INTERPRETER = None


# ------------------------------------------------------------------ fake torch
class Dtype:
    def __init__(self, name, size, code):
        self.name, self.size, self.code = name, size, code

    def __str__(self):
        return self.name

    __repr__ = __str__


F32, F64, I64, U8, I32 = (Dtype('torch.float32', 4, 'f'), Dtype('torch.float64', 8, 'd'),
                          Dtype('torch.int64', 8, 'q'), Dtype('torch.uint8', 1, 'B'), Dtype('torch.int32', 4, 'i'))


class Device:
    def __init__(self, type, index=None):
        self.type, self.index = type, index

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)

    def __hash__(self):
        return hash((self.type, self.index))

    def __repr__(self):
        return '%s:%s' % (self.type, self.index)


class Arena:
    def __init__(self, base, size):
        self.base, self.mem, self.top = base, bytearray(size), 512

    def alloc(self, nbytes):
        address = self.base + self.top
        self.top += (max(nbytes, 1) + 511) // 512 * 512
        if self.top > len(self.mem):
            raise MemoryError('fake arena exhausted')
        return address

    def read(self, address, count):
        return bytes(self.mem[address - self.base:address - self.base + count])

    def write(self, address, raw):
        self.mem[address - self.base:address - self.base + len(raw)] = raw


class Storage:
    def __init__(self, address, size):
        self.address, self.size = address, size

    def data_ptr(self):
        return self.address

    def nbytes(self):
        return self.size


class Scalar:
    def __init__(self, owner, index, value):
        self.owner, self.index, self.value = owner, index, value

    def __ixor__(self, mask):
        return Scalar(self.owner, self.index, self.value ^ mask)


def contiguous_strides(shape):
    strides, step = [], 1
    for dim in reversed(shape):
        strides.append(step)
        step *= max(dim, 1)
    return tuple(reversed(strides))


class Tensor:
    def __init__(self, torch, storage, byte_offset, shape, strides, dtype, device, counter=None):
        self.torch, self.storage, self.byte_offset = torch, storage, byte_offset
        self.shape, self.strides, self.dtype, self.device = tuple(shape), tuple(strides), dtype, device
        self.counter = counter if counter is not None else [0]
        self.pinned = False

    layout = property(lambda self: self.torch.strided)
    is_cuda = property(lambda self: self.device.type == 'cuda')
    _version = property(lambda self: self.counter[0])
    data = property(lambda self: Tensor(self.torch, self.storage, self.byte_offset, self.shape, self.strides,
                                        self.dtype, self.device))

    def element_size(self):
        return self.dtype.size

    def numel(self):
        return len(list(itertools.product(*(range(d) for d in self.shape)))) if self.shape else 1

    def is_contiguous(self):
        if 0 in self.shape:
            return True
        expected = 1
        for dim, stride in zip(reversed(self.shape), reversed(self.strides)):
            if dim != 1:
                if stride != expected:
                    return False
                expected *= dim
        return True

    def data_ptr(self):
        return self.storage.address + self.byte_offset if self.storage.size else 0

    def storage_offset(self):
        return self.byte_offset // self.dtype.size

    def untyped_storage(self):
        return self.storage

    def arena(self):
        return self.torch.device_arena if self.is_cuda else self.torch.host_arena

    def offsets(self):
        for index in itertools.product(*(range(d) for d in self.shape)):
            yield self.byte_offset + sum(i * s for i, s in zip(index, self.strides)) * self.dtype.size

    def logical(self):
        return b''.join(self.arena().read(self.storage.address + o, self.dtype.size) for o in self.offsets())

    def write_logical(self, raw):
        for i, offset in enumerate(self.offsets()):
            self.arena().write(self.storage.address + offset, raw[i * self.dtype.size:(i + 1) * self.dtype.size])
        self.counter[0] += 1

    def share(self, byte_offset, shape, strides, dtype):
        return Tensor(self.torch, self.storage, byte_offset, shape, strides, dtype, self.device, self.counter)

    def detach(self):
        return self.share(self.byte_offset, self.shape, self.strides, self.dtype)

    def cpu(self):
        self.torch.hook('cpu', self)
        self.torch.log.append(('cpu', self.dtype.name, self.shape))
        if not self.is_cuda:
            return self
        return self.torch.make(Device('cpu'), self.shape, self.dtype, self.logical())

    def to(self, device, non_blocking=False):
        self.torch.log.append(('copy', device.type, non_blocking, self.pinned))
        return self if device == self.device else self.torch.make(device, self.shape, self.dtype, self.logical())

    def pin_memory(self):
        self.pinned = True
        self.torch.log.append(('pin',))
        return self

    def contiguous(self):
        return self if self.is_contiguous() else self.torch.make(self.device, self.shape, self.dtype, self.logical())

    def clone(self):
        return self.torch.make(self.device, self.shape, self.dtype, self.logical())

    def copy_(self, other):
        self.write_logical(other.logical())
        return self

    def reshape(self, *shape):
        shape = shape[0] if len(shape) == 1 and isinstance(shape[0], tuple) else shape
        count = len(list(itertools.product(*(range(d) for d in self.shape)))) if self.shape else 1
        shape = tuple(count if d == -1 else d for d in shape)
        if not self.is_contiguous():
            raise ValueError('fake reshape needs a contiguous tensor')
        return self.share(self.byte_offset, shape, contiguous_strides(shape), self.dtype)

    def view(self, dtype):
        if not self.is_contiguous():
            raise ValueError('fake view needs a contiguous tensor')
        shape = self.shape[:-1] + (self.shape[-1] * self.dtype.size // dtype.size,)
        return self.share(self.byte_offset, shape, contiguous_strides(shape), dtype)

    def t(self):
        return self.share(self.byte_offset, self.shape[::-1], self.strides[::-1], self.dtype)

    def expand(self, count):
        return self.share(self.byte_offset, (count,), (0,), self.dtype)

    def tolist(self):
        raw = self.logical()
        values = struct.unpack('<%d%s' % (len(raw) // self.dtype.size, self.dtype.code), raw)
        return list(values)

    def numpy(self):
        return self.logical()

    def __getitem__(self, key):
        if isinstance(key, slice):
            start, stop, step = key.indices(self.shape[0])
            assert step == 1
            return self.share(self.byte_offset + start * self.strides[0] * self.dtype.size, (max(0, stop - start),),
                              self.strides, self.dtype)
        return Scalar(self, key, self.tolist()[key])

    def __setitem__(self, key, value):
        raw = bytearray(self.logical())
        raw[key] = value.value if isinstance(value, Scalar) else value
        self.write_logical(bytes(raw))


class Stream:
    def __init__(self, torch, device, handle, token='default'):
        self.torch, self.device, self.cuda_stream, self.token = torch, device, handle, token

    def synchronize(self):
        self.torch.hook('sync', self)
        self.torch.log.append(('sync', self.cuda_stream))
        self.torch.pending.discard(self.cuda_stream)

    def __eq__(self, other):
        return isinstance(other, Stream) and (self.device, self.cuda_stream, self.token) == \
            (other.device, other.cuda_stream, other.token)

    def __hash__(self):
        return hash((self.cuda_stream, self.token))


class Event:
    def __init__(self, torch):
        self.torch, self.stream = torch, None

    def record(self, stream):
        self.stream = stream

    def query(self):
        return self.stream.cuda_stream not in self.torch.pending


class Cuda:
    def __init__(self, torch):
        self.torch = torch

    is_available = staticmethod(lambda: True)
    current_device = staticmethod(lambda: 0)
    max_memory_allocated = staticmethod(lambda: 123)

    def get_device_properties(self, index):
        return types.SimpleNamespace(name='Fake GPU', major=12, minor=1)

    def current_stream(self, device=None):
        return self.torch.current

    def default_stream(self, device=None):
        return self.torch.default

    def Stream(self, device=None):
        self.torch.streams += 1
        return Stream(self.torch, Device('cuda', 0), 0x1000 + self.torch.streams, 'side')

    @contextlib.contextmanager
    def stream(self, stream):
        previous, self.torch.current = self.torch.current, stream
        try:
            yield
        finally:
            self.torch.current = previous

    def Event(self):
        return Event(self.torch)

    def _sleep(self, cycles):
        self.torch.pending.add(self.torch.current.cuda_stream)
        self.torch.log.append(('sleep', cycles))

    def synchronize(self):
        self.torch.log.append(('global_sync',))


class Torch:
    Tensor = Tensor
    float32, float64, int64, uint8, int32 = F32, F64, I64, U8, I32
    strided = 'strided'
    __version__ = 'fake'

    def __init__(self):
        self.log, self.hooks, self.pending, self.streams, self.made = [], {}, set(), 0, []
        self.device_arena, self.host_arena = Arena(0x7000_0000_0000, 1 << 24), Arena(0x5000_0000_0000, 1 << 24)
        self.version = types.SimpleNamespace(cuda='fake')
        self.default = Stream(self, Device('cuda', 0), 0)
        self.current = self.default
        self.cuda = Cuda(self)

    def hook(self, name, value):
        if name in self.hooks:
            self.hooks[name](value)

    def device(self, kind, index=None):
        return Device(kind, index)

    def make(self, device, shape, dtype, raw=None):
        shape = tuple(shape)
        count = len(list(itertools.product(*(range(d) for d in shape)))) if shape else 1
        size = count * dtype.size
        arena = self.device_arena if device.type == 'cuda' else self.host_arena
        storage = Storage(arena.alloc(size) if size else 0, size)
        result = Tensor(self, storage, 0, shape, contiguous_strides(shape), dtype, device)
        self.made.append(weakref.ref(result))
        if raw:
            result.write_logical(raw)
            result.counter[0] = 0
        return result

    def tensor(self, values, dtype=None, device=None):
        return self.make(device or Device('cpu'), (len(values),), dtype, struct.pack('<%d%s' % (len(values), dtype.code), *values))

    def empty(self, *shape, dtype=None, device=None):
        self.hook('empty', shape)
        return self.make(device or Device('cpu'), shape, dtype)

    def zeros(self, *shape, dtype=F32, device=None):
        return self.make(device or Device('cpu'), shape, dtype)

    def frombuffer(self, buffer, dtype=None):
        return self.make(Device('cpu'), (len(buffer) // dtype.size,), dtype, bytes(buffer))


class Native:
    """Bit-exact fake of the C ABI including its host-side argument rejection."""

    def __init__(self, torch, status=0):
        self.torch, self.status, self.calls, self.reads, self.hook = torch, status, [], [], None

    def __call__(self, ptrs, lens, count, out, stream):
        ptrs, lens, out, stream = (v or 0 for v in (ptrs, lens, out, stream))
        self.calls.append((ptrs, lens, count, out, stream))
        if self.hook:
            self.hook()
        if count > 2**32 - 1:
            return 1
        if count == 0:
            return 0
        if not ptrs or not lens or not out:
            return 1
        arena, occurrences = self.torch.device_arena, []
        for i in range(count):
            address, = struct.unpack('<q', arena.read(ptrs + 8 * i, 8))
            length, = struct.unpack('<q', arena.read(lens + 8 * i, 8))
            occurrences.append((address, length))
            arena.write(out + 32 * i, hashlib.sha256(arena.read(address, length) if length else b'').digest())
        self.reads.append(occurrences)
        self.torch.log.append(('launch', stream))
        return self.status


@contextlib.contextmanager
def fake_modules(torch):
    assert 'torch' not in sys.modules
    sys.modules['torch'] = torch
    try:
        yield
    finally:
        del sys.modules['torch']


def pattern(count, salt=0):
    return mod.pattern(count, salt)


class Base(unittest.TestCase):
    def setUp(self):
        self.torch = Torch()
        self.device = self.torch.device('cuda', 0)
        self.native = Native(self.torch)
        self.hasher = mod.Sha256Native(self.torch, self.native)

    def leaf(self, raw):
        if not raw:
            return self.torch.empty(0, dtype=F32, device=self.device)
        return self.torch.frombuffer(bytearray(raw), dtype=F32).to(self.device)

    def order(self, *names):
        positions = [next(i for i, e in enumerate(self.torch.log) if e[0] == name) for name in names]
        self.assertEqual(positions, sorted(positions))


# ------------------------------------------------------------------ source contract + ABI
class AbiTests(Base):
    def test_cuda_signature_matches_the_bound_ctypes_abi(self):
        source = CUDA.read_text()
        match = re.search(r'extern "C" cudaError_t sfora_sha256_occurrences\((.*?)\) \{', source, re.S)
        params = [' '.join(p.split()) for p in match[1].split(',')]
        types_ = [re.sub(r'\s*\w+$', '', p) for p in params]
        self.assertEqual(types_, ['const unsigned char* const*', 'const uint64_t*', 'uint64_t', 'unsigned char*', 'cudaStream_t'])
        lib = types.SimpleNamespace(sfora_sha256_occurrences=lambda *a: 0)
        function = mod.bind_abi(lib, ctypes)
        self.assertEqual(function.argtypes, [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p])
        self.assertIs(function.restype, ctypes.c_int)
        with self.assertRaisesRegex(ValueError, 'symbol missing'):
            mod.bind_abi(types.SimpleNamespace(), ctypes)

    def test_real_ctypes_five_argument_round_trip(self):
        proto = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p)
        callback = proto(self.native)
        pointer = ctypes.CFUNCTYPE(ctypes.c_int)(ctypes.cast(callback, ctypes.c_void_p).value)
        function = mod.bind_abi(types.SimpleNamespace(sfora_sha256_occurrences=pointer), ctypes)
        with self.assertRaises(TypeError):
            function(1, 2, 3, 4)
        raws = [pattern(n, n) for n in (1, 13, 16)]
        hasher = mod.Sha256Native(self.torch, function)
        self.assertEqual(hasher.digests([self.leaf(r) for r in raws]), [sha(r) for r in raws])
        self.assertEqual(self.native.calls[-1][4], 0)
        self.assertEqual(function(None, None, 0, None, None), 0)
        self.assertEqual(function(None, None, 2**32, None, None), 1)
        self.assertEqual(function(None, None, 1, None, None), 1)


# ------------------------------------------------------------------ pre-enqueue validation
class PlanTests(Base):
    def plan(self, leaves):
        return mod.plan_occurrences(self.torch, leaves, self.device)

    def test_valid_scalar_empty_offset_duplicate(self):
        base = self.leaf(pattern(40, 7))
        scalar, empty = base[5:6].reshape(()), self.leaf(b'')
        spans = self.plan([base, scalar, empty, base[3:20], base[3:20], base[2:12], base[7:17]])
        self.assertEqual([n for _, n in spans], [160, 4, 0, 68, 68, 40, 40])
        self.assertEqual(spans[1][0], base.data_ptr() + 20)
        self.assertEqual(spans[2], (0, 0))
        self.assertEqual(spans[3], spans[4])

    def test_rejections(self):
        good = self.leaf(pattern(8))
        host = self.torch.zeros(4)
        wrong_device = self.torch.make(Device('cuda', 1), (4,), F32)
        odd = good.share(2, (4,), (1,), F32)
        beyond = good.share(8 * 4, (4,), (1,), F32)
        null = self.torch.make(self.device, (4,), F32)
        null.storage = Storage(0, 16)
        stale = good.share(4, (4,), (1,), F32)
        stale.data_ptr = lambda: good.data_ptr() + 8
        cases = {'cpu': host, 'device': wrong_device, 'float64': self.torch.make(self.device, (4,), F64),
                 'int64': self.torch.make(self.device, (4,), I64), 'int32': self.torch.make(self.device, (4,), I32), 'transposed': self.torch.make(self.device, (4, 4), F32).t(),
                 'stride_zero': self.torch.make(self.device, (1,), F32).expand(4), 'unaligned': odd,
                 'past_storage': beyond, 'null_nonempty': null, 'storage_mismatch': stale, 'non_tensor': 3}
        for name, value in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.plan([good, value])
        with self.assertRaises(ValueError):
            self.plan((good,))
        with mock.patch.object(mod, 'MAX_COUNT', 1), self.assertRaisesRegex(ValueError, 'within 2'):
            self.plan([good, good])

    def test_rejection_happens_before_any_device_effect(self):
        good = self.leaf(pattern(8))
        before = list(self.torch.log)
        for bad in (self.torch.zeros(4), self.torch.make(self.device, (4,), F64)):
            with self.assertRaises(ValueError):
                self.hasher.digests([good, bad])
        self.assertEqual(self.torch.log, before)
        self.assertEqual(self.native.calls, [])
        self.assertFalse(self.hasher.poisoned)


# ------------------------------------------------------------------ the caller
class CallerTests(Base):
    def test_parity_all_lengths_and_zero(self):
        self.assertEqual(self.hasher.digests([]), [])
        self.assertEqual(self.native.calls, [])
        self.assertEqual(self.torch.log, [])
        self.assertEqual(self.hasher.digests([]), [])
        for count in (1, 127, 128, 129, 130, 257):
            raws = [pattern(n % 40, n) for n in range(count)]
            self.assertEqual(self.hasher.digests([self.leaf(r) for r in raws]), [sha(r) for r in raws])
            self.assertEqual(len(self.native.reads[-1]), count)
        for n in range(0, 80):
            raw = pattern(n, n + 3)
            self.assertEqual(self.hasher.digests([self.leaf(raw)]), [sha(raw)])

    def test_special_bit_patterns_are_literal(self):
        raw = struct.pack('<8I', *mod.SPECIAL)
        self.assertEqual(self.hasher.digests([self.leaf(raw)]), [sha(raw)])

    def test_aliases_duplicates_and_order_hash_every_occurrence(self):
        base_raw = pattern(40, 7)
        base = self.leaf(base_raw)
        view, same = base[3:20], base[3:20]
        leaves = [base, view, same, view, base[2:12], base[7:17]]
        raws = [base_raw, base_raw[12:80], base_raw[12:80], base_raw[12:80], base_raw[8:48], base_raw[28:68]]
        self.assertEqual(self.hasher.digests(leaves), [sha(r) for r in raws])
        self.assertEqual(len(self.native.calls), 1)
        self.assertEqual(len(self.native.reads[0]), 6)
        self.assertEqual(self.hasher.digests(leaves[::-1]), [sha(r) for r in raws[::-1]])
        self.assertEqual(self.native.reads[1], self.native.reads[0][::-1])

    def test_current_stream_and_nonblocking_ordering(self):
        side = self.torch.cuda.Stream(device=self.device)
        leaves = [self.leaf(pattern(16))]
        for stream in (self.torch.default, side):
            start = len(self.torch.log)
            with self.torch.cuda.stream(stream):
                self.assertEqual(self.hasher.digests(leaves), [sha(pattern(16))])
            log = self.torch.log[start:]
            self.assertEqual(self.native.calls[-1][4], stream.cuda_stream)
            kinds = [e[0] for e in log]
            self.assertEqual([k for k in kinds if k in ('pin', 'launch', 'sync', 'cpu')],
                             ['pin', 'pin', 'launch', 'sync', 'cpu'])
            tables = [e for e in log if e[0] == 'copy']
            self.assertEqual(len(tables), 2)
            self.assertTrue(all(e[1:] == ('cuda', True, True) for e in tables))
            self.assertNotIn(('global_sync',), log)
            self.assertLess(max(i for i, e in enumerate(log) if e[0] == 'copy'), kinds.index('launch'))

    def test_stream_identity_rejections(self):
        leaves = [self.leaf(pattern(16))]
        self.torch.current = Stream(self.torch, self.device, 0, 'rogue')
        with self.assertRaisesRegex(ValueError, 'current CUDA stream'):
            self.hasher.digests(leaves)
        self.torch.current = Stream(self.torch, self.torch.device('cuda', 1), 0x77, 'wrong-device')
        with self.assertRaisesRegex(ValueError, 'current CUDA stream'):
            self.hasher.digests(leaves)
        self.assertEqual(self.native.calls, [])
        self.torch.current = self.torch.default
        side = self.torch.cuda.Stream(device=self.device)
        self.native.hook = lambda: setattr(self.torch, 'current', self.torch.default)
        with self.torch.cuda.stream(side), self.assertRaisesRegex(ValueError, 'stream changed during launch'):
            self.hasher.digests(leaves)
        self.assertIn(('sync', side.cuda_stream), self.torch.log)

    def test_stream_change_between_table_copies_and_launch_is_rejected(self):
        leaves = [self.leaf(pattern(16))]
        side = self.torch.cuda.Stream(device=self.device)
        self.torch.hooks['empty'] = lambda shape: setattr(self.torch, 'current', self.torch.default)
        with self.torch.cuda.stream(side), self.assertRaisesRegex(ValueError, 'changed before launch'):
            self.hasher.digests(leaves)
        self.assertEqual(self.native.calls, [])
        self.assertIn(('sync', side.cuda_stream), self.torch.log)
        self.assertFalse(self.hasher.poisoned)

    def test_buffer_validation_after_enqueue_still_drains(self):
        leaves = [self.leaf(pattern(16)), self.leaf(pattern(5))]
        for name, make_bad in (('alias', lambda t, dev: leaves[0].share(0, (64,), (1,), U8)),
                               ('dtype', lambda t, dev: t.make(dev, (64,), F32)),
                               ('small', lambda t, dev: t.make(dev, (8,), U8))):
            with self.subTest(name):
                self.torch.log.clear()
                self.native.calls.clear()
                with mock.patch.object(Torch, 'empty', lambda t, *shape, dtype=None, device=None: make_bad(t, device)):
                    with self.assertRaises(ValueError):
                        self.hasher.digests(leaves)
                self.assertEqual(self.native.calls, [])
                self.assertIn(('sync', 0), self.torch.log)
                self.assertFalse(self.hasher.poisoned)
        self.assertEqual(self.hasher.digests(leaves), [sha(pattern(16)), sha(pattern(5))])

    def test_launch_completion_readback_failures_return_nothing_and_recover(self):
        leaves = [self.leaf(pattern(16)), self.leaf(pattern(5))]
        expected = [sha(pattern(16)), sha(pattern(5))]
        injections = {'status': lambda: setattr(self.native, 'status', 5),
                      'sync_once': lambda: self.torch.hooks.__setitem__('sync', self.fail_once()),
                      'cpu_once': lambda: self.torch.hooks.__setitem__('cpu', self.fail_once()),
                      'fault': lambda: setattr(self.hasher, 'fault', self.fault_once('completed'))}
        for name, arm in injections.items():
            with self.subTest(name):
                arm()
                with self.assertRaises((ValueError, RuntimeError)):
                    self.hasher.digests(leaves)
                self.native.status = 0
                self.torch.hooks.clear()
                self.hasher.fault = None
                self.assertFalse(self.hasher.poisoned or self.hasher.quarantine)
                self.assertEqual(self.hasher.digests(leaves), expected)

    @staticmethod
    def fail_once():
        seen = []

        def hook(value):
            if not seen:
                seen.append(1)
                raise RuntimeError('injected')
        return hook

    @staticmethod
    def fault_once(point):
        seen = []

        def fault(name):
            if name == point and not seen:
                seen.append(1)
                raise mod.InjectedFault('injected')
        return fault

    def test_owners_survive_until_drain_then_release(self):
        refs, alive = {}, []
        leaves = [self.leaf(pattern(16)), self.leaf(pattern(5))]
        for i, leaf in enumerate(leaves):
            refs[i] = weakref.ref(leaf)
        del leaf

        def hook(stream):
            alive.append([ref() is not None for ref in refs.values()])
            leaves.clear()  # the caller drops every reference during the drain
        self.native.status = 9
        self.torch.hooks['sync'] = hook
        made = len(self.torch.made)
        try:  # not assertRaises: it clears the traceback frames and would hide a pinned owner
            self.hasher.digests(leaves)
        except ValueError as error:
            kept = error
        gc.collect()
        self.assertEqual(alive, [[True, True]], 'inputs were not retained through the drain')
        self.assertTrue(all(ref() is None for ref in refs.values()), 'error frames pinned the inputs')
        self.assertEqual([ref for ref in self.torch.made[made:] if ref() is not None], [],
                         'tables, staging or output stayed pinned by the retained error')
        self.assertFalse(self.hasher.poisoned or self.hasher.quarantine)
        frames = []
        trace = kept.__traceback__
        while trace is not None:
            frames.append(trace.tb_frame.f_code.co_name)
            trace = trace.tb_next
        self.assertIn('digests', frames)
        self.torch.hooks.clear()
        self.native.status = 0
        self.assertEqual(self.hasher.digests([self.leaf(pattern(16)), self.leaf(pattern(5))]), [sha(pattern(16)), sha(pattern(5))])

    def test_failed_drain_quarantines_every_owner_and_poisons(self):
        refs = []
        leaves = [self.leaf(pattern(16))]
        refs.append(weakref.ref(leaves[0]))
        self.native.status = 3
        self.torch.hooks['sync'] = lambda stream: (_ for _ in ()).throw(RuntimeError('drain failed'))
        try:
            self.hasher.digests(leaves)
        except ValueError as error:
            kept = error
        self.assertTrue(any('drain also failed' in note for note in kept.__notes__))
        del leaves
        gc.collect()
        self.assertTrue(self.hasher.poisoned and len(self.hasher.quarantine) == 1)
        self.assertIsNotNone(refs[0](), 'owner released after a failed drain')
        self.assertGreaterEqual(len(self.hasher.quarantine[0]), 4)
        self.torch.hooks.clear()
        calls = len(self.native.calls)
        with self.assertRaisesRegex(ValueError, 'poisoned'):
            self.hasher.digests([self.leaf(pattern(4))])
        self.assertEqual(len(self.native.calls), calls)

    def test_keyboard_interrupt_still_drains_and_propagates(self):
        leaves = [self.leaf(pattern(16))]
        sync_calls = []

        def hook(stream):
            sync_calls.append(1)
            if len(sync_calls) == 1:
                raise KeyboardInterrupt
        self.torch.hooks['sync'] = hook
        with self.assertRaises(KeyboardInterrupt):
            self.hasher.digests(leaves)
        self.assertEqual(len(sync_calls), 2)
        self.assertFalse(self.hasher.poisoned)

    def test_oom_after_table_copies_drains(self):
        def boom(shape):
            raise MemoryError('injected oom')
        self.torch.hooks['empty'] = boom
        with self.assertRaises(MemoryError):
            self.hasher.digests([self.leaf(pattern(4))])
        self.assertIn(('sync', 0), self.torch.log)
        self.assertEqual(self.native.calls, [])
        self.torch.hooks.clear()
        self.assertEqual(self.hasher.digests([self.leaf(pattern(4))]), [sha(pattern(4))])

    def test_reentry_rejected(self):
        leaves = [self.leaf(pattern(4))]
        seen = []

        def nested():
            try:
                self.hasher.digests(leaves)
            except ValueError as error:
                seen.append(str(error))
        self.native.hook = nested
        self.assertEqual(self.hasher.digests(leaves), [sha(pattern(4))])
        self.assertEqual(len(seen), 1)
        self.assertIn('reentrant', seen[0])


# ------------------------------------------------------------------ cursor + original serializer
def original_fingerprint(path):
    nodes = [n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint']
    return mod.load_fingerprint(path.read_bytes())[0], ast.dump(nodes[0])


class CursorTests(Base):
    def setUp(self):
        super().setUp()
        self.originals = {name: original_fingerprint(path) for name, path in SERIALIZERS.items()}
        stack = contextlib.ExitStack()
        stack.enter_context(fake_modules(self.torch))
        self.addCleanup(stack.close)

    def trees(self):
        a, b, c = self.leaf(pattern(16, 1)), self.leaf(pattern(17, 2)), self.leaf(pattern(1, 3))
        return {'dict': {'z': a, 'a': (b, [c, c]), 5: a, 'n': (1, 'x', None, 2.5, True)},
                'dup3': (a, a, a), 'list': [a, a], 'tuple': (a, a), 'shape44': (a.reshape(4, 4),),
                'shape28': (a.reshape(2, 8),), 'flat': (a,), 'empty': [], 'emptyleaf': {'k': [self.leaf(b'')]},
                'scalar': (c.reshape(()),), 'tensor_key': {c: a}, 'bytes': {'b': b'x', 'i': 1}}

    def test_serializer_copies_are_identical_original_code(self):
        self.assertEqual(self.originals['probe'][1], self.originals['mlp'][1])
        trainer = ROOT / 'scripts/train_siglip2_substrate_adaptation.py'
        self.assertEqual(self.originals['probe'][1], original_fingerprint(trainer)[1])

    def test_native_matches_original_typed_tree(self):
        for name, (function, _) in self.originals.items():
            for label, tree in self.trees().items():
                with self.subTest('%s %s' % (name, label)):
                    before = len(self.native.calls)
                    self.assertEqual(mod.native_fingerprint(self.torch, self.hasher, function, tree), function(tree))
                    self.assertEqual(len(self.native.calls) - before, 0 if label in ('empty', 'bytes') else 1)

    def test_distinct_structures_remain_distinct_and_no_dedup(self):
        function = self.originals['probe'][0]
        trees = self.trees()
        prints = {k: mod.native_fingerprint(self.torch, self.hasher, function, trees[k])
                  for k in ('list', 'tuple', 'shape44', 'shape28', 'flat', 'dup3')}
        self.assertEqual(len(set(prints.values())), 6)
        reads = self.native.reads[-1]
        self.assertEqual(len(reads), 3)
        self.assertEqual(len(set(reads)), 1)

    def test_visit_order_includes_dict_keys_and_sorted_repr(self):
        a, b = self.leaf(pattern(4, 1)), self.leaf(pattern(5, 2))
        found = mod.collect_occurrences(self.torch, {'b': b, 'a': [a, {'k': a}], 3: b})
        self.assertEqual([id(t) for t in found], [id(a), id(a), id(b), id(b)])
        key = self.leaf(pattern(2, 9))
        order = mod.collect_occurrences(self.torch, {key: a})
        self.assertEqual([id(t) for t in order], [id(key), id(a)])

    def test_wrong_digest_changes_the_result(self):
        function = self.originals['probe'][0]
        tree = (self.leaf(pattern(16, 1)),)
        good = mod.native_fingerprint(self.torch, self.hasher, function, tree)

        def corrupting(*args):
            status = self.native(*args)
            self.torch.device_arena.write(args[3], b'\x00' * 32)
            return status
        bad = mod.Sha256Native(self.torch, corrupting)
        self.assertNotEqual(mod.native_fingerprint(self.torch, bad, function, tree), good)

    def test_cursor_failures(self):
        key = (1, 0, 'torch.float32', (4,))
        fact = ('torch.float32', (4,), '0' * 64)
        cursor = mod.OccurrenceCursor([key, key], [fact, fact])
        self.assertEqual(cursor.get(key), fact)
        with self.assertRaisesRegex(ValueError, 'missing'):
            cursor.finish()
        cursor = mod.OccurrenceCursor([key], [fact])
        cursor.get(key)
        with self.assertRaisesRegex(ValueError, 'extra'):
            cursor.get(key)
        for bad in ((2, 0, 'torch.float32', (4,)), (1, 1, 'torch.float32', (4,)), (1, 0, 'torch.float64', (4,)),
                    (1, 0, 'torch.float32', (5,)), (1, 0, 'torch.float32', [4])):
            with self.subTest(bad), self.assertRaisesRegex(ValueError, 'differs'):
                mod.OccurrenceCursor([key], [fact]).get(bad)
        cursor = mod.OccurrenceCursor([key], [fact])
        cursor.get(key)
        cursor.finish()
        for call in (lambda: cursor.get(key), cursor.finish):
            with self.assertRaisesRegex(ValueError, 'closed|missing'):
                call()
        with self.assertRaises(ValueError):
            mod.OccurrenceCursor([key], [('torch.float32', [4], '0' * 64)])
        with self.assertRaises(ValueError):
            mod.OccurrenceCursor([key], [('torch.float32', (4,), 'A' * 64)])
        with self.assertRaises(ValueError):
            mod.OccurrenceCursor([key, key], [fact])

    def test_serializer_seam_rejects_extra_missing_reordered_and_mutated(self):
        real = self.originals['probe'][0]
        a, b = self.leaf(pattern(16, 1)), self.leaf(pattern(17, 2))
        tree = (a, b)

        def run(function, value=tree):
            return mod.native_fingerprint(self.torch, self.hasher, function, value)
        self.assertEqual(run(real), real(tree))

        def extra(value, frozen=None):
            real(value, frozen)
            frozen.get((0, 0, 'x', ()))
        def missing(value, frozen=None):
            return real((value[0],), frozen)
        def reordered(value, frozen=None):
            return real((value[1], value[0]), frozen)
        for label, function in (('extra', extra), ('missing', missing), ('reordered', reordered)):
            with self.subTest(label), self.assertRaises(ValueError):
                run(function)
        # in-place version bump during hashing is caught by the exact key comparison
        self.native.hook = lambda: a.counter.__setitem__(0, a.counter[0] + 1)
        with self.assertRaisesRegex(ValueError, 'differs'):
            run(real)
        self.native.hook = None
        # a failed serializer leaves a closed, non-retained cursor
        captured = []

        def spy(value, frozen=None):
            captured.append(frozen)
            raise RuntimeError('serializer failed')
        with self.assertRaises(RuntimeError):
            run(spy)
        self.assertFalse(captured[0].open)
        self.assertEqual(captured[0].keys, ())


# ------------------------------------------------------------------ authority fixtures
def interpreter_fact():
    global INTERPRETER
    if INTERPRETER is None:
        INTERPRETER = fact(Path(sys.executable).resolve())
    return INTERPRETER


class Fixture:
    def __init__(self, directory):
        self.dir = Path(directory).resolve()
        self.handles, self.static = [], {}
        self.static.update(source=fact(CUDA), contract=fact(ROOT / 'docs/gpu_sha256_source_contract_2026-10-09.md'),
                           interpreter=interpreter_fact(), driver=fact(SCRIPT), test=fact(HERE / 'test_cuda_sha256_native.py'),
                           probe=fact(SERIALIZERS['probe']), mlp=fact(SERIALIZERS['mlp']))
        for name in ('nvcc', 'gxx', 'compile', 'evidence', 'log', 'proof_evidence'):
            self.static[name] = self.file(name, ('fixture ' + name).encode())
        self.static['library'] = self.file('libsha256_occurrences.so', b'fixture library ' * 64)
        self.static['runtime'] = self.file('libcudart.so.13', b'fixture runtime ' * 64)
        self.locks = []
        for i in range(2):
            path = self.dir / ('lock%d' % i)
            handle = open(path, 'wb')
            self.handles.append(handle)
            self.locks.append({'path': str(path), 'fd': handle.fileno()})

    def close(self):
        for handle in self.handles:
            handle.close()

    def file(self, name, raw):
        path = self.dir / name
        path.write_bytes(raw)
        return {'path': str(path), 'sha256': sha(raw)}

    def json_file(self, name, record):
        return self.file(name, json.dumps(record, sort_keys=True).encode())

    def render(self, edit=None):
        s = self.static

        def emit(stage, name, record):
            if edit:
                edit(stage, record)
            return self.json_file(name, record)
        build = emit('build', 'build-authority.json', {
            'schema': mod.BUILD, 'source': s['source'], 'contract': s['contract'],
            'toolchain': [{'role': 'nvcc', 'file': s['nvcc']}, {'role': 'host_compiler', 'file': s['gxx']}],
            'compile_script': s['compile'], 'target': 'sm_121',
            'flags': ['-O3', '--shared', '-Xcompiler', '-fPIC', '-gencode=arch=compute_121,code=sm_121'],
            'environment': {'LC_ALL': 'C'}, 'output_dir': str(self.dir / 'build-out'), 'evidence': [s['evidence']]})
        receipt = emit('receipt', 'build-receipt.json', {
            'schema': mod.RECEIPT, 'build_authority': build, 'exit_status': 0, 'library': s['library'],
            'sass_targets': ['sm_121'], 'ptx_images': 0, 'logs': [s['log']]})
        proof = emit('provenance', 'provenance.json', {
            'schema': mod.PROVENANCE, 'file': s['runtime'], 'origin': 'root frozen fixture', 'evidence': [s['proof_evidence']]})
        files = {s[k]['path']: {'sha256': s[k]['sha256'], 'size': Path(s[k]['path']).stat().st_size} for k in ('library', 'runtime')}
        inventory = emit('inventory', 'inventory.json', {'schema': mod.INVENTORY, 'files': files})
        return emit('native', 'native-authority.json', {
            'schema': mod.NATIVE, 'sources': {'driver': s['driver'], 'test': s['test'], 'probe_serializer': s['probe'],
                                              'mlp_serializer': s['mlp']},
            'build_authority': build, 'build_receipt': receipt, 'library': s['library'],
            'runtime_files': [{'file': s['runtime'], 'provenance': proof}], 'mapping_inventory': inventory,
            'interpreter': s['interpreter'], 'device': {'index': 0, 'name': 'Fake GPU', 'capability': [12, 1]},
            'resource_policy': dict(mod.POLICY), 'locks': self.locks})


def at(stage, change):
    def edit(current, record):
        if current == stage:
            change(record)
    return edit


class AuthorityBase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='sha-native-')
        self.addCleanup(directory.cleanup)
        self.fixture = Fixture(directory.name)
        self.addCleanup(self.fixture.close)


class AuthorityTests(AuthorityBase):
    def test_positive_read(self):
        admitted = mod.read_native_authority(self.fixture.render())
        self.assertEqual(admitted.build['target'], 'sm_121')
        self.assertEqual(admitted.record['resource_policy'], mod.POLICY)
        self.assertEqual(set(admitted.inventory), {self.fixture.static['library']['path'], self.fixture.static['runtime']['path']})

    def rejects(self, edit, pattern=None):
        with self.assertRaisesRegex(ValueError, pattern) if pattern else self.assertRaises(ValueError):
            mod.read_native_authority(self.fixture.render(edit))

    def test_native_record_negatives(self):
        s = self.fixture.static
        cases = {
            'extra_key': at('native', lambda r: r.update(extra=1)),
            'missing_key': at('native', lambda r: r.pop('locks')),
            'schema': at('native', lambda r: r.update(schema='x')),
            'sources_missing': at('native', lambda r: r['sources'].pop('mlp_serializer')),
            'sources_extra': at('native', lambda r: r['sources'].update(more=s['test'])),
            'stale_receipt_link': at('receipt', lambda r: r.update(build_authority=s['evidence'])),
            'library_differs': at('native', lambda r: r.update(library=s['runtime'])),
            'runtime_not_listed': at('native', lambda r: r.update(runtime_files=[])),
            'runtime_row_keys': at('native', lambda r: r['runtime_files'][0].update(x=1)),
            'interpreter': at('native', lambda r: r.update(interpreter=s['nvcc'])),
            'device_capability': at('native', lambda r: r['device'].update(capability=[9, 0])),
            'device_extra': at('native', lambda r: r['device'].update(x=1)),
            'device_bool': at('native', lambda r: r['device'].update(index=True)),
            'lib_wrong_sha': at('native', lambda r: r.update(library={'path': s['library']['path'], 'sha256': '0' * 64})),
            'relative_path': at('native', lambda r: r.update(library={'path': 'libsha256.so', 'sha256': s['library']['sha256']})),
            'dup_files': at('native', lambda r: r['sources'].update(test=s['driver'])),
        }
        for name, edit in cases.items():
            with self.subTest(name):
                self.rejects(edit)

    def test_policy_must_be_exactly_the_original_limits(self):
        for key, bad in (('body_seconds', 299), ('body_seconds', 301), ('whole_process_seconds', 1499),
                         ('whole_process_seconds', 900), ('exit_reserve_seconds', 120), ('exit_reserve_seconds', 301),
                         ('host_bytes', 8 * 1024**3 - 1), ('swap_bytes', 1), ('swap_bytes', False),
                         ('cuda_allocated_bytes_exclusive', 10**10 + 1)):
            with self.subTest('%s=%r' % (key, bad)):
                self.rejects(at('native', lambda r, key=key, bad=bad: r['resource_policy'].update({key: bad})), 'policy')
        self.rejects(at('native', lambda r: r['resource_policy'].update(extra=1)), 'policy')

    def test_build_authority_is_prospective_and_sass_only(self):
        s = self.fixture.static
        flags = ['-gencode=arch=compute_121,code=sm_121']
        cases = {
            'product_key': at('build', lambda r: r.update(library=s['library'])),
            'binary_hash_key': at('build', lambda r: r.update(output_sha256='0' * 64)),
            'arch_shorthand': at('build', lambda r: r['flags'].append('-arch=sm_121')),
            'ptx_in_code': at('build', lambda r: r['flags'].__setitem__(-1, '-gencode=arch=compute_121,code=compute_121')),
            'sass_and_ptx': at('build', lambda r: r['flags'].__setitem__(-1, '-gencode=arch=compute_121,code=[sm_121,compute_121]')),
            'two_gencode': at('build', lambda r: r['flags'].append('-gencode=arch=compute_90,code=sm_90')),
            'no_gencode': at('build', lambda r: r.update(flags=['-O3'])),
            'bare_gencode': at('build', lambda r: r.update(flags=['-gencode', 'arch=compute_121,code=sm_121'])),
            'wrong_target_flag': at('build', lambda r: r['flags'].__setitem__(-1, '-gencode=arch=compute_90,code=sm_90')),
            'ptx_flag': at('build', lambda r: r['flags'].append('--ptx')),
            'bad_target': at('build', lambda r: r.update(target='sm_x')),
            'missing_nvcc': at('build', lambda r: r['toolchain'].pop(0)),
            'dup_role': at('build', lambda r: r['toolchain'].append(dict(r['toolchain'][0]))),
            'wrong_source_name': at('build', lambda r: r.update(source=s['evidence'])),
            'env_lowercase': at('build', lambda r: r['environment'].update(path='x')),
            'output_relative': at('build', lambda r: r.update(output_dir='out')),
            'output_dotdot': at('build', lambda r: r.update(output_dir='/tmp/../out')),
            'no_evidence': at('build', lambda r: r.update(evidence=[])),
        }
        self.assertEqual(flags[0], '-gencode=arch=compute_121,code=sm_121')
        for name, edit in cases.items():
            with self.subTest(name):
                self.rejects(edit)

    def test_receipt_provenance_inventory_negatives(self):
        s = self.fixture.static
        cases = {
            'exit_status': at('receipt', lambda r: r.update(exit_status=1)),
            'exit_bool': at('receipt', lambda r: r.update(exit_status=False)),
            'ptx_images': at('receipt', lambda r: r.update(ptx_images=1)),
            'targets': at('receipt', lambda r: r.update(sass_targets=['sm_90'])),
            'receipt_extra': at('receipt', lambda r: r.update(x=1)),
            'no_logs': at('receipt', lambda r: r.update(logs=[])),
            'proof_wrong_file': at('provenance', lambda r: r.update(file=s['library'])),
            'proof_no_origin': at('provenance', lambda r: r.update(origin=' ')),
            'proof_self_evidence': at('provenance', lambda r: r.update(evidence=[s['runtime']])),
            'proof_no_evidence': at('provenance', lambda r: r.update(evidence=[])),
            'proof_bad_evidence': at('provenance', lambda r: r.update(evidence=['x'])),
            'inventory_missing_lib': at('inventory', lambda r: r['files'].pop(s['library']['path'])),
            'inventory_hash': at('inventory', lambda r: r['files'][s['runtime']['path']].update(sha256='1' * 64)),
            'inventory_directory': at('inventory', lambda r: r['files'].update({'/usr/lib': {'sha256': '1' * 64, 'size': 1}})),
            'inventory_entry': at('inventory', lambda r: r['files'][s['library']['path']].update(x=1)),
            'inventory_relative': at('inventory', lambda r: r['files'].update({'lib.so': {'sha256': '1' * 64, 'size': 1}})),
        }
        for name, edit in cases.items():
            with self.subTest(name):
                self.rejects(edit)

    def test_file_bounds_and_hashing_are_separate(self):
        raw = b'x' * (mod.JSON_LIMIT + 1)
        big_json = self.fixture.file('big.json', raw)
        with self.assertRaisesRegex(ValueError, 'limit'):
            mod.read_bytes(big_json, mod.JSON_LIMIT)
        self.assertEqual(mod.hash_file(big_json).st_size, len(raw))
        huge = self.fixture.dir / 'huge.bin'  # sparse: binaries above 64MiB must hash without the JSON cap
        with open(huge, 'wb') as stream:
            stream.truncate(64 * 1024**2 + 4096)
        digest = hashlib.sha256()
        for _ in range(64 * 1024):
            digest.update(bytes(1024))
        digest.update(bytes(4096))
        self.assertEqual(mod.hash_file({'path': str(huge), 'sha256': digest.hexdigest()}).st_size, 64 * 1024**2 + 4096)
        with mock.patch.object(mod, 'JSON_LIMIT', 8):
            with self.assertRaises(ValueError):
                mod.read_json(self.fixture.json_file('a.json', {'schema': 'x'}), 'x', {'schema'})
        self.assertEqual(mod.hash_file(self.fixture.static['library']).st_size, 1024)
        for name, bad in (('missing', {'path': str(self.fixture.dir / 'none'), 'sha256': '0' * 64}),
                          ('symlink', None), ('directory', {'path': str(self.fixture.dir), 'sha256': '0' * 64}),
                          ('extra', {**self.fixture.static['log'], 'x': 1}), ('upper', {**self.fixture.static['log'], 'sha256': 'A' * 64})):
            if name == 'symlink':
                link = self.fixture.dir / 'link'
                link.symlink_to(self.fixture.static['log']['path'])
                bad = {'path': str(link), 'sha256': self.fixture.static['log']['sha256']}
            with self.subTest(name), self.assertRaises((ValueError, OSError)):
                mod.hash_file(bad)
        changed = self.fixture.file('mutable', b'before')
        Path(changed['path']).write_bytes(b'after!')
        with self.assertRaisesRegex(ValueError, 'differs'):
            mod.hash_file(changed)

    def test_duplicate_json_keys_and_nonfinite_rejected(self):
        for name, raw in (('dup', b'{"schema":"x","schema":"x"}'), ('nan', b'{"schema":"x","v":NaN}')):
            with self.subTest(name), self.assertRaises(ValueError):
                mod.read_json(self.fixture.file(name + '.json', raw), 'x', {'schema', 'v'})


# ------------------------------------------------------------------ library + mapping
class LibraryTests(AuthorityBase):
    def setUp(self):
        super().setUp()
        self.admitted = mod.read_native_authority(self.fixture.render())
        self.library = self.fixture.static['library']['path']
        self.runtime = self.fixture.static['runtime']['path']
        self.maps = self.fixture.dir / 'maps'

    def line(self, path, ino=None, dev=None, suffix=''):
        info = os.stat(path)
        return '7f0000000000-7f0000100000 r-xp 00000000 %02x:%02x %d %s%s' % (
            os.major(dev if dev is not None else info.st_dev), os.minor(dev if dev is not None else info.st_dev),
            info.st_ino if ino is None else ino, path, suffix)

    def cdll(self, *paths, record=None):
        def load_it(path):
            if record is not None:
                record.append((path, os.readlink(path)))
            self.maps.write_text('\n'.join(self.line(p) for p in paths) + '\n')
            return types.SimpleNamespace(sfora_sha256_occurrences=lambda *a: 0)
        return load_it

    def test_loads_the_authenticated_inode_through_the_descriptor(self):
        self.maps.write_text('')
        record = []
        fd, lib, after = mod.open_library(self.admitted, self.cdll(self.library, record=record), str(self.maps))
        self.addCleanup(os.close, fd)
        self.assertRegex(record[0][0], r'^/proc/self/fd/\d+$')
        self.assertEqual(record[0][1], self.library)
        self.assertEqual(set(after), {self.library})
        self.assertEqual(mod.mapped_files(self.admitted.inventory, str(self.maps)), after)

    def test_library_and_mapping_negatives(self):
        self.maps.write_text('')
        Path(self.library).write_bytes(b'tampered after the authority froze it')
        with self.assertRaisesRegex(ValueError, 'differs'):
            mod.open_library(self.admitted, self.cdll(self.library), str(self.maps))
        Path(self.library).write_bytes(b'fixture library ' * 64)
        outside = self.fixture.file('libother.so', b'other')['path']
        cases = {
            'outside_inventory': [self.library, outside],
            'not_loaded': [],
            'only_runtime': [self.runtime],
        }
        for name, paths in cases.items():
            self.maps.write_text('')
            with self.subTest(name), self.assertRaises(ValueError):
                mod.open_library(self.admitted, self.cdll(*paths), str(self.maps))
        for name, text in (('deleted', self.line(self.library, suffix=' (deleted)')),
                           ('inode', self.line(self.library, ino=1)),
                           ('device', self.line(self.library, dev=os.makedev(1, 1)))):
            def load_bad(path, text=text):
                self.maps.write_text(text + '\n')
                return object()
            self.maps.write_text('')
            with self.subTest(name), self.assertRaises(ValueError):
                mod.open_library(self.admitted, load_bad, str(self.maps))
        link = self.fixture.dir / 'libalias.so'
        link.symlink_to(self.library)
        with self.assertRaises(ValueError):
            mod.mapped_files({str(link): {'sha256': '0' * 64, 'size': 1}}, self.write_maps(self.line(str(link))))
        with self.assertRaises(ValueError):
            mod.mapped_files({self.library: {'sha256': '0' * 64, 'size': 1}}, self.write_maps(self.line(self.library)))
        self.assertEqual(mod.mapped_files({}, self.write_maps('7f00-7f01 rw-p 00000000 00:00 0 [heap]\n')), {})

    def write_maps(self, text):
        self.maps.write_text(text + '\n')
        return str(self.maps)

    def test_undeclared_inventoried_mapping_is_rejected(self):
        extra = self.fixture.file('libextra.so', b'inventoried but not declared')

        def add(stage, record):
            if stage == 'inventory':
                record['files'][extra['path']] = {'sha256': extra['sha256'], 'size': Path(extra['path']).stat().st_size}
        admitted = mod.read_native_authority(self.fixture.render(add))
        self.assertIn(extra['path'], admitted.inventory)
        self.maps.write_text('')
        with self.assertRaisesRegex(ValueError, 'beyond the declared'):
            mod.open_library(admitted, self.cdll(self.library, extra['path']), str(self.maps))
        self.maps.write_text('')
        fd, _, after = mod.open_library(admitted, self.cdll(self.library, self.runtime), str(self.maps))
        os.close(fd)
        self.assertEqual(set(after), {self.library, self.runtime})

    def test_descriptor_closed_on_failure(self):
        before = len(os.listdir('/proc/self/fd'))
        self.maps.write_text('')
        with self.assertRaises(ValueError):
            mod.open_library(self.admitted, self.cdll(), str(self.maps))
        self.assertEqual(len(os.listdir('/proc/self/fd')), before)


# ------------------------------------------------------------------ CLI + admission
class CliTests(AuthorityBase):
    def setUp(self):
        super().setUp()
        self.authority = self.fixture.render()
        self.output = str(self.fixture.dir / 'out')
        self.recorder = []

    def argv(self, stage='smoke', unit=None, authority=None, output=None):
        return mod.cli(stage, authority or self.authority, unit, output or self.output)

    @property
    def body(self):
        recorder = self.recorder

        def body(context, output):
            context.source.check()
            context.locks.check()
            recorder.append((context, output))
            return 'ran'
        return body

    def run_main(self, argv, bodies=None):
        with mock.patch.dict(mod.STAGE_BODIES, bodies or {'smoke': self.body}, clear=True):
            return mod.main(argv)

    def test_smoke_admits_then_runs_the_body_once(self):
        self.assertEqual(self.run_main(self.argv()), 'ran')
        (context, output), = self.recorder
        self.assertEqual(output, self.output)
        self.assertEqual(context.fact, self.authority)
        self.assertEqual(context.authority.build['target'], 'sm_121')
        self.assertFalse(os.path.lexists(self.output))
        self.assertFalse(NATIVE & {name.split('.')[0] for name in sys.modules})

    def test_argv_contract(self):
        good = self.argv()
        swapped = [good[0], *good[3:5], *good[1:3], *good[5:]]
        cases = {'swapped_order': swapped, 'extra': good + ['--unit', 'x'], 'missing_output': good[:-2],
                 'bad_stage': ['bogus', *good[1:]], 'abbrev': [a.replace('--authority-sha256', '--authority-s') for a in good],
                 'wrong_sha': [*good[:4], '0' * 64, *good[5:]], 'upper_sha': [*good[:4], self.authority['sha256'].upper(), *good[5:]],
                 'relative_authority': [good[0], '--authority', 'native-authority.json', *good[3:]],
                 'relative_output': [*good[:-1], 'out'], 'existing_output': [*good[:-1], str(self.fixture.dir)],
                 'missing_parent': [*good[:-1], str(self.fixture.dir / 'x' / 'out')],
                 'smoke_with_unit': mod.cli('smoke', self.authority, self.authority, self.output),
                 'full_without_unit': mod.cli('full', self.authority, None, self.output),
                 'half_unit': [*good[:-2], '--unit', self.authority['path'], '--output', self.output]}
        quiet = contextlib.redirect_stderr(io.StringIO())
        for name, argv in cases.items():
            with self.subTest(name), quiet, self.assertRaises((ValueError, SystemExit)):
                self.run_main(argv, {'smoke': self.body, 'full': self.body, 'timing': self.body})
        self.assertEqual(self.recorder, [])
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
            mod.main(['--help'])

    def test_mutated_authority_never_reaches_the_body(self):
        for name, edit in (('policy', at('native', lambda r: r['resource_policy'].update(whole_process_seconds=900))),
                           ('sources', at('native', lambda r: r['sources'].update(driver=self.fixture.static['test']))),
                           ('locks_fd', at('native', lambda r: r['locks'][0].update(fd=99))),
                           ('locks_one', at('native', lambda r: r['locks'].pop()))):
            with self.subTest(name), self.assertRaises(ValueError):
                self.run_main(self.argv(authority=self.fixture.render(edit), output=str(self.fixture.dir / ('o-' + name))))
        self.assertEqual(self.recorder, [])

    def test_driver_must_be_the_running_module(self):
        edit = at('native', lambda r: r['sources'].update(driver=fact(HERE / 'test_cuda_sha256_source.py')))
        with self.assertRaises(ValueError):
            self.run_main(self.argv(authority=self.fixture.render(edit)))
        other = load('qualify_cuda_sha256_native_other')
        other.STAGE_BODIES.clear()
        other.STAGE_BODIES['smoke'] = self.body
        other.__file__ = str(HERE / 'elsewhere.py')
        with self.assertRaises(ValueError):
            other.main(self.argv())

    def test_full_and_timing_are_unreleased_scaffolding_and_ignore_user_go_units(self):
        unit = self.fixture.json_file('go.json', {'schema': 'cuda-sha256-native-unit-v1', 'stage': 'smoke',
                                                  'decision': 'GO', 'authority': self.authority})
        for stage in ('full', 'timing'):
            argv = mod.cli(stage, self.authority, unit, self.output)
            with self.subTest(stage + ' no body'), self.assertRaisesRegex(ValueError, 'unreleased scaffolding'):
                self.run_main(argv)
            self.assertIsNone(mod.UNIT_READER)
            with self.subTest(stage + ' body present'), self.assertRaisesRegex(ValueError, 'genuine root-owned'):
                self.run_main(argv, {'smoke': self.body, stage: self.body})
        self.assertEqual(self.recorder, [])
        self.assertEqual(set(mod.STAGE_BODIES), {'smoke'})

    def test_script_help_and_flags(self):
        for flags in ((), ('-O',), ('-OO',)):
            with self.subTest(flags):
                done = subprocess.run([sys.executable, '-B', *flags, str(SCRIPT), '--help'], capture_output=True, text=True, timeout=10)
                self.assertEqual(done.returncode, 0, done.stderr)
                self.assertIn('--authority-sha256', done.stdout)
                done = subprocess.run([sys.executable, '-B', *flags, '-c', (
                    'import sys; sys.path.insert(0, %r); import qualify_cuda_sha256_native as m\n'
                    'c = m.OccurrenceCursor([(1, 0, "torch.float32", (1,))], [("torch.float32", (1,), "0" * 64)])\n'
                    'try:\n    c.get((2, 0, "torch.float32", (1,)))\nexcept ValueError:\n    print("rejected")\n'
                    'print([n for n in sys.modules if n.split(".")[0] in %r])' % (str(HERE), sorted(NATIVE)))],
                    capture_output=True, text=True, timeout=10)
                self.assertEqual(done.stdout.split('\n')[:2], ['rejected', '[]'], done.stderr)
        done = subprocess.run([sys.executable, '-B', str(SCRIPT), 'smoke', '--authority', '/x', '--authority-sha256', '0' * 64,
                               '--output', '/tmp/never-created-sha-native'], capture_output=True, text=True, timeout=10)
        self.assertNotEqual(done.returncode, 0)
        self.assertFalse(os.path.lexists('/tmp/never-created-sha-native'))

    def test_compile_and_no_assert_dependence(self):
        with tempfile.TemporaryDirectory() as target:
            py_compile.compile(str(SCRIPT), cfile=str(Path(target) / 'x.pyc'), doraise=True)
            py_compile.compile(__file__, cfile=str(Path(target) / 'y.pyc'), doraise=True)
        tree = ast.parse(SCRIPT.read_text())
        self.assertFalse([n for n in ast.walk(tree) if isinstance(n, ast.Assert)])
        top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = {a.name.split('.')[0] for n in top for a in n.names} | {n.module.split('.')[0] for n in top if isinstance(n, ast.ImportFrom)}
        self.assertFalse(names & NATIVE | names & {'ctypes'})


# ------------------------------------------------------------------ frozen smoke body
def maps_line(path):
    info = os.stat(path)
    return '7f0000000000-7f0000100000 r-xp 00000000 %02x:%02x %d %s\n' % (
        os.major(info.st_dev), os.minor(info.st_dev), info.st_ino, path)


class SmokeTests(AuthorityBase):
    def setUp(self):
        super().setUp()
        self.authority = self.fixture.render()
        self.admitted = mod.read_native_authority(self.authority)
        self.maps = self.fixture.dir / 'maps'
        self.checks = []
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.stack = stack
        self.count = 0
        self.reset()

    def reset(self):
        self.stack.close()
        self.torch = Torch()
        self.native = Native(self.torch)
        self.stack.enter_context(fake_modules(self.torch))
        self.maps.write_text('')
        self.count += 1
        self.output = self.fixture.dir / ('smoke-out-%d' % self.count)

    def cdll(self, path):
        self.paths = getattr(self, 'paths', []) + [(path, os.readlink(path))]
        self.maps.write_text(maps_line(self.fixture.static['library']['path']))
        return types.SimpleNamespace(sfora_sha256_occurrences=self.native)

    def run_smoke(self):
        context = types.SimpleNamespace(authority=self.admitted, fact=self.authority, locks=None,
                                        source=types.SimpleNamespace(check=lambda: self.checks.append(1)))
        return mod.smoke(context, str(self.output), torch=self.torch, cdll=self.cdll, maps=str(self.maps))

    def test_all_checks_pass_and_receipt_is_a_discarded_diagnostic(self):
        receipt = self.run_smoke()
        self.assertEqual(sorted(receipt['checks']), ['injected_failures', 'mutation', 'nondefault_stream_pending',
                                                     'parity', 'raw_abi', 'rejections', 'typed_trees'])
        self.assertEqual(receipt['checks']['raw_abi'], {'count0_nulls': 0, 'count2e32': 1, 'all_null': 1,
                                                         'ptrs_null': 1, 'lens_null': 1, 'out_null': 1})
        self.assertEqual(receipt['checks']['parity']['occurrences'], 129)
        self.assertEqual(receipt['checks']['injected_failures'], {
            'status': 'ValueError', 'launched': 'InjectedFault', 'completed': 'InjectedFault', 'readback': 'InjectedFault'})
        self.assertEqual(receipt['status'], 'SMOKE_DIAGNOSTIC_UNREVIEWED')
        self.assertTrue(all(receipt[flag] is False for flag in mod.FLAGS))
        self.assertEqual(receipt['resource_policy'], mod.POLICY)
        written = json.loads((self.output / 'smoke-receipt.json').read_text())
        self.assertEqual(written['schema'], mod.SMOKE_RECEIPT)
        self.assertEqual(written['authority'], self.authority)
        self.assertEqual(len(self.checks), 2)
        self.assertRegex(self.paths[0][0], r'^/proc/self/fd/\d+$')
        self.assertEqual(self.paths[0][1], self.fixture.static['library']['path'])
        log = self.torch.log
        self.assertGreater(log.index(('global_sync',)), max(i for i, e in enumerate(log) if e[0] == 'sleep'))
        self.assertGreaterEqual(len([e for e in log if e[0] == 'launch' and e[1] >= 0x1000]), 2)

    def arm(self, name):
        if name == 'wrong_digest':
            class Bad(Native):
                def __call__(self, *args):
                    status = Native.__call__(self, *args)
                    if args[2] == 129:
                        self.torch.device_arena.write(args[3] + 32 * 128, b'\x01' * 32)
                    return status
            self.native = Bad(self.torch)
        elif name == 'accepts_nulls':
            class Lax(Native):
                def __call__(self, ptrs, lens, count, out, stream):
                    return 0 if not out else Native.__call__(self, ptrs, lens, count, out, stream)
            self.native = Lax(self.torch)
        elif name == 'not_pending':
            self.torch.cuda._sleep = lambda cycles: None
        elif name == 'poisoned_drain':
            syncs = []
            self.torch.hooks['sync'] = lambda stream: (syncs.append(1), len(syncs) > 25 and (_ for _ in ()).throw(RuntimeError('drain')))
        elif name == 'restore_bumps_version':
            real, seen = mod.flip, []

            def flip(torch, leaf, byte, mask):
                real(torch, leaf, byte, mask)
                seen.append(1)
                if len(seen) == 2:
                    leaf.counter[0] += 1
            self.stack.enter_context(mock.patch.object(mod, 'flip', flip))
        elif name == 'leaky_owner':
            leaked, real = [], mod.Sha256Native

            class Leaky(real):
                def digests(self, tensors):
                    leaked.append(list(tensors))
                    return real.digests(self, tensors)
            self.stack.enter_context(mock.patch.object(mod, 'Sha256Native', Leaky))
        elif name == 'restore_skipped':
            real, seen = mod.flip, []

            def flip(torch, leaf, byte, mask):
                seen.append(1)
                if len(seen) != 2:
                    real(torch, leaf, byte, mask)
            self.stack.enter_context(mock.patch.object(mod, 'flip', flip))
        elif name == 'wrong_device':
            self.torch.cuda.get_device_properties = lambda index: types.SimpleNamespace(name='Other', major=12, minor=1)

    def test_each_defect_fails_closed_and_publishes_nothing(self):
        for name in ('wrong_digest', 'accepts_nulls', 'not_pending', 'poisoned_drain', 'wrong_device', 'restore_bumps_version',
                     'restore_skipped', 'leaky_owner'):
            with self.subTest(name):
                self.reset()
                self.arm(name)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_smoke()
                self.assertFalse(os.path.lexists(self.output))

    def test_mapping_change_during_the_run_fails(self):
        extra = self.fixture.static['runtime']['path']
        calls = []

        class Mapping(Native):
            def __call__(inner, *args):
                calls.append(1)
                if len(calls) == 30:
                    self.maps.write_text(maps_line(self.fixture.static['library']['path']) + maps_line(extra))
                return Native.__call__(inner, *args)
        self.native = Mapping(self.torch)
        with self.assertRaisesRegex(ValueError, 'mapping'):
            self.run_smoke()
        self.assertFalse(os.path.lexists(self.output))

    def test_output_is_exclusive(self):
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_smoke()
        self.assertEqual(list(self.output.iterdir()), [])

    def test_serializer_copies_must_match(self):
        other = self.fixture.file('other.py', b'def fingerprint(value, frozen=None, consumed=None):\n    return "different"\n')
        self.admitted.record['sources']['mlp_serializer'] = other
        with self.assertRaisesRegex(ValueError, 'differ'):
            self.run_smoke()
        self.assertFalse(os.path.lexists(self.output))


if __name__ == '__main__':
    unittest.main(verbosity=1)
