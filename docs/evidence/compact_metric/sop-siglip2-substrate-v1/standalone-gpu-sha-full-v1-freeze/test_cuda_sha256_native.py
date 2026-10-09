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
import math
import mmap
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
EVIDENCE = ROOT / 'docs/evidence/compact_metric/sop-siglip2-substrate-v1'
FIXTURE_FILE = EVIDENCE / 'standalone-gpu-sha-full-fixture-v1/inventory.json'
EXTRACTOR = ROOT / 'scripts/extract_siglip2_vision_source.py'
METADATA = EVIDENCE / 'late-dense-v1/so400-native256-upstream-metadata-v1.json'


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
        self.base, self.mem, self.top = base, mmap.mmap(-1, size), 512  # lazily committed: large fake devices stay cheap

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
    def __init__(self, address, size, arena=None):
        self.address, self.size, self.arena = address, size, arena

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
        return math.prod(self.shape)

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
        return self.storage.arena

    def offsets(self):
        for index in itertools.product(*(range(d) for d in self.shape)):
            yield self.byte_offset + sum(i * s for i, s in zip(index, self.strides)) * self.dtype.size

    def logical(self):
        if self.is_contiguous():  # fast path: a contiguous tensor is one span (also what keeps 19.8 MB leaves cheap)
            return self.arena().read(self.storage.address + self.byte_offset, self.numel() * self.dtype.size)
        return b''.join(self.arena().read(self.storage.address + o, self.dtype.size) for o in self.offsets())

    def _write(self, raw):
        if self.is_contiguous():
            self.arena().write(self.storage.address + self.byte_offset, raw)
            return
        for i, offset in enumerate(self.offsets()):
            self.arena().write(self.storage.address + offset, raw[i * self.dtype.size:(i + 1) * self.dtype.size])

    def write_logical(self, raw):
        self.torch.run_or_defer(lambda: self._write(raw))
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
        self.torch.current.run()  # a copy on the current stream is ordered after that stream's queued work
        return self.torch.make(Device('cpu'), self.shape, self.dtype, self.logical())

    def to(self, device, non_blocking=False):
        self.torch.log.append(('copy', device.type, non_blocking, self.pinned))
        if device == self.device:
            return self
        if self.is_cuda:
            self.torch.current.run()
        return self.torch.make(device, self.shape, self.dtype, self.logical())

    def normal_(self, mean=0.0, std=1.0, generator=None):
        """Deterministic stand-in content (seed and size only); the real Generator.normal_ is an UNVERIFIED assumption."""
        nbytes = self.numel() * self.dtype.size
        block = hashlib.shake_256(b'%d:%d' % (generator.seed, nbytes)).digest(min(nbytes, 4096))
        self.write_logical((block * (nbytes // max(len(block), 1) + 1))[:nbytes])
        return self

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
        count = self.numel()
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
        size = self.dtype.size  # one element of a contiguous 1-D tensor; never a whole-tensor tolist
        raw = self.arena().read(self.storage.address + self.byte_offset + key * size, size)
        return Scalar(self, key, struct.unpack('<' + self.dtype.code, raw)[0])

    def __setitem__(self, key, value):
        size, value = self.dtype.size, value.value if isinstance(value, Scalar) else value
        raw = struct.pack('<' + self.dtype.code, value)
        address = self.storage.address + self.byte_offset + key * size
        self.torch.run_or_defer(lambda: self.arena().write(address, raw))
        self.counter[0] += 1


class Stream:
    def __init__(self, torch, device, handle, token='default'):
        self.torch, self.device, self.cuda_stream, self.token = torch, device, handle, token
        self.queue, self.issued, self.done = [], 0, 0

    def push(self, op):
        self.queue.append(op)
        self.issued += 1

    def run(self, upto=None):
        """Execute this stream's queued work in order; a ('wait', event) item first runs the producer up to the event."""
        while self.queue and (upto is None or self.done < upto):
            op = self.queue.pop(0)
            self.done += 1
            if isinstance(op, tuple):
                op[1].stream.run(op[1].mark)
            else:
                op()

    def wait_event(self, event):
        self.push(('wait', event))

    def synchronize(self):
        self.torch.hook('sync', self)
        self.torch.log.append(('sync', self.cuda_stream))
        self.run()
        self.torch.pending.discard(self.cuda_stream)

    def __eq__(self, other):
        return isinstance(other, Stream) and (self.device, self.cuda_stream, self.token) == \
            (other.device, other.cuda_stream, other.token)

    def __hash__(self):
        return hash((self.cuda_stream, self.token))


class Event:
    def __init__(self, torch):
        self.torch, self.stream, self.mark = torch, None, 0

    def record(self, stream):
        self.stream, self.mark = stream, stream.issued

    def query(self):
        return self.stream.cuda_stream not in self.torch.pending


class Cuda:
    def __init__(self, torch):
        self.torch = torch

    is_available = staticmethod(lambda: True)
    current_device = staticmethod(lambda: 0)

    def max_memory_allocated(self):
        return self.torch.peak

    def get_device_properties(self, index):
        return types.SimpleNamespace(name='Fake GPU', major=12, minor=1)

    def current_stream(self, device=None):
        return self.torch.current

    def default_stream(self, device=None):
        return self.torch.default

    def Stream(self, device=None):
        self.torch.streams += 1
        stream = Stream(self.torch, Device('cuda', 0), 0x1000 + self.torch.streams, 'side')
        self.torch.streams_by_handle[stream.cuda_stream] = stream
        return stream

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

    def __init__(self, device_bytes=1 << 24):
        self.log, self.hooks, self.pending, self.streams, self.made = [], {}, set(), 0, []
        self.device_arena = Arena(0x7000_0000_0000, device_bytes)
        self.version = types.SimpleNamespace(cuda='fake')
        self.default = Stream(self, Device('cuda', 0), 0)
        self.current = self.default
        self.streams_by_handle, self.defer, self.peak = {0: self.default}, False, 123
        self.cuda = Cuda(self)

    def run_or_defer(self, thunk):
        """A write on a stream that is still busy (torch.cuda._sleep) completes later, in that stream's order, when defer is on."""
        if self.defer and self.current.cuda_stream in self.pending:
            self.current.push(thunk)
        else:
            thunk()

    def Generator(self, device=None):
        generator = types.SimpleNamespace(seed=None)
        generator.manual_seed = lambda seed: (setattr(generator, 'seed', seed), generator)[1]
        return generator

    def hook(self, name, value):
        if name in self.hooks:
            self.hooks[name](value)

    def device(self, kind, index=None):
        return Device(kind, index)

    def make(self, device, shape, dtype, raw=None):
        shape = tuple(shape)
        size = math.prod(shape) * dtype.size
        arena = self.device_arena if device.type == 'cuda' else Arena(0x5000_0000_0000, size + 1024)  # host: private, freed with the tensor
        storage = Storage(arena.alloc(size) if size else 0, size, arena)
        result = Tensor(self, storage, 0, shape, contiguous_strides(shape), dtype, device)
        self.made.append(weakref.ref(result))
        if raw:
            result.write_logical(raw)
            result.counter[0] = 0
        return result

    def tensor(self, values, dtype=None, device=None):
        return self.make(device or Device('cpu'), (len(values),), dtype, struct.pack('<%d%s' % (len(values), dtype.code), *values))

    def empty(self, *shape, dtype=None, device=None):
        shape = tuple(shape[0]) if len(shape) == 1 and isinstance(shape[0], (tuple, list)) else shape
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
        launch = self.torch.streams_by_handle.get(stream)
        if launch is not None:
            launch.run()  # stream order: the launch sees exactly what that stream (and its waited events) completed
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
    def __init__(self, directory, own=False):
        self.dir = Path(directory).resolve()
        self.handles, self.static = [], {}
        self.static.update(source=fact(CUDA), contract=fact(ROOT / 'docs/gpu_sha256_source_contract_2026-10-09.md'),
                           interpreter=interpreter_fact(), driver=fact(SCRIPT), test=fact(HERE / 'test_cuda_sha256_native.py'),
                           probe=fact(SERIALIZERS['probe']), mlp=fact(SERIALIZERS['mlp']))
        if own:  # byte-identical private copies so a test can drift a closure FILE without touching the repo
            for key, origin in (('source', CUDA), ('contract', ROOT / 'docs/gpu_sha256_source_contract_2026-10-09.md'),
                                ('driver', SCRIPT), ('test', HERE / 'test_cuda_sha256_native.py'),
                                ('probe', SERIALIZERS['probe']), ('mlp', SERIALIZERS['mlp'])):
                self.static[key] = self.file(origin.name, origin.read_bytes())
        for key, origin in (('fixture', FIXTURE_FILE), ('extractor', EXTRACTOR), ('metadata', METADATA)):
            self.static[key] = self.file('full-fixture-' + origin.name if key == 'fixture' else origin.name, origin.read_bytes()) if own else fact(origin)
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

    def common(self, edit=None):
        """The closure shared by the smoke and the full authority: build authority/receipt, provenance, inventory."""
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
        return types.SimpleNamespace(build=build, receipt=receipt, proof=proof, inventory=inventory, emit=emit)

    def native(self, common, driver, test, schema, name, stage, extra=None):
        s = self.static
        record = {
            'schema': schema, 'sources': {'driver': driver, 'test': test, 'probe_serializer': s['probe'],
                                          'mlp_serializer': s['mlp']},
            'build_authority': common.build, 'build_receipt': common.receipt, 'library': s['library'],
            'runtime_files': [{'file': s['runtime'], 'provenance': common.proof}], 'mapping_inventory': common.inventory,
            'interpreter': s['interpreter'], 'device': {'index': 0, 'name': 'Fake GPU', 'capability': [12, 1]},
            'resource_policy': dict(mod.POLICY), 'locks': self.locks, **(extra or {})}
        return common.emit(stage, name, record)

    def render(self, edit=None):
        s = self.static
        return self.native(self.common(edit), s['driver'], s['test'], mod.NATIVE, 'native-authority.json', 'native')

    # ---- the full stage: a genuine-shaped prior smoke unit chain + the fixture/extractor/config FILEs
    UNIT, INVOCATION = 'sfora-standalone-gpu-sha-smoke-v2', 'e594e58e8ad1495a8150757ca754e3fc'

    def render_full(self, edit=None, chain=None, frozen=None):
        """edit(stage, record) edits the shared closure and both authorities ('prior-native', 'native'); chain(stage, value) edits the prior
        chain ('receipt', 'footer', 'unit', 'launch', 'bootstrap-authority', 'verification'); frozen(full) edits the 'full' extension."""
        s, common = self.static, self.common(edit)
        chain = chain or (lambda stage, value: None)
        out = self.dir / 'smoke-out'
        out.mkdir(exist_ok=True)
        prior_driver, prior_test = self.file('prior_driver.py', b'prior driver'), self.file('prior_test.py', b'prior test')
        prior = self.native(common, prior_driver, prior_test, mod.NATIVE, 'prior-authority.json', 'prior-native')
        argv = [prior_driver['path'], 'smoke', '--authority', prior['path'], '--authority-sha256', prior['sha256'], '--output', str(out)]
        receipt = {
            'schema': mod.SMOKE_RECEIPT, 'status': 'SMOKE_DIAGNOSTIC_UNREVIEWED', 'engineering_only': True, 'authority': prior,
            'library': s['library'], 'device': {'index': 0, 'name': 'Fake GPU', 'capability': [12, 1]},
            'checks': json.loads(json.dumps(mod.SMOKE_CHECKS)), 'native_calls': 24,
            'resources': {'wall_seconds': 3.9, 'body_seconds': 3.8, 'process_peak_rss_kib': 666528, 'peak_cuda_allocated_bytes': 15872},
            'resource_policy': dict(mod.POLICY), 'mappings': sorted([s['library']['path'], s['runtime']['path']]),
            **{flag: False for flag in mod.FLAGS}, 'normal_terminal_required': True,
            'invocation': {'argv': argv, 'python': s['interpreter']['path'], 'pid': 4242, 'optimize': 0,
                           'torch': '2.12.1+cu130', 'cuda': '13.0'}}
        chain('receipt', receipt)
        receipt_fact = self.json_file('smoke-out/smoke-receipt.json', receipt)
        outer = {'memory_events': {k: 0 for k in sorted(mod.EVENT_KEYS)}, 'memory_peak_bytes': 436899840,
                 'swap_current_bytes': 0, 'wall_seconds': 4.05}
        lines = ['Running as unit: %s.service; invocation ID: %s' % (self.UNIT, self.INVOCATION),
                 json.dumps({'bootstrap_exit_pass': True, 'normal_outer_terminal_required': True, 'resources': outer}, sort_keys=True),
                 'Finished with result: success', 'Main processes terminated with: code=exited/status=0', 'Service runtime: 4.393s',
                 'CPU time consumed: 4.352s', 'Memory peak: 1020.0K', 'Memory swap peak: 0B']
        chain('footer', lines)
        raw = ['\n'.join(lines) + '\n']
        chain('log', raw)
        log = self.file('original-smoke.log', raw[0].encode('utf-8', 'surrogateescape'))
        unit = {'both_locks_held': True, 'invocation_id': self.INVOCATION, 'log': log, 'native_peak_rss_kib': 666528,
                'receipt': receipt_fact, 'service_seconds': 4.393, 'unit': self.UNIT}
        chain('unit', unit)
        unit_fact = self.json_file('unit.json', unit)
        bootstrap = self.file('bootstrap.py', b'bootstrap')
        helper = self.file('helper.py', b'helper')
        auth = {'schema': mod.BOOTSTRAP, 'files': {'driver': prior_driver, 'test': prior_test, 'requests': helper,
                                                   'probe_serializer': s['probe'], 'mlp_serializer': s['mlp']},
                'native_authority': prior, 'output': str(out), 'interpreter': s['interpreter']}
        chain('bootstrap-authority', auth)
        auth_fact = self.json_file('bootstrap-authority.json', auth)
        launch = {'bootstrap': bootstrap, 'bootstrap_authority': auth_fact, 'engineering_only': True, 'native_authority': prior,
                  'output': str(out), 'product_go': False, 'quality_go': False, 'schema': mod.LAUNCH, 'speed_go': False, 'unit': self.UNIT,
                  'command': ['/usr/bin/systemd-run', '--user', '--unit=' + self.UNIT, '--wait', '--pipe', '--collect',
                              '--property=RuntimeMaxSec=1500', '--property=MemoryMax=8589934592', '--property=MemorySwapMax=0',
                              '--property=TasksMax=128', '--property=KillMode=control-group', '--property=OOMPolicy=stop',
                              '--setenv=CUDA_VISIBLE_DEVICES=0', '--setenv=OMP_NUM_THREADS=1', '--setenv=OPENBLAS_NUM_THREADS=1',
                              '--setenv=MKL_NUM_THREADS=1', '/venv/bin/python', '-I', '-B', bootstrap['path'], auth_fact['path'],
                              auth_fact['sha256'], bootstrap['sha256']]}
        chain('launch', launch)
        launch_fact = self.json_file('launch.json', launch)
        verification = {'all_smoke_checks': True, 'complete_current_mapped_file_hashes_exit_pass': True, 'engineering_only': True,
                        'exact_authority': True, 'exit_status': 0, 'next': 'full', 'original_helper_and_source_exit_pass': True,
                        'outer_resources': outer, 'product_go': False, 'schema': 'standalone-cuda-sha-smoke-parent-verification-v1',
                        'speed_go': False, 'terminal': unit}
        chain('verification', verification)
        verification_fact = self.json_file('verification.json', verification)
        self.unit = unit_fact
        full = {'prior': {'unit': unit_fact, 'launch': launch_fact, 'verification': verification_fact},
                'fixture': s['fixture'], 'extractor': s['extractor'], 'metadata': s['metadata']}
        if frozen:
            frozen(full)
        return self.native(common, s['driver'], s['test'], mod.NATIVE_FULL, 'native-authority-full.json', 'native', {'full': full})


def at(stage, change):
    def edit(current, record):
        if current == stage:
            change(record)
    return edit


class AuthorityBase(unittest.TestCase):
    own = False

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='sha-native-')
        self.addCleanup(directory.cleanup)
        self.fixture = Fixture(directory.name, self.own)
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

    def test_inventoried_undeclared_mapping_bytes_are_hashed_not_just_inode_and_size(self):
        raw = b'inventoried but not declared'
        extra = self.fixture.file('libextra.so', raw)

        def freeze(digest):
            def add(stage, record):
                if stage == 'inventory':
                    record['files'][extra['path']] = {'sha256': digest, 'size': len(raw)}
            return mod.read_native_authority(self.fixture.render(add))
        wrong = freeze(sha(b'y' * len(raw)))  # right size and inode, bytes the root never froze
        self.maps.write_text(self.line(extra['path']) + '\n')  # already mapped before the dlopen, as Torch libraries are
        before = len(os.listdir('/proc/self/fd'))
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            mod.open_library(wrong, self.cdll(self.library, extra['path']), str(self.maps))
        self.assertEqual(len(os.listdir('/proc/self/fd')), before)
        exact = freeze(extra['sha256'])
        self.maps.write_text(self.line(extra['path']) + '\n')
        fd, _, after = mod.open_library(exact, self.cdll(self.library, extra['path']), str(self.maps))
        os.close(fd)
        self.assertEqual(set(after), {self.library, extra['path']})

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

    def test_timing_stays_unreleased_scaffolding_and_full_never_accepts_a_user_go_unit(self):
        unit = self.fixture.json_file('go.json', {'schema': 'cuda-sha256-native-unit-v1', 'stage': 'smoke',
                                                  'decision': 'GO', 'authority': self.authority})
        self.assertIs(mod.UNIT_READER, mod.read_prior)  # the genuine reader is wired; a GO file is not a unit
        argv = mod.cli('timing', self.authority, unit, self.output)
        with self.subTest('timing no body'), self.assertRaisesRegex(ValueError, 'unreleased scaffolding'):
            self.run_main(argv)
        with self.subTest('timing body present'), self.assertRaises(ValueError):
            self.run_main(argv, {'smoke': self.body, 'timing': self.body})
        with self.subTest('full with a user GO and a smoke authority'), self.assertRaises(ValueError):
            self.run_main(mod.cli('full', self.authority, unit, self.output), {'smoke': self.body, 'full': self.body})
        self.assertEqual(self.recorder, [])
        self.assertEqual(set(mod.STAGE_BODIES), {'smoke', 'full'})

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


class Clock:
    def __init__(self):
        self.now = 5000.0

    def __call__(self):
        return self.now


class StageHarness(AuthorityBase):
    """Fake-torch lifecycle harness shared by the smoke and the full stage tests (no tests of its own)."""
    own = True  # private byte-identical copies of the closure FILEs, so a test can drift any of them
    stage = 'smoke'

    def setUp(self):
        super().setUp()
        self.authority = self.render()
        self.admitted = self.admit()
        self.maps = self.fixture.dir / 'maps'
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
        self.counts, self.events, self.receipt_seen = {'source': 0, 'locks': 0}, [], []
        self.drifts, self.extra_maps, self.library_fd, self.exiting, self.actions = set(), [], None, False, {}

    def cdll(self, path):
        self.paths = getattr(self, 'paths', []) + [(path, os.readlink(path))]
        self.library_fd = int(path.rsplit('/', 1)[1])
        self.maps.write_text(maps_line(self.fixture.static['library']['path']) + ''.join(maps_line(p) for p in self.extra_maps))
        return types.SimpleNamespace(sfora_sha256_occurrences=self.native)

    def spy(self, kind):
        def check():
            self.counts[kind] += 1
            self.events.append(kind)
            self.receipt_seen.append(os.path.lexists(self.output))
            if (kind, self.counts[kind]) in self.actions:
                self.actions[(kind, self.counts[kind])]()
            if (kind, self.counts[kind]) in self.drifts or self.exiting and (kind, 'exit') in self.drifts:
                raise ValueError('injected %s drift' % kind)
        return types.SimpleNamespace(check=check)

    def render(self, edit=None):
        return self.fixture.render(edit)

    def admit(self):
        return mod.read_native_authority(self.authority)

    def context(self):
        return types.SimpleNamespace(authority=self.admitted, fact=self.authority, locks=self.spy('locks'), source=self.spy('source'))

    def run_stage(self, **overrides):
        arguments = dict(torch=self.torch, cdll=self.cdll, maps=str(self.maps))
        arguments.update(overrides)
        return getattr(mod, self.stage)(self.context(), str(self.output), **arguments)

    def run_smoke(self, **overrides):
        return self.run_stage(**overrides)

    def held_error(self, **overrides):
        try:  # not assertRaises: it clears the traceback frames and would hide a pinned owner
            self.run_stage(**overrides)
        except BaseException as error:
            return error
        raise AssertionError('%s returned instead of rejecting' % self.stage)

    @staticmethod
    def frame_locals(error, name):
        """Locals still pinned by the retained error's traceback frame `name` (empty once the frame was cleared)."""
        trace = error.__traceback__
        while trace is not None:
            if trace.tb_frame.f_code.co_name == name:
                return dict(trace.tb_frame.f_locals)
            trace = trace.tb_next
        raise AssertionError('no %s frame in the traceback' % name)

    def fds(self):
        return len(os.listdir('/proc/self/fd'))

    def timed(self, admission=0.0):
        clock = Clock()
        self.stack.enter_context(mock.patch.object(mod, 'time', types.SimpleNamespace(perf_counter=clock)))
        self.stack.enter_context(mock.patch.object(mod, 'STARTED', clock.now - admission))
        return clock

    def on_call(self, number, action):
        seen = []

        def hook():
            seen.append(1)
            if len(seen) == number:
                action()
        self.native.hook = hook

    def on_exit_drain(self, action):
        real, done = self.torch.cuda.synchronize, []

        def synchronize():
            if not done:
                done.append(1)
                action()
            real()
        self.torch.cuda.synchronize = synchronize

    def with_extra(self, raw=b'inventoried but undeclared mapping'):
        """A second, inventoried, undeclared .so that is already mapped before the dlopen, as a Torch library would be."""
        extra = self.fixture.file('libextra.so', raw)

        def add(stage, record):
            if stage == 'inventory':
                record['files'][extra['path']] = {'sha256': extra['sha256'], 'size': len(raw)}
        self.authority = self.render(add)
        self.admitted = self.admit()
        self.reset()
        self.maps.write_text(maps_line(extra['path']))
        self.extra_maps = [extra['path']]
        return extra

    def close_failure(self):
        real, raised = os.close, []

        def close(fd):
            real(fd)
            if fd == self.library_fd and not raised:
                raised.append(1)
                raise OSError('close failed')
        self.stack.enter_context(mock.patch.object(mod.os, 'close', close))

    def spied_closure(self):
        real = mod.read_native_authority

        def spy(fact, *stage):
            self.events.append('closure')
            return real(fact, *stage)
        return mock.patch.object(mod, 'read_native_authority', spy)

    def refuse(self, path):
        raise OSError('dlopen failed')



class SmokeTests(StageHarness):
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
        self.assertEqual(self.counts['source'] + 1, self.counts['locks'])
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


    # ------------------------------------------------ body clock vs whole-process clock
    def test_body_clock_starts_at_body_entry_not_at_process_start(self):
        self.timed(admission=700)  # admission/hashing already spent 700 s of the 1500 s whole-process cap
        receipt = self.run_smoke()
        self.assertLess(receipt['resources']['body_seconds'], mod.POLICY['body_seconds'])
        self.assertGreaterEqual(receipt['resources']['wall_seconds'], 700)

    def test_body_overrun_is_measured_from_body_entry(self):
        clock = self.timed()
        self.on_call(30, lambda: setattr(clock, 'now', clock.now + 301))
        with self.assertRaisesRegex(ValueError, 'body deadline'):
            self.run_smoke()
        self.assertFalse(os.path.lexists(self.output))

    def test_the_last_check_group_is_still_covered_by_the_body_end_guard(self):
        for seconds, admission, message in ((301, 0, 'body deadline'), (10, 1195, 'headroom')):
            with self.subTest(message):
                self.reset()
                clock = self.timed(admission=admission)
                real = self.torch.zeros

                def zeros(*shape, **kwargs):
                    if shape == (1,):  # the last rejection fixture; no native call and no check follows it
                        clock.now += seconds
                    return real(*shape, **kwargs)
                self.torch.zeros = zeros
                with self.assertRaisesRegex(ValueError, message):
                    self.run_smoke()
                self.assertFalse(os.path.lexists(self.output))

    def test_exit_reserve_is_required_before_exit_begins(self):
        clock = self.timed(admission=1195)
        self.on_call(30, lambda: setattr(clock, 'now', clock.now + 10))
        with self.assertRaisesRegex(ValueError, 'headroom'):
            self.run_smoke()
        self.assertFalse(os.path.lexists(self.output))

    def test_exit_work_may_use_the_reserve_but_not_pass_the_whole_cap(self):
        for exit_seconds, accepted in ((600, True), (900, False)):
            with self.subTest(exit_seconds):
                self.reset()
                clock = self.timed(admission=600)
                self.on_exit_drain(lambda: setattr(clock, 'now', clock.now + exit_seconds))
                if accepted:
                    receipt = self.run_smoke()
                    self.assertGreaterEqual(receipt['resources']['wall_seconds'], 1200)
                    self.assertGreater(receipt['resources']['body_seconds'], mod.POLICY['body_seconds'])
                else:
                    with self.assertRaisesRegex(ValueError, 'headroom'):
                        self.run_smoke()
                    self.assertFalse(os.path.lexists(self.output))

    # ------------------------------------------------ repeated Locks/Source checks
    def test_locks_and_source_are_rechecked_at_every_checkpoint(self):
        self.run_smoke()
        self.assertEqual(self.counts['locks'], self.counts['source'] + 1)  # + the post-publication lock check
        self.assertGreaterEqual(self.counts['source'], 9)
        self.assertEqual(self.receipt_seen[:-1], [False] * (len(self.receipt_seen) - 1))
        self.assertTrue(self.receipt_seen[-1], 'the last lock check must follow the receipt write')

    def test_every_single_recheck_is_a_fail_closed_boundary(self):
        self.run_smoke()
        totals = dict(self.counts)
        self.assertGreaterEqual(totals['source'], 9)
        before = self.fds()
        for kind in ('locks', 'source'):
            for number in range(1, totals[kind] + 1):
                with self.subTest(kind=kind, number=number):
                    self.reset()
                    self.drifts = {(kind, number)}
                    with self.assertRaisesRegex(ValueError, 'injected'):
                        self.run_smoke()
                    # only the final post-publication lock check can fail with the diagnostic receipt already written
                    self.assertEqual(os.path.lexists(self.output), (kind, number) == ('locks', totals['locks']))
                    self.assertEqual(self.fds(), before)

    # ------------------------------------------------ complete fresh FILE closure at exit
    def test_exit_drains_then_rehashes_the_fresh_closure_then_source_then_locks_before_the_receipt(self):
        real = mod.read_native_authority
        self.on_exit_drain(lambda: self.events.append('drain'))

        def spy(fact):
            self.events.append('closure')
            return real(fact)
        with mock.patch.object(mod, 'read_native_authority', spy):
            self.run_smoke()
        self.assertEqual(self.events[-5:], ['drain', 'closure', 'source', 'locks', 'locks'])
        self.assertEqual(self.events.count('closure'), 1)

    def test_exit_rehash_catches_drift_in_every_closure_file(self):
        static, directory = self.fixture.static, self.fixture.dir
        paths = {name: Path(static[name]['path']) for name in (
            'nvcc', 'gxx', 'compile', 'evidence', 'log', 'runtime', 'proof_evidence', 'source', 'contract', 'driver',
            'test', 'probe', 'mlp', 'library')}
        paths.update({name: directory / name for name in (
            'build-authority.json', 'build-receipt.json', 'provenance.json', 'inventory.json', 'native-authority.json')})
        self.run_smoke()
        before = self.fds()
        for name, path in paths.items():
            with self.subTest(name):
                self.reset()
                original = path.read_bytes()

                def drift(path=path, original=original):
                    path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                self.on_exit_drain(drift)
                try:
                    with self.assertRaisesRegex(ValueError, 'SHA256|differs'):
                        self.run_smoke()
                finally:
                    path.write_bytes(original)
                self.assertFalse(os.path.lexists(self.output))
                self.assertEqual(self.fds(), before)

    def test_exit_rehash_rejects_a_changed_running_interpreter(self):
        other = self.fixture.static['nvcc']['path']
        self.on_exit_drain(lambda: self.stack.enter_context(mock.patch.object(sys, 'executable', other)))
        with self.assertRaisesRegex(ValueError, 'interpreter'):
            self.run_smoke()
        self.assertFalse(os.path.lexists(self.output))

    # ------------------------------------------------ mapped bytes are authenticated, not just inode/size
    def test_exact_undeclared_inventoried_mapping_is_hashed_and_accepted(self):
        extra = self.with_extra()
        self.assertIn(extra['path'], self.run_smoke()['mappings'])

    def test_undeclared_mapping_edited_in_place_after_open_fails_at_exit(self):
        extra = self.with_extra()
        path = Path(extra['path'])
        self.on_call(30, lambda: path.write_bytes(b'X' + path.read_bytes()[1:]))  # same inode, same size
        with self.assertRaisesRegex(ValueError, 'SHA256|differs'):
            self.run_smoke()
        self.assertFalse(os.path.lexists(self.output))

    # ------------------------------------------------ guaranteed cleanup preserving the primary rejection
    def test_library_descriptor_is_closed_on_success_and_on_every_failure(self):
        before = self.fds()
        self.run_smoke()
        self.assertEqual(self.fds(), before)
        for name in ('wrong_digest', 'poisoned_drain', 'not_pending', 'accepts_nulls'):
            with self.subTest(name):
                self.reset()
                self.arm(name)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_smoke()
                self.assertEqual(self.fds(), before)

    def test_keyboard_interrupt_closes_the_descriptor_and_stays_primary(self):
        before = self.fds()

        def interrupt():
            raise KeyboardInterrupt
        self.on_call(30, interrupt)
        with self.assertRaises(KeyboardInterrupt):
            self.run_smoke()
        self.assertEqual(self.fds(), before)
        self.assertFalse(os.path.lexists(self.output))

    def test_failed_body_releases_error_frame_owners_after_a_good_drain(self):
        self.arm('wrong_digest')
        error = self.held_error()
        self.assertIsInstance(error, ValueError)
        self.assertIn('digest', str(error))
        gc.collect()
        self.assertEqual([ref for ref in self.torch.made if ref() is not None], [], 'error frames pinned device owners')
        self.assertEqual(self.frame_locals(error, 'smoke_checks'), {}, 'error frame locals were not released')
        self.assertIsNotNone(error.__traceback__)

    def test_failed_final_drain_keeps_owners_and_never_masks_the_primary_rejection(self):
        self.arm('wrong_digest')
        self.torch.cuda.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('final drain'))
        before = self.fds()
        error = self.held_error()
        self.assertIsInstance(error, ValueError)
        self.assertIn('digest', str(error))
        self.assertTrue(any('final drain' in note for note in error.__notes__), error.__notes__)
        gc.collect()
        self.assertTrue([ref for ref in self.torch.made if ref() is not None], 'owners released after a failed drain')
        self.assertIn('tensors', self.frame_locals(error, 'smoke_checks'), 'error frames cleared after a failed drain')
        self.assertEqual(self.fds(), before)
        self.assertFalse(os.path.lexists(self.output))

    def test_a_quarantined_hasher_keeps_its_owners_even_after_a_good_final_drain(self):
        self.arm('poisoned_drain')
        error = self.held_error()
        self.assertIsInstance(error, (ValueError, RuntimeError))
        gc.collect()
        self.assertTrue([ref for ref in self.torch.made if ref() is not None], 'quarantined owners were released')
        self.assertIn('tensors', self.frame_locals(error, 'smoke_checks'), 'error frames cleared despite a quarantine')

    def test_cleanup_failure_alone_still_raises_without_a_receipt(self):
        self.close_failure()
        before = self.fds()
        with self.assertRaisesRegex(OSError, 'close failed'):
            self.run_smoke()
        self.assertEqual(self.fds(), before)
        self.assertFalse(os.path.lexists(self.output))

    def test_cleanup_failure_never_replaces_the_primary_rejection(self):
        self.arm('wrong_digest')
        self.close_failure()
        before = self.fds()
        error = self.held_error()
        self.assertIsInstance(error, ValueError)
        self.assertIn('digest', str(error))
        self.assertTrue(any('close failed' in note for note in error.__notes__), error.__notes__)
        self.assertEqual(self.fds(), before)


    # ------------------------------------------------ the independent exit also runs when the body failed
    def test_body_failure_still_attempts_the_full_independent_exit(self):
        self.arm('wrong_digest')
        before = self.fds()
        with self.spied_closure():
            error = self.held_error()
        self.assertIsInstance(error, ValueError)
        self.assertIn('digest', str(error))
        self.assertEqual(getattr(error, '__notes__', []), [])
        self.assertEqual(self.events[-3:], ['closure', 'source', 'locks'])
        self.assertEqual(self.fds(), before)
        self.assertFalse(os.path.lexists(self.output))

    def test_every_exit_check_is_attempted_despite_body_drain_and_other_exit_failures(self):
        self.arm('wrong_digest')
        nvcc = Path(self.fixture.static['nvcc']['path'])
        original = nvcc.read_bytes()

        def drain():
            self.exiting = True
            nvcc.write_bytes(bytes([original[0] ^ 1]) + original[1:])
            raise RuntimeError('final drain')
        self.torch.cuda.synchronize = drain
        self.drifts = {('source', 'exit'), ('locks', 'exit')}
        before = self.fds()
        try:
            with self.spied_closure():
                error = self.held_error()
        finally:
            nvcc.write_bytes(original)
        self.assertIsInstance(error, ValueError)
        self.assertIn('digest', str(error))  # the original body rejection stays primary
        notes = ' | '.join(error.__notes__)
        for fragment in ('final drain', 'SHA256', 'injected source drift', 'injected locks drift'):
            self.assertIn(fragment, notes)
        self.assertEqual(self.events[-3:], ['closure', 'source', 'locks'])
        self.assertEqual(self.fds(), before)
        self.assertFalse(os.path.lexists(self.output))

    def test_load_failure_before_the_mapping_snapshot_still_runs_the_exit_without_one(self):
        self.cdll = self.refuse
        before = self.fds()
        with self.spied_closure():
            error = self.held_error()
        self.assertIsInstance(error, OSError)
        self.assertIn('dlopen failed', str(error))
        self.assertEqual(getattr(error, '__notes__', []), [], 'a missing snapshot must not be an exit failure')
        self.assertEqual(self.events[-3:], ['closure', 'source', 'locks'])
        self.assertEqual(self.fds(), before)

    def test_current_mapped_bytes_are_validated_even_when_the_load_failed(self):
        extra = self.with_extra()
        path = Path(extra['path'])
        path.write_bytes(b'X' + path.read_bytes()[1:])  # same inode and size, bytes the root never froze
        self.cdll = self.refuse
        error = self.held_error()
        self.assertIsInstance(error, OSError)
        self.assertIn('dlopen failed', str(error))
        self.assertTrue(any('SHA256' in note for note in error.__notes__), getattr(error, '__notes__', None))

    def test_import_failure_still_runs_the_cpu_exit_checks(self):
        self.stack.enter_context(mock.patch.dict(sys.modules, {'torch': None}))
        with self.spied_closure():
            error = self.held_error(torch=None)
        self.assertIsInstance(error, ImportError)
        self.assertEqual(getattr(error, '__notes__', []), [])
        self.assertEqual(self.events[-3:], ['closure', 'source', 'locks'])

    def test_publication_that_crosses_the_whole_cap_is_not_silently_accepted(self):
        clock = self.timed(admission=1100)
        real = json.dump

        def dump(*args, **kwargs):
            clock.now += 450
            return real(*args, **kwargs)
        self.stack.enter_context(mock.patch.object(mod.json, 'dump', dump))
        with self.assertRaisesRegex(ValueError, 'headroom'):
            self.run_smoke()


# ------------------------------------------------------------------ the full stage: authority + genuine prior chain
def chain_at(stage, change):
    return lambda current, value: change(value) if current == stage else None


class FullAuthorityTests(AuthorityBase):
    own = True

    def admit(self, **kwargs):
        return mod.read_native_authority(self.fixture.render_full(**kwargs), 'full')

    def test_positive_read_binds_every_extension_file(self):
        admitted = self.admit()
        self.assertEqual(admitted.record['schema'], mod.NATIVE_FULL)
        self.assertEqual(admitted.record['full'].keys(), mod.FULL_KEYS)
        self.assertEqual(admitted.record['full']['prior'].keys(), mod.PRIOR_KEYS)
        self.assertEqual(admitted.record['full']['fixture'], self.fixture.static['fixture'])

    def test_smoke_and_full_records_are_not_interchangeable(self):
        full, smoke = self.fixture.render_full(), self.fixture.render()
        self.assertEqual(mod.read_native_authority(smoke).record['schema'], mod.NATIVE)
        for label, call in (('smoke given full', lambda: mod.read_native_authority(full)),
                            ('full given smoke', lambda: mod.read_native_authority(smoke, 'full')),
                            ('timing', lambda: mod.read_native_authority(full, 'timing'))):
            with self.subTest(label), self.assertRaises(ValueError):
                call()

    def test_extension_negatives(self):
        s = self.fixture.static
        wrong = {'sha256': '0' * 64}
        cases = {
            'extra_key': lambda f: f.update(extra=1), 'missing_fixture': lambda f: f.pop('fixture'),
            'prior_extra': lambda f: f['prior'].update(extra=s['log']), 'prior_missing': lambda f: f['prior'].pop('launch'),
            'duplicate_file': lambda f: f.update(extractor=f['metadata']),
            'reused_closure_file': lambda f: f.update(fixture=s['library']),
            'relative_path': lambda f: f.update(fixture={**s['fixture'], 'path': 'inventory.json'}),
            **{'sha_' + k: (lambda f, k=k: f.update({k: {**f[k], **wrong}})) for k in ('fixture', 'extractor', 'metadata')},
            **{'sha_prior_' + k: (lambda f, k=k: f['prior'].update({k: {**f['prior'][k], **wrong}})) for k in mod.PRIOR_KEYS}}
        for name, edit in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.admit(frozen=edit)

    def test_every_extension_file_is_rehashed_by_a_fresh_admission(self):
        fact = self.fixture.render_full()
        mod.read_native_authority(fact, 'full')
        for name in ('full-fixture-inventory.json', 'extract_siglip2_vision_source.py', 'so400-native256-upstream-metadata-v1.json', 'unit.json',
                     'launch.json', 'verification.json'):
            with self.subTest(name):
                path = self.fixture.dir / name
                original = path.read_bytes()
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                try:
                    with self.assertRaisesRegex(ValueError, 'differs'):
                        mod.read_native_authority(fact, 'full')
                finally:
                    path.write_bytes(original)


class PriorTests(AuthorityBase):
    own = True

    def read(self, unit=None, **kwargs):
        admitted = mod.read_native_authority(self.fixture.render_full(**kwargs), 'full')
        return mod.read_prior(unit or self.fixture.unit, admitted)

    def test_positive_chain_returns_json_facts_bound_to_the_unit(self):
        facts = self.read()
        self.assertEqual(facts['unit'], self.fixture.unit)
        self.assertEqual(facts['invocation_id'], self.fixture.INVOCATION)
        self.assertEqual(facts['unit_name'], self.fixture.UNIT)
        self.assertEqual(json.loads(json.dumps(facts)), facts)
        self.assertIs(mod.UNIT_READER, mod.read_prior)

    def test_only_the_frozen_unit_and_the_smoke_stage_are_read(self):
        admitted = mod.read_native_authority(self.fixture.render_full(), 'full')
        with self.assertRaisesRegex(ValueError, 'did not freeze'):
            mod.read_prior(self.fixture.static['log'], admitted)
        with self.assertRaisesRegex(ValueError, 'only released prior stage'):
            mod.read_prior(self.fixture.unit, admitted, 'full')

    def test_user_written_go_unit_is_never_accepted(self):
        go = self.fixture.json_file('go.json', {'schema': 'cuda-sha256-native-unit-v1', 'stage': 'smoke', 'decision': 'GO',
                                                'authority': self.fixture.static['driver']})
        with self.assertRaises(ValueError):
            self.read(unit=go, frozen=lambda f: f['prior'].update(unit=go))

    def test_semantic_negatives_on_each_chain_record(self):
        def edit_boot(lines, change):
            record = json.loads(lines[1])
            change(record)
            lines[1] = json.dumps(record, sort_keys=True)

        def line(index, text):
            return lambda lines: lines.__setitem__(index, text)
        s, flags = self.fixture.static, mod.FLAGS
        cases = {
            'footer_missing_line': ('footer', lambda l: l.pop()), 'footer_extra_line': ('footer', lambda l: l.append('x')),
            'footer_failed': ('footer', line(2, 'Finished with result: exit-code')),
            'footer_status1': ('footer', line(3, 'Main processes terminated with: code=exited/status=1')),
            'footer_signal': ('footer', line(3, 'Main processes terminated with: code=killed/status=9/KILL')),
            'footer_wrong_unit': ('footer', line(0, 'Running as unit: other.service; invocation ID: ' + self.fixture.INVOCATION)),
            'footer_wrong_id': ('footer', line(0, 'Running as unit: %s.service; invocation ID: %s' % (self.fixture.UNIT, '0' * 32))),
            'footer_runtime': ('footer', line(4, 'Service runtime: 4.394s')), 'footer_swap_peak': ('footer', line(7, 'Memory swap peak: 4.0K')),
            'footer_swapped_lines': ('footer', lambda l: l.insert(2, l.pop(3))),
            'footer_bootstrap_false': ('footer', lambda l: edit_boot(l, lambda r: r.update(bootstrap_exit_pass=False))),
            'footer_oom_kill': ('footer', lambda l: edit_boot(l, lambda r: r['resources']['memory_events'].update(oom_kill=1))),
            'footer_swap': ('footer', lambda l: edit_boot(l, lambda r: r['resources'].update(swap_current_bytes=1))),
            'footer_peak': ('footer', lambda l: edit_boot(l, lambda r: r['resources'].update(memory_peak_bytes=8 * 1024**3 + 1))),
            'footer_wall': ('footer', lambda l: edit_boot(l, lambda r: r['resources'].update(wall_seconds=1500.0))),
            'footer_event_key': ('footer', lambda l: edit_boot(l, lambda r: r['resources']['memory_events'].pop('oom'))),
            'log_no_newline': ('log', lambda raw: raw.__setitem__(0, raw[0][:-1])), 'log_crlf': ('log', lambda raw: raw.__setitem__(0, raw[0].replace('\n', '\r\n'))),
            'log_cut': ('log', lambda raw: raw.__setitem__(0, raw[0][:90])), 'log_empty': ('log', lambda raw: raw.__setitem__(0, '')),
            'log_not_utf8': ('log', lambda raw: raw.__setitem__(0, raw[0] + '\udcff')),
            'unit_extra': ('unit', lambda u: u.update(extra=1)), 'unit_locks_false': ('unit', lambda u: u.update(both_locks_held=False)),
            'unit_locks_int': ('unit', lambda u: u.update(both_locks_held=1)),
            'unit_id_upper': ('unit', lambda u: u.update(invocation_id=u['invocation_id'].upper())),
            'unit_id_other': ('unit', lambda u: u.update(invocation_id='1' * 32)), 'unit_service': ('unit', lambda u: u.update(service_seconds=4.0)),
            'unit_service_bool': ('unit', lambda u: u.update(service_seconds=True)), 'unit_rss': ('unit', lambda u: u.update(native_peak_rss_kib=1)),
            'unit_receipt_is_log': ('unit', lambda u: u.update(receipt=u['log'])),
            'receipt_key': ('receipt', lambda r: r.update(extra=1)), 'receipt_status': ('receipt', lambda r: r.update(status='PASS')),
            'receipt_engineering': ('receipt', lambda r: r.update(engineering_only=False)),
            'receipt_terminal': ('receipt', lambda r: r.update(normal_terminal_required=False)),
            **{'receipt_flag_' + f: ('receipt', lambda r, f=f: r.update({f: True})) for f in flags},
            'receipt_calls': ('receipt', lambda r: r.update(native_calls=23)), 'receipt_calls_bool': ('receipt', lambda r: r.update(native_calls=True)),
            'receipt_occurrences': ('receipt', lambda r: r['checks']['parity'].update(occurrences=128)),
            'receipt_bool_for_int': ('receipt', lambda r: r['checks']['raw_abi'].update(all_null=True)),
            'receipt_check_missing': ('receipt', lambda r: r['checks'].pop('rejections')),
            'receipt_check_extra': ('receipt', lambda r: r['checks'].update(extra=1)),
            'receipt_injected': ('receipt', lambda r: r['checks']['injected_failures'].update(status='InjectedFault')),
            'receipt_version_bool': ('receipt', lambda r: r['checks']['mutation'].update(version=True)),
            'receipt_trees': ('receipt', lambda r: r['checks']['typed_trees'].update(trees=8)),
            'receipt_argv': ('receipt', lambda r: r['invocation']['argv'].__setitem__(1, 'full')),
            'receipt_optimize': ('receipt', lambda r: r['invocation'].update(optimize=1)),
            'receipt_python': ('receipt', lambda r: r['invocation'].update(python=s['nvcc']['path'])),
            'receipt_body': ('receipt', lambda r: r['resources'].update(body_seconds=300.0)),
            'receipt_wall': ('receipt', lambda r: r['resources'].update(wall_seconds=4.2)),
            'receipt_rss': ('receipt', lambda r: r['resources'].update(process_peak_rss_kib=1)),
            'receipt_cuda_cap': ('receipt', lambda r: r['resources'].update(peak_cuda_allocated_bytes=10**10)),
            'receipt_resource_key': ('receipt', lambda r: r['resources'].pop('body_seconds')),
            'receipt_library': ('receipt', lambda r: r.update(library=s['runtime'])),
            'receipt_device': ('receipt', lambda r: r['device'].update(name='Other')),
            'receipt_policy': ('receipt', lambda r: r['resource_policy'].update(body_seconds=301)),
            'receipt_maps_no_library': ('receipt', lambda r: r.update(mappings=[r['mappings'][1]])),
            'receipt_maps_outside': ('receipt', lambda r: r.update(mappings=sorted([*r['mappings'], '/x/libx.so']))),
            'receipt_maps_unsorted': ('receipt', lambda r: r.update(mappings=r['mappings'][::-1])),
            'receipt_authority': ('receipt', lambda r: r.update(authority=s['evidence'])),
            'launch_product': ('launch', lambda l: l.update(product_go=True)), 'launch_quality': ('launch', lambda l: l.update(quality_go=True)),
            'launch_unit': ('launch', lambda l: l.update(unit='other')), 'launch_authority': ('launch', lambda l: l.update(native_authority=s['evidence'])),
            'launch_output': ('launch', lambda l: l.update(output=l['output'] + '2')),
            'launch_property': ('launch', lambda l: l['command'].__setitem__(7, '--property=MemoryMax=1')),
            'launch_no_property': ('launch', lambda l: l['command'].pop(9)), 'launch_extra_arg': ('launch', lambda l: l['command'].append('x')),
            'launch_setenv': ('launch', lambda l: l['command'].__setitem__(12, '--setenv=CUDA_VISIBLE_DEVICES=1')),
            'launch_bootstrap_sha': ('launch', lambda l: l['command'].__setitem__(-1, '0' * 64)),
            'launch_isolation': ('launch', lambda l: l['command'].__setitem__(-6, '-S')),
            'launch_systemd': ('launch', lambda l: l['command'].__setitem__(0, '/tmp/systemd-run')),
            'launch_relative_python': ('launch', lambda l: l['command'].__setitem__(16, 'python')),
            'boot_output': ('bootstrap-authority', lambda b: b.update(output='/x')),
            'boot_driver': ('bootstrap-authority', lambda b: b['files'].update(driver=s['test'])),
            'boot_missing': ('bootstrap-authority', lambda b: b['files'].pop('requests')),
            'boot_authority': ('bootstrap-authority', lambda b: b.update(native_authority=s['evidence'])),
            'boot_interpreter': ('bootstrap-authority', lambda b: b.update(interpreter=s['nvcc'])),
            'verify_terminal': ('verification', lambda v: v['terminal'].update(service_seconds=4.4)),
            'verify_outer': ('verification', lambda v: v['outer_resources'].update(wall_seconds=4.0)),
            'verify_checks': ('verification', lambda v: v.update(all_smoke_checks=False)),
            'verify_mapped': ('verification', lambda v: v.update(complete_current_mapped_file_hashes_exit_pass=False)),
            'verify_product': ('verification', lambda v: v.update(product_go=True)), 'verify_speed': ('verification', lambda v: v.update(speed_go=True)),
            'verify_exit': ('verification', lambda v: v.update(exit_status=1)), 'verify_extra': ('verification', lambda v: v.update(go='GO'))}
        for name, (stage, change) in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.read(chain=chain_at(stage, change))

    def test_prior_authority_closure_must_equal_the_full_authority_closure(self):
        s = self.fixture.static
        cases = {'library': lambda r: r.update(library=s['runtime']), 'device': lambda r: r['device'].update(index=1),
                 'locks': lambda r: r.update(locks=r['locks'][:1]), 'policy': lambda r: r['resource_policy'].update(body_seconds=299),
                 'probe_serializer': lambda r: r['sources'].update(probe_serializer=s['mlp']),
                 'mlp_serializer': lambda r: r['sources'].update(mlp_serializer=s['probe']),
                 'build_receipt': lambda r: r.update(build_receipt=s['log']), 'interpreter': lambda r: r.update(interpreter=s['nvcc']),
                 'runtime_row': lambda r: r['runtime_files'].append(r['runtime_files'][0]), 'runtime_empty': lambda r: r.update(runtime_files=[])}
        for name, change in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.read(edit=at('prior-native', change))

    def test_any_changed_byte_of_any_chain_file_is_substitution(self):
        admitted = mod.read_native_authority(self.fixture.render_full(), 'full')
        self.assertEqual(mod.read_prior(self.fixture.unit, admitted)['invocation_id'], self.fixture.INVOCATION)
        for name in ('unit.json', 'smoke-out/smoke-receipt.json', 'original-smoke.log', 'launch.json', 'bootstrap.py',
                     'bootstrap-authority.json', 'verification.json', 'prior-authority.json', 'prior_driver.py', 'prior_test.py', 'helper.py'):
            with self.subTest(name):
                path = self.fixture.dir / name
                original = path.read_bytes()
                path.write_bytes(bytes([original[0] ^ 1]) + original[1:])
                try:
                    with self.assertRaisesRegex(ValueError, 'differs'):
                        mod.read_prior(self.fixture.unit, admitted)
                finally:
                    path.write_bytes(original)

    def test_wrong_shapes_in_a_forged_record_are_value_errors(self):
        for name, stage, change in (('receipt_checks_str', 'receipt', lambda r: r.update(checks='x')),
                                    ('receipt_resources_none', 'receipt', lambda r: r.update(resources=None)),
                                    ('launch_command_int', 'launch', lambda l: l.update(command=3)),
                                    ('boot_files_list', 'bootstrap-authority', lambda b: b.update(files=[])),
                                    ('verification_terminal_none', 'verification', lambda v: v.update(terminal=None)),
                                    ('unit_unit_int', 'unit', lambda u: u.update(unit=3))):
            with self.subTest(name), self.assertRaises(ValueError):
                self.read(chain=chain_at(stage, change))


GENUINE_ROOT = '/home/riomus/runs/sfora-standalone-gpu-sha-smoke-source-v2/'
FREEZE, RESULT = EVIDENCE / 'standalone-gpu-sha-smoke-v2-freeze', EVIDENCE / 'standalone-gpu-sha-smoke-v2-result'


def genuine_path(path):
    """The committed copies stand in for the root's /home/riomus files (same bytes, same sha256)."""
    if path == '/home/riomus/runs/sfora-standalone-gpu-sha-smoke-v2/smoke-receipt.json':
        return RESULT / 'receipt.json'
    if path == GENUINE_ROOT + 'original-smoke-v2.log':
        return RESULT / 'original.log'
    if path.startswith(GENUINE_ROOT):
        return FREEZE / path[len(GENUINE_ROOT):]
    if path.startswith((str(FREEZE), str(RESULT))):
        return Path(path)
    raise FileNotFoundError(path)


class GenuinePriorTests(unittest.TestCase):
    """The reader against the ORIGINAL smoke-v2 bytes (exact eight-line log, launch, verification, receipt, unit, bootstrap)."""

    def setUp(self):
        prior = json.loads((FREEZE / 'native-authority-v2.json').read_text())
        current = json.loads(json.dumps(prior))
        current['schema'] = mod.NATIVE_FULL
        current['full'] = {'prior': {'unit': fact(RESULT / 'unit.json'), 'launch': fact(FREEZE / 'launch.json'),
                                     'verification': fact(RESULT / 'verification.json')},
                           'fixture': fact(FIXTURE_FILE), 'extractor': fact(EXTRACTOR), 'metadata': fact(METADATA)}
        files = json.loads((FREEZE / 'mapping-inventory-v1.json').read_text())['files']
        self.admitted = types.SimpleNamespace(record=current, inventory=files)
        self.unit, self.overrides = current['full']['prior']['unit'], {}
        self.real_read, self.real_hash = mod.read_bytes, mod.hash_file

    def key(self, name):
        launch = json.loads((FREEZE / 'launch.json').read_text())
        return {'log': GENUINE_ROOT + 'original-smoke-v2.log', 'receipt': '/home/riomus/runs/sfora-standalone-gpu-sha-smoke-v2/smoke-receipt.json',
                'unit': self.unit['path'], 'launch': str(FREEZE / 'launch.json'), 'verification': str(RESULT / 'verification.json'),
                'bootstrap-authority': launch['bootstrap_authority']['path'], 'prior': launch['native_authority']['path']}[name]

    def mutate(self, name, change):
        """Replace one genuine FILE's bytes; the sha256 binding is deliberately bypassed so the SEMANTIC check is what rejects."""
        key = self.key(name)
        raw = (genuine_path(key)).read_bytes()
        if name == 'log':
            self.overrides[key] = change(raw.decode()).encode('utf-8', 'surrogateescape')
        else:
            record = json.loads(raw)
            change(record)
            self.overrides[key] = json.dumps(record, sort_keys=True).encode()

    def read(self):
        def read_bytes(item, limit):
            if item['path'] in self.overrides:
                return self.overrides[item['path']]
            return self.real_read({'path': str(genuine_path(item['path'])), 'sha256': item['sha256']}, limit)

        def hash_file(item):
            if item['path'] not in self.overrides:
                return self.real_hash({'path': str(genuine_path(item['path'])), 'sha256': item['sha256']})
        with mock.patch.object(mod, 'read_bytes', read_bytes), mock.patch.object(mod, 'hash_file', hash_file):
            return mod.read_prior(self.unit, self.admitted)

    def test_the_original_normal0_smoke_v2_chain_is_accepted(self):
        facts = self.read()
        self.assertEqual(facts['invocation_id'], 'e594e58e8ad1495a8150757ca754e3fc')
        self.assertEqual(facts['unit_name'], 'sfora-standalone-gpu-sha-smoke-v2')
        self.assertEqual(facts['service_seconds'], 4.393)
        self.assertEqual(facts['outer_resources']['memory_peak_bytes'], 436899840)
        self.assertEqual(len((RESULT / 'original.log').read_text().splitlines()), 8)

    def test_genuine_log_mutations_and_truncations_are_rejected(self):
        original = (RESULT / 'original.log').read_text()
        lines = original.split('\n')[:-1]
        cases = {'drop_last': '\n'.join(lines[:-1]) + '\n', 'drop_first': '\n'.join(lines[1:]) + '\n', 'drop_middle': '\n'.join(lines[:3] + lines[4:]) + '\n',
                 'no_newline': original[:-1], 'crlf': original.replace('\n', '\r\n'), 'cut': original[:200], 'empty': '', 'extra': original + 'x\n',
                 'blank': original + '\n', 'status1': original.replace('status=0', 'status=1'), 'result': original.replace('success', 'exit-code'),
                 'id': original.replace('e594e58e', 'e594e58f'), 'unit': original.replace('smoke-v2.service', 'smoke-v3.service'),
                 'boot_false': original.replace('"bootstrap_exit_pass": true', '"bootstrap_exit_pass": false'),
                 'oom': original.replace('"oom_kill": 0', '"oom_kill": 1'), 'swap': original.replace('"swap_current_bytes": 0', '"swap_current_bytes": 4096'),
                 'runtime': original.replace('4.393s', '4.394s'), 'swap_peak': original.replace('0B', '1B'), 'peak': original.replace('436899840', '8589934593'),
                 'swapped': '\n'.join([lines[0], lines[2], lines[1], *lines[3:]]) + '\n', 'non_utf8': original + '\udcff'}
        self.assertEqual(self.read()['invocation_id'], 'e594e58e8ad1495a8150757ca754e3fc')
        for name, text in cases.items():
            with self.subTest(name):
                self.overrides.clear()
                self.mutate('log', lambda raw, text=text: text)
                with self.assertRaises(ValueError):
                    self.read()

    def test_genuine_launch_verification_receipt_unit_and_bootstrap_mutations_are_rejected(self):
        cases = {
            'launch_product': ('launch', lambda r: r.update(product_go=True)), 'launch_unit': ('launch', lambda r: r.update(unit='x')),
            'launch_memory': ('launch', lambda r: r['command'].__setitem__(8, '--property=MemoryMax=8589934593')),
            'launch_runtime': ('launch', lambda r: r['command'].__setitem__(7, '--property=RuntimeMaxSec=1501')),
            'launch_drop_swap': ('launch', lambda r: r['command'].pop(9)), 'launch_bootstrap': ('launch', lambda r: r['command'].__setitem__(-1, '0' * 64)),
            'launch_output': ('launch', lambda r: r.update(output='/home/riomus/runs/other')),
            'launch_authority': ('launch', lambda r: r['native_authority'].update(sha256='0' * 64)),
            'verify_terminal': ('verification', lambda r: r['terminal'].update(unit='x')), 'verify_outer': ('verification', lambda r: r['outer_resources'].update(wall_seconds=1.0)),
            'verify_checks': ('verification', lambda r: r.update(all_smoke_checks=False)), 'verify_exit': ('verification', lambda r: r.update(exit_status=1)),
            'verify_product': ('verification', lambda r: r.update(product_go=True)),
            'receipt_flag': ('receipt', lambda r: r.update(speed_go=True)), 'receipt_calls': ('receipt', lambda r: r.update(native_calls=23)),
            'receipt_parity': ('receipt', lambda r: r['checks']['parity'].update(occurrences=128)),
            'receipt_abi': ('receipt', lambda r: r['checks']['raw_abi'].update(out_null=0)),
            'receipt_injected': ('receipt', lambda r: r['checks']['injected_failures'].update(readback='ValueError')),
            'receipt_optimize': ('receipt', lambda r: r['invocation'].update(optimize=2)),
            'receipt_argv': ('receipt', lambda r: r['invocation']['argv'].__setitem__(7, '/home/riomus/runs/other')),
            'receipt_body': ('receipt', lambda r: r['resources'].update(body_seconds=300.1)),
            'receipt_mappings': ('receipt', lambda r: r['mappings'].remove(r['library']['path'])), 'receipt_library': ('receipt', lambda r: r['library'].update(sha256='0' * 64)),
            'unit_locks': ('unit', lambda r: r.update(both_locks_held=False)), 'unit_id': ('unit', lambda r: r.update(invocation_id='f' * 32)),
            'unit_extra': ('unit', lambda r: r.update(decision='GO')), 'unit_rss': ('unit', lambda r: r.update(native_peak_rss_kib=666529)),
            'boot_output': ('bootstrap-authority', lambda r: r.update(output='/home/riomus/runs/other')),
            'boot_driver': ('bootstrap-authority', lambda r: r['files']['driver'].update(sha256='0' * 64)),
            'prior_locks': ('prior', lambda r: r['locks'].pop()), 'prior_library': ('prior', lambda r: r['library'].update(sha256='0' * 64)),
            'prior_runtime': ('prior', lambda r: r['runtime_files'][0]['file'].update(sha256='0' * 64)),
            'prior_probe': ('prior', lambda r: r['sources']['probe_serializer'].update(sha256='0' * 64))}
        for name, (target, change) in cases.items():
            with self.subTest(name):
                self.overrides.clear()
                self.mutate(target, change)
                with self.assertRaises(ValueError):
                    self.read()

    def test_a_user_written_go_unit_cannot_stand_in_for_the_genuine_unit(self):
        self.mutate('unit', lambda r: (r.clear(), r.update(schema='cuda-sha256-native-unit-v1', stage='smoke', decision='GO')))
        with self.assertRaisesRegex(ValueError, 'smoke unit record'):
            self.read()

    def test_the_committed_evidence_is_itself_unchanged(self):
        for name, digest in (('standalone-gpu-sha-smoke-v2-result/receipt.json', '45eff741cc688c616eab66ae1aad0f83e0250fb025c64d2e1d28d3b02c0fc7a8'),
                             ('standalone-gpu-sha-smoke-v2-result/original.log', 'b1e6fc4a01ca51b4a27574f2a10649d049cc24e8f2e1d87c33db14bb8cc33709'),
                             ('standalone-gpu-sha-smoke-v2-freeze/native-authority-v2.json', '8f11965ce5f79a943d2d3aab2ea85cd509de1f5b1a5e6333ef201758f07c1887')):
            self.assertEqual(sha((EVIDENCE / name).read_bytes()), digest)


# ------------------------------------------------------------------ the synthetic inventory (host only)
def replacing(old, new):
    def edit(raw):
        if old not in raw:
            raise AssertionError('test edit did not apply: %r' % old)
        return raw.replace(old, new, 1)
    return edit


class InventoryTests(AuthorityBase):
    own = True

    def record(self, fixture=None, metadata=None, extractor=None, probe=None, mlp=None, bind=True):
        def variant(label, origin, edit=None):
            (self.fixture.dir / label).mkdir(exist_ok=True)
            raw = origin.read_bytes() if edit is None else edit(origin.read_bytes())
            return self.fixture.file('%s/%s' % (label, origin.name), raw)
        meta, ext = variant('meta', METADATA, metadata), variant('ext', EXTRACTOR, extractor)
        record = json.loads(FIXTURE_FILE.read_text())
        if bind:
            record['metadata']['sha256'], record['source']['sha256'] = meta['sha256'], ext['sha256']
        if fixture:
            fixture(record)
        return {'full': {'fixture': self.fixture.json_file('fx.json', record), 'extractor': ext, 'metadata': meta},
                'sources': {'probe_serializer': variant('probe', SERIALIZERS['probe'], probe),
                            'mlp_serializer': variant('mlp', SERIALIZERS['mlp'], mlp)}}

    def test_the_real_fixture_extractor_metadata_and_roles_agree(self):
        inventory = mod.build_inventory(self.record())
        facts = mod.inventory_facts(inventory)
        self.assertEqual((facts['leaves'], facts['bytes'], facts['largest_leaf_bytes']), (448, 1711552256, 19832832))
        self.assertEqual(facts['largest'], 'encoder.layers.0.mlp.fc1.weight')
        self.assertEqual(inventory.shapes[facts['largest']], (4304, 1152))
        self.assertEqual(facts['frozen_leaves'], {'probe': 447, 'mlp': 444})
        self.assertEqual(inventory.roles['probe'], ('head.probe',))
        self.assertEqual(inventory.roles['mlp'], tuple('encoder.layers.26.mlp.%s.%s' % (a, b) for a in ('fc1', 'fc2') for b in ('weight', 'bias')))
        self.assertFalse(any(n.startswith('vision_model.') for n in inventory.shapes))
        self.assertEqual(json.loads(json.dumps(facts)), facts)
        self.assertNotIn('extract_siglip2_vision_source', sys.modules, 'the extractor must be AST-extracted, never imported')

    def test_tampered_fixture_extractor_metadata_or_roles_are_rejected(self):
        def shape(record):
            record['shapes']['head.probe'] = [1, 1, 1151]
        cases = {
            'fixture_shape': dict(fixture=shape), 'fixture_bytes': dict(fixture=lambda r: r.update(bytes=r['bytes'] + 4)),
            'fixture_leaves': dict(fixture=lambda r: r.update(leaves=447)), 'fixture_largest': dict(fixture=lambda r: r.update(largest_leaf_bytes=4)),
            'fixture_probe': dict(fixture=lambda r: r.update(probe=['head.other'])), 'fixture_mlp': dict(fixture=lambda r: r['mlp'].reverse()),
            'fixture_model': dict(fixture=lambda r: r.update(model='google/siglip2-large-patch16-256')),
            'fixture_resolved': dict(fixture=lambda r: r['resolved'].update(vision_use_head=1)),
            'fixture_quality': dict(fixture=lambda r: r.update(quality_read=True)), 'fixture_real': dict(fixture=lambda r: r.update(synthetic=False)),
            'fixture_extra': dict(fixture=lambda r: r.update(extra=1)), 'fixture_schema': dict(fixture=lambda r: r.update(schema='x')),
            'fixture_dropped_leaf': dict(fixture=lambda r: r['shapes'].pop('head.probe')),
            'fixture_unbound_metadata': dict(bind=False, metadata=lambda raw: raw + b'\n'),
            'fixture_unbound_extractor': dict(bind=False, extractor=lambda raw: raw + b'\n'), 'fixture_source_sha': dict(fixture=lambda r: r['source'].update(sha256='0' * 64)),
            'fixture_metadata_sha': dict(fixture=lambda r: r['metadata'].update(sha256='0' * 64)),
            'fixture_metadata_name': dict(fixture=lambda r: r['metadata'].update(path='docs/other.json')),
            'metadata_width': dict(metadata=replacing(b'"hidden_size": 1152,\n      "image_size"', b'"hidden_size": 1024,\n      "image_size"')),
            'metadata_model': dict(metadata=replacing(b'"model": "google/siglip2-so400m-patch16-256"', b'"model": "google/other"')),
            'extractor_prefix': dict(extractor=replacing(b"PREFIX = 'vision_model.'", b"PREFIX = 'vm.'")),
            'extractor_second_prefix': dict(extractor=lambda raw: raw + b"\nPREFIX = 'x'\n"),
            'extractor_no_function': dict(extractor=replacing(b'def expected_vision(', b'def other_vision(')),
            'extractor_profile': dict(extractor=replacing(b'(1152, 27, 4304)', b'(1152, 27, 4305)')),
            'probe_role': dict(probe=replacing(b"PROBE = ('head.probe',)", b"PROBE = ('head.attention.in_proj_bias',)")),
            'probe_shape': dict(probe=replacing(b'PROBE_SHAPES = [[1,1,1152]]', b'PROBE_SHAPES = [[1,1,1151]]')),
            'probe_second': dict(probe=lambda raw: raw + b"\nPROBE = ('head.probe',)\n"),
            'mlp_shape': dict(mlp=replacing(b'MLP_SHAPES = [[4304, 1152]', b'MLP_SHAPES = [[4304, 1151]')),
            'mlp_layer': dict(mlp=replacing(b'"encoder.layers.26.mlp."', b'"encoder.layers.25.mlp."'))}
        for name, kwargs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                mod.build_inventory(self.record(**kwargs))


class LeafTests(Base):
    def test_leaves_are_deterministic_per_name_and_separately_allocated(self):
        generator = self.torch.Generator(self.device)
        first = mod.allocate_leaf(self.torch, self.device, generator, 'a.weight', (3, 5))
        again = mod.allocate_leaf(self.torch, self.device, generator, 'a.weight', (3, 5))
        other = mod.allocate_leaf(self.torch, self.device, generator, 'b.weight', (3, 5))
        self.assertEqual((first.shape, first.dtype, first.device), ((3, 5), F32, self.device))
        self.assertEqual(first.logical(), again.logical())
        self.assertNotEqual(first.logical(), other.logical())
        self.assertNotEqual(first.data_ptr(), again.data_ptr())
        self.assertEqual(len({mod.leaf_seed('n%d' % i) for i in range(448)}), 448)
        self.assertTrue(all(0 <= mod.leaf_seed('n%d' % i) < 2**63 for i in range(448)))

    def test_check_leaves_rejects_aliases_overlaps_and_wrong_allocations(self):
        inventory = types.SimpleNamespace(shapes={'a': (4, 4), 'b': (2, 4)})
        generator = self.torch.Generator(self.device)
        good = {n: mod.allocate_leaf(self.torch, self.device, generator, n, s) for n, s in inventory.shapes.items()}
        self.assertEqual([n for _, n in mod.check_leaves(self.torch, self.device, inventory, good)], [64, 32])
        a = good['a']
        host, wrong = self.torch.zeros(2, 4), self.torch.make(self.device, (2, 4), F64)
        cases = {'alias': {'a': a, 'b': a.reshape(2, 8)}, 'wrong_shape': {'a': a, 'b': good['b'].reshape(8)},
                 'offset_view': {'a': a, 'b': a.reshape(-1)[4:12].reshape(2, 4)}, 'host': {'a': a, 'b': host}, 'dtype': {'a': a, 'b': wrong},
                 'missing': {'a': a}, 'extra': {**good, 'c': a}, 'overlap_whole': {'a': a, 'b': a.reshape(-1)[:8].reshape(2, 4)}}
        for name, leaves in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                mod.check_leaves(self.torch, self.device, inventory, leaves)


# ------------------------------------------------------------------ the full stage body (fake torch, scaled inventory)
REAL_BUILD = mod.build_inventory
ROLE_MLP = tuple('encoder.layers.26.mlp.%s.%s' % (a, b) for a in ('fc1', 'fc2') for b in ('weight', 'bias'))
SMALL = {'head.probe': (1, 1, 6), ROLE_MLP[0]: (4, 6), ROLE_MLP[1]: (4,), ROLE_MLP[2]: (6, 4), ROLE_MLP[3]: (6,),
         'post_layernorm.weight': (6,), 'post_layernorm.bias': (6,), 'embeddings.patch_embedding.bias': (6,),
         'encoder.layers.0.mlp.fc1.weight': (3, 4000), 'encoder.layers.0.self_attn.q_proj.bias': (5,),
         'head.layernorm.bias': (7,), 'head.attention.in_proj_bias': (18,)}


def small_inventory():
    return types.SimpleNamespace(shapes=dict(SMALL), roles={'probe': ('head.probe',), 'mlp': ROLE_MLP},
                                 largest='encoder.layers.0.mlp.fc1.weight')


class FullTests(StageHarness):
    stage = 'full'

    def setUp(self):
        self.inventory, self.leaves = small_inventory(), {}
        super().setUp()

    def render(self, edit=None):
        return self.fixture.render_full(edit)

    def admit(self):
        return mod.read_native_authority(self.authority, 'full')

    def context(self):
        context = super().context()
        context.unit, context.prior = self.fixture.unit, mod.read_prior(self.fixture.unit, self.admitted)
        return context

    def reset(self):
        super().reset()
        self.torch.defer = True
        self.leaves = {}
        real = mod.check_leaves

        def capturing(torch, device, inventory, leaves):
            self.leaves = dict(leaves)  # strong references to exactly the inventory leaves (not the largest-leaf gate copies)
            return real(torch, device, inventory, leaves)
        self.stack.enter_context(mock.patch.object(mod, 'check_leaves', capturing))
        self.stack.enter_context(mock.patch.object(mod, 'build_inventory', lambda record: self.inventory))

    def original(self, name='probe'):
        return original_fingerprint(SERIALIZERS[name])[0]

    def test_full_stage_passes_and_writes_a_discarded_diagnostic(self):
        receipt = self.run_stage()
        self.assertEqual((receipt['schema'], receipt['status']), (mod.FULL_RECEIPT, 'FULL_DIAGNOSTIC_UNREVIEWED'))
        self.assertTrue(receipt['engineering_only'] and receipt['normal_terminal_required'])
        self.assertTrue(all(receipt[flag] is False for flag in mod.FLAGS))
        self.assertEqual(receipt['resource_policy'], mod.POLICY)
        self.assertEqual({k: receipt[k] for k in ('fixture', 'extractor', 'metadata')},
                         {k: self.fixture.static[k] for k in ('fixture', 'extractor', 'metadata')})
        self.assertEqual(receipt['prior']['invocation_id'], self.fixture.INVOCATION)
        written = json.loads((self.output / 'full-receipt.json').read_text())
        self.assertEqual((written['schema'], written['authority'], written['prior']), (mod.FULL_RECEIPT, self.authority, receipt['prior']))
        self.assertFalse((self.output / 'smoke-receipt.json').exists())
        checks = receipt['checks']
        self.assertEqual(sorted(checks), ['comparison', 'inventory', 'largest_leaf', 'workloads'])
        self.assertEqual(checks['inventory']['frozen_leaves'], {'probe': 11, 'mlp': 8})
        self.assertEqual(sorted(checks['largest_leaf']), ['aliases', 'injected_failures', 'mutation', 'parity', 'streams'])
        self.assertEqual(checks['largest_leaf']['injected_failures'], {
            'status': 'ValueError', 'launched': 'InjectedFault', 'completed': 'InjectedFault', 'readback': 'InjectedFault'})
        self.assertEqual({k: v['launches'] for k, v in checks['workloads'].items()}, {'probe': [11, 12], 'mlp': [8, 12], 'largest': [1]})
        comparison = checks['comparison']
        self.assertIn('Outside the clocks', comparison['clock'])
        self.assertEqual(comparison['pairs'], 3)
        for label, row in comparison['workloads'].items():
            self.assertEqual(len(row['pairs']), 3)
            self.assertEqual([p['order'] for p in row['pairs']], [['original', 'candidate'], ['candidate', 'original'], ['original', 'candidate']])
            self.assertTrue(all(len(p[arm + '_call_seconds']) == len(row['trees']) and p[arm + '_seconds'] >= 0
                                for p in row['pairs'] for arm in ('original', 'candidate')))
            self.assertEqual(row['isolated_mechanism_only'], label == 'largest')
        self.assertGreater(len(self.native.calls), receipt['native_calls'])  # injected-failure hashers call the native function directly
        # an independent ORIGINAL fingerprint of the current leaves
        names = list(SMALL)
        probe, mlp = self.original('probe'), self.original('mlp')
        digests = checks['workloads']
        self.assertEqual(digests['probe']['digests'], [probe({n: self.leaves[n] for n in names if n != 'head.probe'}),
                                                       probe({n: self.leaves[n] for n in names})])
        self.assertEqual(digests['mlp']['digests'], [mlp({n: self.leaves[n] for n in names if n not in ROLE_MLP}),
                                                     mlp({n: self.leaves[n] for n in names})])
        self.assertNotEqual(digests['probe']['digests'][0], digests['probe']['digests'][1])
        self.leaves.clear()
        gc.collect()
        self.assertEqual([ref for ref in self.torch.made if ref() is not None], [], 'a finished stage must not pin any device owner')

    def test_every_tree_is_one_launch_in_the_original_visit_order_without_dedup(self):
        self.run_stage()
        want = lambda names: [(self.leaves[n].data_ptr(), 4 * self.leaves[n].numel()) for n in sorted(names, key=repr)]
        names = list(SMALL)
        frozen_probe, frozen_mlp = [n for n in names if n != 'head.probe'], [n for n in names if n not in ROLE_MLP]
        reads = [r for r in self.native.reads if len(r) in (11, 12, 8) or r == want([self.inventory.largest])]
        for tree in (frozen_probe, names, frozen_mlp, names, [self.inventory.largest]):
            self.assertIn(want(tree), reads)
        self.assertFalse([r for r in self.native.reads if len(r) in (19, 20, 23, 24)], 'frozen and full were amalgamated')
        self.assertTrue(all(len({a for a, _ in r}) == len(r) for r in reads if len(r) > 1), 'the inventory has no aliased leaf')

    def test_the_448_name_inventory_runs_the_exact_sequential_groups(self):
        real = REAL_BUILD(self.admitted.record)
        names = list(real.shapes)
        self.assertEqual(len(names), 448)
        self.inventory = types.SimpleNamespace(
            shapes={n: (3, 4000) if n == real.largest else (1 + i % 3,) for i, n in enumerate(names)}, roles=real.roles, largest=real.largest)
        receipt = self.run_stage()
        self.assertEqual({k: v['launches'] for k, v in receipt['checks']['workloads'].items()},
                         {'probe': [447, 448], 'mlp': [444, 448], 'largest': [1]})
        self.assertEqual(receipt['checks']['inventory']['leaves'], 448)
        self.assertEqual(len({leaf.data_ptr() for leaf in self.leaves.values()}), 448)
        self.assertEqual(sorted(len(r) for r in set(map(tuple, self.native.reads)) if len(r) > 7), [444, 447, 448])

    def arm(self, name):
        torch = self.torch
        if name in ('wrong_frozen', 'wrong_full', 'wrong_last'):
            class Bad(Native):
                def __call__(inner, *args):
                    status = Native.__call__(inner, *args)
                    if args[2] == {'wrong_frozen': 11, 'wrong_full': 12, 'wrong_last': 12}[name]:
                        first = 0 if name != 'wrong_last' else 32 * (args[2] - 1)
                        self.torch.device_arena.write(args[3] + first, b'\x01' * 32)
                    return status
            self.native = Bad(torch)
        elif name == 'stale_tree_cache':
            real, cache = mod.native_fingerprint, {}

            def stale(torch_, hasher, original, value):
                key = tuple((t.data_ptr(), 4 * t.numel()) for t in mod.collect_occurrences(torch_, value))
                if key not in cache:
                    cache[key] = real(torch_, hasher, original, value)
                return cache[key]
            self.stack.enter_context(mock.patch.object(mod, 'native_fingerprint', stale))
        elif name == 'reused_leaf_digests':
            real, known = mod.native_fingerprint, {}

            def reuse(torch_, hasher, original, value):
                tensors = mod.collect_occurrences(torch_, value)
                keys = [mod.occurrence_key(t) for t in tensors]
                fresh = [t for t in tensors if t.data_ptr() not in known]
                known.update({t.data_ptr(): d for t, d in zip(fresh, hasher.digests(fresh), strict=True)})
                cursor = mod.OccurrenceCursor(keys, [(k[2], k[3], known[k[0]]) for k in keys])
                result = original(value, frozen=cursor)
                cursor.finish()
                return result
            self.stack.enter_context(mock.patch.object(mod, 'native_fingerprint', reuse))
        elif name == 'pointer_only_dedup':
            class Dedup(Native):
                def __call__(inner, ptrs, lens, count, out, stream):
                    status = Native.__call__(inner, ptrs, lens, count, out, stream)
                    seen, arena = {}, self.torch.device_arena
                    for i in range(count):
                        address, = struct.unpack('<q', arena.read(ptrs + 8 * i, 8))
                        digest = arena.read(out + 32 * i, 32)
                        if address in seen:
                            arena.write(out + 32 * i, seen[address])
                        elif address:
                            seen[address] = digest
                    return status
            self.native = Dedup(torch)
        elif name == 'wrong_stream_handle':
            class Wrong(Native):
                def __call__(inner, ptrs, lens, count, out, stream):
                    return Native.__call__(inner, ptrs, lens, count, out, 0)
            self.native = Wrong(torch)
        elif name == 'no_wait_event':
            self.stack.enter_context(mock.patch.object(Stream, 'wait_event', lambda self, event: None))
        elif name == 'not_pending':
            torch.cuda._sleep = lambda cycles: None
        elif name == 'poisoned_drain':
            syncs = []
            torch.hooks['sync'] = lambda stream: (syncs.append(1), len(syncs) > 12 and (_ for _ in ()).throw(RuntimeError('drain')))
        elif name == 'leaky_owner':
            leaked, real = [], mod.Sha256Native

            class Leaky(real):
                def digests(self, tensors):
                    leaked.append(list(tensors))
                    return real.digests(self, tensors)
            self.stack.enter_context(mock.patch.object(mod, 'Sha256Native', Leaky))
        elif name == 'restore_bumps_version':
            real, seen = mod.flip, []

            def flip(torch_, leaf, byte, mask):
                real(torch_, leaf, byte, mask)
                seen.append(1)
                if len(seen) == 2:
                    leaf.counter[0] += 1
            self.stack.enter_context(mock.patch.object(mod, 'flip', flip))
        elif name == 'restore_skipped':
            real, seen = mod.flip, []

            def flip(torch_, leaf, byte, mask):
                seen.append(1)
                if len(seen) != 2:
                    real(torch_, leaf, byte, mask)
            self.stack.enter_context(mock.patch.object(mod, 'flip', flip))
        elif name == 'wrong_device':
            torch.cuda.get_device_properties = lambda index: types.SimpleNamespace(name='Other', major=12, minor=1)
        elif name in ('aliased_leaves', 'wrong_shape', 'wrong_dtype', 'host_leaf'):
            real, made = mod.allocate_leaf, {}

            def allocate(torch_, device, generator, leaf_name, shape):
                leaf = real(torch_, device, generator, leaf_name, shape)
                if leaf_name == 'post_layernorm.weight':
                    made[leaf_name] = leaf
                if leaf_name != 'post_layernorm.bias':
                    return leaf
                return {'aliased_leaves': made['post_layernorm.weight'], 'wrong_shape': torch_.make(device, (shape[0] + 1,), F32),
                        'wrong_dtype': torch_.make(device, shape, F64), 'host_leaf': torch_.zeros(*shape)}[name]
            self.stack.enter_context(mock.patch.object(mod, 'allocate_leaf', allocate))
        else:
            raise AssertionError(name)

    def test_each_defect_fails_closed_and_publishes_nothing(self):
        before = self.fds()
        for name in ('wrong_frozen', 'wrong_full', 'wrong_last', 'stale_tree_cache', 'reused_leaf_digests', 'pointer_only_dedup',
                     'wrong_stream_handle', 'no_wait_event', 'not_pending', 'poisoned_drain', 'leaky_owner', 'restore_bumps_version', 'restore_skipped',
                     'wrong_device', 'aliased_leaves', 'wrong_shape', 'wrong_dtype', 'host_leaf'):
            with self.subTest(name):
                self.reset()
                self.arm(name)
                with self.assertRaises((ValueError, RuntimeError)):
                    self.run_stage()
                self.assertFalse(os.path.lexists(self.output))
                self.assertEqual(self.fds(), before)


    # ---- every guard and the exit rehash the extension FILEs; the exit also re-reads the whole prior chain
    def drift_bytes(self, name):
        path = self.fixture.dir / name
        original = path.read_bytes()
        return path, original, bytes([original[0] ^ 1]) + original[1:]

    def test_extension_file_drift_is_caught_by_a_body_guard(self):
        self.run_stage()
        total = self.counts['source']
        self.assertGreater(total, 30)
        for name in ('full-fixture-inventory.json', 'extract_siglip2_vision_source.py', 'so400-native256-upstream-metadata-v1.json'):
            for number in (2, total - 5):
                with self.subTest(name, guard=number):
                    self.reset()
                    path, original, changed = self.drift_bytes(name)
                    self.actions[('source', number)] = lambda path=path, changed=changed: path.write_bytes(changed)
                    try:
                        with self.assertRaisesRegex(ValueError, 'differs'):
                            self.run_stage()
                    finally:
                        path.write_bytes(original)
                    self.assertFalse(os.path.lexists(self.output))

    def test_exit_rehashes_the_extension_and_rereads_the_whole_prior_chain(self):
        self.run_stage()
        before = self.fds()
        for name in ('full-fixture-inventory.json', 'extract_siglip2_vision_source.py', 'so400-native256-upstream-metadata-v1.json',
                     'smoke-out/smoke-receipt.json', 'original-smoke.log', 'prior_driver.py'):
            with self.subTest(name):
                self.reset()
                path, original, changed = self.drift_bytes(name)
                self.on_exit_drain(lambda path=path, changed=changed: path.write_bytes(changed))
                try:
                    with self.assertRaisesRegex(ValueError, 'differs'):
                        self.run_stage()
                finally:
                    path.write_bytes(original)
                self.assertFalse(os.path.lexists(self.output))
                self.assertEqual(self.fds(), before)

    def test_a_semantic_change_of_the_prior_chain_at_exit_is_not_accepted(self):
        def forge():  # a consistently re-hashed forgery is still caught: the chain is re-read and must equal the admitted facts
            real = mod.read_prior
            self.stack.enter_context(mock.patch.object(mod, 'UNIT_READER', lambda *a: {**real(*a), 'invocation_id': '0' * 32}))
        self.on_exit_drain(forge)
        with self.assertRaisesRegex(ValueError, 'prior smoke unit chain differs at exit'):
            self.run_stage()
        self.assertFalse(os.path.lexists(self.output))

    def test_the_exit_runs_the_same_independent_checks_when_the_body_failed(self):
        self.arm('wrong_full')
        with self.spied_closure():
            error = self.held_error()
        self.assertIn('ORIGINAL', str(error))
        self.assertEqual(getattr(error, '__notes__', []), [])
        self.assertEqual(self.events[-3:], ['closure', 'source', 'locks'])

    # ---- caps
    def test_body_cap_cuda_cap_and_locks_are_enforced_inside_the_full_body(self):
        clock = self.timed()
        self.on_call(40, lambda: setattr(clock, 'now', clock.now + 301))
        with self.assertRaisesRegex(ValueError, 'body deadline'):
            self.run_stage()
        self.reset()
        self.on_call(40, lambda: setattr(self.torch, 'peak', 10**10))
        with self.assertRaisesRegex(ValueError, 'resource cap'):
            self.run_stage()
        self.reset()
        self.drifts = {('locks', 20)}
        with self.assertRaisesRegex(ValueError, 'injected locks drift'):
            self.run_stage()
        self.assertFalse(os.path.lexists(self.output))

    def test_the_whole_clock_leaves_the_exit_reserve_before_the_exit(self):
        clock = self.timed(admission=1195)
        self.on_call(40, lambda: setattr(clock, 'now', clock.now + 10))
        with self.assertRaisesRegex(ValueError, 'headroom'):
            self.run_stage()
        self.assertFalse(os.path.lexists(self.output))

    # ---- lifetimes
    def test_failure_releases_every_device_owner_after_a_good_drain_and_keeps_them_after_a_bad_one(self):
        self.arm('wrong_full')
        error = self.held_error()
        self.leaves.clear()
        gc.collect()
        self.assertEqual([ref for ref in self.torch.made if ref() is not None], [], 'error frames pinned device owners')
        self.assertEqual(self.frame_locals(error, 'full_checks'), {})
        self.reset()
        self.arm('wrong_full')
        self.torch.cuda.synchronize = lambda: (_ for _ in ()).throw(RuntimeError('final drain'))
        error = self.held_error()
        self.assertTrue(any('final drain' in note for note in error.__notes__), error.__notes__)
        self.assertIn('leaves', self.frame_locals(error, 'full_checks'))

    def test_a_quarantined_hasher_keeps_its_owners_in_the_full_stage(self):
        self.arm('poisoned_drain')
        error = self.held_error()
        self.assertIsInstance(error, (ValueError, RuntimeError))
        gc.collect()
        self.assertTrue([ref for ref in self.torch.made if ref() is not None], 'quarantined owners were released')

    def test_the_library_descriptor_is_closed_and_the_receipt_is_exclusive(self):
        before = self.fds()
        self.run_stage()
        self.assertEqual(self.fds(), before)
        self.reset()
        self.output.mkdir()
        with self.assertRaises(FileExistsError):
            self.run_stage()

    def test_no_global_barrier_sits_between_the_table_copies_and_the_readback_of_a_candidate_call(self):
        self.run_stage()
        log = self.torch.log
        complete = [i for i, e in enumerate(log) if e[0] == 'launch' and log[i + 1][0] == 'sync' and log[i + 2][0] == 'cpu']
        self.assertGreater(len(complete), 20)
        self.assertIn(('global_sync',), log)  # the arm and exit drains exist, but never inside a candidate call
        for i in complete:
            start = [j for j in range(i) if log[j][0] == 'pin'][-2]  # two pinned tables per call
            self.assertNotIn(('global_sync',), log[start:i + 3])
            self.assertEqual([e[0] for e in log[start:i + 3] if e[0] in ('pin', 'copy', 'launch', 'sync', 'cpu')],
                             ['pin', 'copy', 'pin', 'copy', 'launch', 'sync', 'cpu'])


class FullCliTests(AuthorityBase):
    own = False  # the driver FILE must be this very script, as for the smoke CLI tests

    def setUp(self):
        super().setUp()
        self.authority = self.fixture.render_full()
        self.output = str(self.fixture.dir / 'cli-out')
        self.recorder = []

    @property
    def body(self):  # a plain closure: requests.Source deep-copies the STAGE_BODIES literal
        recorder = self.recorder

        def body(context, output):
            context.source.check()
            context.locks.check()
            recorder.append((context, output))
            return 'ran'
        return body

    def run_main(self, argv, bodies=None):
        with mock.patch.dict(mod.STAGE_BODIES, bodies or {'smoke': self.body, 'full': self.body}, clear=True):
            return mod.main(argv)

    def test_full_admits_the_authority_then_the_genuine_unit_then_runs_the_body_once(self):
        self.assertEqual(self.run_main(mod.cli('full', self.authority, self.fixture.unit, self.output)), 'ran')
        (context, output), = self.recorder
        self.assertEqual((output, context.unit, context.fact), (self.output, self.fixture.unit, self.authority))
        self.assertEqual(context.prior['invocation_id'], self.fixture.INVOCATION)
        self.assertEqual(context.authority.record['schema'], mod.NATIVE_FULL)
        self.assertFalse(NATIVE & {name.split('.')[0] for name in sys.modules})

    def test_rejections_never_reach_the_full_body(self):
        go = self.fixture.json_file('go.json', {'schema': 'cuda-sha256-native-unit-v1', 'stage': 'smoke', 'decision': 'GO',
                                                'authority': self.authority})
        smoke = self.fixture.render()
        cases = {'user_go_unit': mod.cli('full', self.authority, go, self.output),
                 'unfrozen_unit': mod.cli('full', self.authority, self.fixture.static['log'], self.output),
                 'smoke_authority_for_full': mod.cli('full', smoke, self.fixture.unit, self.output),
                 'full_authority_for_smoke': mod.cli('smoke', self.authority, None, self.output),
                 'timing': mod.cli('timing', self.authority, self.fixture.unit, self.output)}
        for name, argv in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                self.run_main(argv, {'smoke': self.body, 'full': self.body})
        self.assertEqual(self.recorder, [])
        self.assertEqual(set(mod.STAGE_BODIES), {'smoke', 'full'})
        with self.assertRaisesRegex(ValueError, 'unreleased scaffolding'):
            self.run_main(mod.cli('timing', self.authority, self.fixture.unit, self.output), {'smoke': self.body, 'full': self.body})

    def test_a_changed_chain_file_never_reaches_the_body(self):
        path = self.fixture.dir / 'original-smoke.log'
        path.write_bytes(path.read_bytes()[:-1])
        with self.assertRaises(ValueError):
            self.run_main(mod.cli('full', self.authority, self.fixture.unit, self.output))
        self.assertEqual(self.recorder, [])


class RealSizeLeafTests(unittest.TestCase):
    """The real 19,832,832-byte [4304, 1152] leaf through the fake native: sizes, offsets, no tolist, no 1.7 GiB inventory."""

    def setUp(self):
        self.torch = Torch(device_bytes=160 << 20)
        self.device = self.torch.device('cuda', 0)
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(fake_modules(self.torch))
        real = Tensor.tolist

        def tolist(tensor):
            if tensor.numel() > 4096:
                raise AssertionError('a large tolist/bytes-copy oracle is forbidden')
            return real(tensor)
        stack.enter_context(mock.patch.object(Tensor, 'tolist', tolist))
        self.native = Native(self.torch)
        self.hasher = mod.Sha256Native(self.torch, self.native)

    def test_largest_leaf_parity_mutation_and_typed_digest_use_the_streamed_oracle(self):
        leaf = mod.allocate_leaf(self.torch, self.device, self.torch.Generator(self.device), 'encoder.layers.0.mlp.fc1.weight', (4304, 1152))
        self.assertEqual(4 * leaf.numel(), 19832832)
        self.assertEqual(mod.plan_occurrences(self.torch, [leaf], self.device)[0][1], 19832832)
        want = mod.host_digest(torch=self.torch, leaf=leaf)
        self.assertEqual(want, sha(leaf.logical()))
        self.assertEqual(self.hasher.digests([leaf]), [want])
        flat, version = leaf.reshape(-1), leaf._version
        mod.flip(self.torch, flat, 19832831, 0x40)
        changed = mod.host_digest(self.torch, leaf)
        self.assertEqual((leaf._version, changed != want), (version, True))
        self.assertEqual(self.hasher.digests([leaf]), [changed])
        mod.flip(self.torch, flat, 19832831, 0x40)
        self.assertEqual(self.hasher.digests([leaf]), [want])
        original = original_fingerprint(SERIALIZERS['probe'])[0]
        tree = {'encoder.layers.0.mlp.fc1.weight': leaf}
        self.assertEqual(mod.native_fingerprint(self.torch, self.hasher, original, tree), original(tree))
        self.assertEqual(self.native.reads[-1], [(leaf.data_ptr(), 19832832)])

    def test_every_largest_leaf_gate_runs_at_the_real_size(self):
        name, shape = 'encoder.layers.0.mlp.fc1.weight', (4304, 1152)
        generator = self.torch.Generator(self.device)
        big = mod.allocate_leaf(self.torch, self.device, generator, name, shape)
        fingerprints = [original_fingerprint(path)[0] for path in SERIALIZERS.values()]
        calls = []

        def counted(*args):
            calls.append(args)
            return self.native(*args)
        results = mod.largest_gates(self.torch, self.device, mod.Sha256Native(self.torch, counted), self.native, fingerprints,
                                    lambda: None, [], calls, big, name,
                                    lambda: mod.allocate_leaf(self.torch, self.device, generator, name, shape))
        self.assertEqual(results['parity']['bytes'], 19832832)
        self.assertEqual(results['mutation']['bytes'], [0, 9916417, 19832831])
        self.assertEqual(sorted(results['injected_failures']), ['completed', 'launched', 'readback', 'status'])


class SourceShapeTests(unittest.TestCase):
    def functions(self, *names):
        tree = ast.parse(SCRIPT.read_text())
        found = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
        return [found[name] for name in names]

    def test_the_full_oracle_path_has_no_tolist_or_bytes_copy(self):
        for function in self.functions('host_digest', 'largest_gates', 'full_checks', 'timed_arm', 'allocate_leaf', 'check_leaves', 'build_inventory'):
            for node in ast.walk(function):
                if isinstance(node, ast.Attribute):
                    self.assertNotIn(node.attr, ('tolist', 'tobytes', 'to_bytes'), function.name)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, ('bytes', 'bytearray'), function.name)

    def test_comparator_is_the_original_fingerprint_never_the_rejected_helper(self):
        text = SCRIPT.read_text()
        self.assertNotIn('_fingerprint_cuda_dict', text.replace('never `_fingerprint_cuda_dict`', ''))
        self.assertIn("original_fingerprint(value, frozen=cursor)", text)
        self.assertEqual(set(mod.STAGE_BODIES), {'smoke', 'full'})
        self.assertIsNone(getattr(mod, 'timing', None))

    def test_released_stage_flow_uses_only_the_shared_lifecycle(self):
        calls = {n.func.id for f in self.functions('smoke', 'full') for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertEqual(calls, {'run_stage'})
        body = {n.func.id for f in self.functions('smoke_body', 'full_body') for n in ast.walk(f) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        self.assertLessEqual({'open_stage', 'smoke_checks', 'full_checks'}, body)


class SmokeSourceInverseTests(unittest.TestCase):
    """What the passed native smoke-v2 ran is AST-identical to the frozen smoke-v2 driver except an explicit, finite list."""
    FROZEN = FREEZE / 'qualify_cuda_sha256_native.py'
    CHANGED = {'STAGE_BODIES', 'UNIT_READER', 'exit_pass', 'main', 'parser', 'read_json', 'read_native_authority', 'smoke', 'smoke_body',
               'smoke_checks'}

    @staticmethod
    def definitions(tree):
        found = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                found[node.name] = ast.dump(node)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    for name in ([target.id] if isinstance(target, ast.Name) else [e.id for e in getattr(target, 'elts', ())]):
                        found[name] = ast.dump(node)
        return found

    def setUp(self):
        raw = self.FROZEN.read_bytes()
        self.assertEqual(sha(raw), '4786116278231bf17c36dd83dc4ac769df032f27678957a63297b3ca88399174')
        self.old, self.new = ast.parse(raw), ast.parse(SCRIPT.read_text())

    def test_only_the_listed_smoke_definitions_changed_and_none_was_removed(self):
        old, new = self.definitions(self.old), self.definitions(self.new)
        self.assertEqual(sorted(set(old) - set(new)), [])
        self.assertEqual({name for name in old if old[name] != new[name]}, self.CHANGED)

    @staticmethod
    def injected_span(body):
        names = lambda n: isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple) and [e.id for e in n.targets[0].elts] == ['injected', 'kept']
        first = next(i for i, n in enumerate(body) if names(n))
        last = next(i for i, n in enumerate(body) if ast.unparse(n) == "results['injected_failures'] = injected")
        return first, last

    def test_smoke_checks_differ_only_by_the_extracted_injected_failure_block(self):
        old = next(n for n in self.old.body if isinstance(n, ast.FunctionDef) and n.name == 'smoke_checks').body
        new = next(n for n in self.new.body if isinstance(n, ast.FunctionDef) and n.name == 'smoke_checks').body
        first, last = self.injected_span(old)
        self.assertEqual([ast.dump(n) for n in new[:first]], [ast.dump(n) for n in old[:first]])
        self.assertEqual([ast.dump(n) for n in new[first + 1:]], [ast.dump(n) for n in old[last + 1:]])
        self.assertIn("results['injected_failures'] = injected_failures(", ast.unparse(new[first]))

    def test_the_extracted_injected_failure_loop_is_the_released_loop_with_only_its_inputs_parameterised(self):
        old = next(n for n in self.old.body if isinstance(n, ast.FunctionDef) and n.name == 'smoke_checks').body
        helper = next(n for n in self.new.body if isinstance(n, ast.FunctionDef) and n.name == 'injected_failures').body
        first, last = self.injected_span(old)
        released = ast.Module(body=old[first:last], type_ignores=[])  # the assignments, the loop and `del kept`

        class Parameterise(ast.NodeTransformer):
            def visit_Assign(self, node):
                if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'owners':
                    return ast.Assign(targets=node.targets, value=ast.Call(func=ast.Name('make_owners', ast.Load()), args=[], keywords=[]))
                return self.generic_visit(node)

            def visit_Subscript(self, node):
                name = node.value.id if isinstance(node.value, ast.Name) else None
                if name in ('tensors', 'expected') and ast.unparse(node.slice) == ':3':
                    return ast.Name('follow' if name == 'tensors' else 'want', ast.Load())
                return self.generic_visit(node)
        released = Parameterise().visit(released)
        ast.fix_missing_locations(released)
        shared = ast.Module(body=[n for n in helper[1:-1]], type_ignores=[])  # without the docstring and the return
        self.assertEqual(ast.dump(released), ast.dump(shared))

    def test_smoke_still_has_exactly_the_released_receipt_shape_and_file(self):
        self.assertIn("(stage + '-receipt.json')", SCRIPT.read_text())
        self.assertEqual(mod.SMOKE_RECEIPT_KEYS, {'schema', 'status', 'engineering_only', 'authority', 'library', 'device', 'checks', 'native_calls',
                                                  'resources', 'resource_policy', 'mappings', *mod.FLAGS, 'normal_terminal_required', 'invocation'})
        self.assertEqual(mod.SMOKE_CHECKS['parity']['lengths'], [4 * n for n in mod.LENGTHS])


if __name__ == '__main__':
    unittest.main(verbosity=1)
