#!/usr/bin/env python3
"""Standalone CUDA SHA-256 native caller + frozen smoke entrypoint; UNRUN, UNQUALIFIED.

SOURCE ONLY. Nothing here was compiled, loaded, launched or timed; no Torch/NumPy/GPU/SSH
work happens at import or in the tests. Native execution is root-only. No speed, quality,
SOTA or product claim follows from this file. It calls rust/sfora-cuda-sha256/
sha256_occurrences.cu unchanged; serving, defaults, helpers, math and the original
serializers are untouched. full and timing are UNRELEASED SCAFFOLDING, not qualifiers.

C ABI (exactly five arguments; cudaError_t is the C enum int):
  sfora_sha256_occurrences(const uchar* const* ptrs, const uint64* lens, uint64 count,
                           uchar* out, cudaStream_t current_stream) -> int
  ctypes: argtypes=(c_void_p,c_void_p,c_uint64,c_void_p,c_void_p), restype=c_int.
  Raw count-zero, count=2**32 and null-top-level-nonzero ABI calls are ROOT smoke checks
  made on the bound function directly; digests([]) only skips the call as a convenience.

API (torch/ctypes imports are lazy; module load is stdlib plus the stdlib-only
qualify_connected_serving_requests):
  bind_abi(lib, ctypes)                    bind and return the one function
  plan_occurrences(torch, tensors, device) -> [(address, nbytes)]; PURE HOST validation before
        anything is enqueued. Leaf: CUDA torch.Tensor on the active device, float32, strided,
        is_contiguous, nbytes%4==0, nbytes<=(2**64-1)//8, scalar=4 bytes, empty=0 bytes
        (address 0), nonempty needs a nonzero 4-aligned address inside its storage (offset
        views allowed). No gather/cast/copy/fallback, no pointer dedup, no cache.
  Sha256Native(torch, function, fault=None).digests(tensors) -> [64 lowercase hex], one per
        occurrence in order; duplicates and aliases are hashed afresh. fault(point) is an
        injection seam used only by the smoke body (points launched/completed/readback).
  collect_occurrences(torch, value)        tensors in the ORIGINAL fingerprint() visit order
  OccurrenceCursor(keys, facts)            single-use per-call frozen= object: .get(key) must
        equal the next occurrence key (data_ptr, _version, dtype str, shape tuple) and returns
        (dtype str, shape tuple, hex); extra/missing/reordered/mutated/reused all raise; never
        indexed by pointer, never retained (close() drops it, also on failure).
  native_fingerprint(torch, hasher, original_fingerprint, value) = collect -> digests (checked
        completion + readback) -> original_fingerprint(value, frozen=cursor) ->
        cursor.finish(). The ORIGINAL serializer frames every container/type/key/dtype/shape.

Lifecycle of one digests() call. Phase A (no device effect): list, count<=2**32-1, every leaf,
the current CUDA stream (handle 0 only when it is the default stream), busy and poison flags.
Phase B (from the first table copy): int64 pointer/length tables (pinned host staging, then a
NON-BLOCKING copy, so stream order and not a host barrier orders producers) and a separately
allocated uint8 output of 32*count are enqueued on the current stream; capacity, dtype,
contiguity, device and output-vs-input/table non-overlap are checked; the stream is re-read
immediately before and after the one native call; a nonzero status is an error. Completion =
stream.synchronize() (stream scoped, never a global barrier), then readback, then a length
check; only then are digests produced. The finally block drains (stream.synchronize()) whenever
Phase B began and readback was not checked. A drain failure quarantines every owner (inputs,
tables, output) in Sha256Native.quarantine, sets poisoned and refuses all later calls; owners
and error-frame locals are released only after a successful drain. No digest is returned on
any launch, completion or readback failure. One call at a time (re-entry rejected).

FILE={path:canonical absolute regular file,sha256:actual 64 hex}. JSON FILEs are read through a
separate 4MiB bound. Every binary/text FILE (compiler, nvcc, library, cuDNN/runtime files,
source, logs) gets a fresh STREAMED sha256 over an opened O_NOFOLLOW regular fd with NO size cap.
requests.Source keeps its own 2MiB source bound. Roots supply the actual hashes; none is
defaulted or guessed and no directory is ever granted. Strict JSON, no unknown/duplicate keys.
 cuda-sha256-build-authority-v1 (PROSPECTIVE, frozen BEFORE compilation, no product keys):
  schema, source (FILE named sha256_occurrences.cu), contract (FILE), toolchain (list of
  {role,file}; roles unique, nvcc and host_compiler required), compile_script (FILE), target
  (sm_NN[a]), flags (list; exactly one -gencode=arch=compute_NN,code=sm_NN for that target and
  no -arch/-code/--gpu-*/-ptx: SASS only, no PTX), environment ({NAME:str}), output_dir
  (canonical absolute path string, not created here), evidence (nonempty FILE list).
 cuda-sha256-build-receipt-v1 (produced by the root's build): schema, build_authority (FILE),
  exit_status (0), library (FILE), sass_targets ([target]), ptx_images (0), logs (FILE list).
 cuda-sha256-native-authority-v1 (EXACT produced binary/runtime closure):
  schema, sources ({driver,test,probe_serializer,mlp_serializer}:FILE; driver is this running
  module), build_authority, build_receipt, library (equal to the receipt's), runtime_files
  (nonempty list of {file,provenance}; provenance=cuda-sha256-native-file-provenance-v1
  {schema,file,origin,evidence[FILE]} with evidence independent of file/provenance),
  mapping_inventory (FILE cuda-sha256-mapping-inventory-v1 {schema,files:{canonical path:
  {sha256,size}}}: the explicit complete permitted '.so' mapping set, containing library and
  every runtime_files FILE; every key must be an existing canonical regular file, so no
  directory or glob entry exists), interpreter (FILE equal to the running
  executable), device ({index,name,capability:[major,minor]}, capability matching the target),
  resource_policy (EXACTLY body_seconds 300, whole_process_seconds 1500, exit_reserve_seconds
  300, host_bytes 8GiB, swap_bytes 0, cuda_allocated_bytes_exclusive 10_000_000_000), locks (the
  two original inherited exclusive lifetime {path,fd}, verified by requests.Locks).
 open_library hashes an OPENED regular-file descriptor, dlopens that same inode through
 /proc/self/fd, and requires the maps entry to be that canonical undeleted inode, every
 file-backed .so mapping to be inventoried with its frozen size and live device/inode, and only
 declared files to appear. Declared FILEs are freshly hashed; other inventoried mappings are
 authenticated by size+device/inode only (root decision). The set is re-read after the run.
 UNIT: a user-written {decision:GO,authority} is NEVER accepted. UNIT_READER stays None until a
 genuine root-owned original unit/footer/locks/resources/exit/source reader exists, so --unit and
 therefore full/timing fail closed at admission.

CLI (argv order is canonical and exact; invoke by the canonical absolute script path):
  qualify_cuda_sha256_native.py STAGE --authority FILE.json --authority-sha256 SHA
      [--unit FILE.json --unit-sha256 SHA] --output NEWDIR            STAGE in smoke|full|timing
 --unit is forbidden for smoke and required for full/timing. NEWDIR is a canonical absolute
 path whose parent exists and which does not exist. smoke admits (authority, sources, FILEs,
 interpreter, locks, resources policy) and then runs the frozen smoke body; STAGE_BODIES has no
 full/timing entry, so those stages fail closed. The smoke body checks: raw ABI nulls/oversize,
 hashlib parity for byte lengths 0,4,52-68,116-132 with scalar/empty/offset/duplicate/overlap
 over 129 occurrences, typed-tree equality with the ORIGINAL fingerprint() extracted by AST
 from the authenticated probe and MLP serializer FILEs, occurrence order, same-version .data
 mutation/restore, a nondefault current stream with a mutation still pending hashed before any
 oracle copy, labeled InjectedFault launch/completion/readback failures with real work
 outstanding, drain, released owners under a retained error and a healthy next call, and
 pre-enqueue rejections with zero native calls. It writes one diagnostic receipt (all
 eligibility flags false) only after every check passes. The 447+448/444+448 inventories,
 timing and every qualification gate are separate root releases.

UNVERIFIED NATIVE ASSUMPTIONS (the tests use a fake torch/ABI and cannot check them): Tensor.data
carries a fresh version counter; torch.cuda._sleep plus Event.query prove a mutation is still
pending (the check fails closed if it is not); pinned non_blocking copies and the caching
allocator never insert a host barrier; the CUDA runtime/driver mappings fit the explicit
inventory. Not integrated: encoder_facts callers, serving, extraction ledgers, bridge provenance.
"""
import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time
from types import SimpleNamespace
import weakref

import qualify_connected_serving_requests as requests

require = requests.require
ABI_SYMBOL = 'sfora_sha256_occurrences'
MAX_COUNT = 2**32 - 1
MAX_LEAF_BYTES = (2**64 - 1) // 8
JSON_LIMIT = 4 * 1024**2
STAGES = ('smoke', 'full', 'timing')
PRIOR = {'full': 'smoke', 'timing': 'full'}
POLICY = {'body_seconds': 300, 'whole_process_seconds': 1500, 'exit_reserve_seconds': 300,
          'host_bytes': 8 * 1024**3, 'swap_bytes': 0, 'cuda_allocated_bytes_exclusive': 10_000_000_000}
BUILD, RECEIPT = 'cuda-sha256-build-authority-v1', 'cuda-sha256-build-receipt-v1'
NATIVE, PROVENANCE = 'cuda-sha256-native-authority-v1', 'cuda-sha256-native-file-provenance-v1'
INVENTORY, SMOKE_RECEIPT = 'cuda-sha256-mapping-inventory-v1', 'cuda-sha256-native-smoke-v1'
BUILD_KEYS = {'schema', 'source', 'contract', 'toolchain', 'compile_script', 'target', 'flags',
              'environment', 'output_dir', 'evidence'}
RECEIPT_KEYS = {'schema', 'build_authority', 'exit_status', 'library', 'sass_targets', 'ptx_images', 'logs'}
SOURCE_KEYS = {'driver', 'test', 'probe_serializer', 'mlp_serializer'}
NATIVE_KEYS = {'schema', 'sources', 'build_authority', 'build_receipt', 'library', 'runtime_files',
               'mapping_inventory', 'interpreter', 'device', 'resource_policy', 'locks'}
ARCH_FLAGS = ('-arch', '--gpu-architecture', '-code', '--gpu-code', '-ptx', '--ptx', '-gencode',
              '--generate-code')
HEX = re.compile('[0-9a-f]{64}')
FLAGS = ('quality_read', 'quality_eligible', 'qualification_eligible', 'state_reuse_eligible',
         'optimization_eligible', 'product_go', 'speed_go')
STARTED = time.perf_counter()
UNIT_READER = None  # the root wires the genuine terminal/provenance reader here; a GO file is never enough


class InjectedFault(RuntimeError):
    """A labeled test failure injected by the smoke body; never a real CUDA error."""


# ---------------------------------------------------------------- FILE authentication
def canonical(path):
    require(type(path) is str, 'canonical absolute path required')
    value = Path(path)
    require(value.is_absolute() and str(value) == path and value.resolve() == value and
            not value.is_symlink(), 'canonical absolute path required')
    return value


def open_regular(fact):
    require(type(fact) is dict and fact.keys() == {'path', 'sha256'} and type(fact['sha256']) is str and
            HEX.fullmatch(fact['sha256']), 'actual FILE required')
    fd = os.open(canonical(fact['path']), os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode):
        os.close(fd)
        raise ValueError('regular FILE required')
    return fd, info


def stream_hash(fd):
    digest, offset = hashlib.sha256(), 0
    while True:
        block = os.pread(fd, 1 << 20, offset)
        if not block:
            return digest.hexdigest(), offset
        digest.update(block)
        offset += len(block)


def hash_file(fact):
    """Fresh streamed hash with no size cap; returns the opened file's stat."""
    fd, info = open_regular(fact)
    try:
        digest, size = stream_hash(fd)
        after = os.fstat(fd)
        require(digest == fact['sha256'] and size == info.st_size == after.st_size and
                after.st_mtime_ns == info.st_mtime_ns, 'current FILE size/SHA256 differs')
        return info
    finally:
        os.close(fd)


def read_bytes(fact, limit):
    fd, info = open_regular(fact)
    try:
        require(info.st_size <= limit, 'bounded FILE exceeds its limit')
        raw, offset = [], 0
        while offset <= limit:
            block = os.pread(fd, min(1 << 20, limit + 1 - offset), offset)
            if not block:
                break
            raw.append(block)
            offset += len(block)
        raw = b''.join(raw)
        require(len(raw) <= limit and hashlib.sha256(raw).hexdigest() == fact['sha256'],
                'current FILE size/SHA256 differs')
        return raw
    finally:
        os.close(fd)


def read_json(fact, schema, keys):
    record = requests.strict_json(read_bytes(fact, JSON_LIMIT))
    require(type(record) is dict and record.keys() == keys and record['schema'] == schema,
            'exact %s required' % schema)
    return record


def file_list(rows):
    require(type(rows) is list and rows and len({r['path'] for r in rows if type(r) is dict and 'path' in r}) ==
            len(rows), 'nonempty distinct FILE list required')
    for fact in rows:
        hash_file(fact)


# ---------------------------------------------------------------- authority records
def read_build_authority(fact):
    record = read_json(fact, BUILD, BUILD_KEYS)
    for name in ('source', 'contract', 'compile_script'):
        hash_file(record[name])
    require(Path(record['source']['path']).name == 'sha256_occurrences.cu' and
            Path(record['contract']['path']).name == 'gpu_sha256_source_contract_2026-10-09.md',
            'the CUDA source and contract FILEs are required')
    tools = record['toolchain']
    require(type(tools) is list and all(type(r) is dict and r.keys() == {'role', 'file'} and
            type(r['role']) is str and re.fullmatch('[a-z0-9_.-]+', r['role']) for r in tools) and
            len({r['role'] for r in tools}) == len(tools) and
            {'nvcc', 'host_compiler'} <= {r['role'] for r in tools}, 'exact toolchain roles required')
    file_list([r['file'] for r in tools])
    target = record['target']
    require(type(target) is str and re.fullmatch('sm_[0-9]{2,3}[a-z]?', target), 'exact SM target required')
    flags = record['flags']
    require(type(flags) is list and all(type(f) is str and f and f == f.strip() for f in flags) and
            [f for f in flags if f.startswith(ARCH_FLAGS)] ==
            ['-gencode=arch=compute_%s,code=%s' % (target[3:], target)], 'SASS-only single gencode required')
    env = record['environment']
    require(type(env) is dict and all(type(k) is str and re.fullmatch('[A-Z_][A-Z0-9_]*', k) and
            type(v) is str for k, v in env.items()), 'exact build environment required')
    out = record['output_dir']
    require(type(out) is str and Path(out).is_absolute() and str(Path(out)) == out and
            '..' not in Path(out).parts, 'canonical build output path required')
    file_list(record['evidence'])
    return record


def read_build_receipt(fact, build_fact, build):
    receipt = read_json(fact, RECEIPT, RECEIPT_KEYS)
    require(receipt['build_authority'] == build_fact and type(receipt['exit_status']) is int and
            receipt['exit_status'] == 0 and receipt['sass_targets'] == [build['target']] and
            type(receipt['ptx_images']) is int and receipt['ptx_images'] == 0,
            'successful SASS-only build receipt for the frozen authority required')
    hash_file(receipt['library'])
    file_list(receipt['logs'])
    return receipt


def read_inventory(fact, declared):
    inventory = read_json(fact, INVENTORY, {'schema', 'files'})
    files = inventory['files']
    require(type(files) is dict and files and all(type(e) is dict and e.keys() == {'sha256', 'size'} and
            type(e['sha256']) is str and HEX.fullmatch(e['sha256']) and type(e['size']) is int and
            e['size'] > 0 and canonical(p).is_file() for p, e in files.items()), 'explicit mapping inventory required')
    for item in declared:
        require(files.get(item['path'], {}).get('sha256') == item['sha256'], 'declared FILE missing from inventory')
    return files


def read_native_authority(fact):
    record = read_json(fact, NATIVE, NATIVE_KEYS)
    sources = record['sources']
    require(type(sources) is dict and sources.keys() == SOURCE_KEYS, 'exact source FILE set required')
    for item in sources.values():
        hash_file(item)
    build = read_build_authority(record['build_authority'])
    receipt = read_build_receipt(record['build_receipt'], record['build_authority'], build)
    require(record['library'] == receipt['library'], 'receipt library differs')
    rows = record['runtime_files']
    require(type(rows) is list and rows, 'explicit runtime FILE provenance required')
    runtime = []
    for row in rows:
        require(type(row) is dict and row.keys() == {'file', 'provenance'}, 'exact runtime FILE row required')
        hash_file(row['file'])
        proof = read_json(row['provenance'], PROVENANCE, {'schema', 'file', 'origin', 'evidence'})
        require(proof['file'] == row['file'] and type(proof['origin']) is str and proof['origin'].strip(),
                'runtime FILE provenance required')
        file_list(proof['evidence'])
        require(not {row['file']['path'], row['provenance']['path']} & {e['path'] for e in proof['evidence']},
                'independent runtime FILE provenance required')
        runtime.append(row['file'])
    inventory = read_inventory(record['mapping_inventory'], [record['library'], *runtime])
    paths = [record['library'], record['build_authority'], record['build_receipt'], record['mapping_inventory'],
             *sources.values(), *runtime, *(r['provenance'] for r in rows)]
    require(len({p['path'] for p in paths}) == len(paths), 'distinct authority FILEs required')
    hash_file(record['interpreter'])
    require(Path(sys.executable).resolve() == Path(record['interpreter']['path']), 'running interpreter differs')
    device = record['device']
    require(type(device) is dict and device.keys() == {'index', 'name', 'capability'} and
            type(device['index']) is int and device['index'] >= 0 and type(device['name']) is str and
            device['name'] and type(device['capability']) is list and len(device['capability']) == 2 and
            all(type(v) is int and v >= 0 for v in device['capability']) and
            'sm_%d%d' % tuple(device['capability']) == re.sub('[a-z]$', '', build['target']),
            'device identity matching the build target required')
    policy = record['resource_policy']
    require(type(policy) is dict and policy == POLICY and all(type(v) is int for v in policy.values()),
            'exact body300/whole1500/exit-reserve300/8GiB/zero-swap/CUDA<1e10 policy required')
    return SimpleNamespace(record=record, build=build, receipt=receipt, inventory=inventory, runtime=runtime)


# ---------------------------------------------------------------- mapping + library
def mapped_files(inventory, maps='/proc/self/maps'):
    result = {}
    for line in Path(maps).read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6 or not fields[5].startswith('/') or '.so' not in fields[5]:
            continue
        raw = fields[5]
        require(not raw.endswith(' (deleted)'), 'deleted native mapping rejected')
        path = Path(raw)
        require(str(path) == raw and path.resolve() == path and not path.is_symlink(), 'noncanonical native mapping rejected')
        major, minor = (int(v, 16) for v in fields[3].split(':'))
        identity = os.makedev(major, minor), int(fields[4])
        entry = inventory.get(raw)
        require(entry is not None, 'native mapping outside the frozen inventory')
        info = path.stat()
        require((info.st_dev, info.st_ino) == identity and info.st_size == entry['size'],
                'native mapping inode/size differs from the frozen inventory')
        require(result.setdefault(raw, identity) == identity, 'conflicting native mappings')
    return result


def open_library(authority, cdll, maps='/proc/self/maps'):
    """Hash an opened descriptor, dlopen that same inode, verify the mapping set; keep fd to exit."""
    fact = authority.record['library']
    fd, info = open_regular(fact)
    try:
        digest, size = stream_hash(fd)
        require(digest == fact['sha256'] and size == info.st_size == authority.inventory[fact['path']]['size'],
                'library descriptor hash/size differs')
        before = mapped_files(authority.inventory, maps)
        lib = cdll('/proc/self/fd/%d' % fd)
        after = mapped_files(authority.inventory, maps)
        declared = {fact['path'], *(r['path'] for r in authority.runtime)}
        require(after.get(fact['path']) == (info.st_dev, info.st_ino), 'loaded library is not the authenticated inode')
        require(before.items() <= after.items() and after.keys() - before.keys() <= declared,
                'native mapping set changed beyond the declared FILEs')
        return fd, lib, after
    except BaseException:
        os.close(fd)
        raise


def bind_abi(lib, ctypes):
    try:
        function = getattr(lib, ABI_SYMBOL)
    except AttributeError:
        raise ValueError('native ABI symbol missing') from None
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_void_p]
    function.restype = ctypes.c_int
    return function


# ---------------------------------------------------------------- the caller
def plan_occurrences(torch, tensors, device):
    require(type(tensors) is list and len(tensors) <= MAX_COUNT, 'occurrence list within 2**32-1 required')
    spans = []
    for leaf in tensors:
        require(isinstance(leaf, torch.Tensor) and leaf.is_cuda and leaf.device == device,
                'CUDA tensor on the active device required')
        require(leaf.dtype is torch.float32 and leaf.layout is torch.strided and leaf.element_size() == 4 and
                leaf.is_contiguous(), 'contiguous strided float32 leaf required')
        size = leaf.numel() * 4
        require(type(size) is int and size % 4 == 0 and size <= MAX_LEAF_BYTES, 'leaf byte length differs')
        if size == 0:
            spans.append((0, 0))
            continue
        address, offset, storage = leaf.data_ptr(), leaf.storage_offset() * 4, leaf.untyped_storage()
        require(type(address) is int and 0 < address < 2**63 and address % 4 == 0 and
                address == storage.data_ptr() + offset and offset + size <= storage.nbytes(), 'invalid leaf span')
        spans.append((address, size))
    return spans


class Sha256Native:
    def __init__(self, torch, function, fault=None):
        self.torch, self.function, self.fault = torch, function, fault
        self.poisoned, self.quarantine, self.busy = False, [], False

    def table(self, values, device):
        host = self.torch.tensor(values, dtype=self.torch.int64).pin_memory()
        return host, host.to(device, non_blocking=True)

    def digests(self, tensors):
        torch = self.torch
        require(not self.poisoned, 'native context poisoned by a failed drain')
        require(not self.busy, 'native call is not reentrant')
        device = torch.device('cuda', torch.cuda.current_device())
        spans = plan_occurrences(torch, tensors, device)
        count = len(spans)
        if count == 0:
            return []
        stream = torch.cuda.current_stream(device)
        handle = stream.cuda_stream
        require(type(handle) is int and handle >= 0 and stream.device == device and
                (handle != 0 or stream == torch.cuda.default_stream(device)), 'current CUDA stream required')

        def current():
            now = torch.cuda.current_stream(device)
            return now.cuda_stream, now.device

        owners, started, completed, failure = list(tensors), False, False, None
        host = ptrs = lens = out = buffer = None
        self.busy = True
        try:
            started = True
            host, ptrs = self.table([a for a, _ in spans], device)
            owners.extend((host, ptrs))
            host, lens = self.table([n for _, n in spans], device)
            owners.extend((host, lens))
            out = torch.empty(32 * count, dtype=torch.uint8, device=device)
            owners.append(out)
            for buffer, need, dtype in ((ptrs, count, torch.int64), (lens, count, torch.int64), (out, 32 * count, torch.uint8)):
                require(buffer.is_cuda and buffer.device == device and buffer.dtype is dtype and
                        buffer.is_contiguous() and buffer.numel() >= need and buffer.data_ptr() % 8 == 0,
                        'native buffer layout differs')
            low, high = out.data_ptr(), out.data_ptr() + 32 * count
            require(all(n == 0 or a + n <= low or a >= high for a, n in spans), 'output overlaps an input leaf')
            require(all(b.data_ptr() + 8 * count <= low or b.data_ptr() >= high for b in (ptrs, lens)),
                    'output overlaps an input array')
            require(current() == (handle, device), 'current CUDA stream changed before launch')
            status = self.function(ptrs.data_ptr(), lens.data_ptr(), count, out.data_ptr(), handle)
            require(current() == (handle, device), 'current CUDA stream changed during launch')
            if self.fault is not None:
                self.fault('launched')
            require(type(status) is int and status == 0, 'native launch status %r' % (status,))
            stream.synchronize()
            if self.fault is not None:
                self.fault('completed')
            raw = bytes(out.cpu().tolist())
            if self.fault is not None:
                self.fault('readback')
            require(len(raw) == 32 * count, 'native readback length differs')
            completed = True
            return [raw[i:i + 32].hex() for i in range(0, len(raw), 32)]
        except BaseException as error:
            failure = error
            raise
        finally:
            self.busy = False
            drained = True
            if started and not completed:
                try:
                    stream.synchronize()
                except BaseException as drain:
                    drained = False
                    self.poisoned = True
                    self.quarantine.append(owners)
                    if failure is None:
                        raise
                    failure.add_note('native drain also failed; owners quarantined: ' + repr(drain))
            if drained:
                owners.clear()
            host = ptrs = lens = out = buffer = tensors = spans = stream = failure = None


def collect_occurrences(torch, value):
    found = []

    def visit(item):
        if isinstance(item, torch.Tensor):
            found.append(item)
        elif isinstance(item, dict):
            for key in sorted(item, key=repr):
                visit(key)
                visit(item[key])
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
    visit(value)
    return found


def occurrence_key(leaf):
    return leaf.data_ptr(), leaf._version, str(leaf.dtype), tuple(leaf.shape)


class OccurrenceCursor:
    def __init__(self, keys, facts):
        require(len(keys) == len(facts) and all(type(f) is tuple and len(f) == 3 and type(f[0]) is str and
                type(f[1]) is tuple and type(f[2]) is str and HEX.fullmatch(f[2]) for f in facts),
                'one exact (dtype, shape, digest) fact per occurrence required')
        self.keys, self.facts, self.index, self.open = list(keys), list(facts), 0, True

    def get(self, key):
        require(self.open, 'occurrence cursor is closed')
        require(self.index < len(self.keys), 'extra occurrence')
        require(key == self.keys[self.index], 'occurrence differs from the hashed leaf')
        self.index += 1
        return self.facts[self.index - 1]

    def finish(self):
        require(self.open and self.index == len(self.keys), 'missing occurrence')
        self.close()

    def close(self):
        self.keys = self.facts = ()
        self.open = False


def native_fingerprint(torch, hasher, original_fingerprint, value):
    tensors = collect_occurrences(torch, value)
    keys = [occurrence_key(leaf) for leaf in tensors]
    digests = hasher.digests(tensors)
    require(len(digests) == len(tensors), 'one digest per occurrence required')
    cursor = OccurrenceCursor(keys, [(k[2], k[3], d) for k, d in zip(keys, digests, strict=True)])
    try:
        result = original_fingerprint(value, frozen=cursor)
        cursor.finish()
        return result
    finally:
        cursor.close()


def load_fingerprint(raw):
    """The original fingerprint() exactly as written, extracted by AST from authenticated bytes."""
    nodes = [n for n in ast.parse(raw).body if isinstance(n, ast.FunctionDef) and n.name == 'fingerprint']
    require(len(nodes) == 1, 'exactly one original fingerprint() required')
    namespace = {'hashlib': hashlib}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original fingerprint>', 'exec', dont_inherit=True), namespace)
    return namespace['fingerprint'], ast.dump(nodes[0])


# ---------------------------------------------------------------- frozen smoke body
LENGTHS = (0, 1, 13, 14, 15, 16, 17, 29, 30, 31, 32, 33)  # float32 elements: 0,4,52-68,116-132 bytes
SPECIAL = (0x00000000, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC00001, 0xFFC12345, 0x00000001, 0x3F800000)


def pattern(elements, salt):
    """Deterministic asymmetric float32 bit patterns incl. literal signed zero/inf/NaN payloads."""
    words = [SPECIAL[(i + salt) % len(SPECIAL)] if i < len(SPECIAL) else
             (i * 2654435761 + salt * 40503 + 0x9E3779B9) & 0xFFFFFFFF for i in range(elements)]
    return struct.pack('<%dI' % elements, *words)


def device_leaf(torch, device, raw):
    if not raw:
        return torch.empty(0, dtype=torch.float32, device=device)
    return torch.frombuffer(bytearray(raw), dtype=torch.float32).to(device)


def leaf_bytes(torch, leaf):
    return bytes(leaf.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist())


def flip(torch, leaf, byte, mask):
    view = leaf.data.view(torch.uint8)
    view[byte] ^= mask


def xor_byte(raw, byte, mask):
    value = bytearray(raw)
    value[byte] ^= mask
    return bytes(value)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def smoke_cases(torch, device):
    """129 occurrences: lengths, scalar, offset, overlap and duplicate aliases, cycled past 128."""
    raws = [pattern(n, i) for i, n in enumerate(LENGTHS)]
    leaves = [device_leaf(torch, device, raw) for raw in raws]
    base_raw = pattern(40, 7)
    base = device_leaf(torch, device, base_raw)
    items = list(zip(leaves, raws, strict=True))
    items.append((base[5:6].reshape(()), base_raw[20:24]))
    items.append((base[3:20], base_raw[12:80]))
    items.append((base[2:12], base_raw[8:48]))
    items.append((base[7:17], base_raw[28:68]))
    items.extend((items[3], items[3]))
    cases = [items[i % len(items)] for i in range(129)]
    return [c[0] for c in cases], [c[1] for c in cases], leaves, raws, base, base_raw


def smoke_checks(torch, device, function, fingerprints, guard):
    calls = []

    def counted(*args):
        calls.append(args)
        return function(*args)
    hasher = Sha256Native(torch, counted)
    results = {}
    handle = torch.cuda.current_stream(device).cuda_stream

    # raw ABI: count zero with nulls, count 2**32, null top-level arguments with nonzero count
    leaf = device_leaf(torch, device, pattern(4, 3))
    probe = hasher.digests([leaf])
    require(probe == [sha(pattern(4, 3))], 'ABI warm-up parity differs')
    host_ptr, host_len = hasher.table([leaf.data_ptr()], device), hasher.table([16], device)
    scratch = torch.empty(32, dtype=torch.uint8, device=device)
    valid = (host_ptr[1].data_ptr(), host_len[1].data_ptr(), scratch.data_ptr())
    raw = {'count0_nulls': function(None, None, 0, None, handle),
           'count2e32': function(valid[0], valid[1], 2**32, valid[2], handle),
           'all_null': function(None, None, 1, None, handle),
           'ptrs_null': function(None, valid[1], 1, valid[2], handle),
           'lens_null': function(valid[0], None, 1, valid[2], handle),
           'out_null': function(valid[0], valid[1], 1, None, handle)}
    torch.cuda.current_stream(device).synchronize()
    require(raw['count0_nulls'] == 0 and all(v == 1 for k, v in raw.items() if k != 'count0_nulls'),
            'raw ABI status differs')
    results['raw_abi'] = raw
    guard()

    tensors, raws, leaves, leaf_raws, base, base_raw = smoke_cases(torch, device)
    require(len(tensors) == 129 and all(leaf_bytes(torch, t) == r for t, r in zip(tensors, raws, strict=True)),
            'fixture bytes differ from the host oracle')
    expected = [sha(r) for r in raws]
    before = len(calls)
    require(hasher.digests(tensors) == expected, 'raw digest parity differs')
    require(len(calls) == before + 1, 'one native call per occurrence list required')
    reverse = hasher.digests(tensors[::-1])
    require(reverse == expected[::-1] and reverse != expected, 'occurrence order differs')
    results['parity'] = {'occurrences': len(tensors), 'lengths': [len(r) for r in leaf_raws]}
    guard()

    # typed trees against the ORIGINAL fingerprint(), both serializer copies
    sixteen = leaves[5]
    trees = [{'a': leaves[3], 'b': (leaves[6], [leaves[7], leaves[7]]), 3: leaves[2], 'z': tensors[12], 'n': (1, 'x', None, 2.5, True)},
             (sixteen, sixteen), [sixteen, sixteen], (sixteen.reshape(4, 4),), (sixteen.reshape(2, 8),), (sixteen,),
             {'k': sixteen, 7: sixteen}, {'k': [leaves[0], leaves[1]]}, []]
    tree_prints = []
    for original in fingerprints:
        tree_prints.append([native_fingerprint(torch, hasher, original, tree) for tree in trees])
        require(tree_prints[-1] == [original(tree) for tree in trees], 'typed tree fingerprint differs')
    require(tree_prints[0] == tree_prints[1] and len(set(tree_prints[0][1:8])) == 7, 'tree framing collisions')
    results['typed_trees'] = {'trees': len(trees), 'serializers': len(fingerprints)}
    guard()

    # same-version byte mutation and exact restoration
    target, target_raw = leaves[6], leaf_raws[6]
    version = target._version
    flip(torch, target, 5, 0x40)
    changed = xor_byte(target_raw, 5, 0x40)
    require(target._version == version and leaf_bytes(torch, target) == changed, 'same-version mutation precondition differs')
    require(hasher.digests([target]) == [sha(changed)] != [sha(target_raw)], 'mutated digest differs')
    flip(torch, target, 5, 0x40)
    require(target._version == version and hasher.digests([target]) == [sha(target_raw)], 'restored digest differs')
    results['mutation'] = {'version': version, 'byte': 5, 'mask': 0x40}
    guard()

    # nondefault current stream: mutation still pending when hashing is enqueued, oracle copy last
    side = torch.cuda.Stream(device=device)
    with torch.cuda.stream(side):
        require(torch.cuda.current_stream(device).cuda_stream != 0, 'nondefault current stream required')
        require(hasher.digests([target]) == [sha(target_raw)], 'side-stream warm-up differs')
        torch.cuda._sleep(200_000_000)
        flip(torch, target, 9, 0x04)
        marker = torch.cuda.Event()
        marker.record(side)
        require(not marker.query(), 'mutation was not pending; inconclusive')
        pending = hasher.digests([target])
        oracle = leaf_bytes(torch, target)
    side.synchronize()
    require(oracle == xor_byte(target_raw, 9, 0x04) and pending == [sha(oracle)], 'pending-mutation digest differs')
    flip(torch, target, 9, 0x04)
    require(hasher.digests([target]) == [sha(target_raw)], 'side-stream restoration differs')
    results['nondefault_stream_pending'] = {'byte': 9, 'mask': 0x04}
    guard()

    # injected failures with real work outstanding; owners must drain, be released, then a healthy call works
    injected, kept = {}, {}
    for point in ('status', 'launched', 'completed', 'readback'):
        seen = []
        if point == 'status':
            def once(*args, seen=seen):
                result = function(*args)
                return result if seen else seen.append(1) or 7
            failing = Sha256Native(torch, once)
        else:
            def fault(name, seen=seen, point=point):
                if name == point and not seen:
                    seen.append(1)
                    raise InjectedFault('injected ' + point)
            failing = Sha256Native(torch, function, fault)
        owners = [device_leaf(torch, device, raw) for raw in raws[:3]]
        refs = [weakref.ref(leaf) for leaf in owners]
        try:
            failing.digests(owners)
        except (ValueError, InjectedFault) as error:
            kept[point] = error  # a retained error must not pin the drained owners
            injected[point] = type(error).__name__
        else:
            raise ValueError('injected %s failure returned digests' % point)
        del owners
        gc.collect()
        require(all(ref() is None for ref in refs), 'drained failure kept its owners alive')
        require(not failing.poisoned and not failing.quarantine and seen, 'injected failure left a poisoned context')
        require(failing.digests(tensors[:3]) == expected[:3], 'healthy call after injected failure differs')
    del kept
    results['injected_failures'] = injected
    guard()

    # pre-enqueue rejections never reach the native function
    calls_before = len(calls)
    bad = {'cpu': torch.zeros(4), 'float64': torch.zeros(4, dtype=torch.float64, device=device),
           'transposed': torch.zeros(4, 4, device=device).t(), 'stride_zero': torch.zeros(1, device=device).expand(4),
           'tuple': (leaves[1],), 'int_list': [1]}
    rejected = []
    for name, value in bad.items():
        try:
            hasher.digests(value if isinstance(value, (tuple, list)) else [value])
        except ValueError:
            rejected.append(name)
    require(rejected == list(bad) and len(calls) == calls_before, 'pre-enqueue rejection differs')
    results['rejections'] = rejected
    return results, len(calls)


def resources(torch):
    import resource
    swap = re.search(r'VmSwap:\s+(\d+) kB', Path('/proc/self/status').read_text())
    require(swap is not None and int(swap[1]) == POLICY['swap_bytes'], 'swap use differs')
    facts = {'wall_seconds': time.perf_counter() - STARTED,
             'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
    require(facts['wall_seconds'] < POLICY['body_seconds'], 'body deadline differs')
    requests.check_resources(facts, POLICY, reserve=True)
    return facts


def smoke(context, output, torch=None, cdll=None, maps='/proc/self/maps'):
    import ctypes
    if torch is None:
        import importlib
        torch = importlib.import_module('torch')
    if cdll is None:
        cdll = ctypes.CDLL
    authority, record = context.authority, context.authority.record
    context.source.check()
    prints = []
    for name in ('probe_serializer', 'mlp_serializer'):
        function, dump = load_fingerprint(read_bytes(record['sources'][name], 2 * 1024**2))
        prints.append((function, dump))
    require(prints[0][1] == prints[1][1], 'probe and MLP original fingerprint() differ')
    spec = record['device']
    require(torch.cuda.is_available() and torch.cuda.current_device() == spec['index'], 'frozen CUDA device required')
    props = torch.cuda.get_device_properties(spec['index'])
    require(props.name == spec['name'] and [props.major, props.minor] == spec['capability'], 'frozen CUDA device differs')
    device = torch.device('cuda', spec['index'])
    def guard():
        return resources(torch)
    guard()
    fd, lib, mappings = open_library(authority, cdll, maps)
    function = bind_abi(lib, ctypes)
    checks, native_calls = smoke_checks(torch, device, function, [p[0] for p in prints], guard)
    torch.cuda.synchronize()
    require(mapped_files(authority.inventory, maps) == mappings, 'native mapping inventory changed during the run')
    context.source.check()
    final = resources(torch)
    receipt = {'schema': SMOKE_RECEIPT, 'status': 'SMOKE_DIAGNOSTIC_UNREVIEWED', 'engineering_only': True,
               'authority': context.fact, 'library': record['library'], 'device': spec, 'checks': checks,
               'native_calls': native_calls, 'resources': final, 'resource_policy': POLICY, 'mappings': sorted(mappings),
               **{flag: False for flag in FLAGS}, 'normal_terminal_required': True,
               'invocation': {'argv': sys.argv, 'python': str(Path(sys.executable).resolve()), 'pid': os.getpid(),
                              'optimize': sys.flags.optimize, 'torch': torch.__version__, 'cuda': torch.version.cuda}}
    os.mkdir(output, 0o700)
    with open(Path(output) / 'smoke-receipt.json', 'x') as stream:
        json.dump(receipt, stream, sort_keys=True, indent=1)
        stream.flush()
        os.fsync(stream.fileno())
    return receipt


STAGE_BODIES = {'smoke': smoke}


# ---------------------------------------------------------------- CLI
def cli(stage, authority, unit, output):
    argv = [stage, '--authority', authority['path'], '--authority-sha256', authority['sha256']]
    if unit is not None:
        argv += ['--unit', unit['path'], '--unit-sha256', unit['sha256']]
    return argv + ['--output', output]


def parser():
    result = argparse.ArgumentParser(prog='qualify_cuda_sha256_native.py', allow_abbrev=False,
                                     description='Standalone CUDA SHA-256 native caller; smoke only, root-run.')
    result.add_argument('stage', choices=STAGES)
    result.add_argument('--authority', required=True)
    result.add_argument('--authority-sha256', required=True)
    result.add_argument('--unit')
    result.add_argument('--unit-sha256')
    result.add_argument('--output', required=True)
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    authority = {'path': args.authority, 'sha256': args.authority_sha256}
    unit = None if args.unit is None else {'path': args.unit, 'sha256': args.unit_sha256}
    require((args.unit is None) == (args.unit_sha256 is None) and (args.stage == 'smoke') == (args.unit is None),
            '--unit is forbidden for smoke and required with its hash for full/timing')
    require(argv == cli(args.stage, authority, unit, args.output), 'fixed canonical CLI order required')
    output = Path(args.output)
    require(output.is_absolute() and str(output) == args.output and output.parent.is_dir() and
            output.parent.resolve() == output.parent and not os.path.lexists(output), 'new canonical output directory required')
    require(args.stage in STAGE_BODIES, '%s stage body is unreleased scaffolding' % args.stage)
    if unit is not None:
        require(UNIT_READER is not None, 'genuine root-owned terminal unit reader required; a user-written GO is never accepted')
        UNIT_READER(unit, PRIOR[args.stage], authority)
    admitted = read_native_authority(authority)
    source = requests.Source(sys.modules[__name__], admitted.record['sources']['driver'])
    locks = requests.Locks(admitted.record['locks'])
    context = SimpleNamespace(authority=admitted, fact=authority, source=source, locks=locks)
    return STAGE_BODIES[args.stage](context, args.output)


if __name__ == '__main__':
    main()
