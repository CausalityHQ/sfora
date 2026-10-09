#!/usr/bin/env python3
"""Standalone CUDA SHA-256 native caller + frozen smoke and full entrypoints; UNQUALIFIED.

The smoke stage passed natively (root, normal0). The full stage below is SOURCE ONLY: never compiled-in, launched or
timed from here; no Torch/NumPy/GPU/SSH work happens at import or in the tests. Native execution is root-only. No speed,
quality, SOTA or product claim follows from this file. It calls rust/sfora-cuda-sha256/
sha256_occurrences.cu unchanged; serving, defaults, helpers, math and the original
serializers are untouched. timing stays UNRELEASED SCAFFOLDING (no STAGE_BODIES entry); a full engineering receipt is
never production, optimization or qualification eligibility.

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
 declared files to appear. EVERY mapped .so (declared or not) is then freshly stream-hashed through
 an O_NOFOLLOW fd against the sha256 the root froze in the inventory, with the opened inode equal to
 the mapped one: a mapping whose bytes are not authenticated fails closed. The set is re-read and
 every mapped file re-hashed at exit. Unmapped inventory entries are not hashed here.
 cuda-sha256-native-authority-full-v1 (the full stage only; smoke accepts ONLY the v1 record above): the v1 keys plus
  full = {prior:{unit,launch,verification}, fixture, extractor, metadata}, all FILEs, distinct from every other FILE,
  hashed at admission, rehashed (fixture/extractor/metadata) at every full-body guard and (everything, plus the whole
  prior chain re-read) at exit. prior.unit must equal --unit. The closure FILEs (build authority/receipt, library,
  runtime files + provenance, mapping inventory, interpreter, probe/MLP serializers) must be byte-identical (sha256) to
  the prior smoke authority's; their paths may differ. fixture = cuda-sha256-synthetic-vision-inventory-v1.
 UNIT: a user-written {decision:GO,authority} is NEVER accepted. UNIT_READER = read_prior(unit, admitted, 'smoke')
 authenticates the ORIGINAL normal0 smoke unit from its FILEs: the schema-less unit record {both_locks_held,
 invocation_id, log, native_peak_rss_kib, receipt, service_seconds, unit}; the smoke receipt (exact keys, flags false, the
 24 native calls and every smoke check, invocation argv/python, resources inside the policy, mappings inside the frozen
 inventory); the eight-line systemd terminal footer (unit + invocation ID, bootstrap_exit_pass, zero cgroup events, zero
 swap, peak <= 8 GiB, success, code=exited/status=0, runtime == service_seconds); the root launch (unit name, properties,
 environment, bootstrap command, native authority, output); the bootstrap and its authority; the prior native authority
 and its driver/test bytes; and the parent verification (terminal == the unit record, outer resources == the footer's).

CLI (argv order is canonical and exact; invoke by the canonical absolute script path):
  qualify_cuda_sha256_native.py STAGE --authority FILE.json --authority-sha256 SHA
      [--unit FILE.json --unit-sha256 SHA] --output NEWDIR            STAGE in smoke|full|timing
 --unit is forbidden for smoke and required for full/timing. NEWDIR is a canonical absolute
 path whose parent exists and which does not exist. smoke admits (authority, sources, FILEs,
 interpreter, locks, resources policy) and then runs the frozen smoke body; full additionally runs UNIT_READER
 before the body; STAGE_BODIES has no timing entry, so timing fails closed. The smoke body checks: raw ABI nulls/oversize,
 hashlib parity for byte lengths 0,4,52-68,116-132 with scalar/empty/offset/duplicate/overlap
 over 129 occurrences, typed-tree equality with the ORIGINAL fingerprint() extracted by AST
 from the authenticated probe and MLP serializer FILEs, occurrence order, same-version .data
 mutation/restore, a nondefault current stream with a mutation still pending hashed before any
 oracle copy, labeled InjectedFault launch/completion/readback failures with real work
 outstanding, drain, released owners under a retained error and a healthy next call, and
 pre-enqueue rejections with zero native calls. It writes one diagnostic receipt (all
 eligibility flags false) only after every check AND the exit pass succeed.

 The full body (same admission, clocks, guards, exit and receipt lifecycle through run_stage/open_stage; synthetic
 inventory only, no model weights, no quality read). build_inventory runs the ORIGINAL expected_vision/PREFIX/PROFILES
 (AST-extracted from the authenticated extractor bytes) over the authenticated upstream metadata config/model, strips the
 exact 'vision_model.' prefix and requires the fixture, 448 leaves, 1711552256 bytes, the 19832832-byte largest leaf and the
 PROBE/MLP roles of both original serializers (447 / 444 frozen) to agree. Correctness gates come first, on a real-size
 separately allocated largest leaf ([4304, 1152], deterministic seeded Generator.normal_): raw and typed parity against the
 ORIGINAL streamed host digest (D2H + uint8 numpy memoryview hashlib; never tolist/bytes), three same-version byte mutations
 with exact restoration, offset/duplicate/overlapping aliases and occurrence order, a nondefault current stream, an explicit
 producer-stream event -> consumer wait_event dependency with the candidate run before the oracle copy and no host barrier,
 and the retained injected status/launched/completed/readback failures with a checked drain, released owners and a healthy
 follow-up. Then all 448 leaves are allocated (separate storages, unique, non-overlapping) and each of probe 447 -> validate
 -> 448 (PROBE serializer), MLP 444 -> validate -> 448 (MLP serializer) and the isolated largest leaf is compared with the
 ORIGINAL fingerprint of the same current bytes, with exactly one native launch per tree (no amalgamation, deduplication,
 caching or reuse). A role-leaf mutation must change only the full digest. Only then PAIRS=3 alternating original/candidate
 arms per workload are timed (order swaps every pair, every arm digest-checked); the clock scope is CLOCK, guard() checks sit
 outside the clocks, and nothing is thresholded or projected. timing and every qualification gate stay separate releases.

 Smoke clocks and exit (policy unchanged: body 300 / whole 1500 / exit reserve 300). The body clock
 starts at smoke() entry (torch import and CUDA init count; admission does not) and is checked < 300
 only up to the body-end guard; the whole clock runs from module load and must leave the 300 s
 reserve at that guard (< 1200), then only < 1500 after the exit work. guard() (start, after each
 check group, body end) = Source.check + Locks.check + resources. The exit runs after the body on EVERY
 path, success or failure, each check attempted even after a body, drain or earlier exit failure
 (errors are collected, never short-circuited): device drain (only if Torch loaded) -> current
 mapping set (equal to the post-open snapshot when one exists; a load that failed before the
 snapshot is not an exit failure) with every mapped .so re-hashed -> a FRESH
 read_native_authority(authority) (authority, 4 sources, build authority + CUDA source/contract/
 compile script/compilers/evidence, receipt/library/logs, runtime files + provenance + evidence,
 inventory, interpreter; record and inventory must equal the admitted ones) -> Source.check ->
 Locks.check -> close the library fd -> final resources. Only if the body and all of that
 succeeded is the receipt written; afterwards Locks.check and the whole cap are checked once more,
 and a failure there raises while the diagnostic receipt (normal_terminal_required) remains on disk.
 The body rejection stays primary (requests.raise_failures) and every other error is attached.
 One try/except/finally spans it: the fd is closed once on every path, and only after a good
 drain with no quarantined hasher are the error frames' owners released (traceback.clear_frames).
 Limits: "uncached" = no digest/descriptor/authority reuse, not a page-cache bypass; Python source
 modules and non-.so mappings are outside the closure; the 8 GiB RSS cap is peak-RSS only.

UNVERIFIED NATIVE ASSUMPTIONS (the tests use a fake torch/ABI and cannot check them): Tensor.data
carries a fresh version counter; torch.cuda._sleep plus Event.query prove a mutation is still
pending (the check fails closed if it is not); pinned non_blocking copies and the caching
allocator never insert a host barrier; the CUDA runtime/driver mappings fit the explicit
inventory. Full stage, additionally: torch.Generator(device).manual_seed + Tensor.normal_(generator=) is deterministic and
allocates no host copy; untyped_storage().nbytes() of a fresh torch.empty equals its requested bytes; reshape(-1) of a
contiguous leaf is a view and .data of it shares storage without bumping the original's version; Stream.wait_event makes a
later current-stream launch see the producer's earlier write; the 19.8 MB single-occurrence kernel finishes inside the body
clock. Not integrated: encoder_facts callers, serving, extraction ledgers, bridge provenance.
"""
import argparse
import ast
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import sys
import time
import traceback
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
NATIVE_FULL, FULL_RECEIPT = 'cuda-sha256-native-authority-full-v1', 'cuda-sha256-native-full-v1'
FIXTURE = 'cuda-sha256-synthetic-vision-inventory-v1'
BUILD_KEYS = {'schema', 'source', 'contract', 'toolchain', 'compile_script', 'target', 'flags',
              'environment', 'output_dir', 'evidence'}
RECEIPT_KEYS = {'schema', 'build_authority', 'exit_status', 'library', 'sass_targets', 'ptx_images', 'logs'}
SOURCE_KEYS = {'driver', 'test', 'probe_serializer', 'mlp_serializer'}
NATIVE_KEYS = {'schema', 'sources', 'build_authority', 'build_receipt', 'library', 'runtime_files',
               'mapping_inventory', 'interpreter', 'device', 'resource_policy', 'locks'}
FULL_KEYS = {'prior', 'fixture', 'extractor', 'metadata'}  # the full stage's one extra authority key: 'full'
PRIOR_KEYS = {'unit', 'launch', 'verification'}
FIXTURE_KEYS = {'bytes', 'largest_leaf_bytes', 'leaves', 'metadata', 'mlp', 'model', 'probe', 'quality_read', 'resolved',
                'schema', 'shapes', 'source', 'synthetic'}
LEAVES, BYTES, LARGEST_BYTES = 448, 1711552256, 19832832  # the frozen synthetic inventory facts, re-derived and compared
ARCH_FLAGS = ('-arch', '--gpu-architecture', '-code', '--gpu-code', '-ptx', '--ptx', '-gencode',
              '--generate-code')
HEX = re.compile('[0-9a-f]{64}')
FLAGS = ('quality_read', 'quality_eligible', 'qualification_eligible', 'state_reuse_eligible',
         'optimization_eligible', 'product_go', 'speed_go')
STARTED = time.perf_counter()  # whole-process clock (module load); the body clock starts at smoke() entry


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


def read_keys(fact, keys, label):
    record = requests.strict_json(read_bytes(fact, JSON_LIMIT))
    require(type(record) is dict and record.keys() == keys, 'exact %s required' % label)
    return record


def read_json(fact, schema, keys):
    record = read_keys(fact, keys, schema)
    require(record.get('schema') == schema, 'exact %s required' % schema)
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


def read_full_extension(extension):
    """The full stage's prospectively bound FILEs: the prior smoke chain and the fixture/extractor/config."""
    require(type(extension) is dict and extension.keys() == FULL_KEYS and type(extension['prior']) is dict and
            extension['prior'].keys() == PRIOR_KEYS, 'exact full-stage extension required')
    rows = [*extension['prior'].values(), extension['fixture'], extension['extractor'], extension['metadata']]
    for item in rows:
        hash_file(item)
    return rows


def read_native_authority(fact, stage='smoke'):
    full = stage == 'full'
    require(stage in ('smoke', 'full'), 'a smoke or full native authority is required')
    record = read_json(fact, NATIVE_FULL if full else NATIVE, NATIVE_KEYS | {'full'} if full else NATIVE_KEYS)
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
    extension = read_full_extension(record['full']) if full else []
    paths = [record['library'], record['build_authority'], record['build_receipt'], record['mapping_inventory'],
             *sources.values(), *runtime, *(r['provenance'] for r in rows), *extension]
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


# ---------------------------------------------------------------- the genuine prior smoke unit
UNIT_KEYS = {'both_locks_held', 'invocation_id', 'log', 'native_peak_rss_kib', 'receipt', 'service_seconds', 'unit'}
LAUNCH, BOOTSTRAP = 'cuda-sha256-smoke-root-launch-v1', 'cuda-sha256-smoke-bootstrap-v1'
VERIFICATION = 'standalone-cuda-sha-smoke-parent-verification-v1'
LAUNCH_KEYS = {'bootstrap', 'bootstrap_authority', 'command', 'engineering_only', 'native_authority', 'output',
               'product_go', 'quality_go', 'schema', 'speed_go', 'unit'}
BOOTSTRAP_KEYS = {'schema', 'files', 'native_authority', 'output', 'interpreter'}
BOOTSTRAP_FILES = {'driver', 'test', 'requests', 'probe_serializer', 'mlp_serializer'}
VERIFICATION_KEYS = {'all_smoke_checks', 'complete_current_mapped_file_hashes_exit_pass', 'engineering_only',
                     'exact_authority', 'exit_status', 'next', 'original_helper_and_source_exit_pass', 'outer_resources',
                     'product_go', 'schema', 'speed_go', 'terminal'}
SMOKE_RECEIPT_KEYS = {'schema', 'status', 'engineering_only', 'authority', 'library', 'device', 'checks', 'native_calls',
                      'resources', 'resource_policy', 'mappings', *FLAGS, 'normal_terminal_required', 'invocation'}
OUTER_KEYS = {'memory_events', 'memory_peak_bytes', 'swap_current_bytes', 'wall_seconds'}
EVENT_KEYS = {'high', 'low', 'max', 'oom', 'oom_group_kill', 'oom_kill'}
SMOKE_NATIVE_CALLS = 24
SMOKE_CHECKS = {  # exactly what the released smoke body reports when every check passes (mutation.version is an observation)
    'injected_failures': {'completed': 'InjectedFault', 'launched': 'InjectedFault', 'readback': 'InjectedFault',
                          'status': 'ValueError'},
    'mutation': {'byte': 5, 'mask': 64, 'version': 0},
    'nondefault_stream_pending': {'byte': 9, 'mask': 4},
    'parity': {'lengths': [4 * n for n in (0, 1, 13, 14, 15, 16, 17, 29, 30, 31, 32, 33)], 'occurrences': 129},
    'raw_abi': {'all_null': 1, 'count0_nulls': 0, 'count2e32': 1, 'lens_null': 1, 'out_null': 1, 'ptrs_null': 1},
    'rejections': ['cpu', 'float64', 'transposed', 'stride_zero', 'tuple', 'int_list'],
    'typed_trees': {'serializers': 2, 'trees': 9}}


def canon(value):
    return json.dumps(value, sort_keys=True)  # type-exact comparison: True never equals 1


def number(value, high):
    return type(value) in (int, float) and math.isfinite(value) and 0 < value < high


def sha_of(item):
    return item.get('sha256') if type(item) is dict else None


def read_footer(raw, unit):
    """The original systemd --wait --pipe terminal of a normal0 service: exactly eight lines, no truncation."""
    require(raw.endswith(b'\n'), 'complete terminal log required')
    try:
        lines = raw.decode('utf-8').split('\n')[:-1]
    except UnicodeDecodeError:
        raise ValueError('terminal log is not UTF-8') from None
    require(len(lines) == 8, 'exact eight-line normal terminal footer required')
    head = re.fullmatch(r'Running as unit: ([a-z0-9][a-z0-9-]*)\.service; invocation ID: ([0-9a-f]{32})', lines[0])
    require(head is not None and head[1] == unit['unit'] and head[2] == unit['invocation_id'],
            'terminal log unit/invocation ID differs')
    boot = requests.strict_json(lines[1])
    require(type(boot) is dict and boot.keys() == {'bootstrap_exit_pass', 'normal_outer_terminal_required', 'resources'} and
            boot['bootstrap_exit_pass'] is True and boot['normal_outer_terminal_required'] is True,
            'bootstrap exit pass required')
    outer = boot['resources']
    require(type(outer) is dict and outer.keys() == OUTER_KEYS and type(outer['memory_events']) is dict and
            outer['memory_events'].keys() == EVENT_KEYS and
            all(type(v) is int and v == 0 for v in outer['memory_events'].values()) and
            type(outer['memory_peak_bytes']) is int and 0 < outer['memory_peak_bytes'] <= POLICY['host_bytes'] and
            type(outer['swap_current_bytes']) is int and outer['swap_current_bytes'] == POLICY['swap_bytes'] and
            number(outer['wall_seconds'], POLICY['whole_process_seconds']), 'outer unit resources differ')
    runtime = re.fullmatch(r'Service runtime: ([0-9]+\.[0-9]{3})s', lines[4])
    require(lines[2:4] == ['Finished with result: success', 'Main processes terminated with: code=exited/status=0'] and
            runtime is not None and float(runtime[1]) == unit['service_seconds'] and
            re.fullmatch(r'CPU time consumed: [0-9]+\.[0-9]{3}s', lines[5]) and
            re.fullmatch(r'Memory peak: [0-9.]+[BKMG]', lines[6]) and lines[7] == 'Memory swap peak: 0B',
            'normal0 terminal footer differs')
    return outer


def prior_facts(unit_fact, admitted, stage):
    """Authenticate the ORIGINAL normal0 smoke unit the full authority froze: unit record, receipt, terminal log, launch,
    bootstrap, prior native authority and its source bytes, parent verification. Returns JSON-able facts. A user-written
    decision:GO record, or any forged/truncated/substituted/failed part, raises. Nothing is inferred from smoke metadata."""
    require(stage == 'smoke', 'the only released prior stage is smoke')
    current = admitted.record
    frozen = current['full']['prior']
    require(unit_fact == frozen['unit'], 'the authority did not freeze this unit FILE')
    unit = read_keys(unit_fact, UNIT_KEYS, 'smoke unit record')
    require(unit['both_locks_held'] is True and type(unit['invocation_id']) is str and
            re.fullmatch('[0-9a-f]{32}', unit['invocation_id']) and type(unit['unit']) is str and
            re.fullmatch('[a-z0-9][a-z0-9-]*', unit['unit']) and type(unit['native_peak_rss_kib']) is int and
            unit['native_peak_rss_kib'] > 0 and number(unit['service_seconds'], POLICY['whole_process_seconds']),
            'normal0 unit identity required')
    receipt = read_json(unit['receipt'], SMOKE_RECEIPT, SMOKE_RECEIPT_KEYS)
    outer = read_footer(read_bytes(unit['log'], 1024**2), unit)
    launch = read_json(frozen['launch'], LAUNCH, LAUNCH_KEYS)
    prior = read_json(receipt['authority'], NATIVE, NATIVE_KEYS)
    # the prior closure is the current closure: same bytes (paths may differ), device, policy and original locks
    rows, now = prior['runtime_files'], current['runtime_files']
    sources, mine = prior['sources'], current['sources']
    require(type(rows) is list and len(rows) == len(now) and all(type(r) is dict for r in rows) and
            type(sources) is dict and
            [(sha_of(r.get('file')), sha_of(r.get('provenance'))) for r in rows] ==
            [(r['file']['sha256'], r['provenance']['sha256']) for r in now] and
            all(sha_of(prior[k]) == current[k]['sha256'] for k in
                ('build_authority', 'build_receipt', 'library', 'interpreter', 'mapping_inventory')) and
            all(sha_of(sources.get(k)) == mine[k]['sha256'] for k in ('probe_serializer', 'mlp_serializer')) and
            prior['device'] == current['device'] and prior['resource_policy'] == current['resource_policy'] == POLICY and
            prior['locks'] == current['locks'], 'prior smoke closure differs from the full authority')
    hash_file(sources['driver'])
    hash_file(sources['test'])
    # launch + bootstrap
    command, name, output = launch['command'], unit['unit'], launch['output']
    props = ['--property=RuntimeMaxSec=%d' % POLICY['whole_process_seconds'], '--property=MemoryMax=%d' % POLICY['host_bytes'],
             '--property=MemorySwapMax=%d' % POLICY['swap_bytes'], '--property=TasksMax=128',
             '--property=KillMode=control-group', '--property=OOMPolicy=stop']
    boot_fact, boot_auth_fact = launch['bootstrap'], launch['bootstrap_authority']
    require(launch['engineering_only'] is True and launch['product_go'] is False and launch['quality_go'] is False and
            launch['speed_go'] is False and launch['unit'] == name and launch['native_authority'] == receipt['authority'] and
            type(output) is str and unit['receipt']['path'] == str(Path(output) / 'smoke-receipt.json') and
            type(command) is list and all(type(c) is str for c in command) and len(command) == 23 and
            command[:6] == ['/usr/bin/systemd-run', '--user', '--unit=' + name, '--wait', '--pipe', '--collect'] and
            command[6:12] == props and command[12:16] == ['--setenv=CUDA_VISIBLE_DEVICES=0', '--setenv=OMP_NUM_THREADS=1',
                                                          '--setenv=OPENBLAS_NUM_THREADS=1', '--setenv=MKL_NUM_THREADS=1'] and
            Path(command[16]).is_absolute() and command[17:] == ['-I', '-B', boot_fact['path'], boot_auth_fact['path'],
                                                                 boot_auth_fact['sha256'], boot_fact['sha256']],
            'exact root launch of the prior unit required')
    hash_file(boot_fact)
    bootstrap = read_json(boot_auth_fact, BOOTSTRAP, BOOTSTRAP_KEYS)
    files = bootstrap['files']
    require(type(files) is dict and files.keys() == BOOTSTRAP_FILES and bootstrap['native_authority'] == receipt['authority'] and
            bootstrap['output'] == output and bootstrap['interpreter'] == prior['interpreter'] and
            all(files[k] == sources[k] for k in ('driver', 'test', 'probe_serializer', 'mlp_serializer')),
            'bootstrap authority differs from the prior native authority')
    hash_file(files['requests'])
    # the receipt written by the prior driver
    argv = [sources['driver']['path'], 'smoke', '--authority', receipt['authority']['path'], '--authority-sha256',
            receipt['authority']['sha256'], '--output', output]
    invocation, facts, checks = receipt['invocation'], receipt['resources'], receipt['checks']
    require(type(invocation) is dict and invocation.keys() == {'argv', 'python', 'pid', 'optimize', 'torch', 'cuda'} and
            invocation['argv'] == argv and invocation['python'] == prior['interpreter']['path'] and
            type(invocation['pid']) is int and invocation['pid'] > 0 and type(invocation['optimize']) is int and
            invocation['optimize'] == 0 and all(type(invocation[k]) is str and invocation[k] for k in ('torch', 'cuda')),
            'exact normal0 invocation required')
    require(receipt['status'] == 'SMOKE_DIAGNOSTIC_UNREVIEWED' and receipt['engineering_only'] is True and
            receipt['normal_terminal_required'] is True and all(receipt[f] is False for f in FLAGS) and
            receipt['library'] == prior['library'] and receipt['device'] == prior['device'] and
            receipt['resource_policy'] == POLICY and type(receipt['native_calls']) is int and
            receipt['native_calls'] == SMOKE_NATIVE_CALLS and type(checks) is dict and type(checks.get('mutation')) is dict and
            type(checks['mutation'].get('version')) is int and
            canon({**checks, 'mutation': {**checks['mutation'], 'version': 0}}) == canon(SMOKE_CHECKS),
            'every smoke check must have passed in the prior receipt')
    require(type(facts) is dict and facts.keys() == {'wall_seconds', 'body_seconds', 'process_peak_rss_kib', 'peak_cuda_allocated_bytes'} and
            number(facts['body_seconds'], POLICY['body_seconds']) and
            facts['process_peak_rss_kib'] == unit['native_peak_rss_kib'] and
            number(facts['wall_seconds'], POLICY['whole_process_seconds']) and
            facts['wall_seconds'] <= outer['wall_seconds'] <= unit['service_seconds'], 'prior resources differ')
    requests.check_resources(facts, POLICY, reserve=False)
    mapped, declared = receipt['mappings'], {prior['library']['path'], *(r['file']['path'] for r in rows)}
    require(type(mapped) is list and mapped == sorted(set(mapped)) and declared <= set(mapped) and
            set(mapped) <= set(admitted.inventory), 'prior mapping set differs from the frozen inventory')
    # the parent's verification of exactly this unit
    verified = read_json(frozen['verification'], VERIFICATION, VERIFICATION_KEYS)
    require(canon(verified['terminal']) == canon(unit) and canon(verified['outer_resources']) == canon(outer) and
            all(verified[k] is True for k in ('all_smoke_checks', 'complete_current_mapped_file_hashes_exit_pass',
                                               'engineering_only', 'exact_authority', 'original_helper_and_source_exit_pass')) and
            verified['product_go'] is False and verified['speed_go'] is False and type(verified['exit_status']) is int and
            verified['exit_status'] == 0, 'parent verification of the prior unit differs')
    return {'unit': unit_fact, 'unit_name': name, 'invocation_id': unit['invocation_id'], 'receipt': unit['receipt'],
            'log': unit['log'], 'launch': frozen['launch'], 'verification': frozen['verification'],
            'authority': receipt['authority'], 'service_seconds': unit['service_seconds'], 'outer_resources': outer}


def read_prior(unit_fact, admitted, stage='smoke'):
    try:
        return prior_facts(unit_fact, admitted, stage)
    except (KeyError, TypeError, IndexError, AttributeError) as error:
        raise ValueError('prior smoke unit record has the wrong shape: %r' % (error,)) from error


UNIT_READER = read_prior  # a user-written decision:GO record is never accepted; only this genuine chain reader is


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


def authenticate_mappings(inventory, mappings):
    """Fresh streamed hash of every mapped .so inode against the digest the root froze for it; no size/inode-only trust."""
    for path, identity in mappings.items():
        entry = inventory[path]
        info = hash_file({'path': path, 'sha256': entry['sha256']})
        require((info.st_dev, info.st_ino) == identity and info.st_size == entry['size'],
                'mapped native file is not the authenticated inode')


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
        authenticate_mappings(authority.inventory, after)
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


def injected_failures(torch, function, hashers, make_owners, follow, want):
    """Labeled InjectedFault/status failures with real work outstanding. Each must return no digest, drain, release every
    owner despite the retained error, leave the context healthy and allow a correct next call. Shared by smoke and full."""
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
        hashers.append(failing)
        owners = make_owners()
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
        require(failing.digests(follow) == want, 'healthy call after injected failure differs')
    del kept
    return injected


def smoke_checks(torch, device, function, fingerprints, guard, hashers):
    calls = []

    def counted(*args):
        calls.append(args)
        return function(*args)
    hasher = Sha256Native(torch, counted)
    hashers.append(hasher)
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
    results['injected_failures'] = injected_failures(
        torch, function, hashers, lambda: [device_leaf(torch, device, raw) for raw in raws[:3]], tensors[:3], expected[:3])
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


def resources(torch, body_started, final=False):
    """body300 counts from body entry, whole1500 from process start. Before the exit work the whole clock must still
    leave the 300 s exit reserve; once the exit work is done (final) only the whole cap remains."""
    import resource
    swap = re.search(r'VmSwap:\s+(\d+) kB', Path('/proc/self/status').read_text())
    require(swap is not None and int(swap[1]) == POLICY['swap_bytes'], 'swap use differs')
    now = time.perf_counter()
    facts = {'wall_seconds': now - STARTED, 'body_seconds': now - body_started,
             'process_peak_rss_kib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             'peak_cuda_allocated_bytes': 0 if torch is None else torch.cuda.max_memory_allocated()}  # none if Torch never loaded
    require(final or facts['body_seconds'] < POLICY['body_seconds'], 'body deadline differs')
    requests.check_resources(facts, POLICY, reserve=not final)
    return facts


def exit_pass(context, held, maps, body_started, stage='smoke'):
    """The independent exit, run after the body on EVERY path. Each check is attempted even after a body, drain or
    earlier exit failure; the errors are returned, never raised. Nothing admitted earlier is trusted or reused."""
    errors, authority, torch = [], context.authority, held.torch

    def attempt(check):
        try:
            check()
        except BaseException as error:
            errors.append(error)

    def drain():
        held.drain_tried = True
        torch.cuda.synchronize()
        held.drained = True

    def mapped():  # a missing post-open snapshot (load failed first) is not an exit failure; the bytes still are checked
        current = mapped_files(authority.inventory, maps)
        require(held.mappings is None or current == held.mappings, 'native mapping inventory changed during the run')
        authenticate_mappings(authority.inventory, current)

    def closure():  # every authority/source/compiler/evidence/runtime/interpreter/inventory FILE, freshly hashed
        fresh = read_native_authority(context.fact) if stage == 'smoke' else read_native_authority(context.fact, stage)
        require(fresh.record == authority.record and fresh.inventory == authority.inventory, 'admitted closure differs at exit')
        if stage == 'full':  # the whole prior smoke chain is read again from its FILEs, never from the admitted facts
            require(UNIT_READER(context.unit, fresh, PRIOR[stage]) == context.prior, 'prior smoke unit chain differs at exit')

    def release():
        fd, held.fd = held.fd, None
        if fd is not None:
            os.close(fd)

    def final():
        held.final = resources(torch, body_started, final=True)

    if torch is not None:
        attempt(drain)
        attempt(mapped)
    attempt(closure)
    attempt(context.source.check)
    attempt(context.locks.check)
    attempt(release)
    attempt(final)
    return errors


def teardown(held, hashers, failure):
    """Last resort cleanup: drain if the exit never tried; only after a good drain with no quarantined hasher release the
    error frames' owners; always close the fd. Returns the errors."""
    errors, torch = [], held.torch
    if torch is not None and not held.drain_tried:
        held.drain_tried = True
        try:
            torch.cuda.synchronize()
            held.drained = True
        except BaseException as error:
            errors.append(error)
    if failure is not None and held.drained and not any(h.poisoned or h.quarantine for h in hashers):
        traceback.clear_frames(failure.__traceback__)
    if held.fd is not None:
        fd, held.fd = held.fd, None
        try:
            os.close(fd)
        except BaseException as error:
            errors.append(error)
    return errors


def open_stage(context, held, cdll, maps, body_started, extra=()):
    """Admission shared by smoke and full: Torch + frozen device + both original serializers + one authenticated library
    bound through the checked ABI. guard() = Source + Locks + extra FILE checks + resources."""
    import ctypes
    if held.torch is None:
        import importlib
        held.torch = importlib.import_module('torch')
    torch = held.torch
    if cdll is None:
        cdll = ctypes.CDLL
    authority, record = context.authority, context.authority.record
    context.source.check()
    context.locks.check()
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
        context.source.check()
        context.locks.check()
        for check in extra:
            check()
        return resources(torch, body_started)
    guard()
    held.fd, lib, held.mappings = open_library(authority, cdll, maps)
    function = bind_abi(lib, ctypes)
    return torch, device, function, [p[0] for p in prints], guard


def smoke_body(context, held, hashers, cdll, maps, body_started):
    torch, device, function, fingerprints, guard = open_stage(context, held, cdll, maps, body_started)
    result = smoke_checks(torch, device, function, fingerprints, guard, hashers)
    guard()
    return result


def run_stage(stage, body, context, output, torch, cdll, maps):
    """The one lifecycle: body, then the independent exit on EVERY path, then (only if all passed) the diagnostic receipt."""
    body_started = time.perf_counter()
    held = SimpleNamespace(torch=torch, fd=None, mappings=None, final=None, drain_tried=False, drained=False)
    hashers, failures, result, receipt = [], [], None, None
    try:
        try:
            result = body(context, held, hashers, cdll, maps, body_started)
        except BaseException as error:
            failures.append(error)
        failures.extend(exit_pass(context, held, maps, body_started, stage))
        if not failures:
            record, (checks, native_calls) = context.authority.record, result
            receipt = {'schema': SMOKE_RECEIPT if stage == 'smoke' else FULL_RECEIPT,
                       'status': 'SMOKE_DIAGNOSTIC_UNREVIEWED' if stage == 'smoke' else 'FULL_DIAGNOSTIC_UNREVIEWED',
                       'engineering_only': True,
                       'authority': context.fact, 'library': record['library'], 'device': record['device'], 'checks': checks,
                       'native_calls': native_calls, 'resources': held.final, 'resource_policy': POLICY,
                       'mappings': sorted(held.mappings), **{flag: False for flag in FLAGS}, 'normal_terminal_required': True,
                       'invocation': {'argv': sys.argv, 'python': str(Path(sys.executable).resolve()), 'pid': os.getpid(),
                                      'optimize': sys.flags.optimize, 'torch': held.torch.__version__,
                                      'cuda': held.torch.version.cuda}}
            if stage == 'full':
                receipt.update(prior=context.prior, **{k: record['full'][k] for k in ('fixture', 'extractor', 'metadata')})
            os.mkdir(output, 0o700)
            with open(Path(output) / (stage + '-receipt.json'), 'x') as stream:
                json.dump(receipt, stream, sort_keys=True, indent=1)
                stream.flush()
                os.fsync(stream.fileno())
            context.locks.check()  # publication is inside the whole cap; a failure here leaves the receipt but fails the run
            resources(held.torch, body_started, final=True)
    except BaseException as error:
        failures.append(error)
    finally:
        failures.extend(teardown(held, hashers, failures[0] if failures else None))
        if failures:
            requests.raise_failures(failures)
    return receipt


def smoke(context, output, torch=None, cdll=None, maps='/proc/self/maps'):
    return run_stage('smoke', smoke_body, context, output, torch, cdll, maps)


# ---------------------------------------------------------------- full synthetic inventory stage
PAIRS = 3
CLOCK = ('time.perf_counter around each complete arm of a sequential workload (frozen tree, digest validation, full tree). '
         'Candidate inside the clock: leaf traversal, validation, pinned table construction and H2D transfers, launch, '
         'stream completion, readback, hexadecimal conversion, original framing. Original inside: per-leaf D2H, streamed '
         'hashlib, original framing. Outside the clocks: guard() source/lock/resource/FILE rehash checks and a device drain '
         'before each arm. Descriptive observation only; no threshold, p99, GO, quality or product claim.')


def load_original(raw, label, functions=(), assigns=()):
    """ORIGINAL top-level functions/constants exactly as written, extracted by AST from authenticated bytes."""
    nodes = [n for n in ast.parse(raw).body
             if (isinstance(n, ast.FunctionDef) and n.name in functions) or
             (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name) and
              n.targets[0].id in assigns)]
    names = sorted(n.name if isinstance(n, ast.FunctionDef) else n.targets[0].id for n in nodes)
    require(names == sorted((*functions, *assigns)), 'exactly one original %s definition each required' % label)
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<original %s>' % label, 'exec', dont_inherit=True), namespace)
    return namespace


def leaf_nbytes(shape):
    return 4 * math.prod(shape)


def build_inventory(record):
    """Host only. The ORIGINAL expected_vision(config, model) over the authenticated extractor and upstream metadata gives
    the leaf names/shapes (exact 'vision_model.' prefix removed); the authenticated fixture and the PROBE/MLP roles of BOTH
    original serializers must agree with it exactly. No Torch, no allocation."""
    ext, sources = record['full'], record['sources']
    fixture = read_json(ext['fixture'], FIXTURE, FIXTURE_KEYS)
    metadata = requests.strict_json(read_bytes(ext['metadata'], JSON_LIMIT))
    require(type(metadata) is dict and type(metadata.get('config')) is dict and type(metadata.get('model')) is str,
            'upstream metadata with config and model required')
    for key, bound in (('metadata', ext['metadata']), ('source', ext['extractor'])):
        item = fixture[key]
        require(type(item) is dict and item.keys() == {'path', 'sha256'} and item['sha256'] == bound['sha256'] and
                type(item['path']) is str and Path(item['path']).name == Path(bound['path']).name,
                'fixture is not bound to the authenticated %s FILE' % key)
    original = load_original(read_bytes(ext['extractor'], 2 * 1024**2), 'extractor', ('expected_vision',), ('PREFIX', 'PROFILES'))
    expected, resolved = original['expected_vision'](metadata['config'], metadata['model'])
    prefix = original['PREFIX']
    require(prefix == 'vision_model.' and all(n.startswith(prefix) for n in expected), 'exact vision_model. prefix required')
    shapes = {n[len(prefix):]: tuple(shape) for n, shape in expected.items()}
    sizes = {n: leaf_nbytes(shape) for n, shape in shapes.items()}
    require(len(shapes) == len(expected) == LEAVES and sum(sizes.values()) == BYTES and max(sizes.values()) == LARGEST_BYTES and
            all(type(d) is int and d > 0 for shape in shapes.values() for d in shape), 'frozen synthetic inventory facts differ')
    require(fixture['model'] == metadata['model'] and canon(fixture['resolved']) == canon(resolved) and
            canon(fixture['shapes']) == canon({n: list(shape) for n, shape in shapes.items()}) and
            (fixture['leaves'], fixture['bytes'], fixture['largest_leaf_bytes']) == (LEAVES, BYTES, LARGEST_BYTES) and
            fixture['synthetic'] is True and fixture['quality_read'] is False, 'fixture differs from the ORIGINAL expected_vision')
    probe = load_original(read_bytes(sources['probe_serializer'], 2 * 1024**2), 'probe roles', assigns=('PROBE', 'PROBE_SHAPES'))
    mlp = load_original(read_bytes(sources['mlp_serializer'], 2 * 1024**2), 'MLP roles', assigns=('MLP', 'MLP_SHAPES'))
    roles = {'probe': tuple(probe['PROBE']), 'mlp': tuple(mlp['MLP'])}
    require(all(all(type(n) is str for n in r) and len(set(r)) == len(r) and set(r) <= shapes.keys() for r in roles.values()) and
            list(roles['probe']) == fixture['probe'] and list(roles['mlp']) == fixture['mlp'] and
            probe['PROBE_SHAPES'] == [list(shapes[n]) for n in roles['probe']] and
            mlp['MLP_SHAPES'] == [list(shapes[n]) for n in roles['mlp']] and
            (len(shapes) - len(roles['probe']), len(shapes) - len(roles['mlp'])) == (LEAVES - 1, LEAVES - 4),
            'PROBE/MLP roles differ from the inventory')
    return SimpleNamespace(shapes=shapes, roles=roles, largest=min(n for n, size in sizes.items() if size == LARGEST_BYTES))


def inventory_facts(inventory):
    sizes = {n: leaf_nbytes(shape) for n, shape in inventory.shapes.items()}
    return {'leaves': len(sizes), 'bytes': sum(sizes.values()), 'largest': inventory.largest,
            'largest_leaf_bytes': sizes[inventory.largest], 'roles': {k: list(v) for k, v in inventory.roles.items()},
            'frozen_leaves': {k: len(sizes) - len(v) for k, v in inventory.roles.items()}}


def leaf_seed(name):
    return int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], 'little') >> 1


def allocate_leaf(torch, device, generator, name, shape):
    """One separately allocated deterministic FP32 leaf; the seed depends on the leaf name only."""
    generator.manual_seed(leaf_seed(name))
    leaf = torch.empty(shape, dtype=torch.float32, device=device)
    leaf.normal_(0.0, 0.02, generator=generator)
    return leaf


def check_leaves(torch, device, inventory, leaves):
    names = list(inventory.shapes)
    require(leaves.keys() == inventory.shapes.keys(), 'allocated leaf names differ from the inventory')
    spans = plan_occurrences(torch, [leaves[n] for n in names], device)
    require(all(tuple(leaves[n].shape) == inventory.shapes[n] and leaves[n].storage_offset() == 0 and
                leaves[n].untyped_storage().nbytes() == size == leaf_nbytes(inventory.shapes[n]) and size > 0
                for n, (_, size) in zip(names, spans, strict=True)), 'leaf is not a separate exact allocation')
    ordered = sorted(spans)
    require(all(a + size <= b for (a, size), (b, _) in zip(ordered, ordered[1:])), 'leaf allocations overlap or alias')
    return spans


def host_digest(torch, leaf):
    """The ORIGINAL serializer's own streamed oracle: D2H, uint8 numpy view, hashlib over the memoryview; no tolist/bytes copy."""
    raw = leaf.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    return hashlib.sha256(memoryview(raw)).hexdigest()


def largest_gates(torch, device, hasher, function, fingerprints, guard, hashers, calls, big, name, fresh):
    """Correctness gates on the real-size largest leaf, before the whole inventory is even allocated."""
    nbytes, results = 4 * big.numel(), {}
    want = host_digest(torch, big)
    require(hasher.digests([big]) == [want], 'largest-leaf digest parity differs')
    for original in fingerprints:
        require(native_fingerprint(torch, hasher, original, {name: big}) == original({name: big}), 'largest-leaf typed digest differs')
    results['parity'] = {'bytes': nbytes, 'digest': want}
    guard()

    flat, version, seen = big.reshape(-1), big._version, {want}
    bytes_ = [0, nbytes // 2 + 1, nbytes - 1]
    for byte in bytes_:
        flip(torch, flat, byte, 0x40)
        changed = host_digest(torch, big)
        require(big._version == version and changed not in seen and hasher.digests([big]) == [changed],
                'same-version mutation digest differs')
        seen.add(changed)
        flip(torch, flat, byte, 0x40)
        require(big._version == version and hasher.digests([big]) == [want], 'restored largest-leaf digest differs')
    results['mutation'] = {'bytes': bytes_, 'mask': 0x40, 'version': version}
    guard()

    count = nbytes // 4
    views = [big, flat[1:4097], flat[1:4097], flat[4096:8193], big, flat[count - 1:], flat[:4096]]  # last: same address, shorter length
    wants, start = [host_digest(torch, view) for view in views], len(calls)
    require(hasher.digests(views) == wants, 'offset/duplicate/overlapping alias digests differ')
    require(hasher.digests(views[::-1]) == wants[::-1] and wants != wants[::-1], 'occurrence order differs')
    require([c[2] for c in calls[start:]] == [len(views)] * 2, 'one native call per occurrence list required')
    results['aliases'] = {'occurrences': len(views), 'reversed': True}
    guard()

    side = torch.cuda.Stream(device=device)
    with torch.cuda.stream(side):
        require(torch.cuda.current_stream(device).cuda_stream != 0, 'nondefault current stream required')
        require(hasher.digests([big]) == [want], 'side-stream digest differs')
    side.synchronize()
    producer, consumer, event = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device), torch.cuda.Event()
    with torch.cuda.stream(producer):  # the producer's write is still pending when the consumer is enqueued
        torch.cuda._sleep(200_000_000)
        flip(torch, flat, 3, 0x20)
        event.record(producer)
        require(not event.query(), 'producer write was not pending; inconclusive')
    with torch.cuda.stream(consumer):  # explicit event dependency, no host barrier; the oracle copy comes last
        consumer.wait_event(event)
        pending = hasher.digests([big])
        oracle = host_digest(torch, big)
    consumer.synchronize()
    producer.synchronize()
    require(oracle != want and pending == [oracle], 'producer-event dependent digest differs')
    flip(torch, flat, 3, 0x20)
    require(big._version == version and hasher.digests([big]) == [want], 'restored digest after the event dependency differs')
    results['streams'] = {'side': True, 'producer_event': {'byte': 3, 'mask': 0x20, 'distinct_streams': True}}
    guard()

    results['injected_failures'] = injected_failures(torch, function, hashers, lambda: [fresh()], [big], [want])
    guard()
    return results


def timed_arm(torch, run, trees, wants):
    torch.cuda.synchronize()  # outside the clock
    started, seconds = time.perf_counter(), []
    for tree, want in zip(trees, wants, strict=True):
        begin = time.perf_counter()
        got = run(tree)
        seconds.append(time.perf_counter() - begin)
        require(got == want, 'typed digest differs from the ORIGINAL fingerprint')
    return time.perf_counter() - started, seconds


def full_checks(torch, device, function, fingerprints, guard, hashers, inventory):
    calls = []

    def counted(*args):
        calls.append(args)
        return function(*args)
    hasher = Sha256Native(torch, counted)
    hashers.append(hasher)
    shapes, largest = inventory.shapes, inventory.largest
    names = list(shapes)
    results = {'inventory': inventory_facts(inventory)}
    generator = torch.Generator(device=device)
    tiny = pattern(16, 5)
    require(hasher.digests([device_leaf(torch, device, tiny)]) == [sha(tiny)], 'full warm-up parity differs')
    guard()

    big = allocate_leaf(torch, device, generator, largest, shapes[largest])
    results['largest_leaf'] = largest_gates(torch, device, hasher, function, fingerprints, guard, hashers, calls, big, largest,
                                            lambda: allocate_leaf(torch, device, generator, largest, shapes[largest]))
    del big
    leaves = {n: allocate_leaf(torch, device, generator, n, shapes[n]) for n in names}
    check_leaves(torch, device, inventory, leaves)
    guard()

    work = {label: (fingerprints[i], [{n: leaves[n] for n in names if n not in inventory.roles[label]}, dict(leaves)])
            for i, label in enumerate(('probe', 'mlp'))}
    work['largest'] = (fingerprints[0], [{largest: leaves[largest]}])
    expected, launches = {}, {}
    for label, (original, trees) in work.items():  # sequential frozen -> validate -> full, each vs the ORIGINAL on current bytes
        expected[label], start = [], len(calls)
        for tree in trees:
            digest = original(tree)
            require(native_fingerprint(torch, hasher, original, tree) == digest,
                    '%s typed digest differs from the ORIGINAL fingerprint' % label)
            expected[label].append(digest)
        launches[label] = [c[2] for c in calls[start:]]
        require(launches[label] == [len(tree) for tree in trees], 'one native launch per tree, no amalgamation or reuse')
        guard()
    for label in ('probe', 'mlp'):  # a role leaf is outside the frozen tree: only the full digest may change, and it must
        original, trees = work[label]
        leaf = leaves[inventory.roles[label][0]]
        flat, version = leaf.reshape(-1), leaf._version
        flip(torch, flat, 4 * leaf.numel() - 1, 0x01)
        require(leaf._version == version, 'same-version role mutation precondition differs')
        full_now = native_fingerprint(torch, hasher, original, trees[1])
        require(native_fingerprint(torch, hasher, original, trees[0]) == expected[label][0] and
                full_now != expected[label][1] and full_now == original(trees[1]), '%s role mutation digests differ' % label)
        flip(torch, flat, 4 * leaf.numel() - 1, 0x01)
        require(native_fingerprint(torch, hasher, original, trees[1]) == expected[label][1], '%s restored digest differs' % label)
        guard()
    results['workloads'] = {label: {'digests': expected[label], 'launches': launches[label]} for label in work}

    comparison = {}
    for label, (original, trees) in work.items():
        runs = {'original': original, 'candidate': lambda tree, original=original: native_fingerprint(torch, hasher, original, tree)}
        pairs, start = [], len(calls)
        for pair in range(PAIRS):
            row = {'order': ['original', 'candidate'] if pair % 2 == 0 else ['candidate', 'original']}
            for arm in row['order']:
                row[arm + '_seconds'], row[arm + '_call_seconds'] = timed_arm(torch, runs[arm], trees, expected[label])
                guard()
            pairs.append(row)
        require([c[2] for c in calls[start:]] == launches[label] * PAIRS, 'timed candidate launches differ')
        comparison[label] = {'trees': [len(tree) for tree in trees], 'pairs': pairs,
                             'isolated_mechanism_only': label == 'largest'}
    results['comparison'] = {'clock': CLOCK, 'pairs': PAIRS, 'workloads': comparison}
    return results, len(calls)


def full_body(context, held, hashers, cdll, maps, body_started):
    record = context.authority.record
    inventory = build_inventory(record)  # host only, before any CUDA work
    files = [record['full'][k] for k in ('fixture', 'extractor', 'metadata')]
    torch, device, function, fingerprints, guard = open_stage(
        context, held, cdll, maps, body_started, (lambda: [hash_file(item) for item in files],))
    result = full_checks(torch, device, function, fingerprints, guard, hashers, inventory)
    guard()
    return result


def full(context, output, torch=None, cdll=None, maps='/proc/self/maps'):
    return run_stage('full', full_body, context, output, torch, cdll, maps)


STAGE_BODIES = {'smoke': smoke, 'full': full}


# ---------------------------------------------------------------- CLI
def cli(stage, authority, unit, output):
    argv = [stage, '--authority', authority['path'], '--authority-sha256', authority['sha256']]
    if unit is not None:
        argv += ['--unit', unit['path'], '--unit-sha256', unit['sha256']]
    return argv + ['--output', output]


def parser():
    result = argparse.ArgumentParser(prog='qualify_cuda_sha256_native.py', allow_abbrev=False,
                                     description='Standalone CUDA SHA-256 native caller; smoke and full stages, root-run.')
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
    admitted = read_native_authority(authority, args.stage)
    prior = None
    if unit is not None:
        require(UNIT_READER is not None, 'genuine root-owned terminal unit reader required; a user-written GO is never accepted')
        prior = UNIT_READER(unit, admitted, PRIOR[args.stage])
    source = requests.Source(sys.modules[__name__], admitted.record['sources']['driver'])
    locks = requests.Locks(admitted.record['locks'])
    context = SimpleNamespace(authority=admitted, fact=authority, source=source, locks=locks, unit=unit, prior=prior)
    return STAGE_BODIES[args.stage](context, args.output)


if __name__ == '__main__':
    main()
