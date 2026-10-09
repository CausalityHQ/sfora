# Standalone GPU SHA-256 source contract — 2026-10-09

**Source prototype only. CUDA compilation, native execution, native authority,
performance, integration and release are UNRUN.** The root approved this finite
design before production writes. Base:
`6ea5e22be106b6dde0283a02094ecb1d0cad8c23`.

This work owns only `rust/sfora-cuda-sha256/sha256_occurrences.cu`,
`scripts/test_cuda_sha256_source.py`, and this document. There are no Cargo,
dependency, runtime, bridge, serializer, authority or serving-default changes.
The independently assigned serving rollback is outside this scope.

## Finite enqueue API

The source declares exactly one exported C ABI entry point:

```cpp
cudaError_t sfora_sha256_occurrences(
    const unsigned char* const* device_ptrs,
    const uint64_t* device_byte_lengths,
    uint64_t occurrence_count,
    unsigned char* device_output,
    cudaStream_t current_stream);
```

`device_ptrs` and `device_byte_lengths` are **device-resident arrays**, ordered
by leaf occurrence. Count is in `[0, UINT32_MAX]`. A nonzero count needs arrays
of at least that many entries and output capacity of at least `count * 32`
bytes. Output slot `i` is the 32-byte SHA-256 of exactly the byte span at entry
`i`, with the standard big-endian digest-byte order. Each call reads every
occurrence afresh. Identical pointers and overlapping input aliases still get
distinct output slots and distinct hashing work. Nothing is deduplicated,
cached or retained between calls.

Count zero returns `cudaSuccess` without launching or dereferencing anything;
null arrays/output are permitted in that case. The host entry point rejects
counts above `UINT32_MAX` and null top-level arrays/output for nonzero count.
It does **not** dereference device arrays to validate their contents.

Before calling, the caller must validate all of the following:

- Every occurrence represents a contiguous FP32 leaf's logical bytes, with
  byte length divisible by four and at most `UINT64_MAX / 8`. Scalar means four
  bytes; empty means zero bytes and permits a null leaf pointer. Nonempty
  leaves need a valid readable span. A contiguous offset view is permitted.
- Arrays, leaf spans and output reside on the active device and have the
  required capacities. Pointer/length arrays have their element alignment.
  Output cannot overlap any input leaf span or either input array. Input
  spans may overlap each other.
- The explicit stream is the caller's **current CUDA stream** on that device.
  Handle zero is accepted when it is actually the caller's current default
  stream. No implicit stream lookup, substitution or private stream exists.

Strided/noncontiguous views, other dtypes and invalid spans are outside this
API. This source supplies no gather, generic whitelist or fallback. It does
not infer contiguity or dtype from a raw pointer. Those obligations would have
to be enforced by a separately reviewed caller before any integration.

## Ownership, ordering and errors

The caller owns all tensors/storage, pointer and length arrays, output, and
any caller scratch. Keep every owner alive until error-checked completion and
readback. Input bytes and arrays must remain immutable while the kernel reads
them. Order preceding mutations on the supplied stream and establish explicit
dependencies on producers using other streams. Neither this API nor the
original serializer promises an atomic snapshot against concurrent mutation.

The function enqueues one logical thread per leaf and returns
`cudaGetLastError()` after launch. Success is an enqueue/launch status, **not**
completed or host-readable output. The caller must attribute/check prior CUDA
errors, check asynchronous completion, order and check digest readback, and
withhold all digests on any error. On a failure after work may have been
enqueued, retain owners through checked draining or safe device teardown;
returning an error does not grant permission to recycle live storage. A later
healthy call is part of the required native error-lifetime test.

The primitive allocates no buffers, compiles nothing, loads nothing and
synchronizes nothing. Thread-local state and message schedule are its only
scratch. SHA chaining is sequential within each leaf; parallelism is across
occurrences. This is the first source hypothesis. Register spills, occupancy,
divergence, load behavior and speed are unknown.

## Bytes and the original identity

All tensor input accesses are unsigned byte loads. There is no floating point
operation or conversion, numeric signed-zero normalization, NaN
canonicalization, native-word reinterpretation, or backing-storage-gap read.
Padding appends `0x80`, zero bytes and the original bit length in big-endian
order, including the two-block boundary. The source accepts null empty leaves
without reading their pointers. SHA message words and output use big-endian
ordering while input tensor bytes remain literal and unchanged.

A raw leaf digest is **not** `fingerprint(tensor)` or a complete tree digest.
The original `src/sfora/connected_probe_inference.py` fingerprint bytes are
untouched. Its `Tensor` tag, dtype, shape, lowercase hexadecimal leaf digest,
typed tuples/lists/dicts, `repr` key ordering and length framing remain host
responsibilities. No new framing implementation is introduced here.

## Source falsifier and its limit

Run the narrow stdlib-only check with a bounded address space and deadline:

```bash
timeout 30s bash -c 'ulimit -v 1048576; exec python3 scripts/test_cuda_sha256_source.py'
```

The test extracts the source constants and finite helper, input-word,
schedule, state initialization/register, round, chaining and output
expressions. A restricted integer-expression evaluator executes those
operations; Python supplies block/round iteration and occurrence orchestration.
Control-loop and kernel/launch snippets receive source-structure checks only.
Unsupported expression syntax fails rather than silently selecting another
implementation. This is a source falsifier, not a CUDA emulator or compiler.

Independent checks use `hashlib.sha256` for digest equality and integer square
and cube roots of independently enumerated primes for all IV/K constants.
Vectors include empty, `abc`, a published multiblock known-answer message,
every raw byte length from 0 through 256, deterministic longer bytes, scalar,
FP32 element counts 13–17 and 29–33 (52–68 and 116–132 bytes), literal signed
zero/infinity/NaN payload bits, offset spans, reordered occurrences, duplicate
aliases, byte mutation and restoration. No floats or Torch objects are used.
The non-multiple-of-four byte vectors exercise internal padding arithmetic;
they do not extend the FP32 caller contract.

Each of the 64 round constants and eight initial constants gets a wrong-bit
mutant whose digest must differ from the oracle. Additional mutants exercise
schedule indices, padding thresholds/marker/length endian, round updates,
initial register selection, chaining register order, rotate direction, input
word endian and output endian. These checks test whether the source falsifier
can detect those specific mistakes. A passing run cannot certify memory
safety, native integer lowering, the whole kernel's execution, launch admission,
stream/race behavior, alias lifetimes or performance. The occurrence tests
execute extracted leaf operations repeatedly on CPU views; they do not prove
that native occurrence dispatch reads fresh bytes.

## Root-owned gates — all UNRUN here

The committed research and two critiques under
`docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-gpu-fresh-sha-research-20261009/`
permit only this source prototype. Their original-result latency projections
and 0.35/0.7-second thresholds are not approved gates or predictions here.
The rejected four-worker CPU helper's 6.05–6.11 seconds versus original
1.57–1.60 seconds remains **REJECT**, not the benchmark baseline.

Before any native compile/load/launch, the root must freeze finite compiler
and toolchain identities, exact flags/target, source and resulting ELF/device
binary hashes, CUDA runtime/driver dependencies, and interpreter/oracle
identity. Exact SASS-only device code for the target, with no unpinned PTX JIT,
requires a new finite native-authority decision. An existing compiler/library
admission or trusted directory does not admit this source or its future binary.
No native build command, loader or authority exception is supplied here.

Root-owned native correctness work must cover byte-for-byte raw digests and
complete original typed-tree equality, all padding cases, the 19,832,832-byte
largest leaf, offset/duplicated/shared aliases, zero/scalar, fresh same-version
byte mutations and restoration, and the explicit treatment of transposed or
stride-zero views outside this API. Exercise pending writes on a nondefault
current stream without a concealing global barrier, other-stream dependencies,
race prevention, retained owners, launch/completion/readback errors, draining
and a healthy subsequent call. Validate the count-zero and invalid host
arguments natively as well. All are UNRUN.

Any later standalone engineering benchmark must preserve the actual full-tree
sequence: **447 frozen occurrences, check, then 448 full occurrences** for the
probe; separately **444 frozen, check, then 448 full** for the MLP. Never merge
the two dictionaries into one launch or deduplicate repeated leaves. Measure
the full completed wrapper/readback/framing boundary against the **original
serializer**, alongside the isolated largest leaf. Three alternating
engineering timing pairs, resource ceilings and any prospective stop rule
need separate root release; no speed claim follows from a single-leaf result.

Full-suite execution and commit stay on HOLD until the root's serial release
(at most 120 seconds and 1 GiB address space). Integrated B1/B32/tail parity,
resource/lifecycle/uncached-exit gates, protocol-matched quality and speed,
and 10,000 interleaved paired calls with a p99 confidence interval remain
unchanged root responsibilities. This source work grants no production GO.
