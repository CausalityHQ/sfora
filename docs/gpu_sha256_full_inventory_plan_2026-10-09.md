Proceed with **one standalone harness and a two-stage build/load authority freeze**. Keep serving unchanged and wait for root to collect SSH54145 before releasing native work.

The inspected files are unchanged from `d74365e`, despite HEAD advancing during inspection. The CUDA source still hashes to `f4547f923c5cb3acf893ffc1186e8392c343a6e27951ea3812b430993e621fd1`.

1. **Implement only the missing caller and its tests.**

   Add `scripts/qualify_cuda_sha256_native.py` and `scripts/test_cuda_sha256_native.py`. Keep the CUDA primitive unchanged unless compilation or a falsifier identifies a defect.

   The harness owns admission, explicit `ctypes` ABI binding, tensor validation, device pointer/length tables, current-stream launch, checked completion/readback, and cleanup. Before enqueue, require contiguous CUDA FP32 leaves on one active device, valid spans and capacities, contract-compliant lengths, and separately allocated output. Reject transposes, stride-zero views, other dtypes and CPU leaves before launching anything. No gather or fallback is needed for this first scope.

   Reuse the **original `fingerprint`** for outer framing. Its existing `frozen.get(key)` seam can consume a private, per-call occurrence cursor containing freshly computed `(dtype, shape, lowercase_digest_hex)` facts. The cursor must verify each metadata key, consume exactly one result per occurrence, and reject missing, extra or reordered results. It must never index results by pointer or retain them between calls. This preserves duplicate hashing without introducing another serializer.

   I verified AST equality between the [probe serializer](/home/rb/worktrees/sfora-positive-causality/src/sfora/connected_probe_inference.py:740), MLP serializer and original training serializer.

2. **Freeze build inputs, then separately admit the produced binary.**

   Put the prospective artifacts under `docs/evidence/compact_metric/sop-siglip2-substrate-v1/standalone-gpu-sha-native-v1/`:

   - `source-manifest.json`: CUDA source, contract, source test, new harness/test, original serializer sources, reused admission helpers, fixture source and configuration.
   - `build-authority.json` and `compile.sh`: canonical tool paths, hashes, exact environment, compiler arguments, target, resource ceiling and exclusive output directory.
   - `build-receipt.json`: actual exit status, command, versions, logs, ELF hash, extracted device-image hashes and dependency inspection.
   - `native-authority.json`: explicit executable/runtime FILE closure, binary receipt, GPU identity, stage, resource policy and oracle identities.

   Root first authenticates the supplied nvcc path, adjacent headers and `/usr/bin/g++`, resolving symlinks and recording actual version/hash evidence. Include nvcc configuration, invoked CUDA subtools, host compiler/linker, consumed headers/libraries and inspection tools—not merely `nvcc` and `cuda_runtime.h`.

   Compile once with explicit host compiler, optimization/PIC/shared-library flags, explicit CUDA runtime selection, and **only `code=sm_<verified target>` device output**. Determine the target from authenticated spark-2751 evidence; do not guess it from the hostname. Inspect the resulting fatbinary to require the intended SASS image and no embedded PTX fallback. Compiler-generated intermediate PTX is build evidence, not permission for runtime JIT.

   Freeze the actual `libcudart`, driver-side dependencies, interpreter, Torch/NumPy and `hashlib`/crypto oracle closure. Existing directory trust or `DT_NEEDED` names are insufficient.

3. **Use a finite loader; do not extend the Cutile grant implicitly.**

   [CombinedAuthority](/home/rb/worktrees/sfora-positive-causality/scripts/connected_control_native_authority.py:111) hardcodes the archived Cutile schema, build and binary. Reuse its FILE/provenance, fresh-hash and mapped-inode checks as patterns; this SHA library needs its own authority.

   Hash an opened regular-file descriptor, load that same authenticated inode, retain its canonical file through process exit, and check mapped device/inode identities and the complete permitted mapping set before and after execution. Reject unexpected dependencies and changed/deleted files.

   Do **not** copy [the factorized loader](/home/rb/worktrees/sfora-positive-causality/src/sfora/factorized_residual_native.py:142) verbatim: it unlinks its temporary load alias, whereas the connected authority rejects deleted mappings. Bind all five argument types and `cudaError_t` explicitly. No runtime compilation, extension cache or blanket native-directory exception.

4. **Run the smallest first native falsifier before allocating the full inventory.**

   After root releases the frozen smoke authority, run one process on spark-2751, with a proposed **30-second smoke-body ceiling** inside the existing native resource envelope. Use less than 64 KiB of fixture data.

   Test:

   - Count zero with null arrays/output; oversized count and null nonzero top-level arguments.
   - FP32 byte lengths `0, 4, 52, 56, 60, 64, 68, 116, 120, 124, 128, 132`; asymmetric bytes and literal signed-zero/NaN payloads.
   - A contiguous offset view, duplicate and overlapping aliases, and **129 occurrences** to cross the 128-thread launch boundary.
   - A nondefault current stream with pending byte mutation, followed immediately by candidate hashing. **Run the candidate before the oracle copy**, with no preceding global barrier.
   - Same-version `.data` byte mutation and exact restoration; verify the unchanged version counter explicitly.

   Require raw digest equality with `hashlib` and complete typed-tree equality with the original serializer. A wrong digest, stale mutation result, ordering failure, missing occurrence, timeout or authority mismatch stops this stage. A pass establishes only this smoke gate.

5. **Complete correctness and error-lifetime gates before timing.**

   Add the **19,832,832-byte `[4304,1152]` leaf**, all admitted padding cases, nested typed trees, differing shapes with identical raw bytes, tuple/list and key-type distinctions, aliases and occurrence reordering.

   For streams, include default and nondefault streams, plus an explicit event dependency from another producer stream. Freeze input immutability during hashing; this API provides no atomic snapshot against concurrent mutation.

   For errors, distinguish:

   - Pre-enqueue rejection.
   - Launch-status failure.
   - Completion/readback failure after real work has been queued.

   Inject failures at the harness’s checked boundaries while real benign work remains outstanding. Require no digest publication, owners retained until a real checked drain completes, cleared retained references afterward, and a healthy subsequent call. Label injected errors honestly. If draining itself fails, quarantine ownership until process teardown; do not pretend a poisoned context supports recovery. Avoid deliberate illegal-memory-access kernels on the shared device.

   Add source-level negative tests for altered authority facts, binary identity, ABI binding, occurrence-cursor ordering and serializer framing. Reuse the existing source-check evidence while its inputs remain unchanged.

6. **Only then release the matched engineering comparison.**

   Generate one deterministic synthetic 448-leaf inventory using `expected_vision()` in `scripts/extract_siglip2_vision_source.py` and the pinned `late-dense-v1/so400-native256-upstream-metadata-v1.json`. Remove the extractor’s exact `vision_model.` prefix and verify the resulting names against the runtime PROBE/MLP roles. Allocate each parameter separately; do not substitute repeated aliases for the real workload.

   Measure separately:

   - Probe: **447 frozen → validate digest → 448 full**.
   - MLP: **444 frozen → validate digest → 448 full**.
   - Largest leaf alone, labelled as an isolated mechanism measurement.

   Use three alternating original/candidate pairs per workload. Include traversal, validation, table construction/transfers, launch, completion, readback, hexadecimal conversion and original framing. Record admission/build costs separately and disclose source checks outside the clock. Preserve occurrence order and both launches; no amalgamation, deduplication or reuse across calls.

   The comparator is original `fingerprint`, never `_fingerprint_cuda_dict`: the four-worker helper remains rejected at 6.05–6.11 seconds versus 1.57–1.60 seconds. Report observations and resources without adopting the projected thresholds.

**Hard blockers today:** no compiled artifact, verified toolchain/runtime closure, SHA-specific native authority, validated caller, or native stream/error-lifetime evidence exists. The inspected source does not establish a specific SHA arithmetic defect; compilation and execution remain unproven. Full integration would additionally reach both `encoder_facts` callers, extraction ledgers/tests and installed bridge provenance, outside this standalone slice.

Keep the existing body/whole-process limits, exit reserve, 8 GiB host ceiling, zero swap, CUDA allocation ceiling and lifetime locks. A standalone pass leaves B1/B32/tail parity, lifecycle/uncached exit, scientific gates and 10,000 interleaved paired-call p99 qualification unchanged.

Skipped: edits, SSH, compiler/native/GPU/image execution, test reruns and duplicate consultations.  
Risk: even byte-exact native results may be slower at the complete sequential-tree boundary.

## Current root release: source implementation only

The native smoke-v2 has now passed, as retained in standalone-gpu-sha-smoke-v2-result: normal0, invocatione594e58e8ad1495a8150757ca754e3fc, 24 native calls;129 raw occurrences;9 typed trees with both original serializers; mutation/pending-stream/injected-error checks; outer cgroup436899840B/events0/swap0, CUDApeak15872B. All eligibility flags false. Original smoke-v1 FAIL remains recorded.

Implement the smallest extension of the existing caller/tests for one full synthetic correctness plus descriptive comparison stage. Use standalone-gpu-sha-full-fixture-v1/inventory.json (448 separate leaves,1711552256 bytes, largest19832832B) derived from the exact original expected_vision(config,model). Preserve original source/model roles, separate447+448 PROBE and444+448 MLP boundary checks, original fingerprint baseline and existing clocks/resources/native source/locks/full exit. Never combine, cache or deduplicate. Keep timing stage unreleased; a full engineering receipt is not production optimization eligibility.

A genuine full-stage prior reader must authenticate actual smoke receipt+original log+launch+parent verification+source/native authority, exact normal0 invocation/footer/resources/locks and checks, NOT a decision:GO file. Bind fixture/extractor/config FILEs prospectively to this prior/extension authority and freshly rehash them on guards/exit. Do not infer qualification from smoke metadata. New future source/execution/unit hashes remain parent supplied. Root still owns native release, acceptance and integration. Before code publish exact schema/API/source inverse plan; full suite/commit held for source review.

