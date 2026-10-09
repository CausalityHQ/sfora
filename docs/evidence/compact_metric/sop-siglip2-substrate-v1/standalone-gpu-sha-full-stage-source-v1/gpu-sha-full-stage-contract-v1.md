# GPU SHA-256 FULL-stage source contract v1 (child sfora/gpu-sha-full-inventory-20261009)

Scope: scripts/qualify_cuda_sha256_native.py + scripts/test_cuda_sha256_native.py ONLY. Base e2014d50e9fd1d90d9a40437cf17ad2d0338d226 (ff-only
from f860e061). No C++/serving change. No Torch/NumPy/native/GPU/SSH here. Full suite and commit are HELD for root source review / serial slot.
Everything below is SOURCE ONLY; the full stage is UNRUN. All eligibility flags stay false; no GO, threshold, estimate, p99, quality or product claim.

## 1. Schemas

### 1.1 Full native authority (new, full stage only)
`cuda-sha256-native-authority-full-v1` = exactly the keys of `cuda-sha256-native-authority-v1` plus `full`.
Smoke continues to accept ONLY the v1 record with its exact v1 key set (a full record given to smoke, or a v1 record given to full, is rejected).
All v1 validation is unchanged (sources, build authority/receipt, library, runtime FILE provenance, mapping inventory, interpreter, device,
resource policy body300/whole1500/exit300/8GiB/zero-swap/CUDA<1e10, the two inherited locks).

`full` = exactly `{prior, fixture, extractor, metadata}`; `prior` = exactly `{unit, launch, verification}`. Every member is a FILE
`{path: canonical absolute regular file, sha256: actual}`:
- prior.unit: the ORIGINAL smoke-v2 `unit.json` (schema-less record, exact keys
  `{both_locks_held, invocation_id, log, native_peak_rss_kib, receipt, service_seconds, unit}`). It MUST equal the CLI `--unit`/`--unit-sha256` FILE.
- prior.launch: `launch.json` (`cuda-sha256-smoke-root-launch-v1`).
- prior.verification: the parent verification (`standalone-cuda-sha-smoke-parent-verification-v1`).
- fixture: `standalone-gpu-sha-full-fixture-v1/inventory.json` (`cuda-sha256-synthetic-vision-inventory-v1`, exact keys
  `{bytes, largest_leaf_bytes, leaves, metadata, mlp, model, probe, quality_read, resolved, schema, shapes, source, synthetic}`).
- extractor: scripts/extract_siglip2_vision_source.py bytes (source of the ORIGINAL `expected_vision`, `PREFIX`, `PROFILES`).
- metadata: `late-dense-v1/so400-native256-upstream-metadata-v1.json` (`config` + `model`).
All six are hashed at admission, are part of the distinct-FILE set (no FILE may be reused), are rehashed by every full-body `guard()` (fixture, extractor,
metadata) and again at exit (fresh whole-authority re-read, plus the whole prior chain is re-read from its FILEs and must equal the admitted facts).
No new native directory, mapping, library or pin is introduced. The prior smoke authority's closure FILEs must be BYTE-identical (sha256; paths may differ)
to the full authority's: build_authority, build_receipt, library, runtime_files (+provenance), mapping_inventory, interpreter, probe_serializer, mlp_serializer;
and device, resource_policy and locks must be equal. Fixture `metadata.sha256`/`source.sha256` must equal the bound metadata/extractor FILEs
(basenames equal; the repo-relative path in the fixture is not used).

### 1.2 Full receipt (written only after every check AND the complete exit pass)
`full-receipt.json`, schema `cuda-sha256-native-full-v1`, status `FULL_DIAGNOSTIC_UNREVIEWED`, engineering_only true, all FLAGS
(quality_read, quality_eligible, qualification_eligible, state_reuse_eligible, optimization_eligible, product_go, speed_go) false,
normal_terminal_required true. Same keys as the smoke receipt plus `prior` (the facts returned by read_prior), `fixture`, `extractor`, `metadata`.
`checks` = `{inventory, largest_leaf, workloads, comparison}`; the smoke receipt is byte-for-byte the same shape as before (file `smoke-receipt.json`).

## 2. Prior smoke unit reader (UNIT_READER = read_prior(unit, admitted, 'smoke'))
No `decision:GO` record is accepted (a user-written GO has the wrong keys). Authenticated from FILEs, fresh bytes, no cache:
1. unit.json equals the FILE frozen in the authority; exact keys; both_locks_held is true; invocation_id 32 lowercase hex; unit name; native_peak_rss_kib>0; service_seconds finite < 1500.
2. smoke receipt (exact 20 keys, schema cuda-sha256-native-smoke-v1, status SMOKE_DIAGNOSTIC_UNREVIEWED, engineering_only, normal_terminal_required, all FLAGS false);
   authority = the prior authority FILE; library/device/resource_policy equal; native_calls == 24; `checks` equal (type-exact) to the released smoke checks
   (raw_abi, parity 129 occurrences + 12 lengths, typed_trees 9/2 serializers, mutation, nondefault_stream_pending, 4 injected failures, 6 rejections; mutation.version is an observation);
   invocation `{argv, python, pid, optimize, torch, cuda}` with argv == [prior driver, smoke, --authority, authority path, --authority-sha256, sha, --output, launch output],
   python == interpreter, optimize 0; resources finite and inside body300/whole1500/8GiB/CUDA<1e10 (requests.check_resources), process_peak_rss_kib == unit.native_peak_rss_kib,
   receipt wall <= terminal wall <= service_seconds; mappings sorted/unique, contain the library and runtime files, subset of the frozen inventory.
3. terminal log: EXACTLY eight lines ending in one newline: `Running as unit: <unit>.service; invocation ID: <id>` (equal to the unit record), the bootstrap JSON
   (bootstrap_exit_pass and normal_outer_terminal_required true; cgroup events high/low/max/oom/oom_group_kill/oom_kill all 0; swap 0; peak 0<..<=8GiB; wall<1500),
   `Finished with result: success`, `Main processes terminated with: code=exited/status=0`, `Service runtime: N.NNNs` (== service_seconds), CPU time, Memory peak, `Memory swap peak: 0B`.
4. launch: schema, flags false, unit == unit record, native_authority == receipt.authority, output, receipt path == output/smoke-receipt.json; command exactly 23 items:
   systemd-run --user --unit=<unit> --wait --pipe --collect, the six properties (RuntimeMaxSec=1500 MemoryMax=8589934592 MemorySwapMax=0 TasksMax=128 KillMode=control-group OOMPolicy=stop),
   the four --setenv, an absolute python, `-I -B`, bootstrap path, bootstrap-authority path, its sha256, the bootstrap sha256.
5. bootstrap FILE hashed; bootstrap authority (`cuda-sha256-smoke-bootstrap-v1`): files {driver,test,requests,probe_serializer,mlp_serializer}, driver/test/serializers equal the prior authority's sources,
   native_authority, output, interpreter equal; helper hashed.
6. prior native authority (v1 exact keys) read from the receipt's FILE; its driver and test source bytes hashed; closure identical to the full authority (see 1.1).
7. parent verification: schema, terminal == unit record, outer_resources == footer resources, all_smoke_checks/complete_current_mapped_file_hashes_exit_pass/engineering_only/exact_authority/original_helper_and_source_exit_pass true,
   product_go/speed_go false, exit_status 0.
Any shape error in a forged record is converted to ValueError. Forged, truncated, substituted (any FILE byte change fails its sha256), and failed (non-normal0, non-success, nonzero events, any check flipped) units reject.

## 3. API (new/changed names)
- `read_native_authority(fact, stage='smoke')`, `read_full_extension`, `read_keys`, `read_json` (message unchanged).
- `read_prior`, `prior_facts`, `read_footer`, constants `UNIT_KEYS LAUNCH BOOTSTRAP VERIFICATION SMOKE_CHECKS SMOKE_NATIVE_CALLS ...`; `UNIT_READER = read_prior`.
- Shared lifecycle: `open_stage(context, held, cdll, maps, body_started, extra=())` (Torch + frozen device + both original serializers + one authenticated library + checked ABI + guard),
  `run_stage(stage, body, ...)` (body -> independent exit on EVERY path -> receipt), `exit_pass(..., stage)`, `injected_failures(...)` (shared by smoke and full), `smoke_body`, `smoke`, `full_body`, `full`.
  Smoke observable semantics, clocks, guards, exit and receipt are unchanged.
- Full: `load_original` (AST extraction of the ORIGINAL top-level function/constant definitions), `build_inventory(record)`, `inventory_facts`, `leaf_nbytes`, `allocate_leaf`, `check_leaves`,
  `host_digest`, `largest_gates`, `timed_arm`, `full_checks`.
- `STAGE_BODIES = {'smoke': smoke, 'full': full}`; `timing` has NO body and fails closed ("unreleased scaffolding"). `main` admits the authority for the stage, then (full only) UNIT_READER, then Source/Locks.

## 4. Full-stage checks (order)
1. Host only: `build_inventory`. ORIGINAL expected_vision(config, model) (AST from the authenticated extractor) over the metadata; exact `vision_model.` prefix removed; 448 leaves, 1711552256 bytes, largest 19832832 B ([4304,1152]);
   fixture agrees; PROBE/PROBE_SHAPES (probe serializer) and MLP/MLP_SHAPES (MLP serializer) roles agree with the fixture and shapes; frozen counts 447 / 444.
2. Admission, device and library as smoke (open_stage). Tiny warm-up parity.
3. Real-size largest leaf (separately allocated, seeded Generator.normal_), BEFORE the whole inventory: raw and typed (both serializer copies) parity vs the ORIGINAL streamed host digest;
   three same-version byte mutations (first, middle+1, last byte) with exact restoration; offset/duplicate/overlapping/tail aliases, one native call per occurrence list, reversed order; nondefault current stream;
   explicit producer-stream write -> Event -> consumer `wait_event` with the candidate run BEFORE the oracle copy and no host barrier; retained injected status/launched/completed/readback failures
   with checked drain, released owners, healthy follow-up.
4. All 448 leaves allocated separately (exact storage bytes, unique, non-overlapping; total bytes) then, per workload, sequential and each tree compared with the ORIGINAL fingerprint of the same current bytes:
   probe 447 frozen -> validate -> 448 full (PROBE serializer); MLP 444 frozen -> validate -> 448 full (MLP serializer); isolated largest. Exactly one native launch per tree ([447,448], [444,448], [1]); never amalgamated,
   deduplicated, cached or reused. A role-leaf same-version mutation changes ONLY the full digest (frozen unchanged), equals the ORIGINAL, and restores.
5. Only then PAIRS=3 alternating original/candidate arms per workload (order swaps each pair); every arm's digests are compared with the ORIGINAL; CLOCK documents what is inside (traversal, validation, table
   construction/H2D, launch, stream completion, readback, hex, framing) and outside (guard() source/lock/resource/FILE rehash, device drain) the clocks. No threshold/projection/p99/GO.
6. guard() (Source + Locks + fixture/extractor/metadata rehash + resources) between every group; the complete uncached exit (drain, mapped-file hashes, fresh whole authority + whole prior chain, Source, Locks, fd close, final resources) runs on every path.
Oracle: the ORIGINAL serializer's own D2H + uint8 numpy memoryview hashlib; no tolist, no bytes copy of a leaf.

## 5. Source inverse (existing assertions that change)
Only `CliTests.test_full_and_timing_are_unreleased_scaffolding_and_ignore_user_go_units` is inverted narrowly: UNIT_READER is the genuine reader, `set(STAGE_BODIES) == {'smoke','full'}`, `timing` stays
unreleased, user-written GO units are rejected for full. All other existing smoke/source/AST/CLI/exit/lifetime tests are kept; the fake torch gains fast contiguous paths, a stream-queue hazard model,
a seeded Generator and lazily committed arenas (the existing suite became faster). No existing assertion is weakened.

## 6. Tests (fake/virtual only; narrow, <=15 s, AS 1 GiB)
Authority: full positive/negatives (each extension FILE hash, extra/missing keys, duplicates, smoke<->full schema swap). Prior reader: synthetic chain + the genuine committed smoke-v2 chain (path remap of the committed copies) positive,
and negatives on the genuine exact eight-line log / launch / verification / receipt / unit / bootstrap authority and on the synthetic chain (forged, truncated, substituted, failed, GO-file, closure mismatch).
Inventory: real fixture/extractor/metadata; tampered fixture/extractor/metadata/serializer roles. Stage: scaled inventory through the `build_inventory` seam (the 448-name case once, a small role-faithful case for defects),
real-size 19,832,832-byte virtual leaf (no tolist, no 1.7 GiB allocation), producer-event/stream mutants, stale-cache/dedup/amalgamation mutants, injected-error lifetime and quarantine, guard/exit drift of every extension FILE and prior chain FILE,
body/whole/exit caps, CUDA cap, locks, fd cleanup, no-receipt-on-failure. AST checks: no Assert; no tolist/bytes copy in the full oracle path; the rejected four-worker helper is not the comparator.

## 7. Unverified native assumptions (cannot be tested here)
torch.Generator(device).manual_seed + Tensor.normal_(generator=) determinism and no host copy; untyped_storage().nbytes() == requested bytes; reshape(-1) view and `.data` same-version mutation; Stream.wait_event ordering;
torch.cuda._sleep/Event.query pending proof; the 19.8 MB single-occurrence kernel time inside the 300 s body clock. A native pass would still establish only this engineering stage.
