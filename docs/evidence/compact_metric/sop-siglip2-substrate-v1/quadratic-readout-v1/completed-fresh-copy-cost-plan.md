**GO for one bounded change: batch fresh CUDA tensor bytes within each fingerprint call. Qualification remains NO-GO until a fresh TRAIN1000 endpoint completes.** The profile supports this intervention; it does not establish that it will fit 300 seconds.

I authenticated all four profile-result files and inspected the complete pstats: 10,649 function records and 25,322 caller edges, without stripping filenames. Trainer, primitive and frame-helper hashes match the original source-v5 freeze.

The measured caller ledger is:

| Full caller/function key | Calls | Profiled seconds |
|---|---:|---:|
| `/home/riomus/runs/sfora-native256-source-cpu-v4/extract_siglip2_vision_source.py:77:stream_range` → buffered `readinto` | 38,759 | 24.152953 self |
| `/home/riomus/runs/sfora-native256-adaptation-source-v6/train_siglip2_substrate_adaptation.py:102:bound_file` → buffered `readinto` | 69,060 | 17.585515 self |
| `/home/riomus/runs/sfora-so400-genuine-view-train-source-v1/train_siglip2_genuine_views.py:146:bound_file` → buffered `read` | 37,614 | 11.362159 self |
| `/home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py:107:bound_file` → buffered `read` | 10,776 | 5.868161 self |
| `/home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py:1050:update` → `integrity` at line 916 of the same full filename | 52 | 3.340989 cumulative |
| `/home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py:916:integrity` → `check_complement` at line 911 of the same full filename | 60 | 3.153233 cumulative |
| `/home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py:911:check_complement` → `/home/riomus/runs/sfora-so400-quadratic-readout-source-v5/quadratic_encoder_frames.py:95:fingerprint` | 60 | 3.151194 cumulative |
| `/home/riomus/runs/sfora-so400-quadratic-readout-source-v5/train_siglip2_quadratic_readout.py:1050:update` → `/home/riomus/runs/sfora-so400-quadratic-readout-source-v5/quadratic_encoder_frames.py:95:fingerprint` | 26 | 2.128187 cumulative |
| `/home/riomus/runs/sfora-so400-quadratic-readout-source-v5/quadratic_encoder_frames.py:1062:visit` → `Tensor.cpu` | 2,676 | 3.677849 self |

Whole-profile buffered reads consumed 59.670496 self seconds; SHA updates consumed 20.545423. Receipt phase clocks place admission at 66.490091 seconds, native admission at 23.169315, and fresh exit rehash at 19.906452. These are principally invocation-level costs. The repeated update path instead performs two fresh complement checks and one complete-state fingerprint per update. Its 26 updates consumed 6.828391 cumulative seconds.

Cumulative windows overlap and must not be summed indiscriminately. `.cpu()` includes copying and synchronization waits; it does not identify GPU kernel time, transferred bytes or avoidable synchronization separately. Generated serializer bodies also share pstats filename/line/name keys, making their individual visitor totals misleading. The wrapper and native-method caller edges above provide the useful attribution. No TRAIN1000 forecast follows from these observations.

**The minimal implementation boundary is [quadratic_encoder_frames.py](/home/rb/worktrees/sfora-positive-causality/scripts/quadratic_encoder_frames.py:49), plus one differential test in the existing test file.**

1. Extend `seal`’s adapted serializer so each `fingerprint(value, consumed=None)` invocation gathers CUDA tensor occurrences in canonical traversal order. Form contiguous flattened **uint8 views**, concatenate by exact device, and perform at most one blocking `.cpu()` per nonempty device buffer.
2. Feed each occurrence’s byte slice into the original tensor branch. Preserve its **individual SHA256**, original dtype and shape, and every original typed frame. Hashing the concatenated buffer as one tensor would be incorrect.
3. Keep snapshots local to that invocation. Repeated aliases receive separate occurrences; no pointer/version memo, retained tensor facts or cross-boundary reuse. CPU tensors and calls with `consumed` retain the existing path, preserving checkpoint-page consumption.
4. Preserve capsule ownership checks and the pinned `a1684917…b543` original serializer. Trainer integrity boundaries, optimizer, RNG, math, source admission and complete fresh exit traversal remain intact.

This targets demonstrated recurring cost without replacing mandatory provenance validators. The profile identifies expensive admission reads but does not prove a removable duplicate set with equivalent predicates, inventory and start/exit semantics.

The bounded falsifier is:

- **Stdlib first, ≤5 seconds:** add `ContractTests.test_fingerprint_batches_fresh_cuda_bytes`, using the existing fake-torch approach. Compare complete fingerprints against the untouched original serializer for mixed CPU/CUDA leaves, mixed dtypes, empty/scalar tensors, repeated aliases, tuple/list and integer/string-key distinctions. Assert transfer counts, identical `consumed` order, all foreign/replaced/conflicting capsule rejections, and changed fingerprints after unchanged-version `.data` mutations between calls. Check both consecutive integrity boundaries independently.
- **Root-only native comparison, one discarded diagnostic:** retain full admission and fresh exit, both locks, 8GiB/noSwap/events0 and CUDA allocated `<10GB`; whole-service cap 300 seconds, comparison body ≤10 seconds. Use identical admitted live tensor values and equivalent owned metadata capsules for baseline v5 and proposed serializers. Run eight alternating paired samples of **two complement fingerprints plus one complete-state fingerprint**. Require exact digest equality and a lower paired median for the proposed path. Include real strided tensors and `.data` mutation rejection. A mismatch, resource failure, body timeout or no median improvement ends this intervention. This diagnostic remains permanently ineligible for qualification or state reuse.

The main risk is that `.cpu()` time reflects bulk bandwidth or pending GPU work rather than many avoidable waits. Concatenation adds a device copy and temporary device/host storage, so it could be slower. Incorrect slice offsets, traversal order or byte reinterpretation could corrupt canonical hashes. The differential test addresses framing; native checks must establish actual byte equivalence and cost.

If the falsifier passes, freeze the new exact four-file closure, run **fresh CPU120 → control mechanics300 → candidate mechanics300 → fresh control061 TRAIN1000/300 once**. Every original predicate, replay, reload, fresh exit and whole-service terminal must pass. Only accepted control completion admits candidate061.

Any failure closes this correction as engineering **NO-GO**. Preserve the 531/666/705 timeouts; do not resume partial states or alter caps. Accepted paired endpoints then enter the unchanged selection/validation gates and subsequent full production SOP+InShop quality and matched public-speed qualification. A transfer improvement alone satisfies none of those goals.

No files were edited, native packages imported or jobs launched. The workspace advanced to `a93bde52` during inspection through evidence-document changes; the frozen source bytes remained unchanged.
