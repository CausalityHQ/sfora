**Verdict: conditional source GO. The science and provenance hold up. Two small fixes are needed before commit or freeze. Native memory fit is untouched by this change and still unproven.**

I checked HEAD `801b567a`, the two uncommitted scripts, the frozen base, both lifetime critiques, the reconciliation and the three `/tmp` logs. I edited no repo files and ran no repo tests or native code. I did run one generic Python snippet in `/tmp`, described under B1.

## Blockers before commit/freeze

### B1. The cleanup check on the failure path can never pass, so it skips the state check and reports a leak that isn't there
- **What happens:** when the error is raised inside a function the helper calls, that function's stack frame stays alive in the traceback. Any tracked tensor it was given stays alive with it. This covers `torch.autograd.grad` (`out` = loss), the model forward (`pixel_values`), `loss_terms` (`raw`) and `route_close` (`right` = the replay ranking gradient).
- **The effect:** inside `release_pass`, the weakref check fails. The payload/state check after it is skipped. The original error does survive, but it gets the note "graph lifetime survived release", which reads like a memory leak.
- **Already present in the tests:** the existing `total` fault and `probe(fault=True)` cases hit this path today. They pass only because `rejects` checks `str(error)` and never looks at notes.
- **Evidence:** I measured the mechanism in a standalone Python 3.14 snippet. Without clearing frames the tracked object was still alive after release (`True`); with `traceback.clear_frames` it was gone (`False`).
- **Memory is still freed later:** `arm_run` already calls `traceback.clear_frames(primary.__traceback__)` at `scripts/train_siglip2_rank_routed_mlp.py:2047-2049`, so this is a wrong check and a wrong label, not a native leak.
- **Fix (about 2 lines):** in the `finally` of `route_full_reference`, add `if primary is not None: traceback.clear_frames(primary.__traceback__)` before `release_pass()`. That is the same pattern `arm_run` uses; the helper's own running frame is skipped automatically.
- **Test:** in the sequential falsifier's `total` mode, catch the error inline and assert `not getattr(error, '__notes__', None)`. This should fail now and pass after the fix.

### B2. The new replay checks and the cross-graph comparison are never broken by a mutant
- A grep of the test file finds zero hits for these messages:
  - `full B64 replay raw differs`
  - `replay ranking/membership differs`
  - `full encoder ranking-only`
  - `full ranking encoder gradient absent`
- The encoder comparison between graph 0's total and graph 1's rank gradient carries the science of this redesign. A mutant that compares the ranking gradient with itself, or drops that loop, passes every layer.
- **Fix (about 10 lines):** add more `injected` modes on graph 1 (`forward_id == 1`), each expecting its own message, the RNG restored, and `len(graphs) == 2`:
  1. Perturb `raw` in the `connected` wrapper.
  2. Wrap `loss_terms` and change the rank value, and separately `selected['active']`.
  3. Scale the rank-gradient oracle while the forward facts stay identical.

## Should-fix (labels only, no change to science)
- **S1. Pixels and features share one replay check and one message.** Pixels are checked only after the roughly 20 s forward, together with features, so a loader failure looks the same as kernel nondeterminism (Opus B2). Fingerprint pixels right after `torch.cat` and give each check its own message. This costs no memory: the CPU fingerprint is a view, not a copy (`detach().cpu().contiguous().view(uint8).numpy()`).
- **S2. The static observer contract no longer covers this helper.** `observer_source_contract` now parses the restored old source, because the new helper's three conditional phase names and its `range(2)` loop would fail its rules. The pinned bytes of the helper (SHA `89123971`, which matches the current file) plus the runtime 24-phase sequence check cover it. I enumerated the paths: at most 12 records per call on any path, so 119 ≤ 128 is correct. The 113 − 18 + 24 arithmetic is hand-derived; optionally add an AST assert on the live helper (9 call sites, names drawn from the fixed 12).
- **S3. `K == replay['K']` proves nothing.** Both passes receive the same argument. The membership part of `selected` is a pure function of batch and bank, so the real replay evidence is `active` plus the rank fingerprint. Don't cite K equality.
- **S4. Minor cost.** The `finally` calls `release_pass` a third time on success, adding one payload fingerprint per view (~1–1.4 s by Opus's estimate). Harmless.

## Verified sound (from source)
- **One encoder backward per graph, without retaining the graph:**
  - The regression gradient comes from detached features.
  - Original A/C requests only A and C, so autograd skips the encoder nodes.
  - Order on graph 0 is regression (retained) → original A/C (retained) → total (released).
  - The RED log is the old helper failing at its second encoder backward on the same graph; GREEN passes.
- **Original A/C provenance:** the total-versus-original A/C check, nonzero rank encoder gradients and the existing tolerances are unchanged. The microbatch comparisons downstream still use `result['gradients']`.
- **Replay:**
  - The RNG is restored before pass 2.
  - The payload is checked again between passes, so a state change fails before any replay.
  - The features, raw output and rank scalar must match exactly by fingerprint.
  - `selected` holds only Python lists and ints, so `deepcopy` and `==` are safe.
- **Ownership:** every created tensor gets a weakref at creation. `selected`, `facts` and `replay` hold no tensors. `result` is cleared on failure, and the original error is preserved via `add_note`.
- **Exact inverse:**
  - HEAD's trainer equals the frozen base `2fb4f237`, and that base file is tracked in git.
  - The helper slice SHA matches the current file.
  - The rest of the file is pinned through the full SHA of the restored file.

## Native and allocator risk (my inference, not source blockers)
- **Bitwise replay is unproven.** Exact replay assumes the CPU kernels give bitwise-identical results on freshly allocated buffers. Some BLAS paths depend on memory alignment (Intel MKL ships a reproducibility mode, CNR, for exactly that reason), and this repo has no B64 determinism evidence. The code fails closed, which is correct. If it fires, record it as a determinism finding; don't add a tolerance.
- **More peak events.** The run now does 4 B64 forward peaks instead of 2. The augmented view's retained memory and the 276 micro events after the full reference are unchanged. A fresh CPU gate is therefore a localisation experiment. Write down per-boundary predictions before launch:
  - events at canonical `full_total_gradients`
  - `memory.current` at `full_pass_released` versus `replay_pass_released`
  - events in the augmented view
- **Still missing:** measured observation byte totals, any evidence of B64 determinism, and observation of whether the allocator actually releases memory.

I saved this verdict to memory as `sfora-sequential-b64-reference-source-review.md`.
