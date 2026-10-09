# Critique of the CPU-v2 two-pass full-reference lifetime proposal

**Verdict: HOLD. The science is fine, but the run's own evidence says this change will not get through the memory gate.** The proposal computes a genuine second gradient rather than deriving one, and it weakens no proof. But it targets only 228 of the 1,643 memory-limit events. Its extra work also lands where the log shows memory is not given back.

I checked HEAD `da86f1a5`, the source SHA `2fb4f237`, the proposal JSON, `original.log` (all 111 observation records) and `verification.json`. Nothing in the repo was edited and nothing was executed.

## Blocking findings

### B1. The memory argument covers less than a seventh of the violation, and the new work may make it worse

These numbers are deltas computed from `original.log`. "Reclaimed" is the `pgsteal_direct` delta, converted to MiB.

| Section | `max` events | Reclaimed | Changed by the proposal? |
|---|---|---|---|
| Canonical `total` gradient, after the retained rank backward | 0→228 | 53 MiB | Yes |
| Augmented `ranking`, the **first** encoder backward on a fresh graph | 228→842 | 112 MiB | No: pass-1 `total6` has the same structure |
| Augmented `total` | 842→1367 | 129 MiB | Yes |
| Augmented split and micros, with **no full graph alive** | 1367→1643 | 72 MiB | No |

- **At least 890 of the 1,643 events (614 + 276) are in sections the proposal leaves structurally unchanged.** Only the memory level they start from would differ.
- **Releasing the augmented graph freed almost nothing.** After release, settled memory went 5596→5584 MiB (anon 4961→4948). The same release in the canonical view went 5655→3828.
- **The augmented view also started higher**: 4459 MiB at `full_begin` versus 3648 for canonical.
- **The proposal repeats the pattern that held on to memory.** It adds a second B64 graph per view, built right after the previous one is released, and that is exactly the sequence after which memory stayed high. Canonical pass 2 may then behave like the augmented view did. Augmented pass 2 may start at about 5.58 GiB, and a B64 forward alone peaked about 3.75 GiB above its starting level in canonical (3731→7482 MiB). The risk points toward more events, not fewer.
- **There is one real argument in its favour.** Pass-1 `total6` starts from roughly the level the current `ranking` backward started from (5037 MiB, versus 5644 for the current `total`). That `ranking` backward peaked at 7825 MiB with no events, and the canonical overshoot was only about 53 MiB. So the canonical bracket plausibly clears. Nothing in the evidence predicts the same for the augmented view.
- **Missing evidence:** we can't tell whether the retained memory comes from the micro graphs interleaving with the B64 graph, or from any second B64 cycle. The allocator mechanism is unobserved; glibc arena or mmap-threshold retention fits the data but is unverified.
- **After the first event, `memory.peak` stays pinned at 8192 MiB**, so every later margin can only be read from event counts and `pgsteal`.

**Repair:** don't launch this as a fix. If a native unit runs, label it as a localisation experiment. Before it runs, write down predictions for each boundary:
1. Canonical pass-1 `total` adds 0 events.
2. Settled memory after canonical pass-2 release, compared with after pass 1. This one comparison separates the two retention hypotheses.
3. Events in the augmented view.

### B2. Pass-2 replay checks have a gap, and a failure would be mislabelled

- **What actually makes the replay identical:**
  - `pixels_for` already requires that global RNG is unchanged (`qualify_actual_objective_encoder_gradients.py:186-189`).
  - The augmentation is seeded per row.
  - The model is in `.eval()` (`train_siglip2_rank_routed_mlp.py:768`).

  Resetting the RNG is therefore only hygiene. Exact equality of features, raw and the rank scalar depends entirely on the B64 CPU kernels (SDPA, GEMM, aarch64) being bitwise deterministic within one process. **There is no evidence of that.**
- **Repair:**
  - Check the pass-2 pixel and feature fingerprints immediately after the forward, before `loss_terms` or any backward.
  - Give that failure its own message, e.g. "full B64 replay nondeterministic", so it can't be confused with a memory or science failure.
- **Missing predicate:** pass 2 must require `selected` (membership plus `active`) and `K` to equal pass 1. Rank-scalar equality alone is weaker. `loss_terms` returns `{**membership,'active':active}` (`train_siglip2_identity_diversity.py:1293`).

### B3. Error-path cleanup

- **No lifetime check on the error path.** `refs` is filled only on success (`train_siglip2_rank_routed_mlp.py:1366`), so when there is an error, freeing relies entirely on `temporary.clear()`.
- **Repair:**
  - Keep every tensor from both passes, including the pass-2 `original` MSE and the rank gradient tuple, inside a container that the `finally` block clears. Don't use plain locals: the traceback would keep them alive, which brings back the traceback-retention problem the cleanup fix addressed.
  - On a pass-2 failure, also clear the detached pass-1 `result` (about 40 MiB).
  - Keep the existing pattern at lines 1371–1379: RNG restore, then the payload fingerprint check, and never replace the primary error.
- **Inter-pass gate:** build the weakrefs when each object is created. Cover `pixels`, `features`, `raw`, `detached`, `original`, `mse`, `rank`, `loss`, `original_loss`, and the regression, original A/C and total tuples. "Weakref is dead" proves the Python objects are gone; it says nothing about memory being returned to the OS (the proposal already says this).

### B4. The proposed falsifier would wrongly flag the allowed calls, and it can't see the retain-graph mutant

- **Calls can't be classified by their member lists.** The regression call passes all six members with `allow_unused` (line 1342), yet it never traverses the encoder because its input is detached.
  - **Repair:** the stand-in should track reachability. Count an "encoder backward" only when an encoder member is requested **and** the output depends on the live features node.
- **The planned negative cases don't cover `retain_graph`.** A pass-1 `total` run with `retain_graph=True` would still pass the weakref gate after the clear.
  - **Repair:** the stand-in must reject `retain_graph=True` on any encoder backward, and require it on the earlier regression and original A/C calls.
- **Add a negative case** where pass 2 reuses the pass-1 pixel tensor. The inter-pass weakref gate must reject it.

## What is sound (no change needed)

- **Science:**
  - Pass 1 keeps three checks unchanged: regression gradients absent from the encoder (`None` or zero), `total` A/C equal to original A/C, and nonzero rank encoder gradients.
  - Comparing `total` (graph 1) with `rank` (graph 2) is a genuine independent computation. It is no weaker than the current same-graph check: because the MSE comes from detached features, the encoder gradient of `total` equals the rank gradient whenever the computation is deterministic.
  - The return shape and the full-versus-micro and actual-optimizer checks are unchanged. No gradient is derived.
- **Cost:** each B64 forward took about 19.9 s. Two extra forwards, plus pixels and an inter-pass payload fingerprint (about 1–1.4 s each), add roughly 45 s, bringing about 326 s to about 371 s against the 600 s cap. The number of encoder backwards is unchanged.
- **Observer budget:** the current run wrote 111 records totalling 115,664 bytes, with the largest at 2,415 bytes. Adding 12 records of about 1.8 KB each keeps the total near 137 KB, under 262,144. The count of 123 is under 128 but leaves only 5 spare. The worst-case count must include the release records from both passes on error exits.

## Smallest complete implementation

Change only `route_full_reference`:
1. Forward once and run pass 1: regression check, original A/C, then `total6` without retaining the graph.
2. Keep only the detached `total6`, the three scalars, and the fingerprints of pixels, features and raw. Clear everything else, check the weakrefs, restore the RNG and verify the payload fingerprint.
3. Forward again (pass 2), checking fingerprints, membership and `K` immediately. Compute `rank6` without retaining the graph, then compare.
4. Clear pass 2 and run the existing `finally` block.

This is about 30 changed lines. Don't add allocator controls or cap changes.

The verdict is saved to memory as `sfora-rank-routing-cpu-v2-lifetime-proposal-critique.md`.
