**Verdict: no exact source owner is proven for the failing pressure, so the deliverable is one bounded metadata observation contract, not a repair.** One science-neutral application residency is proven at the first bracket but is quantitatively insufficient, and I recommend it not be applied alone.

## Ownership trace (from `memory-boundaries.json` and the 117 `original.log` rows, source `69302dab`)

| Bracket (phase, view) | cur before → peak (GiB) | events | What is provably alive |
|---|---|---|---|
| pass-0 total backward, canonical (`route_full_reference` line 1396) | 4.95 → 7.51 | 0 | pixels 48 MiB, graph saved tensors, original_A_C, loss scalars |
| **pass-0 total backward, augmented (first event)** | 5.44 → 8.00 = cap | 105 (24 MB file reclaim) | identical owner set, identical shapes |
| replay ranking backward, augmented | 5.45 → cap | +27 | same |
| micro-0 `route_split` witness, augmented (B16) | 5.46 → cap | +229 (46 MB reclaim) | only micro graph, 12 MiB pixels |
| later micro grads + step | 5.46 | +63 | unchanged |

Facts that separate proven application ownership from unproven residency:

- **Shapes are view-independent.** Batch, K, members and graph structure are the same in both views, so the application's bracket owner set is identical. The difference is baseline only: augmented `full_begin` current is 4.42 GiB versus 3.57 canonical, and post-reference current is 5.46 versus 3.81.
- **Proven application delta between views is about 40 MiB.** Member `.grad` from the four canonical micro backwards plus `ranking_total`. Every per-view accumulator and every B64 or micro temporary is weakref-proven dead before the augmented view starts.
- **Release asymmetry is allocator-shaped, not application-shaped.** After canonical pass-0 release anon stayed at 4.12 GiB. After canonical replay release anon dropped 1.41 GiB. After both augmented releases anon stayed at 4.58. Python ownership was identical in all four cases.
- **Micro-0 split hitting the cap at B16** means the rank-loss backward transient is roughly 2.5 GiB even at 16 anchors. The bank rank loss (`member_bank_rank_loss` over the 6355-row bank) is the consistent source of that transient, but I did not read its tensor shapes, so this is unproven and must not become a target.
- **One proven owner at the first bracket:** `temporary['pixels']` (48 MiB) stays alive through the B64 backward. Its only use after the forward is the fingerprint at line 1363 (`facts = {... 'pixels' ...}`). No autograd node saves it, because the patch-embedding weight and the input do not require grad. Popping it after the fingerprint is science-neutral, replay-neutral and RNG-neutral.
- **Why that is not the correction:** cumulative overshoot across brackets was about 90 MB and the micro-phase brackets run at a baseline 1.04 GiB above canonical with no B64 temporaries alive at all. A 48 MiB pop would remove the first event and almost certainly not the gate failure. Applying it alone is a speculative memory-fit attempt, which the freeze forbids.

## The contract: metadata accounting at the first-event bracket

Two fields, no new rows, no tensor retention, no behavior change.

1. **`malloc` field in every `phase_observe` row** (seam: the `row` dict at lines 248 to 257). Read glibc `mallinfo2` through `ctypes.CDLL(None)` and record `arena, hblkhd, uordblks, fordblks, keepcost` as integers. If the symbol is absent record `null`. This is pure stdlib and separates freed-but-retained arena bytes from in-use bytes.
2. **`graph` field on the pass-0 `full_forward`, `full_loss` and `full_total_gradients` rows only** (seam: wrap lines 1359 to 1388 of `route_full_reference` in `torch.autograd.graph.saved_tensors_hooks(pack, unpack)` where `pack` records `nbytes` keyed by storage `data_ptr` into a dict of ints and returns the tensor unchanged). Record `saved_count`, `unique_bytes`, and the three largest `(shape, bytes)` entries. Store the dict in `context['phase_observation']` so `phase_observe` can emit it. Caveat to record in the freeze: under saved-tensor hooks PyTorch skips the in-place version check on saved tensors. Numerics are unchanged, and the existing replay and `route_close` gates stay in force and would catch any drift.

Mutually distinguishing predictions, evaluated on the augmented view at `full_pass_released` relative to its own `full_begin`:

- **P-ALLOC:** `fordblks + keepcost` grows by at least half of the anon growth and `uordblks + hblkhd` returns to within 10 percent of `full_begin`. The residency is glibc-retained free memory. No source correction exists under the freeze's constraints.
- **P-NATIVE:** `uordblks + hblkhd` stays at least 0.5 GiB above `full_begin` while `unique_bytes` plus declared owners explain under 30 percent of that excess. Native workspace holds it. No source correction exists.
- **P-APP:** `unique_bytes` plus declared owners explain at least 80 percent of the current growth at `full_original_A_C_gradients`. An application owner exists and the top census entries name it. Only then is a targeted source correction justified, and the pixel pop joins it.

A secondary check on the baseline drift: P-ALLOC predicts `fordblks` of roughly 0.6 GiB or more at augmented `full_begin` versus near zero at canonical `full_begin`.

**Stdlib falsifier.** One `python3 -I` script over the log: parse JSON lines, require both fields on the named rows, require the 128-record and 256 KiB bounds unchanged, evaluate the three predicates, and exit 0 only if exactly one holds. Exit 2 if none or several hold, exit 3 if `malloc` is null. Plus a one-line pre-check on the target host that `mallinfo2` exists and its struct is ten `size_t` fields. The implementer should add one torch unit in the existing test file pattern proving a tensor saved under the hook dies by weakref after graph release.

**Resource and cost consequences.** Rows are unchanged in count (117 of 128 used), each grows by roughly 160 bytes, total stays near 136 KiB under the 256 KiB bound. `mallinfo2` is microseconds per call under arena locks. The pack hook is a few hundred Python calls on pass 0 only. Source SHA changes, so this requires a new freeze recorded as diagnostics-only with the same stop rule. This plan launches nothing. The contract is exercised only when the root next decides to run the original gate.

**Inverse.** Delete the `malloc` line and the hook wrapper. The falsifier then fails on missing fields, which is the intended signal.

**Stop rule.** If P-ALLOC or P-NATIVE holds, declare the original unchanged CPU600/8 GiB qualification not source-repairable and hand the cap or host decision to the operator. If P-APP holds, make exactly one correction at the named owner. If the result is undetermined, stop without a second contract.

**Risks.** The rank-loss transient explanation is unproven and must not be acted on. `mallinfo2` sums all arenas but not THP rounding, so P-ALLOC can under-count. The hook touches the graph-creation seam of the reference pass, so the freeze reconciliation must state it is metadata-only.
