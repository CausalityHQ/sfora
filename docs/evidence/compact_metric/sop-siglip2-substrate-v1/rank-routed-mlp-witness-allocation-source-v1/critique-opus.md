# Original labelled research_critic review 503e4b42d7524cad

**Verdict: the source change is safe to run (SOURCE GO).** I found no blockers in the 11 changed lines. But the original CPU-v3 log predicts that the one fresh CPU600 run will still fail the zero-event gate. It will fail later, in the augmented view's split/micro steps, which this change does not touch. That is an inference from the log, not a measurement.

I did not edit the repo and ran no tests, imports or native code. The production snapshot reads as `1628700d` in both `/tmp` and the worker. The baseline is `69302dab` in both HEAD `scripts/` and `cpu-v3-freeze`. The test file read as `ec17ce0e`. Replaying `SCHEDULING_INVERSE_EDITS` on the snapshot as plain bytes gives back `69302dab` exactly. I make no full-suite claim.

## Success and error paths: no source blockers
- **Same branch conditions.** The new guard (`candidate and step == 1 and cpu`, line 1165) is true exactly when the old `route is not None and cpu` was true. Control and GPU runs never bind `full_reference`.
- **No second owner of the reference result.** The returned dictionary is installed into `route['full']` and the alias is deleted (1171–1172). The existing release check (1236–1240) still applies.
- **RNG and checks unchanged.** `zeros_like` uses no RNG, and the reference still restores the RNG and checks the payload hash (`release_pass`). All math, exact-equality gates and tolerances are byte-identical. The replay pixel check now fails before the replay forward (1361–1362), with the same message. That is strictly stronger.
- **Error path is covered.** If anything fails between the reference returning and `route['full']` being set, the result stays alive in the frame. `arm_run` clears frames on the error path (2074), the same pattern as before.
- **Pixels are not saved by autograd.** Only `encoder.layers.26.mlp.*` requires grad (lines 88 and 1104), so the patch-embedding layer saves no input. `del temporary['pixels']` (1366) drops the last Python owner. The fake `saved_input` test mode shows this is not enforced at runtime. Treat it as source inference only, because the run cannot observe it.
- **Observation budget is unchanged.** The number of `phase_observe` calls is the same: 117 records against the 128 limit, and 117,180 bytes against the 256 KiB limit.

## Gaps to settle before the one CPU600 run
Row numbers below are 0-based positions in the observation stream of `original.log`.

1. **The memory saving in the failing view is probably not 115 MiB of RSS (native behaviour unknown).**
   - In the canonical view, allocating the per-view buffers raised `memory.current` from 3531 to 3651 MiB (rows 16→17).
   - In the augmented view, where the failure happened, it went from 4522 to 4522 MiB, with anon unchanged at 3593 (rows 61→62). Those buffers reused canonical memory that had been freed but was still resident.
   - So moving them only frees allocator holes, which helps only if the reference's own allocations reuse them.
   - The 48 MiB of pixels (50,331,648 B) is above glibc's 32 MiB dynamic mmap ceiling, so it likely really returns to the OS. This assumes the service uses glibc malloc with no allocator override, which I did not check.
2. **The direct reclaim around the reference was small.** It is context only, not a gate or threshold.
   - `pgsteal_direct` went from 0 to 6256 pages (24.4 MiB) across the augmented total backward (rows 67→68). The replay ranking gradients added 7.1 MiB.
   - These are not measurements of how far over the limit the run went. They are consistent with a 48 MiB pixel release possibly clearing the reference phase.
3. **Most likely outcome is still FAIL (inference).** With no reference alive, augmented events grew from 132 to 424 (rows 74→104) and direct reclaim added 58 MiB, 46 MiB of it during the split (79→80).
   - That happened on a base of about 5.59 GiB.
   - The augmented reference released almost nothing: `full_begin` 4522 → `full_graph_released` 5590 MiB (+1068), anon +1097. The canonical replay release went from 5341 to 3902 MiB, leaving the canonical reference at only +251.
   - The delta does not change what is live during the micro loop. Pre-register before launch that a PASS is not expected and that the readouts are:
     - (a) where the first augmented event occurs;
     - (b) the augmented post-reference retention, compared with +1068/+1097 MiB.
   - Without that, the single run's result is uninterpretable.
4. **Compare runs by phase and view, not row order.** `view_accumulators` now comes after `full_graph_released`, so "first positive event bracket" must not be read by row index or by the preceding phase.
5. **The "pixel ownership ends after forward" prediction cannot be checked in the run.** Smallest fix: state it as source-inferred. Alternatively, capture the weakref to the concatenated pixels and add `require(ref() is None)` after the `del` (2 lines). That adds a new rejection, so it is optional.

The GPU admission path is unchanged: only a complete normal exit-0 receipt plus the outer terminal admits GPU work.

I also saved this verdict in my own memory notes, outside the repo.

