# Read-only prerequisite admission audit

Actual GPT-6.1 Sol XHigh; consultation b5bf232a633f48f2; terminal exit0. No measured speedup or complete 300s budget claim.

Recommend one small intervention: **reuse the authenticated `legacy['admission']` reader only in nearest-ranking’s startup prerequisite terminal admission.** This removes a demonstrable redundant inventory scan. It does **not** yet establish a credible 300-second lifecycle budget.

All source findings below refer to committed `40426ca3`; the inspected fitter, quadratic, genuine and adaptation files match the SHA pins in the accepted CPU receipt.

The authentication path is:

- [Nearest authority](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_nearest_ranking.py:214) authenticates its exact3 closure, loads the pinned fitter, calls `fitter.authority`, then admits the accepted concat terminal.
- [Fitter authority](/home/rb/worktrees/sfora-positive-causality/scripts/fit_siglip2_prototype_residual.py:226) authenticates original CPU/source facts and calls quadratic `old.authority`. Its historical terminal loops at **418–419**, and ordinary terminal loop at **1073–1074**, nevertheless call its uncached `bound_file` again.
- [Quadratic authority](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:284) creates an actual authenticated `original.FlatAdmission`, registers its already-authenticated guards as verified, and retains it as `legacy['admission']`. Thus nearest already has a legitimate reusable reader.
- [FlatAdmission](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_substrate_adaptation.py:214) checks canonical paths, digest syntax, conflicting SHA/size and conflicting stage authority; `bound_file` hashes previously unverified paths, while `read_json` authenticates bounded bytes and parses those bytes.

The original adaptation `authority` creates a local reader and drops it before native execution. That is distinct from this actual nearest path: quadratic retains a separately instantiated, authenticated reader. No change to the original adaptation lifecycle is needed.

[Nearest `admit_terminal`](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_nearest_ranking.py:1188) currently authenticates receipt/authority JSON through its double-read helper, then unconditionally streams **every** receipt `input_guards` entry at **1210–1211**, even where the retained original reader has already verified that path. The accepted nearest CPU receipt inventories **15,152 paths**; all **15,140 fitter CPU paths** occur in it with matching digests. This establishes substantial overlap, not a measured saving or proof that every path is already reader-verified.

One bounded implementation plan:

1. **Modify only the admission region of `scripts/train_siglip2_nearest_ranking.py`, after the memory child is integrated.** Extend the existing API to:
   ```python
   read_json(fact, guards, *, admission=None)
   ```
   Preserve `file_fact(fact)`. With an explicitly supplied reader, use its `descriptor_json(fact, guards)`; otherwise retain existing behavior.

2. Inside `admit_terminal`, obtain the existing `legacy['admission']`. Require its exact authenticated original class and unchanged bound methods. Supply it to the two receipt/authority JSON reads and replace the final inventory call with:
   ```python
   admission.bound_file(guards, p, h)
   ```
   Keep `authority(args)` and `admit_terminal(context, unit, phase, arm)` signatures unchanged. Keep all launch, source, code, interpreter, argv, prerequisite inventory, cgroup, invocation uniqueness and terminal predicates unchanged, including the authenticated original terminal adapter.

3. Add **`scripts/test_siglip2_nearest_ranking_startup_admission.py`**, independently of the active trainer/test memory slice. Keep fitter and all historical helper bytes unchanged: fitter SHA `95295794…adc13b` is part of the accepted concat authority. Editing it would expand this task into an authority migration.

Predicate correspondence is narrow: nearest’s full-byte inventory authentication moves to `FlatAdmission.bound_file`; JSON descriptor validation remains nearest-owned, and byte authentication/parsing moves to `FlatAdmission.descriptor_json`. Stage guard dictionaries and `required_guards` snapshots remain separate and unchanged. **Never seed new `verified` entries merely from a receipt or guard union.**

The source-only RED→GREEN falsifier should execute the real nearest admission function with the real original reader and tiny valid terminal fixtures, bounded to **5 seconds and 16 MiB**:

- Preauthenticate one bulk fixture through that reader; admit two distinct valid terminal fixtures referencing it and one newly encountered bulk path. Count actual full-byte reads: GREEN requires **one per unique bulk path across the reuse window**, with both stage inventories populated. Current nearest source is RED because it rereads the preauthenticated path.
- Reject conflicting SHA, conflicting size, stage guard conflict, missing/noncanonical files, and wrong bytes on first authentication.
- Reject authenticated duplicate-key/nonfinite JSON and cap violations. Mutating a returned parsed object must not affect the next parse from authenticated cached bytes.
- Reject altered terminal predicates and substituted reader methods. Preserve terminal-adapter mutation checks.
- Exercise the unchanged fresh exit path: same-size byte tampering with restored timestamps must fail SHA authentication. Successful exit must still perform its fresh reads.

**Trust limit:** this is reuse of startup authentication, not continuous file authentication. Same-size changes after verification need not fail the cached startup lookup; authenticated JSON remains the original snapshot, and the complete uncached exit must reject changed bytes. Stat/mtime supplies no authentication. Runtime loads, payload checks, parity, scientific predicates and [exit hashing](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_nearest_ranking.py:1240) retain their current behavior. The main risk is accidentally allowing this reader into runtime or exit calls.

The timing prerequisite remains open:

| Evidence | Seconds |
|---|---:|
| Mechanics startup admission / preparation | 122.321 / 35.858 |
| Seventeen updates: compute / including integrity | 14.900 / 50.310 |
| Rejection after failed restore | 284.523 |
| Original updates 9–17, used as replay estimate | 26.529 |
| Accepted CPU fresh exit hashing | 41.059 |

Using the recorded durations gives **352.111 seconds before additional remaining reload/proof work**, minus measured startup saving `S`, plus the unknown restore-cost change. Meeting 300 therefore needs approximately **52.111 seconds plus remaining work** in savings, subject to that restore change. This intervention’s saving is unmeasured; the CPU own-exit scan’s 14.656 seconds cannot be substituted for its timing.

The plan is justified as a small, testable reduction in redundant reads. It is insufficient evidence to close timing. Refreeze changed nearest bytes and obtain fresh CPU qualification before root considers further qualification; keep DGX mechanics withheld until both memory and timing prerequisites are closed.

Read-only audit completed. No files, tests, native runs, jobs, children, reviews or operator messages were initiated.
