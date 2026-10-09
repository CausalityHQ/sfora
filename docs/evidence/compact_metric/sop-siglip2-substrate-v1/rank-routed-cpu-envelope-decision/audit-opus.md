The 12 GiB CPU protocol is **not source-feasible under the current frozen contract**. It becomes feasible only if the operator explicitly amends that contract; with that amendment there is one minimal delta, given below. Without it, the GPU stays held.

Evidence: HEAD `667752fa3c57bd4afb2fea0b1ecb772e253325b8`. `scripts/train_siglip2_rank_routed_mlp.py` is byte-identical to the frozen v4 copy (`1628700d`), and the test file matches `d9a68a05`. The audit was read-only stdlib/grep: nothing was run, imported or edited.

## Every predicate that assumes 8 GiB

**Pinned historical predicates (exact equality, cannot be edited):**

| # | Location | Predicate | Effect of a 12 GiB CPU unit |
|---|---|---|---|
| P1 | `qualify_siglip2_substrate_cpu.py:37,514-516` (`eacd32d2`, the `source_driver`) | `int(memory.max) == 8 GiB` and `memory.peak <= 8 GiB` | Raises on the first call |
| P2 | `initialize_siglip2_substrate_fit.py:59,159-160` (the `genuine.reference`/`init`) | `memory.max == "8589934592"` and `peak <= 8 GiB` | Raises |
| P3 | `train_siglip2_substrate_adaptation.py:493,516` (`a1684917`, matches `TERMINAL_SOURCE_SHA`) | Original reader: `native_peak_rss_kib <= 8*1024**2`, plus P2 on `cgroup_before`, `cgroup_after` and the final footer | No later phase can admit the CPU unit |
| P1' | Original trainer `integrity()`, `export-exit-scan-ab-v1-freeze/train_siglip2_identity_diversity.py:1112` | `source.cgroup_memory()` | The recompiled integrity in `_initializer_runtime` (`:403-449`) swaps only the `policy()` call; this cgroup call is still live |
| — | `fit_siglip2_prototype_residual.py:346-367` | The reader may adapt only runtime parsing; pinned by `TERMINAL_AST_SHA` and `ADAPTED_TERMINAL_AST_SHA` (`51d6678a…`) | A cap change is not covered by that precedent |

**In the new source (editable, but frozen at `1628700d`):**
- `:126-129` `policy()` returns `host_bytes = 8*1024**3` for every phase.
- `:147` `check_launch` and `:2133` `check_terminal` require `resource_policy == policy(phase)`.
- `:1004-1017` `resource_check` calls P1 (`:1007`), P2 (`:1010`) and `ru_maxrss <= host_bytes//1024` (`:1012`). It runs at `:2276`, `:2302`, `:2336`, once per seed in `cpu_run`, and inside every `integrity()`.
- `:2143` hard-codes `process_peak_rss_kib <= 8*1024**2`. The run also checks its own receipt against this at `:2334`.
- `:2210` and `:2221` route each selected unit through the unchanged original reader, so P3 applies.
- The contract text says the same thing in four places: `:13` "Historical policies unchanged", `:44` "8GiB", `:316` "No historical policy… rebound", and `:2195` "unchanged original uncached UNIT reader".

**Test (`d9a68a05`):** `:387` asserts `d.policy(p) == c.policy(p)` for cpu, mechanics and train. `:1484` has a fixture with `memory.max = '8589934592'`.

**Launch and freeze:** `cpu-v4-launch.sh` sets `MemoryMax=8589934592`. The authority and freeze set `resource_policy.host_bytes`. The freeze `stop_rule` says "No unchanged retry or cap changes".

**Not cap-dependent:**
- `zero_events` (quadratic readout `:194-197`).
- `phase_observe`, which only reads values.
- The stop footer.
- The original CPU-v5 (CPU500, admitted under its own `policy`, original trainer `:197-199`).
- The actual-gradient witness (`witness.POLICY`, its RSS check at `:548`). It is admitted, not re-run, so it is unaffected.

## Where raising only the new CPU unit breaks semantics

1. **An unchanged-source 12 GiB run fails by construction.** P1 raises at the first `resource_check` (`:2276`), before any arm work. That would be a guaranteed wasted run.
2. **Mechanics and TRAIN admit `selected_cpu` through the unchanged original reader** (P3 plus P2). A 12 GiB unit can only be admitted if that reader is adapted, which directly contradicts `:13`, `:316` and `:2195`. That is a change of contract, not a fix.
3. **`check_terminal` ties every phase to one source** (`execution_sha256` and `code` identity). The 12 GiB CPU run therefore has to use the exact source that will later run mechanics and TRAIN, so all phases must be in the delta before it runs. Any later source change means a new CPU run.
4. **Product limit, kept separate.** The CPU phase is the only place CPU FP32 serving parity is run (`load_inference(…, witness['device'])` at about `:1992`). Under a 12 GiB envelope, CPU FP32 serving at an 8 GiB host becomes **unqualified**. Mechanics and TRAIN serve on CUDA and never re-establish it. A 12 GiB pass also gives no evidence that mechanics or TRAIN fit an 8 GiB host or the CUDA allocation limit.

## The one minimal delta (only after the operator decision)

This would be a new source v5 for all phases. It reuses the existing AST-substitute pattern with the inverse-restore proof (`:403-449`).

- **S1.** `policy()` uses `12*1024**3` when phase is cpu and `8*1024**3` otherwise. Seconds, swap and the CUDA limit stay as they are.
- **S2.** `resource_check` in the cpu phase uses compiled copies of the genuine `cgroup_memory` and `admit_cgroup`. In each, the single `POLICY['host_bytes']` node is replaced with `_connected_host_bytes`. The proofs are a substitution count of exactly 1 per site and an inverse that restores the genuine AST. Mechanics and TRAIN keep the genuine functions.
- **S3.** Extend the `integrity` substitution so that `source.cgroup_memory()` becomes the S2 phase dispatcher, with a new adapted SHA.
- **S4.** A CPU-only adapted terminal reader, layered on the fitter's adapted AST: `self.init.admit_cgroup` becomes the 12 GiB copy from S2, and the literal `8 * 1024**2` becomes `12 * 1024**2`. It needs an inverse-restore proof back to `51d6678a…` and a new pinned SHA. It is used only for `admit_terminal(…,'cpu',…)`. `selected_mechanics`, `fresh_control`, the original CPU-v5 and the actual-gradient witness keep the genuine reader.
- **S5.** Replace the `:2143` literal with `policy(phase)['host_bytes']//1024`. Update the docstring and comments at `:13`, `:44`, `:316` and `:2195` to declare the CPU amendment.
- **Authority:** a new source root, `execution.json`, authority and unit name. Set `resource_policy.cpu.host_bytes = 12884901888` and `MemoryMax=12884901888`, with `MemorySwapMax=0` and `RuntimeMaxSec=600` unchanged. The `original_cpu`, `actual_gradient` and `witness` blocks stay byte-identical. Record a new stop rule; the v3 and v4 FAILs remain on record.
- **Test:** `:387` becomes equality for mechanics and train plus an explicit 12 GiB assertion for cpu. Mutants:
  - In the cpu phase, `memory.max` of 8 GiB is rejected and 12 GiB is accepted.
  - In mechanics or train, 12 GiB is rejected.
  - The CPU reader rejects RSS above 12 GiB, any wrong max, and any event.
  - The mechanics reader still rejects 12 GiB.
  - Every substitution restores to its genuine AST.
  - The CPU-v5 and actual-gradient readers are the same objects as before.
  - The `:1484` fixture is kept for mechanics.

**Bounded falsifying test (stdlib source gate only, no native run):** the delta is falsified if any inverse-restore fails to equal its genuine pinned AST, if any substitution count is not 1, or if a mechanics-phase cgroup with `memory.max` 12884901888 is accepted. The single native CPU run afterwards keeps every existing gate: zero events, zero swap, B64 and micro proofs, cleanup and exit, both locks.

**Dependency ledger:**
- Python `9258c53d…`.
- Witness 7 files, as listed in v4 authority.
- Original CPU-v5 authority `8fcfb9ae…` and receipt `d2239da8…`.
- Actual-gradient authority `84763969…`, receipt `e7da19ba…`, log `3dd08431…`.
- Original trainer `840c5d82…`.
- `TERMINAL_SOURCE_SHA` `a1684917…`, `ADAPTED_TERMINAL_AST_SHA` `51d6678a…`, integrity AST pins `c7b564cf…`/`8ad7272e…`.
- `source_driver` `eacd32d2…`, exporter `163bee8b…`.
- **Not verifiable locally:**
  - The DGX pin for the `init` module; the local copy is `7ac147b8`.
  - Whether `/home/riomus/runs/sfora-native256-initialized-cpu-final-footer-v1.py` asserts anything beyond what it reports.
  - Any `MemoryMax`/`MemoryHigh` on the parent slices (`user@1000.service/app.slice`).

**Operator decision required:** they would have to approve the following, in substance:

> "Amend the rank-routed contract so that the new CPU600 witness alone runs at host 12 GiB. That unit is admitted by an AST-proven, constant-only adaptation of three genuine 8 GiB predicates, not the unchanged original reader. CPU FP32 serving at 8 GiB becomes unqualified. Mechanics, TRAIN and serving keep 8 GiB. The v3 and v4 8 GiB FAILs stand."

**Risks:**
- The roughly 1.146 GB of retention with no identified owner is untouched; this is an envelope change, not a fix.
- A pass says nothing about whether mechanics or TRAIN fit at 8 GiB.
- Any further source edit invalidates the CPU unit.

**If the operator declines:** no native action is authorized. Removing or relaxing the `selected_cpu` requirement (`:151`, `:161`) would bypass a gate, so the GPU stays held. Any further work, including drafting this delta, needs new authorization.
