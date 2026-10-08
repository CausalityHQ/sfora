## Verdict: HOLD

- **Source worker's batching commit `bcc3f26f`:** keep it unfrozen and unadmitted.
- **Separate v5 original first-decision owner:** sound in principle, but on HOLD until it is measured.
- **AST-correspondent old reader adapter:** KILL.
- **Cap escalation:** not recommended.

### Why: arithmetic from committed logs, same execution `a4ca55…`

I checked this with a local stdlib script that only reads the committed logs.

| Run | Admitted | Exit-rehash end | Service time |
|---|---|---|---|
| first-cpu v1 / v3 / v4 / v5 | 359–368 s | 476–486 s | 478.8–489.1 s of 500 |
| full-cpu-v1 | no marker by about 497 s (Python-relative) | — | 500.114 s, timed out |

- **first-cpu-v5 had only 10.9 s of headroom.** It is the same source as the full run.
- **After the admitted marker, the first stage took 114.8–123.2 s** (qualification plus exit rehash).
- **The full stage does a superset of that work** after admission: 4 endpoints instead of 2, and a larger guard set.
- **So unchanged v5 full CPU needs at least 614.8 s, at least 23% over the cap.** Any source change must save at least about 115 s. A realistic target is about 160 s, assuming qualification roughly doubles and the exit rehash grows.

### Counterexamples

1. **The new batched `accept_unit` loop is not on the full-CPU critical path.**
   - In `phase=cpu`, the only `accept_unit` call is `first_selection`.
   - Source equality means that call has to go through the immutable v5 reader. That reader scans 20,381 files / 25.7 GB serially.
   - In this unit, `bcc3f26f` therefore only speeds up the archived-concat loop. That loop was already inside first-cpu-v5's 368 s.
   - To rescue CPU500, that loop alone would need to take at least ~153 s serially at 4× speedup, or at least ~230 s at 2×.
   - For scale: first-score-v2 spent +160.7 s on four whole `accept_unit` calls, each including its own serial loop. **Projected outcome: FAIL.**

2. **Fixing CPU500 would probably just move the timeout to a later unit.**
   - Every later full-stage unit (4 × export1500, score700, VAL) re-admits first061 serially through v5.
   - Each also accepts receipts whose `input_guards` include all of the first-score guards (receipts grow with nesting).
   - The first-stage runs already used 617.9 of 700 s (score) and 1264–1289 of 1500 s (export).
   - The underlying problem is reading every occurrence fresh as nested receipts grow, under caps inherited from the first stage. Batching gives a constant-factor speedup, which doesn't fix that growth.

3. **Copying `original_owner_context` verbatim breaks the binding to the full launch.** It sets `owner['launch']` to the first-score authority, so `check_receipt` would compare the receipt with its own launch on every procedure key. The owner must keep `launch` as the current full launch, by identity.

4. **Copied state would break later checks.** If `accepted_units` or `legacy['invocations']` are copied instead of shared, `native_start`'s distinct-unit check misses first061. The owner also has to run between `admit_endpoints[:2]` and `[2:]`, bound by identity to the stage-first `costs` object.

5. **The AST adapter would pass the genuineness checks without being genuine.**
   - `cli()` uses `Path(__file__)`. To match the receipt's argv, an adapter carrying `check_receipt` would have to claim v5's file path while running different bytes.
   - `guard_helpers` only checks the file path, the file hash and code identity captured after load. It would accept that impostor.
   - The only acceptable shape: v6 owns the I/O composition and calls the genuine `v5.check_receipt(owner, …)`. It is only worth building if G1 (below) shows the first-score loop matters.

6. **v5's full-stage admission of first061 has never completed on real data.** Its predicates are unproven there, not just its timing.

### Minimum v5 owner spec, if G1 funds it

- **Pins in v6:**
  - `FIRST_DECISION_OWNER = {root: v5, exec: a4ca55…, code: 919a05… / df1e23…}`
  - `FIRST_SELECTION_UNIT` = the exact 7-key unit, receipt `bc80bef…`.
  - Load the owner only when `stage=='full'` and `launch.first_selection` equals the pinned unit.
  - The v3 export owner predicate stays first-score only, unchanged.
- **Before use:**
  - `closure(v5)` must equal the pinned code; then `load_authenticated('_connected_first_owner_v5', …)`.
  - Value-check the v5 constants against v6: SCHEMA, COST_POLICY, READINESS, PANELS, SEEDS, ORDER, METRICS, TRAINING, the evaluator/nearest/genuine/reference pins, both scope SHAs, and `policy(p)` for all three phases.
- **Transplant exactly three members:**
  - `args`: same values except `execution_sha256` (v5's).
  - `code`: v5's.
  - `common_guards`: remove exactly the three v6 closure entries and add the three v5 entries, which must be disjoint.
  - Every other member must be the same object as in the current context (`is`), with `owner.keys()==context.keys()`.
- **One-shot call:** a flag captured in the loader closure allows exactly one call, then require all of:
  - decision `CONTINUE`
  - receipt sha `bc80bef…`
  - execution and source code equal to v5's
  - endpoints equal to `launch.endpoints[:2]`
- **Lexical capture:**
  - The loader closure holds the module and a deep snapshot of it.
  - It checks that the context slot, the captured module and `sys.modules[name]` are the same object.
  - It checks the snapshot against the closure's own copy, never the one in the context. That rejects a forged owner and snapshot replaced together.
- **Closures before and at exit:** add v5 to the fresh closure list in `exit_rehash` (with `{}` guards) and to `guard_helpers`.
- **Negative tests:**
  - stage-first launch
  - any unit field changed
  - phase cpu/export, panel validation, or roles `selected_cpu`/exports/`selection_go`
  - a second call
  - rebinding `check_receipt` or mutating `COST_POLICY`
  - forging the context slot and `helper_snapshots` together
  - owner launch set to the first-score launch
  - a v5 byte mutation with restored mtime, caught at exit
  - full score with `exports == ORIGINAL_EXPORT_UNITS`

### Cheapest falsifier (F0, stdlib, runnable now)

The script I ran parses the first-cpu v1/v3/v4/v5 and full-cpu-v1 logs plus `failure.json`. It asserts `lb = (500 − wrap) + min(post) + wrap > 500` and prints **614.8 s, saving required ≥ 114.8 s**.

A second static check is enough on its own: once `first_selection` goes through v5, `bcc3f26f` has exactly one batched call site in phase cpu (the concat loop).

### Next gate (G1): one root-only diagnostic unit, run once

Constraints: stdlib only, read-only, nothing admitted, no Torch/native/images, no output directory, no receipt. Use the same `drop_caches`, locks and cgroup properties, with its own declared diagnostic budget of about 900 s. That budget is not a gate cap.

1. **Phase timing:** load frozen v5 by exact hash and call only `authority(args)` for the frozen full-CPU authority, using a never-created output path.
   - A daemon thread samples `sys._current_frames()` every 0.25 s and records wall time per `authority`, `accept_unit` and `admit_endpoints` line.
   - Stop at return and never call `native_start`.
   - This also shows, for the first time, that the real full branch passes on real data.
2. **Throughput:** after a second `drop_caches`, hash the exact concat and first-score `input_guards` lists serially and with fixed-4 workers. Use 1 MiB reads plus DONTNEED, matching `bound_file`.
3. **Decision rule:**
   - Project v6 at all three caps: CPU500, all four full exports (1500) and the full score (700).
   - CPU projection = pre-admit − serial concat + batched concat + 2×41.6 s + exit rehash scaled by guard bytes. Project exports and score the same way.
   - Freeze `bcc3f26f` plus the owner only if every projection is at most 90% of its unchanged cap (450 / 1350 / 630 s).
   - Otherwise do no more source work. The measured breakdown goes to the operator as a cap or read-per-occurrence decision; that isn't a call for the source worker.
   - Any speedup stays prospective until one new cold original full CPU500 run.

### Root cause

`check_receipt` requires the receipt's producer source to equal the current verifier source. That is why every evaluator change forces another owner transplant. Supply-chain attestation schemes (SLSA, in-toto) keep the producer's identity separate from the verifier's. That is a policy change only the operator can make, so I'm noting it rather than recommending it.

I saved this verdict to project memory.
