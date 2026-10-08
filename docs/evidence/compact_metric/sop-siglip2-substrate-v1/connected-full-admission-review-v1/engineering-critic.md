**HOLD qualification. GO for one source-only action: add a narrowly scoped source-v5 owner for the exact accepted first061 CONTINUE receipt, using its unchanged `accept_unit`.** Keep the batching worker’s scope intact. Defer the old-reader batching adapter until evidence shows it is needed.

Concrete findings:

1. **Batching alone cannot qualify full stage.** The genuine reader compares both receipt execution and evaluator code against its context. Changing either evaluator or test bytes invalidates the accepted first score. I reproduced this rejection using the actual predicate and accepted receipt. Weakening equality would erase provenance. [Original reader](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v1-freeze/evaluate_siglip2_connected_mlp.py:1116)

2. **Calling the old reader retains its serial scan.** Its `accept_unit` resolves `bound_file` through the original module, so batching the current evaluator does not accelerate those 20,381 historical guard checks. The 25,735,111,462-byte inventory establishes neither elapsed hashing time nor a predicted saving. This limits the optimization claim; it does not establish that an adapter is necessary. [Original scan](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v1-freeze/evaluate_siglip2_connected_mlp.py:1190)

3. **Reusing the export-owner composition would grant the wrong authority.** That composition substitutes six context members and admits a historical CPU to validate first-stage exports. First-decision admission needs neither that CPU admission nor export dispatch. Its source-v3 route must retain the existing first-selection-only restriction. [Existing owner composition](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v1-freeze/evaluate_siglip2_connected_mlp.py:135)

4. **The timeout does not identify the expensive phase.** The admitted marker follows `native_start`, so its absence cannot distinguish authority work from native startup. The authenticated terminal result remains killed/TERM/timeout at 500.114 seconds despite footer status zero. No qualification or cap increase follows from that evidence. [Run ordering](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v1-freeze/evaluate_siglip2_connected_mlp.py:1257), [failure record](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v1/failure.json)

The minimum coherent reader-context transplant has **four substitutions**:

| Member | Required substitution/check |
|---|---|
| `root` | Exact canonical source-v5 root; separate from current source, helper roots and output. |
| `args` | A copy of current args with **only** `execution_sha256` replaced by the pinned v5 execution. Check every remaining value against current args. This is a reader context, never an argument to old `authority` or `run`. |
| `code` | Exact pinned v5 evaluator/test hashes, freshly authenticated through its execution closure. |
| `common_guards` | Remove exactly the current evaluator/test/execution entries and insert exactly the three v5 entries. Require the current entries to exist with their expected hashes, reject conflicting/overlapping closure membership, and preserve every other entry. |

Require identical context key sets and `owner[k] is context[k]` for every other member. In particular, retain the **current full launch**, training/score contexts, scientific helpers, terminal reader, preparation costs, endpoint records, cost dictionary, guard ledger, invocation ledger, accepted-unit list and any current CPU.

Keeping the full launch is deliberate: the unchanged reader already selects its first two endpoints and seed061 costs when explicitly called with `stage='first', panel='selection'`. I exercised its complete metadata check with that arrangement successfully. No historical launch or CPU substitution is needed.

Admission must additionally pin the **complete UNIT**, not merely its receipt hash:

- Receipt `bc80bef471cd05c7ab842258a9b86f3e8813f5dabc5235e2202399f5c607028d`.
- Invocation `c47869c3b2d24e70a2213545e371265e`.
- Exact receipt/log paths and hashes, unit name, duration, RSS and lock assertion from the accepted [UNIT](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-first-selection-score-v2/unit.json).
- Original score authority hash `d1134fdf92c942f7ba65ee0aa814cdc735732f14a035420785bb2160af41f712`.

Dispatch once to genuine v5 `accept_unit(..., 'score', stage='first', panel='selection')`, then require CONTINUE and the exact first endpoint pair before admitting069. Historical score validation retains its **700-second** policy; the new CPU service independently retains **500 seconds**.

The owner and snapshots need an independent trust anchor. Capture the authenticated module, expected descriptor, module/spec identities, function code/defaults, namespace bindings and copied mutable literals in a lexical closure held outside replaceable context slots. Check against that capture immediately before and after admission and at exit. Comparing a context-supplied owner with a context-supplied snapshot permits both to be forged together. Preserve the existing [independent capture pattern](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-evaluation-full-cpu-v1-freeze/evaluate_siglip2_connected_mlp.py:123).

Freshly authenticate v5’s complete three-file closure before use and at exit. Keep both old and current closures in the accumulated guard ledger; the three-entry substitution applies only to the owner’s common-input expectation. Preserve genuine receipt, CLI, resource, terminal, origin, interpreter and decision validation.

The wrapper’s sole historical role is **first-selection prerequisite**. Reject its use for CPU, export, full selection GO or validation scoring. Full validation may consume that prerequisite only while the unchanged current reader separately requires full selection GO. Historical source-v3 exports remain unavailable to full stage.

For the optional adapter, **HOLD now**. If measured attribution later justifies it, a separately owned adapter can be defensible only with:

- An exact, reversible AST correspondence to pinned v5 `accept_unit`, changing only its bulk guard loop and explicitly audited original-function bindings.
- Genuine unchanged old `check_receipt`, terminal and native-origin validators.
- Original `bound_file` semantics, four workers, fresh reads per occurrence, no deduplication/cache, joined completion and ordered atomic bulk publication.
- Independent authentication of the adapter and its bindings; no monkeypatching old globals or manufacturing a replacement original reader with patched globals.

Preserve the existing pre-loop invocation-ledger effect. On rejection, prohibit accepted-unit publication or native advancement. Concurrent scheduling and failure-message precedence can change; acceptance predicates cannot.

The cheapest existing executable falsifier is the worker’s stdlib-only scan check, which also exercises the historical source rejection:

```bash
rtk proxy timeout 120 prlimit --as=1073741824 -- python3 -B -c \
"import runpy; runpy.run_path('/home/rb/worktrees/sfora-connected-full-admission-fresh-batch-20261008/scripts/test_connected_mlp_evaluation.py')['actual_admission_scan_falsifier']()"
```

For the proposed composition, add **one case to the existing owner fixture**, exercising the actual new wrapper and unchanged v5 reader. Its minimum counterexamples are:

- Replace owner **and** snapshot together; mutate code/defaults/globals after capture.
- Omit a shared common guard, remove an extra guard, or alter either closure hash.
- Substitute UNIT/log/invocation, CLI authority, endpoint order, cost or preparation data.
- Route old first CONTINUE as full GO, CPU, export or VAL authority.
- Supply timeout terminal evidence with command status zero.
- Invoke old authority, admit first061 twice, or reach069 after rejection.

Use spies only for unavailable external I/O; replacing the reader or scientific decision with an unconditional success would invalidate the test. Then run the affected source-only suite once.

**Exact next qualification gate:** after that source proof passes, freeze the composed current source and run **one new cold original full CPU500**, with unchanged 8 GiB, zero swap, both locks, all four ordered endpoints, complete native/parity checks, original normal-exit terminal validation and fresh exit hashing. Only its PASS permits four fresh current-stage exports1500, then full score700; only full selection GO permits sealed VAL. Any speed benefit remains prospective.

I verified the worker production diff reverses exactly to frozen v5 bytes and AST, and authenticated the preserved CPU failure log plus both TRAIN069 receipt/log bindings. The metadata falsifier did **not** qualify remote file/origin admission or a new owner implementation. No files, jobs or operator channels were changed.
