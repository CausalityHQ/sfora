# Compact ranking completion-path plan

Original read-only specialist 5026eaeaefc44b82: GPT-6.1 Sol XHigh, terminal exit 0, 544 seconds. No native jobs or edits. Original mechanics timeout remains FAIL.

Root decision: implement only timing wrappers around the existing bundle, origin and complete-exit operations. Preserve all predicates and both native audit authentications. One new CPU500 gate and one discarded control061 mechanics300 collect the measured bottleneck; no candidate, TRAIN or quality job starts from partial evidence. A complete mechanics pass still requires a separate root assessment of useful TRAIN128 feasibility, without treating an estimate as measurement or changing scientific thresholds.

+**Blocker: the evidence supports targeted instrumentation, not yet a trustworthy speed correction.** Recommend one instrumentation-only change, then one CPU500 → discarded control061 mechanics300 sequence and stop.

Inspected `master` at `cd28be86`; compact, nearest and fitter source hashes match the frozen evidence. No files were edited or runs launched.

The [original mechanics log](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/prototype-residual-ridge-v1/compact-ranking-mechanics-control-v1/original.log:142) establishes:

| Segment | Measured seconds |
|---|---:|
| Final independent reload finished | 130.767 elapsed |
| Bundle preparation finished | 142.091 elapsed |
| Bundle preparation end → source-exit begin | 86.182 |
| Source exit | 47.920 |
| Fitter’s own union within source exit | 14.657 |
| Compact’s own exit | 17.687 |
| Compact exit finished | 293.881 elapsed |
| Final post-exit audit | **Unmeasured; unfinished** |

The 300.188s service clock and trainer timestamps have different starts; their difference is not a precise audit duration. CPU’s corresponding tail is 12.079s **including audit, resource checks and receipt work**, not an isolated CUDA audit measurement.

The first17 updates total 14.236s; replay17 totals 12.094s. Using the measured steady update median, replacing mechanics’ 34 updates with TRAIN128 adds approximately **76.30s of update work alone**. Mechanics passing narrowly would therefore leave TRAIN128 timing unproved.

Redundant reads are real:

- [Nearest API acquisition](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_nearest_ranking.py:407) authenticates; the subsequent audit authenticates again at line 504.
- [Original origin collection](/home/rb/worktrees/sfora-positive-causality/scripts/qualify_siglip2_substrate_cpu.py:480) hashes observed files; [quadratic audit](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_quadratic_readout.py:352) hashes them again without an admission reader; compact’s lines 1404–1405 add another pass.
- Fitter and compact repeat overlapping guard inventories. Compact’s CPU inventory shares **15,353 paths** with nearest CPUv5.

However, the vendor proof contains only **4,280,729 metadata bytes**, versus **409,246,232 supplemental-library bytes**. Metadata-only reuse has no demonstrated sufficient saving. Broad startup-reader reuse would violate [the explicit fresh supplemental-byte checks](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_nearest_ranking.py:562).

The one actionable change:

1. Modify only [train_siglip2_compact_ranking.py](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_compact_ranking.py:1353) and [its existing test file](/home/rb/worktrees/sfora-positive-causality/scripts/test_siglip2_compact_ranking.py).
2. Reuse `timed(context, name)` around bundle qualification in CPU/GPU callers, GPU post-calibration audit, run’s post-run audit, origin-guard promotion, and the final API acquisition and audit separately. At lines 1367–1368:

   ```python
   with timed(context, 'post_exit_api_authentication'):
       api = context['nearest'].native_source_api(context)
   with timed(context, 'post_exit_origin_audit'):
       api.audit_origins(context['legacy'], require_exact=context['args'].phase != 'cpu')
   ```

3. Preserve this predicate ledger without substitutions:

   | Predicate | Retained API/location |
   |---|---|
   | Vendor proof, ownership, size and exact-four authority | nearest `bind_native_authority`, line 326 |
   | Original source/function/default/global/private bindings; fresh supplemental bytes | nearest `authenticate`, line 527 |
   | Actual origins, original conflicts and exact-four CUDA membership | nearest `audit_origins`, line 503; final call stays **after complete exit** |
   | FIT/TRAIN mappings, original closures and fresh exit reader | quadratic `exit_rehash`, line 400 |
   | Full fitter union | fitter `exit_rehash`, line 1172 |
   | Complete compact guards and exact2/external exact3 closures | compact `exit_rehash`, line 1353 |

The fastest falsifier is one stdlib test, `ContractTests.test_completion_timing_preserves_checks`, bounded to five seconds with tiny FILE fixtures. Require inverse-AST equality after removing only the named timing wrappers and reversing the final-call split. Execute the real exit function against trace fixtures: preserve call order, every current guard read, final `require_exact=True`, restored-mtime byte-tamper rejection, and exception propagation. Reject mutants that omit a hash, move the final audit earlier, or swallow its failure.

```bash
timeout 5s python3 -B scripts/test_siglip2_compact_ranking.py ContractTests.test_completion_timing_preserves_checks
```

Root then refreezes the exact2 closure and launches **one fresh CPU500**. Only complete acceptance admits **one discarded control061 mechanics300**, with unchanged locks, resources, reloads, portability and terminal checks. Collect its original terminal result and stop—even if it passes. Timing “end” markers are not qualification receipts. Preserve the original FAIL; admit no candidate, TRAIN or quality work from this diagnostic sequence.
