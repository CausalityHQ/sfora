# Original labelled engineering_critic review 9895240c6ef746b4

**What this change does:** delays unused per-view buffers until the genuine B64 reference returns, fingerprints pixels before forward, and deletes the caller’s pixel reference afterward. I found **no demonstrated production correctness, security, or admission regression** in this delta.

1. **Should fix before qualification: test allocation failure after a successful reference.**  
   The new [allocation order](/tmp/rank-routing-witness-allocation-review-v1/train_siglip2_rank_routed_mlp.py:1167) creates a transition where the reference succeeds but a subsequent `zeros_like` fails, leaving `full_reference` and possibly partial buffers in the failed update’s traceback. The added [failure test](/home/rb/worktrees/sfora-rank-routing-witness-allocation-20261009/scripts/test_siglip2_rank_routed_mlp.py:1126) only fails inside the reference.

   Existing [arm cleanup](/tmp/rank-routing-witness-allocation-review-v1/train_siglip2_rank_routed_mlp.py:2069) appears sufficient; **this is a coverage gap, not an established leak**. Smallest falsifier: inject failure at the first and an intermediate delayed allocation, retain the exception, and verify witness/partial-buffer release, preserved primary error, one state cleanup, and no micro execution or optimizer step. No production change is presently justified.

2. **The frozen source-verification gate remains required before the CPU attempt.**  
   The proposed tests load their sibling `DRIVER`, not the reviewed `/tmp` snapshot ([binding](/home/rb/worktrees/sfora-rank-routing-witness-allocation-20261009/scripts/test_siglip2_rank_routed_mlp.py:32)). Bind the final test receipt to the exact production/test hashes and complete the already planned byte/AST inverse and bounded source suite. The [committed plan](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/rank-routed-mlp-v3-memory-plan/root-plan.md:11) explicitly requires this. Narrow reported successes cannot substitute for that gate.

The production paths otherwise support the intended change:

- Outer cross-view accumulators remain intact. The returned dictionary is installed directly, and its temporary alias is deleted before micro execution ([source](/tmp/rank-routing-witness-allocation-review-v1/train_siglip2_rank_routed_mlp.py:1161)).
- Independent total-six and ranking-six B64 gradient calls, original A/C provenance, both-view full/micro comparisons, and unscaled optimizer correspondence remain unchanged ([reference](/tmp/rank-routing-witness-allocation-review-v1/train_siglip2_rank_routed_mlp.py:1376), [comparison](/tmp/rank-routing-witness-allocation-review-v1/train_siglip2_rank_routed_mlp.py:1509)).
- Pixel fingerprints now describe **pre-forward** contents; replay pixel mismatch rejects before model execution. Feature/raw/rank/membership checks remain afterward. This is an intentional validation-timing change, not literal equivalence for a hypothetical input-mutating forward ([source](/tmp/rank-routing-witness-allocation-review-v1/train_siglip2_rank_routed_mlp.py:1360)).
- Creation-time weakrefs, final lifetime checks, RNG restoration, and state audits remain. Deleting the caller’s pixel reference does not establish storage or autograd release.

**Native unknown:** the 121,077,696-byte buffer inventory plus 50,331,648-byte caller pixels measures declared overlap, not guaranteed RSS savings. Original CPUv3 remains FAIL with 424 max events and no accepted CPU receipt ([verification](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/rank-routed-mlp-cpu-v3-result/verification.json:167)).

**Verdict:** close the narrow test gap and frozen source gate, then proceed only to the planned single CPU600/8GiB/zero-event/zero-swap/both-lock qualification. GPU admission still requires the complete normal-zero receipt and outer terminal.

Exact read SHA-256 values, unchanged on recheck:

- Production: `1628700dcf502a7f9669d572eb06e3370bf875cb3aaa7d2aa464108c2cc5c572`
- Mutable proposed tests: `ec17ce0e578de25312faf59087410813ad3f1b6e0976ce63433c40146b2c731c`

No files edited; no tests, imports, native workloads, SSH, children, or consultations run. Full-suite success and native fit are unverified.

