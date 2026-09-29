# Bounded next intervention after dense100

Original read-only Opus5.5 consultation `8e33b776233a43e2` completed and collected, exit0. No native model jobs, edits, quality reads or additional consultation by the specialist. Parent retains SAME active production goal and checkpoint ownership.

## Parent reconciliation

Source confirms ArcFace sees all2004 fit proxies and corrected SmoothAP sees the full fit bank. Batch negative grouping alone cannot expose additional negatives. The selected intervention is fit-only confusing-anchor reweighting, preserving full product coverage, the corrected objective, dense10 native constructor and exact packed scorer. Prior AP gain/top1 churn motivates this hypothesis; it does not establish the cause.

The proposed32uniform/32hard allocation, hard-pool320rows and32product minima are unmeasured preregistration suggestions, not established scientific thresholds. Thirty-two hard products alone does not guarantee32 available products after excluding uniform products. A bounded schedule must reject insufficient distinct choices without an infinite greedy loop. Parent must freeze a feasible deterministic policy and one finite-budget rule before census/training; no threshold changes after seeing quality.

The specialist's phrase 'FP16 params with F32 master copies' is inaccurate: qualified models/head/proxy parameters are float32 with FP16 autocast. Retain the actual existing precision/scaler contract.

Archived dense uniform checkpoints may be quality controls for an isolated schedule change only after same-source/same-seed constructor, objective, optimizer, augmentation and full uniform-schedule equality are independently verified for bothseeds. Same sampled inputs cannot be required between treatment and control because they are the treatment. Historical timing ratios remain historical, not contemporaneous overhead. Existing17 replay for one seed alone is insufficient to certify a new sampler's determinism for bothseeds. No control-reuse waiver is granted by this plan.

Next decisive step: minimal fit-bank margin census and schedule qualification, CPU-only on DGX under120s/8GiB/noSwap; never use held identities/quality to select anchors. Precompute candidate schedules once, authenticate source/bank/target/schedule hashes and preserve existing uniform paths. Follow with constructor/17step qualification and a fixed paired training pilot only if scientific/resource/control gates pass. Useful model path: targeting confusing fit anchors then immediate serving/freshconfirmation for a survivor. No additional architecture/site/LR sweep.

## Specialist result (unmodified)

I found no blocker in the source code. There is one real risk in the premise, and a CPU census on the fit bank that costs no GPU time tests it before any training.

## Finding that shapes the plan (from reading the source)
Swapping which examples go into a batch does not give the model harder negatives here, because both loss terms already compare against every negative:
- ArcFace uses all 2004 half-arm proxies (`pe_large_coverage.py:115`).
- The corrected SmoothAP8 rank term (`pe_native_valid_anchor.py:17-21` → `member_bank_rank_loss`, `train_sop_siglip2_compact.py:324`) scores each anchor against the whole detached fit bank (`pe_large_coverage.py:77`).

So the only thing a "hard-negative-aware" sampler can change is **how often each anchor gets trained**, and which augmentations it gets (augmentation is keyed on id and step). The memory note "paired-positive sampler REJECT" (2026-09-29) found a batch-composition change inert under this same anchor-vs-detached-bank loss. The plan below is therefore an anchor-reweighting test, not negative mining.

## Plan
1. **New `scripts/large_hard_schedule.py`** (numpy only, with an assert-based `__main__` self-check). It defines `schedule(target, bank, seed)` and replaces `pe_large_coverage.py:32-46` without touching it:
   - Inputs are only the half-arm fit `target` and teacher `bank`, the same ones the dense run uses (`proof["arms"]["half"]`).
   - For each fit row i, excluding i itself: margin m_i = (highest cosine to any other row of the same product) − (highest cosine to any row of another product). The hard pool is the rows with m_i < 0. Products with a single row are excluded.
   - Each of the 100 steps has 64 slots: 32 uniform slots taken in the existing `order` (stride 32) and 32 hard slots from a seeded, fixed cycle through the hard pool. A hard row whose product is already in the batch is skipped greedily.
   - Keep all the existing asserts: shape (100, 64), 64 distinct products per batch, and every product covered (3200 uniform slots ≥ 2004 products).
   - The 32/32 split is fixed now, with no sweep. The held set (6354 queries / 6245 gallery) and its rescue/loss lists are never inputs.
2. **New `scripts/train_large_dense_hard.py`**, copied from `train_large_dense_pilot.py` with only these changes:
   - line 88 calls the new schedule;
   - lines 101-104 and 117-123 compare against the archived uniform steps, so they cannot hold and are removed. Instead, the receipt records the schedule SHA and a per-step anchor list;
   - everything else stays byte-identical: native400 + head128, FP16 params with F32 master copies, optimizer check, the <10 GB CUDA assert, the same F5 source and both seeds, 100 updates.
   - Checkpoints go to new unit paths, so the dense 179032 and 179041 checkpoints are preserved.

## Whether the archived control can be reused
For the question "hard schedule vs uniform schedule", the archived dense runs are the matched control, so no new control run is needed. That holds only if all three of these are true:
- `schedule(hard_slots=0)` equals `coverage.schedule(target, seed)` exactly (checked on CPU);
- a diff shows the new trainer differs from the dense pilot only at the lines listed in step 2;
- the stack is deterministic, which rests on the existing 17-step bitwise replay for seed 179032.

If any of these fails, a fresh uniform control is required. Score the hard and uniform exports in one scorer call on the same held set:
- **Primary:** hard T vs dense T, paired.
- **Secondary:** the frozen floors against the original control C, unchanged.

## Cost
Per-step compute is unchanged (batch of 64, same terms). The share of rank-active steps may shift, so report it rather than predicting it. Expected cost is:
- 2 × (training wall as in the dense receipts, not re-estimated here);
- 2 × one v2 export, which took about 290 s against the 300 s cap, so there is only about 9 s of headroom;
- 1 × scorer at about 80 s.

The schedule arrays and their SHA must be produced by the CPU qualification and only re-checked in the GPU job, so no startup time is added under the 300 s cap. I'm not projecting any quality gain.

## The one cheapest falsifying qualification (native CPU, fit-only, no GPU)
Compute the hard pool from `teacher.fit.npy` and the half-arm `target`, and report:
- the hard-row count H and the number of distinct hard products;
- the enrichment factor (hard share of anchor slots, hard vs uniform);
- rows per product in the hard pool;
- the result of the `hard_slots=0` equality check.

**Actionable blocker:** fit-set confusion may already be saturated, because the F5 teacher was trained on these rows and TRAIN rank was earlier measured at 0.02. Freeze these kill rules before running:
- **KILL** if there are fewer than 32 distinct hard products (the 32 hard slots per step cannot be filled);
- **KILL** if H < 320, meaning each hard row would be repeated more than 10×, which is memorisation rather than signal;
- **Blocker, not KILL,** if the equality check fails: that only means the archived control cannot be reused.

The thresholds are preregistration choices I proposed, not measured values.

**Leakage and control risks:** held-derived confusion must never pick products (the 179032 and 179041 rescue/loss lists are off-limits). Use the half arm only; the full arm has 3997 classes and must not be used. Confirm that held products are disjoint from fit products before running.
