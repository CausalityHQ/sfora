# Completed continuation review

## research_critic

# Critique: paired 256-step layer-26 MLP continuation (dc5280c, seed179061 first)

**Verdict: don't start 256-step training yet.** The idea isn't dead, but the existing artifacts can answer the deciding question for a fraction of the cost. If it does go ahead, it should continue from the admitted step-128 checkpoints, not restart from initial weights. Under the current caps, a fresh 256-step run can't be done at all. This answer releases no experiment.

I edited no repo files. All numbers below come from committed evidence, read with stdlib Python only. The census figures are my own computation as critic, not root-verified evidence. I did save a short memory note about this verdict in `~/.claude/projects/.../memory/`.

## 1. What the code allows: fresh 256 vs continuing from 128

- **No learning-rate schedule.** The learning rate is constant (1e-4 for the head, 1e-5 for the MLP), with AdamW weight decay 0.05 and clip 1.0. The CUDA GradScaler is asserted to stay fixed at 128 (`train_siglip2_connected_mlp.py:1039`).
- **Augmentation randomness is pre-drawn.** It comes from PCG64 masks, not the global RNG. `save` stores optimizer, scaler, counter and both RNG states, and mechanics already proved that 17 straight updates equal 8 + save/restore + 9.
- **So continuing from step 128 is the same algorithm as a fresh 256-step run.** There is no scientific reason to prefer the fresh run.
- **A fresh 256-step run breaks the TRAIN3000 cap.** The 128-step core is about 1678 s per arm, so 256 steps is about 3360 s of core alone. Running it would need a cap change, which is ruled out. A 128-step continuation fits at roughly 2.2 ks per arm.
- **"Change only `'updates':128` on line 91" is false.** The value 128 is also hard-coded in:
  - the `update()` guard (`1<=step<=128`, :980)
  - the `arm_run` step total (:1522)
  - `check_terminal`, which requires `training_updates == 128` (:1631)
  - `check_scope_schedule_fact`, which requires `len == 128`
  - `class_visits` (`range(128)`)
  - the `(128,64)` schedule shape checks
  - `control_schedule_sha256`
  - the `scope_schedule` docstring ("only first128 are used, never search for another batch")
- **The recipe is part of the checkpoint's identity.** A 128-step terminal checkpoint will be rejected under a 256-step recipe. Continuing therefore needs a new, explicit link from the old identity to the new one, not a constant change.
- **Steps 129–256 have no admitted schedule.** Rows 128–255 and their masks are fully determined by the unchanged 1000-row PCG64 generator, so using them is not a search. They still need a new CPU authority.
- **Repeat exposure doubles.** About 16.3 visits per class over 1008 classes, and about 2.6 draws per FIT image per view instead of about 1.3.

## 2. What the existing evidence already says (more than displacement does)

**Displacement.** The weight change is about 21% of the maximum possible if every Adam step pointed the same way. For comparison, a random walk would give about 9%. Weights and biases move by the same per-coordinate fraction, so the "biases barely moved" reading is just a side effect of the biases' large norms. Norm shrinkage matches the weight-decay prediction. Fable's "parameter norm ≈60" is actually 102. None of this says whether more training would help.

**The errors outside the shared core are used up** (`query-transition-diagnostic.json`):
- Seed061: 11 errors fixed and 5 new errors. Of the 45 queries wrong under both arms, 44 are the core44 and 1 is not.
- So the 128-step candidate already fixed 11 of the 12 control errors outside the core44. Any further gain on seed061 has to come from the core44 or from undoing the 5 losses.
- Both effects repeat across seeds: 9 queries fixed under both seeds, 5 broken under both.

**Core44 margins move in a consistent direction** (my census computation):

| Seed | Median margin change (candidate − control) | Queries improved | Rank better / same / worse |
|---|---|---|---|
| 061 | +0.019 | 32/44 | 20/19/5 |
| 069 | +0.021 | 31/44 | 18/19/7 |

- The per-query margin changes correlate at r = 0.955 across the two seeds.
- This is not a scale artifact. Similarity rises overall (median best positive +0.064, top impostor +0.043). That general rise is also a warning sign for crowding around popular impostors, which would cost queries that are currently correct.
- Extrapolating along the same direction, core44 queries that would flip to correct:

| Dose (in 128-step units) | Seed 061 | Seed 069 |
|---|---|---|
| √2 (random walk) | 6 | 8 |
| 2 (fully coherent) | 11 | 13 |

- The 10 core44 queries closest to flipping need 1.0–1.9× the 128-step dose.

**What this can't tell you:**
- How many currently-correct queries get broken. Margins for the 1673 correct queries were never measured, and the 5 seed-consistent losses suggest losses will grow too.
- Bias from how the core44 were selected: they were picked because every endpoint got them wrong.
- How much the control arm drifts by step 256.

So whether 256 steps can push the product interval's lower bound above zero depends on gains minus losses. Only the loss side is missing.

## 3. The one cheaper measurement to run first

A CPU-only census of the four already-admitted exported endpoint outputs (control and candidate, seeds 061 and 069). No images, training or GPU; same scorer and panel; under the census-v1 resource policy, which peaked at 3.3 GB. Freeze the rule before reading anything:

- **Per query, all 1734:** margin = best positive − best impostor, for each endpoint; change = candidate − control.
- **Self-check:** projecting at dose k = 1 must reproduce the actual candidate R@1 vector exactly. If it doesn't, the measurement is invalid.
- **Project** margin = control + k × change, for k ∈ {√2, 2}, counting flips both ways (errors fixed and correct queries broken). Hold the control at its step-128 value.
- **Interval:** feed the projected R@1 vectors into the unchanged paired product/query interval helper (5000 draws, seed 179019).
- **STOP** if the projected two-seed product lower bound at k = 2 (the optimistic case) is ≤ 0. Close the lane; 256 is not proposed.
- **If it passes:** 256 is justified only as a dose-response falsifier, carrying the recorded prediction band. Passing does not validate the method.

## 4. If the measurement passes: the smallest 256-step plan

**Preconditions (otherwise STOP; never raise the cap):**
- The step-128 terminal checkpoints must exist with their current bytes on disk.
- Each must re-hash to its admitted `terminal_state_sha256`.

**Gates, in order:**
1. **Source and review.** Bump the recipe to 256; the step guards run from 129 to 256. Steps 129–256 use rows 128–255 of the existing generator, with the first 128 rows byte-equal. Add an explicit link to the parent checkpoint's identity, and a new evaluator authority with a new execution SHA.
2. **CPU (600 s policy).** Two independent replays of the schedule and masks. The 256-step class-visit histogram is computed prospectively and frozen.
3. **Mechanics (1200 s policy per arm, both arms).**
   - Restore step 128 and check payload digest equality and image-witness equality.
   - 17 straight steps (129–145) must equal 8 + save/restore + 9, as a complete payload digest.
4. **Paired TRAIN for seed061.** The control continuation runs first, then the candidate names it as its fresh control. Gates unchanged: 3000 s, 8 GiB, zero swap and memory events, CUDA under 10 GB, both cost ratios ≤ 1.50.
5. **Export and score.** Both exports, then the FIRST stage: the immediate gate, `first_gate`, and the unchanged floors (R@1 .96482, mAP@R .81778).
6. **FULL stage only if FIRST passes.** Run seed069, then require both product and query lower bounds above zero.

**Stop rules:**
- FIRST fails: close; no seed069 run.
- FULL fails: close the lane. Declare now that no 512-step or other dose follows, so dose isn't tuned against an already-exposed panel.
- The 128-step KILL stays on record either way.

**Cost and resource risks:**
- About 10 ks of DGX time for seed061 (two continuations at about 2.2 ks, two exports at about 1.25 ks, mechanics, CPU, score) and about 9 ks more for FULL.
- The bigger risk is in-scope page cache (pages the process has read, charged to its 8 GiB cgroup), not compute. The TRAIN128 host peak was 6.02 GB of 8 GiB, and the FS diagnostic and exit rehash have already hit memory.max events. Checkpoint restore and rehash add reads.

## 5. Assumptions I'd challenge and evidence that's missing

- **Even a FULL pass doesn't move production.** It is selection on an already-exposed TRAIN panel, with the step count chosen after seeing that panel's margins. A VAL stage is required before any claim.
  - Gains on held panels have previously disappeared on the official split (the 2000→4000 precedent).
  - This panel has a 1715-image gallery against roughly 12.6k images in the official split.
- **SOP impact is unmeasured.** Fine-tuning layer 26 on In-Shop alone changes shared backbone weights. Joint SOP/In-Shop production needs an SOP no-harm gate, unless the deployment contract allows separate backbones. That contract needs stating.
- **Fable's stop rule doesn't match the actual gates.** Its "seed-061 lower bound" rule and its 2%/10% thresholds should stay rejected. FIRST computes no interval.
- The GPU SHA work is separate, unqualified, and has no bearing on this decision.

## engineering_critic

What this change does: It tests whether another 128 updates improve the paired MLP result, keeping the scientific recipe unchanged. **One matched continuation to 256 total steps is justified for prospective qualification. The current implementation is not ready to run it, and this review releases no experiment.**

The rationale is the earlier positive paired R@1 movement and AP improvement—not the displacement percentage. Original **KILL**, **CLOSE_FS**, and the unmet joint quality-and-speed goal remain unchanged.

1. **Must fix: continuation is not an admitted operation.**  
   `arm_run()` always starts fresh. `update()` rejects step129, and the inherited validator requires a `(128,64)` schedule and counter≤128. `restore()` preserves Adam moments, scaler, counter and CPU/CUDA RNG, but also requires the saved method/source identity to match the current authority. Changing `RECIPE['updates']` cannot extend these contracts. The displacement read examined inference endpoints, which omit optimizer/RNG state; it does not qualify resumability. See [training entry and replay](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_connected_mlp.py:1516), [restore](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_connected_mlp.py:918), and [inherited payload validation](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_identity_diversity.py:1028).

   The smallest valid route is a newly frozen continuation protocol that first admits each **complete original TRAIN128 checkpoint under its original authority**, then explicitly binds that immutable parent into the new protocol. Preserve the original `state_reuse_eligible:false` receipt; do not relabel it. Resetting Adam, restarting the schedule, or using the inference bundle would test a different intervention.

2. **Must fix: fresh256 is incompatible with the current time envelope at observed throughput.**  
   Seed061 control/candidate TRAIN128 consumed **1,703.062/1,757.754 core seconds**. Doubling core work already projects to **3,406/3,516 seconds**, exceeding TRAIN3000 before other work. Holding historical overhead constant projects whole-service times of approximately **3,910/4,053 seconds**. These are estimates, not measured256 results. The last32 updates still took roughly13 seconds each, so startup amortization does not explain away the problem. Sources: [control receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-train-control-179061-v1/receipt.json), [candidate receipt](/home/rb/worktrees/sfora-positive-causality/docs/evidence/compact_metric/sop-siglip2-substrate-v1/connected-mlp-train-candidate-179061-v1/receipt.json), [resource policy](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_connected_mlp.py:120).

   Choose a prospectively admitted128→256 continuation. Do not raise the ceiling or invent segmentation after a timeout. Failure to admit continuation means stop.

3. **Must fix: Fable’s proposed decision rules differ materially from the frozen rules.**  
   FIRST has **no confidence intervals**. FULL requires, for **both R@1 and AP**, mean gain≥0.002 and product lower95>0, plus positive R@1/nonnegative AP per seed and all source/concat floors. Fable’s single-seed lower-bound stop would reject under a different policy; its abbreviated FULL rule omits required gain floors. See [FIRST/FULL dispatch](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_identity_diversity.py:917) and [exact quality gate](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_genuine_views.py:338).

   Reuse that decision arithmetic unchanged, with new method-bound receipts. Neither historical FIRST CONTINUE nor any old GO can authorize the new endpoints.

4. **Should fix: distinguish utility from evidence of a budget effect.**  
   A positive candidate256-minus-control256 result establishes the new paired comparison. It does not by itself show improvement over128. Report candidate256-minus-candidate128 and the change in paired advantage, using the existing128 results as descriptive references. Do not promote these comparisons into new acceptance thresholds.

   My read of the saved061 traces also weakens a simple underfitting story: candidate mean ranking loss was0.054556 over steps65–96 and0.055299 over97–128. Batches differ, so this proves neither convergence nor exhaustion. Core44 filename counts likewise cannot establish the error mechanism.

The smallest prospective plan is:

- **Freeze one method and exact schedule extension.** Use the original generator’s first256 rows from its existing1000-step construction, preserving the first128 exactly; do not repeat128 rows or reseed at the boundary. Authenticate suffix rows, masks, class/image identities and provenance. Keep constant learning rates, optimizer groups, clipping, scaler, fixed views, CONTROL1008/6355 scope, loss, cached training gallery and serving arithmetic unchanged. The existing generator already constructs all1000 before truncation: [schedule implementation](/home/rb/worktrees/sfora-positive-causality/scripts/train_siglip2_identity_diversity.py:393). Freeze actual source bytes and hashes only after implementation.
- **New CPU qualification.** Test full parent admission, the new typed state/schedule contract, boundary update and independent reload/export. Mutation tests must reject wrong parent/arm/seed, altered moments or RNG, counter reset, schedule substitution, changed current bytes despite restored timestamps, and old128 endpoint/GO substitution. CPU checks cannot substitute for CUDA continuation equivalence.
- **New paired mechanics at the boundary.** For each admitted arm, compare128→145 uninterrupted against independently restored128→136, serialize/release/restore, then137→145. Require exact complete terminal state and same-role raw/unit/packed parity. Retain all ownership, independent storage, source, current-byte and cleanup checks. Mechanics state remains discarded.
- **New paired TRAIN061.** Restore the original128 parent independently and execute exactly129–256. Run control first; candidate requires that new control’s complete normal-terminal admission. Replay the first17 continuation diagnostics against new mechanics. Record starting128, executed128 and terminal256 explicitly.
- **New evaluator CPU, exports and FIRST score.** Update every128-specific endpoint binding and payload validator; the evaluator currently explicitly authenticates step128: [payload admission](/home/rb/worktrees/sfora-positive-causality/scripts/evaluate_siglip2_connected_mlp.py:1456). Require complete production UNIT admission, independent export/readback parity and original scoring over all1734 queries,1715 gallery images and498 products. Both query and gallery use their own updated endpoint.
- **Only new FIRST CONTINUE permits the analogous069 stages.** FULL retains equal-seed paired deltas, shared5000/179019 product/query bootstrap draws, both uncertainty summaries, unchanged floors and stops. Any engineering/resource failure stops before quality; FIRST KILL stops before069; FULL KILL closes this256 proposal. No adaptive checkpoints, loss changes, further budget ladder, official or VAL read follows here.

Apply the unchanged≤1.50 core and whole-service limits to the newly matched continuation units. Report the complete128+128 costs separately, including historical prefixes and all qualification/export overhead; continuation is not a free256-step run. A comparison of cumulative costs must not conceal a failing new-unit ratio.

Resource risk is substantial despite the small scientific change. Historical CPU qualification, paired mechanics, two128-step training units and two exports already sum to roughly **2.6 hours before evaluator qualification/scoring**. Additional parent authentication consumes time and file-cache memory. Preserve CPU600/mechanics1200/TRAIN3000 and evaluator700/export1500/score700 limits,8GiB host, zero swap/disallowed events, CUDA<10GB, both locks and full uncached exit. Stop on failure without cap rescue.

I verified local archive hashes and that current trainer/test bytes match the accepted128 source. **Verdict: qualify one paired continuation design; do not launch the current proposal.** GPU-SHA remains a separate, unqualified engineering track.

Not checked: remote checkpoint bytes or native execution; no files edited, tests launched, images opened, SSH or children used.  
Risk: another pass on this repeatedly exposed TRAIN-selection panel cannot establish official generalization, SOTA or serving speed.

## Root disposition

Do not launch continuation yet. Independently verify the suggested all-query margin census from existing outputs, including loss-side counts. Linear margin extrapolation is descriptive and cannot establish a valid training-dose prediction or change frozen acceptance thresholds. Preserve original KILL and CLOSE_FS. Before any continuation, authenticate complete original optimizer/RNG checkpoints and prospectively qualify new schedule, source, mechanics and endpoint contracts. GPU SHA is separate and remains native-unqualified.

