## research_critic: claude-opus-5-5

Original consultation 05054856e1764d93, group a258d9307a9a4577; status completed, exit 0.

# Verdict: REJECT dense adaptation of blocks 10–11 as the next intervention

The idea looks new relative to the killed rank‑8 q/v arm, but it is really a move of the freeze boundary from 12 to 10. The repo has already measured that boundary, on the same data family, at the same dense LR (1e‑5), with 10× the budget. R1 did not respond, and mAP got worse with more lower-stack plasticity. So this arm is predicted to fail the +0.20pp R1 floor, and it cannot help the speed half of the goal. I read the files only; nothing in the repo was edited or run.

## 1. Evidence that decides it

The earlier results come from `docs/inshop_true_freeze_gate_2026-09-26.md:19` and `docs/inshop_freeze16_gate_2026-09-27.md:69`. Setup: seed 179024, 1,000 updates, vision LR 1e‑5, ArcFace+SmoothAP, TRAIN‑held symmetric panel of 12,599 images from unseen products.

| Trainable lower stack | R@1 | mAP@R |
|---|---:|---:|
| All 24 blocks + embeddings | 98.555 | 0.8267 |
| Blocks 0–11 frozen, embeddings trainable | 98.643 | 0.8377 |
| Embeddings + blocks 0–11 frozen (current recipe) | 98.548 | **0.8391** |
| Embeddings + blocks 0–15 frozen | 98.460 | 0.8315 |

- **Full fine‑tuning already contains the dense 10–11 arm.** It trained every lower block, including the MLPs, for 10× the updates. R1 moved by one query (interval about ±0.19pp) and mAP fell 1.24pp.
- **The one multi-seed comparison agrees.** Across three seeds, the older freeze‑12 arm gave positive mean mAP against full fine‑tuning and failed a per‑seed R1 interval (`true_freeze_gate:3`).
- **The pattern is always mAP moving and R1 not.** No boundary change in the ledger has moved R1 by more than 0.18pp. The +0.20pp floor means about 13 of roughly 208 missed queries (a 6% relative error cut) in 100 updates.
- **The q/v arm showed the same signature.** AP rose 0.094pp with an interval above zero; R1 rose 0.008pp. That matches the earlier finding that SmoothAP moves mAP but not R1. The evidence points to the objective, not lower-stack capacity, as the R1 limit.

**Caveats:** the earlier curve started from pretrained weights rather than F5, used BF16 rather than FP16, a different panel shape, and mostly one seed. That stops it counting as a replication. It does not give any reason to expect two blocks for 100 updates to beat twelve blocks for 1,000.

## 2. What the q/v KILL does and doesn't show

- **It is not a universal architecture kill.** It closes one allocation: rank 8, q/v only, blocks 0–11, factor LR 1e‑4, 100 updates.
- **The dense 10–11 arm has the same weakness.** A 100‑update result, pass or fail, is conditional on the budget.
- **Only a pass matters, and it would still have to be retrained at production budget.** At that budget the measured curve predicts null R1 and slightly lower mAP. That weak expected value is the reason to reject; the q/v result alone would not be.
- **The q/v hypothesis is already weakened.** The full‑rank, all‑sublayer, all‑lower‑block version of "q/v was too restricted" was tested at 1,000 updates and was null for R1.
- **Neither outcome of the new arm could isolate a cause.** It changes rank, sublayers (adding k, output and MLP), depth (0–11 to 10–11), LR (1e‑4 to 1e‑5) and parameter count (0.39M to 25.2M) all at once. A pass could not be credited to the MLPs; a fail could not exclude MLP plasticity in blocks 0–9.

## 3. Limitations even if it runs

- **No speed contribution.** The architecture is the same 400 tensors, so latency is unchanged by construction.
- **The cost gate cannot fail.** Backward only goes through 14 blocks instead of 12, so I'd expect a cost ratio around 1.05–1.10, against 1.245 for the q/v arm. Peak memory is also bounded by the q/v arm's 7.84 GB. A cost PASS tells you nothing.
- **Selection bias.** The arm was chosen after control and candidate quality on this panel had been read. The panel has now been observed at least three times, so any result is exploratory.
- **Gradient clipping is active on every step.** Pre‑clip norms run 37–61 against a clip of 1 (mechanics‑v4 receipt). Adding 25M parameters to the global norm changes the per‑step clip factor for the upper blocks, head and proxies. Adam is scale‑invariant within a step, so this is only a per‑step reweighting and a minor confound. It still needs logging.

## 4. If the operator overrides: amendments needed before training

**Essential amendments:**
1. **Disclose the prior and the predicted outcome in the gate doc:** R1 null, mAP flat or slightly down.
2. **Pre-declare what each outcome means.**
   - FAIL closes lower‑stack plasticity for R1 under this objective, LR and budget. Combined with the 0/12/16 curve, that closes the family for this recipe, but not universally.
   - PASS only authorizes retraining at production budget, where the 1,000‑update curve must be overturned, followed by fresh confirmation.
3. **Log per‑group gradient norms every step:** blocks 10–11, blocks 12–23, and head/proxy.
4. **Keep all floors unchanged.** State that the cost gate cannot fail and that no speed claim is possible.

**Smallest implementation.** Write a new copy of `train_large_lower_pilot.py`. Do not touch the shared modules, because that would change the source hashes.

- **Setup order:**
  - Call `coverage.load_native` unchanged (400‑key strict load, source hashes, freeze‑12 inventory).
  - Check that inventory and the frozen digest against the proof, as `old.fresh` does, *before* unfreezing.
  - Call `layers[10].requires_grad_(True)` and `layers[11].requires_grad_(True)`.
  - Only then build `coverage.parameters(...)` and the AdamW optimizer, with the same groups as `pe_large_optimization.py:75`.
- **Why the order matters.** `old.fresh` builds both the parameter list and the optimizer inside itself (`pe_large_optimization.py:65-75`). Unfreezing after it leaves blocks 10–11 out of both, and no per‑step check catches it. Blocks 10–11 receive gradients but are never stepped, so the run is a silent null arm that ends bitwise identical to control.
- **Frozen invariant.** Use explicit clones of the 163 tensors in embeddings and blocks 0–9 (3 + 10×16), plus a fingerprint of the buffers. Do not use `frozen_state`: its roots are hardcoded to blocks 0..11 (`pe_core_training.py:55`), so blocks 10–11 would be included and it would fail after training. Don't call `frozen_digest` after training at all.
- **Drop the adapter code:** the `install`/`merge` calls and the mechanics first‑17 replay assert (`train_large_lower_pilot.py:160`). The checkpoint is simply the native state dict.
- **Export and scoring.** Copy the export driver with the run paths and training hash swapped; only the two candidates need exporting. Reuse the control held wires by SHA (`34635dc4…` and `9c7d6621…`) and the scorer unchanged.

**Qualification that could falsify the run:**
1. **Counts:** exactly 163 frozen and 237 trainable vision tensors, all 32 block‑10/11 tensors trainable, and 240 optimizer members.
2. **Optimizer identity:** the ordered list of optimizer parameter IDs equals the parameter list, and the dense group contains every block‑10/11 parameter.
3. **Initial state:** `initial_fingerprint` equals the archived `initial_state_sha256` for each seed.
4. **Negative controls, run on CPU:** the qualifier must reject (a) unfreezing after `old.fresh`, (b) a different boundary such as only block 11 or blocks 9–11, and (c) a perturbed block‑9 tensor.
5. **Step‑1 exact parity with the archived control:** ce, rank, loss, scale and the block 12–23 gradient norms must match exactly, and the pre‑clip norm must be strictly larger. This already held for the q/v arm. For seed 179032, ce was 2.399176239967346 in both, the gradients were identical, and pre‑clip went from 48.0517 to 48.0852.
6. **Every step:** block 10 and block 11 gradient norms are positive and finite. `old.step` already checks that frozen tensors have no gradient and an unchanged version counter. All 100 image hashes replay.
7. **End of run:**
   - the 163 frozen tensors equal the pinned F5 source, both in memory and on disk
   - buffers are unchanged
   - all 32 block‑10/11 tensors and every upper group have changed
   - the whole‑model SHA differs from the control's
   - strict 400‑tensor reload gives exact whole, head and packed parity.
8. **Discarded 17‑update run:** scientifically redundant, since the 100‑update job carries every step check, writes the checkpoint only after all checks pass, and runs under the same caps. Keep it (about 40s) only if the ledger process requires it.

I saved this verdict to memory as `sfora-dense-lower-10-11-reject-verdict`.

## engineering_critic: gpt-6-astra

Original consultation 9b877ed462ee4884, group a258d9307a9a4577; status completed, exit 0.

**ADMIT the chosen intervention to bounded qualification. REJECT immediate training admission or any release claim using the existing proofs.** Dense blocks 10–11 are a coherent next experiment; their correctness and resource feasibility remain unqualified.

Concrete findings, in priority order:

1. **P1 — Unfreezing after `fresh()` can silently omit the intervention.** `fresh()` captures both `state["params"]` and AdamW membership after the original 0–11 freeze. Changing `requires_grad` afterward does not add those parameters to optimization, unscaling, clipping, or optimizer zeroing. Comparing the optimizer against that same stale list would still pass. Select the boundary before constructing both lists, then verify exact, duplicate-free IDs against independently enumerated expected names. [pe_large_optimization.py:65](/home/rb/worktrees/sfora-positive-causality/scripts/pe_large_optimization.py:65)

2. **P1 — The inherited frozen digest cannot represent this boundary.** `frozen_state()` includes blocks 10–11 through hardcoded prefixes even when the inventory marks them trainable. I reproduced this with the extracted function and a stdlib-only fake model. Legitimate updates would fail this invariant; deleting the check would remove protection. The replacement must explicitly cover embeddings and blocks 0–9, including named buffers, and exclude 10–11. [pe_core_training.py:51](/home/rb/worktrees/sfora-positive-causality/scripts/pe_core_training.py:51)

3. **P1 — Existing admission evidence does not qualify the dense candidate.** Pilot CPU startup constructs `old.fresh()`, whereas training constructs `mechanics.fresh()`. That was supported by separate adapter qualification; it supplies no dense-boundary proof. The first-17 diagnostic replay also targets the old adapter, and the terminal path explicitly merges adapters. Require the same dense constructor in CPU qualification, discarded mechanics, and pilot execution; bind replay to the new dense mechanics receipt. Initial native tensors may match historical controls while optimizer roles necessarily differ. [train_large_lower_pilot.py:110](/home/rb/worktrees/sfora-positive-causality/scripts/train_large_lower_pilot.py:110), [train_large_lower_pilot.py:158](/home/rb/worktrees/sfora-positive-causality/scripts/train_large_lower_pilot.py:158)

4. **P1 — Preserve corrected objective routing and extend the gradient audit.** The shared step calls `coverage.terms`, whose default skips rank loss for inactive batches; the pilot explicitly substitutes corrected valid-anchor terms. Retain that substitution in every candidate execution path. The shared positive-gradient audit covers only 12–23. Extend coverage to 10–23 and separately verify the newly enabled MLP and attention-output pathways; a positive aggregate block norm alone does not establish their participation. [pe_large_optimization.py:174](/home/rb/worktrees/sfora-positive-causality/scripts/pe_large_optimization.py:174), [pe_large_coverage.py:113](/home/rb/worktrees/sfora-positive-causality/scripts/pe_large_coverage.py:113)

5. **P2 — Runtime feasibility and the cost gate are different checks.** Existing mechanics admits `median × 100 < 269s`; the historical ≤1.50 screen corresponds to approximately **174–175s training wall and 1.732s median update**. A run can fit the execution cap and still fail cost. Re-estimate construction, training, and reload overhead for this candidate. Also move CUDA peak accounting before candidate construction: the current pilot resets it afterward, excluding earlier transient peaks from its report. [train_large_lower_mechanics.py:169](/home/rb/worktrees/sfora-positive-causality/scripts/train_large_lower_mechanics.py:169), [train_large_lower_pilot.py:135](/home/rb/worktrees/sfora-positive-causality/scripts/train_large_lower_pilot.py:135)

**Smallest implementation:** one fixed dense constructor plus an explicit frozen-prefix invariant. Load and authenticate F5 using existing machinery; enable all parameters in blocks 10–11 before building optimizer membership; retain dense LR `1e-5`, head/proxy LR `1e-4`, and existing optimizer settings. Reuse the corrected update, schedule, packing, and scoring logic. This candidate is already native400: remove adapter installation, factor-specific checks, and merge operations from its path. Leave historical boundary helpers and proofs intact.

**Scientifically essential amendment before training:** freeze the estimand as “adding dense adaptation of blocks 10–11 to the original control under exactly 100 updates.” This does **not** isolate MLP causality: compared with the failed adapter arm, placement, parameterization, trainable capacity, and rates differ. No additional arm or sweep is needed for this bounded question. Predeclare that a survivor advances to fresh confirmation and serving qualification, without further selection on this observed panel.

The falsifying qualification should be:

- **CPU, ≤120s:** authenticate source, both seeds’ unchanged initial tensors, native400/head/packed initial parity, exact parameter roles and optimizer IDs/rates. Reject deliberately wrong boundary, missing/duplicate optimizer member, changed source/seed, and a mutated frozen parameter or buffer. Exercise the actual dense constructor.
- **One discarded 17-update GPU run, ≤120s:** prove finite gradients for every optimizer member, positive gradients across blocks 10–23 and the retained head/proxy groups, and actual changes in the newly enabled MLP/output weights. Require frozen parameter/buffer equality, corrected inactive-batch recovery, one optimizer/scaler/bank update per step, archived input replay, and independent native400/whole/head/normalized-packed reload parity. Discard its weights.
- **Only after qualification:** two fresh 100-update candidates with the declared seeds, all 100 archived input hashes replayed, and unchanged fitting/evaluation partitions. Keep both locks, 8GiB/no-swap limits, CUDA allocation below 10GB, and separate 300s pilot/export units. Measure actual cost against the disclosed historical denominators; mechanics estimates cannot certify that final screen. Complete independent full-12,599-image export parity before scoring.

Retain the exact gate: each seed R@1 delta >0 and AP delta ≥0; both mean gains ≥0.20pp; both product-bootstrap lower bounds >0; both cost ratios ≤1.50 per seed. The scorer correctly represents 0.20pp as `.002`. Its `"pass": true` means successful evaluation, so admission must follow `"decision"` and the quality/cost fields. [score_large_lower_pilot.py:81](/home/rb/worktrees/sfora-positive-causality/scripts/score_large_lower_pilot.py:81)

The previous result kills its fixed 100-update allocation, not the architecture universally. Reused-panel intervals remain exploratory and conditional on two seeds. Historical training cost establishes neither contemporaneous overhead nor serving speed. Full protocol-matched SOP **and** In-Shop external quality **and** speed remain mandatory and unmet.

No files edited, Torch imported, GPU/SSH or training jobs started, or official quality results read.

