# What did NOT work — Sfora retrieval (as of 2026-10-24)

Scope: learned image → top-k retrieval (SOP, In-Shop, Cars, Aircraft) and its serving library.
Section A is measured in the current campaign: each row was pre-registered, used TRAIN-only selection, and has a committed receipt.
Sections B–C come from dated memory notes and committed verdicts of earlier campaigns. They are not re-verified here; check the cited verdict before relying on a number.

## A. Current campaign: SOP SigLIP2 So400m (2026-10-16 → 10-24), all measured

Shared setup:
- **Baseline:** So400m 280 px, 128-D head, 1000 updates, seed 179024. Held R@1 93.659, mAP@R 80.144.
- **Held split:** 5,851 held queries against the full 59,551-image TRAIN gallery, packed int8.
- **Statistics:** Δ is the paired product bootstrap; the 95% interval is in brackets.

| # | Intervention | Result | Decision | Evidence |
|---|---|---|---|---|
| A1 | Equal-weight Large + So400m embedding ensemble (In-Shop held) | ΔR@1 +0.21 [−0.02, +0.44] vs So400m | FAIL | 42c83a6a |
| A2 | Test-time downscaling, no retraining (So400m trained at 384, tested at 280 / 224) | 92.36 / 89.56 held vs Large 92.82; 336 passed | 280/224 FAIL | 800bda09 |
| A3 | 2000 vs 1000 updates (constant LR) | ΔR@1 +0.21 [−0.21, +0.63]; ΔmAP −0.14 | KILL | 36107cfb |
| A4 | Offline gallery flip-averaging (query plain) | ΔR@1 +0.09 [−0.15, +0.32]; both sides +0.14 [−0.11, +0.39] | KILL | 82585309 |
| A5 | Freeze 9 instead of 13 of 27 blocks | ΔR@1 −0.22 [−0.59, +0.15]; +233 s, +5 GB | KILL | 92f5af1b |
| A6 | Bigger substrate: pretrained SigLIP2 giant-opt/16@256 vs So400m@384 | −0.38 pp packed full-gallery R@1 (needed ≥ +1.0) | CLOSE | 3a0b7a75 |
| A7 | Super-category-homogeneous batches (16 products × 4 images, one category per batch) | ΔR@1 +0.14 [−0.21, +0.50]; **ΔmAP −0.58 [−0.97, −0.21]** | KILL (harmful) | 712f3b30 |
| A8 | EMA weight shadow, decay 0.996 (~250-step average) | **ΔR@1 −0.55 [−0.92, −0.17]; ΔmAP −1.49** | KILL (harmful) | 2ba9b4e3 |

What these say:
- **Recipe knobs:** budget, freeze boundary, flip TTA and averaging all sit at a local optimum for this recipe.
- **Batch composition (A7):** the full-fit member bank already supplies global hard negatives, so batch composition does not matter (the XBM argument).
- **Averaging (A8):** at constant LR the trajectory is still improving at step 1000, so averaging lags.
- **Substrate scale (A6):** gains stop beyond So400m.

What did work in the same campaign, for contrast (official SOP TEST, packed int8):

| Model | Official R@1 | Gallery |
|---|---|---|
| Large 128-D | 91.660 | |
| So400m@384 128-D | 93.131 | |
| So400m@280 128-D | 92.594 | |
| So400m@280 512-D | 93.480 | 514 B/row |
| So400m@384 512-D | **93.967** | 514 B/row |

Effective levers were the substrate (Large → So400m) and the embedding width (128 → 512).

## B. Earlier rejected lanes (memory notes; see the cited verdict)

### In-Shop quality
- **So400m substrate at full scale:** official R@1 96.244 vs Large 96.230. KILL on R@1; mAP +0.65 kept as a Pareto point only.
  - TRAIN-held pilot gains transferred poorly to official: +0.67 held became +0.01 official.
- **Teacher-init "retained256", and its direct-supervision replay:** KILL (−0.016 / −0.110). The width gain decays from +2.3 at step 0 to about 0 by 1000.
- **Trained 288 px:** REJECT as a premise. In-Shop images are 256×256, so 288 only supersamples tokens.
- **Freezing more or fewer lower blocks (dense 10–11):** REJECT. The freeze-boundary curve was flat in R@1.
- **All-bank log-NCA replacing SmoothAP:** REJECT.
- **Negative-column margin SmoothAP:** REJECT. An equivalent live-top1 margin had already measured net-zero.
- **32×2 paired-positive sampler:** REJECT. Inert under a detached-bank loss.
- **DINOv2 substrate:** rejected at 43.5% vs 81.7% held. DINOv3 is unavailable (gated).
- **UniCOM / run-averaging family on the In-Shop 1.55 pt gap:** oracle ceiling 96.286 < 96.7 target. Variance/averaging caps at about +0.55.

### SOP before So400m
- **Frozen-source relational/anchoring routes:** capped (factorial verdict, 2026-09-08).
- **Nested-rank proxy-anchor-128:** collapsed to 0.272 mAP@R vs source 0.518 (Adam absent-proxy drift).
- **Depth-folded L/14, slot-pooling readouts, asymmetric query-student vs fixed gallery:** not funded or rejected. Serving is decode- and encoder-bound, with search under 3% of cost.

### Cross-dataset invention passes (Cars, Aircraft, "Pass209", Aug–Sep 2026)
- About 13 invention passes, plus two architecture-relaxation spikes, returned NONE or REJECT. Mechanisms killed by algebraic reduction or by failed falsifiers include FVCG, MET-ADC, ZCC, DRIP, IGVF, CSPD, FRV and UDD.
- Frozen-teacher relaxation was rejected.
- VehicleID was disqualified by a dual-protocol trap.

### Serving / compaction
- **Noise-shaped PQ (64 B):** never beat the deployed iso+sqL2 in absolute terms.
- **Custom cutile-rs encode kernel:** rejected. Encode is host-launch-bound, so the order to try is large batch, then CUDA graph, then torch.compile.
- **Connected-MLP probe arms:** KILL, explained by the Adam coordinate budget (about 1/93 of the MLP's L2 budget).

## C. Process problems that slowed progress (honest account)

1. **Assurance cost dwarfed experiment cost.** Many weeks (Oct 7–15) went into source, lifetime and parity reviews of serving and installed-environment plumbing (reviews: 2775cc31, a4cd794c, de10686a, 2ea0535f…). Few quality measurements came out of that period. Most real quality movement came in the 8 days of short TRAIN-only pilots.
2. **Invention-first instead of measurement-first.** Long chains of consult/critique passes produced NONE verdicts. The cheapest falsifier was usually a single 30-minute TRAIN pilot.
3. **Pilot-to-official transfer misread on In-Shop.** Full-TRAIN models contaminate the TRAIN-held split, so held gains there did not predict official. SOP held gains transferred fully, at 1.0–1.15× the held delta.
4. **Operational traps:**
   - A bare `systemd-run --user` unit dies when the last ssh session closes (Linger=no). Fix: a detached `setsid` driver running `systemd-run --wait`.
   - Piping stdin into ssh together with `</dev/null` produced empty files.
   - Monkeypatched defaults are overridden when callers pass the argument explicitly (`interpolate_pos_encoding=False`).
   - In torch 2.12, `torch.optim.optimizer` must be imported explicitly.
5. **Delegation stalls.** Specialist children died at MCP initialize, or blocked on a codex "Hooks need review" prompt that the owner had no supported control to decline. Message queues to the orchestrator were frequently full.
6. **Pinned-source friction.** Receipts pin historical trainer bytes, so every change needs a new wrapper and staged source. Done consistently, this kept the provenance valid.

## D. Added 2026-10-24 (later)

- **The premise rejection "trained 288 px on 256 px In-Shop only supersamples" was wrong.** It was never measured.
  - Pretrained So400m/14 at its native 384 beats 256 by **+5.67 pp** packed full-TRAIN R@1 on In-Shop TRAIN (447bddc4).
  - Every In-Shop So400m run (official 96.244) had been run at 256, downscaled from the model's native 384.
  - Lesson: measure a cheap zero-training screen before rejecting a lever on argument alone.
- **`torch.compile` (max-autotune) on top of fp16-native serving: +2.5 % end to end, discarded** (bar was 5 %).
  - End-to-end encode is now partly bound by decode and preprocessing.
  - The next serving lever is the input pipeline, not kernels.
- **Process: a specialist child's brief never became a turn.** The child sat idle for 30+ min with its durable message still queued.
  - The child was stopped and the work was done in-root, without a duplicate child.
- **In-Shop 280 px efficiency point: KILL** (735d57ed). Held R@1 −0.23 [−0.39, −0.06] and mAP@R −1.33 vs the 384 pilot, which misses the −0.30 non-inferiority bar.
  - In-Shop needs 384 more than SOP did. Resolution is the lever that won In-Shop, so cutting it gives the gain back.
  - Efficiency has to come from the arithmetic at 384, not from fewer tokens.
- **In-Shop Large/16@384 as a faster substitute: KILL** (3b289e72).
  - Pretrained, 384 helped Large as much as So400m: +5.01 pp vs 256.
  - Fine-tuned at 512-D: held 97.94 vs So400m@384 98.61 (−0.67 [−0.88, −0.46], mAP −3.46). That is also below So400m@256 (98.02).
  - **Lesson:** "substrate ≈ 0 at matched 256" (native256 campaign, 128-D, older recipe) does not carry over to the 512-D member-bank recipe. Substrate and resolution are both needed.
  - The efficiency gap is now a model-level problem. Remaining options: distill So400m@384 into a cheaper student, or token reduction inside So400m.
- **Training-free token merging (ToMe, start layer 4, r 24) on So400m@384: KILL.**
  - GPU speed 1.21× (bar 1.30×). Held In-Shop R@1 −0.86 [−1.07, −0.66].
  - End to end it was 2× *slower*: the gather/scatter merges break the compiled graph.
  - With the 280 px KILL, In-Shop retrieval is token-sensitive at 384. Dropping or merging tokens costs accuracy.
- **Efficiency state after five falsifiers** (280 px, Large@384, ToMe killed; workers and fp16/FP8 kept):
  - The serving frontier is So400m@384 FP8 at ~101 img/s e2e / 18.5 ms p50.
  - Remaining options all need training: a distilled cheaper student, or merging-aware fine-tuning. None is pre-registered yet.
- **Distillation So400m@384 → Large/16@384 (cosine + batch-similarity MSE, weight 1.0, 1000 updates): KILL.**
  - Held 97.97 vs undistilled Large 97.94 (+0.02). Vs the teacher pilot: −0.65 [−0.86, −0.44], mAP −3.41.
  - Embedding-level distillation at this budget doesn't transfer the So400m advantage. The Large substrate ceiling under this recipe is about 97.95 held.
- **Efficiency status:** the cheap and medium falsifiers are exhausted (280 px, Large@384, ToMe, distillation). Kept efficiency wins: workers, fp16, compile at 384, and FP8 (end to end 42 → 101 img/s, p50 52 → 18.5 ms).
  - The remaining distance to Large-level speed (~190 img/s) has no causally supported cheap lever left in this campaign.
  - Longer-horizon options, unscreened: merge-aware fine-tuning, structured pruning of So400m blocks, and FP8 attention kernels. Each is a multi-hour project with a low prior from today's evidence.
- **Depth pruning So400m@384 (drop the last 8 of 27 blocks, fine-tuned): KILL by margin.**
  - Held 98.26 vs full depth 98.61 (−0.35 [−0.53, −0.17]; bar −0.30), mAP −0.87.
  - It is still the best speed/quality trade found: it beats So400m@256 (98.02), Large@384 (97.94) and distilled Large (97.97) at ~1.4× fewer FLOPs.
  - If the operator accepts a ~0.35 pp R@1 cost, this is the lane to fund: dropping fewer blocks (4–6), or depth pruning combined with FP8.
- **Resolution above 384 (pretrained So400m 448 / 512): CLOSE.** 448: −0.18 pp. 512: +1.05 pp (bar +2.0). Resolution gains saturate at the model's native 384.
- **1024-D head (In-Shop So400m@384 pilot): KILL.** Held R@1 −0.15 [−0.27, −0.03] vs 512-D; mAP −0.05 (not significant). Width saturates at 512.
- **In-Shop exposure beyond ~5 epochs: KILL.** Pilot 1500 vs 1000 updates: +0.08 [−0.07, +0.22], mAP +0.31 (not significant). No 3000-update full run. The 2000-update full-TRAIN incumbent (97.039) sits at the exposure optimum.
- **SOP exposure on the 384/512 recipe (2000 vs 1000 updates): KILL.** −0.03 [−0.35, +0.29], mAP −0.06.
  - SOP saturates at ~1.2 epochs, unlike In-Shop (~5 epochs). With ~11k SOP products seen once each, the member bank already covers the data.
- **SOP full-TRAIN (59,551 rows) vs the 90% fit split, 384/512, 1000 updates: KILL.**
  - Official 93.721 vs 93.967 (−0.25 [−0.36, −0.14]), mAP −0.45.
  - Likely causes: at fixed updates, 11% more products dilutes per-product exposure, and seed variance (SD ~0.2 pp) may contribute.
  - SOP data and exposure levers are both closed for this recipe.
- **Parity lesson:** official receipts hold both `float_quality` and `packed_quality`. Always gate on `packed_quality` (the headline number).

## E. Fundamental-redesign screens (2026-10-31; contextless Fable proposals, training-free, all measured)
- **Token MaxSim re-rank (9×9 final-block tokens, PCA256, fused with the global score): KILL, harmful.** −0.62 [−0.95, −0.34], mAP −1.28.
- **Background context vector (1 − attention pooling): KILL, harmful.** −0.36, mAP −0.87. Session/background cues are not a free signal.
- **Attention-guided zoom second view: KILL.** Zoom alone −0.44; fused −0.09 / −0.19 (not significant), mAP worse. The median attention box already covers 78 % of the frame.
- What survives: held R@32 is 99.33 % vs R@1 94.92 %, so the re-ranking ceiling is 4.4 pp. Untrained token evidence only adds noise; only a *learned* re-ranker remains untested in this family. Evidence 7c88f332 + sop-so400m-token-cache-screen-v1.
- Ensembles of the same recipe: the two-seed SOP ensemble IS kept (official 94.129, +0.16), but only about half the held gain (+0.34) transferred.
- **Learned token-pair re-ranker (2-layer cross-attention over 81+81 cached tokens, top-32): KILL on R@1.** fp16 −0.07 [−0.63, +0.46]; int8 0.00.
  - mAP@R improved significantly: +0.86 [+0.31, +1.39], R@10 +0.24. That is a Pareto note at +21–41 KB/image, not a kept change.
  - Overfit FIT lists early (best inner-val at step 4k of 27k).
  - With the training-free token/context/zoom KILLs, the "pooling discards information" family cannot move rank-1 on this system. Remaining rank-1 errors behave like near-duplicate ambiguity. Evidence c4df2d53, sop-so400m-token-reranker-v1.
- **VLM pairwise verifier (Qwen3-VL-8B, top-3, margin-gated): NO-SIGNAL.** +0.07 [−0.14, +0.28]; 22 fixed, 18 broken. On hard pairs the VLM is a weaker judge than the embedding.
- **Error anatomy (the important result).** About 2/3 of inspected held errors are detail or label close-ups: stamps, rating plates, parts, undersides. They show little of the product and their positives are full shots. Near-duplicate listings are NOT the main residual.
  - Every single-image re-scoring idea (tokens, zoom, context, learned re-ranker, VLM) is therefore closed for rank-1.
  - Next levers must use legible text or cross-image structure. Evidence: sop-vlm-verifier-probe-v1.
- **OCR text index (diagnostic): CLOSE.** Errors whose query text matches a positive but not the wrong top-1: 2.4 % [1.1, 4.8] (bar ≥ 10 %). Text overlaps the wrong image more often (19.5 %) than a positive (12.5 %). Evidence abba3547, sop-ocr-text-overlap-diag-v1.
- **Gallery competition / hubness normalisation (ρ, typed by detail-ness, leave-query-out, CV): KILL.**
  - Full gallery: +0.10 [−0.12, +0.34]. Held-only: mAP −0.33.
  - Close-ups are not distinguishable hubs (AUC 0.55).
- **Multi-view token coverage, ORACLE ceiling with true product clusters: KILL.** −0.55 [−0.80, −0.32] R@1. Union over views adds +0.02 over the best single view. Even perfect clustering cannot make token evidence fix rank-1.
- **Provenance / JPEG-metadata kernel: not run, rejected on validity.** It exploits upload-session encoding artefacts, not product appearance.
- Evidence e811cc9a, sop-gallery-d2-d3-screen-v1.
- **Detail-crop training (RandomResizedCrop scale 0.15–1.0 instead of 0.8–1.0), 384/512, seed 179024: KILL.** Held R@1 +0.00 [−0.30, +0.29], mAP +0.06 ns. Fixed 34 baseline errors, broke 34. Patch verified active (different first batches, first loss 9.86 vs 5.49). Training-data exposure to close-ups does not fix the close-up → full-shot errors. Evidence 923c69eb, sop-so400m-384d512-detailcrop-179024-v1.
- **Third ensemble member (detail-crop model, or 179024@2000) added to the 2-seed incumbent: KILL.** R@1 −0.10 [−0.29, +0.08] / −0.07; mAP +0.17 significant (mAP-only Pareto note at 3× encode). Ensembling saturates at 2 members for rank-1. Evidence e658a480, sop-so400m-384d512-ensemble-member-screen-v2.

## F. Redesign proposals v2 screens (2026-10-31; contextless specialist 5814c6f4, all on cached held features)
- **Role-conditioned pair scoring (per-image "close-up-ness" scalar, ~2 params on top-32): KILL.** Oracle pre-kill passed (E1 = 88/5,851: errors where every item above the positive is more atypical than it), but a learned role (ridge on FIT, corr 0.61 with oracle) cross-fitted gives R@1 −0.05 [−0.28, +0.19], top-32 mAP −0.62 [−0.78, −0.47]; β flips sign between halves. 24 fixed / 27 broken.
- **Scene-context second vector (object-suppressed pooled frozen tokens): KILL at pre-kill.** On 255 errors with the positive in top-32, context similarity prefers the true positive over the wrong top-1 only 21.6 % (bar 65 %). Context points to the wrong product.
- **Listing-level index (score + β·sim to the rest of g's listing), ORACLE listings: KILL.** −0.36 at β 0.1, worse for larger β. Together with the D3 oracle token union, gallery set structure cannot help rank-1, even with perfect clusters.
- Not run: asymmetric close-up adapter (#3, an information-limit risk; detail-crop training already showed no net gain) and regional gallery descriptors (#4, same family as token MaxSim/zoom KILLs). Evidence: scripts diag_sop_role_order_e1_v1.py, screen_sop_role_rescore_v1.py; proposals docs/fundamental_redesign_proposals_v2_2026-10-31.md.
- **Freeze 18 of 27 blocks (to share more trunk in the served ensemble; −14 % encode predicted): KILL.** Held R@1 −0.27 [−0.54, 0.00], mAP −0.90 [−1.22, −0.58]. The freeze curve is not flat above 13 on SOP. Evidence 425d4513, sop-so400m-384d512-freeze18-179024-v1.
- **Static per-layer FP8 activation scales (instead of dynamic per-row) on the served ensemble: KILL.** 1.065× GPU-only (bar 1.10×), drift max 0.0145. The quantisation reductions are already fused with GELU/LN by the compiler. Evidence: sop-shared-trunk-ensemble-v1.
- **Segment compile of the FP8 ensemble (whole trunk / upper-stack graphs): KILL as default.** +4.7 % e2e, but held R@1 95.078 < gate 95.166. Kept as opt-in `compile="segment"`.
- **NVFP4 / MXFP8 block-scaled GEMMs on GB10 (cuBLAS via F.scaled_mm): no speed lever.** fc1 139 (FP8 rowwise) vs 98 (MXFP8) vs 133 (NVFP4) TFLOPS; fc2 194/184/218; qkvo 127/137/131. Inductor has no Triton FP8 GEMM template on sm_121.
- **Fused FP8 encoder (Triton add+LN/GELU/attn quant, fused QKV, fc1+GELU→MXFP8): KILL on held gate.** 1.171× GPU-only, e2e 76.3 vs 63.7 img/s, b1 p50 −18 %, but held R@1 95.129 (mx) / 95.061 (plain) < 95.166. FP8 variants all sit within ±0.07 of each other, so the point gate is noise-limited (fix the gate design before the next numerics change, never post hoc).
- **Per-layer FP8 serving on the 2000-update In-Shop incumbent: FAIL parity** (net −12/14,218; it passed on the 1000-update model). Replaced by the fused FP8 encoder (net 0, +9.9 %). Evidence inshop-fused-fp8-v1.
- **SOP single-model fused FP8 (mx) as a speed Pareto point: FAIL parity.** Net −18 (gate −13/60,502), 113.5 img/s; 240 fixed / 258 broken. FP8 churn on SOP near-ties is ~500 flips regardless of kernel.
- **H2D prefetch on a side stream for the served SOP ensemble: KILL.** Bitwise-equal output, but base 64.6–65.6 vs prefetch 64.4–65.4 img/s: the copy is already hidden; the e2e gap to GPU-only is decode/worker-bound and small.
- **SOP ensemble on CUTLASS FP8 kernels: FAIL held gate.** 95.044 < 95.166 at 93.3 img/s e2e (+46 % vs served 63.7). Same SOP FP8 near-tie churn.
