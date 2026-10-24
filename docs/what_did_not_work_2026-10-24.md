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
