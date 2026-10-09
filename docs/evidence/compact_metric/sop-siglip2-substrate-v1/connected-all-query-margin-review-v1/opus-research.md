{"id":"0a22ddae72cc42f9","model":"claude-opus-5-5","status":"completed","exit_code":0}

# Decision after the all-query margin census: option (b), don't qualify the 128→256 continuation

**Verdict:** stop the duration lane. The one next intervention I'd fund is a rank-routed encoder arm at 128 steps (defined below). A one-step gradient check should come first. The original KILL, the closed FS branch, and the FIRST/FULL gates are unchanged.

I edited no repo files and ran nothing native. Every number below is my own stdlib computation on the committed `receipt.json` and the TRAIN128 receipts. The root has not verified them. I saved one memory note about this verdict.

## 1. What the census actually shows

**Evidence:**
- **The shifts are real, not noise.** Comparing the two controls (seed 069 minus seed 061, using seed 061's cohorts) gives median deltas of +0.0009 and +0.0017. The candidate shifts (−0.0086 and +0.023) are 10–20 times larger. Selecting cohorts on the other seed's control gives the same numbers, so regression to the mean doesn't explain them.
- **"Worsened on 1035/1678 correct queries" is not a loss signal at this gallery size.** Margin change ≈ 0.0147 − 0.085 × control margin (correlation −0.34). Grouped by control margin:

| Control margin | Share of queries improved |
|---|---|
| below 0 | 73–82% |
| 0 to 0.05 | 68–69% |
| 0.05 to 0.10 | 54–60% |
| above 0.20 | 22–35% |

  The worsening sits on queries far from the boundary. Please don't cite the erosion counts as the reason to stop.
- **Impostor scores rise almost uniformly.** For the same impostor item, Δscore ≈ +0.057 − 0.018 × control score, which is about +0.047 everywhere.
- **Positive scores rise by level.** Δscore ≈ +0.158 − 0.145 × control score: hard positives gain about +0.08, near-duplicates about +0.02 (cosine is capped at 1).
- **There is a real discriminative part.** At the same control score, positives gain +0.007 to +0.032 more than impostors, on both seeds.

So the candidate is two things together: a gain on hard positives, plus a lift in all similarities that doesn't depend on class.

**Source facts behind this:**
- The MSE target P is the class mean of the starting model's own canonical outputs T (`scripts/train_siglip2_compact_ranking.py:413-420`). The MSE is therefore a pull toward each image's own class centre, for both the canonical and augmented views. It has no term pushing classes apart.
- At step 1, SmoothAP supplies only about 11–13% of the MLP gradient norm:
  - seed 061: fc2.weight 0.49 of 4.50, fc1.weight 0.24 of 1.81
  - seed 069: fc2.weight 0.57 of 4.77, fc1.weight 0.24 of 1.92
  - These are norms with the two views combined in quadrature, not directions.
- **Training loss, last 32 steps:**

| Seed | MSE candidate vs control | Rank loss candidate vs control |
|---|---|---|
| 061 | 0.4735 vs 0.5925 (−20%) | 0.0553 vs 0.0599 (−8%) |
| 069 | 0.4686 vs 0.5853 (−20%) | 0.0524 vs 0.0578 (−9%) |

  The candidate's rank loss was flat over the last two quarters (0.0546 → 0.0553). Its MSE was still falling, but slowly (−2.5%).
- SmoothAP depends only on score differences, so a shift that moves all of a query's scores together costs nothing. It is also scored against a gallery bank cached from the original encoder (`train_siglip2_identity_diversity.py:1239-1249`). So training never sees the gallery side contracting too, and that only happens at serving time.

**Hypothesis (not established):** the pull toward class centres teaches pose and view invariance, which explains the hard-positive gain; 42 of the core44 have a positive in a different pose. Because it is mostly pull with little push, the same update also raises every similarity. That lift is the part a 7× larger official gallery, or SOP's shared layer 26, would punish, and this 1715-image panel can't show it.

## 2. Why not continue

- **The only remaining upside is core44.** The candidate already fixed 11 of 12 seed-061 errors outside it. Projecting any further core44 gain is exactly the dose extrapolation you ruled out, so nothing admissible supports a net benefit.
- **More steps keep the same mix.** The gain and the uniform lift come from the same pull term, so continuing can't separate them. Even a FULL pass would be a dose chosen on an already-exposed panel.
- **It is the most expensive route.** It needs:
  - typed optimizer/RNG resume admission
  - surgery on every hard-coded 128
  - a new schedule authority
  - new boundary mechanics
  - evaluator rebinding
  - about 19 ks of DGX time

## 3. The one intervention: rank-routed encoder

The arm is identical to the candidate except that the encoder MLP gets gradient only from SmoothAP. The MSE still trains A/C, with the same values as before.

**Minimal source scope:**
- In the candidate branch of `update()` (`scripts/train_siglip2_connected_mlp.py:999-1006`):
  - Take `mse` from `trainer.raw_features(context,state,features)`. That path already detaches the encoder: `fullfeature_raw_features` and the readout detach `x`.
  - Take `rank`/`selected` from the live `connected.raw_features(...)`.
  - Two calls to `loss_terms` are acceptable; the extra gallery readout pass is cheap.
- Add a new arm/method identity string and its evaluator binding.
- Everything else stays the same: 128 updates, schedule rows 0–127, learning rates, clip, scaler, views, scope, cached bank.
- The new `update()` has to pass the existing 128-step mechanics again; the other 128-step contracts carry over unchanged.
- The control code path is untouched. Whether the accepted control061 can stand in as `fresh_control` is your protocol call.

**Pretraining falsifier.** Run it inside the CPU qualification that's needed anyway, on the first scheduled B64:
- For fc1 and fc2, compute cos(g_rank, g_mse), where g_mse = g_total − g_rank. Both tensors already exist at step 1 as `ranking_total` and `p.grad` before clipping.
- If the two are strongly aligned, the routing can't separate pull from ranking. Then skip TRAIN and close the MLP encoder lane. The cutoff should be frozen before reading; I'd use 0.5. That is a feasibility constant, not a quality gate.

**Running order and stops:**
1. Seed 061 TRAIN128, export, then the unchanged FIRST stage.
2. FIRST KILL closes the MLP encoder lane, and continuation is not revived.
3. FIRST CONTINUE leads to seed 069 and the frozen FULL stage.

**Descriptive read (decides nothing):** rerun the same census decomposition as above (impostor lift and positive-minus-impostor at matched score) for the new arm against control and against the old candidate.
- **If the gain comes from the pull term (my prior):** the new arm will look like control, with churn-level R@1 changes and an impostor lift within the ±0.002 seed noise. The nearby Source1024 MAIN rank-only128 arm in `docs/inshop_source_main_100_gate_2026-09-28.md` was churn (+0.079pp, 30 gained/25 lost). It is a different trainer, but it points the same way.
- **If ranking drives the gain:** the matched-score gain stays while the lift disappears.

## 4. Confounds

- **Adam:** it normalises steps per coordinate, so a gradient that is only about 12% of the norm still moves the weights at full learning-rate size. A null result could be amplified noise rather than "ranking carries no signal." Check displacement and its coherence against the candidate's 0.828% aggregate.
- **The arm doesn't fix the cached-gallery asymmetry.** Both the candidate and the new arm still score against the frozen bank.
- **Two effects are bundled.** Removing the MSE from the encoder also removes the pull of augmented views toward the canonical centre.
- **Measurement limits:**
  - The census only records the best positive and the top impostor, so the "global lift" reading is indirect.
  - 433 of 1734 top impostors changed item.
  - My gradient shares are norms with views combined in quadrature; they could be up to √2 larger.
- **Scope limits:**
  - The panel has already been read many times.
  - SOP no-harm is unmeasured for any change to layer 26.
  - VAL and official reads remain required before any production claim.

The GPU SHA work is separate, and I didn't review it.

