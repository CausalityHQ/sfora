# Protocol judgement — In-Shop averaging arms (2026-07-31)

> **RETRACTED (2026-08-03).** Every score and sigma in this judgement came from
> DeepFashion `img_highres`, not the official centered 256-pixel In-Shop retrieval
> corpus. The instruction to run `measure_selection_bias.py` as a correction is
> also invalid: that script measures a local peak gap, not selection bias. This file
> is retained only as a record of the contaminated decision path and must not guide
> GPU work or method verdicts.

In-Shop seed 0, paired against `proxy_anchor` seed 0 = 0.9024:

| arm | R@1 | Δ |
| --- | ---: | ---: |
| `pa_ema_avg_bnfix` | 0.9043 | +0.19 |
| `pa_dual_ema_bnfix` | 0.9044 | **+0.20** |

## 1. `pa_dual_ema`'s distinctive claim is dead — and it died cheaply

The registered prediction was that decoupling the timescales should land **near the sum
of the two best cells, not near either alone**, and that it fails if it merely matches
what a single EMA already gives.

Dual-EMA beats plain averaging by **+0.01 pt**, against a dataset σ of 0.12. The two arms
are indistinguishable. Whatever the CUB factorial suggested about the roles wanting
opposite timescales, **decoupling them buys nothing on In-Shop**.

That is Gate 3 doing its job: the claim was falsifiable, one 2.2 h run falsified it, and
no confirmation seeds are owed to a hypothesis whose own condition says it failed. Record
it in `docs/method_search_verdict.md` with that mechanism — *decoupling target and
evaluation timescales is not the constraint; a single EMA was not losing anything
recoverable* — and do **not** spend further GPU on dual-EMA.

## 2. Averaging itself is attenuated, not confirmed

+0.41 on CUB, +0.19 on In-Shop. Roughly half, and at ~1.6σ it is not established at one
seed. This is the only part still worth GPU, and only because it is cheap to settle.

**Run `pa_ema_avg_bnfix` In-Shop seeds 1 and 2 — two runs, ~4.5 h.** Nothing else. If the
three-seed mean stays near +0.2, averaging is a real but marginal effect that does **not**
support a method claim, and the honest conclusion is that the CUB result was
dataset-specific. If it rises toward +0.4, it replicates and earns Cars.

Do not run more dual-EMA seeds, and do not start the momentum sweep until this resolves —
sweeping a hyperparameter of an effect that may not replicate is backwards.

## 3. Fix the analyser first — it is currently blind to these arms

`pa_ema_avg_bnfix` and `pa_dual_ema_bnfix` are missing from `ARMS` in
`scripts/analyze_reference_matrix.py`, so both report "incomplete" and no paired t or sign
test is computed for them. Register both against `proxy_anchor`. Check
`PAIRED_CONTROL` — these pair with the plain base, unlike the `narrow*` arms which pair
with their own weakened control.

## 4. Then report both numbers

Run `scripts/measure_selection_bias.py` over the In-Shop artifacts. Averaging arms collect
less best-over-training bonus than the base, so the raw Δ understates them — on CUB that
correction reversed the ranking. Quote raw **and** corrected.
