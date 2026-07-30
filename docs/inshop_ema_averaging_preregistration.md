# Pre-registration: does weight averaging replicate on In-Shop?

**Written 2026-07-30 before any `pa_ema_avg_bnfix` artifact exists.**

## Experiment

Run `pa_ema_avg_bnfix` at seed 0 against the existing current-digest In-Shop
Proxy Anchor seed-0 baseline. The baseline's three-seed mean is 0.9035 and its
across-seed standard deviation is 0.12 pt.

The derived recipe is
`proxy_anchor.inshop.official-51db570.pa_ema_avg_bnfix`, digest
`80f57f183966d6adc868d2db319626d4a87160df7df44a2a82bd5af18d80d0f6`.
It evaluates the EMA weights at momentum 0.99 and EMA-averages the BatchNorm
buffers (`ema_teacher_ema_buffers=True`).

Plain `pa_ema_avg_fast` is prohibited on In-Shop. Its EMA weights would be paired
with the student's last-step BatchNorm running statistics because buffers are
hard-copied by default. That is only a valid control on frozen-BatchNorm CUB,
where the buffers never move.

## Predictions

CUB weight averaging at momentum 0.99 gives +0.414 pt raw and +0.732 pt after
removing the differential best-over-training selection bonus. In-Shop has much
lower run noise and a larger training set, so the prediction is deliberately
smaller:

1. **Raw paired gain:** **+0.20 to +0.50 pt** at seed 0.
2. **Selection-corrected paired gain:** **+0.25 to +0.60 pt**.
3. **Protocol under-credit:** correction increases the averaging gain by
   **+0.05 to +0.20 pt**, because the averaged model should collect a smaller
   winner's-curse bonus than the student.

The run is counted as a clear off-CUB replication if the raw paired gain is at
least **+0.24 pt** (two In-Shop baseline standard deviations) and the corrected
gain is positive. It is counted as a clear negative if the raw and corrected
gains are both at most **+0.10 pt**. Anything between those thresholds is
inconclusive and earns additional paired seeds rather than a positive claim.

## Falsification

- The cross-dataset averaging claim fails if the result is a clear negative.
- The quantitative prediction fails if either raw or corrected gain falls
  outside its registered interval, even if the replication threshold passes.
- The protocol-bias prediction fails if correction does not increase the paired
  gain, or increases it by more than 0.20 pt.

Both raw best-over-training and selection-corrected values must be reported.
The corrected estimator does not replace the benchmark metric; their ranking
difference is itself a finding.
