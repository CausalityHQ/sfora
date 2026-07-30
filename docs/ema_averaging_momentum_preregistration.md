# Pre-registration: the short-window weight-averaging sweep

**Written 2026-07-30 before any `pa_ema_avg_m95` or `pa_ema_avg_m90` run.**
This document fixes the predictions, recipe digests, reporting rule, and failure
conditions before the deciding artifacts exist.

## Why this is the priority

Plain weight averaging is the strongest single intervention measured here after
removing each arm's best-over-training selection bonus:

| CUB comparison | raw best-over-training Δ | selection-corrected Δ |
| --- | ---: | ---: |
| `pa_ema_avg_fast` (m=0.99) − `proxy_anchor` | +0.414 pt | **+0.732 pt** |
| `pa_ema_avg` (m=0.999) − `proxy_anchor` | +0.059 pt | **+0.610 pt** |
| `pa_distill` − `proxy_anchor` | +0.658 pt | +0.592 pt |

The reversal is part of the result. An averaged model has a smoother evaluation
curve, so a maximum over training gives it a smaller winner's-curse bonus than it
gives the student baseline. Every sweep result must therefore report both the raw
benchmark value and the leave-one-out-neighbour correction from
`scripts/measure_selection_bias.py`. Neither may be substituted for the other.

Weight averaging is old (Polyak; Izmailov et al., UAI 2018), but the literature
check found no benchmark-matched evaluation of SWA or another weight-averaging
method on Proxy Anchor or HIST zero-shot retrieval. The technique is not novel;
this evaluation is unoccupied.

## Frozen recipes

| arm | momentum | approximate time constant | recipe digest |
| --- | ---: | ---: | --- |
| `pa_ema_avg` | 0.999 | 1000 steps | `0eaa02af1223aaf4f2964d193197f82919dfa7d23f11d733058d4d8e9e80021e` |
| `pa_ema_avg_fast` | 0.99 | 100 steps | `cbccf36a1c8b6ee5108ae72a8ba1a606732c24b4dc37db2f5ce3aab237951c4c` |
| `pa_ema_avg_m95` | 0.95 | 20 steps | `5d2a132d8bf888e41b22f48c2d1c6a2654a209877603ab2d39883bef753f734b` |
| `pa_ema_avg_m90` | 0.90 | 10 steps | `0f636568f74ca7b551595d819448a48e92eebd2b14e8bc4441dbcf0aca15114e` |

All comparisons are paired by seed against the current-digest CUB
`proxy_anchor` artifacts. The sweep runs seeds 0–2 for both new cells; no
adaptive winner selection is allowed.

## Predictions

The mechanism predicts an inverted trade-off. At very slow momentum the average
retains harmful initialisation; at very fast momentum it approaches the noisy
student and ceases to average meaningfully. The two existing points put the
optimum near 0.99.

1. **Raw ordering.** The mean raw gain will order
   `m=0.99 > m=0.95 > m=0.90`, with:

   - `m=0.95`: **+0.15 to +0.40 pt**
   - `m=0.90`: **0.00 to +0.25 pt**

2. **Corrected ordering.** The selection-corrected mean gain will have the same
   order, with:

   - `m=0.95`: **+0.35 to +0.65 pt**
   - `m=0.90`: **+0.15 to +0.45 pt**

3. **Protocol-bias gradient.** The correction will increase the apparent gain
   at both new momenta, but by less as momentum falls:

   - `m=0.95`: corrected minus raw **+0.10 to +0.30 pt**
   - `m=0.90`: corrected minus raw **0.00 to +0.20 pt**

   This follows because a shorter average should smooth the evaluated trajectory
   less and therefore surrender less winner's-curse bonus to the student.

4. **Positive paired signs.** At least two of three paired raw deltas will be
   positive at `m=0.95`. `m=0.90` may include zero or negative seeds because its
   ten-step window is close to the student.

## What falsifies what

- **Averaging as a real intervention fails** if either new arm has a non-positive
  selection-corrected mean, or if fewer than two of three `m=0.95` raw paired
  deltas are positive.
- **The momentum mechanism fails** if `m=0.95` equals or exceeds `m=0.99`, or if
  `m=0.90` exceeds `m=0.95`, on either the raw or corrected mean. That would mean
  the predicted short-window turnover was placed incorrectly or does not exist.
- **The protocol-bias mechanism fails** if correction does not increase the mean
  gain for either new arm, or if the correction gap is larger at `m=0.90` than at
  `m=0.95`.
- Landing outside a numeric interval is recorded as a failed quantitative
  prediction even if the qualitative ordering survives.

Three seeds cannot establish a small effect with the exact sign test: its
two-sided floor is 0.25. This sweep maps the curve and tests mechanism; it does
not turn an n=3 cell into publishable significance.
