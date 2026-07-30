# Iterative search protocol for a novel similarity-learning method

**Frozen 2026-07-30 before continuing candidate 1.** This protocol governs the
search until either:

1. a genuinely novel, single-model similarity-learning method produces a
   digest-pinned, preregistered improvement that confirms out-of-sample and
   replicates on a second benchmark; or
2. the measurement record and primary-literature audit support the explicit
   conclusion that no defensible candidate remains.

The standing target is not a new evaluation of an old method. Polyak/SWA weight
averaging can be a valuable measurement result, but it is not a novel method.
Likewise, a transductive or multi-view ensemble does not satisfy the target.

## Why the gates are ordered this way

Fifteen GPU candidates failed. Their failures were structured: they either added
regularisation to an already-fitting base or changed how existing similarities
were scored without changing what supervision existed. Several further ideas
were implemented or developed before prior art was found. The order below kills
unsupported, occupied, or statistically unreadable candidates before they consume
the expensive part of the loop.

For every candidate, run the gates in order and stop at the first failure.
Commit and push the evidence after every gate. Never silently rescue a failed
candidate by changing its mechanism, threshold, or recipe after seeing the result;
that is a new candidate and starts again at gate 1.

## Gate 1 — provenance of the idea

The candidate must follow from a measurement in this repository, not an analogy
to success in another field.

Record:

- the exact measured observation;
- the numeric values that motivate the intervention;
- why the proposed mechanism follows from those values;
- what previously failed mechanism it avoids.

**Candidate 1, `pa_dual_ema`, is the model.** The EMA factorial found that the
distillation-target role prefers momentum 0.999, while the evaluated-average role
prefers 0.99:

| role | momentum 0.999 | momentum 0.99 |
| --- | ---: | ---: |
| distillation target, seed 0 | +0.91 pt | +0.30 pt |
| evaluated average, seed 0 | +0.07 pt | +0.46 pt |

A single EMA must choose one timescale and underserve one role.
`pa_dual_ema` uses a slow 0.999 distillation teacher and a separate fast 0.99
evaluation average.

Failure at gate 1 means the idea is recorded as unsupported and receives no
literature or GPU work.

## Gate 2 — prior art before GPU

Search primary literature for:

- the exact mechanism;
- equivalent formulations under different terminology;
- the combination of its components, not merely each component separately;
- benchmark-matched and adjacent-task uses.

Examples of expensive misses this gate must prevent:

- the teacher BatchNorm fix was EMAN (Cai et al., CVPR 2021);
- gauge fixing was occupied by Hoffer et al. (ICLR 2018) and Pernici et al.
  (TNNLS 2022);
- snapshot/trajectory ensembling was occupied by Huang et al. (ICLR 2017) and
  Izmailov et al. (UAI 2018);
- multi-centre assignment was occupied by SoftTriple (ICCV 2019) and balanced
  prototype assignment by SwAV.

If the method exists, add the citation and equivalence argument to
`docs/method_search_verdict.md`, commit and push, then start the next candidate.
Do not spend GPU to rediscover prior art.

Passing this gate means only that no equivalent method was found after a real
search. It is not proof of novelty; all later claims remain qualified.

## Gate 3 — preregister the deciding result

Before the deciding run, commit and push:

- the exact recipe selector and full digest;
- the paired baseline digest;
- a numeric expected effect;
- a success threshold;
- an explicit falsification condition;
- the seed roles: hypothesis-generating, screening, or confirmation;
- the raw and selection-corrected predictions when they differ.

Four numerical explanations have already been preregistered and all four failed.
That is the protocol working.

No result may be interpreted against a preregistration written after its artifact
timestamp.

## Gate 4 — screen on In-Shop, not CUB

Default screen: one paired In-Shop seed.

- In-Shop across-seed sigma is about 0.12 pt.
- CUB across-seed sigma is about 0.57 pt.
- A CUB comparison needs roughly 5–17 paired seeds for a +0.5 pt effect,
  depending on the specific paired sd.
- A one-seed CUB screen already produced a false +0.52 pt lead here.

For trainable-BatchNorm datasets, every evaluated EMA must average its BatchNorm
buffers as well as its weights. Use `pa_ema_avg_bnfix`-style
`ema_teacher_ema_buffers=True`; plain `pa_ema_avg_fast` is invalid on In-Shop.
If a training teacher is forward-passed, its normalization mode must also follow
the established EMAN-compatible recipe. A half-averaged model is a confounded
screen, not a negative.

The default screen passes only if the preregistered In-Shop threshold is met.
An already-running CUB artifact may be allowed to finish as mechanism evidence,
but it does not replace the required In-Shop screen and must not be promoted from
n=1.

## Gate 5 — confirm out-of-sample

Never quote the seeds that generated the hypothesis as the confirmation estimate.
The Proxy Anchor distillation effect was +0.890 pt on hypothesis-generating seeds
and +0.427 pt on new seeds.

Predeclare fresh confirmation seeds. Report:

- every paired delta;
- mean and paired sd;
- paired t-test;
- exact paired sign test;
- the hypothesis-generating estimate separately from the out-of-sample estimate.

An sd from two or three runs is not treated as stable. In this project a
three-seed paired sd of 0.153 became 0.367 at six seeds.

## Gate 6 — report raw and selection-corrected

Run `scripts/measure_selection_bias.py`, pinned to the current recipe digests.
Report both:

1. raw best-over-training Recall@1, which is the field's benchmark metric; and
2. the leave-one-out-neighbour selection-corrected estimate.

Best-over-training bonuses measured here range from roughly 0.35 to 0.84 pt and
differ between arms by enough to reverse their ranking. Stable methods are
systematically under-credited because their smoother curves collect a smaller
winner's-curse bonus.

The corrected number does not replace the raw benchmark number. The difference
between them is part of the result.

## Gate 7 — replicate on a second dataset

After In-Shop screening and fresh confirmation, run the same frozen mechanism on
Cars196 or CUB with dataset-correct recipes. Preregister the second-dataset number
and failure condition before its deciding artifact.

One dataset is an observation. Two datasets are a result.

The current practical reference levels are:

- CUB: HIST 0.7082, Proxy Anchor 0.6919. A credible single-model arm above about
  0.715 is materially interesting.
- In-Shop: HIST 0.9038, Proxy Anchor 0.9035. A credible single-model arm above
  about 0.9038 must also clear the paired noise and preregistered threshold.

IDEAL does not close the CUB ceiling: its 72.3 is measured from a weak 69.7 HIST
baseline and uses four-view inference. The open ceiling motivates careful search;
it does not lower any gate.

## When a candidate dies

Immediately add it to `docs/method_search_verdict.md` with:

- the motivating measurement;
- the recipe digest and paired baseline digest;
- the gate it failed;
- every raw and corrected number available;
- the mechanism of failure, not just the score;
- relevant prior art.

Commit and push that negative before generating the next candidate.

## Generating the next candidate

Prefer the only class not exhausted by the first fifteen failures: change **what
supervision exists**, rather than merely changing its score, mining weight,
regularisation strength, or inference rule.

Candidate generation still begins at gate 1. Plausibility, elegance, an open
leaderboard ceiling, or novelty of terminology cannot substitute for a motivating
measurement.

## Search stopping rule

Conclude that no defensible novel method remains when:

1. every mechanism directly motivated by the repository's measurements has
   failed a gate or is occupied by prior art;
2. the surviving positive interventions are established techniques or evaluation
   findings rather than novel supervision;
3. no candidate can state a numeric prediction whose causal link to a measured
   defect survives adversarial review.

At that point, do not generate an armchair candidate to keep the loop alive.
Write up the digest-pinned measurement contributions: comparison-specific power,
fixed-seed nondeterminism, winner's-curse reversal, provenance failures, and the
measured cost of missing EMAN behavior in DML.
