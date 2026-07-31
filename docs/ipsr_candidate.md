# Candidate 20: interventional principal-stratum ranking (IPSR)

**Status: DEAD at Gate 4 on 2026-07-31.** The preregistered diagnostic passed
and training remained mechanically healthy, but the complete In-Shop screen
failed its absolute performance threshold. Gate 3 was committed before
implementation, inversion measurement, or IPSR GPU work. Gate 1 is in
`post_arcg_candidate_batch.md`; the narrowly live Gate-2 audit is in
`ipsr_prior_art.md`.

## Fixed operating representation and response signature

Use the official In-Shop training split and the already materialized seed-0
official Proxy Anchor BN-Inception checkpoint after exactly epoch 10 / step
1,440. Use ARCG's frozen six-view panel, preprocessing, five scalar cosine
responses, training-split median/MAD standardization, within-image centring,
normalization, and response-agreement threshold 0.5 without modification.

The existing diagnostic retained 36.31% of same-class edges. Re-running the six
forwards is required because only aggregate graph diagnostics—not per-image
signatures—were persisted. This costs approximately 7 GPU-minutes and is a
diagnostic cost, not a training screen.

## Fixed preference construction

For each anchor image `a`, consider only different images with the same training
identity. Let compatible peers have response-signature cosine at least 0.5 and
incompatible peers have cosine below 0.5.

1. Choose `u`, the incompatible peer with highest centre-embedding cosine to
   `a` (ties by stable training-row index).
2. Among compatible peers whose centre cosine is strictly lower than
   `cos(a,u)`, choose `p` with the highest centre cosine (ties by row index).
3. If either set is empty or no such `p` exists, `a` has no IPSR preference.
4. Otherwise register the ordinal target `cos(a,p) > cos(a,u)`.

This chooses the *smallest observed contradiction* to response compatibility,
not the farthest compatible peer. Every retained target is unsatisfied at
construction, but the rule avoids demanding gratuitously large within-class
reordering. There is at most one preference per anchor. No different-class pair,
transformed view, generated feature, or base-distance label enters the target.

## CPU diagnostic prediction and kill rule

After the registered six-view export, preference construction is CPU-only. The
prediction is that **70% of training images** and **70% of identities with at
least three images** receive at least one contradicted preference. The candidate
may proceed only if:

- anchor coverage is at least **50%**;
- class coverage is at least **50%**; and
- mean initial zero-margin Bradley–Terry loss is at least **0.70**.

The per-preference loss is
`softplus(cos(a,u) - cos(a,p))`. It exceeds `log(2) = 0.6931` by construction;
the 0.70 condition catches numerical or implementation mistakes rather than
serving as a tunable effect-size threshold. Failure of any condition kills IPSR
before training. The measured values may not change the selection rule or
thresholds.

### Diagnostic result

The unchanged epoch-10 export produced **16,455 preferences**, covering
**63.58% of training anchors** (prediction 70%, required at least 50%) and
**73.36% of eligible identities** (prediction 70%, required at least 50%). Mean
initial Bradley–Terry loss was **0.7359** (required at least 0.70). All three
conditions passed. The independently rerun ARCG statistics reproduced exactly,
confirming that preference construction used the registered response graph.

## Registered training operator

At the start of epoch 11, build preferences from the current student after ten
completed epochs. Keep the entire Proxy Anchor objective, including its
own-class positive term, unchanged. Add the mean zero-margin Bradley–Terry loss
with fixed weight **1.0**. For each sampled anchor, compare its current embedding
to the detached epoch-10 centre embeddings of its registered `p` and `u`.
Anchors without a preference contribute zero IPSR loss. Refresh signatures,
preferences, and detached targets once at the start of epoch 41 from the current
student; do not use an EMA.

Weight 1.0 is preregistered because the expected ranking loss is approximately
0.7 versus the measured pre-activation Proxy Anchor loss 2.36: it introduces a
material but non-dominant gradient without a tuned scale or absolute pair
margin. The implementation must log preference coverage and ranking loss
separately. If total loss collapses or Proxy Anchor's positive term is removed,
the implementation is invalid rather than a candidate result.

## Registered Gate-4 screen and controls

Screen seed 0 on In-Shop only under the digest-pinned official recipe. Predicted
raw best-over-training R@1 is **0.9080**. The candidate is falsified below
**0.9059**, the same two-sigma-above-Proxy-Anchor threshold registered for ARCG.
Report raw and selection-corrected results if a complete screen reaches Gate 6.

Clearing the absolute threshold does not establish IPSR's novel mechanism. It
must then strictly beat seed-matched controls with the same preference count and
loss:

1. a distance-only ordinal control whose preferences are constructed without
   intervention responses; and
2. a deterministic random within-class inversion control.

Failure to beat either control kills IPSR before extra seeds or a second
dataset. Only a full-method win over both controls may advance to out-of-sample
seeds and then Cars196 or CUB replication.

## Gate-4 result

The full run reproduced the operating signal at activation: **16,303
preferences**, anchor coverage **62.99%**, class coverage **73.31%**, and initial
ranking loss **0.7352**. Unlike ARCG, IPSR retained the full Proxy Anchor loss.
Total loss stayed on its ordinary scale, approximately 1.3–2.7 after
activation, while ranking loss remained nonzero around 0.71–0.73. At the
epoch-40 refresh the signal also persisted: 16,401 preferences, 63.37% anchor
coverage, 73.81% class coverage, and initial loss 0.7223. The candidate neither
collapsed nor erased its own graph.

It nevertheless failed the registered performance condition. Raw
best-over-training In-Shop R@1 was **0.9034 at epoch 51**, below the 0.9059
minimum and below the existing HIST 0.9038. Against paired Proxy Anchor seed 0,
the raw delta was only **+0.091 pt**. Selection correction reports IPSR
**0.9013** versus Proxy Anchor **0.9007** for the shared seed, a corrected delta
of only **+0.060 pt**. The small positive ordering reversal is not decisive:
both deltas are below one measured In-Shop sigma (0.12 pt), and the absolute
registered gate failed.

No distance or random controls, additional seeds, or second dataset are
permitted. Mechanistically, response compatibility identified stable nuisance
strata but did not identify retrieval relevance. The ranking loss fell only
from 0.7352 to roughly 0.715 and forced one same-identity appearance ordering
without improving unseen-identity retrieval. Matching crop/flip sensitivity is
a measurable relation, but it is not evidence that one same-class peer should
rank ahead of another for identity retrieval. IPSR is closed rather than
escalated on its sub-sigma paired gain.
