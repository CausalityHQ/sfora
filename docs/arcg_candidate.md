# Candidate 19: augmentation-response compatibility graph (ARCG)

**Status: DEAD at Gate 4 on 2026-07-31.** The operating-point diagnostic passed,
but the training operator catastrophically removed useful positive supervision.
Gate 3 was committed as
`c652d1c` before implementation, model export, or any ARCG GPU work. Gate 1
provenance and the narrowly live Gate 2
audit are recorded in `post_rspg_candidate_batch.md` and `arcg_prior_art.md`.

## Claim and provenance

ARCG compares how two different same-class images respond to the same fixed
panel of controlled image interventions. Agreement keeps their labelled pair
positive; disagreement changes it to unknown. This changes which supervision
exists rather than adding a loss or changing a similarity score.

Two repository measurements motivate it. RSPG's cross-class signature retained
64.49% of same-class CUB pairs but only 8.66% on In-Shop, showing that rival
identity carries little within-class information on CUB and is strongly
dataset-dependent. Independently, the region experiment improved from 0.5775
with fixed coordinates to 0.6442 with position-tolerant MaxSim (+6.67 points),
evidence that pose/crop displacement is a real nuisance. ARCG therefore derives
structure from within-image intervention responses, not rival classes.

## Registered operating-point diagnostic

The diagnostic dataset is the **official In-Shop training split** (25,882
images). Its representation source is a seed-0 official Proxy Anchor
BN-Inception model after exactly epoch 10 / step 1,440, trained with the
digest-pinned In-Shop recipe, no test evaluation and no checkpoint selection.
The existing embedding pack is insufficient because it contains no model with
which to evaluate interventions. If the checkpoint cannot be reconstructed
exactly, ARCG is blocked rather than decided on a one-step or foreign-dataset
representation.

For each image, evaluate these six deterministic 224-pixel views after the
reference resize-to-256 operation:

1. centre crop (the anchor);
2. horizontal flip of the centre crop;
3. left crop;
4. right crop;
5. top crop;
6. bottom crop.

All use the reference BN-Inception RGB-to-BGR and normalization path. For each
non-anchor view, record the scalar cosine displacement from the anchor,
`1 - cos(z_anchor, z_view)`. The five-element response signature is standardized
per intervention using the training-split median and MAD (with a registered
`1e-6` floor), then centred within image and L2-normalized. An image whose
centred norm is below `1e-6` has no eligible edges. Two distinct same-class
images are eligible exactly when their signature cosine is at least **0.5**.
Base-image distance is not an input to this decision.

This panel tests spatial response only. Photometric interventions are excluded
because their strength would add tunable choices and BN-Inception's BGR-space
preprocessing makes an apparently simple colour transform less canonical.

### Diagnostic prediction and kill rule

The preregistered prediction is an eligible same-class edge density of **0.20**
on In-Shop. The signal is eligible to proceed only if all of the following hold:

- density is in **[0.05, 0.50]**;
- at least **50%** of classes with three or more images split into more than one
  connected component;
- among each class's closest quartile by anchor-embedding cosine distance, at
  least **5%** of pairs are rejected; and
- among each class's farthest quartile, at least **5%** of pairs are accepted.

The last two tests are the registered asymmetry: the operator must reject close
pairs with disagreeing responses and accept distant pairs with agreeing
responses. Failure of any condition kills ARCG before a training screen because
the graph is vacuous, fragmented beyond use, or empirically collapses into
ordinary distance mining.

The expected cost is approximately 20 minutes of GPU time to reproduce and save
the exact epoch-10 model, plus at most 20 minutes for six deterministic passes.
The graph calculation is CPU-only after export.

### Diagnostic result

The seed-0 epoch-10 checkpoint completed at step 1,440. The registered six-view
diagnostic then measured **55,594 / 153,115 eligible edges, density 0.3631**
(prediction 0.20; allowed interval [0.05, 0.50]). All remaining conditions also
passed: **80.93%** of eligible classes had multiple graph components (required
at least 50%), **53.07%** of closest-quartile pairs were rejected (required at
least 5%), and **28.02%** of farthest-quartile pairs were accepted (required at
least 5%). Every image had a valid response signature.

This is a strong mechanism diagnostic: response agreement is not a disguised
embedding-distance rule because it frequently rejects close pairs and accepts
distant ones. It says nothing yet about retrieval quality; only the Gate 4
screen can establish that.

## Registered In-Shop screen

Only if the diagnostic passes, the full ARCG arm will use the same fixed panel,
normalization, threshold, and warm-up epoch; none may be tuned from the
diagnostic. **Prospective correction before implementation or screening:** the
initial Gate-3 text referred to a “refresh schedule” without naming one. That is
an underspecified preregistration. The executable rule is now fixed as graph
construction at epoch 10 and one refresh at epoch 40, both from the current
student. No EMA is introduced because it is not part of ARCG's mechanism. This
correction is uncontaminated by retrieval results: no ARCG training arm exists
and no screen has run. Seed 0 is the Gate 4 screen. The predicted raw best-over-
training In-Shop R@1 is **0.9080**. It is falsified below **0.9059**, which is
0.24 point above the paired Proxy Anchor mean and two times the measured
0.12-point dataset sigma. Raw and selection-corrected values must both be
reported.

A headline win is not sufficient for the novelty claim. Before any second-
dataset escalation, the full method must strictly beat seed-matched controls
for (1) continuous soft response weighting and (2) an ordinary anchor-embedding
distance gate. Failure to beat either control kills the claimed gate mechanism.

## Gate-4 result: objective collapse

The full seed reproduced the diagnostic rather than invalidating it: **55,729 /
153,115 edges, density 0.3640**, multi-component fraction **0.8184**, closest-
quartile rejection **0.5337**, and farthest-quartile acceptance **0.2800**. Thus
augmentation response really does define selective, non-distance intra-class
structure on In-Shop.

The training operator nevertheless failed immediately. Before activation,
epoch-10 raw R@1 was **0.8463** and loss was **2.3593**. In the first hundred
steps after replacing Proxy Anchor's own-class proxy-positive term with the
registered graph-positive term, loss fell to **0.0017** and epoch-11 R@1 fell to
**0.7005**. Loss then reached **0.0005** while R@1 declined monotonically to
**0.6637 at epoch 15**. The run was terminated at that point: a virtually zero
objective supplies no recovery force, and spending the remaining GPU time could
not test the proposed mechanism more clearly.

This is an early mechanistic kill, not a completed benchmark. The partial
epoch-10 best is far below the registered 0.9059 falsification threshold and is
the pre-activation model, not an ARCG result. No final artifact exists, so a
selection-corrected estimate is neither available nor defensible; the protocol's
Gate 6 was not reached. No controls, additional seeds, or second dataset are
permitted.

The mechanism matches but sharpens RSPG's failure. Same-class graph neighbours
already have cosine similarity beyond Proxy Anchor's positive margin at graph
construction, so their detached positive loss is almost zero. Removing the
own-class proxy term therefore removes the only unsatisfied attractive force;
the different-class proxy term is quickly satisfied alone. Keeping the proxy
positive and merely adding or weighting graph pairs would no longer be the
claimed positive-to-unknown gate—it would be another regularizer or soft
reweighting method, both occupied by prior art. ARCG is therefore closed rather
than retuned with a new pair margin after seeing the collapse.
