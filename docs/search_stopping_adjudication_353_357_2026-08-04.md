# Corrected search-stopping adjudication after candidate 357

Date: 2026-08-04.

## Decision

**No Gate-1-compliant, Gate-2-surviving method is identified under the current
constraints. No candidate GPU run is justified.** This is an evidence-bounded
stopping decision, not a theorem that no future similarity-learning method can
exist.

The decision follows 357 numbered mechanisms, corrected two-seed In-Shop
reference artifacts, prospective functional checks, repeated primary-source
audits, and a final adversarial `claude-fable-5` proof-check. The Fable pass was
read-only, maximum effort, had no web access or model fallback, and was asked to
falsify rather than endorse stopping. An independent primary-source horizon
scan was performed in parallel. It found no omitted mechanism among current
work on potential-field DML, ESA, DADA, Proxy-AN, projected-hypersphere DML,
Shadow Loss, or recent generalized-DML formulations; these were already in the
repository's horizon and mechanism audits.

## Final five falsification attempts

### 353. Within-run dual-head identity-disjoint co-training

Share a trunk, train two heads on disjoint identity folds, and make each head
match the other's relational Gram on identities whose labels it did not see.
This is class-disjoint episodic learning plus relational/mutual distillation and
consistency regularization. It also turns the descriptive seen/unseen gap into
a causal premise. Candidates 61, 228, 336, 348, and 349 already close those
operators. **Dead at Gates 1 and 2.**

### 354. Cross-seed churn regularization

Use the measured top-1 row disagreement to penalize sensitivity to
initialization or a lagged model. A single run cannot observe cross-seed churn;
lagged-self targets are temporal self-distillation and flatness penalties are
regularization. The new measurement also says most errors are stable, not that
the minority churn causes them. **Dead at Gates 1 and 2.**

### 355. Fixed proxy constellation with confusion-aware assignment

Freeze proxies to an equiangular constellation, then assign classes to vertices
by a warm-up confusion cost. Fixed classifiers/gauge fixing are occupied; the
assignment transmits hard-negative-class confusion and is therefore mining.
The verified proxy-margin failure rate supplies no proxy-placement pathology.
**Dead at Gates 1 and 2.**

### 356. Common-random-number augmentation across same-class images

Couple augmentation draws for different same-class images while leaving the
loss unchanged. There is no verified augmentation-covariance pathology, and
coupling changes the variance of an estimator of the same expectation rather
than its supervision. In Proxy Anchor, same-class images interact only through
their shared proxy, so draw covariance has no new pair-level first-order
pathway. **Dead at Gates 1 and 2.**

### 357. Train-time query/gallery role simulation

Split training images into query/gallery roles and optimize a smoothed top-1
event. This is listwise retrieval optimization, already occupied by RS@k and
related ranking losses. Filename-derived roles add dataset metadata, and the
verified training leave-one-out R@1 near 0.9955 leaves almost no unsolved
training event. **Dead at Gates 1 and 2.**

## The stopping argument that survives scrutiny

With label-only training pixels, one model at roughly baseline cost, and one
normalized 512-dimensional cosine descriptor at inference, a gradient-bearing
training objective must choose a referent and a rule for routing gradient to
that referent.

The available referents are:

1. another sample or a collection of samples;
2. a learned class parameter, proxy, centre, or distribution;
3. a transformed or synthesized version of a sample;
4. a past, perturbed, or independently trained model copy;
5. a held-out label fold or episode;
6. the test population;
7. external symbols, annotations, data, or models; or
8. no external referent, leaving a parameter, architecture, or representation
   constraint.

The executable routing families are correspondingly pair/list weighting and
mining; proxy/multi-centre/distributional learning; augmentation, invariance,
equivariance, and synthesis; distillation and ensembles; episodic/meta-learning;
test-time or transductive adaptation; added supervision; similarity/kernel or
architecture changes; parameter-space/optimizer dynamics; and batch/data
construction. The catalogue contains primary-source or corrected empirical
closures for every one. Current Gram geometry, pixel statistics, controlled
input response, training trajectories, cross-seed variation, and batch
stochasticity have also been audited as candidate information channels.

This referent/channel enumeration is the load-bearing argument. Proxy spanning
is not: spanning only supplies a conservation statement and does not prove
operator completeness.

## Required corrections to earlier stopping language

1. The earlier nine-family shorthand omitted architecture,
   parameter/optimizer dynamics, and batch/data construction. Those must remain
   explicit; otherwise the taxonomy is technically incomplete.
2. After reliability audit 321, many historical benchmark deaths are
   quarantined. The stopping weight therefore rests primarily on
   mechanism-level prior-art occupancy, which is unaffected by bad artifacts,
   and only secondarily on the smaller corrected empirical graveyard.
3. The defensible claim is “no presently justified supervision primitive under
   these constraints,” not “no publishable method can ever exist.”

## Reopening boundary

A new arm requires a prospectively validated information channel that creates
a referent outside the enumeration or a primary-source finding that vacates a
load-bearing occupancy ruling. Relaxing the task—extra annotations, external
models, transductive inference, multiple deployed models, or materially higher
cost—can also reopen engineering research, but must be disclosed as a changed
claim.

Fable proposed stratifying the 908 stable In-Shop test errors by pixel and
acquisition covariates as the narrowest diagnostic. That can audit corpus
quality, but it cannot cleanly generate an In-Shop method: stable-error
membership uses official test labels, so designing on it would contaminate the
benchmark. It would require a separately preregistered untouched dataset or
method family for confirmation before any training action. Running it merely
because the GPU is idle would not satisfy the protocol.

