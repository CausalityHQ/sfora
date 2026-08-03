# Audit 322: static-checkpoint falsifiers

Date: 2026-08-03.  Registered before running either deciding statistic.

## Evidence and purpose

Use only the digest-bound corrected official In-Shop final training pack and
seed-0 Proxy Anchor checkpoint admitted by audit 321.  This is not a method
screen.  It tests whether the one surviving descriptive signal contains
anything beyond ordinary margins, and whether its small error count is materially
explained by cross-identity visual near-duplicates.

The pack contains 25,882 images but only about 129 leave-one-out errors.  No
subgroup analysis may be presented as precise merely because the total row count
is large.

## Falsifier A: margin sufficiency

For each image compute, in float64 cosine:

- proxy margin: similarity to its labelled proxy minus similarity to the nearest
  foreign proxy;
- image margin: similarity to the nearest same-class image minus similarity to
  the nearest foreign image;
- the already frozen agreement bit: nearest-foreign-image and
  nearest-foreign-proxy classes agree;
- leave-one-out error.

Standardize the two margins.  Fit an unpenalized binomial logistic base model
with intercept, both margins, both squared terms, and their interaction.  Fit
the nested model with the agreement bit added.  Report both log likelihoods,
the one-degree likelihood-ratio statistic/p-value, coefficient, and event
counts.  The polynomial terms are fixed now to avoid declaring a nonlinear
margin effect to be residual agreement after the fact.

- If agreement does not improve the model at `p < 0.01`, its risk ratio is
  adequately explained by ordinary margin geometry for this diagnostic; using
  it is margin-based hard mining.
- If it does improve at `p < 0.01`, the residual information is precisely that
  an instance graph and a proxy graph concur; acting on it is graph-consistency
  supervision.

Either branch closes the static confusion-agreement line as a novelty source.
The p-value is a descriptive nested-model test on a hypothesis-generating seed,
not confirmatory population inference.

## Falsifier B: cross-identity near-duplicate hygiene

Construct mutual nearest-foreign-image pairs.  Restrict inspection to pairs at
least as similar as the global median nearest-same-class cosine.  On the original
official training JPEGs, flag a pair if its 64-bit DCT perceptual-hash Hamming
distance is at most 4, or if the Pearson correlation between directly resized
64x64 grayscale pixels is at least 0.98.  Record paths, labels, cosine, hash
distance, and pixel correlation for every flag.

Call near-duplicate ambiguity material if at least 13 of the leave-one-out
errors (10% of the approximately 129 known before this audit) have a flagged
mutual foreign nearest neighbour.  This would shrink the already small
learnable event pool; it authorizes data-hygiene reporting, not class merging,
relabeling, or a new loss.  Below 13, this narrow duplicate explanation fails;
it does not prove labels are semantically perfect.

## Fixed interpretation

No outcome authorizes GPU work.  Embedding/proxy/label statistics from one
checkpoint are Gram-invariant observables.  A proposed first-order action must
still be reduced to its actual supervision relation; scalar feedback through
the existing proxy atoms is weighting/mining, not a new primitive.

## Execution defect discovered before accepting a result

The first execution exposed 12 singleton training identities.  Their nearest
same-class similarity and image margin are undefined (`-inf`).  The initial
script standardized those values, allowed NaNs through its optimizer and JSON
writer, and emitted a meaningless `p=1.0` while exiting successfully.  That
margin result is rejected in full.

The correction excludes only rows whose registered margin predictors are
non-finite, reports their row and event counts, fails if too little finite data
remain, and forbids NaN JSON serialization.  The model terms, `p < 0.01`
decision threshold, near-duplicate thresholds, and 13-error materiality rule
are unchanged.  This is missing-predictor handling forced by an undefined
quantity, not a result-dependent threshold change.  A regression test injects
an undefined singleton margin and requires its explicit exclusion.
