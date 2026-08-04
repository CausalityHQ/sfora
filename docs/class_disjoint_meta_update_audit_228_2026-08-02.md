# Candidate 228: class-disjoint meta-update supervision (CDMS)

Date: 2026-08-02. Status: **DEAD at Gates 1 and 2**. No diagnostic,
implementation, or GPU.

## Proposal

Split sampled training identities into disjoint sets A and B. Virtually update
the encoder with Proxy Anchor on A, then compute a proxy-free differentiable
retrieval loss on B under the updated parameters and differentiate through the
virtual step. The intended object was whether an update learned from one
identity population improves retrieval on a disjoint identity population.

## Gate 1

The cited repository measurements—99.975% proxy-to-own-centroid ownership,
70.303% reverse ownership, 65.308% image-level ownership, and the `0.9656`
versus `0.8865` leave-one-out R@1 split—are all within-training proxy-calibration
statistics. They do not measure whether gradients learned on one identity set
transfer to another. Test identities lacking proxies is the benchmark protocol,
not an observed failure of cross-class update transfer. Candidate 228 therefore
has no matched provenance quantity.

## Gate 2: exact reduction

For `theta' = theta - eta grad L_A(theta)`, Taylor expansion gives

```
L_B(theta') = L_B(theta)
              - eta <grad L_B(theta), grad L_A(theta)>
              + O(eta^2).
```

Thus `L_A + lambda L_B(theta')` is ordinary multi-task training plus, at leading
order, a gradient-alignment regularizer. This is the mechanism of MLDG (Li et
al., AAAI 2018), the first-order meta-learning analysis of Reptile, and
Fish-style inter-domain gradient matching. Keeping the exact meta-gradient
requires `(I - eta H_A) grad L_B`; dropping the Hessian to meet the proposed
cost target removes the distinguishing term to first order and leaves Proxy
Anchor on A plus a proxy-free pair/listwise loss on B.

The application is occupied too. The repository had already recorded Zheng,
Lu, and Zhou, *Deep Metric Learning With Adaptively Composite Dynamic
Constraints* (TPAMI 2023), which uses disjoint-label episodes and held-out
retrieval through a virtual update. MASF applies a metric-style outer loss under
virtually updated parameters; M3L uses meta-train/meta-test identity splits and a nonparametric
meta-test objective because parametric identity classifiers do not transfer.
PADS is bilevel policy learning on held-out DML classes. Candidate 61 in this
repository previously died on the same pseudo-seen/pseudo-unseen construction.

Withholding B's proxies changes neither conclusion: its retrieval loss is an
ordinary supervised pair/listwise objective, and choosing different inner and
outer losses is a standard episodic-meta-learning degree of freedom.

## Verdict

The one-sentence mechanism death is: beyond `PA(A) + ranking(B)`, CDMS adds only
`-eta <grad L_A,grad L_B>`, an occupied gradient-alignment regularizer, while the
cheap first-order implementation deletes exactly that addition. No measurement
or GPU run is warranted.
