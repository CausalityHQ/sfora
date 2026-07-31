# CAS — conformal acceptance-set similarity

**Status: DEAD at Gate 1 after a CPU/data-card feasibility check on 2026-07-31.
No implementation or GPU use.** The proposal below was recorded before that
check and the prior-art audit.

## Repository provenance

Three repository measurements constrain the next operation:

1. Proxy Anchor's single positive proxy treats all images of a class as draws
   toward one point, while the datasets visibly contain pose, viewpoint, and
   occlusion modes.
2. Explicitly splitting that support with sub-centre Proxy Anchor lost about
   1.7 CUB points, so hard latent modes are not a supported remedy.
3. RSPG and ARCG found real within-class relational structure but replacing
   positive attraction with selected graph edges self-erased; IPSR retained PA
   but showed that augmentation-response order is not retrieval relevance.

The remaining question is whether the class label should specify a *region of
acceptable support* rather than either a point or selected positive edges.

## Cross-disciplinary mechanism

CAS borrows leave-one-out calibration from conformal prediction. At a frozen
warm-up checkpoint, each training identity defines an empirical distribution
of leave-one-out nonconformity scores. A sample is supervised to lie inside its
own class's calibrated acceptance set and outside every negative class's set.
Once it is safely inside, no force pulls it farther toward a class centre.

The proposed score must use only training identities and must remain symmetric
at retrieval. One concrete form is a smooth leave-one-out k-neighbour energy
within each class, calibrated by that class's empirical quantile. Training adds
hinges on own-class acceptance and negative-class rejection while retaining the
ordinary PA negative term; it does not introduce sub-centres or declare
particular same-class pairs to be positives.

The claimed distinction is supervision by *set membership with finite-sample
calibration*, rather than attraction to prototypes, pair mining, or partitioning
one class into modes.

## Gate-2 attack required

Before any diagnostic or code, search primary sources for:

- conformal prediction used as a differentiable training loss for metric
  learning or image retrieval;
- conformal nearest-neighbour / conformal similarity learning;
- hypersphere, class-region, range, and set-based deep metric losses;
- neighborhood components analysis and kNN surrogate losses;
- one-class and open-set metric learning with class-conditional acceptance
  regions.

CAS is dead if existing work already trains a retrieval embedding by requiring
leave-one-out conformal acceptance for the labelled class and rejection by
other classes. It is also dead if the quantile merely reduces algebraically to
hard-positive mining or a class-radius hinge without conformal calibration
changing the supervision operator.

## Gate-1 feasibility result

The official In-Shop training partition has 25,882 images across 3,997
identities, mean 6.48 images per identity but a sharply concentrated low-count
distribution:

- 12 identities have one image, 10 have two, and 416 have three;
- 1,575 have four images and 671 have five;
- only a small tail supplies enough observations for a resolved empirical
  class-conditional quantile.

Thus the intended class-conditional conformal calibration has only three or
four leave-one-out calibration scores for most usable identities. At nominal
coverage levels relevant to rejection, its attainable p-values/quantiles are
too coarse to express the proposed supervision. The one- and two-image classes
cannot calibrate it at all.

Pooling nonconformity scores across identities would solve the sample-count
problem only by removing the mechanism's defining class-conditional acceptance
set. What remains is a global kNN or class-radius hinge, directly adjacent to
LMNN, nearest-neighbour Gaussian-kernel DML, and hard-positive/range losses.
CAS therefore fails repository-grounded feasibility at Gate 1; it is not
rescued by redefining the candidate after seeing the class counts.
