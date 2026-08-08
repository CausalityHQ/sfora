# Pass 195 — Cross-Identity Displacement Transport (DEAD at Gate 2)

## Gate 1: measured motivation

The corrected In-Shop transfer audit found a cardinality-matched positive-transfer
deficit of **−0.04968** (unseen-minus-seen nearest-positive performance). A second
audit found that within-class subspace variance is not a pure nuisance axis:
within-class structure explains roughly **35–37%** of target variance versus
**38–40%** of between-class variance. The proposed mechanism would therefore match
distributions of same-class displacement vectors across different training
identities, while ordinary Proxy Anchor handles identity separation. Deployment
would remain one ordinary descriptor.

This is a valid measured premise, but no implementation or GPU run was authorized.

## Gate 2: prior-art audit

The mechanism is already occupied at the level that matters:

* Lin et al., *Deep Variational Metric Learning* (ECCV 2018), explicitly models
  intra-class variation rather than collapsing every class to a point:
  <https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html>.
* *Variance-Preserving Deep Metric Learning* directly preserves intra-class
  variance: <https://openreview.net/pdf?id=ufn8WpXft5>.
* FTL/shared-versus-class-dependent covariance, DiVA/MIC complementary heads,
  Deep Relational Metric Learning, and IAA / *Learning Intra-Batch Connections*
  already transfer or regularize cross-instance relations and variation:
  <https://proceedings.mlr.press/v139/seidenschwarz21a.html>.

Matching displacement distributions across identities is a distributional
variant of the same cross-instance variation-transfer object. It is not clearly
distinct from covariance/variance preservation or relational metric supervision;
the distinction would be cosmetic without a new data flow or supervision source.

**Disposition: DEAD at Gate 2.** No implementation, CPU benchmark, or GPU run.
The measurement remains useful: the positive-transfer deficit and variance ratio
say that simply projecting away within-class variation is unsafe, but they do not
justify another occupied variation-preservation loss.
