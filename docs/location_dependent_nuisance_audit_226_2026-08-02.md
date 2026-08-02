# Candidate 226: location-dependent nuisance field

Date: 2026-08-02. Status: **DEAD before Gate 3**. No diagnostic,
implementation, or GPU.

## Provenance ruling

Candidate 225 found fold-mean cross-identity within/between trace ratios
`0.9312`, `0.9287`, and `0.9345`; random subspaces were approximately `1.0`.
A fixed nuisance-heavy subspace therefore did not transfer. A location-dependent
field is logically compatible with that negative—a field can cancel when pooled
globally—but the result is not positive evidence for such a field.

The tempting residual fact, that real-label source covariances score above the
permuted-label control (`~0.93` versus `~0.70`), is explained by centering. Under
permuted labels the class means approach the global mean and the estimated
within scatter approaches total scatter; its leading directions are therefore
between-class-heavy. Real class centering removes that mass. The ordering
`permuted < real < random` does not establish a transferable nuisance direction.
Candidate 226 consequently fails Gate 1 rather than treating extra model
capacity as a rescue after falsification.

## Independent Gate-2 closure

Had the field existed, its proposed operation was to estimate local within-class
tangent covariance as a smooth function of embedding location, parallel
transport it to a query point, and contract or quotient only along its leading
directions. Every executable form is occupied:

- preconditioning a gradient is a learned local/Riemannian metric;
- penalizing sensitivity along the field is Tangent Prop or the Manifold
  Tangent Classifier with a different tangent estimator;
- transforming the deployed point is a nonlinear nuisance projection/feedforward
  embedding and still ends in the required single cosine vector.

Relevant primary precedents include Simard et al., Tangent Prop (NeurIPS 1991);
Hastie and Tibshirani, discriminant adaptive nearest-neighbour classification
(TPAMI 1996); Sugiyama, Local Fisher Discriminant Analysis (JMLR 2007); Rifai et
al., Contractive Auto-Encoders (ICML 2011) and Manifold Tangent Classifier
(NeurIPS 2011); Hauberg et al., A Geometric Take on Metric Learning (NeurIPS
2012); and Huang et al., Local Similarity-Aware Deep Feature Embedding (NeurIPS
2016). Identity-label covariance rather than a decoder Jacobian changes the
estimator, not the operator.

## Rejected follow-up measurement

An independent audit proposed measuring cross-seed reproducibility of
within-class residual Gram matrices. It is not run. A positive result would
produce a stable scalar relation for each same-class pair; consuming it means
pair gating, weighting, contextual ranking, or an auxiliary agreement loss,
all already occupied. A negative result would close a line already unsupported
by candidate 225. Under the project's Gate-2-before-measurement rule, neither
branch licenses a novel operator, so CPU time and another post-hoc statistic are
unwarranted.

