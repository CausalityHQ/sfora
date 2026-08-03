# Physics and chemistry operator audit 238

Date: 2026-08-03. Prior art checked online before implementation or GPU work.

- **Boltzmann/free-energy class landscapes:** Bhat et al., *A Free Energy Based
  Approach for Distance Metric Learning* (KDD 2019), already derives a metric
  objective and analytical Boltzmann solution from free energy. Directional and
  probabilistic DML additionally model class distributions on the hypersphere.
- **Renormalization/coarse-graining:** information-theoretic RG networks and
  scale-invariant feature learning already turn preservation across spatial
  scales into an objective. In image DML this executes as multiscale pooling or
  augmentation consistency, with no residual supervision relation.
- **Quantum density matrices:** quantum metric-learning classifiers already
  represent class ensembles as density matrices and compare them with quantum
  kernels. Classical SPD/distributional DML occupies the same executable
  covariance geometry; compressing it to the required single cosine vector is a
  pooling or kernel approximation.
- **Reaction correspondence:** chemical reaction representation learning gains
  its signal from observed atom correspondences between reactants and products.
  The image benchmarks contain no analogous cross-image correspondence channel;
  estimating one from embeddings becomes local matching or reconstruction.

**Verdict: no candidate survives Gates 1 and 2.** Thermodynamic names reduce to
existing energy/probabilistic DML, RG reduces to scale consistency, density
matrices reduce to distributional metrics, and reaction learning relies on an
annotation absent here. No implementation or GPU run follows.

Primary sources:

- https://www.kdd.org/kdd2019/accepted-papers/view/a-free-energy-based-approach-for-distance-metric-learning
- https://www.nature.com/articles/s41567-018-0081-4
- https://pmc.ncbi.nlm.nih.gov/articles/PMC9687469/
- https://link.springer.com/article/10.1186/s13321-026-01201-w
