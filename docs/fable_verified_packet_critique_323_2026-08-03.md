# Audit 323: Fable critique on the verified evidence packet

Date: 2026-08-03.  Claude Fable was given only the audit-321 corrected
In-Shop facts in a self-contained prompt, with no repository or web tools.  An
earlier Fable call exceeded its budget without returning text; no conclusion is
attributed to that failed call.

## Strongest correction

The restriction is not merely that prior art occupies many geometric ideas.  A
single embeddings/proxies/labels checkpoint exposes only Gram-invariant
statistics.  Here 3,997 proxies in 512 dimensions span the embedding space, so
a scalar-driven first-order correction can be expressed through existing
image/proxy gradient atoms.  Without a new target, input intervention, or
independent temporal observation, such a proposal reduces to weighting/mining.

Fable also corrected the apparent sample size.  The 25,882 rows contain only
129 leave-one-out errors: about 97 in the agreement subset and 32 outside it.
Further conditioning is therefore event-limited even before considering that
only one seed is verified.

## Candidate reductions

Fable returned zero live methods.  Its five proposed CPU measurements reduced
as follows:

1. margin sufficiency: margin weighting/mining if agreement is redundant;
   graph consistency if agreement has residual information;
2. cross-ID near-duplicate inspection: data cleaning or target correction, not
   a novel supervision method;
3. error-direction span decomposition: a combination of existing proxy
   gradient atoms;
4. shared confusion subspace: output projection/whitening regularization;
5. proxy-centroid displacement: proxy consistency or reweighted proxy updates.

Audits 321--322 executed the first two.  The margin test was even less
informative than expected: finite-row leave-one-out error is definitionally the
sign of the image margin, giving complete separation.  The duplicate audit
found one near-identical cross-ID pair and only 2 / 129 affected errors.

## Gate-2 attack on the two stated escape routes

Fable identified input-space/Jacobian evidence and training trajectories as
artifacts that are not present in a static embedding pack.  They are possible
sources of new measurements, but not currently defensible method classes:

- Input-gradient norm control is established generic Jacobian regularization
  (Varga et al., *Gradient Regularization Improves Accuracy of Discriminative
  Models*, 2017).  Within DML, Fu et al., *Deep Metric Learning with
  Self-Supervised Ranking* (AAAI 2021), already creates input transformations
  to simulate and preserve local intra-class structure.  Augmentation/Jacobian
  response used as an invariant, ranking, or tangent penalty is therefore
  regularization/equivariance/self-supervised ranking unless a genuinely new
  target relation is first measured.
- Trajectory-conditioned sample emphasis is curriculum or self-paced metric
  learning.  CurricularFace (Huang et al., CVPR 2020) changes easy/hard sample
  importance with training stage; Balanced Self-Paced Metric Learning (Zhang et
  al., AAAI 2023) adaptively excludes extremely hard/noisy samples by weighting;
  and DML with adaptively composite dynamic constraints explicitly changes
  constraints across training stages.  A stored trajectory alone does not
  create a new supervision relation.

Primary sources:

- https://arxiv.org/abs/1712.09936
- https://doi.org/10.1609/aaai.v35i2.16226
- https://openaccess.thecvf.com/content_CVPR_2020/html/Huang_CurricularFace_Adaptive_Curriculum_Learning_Loss_for_Deep_Face_Recognition_CVPR_2020_paper.html
- https://doi.org/10.1609/aaai.v37i2.26324
- https://pubmed.ncbi.nlm.nih.gov/37018614/

## Decision

There is no live candidate and no justified GPU acquisition from this evidence
packet.  A second seed would improve reliability, but it cannot rescue a
mechanism already reduced to occupied supervision.  The next loop may open only
from a newly verified **non-Gram** training-data measurement that specifies a
new target relation before implementation.  Merely computing Jacobians or
saving trajectories is not enough.

This is evidence that the current search boundary is exhausted, not a theorem
that no novel similarity-learning method can exist.  The honest claim is: after
323 recorded candidates/audits, there is no measurement-motivated,
prior-art-surviving method that warrants another benchmark run.
