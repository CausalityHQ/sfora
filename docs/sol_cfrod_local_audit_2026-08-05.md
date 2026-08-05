# CF-ROD authoritative local audit

Date: 2026-08-05.

Verdict: **DEAD at Gates 1 and 2.** No diagnostic, preregistration,
implementation, or GPU run follows.

Exact artifacts:

- `docs/sol_cfrod_proposal_pass16_2026-08-05.md`
- `docs/sol_cfrod_review_prompt_2026-08-05.txt`
- `docs/sol_cfrod_review_2026-08-05.md`

The proposer and reviewer were independent GPT-5.6-Sol shell-fallback
sessions after their respective native Codex jobs failed before provider
receipt. Both exact answers were extracted from separate Codex JSONL
transcripts and SHA-256 checked before the local verdict.

## Gate 1: identity exclusion does not identify useful factors

CF-ROD's causal premise is identity-conditioned overcompression: supervised
training allegedly removes within-training-identity pose, trim, pattern, part,
or viewpoint axes that later distinguish unseen identities. Its cross-fitted
teacher is trained without the held-out identity; the method class-centers that
teacher's descriptors and distills their residual coordinates.

No verified repository artifact shows that corrected retrieval errors are
caused by this overcompression, that a held-out-identity teacher preserves the
missing useful axes, or that its centered residual predicts error repair. The
verified augmentation-response relation shows that controlled response
structure exists, not that an excluded teacher's raw residual is causal or
beneficial. Corrected In-Shop error overlap likewise establishes repeatable
query difficulty, not residual-factor utility.

For identity `y`, `C_y T_{-y}(X_y)` is simply whatever variation an
ImageNet-initialized network trained to separate other identities retains. It
can encode crop, pose, illumination, background, acquisition series, or
ImageNet features. Centering removes a batch mean; it does not distinguish
future identity information from nuisance. Exclusion proves non-exposure, not
identification. This is the decisive Gate-1 failure.

The proposed `+0.007/+0.004/+0.004` CUB/Cars/SOP gains have no measured
effect-size analogue, pilot, or causal prevalence estimate. They are invented
forecasts, not consequences of repository evidence.

## Gate 2: cross-fitted distillation and every component are occupied

The proposal's novelty search missed its closest primitive. [Dao et al.,
*Knowledge Distillation as Semiparametric Inference*, ICLR
2021](https://arxiv.org/abs/2104.09732) explicitly introduce cross-fitted
knowledge distillation: teachers are trained excluding each fold and queried
only on held-out examples, including image experiments. Their theory has a
defined Bayes-probability nuisance target; CF-ROD has no corresponding estimand
for useful residual factors.

The remaining components are also established:

- [Learning Metrics from Teachers](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_Learning_Metrics_From_Teachers_Compact_Networks_for_Image_Embedding_CVPR_2019_paper.pdf)
  transfers teacher image embeddings.
- [Relational Knowledge Distillation](https://arxiv.org/abs/1904.05068)
  transfers pair distances and angles and reports metric-learning results.
- [Deep Variational Metric Learning](https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html)
  separates class centers from intra-class variation for unseen-class DML.
- [Adaptive Learnable Assessment](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Deep_Metric_Learning_via_Adaptive_Learnable_Assessment_CVPR_2020_paper.html)
  uses disjoint-label episodes to optimize DML generalization.
- [Geometry-Aware Representational Alignment](https://arxiv.org/abs/2509.25253)
  uses centered/Procrustes representational comparisons for distillation.

The exact assembly of identity-level folds, batch-class centering, an EMA
Procrustes map, and PFML may be unpublished. It is a narrow combination of
cross-fitted KD, feature/relational distillation, center-residual separation,
and orthogonal alignment. Without an identified new target or measured causal
premise, that assembly is not a defensible new similarity-learning mechanism.

## Formal and control failures

The independent review's mathematical corrections are accepted:

1. At a class-coded student residual `B=0`, the fold loss is
   `||A||^2/(||A||^2 + 1e-6 n)`, strictly below one, not one.
2. The stated gradient omits the stabilizer and outer four-fold factor.
3. More importantly, a nonzero ambient descriptor gradient need not give a
   nonzero network gradient. Unit normalization projects it through
   `(I - zz^T)`; a radial gradient vanishes. The proposed proof therefore does
   not exclude class-code stationary points in the actual normalized model.
4. Adding `1e-4 I` to the EMA cross-covariance before SVD is not coordinate-
   neutral. Teacher and student heads have arbitrary bases, and the ridge
   biases the alignment toward the identity map.
5. The method calls a two- or four-image augmented batch residual a class
   residual. Its target changes with batch companions and crops and is not a
   full-identity conditional mean.
6. The PFML anchor changes batch size, proxy learning rate, schedule, BN policy,
   and other recipe details, so published PFML means cannot be treated as a
   reproduced paired baseline.

The controls leave the central confounds unresolved. A teacher trained on a
random 75% image subset containing every identity may also see the exact image,
so the proposed leaky control mixes identity exposure with image leakage.
Missing controls include ImageNet-checkpoint-only residuals,
image-held-out/identity-seen teachers, a normal full-data teacher, random
teacher residuals, centered RKD, and exact corrected PFML under the proposed
recipe. The pose/view concentration diagnostic has no legal identity-label-only
definition.

Teacher evaluation mode is also unspecified. If train mode is used during
student training, BatchNorm buffers update on excluded identities and violate
the claimed strict exclusion.

## Conclusion

CF-ROD is an expensive experimental heuristic (~2.83x training compute), not a
Gate-1/2 survivor. Its useful idea—simulate non-exposure at the identity
level—is already within cross-fitted distillation and episodic meta-DML, while
its proposed residual has no identified relationship to corrected benchmark
errors. No GPU is authorized. The next blind proposer remains mechanism-neutral
and does not receive this failure ledger.
