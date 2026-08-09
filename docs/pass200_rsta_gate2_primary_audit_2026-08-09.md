# Pass 200 RSTA Gate-2 Primary-Source Audit — 2026-08-09

## Scope and verdict

This was a read-only, primary-source audit of the exact RSTA object, data flow,
and training decision. No RSTA diagnostic result or candidate value was
inspected.

**Verdict: LIVE-NARROW.** No audited source instantiates the exact conjunction

`b_i = sum_j K_ij dbar_j -> stopgrad(s_i),  s_i = K_ii dbar_i`

through a differentiated, receiver-specific cosine-plus-log-norm penalty while
leaving sample eligibility and weights unchanged. Broad claims such as "a new
NTK regularizer," "a new functional-motion objective," or "a new use of
influence" are not defensible.

**Exact mechanism sentence:** RSTA differentiates the full contextual
PA-induced receiver motion toward its stopped diagonal self-motion in both
angle and magnitude; nearby work either scores or samples the full motion,
regularizes the kernel globally, attributes influence, or aligns
parameter/input/teacher gradients, but does not use this exact object, target,
data flow, and training decision.

## Collision map

| Primary source | Occupied object and decision | Exact non-collision with RSTA |
| --- | --- | --- |
| [DoCL, AISTATS 2021](https://proceedings.mlr.press/v130/zhou21a/zhou21a.pdf) | Equation 3 gives functional motion under selected samples. Equation 7 scores a receiver by the inner product of its raw residual with its full-data functional motion, then uses the score for weighted curriculum sampling. | This is the closest collision with `b_i` and the raw-cotangent control, but it has no `K_ii dbar_i` self target, stop-gradient, or differentiated cosine-plus-norm regularizer. Its decision point is eligibility. |
| [MGS, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/file/67b0579a7298d9cf39c59404d867bdd7-Paper-Conference.pdf) | Equations 1--3 explicitly contain self motion `K_ii d_i` and batch motion `sum_j K_ij d_j`. Its training action penalizes global empirical-kernel trace or log-determinant/eigenvalue summaries. | It never aligns the two receiver-specific, loss-weighted vectors. This is an ingredient collision, not an operator collision. |
| [NINT v2](https://arxiv.org/html/2511.15487v2) | Equation 7 and Algorithm 1 score the norm of receiver rows of `K g`; Appendix Section 11 specifies VJP followed by JVP, `J(J^T g) = K g`. The score changes coordinate selection. | It has no self vector `K_ii g_i`, direction comparison, stop-gradient, or tangent-field regularizer. |
| [Charpiat et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html) | Defines normalized cross-example Jacobian/kernel similarity by how an update for one input changes another and proposes differentiable training regularization. | It has no contextual loss cotangent, full-batch `b_i`, diagonal self-motion target, or same-receiver cosine-plus-norm loss. It nevertheless occupies direct differentiable tangent-field shaping and is closer than generic Jacobian-norm regularization. |
| [KAR, ICML 2025](https://proceedings.mlr.press/v267/li25br.html) | Aligns an INR's NTK with a derived optimal kernel through a kernel-alignment regularizer. | Its target is a kernel, not a receiver-specific loss-induced motion; it has no `b_i -> s_i` data flow. |
| [Koh and Liang influence functions, ICML 2017](https://proceedings.mlr.press/v70/koh17a.html); [TracIn, NeurIPS 2020](https://proceedings.neurips.cc/paper/2020/hash/e6385d39ec9394f2f3a354d9d2b88eec-Abstract.html) | Estimate how training examples affect a test prediction or loss using Hessian-preconditioned derivatives or checkpoint gradient products, for attribution and debugging. | They do not define a current-batch receiver self target or differentiate a tangent-motion regularizer. |
| [GULF, ICML 2020](https://proceedings.mlr.press/v119/johnson20b.html) | Constructs a guide function by functional or mirror descent and fits the network toward that guide. | The guide is an externally constructed functional-gradient target, not the realized empirical-tangent motion `J J^T dbar` or its diagonal self component. |
| [Gradient Agreement](https://arxiv.org/abs/1810.08178); [Gradient Knowledge Distillation](https://arxiv.org/abs/2211.01071) | The former compares parameter gradients and reweights task contributions; the latter aligns teacher and student input/embedding gradients. | Neither maps total and self-induced fields through one named receiver's output Jacobian in the same model. |
| [DML-ALA, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Deep_Metric_Learning_via_Adaptive_Learnable_Assessment_CVPR_2020_paper.html) | Meta-learns scalar sample weights from a validation objective. | It changes sample weights rather than the empirical tangent field. |
| [Moving in the Right Direction, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/papers/Mohan_Moving_in_the_Right_Direction_A_Regularization_for_Deep_Metric_CVPR_2020_paper.pdf) | Regularizes directions of sampled-pair displacement in the DML embedding space and encourages orthogonality. | Its vectors are current embedding differences, not parameter-tangent motions. This is a close DML direction-regularization analogy, not an exact collision. |
| [DML empirical influence function, NeurIPS 2022](https://openreview.net/pdf?id=ocg4JWjYZ96) | Quantifies training subsets responsible for DML generalization errors and can relabel suspected noisy examples. | Its decision is attribution and data correction, not receiver-specific tangent-field training. |

### NINT implementation caveat

The NINT paper's intended `K g` object is conceptual prior art even though the
[official commit, lines 226--237](https://github.com/chen2hang/NTK_Guided_Implicit_Neural_Teaching/blob/293bcd2/src/nint.py#L226-L237)
unpacks `torch.func.jvp` as `w_flat, _`. PyTorch returns `(primal, tangent)`, so
the released score uses the primal output rather than the claimed JVP. This code
defect weakens the release as evidence that it computes `b_i`; it does not erase
the paper's prior disclosure of the intended object and sampling decision.

## Load-bearing non-collapse condition

If `K_ii` is approximately scalar-isotropic on the descriptor tangent space,
then `s_i` is approximately collinear with `dbar_i`. RSTA's angular target then
collapses to a normalized DoCL-like raw-cotangent target; only its magnitude
calibration remains distinct. The preregistered requirement that
`A_self - A_desc` be positive with a positive bootstrap lower bound is therefore
load-bearing, not a supporting diagnostic.

"Self" must also remain qualified as **receiver-self parameter path with a
contextual PA cotangent**. Because `dbar_i` comes from the full PA batch graph,
`s_i` is not sample-isolated influence.

## Diagnostic/specification correspondence

No fatal Stage-A implementation/preregistration mismatch was found:

- `exact_contextual_rsta_fields` uses the production full-batch Proxy Anchor
  loss, detached proxies, and a differentiable contextual `dbar`.
- `exact_kernel_fields` uses one global VJP/JVP for `b`, followed by serial
  receiver-only VJP/JVPs and extraction of the receiver row for `s`.
- The diagnostic uses normalized descriptors, the full B=180 graph, the full
  trainable encoder parameter tree, and a bufferless train-mode BatchNorm clone,
  matching the registered diagnostic algebra.
- Eight receivers per artificial diagnostic batch versus one receiver per
  training batch is an explicit Stage-A measurement design, not a mechanism
  substitution.

## Training implementation gap

There is no RSTA training operator under `src/`. Therefore the executable
training path does not yet verify complete `stopgrad(s)` semantics, the
one-receiver hash, warmup gating, sample-ID fail-closed behavior, proxy
detachment only inside RSTA, or second-order differentiation through `b`. This
is not fatal to the explicitly pretraining Stage-A diagnostic, but Stage B or
benchmark authorization should require a gradient-level fixture proving those
exact semantics before treating the registered mechanism as implemented.
