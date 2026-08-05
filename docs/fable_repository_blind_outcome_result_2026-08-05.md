# Repository-blind Fable outcome and adjudication

Date: 2026-08-05.

## Isolation and execution

Fable received only the open-set image-similarity problem, deployment and data
constraints, audited external operating points, and the required standard of
evidence in `docs/fable_repository_blind_outcome_brief_2026-08-04.md`. It was
not given a proposed mechanism or the repository's candidate catalogue. Local
filesystem and shell tools were disabled; only web search and fetch were
available. The model was `fable` at maximum effort.

The first execution spawned four literature-search workers but the command-line
client killed them at its default 600-second print-mode background ceiling. No
scientific result from that interrupted execution is counted. The identical
prompt was rerun with `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0`, allowing the
workers to finish. That corrected execution completed and returned **`NONE`**.

The command used no session persistence. The following is a faithful structured
record recovered from the controller's completed stdout, not a verbatim Claude
transcript.

## Fable's answer

Fable found no method that was both genuinely novel, rather than a port or
component conjunction, and defensibly likely to cross a matched-capacity
benchmark horizon. Its rejected routes included contextual similarity, hubness
and mutual-neighbour rules, local intrinsic dimensionality, shared nuisance
subspaces, augmentation-strength ordinal targets, episodic pseudo-open-set
training, diversity regularizers, ETF/SAM/norm/VICReg and mixed-curvature ports,
GAPan, and external-teacher SAGA.

It proposed one missing measurement instead of a method: train a strong baseline
on class folds and decompose held-out-class retrieval errors into four purported
sources:

1. `E_noise`, label or duplicate artifacts identified by manual audit;
2. `E_geom`, errors corrected by frozen-embedding k-reciprocal, query-expansion,
   or contextual reranking;
3. `E_metric`, errors corrected by a whitened LDA, Mahalanobis, or rank-truncated
   linear metric at ranks 8, 32, and 128 on 512-D and 2048-D features; and
4. `E_feat`, the residual not corrected by the preceding interventions.

Its proposed design was four class folds by three seeds on CUB and Cars196,
optionally In-Shop. It suggested that `E_metric >= 40%` on both datasets could
justify an objective that closes the identity-to-oracle metric gap, whereas
`E_feat >= 60%` would pre-falsify further loss-space work and `E_noise >= 30%`
would indicate a material data ceiling.

## Primary-source correction of the benchmark discussion

Three factual checks were material:

- PFML does report five-run ResNet-50/512-D Recall@1 of **73.4 +/- 0.3 CUB**
  and **92.7 +/- 0.3 Cars196**. This is correctly represented in the repository:
  https://openaccess.thecvf.com/content/CVPR2025/papers/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.pdf
- GCA-HNG reports ResNet-50/512-D Proxy Anchor results of **72.3 +/- 0.3 CUB**,
  **91.1 +/- 0.2 Cars196**, and **93.6 +/- 0.1 In-Shop**, against its own
  **68.5 +/- 0.3**, **88.4 +/- 0.1**, and **92.2 +/- 0.1** baselines. It is an
  IJCV paper, not an unreviewed result, but its In-Shop value does not exceed the
  already-audited 93.9 VAPNet horizon:
  https://arxiv.org/abs/2411.13145
- Fable's suggestion that AdvRF's **76.6 CUB / 94.9 Cars196** values were likely
  bounding-box-cropped is false. AdvRF states that inputs are resized to 256 and
  randomly cropped to 224 during training. Its CUB bounding-box ablation scores
  **73.9**, while the learned AdvRF localization scores **76.6**. The reported
  76.6/94.9 values therefore remain the comparable full-image external horizon:
  https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_Adversarial_Reconstruction_Feedback_for_Robust_Fine-grained_Generalization_ICCV_2025_paper.pdf

The outcome brief's 0.766 CUB, 0.949 Cars196, and 0.939 In-Shop targets remain
unchanged.

## Adjudication of the proposed measurement

The decomposition is a useful way to state the unresolved scientific question,
but it is **not ready to preregister or run**.

First, the four quantities are not a partition as written. A query can be fixed
by both contextual reranking and a learned metric, while a duplicate can also
be fixed by either. An explicit ordered attribution rule would make the buckets
disjoint, but their sizes would then depend on the arbitrary order. A set-valued
overlap report would be honest but would not support the proposed percentage
decision rules.

Second, `E_metric` leaks labels if the oracle metric is fitted and evaluated on
the same held-out identities or images. A nested class split avoids direct
identity leakage but then estimates transfer of one regularized metric estimator,
not an intrinsic metric deficit. A support/query split within held-out classes
instead measures few-shot adaptation, which is outside the zero-shot deployment
problem. Candidate 371 already rejected the corresponding demanded-rank oracle
for exactly this identification failure.

Third, most components are already measured or occupied. The In-Shop
neighbourhood audit measured reciprocity and contextual-error structure;
contextual optimization and distillation are occupied by Liao et al. and Wu et
al. Candidate 310 catalogues Mahalanobis and geometric-space tuning, and
candidate 371 rejects rank-truncated held-out-class metrics as an action map.
The static-checkpoint falsifiers already performed a prospectively specified
near-duplicate audit and exposed how an apparently informative error covariate
can be outcome-defining by construction. Recombining these probes does not
create a new training relation.

Fourth, the proposed action is not identified. Large reranking headroom points
to a transductive operator excluded at deployment; large metric-oracle headroom
does not say which inductive training target will transfer to unseen classes;
and a large residual only says that the chosen probes did not repair the error.

Finally, the cost estimate is understated. Four class folds by three seeds is
**12 training runs per dataset**, before any additional baseline or robustness
arm. Even allowing smaller folds, this is not approximately two ordinary runs
per dataset. CUB plus Cars would consume 24 folded training runs without a
surviving information-to-action map.

## Verdict

**`NONE` is accepted; no candidate, diagnostic, implementation, or GPU run is
authorized.** The proposed decomposition is retained as a clear statement of
unknowns, not as a passed Gate-1 measurement plan. It overlaps candidate 371
and existing error-geometry audits, contains a label-leakage ambiguity, lacks a
well-defined partition, understates cost, and supplies no unoccupied action for
any outcome branch.

This result is still useful. A genuinely repository-blind, web-enabled search
failed to find a defensible method, and its best fallback independently
converged on *measurement before mechanism*. The independent convergence does
not prove that no method exists; it strengthens the evidence that another GPU
arm is unjustified until a measurement has both a leakage-safe estimand and a
novel outcome-to-action map.
