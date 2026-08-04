# Candidate 368: Correspondence-Distilled Embeddings invention and collision audit

Date: 2026-08-04.

## Isolation correction

The first attempted fourth Fable pass was not independent. Although the CLI
allow-list named only web tools, Claude auto-read `HANDOFF_BRIEF.md` and
`RSPG_TASK.md` from the worktree before recognizing the requested isolation.
It disclosed the contamination and proposed essentially candidate 365 again:
proxy-nullspace relational distillation, with a forecast below the external
horizon and a mechanism contradicted by the In-Shop comparison. That output is
not counted as a candidate.

The corrected pass ran `claude-fable-5` from an empty temporary directory with
filesystem, shell, repository-search, edit, skill, and agent tools explicitly
disabled. It received only the outcome-only problem statement, audited SOTA
lanes, local corrected In-Shop references, and primary-source web access. This
is the first fourth-pass output that satisfies the intended catalogue-blind
design.

## Frozen proposal

Fable proposed **Correspondence-Distilled Embeddings (CoDE)**:

1. use the EMA model's final spatial tokens to compute pairwise entropic
   optimal-transport similarity between training images;
2. train that local similarity with an ordinary supervised metric loss;
3. distil its in-batch row distribution into the deployed GAP descriptor with
   listwise KL; and
4. discard the token head and EMA copy at inference.

The frozen headline prediction was CUB R@1 **0.775 +/- 0.006** at five seeds
with ResNet-50, 2048 dimensions, and 200 epochs. Its proposed pre-training gate
was to use OT for top-20 reranking on existing training-side validation
checkpoints and require at least **+0.8 R@1** before training.

## Gate 1

The proposal does not have positive repository provenance. The directly
relevant corrected measurements point against local matching as the missing
signal:

- the trained `region_pa` arm scored 0.6466 CUB, about **-3.6 points** against
  paired Proxy Anchor;
- changing its fixed-coordinate comparison to position-tolerant MaxSim
  recovered 6.7 points relative to the broken regional score but still did not
  recover the global baseline; and
- the frozen Cars global-to-MaxSim probe changed 0.8306 to 0.8159,
  **-1.47 points**.

The proposal instead inferred its premise from external correspondence methods
and from a cross-lane comparison confounded by backbone, pretraining corpus,
method, descriptor dimension, and recipe. Its information-theoretic claim that
class labels supply only `log2(C)` bits per image is not a valid bound on the
training signal: pretrained features, image-dependent gradients, augmentation,
and within-class input variation remain present. The OT teacher is also not
label-orthogonal after its own supervised matching loss is applied.

## Gate 2

The executable mechanism is already recorded exactly as candidate **146,
local-similarity-to-global distillation**. That audit names both halves:

- Zhao et al., **DIML** (ICCV 2021) computes optimal matching flow between
  convolutional feature maps and uses the resulting structural similarity in
  DML objectives, including Proxy Anchor; and
- Roth et al., **S2SD** (ICML 2021) transfers auxiliary similarity matrices
  into the deployed DML embedding with no auxiliary inference path.

Lebailly et al., **Global-Local Self-Distillation for Visual Representation
Learning** (WACV 2023) is an additional direct local-to-global distillation
neighbour. Candidate 6 independently rejected matched-patch supervision because
DIML already lets cross-image spatial correspondences determine the training
similarity. Replacing candidate 146's MaxSim teacher by Sinkhorn OT, using an
EMA copy, or adding a supervised token loss changes the teacher estimator and
stabilization recipe. It does not create a new supervision relation.

The proposal's numerical case is additionally not a registered additive
prediction: it sums guessed gains for supervised local matching, distillation,
and EMA even though additivity already failed prospectively in this project and
corrected EMA distillation regressed on In-Shop. Its 2048-D/200-epoch forecast
also does not identify a matched local baseline at that capacity.

Primary sources:

- Zhao et al., *Towards Interpretable Deep Metric Learning With Structural
  Matching*, ICCV 2021:
  https://openaccess.thecvf.com/content/ICCV2021/html/Zhao_Towards_Interpretable_Deep_Metric_Learning_With_Structural_Matching_ICCV_2021_paper.html
- Roth et al., *Simultaneous Similarity-based Self-Distillation for Deep Metric
  Learning*, ICML 2021:
  https://proceedings.mlr.press/v139/roth21a.html
- Lebailly et al., *Global-Local Self-Distillation for Visual Representation
  Learning*, WACV 2023:
  https://openaccess.thecvf.com/content/WACV2023/papers/Lebailly_Global-Local_Self-Distillation_for_Visual_Representation_Learning_WACV_2023_paper.pdf

Repository records:

- `docs/recent_fg_operator_scan_2026-08-01.md`, candidate 146;
- `docs/matched_patch_supervision_candidate.md`, candidate 6;
- `docs/post_fidelity_candidate_batch_245_252_2026-08-03.md`, candidate 251;
- `docs/search_stopping_update_2026-08-02.md`.

## Verdict

**DEAD at Gates 1 and 2. No diagnostic, implementation, preregistration, or
GPU.** CoDE is a clean, catalogue-blind rediscovery of the already occupied
DIML plus similarity/local-to-global distillation family. The failed first
attempt is retained as a process finding: a CLI tool allow-list is not an
isolation boundary; future blind passes must run outside the repository with
file and shell tools explicitly denied.
