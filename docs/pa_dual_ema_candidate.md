# Candidate 1: dual-timescale EMA

Status: iterative search protocol in progress.

## Gate 1 — provenance: PASS

The idea comes directly from the CUB EMA factorial, not an analogy.

Against the paired Proxy Anchor seed-0 baseline:

| EMA role | momentum 0.999 | momentum 0.99 |
| --- | ---: | ---: |
| distillation target; evaluate student | +0.91 pt | +0.30 pt |
| no distillation; evaluate EMA weights | +0.07 pt | +0.46 pt |

The roles prefer opposite timescales. At momentum 0.999 the average retains
5.3% of its initialization after CUB's 2,940 steps; this contaminates an
evaluated model whose embedding head began random. At momentum 0.99 the
initial contribution is effectively zero, which helps evaluation, but the
distillation target loses 0.61 pt on the hypothesis-generating seed.

A single EMA must compromise. `pa_dual_ema` maintains:

- a slow momentum-0.999 teacher used only for relational distillation; and
- a separate fast momentum-0.99 average used as the evaluated retrieval model.

This changes the temporal supervision available during training while retaining
the independently measured evaluation average. It does not rescue either of the
two refuted stories about headroom or embedding capacity.

The first attempted CUB run was stopped at epoch 5, produced no result artifact,
and is not evidence.

## Gate 2 — prior art: QUALIFIED PASS

Search completed 2026-07-30 before restarting the candidate. Queries covered
dual/multiple EMA teachers, separate EMA schedules, teacher-role disentanglement,
distillation combined with weight averaging, and EMA evaluation.

Adjacent work is substantial:

- Mean Teacher established an EMA copy as a consistency target
  (Tarvainen & Valpola, NeurIPS 2017).
- SWA and later EMA studies established averaged weights as the deployed model
  (Izmailov et al., UAI 2018;
  [Morales-Brotons et al., TMLR 2024](https://arxiv.org/abs/2411.18704)).
- *Pseudo Labelling for Enhanced Masked Autoencoders* uses separate EMA teachers
  for pseudo-label and reconstruction roles and explicitly allows different
  schedules
  ([Srinivasa, BMVC 2024](https://bmva-archive.org.uk/bmvc/2024/papers/Paper_737/paper.pdf)).
- Dual-EMA self-distillation with two decay rates has been used to provide two
  levels of soft-label supervision outside retrieval.
- Ensemble distillation has been combined with weight averaging into a single
  inference model
  ([Nam et al., ICML 2022](https://proceedings.mlr.press/v162/nam22a.html)).
- Switch EMA feeds averaged parameters back into optimization rather than
  maintaining independent role-specific copies
  ([Li et al., 2024](https://arxiv.org/abs/2402.09240)).

No primary source found uses the exact mechanism: one slow EMA of a single
student solely as a relational-distillation target, while a separate fast EMA of
that same student is the deployed single retrieval model. No benchmark-matched
use was found on CUB, Cars196, or In-Shop.

This is only a qualified novelty claim. The components and the general principle
of role-specific EMA timescales are prior art. What remains potentially novel is
the measured DML-specific conflict between the two roles and this exact
single-student resolution. If the result is not materially above both components,
there is neither an empirical result nor enough distinction from adjacent work
to defend.
