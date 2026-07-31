# Candidate 1: dual-timescale EMA

Status: **failed gate 4 on In-Shop**.

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

## Gate 3 — preregistration: PASS

**Committed before either In-Shop averaging artifact exists.** The stopped CUB
attempt reached only epoch 5 and emitted no artifact, so it did not reveal a
deciding result.

### Frozen In-Shop recipes

| arm | selector | full recipe digest |
| --- | --- | --- |
| baseline | `auto` | `16a3bc844c81b53dc4d9f501d55d2fa5bbca174318edb5fbfd48107a411ee06a` |
| average-only control | `pa_ema_avg_bnfix` | `80f57f183966d6adc868d2db319626d4a87160df7df44a2a82bd5af18d80d0f6` |
| candidate | `pa_dual_ema_bnfix` | `79f9d35c4eeabf6cda08fc6f5e1b19a1d9440db4e05b4a7de2bc5cb7ec8fde3d` |

Both derived arms use momentum 0.99 for the evaluated average and
`ema_teacher_ema_buffers=True`. The candidate alone adds a slow 0.999
distillation teacher, relational loss weight 1.0, and
`ema_teacher_train_mode=True`. Thus trainable BatchNorm cannot create a
half-averaged evaluation model or the previously measured teacher/student
normalization mismatch.

Seed 0 is a **screening seed**, not confirmation. The current baseline artifact is
0.9024 at seed 0; its three-seed mean is 0.9035.

### Numeric predictions

The honest prediction is that candidate 1 fails its novelty screen on In-Shop.
EMAN-correct distillation is −0.04 pt against Proxy Anchor on this dataset, so
the slow teacher has no measured benefit to add to the fast average.

1. `pa_ema_avg_bnfix − proxy_anchor`, raw: **+0.20 to +0.50 pt**.
2. `pa_dual_ema_bnfix − proxy_anchor`, raw: **+0.20 to +0.50 pt**.
3. `pa_dual_ema_bnfix − pa_ema_avg_bnfix`, raw: **−0.10 to +0.10 pt**,
   centred on zero.
4. Selection correction should increase each averaging arm's gain over Proxy
   Anchor by **+0.05 to +0.20 pt**. The corrected dual-minus-average difference
   should remain within **±0.10 pt** because both evaluate the same fast average.

### Gate-4 success and falsification

Candidate 1 passes the one-seed In-Shop screen only if all are true:

- raw `dual − average-only` is at least **+0.24 pt** (two In-Shop baseline
  standard deviations);
- selection-corrected `dual − average-only` is positive;
- raw dual R@1 is at least **0.9048**, clearing the paired Proxy Anchor seed by
  +0.24 pt and the 0.9038 HIST reference.

It fails candidate gate 4 if any condition is missed. A gain over Proxy Anchor
that is already supplied by `pa_ema_avg_bnfix` is a positive result for old
weight averaging and a negative result for the novel dual-timescale combination.

If it unexpectedly passes, seeds 1–3 are fresh confirmation seeds. Seed 0 will
never be included in the quoted confirmation estimate.

## Gate 4 — In-Shop screen: FAIL

Both preregistered seed-0 artifacts completed on 2026-07-31. The queue stopped
after the candidate for protocol judgement.

| arm | digest | raw best R@1 | selection-corrected R@1 | selection bonus |
| --- | --- | ---: | ---: | ---: |
| Proxy Anchor | `16a3bc844c81` | 0.9024 | 0.9015 | +0.201 pt |
| `pa_ema_avg_bnfix` | `80f57f183966` | 0.9043 | 0.9033 | +0.100 pt |
| `pa_dual_ema_bnfix` | `79f9d35c4eea` | 0.9044 | 0.9040 | +0.037 pt |

The digest-pinned output files are:

- `image_end_to_end_inshop.pa_ema_avg_bnfix.proxy_anchor.inshop.official-51db570.pa_ema_avg_bnfix.80f57f183966_seed0.json`
- `image_end_to_end_inshop.pa_dual_ema_bnfix.proxy_anchor.inshop.official-51db570.pa_dual_ema_bnfix.79f9d35c4eea_seed0.json`

The raw averaging gain over the paired seed-0 baseline was **+0.183 pt** and
the corrected gain was **+0.255 pt**. This misses the preregistered +0.20 to
+0.50 pt raw prediction by 0.017 pt at its lower boundary. The raw dual gain
was **+0.197 pt**, corrected to **+0.332 pt**.

The deciding dual-minus-average comparison was:

- raw: **+0.014 pt** (`0.9044` versus `0.9043`);
- selection-corrected: **+0.077 pt** (`0.9040` versus `0.9033`).

Selection correction therefore behaved in the predicted direction, but the
candidate failed two of its three conjunctive success conditions:

1. +0.014 pt raw is far below the required +0.24 pt;
2. +0.077 pt corrected is positive, satisfying only this condition;
3. 0.9044 raw is below the required 0.9048.

### Mechanism verdict

Separating the EMA timescales does not resolve a hidden conflict on In-Shop.
Once the same BN-correct fast average is evaluated in both arms, adding the slow
0.999 relational-distillation teacher changes raw Recall@1 by effectively zero.
This agrees with the prior three-seed In-Shop result in which BN-correct
distillation was already 0.04 pt below Proxy Anchor. The factorial conflict was
CUB-specific evidence, not a transferable supervision mechanism.

Candidate 1 stops here. It receives no confirmation seeds and cannot support a
novel-method claim.
