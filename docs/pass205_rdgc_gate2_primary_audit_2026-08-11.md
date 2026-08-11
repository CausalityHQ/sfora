# Pass 205 RDGC Gate-2 Primary-Source Audit — 2026-08-11

## Authority, scope, and chronology

This is a read-only primary-source audit of the exact Receiver-Diagonal Gain
Calibration (RDGC) candidate frozen at:

```text
path   = docs/pass205_rdgc_candidate_2026-08-10.md
commit = 30d533e532d0f22c8b1e474987001685a4aa3488
sha256 = 2a86f11f8d6a4563610b0585db74c372903bdbf7deabd580fa929114fda2af0f
```

The scientific candidate predates this audit. Its first source implementation
also predates this audit because the missing durable literature authority was
discovered only at the manifest-provenance gate. This audit is nevertheless
prospective to every RDGC scientific observation: no RDGC GPU process,
preliminary, panel, result, or candidate value existed or was inspected when
this document was written. The chronology is disclosed rather than relabelled.

The original candidate required this audit before implementation. That
precondition was breached: source commits through `291ccbf` preceded the audit.
The remedy is not to erase the chronology. The source must be reopened under a
new amendment and repair plan, then independently re-reviewed against this
audit before any manifest or execution.

**Verdict: `LIVE-NARROW` as a bounded prior-art verdict.** Within the frozen
fourteen-source corpus below, no
source applies a differentiated logarithmic penalty to the gain mismatch
between one named receiver's full contextual empirical-tangent response and
the stopped norm of its diagonal receiver path while leaving eligibility,
sample weights, proxies, optimizer metric, and inference unchanged.

This is a bounded novelty audit, not proof that RDGC works and not a claim that
all uses of tangent kernels, gain normalization, preconditioning, or metric
learning are new. Only the exact narrow mechanism remains live.

## Frozen primary-source identifiers

The exact ordered `primary_source_ids` array is:

```json
[
  "pmlr-v130-zhou21a",
  "neurips-2022-67b0579a7298d9cf39c59404d867bdd7",
  "arxiv-2511.15487v2",
  "neurips-2019-c61f571dbd2fb949d3fe5ae1608dd48b",
  "pmlr-v80-chen18a",
  "pmlr-v37-martens15",
  "neurips-2025-4522de4178bddb36b49aa26efad537cf",
  "pmlr-v108-barshan20a",
  "neurips-2023-8249b30d877c91611fd8c7aa6ac2b5fe",
  "pmlr-v162-rame22a",
  "cvpr-2022-kim-adaface",
  "cvpr-2021-meng-magface",
  "cvpr-2019-zhang-adacos",
  "arxiv-1708.03888"
]
```

## Exact candidate tuple

For receiver `r`, with contextual PA cotangents `dbar_j`, the audited object is

```text
b_r = J_r sum_j J_j^T dbar_j
s_r = J_r J_r^T dbar_r
R_RDGC = 0.5 * log((||b_r|| + 1e-8)
                   / (stopgrad(||s_r||) + 1e-8))^2
```

The registered Stage-B diagnostic differentiates `R_RDGC`, norm-matches its
parameter correction to the ordinary Proxy Anchor direction, and evaluates a
frozen 0.10-ratio virtual JVP update. That is a no-training diagnostic decision,
not a training rule. A possible future training form would be
`L_PA + lambda_G R_RDGC`; neither `lambda_G` nor a schedule is selected or
authorized.

## Collision map

| ID and primary source | Occupied object and decision | Exact non-collision with RDGC |
| --- | --- | --- |
| `pmlr-v130-zhou21a` — [DoCL, AISTATS 2021](https://proceedings.mlr.press/v130/zhou21a.html) | Scores examples from residuals and full functional learning dynamics, then changes curriculum weights. | It has no receiver-diagonal `J_rJ_r^T dbar_r` scalar reference and no differentiated log gain-ratio correction. Its decision acts on data weighting. |
| `neurips-2022-67b0579a7298d9cf39c59404d867bdd7` — [Model Gradient Similarity, NeurIPS 2022](https://papers.nips.cc/paper_files/paper/2022/hash/67b0579a7298d9cf39c59404d867bdd7-Abstract-Conference.html) | Exposes diagonal and cross-example empirical-tangent ingredients and regularizes global kernel summaries. | This is an ingredient collision, but it does not compare one receiver's contextual full and diagonal gains through RDGC's scalar loss. |
| `arxiv-2511.15487v2` — [NINT v2](https://arxiv.org/html/2511.15487v2) | Uses the norm of an intended `K g` functional motion to select coordinates. | It has no diagonal receiver reference or differentiated gain-ratio correction; its decision changes selection. The implementation caveat in the bound RSTA audit remains applicable. |
| `neurips-2019-c61f571dbd2fb949d3fe5ae1608dd48b` — [Charpiat et al., NeurIPS 2019](https://proceedings.neurips.cc/paper/2019/hash/c61f571dbd2fb949d3fe5ae1608dd48b-Abstract.html) | Defines cross-example similarity through model tangent behavior and permits differentiable shaping. | It occupies differentiable tangent-field shaping broadly, but not the contextual loss-induced full/diagonal scalar gain object. |
| `pmlr-v80-chen18a` — [GradNorm, ICML 2018](https://proceedings.mlr.press/v80/chen18a.html) | Adapts multitask loss weights by balancing parameter-gradient training rates. | It does not form a named receiver's output-space full/diagonal empirical-tangent gain ratio. |
| `pmlr-v37-martens15` — [K-FAC, ICML 2015](https://proceedings.mlr.press/v37/martens15.html) | Uses a Kronecker-factored Fisher approximation as an optimizer preconditioner. | It changes the parameter-space optimization metric rather than differentiating RDGC's receiver-specific scalar target. |
| `neurips-2025-4522de4178bddb36b49aa26efad537cf` — [NTKMTL, NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/hash/4522de4178bddb36b49aa26efad537cf-Abstract-Conference.html) | Uses extended-NTK spectra to balance multitask convergence. | Its target is task-level convergence balance, not a receiver's contextual full/diagonal gain-ratio loss. |
| `pmlr-v108-barshan20a` — [RelatIF, AISTATS 2020](https://proceedings.mlr.press/v108/barshan20a.html) | Ratios local prediction influence to global model influence for explanatory-example selection. | The ratio is an attribution criterion, not a differentiated current-batch receiver correction. |
| `neurips-2023-8249b30d877c91611fd8c7aa6ac2b5fe` — [Automatic Clipping, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8249b30d877c91611fd8c7aa6ac2b5fe-Abstract-Conference.html) | Normalizes per-example parameter gradients for differentially private optimization without a tuned clipping threshold. | It motivates the registered per-example-gradient-normalized control, but has no receiver output-space full/diagonal tangent gain. This archival record replaces the original candidate's mislabeled 2022 OpenReview submission reference. |
| `pmlr-v162-rame22a` — [Fishr, ICML 2022](https://proceedings.mlr.press/v162/rame22a.html) | Matches domain-level gradient variances as Fisher-information proxies for invariance. | It does not define a named receiver's contextual full/diagonal gain or RDGC correction. |
| `cvpr-2022-kim-adaface` — [AdaFace, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Kim_AdaFace_Quality_Adaptive_Margin_for_Face_Recognition_CVPR_2022_paper.html) | Uses feature norm as an image-quality proxy to adapt the face-recognition margin. | It changes the logit loss from embedding quality; it does not measure an empirical-tangent response. |
| `cvpr-2021-meng-magface` — [MagFace, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Meng_MagFace_A_Universal_Representation_for_Face_Recognition_and_Quality_Assessment_CVPR_2021_paper.html) | Couples embedding magnitude to sample quality through an adaptive angular margin and regularizer. | This occupies magnitude-aware metric learning but not contextual full/diagonal tangent gain calibration. |
| `cvpr-2019-zhang-adacos` — [AdaCos, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Zhang_AdaCos_Adaptively_Scaling_Cosine_Logits_for_Effectively_Learning_Deep_Face_CVPR_2019_paper.html) | Adapts the global scale of cosine logits during training. | It calibrates logit scale, not one receiver's empirical-tangent gain or parameter correction. |
| `arxiv-1708.03888` — [LARS](https://arxiv.org/abs/1708.03888) | Applies a layerwise trust ratio to parameter updates for large-batch training. | It motivates the registered layerwise-trust-ratio control, but does not use receiver-specific output-space full/diagonal gains. |

## Binding to the RSTA Gate-2 audit

The prior audit is frozen as:

```text
path   = docs/pass200_rsta_gate2_primary_audit_2026-08-09.md
commit = 9d0cc9646607e1637f593457a507dce547d7d4b8
sha256 = 3efad753b1328c1a23188dfb1422cf86fa1376e625434a6ea419b24dfc0caf0b
```

RSTA's directional self target did not survive its registered Stage-A gate.
RDGC therefore cannot be described as a continuation of a validated directional
mechanism. It is the isolated magnitude-calibration ablation of RSTA's already
registered cosine-plus-log-norm operator: RDGC drops the failed angular term and
retains the scalar log-norm term. It is not a wholly independent new object.
The bounded prior-art question is whether this magnitude-only ablation has an
exact external collision in the fourteen-source corpus; none was found.

The prior strong full-motion and norm-ratio observations are hypothesis
generation only. They are not RDGC validation evidence. However, the original
candidate's `E_r,180` statistic is effectively the same full/diagonal absolute
log-norm ratio already observed under RSTA (apart from epsilon and fresh rows).
Its preregistered low threshold was therefore contaminated and cannot serve as
a fresh SURVIVE or CLOSE predicate. The repaired diagnostic must retain
`E_r,180` and its seed/context summaries transparently as descriptive evidence
but remove both corresponding decision clauses. No replacement threshold may
be selected from the prior value. The other receiver-specificity, context,
heterogeneity, and global-scalar predicates remain fresh and binding.

## Narrow claim and mandatory falsifiers

The only claim left live is:

> RDGC differentiates the logarithmic gain error between a named receiver's
> full contextual empirical-tangent response and the stopped scalar norm of its
> diagonal receiver path.

Forbidden claims include a new NTK regularizer generally, a new use of
functional motion, a new form of gradient normalization, a new preconditioner,
or a generally novel metric-learning magnitude loss.

The magnitude-only ablation remains live only with all registered panel
comparisons intact:

1. ordinary Proxy Anchor virtual update;
2. raw-cotangent angular correction;
3. generic full-motion damping;
4. batch-global scalar gain;
5. scalar-diagonal/raw gain;
6. per-example-gradient-normalized gain;
7. layerwise LARS-style trust-ratio direction.

The fresh nested-contributor preliminary must close RDGC before the expensive
panel if the receiver-specific diagonal reference aliases a global scalar,
lacks stable context transfer, or lacks material heterogeneity. `E_r,180`
remains recorded but non-gating. If the preliminary survives,
the full panel must show paired lower-bound and seed-consistent improvements in
both alignment and margin slope against Proxy Anchor and every control, while
also rejecting correction-direction aliasing. The repaired CLOSE predicates
take precedence over PASS. No threshold may be tuned from prior RSTA values or
after observing RDGC values.

## Conclusion

No exact external collision was found in the frozen corpus. The verdict is
`LIVE-NARROW`, solely for the magnitude-only RSTA ablation's scalar
receiver-conditioned full/diagonal gain mechanism and only under the repaired
falsifiers above. The audit neither predicts improvement nor authorizes
training, production use, or a scientific run without the separately reviewed
source and manifest handoff.
