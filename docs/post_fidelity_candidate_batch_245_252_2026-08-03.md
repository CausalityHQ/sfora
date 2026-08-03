# Post-fidelity cross-field candidate batch 245--252

Date: 2026-08-03. Gate-1/2 audit conducted while the corrected official SOP
baseline was running. No candidate in this batch authorizes implementation or
GPU work.

The batch used the newly measured recipe defects as possible provenance rather
than assuming that a bug fix itself is a method. An independent Claude Sonnet
pass proposed eight mechanisms; repository evidence and primary prior-art
records were then checked adversarially.

| id | proposed mechanism | verdict |
| --- | --- | --- |
| 245 | exposure-equalized proxy radial clocks | **DEAD at Gate 2.** Cosine proxy radius changes the angular normalization Jacobian, but fixing/adapting that gauge is weight normalization, spherical/Riemannian optimization, or layerwise adaptive-rate control. Candidate 34 already rejected exposure-gated proxy updates, and the early partial-run loss difference does not establish retrieval benefit. |
| 246 | nested-dimension dropout | **DEAD at Gates 1 and 2.** No repository measurement says the 512-D budget is the error source. Nested Dropout (Rippel et al., ICML 2014) and Matryoshka Representation Learning (Kusupati et al., NeurIPS 2022) occupy ordered prefixes. |
| 247 | superclass-environment invariant risk | **DEAD at Gate 2.** SOP superclass IDs do not exist with matching semantics on CUB/Cars/In-Shop; IRM (Arjovsky et al., 2019) is the operator, and an invariance penalty is regularization rather than new supervision. |
| 248 | Kalman-filtered per-class anchors | **DEAD at Gate 2.** Adaptive state estimation changes EMA gain per proxy but remains teacher/prototype smoothing. The cross-dataset EMA failure is evidence against spending a run, not provenance for another gain estimator. |
| 249 | balanced incomplete-block batches | **ALREADY DEAD as candidate 204.** The exact proposed operator was audited in `pair_coverage_batch_design_audit_2026-08-02.md`: GCBS, combinatorial designs, incomplete U-statistic SGD, and XBM occupy it, while uniform random sampling already equalizes the first moment. |
| 250 | frozen error-correcting class codewords | **DEAD at Gate 2.** This is ECOC/deep hashing and a fixed-proxy classifier. A combinatorial target supplies no data-derived intra-class relation and changes geometry, not supervision. |
| 251 | train-only local-token MIL | **ALREADY DEAD.** Claude incorrectly treated the -1.47-point Cars MaxSim probe as the only evidence. `region_pa` was itself trained with local evidence and lost about 3.6 points; Bag Exponential Loss, attention MIL, DIML, and DeepEMD occupy latent/top-k bag supervision. See candidates 6, 25, and 230b. |
| 252 | complex phase/magnitude nuisance split or confusability bandit | **DEAD at Gate 1/2.** Nothing identifies phase as a nuisance carrier, so the complex reparameterization has no supervision for its proposed roles. A class-confusion bandit is automated curriculum/hard-class sampling at a coarser index, already an occupied sampling action. |

## Result

There is no shortlist survivor to run. The apparent strongest proposal,
train-only local-token MIL, arose from omitting an existing trained-arm negative
and prior-art entry; restoring those facts kills it. The speed proposal, balanced
block sampling, exactly duplicates candidate 204. This is a useful independent
failure: even when prompted to avoid renamed weighting/mining/regularization,
the generated mechanisms converged back to already catalogued operators.

The next legitimate reopening point is the artifact-verified final SOP
checkpoint. In particular, proxy norm versus class count and per-class retrieval,
superclass-conditioned nearest errors, and class-size-conditioned cohesion are
new measurements only after that artifact exists. They may falsify proposed
mechanisms for free; they must not be converted into a GPU arm before Gate 2.

Primary sources used for the new rulings:

- Salimans and Kingma, *Weight Normalization*, NeurIPS 2016,
  https://arxiv.org/abs/1602.07868; Bonnabel, *Stochastic Gradient Descent on
  Riemannian Manifolds*, IEEE TAC 2013,
  https://doi.org/10.1109/TAC.2013.2254619.
- Rippel et al., *Learning Ordered Representations with Nested Dropout*, ICML
  2014, https://proceedings.mlr.press/v32/rippel14.html; Kusupati et al.,
  *Matryoshka Representation Learning*, NeurIPS 2022,
  https://arxiv.org/abs/2205.13147.
- Arjovsky et al., *Invariant Risk Minimization*, 2019,
  https://arxiv.org/abs/1907.02893.
- Dietterich and Bakiri, *Solving Multiclass Learning Problems via
  Error-Correcting Output Codes*, JAIR 1995,
  https://doi.org/10.1613/jair.105.
- Graves et al., *Automated Curriculum Learning for Neural Networks*, ICML
  2017, https://proceedings.mlr.press/v70/graves17a.html.

Candidate 249's and 251's complete primary-source lists are already preserved in
the cited candidate-204 and candidate-230 audits rather than duplicated here.
