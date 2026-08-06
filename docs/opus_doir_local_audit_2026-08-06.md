# Local protocol audit: DOIR (Pass 28)

Date: 2026-08-06. Frozen proposal: `docs/opus_doir_proposal_pass28_2026-08-06.md` at commit `a8650cf`. Independent review consultation: `c97fba8c9fa94e66` (started only after the proposal and review prompt were committed). No implementation or GPU is authorized by this audit.

## Verdict

**DEAD at Gate 1; no GPU.** The proposal supplies no repository measurement showing that low identity-Fisher rank causes corrected zero-shot retrieval error, and the closest repository rank investigation explicitly found that proxy-span rank was dimensionally guaranteed rather than causal evidence. More decisively, the claimed escape theorem is false for the factorized Fisher actually optimized: differentiating a log-determinant with respect to a missing eigenvalue does not imply a nonzero training gradient that can create that eigenmode. The proposal also forecasts no statistically defensible frontier crossing on either CUB or Cars, makes In-Shop non-frozen, and places its only claimed significant crossing on SOP, outside the standing three-benchmark objective.

The class-marginal Fisher formula and scale-neutral matrix gradient are useful calculations. They do not rescue the method.

## Gate 1: no eligible measured provenance

DOIR begins from a plausible story—seen identities use only a low-dimensional discriminative subspace—but does not name a measurement from this repository that establishes it. The repository's relevant evidence points the other way:

- Candidate 371 showed that the centered span of 75 proxies having rank at most 74 is a dimensional identity, not evidence of channel starvation. It also found that a proxy-rank lever does not isolate representation-gradient rank because normalization and the shared nonlinear backbone spread parameter updates beyond the proxy span.
- Candidate 225's prospective disjoint-identity nuisance-frame test did not find a transferable linear nuisance subspace: corrected ratios were `0.9312`, `0.9287`, and `0.9345`, all below its `1.15` falsifier.
- The CCRB audit found no corrected artifact relating within-class covariance rank to retrieval and showed that filling dimensions does not identify what information fills them.

DOIR introduces a different matrix, but it supplies no saved identity-Fisher spectrum, no longitudinal link between that spectrum and corrected retrieval, and no intervention isolating it. It is therefore an armchair mechanism under Gate 1, not a measurement-derived candidate.

## Formula audit

### What is correct

For proxy posterior `pi_ij`, class marginal `q_ic`, responsibility-weighted class proxy mean `m_ic`, and global proxy mean `mbar_i`,

```
grad_z log q_ic = s (m_ic - mbar_i)
F_i = s^2 sum_c q_ic (m_ic-mbar_i)(m_ic-mbar_i)^T
```

is correct. Tangent projection gives the Fisher of the class likelihood under infinitesimal motion on the normalized sphere. The trace-normalized log-det penalty has its minimum at an isotropic positive spectrum, and the stated matrix derivative is scale-neutral: `tr((dR/dF) F)=0`.

The familiar categorical bound `rank(F_i) <= C-1` is per sample, not necessarily a bound on the batch average. Because the proposal uses multiple proxies per class, `m_ic` and the tangent plane vary with the sample; summing their factors can in principle reach rank 512. It would therefore be wrong to kill DOIR merely because CUB has 100 training identities. With one fixed proxy per class, the global rank would be at most `C-1`; with `K=15`, that simple global bound no longer applies.

### The escape lemma is false in the optimized parameterization

Write the batch Fisher as `F = V V^T`, where columns of `V` are the projected, confusion-weighted score vectors. For a spectral penalty `R(F)`,

```
grad_V R = 2 (grad_F R) V.
```

If a direction `u` is exactly absent, `u^T V=0`. The update above is formed from the existing columns of `V`; the large formal derivative `dR/dlambda_u` at a zero eigenvalue does not itself create a component of `V` along `u`. At an exact zero singular mode, the first-order factor gradient into that mode is zero. This is the standard rank-deficient Gram-barrier trap. Jitter bounds the matrix inverse but does not change the missing factor.

Lemma 2 therefore confuses optimization in the cone of free PSD matrices with optimization through proxy/embedding factors. Its claim that the idle set is not invariant and is escaped at non-vanishing rate is unproved and false in the exact rank-deficient case.

Lemma 1 is also too strong for a deep normalized network. A zero component of `grad_z ell` does not imply exponential decay of the corresponding component of `z`: AdamW decays shared parameters, not embedding coordinates, and updates caused by other samples/directions pass through a shared nonlinear backbone and can rotate every output component. Training proxies also change the score span. The proposed continuous coordinate-wise decay and preservation argument is not a valid consequence of parameter weight decay.

### Singular-confidence failure

The proposal says a confidently classified sample contributes exactly zero. If the whole batch is exactly confident, `F=0` and `d F / tr(F)` is undefined; no trace-denominator floor is specified. Near that state the normalization contributes a `1/tr(F)` factor, so arbitrarily small softmax/rounding residuals determine a scale-neutral direction with potentially very large gradients. FP32 Cholesky jitter repairs `logdet(tilde F + eps I)` but not division by `tr(F)`.

## Degeneracies and controls

1. **Within-class proxy scatter is not blocked.** The statement that redistributions preserving every `m_ic` leave `F_i` unchanged is tautological. The trainable proxies are free to change `m_ic(z)` across samples. With 15 proxies per CUB/Cars class they can manufacture a high-rank class-score Jacobian through within-class responsibility switching. This is exactly the extra degree of freedom that avoids the one-proxy `C-1` rank ceiling, so it cannot simultaneously be dismissed as irrelevant.

2. **The parked-proxy exponent is unsupported.** A single rival class whose logit is lower by `s Delta` has posterior mass of order `exp(-s Delta)`, and its covariance cross-weight against the dominant class is also first order in that mass, not generally `exp(-2 s Delta)`. The claimed `Delta <= 0.072` shell is therefore too tight by roughly a factor of two in the simple dominant-class case.

3. **C8 is exactly a no-op as written.** `F` does not use the ground-truth sample label. Applying a fixed permutation to class names merely relabels the index `c` in `sum_c`; the grouped proxy sets and the resulting matrix are unchanged. It cannot test semantic-free noise and will match DOIR bit-for-bit.

4. **C3's detach description is reversed.** Detaching `q,m` also detaches their proxy dependence. The only remaining path is through the tangent projector `Pi(z)`, hence into the embedding, not “only into proxies.” C4 (detach `P` but retain posterior dependence on `z`) can be an embedding-only path, but C3 is not its proxy-only complement.

5. **C10 does not measure an unseen-identity Fisher.** On test images it remains the Fisher of the *seen training-proxy classifier*. It can be reported post hoc without labels, but a higher effective rank of seen-proxy confusion on OOD images is not evidence that unseen identity decisions gained rank. Using the test statistic for model choice would also violate the training-identity-only selection rule.

6. **Five seed pairs cannot support the registered mediation correlation.** With `n=5`, a threshold such as `r=0.5` is extremely unstable and does not provide a meaningful mediation test. Raw and selection-corrected retrieval values are also absent from the reporting plan despite the repository's measured differential selection bias.

## Gate 2 adjacency

No exact primary-source collision is needed to reject DOIR at Gate 1, and this audit does not claim one without evidence. The surviving action is nevertheless tightly occupied: make a learned representation/proxy geometry use more spectral directions through a determinant or covariance barrier.

- NIR is benchmark-matched proxy DML explicitly motivated by proxy losses losing class-local structure; it preserves non-isotropic sample structure rather than isotropizing a classifier Fisher.
- VCReg and related variance/covariance regularizers explicitly prevent dimensional and gradient starvation for transfer.
- Candidate 156 already rejected a same-class log-determinant volume objective as occupied variance preservation; CCRB rejected a trace-inverse spectral barrier for the same action and its content-selection failure.
- D-optimal design and Fisher-information maximization are established uses of `log det F`; changing the optimized coordinates from parameters/data acquisition to embedding coordinates may be an application-level distinction, but it does not repair provenance or the false escape theorem.

Primary/reference sources:

- Roth, Vinyals, and Akata, *Non-Isotropy Regularization for Proxy-Based Deep Metric Learning*, CVPR 2022: <https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html>
- Garrido et al., *RankMe*, ICML 2023: <https://proceedings.mlr.press/v202/garrido23a.html>
- Jia and Su, *Information-Theoretic Local Minima Characterization and Regularization*, ICML 2020: <https://proceedings.mlr.press/v119/jia20a.html>
- Xu et al., *L_DMI: A Novel Information-theoretic Loss Function for Training Deep Nets Robust to Label Noise*, NeurIPS 2019: <https://proceedings.neurips.cc/paper/2019/hash/8a1ee9f2b7abe6e88d1a479ab6a42c5e-Abstract.html>
- Repository rank audit: `docs/fable_rank_demand_measurement_audit_371_2026-08-04.md`.
- Repository spectral-barrier audit: `docs/fable_ccrb_nearmiss_audit_2026-08-05.md`.

## Forecast and objective mismatch

The frozen proposal itself computes that CUB `0.735` versus PFML `0.734` and Cars `0.930` versus `0.927` are not statistically defensible crossings. In-Shop is explicitly “soft, not frozen.” The sole forecast significant crossing is SOP `0.833` versus `0.829`, while the standing objective requires a benchmarked improvement on CUB, Cars196, or In-Shop. Even a perfectly executed result matching the proposal would not yet fulfill that objective.

## Local disposition

Preserve the class-marginal embedding-coordinate Fisher derivation and the warning that a per-sample `C-1` rank bound does not automatically bound an average with sample-varying multi-proxy means. Reject DOIR as a method. Its causal premise lacks repository provenance, both load-bearing lemmas overclaim deep-network dynamics, the claimed spectral escape fails through a Gram factor, two decisive controls are malformed, and its registered forecast does not cross an in-scope frontier.

## Independent-review reconciliation

The frozen cold review (`docs/opus_doir_review_2026-08-06.md`) agrees with the
DEAD disposition and supplies a stronger, constructive mechanism failure. In a
synthetic instance using the proposal's own dimensions and hyperparameters, it
held every embedding fixed and moved only proxies along an exact base-loss null
space. The barrier fell from `1.1941` to `0.3458`, Fisher effective rank rose
from `154.9` to `355.7`, and the base loss stayed exactly `2.0739198`. Thus the
registered signatures `R down` and Fisher effective-rank up can be produced
without changing the deployed representation or its retrieval result. The
reviewer correctly limits this experiment: it does not prove that real joint
training has zero transient effect, but it does prove that the proposed
signatures and controls cannot attribute such an effect to embedding repair.

The review independently confirmed that the batch-averaged Fisher can be full
rank even though each sample Fisher has rank at most `C-1`; that correction is
retained. It also quantified why Lemma 1 cannot carry the mechanism: with
`lr=1e-4` and AdamW decay `1e-4`, the direct multiplicative decay is only
`1e-8` per step, and the alleged common idle direction is generically absent
when the many sample-dependent multi-proxy score vectors span 512 dimensions.
For Lemma 2, an exactly absent factor direction has zero first-order parameter
gradient, so the PSD-cone derivative is not an escape theorem.

The independent Gate-2 search did **not** find an exact object-level collision:
the class-marginal posterior Fisher in embedding coordinates with a
trace-normalized log-determinant barrier may be a new object. That narrow
novelty does not rescue an unmeasured causal premise or a proxy-only degenerate
path. The nearest established actions remain Fisher/D-optimal design,
MCR-squared, dimensional-collapse/covariance regularization, and NIR. The cold
review also confirms that C8 is an exact label-permutation no-op and that C10
fits seen proxies to test embeddings rather than measuring unseen identity
information. For C3, the proposal's tangent projection means detaching `q` and
`m` can at most leave a projector-only embedding path; it still does not create
the claimed proxy-only control. No GPU work is justified.
