# CRS-PFML authoritative local audit

Date: 2026-08-05.

Verdict: **DEAD at Gates 1 and 2.** No diagnostic, preregistration,
implementation, or GPU run follows.

Exact frozen artifacts are
`docs/sol_crs_pfml_proposal_pass17_2026-08-05.md` and
`docs/sol_crs_pfml_review_2026-08-05.md`. Separate GPT-5.6-Sol shell-fallback
sessions produced and reviewed the proposal after their native jobs failed
before provider receipt. Both answers were extracted from separate Codex JSONL
transcripts and SHA-256 checked before this verdict.

## Gate 1: no verified causal headroom

CRS-PFML assumes corrected zero-shot errors arise because high-margin
descriptor directions encode photograph-specific background, pose, crop,
illumination, or augmentation response rather than identity-reproducible
evidence. It proposes to estimate repeatability from two different training
photographs with the same label.

No verified repository measurement establishes that premise or relates a
reliability spectrum to corrected retrieval errors. The audited IPSR pack shows
controlled augmentation-response relations exist, not that nonrepeatable
directions cause errors or that same-label cross-photo covariance isolates
transferable morphology. Corrected In-Shop training leave-one-out R@1 near
0.995 and stable official-query difficulty do not identify a missing
repeatability mechanism.

The forecasts `+0.008/+0.005/+0.005` on CUB/Cars/SOP are unsupported by a
pilot, spectrum diagnostic, nuisance-stratified error rate, or effect-size
analogue. Gate 1 therefore fails independently of the formal defects below.

## Gate 2: the central estimator is wrong under its sampler

For identity `y` with `n_y` finite photographs, define residuals around the
finite-set mean, with `sum_i e_{y,i}=0`. If the method samples an ordered pair
of distinct photographs without replacement, then

```
E[e_i e_j^T | y, i != j]
  = -1 / (n_y (n_y - 1)) * sum_i e_i e_i^T,
```

not zero. The two photograph residuals are mechanically dependent. Therefore
the cross-covariance equals the desired between-identity component minus a
within-identity photograph covariance term, scaled most strongly when classes
have few images. This directly contradicts the frozen repeated-measurement
derivation.

The loss uses `C C^T`, so it removes correlation sign. A negatively correlated
without-replacement nuisance direction is rewarded exactly like positive
repeatability. The method can therefore amplify what it claims to suppress.
Repair requires a different estimator or objective.

## The random-projection anti-concentration proof is false

Let `A` be any orthonormal `512 x 16` matrix and encode every training identity
as `z(x)=A c_y`, with unit 16-D codes `c_y` in general position. This is a fixed
rank-16 identity lookup representation. For a Haar `512 x 16` probe `R`, the
matrix `H=R^T A` is square and invertible with probability one, because its
determinant is a nonzero polynomial (for example, it equals one at `R=A`).

Hence every countable sequence of random probes maps the same rank-16 identity
code to a full-rank, perfectly repeatable 16-D projected code almost surely.
After whitening, all sixteen reliability directions can approach one (modulo
the ridge). Resampling `R` does not spread information across 512 descriptor
dimensions and does not prevent concentration in a fixed 16-D subspace. The
proposal's load-bearing novelty and degeneracy claim is false.

Other accepted review findings:

- pooled whitening is not ordinary CCA except when branch marginals match;
- `-logdet(CC^T + eps I)` maximizes a ridged product of squared singular
  values, not a weakest-reliability objective;
- the CRS gradient is zero at constant collapse, so PFML alone prevents it;
- `m_C` is a signed EMA with no protection against a small/sign-changing
  denominator;
- the PFML sampler, augmentation, and universal `(delta, alpha)=(0.2,2)` recipe
  are not the published matched frontier recipe.

## Occupied mechanism neighborhood

The exact wrapper may be unreported, but its primitives and scientific target
are crowded:

- [Deep CCA](https://proceedings.mlr.press/v28/andrew13.html), supervised CCA,
  and multiview discriminant analysis learn correlated shared representations
  across paired views.
- [Probabilistic CCA](https://www.di.ens.fr/~fbach/probacca.pdf) already uses a
  shared-latent plus view-noise model.
- [Barlow Twins](https://arxiv.org/abs/2103.03230),
  [VICReg](https://openreview.net/pdf?id=iWpcWZ8phD), and
  [W-MSE](https://proceedings.mlr.press/v139/ermolov21a.html) occupy
  cross-view agreement, whitening, variance preservation, and redundancy
  reduction.
- [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html),
  face recognition, and re-ID use distinct same-label photographs as positive
  evidence.
- [MCR2](https://proceedings.neurips.cc/paper_files/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html),
  Anti-Collapse DML, and [NIR](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html)
  occupy distributed covariance/rank and coding-rate pressure in DML.

Calling two different photos “cross-fitted” does not create statistical
cross-fitting: no nuisance estimator is trained on one fold and evaluated on a
held-out fold.

## Controls and conclusion

The controls do not rescue identification. Same-image CRS halves raw-image
diversity relative to distinct-photo CRS; pair shuffling removes all label
signal; fixed `R` does not test the rank-16 counterexample; and no
cross-acquisition/background/source control separates morphology from stable
identity shortcuts. Distinct-photo VICReg/W-MSE, marginal coding-rate, and an
exact current-recipe PFML baseline are missing.

CRS-PFML is therefore not a Gate-1/2 survivor. Its central covariance premise
is wrong under without-replacement sampling, its squared operator can reward
negative nuisance, and its random-probe rank claim has an exact counterexample.
No GPU is authorized.
