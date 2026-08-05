# Verdict: DEAD

The proposal’s central causal claim is mathematically false under its own sampler and loss. Distinct photographs sampled without replacement do not have conditionally independent nuisance, and the squared canonical operator can actively reward the resulting nuisance covariance. Separately, the claimed random-projection anti-concentration guarantee is defeated by a fixed rank-16 identity code.

These are failures of the frozen method, not missing exposition. Repairing them would require a different estimator or objective.

## Decisive failure: the nuisance covariance is not zero

Let identity \(y\) have \(n_y\) photographs and define photograph residuals around its finite-set mean:

\[
e_{y,i}=v_{y,i}-s_y,\qquad \sum_{i=1}^{n_y}e_{y,i}=0.
\]

The proposal samples an ordered distinct pair \(i\ne j\) without replacement. Therefore

\[
\mathbb E[e_{y,i}e_{y,j}^{\top}\mid y,i\ne j]
=
-\frac{1}{n_y(n_y-1)}
\sum_i e_{y,i}e_{y,i}^{\top},
\]

not zero. Conditional independence is mechanically violated because observing the first photograph changes the distribution of the second.

Consequently,

\[
\operatorname{Cov}(v^{(1)},v^{(2)})
=
\operatorname{Cov}(s_y)
-
\mathbb E_y\!\left[
\frac{\Sigma_{\text{photo},y}}{n_y-1}
\right],
\]

up to covariance convention and augmentation terms. This effect is strongest for identities with few images—the proposal’s stated SOP risk.

Worse, \(CC^\top\) discards correlation sign. A negatively repeatable photograph residual produced by without-replacement sampling contributes exactly like positive reliability. Thus the objective can reward the nuisance it claims to remove.

The repeated-measure interpretation also omits the need to specify which reliability model is intended: consistency, absolute agreement, fixed acquisition effects, or random acquisition effects can yield different estimands, as classical ICC work emphasizes. [Shrout and Fleiss](https://pubmed.ncbi.nlm.nih.gov/18839484/) formalized these distinctions decades ago.

## The requested rank-16 counterexample exists

Choose an orthonormal matrix \(A\in\mathbb R^{512\times16}\). Give every training identity a unit code \(c_y\in\mathbb R^{16}\), chosen in affine general position, and let every photograph of that identity map to

\[
z(x)=Ac_y.
\]

This is a normalized, fixed rank-16 training-identity lookup code. With generic codes, every sampled set of 32 identities has centered covariance of rank 16.

For a fresh Haar probe \(R\), define

\[
H_R=R^\top A\in\mathbb R^{16\times16}.
\]

The polynomial \(\det(R^\top A)\) is not identically zero—it equals one when \(R=A\)—so

\[
\Pr[\det(H_R)=0]=0.
\]

Therefore, with probability one over every countable sequence of training iterations:

\[
v_y^{(1)}=v_y^{(2)}=H_Rc_y,
\]

and the projected batch remains full rank and perfectly repeatable on every iteration. With ridge regularization, its reliability eigenvalues are merely changed from \(1\) to \(\sigma_i/(\sigma_i+\varepsilon)\).

Random resampling may softly prefer better-conditioned, more distributed representations, but it does not prevent concentration in a fixed 16-dimensional subspace. No determinant factor is pinned to \(10^{-3}\) for this construction. The frozen anti-concentration statement is therefore false.

## Equation and operation audit

| Operation | Finding |
|---|---|
| Descriptor/proxy normalization | Valid except that vectors below \(10^{-12}\) are not actually unit normalized. |
| PFML potentials | The piecewise potentials and total-energy structure substantially match published PFML. At \(d=\delta\), either branch agrees. [PFML primary paper](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html) |
| PFML collapse claim | At exact cross-class collapse, \(d^{-\alpha}\) is infinite/undefined, not merely “unsatisfied.” |
| PFML matching | The paper selects \(\delta\) and \(\alpha\) by cross-validation; the frozen universal \((0.2,2)\), sampler, and much stronger augmentation recipe are not demonstrated to reproduce the cited PFML frontier. |
| Sampling | Distinct-without-replacement contradicts conditional independence, as derived above. |
| \(S_{11},S_{22}\) | Ordinary ridge sample covariances; correct. |
| \(S_{12}\) | A symmetrized cross-covariance, not the ordinary empirical cross-covariance. It deletes antisymmetric cross-view structure. |
| Pooled \(S\) | Valid positive-definite pooled covariance, but not ordinary two-view CCA whitening. |
| “Canonical spectrum” | Standard CCA uses \(S_{11}^{-1/2}S_{12}S_{22}^{-1/2}\) and its singular values. Pooled whitening agrees only under equal branch marginals, approximately in population—not in each 32-identity batch. [Deep CCA](https://proceedings.mlr.press/v28/andrew13.html) |
| Latent-noise interpretation | This is close to probabilistic CCA’s shared-latent/view-noise model, which the novelty search omitted. [Bach and Jordan](https://www.di.ens.fr/~fbach/probacca.pdf) |
| Log determinant | \(-\frac1{16}\sum_i\log(\sigma_i(C)^2+10^{-3})\) maximizes a ridged product of squared singular values, not the product of canonical repeatabilities. It is not a weakest-eigenvalue objective. |
| Correlation sign | \(CC^\top\) rewards \(C=-I\) just as much as \(C=I\); anti-repeatability is accepted. |
| Constant collapse | The stated loss value \(-\log 10^{-3}\) is correct, but the CRS gradient is exactly zero at \(C=0\). Avoidance then relies wholly on PFML. |
| Rank \(r<16\) | The rank bound and \(16-r\) ridge factors are correct. Rank exactly 16 defeats the claimed broader protection. |
| Random \(R\) | Gaussian thin QR produces a generic Haar/Stiefel probe under the usual QR sign convention. Rotational invariance holds only in distribution/expectation, not for an individual sampled loss. |
| Eigendecomposition gradient | Defined generically, but eigenvector derivatives are ill-conditioned or undefined at repeated eigenvalues. Clamping also creates zero-gradient regions and a kink at \(10^{-3}\). |
| Ridge/clamp | Since \(S\) already contains \(10^{-3}I\), the same eigenvalue clamp is theoretically redundant except for numerical error. |
| EMA scaling | \(m_U\) uses \(|U|\), while \(m_C\) uses signed \(L_{\rm CRS}\). The latter can be slightly negative because factors can reach \(1.001\). No safeguard handles a small or sign-changing \(m_C\). |
| EMA semantics | Normalization makes the effective gradient weight depend on the entire training trajectory. The timing of EMA update versus loss evaluation is unspecified. |
| Schedule | Algebraically continuous at epochs 10 and 30. Whether \(t\) is zero- or one-indexed is unspecified. |
| Cost | The CRS FLOP estimate relative to a ResNet forward is plausible but unmeasured. “Two eigendecompositions” is inconsistent with the one explicitly specified; the log determinant’s algorithm is undefined. Wall-time and memory claims need measurement. |
| Deployment | One model, one view, one normalized 512-D descriptor and cosine retrieval are Lane-A legal. |
| Forecast differences | \(+0.008,+0.005,+0.005\) are arithmetically correct. |
| Forecast tests | The thresholds use \(2\) instead of a small-sample \(t\) critical value and assume independent means. Welch-style 95% thresholds are approximately \(0.7392,0.9314,0.8319\), so forecasts still cross them, but Cars clears by only about \(0.0006\). There is no multiplicity correction or paired-seed covariance model. |
| Evidence | Forecasts are not measurements. No verified spectrum, nuisance-stratified reliability result, or retrieval experiment supports the causal error mode. |

## Causal interpretation does not follow

Repeatability identifies anything stable within a training label:

- training-label lookup;
- background, photographer, seller, web-source, or acquisition-series signature;
- pose conventions;
- catalog layout;
- the physical product instance on SOP;
- genuinely transferable morphology.

The objective contains no variable or intervention that distinguishes these causes.

Moreover, CUB labels are bird species, not individual birds; the official dataset describes 200 categories and warns that some images overlap ImageNet. [Official CUB page](https://www.vision.caltech.edu/datasets/cub_200_2011/) Cars labels are make/model/year categories, while SOP labels are eBay product instances. These benchmarks therefore instantiate substantially different meanings of “identity.” A common repeated-measure causal model is unsupported.

Face recognition and person re-identification have long trained on different photographs of the same identity while confronting pose, camera, illumination, and background confounding. See [FaceNet](https://openaccess.thecvf.com/content_cvpr_2015/html/Schroff_FaceNet_A_Unified_2015_CVPR_paper.html) and camera-aware re-ID consistency work such as [Wu et al.](https://openaccess.thecvf.com/content_ICCV_2019/html/Wu_Unsupervised_Person_Re-Identification_by_Camera-Aware_Similarity_Consistency_Learning_ICCV_2019_paper.html). Distinct-photo agreement alone is not evidence that transferable morphology has been isolated.

## Prior-art audit

The novelty search is materially incomplete.

- CCA, supervised CCA, multiview discriminant analysis, and Deep CCA already learn correlated or shared representations across paired views. Supervised variants explicitly use within-class/inter-view correlation; see [I2SCA](https://ojs.aaai.org/index.php/AAAI/article/view/8986), [multiview discriminant analysis](https://vipl.ict.ac.cn/resources/codes/2016/202205/P020220601507044760660.pdf), and [Deep CCA](https://proceedings.mlr.press/v28/andrew13.html).

- The shared-signal plus independent-view-noise model is standard probabilistic CCA, not a new estimand. [Bach and Jordan](https://www.di.ens.fr/~fbach/probacca.pdf)

- Cross-view agreement plus redundancy control is the core family containing [Barlow Twins](https://icml.cc/virtual/2021/spotlight/10300), [VICReg](https://openreview.net/pdf?id=iWpcWZ8phD), and [W-MSE](https://proceedings.mlr.press/v139/ermolov21a.html). Replacing same-image pairs with labeled same-class pairs is a supervised/multiview adaptation, not a categorical departure.

- Distinct-instance same-label compactness is standard in [Supervised Contrastive Learning](https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html), face recognition, and re-ID.

- Distributed covariance/rank and log-determinant-style objectives already appear in [MCR\(^2\)](https://proceedings.neurips.cc/paper_files/paper/2020/hash/6ad4174eba19ecb5fed17411a34ff5e6-Abstract.html), coding-rate Anti-Collapse DML, and covariance/non-isotropy DML. [Anti-Collapse DML](https://arxiv.org/abs/2407.03106), [NIR](https://openaccess.thecvf.com/content/CVPR2022/html/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.html)

- Random-subspace learning substantially predates this proposal, from [Ho’s random-subspace method](https://doi.org/10.1109/34.709601) through per-step redrawn neural optimization subspaces. [Gressmann et al.](https://arxiv.org/abs/2011.04720)

- Augmentation, sampling, and proxy-distribution changes are established major DML factors; see [PADS](https://openaccess.thecvf.com/content_CVPR_2020/html/Roth_PADS_Policy-Adapted_Sampling_for_Visual_Similarity_Learning_CVPR_2020_paper.html), [DADA](https://ojs.aaai.org/index.php/AAAI/article/view/29400), and the controlled training study by [Roth et al.](https://proceedings.mlr.press/v119/roth20a).

The exact combination may still be unreported, but the frozen proposal overstates conceptual separation from supervised CCA/redundancy-reduction and repeated-measure reliability.

## Controls do not isolate the mechanism

Several controls are confounded or underdefined:

- Same-image CRS uses only 32 distinct raw photographs duplicated into two augmentations, whereas distinct-photo CRS uses 64. It changes raw-image diversity and batch-normalization statistics.

- Positive MSE tests pairwise compactness but not supervised redundancy reduction plus variance preservation.

- The “Barlow control” lacks its exact normalization, off-diagonal coefficient, loss equation, ridge, and whether it uses a projector. It is not executable as frozen.

- No distinct-photo VICReg or W-MSE control is included.

- No marginal log-determinant/coding-rate control isolates ordinary distributed-rank regularization from cross-photo reliability.

- No random-projection loss independent of pair matching isolates stochastic-subspace regularization.

- Pair shuffling removes all supervised shared signal, not just distinct-photo reliability; an 80% gain-removal threshold has no causal identification interpretation.

- The fixed-\(R\) control tests resampling but does not test the rank-16 counterexample.

- No background masking, source/camera stratification, pose stratification, cross-acquisition pairing, or training-identity-probe test distinguishes morphology from stable shortcuts.

- No held-out-training-identity measurement verifies that CRS improves a reliability spectrum before retrieval is inspected.

The broader concern is well established: DML results are highly sensitive to recipe and mini-batch sampling, so a paper baseline is not “matched” merely by retaining its backbone and descriptor dimension. [Roth et al.](https://proceedings.mlr.press/v119/roth20a), [Musgrave et al.](https://mlanthology.org/eccv/2020/musgrave2020eccv-metric/)

## Protocol and remaining underdefinition

The declared deployment is legal, and the stated prohibition on test-set tuning is appropriate. No explicit external training data beyond permitted ImageNet initialization are proposed.

Nevertheless:

- CUB’s official maintainers explicitly warn of CUB–ImageNet image overlap. ImageNet initialization is permitted by the rubric, but this remains real benchmark contamination and must be reported, not cured by recording a checkpoint hash. [Official warning](https://www.vision.caltech.edu/datasets/cub_200_2011/)

- The identity-disjoint validation folds, fold-selection rule, number of tuning trials, and final retraining seed policy are unspecified.

- “Hyperparameters remain fixed” conflicts with the later statement that hyperparameter selection must use validation folds.

- The exact PFML matched recipe is not fixed: sampler, batch size, augmentations, \(\delta/\alpha\) selection, and baseline reproduction are unresolved.

- Adam betas, Adam versus decoupled weight decay semantics, image normalization, proxy initialization, epoch length, EMA update order, QR sign handling, precision, log-determinant implementation, and behavior on nonfinite PFML distances are unspecified.

- The method is called “cross-fitted,” but no estimator is trained on one fold and evaluated on a held-out fold in the statistical cross-fitting sense.

In short, the forecasts cross the supplied point frontier, but there is no verified result behind them, and the proposed experiment cannot validate the claimed causal mechanism with the frozen controls. The exact without-replacement covariance error and rank-16 counterexample are already sufficient for **DEAD**.
