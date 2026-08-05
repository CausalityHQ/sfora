## Verdict: DEAD

The frozen proposal is not a well-defined differentiable algorithm, and its claimed 90% certificate is false after the specified clipping. These are specification-level failures, not empirical uncertainties.

### Decisive reason

The proposal equates two different objects:

\[
(\hat\xi,\hat\beta)=\arg\min \ell(\xi,\beta)
\]

and the output of “eight damped Newton iterations.” Eight iterations are not generally the constrained MLE, while the frozen text supplies no damping rule, line search, projection, barrier, active-set treatment, or infeasible-initialization recovery. Consequently, \(q_i\), its gradient, and therefore the trained model are not uniquely determined.

This cannot be repaired without defining a different proposal.

### 1. Mathematical audit

- The GPD CDF, support condition \(1+\xi e/\beta>0\), likelihood for \(\xi\ne0\), and the *unclipped* maximum-quantile algebra are correct under a fitted iid tail model.
- The optimization objective is undefined at \(\xi=0\), even though \(0\) is inside the feasible interval. Only the CDF and \(q_i\) limits are supplied. The missing likelihood limit is \(k\log\beta+\sum e_j/\beta\).
- The initialization is not guaranteed feasible. At \(\xi=-0.1\), feasibility requires \(\max e<10\beta\), which need not hold for a concentrated set of exceedances.
- The constrained solution may be on the \(\xi\), \(\beta\), or support boundary. Projection and active-set transitions are nonsmooth and unspecified.
- Eight Newton steps need not converge, need not remain feasible, and do not implement the displayed `argmin`.
- The fit is nonunique when all exceedances are zero: \(\beta\) is driven to its lower bound while \(\xi\) is unidentified. Thus the collapse case does not produce a unique \(q_i\).
- The top-64 map is only piecewise differentiable. At rank swaps or ties, the threshold and membership gradient require a tie convention that is not provided. The statement \(s_{ij}\ge u_i\) can also select more than 64 observations under ties while the likelihood still sums exactly \(k=64\).
- The branch at \(|\hat\xi|=10^{-4}\), constraint activation, top-\(k\), `max`, and clipping all contradict the unqualified description “differentiable.”
- “Clip \(q_i\) to \(1-10^{-4}\)” does not formally say upper clamp, lower clamp, or replacement. The intended upper clamp is inferable but is not executable syntax.
- \(E\) in \(L_{\rm PF}=|E|^{-1}\sum_{r\in E}\cdots\) is never defined. It is unclear whether proxies are evaluation points or merely field sources.
- \(z\notin\mathbb S^{511}\) exactly because division uses \(\|Wh\|+\epsilon\), and \(z=0\) is possible.
- At constant collapse, \(\psi_{\rm rep}(0)=0^{-\alpha}\) is infinite/undefined and Euclidean distance has no ordinary gradient at coincidence. The claimed “nonzero descent directions” do not follow from the frozen equations.

The classical POT result is asymptotic tail approximation for suitable distributions and thresholds; it does not confer validity on an arbitrary adaptive queue fit. See the primary results of [Pickands (1975)](https://doi.org/10.1214/aos/1176343003) and [Balkema–de Haan (1974)](https://doi.org/10.1214/aop/1176996548).

#### Certificate failure

Before clipping, solving

\[
\frac{k}{n_i}\bar G(q_i-u_i)=r_M
\]

does give the stated plug-in quantile if negatives are iid from the fitted distribution.

But if the fitted \(q_i>0.9999\), the proposal replaces it by a smaller value. Then

\[
P(\max S_j^-\le q_i^{\rm clipped})<0.90
\]

is entirely possible. The implication advertised as a certificate therefore does not survive the executable clipping operation.

Moreover, 0.90 is a model quantile, not a statistical confidence level. Parameter uncertainty from only 64 tail observations is ignored.

### 2. Monotonicity fails

Lowering an individual negative does **not** always lower \(q_i\) or the loss:

- Lowering any negative outside the top 64 changes nothing.
- When \(q_i\) is upper-clipped, small reductions can leave both target and loss unchanged.
- Within the top 64, lowering one observation refits \(\xi,\beta,u\). There is no sign constraint on the implicit derivatives of the MLE; a reduction can make the estimated tail heavier and increase the extrapolated quantile.
- Lowering the 64th observation changes the threshold and therefore every exceedance simultaneously.

Thus EGR is not an isotonic negative-ranking loss.

### 3. The queue does not identify the deployment maximum

The required equality \(P(\max S_j^-\le s)=F(s)^M\) needs conditional independence and a common relevant distribution. The queue instead contains:

- repeated images and identities;
- class-balanced rather than gallery-frequency sampling;
- descriptors from different historical networks;
- similarities changing because the query and encoder are actively optimized against that queue;
- seen-identity nuisance structure, while deployment identities are disjoint.

FIFO turnover prevents permanent memorization; it does not establish exchangeability, stationarity, tail calibration, or transfer to unseen identities.

There is also a serious SOP execution issue. With 11,318 training identities and a 64-step queue, a without-replacement class-balanced epoch sampler will normally not revisit an identity before its queue entries expire. Then \(\mathcal I\) can be empty and EGR contributes no SOP gradient. Sampling identities with replacement would change this, but the frozen sampler does not specify which behavior is intended.

### 4. Causal mechanism

Rare-impostor collision is assumed, not measured. No proposed diagnostic establishes:

- what fraction of R@1 errors are nuisance-driven isolated impostors;
- whether those errors have a stable GPD tail;
- whether a tail fitted on seen identities calibrates held-out identities;
- whether improvement comes specifically from gallery-size extrapolation.

Computationally, EGR is a flexible symmetric function of the observed top 64 queue scores followed by a positive-ranking softplus. It creates no synthetic unseen negative. Locally it is another top-\(k\) hard-ranking surrogate with data-dependent weights. It is not exactly CVaR or hot LSE, but the frozen evidence cannot distinguish it causally from that occupied family.

### 5. Closest primary prior art

The novelty search is materially incomplete:

- **WEINCE** is the closest conceptual prior: an EVT-motivated correction for top-1 contrastive outcomes with bounded cosine similarity, anchor-wise online tail statistics, no trainable tail parameters, and altered hard-negative gradient allocation. It explicitly discusses finite-batch uncertainty and uses stop-gradient tail fits. [WEINCE, ICML 2026](https://arxiv.org/abs/2606.00262)
- **TriSim** already fits a GPD to high-similarity tail observations and feeds the resulting probabilities into a triplet training loss. Its modality and purpose differ, but it directly anticipates “train-time GPD over similarity extremes affects retrieval ranking.” [TriSim, CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/papers/Zheng_TriSim_Tri-Dimensional_Similarity_Modeling_with_Extreme_Value_Theory_for_False-Negative_CVPR_2026_paper.pdf)
- **OpenMax** fits EVT tails in representation space for open-set inference, while **EVM** builds EVT-based variable-bandwidth recognition regions. They do not implement this deployment lane but establish the embedding-space EVT lineage. [OpenMax](https://openaccess.thecvf.com/content_cvpr_2016/html/Bendale_Towards_Open_Set_CVPR_2016_paper.html), [EVM](https://doi.org/10.1109/TPAMI.2017.2707495)
- **Cross-Batch Memory** already supplies the detached FIFO embedding queue, current-query gradients, and expanded hard-negative pool used here. [XBM, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html)
- Tail-focused training is occupied by CVaR/KL-DRO pAUC optimization. [Zhu et al.](https://arxiv.org/abs/2203.00176)
- Observed-rank optimization is occupied by [Smooth-AP](https://arxiv.org/abs/2007.12163) and [SupRank/ROADMAP](https://arxiv.org/abs/2309.08250).

The exact conjunction “supervised zero-shot DML plus POT extrapolation to a nominal larger gallery” may still be narrower novelty. The broad frozen claim is nevertheless untenable without WEINCE and TriSim.

### 6. Controls are insufficient

The proposed controls are useful but do not isolate the claimed mechanism. Missing decisive controls include:

- queue-size and queue-age sweeps;
- held-out-training-identity tail calibration and maximum-coverage tests;
- empirical top-\(k\) penalties matched per-anchor in loss and rank-gradient profile, not merely mean gradient norm;
- an \(M\) or \(\rho\) sweep testing predicted calibration;
- GPD goodness-of-fit and threshold-stability diagnostics;
- stop-gradient versus implicit-fit-gradient ablation;
- WEINCE-style endpoint correction;
- a sampler audit showing nonempty \(\mathcal I\), especially on SOP.

The shuffled-tail control alters query hardness and may create support-incompatible parameter/threshold combinations, so it is not a clean calibration null.

### 7. Forecasts and matched frontier

The forecast gains are arithmetic assertions, not deductions. No measured tail fit, calibration, causal-error prevalence, gradient behavior, or pilot variance supports \(+0.007,+0.004,+0.005\).

More importantly, this is not the published PFML recipe. PFML reports Adam at \(5\times10^{-4}\), proxies at 100× learning rate, and cross-validated \(\alpha,\delta\); EGR-PFML substitutes AdamW, differential learning rates, weight decay, warm-up, cosine decay, clipping, and stronger augmentation. [PFML primary paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.pdf)

A paired CI from newly rerun PFML/EGR seeds could establish a comparison under the new recipe, but it cannot turn the published PFML numbers into paired evidence. The forecast means numerically exceed the supplied Lane-A references, but they do not quantitatively follow from any premise.

### 8. Legal and protocol audit

On its face, deployment is legal: one ResNet-50, one view, one normalized 512-D descriptor, and ordinary nearest-neighbour retrieval. The queue and EVT machinery are training-only, and no forbidden data source is specified.

Protocol ambiguities remain:

- “audited \(\alpha,\delta\)” versus fallback \((4,0.5)\) defines two methods.
- The identity-sampling replacement policy is missing.
- Validation subdivisions, selection metric, epoch-selection rule, and tuning budget are not frozen.
- Exact test resize/interpolation is unspecified.
- “All hyperparameters frozen across datasets” conflicts literally with dataset-dependent proxy counts and \(M\), even if those differences are intentional.
- The claimed PFML reproduction changes optimizer and augmentation recipe substantially.

Experiments might show that some completed implementation improves retrieval, but that implementation would necessarily supply substantive missing choices and would therefore be a different proposal.
