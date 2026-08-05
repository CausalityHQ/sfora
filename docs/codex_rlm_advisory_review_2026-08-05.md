# Verdict: DEAD

The proposal’s claimed technical heart—the “provably” monotone, anti-gaming return-level estimator—is mathematically false for the frozen computation. Fixing it would require changing the estimator or gradient path, which is a different proposal.

## 1. EVT/PWM and proof audit

The basic statistical formulas are mostly correct:

- Translating Hosking–Wallis’s sign convention to the standard GPD shape `xi`, `xi_hat = 2 - lambda_1/lambda_2` and `sigma_hat = (1-xi_hat)lambda_1` is correct.
- The POT return level `u + (sigma/xi){(N pi)^xi - 1}` and its `xi -> 0` limit `u + sigma log(N pi)` are correct.
- For an actual `1-1/N` quantile and `N` independent draws, the maximum exceeds it with probability `1-(1-1/N)^N -> 1-e^-1`. For `K' ~= K-1` and `N=c(K-1)`, the probability is approximately `1-e^(-1/c)=0.221` at `c=4`, not exactly `1/c=0.25`.

These facts do not rescue the method.

### Fatal monotonicity counterexample

Let

```
q = (1-xi_hat)A
s_bar_T = mean_{top k}(s_c).
```

The forward target is

```
m_hat = u + q(s_bar_T-u) = q s_bar_T + (1-q)u.
```

The `(k+1)`-st largest similarity determines `u` but is not a tail member. Increasing that similarity without changing membership changes the actual forward target by

```
Delta m_hat / Delta u = 1-q.
```

Under every frozen setting, `q>1`. For example:

- CUB/Cars with `xi=0`: `A=log(4*99*0.1)=3.68`, so the slope is `-2.68`.
- SOP with `xi=0`: `A=log(4*11317*0.1)=8.42`, so the slope is `-7.42`.

Thus raising the threshold impostor similarity can sharply *lower* the return-level target. Stop-gradient makes autodiff report zero for that path; it does not make the forward function coordinatewise non-decreasing. The proof illegitimately substitutes a surrogate backward derivative for forward monotonicity.

Two corollaries also fail:

- Nonnegative coordinate pseudogradients do not imply that no parameter-space descent direction can increase an impostor; all similarities are coupled through embeddings and proxies.
- Because `t=min(max(m_hat,s_max),0.9999)`, if `s_max>0.9999`, then `t<s_max`. Therefore the unconditional claim `t>=s_max`, and hence the claimed realized-max upper bound, is false.

The cited PWM source also warns that large samples may be needed before asymptotic approximations are useful. CUB/Cars supply only about ten exceedances per anchor, below even the smallest `n=15` simulation considered there. The asserted `O((pi K)^-1/2)` error is not justified for adaptively generated, dependent proxy scores. [Hosking & Wallis, 1987](https://www.stat.cmu.edu/technometrics/80-89/VOL-29-03/v2903339.pdf)

No optimum is proved. Assumption A3 simply assumes the desired end-of-training margins; it is not derived from the joint CE/RLM optimization.

## 2. Wrong-population identification

A GPD fit to seen-class proxies does not identify the deployment impostor tail:

- Every proxy has been repeatedly optimized against the same training embeddings.
- Deployment candidates are images of unseen identities, not sub-center proxies.
- Taking a maximum over `M` learned proxies changes the score law again.
- Proxy scores share the backbone and proxy matrix and are neither independent nor fresh class draws.

The image twin does not repair this. Its negatives are still embeddings of actively optimized, seen identities. With batch 128, it fits only roughly 31–32 image exceedances per anchor regardless of SOP/In-Shop’s thousands of classes. Extrapolating those observations to `c|X_train|` does not turn them into unseen-identity observations.

This is especially consequential because class-disjoint DML test distributions can exhibit unspecified and split-dependent shifts; that is an empirically documented issue, not something implied away by “exchangeability.” [Milbich et al., NeurIPS 2021](https://papers.nips.cc/paper/2021/file/d1f255a373a3cef72e03aa9d980c7eca-Paper.pdf)

## 3. Causal claim and occupied surrogate

“Fresh-impostor max-margin miscalibration” is assumed, not measured. There is no diagnostic comparing optimized seen-class maxima, held-out-class maxima, proxy versus image tails, their evolution during training, or predicted versus empirical return levels.

Dwork et al. study adaptive reuse of data for statistical analyses and holdout validation. They do not prove that SGD suppression of class proxies produces the particular downward bias claimed here. [Dwork et al., 2015](https://papers.nips.cc/paper_files/paper/2015/hash/bad5f33780c42f2588878a9d07405083-Abstract.html)

When the EVT branch is active, its negative-score pseudogradient is simply `common scalar / k` on each selected top-k negative. That is uniform top-tail/CVaR-like hard-negative pressure. When the guard is active, it is ordinary hardest-negative pressure. Consequently, the frozen optimizer alternates between two already occupied ranking mechanisms; the extrapolated forward scalar mainly controls activation and hardness. Primary neighboring work includes hard-negative tilting, top-tail/CVaR pAUC optimization, proxy aggregation, and large-negative memories. [Robinson et al., ICLR 2021](https://openreview.net/pdf?id=CR1XOQ0UTh-), [Zhu et al., ICML 2022](https://proceedings.mlr.press/v162/zhu22g.html), [Proxy-Anchor](https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html), [XBM](https://openaccess.thecvf.com/content_CVPR_2020/html/Wang_Cross-Batch_Memory_for_Embedding_Learning_CVPR_2020_paper.html).

## 4. Novelty assessment

The exact narrow combination—end-to-end descriptor learning with a stopped-gradient POT return-level target—was not located in the primary literature checked. That narrow novelty is therefore **unresolved**, not established.

The proposal’s novelty survey contains material errors:

- EVM is not merely post-hoc calibration. Its primary paper describes an EVT-derived classifier learned from training data and explicitly distinguishes it from post-hoc approaches. It does not end-to-end train the backbone, so it is close prior rather than exact anticipation. [Rudd et al., EVM](https://www.wjscheirer.com/papers/wjs_tpami2017_evm.pdf)
- PFML means **Potential Field Based Metric Learning**, not “probabilistic multi-proxy class representation.” [PFML, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html)
- TCM is closer than portrayed: it is explicitly a train-time open-world margin regularizer that selectively penalizes hard pairs, although it does not use return levels. [TCM, ICLR 2024](https://proceedings.iclr.cc/paper_files/paper/2024/hash/16336d94a5ffca8de019087ab7fe403f-Abstract-Conference.html)

Novelty cannot compensate for an invalid frozen mechanism.

## 5. Controls are not decisive

C1 correctly controls the incremental RLM effect under the same multi-proxy base, and C2/C3 are useful. But essential controls are missing:

- fixed-coefficient top-k mean/CVaR hinge;
- swept coefficient `q` with no EVT fit;
- threshold-plus-top-tail contrast with fixed `q`;
- empirical held-out-identity return-level calibration;
- single-proxy versus `M=2/4` capacity;
- image-level realized-max control with exactly the same weighting as the twin.

C2 is ambiguous about whether it replaces both proxy and image EVT terms. C7 shows whether the image term helps performance; it cannot show that the term repairs the population shift.

## 6. Forecasts and lane arithmetic

The reported PFML Lane-A numbers agree with its primary paper, and DADA is a genuine AAAI 2024 proxy-domain adaptation method. [PFML](https://openaccess.thecvf.com/content/CVPR2025/papers/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.pdf), [DADA](https://ojs.aaai.org/index.php/AAAI/article/view/29400)

However:

- The R@1 forecasts are unsupported guesses; no empirical or statistical bridge connects the 22% breach heuristic to the proposed R@1 gains.
- Cars is internally impossible as stated: if `[0.910,0.926]` contains 90% probability, then `P(X>0.927)<=0.10`, not `0.14`.
- The SOP arithmetic `0.829+2sqrt(0.002^2+0.003^2)=0.8362` is numerically correct, but it combines seed standard deviations, not standard errors of five-run means. It is therefore not a conventional significance threshold.
- The medians cross only the supplied mean references on SOP and In-Shop; neither crosses the proposal’s own 2-sigma heuristic.
- In-Shop has no reference uncertainty, so a statistically calibrated frontier-crossing probability cannot be obtained from the supplied evidence.

## 7. Hidden risks

Deployment is legally Lane A: one ResNet-50, one view, one 512-D descriptor, ordinary cosine retrieval. The class-disjoint validation plan also uses only official training data.

Remaining risks are substantial:

- Eighteen hyperparameter combinations per dataset, plus epoch selection, controls, and five seeds create large unreported tuning and selection compute.
- A single 20% class fold is particularly noisy on CUB/Cars.
- “Zero extra memory” is literally false: the image similarity matrix, sorting/top-k state, and workspaces require additional training memory.
- The <2% overhead claim is unbenchmarked.
- `N_img=c|X_train|` gives dataset-dependent implied breach rates because test galleries do not share the same train/gallery size ratio.
- SOP label noise directly contaminates the extreme tail; giving a false negative weight `1/k` does not make the fitted tail or return level robust.
- The frozen fixed class splits are known to conceal variable unseen-class distribution shifts.

**Single decisive reason:** the stop-gradient construction does not make the forward return-level target monotone; for the frozen values, increasing the threshold impostor lowers the claimed conservative target. That directly invalidates the proposal’s central proof and anti-gaming justification.
