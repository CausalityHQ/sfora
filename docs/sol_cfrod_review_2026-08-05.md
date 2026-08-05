# Verdict: DEAD

The method is legal in broad deployment form and its main loss is plausibly trainable, but the frozen proposal fails its central causal claim, its memorization proof, its control design, and its quantitative forecast standard.

The single strongest decisive reason is this:

> Identity exclusion establishes only that the teacher did not train on the held-out identity. It does not identify the centered teacher residual as useful zero-shot identity information.

For an excluded identity \(y\),

\[
C_yT_{-y}(X_y)
\]

is merely the variation retained by an ImageNet-initialized network optimized to separate other training identities. It can encode pose, crop, background, illumination, annotation artifacts, or ImageNet features just as readily as factors useful for future identity retrieval. Centering removes the batch class mean; it does not separate causal identity factors from nuisance factors. No estimand, intervention contrast, exclusion restriction, invariance assumption, or measured premise establishes the claimed “precisely that geometry.” The proposal itself lists nuisance preservation as a principal risk, contradicting the stronger causal wording.

Repair would require new controls or assumptions, so it cannot be credited to the frozen proposal.

## 1. Equation and operation audit

### Correct or essentially correct

- If every sampled identity appears at least once, then

  \[
  P=Y(Y^\top Y)^{-1}Y^\top
  \]

  is the projector onto batchwise identity-constant vectors, while \(C=I-P\) is symmetric and idempotent.

- \(CZ\) and \(CT\) are batchwise—not datasetwise—within-identity residuals. Centering preserves pairwise differences between members of the same sampled identity.

- For the unregularized problem

  \[
  \min_{Q^\top Q=I}\|B-AQ\|_F^2,
  \]

  an SVD \(A^\top B=U\Sigma V^\top\) gives \(Q=UV^\top\). Thus the stated Procrustes orientation is correct.

- The EMA direction \(A^\top B\), stopped gradient through the state and \(Q\), the lambda schedule, reflection permission, and cosine/Euclidean test equivalence are coherent.

- The rough FLOP arithmetic is internally correct under the proposal’s approximation:

  \[
  \frac{300(3F)+200(3F)+200F}{200(3F)}=2.833.
  \]

  It remains only an approximation: identity folds need not contain equal numbers of images, PFML’s pair/proxy computations are ignored, and the long-PFML learning-rate schedule is unspecified.

### Incorrect claims

1. **The class-code loss is not exactly 1.**

   At \(B=0\),

   \[
   L_k
   =\frac{\|AQ\|_F^2}{\|A\|_F^2+10^{-6}n_k}
   =\frac{\|A\|_F^2}{\|A\|_F^2+10^{-6}n_k}<1.
   \]

   The four-fold loss is the average of four such ratios, not 1.

2. **The displayed gradient is incomplete.**

   For one fold, holding \(Q\) fixed,

   \[
   \nabla_Z L_k
   =\frac{2C(B-AQ)}{\|A\|_F^2+10^{-6}n_k}.
   \]

   At \(B=0\), this reduces to \(-2AQ/D_k\), since \(CA=A\). The complete CF-ROD gradient also carries the outer \(1/4\). The proposal omits both the denominator’s \(10^{-6}n_k\) and the four-fold factor.

3. **The claimed “not stationary in embedding space” conclusion ignores mandatory normalization.**

   The descriptors lie on the unit sphere. The realizable gradient with respect to pre-normalization features is the tangent projection

   \[
   (I-zz^\top)\nabla_zL.
   \]

   A nonzero Euclidean \(\nabla_ZL\) can therefore produce zero network gradient whenever it is radial. This is not a “pathological Jacobian”; it is the ordinary Jacobian of the specified \(L_2\) normalization.

   The loophole is especially direct for a two-image identity, where centered teacher rows are \(a,-a\). A class-coded student vector parallel to \(aQ\) can have nonzero ambient gradients that normalization annihilates. Thus the proof does not exclude all class-code stationary points in the model actually specified.

4. **The ridge changes the Procrustes problem.**

   SVD is applied to

   \[
   S_k+10^{-4}I,
   \]

   not the EMA cross-covariance \(S_k\). This introduces an identity-coordinate preference. Because teacher and student projection heads have arbitrary coordinate bases, it is not a neutral numerical stabilization and \(Q_k\) is not generally the optimizer of the displayed current-batch residual loss.

5. **The repulsion definition is ill-typed as written.**

   Inside the scalar function \(\phi_{\rm rep}(d)\), the term is written \(d_\epsilon^{-\alpha}\), although \(d_\epsilon\) was defined as a two-argument function. Interpreting it as \(d^{-\alpha}\) is an uncredited repair.

6. **“Class-centered residual” is overstated.**

   The center is computed from only two or four co-sampled, independently augmented images—not from the identity’s complete training-image distribution. The target for an image changes with its batch companions and their crops.

### PFML mismatch

The attractive and repulsive signs have the intended qualitative directions, but this is not a faithful compute-matched reproduction of the audited PFML recipe. The primary PFML paper reports, for ResNet-50, batch size 100, proxy learning rate 0.01, and dataset-dependent BN freezing/warm-up. CF-ROD changes the batch to 128, proxy LR to 0.05, introduces a particular cosine schedule, and leaves BN policy unspecified. [The PFML paper also emphasizes that sample–sample interactions are part of its intra-class feature mechanism.](https://arxiv.org/html/2405.18560v4)

Therefore the table’s assertion that the altered “PFML-200” reproduces \(0.734/0.927/0.829\) has no measured basis. DML comparisons are known to be sensitive to sampler and training-protocol changes; this is precisely the issue examined by [A Metric Learning Reality Check](https://arxiv.org/abs/2003.08505).

## 2. Causal and identifiability audit

The counterfactual language is invalid.

Training \(T_{-y}\) without identity \(y\) gives one observed predictor under one training regime. It does not estimate a causal contrast such as

\[
T_{-y}(x)-T_{+y}(x),
\]

nor does centering identify which latent factors become identity-defining under the train/test identity intervention.

Cross-fitting has a rigorous role when out-of-fold nuisance predictions estimate a defined statistical target. That logic was developed for distillation before this proposal: Dao et al. trained fold-excluded teachers and queried each teacher only on its held-out examples, explicitly calling the method cross-fitted knowledge distillation. Their guarantees depend on a defined Bayes-probability nuisance target; CF-ROD supplies no analogous target for “useful within-identity factors.” [Knowledge Distillation as Semiparametric Inference, ICLR 2021](https://arxiv.org/abs/2104.09732).

The teacher can instead preserve:

- ImageNet appearance features;
- dimensions useful for separating other training identities;
- crop and color-jitter sensitivity;
- background and acquisition conditions;
- viewpoint or pose that helps, harms, or is irrelevant depending on the dataset.

Identity exclusion prevents direct target-class label fitting. It does not distinguish these possibilities.

## 3. Memorization and optimum claims

The algebra only rules out the unconstrained point \(Z=YA\) as a stationary point of the residual loss under additional assumptions. It does not prove the claimed optimum or prevent neural shortcuts.

Specifically:

- The loss value and gradient are misstated.
- Unit-normalization can annihilate an ambient nonzero gradient.
- PFML itself can be locally flat for same-class distances below \(\delta\) and different-class distances beyond \(\delta\).
- A high-capacity network can memorize augmentation-dependent image targets without learning transferable factors.
- \(Q\) is student-dependent through its EMA, so the fixed-\(Q\) calculation is not a characterization of the complete training dynamics.
- “Copying the teacher cannot minimize the objective because PFML enforces labels” does not prove incompatibility: a network can preserve teacher residuals while changing identity centroids, and that is almost exactly what the combined objective requests.

The proposal commendably admits neural memorization remains possible, but that admission means the preceding “direct exclusion” is much narrower than claimed.

## 4. Closest primary prior art

I did not find the exact four-part composition—identity-fold teachers, class-centered output residuals, stopped online orthogonal alignment, single DML student. Narrow combination novelty is therefore plausible. The claimed search and primitive-level novelty are nevertheless incomplete.

Most important omitted or underweighted precedents:

- **Cross-fitted KD already exists.** Dao et al. train teachers excluding each fold and use their held-out predictions as student targets, including image experiments. This is substantially closer than the proposal’s econometric-only citation. [Dao et al., ICLR 2021](https://arxiv.org/abs/2104.09732).

- **Intra-class variation separated from class centers is established DML motivation.** DVML explicitly disentangles class centers from intra-class variance to improve unseen-class generalization on CUB, Cars, and SOP. [Deep Variational Metric Learning, ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Xudong_Lin_Deep_Variational_Metric_ECCV_2018_paper.html).

- **Metric embedding distillation is established.** Teacher-to-student image-embedding transfer predates CF-ROD. [Learning Metrics from Teachers, CVPR 2019](https://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_Learning_Metrics_From_Teachers_Compact_Networks_for_Image_Embedding_CVPR_2019_paper.pdf).

- **Relational geometry transfer is occupied by RKD**, which transfers pairwise distances and triplet angles rather than logits. [RKD, CVPR 2019](https://arxiv.org/abs/1904.05068).

- **DML already targets retained intra-class relational structure.** [Deep Relational Metric Learning, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Zheng_Deep_Relational_Metric_Learning_ICCV_2021_paper.html).

- **Disjoint-label episodic meta-DML is established.** ALA explicitly samples disjoint-label train/validation subsets to optimize unseen-identity generalization. [ALA, CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Zheng_Deep_Metric_Learning_via_Adaptive_Learnable_Assessment_CVPR_2020_paper.html).

- **Feature-only face distillation without student identity supervision exists.** [Rethinking Feature-Based Knowledge Distillation for Face Recognition, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Li_Rethinking_Feature-Based_Knowledge_Distillation_for_Face_Recognition_CVPR_2023_paper.html).

- **Face KD transfers intra-class similarity distributions.** [ICD-Face, ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Yu_ICD-Face_Intra-class_Compactness_Distillation_for_Face_Recognition_ICCV_2023_paper.html).

- **Unlabeled-face teacher/student representation learning is occupied.** [ProS, WACV 2024](https://openaccess.thecvf.com/content/WACV2024/html/Di_ProS_Facial_Omni-Representation_Learning_via_Prototype-Based_Self-Distillation_WACV_2024_paper.html).

- **Centered/Procrustes geometry-aware distillation is directly occupied by 2025 work**, which proposes Procrustes distance and centered representational comparisons as distillation losses. [Knowledge Distillation Through Geometry-Aware Representational Alignment](https://arxiv.org/abs/2509.25253).

- Meta-domain generalization and re-ID are adjacent rather than duplicative: [MLDG](https://arxiv.org/abs/1710.03463), [Meta Distribution Alignment for Re-ID](https://openaccess.thecvf.com/content/CVPR2022/html/Ni_Meta_Distribution_Alignment_for_Generalizable_Person_Re-Identification_CVPR_2022_paper.html), and [multi-teacher similarity distillation for re-ID](https://openaccess.thecvf.com/content_CVPR_2019/html/Wu_Distilled_Person_Re-Identification_Towards_a_More_Scalable_System_CVPR_2019_paper.html).

Thus novelty, if any, is the narrow assembly—not cross-fitting, held-out predictions as targets, residual/center separation, relational DML, feature KD, or Procrustes distillation individually.

## 5. Control audit

The controls do not isolate the stated mechanism.

1. **Long PFML** tests extra optimization compute, but not ImageNet feature preservation, teacher ensembling, or recipe changes. Its “approximately 567 epochs” and corresponding cosine schedule are unspecified.

2. **Leaky teacher** confounds identity exposure with exact-image exposure. A teacher trained on random images from every identity may have trained on the very image it later labels. The missing control is image-held-out but identity-seen teachers.

3. **Uncentered OOF** isolates removal of batch identity means reasonably well, but changes the target’s scale and between-class geometry simultaneously.

4. **OOF-RKD** does not isolate coordinates versus relations if RKD is applied to uncentered descriptors. It simultaneously changes centering, target type, normalization, and pair/triplet weighting. Its loss weights and pair/angle sampling are also unspecified.

Essential missing controls include:

- ImageNet-checkpoint-only residual distillation;
- a normal full-data or ensemble teacher;
- image-held-out/identity-seen teachers;
- centered-residual RKD;
- random or untrained teacher residuals;
- exact PFML-200 using the audited PFML recipe;
- the changed PFML recipe without CF-ROD;
- \(Q=I\), batch Procrustes, and EMA-Procrustes variants.

Without the ImageNet-only and image-held-out controls, the experiment cannot distinguish identity exclusion from ordinary pretrained-feature preservation or mere removal of exact-image leakage.

## 6. Forecast audit

The normal-approximation arithmetic is numerically correct under its assumptions:

- CUB SE: \(0.00224\)
- Cars SE: \(0.00190\)
- SOP SE: \(0.00126\)

But the assumptions and conclusions are not justified.

- These are forecasts without pilot measurements, residual-utility measurements, learning curves, effect-size analogues, or control results.
- With only five runs per method, a \(t\)- rather than known-variance normal interval is appropriate. For Cars, the difference interval is approximately

  \[
  0.004\pm2.306(0.00190)\approx[-0.0004,0.0084],
  \]

  so the forecast does not establish frontier crossing at 95%.
- Historical PFML variance does not capture systematic uncertainty from changing batch size, optimizer details, BN policy, augmentation, and proxy LR.
- The proposal alternates between independent-run inference in the table and “paired PFML-200” falsification. The pairing and paired estimator are not defined.
- “Absolute CF-ROD interval overlaps the frontier” improperly treats the frontier as a fixed constant despite giving it uncertainty. The difference distribution must include both uncertainties.
- The forecast that 567-epoch PFML slightly degrades on CUB and Cars is unsupported.
- CF-ROD is 2.83× the audited frontier’s training cost. It is only compute-matched to the hypothetical long-PFML control, whose outcome is itself guessed.

No quantitative premise supports \(+0.007,+0.004,+0.004\).

## 7. Protocol, contamination, and executability

No direct test-image or test-label use is specified, and deployment satisfies the one-model, one-view, 512-D requirement. The broad data regime is therefore legal.

However, the frozen executable specification leaves material operations undefined:

- initialization of the new 512-D projection \(W\);
- proxy initialization and whether weight decay applies to proxies;
- Stage-I sampler, batch size, augmentation, BN policy, warm-up, and checkpoint rule;
- whether frozen teachers are put in evaluation mode during Stage II;
- if not, teacher BN running statistics would update on excluded identities, contradicting strict identity exclusion;
- exact random-fold algorithm and treatment of unequal image counts;
- replacement/drop-last behavior;
- augmentation parameters beyond “standard” color jitter;
- test resize dimension before center crop;
- EMA update-before-loss versus update-after-loss convention;
- numerical precision/device of the \(512^2\) EMA and SVD;
- the construction and mapping of the four leaky teachers;
- the complete OOF-RKD formula and weights;
- the 567-epoch learning-rate schedule;
- the definition of “differs strongly in pose/view.”

That last diagnostic may itself violate the “identity labels only” condition if it uses CUB attributes, pose labels, human annotation, or an external pretrained estimator. No legal identity-label-only definition is supplied.

Finally, hashing benchmark manifests does not detect overlap with ImageNet-1K. The proposal acknowledges this, and ImageNet initialization is expressly permitted, so it is not disqualifying—but it prevents treating the held-out teacher residual as uncontaminated evidence derived solely from other benchmark identities.

## Bottom line

CF-ROD is a plausible experimental heuristic, but the frozen proposal presents it as more than the evidence permits. Identity exclusion does not causally identify useful residual factors; the degeneracy proof is false for the normalized model; the closest cross-fitted-KD prior is omitted; controls leave the main confounds unresolved; the PFML anchor uses a changed recipe; and the forecast is invented rather than derived.

Those are substantive, not editorial, defects. Fixing them would create a different proposal.
