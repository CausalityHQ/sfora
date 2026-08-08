# Pass 181 — Class-Influence Entropy Backpropagation (CIEB)

## Frozen blind proposal

For normalized descriptor coordinates `j`, estimate a class-ownership score
`a_cj=((mu_cj-mu_j)^2)/(sigma_cj^2+eps)`, normalize it over classes to `q_cj`,
and compute class-influence entropy
`h_j=-sum_c q_cj log(q_cj)/log(C)`. Multiply the Proxy Anchor gradient for
coordinate `j` by a normalized `(h_j+eps)^alpha`, while leaving the forward
descriptor and deployment unchanged. The premise is that low-entropy
coordinates are owned by individual training classes and hurt unseen-class
transfer; a CPU falsifier would correlate entropy with held-out retrieval and
compare low-entropy versus random coordinate ablations.

## Gate 2 audit

Gate 2 is **DEAD**. Jin et al., *A Weighting Method for Feature Dimension by
Semisupervised Learning With Entropy* (IEEE TNNLS 2023), explicitly derives
entropy-based feature-dimension weights using whole- and within-class entropy
for classification and dimensionality reduction. Kpotufe et al., *Gradients
weights improve regression and classification* (JMLR 2016), establishes
feature-wise gradient weighting from estimated coordinate variation. Gradient
reweighting and gradient-centralization methods occupy the train-time
preconditioning route. CIEB changes only the estimator of coordinate weights
from entropy/variation to class ownership; the supervised object and the
gradient action are the same. No CPU or GPU work is authorized.

## Repaired Gate-2 re-audit and Gate-1 preregistration (2026-08-08)

**Revised status: LIVE-NARROW at Gate 2; Gate 1 unresolved.** The cited entropy
method weights the forward metric, and Gradient Weighting reweights inputs for a
second-pass nonparametric predictor. Neither uses label-derived ownership entropy as
a backward-only preconditioner of learned DML coordinates. That difference must beat
forward-weighting and random-coordinate controls, but it is not an exact prior-art
collision under the repaired rule.

The operator's previously omitted constants are frozen before any value is inspected:
`alpha=1`, `epsilon=1e-6`, and
`w_j=(h_j+epsilon)/mean_k(h_k+epsilon)`, with no clipping.

### Exact Stage-A necessary-condition screen

The earlier prose was not executable: it left the ownership estimator, fold
aggregation, split-half choice, bottom-decile rounding, matched-mask construction,
standardizer, bootstrap unit, and four-seed CV rule to the analyst.  The following
specification supersedes it before any CIEB statistic is inspected.

For every seed, use float64 and fail on any nonfinite value.  For an estimator class
set `E`, define class-balanced quantities

`mu_cj = mean_{i:y_i=c}(z_ij)`, `mu_j = mean_{c in E}(mu_cj)`, and
`sigma2_cj = mean_{i:y_i=c}((z_ij-mu_cj)^2)`, using population denominator `n_c`.
Singleton variance is therefore zero; no shrinkage is introduced.  Then

`a_cj = (mu_cj-mu_j)^2 / (sigma2_cj + 1e-6)`.

For `A_j=sum_c a_cj > 0`, set `q_cj=a_cj/A_j`; when `A_j=0`, set
`q_cj=1/|E|`.  With natural logarithms and `0 log 0 = 0`, define

`h_j = -sum_c q_cj log(q_cj) / log(|E|)`.

Require at least two estimator classes.  A constant coordinate is assigned `h_j=1`,
not spuriously treated as class-owned.  Full-training weights remain
`w_j=(h_j+1e-6)/mean_k(h_k+1e-6)` and use population
`CV=std_ddof0(w)/mean(w)`.

Use SHA-256 domains exactly as follows.  A class fold is the unsigned big-endian
integer from the first eight bytes of
`SHA256("pass181-cieb-stage-a-v1|fold|<canonical-int-label>")` modulo five.  Within a
held-out identity, order rows by
`(SHA256("pass181-cieb-stage-a-v1|split|<example_id>"), example_id)` and assign
alternating rows gallery/query starting with gallery.  Exclude identities with fewer
than two images.  Average queries within identity and identities equally.  Any zero
post-ablation norm invalidates the seed rather than being silently excluded.

For each of the five held-out folds, estimate a separate entropy vector from the
other four folds.  Select exactly `ceil(0.10*512)=52` target coordinates ordered by
`(entropy ascending, coordinate index ascending)`.  For stability, compute all three
unique two-versus-two partitions of those four estimator folds.  Spearman uses average
ranks; a constant vector gives correlation zero.  The seed statistic is the median of
all `5*3=15` correlations, so no split may be chosen after inspection.

For nuisance-matched masks define, on the same estimator set,

`v_j = mean_c mean_{i:y_i=c}((z_ij-mu_j)^2)` and
`r_j = mean_c (mu_cj * p_cj)`,

where `p_c` is the unit own-class proxy.  Independently rank
`(log(v_j+1e-6), r_j)` with coordinate-index tie breaks and divide each rank into
eight equal-frequency bins.  The Cartesian product forms fixed 8-by-8 strata.  Every
control mask contains exactly the target mask's count from each stratum.  Within a
cell, select by
`SHA256("pass181-cieb-stage-a-v1|mask|<seed>|<fold>|<replicate>|<cell>|<coordinate>")`.
Scan replicate numbers from zero and retain the first 1,000 unique 52-coordinate masks
that differ from the target.  No outcome enters matching.  Record covariate balance
and the canonical SHA-256 of the `uint16[1000,52]` matrix.

For every unablated held-out query, select its 32 highest-cosine foreign gallery rows
once, with ties by split hash then example ID, and freeze them for target and all
controls.  Use every same-identity gallery row as positives.  Mask both query and
gallery, renormalize, and compute

`m = tau*logmeanexp(s_pos/tau) - tau*logmeanexp(s_foreign32/tau)`, `tau=0.05`.

For identity `i` and mask `b`, average `m_b-m_unablated` over its queries.  R@1 is
descriptive only for unablated and target masks; control-mask R@1 is not computed or
selected.

For seed `s`, let `T_s` be the equal-identity target effect, `R_sb` the equal-identity
effect of control mask `b`, `D_s=T_s-mean_b R_sb`, `S_s=sd_ddof1,b(R_sb)`, and
`Z_s=D_s/S_s`.  `S_s<=1e-12` invalidates the diagnostic.  Pool seeds equally:
`D_pooled=mean_s D_s`, `Z_pooled=mean_s Z_s`.

Bootstrap complete eligible identity labels jointly across all four seeds with 10,000
NumPy `PCG64` replicates and seed 181, recording the NumPy version.  Retain observed
`S_s` as frozen nuisance standardizers; the statistic is the equal-seed mean of each
resampled identity advantage divided by `S_s`.  The one-sided 95% lower bound is the
ordinary 2.5th percentile.

The screen **passes onward** only if stability is at least `0.60` in every seed, CV is
at least `0.10` in every seed, `Z_pooled>=0.05`, the bootstrap lower bound is positive,
and `D_s>0` in all four seeds.  A fail takes precedence if stability is below `0.30`
in at least three seeds, CV is below `0.05` in at least three seeds, or
`D_pooled<=0`.  Everything else is unresolved.

Execution is staged without changing the verdict: entropy stability and CV are
computed first and may trigger their registered early fail before constructing 1,000
matched masks.  This avoids expensive arithmetic when the estimator is already
unstable or effectively constant; no CPU wall-clock measurement enters the decision.

Any artifact-binding mismatch or threshold failure stops CIEB before implementation
or GPU.  Reconstructed **training** descriptors must match the digest-bound final
training pack at `2e-5`.  The legacy pre-head query export is known to be batch-shape
dependent and is not used.  Official R@1 is bound independently by recomputing it from
the immutable final query/gallery packs and matching the report and retrieval audit,
exactly as adjudicated before Pass159.  Query/gallery labels and IDs contribute no
candidate statistic.  Require a 512-D head, one proxy for every and only training
label, identical training example-ID order across seeds, and record hashes of the
manifest, this proposal, diagnostic source, and output schema.

A Stage-A pass establishes only that the statistic identifies harmful coordinates; it
**cannot pass Gate 1**, because post-training forward ablation is a different
intervention from backward preconditioning and the checkpoint already saw the held-out
training identities.  Conversely a failure means the registered necessary-condition
screen failed and implementation is unauthorized; it does not prove that every
possible entropy preconditioner is mathematically impossible.

The decisive Stage B is a class-disjoint bounded head-training/influence experiment.
Fit a frozen-trunk 512-D head and proxies to epoch 10 on 80% of In-Shop training
identities only; neither images nor labels from the remaining 20% may enter fitting or
entropy estimation. Compare equal-parameter-norm updates from CIEB with ordinary PA,
coordinate-permuted entropy, variance-matched diagonal weights, forward weighted
distance using the same entropy, and inverse entropy. On held-out identities, score
alignment with a proxy-free supervised-retrieval gradient. CIEB must retain at least
`0.20` update residual outside ordinary PA (`<0.10` fails), improve alignment over the
strongest control by `>=0.02`, have a positive identity-clustered 95% lower bound and
every seed positive, and match the sign and magnitude (within 20%) of a small stateless
step. A nonpositive mean or at least three of four nonpositive seeds fails; otherwise
the result is unresolved. No benchmark training run is authorized before Stage B.
Full re-audit:
`docs/repaired_gate2_reaudit_pass159_pass181_2026-08-08.md`.

## Stage-A result

**FAIL.** Split-half entropy-rank stability was only `0.0355`–`0.0864` across the
four seeds, and weight CV was `0.0385`–`0.0407`; all four seeds crossed both registered
early-fail thresholds. The matched-mask stage was not executed. No Stage B or GPU run
is authorized. Full result: `docs/pass181_cieb_stage_a_result_2026-08-09.md`.
