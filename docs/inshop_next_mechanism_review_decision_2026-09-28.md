# Next-mechanism review: reject the proposed reopening

The original Fable consultation `ae2da85c13b04bee` completed with exit0 in
**597 seconds**, under a $6/1,800-second maximum (not measured spend).
Its [raw answer and receipt](evidence/compact_metric/sop-siglip2-substrate-v1/inshop-next-mechanism-research-v1/consultation.json)
are preserved. No review duplicate or GPU experiment was started.

## Valid observation

The older official-result recipe used **23,342 fit images / 3,598 products**.
The later unseen-gallery recipe uses **13,283 fit images / 2,004 products**,
with **12,599 held images / 1,993 products**. Its absolute TRAIN scores are
not official-quality forecasts, and its current checkpoint has no official
result. Recipe contrasts under this split remain useful conditional evidence;
they do not answer the full deployment-regime question by themselves.

## Independently checked reasons not to run the recommendation

1. **It reopens an explicitly closed scaling arm.** The
   [frozen data-scaling gate](inshop_fit_product_scaling_gate_2026-09-27.md)
   measured full-minus-half R@1 **+0.3651 percentage points** with a positive
   product interval **[+0.1535,+0.5740]**, but missed the predeclared **+0.4**
   floor. Its decision explicitly stops before all-TRAIN retraining. Initial
   PCA, classifier, image diversity, batch schedules and repeat exposure also
   changed. This is not an isolated causal class-count effect.
2. **Its proposed rescue stratification uses a rejected instrument.** The
   [hard-product readout](inshop_siglip2_unseen_gallery_gate_2026-09-26.md)
   selected499 products/4,198 queries with only92 control misses and1.52×
   miss enrichment, below its frozen3×/100-miss requirements. It was rejected
   for method selection. New15%/10% post-hoc rescue thresholds cannot validate
   this instrument or override the original scaling decision.
3. **Its impossibility claims do not follow.** A negative subgroup effect
   cannot prove all-TRAIN learning cannot close the official gap. Positive
   subgroup rescue likewise does not validate extrapolation to a different
   identity count, gallery distribution, update budget or official split.
   No additional scoring was run to try to rescue a closed arm.
4. **One saturation number belongs to another checkpoint/regime.** The
   99.3613% fit Recall@1 came from the older23,342-fit-image bank model, not
   proof that every later13,283-image recipe memorizes its training set or
   that optimization can no longer improve generalization. Frozen-feature
   head tests do not universally measure full-encoder adaptation, but they
   remain valid tests of the fixed cached configurations actually evaluated.
5. **Predictions and cost assumptions are not measurements.** The projected
   official +0.9-point gain and28-minute/seed retrain estimate are unverified.
   Serving tails must be qualified on the actual checkpoint; identical dense
   architecture alone does not make p99 checkpoint-independent.

Decision: **KILL this proposed hard-stratum rescue and all-TRAIN reopening**.
No new source, method, loss, data, training budget, official read or production
default is selected from this review. It found no supported distinct candidate;
that is a limited search outcome, not proof no better algorithm exists.

## Current prerequisite and scope

Normal Hugging Face credential discovery was checked again after the review:
`get_token()` is absent on both Devbox and DGX Spark. No token contents were
read or printed. Authenticated DINOv3 owner grant remains **unknown**. The
previously reported source-access prerequisite has not changed. No model
download, secret search, alternate mirror or repeated approval request followed.

Existing qualified production work and all remote evidence are preserved.
No job or consultation remains active. The complete SOP/In-Shop quality and
matched image-to-top-k speed objective remains unmet. Further work requires
a supported, causally distinct candidate or availability of the already
requested source-access prerequisite; another duplicate recipe search is not
licensed by this result. There is no new quality, training or serving number.
