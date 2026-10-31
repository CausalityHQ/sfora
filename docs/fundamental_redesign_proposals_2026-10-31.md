# Fundamental-redesign proposals (contextless Fable specialist eb0f33382ca64e84, 2026-10-31)

Input: docs/fundamental_redesign_brief_2026-10-31.md only. Simulated perspectives are not evidence; every number below is the specialist's, unverified until measured.

Brief read in full. No other inputs are needed, so the plan below works only from that file. Simulated perspectives are labelled as such and are not evidence.

**Recommendation.** Run one shared cache pass over the fine-tuned So400m on all 59,551 TRAIN images plus the held queries, storing per image the global vector, the final-block patch tokens pooled to a 9×9 grid, the block-13 tokens pooled the same way, and the attention-pooling head's attention map. That one pass feeds three training-free checks in minutes and one learned candidate in a quarter hour. The learned token-level pair re-ranker is the only candidate with primary-literature support for a gain of this size on top of a strong global descriptor, and it is the only one that adds information the FAILED list never touched: the per-patch evidence that attention pooling discards. Everything else is ranked below it.

## Perspectives (simulated, not evidence)

| Perspective | Would argue | Would predict | Would object |
|---|---|---|---|
| Metric-learning researcher | The embedding objective is saturated. Updates, data, freeze depth, head width and batch composition are all inert, so the pooled vector has no unused information left. Gains must come from evidence outside the vector. | A learned re-ranker over local tokens recovers a fifth to a third of errors whose positive sits in the top 32. | SOP has duplicate listings labelled as different products, which caps R@1 below 100. UNVERIFIED fraction. Re-rankers can lower mAP@R if they only fix the top position. |
| IR systems engineer | Search is under 3% of serving cost, so a 20× byte budget and a 10× search budget are affordable. Anything that runs the backbone on candidates per query is not. | A two-layer re-ranker on cached compressed tokens costs under 5% throughput. Keypoint matching or a VLM judge costs 10× to 100×. | Token stores change the index from flat to two-tier, which adds a failure surface. |
| Vision representation researcher | Fine-grained cues die twice: once when a non-square photo is squashed to 384 px with the background included, and once when 729 tokens collapse to one vector. SigLIP2 trained its dense features explicitly, so tokens are usable without a new backbone. | Attention-guided zoom recovers part of the pixel gain the pretrained 512 screen showed, without position-embedding interpolation. | Fine-tuning the upper 14 blocks for a pooled objective pushes tokens toward redundancy with the global, so final-block tokens may carry little extra. Block-13 tokens are frozen pretrained features and avoid this. |
| Statistician | Two thousand held queries cannot certify a half-point effect. For paired ΔR@1 with a 2% to 3% discordant rate the standard error is about 0.35 pp. | Only effects at or above +0.7 pp will pass the lower-95 gate at 2k queries. All held images as queries, about 5.9k, bring the certifiable effect to about +0.4 pp. | Six falsifiers need a pre-registered order and one official read per kept change. Retrain-based candidates must be seed-matched because seed SD is 0.2 pp. Training-free candidates share the backbone so seed noise cancels, which favours them. |
| Skeptical production engineer | Thirteen theory passes failed. Only cheap single-run falsifiers have ever paid off here. New model dependencies and new storage tiers are debt. | Training-free checks come back negative more often than not, and the one learned candidate decides the campaign. | Keypoint matchers and VLM judges are research probes, not serving components. Any candidate that needs raw JPEGs at query time doubles storage. |

## Candidates

**1. Learned token-level pair re-ranker on cached compressed patch tokens.** Schema, structure and algorithm change.
- **Mechanism.** Store a small grid of compressed patch tokens per image beside the global vector. A two-layer cross-attention scorer re-ranks the top 32 global candidates per query using token-to-token evidence. FAILED covers only global-vector re-ranking, which adds no new information. Token merging in FAILED was an encoder speed trick, not a storage format.
- **Prediction.** SOP R@1 +0.5 to +1.2, mAP@R +0.3 to +0.8. Assumption: held R@32 is at or above 98.5, UNVERIFIED and measured in-run, and the scorer net-fixes a fifth to a third of those recoverable errors. Literature: Reranking Transformers on SOP reported roughly +2.7 R@1 over their global baseline at the 82 level, Tan, Yuan and Ordonez, ICCV 2021, exact figure UNVERIFIED. AMES, Suma et al., ECCV 2024, shows memory-efficient local-descriptor re-ranking beating strong globals on landmark benchmarks.
- **Falsifier.** One run on the shared cache. Pool 27×27 tokens to 9×9 by averaging, PCA to 256 dims, keep fp16 in host RAM for both final-block and block-13 tokens. Mine top-32 lists for every fit query against the fit gallery with self excluded. Train a 2-layer, width-256, 4-head transformer over the concatenated query and candidate tokens with segment and 2D position embeddings, listwise softmax over the 32 candidates plus pairwise BCE, with the global cosine fed as a scalar feature. Stage 0 before training: ColBERT-style MaxSim over the same tokens fused with the global by z-scored sum. Score held queries against the full 59,551 gallery. KILL if paired-bootstrap lower-95 of ΔR@1 is at or below 0, or if point ΔmAP@R is below −0.2. Pass to one official read only if lower-95 is above 0 and the point estimate is at or above +0.5.
- **Serving delta.** Shown in the table below. Rerank compute is about 3% of one encoder pass per query.
- **Most likely failure here.** Fit-split lists are already solved by the embedding that trained on them, so the scorer trains on an easier distribution than held. Mitigation inside the same run: train only on lists where the top-2 cosine margin is small, or mine lists with the frozen block-13 tokens.

**2. Attention-guided object-centric second view at native 384.** Format and schema change.
- **Mechanism.** Use the pooling head's attention map to box the product, crop the original JPEG with a margin, resize to 384, encode a second time. Store the zoomed vector alone or beside the full-frame vector. FAILED tested whole-frame resolution changes, which hit position-embedding interpolation. This spends the native 729 tokens on the object instead.
- **Prediction.** R@1 +0.3 to +0.8 training-free, up to +1.0 if the encoder is then fine-tuned with zoom views. mAP@R +0.2 to +0.5. Assumption: the median attention box covers at most 60% of the frame, UNVERIFIED and measured in-run. The pretrained 512 screen's +1.05 shows the pixel budget still matters. Literature on saliency zoom for fine-grained recognition: Fu et al., RA-CNN, CVPR 2017; Recasens et al., ECCV 2018. Magnitudes there are for classification and UNVERIFIED for retrieval.
- **Falsifier.** Second encoder pass over all 59,551 gallery images plus held queries, about 10 minutes beyond the shared cache. Evaluate three scorers: zoom only, sum of the two cosines, average of the two unit vectors renormalised. Early diagnostic: if the median box exceeds 80% of the frame, stop. KILL if all three variants have lower-95 ΔR@1 at or below 0.
- **Most likely failure here.** Crops below 0.8 of the frame are out of the training crop range, so the fine-tuned encoder sees a distribution shift. Many eBay photos already fill the frame, leaving no leverage.

**3. Context vector from the complement of the attention map.** Schema change, zero extra encoder cost.
- **Mechanism.** Pool the same tokens with weights proportional to one minus the pooling attention, project with a 128-dim PCA, store as a second small vector. Score is global cosine plus a small weight times context cosine. Same-listing photos share a session, meaning background, floor, lighting and camera, and that cue is legitimately predictive under this protocol. Nothing in FAILED separates object evidence from session evidence.
- **Prediction.** R@1 +0.0 to +0.6, mAP@R ±0.3. Mechanism and magnitude UNVERIFIED, no primary literature on products. Could be negative.
- **Falsifier.** Five minutes on the shared cache. Grid the weight over 0.1, 0.2 and 0.3 on fit queries only, then read held once. KILL if lower-95 ΔR@1 is at or below 0 at the fit-chosen weight.
- **Most likely failure here.** ArcFace already rewards any discriminative cue, so the global vector encodes session already. White-background listings carry no context, and one seller listing several identical items shares a session across products, which inverts the cue.

**4. Training-free geometric verification over keypoints.** Schema and algorithm change.
- **Mechanism.** Extract ALIKED or SuperPoint keypoints per image, match the query against each top-20 candidate with LightGlue, and add a log-inlier term to the cosine, or use inliers only to break ties within a cosine margin. Same physical item re-photographed yields geometrically consistent matches. FAILED has no local-feature route.
- **Prediction.** R@1 +0.2 to +0.8, mAP@R +0.1 to +0.4. Assumption: at least 30% of fine-grained confusions involve the same item or scene re-photographed, UNVERIFIED. Literature: DELG local re-ranking adds roughly +2 mAP on ROxford-Medium over its global, Cao, Araujo and Sim, ECCV 2020, figure UNVERIFIED and from landmarks. LightGlue: Lindenberger, Sarlin and Pollefeys, ICCV 2023.
- **Falsifier.** Keypoints for 61.5k images at up to 640 px, then 40k pair matches for 2k held queries against the top 20. Choose the weight and the margin gate on fit queries. KILL if lower-95 ΔR@1 is at or below 0 for the fit-chosen setting. Timings on GB10 are UNVERIFIED and estimated at 40 minutes.
- **Most likely failure here.** Smooth consumer products give few repeatable keypoints, and two listings of one model line share the same logo and button geometry, so inlier counts do not separate them. Low source resolution makes this worse.

**5. Vision-language-model pairwise verifier on the top 3, run as a ceiling probe.** Algorithm change, not deployable at current serving cost.
- **Mechanism.** For queries whose top-2 cosine margin is small, show the query and each top-3 candidate side by side to an open VLM such as Qwen2.5-VL-7B and re-rank by the yes-logit for "same product listing". It reads printed text, counts features and compares details, which also subsumes an OCR schema. Nothing in FAILED uses a generative verifier.
- **Prediction.** R@1 +0.3 to +1.5 if VLM fine-grained same-item judgement beats the embedding on margin-ambiguous cases, UNVERIFIED. Without the margin gate, harm is likely because VLMs call viewpoint changes "different". Reference: Qwen2.5-VL technical report, Bai et al., 2025.
- **Falsifier.** About 6k pairs for 2k held queries, estimated 40 to 90 minutes on GB10 at 7B, UNVERIFIED. KILL if lower-95 ΔR@1 is at or below 0. The failure result is itself the payload: if a strong generic verifier cannot fix margin-ambiguous top-3 cases, the duplicate-listing ceiling hypothesis gains weight and candidates 1, 3 and 4 lose prior.
- **Most likely failure here.** The VLM demotes true positives seen from different angles more often than it promotes them.

**6. Analogy query expansion from the labelled TRAIN bank.** Algorithm and query-formulation change, lottery ticket.
- **Mechanism.** For each query, find its nearest TRAIN images, take their product-mates' embedding differences as view-variation vectors, add scaled copies to the query, and score the gallery by the maximum over the expanded query set. This borrows viewpoint variation from similar labelled products to reach extreme-viewpoint positives. FAILED expansions were all unlabelled gallery-graph methods.
- **Prediction.** R@1 +0.0 to +0.3, mAP@R ±0.2. Literature analogue: feature hallucination by analogy, Hariharan and Girshick, ICCV 2017. Transfer to retrieval UNVERIFIED.
- **Falsifier.** Fifteen minutes on the shared cache. Grid the neighbour count and scale on fit, read held once. KILL if lower-95 ΔR@1 is at or below 0.
- **Most likely failure here.** Variation vectors are product-specific noise in a space already trained for viewpoint invariance, and the max over an expanded set raises false positives.

Serving cost deltas against the current 514 bytes per image and about 101 images per second end to end:

| Candidate | Bytes per image | Encode cost | Search cost |
|---|---|---|---|
| 1 token re-ranker, light tier | 5 to 11 KB, 10× to 21× | +<1% for a projection | +3% of an encoder pass per query |
| 1 heavy tier, full tokens with a pair-ViT | ~840 KB int8 | none extra | ~10 encoder passes per query, not deployable |
| 2 zoom view | 514 or 1,028 | 2 encoder passes, throughput about halved | 1× or 2× of <3% |
| 3 context vector | 644 | none | negligible |
| 4 geometric verification | 17 to 256 KB | +5 to 10 ms per image, UNVERIFIED | 20 matcher calls per query, becomes dominant |
| 5 VLM verifier | 514 plus retained JPEGs | none | 1 to 3 s per gated query, about 100× |
| 6 analogy expansion | 514 plus a 30 MB TRAIN bank | none | one extra small kNN |

## Ranking and run plan

| Rank | Candidate | P(≥ +1 pp official) | Expected ΔR@1 | Deployable | Falsifier wall time |
|---|---|---|---|---|---|
| 1 | Token re-ranker | 0.25 | +0.5 | yes | 60 min shared cache + 20 min |
| 2 | Zoom view | 0.12 | +0.3 | yes | +10 min on the cache |
| 3 | VLM verifier | 0.20 | +0.4 | no, probe only | 40 to 90 min |
| 4 | Geometric verification | 0.10 | +0.25 | marginal | 40 min |
| 5 | Context vector | 0.07 | +0.1 | yes | +5 min on the cache |
| 6 | Analogy expansion | 0.04 | +0.05 | yes | 15 min |

Probabilities are judgement, UNVERIFIED. Run order: the shared cache first, then stage-0 MaxSim, zoom and context in the same session, then train the re-ranker. Run the VLM probe only if candidate 1 is killed or passes weakly, because its outcome re-prices every re-ranking idea. Candidates 4 and 6 run only after 1 to 3 have reported. Use all held images as queries for the training-free checks, since the cost is nil and it halves the certifiable effect size.

Considered and excluded as adjacent to FAILED or structurally inert:
- **Database-side augmentation, alpha query expansion, size-constrained gallery partition.** Same graph family as k-reciprocal and diffusion, and roughly symmetric errors mean both listings pick each other, which no reciprocity rule separates.
- **Learned pair metric on global vectors.** No new information beyond the vector.
- **Per-super-category expert heads.** The homogeneous-batch and 1024-head results say neither negatives nor capacity bind.
- **NaFlex aspect-preserving checkpoint.** Resolution family, small expected effect, full retrain.
- **Pseudo-label adaptation on TEST images.** Protocol grey zone and no domain shift to exploit.
- **Merging duplicate TRAIN listings.** Session cues are legitimately predictive under same-listing retrieval, so merging could remove a useful signal.
- **Frozen pretrained semantic branch.** Ensemble family, capped near +0.3 by the two ensemble results.

Risks that cut across all six: the duplicate-listing ceiling may put +1 pp out of reach, and only the VLM probe plus a margin diagnostic on held errors can measure it. Six tests at the same gate inflate false positives, so keep the pre-registered order and one official read per kept change. Candidate 1 trains on fit lists the embedding has already overfit, so its held transfer is the single biggest technical risk.