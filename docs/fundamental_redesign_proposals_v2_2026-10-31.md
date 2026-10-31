# SOP redesign: five candidate changes, ranked, with one ~90-minute screen to test them first

This is read-only work: I ran no code and changed no files. The analysis uses only the brief and the FAILED list. Every forecast, threshold and timing below is **UNVERIFIED** unless it is arithmetic from the brief's own figures.

## Perspectives (simulated, not evidence)

| Perspective | Argues | Predicts | Objects to |
|---|---|---|---|
| Metric-learning researcher | Losses, samplers, margins and log-NCA are used up. The remaining choice is how similarity is *scored*: with a per-image uncertainty, cosine is no longer a sufficient statistic. | An objective tweak alone will not give +1. | Re-rankers that are tuned on the held split being judged. |
| Retrieval systems engineer | R@32 = 99.33 % means recall is solved. This is a rank-1 precision problem inside a 32-item shortlist, so extra work should run only on hard queries. | Cheap low-capacity score fusion is the most robust approach. | Anything that multiplies encoder passes, since encode dominates cost. |
| Vision representation researcher | A close-up covers about 3×3 patches of a 27×27 grid in the full shot. Token MaxSim and attention zoom never matched scales. Scene and lighting statistics sit in the frozen early blocks. | Regional descriptors help only where the detail is visible in a full shot. | Rating plates and undersides never appear in full shots. |
| Photo/"same photo session" analyst | Images from one listing share lighting, white balance, surface and framing habits, and close-ups keep some of that. | Context is the only real information channel for close-ups. | Sensor noise or JPEG-quantisation forensics. Those are provenance cues and are invalid. |
| Statistician | Assuming held R@1 ≈ 94 %, the 2k held queries give about 120 errors, ~80 of them close-ups, so +1 pp means about 20 net fixes. With ~25 queries changed, the paired 95 % CI is about ±0.5 pp. | Effects below ~+0.5 pp cannot be certified on 2k queries. | Looking only at errors counts fixes but never breaks. Any tuned parameter needs product-disjoint cross-fitting. |
| Skeptical production engineer | 13 theory passes produced nothing. Measure the ceilings before building anything. | Every candidate fails. | Transductive or session cues that are really dataset artefacts. Those need an admissibility ruling first. |

**Falsifiable claims that come out of this:**
- **H1:** most wrong top-1s for close-up queries are themselves close-ups of other products, and those errors come in mutual pairs.
- **H2:** scene/context similarity prefers the true positive over the wrong top-1 well above 50 % of the time.
- **H3:** in fewer than 30 % of close-up errors is the depicted detail visible in a positive full shot.
- **H4:** if no visual cue links ≥ 60 % of close-up errors to their positives, +1 pp on SOP R@1 is unreachable by valid means.

## Shared setup for every screen: one recording encode pass (≤ 25 min)

Run the current single model once over 59,551 TRAIN images and record, per image:
- the pooled 1152-D vector `p`;
- the 512-D head output and its norm before L2-normalisation;
- the attention-pool weights over the 729 tokens;
- for blocks 6 and 12 (both frozen, so identical to pretrained): the token mean weighted by (1 − attention) and the per-channel standard deviation;
- 13 regional descriptors: the trained attention-pool head plus linear head applied to token windows (2×2 grid of 14×14, 3×3 grid of 9×9).

That is about 1.3 GB in host RAM. The test set is the 2k held queries against the full gallery, re-scoring the top-32 candidates.

**Generic kill rule (G).** Fit any parameter on one product-half of the queries, evaluate on the other half, then swap. KILL if any of these holds:
- paired ΔR@1 point estimate < +0.3 pp;
- product-bootstrap CI lower bound ≤ 0;
- ΔmAP@R < −0.3;
- R@1 on queries that are not close-ups drops by more than 0.1 pp.

Use all held queries for the CI if the pre-registration allows; they cost no extra encoding.

**Role score `r` (how "close-up-like" an image is).** Either the free pre-normalisation head norm (feature norm tracks image quality, as in MagFace, Meng et al. CVPR 2021) or a cross-fitted linear probe from `p` to leave-one-out distance from the image's own product. Pick by AUC on half A.

## Candidates, ranked by expected gain × probability

### 1. Uncertainty-conditioned pair scoring
**What changes:** data schema (one extra scalar per image) and algorithm (new similarity function).

1. **Mechanism.** Store a per-image uncertainty/role scalar and re-score the top-32 with a calibrated logit P(same | s, r_q, r_g) of about 8 parameters. This is the scalar-σ case of PFE's mutual-likelihood score (Shi & Jain, ICCV 2019), or condition-dependent score normalisation as in speaker verification (Auckenthaler et al. 2000). Cross-product close-up↔close-up pairs then get demoted, and because those errors are mutual, each demoted pair can fix two queries.
   - *Not in FAILED:* hubness/competition normalisation was a gallery-side correction from neighbourhood statistics. It had no query-side term, and its AUC of 0.55 measured hubness, not view role. The cross-attention re-ranker was a high-capacity token model and overfit; this one uses no tokens and ~8 parameters.
2. **Predicted effect.** R@1 +0.2 to +0.6 and mAP@R −0.3 to +0.3, assuming in ≥ 30 % of close-up errors the positive is the best-ranked candidate that is not a close-up. UNVERIFIED.
3. **Falsifier (~20 min on the recorded data).**
   - Pre-kill: count E1, the errors where every candidate ranked above the best positive F* has r_g > r_F*. KILL if E1 < 10 per 2k queries, which means a gross ceiling below +0.5 pp.
   - Then fit the logistic calibrator on half A, evaluate on half B, swap, and apply rule G.
4. **Serving cost.** +2 B/image (+0.4 %), encode +0, search +32 scalar operations per query.
5. **Most likely failure.** The wrong top-1s are full shots of look-alike products rather than close-ups, which disproves H1, so no role prior can reorder them.

### 2. Two-vector record: identity + scene context
**What changes:** schema and format (two vectors per image) and algorithm (fusion gated by role).

1. **Mechanism.** Add a 64-D int8 context vector, pooled from frozen-block tokens with the object suppressed by (1 − attention) plus channel statistics (AdaIN-style style statistics, Huang & Belongie 2017). Fuse as s + α(r_q)·s_ctx in the top-32. The listing's shared session cues (lighting, white balance, surface) are the only information channel that can link a close-up to its listing.
   - *Not in FAILED:* every failed re-ranker matched object content (tokens, zoom, cross-attention, VLM, OCR). The JPEG/provenance reject used file fields, not pixels.
2. **Predicted effect.** R@1 +0.3 to +1.0 and mAP@R +0.3 to +1.0, assuming the cue audit (see plan) finds a context cue in ≥ 40 % of close-up errors. UNVERIFIED.
3. **Falsifier (~20 min on the recorded data, no training).**
   - Pre-kill: among close-up errors where the positive is in the top-32, measure P(s_ctx(q, F*) > s_ctx(q, w)), where w is the wrong top-1. KILL if below 65 %; with ~80 errors, 61 % is the 95 % chance line.
   - Then cross-fit α and apply rule G.
   - Only if both pass: a ~60-min run training a contrastive context head on fit.
4. **Serving cost.** +64 B/image (+12 %). Encode about +1 %, because the tokens are already computed. Search is negligible.
5. **Most likely failure.** Close-ups are shot with macro or flash and don't inherit scene statistics. Stores that photograph many products on the same backdrop create new cross-product confusions.
   - **Validity:** the root must rule this admissible before any official TEST read.

### 3. Asymmetric close-up query expert
**What changes:** format (separate query-side mapping) and algorithm (gated asymmetric retrieval).

1. **Mechanism.** When a query is flagged as a close-up, pass `p` through a small residual adapter that maps into the **frozen** gallery space, trained against the gallery with positives = same product (asymmetric/compatible retrieval: Budnik & Avrithis, CVPR 2021; Shen et al. BCT, CVPR 2020). The gallery and full-shot queries are untouched, so the "broke 34" breakage pattern cannot occur outside close-up queries.
   - *Not in FAILED:* detail-crop training changed the *shared* embedding using synthetic crops. This changes only gated real close-up queries.
2. **Predicted effect.** R@1 +0.1 to +0.5, mAP@R ≈ 0. UNVERIFIED.
3. **Falsifier (60–70 min).** Train the baseline recipe plus an adapter branch on detached inputs, applied to fit anchors flagged as close-ups. Compare within the same run, expert on vs off, with an identical gallery. KILL if the close-up subgroup gains less than +2 pp, or under rule G.
4. **Serving cost.** 0 B; encode +≈0 (an MLP on `p` for about 10–15 % of queries).
5. **Most likely failure.** The information limit: without a product-specific cue, the expert can only move a close-up toward its category mean. Too few real close-ups in TRAIN to train it.

### 4. Scale-matched regional gallery descriptors
**What changes:** format (multi-vector gallery records) and algorithm (max over regions).

1. **Mechanism.** Store 13 regional descriptors per gallery image, made by applying the trained attention-pool head to token windows (R-MAC-style, Tolias et al., ICLR 2016). For gated close-up queries, score max(s, s_region − δ), so a close-up can match the zoomed-in region of a full shot at the right scale.
   - *Not in FAILED:* MaxSim worked at token scale (~42 px), and zoom used a single 78 % box, so neither matched scales.
2. **Predicted effect.** R@1 0 to +0.4 (only the visible-part subset), mAP@R 0 to +0.5. UNVERIFIED.
3. **Falsifier (~15 min on the recorded data).** KILL without any compute if the cue audit shows < 30 % of close-up errors have the detail visible in a positive full shot (that is, if H3 holds). Otherwise cross-fit δ and apply rule G.
4. **Serving cost.** 514 B → ~7.2 KB per image (14×). Encode about +3 %, search negligible.
5. **Most likely failure.** Plates and undersides are invisible in full shots, and generic parts such as rims match windows from every product in the category, the same failure as token MaxSim.

### 5. Two-level listing index (transductive)
**What changes:** data structure (cluster index) and algorithm (listing-level scoring).

1. **Mechanism.** Build high-precision listing clusters from mutual nearest neighbours among images with low r. Score gated queries as s(q,g) + β·s(q, mean of the rest of g's listing).
   - *Not in FAILED as-is:* k-reciprocal and diffusion were not gated by role, and the oracle-cluster test was a token union rather than a global set score. Still, this is the closest relative of the failed family.
2. **Predicted effect.** R@1 ≤ +0.3, mAP@R up to +1. UNVERIFIED.
3. **Falsifier (~10 min).** First run it with the **oracle** (true) listings. KILL the whole transductive family if the oracle ΔR@1 < +0.5.
4. **Serving cost.** +4 B cluster ID per image plus cluster means; an offline graph build; harder incremental updates.
5. **Most likely failure.** Aggregation rewards same-category listings rather than identity. **Validity:** needs a ruling on using the test gallery's structure.

## Recommended plan (fits in ~90 min)

1. **One recording pass** (≤ 25 min), producing everything listed in the shared setup.
2. **Manual cue audit, in parallel (~30 min, done by a human).** Take 50 close-up errors, view the query, the wrong top-1 and the best positive F*, and label the cue that links the query to F*:
   - **V:** the detail is visible in F*;
   - **C:** shared context (scene, lighting, surface);
   - **T:** text;
   - **N:** nothing.

   This sizes the ceilings: candidate 4 can do at most V, candidate 2 at most C, and candidate 1 needs no cue.
3. **Run the pre-kills and rule G for candidates 1, 2, 4 and 5** on the recorded data, in that order.
4. **Only survivors get a training run.** Candidate 3 runs only if the audit shows C or V ≥ 30 %.

**Stop rule:** if N ≥ 60 % (H4) and candidate 1's E1 < 20, declare SOP R@1 limited by close-ups at about the current level and move effort to mAP@R, In-Shop or efficiency.

## Risks
- Fitting to the held split: everything must be cross-fitted and the official TEST read once.
- Small subgroups: with ~80 close-up errors, effects below +0.5 pp cannot be certified.
- Admissibility of candidates 2 and 5.
- `r` may fail to separate close-ups. Check its AUC before trusting candidate 1's ceiling.

My overall estimate that any of these reaches +1 pp on official SOP is about 10–15 % (UNVERIFIED). The measurement that decides it is the cue audit together with E1, not a training run.

**Considered and dropped:**
- Aspect-preserving NaFlex input: no link to the error anatomy.
- Masking close-up positives in the ranking loss: the same family as log-NCA, which failed.
- Sinkhorn / dual-softmax scoring: the hubness family, which failed.
- Sensor or JPEG forensics: provenance cues, invalid.
- Adapting the model on the test gallery: violates TRAIN-only.
- End-to-end vMF uncertainty embeddings: only as a follow-up if candidate 1 passes.