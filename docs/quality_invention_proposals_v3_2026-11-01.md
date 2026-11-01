Brief read. The deliverable below follows its "What to produce" order: simulated perspectives first, then six ranked candidates with falsifiers. No files touched, no code run.

**Summary.** One candidate has a measured prior in the brief itself: the 512-input pretrained screen gained +1.05 and was rejected only against a cost-justified +2.0 bar that no longer applies. A natively trained 512 checkpoint, fine-tuned under the existing recipe, is the only candidate I rate above 30 % to clear +0.3 on held. Everything else is a 10 to 20 % shot.

## Simulated perspectives (simulations, not evidence)

- **Metric-learning researcher.** Argues the objective is saturated: full-bank SmoothAP already supplies global hard negatives, which is why batch composition, log-NCA, margins and pair samplers were inert. Predicts loss changes give at most +0.2. Objects that the member bank is refreshed about once per row in 1.2 epochs, so SmoothAP aims its gradient at stale near-ties during the only phase where embeddings move; would test bank freshness before any new loss.
- **Vision-foundation-model researcher.** Argues the fine-tune is 1.2 epochs at lr 1e-5, so the final representation is mostly the pretrained one, and headroom lives in resolution and pretraining, not in heads or re-rankers. Predicts a natively trained 512 tower beats interpolated 384 weights at 512. Objects that giant-opt lost to So400m pretrained, so "stronger tower" does not transfer automatically, and warns the pretrained +1.05 may shrink after fine-tuning as the Large distillation did.
- **Data-centric researcher.** Argues two thirds of errors are an information problem: a stamp close-up carries little identity, and the full-shot positive renders that stamp in about one patch. Predicts only more identity-carrying pixels or more product-identity supervision move R@1. Objects that synthetic detail crops were exactly zero-sum (fixed 34, broke 34), so crop tricks are dead; would consider external product-identity data with a protocol caveat.
- **Statistician.** With about 297 held errors and typical discordant counts of 70 to 120 per comparison, the paired SE is roughly 0.14 to 0.19 pp, so +0.3 is a 2-SE effect and +0.5 is 3 SE. Predicts most candidates below +0.3 will land "ns" and that six screens at one-sided 2.5 % carry about a 14 % family-wise false-positive rate. Objects to any official read without a second-seed confirmation of a passing screen.
- **Skeptical practitioner.** Thirteen theory-led passes produced nothing; every kept gain came from a one-run falsifier. Argues only candidates with a measured prior deserve GPU time first. Predicts re-ranking of any kind is dead given the oracle results. Objects to transductive use of TEST on protocol grounds and to external data without near-duplicate screening against TEST.
- **Systems engineer.** Notes cost is now secondary, so a 1.4× to 1.5× encoder is affordable, storage stays 514 B per row for every candidate below, and GB10 memory is not a constraint at 1024 tokens with batch 64 in bf16. Warns that bank re-encodes add 25 to 40 minutes per run and that variable-length (NaFlex) batching is real engineering.

## Candidates, ranked by expected gain × probability

All predicted magnitudes are UNVERIFIED unless a source is named. "Held" means the 5,851-query held split with the paired product bootstrap, as in the brief.

| Rank | Candidate | Held ΔR@1 if it works | P(pass) | Expected value | Falsifier wall time |
|---|---|---|---|---|---|
| 1 | Native-512 SigLIP2 So400m tower | +0.3 to +0.8 | 0.35 | ≈ +0.18 | 50 to 70 min |
| 2 | Fresh SmoothAP member bank | +0.1 to +0.4 | 0.20 | ≈ +0.06 | ≈ 60 min |
| 3 | Non-SigLIP backbone census | +0.3 to +0.6 | 0.15 | ≈ +0.06 | 30 + 60 min |
| 4 | Products-10K intermediate fine-tune | +0.2 to +0.5 | 0.15 | ≈ +0.05 | 70 to 90 min |
| 5 | Crop-to-whole self-consistency term | +0.1 to +0.3 | 0.12 | ≈ +0.03 | ≈ 35 min |
| 6 | WiSE-FT interpolation toward pretrained | 0.0 to +0.2 | 0.10 | ≈ +0.02 | ≈ 40 min, no training |

**1. Native-512 tower: swap to the SigLIP2 So400m patch16-512 checkpoint (1024 tokens), same recipe.**
Mechanism. Two thirds of errors are close-ups whose identity cues occupy about one patch in the full-shot positive at 384; at native 512 the same cue spans about 1.8× the pixels and the position embeddings and attention were trained at that density, so the full shot can encode what the close-up encodes. The steep loss below native (−1.3 at 280) shows the model is resolution-bound in this regime.
Not covered by FAILED. The 448/512 entries are pretrained screens judged against a +2.0 bar, and the "interpolation saturates at native 384" note implies the 384 checkpoint was interpolated. A natively trained 512 checkpoint, fine-tuned, under a quality-only objective is unmeasured. The SigLIP2 report (Tschannen et al. 2025, arXiv:2502.14786) ships So400m at 384 and 512 and reports gains for the 512 variant on its detail-heavy evaluations; the SOP magnitude is UNVERIFIED.
Prediction. Assume the pretrained +1.05 shrinks by half under fine-tuning: ΔR@1 +0.3 to +0.8, ΔmAP@R +0.5 to +1.5. In-Shop should move less (280 cost it only −0.23).
Falsifier. Optional 10-minute stage 0: pretrained screen of patch16-512 and patch16-naflex at 1024 tokens against the So400m@384 pretrained baseline, to separate token count from aspect preservation. Then one 1000-update run with the better checkpoint at 512 input, RandomResizedCrop(512, 0.8 to 1.0), 13 of 27 blocks frozen, same losses and learning rates, batch 64 (micro-batch 32×2 if needed). KILL if held ΔR@1 < +0.30 or the 95 % interval includes 0. On PASS, train a second seed, compare the 512 two-seed ensemble against the existing ensemble, then one official read.
Cost. Encoder FLOPs about 1.45× (tokens 1.40×, attention 1.97×), serving roughly 101 to 70 img/s, storage unchanged, train time 1.5× to 2×.
Most likely failure. Many SOP source images are below 512 px on the long side, so the 512 input is upsampled and carries no new detail; fine-tuning at 384 already extracts what exists and the pretrained gap collapses to zero.

**2. Fresh member bank: periodic full re-encode instead of last-batch-view refresh.**
Mechanism. Over 1000 updates each of about 53.6k fit rows is refreshed about 1.2 times, so a typical row is about 500 updates stale, and it is stale during the early phase where drift is largest (XBM, Wang et al. CVPR 2020, measures drift as decaying with iteration). SmoothAP concentrates gradient on the positives and negatives at the query's rank boundary, so a stale bank aims that gradient at pairs that are no longer the near-ties.
Not covered by FAILED. No entry varies bank freshness. The 2000-update and EMA entries change the trajectory, not staleness. The 32×2 sampler being inert "under a detached bank" is consistent with staleness mattering.
Prediction. ΔR@1 +0.1 to +0.4, ΔmAP@R +0.3 to +1.0 since mAP depends on the whole ranking. Assumption: drift over 500 updates exceeds the near-tie margins, which the FP8 flip count shows are around 1e-3 cosine.
Falsifier. One run, same recipe, no-grad eval-mode re-encode of all fit rows at updates 150, 350, 600 and 850 (drift is front-loaded). Each re-encode is about 6 to 9 minutes depending on whether decoded tensors are cached. Log the median cosine between the stale row and its fresh value at each re-encode. KILL if held ΔR@1 < +0.25 and ΔmAP@R < +0.5. If median drift is below 0.01 cosine at every re-encode, the mechanism is absent regardless of the headline.
Cost. Train time +40 to +60 %. Serving unchanged. Stacks with candidate 1.
Most likely failure. Slow drift: at lr 1e-5 the bank ordering is set by pretrained features plus the PCA head and barely changes, so fresh equals stale.

**3. Backbone census outside the SigLIP2 family, then one fine-tune of the survivor.**
Mechanism. The final representation is mostly the pretrained one, so a differently pretrained tower is the cheapest way to change what the embedding is sensitive to. Perception Encoder Core (Bolya et al. 2025, arXiv:2504.13181) and AIMv2 (Fini et al. 2024, arXiv:2411.14402) use objectives with dense or decoding terms that are claimed to retain more local detail than pure contrastive training.
Not covered by FAILED. Screened towers are SigLIP2 giant-opt, SigLIP2 Large and DINOv2; no CLIP-family tower from another lab.
Prediction. UNVERIFIED. The giant-opt result shows benchmark strength does not transfer to SOP. P(any of three beats So400m pretrained by +0.5) about 0.25; P(fine-tuned survivor ≥ +0.3) about 0.15.
Falsifier. Stage 1, ≤ 30 min: pretrained screen of PE-Core G/14@448, AIMv2-H/14@448 and one DFN or MetaCLIP-2 H/14 tower at native resolution: encode TRAIN, PCA head to 512, held R@1. KILL any tower below So400m pretrained +0.5. Stage 2, ≤ 60 min: standard recipe on the single survivor with about half the blocks frozen. KILL if ΔR@1 < +0.3 against the So400m single.
Cost. 2 to 6 GB download per tower; serving 1× to 3× depending on the survivor; storage unchanged.
Most likely failure. SOP identity needs same-object-rephotographed sensitivity that WebLI alt-text training happens to supply unusually well, and every alternative loses at the pretrained screen as giant-opt did.

**4. Intermediate fine-tune on Products-10K before the SOP fine-tune.**
Mechanism. SOP gives about 5 images per product, a thin sample of the part-versus-whole relation; Products-10K (Bai et al. 2020, arXiv:2008.10545) adds about 10k SKUs and 140k mixed studio and user images, so the upper blocks start the SOP fine-tune already adapted to product identity rather than alt-text. Protocol note: extra pretraining data is accepted practice on SOP (UNICOM, An et al. ICLR 2023, pretrains on LAION-400M clusters) provided TEST is untouched; dedupe the external set against SOP TEST by pretrained near-duplicate search before use.
Not covered by FAILED. All entries use SOP TRAIN only.
Prediction. +0.2 to +0.5 R@1, similar mAP, UNVERIFIED. Counter-evidence inside the brief: full TRAIN at fixed updates was −0.25, so more data does not help by itself under this recipe.
Falsifier. Download and dedupe, then stage A of 1000 updates with the existing ArcFace plus bank-SmoothAP on Products-10K, then stage B equal to the standard SOP recipe from stage-A weights, held eval. About 70 to 90 minutes if the download is fast. KILL if ΔR@1 < +0.3 or the interval includes 0.
Cost. Training 2×, no serving cost, one-time download.
Most likely failure. Negative transfer: packaged-goods SKUs are far from user photos of furniture and appliances, and 1.2 epochs of SOP cannot undo the shift.

**5. Crop-to-whole self-consistency as an auxiliary term.**
Mechanism. Pull the embedding of a 5 to 30 % area random crop toward the detached embedding of its own full image, while the product-level losses see only global views; a real close-up query then reaches its full shot by transitivity through the crop. DINO (Caron et al. ICCV 2021) uses the same local-to-global alignment.
Not covered by FAILED. RandomResizedCrop 0.15 to 1.0 fed crops to the class and ranking losses as product samples, which forces crops to compete against other products' full shots. Here the class loss never sees crops. Honest caveat: the transitivity mechanism is the same, so this is the nearest to covered of the six.
Prediction. +0.1 to +0.3, UNVERIFIED.
Falsifier. One run, consistency weight 1.0, crops at 192 px (cheap extra forward). About 35 minutes. KILL if ΔR@1 < +0.3.
Cost. Train time +15 %, serving unchanged.
Most likely failure. The same zero-sum trade: stamp A moves toward kettle A, stamp B toward kettle B, and when the kettles look alike nothing changes.

**6. WiSE-FT interpolation of the fine-tuned upper 14 blocks and head toward their initial weights.**
Mechanism. Product-disjoint TEST is a distribution shift; weight interpolation between pretrained and fine-tuned weights improves shifted-distribution accuracy at no training cost (Wortsman et al. CVPR 2022). Head interpolates toward its PCA initialisation.
Not covered by FAILED. EMA is a temporal average along one trajectory, and "retained teacher init" is a training-time anchor; post-hoc interpolation is neither.
Prediction. 0.0 to +0.2, UNVERIFIED. The recipe is already a mild fine-tune selected on held, so little room.
Falsifier. Evaluate α in {0.6, 0.7, 0.8, 0.9} on held with the existing checkpoint, about 8 minutes each. KILL if the best α is < +0.25 over α = 1. Any pass must be confirmed on the second seed before touching the ensemble.
Cost. Zero.
Most likely failure. Interpolation just removes useful adaptation.

**Excluded on purpose.** Transductive use of TEST images (protocol breach). Any new re-ranker (oracle role, oracle listing and oracle cluster results already bound this family near zero). Bank-centroid ArcFace prototypes (class weights already approximate class means after 1000 updates). Asymmetric two-head scoring (no mechanism for a symmetric target).

**Sequencing.** Run candidate 1 first; it has the only measured prior. Run candidate 2 concurrently on the existing 384 tower since it stacks with any backbone. Confirm any pass with a second seed before the single official read, per the statistician's family-wise caveat.