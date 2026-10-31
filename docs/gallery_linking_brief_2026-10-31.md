# Brief: invent fundamental changes (data schema / format / structures / algorithms) for Sfora image retrieval

You have no history with this project. Everything you need is in this file. Do not assume anything about
what the team prefers, plans or has tried beyond what is listed under FAILED.

## Task
Image → top-k retrieval of the same product. Two public benchmarks with fixed protocols:
- Stanford Online Products (SOP): 59,551 TRAIN images / 11,318 products; 60,502 TEST images / 11,316 products
  (disjoint products), 12 super-categories (bicycle, cabinet, chair, coffee maker, fan, kettle, lamp, mug, sofa,
  stapler, table, toaster). Images are user-uploaded product photos: ~5.3 images per product, mixed backgrounds,
  viewpoints, lighting; often the same object re-photographed. Metric: Recall@1 with symmetric query/gallery over
  all TEST images, self excluded; also mAP@R.
- DeepFashion In-Shop: 25,882 TRAIN images / 3,997 products; TEST query 14,218 / gallery 12,612 images.

## The system as it exists now (state, not advice)
Data format and schema:
- Raw JPEG → RGB → resize to 384×384 (no crop at test; RandomResizedCrop(384, 0.8–1.0) + hflip at train) →
  SigLIP2 So400m/14 vision tower (27 blocks, width 1152, 729 patch tokens, attention-pooled 1152-D output).
- Head: L2-normalise the 1152-D pooled vector → Linear(1152 → 512) → L2-normalise.
- Storage: signed int8 quantisation of the unit vector (round(x·127)) plus one fp16 inverse norm → 514 bytes per
  gallery image. Flat array, no index structure.
- Search: exact brute-force cosine via int8 GEMM over the whole gallery, stable top-k. Search is <3 % of serving
  cost; decode + encoder dominate (~101 images/s end-to-end on one GB10 GPU with FP8 linears).
- Each image is one vector. No multi-vector, no tokens kept, no metadata, no category labels, no text used at
  train or test time (SigLIP2's text tower is discarded).

Training algorithm:
- Fine-tune on TRAIN only. Lower 13 of 27 blocks + patch/position embeddings frozen; upper 14 blocks + head trained.
- Objective: ArcFace (margin 0.3, scale 64) over product classes + 8.0 × SmoothAP ranking loss computed between the
  batch and a detached "member bank" holding the current embedding of every fit-split TRAIN image (bank rows are
  refreshed from their last batch view). Batches: 16 products × 4 images = 64, coverage-first identity sampling.
- AdamW (wd 0.05), vision lr 1e-5, head lr 1e-4, constant LR, grad clip 1.0, bf16 autocast, 1000 updates
  (SOP: ~1.2 image-epochs; In-Shop: 2000 updates ≈ 5 epochs). Head initialised from PCA of pretrained features.
- Selection: product-disjoint 90/10 split of TRAIN (fit/held); held queries vs the full TRAIN gallery; paired
  product-bootstrap 95 % interval must exclude 0. One official TEST read per kept change.

Current measured state (official TEST, packed int8): SOP R@1 93.967 / mAP@R 82.69; In-Shop R@1 97.039 / mAP 84.88.
Single training seed; seed SD ≈ 0.2 pp.

Error structure that is known (SOP, from earlier analyses): the remaining R@1 errors are dominated by
fine-grained confusions inside a super-category (near-identical products, e.g. two kettles of the same model line),
plus some queries whose only positives are extreme-viewpoint shots. Errors are roughly symmetric (query↔gallery).

## FAILED (measured; do not re-propose these as-is)
Each was pre-registered and measured on the held split with the paired bootstrap; Δ is vs the then-baseline.
Quality (SOP unless noted):
- 2000 vs 1000 updates (two recipes): ΔR@1 +0.21 [−0.21,+0.63] and −0.03 [−0.35,+0.29]. SOP saturates ~1.2 epochs.
- Full TRAIN (all rows) instead of the 90 % fit split, fixed updates: official −0.25 [−0.36,−0.14].
- Freeze 9 instead of 13 blocks: −0.22 [−0.59,+0.15]. Freeze-boundary curve flat on In-Shop too.
- EMA weight averaging (0.996): −0.55 [−0.92,−0.17], mAP −1.49.
- Super-category-homogeneous batches (harder in-batch negatives): R@1 +0.14 ns, mAP −0.58 [−0.97,−0.21].
  Interpretation: the full-bank SmoothAP already supplies global hard negatives; batch composition is inert.
- Horizontal-flip test-time augmentation (gallery or both sides): +0.09 / +0.14, not significant.
- Test-time resolution below native 384 (280, 224): −1.3 / −4.1 (SOP); In-Shop 280: −0.23, mAP −1.33.
- Resolution above native (448, 512 pretrained screens): −0.18 / +1.05 (below the +2.0 bar); position-embedding
  interpolation saturates at native 384.
- Head width 1024 vs 512: −0.15 [−0.27,−0.03]. 128 → 512 had helped; 512 → 1024 does not.
- Bigger backbone SigLIP2 giant-opt (1B): −0.38 pp pretrained vs So400m. DINOv2: far worse. DINOv3 unavailable.
- Smaller backbone SigLIP2 Large/16@384 under the same recipe: −0.67 [−0.88,−0.46], mAP −3.46.
- Distilling So400m@384 → Large@384 (cosine + batch-similarity MSE): +0.02 over undistilled Large; no transfer.
- Depth pruning (drop last 8 of 27 blocks, fine-tuned): −0.35 [−0.53,−0.17]. Training-free token merging: −0.86.
- Large + So400m embedding ensemble (In-Shop): +0.21 [−0.02,+0.44]. (A two-seed same-model ensemble is +0.34 held.)
- Earlier recipe-level rejects: replacing SmoothAP with log-NCA; negative-column margin on SmoothAP; 32×2
  paired-positive sampler (inert under a detached bank); proxy-anchor nested-rank head (collapsed); "retained 256-D
  teacher init" (gain decays to 0 by step 1000); frozen-source relational/anchoring re-ranking routes (capped).
- Compression: 64-byte noise-shaped PQ never beat plain int8 + sqL2. k-reciprocal / diffusion re-ranking lanes were
  tried in older campaigns without a kept gain.
Process (so you calibrate ambition): ~13 "invention passes" that argued from theory produced no kept gain; every kept
gain came from a 30–60 minute single-run falsifier on the held split.

## What to produce
Assume nothing short of a structural change moves R@1 by ≥ +1 pp now. First list which perspectives would examine
this and what each would argue, predict and object to (e.g. metric-learning researcher, information-retrieval
systems engineer, computer-vision representation researcher, statistician, skeptical production engineer). Then
synthesise ≤ 6 candidate FUNDAMENTAL changes, each changing at least one of: data schema (what is stored per
image/product), data format (vector vs tokens vs multi-vector vs structured), data structure (index/graph/hierarchy),
or algorithm (objective, inference procedure, re-ranking, query formulation). For each:
1. mechanism, in two sentences, and why the FAILED list does not already cover it;
2. predicted effect on SOP R@1 and mAP@R with a stated assumption (cite primary literature if you use a number);
3. the cheapest single-run TRAIN-only falsifier on one GB10 GPU (≤ ~90 min, 128 GB host RAM, 2k-image held
   query set vs 59,551 gallery), including what result would KILL it;
4. serving cost delta (bytes per image, encode cost, search cost);
5. the single most likely reason it fails on this specific system.
Rank them by expected gain × probability. Mark every unverifiable claim UNVERIFIED. Simulated perspectives are not
evidence. Read-only: no code execution, no downloads.

## UPDATE 2026-10-31 — measured since this brief was first written (all FAILED for rank-1, TRAIN-held, 5,851 queries)
- Training-free token MaxSim re-rank (9×9 tokens): −0.62 [−0.95,−0.34]. Background context vector: −0.36. Attention zoom view: −0.09 ns (product already fills ~78 % of frame).
- LEARNED token-pair re-ranker (2-layer cross-attention, top-32): R@1 −0.07 ns (mAP +0.86 significant). Overfits fit lists.
- VLM (Qwen3-VL-8B) pairwise "same product?" verifier on top-3: +0.07 ns; on hard pairs the VLM judges worse than the embedding.
- OCR text match: only 2.4 % of errors have query text matching a positive but not the wrong top-1.
- Two-seed embedding ensemble: KEPT, official 94.129 (+0.16).
- Held R@32 = 99.33 % vs R@1 94.92 %.
ERROR ANATOMY (manual inspection of 15 held errors): ~2/3 are DETAIL / LABEL close-ups (maker stamps, rating plates,
wheel rim, chair lever, kettle lid, furniture underside). They show little of the product; their positives are mostly
full-product shots; their wrong top-1 is another product's similar close-up. The rest are full shots of near-identical
models. Conclusion so far: no single-image re-scoring can resolve a stamp; only information from OTHER images can.

## NEW QUESTION
Invent ≤ 4 fundamental designs that use multi-image / gallery structure to resolve partial-view queries under the
SOP protocol (symmetric: every TEST image is a query against all other TEST images, unlabeled; self excluded). Past
graph methods FAILED (k-reciprocal, diffusion, alpha-QE, database-side augmentation; errors are symmetric so
reciprocity rules don't separate). Explain precisely why each new design differs from those. Consider e.g. learned
set/graph models trained on TRAIN that propagate identity from full-view neighbours to detail shots, "view-type"
aware retrieval, learned transductive clustering, or anything else. For each: mechanism; why not covered by FAILED;
protocol legitimacy (is using the unlabeled TEST gallery jointly allowed / how to make it legitimate); predicted
effect; cheapest TRAIN-only falsifier on one GB10 (≤ 90 min) using cached per-image 512-D vectors + 9×9×1152 tokens of
all 59,551 TRAIN images; KILL condition; serving cost. Rank. Mark UNVERIFIED claims. Simulated perspectives are not
evidence. Read-only.
