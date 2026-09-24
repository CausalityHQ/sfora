# B/16 resolution screen plan

## Goal

Test whether source-preserving 336-pixel detail improves a compact UNICOM B/16 retrieval profile at an acceptable image-to-result cost. All measurements use SOP TRAIN identities and remain exploratory.

## Gate 1: adapter correctness

1. Add a narrow B/16 resolution adapter that uses the unchanged 224-pixel graph and pretrained feature head. At 336, interpolate the position grid to 21×21 and area-resample final tokens back to 14×14.
2. Test invalid geometry and exact 224 parity with a small deterministic model before deploying.
3. On DGX, load the authenticated checkpoint and eight SOP TRAIN images; require exact 224 parity, finite 336 outputs, and record source, script, and image identities.

## Gate 2: paired serving feasibility

Build the same packed gallery and scorer for A (native 224), B (native 336), and C (224 tensor upsampled to 336). Time image decode, encoding, packing, and exact top-10 separately on the same query images at batch 1 and 32. Use reversed arm order, explicit synchronization, and raw samples. Stop if B batch-1 p50 exceeds 0.7 times a matched L/14@336 replay on the same harness, or if adapter parity fails. This is only a cost screen, not p99 certification.

## Gate 3: frozen quality

On a Gate 2 pass, encode the 59,551 SOP TRAIN rows once per arm. Score the 5,851-image product-disjoint holdout against itself and the full TRAIN gallery, first at native width and then with per-arm PCA-128 fitted only on 53,700 fit rows and packed to 130 bytes. Record per-query scores and product-bootstrap uncertainty. Never read SOP TEST images or labels. A frozen quality loss is diagnostic; it does not alone rule out matched fine-tuning.

## Gate 4: matched training

If the cost screen permits, train A/B/C for the same 1,000 full-backbone ArcFace updates, batch sequence, seed, and compact-head protocol. Evaluate packed full-gallery Recall@1, holdout mAP@R, B-vs-C detail effect, CUB transfer, training throughput, peak memory, and end-to-end latency. Apply the advance gates in the design spec, then seek independent seeds and official protocols only for a surviving candidate.

Each gate runs as one durable DGX job with its original exit status and raw artifact. Commit and push script plus decisions without unrelated worktree files.
