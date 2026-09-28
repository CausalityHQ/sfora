# Wider supervised training, fixed compact serving: one cheap capacity gate

ArcFace derivative and teacher transfer are closed. The earlier source
classifier diagnostic found+3.3203pp fit leave-query-out accuracy in1024D
versus128D, with42.37% source gradient outside the narrow span. Its added
source-centroid auxiliary saturated and was killed. A wider **main** trained
head is a different route for those coordinates: they receive the unchanged
ArcFace+bank supervision, rather than a weak auxiliary. Previously measured
trained raw1024 cosine does not bound a separately supervised256D head.

Fixed proposal: train one256D affine head, then fit an **uncentered**128-row
orthonormal map on normalized256D fit outputs only. Fold that map into the
affine head (C W,C b). Unit normalization before the final linear map cancels
under final unit normalization when no extra center/bias is introduced.
Export and evaluate the actual folded128D path; floating association changes
are not promised bit-identical to a two-matmul path. Serving remains one
1024→128 head and the existing130-byte packed representation. No new serving
module, wire format, encoder, model source, loss or teacher is proposed.

Before any implementation/training, one≤120s CPU fit-cache-only diagnostic:

- Original13283/2004 fit products/cache, same512 metadata-selected products
  and leave-query-out prototype scoring from the earlier source-classifier
  panel. No new panel selection, held/official read or fitted held statistic.
- PCA256 from fit normalized source, native128 from its leading128 components;
  replay narrow PCA hash and previous435/512 prototype hits. Report256D and
  folded128D fit prototype hits and paired product-bootstrap intervals.
- Uncentered compactor uses deterministic float64 SVD/sign convention. Check
  orthonormality, actual folded-head arithmetic and a small float64 normalized
  composition identity fixture. This is not a kernel parity certification.
- Use unchanged ArcFace margin0.3/scale64 plus8 detached bank, same initial
  class means and first17 original batches. Quantify wide main source-gradient
  energy outside narrow PCA rows plus the radial normalization axis.
- GO only if wide-minus-narrow prototype accuracy≥+1pp, paired95 lower>0;
  folded128-minus-narrow point≥−1pp; median unused wide-gradient fraction≥20%;
  median wide/narrow source-gradient norm ratio in[0.25,4]; all1088 gradients
  finite/nonzero, orthogonality/parity/inventory/authority pass, CPU≤120s.

Failure kills this fixed256→128 capacity configuration; no width/compactor/
floor search. A positive result only supports consequential design review and
one bounded paired encoder smoke before a separately frozen TRAIN retrieval
gate. It proves no generalization, SOTA, novelty or training/serving speed win.
Keep original128 baseline and production source unchanged during diagnosis.

## Terminal CPU result

Original computation exited1 while serializing a NumPy boolean; no receipt
was saved. Cast only that criterion to a native bool and reran once with the
unchanged gate. The repaired run exited0 in9.202887s. Receipt
`evidence/compact_metric/sop-siglip2-substrate-v1/inshop-wide-main-head-cache-v1.json`
SHA256 `2698063c7c5720b85dbd643bde678d61550c6a65dffbb4c598910d49779d474e`.

| Frozen fit-only measure | Result | Decision |
|---|---:|---|
| Native128 leave-query-out prototype accuracy |435/512,84.96094%|Control|
| Wide256 prototype accuracy |450/512,87.89063%|+2.92969pp,95%[1.5625,4.4922]pp|
| Actual folded128 prototype accuracy |435/512,84.96094%|Pass gross floor|
| Median wide gradient outside narrow span/radial axis |34.90907%|Pass20% floor|
| Median wide/narrow source-gradient norm ratio |0.9072373|Pass[0.25,4]|
| Folded normalized float maximum error |1.4305115e-6|Pass1e-5|
| Native narrow null residual maximum |8.8183050e-7|Pass1e-5|

All1088 gradients and frozen authority/budget guards passed. Independently
recounted hit arrays, medians, native-bool criteria and current probe source
hash from the saved receipt. This is **GO_BOUNDED_ENCODER_DESIGN**, not a
retrieval gain: no packed R@1/mAP@R, held evaluation, training wall, VRAM or
image-to-top-k measurements were made. Production128 head, checkpoint and
package remain unchanged. Consequential Opus/Astra review group
`579a7c2e61c745dd` was started once before GPU implementation; Spark was idle.
