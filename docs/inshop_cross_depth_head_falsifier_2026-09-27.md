# In-Shop cross-depth head falsifier, 27 September 2026

The 22-block SigLIP2 encoder trained 11.88% faster at 100 updates but lost
0.009100 packed mAP@R on 12,599 product-disjoint official **TRAIN** held
images, beyond its frozen floor. The 24-block and 22-block pretrained source
caches already contain the same 25,882 TRAIN rows. Test one fit-only linear
transfer before paying for another training run: align normalized 22-block
source vectors to normalized 24-block vectors by orthogonal Procrustes with
translation on the 13,283 fit-product rows, then compose the fitted map with
the 24-block fit-only PCA-128 head. The composed affine head fits the existing
single 1024-to-128 linear layer and adds **zero** serving operations. Compare
its packed held-only symmetric retrieval with the 22-block's own fit-only
PCA-128 and the 24-block fit-only PCA-128 on identical held rows. This is a
source-initialization probe, not trained-model or official TEST evidence.

Freeze this kill rule before reading held quality: advance to one matched
100-update 22-block training screen only if the transferred head exceeds the
22-block own-PCA packed mAP@R by at least **0.005**, its paired
product-bootstrap 95% lower bound for that mAP difference is **positive**,
and packed R@1 does not decline. Require exact source-cache/model/row hashes,
fit-only PCA and mapping, finite outputs and exact 130-byte packed scoring.
Otherwise stop this mapping arm with no retraining or official read. If it
passes, count the extra source-cache export when judging total training cost;
the existing separate exports took **216.163 s** for 24 blocks and
**180.906 s** for 22 blocks. A one-pass two-depth cache exporter would need
measurement before any production speed claim. Layer dropping and linear
feature distillation are prior art; no novelty or SOTA claim follows.
