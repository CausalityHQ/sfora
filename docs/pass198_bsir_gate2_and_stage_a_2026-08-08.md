# Pass 198 — Batch-Shape Invariant Retrieval (BSIR)

## Measured provenance

The artifact-bound Pass159 audit exposed a repeatable deployment-path defect in all
four corrected In-Shop Proxy Anchor checkpoints.  The legacy batch-256 reconstruction
and canonical batch-128 export differed in exactly the last 138 of 14,218 query rows,
because those rows were evaluated as a batch of 138 versus batches of 128 and 10.  The
worst coordinate discrepancy was `6.39e-4`–`8.45e-4`.  Train and gallery remainders
were identical under both batch sizes and agreed within `1.35e-7`.

This proves that the deployed eval-mode descriptor is numerically a function of outer
tensor shape.  It does **not** yet prove a retrieval error or a direction that can
improve R@1.  That distinction is the Gate-1 question below.

## Frozen candidate

Retain ordinary Proxy Anchor training.  On a fixed subset of steps, after the ordinary
train-mode forward has updated BatchNorm buffers once, snapshot every module's
individual training flag and perform two gradient-enabled eval-mode shadow forwards
of the same transformed anchor images.  Legal training images pad the outer tensor to
shapes 128 and 256; padding labels enter neither loss and padding rows receive no
descriptor loss.  Restore every saved module flag before backward.  Penalize squared
distance between the anchors' normalized deployed descriptors, normalized by the
same discrepancy measured once at ImageNet initialization, with fixed coefficient
`lambda=0.05`.  Inference remains one model, one view, and one 512-D descriptor.

The method is intended to make supervised retrieval invariant to finite-precision,
tensor-shape-dependent variation in the deployed descriptor.  It is not a claim that
ordinary BatchNorm batch-composition dependence is new.

## Gate 2 — LIVE-NARROW

An adversarial primary-source audit found no method with the same training object,
data flow, and decision point.  Batch Renormalization corrects train-mode minibatch
statistics and train/eval mismatch; Ghost and Virtual BatchNorm change the
mathematical training normalizer; EvalNorm and EMAN repair evaluation or teacher
statistics; R-Drop, the Pi-model, and Mean Teacher impose consistency under dropout,
augmentation, or changing weights.  None compares the same eval-mode retrieval
descriptor under two outer tensor shapes that should be identical in exact arithmetic.
Recent numerical-invariance work repairs arithmetic rather than learning a paired
retrieval consistency penalty.

The closest internal proposal, Pass134 counterfactual batch-composition invariance,
changes semantic peers in train mode and measures genuine BatchNorm coupling.  BSIR
holds the anchor pixels and eval-mode state fixed and changes only outer execution
shape.  Under the repaired exact-mechanism rule, Pass134 is an adjacent mandatory
control rather than a Gate-2 death.

Primary neighbours:

- Sergey Ioffe, *Batch Renormalization*, NeurIPS 2017,
  <https://proceedings.neurips.cc/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf>.
- Elad Hoffer et al., *Train longer, generalize better*, NeurIPS 2017,
  <https://papers.neurips.cc/paper/6770-train-longer-generalize-better-closing-the-generalization-gap-in-large-batch-training-of-neural-networks.pdf>.
- Tim Salimans et al., *Improved Techniques for Training GANs*, NeurIPS 2016,
  <https://papers.nips.cc/paper/2016/file/8a3363abe792db2d8761d6403605aeb7-Paper.pdf>.
- Xiaobo Liang et al., *R-Drop*, NeurIPS 2021,
  <https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html>.
- Sungyeon Kim et al., *Proxy Anchor Loss for Deep Metric Learning*, CVPR 2020,
  <https://openaccess.thecvf.com/content_CVPR_2020/html/Kim_Proxy_Anchor_Loss_for_Deep_Metric_Learning_CVPR_2020_paper.html>.

The audit also identified a mechanistic risk: the residual and the difference between
the two execution-shape Jacobians may both be tiny, so a symmetric consistency
gradient can be second-order-small and dominated by rounding noise.  A stop-gradient
branch would be a different operator and is not silently substituted.

## Frozen Stage-A retrieval-causality diagnostic

This diagnostic is committed before inspecting any neighbour or correctness change.
It uses the four immutable, digest-bound artifacts in
`docs/pass159_stage_a_manifest.json` and no new model execution:

1. Reconstruct normalized legacy query descriptors from each seed's saved pre-head
   query features and exact checkpoint head.  Require exact label/order binding to the
   canonical query pack.  Only rows 14,080–14,217 may differ materially; the earlier
   rows and canonical gallery remain immutable controls.
2. For each of the 552 seed-row pairs, use chunked float64 cosine retrieval against
   that seed's canonical gallery.  Record descriptor L2 and angular drift, canonical
   and legacy nearest-gallery row and identity, canonical top-1/top-2 cosine margin,
   the sufficient stability certificate `margin > 2*||delta_q||_2`, and canonical
   versus legacy correctness.
3. Recompute full R@1 by replacing only the 138 canonical tail queries with their
   legacy counterparts.  Record nearest-identity flips, correct-to-wrong,
   wrong-to-correct, net R@1 change, and absolute correctness changes per seed.

The candidate **passes onward** to an implementation preregistration only if at least
two of four seeds each have at least three absolute correctness changes among the 138
exposed rows (at least `0.0211` full-dataset R@1 point per qualifying seed), and the
pooled four-seed rate has at least twelve absolute correctness changes.  This is an
effect-existence threshold, not a claim that either numerical realization is better.

The candidate **fails Gate 1** if no correctness change occurs in all 552 pairs, or if
fewer than two seeds have three changes and the pooled count is below twelve.  Other
outcomes are unresolved.  Nearest-identity flips without correctness changes are
reported but cannot pass: numerical non-invariance alone is an integrity issue, not
evidence for improving retrieval quality.

Even a Stage-A pass would authorize only a bounded matched-control falsifier.  BSIR
would have to beat ordinary continuation, equal-compute duplicate shadows, Batch
Renormalization, and Pass134-style semantic co-batch invariance on canonical R@1;
robustness without a quality gain does not satisfy this project's objective.
