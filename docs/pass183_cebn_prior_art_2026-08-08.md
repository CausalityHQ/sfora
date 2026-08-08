# Pass 183 — Class-Excluded Batch Normalization prior-art audit

## Claim tested

Class-Excluded Batch Normalization (CE-BN) computes the training-time mean and
variance for example `i` from batch examples whose labels differ from `y_i`.
For the first matched screen it is inserted at the embedding head (the smallest
auditable implementation); the existing backbone BN layers are otherwise
unchanged. At inference, the ordinary single-model descriptor path is used.

## Gate 1 provenance

This is specific to the corrected repository lane: In-Shop uses trainable
BatchNorm (`freeze_batch_norm=False`), while CUB freezes it. The project has
already measured that this distinction changes the behavior of EMA teachers and
invalidates a half-averaged BN buffer. CE-BN asks whether the same trainable
batch-statistics channel is also coupling examples of the same identity during
Proxy Anchor training. It is therefore an In-Shop-only hypothesis, not a generic
normalization substitution.

## Gate 2 search

The primary BatchNorm paper defines normalization from current mini-batch
statistics (Ioffe & Szegedy, ICML 2015):
<https://proceedings.mlr.press/v37/ioffe15.pdf>.
Balanced batch structure was shown to affect learned representations and
conditional test behavior (Bjorck et al., 2018):
<https://arxiv.org/abs/1802.07590>.
Conditional BatchNorm changes affine parameters from side information rather
than excluding same-label examples (de Vries et al., 2017):
<https://arxiv.org/abs/1703.06868>.
CrossNorm exchanges statistics between examples/domains rather than computing a
per-example leave-own-class-out statistic (Zhou et al., ICLR 2021):
<https://openreview.net/forum?id=SklDY1StwB>.
Cross-Iteration BN pools statistics across iterations to solve small-batch
estimation, not label-conditioned positive removal (Yao et al., CVPR 2021):
<https://openaccess.thecvf.com/content/CVPR2021/html/Yao_Cross-Iteration_Batch_Normalization_CVPR_2021_paper.html>.

I found no primary paper that computes BN moments separately for each example
after removing all examples of that example's class and then uses the result as
the training normalization in supervised metric learning. CE-BN is therefore
**LIVE-NARROW at Gate 2**, with the distinction being label-excluded statistics,
not class-conditioned affine parameters, statistic exchange, or balanced
batches.
