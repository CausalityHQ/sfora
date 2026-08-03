# Candidate 332: adversarial audit of distinct-part set matching

Date: 2026-08-03. This Gate-1/Gate-2 audit preceded implementation and used no
GPU. A Fable 5 critic independently reviewed the direction; every claim below
was then checked against the repository record or the linked primary source.
One error in the critic's draft was rejected: for one part per image,
`det(A^T B)` is the ordinary dot product, not the two-vector Gram determinant.

## Proposed mechanism

Represent each image by a set of local descriptors, and make similarity reward
several mutually distinct correspondences rather than one MaxSim match. Candidate
forms were a determinant or permanent of the cross-part similarity matrix,
principal-angle/Grassmann similarity between part subspaces, or a DPP-style
log-determinant rewarding non-redundant parts.

## Gate 1: dead on provenance

The only repository number that appeared to motivate this family was the
`0.5775 -> 0.6442` (+6.67-point) recovery from fixed-slot regional comparison to
MaxSim. Audit 231 established that this was only an evaluation repair *inside an
already losing trained regional arm*: `region_pa` averaged 0.6466 against Proxy
Anchor 0.6825, about **-3.6 points**. The clean frozen Cars probe instead favored
one global vector, **0.8306 global versus 0.8159 MaxSim (-1.47 points)**. The
current digest-bound corrected In-Shop packet has no local-token observable.

Thus the available measured evidence contradicts, rather than motivates, adding
part-set matching. This independently satisfies the Gate-1 kill condition.

## Gradient reduction

Let `A,B in R^(d x k)` contain the two images' part vectors and let
`M=A^T B`. For an invertible square cross-Gram, the locally defined
`S=log|det(M)|` has

```
dS = tr(M^-1 dM)
dS/dA = B M^-T
dS/dB = A M^-1.
```

Each part therefore receives a linear combination of the other image's existing
parts, with weights computed from the cross-Gram. A within-set
`log det(A^T A)` diversity term similarly gives inverse-Gram-weighted repulsion.
QR/SVD principal-angle forms and differentiable assignment relaxations change
the coefficient function, not the relation being supervised. A determinant can
also be signed, zero, or ill-conditioned; using its square, absolute value, or a
jittered log determinant changes numerical behavior but not this reduction.

The label channel remains the same: the loss still attracts labelled same-class
image sets and separates labelled different-class sets. Candidate 332 therefore
falls in the protocol's already failed class (b), a similarity/weighting change
rather than a change to what supervision exists.

## Gate 2: dead independently on prior art

The individual forms are established:

- Wolf and Shashua, *Learning over Sets using Kernel Principal Angles* (JMLR
  2003), constructs positive-definite similarities between vector sets from
  principal angles and cross-set inner products.
- Vishwanathan and Smola, *Binet-Cauchy Kernels* (NeurIPS 2004), explicitly uses
  compound matrices/exterior powers and determinants to build kernels.
- Huang et al., *Projection Metric Learning on Grassmann Manifold* (CVPR 2015),
  learns a discriminative metric directly over image-set subspaces.
- Wei et al., *Grassmann Pooling as Compact Homogeneous Bilinear Pooling for
  Fine-Grained Visual Classification* (ECCV 2018), applies Grassmann pooling to
  local convolutional features in fine-grained recognition.
- Zhang, Kjellstrom, and Mandt, *Determinantal Point Processes for Mini-Batch
  Diversification* (UAI 2017), already places DPP diversity in the training
  loop. In this candidate, using it for part diversity is still representation
  regularization or sampling.
- Bo and Sminchisescu, *Efficient Match Kernel between Sets of Features for
  Visual Recognition* (NeurIPS 2009), and compact bilinear pooling establish
  efficient local-set match-kernel and higher-order pooling machinery.

The repository also already tested the application-level neighbors: regional
Proxy Anchor lost about 3.6 points; local MaxSim lost 1.47 points in the clean
frozen Cars probe; and its prior-art catalogue covers DIML, DeepEMD, attention
MIL, bilinear/covariance pooling, DPP/log-det diversity, and subspace metric
learning. Changing the algebra to a determinant does not open a new supervision
primitive.

## Fable's attempted alternatives

The critic generated three alternatives from the verified packet, but none was
new or live:

1. Merge the one near-identical cross-ID In-Shop pair. The preregistered audit
   found only 2 of 129 leave-one-out errors, below its locked 13-error
   materiality threshold; this is dataset repair, not a method.
2. Treat 12 singleton training identities as censored positives. Proxy
   attraction already supplies their positive supervision, and masking undefined
   pair relations reduces to the censored-pair/listwise family recorded as
   candidate 328.
3. Gate positives by epoch-10-to-final relation persistence. Candidate 317
   already killed the same observation at Gate 1: persistence was 64.99%, but
   its error difference was only 0.133 point on one trajectory; the executable
   operator is the occupied positive-eligibility interface that self-erased in
   RSPG.

No alternative survives Gate 1 and Gate 2, so none is assigned a new candidate
number and no GPU is authorized. The repaired PFML Cars reference remains
baseline infrastructure: its independently verified final artifact must supply
a genuinely new measurement before another implementation is justified.

## Primary sources

- Wolf and Shashua (JMLR 2003): https://www.jmlr.org/papers/v4/wolf03a.html
- Vishwanathan and Smola (NeurIPS 2004):
  http://alex.smola.org/papers/2004/VisSmo04b.pdf
- Huang et al. (CVPR 2015):
  https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Huang_Projection_Metric_Learning_2015_CVPR_paper.html
- Wei et al. (ECCV 2018): https://doi.org/10.1007/978-3-030-01219-9_22
- Zhang et al. (UAI 2017): https://arxiv.org/abs/1705.00607
- Bo and Sminchisescu (NeurIPS 2009):
  http://ai.cs.washington.edu/www/media/papers/nips09a.pdf
- Gao et al., *Compact Bilinear Pooling* (CVPR 2016):
  https://doi.org/10.1109/CVPR.2016.41
