# Pass 162 — identity-conditional bottleneck completion (2026-08-08)

## Proposal and provenance

The repaired measurement-conditioned lane used the verified four-seed
unseen-minus-seen nearest-positive gap (`-0.04968`) to propose ICBC: a training-
only decoder receives a global 512-D descriptor from one image and visible
patches from a different same-identity image, and must reconstruct masked target
pixels better than a foreign descriptor.  The intent was to add cross-view
visual information to the deployed descriptor and improve unseen retrieval.

## Adversarial review

The frozen proposal is **DEAD before GPU** for independent reasons:

1. Its `+0.25`-point forecast compared seed-0 (`0.9137`) against `0.9162`,
   while the verified four-seed final baseline is `0.9153889 ± 0.0013196`;
   against that baseline the forecast is only `+0.000811`, below an existing
   seed-1 result.  The forecast therefore does not cross a meaningful screen.
2. The frozen equations leave pixel normalization, aggregation, ranking loss,
   source detachment, energy normalization, and coefficient scale ambiguous;
   the gradient is not uniquely reproducible.
3. Reconstruction decodability does not identify improved cosine geometry.
   A decoder can exploit seen-identity lookup, nuisance/background/camera
   statistics, or an invertible latent transform while same-identity cosines
   remain unchanged.  A ridge CPU probe also does not test the nonlinear decoder
   used during training.
4. Zheng et al., CVPR 2019, already reconstruct same-identity images from a
   different image's appearance code to improve re-identification.  CroCo,
   reconstruction-based disentanglement, VehicleMAE, DiVA, AdvRF, and VAPNet
   occupy the adjacent components.  Masking and foreign-energy ranking are an
   implementation conjunction, not a new supervision primitive.

No CPU or GPU run is authorized.  This is a substantive negative, not a claim
that every cross-image auxiliary objective is impossible.
