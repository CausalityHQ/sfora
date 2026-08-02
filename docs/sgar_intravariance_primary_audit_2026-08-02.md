# SGAR intra-variance primary-source audit

Date: 2026-08-02. Source inspected: Liu et al., *Deep Metric Learning Assisted
by Intra-variance in a Semi-supervised View of Learning*, arXiv:2304.10941 v1;
version of record EAAI 131 (2024), 107885. The accessible arXiv draft contains
unfinished template text, so table-transcription findings below are explicitly
version-qualified; the journal abstract confirms the same mechanism.

## Mechanism

SGAR synthesizes post-backbone latent vectors
`r_i^j = r_i + j alpha u_i^j`, with isotropic Gaussian unit directions and
`j=1..5`. A gate selects source positives near a generation boundary. Its
ranking loss orders synthetic points by the construction index `j` and adds an
anchor term relating synthetics to real positives.

No real-image intra-class relation supplies the ordinal label. The order is
true by construction and the Gaussian direction is not conditioned on observed
appearance structure. Mechanistically this is the synthesis-intensity ranking
of Fu et al.'s SR/SSR line with a different gate and all-pairs ordering. It is a
radial smoothness regularizer, not non-generative expanded supervision.

## Evidence audit

The method uses ImageNet-pretrained GoogleNet or BN-Inception, 512 dimensions,
standard class-disjoint splits and single-view evaluation. Synthesis occurs
after the backbone and is plausibly about the +6.6% overhead measured for the
nearly identical SSR construction, though SGAR does not report cost.

Reported BN-Inception/512 changes over quoted Proxy Anchor baselines are:

- CUB 68.4 -> 68.8;
- Cars196 86.1 -> 86.6;
- SOP 78.6 -> 79.4 as printed;
- In-Shop 91.6 -> 91.9.

There are no seeds, error bars, confidence intervals, or stated checkpoint
estimand. Six hyperparameters are swept using CUB test R@1. The +0.3--0.5 point
gains are the same scale as this repository's measured RS@k selection bonus of
**+0.427 point** and below CUB's measured 0.57-point seed standard deviation.

The accessible arXiv table has multiple transcription defects. Its SOP PA row
uses 78.6 where the Proxy Anchor source reports 79.1, reducing the claimed SOP
delta from +0.8 to +0.3. It copies an impossible SSR CUB sequence with R@8 below
R@4 (80.3 instead of source 90.3), and compares against SSR's weak Triplet
variant while omitting SSR's stronger Margin variant. Neither claimed change
over SSR—the generation gate and revised ranking loss—is ablated.

## Search consequence

SGAR does not close BLENDER's remaining non-generative, data-only direction:
every added ranking label is attached to a synthetic vector. It instead
strengthens the existing SR/SSR synthesis-ranking closure used for candidates 5
and 22. Its own Eq. 2 gestures toward ordering real positive images, but the
implementation replaces that with constructed synthetic order; ARCG/IPSR are
the repository's direct real-image tests and already failed at Gate 4.

Primary sources:

- https://arxiv.org/abs/2304.10941
- https://www.sciencedirect.com/science/article/pii/S0952197624000435
