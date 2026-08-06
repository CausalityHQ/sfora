# Cold Opus review — IRH (Pass 63)

**Decision: DEAD at Gates 1–2; no GPU.**

The decisive equation is that `ell_i` and `s_i` are detached state, so IRH’s
gradient is only `-sum_i s_i grad r_i`: a per-example signed Pi-model/VAT
consistency term. The divergence-free transport, staircase, controller, and
implicit-threshold framing are outside autograd; the claimed interaction is a
scalar weight on an occupied consistency loss.

The transport itself is published: Mots’oehli et al. (ICCV 2025 CV4DC,
arXiv:2507.05536) use `v=(d_y psi,-d_x psi)` from a smooth random field with a
pixel displacement scale for zero-divergence swirl distortions. Saliency-guided
warping and the evidence mixture are adjacent Instance-Warp and SaliencyMix.
Persistent per-example radius staircases are occupied by IAAT
(arXiv:1910.08051), and signed consistency variants by Equivariant Contrastive
Learning (ICLR 2022).

The state is also statistically unidentifiable: with only 25% of a batch
probed, each image receives roughly 15–25 trials over training, giving a
stationary staircase noise floor around .25–.4 nats. Thus F0’s threshold-spread
test passes from estimator noise even when true spread is zero, and the negative
half becomes near-random inconsistency maximization. Finally, area preservation
does not preserve information: the stream-function flow has condition number
that grows with displacement, bilinear resampling destroys detail, and the
controller changes the ruler while states persist. The ECT area confound is
replaced by a resampling confound; response monotonicity and threshold roots are
not guaranteed.

Primary sources: Mots’oehli et al. 2025; IAAT 2019; Karmali et al. 2016
staircase efficiency; Equivariant Contrastive Learning 2022; Instance-Warp
2024; SaliencyMix.
