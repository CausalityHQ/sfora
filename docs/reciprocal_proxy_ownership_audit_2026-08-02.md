# Candidate 184: reciprocal proxy ownership — algebraic Gate-2 death

Checked before implementation or GPU work on 2026-08-02.

## Provenance

On the exact In-Shop epoch-10 Proxy Anchor export, **99.975%** of proxies choose
their labelled empirical centroid, but only **70.303%** of centroids choose their
labelled proxy. Images won by their own proxy have leave-one-out R@1 **0.9656**,
versus **0.8865** when a foreign proxy wins, a **7.91-point** conditional gap.

Candidate 184 proposed leave-one-out batch centroids and bidirectional
cross-entropy on the proxy-by-centroid similarity matrix: each proxy must choose
its own centroid and each centroid must choose its own proxy.

## Adversarial review and adjudication

Claude called the construction live because it couples heterogeneous proxy and
centroid objects. That conclusion does not survive algebra. Proxy Anchor already
uses every proxy as an anchor against labelled positive and negative samples;
replacing its sample set by class centroids is the row-normalised direction.
The column-normalised direction is ordinary Proxy-NCA/class softmax applied to
the centroid. Their sum is the familiar symmetric contrastive cross-entropy on
one similarity matrix. Leave-one-out changes how the class point is estimated,
not the ground-truth matching or supervision relation.

Close primary work also directly calibrates proxies toward empirical class
features: Li et al., *Robust Calibrate Proxy Loss for Deep Metric Learning*
(<https://arxiv.org/abs/2304.09162>) adds a calibration loss constraining proxy
optimisation toward class-feature centres, while Center Contrastive Loss
(<https://arxiv.org/abs/2308.00458>) maintains a class-centre bank and contrasts
queries against those centres.

**DEAD at Gate 2.** A doubly-stochastic or mutual-assignment version would be
balanced assignment/optimal transport, already closed by candidate 175. The
measurement remains useful as an error-risk diagnostic but offers only another
aggregation of existing class-to-proxy labels.
