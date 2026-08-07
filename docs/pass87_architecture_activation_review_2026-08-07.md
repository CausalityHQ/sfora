# Pass 87 — architecture/activation batch review (2026-08-07)

## Gate 1 and Gate 2 outcome

The batch was generated from the measured `-0.04968` unseen-minus-seen
positive-transfer gap and persistent cross-seed error overlap, then checked
against primary literature before any GPU work.

### Potential-field DML — dead at Gate 2

Bhatnagar and Ahuja, *Potential Field Based Deep Metric Learning*, CVPR 2025,
represents every embedding as a decaying attractive/repulsive field and
superposes fields rather than mining tuplets. This is a direct occupied DML
objective in the same CUB/Cars/SOP retrieval setting. No GPU run.

### Disentangled DML — dead at Gate 2

The 2025 AAAI *Deep Disentangled Metric Learning* paper applies
information-bottleneck/class-agnostic regularization to an existing DML
objective. It adds no new supervision referent or deployment representation,
and our diagnostics do not identify a factor that it could recover. No GPU run.

### Spherical Embedding Expansion — dead at Gate 2

The IJCAI 2025 *SEE: Spherical Embedding Expansion for Improving Deep Metric
Learning* augments training with synthetic Max-Mahalanobis centers. Embedding
augmentation/virtual positives are occupied by embedding-expansion and
proxy/center families, and it changes no supervision referent. No GPU run.

## Result

No member clears Gate 2. The DGX remains idle. The next search must target a
new referent or a measurable within-class factor, rather than another scalar
loss, embedding regularizer, or synthetic-center construction.
