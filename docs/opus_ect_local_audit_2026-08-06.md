# ECT local Gate 1–2 audit (Pass 62)

## Gate 1 — provenance

ECT is motivated by a measured CUB failure decomposition: 48.1% of failed
queries had a nearer own-class centroid but a wrong nearest image (local
within-class evidence failure), while 51.9% had between-class centroid overlap.
The proposal explicitly targets both failure modes through redundant
part-level evidence rather than a scalar contraction trade-off. This is a
repository measurement, not an armchair analogy, so Gate 1 is provisionally
passed.

## Gate 2 — prior-art hazards

The components overlap heavily with mixup/CutMix, SnapMix (CAM-weighted
proportional targets), Attentive CutMix and saliency patch methods, erasure and
attention-diffusion methods, and metric-learning mixup/i-Mix/Metrix. The
candidate distinction is narrower: masks are selected from the *anchor’s own
current evidence* and composites use a two-sided, winner-take-all target with a
must-switch contradiction plus an adaptive minority-repulsion guard. A cold
review must determine whether that target/mask combination is already present,
whether the loss is simply SnapMix or hard CutMix under another name, and
whether the online evidence map is a valid causal signal rather than a saliency
heuristic.

The specification also has engineering risks: the frozen pooled descriptor must
match the control (norm-weighted pooling versus standard GeM); the 1.5x compute
and distinct-image exposure must be matched; composite masks must not leak
same-class information; and `L_rep` must be skipped exactly for same-class
pairs. The proposed `A2` (soft target), `A3` (erase-only), and `A4` (random mask)
controls are necessary to distinguish the claimed mechanism.

**Decision:** Gate 1 passes provisionally; Gate 2 remains pending the mandatory
cold primary-art review. No GPU is authorized yet.
