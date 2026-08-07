# Pass 126 — cross-field triage (2026-08-07)

This pass searched outside ordinary DML losses while the Pass-119 DGX
controller was still running.  No GPU run is authorized from this pass.

## Gate-2 outcomes

### Error-correcting class-code supervision — DEAD

The proposal was to assign each training class a fixed error-correcting or
Hadamard codeword, train several binary proxy heads, and deploy their
concatenation as one descriptor.  Error-correcting output codes are a
standard multiclass construction, and multi-head/multi-proxy metric learning
already supplies the same decomposition.  Changing the codebook does not
change the supervised object enough to support a novelty claim.

### Hebbian/anti-Hebbian proxy updates — DEAD

The proposal was to replace backpropagated proxy updates with a Hebbian
attraction plus anti-Hebbian repulsion rule.  FastHebb and supervised Hebbian
learning already cover scalable Hebbian image learning; in DML the proposed
update is an alternative parameterization of pair/proxy weighting, not new
supervision.  No benchmark-specific distinction survived adversarial review.

### Physics potential-field embedding — DEAD

The proposal was to train embeddings as particles under learned attractive
same-class and repulsive different-class forces.  Potential Field Based Deep
Metric Learning (CVPR 2025) explicitly occupies this mechanism, including the
physics motivation and retrieval objective.  Reusing a different potential or
integrator would be a loss swap, not a defensible new method.

## Remaining live path

SRC remains live-narrow because it supervises real-image leave-one-out
coalitions against omission-specific complementary proxy targets.  It still
requires the queued ECTR controls and selection analysis before any GPU run.

## Primary sources checked

- Lagani et al., *FastHebb*, arXiv:2207.03172:
  https://arxiv.org/abs/2207.03172
- Alemanno et al., *Supervised Hebbian Learning*, arXiv:2203.01304:
  https://arxiv.org/abs/2203.01304
- Bhatnagar et al., *Potential Field Based Deep Metric Learning*, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html
- Dietterich and Bakiri, *Solving Multiclass Learning Problems via Error-Correcting Output Codes*, J. AI Research 1995:
  https://www.jair.org/index.php/jair/article/view/10172
