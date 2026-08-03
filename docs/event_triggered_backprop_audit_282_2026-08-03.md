# Candidate 282: event-triggered metric backpropagation

**Verdict: DEAD at Gate 2; no implementation or GPU.**

## Proposed mechanism and provenance

The audited SOP operation count establishes that image-network computation, rather
than Proxy Anchor's sample--proxy matrix, dominates training. A biology/neuromorphic
analogy suggested propagating an expensive update only when an example's proxy force
or loss crosses a threshold. Low-force examples would take a forward pass but skip
backpropagation; a stale score cache could eventually skip some forward passes too.

## Gate 2 prior-art attack

This is an established workload-skipping family, not a new DML method.

- Brock et al., *FreezeOut: Accelerate Training by Progressively Freezing Layers*
  (2017), progressively remove layers from backpropagation and report wall-clock
  savings: <https://arxiv.org/abs/1706.04983>.
- Jiang et al., *Accelerating Deep Learning by Focusing on the Biggest Losers*
  (2019), introduce Selective-Backprop: use per-example forward loss to decide whether
  to compute a gradient, including a stale-score variant that also skips forward work:
  <https://arxiv.org/abs/1910.00762>.
- Parameter-efficient fine-tuning and LoRA freeze the expensive pretrained body and
  update small adapters. A 2026 AAAI student abstract already explicitly names a
  LoRA/adapter deep-metric-learning application:
  <https://doi.org/10.1609/aaai.v40i48.42270>.

Replacing Selective-Backprop's scalar loss priority with Proxy Anchor force magnitude
does not change the operator: it still prioritizes examples using a quantity derived
from the current objective. It creates no new supervision and has no repository
measurement showing that force is a better compute-allocation signal than loss. The
event-driven biological name therefore supplies neither novelty nor Gate-1 evidence
for the proposed trigger.

