# Cross-field candidate batch: candidates 153--155

Date: 2026-08-01. This round deliberately imported mechanisms from immunology,
sheaf theory, and constrained control/optimization instead of starting from a
named DML loss. The shared repository premise is that disconnected same-class
In-Shop modes retrieve **3.534 R@1 points better** after exact class-size
matching, so indiscriminate positive contraction is the wrong intervention.

## 153. Immune negative-selection metric learning

The proposal learned class-specific exclusion detectors and gave images no
own-class attraction. Claude initially marked this live. The adversarial
algebraic pass reversed that verdict.

For normalized embedding `z_i` and detector `d_c`, the proposed loss

`sum_i sum_(c != y_i) softplus(z_i dot d_c - tau)`

is exactly Proxy Anchor's foreign-proxy negative term without its positive
term. It admits non-informative symmetric/orthogonal solutions. Anchoring
detectors to foreign examples makes it one-vs-rest or complementary-label
classification; multiple coverage balls make it classical real-valued
artificial-immune negative selection or multi-proxy classification. Ishida et
al., *Learning from Complementary Labels* (NeurIPS 2017), formalizes learning
from “not this class” labels, and real-valued negative-selection algorithms
already learn detector coverage in continuous spaces. Candidate 153 is **DEAD
AT GATE 2**.

## 154. Sheaf transport between class modes

The proposal assigned vector-space fibers to within-class modes, learned
orthogonal edge transports, penalized cycle holonomy, and gauge-fixed the
result to a single embedding. Neural Sheaf Diffusion (Bodnar et al., NeurIPS
2022) already learns vector spaces and linear restriction maps on graph nodes
and edges to control diffusion and prevent oversmoothing in heterophilic
graphs:
https://papers.neurips.cc/paper_files/paper/2022/hash/75c45fca2aa416ada062b26cc4fb7641-Abstract-Conference.html.
Together with the group-action/cycle-consistency prior recorded for candidate
152, holonomy is a geometric consistency regularizer on an existing graph, not
a new supervision source. Candidate 154 is **DEAD AT GATE 2**.

## 155. Viability-constrained DML updates

The proposal optimized negative separation only inside the tangent cone that
preserves each image's current same-class kNN ordering. This replaces a weighted
loss with projected constrained optimization, but the executable object is a
rank-preservation constraint plus gradient projection. Trust-region constrained
optimization is established, and PCGrad (Yu et al., NeurIPS 2020) explicitly
projects updates away from conflicting gradients:
https://proceedings.neurips.cc/paper/2020/file/3fe78a8acf5fda99de95303940a2420c-Paper.pdf.
The sample-specific kNN inequalities are listwise/rank regularization; changing
their enforcement from a penalty to a projection is an optimizer choice.
Candidate 155 is **DEAD AT GATE 2**.

## Verdict

Cross-field adaptation is allowed and useful, but analogy is not novelty. All
three candidates reduce to established supervision or optimization objects.
No diagnostic, implementation, or GPU run follows.
