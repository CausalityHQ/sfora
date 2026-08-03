# Candidate 319: confusion-agreement supervision

## Gate 1

In corrected In-Shop seed-0 training embeddings, I compared two independent
signals for each image's hardest foreign class: nearest foreign image and
nearest foreign proxy. They agreed for **15.69%** of samples. Those samples had
**2.3886%** leave-one-out retrieval error, versus **0.1466%** when the signals
disagreed (overall **0.4984%**). Thus cross-level confusion agreement is a strong
error-localizing observable.

## Gate 2 verdict: dead

An implementable intervention would either upweight the agreeing foreign proxy
as a hard negative or add a graph edge among confusing classes. The former is
hard-negative/proxy weighting (Proxy Anchor, Proxy-ISA, GPW); the latter is
confusion-graph or graph-consistency metric learning (Jin et al., IJCAI 2017;
Deep Consistent Graph Metric Learning, AAAI 2021; ProxyGML, NeurIPS 2020).
Neither changes what supervision exists in a defensible new way. No GPU was
used. The agreement/error statistic is retained as a diagnostic, not a method.
