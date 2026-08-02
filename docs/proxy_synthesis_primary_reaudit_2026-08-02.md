# Proxy Synthesis primary-source re-audit

Date: 2026-08-02. This challenged the prior used to close synthetic/reserved
unknown-class supervision after the cross-field batch.

## Verdict

**Closure confirmed.** Gu, Ko, and Kim, *Proxy Synthesis: Learning with
Synthetic Classes for Deep Metric Learning* (AAAI 2021), does not merely mix
samples around observed proxies. It generates both synthetic embeddings and
synthetic proxies that operate as synthetic classes inside proxy-based losses,
explicitly to mimic unseen classes and improve class-disjoint generalization.

Therefore candidate 208's reserved Good--Turing proxies and candidate 210's
virtual class for a binding-error composite remain mechanism-level
instantiations of occupied synthetic-class supervision. Different rules for
estimating how many virtual classes to add or how to construct their examples
change the estimator/content, not the primitive.

Candidate 208 is independently dead on repository arithmetic: its estimated
unseen mass was only 0.18--0.30%, at or below the already insufficient 0.253%
ensemble all-miss rescue event.

Primary source: https://arxiv.org/abs/2103.15454
