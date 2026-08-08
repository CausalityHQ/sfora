# Pass 194 — Confusion-complement supervision (DEAD at Gate 2)

## Gate 1: measured motivation

The corrected CUB error decomposition attributes **51.9%** of failed queries to
between-class centroid overlap. A proposed response was to build a detached
proxy-confusion graph, select the top confused foreign classes for each target,
and add a signed counterfactual target that explicitly separates the true proxy
from only those confused proxies. The intended change was to target supervision,
not merely change the similarity kernel.

This is a valid measurement-conditioned premise, but it is only a premise; no
GPU run was authorized.

## Gate 2: mechanism audit

The candidate is not defensibly distinct from existing operators:

* ProxyGML (Zhu et al., NeurIPS 2020) constructs similarity subgraphs between
  samples and global proxies and uses reverse label propagation to adjust
  neighbour relationships according to ground-truth labels. Its decision point
  is already a confusion-aware proxy relation, rather than a plain all-negative
  proxy loss. Primary source: <https://papers.nips.cc/paper/2020/hash/ce016f59ecc2366a43e1c96a4774d167-Abstract.html>.
* Proxy Anchor already applies a positive target to the true proxy and negative
  pressure to every foreign proxy. Restricting that pressure to the top confused
  foreign proxies and changing its sign/weight is a hard-negative or adaptive
  margin reweighting, not a new supervision object.
* Proxy Synthesis (Gu, Ko, Kim, AAAI 2021) explicitly creates competitive
  synthetic classes between original classes to add confusion-focused proxy
  supervision. Primary source: <https://ojs.aaai.org/index.php/AAAI/article/view/16236>.

Mechanism-level distinction fails: the proposed “counterfactual target” either
reduces algebraically to a sparse weighted negative-proxy term (Proxy Anchor /
hard-negative mining), or to a proxy graph with label-aware confusion edges
(ProxyGML). It does not introduce a new training object or data flow.

**Disposition: DEAD at Gate 2.** No implementation, CPU benchmark, or GPU run.
The useful lesson is that the between-class error measurement motivates focusing
on confusion, but it does not license another confusion-weighting method.
