# 2026 horizon audit: Hierarchical Ranking in Hyperbolic Space

Date: 2026-08-04.

## Source and claim

Zhang and Li, *Hierarchical Ranking in Hyperbolic Space: A Novel Approach to
Metric Learning*, **Neural Networks 199 (2026), 108658**, is a reviewed journal
paper, not an unreviewed search result.

Primary records:

- DOI: https://doi.org/10.1016/j.neunet.2026.108658
- PubMed: https://pubmed.ncbi.nlm.nih.gov/41691828/
- author-posted SSRN record: https://ssrn.com/abstract=5131390

The abstract claims Recall@1 improvements of **+2.4 points on CUB-200-2011**
and **+1.6 points on Cars196** over the state of the art selected by the paper.
The method infers an implicit hierarchy from distances among learned class
proxies in a Poincare ball, generates proxy-ranking pseudo-labels, and trains a
hierarchical ranking loss rather than maintaining explicit clusters.

## What can and cannot be concluded

The full comparison table was unavailable through the journal endpoint and the
author-posted PDF returned HTTP 403 during this audit. Therefore the abstract's
deltas cannot be converted honestly into absolute R@1 values, and the audit
cannot yet bind its backbone, descriptor dimension, seed count, uncertainty,
selection rule, or exact In-Shop result. It would be wrong to add 2.4/1.6 to
this repository's preferred horizon numbers: the paper may use a different
baseline set and capacity.

The method also appears to deploy hyperbolic distance, whereas this project's
current target is one normalized descriptor compared by cosine similarity.
Until the full evaluation is inspected, it is a material general-DML horizon
warning but not a replacement for the comparable cosine-family bars of 0.766
CUB, 0.949 Cars196, and 0.939 In-Shop.

## Mechanism consequence

The prior-art consequence does not depend on the inaccessible table. Latent
class hierarchy inferred from proxy geometry, hyperbolic representation, and
multi-level proxy ranking are occupied. A new candidate cannot claim novelty
by deriving a hierarchy from learned proxies and applying a ranked or
level-weighted loss, even if it returns to Euclidean or cosine inference.

This strengthens the existing Gate-2 deaths for hierarchy/proxy-ranking ideas
and raises the risk that a stale-horizon invention would be misreported as
state of the art. It does not authorize an implementation or GPU run.

