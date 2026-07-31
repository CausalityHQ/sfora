# RFSS — rooted factor spanning supervision

**Status: DEAD at Gate 2 on 2026-07-31; no diagnostic, implementation, or GPU
use.** Gate 1 was recorded before the audit below.

## Repository provenance

The ARCG operating-point graph reproduced two facts simultaneously:

- its response-compatible edge density was 0.3640 and **81.84% of classes had
  multiple connected components**, so it preserved substantial intra-class
  structure;
- replacing Proxy Anchor's positive term with those edges removed persistent
  attraction. The graph loss fell from 2.3593 to 0.0017 immediately after
  activation and R@1 collapsed from 0.8463 toward 0.6637.

RSPG independently failed through the same positive-to-unknown interface. The
measured mechanism is therefore not simply “the selected edges were wrong.” A
disconnected selective graph has no force tying all observations to one identity,
while restoring full proxy attraction returns to indiscriminate point collapse.

## Cross-disciplinary mechanism

RFSS borrows rooted minimum-connectivity design from communication networks.
For each training identity at a frozen warm-up checkpoint:

1. construct a complete within-class graph whose edge cost combines ordinary
   embedding distance with the already measured augmentation-response
   incompatibility;
2. select one minimum spanning tree, which supplies exactly `n_c - 1` edges and
   therefore connects every observation with no redundant all-pairs attraction;
3. retain one persistent root-to-own-proxy positive constraint for the class
   medoid, while retaining all ordinary Proxy Anchor negatives;
4. apply bounded positive margins on the tree edges and root edge, with a single
   preregistered refresh if the diagnostic supports it.

The root prevents ARCG's self-erasure; the tree propagates identity supervision
to every sample; and minimal connectivity avoids forcing all samples directly
onto one proxy. This is neither a disconnected positive gate nor a multi-centre
partition.

## Gate-2 attack required

Before implementation, search primary sources for:

- minimum-spanning-tree or spanning-forest losses in deep metric learning;
- class-wise graph connectivity and topology-preserving metric objectives;
- Easy Positive, nearest-neighbour, anchor-neighbour, and graph-based positive
  mining;
- rooted prototype/graph supervision and manifold embedding;
- spectral connectivity or persistent-homology losses for retrieval.

RFSS is dead if existing work already connects each labelled class by an MST or
equivalent sparse graph plus a prototype root. It is also dead if the rooted
tree reduces mechanistically to established nearest-positive mining with an
ordinary proxy loss, even if the exact implementation is absent.

## Gate-2 result

Xu et al., *Deep Asymmetric Metric Learning via Rich Relationship Mining*
(CVPR 2019), is a direct collision. DAMLRRM explicitly rejects connecting every
possible positive pair and instead builds a minimum-cost spanning tree within
each category, guaranteeing a connected region and a direct or indirect path
between arbitrary positives. It evaluates this tree supervision on CUB,
Cars196, and SOP and attributes improved generalization to relaxing the positive
constraint.

RFSS adds a standard proxy root to DAMLRRM's exact within-class MST operation.
That root addresses the repository's ARCG self-erasure engineering issue, but a
composition of the published tree supervision with ordinary Proxy Anchor is not
a novel method mechanism. Candidate 32 is **DEAD at Gate 2**.
