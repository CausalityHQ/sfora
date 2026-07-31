# CTNI — class-transmitted negative immunity

**Status: DEAD at Gate 2 on 2026-07-31; no implementation or GPU use.** Gate 1
was recorded before the audit below.

## Repository provenance

RSPG measured a sharp dataset split in target-excluded rival signatures:
same-class agreement retained 64.49% of pairs on CUB but only 8.63% at the
In-Shop epoch-10 operating point. On In-Shop, different images of the same
identity therefore expose substantially different rival identities. The
official training split contains 3,997 identities with only 6.48 images per
identity on average and most classes containing four or five images. Each image
is a sparse observation of the negative confusions relevant to its identity.

RSPG/ARCG showed that using relational structure to remove positive attraction
self-erases, while IPSR showed that ordering same-class images by nuisance
response is not relevance. CTNI moves the measured structure to the negative
side: it expands which negative relations are supervised without weakening the
class-positive label.

## Cross-disciplinary mechanism

The analogy is shared immune memory. If one member of identity class `c`
strongly confuses with negative proxy `r`, that exposure becomes class-level
evidence that every member of `c` should reject `r`, even when `r` is currently
easy for a particular member.

At a frozen warm-up checkpoint:

1. for every training image, record its top target-excluded rival proxies;
2. aggregate a robust class exposure set (for example, rivals observed by at
   least two members, with the exact rule reserved for preregistration);
3. retain ordinary Proxy Anchor unchanged;
4. add a bounded margin term requiring every member of the class to reject the
   transmitted exposure proxies.

Unlike ordinary hard-negative mining, eligibility of `(image, negative proxy)`
is determined by *other images with the same label*, not by that image's own
distance. This creates cross-instance negative supervision: class members share
evidence about observed failure modes.

## Gate-2 attack required

Search primary sources before implementation for:

- class-level, group-level, or cross-instance hard-negative propagation in DML;
- memory-bank and proxy methods that aggregate negatives per class and impose
  them on all class members;
- hard-negative class mining in face recognition and person re-identification;
- collaborative filtering / knowledge-transfer methods that share one sample's
  negative relations with same-class peers;
- confusion-graph and class-adaptive margin methods.

CTNI is dead if prior work already transfers a hard negative discovered from
one labelled sample to other samples of that class. It is also dead if Proxy
Anchor's existing all-proxy negative term makes the added eligibility purely an
ordinary static reweighting with no distinct supervision operation.

## Gate-2 result

The second kill condition holds exactly. Proxy Anchor's negative term sums over
every batch image whose label differs from a proxy. Consequently every proposed
`(image from c, rival proxy r != c)` relation is already supervised as negative,
whether or not `r` appears in that image's own top rivals. Transmitting a peer's
rival can only increase an existing pair's weight or margin; it does not add a
new negative relation.

That remaining mechanism is occupied by hard-class/hard-prototype mining and
class-adaptive margins. AdaptiveFace (Liu et al., CVPR 2019) jointly learns
class-dependent margins and mines hard class prototypes for large-class face
recognition. Confusion-Based Metric Learning explicitly uses confusion
structure to regularize zero-shot retrieval, while the broad hard-negative and
pair-weighting literature covers score-dependent emphasis. CTNI's same-class
aggregation is a mining statistic, not a distinct supervision operator.

The measured In-Shop rival diversity remains informative, but Proxy Anchor has
already encoded the complete negative label relation. Candidate 27 fails prior
art/redundancy before preregistration.

Primary sources checked include:

- H. Liu et al., *AdaptiveFace: Adaptive Margin and Sampling for Face
  Recognition*, CVPR 2019.
- *Confusion-Based Metric Learning for Regularizing Zero-Shot Image Retrieval
  and Clustering*, IEEE TNNLS 2022.
