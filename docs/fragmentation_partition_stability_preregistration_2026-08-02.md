# In-Shop fragmentation partition-stability preregistration

Recorded after the binary identity-stability result was known (mean pairwise
Cohen kappa 0.88483), but before computing any component-count agreement,
component partition overlap, or k=2 graph on any pack.

## Question and provenance

The same identities are disconnected under three optimizer seeds, but that can
happen when a class has stable high spread while a different knife-edge 1-NN
edge chooses arbitrary components each run. A latent-substructure interpretation
requires the *membership of the components*, not just disconnectedness, to recur.
This is a CPU-only provenance diagnostic and cannot authorize a method.

## Locked analysis

Use the exact three digest-bound epoch-10 In-Shop Proxy Anchor packs from the
identity-stability test. Refuse any difference or duplication in `example_ids`
or their labels. Within each identity, construct undirected graphs by
symmetrizing directed cosine k-nearest-neighbour edges.

1. On identities whose k=1 graph is disconnected in all three seeds, compute
   pairwise adjusted Rand index (ARI) between component assignments. ARI is
   invariant to component-label permutations and chance-corrected. Report the
   macro mean for each seed pair, their mean/minimum, and the pooled class-size
   weighted values.
2. Report the fraction of that cohort with exactly the same component count in
   all three seeds.
3. Repeat k=1 ARI descriptively on each pair's broader intersection of identities
   disconnected in those two seeds, to expose inflation from selecting on all
   three seeds.
4. Among the stable-k=1 cohort with at least six images, report per-seed and
   all-three fractions still disconnected after symmetrized k=2. Size six is
   required because two k=2 components must each contain at least three nodes;
   identities of size 3--5 are ineligible rather than counted as connected
   evidence against persistence.

No graph degree beyond k=2, threshold, temperature, or class subset may be tuned.

## Prediction and falsification

Before observing these statistics:

- predicted mean macro pairwise k=1 ARI is **> 0.35**, every pair is **> 0.20**,
  and exact three-seed component-count agreement exceeds **0.60**;
- predicted all-three k=2 persistence among eligible identities is **> 0.20**;
- repeatable component structure is falsified if mean macro ARI is **<= 0.15**,
  any pairwise macro ARI is **<= 0**, component-count agreement is **< 0.40**,
  or all-three k=2 persistence is **< 0.05**;
- values between the prediction and falsification boundaries are inconclusive
  and do not motivate an intervention.

The joint interpretation requires all prediction clauses to pass. A pass still
does not prove semantic factors: fixed label noise, near-duplicate groups and
stable acquisition clusters can produce repeatable partitions. It would justify
only a later independent diagnostic against those explanations. A failure kills
the fragmentation-derived supervision line because binary stability would then
describe class spread or graph brittleness, not repeatable component membership.

## Result (2026-08-02)

Every prediction clause passed. The stable three-seed cohort contained 1,439
identities. Pairwise macro ARIs were **0.83755**, **0.84475**, and **0.84213**
(mean **0.84148**, minimum **0.83755**); class-size-weighted values remained
**0.80859--0.81624**. Restricting only to each pair's disconnected intersection
gave essentially the same **0.83988--0.84678**, so selection on all three seeds
did not create the result. Exact component-count agreement was **0.67269**.

Of 1,246 stable-fragmented identities large enough for two k=2 components,
per-seed k=2 disconnection was **0.70706**, **0.68379**, and **0.68379**, and
**0.56340** remained disconnected at k=2 in all three seeds. The partitions are
therefore neither arbitrary within a stably broad class nor predominantly a
single-edge k=1 accident.

The analyzer SHA-256 was
`b2a9415afd5c0dedbec2ae502345ec75468fee7a49d366f4021b3b55397b7bca` and the
result JSON SHA-256 was
`81a81f1c89ca6e5e757f9621cf0b84c1f15c2dd22e676790afb07b215628903e`.

This advances the provenance claim only to repeatable image membership. It does
not distinguish clothing/viewpoint structure from fixed acquisition series,
background, model identity, label noise, or near duplicates. In-Shop filenames
contain explicit series and view-like tokens, enabling that cheaper audit before
any supervision proposal.
