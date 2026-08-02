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
