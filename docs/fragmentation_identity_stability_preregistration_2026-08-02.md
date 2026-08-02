# In-Shop fragmentation identity-stability preregistration

Recorded after seed 1 passed the aggregate replication gates but before computing
any cross-seed class overlap and before seed 2 completed.

## Question

Seed 0 and seed 1 both show about 40% fragmented identities and adjusted
fragmented-minus-connected R@1 gaps of +5.875 and +5.966 points. Aggregate
replication does not establish that fragmentation belongs to the data: every run
could fragment a different random 40% of identities. A supervision mechanism
derived from stable appearance factors requires the same identities to recur.

## Locked analysis

Use the final normalized training packs for seeds 0, 1, and 2. For every identity
with at least three images in all packs, compute the same symmetrized within-class
1-NN disconnectedness indicator used by
`scripts/measure_spectral_class_connectivity.py`. Do not tune graph degree,
temperature, or thresholds.

Report:

1. pairwise raw agreement and Cohen's kappa for the binary fragmentation
   indicator for pairs (0,1), (0,2), and (1,2);
2. mean and minimum pairwise kappa;
3. fractions fragmented in zero, one, two, or three seeds;
4. the class-balanced leave-one-out R@1 difference between identities fragmented
   in all three seeds and identities connected in all three, using each class's
   outcome averaged over the three packs.

Labels and class membership must be identical across packs. Refuse missing or
duplicate identities rather than intersecting silently.

## Prediction and falsification

- Prediction: mean pairwise kappa is **> 0.20**, every pairwise kappa is positive,
  and at least **10%** of eligible identities fragment in all three seeds.
- Intrinsic identity stability is falsified if mean kappa is **<= 0.10**, any
  pairwise kappa is negative, or fewer than **5%** fragment in all three.
- Values between those boundaries are weak/inconclusive and do not motivate a
  data-derived class-factor mechanism.

The stable-all versus connected-all R@1 gap is descriptive, not an additional
gate, because conditioning on stability changes class composition.

This diagnostic cannot itself authorize a method. A pass supplies provenance
for an intrinsic within-class factor; any intervention still needs Gate 2 prior
art clearance and a prospective In-Shop screen.
