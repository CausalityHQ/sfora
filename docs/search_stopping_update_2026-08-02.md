# Evidence-bounded stopping update after candidate 231

Date: 2026-08-02. This is not an impossibility theorem. It records why another
naming round over the current measurements is not scientifically warranted.

## Decision

No candidate currently satisfies all of the search protocol's requirements:

1. an exact motivating measurement from this repository;
2. an executable operator not reducible to an occupied mechanism;
3. novelty surviving primary-source prior-art checks;
4. a mechanism plausible on both an instance-identity dataset (In-Shop) and a
   category-identity dataset (CUB or Cars); and
5. approximately 1x training with one normalized 512-dimensional cosine vector
   at inference.

Candidates 224--231 were generated after the previous stopping audit and did
not change that result. They closed causal relation ablation, transferable
nuisance subspaces, local nuisance fields, cross-instance completion,
class-disjoint meta-updates, generic higher-order output losses, six token/data
crosses, and sliced-quantile pooling. Candidate 225 included a preregistered
three-seed CPU diagnostic rather than only a literature ruling.

## Remaining escape classes

| class | strongest evidence | binding failure |
| --- | --- | --- |
| spatial/token supervision | frozen Cars global `0.8306` vs MaxSim `0.8159`; trained CUB region arm `0.6466` vs PA `0.6825` | no positive matched provenance; concrete correspondence, pooling, part, and completion operators are occupied |
| architecture | no identified architectural defect in the repository | a head/backbone variant is not new supervision; train-only auxiliary heads are established |
| data construction | In-Shop acquisition cosine `0.8199` within vs `0.6396` across and 7.18x gap amplification | the best direct intervention is FDA/FACT/APR; acquisition-session structure has no CUB counterpart |
| discrete/stochastic training | randomized relation-ablation proposal | run-long assignment yields one contrast; short windows re-supply withheld gradients; deployed action is selection/weighting |
| parameter-space dynamics | CUB averaging looked positive | Polyak/Izmailov/EMAN prior art, and In-Shop paired effects `+0.19,-0.12,+0.14` (mean `+0.070` pt) failed replication; the line is closed |
| pixel/Jacobian supervision | ARCG response graph density `0.3631`, non-distance structure | affordable finite differences produced only `+0.060` corrected in IPSR; exact Jacobian objectives are established and exceed the cost bound |

## Why the cross-dataset requirement binds

The strongest surviving measurement is In-Shop acquisition structure. An
In-Shop class is one physical garment photographed in sessions; a CUB class is
a species containing different individual birds. A session-nuisance mechanism
therefore does not define the same supervision object on CUB. Conversely, CUB
part/pose semantics lead to the already dense part-discovery and regional-DML
literature.

The natural abstraction that could bridge them was transferable within-class
variation. Candidate 225 tested its linear form with identity-disjoint folds.
Fold-averaged `rho_32` was `0.9312`, `0.9287`, and `0.9345`, below both one and
the locked `1.15` falsifier. A direction learned as within-class variation on
one identity set was not nuisance-selective on another. Candidate 226 found no
positive provenance for escalating that failed global object into a nonlinear
field, and the corresponding local-metric/tangent operators are prior art.

## Reopening conditions

The search should reopen only when at least one genuinely new information source
arrives. High-value examples are:

- a third dataset such as SOP, whose product-instance classes but independent
  seller imagery can distinguish identity structure from In-Shop session
  leakage;
- an explicitly authorized annotation channel such as CUB parts/attributes or
  dataset acquisition metadata, acknowledging that this changes the claim;
- primary-source evidence that vacates a **mechanism-level** prior-art ruling,
  rather than merely showing that a known method was not evaluated on these
  benchmarks; or
- a new causal measurement identifying which pixel component produces the
  acquisition gap, with an outcome that does not route to an already-occupied
  background, colour, or camera-invariance method.

More GPU or a looser cost bound alone does not reopen the search: the expensive
branches also failed novelty or identification. The averaging momentum sweep,
dual-EMA replication, and further averaging datasets remain cancelled because
the underlying effect failed In-Shop replication.

## Honest conclusion

Under the current observables, deployment constraint, cost bound, measured
effects, and checked literature, **no novel and performant candidate is
identified**. This does not prove none can exist. It does establish that a 232nd
proposal generated from the same evidence would be unconstrained speculation,
not the measurement-led search protocol. The negative catalogue and the
regional-provenance correction are the defensible research result at this
boundary.

