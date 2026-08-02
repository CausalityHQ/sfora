# Search boundary 229: invariant output-space operators

Date: 2026-08-02. Status: **evidence-bounded stopping boundary, not an
impossibility theorem**. No candidate, diagnostic, implementation, or GPU.

## Corrected representation statement

Let `U` collect every embedding-space vector read by a training operator:
current sample embeddings, augmented views, memory/teacher entries, and learned
proxies. If an update field is equivariant under a simultaneous orthogonal gauge
change `U -> UQ`, then it has the form

```
V_i(U,Y) = sum_j W_ij(Gram(U),Y) U_j.
```

For a differentiable scalar invariant loss, the coefficient matrix is the
appropriate symmetric Gram derivative; nonconservative equivariant fields have
the same span form. This is an instance of the representation results in Villar
et al., *Scalars are universal: Equivariant machine learning, structured like
classical physics* (NeurIPS 2021, arXiv:2106.06610), not a new theorem from this
project.

An elementary local argument explains the scalar-loss case. Perturb `u_i` in a
direction orthogonal to the span of all vectors in `U`. Every off-diagonal Gram
entry is unchanged to first order and the diagonal change is even in the
perturbation, so differentiability forces zero derivative in that direction.
The gradient therefore lies in the span of the input vectors.

The node set must include proxies. The narrower statement over batch embeddings
alone is false for Proxy Anchor when batch size is below dimension: proxy terms
can point outside the batch span. Nonsmooth sorting, optimal transport, rank,
and persistence do not escape at differentiability points; they choose
context-dependent sparse or dense coefficients. Under `SO(d)` rather than
`O(d)`, an orientation/chirality direction exists at rank exactly `d-1`, but a
512-dimensional batch would require at least 511 vectors spanning exactly a
hyperplane. It is measure-zero and operationally irrelevant here.

## What the statement does not prove

This is a conservation law, not a novelty taxonomy. When the complete node set
spans the embedding dimension, the span statement is vacuous and coefficient
representations are non-unique. Even otherwise, the admissible `W` is an
infinite-dimensional function class. A contextual, determinant, transport, or
rank loss can have different dynamics and generalization despite sharing the
same gradient form. General Pair Weighting covers pair-based loss formulations;
it does not prove that every global Gram functional is prior art.

Accordingly, this boundary must never replace the project’s specific
mechanism-level reductions. Examples that remain decisive are candidate 225's
`(a+b)(1-c)` contraction identity, candidate 227's conditional-prediction
information, and candidate 228's meta-gradient inner product. Those name the
actual occupied operation.

## Reopening rule

A future output-space proposal must supply at least one of:

1. a genuinely new gauge-covariant node or observed relation, beyond embeddings,
   augmented views, proxies, teachers, and memories already audited; or
2. a specific Gram/context-dependent coefficient rule whose information and
   dynamics are not already represented in the negative catalogue.

Calling a determinant, hypergraph, simplex, optimal-transport plan, persistence
pair, or tuple aggregate “higher order” is not enough: its exact information
source and coefficient rule must be new.

Legitimate mathematical escape classes are parameter-space constraints,
architecture/data construction, pixel Jacobians, discrete batch construction,
spatial/token supervision, and gauge-breaking coordinate penalties. In this
repository they are empirically or bibliographically occupied: EMA/averaging;
Metrix/Embedding Expansion/parts; Tangent Prop/contractive methods; combinatorial
batch design; DIML/CroCo/regional losses; and Barlow Twins/VICReg-style
decorrelation. Candidate 225 also falsified a transferable linear nuisance
subspace; 226 found no provenance for a nonlinear field; 227 closed
cross-instance completion; and 228 closed class-disjoint meta-updates.

## Verdict

No candidate survives this audit. The correct conclusion is narrower than “no
novel method can exist”: **under the current observables, deployment constraint,
cost bound, measurements, and checked literature, no unoccupied supervision
operator is presently identified.** Further naming-level loss invention is not
justified. Reopening requires new measured information or a concrete operator
that defeats the rule above; it cannot rely solely on a new aggregate of the
existing Gram matrix.

