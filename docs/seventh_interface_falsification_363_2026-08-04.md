# Seventh-interface falsification audit (363)

Date: 2026-08-04. Model: `claude-fable-5`, maximum effort, read-only and
web-disabled. No shell, repository mutation, test-label use, subagent, or GPU
was authorized.

## Decision

**NO LIVE INTERFACE OR METHOD.** Five mathematical counterexamples were
constructed to attack the action-interface closure. None supplies a
Gate-1-supported action outside the occupied catalogue. The pass did identify
a real wording defect: the earlier six-interface list is incomplete under a
faithful representation convention and tautological if arbitrary information
may be hidden inside a “group” value. The corrected closure below states the
missing types explicitly.

## Attachment lemma

Let the executable loss touch parameters through finitely many evaluations

`E(theta) = (f_theta(z_1), ..., f_theta(z_M))`

at parameter-independent inputs, plus direct parameter reads. Suppose an
implicit object `u*(theta)` solves

`F(u, E(theta), C) = 0`,

with nonsingular `dF/du`, and the loss is `ell(E, u*, C)`. Implicit
differentiation gives

`grad_theta L = [dell/dE - (dell/du)(dF/du)^-1(dF/dE)] dE/dtheta`.

Thus a fixed-point, equilibrium, KKT, adjoint, or variational-inequality solver
can mix evaluation-gradient atoms but cannot create a new way for observed
data `C` to attach to sample-indexed quantities. Danskin-style differentiation
gives the same typing for argmin/argmax and optimal-transport plans. This lemma
classifies attachments only; it does not prove two objectives semantically
equivalent or replace mechanism-level prior-art checks.

## Five attempted counterexamples

### A. Equilibrium or implicit supervision

An embedding-derived batch kernel `P` and equilibrium similarity

`Z* = (1-alpha) P + alpha P Z* P^T`

with loss against a label relation appears to add a fixed-point referent. The
label datum remains pairwise, however, and the resolvent is differentiation
machinery. The resulting action is contextual-similarity/graph-propagation
supervision or multiplier-based weighting, occupied by candidates 28, 34, 43,
44, 129, 133, 155, 209, 224, and 225. Its only corrected motivation is the
outcome-defining confusion-margin stratification closed by 319 and 350.

### B. Conservation or current-law supervision

A class-confusion flux `J_ab(theta)` with a divergence penalty acts on the class
partition plus per-class scalar sources. Its gradient is a signed routing of
image-to-foreign-class atoms: hard-negative class weighting/mining, balanced
assignment, or projection. Candidates 36, 104, 155, 175, 213, 223, 229, and
320 close the branches. Candidate 36's corrected sparse-null measurement also
reversed the motivating premise. **Dead at Gates 1 and 2.**

### C. Measure-valued supervision

Optimal or Gromov--Wasserstein transport has three target choices. Fixed
embedding atoms are gauge fixing plus assignment; an abstract metric-measure
target is a relational/Gram target; learned support is a proxy or multi-centre
model. The optimal plan weights cost gradients and does not create a new
attachment. Candidates 54, 122, 175, 185, 202, 351, 355, and the batch-OT
horizon close the actions; corrected class-size and fragmentation measurements
supply no causal target. **Dead at Gates 1 and 2.**

### D. Optimal-control boundaries and stopping times

Training-as-control terminal costs are still built from model evaluations; the
adjoint is trajectory machinery. Model-free hitting times on a pixel graph are
pair relations, while model-dependent times are curriculum, temporal
distillation, optimizer dynamics, selective backpropagation, or smoothed
listwise events. Candidates 58, 127, 129, 130, 134, 136, 199, 200, 214, 228,
273, 282, 314, and 354 close these routes. Exact second-order control also
breaks the cost bound. **Dead at Gates 1 and 2.**

### E. Decorated higher-arity incidence

A genuine symmetric three-tensor can define

`L3 = sum_(i<j<k) W_ijk phi(<e_i,e_j>, <e_j,e_k>, <e_k,e_i>)`.

A dense decorated `W` has cubic degrees of freedom and cannot be faithfully
represented by bounded-dimension unary/binary attachments. Encoding it inside
an arbitrary group value would make the taxonomy vacuous. This is the real
typing correction.

It still supplies no candidate. Executable decorated higher-order supervision
is tuple, listwise, hypergraph, or higher-order Gram learning, occupied by HIST,
RLL, and candidates 52--59, 128, 154, 159--163, 170--173, and 332. TIRD provides
a corrected empirical failure, and no verified current measurement is
irreducibly higher-arity. **Dead at Gates 1 and 2.**

## Amended evidence-bounded closure

Assume:

1. the loss touches parameters through finitely many model evaluations,
   activations or input derivatives at parameter-independent points, plus
   direct parameter reads;
2. those points come from training images through parameter-independent maps or
   are fixed probes;
3. training uses labels only, one deterministic model, no transductive/test
   data, roughly baseline cost, and one normalized descriptor at inference;
4. attachments are represented faithfully, with bounded-dimension values rather
   than arbitrary bit-encoding.

Then model-free observations attach as a finite union of:

- decorated finite-arity relational structures, including per-group values and
  higher-order tensors;
- input-space maps, families, or fixed probes;
- fixed or learned embedding-space nodes;
- parameter-space constants or optimizer-state couplings; and
- learned non-embedding auxiliary objects.

Implicit solution machinery preserves those types. In this repository,
higher-order structures route to tuple/listwise/hypergraph objectives; probes
to synthesis or fixed-probe objectives; parameter attachments to optimizer or
model-copy dynamics; and learned auxiliaries to conditional similarity,
weighting, or meta-learning. All are occupied, and none has a corrected
measurement supporting an unoccupied action.

This is not an impossibility theorem. Exact continuum-valued objectives can
escape the finite-evaluation normal form but violate the cost bound, and their
discretizations re-enter it. Occupancy remains an empirical primary-literature
claim checked mechanism by mechanism. Multiple models, extra annotations,
transduction, external knowledge, or higher cost remain disclosed task changes.

## Consequence

The interface axis is closed under the stated assumptions. A legitimate
reopening now requires either a corrected measurement that instantiates one of
the explicit higher-order/probe/parameter/auxiliary types with an unoccupied
action, or primary-source evidence vacating a load-bearing occupancy ruling.
No GPU or further observation-channel naming round follows from this audit.

