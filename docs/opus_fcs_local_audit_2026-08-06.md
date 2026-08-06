# Local protocol audit: FCS (Pass 39)

Date: 2026-08-06. Frozen proposal:
`docs/opus_fcs_proposal_pass39_2026-08-06.md` at commit `65e2420`.
This audit authorizes no implementation or GPU.

## Local verdict

**DEAD at Gate 1; independently recurrent/occupied at Gate 2; no GPU.** The
verified repository packet contains no measurement that a class-private proxy
readout has caused official-query errors through a low-rank governed subspace,
that unseen-identity evidence lies outside the training-proxy span, or that
raising descriptor effective rank improves corrected retrieval. The same
rank-starvation premise was rejected for candidate 371 and Pass 29 DSA: the
`C-1` algebra is a dimensional bound, not a measured causal deficit, and a
99-dimensional continuous space can encode arbitrarily many unseen identities.

The operator also recurs candidate 250, already rejected as frozen
error-correcting class codewords, and sits inside a mature prior-art chain:
ECOC, deep N-ary ECOC with shared networks, problem-dependent and jointly
learned code matrices, and data-informed codeword-to-class assignment. FCS
changes the alphabet to 16 symbols, places the classifiers on disjoint
normalized descriptor slices, and uses a particular alternating assignment.
Those are engineering choices around the same supervision object: decompose a
class label into several class partitions and train one classifier per code
coordinate. The code is still a deterministic recoding of the original class
label and contains no new image-derived supervision.

Three load-bearing mathematical claims fail independently. First, 100 CUB
codeword vectors cannot span `R^512`: the span of any `C x 512` codeword matrix
has rank at most `C`, regardless of balanced symbol use in each block. Second,
the stated test-time margin assumes unseen identities differ in at least eight
code coordinates, but unseen identities have no assigned codeword and never
appear in the separation constraint. Third, the repair-and-partial-revert
procedure does not guarantee a feasible code: reverting only the classes in a
new violating pair can create violations with unreverted new codewords, while
the Gilbert--Varshamov count proves neither coordinate balance nor termination
of the proposed constrained repair.

## Gate 0 and Gate 1: no eligible motivating measurement

The current evidence boundary is
`docs/current_evidence_reliability_audit_321_2026-08-03.md`. Its verified
corrected In-Shop measurements concern official-corpus baseline scores,
foreign-image/foreign-proxy confusion agreement, and stable query errors. It
contains no descriptor-rank, proxy-span, or loss-nullspace intervention.

The closest repository material is explicitly negative as provenance:

- candidate 371 rejected a demanded-rank oracle because centered proxy rank is
  guaranteed algebra and a rank-truncated held-out metric would measure an
  estimator/split, not an intrinsic benchmark dimension;
- Pass 29 DSA records no verified relationship between participation ratio and
  corrected zero-shot retrieval, and separates observed training-identity
  scatter from encoder gain; and
- candidate 250 already classified error-correcting class targets as a change
  of geometry rather than new supervision.

FCS supplies forecasts (`PR` gain, `+1.1` CUB points, and a SOP/In-Shop null),
not artifact-backed repository measurements. Its explanation therefore starts
from an armchair capacity analogy and fails protocol Gate 1 before any GPU.

## Algebra audit

### The claimed full-rank codeword span is impossible on CUB and Cars

Let `V` be the matrix whose `C` rows are the concatenated normalized target
codewords. Irrespective of balance and Hamming separation,

```
rank(V) <= min(C, 512).
```

Thus `rank(V) <= 100` on CUB and `<= 98` on Cars. Proposition 1's conclusion
that `span{phi(codewords)} = R^512` is false. Having every symbol occur in every
block only makes each block's marginal target matrix span its local alphabet;
the same `C` rows couple all blocks, so their concatenation does not multiply
the row rank by `K`.

The weaker claim that a coordinate can receive a gradient is not the claimed
causal result. Softmax gradients sum to zero within a block, vanish when the
prediction equals the smoothed target, and depend on current predictions. It
does not follow that the learned between-identity geometry is full rank, that
new visual evidence was extracted, or that any additional direction transfers.

The baseline comparison is also incomplete. For a normalized descriptor
`z = h/||h||` and proxy-span logit gradient `v`, the pre-normalization gradient
is `(I-zz^T)v/||h||`; it need not lie in the proxy span because of its component
along `z`. More importantly, decoupled weight decay does not selectively
annihilate the proxy-orthogonal direction of a scale-invariant normalized
network. The proposal gives no measurement or derivation that turns the proxy
rank bound into deletion of transferable features.

### The error-correcting guarantee does not transfer to unseen identities

Proposition 3 is conditional on a different identity being separated in at
least `delta_min` code blocks. Constraint (S) establishes that only for the
`C` **training** identities in `Y`. At deployment `Y` is discarded and a test
identity has no codeword. Neither two images of the same unseen identity being
mapped to a common unused tuple nor two different unseen identities occupying
Hamming-separated tuples follows from the training objective. The statement
that `d^K-C` unused tuples are "unseen-identity capacity" is therefore a count
of unused cells, not a generalization theorem or measured mechanism.

### The alternating update is not executable with its stated guarantee

The Gilbert--Varshamov calculation only lower-bounds the size of an
unconstrained q-ary separated code. It does not prove the existence of a code
whose symbol counts are simultaneously floor/ceiling balanced in every
coordinate, nor provide the promised random balanced initializer.

After each independent per-block transportation solution, every destination is
already at its lower or upper allowed count. A one-class repair can only move
from an upper-count symbol to a lower-count symbol. There need not be such a
move that both raises the selected pair's Hamming distance and preserves all
other pairwise distances. Reverting only "the involved classes" to their old
codes does not restore the old globally feasible table because those old codes
can conflict with other classes retained at their new codes. Only reverting
the whole table is feasible by induction, which is not the frozen algorithm.

The claimed update cost is also understated: merely forming all `K*C*d`
affinities on SOP is about `32*11318*16 = 5.79 million` entries before solving
32 constrained flows and separation repair; it is not `2.5 million ops` total.
This is secondary to the scientific deaths but invalidates the quoted cost
derivation.

### The proposed controls do not rescue identification

Control C5 changes `K`, `d`, block dimensionality, Hamming-distance threshold,
and over-subscription simultaneously. Its last CUB point has `d=128>C=100`,
where the balance condition permits unused symbols and Proposition 1's own
premise `C>=d` fails. A monotone curve cannot uniquely identify
over-subscription.

Control C6 is tuned to match **test** effective rank. If test identities are
used to tune the regularizer, it violates the no-test-selection rule; if a
training-identity holdout is used, it no longer guarantees rank matching on the
reported test set. Either reading makes the claimed decisive attribution
non-executable as written.

## Gate 2: recurrence and primary prior art

The nearest mechanism is not merely DREML. It is the intersection of established
ECOC components:

1. Dietterich and Bakiri, *Solving Multiclass Learning Problems via
   Error-Correcting Output Codes* (JAIR 1995), established distributed class
   codewords and one classifier per code coordinate:
   <https://arxiv.org/abs/cs/9501101>.
2. Zhang et al., *Deep N-ary Error Correcting Output Codes* (2020), explicitly
   decomposes classes into N-ary meta-classes and evaluates partial and full
   parameter-sharing neural architectures:
   <https://arxiv.org/abs/2009.10465>.
3. Song, Kang, and Tay, *Error-Correcting Output Codes with Ensemble Diversity
   for Robust Learning in Neural Networks* (AAAI 2021), jointly trains a neural
   ECOC while maximizing row Hamming distance and column diversity:
   <https://arxiv.org/abs/1912.00181>.
4. Zhang et al., *Joint learning of error-correcting output codes and
   dichotomizers from data* (Neural Computing and Applications 2012), learns
   the code matrix and classifiers simultaneously by alternating optimization:
   <https://nlpr.ia.ac.cn/2012papers/gjkw/gk34.pdf>.
5. Evron et al., *The Role of Codeword-to-Class Assignments in Error-Correcting
   Codes* (AISTATS 2023), studies problem-dependent assignments that map
   similar classes to similar codewords and demonstrates that the assignment
   controls generalization:
   <https://proceedings.mlr.press/v206/evron23a.html>.

FCS's disjoint descriptor slices and ordinary-cosine deployment are not shown
to create a new supervision relation; they make the vector of N-ary classifier
outputs serve as the retrieval descriptor. Its per-epoch affinity assignment
is a new optimizer/constraint selection within the occupied learned-ECOC
family, not a new causal training object. The repository had already reached
the same mechanism-level conclusion for candidate 250 before this pass.

## Protocol disposition

Stop before implementation, preregistration, or GPU. The proposal fails the
earliest required gate and independently repeats an occupied mechanism. A cold
review of the frozen proposal remains mandatory under the current search
protocol; it must be frozen before the final ledger entry is reconciled.
