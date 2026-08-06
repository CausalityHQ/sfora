# Pass 51 local evidence-aware audit: PSPL

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_pspl_proposal_pass51_2026-08-06.md`  
Frozen full-proposal SHA-256: `bba3410c569c3fcc3076441746ec0e620705de52b90c48d648980ce062a0d3d0`  
Durable proposal consultation: `f3bba1083a7b4e5a` (Fable credit failure,
same-job Claude Opus fallback)

This audit was written after freezing the complete proposal and before requesting or
reading an independent review.

## Verdict

**DEAD at Gates 1 and 2, independently; no preregistration, implementation, or
candidate GPU.** Product-Simplex Prototype Lattice supervision (PSPL) splits a
512-dimensional descriptor into 64 orthogonal eight-dimensional groups, assigns
each training class a balanced learned 64-symbol product code over nine fixed
simplex atoms, and trains 64 nine-way classifiers with an equal-group-energy term.

The proposal is an almost exact recurrence of Pass 39 FCS and a stricter geometric
variant of Pass 46 FPC: a deterministic learned N-ary product code of the original
training label. The fixed simplex atoms, online balanced assignment, Hamming repair,
and cosine wrapper change the code geometry and optimizer, but not what supervision
exists. No verified repository measurement attributes corrected zero-shot retrieval
errors to free prototype coordinates or shows that sharing code digits across class
labels creates transferable visual factors.

The central causal claim also does not follow from the construction. An encoder may
first identify a training class privately and then emit that class's injective
64-digit code. Shared output atoms do not force a shared visual computation, and an
unseen identity is never assigned or trained toward one of the purportedly
"reserved" lattice points. The proposal acknowledges the first escape, converting
its advertised structural guarantee into an unmeasured optimization-bias
conjecture.

## Gate 1: no measured prototype-privacy cause

The verified evidence packet contains no intervention showing that a free class
prototype table causes official-query errors, that the rank or parameter count of
that table is a bottleneck, or that recoding class labels into shared coordinates
improves independently selected/final retrieval on unseen identities. Stable error
overlap does not identify prototype privacy. Candidate 371 established that a
`C-1` centered-proxy rank ceiling is dimensional algebra rather than a measurement
of demanded representation rank. Candidate 63 found that even full-rank fitted
compression lost 1.215 CUB points, while rank 128 lost 2.785 points. Passes 29 DSA,
39 FCS, 41 NSRC, 43 AXE, and 46 FPC already supplied missing or adverse evidence for
closely related attempts to reshape, fill, code, or defend directions relative to a
finite training-class table.

The proposal's own mechanism is explicitly not an information mechanism: the full
code is a deterministic function of the label. An injective code retains exactly
`H(c)` bits about the training class, just as a distinct free proxy does. The claimed
benefit is therefore only that SGD may prefer a reusable computation. No repository
artifact measures that preference or connects it to corrected query errors.

## The structural guarantee is false

Balance guarantees only that each *digit value* is shared. It does not constrain the
function used to compute that digit. A sufficiently expressive backbone can implement

```text
image -> private training-class classifier -> stored 64-digit class code.
```

The joint code is deliberately Hamming-separated and therefore class-identifying.
The lower bound `ceil(log_K C)` says how many base-`K` digits are needed to name a
class; it is not a lower bound on shared visual computation. The proposal's statement
that "no single group can separate any pair of classes" is literally false: one
group separates every pair assigned different digits. The true, much weaker fact is
that one group cannot uniquely identify all `C` classes.

The numerical sharing guarantees are also wrong under the frozen rule
`beta=ceil(0.25 C/K)`:

| dataset | `C` | guaranteed minimum classes per digit |
|---|---:|---:|
| CUB | 100 | `floor(100/9)-3 = 8` |
| Cars196 train | 98 | `floor(98/9)-3 = 7`, not at least 8 |
| SOP train | 11,318 | `floor(11318/9)-315 = 942`, not at least 1,000 |
| In-Shop train | 3,997 | `floor(3997/9)-112 = 332`, not at least 1,000 |

Even if all advertised counts were correct, sharing an output parameter among labels
would not show that those labels share the evidence which activates it.

## "Reserved geometry" does not supervise an unseen identity

Lemma 1 correctly gives the cosine of two *constructed* lattice points as a function
of code Hamming distance. The corollary then assumes the missing premise: that a test
identity "lands near a lattice point." No training term assigns a code to an unseen
identity, makes its images agree on an unoccupied code, or gives unused codes semantic
meaning. The codebook and code table are discarded at deployment. Therefore the
`K^G-C` unused tuples are merely unoccupied vectors, not reserved identity slots.

For any function that realizes the training codes, its outputs on disjoint unseen
support remain unconstrained by the code table. This is the same unused-code failure
already recorded for FCS and the same class-private-backbone escape recorded for FPC.
PSPL does not repair the proposal's own universal-approximation counterexample; it
only swaps the training-support target set.

## Executable mathematics and controls fail internally

### The displayed normalization gradient is not the implemented gradient

For `q(u)=u/(||u||+epsilon)`, the exact Jacobian is

```text
J_q(u) = I/(r+epsilon) - u u^T/[r (r+epsilon)^2],  r=||u||,
```

not `(I-q q^T)/r`. At `u=0` the implemented limit is `I/epsilon`, whereas the
displayed expression is undefined. The claimed proof-level starved-group dynamics
therefore do not follow from the frozen implementation.

The same epsilon invalidates Proposition 2's proof as written. For every finite
nonzero `u`, `||q(u)||=r/(r+epsilon)<1`, so `q(u)` is not on the unit sphere and can
never equal a unit simplex vertex. At the claimed point `f=mu(k)`, each block has
`r=1/8` and `q(u)=(1/8)/(1/8+epsilon) s_k`, not `s_k`. The conclusion may have a
nearby limiting directional argument, but the supplied spherical proof does not
establish exact attainment or uniqueness for the implemented objective.

### The assignment update is not executable as claimed

- The pairwise-independence cost contains `log n_{gk,g'k'}` with no smoothing.
  Balanced marginal counts do not prevent a joint count from being zero, so the
  frozen E-step can produce `log(0)`.
- A `C x 576` by `576 x C` Hamming matrix costs about 73.8 billion multiply-adds at
  SOP's 11,318 training classes and materializes roughly 128 million pair entries.
  This is not the stated three-second negligible CPU update, and up to 20 repair
  passes can cost more.
- Moving one member of a violating pair can create another violation. Reverting a
  class to its previous code can violate the current balance and separation state.
  No convergence, feasibility, or authoritative-state rule is supplied.
- Epoch-zero centroids pass through a randomly initialized embedding head. They are
  not learned PSPL factors, and the proposal does not specify a stable warm-start
  state for the first assignment.

### Several frozen controls are impossible or contradictory

- C2 asks for a single `C`-vertex simplex in 512 dimensions. Such a simplex requires
  `C-1` dimensions, so it is possible for CUB/Cars but impossible for SOP
  (`C=11,318`) and In-Shop (`C=3,997`). The forecast table nevertheless reports an
  SOP C2 value of 0.803.
- C9 varies `G` over `{8,16,32,64,128}` at fixed `G(K-1)=512`, which necessarily
  changes `K` from 65 through 5 and contradicts the all-dataset frozen `K=9` claim.
  A fixed `h_min=24` is impossible when `G` is 8 or 16; the registered `h_min=40`
  is also impossible for `G<40`.
- C4's class-private 64-vector target bank is not matched in parameter count or
  target geometry to the frozen 576-atom PSPL bank, so it does not isolate sharing
  without a more precise construction.
- The held-out-identity group-cell top-1 diagnostic has no target for an identity
  excluded from the fit code table. Assigning held-out classes from their labeled
  centroids makes the diagnostic a new transductive validation procedure and tests
  assignment fit, not whether a code learned on other identities transfers.

## Gate 2: exact internal recurrence and occupied primary mechanisms

Pass 39 FCS already:

1. split a 512-dimensional retrieval descriptor into normalized blocks;
2. assigned every training identity a balanced, Hamming-separated multi-symbol code;
3. trained each descriptor block to predict its class-code symbol; and
4. updated class codes from current model affinities.

PSPL changes 32 sixteen-dimensional blocks to 64 eight-dimensional blocks, uses a
nine-vertex regular simplex for each symbol, adds equal block energy, and wraps the
assignment in min-cost flow plus hysteresis. Its supervision relation is unchanged.
Pass 46 FPC independently repeated the same learned product-code label object with
shared learned codebooks and entropic assignment. Both were rejected because an
injective recoding of class labels creates no supervision for unseen labels.

Primary sources occupy every novelty-bearing component:

- Dietterich and Bakiri, *Solving Multiclass Learning Problems via
  Error-Correcting Output Codes* (JAIR 1995), established distributed class
  codewords and one prediction task per coordinate:
  https://arxiv.org/abs/cs/9501101
- Zhang et al., *Deep N-ary Error Correcting Output Codes* (2020), decomposes
  multiclass labels into N-ary subproblems and supplies shared-parameter deep
  architectures:
  https://arxiv.org/abs/2009.10465
- Song, Kang, and Tay, *Error-Correcting Output Codes with Ensemble Diversity for
  Robust Learning in Neural Networks* (AAAI 2021), jointly optimizes neural ECOC
  code separation and task diversity:
  https://doi.org/10.1609/aaai.v35i11.17169
- Evron et al., *The Role of Codeword-to-Class Assignments in Error-Correcting
  Codes* (AISTATS 2023), studies similarity-preserving, problem-dependent
  codeword-to-class assignment and its generalization consequences:
  https://proceedings.mlr.press/v206/evron23a.html
- Yang et al., *Inducing Neural Collapse in Imbalanced Learning* (NeurIPS 2022),
  fixes the final classifier to a simplex ETF during training:
  https://proceedings.neurips.cc/paper_files/paper/2022/hash/f7f5f501282771c96bb3fedcc96bedfe-Abstract.html
- Yu et al., *Product Quantization Network for Fast Image Retrieval* (ECCV 2018),
  and Klein and Wolf, *End-to-End Supervised Product Quantization for Image Search
  and Retrieval* (CVPR 2019), learn product/block codebooks and soft/hard codes with
  supervised end-to-end image retrieval objectives:
  https://openaccess.thecvf.com/content_ECCV_2018/html/Tan_Yu_Product_Quantization_Network_ECCV_2018_paper.html
  and
  https://openaccess.thecvf.com/content_CVPR_2019/html/Klein_End-To-End_Supervised_Product_Quantization_for_Image_Search_and_Retrieval_CVPR_2019_paper.html

No single source must reproduce every wrapper. The claimed conjunction is already an
exact repository recurrence, while its pieces are public fixed-ETF classification,
learned N-ary ECOC/class assignment, and product-code retrieval. Online min-cost flow,
cosine normalization, and equal block energy are optimizer/geometric choices within
that occupied supervision object.

## Protocol and frontier failures

The protocol requires a paired corrected In-Shop screen first after Gates 1--3. PSPL
instead labels In-Shop secondary and declares Cars196 decisive. It supplies no
same-seed, current-digest corrected Proxy Anchor control, no independently selected or
final metric, and no out-of-sample confirmation stage. Its significance arithmetic
combines a forecast standard error with an external published standard error as if
the arms were independent; it is not a paired comparison.

The proposal predicts no general frontier crossing: 0.738 CUB and 0.934 Cars are
below the audited higher-capacity 0.766 and 0.949 observations, while 0.929 In-Shop
is below 0.939/0.9448. In its own matched Lane-A framing it predicts a potentially
decisive Cars result only, but that forecast is not derived from any measured premise.
Even if the number landed, the exact FCS/FPC recurrence would prevent a novelty claim.

## Correct pieces worth preserving

- Lemma 1's cosine--Hamming identity is correct for the constructed lattice.
- Equal block-energy KL is a coherent way to prevent a descriptor from allocating
  all norm to one block.
- Random-code, learned-assignment, class-private-target, global-normalization, and
  code-initialization controls are directionally useful.
- Deployment is legal and clean: one model, one view, fixed 512-dimensional cosine
  descriptor, with all codes discarded.
- The proposal honestly states that its supervision carries no new Shannon
  information and that class-first memorization remains possible.

Those correct pieces do not rescue the candidate. They expose the decisive lesson:
**sharing label-code coordinates does not imply sharing the visual computation that
produces them, and empty geometric slots are not supervision for unseen identities.**

## Disposition

Stop before preregistration, implementation, or GPU. Freeze this local audit, then
obtain the protocol-mandated independent cold review of the exact proposal. Reconcile
that review into the method-search ledger. Any repair that introduces a real
cross-identity visual target rather than a deterministic recoding of the class label
is a new proposal and must restart the blind-generation protocol.
