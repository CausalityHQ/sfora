# Pass 46 local evidence-aware audit: FPC

Date: 2026-08-06 UTC  
Frozen proposal: `docs/opus_fpc_proposal_pass46_2026-08-06.md`  
Frozen full-proposal SHA-256: `4e5fd9fffe5338013c37f2e392a011de4773ac15b3f6a467a2c756bd9d1768d1`  
Exact provider answer SHA-256 (before repository terminal newline): `2d3cb249fbeb82d44c11f18caa11d45bf8356f833c7f15bb367a1633501ad20d`  
Durable proposal consultation: `2cdadbcf836940d9` (Fable credit failure, same-job Opus fallback)

This audit was written after freezing the exact complete provider answer and
before requesting or reading an independent review.

## Verdict

**DEAD at Gates 1 and 2, independently; no preregistration, implementation, or
candidate GPU.** Factorial Proxy Codes (FPC) replaces each free class proxy by
the concatenation of one selected vector from each of eight small shared
codebooks. It learns balanced class-to-code assignments with entropic OT,
anneals them to hard codes, and applies ordinary Proxy Anchor plus head-only
separation and balance terms.

FPC has no eligible measured provenance, repeats Pass 39 FCS and learned
ECOC/product-code supervision, and fails its own central capacity argument.
The proposal additionally predicts no crossing of the audited Lane-A frontier.

## Gate 1: no measured head-capacity cause

The proposal attributes zero-shot error to a free proxy table permitting
class-index memorization and claims that codeword sharing will move supervision
from about 30 images per target to 733. No corrected repository measurement
shows that free-proxy parameter count or proxy-table rank causes official-query
errors, that codeword sharing reveals additional visual evidence, or that
restricting a class table repairs retrieval on unseen identities.

The repository evidence is adverse. Candidate 371 established that the
`C-1` rank ceiling of centered training proxies is dimensional algebra, not a
measurement of demanded representation rank. Passes 29 DSA, 30 NSP, 39 FCS,
41 NSRC, and 43 AXE independently failed variants of the same premise: altering,
erasing, filling, coding, or defending directions relative to a finite
training-label table does not create a new source of visual supervision.
Candidate 63 additionally found that a full-rank fitted compression already
lost 1.215 CUB points, while rank 128 lost 2.785 points.

The motivating sample-count comparison is internally inconsistent. Official
CUB training has 5,864 images over 100 identities, or 58.64 images per class,
not about 30. With 8 codewords per block, 12.5 classes and therefore about 733
images update a codeword; the honest comparison is 733 versus 58.6, not 733
versus 30. More importantly, more images update a *parameter* only because
different labels are forced to share it. That is not evidence that those
images share a transferable visual factor.

## Central mathematics: the claimed bottleneck is not the stated bottleneck

### The factorial proxy set cannot span 512 dimensions

For block `b`, every hard or soft proxy component lies in
`span(U^(b))`, whose rank is at most `C`. The concatenated proxy table therefore
lies in the direct sum of those block spans and has linear rank at most

```
sum_b rank(U^(b)) <= m C.
```

For the frozen CUB/Cars setting `m=8, C=8`, this is at most **64**, not the
claimed full 512-dimensional span. Normalising each convex combination does
not leave its block's linear span. This directly invalidates the advertised
distinction from a low-rank head and makes control C1 (`r=56`) a near-rank
control rather than a categorically different mechanism. The proposal's own
Pass 39 predecessor made the same impossible full-span claim for a finite
class-code table.

### Twenty-four bits do not remove class identity

An injective deterministic code of one of `K` training labels retains exactly
`H(c)=log2 K` bits about a uniform class variable, regardless of whether its
storage alphabet has 24 available bits or 512 real coordinates. On CUB,
`log2 100 = 6.64` bits; a 24-bit code space is substantially overcomplete for
the class index. Comparing `m log2 C` to “512 continuous dimensions” mixes code
capacity with parameter dimension and does not bound class-attributable mutual
information in the claimed way. FPC deliberately preserves an injective map of
every training label, so it does not make class-index memorization infeasible.

The encoder can still implement each class code as an arbitrary training-image
classifier. Sharing a codeword parameter across labels does not make the visual
evidence activating that parameter shared, and Proposition 2 only constrains
the linear head. It does not exclude class-private functions in the shared
backbone or embedding head. The proposal acknowledges this escape, which turns
the purported capacity theorem into an unmeasured optimization-bias conjecture.

### The PA limiting argument does not establish transported factors

Proposition 1 constructs a class-collapsed feasible solution, but this is not a
comparison theorem showing FPC excludes it; an injective factorial code admits
the same class-constant mapping. Its displayed PA bound also suppresses the
actual positive/negative normalizers and invokes a limit in
`alpha(1-rho0-2 delta)` although `alpha=32` is frozen and spherical packing
constrains `rho0`.

Lemma 1 correctly characterises equality in the blockwise Cauchy--Schwarz
bound. It does not show the finite-temperature PA optimum reaches similarity
one, nor that a selected code coordinate corresponds to a reusable appearance
factor. Equal block energy and alignment to a deterministic label code are
compatible with pure label memorization.

## Executability defects

- The displayed OT problem has exact row marginals but only a KL penalty on
  column marginals. Ordinary Sinkhorn iterations solve equality-constrained
  marginals; “three Sinkhorn iterations” does not specify an optimizer for the
  stated KL-relaxed problem.
- Hard-phase repair overrides assignments after the differentiable OT solve.
  The proposal does not specify which state is authoritative, how overrides
  receive gradients, or how Hamming repair preserves the balance objective.
  A runner-up move can create another collision, and no termination proof is
  supplied.
- `O(K^2 m)` repair is about a billion block comparisons at SOP/In-Shop class
  counts, not part of the claimed negligible `K x C` overhead.
- C4 (`m=1, C=K`) is not algebraically the frozen PA recipe while assignment
  logits, OT/annealing/projection, hard repair, zero weight decay, proxy learning
  rate, and structural terms/schedules differ. It is at best a close structured
  proxy control unless those paths are explicitly disabled.
- The extra `K*m*C` assignment logits are class-private trainable parameters.
  Counting only the final hard code's nominal bits while optimizing those
  logits throughout the soft phase understates the mechanism's train-time
  class-specific capacity.

## Gate 2: exact internal recurrence and occupied public mechanism

Pass 39 FCS already split the descriptor into blocks, assigned each training
identity a balanced multi-symbol Hamming-separated code, trained the block
targets, and updated class codes from learned affinities. FPC changes the block
target from a fixed atom to a learned shared vector, the assignment update to
entropic OT, and the loss wrapper to Proxy Anchor. The supervision relation is
unchanged: a deterministic, learned, balanced product code of the original
class label. FCS was already rejected as learned ECOC with no unseen-class
supervision.

Primary sources occupy each claimed building block:

- Song, Kang, and Tay, *Error-Correcting Output Codes with Ensemble Diversity
  for Robust Learning in Neural Networks* (AAAI 2021), jointly designs class
  codewords for row Hamming separation and column-task diversity:
  https://doi.org/10.1609/aaai.v35i11.17169
- Evron et al., *The Role of Codeword-to-Class Assignments in Error-Correcting
  Codes* (AISTATS 2023), shows problem-dependent, similarity-preserving
  codeword-to-class assignment controls generalization:
  https://proceedings.mlr.press/v206/evron23a.html
- Shu and Nakayama, *Compressing Word Embeddings via Deep Compositional Code
  Learning* (2018), learns one discrete index per multiple codebooks and
  composes their shared basis vectors end to end with soft relaxation:
  https://arxiv.org/abs/1711.01068
- Martinez et al., *Permute, Quantize, and Fine-tune* (CVPR 2021), explicitly
  represents weight-matrix subvectors by codes and shared codebooks:
  https://openaccess.thecvf.com/content/CVPR2021/papers/Martinez_Permute_Quantize_and_Fine-Tune_Efficient_Compression_of_Neural_Networks_CVPR_2021_paper.html
- Product-quantization retrieval networks already learn multiple block
  codebooks jointly with image retrieval objectives; for example Jang and Cho,
  *Generalized Product Quantization Network* (CVPR 2020):
  https://openaccess.thecvf.com/content_CVPR_2020/papers/Jang_Generalized_Product_Quantization_Network_for_Semi-Supervised_Image_Retrieval_CVPR_2020_paper.pdf

No one source in this list need match every FPC wrapper. The novelty-bearing
conjunction—balanced learned class codes plus shared multi-codebook composed
vectors—is already the conjunction of learned ECOC and compositional
codebook/weight quantization, and it is an exact internal recurrence of FCS.
Balanced OT and soft-to-hard scheduling change the optimizer, not what
supervision exists.

## Standing-objective failure

The proposal candidly forecasts FPC below its own cited Lane-A frontier on
every primary dataset: CUB `0.712` versus `0.734` and Cars `0.905` versus
`0.927`, with only 8-percent and 7-percent subjective crossing probabilities.
It forecasts In-Shop `0.900`, below both the corrected BN-Inception reference
being replicated here (about `0.915` final over the first three completed
seeds) and the proposal's own cited PA+DADA row. Even exact agreement with its
forecast would not fulfill the standing objective.

## Correct pieces worth preserving

- The proposal reports its non-crossing forecast instead of inflating it.
- Random-code, no-balance, soft-only, head-learning-rate, sampler, capacity,
  and structural-term controls are useful if a future *measured* code-sharing
  hypothesis is investigated.
- It keeps deployment legal: one model, one view, a fixed 512-D descriptor,
  and cosine retrieval.
- It explicitly surfaces the nearest untraced shared-codebook neighbour as a
  novelty risk.

## Mechanism and process lesson

A small or discrete training-label head is not evidence of a small
class-memorizing encoder, and an injective recoding of labels does not add
supervision for unseen labels. Count the linear span of the actual constructed
vectors, compare information quantities in the same units, and require a
repository measurement tying the restricted object to corrected retrieval
error before spending GPU. FPC remains dead even if its optimizer wrapper were
novel, because its cause is unmeasured, its key span theorem is false, and its
own expected result is not a frontier crossing.

## Frozen independent review and reconciliation

The independent cold review ran as durable consultation
`81741bf3725d4caa`: Fable exhausted its credits and the same job completed under
Claude Opus. Its exact final answer, after removing only the preserved Fable
credit-failure banner, is frozen at
`docs/opus_fpc_review_2026-08-06.md`, file SHA-256
`35276f2e66b143c1be5c33f2fa447c742e9e0d1fceb45143b57b3a7d8d59dc2e`
and pre-terminal-newline provider-answer SHA-256
`263a34e4a860ab2ede2deaf338e085202fd02d8967a34042eeb457c5d6554359`.

The reviewer independently returns **DEAD**. It sharpens the rank failure from
linear span at most 64 to discriminative affine/difference span at most
`m(C-1)=56`, exactly the rank of FPC's own C1 control. It independently derives
the same mutual-information defect: both an injective FPC code and a distinct
free proxy carry `H(c)=log2(100)=6.64` bits about the uniform CUB class, while
the advertised 24-bit bound is non-binding. The continuous class-private
assignment logits further contradict “nothing more.”

The cold review finds additional independent failures. Proposition 1 puts the
positive-term sign on the negative term; the Welch bound prevents all 100
proxies from having maximum cosine below `-0.1`, so the displayed PA loss does
not approach zero as claimed. With minimum Hamming distance two, the design
permits proxy cosine 0.75 and a maximum score difference only about 0.354,
while the positive term is already saturated near similarity 0.4. Thus the
hard code can cap separation exactly for the nearest labels rather than
transport factors. C4 is not a PA reduction: at `m=1,h=2`, its injectivity
hinge is unsatisfiable and the decorrelation mean is empty. The reviewer also
confirms the relaxed-column OT program is not ordinary balanced Sinkhorn,
repair lacks termination and balance preservation, and global repair at SOP
scale contradicts the claimed cost.

The admitted untraced neighbour resolves independently and exactly. Michalkiewicz
et al., *Few-Shot Single-View 3-D Object Reconstruction with Compositional
Priors* (ECCV 2020), learns per-class attention over multiple codebooks of
vectors shared across classes and composes them into the class embedding. That
is the proposal's shared-class-codebook object; learned/annealed discrete codes,
ECOC row separation and column independence, and the balanced multi-symbol
assignment additionally recur through public work and Pass 39 FCS. The domain
is not DML, but the proposal's own pre-committed novelty rule said this primary
mechanism match triggers re-adjudication, and none of its nominated residual
distinctions survives.

There is no conflict. The local audit killed FPC from missing provenance,
false span and information claims, internal recurrence, and non-crossing
forecasts. The cold review independently reproduces those failures and adds a
false PA bound, adverse margin geometry, a broken sanity control, and the exact
primary neighbour. Correct PA normalization, parameter/bit arithmetic, legal
deployment, honest frontier arithmetic, sampler/learning-rate controls, and
the proposal's ambiguity ledger remain useful. FPC receives no GPU.
