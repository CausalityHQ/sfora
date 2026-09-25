# Packed top-k proof model

From this directory, run `lake exe cache get`, then `lake build SforaProofs`.
The `lean-toolchain` file pins Lean to v4.33.0;
`lakefile.toml` pins Mathlib to v4.33.0, and `lake-manifest.json` locks exact
dependency revisions. The proof files contain no holes or custom axioms.

The finite gallery is a set of unique ordinals. `keyFun` ranks rows by higher
score, then lower ordinal. `rank` counts valid rows ahead of a row, and `topK`
contains the first `min(k, gallery size)` rows. Invalid padded rows are absent
from the gallery. The model assumes a fixed linear order on finite scores;
the selector theorems also apply to already-rounded scores when their tie
ordering agrees with this model. Arithmetic error between two scoring methods
belongs in the separate ε premise. NaNs are outside the model.

`Euclidean.lean` proves that, for a fixed query and finite real vectors,
`2·dot(query, gallery) − ‖gallery‖²` equals the query's squared norm minus
squared Euclidean distance. `euclidean_beats_iff` proves identical pairwise
ranking, including the lower-ordinal rule for exact ties; `euclidean_topK_eq`
proves equality of the selected row sets. This models the algebra in
`src/sfora/sop_evaluation.py`'s prefix scorer **over exact real arithmetic**.
For leave-one-out retrieval, the finite gallery set must omit the query's own
ordinal. The proof does not establish upstream UNICOM's different self-exclusion
and tie behavior, nor that float32 matmul, normalization, or stable sorting
matches real arithmetic. In a constructed CPU float32 near-duplicate probe,
expanded scores collapsed 200 rows to 186 distinct values and selected a
different top row from float64 direct distance; matrix shape also changed some
rounded dot products. These are implementation-level ranking risks, not a
measured SOP dataset error rate.

`MemoryBank.lean` proves an exact real-arithmetic staleness bound for the
member-bank training lane: the absolute difference between fresh and stale
dot scores is at most the sum of `|query_coordinate| × |member_drift_coordinate|`.
`dotScore_stale_error_of_coordinate_bounds` replaces actual drift with explicit
per-coordinate upper bounds. A separately established uniform radius can then
serve as `ErrWithin` for the existing margin and Recall@1 theorems. This proof
does not establish that training keeps drift small, that bank-gradient updates
improve retrieval, or that the compiled fp32/CUDA score equals exact real
arithmetic. Those premises require measured or further formal evidence.

`PackedNorm.lean` bounds the error between ideal cosine of the **quantized
codes** and a score formed with two rounded inverse norms. If each stored
scale has total relative error at most `u`, ideal code cosine has magnitude at
most one, and residual score arithmetic has absolute error at most `ρ`, then
the score error is at most `ρ + 2u + u²`. Under an explicit `u = 2⁻¹¹`
premise, the scale-only term is `4097/4194304 ≈ 0.000977`.
That premise is conditional even for freshly packed rows, because the code
computes the norm and reciprocal in f32 before f16 storage; the wire validator
also accepts neighboring f16 inverse norms, which may need a larger `u`.
`packedNorm_errWithin` supplies this bound to the existing robust-margin
top-k theorem. No proof here establishes the actual f16 scale-construction
error, the f32 residual `ρ`, the descriptor-to-code quantization error, or
the required score-margin distribution. `commonQueryScale_topK_eq` proves
that a fixed positive query-scale factor does not change ranking. In exact
score arithmetic (`ρ = 0`), `robust_mem_roundedNormScore` therefore needs
only the gallery-scale radius `u` and a true-winner margin greater than `2u`.
These premises must be established separately before the conditional theorem
yields a numeric recall guarantee for the compiled scorer.

`topK_merge` proves that top `k` of the union of each block's top `k` is the
global top `k`. Blocks may overlap and may contain fewer than `k` rows; they
must each be subsets of the gallery and together cover it. `orderedTopK_merge`
gives equality of the deterministically ordered output;
`orderSelected_pairwise` proves descending score and ascending ordinal order,
and `orderSelected_nodup` proves no repeated ordinals. `iterated_merge_exact`
extends the set result to any finite chain of valid merge levels. `card_topK` proves that
exactly `min(k, gallery size)` rows are returned. `blockRows_cover` and
`topK_merge_blocks` instantiate the abstract theorem for valid ordinals
`0..rows-1` partitioned by a positive width. The proof concerns the abstract
rank selector, not the compiled GPU kernel.

`ErrWithin` assumes ε ≥ 0 and every valid approximate score lies between the
true score minus ε and the true score plus ε. A *robust* true winner has exact score more
than `2ε` above every true outsider. `robust_mem_topK_approx` proves its
survival; `robust_fraction_le_recall` proves that the robust fraction is a
lower bound on top-k recall when the true result is nonempty. If every true
winner has that margin, `topK_approx_eq_of_full_margin` proves equality of the
true and approximate selected sets. The error and margin premises must be
established separately for any real scorer and query cohort. They cannot be
inferred from Lean alone. The RC4 bit-exact check compares two implementations
of the same packed score, so it does not supply an ε bound between packed and
float descriptor scores. No Pet/In-Shop margin census or score-error bound is
in the RC4 receipt; a measured or analytic bound is needed to make this
conditional theorem a numeric quality claim. A uniform ε can be the maximum
of per-row errors for one query, though row-specific bounds could be tighter.

`top1_label_correct_of_positive_margin` gives a label-level guarantee: if one
positive gallery row scores more than `2ε` above every negative under an ideal
scorer, every top-1 row under the observed scorer is positive, even if another
positive becomes the winner. `certified_queries_le_top1_hits` counts such
queries in a finite panel and proves their count is a lower bound on observed
top-1 hits; an empty gallery cannot count as a hit. For a nonempty panel,
`certified_fraction_le_top1_recall` divides both counts by the same number of
queries. The gallery for a leave-one-out query must exclude its own row.
To turn this into a numeric recall bound, measure or prove the pointwise error
between the chosen ideal scorer and the compiled deployed scorer on every
eligible query-gallery pair, then count the certified margins. The existing
inverse-norm bound alone compares scores of quantized codes and does not bound
the original float descriptor against its quantized code.

For one finite SOP TRAIN product-disjoint holdout, the
[SigLIP2 ArcFace score-pair receipt](../docs/evidence/compact_metric/sop-siglip2-substrate-v1/float-packed-margin-certificate-v1.json)
measures the maximum fp32 float-to-packed formula score error against every
nonself gallery row for each of 5,851 queries. The positive-versus-negative
margin is greater than twice that radius for 5,111 queries, so the theorem
certifies 87.35% Recall@1 for these specified finite computed scores; observed
packed Recall@1 is 90.46%. This does not establish an analytic quantization
error bound, compiled CUDA arithmetic identity, or generalization to unseen
queries. The stored per-query radii and margins make each premise auditable.

`card_candidates_le` bounds the distinct merged logical candidates by
`blocks*k`. Under an explicit per-block cap of `k`, `candidate_count_le`
bounds emitted logical records even when blocks overlap, and
`candidate_bytes_le` converts that count to fixed-record payload bytes. The
kernel uses 16 physical lanes per block for logical `k=10`, so
`allocatedCandidateSlots` models the larger padded slot count. At 1,000,003
rows and block width 128 the checked counts are 7,813 blocks, at most 78,130
logical candidates, and 125,008 physical lanes per query. For one f32 score
and one i32 ordinal per lane, the local output payload is 1,000,064 bytes
at batch 1 or 32,002,048 bytes at batch 32. These counts omit merge tensors,
other GPU buffers, and allocator overhead. The `blockCount` expression
assumes positive block width, and the partition and block-specific bounds
explicitly require it. `symbolic_latency_le`
sums explicit assumptions on scoring, selection, merge, transfer, and host
stages. Its merge premise must cover every merge level. For the fused
score-and-select launch, one may set `select=0` and charge the whole launch to
`score`; query copies and per-call allocation belong in an explicit stage
bound as well. The theorem does not establish these premises or a percentile
latency bound.

`denseScoreProducts_le_padded` relates valid-row products to the padded
block-by-dimension count used in the symbolic scoring premise. Neither count
is a measured runtime model of the tiled GPU kernel.

The production path is `rust/sfora-cutile-int8-score/src/topk.rs`: it uses
128-row score blocks, a retained width of 10, a 16-lane local output tile, and
a 2048-wide merge tile. It masks padded rows and selects the lowest ordinal
on equal finite scores. `src/wire.rs` rejects non-finite or non-positive packed
inverse norms. `signed_byte_dot_128_fits_i32` bounds the full 128-term
signed-byte dot product by 2²¹ in magnitude, including `-128` inputs; this
fits the kernel's i32 accumulator. The Rust scalar oracle accumulates into i64
because its public packed-row constructor accepts wider descriptors. The scalar
oracle uses `total_cmp` while the kernel uses float equality for ties; their
ordering agreement also relies on finite scores without signed-zero
differences. The Lean model does not verify the Rust/Cutile source,
CUDA `f32` behavior, memory allocation, synchronization, or the public Python
interface. Short local or merge tiles can carry duplicate ordinals or `-∞`
placeholders in their unused lanes. The set model omits those records.
Production accepts galleries with at least `k=10` rows and repeatedly merges
until one group remains; a source-level refinement proof would need to show
that placeholders never displace ten valid finite winners and that duplicate
masking preserves the modeled set at every level. The production constructor
checks that the 128-row padded gallery has at most `2,147,483,520` rows, the
greatest multiple of 128 below `i32::MAX`; signed kernel ordinals therefore
remain below the `i32::MAX` placeholder. The host output check rejects every
negative, sentinel, or out-of-gallery ordinal before converting to `u32`.
`padded_ordinal_lt_i32_sentinel` and `gallery_rows_lt_i32_sentinel` prove the
arithmetic implication of that padded-row premise in Lean.
These source-level guards do not establish a refinement proof of the kernel.
A hardware execution model would also be needed for wall-time and allocation
guarantees.

Measured p50/p95/p99 latency, throughput, peak GPU allocation, RSS, and
Pet/In-Shop recall remain empirical claims. The authenticated RC4 decision
and limits are in `../docs/rc4_packed_search_decision_table.md` and
`../docs/evidence/packed_topk_rc4_decision_v2.json`. The RC4 candidate failed
its paired public API p99 gate, so these theorems do not promote it over RC3.
