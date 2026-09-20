# Shrinkage Within-Class Whitening for Compact Similarity

## Status

This is claim-ineligible development evidence. Cars, CUB, Pet, and In-Shop had
already been observed by the project before this candidate family was defined.
The measurements establish a mechanism and a prospective selection rule; they
are not a fresh-dataset or publication claim.

The experiment compares five 128-dimensional signed-int8 affine encoders over
frozen teacher embeddings:

- ordinary centered PCA;
- the existing PCA-initialized learned projection;
- Ledoit-Wolf within-class whitening followed by PCA in whitened space;
- the same whitening initialization followed by the unchanged learned recipe;
- a Fisher/LDA hybrid whose unused dimensions are filled by whitened PCA.

Every deployed arm remains one 768-to-128 affine transform, unit normalization,
and one signed byte per output dimension. Whitening is a fitting procedure, not
an added serving component.

## Evaluation results

All values below are measured, not projected. MAP@R and Recall@1 use the
repository's deterministic cosine scorer over restored signed-int8 codes.

| Dataset and split | PCA MAP@R / R@1 | Existing learned | Whitening-only | Whitening-init learned | LDA hybrid |
|---|---:|---:|---:|---:|---:|
| Stanford Cars, 98 unseen evaluation classes | 0.767869 / 0.970852 | 0.825656 / 0.972574 | 0.846998 / 0.972205 | **0.848032 / 0.972820** | 0.846578 / 0.973189 |
| CUB-200, 100 unseen evaluation classes | **0.719066 / 0.912146** | 0.718743 / 0.905495 | 0.703364 / 0.905145 | 0.702875 / 0.904795 | 0.701715 / 0.899895 |
| Oxford-IIIT Pet, 18 unseen evaluation classes | 0.847639 / 0.962275 | 0.862759 / **0.969595** | 0.870446 / 0.963964 | **0.871733** / 0.967342 | 0.870251 / 0.963964 |
| In-Shop official query/gallery, rank-finished L/14 teacher | 0.777572 / 0.946125 | **0.800020 / 0.954283** | 0.775164 / 0.933887 | 0.793342 / 0.944437 | 0.773982 / 0.935082 |

Cars is the decisive positive mechanism result. Whitening initialization improves
MAP@R by 0.022377 over the existing learned head, by 0.021943 over the float-768
teacher (0.826089), and by 0.030402 over the previously measured equal-byte OPQ
control (0.817630). The float-to-int8 loss for the selected Cars arm is only
0.000110 MAP@R.

CUB and In-Shop are equally decisive negative controls: whitening must not be a
universal default. Pet improves MAP@R but reduces Recall@1, and the paired
class-cluster bootstrap interval for its learned-minus-LDA MAP delta crosses
zero. That makes Pet an abstention case rather than evidence for unconditional
selection.

## Fit-only evidence

The following row-weighted values use only deterministic class folds inside the
fit split. Evaluation labels are not used in these calculations.

| Dataset | PCA | Existing learned | Whitening-only | Whitening-init learned | LDA hybrid | Highest fold MAP@R |
|---|---:|---:|---:|---:|---:|---|
| Cars | 0.772612 | 0.805720 | 0.818385 | **0.819107** | 0.818386 | whitening-init |
| CUB | **0.865754** | 0.858946 | 0.852255 | 0.851080 | 0.852058 | PCA |
| Pet | 0.891576 | 0.902806 | **0.904806** | 0.904202 | 0.904741 | whitening-only |
| In-Shop | 0.856372 | **0.871374** | 0.849577 | not measured in folds | not measured in folds | existing learned |

The direction agrees with the best evaluation family on Cars, CUB, and In-Shop.
Pet's whitening MAP advantage over the existing learned head is only 0.001999
and its fit-only Recall@1 is lower by 0.002118, correctly identifying it as an
unstable choice under a conservative guard.

This agreement is retrospective development evidence, not prospective selector
validation. In particular, the previously frozen production rule required a
nonnegative Recall@1 gain and therefore would reject the Cars whitening arm on
its fit folds. No document should describe the new family as preregistered or
the current production selector as having selected it.

## Mechanism

Whitening-only slightly exceeds the Fisher/LDA hybrid on Cars, Pet, and CUB.
Therefore the Cars gain is not explained by class-mean discriminant directions.
It comes from shrinkage-regularized within-class conditioning before the
128-dimensional projection. Starting the unchanged ranking learner from that
geometry adds another 0.001034 MAP@R on Cars and 0.001287 on Pet, but loses on
CUB and In-Shop.

The practical diagnosis is that the existing learner's PCA initialization and
strong anchor can constrain optimization to a poor geometry on some domains.
Within-class whitening supplies a better initial metric when fit-only evidence
supports it. This is classical WCCN/Fisher-family machinery used as a robust
control and initializer, not a claim of inventing LDA.

## Numerical and protocol audit

- Cars uses 98 fit classes and 98 disjoint evaluation classes. CUB uses two
  disjoint named sets of 100 classes; Pet uses 19 fit and 18 disjoint named
  classes. In-Shop train identities are disjoint from query/gallery identities.
- An independent `scipy.linalg.eigh(B, W)` audit on Cars has generalized
  relative residual 3.024e-15, W-orthogonality residual 2.204e-14, and minimum
  principal cosine 0.999999999999996 against the whitening construction.
- The retained runner copies caller inputs before normalization, rejects fewer
  than 128 input dimensions, checks finite nonzero rows, derives numerical
  between-class rank, and uses a query-weighted paired class-cluster bootstrap.
- Cars, CUB, and Pet receipts bind the frozen feature archive, candidate module,
  learned checkpoint, and historical selector receipt by SHA-256. In-Shop adds
  the exact train/query/gallery archive and official selector receipt.
- Exact executed drivers are retained beside the receipts. The four panel runs
  completed in 33.55, 11.42, 20.50, and 34.61 seconds respectively on the DGX
  Spark host; these are experiment wall times, not serving latency benchmarks.

Independent Fable 5.1 and GPT-6 Astra reviews agree that the Cars result is
mathematically credible and that unconditional whitening or a generic selector
claim is not yet supported. Their concrete corrections—bootstrap estimand,
input mutation, insufficient dimensions, numerical rank, inclusion of PCA, and
the In-Shop large-class control—are incorporated in the retained evidence.

## Prospective decision boundary

The smallest safe next rule is a three-family selector:

1. choose the incumbent between PCA and PCA-initialized learning using the
   existing fit-only rule;
2. promote whitening-initialized learning only when its fit-fold MAP@R exceeds
   the incumbent by at least 0.003 and its Recall@1 is no more than 0.003 lower;
3. otherwise retain the incumbent.

Those thresholds are post-hoc on the current development panel. They must be
frozen before one untouched-domain evaluation and must not be presented as
validated until that evaluation completes. No tuning is permitted on the
untouched result.

## Performance decision

No custom CUDA, cuTile, or CUDA-Oxide kernel is justified by this evidence. The
accepted representation still serves through the same small affine projection,
normalization, and int8 conversion. Existing GB10 evidence places batch-one
encoding around 20 microseconds with graph copy/replay and 34 microseconds with
the compiled path; these fitting experiments finish in seconds. Kernel work is
deferred until an end-to-end profile shows a material kernel bottleneck after
the representation and selector are accepted.

## Evidence

Exact receipts and drivers are under
`docs/evidence/compact_metric/shrinkage-whitening-panel-v1/`.

- Cars receipt SHA-256: `d2bdb71f9b183131d89ba885dd03e22090dc8bf46471b218acc0950ee6059628`
- CUB receipt SHA-256: `edb23d4cfb81787f2facd6e35074851a1b438e9236d9ff04cade23b1c851f77e`
- Pet receipt SHA-256: `648f5c6131e9b396b49c0ca9aac46e5901256bb2b88b39560e99047964d46036`
- In-Shop receipt SHA-256: `d80ff64abe45287e5e9fd1c1148af60499267f480f27b9aaa14bc858c28f57e5`
- Class-disjoint panel driver SHA-256: `245967014642b76aff74625ac5b40de48690d7dbe4d51bb6084fa559f177cea2`
- In-Shop driver SHA-256: `34a6e15fa3188f415bf70559277563db4bb2828115451886287f687ecd7c174d`
