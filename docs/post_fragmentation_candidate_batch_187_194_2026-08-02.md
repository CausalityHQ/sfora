# Post-fragmentation candidate batch: 187--194

Date: 2026-08-02. Generated after candidate 186 died at Gate 2 and the search
boundary update recorded no defensible arm. No candidate in this batch
authorises GPU work; none reached preregistration.

## Round constraints

Information budget: official training pixels and class labels only. No external
model, no dataset metadata (including In-Shop filename acquisition tokens), no
text, no generator. Roughly 1x training cost. One deterministic 512-D cosine
embedding at inference.

Required class: the mechanism must change **what cross-instance supervision
exists**, not how an existing relation is scored, weighted, selected, or
optimised. Explicitly excluded operator families: loss reweighting, pair/sample
miners, proxies and subcentres, clustering-derived pseudo-labels, covariance and
diversity regularisers, gradient surgery/projection, substituted similarity
functions, augmentation auxiliaries and response signatures, contextual kNN,
synthesis, optimal transport, ensembles.

## A useful taxonomy, not a completeness theorem

Under this budget the model is a **separable** encoder: `s(A,B) = cos(f(A), f(B))`
depends on `A` and `B` only through their own pixels. Any cross-instance
supervision available to the objective is therefore a functional of training
observables, the label partition `P`, and possibly the current Gram matrix `S`.
Five common operations on `P` cover the executable proposals in this batch:

| operation on `P` | resulting supervision | status |
|---|---|---|
| identity | all same-class pairs positive | ordinary DML (Proxy Anchor/HIST) |
| refinement | subcentres, mode pseudo-identities | excluded; candidates 185, 40; SoftTriple |
| coarsening | class hierarchy, graded negatives | occupied; HIER, Beyond Binary Supervision |
| weighting | soft/graded relations | excluded loss reweighting |
| subsetting | eligible-positive graphs | excluded miners; candidates 5, 18, 19, 68, 133 |

This is a routing taxonomy, **not a proof that partitions admit only five
functionals**. Higher-order relations can be written, but they require an
observed correspondence or incidence beyond class co-membership; otherwise the
extra tuple positions are arbitrary. Each candidate below is adjudicated on its
own mechanism. The group-level conclusion is evidence-bounded and must not be
quoted as an impossibility theorem.

---

## 187. Directed retrieval-obligation supervision

**Provenance.** `docs/neighborhood_error_audit.md`, exact epoch-10 In-Shop pack
`reports/emb/inshop_pa_epoch10_operating.train.npz`, 25,882 images: leave-one-out
train R@1 **0.938181**; mutual top-1 rate **0.567035**; R@1 **0.969065** among
mutual top-1 cases versus **0.897733** otherwise; the 1.03% of queries lacking
top-10 reciprocity retrieve at **0.406015**. Supervision is symmetric but the
measured retrieval obligation plainly is not.

**Mechanism.** Replace the symmetric same-class relation with a *directed* duty:
for an ordered same-class pair `(A,B)`, `B` must sit inside `A`'s positive
neighbourhood, while `A` carries no obligation toward `B`.

**Algebraic attack.** The duty is `s(A,B) > s(A,C)` for foreign `C`; its converse
is `s(A,B) > s(B,C')`. Because `s` is symmetric, the *union over all ordered
same-class pairs* of these constraint sets is literally the per-anchor constraint
set already optimised by NCA-family, Ranked List Loss (Wang et al., CVPR 2019),
and Smooth-AP (Brown et al., ECCV 2020) — candidate 148 killed exactly this.
Asymmetry survives only if the direction set is a strict, criterion-chosen subset,
which is pair mining (excluded). And no asymmetry can be *deployed*: a symmetric
cosine kernel cannot represent a directed relation at inference, so any retained
direction is a training-time coefficient, i.e. loss reweighting (excluded).
**DEAD — subsetting/weighting row; vacuous at inference.**

## 188. Population-ablation invariance supervision

**Provenance.** `docs/fragmentation_confounding_preregistration_2026-08-02.md`:
the +5.875-point adjusted fragmentation contrast is defined only relative to a
class population. `docs/proxy_geometry_audit.md`: 99.975% of proxies own their
centroid but only **70.303%** of centroids own their proxy, and only **65.308%**
of images score their labelled proxy highest — relations are demonstrably
population-conditional.

**Mechanism.** Supervise that a same-class relation is invariant to which *other*
identities are present, making population-exchangeability itself the label.

**Algebraic attack.** At fixed parameters `s(A,B)` does not depend on the training
population at all — the encoder is separable, so the constraint is identically
satisfied and supervises nothing. The only population-dependent object is the
*solution*, so the proposal is a statement about the training procedure. Realising
it needs either several models trained on subsets (ensembles, excluded, and >1x),
or a min-max over subpopulations, which is distributionally robust optimisation:
Sagawa et al., *Distributionally Robust Neural Networks* (ICLR 2020), and Qi et
al., *A Simple and Effective Framework for Pairwise Deep Metric Learning*
(ECCV 2020), which already optimises DML pair weights over an uncertainty set.
**DEAD — vacuous as stated, occupied when made non-vacuous.**

## 189. Model-free pixel-domain eligibility graph

**Provenance.** `docs/results.md` §regional: fixed-slot region cosine **0.5775**
versus position-tolerant MaxSim **0.6442**, a **+6.67-point** recovery, showing
that raw position-tolerant appearance agreement carries real information.
`docs/local_chord_expansion_candidate.md`: CUB top-5 positive-neighbour Jaccard
**0.411** against **0.045** by chance (**9.06x**).

**Mechanism.** Compute a once-only, model-free pixel-domain agreement between
same-class images and let it decide which same-class pairs carry positive
supervision — no learned encoder anywhere in the estimator.

**Algebraic attack.** This is discrete same-class eligibility with a new edge
estimator: the subsetting row, and an excluded miner. It is directly occupied.
Xu et al., *Deep Asymmetric Metric
Learning via Rich Relationship Mining* (DAMLRRM, CVPR 2019), reject the
all-positive constraint using a per-class visual-distance spanning tree on CUB,
Cars196 and SOP. Kim et al., *Deep Metric Learning Beyond Binary Supervision*
(CVPR 2019), occupies deriving continuous relations from a non-label source.
Candidates 4, 5, 6, 68 and 133 already died here. **DEAD.**

## 190. Cross-class analogical quadruple supervision

**Provenance.** Five aligned final CUB HERD packs: class-pair means explain
**52.57--58.90%** of cross-class similarity variance and reproduce at Pearson
**0.9037**; the image-by-image interaction retains **4.75%** of variance with
cross-seed Pearson **0.5710** (verdict §§51--52).

**Mechanism.** Supervise `f(A) - f(B) ~ f(C) - f(D)` where `(A,B)` are same-class
in `c1` and `(C,D)` same-class in `c2`: a four-real-image analogy relation that no
pairwise label expresses.

**Algebraic attack.** The label partition supplies no correspondence saying that
the transformation from `A` to `B` is the same latent factor as the transformation
from `C` to `D`; arbitrary within-class pairing therefore invents rather than
observes the proposed quadruple label. Expanding
`||(f(A)-f(B)) - (f(C)-f(D))||^2` on normalised embeddings gives a fixed affine
function of six Gram entries. That algebra alone does not prove non-novelty, but
it shows that the executable arm is a contrast over existing similarities, not
an independently measured four-way fact. The vector form is adjacent prior art:
TraVeLGAN (Amodio and Krishnaswamy, CVPR 2019) preserves equal transformation
vectors across images, and Difference Vector Equalization (AAAI 2026) equalises
embedding difference vectors across samples — candidate 30. Those different
tasks do not alone occupy supervised DML, but they eliminate novelty of the
vector-equality operator. Empirically, TIRD
(candidate 52) trained the closed 2x2 form of this contrast and lost **7.237**
raw In-Shop points. **DEAD at Gate 1 for absent observed correspondence, with an
occupied operator and a negative in-repo analogue.**

## 191. Observed mode-count preservation as an explicit label

**Provenance.** `docs/fragmentation_replication_preregistration_2026-08-02.md`:
**40.33%** of eligible In-Shop within-class 1-NN graphs are disconnected at the
epoch-10 operating point. Adjusted disconnected-minus-connected class-balanced
leave-one-out R@1 is **+5.875 points** at **60.43%** coverage.

**Mechanism.** Give each class an integer target equal to its observed number of
within-class components and supervise the embedding to retain it.

**Algebraic attack.** The component count is read off the current embedding, so
the target is endogenous — the same defect that retracted candidates 106 and 114:
the model is rewarded for a partition it creates. Making it exogenous requires a
frozen pixel-space estimator, which is candidate 189. As an objective, any
differentiable relaxation of component count is the class Laplacian's algebraic
connectivity (candidate 136, killed at Gate 1) or a spreading/diversity penalty
(candidate 137; Ranked List Loss; Anti-Collapse; reverse contrastive), both
excluded covariance/diversity regularisers; the discrete version is mode
pseudo-identities (candidate 185, occupied by Easy Positive and
Divide-and-Conquer). Candidate 186's causal-direction objection also applies
unchanged: fragmentation correlates only **+0.04754** with class R@1 while the
variable the edit spends, mean within-class cosine, correlates **+0.41302**.
**DEAD.**

## 192. Absolute margin-distribution supervision

**Provenance.** `docs/neighborhood_error_audit.md`: mean top1--top2 cosine margin
**0.033816** for correct versus **0.023650** for incorrect retrievals.
`docs/proxy_geometry_audit.md`: ownership margin **+0.0305** for correct versus
**-0.0309** for errors, and R@1 **0.6441** among the 118 images with margin at
most -0.20.

**Mechanism.** Make the supervised object the achieved positive-minus-negative
gap targeted at a fixed absolute value and distribution, rather than its sign.

**Algebraic attack.** A target on the gap is the definition of a margin loss:
FaceNet's triplet margin (Schroff et al., CVPR 2015) and the learned-threshold
margin loss of Wu et al., *Sampling Matters in Deep Embedding Learning*
(ICCV 2017), occupy it; shaping the penalty as a function of the gap is loss-shape
learning, already killed as candidate 65. The relation set is unchanged — only the
penalty applied to it moves. This is the weighting row. **DEAD.**

## 193. Cardinality-invariant relation supervision

**Provenance.** `docs/proxy_geometry_audit.md`: proxy-centroid cosine correlates
**0.2760** with training identity size, rising monotonically from **0.08396**
(1--3 images) to **0.19061** (21+ images). In-Shop identity sizes are 12/10/416/
1,575/671 classes at 1/2/3/4/5 images.

**Mechanism.** Supervise so the learned same-class relation is identical in form
regardless of how many examples the identity has.

**Algebraic attack.** Cardinality enters a separable model only through sampling
frequency, per-example or per-class coefficients, or the margin/logit offset.
Those are class-balanced loss (Cui et al., CVPR 2019), logit adjustment (Menon et
al., ICLR 2021), and count-adaptive margins (AdaptiveFace, CVPR 2019). Candidates
100 and 141 already died on precisely this, and candidate 34 showed the exposure
premise was arithmetically wrong (`0.044/176` versus `1/3997` are both ~0.00025).
The weighting row. **DEAD.**

## 194. Transitive-closure certificate supervision

**Provenance.** The fragmentation pair above (**40.33%** disconnected,
**+5.875** adjusted points) plus `docs/method_search_verdict.md` §13: CUB
within-class local relation structure is stable (pair-rank Spearman **0.863**,
top-5 positive Jaccard **0.411**) while global class-centred residual modes are
not (cross-run ARI **0.06--0.07**).

**Mechanism.** Replace "all same-class pairs are positive" with the existential
relation "there exists a chain of high-similarity steps inside the class joining
`A` to `B`" — supervise connectivity rather than adjacency, so distant same-class
images are never directly attracted.

**Algebraic attack.** This is DAMLRRM verbatim: Xu et al. (CVPR 2019) construct a
per-class minimum-cost spanning tree so that every positive pair is connected
"directly or indirectly", explicitly to avoid all-pairs collapse, and evaluate on
CUB, Cars196 and SOP. Candidates 32 and 68 already died against it. Any
differentiable relaxation of "a path exists" is either algebraic connectivity
(candidate 136, Gate-1 dead) or a soft-min over paths, i.e. graph-based positive
propagation occupied by ProxyGML (Zhu et al., NeurIPS 2020) and STML (Kim et al.,
CVPR 2022). The chain edges are themselves chosen from the current embedding, so
the subsetting row applies as well. **DEAD.**

---

## Verdict

**NONE.** No candidate in 187--194 survives. Seven die at Gate 2 against named
primary sources; 188 dies earlier, on vacuity, because a separable encoder makes
its constraint identically true.

The batch is more informative as a group than individually. Each candidate
attacked a different route to new information — asymmetry of the retrieval duty
(187), the training population (188), a non-learned pixel estimator (189),
higher-arity real-image tuples (190), an integer structural label (191), the
absolute scale of the achieved gap (192), class cardinality (193), and
existential rather than universal class connectivity (194). All eight land in
the five-row table above. The excluded families cover every executable proposal
in this round; that empirical coverage must not be upgraded into a proof that no
other operation exists.

This remains evidence-bounded, not an impossibility proof. The round reopens if
either (a) an experiment exposes an information-bearing cross-instance relation
whose estimator uses neither the current model state nor a frozen appearance
proxy, or (b) a primary-source audit shows one of the occupying works does not in
fact perform the mechanism attributed to it. The pending seed-1/seed-2
fragmentation replication is the nearest scheduled measurement that could do (a);
it cannot do so by itself, because a confirmed marker still needs an operator
outside the table.

## Primary sources cited in this batch

- Wang et al., *Ranked List Loss for Deep Metric Learning* (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper.html
- Brown et al., *Smooth-AP* (ECCV 2020): https://arxiv.org/abs/2007.12163
- Sagawa et al., *Distributionally Robust Neural Networks* (ICLR 2020): https://openreview.net/forum?id=ryxGuJrFvS
- Qi et al. (ECCV 2020): https://arxiv.org/abs/1912.11194
- Xu et al., *Deep Asymmetric Metric Learning via Rich Relationship Mining* (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/html/Xu_Deep_Asymmetric_Metric_Learning_via_Rich_Relationship_Mining_CVPR_2019_paper.html
- Kim et al., *Deep Metric Learning Beyond Binary Supervision* (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/papers/Kim_Deep_Metric_Learning_Beyond_Binary_Supervision_CVPR_2019_paper.pdf
- Amodio and Krishnaswamy, *TraVeLGAN* (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/html/Amodio_TraVeLGAN_Image-To-Image_Translation_by_Transformation_Vector_Learning_CVPR_2019_paper.html
- Schroff et al., *FaceNet* (CVPR 2015): https://arxiv.org/abs/1503.03832
- Wu et al., *Sampling Matters in Deep Embedding Learning* (ICCV 2017): https://arxiv.org/abs/1706.07567
- Cui et al., *Class-Balanced Loss* (CVPR 2019): https://arxiv.org/abs/1901.05555
- Menon et al., *Long-tail learning via logit adjustment* (ICLR 2021): https://openreview.net/forum?id=37nvvqkCo5
- Liu et al., *AdaptiveFace* (CVPR 2019): https://openaccess.thecvf.com/content_CVPR_2019/html/Liu_AdaptiveFace_Adaptive_Margin_and_Sampling_for_Face_Recognition_CVPR_2019_paper.html

## Independent-review correction

Claude's initial draft incorrectly cited Lee et al., *Correlation Verification
for Image Retrieval* (CVPR 2022), as positive-pair preselection. Primary-source
inspection shows that CVNet is a learned dense-correlation geometric re-ranker;
it does not support that claim. The citation was removed before commit. Candidate
189 remains dead because DAMLRRM directly implements visual-distance-based sparse
same-class eligibility, and because this round explicitly excludes another
positive miner.
- Zhu et al., *ProxyGML* (NeurIPS 2020): https://proceedings.neurips.cc/paper_files/paper/2020/hash/ce016f59ecc2366a43e1c96a4774d167-Abstract.html
- Kim et al., *STML* (CVPR 2022): https://arxiv.org/abs/2203.16294
