# Cross-field candidate batch 156--158

Date: 2026-08-01. This batch was generated from the two strongest surviving
repository measurements: disconnected within-class In-Shop graphs correlate
with **+3.534 Recall@1 points** after exact class-size matching, while repairing
the evaluation metric of a trained `region_pa` arm from fixed-slot cosine to
MaxSim recovered **+6.67 points**. This is not a frozen-feature gain; the actual
frozen Cars probe found MaxSim **-1.47 points** below global pooling. The batch
was reported before implementation.
No diagnostic, implementation, or GPU run followed.

## 156. Same-class set-volume supervision

The proposed import from determinantal design was to require a same-class batch
to retain linearly independent visual evidence rather than contract to a point.
This does not survive Gate 2. Its action is an intra-class diversity or variance
constraint; variance-preserving DML, group-sensitive triplet sampling, and
self-supervised intra-class ranking already preserve such structure. A log-det
or DPP estimator changes the scalar functional, not the supervision relation.

Closest mechanisms include Bai et al., *Incorporating Intra-Class Variance to
Fine-Grained Visual Recognition* (2017), and Fu et al., *Deep Metric Learning
with Self-Supervised Ranking* (AAAI 2021). Candidate 156 is **DEAD AT GATE 2**.

## 157. Cross-view error-correcting evidence

The coding-theory proposal required two images of the same class to remain
matchable after independently erasing subsets of their local evidence, without
assigning a training-class codeword. Algebraically this is robustness to feature
or region dropout plus the original pair label. If explicit parity/syndrome
coordinates are introduced, it becomes supervised hashing/ECOC; if they are
not, it is erasure augmentation or consistency regularisation. Candidate 151
already closed the coding version. Candidate 157 is **DEAD AT GATE 2**.

Claude's alternative learned class-syndrome decoder also fails the open-set
task structurally: unseen test identities have no learned syndrome. Replacing
class syndromes by pairwise syndromes reduces to a learned distance or hash.

## 158. Residual retrieval-error boosting

The proposal assigned a second metric head only those retrieval errors not
already rescued by local evidence. BIER (*Boosting Independent Embeddings
Robustly*, Opitz et al., BMVC 2018) already divides the embedding into an online
gradient-boosted ensemble and reweights samples for later learners. The
repository's stronger all-five-miss diagnostic also supplies no Gate-1 case for
a special rescue operator: concatenation rescued only **15/5,924 = 0.253%** of
all queries after every component missed. Candidate 158 is **DEAD AT GATE 2**.

## Process result

Names imported from coding theory, experimental design, or boosting do not make
an operator new. In this batch every usable action reduces respectively to
variance preservation, erasure consistency/hashing, or boosted metric heads.
The next batch must define a higher-order supervision relation whose effect
cannot be written as pair reweighting, an auxiliary prediction target, or
diversity regularisation.
