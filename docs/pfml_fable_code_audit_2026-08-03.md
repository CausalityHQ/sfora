# Fable code-to-primary-source PFML audit

Date: 2026-08-03. Completed while the repaired fixed-interpretation Cars run
had reported only epochs 10--60. Fable 5 compared the extracted paper and
supplement with the executable loss, protocol, optimizer, and CLI paths. The
main agent independently checked the cited source passages and code. No code or
GPU action was delegated to Fable.

## Verdict

**No new executable Eq. 1--6 bug was found.** Signs, branch inequalities,
sample/proxy populations, proxy--proxy terms, ordered double counting, and
gradient direction match the primary equations. The run remains a disclosed
fixed interpretation, not an exact reproduction.

The audit did find one primary-source contradiction and two overclaims in local
documentation/comments that are corrected with this record.

## Equation and population checks

- Attraction is `-delta^-alpha` inside the strict delta ball and `-d^-alpha`
  outside. Repulsion is `d^-alpha` inside and `delta^-alpha` outside. The code's
  `where` branches and signs match.
- Concatenating the batch and all class proxies, then masking by label, produces
  Eq. 4/5: same-class batch samples and proxies attract; different-class batch
  samples and proxies repel.
- Eq. 6 evaluates every batch embedding in its class field and every proxy in
  its class field. Each distinct unordered pair consequently occurs once in
  each direction. The ordered off-diagonal code matches this factor of two.
- With batch 100 as 25 classes x 4 samples and 98 x 15 proxies, a loss contains
  1,570 points, 2,463,330 ordered off-diagonal pairs, 23,880 same-class pairs,
  and 2,439,450 different-class pairs. At delta 0.2 and alpha 3, the force-free
  energy floor is about 301,946,250. This independently explains the roughly
  300-million loss scale and why raw energy near the floor says little about
  remaining force.

## Primary-source learning-rate contradiction

The main paper's Section 4.1 says Adam learning rate **`5e-4`** and proxies at
100x, implying `0.05`. The official supplement's dataset/architecture-specific
Table 5 says **`1e-4`** and proxy LR **`0.01`** for Cars R50/512. The two sources
cannot both describe the same execution.

The preregistration already specified that dataset-specific Table 5 takes
precedence, so the active run remains `1e-4` / `0.01`; changing it after the
trajectory was visible would be tuning. But any failure must list this 5x
intra-source contradiction prominently. It is a plausible source of a large
gap and prevents an exact-reproduction claim.

## Loss-scale wording correction

Eq. 6 is written as a raw sum, so the active implementation is the defensible
literal reading. It is not legitimate to claim that the authors' unavailable
code *must* have used the sum. With Adam's coupled decay, replacing the sum by a
pair mean is not a harmless global rescaling: it changes the data-gradient to
weight-decay ratio by roughly the pair count (and changes the effect of Adam
epsilon). Conversely, the raw sum can make nominal `1e-4` decay effectively
tiny until the field approaches equilibrium.

Therefore the historical mean-scaled run is invalid evidence about this fixed
interpretation, but neither scale is established as the authors' unpublished
implementation. “Literal fixed interpretation” replaces “required.”

## Ranked ambiguity surface if the fixed run fails

1. undisclosed Cars-selected delta/alpha within the paper's search grid;
2. main-text `5e-4` versus Table-5 `1e-4` base LR contradiction;
3. unavailable loss reduction and its coupled-decay interaction;
4. balanced 25x4 sampling versus random shuffled/drop-last sampling;
5. absent versus present gradient clipping under raw-sum gradients;
6. head initialization, pretrained-weight digest, and augmentation details;
7. normalized proxy directions versus unconstrained proxy positions;
8. frozen BN statistics with trainable affine versus freezing affine too.

No item is relabeled a bug because the authors do not disclose the needed
choice. A failure falsifies the prospective local interpretation and supplies
no license to tune these choices after seeing the outcome.

## Corrected stale causal statement

An older `symmetric_potential` docstring called the historical mean-scaled PFML
collapse empirical confirmation that short-range repulsion necessarily causes
collapse. That result was confounded by the loss-scale/coupled-decay defect and
cannot identify the mechanism. The code comment now labels the motivation
heuristic; no executable behavior changes.

## Primary sources

- Bhatnagar and Ahuja, *Potential Field Based Deep Metric Learning*, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html
- Official supplement:
  https://openaccess.thecvf.com/content/CVPR2025/supplemental/Bhatnagar_Potential_Field_Based_CVPR_2025_supplemental.zip
