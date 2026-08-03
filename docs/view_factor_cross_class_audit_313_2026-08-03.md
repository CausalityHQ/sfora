# Candidate 313: cross-class view-factor supervision

## Gate 1

RSPG showed that rival-class structure is nearly vacuous on CUB (64% of
within-class pairs retained) but selective on In-Shop (8.6%). A proposed next
mechanism was to infer a within-class factor such as viewpoint from cross-class
appearance matches, and use agreement of that factor as additional positive
supervision.

## Gate 2 verdict: dead

This is not an unexplored supervision primitive. Wang et al., *Exploiting
View-Specific Appearance Similarities Across Classes for Zero-Shot Pose
Prediction: A Metric Learning Approach* (AAAI 2016), explicitly learns
cross-class view-specific similarities and transfers them to unseen classes.
Viewpoint-aware metric learning in vehicle re-identification (Chu et al., ICCV
2019) likewise uses viewpoint-conditioned positive/negative constraints. The
proposed factor agreement would therefore be a reimplementation of supervised
view-aware metric learning, not a defensible new method. No implementation or
GPU was used.

Sources: AAAI paper, https://ojs.aaai.org/index.php/AAAI/article/view/10472;
Chu et al., https://openaccess.thecvf.com/content_ICCV_2019/papers/Chu_Vehicle_Re-Identification_With_Viewpoint-Aware_Metric_Learning_ICCV_2019_paper.pdf.
