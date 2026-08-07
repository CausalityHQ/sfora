# Pass 75 — NONE; BN artifact and positive-side redirection

Claude's adversarial review generated Identity-Agnostic Normalization
Calibration and killed it at Gate 2: Batch Renormalization, EvalNorm,
AdaBN/PreciseBN, EMAN, and MetaBIN already occupy the mechanism. No training
run was authorized.

The required positive-side diagnostic on matched trained pre-head features
showed mean unseen-minus-seen changes of **−0.05305** for nearest positive,
**+0.03220** for nearest foreign, and **−0.08523** for the P−F margin.
Foreign crowding explains only about 38% of the margin degradation; the
dominant measured issue is within-class positive collapse.

The preregistered BN-buffer placebo also collapsed the foreign effect to about
−0.000015 and positive/margin changes to near zero. The original +0.032
foreign shift was train-specific BatchNorm covariate shift, not a learned
unseen-class mechanism. The placebo deliberately collapses features and is a
diagnostic, not a recipe.

This closes the foreign-crowding/normalization family. The remaining measured
lead is within-class positive-side degradation on In-Shop. Any new candidate
must derive Gate-1 provenance from that term, pass primary-literature review,
and be preregistered before GPU training.
