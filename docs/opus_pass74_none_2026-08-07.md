# Pass 74 — no GPU candidate; geometry premise requires repair

The blind invention/prior-art review returned **NONE**. Every proposed
intervention was either already occupied (Non-Isotropy Regularization,
rho-spectrum regularization, Proxy Synthesis, DDML, SEE, coding-rate and
late-phase collapse controls) or fell under the repository's closure lemmas.
No training run is authorized from this pass.

More importantly, the Gate-1 artifact in `ce79af5` is not a valid causal
comparison. The untrained diagnostic used GAP-only 1024-D features, while the
trained model evaluates average-plus-max pooling followed by a signed 1024→512
head. It also compared nearest-facility maxima over 25,882 train rows against
12,612 gallery rows. Extreme-value pool-size bias can be of the same order as
the observed +0.0109 untrained excess. The reported “training-induced” claim
is therefore withdrawn pending a matched control.

The repair is preregistered here before any rerun: use identical average-plus-
max pooled features, identical embedding head, rank-matched foreign order
statistics (or repeated equal-size subsampling), and report the positive-minus-
foreign margin as well as foreign similarity. Kill the provenance premise if
the untrained unseen excess is ≥0.020; pass only if it is <0.010, the trained
excess is >0.020 with non-overlapping four-seed intervals, and the foreign
component explains at least half of the margin change. This is a diagnostic,
not a new method, and no GPU training should begin until it passes.

Primary literature checks found no unoccupied intervention: [NIR](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Non-Isotropy_Regularization_for_Proxy-Based_Deep_Metric_Learning_CVPR_2022_paper.pdf), [rho-spectrum](http://proceedings.mlr.press/v119/roth20a/roth20a.pdf), [Proxy Synthesis](https://ojs.aaai.org/index.php/AAAI/article/view/16236), [DDML](https://ojs.aaai.org/index.php/AAAI/article/view/34184), and [SEE](https://www.ijcai.org/proceedings/2025/1214.pdf).
