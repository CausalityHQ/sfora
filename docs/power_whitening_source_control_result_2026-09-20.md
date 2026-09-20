# Power-whitening source-control result

This claim-ineligible causal diagnostic compared the selected supervised
within-class covariance transform with shuffled-label and label-free
total-covariance controls on the two largest-gain regimes, Cars and Food101.
Every grid arm was selected using actual signed-int8 codes on fit-only class
folds. The terminal result is 23,747 bytes with SHA-256
`e58d6ac2a9c53238837165dfdd86b0fb10e5f8224ca298825b613d0ff3893e18`.

| Dataset | Within-class | Shuffled labels | Total covariance | Ledoit-Wolf alpha=0.5 |
|---|---:|---:|---:|---:|
| Cars mAP@R / R@1 | 0.860658 / 0.974419 | 0.791897 / 0.970729 | 0.792175 / 0.971221 | 0.846998 / 0.972205 |
| Food101 mAP@R / R@1 | 0.766113 / 0.941333 | 0.750792 / 0.941412 | 0.750879 / 0.941255 | 0.761514 / 0.939922 |

The label-free total-covariance arm trails supervised within-class covariance
by 0.068483 mAP@R on Cars and 0.015234 on Food. Their mean difference is
`-0.041858`, far beyond the frozen `-0.005` label-decorative gate. Shuffled
labels closely reproduce total covariance (within 0.00028 Cars and 0.00009
Food), supporting the control rather than suggesting an implementation fault.

The selected cells also differ: supervised within-class covariance chooses
`0.75:1.00` Cars and `0.50:1.00` Food, while shuffled and total covariance
choose much weaker `0.25` powers. Fixed Ledoit-Wolf shrinkage removes the grid
and retains most of the gain, but misses the selected transform by 0.013660 on
Cars and 0.004599 on Food; the grid cannot yet be retired under the frozen
within-0.005 rule.

This closes the universal unlabeled-whitening interpretation. The current
product is a generic supervised domain adapter fitted on labeled, disjoint
classes from the deployment domain. Because WCCN and power whitening are prior
art, these results alone are not a novel similarity-learning method. The next
algorithmic falsifier is the independently proposed power-initialized affine
learner with a fixed coverage/nearest-positive objective, aimed at improving
Recall@1 without surrendering the within-class mAP gain.

The complete source-control run took 75.919 seconds on the DGX.
