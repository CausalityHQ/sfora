# NC-Init + perturbation injection prior-art audit

Date: 2026-08-04. Primary sources:

- Park et al., *Neural Collapse-Informed Initialization with Perturbation
  Injection in Classification-based Metric Learning*, AAAI 2026,
  DOI [`10.1609/aaai.v40i10.37777`](https://doi.org/10.1609/aaai.v40i10.37777).
- Authors' [code](https://github.com/jinnnnnnnnn/NC_init_PI), inspected at
  commit `06caa23b59113ce81e0a82461c6beffc24120887`.

## Mechanism occupied

The method freezes a pretrained backbone briefly, extracts each benchmark
training class's principal feature direction, initializes that class's proxy
along the direction, then injects small isotropic Gaussian perturbations into
the proxy during subsequent Proxy Anchor training. Its stated object is to
preserve pretrained neural-collapse geometry while limiting cumulative proxy
drift and smoothing the embedding space.

This is direct DML prior art for all of the following candidate descriptions:

- training-class proxies initialized from frozen-backbone class PCA or neural-
  collapse directions;
- penalizing or bounding proxy drift from those data-dependent directions;
- noise injection around a pretrained/data-dependent proxy initialization;
- a warm-up that fits the projection/proxy head before joint fine-tuning.

Changing PCA to a nearby class-direction estimator, changing Gaussian noise to
another isotropic perturbation, or adding an explicit drift penalty would be a
variant, not a new supervision relation.

## Numerical and evidential boundary

The paper's headline experiments use ImageNet-21K initialization. Reported
single-model Recall@1 includes:

| backbone/method | CUB | Cars196 | SOP | In-Shop |
| --- | ---: | ---: | ---: | ---: |
| ResNet-50 PA reproduction | 81.8 | 86.8 | 81.8 | 92.0 |
| ResNet-50 PA + NC-Init/PI | 83.2 | 89.2 | 82.1 | 92.5 |
| HypViT | 85.6 | 86.5 | 85.9 | 92.5 |
| HypViT + NC-Init/PI | 85.9 | 88.5 | 86.3 | 92.9 |

These values do not raise this project's absolute CUB (87.8), Cars (94.9), or
In-Shop (93.9) horizons. They also do not replace the ImageNet-1K comparable
lane: the gain and the much stronger CUB baseline are reported after changing
pretraining to ImageNet-21K.

No seed count, standard deviation, confidence interval, or error bar appears in
the paper. The released entrypoint calls `seed_everything()` with a fixed
default seed of 1 and exposes no seed CLI argument. The evidence should
therefore be treated as a single-run reviewed result unless the authors provide
otherwise. It establishes prior art, not a precision estimate.

## Search consequence

This paper does not alter the outcome-only Fable target, but it materially
narrows the initialization/optimization search space. A Fable proposal based on
recovering frozen-backbone class directions, neural-collapse proxy geometry,
proxy drift control, or perturbing those proxies fails Gate 2 unless it defines
a genuinely different mathematical object and supplies a causal measurement
beyond this method's one-seed evidence.
