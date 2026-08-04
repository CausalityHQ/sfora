# IMSDO 2025 margin/feature-variance prior-art audit

Date: 2026-08-04. Primary source: Winston and Kang, *IMSDO: Deep Metric
Learning with Incremental Margin and Standard Deviation Optimization*,
Neurocomputing 655 (2025), 131376,
[DOI 10.1016/j.neucom.2025.131376](https://doi.org/10.1016/j.neucom.2025.131376).

IMSDO is direct benchmark-matched prior art on CUB, Cars196, SOP and In-Shop.
It linearly/logarithmically schedules a triplet margin upward during training as a
curriculum, and adds a mean-squared penalty that drives average feature-map
standard deviation toward a selected target in order to suppress insignificant
maps. The paper sweeps margin schedules, target deviations and the combined-loss
weight on benchmark performance.

Consequently, the following are occupied and cannot be presented here as new:

- warm-up or curriculum schedules on a metric-learning margin;
- progressive easy-to-hard separation implemented by increasing that margin;
- directly targeting feature-map standard deviation to select/suppress channels;
- combining those two operations with triplet or proxy losses.

This source reinforces the existing mechanism reduction in candidate 108:
norm/dispersion-conditioned dynamic margins and feature-spread control are
established actions. Changing the growth curve, applying the statistic per layer,
or substituting variance/MAD does not create a new supervision relation. The
accessible source does not provide enough protocol detail or uncertainty to alter
the project's numerical SOTA horizon; its role is Gate-2 occupancy.
