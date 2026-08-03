# Proxy initialization fidelity audit 242

Date: 2026-08-03.

## Defect and mechanism

The pinned Proxy Anchor source creates a Gaussian proxy matrix and applies
`nn.init.kaiming_normal_(proxies, mode='fan_out')`. SFORA instead sampled a
Gaussian matrix and normalized every proxy row to unit length before registering
the parameter.

Both forward passes normalize proxies for cosine similarity, but the parameter
dynamics are not equivalent. If `p` is normalized in the forward pass, its
Jacobian scales as `1/||p||`; AdamW moments, epsilon, and decoupled weight decay
also act on the raw parameter. Upstream SOP starts 512-dimensional proxy rows at
an expected norm determined by fan-out over 11,318 classes, not norm one. This is
therefore a training-recipe defect even though the initial proxy directions have
the same spherical distribution.

The first original-resolution corrected SOP launch was stopped at step 300 when
this was found. It produced no report and supplies no score.

## Scope of historical evidence

All historical Proxy Anchor and PA-derived runs in this repository used the
unit-normal proxy initialization. Their paired candidate deltas remain
observations under a shared modified initialization, but their claim to execute
the exact official PA recipe is withdrawn. In particular, absolute PA-to-paper
and PA-to-HIST comparisons mix this initialization deviation with the method
difference. This audit does not assert a score direction or magnitude before a
corrected run.

## Repair

`ImageEndToEndConfig.proxy_initialization` now distinguishes the archived
`unit_normal` behavior from `kaiming_normal`. Official Proxy Anchor recipes lock
the latter. Their recipe digests intentionally changed; unrelated HIST digests
remain fixed. A regression test verifies that Kaiming-initialized proxy rows are
not silently unit-normalized. The corrected SOP digest is
`6212b9499c00cf19ad4a53344ea348a3ea903bced426a74ef3315695afab3d00`.
