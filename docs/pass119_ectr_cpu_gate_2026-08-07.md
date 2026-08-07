# Pass 119 ECT-R CPU gate

Date: 2026-08-07. This is inference-only analysis of the retained
`ect_zero_probe.json` generated from the corrected In-Shop epoch-10 checkpoint;
no weights, split, or test images were changed.

ECT-R freezes β=`.85`, uses the relative switch hinge
`[.20-(cos(z_c,z_b)-cos(z_c,z_a))]_+`, and freezes the plateau threshold at
the epoch-10 median anchor cosine `.53`.

| n | plateau active | switch active | area/gap Pearson r | area range |
|---:|---:|---:|---:|---:|
| 64 | 48.4375% | 46.8750% | -0.08351 | 0.1633–0.4898 |

The registered CPU thresholds were 5–95% activation for both hinges, absolute
area/gap correlation below `.20`, and area-matched plateau/switch construction.
All pass. The old ECT failure (100% switch activation from an absolute
partner-cosine hinge and β-dependent area) is not reused as evidence.

Disposition: **Gate 3 passed; corrected In-Shop Gate-4 implementation is
authorized.** The deciding run must include A0, A2, A4, A5, A6 and A7 with
identical pooling, sampler, optimizer, composite count, and replaced-area
matching. No Cars/CUB follow-up is authorized unless the full ECT-R arm beats
both soft-target and area-matched random-mask controls at the preregistered
In-Shop thresholds.

