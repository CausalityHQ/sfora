# PFML prelaunch fidelity audit

Date: 2026-08-03. This audit precedes any repaired PFML GPU step or result. It
uses the CVPR 2025 paper, its official supplement, the current executable path,
and the official Proxy Anchor source as an adjacent implementation reference.
PFML has no discoverable authors' training source, so undisclosed choices remain
assumptions rather than source matches.

## Verdict

**GO for one fixed-interpretation Cars196 reference after one repaired CLI bug
and a fail-closed smoke. Do not call it source-exact.** The paper/supplement pin
the main method and most high-impact hyperparameters, but do not pin the train
augmentation, sampler, proxy normalization, gradient clipping, embedding-head
initializer, pretrained-weight digest, or checkpoint-selection cadence.

This run is occupied baseline infrastructure, not candidate evidence. Its value
is to test the repaired Eq. 6 implementation and produce a modern final
checkpoint/trajectory from which a new non-Gram measurement can be derived.

## Confirmed launch-blocking bug and repair

`config_for_protocol("pfml-resnet50-512")` correctly set `potential_alpha=3`,
but the real CLI declared an unrelated concrete default `--potential-alpha 4`
and unconditionally copied it over the protocol config. Thus a command that
named the PFML protocol but omitted that flag silently ran alpha 4. Direct
protocol unit tests did not exercise this path, and one CLI test was itself
stale (`cosine` expected although the repaired preset specifies no schedule).

Commit `aa878e8` makes potential delta/alpha optional CLI overrides, falls back
to the resolved protocol values, and regression-tests the actual command path:
PFML now resolves to delta 0.2, alpha 3, Adam, and no LR schedule. The focused
PFML/CLI test set passes (27 tests) and Ruff is clean.

## Source-matched surface

The official supplement's Table 5 specifies, for Cars196 ResNet-50/512:

- Adam, base/head LR `1e-4`, proxy LR `0.01`, weight decay `1e-4`;
- batch size 100, 200 epochs, one warm-up epoch, frozen BatchNorm;
- 15 proxies per class.

The main paper specifies L2-normalized 512-D output, a standard ResNet average
pool with only the final FC changed, 224-pixel inputs, and 256-resize/224-center-
crop evaluation. It searches delta in `[0.1,0.3]` and alpha in `{0,...,6}`;
the fixed local interpretation remains delta 0.2 and alpha 3 as prospectively
registered.

The executable optimizer groups resolve to `1e-4` backbone, `1e-4` head and
`0.01` proxies. Adam's coupled `1e-4` weight decay applies to all groups. The
one-epoch warm-up freezes the pretrained backbone while training the new head
and proxies. That meaning is not defined in the PFML supplement, but it matches
the official Proxy Anchor source's explicit comment and implementation:
"Train only new params" before unfreezing at the requested epoch.

## Eq. 1--6 algebra

For the union of batch embeddings and all class proxies, Eq. 6 evaluates each
point under its own class field. Therefore every distinct unordered pair
appears in both directions. The current all-points matrix and ordered
off-diagonal sum match that population: sample--sample, sample--proxy, and
proxy--proxy interactions are all present.

- A same-label pair contributes the constant `-delta^-alpha` inside delta and
  `-d^-alpha` outside.
- A different-label pair contributes `d^-alpha` inside delta and the constant
  `delta^-alpha` outside.
- Self-attraction is a constant with zero gradient. Excluding diagonal terms
  changes only the reported energy offset, not optimization.
- Clamping distance at `1e-4` affects exact/near-exact collisions only and
  prevents an infinite repulsive value. It is a disclosed numerical safeguard.
- Returning the raw ordered sum is required by Eq. 6. The historical mean-loss
  implementation changed the ratio of data gradient to Adam's coupled weight
  decay by millions and its collapsed result remains invalid evidence.

## Unresolved source ambiguities

The fixed run retains the already registered local assumptions: four samples
per class, resize-256 followed by random-resized-crop-224 plus flip, torchvision
ImageNet V1 weights, default FC initialization, normalized proxy directions,
trainable BatchNorm affine values with frozen running statistics, and no
gradient clipping. These choices are not established by the PFML paper.

The official Proxy Anchor source instead supplies plausible adjacent choices:
random shuffled/drop-last batches unless `--IPC` is explicitly passed,
random-resized-crop without a preceding resize, Kaiming head/proxy
initialization, frozen BatchNorm affine parameters, proxy normalization, and
elementwise gradient clipping at 10. Importing that entire recipe after the
PFML preregistration would replace one undocumented interpretation with
another; it would not create source evidence. Any result must therefore say
"fixed interpretation" and list these assumptions. Failure may falsify this
implementation, but cannot by itself falsify PFML's published result.

## Fail-closed smoke and reporting contract

Before the 200-epoch run, a one-step full Cars batch must satisfy all of:

1. serialized config says ResNet-50/512, alpha 3, delta 0.2, 15 proxies/class,
   Adam, `1e-4` base/head LR, `0.01` proxy LR, and no LR schedule;
2. the official split contains 98 train and 98 test classes with no overlap;
3. loss, model parameters, proxy parameters, and their gradients are finite;
4. head and proxy gradient norms are nonzero, while the backbone is frozen in
   warm-up;
5. the output is explicitly marked a one-step smoke and cannot satisfy the
   completed-artifact controller.

The deciding run evaluates test retrieval at a fixed ten-epoch cadence. Report
raw best-over-training R@1 and epoch as a test-selected diagnostic, plus the
unrestored final-epoch R@1 as primary. The historical local-neighbour peak-gap
estimator is not a selection correction and will not be reported as one. Save
the final checkpoint with `artifact_selection=final_training_state`; export
final embeddings separately and verify their scorer before using them as
candidate provenance.

## Smoke result

The repaired one-step Cars196 smoke completed successfully before the deciding
run. Its artifact SHA-256 is
`606fb5ed2b9ac5473db2b5bef94691b92f7e004b0c5ba2c54e74cc56e81d52e9`.
The serialized executed config contains ResNet-50/512, Adam, base/head LR
`1e-4`, proxy multiplier 100 (`0.01` effective), weight decay `1e-4`, 15
proxies/class, delta 0.2, **alpha 3**, no schedule, and one warm-up epoch. The
raw ordered potential was finite (`304,921,600`), the optimizer step completed,
the complete JSON contains no NaN or infinity, and final scoring remained
finite. The one-step R@1 `0.413602` is an initialization/smoke observation and
is excluded from benchmark reporting.

An independent reload verified 8,054 train images in 98 classes labelled 0--97
and 8,131 test images in 98 classes labelled 98--195, with an empty class
intersection. Peak observed GPU allocation was 2,808 MiB. Dataset materializing
plus eight forked loader workers gave approximately 48 GB parent RSS (mostly
shared copy-on-write pages) and left at least 65 GiB host memory available; this
is high but did not approach the 121-GiB host limit. The smoke therefore clears
the finite-state, split, configuration, and memory gates for the fixed 200-epoch
run.

## Primary sources

- Bhatnagar and Ahuja, *Potential Field Based Deep Metric Learning*, CVPR 2025:
  https://openaccess.thecvf.com/content/CVPR2025/html/Bhatnagar_Potential_Field_Based_Deep_Metric_Learning_CVPR_2025_paper.html
- Official PFML supplement:
  https://openaccess.thecvf.com/content/CVPR2025/supplemental/Bhatnagar_Potential_Field_Based_CVPR_2025_supplemental.zip
- Official Proxy Anchor implementation:
  https://github.com/sung-yeon-kim/Proxy-Anchor-CVPR2020
