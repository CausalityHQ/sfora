# Pass 129 — CEA Gate-1 result and Gate-3 preregistration

## Gate-1 result (before any CEA training run)

The CPU diagnostic used the trained In-Shop checkpoint
`arcg_inshop_pa_epoch10_seed0.pt`, not a random or one-step model.  It formed
gradient-times-activation class-evidence maps at the BN-Inception spatial
branches and compared different same-class images.

The result is **0.6374427 AUC** for evidence-map agreement to distinguish the
closest quartile from the most distant quartile of same-class pairs, with
**25.1928%** of same-class pairs retained by the upper-quartile agreement gate.
There were 389 usable pairs from 128 train images.  The diagnostic is CPU-only;
its complete JSON output is `docs/evidence/cea_gate.inshop.cpu.json`.

This clears the preregistered Gate-1 thresholds (AUC gain over chance at least
0.05 and coverage at least 5%).  It does not establish that the gate improves
retrieval: the same evidence signature could merely restate embedding distance.

## Gate-3 preregistration

The paired corrected In-Shop Proxy Anchor reference is R@1 **0.9163033**.
Before implementation or a deciding GPU run, CEA predicts:

* raw best-over-training R@1: **0.9191**;
* independently frozen checkpoint R@1: **0.9182**.

The deciding run is falsified if the frozen checkpoint is below **0.9175**, or
if CEA fails to beat both (a) ordinary Proxy Anchor and (b) a matched
distance-only positive gate by **0.0010**.  The distance-only control retains
the same pair budget and uses the embedding-distance threshold without
evidence maps.  Every arm must report raw best and frozen values; the local
peak-gap output is descriptive only and is not called a selection correction.

The CEA claim remains narrow: the training-time object is a hard
positive-to-unknown decision between two different labelled images based on
agreement of their class-evidence-drop signatures.  It is not a saliency
auxiliary loss, a soft pair weight, or an inference-time reranker.

No GPU run is authorized here until the active Pass-119 controller's corrected
random artifact and analysis marker are complete.
