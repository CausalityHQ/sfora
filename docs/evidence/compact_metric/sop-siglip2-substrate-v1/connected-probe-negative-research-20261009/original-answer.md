Verdict: the probe null is an optimizer-budget artifact, not evidence about pooling. Coverage and stale gallery are already excluded as causes by existing arms. The one zero-training mechanism worth running is query-only asymmetric serving of the archived MLP candidate, falsified from archived wires and the FIT cache.

## Observations (receipts and source, no inference)

- **Probe arm.** 1152 parameters at AdamW lr 1e-5, 128 updates, clip 1.0, decay 0.05. R@1 identical on all 1734 queries; AP moved on 102 queries, net −0.0038 pp.
- **MLP arm, same recipe.** Last-block MLP, about 9.9M parameters, same lr 1e-5, same 128 updates, same stale-gallery objective. Two-seed mean +0.346 pp R@1 and +2.300 mAP@R. Killed only because the product-bootstrap R@1 interval [−0.085, +0.828] crossed zero.
- **Coverage arms.** The 2016-class identity-diversity arm returned R@1 equal to control (96.83 both, mAP +0.16). The identity-mix arm was negative (−0.29 R@1, −0.62 AP).
- **Stale gallery.** For every encoder-side arm the ranking gallery is the cached frozen-encoder canonical features through current A/C, and the prototype MSE targets class means from the same cache. Export and scoring re-encode both sides with the adapted encoder. The freshness counterfactual (SS/SF/FS/FF) was designed three times and never produced a cell. All three failed on a control-tap parity predicate caused by B32-versus-B6 tail bytes, not on science.
- **Error structure.** 56 errors on the panel. 44 are shared across all four MLP endpoints, 41 of those with the same top impostor product, all margins negative, best positive at median rank 2.5 to 3.

## Ranked causal reasoning

1. **Optimizer geometry is the proximate cause of the probe null.** With Adam, each coordinate moves at most about lr per step (3.16× lr in the worst bias-correction case), independent of gradient size. The displacement budget therefore scales as √d × lr × T. Over 128 steps the probe's L2 budget is 0.04 to 0.14, while the MLP's is 4 to 13, roughly 93× larger. HF initialises this probe with unit-variance randn, so a 0.04 to 0.14 move on a norm-34 vector is 0.1 to 0.4 percent, which changes attention logits by that fraction and pooled features at the 1e-3 level. That is exactly the signature observed: zero R@1 flips, near-tie AP flips on 6 percent of queries. The MLP arm moved quality under the identical lr, steps, and gallery, so the budget, not the recipe, is what differed. This is not "gradient was small", and it does not justify an LR sweep. It means the probe arm never tested whether the pooling query is a useful direction.
2. **Stale gallery is real but secondary.** It is a train/serve asymmetry shared with the MLP arm, which still gained 2.3 mAP@R. It becomes decision-relevant only where displacement is material, meaning the MLP candidate, and its contribution there is unmeasured. The root's dataflow audit already names the exact open question: whether same-image frozen versus adapted gallery features differ enough to change nearest margins.
3. **Coverage is closed.** Two arms, one null and one negative. The readout is not data-starved on this substrate.
4. **Representation is untested by the probe arm** and only weakly tested by the MLP arm. The core-44 with one shared impostor product per query across all endpoints is consistent with either near-duplicate product listings or a genuine shared blind spot; the census could not split these.

**Gate arithmetic that binds everything.** Fixing all 12 non-core errors yields at most +0.69 pp R@1. The frozen product-bootstrap lower-bound rule needs roughly +0.45 pp with low variance. The MLP's +0.35 pp came from 6 net queries. No readout or encoder arm can pass the panel R@1 gate without touching the core-44, so continued readout-side arms on this panel are expected to fail by construction.

## ONE recommendation

**Mechanism: query-only asymmetric serving of the archived MLP candidate.** Keep the gallery on the frozen encoder plus the candidate's A/C, adapt only the query path. This is the condition the objective actually optimised; symmetric re-encoding at export was an untested extrapolation. Serving cost is unchanged, the gallery wire never needs re-indexing, and the only new bundle content is the four MLP tensors as a query-side overlay.

**Falsifier, TRAIN-only, zero training, no live encoder.** Compute the FS cell: archived candidate query unit descriptors from the existing export wires, scored against the 1715 gallery rows' frozen 1152-d FIT-cache features passed through the candidate's A/C readout on CPU FP32, with the original scorer and the frozen 5000-draw product bootstrap. Pair against archived FF (symmetric candidate) and SS (control) for both seeds. Using the FIT cache instead of a live tap removes the B6-tail parity blocker entirely; the only new computation is 1715 readouts.

**Stop rule.**
- Kill if FS − FF ≤ 0 on paired R@1 for either seed, or FS − SS fails the unchanged lower-bound rule. That closes the stale-gallery family and asymmetric serving. The remaining lever is core-44, which forces a root decision on whether the panel R@1 gate is attainable at all.
- Continue only if FS beats FF on both seeds and clears the frozen lower bound. The MLP candidate then re-enters at the same-four VAL gate under an asymmetric serving contract with no new training.

**Bounded scope and cost.** One read-only CPU job per seed. Prior freshness runs sat at 20 s and under 4.5 GB host before failing; this version does less. Inputs are the two candidate export directories, the two inference bundles, the FIT cache rows mapped by original_fit_index, and the frozen scorer. No CUDA, no images, no training, no edits to the trainer. Fold two free reads into the same job, since they read the same archives: the probe and MLP relative weight displacement from the FP32 payloads, which pins the probe interpretation above with a prediction of under 0.5 percent, and the frozen-feature cosine between each core-44 impostor product and the true product's gallery images, which tells the root whether the core is duplicate listings or a representation gap before any further panel arm is funded.

## Route to publishable quality and measured speed

Quality claims should move to mAP@R as co-primary on the official In-Shop and SOP protocols, where the MLP family's +2.3 mAP@R is the only positive signal in this substrate line. That is a root freeze decision. On speed, the serving path cannot yet be measured honestly: every call to inference_outputs runs encoder_facts, which SHA-256s the frozen-447 set and then the full state dict, about 1.6 GB per request. That accounts for the 1.69 s batch-1 median and most of the 2.0 s batch-32 figure. Product latency is undefined until that fingerprint moves to load time, which is the root's existing fresh-SHA track and not a new arm.

Verdict saved to memory for later sessions.
