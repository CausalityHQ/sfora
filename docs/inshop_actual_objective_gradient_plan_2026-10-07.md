Implement **one discarded actual-objective micro16 witness**, in a new driver and one stdlib test. Keep the frozen qualifier, trainer, readout, loss, and active DGX observation unchanged.

The checkout is `4bd46dcf`, one commit after `7bb4d0f0`; the requested source files are unchanged between them. CPU-v5 used the trainer at `964baa20`, whose bytes match its accepted freeze. Load that original native source-v5 module; the current trainer’s file hash differs.

1. **Authenticate the accepted initializer before native work.**

   Use `/home/riomus/runs/sfora-so400-identity-diversity-cpu-v5/initializer-control-179061.pt`, selected from the accepted receipt’s `qualifications` entry with `arm="control"`, `seed=179061`, and CPU identity.

   | Authority | SHA256 |
   |---|---|
   | CPU-v5 `receipt.json` | `d2239da896465ee004ebb95c9316713d634fa457c91c708c6c410769a8c17b03` |
   | Initializer file bytes | `8b89897fdaaad94770e40bd2e69422734709e6d2d73710aa496610bdb2882454` |
   | Complete typed initializer payload | `4a09ad011dfc4bc1897482b96ed69f882efe0c0962833a280f58a3cbd5adeaee` |
   | CPU-v5 trainer execution | `bd081a02a49f1f0dd8305b78bb3f8aa91f5bb7331f3045325a185971fb7c04a8` |
   | Frozen trainer bytes | `840c5d8277a89ccdac02c9e231cbe6eddf386e2b23915ecd1bbec1136c51dee8` |
   | Original CPU launch authority | `8fcfb9ae55c33ed842c68ab01b59fea378f1258b4b784d5874e40e9f49801eac` |

   The trainer root is `/home/riomus/runs/sfora-so400-identity-diversity-train-source-v5`.

   Reuse its `authority()` and `admit_terminal(..., "cpu", "control", 179061)` with the accepted CPU UNIT. Keep prospective driver guards separate until that historical admission finishes. The new witness authority binds its actual driver/test/helper/interpreter bytes, fresh output, both locks, and unchanged **300 seconds / 8 GiB / zero swap / CUDA <10 GB** policy. Compute prospective hashes after implementation.

2. **Restore the CPU initializer without rebuilding legacy native state.**

   Add one private model-free loader. Use `torch.load(map_location="cpu", weights_only=True, mmap=True)`, the original `CheckpointPages`, uncached typed `fingerprint()`, and `check_payload(..., step=0)`. Populate the validator context’s expected hashes from the admitted qualification; retain only required provenance/scope metadata, then use the original `restore()`.

   Preserve the **typed disk identity**. Compare its JSON form against the receipt identity; passing receipt JSON directly to restore can change tuple-valued optimizer fields.

   Require the original exact `PAYLOAD_KEYS`: schema, identity, source, A/C, all `STATIC_KEYS`, optimizer, scaler, counter, CPU/CUDA RNG, and numerical flags. In particular:

   - `A`: FP32 `[128,160]`, accepted A0; `C`: FP32 `[128,1152]`, exact zero.
   - Frozen head: `primary.weight`, `primary.bias`, `down.weight`, `up.weight`; buffers `center`, `preactivation_std`.
   - Means `linear[32]`, `concat[160]`; `mu_train[1152]`.
   - Both CPU views `[6355,1152]`; teachers `T`, `V`, `P[1008,128]`, `counts[1008]`, scalar `e0`.
   - Targets/original rows `[6355]`, both frozen schedules/masks/provenance, counter zero, empty AdamW moments.

   Run original `integrity()` and `canonical_initial_witness()` before encoder construction. **Do not call `prepare_native()`, `prepare_scope()`, or `fitter.prepare_original()`.** The initializer already supplies the admitted common statistics and data; those paths reconstruct legacy warm state unnecessarily.

3. **Use the exact first scheduled micro16 and genuine pixels.**

   ```python
   batch = cpu_state["schedules"]["179061"][0].tolist()
   anchors = batch[:16]
   K = trainer.ranking_membership(cpu_state["ranking_bank"], batch)["valid"]
   ```

   The bound anchors are:

   ```text
   185,419,3802,1536,5675,3633,5030,3737,
   5622,1070,6090,343,1312,4889,5235,1321
   ```

   **K is 63 for the complete B64**, although all first sixteen anchors are valid. Preserve regression denominator `128*e0` and ranking denominator `2*K = 126`.

   Reuse the authenticated genuine exporter’s selected manifest and `image_rows_node()` to construct its original `ImageRows(..., augment=True)`. Follow its pixel preparation exactly: hash the TRAIN file, decode RGB, canonical copy or `isolated_view()` with seed `179081 + original_train_row`, then the original processor’s `images=..., return_tensors="pt"`. Bind image/RGB/pixel facts and check global CPU RNG preservation. Process both views, one lifetime at a time.

4. **Run the real loss through one genuine encoder.**

   Reuse the original source driver’s `package_origins()`, `fresh_source()` and runtime checks, as in the current qualifier. Internally this authenticates the Siglip constructor, reads bare vision tensors with `safe_open`, performs strict state loading, and constructs the local torchvision-backed processor.

   Select exactly:

   ```text
   encoder.layers.26.mlp.fc1.weight
   encoder.layers.26.mlp.fc1.bias
   encoder.layers.26.mlp.fc2.weight
   encoder.layers.26.mlp.fc2.bias
   ```

   Keep the other 444 parameter names frozen. Use independent GPU copies of the accepted small readout/statistics; leave the restored CPU state pristine. Create no GPU optimizer.

   For each view, first run a **live frozen-control forward**, then enable the four MLP tensors and run the same pixels under the same recorded FP32 numerical/RNG role:

   ```python
   x = F.normalize(model(pixel_values=pixels).pooler_output.float(), dim=1)
   raw = connected.raw_features(x, head, A, means, C, mu, primitive, readout)
   mse, rank, facts = trainer.loss_terms(context, gallery_state, raw, anchors, K)
   ```

   `loss_terms()` accepts the connected query directly. Its existing `ranking_gallery()` sends all 6355 cached canonical features through current A/C and normalizes the result. Its existing membership/SmoothAP functions preserve official-image self exclusion and **all positives**. No loss adapter is needed.

   Compare initial live-control/connected raw, unit, scores and loss values within the declared numerical role. Cached features came from different extraction arithmetic; cached/live equality is not this pairing.

5. **Make connectivity falsifiable, then release everything.**

   Use `torch.autograd.grad()` to measure regression and **SmoothAP separately**. Require finite, nonzero SmoothAP gradients for all four MLP tensors, plus finite nonzero total gradients; verify both views and absent gradients for the 444 frozen names.

   Split query/gallery A/C into equal independent leaves and reuse `loss_terms()` to verify nonzero query and gallery contributions and `tied ≈ query + gallery`, using the existing comparison tolerances.

   Clear gradients and run the original detached query expression through the same complete gallery. Require forward-identical values and matching A/C gradients, with zero/absent encoder gradients. A `raw.requires_grad` check alone cannot establish this.

   Release every graph, gradient, split state, GPU readout copy, pixels, processor and encoder—including loop-local references—using `finally` cleanup and weakrefs. Verify all 448 encoder tensor bytes, config, processor and nonpersistent buffer facts. Restore the admitted CPU flags/RNG; rerun original CPU `integrity()` and payload fingerprint. Then release CPU state and run the original full `exit_rehash()` through a fresh uncached reader. Synchronize CUDA; never reset its peak.

The main pitfalls are typed identity conversion, CPU views accidentally copied into multiple retained states, graph references surviving between views, and calling trainer integrity while a registered encoder remains live. Original integrity explicitly requires model release.

The bounded stdlib test should check exact authority/initializer selection, first-micro16 routing, full-B64 denominator, all-positive/self-exclusion membership, four/444 roles, cleanup ordering, and rejection of stale hashes or a zero-gradient witness. I ran the read-only source/metadata portion: **PASS**, including `K=63`, denominator 126, stale-bank rejection, and retained API correspondence. Native payload loading, gradients and resource qualification remain **UNRUN**.

**The concrete resource dependency is the unchanged original admission plus complete uncached exit scan.** Their cost with this witness has not been measured, so source inspection cannot promise completion within 300 seconds. Preserve that bound. The historical cached ~101-second control is contextual evidence; the proposal’s unchanged ≤1.50 cost gate requires a fresh live frozen control.

This witness qualifies only actual-objective connectivity. It performs zero updates, discards all state, reads no quality, and leaves fresh TRAIN gated. The current `nearest_ranking_readout.py` keeps its input graph intact; do not attribute a detach fault to an older layer without binding the actual frozen run source.
