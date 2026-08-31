# Pass209 M4 Objective Rescue Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one authenticated, claim-ineligible objective-rescue receipt from the three preregistered fp32 frozen substrates, then combine it with terminal M1/M3 evidence to admit exactly one broad method family without test leakage.

**Architecture:** A focused `sfora` library module owns framed descriptor bytes, exact CPU scoring, strict evidence types, duplicate hashing, confusion-pair clustered bootstrap, and the decision adapter. A thin offline runner performs one frozen encoding cell and atomically publishes its three artifacts; a separate analyzer authenticates all three terminal cells plus the M2 manifest and recomputes every derived quantity. A deployment wrapper stages the reviewed source on DGX and enforces the registered resource/resume contract without inspecting partial scientific results.

**Tech Stack:** Python 3.12, PyTorch fp32, NumPy PCG64, Hugging Face Transformers/Datasets in offline mode, pytest, Ruff, mypy.

**Spec:** `docs/pass209_m4_objective_separability_protocol_2026-08-30.md`

## Global Constraints

- Operate only in `/home/rb/worktrees/sfora-emafactorial`; never touch Borsuk.
- Heavy model inference, builds, profiling, and scientific execution run on DGX `spark-2751`; local execution is limited to edits and focused lightweight tests.
- The only cells are DINOv2-L, SigLIP2-so400m, and SigLIP-so400m at their exact registered revisions and readouts; SigLIP-base is excluded.
- Cars train classes `82..97` are the only data visible to M4. Classes `49..81`, `98..195`, M1 clean metrics, and historical official-test embeddings are unavailable.
- Each cell uses batch 8, full fp32 CUDA descriptor inference on the registered DGX, query block 32, deterministic algorithms, TF32 disabled, and one-thread CPU reference scoring; receipts bind the complete GPU/driver/CUDA/cuDNN stack.
- Each cell publishes exactly one create-new authority receipt, framed descriptor plane, and ordered query-evidence table; partial output is never scientific evidence.
- DINOv2-L and SigLIP2-so400m are the only non-selecting rescue devices. SigLIP-so400m supplies the frozen error manifest and is descriptive only.
- The analyzer uses the exact 103-row manifest, primary pair `(82,83)`, 10,000 unordered-confusion-pair cluster resamples, a census materiality threshold, and the spec's first-match decision adapter.
- M4 cannot select a model, layer, loss, schedule, crop policy, checkpoint, or winning frozen device. It can admit only `cross-class-transfer`, `input-evidence-capacity`, or no family.
- Do not launch an M4 GPU cell until the sole active three-seed M1/M3 control releases the DGX.

---

### Task 1: Framed descriptor and canonical cell authority

**Files:**
- Create: `src/sfora/pass209_m4.py`
- Create: `tests/test_pass209_m4.py`

**Interfaces:**
- Produces: `M4CellSpec`, `M4DescriptorHeader`, `encode_descriptor_file(header, descriptors) -> bytes`, `decode_descriptor_file(payload) -> tuple[M4DescriptorHeader, torch.Tensor]`, `canonical_json_bytes(value) -> bytes`, and `publish_new_outputs(outputs) -> None`.

- [ ] **Step 1: Write the failing framed-codec test.** Require the exact 16-byte `b"SFORA-M4-F32-V1\n"` magic, little-endian `u64` header length, sorted compact JSON plus one LF, row-major little-endian f32 payload, and exact round trip.
- [ ] **Step 2: Run the single RED.**

  ```bash
  uv run --offline --locked pytest -q -p no:cacheprovider tests/test_pass209_m4.py::test_descriptor_codec_has_exact_framing_and_round_trips
  ```

  Expected: import failure because `sfora.pass209_m4` does not exist.
- [ ] **Step 3: Implement the minimal codec.** Parse with `struct.Struct("<Q")`, require exact header keys and concrete JSON types, and convert the payload through `numpy.frombuffer(..., dtype="<f4")` before copying into a contiguous `torch.float32` tensor.
- [ ] **Step 4: Add strict mutations.** Reject wrong magic, header order/whitespace/newline, short/long header, shape/byte arithmetic drift, padding, extra bytes, nonfinite rows, zero norms, norm error above `1e-6`, digest drift, and bool-as-int values.
- [ ] **Step 5: Implement atomic three-file publication.** Preflight all destinations and owned partials, create every partial with `xb`, fsync files, hard-link only after all writes succeed, fsync directories, and remove only owned partials/output links on failure.
- [ ] **Step 6: Run the codec tests GREEN and commit the slice.**

### Task 2: Exact scorer and ordered query evidence

**Files:**
- Modify: `src/sfora/pass209_m4.py`
- Modify: `tests/test_pass209_m4.py`

**Interfaces:**
- Produces: `configure_reference_scorer() -> ScorerEnvironment`, `score_descriptor_plane(descriptors, examples, *, block_size=32) -> tuple[QueryEvidence, ...]`, and `validate_query_evidence(...) -> None`.

- [ ] **Step 1: Write a failing hand-derived scorer test.** Use normalized f32 rows with an exact tie, signed zero, a same-label neighbor, and a different-label neighbor. Require exact `float32.view(torch.int32)` bit patterns and lowest-row tie resolution.
- [ ] **Step 2: Run the scorer RED and preserve the missing-interface failure.**
- [ ] **Step 3: Implement the scorer authority.** Set intra-op and inter-op threads to one before scoring, enable deterministic algorithms, reject CUDA tensors/autocast, use contiguous CPU f32 inputs and blocks of exactly 32, compute `query_block @ descriptors.T`, set diagonal to `-inf`, and use an explicit lexicographic row scan rather than backend-dependent top-k tie behavior.
- [ ] **Step 4: Bind the environment.** Record `torch.__version__`, `torch.__config__.show()`, BLAS identity/version, CPU architecture/ISA flags, thread counts, deterministic state, `uv.lock` SHA-256, Pillow/libjpeg and Transformers versions, scorer schema, and a fixed synthetic self-test digest. Fail before cell execution if the self-test differs.
- [ ] **Step 5: Add differential/mutation tests.** Compare the blocked scorer to a scalar f32 oracle for complete blocks and the ragged final one-row block; mutate score bits, row IDs, example IDs, labels, nearest/best-same/best-different rows, margins, tie ordering, table cardinality, and canonical bytes.
- [ ] **Step 6: Run the focused scorer nodes GREEN and commit.**

### Task 3: Objective rescue statistics

**Files:**
- Modify: `src/sfora/pass209_m4.py`
- Modify: `tests/test_pass209_m4.py`

**Interfaces:**
- Produces: `rgb_record_sha256(image) -> str`, `dominant_pair_rescuable(rows) -> bool`, `bootstrap_reachable(rows) -> BootstrapEvidence`, and `analyze_rescue(cells, manifest, examples) -> M4Evidence`.

- [ ] **Step 1: Write failing RGB and duplicate tests.** Hash exactly `b"SFORA-M4-RGB-V1\n" + struct.pack("<II", width, height) + rgb_bytes`; prove different dimensions, pixels, padding, or modes cannot alias and verify gallery-wide different-label matches rather than top-1-only matches.
- [ ] **Step 2: Write failing dominant-census tests.** Use exact edge cases `15/63`, `16/63`, and `63/63`; require the descriptive `>=.25` threshold, concrete integer counts, and no iid confidence interval or multiplicity field.
- [ ] **Step 3: Write failing clustered-bootstrap tests.** Construct lexicographically sorted nonempty unordered-pair blocks, duplicate each complete block per PCG64-v3 draw, keep both directions of `(82,83)` inseparable, and require exactly 10,000 little-endian f64 values plus NumPy `inverted_cdf` p2.5/p10/p97.5 and digest.
- [ ] **Step 4: Implement P1/P2/P3.** Derive `universal_three_device_error = not reachable`; keep selecting-device and minor-pair tables descriptive; compute `dominant_pair_rescuable` only from the two adjusted dominant-pair bounds.
- [ ] **Step 5: Add leakage and authority mutations.** Reject added devices/pairs, promoted descriptive results, any class outside `82..97`, incorrect 103-row manifest membership, nonzero SigLIP-so400m rescue, independent universal-error input, an iid interval field, wrong bootstrap seed/order/pair partition, and a p10 derived from the dominant panel rather than global reachability.
- [ ] **Step 6: Run statistics tests GREEN and commit.**

### Task 4: Strict offline frozen-cell runner

**Files:**
- Create: `scripts/run_pass209_m4_cell.py`
- Create: `tests/test_run_pass209_m4_cell.py`
- Modify: `src/sfora/pass209_m4.py`

**Interfaces:**
- Consumes: the three literal `M4CellSpec` entries and existing Cars loader/materializer.
- Produces: CLI `run_pass209_m4_cell.py --cell {dinov2-large,siglip2-so400m,siglip-so400m} --receipt PATH --descriptors PATH --queries PATH --execute`.

- [ ] **Step 1: Write a failing strict-CLI test.** Require all three new output paths, exact cell name, explicit `--execute`, offline environment, source revision/tree digest, prerequisite receipt, dataset sequence digest, batch 8, query block 32, and refusal of test/clean-class/model/layer/readout override flags.
- [ ] **Step 2: Run the CLI RED.**
- [ ] **Step 3: Implement authority loading before model loading.** Authenticate the prerequisite receipt SHA, the SigLIP-so400m v2 legacy descriptor digest authority (the two v1 receipts have none and must not be supplemented), source/tree, exact Cars revision and 1,345 example sequence, cell revision/readout, expected correct count, output absence, and scorer self-test.
- [ ] **Step 4: Implement frozen fp32 encoding.** Reuse the established RGB materializer and registered Transformers readouts, enforce processor shape, full fp32 tower/readout, unit normalization, deterministic CUDA, TF32 off, and bind GPU product/UUID/compute capability, driver, CUDA runtime/build, and cuDNN. Emit batch-boundary progress receipts and enforce the exact aggregate-count gate.
- [ ] **Step 5: Score on CPU and publish once.** Move only the final descriptor plane to CPU, release CUDA model/cache, invoke Task 2, build all three payloads in memory or owned partials, and atomically publish them together.
- [ ] **Step 6: Add fake-model integration tests.** Mutation-lock readout, revision, processor shape, count, batch ordering, resume identity, overwrite refusal, partial cleanup, and proof that no clean/test split is loaded.
- [ ] **Step 7: Run the runner tests, Ruff, mypy, py_compile, and commit.**

### Task 5: Three-cell analyzer and M1/M3 adapter

**Files:**
- Create: `scripts/analyze_pass209_m4.py`
- Create: `tests/test_analyze_pass209_m4.py`
- Modify: `src/sfora/pass209_m4.py`

**Interfaces:**
- Produces: `load_m4_cells(paths) -> tuple[M4Cell, M4Cell, M4Cell]`, `m4_receipt_bytes(...) -> bytes`, and `adapt_m3_m4(m3_receipts, m4_receipt) -> FamilyDecision`.

- [ ] **Step 1: Write a failing coherent nine-file analyzer test.** Use three small descriptor planes but literal production cell identities and a coherent manifest. Require full descriptor/query recomputation, P1/P2/P3, bootstrap, provenance, `claim_eligible=false`, and one trailing LF.
- [ ] **Step 2: Run the analyzer RED.**
- [ ] **Step 3: Implement strict all-input authentication.** Require three unique registered cells, every receipt/file cross-digest, exact source/data/example identity, expected aggregate counts, descriptor/query equality, manifest SHA, and absence of any partial input.
- [ ] **Step 4: Implement canonical result validation.** Recompute every count, overlap, census threshold, percentile, vector digest, and descriptive table before serialization; reject nonfinite values and derived-field drift. Require SigLIP-so400m's exact 103 incorrect query positions to match the source manifest and its rescue rate to be zero. Permit a different incorrect nearest row only as a published descriptive CPU/CUDA divergence with both row IDs, score bits, and margin; it cannot rewrite the source-manifest pair.
- [ ] **Step 5: Implement the first-match adapter.** Accept only three authenticated terminal M1/M3 seed receipts. Return `F4-TRANSFER` for T-low plus reachable p10 `>=.25`; `F4-CAPACITY` for T-high plus dominant-pair rescue; otherwise `F4-NONE`. Prove rule precedence, threshold equality, and no model/device identity escapes into the admitted family.
- [ ] **Step 6: Add mutation matrices.** Cover every role digest/key/type, device selection, pair selection, overlap, bootstrap, M3 seed/cardinality/ratio/terminal state, and prohibited class/model fields.
- [ ] **Step 7: Run analyzer tests GREEN and commit.**

### Task 6: DGX deployment and resource/resume envelope

**Files:**
- Create: `scripts/run_pass209_m4_objective_rescue_v1.sh`
- Create: `scripts/deploy_pass209_m4_objective_rescue_v1.sh`
- Create: `tests/test_run_pass209_m4_objective_rescue_v1.py`
- Create: `tests/test_deploy_pass209_m4_objective_rescue_v1.py`

**Interfaces:**
- Produces: one source-manifested DGX campaign that runs the three cells serially and the analyzer once, with no candidate training.

- [ ] **Step 1: Write failing shell-contract tests.** Require exact source revision/tree manifest, offline/cache preflight, unique create-new output root, serial cell order, no result inspection between cells, combined RSS plus CUDA-reserved 64-GiB stop, PSI `avg10 >= .50`, swap delta 256 MiB, 20-minute progress timeout, authenticated same-cell resume, and explicit process/temp clearance.
- [ ] **Step 2: Implement the remote runner.** Launch exactly one process group, retain its PID, sample pressure without overlapping jobs, terminate the group on a registered stop, and emit only an operational stop receipt until all cells are terminal.
- [ ] **Step 3: Implement deployment.** Build a content-addressed source archive from tracked Sfora files, transfer to DGX, verify digest before extraction, assert no active M1/M3 process before launch, and refuse an existing campaign output.
- [ ] **Step 4: Add failure-injection tests.** Cover missing model cache, source mismatch, extra output, stale partial, first/second/third cell failure, analyzer failure, resource stop, resumable checkpoint, and refusal to duplicate a live/original run.
- [ ] **Step 5: Mutation-lock terminal semantics.** An authority failure produces terminal `failed`, runs no adapter, admits no candidate, and has the operational `F4-NONE` consequence while permitting a separately preregistered future objective measurement.
- [ ] **Step 6: Run shell-focused tests and static checks GREEN; commit.**

### Task 7: Final assurance, review, and frozen delivery

**Files:**
- Modify only demonstrated defects in the files above.
- Track: `docs/pass209_m4_objective_separability_protocol_2026-08-30.md`
- Track: `docs/superpowers/plans/2026-08-31-pass209-m4-objective-rescue.md`

- [ ] **Step 1: Run focused tests.**

  ```bash
  uv run --offline --locked pytest -q -p no:cacheprovider \
    tests/test_pass209_m4.py \
    tests/test_run_pass209_m4_cell.py \
    tests/test_analyze_pass209_m4.py \
    tests/test_run_pass209_m4_objective_rescue_v1.py \
    tests/test_deploy_pass209_m4_objective_rescue_v1.py
  ```

- [ ] **Step 2: Run static assurance.**

  ```bash
  uv run --offline --locked ruff check src/sfora/pass209_m4.py scripts/run_pass209_m4_cell.py scripts/analyze_pass209_m4.py tests/test_pass209_m4.py tests/test_run_pass209_m4_cell.py tests/test_analyze_pass209_m4.py
  uv run --offline --locked mypy --strict src/sfora/pass209_m4.py scripts/run_pass209_m4_cell.py scripts/analyze_pass209_m4.py
  python3 -m py_compile src/sfora/pass209_m4.py scripts/run_pass209_m4_cell.py scripts/analyze_pass209_m4.py
  git diff --check
  ```

- [ ] **Step 3: Run the repository's full assurance command once on DGX.** Do not overlap it with the M1/M3 control or M4 inference. Preserve the original terminal and repair only demonstrated failures through focused RED/GREEN cycles before one final full rerun.
- [ ] **Step 4: Obtain a cold cross-provider review.** Require review of the final diff against the protocol, especially selecting-device exclusion, scorer authority, bootstrap, adapter precedence, resume behavior, and class-access surface.
- [ ] **Step 5: Commit and push verified repairs.** Stage only the named Sfora files, preserve configured identity, add no attribution trailer, push `HEAD:refs/heads/devbox/emafactorial`, and verify local/remote SHA equality and clean status except protected user-owned files.

### Task 8: Execute M4 and admit one method family

**Files:**
- Create only ignored generated evidence under the registered DGX output root and `reports/generated/`.

- [ ] **Step 1: Wait for terminal M1/M3 evidence.** Authenticate all three seed receipts and confirm no control process remains before occupying DGX.
- [ ] **Step 2: Launch exactly one reviewed M4 campaign.** Preserve original PID, pressure, batch progress, cell terminals, and analyzer terminal. Never inspect partial descriptors or overlap until all three cells are sealed.
- [ ] **Step 3: Verify the canonical M4 receipt independently.** Recompute all nine input digests, descriptor/query tables, P1/P2/P3, bootstrap digest/percentiles, and decision inputs.
- [ ] **Step 4: Run the adapter once.** Combine only terminal M1/M3 and M4 receipts. Freeze exactly one outcome: `F4-CAPACITY`, `F4-TRANSFER`, or `F4-NONE`.
- [ ] **Step 5: Preregister the corresponding Fable-supported method family before implementation.** Capacity admits BENR or its preregistered compute fallback ISP; transfer admits CGCF; none admits no candidate. The selected method still requires its own literature occupancy review, implementation plan, paired clean gate, and sealed official comparison.
