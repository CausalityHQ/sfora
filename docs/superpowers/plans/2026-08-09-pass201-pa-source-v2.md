# Pass201 Ordinary-PA Source v2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce one prospectively authorized, immutable ordinary Proxy Anchor seed-0 source whose provenance is strong enough to activate and run the frozen Pass201 CIS operator diagnostic.

**Architecture:** A pure contract module owns strict schemas, canonical encodings, Git/file/tree bindings, restricted checkpoint metadata, and non-clobber publication. A small controller has three modes: capture prospective authority twice through the real CLI boundary, run the one authorized training child, and derive two identical post-run sidecars before publishing a receipt last. A separate activation tuple commits the sole valid receipt before any ordinary report metric or Pass201 candidate value is read.

**Tech Stack:** Python 3.12, standard-library JSON/hashlib/os/zipfile/pickle/subprocess, Typer production CLI capture, Git object verification, pytest, Ruff, DGX CUDA training.

## Global Constraints

- Authoritative reviewed design: `docs/superpowers/specs/2026-08-09-pass201-pa-source-v2-design.md` at full commit `6d781b3`; implement its normative schemas and literals exactly.
- This work creates source authority only. No CIS/Pass201 operator tensor, context statistic, decision, candidate value, or report method metric may be read before activation.
- The ordinary PA source truthfully uses official In-Shop query/gallery evaluation; `candidate_values_computed=false` refers only to Pass201 values.
- Authorization is two commits: reviewed code commit `C`, followed by commit `A` adding only `docs/pass201_pa_source_v2_prelaunch.json`. `A` has exactly one parent `C`.
- The single attempt runs only from a new clean detached checkout at `A` on DGX `spark-2751`; its private mode-0700 run directory is absent beforehand.
- The first authorized attempt is either automatically activated if complete and valid or blocks. No second receipt, source selection, tolerance widening, or report-score inspection is allowed.
- Local tests use bounded fixtures and never materialize the full model/dataset. Real In-Shop capture, full import/image trees, training, and checkpoint validation run only on DGX.
- CPU/GPU wall time is operational evidence only. No scientific decision uses host timing.
- Strict TDD: observe a focused RED, implement minimal GREEN, run the fresh focused gate, then review. Never overlap tests or implementation subagents.
- Never touch `HANDOFF_BRIEF.md`, `RSPG_SPECDEFECT.md`, or `RSPG_TASK.md`.

## File Structure

- Create `scripts/pass201_pa_source_v2_contract.py`: pure schemas, strict JSON, framed hashing, Merkle/import-root bindings, Git topology checks, restricted checkpoint metadata, immutable file evidence, and non-clobber publication.
- Create `scripts/run_pass201_pa_source_v2.py`: `freeze-authority`, `run`, and private `derive-sidecars` orchestration only.
- Create `tests/test_pass201_pa_source_v2_contract.py`: pure/unit and malicious-input contract tests.
- Create `tests/test_run_pass201_pa_source_v2.py`: temp-Git, synthetic-In-Shop, production-capture, real child-process, failure-order, and CLI tests.
- Create later, never in `C`: `docs/pass201_pa_source_v2_prelaunch.json`.
- Create only after a successful receipt, in one later activation commit: `docs/pass201_pa_source_v2_activation.json`.
- Create for activation: `scripts/activate_pass201_pa_source_v2.py` and `tests/test_activate_pass201_pa_source_v2.py`.

---

### Task 1: Canonical Contract, File/Tree Authority, and Strict Schemas

**Files:**
- Create: `scripts/pass201_pa_source_v2_contract.py`
- Create: `tests/test_pass201_pa_source_v2_contract.py`

**Interfaces:**
- Produces `load_strict_json_bytes(data: bytes) -> dict[str, Any]` and `canonical_json_bytes(value: Any) -> bytes`.
- Produces immutable `RepoBlob`, `ExternalFileBinding`, `MerkleBinding`, `ImportRootBinding`, `OutputEvidence`, `PrelaunchAuthority`, and `CompleteReceipt` dataclasses.
- Produces `bind_external_file(path: Path) -> ExternalFileBinding`, `bind_repo_blob(repo: Path, revision: str, path: PurePosixPath) -> RepoBlob`, `bind_merkle(root: Path) -> MerkleBinding`, and `bind_import_roots(interpreter: Path, env: Mapping[str, str], checkout: Path) -> tuple[ImportRootBinding, ...]`.
- Produces `validate_prelaunch(payload: object) -> PrelaunchAuthority` and `validate_complete_receipt(payload: object, authority: PrelaunchAuthority) -> CompleteReceipt`.

  Define the immutable public records with these exact Python fields; nested strict JSON remains in `payload` only after full validation:

  ```python
  @dataclass(frozen=True)
  class RepoBlob:
      path: PurePosixPath
      git_mode: Literal["100644", "100755"]
      byte_count: int
      sha256: str
      git_blob: str

  @dataclass(frozen=True)
  class ExternalFileBinding:
      path: Path
      mode: int
      device: int
      inode: int
      byte_count: int
      sha256: str

  @dataclass(frozen=True)
  class MerkleBinding:
      root: Path
      algorithm: Literal["pass201-length-framed-merkle-v1"]
      count: int
      byte_count: int
      root_sha256: str

  @dataclass(frozen=True)
  class ImportTreeBinding:
      algorithm: Literal["pass201-import-tree-v1"]
      regular_count: int
      symlink_count: int
      byte_count: int
      root_sha256: str

  @dataclass(frozen=True)
  class ExternalFileImportTarget:
      link_relative_path: PurePosixPath
      target_text: str
      resolved_path: Path
      kind: Literal["file"]
      file: ExternalFileBinding

  @dataclass(frozen=True)
  class ImportDirectoryBinding:
      root: Path
      tree: ImportTreeBinding
      external_symlink_targets: tuple[
          "ExternalFileImportTarget | ExternalDirectoryImportTarget", ...
      ]

  @dataclass(frozen=True)
  class ExternalDirectoryImportTarget:
      link_relative_path: PurePosixPath
      target_text: str
      resolved_path: Path
      kind: Literal["directory"]
      directory: ImportDirectoryBinding

  @dataclass(frozen=True)
  class NonexistentImportRoot:
      entry: str
      status: Literal["nonexistent"] = "nonexistent"

  @dataclass(frozen=True)
  class ZipImportRoot:
      entry: str
      file: ExternalFileBinding
      status: Literal["zip"] = "zip"

  @dataclass(frozen=True)
  class DirectoryImportRoot:
      entry: str
      directory: ImportDirectoryBinding
      status: Literal["directory"] = "directory"

  ImportRootBinding: TypeAlias = NonexistentImportRoot | ZipImportRoot | DirectoryImportRoot

  @dataclass(frozen=True)
  class PrelaunchAuthority:
      payload: Mapping[str, Any]
      source_commit: str
      checkout_root: Path
      expected_config_bytes: bytes
      expected_config_sha256: str
      expected_train_steps: int
      steps_per_epoch: int
      total_epochs: int

  @dataclass(frozen=True)
  class CompleteReceipt:
      payload: Mapping[str, Any]
      authorization_commit: str
      output_evidence: Mapping[str, "OutputEvidence"]
  ```

- [ ] **Step 1: Write strict JSON/schema RED tests**

  Add tests that independently inject duplicate keys, NaN/Infinity, Boolean-as-integer, float-as-integer, uppercase/short hashes, absolute repository paths, relative external paths, missing/extra nested keys, false success literals, a third sidecar child, wrong algorithm IDs, and open-ended absence/input-hash keys. Assert rejection before any file/Git/model sentinel.

  ```python
  @pytest.mark.parametrize("raw", [b'{"x":1,"x":2}', b'{"x":NaN}'])
  def test_strict_json_rejects_ambiguous_bytes(raw: bytes) -> None:
      with pytest.raises(ValueError):
          load_strict_json_bytes(raw)

  def test_prelaunch_rejects_false_required_predicate(valid_prelaunch: dict[str, Any]) -> None:
      valid_prelaunch["postconditions"]["require_source_equal"] = False
      with pytest.raises(ValueError, match="require_source_equal"):
          validate_prelaunch(valid_prelaunch)
  ```

- [ ] **Step 2: Run the schema RED**

  Run: `.venv/bin/pytest -q tests/test_pass201_pa_source_v2_contract.py -k 'strict_json or schema'`

  Expected: collection/import fails because the contract module/interfaces do not exist; fixture setup itself succeeds.

- [ ] **Step 3: Implement canonical JSON and exact schema validation**

  Use `json.loads(..., parse_constant=reject, object_pairs_hook=reject_duplicate_keys)`. Recursively enforce exact JSON types (`type(x) is int`, never `isinstance(x, int)`), finite numbers, exact key sets, literal success values, exact two-child ordinals/PID distinctness, exact six absence keys, and exact eight input-hash keys. Emit sorted compact UTF-8 JSON with one newline.

  ```python
  def canonical_json_bytes(value: Any) -> bytes:
      validate_json_native(value)
      return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")

  def load_strict_json_bytes(data: bytes) -> dict[str, Any]:
      value = json.loads(data.decode("utf-8"), parse_constant=reject_constant,
                         object_pairs_hook=unique_object)
      require_exact_json_tree(value)
      return require_dict(value)
  ```

- [ ] **Step 4: Write descriptor-safe file and Merkle RED tests**

  Cover `O_NOFOLLOW`, regular-file type, device/inode/size/mtime stability, path escape, duplicate normalized names, symlink rejection for ordinary Merkle, raw UTF-8 ordering, exact u64 framing, mutation during read, and a known hand-computed three-leaf digest.

  Run: `.venv/bin/pytest -q tests/test_pass201_pa_source_v2_contract.py -k 'external_file or ordinary_merkle'`

  Expected: fail on missing binding functions.

- [ ] **Step 5: Implement file and ordinary Merkle bindings**

  Stream from already-open descriptors. Re-stat before/after. Traverse without following links and encode exact `{relative_path,bytes,sha256}` leaves using the design’s `leaf\0`/`node\0` domain separation and odd-node duplication.

  ```python
  def bind_merkle(root: Path) -> MerkleBinding:
      leaves = tuple(sorted(iter_regular_leaves(root), key=lambda x: x.relative_utf8))
      hashes = [sha256(b"leaf\0" + frame(canonical_json_bytes(x.payload))).digest()
                for x in leaves]
      return MerkleBinding(root=root.resolve(strict=True), count=len(leaves),
                           byte_count=sum(x.byte_count for x in leaves),
                           root_sha256=reduce_merkle(hashes).hex())
  ```

  Run the same focused command and require PASS.

- [ ] **Step 6: Write import-root RED tests**

  Build an interpreter-path fixture containing a real directory, internal symlink, external file symlink, external directory symlink, ZIP path, nonexistent path, `.pth`, and `sitecustomize`. Assert ordered tagged records, target-string binding, external-target recursion, cycle rejection, checkout/cwd exclusion, pre-existing `.pyc` inclusion, and changed-target detection.

  Run: `.venv/bin/pytest -q tests/test_pass201_pa_source_v2_contract.py -k 'import_root'`

  Expected: fail on missing tagged import-root implementation.

- [ ] **Step 7: Implement import-root binding**

  Obtain effective `sys.path` from a fresh bound interpreter child under the replacement environment. Preserve order; use explicit `nonexistent`, `zip`, and `directory` tags. Do not follow symlinks during directory traversal; encode links as leaves and recursively bind every external resolved target.

  ```python
  def bind_import_roots(interpreter: Path, env: Mapping[str, str],
                        checkout: Path) -> tuple[ImportRootBinding, ...]:
      entries = query_effective_sys_path(interpreter, env)
      return tuple(bind_import_entry(entry, checkout=checkout,
                                     active_realpaths=frozenset())
                   for entry in entries if not is_bound_checkout_entry(entry, checkout))
  ```

  Run the same focused command and require PASS.

- [ ] **Step 8: Write strict Git topology RED tests**

  In a real temporary repository, require detached `A`, exactly one parent `C`, exactly one `A 100644 docs/pass201_pa_source_v2_prelaunch.json` change, manifest bytes equal `A:path`, exact repo blob SHA/content, and empty `git status --porcelain=v1 -z`. Bind and verify the resolved Git executable itself. Test extra parent/change, executable mode, dirty/untracked path, symbolic branch, stale blob, and substituted Git failures.

- [ ] **Step 9: Run Git RED**

  Run: `.venv/bin/pytest -q tests/test_pass201_pa_source_v2_contract.py -k 'git_topology'`

  Expected: fail on missing `validate_authorization_topology`.

- [ ] **Step 10: Implement Git topology and run GREEN**

  ```python
  def validate_authorization_topology(repo: Path, authority: PrelaunchAuthority) -> str:
      head = git_text(repo, "rev-parse", "HEAD")
      require_detached(repo)
      require_exact_parent(repo, head, authority.source_commit)
      require_single_manifest_addition(repo, authority.source_commit, head)
      require_empty_porcelain(repo)
      return head
  ```

  Run the focused `git_topology` tests and require PASS.

- [ ] **Step 11: Verify and commit Task 1**

  Run the focused test file serially, Ruff on both new files, `python -m py_compile scripts/pass201_pa_source_v2_contract.py`, and `git diff --check`. Commit only the contract/test with message `implement Pass201 source v2 contracts` and write the ignored SDD task report.

---

### Task 2: Restricted Checkpoint Metadata and Non-Clobber Publication

**Files:**
- Modify: `scripts/pass201_pa_source_v2_contract.py`
- Modify: `tests/test_pass201_pa_source_v2_contract.py`

**Interfaces:**
- Produces `read_restricted_checkpoint_metadata(path: Path, authority: PrelaunchAuthority) -> CheckpointMetadata` without importing torch or reading any storage member.
- Produces `publish_new_file(path: Path, data: bytes, *, mode: int = 0o444) -> OutputEvidence` and `hash_open_regular(path: Path) -> OutputEvidence`.

  ```python
  @dataclass(frozen=True)
  class CheckpointArch:
      backbone_name: Literal["bn_inception"]
      pretrained_weights: Literal["bn_inception_52deb4733"]
      head_pooling: Literal["avg_max"]
      embedding_dimensions: Literal[512]
      embedding_head_init: Literal["kaiming_normal"]
      embedding_layer_norm: Literal[False]

  @dataclass(frozen=True)
  class CheckpointMetadata:
      data_pickle_sha256: str
      top_keys: tuple[str, ...]
      artifact_selection: Literal["final_training_state"]
      evaluation_model_source: Literal["student"]
      training_step: int
      arch: CheckpointArch
      arch_sha256: str
      training_config_sha256: str
      state_dict_key_count: int
      state_dict_storage_materialized: Literal[False] = False

  @dataclass(frozen=True)
  class OutputEvidence:
      path: PurePosixPath
      file_type: Literal["regular"]
      mode: int
      byte_count: int
      sha256: str
  ```

- [ ] **Step 1: Write safe/malicious checkpoint RED tests**

  Generate tiny ZIP fixtures with one valid `*/data.pkl` and inert storage references. Independently test duplicate members, traversal names, encryption, >100,000 members, oversized archive/metadata declarations, zero/two `data.pkl`, trailing pickle value, extension opcode, forbidden global, malformed persistent ID/rebuild, tensor-storage member open sentinel, wrong six-key set, wrong config types, non-final state, EMA source, architecture/step mismatch, and `torch` import sentinel.

  ```python
  def test_checkpoint_reader_never_opens_storage(valid_checkpoint_zip: Path,
                                                 monkeypatch: pytest.MonkeyPatch) -> None:
      opened: list[str] = []
      monkeypatch.setattr(zipfile.ZipFile, "open", recording_zip_open(opened))
      metadata = read_restricted_checkpoint_metadata(valid_checkpoint_zip, authority())
      assert metadata.state_dict_storage_materialized is False
      assert not any("/data/" in name for name in opened)

  def test_checkpoint_rejects_forbidden_global(tmp_path: Path) -> None:
      checkpoint = write_pickle_zip(tmp_path, reduce_to_global("os", "system"))
      with pytest.raises(ValueError, match="forbidden pickle global"):
          read_restricted_checkpoint_metadata(checkpoint, authority())
  ```

- [ ] **Step 2: Run the checkpoint RED**

  Run: `.venv/bin/pytest -q tests/test_pass201_pa_source_v2_contract.py -k 'checkpoint or pickle or storage'`

- [ ] **Step 3: Implement restricted metadata decoding**

  Enforce the design’s literal global allowlist through controller-owned inert stubs. Accept only exact `("storage", storage_stub, ascii_key, location, nonnegative_int_size)` IDs. Read only the central directory and unique `data.pkl`; never open `data/`. Require sorted top keys, exact scalar literals, exact architecture, prebound step, and canonical training config bytes. Record `state_dict_key_count`, `data_pickle_sha256`, and `state_dict_storage_materialized=false`.

  ```python
  def read_restricted_checkpoint_metadata(
      path: Path, authority: PrelaunchAuthority
  ) -> CheckpointMetadata:
      with open_regular_no_follow(path) as checkpoint:
          with zipfile.ZipFile(checkpoint) as archive:
              data_pickle = validate_and_read_unique_metadata(archive, max_members=100_000,
                                                              max_bytes=64 << 20)
      root = RestrictedMetadataUnpickler(io.BytesIO(data_pickle)).load_one()
      return validate_checkpoint_root(root, authority, sha256_bytes(data_pickle))
  ```

- [ ] **Step 4: Write atomic publication RED tests**

  Test pre-existing destination, symlink destination/parent, temp write/fsync/link/directory-fsync failures, content mutation, and concurrent creator. Assert no replacement, stale-success receipt, or accepted partial output. Verify report/checkpoint/log descriptor hashing rejects mutation and changes permissions read-only only after a successful hash.

  Run: `.venv/bin/pytest -q tests/test_pass201_pa_source_v2_contract.py -k 'publication or output_evidence'`

  Expected: fail on missing exclusive publication behavior.

- [ ] **Step 5: Implement publication and evidence**

  Write/fsync a mode-0600 sibling temporary, hard-link to the absent destination, fsync the parent, unlink the temporary, reopen with `O_NOFOLLOW`, verify regular identity/content, then chmod read-only where required. Receipt publication is a separate call performed last by Task 3.

  ```python
  def publish_new_file(path: Path, data: bytes, *, mode: int = 0o444) -> OutputEvidence:
      temp = create_exclusive_sibling(path, 0o600)
      try:
          write_all_and_fsync(temp, data)
          os.link(temp.name, path, follow_symlinks=False)
          fsync_directory(path.parent)
      finally:
          unlink_if_present(temp.name)
      os.chmod(path, mode, follow_symlinks=False)
      return hash_open_regular(path)
  ```

  Run the same focused command and require PASS.

- [ ] **Step 6: Verify and commit Task 2**

  Run the full contract test file once, Ruff, `py_compile`, and `git diff --check`. Commit only contract/test with message `bind Pass201 source checkpoint metadata` and append the SDD report.

---

### Task 3: Production CLI Capture and Deterministic Sidecars

**Files:**
- Create: `scripts/run_pass201_pa_source_v2.py`
- Create: `tests/test_run_pass201_pa_source_v2.py`
- Modify: `scripts/pass201_pa_source_v2_contract.py`
- Modify: `tests/test_pass201_pa_source_v2_contract.py`

**Interfaces:**
- Produces `capture_authority(argv: Sequence[str], dataset_root: Path) -> CapturedAuthority`.
- Produces `derive_resolved_config(report_bytes: bytes, checkpoint: CheckpointMetadata, authority: PrelaunchAuthority) -> bytes`.
- Produces `derive_train_manifest(capture: CapturedAuthority, authority: PrelaunchAuthority) -> bytes`.
- CLI private mode: `derive-sidecars --manifest PATH --report PATH --checkpoint PATH --output-dir PATH` emits canonical config/manifest bytes to stdout framing only; it cannot train or publish.

  ```python
  @dataclass(frozen=True)
  class CapturedAuthority:
      config_bytes: bytes
      recipe_id: str
      recipe_digest: str
      train_count: int
      query_count: int
      gallery_count: int
      protocol: str
      protocol_name: str
      rows: tuple[tuple[int, str, int], ...]
      resolved_membership_sha256: str
      resolved_train_steps: int
      steps_per_epoch: int
      total_epochs: int
  ```

- [ ] **Step 1: Write production-capture RED integration test**

  Create a real tiny In-Shop tree and partition. Invoke the real Typer app with frozen-shape argv after monkeypatching only `sfora.cli.run_image_end_to_end_benchmark`. The capture function must receive real train/query/gallery lists and final `ImageEndToEndConfig`, call the exact split/noise/schedule suffix, emit authority, and raise the controller sentinel before model factory/report writer. Assert exactly one boundary call, no output/temp path, and genuine In-Shop-like `example_id` is never interpreted as a path.

  ```python
  def test_capture_uses_real_cli_boundary_without_training(tiny_inshop: Path,
                                                           monkeypatch: pytest.MonkeyPatch) -> None:
      monkeypatch.setattr(image_end_to_end, "build_image_encoder", forbidden("model"))
      captured = capture_authority(frozen_test_argv(tiny_inshop), tiny_inshop)
      assert captured.protocol == "query_gallery"
      assert captured.rows == tuple((i, row.example_id, int(row.label))
                                    for i, row in enumerate(expected_optimization_rows(tiny_inshop)))
      assert not declared_output_paths(tiny_inshop).intersection(existing_paths(tiny_inshop))
  ```

- [ ] **Step 2: Run capture RED**

  Run: `.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py -k 'production_capture'`.

  Expected: fail on missing `capture_authority`; no model/report sentinel fires.

- [ ] **Step 3: Implement capture and run GREEN**

  Implement a private capture exception deriving directly from `BaseException`. Call the exact named production functions and construct contiguous zero-based rows. Resolve each `ImageExample.image` strictly under the bound resolved image root and hash path/content membership without reading report metrics.

  ```python
  def capture_authority(argv: Sequence[str], dataset_root: Path) -> CapturedAuthority:
      captured: list[CapturedAuthority] = []
      with patch.object(sfora.cli, "run_image_end_to_end_benchmark",
                        capture_then_raise(captured, dataset_root)):
          invoke_real_typer(argv)
      require(len(captured) == 1, "production boundary count")
      return captured[0]
  ```

  Run the focused `production_capture` tests and require PASS.

- [ ] **Step 4: Add two-process identity and expected-config tests**

  Spawn two fresh capture children. Assert byte-identical config, bundle counts/protocol, recipe ID/digest, rows, membership, and schedule. Mutate environment, recipe blob, partition, label mapping, split/noise parameter, schedule, and JSON scalar type; assert fail closed. Require exact In-Shop operating point: ordinary `proxy_anchor`, seed integer `0`, one proxy/class, BN-Inception, batch 180, drop-last, 60 epochs, official full partition, and the design’s prebound recipe digest.

- [ ] **Step 5: Implement resolved-config derivation**

  Strict-parse report bytes, canonicalize exactly `report.config`, and require byte identity with the prospective config and restricted checkpoint config. Do not access `report.methods` or any metric key. Test a mapping wrapper that raises if those keys are touched.

  ```python
  def derive_resolved_config(report_bytes: bytes, checkpoint: CheckpointMetadata,
                             authority: PrelaunchAuthority) -> bytes:
      report = load_strict_json_bytes(report_bytes)
      require_exact_keys(report, EXPECTED_REPORT_KEYS)
      config = canonical_json_bytes(report["config"])
      require(config == authority.expected_config_bytes, "report config drift")
      require(sha256_bytes(config) == checkpoint.training_config_sha256,
              "checkpoint config drift")
      return config
  ```

- [ ] **Step 6: Implement exact train-manifest schema**

  Emit only the design’s six top-level keys and exact nested keys. Include literal ordered call graph, sorted `FileBinding` source list, exact dataset authority, rows, row/identity counts, ordered-row hash, and resolved-membership hash. Test every nested missing/extra/type mutation and the exact membership of `Img -> img` and resolved `img/img`.

  ```python
  def derive_train_manifest(capture: CapturedAuthority,
                            authority: PrelaunchAuthority) -> bytes:
      payload = build_exact_train_manifest_payload(capture, authority)
      validate_train_manifest(payload, authority)
      return canonical_json_bytes(payload)
  ```

- [ ] **Step 7: Implement private derive-sidecars mode and child framing**

  Give each child immutable manifest/report/checkpoint inputs, return length-prefixed config and manifest bytes plus hashes, and forbid alternative paths/algorithms. Require two distinct child PIDs and exact byte identity before publication.

- [ ] **Step 8: Write schedule/epoch RED tests**

  Capture the literal `_resolve_training_schedule` tuple and test `steps_per_epoch > 0`, `drop_last_train_batch is True`, `steps_per_epoch == max(1, optimization_count // batch_size)`, checkpoint `training_step` equality to `resolved_train_steps`, exact divisibility, and one-based `training_step // steps_per_epoch == total_epochs`. Mutate each term and require failure before sidecar/receipt publication. Do not read a report method value to obtain an executed-step field; those values remain forbidden before activation.

- [ ] **Step 9: Implement schedule binding and run GREEN**

  Store the exact prospective tuple in `A`; require the two capture children, restricted checkpoint metadata, derived sidecars, and receipt to match it. Run: `.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py -k 'schedule or completed_epoch'`.

  ```python
  def validate_completed_epoch(capture: CapturedAuthority, checkpoint_step: int,
                               optimization_count: int,
                               batch_size: int) -> int:
      require(capture.steps_per_epoch == max(1, optimization_count // batch_size),
              "drop-last schedule drift")
      require(checkpoint_step == capture.resolved_train_steps,
              "training step drift")
      require(checkpoint_step % capture.steps_per_epoch == 0, "partial epoch")
      completed = checkpoint_step // capture.steps_per_epoch
      require(completed == capture.total_epochs, "completed epoch drift")
      return completed
  ```

- [ ] **Step 10: Verify and commit Task 3**

  Run both focused files serially, Ruff, `py_compile` for both scripts, and `git diff --check`. Commit only the four task files with message `capture Pass201 source v2 authority` and append the SDD report.

---

### Task 4: One-Shot Training Controller and Complete Receipt

**Files:**
- Modify: `scripts/run_pass201_pa_source_v2.py`
- Modify: `tests/test_run_pass201_pa_source_v2.py`

**Interfaces:**
- CLI public modes:
  - `freeze-authority --frozen-absence-checked-utc <RFC3339-UTC> --output docs/pass201_pa_source_v2_prelaunch.json`
  - `run --manifest docs/pass201_pa_source_v2_prelaunch.json`
- Produces the six private-run-directory outputs, with `receipt.json` published last.

  ```python
  @dataclass(frozen=True)
  class FreezeArgs:
      checkout_root: Path
      dataset_root: Path
      python_path: Path
      frozen_absence_checked_utc: str
      output_path: Path

  @dataclass(frozen=True)
  class RunningChild:
      process: subprocess.Popen[bytes]
      started_utc: str

      @property
      def pid(self) -> int:
          return self.process.pid

  @dataclass(frozen=True)
  class CompletedChild:
      pid: int
      started_utc: str
      ended_utc: str
      returncode: Literal[0]
  ```

- [ ] **Step 1: Write controller ordering RED test**

  Use a real child script and an event ledger. Require order: strict manifest -> detached exact Git topology -> replacement runtime/interpreter/import/source/pretrained/dataset bindings -> frozen/preflight absence -> create mode-0700 run dir/lock -> one child -> exit zero -> postflight equality -> hash/freeze report/checkpoint/log -> restricted metadata -> two sidecar children -> publish config/manifest -> publish receipt last. Every injected failure stops before the next phase and cannot produce a complete receipt.

  ```python
  def test_controller_publishes_receipt_last(controller_fixture: ControllerFixture) -> None:
      controller_fixture.run()
      assert controller_fixture.events == EXPECTED_SUCCESS_EVENT_ORDER
      assert controller_fixture.publish_order[-1] == "receipt.json"

  @pytest.mark.parametrize("failure_event", EXPECTED_SUCCESS_EVENT_ORDER[:-1])
  def test_controller_failure_never_publishes_receipt(controller_fixture: ControllerFixture,
                                                       failure_event: str) -> None:
      controller_fixture.fail_at(failure_event)
      with pytest.raises(RuntimeError):
          controller_fixture.run()
      assert not controller_fixture.receipt.exists()
  ```

  Run: `.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py -k 'controller_order'`

  Expected: fail because the controller phases do not exist.

- [ ] **Step 2: Implement freeze-authority mode**

  Resolve and bind controller/source blobs, Python/Git/interpreter/import roots, `pyproject.toml`, `uv.lock`, canonical package list, pretrained checkpoint, partition, literal resolved image root/tree and `Img -> img` link, exact command/replacement environment, two capture children, config/recipe/bundle/rows/membership/schedule, private output paths, and timestamped `ENOENT` evidence. Emit canonical manifest bytes only; never train or read candidate paths.

  ```python
  def freeze_authority(args: FreezeArgs) -> bytes:
      first = run_capture_child(args)
      second = run_capture_child(args)
      require(first == second, "capture children disagree")
      payload = build_prelaunch_payload(args, first, bind_runtime(args))
      validate_prelaunch(payload)
      return canonical_json_bytes(payload)
  ```

  Run: `.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py -k 'freeze_authority or controller_order'`

  Expected: freeze-focused cases pass; launch/postflight ordering remains RED until their steps.

- [ ] **Step 3: Write exact C-to-A and environment RED tests**

  In temp Git, prove freeze output validates only after it is the sole added file in child `A`; run mode rejects branch HEAD, extra diff/parent, ambient env leak, changed interpreter/Git/import root/pretrained/data/source, false query/gallery scope, existing run dir, and any output/temp collision.

  Run: `.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py -k 'run_preflight or replacement_environment'`

  Expected: fail before a child is launched.

- [ ] **Step 4: Implement run preflight and child launch**

  Use the bound absolute Python with exact `-m sfora.cli image-end-to-end ...` argv and complete replacement environment including `PYTHONDONTWRITEBYTECODE=1`. Hold the directory lock; open log exclusively; invoke exactly once; record PID/timestamps/return code. Do not shell-expand, retry, or accept extra arguments.

  ```python
  def run_authorized_source(manifest_path: Path) -> None:
      authority = validate_runtime_preflight(manifest_path)
      with create_and_lock_private_run_directory(authority) as run_dir:
          running = launch_once(authority, run_dir)
          returncode = running.process.wait()
          require(returncode == 0, "ordinary PA child failed")
          completed = CompletedChild(pid=running.pid,
                                     started_utc=running.started_utc,
                                     ended_utc=utc_now_rfc3339(),
                                     returncode=0)
          publish_postflight(authority, completed, run_dir)
  ```

  Run the same focused preflight/environment command and require PASS.

- [ ] **Step 5: Write postflight/receipt RED tests**

  Mutate every source/data/runtime/pretrained binding, report/config/checkpoint scalar, output type/hash, child sidecar, membership, algorithm literal, receipt success flag, and scope flag. Assert receipt absent or invalid. Test ordinary report method-map access sentinel remains zero.

  Run: `.venv/bin/pytest -q tests/test_run_pass201_pa_source_v2.py -k 'postflight or complete_receipt'`

  Expected: fail on missing receipt construction/publication.

- [ ] **Step 6: Implement postflight and receipt**

  Rebind every authority, require exact equality, descriptor-hash outputs, read restricted checkpoint metadata, derive sidecars twice, publish both, then assemble the exact complete receipt. Set truthful scope fields and publish receipt last. The receipt never hashes itself and records no report metric.

  ```python
  def publish_postflight(authority: PrelaunchAuthority, process: CompletedChild,
                         run_dir: Path) -> None:
      post = require_pre_post_identity(authority, bind_runtime_after(authority))
      scientific = freeze_scientific_outputs(run_dir)
      metadata = read_restricted_checkpoint_metadata(scientific.checkpoint, authority)
      sidecars = require_two_identical_sidecar_children(authority, scientific, metadata)
      publish_new_file(run_dir / "resolved_config.json", sidecars.config)
      publish_new_file(run_dir / "train_manifest.json", sidecars.manifest)
      receipt = build_and_validate_receipt(authority, process, post, scientific,
                                           metadata, sidecars)
      publish_new_file(run_dir / "receipt.json", canonical_json_bytes(receipt))
  ```

  Run the same focused postflight/receipt command and require PASS, then rerun
  `controller_order` and require the complete ordering suite PASS.

- [ ] **Step 7: Run final local assurance and independent review**

  Run both test files serially once, Ruff, `py_compile`, and `git diff --check`. Request a cold reviewer against the committed design; fix Critical/Important findings with separate RED/GREEN rounds. Commit final controller/test changes as `run authenticated Pass201 PA source v2`. This resulting reviewed commit is `C`.

---

### Task 5: Freeze Authorization Commit A and Launch on DGX

**Files:**
- Create: `docs/pass201_pa_source_v2_prelaunch.json`
- Remote-only outputs: `reports/generated/pass201_source_v2/run-v2/{report.json,checkpoint.pt,training.log,resolved_config.json,train_manifest.json,receipt.json}`

**Interfaces:**
- Consumes reviewed `C` and the DGX dataset/runtime.
- Produces authorization `A` and exactly one immutable complete source attempt.

- [ ] **Step 1: Prepare the final bound DGX checkout at C and confirm queue**

  Create the one final isolated checkout at the literal path that will be committed in `A`; fetch and detach it at `C`. Verify hostname `spark-2751`, no GPU compute process/controller/watcher, exact dataset/image/pretrained/runtime availability, empty porcelain status, and absent private run directory. Do not use the dirty historical worktree and do not move/copy the checkout after authority capture.

- [ ] **Step 2: Generate prelaunch bytes twice in that exact checkout**

  Compute one RFC3339 UTC value operationally, then run the exact command twice in fresh processes from `C` at the final bound path:

  ```text
  <bound-python> scripts/run_pass201_pa_source_v2.py freeze-authority \
    --frozen-absence-checked-utc <same-value-both-times> \
    --output <two-distinct-temporary-output-paths>
  ```

  Require byte identity, then exclusively move the accepted canonical bytes to `docs/pass201_pa_source_v2_prelaunch.json` in that checkout. Its absolute checkout/cwd/import/runtime bindings therefore remain true after the next detached commit.

- [ ] **Step 3: Commit A in place and review the edge**

  While detached in the same DGX checkout, commit only `docs/pass201_pa_source_v2_prelaunch.json` with message `freeze Pass201 PA source v2 launch`. Verify `A` has one parent exactly `C`, `git diff-tree C A` has the sole 100644 addition, the blob equals current bytes, the checkout path is unchanged, and porcelain is empty. Independently review `C..A` and the live runtime bindings, then push detached `HEAD` to `origin/devbox/emafactorial`. Fast-forward the local branch to the pushed `A`; do not create a second execution checkout.

- [ ] **Step 4: Launch exactly once from detached A**

  In the same bound DGX checkout now detached at `A`, reconfirm exact realpath, empty porcelain status, unchanged runtime/import/data bindings, and absent run directory, then start the single controller through the background execution mechanism. Retain its PID/session and bounded log; never start a duplicate. Poll the same job at intervals no longer than 55 seconds while doing independent work.

- [ ] **Step 5: Collect exit and validate receipt structurally**

  Require controller exit zero and a complete receipt. Validate all pre/post bindings, output hashes, restricted checkpoint metadata, two-child identity, truthful query/gallery scope, and `pass201_candidate_paths_read=false`. Do not parse `report.methods` or any metric value.

---

### Task 6: Non-Selective Activation and Pass201 Handoff

**Files:**
- Create: `docs/pass201_pa_source_v2_activation.json`
- Create: `scripts/activate_pass201_pa_source_v2.py`
- Create: `tests/test_activate_pass201_pa_source_v2.py`

**Interfaces:**
- Produces one committed activation tuple binding `A`, its manifest blob/hash, the sole receipt hash, and five output evidences.
- Supplies immutable source-v2 authority to the separate Pass201 Task3 repair; it does not itself load checkpoint tensors or score CIS.
- Produces `ActivationFailure(reason: Literal["BLOCKED_SOURCE_ARTIFACT_UNAVAILABLE", "INVALID_OPERATING_POINT_MISMATCH"])`, `build_activation(authority_path: Path) -> bytes`, and `write_activation(authority_path: Path, output_path: Path) -> OutputEvidence`.
- CLI command is exactly `activate --authority docs/pass201_pa_source_v2_prelaunch.json --output docs/pass201_pa_source_v2_activation.json`; it rejects extra/alternate modes and an existing output.

  ```python
  class ActivationReason(StrEnum):
      BLOCKED = "BLOCKED_SOURCE_ARTIFACT_UNAVAILABLE"
      INVALID = "INVALID_OPERATING_POINT_MISMATCH"

  @dataclass(frozen=True)
  class ActivationFailure(Exception):
      reason: ActivationReason
      detail: str
  ```

- [ ] **Step 1: Write activation RED tests**

  Test the exact activation schema and automatic policy. Missing authenticated receipt/report/checkpoint/log/config/manifest is `BLOCKED_SOURCE_ARTIFACT_UNAVAILABLE`. A present malformed/mismatched manifest, receipt, sidecar, config, checkpoint, source, derivation, or output is `INVALID_OPERATING_POINT_MISMATCH`. Use an open/JSON denylist proving `report.methods`, ordinary metrics, CIS paths, and Pass201 result paths remain unread. A second receipt or `attempt_ordinal != 1` must be invalid.

  ```python
  def test_missing_authenticated_source_is_blocked(activation_fixture: ActivationFixture) -> None:
      activation_fixture.receipt.unlink()
      with pytest.raises(ActivationFailure) as raised:
          build_activation(activation_fixture.authority)
      assert raised.value.reason is ActivationReason.BLOCKED

  def test_present_mismatch_is_invalid_without_metric_read(
      activation_fixture: ActivationFixture, monkeypatch: pytest.MonkeyPatch
  ) -> None:
      activation_fixture.mutate_receipt_hash()
      monkeypatch.setattr(json, "loads", deny_report_method_access(json.loads))
      with pytest.raises(ActivationFailure) as raised:
          build_activation(activation_fixture.authority)
      assert raised.value.reason is ActivationReason.INVALID
  ```

- [ ] **Step 2: Run activation RED**

  Run: `.venv/bin/pytest -q tests/test_activate_pass201_pa_source_v2.py`

  Expected: fail because the activation module and exact reason mapping do not exist.

- [ ] **Step 3: Implement metric-blind activation**

  Strict-load `A`, its sole complete receipt, config/manifest sidecars, and restricted checkpoint structure through the contract module. Hash report bytes without parsing its `methods`. Emit canonical activation bytes with exact attempt/policy/read-scope literals. `build_activation` returns bytes; `write_activation` publishes them exclusively with the contract helper.

  ```python
  def build_activation(authority_path: Path) -> bytes:
      try:
          inputs = require_all_authenticated_paths(authority_path)
      except FileNotFoundError as error:
          raise ActivationFailure(ActivationReason.BLOCKED, str(error)) from error
      try:
          payload = validate_and_build_activation(inputs)
      except (ValueError, OSError) as error:
          raise ActivationFailure(ActivationReason.INVALID, str(error)) from error
      return canonical_json_bytes(payload)

  def write_activation(authority_path: Path, output_path: Path) -> OutputEvidence:
      data = build_activation(authority_path)
      return publish_new_file(output_path, data, mode=0o444)
  ```

- [ ] **Step 4: Run activation GREEN and review**

  Run the focused activation test, Ruff, `py_compile`, and `git diff --check`. Request a cold review proving the trust graph is acyclic/non-selective and the failure mapping exact. Fix material findings through new RED/GREEN cycles. Commit only the activation script/test with message `implement Pass201 source v2 activation`; call this immutable code commit `E` and push it. In the existing DGX checkout, first rehash the untouched source-v2 outputs, then fetch and detach at `E` without cleaning or replacing those untracked outputs. `E` precedes and is bound by the activation-data commit.

- [ ] **Step 5: Generate, commit, and push activation alone**

  In the same bound DGX checkout that contains the sole receipt and immutable outputs,
  run exactly:

  ```text
  <bound-python> scripts/activate_pass201_pa_source_v2.py activate \
    --authority docs/pass201_pa_source_v2_prelaunch.json \
    --output docs/pass201_pa_source_v2_activation.json
  ```

  Require the destination was absent and exclusive publication succeeded. Commit only `docs/pass201_pa_source_v2_activation.json` with message `activate Pass201 PA source v2`; remote run outputs remain untracked and must not be staged. Verify exact schema, `attempt_ordinal=1`, automatic acceptance policy, and acyclic trust graph. Push detached HEAD to `origin/devbox/emafactorial`, then fast-forward the local branch.

- [ ] **Step 6: Hand off to a separate Pass201 repair plan**

  End this plan after the reviewed activation commit. Create a separate reviewed plan for `scripts/diagnose_pass201_cis_operator.py` covering the already-known Task3 blockers: real six-key checkpoint shape, nonempty replay tensors, full scalar replay, environment/factory isolation, integrity-before-science ordering, digest recomputation, exact child prefixes, partial failure audit, atomic replacement, and genuine In-Shop ID loading. That later plan alone owns binding replay, smoke, scientific diagnosis, CIS judgment, and ledger updates; this source plan must not implement or run them.
