# Pass201 Ordinary-PA Source v2 Design

## Decision

Run one new ordinary Proxy Anchor seed-0 source under a prospectively committed,
self-auditing controller. The previous source remains an authenticated training
observation but is insufficient as Pass201 authority because the wrapper that
claimed pre/post replay was not itself bound before execution. Do not salvage it
with a post-hoc envelope.

Three approaches were considered:

1. **New source v2 (chosen):** commit controller code, then commit a prelaunch
   authorization while all outputs are absent, then execute once in a clean checkout.
   Cost is about 1.7 DGX-hours; provenance is direct and non-exceptional.
2. **External envelope around the old source:** avoids GPU work but cannot
   retrospectively authenticate the unbound wrapper's claimed contemporaneous
   pre/post checks. Rejected by cold review.
3. **Close Pass201 as blocked:** scientifically valid but discards the strongest
   currently isolated measurement branch before paying a modest source-control cost.

The operator previously granted full autonomy and requested continuous execution,
so this documented choice serves as the design approval without a blocking question.

## Scope

The source run is ordinary corrected In-Shop Proxy Anchor, seed `0`, with the exact
current production recipe and no CIS/operator auxiliary. It produces only source
authority for the already frozen Pass201 diagnostic. No Pass201 operator tensor,
context statistic, decision, or candidate value is computed.

New tracked files:

- `scripts/run_pass201_pa_source_v2.py`: controller, audit, restricted checkpoint
  metadata reader, and deterministic sidecar derivation entry point;
- `tests/test_run_pass201_pa_source_v2.py`: bounded unit plus real temp-Git,
  synthetic-In-Shop, restricted-pickle, and child-process integration suite;
- `docs/pass201_pa_source_v2_prelaunch.json`: generated only after controller review,
  committed while every output is absent.

Remote outputs live in one private run directory,
`reports/generated/pass201_source_v2/run-v2/`, which must itself be absent at
preflight:

- `report.json`;
- `checkpoint.pt`;
- `training.log`;
- `resolved_config.json`;
- `train_manifest.json`;
- `receipt.json`.

## Two-commit authorization

1. Implement and independently review controller/tests; commit reviewed source as
   commit `C`.
2. Generate the prelaunch manifest from `C`. It binds the complete scientific source
   blob set, controller, CLI entry/dependencies, exact command, environment, dataset
   authority, output paths, sidecar algorithms, and requirement that all six outputs
   are absent. Commit only this manifest as authorization commit `A`; no source file
   changes between `C` and `A`.
3. Independently review `C..A`. Run only a clean detached checkout at `A`. At runtime
   the controller verifies current `HEAD=A`, current manifest bytes equal `A:path`,
   every declared source/controller blob equals `C:path`, and output absence before
   training.

The manifest never embeds its own commit. The receipt records actual `A`, manifest
path/hash, and all runtime checks. Activation later validates `A:path` and the exact
receipt bytes, making the graph acyclic.

## Frozen command

The controller launches exactly one subprocess from the isolated checkout root with
the manifest-bound absolute virtual-environment Python executable and argv:

```text
<manifest.execution.python.path> -m sfora.cli image-end-to-end
--dataset-name inshop
--dataset-root /home/riomus/datasets/inshop_official_standard
--objectives proxy_anchor
--recipe auto
--num-workers 8
--seed 0
--save-model-path reports/generated/pass201_source_v2/run-v2/checkpoint.pt
--output reports/generated/pass201_source_v2/run-v2/report.json
```

Combined stdout/stderr is written directly to the frozen log path. The controller
does not shell-expand the command and accepts no alternative objective, seed, recipe,
dataset, output, or extra argument.

## Prelaunch manifest schema

Top-level keys are exactly:

```text
schema_version, status, purpose, source_commit, authorization,
controller, source, execution, dataset, outputs, sidecars, postconditions
```

- `schema_version = "pass201-pa-source-v2-prelaunch-v1"`, `status = "frozen"`.
- `source_commit` is full `C`.
- `authorization` has exact `manifest_path` and requires runtime Git-blob equality;
  it contains no self hash or self commit.
- `controller` has exact path, SHA-256, and Git-blob SHA at `C`.
- `source` lists every repo-relative source/dependency path and SHA-256, plus the
  canonical complete `src/sfora/**/*.py` Merkle algorithm and digest at `C`.
- `execution` contains exact checkout root requirement, cwd, a replacement
  environment map, the Python executable path/bytes/mode/interpreter identity, argv,
  objective, seed, expected canonical config bytes/SHA, and operating-point
  invariants.
- `dataset` binds root, partition path/bytes/SHA/line count, exact bundle
  train/query/gallery counts and protocol, image count, image-tree path/content
  Merkle algorithm/digest, and requires every production-resolved optimization-image
  path/content pair to be a member of both the preflight and postflight trees. This
  is membership evidence, not a claim to have traced every file open.
- `outputs` contains the exact private run directory, six exact repo-relative paths,
  and the required-absence Boolean for each.
- `sidecars` binds the exact config/manifest schema and algorithm IDs below.
- `postconditions` requires exit zero, byte-identical pre/post source/data digests,
  exact output hashes, restricted checkpoint metadata, two-process sidecar identity,
  and receipt publication only after every success predicate.

Every nested object has an exact schema validated by controller tests. JSON parsing
rejects duplicate keys, nonfinite values, wrong scalar types, and unknown keys.

### Normative encoding and nested schemas

This section is normative and overrides any earlier shorthand. `str` means a
nonempty Unicode JSON string; `sha256` means a lowercase 64-hex string; `git_oid`
means a lowercase 40-hex Git object ID; `int` means JSON integer but not Boolean;
`bool` means JSON Boolean; `repo_path` means a normalized relative POSIX UTF-8 path
with no empty, `.` or `..` component; `abs_path` means a normalized absolute POSIX
UTF-8 path whose components after the leading slash obey the same rule. Objects
reject unknown/missing keys.
Arrays have the stated order and reject duplicates where described. All authority
JSON uses UTF-8 canonical JSON with sorted keys, compact separators, no ASCII
escaping, no NaN/Infinity, and exactly one trailing newline. File hashes are SHA-256
over literal bytes. An ordered record hash is SHA-256 over the concatenation of, for
each record, an unsigned big-endian 64-bit canonical-byte length followed by those
canonical bytes. Merkle leaves are the same length-framed canonical record bytes;
leaf hash is `SHA256(b"leaf\0" + framed_record)`, parent hash is
`SHA256(b"node\0" + left + right)`, duplicating the last node at an odd level, and
the empty root is `SHA256(b"empty\0")`.

Every `MerkleBinding` uses the same traversal contract. Resolve the declared root
once, require a real directory, recursively enumerate without following symlinks,
and fail on every symlink, non-regular non-directory entry, non-UTF-8 name, duplicate
normalized relative path, or escape from the root. Directories contribute no leaf.
Each regular-file leaf record is exactly
`{relative_path:repo_path,bytes:int,sha256:sha256}` where the path is relative to the
resolved root; leaves are sorted by raw UTF-8 path bytes and hashed with the
length-framing above. Open each file with `O_RDONLY|O_NOFOLLOW`; require the same
device, inode, mode, size, and `mtime_ns` from `fstat` before and after the streamed
read. The binding's `count` and `bytes` are the leaf count and sum of leaf bytes.
Pre-existing `.pyc`/`__pycache__` files are ordinary included leaves; none is
ignored.

Python import roots use a separate
`ImportTreeBinding = {algorithm:"pass201-import-tree-v1",regular_count:int,
symlink_count:int,bytes:int,root_sha256:sha256}`. Its traversal does not follow
symlinks. A regular leaf is exactly
`{kind:"file",relative_path:repo_path,bytes:int,sha256:sha256}` and uses the same
mutation-safe descriptor read above. A symlink leaf is exactly
`{kind:"symlink",relative_path:repo_path,target_text:str,resolved_path:abs_path,
resolved_scope:"internal"|"external"}` and is read with `lstat/readlink/lstat`,
requiring device/inode/mode/target stability. Leaves of both kinds are canonicalized,
sorted by `(relative_path UTF-8 bytes, kind)`, and length-framed into the tree hash.
An internal resolved target must lie under the same root; its real regular file or
directory contents are reached at their non-symlink location and therefore already
occur in that root's traversal. Every external resolved target is represented once
in `external_symlink_targets` and recursively bound as a descriptor-safe file or
`ImportDirectoryBinding`; repeated targets use identical bindings and cycles fail.

Effective `sys.path` entries retain interpreter order. Empty entries resolve to the
frozen cwd; the checkout root and bound checkout `src` are excluded because repo
authority already binds them. A nonexistent entry is retained explicitly with
`status="nonexistent"`; it is not traversed. A regular ZIP entry is bound as an
`ExternalFileBinding`. A real directory uses `ImportDirectoryBinding`. Every other
entry type fails closed. The controller requires the same ordered tagged list before
and after training.

Common records are exact:

- `FileBinding = {path:repo_path, git_mode:"100644"|"100755", bytes:int,
  sha256:sha256, git_blob:git_oid}`;
- `ExternalFileBinding = {path:abs_path, mode:int, device:int, inode:int,
  bytes:int, sha256:sha256}`;
- `MerkleBinding = {root:repo_path|abs_path,
  algorithm:"pass201-length-framed-merkle-v1", count:int, bytes:int,
  root_sha256:sha256}`;
- `OutputAuthority = {path:repo_path, required_absent:true}` in prelaunch and
  `OutputEvidence = {path:repo_path, file_type:"regular", mode:int, bytes:int,
  sha256:sha256}` postflight;
- `BundleCounts = {train:int, query:int, gallery:int,
  protocol:"query_gallery", protocol_name:str}`.
- `ImportRootBinding` is the tagged union
  `{entry:str,status:"nonexistent"}` or
  `{entry:str,status:"zip",file:ExternalFileBinding}` or
  `{entry:str,status:"directory",directory:ImportDirectoryBinding}`.
  `ImportDirectoryBinding = {root:abs_path,tree:ImportTreeBinding,
  external_symlink_targets:[ExternalImportTarget]}`. `ExternalImportTarget` is
  `{link_relative_path:repo_path,target_text:str,resolved_path:abs_path,
  kind:"file",file:ExternalFileBinding}` or the same prefix with
  `kind:"directory",directory:ImportDirectoryBinding`.

The prelaunch objects have exactly these nested keys and types:

- `authorization = {manifest_path:repo_path, required_parent_commit:git_oid,
  required_diff_paths:[repo_path], required_diff_status:["A"],
  required_diff_modes:["100644"], clean_policy:"empty-porcelain-v1-z",
  frozen_absence_checked_utc:str,
  frozen_absence:{run_directory:"ENOENT",report:"ENOENT",checkpoint:"ENOENT",
  log:"ENOENT",resolved_config:"ENOENT",train_manifest:"ENOENT",
  receipt:"ENOENT"}}`;
- `controller = FileBinding`;
- `source = {files:[FileBinding], python_tree:MerkleBinding,
  pyproject:FileBinding, lockfile:FileBinding, equivalence_test_id:str}`;
- `execution = {checkout_root:abs_path, cwd:abs_path,
  python:ExternalFileBinding, python_realpath:abs_path, python_version:str,
  git:ExternalFileBinding, python_packages:{bytes:int,sha256:sha256},
  python_import_roots:[ImportRootBinding], environment:{str:str},
  environment_policy:"replace", argv:[str], objective:"proxy_anchor", seed:0,
  expected_config_json:str, expected_config_sha256:sha256,
  recipe_id:str, recipe_digest:sha256, schedule:{resolved_train_steps:int,
  steps_per_epoch:int, total_epochs:int},
  pretrained_checkpoint:ExternalFileBinding}`;
- `dataset = {root:"/home/riomus/datasets/inshop_official_standard",
  partition:ExternalFileBinding, partition_lines:int, bundle:BundleCounts,
  declared_image_root:"/home/riomus/datasets/inshop_official_standard/Img/img",
  resolved_image_root:"/home/riomus/datasets/inshop_official_standard/img/img",
  image_root_link:{path:"/home/riomus/datasets/inshop_official_standard/Img",
  target:"img",lstat_mode:int}, image_tree:MerkleBinding,
  image_tree_leaf_base:"resolved_image_root",
  image_tree_leaf_schema:"relative_path,size,sha256",
  selection_policy:"full_official_partition",
  optimization_authority:{algorithm_id:"pass201-production-invocation-capture-v1",
  row_count:int,identity_count:int,ordered_row_sha256:sha256,
  resolved_membership_sha256:sha256}}`;
- `outputs = {run_directory:repo_path, run_directory_required_absent:true,
  report:OutputAuthority, checkpoint:OutputAuthority, log:OutputAuthority,
  resolved_config:OutputAuthority, train_manifest:OutputAuthority,
  receipt:OutputAuthority}`;
- `sidecars = {config_algorithm:"pass201-resolved-config-v2",
  manifest_algorithm:"pass201-inshop-benchmark-row-suffix-v2",
  schedule_algorithm:"pass201-inshop-completed-epoch-v1",
  config_schema:"canonical-json-object-v1",
  manifest_schema:"pass201-train-manifest-v1"}`;
- `postconditions = {required_exit_code:0, require_source_equal:true,
  require_partition_equal:true, require_image_tree_equal:true,
  require_two_process_sidecar_identity:true,
  require_restricted_checkpoint_metadata:true,
  require_complete_receipt:true}`.

The manifest's top-level `schema_version`, `status`, and `purpose` are strings,
`source_commit` is `git_oid`, and all remaining top-level values are the exact
objects above. `execution.cwd` must be byte-identical to `execution.checkout_root`.
Its `environment` is a complete replacement map, not an overlay. Its
keys are exactly `HOME`, `PATH`, `PYTHONPATH`, `PYTHONNOUSERSITE`,
`PYTHONDONTWRITEBYTECODE`,
`LD_LIBRARY_PATH`, `CUDA_VISIBLE_DEVICES`, `CUBLAS_WORKSPACE_CONFIG`,
`PYTHONHASHSEED`, `LC_ALL`, `LANG`, `TZ`, `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`XDG_CACHE_HOME`, and `TORCH_HOME`; values are frozen in `A`. `PATH` contains only
the bound virtual-environment directory and
system `/usr/bin:/bin`; `PYTHONPATH` is the detached checkout's `src`. The Python
file, its resolved realpath target, the resolved Git executable, `pyproject.toml`,
`uv.lock`, and the canonical sorted `python -m pip freeze --all` output are hashed in
`A`; the controller repeats all checks before launch. Package names are not treated
as byte identity: a clean interpreter child emits its ordered effective `sys.path`;
after excluding the bound checkout root and `src`, every ordered effective entry
(stdlib, site-packages, `.pth` additions, ZIPs, nonexistent entries, and any entry
containing `sitecustomize`/`usercustomize`) is recorded as an `ImportRootBinding`.
The controller recomputes the exact tagged list before and after training. Entries
outside the frozen list or a changed binding fail closed.

`PYTHONNOUSERSITE` and `PYTHONDONTWRITEBYTECODE` are both exactly `"1"`; this keeps
the bound import roots byte-stable while the training child imports them.
Pre-existing bytecode remains included in their Merkle bindings.

The expected config in `A` is generated by controller `freeze-authority` mode at
`C`, before authorization, using the real bound dataset and frozen argv. Two fresh
capture children invoke the real Typer application after replacing only
`sfora.cli.run_image_end_to_end_benchmark` with a controller-owned capture function.
That function receives the production-resolved train/query/gallery arguments and
`ImageEndToEndConfig`, derives the exact split/noise/schedule/row suffix, emits only
authority bytes, then raises a controller-owned `BaseException` sentinel before
model construction or report writing. Each child must reach the sentinel exactly
once, create none of the output or known report-temporary paths, and emit
byte-identical config, bundle, rows, resolved-membership, recipe, and schedule
records. A real synthetic-In-Shop integration test at `C` exercises this same
capture boundary. `A` freezes the resulting config bytes/SHA, recipe ID/digest,
exact bundle counts/protocol, optimization authority, and exact schedule tuple.
Agreement between later report and checkpoint is not the source of these
commitments.

`dataset.image_tree.root` must equal the literal `resolved_image_root`; every leaf
path is UTF-8-relative to that root. The controller separately validates the exact
`Img -> img` symlink with `lstat`/`readlink` before and after the run, then resolves
each optimization `ImageExample.image` strictly and requires it to lie under the
literal resolved root and match a tree leaf.

The receipt reuses the common records and has exact nested schemas:

- `authorization = {authorization_commit:git_oid, source_commit:git_oid,
  manifest_path:repo_path, manifest_bytes:int, manifest_sha256:sha256,
  manifest_git_blob:git_oid, parent_verified:true, single_addition_verified:true,
  detached_head_verified:true, clean_policy_verified:true}`;
- `controller = {file:FileBinding, python:ExternalFileBinding,
  python_packages:{bytes:int,sha256:sha256}, source_tree:MerkleBinding}`;
- `command = {cwd:abs_path, environment:{str:str}, argv:[str]}`;
- `preflight = {started_utc:str, run_directory_absent:true,
  source_tree:MerkleBinding, partition:ExternalFileBinding,
  image_tree:MerkleBinding, pretrained_checkpoint:ExternalFileBinding,
  outputs_absent:{report:true,checkpoint:true,log:true,resolved_config:true,
  train_manifest:true,receipt:true}}`;
- `process = {pid:int, started_utc:str, ended_utc:str, exit_code:0}`;
- `postflight = {ended_utc:str, source_tree:MerkleBinding,
  partition:ExternalFileBinding, image_tree:MerkleBinding,
  pretrained_checkpoint:ExternalFileBinding, source_equal:true,
  partition_equal:true, image_tree_equal:true,
  pretrained_checkpoint_equal:true}`;
- `outputs = {report:OutputEvidence, checkpoint:OutputEvidence,
  log:OutputEvidence, resolved_config:OutputEvidence,
  train_manifest:OutputEvidence}`;
- `checkpoint_metadata = {literal_top_keys:["arch","artifact_selection",
  "evaluation_model_source","state_dict","training_config","training_step"],
  artifact_selection:"final_training_state", evaluation_model_source:"student",
  arch:{backbone_name:"bn_inception",pretrained_weights:"bn_inception_52deb4733",
  head_pooling:"avg_max",embedding_dimensions:512,
  embedding_head_init:"kaiming_normal",embedding_layer_norm:false}, training_step:int,
  training_config_sha256:sha256, state_dict_storage_materialized:false}`;
- `sidecar_derivation = {config_algorithm:"pass201-resolved-config-v2",
  manifest_algorithm:"pass201-inshop-benchmark-row-suffix-v2",
  schedule_algorithm:"pass201-inshop-completed-epoch-v1",
  source_files:[FileBinding],
  input_hashes:{manifest:sha256,source_tree:sha256,partition:sha256,
  image_tree:sha256,pretrained_checkpoint:sha256,report:sha256,
  checkpoint:sha256,expected_config:sha256},
  child_processes:[{ordinal:1|2,pid:int,config_sha256:sha256,
  manifest_sha256:sha256}],
  row_count:int,identity_count:int,ordered_row_sha256:sha256,
  resolved_membership_count:int,resolved_membership_sha256:sha256,
  membership_covered_by_preflight:true,membership_covered_by_postflight:true}`;
- `scope = {ordinary_source_uses_official_query_gallery:true,
  uses_pass201_operator_data:false,pass201_candidate_paths_read:false,
  authorized_action:"source_binding_only"}`.

The receipt's `preflight` absence map records evidence established before creating
the private directory. Postflight never claims absence: it records regular-file
type, mode, size, and descriptor-derived hash. The later activation commit must,
before any Pass201 operator/candidate path is read, bind the exact tuple
`(authorization commit A, A:path Git blob and SHA-256, receipt SHA-256, report
SHA-256, checkpoint SHA-256, log SHA-256, resolved-config SHA-256, train-manifest
SHA-256)`. The activation commit is only a container and is never embedded in these
artifacts.

`child_processes` has exactly two elements in ordinal order, with distinct positive
PIDs and identical config/manifest hashes. `source_files` is sorted by UTF-8 path
bytes and nonempty. No other dynamic-key map occurs in either authority schema; the
environment is the one explicitly enumerated above.
The three receipt algorithm literals must also equal the corresponding three
literals in `A.sidecars`; equality is checked before receipt publication.

### Non-selective activation

The single authorization permits exactly one training invocation and therefore at
most one receipt. The first authorized attempt is accepted automatically if and only
if its complete receipt validates; an incomplete/invalid first attempt blocks this
authority and cannot be replaced by a second receipt. Before activation is committed,
the activation process may read raw bytes and the restricted structural metadata
above, but it must not parse `report.methods`, any ordinary PA metric value, or any
Pass201 operator/candidate path.

The activation document has exactly:

```text
schema_version:"pass201-pa-source-v2-activation-tuple-v1",
status:"frozen",
attempt_ordinal:1,
source_commit:git_oid,
authorization_commit:git_oid,
manifest:{path:repo_path,bytes:int,sha256:sha256,git_blob:git_oid},
receipt:{path:repo_path,bytes:int,sha256:sha256},
outputs:{report:OutputEvidence,checkpoint:OutputEvidence,log:OutputEvidence,
resolved_config:OutputEvidence,train_manifest:OutputEvidence},
acceptance_policy:"activate_single_complete_valid_v2_receipt_else_block",
report_method_values_read:false,
pass201_paths_read:false,
authorized_action:"source_binding_before_pass201_computation"
```

It is committed as the only file in a later activation commit. That commit is the
external trust root recorded by Pass201 replay; it is a container and is never
embedded in the activation document.

### Restricted checkpoint metadata contract

The restricted metadata child uses only `os.open(..., O_RDONLY|O_NOFOLLOW)`,
`fstat`, `zipfile`, `pickle`, and `pickletools`; importing `torch` is forbidden in
that child. The separate authority/row-capture children may import production
modules but never open the produced checkpoint or any Pass201 candidate path. The
metadata child rejects an archive larger than 2 GiB, more than
100,000 ZIP members, encrypted members, duplicate member names, non-regular input,
or a `data.pkl` payload larger than 64 MiB. There must be exactly one member whose
POSIX name is `data.pkl` or ends in `/data.pkl`. The reader opens no `data/` storage
member. `find_class` accepts only
`collections.OrderedDict`, `torch._utils._rebuild_tensor`,
`torch._utils._rebuild_tensor_v2`, `torch._utils._rebuild_tensor_v3`,
`torch.ByteStorage`, `torch.CharStorage`, `torch.ShortStorage`, `torch.IntStorage`,
`torch.LongStorage`, `torch.HalfStorage`, `torch.FloatStorage`,
`torch.DoubleStorage`, `torch.BoolStorage`, `torch.BFloat16Storage`,
`torch.ComplexFloatStorage`, `torch.ComplexDoubleStorage`,
`torch.storage.TypedStorage`, and `torch.storage.UntypedStorage`; every accepted
callable/class is replaced with a metadata-only stub. `persistent_load` accepts
only a tuple `("storage", storage_class_stub, ascii_key, location, int_size)` with
nonnegative size and returns a non-indexable storage stub. Any reduction other than
these fails closed.

The unpickled root must be an ordinary `dict` with exactly the six keys frozen in
the receipt schema. `state_dict` must be an `OrderedDict[str, tensor_stub]`; its
contents may be counted and structurally validated but never materialized. The
reader requires exact `artifact_selection="final_training_state"`, exact
`evaluation_model_source="student"`, exact architecture fields shown above,
`training_step` equal to the prebound schedule step, and canonical
`training_config` bytes equal to `A.execution.expected_config_json`. All string,
integer, Boolean, list, and object types are checked exactly; Python truthiness or
numeric coercion is forbidden.

## Controller process and receipt

Before importing torch, reading checkpoint tensor storage, or invoking training, the
controller:

1. validates strict manifest schema and current `A:path` equality;
2. validates detached `HEAD=A`, that `A` has exactly one parent equal to `C`, and
   that `C..A` contains exactly one added regular file at the authorization path,
   with the declared mode and bytes equal to the current manifest; it then validates
   controller and every declared Git blob/content hash and the clean-worktree policy;
3. recomputes partition and full image-tree authority;
4. verifies every output path is absent and its parent is inside the checkout;
5. records exact preflight evidence in memory.

The authorization names a unique run directory that must not exist. The controller
creates it with mode `0700`, holds an exclusive advisory lock in its already-open
directory descriptor, rejects symlinks/non-regular files, and places all six outputs
inside it. In the trusted single-user DGX threat model this directory is the
non-clobber boundary: the training process may update its own partial report, but no
unrelated process is authorized to write there. The controller opens the log with
exclusive creation, runs the exact subprocess once with a complete replacement
environment, captures the exact return code, and recomputes all source and data
bindings through already-open regular-file descriptors. A nonzero exit or any
postflight mismatch publishes no successful receipt or sidecar and makes the source
unusable.

On success it hashes immutable report/checkpoint/log bytes, reads only the
checkpoint ZIP `data.pkl` with a restricted unpickler that never opens tensor-storage
members, and derives both sidecars twice in separate fresh subprocesses. The two
canonical byte streams must match exactly. It publishes sidecars and receipt to
initially absent paths using fsynced sibling temporaries plus atomic non-clobber
hard-link publication and parent-directory fsync.

Receipt top-level keys are exactly:

```text
schema_version, status, candidate_values_computed, authorization,
controller, command, preflight, process, postflight, outputs,
checkpoint_metadata, sidecar_derivation, scope
```

- `schema_version = "pass201-pa-source-v2-receipt-v1"`;
- `status = "complete"`, `candidate_values_computed = false`;
- `authorization` records actual `A`, manifest path/bytes/SHA/Git-blob identity, and
  declared `C`;
- `controller` records path/hash/blob and executing Python/environment identity;
- `command` records exact cwd/environment/argv;
- `preflight` persists complete source/data digests and exact absence evidence;
  `postflight` persists complete source/data digests plus output presence/type/size/
  descriptor-hash evidence and never calls that post-run state "absence";
- `process` records exit `0` and supporting start/end UTC timestamps;
- `outputs` records path, byte count, and SHA-256 for report/checkpoint/log/config/
  train manifest; receipt does not hash itself;
- `checkpoint_metadata` records literal top-key set, artifact selection, evaluation
  source, training step, canonical training-config hash, and
  `state_dict_storage_materialized=false`;
- `sidecar_derivation` records algorithm IDs, source blobs, input hashes, both child
  output hashes, row/identity counts, ordered-row digest, and production-resolved
  optimization-image membership;
- `scope` records `ordinary_source_uses_official_query_gallery=true`,
  `uses_pass201_operator_data=false`, `pass201_candidate_paths_read=false`, and
  `authorized_action="source_binding_only"`.

## Deterministic sidecars

### Resolved config

Algorithm ID: `pass201-resolved-config-v2`. Authorization `A` already contains the
expected canonical config bytes and SHA generated at `C` from the frozen argv,
resolved recipe, and bound bundle counts. Parse the report with strict JSON, take
exactly its `config` object, require JSON-native exact types, and encode as
UTF-8 canonical JSON (`sort_keys=true`, compact separators, `ensure_ascii=false`,
`allow_nan=false`) plus one final newline. Require it and the restricted checkpoint
`training_config` to be byte-identical to the prelaunch value, with exact
`objectives=["proxy_anchor"]`, exact integer `seed=0`, and exact prebound recipe ID
and recipe digest.

### Train manifest

Algorithm ID: `pass201-inshop-benchmark-row-suffix-v2`. At source commit `C`, use
this exact benchmark-side row-derivation suffix and no reimplementation:

1. `src/sfora/data.py::load_image_retrieval_bundle` with
   `dataset_name="inshop"`, the manifest-bound dataset root,
   `limit_per_class=None`, `train_min_per_class=None`,
   `evaluation_min_per_class=None`, `max_classes=None`, and `seed=0`;
2. `src/sfora/image_end_to_end.py::ImageEndToEndConfig.model_validate` on the exact
   canonical report config;
3. `src/sfora/image_end_to_end.py::_checkpoint_train_validation_split` on
   `bundle.train`, with fraction equal to
   `config.checkpoint_selection_validation_fraction` only when
   `config.checkpoint_selection_interval > 0`, otherwise exactly `0.0`, and seed
   `config.seed`;
4. `src/sfora/image_end_to_end.py::_apply_training_label_noise` on the optimization
   result with `fraction=config.label_noise_fraction` and `seed=config.seed`;
5. enumerate that returned list without a transform; row `sample_index` is the
   zero-based list position, `example_id` is the literal `ImageExample.example_id`,
   and `label` is exact `int(ImageExample.label)`.

Bind the exact `src/sfora/data.py`, `src/sfora/image_end_to_end.py`, and imported
model/config dependency Git blobs at `C`. Require full official-partition policy,
all three class/cap arguments above to be `None`, and exact bundle counts/protocol
before split/noise. Any divergence is invalid.

Emit schema `pass201-train-manifest-v1` with exact top-level keys
`schema_version`, `algorithm_id`, `source_commit`, `dataset_authority`, `rows`, and
`derivation`. Each row has exactly `sample_index`, `example_id`, `label`; sample
indices are contiguous zero-based integers in emitted order. The nested schemas are
exact:

```text
schema_version:"pass201-train-manifest-v1",
algorithm_id:"pass201-inshop-benchmark-row-suffix-v2",
source_commit:git_oid,
dataset_authority:{
  root:"/home/riomus/datasets/inshop_official_standard",
  partition_sha256:sha256,
  resolved_image_root:"/home/riomus/datasets/inshop_official_standard/img/img",
  image_tree_sha256:sha256,
  bundle:BundleCounts,
  selection_policy:"full_official_partition"
},
rows:[{sample_index:int,example_id:str,label:int}],
derivation:{
  call_graph:[
    "sfora.cli._load_cli_image_retrieval_bundle",
    "sfora.image_recipes.resolve_recipe",
    "sfora.image_recipes.config_for_recipe",
    "sfora.image_recipes.mark_recipe_config_modified",
    "sfora.image_end_to_end._checkpoint_train_validation_split",
    "sfora.image_end_to_end._apply_training_label_noise",
    "sfora.image_end_to_end._resolve_training_schedule"
  ],
  source_files:[FileBinding],
  resolved_config_sha256:sha256,
  row_count:int,
  identity_count:int,
  ordered_row_sha256:sha256,
  resolved_membership_count:int,
  resolved_membership_sha256:sha256
}
```

The call-graph array above is literal and ordered. `source_files` is the nonempty
UTF-8-path-sorted list of the bound blobs defining every named callable and imported
schema/model dependency; its exact path list is committed in `A`. Row count equals
the array length, identity count equals the cardinality of exact integer labels, and
the ordered-row hash uses the normative framing above.

At `C`, a real synthetic In-Shop integration test must demonstrate that the
controller's expected-config builder and row suffix equal the config and ordered
optimization rows captured at the production CLI boundary after
`_load_cli_image_retrieval_bundle`, `resolve_recipe`, `config_for_recipe`, runtime
config assembly, and `mark_recipe_config_modified`. The manifest binds all of those
source blobs and the equivalence-test ID.

For every stable sample index, resolve the production dataset's source path,
require it remains within the bound dataset root, and require its path/content pair
participates in the recomputed bound image-tree Merkle. `example_id` is never treated
as a filesystem path.

### Schedule/epoch derivation

Algorithm ID: `pass201-inshop-completed-epoch-v1`. Call
`src/sfora/image_end_to_end.py::_resolve_training_schedule` with the validated config,
`optimization_example_count=len(optimization_examples)`, and their ordered integer
labels. Let its exact return be `(resolved_train_steps, steps_per_epoch,
total_epochs)`. Require `steps_per_epoch > 0`, checkpoint `training_step` equal to
`resolved_train_steps`, and `training_step % steps_per_epoch == 0`. Define
`checkpoint_epoch` as the **one-based count of fully completed epochs**,
`training_step // steps_per_epoch`; require it equals `total_epochs`. For the frozen
drop-last policy the bound function computes
`steps_per_epoch=max(1, optimization_example_count // config.batch_size)`; require
`drop_last_train_batch=true` and freeze the exact returned tuple in `A`. No separate
CLI display estimate is permitted. Bind the exact schedule-function Git blob at `C`.

## Failure behavior

The controller uses no scientific Pass201 reason code because it is pre-activation.
Any missing input, preflight mismatch, output collision, training failure, postflight
mismatch, metadata mismatch, sidecar disagreement, or publication failure exits
nonzero and cannot emit `status=complete`. A failure audit may be written only to a
separate uniquely addressed operational log; it is never accepted as source
authority.

During later Pass201 activation, an absent complete v2 receipt/output maps to
`BLOCKED_SOURCE_ARTIFACT_UNAVAILABLE`; a present but mismatched receipt, source,
sidecar, derivation, or artifact maps to `INVALID_OPERATING_POINT_MISMATCH`.

## Testing and success criteria

Local tests use bounded fixtures but are not mocks-only. They must include a real
temporary two-commit Git repository that checks exact parent/diff/mode/blob and
detached/dirty failures; a real synthetic In-Shop partition/tree through the
production loader and CLI-boundary config/row equivalence test; safe and malicious
ZIP/pickle fixtures exercising every allowlist/size/storage rejection; a real child
process proving replacement-environment/argv/exit identity; symlink, output
collision, partial report, and publication-failure cases; and a two-fresh-process
byte-identity test. Together they prove strict schemas, Git-blob/source/data
validation, private-directory/non-clobber behavior, exact subprocess argv,
pre/post mismatch failure, receipt completeness, exact config types/bytes,
production dataset row equality, example-ID/path separation, membership coverage,
and that the restricted metadata child imports no torch, materializes no tensor
storage, and no process reads a CIS or Pass201 operator/candidate path during
binding. Hashing the prospectively bound pretrained checkpoint and structurally
reading the newly produced PA checkpoint are explicitly required source-binding
actions, not forbidden model-path access.

Before launch, an independent reviewer must approve controller code and the later
`C..A` prelaunch diff. The DGX run is valid only from a clean detached checkout at
`A`, with no other compute process. CPU/GPU wall time is operational only. Success is
one immutable complete receipt and its five bound outputs; it does not establish any
method effect.
