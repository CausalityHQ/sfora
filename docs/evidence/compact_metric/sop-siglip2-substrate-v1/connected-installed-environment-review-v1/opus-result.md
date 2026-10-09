## Review: installed-environment correspondence helper (HEAD `14eee1543a712707ac211eb0b13cb9a3ab20146a`)

**Verdict: CONDITIONAL GO, source-only.** The helper is admissible and doesn't duplicate anything that exists. It needs four exact fixes, B1–B4 below. It is not overbroad as long as it keeps to the 32 descriptors and claims nothing beyond correspondence. I made no repo edits and no native or SSH reads.

### What I checked against the committed evidence
I recomputed these with stdlib `-I`, without trusting the owner JSON's booleans. All hold:
- **Evidence pins match:** the bundle hashes to `e12429ef…` (same as `verification.json`), the owners file to `8e38d2e5…`, and the collector to `0523c90a…`. All are tracked at 14eee154.
- **One original root:** all 32 RECORD keys sit under a single `…/.venv/lib/python3.13/site-packages`.
- **Members:** 1754 members, 1754 unique. For each one:
  - `file == R + "/" + record_relative_path`, with no `''`, `.`, `..` or leading `/`;
  - `sha256 == bundle.environment.files[file]`;
  - `record == key` and `record_sha256` equals its profile's value;
  - `metadata.path == <dist-info>/METADATA`.
- **The rest of `env.files`:** exactly 13 paths remain. All are in `native_files` and all are outside R. 249 of the selected members are native.
- **Packages:** the 6 roots and origins sit under R and their origins are in `env.files`. Each version equals its owning distribution's METADATA version (PIL→pillow 12.2.0, torch 2.12.1, and so on). `vision_constructor` is owned by transformers 5.12.1.
- **Read cost:** about 3.9 GiB of member bytes per call; the largest single file is 601 MiB, so files must be hashed as a stream.

### Is identical original RECORD bytes enough for cross-root correspondence?
**Yes, and it is the only binding the evidence supports.** The owners file carries only the RECORD SHA, not the RECORD rows, so you can't compare row by row without new original evidence. Matching RECORD bytes mean an identical root-relative row set. The selected rel paths contain no `..` (1754/1754), so `target_root/rel` is a strict correspondence.

**Practical consequence:** the original collector's own comment says `../` console-script rows exist. Their hashes cover shebangs rewritten with the absolute interpreter path. Installers also add rows (INSTALLER, REQUESTED, direct_url.json). So a fresh pip or uv install at a different prefix will most likely fail closed on the dists with scripts. A byte-copy relocation of the dist-info folders plus members is realistically the only way to pass. Failing closed here is correct; don't add normalization.

### Blockers (must be in the source before GO)
- **B1. Selection omission and absolute-path escalation.** Don't define the system anchors as `env.files − ⋃owners`. With that rule, dropping one owner member silently reclassifies a wheel file as an anchor opened at its original absolute path. The rule should be:
  - every `env.files` path under R must be owned exactly once;
  - every path not under R is an anchor;
  - no owned path is outside R.

  Then check `owned ∪ anchors == env.files` and `owned ∩ anchors == ∅`, plus `native_files ⊆ env.files`.
- **B2. Authenticate and parse the same bytes.** The helper takes `bundle_bytes` and `owners_bytes` plus trusted SHA constants from committed caller source. It hashes, then parses those same bytes. Never take a SHA from `verification.json` or the terminal JSON at runtime. Ignore every boolean flag in the owners file (`unique_owner_pass` etc.) and recompute instead.
- **B3. Exact target RECORD rows.** The original collector used lenient `urlsafe_b64decode` (which drops invalid characters) and `resolve()`. The target side must do the following on the same fresh RECORD bytes it hashed:
  - parse with `csv`;
  - require exactly one row per selected rel;
  - require `row == [rel, "sha256=" + b64url_nopad(bytes.fromhex(sha)), str(bytes)]` by string equality;
  - require exactly one METADATA row matching the metadata SHA and size.

  This re-proves ownership at the target from the RECORD bytes instead of trusting the JSON label.
- **B4. Root and injectivity.** `installed_root` must be a canonical absolute path, walked with nofollow. Reject it if:
  - it equals R, or one contains the other (otherwise an identity "relocation" opens original paths and proves nothing);
  - any anchor lies under it, or it lies under an anchor's directory.

  Also require the mapped image to have exactly 1767 distinct values.

### Minimal API
Put this in a new `src/sfora/connected_installed_environment.py`. Factor the existing `_installed_sha` walk into `_open_installed(path) -> fd`, so it can return `(sha, size)` and capped bytes (16 MiB, as in the collector, for RECORD and METADATA). Leave `materialize_identity` behaviour unchanged.

```python
def project_environment(bundle: bytes, owners: bytes, *, bundle_sha256: str,
                        owners_sha256: str, installed_root: str) -> JSONObject  # lexical only
def restore_environment(projected: object, bundle: bytes, owners: bytes, *, bundle_sha256: str,
                        owners_sha256: str, installed_root: str) -> JSONObject   # exact inverse or ValueError
def materialize_environment(projected: object, bundle: bytes, owners: bytes, *, bundle_sha256: str,
                            owners_sha256: str, installed_root: str) -> JSONObject  # fresh reads, returns EXPECTED
```

Projected schema, with the same shape as `bundle.environment` and the same SHA values:
```json
{"schema":"connected-installed-environment-correspondence-v1",
 "original":{"bundle_sha256":"…","owners_sha256":"…","root":"<R, inert>"},
 "installed_root":"<T>",
 "distributions":[{"record":"numpy-2.5.0.dist-info/RECORD","record_sha256":"…",
                   "name":"numpy","version":"2.5.0","metadata":{"sha256":"…","bytes":6584}}],   // 32, sorted
 "environment":{"files":{"<T/rel or anchor>":"sha"},          // 1767
                "native_files":{…},                           // 262
                "packages":{"torch":{"root":"T/torch","origin":"T/torch/__init__.py","version":"2.12.1"}},
                "vision_constructor":"T/transformers/models/siglip/modeling_siglip.py"}}
```

`restore_environment` maps the R/T prefix back on owned paths only and requires `json.dumps(restored) == json.dumps(bundle.environment)`, the same exactness test as `restore_identity`.

`materialize_environment` reads in this order:
1. each of the 32 RECORDs (SHA), then its rows (B3);
2. METADATA (size and SHA);
3. the 1754 members (`fstat` size before hashing, then streamed SHA);
4. the 13 anchors (SHA).

Every read goes through the nofollow walk, requires a regular file, and uses `O_NONBLOCK`. Nothing is imported. `packages` minus `version` is exactly what `materialize_identity(installed_roots=…)` needs, so this replaces today's hand-bound installed roots.

Parsing Name and Version out of METADATA is redundant once its SHA and size are pinned. If you keep it, use exact single-header equality as the collector does, with no name normalization. Dist-info folder names are normalized (`pyyaml-…` vs `PyYAML`, `pillow` vs PIL).

### Falsifying stdlib tests (synthetic tmp R, T and anchor directory)
1. A valid-JSON owners or bundle file whose SHA doesn't match the pin is rejected before it is parsed.
2. Omitting one owned member is rejected (B1). It must not turn into an anchor; assert no open under R or any absolute original path.
3. These owner-file faults are rejected:
   - a member that isn't in `env.files`;
   - the same file listed under two profiles;
   - a member whose SHA differs from the bundle;
   - `rel` of `../x`, `a/./b` or an absolute path;
   - `file != R/rel`;
   - two profiles under different roots.
4. These target RECORD faults are rejected:
   - one appended row (pip INSTALLER);
   - CRLF line endings;
   - padded or otherwise non-canonical b64;
   - a duplicate selected row;
   - a METADATA row that differs.
5. These member faults are rejected:
   - one flipped byte at the same size;
   - a size change;
   - a leaf symlink to identical bytes;
   - a symlinked parent directory;
   - a FIFO (must reject without hanging);
   - a missing file.
6. A METADATA byte flip is rejected, and so is a package version that differs from its owner's version.
7. These installed roots are rejected:
   - equal to R, or nested inside or around R;
   - relative, or with a trailing slash or `..`;
   - one containing an anchor.
8. Round trip: `restore(project(x)) == x.environment`. A projected file with one key added, dropped, or with a changed value or path is rejected.
9. An unselected RECORD row whose file is missing on disk, plus a planted `foo.pth`, still passes. A monkeypatched open helper must log exactly 32+32+1754+13 leaf opens, proving there are no extra reads.
10. A member `.py` that raises when imported still passes, `sys.modules` keys are unchanged, and `ctypes` is not newly imported.

### Non-claims the result must state
These are parent-owned and not blockers here:
- **Not unique ownership at the target.** The other 164 dist-info folders are unread. A second `torch-*.dist-info` could change what `importlib.metadata` returns, which transformers' version checks consume. If you want that covered without reads, compare `listdir(T)` dist-info names against the 196 names pinned in the collector's `EXPECTED_RECORDS`.
- **Not import resolution.** Timestamp `.pyc` files in `__pycache__` run instead of the hashed `.py` (`-B` doesn't stop reading them; that needs `-X pycache_prefix`). Other gaps: `.pth`/sitecustomize, planted `pkg/_C/` folders that shadow `.so` extensions, and RPATH or `ld.so` resolution.
- **Interpreter not bound.** The 13 anchors don't include the interpreter or libpython, so "same admitted Python/ISA" is an unverified caller precondition.
- **Anchors are on-disk only.** Their hashes are a check after mapping, not of the mapped images. Time-of-check vs time-of-use still applies.
- **Still unrun gates:** tensor/native conversion, the same-batch B1/B2/B32 runs with wires/top-10/ties/lifetime, and the genuine target interpreter all remain parent-owned and UNRUN.

I saved this verdict to memory as `sfora-installed-environment-correspondence-review`.
